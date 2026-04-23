import json
import os
import shutil
import tempfile
import uuid
from typing import Dict, List, Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from logging_setup import get_logger
from rate_limiter import RateLimiter, RateLimitError

log = get_logger(__name__)


# --------------------------------------------------------------------------- Embeddings factory
def build_embeddings(provider: str, api_key: str) -> Embeddings:
    """Return the right embedding model for the chosen provider.

    • openai → OpenAIEmbeddings
    • gemini → GoogleGenerativeAIEmbeddings with "models/gemini-embedding-001".
              The older "models/embedding-001" and "text-embedding-004" aliases
              have been removed from the v1beta endpoint, so we pin to the
              current stable embedContent-capable model.
    """
    provider = (provider or "openai").lower()
    if provider == "openai":
        return OpenAIEmbeddings(api_key=api_key)
    elif provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key,
        )
    else:
        raise ValueError(f"Unknown provider '{provider}'.")


# --------------------------------------------------------------------------- paths
# Root under which every session's FAISS index lives as its own subdirectory.
# Per-session isolation prevents one user's upload from being embedded into
# another user's index in a multi-user deploy.
DEFAULT_VECTOR_ROOT = os.getenv("TA_VECTOR_ROOT", "vector_store")

# Marker file we write on index creation. `load()` refuses to open a persisted
# directory unless the marker matches what this RAGService instance expects.
# This makes FAISS's `allow_dangerous_deserialization=True` safer: we only
# deserialize stores we know were written by *this* app, in *this* session.
_MARKER_NAME = ".session_marker"


def _session_dir(root: str, session_id: str) -> str:
    """Filesystem-safe per-session subdirectory."""
    # Only allow alnum/underscore/dash in the session component. This is
    # defense in depth — Streamlit session IDs are already UUIDs, but we
    # sanitize anyway to prevent path traversal if this ever gets called
    # with user input.
    safe = "".join(c for c in (session_id or "default") if c.isalnum() or c in "-_")
    safe = safe[:64] or "default"
    return os.path.join(root, safe)


class RAGService:
    """FAISS wrapper with per-file chunk tracking, per-session isolation, and
    upload size enforcement.

    Args:
        session_id: Unique ID for this user session. If omitted, falls back
            to "default" which is only safe in single-user mode.
        persist_root: Root directory under which each session gets its own
            subdirectory. Overridden by the TA_VECTOR_ROOT env var if set.
        rate_limiter: Optional RateLimiter used to enforce per-file and
            per-session upload size caps.
    """

    def __init__(
        self,
        persist_root: str = DEFAULT_VECTOR_ROOT,
        provider: str = "openai",
        api_key: str = "",
        session_id: Optional[str] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.session_id = session_id or "default"
        self.persist_dir = _session_dir(persist_root, self.session_id)
        self.index_path = os.path.join(self.persist_dir, "faiss_index")
        self.manifest_path = os.path.join(self.persist_dir, "manifest.json")
        self.marker_path = os.path.join(self.persist_dir, _MARKER_NAME)
        self.provider = provider
        self.api_key = api_key
        self.rate_limiter = rate_limiter
        self.embeddings = build_embeddings(provider, api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        self.vector_store = None
        self.processed_files: set = set()
        # filename -> list of chunk UUIDs inside the FAISS index. Used so we can
        # call vector_store.delete(ids) when a file is removed.
        self.file_chunks: Dict[str, List[str]] = {}
        # filename -> size in bytes, so we can refund RateLimiter quota on
        # removal.
        self.file_sizes: Dict[str, int] = {}
        log.debug(
            "RAGService init: provider=%s session=%s dir=%s",
            provider, self.session_id, self.persist_dir,
        )

    # -------------------------------------------------------------- hardening
    def _write_marker(self) -> None:
        os.makedirs(self.persist_dir, exist_ok=True)
        with open(self.marker_path, "w", encoding="utf-8") as f:
            json.dump(
                {"session_id": self.session_id, "app": "ai-teaching-assistant"},
                f,
            )

    def _marker_ok(self) -> bool:
        """True if the marker file belongs to our session.

        The marker is our defense against loading a FAISS pickle written by
        someone else (FAISS needs allow_dangerous_deserialization=True, which
        runs pickle.load under the hood). If the marker is absent or has a
        different session_id, we refuse to load.
        """
        if not os.path.exists(self.marker_path):
            log.warning("FAISS load refused: missing marker in %s", self.persist_dir)
            return False
        try:
            with open(self.marker_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            log.warning("FAISS load refused: unreadable marker (%s)", e)
            return False
        if data.get("session_id") != self.session_id:
            log.warning(
                "FAISS load refused: marker session %r != current %r",
                data.get("session_id"), self.session_id,
            )
            return False
        if data.get("app") != "ai-teaching-assistant":
            log.warning("FAISS load refused: marker not from this app")
            return False
        return True

    # ------------------------------------------------------------------ load/save
    def load(self) -> bool:
        if not (
            os.path.isdir(self.index_path) and os.path.exists(self.manifest_path)
        ):
            return False
        if not self._marker_ok():
            return False
        try:
            self.vector_store = FAISS.load_local(
                self.index_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            self.processed_files = set(manifest.get("files", []))
            self.file_chunks = {
                k: list(v) for k, v in manifest.get("file_chunks", {}).items()
            }
            self.file_sizes = {
                k: int(v) for k, v in manifest.get("file_sizes", {}).items()
            }
            # Back-compat: older manifests had no file_chunks; degrade gracefully.
            for fname in self.processed_files:
                self.file_chunks.setdefault(fname, [])
                self.file_sizes.setdefault(fname, 0)
            # Replay upload sizes into the rate limiter so caps survive reload.
            if self.rate_limiter:
                for size in self.file_sizes.values():
                    try:
                        self.rate_limiter.check_upload("(restored)", size)
                    except RateLimitError:
                        # If the restored index exceeds caps, log but accept —
                        # the user already paid that cost in a prior session.
                        log.warning(
                            "restored index exceeds current upload caps; "
                            "accepting but new uploads will be blocked"
                        )
                        break
            log.info(
                "loaded persisted index: %d files (%d chunks)",
                len(self.processed_files),
                sum(len(v) for v in self.file_chunks.values()),
            )
            return True
        except Exception as e:
            log.exception("failed to load persisted index: %s", e)
            self.vector_store = None
            self.processed_files = set()
            self.file_chunks = {}
            self.file_sizes = {}
            return False

    def save(self) -> None:
        if self.vector_store is None:
            return
        os.makedirs(self.persist_dir, exist_ok=True)
        self.vector_store.save_local(self.index_path)
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "files": sorted(self.processed_files),
                    "file_chunks": self.file_chunks,
                    "file_sizes": self.file_sizes,
                },
                f,
                indent=2,
            )
        self._write_marker()
        log.debug("index saved: %d files", len(self.processed_files))

    def clear_index(self) -> None:
        """Wipe the in-memory store AND delete the persisted directory."""
        # Refund all quota before wiping so a fresh session doesn't inherit
        # stale accounting.
        if self.rate_limiter:
            for size in self.file_sizes.values():
                self.rate_limiter.release_upload(size)
        self.vector_store = None
        self.processed_files = set()
        self.file_chunks = {}
        self.file_sizes = {}
        if os.path.isdir(self.persist_dir):
            try:
                shutil.rmtree(self.persist_dir)
            except OSError as e:
                log.warning("could not remove %s: %s", self.persist_dir, e)
        log.info("index cleared for session %s", self.session_id)

    # ---------------------------------------------------------------- ingestion
    def process_files(self, uploaded_files) -> None:
        """Ingest Streamlit uploaded files, one file at a time so we can record
        chunk IDs per source filename.

        Enforces per-file and per-session size caps via the RateLimiter BEFORE
        calling the embedding API. Raises RateLimitError on violation.
        """
        for uploaded_file in uploaded_files:
            if uploaded_file.name in self.processed_files:
                continue  # idempotent — don't duplicate chunks

            # Enforce upload caps before we pay for embeddings.
            data = uploaded_file.getvalue()
            size = len(data)
            if self.rate_limiter:
                # Will raise RateLimitError; let the caller surface the message.
                self.rate_limiter.check_upload(uploaded_file.name, size)

            docs = self._load_file(uploaded_file)
            if not docs:
                # Release the reservation since nothing actually got indexed.
                if self.rate_limiter:
                    self.rate_limiter.release_upload(size)
                continue
            chunks = self.text_splitter.split_documents(docs)
            ids = [str(uuid.uuid4()) for _ in chunks]
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(
                    chunks, self.embeddings, ids=ids
                )
            else:
                self.vector_store.add_documents(chunks, ids=ids)
            self.file_chunks[uploaded_file.name] = ids
            self.file_sizes[uploaded_file.name] = size
            self.processed_files.add(uploaded_file.name)
            log.info(
                "ingested %s (%d bytes, %d chunks)",
                uploaded_file.name, size, len(ids),
            )
        self.save()

    def _load_file(self, uploaded_file) -> List[Document]:
        suffix = f".{uploaded_file.name.split('.')[-1]}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        try:
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(tmp_path)
            elif uploaded_file.name.endswith(".txt"):
                loader = TextLoader(tmp_path, encoding="utf-8")
            else:
                log.warning("unsupported file type: %s", uploaded_file.name)
                return []
            docs = loader.load()
            for d in docs:
                d.metadata["source_name"] = uploaded_file.name
            return docs
        except Exception as e:
            log.exception("failed to parse %s: %s", uploaded_file.name, e)
            return []
        finally:
            try:
                os.unlink(tmp_path)
            except OSError as e:
                log.debug("could not delete temp file %s: %s", tmp_path, e)

    def remove_file(self, filename: str) -> bool:
        """Delete a single file's chunks from the FAISS index."""
        ids = self.file_chunks.get(filename)
        size = self.file_sizes.get(filename, 0)
        if not ids or self.vector_store is None:
            # Nothing to delete — just make sure our manifest is consistent.
            self.processed_files.discard(filename)
            self.file_chunks.pop(filename, None)
            self.file_sizes.pop(filename, None)
            if size and self.rate_limiter:
                self.rate_limiter.release_upload(size)
            return False
        try:
            self.vector_store.delete(ids)
        except Exception as e:
            log.exception("FAISS delete failed for %s: %s", filename, e)
            return False
        self.file_chunks.pop(filename, None)
        self.file_sizes.pop(filename, None)
        self.processed_files.discard(filename)
        if self.rate_limiter and size:
            self.rate_limiter.release_upload(size)
        if not self.file_chunks:
            # Index is now empty — wipe persistence so no stale files remain.
            self.clear_index()
        else:
            self.save()
        log.info("removed %s from index", filename)
        return True

    # --------------------------------------------------------------- retrieval
    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        if self.vector_store is None or not query:
            return []
        return self.vector_store.similarity_search(query, k=k)

    def get_retriever(self, k: int = 4):
        if self.vector_store:
            return self.vector_store.as_retriever(search_kwargs={"k": k})
        return None

    def has_index(self) -> bool:
        return self.vector_store is not None and bool(self.processed_files)


def format_context(docs: List[Document]) -> str:
    if not docs:
        return "(no context retrieved)"
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source_name", "unknown")
        page = d.metadata.get("page")
        tag = f"[{i}] {src}" + (f" (p.{page+1})" if isinstance(page, int) else "")
        parts.append(f"{tag}\n{d.page_content.strip()}")
    return "\n\n".join(parts)


def format_sources(docs: List[Document]) -> List[dict]:
    seen = []
    out = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source_name", "unknown")
        page = d.metadata.get("page")
        key = (src, page)
        if key in seen:
            continue
        seen.append(key)
        out.append(
            {
                "id": i,
                "source": src,
                "page": (page + 1) if isinstance(page, int) else None,
                "snippet": d.page_content.strip()[:240],
                "web": bool(d.metadata.get("web")),
                "title": d.metadata.get("title"),
                "url": d.metadata.get("url"),
            }
        )
    return out
