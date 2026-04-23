"""RAGService — chunk tracking, session isolation, FAISS marker defense,
rate-limit integration. Uses a fake Embeddings backend so tests run offline
with zero API cost.
"""
import json
import os
import shutil

import pytest
from langchain_core.embeddings import Embeddings

from rag_service import RAGService, _session_dir
from rate_limiter import RateLimiter, RateLimitError


# -------------------------------------------------------- fake embeddings
class FakeEmbeddings(Embeddings):
    """Deterministic hash-based embeddings so FAISS has something to index
    without network / API calls."""

    def _vec(self, text: str):
        # 8-dimensional stable bag-of-words-ish vector.
        v = [0.0] * 8
        for tok in text.lower().split():
            v[hash(tok) % 8] += 1.0
        n = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


# ---------------------------------------------------------- fake upload
class FakeUpload:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._data = content

    def getvalue(self):
        return self._data


# ---------------------------------------------------------------- fixtures
@pytest.fixture
def rag(tmp_path, monkeypatch):
    """A RAGService with FakeEmbeddings, a fresh tmp persist dir, and a
    permissive rate limiter."""
    # Override build_embeddings so RAGService doesn't try to import openai.
    import rag_service as rs
    monkeypatch.setattr(rs, "build_embeddings", lambda p, k: FakeEmbeddings())
    return RAGService(
        persist_root=str(tmp_path),
        provider="openai",
        api_key="fake",
        session_id="session-a",
        rate_limiter=RateLimiter(
            llm_per_min=100, llm_per_session=100,
            max_file_bytes=1_000_000, max_total_bytes=10_000_000,
        ),
    )


# -------------------------------------------------------------- basic ingest
class TestIngest:
    def test_process_txt_file_creates_index(self, rag):
        rag.process_files([FakeUpload("a.txt", b"Python is a programming language.")])
        assert "a.txt" in rag.processed_files
        assert rag.has_index()
        assert len(rag.file_chunks["a.txt"]) >= 1

    def test_retrieve_returns_documents(self, rag):
        rag.process_files([FakeUpload("a.txt", b"Python is great for scripting.")])
        docs = rag.retrieve("python scripting", k=1)
        assert len(docs) == 1
        assert "Python" in docs[0].page_content

    def test_idempotent_on_same_filename(self, rag):
        rag.process_files([FakeUpload("a.txt", b"Content one")])
        n1 = len(rag.file_chunks["a.txt"])
        rag.process_files([FakeUpload("a.txt", b"Content two")])  # same name
        # Chunks should NOT double — we skip files we've already seen.
        assert len(rag.file_chunks["a.txt"]) == n1


# --------------------------------------------------------- chunk removal
class TestRemoveFile:
    def test_remove_drops_chunks_and_manifest(self, rag):
        rag.process_files([
            FakeUpload("a.txt", b"Alpha content for testing"),
            FakeUpload("b.txt", b"Beta content for testing"),
        ])
        assert rag.remove_file("a.txt")
        assert "a.txt" not in rag.processed_files
        assert "a.txt" not in rag.file_chunks
        # b.txt is still searchable.
        docs = rag.retrieve("beta", k=2)
        assert any("Beta" in d.page_content for d in docs)

    def test_remove_last_file_wipes_dir(self, rag):
        rag.process_files([FakeUpload("a.txt", b"Only file")])
        rag.remove_file("a.txt")
        # Dir should be gone — otherwise stale indexes linger.
        assert not os.path.isdir(rag.persist_dir)

    def test_remove_refunds_upload_quota(self, rag):
        rag.process_files([FakeUpload("a.txt", b"X" * 500)])
        used_before = rag.rate_limiter.status()["uploaded_bytes"]
        rag.remove_file("a.txt")
        used_after = rag.rate_limiter.status()["uploaded_bytes"]
        assert used_before == 500
        assert used_after == 0


# ----------------------------------------------------- session isolation
class TestSessionIsolation:
    def test_different_sessions_use_different_dirs(self, tmp_path, monkeypatch):
        import rag_service as rs
        monkeypatch.setattr(rs, "build_embeddings", lambda p, k: FakeEmbeddings())
        a = RAGService(persist_root=str(tmp_path), session_id="alice")
        b = RAGService(persist_root=str(tmp_path), session_id="bob")
        assert a.persist_dir != b.persist_dir
        assert "alice" in a.persist_dir
        assert "bob" in b.persist_dir

    def test_session_id_is_path_sanitized(self, tmp_path):
        # Path-traversal attempts must be scrubbed.
        d = _session_dir(str(tmp_path), "../../etc/passwd")
        # The ".." and "/" should be stripped; the dir must stay under root.
        normalized = os.path.normpath(d)
        assert normalized.startswith(os.path.normpath(str(tmp_path)))

    def test_load_from_other_session_rejected(self, tmp_path, monkeypatch):
        """A malicious swap: session B points at the dir session A wrote.
        The marker file must make B refuse to load."""
        import rag_service as rs
        monkeypatch.setattr(rs, "build_embeddings", lambda p, k: FakeEmbeddings())

        a = RAGService(persist_root=str(tmp_path), session_id="alice")
        a.process_files([FakeUpload("a.txt", b"alice secret notes")])

        # Same on-disk directory, different session_id. This simulates an
        # attacker dropping their own store into our persist root.
        b = RAGService(persist_root=str(tmp_path), session_id="bob")
        # Move alice's dir to bob's expected path.
        shutil.move(a.persist_dir, b.persist_dir)
        # Marker says "alice" but b.session_id="bob" — load must refuse.
        assert not b.load()
        assert not b.has_index()


# ----------------------------------------------------- persistence + marker
class TestPersistence:
    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        import rag_service as rs
        monkeypatch.setattr(rs, "build_embeddings", lambda p, k: FakeEmbeddings())

        a = RAGService(persist_root=str(tmp_path), session_id="s1")
        a.process_files([FakeUpload("x.txt", b"content for x")])

        # New instance, same session_id — should load the persisted index.
        b = RAGService(persist_root=str(tmp_path), session_id="s1")
        assert b.load()
        assert b.has_index()
        assert "x.txt" in b.processed_files

    def test_missing_marker_refuses_load(self, tmp_path, monkeypatch):
        import rag_service as rs
        monkeypatch.setattr(rs, "build_embeddings", lambda p, k: FakeEmbeddings())
        a = RAGService(persist_root=str(tmp_path), session_id="s1")
        a.process_files([FakeUpload("x.txt", b"content")])
        # Corrupt: delete the marker.
        os.remove(a.marker_path)

        b = RAGService(persist_root=str(tmp_path), session_id="s1")
        assert not b.load()

    def test_corrupt_marker_refuses_load(self, tmp_path, monkeypatch):
        import rag_service as rs
        monkeypatch.setattr(rs, "build_embeddings", lambda p, k: FakeEmbeddings())
        a = RAGService(persist_root=str(tmp_path), session_id="s1")
        a.process_files([FakeUpload("x.txt", b"content")])
        with open(a.marker_path, "w") as f:
            f.write("{not json")

        b = RAGService(persist_root=str(tmp_path), session_id="s1")
        assert not b.load()


# ----------------------------------------------------- upload caps
class TestUploadCaps:
    def test_oversize_file_rejected_before_embedding(self, tmp_path, monkeypatch):
        import rag_service as rs
        monkeypatch.setattr(rs, "build_embeddings", lambda p, k: FakeEmbeddings())
        rl = RateLimiter(max_file_bytes=100, max_total_bytes=10_000)
        rag = RAGService(
            persist_root=str(tmp_path),
            session_id="s1",
            rate_limiter=rl,
        )
        big = FakeUpload("big.txt", b"X" * 500)
        with pytest.raises(RateLimitError, match="max per file"):
            rag.process_files([big])
        # Index must not have been created.
        assert not rag.has_index()
