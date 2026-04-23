"""Optional DuckDuckGo web search with retries.

Returns LangChain Document objects with source metadata so web results can be
merged with RAG retrieval and rendered by the existing format_context /
format_sources helpers.

Reliability:
- Retries transient failures (DNS, connect, rate-limit) with exponential
  backoff. DDG is flaky — one ConnectError should not surface as a broken
  web-search feature.
- On final failure, returns `[]` AND sets a module-level `last_error` string
  so the UI can show a "web search unavailable" notice instead of silently
  going grounded-only. Logs every failure.
"""
from __future__ import annotations

import time
from typing import List, Optional

from langchain_core.documents import Document

from logging_setup import get_logger

log = get_logger(__name__)

# Number of attempts and the base backoff. Tight on purpose — this sits inside
# a user-visible request path, so we'd rather return quickly with `[]` than
# hang for 30s retrying a dead service.
_MAX_ATTEMPTS = 3
_BASE_BACKOFF_S = 0.5  # 0.5s, 1.0s, 2.0s

# Last non-empty error message from a failed call. Used by the UI to show a
# "web search unavailable: <reason>" banner.
last_error: Optional[str] = None


def _load_ddgs():
    """The library was renamed duckduckgo_search → ddgs; support both."""
    try:
        from ddgs import DDGS  # type: ignore
        return DDGS
    except ImportError:
        pass
    try:
        from duckduckgo_search import DDGS  # type: ignore
        return DDGS
    except ImportError:
        log.info("ddgs / duckduckgo_search not installed; web search disabled")
        return None


def search_web(query: str, max_results: int = 3) -> List[Document]:
    """Run a DuckDuckGo search with retries. Returns [] on persistent failure."""
    global last_error
    if not query or max_results <= 0:
        return []

    DDGS = _load_ddgs()
    if DDGS is None:
        last_error = "DuckDuckGo library not installed (pip install ddgs)."
        return []

    last_exc: Optional[BaseException] = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            with DDGS() as ddgs:
                hits = list(ddgs.text(query, max_results=max_results))
            # Success clears the previous failure state so the UI banner goes
            # away after recovery.
            last_error = None
            log.debug("ddg query ok on attempt %d: %d hits", attempt, len(hits))
            return _to_documents(hits)
        except Exception as e:  # noqa: BLE001 - we genuinely want all failures
            last_exc = e
            backoff = _BASE_BACKOFF_S * (2 ** (attempt - 1))
            log.warning(
                "ddg query failed (attempt %d/%d): %s: %s — retrying in %.1fs",
                attempt, _MAX_ATTEMPTS, type(e).__name__, str(e)[:200], backoff,
            )
            if attempt < _MAX_ATTEMPTS:
                time.sleep(backoff)

    # All attempts exhausted.
    last_error = f"{type(last_exc).__name__}: {str(last_exc)[:200]}"
    log.error("ddg query failed after %d attempts: %s", _MAX_ATTEMPTS, last_error)
    return []


def _to_documents(hits: list) -> List[Document]:
    docs: List[Document] = []
    for h in hits:
        title = (h.get("title") or "").strip() or "(untitled)"
        url = (h.get("href") or h.get("url") or "").strip()
        body = (h.get("body") or h.get("snippet") or "").strip()
        if not body:
            continue
        docs.append(
            Document(
                page_content=f"{title}\n{body}",
                metadata={
                    "source_name": url or title,
                    "web": True,
                    "title": title,
                    "url": url,
                },
            )
        )
    return docs
