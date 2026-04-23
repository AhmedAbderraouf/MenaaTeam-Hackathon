"""Per-session rate limiting and cost controls.

Three independent guards per session:
- `llm_calls_per_min`: sliding-window request limiter for any LLM call
  (answer / grade / quiz / plan). Prevents burst abuse.
- `llm_calls_per_session`: hard ceiling on LLM calls per session so a single
  user can't rack up unbounded API cost.
- Upload size caps: max-bytes-per-file and max-bytes-total-per-session,
  enforced by RAGService before any embedding cost is incurred.

All limits are configurable via env vars. Defaults target a classroom-style
deployment (generous enough for normal use, small enough to cap abuse).

The `RateLimiter` object is designed to live in `st.session_state`, one per
user session, so counts naturally reset when the session ends.
"""
from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

from logging_setup import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------- defaults
def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        log.warning("Invalid %s env value; using default %d", name, default)
        return default


DEFAULT_LLM_PER_MIN       = _int_env("TA_LLM_RPM",        20)     # 20 req/min
DEFAULT_LLM_PER_SESSION   = _int_env("TA_LLM_SESSION_CAP", 200)   # 200/session
DEFAULT_MAX_FILE_BYTES    = _int_env("TA_MAX_FILE_BYTES",  200 * 1024 * 1024)  # 200 MB
DEFAULT_MAX_TOTAL_BYTES   = _int_env("TA_MAX_TOTAL_BYTES", 200 * 1024 * 1024)  # 200 MB


class RateLimitError(RuntimeError):
    """Raised when a session exceeds a configured cap.

    The message is safe to surface to users via `st.error(...)`.
    """


@dataclass
class RateLimiter:
    llm_per_min: int = DEFAULT_LLM_PER_MIN
    llm_per_session: int = DEFAULT_LLM_PER_SESSION
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES

    # Mutable state - per session.
    _calls: Deque[float] = field(default_factory=deque)   # timestamps of LLM calls
    _session_total_calls: int = 0
    _uploaded_bytes: int = 0

    # ------------------------------------------------------------- LLM calls
    def check_llm_call(self) -> None:
        """Raise RateLimitError if this LLM call would exceed either cap."""
        now = time.time()
        # Drop timestamps older than 60s (sliding window).
        cutoff = now - 60.0
        while self._calls and self._calls[0] < cutoff:
            self._calls.popleft()

        if self._session_total_calls >= self.llm_per_session:
            log.warning(
                "session LLM cap hit: %d/%d",
                self._session_total_calls, self.llm_per_session,
            )
            raise RateLimitError(
                f"You've reached the per-session limit of "
                f"{self.llm_per_session} requests. Reload the page to start a "
                f"fresh session."
            )
        if len(self._calls) >= self.llm_per_min:
            oldest = self._calls[0]
            wait_s = max(1, int(60 - (now - oldest)))
            log.info("per-minute cap hit: %d in last minute", len(self._calls))
            raise RateLimitError(
                f"Too many requests — limit is {self.llm_per_min}/min. "
                f"Please wait ~{wait_s}s and try again."
            )

        # Reserve the slot now. We count the intent to call; failed calls still
        # count because they burn partial cost (network, tokens on error).
        self._calls.append(now)
        self._session_total_calls += 1
        log.debug(
            "llm_call reserved: rpm=%d session_total=%d",
            len(self._calls), self._session_total_calls,
        )

    # ------------------------------------------------------------- uploads
    def check_upload(self, filename: str, size_bytes: int) -> None:
        """Raise RateLimitError if this upload would exceed size caps."""
        if size_bytes > self.max_file_bytes:
            log.warning(
                "oversize file rejected: %s %d > %d",
                filename, size_bytes, self.max_file_bytes,
            )
            raise RateLimitError(
                f"'{filename}' is {_fmt_mb(size_bytes)} — max per file is "
                f"{_fmt_mb(self.max_file_bytes)}."
            )
        projected = self._uploaded_bytes + size_bytes
        if projected > self.max_total_bytes:
            log.warning(
                "session upload cap would be exceeded: %d + %d > %d",
                self._uploaded_bytes, size_bytes, self.max_total_bytes,
            )
            raise RateLimitError(
                f"Upload would exceed session total "
                f"({_fmt_mb(self.max_total_bytes)}). Remove some files first."
            )
        self._uploaded_bytes = projected
        log.debug(
            "upload accepted: %s (%d bytes, session total=%d)",
            filename, size_bytes, self._uploaded_bytes,
        )

    def release_upload(self, size_bytes: int) -> None:
        """Called when a file is removed from the index — refunds quota."""
        self._uploaded_bytes = max(0, self._uploaded_bytes - size_bytes)

    # ----------------------------------------------------------- diagnostics
    def status(self) -> dict:
        """Snapshot used by the UI to show current usage."""
        return {
            "llm_minute": len(self._calls),
            "llm_minute_cap": self.llm_per_min,
            "llm_session": self._session_total_calls,
            "llm_session_cap": self.llm_per_session,
            "uploaded_bytes": self._uploaded_bytes,
            "max_total_bytes": self.max_total_bytes,
        }


def _fmt_mb(n: int) -> str:
    return f"{n / (1024 * 1024):.1f} MB"
