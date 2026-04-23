"""Centralized logging configuration.

One `get_logger(name)` helper used throughout the project. Logs go to stderr
with a consistent structured format; level is controlled via the
`TA_LOG_LEVEL` env var (default INFO).

Why not just `logging.getLogger(__name__)` directly?
- We want a single place to tweak format / level.
- We want `logging.basicConfig` to only run ONCE even if multiple modules
  import this (Streamlit reruns the script on every interaction).
"""
import logging
import os
import sys
from typing import Optional

_CONFIGURED = False


def _configure_once() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.getenv("TA_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    # Structured-ish single-line format: timestamp | level | logger | msg
    # Good enough for grep / log aggregators without pulling in a JSON lib.
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S"

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    root = logging.getLogger("ta")
    root.setLevel(level)
    # Wipe any handlers from a prior Streamlit rerun so we don't duplicate.
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.propagate = False  # don't double-log through the global root logger

    _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a namespaced logger under the 'ta' root.

    Example:
        log = get_logger(__name__)
        log.info("processed %d files", n)
    """
    _configure_once()
    if name:
        # Normalize so module path "rag_service" becomes "ta.rag_service"
        short = name.rsplit(".", 1)[-1]
        return logging.getLogger(f"ta.{short}")
    return logging.getLogger("ta")
