"""web_search retry / fallback behavior — no real network calls."""
import web_search


class _FakeDDGS:
    """Scriptable DDGS stand-in. `script` is a list of outcomes applied in
    order; each outcome is either `Exception(...)` to raise or a list of
    hits to return.
    """
    def __init__(self, script):
        self._script = script
        self._i = 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def text(self, query, max_results=3):
        outcome = self._script[self._i]
        self._i += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class TestWebSearchRetry:
    def test_succeeds_on_first_try(self, monkeypatch):
        script = [[
            {"title": "T", "href": "http://x", "body": "content"},
        ]]
        monkeypatch.setattr(web_search, "_load_ddgs", lambda: lambda: _FakeDDGS(script))
        web_search.last_error = "stale"
        docs = web_search.search_web("query")
        assert len(docs) == 1
        assert docs[0].metadata["url"] == "http://x"
        # Previous failure state must be cleared on success.
        assert web_search.last_error is None

    def test_retries_then_succeeds(self, monkeypatch):
        script = [
            ConnectionError("boom"),
            [{"title": "T", "href": "http://x", "body": "content"}],
        ]
        monkeypatch.setattr(web_search, "_load_ddgs", lambda: lambda: _FakeDDGS(script))
        # Make the backoff near-zero so this test stays fast.
        monkeypatch.setattr(web_search, "_BASE_BACKOFF_S", 0.001)
        docs = web_search.search_web("q")
        assert len(docs) == 1

    def test_persistent_failure_returns_empty_and_sets_last_error(self, monkeypatch):
        script = [ConnectionError("dns"), ConnectionError("dns"), ConnectionError("dns")]
        monkeypatch.setattr(web_search, "_load_ddgs", lambda: lambda: _FakeDDGS(script))
        monkeypatch.setattr(web_search, "_BASE_BACKOFF_S", 0.001)
        web_search.last_error = None
        docs = web_search.search_web("q")
        assert docs == []
        assert web_search.last_error is not None
        assert "ConnectionError" in web_search.last_error

    def test_no_library_returns_empty(self, monkeypatch):
        monkeypatch.setattr(web_search, "_load_ddgs", lambda: None)
        docs = web_search.search_web("q")
        assert docs == []
        assert "library not installed" in (web_search.last_error or "").lower()

    def test_empty_query_returns_empty_without_hitting_library(self, monkeypatch):
        called = {"n": 0}
        def _loader():
            called["n"] += 1
            return lambda: _FakeDDGS([])
        monkeypatch.setattr(web_search, "_load_ddgs", _loader)
        assert web_search.search_web("") == []
        assert web_search.search_web("   ", max_results=3) == []
        # The library loader should never have been called.
        assert called["n"] == 0
