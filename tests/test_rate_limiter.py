"""RateLimiter — deterministic, no sleeps longer than a few ms."""
import time

import pytest

from rate_limiter import RateLimiter, RateLimitError


class TestLLMPerMinute:
    def test_under_cap_is_fine(self):
        rl = RateLimiter(llm_per_min=5, llm_per_session=100)
        for _ in range(5):
            rl.check_llm_call()  # should not raise

    def test_over_cap_raises(self):
        rl = RateLimiter(llm_per_min=3, llm_per_session=100)
        for _ in range(3):
            rl.check_llm_call()
        with pytest.raises(RateLimitError, match=r"\d+/min"):
            rl.check_llm_call()

    def test_window_slides_after_60s(self):
        rl = RateLimiter(llm_per_min=2, llm_per_session=100)
        rl.check_llm_call()
        rl.check_llm_call()
        # Manually age the timestamps so we don't sleep 60s in tests.
        rl._calls[0] -= 61
        rl._calls[1] -= 61
        # Third call should now succeed because the window slid.
        rl.check_llm_call()


class TestSessionCap:
    def test_session_cap_caps_total_regardless_of_rate(self):
        rl = RateLimiter(llm_per_min=1000, llm_per_session=3)
        for _ in range(3):
            rl.check_llm_call()
        with pytest.raises(RateLimitError, match="per-session limit"):
            rl.check_llm_call()


class TestUploadSize:
    def test_single_file_over_cap_rejected(self):
        rl = RateLimiter(max_file_bytes=1000)
        with pytest.raises(RateLimitError, match="max per file"):
            rl.check_upload("big.pdf", 2000)

    def test_accumulated_uploads_capped(self):
        rl = RateLimiter(max_file_bytes=1_000_000, max_total_bytes=1500)
        rl.check_upload("a.txt", 800)
        rl.check_upload("b.txt", 600)   # total=1400, still under
        with pytest.raises(RateLimitError, match="exceed session total"):
            rl.check_upload("c.txt", 200)   # would hit 1600

    def test_release_refunds_quota(self):
        rl = RateLimiter(max_total_bytes=1000)
        rl.check_upload("a.txt", 800)
        rl.release_upload(800)
        # After refund we should fit another 800-byte file.
        rl.check_upload("b.txt", 800)

    def test_status_reflects_state(self):
        rl = RateLimiter(llm_per_min=10, llm_per_session=100)
        rl.check_llm_call()
        rl.check_upload("a.txt", 500)
        s = rl.status()
        assert s["llm_minute"] == 1
        assert s["llm_session"] == 1
        assert s["uploaded_bytes"] == 500
