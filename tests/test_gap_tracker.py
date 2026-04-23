"""GapTracker — topic extraction + adaptive difficulty."""
from gap_tracker import GapTracker, extract_topic


class TestExtractTopic:
    def test_returns_first_content_word(self):
        assert extract_topic("what is recursion?") == "recursion"
        assert extract_topic("how do lists work?") == "lists"

    def test_ignores_stopwords(self):
        # "what", "is", "a" all filtered — returns the real content word.
        assert extract_topic("what is a function") == "function"

    def test_returns_none_when_only_stopwords(self):
        assert extract_topic("what is it") is None
        assert extract_topic("") is None
        assert extract_topic(None) is None


class TestDifficultyAdjust:
    def test_three_correct_raises_difficulty(self):
        t = GapTracker()
        t.set_baseline(3)
        for _ in range(3):
            t.record_result("topicA", True)
        assert t.difficulty == 4

    def test_three_wrong_lowers_difficulty(self):
        t = GapTracker()
        t.set_baseline(3)
        for _ in range(3):
            t.record_result("topicA", False)
        assert t.difficulty == 2

    def test_bounded_at_min_and_max(self):
        t = GapTracker()
        t.set_baseline(5)
        for _ in range(10):
            t.record_result("x", True)
        assert t.difficulty == 5  # capped

        t2 = GapTracker()
        t2.set_baseline(1)
        for _ in range(10):
            t2.record_result("x", False)
        assert t2.difficulty == 1  # capped


class TestWeakTopics:
    def test_topic_becomes_weak_after_repeated_errors(self):
        t = GapTracker()
        t.record_result("loops", False)
        t.record_result("loops", False)
        # Two incorrect, no correct → weak.
        assert "loops" in t.weak_topics()

    def test_topic_not_weak_if_mostly_correct(self):
        t = GapTracker()
        t.record_result("loops", True)
        t.record_result("loops", True)
        t.record_result("loops", False)
        assert "loops" not in t.weak_topics()

    def test_repeated_question_asks_surface_as_weak(self):
        t = GapTracker()
        for _ in range(3):
            t.record_question_asked("pointers")
        assert "pointers" in t.weak_topics()
