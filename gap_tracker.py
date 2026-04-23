"""Tracks per-session weak topics and adaptive quiz difficulty (1..5)."""
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# Stopwords used to keep common question words ("what", "can", "how"...) out of
# the weak-topic list. Kept deliberately small — this is a rough heuristic, not
# a full NLP pipeline.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "could",
    "do", "does", "did", "for", "from", "has", "have", "had", "how", "i", "if",
    "in", "is", "it", "its", "me", "my", "not", "of", "on", "or", "please",
    "provide", "show", "so", "some", "tell", "than", "that", "the", "their",
    "then", "there", "they", "this", "to", "us", "was", "we", "were", "what",
    "when", "where", "which", "who", "why", "will", "with", "would", "you",
    "your", "about", "more", "information", "explain", "describe", "define",
    "give", "help", "example", "examples",
}


def extract_topic(text: str) -> Optional[str]:
    """Pick the first non-stopword token from a question.

    Returns None when nothing meaningful is left — the caller should skip
    recording in that case instead of inventing a topic like "what".
    """
    if not text:
        return None
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]{1,}", text.lower())
    for tok in tokens:
        if tok not in _STOPWORDS and len(tok) > 2:
            return tok
    return None


@dataclass
class GapTracker:
    baseline: int = 2
    difficulty: int = 2
    correct: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    incorrect: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    topic_asks: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    recent_streak: List[bool] = field(default_factory=list)  # True = correct

    MIN = 1
    MAX = 5

    def set_baseline(self, level: int) -> None:
        self.baseline = max(self.MIN, min(self.MAX, level))
        self.difficulty = self.baseline

    def record_question_asked(self, topic: Optional[str]) -> None:
        """Only records if `topic` is a meaningful, non-stopword term."""
        if not topic:
            return
        topic = topic.lower().strip()
        if not topic or topic in _STOPWORDS or len(topic) <= 2:
            return
        self.topic_asks[topic] += 1

    def record_result(self, topic: Optional[str], is_correct: bool) -> None:
        topic = (topic or "general").lower().strip()
        if topic in _STOPWORDS or len(topic) <= 2:
            topic = "general"
        if is_correct:
            self.correct[topic] += 1
        else:
            self.incorrect[topic] += 1
        self.recent_streak.append(is_correct)
        self.recent_streak = self.recent_streak[-5:]
        self._adjust_difficulty()

    def _adjust_difficulty(self) -> None:
        if len(self.recent_streak) < 3:
            return
        recent = self.recent_streak[-3:]
        score = sum(recent)
        if score >= 3 and self.difficulty < self.MAX:
            self.difficulty += 1
        elif score <= 1 and self.difficulty > self.MIN:
            self.difficulty -= 1

    def weak_topics(self, min_errors: int = 2) -> List[str]:
        weak = [
            t
            for t, errs in self.incorrect.items()
            if errs >= min_errors and errs > self.correct.get(t, 0)
        ]
        weak += [
            t for t, n in self.topic_asks.items()
            if n >= 3 and t not in weak
        ]
        return [t for t in weak if t != "general"]

    def difficulty_label(self) -> str:
        return {
            1: "very easy (recall)",
            2: "easy (basic concepts)",
            3: "medium (application)",
            4: "hard (analysis)",
            5: "very hard (synthesis / edge cases)",
        }[self.difficulty]

    def suggestion(self) -> str:
        weak = self.weak_topics()
        if not weak:
            return ""
        topics = ", ".join(weak[:3])
        return f"You seem to struggle with {topics}. Want more practice on these?"
