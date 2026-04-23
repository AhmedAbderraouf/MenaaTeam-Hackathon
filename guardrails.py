"""Prompt-injection detection, self-harm safety layer, and refusal helpers."""
import re

# --------------------------------------------------------------- crisis / self-harm
# Teaching assistants are not therapists. When a user types something that
# looks like crisis content we must short-circuit all grading / QA / quiz
# logic and surface professional resources instead of a 0/100 score.
CRISIS_PATTERNS = [
    r"\bkill (?:my ?self|myself)\b",
    r"\bkms\b",
    r"\bsuicid(?:e|al)\b",
    r"\bend (?:my|this) life\b",
    r"\btake my (?:own )?life\b",
    r"\bwant(?:ing)? to die\b",
    r"\bi (?:just )?want to die\b",
    r"\bbetter off dead\b",
    r"\bno (?:reason|point) (?:to|in) liv(?:e|ing)\b",
    r"\bno reason to be alive\b",
    r"\bharm(?:ing)? (?:my ?self|myself)\b",
    r"\bhurt(?:ing)? (?:my ?self|myself)\b",
    r"\bself[- ]?harm\b",
    r"\bcut(?:ting)? (?:my ?self|myself)\b",
    r"\boverdose\b",
]

CRISIS_RESPONSE = (
    "I'm really sorry you're feeling this way — this is outside what a "
    "course-materials assistant can help with, but please reach out to "
    "someone who can:\n\n"
    "• **US / Canada:** call or text **988** (Suicide & Crisis Lifeline)\n"
    "• **UK & Ireland:** Samaritans — **116 123** (free, 24/7)\n"
    "• **Anywhere else:** find a hotline at "
    "https://findahelpline.com or https://www.iasp.info/resources/Crisis_Centres/\n"
    "• If you are in immediate danger, call your local emergency number.\n\n"
    "You deserve support from someone trained to help. Please talk to a "
    "person you trust or reach out to one of the services above."
)


def detect_crisis(text: str) -> bool:
    """True if the input looks like self-harm or suicide crisis content."""
    if not text:
        return False
    lowered = text.lower()
    return any(re.search(p, lowered) for p in CRISIS_PATTERNS)


INJECTION_PATTERNS = [
    r"ignore (?:all |any |the )?(?:previous|prior|above) (?:instructions?|prompts?|rules?)",
    r"disregard (?:all |any |the )?(?:previous|prior|above) (?:instructions?|prompts?|rules?)",
    r"forget (?:all |any |the )?(?:previous|prior|above) (?:instructions?|prompts?|rules?)",
    r"act as (?:a |an )?(?!teaching|ta|tutor|professor)",
    r"pretend (?:to be|you are)",
    r"you are now",
    r"system prompt",
    r"developer mode",
    r"jailbreak",
    r"reveal (?:your |the )?(?:prompt|instructions|system)",
    r"bypass (?:the )?(?:rules|guardrails|safety)",
    r"override (?:the )?(?:rules|system|instructions)",
]

REFUSAL_MESSAGE = (
    "I can only answer based on the provided course materials."
)


def detect_injection(text: str) -> bool:
    """Return True if the input looks like a prompt-injection attempt."""
    if not text:
        return False
    lowered = text.lower()
    return any(re.search(p, lowered) for p in INJECTION_PATTERNS)


def sanitize_user_input(text: str) -> str:
    """Strip embedded instruction markers so retrieved or user text cannot
    impersonate the system role when concatenated into a prompt."""
    if not text:
        return ""
    cleaned = re.sub(r"(?i)system\s*:", "", text)
    cleaned = re.sub(r"(?i)assistant\s*:", "", cleaned)
    return cleaned.strip()
