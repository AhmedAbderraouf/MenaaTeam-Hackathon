"""Per-session student profile used to adapt explanations and quiz difficulty."""
from dataclasses import dataclass, field

EDUCATION_LEVELS = ["high school", "undergraduate", "graduate"]
FAMILIARITY_LEVELS = ["beginner", "intermediate", "advanced"]


@dataclass
class StudentProfile:
    education_level: str = "undergraduate"
    familiarity: str = "beginner"
    learning_goal: str = ""

    # Difficulty baseline 1..5 derived from familiarity
    def baseline_difficulty(self) -> int:
        return {"beginner": 2, "intermediate": 3, "advanced": 4}.get(
            self.familiarity, 2
        )

    def style_guidance(self) -> str:
        """Instruction snippet injected into LLM system prompts."""
        if self.familiarity == "beginner":
            tone = (
                "Use simple language, short sentences, and at least one concrete "
                "example or analogy. Define any jargon before using it."
            )
        elif self.familiarity == "advanced":
            tone = (
                "Be concise and technical. Skip introductory background. Use precise "
                "terminology and focus on nuance, trade-offs, and edge cases."
            )
        else:
            tone = (
                "Be balanced: explain the idea clearly, then add one technical detail "
                "or example."
            )
        goal = (
            f" The student's learning goal is: {self.learning_goal}."
            if self.learning_goal
            else ""
        )
        return (
            f"Audience: {self.education_level} student, {self.familiarity} level. "
            f"{tone}{goal}"
        )
