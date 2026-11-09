from __future__ import annotations

from enum import Enum


class Difficulty(str, Enum):
    ONE_HOP = "1-hop"
    TWO_HOP = "2-hop"
    THREE_HOP = "3-hop"
    COUNTERFACTUAL = "counterfactual"


class DifficultyClassifier:
    """Assign reasoning difficulty labels for benchmark QA items."""

    def classify(
        self,
        reasoning_steps: int,
        requires_arithmetic: bool,
        has_counterfactual: bool,
    ) -> Difficulty:
        """Classify by steps, arithmetic, and counterfactual requirement."""
        if has_counterfactual:
            return Difficulty.COUNTERFACTUAL
        if reasoning_steps <= 1 and not requires_arithmetic:
            return Difficulty.ONE_HOP
        if reasoning_steps <= 2:
            return Difficulty.TWO_HOP
        return Difficulty.THREE_HOP

    def classify_from_metadata(self, metadata: dict[str, object]) -> Difficulty:
        """Classify from metadata dictionary fields."""
        steps_raw = metadata.get("reasoning_steps", 1)
        steps = int(steps_raw) if isinstance(steps_raw, (int, float, str)) else 1
        needs_math = bool(metadata.get("requires_arithmetic", False))
        is_counterfactual = bool(metadata.get("counterfactual", False))
        return self.classify(
            reasoning_steps=steps,
            requires_arithmetic=needs_math,
            has_counterfactual=is_counterfactual,
        )
