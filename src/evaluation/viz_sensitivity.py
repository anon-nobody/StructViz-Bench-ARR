from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class VizSensitivityResult:
    """Per-question sensitivity analysis across visualization formats.

    Attributes:
        sensitivity_score: Std-dev of accuracies across viz types.
        best_viz: Visualization type with highest accuracy.
        worst_viz: Visualization type with lowest accuracy.
        accuracies: Mapping viz_type -> accuracy.
        flip_rate: Fraction of viz-pair transitions where correctness changes.
        consistency_ratio: 1.0 if all viz types agree, 0.0 otherwise.
        gap_pp: Best - worst accuracy in percentage points.
        visual_only_sensitivity: Std-dev excluding text_only format.
    """

    sensitivity_score: float
    best_viz: str
    worst_viz: str
    accuracies: dict[str, float]
    flip_rate: float = 0.0
    consistency_ratio: float = 1.0
    gap_pp: float = 0.0
    visual_only_sensitivity: float = 0.0


class VizSensitivityAnalyzer:
    """Analyze model sensitivity to visualization format changes."""

    @staticmethod
    def _std(values: list[float]) -> float:
        """Population standard deviation."""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return variance**0.5

    @staticmethod
    def compute_flip_rate(viz_accuracy: dict[str, float]) -> float:
        """Fraction of visualization pairs where correctness status differs.

        For binary EM values (0/1), this measures how often switching
        the visualization format flips the answer from correct to incorrect
        or vice versa.

        Args:
            viz_accuracy: Mapping viz_type -> accuracy (0.0 or 1.0 for EM).

        Returns:
            Flip rate in [0, 1]. 0 = all formats agree, 1 = every pair disagrees.
        """
        values = list(viz_accuracy.values())
        if len(values) < 2:
            return 0.0
        pairs = 0
        flips = 0
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                pairs += 1
                if values[i] != values[j]:
                    flips += 1
        return flips / pairs if pairs > 0 else 0.0

    @staticmethod
    def compute_consistency(viz_accuracy: dict[str, float]) -> float:
        """Whether all visualization formats produce the same correctness.

        Args:
            viz_accuracy: Mapping viz_type -> accuracy (0.0 or 1.0 for EM).

        Returns:
            1.0 if all formats agree (all correct or all wrong), 0.0 otherwise.
        """
        values = list(viz_accuracy.values())
        if len(values) < 2:
            return 1.0
        return 1.0 if len(set(values)) == 1 else 0.0

    def analyze(self, viz_accuracy: dict[str, float]) -> VizSensitivityResult:
        """Compute full sensitivity analysis for one question across viz types."""
        if not viz_accuracy:
            return VizSensitivityResult(0.0, "n/a", "n/a", {})

        values = list(viz_accuracy.values())
        best_viz = max(viz_accuracy.items(), key=lambda item: item[1])[0]
        worst_viz = min(viz_accuracy.items(), key=lambda item: item[1])[0]

        sensitivity = self._std(values)
        flip_rate = self.compute_flip_rate(viz_accuracy)
        consistency = self.compute_consistency(viz_accuracy)
        gap_pp = (max(values) - min(values)) * 100.0

        # Visual-only sensitivity (excluding text_only).
        visual_only = {k: v for k, v in viz_accuracy.items() if k != "text_only"}
        visual_only_sensitivity = self._std(list(visual_only.values()))

        return VizSensitivityResult(
            sensitivity_score=sensitivity,
            best_viz=best_viz,
            worst_viz=worst_viz,
            accuracies=viz_accuracy,
            flip_rate=flip_rate,
            consistency_ratio=consistency,
            gap_pp=gap_pp,
            visual_only_sensitivity=visual_only_sensitivity,
        )

    def analyze_group(
        self,
        grouped: dict[tuple[str, str], dict[str, float]],
    ) -> dict[tuple[str, str], VizSensitivityResult]:
        """Analyze all question-model groups keyed by (question_id, model_name)."""
        return {key: self.analyze(viz_scores) for key, viz_scores in grouped.items()}

    def aggregate_flip_rate(
        self,
        results: dict[tuple[str, str], VizSensitivityResult],
    ) -> float:
        """Average flip rate across all analyzed question-model groups."""
        if not results:
            return 0.0
        return sum(r.flip_rate for r in results.values()) / len(results)

    def aggregate_consistency(
        self,
        results: dict[tuple[str, str], VizSensitivityResult],
    ) -> float:
        """Average consistency ratio across all analyzed question-model groups."""
        if not results:
            return 1.0
        return sum(r.consistency_ratio for r in results.values()) / len(results)
