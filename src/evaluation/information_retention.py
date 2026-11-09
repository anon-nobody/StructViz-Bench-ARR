"""Information retention analysis for visualization formats.

Separates two sources of MLLM performance degradation:
1. **Information loss**: The visualization itself discards data (e.g., GAF
   applies a cosine transform that destroys exact values).
2. **Reasoning failure**: The data is present in the image but the model
   fails to extract or reason about it.

By assigning each visualization a *value retrievability score* and correlating
it with EM, we can attribute performance gaps to the right cause.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


# ── Retrievability scores ────────────────────────────────────────────────────
#
# These are expert-assigned scores (0 = no individual values recoverable,
# 1 = all values exactly readable) based on the information-theoretic
# properties of each rendering.  They are constant per format, not learned.
#
# Justifications:
#   text_only:        All values present as text → 1.0
#   table_image:      Rendered table with exact numbers → 0.95
#   bar_chart:        Values readable from bar heights + axis → 0.80
#   line_plot:        Values readable at discrete points → 0.75
#   heatmap:          Color-to-value mapping with colorbar → 0.55
#   scatter_plot:     Individual points readable but overlap → 0.60
#   node_link:        Structure visible, labels if <30 nodes → 0.70
#   circular_layout:  Same as node_link, denser layout → 0.65
#   adjacency_matrix: Binary presence readable from grid → 0.50
#   gaf:              Cosine of scaled values; original values not
#                     recoverable without inverse transform → 0.25
#   recurrence_plot:  Binary thresholded distance matrix;
#                     temporal order partially lost → 0.15

_DEFAULT_RETRIEVABILITY: dict[str, float] = {
    # ── Tabular ──
    "text_only": 1.00,
    "table_image": 0.95,
    "bar_chart": 0.80,
    "scatter_plot": 0.60,
    "heatmap": 0.55,
    # ── Time Series ──
    "line_plot": 0.75,
    # "heatmap" already defined above (shared key)
    "gaf": 0.25,
    "recurrence_plot": 0.15,
    # ── Graph ──
    "node_link": 0.70,
    "circular_layout": 0.65,
    "adjacency_matrix": 0.50,
}


@dataclass(slots=True)
class RetentionAnalysisResult:
    """Information retention analysis for one visualization format.

    Attributes:
        viz_type: Visualization type name.
        retrievability: Expert-assigned value retrievability score.
        mean_em: Mean EM for this visualization across evaluated items.
        efficiency: EM / retrievability — how well the model utilizes
            the available information (>1.0 unlikely but possible with
            lucky guessing on low-retrievability formats).
        category: "information_loss" if retrievability < 0.5 and EM is
            proportionally low, "reasoning_failure" if retrievability is
            high but EM is low, "adequate" otherwise.
    """

    viz_type: str
    retrievability: float
    mean_em: float
    efficiency: float
    category: str


class InformationRetentionAnalyzer:
    """Analyze the relationship between information retention and EM."""

    def __init__(
        self,
        retrievability_scores: dict[str, float] | None = None,
    ) -> None:
        self._scores = dict(retrievability_scores or _DEFAULT_RETRIEVABILITY)

    def get_retrievability(self, viz_type: str) -> float:
        """Look up the retrievability score for a visualization type.

        Args:
            viz_type: Visualization type name.

        Returns:
            Retrievability in [0, 1], defaulting to 0.5 for unknown types.
        """
        return self._scores.get(viz_type, 0.5)

    def _categorize(self, retrievability: float, mean_em: float) -> str:
        """Categorize the performance gap source."""
        if retrievability < 0.4:
            return "information_loss"
        if retrievability >= 0.6 and mean_em < 0.30:
            return "reasoning_failure"
        if mean_em >= 0.40:
            return "adequate"
        return "mixed"

    def analyze_per_viz(
        self,
        records: list[dict[str, object]],
    ) -> list[RetentionAnalysisResult]:
        """Compute retention analysis grouped by visualization type.

        Args:
            records: Evaluation result rows with ``viz_type`` and
                ``exact_match`` fields.

        Returns:
            Sorted list of per-viz-type retention results.
        """
        by_viz: dict[str, list[float]] = defaultdict(list)
        for r in records:
            viz = str(r.get("viz_type", ""))
            by_viz[viz].append(float(r.get("exact_match", 0.0)))

        results: list[RetentionAnalysisResult] = []
        for viz_type in sorted(by_viz):
            ems = by_viz[viz_type]
            mean_em = sum(ems) / len(ems) if ems else 0.0
            retrievability = self.get_retrievability(viz_type)
            efficiency = mean_em / retrievability if retrievability > 0.0 else 0.0
            category = self._categorize(retrievability, mean_em)
            results.append(
                RetentionAnalysisResult(
                    viz_type=viz_type,
                    retrievability=retrievability,
                    mean_em=mean_em,
                    efficiency=efficiency,
                    category=category,
                )
            )
        return results

    def compute_correlation(
        self,
        records: list[dict[str, object]],
    ) -> float:
        """Pearson correlation between retrievability and EM across viz types.

        A high positive correlation supports the claim that performance gaps
        are driven by information loss rather than reasoning failure.

        Args:
            records: Evaluation result rows.

        Returns:
            Pearson r in [-1, 1], or 0.0 if insufficient data.
        """
        analysis = self.analyze_per_viz(records)
        if len(analysis) < 3:
            return 0.0

        xs = [a.retrievability for a in analysis]
        ys = [a.mean_em for a in analysis]
        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n

        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / n
        std_x = (sum((x - mean_x) ** 2 for x in xs) / n) ** 0.5
        std_y = (sum((y - mean_y) ** 2 for y in ys) / n) ** 0.5

        if std_x == 0.0 or std_y == 0.0:
            return 0.0
        return cov / (std_x * std_y)
