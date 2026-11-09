"""Visual-only analysis: metrics computed without text_only format.

Addresses the the review venue concern that text_only always dominates, making the
benchmark appear to show "visualization hurts".  By reporting visual-only
metrics separately, we reframe the finding as "current MLLMs have immature
visual reasoning" rather than "visualization is unnecessary".
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(slots=True)
class VisualOnlyResult:
    """Visual-only analysis for one model or modality slice.

    Attributes:
        visual_only_em: Mean EM excluding text_only records.
        text_only_em: Mean EM for text_only records only.
        text_gap_pp: text_only_em - visual_only_em in percentage points.
        best_visual: Best-performing visual format (excluding text_only).
        best_visual_em: EM of the best visual format.
        visual_sensitivity_std: Std-dev of EM across visual-only formats.
        visual_format_ranking: Formats sorted by EM descending.
    """

    visual_only_em: float
    text_only_em: float
    text_gap_pp: float
    best_visual: str
    best_visual_em: float
    visual_sensitivity_std: float
    visual_format_ranking: list[tuple[str, float]]


class VisualOnlyAnalyzer:
    """Analyze benchmark results with text_only excluded."""

    @staticmethod
    def filter_visual_only(
        records: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """Return records where viz_type is not text_only.

        Args:
            records: Evaluation result rows.

        Returns:
            Filtered list (does not mutate input).
        """
        return [r for r in records if str(r.get("viz_type", "")) != "text_only"]

    @staticmethod
    def filter_text_only(
        records: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """Return records where viz_type is text_only.

        Args:
            records: Evaluation result rows.

        Returns:
            Filtered list.
        """
        return [r for r in records if str(r.get("viz_type", "")) == "text_only"]

    def analyze(
        self,
        records: list[dict[str, object]],
    ) -> VisualOnlyResult:
        """Compute visual-only metrics from a set of evaluation records.

        Args:
            records: Evaluation result rows with ``viz_type`` and ``exact_match``.

        Returns:
            VisualOnlyResult with gap analysis and ranking.
        """
        visual = self.filter_visual_only(records)
        text = self.filter_text_only(records)

        visual_em = (
            sum(float(r.get("exact_match", 0.0)) for r in visual) / len(visual)
            if visual
            else 0.0
        )
        text_em = (
            sum(float(r.get("exact_match", 0.0)) for r in text) / len(text)
            if text
            else 0.0
        )
        text_gap = (text_em - visual_em) * 100.0

        # Per-format EM (visual only).
        by_format: dict[str, list[float]] = defaultdict(list)
        for r in visual:
            by_format[str(r.get("viz_type", ""))].append(
                float(r.get("exact_match", 0.0))
            )

        format_ems: dict[str, float] = {
            fmt: sum(vals) / len(vals) for fmt, vals in by_format.items() if vals
        }

        ranking = sorted(format_ems.items(), key=lambda x: -x[1])
        best_visual = ranking[0][0] if ranking else "n/a"
        best_visual_em = ranking[0][1] if ranking else 0.0

        # Visual-only sensitivity (std across formats).
        if len(format_ems) >= 2:
            vals = list(format_ems.values())
            mean = sum(vals) / len(vals)
            variance = sum((v - mean) ** 2 for v in vals) / len(vals)
            visual_std = variance**0.5
        else:
            visual_std = 0.0

        return VisualOnlyResult(
            visual_only_em=visual_em,
            text_only_em=text_em,
            text_gap_pp=text_gap,
            best_visual=best_visual,
            best_visual_em=best_visual_em,
            visual_sensitivity_std=visual_std,
            visual_format_ranking=ranking,
        )

    def analyze_by_modality(
        self,
        records: list[dict[str, object]],
    ) -> dict[str, VisualOnlyResult]:
        """Compute visual-only analysis per modality.

        Args:
            records: Evaluation result rows.

        Returns:
            Mapping modality -> VisualOnlyResult.
        """
        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for r in records:
            grouped[str(r.get("modality", "unknown"))].append(r)
        return {modality: self.analyze(group) for modality, group in sorted(grouped.items())}

    def analyze_by_model(
        self,
        records: list[dict[str, object]],
    ) -> dict[str, VisualOnlyResult]:
        """Compute visual-only analysis per model.

        Args:
            records: Evaluation result rows.

        Returns:
            Mapping model -> VisualOnlyResult.
        """
        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for r in records:
            grouped[str(r.get("model", "unknown"))].append(r)
        return {model: self.analyze(group) for model, group in sorted(grouped.items())}
