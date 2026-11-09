"""Paired statistical analysis for visualization sensitivity.

Leverages the repeated-measures structure of StructViz-Bench (same question
evaluated across multiple visualization formats) to compute item-level
sensitivity metrics with proper paired statistics.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass


@dataclass(slots=True)
class PairedSensitivityResult:
    """Aggregate paired sensitivity statistics.

    Attributes:
        flip_rate: Fraction of items where at least one viz pair disagrees.
        consistency_rate: Fraction of items where all viz types agree.
        mean_gap_pp: Mean best-worst gap in percentage points.
        median_gap_pp: Median best-worst gap in percentage points.
        gap_ci_lower: 95% bootstrap CI lower bound for mean gap.
        gap_ci_upper: 95% bootstrap CI upper bound for mean gap.
        n_items: Number of items analyzed.
    """

    flip_rate: float
    consistency_rate: float
    mean_gap_pp: float
    median_gap_pp: float
    gap_ci_lower: float
    gap_ci_upper: float
    n_items: int


@dataclass(slots=True)
class PairedComparisonResult:
    """Paired comparison between two visualization types.

    Attributes:
        viz_a: First visualization type.
        viz_b: Second visualization type.
        mean_diff: Mean EM difference (A - B).
        ci_lower: 95% bootstrap CI lower bound.
        ci_upper: 95% bootstrap CI upper bound.
        p_value: Two-sided bootstrap p-value.
        n_paired: Number of paired items.
        effect_size: Cohen's d effect size.
    """

    viz_a: str
    viz_b: str
    mean_diff: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n_paired: int
    effect_size: float


class PairedAnalyzer:
    """Paired analysis exploiting the repeated-measures benchmark design."""

    def __init__(self, seed: int = 42, n_bootstrap: int = 10000) -> None:
        self._seed = seed
        self._n_boot = n_bootstrap

    def _group_by_item(
        self,
        records: list[dict[str, object]],
    ) -> dict[str, dict[str, float]]:
        """Group records by question_id → {viz_type: exact_match}."""
        grouped: dict[str, dict[str, float]] = defaultdict(dict)
        for r in records:
            qid = str(r.get("question_id", ""))
            viz = str(r.get("viz_type", ""))
            em = float(r.get("exact_match", 0.0))
            grouped[qid][viz] = em
        return dict(grouped)

    def compute_item_sensitivity(
        self,
        records: list[dict[str, object]],
        exclude_text_only: bool = False,
    ) -> PairedSensitivityResult:
        """Compute item-level sensitivity with bootstrap CIs.

        Args:
            records: Evaluation result rows.
            exclude_text_only: If True, exclude text_only from analysis.

        Returns:
            PairedSensitivityResult with flip rate, gap stats, and CIs.
        """
        grouped = self._group_by_item(records)
        gaps: list[float] = []
        flips = 0
        consistent = 0

        for viz_scores in grouped.values():
            scores = {
                k: v for k, v in viz_scores.items()
                if not (exclude_text_only and k == "text_only")
            }
            if len(scores) < 2:
                continue
            vals = list(scores.values())
            gap = (max(vals) - min(vals)) * 100.0
            gaps.append(gap)
            unique = set(vals)
            if len(unique) > 1:
                flips += 1
            else:
                consistent += 1

        n = len(gaps)
        if n == 0:
            return PairedSensitivityResult(
                flip_rate=0.0,
                consistency_rate=1.0,
                mean_gap_pp=0.0,
                median_gap_pp=0.0,
                gap_ci_lower=0.0,
                gap_ci_upper=0.0,
                n_items=0,
            )

        mean_gap = sum(gaps) / n
        sorted_gaps = sorted(gaps)
        median_gap = sorted_gaps[n // 2]

        # Bootstrap CI for mean gap.
        import random

        rng = random.Random(self._seed)
        boot_means: list[float] = []
        for _ in range(self._n_boot):
            sample = [rng.choice(gaps) for _ in range(n)]
            boot_means.append(sum(sample) / n)
        boot_means.sort()
        ci_lower = boot_means[int(self._n_boot * 0.025)]
        ci_upper = boot_means[int(self._n_boot * 0.975)]

        total = flips + consistent
        return PairedSensitivityResult(
            flip_rate=flips / total,
            consistency_rate=consistent / total,
            mean_gap_pp=mean_gap,
            median_gap_pp=median_gap,
            gap_ci_lower=ci_lower,
            gap_ci_upper=ci_upper,
            n_items=total,
        )

    def paired_viz_comparison(
        self,
        records: list[dict[str, object]],
        viz_a: str,
        viz_b: str,
    ) -> PairedComparisonResult:
        """Paired bootstrap comparison between two visualization types.

        Args:
            records: Evaluation result rows.
            viz_a: First visualization type.
            viz_b: Second visualization type.

        Returns:
            PairedComparisonResult with mean diff, CI, p-value, effect size.
        """
        grouped = self._group_by_item(records)
        diffs: list[float] = []
        for viz_scores in grouped.values():
            if viz_a in viz_scores and viz_b in viz_scores:
                diffs.append(viz_scores[viz_a] - viz_scores[viz_b])

        n = len(diffs)
        if n < 3:
            return PairedComparisonResult(
                viz_a=viz_a,
                viz_b=viz_b,
                mean_diff=0.0,
                ci_lower=0.0,
                ci_upper=0.0,
                p_value=1.0,
                n_paired=n,
                effect_size=0.0,
            )

        observed = sum(diffs) / n
        std = (sum((d - observed) ** 2 for d in diffs) / n) ** 0.5
        effect_size = observed / std if std > 0 else 0.0

        import random

        rng = random.Random(self._seed)
        boot_means: list[float] = []
        for _ in range(self._n_boot):
            sample = [rng.choice(diffs) for _ in range(n)]
            boot_means.append(sum(sample) / n)
        boot_means.sort()

        ci_lower = boot_means[int(self._n_boot * 0.025)]
        ci_upper = boot_means[int(self._n_boot * 0.975)]
        p_value = 2.0 * min(
            sum(1 for b in boot_means if b <= 0) / self._n_boot,
            sum(1 for b in boot_means if b >= 0) / self._n_boot,
        )

        return PairedComparisonResult(
            viz_a=viz_a,
            viz_b=viz_b,
            mean_diff=observed,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=min(p_value, 1.0),
            n_paired=n,
            effect_size=effect_size,
        )

    @staticmethod
    def holm_bonferroni_correction(
        p_values: list[float],
        alpha: float = 0.05,
    ) -> list[dict[str, float | bool]]:
        """Apply Holm-Bonferroni step-down correction to a list of p-values.

        Controls family-wise error rate at level alpha across multiple
        comparisons. The smallest p-value is compared to alpha/m, the next
        to alpha/(m-1), etc., where m is the total number of comparisons.

        Args:
            p_values: Raw two-sided p-values from independent tests.
            alpha: Family-wise error rate (default 0.05).

        Returns:
            List of dicts with raw_p, adjusted_p, threshold, reject (bool),
            in the same order as input p_values.
        """
        m = len(p_values)
        if m == 0:
            return []
        # Sort indices by p-value ascending.
        order = sorted(range(m), key=lambda i: p_values[i])
        results: list[dict[str, float | bool]] = [
            {"raw_p": 0.0, "adjusted_p": 0.0, "threshold": 0.0, "reject": False}
            for _ in range(m)
        ]
        prev_adjusted = 0.0
        for rank, idx in enumerate(order):
            p = p_values[idx]
            threshold = alpha / (m - rank)
            # Adjusted p = max over current and all earlier of (m-k)*p_k.
            adjusted = max(prev_adjusted, (m - rank) * p)
            adjusted = min(adjusted, 1.0)
            prev_adjusted = adjusted
            results[idx] = {
                "raw_p": p,
                "adjusted_p": adjusted,
                "threshold": threshold,
                "reject": adjusted < alpha,
            }
        return results

    def best_vs_worst_paired(
        self,
        records: list[dict[str, object]],
        modality: str | None = None,
        exclude_text_only: bool = False,
    ) -> PairedComparisonResult:
        """Paired comparison between best and worst viz types by mean EM.

        Args:
            records: Evaluation result rows.
            modality: Optional modality filter.
            exclude_text_only: If True, exclude text_only.

        Returns:
            PairedComparisonResult for best vs worst.
        """
        filtered = records
        if modality:
            filtered = [r for r in records if str(r.get("modality", "")) == modality]
        if exclude_text_only:
            filtered = [r for r in filtered if str(r.get("viz_type", "")) != "text_only"]

        # Find best and worst viz types by aggregate EM.
        by_viz: dict[str, list[float]] = defaultdict(list)
        for r in filtered:
            by_viz[str(r.get("viz_type", ""))].append(float(r.get("exact_match", 0.0)))

        if len(by_viz) < 2:
            return PairedComparisonResult(
                viz_a="n/a", viz_b="n/a", mean_diff=0.0,
                ci_lower=0.0, ci_upper=0.0, p_value=1.0,
                n_paired=0, effect_size=0.0,
            )

        viz_means = {v: sum(ems) / len(ems) for v, ems in by_viz.items()}
        best = max(viz_means, key=lambda v: viz_means[v])
        worst = min(viz_means, key=lambda v: viz_means[v])

        return self.paired_viz_comparison(filtered, best, worst)

    def glmm_summary(
        self,
        records: list[dict[str, object]],
    ) -> dict[str, object]:
        """Compute a simplified fixed-effects logistic regression summary.

        Uses a question-level aggregation approach as an approximation
        to mixed-effects logistic regression. For each visualization type,
        computes the odds ratio relative to text_only baseline.

        Args:
            records: Evaluation result rows.

        Returns:
            Dict with viz_type -> {odds_ratio, log_odds, n} mappings.
        """
        by_viz: dict[str, list[float]] = defaultdict(list)
        for r in records:
            viz = str(r.get("viz_type", ""))
            by_viz[viz].append(float(r.get("exact_match", 0.0)))

        baseline_key = "text_only"
        if baseline_key not in by_viz:
            return {"error": "no text_only baseline found"}

        baseline_ems = by_viz[baseline_key]
        baseline_p = sum(baseline_ems) / len(baseline_ems) if baseline_ems else 0.5
        baseline_p = max(0.01, min(0.99, baseline_p))
        baseline_log_odds = math.log(baseline_p / (1.0 - baseline_p))

        results: dict[str, object] = {}
        for viz, ems in sorted(by_viz.items()):
            p = sum(ems) / len(ems) if ems else 0.5
            p = max(0.01, min(0.99, p))
            log_odds = math.log(p / (1.0 - p))
            odds_ratio = math.exp(log_odds - baseline_log_odds)
            results[viz] = {
                "mean_em": p,
                "log_odds": log_odds,
                "odds_ratio_vs_text": odds_ratio,
                "n": len(ems),
            }
        return results
