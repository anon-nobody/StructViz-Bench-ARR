"""Majority-baseline analysis and trivial-task filtering.

the review venue reviewers will note that some task types have majority-class baselines
above 80%, inflating aggregate EM.  This module identifies such trivial tasks,
computes chance-adjusted scores, and produces filtered metrics.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass(slots=True)
class TaskBaseline:
    """Majority-class baseline statistics for one task type."""

    task: str
    modality: str
    majority_class: str
    majority_ratio: float
    item_count: int


@dataclass(slots=True)
class AdjustedMetrics:
    """Metrics after filtering out trivial tasks."""

    em: float
    f1: float
    numeric: float
    total_items: int
    excluded_tasks: list[str]
    excluded_items: int


class MajorityBaselineAnalyzer:
    """Detect trivial tasks and compute chance-adjusted metrics."""

    def compute_majority_baseline(
        self,
        items: list[dict[str, object]],
    ) -> list[TaskBaseline]:
        """Compute majority-class ratio per (modality, task) group.

        Args:
            items: Benchmark items with ``task``, ``modality``, ``answer`` fields.

        Returns:
            Sorted list of TaskBaseline results.
        """
        grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
        for item in items:
            key = (str(item.get("modality", "unknown")), str(item.get("task", "unknown")))
            grouped[key].append(str(item.get("answer", "")))

        results: list[TaskBaseline] = []
        for (modality, task), answers in sorted(grouped.items()):
            counter = Counter(answers)
            most_common_answer, most_common_count = counter.most_common(1)[0]
            results.append(
                TaskBaseline(
                    task=task,
                    modality=modality,
                    majority_class=most_common_answer,
                    majority_ratio=most_common_count / len(answers),
                    item_count=len(answers),
                )
            )
        return results

    def identify_trivial_tasks(
        self,
        items: list[dict[str, object]],
        threshold: float = 0.80,
    ) -> list[TaskBaseline]:
        """Return task types whose majority baseline exceeds *threshold*.

        Args:
            items: Benchmark items.
            threshold: Majority ratio cutoff (default 0.80).

        Returns:
            List of trivial TaskBaseline entries.
        """
        baselines = self.compute_majority_baseline(items)
        return [b for b in baselines if b.majority_ratio > threshold]

    def compute_adjusted_metrics(
        self,
        records: list[dict[str, object]],
        trivial_tasks: list[TaskBaseline],
    ) -> AdjustedMetrics:
        """Recompute aggregate metrics after removing trivial-task records.

        Args:
            records: Evaluation result rows with ``task``, ``modality``,
                ``exact_match``, ``f1``, ``numeric_accuracy`` fields.
            trivial_tasks: Trivial tasks to exclude.

        Returns:
            AdjustedMetrics with filtered aggregate scores.
        """
        exclude_keys = {(t.modality, t.task) for t in trivial_tasks}
        filtered = [
            r
            for r in records
            if (str(r.get("modality", "")), str(r.get("task", ""))) not in exclude_keys
        ]
        excluded_count = len(records) - len(filtered)

        if not filtered:
            return AdjustedMetrics(
                em=0.0,
                f1=0.0,
                numeric=0.0,
                total_items=0,
                excluded_tasks=[f"{t.modality}:{t.task}" for t in trivial_tasks],
                excluded_items=excluded_count,
            )

        n = len(filtered)
        return AdjustedMetrics(
            em=sum(float(r.get("exact_match", 0.0)) for r in filtered) / n,
            f1=sum(float(r.get("f1", 0.0)) for r in filtered) / n,
            numeric=sum(float(r.get("numeric_accuracy", 0.0)) for r in filtered) / n,
            total_items=n,
            excluded_tasks=[f"{t.modality}:{t.task}" for t in trivial_tasks],
            excluded_items=excluded_count,
        )

    @staticmethod
    def chance_adjusted_score(em: float, baseline: float) -> float:
        """Normalize EM by chance level: (EM - baseline) / (1 - baseline).

        Returns 0.0 when EM ≤ baseline (no better than random), and 1.0 for
        perfect performance.  Avoids division by zero when baseline = 1.0.

        Args:
            em: Exact-match accuracy in [0, 1].
            baseline: Majority-class baseline in [0, 1].

        Returns:
            Chance-adjusted score.
        """
        if baseline >= 1.0:
            return 0.0
        adjusted = (em - baseline) / (1.0 - baseline)
        return max(0.0, adjusted)
