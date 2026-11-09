"""Tests for baseline_analyzer module."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.baseline_analyzer import MajorityBaselineAnalyzer


class TestMajorityBaselineAnalyzer:
    """Tests for MajorityBaselineAnalyzer."""

    def setup_method(self) -> None:
        self.analyzer = MajorityBaselineAnalyzer()

    def test_compute_majority_baseline(self) -> None:
        items = [
            {"modality": "graph", "task": "connectivity", "answer": "yes"},
            {"modality": "graph", "task": "connectivity", "answer": "yes"},
            {"modality": "graph", "task": "connectivity", "answer": "yes"},
            {"modality": "graph", "task": "connectivity", "answer": "no"},
        ]
        baselines = self.analyzer.compute_majority_baseline(items)
        assert len(baselines) == 1
        assert baselines[0].majority_class == "yes"
        assert baselines[0].majority_ratio == 0.75
        assert baselines[0].item_count == 4

    def test_identify_trivial_tasks(self) -> None:
        items = [
            {"modality": "graph", "task": "connectivity", "answer": "yes"},
            {"modality": "graph", "task": "connectivity", "answer": "yes"},
            {"modality": "graph", "task": "connectivity", "answer": "yes"},
            {"modality": "graph", "task": "connectivity", "answer": "yes"},
            {"modality": "graph", "task": "connectivity", "answer": "yes"},
            {"modality": "graph", "task": "connectivity", "answer": "no"},
            {"modality": "tabular", "task": "value_extraction", "answer": "10"},
            {"modality": "tabular", "task": "value_extraction", "answer": "20"},
            {"modality": "tabular", "task": "value_extraction", "answer": "30"},
        ]
        # 5/6 = 0.833 > 0.80 threshold
        trivial = self.analyzer.identify_trivial_tasks(items, threshold=0.80)
        assert len(trivial) == 1
        assert trivial[0].task == "connectivity"

    def test_compute_adjusted_metrics(self) -> None:
        from src.evaluation.baseline_analyzer import TaskBaseline

        records = [
            {"modality": "graph", "task": "connectivity", "exact_match": 1.0, "f1": 1.0, "numeric_accuracy": 0.0},
            {"modality": "graph", "task": "connectivity", "exact_match": 1.0, "f1": 1.0, "numeric_accuracy": 0.0},
            {"modality": "tabular", "task": "value_extraction", "exact_match": 0.5, "f1": 0.6, "numeric_accuracy": 0.5},
            {"modality": "tabular", "task": "value_extraction", "exact_match": 0.3, "f1": 0.4, "numeric_accuracy": 0.3},
        ]
        trivial = [
            TaskBaseline(task="connectivity", modality="graph", majority_class="yes", majority_ratio=0.9, item_count=10),
        ]
        adj = self.analyzer.compute_adjusted_metrics(records, trivial)
        assert adj.total_items == 2
        assert adj.excluded_items == 2
        assert abs(adj.em - 0.4) < 1e-6

    def test_chance_adjusted_score(self) -> None:
        assert abs(MajorityBaselineAnalyzer.chance_adjusted_score(0.8, 0.5) - 0.6) < 1e-9
        assert MajorityBaselineAnalyzer.chance_adjusted_score(0.5, 0.5) == 0.0
        assert MajorityBaselineAnalyzer.chance_adjusted_score(0.3, 0.5) == 0.0
        assert abs(MajorityBaselineAnalyzer.chance_adjusted_score(1.0, 0.5) - 1.0) < 1e-9
        assert MajorityBaselineAnalyzer.chance_adjusted_score(0.5, 1.0) == 0.0
