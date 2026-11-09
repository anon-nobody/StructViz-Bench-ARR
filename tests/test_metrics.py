from __future__ import annotations

from src.evaluation.metrics import (
    compute_metrics,
    exact_match,
    numerical_accuracy,
    token_f1,
)
from src.evaluation.viz_sensitivity import VizSensitivityAnalyzer


def test_exact_match() -> None:
    assert exact_match("Yes", "yes") == 1.0


def test_token_f1_partial_overlap() -> None:
    assert token_f1("value is 10", "10") > 0.0


def test_numerical_accuracy() -> None:
    assert numerical_accuracy("1.0001", "1.0", abs_tolerance=0.05) == 1.0


def test_compute_metrics_bundle() -> None:
    metrics = compute_metrics("1", "1")
    assert metrics.exact == 1.0
    assert metrics.numeric == 1.0


def test_viz_sensitivity_analyzer() -> None:
    analyzer = VizSensitivityAnalyzer()
    result = analyzer.analyze({"bar_chart": 1.0, "heatmap": 0.0, "text_only": 1.0})
    assert result.best_viz in {"bar_chart", "text_only"}
    assert result.worst_viz == "heatmap"
    assert result.sensitivity_score > 0.0
