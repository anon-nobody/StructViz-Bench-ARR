"""Tests for answer_extractor module."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.answer_extractor import extract_answer, recompute_with_extraction


class TestExtractAnswer:
    """Tests for the extract_answer function."""

    def test_short_prediction_returned_as_is(self) -> None:
        assert extract_answer("42") == "42"
        assert extract_answer("yes") == "yes"

    def test_explicit_answer_is_pattern(self) -> None:
        text = "Looking at the chart, I can see the values. The answer is 42."
        assert extract_answer(text) == "42"

    def test_answer_colon_pattern(self) -> None:
        text = "After careful analysis of all the data points...\nAnswer: 55.3"
        assert extract_answer(text) == "55.3"

    def test_therefore_pattern(self) -> None:
        text = (
            "The bar chart shows revenue of $100M in Q1 and $150M in Q2. "
            "The difference is $50M. Therefore, 50"
        )
        assert extract_answer(text) == "50"

    def test_thus_pattern(self) -> None:
        text = "We observe a clear upward trend across all quarters. Thus, increasing"
        assert extract_answer(text) == "increasing"

    def test_boxed_answer(self) -> None:
        text = "Let me work through this step by step... The result is \\boxed{3.14}"
        assert extract_answer(text) == "3.14"

    def test_bold_final_answer(self) -> None:
        text = "After analyzing the visualization carefully...\n**42**"
        assert extract_answer(text) == "42"

    def test_numeric_last_line(self) -> None:
        text = (
            "The chart shows several data points. Let me calculate the average.\n"
            "Sum = 100, Count = 4\n"
            "25.0"
        )
        assert extract_answer(text) == "25.0"

    def test_short_last_line(self) -> None:
        text = (
            "Based on the bar chart, I can see that the company with the highest "
            "revenue is clearly shown in the tallest bar.\n"
            "Company A"
        )
        assert extract_answer(text) == "Company A"

    def test_verbose_claude_response(self) -> None:
        text = (
            "Looking at the combined visualization, I need to analyze both the "
            "tabular data and the time series plot. The table shows quarterly "
            "revenues for three companies, while the line chart displays their "
            "stock price trends over the same period.\n\n"
            "From the table, Company B has the highest Q3 revenue at $85M.\n"
            "From the time series, Company B's stock also shows an upward trend.\n\n"
            "The answer is Company B"
        )
        assert extract_answer(text) == "Company B"

    def test_empty_input(self) -> None:
        assert extract_answer("") == ""
        assert extract_answer("   ") == ""

    def test_fallback_returns_original(self) -> None:
        # Long text without any extraction pattern should return stripped original.
        text = "A" * 100
        assert extract_answer(text) == text

    def test_result_pattern(self) -> None:
        text = "I calculated the values step by step.\nResult: 7.5"
        assert extract_answer(text) == "7.5"


class TestRecomputeWithExtraction:
    """Tests for recompute_with_extraction."""

    def test_basic_recomputation(self) -> None:
        rows = [
            {
                "prediction": "Looking at all the data carefully, the answer is 42.",
                "answer": "42",
                "exact_match": 0.0,
                "f1": 0.0,
                "numeric_accuracy": 0.0,
            },
        ]
        result = recompute_with_extraction(rows)
        assert result[0]["prediction_extracted"] == "42"
        assert result[0]["exact_match"] == 1.0

    def test_already_correct_unchanged(self) -> None:
        rows = [
            {
                "prediction": "yes",
                "answer": "yes",
                "exact_match": 1.0,
                "f1": 1.0,
                "numeric_accuracy": 0.0,
            },
        ]
        result = recompute_with_extraction(rows)
        assert result[0]["exact_match"] == 1.0

    def test_numeric_extraction(self) -> None:
        rows = [
            {
                "prediction": "After careful analysis, the value is approximately 3.14.\nAnswer: 3.14",
                "answer": "3.14",
                "exact_match": 0.0,
                "f1": 0.0,
                "numeric_accuracy": 0.0,
            },
        ]
        result = recompute_with_extraction(rows)
        assert result[0]["exact_match"] == 1.0
