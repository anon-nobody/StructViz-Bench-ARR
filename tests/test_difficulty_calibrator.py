"""Tests for difficulty_calibrator module."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.difficulty_calibrator import DifficultyCalibrator


class TestDifficultyCalibrator:
    """Tests for DifficultyCalibrator."""

    def setup_method(self) -> None:
        self.calibrator = DifficultyCalibrator()

    def _make_records(
        self,
        qid: str,
        difficulty: str,
        ems: list[float],
    ) -> list[dict[str, object]]:
        return [
            {"question_id": qid, "difficulty": difficulty, "exact_match": em}
            for em in ems
        ]

    def test_empirical_difficulty_easy(self) -> None:
        records = self._make_records("q1", "1-hop", [1.0, 1.0, 0.8, 0.7])
        items = self.calibrator.compute_empirical_difficulty(records)
        assert len(items) == 1
        assert items[0].empirical_label == "easy"
        assert items[0].mean_em > 0.6

    def test_empirical_difficulty_hard(self) -> None:
        records = self._make_records("q2", "3-hop", [0.0, 0.1, 0.2, 0.1])
        items = self.calibrator.compute_empirical_difficulty(records)
        assert len(items) == 1
        assert items[0].empirical_label == "hard"
        assert items[0].mean_em < 0.3

    def test_empirical_difficulty_medium(self) -> None:
        records = self._make_records("q3", "2-hop", [0.5, 0.4, 0.3, 0.5])
        items = self.calibrator.compute_empirical_difficulty(records)
        assert len(items) == 1
        assert items[0].empirical_label == "medium"

    def test_discrimination_index(self) -> None:
        records = self._make_records("q1", "2-hop", [0.0, 0.0, 1.0, 1.0])
        items = self.calibrator.compute_empirical_difficulty(records)
        # Top 27% all correct, bottom 27% all wrong → discrimination = 1.0
        assert items[0].discrimination == 1.0

    def test_validate_alignment(self) -> None:
        records = (
            self._make_records("q1", "1-hop", [0.9, 0.8, 1.0, 0.7])
            + self._make_records("q2", "2-hop", [0.5, 0.4, 0.5, 0.4])
            + self._make_records("q3", "3-hop", [0.1, 0.2, 0.0, 0.1])
        )
        report = self.calibrator.validate_difficulty_alignment(records)
        # 1-hop → easy, 2-hop → medium, 3-hop → hard: positive correlation
        assert report.spearman_rho > 0.0
        assert report.alignment_rate > 0.0
        assert len(report.per_level_em) == 3
        assert len(report.irt_difficulties) == 3

    def test_irt_harder_items_have_higher_difficulty(self) -> None:
        records = (
            self._make_records("easy_q", "1-hop", [0.9, 0.9, 0.9])
            + self._make_records("hard_q", "3-hop", [0.1, 0.1, 0.1])
        )
        report = self.calibrator.validate_difficulty_alignment(records)
        # IRT difficulty: higher value = harder.
        assert report.irt_difficulties["hard_q"] > report.irt_difficulties["easy_q"]

    def test_spearman_perfect_correlation(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        rho = DifficultyCalibrator._spearman(x, y)
        assert abs(rho - 1.0) < 1e-6

    def test_spearman_inverse_correlation(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [5.0, 4.0, 3.0, 2.0, 1.0]
        rho = DifficultyCalibrator._spearman(x, y)
        assert abs(rho - (-1.0)) < 1e-6

    def test_empty_records(self) -> None:
        report = self.calibrator.validate_difficulty_alignment([])
        assert report.spearman_rho == 0.0
        assert report.alignment_rate == 0.0
