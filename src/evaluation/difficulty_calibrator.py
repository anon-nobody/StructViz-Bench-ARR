"""Empirical difficulty calibration via IRT and discrimination analysis.

Validates programmatic difficulty labels (1-hop, 2-hop, 3-hop, counterfactual)
against actual model performance.  Provides:
- Empirical difficulty bins based on observed EM
- Item discrimination index (upper 27% vs lower 27%)
- Spearman rank correlation between assigned and empirical difficulty
- 1-Parameter IRT (Rasch model) difficulty estimation
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass


# ── Difficulty label → numeric rank for correlation ──────────────────────────

_DIFFICULTY_RANK: dict[str, int] = {
    "1-hop": 1,
    "2-hop": 2,
    "3-hop": 3,
    "counterfactual": 4,
}


@dataclass(slots=True)
class EmpiricalDifficulty:
    """Empirical difficulty for one question.

    Attributes:
        question_id: Unique question identifier.
        assigned: Programmatic difficulty label.
        assigned_rank: Numeric rank of assigned difficulty.
        mean_em: Mean EM across all models and viz types.
        empirical_label: Empirical difficulty based on EM thresholds.
        discrimination: Discrimination index (upper - lower 27% EM).
    """

    question_id: str
    assigned: str
    assigned_rank: int
    mean_em: float
    empirical_label: str
    discrimination: float


@dataclass(slots=True)
class CalibrationReport:
    """Full difficulty calibration report.

    Attributes:
        items: Per-question empirical difficulty.
        spearman_rho: Rank correlation between assigned-difficulty rank (1-4)
            and binned empirical-difficulty rank (1-3, easy/medium/hard).
            This is the *binned* method which can suffer information loss.
        spearman_rho_raw: Rank correlation between assigned-difficulty rank (1-4)
            and per-question raw mean EM treated as a continuous variable.
            This is the standard, recommended method.
        alignment_rate: Fraction of items where assigned ≈ empirical.
        mean_discrimination: Average discrimination index.
        per_level_em: Mean EM per assigned difficulty level.
        irt_difficulties: 1PL IRT difficulty per question (if computed).
    """

    items: list[EmpiricalDifficulty]
    spearman_rho: float
    spearman_rho_raw: float
    alignment_rate: float
    mean_discrimination: float
    per_level_em: dict[str, float]
    irt_difficulties: dict[str, float]


class DifficultyCalibrator:
    """Validate and calibrate benchmark difficulty labels."""

    def __init__(
        self,
        easy_threshold: float = 0.60,
        hard_threshold: float = 0.30,
    ) -> None:
        """Initialize with empirical difficulty thresholds.

        Args:
            easy_threshold: EM above this → "easy".
            hard_threshold: EM below this → "hard".
        """
        self._easy = easy_threshold
        self._hard = hard_threshold

    def _empirical_label(self, mean_em: float) -> str:
        if mean_em >= self._easy:
            return "easy"
        if mean_em <= self._hard:
            return "hard"
        return "medium"

    def _empirical_rank(self, label: str) -> int:
        return {"easy": 1, "medium": 2, "hard": 3}.get(label, 2)

    def compute_empirical_difficulty(
        self,
        records: list[dict[str, object]],
    ) -> list[EmpiricalDifficulty]:
        """Compute empirical difficulty for each question.

        Args:
            records: Evaluation result rows with ``question_id``,
                ``difficulty``, ``exact_match`` fields.

        Returns:
            Per-question empirical difficulty results.
        """
        # Group by question_id.
        by_question: dict[str, list[dict[str, object]]] = defaultdict(list)
        for r in records:
            qid = str(r.get("question_id", ""))
            by_question[qid].append(r)

        results: list[EmpiricalDifficulty] = []
        for qid, rows in sorted(by_question.items()):
            ems = [float(r.get("exact_match", 0.0)) for r in rows]
            mean_em = sum(ems) / len(ems) if ems else 0.0

            assigned = str(rows[0].get("difficulty", "unknown"))
            assigned_rank = _DIFFICULTY_RANK.get(assigned, 0)

            empirical_label = self._empirical_label(mean_em)

            # Discrimination: sort by EM, compare top 27% vs bottom 27%.
            sorted_ems = sorted(ems)
            n = len(sorted_ems)
            k = max(1, int(n * 0.27))
            lower = sum(sorted_ems[:k]) / k
            upper = sum(sorted_ems[-k:]) / k
            discrimination = upper - lower

            results.append(
                EmpiricalDifficulty(
                    question_id=qid,
                    assigned=assigned,
                    assigned_rank=assigned_rank,
                    mean_em=mean_em,
                    empirical_label=empirical_label,
                    discrimination=discrimination,
                )
            )
        return results

    def validate_difficulty_alignment(
        self,
        records: list[dict[str, object]],
    ) -> CalibrationReport:
        """Full calibration report comparing assigned vs empirical difficulty.

        Args:
            records: Evaluation result rows.

        Returns:
            CalibrationReport with Spearman ρ, alignment rate, and per-level EM.
        """
        items = self.compute_empirical_difficulty(records)
        if not items:
            return CalibrationReport(
                items=[],
                spearman_rho=0.0,
                spearman_rho_raw=0.0,
                alignment_rate=0.0,
                mean_discrimination=0.0,
                per_level_em={},
                irt_difficulties={},
            )

        # Method 1 (binned): assigned rank (1-4) vs binned empirical rank (1-3).
        # Information-lossy due to discretization; reported for backward
        # compatibility but should not be the headline correlation.
        assigned_ranks = [float(i.assigned_rank) for i in items if i.assigned_rank > 0]
        empirical_ranks = [
            float(self._empirical_rank(i.empirical_label))
            for i in items
            if i.assigned_rank > 0
        ]
        rho = self._spearman(assigned_ranks, empirical_ranks)

        # Method 2 (standard): assigned rank vs raw mean EM treated as a
        # continuous score. This is the recommended Spearman computation
        # and avoids the information loss from binning.
        raw_em = [i.mean_em for i in items if i.assigned_rank > 0]
        rho_raw = self._spearman(assigned_ranks, raw_em)

        # Alignment: assigned difficulty direction matches empirical.
        aligned = 0
        total_valid = 0
        for item in items:
            if item.assigned_rank == 0:
                continue
            total_valid += 1
            assigned_hard = item.assigned_rank >= 3
            empirical_hard = item.empirical_label == "hard"
            assigned_easy = item.assigned_rank <= 1
            empirical_easy = item.empirical_label == "easy"
            if (assigned_hard and empirical_hard) or (assigned_easy and empirical_easy):
                aligned += 1
            elif item.empirical_label == "medium":
                aligned += 1  # medium is always "aligned enough"
        alignment_rate = aligned / total_valid if total_valid > 0 else 0.0

        # Mean discrimination.
        mean_disc = (
            sum(i.discrimination for i in items) / len(items) if items else 0.0
        )

        # Per-level EM.
        level_ems: dict[str, list[float]] = defaultdict(list)
        for item in items:
            level_ems[item.assigned].append(item.mean_em)
        per_level_em = {
            level: sum(vals) / len(vals) for level, vals in level_ems.items() if vals
        }

        # 1PL IRT difficulty estimation.
        irt_diffs = self._compute_irt_difficulties(items)

        return CalibrationReport(
            items=items,
            spearman_rho=rho,
            spearman_rho_raw=rho_raw,
            alignment_rate=alignment_rate,
            mean_discrimination=mean_disc,
            per_level_em=per_level_em,
            irt_difficulties=irt_diffs,
        )

    @staticmethod
    def _spearman(x: list[float], y: list[float]) -> float:
        """Compute Spearman rank correlation coefficient.

        Args:
            x: First variable values.
            y: Second variable values (same length).

        Returns:
            Spearman ρ in [-1, 1], or 0.0 if insufficient data.
        """
        n = len(x)
        if n < 3 or len(y) != n:
            return 0.0

        def _rank(values: list[float]) -> list[float]:
            indexed = sorted(enumerate(values), key=lambda t: t[1])
            ranks = [0.0] * n
            i = 0
            while i < n:
                j = i
                while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
                    j += 1
                avg_rank = (i + j) / 2.0 + 1.0
                for k in range(i, j + 1):
                    ranks[indexed[k][0]] = avg_rank
                i = j + 1
            return ranks

        rx = _rank(x)
        ry = _rank(y)
        d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry))
        return 1.0 - (6.0 * d_sq) / (n * (n * n - 1))

    @staticmethod
    def _compute_irt_difficulties(
        items: list[EmpiricalDifficulty],
    ) -> dict[str, float]:
        """Estimate 1PL (Rasch) item difficulty from mean EM.

        Uses the logit transform: difficulty = -log(p / (1-p)) where p = mean_em.
        Positive values = harder items, negative = easier.

        Args:
            items: Per-question empirical difficulty results.

        Returns:
            Mapping question_id -> IRT difficulty parameter.
        """
        result: dict[str, float] = {}
        for item in items:
            p = item.mean_em
            # Clamp to avoid log(0) or log(inf).
            p = max(0.01, min(0.99, p))
            result[item.question_id] = -math.log(p / (1.0 - p))
        return result
