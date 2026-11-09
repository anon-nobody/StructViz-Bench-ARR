from __future__ import annotations

from collections import defaultdict


class HallucinationRateTracker:
    """Track hallucination rate by visualization type."""

    def compute(self, records: list[dict[str, object]]) -> dict[str, float]:
        """Compute hallucination ratio per viz type from prediction records."""
        totals: dict[str, int] = defaultdict(int)
        hallucinations: dict[str, int] = defaultdict(int)
        for row in records:
            viz = str(row["viz_type"])
            totals[viz] += 1
            if bool(row.get("hallucinated", False)):
                hallucinations[viz] += 1
        return {
            viz: hallucinations[viz] / totals[viz] if totals[viz] else 0.0
            for viz in sorted(totals)
        }
