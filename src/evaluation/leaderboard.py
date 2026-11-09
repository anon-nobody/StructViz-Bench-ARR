from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class LeaderboardRow:
    model: str
    overall: float
    tab_viz: float
    ts_viz: float
    graph_viz: float
    mixed: float
    sensitivity: float
    visual_only_em: float = 0.0
    text_gap_pp: float = 0.0
    adjusted_em: float = 0.0


class LeaderboardGenerator:
    """Generate markdown leaderboard tables for StructViz-Bench."""

    def generate_main_table(self, rows: list[LeaderboardRow]) -> str:
        """Generate main model-by-metric leaderboard markdown table."""
        header = (
            "| Model | Overall | Tab-Viz | TS-Viz | Graph-Viz | Mixed"
            " | Sensitivity | Vis-Only | Text Gap | Adj. EM |\n"
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
        body = "\n".join(
            f"| {row.model} | {row.overall:.3f} | {row.tab_viz:.3f} | {row.ts_viz:.3f} | "
            f"{row.graph_viz:.3f} | {row.mixed:.3f} | {row.sensitivity:.3f} | "
            f"{row.visual_only_em:.3f} | {row.text_gap_pp:.1f} | {row.adjusted_em:.3f} |"
            for row in rows
        )
        return f"{header}\n{body}" if body else header

    def generate_difficulty_table(
        self, difficulty_scores: dict[str, dict[str, float]]
    ) -> str:
        """Generate per-difficulty markdown breakdown for each model."""
        header = "| Model | 1-hop | 2-hop | 3-hop | Counterfactual |\n|---|---:|---:|---:|---:|"
        lines: list[str] = [header]
        for model, scores in sorted(difficulty_scores.items()):
            lines.append(
                "| "
                f"{model} | {scores.get('1-hop', 0.0):.3f} | {scores.get('2-hop', 0.0):.3f} | "
                f"{scores.get('3-hop', 0.0):.3f} | {scores.get('counterfactual', 0.0):.3f} |"
            )
        return "\n".join(lines)

    def generate_report(
        self,
        rows: list[LeaderboardRow],
        difficulty_scores: dict[str, dict[str, float]],
    ) -> str:
        """Generate complete leaderboard markdown report."""
        main = self.generate_main_table(rows)
        difficulty = self.generate_difficulty_table(difficulty_scores)
        return "\n\n".join(
            ["## StructViz-Bench Leaderboard", main, "## Per-Difficulty", difficulty]
        )
