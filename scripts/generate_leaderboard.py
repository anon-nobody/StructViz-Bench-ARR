from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false

import argparse
from collections import defaultdict
from pathlib import Path

from src.evaluation.evaluator import EvaluationRecord, Evaluator
from src.evaluation.leaderboard import LeaderboardGenerator, LeaderboardRow


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for leaderboard generation."""
    parser = argparse.ArgumentParser(
        description="Generate leaderboard from real prediction JSONL files."
    )
    parser.add_argument(
        "--results", type=Path, required=True, help="Results directory."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional markdown output path. Defaults to <results>/leaderboard.md.",
    )
    return parser.parse_args()


def _to_records(rows: list[dict[str, object]]) -> list[EvaluationRecord]:
    return [Evaluator._record_from_row(row) for row in rows]


def _gather_prediction_rows(results_dir: Path) -> list[dict[str, object]]:
    files = sorted(results_dir.glob("*.jsonl"))
    rows: list[dict[str, object]] = []
    for file_path in files:
        rows.extend(Evaluator.load_records(file_path, as_rows=True))
    return rows


def main() -> None:
    """Read real predictions, compute metrics, and write leaderboard markdown."""
    args = parse_args()
    output_path = args.output or (args.results / "leaderboard.md")

    prediction_rows = _gather_prediction_rows(args.results)
    records = _to_records(prediction_rows)
    evaluator = Evaluator()

    by_model: dict[str, list[EvaluationRecord]] = defaultdict(list)
    for record in records:
        by_model[record.model].append(record)

    leaderboard_rows: list[LeaderboardRow] = []
    difficulty_scores: dict[str, dict[str, float]] = {}
    for model_name, model_records in sorted(by_model.items()):
        summary = evaluator.summarize(model_records)
        modality = evaluator.summarize_by_modality(model_records)
        difficulty = evaluator.summarize_by_difficulty(model_records)
        sensitivity = evaluator.summarize_viz_sensitivity(model_records)

        leaderboard_rows.append(
            LeaderboardRow(
                model=model_name,
                overall=summary["overall_exact"],
                tab_viz=modality.get("tabular", {}).get("exact", 0.0),
                ts_viz=modality.get("timeseries", {}).get("exact", 0.0),
                graph_viz=modality.get("graph", {}).get("exact", 0.0),
                mixed=modality.get("mixed", {}).get("exact", 0.0),
                sensitivity=sensitivity["avg_sensitivity"],
            )
        )
        difficulty_scores[model_name] = {
            key: value["exact"]
            for key, value in sorted(difficulty.items())
            if key in {"1-hop", "2-hop", "3-hop", "counterfactual"}
        }

    report = LeaderboardGenerator().generate_report(
        rows=sorted(leaderboard_rows, key=lambda row: row.overall, reverse=True),
        difficulty_scores=difficulty_scores,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Leaderboard markdown: {output_path}")


if __name__ == "__main__":
    main()
