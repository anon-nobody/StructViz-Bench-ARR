"""DEPRECATED: Use scripts/run_fullscale_eval.py instead.

This script is a legacy entrypoint kept for reference only.
The canonical evaluation script is run_fullscale_eval.py.
"""

from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from src.evaluation.evaluator import Evaluator, EvaluationRecord
from src.models.model_factory import create_model
from src.utils.io_utils import read_benchmark_items


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for full multi-model evaluation."""
    parser = argparse.ArgumentParser(
        description="Run full StructViz-Bench multi-model evaluation."
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to evaluation YAML config."
    )
    return parser.parse_args()


def _mean_exact(records: list[EvaluationRecord]) -> float:
    if not records:
        return 0.0
    return sum(record.metrics.exact for record in records) / len(records)


def _build_summary_payload(
    all_records: list[EvaluationRecord],
    by_model: dict[str, list[EvaluationRecord]],
    evaluator: Evaluator,
) -> dict[str, Any]:
    """Build aggregate summary JSON for all evaluated models."""
    model_rows: list[dict[str, Any]] = []
    for model_name, records in sorted(by_model.items()):
        model_rows.append(
            {
                "model": model_name,
                "overall": evaluator.summarize(records),
                "by_modality": evaluator.summarize_by_modality(records),
                "by_difficulty": evaluator.summarize_by_difficulty(records),
                "sensitivity": evaluator.summarize_viz_sensitivity(records),
            }
        )

    return {
        "num_records": len(all_records),
        "overall_exact": _mean_exact(all_records),
        "models": model_rows,
    }


def main() -> None:
    """Evaluate all configured models and write predictions plus summary."""
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    outputs = cfg.get("outputs", {})
    batch_size = int(cfg.get("batch_size", 1))
    rendered_path = Path(
        cfg.get("inputs", {}).get(
            "rendered_items_jsonl", "benchmark/rendered_items.jsonl"
        )
    )

    rendered_items = [item.to_dict() for item in read_benchmark_items(rendered_path)]
    evaluator = Evaluator()

    all_records: list[EvaluationRecord] = []
    records_by_model: dict[str, list[EvaluationRecord]] = defaultdict(list)
    for model_config in cfg.get("models", []):
        model = create_model(Path(str(model_config)))
        records = evaluator.evaluate_batch(
            model=model, items=rendered_items, batch_size=batch_size
        )
        all_records.extend(records)
        records_by_model[model.name].extend(records)

    predictions_path = Path(
        outputs.get("predictions_jsonl", "results/predictions.jsonl")
    )
    evaluator.export_records(all_records, predictions_path)

    summary_payload = _build_summary_payload(all_records, records_by_model, evaluator)
    summary_path = Path(outputs.get("summary_json", "results/summary.json"))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"Evaluated models: {len(records_by_model)}")
    print(f"Total predictions: {len(all_records)}")
    print(f"Predictions JSONL: {predictions_path}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
