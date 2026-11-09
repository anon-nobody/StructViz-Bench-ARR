"""DEPRECATED: Use scripts/run_fullscale_eval.py instead.

This script is a legacy entrypoint kept for reference only.
The canonical evaluation script is run_fullscale_eval.py.
"""

from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false

import argparse
from pathlib import Path

import yaml

from src.evaluation.evaluator import Evaluator
from src.models.model_factory import create_model
from src.utils.io_utils import read_benchmark_items


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for single-model benchmark evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate one model on rendered benchmark items."
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Evaluation config YAML path."
    )
    parser.add_argument(
        "--model-config", type=Path, required=True, help="Single model YAML path."
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        required=True,
        help="Rendered BenchmarkItem JSONL path.",
    )
    return parser.parse_args()


def main() -> None:
    """Run one-model evaluation and export per-item prediction JSONL."""
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    model = create_model(args.model_config)
    evaluator = Evaluator()
    rendered_items = [item.to_dict() for item in read_benchmark_items(args.benchmark)]
    records = evaluator.evaluate_batch(
        model=model,
        items=rendered_items,
        batch_size=int(cfg.get("batch_size", 1)),
    )

    output_base = Path(
        cfg.get("outputs", {}).get("predictions_jsonl", "results/predictions.jsonl")
    )
    output_path = output_base.with_name(
        f"{output_base.stem}_{model.name}{output_base.suffix}"
    )
    evaluator.export_records(records, output_path)

    print(f"Model evaluated: {model.name}")
    print(f"Items evaluated: {len(records)}")
    print(f"Predictions JSONL: {output_path}")


if __name__ == "__main__":
    main()
