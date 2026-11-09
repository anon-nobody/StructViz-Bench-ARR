"""Recompute metrics on existing result files using answer extraction.

Reads an input JSONL file of model predictions, applies the answer
extraction pipeline to each prediction, recomputes exact_match / f1 /
numeric_accuracy, and writes the updated records to an output file.
Also prints a before/after comparison summary to stdout.

Usage:
    python scripts/recompute_with_extraction.py \
        --input results/full_claude.jsonl \
        --output results/full_claude_extracted.jsonl
"""

from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false

import argparse
import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pathlib import Path
from typing import Any

from src.evaluation.answer_extractor import extract_answer
from src.evaluation.metrics import compute_metrics
from src.utils.io_utils import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Recompute metrics with answer extraction on existing JSONL results."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input JSONL results file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write updated JSONL results.",
    )
    parser.add_argument(
        "--abs-tolerance",
        type=float,
        default=0.05,
        help="Absolute numeric tolerance (default: 0.05).",
    )
    parser.add_argument(
        "--rel-tolerance",
        type=float,
        default=0.01,
        help="Relative numeric tolerance (default: 0.01).",
    )
    return parser.parse_args()


def recompute_record(
    record: dict[str, Any],
    abs_tolerance: float = 0.05,
    rel_tolerance: float = 0.01,
) -> dict[str, Any]:
    """Recompute metrics for a single record using answer extraction.

    Args:
        record: Original evaluation record dict with ``prediction`` and
            ``answer`` fields.
        abs_tolerance: Absolute numeric tolerance.
        rel_tolerance: Relative numeric tolerance.

    Returns:
        Updated record dict with extracted prediction and recomputed
        ``exact``, ``f1``, and ``numeric`` metric fields.  The original
        raw prediction is preserved as ``prediction_raw``.
    """
    prediction_raw = str(record.get("prediction", ""))
    answer = str(record.get("answer", ""))

    extracted = extract_answer(prediction_raw)
    metrics = compute_metrics(
        extracted,
        answer,
        abs_tolerance=abs_tolerance,
        rel_tolerance=rel_tolerance,
        extract=False,  # Already extracted above.
    )

    updated = dict(record)
    updated["prediction_raw"] = prediction_raw
    updated["prediction"] = extracted
    updated["exact"] = metrics.exact
    updated["f1"] = metrics.f1
    updated["numeric"] = metrics.numeric
    return updated


def _compute_mean(records: list[dict[str, Any]], key: str) -> float:
    """Compute mean of a numeric field across records."""
    values = [float(r.get(key, 0.0)) for r in records]
    if not values:
        return 0.0
    return sum(values) / len(values)


def print_comparison(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> None:
    """Print a before/after comparison summary to stdout.

    Args:
        before: Original records.
        after: Records with recomputed metrics.
    """
    n = len(before)
    print(f"\n{'=' * 60}")
    print(f"Recomputation Summary  ({n} records)")
    print(f"{'=' * 60}")

    for metric in ("exact", "f1", "numeric"):
        old_key = metric
        # Try alternate key names used in some result files.
        if metric == "exact" and not any(metric in r for r in before):
            old_key = "exact_match"
        before_mean = _compute_mean(before, old_key)
        after_mean = _compute_mean(after, metric)
        delta = after_mean - before_mean
        arrow = "+" if delta >= 0 else ""
        print(f"  {metric:>10s}:  {before_mean:.4f}  ->  {after_mean:.4f}  ({arrow}{delta:.4f})")

    # Count flipped predictions (wrong -> correct)
    flipped_correct = 0
    flipped_wrong = 0
    for old, new in zip(before, after):
        old_em = float(old.get("exact", old.get("exact_match", 0.0)))
        new_em = float(new.get("exact", 0.0))
        if old_em == 0.0 and new_em == 1.0:
            flipped_correct += 1
        elif old_em == 1.0 and new_em == 0.0:
            flipped_wrong += 1

    print(f"\n  Flipped correct  (0->1): {flipped_correct}")
    print(f"  Flipped wrong    (1->0): {flipped_wrong}")
    print(f"{'=' * 60}\n")


def main() -> None:
    """Entry point: read, recompute, write, and print summary."""
    args = parse_args()

    input_path: Path = args.input
    output_path: Path = args.output

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    records = read_jsonl(input_path)
    if not records:
        print(f"Warning: no records found in {input_path}", file=sys.stderr)
        sys.exit(0)

    print(f"Loaded {len(records)} records from {input_path}")
    print("Applying answer extraction and recomputing metrics...")

    updated: list[dict[str, Any]] = [
        recompute_record(
            r,
            abs_tolerance=args.abs_tolerance,
            rel_tolerance=args.rel_tolerance,
        )
        for r in records
    ]

    write_jsonl(output_path, updated)
    print(f"Wrote {len(updated)} updated records to {output_path}")

    print_comparison(records, updated)


if __name__ == "__main__":
    main()
