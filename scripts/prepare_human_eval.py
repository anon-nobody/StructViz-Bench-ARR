from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rendering.render_pipeline import RenderPipeline
from src.utils.io_utils import BenchmarkItem, read_benchmark_items, save_image

MODALITIES = ["tabular", "timeseries", "graph"]
DIFFICULTIES = ["1-hop", "2-hop", "3-hop", "counterfactual"]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for human evaluation preparation.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Prepare stratified sample and annotation materials for human evaluation.",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        required=True,
        help="Path to benchmark JSONL file (base_items.jsonl).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/human_eval"),
        help="Output directory for images, CSV, and instruction document.",
    )
    parser.add_argument(
        "--num-items",
        type=int,
        default=100,
        help="Number of items to sample (stratified by modality x difficulty).",
    )
    parser.add_argument(
        "--num-annotators",
        type=int,
        default=3,
        help="Number of annotators (controls number of CSV row copies).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def stratified_sample(
    items: list[BenchmarkItem],
    num_items: int,
    rng: random.Random,
) -> list[BenchmarkItem]:
    """Sample items stratified by (modality, difficulty).

    Distributes the budget evenly across all observed (modality, difficulty)
    buckets, then fills remaining slots from the leftover pool.

    Args:
        items: Full list of benchmark items.
        num_items: Target number of items to sample.
        rng: Seeded random generator.

    Returns:
        Stratified sample of benchmark items.
    """
    buckets: dict[tuple[str, str], list[BenchmarkItem]] = defaultdict(list)
    for item in items:
        buckets[(item.modality, item.difficulty)].append(item)

    per_bucket = max(1, num_items // len(buckets)) if buckets else 1
    sampled: list[BenchmarkItem] = []
    sampled_ids: set[str] = set()

    for key in sorted(buckets.keys()):
        pool = buckets[key]
        k = min(per_bucket, len(pool))
        chosen = rng.sample(pool, k)
        sampled.extend(chosen)
        sampled_ids.update(item.question_id for item in chosen)

    # Fill remaining slots from un-sampled items.
    remaining = num_items - len(sampled)
    if remaining > 0:
        leftover = [item for item in items if item.question_id not in sampled_ids]
        fill = rng.sample(leftover, min(remaining, len(leftover)))
        sampled.extend(fill)

    return sampled[:num_items]


def select_viz_types(
    item: BenchmarkItem,
    num_viz: int,
    rng: random.Random,
) -> list[str]:
    """Select random visualization types for an item.

    If the item has fewer applicable viz methods than requested, all are returned.

    Args:
        item: Benchmark item with viz_methods list.
        num_viz: Number of visualization types to select.
        rng: Seeded random generator.

    Returns:
        List of selected visualization type names.
    """
    available = list(item.viz_methods)
    if len(available) <= num_viz:
        return available
    return rng.sample(available, num_viz)


def render_items(
    items: list[BenchmarkItem],
    viz_selections: dict[str, list[str]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Render each item-visualization pair and save as PNG.

    Args:
        items: Sampled benchmark items.
        viz_selections: Mapping question_id -> list of selected viz types.
        output_dir: Root output directory for images.

    Returns:
        List of dicts with item_id, question, answer, viz_type, image_path,
        modality, and difficulty.
    """
    pipeline = RenderPipeline()
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for item in items:
        viz_types = viz_selections[item.question_id]
        rendered_all = pipeline.render_all(
            {"modality": item.modality, "data": item.data}
        )

        for viz_type in viz_types:
            if viz_type not in rendered_all:
                continue
            image = rendered_all[viz_type]
            image_filename = f"{item.question_id}_{viz_type}.png"
            image_path = image_dir / item.modality / image_filename
            save_image(image_path, image)

            records.append(
                {
                    "item_id": item.question_id,
                    "question": item.question,
                    "answer": item.answer,
                    "viz_type": viz_type,
                    "image_path": str(image_path),
                    "modality": item.modality,
                    "difficulty": item.difficulty,
                }
            )

    return records


def generate_annotation_csv(
    records: list[dict[str, Any]],
    num_annotators: int,
    output_dir: Path,
) -> Path:
    """Generate an annotation spreadsheet for human evaluators.

    Creates one row per (item, viz_type, annotator) combination.  The
    ``human_answer``, ``human_confidence``, and ``time_seconds`` columns
    are left blank for annotators to fill in.

    Args:
        records: Rendered item records from ``render_items``.
        num_annotators: Number of annotator copies per item-viz pair.
        output_dir: Output directory for the CSV file.

    Returns:
        Path to the generated CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "annotation_sheet.csv"

    fieldnames = [
        "item_id",
        "question",
        "viz_type",
        "image_path",
        "human_answer",
        "human_confidence",
        "time_seconds",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            for _annotator_idx in range(num_annotators):
                writer.writerow(
                    {
                        "item_id": record["item_id"],
                        "question": record["question"],
                        "viz_type": record["viz_type"],
                        "image_path": record["image_path"],
                        "human_answer": "",
                        "human_confidence": "",
                        "time_seconds": "",
                    }
                )

    return csv_path


def generate_instruction_document(output_dir: Path, num_items: int) -> Path:
    """Generate an HTML instruction document for annotators.

    Describes the evaluation task, rating scales, and workflow. The HTML
    format allows easy printing or browser-based reading.

    Args:
        output_dir: Output directory for the instruction file.
        num_items: Number of items in the evaluation set.

    Returns:
        Path to the generated HTML file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "annotator_instructions.html"

    html_content = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>StructViz-Bench Human Evaluation Instructions</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto;
         line-height: 1.6; color: #333; }}
  h1 {{ color: #1a5276; border-bottom: 2px solid #1a5276; padding-bottom: 8px; }}
  h2 {{ color: #2c3e50; margin-top: 32px; }}
  .highlight {{ background: #fef9e7; padding: 12px; border-left: 4px solid #f39c12;
                margin: 16px 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th, td {{ border: 1px solid #bdc3c7; padding: 8px 12px; text-align: left; }}
  th {{ background: #ecf0f1; }}
  code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
  ol li {{ margin-bottom: 8px; }}
</style>
</head>
<body>

<h1>StructViz-Bench Human Evaluation</h1>

<h2>1. Overview</h2>
<p>You will evaluate <strong>{num_items} benchmark items</strong>, each rendered as
a data visualization image. Your task is to answer a question about the
visualized data as accurately as possible.</p>

<h2>2. Materials</h2>
<ul>
  <li><strong>Images:</strong> Located in the <code>images/</code> subdirectory,
      organized by data modality (tabular, timeseries, graph).</li>
  <li><strong>Annotation sheet:</strong> <code>annotation_sheet.csv</code> with one
      row per (item, visualization type) pair.</li>
</ul>

<h2>3. Annotation Workflow</h2>
<ol>
  <li>Open the annotation CSV in your preferred spreadsheet editor.</li>
  <li>For each row, locate the image referenced in <code>image_path</code>.</li>
  <li>Read the <code>question</code> and study the visualization carefully.</li>
  <li>Write your answer in the <code>human_answer</code> column.</li>
  <li>Rate your confidence (1-5) in the <code>human_confidence</code> column.</li>
  <li>Record the time you spent (in seconds) in <code>time_seconds</code>.</li>
</ol>

<h2>4. Confidence Scale</h2>
<table>
  <tr><th>Score</th><th>Meaning</th></tr>
  <tr><td>1</td><td>Very uncertain -- guessing</td></tr>
  <tr><td>2</td><td>Somewhat uncertain</td></tr>
  <tr><td>3</td><td>Moderately confident</td></tr>
  <tr><td>4</td><td>Fairly confident</td></tr>
  <tr><td>5</td><td>Very confident -- certain of the answer</td></tr>
</table>

<h2>5. Answer Format Guidelines</h2>
<div class="highlight">
  <p><strong>Important:</strong> Provide concise, exact answers. For numerical
  questions, give the number only (e.g., <code>42</code> or <code>3.14</code>).
  For categorical questions, use the exact label shown in the visualization.
  Do not include units unless the question specifically asks for them.</p>
</div>

<h2>6. Data Modalities</h2>
<table>
  <tr><th>Modality</th><th>Description</th></tr>
  <tr><td>Tabular</td><td>Tables rendered as heatmaps, bar charts, HTML tables, etc.</td></tr>
  <tr><td>Time series</td><td>Temporal data shown as line plots, area charts, etc.</td></tr>
  <tr><td>Graph</td><td>Network/graph structures displayed as node-link diagrams, etc.</td></tr>
</table>

<h2>7. Tips</h2>
<ul>
  <li>Take breaks to avoid fatigue -- annotation quality matters more than speed.</li>
  <li>If the visualization is genuinely unreadable, write <code>UNREADABLE</code>
      and set confidence to 1.</li>
  <li>Do not consult external resources or other annotators during evaluation.</li>
</ul>

</body>
</html>"""

    html_path.write_text(html_content, encoding="utf-8")
    return html_path


def main() -> None:
    """Prepare human evaluation materials from a benchmark JSONL file.

    Performs stratified sampling, renders visualization images, generates
    an annotation CSV, and writes annotator instructions.
    """
    args = parse_args()
    rng = random.Random(args.seed)

    project_root = Path(__file__).resolve().parents[1]
    benchmark_path = (
        args.benchmark if args.benchmark.is_absolute() else project_root / args.benchmark
    )
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    )

    print(f"Loading benchmark from {benchmark_path} ...")
    items = read_benchmark_items(benchmark_path)
    print(f"  Loaded {len(items)} items.")

    # Stratified sample.
    sampled = stratified_sample(items, args.num_items, rng)
    print(f"  Sampled {len(sampled)} items (stratified by modality x difficulty).")

    # Select 3 random viz types per item.
    num_viz_per_item = 3
    viz_selections: dict[str, list[str]] = {}
    for item in sampled:
        viz_selections[item.question_id] = select_viz_types(item, num_viz_per_item, rng)

    # Render images.
    print("Rendering images ...")
    records = render_items(sampled, viz_selections, output_dir)
    print(f"  Rendered {len(records)} item-visualization pairs.")

    # Generate annotation CSV.
    csv_path = generate_annotation_csv(records, args.num_annotators, output_dir)
    print(f"  Annotation CSV: {csv_path}")

    # Generate instruction document.
    instructions_path = generate_instruction_document(output_dir, args.num_items)
    print(f"  Annotator instructions: {instructions_path}")

    # Summary manifest.
    manifest = {
        "seed": args.seed,
        "num_items": len(sampled),
        "num_viz_per_item": num_viz_per_item,
        "num_annotators": args.num_annotators,
        "total_annotation_rows": len(records) * args.num_annotators,
        "total_images": len(records),
    }
    print("\nManifest:")
    for key, value in manifest.items():
        print(f"  {key}: {value}")

    print(f"\nHuman evaluation materials ready in {output_dir}")


if __name__ == "__main__":
    main()
