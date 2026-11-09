from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rendering.render_pipeline import RenderPipeline
from src.utils.io_utils import (
    BenchmarkItem,
    read_benchmark_items,
    save_image,
    write_benchmark_items,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for benchmark rendering."""
    parser = argparse.ArgumentParser(
        description="Render all benchmark items for all modality viz methods."
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="Input BenchmarkItem JSONL path."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for images/JSONL.",
    )
    return parser.parse_args()


def main() -> None:
    """Render all visualization variants and save rendered benchmark JSONL."""
    args = parse_args()
    pipeline = RenderPipeline()
    base_items = read_benchmark_items(args.input)

    rendered_items: list[BenchmarkItem] = []
    for item in base_items:
        rendered = pipeline.render_all({"modality": item.modality, "data": item.data})
        for viz_name, image in rendered.items():
            image_path = (
                args.output_dir
                / "benchmark"
                / "rendered"
                / item.modality
                / f"{item.question_id}_{viz_name}.png"
            )
            save_image(image_path, image)
            rendered_items.append(
                BenchmarkItem(
                    question_id=item.question_id,
                    question=item.question,
                    answer=item.answer,
                    modality=item.modality,
                    data_id=item.data_id,
                    task=item.task,
                    difficulty=item.difficulty,
                    viz_methods=list(item.viz_methods),
                    data=item.data,
                    metadata=dict(item.metadata),
                    image_path=str(image_path),
                    viz_type=viz_name,
                    source=item.source,
                )
            )

    output_jsonl = args.output_dir / "rendered_items.jsonl"
    write_benchmark_items(output_jsonl, rendered_items)
    print(f"Rendered items: {len(rendered_items)}")
    print(f"Rendered benchmark JSONL: {output_jsonl}")


if __name__ == "__main__":
    main()
