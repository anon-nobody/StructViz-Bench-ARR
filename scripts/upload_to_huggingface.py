from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, Features, Image, Value
from huggingface_hub import HfApi


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare StructViz-Bench dataset and upload it to HuggingFace Hub."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help='Target HuggingFace dataset repo id, e.g. "anonymous/StructViz-Bench".',
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("benchmark"),
        help="Path to benchmark directory containing JSONL files and rendered images.",
    )
    parser.add_argument(
        "--dataset-card",
        type=Path,
        default=Path("DATASET_CARD.md"),
        help="Path to dataset card markdown file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare release artifacts locally without uploading to HuggingFace.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token. If omitted, HF_TOKEN environment variable is used.",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _collect_image_index(rendered_root: Path) -> dict[tuple[str, str], Path]:
    index: dict[tuple[str, str], Path] = {}
    for path in rendered_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        stem = path.stem
        if "_" not in stem:
            continue
        question_id, viz_type = stem.rsplit("_", 1)
        index[(question_id, viz_type)] = path
    return index


def _load_merged_items(base_path: Path, realworld_path: Path) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}

    for source_name, default_source, path in [
        ("base_items", "synthetic", base_path),
        ("realworld_test", "real_world", realworld_path),
    ]:
        for item in _load_jsonl(path):
            question_id = str(item["question_id"])
            if question_id in merged:
                continue
            normalized = dict(item)
            normalized["source"] = str(item.get("source", default_source))
            normalized["_source_file"] = source_name
            merged[question_id] = normalized

    return list(merged.values())


def _prepare_rows(
    items: list[dict[str, Any]],
    image_index: dict[tuple[str, str], Path],
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    missing_images = 0

    for item in items:
        question_id = str(item["question_id"])
        viz_methods = item.get("viz_methods", [])
        if not isinstance(viz_methods, list):
            continue

        for viz_type_raw in viz_methods:
            viz_type = str(viz_type_raw)
            image_path = image_index.get((question_id, viz_type))
            if image_path is None:
                missing_images += 1
                continue

            rows.append(
                {
                    "question_id": question_id,
                    "question": str(item.get("question", "")),
                    "answer": str(item.get("answer", "")),
                    "modality": str(item.get("modality", "")),
                    "source": str(item.get("source", "unknown")),
                    "viz_type": viz_type,
                    "difficulty": str(item.get("difficulty", "")),
                    "task_type": str(item.get("task", item.get("task_type", ""))),
                    "image": {
                        "path": None,
                        "bytes": image_path.read_bytes(),
                    },
                }
            )

    return rows, missing_images


def _build_dataset(rows: list[dict[str, Any]], release_dir: Path) -> DatasetDict:
    features = Features(
        {
            "question_id": Value("string"),
            "question": Value("string"),
            "answer": Value("string"),
            "modality": Value("string"),
            "source": Value("string"),
            "viz_type": Value("string"),
            "difficulty": Value("string"),
            "task_type": Value("string"),
            "image": Image(),
        }
    )

    test_ds = Dataset.from_list(rows, features=features)
    dataset_dict = DatasetDict({"test": test_ds})
    dataset_dict.save_to_disk(str(release_dir / "hf_dataset"))
    return dataset_dict


def _print_summary(
    repo_id: str,
    release_dir: Path,
    rows: list[dict[str, Any]],
    unique_items: int,
    missing_images: int,
    dry_run: bool,
) -> None:
    modalities = sorted({row["modality"] for row in rows})
    viz_types = sorted({row["viz_type"] for row in rows})

    print("StructViz-Bench HuggingFace release summary")
    print(f"- Repo ID: {repo_id}")
    print(f"- Release directory: {release_dir}")
    print("- Split: test")
    print(
        "- Features: question_id, question, answer, modality, source, "
        "viz_type, difficulty, task_type, image"
    )
    print(f"- Unique benchmark items merged: {unique_items}")
    print(f"- Prepared rendered rows (test): {len(rows)}")
    print(f"- Missing rendered combinations skipped: {missing_images}")
    print(f"- Modalities: {', '.join(modalities)}")
    print(f"- Visualization types: {', '.join(viz_types)}")
    if dry_run:
        print("- Mode: dry-run (no upload performed)")
    else:
        print("- Mode: upload completed")


def main() -> None:
    args = parse_args()

    data_dir = args.data_dir.resolve()
    dataset_card = args.dataset_card.resolve()
    release_dir = Path("release/huggingface").resolve()

    base_items_path = data_dir / "base_items.jsonl"
    realworld_path = data_dir / "realworld_test.jsonl"
    rendered_root = data_dir / "rendered"

    required_paths = [base_items_path, realworld_path, rendered_root, dataset_card]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    if release_dir.exists():
        shutil.rmtree(release_dir)
    release_dir.mkdir(parents=True, exist_ok=True)

    staged_data_dir = release_dir / "benchmark"
    staged_rendered_root = staged_data_dir / "rendered"

    _copy_file(base_items_path, staged_data_dir / "base_items.jsonl")
    _copy_file(realworld_path, staged_data_dir / "realworld_test.jsonl")
    shutil.copytree(rendered_root, staged_rendered_root, dirs_exist_ok=True)
    _copy_file(dataset_card, release_dir / "README.md")

    merged_items = _load_merged_items(base_items_path, realworld_path)
    image_index = _collect_image_index(staged_rendered_root)
    rows, missing_images = _prepare_rows(
        items=merged_items,
        image_index=image_index,
    )
    dataset_dict = _build_dataset(rows=rows, release_dir=release_dir)

    if not args.dry_run:
        token = args.token or os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError("Missing HuggingFace token. Pass --token or set HF_TOKEN.")

        api = HfApi(token=token)
        api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)
        dataset_dict.push_to_hub(repo_id=args.repo_id, token=token)
        api.upload_file(
            path_or_fileobj=str(release_dir / "README.md"),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
        )

    _print_summary(
        repo_id=args.repo_id,
        release_dir=release_dir,
        rows=rows,
        unique_items=len(merged_items),
        missing_images=missing_images,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
