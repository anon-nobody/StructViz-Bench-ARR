from __future__ import annotations

import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPACE_DIR = PROJECT_ROOT / "release" / "huggingface" / "human_eval_space"
DATA_DIR = SPACE_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"
TASK_A_PATH = PROJECT_ROOT / "results" / "human_eval" / "task_a_items.jsonl"
TASK_B_PATH = PROJECT_ROOT / "results" / "human_eval" / "task_b_pairs.jsonl"


def load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def infer_modality(question_id: str) -> str:
    return question_id.split("_", 1)[0]


def source_roots() -> list[Path]:
    return [
        PROJECT_ROOT
        / "release"
        / "huggingface"
        / "benchmark"
        / "rendered"
        / "benchmark"
        / "rendered",
        PROJECT_ROOT / "benchmark" / "rendered",
    ]


def resolve_image(question_id: str, viz_type: str) -> Path | None:
    modality = infer_modality(question_id)
    filename = f"{question_id}_{viz_type}.png"
    for root in source_roots():
        candidate = root / modality / filename
        if candidate.exists():
            return candidate
    return None


def copy_image(question_id: str, viz_type: str) -> str | None:
    source = resolve_image(question_id, viz_type)
    if source is None:
        return None
    modality = infer_modality(question_id)
    target_dir = IMAGE_DIR / modality
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    if not target.exists():
        shutil.copy2(source, target)
    return str(target.relative_to(SPACE_DIR))


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (SPACE_DIR / "responses").mkdir(parents=True, exist_ok=True)

    task_a_items = load_jsonl(TASK_A_PATH)
    task_b_items = load_jsonl(TASK_B_PATH)

    rewritten_a: list[dict[str, object]] = []
    rewritten_b: list[dict[str, object]] = []

    copied = 0
    missing: list[str] = []

    for item in task_a_items:
        item_copy = dict(item)
        image_rel = copy_image(str(item["question_id"]), str(item["viz_type"]))
        if image_rel is None:
            missing.append(f"{item['question_id']} [{item['viz_type']}]")
        else:
            copied += 1
            item_copy["image_path"] = image_rel
            rewritten_a.append(item_copy)

    for item in task_b_items:
        item_copy = dict(item)
        image_a = copy_image(str(item["question_id"]), str(item["viz_a"]))
        image_b = copy_image(str(item["question_id"]), str(item["viz_b"]))
        if image_a is None:
            missing.append(f"{item['question_id']} [{item['viz_a']}]")
        else:
            copied += 1
            item_copy["image_a_path"] = image_a
        if image_b is None:
            missing.append(f"{item['question_id']} [{item['viz_b']}]")
        else:
            copied += 1
            item_copy["image_b_path"] = image_b
        if "image_a_path" in item_copy and "image_b_path" in item_copy:
            rewritten_b.append(item_copy)

    with (DATA_DIR / "task_a_items.jsonl").open("w") as handle:
        for item in rewritten_a:
            handle.write(json.dumps(item, ensure_ascii=True) + "\n")

    with (DATA_DIR / "task_b_pairs.jsonl").open("w") as handle:
        for item in rewritten_b:
            handle.write(json.dumps(item, ensure_ascii=True) + "\n")

    summary = {
        "task_a_requested": len(task_a_items),
        "task_a_exported": len(rewritten_a),
        "task_b_requested": len(task_b_items),
        "task_b_exported": len(rewritten_b),
        "image_references_copied": copied,
        "missing_image_references": len(missing),
    }
    with (DATA_DIR / "bundle_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Prepared HF Space bundle in {SPACE_DIR}")
    print(
        f"Task A items exported with images: {len(rewritten_a)} / {len(task_a_items)}"
    )
    print(
        f"Task B items exported with images: {len(rewritten_b)} / {len(task_b_items)}"
    )
    print(f"Image references copied: {copied}")
    print(f"Missing image references: {len(missing)}")
    if missing:
        print("Examples of missing images:")
        for sample in missing[:10]:
            print(f"  - {sample}")


if __name__ == "__main__":
    main()
