from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPACE_DIR = PROJECT_ROOT / "release" / "huggingface" / "human_eval_space"
IMAGE_DIR = SPACE_DIR / "data" / "images"
TASK_FILES = [
    SPACE_DIR / "data" / "task_a_items.jsonl",
    SPACE_DIR / "data" / "task_b_pairs.jsonl",
]


def safe_name(name: str) -> str:
    return (
        name.replace("::", "__").replace(":", "_").replace("=", "-").replace(" ", "_")
    )


def rename_images() -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not IMAGE_DIR.exists():
        return mapping
    for path in sorted(IMAGE_DIR.rglob("*.png")):
        safe = safe_name(path.name)
        if safe == path.name:
            mapping[path.name] = path.name
            continue
        target = path.with_name(safe)
        path.rename(target)
        mapping[path.name] = target.name
    return mapping


def rewrite_task_files(mapping: dict[str, str]) -> None:
    for task_file in TASK_FILES:
        rows = []
        with task_file.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                for key in ("image_path", "image_a_path", "image_b_path"):
                    value = row.get(key)
                    if isinstance(value, str):
                        old_name = Path(value).name
                        new_name = mapping.get(old_name)
                        if new_name:
                            row[key] = f"images/{new_name}"
                rows.append(row)
        with task_file.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    mapping = rename_images()
    rewrite_task_files(mapping)
    print(f"Renamed {sum(1 for k, v in mapping.items() if k != v)} images")


if __name__ == "__main__":
    main()
