from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPACE_DIR = PROJECT_ROOT / "release" / "huggingface" / "human_eval_space"
IMAGE_DIR = SPACE_DIR / "data" / "images"
SPACE_REPO = os.environ.get("HF_SPACE_REPO", "ANON-ORG/structviz-bench-human-eval")
HF_TOKEN = os.environ.get("HF_TOKEN")


def main() -> None:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN environment variable must be set.")
    if not IMAGE_DIR.exists():
        raise RuntimeError(f"Image directory not found: {IMAGE_DIR}")

    files = [path for path in IMAGE_DIR.rglob("*") if path.is_file()]
    print(f"Uploading {len(files)} image files to {SPACE_REPO} ...")

    api = HfApi(token=HF_TOKEN)
    operations = []
    for path in sorted(files):
        rel_path = Path("images") / path.name
        operations.append(
            CommitOperationAdd(
                path_in_repo=str(rel_path).replace("\\", "/"),
                path_or_fileobj=str(path),
            )
        )
        print(f"Prepared {rel_path}")
    api.create_commit(
        repo_id=SPACE_REPO,
        repo_type="space",
        operations=operations,
        commit_message="Upload human eval image assets",
    )
    print("Upload complete.")


if __name__ == "__main__":
    main()
