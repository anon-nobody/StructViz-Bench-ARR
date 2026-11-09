"""Fetch ONLY the ChartQA test split (png, tables, annotations, question JSONs).

Reads the official release zip ``ChartQA Dataset.zip`` from the HF dataset repo
``ahmed-masry/ChartQA`` (pinned revision) with HTTP range requests, so the train/val members
are never downloaded (~60 MB transferred instead of 875 MB).

Usage:
    .venv/bin/python scripts/review/fetch_chartqa.py --out data/external/chartqa
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import fsspec

REPO = "ahmed-masry/ChartQA"
REVISION = "af8b6f5c08c95085271561c2a3f9d15f2b5a9031"
URL = (f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/"
       "ChartQA%20Dataset.zip")
PREFIX = "ChartQA Dataset/test/"


def main() -> None:
    """Extract the test members of the release zip."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("data/external/chartqa"))
    args = ap.parse_args()
    dest = args.out / "test"
    dest.mkdir(parents=True, exist_ok=True)
    handle = fsspec.open(URL, block_size=8 * 2**20, cache_type="readahead").open()
    zf = zipfile.ZipFile(handle)
    members = [m for m in zf.infolist() if m.filename.startswith(PREFIX) and not m.is_dir()]
    members.sort(key=lambda m: m.header_offset)  # sequential reads -> readahead cache hits
    n = 0
    for m in members:
        rel = m.filename[len(PREFIX):]
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == m.file_size:
            continue
        with zf.open(m) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
        n += 1
        if n % 500 == 0:
            print(f"{n}/{len(members)}", flush=True)
    digest = hashlib.sha256()
    for p in sorted(dest.rglob("*")):
        if p.is_file():
            digest.update(str(p.relative_to(dest)).encode())
            digest.update(p.read_bytes())
    meta = {"repo": REPO, "revision": REVISION, "url": URL, "member_prefix": PREFIX,
            "n_files": len(members), "tree_sha256": digest.hexdigest()}
    (args.out / "fetch_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
