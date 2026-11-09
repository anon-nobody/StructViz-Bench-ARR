"""Regression check: RenderPipeline(suite="v1") still reproduces the released v1 images.

Picks 4 items per modality (smallest / 1/3 / 2/3 / largest payload among items that have a
stored v1 image), re-renders them and compares against the stored PNGs:
    tabular            -> benchmark/rendered/benchmark/rendered/tabular/
    timeseries, graph  -> benchmark/render_mm/benchmark/rendered/{timeseries,graph}/
Prints MATCH (pixel-identical; BYTES if also byte-identical when re-encoded) or DIFF with the
max absolute channel difference and the fraction of differing pixels.

Usage: python scripts/analysis/check_v1_unchanged.py [--input benchmark/realworld_test.jsonl]
"""

from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false

import argparse
import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.rendering.render_pipeline import RenderPipeline  # noqa: E402
from src.utils.io_utils import BenchmarkItem  # noqa: E402

STORED_DIRS = {
    "tabular": ROOT / "benchmark/rendered/benchmark/rendered/tabular",
    "timeseries": ROOT / "benchmark/render_mm/benchmark/rendered/timeseries",
    "graph": ROOT / "benchmark/render_mm/benchmark/rendered/graph",
}
ITEMS_PER_MODALITY = 4


def _size(row: dict) -> int:
    return len(row["data"]["nodes"]) if row["modality"] == "graph" else len(row["data"])


def main() -> int:
    """Run the check; return 0 if every image is pixel-identical."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "benchmark/realworld_test.jsonl")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    rows = [json.loads(line) for line in args.input.open() if line.strip()]
    pipeline = RenderPipeline(suite="v1")
    n_match = n_bytes = n_total = 0
    for modality, stored_dir in STORED_DIRS.items():
        available = set(p.name for p in stored_dir.iterdir())
        cands = [
            r for r in rows
            if r["modality"] == modality and f"{r['question_id']}_text_only.png" in available
        ]
        cands.sort(key=lambda r: (_size(r), r["question_id"]))
        picks = [cands[int(round(q * (len(cands) - 1) / (ITEMS_PER_MODALITY - 1)))]
                 for q in range(ITEMS_PER_MODALITY)]
        for row in picks:
            item = BenchmarkItem.from_dict(row)
            rendered = pipeline.render_all({"modality": item.modality, "data": item.data})
            for viz, image in rendered.items():
                name = f"{item.question_id}_{viz}.png"
                stored_path = stored_dir / name
                n_total += 1
                if not stored_path.exists():
                    print(f"MISSING  {modality:<10} {name}")
                    continue
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                same_bytes = buf.getvalue() == stored_path.read_bytes()
                new = np.asarray(image.convert("RGB"), dtype=np.int16)
                old = np.asarray(Image.open(stored_path).convert("RGB"), dtype=np.int16)
                if new.shape != old.shape:
                    print(f"DIFF     {modality:<10} {name}  shape {old.shape} -> {new.shape}")
                    continue
                diff = np.abs(new - old)
                if diff.max() == 0:
                    n_match += 1
                    n_bytes += int(same_bytes)
                    tag = "BYTES" if same_bytes else "pixels"
                    print(f"MATCH    {modality:<10} {name}  (size={_size(row)}, identical {tag})")
                else:
                    frac = float((diff.max(axis=2) > 0).mean())
                    print(f"DIFF     {modality:<10} {name}  (size={_size(row)}) "
                          f"max|d|={int(diff.max())} differing_px={frac:.4%}")
    print(f"\n{n_match}/{n_total} pixel-identical ({n_bytes} also byte-identical PNG encodings)")
    return 0 if n_match == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
