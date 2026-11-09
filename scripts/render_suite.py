"""Render the answer-complete "v2" suite or the tabular calculation-aid "assist" suite.

Examples:
    python scripts/render_suite.py --suite v2 --input benchmark/realworld_test.jsonl \
        --output-dir benchmark/render_v2 --shard 0/8
    python scripts/render_suite.py --suite v2 --output-dir benchmark/render_v2 --merge
    python scripts/render_suite.py --suite assist --input benchmark/realworld_test.jsonl \
        --output-dir benchmark/render_assist

Images go to ``<output-dir>/<modality>/<question_id>_<viz>.png``; one manifest row per image
(``manifest_shard{i}.jsonl`` when sharded, else ``manifest.jsonl``). The v1 renderers and the
released images under ``benchmark/rendered`` / ``benchmark/render_mm`` are not touched.
"""

from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false

import argparse
import collections
import hashlib
import io
import json
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from src.rendering import graph_renderers, tabular_renderers, timeseries_renderers  # noqa: E402
from src.rendering import v2_renderers  # noqa: E402
from src.rendering.render_pipeline import RenderPipeline  # noqa: E402
from src.utils.io_utils import BenchmarkItem  # noqa: E402

_QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", choices=["v2", "assist"], required=True)
    parser.add_argument("--input", type=Path, default=Path("benchmark/realworld_test.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard", type=str, default=None, help="i/n: render items idx %% n == i")
    parser.add_argument("--only-qids", type=Path, default=None,
                        help="Text file with one question_id per line (or a JSON list).")
    parser.add_argument("--modality", choices=["tabular", "timeseries", "graph"], default=None)
    parser.add_argument("--merge", action="store_true",
                        help="Merge manifest_shard*.jsonl (+ manifest_update_*.jsonl) into "
                             "manifest.jsonl and summarise.")
    parser.add_argument("--update", action="store_true",
                        help="Re-render in place: overwrite only images whose PNG bytes changed "
                             "vs <output-dir>/manifest.jsonl; every row (with current constants "
                             "and a 'rerendered' flag) goes to manifest_update[_shard{i}].jsonl "
                             "(apply with --merge).")
    return parser.parse_args()


def is_assist_question(row: dict[str, Any]) -> bool:
    """Tabular comparison questions, or aggregation questions asking for an average.

    "average" must appear outside quoted column names (a column called 'average_x' does not
    make a sum/max question an averaging one).
    """
    if row.get("modality") != "tabular":
        return False
    task = str(row.get("task", ""))
    if task == "comparison":
        return True
    if task == "aggregation":
        unquoted = _QUOTED.sub(" ", str(row.get("question", "")))
        return "average" in unquoted.lower()
    return False


def renderer_constants(suite: str) -> dict[str, Any]:
    """Constants that determine the rendered pixels (stored in every manifest row)."""
    return {
        "suite": suite,
        "matplotlib": matplotlib.__version__,
        "v1_TABLE_IMAGE_MAX_ROWS": tabular_renderers.TABLE_IMAGE_MAX_ROWS,
        "v1_TEXT_VIEW_MAX_ROWS": tabular_renderers.TEXT_VIEW_MAX_ROWS,
        "v1_HEATMAP_MAX_ROWS": tabular_renderers.HEATMAP_MAX_ROWS,
        "v1_TS_TEXT_VIEW_MAX_POINTS": timeseries_renderers.TEXT_VIEW_MAX_POINTS,
        "v1_GRAPH_TEXT_VIEW_MAX_EDGES": graph_renderers.TEXT_VIEW_MAX_EDGES,
        **{k: list(v) if isinstance(v, tuple) else v for k, v in v2_renderers.v2_constants().items()},
    }


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Read, filter and shard the input rows (deterministic order)."""
    with args.input.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if args.suite == "assist":
        rows = [row for row in rows if is_assist_question(row)]
    if args.modality:
        rows = [row for row in rows if row["modality"] == args.modality]
    if args.only_qids:
        text = args.only_qids.read_text().strip()
        wanted = set(json.loads(text)) if text.startswith("[") else set(text.split())
        rows = [row for row in rows if row["question_id"] in wanted]
    if args.shard:
        shard_i, shard_n = (int(part) for part in args.shard.split("/"))
        rows = [row for k, row in enumerate(rows) if k % shard_n == shard_i]
    return rows


def merge(out_dir: Path) -> None:
    """Merge shard manifests and print a summary."""
    rows: list[dict[str, Any]] = []
    shard_paths = sorted(out_dir.glob("manifest_shard*.jsonl"))
    for path in shard_paths or [out_dir / "manifest.jsonl"]:
        rows.extend(json.loads(line) for line in path.open() if line.strip())
    update_paths = sorted(out_dir.glob("manifest_update*.jsonl"))
    if update_paths:
        by_key = {(r["question_id"], r.get("viz_type", "")): r for r in rows}
        n_updates = 0
        for path in update_paths:
            for line in path.open():
                if line.strip():
                    row = json.loads(line)
                    by_key[(row["question_id"], row.get("viz_type", ""))] = row
                    n_updates += 1
        rows = list(by_key.values())
        print(f"applied {n_updates} updated rows from {len(update_paths)} update manifests")
    rows.sort(key=lambda r: (r["question_id"], r.get("viz_type", "")))
    if shard_paths or update_paths:
        with (out_dir / "manifest.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    ok = [r for r in rows if "error" not in r]
    errors = [r for r in rows if "error" in r]
    print(f"merged rows: {len(rows)}  images: {len(ok)}  errors: {len(errors)}")
    print("images per (modality, viz):")
    for key, count in sorted(collections.Counter((r["modality"], r["viz_type"]) for r in ok).items()):
        print(f"  {key[0]:<10} {key[1]:<17} {count}")
    print("canvas sizes (w x h: count) per modality/viz:")
    sizes = collections.defaultdict(collections.Counter)
    for r in ok:
        sizes[(r["modality"], r["viz_type"])][f"{r['width']}x{r['height']}"] += 1
    for key in sorted(sizes):
        print(f"  {key[0]:<10} {key[1]:<17} {dict(sizes[key].most_common())}")
    for r in errors:
        print("ERROR", r["question_id"], r["error"])


def main() -> None:
    """Render the requested suite."""
    args = parse_args()
    if args.merge:
        merge(args.output_dir)
        return
    rows = load_rows(args)
    pipeline = RenderPipeline(suite=args.suite)
    constants = renderer_constants(args.suite)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "manifest_update" if args.update else "manifest"
    manifest_name = (
        f"{prefix}_shard{args.shard.split('/')[0]}.jsonl" if args.shard else f"{prefix}.jsonl"
    )
    old_sha: dict[tuple[str, str], str] = {}
    if args.update:
        with (args.output_dir / "manifest.jsonl").open() as handle:
            for line in handle:
                if line.strip():
                    r = json.loads(line)
                    old_sha[(r["question_id"], r.get("viz_type", ""))] = r.get("sha256", "")
    n_unchanged = 0
    manifest_path = args.output_dir / manifest_name
    start = time.time()
    n_images = 0
    failures: list[str] = []
    with manifest_path.open("w") as manifest:
        for k, row in enumerate(rows):
            item = BenchmarkItem.from_dict(row)
            try:
                rendered = pipeline.render_all({"modality": item.modality, "data": item.data})
            except Exception as exc:  # noqa: BLE001 - record and continue
                failures.append(item.question_id)
                manifest.write(json.dumps({
                    "question_id": item.question_id,
                    "modality": item.modality,
                    "task": item.task,
                    "suite": args.suite,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=3),
                }) + "\n")
                continue
            dropped: list[str] = []
            if item.modality == "tabular":
                num_cols = [str(c) for c in item.data.select_dtypes(include="number").columns]
                dropped = num_cols[v2_renderers.V2_BAR_MAX_PANELS:]
            for viz_name, image in rendered.items():
                path = args.output_dir / item.modality / f"{item.question_id}_{viz_name}.png"
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                data = buffer.getvalue()
                digest = hashlib.sha256(data).hexdigest()
                unchanged = (
                    args.update
                    and old_sha.get((item.question_id, viz_name)) == digest
                    and path.exists()
                )
                if unchanged:
                    n_unchanged += 1
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                    n_images += 1
                row_constants = constants
                if dropped and viz_name in ("bar_chart", "bar_rows", "bar_rows_means"):
                    row_constants = {**constants, "bar_chart_dropped_columns": dropped,
                                     "bar_chart_note": f"bar panels capped at "
                                     f"{v2_renderers.V2_BAR_MAX_PANELS}; "
                                     f"{len(dropped)} numeric columns not drawn"}
                manifest.write(json.dumps({
                    "question_id": item.question_id,
                    "modality": item.modality,
                    "task": item.task,
                    "viz_type": viz_name,
                    "image_path": str(path),
                    "width": image.width,
                    "height": image.height,
                    "sha256": digest,
                    "suite": args.suite,
                    "renderer_constants": row_constants,
                    **({"rerendered": not unchanged} if args.update else {}),
                }) + "\n")
            if (k + 1) % 100 == 0:
                print(f"[{args.shard or 'all'}] {k + 1}/{len(rows)} items, {n_images} images, "
                      f"{time.time() - start:.0f}s", flush=True)
    if args.update:
        print(f"[{args.shard or 'all'}] update: {n_images} images re-rendered, "
              f"{n_unchanged} unchanged")
    print(f"[{args.shard or 'all'}] done: {len(rows)} items, {n_images} images, "
          f"{len(failures)} failures, {time.time() - start:.0f}s -> {manifest_path}")
    for qid in failures:
        print("FAILED", qid)


if __name__ == "__main__":
    main()
