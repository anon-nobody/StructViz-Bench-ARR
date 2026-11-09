"""Build the ChartQA external-validity set: real charts + human questions, format-controlled.

Steps (all deterministic):
  1. Load ChartQA test_human questions; parse each chart's CSV table. A table "parses cleanly"
     if: first column = row labels (kept as strings), every other column converts to float
     after stripping thousands separators, %, currency symbols and footnote '*', with no
     empty/unparseable cell; >= 1 numeric column; 1..40 rows.
  2. If more than SAMPLE_N questions qualify, draw a stratified (by chart type, proportional,
     largest remainder) sample of SAMPLE_N questions with random.Random(0).
  3. Re-render each qualifying table with RenderPipeline(suite="v2") (5 tabular formats) and
     copy the original ChartQA PNG (resized to <= 1024 px long side) as viz "original_chart".
  4. Write the benchmark JSONL (for eval_local_suite.py --benchmark) and the two manifests.

    .venv/bin/python scripts/review/build_chartqa.py
"""

from __future__ import annotations

# pyright: reportMissingImports=false

import collections
import hashlib
import io
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from render_suite import renderer_constants  # noqa: E402
from src.rendering import v2_renderers  # noqa: E402
from src.rendering.render_pipeline import RenderPipeline  # noqa: E402

SRC = ROOT / "data/external/chartqa/test"
IMG_DIR = ROOT / "benchmark/render_chartqa"
OUT_DIR = ROOT / "results/review/chartqa"
MAX_ROWS = 40
SAMPLE_N = 800
SEED = 0
ORIG_MAX_SIDE = 1024
SUITE = "chartqa"
ORIGINAL_VIZ = "original_chart"
TABULAR_VIZ = ["bar_chart", "heatmap", "table_image", "scatter_plot", "text_only"]

_NUM = re.compile(r"^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$")
# Questions whose wording refers to the original chart's visual encoding (colours, positions);
# such references may not transfer to a re-rendering. Reported as a flag/subset, not removed.
_VISUAL_REF = re.compile(
    r"\b(red|blue|green|orange|yellow|purple|pink|gr[ae]y|black|brown|navy|violet|maroon|"
    r"teal|colou?r(ed|s)?|leftmost|rightmost|left|right|top|bottom|upper|lower|middle|"
    r"tallest|shortest|longest bar)\b", re.I)


def source_of(imgname: str) -> str:
    """ChartQA source site, from the release's file-naming convention."""
    stem = imgname.rsplit(".", 1)[0]
    if stem.startswith(("two_col", "multi_col")):
        return "statista"
    if stem.startswith("OECD"):
        return "oecd"
    if stem.isdigit():
        return "owid" if len(stem) == 14 else "pew"
    return "unknown"


def to_float(cell: str) -> float | None:
    """Parse one cell; None if not a clean number."""
    s = cell.strip().replace(",", "").replace("−", "-")
    s = s.rstrip("*").strip()
    s = re.sub(r"^[$€£¥]", "", s)
    s = s[:-1].strip() if s.endswith("%") else s
    return float(s) if _NUM.match(s) else None


def parse_table(path: Path) -> tuple[pd.DataFrame | None, str]:
    """Return (DataFrame, "ok") or (None, reason)."""
    try:
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as exc:  # noqa: BLE001
        return None, f"csv_error:{type(exc).__name__}"
    if raw.shape[1] < 2:
        return None, "lt2_columns"
    if len(raw) == 0:
        return None, "empty"
    label_col = raw.columns[0]
    out: dict[str, Any] = {str(label_col): [str(v).strip() for v in raw[label_col]]}
    for col in raw.columns[1:]:
        vals = [to_float(str(v)) for v in raw[col]]
        if any(v is None for v in vals):
            return None, "non_numeric_or_missing_cell"
        out[str(col)] = vals
    df = pd.DataFrame(out)
    if len(df) > MAX_ROWS:
        return None, f"gt{MAX_ROWS}_rows"
    return df, "ok"


def stratified_sample(items: list[dict[str, Any]], n: int, key: str) -> list[dict[str, Any]]:
    """Proportional allocation (largest remainder), then random.Random(SEED) within strata."""
    strata: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for it in items:
        strata[it[key]].append(it)
    total = len(items)
    quotas = {k: n * len(v) / total for k, v in strata.items()}
    alloc = {k: math.floor(q) for k, q in quotas.items()}
    rest = n - sum(alloc.values())
    for k in sorted(quotas, key=lambda k: (-(quotas[k] - alloc[k]), k))[:rest]:
        alloc[k] += 1
    rng = random.Random(SEED)
    chosen: list[dict[str, Any]] = []
    for k in sorted(strata):
        pool = sorted(strata[k], key=lambda it: it["question_id"])
        chosen.extend(rng.sample(pool, alloc[k]))
    return sorted(chosen, key=lambda it: it["question_id"])


def png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write to a temp name, then rename (watchers never see a partial manifest)."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> None:  # noqa: PLR0915
    human = json.loads((SRC / "test_human.json").read_text())
    counts: dict[str, Any] = {"test_human_questions": len(human),
                              "test_human_charts": len({h["imgname"] for h in human})}
    reasons: collections.Counter[str] = collections.Counter()
    chart_reasons: dict[str, str] = {}
    tables: dict[str, pd.DataFrame] = {}
    for img in sorted({h["imgname"] for h in human}):
        df, why = parse_table(SRC / "tables" / (img.rsplit(".", 1)[0] + ".csv"))
        chart_reasons[img] = why
        if df is not None:
            tables[img] = df
    reasons.update(chart_reasons.values())
    counts["chart_filter_reasons"] = dict(reasons.most_common())

    items: list[dict[str, Any]] = []
    per_chart_k: collections.Counter[str] = collections.Counter()
    q_reasons: collections.Counter[str] = collections.Counter()
    for h in human:
        img = h["imgname"]
        stem = img.rsplit(".", 1)[0]
        k = per_chart_k[img]
        per_chart_k[img] += 1
        q_reasons[chart_reasons[img]] += 1
        if img not in tables:
            continue
        ann = json.loads((SRC / "annotations" / f"{stem}.json").read_text())
        ctype = str(ann.get("type", "unknown"))
        df = tables[img]
        n_num = df.shape[1] - 1
        title = (ann.get("general_figure_info", {}).get("title", {}) or {}).get("text")
        items.append({
            "question_id": f"chartqa_{stem}_{k}",
            "question": str(h["query"]),
            "answer": str(h["label"]),
            "modality": "tabular",
            "data_id": f"chartqa_{stem}",
            "task": f"chartqa_{ctype}",
            "difficulty": "chartqa_human",
            "viz_methods": [*TABULAR_VIZ, ORIGINAL_VIZ],
            "metadata": {
                "chart_type": ctype, "chartqa_source": source_of(img), "imgname": img,
                "split": "test_human", "n_rows": int(len(df)), "n_numeric_cols": n_num,
                "bar_chart_drops_columns": n_num > v2_renderers.V2_BAR_MAX_PANELS,
                "visual_reference": bool(_VISUAL_REF.search(str(h["query"]))),
                "original_title": title,
            },
            "_df": df,
            "source": f"chartqa_{source_of(img)}",
            "chart_type": ctype,
        })
    counts["question_filter_reasons"] = dict(q_reasons.most_common())
    counts["qualifying_questions"] = len(items)
    counts["qualifying_charts"] = len({it["data_id"] for it in items})
    counts["qualifying_by_chart_type"] = dict(collections.Counter(it["chart_type"] for it in items))
    if len(items) > SAMPLE_N:
        items = stratified_sample(items, SAMPLE_N, "chart_type")
    counts["sampled_questions"] = len(items)
    counts["sampled_charts"] = len({it["data_id"] for it in items})
    for key, fn in [("by_chart_type", lambda it: it["chart_type"]),
                    ("by_source", lambda it: it["metadata"]["chartqa_source"]),
                    ("visual_reference", lambda it: it["metadata"]["visual_reference"]),
                    ("bar_chart_drops_columns",
                     lambda it: it["metadata"]["bar_chart_drops_columns"])]:
        counts[f"sampled_{key}"] = {str(k): v for k, v in
                                    sorted(collections.Counter(map(fn, items)).items())}

    pipeline = RenderPipeline(suite="v2")
    constants = renderer_constants("v2")
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rendered_cache: dict[str, dict[str, bytes]] = {}
    orig_cache: dict[str, tuple[bytes, int, int, int, int]] = {}
    manifest: list[dict[str, Any]] = []
    render_errors: list[str] = []
    for n_it, it in enumerate(items, 1):
        did, df = it["data_id"], it["_df"]
        if did not in rendered_cache:
            try:
                imgs = pipeline.render_all({"modality": "tabular", "data": df})
                rendered_cache[did] = {v: png_bytes(imgs[v]) for v in TABULAR_VIZ}
            except Exception as exc:  # noqa: BLE001
                render_errors.append(f"{did}: {type(exc).__name__}: {exc}")
                rendered_cache[did] = {}
            src_img = Image.open(SRC / "png" / it["metadata"]["imgname"]).convert("RGB")
            w0, h0 = src_img.size
            scale = min(1.0, ORIG_MAX_SIDE / max(w0, h0))
            if scale < 1.0:
                src_img = src_img.resize((round(w0 * scale), round(h0 * scale)),
                                         Image.Resampling.LANCZOS)
            orig_cache[did] = (png_bytes(src_img), src_img.width, src_img.height, w0, h0)
        base = {"question_id": it["question_id"], "question": it["question"],
                "answer": it["answer"], "modality": "tabular", "task": it["task"],
                "chart_type": it["chart_type"], "chartqa_source": it["metadata"]["chartqa_source"],
                "data_id": did, "suite": SUITE}
        for viz in TABULAR_VIZ:
            data = rendered_cache[did].get(viz)
            if data is None:
                continue
            path = IMG_DIR / f"{it['question_id']}_{viz}.png"
            path.write_bytes(data)
            with Image.open(io.BytesIO(data)) as im:
                w, h = im.size
            row_constants = constants
            if viz == "bar_chart" and it["metadata"]["bar_chart_drops_columns"]:
                dropped = [str(c) for c in df.columns[1:]][v2_renderers.V2_BAR_MAX_PANELS:]
                row_constants = {**constants, "bar_chart_dropped_columns": dropped}
            manifest.append({**base, "viz_type": viz,
                             "image_path": str(path.relative_to(ROOT)), "width": w, "height": h,
                             "sha256": hashlib.sha256(data).hexdigest(),
                             "renderer_suite": "v2", "renderer_constants": row_constants})
        data, w, h, w0, h0 = orig_cache[did]
        path = IMG_DIR / f"{it['question_id']}_{ORIGINAL_VIZ}.png"
        path.write_bytes(data)
        manifest.append({**base, "viz_type": ORIGINAL_VIZ,
                         "image_path": str(path.relative_to(ROOT)), "width": w, "height": h,
                         "sha256": hashlib.sha256(data).hexdigest(),
                         "original_width": w0, "original_height": h0,
                         "original_file": f"data/external/chartqa/test/png/"
                                          f"{it['metadata']['imgname']}"})
        if n_it % 100 == 0:
            print(f"{n_it}/{len(items)} items", flush=True)

    bench_rows = []
    for it in items:
        df = it["_df"]
        bench_rows.append({
            k: v for k, v in it.items() if k not in ("_df", "chart_type")
        } | {"data_format": "dataframe_records",
             "data": json.loads(df.to_json(orient="records")),
             "data_columns": [str(c) for c in df.columns]})
    write_jsonl_atomic(OUT_DIR / "chartqa_items.jsonl", bench_rows)
    counts["render_errors"] = render_errors
    counts["manifest_rows"] = len(manifest)
    counts["manifest_rows_by_viz"] = dict(collections.Counter(r["viz_type"] for r in manifest))
    (OUT_DIR / "build_counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    write_jsonl_atomic(OUT_DIR / "manifest_chartqa_original.jsonl",
                       [r for r in manifest if r["viz_type"] == ORIGINAL_VIZ])
    write_jsonl_atomic(OUT_DIR / "manifest_chartqa.jsonl", manifest)
    print(json.dumps({k: v for k, v in counts.items() if k != "render_errors"}, indent=2))
    print(f"render errors: {len(render_errors)}", *render_errors[:10], sep="\n")


if __name__ == "__main__":
    main()
