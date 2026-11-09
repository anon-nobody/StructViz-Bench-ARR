#!/usr/bin/env python3
"""Build the consistency-LoRA training manifest from the benchmark's rendered images.

For each base question_id it collects the question text + answer (from a results JSONL) and
all available per-format PNGs (from the render dir), emitting one JSONL line:
    {"question_id","question","answer","images":{viz_type: abs_path, ...}}
Only questions with >=2 formats are kept (needed for the consistency term).

Usage:
    python build_pair_manifest.py \
        --qa results/full_gpt4o.jsonl \
        --img_root benchmark/rendered/benchmark/rendered \
        --out scripts/mitigation/pairs.jsonl
"""
from __future__ import annotations
import argparse, json, os, glob

VIZ = ["bar_chart", "heatmap", "scatter_plot", "table_image", "text_only",
       "line_plot", "gaf", "recurrence_plot", "node_link", "adjacency_matrix",
       "circular_layout"]  # longest-first match to avoid prefix ambiguity
VIZ.sort(key=len, reverse=True)


def parse_qid_viz(fname):
    stem = fname[:-4] if fname.endswith(".png") else fname
    for v in VIZ:
        if stem.endswith("_" + v):
            return stem[: -(len(v) + 1)], v
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True, help="results JSONL with question_id/question/answer")
    ap.add_argument("--img_root", required=True, help="dir tree containing <qid>_<viz>.png")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    qa = {}
    for line in open(a.qa):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        qid = r.get("question_id")
        if qid and qid not in qa:
            qa[qid] = (r.get("question", ""), str(r.get("answer", "")))

    images = {}  # qid -> {viz: path}
    for path in glob.glob(os.path.join(a.img_root, "**", "*.png"), recursive=True):
        qid, viz = parse_qid_viz(os.path.basename(path))
        if qid and viz:
            images.setdefault(qid, {})[viz] = os.path.abspath(path)

    n = kept = 0
    with open(a.out, "w") as f:
        for qid, imgs in images.items():
            n += 1
            if qid not in qa or len(imgs) < 2:
                continue
            q, ans = qa[qid]
            f.write(json.dumps({"question_id": qid, "question": q, "answer": ans,
                                "images": imgs}) + "\n")
            kept += 1
    print(f"scanned {n} question_ids with images; wrote {kept} usable (>=2 formats) to {a.out}")


if __name__ == "__main__":
    main()
