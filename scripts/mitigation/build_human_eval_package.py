#!/usr/bin/env python3
"""Build a self-contained, distributable human-evaluation package (Task A: answer legibility).

Draws a stratified sample of 100 (question, visualization) instances balanced across the three
modalities and spread over formats (and tasks where possible), copies the rendered images into
the package so it is self-contained, and writes one rating sheet per annotator plus instructions
and an assembly script that feeds scripts/mitigation/human_eval_aggregate.py.

Inputs (already produced earlier in the pipeline):
  - scripts/mitigation/pairs.jsonl          (tabular: qid, question, answer, images{viz:abs_path})
  - scripts/mitigation/pairs_tsgraph.jsonl  (timeseries+graph)
  - results/full_gpt4o.jsonl                (qid -> task, for stratification/context; optional)

Usage:
  python scripts/mitigation/build_human_eval_package.py --out human_eval_package --n 100
"""
from __future__ import annotations
import argparse, csv, json, os, random, shutil
import os

MODS = ["tabular", "timeseries", "graph"]


def load_pairs(*paths):
    items = []
    for p in paths:
        if not os.path.exists(p):
            continue
        for line in open(p):
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def task_map(path):
    m = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                r = json.loads(line)
                m.setdefault(r.get("question_id"), r.get("task", ""))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="human_eval_package")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--annotators", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    root = os.environ.get("STRUCTVIZ_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    pairs = load_pairs(os.path.join(root, "scripts/mitigation/pairs.jsonl"),
                       os.path.join(root, "scripts/mitigation/pairs_tsgraph.jsonl"))
    tasks = task_map(os.path.join(root, "results/full_gpt4o.jsonl"))

    by_mod = {m: [] for m in MODS}
    for it in pairs:
        mod = it["question_id"].split("_")[0]
        if mod in by_mod:
            by_mod[mod].append(it)

    per_mod = [a.n // 3 + (1 if i < a.n % 3 else 0) for i in range(3)]  # 34/33/33
    selected = []
    for mod, k in zip(MODS, per_mod):
        pool = by_mod[mod][:]
        rng.shuffle(pool)
        fmts = sorted({v for it in pool for v in it["images"]})
        picked = 0; fi = 0; seen_task = set()
        # first pass: prefer task diversity; round-robin over formats
        for it in pool:
            if picked >= k:
                break
            viz = fmts[fi % len(fmts)]
            if viz not in it["images"]:
                viz = rng.choice(list(it["images"]))
            path = it["images"][viz]
            if not os.path.exists(path):
                continue
            selected.append({"question_id": it["question_id"], "modality": mod,
                             "task": tasks.get(it["question_id"], ""),
                             "viz_type": viz, "question": it["question"],
                             "ground_truth": str(it["answer"]), "src_image": path})
            picked += 1; fi += 1
    rng.shuffle(selected)

    # write package
    outdir = os.path.join(root, a.out)
    imgdir = os.path.join(outdir, "images")
    os.makedirs(imgdir, exist_ok=True)
    with open(os.path.join(outdir, "items.jsonl"), "w") as fj:
        for i, s in enumerate(selected):
            item_id = f"H{i:03d}"
            ext = os.path.splitext(s["src_image"])[1] or ".png"
            rel = f"images/{item_id}_{s['modality']}_{s['viz_type']}{ext}"
            shutil.copy2(s["src_image"], os.path.join(outdir, rel))
            s["item_id"] = item_id; s["image"] = rel
            fj.write(json.dumps({k: s[k] for k in
                     ["item_id", "question_id", "modality", "task", "viz_type",
                      "question", "ground_truth", "image"]}) + "\n")

    cols = ["item_id", "modality", "task", "viz_type", "image", "question",
            "ground_truth", "rating", "your_answer", "notes"]
    for ann in range(1, a.annotators + 1):
        with open(os.path.join(outdir, f"ratings_annotator{ann}.csv"), "w", newline="") as fc:
            w = csv.writer(fc); w.writerow(cols)
            for s in selected:
                w.writerow([s["item_id"], s["modality"], s["task"], s["viz_type"], s["image"],
                            s["question"], s["ground_truth"], "", "", ""])
    print(f"package: {outdir}")
    print(f"  items: {len(selected)}  (per modality: "
          f"{ {m: sum(1 for s in selected if s['modality']==m) for m in MODS} })")
    print(f"  images copied: {len(selected)}")
    print(f"  sheets: ratings_annotator1..{a.annotators}.csv")


if __name__ == "__main__":
    main()
