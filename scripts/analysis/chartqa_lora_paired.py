"""Paired chart-cluster comparisons for the ChartQA LoRA transfer test.

For lambda0 and lambda1 vs. base, on (a) the original charts and (b) the five re-renderings
pooled, restricted to questions without visual references: difference in relaxed accuracy with a
95% percentile bootstrap over charts (B=5000, seed 0). Writes chartqa_lora_paired.json.
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/analysis"))
import chartqa_score as cs  # noqa: E402

D = ROOT / "results/review/chartqa"
items = {json.loads(l)["question_id"]: json.loads(l) for l in (D / "chartqa_items.jsonl").open()}


def load(fn: str) -> dict:
    out = {}
    for line in (D / fn).open():
        r = json.loads(line)
        it = items[r["question_id"]]
        out[(r["question_id"], r["viz_type"])] = float(cs.relaxed_correct(str(r.get("prediction", "")), str(it["answer"])))
    return out


def chart_of(qid: str) -> str:
    it = items[qid]
    return str(it.get("data_id") or it.get("chart_id") or qid)


def visual(qid: str) -> bool:
    return bool(items[qid]["metadata"].get("visual_reference"))


def paired(a: dict, b: dict, fmts: set[str]) -> dict:
    by = defaultdict(lambda: [0.0, 0])
    for k, v in a.items():
        q, f = k
        if f not in fmts or visual(q) or k not in b:
            continue
        c = by[chart_of(q)]
        c[0] += v - b[k]
        c[1] += 1
    cl = list(by.values())
    est = 100 * sum(x for x, _ in cl) / sum(n for _, n in cl)
    rng = random.Random(0)
    boots = []
    for _ in range(5000):
        s = [cl[rng.randrange(len(cl))] for _ in cl]
        boots.append(100 * sum(x for x, _ in s) / sum(n for _, n in s))
    boots.sort()
    return {"diff_pp": round(est, 2), "ci": [round(boots[124], 2), round(boots[4874], 2)],
            "n_pairs": sum(n for _, n in cl), "n_charts": len(cl)}


def main() -> None:
    base = {**load("pred_lora_base_original.jsonl"), **load("pred_lora_base_all.jsonl")}
    res = {}
    rer = {"bar_chart", "heatmap", "table_image", "scatter_plot", "text_only"}
    for s in ("lambda0", "lambda1"):
        m = {**load(f"pred_lora_{s}_original.jsonl"), **load(f"pred_lora_{s}_all.jsonl")}
        res[s] = {"original": paired(m, base, {"original_chart"}), "rerendered": paired(m, base, rer)}
    out = ROOT / "scripts/analysis/chartqa_lora_paired.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
