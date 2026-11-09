#!/usr/bin/env python3
"""Evaluate format sensitivity BEFORE vs AFTER consistency-LoRA (Section 6 numbers).

For each eval question and each visualization format, greedily generate an answer, score it
against the ground truth (exact match with numeric tolerance / lenient substring), and report
per-format EM, the best-worst sensitivity gap, and the Consistency Rate (agreement of the
model's predictions across a question's formats). Run once with --adapter to get the AFTER
row; run without it for BEFORE. Same resolution as training for a fair comparison.

USAGE (same env as training):
  # before (base model):
  python eval_lora.py --manifest scripts/mitigation/pairs_eval.jsonl
  # after (base + LoRA adapter):
  python eval_lora.py --manifest scripts/mitigation/pairs_eval.jsonl --adapter checkpoints/consistency_lora
"""
from __future__ import annotations
import argparse, json, os, re
from collections import defaultdict, Counter


def norm(s):
    s = str(s).strip().lower()
    m = re.search(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    return m.group() if m else s


def match(pred, gold, abs_tol=0.05, rel_tol=0.01):
    """Tolerance-based numeric match, else exact normalised string match.

    An earlier version also returned True when the gold string appeared anywhere inside a
    longer answer (substring credit). That rule was removed after review: the paper's LoRA
    table is now computed by rescore_lora.py from the released records without it.
    """
    p, g = norm(pred), norm(gold)
    if p == g:
        return True
    try:
        pf, gf = float(p), float(g)
        return abs(pf - gf) <= max(abs_tol, rel_tol * abs(gf))
    except ValueError:
        return False


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir; omit for base model")
    ap.add_argument("--max_pixels", type=int, default=200704)
    ap.add_argument("--min_pixels", type=int, default=50176)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="cap #questions (0 = all)")
    ap.add_argument("--dump", default=None, help="write per-question records (JSONL) for offline stats")
    return ap.parse_args()


def main():
    a = parse_args()
    import torch
    from transformers import AutoProcessor
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
    except ImportError:
        from transformers import AutoModelForImageTextToText as VLModel
    from qwen_vl_utils import process_vision_info

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(a.model_id, max_pixels=a.max_pixels,
                                              min_pixels=a.min_pixels)
    model = VLModel.from_pretrained(a.model_id, torch_dtype=torch.bfloat16).to(device).eval()
    tag = "BASE"
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter).eval()
        tag = "AFTER-LoRA"

    items = [json.loads(l) for l in open(a.manifest) if l.strip()]
    if a.limit:
        items = items[:a.limit]

    fmt_hit = defaultdict(lambda: [0, 0])   # (modality,viz) -> [correct, total]
    cr = defaultdict(lambda: [0.0, 0])       # modality -> [cr_sum, n]
    records = []                              # per-question, for offline paired stats
    for r in items:
        q, gold = r["question"], str(r["answer"])
        mod = r["question_id"].split("_")[0]   # tabular / timeseries / graph
        preds = {}; correct = {}
        for viz, path in r["images"].items():
            if not os.path.exists(path):
                continue
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": path, "max_pixels": a.max_pixels,
                 "min_pixels": a.min_pixels},
                {"type": "text", "text": q + " Answer concisely. Give only the final answer."}]}]
            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            imgs, vids = process_vision_info(msgs)
            inp = processor(text=[text], images=imgs, videos=vids, do_resize=False,
                            return_tensors="pt").to(device)
            with torch.no_grad():
                gen = model.generate(**inp, max_new_tokens=a.max_new_tokens, do_sample=False)
            out = processor.batch_decode(
                [gen[0][inp["input_ids"].shape[1]:]], skip_special_tokens=True)[0].strip()
            preds[viz] = out
            correct[viz] = int(match(out, gold))
            fmt_hit[(mod, viz)][1] += 1
            fmt_hit[(mod, viz)][0] += correct[viz]
        if len(preds) >= 2:
            normed = [norm(p) for p in preds.values()]
            agree = tot = 0
            for i in range(len(normed)):
                for j in range(i + 1, len(normed)):
                    tot += 1; agree += (normed[i] == normed[j])
            cr[mod][0] += agree / tot; cr[mod][1] += 1
        records.append({"qid": r["question_id"], "modality": mod,
                        "correct": correct, "preds": {k: norm(v) for k, v in preds.items()}})

    print(f"=== {tag} (n={len(items)} questions, res={a.max_pixels}px) ===")
    for mod in ["tabular", "timeseries", "graph"]:
        ems = {v: 100 * c / t for (m, v), (c, t) in fmt_hit.items() if m == mod and t}
        if not ems:
            continue
        gap = max(ems.values()) - min(ems.values())
        crm = 100 * cr[mod][0] / cr[mod][1] if cr[mod][1] else 0.0
        fmts = "  ".join(f"{v}={ems[v]:.1f}" for v in sorted(ems, key=lambda x: -ems[x]))
        print(f"[{mod:10}] gap={gap:4.1f}pp  CR={crm:4.1f}%  ({cr[mod][1]} q)  | {fmts}")
    if a.dump:
        import json as _json
        with open(a.dump, "w") as fh:
            for rec in records:
                fh.write(_json.dumps(rec) + "\n")
        print(f"dumped {len(records)} per-question records -> {a.dump}")


if __name__ == "__main__":
    main()
