#!/usr/bin/env python3
"""Vision-attention probe for StructViz-Bench (mechanism, Section 5).

Question: when a format is hard to read, does the model attend to the image differently?
For each (question, image) we run one forward with output_attentions and measure the fraction
of attention mass at the answer-generation position that lands on IMAGE tokens (vs. text).
We average this "image-attention share" per visualization format. Hypothesis: lossy/hard
formats (recurrence_plot, gaf, scatter) receive lower or less-focused image attention than
readable ones (table_image, line_plot).

NOTE: attention weights are only returned with attn_implementation="eager" (SDPA/flash return
None). Eager attention is memory-heavy, so we use small images and few samples per format.

REQUIREMENTS: same env as train_consistency_lora.py (transformers>=4.49, qwen-vl-utils, torch).
USAGE:
    CUDA_VISIBLE_DEVICES=3 HF_HOME=/.../.cache/huggingface \
    python attention_probe.py --manifest scripts/mitigation/pairs.jsonl --per_format 30
"""
from __future__ import annotations
import argparse, json, os, random
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--manifest", required=True)
    p.add_argument("--per_format", type=int, default=30, help="images sampled per viz format")
    p.add_argument("--max_pixels", type=int, default=200704)   # 256*28*28, keep small for eager attn
    p.add_argument("--min_pixels", type=int, default=50176)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    import torch
    from transformers import AutoProcessor
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
    except ImportError:
        from transformers import AutoModelForImageTextToText as VLModel
    from qwen_vl_utils import process_vision_info

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model_id, max_pixels=args.max_pixels,
                                              min_pixels=args.min_pixels)
    model = VLModel.from_pretrained(args.model_id, torch_dtype=torch.bfloat16,
                                    attn_implementation="eager").to(device).eval()
    img_tok = model.config.image_token_id

    # collect (viz, image_path, question) samples
    by_fmt = defaultdict(list)
    for line in open(args.manifest):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        for viz, path in r.get("images", {}).items():
            if os.path.exists(path):
                by_fmt[viz].append((path, r["question"]))
    share = {}
    for viz, samples in by_fmt.items():
        rng.shuffle(samples)
        vals = []
        for path, q in samples[:args.per_format]:
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": path, "max_pixels": args.max_pixels,
                 "min_pixels": args.min_pixels},
                {"type": "text", "text": q}]}]
            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            images, videos = process_vision_info(msgs)
            inp = processor(text=[text], images=images, videos=videos, do_resize=False,
                            return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inp, output_attentions=True)
            attn = out.attentions[-1][0].float().mean(0)          # [seq, seq] last layer, mean heads
            last = attn[-1]                                        # attn from the answer-gen position
            img_mask = (inp["input_ids"][0] == img_tok)
            vals.append(float(last[img_mask].sum() / (last.sum() + 1e-9)))
        if vals:
            share[viz] = 100 * sum(vals) / len(vals)
    print(f"{'viz_format':18} {'image-attention share %':>24} {'n':>5}")
    for viz, s in sorted(share.items(), key=lambda kv: -kv[1]):
        print(f"{viz:18} {s:24.1f} {min(len(by_fmt[viz]), args.per_format):5d}")
    print("\nHypothesis: readable formats (table_image/line_plot) draw more focused image "
          "attention than lossy ones (recurrence_plot/gaf). Compare against per-format EM.")


if __name__ == "__main__":
    main()
