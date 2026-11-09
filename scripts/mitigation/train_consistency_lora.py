#!/usr/bin/env python3
"""Consistency-regularized LoRA fine-tuning of Qwen2.5-VL-7B for format robustness.

Idea (StructViz-Bench mitigation, axis B): the same (data, question) rendered in two
visualization formats should yield the same answer. We fine-tune with
    loss = task_LM_loss(format A) + task_LM_loss(format B)
           + lambda * KL_consistency(answer-token distributions of A vs B)
plus format augmentation (each step samples a different format pair per question). Only LoRA
adapters on the LLM are trained; the vision encoder is frozen. This is the training-based
counterpart to the training-free selector/ensemble in scripts/mitigation/.

REQUIREMENTS (install on a GPU box; NOT runnable here):
    pip install "transformers>=4.49" peft accelerate qwen-vl-utils torch pillow
Base model Qwen2.5-VL-7B-Instruct is already in the local HF cache.

DATA MANIFEST (JSONL, one base question per line):
    {"question_id": "...", "question": "...", "answer": "...",
     "images": {"bar_chart": "/abs/....png", "table_image": "/abs/....png", ...}}
Build it from the benchmark's rendered images (each question_id has one PNG per viz_type).
Only questions with >=2 image formats are usable for the consistency term.

USAGE:
    python train_consistency_lora.py --manifest pairs.jsonl --output_dir ckpt/consist_lora \
        --lambda_consistency 1.0 --epochs 1 --lr 1e-4
    # quick shape/sanity check without a full run:
    python train_consistency_lora.py --manifest pairs.jsonl --dry_run
"""
from __future__ import annotations

import argparse
import json
import os
import random

# --- heavy deps imported lazily so --help / py_compile work without a GPU env ---


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.expanduser(
        "~/.cache/huggingface/hub"),  # informational; HF resolves by repo id below
        help="(unused) local cache note")
    p.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--manifest", required=True, help="JSONL of base questions with per-format images")
    p.add_argument("--output_dir", default="checkpoints/consistency_lora")
    p.add_argument("--lambda_consistency", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad_accum", type=int, default=8, help="pairs per optimizer step")
    p.add_argument("--max_pixels", type=int, default=1280 * 28 * 28)
    p.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry_run", action="store_true", help="build one pair, print shapes, exit")
    return p.parse_args()


# ----------------------------- data -----------------------------
def load_manifest(path):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            imgs = {k: v for k, v in r.get("images", {}).items() if v and os.path.exists(v)}
            if len(imgs) >= 2:
                items.append({"q": r["question"], "a": str(r["answer"]), "images": imgs})
    if not items:
        raise SystemExit("No usable items (need >=2 existing image formats per question).")
    return items


def sample_pair(item, rng):
    """Pick two distinct formats of the same question."""
    v1, v2 = rng.sample(list(item["images"].keys()), 2)
    return (item["images"][v1], item["q"], item["a"]), (item["images"][v2], item["q"], item["a"])


# ------------------------ tokenization/labels ------------------------
def build_inputs(processor, image_path, question, answer, max_pixels, min_pixels, device):
    """Return processor inputs (prompt+answer) and the answer-token span [start,end)."""
    from qwen_vl_utils import process_vision_info

    user = [{"role": "user", "content": [
        {"type": "image", "image": image_path, "max_pixels": max_pixels, "min_pixels": min_pixels},
        {"type": "text", "text": question}]}]
    # prompt (with generation primer) -> its token length is the label mask boundary
    prompt_text = processor.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
    full_text = prompt_text + answer + processor.tokenizer.eos_token

    images, videos = process_vision_info(user)  # Qwen2.5-VL: default image_patch_size=14
    prompt_inputs = processor(text=[prompt_text], images=images, videos=videos,
                              do_resize=False, return_tensors="pt")
    full_inputs = processor(text=[full_text], images=images, videos=videos,
                            do_resize=False, return_tensors="pt")
    prompt_len = prompt_inputs["input_ids"].shape[1]
    seq_len = full_inputs["input_ids"].shape[1]

    labels = full_inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100  # mask the prompt + image tokens; train only on the answer
    full_inputs["labels"] = labels
    full_inputs = {k: v.to(device) for k, v in full_inputs.items()}
    return full_inputs, prompt_len, seq_len  # answer span = [prompt_len-1, seq_len-1) in logits


def consistency_kl(logits_a, span_a, logits_b, span_b):
    """Symmetric KL between the two formats' next-token distributions over the (identical)
    answer tokens. Answer token *count* is the same for both formats; align step-by-step."""
    import torch
    import torch.nn.functional as F

    sa0, sa1 = span_a
    sb0, sb1 = span_b
    n = min(sa1 - sa0, sb1 - sb0)
    if n <= 0:
        return logits_a.new_zeros(())
    la = logits_a[0, sa0:sa0 + n]          # [n, vocab]
    lb = logits_b[0, sb0:sb0 + n]
    pa, pb = F.log_softmax(la, -1), F.log_softmax(lb, -1)
    kl = 0.5 * (F.kl_div(pa, pb, log_target=True, reduction="batchmean")
                + F.kl_div(pb, pa, log_target=True, reduction="batchmean"))
    return kl


# ----------------------------- train -----------------------------
def main():
    args = parse_args()
    random.seed(args.seed)
    rng = random.Random(args.seed)

    import torch
    # Fix LoRA initialisation and dropout as well (added for the multi-seed rerun).
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    from transformers import AutoProcessor
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
    except ImportError:  # newer transformers unifies under this class
        from transformers import AutoModelForImageTextToText as VLModel
    from peft import LoraConfig, get_peft_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model_id, max_pixels=args.max_pixels,
                                              min_pixels=args.min_pixels)
    _dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = VLModel.from_pretrained(args.model_id, torch_dtype=_dtype, device_map=None).to(device)
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],  # LLM only; vision tower frozen
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()   # cut activation memory
    model.enable_input_require_grads()       # required for grad-checkpointing with a frozen base

    items = load_manifest(args.manifest)
    print(f"{len(items)} usable questions (>=2 formats).")

    if args.dry_run:
        (imgA, q, a), (imgB, _, _) = sample_pair(items[0], rng)
        inA, pA, sA = build_inputs(processor, imgA, q, a, args.max_pixels, args.min_pixels, device)
        inB, pB, sB = build_inputs(processor, imgB, q, a, args.max_pixels, args.min_pixels, device)
        outA = model(**inA)
        outB = model(**inB)
        task = outA.loss + outB.loss
        cons = consistency_kl(outA.logits, (pA - 1, sA - 1), outB.logits, (pB - 1, sB - 1))
        loss = task + args.lambda_consistency * cons
        loss.backward()
        print(f"DRY-RUN OK  inA={tuple(inA['input_ids'].shape)} inB={tuple(inB['input_ids'].shape)} "
              f"task_loss={task.item():.4f} consistency_KL={float(cons):.4f} "
              f"total={loss.item():.4f}  (forward+backward succeeded)")
        return

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()
    step = 0
    for epoch in range(args.epochs):
        order = list(range(len(items)))
        rng.shuffle(order)
        opt.zero_grad()
        for i, idx in enumerate(order):
            (imgA, q, a), (imgB, _, _) = sample_pair(items[idx], rng)
            inA, pA, sA = build_inputs(processor, imgA, q, a, args.max_pixels, args.min_pixels, device)
            inB, pB, sB = build_inputs(processor, imgB, q, a, args.max_pixels, args.min_pixels, device)
            outA = model(**inA)
            outB = model(**inB)
            task = outA.loss + outB.loss
            cons = consistency_kl(outA.logits, (pA - 1, sA - 1),
                                  outB.logits, (pB - 1, sB - 1))
            loss = (task + args.lambda_consistency * cons) / args.grad_accum
            loss.backward()
            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad(); step += 1
                if step % 20 == 0:
                    print(f"epoch {epoch} step {step} task={task.item():.3f} "
                          f"cons={float(cons):.4f}")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)      # saves LoRA adapters only
    processor.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapters to {args.output_dir}. "
          f"Evaluate with the benchmark eval pipeline (load base + adapters) and compare "
          f"the sensitivity gap / Consistency Rate before vs after.")


if __name__ == "__main__":
    main()
