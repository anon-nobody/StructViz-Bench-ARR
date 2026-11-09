#!/usr/bin/env bash
# lambda=0 control: identical data/epochs/lr/resolution/seed to the reported run, consistency
# term switched off. Isolates what the consistency loss adds over plain multi-format SFT.
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python; M=scripts/mitigation; OUT=checkpoints/consistency_lora_mm_lambda0
echo "[$(date +%T)] train lambda=0"
$PY $M/train_consistency_lora.py --manifest $M/pairs_all_train.jsonl --output_dir $OUT \
    --lambda_consistency 0.0 --epochs 1 --lr 1e-4 --max_pixels 200704 --seed 42
echo "[$(date +%T)] eval lambda=0 adapter"
$PY $M/eval_lora.py --manifest $M/pairs_all_eval.jsonl --adapter $OUT --max_pixels 200704 \
    --dump $M/after_records_lambda0.jsonl
echo "[$(date +%T)] stats: base -> lambda0"
$PY $M/stats_lora.py --base $M/base_records_full.jsonl --after $M/after_records_lambda0.jsonl --B 5000
echo "[$(date +%T)] stats: lambda0 -> lambda1 (reported)"
$PY $M/stats_lora.py --base $M/after_records_lambda0.jsonl --after $M/after_records_full.jsonl --B 5000
echo "[$(date +%T)] DONE"
