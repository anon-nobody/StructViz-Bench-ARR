#!/usr/bin/env bash
# ACL review round 3: multi-format LoRA on an object-disjoint split, three seeds, with and
# without the consistency term, plus the base model on the same evaluation objects.
# Usage: GPU=<id> bash scripts/mitigation/run_r3_gpu.sh   (needs ~40 GB free on that GPU)
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${GPU:?set GPU}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python; M=scripts/mitigation; O=$M/r3_objsplit; mkdir -p $O
$PY $M/build_object_split.py
echo "[$(date +%T)] base"
$PY $M/eval_lora.py --manifest $M/pairs_objsplit_eval.jsonl --max_pixels 200704 --dump $O/base.jsonl
for LAM in 0.0 1.0; do for SEED in 1 2 3; do
  CK=checkpoints/r3_objsplit_l${LAM}_s${SEED}
  echo "[$(date +%T)] train lambda=$LAM seed=$SEED"
  $PY $M/train_consistency_lora.py --manifest $M/pairs_objsplit_train.jsonl --output_dir $CK \
      --lambda_consistency $LAM --epochs 1 --lr 1e-4 --max_pixels 200704 --seed $SEED
  $PY $M/eval_lora.py --manifest $M/pairs_objsplit_eval.jsonl --adapter $CK --max_pixels 200704 \
      --dump $O/l${LAM}_s${SEED}.jsonl
  $PY $M/stats_lora.py --base $O/base.jsonl --after $O/l${LAM}_s${SEED}.jsonl --B 5000 \
      > $O/stats_l${LAM}_s${SEED}.txt
done; done
echo "[$(date +%T)] DONE"
