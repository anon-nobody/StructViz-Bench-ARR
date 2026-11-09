#!/usr/bin/env bash
# ChartQA transfer test of the paper's multi-format LoRA (Table tab:lora / App. app:lora).
# base Qwen2.5-VL-7B, lambda=0 adapter (plain multi-format SFT), lambda=1 adapter (reported,
# consistency-regularised), each on the ORIGINAL ChartQA charts and on all 6 renderings,
# at the training resolution (max_pixels=200704; min_pixels=50176 as in eval_lora.py),
# greedy, 16 new tokens. Resumable (rows already in an output file are skipped).
#   GPU=0 bash scripts/review/run_lora_chartqa.sh            # run
#   DRY=1 bash scripts/review/run_lora_chartqa.sh            # parse args + join only (CPU)
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python; D=results/review/chartqa
GPU_ARGS=(); [[ -n "${GPU:-}" ]] && GPU_ARGS=(--gpu "$GPU")
DRY_ARGS=(); [[ -n "${DRY:-}" ]] && DRY_ARGS=(--dry_run)
declare -A ADAPTER=([base]="" [lambda0]=checkpoints/consistency_lora_mm_lambda0 \
                    [lambda1]=checkpoints/consistency_lora_mm)
declare -A MANIFEST=([original]=$D/manifest_chartqa_original.jsonl [all]=$D/manifest_chartqa.jsonl)
for name in original all; do
  for setting in base lambda0 lambda1; do
    AD=(); [[ -n "${ADAPTER[$setting]}" ]] && AD=(--adapter "${ADAPTER[$setting]}")
    echo "[$(date +%T)] $setting / $name"
    $PY scripts/review/eval_lora_chartqa.py --manifest "${MANIFEST[$name]}" \
        --benchmark $D/chartqa_items.jsonl --setting "$setting" "${AD[@]}" \
        --max_pixels 200704 --min_pixels 50176 --max_new_tokens 16 \
        --output "$D/pred_lora_${setting}_${name}.jsonl" "${GPU_ARGS[@]}" "${DRY_ARGS[@]}"
  done
done
[[ -z "${DRY:-}" ]] && $PY scripts/analysis/chartqa_score.py
echo "[$(date +%T)] DONE"
