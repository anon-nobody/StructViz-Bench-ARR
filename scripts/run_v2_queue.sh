#!/usr/bin/env bash
# Queue-based launcher for the v2 / assist / no-image Qwen runs on shared GPUs.
# Each GPU worker claims the next unclaimed task, waits until its GPU has >= 22 GB free,
# runs the (resumable) evaluation, and releases the claim on failure so a free worker retries.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
Q=results/v2/queue; mkdir -p "$Q" results/v2/logs
TASKS=(noimage assist v2s0 v2s1 v2s2 v2s3)
cmd_for() {
  case "$1" in
    noimage) echo "$PY scripts/eval_local_suite.py --model qwen --no-image --output results/v2/noimage_qwen.jsonl --suite no_image --log results/v2/logs/noimage.log";;
    assist)  echo "$PY scripts/eval_local_suite.py --model qwen --manifest benchmark/render_assist/manifest.jsonl --suite assist --output results/v2/assist_qwen.jsonl --log results/v2/logs/assist.log";;
    v2s*)    i=${1#v2s}; echo "$PY scripts/eval_local_suite.py --model qwen --manifest benchmark/render_v2/manifest.jsonl --suite v2 --shard $i/4 --output results/v2/v2_qwen_shard$i.jsonl --log results/v2/logs/v2_shard$i.log";;
  esac
}
worker() {
  gpu=$1
  while true; do
    task=""
    for t in "${TASKS[@]}"; do
      [ -e "$Q/$t.done" ] && continue
      if mkdir "$Q/$t.claim" 2>/dev/null; then task=$t; break; fi
    done
    [ -z "$task" ] && { echo "[gpu$gpu] $(date -u +%H:%M) no tasks left"; return; }
    while true; do
      total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$gpu")
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")
      free=$(( total - used ))
      [ "$free" -ge 20000 ] && break
      sleep 60
    done
    echo "[gpu$gpu] $(date -u +%H:%M) start $task (free ${free}MB)"
    if $(cmd_for "$task") --gpu "$gpu" >> "results/v2/logs/worker_gpu$gpu.log" 2>&1; then
      touch "$Q/$task.done"; echo "[gpu$gpu] $(date -u +%H:%M) done $task"
    else
      echo "[gpu$gpu] $(date -u +%H:%M) FAILED $task (claim released for retry)"
      rmdir "$Q/$task.claim"; sleep 120
    fi
  done
}
for g in 0 2 1 3; do worker "$g" & done
wait
echo "ALL TASKS DONE $(date -u)"
