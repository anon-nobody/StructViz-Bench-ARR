#!/usr/bin/env bash
# GPU queue for the review-revision runs (claim/release pattern of run_v2_queue.sh).
# Workers on GPUs 0, 2, 3 (never GPU 1) claim tasks in priority order, wait until their GPU has
# enough free memory, run the resumable job, and release the claim on failure so a free worker
# retries. A task counts as done only if its log ends with the job's own completion line.
# ADOPT="gpu:pid:task ..." lets a restarted queue wait for jobs launched by a previous queue
# instance and finalize them (used when the task order was changed mid-run).
cd "$(dirname "$0")/.."
PY=.venv/bin/python
R=results/review; Q=$R/queue; L=$R/logs; C=$R/chartqa; mkdir -p "$Q" "$L"
V2=benchmark/render_v2/manifest.jsonl
SUB=benchmark/render_v2/manifest_subset32b.jsonl
EV="$PY scripts/eval_local_suite.py"
# Priority (team-lead, 2026-11-06): 32B shards, Qwen3-VL v2, ChartQA (7B, Qwen3-VL), LoRA ChartQA
# transfer, CoT v2 (+ system-prompt-only), InternVL tiles, reduced TTA sample, CoT v1 audit last.
TASKS=(q32s0 q32s1 q32s2 q3vl cq_qwen cq_qwen3vl8b
       lora_base_original lora_lambda0_original lora_lambda1_original
       lora_base_all lora_lambda0_all lora_lambda1_all
       cotv2 cotv2_sys ivl2_4b ivl2_8b ivl25_t6 tta0 tta1 tta2 cotv1)
TTA_MAN=$R/tta_manifest.jsonl   # 50 fully-answerable questions per modality x 5 formats
TTA_CAP=2560                    # stage-1 cap: fits a 200-point series as i,v lines (<=2426 tok)
# group name -> member tasks; queue/group_<name>.done is written when all members are done
declare -A GROUPS_=([q32_merge]="q32s0 q32s1 q32s2" [qwen3vl_v2]="q3vl"
  [chartqa_lora]="cq_qwen cq_qwen3vl8b lora_base_original lora_lambda0_original lora_lambda1_original lora_base_all lora_lambda0_all lora_lambda1_all"
  [cot_v2]="cotv2 cotv2_sys" [internvl]="ivl2_4b ivl2_8b ivl25_t6" [tta]="tta0 tta1 tta2" [cot_v1]="cotv1")
CHARTQA_TASKS=(cq_qwen cq_qwen3vl8b lora_base_original lora_lambda0_original lora_lambda1_original
               lora_base_all lora_lambda0_all lora_lambda1_all)

cmd_for() {  # full command; GPU_ID is substituted by the worker
  case "$1" in
    q32s*) i=${1#q32s}; echo "$EV --model qwen32b-hf --manifest $V2 --suite v2 --shard $i/3 --output $R/v2_qwen32b_shard$i.jsonl --log $L/$1.log --gpu GPU_ID";;
    q3vl)  echo "env PYTHONPATH=.tf_new $EV --model qwen3vl8b --manifest $V2 --suite v2 --output $R/v2_qwen3vl8b.jsonl --log $L/$1.log --gpu GPU_ID";;
    cq_qwen) echo "$EV --model qwen --manifest $C/manifest_chartqa.jsonl --benchmark $C/chartqa_items.jsonl --suite chartqa --output $C/pred_qwen_chartqa.jsonl --log $L/$1.log --gpu GPU_ID";;
    cq_qwen3vl8b) echo "env PYTHONPATH=.tf_new $EV --model qwen3vl8b --manifest $C/manifest_chartqa.jsonl --benchmark $C/chartqa_items.jsonl --suite chartqa --output $C/pred_qwen3vl8b_chartqa.jsonl --log $L/$1.log --gpu GPU_ID";;
    lora_*) rest=${1#lora_}; setting=${rest%_*}; name=${rest##*_}
            case $setting in base) ad="";; lambda0) ad="--adapter checkpoints/consistency_lora_mm_lambda0";; lambda1) ad="--adapter checkpoints/consistency_lora_mm";; esac
            case $name in original) man=$C/manifest_chartqa_original.jsonl;; all) man=$C/manifest_chartqa.jsonl;; esac
            # Same arguments as scripts/review/run_lora_chartqa.sh (that wrapper is not called: its
            # final scorer call would write into paper_tables/).
            echo "env HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY scripts/review/eval_lora_chartqa.py --manifest $man --benchmark $C/chartqa_items.jsonl --setting $setting $ad --max_pixels 200704 --min_pixels 50176 --max_new_tokens 16 --output $C/pred_lora_${setting}_${name}.jsonl --gpu GPU_ID";;
    tta*)  i=${1#tta}; echo "$EV --model qwen --transcribe-then-answer --transcribe-max-new-tokens $TTA_CAP --manifest $TTA_MAN --suite v2_subset_tta_sample --shard $i/3 --output $R/tta_qwen_shard$i.jsonl --log $L/$1.log --gpu GPU_ID";;
    cotv2) echo "$EV --model qwen --cot --cot-user-suffix --manifest $R/cot_v2_manifest.jsonl --suite v2_cot --output $R/cot_v2_qwen.jsonl --log $L/$1.log --gpu GPU_ID";;
    cotv2_sys) echo "$EV --model qwen --cot --manifest $R/cot_v2_manifest.jsonl --suite v2_cot_sysonly --output $R/cot_v2_qwen_sysprompt_only.jsonl --log $L/$1.log --gpu GPU_ID";;
    cotv1) echo "$EV --model qwen --cot --cot-prompt ablation_v1 --manifest $R/cot_v1_manifest.jsonl --suite v1_ablation_cot --output $R/cot_v1_qwen.jsonl --log $L/$1.log --gpu GPU_ID";;
    ivl2_4b)  echo "$EV --model internvl2-4b --internvl-max-tiles 6 --manifest $SUB --suite v2_subset --output $R/v2sub_internvl2_4b.jsonl --log $L/$1.log --gpu GPU_ID";;
    ivl2_8b)  echo "$EV --model internvl2-8b --internvl-max-tiles 6 --manifest $SUB --suite v2_subset --output $R/v2sub_internvl2_8b.jsonl --log $L/$1.log --gpu GPU_ID";;
    ivl25_t6) echo "$EV --model internvl --internvl-max-tiles 6 --manifest $SUB --suite v2_subset --output $R/v2sub_internvl25_8b_tiles6.jsonl --log $L/$1.log --gpu GPU_ID";;
  esac
}
need_mb() { case "$1" in q32s*) echo 72000;; ivl*) echo 14000;; *) echo 24000;; esac; }
finished_ok() {  # the job's own completion line must be at the end of its log
  case "$1" in
    lora_*) tail -n 3 "$L/$1.log" | grep -qE "^DONE |pending=0 " ;;
    *)      tail -n 3 "$L/$1.log" | grep -qE "\] DONE: |Nothing to do" ;;
  esac
}
merge_if_complete() {  # $1 shard prefix, $2 merged output, $3 manifest, $4 shard file prefix
  for i in 0 1 2; do [ -e "$Q/$1$i.done" ] || return 0; done
  mkdir "$Q/merge_$1.lock" 2>/dev/null || return 0
  $PY scripts/merge_shards.py $(for i in 0 1 2; do echo "$4$i.jsonl"; done) --output "$2" \
      --manifest "$3" > "$L/merge_$1.log" 2>&1 && touch "$Q/merge_$1.done"
}
score_chartqa_if_complete() {
  for t in "${CHARTQA_TASKS[@]}"; do [ -e "$Q/$t.done" ] || return 0; done
  mkdir "$Q/chartqa_score.lock" 2>/dev/null || return 0
  $PY scripts/analysis/chartqa_score.py --out-tex "$C/tab_chartqa.tex" > "$L/chartqa_score.log" 2>&1 \
      && touch "$Q/chartqa_score.done"
}
groups_check() {  # group marker = all members done (+ merge / scoring where applicable)
  for g in "${!GROUPS_[@]}"; do
    [ -e "$Q/group_$g.done" ] && continue
    ok=1; for t in ${GROUPS_[$g]}; do [ -e "$Q/$t.done" ] || ok=0; done
    case $g in q32_merge) [ -e "$Q/merge_q32s.done" ] || ok=0;; tta) [ -e "$Q/merge_tta.done" ] || ok=0;;
               chartqa_lora) [ -e "$Q/chartqa_score.done" ] || ok=0;; esac
    [ $ok = 1 ] && date -u +%FT%H:%M:%S > "$Q/group_$g.done"
  done
}
finalize() {  # $1 task, $2 gpu
  if finished_ok "$1"; then
    touch "$Q/$1.done"; echo "[gpu$2] $(date -u +%FT%H:%M) done $1"
    merge_if_complete q32s "$R/v2_qwen32b.jsonl" "$V2" "$R/v2_qwen32b_shard"
    merge_if_complete tta "$R/tta_qwen.jsonl" "$TTA_MAN" "$R/tta_qwen_shard"
    score_chartqa_if_complete
    groups_check
  else
    echo "[gpu$2] $(date -u +%FT%H:%M) FAILED $1 (claim released for retry)"
    rm -f "$Q/$1.claim/gpu" "$Q/$1.claim/start"; rmdir "$Q/$1.claim"; sleep 120
  fi
}
worker() {
  gpu=$1
  for a in ${ADOPT:-}; do
    IFS=: read -r ag apid atask <<< "$a"
    [ "$ag" = "$gpu" ] || continue
    echo "[gpu$gpu] $(date -u +%FT%H:%M) adopted $atask (pid $apid); waiting for it"
    while kill -0 "$apid" 2>/dev/null; do sleep 60; done
    finalize "$atask" "$gpu"
  done
  while true; do
    task=""
    for t in "${TASKS[@]}"; do
      [ -e "$Q/$t.done" ] && continue
      if mkdir "$Q/$t.claim" 2>/dev/null; then task=$t; break; fi
    done
    [ -z "$task" ] && { echo "[gpu$gpu] $(date -u +%FT%H:%M) no tasks left"; return; }
    need=$(need_mb "$task")
    while true; do
      total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$gpu")
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")
      free=$(( total - used ))
      [ "$free" -ge "$need" ] && break
      echo "[gpu$gpu] $(date -u +%FT%H:%M) waiting for $task: free ${free}MB < ${need}MB"
      sleep 120
    done
    echo "$gpu" > "$Q/$task.claim/gpu"; date -u +%FT%H:%M:%S > "$Q/$task.claim/start"
    echo "[gpu$gpu] $(date -u +%FT%H:%M) start $task (free ${free}MB)"
    cmd=$(cmd_for "$task"); cmd=${cmd//GPU_ID/$gpu}
    case "$task" in
      lora_*) $cmd >> "$L/$task.log" 2>&1 ;;
      *)      $cmd >> "$L/worker_gpu$gpu.log" 2>&1 ;;
    esac
    finalize "$task" "$gpu"
  done
}
for g in 0 2 3; do worker "$g" & done
wait
echo "ALL TASKS DONE $(date -u)"; touch "$Q/ALL_DONE"
