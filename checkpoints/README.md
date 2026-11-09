# Consistency-regularized LoRA

`consistency_lora_mm/adapter_config.json` is the exact PEFT configuration of the adapter
reported in the paper (jointly trained on tabular + time-series + graph).

The adapter weights (`adapter_model.safetensors`, ~190 MB) exceed the size limit of this
anonymized host and are therefore **not** included here. They are reproducible from the
released code and this config:

```bash
python scripts/mitigation/train_consistency_lora.py \
    --pairs scripts/mitigation/pairs_all_train.jsonl \
    --max_pixels 200704 --epochs 1
python scripts/mitigation/eval_lora.py  --dump records.jsonl
python scripts/mitigation/stats_lora.py --records records.jsonl   # paired bootstrap, B=5000
```

Weights will be published with the camera-ready version.

## λ=0 control

`consistency_lora_mm_lambda0/adapter_config.json` is the configuration of the control run
reported alongside it in the paper: identical data, schedule, and seed, with
`--lambda_consistency 0`. Reproduce both, then compare, with:

```bash
scripts/mitigation/run_lambda0_control.sh
```

The per-question records for base, λ=0 and λ=1 (`scripts/mitigation/*_records_*.jsonl`) are
released so the paired statistics in the paper can be recomputed without a GPU:

```bash
python scripts/mitigation/stats_lora.py --base scripts/mitigation/after_records_lambda0.jsonl \
    --after scripts/mitigation/after_records_full.jsonl --B 5000
```
