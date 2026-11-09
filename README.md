# StructViz-Bench

A Unified Benchmark for Evaluating MLLM Reasoning over Visualized Structured Data

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

StructViz-Bench is a unified benchmark for systematically evaluating how visualization format affects MLLM reasoning across tabular, time-series, and graph data. By keeping underlying data and question semantics fixed while varying the visual representation, it enables direct measurement of visualization sensitivity across 14 visualization types (18,315 rendered instances per model; seven models in the paper).

## What's new for this version

- **Expressive-suite multi-model runs.** The corrected (v2) rendering suite is now also run with
  Qwen2.5-VL-32B and Qwen3-VL-8B on all questions, and with InternVL2-4B, InternVL2-8B and
  InternVL2.5-8B (6 tiles) on the 1,096-question strict subset. Predictions are in
  `results/review/v2_*.jsonl` and `results/review/v2sub_*.jsonl`, summarised by
  `scripts/analysis/v2_multimodel.py` and `v2_multimodel_normalized.py`.
- **Canonical scoring.** `scripts/analysis/scoring.py` defines `canonical_em`, a label-normalised
  exact match applied to every result file (`canonical_rescore.py`, `canonical_extras.py`; unit
  tests in `tests/test_scoring.py`). `python scripts/acl_numbers/recompute.py` recomputes every
  data-derived number of the paper under this metric into `scripts/acl_numbers/recomputed.json`.
- **Transcription experiment.** In transcribe-then-answer, Qwen2.5-VL-7B first transcribes each
  rendering to plain text and then answers from its own transcript alone
  (`results/review/tta_qwen.jsonl`, `scripts/analysis/tta_analysis.py`). Chain-of-thought
  re-runs are in `results/review/cot_*` (`scripts/analysis/cot_v2_audit.py`).
- **ChartQA external test (scripts only).** `scripts/review/fetch_chartqa.py` and
  `build_chartqa.py` download ChartQA and build the test items; `scripts/review/eval_lora_chartqa.py`,
  `scripts/analysis/chartqa_score.py` and `chartqa_lora_paired.py` score them. ChartQA is
  GPL-3.0, so its data, renderings and per-question predictions are not included; only the
  aggregate results (`scripts/analysis/chartqa_*.json`) are.
- **ETT rebuild.** The ETT-derived items ship without their series and answers, together with a
  script that rebuilds them from the original ETT files (see "ETT-derived items" below).
- **Further analyses** with their result JSONs in `scripts/analysis/`: error taxonomy, template
  difficulty, renderer-parameter sensitivity, the answer-absent bound (`h1_bound.py`) and a
  signal-detection view of the format gap (`h7_sdt.py`).

## Key Results

### Overall Leaderboard
| Model | Overall EM (%) | Tabular (%) | Time Series (%) | Graph (%) |
|---|---|---|---|---|
| Gemini Flash | 36.2 | 42.4 | 27.2 | 39.3 |
| GPT-4o | 34.4 | 41.4 | 21.8 | 43.6 |
| Qwen2.5-VL-7B | 32.4 | 37.7 | 24.3 | 35.3 |
| Claude Sonnet | 26.8 | 30.2 | 17.5 | 39.7 |

### Visualization Sensitivity Gaps
| Model | Tabular Gap (pp) | Time Series Gap (pp) | Graph Gap (pp) |
|---|---|---|---|
| Gemini Flash | 40.5 | 19.2 | 21.7 |
| GPT-4o | 36.7 | 25.7 | 23.0 |
| Qwen2.5-VL-7B | 37.1 | 13.6 | 14.8 |
| Claude Sonnet | 28.4 | 12.0 | 14.7 |


### Extension Experiments

| Experiment | Description | Status |
|---|---|---|
| Viz-Removal Ablation | Leave-one-out pooling analysis | Run, but not used in the paper (the pooled mean rises by construction when the worst format is removed) |
| Prompt Sensitivity | 4 prompt variants (concise, detailed, CoT, minimal) | Complete |
| Mixed-Type (Level 2) | Cross-modal pilot: one (table, series, graph) triple, 30 unique problems, 600 item ids (15–25 repeats each) | Pilot only; no significance claims |


## Benchmark Structure

```text
StructViz-Bench
├── Level 1: Single-Type
│   ├── Tabular      (value extraction, trend analysis, comparison)
│   ├── Time Series  (forecasting, anomaly detection, pattern classification)
│   └── Graph        (connectivity, shortest path, community)
├── Level 2: Mixed-Type
│   ├── Tabular + Time Series
│   ├── Tabular + Graph
│   └── Time Series + Graph
└── Level 3: Reasoning Depth
    ├── 1-hop
    ├── 2-hop
    ├── 3-hop
    └── Counterfactual
```

- **Total items:** 3,795
- **Rendered instances per model:** 18,315
- **Models in the paper:** 7 (GPT-4o, Gemini Flash, Gemini-2.5, Claude Sonnet, Qwen2.5-VL-7B/32B, InternVL2.5-8B); an eighth run (`results/full_llava.jsonl`, LLaVA-OneVision) is released but not used in the paper because it is degenerate: exact match is identical across all formats of every modality and the per-question flip rate is 0%, i.e. its outputs did not depend on the image

## Visualization Families

- **Tabular:** `bar_chart`, `heatmap`, `table_image`, `scatter_plot`, `text_only`
- **Time Series:** `line_plot`, `gaf`, `recurrence_plot`, `heatmap`, `text_only`
- **Graph:** `node_link`, `adjacency_matrix`, `circular_layout`, `text_only`

## Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd StructViz-Bench
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.

# 1. Generate benchmark data (synthetic only)
python scripts/generate_benchmark.py \
  --config configs/generation.yaml \
  --output benchmark/base_items.jsonl

# 1b. Include real-world datasets (requires data/external/)
python scripts/generate_benchmark.py \
  --config configs/generation.yaml \
  --output benchmark/realworld_test.jsonl \
  --include-realworld

# 2. Render visualizations
python scripts/render_all.py \
  --input benchmark/base_items.jsonl \
  --output-dir benchmark/rendered/

# 3. Run evaluation (single model, requires API keys)
python scripts/run_fullscale_eval.py \
  --model gpt4o \
  --benchmark benchmark/realworld_test.jsonl \
  --output-dir results/

# 4. Generate analysis report
python scripts/analyze_fullscale_results.py --results-dir results/

# 5. Ablation studies (offline recomputation from existing results)
python scripts/run_ablation.py viz-removal \
  --results-dir results/ --output-dir results/ablation/

# 5b. Prompt sensitivity (requires live model inference)
python scripts/run_ablation.py prompt-sensitivity \
  --model gpt4o --benchmark benchmark/realworld_test.jsonl

```

## Repository Layout

```
StructViz-Bench/
├── README.md
├── REPRODUCTION.md          # Step-by-step reproduction guide
├── DATASET_CARD.md          # HuggingFace dataset card
├── LICENSE                  # Apache-2.0 (code); CC-BY-4.0 (data)
├── requirements.txt
├── configs/                 # Experiment configurations
├── data/                    # Benchmark data (generated)
├── results/                 # Evaluation outputs
├── scripts/                 # All runnable scripts
├── src/                     # Source modules
└── tests/                   # Unit tests
```

## Citation

```bibtex
@misc{structvizbench2026,
  title={StructViz-Bench: A Controlled Study of Visualization-Format Dependence in Multimodal LLM Reasoning over Structured Data},
  author={Anonymous},
  note={Under review},
  year={2026}
}
```

## License

- **Code**: Apache-2.0
- **Benchmark data**: CC-BY-4.0, except the ETT-derived items (see below), which are not
  redistributed and must be rebuilt from the original ETT files

## ETT-derived items

The 840 time-series questions built from ETT windows (`source == "ett"`, `data_id` starting with
`ett_`) are released **without** their series and answer keys. ETT
([zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset)) is licensed CC BY-ND 4.0, which
does not permit sharing adapted material. The windowed series, the answers computed from them and
the two human-audit images that render them are therefore not distributed. In
`benchmark/realworld_test.jsonl` these rows keep their question, task, difficulty and metadata,
with `data` and `answer` set to `null` and `"ett_redacted": true`. In `results/` and the human-eval
files their `answer` / `ground_truth` is null or empty. Model predictions, raw responses and
stored scores (`exact_match`, `f1`, ...) are kept unchanged, except for model outputs that copy
ETT values. In `results/v2/rawtext_qwen.jsonl` the model was given the ETT series as text, so on
the ETT rows `prediction` and `raw_response` are null (`"prediction_redacted": true`). In
`results/review/tta_qwen.jsonl` the transcripts of ETT renderings and the answers given from
them (`transcript`, `raw_response`), and in `results/review/cot_*_qwen*.jsonl` the
chain-of-thought text (`raw_response` and the text fields derived from it), are null on the ETT
rows (`"model_output_redacted": true`). Their `exact_match` and other scores are kept. The
rebuild cannot regenerate these model outputs, because they are not derived from ETT by our
code, so they stay redacted after a restore. Re-running `tta_analysis.py` therefore cannot
recompute transcript fidelity for ETT time series; the shipped `tta_analysis_results.json` holds
those values.

To rebuild the items from the original ETT CSVs (fetched from ETDataset at commit `1d16c8f`,
sha256-checked) and restore every other redacted value byte-for-byte:

```bash
pip install -r requirements.txt
python scripts/data/rebuild_ett.py --download --restore .
```

Pass `--data-dir <path/to/ETT-small>` instead of `--download` to use a local copy. The script
regenerates the items with the benchmark's own generator and config, checks each one against the
hashes in `data/ett_redaction_manifest.json` (hashes only, no values), re-renders the removed
images and verifies every restored file against its original sha256 (files with redacted model
outputs against their original minus those fields).
`scripts/verify_paper_numbers.py` asks for this step when it finds redacted items. The 30 rows
labelled `ett` whose `data_id` starts with `nx_` are NetworkX degree sequences (Karate Club,
Les Misérables, Florentine Families), not ETT data, and ship in full.

## Corrected rendering suite (v2), no-image baseline, and calculation-aid experiment

The audit in the paper found that many v1 renderings do not encode the queried answer
(bar charts draw column means only, scatter plots two columns, table/text views the first
12/14 rows, graph drawings label nodes only below 30 nodes). `scripts/render_suite.py --suite v2`
re-renders every item so that this "absent" share drops to 21% / 36% / 0% (tables / series /
graphs; GAF and recurrence plots are kept as deliberately lossy transforms), and
`--suite assist` renders the 255 mean-comparison questions in a 2x2 (table / row-level bars,
with / without the column means). The v1 renderers are byte-identical
(`scripts/analysis/check_v1_unchanged.py`).

- Manifests with image sha256 and sizes: `benchmark/render_v2/manifest.jsonl`,
  `benchmark/render_assist/manifest.jsonl` (images regenerate deterministically; 1.6 GB, not shipped).
- Evaluation: `scripts/eval_local_suite.py` (Qwen2.5-VL-7B, revision cc594898, greedy, raw
  responses kept; `--no-image` sends the question alone with the image phrase removed from the
  system prompt); `scripts/run_v2_queue.sh` schedules the runs on shared GPUs;
  `scripts/merge_shards.py` merges and checks shards.
- Results: `results/v2/v2_qwen.jsonl` (18,975 rows), `results/v2/noimage_qwen.jsonl` (3,795),
  `results/v2/assist_qwen.jsonl` (1,020), `results/v2/rawtext_qwen.jsonl` (3,795; the same
  model given the data as text, `--raw-text`). A second open model, InternVL2.5-8B (revision
  e9e4c0dc, 448-px single tile), was run on the 1,096 strictly fully-answerable questions
  (`benchmark/render_v2/fully_answerable_qids.txt`, `manifest_subset32b.jsonl`):
  `results/v2/v2_internvl_subset.jsonl` (5,480), `noimage_internvl.jsonl` (1,096),
  `assist_internvl.jsonl` (1,020). Analysis: `scripts/analysis/v2_analysis.py`
  (`v2_analysis_output.txt`, `v2_results.json`; `--strict-answerable` writes
  `v2_results_strict.json`, `--model internvl` writes `v2_results_internvl_strict.json`);
  `verify_paper_numbers.py` checks the paper's Section 5(iv) numbers against these files.
- Human audits: `human_eval_package/` (100 v1 renderings, two raters) and `human_eval_package_v2/`
  (30 corrected renderings re-judged by the same two raters; 22 repeat v1 items). Each holds the
  items, images, returned sheets (notes withheld until de-anonymisation), a recomputation of every
  disputed key (`analysis/`), and the aggregation scripts; the harness recomputes the appendix counts.
