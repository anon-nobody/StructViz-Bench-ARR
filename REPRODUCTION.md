# Reproduction Guide for StructViz-Bench

This guide provides step-by-step instructions to reproduce the full-scale evaluation results for StructViz-Bench.

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (required for local inference with Qwen2.5-VL-7B)
- API keys for proprietary models:
    - OpenAI (GPT-4o)
    - Anthropic (Claude Sonnet)
    - Google AI (Gemini Flash)

## Environment Setup

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set PYTHONPATH (required for all scripts):
```bash
export PYTHONPATH=.
```

4. Configure API keys as environment variables:
```bash
export OPENAI_API_KEY="your_key_here"
export ANTHROPIC_API_KEY="your_key_here"
export GEMINI_API_KEY="your_key_here"
```

## Step 1: Data Generation

Generate the base benchmark items from structured data sources:

```bash
# Synthetic only (1,500 items)
python scripts/generate_benchmark.py \
  --config configs/generation.yaml \
  --output benchmark/base_items.jsonl

# Full benchmark including real-world datasets (3,795 items)
python scripts/generate_benchmark.py \
  --config configs/generation.yaml \
  --output benchmark/realworld_test.jsonl \
  --include-realworld
```

- **Expected Output:** `benchmark/realworld_test.jsonl` with 3,795 items.
- **Validation:** Verify the counts per modality:
    - Tabular: 1,765 (500 synthetic + 1,265 SciTabAlign)
    - Time Series: 1,370 (500 synthetic + 870 with `source` `ett`: 840 ETT-derived + 30 NetworkX degree sequences, `data_id` `nx_*_degree`, which are not ETT data)
    - Graph: 660 (500 synthetic + 160 NetworkX)

## Step 2: Visualization Rendering

Render each benchmark item into all applicable visualization styles:

```bash
python scripts/render_all.py \
  --input benchmark/realworld_test.jsonl \
  --output-dir benchmark/rendered/
```

- **Expected Output:** Rendered images stored in `benchmark/rendered/`.

## Step 3: Model Evaluation

Run the evaluation for each model using the `scripts/run_fullscale_eval.py` script.

```bash
# Evaluate GPT-4o
python scripts/run_fullscale_eval.py --model gpt4o \
  --benchmark benchmark/realworld_test.jsonl --output-dir results/

# Evaluate Gemini Flash
python scripts/run_fullscale_eval.py --model gemini \
  --benchmark benchmark/realworld_test.jsonl --output-dir results/

# Evaluate Claude Sonnet
python scripts/run_fullscale_eval.py --model claude \
  --benchmark benchmark/realworld_test.jsonl --output-dir results/

# Evaluate Qwen2.5-VL-7B
python scripts/run_fullscale_eval.py --model qwen \
  --benchmark benchmark/realworld_test.jsonl --output-dir results/
```

- **Note:** Each model evaluation produces approximately 18,315 rows in a JSONL file.
- **Resumability:** The script supports checkpointing; if interrupted, it will resume from the last processed item.
- **Costs:** Be aware of API costs associated with running full-scale evaluations on proprietary models.

## Step 4: Analysis

Aggregate the evaluation results and generate performance tables:

```bash
python scripts/analyze_fullscale_results.py --results-dir results/
```

- **Expected Output:** `results/full_analysis.md` and `results/full_tables.txt`.

## Step 5: Figure Generation

Generate the figures used in the paper:

```bash
python scripts/generate_paper_figures.py --split full --results-dir results/ --output-dir paper/figures_full/
```

- **Expected Output:** 6 PDF figures generated in `paper/figures_full/`.


## Step 6: Ablation Studies

Run the visualization removal and prompt sensitivity ablation studies:

```bash
# Viz-removal ablation (leave-one-out)
python scripts/run_ablation.py viz-removal

# Prompt sensitivity ablation
python scripts/run_ablation.py prompt-sensitivity --model gpt4o
python scripts/run_ablation.py prompt-sensitivity --model gemini
python scripts/run_ablation.py prompt-sensitivity --model claude
python scripts/run_ablation.py prompt-sensitivity --model qwen
```

- **Expected Output:** `results/ablation/viz_removal_summary.csv`, `results/ablation/prompt_sensitivity_summary.csv`

## Step 7: Mixed-Type Evaluation

Generate and evaluate mixed-type benchmark items:

```bash
# Generate mixed-type benchmark items
python scripts/generate_mixed_benchmark.py

# Run mixed-type evaluation
python scripts/run_mixed_eval.py --model gpt4o
python scripts/run_mixed_eval.py --model gemini
python scripts/run_mixed_eval.py --model claude
```

- **Expected Output:** `results/mixed_{model}.jsonl`, `results/mixed_analysis_summary.csv`

## Step 8: Extension Analysis

Analyze the extension experiment results:

```bash
python scripts/analyze_extensions.py prompt
python scripts/analyze_extensions.py mixed
```

- **Expected Output:** Summary CSVs, LaTeX tables, and comparison figures in `results/ablation/` and `results/`

## Troubleshooting

- **API Rate Limits:** If you encounter rate limit errors, use the `--rpm` flag to limit requests per minute.
- **Interrupted Runs:** All evaluation scripts are resumable via JSONL checkpointing. Simply restart the command.

- **Extension Scripts:** Ablation and mixed-type evaluation scripts support checkpointing and resumability.
- **Claude Instability:** For specific errors encountered during Claude evaluation, use `scripts/retry_claude_errors.py` for targeted retries of failed instances.
