---
language:
- en
license: cc-by-4.0
task_categories:
- visual-question-answering
- question-answering
tags:
- multimodal
- reasoning
- structured-data
- visualization
- benchmark
size_categories:
- 10K<n<100K
---

# Dataset Card for StructViz-Bench

## Dataset Summary

StructViz-Bench is a comprehensive benchmark for evaluating Multimodal Large Language Model (MLLM) reasoning over visualized structured data. It systematically measures **visualization sensitivity**—the degree to which a model's reasoning performance changes when the same underlying data is presented in different visual formats. The benchmark covers three primary modalities: tabular, time-series, and graph data, across 14 distinct visualization types.

## Supported Tasks

- **Visual Question Answering (VQA):** Answering questions based on visualized structured data.
- **Multimodal Reasoning:** Evaluating the model's ability to perform multi-hop and counterfactual reasoning over visual inputs.
- **Visualization Robustness Evaluation:** Measuring the consistency of model performance across different visual representations of the same data.

## Languages

The dataset is in English.

## Dataset Structure

### Data Instances

The benchmark consists of 3,795 base items, which are expanded into 18,315 rendered instances per model evaluation (depending on the number of visualization variants per item).

### Data Fields

- `question_id`: Unique identifier for the question.
- `question`: The natural language question.
- `answer`: The ground truth answer.
- `modality`: The data modality (tabular, timeseries, or graph).
- `source`: The source of the data (synthetic or real-world).
- `viz_type`: The specific visualization type used for the instance.
- `difficulty`: The reasoning depth (1-hop, 2-hop, 3-hop, or counterfactual).
- `task_type`: The type of reasoning task (e.g., extraction, comparison, trend).
- `image_path`: Path to the rendered visualization image.

### Data Splits

StructViz-Bench is an evaluation-only benchmark. It does not provide training or validation splits.

## Dataset Creation

### Curation Rationale

Existing benchmarks often focus on a single data type or a fixed visualization style. StructViz-Bench was created to address the gap in understanding how visualization choices impact MLLM reasoning consistency.

### Source Data

- **Synthetic:** Programmatically generated data from parameterized templates to ensure controlled complexity.
- **Real-world:**
    - Tabular: Derived from SciTabAlign.
    - Time Series: 840 items derived from ETT (Electricity Transformer Temperature) datasets, released as a rebuild script (see "ETT-derived items" below); plus 30 NetworkX degree-sequence time-series items (Karate Club, Les Misérables, Florentine Families; `data_id` `nx_*_degree`), which carry the label `source` `ett` in the released file and are not ETT data.
    - Graph: 4 real-world social network datasets from NetworkX (Karate Club, Les Miserables, Florentine Families, Davis Southern Women) plus 12 topology-template graphs generated from standard models (Watts-Strogatz, Barabasi-Albert, Caveman, Random Geometric). Each graph carries a `source_category` field (`real_world` or `topology_template`) to distinguish provenance.

### ETT-derived items

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

### Annotations

Annotations (questions and answers) are automated using quality-validated templates and programmatic solvers to ensure accuracy and scalability.

## Considerations for Using the Data

### Intended Use

- Benchmarking the multimodal reasoning capabilities of MLLMs.
- Researching visualization sensitivity and robustness in vision-language models.

### Limitations

- **Not intended for training:** This is an evaluation benchmark; using it for training may lead to overfitting on the specific templates and visualization styles.
- **Synthetic Bias:** While it includes real-world data, the synthetic generation component may not capture the full complexity and noise of all real-world visualization scenarios.

## Citation

```bibtex
@misc{structvizbench2026,
  title={StructViz-Bench: A Controlled Study of Visualization-Format Dependence in Multimodal LLM Reasoning over Structured Data},
  author={Anonymous},
  note={Under review},
  year={2026}
}
```
