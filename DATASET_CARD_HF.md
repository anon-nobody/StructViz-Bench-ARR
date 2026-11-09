---
language:
- en
license: cc-by-4.0
task_categories:
- visual-question-answering
- image-text-to-text
tags:
- mllm
- visualization
- benchmark
- structured-data
- multimodal
size_categories:
- 1K<n<10K
dataset_info:
  features:
  - name: question_id
    dtype: string
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: modality
    dtype: string
  - name: source
    dtype: string
  - name: viz_type
    dtype: string
  - name: difficulty
    dtype: string
  - name: task
    dtype: string
  - name: data_id
    dtype: string
  - name: data
    dtype: string
  - name: metadata
    dtype: string
  splits:
  - name: base_items
    num_examples: 1500
  - name: mixed_items
    num_examples: 600
  - name: realworld_test
    num_examples: 3795
  config_name: default
---

# Dataset Card for StructViz-Bench

## Dataset Description

- **Homepage:** withheld for anonymous review; all files are in this repository
- **Repository (anonymized for review):** [https://anonymous.4open.science/r/StructViz-Bench](https://anonymous.4open.science/r/StructViz-Bench)
- **Paper:** StructViz-Bench: A Unified Benchmark for Evaluating MLLM Reasoning over Visualized Structured Data
- **Point of Contact:** Anonymous
- **License:** CC-BY-4.0

### Dataset Summary

StructViz-Bench is a benchmark for evaluating Multimodal Large Language Model (MLLM) reasoning over visualized structured data. It systematically measures **visualization sensitivity** -- the degree to which a model's reasoning performance changes when the same underlying data is presented in different visual formats.

The benchmark encompasses three primary data modalities (tabular, time-series, and graph), 14 distinct visualization types, 4 question difficulty levels (1-hop, 2-hop, 3-hop, and counterfactual), and a diverse set of reasoning tasks. It contains 3,795 base items, which expand to 18,315 rendered instances per model. Released evaluations cover seven models (GPT-4o, Gemini Flash, Gemini-2.5, Claude Sonnet, Qwen2.5-VL-7B/32B, InternVL2.5-8B), 18,315 rendered instances each, plus the corrected v2 suite, a no-image baseline and a calculation-aid experiment on Qwen2.5-VL-7B.

### Supported Tasks

- **Visual Question Answering (VQA):** Answering natural language questions grounded in visualized structured data, requiring extraction, comparison, aggregation, and multi-step reasoning.
- **Multimodal Reasoning:** Evaluating the capacity of models to perform multi-hop inference and counterfactual reasoning over visual inputs derived from structured data.
- **Visualization Robustness Evaluation:** Quantifying the consistency of model performance across different visual representations of identical underlying data.
- **Cross-Modal Reasoning:** Assessing model performance on composite items that require simultaneous reasoning over two distinct data modalities (e.g., tabular + time-series, tabular + graph, time-series + graph).

### Languages

All questions, answers, and metadata are in English (en).

## Dataset Structure

### Data Instances

A representative example from `base_items.jsonl`:

```json
{
  "question_id": "tabular_000003::difficulty=1-hop",
  "question": "Which 'ticker' has the highest 'close_price' value?",
  "answer": "BET",
  "modality": "tabular",
  "data_id": "finance_stock_prices",
  "task": "ranking",
  "difficulty": "1-hop",
  "viz_methods": ["bar_chart", "heatmap", "table_image", "scatter_plot", "text_only"],
  "metadata": {"domain": "finance", "description": "Monthly stock activity for a set of listed companies."},
  "data_format": "dataframe_records",
  "data": [ ... ],
  "data_columns": ["ticker", "period", "close_price", "trade_volume_k", "daily_return_pct", "market_segment"]
}
```

### Data Fields

| Field | Type | Description |
|---|---|---|
| `question_id` | string | Unique identifier encoding modality, item index, and difficulty level. |
| `question` | string | The natural language question posed to the model. |
| `answer` | string | The ground-truth answer, generated via programmatic solvers. |
| `modality` | string | The data modality: `tabular`, `timeseries`, `graph`, `mixed_tab_ts`, `mixed_tab_graph`, or `mixed_ts_graph`. |
| `source` | string | Provenance of the underlying data: `synthetic`, `scitabalign`, `ett`, or `networkx_realworld`. `ett` labels the 840 ETT-derived time-series items and also 30 NetworkX degree-sequence time-series items (Karate Club, Les Misérables, Florentine Families; `data_id` `nx_*_degree`), which are not ETT data; the label is kept as released for reproducibility. |
| `viz_methods` | list[string] | Applicable visualization types for the item (e.g., `bar_chart`, `heatmap`, `line_plot`, `node_link`, `text_only`). |
| `difficulty` | string | Reasoning depth: `1-hop`, `2-hop`, `3-hop`, or `counterfactual`. |
| `task` | string | The reasoning task category (e.g., `value_extraction`, `ranking`, `comparison`, `trend_analysis`, `aggregation`, `filtering`, `correlation`, `forecasting`, `peak_identification`, `pattern_classification`, `anomaly_detection`, `connectivity`, `shortest_path`, `community`, `degree_query`). |
| `data_id` | string | Identifier for the underlying data instance. |
| `data` | object | The raw structured data used to generate the visualization and formulate the question. |
| `metadata` | object | Domain-specific metadata including domain name, description, and column or feature descriptions. |
| `data_format` | string | Format descriptor for the data field (e.g., `dataframe_records`, `json_dict`). |
| `data_columns` | list[string] | Column names for tabular data (when applicable). |

### Data Splits

StructViz-Bench is an **evaluation-only** benchmark and does not provide training or validation splits. The data is organized into three files:

| File | Items | Description |
|---|---|---|
| `base_items.jsonl` | 1,500 | Synthetic-only single-modality items (500 tabular + 500 time-series + 500 graph). |
| `realworld_test.jsonl` | 3,795 | Full benchmark: all 1,500 synthetic items plus 2,295 real-world items (1,265 SciTabAlign tabular + 840 ETT-derived time-series (released as a rebuild script) + 30 NetworkX degree-sequence time-series labelled source 'ett' + 160 NetworkX graph). This is the primary evaluation file used for all paper results. |
| `mixed_items.jsonl` | 600 | Cross-modal composite items (200 tab+ts, 200 tab+graph, 200 ts+graph). |

Note: `realworld_test.jsonl` is a superset of `base_items.jsonl`. The name reflects that it includes real-world sources; use the `source` field (`synthetic`, `scitabalign`, `ett`, `networkx_realworld`) to filter by provenance. To separate ETT-derived items from the 30 NetworkX degree-sequence items that also carry `source` `ett`, use `data_id` (`ett_*` vs. `nx_*_degree`).

When expanded across all applicable visualization types, the base items yield 18,315 rendered instances per model. Seven models were evaluated on the released suite.

## Dataset Creation

### Curation Rationale

Existing multimodal benchmarks typically focus on a single data type (e.g., charts or tables) or employ a fixed visualization style per data instance. This design conflates the difficulty of the underlying reasoning task with the difficulty imposed by the specific visual encoding. StructViz-Bench was created to fill this gap by decoupling data content from visual representation, thereby enabling controlled measurement of visualization sensitivity in MLLM reasoning.

### Source Data

The benchmark draws on both synthetic and real-world data sources:

- **Synthetic data:** Programmatically generated from parameterized templates across multiple domains (finance, healthcare, environmental science, etc.) to ensure controlled complexity and balanced coverage of reasoning tasks.
- **Real-world tabular data:** Derived from SciTabAlign, a collection of scientific tables with structured annotations.
- **Real-world time-series data:** 840 items derived from the ETT (Electricity Transformer Temperature) dataset, containing multivariate sensor readings, released as a rebuild script; plus 30 NetworkX degree-sequence time-series items (Karate Club, Les Misérables, Florentine Families; `data_id` `nx_*_degree`), which carry the label `source` `ett` in the released file and are not ETT data. These items are released redacted (CC BY-ND 4.0); see "ETT-derived items" under Additional Information.
- **Real-world graph data:** Classic and empirical graph structures generated via NetworkX, including social networks and infrastructure topologies.

### Annotations

All questions and ground-truth answers are generated programmatically using quality-validated templates and deterministic solvers. This approach ensures:

- Correctness: Answers are computed directly from the underlying data, eliminating human annotation errors.
- Scalability: The template-based system supports systematic generation across modalities, difficulty levels, and task types.
- Reproducibility: Given the same data and configuration, the benchmark can be regenerated identically.

### Annotation Process

No manual annotation was performed. The question-answer generation pipeline consists of:

1. Data instantiation from parameterized templates or real-world sources.
2. Question generation via modality-specific and task-specific template functions.
3. Answer computation using deterministic programmatic solvers.
4. Quality validation through automated consistency checks and unit tests.

## Considerations for Using the Data

### Social Impact

StructViz-Bench is intended to advance the scientific understanding of MLLM capabilities and limitations in interpreting visualized structured data. It may contribute to improved model robustness and more reliable deployment of vision-language systems in data analysis workflows.

### Biases

- **Domain coverage:** While the benchmark spans multiple domains (finance, healthcare, environmental science, infrastructure), it does not exhaustively represent all possible application domains.
- **Visualization design:** The 14 visualization types selected represent common and widely-used formats but do not cover all possible visual encodings (e.g., 3D visualizations, interactive dashboards).
- **Language:** All content is in English, limiting direct applicability to multilingual evaluation settings.

### Limitations

- **Evaluation-only:** This benchmark is designed exclusively for evaluation. Using it for model training may lead to overfitting on the specific templates and visualization styles employed.
- **Synthetic component:** Although real-world data sources are included, the synthetic generation component may not fully capture the noise, irregularities, and complexity of all real-world visualization scenarios.
- **Static visualizations:** The benchmark evaluates static rendered images only and does not address interactive or dynamic visualization modalities.
- **Answer format:** Ground-truth answers are expressed as short strings (exact match evaluation), which may not capture the full spectrum of valid natural language responses.

## Additional Information

### Licensing Information

This dataset is released under the [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/).

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

### Citation Information

```bibtex
@misc{structvizbench2026,
  title={StructViz-Bench: A Controlled Study of Visualization-Format Dependence in Multimodal LLM Reasoning over Structured Data},
  author={Anonymous},
  note={Under review},
  year={2026}
}
```

### Dataset Statistics

| Statistic | Value |
|---|---|
| Total base items | 3,795 |
| Single-modality items (base_items) | 1,500 |
| Cross-modal items (mixed_items) | 600 |
| Real-world items (realworld_test) | 3,795 |
| Visualization types | 14 |
| Data modalities | 3 (tabular, timeseries, graph) |
| Difficulty levels | 4 (1-hop, 2-hop, 3-hop, counterfactual) |
| Rendered instances per model | 18,315 |
| Rendered instances per model | 18,315 (seven models reported in the paper) |
| Task categories | 49 |

### Visualization Types

**Tabular modality (5 types):**
- `bar_chart`, `heatmap`, `table_image`, `scatter_plot`, `text_only`

**Time-series modality (5 types):**
- `line_plot`, `gaf` (Gramian Angular Field), `recurrence_plot`, `heatmap`, `text_only`

**Graph modality (4 types):**
- `node_link`, `adjacency_matrix`, `circular_layout`, `text_only`

### Contact

For questions or issues regarding this dataset, please open an issue in the associated repository.
