# StructViz-Bench Full-Scale Comparative Analysis

- Total evaluated rows: 91,575
- Models: GPT-4o, Gemini Flash, Qwen2.5-VL-7B, Claude Sonnet, InternVL2.5-8B

## Table 1. Overall Performance
| Model | EM (%) | F1 (%) | Numeric (%) |
|---|---|---|---|
| GPT-4o | 34.41 | 31.35 | 10.63 |
| Gemini Flash | 36.24 | 33.13 | 11.89 |
| Qwen2.5-VL-7B | 32.38 | 29.89 | 8.95 |
| Claude Sonnet | 26.83 | 23.37 | 11.89 |
| InternVL2.5-8B | 20.26 | 20.25 | 5.33 |

## Table 2. EM by Modality
| Model | Tabular (%) | Time Series (%) | Graph (%) |
|---|---|---|---|
| GPT-4o | 41.41 | 21.84 | 43.64 |
| Gemini Flash | 42.37 | 27.15 | 39.32 |
| Qwen2.5-VL-7B | 37.75 | 24.32 | 35.34 |
| Claude Sonnet | 30.18 | 17.55 | 39.70 |
| InternVL2.5-8B | 21.71 | 16.10 | 26.21 |

## Table 3. Visualization Sensitivity Gaps
| Model | Modality | Best Viz | Best EM (%) | Worst Viz | Worst EM (%) | Gap (pp) |
|---|---|---|---|---|---|---|
| GPT-4o | tabular | table_image | 59.49 | bar_chart | 22.78 | 36.71 |
| GPT-4o | timeseries | text_only | 35.84 | recurrence_plot | 10.15 | 25.69 |
| GPT-4o | graph | text_only | 56.36 | adjacency_matrix | 33.33 | 23.03 |
| Gemini Flash | tabular | text_only | 61.13 | scatter_plot | 20.62 | 40.51 |
| Gemini Flash | timeseries | text_only | 38.69 | gaf | 19.49 | 19.20 |
| Gemini Flash | graph | text_only | 53.48 | adjacency_matrix | 31.82 | 21.67 |
| Qwen2.5-VL-7B | tabular | table_image | 54.56 | scatter_plot | 17.51 | 37.05 |
| Qwen2.5-VL-7B | timeseries | text_only | 32.85 | recurrence_plot | 19.20 | 13.65 |
| Qwen2.5-VL-7B | graph | text_only | 43.03 | adjacency_matrix | 28.18 | 14.85 |
| Claude Sonnet | tabular | table_image | 44.53 | scatter_plot | 16.15 | 28.39 |
| Claude Sonnet | timeseries | line_plot | 23.28 | recurrence_plot | 11.31 | 11.97 |
| Claude Sonnet | graph | text_only | 49.55 | node_link | 34.85 | 14.70 |
| InternVL2.5-8B | tabular | table_image | 30.93 | scatter_plot | 11.73 | 19.21 |
| InternVL2.5-8B | timeseries | line_plot | 19.71 | gaf | 11.09 | 8.61 |
| InternVL2.5-8B | graph | text_only | 30.91 | node_link | 23.33 | 7.58 |

## Table 4. EM by Difficulty
| Model | 1-hop (%) | 2-hop (%) | 3-hop (%) | Counterfactual (%) |
|---|---|---|---|---|
| GPT-4o | 21.20 | 37.47 | 31.09 | 78.80 |
| Gemini Flash | 23.22 | 41.47 | 33.12 | 74.02 |
| Qwen2.5-VL-7B | 21.26 | 35.58 | 26.38 | 72.93 |
| Claude Sonnet | 19.94 | 35.31 | 15.15 | 47.18 |
| InternVL2.5-8B | 14.39 | 25.77 | 15.68 | 33.98 |

## Table 5. EM by Source
| Model | Synthetic (%) | Real-world (%) |
|---|---|---|
| GPT-4o | 33.51 | 34.96 |
| Gemini Flash | 35.43 | 36.74 |
| Qwen2.5-VL-7B | 32.24 | 32.46 |
| Claude Sonnet | 25.71 | 27.51 |
| InternVL2.5-8B | 19.09 | 20.99 |

## Key Findings
- Best overall EM: **Gemini Flash** (36.24%).
- Visualization sensitivity remains substantial across all modalities and models.
- Full tables above should be copied into paper updates (replace pilot numbers).
