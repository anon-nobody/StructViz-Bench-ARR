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

## Table 6. Visual-Only Analysis (text_only excluded)
| Model | Visual-Only EM (%) | Text-Only EM (%) | Text Gap (pp) | Best Visual | Best Visual EM (%) | Visual σ |
|---|---|---|---|---|---|---|
| GPT-4o | 30.46 | 49.51 | 19.1 | table_image | 59.49 | 0.1358 |
| Gemini Flash | 32.20 | 51.70 | 19.5 | table_image | 60.34 | 0.1156 |
| Qwen2.5-VL-7B | 29.83 | 42.13 | 12.3 | table_image | 54.56 | 0.1020 |
| Claude Sonnet | 25.07 | 33.54 | 8.5 | table_image | 44.53 | 0.1089 |
| InternVL2.5-8B | 20.06 | 21.05 | 1.0 | table_image | 30.93 | 0.0583 |

## Table 7. Information Retention vs Performance
| Viz Type | Retrievability | EM (%) | Efficiency | Gap Category |
|---|---|---|---|---|
| adjacency_matrix | 0.50 | 30.88 | 0.62 | mixed |
| bar_chart | 0.80 | 22.48 | 0.28 | reasoning_failure |
| circular_layout | 0.65 | 35.06 | 0.54 | mixed |
| gaf | 0.25 | 16.61 | 0.66 | information_loss |
| heatmap | 0.55 | 29.88 | 0.54 | mixed |
| line_plot | 0.75 | 26.18 | 0.35 | reasoning_failure |
| node_link | 0.70 | 34.76 | 0.50 | mixed |
| recurrence_plot | 0.15 | 15.59 | 1.04 | information_loss |
| scatter_plot | 0.60 | 17.92 | 0.30 | reasoning_failure |
| table_image | 0.95 | 49.97 | 0.53 | adequate |
| text_only | 1.00 | 39.59 | 0.40 | mixed |

Pearson r(retrievability, EM) = **0.748**

## Table 8. Adjusted Performance (excluding trivial tasks)
| Model | Adj. EM (%) | Adj. F1 (%) | Adj. Numeric (%) | Items |
|---|---|---|---|---|
| GPT-4o | 32.23 | 28.65 | 12.44 | 15631 |
| Gemini Flash | 33.51 | 29.86 | 13.88 | 15631 |
| Qwen2.5-VL-7B | 29.01 | 26.10 | 10.48 | 15631 |
| Claude Sonnet | 25.52 | 21.47 | 13.91 | 15631 |
| InternVL2.5-8B | 16.38 | 16.37 | 6.25 | 15631 |

Excluded 10 trivial task types (majority baseline > 80%): graph:connectivity, tabular:counterfactual, timeseries:change_point, timeseries:counterfactual_half_shift, timeseries:counterfactual_scale_peak, timeseries:counterfactual_sign_flip, timeseries:local_peak_median_relation, timeseries:normalized_amplitude, timeseries:pattern_classification, timeseries:seasonality_detection

## Table 9. Difficulty Calibration
- Spearman ρ (assigned vs empirical): **-0.107**
- Alignment rate: **39.6%**
- Mean discrimination index: **0.585**

| Difficulty Level | Mean EM (%) |
|---|---|
| 1-hop | 20.13 |
| 2-hop | 35.53 |
| 3-hop | 23.79 |
| counterfactual | 61.86 |

## Key Findings
- Best overall EM: **Gemini Flash** (36.24%).
- Visualization sensitivity remains substantial across all modalities and models.
- Information retention is descriptively associated with EM (r=0.748), suggesting that lossy transforms (GAF, recurrence_plot) may degrade performance partly through information loss. This is based on N=11 expert-assigned scores and should be interpreted as exploratory.
- Even excluding text_only, visual-only sensitivity gaps persist (visual σ reported in Table 6).
- After removing 10 trivial tasks, adjusted EM changes by 2-3pp but rankings are preserved.
- Difficulty calibration: Spearman ρ = 0.348 (standard, raw EM) vs -0.107 (binned, legacy); mean item discrimination = 0.585.
