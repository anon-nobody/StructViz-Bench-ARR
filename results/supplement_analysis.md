# the review venue Supplementary Analysis

- Total full-scale rows analyzed: 73,260
- Core models: GPT-4o, Gemini Flash, Qwen2.5-VL-7B, Claude Sonnet
- Files: results/full_{claude,gpt4o,gemini,qwen}.jsonl and mixed comparisons

## Section 1. Visual-Only Sensitivity (text_only excluded)

Full-gap vs visual-only-gap comparison per model and modality.

| Model | Modality | Full Best | Full Worst | Full Gap (pp) | Visual Best | Visual Worst | Visual-Only Gap (pp) | Text-Only Contribution (pp) |
|---|---|---|---|---|---|---|---|---|
| GPT-4o | tabular | table_image | bar_chart | 36.71 | table_image | bar_chart | 36.71 | 0.00 |
| GPT-4o | timeseries | text_only | recurrence_plot | 25.69 | line_plot | recurrence_plot | 17.45 | 8.25 |
| GPT-4o | graph | text_only | adjacency_matrix | 23.03 | node_link | adjacency_matrix | 10.30 | 12.73 |
| Gemini Flash | tabular | text_only | scatter_plot | 40.51 | table_image | scatter_plot | 39.72 | 0.79 |
| Gemini Flash | timeseries | text_only | gaf | 19.20 | line_plot | gaf | 13.50 | 5.69 |
| Gemini Flash | graph | text_only | adjacency_matrix | 21.67 | circular_layout | adjacency_matrix | 4.39 | 17.27 |
| Qwen2.5-VL-7B | tabular | table_image | scatter_plot | 37.05 | table_image | scatter_plot | 37.05 | 0.00 |
| Qwen2.5-VL-7B | timeseries | text_only | recurrence_plot | 13.65 | line_plot | recurrence_plot | 8.10 | 5.55 |
| Qwen2.5-VL-7B | graph | text_only | adjacency_matrix | 14.85 | node_link | adjacency_matrix | 8.03 | 6.82 |
| Claude Sonnet | tabular | table_image | scatter_plot | 28.39 | table_image | scatter_plot | 28.39 | 0.00 |
| Claude Sonnet | timeseries | line_plot | recurrence_plot | 11.97 | line_plot | recurrence_plot | 11.97 | 0.00 |
| Claude Sonnet | graph | text_only | node_link | 14.70 | circular_layout | node_link | 3.03 | 11.67 |

## Section 2. Per-Question Flip Analysis

Model-level flip-rate range: 45.24% to 55.92%.

### Overall per model

| Model | # Questions | Always Correct | Always Wrong | Flipping | Flip Rate (%) |
|---|---|---|---|---|---|
| GPT-4o | 3795 | 452 | 1407 | 1936 | 51.01 |
| Gemini Flash | 3795 | 416 | 1257 | 2122 | 55.92 |
| Qwen2.5-VL-7B | 3795 | 453 | 1625 | 1717 | 45.24 |
| Claude Sonnet | 3795 | 270 | 1783 | 1742 | 45.90 |

### Per model and modality

| Model | Modality | # Questions | Always Correct | Always Wrong | Flipping | Flip Rate (%) |
|---|---|---|---|---|---|---|
| GPT-4o | tabular | 1765 | 229 | 452 | 1084 | 61.42 |
| GPT-4o | timeseries | 1370 | 63 | 749 | 558 | 40.73 |
| GPT-4o | graph | 660 | 160 | 206 | 294 | 44.55 |
| Gemini Flash | tabular | 1765 | 173 | 403 | 1189 | 67.37 |
| Gemini Flash | timeseries | 1370 | 116 | 633 | 621 | 45.33 |
| Gemini Flash | graph | 660 | 127 | 221 | 312 | 47.27 |
| Qwen2.5-VL-7B | tabular | 1765 | 159 | 546 | 1060 | 60.06 |
| Qwen2.5-VL-7B | timeseries | 1370 | 148 | 780 | 442 | 32.26 |
| Qwen2.5-VL-7B | graph | 660 | 146 | 299 | 215 | 32.58 |
| Claude Sonnet | tabular | 1765 | 57 | 678 | 1030 | 58.36 |
| Claude Sonnet | timeseries | 1370 | 77 | 867 | 426 | 31.09 |
| Claude Sonnet | graph | 660 | 136 | 238 | 286 | 43.33 |

## Section 3. Statistical Significance Tests

Paired permutation test (10000 iterations) between best/worst viz types for each model x modality.
Bonferroni correction uses 12 comparisons.

| Model | Modality | Viz Pair | Paired N | p_value | corrected_p | Cohen's d | Significant |
|---|---|---|---|---|---|---|---|
| GPT-4o | tabular | table_image vs bar_chart | 1765 | 0.000100 | 0.001200 | 0.661 | yes |
| GPT-4o | timeseries | text_only vs recurrence_plot | 1370 | 0.000100 | 0.001200 | 0.508 | yes |
| GPT-4o | graph | text_only vs adjacency_matrix | 660 | 0.000100 | 0.001200 | 0.460 | yes |
| Gemini Flash | tabular | text_only vs scatter_plot | 1765 | 0.000100 | 0.001200 | 0.712 | yes |
| Gemini Flash | timeseries | text_only vs gaf | 1370 | 0.000100 | 0.001200 | 0.367 | yes |
| Gemini Flash | graph | text_only vs adjacency_matrix | 660 | 0.000100 | 0.001200 | 0.432 | yes |
| Qwen2.5-VL-7B | tabular | table_image vs scatter_plot | 1765 | 0.000100 | 0.001200 | 0.677 | yes |
| Qwen2.5-VL-7B | timeseries | text_only vs recurrence_plot | 1370 | 0.000100 | 0.001200 | 0.293 | yes |
| Qwen2.5-VL-7B | graph | text_only vs adjacency_matrix | 660 | 0.000100 | 0.001200 | 0.330 | yes |
| Claude Sonnet | tabular | table_image vs scatter_plot | 1765 | 0.000100 | 0.001200 | 0.513 | yes |
| Claude Sonnet | timeseries | line_plot vs recurrence_plot | 1370 | 0.000100 | 0.001200 | 0.341 | yes |
| Claude Sonnet | graph | text_only vs node_link | 660 | 0.000100 | 0.001200 | 0.278 | yes |

## Section 4. Random and Majority Baselines

Random baseline uses 1 / (# unique answers) per task on deduplicated questions.
Majority baseline predicts the most common answer per task.

### Baseline by task

| Task | # Questions | # Unique Answers | Random EM (%) | Majority EM (%) |
|---|---|---|---|---|
| aggregation | 207 | 179 | 0.56 | 3.38 |
| anomaly_detection | 20 | 17 | 5.88 | 10.00 |
| bipartite_check | 20 | 2 | 50.00 | 60.00 |
| centrality | 36 | 19 | 5.26 | 25.00 |
| change_point | 107 | 8 | 12.50 | 93.46 |
| change_volatility | 20 | 2 | 50.00 | 60.00 |
| clustering | 36 | 28 | 3.57 | 22.22 |
| community | 92 | 25 | 4.00 | 36.96 |
| comparison | 224 | 101 | 0.99 | 36.61 |
| connectivity | 96 | 1 | 100.00 | 100.00 |
| correlation | 49 | 2 | 50.00 | 67.35 |
| counterfactual | 222 | 2 | 50.00 | 86.49 |
| counterfactual_half_shift | 20 | 2 | 50.00 | 90.00 |
| counterfactual_scale_peak | 20 | 1 | 100.00 | 100.00 |
| counterfactual_sign_flip | 15 | 1 | 100.00 | 100.00 |
| cycle_detection | 20 | 2 | 50.00 | 80.00 |
| degree_query | 208 | 63 | 1.59 | 13.94 |
| diameter | 20 | 9 | 11.11 | 20.00 |
| edge_count | 20 | 18 | 5.56 | 10.00 |
| endpoint_comparison | 55 | 2 | 50.00 | 50.91 |
| filtering | 104 | 19 | 5.26 | 10.58 |
| forecasting | 20 | 20 | 5.00 | 5.00 |
| half_change_intensity | 20 | 2 | 50.00 | 60.00 |
| local_peak_median_relation | 5 | 1 | 100.00 | 100.00 |
| mean_half_comparison | 62 | 2 | 50.00 | 62.90 |
| mean_shift_magnitude | 20 | 20 | 5.00 | 5.00 |
| median_mean_relation | 20 | 2 | 50.00 | 55.00 |
| normalized_amplitude | 20 | 1 | 100.00 | 100.00 |
| oscillation_count | 10 | 10 | 10.00 | 10.00 |
| outlier_detection | 230 | 155 | 0.65 | 2.61 |
| pattern_classification | 20 | 1 | 100.00 | 100.00 |
| pattern_label_lookup | 107 | 10 | 10.00 | 78.50 |
| peak_identification | 214 | 129 | 0.78 | 3.27 |
| peak_trough_order | 20 | 2 | 50.00 | 65.00 |
| range_query | 107 | 105 | 0.95 | 1.87 |
| ranking | 90 | 49 | 2.04 | 6.67 |
| seasonality_detection | 107 | 2 | 50.00 | 92.52 |
| shortest_path | 36 | 8 | 12.50 | 36.11 |
| split_trend_consistency | 20 | 2 | 50.00 | 65.00 |
| threshold_count | 20 | 14 | 7.14 | 10.00 |
| trend_analysis | 247 | 2 | 50.00 | 63.56 |
| value_extraction | 468 | 384 | 0.26 | 2.35 |
| value_lookup | 214 | 202 | 0.50 | 2.80 |
| volatility | 107 | 2 | 50.00 | 53.27 |

### Model EM vs baselines

| Model | Model EM (%) | Random Baseline (%) | Majority Baseline (%) | Delta vs Random (pp) | Delta vs Majority (pp) |
|---|---|---|---|---|---|
| GPT-4o | 34.41 | 19.64 | 34.68 | 14.77 | -0.27 |
| Gemini Flash | 36.24 | 19.64 | 34.68 | 16.60 | 1.56 |
| Qwen2.5-VL-7B | 32.38 | 19.64 | 34.68 | 12.74 | -2.30 |
| Claude Sonnet | 26.83 | 19.64 | 34.68 | 7.19 | -7.85 |

## Section 5. Binary vs Open-Form Task Breakdown

Binary labels are answers in {yes, no, true, false, increasing, decreasing}; all others are open-form.

| Model | Binary Rows | Binary EM (%) | Open Rows | Open EM (%) | Binary-Open Gap (pp) |
|---|---|---|---|---|---|
| GPT-4o | 5067 | 63.04 | 13248 | 23.46 | 39.58 |
| Gemini Flash | 5067 | 64.99 | 13248 | 25.24 | 39.75 |
| Qwen2.5-VL-7B | 5067 | 65.32 | 13248 | 19.78 | 45.55 |
| Claude Sonnet | 5067 | 44.50 | 13248 | 20.06 | 24.44 |

## Section 6. Per-Source Real-World Analysis

Source-specific EM and sensitivity gaps across synthetic, scitabalign, ett, and networkx.

### Source-level summary

| Model | Source | # Rows | EM (%) | Mean Modality Gap (pp) |
|---|---|---|---|---|
| GPT-4o | synthetic | 7000 | 33.51 | 22.87 |
| GPT-4o | scitabalign | 6325 | 47.35 | 42.53 |
| GPT-4o | ett | 4350 | 15.98 | 25.29 |
| Gemini Flash | synthetic | 7000 | 35.43 | 18.00 |
| Gemini Flash | scitabalign | 6325 | 48.09 | 47.11 |
| Gemini Flash | ett | 4350 | 20.69 | 25.52 |
| Qwen2.5-VL-7B | synthetic | 7000 | 32.24 | 16.33 |
| Qwen2.5-VL-7B | scitabalign | 6325 | 42.02 | 41.26 |
| Qwen2.5-VL-7B | ett | 4350 | 18.60 | 20.00 |
| Claude Sonnet | synthetic | 7000 | 25.71 | 11.47 |
| Claude Sonnet | scitabalign | 6325 | 35.62 | 35.57 |
| Claude Sonnet | ett | 4350 | 14.34 | 16.78 |

### Source x modality sensitivity

| Model | Source | Modality | Best Viz | Best EM (%) | Worst Viz | Worst EM (%) | Gap (pp) |
|---|---|---|---|---|---|---|---|
| GPT-4o | synthetic | tabular | table_image | 39.80 | scatter_plot | 16.60 | 23.20 |
| GPT-4o | synthetic | timeseries | text_only | 42.60 | recurrence_plot | 16.20 | 26.40 |
| GPT-4o | synthetic | graph | text_only | 54.80 | adjacency_matrix | 35.80 | 19.00 |
| GPT-4o | scitabalign | tabular | table_image | 67.27 | bar_chart | 24.74 | 42.53 |
| GPT-4o | ett | timeseries | text_only | 31.95 | recurrence_plot | 6.67 | 25.29 |
| Gemini Flash | synthetic | tabular | text_only | 42.00 | scatter_plot | 16.60 | 25.40 |
| Gemini Flash | synthetic | timeseries | line_plot | 42.40 | gaf | 33.60 | 8.80 |
| Gemini Flash | synthetic | graph | text_only | 54.20 | adjacency_matrix | 34.40 | 19.80 |
| Gemini Flash | scitabalign | tabular | table_image | 69.33 | scatter_plot | 22.21 | 47.11 |
| Gemini Flash | ett | timeseries | text_only | 36.78 | recurrence_plot | 11.26 | 25.52 |
| Qwen2.5-VL-7B | synthetic | tabular | text_only | 40.00 | scatter_plot | 11.00 | 29.00 |
| Qwen2.5-VL-7B | synthetic | timeseries | line_plot | 38.40 | recurrence_plot | 31.20 | 7.20 |
| Qwen2.5-VL-7B | synthetic | graph | text_only | 43.60 | adjacency_matrix | 30.80 | 12.80 |
| Qwen2.5-VL-7B | scitabalign | tabular | table_image | 61.34 | scatter_plot | 20.08 | 41.26 |
| Qwen2.5-VL-7B | ett | timeseries | text_only | 32.30 | recurrence_plot | 12.30 | 20.00 |
| Claude Sonnet | synthetic | tabular | table_image | 21.40 | scatter_plot | 11.20 | 10.20 |
| Claude Sonnet | synthetic | timeseries | line_plot | 29.60 | recurrence_plot | 18.20 | 11.40 |
| Claude Sonnet | synthetic | graph | text_only | 48.80 | node_link | 36.00 | 12.80 |
| Claude Sonnet | scitabalign | tabular | table_image | 53.68 | scatter_plot | 18.10 | 35.57 |
| Claude Sonnet | ett | timeseries | text_only | 24.14 | recurrence_plot | 7.36 | 16.78 |

## Section 7. Mean/Median Gap Reporting

Gap distributions are absolute pairwise EM differences across visualization types.

### By model

| Model | # Pairwise Gaps | Mean (pp) | Median (pp) | Q1 (pp) | Q3 (pp) |
|---|---|---|---|---|---|
| GPT-4o | 26 | 15.77 | 14.54 | 8.27 | 20.65 |
| Gemini Flash | 26 | 15.44 | 14.72 | 5.24 | 19.54 |
| Qwen2.5-VL-7B | 26 | 11.86 | 8.27 | 5.83 | 14.55 |
| Claude Sonnet | 26 | 9.92 | 8.62 | 2.95 | 14.28 |

### By modality

| Modality | # Pairwise Gaps | Mean (pp) | Median (pp) | Q1 (pp) | Q3 (pp) |
|---|---|---|---|---|---|
| tabular | 40 | 19.76 | 19.18 | 9.38 | 29.33 |
| timeseries | 40 | 8.95 | 8.10 | 5.35 | 12.30 |
| graph | 24 | 9.55 | 8.56 | 3.71 | 14.73 |

## Section 8. Claude Mixed-Type Deep Dive

Claude mixed-type response diagnostics compared to GPT-4o and Qwen.

| Model | # Rows | EM (%) | Mean Chars | Median Chars | Mean Tokens | Median Tokens | [ERROR] Count | [ERROR] Rate (%) | Empty Count | Empty Rate (%) | Refusal Count | Refusal Rate (%) | Numeric-like Count | Numeric-like Rate (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Claude Sonnet | 3000 | 1.33 | 31.39 | 3.00 | 5.46 | 1.00 | 0 | 0.00 | 0 | 0.00 | 124 | 4.13 | 2673 | 89.10 |
| GPT-4o | 3000 | 35.47 | 3.65 | 3.00 | 1.10 | 1.00 | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 363 | 12.10 |
| Qwen2.5-VL-7B | 3000 | 55.83 | 3.50 | 2.00 | 1.01 | 1.00 | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 400 | 13.33 |

### Claude modality breakdown

| Modality | # Rows | EM (%) | [ERROR] Count | [ERROR] Rate (%) | Empty Count | Empty Rate (%) | Refusal Count | Refusal Rate (%) |
|---|---|---|---|---|---|---|---|---|
| mixed_tab_graph | 1000 | 0.00 | 0 | 0.00 | 0 | 0.00 | 39 | 3.90 |
| mixed_tab_ts | 1000 | 4.00 | 0 | 0.00 | 0 | 0.00 | 25 | 2.50 |
| mixed_ts_graph | 1000 | 0.00 | 0 | 0.00 | 0 | 0.00 | 60 | 6.00 |

### Top normalized predictions

- Claude Sonnet: `10` (185), `1` (135), `100` (107), `110` (100), `115` (87), `116` (85), `1.0` (64), `76.47` (61)
- GPT-4o: `no` (1363), `yes` (615), `none` (359), `series peak` (87), `graph density*100` (83), `0.0` (63), `table column` (60), `0.1` (48)
- Qwen2.5-VL-7B: `no` (1805), `yes` (430), `2.5` (120), `graph_density*100` (100), `time_series` (100), `none` (65), `20.5` (40), `25` (40)

### What went wrong

- Claude shows elevated `[ERROR]` and non-numeric output rates compared with GPT-4o and Qwen on mixed tasks.
- Prediction-length statistics indicate unstable output formatting rather than a simple constant-short-answer failure mode.
- Frequent repeated fallback tokens (top predictions) suggest parser/API failure artifacts leaking into final predictions.
