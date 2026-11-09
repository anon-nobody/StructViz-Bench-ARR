# StructViz-Bench Human Evaluation Form

## Instructions

- Evaluators: 3
- Use the provided JSONL files to inspect each item in order.
- Record one judgment per item in the tables below.
- Keep notes concise and specific when selecting Ambiguous/Incorrect/Implausible/Unclear.

## Task A - Answer Correctness Verification

For each item, inspect the visualization, question, and ground-truth answer.

Rating scale:
- Correct: Ground-truth answer is unambiguously correct.
- Ambiguous: Answer is defensible but the item allows multiple interpretations.
- Incorrect: Ground-truth answer appears wrong.

| # | question_id | modality | difficulty | source | viz_type | Evaluator judgment | Notes |
|---|---|---|---|---|---|---|---|
| 1 | tabular_000075::difficulty=1-hop | tabular | 1-hop | synthetic | bar_chart |  |  |
| 2 | tabular_000180::difficulty=1-hop | tabular | 1-hop | synthetic | bar_chart |  |  |
| 3 | tabular_000050::difficulty=1-hop | tabular | 1-hop | synthetic | heatmap |  |  |
| 4 | tabular_000450::difficulty=1-hop | tabular | 1-hop | synthetic | scatter_plot |  |  |
| 5 | tabular_000251::difficulty=1-hop | tabular | 1-hop | synthetic | bar_chart |  |  |
| 6 | tabular_000868::difficulty=1-hop | tabular | 1-hop | scitabalign | scatter_plot |  |  |
| 7 | tabular_001444::difficulty=1-hop | tabular | 1-hop | scitabalign | table_image |  |  |
| 8 | tabular_001233::difficulty=1-hop | tabular | 1-hop | scitabalign | scatter_plot |  |  |
| 9 | tabular_001564::difficulty=1-hop | tabular | 1-hop | scitabalign | table_image |  |  |
| 10 | tabular_001362::difficulty=1-hop | tabular | 1-hop | scitabalign | table_image |  |  |
| 11 | tabular_000139::difficulty=2-hop | tabular | 2-hop | synthetic | scatter_plot |  |  |
| 12 | tabular_000086::difficulty=2-hop | tabular | 2-hop | synthetic | text_only |  |  |
| 13 | tabular_000111::difficulty=2-hop | tabular | 2-hop | synthetic | scatter_plot |  |  |
| 14 | tabular_000164::difficulty=2-hop | tabular | 2-hop | synthetic | table_image |  |  |
| 15 | tabular_000434::difficulty=2-hop | tabular | 2-hop | synthetic | heatmap |  |  |
| 16 | tabular_001258::difficulty=2-hop | tabular | 2-hop | scitabalign | table_image |  |  |
| 17 | tabular_000794::difficulty=2-hop | tabular | 2-hop | scitabalign | scatter_plot |  |  |
| 18 | tabular_001411::difficulty=2-hop | tabular | 2-hop | scitabalign | bar_chart |  |  |
| 19 | tabular_000859::difficulty=2-hop | tabular | 2-hop | scitabalign | heatmap |  |  |
| 20 | tabular_000812::difficulty=2-hop | tabular | 2-hop | scitabalign | heatmap |  |  |
| 21 | tabular_000141::difficulty=3-hop | tabular | 3-hop | synthetic | table_image |  |  |
| 22 | tabular_000339::difficulty=3-hop | tabular | 3-hop | synthetic | text_only |  |  |
| 23 | tabular_000294::difficulty=3-hop | tabular | 3-hop | synthetic | table_image |  |  |
| 24 | tabular_000167::difficulty=3-hop | tabular | 3-hop | synthetic | text_only |  |  |
| 25 | tabular_001395::difficulty=3-hop | tabular | 3-hop | scitabalign | scatter_plot |  |  |
| 26 | tabular_001358::difficulty=3-hop | tabular | 3-hop | scitabalign | table_image |  |  |
| 27 | tabular_001318::difficulty=3-hop | tabular | 3-hop | scitabalign | scatter_plot |  |  |
| 28 | tabular_001613::difficulty=3-hop | tabular | 3-hop | scitabalign | bar_chart |  |  |
| 29 | tabular_000373::difficulty=counterfactual | tabular | counterfactual | synthetic | text_only |  |  |
| 30 | tabular_000048::difficulty=counterfactual | tabular | counterfactual | synthetic | scatter_plot |  |  |
| 31 | tabular_000122::difficulty=counterfactual | tabular | counterfactual | synthetic | text_only |  |  |
| 32 | tabular_000224::difficulty=counterfactual | tabular | counterfactual | synthetic | bar_chart |  |  |
| 33 | tabular_000619::difficulty=counterfactual | tabular | counterfactual | scitabalign | scatter_plot |  |  |
| 34 | tabular_001717::difficulty=counterfactual | tabular | counterfactual | scitabalign | bar_chart |  |  |
| 35 | tabular_001271::difficulty=counterfactual | tabular | counterfactual | scitabalign | bar_chart |  |  |
| 36 | tabular_001470::difficulty=counterfactual | tabular | counterfactual | scitabalign | scatter_plot |  |  |
| 37 | timeseries_000250::difficulty=1-hop | timeseries | 1-hop | synthetic | text_only |  |  |
| 38 | timeseries_000305::difficulty=1-hop | timeseries | 1-hop | synthetic | line_plot |  |  |
| 39 | timeseries_000431::difficulty=1-hop | timeseries | 1-hop | synthetic | recurrence_plot |  |  |
| 40 | timeseries_000252::difficulty=1-hop | timeseries | 1-hop | synthetic | recurrence_plot |  |  |
| 41 | timeseries_001340::difficulty=1-hop | timeseries | 1-hop | ett | heatmap |  |  |
| 42 | timeseries_001511::difficulty=1-hop | timeseries | 1-hop | ett | gaf |  |  |
| 43 | timeseries_001314::difficulty=1-hop | timeseries | 1-hop | ett | recurrence_plot |  |  |
| 44 | timeseries_001374::difficulty=1-hop | timeseries | 1-hop | ett | recurrence_plot |  |  |
| 45 | timeseries_000182::difficulty=2-hop | timeseries | 2-hop | synthetic | gaf |  |  |
| 46 | timeseries_000158::difficulty=2-hop | timeseries | 2-hop | synthetic | text_only |  |  |
| 47 | timeseries_000484::difficulty=2-hop | timeseries | 2-hop | synthetic | text_only |  |  |
| 48 | timeseries_000161::difficulty=2-hop | timeseries | 2-hop | synthetic | text_only |  |  |
| 49 | timeseries_001349::difficulty=2-hop | timeseries | 2-hop | ett | heatmap |  |  |
| 50 | timeseries_001748::difficulty=2-hop | timeseries | 2-hop | ett | heatmap |  |  |
| 51 | timeseries_001337::difficulty=2-hop | timeseries | 2-hop | ett | gaf |  |  |
| 52 | timeseries_001487::difficulty=2-hop | timeseries | 2-hop | ett | recurrence_plot |  |  |
| 53 | timeseries_000119::difficulty=3-hop | timeseries | 3-hop | synthetic | heatmap |  |  |
| 54 | timeseries_000193::difficulty=3-hop | timeseries | 3-hop | synthetic | line_plot |  |  |
| 55 | timeseries_000442::difficulty=3-hop | timeseries | 3-hop | synthetic | heatmap |  |  |
| 56 | timeseries_000141::difficulty=3-hop | timeseries | 3-hop | synthetic | gaf |  |  |
| 57 | timeseries_000024::difficulty=counterfactual | timeseries | counterfactual | synthetic | heatmap |  |  |
| 58 | timeseries_000346::difficulty=counterfactual | timeseries | counterfactual | synthetic | recurrence_plot |  |  |
| 59 | timeseries_000172::difficulty=counterfactual | timeseries | counterfactual | synthetic | line_plot |  |  |
| 60 | timeseries_000198::difficulty=counterfactual | timeseries | counterfactual | synthetic | line_plot |  |  |
| 61 | graph_000404::difficulty=1-hop | graph | 1-hop | synthetic | circular_layout |  |  |
| 62 | graph_000206::difficulty=1-hop | graph | 1-hop | synthetic | text_only |  |  |
| 63 | graph_000405::difficulty=1-hop | graph | 1-hop | synthetic | node_link |  |  |
| 64 | graph_000101::difficulty=1-hop | graph | 1-hop | synthetic | text_only |  |  |
| 65 | graph_001632::difficulty=1-hop | graph | 1-hop | networkx_realworld | circular_layout |  |  |
| 66 | graph_001610::difficulty=1-hop | graph | 1-hop | networkx_realworld | node_link |  |  |
| 67 | graph_001522::difficulty=1-hop | graph | 1-hop | networkx_realworld | circular_layout |  |  |
| 68 | graph_001622::difficulty=1-hop | graph | 1-hop | networkx_realworld | circular_layout |  |  |
| 69 | graph_000309::difficulty=2-hop | graph | 2-hop | synthetic | node_link |  |  |
| 70 | graph_000313::difficulty=2-hop | graph | 2-hop | synthetic | circular_layout |  |  |
| 71 | graph_000388::difficulty=2-hop | graph | 2-hop | synthetic | adjacency_matrix |  |  |
| 72 | graph_000213::difficulty=2-hop | graph | 2-hop | synthetic | adjacency_matrix |  |  |
| 73 | graph_001573::difficulty=2-hop | graph | 2-hop | networkx_realworld | text_only |  |  |
| 74 | graph_001514::difficulty=2-hop | graph | 2-hop | networkx_realworld | adjacency_matrix |  |  |
| 75 | graph_001593::difficulty=2-hop | graph | 2-hop | networkx_realworld | text_only |  |  |
| 76 | graph_001545::difficulty=2-hop | graph | 2-hop | networkx_realworld | adjacency_matrix |  |  |
| 77 | graph_000269::difficulty=3-hop | graph | 3-hop | synthetic | node_link |  |  |
| 78 | graph_000318::difficulty=3-hop | graph | 3-hop | synthetic | circular_layout |  |  |
| 79 | graph_000315::difficulty=3-hop | graph | 3-hop | synthetic | node_link |  |  |
| 80 | graph_000393::difficulty=3-hop | graph | 3-hop | synthetic | node_link |  |  |
| 81 | graph_001556::difficulty=3-hop | graph | 3-hop | networkx_realworld | text_only |  |  |
| 82 | graph_001637::difficulty=3-hop | graph | 3-hop | networkx_realworld | adjacency_matrix |  |  |
| 83 | graph_001577::difficulty=3-hop | graph | 3-hop | networkx_realworld | node_link |  |  |
| 84 | graph_001647::difficulty=3-hop | graph | 3-hop | networkx_realworld | text_only |  |  |
| 85 | graph_000323::difficulty=counterfactual | graph | counterfactual | synthetic | circular_layout |  |  |
| 86 | graph_000347::difficulty=counterfactual | graph | counterfactual | synthetic | node_link |  |  |
| 87 | graph_000296::difficulty=counterfactual | graph | counterfactual | synthetic | text_only |  |  |
| 88 | graph_000474::difficulty=counterfactual | graph | counterfactual | synthetic | adjacency_matrix |  |  |
| 89 | graph_001529::difficulty=counterfactual | graph | counterfactual | networkx_realworld | circular_layout |  |  |
| 90 | graph_001638::difficulty=counterfactual | graph | counterfactual | networkx_realworld | adjacency_matrix |  |  |
| 91 | graph_001528::difficulty=counterfactual | graph | counterfactual | networkx_realworld | adjacency_matrix |  |  |
| 92 | graph_001579::difficulty=counterfactual | graph | counterfactual | networkx_realworld | text_only |  |  |
| 93 | timeseries_000246::difficulty=counterfactual | timeseries | counterfactual | synthetic | text_only |  |  |
| 94 | timeseries_000220::difficulty=3-hop | timeseries | 3-hop | synthetic | recurrence_plot |  |  |
| 95 | timeseries_001354::difficulty=1-hop | timeseries | 1-hop | ett | gaf |  |  |
| 96 | timeseries_000140::difficulty=3-hop | timeseries | 3-hop | synthetic | heatmap |  |  |
| 97 | timeseries_000347::difficulty=counterfactual | timeseries | counterfactual | synthetic | line_plot |  |  |
| 98 | timeseries_001634::difficulty=1-hop | timeseries | 1-hop | ett | recurrence_plot |  |  |
| 99 | graph_001538::difficulty=counterfactual | graph | counterfactual | networkx_realworld | adjacency_matrix |  |  |
| 100 | timeseries_000268::difficulty=3-hop | timeseries | 3-hop | synthetic | heatmap |  |  |

## Task B - Visualization Sensitivity Plausibility

For each pair, decide whether it is plausible that a model answers correctly from viz_a but
incorrectly from viz_b.

Rating scale:
- Plausible: One visualization is meaningfully harder for the same question.
- Implausible: Both visualizations make the answer similarly obvious.
- Unclear: Insufficient clarity to decide.

| # | question_id | viz_a (correct) | viz_b (wrong) | em_a | em_b | Evaluator judgment | Notes |
|---|---|---|---|---|---|---|---|
| 1 | tabular_000006::difficulty=1-hop | table_image | bar_chart | 1.0 | 0.0 |  |  |
| 2 | tabular_000077::difficulty=1-hop | heatmap | bar_chart | 1.0 | 0.0 |  |  |
| 3 | tabular_000031::difficulty=1-hop | table_image | bar_chart | 1.0 | 0.0 |  |  |
| 4 | tabular_001444::difficulty=1-hop | table_image | bar_chart | 1.0 | 0.0 |  |  |
| 5 | tabular_001256::difficulty=1-hop | table_image | bar_chart | 1.0 | 0.0 |  |  |
| 6 | tabular_001416::difficulty=1-hop | heatmap | bar_chart | 1.0 | 0.0 |  |  |
| 7 | tabular_000012::difficulty=2-hop | bar_chart | heatmap | 1.0 | 0.0 |  |  |
| 8 | tabular_000435::difficulty=2-hop | scatter_plot | bar_chart | 1.0 | 0.0 |  |  |
| 9 | tabular_001593::difficulty=2-hop | heatmap | bar_chart | 1.0 | 0.0 |  |  |
| 10 | tabular_001549::difficulty=2-hop | heatmap | bar_chart | 1.0 | 0.0 |  |  |
| 11 | tabular_000220::difficulty=3-hop | table_image | bar_chart | 1.0 | 0.0 |  |  |
| 12 | tabular_000115::difficulty=3-hop | bar_chart | table_image | 1.0 | 0.0 |  |  |
| 13 | tabular_001151::difficulty=3-hop | text_only | bar_chart | 1.0 | 0.0 |  |  |
| 14 | tabular_000976::difficulty=3-hop | text_only | bar_chart | 1.0 | 0.0 |  |  |
| 15 | tabular_000124::difficulty=counterfactual | bar_chart | scatter_plot | 1.0 | 0.0 |  |  |
| 16 | tabular_000099::difficulty=counterfactual | text_only | bar_chart | 1.0 | 0.0 |  |  |
| 17 | tabular_000799::difficulty=counterfactual | heatmap | bar_chart | 1.0 | 0.0 |  |  |
| 18 | tabular_001377::difficulty=counterfactual | bar_chart | heatmap | 1.0 | 0.0 |  |  |
| 19 | timeseries_000325::difficulty=1-hop | line_plot | gaf | 1.0 | 0.0 |  |  |
| 20 | timeseries_000303::difficulty=1-hop | heatmap | gaf | 1.0 | 0.0 |  |  |
| 21 | timeseries_001801::difficulty=1-hop | text_only | gaf | 1.0 | 0.0 |  |  |
| 22 | timeseries_001027::difficulty=1-hop | text_only | gaf | 1.0 | 0.0 |  |  |
| 23 | timeseries_000313::difficulty=2-hop | heatmap | gaf | 1.0 | 0.0 |  |  |
| 24 | timeseries_000238::difficulty=2-hop | heatmap | gaf | 1.0 | 0.0 |  |  |
| 25 | timeseries_001689::difficulty=2-hop | text_only | gaf | 1.0 | 0.0 |  |  |
| 26 | timeseries_001749::difficulty=2-hop | gaf | heatmap | 1.0 | 0.0 |  |  |
| 27 | timeseries_000242::difficulty=3-hop | gaf | recurrence_plot | 1.0 | 0.0 |  |  |
| 28 | timeseries_000318::difficulty=3-hop | gaf | recurrence_plot | 1.0 | 0.0 |  |  |
| 29 | timeseries_000024::difficulty=counterfactual | gaf | heatmap | 1.0 | 0.0 |  |  |
| 30 | timeseries_000346::difficulty=counterfactual | gaf | recurrence_plot | 1.0 | 0.0 |  |  |
| 31 | graph_000050::difficulty=1-hop | text_only | adjacency_matrix | 1.0 | 0.0 |  |  |
| 32 | graph_000251::difficulty=1-hop | text_only | adjacency_matrix | 1.0 | 0.0 |  |  |
| 33 | graph_001520::difficulty=1-hop | node_link | adjacency_matrix | 1.0 | 0.0 |  |  |
| 34 | graph_001621::difficulty=1-hop | text_only | adjacency_matrix | 1.0 | 0.0 |  |  |
| 35 | graph_000462::difficulty=2-hop | circular_layout | adjacency_matrix | 1.0 | 0.0 |  |  |
| 36 | graph_000233::difficulty=2-hop | text_only | adjacency_matrix | 1.0 | 0.0 |  |  |
| 37 | graph_001514::difficulty=2-hop | text_only | adjacency_matrix | 1.0 | 0.0 |  |  |
| 38 | graph_001603::difficulty=2-hop | text_only | adjacency_matrix | 1.0 | 0.0 |  |  |
| 39 | graph_000290::difficulty=3-hop | circular_layout | adjacency_matrix | 1.0 | 0.0 |  |  |
| 40 | graph_000092::difficulty=3-hop | node_link | adjacency_matrix | 1.0 | 0.0 |  |  |
| 41 | graph_001576::difficulty=3-hop | circular_layout | adjacency_matrix | 1.0 | 0.0 |  |  |
| 42 | graph_001586::difficulty=3-hop | circular_layout | adjacency_matrix | 1.0 | 0.0 |  |  |
| 43 | graph_000373::difficulty=counterfactual | node_link | adjacency_matrix | 1.0 | 0.0 |  |  |
| 44 | graph_000171::difficulty=counterfactual | adjacency_matrix | text_only | 1.0 | 0.0 |  |  |
| 45 | graph_001608::difficulty=counterfactual | adjacency_matrix | circular_layout | 1.0 | 0.0 |  |  |
| 46 | graph_001648::difficulty=counterfactual | adjacency_matrix | circular_layout | 1.0 | 0.0 |  |  |
| 47 | timeseries_001719::difficulty=2-hop | heatmap | gaf | 1.0 | 0.0 |  |  |
| 48 | timeseries_000293::difficulty=3-hop | gaf | recurrence_plot | 1.0 | 0.0 |  |  |
| 49 | timeseries_000221::difficulty=counterfactual | gaf | recurrence_plot | 1.0 | 0.0 |  |  |
| 50 | timeseries_001304::difficulty=1-hop | text_only | gaf | 1.0 | 0.0 |  |  |

## Metadata

- Random seed: 42
- Task A items: 100
- Task B pairs: 50
