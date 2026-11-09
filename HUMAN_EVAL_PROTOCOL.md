# Human Evaluation Protocol for StructViz-Bench

> **Status: protocol defined; annotations in progress.**
> Ground-truth answers are computed programmatically and are correct with respect
> to the underlying data by construction. This protocol covers the separate
> question of whether each answer is legible and unambiguous *from the rendered
> image alone*. The ready-to-distribute annotation package — stratified sample,
> per-annotator sheets, the offline rating tool, and the aggregator — is in
> [`human_eval_package/`](human_eval_package/); agreement statistics will be
> reported once the three independent annotations are returned.

## Purpose

Validate benchmark quality by measuring human agreement with ground-truth answers
and assessing whether visualization sensitivity is a genuine phenomenon (not an
artifact of flawed questions or answers).

## Scope

- **Sample size**: 100 randomly stratified items (balanced across 3 modalities,
  4 difficulty levels, and both source types)
- **Evaluators**: 3 evaluators with quantitative background
- **Estimated time**: ~2 hours per evaluator

## Task Design

### Task A: Answer Correctness Verification (all 100 items)

For each item, the evaluator sees:
1. The rendered visualization image
2. The question text
3. The ground-truth answer

They rate:
- **Correct**: The ground-truth answer is unambiguously correct given the visualization
- **Ambiguous**: The answer is defensible but the question/visualization admits other interpretations
- **Incorrect**: The ground-truth answer appears wrong

**Target**: >= 95% items rated "Correct" by majority vote.

### Task B: Visualization Sensitivity Plausibility (50 paired items)

For 50 items selected where models showed sensitivity (correct on one viz, wrong on another):
1. Show both visualizations side by side (e.g., bar_chart vs scatter_plot)
2. Show the question and ground-truth answer
3. Ask: "Is it plausible that a model could answer this correctly from visualization A
   but incorrectly from visualization B?"

They rate:
- **Plausible**: The information is harder to extract from one visualization
- **Implausible**: Both visualizations make the answer equally obvious
- **Unclear**: Cannot determine

**Target**: >= 70% items rated "Plausible" by majority vote.

## Sampling Strategy

```python
# Stratified random sampling
# 100 items total:
#   - 34 tabular, 33 timeseries, 33 graph
#   - Within each modality: ~8 per difficulty level
#   - 50% synthetic, 50% real-world (where available)
#   - For Task B: select from items where >= 2 models show viz sensitivity
```

## Metrics Reported in Paper

1. **Human accuracy**: % items where ground-truth matches human consensus
2. **Inter-annotator agreement**: Fleiss' kappa across 3 evaluators
3. **Sensitivity plausibility rate**: % of sensitivity pairs rated "Plausible"

## Implementation

Script: `scripts/prepare_human_eval.py`
- Samples 100 items from benchmark
- Generates evaluation sheets (HTML or spreadsheet format)
- Collects and aggregates responses

## Integration with Paper

Add to Section 3.4 (Quality Control) or as a new subsection:
> "Human evaluation on a stratified sample of 100 items confirms benchmark quality:
> X% of ground-truth answers are rated correct by majority vote (Fleiss' kappa = Y),
> and Z% of visualization sensitivity pairs are rated plausible by evaluators."
