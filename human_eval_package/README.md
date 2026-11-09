# StructViz-Bench — human image-answerability audit (release copy)

Two annotators with quantitative backgrounds independently rated 100 (question, rendering)
pairs of the original (v1) suite, seeing only the rendered image, the question and the released
key, as `Correct` / `Ambiguous` / `Incorrect` (is the key clearly right from the image alone?),
optionally recording their own answer. A third annotator did not complete the task in time.
The protocol and results are in the paper's appendix "Human image-answerability audit".

## Contents
- `items.jsonl` — the 100 sampled items (34 tabular / 33 time-series / 33 graph; all 14 formats).
- `images/` — the 100 rendered images shown to the raters.
- `ratings_annotator1.csv`, `ratings_annotator2.csv` — the returned sheets. Columns: `item_id,
  modality, task, viz_type, image, question, ground_truth, rating, your_answer, notes`. Free-text
  own answers are released in English translation; the free-text `notes` are withheld until
  de-anonymisation (placeholder text in the cell).
- `analysis/adjudication_2raters.txt` — per-item recomputation: whether the answer function
  re-run on the full data agrees with the key, and what the same function returns on the rows /
  points actually visible in the rendering (notes column withheld).
- `assemble_ratings.py`, `human_eval_aggregate.py` — aggregation (agreement, Cohen's kappa,
  own-answer accuracy). Own answers are matched to the key with the same rule as the model
  scorer: exact match after lower-casing, or numeric within 0.05 absolute / 1% relative; no
  substring credit.

## Reproduce the appendix numbers
```bash
python assemble_ratings.py --sheets ratings_annotator1.csv ratings_annotator2.csv --out ratings_long.csv
python human_eval_aggregate.py --ratings ratings_long.csv
```
`scripts/verify_paper_numbers.py` (repository root) also recomputes the reported counts,
agreement, kappa and own-answer figures from these two sheets.
