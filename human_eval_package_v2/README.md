# StructViz-Bench — re-rating on the corrected (v2) renderings (release copy)

The same two annotators as in `human_eval_package/` judged 30 corrected-suite renderings
(10 tabular / 10 graph / 10 time series). 22 items repeat a v1 item with the same question and
format (`v1_item_id` in `items.jsonl`); 8 are new time-series items on the untruncated listing.
The two returned sheets are identical on all 30 items and are therefore reported in the paper as
one judgement, not as inter-rater agreement.

## Contents
- `items.jsonl` — the 30 items (question, key, v2 image sha256, `v1_item_id` when repeated).
- `images/` — the 30 v2 images shown to the raters.
- `ratings_annotator1.csv`, `ratings_annotator2.csv` — returned sheets; free-text notes withheld
  until de-anonymisation (placeholder text in the cell).
- `analysis/adjudication_v2.txt` — before/after counts on the 22 repeated items and a
  recomputation of every non-Correct item from the source object (no key is wrong).
- `assemble_ratings.py`, `human_eval_aggregate.py` — aggregation.

`scripts/verify_paper_numbers.py` (repository root) recomputes the appendix's counts from these
sheets and the v1 sheets.
