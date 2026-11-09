#!/usr/bin/env python3
"""Object-disjoint split for the multi-format LoRA study (ACL review round 3).

Pools pairs_all_train.jsonl and pairs_all_eval.jsonl (44 objects, every object in both), splits
the objects in half per modality with random.Random(0), and writes pairs_objsplit_train.jsonl /
pairs_objsplit_eval.jsonl so that no evaluation object is seen in training.
"""
from __future__ import annotations

import json
import os
import random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def main() -> None:
    q2d = {}
    with open(os.path.join(ROOT, "benchmark", "base_items.jsonl")) as fh:
        for line in fh:
            r = json.loads(line)
            q2d[r["question_id"]] = (r["data_id"], r["modality"])
    rows = []
    for name in ("pairs_all_train.jsonl", "pairs_all_eval.jsonl"):
        with open(os.path.join(HERE, name)) as fh:
            rows += [json.loads(line) for line in fh]
    objs = defaultdict(set)
    for r in rows:
        d, m = q2d[r["question_id"]]
        objs[m].add(d)
    rng = random.Random(0)
    train_objs = set()
    for m in sorted(objs):
        o = sorted(objs[m])
        rng.shuffle(o)
        train_objs |= set(o[: len(o) // 2])
    tr = [r for r in rows if q2d[r["question_id"]][0] in train_objs]
    ev = [r for r in rows if q2d[r["question_id"]][0] not in train_objs]
    for name, part in (("pairs_objsplit_train.jsonl", tr), ("pairs_objsplit_eval.jsonl", ev)):
        with open(os.path.join(HERE, name), "w") as fh:
            for r in part:
                fh.write(json.dumps(r) + "\n")
    n_ev = len({q2d[r["question_id"]][0] for r in ev})
    print(f"train {len(tr)} q / {len(train_objs)} objects; eval {len(ev)} q / {n_ev} objects")


if __name__ == "__main__":
    main()
