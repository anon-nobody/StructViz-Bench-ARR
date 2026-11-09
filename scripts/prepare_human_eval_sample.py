from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

MODALITIES = ("tabular", "timeseries", "graph")
DIFFICULTIES = ("1-hop", "2-hop", "3-hop", "counterfactual")
SOURCE_TYPES = ("synthetic", "real-world")

TASK_A_SIZE = 100
TASK_B_SIZE = 50
SEED = 42


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(f"{content}\n", encoding="utf-8")


def source_type_from_source(source: str) -> str:
    return "synthetic" if source == "synthetic" else "real-world"


def allocate_even(total: int, categories: tuple[str, ...]) -> dict[str, int]:
    base = total // len(categories)
    remainder = total % len(categories)
    return {
        category: base + (1 if i < remainder else 0)
        for i, category in enumerate(categories)
    }


def build_question_records(rows: list[dict]) -> list[dict]:
    by_question: dict[str, dict] = {}
    for row in rows:
        qid = str(row["question_id"])
        if qid not in by_question:
            by_question[qid] = {
                "question_id": qid,
                "question": str(row["question"]),
                "answer": str(row["answer"]),
                "modality": str(row["modality"]),
                "difficulty": str(row["difficulty"]),
                "source": str(row["source"]),
                "source_type": source_type_from_source(str(row["source"])),
                "viz_rows": [],
            }
        by_question[qid]["viz_rows"].append(row)

    for record in by_question.values():
        record["viz_rows"] = sorted(
            record["viz_rows"], key=lambda x: str(x.get("viz_type", ""))
        )

    return sorted(by_question.values(), key=lambda x: x["question_id"])


def choose_viz_type(viz_rows: list[dict], rng: random.Random) -> str:
    viz_types = sorted({str(row["viz_type"]) for row in viz_rows})
    if not viz_types:
        raise ValueError("No viz_type available for question.")
    return rng.choice(viz_types)


def build_task_a_pool(question_records: list[dict], rng: random.Random) -> list[dict]:
    pool: list[dict] = []
    for record in question_records:
        pool.append(
            {
                "question_id": record["question_id"],
                "question": record["question"],
                "answer": record["answer"],
                "modality": record["modality"],
                "difficulty": record["difficulty"],
                "source": record["source"],
                "viz_type": choose_viz_type(record["viz_rows"], rng),
                "source_type": record["source_type"],
            }
        )
    return pool


def build_task_b_pool(question_records: list[dict]) -> list[dict]:
    pool: list[dict] = []
    for record in question_records:
        correct = [
            row
            for row in record["viz_rows"]
            if float(row.get("exact_match", 0.0)) >= 1.0
        ]
        wrong = [
            row
            for row in record["viz_rows"]
            if float(row.get("exact_match", 0.0)) <= 0.0
        ]
        if not correct or not wrong:
            continue

        viz_a = sorted(correct, key=lambda x: str(x["viz_type"]))[0]
        viz_b = sorted(wrong, key=lambda x: str(x["viz_type"]))[0]
        pool.append(
            {
                "question_id": record["question_id"],
                "question": record["question"],
                "answer": record["answer"],
                "viz_a": str(viz_a["viz_type"]),
                "viz_b": str(viz_b["viz_type"]),
                "em_a": float(viz_a.get("exact_match", 0.0)),
                "em_b": float(viz_b.get("exact_match", 0.0)),
                "modality": record["modality"],
                "difficulty": record["difficulty"],
                "source_type": record["source_type"],
            }
        )
    return pool


def sample_stratified(items: list[dict], n: int, rng: random.Random) -> list[dict]:
    if n > len(items):
        raise ValueError(
            f"Requested {n} items but only {len(items)} candidates are available."
        )

    buckets: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for item in items:
        key = (item["modality"], item["difficulty"], item["source_type"])
        buckets[key].append(item)

    for key in buckets:
        rng.shuffle(buckets[key])

    stratum_keys = [
        (modality, difficulty, source_type)
        for modality in MODALITIES
        for difficulty in DIFFICULTIES
        for source_type in SOURCE_TYPES
    ]
    stratum_targets = allocate_even(n, tuple("|".join(key) for key in stratum_keys))

    selected: list[dict] = []
    selected_ids: set[str] = set()

    for key in stratum_keys:
        stratum_name = "|".join(key)
        target = stratum_targets[stratum_name]
        pool = buckets.get(key, [])
        take = min(target, len(pool))
        for item in pool[:take]:
            selected.append(item)
            selected_ids.add(item["question_id"])

    if len(selected) == n:
        return selected

    modality_targets = allocate_even(n, MODALITIES)
    difficulty_targets = allocate_even(n, DIFFICULTIES)
    source_targets = allocate_even(n, SOURCE_TYPES)

    selected_modality = Counter(item["modality"] for item in selected)
    selected_difficulty = Counter(item["difficulty"] for item in selected)
    selected_source = Counter(item["source_type"] for item in selected)
    selected_stratum = Counter(
        "|".join((item["modality"], item["difficulty"], item["source_type"]))
        for item in selected
    )

    remaining = [item for item in items if item["question_id"] not in selected_ids]
    rng.shuffle(remaining)

    while len(selected) < n and remaining:
        best_index = -1
        best_score = -1
        for idx, item in enumerate(remaining):
            stratum_name = "|".join(
                (item["modality"], item["difficulty"], item["source_type"])
            )
            score = 0
            score += 4 * max(
                0, stratum_targets[stratum_name] - selected_stratum[stratum_name]
            )
            score += 2 * max(
                0,
                modality_targets[item["modality"]]
                - selected_modality[item["modality"]],
            )
            score += 2 * max(
                0,
                difficulty_targets[item["difficulty"]]
                - selected_difficulty[item["difficulty"]],
            )
            score += max(
                0,
                source_targets[item["source_type"]]
                - selected_source[item["source_type"]],
            )

            if score > best_score:
                best_score = score
                best_index = idx

        if best_index < 0:
            break

        chosen = remaining.pop(best_index)
        selected.append(chosen)
        selected_ids.add(chosen["question_id"])
        selected_modality[chosen["modality"]] += 1
        selected_difficulty[chosen["difficulty"]] += 1
        selected_source[chosen["source_type"]] += 1
        selected_stratum[
            "|".join((chosen["modality"], chosen["difficulty"], chosen["source_type"]))
        ] += 1

    if len(selected) < n:
        raise ValueError(
            f"Unable to produce a sample of size {n}; only got {len(selected)}."
        )

    return selected


def strip_internal_fields(rows: list[dict]) -> list[dict]:
    cleaned: list[dict] = []
    for row in rows:
        cleaned.append(
            {
                k: v
                for k, v in row.items()
                if k not in {"source_type", "modality", "difficulty"}
            }
        )
    return cleaned


def write_evaluation_form(path: Path, task_a: list[dict], task_b: list[dict]) -> None:
    lines: list[str] = []
    lines.append("# StructViz-Bench Human Evaluation Form")
    lines.append("")
    lines.append("## Instructions")
    lines.append("")
    lines.append("- Evaluators: 3")
    lines.append("- Use the provided JSONL files to inspect each item in order.")
    lines.append("- Record one judgment per item in the tables below.")
    lines.append(
        "- Keep notes concise and specific when selecting Ambiguous/Incorrect/Implausible/Unclear."
    )
    lines.append("")
    lines.append("## Task A - Answer Correctness Verification")
    lines.append("")
    lines.append(
        "For each item, inspect the visualization, question, and ground-truth answer."
    )
    lines.append("")
    lines.append("Rating scale:")
    lines.append("- Correct: Ground-truth answer is unambiguously correct.")
    lines.append(
        "- Ambiguous: Answer is defensible but the item allows multiple interpretations."
    )
    lines.append("- Incorrect: Ground-truth answer appears wrong.")
    lines.append("")
    lines.append(
        "| # | question_id | modality | difficulty | source | viz_type | Evaluator judgment | Notes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for i, row in enumerate(task_a, start=1):
        lines.append(
            f"| {i} | {row['question_id']} | {row['modality']} | {row['difficulty']} | {row['source']} | {row['viz_type']} |  |  |"
        )

    lines.append("")
    lines.append("## Task B - Visualization Sensitivity Plausibility")
    lines.append("")
    lines.append(
        "For each pair, decide whether it is plausible that a model answers correctly from viz_a but"
    )
    lines.append("incorrectly from viz_b.")
    lines.append("")
    lines.append("Rating scale:")
    lines.append(
        "- Plausible: One visualization is meaningfully harder for the same question."
    )
    lines.append(
        "- Implausible: Both visualizations make the answer similarly obvious."
    )
    lines.append("- Unclear: Insufficient clarity to decide.")
    lines.append("")
    lines.append(
        "| # | question_id | viz_a (correct) | viz_b (wrong) | em_a | em_b | Evaluator judgment | Notes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for i, row in enumerate(task_b, start=1):
        lines.append(
            f"| {i} | {row['question_id']} | {row['viz_a']} | {row['viz_b']} | {row['em_a']} | {row['em_b']} |  |  |"
        )

    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"- Random seed: {SEED}")
    lines.append(f"- Task A items: {len(task_a)}")
    lines.append(f"- Task B pairs: {len(task_b)}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_distribution(label: str, rows: list[dict]) -> None:
    modality_counts = Counter(row["modality"] for row in rows)
    difficulty_counts = Counter(row["difficulty"] for row in rows)
    source_counts = Counter(row["source_type"] for row in rows)
    print(f"{label} modality distribution: {dict(modality_counts)}")
    print(f"{label} difficulty distribution: {dict(difficulty_counts)}")
    print(f"{label} source-type distribution: {dict(source_counts)}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    results_file = project_root / "results" / "full_gpt4o.jsonl"
    output_dir = project_root / "results" / "human_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    rows = load_jsonl(results_file)
    question_records = build_question_records(rows)

    task_a_pool = build_task_a_pool(question_records, rng)
    task_a_sample = sample_stratified(task_a_pool, TASK_A_SIZE, rng)

    task_b_pool = build_task_b_pool(question_records)
    task_b_sample = sample_stratified(task_b_pool, TASK_B_SIZE, rng)

    task_a_output = [
        {
            "question_id": row["question_id"],
            "question": row["question"],
            "answer": row["answer"],
            "modality": row["modality"],
            "difficulty": row["difficulty"],
            "source": row["source"],
            "viz_type": row["viz_type"],
        }
        for row in task_a_sample
    ]

    task_b_output = strip_internal_fields(task_b_sample)

    write_jsonl(output_dir / "task_a_items.jsonl", task_a_output)
    write_jsonl(output_dir / "task_b_pairs.jsonl", task_b_output)
    write_evaluation_form(
        output_dir / "evaluation_form.md", task_a_sample, task_b_sample
    )

    print_distribution("Task A", task_a_sample)
    print_distribution("Task B", task_b_sample)
    print(f"Wrote: {output_dir / 'task_a_items.jsonl'}")
    print(f"Wrote: {output_dir / 'task_b_pairs.jsonl'}")
    print(f"Wrote: {output_dir / 'evaluation_form.md'}")


if __name__ == "__main__":
    main()
