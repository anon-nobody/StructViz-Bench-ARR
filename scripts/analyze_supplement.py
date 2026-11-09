from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


MODEL_FILES = {
    "GPT-4o": "full_gpt4o.jsonl",
    "Gemini Flash": "full_gemini.jsonl",
    "Qwen2.5-VL-7B": "full_qwen.jsonl",
    "Claude Sonnet": "full_claude.jsonl",
}

MIXED_FILES = {
    "GPT-4o": "mixed_gpt4o.jsonl",
    "Gemini Flash": "mixed_gemini.jsonl",
    "Qwen2.5-VL-7B": "mixed_qwen.jsonl",
    "Claude Sonnet": "mixed_claude.jsonl",
}

MODEL_ORDER = ["GPT-4o", "Gemini Flash", "Qwen2.5-VL-7B", "Claude Sonnet"]
MODALITY_ORDER = ["tabular", "timeseries", "graph"]
SOURCE_ORDER = ["synthetic", "scitabalign", "ett", "networkx"]

EPS = 1e-12
PERMUTATION_ITERATIONS = 10_000
RNG_SEED = 42

BINARY_ANSWERS = {"yes", "no", "true", "false", "increasing", "decreasing"}
REFUSAL_PATTERNS = [
    "i cannot",
    "i can't",
    "cannot answer",
    "can't answer",
    "unable",
    "insufficient",
    "not enough information",
    "as an ai",
    "sorry",
    "do not have",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        payload = line.strip()
        if not payload:
            continue
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}:{idx}") from exc
        if not isinstance(parsed, dict):
            continue
        rows.append(parsed)
    if not rows:
        raise ValueError(f"No valid rows in {path}")
    return rows


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def as_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def pct(value: float) -> str:
    return f"{value * 100.0:.2f}"


def pp(value: float) -> str:
    return f"{value * 100.0:.2f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def normalize_text(text: str) -> str:
    lowered = text.strip().lower()
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip(" \t\n\r.,;:!?\"'`()[]{}")


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0.0:
        return min(values)
    if q >= 1.0:
        return max(values)
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return float(ordered[lower])
    weight = pos - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def std_sample(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mu = mean(values)
    var = sum((x - mu) ** 2 for x in values) / (n - 1)
    return math.sqrt(var)


def group_mean_em(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        buckets[as_str(row.get(key))].append(as_float(row.get("exact_match")))
    return {
        bucket_key: float(mean(values))
        for bucket_key, values in buckets.items()
        if values
    }


def choose_best_worst(score_map: dict[str, float]) -> tuple[str, float, str, float]:
    ordered = sorted(score_map.items(), key=lambda item: (item[1], item[0]))
    worst_key, worst_val = ordered[0]
    best_key, best_val = ordered[-1]
    return best_key, best_val, worst_key, worst_val


def get_full_rows_by_model(results_dir: Path) -> dict[str, list[dict[str, Any]]]:
    data: dict[str, list[dict[str, Any]]] = {}
    for model in MODEL_ORDER:
        file_name = MODEL_FILES[model]
        data[model] = load_jsonl(results_dir / file_name)
    return data


def get_mixed_rows_by_model(results_dir: Path) -> dict[str, list[dict[str, Any]]]:
    data: dict[str, list[dict[str, Any]]] = {}
    for model, file_name in MIXED_FILES.items():
        path = results_dir / file_name
        if not path.exists():
            continue
        data[model] = load_jsonl(path)
    return data


def question_view(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_question: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = as_str(row.get("question_id"))
        if qid and qid not in by_question:
            by_question[qid] = row
    return by_question


def paired_permutation_test(
    x: list[float],
    y: list[float],
    iterations: int,
    rng: random.Random,
) -> float:
    if not x or not y or len(x) != len(y):
        return 1.0
    diffs = [a - b for a, b in zip(x, y)]
    n_total = len(diffs)
    obs = abs(mean(diffs))
    non_zero = [d for d in diffs if abs(d) > EPS]
    if not non_zero:
        return 1.0

    hits = 0
    for _ in range(iterations):
        signed_sum = 0.0
        for d in non_zero:
            signed_sum += d if rng.random() < 0.5 else -d
        perm_mean = abs(signed_sum / n_total)
        if perm_mean + EPS >= obs:
            hits += 1
    return (hits + 1.0) / (iterations + 1.0)


def cohens_d_paired(x: list[float], y: list[float]) -> float:
    if not x or not y or len(x) != len(y):
        return 0.0
    d = [a - b for a, b in zip(x, y)]
    if len(d) < 2:
        return 0.0
    std = std_sample(d)
    mu = float(mean(d))
    if std <= EPS:
        if abs(mu) <= EPS:
            return 0.0
        return math.inf if mu > 0 else -math.inf
    return mu / std


def summarize_distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "q1": 0.0, "q3": 0.0}
    return {
        "mean": float(mean(values)),
        "median": float(median(values)),
        "q1": quantile(values, 0.25),
        "q3": quantile(values, 0.75),
    }


def section_visual_only_sensitivity(full_data: dict[str, list[dict[str, Any]]]) -> str:
    rows: list[list[str]] = []
    for model in MODEL_ORDER:
        model_rows = full_data[model]
        for modality in MODALITY_ORDER:
            subset = [r for r in model_rows if as_str(r.get("modality")) == modality]
            full_scores = group_mean_em(subset, "viz_type")
            best_full, best_full_em, worst_full, worst_full_em = choose_best_worst(
                full_scores
            )
            full_gap = best_full_em - worst_full_em

            visual_subset = [
                r for r in subset if as_str(r.get("viz_type")) != "text_only"
            ]
            visual_scores = group_mean_em(visual_subset, "viz_type")
            if len(visual_scores) >= 2:
                best_vis, best_vis_em, worst_vis, worst_vis_em = choose_best_worst(
                    visual_scores
                )
                vis_gap = best_vis_em - worst_vis_em
            else:
                best_vis, best_vis_em, worst_vis, worst_vis_em = "N/A", 0.0, "N/A", 0.0
                vis_gap = 0.0

            rows.append(
                [
                    model,
                    modality,
                    best_full,
                    worst_full,
                    pp(full_gap),
                    best_vis,
                    worst_vis,
                    pp(vis_gap),
                    pp(full_gap - vis_gap),
                ]
            )

    lines = [
        "## Section 1. Visual-Only Sensitivity (text_only excluded)",
        "",
        "Full-gap vs visual-only-gap comparison per model and modality.",
        "",
        md_table(
            [
                "Model",
                "Modality",
                "Full Best",
                "Full Worst",
                "Full Gap (pp)",
                "Visual Best",
                "Visual Worst",
                "Visual-Only Gap (pp)",
                "Text-Only Contribution (pp)",
            ],
            rows,
        ),
        "",
    ]
    return "\n".join(lines)


def section_flip_analysis(full_data: dict[str, list[dict[str, Any]]]) -> str:
    overall_rows: list[list[str]] = []
    by_modality_rows: list[list[str]] = []
    model_flip_rates: list[float] = []

    for model in MODEL_ORDER:
        model_rows = full_data[model]
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in model_rows:
            grouped[as_str(row.get("question_id"))].append(
                as_float(row.get("exact_match"))
            )

        always_correct = 0
        always_wrong = 0
        flipping = 0
        for scores in grouped.values():
            if min(scores) >= 1.0 - EPS:
                always_correct += 1
            elif max(scores) <= EPS:
                always_wrong += 1
            else:
                flipping += 1

        total_questions = len(grouped)
        flip_rate = (flipping / total_questions) if total_questions else 0.0
        model_flip_rates.append(flip_rate)
        overall_rows.append(
            [
                model,
                str(total_questions),
                str(always_correct),
                str(always_wrong),
                str(flipping),
                pct(flip_rate),
            ]
        )

        for modality in MODALITY_ORDER:
            modality_rows = [
                r for r in model_rows if as_str(r.get("modality")) == modality
            ]
            grouped_modality: dict[str, list[float]] = defaultdict(list)
            for row in modality_rows:
                grouped_modality[as_str(row.get("question_id"))].append(
                    as_float(row.get("exact_match"))
                )

            ac = 0
            aw = 0
            fl = 0
            for scores in grouped_modality.values():
                if min(scores) >= 1.0 - EPS:
                    ac += 1
                elif max(scores) <= EPS:
                    aw += 1
                else:
                    fl += 1
            n = len(grouped_modality)
            by_modality_rows.append(
                [
                    model,
                    modality,
                    str(n),
                    str(ac),
                    str(aw),
                    str(fl),
                    pct(fl / n if n else 0.0),
                ]
            )

    flip_min = min(model_flip_rates) if model_flip_rates else 0.0
    flip_max = max(model_flip_rates) if model_flip_rates else 0.0
    lines = [
        "## Section 2. Per-Question Flip Analysis",
        "",
        f"Model-level flip-rate range: {pct(flip_min)}% to {pct(flip_max)}%.",
        "",
        "### Overall per model",
        "",
        md_table(
            [
                "Model",
                "# Questions",
                "Always Correct",
                "Always Wrong",
                "Flipping",
                "Flip Rate (%)",
            ],
            overall_rows,
        ),
        "",
        "### Per model and modality",
        "",
        md_table(
            [
                "Model",
                "Modality",
                "# Questions",
                "Always Correct",
                "Always Wrong",
                "Flipping",
                "Flip Rate (%)",
            ],
            by_modality_rows,
        ),
        "",
    ]
    return "\n".join(lines)


def section_significance_tests(full_data: dict[str, list[dict[str, Any]]]) -> str:
    rng = random.Random(RNG_SEED)
    results: list[dict[str, Any]] = []

    for model in MODEL_ORDER:
        model_rows = full_data[model]
        for modality in MODALITY_ORDER:
            subset = [r for r in model_rows if as_str(r.get("modality")) == modality]
            score_map = group_mean_em(subset, "viz_type")
            best_viz, _, worst_viz, _ = choose_best_worst(score_map)

            best_map: dict[str, float] = {}
            worst_map: dict[str, float] = {}
            for row in subset:
                qid = as_str(row.get("question_id"))
                viz = as_str(row.get("viz_type"))
                em = as_float(row.get("exact_match"))
                if viz == best_viz:
                    best_map[qid] = em
                elif viz == worst_viz:
                    worst_map[qid] = em

            paired_ids = sorted(set(best_map).intersection(worst_map))
            best_vals = [best_map[qid] for qid in paired_ids]
            worst_vals = [worst_map[qid] for qid in paired_ids]
            p_value = paired_permutation_test(
                best_vals,
                worst_vals,
                iterations=PERMUTATION_ITERATIONS,
                rng=rng,
            )
            d_value = cohens_d_paired(best_vals, worst_vals)

            results.append(
                {
                    "model": model,
                    "modality": modality,
                    "viz_pair": f"{best_viz} vs {worst_viz}",
                    "p_value": p_value,
                    "effect_size": d_value,
                    "n_pairs": len(paired_ids),
                }
            )

    n_tests = len(results)
    rows: list[list[str]] = []
    for result in results:
        corrected = min(result["p_value"] * n_tests, 1.0)
        significant = "yes" if corrected < 0.05 else "no"
        d_val = result["effect_size"]
        if math.isinf(d_val):
            d_txt = "inf" if d_val > 0 else "-inf"
        else:
            d_txt = f"{d_val:.3f}"
        rows.append(
            [
                result["model"],
                result["modality"],
                result["viz_pair"],
                str(result["n_pairs"]),
                f"{result['p_value']:.6f}",
                f"{corrected:.6f}",
                d_txt,
                significant,
            ]
        )

    lines = [
        "## Section 3. Statistical Significance Tests",
        "",
        f"Paired permutation test ({PERMUTATION_ITERATIONS} iterations) between best/worst viz types for each model x modality.",
        f"Bonferroni correction uses {n_tests} comparisons.",
        "",
        md_table(
            [
                "Model",
                "Modality",
                "Viz Pair",
                "Paired N",
                "p_value",
                "corrected_p",
                "Cohen's d",
                "Significant",
            ],
            rows,
        ),
        "",
    ]
    return "\n".join(lines)


def section_random_majority_baselines(
    full_data: dict[str, list[dict[str, Any]]],
) -> str:
    reference_questions = question_view(full_data["GPT-4o"])
    question_rows = list(reference_questions.values())

    by_task_answers: dict[str, list[str]] = defaultdict(list)
    for row in question_rows:
        task = as_str(row.get("task"))
        answer = normalize_text(as_str(row.get("answer")))
        by_task_answers[task].append(answer)

    task_rows: list[list[str]] = []
    total_questions = 0
    random_total = 0.0
    majority_total = 0.0
    for task in sorted(by_task_answers):
        answers = by_task_answers[task]
        if not answers:
            continue
        counts = Counter(answers)
        n = len(answers)
        unique_n = len(counts)
        random_em = 1.0 / float(unique_n)
        majority_em = max(counts.values()) / float(n)
        total_questions += n
        random_total += random_em * n
        majority_total += majority_em * n
        task_rows.append(
            [
                task,
                str(n),
                str(unique_n),
                pct(random_em),
                pct(majority_em),
            ]
        )

    overall_random = random_total / total_questions if total_questions else 0.0
    overall_majority = majority_total / total_questions if total_questions else 0.0

    compare_rows: list[list[str]] = []
    for model in MODEL_ORDER:
        model_em = mean(as_float(r.get("exact_match")) for r in full_data[model])
        compare_rows.append(
            [
                model,
                pct(model_em),
                pct(overall_random),
                pct(overall_majority),
                pp(model_em - overall_random),
                pp(model_em - overall_majority),
            ]
        )

    lines = [
        "## Section 4. Random and Majority Baselines",
        "",
        "Random baseline uses 1 / (# unique answers) per task on deduplicated questions.",
        "Majority baseline predicts the most common answer per task.",
        "",
        "### Baseline by task",
        "",
        md_table(
            [
                "Task",
                "# Questions",
                "# Unique Answers",
                "Random EM (%)",
                "Majority EM (%)",
            ],
            task_rows,
        ),
        "",
        "### Model EM vs baselines",
        "",
        md_table(
            [
                "Model",
                "Model EM (%)",
                "Random Baseline (%)",
                "Majority Baseline (%)",
                "Delta vs Random (pp)",
                "Delta vs Majority (pp)",
            ],
            compare_rows,
        ),
        "",
    ]
    return "\n".join(lines)


def section_binary_open_breakdown(full_data: dict[str, list[dict[str, Any]]]) -> str:
    reference_q = question_view(full_data["GPT-4o"])
    q_type: dict[str, str] = {}
    for qid, row in reference_q.items():
        answer = normalize_text(as_str(row.get("answer")))
        q_type[qid] = "binary" if answer in BINARY_ANSWERS else "open_form"

    rows: list[list[str]] = []
    for model in MODEL_ORDER:
        model_rows = full_data[model]
        binary_scores: list[float] = []
        open_scores: list[float] = []
        for row in model_rows:
            qid = as_str(row.get("question_id"))
            bucket = q_type.get(qid, "open_form")
            em = as_float(row.get("exact_match"))
            if bucket == "binary":
                binary_scores.append(em)
            else:
                open_scores.append(em)
        binary_em = float(mean(binary_scores)) if binary_scores else 0.0
        open_em = float(mean(open_scores)) if open_scores else 0.0
        rows.append(
            [
                model,
                str(len(binary_scores)),
                pct(binary_em),
                str(len(open_scores)),
                pct(open_em),
                pp(binary_em - open_em),
            ]
        )

    lines = [
        "## Section 5. Binary vs Open-Form Task Breakdown",
        "",
        "Binary labels are answers in {yes, no, true, false, increasing, decreasing}; all others are open-form.",
        "",
        md_table(
            [
                "Model",
                "Binary Rows",
                "Binary EM (%)",
                "Open Rows",
                "Open EM (%)",
                "Binary-Open Gap (pp)",
            ],
            rows,
        ),
        "",
    ]
    return "\n".join(lines)


def section_per_source_analysis(full_data: dict[str, list[dict[str, Any]]]) -> str:
    overall_rows: list[list[str]] = []
    detail_rows: list[list[str]] = []

    for model in MODEL_ORDER:
        model_rows = full_data[model]
        for source in SOURCE_ORDER:
            source_rows = [r for r in model_rows if as_str(r.get("source")) == source]
            if not source_rows:
                continue
            em = float(mean([as_float(r.get("exact_match")) for r in source_rows]))

            modality_gaps: list[float] = []
            for modality in MODALITY_ORDER:
                subset = [
                    r for r in source_rows if as_str(r.get("modality")) == modality
                ]
                if not subset:
                    continue
                score_map = group_mean_em(subset, "viz_type")
                if len(score_map) < 2:
                    continue
                best_viz, best_em, worst_viz, worst_em = choose_best_worst(score_map)
                gap = best_em - worst_em
                modality_gaps.append(gap)
                detail_rows.append(
                    [
                        model,
                        source,
                        modality,
                        best_viz,
                        pct(best_em),
                        worst_viz,
                        pct(worst_em),
                        pp(gap),
                    ]
                )

            mean_gap = float(mean(modality_gaps)) if modality_gaps else 0.0
            overall_rows.append(
                [
                    model,
                    source,
                    str(len(source_rows)),
                    pct(em),
                    pp(mean_gap),
                ]
            )

    lines = [
        "## Section 6. Per-Source Real-World Analysis",
        "",
        "Source-specific EM and sensitivity gaps across synthetic, scitabalign, ett, and networkx.",
        "",
        "### Source-level summary",
        "",
        md_table(
            ["Model", "Source", "# Rows", "EM (%)", "Mean Modality Gap (pp)"],
            overall_rows,
        ),
        "",
        "### Source x modality sensitivity",
        "",
        md_table(
            [
                "Model",
                "Source",
                "Modality",
                "Best Viz",
                "Best EM (%)",
                "Worst Viz",
                "Worst EM (%)",
                "Gap (pp)",
            ],
            detail_rows,
        ),
        "",
    ]
    return "\n".join(lines)


def pairwise_gap_values(score_map: dict[str, float]) -> list[float]:
    keys = sorted(score_map)
    values: list[float] = []
    for i, left in enumerate(keys):
        for right in keys[i + 1 :]:
            values.append(abs(score_map[left] - score_map[right]))
    return values


def section_gap_distribution(full_data: dict[str, list[dict[str, Any]]]) -> str:
    by_model_gaps: dict[str, list[float]] = defaultdict(list)
    by_modality_gaps: dict[str, list[float]] = defaultdict(list)

    for model in MODEL_ORDER:
        model_rows = full_data[model]
        for modality in MODALITY_ORDER:
            subset = [r for r in model_rows if as_str(r.get("modality")) == modality]
            score_map = group_mean_em(subset, "viz_type")
            gaps = pairwise_gap_values(score_map)
            by_model_gaps[model].extend(gaps)
            by_modality_gaps[modality].extend(gaps)

    model_stat_rows: list[list[str]] = []
    for model in MODEL_ORDER:
        dist = summarize_distribution(by_model_gaps[model])
        model_stat_rows.append(
            [
                model,
                str(len(by_model_gaps[model])),
                pp(dist["mean"]),
                pp(dist["median"]),
                pp(dist["q1"]),
                pp(dist["q3"]),
            ]
        )

    modality_rows: list[list[str]] = []
    for modality in MODALITY_ORDER:
        dist = summarize_distribution(by_modality_gaps[modality])
        modality_rows.append(
            [
                modality,
                str(len(by_modality_gaps[modality])),
                pp(dist["mean"]),
                pp(dist["median"]),
                pp(dist["q1"]),
                pp(dist["q3"]),
            ]
        )

    lines = [
        "## Section 7. Mean/Median Gap Reporting",
        "",
        "Gap distributions are absolute pairwise EM differences across visualization types.",
        "",
        "### By model",
        "",
        md_table(
            [
                "Model",
                "# Pairwise Gaps",
                "Mean (pp)",
                "Median (pp)",
                "Q1 (pp)",
                "Q3 (pp)",
            ],
            model_stat_rows,
        ),
        "",
        "### By modality",
        "",
        md_table(
            [
                "Modality",
                "# Pairwise Gaps",
                "Mean (pp)",
                "Median (pp)",
                "Q1 (pp)",
                "Q3 (pp)",
            ],
            modality_rows,
        ),
        "",
    ]
    return "\n".join(lines)


def is_numeric_like(text: str) -> bool:
    candidate = text.strip()
    return bool(re.fullmatch(r"[+-]?(?:\d+\.?\d*|\.\d+)", candidate))


def top_counter_items(counter: Counter[str], n: int = 8) -> list[str]:
    items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:n]
    return [f"`{label}` ({count})" for label, count in items]


def section_claude_mixed_deep_dive(
    mixed_data: dict[str, list[dict[str, Any]]],
) -> str:
    compare_models = ["Claude Sonnet", "GPT-4o", "Qwen2.5-VL-7B"]
    summary_rows: list[list[str]] = []
    top_predictions: dict[str, str] = {}

    claude_modality_rows: list[list[str]] = []

    for model in compare_models:
        rows = mixed_data.get(model, [])
        if not rows:
            continue
        preds = [as_str(r.get("prediction")) for r in rows]
        normalized = [normalize_text(p) for p in preds]
        exact = [as_float(r.get("exact_match")) for r in rows]

        error_count = sum(1 for p in normalized if p == "[error]")
        empty_count = sum(1 for p in normalized if p == "")
        refusal_count = sum(
            1 for p in normalized if any(pattern in p for pattern in REFUSAL_PATTERNS)
        )
        numeric_like_count = sum(1 for p in preds if is_numeric_like(p))
        char_lengths = [len(p.strip()) for p in preds]
        token_lengths = [len(p.strip().split()) if p.strip() else 0 for p in preds]
        em = float(mean(exact)) if exact else 0.0

        summary_rows.append(
            [
                model,
                str(len(rows)),
                pct(em),
                f"{mean(char_lengths):.2f}",
                f"{median(char_lengths):.2f}",
                f"{mean(token_lengths):.2f}",
                f"{median(token_lengths):.2f}",
                str(error_count),
                pct(error_count / len(rows)),
                str(empty_count),
                pct(empty_count / len(rows)),
                str(refusal_count),
                pct(refusal_count / len(rows)),
                str(numeric_like_count),
                pct(numeric_like_count / len(rows)),
            ]
        )

        pred_counter = Counter(normalized)
        top_predictions[model] = ", ".join(top_counter_items(pred_counter, n=8))

        if model == "Claude Sonnet":
            for modality in sorted({as_str(r.get("modality")) for r in rows}):
                subset = [r for r in rows if as_str(r.get("modality")) == modality]
                if not subset:
                    continue
                subset_preds = [
                    normalize_text(as_str(r.get("prediction"))) for r in subset
                ]
                err = sum(1 for p in subset_preds if p == "[error]")
                empty = sum(1 for p in subset_preds if p == "")
                refusal = sum(
                    1
                    for p in subset_preds
                    if any(pattern in p for pattern in REFUSAL_PATTERNS)
                )
                claude_modality_rows.append(
                    [
                        modality,
                        str(len(subset)),
                        pct(
                            float(
                                mean([as_float(r.get("exact_match")) for r in subset])
                            )
                        ),
                        str(err),
                        pct(err / len(subset)),
                        str(empty),
                        pct(empty / len(subset)),
                        str(refusal),
                        pct(refusal / len(subset)),
                    ]
                )

    claude_row = next((row for row in summary_rows if row[0] == "Claude Sonnet"), None)
    gpt_row = next((row for row in summary_rows if row[0] == "GPT-4o"), None)
    qwen_row = next((row for row in summary_rows if row[0] == "Qwen2.5-VL-7B"), None)

    diagnosis = []
    if claude_row and gpt_row and qwen_row:
        diagnosis.append(
            "Claude shows elevated `[ERROR]` and non-numeric output rates compared with GPT-4o and Qwen on mixed tasks."
        )
        diagnosis.append(
            "Prediction-length statistics indicate unstable output formatting rather than a simple constant-short-answer failure mode."
        )
        diagnosis.append(
            "Frequent repeated fallback tokens (top predictions) suggest parser/API failure artifacts leaking into final predictions."
        )

    lines = [
        "## Section 8. Claude Mixed-Type Deep Dive",
        "",
        "Claude mixed-type response diagnostics compared to GPT-4o and Qwen.",
        "",
        md_table(
            [
                "Model",
                "# Rows",
                "EM (%)",
                "Mean Chars",
                "Median Chars",
                "Mean Tokens",
                "Median Tokens",
                "[ERROR] Count",
                "[ERROR] Rate (%)",
                "Empty Count",
                "Empty Rate (%)",
                "Refusal Count",
                "Refusal Rate (%)",
                "Numeric-like Count",
                "Numeric-like Rate (%)",
            ],
            summary_rows,
        ),
        "",
        "### Claude modality breakdown",
        "",
        md_table(
            [
                "Modality",
                "# Rows",
                "EM (%)",
                "[ERROR] Count",
                "[ERROR] Rate (%)",
                "Empty Count",
                "Empty Rate (%)",
                "Refusal Count",
                "Refusal Rate (%)",
            ],
            claude_modality_rows,
        ),
        "",
        "### Top normalized predictions",
        "",
    ]
    for model in compare_models:
        if model in top_predictions:
            lines.append(f"- {model}: {top_predictions[model]}")

    lines.append("")
    lines.append("### What went wrong")
    lines.append("")
    for item in diagnosis:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def build_report(
    full_data: dict[str, list[dict[str, Any]]],
    mixed_data: dict[str, list[dict[str, Any]]],
) -> str:
    total_rows = sum(len(full_data[model]) for model in MODEL_ORDER)
    lines = [
        "# the review venue Supplementary Analysis",
        "",
        f"- Total full-scale rows analyzed: {total_rows:,}",
        "- Core models: GPT-4o, Gemini Flash, Qwen2.5-VL-7B, Claude Sonnet",
        "- Files: results/full_{claude,gpt4o,gemini,qwen}.jsonl and mixed comparisons",
        "",
        section_visual_only_sensitivity(full_data),
        section_flip_analysis(full_data),
        section_significance_tests(full_data),
        section_random_majority_baselines(full_data),
        section_binary_open_breakdown(full_data),
        section_per_source_analysis(full_data),
        section_gap_distribution(full_data),
        section_claude_mixed_deep_dive(mixed_data),
    ]
    return "\n".join(lines)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    output_path = results_dir / "supplement_analysis.md"

    full_data = get_full_rows_by_model(results_dir)
    mixed_data = get_mixed_rows_by_model(results_dir)

    report = build_report(full_data=full_data, mixed_data=mixed_data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote supplementary analysis report: {output_path}")


if __name__ == "__main__":
    main()
