from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


CORE_MODEL_FILES = {
    "GPT-4o": "full_gpt4o.jsonl",
    "Gemini Flash": "full_gemini.jsonl",
    "Qwen2.5-VL-7B": "full_qwen.jsonl",
    "Claude Sonnet": "full_claude.jsonl",
}

SUPPLEMENTARY_MODEL_FILES = {
    "InternVL2.5-8B": "full_internvl.jsonl",
}

CORE_MODEL_ORDER = ["GPT-4o", "Gemini Flash", "Qwen2.5-VL-7B", "Claude Sonnet"]
SUPPLEMENTARY_MODEL_ORDER = ["InternVL2.5-8B"]
DIFFICULTY_RANK = {"1-hop": 1, "2-hop": 2, "3-hop": 3, "counterfactual": 4}


@dataclass(frozen=True)
class EvalRow:
    """One normalized evaluation row from a model result file."""

    model: str
    question_id: str
    question: str
    answer: str
    prediction: str
    modality: str
    source: str
    viz_type: str
    difficulty: str
    task: str
    exact_match: float
    f1: float
    numeric_accuracy: float

    @property
    def base_question_id(self) -> str:
        marker = "::difficulty="
        if marker in self.question_id:
            return self.question_id.split(marker, maxsplit=1)[0]
        return self.question_id


@dataclass(frozen=True)
class ExampleBlock:
    """Container for one qualitative contrast example in the appendix."""

    category: str
    title: str
    question_id: str
    question: str
    answer: str
    headers: list[str]
    rows: list[list[str]]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for qualitative extraction.

    Returns:
        Parsed namespace containing results directory and output path.
    """
    parser = argparse.ArgumentParser(
        description="Extract high-contrast qualitative examples and write LaTeX appendix block.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing full_*.jsonl files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/qualitative_examples.tex"),
        help="Output .tex file path.",
    )
    parser.add_argument(
        "--include-supplementary-local",
        action="store_true",
        help="Include caveated supplementary local baseline results (InternVL2.5-8B only).",
    )
    return parser.parse_args()


def _parse_json_line(line: str) -> dict[str, object]:
    start = line.find("{")
    if start == -1:
        raise ValueError("Malformed line without JSON object.")
    payload = json.loads(line[start:])
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object per line.")
    return payload


def _to_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Cannot convert to float: {value!r}")


def load_results(
    results_dir: Path,
    include_supplementary_local: bool,
) -> tuple[list[EvalRow], list[str]]:
    """Load and normalize all full-scale evaluation rows.

    Args:
        results_dir: Directory with all `full_*.jsonl` files.

    Returns:
        Flat list of rows with model names attached.
    """
    rows: list[EvalRow] = []
    model_files = dict(CORE_MODEL_FILES)
    model_order = list(CORE_MODEL_ORDER)
    if include_supplementary_local:
        model_files.update(SUPPLEMENTARY_MODEL_FILES)
        model_order.extend(SUPPLEMENTARY_MODEL_ORDER)

    for model, file_name in model_files.items():
        file_path = results_dir / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Missing results file: {file_path}")

        for raw_line in file_path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            item = _parse_json_line(raw_line)
            rows.append(
                EvalRow(
                    model=model,
                    question_id=str(item["question_id"]),
                    question=str(item["question"]),
                    answer=str(item["answer"]),
                    prediction=str(item["prediction"]),
                    modality=str(item["modality"]),
                    source=str(item["source"]),
                    viz_type=str(item["viz_type"]),
                    difficulty=str(item["difficulty"]),
                    task=str(item["task"]),
                    exact_match=_to_float(item["exact_match"]),
                    f1=_to_float(item["f1"]),
                    numeric_accuracy=_to_float(item["numeric_accuracy"]),
                )
            )
    return rows, model_order


def _status(row: EvalRow) -> str:
    return "\\cmark" if row.exact_match >= 1.0 else "\\xmark"


def _truncate_text(text: str, max_chars: int = 80) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _latex_escape(text: str) -> str:
    mapping = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    return "".join(mapping.get(char, char) for char in text)


def _build_indices(
    rows: list[EvalRow],
) -> tuple[
    dict[tuple[str, str], list[EvalRow]],
    dict[tuple[str, str], list[EvalRow]],
    dict[tuple[str, str, str, str], list[EvalRow]],
]:
    by_question_model: dict[tuple[str, str], list[EvalRow]] = {}
    by_question_viz: dict[tuple[str, str], list[EvalRow]] = {}
    by_model_base_modality_viz: dict[tuple[str, str, str, str], list[EvalRow]] = {}

    for row in rows:
        by_question_model.setdefault((row.question_id, row.model), []).append(row)
        by_question_viz.setdefault((row.question_id, row.viz_type), []).append(row)
        by_model_base_modality_viz.setdefault(
            (row.model, row.base_question_id, row.modality, row.viz_type), []
        ).append(row)

    return by_question_model, by_question_viz, by_model_base_modality_viz


def _select_visualization_sensitivity(
    by_question_model: dict[tuple[str, str], list[EvalRow]],
    limit: int,
) -> list[ExampleBlock]:
    def candidate_score(good_row: EvalRow, bad_row: EvalRow) -> float:
        priority = 100.0
        if (
            good_row.modality == "tabular"
            and good_row.viz_type == "text_only"
            and bad_row.viz_type == "scatter_plot"
        ):
            priority = 500.0
        elif (
            good_row.modality == "timeseries"
            and good_row.viz_type in {"line_plot", "text_only"}
            and bad_row.viz_type in {"recurrence_plot", "gaf"}
        ):
            priority = 420.0
        penalty = (1.0 - bad_row.f1) + (1.0 - bad_row.numeric_accuracy)
        return priority + penalty

    candidates: list[tuple[float, EvalRow, EvalRow, str]] = []
    for group in by_question_model.values():
        good_rows = [row for row in group if row.exact_match >= 1.0]
        bad_rows = [row for row in group if row.exact_match < 1.0]
        if not good_rows or not bad_rows:
            continue
        for good_row in good_rows:
            for bad_row in bad_rows:
                if good_row.viz_type == bad_row.viz_type:
                    continue
                score = candidate_score(good_row, bad_row)
                if (
                    good_row.modality == "tabular"
                    and good_row.viz_type == "text_only"
                    and bad_row.viz_type == "scatter_plot"
                ):
                    label = "tabular"
                elif good_row.modality == "timeseries" and bad_row.viz_type in {
                    "recurrence_plot",
                    "gaf",
                }:
                    label = "timeseries"
                else:
                    label = "generic"
                candidates.append((score, good_row, bad_row, label))

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1].question_id,
            item[1].model,
            item[1].viz_type,
            item[2].viz_type,
        ),
        reverse=True,
    )

    selected: list[ExampleBlock] = []
    used_keys: set[tuple[str, str]] = set()

    for preferred_label in ["tabular", "timeseries"]:
        for _, good_row, bad_row, label in candidates:
            key = (good_row.question_id, good_row.model)
            if label != preferred_label or key in used_keys:
                continue
            title = (
                "Visualization sensitivity in tabular data"
                if preferred_label == "tabular"
                else "Time-series encoding sensitivity"
            )
            selected.append(
                ExampleBlock(
                    category="Visualization Sensitivity",
                    title=title,
                    question_id=good_row.question_id,
                    question=good_row.question,
                    answer=good_row.answer,
                    headers=["Model", "Viz Type", "Prediction", "Result"],
                    rows=[
                        [
                            good_row.model,
                            good_row.viz_type,
                            _truncate_text(good_row.prediction),
                            _status(good_row),
                        ],
                        [
                            bad_row.model,
                            bad_row.viz_type,
                            _truncate_text(bad_row.prediction),
                            _status(bad_row),
                        ],
                    ],
                )
            )
            used_keys.add(key)
            break

    for _, good_row, bad_row, _ in candidates:
        key = (good_row.question_id, good_row.model)
        if key in used_keys:
            continue
        selected.append(
            ExampleBlock(
                category="Visualization Sensitivity",
                title="Visualization sensitivity under controlled question semantics",
                question_id=good_row.question_id,
                question=good_row.question,
                answer=good_row.answer,
                headers=["Model", "Viz Type", "Prediction", "Result"],
                rows=[
                    [
                        good_row.model,
                        good_row.viz_type,
                        _truncate_text(good_row.prediction),
                        _status(good_row),
                    ],
                    [
                        bad_row.model,
                        bad_row.viz_type,
                        _truncate_text(bad_row.prediction),
                        _status(bad_row),
                    ],
                ],
            )
        )
        used_keys.add(key)
        if len(selected) >= limit:
            break

    return selected[:limit]


def _select_cross_model_disagreement(
    by_question_viz: dict[tuple[str, str], list[EvalRow]],
    limit: int,
    model_order: list[str],
) -> list[ExampleBlock]:
    candidates: list[tuple[float, list[EvalRow]]] = []
    for group in by_question_viz.values():
        if len(group) < 2:
            continue
        correct = [row for row in group if row.exact_match >= 1.0]
        wrong = [row for row in group if row.exact_match < 1.0]
        if not correct or not wrong:
            continue

        support = min(len(correct), len(wrong))
        severity = sum((1.0 - row.f1) + (1.0 - row.numeric_accuracy) for row in wrong)
        score = support * 10.0 + severity
        ordered_group = sorted(group, key=lambda row: model_order.index(row.model))
        candidates.append((score, ordered_group))

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1][0].question_id,
            item[1][0].viz_type,
        ),
        reverse=True,
    )

    selected: list[ExampleBlock] = []
    used_keys: set[tuple[str, str]] = set()
    for _, group in candidates:
        key = (group[0].question_id, group[0].viz_type)
        if key in used_keys:
            continue
        selected.append(
            ExampleBlock(
                category="Cross-Model Disagreement",
                title="Model-specific strengths and weaknesses on identical input",
                question_id=group[0].question_id,
                question=group[0].question,
                answer=group[0].answer,
                headers=["Model", "Viz Type", "Prediction", "Result"],
                rows=[
                    [
                        row.model,
                        row.viz_type,
                        _truncate_text(row.prediction),
                        _status(row),
                    ]
                    for row in group
                ],
            )
        )
        used_keys.add(key)
        if len(selected) >= limit:
            break
    return selected


def _select_difficulty_scaling(
    by_model_base_modality_viz: dict[tuple[str, str, str, str], list[EvalRow]],
    limit: int,
) -> list[ExampleBlock]:
    candidates: list[tuple[float, EvalRow, EvalRow]] = []

    for group in by_model_base_modality_viz.values():
        one_hop = [
            row for row in group if row.difficulty == "1-hop" and row.exact_match >= 1.0
        ]
        three_hop = [
            row for row in group if row.difficulty == "3-hop" and row.exact_match < 1.0
        ]
        if not one_hop or not three_hop:
            continue
        for easy_row in one_hop:
            for hard_row in three_hop:
                score = 2.0 + (1.0 - hard_row.f1) + (1.0 - hard_row.numeric_accuracy)
                candidates.append((score, easy_row, hard_row))

    by_model_modality_viz: dict[tuple[str, str, str], list[EvalRow]] = {}
    for group in by_model_base_modality_viz.values():
        for row in group:
            key = (row.model, row.modality, row.viz_type)
            by_model_modality_viz.setdefault(key, []).append(row)

    for group in by_model_modality_viz.values():
        one_hop = [
            row for row in group if row.difficulty == "1-hop" and row.exact_match >= 1.0
        ]
        three_hop = [
            row for row in group if row.difficulty == "3-hop" and row.exact_match < 1.0
        ]
        if not one_hop or not three_hop:
            continue
        for easy_row in one_hop:
            for hard_row in three_hop:
                if easy_row.question_id == hard_row.question_id:
                    continue
                task_bonus = 1.0 if easy_row.task == hard_row.task else 0.0
                score = (
                    task_bonus + (1.0 - hard_row.f1) + (1.0 - hard_row.numeric_accuracy)
                )
                candidates.append((score, easy_row, hard_row))

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1].model,
            item[1].modality,
            item[1].viz_type,
            item[1].task,
        ),
        reverse=True,
    )

    selected: list[ExampleBlock] = []
    used_keys: set[tuple[str, str, str, str]] = set()
    for _, easy_row, hard_row in candidates:
        key = (easy_row.model, easy_row.modality, easy_row.viz_type, easy_row.task)
        if key in used_keys:
            continue
        combined_question = (
            f"1-hop: {_truncate_text(easy_row.question, max_chars=120)} | "
            f"3-hop: {_truncate_text(hard_row.question, max_chars=120)}"
        )
        combined_answer = f"1-hop: {easy_row.answer}; 3-hop: {hard_row.answer}"
        rows = sorted(
            [easy_row, hard_row],
            key=lambda row: DIFFICULTY_RANK.get(row.difficulty, 99),
        )
        selected.append(
            ExampleBlock(
                category="Difficulty Scaling",
                title="Reasoning-depth degradation from 1-hop to 3-hop",
                question_id=f"{easy_row.question_id} vs {hard_row.question_id}",
                question=combined_question,
                answer=combined_answer,
                headers=["Model", "Difficulty", "Viz Type", "Prediction", "Result"],
                rows=[
                    [
                        row.model,
                        row.difficulty,
                        row.viz_type,
                        _truncate_text(row.prediction),
                        _status(row),
                    ]
                    for row in rows
                ],
            )
        )
        used_keys.add(key)
        if len(selected) >= limit:
            break
    return selected


def select_examples(rows: list[EvalRow], model_order: list[str]) -> list[ExampleBlock]:
    """Select 6-8 deterministic, high-contrast qualitative examples.

    Args:
        rows: Full evaluation rows across all models.

    Returns:
        Ordered example blocks spanning sensitivity, disagreement, and difficulty.

    Raises:
        RuntimeError: If fewer than six contrasting examples are found.
    """
    by_question_model, by_question_viz, by_model_base_modality_viz = _build_indices(
        rows
    )

    sensitivity = _select_visualization_sensitivity(
        by_question_model=by_question_model, limit=3
    )
    disagreement = _select_cross_model_disagreement(
        by_question_viz=by_question_viz, limit=2, model_order=model_order
    )
    difficulty = _select_difficulty_scaling(
        by_model_base_modality_viz=by_model_base_modality_viz,
        limit=2,
    )

    selected = sensitivity + disagreement + difficulty
    seen: set[tuple[str, str, str]] = {
        (example.category, example.question_id, example.title) for example in selected
    }

    if len(selected) < 6:
        for example in _select_visualization_sensitivity(
            by_question_model=by_question_model,
            limit=8,
        ):
            key = (example.category, example.question_id, example.title)
            if key in seen:
                continue
            selected.append(example)
            seen.add(key)
            if len(selected) >= 6:
                break

    if len(selected) < 6:
        for example in _select_cross_model_disagreement(
            by_question_viz=by_question_viz,
            limit=8,
            model_order=model_order,
        ):
            key = (example.category, example.question_id, example.title)
            if key in seen:
                continue
            selected.append(example)
            seen.add(key)
            if len(selected) >= 6:
                break

    if len(selected) < 6:
        raise RuntimeError(
            "Could not select enough contrasting examples. "
            f"Found {len(selected)} examples; expected at least 6."
        )
    return selected[:8]


def build_latex(examples: list[ExampleBlock]) -> str:
    """Render selected examples into a LaTeX appendix subsection.

    Args:
        examples: Selected qualitative blocks.

    Returns:
        LaTeX content suitable for `\\input{}`.
    """
    lines: list[str] = [
        "% Auto-generated by scripts/extract_qualitative_examples.py",
        "\\subsection{Qualitative Examples}",
        "\\providecommand{\\cmark}{\\textcolor{green!60!black}{\\textbf{Correct}}}",
        "\\providecommand{\\xmark}{\\textcolor{red!70!black}{\\textbf{Incorrect}}}",
        "",
    ]

    linebreak = " \\\\"
    for idx, example in enumerate(examples, start=1):
        lines.append(f"\\paragraph{{Example {idx}: {_latex_escape(example.title)}.}}")
        lines.append(
            f"\\textbf{{Category:}} {_latex_escape(example.category)}{linebreak}"
        )
        lines.append(
            f"\\textbf{{Question ID:}} {_latex_escape(example.question_id)}{linebreak}"
        )
        lines.append(
            f"\\textbf{{Question:}} {_latex_escape(_truncate_text(example.question, 220))}{linebreak}"
        )
        lines.append(
            f"\\textbf{{Ground Truth:}} {_latex_escape(_truncate_text(example.answer, 220))}{linebreak}"
        )

        tabular_spec = "l" * len(example.headers)
        lines.append(f"\\begin{{tabular}}{{{tabular_spec}}}")
        lines.append("\\toprule")
        header_line = " & ".join(_latex_escape(header) for header in example.headers)
        lines.append(f"{header_line}{linebreak}")
        lines.append("\\midrule")
        for row in example.rows:
            escaped = [
                _latex_escape(cell) if cell not in {"\\cmark", "\\xmark"} else cell
                for cell in row
            ]
            lines.append(f"{' & '.join(escaped)}{linebreak}")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\vspace{0.5em}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    """Run extraction pipeline and write the appendix `.tex` file."""
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    results_dir = (
        args.results_dir
        if args.results_dir.is_absolute()
        else project_root / args.results_dir
    )
    output_path = (
        args.output if args.output.is_absolute() else project_root / args.output
    )

    rows, model_order = load_results(
        results_dir=results_dir,
        include_supplementary_local=bool(args.include_supplementary_local),
    )
    examples = select_examples(rows, model_order)
    latex = build_latex(examples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex, encoding="utf-8")
    print(f"Wrote {len(examples)} qualitative examples to: {output_path}")


if __name__ == "__main__":
    main()
