from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownParameterType=false, reportUnknownArgumentType=false
# pyright: reportUnusedCallResult=false

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROMPT_MODELS = ["gpt4o", "gemini", "claude", "qwen"]
MIXED_MODELS = ["gpt4o", "gemini", "claude", "qwen", "internvl", "llava"]
PROMPT_VARIANTS = ["concise", "detailed", "cot", "minimal"]
PROMPT_MODALITIES = ["tabular", "timeseries", "graph"]
MIXED_COMBINATIONS = ["mixed_tab_ts", "mixed_tab_graph", "mixed_ts_graph"]
DIFFICULTY_ORDER = ["1-hop", "2-hop", "3-hop", "counterfactual"]


@dataclass(slots=True)
class MetricBundle:
    """Container for three core evaluation metrics."""

    em: float
    f1: float
    numeric_accuracy: float


def log(msg: str) -> None:
    """Print a timestamped log line."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for extension analysis workflows."""
    parser = argparse.ArgumentParser(
        description="Analyze prompt and mixed-type extension experiment outputs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prompt_parser = subparsers.add_parser(
        "prompt",
        help="Analyze prompt-sensitivity ablation results.",
    )
    prompt_parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Base results directory containing ablation outputs.",
    )
    prompt_parser.add_argument(
        "--models",
        nargs="+",
        default=PROMPT_MODELS,
        help="Model keys to analyze.",
    )
    prompt_parser.add_argument(
        "--variants",
        nargs="+",
        default=PROMPT_VARIANTS,
        help="Prompt variant keys to analyze.",
    )

    mixed_parser = subparsers.add_parser(
        "mixed",
        help="Analyze mixed-type evaluation results.",
    )
    mixed_parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Base results directory containing mixed/full outputs.",
    )
    mixed_parser.add_argument(
        "--models",
        nargs="+",
        default=MIXED_MODELS,
        help="Model keys to analyze.",
    )

    return parser.parse_args()


def resolve_path(project_root: Path, maybe_relative: Path) -> Path:
    """Resolve a potentially relative path against project root."""
    if maybe_relative.is_absolute():
        return maybe_relative
    return project_root / maybe_relative


def warn_missing(path: Path) -> None:
    """Emit a standard warning for a missing result file."""
    log(f"WARNING: missing file, skipping: {path}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records from disk.

    Args:
        path: Path to a JSONL file.

    Returns:
        Parsed rows. Returns an empty list when file is missing.
    """
    if not path.exists():
        warn_missing(path)
        return []

    rows: list[dict[str, Any]] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        payload = line.strip()
        if not payload:
            continue
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            log(f"WARNING: invalid JSON at {path}:{idx}, skipping line")
            continue
        if not isinstance(parsed, dict):
            log(f"WARNING: non-object JSON at {path}:{idx}, skipping line")
            continue
        rows.append(parsed)

    if not rows:
        log(f"WARNING: no valid rows found in {path}")
    return rows


def as_float(value: Any) -> float:
    """Cast a value to float with safe fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def summarize_rows(rows: list[dict[str, Any]]) -> MetricBundle:
    """Compute mean EM/F1/numeric over a row list."""
    if not rows:
        return MetricBundle(em=0.0, f1=0.0, numeric_accuracy=0.0)

    em_total = 0.0
    f1_total = 0.0
    numeric_total = 0.0
    for row in rows:
        em_total += as_float(row.get("exact_match", 0.0))
        f1_total += as_float(row.get("f1", 0.0))
        numeric_total += as_float(row.get("numeric_accuracy", 0.0))

    n_rows = float(len(rows))
    return MetricBundle(
        em=em_total / n_rows,
        f1=f1_total / n_rows,
        numeric_accuracy=numeric_total / n_rows,
    )


def latex_escape(value: str) -> str:
    """Escape text for LaTeX table cells."""
    return value.replace("_", "\\_")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Write dictionaries to CSV with stable column ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def format_pct(value: float) -> str:
    """Format ratio [0,1] as a two-decimal percent string."""
    return f"{value * 100.0:.2f}"


def analyze_prompt(results_dir: Path, models: list[str], variants: list[str]) -> None:
    """Analyze prompt-sensitivity ablation results and produce artifacts.

    Args:
        results_dir: Base results directory.
        models: Model keys to include.
        variants: Prompt variants to include.
    """
    ablation_dir = results_dir / "ablation"
    summary_rows: list[dict[str, Any]] = []
    metric_index: dict[tuple[str, str], dict[str, MetricBundle]] = {}

    for model in models:
        for variant in variants:
            path = ablation_dir / f"prompt_{model}_{variant}.jsonl"
            rows = load_jsonl(path)
            if not rows:
                continue

            overall = summarize_rows(rows)
            by_modality: dict[str, MetricBundle] = {}
            for modality in PROMPT_MODALITIES:
                subset = [
                    row for row in rows if str(row.get("modality", "")) == modality
                ]
                by_modality[modality] = summarize_rows(subset)

            metric_index[(model, variant)] = {
                "overall": overall,
                "tabular": by_modality["tabular"],
                "timeseries": by_modality["timeseries"],
                "graph": by_modality["graph"],
            }

            summary_rows.append(
                {
                    "model": model,
                    "variant": variant,
                    "overall_em": overall.em,
                    "tabular_em": by_modality["tabular"].em,
                    "timeseries_em": by_modality["timeseries"].em,
                    "graph_em": by_modality["graph"].em,
                }
            )

    summary_rows.sort(key=lambda row: (str(row["model"]), str(row["variant"])))

    csv_path = ablation_dir / "prompt_sensitivity_summary.csv"
    write_csv(
        csv_path,
        summary_rows,
        [
            "model",
            "variant",
            "overall_em",
            "tabular_em",
            "timeseries_em",
            "graph_em",
        ],
    )
    log(f"Wrote prompt summary CSV: {csv_path}")

    tex_lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Model & Variant & Overall EM (\\%) & Tabular EM (\\%) & Time Series EM (\\%) & Graph EM (\\%) \\\\",
        "\\midrule",
    ]
    for row in summary_rows:
        tex_lines.append(
            "{} & {} & {} & {} & {} & {} \\\\".format(
                latex_escape(str(row["model"])),
                latex_escape(str(row["variant"])),
                format_pct(as_float(row["overall_em"])),
                format_pct(as_float(row["tabular_em"])),
                format_pct(as_float(row["timeseries_em"])),
                format_pct(as_float(row["graph_em"])),
            )
        )
    tex_lines.extend(["\\bottomrule", "\\end{tabular}"])

    tex_path = ablation_dir / "prompt_sensitivity_table.tex"
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    log(f"Wrote prompt LaTeX table: {tex_path}")

    figure_path = ablation_dir / "prompt_sensitivity_comparison.png"
    make_prompt_figure(summary_rows, figure_path, models=models, variants=variants)

    print_prompt_summary(summary_rows, metric_index, models=models, variants=variants)


def make_prompt_figure(
    summary_rows: list[dict[str, Any]],
    figure_path: Path,
    models: list[str],
    variants: list[str],
) -> None:
    """Create grouped bar chart for prompt variant overall EM."""
    if not summary_rows:
        log("WARNING: no prompt rows available, skipping figure generation")
        return

    row_map: dict[tuple[str, str], float] = {}
    for row in summary_rows:
        row_map[(str(row["model"]), str(row["variant"]))] = as_float(row["overall_em"])

    x_vals = list(range(len(models)))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10.2, 4.8))

    for idx, variant in enumerate(variants):
        offsets = [x + (idx - (len(variants) - 1) / 2.0) * width for x in x_vals]
        y_vals = [100.0 * row_map.get((model, variant), 0.0) for model in models]
        ax.bar(offsets, y_vals, width=width, label=variant)

    ax.set_xticks(x_vals)
    ax.set_xticklabels(models)
    ax.set_ylabel("Exact Match (%)")
    ax.set_title("Prompt-Sensitivity Ablation (Overall EM)")
    ax.legend(frameon=False, ncol=min(len(variants), 4))

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    log(f"Wrote prompt comparison figure: {figure_path}")


def print_prompt_summary(
    summary_rows: list[dict[str, Any]],
    metric_index: dict[tuple[str, str], dict[str, MetricBundle]],
    models: list[str],
    variants: list[str],
) -> None:
    """Print formatted prompt analysis summary to stdout."""
    print("\n=== Prompt Sensitivity Summary ===")
    if not summary_rows:
        print("No prompt-sensitivity result files were found.")
        return

    for model in models:
        model_seen = False
        for variant in variants:
            key = (model, variant)
            if key not in metric_index:
                continue
            model_seen = True
            metrics = metric_index[key]
            overall = metrics["overall"]
            tabular = metrics["tabular"]
            timeseries = metrics["timeseries"]
            graph = metrics["graph"]
            print(
                " - {} / {} | overall EM/F1/Num: {}/{}/{} | tabular: {}/{}/{} | timeseries: {}/{}/{} | graph: {}/{}/{}".format(
                    model,
                    variant,
                    format_pct(overall.em),
                    format_pct(overall.f1),
                    format_pct(overall.numeric_accuracy),
                    format_pct(tabular.em),
                    format_pct(tabular.f1),
                    format_pct(tabular.numeric_accuracy),
                    format_pct(timeseries.em),
                    format_pct(timeseries.f1),
                    format_pct(timeseries.numeric_accuracy),
                    format_pct(graph.em),
                    format_pct(graph.f1),
                    format_pct(graph.numeric_accuracy),
                )
            )
        if not model_seen:
            print(f" - {model}: no available prompt variant files")


def sanitize_difficulty(difficulty: str) -> str:
    """Convert difficulty labels into CSV-friendly suffixes."""
    return difficulty.replace("-", "_").replace(" ", "_")


def mean_em_for(rows: list[dict[str, Any]]) -> float:
    """Compute mean exact match for a set of rows."""
    return summarize_rows(rows).em


def analyze_mixed(results_dir: Path, models: list[str]) -> None:
    """Analyze mixed-type results and compare against single-type baselines.

    Args:
        results_dir: Base results directory.
        models: Model keys to include.
    """
    summary_rows: list[dict[str, Any]] = []

    for model in models:
        mixed_path = results_dir / f"mixed_{model}.jsonl"
        mixed_rows = load_jsonl(mixed_path)
        if not mixed_rows:
            continue

        baseline_path = results_dir / f"full_{model}.jsonl"
        baseline_rows = load_jsonl(baseline_path)
        baseline_em = mean_em_for(baseline_rows) if baseline_rows else 0.0

        mixed_overall_em = mean_em_for(mixed_rows)
        row_payload: dict[str, Any] = {
            "model": model,
            "n_rows": len(mixed_rows),
            "mixed_overall_em": mixed_overall_em,
            "single_baseline_em": baseline_em,
            "cross_modal_delta_em": mixed_overall_em - baseline_em,
        }

        for combo in MIXED_COMBINATIONS:
            subset = [
                row for row in mixed_rows if str(row.get("modality", "")) == combo
            ]
            row_payload[f"{combo}_em"] = mean_em_for(subset)

        seen_difficulties = sorted(
            {
                str(row.get("difficulty", ""))
                for row in mixed_rows
                if str(row.get("difficulty", ""))
            },
            key=lambda value: (
                DIFFICULTY_ORDER.index(value) if value in DIFFICULTY_ORDER else 999,
                value,
            ),
        )
        for difficulty in seen_difficulties:
            subset = [
                row
                for row in mixed_rows
                if str(row.get("difficulty", "")) == difficulty
            ]
            row_payload[f"difficulty_{sanitize_difficulty(difficulty)}_em"] = (
                mean_em_for(subset)
            )

        summary_rows.append(row_payload)

    summary_rows.sort(key=lambda row: str(row["model"]))

    difficulty_columns = collect_difficulty_columns(summary_rows)
    csv_columns = [
        "model",
        "n_rows",
        "mixed_overall_em",
        "single_baseline_em",
        "cross_modal_delta_em",
        "mixed_tab_ts_em",
        "mixed_tab_graph_em",
        "mixed_ts_graph_em",
        *difficulty_columns,
    ]

    csv_path = results_dir / "mixed_analysis_summary.csv"
    write_csv(csv_path, summary_rows, csv_columns)
    log(f"Wrote mixed summary CSV: {csv_path}")

    tex_path = results_dir / "mixed_analysis_table.tex"
    tex_path.write_text(
        build_mixed_latex(summary_rows, difficulty_columns), encoding="utf-8"
    )
    log(f"Wrote mixed LaTeX table: {tex_path}")

    figure_path = results_dir / "mixed_analysis_comparison.png"
    make_mixed_figure(summary_rows, figure_path)

    print_mixed_summary(summary_rows, difficulty_columns)


def collect_difficulty_columns(summary_rows: list[dict[str, Any]]) -> list[str]:
    """Collect and order difficulty columns from mixed summary payloads."""
    cols = {
        key
        for row in summary_rows
        for key in row.keys()
        if key.startswith("difficulty_") and key.endswith("_em")
    }

    ordered: list[str] = []
    for difficulty in DIFFICULTY_ORDER:
        key = f"difficulty_{sanitize_difficulty(difficulty)}_em"
        if key in cols:
            ordered.append(key)
            cols.remove(key)

    ordered.extend(sorted(cols))
    return ordered


def build_mixed_latex(
    summary_rows: list[dict[str, Any]], difficulty_cols: list[str]
) -> str:
    """Build LaTeX table snippet for mixed-type analysis."""
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Model & Mixed EM (\\%) & Single EM (\\%) & Delta (pp) & Tab+TS (\\%) & Tab+Graph (\\%) & TS+Graph (\\%) \\\\",
        "\\midrule",
    ]
    for row in summary_rows:
        lines.append(
            "{} & {} & {} & {:+.2f} & {} & {} & {} \\\\".format(
                latex_escape(str(row.get("model", ""))),
                format_pct(as_float(row.get("mixed_overall_em", 0.0))),
                format_pct(as_float(row.get("single_baseline_em", 0.0))),
                as_float(row.get("cross_modal_delta_em", 0.0)) * 100.0,
                format_pct(as_float(row.get("mixed_tab_ts_em", 0.0))),
                format_pct(as_float(row.get("mixed_tab_graph_em", 0.0))),
                format_pct(as_float(row.get("mixed_ts_graph_em", 0.0))),
            )
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])

    if difficulty_cols:
        col_spec = "l" + "r" * len(difficulty_cols)
        headers = [
            col.removeprefix("difficulty_").removesuffix("_em").replace("_", "-")
            for col in difficulty_cols
        ]
        lines.extend(
            [
                f"\\begin{{tabular}}{{{col_spec}}}",
                "\\toprule",
                "Model & "
                + " & ".join(f"{latex_escape(h)} (\\%)" for h in headers)
                + " \\\\",
                "\\midrule",
            ]
        )
        for row in summary_rows:
            vals = [format_pct(as_float(row.get(col, 0.0))) for col in difficulty_cols]
            lines.append(
                " & ".join([latex_escape(str(row.get("model", ""))), *vals]) + " \\\\"
            )
        lines.extend(["\\bottomrule", "\\end{tabular}", ""])

    return "\n".join(lines)


def make_mixed_figure(summary_rows: list[dict[str, Any]], figure_path: Path) -> None:
    """Create baseline-vs-mixed comparison figure."""
    if not summary_rows:
        log("WARNING: no mixed rows available, skipping figure generation")
        return

    models = [str(row.get("model", "")) for row in summary_rows]
    x_vals = list(range(len(models)))
    width = 0.36
    baseline_vals = [
        100.0 * as_float(row.get("single_baseline_em", 0.0)) for row in summary_rows
    ]
    mixed_vals = [
        100.0 * as_float(row.get("mixed_overall_em", 0.0)) for row in summary_rows
    ]

    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    ax.bar(
        [x - width / 2.0 for x in x_vals], baseline_vals, width=width, label="single"
    )
    ax.bar([x + width / 2.0 for x in x_vals], mixed_vals, width=width, label="mixed")

    ax.set_xticks(x_vals)
    ax.set_xticklabels(models)
    ax.set_ylabel("Exact Match (%)")
    ax.set_title("Single-Type vs Mixed-Type Overall EM")
    ax.legend(frameon=False, ncol=2)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    log(f"Wrote mixed comparison figure: {figure_path}")


def print_mixed_summary(
    summary_rows: list[dict[str, Any]], difficulty_cols: list[str]
) -> None:
    """Print formatted mixed analysis summary to stdout."""
    print("\n=== Mixed-Type Summary ===")
    if not summary_rows:
        print("No mixed evaluation result files were found.")
        return

    for row in summary_rows:
        difficulty_blob = ", ".join(
            f"{col.removeprefix('difficulty_').removesuffix('_em')}: {format_pct(as_float(row.get(col, 0.0)))}"
            for col in difficulty_cols
        )
        print(
            " - {} | mixed={} | single={} | delta={:+.2f}pp | tab+ts={} | tab+graph={} | ts+graph={}{}".format(
                str(row.get("model", "")),
                format_pct(as_float(row.get("mixed_overall_em", 0.0))),
                format_pct(as_float(row.get("single_baseline_em", 0.0))),
                as_float(row.get("cross_modal_delta_em", 0.0)) * 100.0,
                format_pct(as_float(row.get("mixed_tab_ts_em", 0.0))),
                format_pct(as_float(row.get("mixed_tab_graph_em", 0.0))),
                format_pct(as_float(row.get("mixed_ts_graph_em", 0.0))),
                f" | difficulty -> {difficulty_blob}" if difficulty_blob else "",
            )
        )


def main() -> None:
    """Dispatch prompt or mixed analysis subcommand."""
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    results_dir_arg = Path(args.results_dir)
    results_dir = resolve_path(project_root, results_dir_arg)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "prompt":
        analyze_prompt(
            results_dir=results_dir,
            models=list(args.models),
            variants=list(args.variants),
        )
        return

    if args.command == "mixed":
        analyze_mixed(results_dir=results_dir, models=list(args.models))
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
