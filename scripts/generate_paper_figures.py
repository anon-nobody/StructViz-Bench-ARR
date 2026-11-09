from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnusedCallResult=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportPrivateImportUsage=false

import argparse
import json
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PILOT_RESULT_FILES = {
    "GPT-4o": "pilot_gpt4o.jsonl",
    "Gemini Flash": "pilot_gemini.jsonl",
    "Qwen2.5-VL-7B": "pilot_qwen.jsonl",
    "Claude Sonnet": "pilot_claude.jsonl",
}

FULL_RESULT_FILES = {
    "GPT-4o": "full_gpt4o.jsonl",
    "Gemini Flash": "full_gemini.jsonl",
    "Qwen2.5-VL-7B": "full_qwen.jsonl",
    "Claude Sonnet": "full_claude.jsonl",
}

SUPPLEMENTARY_RESULT_FILES = {
    "InternVL2.5-8B": "full_internvl.jsonl",
}

MODEL_ORDER = ["GPT-4o", "Gemini Flash", "Qwen2.5-VL-7B", "Claude Sonnet"]
SUPPLEMENTARY_MODEL_ORDER = MODEL_ORDER + ["InternVL2.5-8B"]
MODALITY_ORDER = ["tabular", "timeseries", "graph"]
DIFFICULTY_ORDER = ["1-hop", "2-hop", "3-hop", "counterfactual"]

MODALITY_LABELS = {
    "tabular": "Tabular",
    "timeseries": "Time Series",
    "graph": "Graph",
}

DIFFICULTY_LABELS = {
    "1-hop": "1-hop",
    "2-hop": "2-hop",
    "3-hop": "3-hop",
    "counterfactual": "Counterfactual",
}


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "--",
            "axes.titleweight": "semibold",
            "axes.labelweight": "medium",
            "figure.dpi": 200,
            "savefig.dpi": 300,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables and figures from StructViz result JSONL files.",
    )
    parser.add_argument(
        "--split",
        choices=["pilot", "full"],
        default="pilot",
        help="Which result files to use (pilot_* or full_*).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing result JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/figures"),
        help="Directory where PDF figures are written.",
    )
    parser.add_argument(
        "--include-supplementary-local",
        action="store_true",
        default=False,
        help="Include supplementary local model results (InternVL2.5-8B).",
    )
    return parser.parse_args()


def load_results(
    results_dir: Path,
    split: str = "pilot",
    include_supplementary: bool = False,
) -> pd.DataFrame:
    result_files = dict(PILOT_RESULT_FILES if split == "pilot" else FULL_RESULT_FILES)
    if include_supplementary and split == "full":
        result_files.update(SUPPLEMENTARY_RESULT_FILES)
    frames: list[pd.DataFrame] = []
    for model, file_name in result_files.items():
        file_path = results_dir / file_name
        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing result file for split='{split}': {file_path}",
            )
        rows = [
            json.loads(line)
            for line in file_path.read_text(encoding="utf-8").splitlines()
        ]
        frame = pd.DataFrame(rows)
        frame["model"] = model
        frame["source_group"] = np.where(
            frame["source"] == "synthetic", "synthetic", "real_world"
        )
        frames.append(frame)

    all_data = pd.concat(frames, ignore_index=True)
    all_data["modality"] = pd.Categorical(
        all_data["modality"], categories=MODALITY_ORDER, ordered=True
    )
    all_data["difficulty"] = pd.Categorical(
        all_data["difficulty"], categories=DIFFICULTY_ORDER, ordered=True
    )
    order = SUPPLEMENTARY_MODEL_ORDER if include_supplementary else MODEL_ORDER
    all_data["model"] = pd.Categorical(
        all_data["model"], categories=order, ordered=True
    )
    return all_data


def _pct(value: float) -> float:
    return 100.0 * float(value)


def make_table_overall(data: pd.DataFrame) -> str:
    grouped = data.groupby("model", observed=True).agg(
        em=("exact_match", "mean"),
        f1=("f1", "mean"),
        numeric_accuracy=("numeric_accuracy", "mean"),
    )

    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & EM (\%) & F1 (\%) & Numeric Accuracy (\%) \\",
        r"\midrule",
    ]
    for model in MODEL_ORDER:
        row = grouped.loc[model]
        lines.append(
            f"{model} & {_pct(row['em']):.1f} & {_pct(row['f1']):.1f} & {_pct(row['numeric_accuracy']):.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def make_table_modality(data: pd.DataFrame) -> str:
    grouped = (
        data.groupby(["model", "modality"], observed=True)["exact_match"]
        .mean()
        .reset_index()
        .pivot(index="model", columns="modality", values="exact_match")
    )

    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & Tabular (\%) & Time Series (\%) & Graph (\%) \\",
        r"\midrule",
    ]
    for model in MODEL_ORDER:
        row = grouped.loc[model]
        lines.append(
            f"{model} & {_pct(row['tabular']):.1f} & {_pct(row['timeseries']):.1f} & {_pct(row['graph']):.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def make_table_viz_per_modality(data: pd.DataFrame) -> str:
    blocks: list[str] = []
    for modality in MODALITY_ORDER:
        subset = data[data["modality"] == modality]
        pivot = (
            subset.groupby(["model", "viz_type"], observed=True)["exact_match"]
            .mean()
            .reset_index()
            .pivot(index="model", columns="viz_type", values="exact_match")
        )
        viz_cols = sorted(pivot.columns.tolist())
        col_spec = "l" + "c" * len(viz_cols)
        header = " & ".join(["Model", *viz_cols]) + " \\\\"

        lines = [
            f"% {MODALITY_LABELS[modality]}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            header,
            r"\midrule",
        ]
        for model in MODEL_ORDER:
            row_vals = [f"{_pct(pivot.loc[model, c]):.1f}" for c in viz_cols]
            lines.append(" & ".join([model, *row_vals]) + " \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def make_table_sensitivity_gap(data: pd.DataFrame) -> str:
    rows: list[tuple[str, str, str, float, str, float, float]] = []
    for model in MODEL_ORDER:
        for modality in MODALITY_ORDER:
            subset = data[(data["model"] == model) & (data["modality"] == modality)]
            scores = (
                subset.groupby("viz_type", observed=True)["exact_match"]
                .mean()
                .sort_values()
            )
            worst_name = scores.index[0]
            worst_score = float(scores.iloc[0])
            best_name = scores.index[-1]
            best_score = float(scores.iloc[-1])
            rows.append(
                (
                    model,
                    MODALITY_LABELS[modality],
                    best_name,
                    best_score,
                    worst_name,
                    worst_score,
                    best_score - worst_score,
                )
            )

    lines = [
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Model & Modality & Best Viz & Best EM (\%) & Worst Viz & Worst EM (\%) & Gap (pp) \\",
        r"\midrule",
    ]
    for model, modality, best_name, best_score, worst_name, worst_score, gap in rows:
        lines.append(
            f"{model} & {modality} & {best_name} & {_pct(best_score):.1f} & {worst_name} & {_pct(worst_score):.1f} & {_pct(gap):.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def make_table_difficulty(data: pd.DataFrame) -> str:
    pivot = (
        data.groupby(["model", "difficulty"], observed=True)["exact_match"]
        .mean()
        .reset_index()
        .pivot(index="model", columns="difficulty", values="exact_match")
    )
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & 1-hop (\%) & 2-hop (\%) & 3-hop (\%) & Counterfactual (\%) \\",
        r"\midrule",
    ]
    for model in MODEL_ORDER:
        row = pivot.loc[model]
        lines.append(
            f"{model} & {_pct(row['1-hop']):.1f} & {_pct(row['2-hop']):.1f} & {_pct(row['3-hop']):.1f} & {_pct(row['counterfactual']):.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def make_table_source(data: pd.DataFrame) -> str:
    pivot = (
        data.groupby(["model", "source_group"], observed=True)["exact_match"]
        .mean()
        .reset_index()
        .pivot(index="model", columns="source_group", values="exact_match")
    )
    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Model & Synthetic (\%) & Real-world (\%) \\",
        r"\midrule",
    ]
    for model in MODEL_ORDER:
        row = pivot.loc[model]
        lines.append(
            f"{model} & {_pct(row['synthetic']):.1f} & {_pct(row['real_world']):.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def _save(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_overview(output_dir: Path) -> None:
    # Compact figure: less vertical whitespace above/below the boxes;
    # boxes occupy most of the canvas so text can be larger without clipping.
    fig, ax = plt.subplots(figsize=(11.5, 2.4))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Boxes fill ~65% of vertical space (y in [0.18, 0.83]); horizontal layout
    # uses 4 boxes of width 0.22 with arrow gaps of width 0.03 between them.
    box_w = 0.22
    box_h = 0.65
    y0 = 0.18
    starts = [0.02, 0.27, 0.52, 0.77]
    boxes = [
        (starts[0], y0, box_w, box_h,
         "Structured Data\n(Tabular,\nTime Series, Graph)", "#F4E6C8"),
        (starts[1], y0, box_w, box_h,
         "Multi-Format Rendering\n(14 Visualization Types)", "#D9EAD3"),
        (starts[2], y0, box_w, box_h,
         "MLLM Evaluation\n(4 core models + 32B)", "#D0E0F0"),
        (starts[3], y0, box_w, box_h,
         "Sensitivity Metrics\n(EM Gap, Flip Rate)", "#F9D5D3"),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=color, ec="#3A3A3A", lw=1.5))
        ax.text(
            x + w / 2, y + h / 2, label,
            ha="center", va="center", fontsize=11,
        )

    # Arrows between consecutive boxes.
    for i in range(len(starts) - 1):
        start_x = starts[i] + box_w
        end_x = starts[i + 1]
        ax.annotate(
            "",
            xy=(end_x, y0 + box_h / 2),
            xytext=(start_x, y0 + box_h / 2),
            arrowprops=dict(arrowstyle="->", lw=1.8, color="#444444"),
        )

    _save(fig, output_dir / "overview.pdf")


def plot_main_results(data: pd.DataFrame, output_dir: Path) -> None:
    pivot = (
        data.groupby(["model", "modality"], observed=True)["exact_match"]
        .mean()
        .reset_index()
        .pivot(index="model", columns="modality", values="exact_match")
        .loc[MODEL_ORDER, MODALITY_ORDER]
    )
    x = np.arange(len(MODEL_ORDER))
    width = 0.23
    colors = ["#355C7D", "#6C5B7B", "#C06C84"]

    fig, ax = plt.subplots(figsize=(8.8, 3.5))
    for idx, modality in enumerate(MODALITY_ORDER):
        vals = 100.0 * pivot[modality].to_numpy()
        ax.bar(
            x + (idx - 1) * width,
            vals,
            width=width,
            label=MODALITY_LABELS[modality],
            color=colors[idx],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, rotation=12, ha="right")
    ax.set_ylabel("Exact Match (%, higher is better)")
    ax.set_title("Main Results by Modality")
    ax.legend(
        frameon=False,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    _save(fig, output_dir / "main_results.pdf")


def plot_viz_sensitivity_heatmap(data: pd.DataFrame, output_dir: Path) -> None:
    pivot = (
        data.groupby(["model", "viz_type"], observed=True)["exact_match"]
        .mean()
        .reset_index()
        .pivot(index="model", columns="viz_type", values="exact_match")
        .loc[MODEL_ORDER]
    )
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    im = ax.imshow(
        100.0 * pivot.to_numpy(), cmap="YlGnBu", aspect="auto", vmin=0, vmax=70
    )
    ax.set_yticks(np.arange(len(MODEL_ORDER)))
    ax.set_yticklabels(MODEL_ORDER)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_title("Visualization-Type Sensitivity (EM %, aggregated)")

    for i in range(len(MODEL_ORDER)):
        for j in range(len(pivot.columns)):
            value = 100.0 * pivot.iloc[i, j]
            ax.text(
                j,
                i,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=7,
                color="#1A1A1A",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Exact Match (%)")
    _save(fig, output_dir / "viz_sensitivity_heatmap.pdf")


def plot_sensitivity_gap(data: pd.DataFrame, output_dir: Path) -> None:
    records: list[dict[str, float | str]] = []
    for model in MODEL_ORDER:
        for modality in MODALITY_ORDER:
            subset = data[(data["model"] == model) & (data["modality"] == modality)]
            scores = subset.groupby("viz_type", observed=True)["exact_match"].mean()
            gap = 100.0 * float(scores.max() - scores.min())
            records.append({"model": model, "modality": modality, "gap": gap})

    gap_df = pd.DataFrame(records)
    x = np.arange(len(MODEL_ORDER))
    width = 0.22
    colors = ["#E07A5F", "#3D405B", "#81B29A"]

    fig, ax = plt.subplots(figsize=(8.8, 3.5))
    for idx, modality in enumerate(MODALITY_ORDER):
        vals = (
            gap_df[gap_df["modality"] == modality]
            .set_index("model")
            .loc[MODEL_ORDER, "gap"]
            .to_numpy()
        )
        ax.bar(
            x + (idx - 1) * width,
            vals,
            width=width,
            label=MODALITY_LABELS[modality],
            color=colors[idx],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, rotation=12, ha="right")
    ax.set_ylabel("Best-Worst EM Gap (percentage points)")
    ax.set_title("Visualization Sensitivity Gap by Modality")
    ax.legend(
        frameon=False,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    _save(fig, output_dir / "sensitivity_gap.pdf")


def plot_difficulty_breakdown(data: pd.DataFrame, output_dir: Path) -> None:
    pivot = (
        data.groupby(["model", "difficulty"], observed=True)["exact_match"]
        .mean()
        .reset_index()
        .pivot(index="model", columns="difficulty", values="exact_match")
        .loc[MODEL_ORDER, DIFFICULTY_ORDER]
    )

    x = np.arange(len(MODEL_ORDER))
    width = 0.18
    colors = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261"]

    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    for idx, difficulty in enumerate(DIFFICULTY_ORDER):
        vals = 100.0 * pivot[difficulty].to_numpy()
        ax.bar(
            x + (idx - 1.5) * width,
            vals,
            width=width,
            label=DIFFICULTY_LABELS[difficulty],
            color=colors[idx],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, rotation=12, ha="right")
    ax.set_ylabel("Exact Match (%)")
    ax.set_title("Difficulty-Level Breakdown")
    ax.legend(
        frameon=False,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    _save(fig, output_dir / "difficulty_breakdown.pdf")


def plot_modality_radar(data: pd.DataFrame, output_dir: Path) -> None:
    pivot = (
        data.groupby(["model", "modality"], observed=True)["exact_match"]
        .mean()
        .reset_index()
        .pivot(index="model", columns="modality", values="exact_match")
        .loc[MODEL_ORDER, MODALITY_ORDER]
    )
    labels = [MODALITY_LABELS[m] for m in MODALITY_ORDER]
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(6.6, 5.4))
    ax = plt.subplot(111, polar=True)
    palette = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]

    for i, model in enumerate(MODEL_ORDER):
        values = (100.0 * pivot.loc[model].to_numpy()).tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=model, color=palette[i])
        ax.fill(angles, values, alpha=0.08, color=palette[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 50)
    ax.set_yticks([10, 20, 30, 40, 50])
    ax.set_yticklabels(["10", "20", "30", "40", "50"])
    ax.set_title("Per-Model Modality Profile (EM %)", y=1.08)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    _save(fig, output_dir / "modality_radar.pdf")


def print_tables(data: pd.DataFrame) -> None:
    table_builders = [
        ("Table 1: Overall performance", make_table_overall),
        ("Table 2: EM by modality", make_table_modality),
        ("Table 3: EM by viz type per modality", make_table_viz_per_modality),
        ("Table 4: Visualization sensitivity gaps", make_table_sensitivity_gap),
        ("Table 5: EM by difficulty", make_table_difficulty),
        ("Table 6: EM by data source", make_table_source),
    ]
    for title, builder in table_builders:
        print("=" * 80)
        print(title)
        print("=" * 80)
        print(builder(data))
        print()


def generate_figures(data: pd.DataFrame, output_dir: Path) -> None:
    _set_plot_style()
    plot_overview(output_dir)
    plot_main_results(data, output_dir)
    plot_viz_sensitivity_heatmap(data, output_dir)
    plot_sensitivity_gap(data, output_dir)
    plot_difficulty_breakdown(data, output_dir)
    plot_modality_radar(data, output_dir)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    split = cast(str, args.split)
    results_dir_arg = cast(Path, args.results_dir)
    output_dir_arg = cast(Path, args.output_dir)

    results_dir = (
        results_dir_arg
        if results_dir_arg.is_absolute()
        else project_root / results_dir_arg
    )
    output_dir = (
        output_dir_arg
        if output_dir_arg.is_absolute()
        else project_root / output_dir_arg
    )

    include_supp = bool(getattr(args, "include_supplementary_local", False))
    data = load_results(
        results_dir=results_dir,
        split=split,
        include_supplementary=include_supp,
    )
    print_tables(data)
    generate_figures(data, output_dir)
    print(f"Saved figures to: {output_dir} (split={split})")


if __name__ == "__main__":
    main()
