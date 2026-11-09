from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _model_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    models = results.get("models")
    if isinstance(models, list):
        return [row for row in models if isinstance(row, dict)]
    return []


def generate_latex_table(results: dict[str, Any], table_type: str = "main") -> str:
    """Generate LaTeX table for main, difficulty, or sensitivity views."""
    rows = _model_rows(results)
    if table_type == "difficulty":
        header = "Model & 1-hop & 2-hop & 3-hop & Counterfactual \\\\"
        body_lines: list[str] = []
        for row in rows:
            scores = row.get("difficulty", {})
            body_lines.append(
                "{} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\".format(
                    row.get("model", "unknown"),
                    float(scores.get("1-hop", 0.0)),
                    float(scores.get("2-hop", 0.0)),
                    float(scores.get("3-hop", 0.0)),
                    float(scores.get("counterfactual", 0.0)),
                )
            )
    elif table_type == "sensitivity":
        header = "Model & Avg. Sensitivity & Hallucination Rate \\\\"
        body_lines = []
        for row in rows:
            body_lines.append(
                "{} & {:.3f} & {:.3f} \\\\".format(
                    row.get("model", "unknown"),
                    float(row.get("avg_sensitivity", 0.0)),
                    float(row.get("avg_hallucination_rate", 0.0)),
                )
            )
    else:
        header = "Model & Overall & Tabular & Time Series & Graph & Mixed \\\\"
        body_lines = []
        for row in rows:
            modality_scores = row.get("modality", {})
            body_lines.append(
                "{} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\".format(
                    row.get("model", "unknown"),
                    float(row.get("overall_exact", 0.0)),
                    float(modality_scores.get("tabular", 0.0)),
                    float(modality_scores.get("timeseries", 0.0)),
                    float(modality_scores.get("graph", 0.0)),
                    float(modality_scores.get("mixed", 0.0)),
                )
            )

    lines = [
        "\\begin{tabular}{lrrrrr}"
        if table_type == "main"
        else "\\begin{tabular}{lrrr}",
        "\\toprule",
        header,
        "\\midrule",
        *body_lines,
        "\\bottomrule",
        "\\end{tabular}",
    ]
    return "\n".join(lines)


def plot_difficulty_breakdown(results: dict[str, Any], output_path: str | Path) -> None:
    """Plot grouped bars per model and difficulty bucket."""
    rows = _model_rows(results)
    if not rows:
        return

    difficulties = ["1-hop", "2-hop", "3-hop", "counterfactual"]
    x = np.arange(len(rows))
    width = 0.18

    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, difficulty in enumerate(difficulties):
        values = [float(row.get("difficulty", {}).get(difficulty, 0.0)) for row in rows]
        ax.bar(x + (idx - 1.5) * width, values, width=width, label=difficulty)

    ax.set_xticks(x)
    ax.set_xticklabels([str(row.get("model", "unknown")) for row in rows], rotation=20)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Difficulty Breakdown")
    ax.legend(loc="upper right")
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_viz_sensitivity(results: dict[str, Any], output_path: str | Path) -> None:
    """Plot heatmap of model performance by visualization type."""
    rows = _model_rows(results)
    if not rows:
        return

    all_viz = sorted(
        {
            str(viz)
            for row in rows
            for viz in row.get("viz_accuracy", {}).keys()
            if isinstance(row.get("viz_accuracy"), dict)
        }
    )
    if not all_viz:
        return

    matrix = np.asarray(
        [
            [float(row.get("viz_accuracy", {}).get(viz, 0.0)) for viz in all_viz]
            for row in rows
        ]
    )

    fig, ax = plt.subplots(figsize=(max(8, len(all_viz) * 0.7), 5.5))
    heatmap = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(all_viz)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels(all_viz, rotation=35, ha="right")
    ax.set_yticklabels([str(row.get("model", "unknown")) for row in rows])
    ax.set_title("Visualization Sensitivity")
    fig.colorbar(heatmap, ax=ax, fraction=0.03, pad=0.04, label="Accuracy")
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_modality_comparison(results: dict[str, Any], output_path: str | Path) -> None:
    """Plot radar chart comparing modality performance for each model."""
    rows = _model_rows(results)
    if not rows:
        return

    labels = ["tabular", "timeseries", "graph", "mixed"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    for row in rows:
        modality = row.get("modality", {})
        values = np.asarray([float(modality.get(label, 0.0)) for label in labels])
        values = np.concatenate([values, [values[0]]])
        ax.plot(angles, values, linewidth=2, label=str(row.get("model", "unknown")))
        ax.fill(angles, values, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Modality Comparison")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def compute_significance(
    results_a: dict[str, Any], results_b: dict[str, Any]
) -> dict[str, float]:
    """Run paired bootstrap significance test on per-item exact outcomes."""
    a_by_item: dict[str, float] = {}
    b_by_item: dict[str, float] = {}
    for row in results_a.get("predictions", []):
        if isinstance(row, dict):
            a_by_item[str(row.get("question_id", ""))] = float(row.get("exact", 0.0))
    for row in results_b.get("predictions", []):
        if isinstance(row, dict):
            b_by_item[str(row.get("question_id", ""))] = float(row.get("exact", 0.0))

    shared_ids = sorted(set(a_by_item) & set(b_by_item))
    if not shared_ids:
        return {
            "delta": 0.0,
            "p_value": 1.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "n": 0,
        }

    diffs = np.asarray(
        [a_by_item[item_id] - b_by_item[item_id] for item_id in shared_ids]
    )
    observed = float(np.mean(diffs))

    rng = np.random.default_rng(42)
    boot = []
    n = len(diffs)
    for _ in range(5000):
        sample = diffs[rng.integers(0, n, n)]
        boot.append(float(np.mean(sample)))
    boot_arr = np.asarray(boot)

    ci_lower = float(np.percentile(boot_arr, 2.5))
    ci_upper = float(np.percentile(boot_arr, 97.5))
    p_value = float(2.0 * min(np.mean(boot_arr <= 0.0), np.mean(boot_arr >= 0.0)))
    return {
        "delta": observed,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n": float(n),
    }
