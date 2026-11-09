from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np


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
MODALITY_ORDER = ["tabular", "timeseries", "graph"]
DIFFICULTY_ORDER = ["1-hop", "2-hop", "3-hop", "counterfactual"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute bootstrap 95% confidence intervals from full-scale JSONL results.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing full_*.jsonl model result files.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10_000,
        help="Number of bootstrap resamples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible bootstrap resampling.",
    )
    parser.add_argument(
        "--include-supplementary-local",
        action="store_true",
        help="Include caveated supplementary local baseline results (InternVL2.5-8B only).",
    )
    return parser.parse_args()


def _resolve_results_dir(results_dir_arg: Path) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    if results_dir_arg.is_absolute():
        return results_dir_arg
    return project_root / results_dir_arg


def _load_jsonl_rows(file_path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(cast(dict[str, float | str], json.loads(line)))
    return rows


def _bootstrap_ci(samples: np.ndarray) -> tuple[float, float]:
    lower = float(np.percentile(samples, 2.5))
    upper = float(np.percentile(samples, 97.5))
    return lower, upper


def compute_cluster_bootstrap_ci(
    rows: list[dict[str, float | str]],
    metric_key: str = "exact_match",
    cluster_key: str = "question_id",
    n_bootstrap: int = 10000,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Cluster-aware bootstrap CI that resamples by base question.

    This accounts for the repeated-measures structure where each base
    question is evaluated across multiple visualization types.  Resampling
    by cluster avoids underestimating CI width from correlated rows.

    Args:
        rows: Result rows with metric and cluster fields.
        metric_key: Field name for the metric to aggregate.
        cluster_key: Field name for the clustering unit.
        n_bootstrap: Number of bootstrap iterations.
        rng: NumPy random generator.

    Returns:
        Dict with value, ci_lower, ci_upper, ci_halfwidth, and n_clusters.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Group rows by cluster.
    from collections import defaultdict

    clusters: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        clusters[str(r.get(cluster_key, ""))].append(float(r.get(metric_key, 0.0)))

    cluster_ids = list(clusters.keys())
    cluster_means = np.array([np.mean(clusters[c]) for c in cluster_ids])
    n_clusters = len(cluster_ids)

    if n_clusters == 0:
        return {"value": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
                "ci_halfwidth": 0.0, "n_clusters": 0}

    observed = float(np.mean(cluster_means))

    boot = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n_clusters, size=n_clusters)
        boot[i] = float(np.mean(cluster_means[idx]))

    ci_lower = float(np.percentile(boot, 2.5))
    ci_upper = float(np.percentile(boot, 97.5))

    return {
        "value": observed,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_halfwidth": (ci_upper - ci_lower) / 2.0,
        "n_clusters": n_clusters,
    }


def _format_value_ci(value: float, ci_low: float, ci_high: float) -> str:
    value_pct = 100.0 * value
    half_width_pct = 50.0 * (ci_high - ci_low)
    return f"{value_pct:.1f} ± {half_width_pct:.1f}"


def _format_table_value(value: float, ci_low: float, ci_high: float) -> str:
    value_pct = 100.0 * value
    half_width_pct = 50.0 * (ci_high - ci_low)
    return f"{value_pct:.1f}{{\\small$\\pm${half_width_pct:.1f}}}"


def _to_code_map(
    values: np.ndarray, order: list[str] | None = None
) -> tuple[np.ndarray, dict[str, int]]:
    if order is None:
        labels = sorted({str(v) for v in values.tolist()})
    else:
        labels = [label for label in order if label in set(values.tolist())]
    code_map = {label: idx for idx, label in enumerate(labels)}
    codes = np.asarray([code_map[str(v)] for v in values.tolist()], dtype=np.int32)
    return codes, code_map


def compute_model_statistics(
    rows: list[dict[str, float | str]],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    exact = np.asarray([float(row["exact_match"]) for row in rows], dtype=np.float64)
    f1 = np.asarray([float(row["f1"]) for row in rows], dtype=np.float64)
    numeric = np.asarray(
        [float(row["numeric_accuracy"]) for row in rows], dtype=np.float64
    )

    modality = np.asarray([str(row["modality"]) for row in rows], dtype=object)
    difficulty = np.asarray([str(row["difficulty"]) for row in rows], dtype=object)
    viz_type = np.asarray([str(row["viz_type"]) for row in rows], dtype=object)

    mod_codes, mod_map = _to_code_map(modality, MODALITY_ORDER)
    diff_codes, diff_map = _to_code_map(difficulty, DIFFICULTY_ORDER)
    viz_labels = sorted({str(v) for v in viz_type.tolist()})
    viz_map = {label: idx for idx, label in enumerate(viz_labels)}
    viz_codes = np.asarray([viz_map[str(v)] for v in viz_type.tolist()], dtype=np.int32)

    modality_to_viz_codes: dict[str, list[int]] = {}
    for mod_label in mod_map:
        mod_code = mod_map[mod_label]
        viz_codes_for_mod = sorted(
            {int(viz_codes[i]) for i in np.where(mod_codes == mod_code)[0].tolist()}
        )
        modality_to_viz_codes[mod_label] = viz_codes_for_mod

    n = exact.shape[0]
    n_mod = len(mod_map)
    n_diff = len(diff_map)
    n_viz = len(viz_map)

    overall_boot = np.zeros(n_bootstrap, dtype=np.float64)
    overall_f1_boot = np.zeros(n_bootstrap, dtype=np.float64)
    overall_numeric_boot = np.zeros(n_bootstrap, dtype=np.float64)
    by_mod_boot = {label: np.zeros(n_bootstrap, dtype=np.float64) for label in mod_map}
    by_diff_boot = {
        label: np.zeros(n_bootstrap, dtype=np.float64) for label in diff_map
    }
    gap_boot = {label: np.zeros(n_bootstrap, dtype=np.float64) for label in mod_map}

    for i in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        exact_s = exact[sample_idx]
        f1_s = f1[sample_idx]
        numeric_s = numeric[sample_idx]
        mod_s = mod_codes[sample_idx]
        diff_s = diff_codes[sample_idx]
        viz_s = viz_codes[sample_idx]

        overall_boot[i] = float(np.mean(exact_s))
        overall_f1_boot[i] = float(np.mean(f1_s))
        overall_numeric_boot[i] = float(np.mean(numeric_s))

        mod_counts = np.bincount(mod_s, minlength=n_mod).astype(np.float64)
        mod_sums = np.bincount(mod_s, weights=exact_s, minlength=n_mod)
        mod_means = np.divide(
            mod_sums,
            mod_counts,
            out=np.zeros_like(mod_sums),
            where=mod_counts > 0,
        )
        for mod_label, mod_code in mod_map.items():
            by_mod_boot[mod_label][i] = float(mod_means[mod_code])

        diff_counts = np.bincount(diff_s, minlength=n_diff).astype(np.float64)
        diff_sums = np.bincount(diff_s, weights=exact_s, minlength=n_diff)
        diff_means = np.divide(
            diff_sums,
            diff_counts,
            out=np.zeros_like(diff_sums),
            where=diff_counts > 0,
        )
        for diff_label, diff_code in diff_map.items():
            by_diff_boot[diff_label][i] = float(diff_means[diff_code])

        viz_counts = np.bincount(viz_s, minlength=n_viz).astype(np.float64)
        viz_sums = np.bincount(viz_s, weights=exact_s, minlength=n_viz)
        viz_means = np.divide(
            viz_sums,
            viz_counts,
            out=np.full_like(viz_sums, np.nan),
            where=viz_counts > 0,
        )
        for mod_label, viz_codes_for_mod in modality_to_viz_codes.items():
            values = viz_means[viz_codes_for_mod]
            values = values[~np.isnan(values)]
            if values.size == 0:
                gap_boot[mod_label][i] = 0.0
            else:
                gap_boot[mod_label][i] = float(np.max(values) - np.min(values))

    overall = float(np.mean(exact))
    overall_f1 = float(np.mean(f1))
    overall_numeric = float(np.mean(numeric))

    by_modality = {
        mod_label: float(np.mean(exact[mod_codes == mod_map[mod_label]]))
        for mod_label in mod_map
    }
    by_difficulty = {
        diff_label: float(np.mean(exact[diff_codes == diff_map[diff_label]]))
        for diff_label in diff_map
    }

    gap_point: dict[str, float] = {}
    for mod_label, viz_codes_for_mod in modality_to_viz_codes.items():
        means: list[float] = []
        for viz_code in viz_codes_for_mod:
            mask = (mod_codes == mod_map[mod_label]) & (viz_codes == viz_code)
            means.append(float(np.mean(exact[mask])))
        gap_point[mod_label] = max(means) - min(means) if means else 0.0

    return {
        "overall_em": {
            "value": overall,
            "ci": _bootstrap_ci(overall_boot),
        },
        "overall_f1": {
            "value": overall_f1,
            "ci": _bootstrap_ci(overall_f1_boot),
        },
        "overall_numeric": {
            "value": overall_numeric,
            "ci": _bootstrap_ci(overall_numeric_boot),
        },
        "modality_em": {
            label: {
                "value": by_modality[label],
                "ci": _bootstrap_ci(by_mod_boot[label]),
            }
            for label in mod_map
        },
        "difficulty_em": {
            label: {
                "value": by_difficulty[label],
                "ci": _bootstrap_ci(by_diff_boot[label]),
            }
            for label in diff_map
        },
        "sensitivity_gap": {
            label: {
                "value": gap_point[label],
                "ci": _bootstrap_ci(gap_boot[label]),
            }
            for label in mod_map
        },
    }


def build_report(stats: dict[str, dict[str, Any]], model_order: list[str]) -> str:
    lines: list[str] = []

    lines.append("# Bootstrap 95% Confidence Intervals (10,000 resamples by default)")
    lines.append("")

    lines.append("## Table 1 rows (LaTeX-ready)")
    for model in model_order:
        model_stats = cast(dict[str, Any], stats[model])
        em = cast(dict[str, Any], model_stats["overall_em"])
        f1 = cast(dict[str, Any], model_stats["overall_f1"])
        numeric = cast(dict[str, Any], model_stats["overall_numeric"])
        em_low, em_high = em["ci"]
        f1_low, f1_high = f1["ci"]
        num_low, num_high = numeric["ci"]
        row = (
            f"{model} & "
            f"{_format_table_value(em['value'], em_low, em_high)} & "
            f"{_format_table_value(f1['value'], f1_low, f1_high)} & "
            f"{_format_table_value(numeric['value'], num_low, num_high)} \\\\"
        )
        lines.append(row)
    lines.append("")

    lines.append("## Overall EM (value ± CI half-width)")
    for model in model_order:
        metric = cast(dict[str, Any], stats[model]["overall_em"])
        ci_low, ci_high = metric["ci"]
        lines.append(f"- {model}: {_format_value_ci(metric['value'], ci_low, ci_high)}")
    lines.append("")

    lines.append("## EM by modality (value ± CI half-width)")
    for model in model_order:
        lines.append(f"- {model}:")
        for modality in MODALITY_ORDER:
            modality_metrics = cast(dict[str, Any], stats[model]["modality_em"])
            metric = cast(dict[str, Any], modality_metrics[modality])
            ci_low, ci_high = metric["ci"]
            lines.append(
                f"  - {modality}: {_format_value_ci(metric['value'], ci_low, ci_high)}"
            )
    lines.append("")

    lines.append("## EM by difficulty (value ± CI half-width)")
    for model in model_order:
        lines.append(f"- {model}:")
        for difficulty in DIFFICULTY_ORDER:
            difficulty_metrics = cast(dict[str, Any], stats[model]["difficulty_em"])
            metric = cast(dict[str, Any], difficulty_metrics[difficulty])
            ci_low, ci_high = metric["ci"]
            lines.append(
                f"  - {difficulty}: {_format_value_ci(metric['value'], ci_low, ci_high)}"
            )
    lines.append("")

    lines.append(
        "## Visualization sensitivity gap by modality (pp, value ± CI half-width)"
    )
    for model in model_order:
        lines.append(f"- {model}:")
        for modality in MODALITY_ORDER:
            gap_metrics = cast(dict[str, Any], stats[model]["sensitivity_gap"])
            metric = cast(dict[str, Any], gap_metrics[modality])
            ci_low, ci_high = metric["ci"]
            lines.append(
                f"  - {modality}: {_format_value_ci(metric['value'], ci_low, ci_high)}"
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    results_dir = _resolve_results_dir(args.results_dir)
    output_path = results_dir / "confidence_intervals.txt"
    model_files = dict(CORE_MODEL_FILES)
    model_order = list(CORE_MODEL_ORDER)
    if cast(bool, args.include_supplementary_local):
        model_files.update(SUPPLEMENTARY_MODEL_FILES)
        model_order.extend(SUPPLEMENTARY_MODEL_ORDER)

    rng = np.random.default_rng(args.seed)
    model_stats: dict[str, dict[str, Any]] = {}

    cluster_rows: list[str] = []
    cluster_rows.append("")
    cluster_rows.append("## Cluster-aware Bootstrap (resampled by question_id)")
    cluster_rows.append(
        "Row-level iid bootstrap (above) ignores the repeated-measures structure"
    )
    cluster_rows.append(
        "where each base question is evaluated across multiple visualization types."
    )
    cluster_rows.append(
        "The cluster-aware bootstrap below resamples by question_id and produces"
    )
    cluster_rows.append(
        "wider, more conservative CIs that properly account for within-question"
    )
    cluster_rows.append("correlation among rendered viz instances.")
    cluster_rows.append("")

    for model in model_order:
        file_path = results_dir / model_files[model]
        if not file_path.exists():
            raise FileNotFoundError(f"Missing result file: {file_path}")
        rows = _load_jsonl_rows(file_path)
        model_stats[model] = compute_model_statistics(
            rows=rows,
            n_bootstrap=args.n_bootstrap,
            rng=rng,
        )

        # Cluster-aware bootstrap on overall EM (resampling by question_id).
        cluster_result = compute_cluster_bootstrap_ci(
            rows=cast(list[dict[str, float | str]], rows),
            metric_key="exact_match",
            cluster_key="question_id",
            n_bootstrap=args.n_bootstrap,
            rng=rng,
        )
        cluster_rows.append(
            f"- {model}: EM = {cluster_result['value']*100:.2f}% "
            f"(cluster 95% CI [{cluster_result['ci_lower']*100:.2f}, "
            f"{cluster_result['ci_upper']*100:.2f}], "
            f"half-width {cluster_result['ci_halfwidth']*100:.2f}pp, "
            f"n_clusters={int(cluster_result['n_clusters'])})"
        )

    report = build_report(model_stats, model_order)
    report += "\n".join(cluster_rows) + "\n"
    output_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote confidence intervals: {output_path}")


if __name__ == "__main__":
    main()
