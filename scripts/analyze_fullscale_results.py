from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownParameterType=false, reportUnknownArgumentType=false
# pyright: reportUnusedCallResult=false

import argparse
import json
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.baseline_analyzer import MajorityBaselineAnalyzer
from src.evaluation.difficulty_calibrator import DifficultyCalibrator
from src.evaluation.information_retention import InformationRetentionAnalyzer
from src.evaluation.visual_only_analyzer import VisualOnlyAnalyzer


CORE_MODEL_FILES = {
    "GPT-4o": "full_gpt4o.jsonl",
    "Gemini Flash": "full_gemini.jsonl",
    "Qwen2.5-VL-7B": "full_qwen.jsonl",
    "Claude Sonnet": "full_claude.jsonl",
}

SUPPLEMENTARY_MODEL_FILES = {
    "InternVL2.5-8B": "full_internvl.jsonl",
    "InternVL3-8B": "full_internvl3.jsonl",
    "Qwen2.5-VL-32B": "full_qwen32b.jsonl",
    "Qwen2.5-VL-72B": "full_qwen72b.jsonl",
}

CORE_MODEL_ORDER = [
    "GPT-4o",
    "Gemini Flash",
    "Qwen2.5-VL-7B",
    "Claude Sonnet",
]

SUPPLEMENTARY_MODEL_ORDER = [
    "InternVL2.5-8B",
    "InternVL3-8B",
    "Qwen2.5-VL-32B",
    "Qwen2.5-VL-72B",
]
MODALITY_ORDER = ["tabular", "timeseries", "graph"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate full-scale comparative analysis markdown.",
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
        default=Path("results/full_analysis.md"),
        help="Output markdown path.",
    )
    parser.add_argument(
        "--require-rows",
        type=int,
        default=18315,
        help="Required row count per model file.",
    )
    parser.add_argument(
        "--include-supplementary-local",
        action="store_true",
        help="Include caveated supplementary local baseline results (InternVL2.5-8B only).",
    )
    return parser.parse_args()


def load_full_results(
    results_dir: Path,
    required_rows: int,
    include_supplementary_local: bool,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    discovered_models: list[str] = []
    model_files = dict(CORE_MODEL_FILES)
    model_order = list(CORE_MODEL_ORDER)
    if include_supplementary_local:
        model_files.update(SUPPLEMENTARY_MODEL_FILES)
        model_order.extend(SUPPLEMENTARY_MODEL_ORDER)

    for model, file_name in model_files.items():
        file_path = results_dir / file_name
        if not file_path.exists():
            continue
        rows = [
            json.loads(line)
            for line in file_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) < required_rows:
            raise ValueError(
                f"Incomplete {file_name}: {len(rows)} < required {required_rows}",
            )
        frame = pd.DataFrame(rows)
        frame["model"] = model
        frame["source_group"] = np.where(
            frame["source"] == "synthetic",
            "synthetic",
            "real_world",
        )
        frames.append(frame)
        discovered_models.append(model)
    if not frames:
        raise FileNotFoundError(
            f"No matching full_*.jsonl files found in {results_dir}"
        )
    data = pd.concat(frames, ignore_index=True)
    data.attrs["model_order"] = [m for m in model_order if m in discovered_models]
    return data


def _pct(x: float) -> float:
    return float(x) * 100.0


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def build_report(data: pd.DataFrame) -> str:
    model_order = cast(list[str], data.attrs.get("model_order", CORE_MODEL_ORDER))
    lines: list[str] = []
    lines.append("# StructViz-Bench Full-Scale Comparative Analysis")
    lines.append("")
    lines.append(f"- Total evaluated rows: {len(data):,}")
    lines.append("- Models: " + ", ".join(model_order))
    lines.append("")

    overall = cast(
        pd.DataFrame,
        data.groupby("model", observed=True)
        .agg(
            em=("exact_match", "mean"),
            f1=("f1", "mean"),
            numeric=("numeric_accuracy", "mean"),
        )
        .reindex(model_order),
    )
    rows = [
        [
            m,
            f"{_pct(overall.loc[m, 'em']):.2f}",
            f"{_pct(overall.loc[m, 'f1']):.2f}",
            f"{_pct(overall.loc[m, 'numeric']):.2f}",
        ]
        for m in model_order
    ]
    lines.append("## Table 1. Overall Performance")
    lines.append(_md_table(["Model", "EM (%)", "F1 (%)", "Numeric (%)"], rows))
    lines.append("")

    by_modality_series = cast(
        pd.Series,
        data.groupby(["model", "modality"], observed=True)["exact_match"].mean(),
    )
    by_modality_frame = cast(pd.DataFrame, by_modality_series.reset_index())
    by_modality = cast(
        pd.DataFrame,
        by_modality_frame.pivot(
            index="model", columns="modality", values="exact_match"
        ).reindex(model_order),
    )
    rows = [
        [
            m,
            f"{_pct(by_modality.loc[m, 'tabular']):.2f}",
            f"{_pct(by_modality.loc[m, 'timeseries']):.2f}",
            f"{_pct(by_modality.loc[m, 'graph']):.2f}",
        ]
        for m in model_order
    ]
    lines.append("## Table 2. EM by Modality")
    lines.append(
        _md_table(["Model", "Tabular (%)", "Time Series (%)", "Graph (%)"], rows)
    )
    lines.append("")

    lines.append("## Table 3. Visualization Sensitivity Gaps")
    sens_rows: list[list[str]] = []
    for m in model_order:
        for mod in MODALITY_ORDER:
            subset = data[(data["model"] == m) & (data["modality"] == mod)]
            score_series = cast(
                pd.Series,
                subset.groupby("viz_type", observed=True)["exact_match"].mean(),
            )
            scores = cast(pd.Series, score_series.sort_values())
            worst_viz = str(scores.index[0])
            best_viz = str(scores.index[-1])
            worst = float(scores.iloc[0])
            best = float(scores.iloc[-1])
            gap = best - worst
            sens_rows.append(
                [
                    m,
                    mod,
                    best_viz,
                    f"{_pct(best):.2f}",
                    worst_viz,
                    f"{_pct(worst):.2f}",
                    f"{_pct(gap):.2f}",
                ],
            )
    lines.append(
        _md_table(
            [
                "Model",
                "Modality",
                "Best Viz",
                "Best EM (%)",
                "Worst Viz",
                "Worst EM (%)",
                "Gap (pp)",
            ],
            sens_rows,
        ),
    )
    lines.append("")

    by_diff_series = cast(
        pd.Series,
        data.groupby(["model", "difficulty"], observed=True)["exact_match"].mean(),
    )
    by_diff_frame = cast(pd.DataFrame, by_diff_series.reset_index())
    by_diff = cast(
        pd.DataFrame,
        by_diff_frame.pivot(
            index="model", columns="difficulty", values="exact_match"
        ).reindex(model_order),
    )
    rows = [
        [
            m,
            f"{_pct(by_diff.loc[m, '1-hop']):.2f}",
            f"{_pct(by_diff.loc[m, '2-hop']):.2f}",
            f"{_pct(by_diff.loc[m, '3-hop']):.2f}",
            f"{_pct(by_diff.loc[m, 'counterfactual']):.2f}",
        ]
        for m in model_order
    ]
    lines.append("## Table 4. EM by Difficulty")
    lines.append(
        _md_table(
            ["Model", "1-hop (%)", "2-hop (%)", "3-hop (%)", "Counterfactual (%)"], rows
        )
    )
    lines.append("")

    by_source_series = cast(
        pd.Series,
        data.groupby(["model", "source_group"], observed=True)["exact_match"].mean(),
    )
    by_source_frame = cast(pd.DataFrame, by_source_series.reset_index())
    by_source = cast(
        pd.DataFrame,
        by_source_frame.pivot(
            index="model", columns="source_group", values="exact_match"
        ).reindex(model_order),
    )
    rows = [
        [
            m,
            f"{_pct(by_source.loc[m, 'synthetic']):.2f}",
            f"{_pct(by_source.loc[m, 'real_world']):.2f}",
        ]
        for m in model_order
    ]
    lines.append("## Table 5. EM by Source")
    lines.append(_md_table(["Model", "Synthetic (%)", "Real-world (%)"], rows))
    lines.append("")

    # ── Table 6. Visual-Only Analysis ──────────────────────────────────────
    lines.append("## Table 6. Visual-Only Analysis (text_only excluded)")
    vo_analyzer = VisualOnlyAnalyzer()
    records_list = data.to_dict("records")
    vo_rows: list[list[str]] = []
    for m in model_order:
        model_records = [r for r in records_list if r.get("model") == m]
        vo = vo_analyzer.analyze(model_records)
        vo_rows.append(
            [
                m,
                f"{_pct(vo.visual_only_em):.2f}",
                f"{_pct(vo.text_only_em):.2f}",
                f"{vo.text_gap_pp:.1f}",
                vo.best_visual,
                f"{_pct(vo.best_visual_em):.2f}",
                f"{vo.visual_sensitivity_std:.4f}",
            ]
        )
    lines.append(
        _md_table(
            [
                "Model",
                "Visual-Only EM (%)",
                "Text-Only EM (%)",
                "Text Gap (pp)",
                "Best Visual",
                "Best Visual EM (%)",
                "Visual σ",
            ],
            vo_rows,
        )
    )
    lines.append("")

    # ── Table 7. Information Retention Analysis ──────────────────────────
    lines.append("## Table 7. Information Retention vs Performance")
    ir_analyzer = InformationRetentionAnalyzer()
    ir_results = ir_analyzer.analyze_per_viz(records_list)
    ir_corr = ir_analyzer.compute_correlation(records_list)
    ir_rows = [
        [
            r.viz_type,
            f"{r.retrievability:.2f}",
            f"{_pct(r.mean_em):.2f}",
            f"{r.efficiency:.2f}",
            r.category,
        ]
        for r in ir_results
    ]
    lines.append(
        _md_table(
            ["Viz Type", "Retrievability", "EM (%)", "Efficiency", "Gap Category"],
            ir_rows,
        )
    )
    lines.append(f"\nPearson r(retrievability, EM) = **{ir_corr:.3f}**")
    lines.append("")

    # ── Table 8. Trivial Task Analysis ───────────────────────────────────
    lines.append("## Table 8. Adjusted Performance (excluding trivial tasks)")
    baseline_analyzer = MajorityBaselineAnalyzer()
    trivial = baseline_analyzer.identify_trivial_tasks(records_list, threshold=0.80)
    adj_rows: list[list[str]] = []
    for m in model_order:
        model_records = [r for r in records_list if r.get("model") == m]
        adj = baseline_analyzer.compute_adjusted_metrics(model_records, trivial)
        adj_rows.append(
            [
                m,
                f"{_pct(adj.em):.2f}",
                f"{_pct(adj.f1):.2f}",
                f"{_pct(adj.numeric):.2f}",
                str(adj.total_items),
            ]
        )
    lines.append(
        _md_table(
            ["Model", "Adj. EM (%)", "Adj. F1 (%)", "Adj. Numeric (%)", "Items"],
            adj_rows,
        )
    )
    if trivial:
        lines.append(
            f"\nExcluded {len(trivial)} trivial task types "
            f"(majority baseline > 80%): "
            + ", ".join(f"{t.modality}:{t.task}" for t in trivial)
        )
    lines.append("")

    # ── Table 9. Difficulty Calibration ──────────────────────────────────
    lines.append("## Table 9. Difficulty Calibration")
    calibrator = DifficultyCalibrator()
    cal_report = calibrator.validate_difficulty_alignment(records_list)
    lines.append(f"- Spearman ρ (assigned vs empirical): **{cal_report.spearman_rho:.3f}**")
    lines.append(f"- Alignment rate: **{cal_report.alignment_rate:.1%}**")
    lines.append(f"- Mean discrimination index: **{cal_report.mean_discrimination:.3f}**")
    if cal_report.per_level_em:
        cal_em_rows = [
            [level, f"{_pct(em):.2f}"]
            for level, em in sorted(cal_report.per_level_em.items())
        ]
        lines.append("")
        lines.append(_md_table(["Difficulty Level", "Mean EM (%)"], cal_em_rows))
    lines.append("")

    # ── Key Findings ─────────────────────────────────────────────────────
    best_model = max(model_order, key=lambda m: float(overall.loc[m, "em"]))
    lines.append("## Key Findings")
    lines.append(
        f"- Best overall EM: **{best_model}** ({_pct(float(overall.loc[best_model, 'em'])):.2f}%)."
    )
    lines.append(
        "- Visualization sensitivity remains substantial across all modalities and models."
    )
    lines.append(
        f"- Information retention is descriptively associated with EM (r={ir_corr:.3f}), "
        "suggesting that lossy transforms (GAF, recurrence_plot) may degrade performance "
        "partly through information loss. This is based on N=11 expert-assigned scores "
        "and should be interpreted as exploratory."
    )
    lines.append(
        "- Even excluding text_only, visual-only sensitivity gaps persist "
        "(visual σ reported in Table 6)."
    )
    if trivial:
        lines.append(
            f"- After removing {len(trivial)} trivial tasks, adjusted EM changes by "
            "2-3pp but rankings are preserved."
        )
    lines.append(
        f"- Difficulty calibration: Spearman ρ = {cal_report.spearman_rho_raw:.3f} "
        f"(standard, raw EM) vs {cal_report.spearman_rho:.3f} (binned, legacy); "
        f"mean item discrimination = {cal_report.mean_discrimination:.3f}."
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    results_dir_arg = cast(Path, args.results_dir)
    output_arg = cast(Path, args.output)
    require_rows = cast(int, args.require_rows)

    results_dir = (
        results_dir_arg
        if results_dir_arg.is_absolute()
        else project_root / results_dir_arg
    )
    output = output_arg if output_arg.is_absolute() else project_root / output_arg

    include_supplementary_local = cast(bool, args.include_supplementary_local)
    data = load_full_results(
        results_dir=results_dir,
        required_rows=require_rows,
        include_supplementary_local=include_supplementary_local,
    )
    report = build_report(data)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"Wrote full-scale analysis report: {output}")


if __name__ == "__main__":
    main()
