"""Figure: where accuracy is lost, per modality, for Qwen2.5-VL-7B (normalised exact match).

For each modality, the question-only prior, the range from worst to best format on the default
suite and on the expressive suite, and the accuracy when the true data are given as text.
Reads scripts/analysis/canonical_results.json; writes paper_tables/figures/fig_ladder.pdf.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
D = json.loads((ROOT / "scripts/analysis/canonical_results.json").read_text())
MODEL = "Qwen2.5-VL-7B"
PRIOR = {"tabular": 24.0, "timeseries": 40.8, "graph": 40.9}  # CV prior, Section 4
NAMES = {"tabular": "Tables", "timeseries": "Time series", "graph": "Graphs"}


def v1_formats(md: str) -> dict:
    return D["perviz"]["canonical"][MODEL][md]


def v2_formats(md: str) -> dict:
    pf = D["v2"]["canonical"][MODEL]["per_format"]
    return {k.split("|")[1]: v["em"] for k, v in pf.items()
            if k.startswith(md + "|") and not k.endswith("text_only_deg")}


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5})
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 1.75), sharey=True)
    rows = ["True data as text", "Expressive suite", "Default suite", "Question-only prior"]
    out = {}
    for ax, md in zip(axes, ["tabular", "timeseries", "graph"]):
        a, b = v1_formats(md), v2_formats(md)
        raw = D["rawtext"]["canonical"][md]["em"]
        y = {r: i for i, r in enumerate(rows)}
        ax.axvline(PRIOR[md], color="#9aa5b4", lw=0.8, ls="--", zorder=0)
        for row, fm, col in [("Default suite", a, "#b0413e"), ("Expressive suite", b, "#24618a")]:
            lo, hi = min(fm.values()), max(fm.values())
            ax.plot([lo, hi], [y[row]] * 2, color=col, lw=3, solid_capstyle="round", alpha=0.35)
            for f, v in fm.items():
                ax.plot(v, y[row], "o", ms=3.2, color=col)
            ax.text(hi + 1.2, y[row], f"{hi - lo:.0f} pt", va="center", fontsize=6.5, color=col)
            out[(md, row)] = (round(lo, 1), round(hi, 1))
        ax.plot(raw, y["True data as text"], "D", ms=4, color="#2f7d4f")
        ax.plot(PRIOR[md], y["Question-only prior"], "s", ms=3.5, color="#6b7686")
        out[(md, "raw")] = round(raw, 1)
        ax.set_title(NAMES[md], fontsize=8)
        ax.set_xlim(0, 85)
        ax.set_xticks([0, 20, 40, 60, 80])
        ax.grid(axis="x", color="#e5e8ec", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlabel("Exact match (%)", fontsize=7)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels(rows)
    axes[0].set_ylim(-0.6, len(rows) - 0.4)
    fig.tight_layout(pad=0.3, w_pad=0.6)
    dst = ROOT / "paper_tables/figures/fig_ladder.pdf"
    fig.savefig(dst)
    print("wrote", dst)
    for k, v in out.items():
        print(k, v)


if __name__ == "__main__":
    main()
