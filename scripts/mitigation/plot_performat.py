#!/usr/bin/env python3
"""Per-format exact-match heatmap (Figure 2). Pure matplotlib, publication style.
Numbers are the released per-format EM (main paper Table 2). Saves a vector PDF."""
import matplotlib
matplotlib.use("Agg")
# TrueType (42) rather than the default Type 3, so the figure's text stays searchable
# and selectable in the submitted PDF.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import os

models = ["GPT-4o", "Gemini Flash", "Qwen2.5-VL-7B", "Claude Sonnet"]
panels = {
    "Tabular": (["bar", "heatmap", "scatter", "table_img", "text_only"], [
        [22.8, 43.6, 23.6, 59.5, 57.6],
        [25.0, 44.7, 20.6, 60.3, 61.1],
        [25.9, 41.7, 17.5, 54.6, 49.0],
        [18.9, 34.8, 16.1, 44.5, 36.5]]),
    "Time series": (["gaf", "heatmap", "line", "recur.", "text_only"], [
        [17.2, 18.5, 27.6, 10.1, 35.8],
        [19.5, 24.9, 33.0, 19.7, 38.7],
        [21.2, 21.0, 27.3, 19.2, 32.8],
        [14.1, 17.0, 23.3, 11.3, 22.0]]),
    "Graph": (["adj_mat", "circular", "node_link", "text_only"], [
        [33.3, 41.2, 43.6, 56.4],
        [31.8, 36.2, 35.8, 53.5],
        [28.2, 33.9, 36.2, 43.0],
        [36.5, 37.9, 34.8, 49.5]]),
}

fig, axes = plt.subplots(1, 3, figsize=(11, 2.9),
                         gridspec_kw={"width_ratios": [5, 5, 4]})
vmin, vmax = 10, 62
for ax, (title, (cols, data)) in zip(axes, panels.items()):
    arr = np.array(data)
    im = ax.imshow(arr, cmap="YlGnBu", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models if ax is axes[0] else [""] * len(models), fontsize=8)
    ax.set_title(title, fontsize=10, pad=4)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 40 else "black")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(models), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="both", length=0)
cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01)
cbar.set_label("Exact match (%)", fontsize=8); cbar.ax.tick_params(labelsize=7)
_out = os.environ.get("OUT_PDF", os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "paper_tables", "fig_performat.pdf"))
fig.savefig(_out,
            bbox_inches="tight")
print(f"saved {_out}")
