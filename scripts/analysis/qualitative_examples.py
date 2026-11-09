"""Write the appendix table of per-format predictions for three example items (v1 suite, seven
models) plus the corrected-suite (v2) predictions of Qwen2.5-VL-7B. Output: paper_tables/tab_qual.tex."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ITEMS = ["tabular_001326::difficulty=2-hop", "timeseries_000027::difficulty=1-hop",
         "graph_000076::difficulty=1-hop"]
MODELS = [("GPT-4o", "full_gpt4o_extracted.jsonl"), ("Gemini-2.0-Flash", "full_gemini_extracted.jsonl"),
          ("Gemini-2.5-Flash", "full_gemini25.jsonl"), ("Claude Sonnet 4", "full_claude_extracted.jsonl"),
          ("Qwen2.5-VL-7B", "full_qwen_extracted.jsonl"), ("Qwen2.5-VL-32B", "full_qwen32b.jsonl"),
          ("InternVL2.5-8B", "full_internvl.jsonl")]
FMT = {"tabular": ["table_image", "text_only", "heatmap", "bar_chart", "scatter_plot"],
       "timeseries": ["line_plot", "heatmap", "gaf", "recurrence_plot", "text_only"],
       "graph": ["node_link", "circular_layout", "adjacency_matrix", "text_only"]}


def esc(s: str) -> str:
    return (str(s).replace("\\", "/").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
            .replace("#", "\\#").replace("$", "\\$"))


def main() -> None:
    base = {}
    for line in (ROOT / "benchmark/realworld_test.jsonl").open():
        r = json.loads(line)
        if r["question_id"] in ITEMS:
            base[r["question_id"]] = r
    pred = {}
    for name, fn in MODELS:
        for line in (ROOT / "results" / fn).open():
            r = json.loads(line)
            if r["question_id"] in ITEMS:
                pred[(name, r["question_id"], r["viz_type"])] = (r["prediction"], float(r.get("exact", r.get("exact_match", 0)) or 0))
    v2 = {}
    for line in (ROOT / "results/v2/v2_qwen.jsonl").open():
        r = json.loads(line)
        if r["question_id"] in ITEMS:
            v2[(r["question_id"], r["viz_type"])] = (r["prediction"], float(r.get("exact_match", 0) or 0))
    out = []
    for qid in ITEMS:
        b = base[qid]; fm = FMT[b["modality"]]
        out.append("\\multicolumn{%d}{@{}p{0.97\\textwidth}@{}}{\\emph{%s} --- %s \\quad key: \\texttt{%s}} \\\\" % (
            len(fm) + 1, esc(qid.split("::")[0]), esc(b["question"]), esc(b["answer"])))
        out.append("model & " + " & ".join("\\texttt{%s}" % esc(f) for f in fm) + " \\\\ \\midrule")
        for name, _ in MODELS:
            cells = []
            for f in fm:
                p, ok = pred.get((name, qid, f), ("--", 0))
                cells.append(("\\textbf{%s}" if ok == 1 else "%s") % esc(p[:14]))
            out.append(esc(name) + " & " + " & ".join(cells) + " \\\\")
        cells = []
        for f in fm:
            p, ok = v2.get((qid, f), ("--", 0))
            cells.append(("\\textbf{%s}" if ok == 1 else "%s") % esc(p[:14]))
        out.append("Qwen2.5-VL-7B, corrected suite & " + " & ".join(cells) + " \\\\ \\midrule")
    tex = ["% per-format predictions for three example items; bold = exact match with the key",
           "\\begin{tabular}{l" + "c" * 5 + "}", "\\toprule"] + out[:-1] + [out[-1].replace(" \\midrule", ""), "\\bottomrule", "\\end{tabular}"]
    (ROOT / "paper_tables/tab_qual.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(tex))


if __name__ == "__main__":
    main()
