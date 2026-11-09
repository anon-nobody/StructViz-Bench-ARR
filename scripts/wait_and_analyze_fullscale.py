from __future__ import annotations

import subprocess
import time
from pathlib import Path


TARGET_ROWS = 18315
POLL_SECONDS = 120


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    full_files = {
        "gpt4o": results_dir / "full_gpt4o.jsonl",
        "gemini": results_dir / "full_gemini.jsonl",
        "claude": results_dir / "full_claude.jsonl",
        "qwen": results_dir / "full_qwen.jsonl",
    }

    print("[watcher] Waiting for all full_* result files to reach 18,315 rows...")
    while True:
        counts = {name: _count_lines(path) for name, path in full_files.items()}
        status = " ".join(f"{k}={v}" for k, v in counts.items())
        print(f"[watcher] {status}", flush=True)

        if all(v >= TARGET_ROWS for v in counts.values()):
            print(
                "[watcher] All model runs complete. Starting full-scale analysis...",
                flush=True,
            )
            break

        time.sleep(POLL_SECONDS)

    figures_cmd = [
        "python",
        "scripts/generate_paper_figures.py",
        "--split",
        "full",
        "--output-dir",
        "paper/figures_full",
    ]
    with (results_dir / "full_tables.txt").open("w", encoding="utf-8") as out:
        _ = subprocess.run(
            figures_cmd,
            cwd=project_root,
            check=True,
            stdout=out,
            stderr=subprocess.STDOUT,
        )

    analysis_cmd = [
        "python",
        "scripts/analyze_fullscale_results.py",
        "--results-dir",
        "results",
        "--output",
        "results/full_analysis.md",
        "--require-rows",
        str(TARGET_ROWS),
    ]
    _ = subprocess.run(
        analysis_cmd,
        cwd=project_root,
        check=True,
    )

    print("[watcher] Full-scale tables written to results/full_tables.txt", flush=True)
    print("[watcher] Full-scale figures written to paper/figures_full/", flush=True)
    print("[watcher] Full-scale report written to results/full_analysis.md", flush=True)


if __name__ == "__main__":
    main()
