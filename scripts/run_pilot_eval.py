from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnusedCallResult=false

import argparse
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

from src.evaluation.metrics import compute_metrics
from src.models.api_models import GPT4oModel, GeminiModel, ClaudeModel
from src.rendering.render_pipeline import RenderPipeline
from src.utils.io_utils import BenchmarkItem, read_jsonl, write_jsonl

# ── API Keys (read from environment) ──────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "gpt4o": {
        "name": "gpt-4o",
        "rpm": 50,
        "env_var": "OPENAI_API_KEY",
        "output": "pilot_gpt4o.jsonl",
    },
    "gemini": {
        "name": "gemini-2.0-flash",
        "rpm": 15,
        "env_var": "GEMINI_API_KEY",
        "output": "pilot_gemini.jsonl",
    },
    "qwen": {
        "name": "Qwen2.5-VL-7B-Instruct",
        "rpm": 999,
        "env_var": "",
        "output": "pilot_qwen.jsonl",
    },
    "claude": {
        "name": "claude-sonnet-4-20250514",
        "rpm": 40,
        "env_var": "ANTHROPIC_API_KEY",
        "output": "pilot_claude.jsonl",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pilot visualization-sensitivity eval on StructViz-Bench.",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("benchmark/realworld_test.jsonl"),
        help="Benchmark JSONL path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for pilot result JSONL files.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["gpt4o", "gemini", "claude", "qwen", "both"],
        default="both",
        help="Which API model(s) to run.",
    )
    parser.add_argument(
        "--num-items",
        type=int,
        default=500,
        help="Total benchmark items to sample before viz expansion.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output JSONL and skip completed item-viz rows.",
    )
    return parser.parse_args()


def _modality_targets(total: int) -> dict[str, int]:
    if total <= 0:
        raise ValueError("--num-items must be > 0")
    base = total // 3
    rem = total % 3
    targets = {
        "tabular": base,
        "timeseries": base,
        "graph": base,
    }
    if rem >= 1:
        targets["tabular"] += 1
    if rem >= 2:
        targets["timeseries"] += 1
    return targets


def _sample_with_source_mix(
    rng: random.Random,
    modality_items: list[BenchmarkItem],
    target: int,
) -> list[BenchmarkItem]:
    synthetic = [item for item in modality_items if item.source == "synthetic"]
    realworld = [item for item in modality_items if item.source != "synthetic"]

    desired_syn = target // 2
    desired_real = target - desired_syn
    take_syn = min(len(synthetic), desired_syn)
    take_real = min(len(realworld), desired_real)

    remaining = target - (take_syn + take_real)
    syn_left = len(synthetic) - take_syn
    real_left = len(realworld) - take_real
    while remaining > 0 and (syn_left > 0 or real_left > 0):
        if real_left > syn_left and real_left > 0:
            take_real += 1
            real_left -= 1
        elif syn_left > 0:
            take_syn += 1
            syn_left -= 1
        elif real_left > 0:
            take_real += 1
            real_left -= 1
        remaining -= 1

    if (take_syn + take_real) < target:
        raise ValueError(
            f"Not enough {modality_items[0].modality if modality_items else 'items'} data to sample target={target}",
        )

    sampled: list[BenchmarkItem] = []
    if take_syn > 0:
        sampled.extend(rng.sample(synthetic, take_syn))
    if take_real > 0:
        sampled.extend(rng.sample(realworld, take_real))
    return sampled


def sample_items(
    benchmark_path: Path, num_items: int, seed: int
) -> list[BenchmarkItem]:
    rows = read_jsonl(benchmark_path)
    all_items = [BenchmarkItem.from_dict(row) for row in rows]
    targets = _modality_targets(num_items)

    by_modality: dict[str, list[BenchmarkItem]] = defaultdict(list)
    for item in all_items:
        by_modality[item.modality].append(item)

    rng = random.Random(seed)
    sampled: list[BenchmarkItem] = []
    for modality in ("tabular", "timeseries", "graph"):
        sampled.extend(
            _sample_with_source_mix(
                rng=rng,
                modality_items=by_modality.get(modality, []),
                target=targets[modality],
            )
        )
    rng.shuffle(sampled)
    return sampled


def _build_model(model_key: str) -> Any:
    if model_key == "gpt4o":
        os_key = MODEL_SPECS["gpt4o"]["env_var"]
        import os

        os.environ[os_key] = OPENAI_API_KEY
        return GPT4oModel(
            name="gpt-4o", requests_per_minute=MODEL_SPECS["gpt4o"]["rpm"]
        )

    if model_key == "claude":
        import os

        os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
        return ClaudeModel(
            name="claude-sonnet-4-20250514",
            requests_per_minute=MODEL_SPECS["claude"]["rpm"],
        )

    if model_key == "qwen":
        from src.models.local_models import QwenVLModel

        return QwenVLModel(name="Qwen2.5-VL-7B-Instruct", device="cuda")

    os_key = MODEL_SPECS["gemini"]["env_var"]
    import os

    os.environ[os_key] = GEMINI_API_KEY
    return GeminiModel(
        name="gemini-2.0-flash",
        requests_per_minute=MODEL_SPECS["gemini"]["rpm"],
    )


def _extract_retry_seconds(error: Exception) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers and isinstance(headers, dict):
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                return None
    return None


def _call_with_retries(
    model: Any,
    question: str,
    image: Any,
    metadata: dict[str, Any],
    max_retries: int = 3,
) -> str:
    for attempt in range(max_retries):
        try:
            return str(model.answer(question=question, image=image, metadata=metadata))
        except Exception as error:  # noqa: BLE001
            message = str(error)
            is_rate_limited = "429" in message or "rate limit" in message.lower()
            if attempt == max_retries - 1:
                print(
                    f"[ERROR] API failed after retries: {message}",
                    file=sys.stderr,
                )
                return "[ERROR]"

            if is_rate_limited:
                wait = _extract_retry_seconds(error)
                if wait is None:
                    wait = 2**attempt
            else:
                wait = 1.5**attempt

            print(
                f"[WARN] API error (attempt {attempt + 1}/{max_retries}): {message}; sleeping {wait:.1f}s",
                file=sys.stderr,
            )
            time.sleep(wait)
    return "[ERROR]"


def _metrics_row(item: BenchmarkItem, viz_type: str, prediction: str) -> dict[str, Any]:
    bundle = compute_metrics(prediction=prediction, answer=item.answer)
    return {
        "question_id": item.question_id,
        "question": item.question,
        "answer": item.answer,
        "prediction": prediction,
        "modality": item.modality,
        "source": item.source,
        "viz_type": viz_type,
        "difficulty": item.difficulty,
        "task": item.task,
        "exact_match": bundle.exact,
        "f1": bundle.f1,
        "numeric_accuracy": bundle.numeric,
    }


def _print_summary(model_name: str, rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[str(row["modality"])][str(row["viz_type"])].append(row)

    print("=== Viz Sensitivity Analysis ===")
    print()
    print(f"Model: {model_name}")

    sensitivity_scores: dict[str, float] = {}
    for modality in ("tabular", "timeseries", "graph"):
        viz_map = grouped.get(modality, {})
        print(f"  {modality}:")
        exact_values: list[float] = []
        for viz_type in sorted(viz_map.keys()):
            rows_for_viz = viz_map[viz_type]
            exact = sum(float(r["exact_match"]) for r in rows_for_viz) / len(
                rows_for_viz
            )
            f1 = sum(float(r["f1"]) for r in rows_for_viz) / len(rows_for_viz)
            numeric = sum(float(r["numeric_accuracy"]) for r in rows_for_viz) / len(
                rows_for_viz
            )
            exact_values.append(exact)
            print(
                f"    {viz_type:<16} exact_match={exact:.3f} f1={f1:.3f} numeric={numeric:.3f}",
            )
        sensitivity_scores[modality] = (
            max(exact_values) - min(exact_values) if exact_values else 0.0
        )

    print()
    print("  Viz Sensitivity Score (per modality):")
    for modality in ("tabular", "timeseries", "graph"):
        print(f"    {modality}: {sensitivity_scores.get(modality, 0.0):.3f}")
    print()


def run_model_eval(
    model_key: str,
    sampled_items: list[BenchmarkItem],
    output_path: Path,
    resume: bool,
) -> list[dict[str, Any]]:
    model = _build_model(model_key)
    rpm = int(MODEL_SPECS[model_key]["rpm"])
    min_interval = 60.0 / float(rpm)

    existing_rows: list[dict[str, Any]] = []
    done_keys: set[tuple[str, str]] = set()
    if resume and output_path.exists():
        existing_rows = read_jsonl(output_path)
        for row in existing_rows:
            done_keys.add(
                (str(row.get("question_id", "")), str(row.get("viz_type", "")))
            )
        print(
            f"[INFO] Resume loaded {len(existing_rows)} existing rows from {output_path}"
        )

    pipeline = RenderPipeline()
    results = list(existing_rows)
    eval_rows = 0
    errors = 0
    last_call_ts = 0.0

    for idx, item in enumerate(sampled_items, start=1):
        try:
            render_input = {
                "modality": item.modality,
                "data": item.data,
                "data_meta": item.metadata.get("data_meta") if item.metadata else None,
                "viz_titles": item.metadata.get("viz_titles", {})
                if item.metadata
                else {},
            }
            rendered = pipeline.render_all(render_input)
        except Exception as error:  # noqa: BLE001
            print(
                f"[ERROR] Render failed for {item.question_id}: {error}",
                file=sys.stderr,
            )
            for viz_type in item.viz_methods:
                key = (item.question_id, viz_type)
                if key in done_keys:
                    continue
                row = _metrics_row(item=item, viz_type=viz_type, prediction="[ERROR]")
                results.append(row)
                done_keys.add(key)
                errors += 1
            continue

        viz_sequence = item.viz_methods if item.viz_methods else list(rendered.keys())
        for viz_type in viz_sequence:
            if viz_type not in rendered:
                continue
            key = (item.question_id, viz_type)
            if key in done_keys:
                continue

            now = time.time()
            elapsed = now - last_call_ts
            if last_call_ts > 0 and elapsed < min_interval:
                time.sleep(min_interval - elapsed)

            prediction = _call_with_retries(
                model=model,
                question=item.question,
                image=rendered[viz_type],
                metadata={"task": item.task},
            )
            last_call_ts = time.time()

            if prediction == "[ERROR]":
                errors += 1

            row = _metrics_row(item=item, viz_type=viz_type, prediction=prediction)
            results.append(row)
            done_keys.add(key)
            eval_rows += 1

        if idx % 50 == 0:
            write_jsonl(output_path, results)
            print(
                f"[INFO] {model.name}: checkpoint at item {idx}/{len(sampled_items)} ({len(results)} rows)",
            )

    write_jsonl(output_path, results)
    print(
        f"[INFO] {model.name}: done, wrote {len(results)} rows to {output_path} (new_rows={eval_rows}, errors={errors})",
    )
    return results


def main() -> None:
    args = parse_args()
    sampled_items = sample_items(
        benchmark_path=args.benchmark,
        num_items=args.num_items,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_order = ["gemini", "gpt4o"] if args.model == "both" else [args.model]

    all_model_rows: dict[str, list[dict[str, Any]]] = {}
    for model_key in model_order:
        output_path = args.output_dir / str(MODEL_SPECS[model_key]["output"])
        rows = run_model_eval(
            model_key=model_key,
            sampled_items=sampled_items,
            output_path=output_path,
            resume=args.resume,
        )
        all_model_rows[model_key] = rows
        _print_summary(model_name=str(MODEL_SPECS[model_key]["name"]), rows=rows)

    if args.model == "both":
        print("=== Combined Analysis Complete ===")
        for model_key in ("gemini", "gpt4o"):
            rows = all_model_rows.get(model_key, [])
            errors = sum(1 for row in rows if str(row.get("prediction")) == "[ERROR]")
            print(
                f"{MODEL_SPECS[model_key]['name']}: total_rows={len(rows)} errors={errors}",
            )


if __name__ == "__main__":
    main()
