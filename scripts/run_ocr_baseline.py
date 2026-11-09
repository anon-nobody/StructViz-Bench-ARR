"""OCR-assisted baseline: perception vs reasoning decomposition.

Evaluates a two-stage pipeline:
  1. Extract text from rendered visualization (via OCR / vision model)
  2. Answer the question using only the extracted text (text-only LLM)

This isolates whether model failures stem from visual perception (reading the
chart) or from reasoning over the extracted content.

Usage:
    python scripts/run_ocr_baseline.py --model gpt4o --ocr-model gpt-4o
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

OCR_PROMPT = (
    "Extract ALL text, numbers, labels, and data values visible in this image. "
    "Return them in a structured format preserving the original layout as closely "
    "as possible. Do not interpret or summarize — just transcribe exactly what you see."
)

QA_PROMPT_TEMPLATE = (
    "Based on the following extracted data, answer the question concisely.\n\n"
    "Extracted data:\n{extracted_text}\n\n"
    "Question: {question}\n\n"
    "Answer (concise, exact value only):"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR-assisted baseline evaluation")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=PROJECT_ROOT / "benchmark" / "structviz_bench.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "ocr_baseline_gpt4o.jsonl",
    )
    parser.add_argument("--ocr-model", default="gpt-4o")
    parser.add_argument("--qa-model", default="gpt-4o")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def extract_text_from_image(image_path: str, model: str) -> str:
    """Stage 1: Use a vision model to OCR/transcribe the image."""
    try:
        import openai

        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        import base64

        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": OCR_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }
            ],
            max_tokens=2048,
            temperature=0.0,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        return f"[OCR_ERROR] {e}"


def answer_from_text(extracted_text: str, question: str, model: str) -> str:
    """Stage 2: Answer the question using only extracted text (no image)."""
    try:
        import openai

        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        prompt = QA_PROMPT_TEMPLATE.format(
            extracted_text=extracted_text, question=question
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.0,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        return f"[QA_ERROR] {e}"


def main() -> None:
    args = parse_args()
    print(f"OCR-assisted baseline: ocr={args.ocr_model}, qa={args.qa_model}")
    print(f"Output: {args.output}")
    print(f"Limit: {args.limit} items")
    print()
    print("NOTE: This script requires rendered images in benchmark/rendered/.")
    print("If images are not pre-rendered, run scripts/render_all.py first.")
    print()

    if not args.benchmark.exists():
        print(f"Benchmark file not found: {args.benchmark}")
        print("This script is a prepared skeleton for the OCR baseline experiment.")
        print("To run it:")
        print(
            "  1. Generate benchmark: python scripts/generate_benchmark.py --config configs/generation.yaml --output benchmark/structviz_bench.jsonl"
        )
        print(
            "  2. Render images: python scripts/render_all.py --config configs/generation.yaml"
        )
        print("  3. Set OPENAI_API_KEY environment variable")
        print("  4. Run: python scripts/run_ocr_baseline.py --limit 500")
        return

    with open(args.benchmark) as f:
        items = [json.loads(line) for line in f]

    import random

    random.seed(args.seed)
    if len(items) > args.limit:
        items = random.sample(items, args.limit)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    for i, item in enumerate(items):
        image_path = item.get("image_path", "")
        if not image_path or not Path(image_path).exists():
            continue

        extracted = extract_text_from_image(image_path, args.ocr_model)
        prediction = answer_from_text(extracted, item["question"], args.qa_model)

        result = {
            "question_id": item["question_id"],
            "question": item["question"],
            "answer": item.get("answer", ""),
            "prediction": prediction,
            "ocr_text": extracted[:500],
            "modality": item.get("modality", ""),
            "viz_type": item.get("viz_type", ""),
        }

        with open(args.output, "a") as out:
            out.write(json.dumps(result) + "\n")

        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(items)}] processed")
        time.sleep(0.5)

    print(f"Done. Results written to {args.output}")


if __name__ == "__main__":
    main()
