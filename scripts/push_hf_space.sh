#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  printf 'Usage: %s <space-repo>\n' "$0" >&2
  printf 'Example: %s ANON-ORG/structviz-bench-human-eval\n' "$0" >&2
  printf 'Set HF_TOKEN env var or use huggingface-cli login.\n' >&2
  exit 1
fi

SPACE_REPO="$1"
# Resolve repo root from this script's location, then point at the human-eval
# space subdirectory. Avoids hardcoding any absolute path.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SPACE_DIR="$REPO_ROOT/release/huggingface/human_eval_space"

if [ -z "${HF_TOKEN:-}" ]; then
  printf 'Error: HF_TOKEN environment variable is not set.\n' >&2
  printf 'Run: export HF_TOKEN="hf_xxx"\n' >&2
  exit 1
fi

REMOTE_URL="https://user:${HF_TOKEN}@huggingface.co/spaces/${SPACE_REPO}"

git -C "$SPACE_DIR" init
git -C "$SPACE_DIR" branch -M main

if git -C "$SPACE_DIR" remote get-url origin >/dev/null 2>&1; then
  git -C "$SPACE_DIR" remote set-url origin "$REMOTE_URL"
else
  git -C "$SPACE_DIR" remote add origin "$REMOTE_URL"
fi

git -C "$SPACE_DIR" add .

if git -C "$SPACE_DIR" diff --cached --quiet; then
  printf 'No staged changes to push.\n'
else
  git -C "$SPACE_DIR" commit -m "deploy human eval space"
fi

git -C "$SPACE_DIR" push -u origin main
