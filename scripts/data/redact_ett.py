"""Remove ETT-derived values from a StructViz-Bench release tree.

The ETT time series (zhouhaoyi/ETDataset) are licensed CC BY-ND 4.0, which does not allow
sharing adapted material. Benchmark items built from ETT windows therefore ship without their
series and answer keys; ``scripts/data/rebuild_ett.py --restore <root>`` regenerates them from
the original ETT CSVs and writes them back byte-for-byte.

What is redacted, for every question whose benchmark row has ``source == "ett"`` and an
``ett_*`` data_id (840 questions; 30 rows labelled ``ett`` are NetworkX degree sequences
with ``nx_*`` data_ids, are not ETT-derived and are left untouched):

* ``benchmark/realworld_test.jsonl``: ``data`` and ``answer`` -> null, ``"ett_redacted": true``.
* every other ``*.jsonl``: ``answer`` / ``ground_truth`` / ``gold`` -> null on rows whose
  ``question_id`` is an ETT question. Model predictions, raw responses and stored scores
  (exact_match, f1, ...) are kept, except:
* model outputs that copy ETT values (``MODEL_OUTPUT_FILES``), on ETT rows only:
  ``results/v2/rawtext_qwen.jsonl`` (raw-text baseline; the model was given the series as text):
  ``prediction`` and ``raw_response`` -> null, ``"prediction_redacted": true``;
  ``results/review/tta_qwen.jsonl`` (transcribe-then-answer): ``transcript`` and
  ``raw_response`` -> null; ``results/review/cot_*_qwen*.jsonl`` (chain of thought):
  ``raw_response`` and the text fields derived from it -> null; both with
  ``"model_output_redacted": true``. Scores are kept. These model outputs are NOT restorable
  (our code does not derive them from ETT); the manifest keeps their sha256 only, and
  ``rebuild_ett.py --restore`` brings those files back to the original minus these fields
  (checked against ``sha256_restore_target``).
* ``*.csv`` with a ``question_id`` column, or an ``item_id`` column resolvable through a sibling
  ``items.jsonl``: ``ground_truth`` / ``answer`` cells -> empty.
* PNG renderings of ETT questions (human-eval package images listed in ``items.jsonl``, and any
  ``<question_id>_<viz>.png``) are deleted.

``data/ett_redaction_manifest.json`` records, without any values, the sha256 of each redacted
question's canonical (data, answer), the sha256 of every touched file before and after
redaction, and the sha256 of every deleted image. Running the script twice is a no-op.

Usage:
    python scripts/data/redact_ett.py <release_root>

Only the standard library is needed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

BENCHMARK_REL = "benchmark/realworld_test.jsonl"
MANIFEST_REL = "data/ett_redaction_manifest.json"
ANSWER_FIELDS = ("answer", "ground_truth", "gold")
CSV_ANSWER_FIELDS = ("ground_truth", "answer", "gold")
SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules"}
REDACTED_FLAG = "ett_redacted"
# Model outputs that copy ETT values (the model saw the series as text, or transcribed / reasoned
# over a rendering that prints them). They are nulled on ETT rows and cannot be rebuilt: they are
# not derived from ETT by our code. Scores (exact_match*, f1, ...) and `prediction` are kept,
# except in the raw-text baseline, whose final prediction is itself a copied value.
PREDICTION_FLAG = "prediction_redacted"
OUTPUT_FLAG = "model_output_redacted"
_COT_TEXT = ("raw_response", "cot_final_answer_text", "prediction_extract_answer",
             "prediction_legacy", "prediction_legacy_trunc256")
MODEL_OUTPUT_FILES: dict[str, tuple[tuple[str, ...], str]] = {
    "results/v2/rawtext_qwen.jsonl": (("prediction", "raw_response"), PREDICTION_FLAG),
    "results/review/tta_qwen.jsonl": (("transcript", "raw_response"), OUTPUT_FLAG),
    "results/review/cot_v1_qwen.jsonl": (_COT_TEXT, OUTPUT_FLAG),
    "results/review/cot_v2_qwen.jsonl": (_COT_TEXT, OUTPUT_FLAG),
    "results/review/cot_v2_qwen_sysprompt_only.jsonl": (_COT_TEXT, OUTPUT_FLAG),
}

ETT_SOURCE = {
    "dataset": "ETDataset (Electricity Transformer Temperature), ETT-small",
    "repository": "https://github.com/zhouhaoyi/ETDataset",
    "commit": "1d16c8f4f943005d613b5bc962e9eeb06058cf07",
    "license": "CC BY-ND 4.0",
    "files_sha256": {
        "ETTh1.csv": "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066",
        "ETTh2.csv": "a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b",
        "ETTm1.csv": "6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e",
        "ETTm2.csv": "db973ca252c6410a30d0469b13d696cf919648d0f3fd588c60f03fdbdbadd1fd",
    },
}
HASH_SPEC = (
    "sha256(json.dumps({'answer': answer, 'data': data}, sort_keys=True, "
    "ensure_ascii=False, separators=(',', ':')).encode('utf-8'))"
)


# ---------------------------------------------------------------------------------------------
# helpers shared with rebuild_ett.py
# ---------------------------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def item_hash(data: Any, answer: Any) -> str:
    """Canonical hash of one ETT question's (data, answer)."""
    payload = json.dumps(
        {"answer": answer, "data": data},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return sha256_bytes(payload.encode("utf-8"))


def is_ett_row(row: dict[str, Any]) -> bool:
    return row.get("source") == "ett" and str(row.get("data_id", "")).startswith("ett_")


def dump_like(obj: dict[str, Any], template_line: str) -> str:
    """Serialise ``obj`` the way the release's JSONL writers did (json.dumps defaults).

    Lines carrying literal non-ASCII characters were written with ensure_ascii=False.
    """
    ascii_only = all(ord(ch) < 128 for ch in template_line)
    return json.dumps(obj, ensure_ascii=ascii_only)


def split_lines(text: str) -> list[str]:
    """Split on "\\n" only, keeping terminators, so a file re-joins byte-for-byte.

    (str.splitlines would also split on U+2028 etc., which may occur inside JSON strings.)
    """
    parts = text.split("\n")
    lines = [part + "\n" for part in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines


def strip_eol(line: str) -> tuple[str, str]:
    body = line.rstrip("\r\n")
    return body, line[len(body):]


def iter_release_files(root: Path, suffix: str) -> list[Path]:
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            if name.endswith(suffix):
                out.append(Path(dirpath) / name)
    return out


def rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def read_csv_rows(raw: bytes) -> tuple[list[list[str]], str]:
    text = raw.decode("utf-8")
    eol = "\r\n" if "\r\n" in text else "\n"
    rows = list(csv.reader(io.StringIO(text, newline="")))
    return rows, eol


def write_csv_rows(rows: list[list[str]], eol: str) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator=eol)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def item_id_map(csv_path: Path) -> dict[str, str]:
    """item_id -> question_id from a sibling items.jsonl (human-eval packages)."""
    items = csv_path.parent / "items.jsonl"
    mapping: dict[str, str] = {}
    if items.exists():
        for line in items.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if "item_id" in row and "question_id" in row:
                    mapping[str(row["item_id"])] = str(row["question_id"])
    return mapping


_PNG_QID = re.compile(r"^(?P<qid>[a-z]+_\d{6}::difficulty=[^_]+)_(?P<viz>[a-z_]+)\.png$")


def suite_from_path(path_rel: str) -> str:
    if "render_v2" in path_rel:
        return "v2"
    if "render_assist" in path_rel:
        return "assist"
    return "v1"


# ---------------------------------------------------------------------------------------------
# redaction
# ---------------------------------------------------------------------------------------------


class Redactor:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.manifest_path = root / MANIFEST_REL
        self.old = (
            json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if self.manifest_path.exists()
            else None
        )
        self.items: dict[str, dict[str, Any]] = dict((self.old or {}).get("items", {}))
        self.files: dict[str, dict[str, Any]] = {
            f["path"]: f for f in (self.old or {}).get("files", [])
        }
        self.images: dict[str, dict[str, Any]] = {
            f["path"]: f for f in (self.old or {}).get("images", [])
        }
        self.answers: dict[str, Any] = {}  # qid -> answer, only while unredacted
        self.ett_qids: set[str] = set()
        self.log: list[str] = []
        self.changed = 0

    # -- benchmark -----------------------------------------------------------------------------

    def redact_benchmark(self) -> None:
        path = self.root / BENCHMARK_REL
        raw = path.read_bytes()
        lines = split_lines(raw.decode("utf-8"))
        out: list[str] = []
        n_new = 0
        n_ett = 0
        for line in lines:
            body, eol = strip_eol(line)
            if not body.strip():
                out.append(line)
                continue
            row = json.loads(body)
            if not is_ett_row(row):
                out.append(line)
                continue
            qid = str(row["question_id"])
            n_ett += 1
            self.ett_qids.add(qid)
            if row.get(REDACTED_FLAG):
                if qid not in self.items:
                    raise SystemExit(f"{qid} is redacted but missing from {MANIFEST_REL}")
                out.append(line)
                continue
            if dump_like(row, body) != body:
                raise SystemExit(f"{BENCHMARK_REL}: row {qid} does not re-serialise exactly")
            digest = item_hash(row["data"], row["answer"])
            if qid in self.items and self.items[qid]["sha256"] != digest:
                raise SystemExit(f"{qid}: values differ from the hash already in the manifest")
            self.items[qid] = {"sha256": digest, "data_id": row["data_id"]}
            self.answers[qid] = row["answer"]
            row["data"] = None
            row["answer"] = None
            row[REDACTED_FLAG] = True
            out.append(dump_like(row, body) + eol)
            n_new += 1
        self._commit_text(path, raw, "".join(out), "benchmark", ["data", "answer"], n_new)
        self.log.append(f"{BENCHMARK_REL}: {n_ett} ETT rows ({n_new} newly redacted)")

    # -- other jsonl ---------------------------------------------------------------------------

    def redact_jsonl(self, path: Path) -> None:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        if not any(q in text for q in self._probe_qids()):
            return
        path_rel = rel(self.root, path)
        out_fields, out_flag = MODEL_OUTPUT_FILES.get(path_rel, ((), PREDICTION_FLAG))
        out: list[str] = []
        target: list[str] = []  # what --restore can reach: answers back, model outputs not
        outputs: list[dict[str, Any]] = []
        n_new = 0
        fields_hit: set[str] = set()
        for line_no, line in enumerate(split_lines(text)):
            body, eol = strip_eol(line)
            if not body.strip():
                out.append(line)
                target.append(line)
                continue
            row = json.loads(body)
            qid = row.get("question_id") if isinstance(row, dict) else None
            if qid not in self.ett_qids or row.get("source") not in (None, "ett"):
                out.append(line)
                target.append(line)
                continue
            present = [k for k in ANSWER_FIELDS if k in row and row[k] is not None]
            redact_outputs = bool(out_fields) and not row.get(out_flag)
            if not present and not redact_outputs:
                out.append(line)
                target.append(line)
                continue
            if dump_like(row, body) != body:
                raise SystemExit(f"{path_rel}: row {qid} does not re-serialise exactly")
            if redact_outputs:
                if not present:
                    raise SystemExit(
                        f"{path_rel}: answers already redacted but model outputs not; run "
                        "scripts/data/rebuild_ett.py --restore first, then redact again"
                    )
                outputs.append({
                    "line": line_no,
                    "question_id": qid,
                    "sha256": sha256_bytes(json.dumps(
                        {k: row[k] for k in out_fields if k in row}, sort_keys=True,
                        ensure_ascii=False, separators=(",", ":"),
                    ).encode("utf-8")),
                })
                for key in out_fields:
                    if key in row:
                        row[key] = None
                row[out_flag] = True
            target.append(dump_like(row, body) + eol)
            for key in present:
                if qid in self.answers and str(row[key]) != str(self.answers[qid]):
                    raise SystemExit(
                        f"{rel(self.root, path)}: {key} of {qid} differs from the benchmark "
                        "answer; a rebuilt item could not restore it"
                    )
                if qid in self.answers and type(row[key]) is not type(self.answers[qid]):
                    raise SystemExit(f"{rel(self.root, path)}: {key} of {qid} has another type")
                row[key] = None
                fields_hit.add(key)
            out.append(dump_like(row, body) + eol)
            n_new += 1
        self._commit_text(path, raw, "".join(out), "jsonl", sorted(fields_hit), n_new)
        if outputs:
            entry = self.files[path_rel]
            entry["unrestorable_fields"] = list(out_fields)
            entry["unrestorable_flag"] = out_flag
            entry["unrestorable_rows"] = outputs
            entry["sha256_restore_target"] = sha256_bytes("".join(target).encode("utf-8"))
            self.log.append(f"{path_rel}: {len(outputs)} rows: {', '.join(out_fields)} nulled "
                            "(model outputs copying ETT values; not restorable)")

    # -- csv -----------------------------------------------------------------------------------

    def redact_csv(self, path: Path) -> None:
        raw = path.read_bytes()
        rows, eol = read_csv_rows(raw)
        if not rows:
            return
        header = rows[0]
        if "question_id" in header:
            key_col = header.index("question_id")
            to_qid: dict[str, str] | None = None
        elif "item_id" in header:
            key_col = header.index("item_id")
            to_qid = item_id_map(path)
        else:
            return
        ans_cols = [header.index(c) for c in CSV_ANSWER_FIELDS if c in header]
        if not ans_cols:
            return
        if write_csv_rows(rows, eol) != raw:
            raise SystemExit(f"{rel(self.root, path)}: CSV does not re-serialise exactly")
        n_new = 0
        for row in rows[1:]:
            if len(row) <= key_col:
                continue
            key = row[key_col]
            qid = to_qid.get(key, "") if to_qid is not None else key
            if qid not in self.ett_qids:
                continue
            hit = False
            for col in ans_cols:
                if col < len(row) and row[col] != "":
                    if qid in self.answers and row[col] != str(self.answers[qid]):
                        raise SystemExit(f"{rel(self.root, path)}: {qid} key differs from benchmark")
                    row[col] = ""
                    hit = True
            n_new += int(hit)
        fields = [header[c] for c in ans_cols]
        self._commit_bytes(path, raw, write_csv_rows(rows, eol), "csv", fields, n_new)

    # -- images --------------------------------------------------------------------------------

    def redact_images(self) -> None:
        targets: dict[str, dict[str, Any]] = {}
        for items in iter_release_files(self.root, "items.jsonl"):
            for line in items.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("question_id") in self.ett_qids and row.get("image"):
                    img = items.parent / str(row["image"])
                    targets[rel(self.root, img)] = {
                        "question_id": row["question_id"],
                        "viz_type": row.get("viz_type"),
                        "suite": row.get("suite", "v1"),
                        "listed_in": rel(self.root, items),
                    }
        for png in iter_release_files(self.root, ".png"):
            match = _PNG_QID.match(png.name)
            if match and match.group("qid") in self.ett_qids:
                path_rel = rel(self.root, png)
                targets.setdefault(path_rel, {
                    "question_id": match.group("qid"),
                    "viz_type": match.group("viz"),
                    "suite": suite_from_path(path_rel),
                    "listed_in": None,
                })
        for path_rel, info in sorted(targets.items()):
            path = self.root / path_rel
            if path.exists():
                digest = sha256_file(path)
                known = self.images.get(path_rel)
                if known and known["sha256"] != digest:
                    raise SystemExit(f"{path_rel}: differs from the image hash in the manifest")
                self.images[path_rel] = {"path": path_rel, "sha256": digest, **info}
                path.unlink()
                self.changed += 1
                self.log.append(f"deleted {path_rel}")
            elif path_rel not in self.images:
                self.log.append(f"WARNING: {path_rel} is listed but absent and not in manifest")

    # -- bookkeeping ---------------------------------------------------------------------------

    def _probe_qids(self) -> list[str]:
        return sorted(self.ett_qids)

    def _commit_text(
        self, path: Path, raw: bytes, text: str, kind: str, fields: list[str], n: int
    ) -> None:
        self._commit_bytes(path, raw, text.encode("utf-8"), kind, fields, n)

    def _commit_bytes(
        self, path: Path, raw: bytes, new: bytes, kind: str, fields: list[str], n: int
    ) -> None:
        path_rel = rel(self.root, path)
        if new == raw:
            return
        known = self.files.get(path_rel)
        if known and sha256_bytes(raw) == known["sha256_original"]:
            known = {**known, "rows_redacted": 0}
        entry = {
            "path": path_rel,
            "kind": kind,
            "fields": sorted(set(fields) | set(known["fields"] if known else [])),
            "rows_redacted": n + (known["rows_redacted"] if known else 0),
            "sha256_original": known["sha256_original"] if known else sha256_bytes(raw),
            "sha256_redacted": sha256_bytes(new),
        }
        self.files[path_rel] = entry
        path.write_bytes(new)
        self.changed += 1
        self.log.append(f"{path_rel}: {n} rows redacted ({', '.join(entry['fields'])})")

    def write_manifest(self) -> None:
        manifest = {
            "description": (
                "ETT-derived StructViz-Bench items removed for licensing (CC BY-ND 4.0). "
                "Hashes only; rebuild with: python scripts/data/rebuild_ett.py --restore ."
            ),
            "ett_source": ETT_SOURCE,
            "generator": {
                "config": "configs/generation.yaml",
                "entry_point": "scripts/generate_benchmark.py::_build_realworld_timeseries_items",
                "question_id_offset": "len(tabular synthetic items) + len(timeseries synthetic "
                "items) from the same config and seed",
            },
            "item_hash": HASH_SPEC,
            "not_redacted": (
                "30 rows labelled source='ett' with data_id nx_*_degree are NetworkX "
                "degree sequences (karate club, Les Miserables, Florentine families), not ETT data."
            ),
            "items": dict(sorted(self.items.items())),
            "files": [self.files[k] for k in sorted(self.files)],
            "images": [self.images[k] for k in sorted(self.images)],
        }
        text = json.dumps(manifest, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.manifest_path.exists() or self.manifest_path.read_text("utf-8") != text:
            self.manifest_path.write_text(text, encoding="utf-8")
            self.changed += 1
            self.log.append(f"wrote {MANIFEST_REL}")

    def run(self) -> None:
        self.redact_benchmark()
        for path in iter_release_files(self.root, ".jsonl"):
            if rel(self.root, path) != BENCHMARK_REL:
                self.redact_jsonl(path)
        for path in iter_release_files(self.root, ".csv"):
            self.redact_csv(path)
        self.redact_images()
        self.check_json_files()
        self.write_manifest()

    def check_json_files(self) -> None:
        """Refuse to finish if a plain .json file carries ETT answers (not handled here)."""
        for path in iter_release_files(self.root, ".json"):
            if rel(self.root, path) == MANIFEST_REL:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            hits = [q for q in self.ett_qids if q in text]
            if not hits:
                continue
            obj = json.loads(text)
            stack = [obj]
            while stack:
                cur = stack.pop()
                if isinstance(cur, dict):
                    if cur.get("question_id") in self.ett_qids and any(
                        cur.get(k) is not None for k in ANSWER_FIELDS + ("data",)
                    ):
                        raise SystemExit(f"{rel(self.root, path)} carries ETT values; extend redact_ett.py")
                    stack.extend(cur.values())
                elif isinstance(cur, list):
                    stack.extend(cur)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="Release tree root (contains benchmark/).")
    parser.add_argument(
        "--allow-private-tree",
        action="store_true",
        help="Run even if the tree contains data/external/ett (the private working copy).",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root / BENCHMARK_REL).exists():
        print(f"error: {root / BENCHMARK_REL} not found", file=sys.stderr)
        return 2
    if (root / "data/external/ett").exists() and not args.allow_private_tree:
        print("error: this looks like the private working tree (data/external/ett exists); "
              "redact a release copy instead, or pass --allow-private-tree", file=sys.stderr)
        return 2
    redactor = Redactor(root)
    redactor.run()
    for line in redactor.log:
        print(line)
    print(f"ETT questions: {len(redactor.ett_qids)}; manifest items: {len(redactor.items)}; "
          f"files touched: {len(redactor.files)}; images removed: {len(redactor.images)}")
    if not redactor.changed:
        print("nothing to do (already redacted)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
