"""Rebuild the ETT-derived StructViz-Bench items from the original ETT CSVs.

The released benchmark ships its ETT questions without series and answer keys (see
``scripts/data/redact_ett.py``; ETT is CC BY-ND 4.0). This script regenerates them with the
benchmark's own generator code and config (``configs/generation.yaml``, seed 42,
``scripts/generate_benchmark.py::_build_realworld_timeseries_items``) and, with ``--restore``,
writes them back into every file the redaction touched.

Not restorable: model outputs that copy ETT values were redacted on ETT rows too (see
``redact_ett.MODEL_OUTPUT_FILES``): the raw-text baseline's ``prediction`` / ``raw_response``
(``"prediction_redacted": true``), the transcribe-then-answer ``transcript`` / ``raw_response``
and the chain-of-thought ``raw_response`` and derived text fields (``"model_output_redacted":
true``). They are model outputs, not derived from ETT by our code, so they cannot be rebuilt and
stay null; their scores are kept. Those files are restored to the original minus these fields
and checked against the manifest's ``sha256_restore_target``; every other file is checked
against its original sha256.

The ETT items depend on nothing but the four CSVs and the config: the window sampler uses its
own ``random.Random(seed + offset)`` and the question generator is deterministic. The only
coupling to the rest of the benchmark is the question-id offset (number of synthetic tabular
plus time-series items), which is recomputed in memory with the same generator code; nothing
else is written.

Examples:
    # restore a redacted release in place (downloads ETT at the pinned commit)
    python scripts/data/rebuild_ett.py --download --restore .

    # use a local ETT-small directory instead
    python scripts/data/rebuild_ett.py --data-dir /path/to/ETT-small --restore .

    # determinism check against an unredacted benchmark file
    python scripts/data/rebuild_ett.py --check benchmark/realworld_test.jsonl

    # just write the ETT rows
    python scripts/data/rebuild_ett.py --output ett_items.jsonl
"""

from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false

import argparse
import io
import json
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import redact_ett as R  # noqa: E402

RAW_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/{commit}/ETT-small/{name}"
DEFAULT_DATA_DIR = Path("data/external/ett/repo/ETT-small")
DEFAULT_DOWNLOAD_DIR = Path("data/external/ett/ETT-small")


# ---------------------------------------------------------------------------------------------
# ETT source files
# ---------------------------------------------------------------------------------------------


def download_ett(dest: Path) -> Path:
    """Fetch the four ETT-small CSVs at the pinned commit and check their sha256."""
    dest.mkdir(parents=True, exist_ok=True)
    commit = R.ETT_SOURCE["commit"]
    for name, digest in R.ETT_SOURCE["files_sha256"].items():
        target = dest / name
        if target.exists() and R.sha256_file(target) == digest:
            print(f"  have {target}")
            continue
        url = RAW_URL.format(commit=commit, name=name)
        print(f"  fetching {url}")
        with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 - fixed URL
            data = response.read()
        if R.sha256_bytes(data) != digest:
            raise SystemExit(f"{name}: downloaded file does not match the pinned sha256")
        target.write_bytes(data)
    return dest


def check_ett_dir(data_dir: Path, allow_unpinned: bool) -> None:
    for name, digest in R.ETT_SOURCE["files_sha256"].items():
        path = data_dir / name
        if not path.exists():
            raise SystemExit(f"missing {path}; pass --data-dir or --download")
        if R.sha256_file(path) != digest:
            msg = f"{path}: sha256 differs from ETDataset@{R.ETT_SOURCE['commit'][:7]}"
            if not allow_unpinned:
                raise SystemExit(msg + " (pass --allow-unpinned to continue anyway)")
            print("WARNING: " + msg)
    extra = sorted(p.name for p in data_dir.glob("ETT*.csv") if p.name not in
                   R.ETT_SOURCE["files_sha256"])
    if extra:
        raise SystemExit(f"{data_dir} has extra ETT*.csv files the loader would pick up: {extra}")


# ---------------------------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------------------------


def generate_ett_rows(data_dir: Path, config: Path) -> list[dict[str, Any]]:
    """Regenerate the ETT benchmark rows exactly as scripts/generate_benchmark.py does."""
    import yaml

    import generate_benchmark as gb
    from src.utils.io_utils import _safe_dumps

    with config.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["real_world"]["timeseries"]["data_dir"] = str(data_dir)
    seed = int(cfg["benchmark"]["seed"])
    # question-id offset = len(tabular) + len(timeseries) synthetic items (in memory only)
    offset = len(gb._build_tabular_items(cfg, seed)) + len(gb._build_timeseries_items(cfg, seed))
    items = gb._build_realworld_timeseries_items(cfg, start_idx=offset)
    rows = [json.loads(_safe_dumps(item.to_dict())) for item in items]
    return [row for row in rows if R.is_ett_row(row)]


def canon(row: dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, ensure_ascii=False)


# ---------------------------------------------------------------------------------------------
# restore
# ---------------------------------------------------------------------------------------------


class Restorer:
    def __init__(self, root: Path, rows: list[dict[str, Any]]) -> None:
        self.root = root
        manifest_path = root / R.MANIFEST_REL
        if not manifest_path.exists():
            raise SystemExit(f"{manifest_path} not found: nothing to restore")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.rows = {str(r["question_id"]): r for r in rows}
        self.failures: list[str] = []
        self.partial: list[str] = []

    def verify_items(self) -> None:
        items = self.manifest["items"]
        missing = sorted(set(items) - set(self.rows))
        if missing:
            raise SystemExit(f"{len(missing)} manifest items were not regenerated, e.g. {missing[:3]}")
        bad = [q for q, info in items.items()
               if R.item_hash(self.rows[q]["data"], self.rows[q]["answer"]) != info["sha256"]]
        if bad:
            raise SystemExit(f"{len(bad)} regenerated items fail the manifest hash, e.g. {bad[:3]}")
        print(f"  {len(items)} regenerated items match their manifest hashes")

    def _restore_jsonl(self, path: Path, kind: str, fields: list[str]) -> bytes:
        out: list[str] = []
        for line in R.split_lines(path.read_bytes().decode("utf-8")):
            body, eol = R.strip_eol(line)
            if not body.strip():
                out.append(line)
                continue
            row = json.loads(body)
            qid = row.get("question_id") if isinstance(row, dict) else None
            if qid not in self.manifest["items"]:
                out.append(line)
                continue
            gen = self.rows[qid]
            if kind == "benchmark":
                if not row.pop(R.REDACTED_FLAG, False):
                    out.append(line)
                    continue
                for key, value in row.items():  # every kept field must agree with the rebuild
                    if key not in ("data", "answer") and gen.get(key) != value:
                        raise SystemExit(f"{qid}: field {key!r} disagrees with the rebuilt item")
                row["data"] = gen["data"]
                row["answer"] = gen["answer"]
            else:
                if row.get("source") not in (None, "ett"):
                    out.append(line)
                    continue
                for key in fields:
                    if key in row and row[key] is None:
                        row[key] = gen["answer"]
            out.append(R.dump_like(row, body) + eol)
        return "".join(out).encode("utf-8")

    def _restore_csv(self, path: Path, fields: list[str]) -> bytes:
        rows, eol = R.read_csv_rows(path.read_bytes())
        header = rows[0]
        if "question_id" in header:
            key_col, to_qid = header.index("question_id"), None
        else:
            key_col, to_qid = header.index("item_id"), R.item_id_map(path)
        cols = [header.index(f) for f in fields if f in header]
        for row in rows[1:]:
            if len(row) <= key_col:
                continue
            qid = to_qid.get(row[key_col], "") if to_qid is not None else row[key_col]
            if qid in self.manifest["items"]:
                for col in cols:
                    if row[col] == "":
                        row[col] = str(self.rows[qid]["answer"])
        return R.write_csv_rows(rows, eol)

    def restore_files(self) -> None:
        for entry in self.manifest["files"]:
            path = self.root / entry["path"]
            current = R.sha256_file(path)
            target = entry.get("sha256_restore_target", entry["sha256_original"])
            if current in (target, entry["sha256_original"]):
                print(f"  already restored {entry['path']}")
                if current != entry["sha256_original"]:
                    self.partial.append(entry["path"])
                continue
            if current != entry["sha256_redacted"]:
                print(f"  WARNING {entry['path']} changed since redaction; restoring anyway")
            if entry["kind"] == "csv":
                new = self._restore_csv(path, entry["fields"])
            else:
                new = self._restore_jsonl(path, entry["kind"], entry["fields"])
            ok = R.sha256_bytes(new) == target
            path.write_bytes(new)
            note = ""
            if "unrestorable_fields" in entry:
                note = (f"; {len(entry['unrestorable_rows'])} rows keep "
                        f"{'/'.join(entry['unrestorable_fields'])} redacted (model outputs, "
                        "not rebuildable)")
                self.partial.append(entry["path"])
            print(f"  {'OK  ' if ok else 'FAIL'} {entry['path']} ({entry['rows_redacted']} rows{note})")
            if not ok:
                self.failures.append(entry["path"])

    def restore_images(self) -> None:
        if not self.manifest["images"]:
            return
        import matplotlib

        matplotlib.use("Agg")
        from src.rendering.render_pipeline import RenderPipeline
        from src.utils.io_utils import BenchmarkItem

        pipelines: dict[str, Any] = {}
        for entry in self.manifest["images"]:
            path = self.root / entry["path"]
            if path.exists() and R.sha256_file(path) == entry["sha256"]:
                print(f"  already present {entry['path']}")
                continue
            suite = entry.get("suite") or "v1"
            pipeline = pipelines.setdefault(suite, RenderPipeline(suite=suite))
            item = BenchmarkItem.from_dict(self.rows[entry["question_id"]])
            images = pipeline.render_all({"modality": item.modality, "data": item.data})
            buffer = io.BytesIO()
            images[entry["viz_type"]].save(buffer, format="PNG")
            data = buffer.getvalue()
            ok = R.sha256_bytes(data) == entry["sha256"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            print(f"  {'OK  ' if ok else 'FAIL'} {entry['path']} (re-rendered, suite {suite})")
            if not ok:
                self.failures.append(entry["path"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=None,
                        help=f"Directory with ETTh1/ETTh2/ETTm1/ETTm2.csv (default "
                             f"{DEFAULT_DATA_DIR}, or {DEFAULT_DOWNLOAD_DIR} with --download).")
    parser.add_argument("--download", action="store_true",
                        help="Fetch the CSVs from github.com/zhouhaoyi/ETDataset at commit "
                             f"{R.ETT_SOURCE['commit'][:7]} (sha256-checked).")
    parser.add_argument("--allow-unpinned", action="store_true",
                        help="Accept CSVs whose sha256 differs from the pinned ones.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "generation.yaml")
    parser.add_argument("--restore", type=Path, default=None, metavar="RELEASE_ROOT",
                        help="Write the rebuilt values back into a redacted release tree.")
    parser.add_argument("--check", type=Path, default=None, metavar="BENCHMARK_JSONL",
                        help="Compare the rebuilt rows with the ETT rows of an unredacted file.")
    parser.add_argument("--output", type=Path, default=None, help="Write the rebuilt rows here.")
    args = parser.parse_args()

    if args.download:
        data_dir = download_ett(args.data_dir or REPO_ROOT / DEFAULT_DOWNLOAD_DIR)
    else:
        data_dir = args.data_dir or REPO_ROOT / DEFAULT_DATA_DIR
    check_ett_dir(data_dir, args.allow_unpinned)

    # The loader globs ETT*.csv in data_dir; stage the four pinned files on their own.
    with tempfile.TemporaryDirectory() as tmp:
        for name in R.ETT_SOURCE["files_sha256"]:
            shutil.copyfile(data_dir / name, Path(tmp) / name)
        rows = generate_ett_rows(Path(tmp), args.config)
    print(f"rebuilt {len(rows)} ETT items from {data_dir}")

    if args.output:
        with args.output.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {args.output}")

    status = 0
    if args.check:
        lines = args.check.read_text(encoding="utf-8").split("\n")
        ref = [json.loads(line) for line in lines if line.strip()]
        ref_ett = {r["question_id"]: r for r in ref if R.is_ett_row(r)}
        gen = {r["question_id"]: r for r in rows}
        same = sum(1 for q, r in ref_ett.items() if q in gen and canon(gen[q]) == canon(r))
        labelled = sum(1 for r in ref if r.get("source") == "ett")
        print(f"check: {same}/{len(ref_ett)} ETT rows identical (canonical JSON, sort_keys); "
              f"{len(gen)} rebuilt; {labelled - len(ref_ett)} rows labelled 'ett' are not "
              "ETT-derived (nx_* data_id) and are not rebuilt")
        status |= int(same != len(ref_ett) or len(gen) != len(ref_ett))

    if args.restore:
        root = args.restore.resolve()
        print(f"restoring {root}")
        restorer = Restorer(root, rows)
        restorer.verify_items()
        restorer.restore_files()
        restorer.restore_images()
        if restorer.failures:
            print(f"RESTORE FAILED for {len(restorer.failures)} files: {restorer.failures}")
            status |= 1
        else:
            print("restore complete: every touched file and image matches its original sha256"
                  + (f", except {', '.join(restorer.partial)}, which matches its restore target "
                     "(ETT-copying model outputs stay redacted)" if restorer.partial else ""))
    return status


if __name__ == "__main__":
    sys.exit(main())
