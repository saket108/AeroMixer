#!/usr/bin/env python3
"""Run unified baseline benchmarks and append comparable rows.

Supports AeroMixer / YOLOv8 / DETR with one consistent output format.
Each baseline can be:
1) command + metrics file
2) metrics file only (if already executed elsewhere)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CANON_COLUMNS = [
    "date_utc",
    "git_commit",
    "model",
    "preset",
    "dataset",
    "split",
    "epochs",
    "batch_size",
    "image_scale",
    "map50",
    "map5095",
    "small_ap",
    "latency_ms_per_image",
    "notes",
]

MAP50_KEYS = [
    "PascalBoxes_Precision/mAP@0.5IOU",
    "mAP@0.5",
    "map50",
    "map_50",
    "metrics/mAP50(B)",
]
MAP5095_KEYS = [
    "PascalBoxes_Precision/mAP@0.5:0.95IOU",
    "mAP@0.5:0.95",
    "map",
    "map_50_95",
    "metrics/mAP50-95(B)",
]
SMALL_AP_KEYS = [
    "SmallObject/AP@0.5IOU",
    "Area/small/mAP@0.5IOU",
    "small_ap",
]


@dataclass
class BaselineSpec:
    name: str
    cmd: str
    metrics: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_commit(root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            text=True,
        )
        return out.strip()
    except Exception:
        return ""


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _from_dict_any(d: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        if key in d:
            v = d[key]
            if isinstance(v, str) and v.startswith("np.float64(") and v.endswith(")"):
                v = v[len("np.float64(") : -1]
            f = _safe_float(v)
            if f is not None:
                return f
    lowered = {str(k).lower(): v for k, v in d.items()}
    for key in keys:
        v = lowered.get(key.lower())
        if isinstance(v, str) and v.startswith("np.float64(") and v.endswith(")"):
            v = v[len("np.float64(") : -1]
        f = _safe_float(v)
        if f is not None:
            return f
    return None


def _parse_log_like(path: Path) -> dict[str, float | None]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    kv = re.findall(r"'([^']+)':\s*([^,\n}]+)", text)
    out: dict[str, float | None] = {}
    for key, raw in kv:
        value = raw.strip()
        if value.startswith("np.float64(") and value.endswith(")"):
            value = value[len("np.float64(") : -1].strip()
        out[key] = _safe_float(value)
    return out


def _parse_metrics(path: Path) -> tuple[float | None, float | None, float | None, str]:
    if not path.exists():
        return None, None, None, f"missing metrics file: {path}"

    ext = path.suffix.lower()
    payload: dict[str, Any] = {}
    try:
        if ext == ".json":
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                if isinstance(raw.get("metrics"), dict):
                    payload = dict(raw["metrics"])
                else:
                    payload = raw
            else:
                return None, None, None, "json payload is not object"
        elif ext == ".csv":
            with open(path, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                return None, None, None, "csv has no rows"
            payload = rows[-1]
        else:
            payload = _parse_log_like(path)
    except Exception as exc:
        return None, None, None, f"parse error: {exc}"

    map50 = _from_dict_any(payload, MAP50_KEYS)
    map5095 = _from_dict_any(payload, MAP5095_KEYS)
    small_ap = _from_dict_any(payload, SMALL_AP_KEYS)
    return map50, map5095, small_ap, "ok"


def _append_rows(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CANON_COLUMNS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CANON_COLUMNS})


def _run_cmd(cmd: str, cwd: Path, dry_run: bool) -> int:
    print(">>", cmd)
    if dry_run or not cmd.strip():
        return 0
    proc = subprocess.run(cmd, cwd=str(cwd), shell=True, check=False)
    return int(proc.returncode)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified baseline benchmark runner.")
    p.add_argument("--dataset", default="unknown_dataset")
    p.add_argument("--split", default="test")
    p.add_argument("--preset", default="benchmark")
    p.add_argument("--epochs", default="")
    p.add_argument("--batch-size", default="")
    p.add_argument("--image-scale", default="")
    p.add_argument("--summary-csv", default="benchmarks/summary.csv")
    p.add_argument("--output-json", default="output/benchmarks/baseline_run.json")
    p.add_argument("--dry-run", action="store_true")

    p.add_argument("--aeromixer-cmd", default="")
    p.add_argument("--aeromixer-metrics", default="")
    p.add_argument("--yolo-cmd", default="")
    p.add_argument("--yolo-metrics", default="")
    p.add_argument("--detr-cmd", default="")
    p.add_argument("--detr-metrics", default="")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    root = _repo_root()
    git_commit = _git_commit(root)
    timestamp = _now_iso()

    specs = [
        BaselineSpec("AeroMixer", args.aeromixer_cmd.strip(), args.aeromixer_metrics.strip()),
        BaselineSpec("YOLOv8", args.yolo_cmd.strip(), args.yolo_metrics.strip()),
        BaselineSpec("DETR", args.detr_cmd.strip(), args.detr_metrics.strip()),
    ]

    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for spec in specs:
        if not spec.cmd and not spec.metrics:
            continue

        started = time.time()
        rc = _run_cmd(spec.cmd, cwd=root, dry_run=args.dry_run)
        runtime_sec = time.time() - started

        map50, map5095, small_ap, metrics_status = (None, None, None, "no metrics path")
        if spec.metrics:
            map50, map5095, small_ap, metrics_status = _parse_metrics(Path(spec.metrics))

        row = {
            "date_utc": timestamp,
            "git_commit": git_commit,
            "model": spec.name,
            "preset": args.preset,
            "dataset": args.dataset,
            "split": args.split,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "image_scale": args.image_scale,
            "map50": map50,
            "map5095": map5095,
            "small_ap": small_ap,
            "latency_ms_per_image": "",
            "notes": (
                f"exit_code={rc};runtime_sec={runtime_sec:.3f};metrics_status={metrics_status};"
                f"cmd={shlex.quote(spec.cmd) if spec.cmd else ''};metrics={spec.metrics}"
            ),
        }
        rows.append(row)
        details.append(
            {
                "baseline": spec.name,
                "exit_code": rc,
                "runtime_sec": runtime_sec,
                "metrics_path": spec.metrics,
                "metrics_status": metrics_status,
                "row": row,
            }
        )

    summary_csv = Path(args.summary_csv)
    output_json = Path(args.output_json)
    if not args.dry_run and rows:
        _append_rows(summary_csv, rows)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(
                {
                    "created_at_utc": timestamp,
                    "git_commit": git_commit,
                    "dataset": args.dataset,
                    "split": args.split,
                    "rows": rows,
                    "details": details,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print(f"Rows prepared: {len(rows)}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Run JSON   : {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
