#!/usr/bin/env python3
"""
Run unified baseline benchmarks (AeroMixer / YOLO / DETR) and write one CSV/JSON summary.

This script is command-driven and model-agnostic:
- you provide a command for each baseline
- optionally provide a metrics file path for each baseline
- script executes, times, parses common mAP keys, and writes unified outputs
"""

import argparse
import csv
import json
import math
import os
import subprocess
import time
from typing import Dict, List, Optional, Tuple


MAP50_KEYS = [
    "mAP@0.5",
    "map50",
    "map_50",
    "metrics/mAP50(B)",
    "PascalBoxes_Precision/mAP@0.5IOU",
]

MAP5095_KEYS = [
    "mAP@0.5:0.95",
    "map",
    "map50_95",
    "map_50_95",
    "metrics/mAP50-95(B)",
]


def _parse_args():
    parser = argparse.ArgumentParser(description="Run baseline benchmark commands and aggregate results.")
    parser.add_argument("--output-root", default="outputs/baseline_benchmarks", type=str)
    parser.add_argument("--tag", default="", type=str, help="Optional experiment tag.")
    parser.add_argument("--seed", default=2, type=int)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")

    parser.add_argument("--aeromixer-cmd", default="", type=str, help="Command string to run AeroMixer baseline.")
    parser.add_argument("--aeromixer-metrics", default="", type=str, help="JSON/CSV metrics file path for AeroMixer.")

    parser.add_argument("--yolo-cmd", default="", type=str, help="Command string to run YOLO baseline.")
    parser.add_argument("--yolo-metrics", default="", type=str, help="JSON/CSV metrics file path for YOLO.")

    parser.add_argument("--detr-cmd", default="", type=str, help="Command string to run DETR baseline.")
    parser.add_argument("--detr-metrics", default="", type=str, help="JSON/CSV metrics file path for DETR.")
    return parser.parse_args()


def _safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _find_metric_value(data: Dict, candidates: List[str]) -> float:
    for key in candidates:
        if key in data:
            return _safe_float(data[key])
    normalized = {str(k).lower(): v for k, v in data.items()}
    for key in candidates:
        low_key = key.lower()
        if low_key in normalized:
            return _safe_float(normalized[low_key])
    return float("nan")


def _parse_metrics_file(path: str) -> Tuple[float, float, str]:
    if not path:
        return float("nan"), float("nan"), "no metrics path provided"
    if not os.path.exists(path):
        return float("nan"), float("nan"), f"metrics file missing: {path}"

    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return float("nan"), float("nan"), "json metrics is not a dict"
            map50 = _find_metric_value(data, MAP50_KEYS)
            map5095 = _find_metric_value(data, MAP5095_KEYS)
            return map50, map5095, "ok"
        except Exception as exc:
            return float("nan"), float("nan"), f"json parse error: {exc}"

    if ext == ".csv":
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                return float("nan"), float("nan"), "csv has no rows"
            row = rows[-1]
            map50 = _find_metric_value(row, MAP50_KEYS)
            map5095 = _find_metric_value(row, MAP5095_KEYS)
            return map50, map5095, "ok"
        except Exception as exc:
            return float("nan"), float("nan"), f"csv parse error: {exc}"

    return float("nan"), float("nan"), f"unsupported metrics extension: {ext}"


def _run_command(cmd: str, dry_run: bool) -> int:
    print(cmd)
    if dry_run:
        return 0
    proc = subprocess.run(cmd, shell=True)
    return int(proc.returncode)


def _write_summary(output_root: str, rows: List[Dict]) -> None:
    os.makedirs(output_root, exist_ok=True)
    json_path = os.path.join(output_root, "benchmark_summary.json")
    csv_path = os.path.join(output_root, "benchmark_summary.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_tasks(args) -> List[Tuple[str, str, str]]:
    return [
        ("aeromixer", args.aeromixer_cmd.strip(), args.aeromixer_metrics.strip()),
        ("yolo", args.yolo_cmd.strip(), args.yolo_metrics.strip()),
        ("detr", args.detr_cmd.strip(), args.detr_metrics.strip()),
    ]


def main():
    args = _parse_args()
    rows: List[Dict] = []

    tasks = _build_tasks(args)
    for baseline_name, cmd, metrics_path in tasks:
        if not cmd:
            continue
        started = time.time()
        exit_code = _run_command(cmd, dry_run=args.dry_run)
        runtime_sec = time.time() - started

        map50, map5095, metrics_status = _parse_metrics_file(metrics_path)
        row = {
            "tag": args.tag,
            "seed": args.seed,
            "baseline": baseline_name,
            "exit_code": exit_code,
            "runtime_sec": round(runtime_sec, 3),
            "map50": None if math.isnan(map50) else round(map50, 6),
            "map50_95": None if math.isnan(map5095) else round(map5095, 6),
            "metrics_path": metrics_path,
            "metrics_status": metrics_status,
            "command": cmd,
        }
        rows.append(row)

    _write_summary(args.output_root, rows)
    print(f"Saved summary: {os.path.join(args.output_root, 'benchmark_summary.csv')}")


if __name__ == "__main__":
    main()
