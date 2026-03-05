#!/usr/bin/env python3
"""Freeze dataset identity (fingerprint + validation) into a version manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import train_any_dataset as ds
import validate_dataset as vd


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dataset_fingerprint(data_dir: Path, max_files: int = 50000) -> dict[str, Any]:
    if not data_dir.exists():
        return {
            "exists": False,
            "num_files_scanned": 0,
            "total_bytes_scanned": 0,
            "truncated": False,
            "meta_sha256": None,
        }

    hasher = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    truncated = False
    errors = 0

    for dirpath, dirnames, filenames in os.walk(str(data_dir)):
        dirnames.sort()
        filenames.sort()
        for name in filenames:
            path = Path(dirpath) / name
            try:
                st = path.stat()
            except Exception:
                errors += 1
                continue
            rel = path.relative_to(data_dir).as_posix()
            hasher.update(rel.encode("utf-8", errors="ignore"))
            hasher.update(str(int(st.st_size)).encode("ascii", errors="ignore"))
            hasher.update(str(int(st.st_mtime_ns)).encode("ascii", errors="ignore"))
            file_count += 1
            total_bytes += int(st.st_size)
            if file_count >= max_files:
                truncated = True
                break
        if truncated:
            break

    return {
        "exists": True,
        "num_files_scanned": file_count,
        "total_bytes_scanned": total_bytes,
        "truncated": truncated,
        "max_files": max_files,
        "stat_errors": errors,
        "meta_sha256": hasher.hexdigest(),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create dataset version manifest for reproducible training."
    )
    p.add_argument(
        "--data", required=True, help="Dataset path: zip/folder/data.yaml/json"
    )
    p.add_argument("--seed", type=int, default=2)
    p.add_argument(
        "--split-ratio", default=None, help="Only for flat YOLO datasets. e.g. 80,10,10"
    )
    p.add_argument(
        "--out",
        default="output/dataset_version.json",
        help="Output dataset version manifest path",
    )
    p.add_argument("--allow-validation-errors", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]
    work = root / "output" / "_auto_data_prep"
    work.mkdir(parents=True, exist_ok=True)

    source_path = ds._resolve_source_path(Path(args.data).expanduser().resolve(), work)
    ratio = ds._parse_split_ratio(args.split_ratio) if args.split_ratio else None
    plan = ds._build_plan(source_path, ratio, args.seed, work)

    validation = vd.validate_dataset_plan(plan)
    if (not validation["ok"]) and (not args.allow_validation_errors):
        print("Dataset validation failed; not freezing version manifest.")
        for err in validation["errors"][:10]:
            print(f"ERROR: {err}")
        print("Use --allow-validation-errors to force manifest generation.")
        return 2

    payload = {
        "created_at_utc": _now_iso(),
        "dataset_source": str(source_path),
        "resolved_plan": {
            "data_dir": str(plan.data_dir),
            "annotation_format": plan.annotation_format,
            "frame_dir": plan.frame_dir,
            "num_classes": int(plan.num_classes),
        },
        "dataset_fingerprint": _dataset_fingerprint(Path(plan.data_dir)),
        "validation": validation,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Dataset version manifest written to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
