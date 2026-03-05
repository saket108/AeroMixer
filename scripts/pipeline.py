#!/usr/bin/env python3
"""Professional single-entry pipeline for AeroMixer.

Modes:
- train: prepare dataset and run training
- eval : prepare dataset and run evaluation
- run  : train then evaluate (recommended)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import build_tiled_yolo_dataset as tiler
import train_any_dataset as ds
import validate_dataset as vd


PRESET_TO_CONFIG = {
    "lite": "config_files/images/aeromixer_images_lite.yaml",
    "full": "config_files/images/aeromixer_images.yaml",
    "prod": "config_files/images/aeromixer_images_prod.yaml",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _work_dir(root: Path) -> Path:
    p = root / "output" / "_auto_data_prep"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _stringify_cmd(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _git_commit(root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(root), text=True
        )
        return out.strip()
    except Exception:
        return None


def _dataset_fingerprint(data_dir: Path, max_files: int = 50000) -> dict:
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


def _prod_guardrail_opts() -> list[str]:
    return [
        "DATA.INPUT_TYPE",
        "image",
        "DATA.NUM_FRAMES",
        "1",
        "DATA.SAMPLING_RATE",
        "1",
        "DATA.OPEN_VOCABULARY",
        "False",
        "MODEL.STM.PREDICT_SEVERITY",
        "False",
        "MODEL.STM.ATTN_TELEMETRY",
        "False",
        "MODEL.STM.ATTN_TELEMETRY_STAGEWISE",
        "False",
        "MODEL.STM.ATTN_TELEMETRY_COMPARE_NOMASK",
        "False",
        "TEST.METRIC",
        "image_ap",
    ]


def _build_common_opts(
    plan: ds.DatasetPlan,
    output_dir: str,
    num_workers: int,
    preset: str,
    disable_guardrails: bool,
) -> list[str]:
    class_count = max(1, int(plan.num_classes))
    opts = [
        "OUTPUT_DIR",
        output_dir,
        "DATA.PATH_TO_DATA_DIR",
        str(plan.data_dir),
        "DATA.FRAME_DIR",
        str(plan.frame_dir),
        "DATA.ANNOTATION_FORMAT",
        str(plan.annotation_format),
        "DATALOADER.NUM_WORKERS",
        str(num_workers),
        "MODEL.STM.ACTION_CLASSES",
        str(class_count),
        "MODEL.STM.OBJECT_CLASSES",
        str(class_count),
        "MODEL.STM.NUM_ACT",
        str(class_count),
        "MODEL.STM.NUM_CLS",
        str(class_count),
    ]
    if preset == "prod" and not disable_guardrails:
        opts.extend(_prod_guardrail_opts())
    return opts


def _build_train_cmd(
    python_exe: str,
    config_file: str,
    output_dir: str,
    plan: ds.DatasetPlan,
    preset: str,
    epochs: int,
    batch_size: int,
    num_workers: int,
    disable_guardrails: bool,
    skip_val_in_train: bool,
    extra_opts: list[str],
) -> list[str]:
    cmd = [python_exe, "train_net.py", "--config-file", config_file]
    if skip_val_in_train:
        cmd.append("--skip-val-in-train")
    opts = _build_common_opts(plan, output_dir, num_workers, preset, disable_guardrails)
    opts.extend(
        [
            "SOLVER.MAX_EPOCH",
            str(epochs),
            "SOLVER.IMAGES_PER_BATCH",
            str(batch_size),
        ]
    )
    if extra_opts:
        opts.extend(extra_opts)
    cmd.extend(opts)
    return cmd


def _resolve_eval_weight(output_dir: Path, explicit_weight: str | None) -> str:
    if explicit_weight:
        return explicit_weight
    # test_net.py resolves MODEL.WEIGHT relative to OUTPUT_DIR when non-absolute.
    return "checkpoints/model_final.pth"


def _build_eval_cmd(
    python_exe: str,
    config_file: str,
    output_dir: str,
    plan: ds.DatasetPlan,
    preset: str,
    num_workers: int,
    disable_guardrails: bool,
    model_weight: str | None,
    extra_opts: list[str],
) -> list[str]:
    cmd = [python_exe, "test_net.py", "--config-file", config_file]
    opts = _build_common_opts(plan, output_dir, num_workers, preset, disable_guardrails)
    opts.extend(["MODEL.WEIGHT", _resolve_eval_weight(Path(output_dir), model_weight)])
    if extra_opts:
        opts.extend(extra_opts)
    cmd.extend(opts)
    return cmd


def _run_command(cmd: list[str], cwd: Path, dry_run: bool) -> int:
    print(">>", _stringify_cmd(cmd))
    if dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    return int(proc.returncode)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AeroMixer one-command pipeline.")
    p.add_argument("--mode", choices=["train", "eval", "run"], default="run")
    p.add_argument(
        "--data", required=True, help="Dataset path: zip/folder/data.yaml/json"
    )
    p.add_argument("--preset", choices=["lite", "full", "prod"], default="lite")
    p.add_argument(
        "--config-file", default=None, help="Optional override for preset config."
    )
    p.add_argument("--output-dir", default="output/pipeline_run")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument(
        "--split-ratio", default=None, help="Only for flat YOLO datasets. e.g. 80,10,10"
    )
    p.add_argument(
        "--tile-size", type=int, default=0, help="Enable YOLO tiling when > 0."
    )
    p.add_argument(
        "--tile-overlap",
        type=float,
        default=0.2,
        help="Tile overlap ratio in [0, 1).",
    )
    p.add_argument(
        "--tile-min-cover",
        type=float,
        default=0.35,
        help="Min GT box coverage ratio to keep in a tile.",
    )
    p.add_argument(
        "--tile-splits",
        default="train,val,test",
        help="Comma-separated splits to tile (train,val,test). Non-selected splits are copied.",
    )
    p.add_argument(
        "--include-empty-tiles",
        action="store_true",
        help="Keep tiles with no labels.",
    )
    p.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip dataset quality validation.",
    )
    p.add_argument(
        "--allow-validation-errors",
        action="store_true",
        help="Continue even if validation fails (not recommended).",
    )
    p.add_argument("--skip-val-in-train", action="store_true")
    p.add_argument(
        "--disable-guardrails",
        action="store_true",
        help="Disable preset guardrails (advanced). In prod preset, guardrails are enabled by default.",
    )
    p.add_argument(
        "--model-weight", default=None, help="Optional weight for eval mode."
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--extra-opts",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra config key/value pairs for train/eval.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    root = _repo_root()
    work = _work_dir(root)
    config_file = args.config_file or PRESET_TO_CONFIG[args.preset]
    output_dir = str(Path(args.output_dir))

    source_path = ds._resolve_source_path(Path(args.data).expanduser().resolve(), work)
    ratio = ds._parse_split_ratio(args.split_ratio) if args.split_ratio else None
    plan = ds._build_plan(source_path, ratio, args.seed, work)
    tiling_report_path = Path(output_dir) / "tiling_report.json"
    tiling_report = None

    if int(args.tile_size) > 0:
        if str(plan.annotation_format).lower() != "yolo":
            print("Tiling currently supports YOLO datasets only.")
            print(f"Resolved annotation_format='{plan.annotation_format}'")
            return 2

        tile_splits = tiler._parse_splits(args.tile_splits)
        tile_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(output_dir).name) or "run"
        tiled_data_dir = (
            work
            / "tiled_datasets"
            / f"{tile_tag}_ts{int(args.tile_size)}_ov{str(args.tile_overlap).replace('.', 'p')}"
        )

        if args.dry_run:
            print("Tiling requested: dry-run mode skips tile materialization.")
            print(f"Planned tiled dataset dir: {tiled_data_dir}")
            tiling_report = {
                "planned_only": True,
                "source_data_dir": str(plan.data_dir),
                "output_data_dir": str(tiled_data_dir),
                "tile_size": int(args.tile_size),
                "overlap": float(args.tile_overlap),
                "min_cover": float(args.tile_min_cover),
                "tile_splits": tile_splits,
                "include_empty_tiles": bool(args.include_empty_tiles),
            }
        else:
            tiling_report = tiler.build_tiled_dataset(
                data_dir=Path(plan.data_dir),
                frame_dir=str(plan.frame_dir),
                out_dir=tiled_data_dir,
                tile_size=int(args.tile_size),
                overlap=float(args.tile_overlap),
                min_cover=float(args.tile_min_cover),
                tile_splits=tile_splits,
                include_empty_tiles=bool(args.include_empty_tiles),
            )
            plan = ds.DatasetPlan(
                data_dir=Path(tiling_report["output_data_dir"]),
                annotation_format="yolo",
                frame_dir="",
                num_classes=int(plan.num_classes),
            )
            _write_json(tiling_report_path, tiling_report)
            print(f"Tiling report: {tiling_report_path}")

    manifest_path = Path(output_dir) / "pipeline_manifest.json"
    validation_report_path = Path(output_dir) / "dataset_validation.json"
    plan_dict = asdict(plan)
    plan_dict["data_dir"] = str(plan_dict["data_dir"])

    manifest = {
        "created_at_utc": _now_iso(),
        "git_commit": _git_commit(root),
        "mode": args.mode,
        "preset": args.preset,
        "config_file": config_file,
        "dataset_source": str(source_path),
        "dataset_plan": plan_dict,
        "dataset_fingerprint": _dataset_fingerprint(Path(plan.data_dir)),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "split_ratio": args.split_ratio,
        "tiling": {
            "enabled": bool(int(args.tile_size) > 0),
            "tile_size": int(args.tile_size),
            "tile_overlap": float(args.tile_overlap),
            "tile_min_cover": float(args.tile_min_cover),
            "tile_splits": args.tile_splits,
            "include_empty_tiles": bool(args.include_empty_tiles),
            "report_path": (
                str(tiling_report_path)
                if (int(args.tile_size) > 0 and not args.dry_run)
                else None
            ),
        },
        "skip_validation": bool(args.skip_validation),
        "allow_validation_errors": bool(args.allow_validation_errors),
        "skip_val_in_train": bool(args.skip_val_in_train),
        "guardrails_enabled": bool(
            args.preset == "prod" and not args.disable_guardrails
        ),
        "dry_run": bool(args.dry_run),
        "extra_opts": list(args.extra_opts),
    }

    train_cmd = _build_train_cmd(
        python_exe=sys.executable,
        config_file=config_file,
        output_dir=output_dir,
        plan=plan,
        preset=args.preset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        disable_guardrails=args.disable_guardrails,
        skip_val_in_train=args.skip_val_in_train,
        extra_opts=list(args.extra_opts),
    )
    eval_cmd = _build_eval_cmd(
        python_exe=sys.executable,
        config_file=config_file,
        output_dir=output_dir,
        plan=plan,
        preset=args.preset,
        num_workers=args.num_workers,
        disable_guardrails=args.disable_guardrails,
        model_weight=args.model_weight,
        extra_opts=list(args.extra_opts),
    )

    manifest["commands"] = {
        "train": _stringify_cmd(train_cmd),
        "eval": _stringify_cmd(eval_cmd),
    }
    if tiling_report is not None:
        manifest["tiling_report"] = tiling_report

    validation = None
    if not args.skip_validation:
        validation = vd.validate_dataset_plan(plan)
        validation_payload = dict(validation)
        validation_payload["dataset_source"] = str(source_path)
        validation_payload["resolved_plan"] = plan_dict
        _write_json(validation_report_path, validation_payload)

        print("Dataset validation:")
        print(
            f"  ok={validation_payload['ok']} "
            f"errors={len(validation_payload['errors'])} "
            f"warnings={len(validation_payload['warnings'])}"
        )
        if validation_payload["errors"]:
            for err in validation_payload["errors"][:5]:
                print(f"  ERROR: {err}")
        if validation_payload["warnings"]:
            for warn in validation_payload["warnings"][:5]:
                print(f"  WARN : {warn}")

    manifest["validation"] = {
        "enabled": not bool(args.skip_validation),
        "report_path": (
            str(validation_report_path) if not args.skip_validation else None
        ),
        "ok": (None if args.skip_validation else bool(validation["ok"])),
        "error_count": (None if args.skip_validation else len(validation["errors"])),
        "warning_count": (
            None if args.skip_validation else len(validation["warnings"])
        ),
    }

    _write_json(manifest_path, manifest)
    print("Pipeline manifest:", manifest_path)
    print("Resolved dataset:")
    print(f"  data_dir          : {plan.data_dir}")
    print(f"  annotation_format : {plan.annotation_format}")
    print(f"  frame_dir         : {plan.frame_dir!r}")
    print(f"  num_classes       : {plan.num_classes}")

    if (
        validation is not None
        and (not validation["ok"])
        and (not args.allow_validation_errors)
    ):
        print("Dataset validation failed. Aborting pipeline.")
        print("Use --allow-validation-errors to continue anyway.")
        return 2

    if args.mode in {"train", "run"}:
        rc = _run_command(train_cmd, cwd=root, dry_run=args.dry_run)
        if rc != 0:
            return rc

    if args.mode in {"eval", "run"}:
        rc = _run_command(eval_cmd, cwd=root, dry_run=args.dry_run)
        if rc != 0:
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
