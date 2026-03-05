#!/usr/bin/env python3
"""Stable inference/evaluation entrypoint with JSON summary output."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import train_any_dataset as ds
import validate_dataset as vd


PRESET_TO_CONFIG = {
    "lite": "config_files/images/aeromixer_images_lite.yaml",
    "full": "config_files/images/aeromixer_images.yaml",
    "prod": "config_files/images/aeromixer_images_prod.yaml",
}

MAP50_KEYS = [
    "PascalBoxes_Precision/mAP@0.5IOU",
    "mAP@0.5",
    "map50",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _work_dir(root: Path) -> Path:
    p = root / "output" / "_auto_data_prep"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _stringify_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


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


def _parse_result_log(path: Path) -> dict[str, float | None]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    kv = re.findall(r"'([^']+)':\s*([^,\n}]+)", text)
    out: dict[str, float | None] = {}
    for key, raw_value in kv:
        value = raw_value.strip()
        if value.startswith("np.float64(") and value.endswith(")"):
            value = value[len("np.float64(") : -1].strip()
        if value.lower() in {"nan", "none"}:
            out[key] = None
            continue
        try:
            out[key] = float(value)
        except Exception:
            continue
    return out


def _primary_map50(metrics: dict[str, float | None]) -> float | None:
    for key in MAP50_KEYS:
        if key in metrics:
            return metrics[key]
    return None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AeroMixer stable inference pipeline.")
    p.add_argument(
        "--data", required=True, help="Dataset path: zip/folder/data.yaml/json"
    )
    p.add_argument("--preset", choices=["lite", "full", "prod"], default="prod")
    p.add_argument(
        "--config-file", default=None, help="Optional override for preset config."
    )
    p.add_argument("--output-dir", default="output/inference_run")
    p.add_argument("--model-weight", default="checkpoints/model_final.pth")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument(
        "--split-ratio", default=None, help="Only for flat YOLO datasets. e.g. 80,10,10"
    )
    p.add_argument("--skip-validation", action="store_true")
    p.add_argument("--allow-validation-errors", action="store_true")
    p.add_argument("--disable-guardrails", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--extra-opts", nargs=argparse.REMAINDER, default=[])
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

    validation = None
    validation_path = Path(output_dir) / "dataset_validation.json"
    if not args.skip_validation:
        validation = vd.validate_dataset_plan(plan)
        payload = dict(validation)
        payload["dataset_source"] = str(source_path)
        payload["resolved_plan"] = asdict(plan)
        payload["resolved_plan"]["data_dir"] = str(plan.data_dir)
        _write_json(validation_path, payload)
        if (not payload["ok"]) and (not args.allow_validation_errors):
            print("Dataset validation failed. Aborting inference.")
            return 2

    cmd = [sys.executable, "test_net.py", "--config-file", config_file]
    opts = _build_common_opts(
        plan=plan,
        output_dir=output_dir,
        num_workers=args.num_workers,
        preset=args.preset,
        disable_guardrails=args.disable_guardrails,
    )
    opts.extend(["MODEL.WEIGHT", str(args.model_weight)])
    if args.extra_opts:
        opts.extend(args.extra_opts)
    cmd.extend(opts)

    manifest = {
        "created_at_utc": _now_iso(),
        "dataset_source": str(source_path),
        "preset": args.preset,
        "config_file": config_file,
        "resolved_plan": {
            "data_dir": str(plan.data_dir),
            "annotation_format": plan.annotation_format,
            "frame_dir": plan.frame_dir,
            "num_classes": plan.num_classes,
        },
        "model_weight": str(args.model_weight),
        "command": _stringify_cmd(cmd),
        "validation": {
            "enabled": not bool(args.skip_validation),
            "report_path": str(validation_path) if not args.skip_validation else None,
            "ok": None if args.skip_validation else bool(validation["ok"]),
        },
    }
    manifest_path = Path(output_dir) / "inference_manifest.json"
    _write_json(manifest_path, manifest)
    print(f"Inference manifest: {manifest_path}")
    print(">>", manifest["command"])

    if args.dry_run:
        return 0

    rc = subprocess.run(cmd, cwd=str(root), check=False).returncode
    if rc != 0:
        return int(rc)

    logs = sorted(Path(output_dir).glob("inference/*/result_image.log"))
    datasets: dict[str, Any] = {}
    for log in logs:
        dataset_name = log.parent.name
        metrics = _parse_result_log(log)
        datasets[dataset_name] = {
            "log_file": str(log),
            "map50": _primary_map50(metrics),
            "metrics": metrics,
        }

    summary = {
        "created_at_utc": _now_iso(),
        "output_dir": output_dir,
        "datasets": datasets,
    }
    summary_path = Path(output_dir) / "inference_summary.json"
    _write_json(summary_path, summary)
    print(f"Inference summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
