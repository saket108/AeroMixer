#!/usr/bin/env python3
"""
Run controlled IoF-tau ablations for STM attention on a fixed image subset.

Experiments:
1) baseline learned tau
2) tau zero
3) tau clamp sweep (e.g., 0.5, 1.0, 2.0)

Outputs:
- per-run training summary JSON (produced by trainer)
- per-run evaluation log with mAP + small/medium/large AP
- aggregate CSV/JSON summary for quick comparison
"""

import argparse
import ast
import csv
import json
import os
import random
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import cv2

try:
    import yaml
except ImportError:
    yaml = None


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [
            line.rstrip("\n")
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]


def _group_by_image(lines: List[str]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        image_rel = parts[0]
        grouped.setdefault(image_rel, []).append(line)
    return grouped


def _sample_grouped_lines(
    grouped: Dict[str, List[str]], ratio: float, seed: int
) -> Tuple[List[str], int]:
    image_keys = sorted(grouped.keys())
    if not image_keys:
        return [], 0
    k = max(1, int(round(len(image_keys) * ratio)))
    rng = random.Random(seed)
    selected = set(rng.sample(image_keys, min(k, len(image_keys))))
    out: List[str] = []
    for key in image_keys:
        if key in selected:
            out.extend(grouped[key])
    return out, len(selected)


def _load_yaml(path: str) -> Dict:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read config. Install `pyyaml`.")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _prepare_subset(
    source_data_dir: str,
    subset_dir: str,
    subset_ratio: float,
    seed: int,
) -> Dict[str, int]:
    os.makedirs(subset_dir, exist_ok=True)
    stats = {}
    for split in ["train", "test"]:
        src = os.path.join(source_data_dir, f"{split}.txt")
        if not os.path.exists(src):
            raise FileNotFoundError(f"Missing split file: {src}")
        lines = _read_lines(src)
        grouped = _group_by_image(lines)
        sampled_lines, sampled_images = _sample_grouped_lines(
            grouped, subset_ratio, seed + (0 if split == "train" else 1)
        )
        dst = os.path.join(subset_dir, f"{split}.txt")
        with open(dst, "w", encoding="utf-8") as f:
            for line in sampled_lines:
                f.write(line + "\n")
        stats[f"{split}_images"] = sampled_images
        stats[f"{split}_anns"] = len(sampled_lines)
    return stats


def _split_aliases(split: str) -> List[str]:
    split_name = str(split).lower()
    if split_name == "train":
        return ["train", "training", "stratified_train"]
    if split_name in ["test", "val", "valid", "validation"]:
        return ["test", "val", "valid", "validation", "stratified_val"]
    return [split_name]


def _find_existing_split_dir(
    base_dir: str, aliases: List[str]
) -> Tuple[Optional[str], Optional[str]]:
    for alias in aliases:
        path = os.path.join(base_dir, alias)
        if os.path.isdir(path):
            return path, alias
    return None, None


def _normalize_relpath(path: str) -> str:
    return path.replace("\\", "/")


def _resolve_yolo_label_path(
    source_data_dir: str, image_split_dir: str, used_split_alias: str, image_path: str
) -> str:
    rel_path_in_split = os.path.relpath(image_path, image_split_dir)
    stem_rel = os.path.splitext(_normalize_relpath(rel_path_in_split))[0]
    parts = [p for p in stem_rel.split("/") if p]
    base_name = os.path.basename(stem_rel) + ".txt"

    candidate_rel_paths = [stem_rel + ".txt", base_name]
    if len(parts) > 1 and parts[0].lower() == "images":
        no_images_rel = "/".join(parts[1:]) + ".txt"
        candidate_rel_paths.append(no_images_rel)
        candidate_rel_paths.append("labels/" + no_images_rel)
        candidate_rel_paths.append("labels/" + os.path.basename(no_images_rel))

    candidate_roots = [
        os.path.join(source_data_dir, "labels", used_split_alias),
        os.path.join(source_data_dir, used_split_alias, "labels"),
        os.path.join(source_data_dir, "labels"),
        os.path.join(source_data_dir, used_split_alias),
        os.path.dirname(image_split_dir),
        source_data_dir,
    ]
    for root in candidate_roots:
        if not os.path.isdir(root):
            continue
        for rel_path in candidate_rel_paths:
            trial = os.path.join(root, rel_path.replace("/", os.sep))
            if os.path.exists(trial):
                return trial
    return os.path.join(source_data_dir, "labels", used_split_alias, base_name)


def _load_yolo_grouped_lines(source_data_dir: str, split: str) -> Dict[str, List[str]]:
    aliases = _split_aliases(split)
    image_split_dir = None
    used_split_alias = None
    images_root = None
    for alias in aliases:
        root_style_dir = os.path.join(source_data_dir, "images", alias)
        split_style_dir = os.path.join(source_data_dir, alias, "images")
        flat_style_dir = os.path.join(source_data_dir, alias)
        if os.path.isdir(root_style_dir):
            images_root = os.path.join(source_data_dir, "images")
            image_split_dir = root_style_dir
            used_split_alias = alias
            break
        if os.path.isdir(split_style_dir):
            images_root = source_data_dir
            image_split_dir = split_style_dir
            used_split_alias = alias
            break
        if os.path.isdir(flat_style_dir):
            images_root = source_data_dir
            image_split_dir = flat_style_dir
            used_split_alias = alias
            break

    if image_split_dir is None or used_split_alias is None or images_root is None:
        raise FileNotFoundError(
            f"YOLO image split for '{split}' not found under {source_data_dir}. Tried aliases: {aliases}."
        )

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    grouped: Dict[str, List[str]] = {}
    for root, _, files in os.walk(image_split_dir):
        for file_name in sorted(files):
            if os.path.splitext(file_name)[1].lower() not in exts:
                continue
            image_path = os.path.join(root, file_name)
            label_path = _resolve_yolo_label_path(
                source_data_dir, image_split_dir, used_split_alias, image_path
            )
            if not os.path.exists(label_path):
                continue

            img = cv2.imread(image_path)
            if img is None:
                continue
            height, width = img.shape[:2]
            if height <= 0 or width <= 0:
                continue

            txt_lines: List[str] = []
            img_rel = _normalize_relpath(os.path.relpath(image_path, images_root))
            with open(label_path, "r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        raise ValueError(
                            f"Bad YOLO label at {label_path}:{line_number}"
                        )
                    cls_id = int(float(parts[0]))
                    x_center, y_center, box_w, box_h = map(float, parts[1:5])
                    x1 = (x_center - box_w / 2.0) * width
                    y1 = (y_center - box_h / 2.0) * height
                    x2 = (x_center + box_w / 2.0) * width
                    y2 = (y_center + box_h / 2.0) * height

                    x1 = max(0.0, min(float(width - 1), x1))
                    y1 = max(0.0, min(float(height - 1), y1))
                    x2 = max(0.0, min(float(width - 1), x2))
                    y2 = max(0.0, min(float(height - 1), y2))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    txt_lines.append(
                        f"{img_rel} {x1:.4f} {y1:.4f} {x2:.4f} {y2:.4f} {cls_id}"
                    )

            if txt_lines:
                grouped[img_rel] = txt_lines
    return grouped


def _detect_source_annotation_format(
    source_data_dir: str, configured_format: str
) -> str:
    fmt = (configured_format or "").strip().lower()
    if fmt and fmt != "auto":
        if fmt in {"txt", "yolo"}:
            return fmt
        raise ValueError(
            f"run_iof_tau_ablation currently supports txt/yolo sources, got DATA.ANNOTATION_FORMAT='{fmt}'."
        )

    if os.path.exists(os.path.join(source_data_dir, "train.txt")):
        return "txt"
    if os.path.isdir(os.path.join(source_data_dir, "labels")) and os.path.isdir(
        os.path.join(source_data_dir, "images")
    ):
        return "yolo"
    for alias in _split_aliases("train"):
        split_images = os.path.join(source_data_dir, alias, "images")
        split_labels = os.path.join(source_data_dir, alias, "labels")
        if os.path.isdir(split_images) and os.path.isdir(split_labels):
            return "yolo"
    raise ValueError(
        "Could not detect source annotation format for ablation subset creation. "
        "Expected txt split files or YOLO images/labels folders."
    )


def _prepare_subset_from_yolo(
    source_data_dir: str,
    subset_dir: str,
    subset_ratio: float,
    seed: int,
) -> Dict[str, int]:
    os.makedirs(subset_dir, exist_ok=True)
    stats = {}
    grouped_train = _load_yolo_grouped_lines(source_data_dir, "train")
    sampled_train_lines, sampled_train_images = _sample_grouped_lines(
        grouped_train, subset_ratio, seed
    )
    train_dst = os.path.join(subset_dir, "train.txt")
    with open(train_dst, "w", encoding="utf-8") as f:
        for line in sampled_train_lines:
            f.write(line + "\n")
    stats["train_images"] = sampled_train_images
    stats["train_anns"] = len(sampled_train_lines)

    grouped_eval = _load_yolo_grouped_lines(source_data_dir, "test")
    sampled_eval_lines, sampled_eval_images = _sample_grouped_lines(
        grouped_eval, subset_ratio, seed + 1
    )
    test_dst = os.path.join(subset_dir, "test.txt")
    with open(test_dst, "w", encoding="utf-8") as f:
        for line in sampled_eval_lines:
            f.write(line + "\n")
    stats["test_images"] = sampled_eval_images
    stats["test_anns"] = len(sampled_eval_lines)
    return stats


def _parse_eval_log(result_log_path: str) -> Dict:
    if not os.path.exists(result_log_path):
        return {}
    text = open(result_log_path, "r", encoding="utf-8").read()
    start = text.find("{")
    if start < 0:
        return {}
    try:
        return ast.literal_eval(text[start:])
    except Exception:
        return {}


def _extract_main_map(eval_metrics: Dict) -> float:
    for key, val in eval_metrics.items():
        k = str(key)
        if (
            k.startswith("PascalBoxes_Precision/mAP@")
            and "PerformanceByCategory" not in k
        ):
            try:
                return float(val)
            except Exception:
                return float("nan")
    return float("nan")


def _extract_small_ap(eval_metrics: Dict) -> float:
    for key, val in eval_metrics.items():
        if str(key).startswith("SmallObject/AP@"):
            try:
                return float(val)
            except Exception:
                return float("nan")
    return float("nan")


def _run_cmd(cmd: List[str], dry_run: bool = False) -> int:
    print(" ".join(cmd))
    if dry_run:
        return 0
    proc = subprocess.run(cmd)
    return int(proc.returncode)


def _write_summary(output_root: str, rows: List[Dict]) -> None:
    os.makedirs(output_root, exist_ok=True)
    json_path = os.path.join(output_root, "ablation_summary.json")
    csv_path = os.path.join(output_root, "ablation_summary.csv")

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


def main():
    parser = argparse.ArgumentParser(description="Run IoF tau ablation experiments.")
    parser.add_argument(
        "--config-file", default="config_files/presets/full.yaml", type=str
    )
    parser.add_argument("--output-root", default="outputs/iof_tau_ablation", type=str)
    parser.add_argument("--subset-ratio", default=0.05, type=float)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--seed", default=2, type=int)
    parser.add_argument("--clamp-values", default="0.5,1.0,2.0", type=str)
    parser.add_argument("--python", default=sys.executable, type=str)
    parser.add_argument(
        "--dataset-name",
        default="",
        type=str,
        help="Override DATA.DATASETS[0] if needed.",
    )
    parser.add_argument(
        "--skip-runs",
        action="store_true",
        help="Do not launch training, only parse existing outputs.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing."
    )
    args = parser.parse_args()

    cfg_dict = _load_yaml(args.config_file)
    data_cfg = cfg_dict.get("DATA", {})
    source_data_dir = data_cfg.get("PATH_TO_DATA_DIR", "")
    frame_dir = data_cfg.get("FRAME_DIR", "")
    configured_format = data_cfg.get("ANNOTATION_FORMAT", "auto")
    datasets = data_cfg.get("DATASETS", ["aircraft"])
    dataset_name = args.dataset_name or (
        datasets[0] if isinstance(datasets, list) and datasets else "aircraft"
    )

    if not source_data_dir:
        raise ValueError(
            "DATA.PATH_TO_DATA_DIR must be set in config or passed via config override."
        )

    source_frame_dir = (
        os.path.join(source_data_dir, frame_dir) if frame_dir else source_data_dir
    )
    source_format = _detect_source_annotation_format(source_data_dir, configured_format)
    subset_frame_dir = source_frame_dir
    print(f"Detected source annotation format: {source_format}")
    subset_dir = os.path.join(
        args.output_root, f"subset_{int(args.subset_ratio * 100)}p_seed{args.seed}"
    )

    if not args.skip_runs:
        if os.path.exists(subset_dir):
            shutil.rmtree(subset_dir)
        if source_format == "txt":
            subset_stats = _prepare_subset(
                source_data_dir, subset_dir, args.subset_ratio, args.seed
            )
        elif source_format == "yolo":
            subset_stats = _prepare_subset_from_yolo(
                source_data_dir, subset_dir, args.subset_ratio, args.seed
            )
            if os.path.isdir(os.path.join(source_data_dir, "images")):
                subset_frame_dir = os.path.join(source_data_dir, "images")
            else:
                subset_frame_dir = source_data_dir
        else:
            raise ValueError(f"Unsupported ablation source format: {source_format}")
        print("Subset prepared:", subset_stats)

    clamp_values = [v.strip() for v in args.clamp_values.split(",") if v.strip()]
    experiments = [
        ("baseline", {"MODEL.STM.IOF_TAU_MODE": "learned"}),
        ("tau_zero", {"MODEL.STM.IOF_TAU_MODE": "zero"}),
    ]
    for clamp_val in clamp_values:
        experiments.append(
            (
                f"tau_clamp_{clamp_val}",
                {
                    "MODEL.STM.IOF_TAU_MODE": "clamp",
                    "MODEL.STM.IOF_TAU_CLAMP_MIN": "0.0",
                    "MODEL.STM.IOF_TAU_CLAMP_MAX": clamp_val,
                },
            )
        )

    rows: List[Dict] = []
    for exp_name, tau_opts in experiments:
        exp_output = os.path.join(args.output_root, exp_name)
        os.makedirs(exp_output, exist_ok=True)

        base_opts = {
            "OUTPUT_DIR": exp_output,
            "DATA.PATH_TO_DATA_DIR": subset_dir,
            "DATA.FRAME_DIR": subset_frame_dir,
            "DATA.ANNOTATION_FORMAT": "txt",
            "SOLVER.MAX_EPOCH": str(args.epochs),
            "MODEL.STM.ATTN_TELEMETRY": "True",
            "MODEL.STM.ATTN_TELEMETRY_STAGEWISE": "True",
            "MODEL.STM.ATTN_TELEMETRY_COMPARE_NOMASK": "False",
        }
        base_opts.update(tau_opts)

        opts_kv: List[str] = []
        for k, v in base_opts.items():
            opts_kv.extend([k, str(v)])

        cmd = [
            args.python,
            "train_net.py",
            "--config-file",
            args.config_file,
            "--seed",
            str(args.seed),
            "--skip-val-in-train",
        ] + opts_kv

        if not args.skip_runs:
            code = _run_cmd(cmd, dry_run=args.dry_run)
            if code != 0:
                print(f"[WARN] experiment {exp_name} failed with exit code {code}")

        train_summary_path = os.path.join(
            exp_output, "inference", "train_metrics_final.json"
        )
        eval_log_path = os.path.join(
            exp_output, "inference", dataset_name, "result_image.log"
        )
        train_summary = {}
        if os.path.exists(train_summary_path):
            with open(train_summary_path, "r", encoding="utf-8") as f:
                train_summary = json.load(f)
        eval_metrics = _parse_eval_log(eval_log_path)

        row = {
            "experiment": exp_name,
            "seed": args.seed,
            "epochs": args.epochs,
            "subset_ratio": args.subset_ratio,
            "iof_tau_mode": tau_opts.get("MODEL.STM.IOF_TAU_MODE", "learned"),
            "iof_tau_clamp_max": tau_opts.get("MODEL.STM.IOF_TAU_CLAMP_MAX", ""),
            "attn_entropy_avg": train_summary.get("attn_entropy_avg", float("nan")),
            "attn_diag_avg": train_summary.get("attn_diag_avg", float("nan")),
            "attn_tau_mean_avg": train_summary.get("attn_tau_mean_avg", float("nan")),
            "refine_l1_avg": train_summary.get("refine_l1_avg", float("nan")),
            "mAP@0.5": _extract_main_map(eval_metrics),
            "SmallObject/AP@0.5": _extract_small_ap(eval_metrics),
            "output_dir": exp_output,
        }
        rows.append(row)

    _write_summary(args.output_root, rows)
    print(f"Saved summary: {os.path.join(args.output_root, 'ablation_summary.csv')}")


if __name__ == "__main__":
    main()
