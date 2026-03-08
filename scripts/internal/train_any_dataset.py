#!/usr/bin/env python3
"""Internal dataset preparation/training helper for AeroMixer.

This helper accepts a dataset path (zip/folder/data.yaml/json), detects layout/format,
optionally creates train/val/test splits, infers class count, and launches train_net.py
with the required config overrides.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class DatasetPlan:
    data_dir: Path
    annotation_format: str
    frame_dir: str
    num_classes: int


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyYAML is required. Install with: pip install pyyaml"
        ) from e

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML dict in {path}, got {type(data).__name__}")
    return data


def _find_data_yaml(root: Path) -> Path | None:
    direct = root / "data.yaml"
    if direct.is_file():
        return direct
    matches = sorted(root.rglob("data.yaml"))
    return matches[0] if matches else None


def _extract_zip_to_dir(zip_path: Path, out_root: Path) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    target = out_root / zip_path.stem
    if target.exists():
        return target
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target)
    return target


def _resolve_source_path(source: Path, work_dir: Path) -> Path:
    if source.is_file() and source.suffix.lower() == ".zip":
        return _extract_zip_to_dir(source, work_dir / "prepared_datasets")
    if source.is_file():
        return source
    if source.is_dir():
        return source
    raise FileNotFoundError(f"Dataset source does not exist: {source}")


def _split_name_from_yaml_entry(entry: Any) -> str | None:
    if entry is None:
        return None
    raw = str(entry).strip()
    if not raw:
        return None
    p = Path(raw)
    if p.name.lower() == "images":
        p = p.parent
    return p.name if p.name else None


def _count_classes_from_data_yaml(data_yaml: dict[str, Any]) -> int | None:
    names = data_yaml.get("names")
    if isinstance(names, list):
        return len(names)
    if isinstance(names, dict):
        return len(names)
    nc = data_yaml.get("nc")
    if isinstance(nc, int) and nc > 0:
        return nc
    return None


def _iter_label_files(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*.txt") if p.is_file()])


def _count_classes_from_yolo_labels(labels_root: Path) -> int:
    max_id = -1
    for txt in _iter_label_files(labels_root):
        try:
            with open(txt, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls = int(float(parts[0]))
                    if cls > max_id:
                        max_id = cls
        except Exception:
            continue
    return max(1, max_id + 1)


def _is_split_layout_train_images(root: Path) -> bool:
    return (root / "train" / "images").is_dir() and (root / "train" / "labels").is_dir()


def _is_split_layout_images_train(root: Path) -> bool:
    return (root / "images" / "train").is_dir() and (root / "labels" / "train").is_dir()


def _is_flat_yolo_layout(root: Path) -> bool:
    return (
        (root / "images").is_dir()
        and (root / "labels").is_dir()
        and not _is_split_layout_images_train(root)
    )


def _infer_format_from_json(json_path: Path) -> str:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "images" in data and "annotations" in data:
        return "coco"
    if isinstance(data, dict) and isinstance(data.get("images"), list):
        # Nested custom JSON format used in AeroMixer dataset loader.
        first = data["images"][0] if data["images"] else {}
        if isinstance(first, dict) and "annotations" in first:
            return "custom_json"
    raise ValueError(f"Could not infer JSON annotation format from {json_path}")


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _create_split_from_flat_yolo(
    dataset_root: Path,
    split_ratio: tuple[int, int, int],
    seed: int,
    out_dir: Path,
) -> Path:
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(
            "Flat YOLO split creation expects <root>/images and <root>/labels."
        )

    images = sorted(
        [
            p
            for p in images_dir.rglob("*")
            if p.suffix.lower() in VALID_IMAGE_EXTS and p.is_file()
        ]
    )
    if not images:
        raise RuntimeError(f"No images found in {images_dir}")

    pairs: list[tuple[Path, Path]] = []
    for img in images:
        rel = img.relative_to(images_dir)
        label = labels_dir / rel.with_suffix(".txt")
        if label.is_file():
            pairs.append((img, label))

    if not pairs:
        raise RuntimeError("No image/label pairs found in flat YOLO layout.")

    rnd = random.Random(seed)
    rnd.shuffle(pairs)

    total = len(pairs)
    tr, va, te = split_ratio
    n_train = (total * tr) // 100
    n_val = (total * va) // 100
    n_test = total - n_train - n_val

    split_bins = {
        "train": pairs[:n_train],
        "val": pairs[n_train : n_train + n_val],
        "test": pairs[n_train + n_val : n_train + n_val + n_test],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, items in split_bins.items():
        for img_src, label_src in items:
            img_dst = out_dir / split_name / "images" / img_src.name
            lbl_dst = out_dir / split_name / "labels" / label_src.name
            _copy_file(img_src, img_dst)
            _copy_file(label_src, lbl_dst)

    return out_dir


def _parse_split_ratio(raw: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 3:
        raise ValueError("--split-ratio must be 'train,val,test' like 80,10,10")
    vals = tuple(int(p) for p in parts)
    if any(v <= 0 for v in vals):
        raise ValueError("--split-ratio values must be > 0")
    if sum(vals) != 100:
        raise ValueError("--split-ratio must sum to 100")
    return vals


def _build_plan(
    source_path: Path,
    split_ratio: tuple[int, int, int] | None,
    seed: int,
    work_dir: Path,
) -> DatasetPlan:
    # 1) data.yaml path given directly
    if source_path.is_file() and source_path.name.lower() == "data.yaml":
        data_yaml = source_path
        root = data_yaml.parent
        ycfg = _load_yaml(data_yaml)
        num_classes = _count_classes_from_data_yaml(
            ycfg
        ) or _count_classes_from_yolo_labels(root / "labels")
        if _is_split_layout_images_train(root):
            return DatasetPlan(
                data_dir=root,
                annotation_format="yolo",
                frame_dir="images",
                num_classes=num_classes,
            )
        return DatasetPlan(
            data_dir=root,
            annotation_format="auto",
            frame_dir="",
            num_classes=num_classes,
        )

    # 2) JSON path
    if source_path.is_file() and source_path.suffix.lower() == ".json":
        mode = _infer_format_from_json(source_path)
        root = source_path.parent
        num_classes = 1
        with open(source_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if mode == "coco":
            cats = data.get("categories", [])
            if cats:
                num_classes = max(1, len(cats))
            else:
                ann = data.get("annotations", [])
                cat_ids = {
                    int(a.get("category_id", 0)) for a in ann if "category_id" in a
                }
                num_classes = max(1, len(cat_ids))
        elif mode == "custom_json":
            names = set()
            ids = set()
            for rec in data.get("images", []):
                for ann in rec.get("annotations", []):
                    if "category_name" in ann:
                        names.add(str(ann["category_name"]))
                    if "category_id" in ann:
                        try:
                            ids.add(int(ann["category_id"]))
                        except Exception:
                            pass
            num_classes = max(1, len(names) if names else len(ids))
        return DatasetPlan(
            data_dir=root, annotation_format=mode, frame_dir="", num_classes=num_classes
        )

    # 3) Directory source
    if not source_path.is_dir():
        raise FileNotFoundError(f"Unsupported dataset source: {source_path}")
    root = source_path

    data_yaml = _find_data_yaml(root)
    if data_yaml is not None:
        ycfg = _load_yaml(data_yaml)
        nc = _count_classes_from_data_yaml(ycfg)
    else:
        nc = None

    if _is_split_layout_train_images(root):
        num_classes = nc or _count_classes_from_yolo_labels(root / "train" / "labels")
        return DatasetPlan(
            data_dir=root,
            annotation_format="yolo",
            frame_dir="",
            num_classes=num_classes,
        )

    if _is_split_layout_images_train(root):
        num_classes = nc or _count_classes_from_yolo_labels(root / "labels" / "train")
        return DatasetPlan(
            data_dir=root,
            annotation_format="yolo",
            frame_dir="images",
            num_classes=num_classes,
        )

    if _is_flat_yolo_layout(root):
        if split_ratio is None:
            raise RuntimeError(
                "Detected flat YOLO dataset (<root>/images + <root>/labels) without train/val/test splits. "
                "Pass --split-ratio (e.g., 80,10,10)."
            )
        split_root = work_dir / "prepared_datasets" / f"{root.name}_split"
        split_root = _create_split_from_flat_yolo(root, split_ratio, seed, split_root)
        num_classes = nc or _count_classes_from_yolo_labels(
            split_root / "train" / "labels"
        )
        return DatasetPlan(
            data_dir=split_root,
            annotation_format="yolo",
            frame_dir="",
            num_classes=num_classes,
        )

    # try JSONs in directory
    json_candidates = sorted(root.rglob("*.json"))
    for js in json_candidates:
        try:
            mode = _infer_format_from_json(js)
        except Exception:
            continue
        return _build_plan(js, split_ratio, seed, work_dir)

    raise RuntimeError(
        "Could not infer dataset format/layout. Expected one of: "
        "split YOLO folders, flat YOLO folders, COCO JSON, or custom JSON."
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train AeroMixer on any uploaded dataset source."
    )
    p.add_argument(
        "--data",
        required=True,
        help="Dataset source path: zip, folder, data.yaml, or json.",
    )
    p.add_argument(
        "--config-file",
        default="config_files/presets/full.yaml",
        help="Base config file.",
    )
    p.add_argument(
        "--output-dir", default="output/auto_any_dataset", help="Run output directory."
    )
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument(
        "--split-ratio",
        default=None,
        help="Only for flat YOLO datasets (images/ + labels/). Format: train,val,test e.g. 80,10,10",
    )
    p.add_argument(
        "--skip-val-in-train", action="store_true", help="Pass through to train_net.py"
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Print resolved command and exit."
    )
    p.add_argument(
        "--extra-opts",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra config overrides passed to train_net.py (key value ...).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    work_dir = repo_root / "output" / "_auto_data_prep"
    work_dir.mkdir(parents=True, exist_ok=True)

    source_path = _resolve_source_path(Path(args.data).expanduser().resolve(), work_dir)
    ratio = _parse_split_ratio(args.split_ratio) if args.split_ratio else None
    plan = _build_plan(source_path, ratio, args.seed, work_dir)

    class_count = max(1, int(plan.num_classes))
    cmd: list[str] = [
        sys.executable,
        "train_net.py",
        "--config-file",
        args.config_file,
    ]
    if args.skip_val_in_train:
        cmd.append("--skip-val-in-train")

    opts: list[str] = [
        "OUTPUT_DIR",
        args.output_dir,
        "DATA.PATH_TO_DATA_DIR",
        str(plan.data_dir),
        "DATA.FRAME_DIR",
        str(plan.frame_dir),
        "DATA.ANNOTATION_FORMAT",
        str(plan.annotation_format),
        "SOLVER.MAX_EPOCH",
        str(args.epochs),
        "SOLVER.IMAGES_PER_BATCH",
        str(args.batch_size),
        "DATALOADER.NUM_WORKERS",
        str(args.num_workers),
        "MODEL.STM.ACTION_CLASSES",
        str(class_count),
        "MODEL.STM.OBJECT_CLASSES",
        str(class_count),
        "MODEL.STM.NUM_ACT",
        str(class_count),
        "MODEL.STM.NUM_CLS",
        str(class_count),
    ]
    if args.extra_opts:
        opts.extend(args.extra_opts)

    cmd.extend(opts)

    print("Resolved training plan:")
    print(f"  source            : {source_path}")
    print(f"  data_dir          : {plan.data_dir}")
    print(f"  annotation_format : {plan.annotation_format}")
    print(f"  frame_dir         : {plan.frame_dir!r}")
    print(f"  num_classes       : {class_count}")
    print("Launching:")
    print("  " + " ".join(cmd))

    if args.dry_run:
        return 0

    proc = subprocess.run(cmd, cwd=str(repo_root), check=False)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
