#!/usr/bin/env python3
"""Dataset validation for AeroMixer (fail-fast quality gate)."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from internal import train_any_dataset as ds


VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _split_aliases(split: str) -> list[str]:
    s = str(split).lower()
    if s == "train":
        return ["train", "training", "stratified_train"]
    if s == "val":
        return ["val", "valid", "validation", "stratified_val"]
    if s == "test":
        return ["test", "val", "valid", "validation", "stratified_val"]
    return [s]


def _norm_key(path: Path) -> str:
    return path.with_suffix("").as_posix().lower()


def _iter_images(root: Path) -> list[Path]:
    return sorted(
        [
            p
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS
        ]
    )


def _iter_labels(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*.txt") if p.is_file()])


def _detect_yolo_layout(data_dir: Path, frame_dir: str) -> str:
    if (data_dir / "train" / "images").is_dir():
        return "split_first"

    frame_root = data_dir / frame_dir if frame_dir else data_dir / "images"
    if (
        frame_root.is_dir()
        and (frame_root / "train").is_dir()
        and (data_dir / "labels" / "train").is_dir()
    ):
        return "images_first"

    if (data_dir / "images").is_dir() and (data_dir / "labels").is_dir():
        return "flat"

    return "unknown"


def _resolve_split_dirs(
    data_dir: Path,
    frame_dir: str,
    layout: str,
    split: str,
) -> tuple[Path | None, Path | None, str | None]:
    aliases = _split_aliases(split)

    if layout == "split_first":
        for a in aliases:
            image_dir = data_dir / a / "images"
            label_dir = data_dir / a / "labels"
            if image_dir.is_dir():
                return image_dir, label_dir if label_dir.is_dir() else None, a
        return None, None, None

    if layout == "images_first":
        frame_root = data_dir / frame_dir if frame_dir else data_dir / "images"
        labels_root = data_dir / "labels"
        for a in aliases:
            image_dir = frame_root / a
            label_dir = labels_root / a
            if image_dir.is_dir():
                return image_dir, label_dir if label_dir.is_dir() else None, a
        return None, None, None

    if layout == "flat":
        image_dir = data_dir / "images"
        label_dir = data_dir / "labels"
        if image_dir.is_dir():
            # Flat layout has no true train/val/test separation.
            return image_dir, label_dir if label_dir.is_dir() else None, "flat"
        return None, None, None

    return None, None, None


def _safe_float(x: str) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _safe_int_from_float(x: str) -> int | None:
    v = _safe_float(x)
    if v is None:
        return None
    i = int(v)
    if abs(v - i) > 1e-6:
        return None
    return i


def _validate_yolo_split(
    split: str,
    image_dir: Path,
    label_dir: Path | None,
    num_classes_expected: int,
) -> dict[str, Any]:
    images = _iter_images(image_dir)
    image_map = {_norm_key(p.relative_to(image_dir)): p for p in images}

    labels: list[Path] = _iter_labels(label_dir) if label_dir is not None else []
    label_map = (
        {_norm_key(p.relative_to(label_dir)): p for p in labels}
        if label_dir is not None
        else {}
    )

    image_keys = set(image_map.keys())
    label_keys = set(label_map.keys())
    missing_label_keys = sorted(image_keys - label_keys)
    orphan_label_keys = sorted(label_keys - image_keys)

    bad_lines = 0
    bad_boxes = 0
    bad_class_ids = 0
    parse_errors = 0
    valid_boxes = 0
    class_hist = [0 for _ in range(max(1, int(num_classes_expected)))]
    classes_seen = set()
    tiny_boxes = 0
    small_boxes = 0
    example_issues: list[str] = []

    for key in sorted(image_keys.intersection(label_keys)):
        label_path = label_map[key]
        try:
            lines = label_path.read_text(encoding="utf-8").splitlines()
        except Exception as e:
            parse_errors += 1
            if len(example_issues) < 10:
                example_issues.append(f"{label_path}: read error: {e}")
            continue

        for ln, raw in enumerate(lines, 1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                bad_lines += 1
                if len(example_issues) < 10:
                    example_issues.append(f"{label_path}:{ln} malformed line")
                continue

            cls_id = _safe_int_from_float(parts[0])
            x = _safe_float(parts[1])
            y = _safe_float(parts[2])
            w = _safe_float(parts[3])
            h = _safe_float(parts[4])
            if cls_id is None or x is None or y is None or w is None or h is None:
                parse_errors += 1
                if len(example_issues) < 10:
                    example_issues.append(f"{label_path}:{ln} parse error")
                continue

            if cls_id < 0 or cls_id >= num_classes_expected:
                bad_class_ids += 1
                if len(example_issues) < 10:
                    example_issues.append(
                        f"{label_path}:{ln} class_id={cls_id} outside [0,{num_classes_expected - 1}]"
                    )
                continue

            vals = [x, y, w, h]
            if any((not math.isfinite(v)) for v in vals):
                bad_boxes += 1
                if len(example_issues) < 10:
                    example_issues.append(f"{label_path}:{ln} non-finite bbox values")
                continue
            if not (
                0.0 <= x <= 1.0
                and 0.0 <= y <= 1.0
                and 0.0 < w <= 1.0
                and 0.0 < h <= 1.0
            ):
                bad_boxes += 1
                if len(example_issues) < 10:
                    example_issues.append(
                        f"{label_path}:{ln} bbox out of normalized range"
                    )
                continue

            valid_boxes += 1
            classes_seen.add(cls_id)
            if cls_id < len(class_hist):
                class_hist[cls_id] += 1
            area = w * h
            if area < 0.0005:
                tiny_boxes += 1
            if area < 0.0025:
                small_boxes += 1

    split_summary = {
        "split": split,
        "image_dir": str(image_dir),
        "label_dir": str(label_dir) if label_dir is not None else None,
        "num_images": len(images),
        "num_label_files": len(labels),
        "missing_label_files": len(missing_label_keys),
        "orphan_label_files": len(orphan_label_keys),
        "valid_boxes": valid_boxes,
        "bad_lines": bad_lines,
        "bad_boxes": bad_boxes,
        "bad_class_ids": bad_class_ids,
        "parse_errors": parse_errors,
        "tiny_box_ratio": (tiny_boxes / valid_boxes) if valid_boxes > 0 else 0.0,
        "small_box_ratio": (small_boxes / valid_boxes) if valid_boxes > 0 else 0.0,
        "classes_seen": sorted(classes_seen),
        "class_hist": class_hist,
        "examples": example_issues,
    }
    split_summary["_image_name_set"] = {p.name.lower() for p in images}
    split_summary["_missing_label_examples"] = missing_label_keys[:10]
    split_summary["_orphan_label_examples"] = orphan_label_keys[:10]
    return split_summary


def _validate_yolo_dataset(plan: ds.DatasetPlan) -> dict[str, Any]:
    data_dir = Path(plan.data_dir)
    layout = _detect_yolo_layout(data_dir, plan.frame_dir)
    warnings: list[str] = []
    errors: list[str] = []

    if layout == "unknown":
        return {
            "ok": False,
            "errors": [f"Could not detect YOLO layout under {data_dir}"],
            "warnings": [],
            "summary": {"layout": layout},
        }
    if layout == "flat":
        warnings.append(
            "Flat YOLO layout detected (images/ + labels/ without split dirs). "
            "Use split generation (e.g. --split-ratio 80,10,10) for reliable train/val/test evaluation."
        )

    split_summaries: dict[str, dict[str, Any]] = {}
    for split in ["train", "val", "test"]:
        image_dir, label_dir, _ = _resolve_split_dirs(
            data_dir, plan.frame_dir, layout, split
        )
        if image_dir is None:
            if split == "train":
                errors.append("Training split directory not found.")
            else:
                warnings.append(f"Split '{split}' directory not found.")
            continue
        split_summary = _validate_yolo_split(
            split, image_dir, label_dir, max(1, int(plan.num_classes))
        )
        split_summaries[split] = split_summary

        if split == "train":
            if split_summary["num_images"] == 0:
                errors.append("Training split has zero images.")
            if split_summary["valid_boxes"] == 0:
                errors.append("Training split has zero valid boxes.")
        if (
            split_summary["bad_lines"] > 0
            or split_summary["bad_boxes"] > 0
            or split_summary["bad_class_ids"] > 0
        ):
            errors.append(
                f"Split '{split}' has invalid labels "
                f"(bad_lines={split_summary['bad_lines']}, bad_boxes={split_summary['bad_boxes']}, "
                f"bad_class_ids={split_summary['bad_class_ids']})."
            )
        if split_summary["orphan_label_files"] > 0:
            errors.append(
                f"Split '{split}' has orphan label files (no matching image)."
            )
        if split_summary["missing_label_files"] > 0:
            warnings.append(
                f"Split '{split}' has {split_summary['missing_label_files']} images without labels "
                "(may be intentional background images)."
            )

    # Leakage check by image filename across available splits.
    def _names(s: str) -> set[str]:
        return split_summaries.get(s, {}).get("_image_name_set", set())

    leaks = {
        "train_val": sorted(_names("train").intersection(_names("val"))),
        "train_test": sorted(_names("train").intersection(_names("test"))),
        "val_test": sorted(_names("val").intersection(_names("test"))),
    }
    for pair, values in leaks.items():
        if values:
            errors.append(
                f"Split leakage detected ({pair}): {len(values)} duplicate image filenames "
                f"(examples: {values[:5]})."
            )

    global_hist = [0 for _ in range(max(1, int(plan.num_classes)))]
    for split_summary in split_summaries.values():
        hist = split_summary["class_hist"]
        for i in range(min(len(hist), len(global_hist))):
            global_hist[i] += int(hist[i])

    nonzero = [x for x in global_hist if x > 0]
    if nonzero:
        imbalance_ratio = max(nonzero) / max(1.0, min(nonzero))
    else:
        imbalance_ratio = 0.0
        warnings.append("No non-zero class counts found in parsed labels.")
    if imbalance_ratio > 20.0:
        warnings.append(
            f"Severe class imbalance detected (max/min positive count ratio = {imbalance_ratio:.2f})."
        )

    # Remove internal helper keys before serialization.
    for split_summary in split_summaries.values():
        split_summary.pop("_image_name_set", None)

    summary = {
        "annotation_format": "yolo",
        "data_dir": str(plan.data_dir),
        "frame_dir": plan.frame_dir,
        "layout": layout,
        "num_classes_expected": int(plan.num_classes),
        "splits": split_summaries,
        "global_class_hist": global_hist,
        "global_class_imbalance_ratio": imbalance_ratio,
        "leakage": {k: len(v) for k, v in leaks.items()},
    }

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "summary": summary,
    }


def _is_valid_yolo_record(
    parts: list[str], num_classes_expected: int
) -> tuple[bool, str]:
    if len(parts) < 5:
        return False, "malformed"
    cls_id = _safe_int_from_float(parts[0])
    x = _safe_float(parts[1])
    y = _safe_float(parts[2])
    w = _safe_float(parts[3])
    h = _safe_float(parts[4])
    if cls_id is None or x is None or y is None or w is None or h is None:
        return False, "parse_error"
    if cls_id < 0 or cls_id >= num_classes_expected:
        return False, "bad_class_id"
    vals = [x, y, w, h]
    if any((not math.isfinite(v)) for v in vals):
        return False, "non_finite"
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
        return False, "bbox_range"
    return True, ""


def _resolve_available_yolo_splits(plan: ds.DatasetPlan) -> dict[str, tuple[Path, Path | None]]:
    data_dir = Path(plan.data_dir)
    layout = _detect_yolo_layout(data_dir, plan.frame_dir)
    out: dict[str, tuple[Path, Path | None]] = {}
    for split in ["train", "val", "test"]:
        image_dir, label_dir, _ = _resolve_split_dirs(
            data_dir=data_dir,
            frame_dir=plan.frame_dir,
            layout=layout,
            split=split,
        )
        if image_dir is not None:
            out[split] = (image_dir, label_dir)
    return out


def _repair_yolo_dataset(
    plan: ds.DatasetPlan,
    *,
    fix_label_lines: bool,
    drop_orphan_labels: bool,
    create_missing_labels: bool,
    fix_leakage: bool,
) -> dict[str, Any]:
    splits = _resolve_available_yolo_splits(plan)
    if not splits:
        return {
            "applied": False,
            "reason": "no yolo splits found",
            "actions": {},
        }

    num_classes = max(1, int(plan.num_classes))
    actions: dict[str, Any] = {
        "splits": {},
        "totals": {
            "label_files_rewritten": 0,
            "invalid_lines_removed": 0,
            "orphan_labels_removed": 0,
            "missing_labels_created": 0,
            "leak_images_removed": 0,
            "leak_labels_removed": 0,
        },
    }

    split_name_to_images: dict[str, list[Path]] = {}
    split_name_to_labels: dict[str, dict[str, Path]] = {}

    # Pass 1: label hygiene inside each split.
    for split, (image_dir, label_dir) in splits.items():
        images = _iter_images(image_dir)
        split_name_to_images[split] = images
        image_key_to_rel = {
            _norm_key(p.relative_to(image_dir)): p.relative_to(image_dir) for p in images
        }
        labels = _iter_labels(label_dir) if label_dir is not None else []
        label_key_to_path = (
            {_norm_key(p.relative_to(label_dir)): p for p in labels} if label_dir is not None else {}
        )
        split_name_to_labels[split] = label_key_to_path

        split_actions = {
            "label_files_rewritten": 0,
            "invalid_lines_removed": 0,
            "orphan_labels_removed": 0,
            "missing_labels_created": 0,
        }

        image_keys = set(image_key_to_rel.keys())
        label_keys = set(label_key_to_path.keys())
        orphan_keys = sorted(label_keys - image_keys)
        missing_keys = sorted(image_keys - label_keys)

        if drop_orphan_labels and label_dir is not None:
            for key in orphan_keys:
                p = label_key_to_path[key]
                try:
                    p.unlink()
                    split_actions["orphan_labels_removed"] += 1
                except Exception:
                    pass

        if create_missing_labels and label_dir is not None:
            for key in missing_keys:
                rel = image_key_to_rel[key]
                out = label_dir / rel.with_suffix(".txt")
                out.parent.mkdir(parents=True, exist_ok=True)
                if not out.exists():
                    out.write_text("", encoding="utf-8")
                    split_actions["missing_labels_created"] += 1

        if fix_label_lines and label_dir is not None:
            current_labels = _iter_labels(label_dir)
            for txt in current_labels:
                try:
                    lines = txt.read_text(encoding="utf-8").splitlines()
                except Exception:
                    continue
                kept: list[str] = []
                removed = 0
                for raw in lines:
                    line = raw.strip()
                    if not line:
                        continue
                    parts = line.split()
                    ok, _ = _is_valid_yolo_record(parts, num_classes)
                    if ok:
                        kept.append(" ".join(parts[:5]))
                    else:
                        removed += 1
                if removed > 0:
                    txt.write_text(("\n".join(kept) + ("\n" if kept else "")), encoding="utf-8")
                    split_actions["label_files_rewritten"] += 1
                    split_actions["invalid_lines_removed"] += removed

        actions["splits"][split] = split_actions
        for k in actions["totals"].keys():
            actions["totals"][k] += split_actions.get(k, 0)

    # Pass 2: split leakage by filename (priority train > val > test).
    if fix_leakage:
        name_index: dict[str, list[tuple[str, Path]]] = defaultdict(list)
        for split, images in split_name_to_images.items():
            for p in images:
                name_index[p.name.lower()].append((split, p))

        priority = {"train": 0, "val": 1, "test": 2}
        for _, refs in name_index.items():
            if len(refs) <= 1:
                continue
            refs_sorted = sorted(refs, key=lambda x: priority.get(x[0], 99))
            keep_split, keep_img = refs_sorted[0]
            _ = (keep_split, keep_img)
            for split, img in refs_sorted[1:]:
                try:
                    img.unlink()
                    actions["totals"]["leak_images_removed"] += 1
                except Exception:
                    pass

                image_dir, label_dir = splits.get(split, (None, None))
                if image_dir is None or label_dir is None:
                    continue
                try:
                    rel = img.relative_to(image_dir).with_suffix(".txt")
                    lbl = label_dir / rel
                    if lbl.exists():
                        lbl.unlink()
                        actions["totals"]["leak_labels_removed"] += 1
                except Exception:
                    pass

    return {
        "applied": True,
        "num_classes_expected": num_classes,
        "options": {
            "fix_label_lines": fix_label_lines,
            "drop_orphan_labels": drop_orphan_labels,
            "create_missing_labels": create_missing_labels,
            "fix_leakage": fix_leakage,
        },
        "actions": actions,
    }


def _validate_generic_json_dataset(plan: ds.DatasetPlan) -> dict[str, Any]:
    data_dir = Path(plan.data_dir)
    json_files = sorted([p for p in data_dir.rglob("*.json") if p.is_file()])
    warnings: list[str] = []
    errors: list[str] = []
    summaries: list[dict[str, Any]] = []

    if not json_files:
        return {
            "ok": False,
            "errors": [
                f"No JSON files found under {data_dir} for annotation format '{plan.annotation_format}'."
            ],
            "warnings": [],
            "summary": {
                "annotation_format": plan.annotation_format,
                "data_dir": str(data_dir),
            },
        }

    parsed_any = False
    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue

        if (
            plan.annotation_format == "coco"
            and "images" in data
            and "annotations" in data
        ):
            parsed_any = True
            summaries.append(
                {
                    "file": str(path),
                    "num_images": len(data.get("images", [])),
                    "num_annotations": len(data.get("annotations", [])),
                    "num_categories": len(data.get("categories", [])),
                }
            )
        elif plan.annotation_format == "custom_json" and isinstance(
            data.get("images"), list
        ):
            parsed_any = True
            num_anns = 0
            for rec in data.get("images", []):
                if isinstance(rec, dict):
                    anns = rec.get("annotations", [])
                    if isinstance(anns, list):
                        num_anns += len(anns)
            summaries.append(
                {
                    "file": str(path),
                    "num_images": len(data.get("images", [])),
                    "num_annotations": num_anns,
                }
            )

    if not parsed_any:
        errors.append(
            f"Could not parse any JSON matching annotation format '{plan.annotation_format}'."
        )

    if len(summaries) > 1:
        warnings.append(
            "Multiple JSON annotation files detected; ensure split routing is intentional."
        )

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "annotation_format": plan.annotation_format,
            "data_dir": str(data_dir),
            "json_files": summaries,
        },
    }


def validate_dataset_plan(plan: ds.DatasetPlan) -> dict[str, Any]:
    fmt = str(plan.annotation_format).lower()
    if fmt == "yolo":
        return _validate_yolo_dataset(plan)
    if fmt in {"coco", "custom_json"}:
        return _validate_generic_json_dataset(plan)
    return {
        "ok": True,
        "errors": [],
        "warnings": [
            f"No deep validator implemented for annotation format '{fmt}'. Skipped."
        ],
        "summary": {"annotation_format": fmt, "data_dir": str(plan.data_dir)},
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate dataset quality for AeroMixer pipeline."
    )
    p.add_argument(
        "--data",
        required=True,
        help="Dataset source path: zip, folder, data.yaml, or json.",
    )
    p.add_argument("--seed", type=int, default=2)
    p.add_argument(
        "--split-ratio", default=None, help="Only for flat YOLO datasets. e.g. 80,10,10"
    )
    p.add_argument(
        "--report-out", default=None, help="Optional JSON path for validation report."
    )
    p.add_argument(
        "--allow-errors",
        action="store_true",
        help="Exit zero even when validation fails.",
    )
    p.add_argument(
        "--fix",
        action="store_true",
        help="Apply safe auto-fixes for YOLO datasets, then re-validate.",
    )
    p.add_argument(
        "--no-fix-label-lines",
        action="store_true",
        help="When --fix is enabled, do not remove malformed/invalid YOLO lines.",
    )
    p.add_argument(
        "--no-drop-orphan-labels",
        action="store_true",
        help="When --fix is enabled, keep orphan label files.",
    )
    p.add_argument(
        "--no-create-missing-labels",
        action="store_true",
        help="When --fix is enabled, do not create empty labels for unlabeled images.",
    )
    p.add_argument(
        "--no-fix-leakage",
        action="store_true",
        help="When --fix is enabled, do not remove duplicate filenames across splits.",
    )
    p.add_argument(
        "--fix-report-out",
        default=None,
        help="Optional JSON path for fix action report.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    work_dir = repo_root / "output" / "_auto_data_prep"
    work_dir.mkdir(parents=True, exist_ok=True)

    source_path = ds._resolve_source_path(
        Path(args.data).expanduser().resolve(), work_dir
    )
    ratio = ds._parse_split_ratio(args.split_ratio) if args.split_ratio else None
    plan = ds._build_plan(source_path, ratio, args.seed, work_dir)

    report = validate_dataset_plan(plan)
    pre_fix_report = dict(report)
    report["dataset_source"] = str(source_path)
    report["resolved_plan"] = {
        "data_dir": str(plan.data_dir),
        "annotation_format": plan.annotation_format,
        "frame_dir": plan.frame_dir,
        "num_classes": plan.num_classes,
    }

    fix_report = None
    if args.fix:
        if str(plan.annotation_format).lower() != "yolo":
            print(
                f"--fix requested, but annotation format is '{plan.annotation_format}'. "
                "Auto-fix currently supports YOLO only."
            )
        else:
            fix_report = _repair_yolo_dataset(
                plan,
                fix_label_lines=(not args.no_fix_label_lines),
                drop_orphan_labels=(not args.no_drop_orphan_labels),
                create_missing_labels=(not args.no_create_missing_labels),
                fix_leakage=(not args.no_fix_leakage),
            )
            report = validate_dataset_plan(plan)
            report["dataset_source"] = str(source_path)
            report["resolved_plan"] = {
                "data_dir": str(plan.data_dir),
                "annotation_format": plan.annotation_format,
                "frame_dir": plan.frame_dir,
                "num_classes": plan.num_classes,
            }
            report["pre_fix"] = pre_fix_report
            report["fix_report"] = fix_report

    if args.report_out:
        out = Path(args.report_out)
    else:
        out = (
            repo_root / "output" / "_auto_data_prep" / "dataset_validation_report.json"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if fix_report is not None:
        fix_out = Path(args.fix_report_out) if args.fix_report_out else out.with_name("dataset_fix_report.json")
        with open(fix_out, "w", encoding="utf-8") as f:
            json.dump(fix_report, f, indent=2)
        print(f"Fix action report: {fix_out}")

    print(f"Validation report: {out}")
    print(
        f"ok={report['ok']} errors={len(report['errors'])} warnings={len(report['warnings'])}"
    )
    if report["errors"]:
        for e in report["errors"][:10]:
            print(f"ERROR: {e}")
    if report["warnings"]:
        for w in report["warnings"][:10]:
            print(f"WARN : {w}")

    if not report["ok"] and not args.allow_errors:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
