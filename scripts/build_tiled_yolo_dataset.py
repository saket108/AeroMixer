#!/usr/bin/env python3
"""Build a tiled YOLO dataset for small-object training/evaluation.

Input can be YOLO split-first or images-first layouts.
Output is always split-first:
  <out_root>/<split>/images/*.jpg
  <out_root>/<split>/labels/*.txt
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2


VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class SplitDirs:
    image_dir: Path
    label_dir: Path


def _split_aliases(split: str) -> list[str]:
    s = str(split).lower()
    if s == "train":
        return ["train", "training", "stratified_train"]
    if s == "val":
        return ["val", "valid", "validation", "stratified_val"]
    if s == "test":
        return ["test", "val", "valid", "validation", "stratified_val"]
    return [s]


def _detect_layout(data_dir: Path, frame_dir: str) -> str:
    if (data_dir / "train" / "images").is_dir():
        return "split_first"
    frame_root = data_dir / frame_dir if frame_dir else data_dir / "images"
    if (
        frame_root.is_dir()
        and (frame_root / "train").is_dir()
        and (data_dir / "labels" / "train").is_dir()
    ):
        return "images_first"
    raise FileNotFoundError(
        f"Unsupported YOLO layout under {data_dir}. "
        "Expected split-first (<split>/images, <split>/labels) or images-first (images/<split>, labels/<split>)."
    )


def _resolve_split_dirs(
    data_dir: Path, frame_dir: str, layout: str, split: str
) -> SplitDirs | None:
    aliases = _split_aliases(split)
    if layout == "split_first":
        for alias in aliases:
            image_dir = data_dir / alias / "images"
            label_dir = data_dir / alias / "labels"
            if image_dir.is_dir():
                return SplitDirs(
                    image_dir=image_dir,
                    label_dir=(
                        label_dir
                        if label_dir.is_dir()
                        else (data_dir / alias / "labels")
                    ),
                )
        return None

    frame_root = data_dir / frame_dir if frame_dir else data_dir / "images"
    labels_root = data_dir / "labels"
    for alias in aliases:
        image_dir = frame_root / alias
        label_dir = labels_root / alias
        if image_dir.is_dir():
            return SplitDirs(
                image_dir=image_dir,
                label_dir=label_dir if label_dir.is_dir() else (labels_root / alias),
            )
    return None


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


def _read_yolo_labels(
    label_path: Path, img_w: int, img_h: int
) -> list[tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []
    out: list[tuple[int, float, float, float, float]] = []
    for ln, raw in enumerate(
        label_path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1
    ):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = _safe_int_from_float(parts[0])
        xc = _safe_float(parts[1])
        yc = _safe_float(parts[2])
        bw = _safe_float(parts[3])
        bh = _safe_float(parts[4])
        if cls is None or xc is None or yc is None or bw is None or bh is None:
            continue
        if not (
            0.0 <= xc <= 1.0
            and 0.0 <= yc <= 1.0
            and 0.0 < bw <= 1.0
            and 0.0 < bh <= 1.0
        ):
            continue
        x1 = (xc - bw / 2.0) * img_w
        y1 = (yc - bh / 2.0) * img_h
        x2 = (xc + bw / 2.0) * img_w
        y2 = (yc + bh / 2.0) * img_h
        out.append((cls, x1, y1, x2, y2))
    return out


def _xyxy_to_yolo(
    x1: float, y1: float, x2: float, y2: float, w: int, h: int
) -> tuple[float, float, float, float]:
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0
    return xc / w, yc / h, bw / w, bh / h


def _grid_positions(length: int, tile: int, overlap: float) -> list[int]:
    if length <= tile:
        return [0]
    stride = max(1, int(round(tile * (1.0 - overlap))))
    positions = list(range(0, max(1, length - tile + 1), stride))
    last = length - tile
    if positions[-1] != last:
        positions.append(last)
    return sorted(set(positions))


def _tile_single_image(
    img_path: Path,
    label_path: Path,
    tile_prefix: str,
    out_img_dir: Path,
    out_lbl_dir: Path,
    tile_size: int,
    overlap: float,
    min_cover: float,
    include_empty_tiles: bool,
) -> dict[str, int]:
    image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if image is None:
        return {"tiles": 0, "tiles_with_boxes": 0, "boxes_written": 0}

    img_h, img_w = image.shape[:2]
    boxes = _read_yolo_labels(label_path, img_w, img_h)
    xs = _grid_positions(img_w, tile_size, overlap)
    ys = _grid_positions(img_h, tile_size, overlap)

    ext = img_path.suffix if img_path.suffix else ".jpg"

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    tiles = 0
    tiles_with_boxes = 0
    boxes_written = 0

    for y in ys:
        for x in xs:
            x2 = min(img_w, x + tile_size)
            y2 = min(img_h, y + tile_size)
            tile = image[y:y2, x:x2]
            tw = x2 - x
            th = y2 - y
            if tw <= 2 or th <= 2:
                continue

            tile_lines: list[str] = []
            for cls, bx1, by1, bx2, by2 in boxes:
                ix1 = max(float(x), bx1)
                iy1 = max(float(y), by1)
                ix2 = min(float(x2), bx2)
                iy2 = min(float(y2), by2)
                iw = max(0.0, ix2 - ix1)
                ih = max(0.0, iy2 - iy1)
                if iw <= 0.0 or ih <= 0.0:
                    continue
                inter = iw * ih
                box_area = max(1e-6, (bx2 - bx1) * (by2 - by1))
                cover = inter / box_area
                if cover < min_cover:
                    continue

                tx1 = ix1 - x
                ty1 = iy1 - y
                tx2 = ix2 - x
                ty2 = iy2 - y
                xc, yc, bw, bh = _xyxy_to_yolo(tx1, ty1, tx2, ty2, tw, th)
                if bw <= 1e-6 or bh <= 1e-6:
                    continue
                tile_lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            if (not tile_lines) and (not include_empty_tiles):
                continue

            tile_name = f"{tile_prefix}__x{x}_y{y}{ext}"
            out_img_path = out_img_dir / tile_name
            out_lbl_path = out_lbl_dir / f"{Path(tile_name).stem}.txt"
            cv2.imwrite(str(out_img_path), tile)
            out_lbl_path.write_text(
                "\n".join(tile_lines) + ("\n" if tile_lines else ""), encoding="utf-8"
            )

            tiles += 1
            if tile_lines:
                tiles_with_boxes += 1
                boxes_written += len(tile_lines)

    return {
        "tiles": tiles,
        "tiles_with_boxes": tiles_with_boxes,
        "boxes_written": boxes_written,
    }


def _copy_split_as_is(in_dirs: SplitDirs, out_split_root: Path) -> dict[str, int]:
    out_img_dir = out_split_root / "images"
    out_lbl_dir = out_split_root / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    images = [
        p
        for p in in_dirs.image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS
    ]
    copied_imgs = 0
    copied_lbls = 0
    for img in images:
        rel = img.relative_to(in_dirs.image_dir)
        dst = out_img_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, dst)
        copied_imgs += 1
        lbl = in_dirs.label_dir / rel.with_suffix(".txt")
        if lbl.exists():
            lbl_dst = out_lbl_dir / rel.with_suffix(".txt")
            lbl_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(lbl, lbl_dst)
            copied_lbls += 1
    return {"images_copied": copied_imgs, "labels_copied": copied_lbls}


def build_tiled_dataset(
    data_dir: Path,
    frame_dir: str,
    out_dir: Path,
    tile_size: int,
    overlap: float,
    min_cover: float,
    tile_splits: list[str],
    include_empty_tiles: bool = False,
) -> dict[str, Any]:
    if tile_size <= 0:
        raise ValueError("tile_size must be > 0")
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0, 1)")
    if not (0.0 < min_cover <= 1.0):
        raise ValueError("min_cover must be in (0, 1]")

    layout = _detect_layout(data_dir, frame_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_report: dict[str, Any] = {}
    for split in ["train", "val", "test"]:
        dirs = _resolve_split_dirs(data_dir, frame_dir, layout, split)
        if dirs is None:
            continue
        out_split_root = out_dir / split

        if split in tile_splits:
            images = [
                p
                for p in dirs.image_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS
            ]
            out_img_dir = out_split_root / "images"
            out_lbl_dir = out_split_root / "labels"
            totals = {
                "source_images": len(images),
                "tiles": 0,
                "tiles_with_boxes": 0,
                "boxes_written": 0,
            }
            for img in images:
                rel = img.relative_to(dirs.image_dir)
                lbl = dirs.label_dir / rel.with_suffix(".txt")
                tile_prefix = rel.with_suffix("").as_posix().replace("/", "__")
                stats = _tile_single_image(
                    img_path=img,
                    label_path=lbl,
                    tile_prefix=tile_prefix,
                    out_img_dir=out_img_dir,
                    out_lbl_dir=out_lbl_dir,
                    tile_size=tile_size,
                    overlap=overlap,
                    min_cover=min_cover,
                    include_empty_tiles=include_empty_tiles,
                )
                totals["tiles"] += stats["tiles"]
                totals["tiles_with_boxes"] += stats["tiles_with_boxes"]
                totals["boxes_written"] += stats["boxes_written"]
            split_report[split] = {"mode": "tiled", **totals}
        else:
            copy_stats = _copy_split_as_is(dirs, out_split_root)
            split_report[split] = {"mode": "copied", **copy_stats}

    # Preserve class metadata when present.
    for name in ["data.yaml", "classes.txt"]:
        src = data_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)

    return {
        "source_data_dir": str(data_dir),
        "output_data_dir": str(out_dir),
        "layout": layout,
        "frame_dir": frame_dir,
        "tile_size": tile_size,
        "overlap": overlap,
        "min_cover": min_cover,
        "tile_splits": tile_splits,
        "include_empty_tiles": include_empty_tiles,
        "splits": split_report,
    }


def _parse_splits(raw: str) -> list[str]:
    valid = {"train", "val", "test"}
    out = []
    for token in raw.split(","):
        s = token.strip().lower()
        if not s:
            continue
        if s not in valid:
            raise ValueError(f"Invalid split '{s}'. Allowed: train,val,test")
        if s not in out:
            out.append(s)
    if not out:
        raise ValueError("No valid splits provided.")
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build tiled YOLO dataset for small-object training/eval."
    )
    p.add_argument("--data-dir", required=True, help="Source dataset directory.")
    p.add_argument(
        "--frame-dir",
        default="",
        help="Frame dir (e.g., 'images' for images-first layout).",
    )
    p.add_argument("--out-dir", required=True, help="Output tiled dataset directory.")
    p.add_argument("--tile-size", type=int, default=640)
    p.add_argument("--overlap", type=float, default=0.2)
    p.add_argument("--min-cover", type=float, default=0.35)
    p.add_argument(
        "--tile-splits",
        default="train,val,test",
        help="Comma-separated: train,val,test",
    )
    p.add_argument(
        "--include-empty-tiles", action="store_true", help="Keep tiles with no labels."
    )
    p.add_argument("--report-out", default=None, help="Optional JSON report path.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    tile_splits = _parse_splits(args.tile_splits)
    report = build_tiled_dataset(
        data_dir=Path(args.data_dir).expanduser().resolve(),
        frame_dir=str(args.frame_dir),
        out_dir=Path(args.out_dir).expanduser().resolve(),
        tile_size=int(args.tile_size),
        overlap=float(args.overlap),
        min_cover=float(args.min_cover),
        tile_splits=tile_splits,
        include_empty_tiles=bool(args.include_empty_tiles),
    )

    report_out = (
        Path(args.report_out).resolve()
        if args.report_out
        else Path(report["output_data_dir"]) / "tiling_report.json"
    )
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Tiling report: {report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
