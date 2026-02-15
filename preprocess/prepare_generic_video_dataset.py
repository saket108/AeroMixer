#!/usr/bin/env python3
"""
Convert raw videos into the AeroMixer generic-video layout.

Outputs:
  1) Frames: <output_root>/frames/<video_id>/<frame_id>.jpg
  2) Split file(s): <output_root>/<split>.txt

Split TXT format expected by VideoDataset:
  <video_id> <frame_id> <x1> <y1> <x2> <y2> <class_id_or_name>

Notes:
- If --annotation-json is not provided, placeholder full-frame boxes are written
  so the pipeline can run a smoke test. Replace placeholders with real labels
  for real training/evaluation.
- Video IDs are generated from relative paths under --videos-dir (extension
  removed), with path separators normalized to '/'.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frames from videos and generate AeroMixer split TXT."
    )
    parser.add_argument("--videos-dir", required=True, type=Path, help="Directory with raw videos.")
    parser.add_argument("--output-root", required=True, type=Path, help="Output dataset root.")
    parser.add_argument(
        "--split",
        default="test",
        type=str,
        help="Split name(s): 'train', 'test', 'all', or comma list like 'train,test' (default: test).",
    )
    parser.add_argument(
        "--annotation-json",
        default=None,
        type=str,
        help=(
            "Optional JSON annotation file. Use '{split}' in the path to load per-split files "
            "(e.g., data/annotations/{split}.json)."
        ),
    )
    parser.add_argument(
        "--bbox-format",
        default="auto",
        choices=["auto", "xyxy", "xywh"],
        help="BBox format for generic JSON records (default: auto).",
    )
    parser.add_argument(
        "--every-n",
        default=1,
        type=int,
        help="Keep one frame every N source frames (default: 1).",
    )
    parser.add_argument(
        "--target-fps",
        default=0.0,
        type=float,
        help="Optional target FPS extraction. If >0, overrides --every-n.",
    )
    parser.add_argument(
        "--max-frames-per-video",
        default=0,
        type=int,
        help="Optional cap per video. 0 means no cap.",
    )
    parser.add_argument(
        "--image-ext",
        default=".jpg",
        type=str,
        help="Frame file extension (default: .jpg).",
    )
    parser.add_argument(
        "--jpeg-quality",
        default=95,
        type=int,
        help="JPEG quality if --image-ext is .jpg/.jpeg (default: 95).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search videos under --videos-dir.",
    )
    parser.add_argument(
        "--overwrite-frames",
        action="store_true",
        help="Delete and rebuild existing extracted frame folders.",
    )
    parser.add_argument(
        "--no-placeholders",
        action="store_true",
        help="Require annotation JSON. Do not generate placeholder boxes.",
    )
    parser.add_argument(
        "--placeholder-class",
        default="0",
        type=str,
        help="Class token for placeholder labels (default: 0).",
    )
    parser.add_argument(
        "--train-ratio",
        default=0.8,
        type=float,
        help="Train split ratio used only when --split includes both train and test with placeholders (default: 0.8).",
    )
    return parser.parse_args()


def _safe_token(value: object) -> str:
    token = str(value).strip().replace("\\", "/")
    token = re.sub(r"\s+", "_", token)
    token = re.sub(r"[^A-Za-z0-9._/-]+", "_", token)
    token = token.strip("._/")
    return token or "item"


def _safe_label(value: object) -> str:
    token = str(value).strip()
    token = re.sub(r"\s+", "_", token)
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", token)
    token = token.strip("._")
    return token or "0"


def _normalize_video_key(value: object) -> str:
    text = str(value).strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    suffix = Path(text).suffix.lower()
    if suffix in VIDEO_EXTS:
        text = str(Path(text).with_suffix("")).replace("\\", "/")
    return text.lower()


def _resolve_target_splits(split_arg: str) -> List[str]:
    token = split_arg.strip().lower()
    if token == "all":
        return ["train", "test"]
    if "," in token:
        parts = [item.strip().lower() for item in token.split(",") if item.strip()]
        if not parts:
            raise ValueError("--split list is empty")
        return parts
    if not token:
        raise ValueError("--split cannot be empty")
    return [token]


def _build_video_id(video_path: Path, videos_dir: Path) -> str:
    rel = video_path.relative_to(videos_dir)
    rel_no_ext = rel.with_suffix("")
    parts = [_safe_token(part) for part in rel_no_ext.parts]
    return "/".join(parts)


def _collect_video_files(videos_dir: Path, recursive: bool) -> List[Path]:
    pattern = "**/*" if recursive else "*"
    videos = []
    for path in videos_dir.glob(pattern):
        if not path.is_file():
            continue
        if path.suffix.lower() in VIDEO_EXTS:
            videos.append(path)
    return sorted(videos)


def _pick_stride(capture: cv2.VideoCapture, every_n: int, target_fps: float) -> int:
    if target_fps <= 0:
        return max(1, every_n)
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if source_fps <= 0:
        return max(1, every_n)
    stride = int(round(source_fps / target_fps))
    return max(1, stride)


def _register_aliases(alias_to_video: Dict[str, Optional[str]], videos_dir: Path, video_path: Path, video_id: str) -> None:
    rel = video_path.relative_to(videos_dir)
    rel_posix = str(rel).replace("\\", "/")
    rel_no_ext = str(rel.with_suffix("")).replace("\\", "/")
    aliases = {
        _normalize_video_key(video_id),
        _normalize_video_key(rel_posix),
        _normalize_video_key(rel_no_ext),
        _normalize_video_key(video_path.name),
        _normalize_video_key(video_path.stem),
    }
    for alias in aliases:
        if not alias:
            continue
        current = alias_to_video.get(alias)
        if current is None:
            alias_to_video[alias] = video_id
        elif current != video_id:
            alias_to_video[alias] = ""


def _resolve_video_id(raw_video: object, alias_to_video: Dict[str, Optional[str]]) -> Optional[str]:
    key = _normalize_video_key(raw_video)
    if key in alias_to_video and alias_to_video[key]:
        return alias_to_video[key]

    # If annotation stores frame file path (e.g. "vid1/000123.jpg"), fallback to parent folder.
    path_like = str(raw_video).replace("\\", "/")
    suffix = Path(path_like).suffix.lower()
    if suffix in IMAGE_EXTS and "/" in path_like:
        parent_key = _normalize_video_key(path_like.rsplit("/", 1)[0])
        if parent_key in alias_to_video and alias_to_video[parent_key]:
            return alias_to_video[parent_key]

    return None


def _parse_frame_id(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        text = str(value)
        nums = re.findall(r"\d+", text)
        if not nums:
            return None
        return int(nums[-1])


def _read_bbox_from_record(record: Dict[str, object]) -> Optional[List[float]]:
    bbox = record.get("bbox") or record.get("box") or record.get("xyxy") or record.get("rect")
    if isinstance(bbox, dict):
        if all(k in bbox for k in ["x1", "y1", "x2", "y2"]):
            return [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]  # type: ignore[index]
        if all(k in bbox for k in ["x", "y", "w", "h"]):
            return [bbox["x"], bbox["y"], bbox["w"], bbox["h"]]  # type: ignore[index]
        return None
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    return [float(v) for v in bbox]


def _bbox_to_xyxy(
    bbox: List[float],
    bbox_format: str,
    frame_size: Optional[Tuple[int, int]],
) -> Optional[Tuple[float, float, float, float]]:
    if len(bbox) != 4:
        return None

    mode = bbox_format
    if mode == "auto":
        x1, y1, x2, y2 = bbox
        mode = "xywh" if (x2 <= x1 or y2 <= y1) else "xyxy"

    if mode == "xywh":
        x, y, w, h = bbox
        if frame_size and max(abs(v) for v in bbox) <= 1.5:
            fw, fh = frame_size
            x *= fw
            w *= fw
            y *= fh
            h *= fh
        x1, y1, x2, y2 = x, y, x + w, y + h
    else:
        x1, y1, x2, y2 = bbox
        if frame_size and max(abs(v) for v in bbox) <= 1.5:
            fw, fh = frame_size
            x1 *= fw
            x2 *= fw
            y1 *= fh
            y2 *= fh

    if frame_size:
        fw, fh = frame_size
        x1 = max(0.0, min(float(fw - 1), x1))
        x2 = max(0.0, min(float(fw - 1), x2))
        y1 = max(0.0, min(float(fh - 1), y1))
        y2 = max(0.0, min(float(fh - 1), y2))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _split_matches(record_split: object, target_split: str) -> bool:
    if record_split is None:
        return True
    text = str(record_split).strip().lower()
    if not text:
        return True
    target = target_split.strip().lower()
    valid = {target}
    if target == "test":
        valid.update({"val", "validation"})
    elif target == "train":
        valid.update({"trainval"})
    return text in valid


def _load_json_records(annotation_json: Path) -> Tuple[List[Dict[str, object]], Dict[str, str]]:
    with annotation_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    categories: Dict[str, str] = {}
    if isinstance(data, dict) and isinstance(data.get("categories"), list):
        for cat in data["categories"]:
            if not isinstance(cat, dict):
                continue
            cat_id = str(cat.get("id", "")).strip()
            cat_name = _safe_label(cat.get("name", cat_id))
            if cat_id:
                categories[cat_id] = cat_name

    # COCO-like: {"images": [...], "annotations": [...]}.
    if isinstance(data, dict) and isinstance(data.get("images"), list) and isinstance(data.get("annotations"), list):
        image_index: Dict[object, Dict[str, object]] = {}
        for image in data["images"]:
            if not isinstance(image, dict):
                continue
            image_id = image.get("id")
            if image_id is None:
                continue
            image_index[image_id] = image

        records: List[Dict[str, object]] = []
        for ann in data["annotations"]:
            if not isinstance(ann, dict):
                continue
            image = image_index.get(ann.get("image_id"))
            if image is None:
                continue
            record = {
                "video": image.get("video_id") or image.get("video") or image.get("vid") or image.get("file_name"),
                "frame": image.get("frame_id") or image.get("frame") or image.get("file_name"),
                "bbox": ann.get("bbox"),
                "label": categories.get(str(ann.get("category_id")), ann.get("category_id")),
                "split": ann.get("split") or image.get("split"),
                "_bbox_format": "xywh",
            }
            records.append(record)
        return records, categories

    if isinstance(data, dict) and isinstance(data.get("annotations"), list):
        return data["annotations"], categories
    if isinstance(data, list):
        return data, categories
    raise ValueError(
        "Unsupported JSON format. Use list, {'annotations': [...]}, or COCO-like {'images': ..., 'annotations': ...}."
    )


def _records_have_split(records: Iterable[Dict[str, object]]) -> bool:
    for rec in records:
        if not isinstance(rec, dict):
            continue
        split = rec.get("split")
        if split is not None and str(split).strip() != "":
            return True
    return False


def _build_annotations_from_json(
    records: Iterable[Dict[str, object]],
    split: str,
    alias_to_video: Dict[str, Optional[str]],
    frame_sizes: Dict[Tuple[str, int], Tuple[int, int]],
    video_default_size: Dict[str, Tuple[int, int]],
    default_bbox_format: str,
) -> List[Tuple[str, int, float, float, float, float, str]]:
    items: List[Tuple[str, int, float, float, float, float, str]] = []
    skipped = 0

    for rec in records:
        if not isinstance(rec, dict):
            continue
        if not _split_matches(rec.get("split"), split):
            continue

        raw_video = rec.get("video") or rec.get("video_id") or rec.get("vid") or rec.get("file_name") or rec.get("filename")
        raw_frame = rec.get("frame") if rec.get("frame") is not None else rec.get("frame_id")
        raw_label = (
            rec.get("label")
            if rec.get("label") is not None
            else rec.get("class")
            if rec.get("class") is not None
            else rec.get("class_id")
            if rec.get("class_id") is not None
            else rec.get("category_id")
            if rec.get("category_id") is not None
            else 0
        )

        if raw_video is None:
            skipped += 1
            continue

        frame_id = _parse_frame_id(raw_frame)
        if frame_id is None:
            # Try deriving frame_id from a path-like token.
            frame_id = _parse_frame_id(str(raw_video))
        if frame_id is None:
            skipped += 1
            continue

        video_id = _resolve_video_id(raw_video, alias_to_video)
        if video_id is None:
            skipped += 1
            continue

        bbox = _read_bbox_from_record(rec)
        if bbox is None:
            skipped += 1
            continue

        frame_size = frame_sizes.get((video_id, frame_id), video_default_size.get(video_id))
        bbox_format = str(rec.get("_bbox_format") or default_bbox_format).lower()
        xyxy = _bbox_to_xyxy(bbox, bbox_format, frame_size)
        if xyxy is None:
            skipped += 1
            continue

        label = _safe_label(raw_label)
        items.append((video_id, frame_id, xyxy[0], xyxy[1], xyxy[2], xyxy[3], label))

    if skipped > 0:
        print(f"[info] skipped {skipped} annotation records that could not be resolved for split '{split}'.")
    return items


def _extract_frames(
    video_path: Path,
    output_dir: Path,
    image_ext: str,
    jpeg_quality: int,
    every_n: int,
    target_fps: float,
    max_frames_per_video: int,
) -> Dict[int, Tuple[int, int]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_sizes: Dict[int, Tuple[int, int]] = {}
    stride = _pick_stride(capture, every_n=every_n, target_fps=target_fps)
    source_idx = 0
    saved = 0

    imwrite_args: List[int] = []
    if image_ext.lower() in [".jpg", ".jpeg"]:
        imwrite_args = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            keep = (source_idx % stride) == 0
            if keep:
                frame_id = source_idx + 1
                frame_path = output_dir / f"{frame_id:06d}{image_ext}"
                if not cv2.imwrite(str(frame_path), frame, imwrite_args):
                    raise RuntimeError(f"Failed to write frame: {frame_path}")
                h, w = frame.shape[:2]
                frame_sizes[frame_id] = (w, h)
                saved += 1
                if max_frames_per_video > 0 and saved >= max_frames_per_video:
                    break
            source_idx += 1
    finally:
        capture.release()

    return frame_sizes


def _write_split_file(output_txt: Path, lines: List[Tuple[str, int, float, float, float, float, str]]) -> None:
    lines_sorted = sorted(lines, key=lambda x: (x[0], x[1], x[6]))
    with output_txt.open("w", encoding="utf-8") as f:
        for vid, frame_id, x1, y1, x2, y2, label in lines_sorted:
            f.write(f"{vid} {frame_id} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {label}\n")


def _build_placeholder_annotations(
    frame_sizes: Dict[Tuple[str, int], Tuple[int, int]],
    target_splits: List[str],
    placeholder_class: str,
    train_ratio: float,
) -> Dict[str, List[Tuple[str, int, float, float, float, float, str]]]:
    items: Dict[str, List[Tuple[str, int, float, float, float, float, str]]] = {split: [] for split in target_splits}
    cls = _safe_label(placeholder_class)

    # For split=all, assign videos to train/test by ratio so both files are meaningful.
    has_train_test = len(target_splits) == 2 and set(target_splits) == {"train", "test"}
    if has_train_test:
        video_ids = sorted({video_id for (video_id, _) in frame_sizes.keys()})
        if len(video_ids) <= 1:
            train_videos = set(video_ids)
        else:
            ratio = max(0.0, min(1.0, float(train_ratio)))
            cut = int(round(len(video_ids) * ratio))
            cut = max(1, min(len(video_ids) - 1, cut))
            train_videos = set(video_ids[:cut])

        for (video_id, frame_id), (w, h) in frame_sizes.items():
            split = "train" if video_id in train_videos else "test"
            items[split].append((video_id, frame_id, 0.0, 0.0, float(max(1, w - 1)), float(max(1, h - 1)), cls))
        return items

    primary = target_splits[0]
    for (video_id, frame_id), (w, h) in frame_sizes.items():
        items[primary].append((video_id, frame_id, 0.0, 0.0, float(max(1, w - 1)), float(max(1, h - 1)), cls))
    return items


def main() -> None:
    args = parse_args()
    videos_dir = args.videos_dir.resolve()
    output_root = args.output_root.resolve()
    image_ext = args.image_ext if args.image_ext.startswith(".") else f".{args.image_ext}"
    target_splits = _resolve_target_splits(args.split)

    if not videos_dir.is_dir():
        raise FileNotFoundError(f"--videos-dir not found: {videos_dir}")
    if args.every_n < 1:
        raise ValueError("--every-n must be >= 1")
    if args.max_frames_per_video < 0:
        raise ValueError("--max-frames-per-video must be >= 0")
    if args.train_ratio < 0.0 or args.train_ratio > 1.0:
        raise ValueError("--train-ratio must be in [0, 1]")

    frames_root = output_root / "frames"
    frames_root.mkdir(parents=True, exist_ok=True)

    videos = _collect_video_files(videos_dir, recursive=args.recursive)
    if not videos:
        raise RuntimeError(f"No video files found under: {videos_dir}")

    alias_to_video: Dict[str, Optional[str]] = {}
    frame_sizes: Dict[Tuple[str, int], Tuple[int, int]] = {}
    video_default_size: Dict[str, Tuple[int, int]] = {}
    video_map: Dict[str, str] = {}

    total_saved_frames = 0
    for idx, video_path in enumerate(videos, start=1):
        video_id = _build_video_id(video_path, videos_dir)
        _register_aliases(alias_to_video, videos_dir, video_path, video_id)
        video_map[str(video_path.relative_to(videos_dir)).replace("\\", "/")] = video_id

        out_dir = frames_root / video_id
        if out_dir.exists() and args.overwrite_frames:
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        extracted = _extract_frames(
            video_path=video_path,
            output_dir=out_dir,
            image_ext=image_ext,
            jpeg_quality=args.jpeg_quality,
            every_n=args.every_n,
            target_fps=args.target_fps,
            max_frames_per_video=args.max_frames_per_video,
        )
        if not extracted:
            print(f"[warn] no frames extracted from {video_path.name}; skipping.")
            continue

        for frame_id, size in extracted.items():
            frame_sizes[(video_id, frame_id)] = size
        first_size = extracted[min(extracted.keys())]
        video_default_size[video_id] = first_size
        total_saved_frames += len(extracted)
        print(f"[{idx}/{len(videos)}] {video_path.name} -> {video_id} ({len(extracted)} frames)")

    if total_saved_frames == 0:
        raise RuntimeError("No frames were extracted from any video.")

    annotations_by_split: Dict[str, List[Tuple[str, int, float, float, float, float, str]]] = {
        split: [] for split in target_splits
    }
    used_placeholders = False

    if args.annotation_json is not None:
        annotation_spec = str(args.annotation_json)
        if "{split}" in annotation_spec:
            for split in target_splits:
                json_path = Path(annotation_spec.format(split=split)).resolve()
                if not json_path.exists():
                    raise FileNotFoundError(f"Annotation file not found for split '{split}': {json_path}")
                records, _ = _load_json_records(json_path)
                items = _build_annotations_from_json(
                    records=records,
                    split=split,
                    alias_to_video=alias_to_video,
                    frame_sizes=frame_sizes,
                    video_default_size=video_default_size,
                    default_bbox_format=args.bbox_format,
                )
                if not items:
                    raise RuntimeError(f"No valid annotations found for split '{split}' in {json_path}")
                annotations_by_split[split] = items
        else:
            json_path = Path(annotation_spec).resolve()
            if not json_path.exists():
                raise FileNotFoundError(f"--annotation-json not found: {json_path}")
            records, _ = _load_json_records(json_path)
            if len(target_splits) > 1 and not _records_have_split(records):
                raise RuntimeError(
                    "Multiple splits requested but JSON has no split field. "
                    "Use --annotation-json with '{split}' in the path or add split labels in records."
                )
            for split in target_splits:
                items = _build_annotations_from_json(
                    records=records,
                    split=split,
                    alias_to_video=alias_to_video,
                    frame_sizes=frame_sizes,
                    video_default_size=video_default_size,
                    default_bbox_format=args.bbox_format,
                )
                if not items:
                    raise RuntimeError(f"No valid annotations found for split '{split}' in {json_path}")
                annotations_by_split[split] = items
    else:
        if args.no_placeholders:
            raise RuntimeError("No annotation source found. Provide --annotation-json or allow placeholders.")
        used_placeholders = True
        annotations_by_split = _build_placeholder_annotations(
            frame_sizes=frame_sizes,
            target_splits=target_splits,
            placeholder_class=args.placeholder_class,
            train_ratio=args.train_ratio,
        )

    for split in target_splits:
        output_txt = output_root / f"{split}.txt"
        output_txt.parent.mkdir(parents=True, exist_ok=True)
        _write_split_file(output_txt, annotations_by_split.get(split, []))
        print(f"[done] wrote split file: {output_txt} ({len(annotations_by_split.get(split, []))} lines)")

    annotation_dir = output_root / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    map_path = annotation_dir / "video_id_map.json"
    with map_path.open("w", encoding="utf-8") as f:
        json.dump(video_map, f, indent=2)

    print(f"[done] wrote frames to: {frames_root}")
    print(f"[done] wrote video-id map: {map_path}")
    if used_placeholders:
        print("[warn] placeholder boxes were used (no annotation JSON provided).")
        print("[warn] replace placeholders with real boxes before real training/evaluation.")


if __name__ == "__main__":
    main()
