#!/usr/bin/env python3
"""
Generic video dataset for AeroMixer.

Expected layout (frame-based):
  <PATH_TO_DATA_DIR>/<FRAME_DIR>/<video_id>/<frame_image_files>

Annotations (either format):
1) TXT:
   - <PATH_TO_DATA_DIR>/train.txt
   - <PATH_TO_DATA_DIR>/test.txt (or val.txt)
   Each line:
     <video_id> <frame_id> <x1> <y1> <x2> <y2> <class_id_or_name>

2) JSON:
   - <PATH_TO_DATA_DIR>/annotations/<split>.json  (split in train/test/val)
   - or <PATH_TO_DATA_DIR>/<split>.json
   JSON can be either a list of records or {"annotations": [...]}.
   Each record should include:
     video/video_id/vid, frame/frame_id, bbox/box/xyxy, and label/class/category.
"""
import json
import logging
import os
import re
from collections import defaultdict

import numpy as np
import torch

import alphaction.dataset.datasets.utils as utils
from alphaction.dataset.datasets.cv2_transform import PreprocessWithBoxes

logger = logging.getLogger(__name__)
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class VideoDataset(torch.utils.data.Dataset):
    """Generic frame-based video dataset."""

    def __init__(self, cfg, split):
        self.cfg = cfg
        self._split = split

        self._sample_rate = int(cfg.DATA.SAMPLING_RATE)
        self._video_length = int(cfg.DATA.NUM_FRAMES)
        if self._sample_rate < 1 or self._video_length < 1:
            raise ValueError("DATA.SAMPLING_RATE and DATA.NUM_FRAMES must be >= 1 for VideoDataset.")

        self.preprocess_with_box = PreprocessWithBoxes(split, cfg.DATA, cfg.IMAGES)

        self.multilabel_action = cfg.MODEL.MULTI_LABEL_ACTION
        self.test_iou_thresh = cfg.TEST.IOU_THRESH
        self.independent_eval = cfg.TEST.INDEPENDENT_EVAL
        self.open_vocabulary = cfg.DATA.OPEN_VOCABULARY
        self.eval_open = cfg.TEST.EVAL_OPEN
        self.use_prior_map = False
        self.prior_boxes_init = getattr(cfg.MODEL, "PRIOR_BOXES_INIT", "")
        self.prior_boxes_test = getattr(cfg.TEST, "PRIOR_BOX_TEST", False)
        self.prior_map = None
        self.text_input = None
        self.vocabulary = {"closed": [], "open": []}

        self.data_dir = cfg.DATA.PATH_TO_DATA_DIR
        frame_dir = getattr(cfg.DATA, "FRAME_DIR", "")
        self.frame_root = os.path.join(self.data_dir, frame_dir) if frame_dir else self.data_dir
        if not os.path.isdir(self.frame_root):
            raise AssertionError(f"Frame directory not found: {self.frame_root}")

        self.video_index = self._index_frame_folders(self.frame_root)
        if len(self.video_index) == 0:
            raise AssertionError(f"No videos (frame folders) found under: {self.frame_root}")

        self.samples, self.class_names = self._load_annotations(split)
        if len(self.samples) == 0:
            raise AssertionError(f"No annotated samples found for split '{split}' in {self.data_dir}")

        self.num_classes = len(self.class_names) if len(self.class_names) > 0 else 1
        if len(self.class_names) == 0:
            self.class_names = ["class_0"]
        self.closed_set_classes = list(self.class_names)
        self.vocabulary["closed"] = list(self.class_names)
        self.vocabulary["open"] = list(self.class_names)

        # Compatibility mappings for open-vocab utilities.
        self.open_to_closed = {i: i for i in range(self.num_classes)}
        self.closed_to_open = {i: i for i in range(self.num_classes)}
        self.open_to_unseen = {}

        if self.open_vocabulary:
            self.text_input = self._build_text_input()

        logger.info(
            "Loaded VideoDataset(split=%s) with %d samples from %d videos and %d classes.",
            split,
            len(self.samples),
            len(self.video_index),
            self.num_classes,
        )

    def __len__(self):
        return len(self.samples)

    @property
    def num_images(self):
        return len(self.samples)

    @property
    def num_videos(self):
        return len(self.video_index)

    def get_sample_info(self, index):
        sample = self.samples[index]
        video_info = self.video_index[sample["video_id"]]
        height, width = video_info["resolution"]
        return {
            "image_id": f"{sample['video_id']},{int(sample['frame_id']):06d}",
            "sample_id": index,
            "video_id": sample["video_id"],
            "frame_id": int(sample["frame_id"]),
            "boxes": sample["boxes"],
            "labels": sample["labels"],
            "height": int(height),
            "width": int(width),
            "resolution": (int(height), int(width)),
        }

    def get_video_info(self, index):
        return self.get_sample_info(index)

    def get_image_info(self, index):
        return self.get_sample_info(index)

    def __getitem__(self, index):
        sample = self.samples[index]
        video = self.video_index[sample["video_id"]]

        seq_positions = self._get_sequence_positions(sample["center_pos"], len(video["paths"]))
        image_paths = [video["paths"][pos] for pos in seq_positions]
        imgs = utils.retry_load_images(image_paths, backend="cv2")

        raw_h, raw_w = video["resolution"]

        boxes = None
        if self._split == "train" and sample["boxes"].shape[0] > 0:
            boxes = sample["boxes"].copy()
            boxes[:, [0, 2]] /= float(raw_w)
            boxes[:, [1, 3]] /= float(raw_h)

        imgs_proc, boxes_proc = self.preprocess_with_box.process(imgs, boxes=boxes)

        pathways = self.cfg.MODEL.BACKBONE.PATHWAYS
        imgs_packed = utils.pack_pathway_output(self.cfg, imgs_proc, pathways=pathways)
        if pathways == 1:
            slow, fast = imgs_packed[0], None
        else:
            slow, fast = imgs_packed[0], imgs_packed[1]

        h, w = slow.shape[-2:]
        whwh = torch.tensor([w, h, w, h], dtype=torch.float32)

        label_arrs = None
        if self._split == "train":
            num_boxes = len(sample["labels"])
            label_arrs = np.zeros((num_boxes, self.num_classes), dtype=np.int32)
            if num_boxes > 0:
                label_arrs[np.arange(num_boxes), sample["labels"]] = 1

        extras = {
            "extra_boxes": None,
            "video_id": sample["video_id"],
            "frame_id": int(sample["frame_id"]),
        }
        boxes_out = boxes_proc if boxes_proc is None else boxes_proc.astype(np.float32)

        return slow, fast, whwh, boxes_out, label_arrs, extras, index

    def _build_text_input(self):
        closed_source, open_source = self._load_vocab_sources()
        closed_text = self._materialize_text_data(closed_source, allow_extra=False)
        if open_source is None:
            open_text = self._copy_text_data(closed_text)
        else:
            open_text = self._materialize_text_data(open_source, allow_extra=True)

        self.vocabulary["closed"] = list(closed_text.keys())
        self.vocabulary["open"] = list(open_text.keys())
        return {"closed": closed_text, "open": open_text}

    def _get_vocab_cfg_value(self, key):
        data_value = str(getattr(self.cfg.DATA, key, "")).strip()
        if data_value:
            return data_value
        return str(getattr(self.cfg.IMAGES, key, "")).strip()

    def _load_vocab_sources(self):
        closed_source = {}
        open_source = None

        vocab_file = self._get_vocab_cfg_value("VOCAB_FILE")
        if vocab_file:
            vocab_path = self._resolve_vocab_path(vocab_file)
            vocab_data = self._read_vocab_data(vocab_path)
            closed_data, open_data = self._extract_vocab_sections(vocab_data)
            closed_source = self._normalize_vocab_entries(closed_data)
            if open_data is not None:
                open_source = self._normalize_vocab_entries(open_data)

        vocab_open_file = self._get_vocab_cfg_value("VOCAB_OPEN_FILE")
        if vocab_open_file:
            vocab_open_path = self._resolve_vocab_path(vocab_open_file)
            open_source = self._normalize_vocab_entries(self._read_vocab_data(vocab_open_path))

        return closed_source, open_source

    def _resolve_vocab_path(self, vocab_file):
        if os.path.isabs(vocab_file):
            vocab_path = vocab_file
        else:
            vocab_path = os.path.join(self.data_dir, vocab_file)
        if not os.path.exists(vocab_path):
            raise AssertionError(f"Vocabulary file not found: {vocab_path}")
        return vocab_path

    def _read_vocab_data(self, vocab_path):
        ext = os.path.splitext(vocab_path)[1].lower()
        if ext == ".json":
            with open(vocab_path, "r", encoding="utf-8") as f:
                return json.load(f)

        data = {}
        with open(vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "\t" in line:
                    key, caption = line.split("\t", 1)
                elif ":" in line:
                    key, caption = line.split(":", 1)
                else:
                    key, caption = line, line
                data[key.strip()] = caption.strip()
        return data

    def _extract_vocab_sections(self, vocab_data):
        if vocab_data is None:
            return {}, None

        if isinstance(vocab_data, dict):
            closed_data = self._find_section_in_json(vocab_data, ["closed", "known", "seen", "base", "train"])
            open_data = self._find_section_in_json(vocab_data, ["open", "novel", "unseen", "all", "eval", "test"])

            if closed_data is None:
                closed_data = self._find_section_in_json(vocab_data, ["classes", "labels", "categories", "names", "vocabulary", "vocab"])

            if closed_data is not None or open_data is not None:
                return (closed_data if closed_data is not None else {}), open_data

        return vocab_data, None

    def _find_section_in_json(self, node, aliases, depth=0):
        if depth > 8:
            return None

        alias_set = {str(alias).strip().lower().replace("-", "_").replace(" ", "_") for alias in aliases}

        if isinstance(node, dict):
            normalized_items = []
            for raw_key, raw_value in node.items():
                key = str(raw_key).strip().lower().replace("-", "_").replace(" ", "_")
                normalized_items.append((key, raw_value))
            for key, value in normalized_items:
                if key in alias_set:
                    return value
            for _, value in normalized_items:
                found = self._find_section_in_json(value, aliases, depth=depth + 1)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = self._find_section_in_json(item, aliases, depth=depth + 1)
                if found is not None:
                    return found

        return None

    def _normalize_vocab_entries(self, data):
        entries = {}
        if data is None:
            return entries

        if isinstance(data, dict):
            for key, value in data.items():
                key_str = str(key).strip()
                if not key_str:
                    continue
                caption = value
                if isinstance(value, dict):
                    caption = value.get("caption", value.get("text", value.get("prompt", key_str)))
                entries[key_str] = {"caption": self._normalize_caption(caption, default_text=key_str)}
            return entries

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    key = item.get("name", item.get("label", item.get("class", item.get("id", item.get("class_id", item.get("category_id", None))))))
                    if key is None:
                        continue
                    key_str = str(key).strip()
                    if not key_str:
                        continue
                    caption = item.get("caption", item.get("text", item.get("prompt", item.get("description", key_str))))
                    entries[key_str] = {"caption": self._normalize_caption(caption, default_text=key_str)}
                else:
                    label = str(item).strip()
                    if label:
                        entries[label] = {"caption": label}
            return entries

        text = str(data).strip()
        if text:
            entries[text] = {"caption": text}
        return entries

    def _normalize_caption(self, caption, default_text):
        if isinstance(caption, dict):
            if "caption" in caption:
                return self._normalize_caption(caption["caption"], default_text)
            if "text" in caption:
                return self._normalize_caption(caption["text"], default_text)
            if "prompt" in caption:
                return self._normalize_caption(caption["prompt"], default_text)
            if len(caption) > 0:
                return self._normalize_caption(next(iter(caption.values())), default_text)
            return str(default_text)
        if isinstance(caption, list):
            cleaned = [str(item).strip() for item in caption if str(item).strip()]
            return cleaned if len(cleaned) > 0 else str(default_text)
        if caption is None:
            return str(default_text)
        text = str(caption).strip()
        return text if text else str(default_text)

    def _materialize_text_data(self, source_entries, allow_extra=False):
        data = {}
        consumed_keys = set()

        source_keys_lut = {str(key): key for key in source_entries.keys()}
        source_keys_lut_lower = {str(key).lower(): key for key in source_entries.keys()}

        for idx, class_name in enumerate(self.class_names):
            candidates = [class_name, str(idx)]
            matched_key = None
            for candidate in candidates:
                candidate_str = str(candidate)
                if candidate_str in source_keys_lut:
                    matched_key = source_keys_lut[candidate_str]
                    break
                candidate_lower = candidate_str.lower()
                if candidate_lower in source_keys_lut_lower:
                    matched_key = source_keys_lut_lower[candidate_lower]
                    break

            if matched_key is None:
                caption = class_name
            else:
                consumed_keys.add(matched_key)
                caption = source_entries[matched_key]["caption"]

            data[class_name] = {"caption": caption}

        if allow_extra:
            for key, value in source_entries.items():
                if key in consumed_keys:
                    continue
                extra_name = self._normalize_open_vocab_name(key)
                if extra_name in data:
                    continue
                data[extra_name] = {"caption": value["caption"]}

        return data

    def _normalize_open_vocab_name(self, key):
        key_str = str(key).strip()
        if key_str.isdigit():
            idx = int(key_str)
            if 0 <= idx < len(self.class_names):
                return self.class_names[idx]
            return f"class_{idx}"
        return key_str

    def _copy_text_data(self, text_data):
        copied = {}
        for key, value in text_data.items():
            caption = value.get("caption", key)
            if isinstance(caption, list):
                copied_caption = list(caption)
            else:
                copied_caption = caption
            copied[key] = {"caption": copied_caption}
        return copied

    def _index_frame_folders(self, frame_root):
        video_index = {}

        for root, _, files in os.walk(frame_root):
            image_files = [
                f for f in sorted(files)
                if os.path.splitext(f)[1].lower() in VALID_IMAGE_EXTS
            ]
            if len(image_files) == 0:
                continue

            video_id = os.path.relpath(root, frame_root).replace("\\", "/")
            if video_id == ".":
                video_id = "_default"

            frame_items = []
            used_ids = set()
            for order, filename in enumerate(image_files):
                stem = os.path.splitext(filename)[0]
                numbers = re.findall(r"\d+", stem)
                if len(numbers) > 0:
                    frame_id = int(numbers[-1])
                else:
                    frame_id = order + 1
                while frame_id in used_ids:
                    frame_id += 1
                used_ids.add(frame_id)
                frame_items.append((frame_id, os.path.join(root, filename)))

            frame_items.sort(key=lambda x: x[0])
            frame_nums = [item[0] for item in frame_items]
            frame_paths = [item[1] for item in frame_items]

            first_frame = utils.retry_load_images([frame_paths[0]], backend="cv2")[0]
            resolution = tuple(first_frame.shape[:2])  # (H, W)

            video_index[video_id] = {
                "frame_nums": frame_nums,
                "paths": frame_paths,
                "resolution": resolution,
            }

        return video_index

    def _load_annotations(self, split):
        # Priority: split txt -> split json.
        txt_candidates = [
            os.path.join(self.data_dir, f"{split}.txt"),
            os.path.join(self.data_dir, "annotations", f"{split}.txt"),
        ]
        for ann_file in txt_candidates:
            if os.path.exists(ann_file):
                return self._load_txt_annotations(ann_file)

        json_candidates = [
            os.path.join(self.data_dir, "annotations", f"{split}.json"),
            os.path.join(self.data_dir, f"{split}.json"),
        ]
        for ann_file in json_candidates:
            if os.path.exists(ann_file):
                return self._load_json_annotations(ann_file, split)

        raise AssertionError(
            f"No annotation file found for split '{split}'. "
            "Expected train/test txt or json under PATH_TO_DATA_DIR."
        )

    def _resolve_frame_position(self, video_id, frame_id):
        frame_nums = self.video_index[video_id]["frame_nums"]
        frame_arr = np.array(frame_nums, dtype=np.int64)
        target = int(frame_id)
        pos = int(np.argmin(np.abs(frame_arr - target)))
        resolved_frame = int(frame_arr[pos])
        return pos, resolved_frame

    def _build_samples(self, grouped_records, class_names):
        samples = []
        skipped = 0
        for (video_id, frame_id), rec in grouped_records.items():
            if video_id not in self.video_index:
                skipped += 1
                continue

            center_pos, resolved_frame = self._resolve_frame_position(video_id, frame_id)
            boxes = np.array(rec["boxes"], dtype=np.float32) if len(rec["boxes"]) else np.zeros((0, 4), dtype=np.float32)
            labels = np.array(rec["labels"], dtype=np.int64) if len(rec["labels"]) else np.zeros((0,), dtype=np.int64)
            samples.append(
                {
                    "video_id": video_id,
                    "frame_id": resolved_frame,
                    "center_pos": center_pos,
                    "boxes": boxes,
                    "labels": labels,
                }
            )

        samples.sort(key=lambda s: (s["video_id"], s["frame_id"]))
        if skipped > 0:
            logger.warning("Skipped %d annotations because their video_id was not found in FRAME_DIR.", skipped)

        if len(class_names) == 0:
            class_names = ["class_0"]

        return samples, class_names

    def _load_txt_annotations(self, ann_file):
        grouped = defaultdict(lambda: {"boxes": [], "labels": []})
        label_to_idx = {}
        class_names = []

        def get_label_idx(label_raw):
            key = str(label_raw).strip()
            if key not in label_to_idx:
                label_to_idx[key] = len(label_to_idx)
                if key.isdigit():
                    class_names.append(f"class_{key}")
                else:
                    class_names.append(key)
            return label_to_idx[key]

        with open(ann_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 7:
                    raise ValueError(f"Bad annotation line in {ann_file}: {line}")
                video_id = parts[0].replace("\\", "/")
                frame_id = int(float(parts[1]))
                x1, y1, x2, y2 = map(float, parts[2:6])
                label_idx = get_label_idx(parts[6])

                grouped[(video_id, frame_id)]["boxes"].append([x1, y1, x2, y2])
                grouped[(video_id, frame_id)]["labels"].append(label_idx)

        return self._build_samples(grouped, class_names)

    def _load_json_annotations(self, ann_file, split):
        with open(ann_file, "r") as f:
            data = json.load(f)

        annotations = data["annotations"] if isinstance(data, dict) and "annotations" in data else data
        if not isinstance(annotations, list):
            raise ValueError(f"Unsupported JSON annotation format in {ann_file}")

        grouped = defaultdict(lambda: {"boxes": [], "labels": []})
        label_to_idx = {}
        class_names = []

        category_name_by_id = {}
        if isinstance(data, dict) and isinstance(data.get("categories"), list):
            for cat in data["categories"]:
                cat_id = str(cat.get("id", ""))
                cat_name = str(cat.get("name", cat_id))
                if cat_id:
                    category_name_by_id[cat_id] = cat_name

        def get_label_idx(label_raw):
            key = str(label_raw).strip()
            if key not in label_to_idx:
                label_to_idx[key] = len(label_to_idx)
                class_names.append(category_name_by_id.get(key, key if not key.isdigit() else f"class_{key}"))
            return label_to_idx[key]

        for rec in annotations:
            if not isinstance(rec, dict):
                continue
            rec_split = str(rec.get("split", "")).lower().strip()
            if rec_split and rec_split not in [split.lower(), "val" if split == "test" else split.lower()]:
                continue

            video_id = rec.get("video") or rec.get("video_id") or rec.get("vid")
            frame_id = rec.get("frame") if rec.get("frame") is not None else rec.get("frame_id")
            bbox = rec.get("bbox") or rec.get("box") or rec.get("xyxy")
            label_raw = (
                rec.get("label")
                if rec.get("label") is not None
                else rec.get("class_id")
                if rec.get("class_id") is not None
                else rec.get("category_id")
                if rec.get("category_id") is not None
                else rec.get("class")
            )

            if video_id is None or frame_id is None or bbox is None or label_raw is None:
                continue
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue

            video_id = str(video_id).replace("\\", "/")
            frame_id = int(float(frame_id))
            x1, y1, x2, y2 = map(float, bbox)
            label_idx = get_label_idx(label_raw)

            grouped[(video_id, frame_id)]["boxes"].append([x1, y1, x2, y2])
            grouped[(video_id, frame_id)]["labels"].append(label_idx)

        return self._build_samples(grouped, class_names)

    def _get_sequence_positions(self, center_pos, num_frames):
        # Build a fixed-length temporal clip around center_pos with boundary clamping.
        half = (self._video_length // 2) * self._sample_rate
        start = center_pos - half
        seq = []
        for i in range(self._video_length):
            pos = start + i * self._sample_rate
            pos = max(0, min(num_frames - 1, pos))
            seq.append(pos)
        return seq
