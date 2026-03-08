#!/usr/bin/env python3
"""
Image-only dataset for AeroMixer.
Supports multiple detection annotation formats:
1) Plain text split files:
    <image_rel_path> x1 y1 x2 y2 class_id [severity]
   with files at <PATH_TO_DATA_DIR>/{train,test}.txt (or annotations.txt).
2) YOLOv5/YOLOv8-style labels:
   <PATH_TO_DATA_DIR>/images/{train,val}/...
   <PATH_TO_DATA_DIR>/labels/{train,val}/...
   with label lines:
    class_id x_center y_center width height
3) COCO JSON:
   <PATH_TO_DATA_DIR>/annotations/instances_{train|val}.json (or similar)
4) Pascal VOC:
   <VOC_ROOT>/Annotations/*.xml, <VOC_ROOT>/JPEGImages/*, and ImageSets/Main/*.txt
5) Custom nested JSON:
   {
     "images": [
       {
         "image_id": "...",
         "file_name": "...jpg",
         "split": "train|valid|test",
         "annotations": [
           {
             "category_id": 1,
             "category_name": "dent",
             "bounding_box_normalized": {"x_center": ..., "y_center": ..., "width": ..., "height": ...},
             "damage_metrics": {"raw_severity_score": ...},
             "risk_assessment": {...},
             "description": "..."
           }
         ]
       }
     ]
   }

Multimodal (Image + Text) Support:
- Open vocabulary detection with text prompts
- Text features from CLIP or other vision-language models
- Text-aware evaluation metrics

Returned samples keep the same tensor structure expected by the rest of the pipeline:
    (primary_input, secondary_input, whwh, boxes, label_arrs, extras, index)
"""
import os
import json
import logging
import re
import numpy as np
import torch
import xml.etree.ElementTree as ET
from collections import defaultdict

import alphaction.dataset.datasets.utils as utils
from alphaction.dataset.datasets.cv2_transform import PreprocessWithBoxes

logger = logging.getLogger(__name__)
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_TILE_KEY_RE = re.compile(r"^(?P<base>.+)__x(?P<x>\d+)_y(?P<y>\d+)$")


class ImageDataset(torch.utils.data.Dataset):
    """Pure image dataset with an image-first sample interface.
    
    Supports multimodal (image + text) mode for open vocabulary detection.
    """

    def __init__(self, cfg, split):
        self.cfg = cfg
        self._split = split
        self.annotation_format = str(getattr(cfg.DATA, "ANNOTATION_FORMAT", "auto")).lower()
        self.input_type = str(getattr(cfg.DATA, "INPUT_TYPE", "image")).lower()
        self.open_vocabulary = cfg.DATA.OPEN_VOCABULARY
        self.eval_open = cfg.TEST.EVAL_OPEN
        self.independent_eval = cfg.TEST.INDEPENDENT_EVAL
        self.multilabel_action = cfg.MODEL.MULTI_LABEL_ACTION
        self.test_iou_thresh = cfg.TEST.IOU_THRESH
        self.use_prior_map = False
        self.prior_boxes_init = getattr(cfg.MODEL, "PRIOR_BOXES_INIT", "")
        self.prior_boxes_test = getattr(cfg.TEST, "PRIOR_BOX_TEST", False)
        self.prior_map = None
        self.text_input = None
        self.vocabulary = {"closed": [], "open": []}
        self._shape_cache = {}
        
        # Multimodal support
        self.multimodal = getattr(cfg.DATA, "MULTIMODAL", False)
        self.text_features = None
        self.text_prompts = None

        if cfg.DATA.NUM_FRAMES != 1 or cfg.DATA.SAMPLING_RATE != 1:
            raise ValueError(
                "ImageDataset expects DATA.NUM_FRAMES=1 and DATA.SAMPLING_RATE=1."
            )

        self.preprocess_with_box = PreprocessWithBoxes(split, cfg.DATA, cfg.IMAGES)

        self.data_dir = cfg.DATA.PATH_TO_DATA_DIR
        frame_dir = getattr(cfg.DATA, "FRAME_DIR", "")
        self.image_dir = os.path.join(self.data_dir, frame_dir) if frame_dir else self.data_dir
        if not os.path.isdir(self.image_dir):
            fallback_image_dir = os.path.join(self.data_dir, "images")
            if os.path.isdir(fallback_image_dir):
                self.image_dir = fallback_image_dir
            elif os.path.isdir(self.data_dir):
                self.image_dir = self.data_dir
            else:
                raise AssertionError(f"Image dir not found: {self.image_dir}")

        self.annotation_mode, samples, classes_set, class_name_map, sample_extras_map = self._load_annotations(split)
        self.image_list = sorted(list(samples.keys()))
        self.annos = {}
        for image_rel_path, entries in samples.items():
            if len(entries) == 0:
                self.annos[image_rel_path] = np.zeros((0, 5), dtype=np.float32)
            else:
                self.annos[image_rel_path] = np.array(entries, dtype=np.float32)

        self.classes = sorted(list(classes_set))
        self.class_id_to_idx = {class_id: idx for idx, class_id in enumerate(self.classes)}
        self.idx_to_class_id = {idx: class_id for class_id, idx in self.class_id_to_idx.items()}
        self.num_classes = len(self.classes) if len(self.classes) > 0 else 1
        if len(self.classes) > 0:
            self.class_names = [
                class_name_map.get(class_id, f"class_{class_id}") for class_id in self.classes
            ]
        else:
            self.class_names = ["class_0"]
        self.closed_set_classes = self.class_names
        self.vocabulary["closed"] = list(self.class_names)
        self.vocabulary["open"] = list(self.class_names)
        if self.open_vocabulary or self.multimodal:
            self.text_input = self._build_text_input()
            # Generate text prompts for each class
            self.text_prompts = self._generate_text_prompts()

        self.tile_metadata = self._build_tile_metadata_map(self.image_list)
        self.samples = []
        for img_rel in self.image_list:
            ann = self.annos[img_rel]
            boxes = ann[:, :4].astype(np.float32) if len(ann) else np.zeros((0, 4), dtype=np.float32)
            if len(ann):
                labels = np.array(
                    [self.class_id_to_idx[int(class_id)] for class_id in ann[:, 4]],
                    dtype=np.int64,
                )
            else:
                labels = np.zeros((0,), dtype=np.int64)
            severities = None
            if len(ann) and ann.shape[1] > 5:
                severities = ann[:, 5].astype(np.float32)
            sample_extras = sample_extras_map.get(img_rel, None) if isinstance(sample_extras_map, dict) else None
            self.samples.append(
                {
                    "image_rel": img_rel,
                    "boxes": boxes,
                    "labels": labels,
                    "severity": severities,
                    "annotation_extras": sample_extras,
                    "tile_meta": self.tile_metadata.get(img_rel),
                }
            )

        logger.info(
            f"Loaded ImageDataset(split={split}, mode={self.annotation_mode}, multimodal={self.multimodal}) with {len(self.samples)} images and "
            f"{sum(len(s['boxes']) for s in self.samples)} total boxes"
        )

    def __len__(self):
        return len(self.samples)

    @property
    def num_images(self):
        return len(self.samples)

    @property
    def num_videos(self):
        # Keep this property for compatibility with shared trainer/evaluator utilities.
        return self.num_images

    def set_multimodal(self, multimodal):
        """Enable or disable multimodal mode.
        
        Args:
            multimodal: Boolean to enable/disable multimodal mode
        """
        self.multimodal = multimodal
        if multimodal and self.text_input is None:
            self.text_input = self._build_text_input()
            self.text_prompts = self._generate_text_prompts()

    def set_text_features(self, text_features):
        """Set text features for multimodal detection.
        
        Args:
            text_features: Numpy array of text features [num_classes, feature_dim]
        """
        self.text_features = text_features

    def get_text_prompts(self, index=None):
        """Get text prompts for the dataset or a specific sample.
        
        Args:
            index: Optional sample index. If None, returns prompts for all classes.
            
        Returns:
            List of text prompts
        """
        if self.text_prompts is None:
            return None
            
        if index is None:
            return self.text_prompts
        
        # Return prompts for a specific sample's labels
        sample = self.samples[index]
        labels = sample["labels"]
        return [self.text_prompts[label] for label in labels]

    def get_text_features(self, index=None):
        """Get text features for the dataset or a specific sample.
        
        Args:
            index: Optional sample index. If None, returns features for all classes.
            
        Returns:
            Numpy array of text features or None if not available
        """
        if self.text_features is None:
            return None
            
        if index is None:
            return self.text_features
        
        # Return features for a specific sample's labels
        sample = self.samples[index]
        labels = sample["labels"]
        return self.text_features[labels]

    def _generate_text_prompts(self):
        """Generate text prompts for each class.
        
        Returns:
            List of text prompts for each class
        """
        prompts = []
        for class_name in self.class_names:
            prompt = self._format_prompt(class_name)
            prompts.append(prompt)
        return prompts

    def _format_prompt(self, class_name):
        template = str(getattr(self.cfg.DATA.TEXT, "PROMPT_TEMPLATE", "a photo of {}")).strip()
        if "{}" in template:
            return template.format(class_name)
        return f"{template} {class_name}".strip()

    def get_sample_info(self, index):
        sample = self.samples[index]
        img_rel = sample["image_rel"]
        height, width = self._get_image_shape(img_rel)
        result = dict(
            image_id=img_rel,
            sample_id=index,
            boxes=sample["boxes"],
            labels=sample["labels"],
            height=height,
            width=width,
            resolution=(height, width),
        )
        if sample.get("severity", None) is not None:
            result["severity"] = sample["severity"]
        if sample.get("annotation_extras", None) is not None:
            result["annotation_extras"] = sample["annotation_extras"]
        if sample.get("tile_meta", None) is not None:
            result["tile_meta"] = sample["tile_meta"]
        
        # Add multimodal information if enabled
        if self.multimodal:
            result["text_prompts"] = self.get_text_prompts(index)
            result["text_features"] = self.get_text_features(index)
        
        return result

    def get_video_info(self, index):
        # Backward-compatible alias used by legacy code paths.
        return self.get_sample_info(index)

    def get_image_info(self, index):
        return self.get_sample_info(index)

    def __getitem__(self, index):
        sample = self.samples[index]
        img_rel = sample["image_rel"]
        img_path = os.path.join(self.image_dir, img_rel)
        imgs = utils.retry_load_images([img_path], backend="cv2")
        raw_resolution = imgs[0].shape[:2]
        self._shape_cache.setdefault(img_rel, raw_resolution)

        boxes = None
        box_labels = sample["labels"]
        if self._split == "train" and sample["boxes"].shape[0] > 0:
            boxes = sample["boxes"].copy()
            boxes[:, [0, 2]] /= float(raw_resolution[1])
            boxes[:, [1, 3]] /= float(raw_resolution[0])

        # preprocess images with boxes and keep mask so labels remain aligned with filtered boxes.
        imgs_proc, boxes_proc, keep_mask = self.preprocess_with_box.process(
            imgs, boxes=boxes, return_keep=True
        )

        if self._split == "train" and keep_mask is not None:
            keep_mask = np.asarray(keep_mask, dtype=bool).reshape(-1)
            if len(box_labels) == keep_mask.shape[0]:
                box_labels = box_labels[keep_mask]
            elif keep_mask.shape[0] > 0:
                # Last-resort safeguard for unexpected shape mismatches.
                box_labels = box_labels[: int(keep_mask.sum())]

            if sample.get("severity", None) is not None:
                severity_vals = sample["severity"]
                if len(severity_vals) == keep_mask.shape[0]:
                    severity_vals = severity_vals[keep_mask]
                else:
                    severity_vals = severity_vals[: int(keep_mask.sum())]
            else:
                severity_vals = None

            annotation_extras = sample.get("annotation_extras", None)
            if isinstance(annotation_extras, list) and len(annotation_extras) == keep_mask.shape[0]:
                annotation_extras = [item for item, k in zip(annotation_extras, keep_mask.tolist()) if k]
        else:
            severity_vals = sample.get("severity", None)
            annotation_extras = sample.get("annotation_extras", None)

        pathways = self.cfg.MODEL.BACKBONE.PATHWAYS
        imgs_packed = utils.pack_pathway_output(self.cfg, imgs_proc, pathways=pathways)
        primary_input, secondary_input = (
            (imgs_packed[0], None) if pathways == 1 else (imgs_packed[0], imgs_packed[1])
        )

        h, w = primary_input.shape[-2:]
        whwh = torch.tensor([w, h, w, h], dtype=torch.float32)

        label_arrs = None
        if self._split == "train":
           # IMPORTANT: STM expects class indices NOT one-hot
            if len(box_labels) == 0:
                label_arrs = np.zeros((0,), dtype=np.int64)
            else:
                label_arrs = box_labels.astype(np.int64)


        extras = {"extra_boxes": None, "image_rel": img_rel, "sample_id": index}
        if severity_vals is not None:
            extras["severity"] = np.asarray(severity_vals, dtype=np.float32)
        if annotation_extras is not None:
            extras["annotation_extras"] = annotation_extras
        if sample.get("tile_meta", None) is not None:
            extras["tile_meta"] = sample["tile_meta"]

        boxes_out = boxes_proc if boxes_proc is None else boxes_proc.astype(np.float32)
        
        # Add multimodal extras if enabled
        if self.multimodal:
            extras["text_prompts"] = self.get_text_prompts(index)
            if self.text_features is not None:
                extras["text_features"] = self.get_text_features(index)

        return primary_input, secondary_input, whwh, boxes_out, label_arrs, extras, index

    def _get_image_shape(self, image_rel_path):
        if image_rel_path not in self._shape_cache:
            img_path = os.path.join(self.image_dir, image_rel_path)
            assert os.path.exists(img_path), f"Image not found: {img_path}"
            im = utils.retry_load_images([img_path], backend="cv2")[0]
            self._shape_cache[image_rel_path] = im.shape[:2]
        return self._shape_cache[image_rel_path]

    def _parse_tiled_key(self, image_rel_path):
        stem = os.path.splitext(str(image_rel_path))[0]
        match = _TILE_KEY_RE.match(stem)
        if match is None:
            return None
        return match.group("base"), int(match.group("x")), int(match.group("y"))

    def _build_tile_metadata_map(self, image_list):
        groups = defaultdict(list)
        for img_rel in image_list:
            parsed = self._parse_tiled_key(img_rel)
            if parsed is None:
                continue
            base_key, x_offset, y_offset = parsed
            height, width = self._get_image_shape(img_rel)
            groups[base_key].append(
                {
                    "image_rel": img_rel,
                    "x": int(x_offset),
                    "y": int(y_offset),
                    "w": int(width),
                    "h": int(height),
                }
            )

        metadata = {}
        for base_key, entries in groups.items():
            if not entries:
                continue

            full_width = max(int(item["x"]) + int(item["w"]) for item in entries)
            full_height = max(int(item["y"]) + int(item["h"]) for item in entries)
            full_width = max(full_width, 1)
            full_height = max(full_height, 1)

            x_positions = sorted({int(item["x"]) for item in entries})
            y_positions = sorted({int(item["y"]) for item in entries})
            x_lookup = {value: idx for idx, value in enumerate(x_positions)}
            y_lookup = {value: idx for idx, value in enumerate(y_positions)}
            x_den = max(len(x_positions) - 1, 1)
            y_den = max(len(y_positions) - 1, 1)

            for item in entries:
                tile_w = int(item["w"])
                tile_h = int(item["h"])
                x_offset = int(item["x"])
                y_offset = int(item["y"])
                center_x = x_offset + 0.5 * tile_w
                center_y = y_offset + 0.5 * tile_h
                left = x_offset / float(full_width)
                top = y_offset / float(full_height)
                right = max(0.0, float(full_width - (x_offset + tile_w))) / float(
                    full_width
                )
                bottom = max(0.0, float(full_height - (y_offset + tile_h))) / float(
                    full_height
                )
                coverage = (tile_w * tile_h) / float(full_width * full_height)
                edge_proximity = 1.0 - min(left, top, right, bottom, 1.0)

                metadata[item["image_rel"]] = {
                    "is_tiled": True,
                    "base_image_id": base_key,
                    "tile_offset_xy": [x_offset, y_offset],
                    "tile_size_wh": [tile_w, tile_h],
                    "full_size_wh": [full_width, full_height],
                    "position_norm": [
                        center_x / float(full_width),
                        center_y / float(full_height),
                    ],
                    "offset_norm": [left, top],
                    "size_norm": [
                        tile_w / float(full_width),
                        tile_h / float(full_height),
                    ],
                    "border_norm": [left, top, right, bottom],
                    "coverage_ratio": coverage,
                    "edge_proximity": edge_proximity,
                    "grid_position": [
                        x_lookup.get(x_offset, 0) / float(x_den),
                        y_lookup.get(y_offset, 0) / float(y_den),
                    ],
                }

        return metadata

    def _load_annotations(self, split):
        loaders = {
            "txt": lambda: self._load_txt_for_split(split),
            "yolo": lambda: self._load_yolo_annotations(split),
            "coco": lambda: self._load_coco_annotations(split),
            "voc": lambda: self._load_voc_annotations(split),
            "custom_json": lambda: self._load_custom_json_annotations(split),
        }

        if self.annotation_format in loaders:
            samples, classes_set, class_name_map, sample_extras_map = loaders[self.annotation_format]()
            return self.annotation_format, samples, classes_set, class_name_map, sample_extras_map

        if self.annotation_format != "auto":
            raise ValueError(
                f"Unsupported DATA.ANNOTATION_FORMAT='{self.annotation_format}'. "
                "Use one of: auto, txt, yolo, coco, voc, custom_json."
            )

        errors = []
        for format_name in ["txt", "yolo", "coco", "voc", "custom_json"]:
            try:
                samples, classes_set, class_name_map, sample_extras_map = loaders[format_name]()
                return format_name, samples, classes_set, class_name_map, sample_extras_map
            except Exception as exc:
                errors.append(f"{format_name}: {exc}")

        raise AssertionError(
            "Could not detect a supported annotation format in PATH_TO_DATA_DIR. "
            + " | ".join(errors)
        )

    def _load_txt_for_split(self, split):
        ann_file = os.path.join(self.data_dir, f"{split}.txt")
        if not os.path.exists(ann_file):
            ann_file = os.path.join(self.data_dir, "annotations.txt")
        if not os.path.exists(ann_file):
            raise AssertionError(
                f"TXT annotation file not found for split '{split}' in {self.data_dir}."
            )
        return self._load_txt_annotations(ann_file)

    def _load_txt_annotations(self, ann_file):
        samples = defaultdict(list)
        classes_set = set()
        with open(ann_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 6:
                    raise ValueError(f"Bad annotation line: {line}")
                img_rel = self._normalize_relpath(parts[0])
                x1, y1, x2, y2 = map(float, parts[1:5])
                cls = int(parts[5])
                severity = float(parts[6]) if len(parts) >= 7 else float("nan")
                samples[img_rel].append([x1, y1, x2, y2, cls, severity])
                classes_set.add(cls)
        if len(samples) == 0:
            raise AssertionError(f"No valid annotations found in {ann_file}")
        return samples, classes_set, {}, {}

    def _load_yolo_annotations(self, split):
        split_aliases = self._split_aliases(split)
        image_split_dir, used_split_alias = self._find_existing_split_dir(self.image_dir, split_aliases)
        if image_split_dir is None:
            raise AssertionError(
                f"YOLO image split directory not found under {self.image_dir}. Tried: {split_aliases}"
            )

        labels_root = os.path.join(self.data_dir, "labels")
        labels_scan_root = labels_root if os.path.isdir(labels_root) else self.data_dir
        class_name_map = self._load_yolo_class_names()

        label_aliases = [used_split_alias] + [alias for alias in split_aliases if alias != used_split_alias]
        label_split_dir, _ = self._find_existing_split_dir(labels_root, label_aliases)
        if label_split_dir is None:
            label_split_dir = labels_scan_root

        samples = defaultdict(list)
        classes_set = set()

        for root, _, files in os.walk(image_split_dir):
            for file_name in sorted(files):
                if os.path.splitext(file_name)[1].lower() not in VALID_IMAGE_EXTS:
                    continue

                img_path = os.path.join(root, file_name)
                img_rel = self._normalize_relpath(os.path.relpath(img_path, self.image_dir))
                rel_path_in_split = os.path.relpath(img_path, image_split_dir)
                label_path = self._resolve_yolo_label_path(image_split_dir, label_split_dir, rel_path_in_split)

                image_boxes = []
                if os.path.exists(label_path):
                    height, width = self._get_image_shape(img_rel)
                    with open(label_path, "r") as f:
                        for line_number, line in enumerate(f, 1):
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split()
                            if len(parts) < 5:
                                raise ValueError(f"Bad YOLO label at {label_path}:{line_number}")

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

                            image_boxes.append([x1, y1, x2, y2, cls_id])
                            classes_set.add(cls_id)

                if len(image_boxes) == 0 and split == "train":
                    continue

                samples[img_rel].extend(image_boxes)

        if len(samples) == 0:
            raise AssertionError(
                f"No images found for split '{split}'. Checked YOLO structure under {image_split_dir} and {label_split_dir}."
            )

        all_classes = self._collect_yolo_class_ids(labels_scan_root)
        if len(class_name_map) > 0:
            all_classes = all_classes.union(set(class_name_map.keys()))
        if len(all_classes) > 0:
            classes_set = all_classes
        return samples, classes_set, class_name_map, {}

    def _load_coco_annotations(self, split):
        ann_file = self._find_coco_annotation_file(split)
        if ann_file is None:
            raise AssertionError(f"COCO annotation JSON not found for split '{split}'.")

        with open(ann_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "images" not in data or "annotations" not in data:
            raise AssertionError(f"Not a valid COCO file: {ann_file}")

        images_by_id = {int(img["id"]): img for img in data["images"] if "id" in img}
        annos_by_image = defaultdict(list)
        classes_set = set()
        class_name_map = {}

        for category in data.get("categories", []):
            try:
                class_id = int(category["id"])
            except Exception:
                continue
            class_name_map[class_id] = str(category.get("name", f"class_{class_id}"))

        for ann in data["annotations"]:
            if ann.get("iscrowd", 0):
                continue
            if "bbox" not in ann or "category_id" not in ann or "image_id" not in ann:
                continue
            image_id = int(ann["image_id"])
            if image_id not in images_by_id:
                continue

            x, y, w, h = map(float, ann["bbox"])
            if w <= 0 or h <= 0:
                continue

            image_info = images_by_id[image_id]
            img_w = float(image_info.get("width", x + w))
            img_h = float(image_info.get("height", y + h))
            x1 = max(0.0, min(img_w - 1.0, x))
            y1 = max(0.0, min(img_h - 1.0, y))
            x2 = max(0.0, min(img_w - 1.0, x + w))
            y2 = max(0.0, min(img_h - 1.0, y + h))
            if x2 <= x1 or y2 <= y1:
                continue

            cls_id = int(ann["category_id"])
            annos_by_image[image_id].append([x1, y1, x2, y2, cls_id])
            classes_set.add(cls_id)

        candidate_roots = [
            self.image_dir,
            self.data_dir,
            os.path.join(self.data_dir, "images"),
            os.path.dirname(ann_file),
            os.path.dirname(os.path.dirname(ann_file)),
        ]

        samples = defaultdict(list)
        for image_id, image_info in images_by_id.items():
            file_name = image_info.get("file_name")
            if not file_name:
                continue
            img_rel = self._resolve_relative_image_path(str(file_name), candidate_roots)
            image_boxes = annos_by_image.get(image_id, [])
            if len(image_boxes) == 0 and self._split == "train":
                continue
            samples[img_rel].extend(image_boxes)

        if len(samples) == 0:
            raise AssertionError(f"No COCO samples found in {ann_file}")

        if len(class_name_map) > 0:
            classes_set = classes_set.union(set(class_name_map.keys()))
        return samples, classes_set, class_name_map, {}

    def _load_voc_annotations(self, split):
        voc_root = self._find_voc_root()
        if voc_root is None:
            raise AssertionError("Pascal VOC root not found.")

        annotation_dir = os.path.join(voc_root, "Annotations")
        image_dir = os.path.join(voc_root, "JPEGImages")
        image_ids = self._read_voc_split_ids(voc_root, split)
        if len(image_ids) == 0:
            raise AssertionError(f"No VOC split file found for split '{split}'.")

        raw_samples = defaultdict(list)
        class_names = set()
        for image_id in image_ids:
            xml_file = os.path.join(annotation_dir, f"{image_id}.xml")
            if not os.path.exists(xml_file):
                continue
            tree = ET.parse(xml_file)
            root = tree.getroot()

            file_name = root.findtext("filename")
            if file_name:
                image_file = os.path.join(image_dir, file_name)
            else:
                image_file = None

            if image_file is None or not os.path.exists(image_file):
                image_file = None
                for ext in VALID_IMAGE_EXTS:
                    trial = os.path.join(image_dir, f"{image_id}{ext}")
                    if os.path.exists(trial):
                        image_file = trial
                        break
            if image_file is None:
                continue

            img_rel = self._normalize_relpath(os.path.relpath(image_file, self.image_dir))
            image_boxes = []
            for obj in root.findall("object"):
                class_name = obj.findtext("name")
                if not class_name:
                    continue
                bbox = obj.find("bndbox")
                if bbox is None:
                    continue
                try:
                    x1 = float(bbox.findtext("xmin"))
                    y1 = float(bbox.findtext("ymin"))
                    x2 = float(bbox.findtext("xmax"))
                    y2 = float(bbox.findtext("ymax"))
                except Exception:
                    continue
                if x2 <= x1 or y2 <= y1:
                    continue

                image_boxes.append([x1, y1, x2, y2, class_name])
                class_names.add(class_name)

            if len(image_boxes) == 0 and self._split == "train":
                continue
            raw_samples[img_rel].extend(image_boxes)

        if len(raw_samples) == 0:
            raise AssertionError(f"No VOC samples found in {voc_root}")

        class_names_sorted = sorted(class_names)
        class_to_id = {name: idx for idx, name in enumerate(class_names_sorted)}
        class_name_map = {idx: name for name, idx in class_to_id.items()}

        samples = defaultdict(list)
        for img_rel, entries in raw_samples.items():
            for x1, y1, x2, y2, class_name in entries:
                samples[img_rel].append([x1, y1, x2, y2, class_to_id[class_name]])

        classes_set = set(class_name_map.keys())
        return samples, classes_set, class_name_map, {}

    def _load_custom_json_annotations(self, split):
        ann_file = self._find_custom_json_annotation_file(split)
        if ann_file is None:
            raise AssertionError(f"Custom JSON annotation file not found for split '{split}'.")

        with open(ann_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not self._looks_like_custom_json(data):
            raise AssertionError(f"Not a supported custom nested JSON file: {ann_file}")

        image_records = data.get("images", [])
        split_aliases = set(self._split_aliases(split))
        if split == "train":
            split_aliases = split_aliases.union({"train"})
        else:
            split_aliases = split_aliases.union({"val", "valid", "validation", "test"})

        candidate_roots = [
            self.image_dir,
            self.data_dir,
            os.path.join(self.data_dir, "images"),
            os.path.dirname(ann_file),
            os.path.dirname(os.path.dirname(ann_file)),
        ]

        samples = defaultdict(list)
        sample_extras_map = {}
        classes_set = set()
        class_name_map = {}
        class_name_to_id = {}
        next_auto_class_id = 0

        for image_rec in image_records:
            if not isinstance(image_rec, dict):
                continue

            rec_split = str(image_rec.get("split", "")).strip().lower()
            if rec_split and rec_split not in split_aliases:
                # Skip explicit non-matching split records.
                continue

            file_name = image_rec.get("file_name", None)
            if file_name is None:
                continue

            img_rel = self._resolve_relative_image_path(str(file_name), candidate_roots)
            img_h, img_w = self._get_image_shape(img_rel)
            img_w = float(img_w)
            img_h = float(img_h)

            image_boxes = []
            image_meta = []
            ann_list = image_rec.get("annotations", [])
            if not isinstance(ann_list, list):
                ann_list = []

            for ann in ann_list:
                if not isinstance(ann, dict):
                    continue

                cls_id_raw = ann.get("category_id", None)
                cls_name_raw = ann.get("category_name", None)
                cls_name = str(cls_name_raw).strip() if cls_name_raw is not None else ""

                cls_id = None
                if cls_id_raw is not None:
                    try:
                        cls_id = int(cls_id_raw)
                    except Exception:
                        cls_id = None

                if cls_id is None:
                    if cls_name:
                        if cls_name not in class_name_to_id:
                            class_name_to_id[cls_name] = next_auto_class_id
                            next_auto_class_id += 1
                        cls_id = class_name_to_id[cls_name]
                    else:
                        # Last-resort fallback class.
                        cls_id = 0

                if cls_name:
                    class_name_map[cls_id] = cls_name
                elif cls_id not in class_name_map:
                    class_name_map[cls_id] = f"class_{cls_id}"
                classes_set.add(cls_id)

                bbox_norm = ann.get("bounding_box_normalized", None)
                if not isinstance(bbox_norm, dict):
                    continue

                try:
                    x_center = float(bbox_norm.get("x_center"))
                    y_center = float(bbox_norm.get("y_center"))
                    box_w = float(bbox_norm.get("width"))
                    box_h = float(bbox_norm.get("height"))
                except Exception:
                    continue

                x1 = (x_center - box_w / 2.0) * img_w
                y1 = (y_center - box_h / 2.0) * img_h
                x2 = (x_center + box_w / 2.0) * img_w
                y2 = (y_center + box_h / 2.0) * img_h

                x1 = max(0.0, min(img_w - 1.0, x1))
                y1 = max(0.0, min(img_h - 1.0, y1))
                x2 = max(0.0, min(img_w - 1.0, x2))
                y2 = max(0.0, min(img_h - 1.0, y2))
                if x2 <= x1 or y2 <= y1:
                    continue

                severity = self._extract_custom_json_severity(ann)
                image_boxes.append([x1, y1, x2, y2, cls_id, severity])

                image_meta.append(
                    {
                        "annotation_id": ann.get("annotation_id", None),
                        "category_name": class_name_map.get(cls_id, f"class_{cls_id}"),
                        "zone_estimation": ann.get("zone_estimation", None),
                        "damage_metrics": ann.get("damage_metrics", None),
                        "risk_assessment": ann.get("risk_assessment", None),
                        "description": ann.get("description", None),
                    }
                )

            if len(image_boxes) == 0 and split == "train":
                continue

            samples[img_rel].extend(image_boxes)
            if len(image_meta) > 0:
                sample_extras_map[img_rel] = image_meta

        if len(samples) == 0:
            raise AssertionError(f"No custom JSON samples found in {ann_file} for split '{split}'")

        return samples, classes_set, class_name_map, sample_extras_map

    def _resolve_yolo_label_path(self, image_split_dir, label_split_dir, rel_path_in_split):
        stem_rel = os.path.splitext(self._normalize_relpath(rel_path_in_split))[0]
        parts = [p for p in stem_rel.split("/") if p]
        base_name = os.path.basename(stem_rel) + ".txt"

        candidate_rel_paths = [stem_rel + ".txt", base_name]
        if len(parts) > 1 and parts[0].lower() == "images":
            no_images_rel = "/".join(parts[1:]) + ".txt"
            candidate_rel_paths.append(no_images_rel)
            candidate_rel_paths.append("labels/" + no_images_rel)
            candidate_rel_paths.append("labels/" + os.path.basename(no_images_rel))

        candidate_roots = [label_split_dir, image_split_dir, self.data_dir]
        for root in candidate_roots:
            if root is None:
                continue
            for rel_path in candidate_rel_paths:
                candidate_path = os.path.join(root, rel_path.replace("/", os.sep))
                if os.path.exists(candidate_path):
                    return candidate_path

        # Return the default candidate for predictable "missing label" behavior.
        return os.path.join(label_split_dir, base_name)

    def _collect_yolo_class_ids(self, labels_root):
        class_ids = set()
        if not os.path.isdir(labels_root):
            return class_ids
        for root, _, files in os.walk(labels_root):
            for file_name in files:
                if not file_name.endswith(".txt"):
                    continue
                label_path = os.path.join(root, file_name)
                with open(label_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) < 1:
                            continue
                        class_ids.add(int(float(parts[0])))
        return class_ids

    def _load_yolo_class_names(self):
        class_name_map = {}
        yaml_candidates = [
            os.path.join(self.data_dir, "data.yaml"),
            os.path.join(self.data_dir, "dataset.yaml"),
        ]
        yaml_file = next((path for path in yaml_candidates if os.path.exists(path)), None)
        if yaml_file is None:
            return class_name_map

        try:
            import yaml

            with open(yaml_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            names = data.get("names", None)
            if isinstance(names, list):
                class_name_map = {idx: str(name) for idx, name in enumerate(names)}
            elif isinstance(names, dict):
                for key, value in names.items():
                    class_name_map[int(key)] = str(value)
        except Exception as exc:
            logger.warning(f"Failed reading YOLO class names from {yaml_file}: {exc}")
        return class_name_map

    def _find_coco_annotation_file(self, split):
        split_aliases = self._split_aliases(split)
        split_tokens = []
        for alias in split_aliases:
            split_tokens.append(alias)
            if alias == "val":
                split_tokens.append("validation")
            if alias == "train":
                split_tokens.append("training")

        candidate_dirs = [
            os.path.join(self.data_dir, "annotations"),
            self.data_dir,
        ]
        candidate_names = []
        for token in split_tokens:
            candidate_names.extend(
                [
                    f"instances_{token}.json",
                    f"instances_{token}2017.json",
                    f"{token}.json",
                    f"{token}_annotations.json",
                ]
            )

        for directory in candidate_dirs:
            if not os.path.isdir(directory):
                continue
            for name in candidate_names:
                ann_file = os.path.join(directory, name)
                if os.path.exists(ann_file):
                    return ann_file
        return None

    def _find_custom_json_annotation_file(self, split):
        split_aliases = self._split_aliases(split)
        split_tokens = []
        for alias in split_aliases:
            split_tokens.append(alias)
            if alias == "val":
                split_tokens.extend(["valid", "validation"])
            if alias == "train":
                split_tokens.append("training")

        parent_dir = os.path.dirname(self.data_dir.rstrip("/\\"))
        candidate_dirs = [
            os.path.join(self.data_dir, "annotations"),
            os.path.join(self.data_dir, "json_splits"),
            os.path.join(parent_dir, "json_splits") if parent_dir else "",
            self.data_dir,
        ]

        candidate_names = []
        for token in split_tokens:
            candidate_names.extend(
                [
                    f"{token}.json",
                    f"{token}_annotations.json",
                    f"aircraft_damage_{token}.json",
                ]
            )

        # 1) Fast path: direct names
        for directory in candidate_dirs:
            if not directory or not os.path.isdir(directory):
                continue
            for name in candidate_names:
                ann_file = os.path.join(directory, name)
                if os.path.exists(ann_file):
                    return ann_file

        # 2) Fallback: scan json files and pick one that matches custom schema and split
        for directory in candidate_dirs:
            if not directory or not os.path.isdir(directory):
                continue
            for name in sorted(os.listdir(directory)):
                if not name.lower().endswith(".json"):
                    continue
                ann_file = os.path.join(directory, name)
                try:
                    with open(ann_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if not self._looks_like_custom_json(data):
                        continue
                    image_recs = data.get("images", [])
                    if not isinstance(image_recs, list):
                        continue
                    for rec in image_recs:
                        if not isinstance(rec, dict):
                            continue
                        rec_split = str(rec.get("split", "")).strip().lower()
                        if not rec_split or rec_split in split_tokens:
                            return ann_file
                except Exception:
                    continue
        return None

    def _looks_like_custom_json(self, data):
        if not isinstance(data, dict):
            return False
        images = data.get("images", None)
        if not isinstance(images, list) or len(images) == 0:
            return False
        first = images[0]
        if not isinstance(first, dict):
            return False
        if "annotations" in first and isinstance(first.get("annotations"), list):
            return True
        return False

    def _extract_custom_json_severity(self, ann):
        damage_metrics = ann.get("damage_metrics", None)
        if isinstance(damage_metrics, dict):
            for key in ["raw_severity_score", "severity_score", "severity", "risk_score"]:
                if key in damage_metrics and damage_metrics[key] is not None:
                    try:
                        return float(damage_metrics[key])
                    except Exception:
                        pass

        risk_assessment = ann.get("risk_assessment", None)
        if isinstance(risk_assessment, dict):
            for key in ["raw_severity_score", "severity_score", "severity", "risk_score"]:
                if key in risk_assessment and risk_assessment[key] is not None:
                    try:
                        return float(risk_assessment[key])
                    except Exception:
                        pass

        return float("nan")

    def _find_voc_root(self):
        candidates = [
            self.data_dir,
            self.image_dir,
            os.path.join(self.data_dir, "VOCdevkit", "VOC2007"),
            os.path.join(self.data_dir, "VOCdevkit", "VOC2012"),
            os.path.join(self.data_dir, "VOC2007"),
            os.path.join(self.data_dir, "VOC2012"),
        ]
        for root in candidates:
            if os.path.isdir(os.path.join(root, "Annotations")) and os.path.isdir(
                os.path.join(root, "JPEGImages")
            ):
                return root
        return None

    def _read_voc_split_ids(self, voc_root, split):
        split_dir = os.path.join(voc_root, "ImageSets", "Main")
        if not os.path.isdir(split_dir):
            return []

        if str(split).lower() == "train":
            candidates = ["train.txt", "trainval.txt"]
        else:
            candidates = ["val.txt", "test.txt", "validation.txt"]
        candidates.extend([f"{alias}.txt" for alias in self._split_aliases(split)])

        for file_name in candidates:
            file_path = os.path.join(split_dir, file_name)
            if not os.path.exists(file_path):
                continue
            image_ids = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    image_ids.append(line.split()[0])
            if len(image_ids) > 0:
                return image_ids
        return []

    def _resolve_relative_image_path(self, file_name, candidate_roots):
        file_name = self._normalize_relpath(file_name)
        if os.path.isabs(file_name) and os.path.exists(file_name):
            return self._normalize_relpath(os.path.relpath(file_name, self.image_dir))

        direct_path = os.path.join(self.image_dir, file_name.replace("/", os.sep))
        if os.path.exists(direct_path):
            return self._normalize_relpath(os.path.relpath(direct_path, self.image_dir))

        for root in candidate_roots:
            if not root:
                continue
            candidate = os.path.join(root, file_name.replace("/", os.sep))
            if os.path.exists(candidate):
                return self._normalize_relpath(os.path.relpath(candidate, self.image_dir))

        return file_name

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

    def _load_vocab_sources(self):
        closed_source = {}
        open_source = None

        vocab_file = str(getattr(self.cfg.IMAGES, "VOCAB_FILE", "")).strip()
        if vocab_file:
            vocab_path = self._resolve_vocab_path(vocab_file)
            vocab_data = self._read_vocab_data(vocab_path)
            closed_data, open_data = self._extract_vocab_sections(vocab_data)
            closed_source = self._normalize_vocab_entries(closed_data)
            if open_data is not None:
                open_source = self._normalize_vocab_entries(open_data)

        vocab_open_file = str(getattr(self.cfg.IMAGES, "VOCAB_OPEN_FILE", "")).strip()
        if vocab_open_file:
            vocab_open_path = self._resolve_vocab_path(vocab_open_file)
            open_source = self._normalize_vocab_entries(self._read_vocab_data(vocab_open_path))

        return closed_source, open_source

    def _extract_vocab_sections(self, vocab_data):
        if vocab_data is None:
            return {}, None

        if isinstance(vocab_data, dict):
            closed_data = self._find_section_in_json(
                vocab_data, ["closed", "known", "seen", "base", "train"]
            )
            open_data = self._find_section_in_json(
                vocab_data, ["open", "novel", "unseen", "all", "eval", "test"]
            )

            if closed_data is None:
                closed_data = self._find_section_in_json(
                    vocab_data, ["classes", "labels", "categories", "names", "vocabulary", "vocab"]
                )

            if closed_data is not None or open_data is not None:
                return (closed_data if closed_data is not None else {}), open_data

        return vocab_data, None

    def _find_section_in_json(self, node, aliases, depth=0):
        if depth > 8:
            return None

        alias_set = {
            str(alias).strip().lower().replace("-", "_").replace(" ", "_") for alias in aliases
        }

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

    def _normalize_vocab_entries(self, data):
        entries = {}
        self._extract_vocab_entries(data, entries)
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

    def _extract_vocab_entries(self, data, entries, key_hint=None, depth=0):
        if data is None or depth > 12:
            return

        if isinstance(data, dict):
            parsed_entry = self._extract_vocab_entry_from_dict(data, fallback_key=key_hint)
            if parsed_entry is not None and self._dict_is_entry_like(data):
                self._add_vocab_entry(entries, parsed_entry[0], parsed_entry[1])
                return

            for raw_key, raw_value in data.items():
                key = str(raw_key).strip()
                if not key:
                    continue
                key_lower = key.lower()
                if self._is_metadata_key(key_lower):
                    continue

                if isinstance(raw_value, dict):
                    parsed_child = self._extract_vocab_entry_from_dict(raw_value, fallback_key=key)
                    if parsed_child is not None:
                        self._add_vocab_entry(entries, parsed_child[0], parsed_child[1])
                    else:
                        next_hint = None if self._is_vocab_container_key(key_lower) else key
                        self._extract_vocab_entries(
                            raw_value, entries, key_hint=next_hint, depth=depth + 1
                        )
                elif isinstance(raw_value, list):
                    if len(raw_value) == 0:
                        continue
                    if self._is_vocab_container_key(key_lower):
                        self._extract_vocab_entries(
                            raw_value, entries, key_hint=None, depth=depth + 1
                        )
                    elif all(not isinstance(item, (dict, list)) for item in raw_value):
                        self._add_vocab_entry(entries, key, raw_value)
                    else:
                        self._extract_vocab_entries(
                            raw_value, entries, key_hint=key, depth=depth + 1
                        )
                elif isinstance(raw_value, (str, int, float)):
                    self._add_vocab_entry(entries, key, raw_value)
            return

        if isinstance(data, list):
            for idx, item in enumerate(data):
                if isinstance(item, dict):
                    parsed_item = self._extract_vocab_entry_from_dict(item, fallback_key=None)
                    if parsed_item is not None:
                        self._add_vocab_entry(entries, parsed_item[0], parsed_item[1])
                    else:
                        next_hint = f"{key_hint}_{idx}" if key_hint else None
                        self._extract_vocab_entries(
                            item, entries, key_hint=next_hint, depth=depth + 1
                        )
                elif isinstance(item, list):
                    self._extract_vocab_entries(item, entries, key_hint=key_hint, depth=depth + 1)
                elif isinstance(item, (str, int, float)):
                    label = str(item).strip()
                    if not label:
                        continue
                    self._add_vocab_entry(entries, label, label)
            return

        if isinstance(data, (str, int, float)):
            value = str(data).strip()
            if not value:
                return
            key = key_hint if key_hint else value
            self._add_vocab_entry(entries, key, value)

    def _extract_vocab_entry_from_dict(self, data, fallback_key=None):
        if not isinstance(data, dict) or len(data) == 0:
            return None

        lowered = {str(k).strip().lower(): v for k, v in data.items()}

        key = None
        for field in ["name", "label", "class", "class_name", "category", "category_name"]:
            value = lowered.get(field, None)
            if value is not None and str(value).strip():
                key = str(value).strip()
                break

        if key is None:
            for field in ["id", "class_id", "category_id", "index"]:
                value = lowered.get(field, None)
                if value is not None and str(value).strip():
                    key = str(value).strip()
                    break

        if key is None and fallback_key is not None and str(fallback_key).strip():
            key = str(fallback_key).strip()

        if key is None:
            return None

        caption = None
        for field in [
            "caption",
            "text",
            "prompt",
            "description",
            "synonyms",
            "templates",
            "phrases",
            "sentences",
        ]:
            if field in lowered and lowered[field] is not None:
                caption = lowered[field]
                break

        if caption is None and "name" in lowered and lowered["name"] is not None:
            caption = lowered["name"]

        if caption is None and len(data) == 1:
            caption = next(iter(data.values()))

        if caption is None:
            caption = key

        return key, caption

    def _dict_is_entry_like(self, data):
        if not isinstance(data, dict):
            return False
        keys = {str(k).strip().lower() for k in data.keys()}
        entry_keys = {
            "id",
            "class_id",
            "category_id",
            "index",
            "name",
            "label",
            "class",
            "class_name",
            "category",
            "category_name",
            "caption",
            "text",
            "prompt",
            "description",
            "synonyms",
            "templates",
            "phrases",
            "sentences",
        }
        return len(keys.intersection(entry_keys)) > 0

    def _add_vocab_entry(self, entries, key, caption):
        key_str = str(key).strip()
        if not key_str or key_str in entries:
            return
        entries[key_str] = {"caption": self._normalize_caption(caption, default_text=key_str)}

    def _is_vocab_container_key(self, key_lower):
        return key_lower in {
            "closed",
            "open",
            "classes",
            "labels",
            "categories",
            "names",
            "vocabulary",
            "vocab",
            "known",
            "seen",
            "base",
            "train",
            "novel",
            "unseen",
            "all",
            "eval",
            "test",
        }

    def _is_metadata_key(self, key_lower):
        return key_lower in {
            "meta",
            "metadata",
            "info",
            "version",
            "license",
            "licenses",
            "description",
            "images",
            "annotations",
            "path",
            "paths",
            "root",
            "dataset",
            "source",
            "url",
        }

    def _materialize_text_data(self, source_entries, allow_extra=False):
        data = {}
        consumed_keys = set()

        source_keys_lut = {str(key): key for key in source_entries.keys()}
        source_keys_lut_lower = {str(key).lower(): key for key in source_entries.keys()}

        for idx, class_name in enumerate(self.class_names):
            class_id = self.idx_to_class_id.get(idx, idx)
            candidates = [class_name, str(class_id), str(idx)]
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
                caption = self._format_prompt(class_name)
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
            class_id = int(key_str)
            if class_id in self.class_id_to_idx:
                return self.class_names[self.class_id_to_idx[class_id]]
            return f"class_{class_id}"
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

    def _split_aliases(self, split):
        split_name = str(split).lower()
        if split_name == "train":
            return ["train", "training", "stratified_train"]
        if split_name in ["test", "val", "valid", "validation"]:
            return ["val", "valid", "validation", "test", "stratified_val"]
        return [split_name]

    def _find_existing_split_dir(self, base_dir, aliases):
        for alias in aliases:
            path = os.path.join(base_dir, alias)
            if os.path.isdir(path):
                return path, alias
        return None, None

    def _normalize_relpath(self, path):
        return path.replace("\\", "/")


# Aliases for backward compatibility
MultimodalImageDataset = ImageDataset
OpenVocabularyDataset = ImageDataset
