import os
import re
from pprint import pformat

import numpy as np
import torch
import torch.nn.functional as F

from ..pascal_wrapper import frame_mAP_pascal


def _make_image_key(image_rel):
    return str(image_rel)


_TILE_KEY_RE = re.compile(r"^(?P<base>.+)__x(?P<x>\d+)_y(?P<y>\d+)$")


def _parse_tiled_key(image_key):
    stem = os.path.splitext(str(image_key))[0]
    m = _TILE_KEY_RE.match(stem)
    if m is None:
        return None
    return m.group("base"), int(m.group("x")), int(m.group("y"))


def _norm_xyxy_to_abs(boxes_norm, width, height):
    boxes = np.asarray(boxes_norm, dtype=np.float32).reshape(-1, 4)
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    out = boxes.copy()
    out[:, [0, 2]] *= float(width)
    out[:, [1, 3]] *= float(height)
    return out


def _abs_xyxy_to_norm(boxes_abs, width, height):
    boxes = np.asarray(boxes_abs, dtype=np.float32).reshape(-1, 4)
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    out = boxes.copy()
    out[:, [0, 2]] /= max(1.0, float(width))
    out[:, [1, 3]] /= max(1.0, float(height))
    return np.clip(out, 0.0, 1.0)


def _pairwise_iou(box, boxes):
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    w = np.clip(xx2 - xx1, a_min=0.0, a_max=None)
    h = np.clip(yy2 - yy1, a_min=0.0, a_max=None)
    inter = w * h
    area_a = max(0.0, (box[2] - box[0]) * (box[3] - box[1]))
    area_b = np.clip(
        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), a_min=0.0, a_max=None
    )
    union = np.clip(area_a + area_b - inter, a_min=1e-6, a_max=None)
    return (inter / union).astype(np.float32)


def _nms_xyxy(boxes, scores, iou_thresh):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    order = np.argsort(-scores)
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = _pairwise_iou(boxes[i], boxes[rest])
        order = rest[ious <= float(iou_thresh)]
    return np.asarray(keep, dtype=np.int64)


def _dedupe_gt_xyxy(boxes, labels, iou_thresh):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if boxes.shape[0] == 0:
        return boxes, labels
    areas = np.clip(
        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), a_min=0.0, a_max=None
    )
    keep_global = []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        if cls_idx.size == 0:
            continue
        order = cls_idx[np.argsort(-areas[cls_idx])]
        kept = []
        for idx in order:
            if len(kept) == 0:
                kept.append(int(idx))
                continue
            ious = _pairwise_iou(boxes[idx], boxes[np.asarray(kept, dtype=np.int64)])
            if np.all(ious <= float(iou_thresh)):
                kept.append(int(idx))
        keep_global.extend(kept)
    keep_global = np.asarray(sorted(set(keep_global)), dtype=np.int64)
    return boxes[keep_global], labels[keep_global]


def _stitch_tiled_predictions(results, targets, nms_iou=0.5, gt_dedup_iou=0.9):
    total = len(targets)
    tile_keys = 0
    groups = {}

    for image_key, target in targets.items():
        parsed = _parse_tiled_key(image_key)
        if parsed is None:
            continue
        tile_keys += 1
        base, off_x, off_y = parsed
        h, w = target["resolution"]
        group = groups.setdefault(
            base,
            {
                "full_w": 0,
                "full_h": 0,
                "gt_boxes_abs": [],
                "gt_labels": [],
                "pred_boxes_abs": [],
                "pred_scores": [],
                "pred_labels": [],
            },
        )
        group["full_w"] = max(int(group["full_w"]), int(off_x + w))
        group["full_h"] = max(int(group["full_h"]), int(off_y + h))

        gt_abs = _norm_xyxy_to_abs(target["bbox"], w, h)
        if gt_abs.shape[0] > 0:
            gt_abs[:, [0, 2]] += float(off_x)
            gt_abs[:, [1, 3]] += float(off_y)
            group["gt_boxes_abs"].append(gt_abs)
            group["gt_labels"].append(np.asarray(target["labels"], dtype=np.int64))

        det = results.get(image_key, None)
        if det is None:
            continue
        det_abs = _norm_xyxy_to_abs(det["boxes"], w, h)
        if det_abs.shape[0] > 0:
            det_abs[:, [0, 2]] += float(off_x)
            det_abs[:, [1, 3]] += float(off_y)
            group["pred_boxes_abs"].append(det_abs)
            group["pred_scores"].append(np.asarray(det["scores"], dtype=np.float32))
            group["pred_labels"].append(np.asarray(det["action_ids"], dtype=np.int64))

    if tile_keys == 0:
        return None, None, 0, 0

    # Require majority tiled naming to avoid accidental remap on normal datasets.
    if tile_keys < max(1, int(round(0.6 * total))):
        return None, None, tile_keys, len(groups)

    stitched_results = {}
    stitched_targets = {}

    for base, group in groups.items():
        full_w = max(1, int(group["full_w"]))
        full_h = max(1, int(group["full_h"]))

        if group["gt_boxes_abs"]:
            gt_boxes_abs = np.concatenate(group["gt_boxes_abs"], axis=0)
            gt_labels = np.concatenate(group["gt_labels"], axis=0)
            gt_boxes_abs, gt_labels = _dedupe_gt_xyxy(
                gt_boxes_abs, gt_labels, gt_dedup_iou
            )
        else:
            gt_boxes_abs = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.zeros((0,), dtype=np.int64)

        stitched_targets[base] = {
            "bbox": _abs_xyxy_to_norm(gt_boxes_abs, full_w, full_h),
            "labels": gt_labels.tolist(),
            "resolution": (full_h, full_w),
        }

        if group["pred_boxes_abs"]:
            pred_boxes_abs = np.concatenate(group["pred_boxes_abs"], axis=0)
            pred_scores = np.concatenate(group["pred_scores"], axis=0)
            pred_labels = np.concatenate(group["pred_labels"], axis=0)

            keep_idx_all = []
            for cls in np.unique(pred_labels):
                cls_idx = np.where(pred_labels == cls)[0]
                if cls_idx.size == 0:
                    continue
                keep_local = _nms_xyxy(
                    pred_boxes_abs[cls_idx], pred_scores[cls_idx], nms_iou
                )
                keep_idx_all.extend(cls_idx[keep_local].tolist())

            if keep_idx_all:
                keep_idx = np.asarray(sorted(keep_idx_all), dtype=np.int64)
                pred_boxes_abs = pred_boxes_abs[keep_idx]
                pred_scores = pred_scores[keep_idx]
                pred_labels = pred_labels[keep_idx]
            else:
                pred_boxes_abs = np.zeros((0, 4), dtype=np.float32)
                pred_scores = np.zeros((0,), dtype=np.float32)
                pred_labels = np.zeros((0,), dtype=np.int64)
        else:
            pred_boxes_abs = np.zeros((0, 4), dtype=np.float32)
            pred_scores = np.zeros((0,), dtype=np.float32)
            pred_labels = np.zeros((0,), dtype=np.int64)

        stitched_results[base] = {
            "boxes": _abs_xyxy_to_norm(pred_boxes_abs, full_w, full_h),
            "scores": pred_scores.astype(np.float32),
            "action_ids": pred_labels.astype(np.int64),
        }

    return stitched_results, stitched_targets, tile_keys, len(groups)


def _prepare_for_image_ap(predictions, dataset, score_thresh=0.0):
    results = {}
    targets = {}

    for sample_id, sample in enumerate(dataset.samples):
        info = dataset.get_sample_info(sample_id)
        image_key = _make_image_key(info["image_id"])
        gt_boxes = sample["boxes"].copy()
        if len(gt_boxes) > 0:
            gt_boxes[:, [0, 2]] /= float(info["width"])
            gt_boxes[:, [1, 3]] /= float(info["height"])
        gt_labels = (sample["labels"] + 1).tolist()
        targets[image_key] = {
            "bbox": gt_boxes,
            "labels": gt_labels,
            "resolution": info["resolution"],
        }

    for sample_id, prediction in enumerate(predictions):
        if len(prediction) == 0:
            continue
        info = dataset.get_sample_info(sample_id)
        image_key = _make_image_key(info["image_id"])

        boxes = prediction[0].numpy()
        scores = (
            torch.sigmoid(prediction[1])
            if dataset.multilabel_action
            else F.softmax(prediction[1], dim=-1)
        )
        scores = scores.numpy()
        # Single-label detection heads should emit one class per query. Treating
        # softmax outputs as multi-label (np.where over all classes) creates
        # duplicate detections and severely hurts precision.
        if dataset.multilabel_action:
            box_ids, class_ids = np.where(scores >= score_thresh)
            valid_ids = class_ids < dataset.num_classes
            box_ids = box_ids[valid_ids]
            class_ids = class_ids[valid_ids]
            if len(box_ids) == 0:
                results[image_key] = {
                    "boxes": np.zeros((0, 4)),
                    "scores": np.array([]),
                    "action_ids": np.array([]),
                }
                continue
            det_boxes = boxes[box_ids, :]
            det_scores = scores[box_ids, class_ids]
            det_action_ids = class_ids + 1
        else:
            fg_scores = scores[:, : dataset.num_classes]
            class_ids = fg_scores.argmax(axis=1)
            best_scores = fg_scores[np.arange(fg_scores.shape[0]), class_ids]
            keep = best_scores >= float(score_thresh)
            if not np.any(keep):
                results[image_key] = {
                    "boxes": np.zeros((0, 4)),
                    "scores": np.array([]),
                    "action_ids": np.array([]),
                }
                continue
            det_boxes = boxes[keep, :]
            det_scores = best_scores[keep]
            det_action_ids = class_ids[keep] + 1

        results[image_key] = {
            "boxes": det_boxes,
            "scores": det_scores,
            "action_ids": det_action_ids,
        }

    return results, targets


def _prepare_for_multimodal_image_ap(
    predictions, dataset, score_thresh=0.0, text_prompts=None
):
    """Prepare predictions for multimodal (image + text) evaluation.

    Args:
        predictions: Model predictions
        dataset: Dataset object
        score_thresh: Score threshold
        text_prompts: Optional text prompts for open vocabulary evaluation

    Returns:
        tuple: (results, targets)
    """
    results = {}
    targets = {}

    # Get class to text mapping
    class_text_map = None
    if text_prompts is not None:
        # Use provided text prompts
        if isinstance(text_prompts, dict):
            class_text_map = text_prompts
        elif isinstance(text_prompts, list):
            class_text_map = {i: text_prompts[i] for i in range(len(text_prompts))}

    for sample_id, sample in enumerate(dataset.samples):
        info = dataset.get_sample_info(sample_id)
        image_key = _make_image_key(info["image_id"])
        gt_boxes = sample["boxes"].copy()
        if len(gt_boxes) > 0:
            gt_boxes[:, [0, 2]] /= float(info["width"])
            gt_boxes[:, [1, 3]] /= float(info["height"])
        gt_labels = (sample["labels"] + 1).tolist()

        # Add text prompts if available
        if class_text_map is not None:
            gt_text_prompts = [
                class_text_map.get(label, f"class_{label}") for label in gt_labels
            ]
        else:
            gt_text_prompts = None

        targets[image_key] = {
            "bbox": gt_boxes,
            "labels": gt_labels,
            "resolution": info["resolution"],
            "text_prompts": gt_text_prompts,
        }

    for sample_id, prediction in enumerate(predictions):
        if len(prediction) == 0:
            continue
        info = dataset.get_sample_info(sample_id)
        image_key = _make_image_key(info["image_id"])

        boxes = prediction[0].numpy()

        # Handle different prediction formats
        if isinstance(prediction[1], torch.Tensor):
            scores = (
                torch.sigmoid(prediction[1])
                if dataset.multilabel_action
                else F.softmax(prediction[1], dim=-1)
            )
            scores = scores.numpy()
        else:
            scores = prediction[1]
        if dataset.multilabel_action:
            box_ids, class_ids = np.where(scores >= score_thresh)
            valid_ids = class_ids < dataset.num_classes
            box_ids = box_ids[valid_ids]
            class_ids = class_ids[valid_ids]
            if len(box_ids) == 0:
                results[image_key] = {
                    "boxes": np.zeros((0, 4)),
                    "scores": np.array([]),
                    "action_ids": np.array([]),
                    "text_prompts": np.array([]),
                }
                continue
            det_boxes = boxes[box_ids, :]
            det_scores = scores[box_ids, class_ids]
            det_class_ids = class_ids
        else:
            fg_scores = scores[:, : dataset.num_classes]
            class_ids = fg_scores.argmax(axis=1)
            best_scores = fg_scores[np.arange(fg_scores.shape[0]), class_ids]
            keep = best_scores >= float(score_thresh)
            if not np.any(keep):
                results[image_key] = {
                    "boxes": np.zeros((0, 4)),
                    "scores": np.array([]),
                    "action_ids": np.array([]),
                    "text_prompts": np.array([]),
                }
                continue
            det_boxes = boxes[keep, :]
            det_scores = best_scores[keep]
            det_class_ids = class_ids[keep]

        # Add text prompts for predictions
        if class_text_map is not None:
            pred_text_prompts = [
                class_text_map.get(int(cid), f"class_{int(cid)}")
                for cid in det_class_ids
            ]
        else:
            pred_text_prompts = [f"class_{int(cid)}" for cid in det_class_ids]

        results[image_key] = {
            "boxes": det_boxes,
            "scores": det_scores,
            "action_ids": det_class_ids + 1,
            "text_prompts": pred_text_prompts,
        }

    return results, targets


def _box_area_pixels(boxes_norm, resolution):
    if boxes_norm is None or len(boxes_norm) == 0:
        return np.zeros((0,), dtype=np.float32)
    h, w = resolution
    widths = np.clip(
        boxes_norm[:, 2] - boxes_norm[:, 0], a_min=0.0, a_max=None
    ) * float(w)
    heights = np.clip(
        boxes_norm[:, 3] - boxes_norm[:, 1], a_min=0.0, a_max=None
    ) * float(h)
    return (widths * heights).astype(np.float32)


def _filter_by_area(results, targets, area_min, area_max):
    filtered_results = {}
    filtered_targets = {}
    total_gt = 0
    total_det = 0

    for image_key, target in targets.items():
        resolution = target["resolution"]
        gt_boxes = target["bbox"]
        gt_labels = np.asarray(target["labels"], dtype=np.int64)
        gt_areas = _box_area_pixels(gt_boxes, resolution)
        gt_mask = (gt_areas >= float(area_min)) & (gt_areas < float(area_max))
        gt_boxes_keep = gt_boxes[gt_mask]
        gt_labels_keep = gt_labels[gt_mask].tolist()
        total_gt += int(gt_mask.sum())

        filtered_targets[image_key] = {
            "bbox": gt_boxes_keep,
            "labels": gt_labels_keep,
            "resolution": resolution,
        }

        det = results.get(image_key, None)
        if det is None:
            continue
        det_boxes = det["boxes"]
        det_scores = np.asarray(det["scores"], dtype=np.float32)
        det_labels = np.asarray(det["action_ids"], dtype=np.int64)
        det_areas = _box_area_pixels(det_boxes, resolution)
        det_mask = (det_areas >= float(area_min)) & (det_areas < float(area_max))
        total_det += int(det_mask.sum())
        filtered_results[image_key] = {
            "boxes": det_boxes[det_mask],
            "scores": det_scores[det_mask],
            "action_ids": det_labels[det_mask],
        }

    return filtered_results, filtered_targets, total_gt, total_det


def _extract_map_key(eval_res):
    for key in eval_res.keys():
        if "Precision/mAP@" in str(key):
            return key
    return None


def _format_iou(iou):
    return f"{float(iou):.2f}".rstrip("0").rstrip(".")


def _ap_key_for_iou(iou):
    return f"PascalBoxes_Precision/mAP@{_format_iou(iou)}IOU"


def _build_ap5095_iou_list(dataset):
    cfg_obj = getattr(dataset, "cfg", None)
    test_cfg = getattr(cfg_obj, "TEST", None) if cfg_obj is not None else None
    iou_min = float(getattr(test_cfg, "AP5095_MIN", 0.5))
    iou_max = float(getattr(test_cfg, "AP5095_MAX", 0.95))
    iou_step = float(getattr(test_cfg, "AP5095_STEP", 0.05))

    iou_min = max(0.0, min(1.0, iou_min))
    iou_max = max(0.0, min(1.0, iou_max))
    if iou_max < iou_min:
        iou_min, iou_max = iou_max, iou_min
    if iou_step <= 0.0:
        iou_step = 0.05

    vals = []
    cur = iou_min
    while cur <= (iou_max + 1e-9):
        vals.append(round(cur, 10))
        cur += iou_step
    if not vals:
        vals = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    return vals


def _extract_map_at_iou(eval_res, iou):
    exact_key = _ap_key_for_iou(iou)
    if exact_key in eval_res:
        try:
            return float(eval_res[exact_key])
        except Exception:
            return None

    suffix = f"@{_format_iou(iou)}IOU"
    for key, value in eval_res.items():
        if str(key).startswith("PascalBoxes_Precision/mAP@") and str(key).endswith(
            suffix
        ):
            try:
                return float(value)
            except Exception:
                return None
    return None


def _maybe_add_ap5095_metrics(dataset, results, targets, eval_res, logger):
    cfg_obj = getattr(dataset, "cfg", None)
    test_cfg = getattr(cfg_obj, "TEST", None) if cfg_obj is not None else None
    report_ap5095 = bool(getattr(test_cfg, "REPORT_AP5095", False))
    if not report_ap5095:
        return eval_res

    iou_list = _build_ap5095_iou_list(dataset)
    if len(iou_list) <= 1:
        return eval_res

    eval_sweep = frame_mAP_pascal(
        results,
        targets,
        dataset.closed_set_classes,
        logger,
        iou_list=iou_list,
    )
    eval_res.update(eval_sweep)

    per_iou = []
    for iou in iou_list:
        v = _extract_map_at_iou(eval_sweep, iou)
        if v is not None:
            per_iou.append(v)
    if per_iou:
        eval_res["PascalBoxes_Precision/mAP@0.5:0.95IOU"] = float(np.mean(per_iou))
    return eval_res


def _compute_area_ap_breakdown(results, targets, dataset, logger):
    iou_value = float(dataset.test_iou_thresh)
    area_bins = [
        ("small", 0.0, 32.0 * 32.0),
        ("medium", 32.0 * 32.0, 96.0 * 96.0),
        ("large", 96.0 * 96.0, float("inf")),
    ]
    out = {}

    for name, area_min, area_max in area_bins:
        det_bin, tgt_bin, num_gt, num_det = _filter_by_area(
            results, targets, area_min, area_max
        )
        out[f"Area/{name}/num_gt"] = int(num_gt)
        out[f"Area/{name}/num_det"] = int(num_det)

        if num_gt <= 0:
            out[f"Area/{name}/mAP@{iou_value}IOU"] = -1.0
            continue

        try:
            eval_bin = frame_mAP_pascal(
                det_bin,
                tgt_bin,
                dataset.closed_set_classes,
                logger,
                iou_list=[iou_value],
            )
            map_key = _extract_map_key(eval_bin)
            map_val = float(eval_bin[map_key]) if map_key is not None else -1.0
        except Exception:
            map_val = -1.0
        out[f"Area/{name}/mAP@{iou_value}IOU"] = map_val

    out[f"SmallObject/AP@{iou_value}IOU"] = out.get(
        f"Area/small/mAP@{iou_value}IOU", -1.0
    )
    return out


def _compute_detection_precision_recall(results, targets, iou_thresh):
    pred_items = []
    gt_lookup = {}
    total_gt = 0

    for image_key, target in targets.items():
        gt_boxes = np.asarray(target.get("bbox", []), dtype=np.float32).reshape(-1, 4)
        gt_labels = np.asarray(target.get("labels", []), dtype=np.int64).reshape(-1)
        if gt_boxes.shape[0] == 0:
            gt_lookup[image_key] = {
                "boxes": gt_boxes,
                "labels": gt_labels,
                "matched": np.zeros((0,), dtype=bool),
            }
            continue
        gt_lookup[image_key] = {
            "boxes": gt_boxes,
            "labels": gt_labels,
            "matched": np.zeros((gt_boxes.shape[0],), dtype=bool),
        }
        total_gt += int(gt_boxes.shape[0])

    for image_key, det in results.items():
        det_boxes = np.asarray(det.get("boxes", []), dtype=np.float32).reshape(-1, 4)
        det_scores = np.asarray(det.get("scores", []), dtype=np.float32).reshape(-1)
        det_labels = np.asarray(det.get("action_ids", []), dtype=np.int64).reshape(-1)
        count = min(det_boxes.shape[0], det_scores.shape[0], det_labels.shape[0])
        for idx in range(count):
            pred_items.append(
                (
                    float(det_scores[idx]),
                    str(image_key),
                    int(det_labels[idx]),
                    det_boxes[idx],
                )
            )

    pred_items.sort(key=lambda item: item[0], reverse=True)

    tp = 0
    fp = 0

    for _score, image_key, det_label, det_box in pred_items:
        gt_info = gt_lookup.get(image_key)
        if gt_info is None or gt_info["boxes"].shape[0] == 0:
            fp += 1
            continue

        label_mask = gt_info["labels"] == int(det_label)
        candidate_idx = np.where(label_mask & (~gt_info["matched"]))[0]
        if candidate_idx.size == 0:
            fp += 1
            continue

        ious = _pairwise_iou(det_box, gt_info["boxes"][candidate_idx])
        if ious.size == 0:
            fp += 1
            continue

        best_local = int(np.argmax(ious))
        best_iou = float(ious[best_local])
        if best_iou >= float(iou_thresh):
            gt_info["matched"][candidate_idx[best_local]] = True
            tp += 1
        else:
            fp += 1

    fn = max(0, int(total_gt - tp))
    precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = float(tp) / float(total_gt) if total_gt > 0 else 0.0
    return {
        f"Detection/Precision@{_format_iou(iou_thresh)}IOU": float(precision),
        f"Detection/Recall@{_format_iou(iou_thresh)}IOU": float(recall),
        f"Detection/TP@{_format_iou(iou_thresh)}IOU": int(tp),
        f"Detection/FP@{_format_iou(iou_thresh)}IOU": int(fp),
        f"Detection/FN@{_format_iou(iou_thresh)}IOU": int(fn),
    }


def _metric_float(eval_res, key, default=None):
    value = eval_res.get(key, default)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return default


def _format_console_metric(value):
    if value is None:
        return "n/a"
    value = float(value)
    if value < 0.0:
        return "n/a"
    return f"{value:.4f}"


def _format_console_eval_summary(eval_res):
    precision = _metric_float(eval_res, "Detection/Precision@0.5IOU")
    recall = _metric_float(eval_res, "Detection/Recall@0.5IOU")
    map50 = _metric_float(eval_res, "PascalBoxes_Precision/mAP@0.5IOU")
    map5095 = _metric_float(eval_res, "PascalBoxes_Precision/mAP@0.5:0.95IOU")
    if map5095 is None:
        map5095 = 0.0
    small_ap = _metric_float(eval_res, "SmallObject/AP@0.5IOU")
    return (
        f"precision={_format_console_metric(precision)} "
        f"recall={_format_console_metric(recall)} "
        f"mAP50={_format_console_metric(map50)} "
        f"mAP50-95={_format_console_metric(map5095)} "
        f"smallAP={_format_console_metric(small_ap)}"
    )


def evaluate_with_text_prompts(
    predictions, dataset, text_prompts, output_folder=None, logger=None
):
    """Evaluate predictions with text prompts for open vocabulary detection.

    Args:
        predictions: Model predictions
        dataset: Dataset object
        text_prompts: Dict or list of text prompts for each class
        output_folder: Folder to save results
        logger: Logger object

    Returns:
        dict: Evaluation results
    """
    if logger is None:
        import logging

        logger = logging.getLogger("alphaction.image_eval")

    logger.info("Evaluating with text prompts for open vocabulary detection.")

    # Prepare predictions with text prompts
    results, targets = _prepare_for_multimodal_image_ap(
        predictions, dataset, text_prompts=text_prompts
    )

    # Evaluate
    eval_res = frame_mAP_pascal(
        results,
        targets,
        dataset.closed_set_classes,
        logger,
        iou_list=[dataset.test_iou_thresh],
    )

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        log_file_path = os.path.join(output_folder, "result_multimodal.log")
        with open(log_file_path, "w") as logf:
            logf.write("Evaluation results (multimodal with text prompts)\n")
            logf.write(pformat(eval_res))

    return eval_res, results


def compute_text_similarity_scores(predictions, text_features, class_names=None):
    """Compute text similarity scores for predictions.

    Args:
        predictions: Model predictions (boxes, scores)
        text_features: Text features from CLIP text encoder
        class_names: Optional class names for mapping

    Returns:
        numpy.ndarray: Updated scores with text similarity
    """
    if not isinstance(predictions, (list, tuple)) or len(predictions) < 2:
        return predictions

    boxes = predictions[0]
    scores = predictions[1]

    if isinstance(scores, torch.Tensor):
        scores = scores.numpy()

    # If text features provided, use them to refine scores
    if text_features is not None and len(text_features) > 0:
        if isinstance(text_features, torch.Tensor):
            text_features = text_features.cpu().numpy()

        # Compute similarity between image features and text features
        # This is a placeholder - actual implementation depends on model architecture
        text_sim = np.dot(scores, text_features.T)
        scores = scores * 0.7 + text_sim * 0.3  # Combine

    return boxes, scores


def do_image_evaluation(
    dataset, predictions, output_folder, logger, metric="image_ap", save_csv=False
):
    del save_csv  # not used for image evaluation
    eval_metric = metric.lower()
    if eval_metric not in ["image_ap", "frame_ap"]:
        raise NotImplementedError(
            "Unsupported metric '{}' for ImageDataset.".format(metric)
        )

    logger.info("Preparing image results for {} evaluation.".format(eval_metric))
    results, targets = _prepare_for_image_ap(predictions, dataset)

    cfg_obj = getattr(dataset, "cfg", None)
    test_cfg = getattr(cfg_obj, "TEST", None) if cfg_obj is not None else None
    stitch_eval = bool(getattr(test_cfg, "TILE_STITCH_EVAL", False))
    if stitch_eval:
        stitch_nms_iou = float(getattr(test_cfg, "TILE_STITCH_NMS_IOU", 0.5))
        stitch_gt_dedup_iou = float(getattr(test_cfg, "TILE_STITCH_GT_DEDUP_IOU", 0.9))
        stitched_results, stitched_targets, tile_keys, stitched_images = (
            _stitch_tiled_predictions(
                results,
                targets,
                nms_iou=stitch_nms_iou,
                gt_dedup_iou=stitch_gt_dedup_iou,
            )
        )
        if stitched_results is not None and stitched_targets is not None:
            results, targets = stitched_results, stitched_targets
            logger.info(
                "Tile-stitch evaluation enabled: remapped %d tile samples into %d full-image groups "
                "(NMS IoU=%.2f, GT dedup IoU=%.2f).",
                tile_keys,
                stitched_images,
                stitch_nms_iou,
                stitch_gt_dedup_iou,
            )
        else:
            logger.info(
                "Tile-stitch evaluation enabled but skipped (tile-like keys found: %d).",
                tile_keys,
            )

    eval_res = frame_mAP_pascal(
        results,
        targets,
        dataset.closed_set_classes,
        logger,
        iou_list=[dataset.test_iou_thresh],
    )
    eval_res = _maybe_add_ap5095_metrics(dataset, results, targets, eval_res, logger)
    eval_res.update(_compute_area_ap_breakdown(results, targets, dataset, logger))
    eval_res.update(
        _compute_detection_precision_recall(
            results, targets, iou_thresh=float(dataset.test_iou_thresh)
        )
    )

    logger.info(
        "Evaluation results (%s): %s",
        eval_metric,
        _format_console_eval_summary(eval_res),
    )
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        log_file_path = os.path.join(output_folder, "result_image.log")
        with open(log_file_path, "w") as logf:
            logf.write("Evaluation results (metric: {})\n".format(eval_metric))
            logf.write(pformat(eval_res))

    return eval_res, results


def do_multimodal_image_evaluation(
    dataset,
    predictions,
    output_folder,
    logger,
    metric="image_ap",
    text_prompts=None,
    save_csv=False,
):
    """Perform multimodal (image + text) evaluation.

    Args:
        dataset: Dataset object
        predictions: Model predictions
        output_folder: Folder to save results
        logger: Logger object
        metric: Evaluation metric
        text_prompts: Text prompts for open vocabulary detection
        save_csv: Whether to save CSV

    Returns:
        tuple: (eval_res, results)
    """
    del save_csv  # not used for image evaluation

    eval_metric = metric.lower()
    if eval_metric not in ["image_ap", "frame_ap"]:
        raise NotImplementedError(
            "Unsupported metric '{}' for ImageDataset.".format(metric)
        )

    logger.info(
        "Preparing multimodal image results for {} evaluation.".format(eval_metric)
    )

    # Use text prompts if provided
    if text_prompts is not None:
        results, targets = _prepare_for_multimodal_image_ap(
            predictions, dataset, text_prompts=text_prompts
        )
    else:
        results, targets = _prepare_for_image_ap(predictions, dataset)

    eval_res = frame_mAP_pascal(
        results,
        targets,
        dataset.closed_set_classes,
        logger,
        iou_list=[dataset.test_iou_thresh],
    )
    eval_res = _maybe_add_ap5095_metrics(dataset, results, targets, eval_res, logger)
    eval_res.update(_compute_area_ap_breakdown(results, targets, dataset, logger))
    eval_res.update(
        _compute_detection_precision_recall(
            results, targets, iou_thresh=float(dataset.test_iou_thresh)
        )
    )

    logger.info(
        "Multimodal evaluation results (%s): %s",
        eval_metric,
        _format_console_eval_summary(eval_res),
    )
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        log_file_path = os.path.join(output_folder, "result_multimodal_image.log")
        with open(log_file_path, "w") as logf:
            logf.write(
                "Multimodal evaluation results (metric: {})\n".format(eval_metric)
            )
            logf.write(pformat(eval_res))

    return eval_res, results
