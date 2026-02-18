import logging
import os
import torch
from tqdm import tqdm
import time
import datetime

from alphaction.dataset.datasets.evaluation import evaluate
from alphaction.utils.comm import get_rank, is_main_process, gather, synchronize, get_world_size
from alphaction.structures.bounding_box import BoxList


def _resolve_model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _stack_metadata(metadata):
    if isinstance(metadata, dict):
        return metadata
    if not isinstance(metadata, (list, tuple)):
        return {}

    merged = {}
    for item in metadata:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            merged.setdefault(key, []).append(value)
    return merged


def _ensure_score_matrix(scores, labels, num_classes, device):
    scores = torch.as_tensor(scores, device=device, dtype=torch.float32)
    if scores.numel() == 0:
        return torch.zeros((0, num_classes), device=device, dtype=torch.float32)

    if scores.ndim == 2:
        if scores.size(1) == num_classes:
            return scores
        if scores.size(1) > num_classes:
            return scores[:, :num_classes]
        pad = torch.zeros((scores.size(0), num_classes - scores.size(1)), device=device, dtype=scores.dtype)
        return torch.cat([scores, pad], dim=1)

    scores = scores.reshape(-1)
    matrix = torch.zeros((scores.size(0), num_classes), device=device, dtype=scores.dtype)
    if labels is None:
        matrix[:, 0] = scores
        return matrix

    labels = torch.as_tensor(labels, device=device).reshape(-1).long()
    labels = labels.clamp(min=0, max=num_classes - 1)
    count = min(scores.size(0), labels.size(0))
    matrix[torch.arange(count, device=device), labels[:count]] = scores[:count]
    return matrix


def _from_result_dict(item, num_classes, device):
    boxes = item.get("boxes", torch.zeros((0, 4), device=device))
    labels = item.get("labels", None)
    scores = item.get("scores", None)

    boxes = torch.as_tensor(boxes, device=device, dtype=torch.float32).reshape(-1, 4)
    if scores is None:
        scores = torch.zeros((boxes.size(0), num_classes), device=device, dtype=torch.float32)
    scores = _ensure_score_matrix(scores, labels, num_classes, device)
    return boxes, scores


def _from_boxlist(item, num_classes, device):
    boxes = item.bbox.to(device=device, dtype=torch.float32).reshape(-1, 4)
    labels = item.get_field("labels") if item.has_field("labels") else None
    scores = item.get_field("scores") if item.has_field("scores") else None
    if scores is None:
        scores = torch.zeros((boxes.size(0), num_classes), device=device, dtype=torch.float32)
    scores = _ensure_score_matrix(scores, labels, num_classes, device)
    return boxes, scores


def _normalize_batch_outputs(outputs, batch_size, num_classes, device):
    if isinstance(outputs, tuple) and len(outputs) == 2:
        first, second = outputs
        if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
            if len(first) == batch_size and len(second) == batch_size:
                # Prefer [scores, boxes] if shapes look right (legacy baseline format).
                if len(first) > 0 and torch.is_tensor(first[0]) and first[0].ndim >= 1:
                    scores_list = [torch.as_tensor(s, device=device, dtype=torch.float32) for s in first]
                    boxes_list = [torch.as_tensor(b, device=device, dtype=torch.float32).reshape(-1, 4) for b in second]
                    pairs = []
                    for boxes, scores in zip(boxes_list, scores_list):
                        pairs.append((boxes, _ensure_score_matrix(scores, None, num_classes, device)))
                    return pairs

                boxes_list = [torch.as_tensor(b, device=device, dtype=torch.float32).reshape(-1, 4) for b in first]
                scores_list = [torch.as_tensor(s, device=device, dtype=torch.float32) for s in second]
                pairs = []
                for boxes, scores in zip(boxes_list, scores_list):
                    pairs.append((boxes, _ensure_score_matrix(scores, None, num_classes, device)))
                return pairs

    if isinstance(outputs, (list, tuple)) and len(outputs) == batch_size:
        pairs = []
        for item in outputs:
            if isinstance(item, dict):
                pairs.append(_from_result_dict(item, num_classes, device))
            elif isinstance(item, BoxList):
                pairs.append(_from_boxlist(item, num_classes, device))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                boxes = torch.as_tensor(item[0], device=device, dtype=torch.float32).reshape(-1, 4)
                scores = _ensure_score_matrix(item[1], None, num_classes, device)
                pairs.append((boxes, scores))
            else:
                empty_boxes = torch.zeros((0, 4), device=device, dtype=torch.float32)
                empty_scores = torch.zeros((0, num_classes), device=device, dtype=torch.float32)
                pairs.append((empty_boxes, empty_scores))
        return pairs

    raise RuntimeError("Unsupported model output format during inference.")


# --------------------------------------------------------
# IMAGE MODE
# --------------------------------------------------------
def compute_on_dataset_image(model, data_loader, device):
    """
    For image detection:
    model returns boxes and class scores/logits in one of multiple supported formats.
    """
    model.eval()
    results_dict = {}
    num_classes = max(1, int(getattr(data_loader.dataset, "num_classes", 1)))

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Image inference"):
            primary_inputs, secondary_inputs, whwh, boxes, _, metadata, idx = batch

            primary_inputs = primary_inputs.to(device)
            whwh = whwh.to(device)
            model_extras = _stack_metadata(metadata)

            outputs = model(
                primary_inputs,
                secondary_inputs,
                whwh,
                boxes=boxes,
                labels=None,
                extras=model_extras,
            )
            normalized = _normalize_batch_outputs(
                outputs,
                batch_size=len(idx),
                num_classes=num_classes,
                device=device,
            )

            for image_id, (box_tensor, score_tensor) in zip(idx, normalized):
                results_dict[image_id] = (box_tensor.cpu(), score_tensor.cpu())

    return results_dict


# --------------------------------------------------------
# MAIN INFERENCE
# --------------------------------------------------------
def inference(
        model,
        data_loader,
        dataset_name,
        output_folder=None,
        metric='image_ap',
        use_cache=False
):
    device = _resolve_model_device(model)
    dataset = data_loader.dataset
    logger = logging.getLogger("alphaction.inference")

    logger.info("Running inference in IMAGE mode")

    start_time = time.time()

    predictions = compute_on_dataset_image(model, data_loader, device)

    synchronize()

    total_time = time.time() - start_time
    logger.info(
        "Total inference time: {} ({} samples)".format(
            str(datetime.timedelta(seconds=total_time)),
            len(dataset)
        )
    )

    predictions = gather(predictions)
    if not is_main_process():
        return

    merged = {}
    for p in predictions:
        merged.update(p)

    return evaluate(
        dataset=dataset,
        predictions=list(merged.values()),
        output_folder=output_folder,
        metric=metric
    )
