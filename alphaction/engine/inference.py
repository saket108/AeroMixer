import logging
import os
import torch
from tqdm import tqdm
import time
import datetime

from alphaction.dataset.datasets.evaluation import evaluate
from alphaction.utils.comm import get_rank, is_main_process, gather, synchronize, get_world_size


def _resolve_model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------
# IMAGE MODE
# --------------------------------------------------------
def compute_on_dataset_image(model, data_loader, device):
    """
    For image detection:
    model returns: boxes + class logits
    """
    model.eval()
    results_dict = {}

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Image inference"):
            primary_inputs, secondary_inputs, whwh, boxes, _, metadata, idx = batch

            primary_inputs = primary_inputs.to(device)
            whwh = whwh.to(device)

            pred_scores, pred_boxes = model(primary_inputs, None, whwh)

            for image_id, b, s in zip(idx, pred_boxes, pred_scores):
                results_dict[image_id] = (b.cpu(), s.cpu())

    return results_dict


# --------------------------------------------------------
# VIDEO MODE (original behavior)
# --------------------------------------------------------
def compute_on_dataset_video(model, data_loader, device):
    model.eval()
    results_dict = {}

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Video inference"):
            primary_inputs, secondary_inputs, whwh, boxes, _, metadata, idx = batch

            primary_inputs = primary_inputs.to(device)
            if secondary_inputs is not None:
                secondary_inputs = secondary_inputs.to(device)
            whwh = whwh.to(device)

            action_score_list, box_list = model(primary_inputs, secondary_inputs, whwh)

            for video_id, box, action_score in zip(idx, box_list, action_score_list):
                results_dict[video_id] = (box.cpu(), action_score.cpu())

    return results_dict


# --------------------------------------------------------
# MAIN INFERENCE
# --------------------------------------------------------
def inference(
        model,
        data_loader,
        dataset_name,
        output_folder=None,
        metric='frame_ap',
        use_cache=False
):
    device = _resolve_model_device(model)
    dataset = data_loader.dataset
    logger = logging.getLogger("alphaction.inference")

    input_type = getattr(dataset, "input_type", "video").lower()

    logger.info(f"Inference mode detected: {input_type.upper()}")

    start_time = time.time()

    if input_type == "image":
        predictions = compute_on_dataset_image(model, data_loader, device)
        metric = "image_ap"   # override metric
    else:
        predictions = compute_on_dataset_video(model, data_loader, device)

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
