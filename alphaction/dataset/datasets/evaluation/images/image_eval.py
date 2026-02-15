import os
from pprint import pformat

import numpy as np
import torch
import torch.nn.functional as F

from ..pascal_wrapper import frame_mAP_pascal


def _make_image_key(image_rel):
    return str(image_rel)


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
        scores = torch.sigmoid(prediction[1]) if dataset.multilabel_action else F.softmax(prediction[1], dim=-1)
        scores = scores.numpy()

        box_ids, class_ids = np.where(scores >= score_thresh)
        valid_ids = class_ids < dataset.num_classes
        box_ids = box_ids[valid_ids]
        class_ids = class_ids[valid_ids]
        if len(box_ids) == 0:
            results[image_key] = {"boxes": np.zeros((0, 4)), "scores": np.array([]), "action_ids": np.array([])}
            continue

        results[image_key] = {
            "boxes": boxes[box_ids, :],
            "scores": scores[box_ids, class_ids],
            "action_ids": class_ids + 1,
        }

    return results, targets


def do_image_evaluation(dataset, predictions, output_folder, logger, metric="image_ap", save_csv=False):
    del save_csv  # not used for image evaluation
    eval_metric = metric.lower()
    if eval_metric not in ["image_ap", "frame_ap"]:
        raise NotImplementedError("Unsupported metric '{}' for ImageDataset.".format(metric))

    logger.info("Preparing image results for {} evaluation.".format(eval_metric))
    results, targets = _prepare_for_image_ap(predictions, dataset)
    eval_res = frame_mAP_pascal(
        results,
        targets,
        dataset.closed_set_classes,
        logger,
        iou_list=[dataset.test_iou_thresh],
    )

    logger.info("Evaluation results ({}):\n{}".format(eval_metric, pformat(eval_res, indent=2)))
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        log_file_path = os.path.join(output_folder, "result_image.log")
        with open(log_file_path, "w") as logf:
            logf.write("Evaluation results (metric: {})\n".format(eval_metric))
            logf.write(pformat(eval_res))

    return eval_res, results
