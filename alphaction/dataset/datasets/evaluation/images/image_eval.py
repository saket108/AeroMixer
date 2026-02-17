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


def _prepare_for_multimodal_image_ap(predictions, dataset, score_thresh=0.0, text_prompts=None):
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
            gt_text_prompts = [class_text_map.get(label, f"class_{label}") for label in gt_labels]
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
            scores = torch.sigmoid(prediction[1]) if dataset.multilabel_action else F.softmax(prediction[1], dim=-1)
            scores = scores.numpy()
        else:
            scores = prediction[1]

        box_ids, class_ids = np.where(scores >= score_thresh)
        valid_ids = class_ids < dataset.num_classes
        box_ids = box_ids[valid_ids]
        class_ids = class_ids[valid_ids]
        
        if len(box_ids) == 0:
            results[image_key] = {
                "boxes": np.zeros((0, 4)), 
                "scores": np.array([]), 
                "action_ids": np.array([]),
                "text_prompts": np.array([])
            }
            continue

        # Add text prompts for predictions
        if class_text_map is not None:
            pred_text_prompts = [class_text_map.get(cid, f"class_{cid}") for cid in class_ids]
        else:
            pred_text_prompts = [f"class_{cid}" for cid in class_ids]

        results[image_key] = {
            "boxes": boxes[box_ids, :],
            "scores": scores[box_ids, class_ids],
            "action_ids": class_ids + 1,
            "text_prompts": pred_text_prompts,
        }

    return results, targets


def evaluate_with_text_prompts(predictions, dataset, text_prompts, output_folder=None, logger=None):
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
    results, targets = _prepare_for_multimodal_image_ap(predictions, dataset, text_prompts=text_prompts)
    
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


def do_multimodal_image_evaluation(dataset, predictions, output_folder, logger, metric="image_ap", 
                                   text_prompts=None, save_csv=False):
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
        raise NotImplementedError("Unsupported metric '{}' for ImageDataset.".format(metric))

    logger.info("Preparing multimodal image results for {} evaluation.".format(eval_metric))
    
    # Use text prompts if provided
    if text_prompts is not None:
        results, targets = _prepare_for_multimodal_image_ap(predictions, dataset, text_prompts=text_prompts)
    else:
        results, targets = _prepare_for_image_ap(predictions, dataset)
    
    eval_res = frame_mAP_pascal(
        results,
        targets,
        dataset.closed_set_classes,
        logger,
        iou_list=[dataset.test_iou_thresh],
    )

    logger.info("Multimodal evaluation results ({}):\n{}".format(eval_metric, pformat(eval_res, indent=2)))
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        log_file_path = os.path.join(output_folder, "result_multimodal_image.log")
        with open(log_file_path, "w") as logf:
            logf.write("Multimodal evaluation results (metric: {})\n".format(eval_metric))
            logf.write(pformat(eval_res))

    return eval_res, results
