# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for computing metrics like precision, recall, CorLoc and etc."""
from __future__ import division

import numpy as np


def compute_precision_recall(scores, labels, num_gt):
  """Compute precision and recall.

  Args:
    scores: A float numpy array representing detection score
    labels: A boolean numpy array representing true/false positive labels
    num_gt: Number of ground truth instances

  Raises:
    ValueError: if the input is not of the correct format

  Returns:
    precision: Fraction of positive instances over detected ones. This value is
      None if no ground truth labels are present.
    recall: Fraction of detected positive instance over all positive instances.
      This value is None if no ground truth labels are present.

  """
  if not isinstance(
      labels, np.ndarray) or labels.dtype != np.bool_ or len(labels.shape) != 1:
    raise ValueError("labels must be single dimension bool numpy array")

  if not isinstance(
      scores, np.ndarray) or len(scores.shape) != 1:
    raise ValueError("scores must be single dimension numpy array")

  if num_gt < np.sum(labels):
    raise ValueError("Number of true positives must be smaller than num_gt.")

  if len(scores) != len(labels):
    raise ValueError("scores and labels must be of the same size.")

  if num_gt == 0:
    return None, None

  sorted_indices = np.argsort(scores)
  sorted_indices = sorted_indices[::-1]
  labels = labels.astype(int)
  true_positive_labels = labels[sorted_indices]
  false_positive_labels = 1 - true_positive_labels
  cum_true_positives = np.cumsum(true_positive_labels)
  cum_false_positives = np.cumsum(false_positive_labels)
  precision = cum_true_positives.astype(float) / (
      cum_true_positives + cum_false_positives)
  recall = cum_true_positives.astype(float) / num_gt
  return precision, recall


def compute_average_precision(precision, recall):
  """Compute Average Precision according to the definition in VOCdevkit.

  Precision is modified to ensure that it does not decrease as recall
  decrease.

  Args:
    precision: A float [N, 1] numpy array of precisions
    recall: A float [N, 1] numpy array of recalls

  Raises:
    ValueError: if the input is not of the correct format

  Returns:
    average_precison: The area under the precision recall curve. NaN if
      precision and recall are None.

  """
  if precision is None:
    if recall is not None:
      raise ValueError("If precision is None, recall must also be None")
    return np.NAN

  if not isinstance(precision, np.ndarray) or not isinstance(recall,
                                                             np.ndarray):
    raise ValueError("precision and recall must be numpy array")
  if not np.issubdtype(precision.dtype, np.floating) or not np.issubdtype(recall.dtype, np.floating):
    raise ValueError("input must be float numpy array.")
  if len(precision) != len(recall):
    raise ValueError("precision and recall must be of the same size.")
  if not precision.size:
    return 0.0
  if np.amin(precision) < 0 or np.amax(precision) > 1:
    raise ValueError("Precision must be in the range of [0, 1].")
  if np.amin(recall) < 0 or np.amax(recall) > 1:
    raise ValueError("recall must be in the range of [0, 1].")
  if not all(recall[i] <= recall[i + 1] for i in range(len(recall) - 1)):
    raise ValueError("recall must be a non-decreasing array")

  recall = np.concatenate([[0], recall, [1]])
  precision = np.concatenate([[0], precision, [0]])

  # Preprocess precision to be a non-decreasing array
  for i in range(len(precision) - 2, -1, -1):
    precision[i] = np.maximum(precision[i], precision[i + 1])

  indices = np.where(recall[1:] != recall[:-1])[0] + 1
  average_precision = np.sum(
      (recall[indices] - recall[indices - 1]) * precision[indices])
  return average_precision


def compute_cor_loc(num_gt_imgs_per_class,
                    num_images_correctly_detected_per_class):
  """Compute CorLoc according to the definition in the following paper.

  https://www.robots.ox.ac.uk/~vgg/rg/papers/deselaers-eccv10.pdf

  Returns nans if there are no ground truth images for a class.

  Args:
    num_gt_imgs_per_class: 1D array, representing number of images containing
        at least one object instance of a particular class
    num_images_correctly_detected_per_class: 1D array, representing number of
        images that are correctly detected at least one object instance of a
        particular class

  Returns:
    corloc_per_class: A float numpy array represents the corloc score of each
      class
  """
  # Divide by zero expected for classes with no gt examples.
  with np.errstate(divide="ignore", invalid="ignore"):
    return np.where(
        num_gt_imgs_per_class == 0, np.nan,
        num_images_correctly_detected_per_class / num_gt_imgs_per_class)


# ============================================================================
# Multimodal (Image + Text) Metrics
# ============================================================================

def compute_text_similarity_scores(image_features, text_features):
  """Compute cosine similarity between image and text features.

  Args:
    image_features: A float numpy array of shape [N, D] representing image features
    text_features: A float numpy array of shape [M, D] representing text features

  Returns:
    similarity_scores: A float numpy array of shape [N, M] representing 
      cosine similarity between image and text features
  """
  # Normalize features
  image_norm = np.linalg.norm(image_features, axis=1, keepdims=True)
  text_norm = np.linalg.norm(text_features, axis=1, keepdims=True)
  
  # Avoid division by zero
  image_features_norm = image_features / (image_norm + 1e-8)
  text_features_norm = text_features / (text_norm + 1e-8)
  
  # Compute cosine similarity
  similarity_scores = np.dot(image_features_norm, text_features_norm.T)
  
  return similarity_scores


def compute_multimodal_precision_recall(image_scores, text_scores, labels, num_gt, alpha=0.5):
  """Compute precision and recall for multimodal (image + text) detection.

  Args:
    image_scores: A float numpy array representing image-based detection scores
    text_scores: A float numpy array representing text similarity scores
    labels: A boolean numpy array representing true/false positive labels
    num_gt: Number of ground truth instances
    alpha: Weight for combining image and text scores (0 = text only, 1 = image only)

  Raises:
    ValueError: if the input is not of the correct format

  Returns:
    precision: Fraction of positive instances over detected ones
    recall: Fraction of detected positive instance over all positive instances
  """
  if not isinstance(labels, np.ndarray) or labels.dtype != np.bool_ or len(labels.shape) != 1:
    raise ValueError("labels must be single dimension bool numpy array")

  if not isinstance(image_scores, np.ndarray) or len(image_scores.shape) != 1:
    raise ValueError("image_scores must be single dimension numpy array")
  
  if not isinstance(text_scores, np.ndarray) or len(text_scores.shape) != 1:
    raise ValueError("text_scores must be single dimension numpy array")

  if num_gt < np.sum(labels):
    raise ValueError("Number of true positives must be smaller than num_gt.")

  if len(image_scores) != len(labels) or len(text_scores) != len(labels):
    raise ValueError("scores and labels must be of the same size.")

  if num_gt == 0:
    return None, None

  # Combine image and text scores
  combined_scores = alpha * image_scores + (1 - alpha) * text_scores

  sorted_indices = np.argsort(combined_scores)
  sorted_indices = sorted_indices[::-1]
  labels = labels.astype(int)
  true_positive_labels = labels[sorted_indices]
  false_positive_labels = 1 - true_positive_labels
  cum_true_positives = np.cumsum(true_positive_labels)
  cum_false_positives = np.cumsum(false_positive_labels)
  precision = cum_true_positives.astype(float) / (
      cum_true_positives + cum_false_positives)
  recall = cum_true_positives.astype(float) / num_gt
  return precision, recall


def compute_text_aware_average_precision(image_scores, text_scores, labels, num_gt, alpha=0.5):
  """Compute Average Precision for multimodal (image + text) detection.

  Args:
    image_scores: A float numpy array representing image-based detection scores
    text_scores: A float numpy array representing text similarity scores
    labels: A boolean numpy array representing true/false positive labels
    num_gt: Number of ground truth instances
    alpha: Weight for combining image and text scores

  Returns:
    average_precision: The area under the precision recall curve
  """
  precision, recall = compute_multimodal_precision_recall(
      image_scores, text_scores, labels, num_gt, alpha)
  
  if precision is None:
    return np.NAN
  
  return compute_average_precision(precision, recall)


def compute_open_vocabulary_metrics(detection_scores, detection_text_prompts, 
                                   gt_labels, gt_text_prompts, num_gt):
  """Compute metrics for open vocabulary detection.

  Args:
    detection_scores: Detection scores for each box
    detection_text_prompts: Text prompts for each detection
    gt_labels: Ground truth labels
    gt_text_prompts: Ground truth text prompts
    num_gt: Number of ground truth instances

  Returns:
    metrics: Dictionary containing precision, recall, and AP
  """
  # Match detections to ground truth based on text similarity
  matched = np.zeros(len(detection_scores), dtype=bool)
  labels = np.zeros(len(detection_scores), dtype=bool)
  
  for gt_idx, gt_prompt in enumerate(gt_text_prompts):
    best_match_idx = -1
    best_similarity = -1
    
    for det_idx, det_prompt in enumerate(detection_text_prompts):
      if matched[det_idx]:
        continue
      
      # Simple string matching (can be replaced with semantic similarity)
      if gt_prompt.lower() == det_prompt.lower():
        similarity = 1.0
      else:
        # Check if detection prompt contains gt prompt or vice versa
        if gt_prompt.lower() in det_prompt.lower() or det_prompt.lower() in gt_prompt.lower():
          similarity = 0.8
        else:
          similarity = 0.0
      
      if similarity > best_similarity:
        best_similarity = similarity
        best_match_idx = det_idx
    
    if best_match_idx >= 0 and best_similarity > 0.5:
      matched[best_match_idx] = True
      if best_similarity >= 0.8:
        labels[best_match_idx] = True
  
  # Compute precision-recall
  precision, recall = compute_precision_recall(detection_scores, labels, num_gt)
  
  metrics = {
    'precision': precision,
    'recall': recall,
    'average_precision': compute_average_precision(precision, recall) if precision is not None else np.NAN
  }
  
  return metrics
