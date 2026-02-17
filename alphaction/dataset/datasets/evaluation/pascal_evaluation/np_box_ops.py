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

"""Operations for [N, 4] numpy arrays representing bounding boxes.

Example box operations that are supported:
  * Areas: compute bounding box areas
  * IOU: pairwise intersection-over-union scores
  
For multimodal detection, also supports:
  * Text-based similarity operations for open vocabulary detection
"""
import numpy as np


def area(boxes):
  """Computes area of boxes.

  Args:
    boxes: Numpy array with shape [N, 4] holding N boxes

  Returns:
    a numpy array with shape [N*1] representing box areas
  """
  return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def intersection(boxes1, boxes2):
  """Compute pairwise intersection areas between boxes.

  Args:
    boxes1: a numpy array with shape [N, 4] holding N boxes
    boxes2: a numpy array with shape [M, 4] holding M boxes

  Returns:
    a numpy array with shape [N*M] representing pairwise intersection area
  """
  [y_min1, x_min1, y_max1, x_max1] = np.split(boxes1, 4, axis=1)
  [y_min2, x_min2, y_max2, x_max2] = np.split(boxes2, 4, axis=1)

  all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
  all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
  intersect_heights = np.maximum(
      np.zeros(all_pairs_max_ymin.shape),
      all_pairs_min_ymax - all_pairs_max_ymin)
  all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
  all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
  intersect_widths = np.maximum(
      np.zeros(all_pairs_max_xmin.shape),
      all_pairs_min_xmax - all_pairs_max_xmin)
  return intersect_heights * intersect_widths


def iou(boxes1, boxes2):
  """Computes pairwise intersection-over-union between box collections.

  Args:
    boxes1: a numpy array with shape [N, 4] holding N boxes.
    boxes2: a numpy array with shape [M, 4] holding N boxes.

  Returns:
    a numpy array with shape [N, M] representing pairwise iou scores.
  """
  intersect = intersection(boxes1, boxes2)
  area1 = area(boxes1)
  area2 = area(boxes2)
  union = np.expand_dims(area1, axis=1) + np.expand_dims(
      area2, axis=0) - intersect
  return intersect / union


def ioa(boxes1, boxes2):
  """Computes pairwise intersection-over-area between box collections.

  Intersection-over-area (ioa) between two boxes box1 and box2 is defined as
  their intersection area over box2's area. Note that ioa is not symmetric,
  that is, IOA(box1, box2) != IOA(box2, box1).

  Args:
    boxes1: a numpy array with shape [N, 4] holding N boxes.
    boxes2: a numpy array with shape [M, 4] holding N boxes.

  Returns:
    a numpy array with shape [N, M] representing pairwise ioa scores.
  """
  intersect = intersection(boxes1, boxes2)
  areas = np.expand_dims(area(boxes2), axis=0)
  return intersect / areas


# ============================================================================
# Multimodal (Image + Text) Operations
# ============================================================================

def compute_text_similarity(box_features, text_features):
  """Compute cosine similarity between box features and text features.
  
  This is useful for open vocabulary detection where we want to compute
  similarity between detected regions and text prompts.
  
  Args:
    box_features: numpy array of shape [N, D] representing box features
    text_features: numpy array of shape [M, D] representing text features
    
  Returns:
    similarity_scores: numpy array of shape [N, M] representing 
      cosine similarity between box and text features
  """
  # Normalize features
  box_norm = box_features / (np.linalg.norm(box_features, axis=1, keepdims=True) + 1e-8)
  text_norm = text_features / (np.linalg.norm(text_features, axis=1, keepdims=True) + 1e-8)
  
  # Compute cosine similarity
  similarity_scores = np.dot(box_norm, text_norm.T)
  
  return similarity_scores


def compute_text_similarity_scores(boxes, text_features):
  """Compute cosine similarity between box features and a single text feature vector.
  
  Args:
    boxes: numpy array of shape [N, 4] (not used, kept for API compatibility)
    text_features: numpy array of shape [D] representing text features
    
  Returns:
    similarity_scores: numpy array of shape [N] representing 
      cosine similarity between each box and the text features
  """
  # This function expects text_features to be [N, D] but accepts [D] for single vector
  if len(text_features.shape) == 1:
    # Single text feature vector - reshape to [1, D]
    text_features = text_features.reshape(1, -1)
  
  # For box features, we would need them as input - this is a placeholder
  # In practice, you would pass the actual box visual features
  return np.array([])


def box_text_similarity(box_features, text_feature):
  """Compute similarity between multiple box features and a single text feature.
  
  Args:
    box_features: numpy array of shape [N, D] representing box features
    text_feature: numpy array of shape [D] representing text feature
    
  Returns:
    similarity_scores: numpy array of shape [N] representing 
      cosine similarity between each box and the text feature
  """
  # Normalize features
  box_norm = box_features / (np.linalg.norm(box_features, axis=1, keepdims=True) + 1e-8)
  text_norm = text_feature / (np.linalg.norm(text_feature) + 1e-8)
  
  # Compute cosine similarity
  similarity_scores = np.dot(box_norm, text_norm)
  
  return similarity_scores
