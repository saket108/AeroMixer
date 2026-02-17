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

"""Numpy BoxMaskList classes and functions for image and multimodal detection."""

import numpy as np
from . import np_box_list


class BoxMaskList(np_box_list.BoxList):
  """Convenience wrapper for BoxList with masks.

  BoxMaskList extends the np_box_list.BoxList to contain masks as well.
  In particular, its constructor receives both boxes and masks. Note that the
  masks correspond to the full image.
  
  For multimodal detection, also supports text prompts and text features.
  """

  def __init__(self, box_data, mask_data):
    """Constructs box collection.

    Args:
      box_data: a numpy array of shape [N, 4] representing box coordinates
      mask_data: a numpy array of shape [N, height, width] representing masks
        with values are in {0,1}. The masks correspond to the full
        image. The height and the width will be equal to image height and width.

    Raises:
      ValueError: if bbox dataset is not a numpy array
      ValueError: if invalid dimensions for bbox dataset
      ValueError: if mask dataset is not a numpy array
      ValueError: if invalid dimension for mask dataset
    """
    super(BoxMaskList, self).__init__(box_data)
    if not isinstance(mask_data, np.ndarray):
      raise ValueError('Mask dataset must be a numpy array.')
    if len(mask_data.shape) != 3:
      raise ValueError('Invalid dimensions for mask dataset.')
    if mask_data.dtype != np.uint8:
      raise ValueError('Invalid dataset type for mask dataset: uint8 is required.')
    if mask_data.shape[0] != box_data.shape[0]:
      raise ValueError('There should be the same number of boxes and masks.')
    self.data['masks'] = mask_data

  def get_masks(self):
    """Convenience function for accessing masks.

    Returns:
      a numpy array of shape [N, height, width] representing masks
    """
    return self.get_field('masks')

  # ============================================================================
  # Multimodal (Image + Text) Support
  # ============================================================================

  def add_text_prompts(self, text_prompts):
    """Add text prompts for each bounding box.
    
    This is useful for open vocabulary detection where each detection
    is associated with a text prompt.
    
    Args:
      text_prompts: List of strings, each string is the text prompt for a box
      
    Raises:
      ValueError: if length doesn't match number of boxes
    """
    if len(text_prompts) != self.num_boxes():
      raise ValueError(f"Length of text_prompts ({len(text_prompts)}) must match "
                       f"number of boxes ({self.num_boxes()})")
    self.data['text_prompts'] = np.array(text_prompts)

  def get_text_prompts(self):
    """Get text prompts for each bounding box.
    
    Returns:
      numpy array of text prompts
    """
    if not self.has_field('text_prompts'):
      return None
    return self.get_field('text_prompts')

  def add_text_features(self, text_features):
    """Add text features for each bounding box.
    
    This is useful for storing CLIP or other text encoder features
    for each detection.
    
    Args:
      text_features: numpy array of shape [N, D] where D is the feature dimension
    """
    if len(text_features) != self.num_boxes():
      raise ValueError(f"Length of text_features ({len(text_features)}) must match "
                       f"number of boxes ({self.num_boxes()})")
    self.data['text_features'] = text_features

  def get_text_features(self):
    """Get text features for each bounding box.
    
    Returns:
      numpy array of text features or None
    """
    if not self.has_field('text_features'):
      return None
    return self.get_field('text_features')

  def has_text_features(self):
    """Check if text features are available.
    
    Returns:
      bool: True if text features are available
    """
    return self.has_field('text_features')
