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

"""Pascal evaluation utilities for image and multimodal detection."""

from . import label_map_util
from . import metrics
from . import np_box_list
from . import np_box_list_ops

# Export functions from label_map_util
__all__ = [
    'label_map_util',
    'metrics', 
    'np_box_list',
    'np_box_list_ops',
]

# Export key functions
__all__.extend([
    'create_category_index',
    'create_category_index_with_text',
    'get_label_map_dict',
    'get_label_map_dict_with_text',
    'create_category_index_from_labelmap',
    'create_category_index_from_labelmap_with_text',
    'create_class_agnostic_category_index',
    'create_class_agnostic_category_index_with_text',
])

__all__.extend([
    'compute_precision_recall',
    'compute_average_precision',
    'compute_cor_loc',
    'compute_text_similarity_scores',
    'compute_multimodal_precision_recall',
    'compute_text_aware_average_precision',
    'compute_open_vocabulary_metrics',
])

__all__.extend([
    'BoxList',
])

__all__.extend([
    'area',
    'intersection',
    'iou',
    'ioa',
    'gather',
    'sort_by_field',
    'non_max_suppression',
    'multi_class_non_max_suppression',
    'scale',
    'clip_to_window',
    'prune_non_overlapping_boxes',
    'prune_outside_window',
    'concatenate',
    'filter_scores_greater_than',
    'change_coordinate_frame',
    'filter_by_text_similarity',
    'sort_by_text_similarity',
])
