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
"""Label map utility functions for image and multimodal detection."""

import logging
logger = logging.getLogger("alphaction.inference")


def _validate_label_map(label_map):
  """Checks if a label map is valid.

  Args:
    label_map: StringIntLabelMap to validate.

  Raises:
    ValueError: if label map is invalid.
  """
  for item in label_map.item:
    if item.id < 1:
      raise ValueError('Label map ids should be >= 1.')


def create_category_index(categories):
  """Creates dictionary of COCO compatible categories keyed by category id.

  Args:
    categories: a list of dicts, each of which has the following keys:
      'id': (required) an integer id uniquely identifying this category.
      'name': (required) string representing category name
        e.g., 'cat', 'dog', 'pizza'.

  Returns:
    category_index: a dict containing the same entries as categories, but keyed
      by the 'id' field of each category.
  """
  category_index = {}
  for cat in categories:
    category_index[cat['id']] = cat
  return category_index


def create_category_index_with_text(categories, text_prompts=None):
  """Creates dictionary of categories keyed by category id with text prompts.

  Args:
    categories: a list of dicts, each of which has the following keys:
      'id': (required) an integer id uniquely identifying this category.
      'name': (required) string representing category name
        e.g., 'cat', 'dog', 'pizza'.
    text_prompts: Optional dict mapping category id to text prompts

  Returns:
    category_index: a dict containing the same entries as categories, but keyed
      by the 'id' field of each category, with added 'text_prompt' field.
  """
  category_index = {}
  for cat in categories:
    cat_with_text = cat.copy()
    if text_prompts and cat['id'] in text_prompts:
      cat_with_text['text_prompt'] = text_prompts[cat['id']]
    else:
      # Generate default text prompt
      cat_with_text['text_prompt'] = f"a photo of {cat['name']}"
    category_index[cat['id']] = cat_with_text
  return category_index


def get_max_label_map_index(label_map):
  """Get maximum index in label map.

  Args:
    label_map: a StringIntLabelMapProto

  Returns:
    an integer
  """
  return max([item.id for item in label_map.item])


def convert_label_map_to_categories(label_map,
                                    max_num_classes,
                                    use_display_name=True):
  """Loads label map proto and returns categories list compatible with eval.

  This function loads a label map and returns a list of dicts, each of which
  has the following keys:
    'id': (required) an integer id uniquely identifying this category.
    'name': (required) string representing category name
      e.g., 'cat', 'dog', 'pizza'.
  We only allow class into the list if its id-label_id_offset is
  between 0 (inclusive) and max_num_classes (exclusive).
  If there are several items mapping to the same id in the label map,
  we will only keep the first one in the categories list.

  Args:
    label_map: a StringIntLabelMapProto or None.  If None, a default categories
      list is created with max_num_classes categories.
    max_num_classes: maximum number of (consecutive) label indices to include.
    use_display_name: (boolean) choose whether to load 'display_name' field
      as category name.  If False or if the display_name field does not exist,
      uses 'name' field as category names instead.
  Returns:
    categories: a list of dictionaries representing all possible categories.
  """
  categories = []
  list_of_ids_already_added = []
  if not label_map:
    label_id_offset = 1
    for class_id in range(max_num_classes):
      categories.append({
          'id': class_id + label_id_offset,
          'name': 'category_{}'.format(class_id + label_id_offset)
      })
    return categories
  for item in label_map.item:
    if not 0 < item.id <= max_num_classes:
      logger.info('Ignore item %d since it falls outside of requested '
                   'label range.', item.id)
      continue
    if use_display_name and item.HasField('display_name'):
      name = item.display_name
    else:
      name = item.name
    if item.id not in list_of_ids_already_added:
      list_of_ids_already_added.append(item.id)
      categories.append({'id': item.id, 'name': name})
  return categories


def load_labelmap(path):
  """Loads label map proto.

  Args:
    path: path to StringIntLabelMap proto text file.
  Returns:
    a StringIntLabelMapProto
  """
  with open(path, 'r') as fid:
    label_map_string = fid.read()
    label_map = string_int_label_map_pb2.StringIntLabelMap()
    try:
      text_format.Merge(label_map_string, label_map)
    except text_format.ParseError:
      label_map.ParseFromString(label_map_string)
  _validate_label_map(label_map)
  return label_map


def get_label_map_dict(label_map_path, use_display_name=False):
  """Reads a label map and returns a dictionary of label names to id.

  Args:
    label_map_path: path to label_map.
    use_display_name: whether to use the label map items' display names as keys.

  Returns:
    A dictionary mapping label names to id.
  """
  label_map = load_labelmap(label_map_path)
  label_map_dict = {}
  for item in label_map.item:
    if use_display_name:
      label_map_dict[item.display_name] = item.id
    else:
      label_map_dict[item.name] = item.id
  return label_map_dict


def get_label_map_dict_with_text(label_map_path, prompt_template="a photo of {}", use_display_name=False):
  """Reads a label map and returns a dictionary of label names to id with text prompts.

  Args:
    label_map_path: path to label_map.
    prompt_template: Template for text prompts (use {} as placeholder for class name)
    use_display_name: whether to use the label map items' display names as keys.

  Returns:
    A tuple of:
      - label_map_dict: Dictionary mapping label names to id
      - text_prompts_dict: Dictionary mapping class id to text prompts
  """
  label_map = load_labelmap(label_map_path)
  label_map_dict = {}
  text_prompts_dict = {}
  
  for item in label_map.item:
    if use_display_name:
      name = item.display_name
    else:
      name = item.name
    
    label_map_dict[name] = item.id
    text_prompts_dict[item.id] = prompt_template.format(name)
  
  return label_map_dict, text_prompts_dict


def create_category_index_from_labelmap(label_map_path):
  """Reads a label map and returns a category index.

  Args:
    label_map_path: Path to `StringIntLabelMap` proto text file.

  Returns:
    A category index, which is a dictionary that maps integer ids to dicts
    containing categories, e.g.
    {1: {'id': 1, 'name': 'dog'}, 2: {'id': 2, 'name': 'cat'}, ...}
  """
  label_map = load_labelmap(label_map_path)
  max_num_classes = max(item.id for item in label_map.item)
  categories = convert_label_map_to_categories(label_map, max_num_classes)
  return create_category_index(categories)


def create_category_index_from_labelmap_with_text(label_map_path, prompt_template="a photo of {}"):
  """Reads a label map and returns a category index with text prompts.

  Args:
    label_map_path: Path to `StringIntLabelMap` proto text file.
    prompt_template: Template for text prompts

  Returns:
    A category index with text prompts, which is a dictionary that maps 
    integer ids to dicts containing categories and text prompts, e.g.
    {1: {'id': 1, 'name': 'dog', 'text_prompt': 'a photo of dog'}, ...}
  """
  label_map = load_labelmap(label_map_path)
  max_num_classes = max(item.id for item in label_map.item)
  categories = convert_label_map_to_categories(label_map, max_num_classes)
  
  # Create text prompts
  text_prompts = {}
  for item in label_map.item:
    if use_display_name:
      name = item.display_name
    else:
      name = item.name
    text_prompts[item.id] = prompt_template.format(name)
  
  return create_category_index_with_text(categories, text_prompts)


def create_class_agnostic_category_index():
  """Creates a category index with a single `object` class."""
  return {1: {'id': 1, 'name': 'object'}}


def create_class_agnostic_category_index_with_text():
  """Creates a category index with a single `object` class and text prompt."""
  return {1: {'id': 1, 'name': 'object', 'text_prompt': 'a photo of object'}}
