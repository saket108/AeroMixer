#!/usr/bin/env python3
"""
Utility functions for dataset loading and processing.

Supports multimodal (image + text) operations for open vocabulary detection.
"""

import logging
import numpy as np
import time
import cv2
import torch
from iopath.common.file_io import g_pathmgr
import os
import pickle
import json


logger = logging.getLogger(__name__)


def retry_load_images(image_paths, retry=10, backend="pytorch"):
    """
    This function is to load images with support of retrying for failed load.
    Args:
        image_paths (list): paths of images needed to be loaded.
        retry (int, optional): maximum time of loading retrying. Defaults to 10.
        backend (str): `pytorch` or `cv2`.
    Returns:
        imgs (list): list of loaded images.
    """
    for i in range(retry):
        imgs = []
        for image_path in image_paths:
            with g_pathmgr.open(image_path, "rb") as f:
                img_str = np.frombuffer(f.read(), np.uint8)
                img = cv2.imdecode(img_str, flags=cv2.IMREAD_COLOR)
            imgs.append(img)

        if all(img is not None for img in imgs):
            if backend == "pytorch":
                imgs = torch.as_tensor(np.stack(imgs))
            return imgs
        else:
            logger.warn("Reading failed. Will retry.")
            time.sleep(1.0)
        if i == retry - 1:
            raise Exception("Failed to load images {}".format(image_paths))


def read_greyscale_image(img_file):
    assert os.path.exists(img_file), "File does not exist!\n{}".format(img_file)
    im = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    im = im.astype(np.float32) / 255.0
    im = torch.from_numpy(im)
    return im


def get_sequence(center_idx, half_len, sample_rate, num_frames):
    """
    Sample frames among the corresponding clip.
    Args:
        center_idx (int): center frame idx for current clip
        half_len (int): half of the clip length
        sample_rate (int): sampling rate for sampling frames inside of the clip
        num_frames (int): number of expected sampled frames
    Returns:
        seq (list): list of indexes of sampled frames in this clip.
    """
    seq = list(range(center_idx - half_len, center_idx + half_len, sample_rate))

    for seq_idx in range(len(seq)):
        if seq[seq_idx] < 0:
            seq[seq_idx] = 0
        elif seq[seq_idx] >= num_frames:
            seq[seq_idx] = num_frames - 1
    return seq

def pack_pathway_output(cfg, frames, pathways=2):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
    Returns:
        frame_list (list): list of tensors with the dimension of
            `channel` x `num frames` x `height` x `width`.
    """
    if cfg.DATA.REVERSE_INPUT_CHANNEL:
        frames = frames[[2, 1, 0], :, :, :]
    if pathways==1:
        frame_list = [frames]
    elif pathways==2:
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // cfg.SLOWFAST.ALPHA
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
    else:
        raise NotImplementedError()
    return frame_list


def load_dets_data(det_file, topk=None):
    assert os.path.exists(det_file), "detection file does not exist: {}".format(det_file)
    with open(det_file, 'rb') as fid:
        data = pickle.load(fid, encoding='iso-8859-1')
    # get list of all frames
    all_dets = dict()
    for vid, dets in data.items():
        for i in list(dets['boxes'].keys()):
            boxes, scores = dets['boxes'][i], dets['scores'][i]
            key = "%s,%05d" % (vid, i)
            if topk is None:
                all_dets[key] = np.hstack((boxes, scores[:, None]))  # (n, 5)
            else:
                indices = np.argsort(scores)[::-1][:topk]  # topK maximum indices
                all_dets[key] = np.hstack((boxes[indices], scores[indices, None]))  # (n, 5)
    return all_dets


# ============================================================================
# Multimodal (Image + Text) Utility Functions
# ============================================================================

def load_text_features(text_features_file, device='cpu'):
    """Load precomputed text features from file.
    
    Args:
        text_features_file: Path to text features file (.npy or .pt)
        device: Device to load features to
        
    Returns:
        numpy array or tensor of text features [num_classes, feature_dim]
    """
    if not os.path.exists(text_features_file):
        logger.warning(f"Text features file not found: {text_features_file}")
        return None
    
    ext = os.path.splitext(text_features_file)[1].lower()
    
    if ext == '.npy':
        text_features = np.load(text_features_file)
        return torch.from_numpy(text_features).to(device) if isinstance(text_features, np.ndarray) else text_features
    elif ext == '.pt':
        return torch.load(text_features_file, map_location=device)
    else:
        logger.warning(f"Unsupported text features format: {ext}")
        return None


def compute_text_similarity(image_features, text_features, normalize=True):
    """Compute similarity between image and text features.
    
    Args:
        image_features: Image features tensor [B, D] or [D]
        text_features: Text features tensor [N, D] or [D]
        normalize: Whether to normalize features before computing similarity
        
    Returns:
        Similarity scores tensor
    """
    if normalize:
        # L2 normalize
        if image_features.dim() == 2:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        else:
            image_features = image_features / image_features.norm()
            
        if text_features.dim() == 2:
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        else:
            text_features = text_features / text_features.norm()
    
    # Compute cosine similarity
    if image_features.dim() == 1:
        return torch.dot(image_features, text_features)
    else:
        return torch.mm(image_features, text_features.t())


def create_text_prompts(class_names, prompt_template="a photo of {}"):
    """Create text prompts for each class.
    
    Args:
        class_names: List of class names
        prompt_template: Template for prompt generation
        
    Returns:
        List of text prompts
    """
    prompts = []
    for class_name in class_names:
        prompt = prompt_template.format(class_name)
        prompts.append(prompt)
    return prompts


def load_multimodal_annotations(ann_file, include_text_prompts=True):
    """Load annotations including text prompts for multimodal detection.
    
    Args:
        ann_file: Path to annotation file
        include_text_prompts: Whether to include text prompts in loading
        
    Returns:
        Dictionary of annotations with optional text prompts
    """
    if not os.path.exists(ann_file):
        logger.warning(f"Annotation file not found: {ann_file}")
        return {}
    
    ext = os.path.splitext(ann_file)[1].lower()
    
    if ext == '.json':
        with open(ann_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    elif ext == '.pkl':
        with open(ann_file, 'rb') as f:
            return pickle.load(f, encoding='iso-8859-1')
    else:
        logger.warning(f"Unsupported annotation format: {ext}")
        return {}


def merge_text_features(text_features_list, method='concat'):
    """Merge multiple text feature arrays.
    
    Args:
        text_features_list: List of text feature arrays
        method: Merge method ('concat', 'average', 'max')
        
    Returns:
        Merged text features
    """
    if not text_features_list:
        return None
    
    # Filter out None values
    valid_features = [f for f in text_features_list if f is not None]
    if not valid_features:
        return None
    
    if method == 'concat':
        return np.concatenate(valid_features, axis=0)
    elif method == 'average':
        return np.mean(valid_features, axis=0)
    elif method == 'max':
        return np.max(valid_features, axis=0)
    else:
        logger.warning(f"Unknown merge method: {method}, using concat")
        return np.concatenate(valid_features, axis=0)


def filter_detections_by_text_score(detections, text_scores, score_threshold=0.0):
    """Filter detections based on combined image and text scores.
    
    Args:
        detections: Detection boxes and scores dictionary
        text_scores: Text similarity scores for each detection
        score_threshold: Minimum combined score threshold
        
    Returns:
        Filtered detections
    """
    if text_scores is None or len(text_scores) == 0:
        return detections
    
    # Combine image scores with text scores
    image_scores = detections.get('scores', np.ones(len(text_scores)))
    combined_scores = image_scores * text_scores
    
    # Filter by threshold
    mask = combined_scores >= score_threshold
    
    filtered_dets = {}
    for key, value in detections.items():
        if hasattr(value, '__getitem__') or isinstance(value, (list, np.ndarray)):
            filtered_dets[key] = value[mask]
        else:
            filtered_dets[key] = value
    
    return filtered_dets
