import torch
import numpy as np
import cv2
import warnings


# ============================================================================
# Text Input Validation (for future text support)
# ============================================================================

def validate_text_input(text, allow_none=True):
    """Validate text input for MHSA.
    
    Args:
        text: Text input (string, list of strings, or None)
        allow_none: Whether to allow None as valid input
        
    Returns:
        tuple: (is_valid, processed_text, error_message)
    """
    if text is None:
        if allow_none:
            return True, None, None
        return False, None, "Text input cannot be None"
    
    # Handle list of texts (batch processing)
    if isinstance(text, (list, tuple)):
        if len(text) == 0:
            return False, None, "Text list cannot be empty"
        # Validate each text in the list
        for i, t in enumerate(text):
            if not isinstance(t, str):
                return False, None, f"Text at index {i} is not a string: {type(t)}"
            if len(t.strip()) == 0:
                return False, None, f"Text at index {i} is empty"
        return True, text, None
    
    # Handle single string
    if isinstance(text, str):
        if len(text.strip()) == 0:
            return False, None, "Text input is empty"
        return True, text.strip(), None
    
    return False, None, f"Invalid text type: {type(text)}. Expected str, list, or None"


# ============================================================================
# Multi-Head Self-Attention Functions
# ============================================================================

def get_multi_head_mask(attentions, threshold=0.6):
    """Create mask for multi-head attention.
    
    Args:
        attentions: Attention weights
        threshold: Threshold for keeping attention values
        
    Returns:
        torch.Tensor: Mask tensor
    """
    nh, np = attentions.size(0), attentions.size(-1)
    # we keep only a certain percentage of the mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=-1, keepdim=True)
    cumval = torch.cumsum(val, dim=-1)
    th_attn = cumval > (1 - threshold)
    idx2 = torch.argsort(idx)  # dim=-1 by default
    th_attn = th_attn.view(nh, -1)
    for head in range(nh):
        th_attn[head] = th_attn[head][idx2[head].view(-1)]
    if len(attentions.size()) == 3:
        th_attn = th_attn.view(nh, -1, np)
    return th_attn


def get_masked_attention_map(attentions, nh, heatmap_size, cam_size, mask=None):
    """Create attention heatmap with masking.
    
    Args:
        attentions: Attention weights
        nh: Number of heads
        heatmap_size: Size of heatmap
        cam_size: Target CAM size
        mask: Optional mask to apply
        
    Returns:
        numpy.ndarray: Attention heatmap
    """
    if mask is not None:
        # apply mask on attention map
        attentions = attentions * mask.float()  # (num_heads, N, L)
    # normalize within each frame
    attentions -= attentions.min(dim=-1, keepdim=True)[0]
    attentions /= attentions.max(dim=-1, keepdim=True)[0]
    
    num_frames = attentions.size(1) if len(attentions.size()) == 3 else 1
    # average over multi-heads as the final attention
    attentions = attentions.reshape(nh, num_frames, heatmap_size[0], heatmap_size[1]).mean(dim=0, keepdim=True)
    
    if cam_size is not None:
        # interpolate
        attentions = torch.nn.functional.interpolate(attentions, size=(cam_size[1], cam_size[0]), mode="bilinear")[0]
    return attentions.cpu().numpy()


# ============================================================================
# MHSA Implementations
# ============================================================================

@torch.no_grad()
def mhsa_clip(image, model, cam_size=None, threshold=0.6, text=None):
    """MHSA for standard CLIP model.
    
    Args:
        image: Input image tensor (B, C, H, W) or (C, H, W)
        model: CLIP model
        cam_size: Target CAM size (optional)
        threshold: Attention threshold
        text: Text input for multimodal (currently not used, reserved for future)
        
    Returns:
        numpy.ndarray: Attention heatmap
        
    Note:
        Text support is reserved for future versions. Currently only image input is supported.
    """
    # Warn if text is provided but not used yet
    if text is not None:
        is_valid, _, error = validate_text_input(text)
        if not is_valid:
            warnings.warn(f"Text validation failed: {error}. Text will be ignored in current version.", UserWarning)
        else:
            warnings.warn("Text input is provided but not yet supported in mhsa_clip. Will be added in future versions.", UserWarning)
    
    # Handle single image or batch
    if len(image.shape) == 3:
        # Single image - add batch dimension
        image = image.unsqueeze(0)
    
    batch_size = image.shape[0]
    
    # get patch token features
    _, attn_last = model.encode_image(image, last_attn_output=True)  # (B, num_heads, L, D)
    nh = attn_last.shape[1] # number of head
    
    # we keep only the output patch attention
    # assume batch_size = 1 for simplicity
    all_heatmaps = []
    for b in range(batch_size):
        attentions = attn_last[b, :, 0, 1:].reshape(nh, -1)  # (num_heads, 7*7)
        heatmap_size = [int(attentions.size(-1)**0.5), int(attentions.size(-1)**0.5)]  # 7
        
        th_attn = get_multi_head_mask(attentions, threshold)
        
        attn_map = get_masked_attention_map(attentions, nh, heatmap_size, cam_size, mask=th_attn)  # (1, H, W)
        all_heatmaps.append(attn_map[0])
    
    if batch_size == 1:
        return all_heatmaps[0]
    
    return np.stack(all_heatmaps, axis=0)


@torch.no_grad()
def mhsa_clipvip(video, model, cam_size=None, threshold=0.6, text=None):
    """MHSA for CLIP-ViP model.
    
    Args:
        video: Input video tensor (B, T, C, H, W)
        model: CLIP-ViP model
        cam_size: Target CAM size (optional)
        threshold: Attention threshold
        text: Text input for multimodal (currently not used, reserved for future)
        
    Returns:
        numpy.ndarray: Attention heatmap
        
    Note:
        Text support is reserved for future versions. Currently only video input is supported.
    """
    # Warn if text is provided but not used yet
    if text is not None:
        is_valid, _, error = validate_text_input(text)
        if not is_valid:
            warnings.warn(f"Text validation failed: {error}. Text will be ignored in current version.", UserWarning)
        else:
            warnings.warn("Text input is provided but not yet supported in mhsa_clipvip. Will be added in future versions.", UserWarning)
    
    # Handle single video or batch
    if len(video.shape) == 4:
        # Single video (T, C, H, W) - add batch dimension
        video = video.unsqueeze(0)
    
    batch_size = video.shape[0]
    num_proxy = model.config.vision_additional_config.add_cls_num + 1
    num_heads = model.config.vision_config.num_attention_heads
    num_frames = video.size(1)

    # run forward pass to get the last block attentions
    _, heatmap_size = model.get_image_features(video, return_ws=True)  # (h,w)
    last_block = list(dict(model.vision_model.encoder.layers.named_children()).values())[-1]
    attn_inter = last_block.attn_probs['inter']  # [B*num_heads, M, M+N*L] where M=4
    attn_intra = last_block.attn_probs['intra']  # [B*num_heads*N, L, M+L] where L=196 if input_size=224
    
    num_patches = attn_intra.shape[-2] # L
    attentions_inter = attn_inter[:, 0, num_proxy:].reshape(-1, num_heads, num_frames, num_patches)[0]  # [B*num_heads, N*L] --> [num_heads, N, L]
    # attentions_intra = attn_intra[:, 0, num_proxy:].reshape(-1, num_heads, num_frames, num_patches)[0]  # [B*num_heads*N, L] --> [num_heads, N, L]
    
    th_attn = get_multi_head_mask(attentions_inter, threshold)
    attn_map = get_masked_attention_map(attentions_inter, num_heads, heatmap_size, cam_size, mask=th_attn)  # (T, H, W)
    
    # temporal weights
    temporal_weights = attn_inter[:, 0, num_proxy:].reshape(-1, num_frames, num_patches).sum(dim=-1)  # [B*num_heads, N]
    temporal_weights = temporal_weights.reshape(-1, num_heads, num_frames)[0].sum(dim=0)  # [N]
    temporal_weights -= temporal_weights.min(dim=-1, keepdim=True)[0]
    temporal_weights /= temporal_weights.max(dim=-1, keepdim=True)[0]
    temporal_weights = temporal_weights.cpu().numpy()
    attn_map = temporal_weights[:, None, None] * attn_map
    
    return attn_map


# ============================================================================
# Batch Processing Utilities
# ============================================================================

def process_batch_mhsa(images, model, cam_size=None, threshold=0.6, text=None, device='cuda'):
    """Process a batch of images with MHSA.
    
    Args:
        images: Batch of images (B, C, H, W) or list of images
        model: CLIP model
        cam_size: Target CAM size
        threshold: Attention threshold
        text: Text input for multimodal
        device: Device to use
        
    Returns:
        list: List of attention heatmaps
    """
    # Convert list to batch tensor
    if isinstance(images, (list, tuple)):
        # Stack images
        if all(isinstance(img, torch.Tensor) for img in images):
            images = torch.stack(images, dim=0)
        elif all(isinstance(img, np.ndarray) for img in images):
            images = torch.from_numpy(np.stack(images, axis=0))
        else:
            raise ValueError("Images must be tensors or numpy arrays")
    
    # Move to device
    if not images.is_cuda:
        images = images.to(device)
    
    # Process with MHSA
    heatmaps = mhsa_clip(images, model, cam_size=cam_size, threshold=threshold, text=text)
    
    return heatmaps


def process_batch_mhsa_vip(videos, model, cam_size=None, threshold=0.6, text=None, device='cuda'):
    """Process a batch of videos with MHSA.
    
    Args:
        videos: Batch of videos (B, T, C, H, W) or list of videos
        model: CLIP-ViP model
        cam_size: Target CAM size
        threshold: Attention threshold
        text: Text input for multimodal
        device: Device to use
        
    Returns:
        list: List of attention heatmaps
    """
    # Convert list to batch tensor
    if isinstance(videos, (list, tuple)):
        # Stack videos
        if all(isinstance(vid, torch.Tensor) for vid in videos):
            videos = torch.stack(videos, dim=0)
        elif all(isinstance(vid, np.ndarray) for vid in videos):
            videos = torch.from_numpy(np.stack(videos, axis=0))
        else:
            raise ValueError("Videos must be tensors or numpy arrays")
    
    # Move to device
    if not videos.is_cuda:
        videos = videos.to(device)
    
    # Process with MHSA
    heatmaps = mhsa_clipvip(videos, model, cam_size=cam_size, threshold=threshold, text=text)
    
    return heatmaps
