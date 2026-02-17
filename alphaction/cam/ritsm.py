import torch
import numpy as np
import cv2
import warnings


# ============================================================================
# Text Input Validation
# ============================================================================

def validate_text_input(text, allow_none=False):
    """Validate text input for RITSM.
    
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


def tokenize_text(text, tokenizer, device='cuda'):
    """Tokenize text input for RITSM.
    
    Args:
        text: Text input (string or list of strings)
        tokenizer: Tokenizer to use
        device: Device to move tokens to
        
    Returns:
        torch.Tensor: Tokenized text
    """
    if text is None:
        return None
    
    # Handle single string
    if isinstance(text, str):
        tokens = tokenizer(text).to(device)
        return tokens
    
    # Handle list of strings (batch)
    if isinstance(text, (list, tuple)):
        tokens = tokenizer(text).to(device)
        return tokens
    
    raise ValueError(f"Invalid text type for tokenization: {type(text)}")


# ============================================================================
# RITSM Implementations
# ============================================================================

def clip_forward(model, image, text):
    """Forward pass for CLIP with image and text.
    
    Args:
        model: CLIP model
        image: Image tensor
        text: Text tensor
        
    Returns:
        tuple: (logits_per_image, encoder_out, text_features)
    """
    # get patch token features
    image_features, encoder_out = model.encode_image(image, transformer_output=True)  # (N, L, D)
    text_features = model.encode_text(text)

    # cosine similarity as logits
    logit_scale = model.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logit_scale * text_features @ image_features.t()

    return logits_per_image, encoder_out, text_features


@torch.no_grad()
def ritsm_clip(image, text, model, device, index=None, cam_size=None, return_logits=False, attn_grad=False):
    """RITSM for standard CLIP model.
    
    Args:
        image: Input image tensor
        text: Text input (string, list of strings, or tokenized tensor)
        model: CLIP model
        device: Device to use
        index: Target class index (optional)
        cam_size: Size of output CAM (optional)
        return_logits: Whether to return logits
        attn_grad: Use attention gradients (currently not used)
        
    Returns:
        numpy.ndarray or tuple: CAM heatmap or (CAM, logits)
    """
    # Validate and process text input
    is_valid, processed_text, error = validate_text_input(text, allow_none=False)
    if not is_valid:
        raise ValueError(f"Text validation failed: {error}")
    
    # Tokenize text if not already tokenized
    if not isinstance(processed_text, torch.Tensor):
        try:
            if hasattr(model, 'tokenizer'):
                text = tokenize_text(processed_text, model.tokenizer, device)
            else:
                # Try to get tokenizer from model
                text = tokenize_text(processed_text, model.transformer, device)
        except Exception as e:
            raise ValueError(f"Text tokenization failed: {e}")
    else:
        text = processed_text
    
    # Handle batch text input
    if text is not None and len(text.shape) == 2:
        # Single text or batch - ensure proper shape
        if text.shape[0] != image.shape[0]:
            # Repeat text to match batch size
            text = text.repeat(image.shape[0], 1, 1)[:image.shape[0]]
    
    # forward pass
    logits_per_image, encoder_out, text_features = clip_forward(model, image, text)
    probs = logits_per_image.softmax(dim=-1)
    if index is None:
        # locate the largest score of img-text pair
        index = np.argmax(logits_per_image.cpu().data.numpy(), axis=-1)
    
    input_size = model.visual.input_resolution  # 224
    patch_features = encoder_out[:, 1:, :]  # (B, 7*7, 768)
    heatmap_size = int(patch_features.size(1)**0.5)  # 7

    # projection
    patch_features = model.visual.ln_post(patch_features)
    if model.visual.proj is not None:
        patch_features = patch_features @ model.visual.proj   # (B, 7*7, 512)

    # normalize
    patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)
    # text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # (K=1, 512)

    # image-text similarity
    it_sim = patch_features @ text_features.t()  # (B, 7*7, K=1)

    # reshape & resize
    image_relevance_all = it_sim[:, :, index].view(-1, 1, heatmap_size, heatmap_size)  # (B, 1, 7, 7)
    image_relevance_all = torch.nn.functional.interpolate(image_relevance_all.float(), size=input_size, mode='bilinear')  # (B, 1, H, W)
    image_relevance = image_relevance_all[0]  # assume batch_size = 1
    image_relevance = image_relevance.reshape(input_size, input_size).detach().cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())     
    # reverse
    image_relevance = np.fabs(1 - image_relevance)

    out = cv2.resize(image_relevance, cam_size) if cam_size is not None else image_relevance
    if return_logits:
        return out, logits_per_image
    return out


@torch.no_grad()
def ritsm_clipvip(video, text, model, device, index=None, cam_size=None, return_logits=False, attn_grad=False, use_mask=False):
    """RITSM for CLIP-ViP model.
    
    Args:
        video: Input video tensor (B, T, C, H, W)
        text: Text input (string, list of strings, or tokenized tensor)
        model: CLIP-ViP model
        device: Device to use
        index: Target class index (optional)
        cam_size: Size of output CAM (optional)
        return_logits: Whether to return logits
        attn_grad: Use attention gradients (currently not used)
        use_mask: Use attention mask
        
    Returns:
        numpy.ndarray or tuple: CAM heatmap or (CAM, logits)
    """
    # Validate and process text input
    is_valid, processed_text, error = validate_text_input(text, allow_none=False)
    if not is_valid:
        raise ValueError(f"Text validation failed: {error}")
    
    # Tokenize text if not already tokenized
    if not isinstance(processed_text, torch.Tensor):
        try:
            if hasattr(model, 'tokenizer'):
                text = tokenize_text(processed_text, model.tokenizer, device)
            else:
                # Try to get tokenizer from model
                text = tokenize_text(processed_text, model.transformer, device)
        except Exception as e:
            raise ValueError(f"Text tokenization failed: {e}")
    else:
        text = processed_text
    
    # Handle batch text input
    if text is not None and len(text.shape) == 2:
        # Single text or batch - ensure proper shape
        if text.shape[0] != video.shape[0]:
            # Repeat text to match batch size
            text = text.repeat(video.shape[0], 1, 1)[:video.shape[0]]
    
    num_proxy = model.config.vision_additional_config.add_cls_num + 1
    eos_idx = text.argmax(dim=-1)
    num_frames = video.size(1)
    
    input_size = model.config.vision_config.image_size  # 224
    patch_size = model.config.vision_config.patch_size  # 16
    num_patches = int(input_size // patch_size)  # 14

    # run forward pass
    out_dict = model(text, video)
    logits_per_image = out_dict['logits_per_image']

    if index is None:
        # locate the largest score of img-text pair
        index = np.argmax(logits_per_image.cpu().data.numpy(), axis=-1)

    # get patch features from the last vision encoder block
    patch_features = out_dict['vision_model_output']['last_hidden_state'][:, num_proxy:, :]  # (B, T*14*14, 768)
    assert num_frames * (num_patches ** 2) == patch_features.size(1)
    
    # layernorm, projection, and normalization
    patch_features = model.vision_model.post_layernorm(patch_features)
    patch_features = model.visual_projection(patch_features)  # 768 --> 512
    patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)  # (B, T*14*14, 512)

    # get the text features
    text_features = out_dict['text_embeds']  # after layernorm, projection, and normalization

    # image-text similarity
    it_sim = patch_features @ text_features.t()  # (B, T*14*14, K=1)
    
    if use_mask:
        th_attn = get_attn_mask(it_sim[0, :, index].view(num_frames, -1), threshold=0.6)
        it_sim = it_sim * th_attn.view(-1).unsqueeze(0).unsqueeze(-1) 
    
    # reshape & resize
    image_relevance_all = it_sim[:, :, index].view(-1, num_frames, num_patches, num_patches)  # (B, T, 14, 14)
    image_relevance_all = torch.nn.functional.interpolate(image_relevance_all.float(), size=input_size, mode='bilinear')  # (B, T, H, W)

    # assume batch_size = 1
    image_relevance_all = image_relevance_all[0]

    all_maps = []
    for image_relevance in image_relevance_all:
        image_relevance = image_relevance.reshape(input_size, input_size).detach().cpu().numpy()
        # normalize and reverse
        image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
        # reverse
        image_relevance = np.fabs(1 - image_relevance)
        atten_map = cv2.resize(image_relevance, cam_size) if cam_size is not None else image_relevance    
        all_maps.append(atten_map)

    out = np.stack(all_maps, axis=0)
    
    if return_logits:
        return out, logits_per_image
    return out


def get_attn_mask(attentions, threshold=0.6):
    """Create attention mask for RITSM.
    
    Args:
        attentions: Attention weights (T, L)
        threshold: Threshold for keeping attention values
        
    Returns:
        torch.Tensor: Attention mask
    """
    nh = attentions.size(0)
    # we keep only a certain percentage of the mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=-1, keepdim=True)
    cumval = torch.cumsum(val, dim=-1)
    th_attn = cumval > (1 - threshold)
    idx2 = torch.argsort(idx)  # dim=-1 by default
    th_attn = th_attn.view(nh, -1)
    for head in range(nh):
        th_attn[head] = th_attn[head][idx2[head].view(-1)]
    return th_attn


# ============================================================================
# Batch Processing Utilities
# ============================================================================

def process_batch_ritsm(images, texts, model, device='cuda', cam_size=None, return_logits=False):
    """Process a batch of images with RITSM.
    
    Args:
        images: Batch of images (B, C, H, W)
        texts: List of text strings or single text string
        model: CLIP model
        device: Device to use
        cam_size: Target CAM size
        return_logits: Whether to return logits
        
    Returns:
        list: List of attention heatmaps
    """
    # Convert list to batch tensor
    if isinstance(images, (list, tuple)):
        if all(isinstance(img, torch.Tensor) for img in images):
            images = torch.stack(images, dim=0)
        elif all(isinstance(img, np.ndarray) for img in images):
            images = torch.from_numpy(np.stack(images, axis=0))
        else:
            raise ValueError("Images must be tensors or numpy arrays")
    
    # Move to device
    if not images.is_cuda:
        images = images.to(device)
    
    # Handle single text or list of texts
    if isinstance(texts, str):
        # Single text for all images
        texts = [texts] * images.shape[0]
    
    # Process with RITSM
    heatmaps = []
    for i in range(images.shape[0]):
        img = images[i:i+1]
        txt = texts[i] if i < len(texts) else texts[-1]
        
        heatmap = ritsm_clip(img, txt, model, device, cam_size=cam_size, return_logits=return_logits)
        heatmaps.append(heatmap)
    
    return heatmaps


def process_batch_ritsm_vip(videos, texts, model, device='cuda', cam_size=None, return_logits=False):
    """Process a batch of videos with RITSM.
    
    Args:
        videos: Batch of videos (B, T, C, H, W)
        texts: List of text strings or single text string
        model: CLIP-ViP model
        device: Device to use
        cam_size: Target CAM size
        return_logits: Whether to return logits
        
    Returns:
        list: List of attention heatmaps
    """
    # Convert list to batch tensor
    if isinstance(videos, (list, tuple)):
        if all(isinstance(vid, torch.Tensor) for vid in videos):
            videos = torch.stack(videos, dim=0)
        elif all(isinstance(vid, np.ndarray) for vid in videos):
            videos = torch.from_numpy(np.stack(videos, axis=0))
        else:
            raise ValueError("Videos must be tensors or numpy arrays")
    
    # Move to device
    if not videos.is_cuda:
        videos = videos.to(device)
    
    # Handle single text or list of texts
    if isinstance(texts, str):
        # Single text for all videos
        texts = [texts] * videos.shape[0]
    
    # Process with RITSM
    heatmaps = []
    for i in range(videos.shape[0]):
        vid = videos[i:i+1]
        txt = texts[i] if i < len(texts) else texts[-1]
        
        heatmap = ritsm_clipvip(vid, txt, model, device, cam_size=cam_size, return_logits=return_logits)
        heatmaps.append(heatmap)
    
    return heatmaps
