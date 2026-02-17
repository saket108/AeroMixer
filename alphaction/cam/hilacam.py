import torch
import numpy as np
import cv2
import warnings


# ============================================================================
# Text Input Validation
# ============================================================================

def validate_text_input(text, allow_none=False):
    """Validate text input for HilaCAM.
    
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
    """Tokenize text input for HilaCAM.
    
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
# HilaCAM Implementations
# ============================================================================

def hilacam_clip(image, text, model, device, index=None, cam_size=None, return_logits=False, attn_grad=True):
    """HilaCAM for standard CLIP model.
    
    Args:
        image: Input image tensor
        text: Text input (string, list of strings, or tokenized tensor)
        model: CLIP model
        device: Device to use
        index: Target class index (optional)
        cam_size: Size of output CAM (optional)
        return_logits: Whether to return logits
        attn_grad: Use attention gradients
        
    Returns:
        numpy.ndarray or tuple: CAM heatmap or (CAM, logits)
    """
    # Validate and process text input
    is_valid, processed_text, error = validate_text_input(text, allow_none=True)
    if not is_valid:
        warnings.warn(f"Text validation failed: {error}. Using empty text.", UserWarning)
        processed_text = ""
    
    # Tokenize text if not already tokenized
    if not isinstance(processed_text, torch.Tensor):
        try:
            if hasattr(model, 'tokenizer'):
                text = tokenize_text(processed_text, model.tokenizer, device)
            else:
                # Try to get tokenizer from model
                text = tokenize_text(processed_text, model.transformer, device)
        except Exception as e:
            warnings.warn(f"Text tokenization failed: {e}. Using empty text.", UserWarning)
            text = None
    else:
        text = processed_text
    
    # Handle batch text input
    if text is not None and len(text.shape) == 2:
        # Single text or batch - ensure proper shape
        if text.shape[0] != image.shape[0]:
            # Repeat text to match batch size
            text = text.repeat(image.shape[0], 1, 1)[:image.shape[0]]
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1)
    if index is None:
        # locate the largest score of img-text pair
        index = np.argmax(logits_per_image.cpu().data.numpy(), axis=-1)
    
    if attn_grad:
        one_hot = np.zeros((1, logits_per_image.size()[-1]), dtype=np.float32)
        # create a tensor equal to the clip score
        one_hot[0, index] = 1
        one_hot = torch.from_numpy(one_hot)
        one_hot = torch.sum(one_hot.cuda() * logits_per_image)
        model.zero_grad()
        # back propergate to the network
        one_hot.requires_grad_(True)
        one_hot.backward(retain_graph=True)
    
    # create a diagonal matrix
    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    # R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    dtype = image_attn_blocks[0].attn_probs.dtype
    R = torch.eye(num_tokens, num_tokens, dtype=torch.float32) #* change to cpu to resolve memory overflow
    # weighted activation
    for blk in image_attn_blocks:
        cam = blk.attn_probs
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        if attn_grad:
            grad = blk.attn_grad
            grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
        cam = cam.clamp(min=0).mean(dim=0)
        # R += torch.matmul(cam, R)
        R += torch.matmul(cam.detach().cpu().to(torch.float32), R)
    R[0, 0] = 0
    image_relevance = R[0, 1:]

    length = image_relevance.shape[-1]
    heatmap_size = int(length**0.5)
    image_relevance = image_relevance.reshape(1, 1, heatmap_size, heatmap_size)
    image_relevance = torch.nn.functional.interpolate(image_relevance.float(), size=224, mode='bilinear')
    image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())    

    out = cv2.resize(image_relevance, cam_size) if cam_size is not None else image_relevance
    if return_logits:
        return out, logits_per_image
    return out

        
def hilacam_clipvip(image, text, model, device, index=None, cam_size=None, return_logits=False, attn_grad=True):
    """HilaCAM for CLIP-ViP model.
    
    Args:
        image: Input image tensor
        text: Text input (string, list of strings, or tokenized tensor)
        model: CLIP-ViP model
        device: Device to use
        index: Target class index (optional)
        cam_size: Size of output CAM (optional)
        return_logits: Whether to return logits
        attn_grad: Use attention gradients
        
    Returns:
        numpy.ndarray or tuple: CAM heatmap or (CAM, logits)
    """
    # Validate and process text input
    is_valid, processed_text, error = validate_text_input(text, allow_none=True)
    if not is_valid:
        warnings.warn(f"Text validation failed: {error}. Using empty text.", UserWarning)
        processed_text = ""
    
    # Tokenize text if not already tokenized
    if not isinstance(processed_text, torch.Tensor):
        try:
            if hasattr(model, 'tokenizer'):
                text = tokenize_text(processed_text, model.tokenizer, device)
            else:
                # Try to get tokenizer from model
                text = tokenize_text(processed_text, model.transformer, device)
        except Exception as e:
            warnings.warn(f"Text tokenization failed: {e}. Using empty text.", UserWarning)
            text = None
    else:
        text = processed_text
    
    # Handle batch text input
    if text is not None and len(text.shape) == 2:
        # Single text or batch - ensure proper shape
        if text.shape[0] != image.shape[0]:
            # Repeat text to match batch size
            text = text.repeat(image.shape[0], 1, 1)[:image.shape[0]]
    
    # run forward pass
    out_dict = model(text, image)
    logits_per_image = out_dict['logits_per_image']

    probs = logits_per_image.softmax(dim=-1)
    if index is None:
        # locate the largest score of img-text pair
        index = np.argmax(logits_per_image.cpu().data.numpy(), axis=-1)
    
    if attn_grad:
        one_hot = np.zeros((1, logits_per_image.size()[-1]), dtype=np.float32)
        # create a tensor equal to the clip score
        one_hot[0, index] = 1
        one_hot = torch.from_numpy(one_hot)
        one_hot = torch.sum(one_hot.cuda() * logits_per_image)
        model.zero_grad()
        # back propergate to the network
        one_hot.requires_grad_(True)
        one_hot.backward(retain_graph=True)
    
    # create a diagonal matrix
    image_attn_blocks = list(dict(model.vision_model.encoder.layers.named_children()).values())
    num_proxy, num_tokens = image_attn_blocks[0].attn_probs['inter'].shape[-2:]
    num_patches = image_attn_blocks[0].attn_probs['intra'].size(1) 
    R = torch.eye(num_tokens, num_tokens, dtype=torch.float32) #* change to cpu to resolve memory overflow
    # weighted activation
    for blk in image_attn_blocks:
        cam = construct_attention(blk.attn_probs['inter'], blk.attn_probs['intra'], image.size(1))  # (B*num_heads, M+N*L, M+N*L)
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        if attn_grad:
            grad = construct_attention(blk.attn_grads['inter'], blk.attn_grads['intra'], image.size(1))  # (B*num_heads, M+N*L, M+N*L)
            grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
        cam = cam.clamp(min=0).mean(dim=0)  # average cam over 12 heads
        # R += torch.matmul(cam, R)
        R += torch.matmul(cam.detach().cpu().to(torch.float32), R)
    R[0, 0] = 0
    image_relevance_all = R[0, num_proxy:].view(image.size(1), num_patches)  # the 1st row (N, L)

    all_maps = []
    for image_relevance in image_relevance_all:
        length = image_relevance.shape[-1]
        heatmap_size = int(length**0.5)
        image_relevance = image_relevance.reshape(1, 1, heatmap_size, heatmap_size)
        image_relevance = torch.nn.functional.interpolate(image_relevance.float(), size=224, mode='bilinear')
        image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
        image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
        atten_map = cv2.resize(image_relevance, cam_size) if cam_size is not None else image_relevance    
        all_maps.append(atten_map)
    
    out = np.stack(all_maps, axis=0)

    if return_logits:
        return out, logits_per_image
    return out



def construct_attention(inter, intra, num_frames):
    """Construct the proxy-guided attention (or gradient) map.
    
    Args:
        inter: Inter-frame attention
        intra: Intra-frame attention
        num_frames: Number of frames
        
    Returns:
        torch.Tensor: Combined attention map
    """
    batch_num_heads, num_proxy, num_tokens = inter.size()  # 12, 4, 4+N*L
    num_patches = intra.size(1)  # L=196
    full_map = torch.zeros((batch_num_heads, num_tokens, num_tokens), dtype=inter.dtype, device=inter.device)
    # the first num_proxy rows
    full_map[:, :num_proxy, :] = inter
    # for each num_patches rows
    intra_rs = intra.view(batch_num_heads, num_frames, num_patches, -1)  # (12, N, L, M+L)
    for t in range(num_frames):
        start = num_proxy + t * num_patches
        end = num_proxy + (t+1) * num_patches
        full_map[:, start: end, :4] = intra_rs[:, t, :, :4]  # proxy column part
        full_map[:, start: end, start: end] = intra_rs[:, t, :, 4:]  # diagnal part
    return full_map
