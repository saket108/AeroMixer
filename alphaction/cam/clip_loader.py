import importlib
import warnings
import os

import torch


# ============================================================================
# Tokenizer Utilities
# ============================================================================

def load_tokenizer(clip_version="ViT-B/32"):
    """Load CLIP tokenizer.
    
    Args:
        clip_version: CLIP model version (default: ViT-B/32)
        
    Returns:
        tokenizer: CLIP tokenizer
    """
    try:
        import clip
        _, tokenizer = clip.load(clip_version, device='cpu')
        return tokenizer
    except ImportError:
        raise ImportError("CLIP is not installed. Please install it: pip install clip")


def load_tokenizer_from_model(model):
    """Load tokenizer from a CLIP model.
    
    Args:
        model: CLIP model
        
    Returns:
        tokenizer: CLIP tokenizer
    """
    if hasattr(model, 'tokenizer'):
        return model.tokenizer
    
    # Try to get tokenizer from the model
    if hasattr(model, 'transformer'):
        # For older CLIP models
        from .simple_tokenizer import SimpleTokenizer
        return SimpleTokenizer()
    
    # Default tokenizer
    return load_tokenizer()


# ============================================================================
# Text Encoder Utilities
# ============================================================================

def get_text_encoder_name(clip_version):
    """Get the text encoder name from CLIP version.
    
    Args:
        clip_version: CLIP model version
        
    Returns:
        str: Text encoder name
    """
    version_lower = clip_version.lower()
    if 'vit-b' in version_lower or 'b/32' in version_lower:
        return 'ViT-B'
    elif 'vit-l' in version_lower or 'l/14' in version_lower:
        return 'ViT-L'
    elif 'rn101' in version_lower:
        return 'RN101'
    elif 'rn50' in version_lower or 'r50' in version_lower:
        return 'RN50'
    else:
        return 'ViT-B'


def load_text_encoder(clip_version="ViT-B/32", device='cuda'):
    """Load CLIP text encoder.
    
    Args:
        clip_version: CLIP model version
        device: Device to load the model
        
    Returns:
        tuple: (text_encoder, tokenizer)
    """
    try:
        import clip
        model, tokenizer = clip.load(clip_version, device=device)
        return model, tokenizer
    except ImportError:
        raise ImportError("CLIP is not installed. Please install it: pip install clip")


# ============================================================================
# OpenCLIP Support
# ============================================================================

def is_openclip_model(clip_version):
    """Check if the version string indicates an OpenCLIP model.
    
    Args:
        clip_version: Model version string
        
    Returns:
        bool: True if OpenCLIP model
    """
    openclip_indicators = ['openclip', 'laion', 'datacomp', 'commonpool']
    version_lower = clip_version.lower()
    return any(indicator in version_lower for indicator in openclip_indicators)


def load_openclip(clip_version, device='cuda'):
    """Load OpenCLIP model.
    
    Args:
        clip_version: OpenCLIP model name (e.g., "ViT-L-14", "laion/CLIP-ViT-L-14-DataComp")
        device: Device to load the model
        
    Returns:
        tuple: (model, preprocess, tokenizer)
    """
    try:
        import open_clip
    except ImportError:
        raise ImportError("OpenCLIP is not installed. Please install it: pip install open-clip")
    
    # Parse model name
    if '/' in clip_version:
        # Format: "creator/model-name"
        model_name = clip_version.split('/')[-1]
    else:
        model_name = clip_version
    
    # Load model and preprocessing
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained='laion2b_s32b_b82k',
        device=device
    )
    
    # Load tokenizer
    tokenizer = open_clip.get_tokenizer(model_name)
    
    return model, preprocess, tokenizer


# ============================================================================
# CLIP Loading Functions
# ============================================================================

def reshape_transform(tensor, height=None, width=None):
    """Transform tensor for CAM visualization.
    
    Args:
        tensor: Input tensor
        height: Target height
        width: Target width
        
    Returns:
        Transformed tensor
    """
    if height or width is None:
        grid_square = len(tensor) - 1
        if grid_square ** 0.5 % 1 == 0:
            height = width = int(grid_square**0.5)
        else:
            raise ValueError("Heatmap is not square, please set height and width.")
    result = tensor[1:, :, :].reshape(
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.permute(2, 0, 1)
    return result.unsqueeze(0)


def _import_clip_module(prefer_hila=False):
    """Import CLIP module.
    
    Args:
        prefer_hila: Whether to prefer hila_clip
        
    Returns:
        module: CLIP module
    """
    if prefer_hila:
        try:
            return importlib.import_module("hila_clip.clip")
        except ModuleNotFoundError:
            pass
    return importlib.import_module("clip")


def _import_open_clip():
    """Import OpenCLIP module.
    
    Returns:
        module: OpenCLIP module
    """
    try:
        return importlib.import_module("open_clip")
    except ImportError:
        return None


def load_clip(clip_version, attn_prob=True, attn_grad=True, attn_last_only=True, resize='adapt', custom=False, model_weight=None):
    """Load CLIP model with various options.
    
    Args:
        clip_version: CLIP model version
        attn_prob: Use attention probabilities
        attn_grad: Use attention gradients
        attn_last_only: Use only last layer attention
        resize: Resize option ('adapt' or 'raw')
        custom: Use custom CLIP
        model_weight: Path to model weights
        
    Returns:
        tuple: (clip_model, preprocess, target_layer, cam_trans, clip_module)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check if OpenCLIP model
    if is_openclip_model(clip_version):
        warnings.warn("OpenCLIP model detected. Use load_openclip() for better support.", UserWarning)
    
    # Load based on version
    if 'vit' in clip_version.lower() and not custom:
        # Use hila CLIP for ViT models
        clip = _import_clip_module(prefer_hila=True)
        clip_model, preprocess = clip.load(clip_version, device=device, jit=False)
    
    elif 'clip-vip' in clip_version.lower():
        # Use CLIP-ViP
        import sys
        sys.path.append("../../")
        from alphaction.modeling.encoders.clipvip import loader
        clip = _import_clip_module(prefer_hila=False)
        clip_model, preprocess = loader.load(clip_version, 
                                             attn_prob=attn_prob,
                                             attn_grad=attn_grad, 
                                             attn_last_only=attn_last_only,
                                             device=device, model_weight=model_weight)

    elif custom:
        # Custom CLIP
        clip = _import_clip_module(prefer_hila=True)
        clip_model, preprocess = clip.load(clip_version, device=device, jit=False)

    else:
        # Standard CLIP
        clip = _import_clip_module(prefer_hila=False)
        clip_model, preprocess = clip.load(clip_version, device=device)

    # Determine target layer based on model type
    if clip_version.startswith("RN"):
        # ResNet models
        target_layer = clip_model.visual.layer4[-1]
        cam_trans = None
    elif 'clip-vip' in clip_version.lower():
        # CLIP-ViP models
        target_layer = clip_model.vision_model.encoder.layers[-1]
        cam_trans = reshape_transform
    else:
        # ViT models (default)
        target_layer = clip_model.visual.transformer.resblocks[-1]
        cam_trans = reshape_transform

    # Handle resize options
    if resize == 'raw':
        # Remove clip resizing
        if not custom:
            raise Exception("Raw input needs to use custom clip.") 
        preprocess.transforms.pop(0)
        preprocess.transforms.pop(0)
    elif resize == 'adapt':
        # Adapt to clip size
        from torchvision import transforms
        crop_size = preprocess.transforms[1].size
        preprocess.transforms.insert(0, transforms.Resize(crop_size))
    
    return clip_model, preprocess, target_layer, cam_trans, clip


def load_clip_from_checkpoint(checkpoint, model):
    """Load CLIP model from checkpoint.
    
    Args:
        checkpoint: Path to checkpoint file
        model: Model to load weights into
        
    Returns:
        Model with loaded weights
    """
    checkpoint = torch.load(checkpoint, map_location='cpu')

    # Use these 3 lines if you use default model setting (not training setting) of the clip
    # For example, if you set context_length to 100 since your string is very long during training
    # checkpoint['model_state_dict']["input_resolution"] = model.input_resolution #default is 224
    # checkpoint['model_state_dict']["context_length"] = model.context_length # default is 77
    # checkpoint['model_state_dict']["vocab_size"] = model.vocab_size 

    model.load_state_dict(checkpoint['model_state_dict'])
    return model


# ============================================================================
# Multimodal Utilities
# ============================================================================

def encode_text(tokenizer, text, device='cuda'):
    """Encode text using CLIP tokenizer.
    
    Args:
        tokenizer: CLIP tokenizer
        text: Text to encode (string or list of strings)
        device: Device to use
        
    Returns:
        torch.Tensor: Tokenized text
    """
    if isinstance(text, str):
        text = [text]
    
    tokens = tokenizer(text).to(device)
    return tokens


def get_text_features(model, tokenizer, texts, device='cuda'):
    """Get text features from CLIP model.
    
    Args:
        model: CLIP model
        tokenizer: CLIP tokenizer
        texts: List of text strings
        device: Device to use
        
    Returns:
        torch.Tensor: Text features
    """
    tokens = encode_text(tokenizer, texts, device)
    
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    return text_features


def get_image_features(model, images, device='cuda'):
    """Get image features from CLIP model.
    
    Args:
        model: CLIP model
        images: Image tensor
        device: Device to use
        
    Returns:
        torch.Tensor: Image features
    """
    with torch.no_grad():
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    return image_features
