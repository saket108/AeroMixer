from .defaults import _C as cfg


# ============================================================================
# Config Utility Functions
# ============================================================================

def get_config():
    """Get a copy of the default config.
    
    Returns:
        CfgNode: Copy of the default configuration
    """
    return cfg.clone()


def merge_config(config_dict):
    """Merge a config dictionary with the default config.
    
    Args:
        config_dict: Dictionary of config values to override
        
    Returns:
        CfgNode: Merged configuration
    """
    config = cfg.clone()
    config.merge_from_file(config_dict)
    return config


def update_config(config, updates):
    """Update config with new values.
    
    Args:
        config: Configuration object to update
        updates: Dictionary of config values to update
        
    Returns:
        CfgNode: Updated configuration
    """
    for key, value in updates.items():
        config[key] = value
    return config


def is_image_mode(config):
    """Check if config is in image mode.
    
    Args:
        config: Configuration object
        
    Returns:
        bool: True if in image mode, False otherwise
    """
    return config.DATA.INPUT_TYPE == "image"


def is_video_mode(config):
    """Check if config is in video mode.
    
    Args:
        config: Configuration object
        
    Returns:
        bool: True if in video mode, False otherwise (always False now since video removed)
    """
    return config.DATA.INPUT_TYPE == "video"


def get_text_config(config):
    """Get text-specific configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        CfgNode: Text configuration
    """
    if hasattr(config, 'DATA.TEXT'):
        return config.DATA.TEXT
    return None


def get_clip_config(config):
    """Get CLIP-specific configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        CfgNode: CLIP configuration
    """
    if hasattr(config, 'MODEL.CLIP'):
        return config.MODEL.CLIP
    return None


def get_clipvip_config(config):
    """Get CLIP-ViP-specific configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        CfgNode: CLIP-ViP configuration
    """
    if hasattr(config, 'MODEL.CLIPViP'):
        return config.MODEL.CLIPViP
    return None


def get_cam_config(config, model_type='clip'):
    """Get CAM-specific configuration for a model type.
    
    Args:
        config: Configuration object
        model_type: Model type ('clip', 'clipvip', or 'viclip')
        
    Returns:
        CfgNode or None: CAM configuration if available
    """
    model_type = model_type.lower()
    
    if model_type == 'clip':
        if hasattr(config, 'MODEL.CLIP.CAM'):
            return config.MODEL.CLIP.CAM
    elif model_type == 'clipvip':
        if hasattr(config, 'MODEL.CLIPViP.CAM'):
            return config.MODEL.CLIPViP.CAM
    
    return None


def set_image_mode(config):
    """Set config to image mode.
    
    Args:
        config: Configuration object
        
    Returns:
        CfgNode: Updated configuration
    """
    config.DATA.INPUT_TYPE = "image"
    config.DATA.NUM_FRAMES = 1
    config.DATA.SAMPLING_RATE = 1
    config.DATA.DATASETS = ['images']
    return config


def get_model_config(config):
    """Get model-specific configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        dict: Model configuration
    """
    model_cfg = {
        'det': config.MODEL.DET,
        'multi_label': config.MODEL.MULTI_LABEL_ACTION,
        'use_roi_feat': config.MODEL.USE_ROI_FEAT,
    }
    
    # Add CLIP config if available
    if hasattr(config, 'MODEL.CLIP'):
        model_cfg['clip'] = {
            'arch': config.MODEL.CLIP.ARCH,
            'context_init': config.MODEL.CLIP.CONTEXT_INIT,
            'freeze_text': config.MODEL.CLIP.FREEZE_TEXT_BACKBONE,
        }
    
    # Add CLIP-ViP config if available
    if hasattr(config, 'MODEL.CLIPViP'):
        model_cfg['clipvip'] = {
            'arch': config.MODEL.CLIPViP.ARCH,
            'clip_name': config.MODEL.CLIPViP.CLIP_NAME,
            'context_init': config.MODEL.CLIPViP.CONTEXT_INIT,
            'freeze_text': config.MODEL.CLIPViP.FREEZE_TEXT_BACKBONE,
        }
    
    return model_cfg


def validate_config(config):
    """Validate configuration values.
    
    Args:
        config: Configuration object
        
    Returns:
        tuple: (is_valid, error_messages)
    """
    errors = []
    
    # Check input type
    if config.DATA.INPUT_TYPE not in ['image', 'video']:
        errors.append(f"Invalid INPUT_TYPE: {config.DATA.INPUT_TYPE}. Must be 'image' or 'video'.")
    
    # Check NUM_FRAMES for image mode
    if config.DATA.INPUT_TYPE == 'image' and config.DATA.NUM_FRAMES != 1:
        errors.append(f"NUM_FRAMES should be 1 for image mode, got {config.DATA.NUM_FRAMES}")
    
    # Check SOLVER config
    if config.SOLVER.IMAGES_PER_BATCH <= 0:
        errors.append(f"IMAGES_PER_BATCH must be positive, got {config.SOLVER.IMAGES_PER_BATCH}")
    
    # Check TEST config
    if config.TEST.IMAGES_PER_BATCH <= 0:
        errors.append(f"IMAGES_PER_BATCH must be positive in TEST, got {config.TEST.IMAGES_PER_BATCH}")
    
    return len(errors) == 0, errors


def print_config(config, print=True):
    """Print configuration values.
    
    Args:
        config: Configuration object
        print: Whether to print to stdout
        
    Returns:
        str: Configuration as string
    """
    config_str = "Configuration:\n"
    config_str += "=" * 50 + "\n"
    
    # Print DATA section
    config_str += "DATA:\n"
    config_str += f"  INPUT_TYPE: {config.DATA.INPUT_TYPE}\n"
    config_str += f"  NUM_FRAMES: {config.DATA.NUM_FRAMES}\n"
    config_str += f"  SAMPLING_RATE: {config.DATA.SAMPLING_RATE}\n"
    config_str += f"  DATASETS: {config.DATA.DATASETS}\n"
    config_str += f"  OPEN_VOCABULARY: {config.DATA.OPEN_VOCABULARY}\n"
    
    # Print TEXT section if available
    if hasattr(config, 'DATA.TEXT'):
        config_str += "\nTEXT:\n"
        config_str += f"  INPUT_TYPE: {config.DATA.TEXT.INPUT_TYPE}\n"
        config_str += f"  MAX_LENGTH: {config.DATA.TEXT.MAX_LENGTH}\n"
        config_str += f"  PROMPT_TEMPLATE: {config.DATA.TEXT.PROMPT_TEMPLATE}\n"
    
    # Print MODEL section
    config_str += "\nMODEL:\n"
    config_str += f"  DET: {config.MODEL.DET}\n"
    config_str += f"  MULTI_LABEL_ACTION: {config.MODEL.MULTI_LABEL_ACTION}\n"
    
    # Print CLIP config if available
    if hasattr(config, 'MODEL.CLIP'):
        config_str += "\nCLIP:\n"
        config_str += f"  ARCH: {config.MODEL.CLIP.ARCH}\n"
        config_str += f"  CONTEXT_INIT: {config.MODEL.CLIP.CONTEXT_INIT}\n"
    
    # Print SOLVER section
    config_str += "\nSOLVER:\n"
    config_str += f"  MAX_EPOCH: {config.SOLVER.MAX_EPOCH}\n"
    config_str += f"  BASE_LR: {config.SOLVER.BASE_LR}\n"
    config_str += f"  IMAGES_PER_BATCH: {config.SOLVER.IMAGES_PER_BATCH}\n"
    
    if print:
        print(config_str)
    
    return config_str
