def normalize_cam_method(cam_method, supported_methods=None, default_for_aeromixer="RITSM"):
    """Normalize CAM method names and resolve aliases (including AEROMIXER).
    
    Supports both image-only and image+text multimodal CAM methods.
    """
    if cam_method is None:
        return ""

    method = str(cam_method).strip()
    if not method:
        return ""

    key = method.lower().replace("-", "").replace("_", "")
    
    # Image-only CAM methods
    image_only_aliases = {
        "gradcam": "GradCAM",
        "scorecam": "ScoreCAM",
        "gradcam++": "GradCAMPlusPlus",
        "gradcampp": "GradCAMPlusPlus",
        "ablationcam": "AblationCAM",
        "xgradcam": "XGradCAM",
        "eigencam": "EigenCAM",
        "eigengradcam": "EigenGradCAM",
        "layercam": "LayerCAM",
        "groupcam": "GroupCAM",
        "sscam1": "SSCAM1",
        "sscam2": "SSCAM2",
        "rawcam": "RawCAM",
        "gradientcam": "GradientCAM",
        "vitgradcam": "ViTGradCAM",
        "rise": "RiseCAM",
        "gscorecam": "GScoreCAM",
        "gscorecambeta": "GScoreCAMBeta",
        "testcam": "TestCAM",
    }
    
    # Multimodal (image + text) CAM methods - CLIP-based
    multimodal_aliases = {
        # RITSM variants
        "ritsm": "RITSM",
        "ritsmclip": "RITSM",
        "ritsmclipvip": "RITSM",
        "ritsm_text": "RITSM_Text",
        "ritsm_multimodal": "RITSM",
        
        # HilaCAM variants
        "hilacam": "HilaCAM",
        "hilaclip": "HilaCAM",
        "hilacamclip": "HilaCAM",
        "hilacam_clipvip": "HilaCAM",
        "hila": "HilaCAM",
        "hilacam_text": "HilaCAM_Text",
        "hilacam_multimodal": "HilaCAM",
        
        # MHSA variants
        "mhsa": "MHSA",
        "mhsacam": "MHSA",
        "mhsa_clip": "MHSA",
        "mhsa_clipvip": "MHSA",
        "mhsa_text": "MHSA_Text",
        "mhsa_multimodal": "MHSA",
        
        # Text-specific CAM methods
        "textcam": "TextCAM",
        "text CAM": "TextCAM",
        "text_heatmap": "TextCAM",
        "text_attention": "TextCAM",
        "multimodal": "MultimodalCAM",
        "image_text": "MultimodalCAM",
        "clip_text": "TextCAM",
        "clip_heatmap": "TextCAM",
        
        # Aliases for default methods
        "aeromixer": default_for_aeromixer,
        "auto": default_for_aeromixer,
        "default": default_for_aeromixer,
    }
    
    # Combine all aliases
    aliases = {**image_only_aliases, **multimodal_aliases}
    
    canonical = aliases.get(key, method)

    if supported_methods is None or canonical == "":
        return canonical

    if canonical not in supported_methods:
        options = ", ".join(sorted(supported_methods))
        raise ValueError(f"Unsupported CAM method '{cam_method}'. Supported methods: {options}")

    return canonical


def get_multimodal_cam_methods():
    """Return a list of supported multimodal (image + text) CAM methods."""
    return [
        "RITSM",
        "RITSM_Text", 
        "HilaCAM",
        "HilaCAM_Text",
        "MHSA",
        "MHSA_Text",
        "TextCAM",
        "MultimodalCAM",
    ]


def is_multimodal_method(cam_method):
    """Check if a CAM method supports image + text multimodal input.
    
    Args:
        cam_method: The CAM method name to check
        
    Returns:
        bool: True if the method supports multimodal (text) input
    """
    normalized = normalize_cam_method(cam_method)
    multimodal_methods = get_multimodal_cam_methods()
    return normalized in multimodal_methods
