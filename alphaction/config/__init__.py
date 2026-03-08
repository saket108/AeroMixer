from .defaults import _C as cfg


def get_config():
    """Get a copy of the default config."""

    return cfg.clone()


def merge_config(config_file):
    """Merge a config file with the default config."""

    config = cfg.clone()
    config.merge_from_file(config_file)
    return config


def update_config(config, updates):
    """Update a config object with top-level values."""

    for key, value in updates.items():
        config[key] = value
    return config


def is_image_mode(config):
    """Check whether the active runtime is image-only."""

    return bool(
        str(getattr(config.DATA, "INPUT_TYPE", "image")).lower() == "image"
        or getattr(config.DATA, "IMAGE_MODE", False)
    )


def is_video_mode(config):
    """Backward-compatible helper for removed video paths."""

    return False


def uses_text_branch(config):
    """Check whether the model should initialize class/prompt text features."""

    return bool(
        getattr(config.DATA, "OPEN_VOCABULARY", False)
        or getattr(config.DATA, "MULTIMODAL", False)
    )


def get_text_config(config):
    """Get text-specific configuration."""

    return getattr(config.DATA, "TEXT", None)


def set_image_mode(config):
    """Normalize config to the supported image-only runtime."""

    config.DATA.INPUT_TYPE = "image"
    config.DATA.IMAGE_MODE = True
    config.DATA.NUM_FRAMES = 1
    config.DATA.SAMPLING_RATE = 1
    config.DATA.DATASETS = ["images"]
    return config


def validate_config(config):
    """Validate configuration values for the active runtime."""

    errors = []

    if config.DATA.INPUT_TYPE != "image":
        errors.append(
            f"Invalid INPUT_TYPE: {config.DATA.INPUT_TYPE}. "
            "This codebase currently supports image mode only ('image')."
        )

    if config.DATA.INPUT_TYPE == "image" and config.DATA.NUM_FRAMES != 1:
        errors.append(f"NUM_FRAMES should be 1 for image mode, got {config.DATA.NUM_FRAMES}")

    if config.SOLVER.IMAGES_PER_BATCH <= 0:
        errors.append(f"IMAGES_PER_BATCH must be positive, got {config.SOLVER.IMAGES_PER_BATCH}")

    if config.TEST.IMAGES_PER_BATCH <= 0:
        errors.append(f"IMAGES_PER_BATCH must be positive in TEST, got {config.TEST.IMAGES_PER_BATCH}")

    return len(errors) == 0, errors


def print_config(config, print=True):
    """Render a compact config summary."""

    config_str = "Configuration:\n"
    config_str += "=" * 50 + "\n"
    config_str += "DATA:\n"
    config_str += f"  INPUT_TYPE: {config.DATA.INPUT_TYPE}\n"
    config_str += f"  NUM_FRAMES: {config.DATA.NUM_FRAMES}\n"
    config_str += f"  SAMPLING_RATE: {config.DATA.SAMPLING_RATE}\n"
    config_str += f"  DATASETS: {config.DATA.DATASETS}\n"
    config_str += f"  OPEN_VOCABULARY: {config.DATA.OPEN_VOCABULARY}\n"
    config_str += f"  MULTIMODAL: {getattr(config.DATA, 'MULTIMODAL', False)}\n"

    if hasattr(config, "DATA.TEXT"):
        config_str += "\nTEXT:\n"
        config_str += f"  INPUT_TYPE: {config.DATA.TEXT.INPUT_TYPE}\n"
        config_str += f"  MAX_LENGTH: {config.DATA.TEXT.MAX_LENGTH}\n"
        config_str += f"  PROMPT_TEMPLATE: {config.DATA.TEXT.PROMPT_TEMPLATE}\n"

    config_str += "\nMODEL:\n"
    config_str += f"  DET: {config.MODEL.DET}\n"
    config_str += f"  MULTI_LABEL_ACTION: {config.MODEL.MULTI_LABEL_ACTION}\n"
    config_str += f"  TEXT_ENCODER: {config.MODEL.TEXT_ENCODER}\n"

    config_str += "\nSOLVER:\n"
    config_str += f"  MAX_EPOCH: {config.SOLVER.MAX_EPOCH}\n"
    config_str += f"  BASE_LR: {config.SOLVER.BASE_LR}\n"
    config_str += f"  IMAGES_PER_BATCH: {config.SOLVER.IMAGES_PER_BATCH}\n"

    if print:
        print(config_str)

    return config_str
