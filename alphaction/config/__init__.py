from pathlib import Path

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


def _find_dataset_yaml(root: Path):
    for name in ("data.yaml", "dataset.yaml"):
        path = root / name
        if path.is_file():
            return path
    return None


def _load_dataset_yaml_names(yaml_path: Path):
    try:
        import yaml
    except Exception:
        return None

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return None

    names = data.get("names", None)
    if isinstance(names, list):
        return [str(name) for name in names]
    if isinstance(names, dict):
        try:
            items = sorted(
                ((int(key), value) for key, value in names.items()),
                key=lambda item: item[0],
            )
        except Exception:
            items = list(names.items())
        return [str(value) for _, value in items]

    nc = data.get("nc", None)
    if isinstance(nc, int) and nc > 0:
        return [f"class_{idx}" for idx in range(int(nc))]
    return None


def _count_yolo_label_classes(labels_root: Path):
    if not labels_root.is_dir():
        return None

    max_id = -1
    for txt in sorted(labels_root.rglob("*.txt")):
        try:
            with open(txt, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls = int(float(parts[0]))
                    if cls > max_id:
                        max_id = cls
        except Exception:
            continue

    if max_id < 0:
        return None
    return max_id + 1


def infer_dataset_class_metadata(config):
    """Best-effort class metadata inference from the configured dataset path."""

    data_dir = str(getattr(config.DATA, "PATH_TO_DATA_DIR", "")).strip()
    if not data_dir:
        return None
    root = Path(data_dir)
    if not root.exists():
        return None

    yaml_path = _find_dataset_yaml(root)
    class_names = _load_dataset_yaml_names(yaml_path) if yaml_path is not None else None

    label_roots = [
        root / "labels" / "train",
        root / "labels" / "val",
        root / "labels" / "valid",
        root / "labels" / "test",
        root / "train" / "labels",
        root / "val" / "labels",
        root / "valid" / "labels",
        root / "test" / "labels",
        root / "labels",
    ]
    for label_root in label_roots:
        label_count = _count_yolo_label_classes(label_root)
        if label_count is None and not class_names:
            continue
        class_count = int(len(class_names)) if class_names else int(label_count)
        if class_names and len(class_names) < class_count:
            class_names = class_names + [
                f"class_{idx}" for idx in range(len(class_names), class_count)
            ]
        if not class_names:
            class_names = [f"class_{idx}" for idx in range(class_count)]
        return {
            "split": label_root.parent.name if label_root.parent != root else "dataset",
            "num_classes": int(class_count),
            "class_names": list(class_names),
        }

    try:
        from alphaction.dataset.datasets.image_dataset import ImageDataset
    except Exception:
        return None

    for split in ("train", "val", "test"):
        try:
            dataset = ImageDataset(config, split)
        except Exception:
            continue

        class_count = max(1, int(getattr(dataset, "num_classes", 0)))
        class_names = list(getattr(dataset, "class_names", []))
        return {
            "split": split,
            "num_classes": class_count,
            "class_names": class_names,
        }

    return None


def auto_sync_dataset_class_counts(config):
    """Align detector class counts to the configured dataset when possible."""

    if not bool(getattr(config.DATA, "AUTO_SYNC_CLASS_COUNTS", True)):
        return None

    metadata = infer_dataset_class_metadata(config)
    if metadata is None:
        return None

    class_count = int(metadata["num_classes"])
    changed = {}
    for field in ("ACTION_CLASSES", "OBJECT_CLASSES", "NUM_ACT", "NUM_CLS"):
        current = int(getattr(config.MODEL.STM, field))
        if current != class_count:
            setattr(config.MODEL.STM, field, class_count)
            changed[field] = {"old": current, "new": class_count}

    metadata["changed"] = changed
    metadata["applied"] = bool(changed)
    return metadata


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
        errors.append(
            f"NUM_FRAMES should be 1 for image mode, got {config.DATA.NUM_FRAMES}"
        )

    if config.SOLVER.IMAGES_PER_BATCH <= 0:
        errors.append(
            f"IMAGES_PER_BATCH must be positive, got {config.SOLVER.IMAGES_PER_BATCH}"
        )

    if config.TEST.IMAGES_PER_BATCH <= 0:
        errors.append(
            f"IMAGES_PER_BATCH must be positive in TEST, got {config.TEST.IMAGES_PER_BATCH}"
        )

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
