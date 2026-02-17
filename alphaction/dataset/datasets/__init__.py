from .concat_dataset import ConcatDataset
from .image_dataset import ImageDataset

__all__ = ["ConcatDataset", "ImageDataset"]

# Multimodal (Image + Text) dataset support
__all__.extend([
    "MultimodalImageDataset",
    "OpenVocabularyDataset",
])

# Try to import multimodal datasets if available
try:
    from .image_dataset import MultimodalImageDataset, OpenVocabularyDataset
except ImportError:
    pass
