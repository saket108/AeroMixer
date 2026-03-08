"""Compatibility aliases for the active image-only detector."""

from .stm_detector import AeroLiteDetector, STMDetector, build_aerolite_detector, build_detection_model


MultimodalActionDetector = AeroLiteDetector
ActionDetector = AeroLiteDetector


__all__ = [
    "AeroLiteDetector",
    "ActionDetector",
    "MultimodalActionDetector",
    "STMDetector",
    "build_aerolite_detector",
    "build_detection_model",
]
