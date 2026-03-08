"""Supported backbone registry for the active AeroLite detector family."""

from alphaction.modeling import registry

from .image_resnet import ImageResNetLiteText


SUPPORTED_BACKBONES = (
    "AeroLite-Det-T",
    "AeroLite-Det-S",
    "AeroLite-Det-B",
)


for backbone_name in SUPPORTED_BACKBONES:
    registry.BACKBONES.register(backbone_name, ImageResNetLiteText)


def build_backbone(cfg):
    conv_body = str(getattr(cfg.MODEL.BACKBONE, "CONV_BODY", "")).strip()
    if conv_body not in registry.BACKBONES:
        supported = ", ".join(SUPPORTED_BACKBONES)
        raise ValueError(
            f"Unsupported backbone '{conv_body}'. Supported backbones: {supported}."
        )
    return registry.BACKBONES[conv_body](cfg)
