from alphaction.modeling import registry
from .image_resnet import ImageResNet, ImageResNetLite
from .vit_utils import ImageVisionTransformerEncoder, ImageVisionTransformerDecoder

# =============================================================================
# Image Backbones (for image + text multimodal models)
# =============================================================================

@registry.BACKBONES.register("ImageResNet-50")
@registry.BACKBONES.register("ImageResNet-101")
@registry.BACKBONES.register("ImageResNet-152")
def build_image_resnet_backbone(cfg):
    model = ImageResNet(cfg)
    return model


@registry.BACKBONES.register("ImageResNet-Lite")
def build_image_resnet_lite_backbone(cfg):
    model = ImageResNetLite(cfg)
    return model


# =============================================================================
# Image Vision Transformer Backbones
# =============================================================================

@registry.BACKBONES.register("ImageViT-B")
@registry.BACKBONES.register("ImageViT-L")
@registry.BACKBONES.register("ImageViT-H")
def build_image_vit_backbone(cfg):
    model = ImageVisionTransformerEncoder(
        img_size=cfg.ViT.IMAGE_SIZE,
        patch_size=cfg.ViT.PATCH_SIZE,
        in_chans=cfg.ViT.IN_CHANS,
        embed_dim=cfg.ViT.EMBED_DIM,
        depth=cfg.ViT.DEPTH,
        num_heads=cfg.ViT.NUM_HEADS,
        mlp_ratio=cfg.ViT.MLP_RATIO,
        qkv_bias=cfg.ViT.QKV_BIAS,
        drop_rate=cfg.ViT.DROP_RATE,
        attn_drop_rate=cfg.ViT.ATTN_DROP_RATE,
        drop_path_rate=cfg.ViT.DROP_PATH_RATE,
    )
    return model


# =============================================================================
# CLIP/ViT Encoders (from external libraries)
# =============================================================================

# OpenAI CLIP
@registry.BACKBONES.register("ViT-B/16")
@registry.BACKBONES.register("ViT-B/32")
@registry.BACKBONES.register("ViT-L/14")
def build_clip_vit_backbone(cfg):
    from alphaction.modeling.encoders.openai_clip.clip_encoder import build_clip_backbone
    model = build_clip_backbone(cfg)
    return model


# CLIP-ViP
@registry.BACKBONES.register("ViP-B/16")
@registry.BACKBONES.register("ViP-B/32")
def build_clipvip_backbone(cfg):
    from alphaction.modeling.encoders.clipvip.clipvip_encoder import build_clipvip_backbone as build_clipvip_model
    image_mode = getattr(cfg.DATA, "INPUT_TYPE", "image") == "image"
    model = build_clipvip_model(cfg, image_mode=image_mode)
    return model


# ViCLIP from InternVideo
@registry.BACKBONES.register("ViCLIP-L/14")
def build_viclip_backbone(cfg):
    from alphaction.modeling.encoders.viclip.viclip_encoder import build_viclip_backbone
    model = build_viclip_backbone(cfg)
    return model


def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)
