"""AeroLite detector for image multimodal models."""

import logging

from torch import nn
import torch
import torch.nn.functional as F

from alphaction.config import is_image_mode, uses_text_branch
from ..backbone import build_backbone
from ..stm_decoder.stm_decoder import build_stm_decoder


logger = logging.getLogger(__name__)

class LayerNorm(nn.Module):
    """LayerNorm that supports channels_first tensors (N,C,...) and channels_last."""

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)

        shape = [1, -1] + [1] * (x.ndim - 2)
        return x * self.weight.view(*shape) + self.bias.view(*shape)



SUPPORTED_DETECTORS = ("AeroLiteDetector", "STMDetector")


class AeroLiteDetector(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.backbone = build_backbone(cfg)

        self.is_image = is_image_mode(cfg)
        if not self.is_image:
            raise RuntimeError(
                "AeroLiteDetector supports image mode only. "
                "Set DATA.INPUT_TYPE='image' and DATA.IMAGE_MODE=True."
            )

        # Build STM decoder with image_mode parameter.
        self.stm_head = build_stm_decoder(cfg, image_mode=self.is_image)

        # Lightweight projection for image mode.
        if self.is_image:
            hidden_dim = cfg.MODEL.STM.HIDDEN_DIM
            in_dim = int(getattr(self.backbone, "dim_embed", hidden_dim))
            self.img_proj = nn.Sequential(
                nn.Conv2d(in_dim, hidden_dim, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            )
            self.pyramid_scales = (1, 2, 4, 8)
            self.pyramid_adapters = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                        nn.ReLU(inplace=True),
                    )
                    for _ in self.pyramid_scales
                ]
            )
            self.use_backbone_fpn = False
            pyramid_channels = getattr(self.backbone, "pyramid_channels", None)
            if isinstance(pyramid_channels, (list, tuple)) and len(pyramid_channels) >= 3:
                c3_ch, c4_ch, c5_ch = [int(ch) for ch in pyramid_channels[-3:]]
                self.use_backbone_fpn = True
                self.fpn_lateral = nn.ModuleList(
                    [
                        nn.Conv2d(c3_ch, hidden_dim, kernel_size=1),
                        nn.Conv2d(c4_ch, hidden_dim, kernel_size=1),
                        nn.Conv2d(c5_ch, hidden_dim, kernel_size=1),
                    ]
                )
                self.fpn_output = nn.ModuleList(
                    [
                        nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                        nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                        nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    ]
                )
                self.fpn_extra = nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                )

        logger.info(
            "AeroLiteDetector initialized in image-only detection mode (backbone=%s).",
            cfg.MODEL.BACKBONE.CONV_BODY,
        )

    def _merge_extras(self, extras, labels=None):
        if isinstance(extras, dict):
            merged = dict(extras)
        elif isinstance(extras, (list, tuple)):
            merged = {}
            for item in extras:
                if not isinstance(item, dict):
                    continue
                for key, value in item.items():
                    merged.setdefault(key, []).append(value)
        else:
            merged = {}

        if labels is not None and "labels" not in merged:
            merged["labels"] = labels
        return merged

    def _collect_numeric_tensors(self, value, bucket):
        if value is None:
            return
        if torch.is_tensor(value):
            if value.numel() > 0:
                bucket.append(value)
            return
        if isinstance(value, dict):
            for child in value.values():
                self._collect_numeric_tensors(child, bucket)
            return
        if isinstance(value, (list, tuple)):
            for child in value:
                self._collect_numeric_tensors(child, bucket)
            return
        try:
            tensor = torch.as_tensor(value)
        except Exception:
            return
        if tensor.numel() > 0:
            bucket.append(tensor)

    def _stack_text_features(self, value, device, dtype):
        candidates = []
        self._collect_numeric_tensors(value, candidates)

        prepared = []
        for tensor in candidates:
            tensor = tensor.to(device=device, dtype=dtype)
            if tensor.ndim == 0:
                continue
            if tensor.ndim == 1:
                prepared.append(tensor.unsqueeze(0))
            else:
                prepared.append(tensor.reshape(-1, tensor.shape[-1]))

        if not prepared:
            return None

        dim = prepared[0].size(-1)
        aligned = []
        for tensor in prepared:
            if tensor.size(-1) > dim:
                aligned.append(tensor[:, :dim])
            elif tensor.size(-1) < dim:
                pad = torch.zeros(
                    (tensor.size(0), dim - tensor.size(-1)),
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
                aligned.append(torch.cat([tensor, pad], dim=1))
            else:
                aligned.append(tensor)

        return torch.cat(aligned, dim=0)

    def _encode_backbone_text(self, device, dtype):
        if not hasattr(self.backbone, "forward_text"):
            return None

        try:
            text = self.backbone.forward_text(device=device)
        except TypeError:
            try:
                text = self.backbone.forward_text()
            except Exception:
                return None
        except Exception:
            return None

        return self._stack_text_features(text, device=device, dtype=dtype)

    def _resolve_text_features(self, extras, device, dtype):
        text_features = self._stack_text_features(extras.get("text_features"), device=device, dtype=dtype)
        if text_features is not None:
            return text_features

        if uses_text_branch(self.cfg):
            return self._encode_backbone_text(device=device, dtype=dtype)

        return None

    def _pick_feature_tensor(self, features):
        if torch.is_tensor(features):
            return features
        if isinstance(features, (list, tuple)):
            for item in features:
                tensor = self._pick_feature_tensor(item)
                if tensor is not None:
                    return tensor
        return None

    def _to_image_feature(self, feat):
        if feat is None:
            return None
        if feat.dim() == 5:
            return feat.mean(dim=2)
        if feat.dim() != 4:
            raise ValueError(f"Expected 4D/5D feature map, got shape={tuple(feat.shape)}")
        return feat

    def _build_image_pyramid(self, feats):
        if feats.dim() != 4:
            raise ValueError(f"Expected 4D feature map for pyramid, got shape={tuple(feats.shape)}")

        h, w = feats.shape[-2:]
        pyramid = []
        for scale, adapter in zip(self.pyramid_scales, self.pyramid_adapters):
            target_h = max(1, h // scale)
            target_w = max(1, w // scale)
            if target_h == h and target_w == w:
                level = feats
            else:
                level = F.interpolate(
                    feats,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
            level = adapter(level)
            pyramid.append(level.unsqueeze(2))
        return pyramid

    def _build_backbone_fpn(self, multiscale_feats):
        if not isinstance(multiscale_feats, (list, tuple)) or len(multiscale_feats) < 3:
            raise ValueError("Expected at least three feature levels for FPN.")

        c3 = self._to_image_feature(multiscale_feats[-3])
        c4 = self._to_image_feature(multiscale_feats[-2])
        c5 = self._to_image_feature(multiscale_feats[-1])

        l3 = self.fpn_lateral[0](c3)
        l4 = self.fpn_lateral[1](c4)
        l5 = self.fpn_lateral[2](c5)

        p5 = l5
        p4 = l4 + F.interpolate(p5, size=l4.shape[-2:], mode="nearest")
        p3 = l3 + F.interpolate(p4, size=l3.shape[-2:], mode="nearest")

        p3 = self.fpn_output[0](p3)
        p4 = self.fpn_output[1](p4)
        p5 = self.fpn_output[2](p5)
        p6 = self.fpn_extra(p5)

        return [p3.unsqueeze(2), p4.unsqueeze(2), p5.unsqueeze(2), p6.unsqueeze(2)]

    def forward_image(self, primary_inputs, whwh, boxes=None, labels=None, extras=None):
        extras = self._merge_extras(extras, labels=labels)

        cls_feat = None
        patch_feat = None
        multiscale_feats = None

        if self.use_backbone_fpn and hasattr(self.backbone, "forward_multiscale"):
            backbone_out = self.backbone.forward_multiscale([primary_inputs])
            if isinstance(backbone_out, (list, tuple)) and len(backbone_out) >= 1:
                multiscale_feats = backbone_out[0]
                if isinstance(backbone_out, (list, tuple)) and len(backbone_out) > 1 and torch.is_tensor(backbone_out[1]):
                    cls_feat = backbone_out[1]
                if isinstance(multiscale_feats, (list, tuple)) and len(multiscale_feats) > 0:
                    patch_feat = multiscale_feats[-1]

        if patch_feat is None:
            backbone_out = self.backbone([primary_inputs])
            if isinstance(backbone_out, (list, tuple)):
                patch_feat = backbone_out[0]
                if len(backbone_out) > 1 and torch.is_tensor(backbone_out[1]):
                    cls_feat = backbone_out[1]
            else:
                patch_feat = backbone_out

        if self.use_backbone_fpn and multiscale_feats is not None:
            mapped_features = self._build_backbone_fpn(multiscale_feats)
            feature_dtype = mapped_features[0].dtype
        else:
            feats = self._pick_feature_tensor(patch_feat)
            if feats is None:
                raise RuntimeError("AeroLiteDetector could not extract a tensor feature map from backbone output.")
            feats = self._to_image_feature(feats)
            feats = self.img_proj(feats)
            mapped_features = self._build_image_pyramid(feats)
            feature_dtype = feats.dtype

        text_features = self._resolve_text_features(extras, device=primary_inputs.device, dtype=feature_dtype)

        return self.stm_head(
            mapped_features,
            whwh,
            gt_boxes=boxes,
            labels=labels,
            text_features=text_features,
            tau_inv=100,
            cls_feat=cls_feat,
            patch_feat=patch_feat,
            text_token_feats=None,
            extras=extras,
        )

    def forward(self, primary_inputs, secondary_inputs=None, whwh=None,
                boxes=None, labels=None, extras=None, part_forward=-1):
        if whwh is None:
            raise ValueError("AeroLiteDetector.forward requires `whwh` image sizes.")
        return self.forward_image(primary_inputs, whwh, boxes, labels, extras)


STMDetector = AeroLiteDetector


def build_detection_model(cfg):
    det_name = str(getattr(cfg.MODEL, "DET", "AeroLiteDetector")).strip()
    if det_name not in SUPPORTED_DETECTORS:
        supported = ", ".join(SUPPORTED_DETECTORS)
        raise ValueError(f"Unsupported detector '{det_name}'. Supported detectors: {supported}.")
    return AeroLiteDetector(cfg)


def build_aerolite_detector(cfg):
    return AeroLiteDetector(cfg)
