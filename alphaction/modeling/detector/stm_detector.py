"""
STM Detector for image/video multimodal models.
"""

from torch import nn
import torch
import torch.nn.functional as F

from ..backbone import build_backbone
from ..stm_decoder.stm_decoder import build_stm_decoder

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



class STMDetector(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.backbone = build_backbone(cfg)

        # Detect dataset mode - check both INPUT_TYPE and IMAGE_MODE.
        self.is_image = cfg.DATA.INPUT_TYPE == "image" or getattr(cfg.DATA, "IMAGE_MODE", False)

        # Build STM decoder with image_mode parameter.
        self.stm_head = build_stm_decoder(cfg, image_mode=self.is_image)

        # Lightweight projection for image mode.
        if self.is_image:
            hidden_dim = cfg.MODEL.STM.HIDDEN_DIM
            in_dim = int(getattr(self.backbone, "dim_embed", hidden_dim))
            self.img_proj = nn.Sequential(
                nn.Conv2d(in_dim, hidden_dim, 1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            )

        print(">>>> STMDetector running in", "IMAGE MODE" if self.is_image else "VIDEO MODE")

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

        if self.cfg.DATA.OPEN_VOCABULARY:
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

    # --------------------------------------------------------
    # IMAGE FORWARD
    # --------------------------------------------------------
    def forward_image(self, primary_inputs, whwh, boxes=None, labels=None, extras=None):
        extras = self._merge_extras(extras, labels=labels)

        backbone_out = self.backbone([primary_inputs])
        cls_feat = None
        patch_feat = None

        if isinstance(backbone_out, (list, tuple)):
            patch_feat = backbone_out[0]
            if len(backbone_out) > 1 and torch.is_tensor(backbone_out[1]):
                cls_feat = backbone_out[1]
        else:
            patch_feat = backbone_out

        feats = self._pick_feature_tensor(patch_feat)
        if feats is None:
            raise RuntimeError("STMDetector could not extract a tensor feature map from backbone output.")

        # Remove temporal dim if it exists.
        if feats.dim() == 5:
            feats = feats.mean(dim=2)

        feats = self.img_proj(feats)

        # Fake multi-scale list for decoder compatibility.
        mapped_features = [feats.unsqueeze(2)] * 4

        text_features = self._resolve_text_features(extras, device=primary_inputs.device, dtype=feats.dtype)

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

    # --------------------------------------------------------
    # VIDEO FORWARD
    # --------------------------------------------------------
    def forward_video(self, primary_inputs, secondary_inputs, whwh, boxes, labels, extras):
        extras = self._merge_extras(extras, labels=labels)

        if self.backbone.num_pathways == 1:
            features = self.backbone([primary_inputs])
        else:
            features = self.backbone([primary_inputs, secondary_inputs])

        text_features = self._resolve_text_features(extras, device=primary_inputs.device, dtype=primary_inputs.dtype)

        return self.stm_head(
            features,
            whwh,
            gt_boxes=boxes,
            labels=labels,
            text_features=text_features,
            tau_inv=100,
            cls_feat=None,
            patch_feat=None,
            text_token_feats=None,
            extras=extras,
        )

    # --------------------------------------------------------
    # UNIVERSAL FORWARD
    # --------------------------------------------------------
    def forward(self, primary_inputs, secondary_inputs, whwh,
                boxes=None, labels=None, extras=None, part_forward=-1):

        if self.is_image:
            return self.forward_image(primary_inputs, whwh, boxes, labels, extras)
        return self.forward_video(primary_inputs, secondary_inputs, whwh, boxes, labels, extras)


def build_detection_model(cfg):
    return STMDetector(cfg)
