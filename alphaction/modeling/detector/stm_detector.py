"""AeroLite detector for image multimodal models."""

import logging

from torch import nn
import torch
import torch.nn.functional as F

from alphaction.config import is_image_mode, uses_text_branch
from ..backbone import build_backbone
from ..stm_decoder.stm_decoder import build_stm_decoder


logger = logging.getLogger(__name__)


def _resolve_text_encoder_dim(cfg):
    name = str(getattr(cfg.MODEL, "TEXT_ENCODER", "LITE_TEXT")).strip() or "LITE_TEXT"
    encoder_cfg = getattr(cfg.MODEL, name, None)
    if encoder_cfg is None:
        return int(getattr(cfg.MODEL.STM, "HIDDEN_DIM", 256))
    return int(
        getattr(encoder_cfg, "EMBED_DIM", getattr(cfg.MODEL.STM, "HIDDEN_DIM", 256))
    )


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
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )

        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)

        shape = [1, -1] + [1] * (x.ndim - 2)
        return x * self.weight.view(*shape) + self.bias.view(*shape)


class ScaleTextRouter(nn.Module):
    """Route multiscale image features with prompt-conditioned level gates."""

    def __init__(
        self,
        feature_dim,
        text_dim,
        num_levels,
        hidden_dim=128,
        gain=0.75,
        temperature=1.0,
    ):
        super().__init__()
        self.num_levels = int(num_levels)
        self.gain = float(gain)
        self.temperature = max(float(temperature), 1e-4)
        self.feature_proj = nn.Linear(int(feature_dim), int(hidden_dim))
        self.text_proj = nn.Linear(int(text_dim), int(hidden_dim), bias=False)
        self.router_norm = nn.LayerNorm(int(hidden_dim))
        self.router_logits = nn.Linear(int(hidden_dim), 1)
        # Earlier FPN levels correspond to smaller object priors.
        self.register_buffer(
            "level_object_scales",
            torch.linspace(
                0.65, 1.55, steps=max(self.num_levels, 1), dtype=torch.float32
            ),
        )

    def _pool_feature(self, feat):
        if feat.dim() == 5:
            feat = feat.mean(dim=2)
        if feat.dim() != 4:
            raise ValueError(
                f"Expected routed feature map to be 4D/5D, got shape={tuple(feat.shape)}"
            )
        return feat.flatten(2).mean(dim=-1)

    def forward(self, features, text_context):
        if not features or text_context is None:
            return features, None

        if text_context.ndim == 1:
            text_context = text_context.unsqueeze(0)

        level_tokens = torch.stack(
            [self._pool_feature(feat) for feat in features], dim=1
        )
        joint = self.feature_proj(level_tokens) + self.text_proj(
            text_context
        ).unsqueeze(1)
        logits = (
            self.router_logits(self.router_norm(torch.tanh(joint))).squeeze(-1)
            / self.temperature
        )
        level_weights = torch.softmax(logits, dim=1)

        centered = (
            level_weights - (1.0 / max(level_weights.size(1), 1))
        ) * level_weights.size(1)
        routed_features = []
        for idx, feat in enumerate(features):
            gate = 1.0 + self.gain * centered[:, idx]
            gate = gate.clamp(min=0.25, max=2.50)
            gate = gate.view(feat.size(0), *([1] * (feat.dim() - 1)))
            routed_features.append(feat * gate)

        level_scales = self.level_object_scales[: level_weights.size(1)].to(
            device=level_weights.device,
            dtype=level_weights.dtype,
        )
        object_scale = (level_weights * level_scales.unsqueeze(0)).sum(dim=1)
        routing_summary = {
            "level_weights": level_weights,
            "object_scale": object_scale,
        }
        return routed_features, routing_summary


class TileGlobalContextFusion(nn.Module):
    """Fuse tile position metadata with pooled multiscale features."""

    def __init__(
        self,
        feature_dim,
        text_dim,
        num_levels,
        meta_dim=12,
        hidden_dim=128,
        level_gain=0.35,
        blend=0.20,
    ):
        super().__init__()
        self.num_levels = int(num_levels)
        self.level_gain = float(level_gain)
        self.blend = float(blend)
        self.feature_proj = nn.Linear(int(feature_dim), int(hidden_dim))
        self.meta_proj = nn.Sequential(
            nn.Linear(int(meta_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
        )
        self.text_proj = nn.Linear(int(text_dim), int(hidden_dim), bias=False)
        self.level_norm = nn.LayerNorm(int(hidden_dim))
        self.level_logits = nn.Linear(int(hidden_dim), 1)
        self.context_proj = nn.Linear(int(hidden_dim), int(text_dim))

    def _pool_feature(self, feat):
        if feat.dim() == 5:
            feat = feat.mean(dim=2)
        if feat.dim() != 4:
            raise ValueError(
                f"Expected tile context feature map to be 4D/5D, got shape={tuple(feat.shape)}"
            )
        return feat.flatten(2).mean(dim=-1)

    def forward(self, features, tile_meta, text_context=None):
        if not features or tile_meta is None:
            return features, None

        level_tokens = torch.stack(
            [self._pool_feature(feat) for feat in features], dim=1
        )
        meta_context = self.meta_proj(tile_meta)
        joint = self.feature_proj(level_tokens) + meta_context.unsqueeze(1)

        if text_context is not None:
            if text_context.ndim == 1:
                text_context = text_context.unsqueeze(0)
            joint = joint + self.text_proj(text_context).unsqueeze(1)

        logits = self.level_logits(self.level_norm(torch.tanh(joint))).squeeze(-1)
        level_weights = torch.softmax(logits, dim=1)

        centered = (
            level_weights - (1.0 / max(level_weights.size(1), 1))
        ) * level_weights.size(1)
        routed_features = []
        for idx, feat in enumerate(features):
            gate = 1.0 + self.level_gain * centered[:, idx]
            gate = gate.clamp(min=0.50, max=2.00)
            gate = gate.view(feat.size(0), *([1] * (feat.dim() - 1)))
            routed_features.append(feat * gate)

        fused_visual = (level_weights.unsqueeze(-1) * self.feature_proj(level_tokens)).sum(
            dim=1
        )
        context = self.context_proj(torch.tanh(fused_visual + meta_context))
        context = F.normalize(context, dim=-1, eps=1e-6)

        summary = {
            "context": context,
            "blend": self.blend,
            "level_weights": level_weights,
            "coverage": tile_meta[:, 4],
            "edge_proximity": tile_meta[:, 9],
            "tile_meta": tile_meta,
        }
        return routed_features, summary


class DefectPrototypeMemory(nn.Module):
    """Learn class prototypes from boxed defect regions and fuse them into text prompts."""

    def __init__(
        self,
        feature_dim,
        text_dim,
        num_classes,
        momentum=0.90,
        blend=0.35,
        context_blend=0.25,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.momentum = float(momentum)
        self.blend = float(blend)
        self.context_blend = float(context_blend)
        self.visual_proj = nn.Linear(int(feature_dim), int(text_dim), bias=False)
        self.visual_norm = nn.LayerNorm(int(text_dim))
        self.register_buffer(
            "prototype_bank", torch.zeros(self.num_classes, int(text_dim))
        )
        self.register_buffer(
            "prototype_counts", torch.zeros(self.num_classes, dtype=torch.float32)
        )
        self.register_buffer(
            "prototype_initialized",
            torch.zeros(self.num_classes, dtype=torch.bool),
        )

    def _prepare_text_bank(self, text_features, device, dtype):
        if text_features is None:
            return None
        if not torch.is_tensor(text_features):
            text_features = torch.as_tensor(text_features)
        text_features = text_features.to(device=device, dtype=dtype)
        if text_features.ndim == 1:
            text_features = text_features.unsqueeze(0)
        elif text_features.ndim == 3:
            text_features = text_features.mean(dim=1)
        elif text_features.ndim > 3:
            text_features = text_features.reshape(-1, text_features.shape[-1])
        if text_features.numel() == 0:
            return None
        if text_features.size(0) > self.num_classes:
            return text_features[: self.num_classes]
        if text_features.size(0) < self.num_classes:
            fill = text_features.mean(dim=0, keepdim=True).expand(
                self.num_classes - text_features.size(0), -1
            )
            return torch.cat([text_features, fill], dim=0)
        return text_features

    def _to_feature_map(self, feature_map):
        if feature_map is None:
            return None
        if feature_map.dim() == 5:
            feature_map = feature_map.mean(dim=2)
        if feature_map.dim() != 4:
            raise ValueError(
                f"Expected 4D/5D prototype feature map, got shape={tuple(feature_map.shape)}"
            )
        return feature_map

    def _extract_class_ids(self, label, device):
        if label is None:
            return None
        label = torch.as_tensor(label, device=device)
        if label.numel() == 0:
            return None
        if label.ndim == 1:
            class_ids = label.long()
        elif label.ndim == 2:
            class_ids = torch.argmax(label, dim=-1).long()
        else:
            return None
        class_ids = class_ids[(class_ids >= 0) & (class_ids < self.num_classes)]
        return class_ids if class_ids.numel() > 0 else None

    def _project_visual(self, visual_feats):
        projected = self.visual_proj(visual_feats)
        projected = self.visual_norm(projected)
        return F.normalize(projected, dim=-1, eps=1e-6)

    def _normalize_boxes(self, boxes_img, frame_size, device):
        boxes = torch.as_tensor(boxes_img, dtype=torch.float32, device=device)
        if boxes.numel() == 0:
            return boxes.reshape(0, 4)
        boxes = boxes.reshape(-1, 4)
        if boxes.max() <= 1.5:
            scale = frame_size.view(1, 4).to(device=device, dtype=boxes.dtype)
            boxes = boxes * scale
        return boxes

    def _pool_box_features(self, feature_map, boxes, labels, whwh):
        pooled_feats = []
        pooled_labels = []
        batch_size, channels, height, width = feature_map.shape
        for idx in range(batch_size):
            class_ids = None
            if labels is not None and idx < len(labels):
                class_ids = self._extract_class_ids(labels[idx], device=feature_map.device)
            if class_ids is None or class_ids.numel() == 0:
                continue

            boxes_img = self._normalize_boxes(
                boxes[idx],
                frame_size=whwh[idx],
                device=feature_map.device,
            )
            if boxes_img.numel() == 0:
                continue

            num_pairs = min(boxes_img.size(0), class_ids.numel())
            if num_pairs <= 0:
                continue

            img_w = max(float(whwh[idx, 0].item()), 1.0)
            img_h = max(float(whwh[idx, 1].item()), 1.0)
            feat_map = feature_map[idx]
            for box, class_id in zip(boxes_img[:num_pairs], class_ids[:num_pairs]):
                x1 = int(torch.floor(box[0] / img_w * width).item())
                y1 = int(torch.floor(box[1] / img_h * height).item())
                x2 = int(torch.ceil(box[2] / img_w * width).item())
                y2 = int(torch.ceil(box[3] / img_h * height).item())

                x1 = max(0, min(width - 1, x1))
                y1 = max(0, min(height - 1, y1))
                x2 = max(x1 + 1, min(width, x2))
                y2 = max(y1 + 1, min(height, y2))

                region = feat_map[:, y1:y2, x1:x2]
                if region.numel() == 0:
                    region = feat_map[:, y1 : y1 + 1, x1 : x1 + 1]
                pooled_feats.append(region.mean(dim=(1, 2)))
                pooled_labels.append(int(class_id.item()))

        if not pooled_feats:
            return None, None
        return torch.stack(pooled_feats, dim=0), torch.tensor(
            pooled_labels, device=feature_map.device, dtype=torch.long
        )

    def _update_memory(self, feature_map, boxes, labels, whwh):
        if boxes is None or labels is None:
            return
        pooled_feats, pooled_labels = self._pool_box_features(
            feature_map, boxes=boxes, labels=labels, whwh=whwh
        )
        if pooled_feats is None or pooled_labels is None or pooled_feats.numel() == 0:
            return

        visual_proto = self._project_visual(pooled_feats.detach())
        with torch.no_grad():
            for class_id in pooled_labels.unique(sorted=True):
                cls_mask = pooled_labels == class_id
                cls_proto = visual_proto[cls_mask].mean(dim=0)
                bank_idx = int(class_id.item())
                if self.prototype_initialized[bank_idx]:
                    updated = (
                        self.momentum * self.prototype_bank[bank_idx]
                        + (1.0 - self.momentum) * cls_proto
                    )
                else:
                    updated = cls_proto
                self.prototype_bank[bank_idx] = F.normalize(
                    updated, dim=0, eps=1e-6
                )
                self.prototype_counts[bank_idx] += float(cls_mask.sum().item())
                self.prototype_initialized[bank_idx] = True

    def _build_similarity_context(self, feature_map, prototype_bank):
        pooled = feature_map.flatten(2).mean(dim=-1)
        projected = self._project_visual(pooled)
        logits = torch.matmul(projected, prototype_bank.t())
        weights = torch.softmax(logits, dim=-1)
        return torch.matmul(weights, prototype_bank)

    def _build_label_context(self, labels, prototype_bank, device, dtype):
        if labels is None:
            return None
        contexts = []
        default_context = prototype_bank.mean(dim=0)
        for label in labels:
            class_ids = self._extract_class_ids(label, device=device)
            if class_ids is None or class_ids.numel() == 0:
                contexts.append(default_context)
                continue
            valid_mask = self.prototype_initialized[class_ids].to(device=device)
            if bool(valid_mask.any()):
                contexts.append(prototype_bank[class_ids[valid_mask]].mean(dim=0))
            else:
                contexts.append(default_context)
        if not contexts:
            return None
        return torch.stack(contexts, dim=0).to(device=device, dtype=dtype)

    def forward(self, text_features, feature_map, whwh, boxes=None, labels=None):
        feature_map = self._to_feature_map(feature_map)
        if feature_map is None:
            return text_features, None

        text_bank = self._prepare_text_bank(
            text_features,
            device=feature_map.device,
            dtype=feature_map.dtype,
        )
        if text_bank is None:
            return text_features, None

        if self.training:
            self._update_memory(feature_map, boxes=boxes, labels=labels, whwh=whwh)

        initialized = self.prototype_initialized.to(device=feature_map.device)
        if not bool(initialized.any()):
            return text_bank, None

        prototype_bank = self.prototype_bank.to(
            device=feature_map.device,
            dtype=text_bank.dtype,
        )
        fused_text = text_bank.clone()
        fused_text[initialized] = F.normalize(
            (1.0 - self.blend) * text_bank[initialized]
            + self.blend * prototype_bank[initialized],
            dim=-1,
            eps=1e-6,
        )

        similarity_context = self._build_similarity_context(
            feature_map, prototype_bank[initialized]
        )
        label_context = self._build_label_context(
            labels=labels,
            prototype_bank=prototype_bank,
            device=feature_map.device,
            dtype=text_bank.dtype,
        )
        image_context = similarity_context
        if label_context is not None and label_context.size(0) == similarity_context.size(0):
            image_context = F.normalize(
                (1.0 - self.context_blend) * label_context
                + self.context_blend * similarity_context,
                dim=-1,
                eps=1e-6,
            )

        summary = {
            "image_context": image_context,
            "context_blend": self.context_blend,
            "initialized_mask": initialized,
            "counts": self.prototype_counts.to(device=feature_map.device),
        }
        return fused_text, summary


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
            if (
                isinstance(pyramid_channels, (list, tuple))
                and len(pyramid_channels) >= 3
            ):
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
                    nn.Conv2d(
                        hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1
                    ),
                    nn.ReLU(inplace=True),
                )
            self.scale_text_router = None
            self.tile_global_context = None
            self.prototype_memory = None
            if uses_text_branch(cfg) and bool(
                getattr(cfg.MODEL.STM, "PROTOTYPE_MEMORY", False)
            ):
                self.prototype_memory = DefectPrototypeMemory(
                    feature_dim=hidden_dim,
                    text_dim=_resolve_text_encoder_dim(cfg),
                    num_classes=int(cfg.MODEL.STM.OBJECT_CLASSES),
                    momentum=float(
                        getattr(cfg.MODEL.STM, "PROTOTYPE_MEMORY_MOMENTUM", 0.90)
                    ),
                    blend=float(
                        getattr(cfg.MODEL.STM, "PROTOTYPE_MEMORY_BLEND", 0.35)
                    ),
                    context_blend=float(
                        getattr(
                            cfg.MODEL.STM, "PROTOTYPE_MEMORY_CONTEXT_BLEND", 0.25
                        )
                    ),
                )
            if uses_text_branch(cfg) and bool(
                getattr(cfg.MODEL.STM, "SCALE_TEXT_ROUTING", False)
            ):
                self.scale_text_router = ScaleTextRouter(
                    feature_dim=hidden_dim,
                    text_dim=_resolve_text_encoder_dim(cfg),
                    num_levels=len(self.pyramid_scales),
                    hidden_dim=int(
                        getattr(cfg.MODEL.STM, "SCALE_TEXT_ROUTING_HIDDEN", 128)
                    ),
                    gain=float(getattr(cfg.MODEL.STM, "SCALE_TEXT_ROUTING_GAIN", 0.75)),
                    temperature=float(
                        getattr(cfg.MODEL.STM, "SCALE_TEXT_ROUTING_TEMP", 1.0)
                    ),
                )
            if uses_text_branch(cfg) and bool(
                getattr(cfg.MODEL.STM, "TILE_GLOBAL_CONTEXT", False)
            ):
                self.tile_global_context = TileGlobalContextFusion(
                    feature_dim=hidden_dim,
                    text_dim=_resolve_text_encoder_dim(cfg),
                    num_levels=len(self.pyramid_scales),
                    hidden_dim=int(
                        getattr(cfg.MODEL.STM, "TILE_GLOBAL_CONTEXT_HIDDEN", 128)
                    ),
                    level_gain=float(
                        getattr(cfg.MODEL.STM, "TILE_GLOBAL_CONTEXT_LEVEL_GAIN", 0.35)
                    ),
                    blend=float(
                        getattr(cfg.MODEL.STM, "TILE_GLOBAL_CONTEXT_BLEND", 0.20)
                    ),
                )
        else:
            self.scale_text_router = None
            self.tile_global_context = None

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
        text_features = self._stack_text_features(
            extras.get("text_features"), device=device, dtype=dtype
        )
        if text_features is not None:
            return text_features

        if uses_text_branch(self.cfg):
            return self._encode_backbone_text(device=device, dtype=dtype)

        return None

    def _build_tile_meta_tensor(self, extras, batch_size, device, dtype):
        if not isinstance(extras, dict):
            return None

        raw_meta = extras.get("tile_meta")
        if raw_meta is None:
            return None
        if isinstance(raw_meta, dict):
            raw_meta = [raw_meta]
        if not isinstance(raw_meta, (list, tuple)):
            return None

        meta_vectors = []
        has_tiled_sample = False
        for idx in range(batch_size):
            item = raw_meta[idx] if idx < len(raw_meta) else None
            if not isinstance(item, dict) or not bool(item.get("is_tiled", False)):
                meta_vectors.append(torch.zeros(10, device=device, dtype=dtype))
                continue

            has_tiled_sample = True
            position = torch.as_tensor(
                item.get("position_norm", [0.5, 0.5]), device=device, dtype=dtype
            ).reshape(-1)[:2]
            size = torch.as_tensor(
                item.get("size_norm", [1.0, 1.0]), device=device, dtype=dtype
            ).reshape(-1)[:2]
            border = torch.as_tensor(
                item.get("border_norm", [0.0, 0.0, 0.0, 0.0]),
                device=device,
                dtype=dtype,
            ).reshape(-1)[:4]
            coverage = torch.tensor(
                [float(item.get("coverage_ratio", 1.0))], device=device, dtype=dtype
            )
            edge = torch.tensor(
                [float(item.get("edge_proximity", 0.0))], device=device, dtype=dtype
            )
            grid = torch.as_tensor(
                item.get("grid_position", [0.0, 0.0]), device=device, dtype=dtype
            ).reshape(-1)[:2]
            meta_vectors.append(
                torch.cat([position, size, coverage, border, edge, grid], dim=0)
            )

        if not has_tiled_sample:
            return None
        return torch.stack(meta_vectors, dim=0)

    def _extract_class_ids(self, label, device, num_classes):
        if label is None:
            return None
        label = torch.as_tensor(label, device=device)
        if label.numel() == 0:
            return None
        if label.ndim == 1:
            class_ids = label.long()
        elif label.ndim == 2:
            class_ids = torch.argmax(label, dim=-1).long()
        else:
            return None
        class_ids = class_ids[(class_ids >= 0) & (class_ids < num_classes)]
        return class_ids if class_ids.numel() > 0 else None

    def _build_text_context(self, text_features, labels, batch_size, device, dtype):
        if text_features is None:
            return None

        if text_features.ndim == 1:
            text_bank = text_features.unsqueeze(0)
        elif text_features.ndim == 2:
            text_bank = text_features
        elif text_features.ndim == 3:
            text_bank = text_features.mean(dim=1)
        else:
            text_bank = text_features.reshape(-1, text_features.shape[-1])

        text_bank = text_bank.to(device=device, dtype=dtype)
        if text_bank.numel() == 0:
            return None

        default_context = text_bank.mean(dim=0)
        contexts = []
        for idx in range(batch_size):
            class_ids = None
            if labels is not None and idx < len(labels):
                class_ids = self._extract_class_ids(
                    labels[idx], device=device, num_classes=text_bank.size(0)
                )

            if class_ids is not None and class_ids.numel() > 0:
                contexts.append(text_bank[class_ids.unique()].mean(dim=0))
            else:
                contexts.append(default_context)

        return torch.stack(contexts, dim=0)

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
            raise ValueError(
                f"Expected 4D/5D feature map, got shape={tuple(feat.shape)}"
            )
        return feat

    def _build_image_pyramid(self, feats):
        if feats.dim() != 4:
            raise ValueError(
                f"Expected 4D feature map for pyramid, got shape={tuple(feats.shape)}"
            )

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
                if (
                    isinstance(backbone_out, (list, tuple))
                    and len(backbone_out) > 1
                    and torch.is_tensor(backbone_out[1])
                ):
                    cls_feat = backbone_out[1]
                if (
                    isinstance(multiscale_feats, (list, tuple))
                    and len(multiscale_feats) > 0
                ):
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
                raise RuntimeError(
                    "AeroLiteDetector could not extract a tensor feature map from backbone output."
                )
            feats = self._to_image_feature(feats)
            feats = self.img_proj(feats)
            mapped_features = self._build_image_pyramid(feats)
            feature_dtype = feats.dtype

        text_features = self._resolve_text_features(
            extras, device=primary_inputs.device, dtype=feature_dtype
        )
        text_context = None
        if self.prototype_memory is not None and text_features is not None:
            text_features, prototype_summary = self.prototype_memory(
                text_features=text_features,
                feature_map=mapped_features[0],
                whwh=whwh,
                boxes=boxes,
                labels=labels,
            )
            if prototype_summary is not None:
                extras = dict(extras)
                extras["prototype_memory"] = prototype_summary

        if text_features is not None and (
            self.scale_text_router is not None or self.tile_global_context is not None
        ):
            text_context = self._build_text_context(
                text_features,
                labels=labels,
                batch_size=primary_inputs.size(0),
                device=primary_inputs.device,
                dtype=feature_dtype,
            )

        if self.scale_text_router is not None and text_context is not None:
            mapped_features, routing_summary = self.scale_text_router(
                mapped_features, text_context
            )
            if routing_summary is not None:
                extras = dict(extras)
                extras["scale_routing"] = routing_summary
        if self.tile_global_context is not None:
            tile_meta = self._build_tile_meta_tensor(
                extras,
                batch_size=primary_inputs.size(0),
                device=primary_inputs.device,
                dtype=feature_dtype,
            )
            mapped_features, tile_summary = self.tile_global_context(
                mapped_features, tile_meta, text_context=text_context
            )
            if tile_summary is not None:
                extras = dict(extras)
                extras["tile_global_context"] = tile_summary

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

    def forward(
        self,
        primary_inputs,
        secondary_inputs=None,
        whwh=None,
        boxes=None,
        labels=None,
        extras=None,
        part_forward=-1,
    ):
        if whwh is None:
            raise ValueError("AeroLiteDetector.forward requires `whwh` image sizes.")
        return self.forward_image(primary_inputs, whwh, boxes, labels, extras)


STMDetector = AeroLiteDetector


def build_detection_model(cfg):
    det_name = str(getattr(cfg.MODEL, "DET", "AeroLiteDetector")).strip()
    if det_name not in SUPPORTED_DETECTORS:
        supported = ", ".join(SUPPORTED_DETECTORS)
        raise ValueError(
            f"Unsupported detector '{det_name}'. Supported detectors: {supported}."
        )
    return AeroLiteDetector(cfg)


def build_aerolite_detector(cfg):
    return AeroLiteDetector(cfg)
