"""Image backbones for pure image and lightweight image+text detection."""

import hashlib
import re
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_AEROLITE_VARIANTS = {
    "aerolite-det-t": {
        "stem_channels": 32,
        "stage_channels": (64, 128, 256),
        "block_depths": (1, 1, 1),
    },
    "aerolite-det-s": {
        "stem_channels": 48,
        "stage_channels": (96, 192, 384),
        "block_depths": (1, 2, 2),
    },
    "aerolite-det-b": {
        "stem_channels": 64,
        "stage_channels": (128, 256, 512),
        "block_depths": (2, 2, 2),
    },
}

_SUPPORTED_AEROLITE_VARIANTS = tuple(_AEROLITE_VARIANTS.keys())
_PUBLIC_AEROLITE_VARIANTS = ("AeroLite-Det-T", "AeroLite-Det-S", "AeroLite-Det-B")


def _prepare_image_input(x, source_name):
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            raise ValueError(f"{source_name} received an empty input list.")
        x = x[0]

    if x.ndim == 5:
        x = x.mean(dim=2)

    if x.ndim != 4:
        raise ValueError(f"{source_name} expects 4D input after preprocessing, got shape={tuple(x.shape)}")
    return x


def _resolve_aerolite_variant(cfg):
    conv_body = str(getattr(cfg.MODEL.BACKBONE, "CONV_BODY", "")).strip().lower()
    if conv_body not in _AEROLITE_VARIANTS:
        supported = ", ".join(_PUBLIC_AEROLITE_VARIANTS)
        raise ValueError(
            f"Unsupported AeroLite variant '{conv_body}'. Supported variants: {supported}."
        )
    return conv_body, _AEROLITE_VARIANTS[conv_body]


class ImageResNetLite(nn.Module):
    """Internal CNN pyramid used by the AeroLite detector family."""

    def __init__(self, cfg, stem_channels=64, stage_channels=(128, 256, 512), block_depths=(1, 1, 1)):
        super(ImageResNetLite, self).__init__()

        self.num_pathways = 1
        if len(stage_channels) != 3 or len(block_depths) != 3:
            raise ValueError("ImageResNetLite expects three stage channels and three stage depths.")

        self.conv1 = nn.Conv2d(3, stem_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stem_channels)
        self.relu = nn.ReLU(inplace=True)

        self.block2 = self._make_stage(stem_channels, stage_channels[0], stride=2, depth=block_depths[0])
        self.block3 = self._make_stage(stage_channels[0], stage_channels[1], stride=2, depth=block_depths[1])
        self.block4 = self._make_stage(stage_channels[1], stage_channels[2], stride=2, depth=block_depths[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dim_out = int(stage_channels[-1])
        self.dim_embed = int(stage_channels[-1])
        self.pyramid_channels = [int(ch) for ch in stage_channels]
        self.stem_channels = int(stem_channels)
        self.stage_channels = tuple(int(ch) for ch in stage_channels)
        self.block_depths = tuple(int(depth) for depth in block_depths)

    def _make_block(self, in_ch, out_ch, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _make_stage(self, in_ch, out_ch, stride, depth):
        blocks = [self._make_block(in_ch, out_ch, stride=stride)]
        for _ in range(max(0, int(depth) - 1)):
            blocks.append(self._make_block(out_ch, out_ch, stride=1))
        return nn.Sequential(*blocks)

    def _prepare_input(self, x):
        return _prepare_image_input(x, "ImageResNetLite")

    def forward_multiscale(self, x):
        x = self._prepare_input(x)

        x = self.relu(self.bn1(self.conv1(x)))
        c3 = self.block2(x)
        c4 = self.block3(c3)
        c5 = self.block4(c4)

        cls_feat = self.avgpool(c5)
        cls_feat = torch.flatten(cls_feat, 1)
        return (c3, c4, c5), cls_feat

    def forward(self, x):
        multiscale_feats, cls_feat = self.forward_multiscale(x)
        patch_feat = multiscale_feats[2]
        return patch_feat, cls_feat


def _normalize_vocab_variants(value, default_text):
    if isinstance(value, dict):
        value = value.get("caption", value.get("text", value.get("prompt", default_text)))

    if isinstance(value, (list, tuple)):
        variants = [str(item).strip() for item in value if str(item).strip()]
    else:
        text = str(value).strip() if value is not None else ""
        variants = [text] if text else []

    if not variants:
        fallback = str(default_text).strip()
        variants = [fallback if fallback else "object"]

    return variants


class LiteTextEncoder(nn.Module):
    """Small text encoder for class prompts and closed/open vocabulary labels."""

    def __init__(self, cfg):
        super().__init__()

        text_cfg = getattr(cfg.MODEL, "LITE_TEXT")
        self.embed_dim = int(getattr(text_cfg, "EMBED_DIM", cfg.MODEL.STM.HIDDEN_DIM))
        self.max_tokens = int(getattr(text_cfg, "MAX_TOKENS", 12))
        self.max_variants = int(getattr(text_cfg, "MAX_VARIANTS", 4))
        self.context_tokens = int(getattr(text_cfg, "CONTEXT_TOKENS", 4))
        self.vocab_size = max(128, int(getattr(text_cfg, "VOCAB_SIZE", 4096)))
        self.dropout = float(getattr(text_cfg, "DROPOUT", 0.0))
        ff_dim = int(getattr(text_cfg, "FFN_DIM", self.embed_dim * 2))
        num_layers = int(getattr(text_cfg, "NUM_LAYERS", 2))
        num_heads = max(1, int(getattr(text_cfg, "NUM_HEADS", 4)))

        self.token_embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)
        self.position_embedding = nn.Parameter(
            torch.zeros(self.max_tokens + self.context_tokens, self.embed_dim)
        )
        self.context_prompt = (
            nn.Parameter(torch.zeros(self.context_tokens, self.embed_dim))
            if self.context_tokens > 0
            else None
        )
        self.input_norm = nn.LayerNorm(self.embed_dim)
        self.input_dropout = nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity()

        if num_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=self.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            self.transformer = None

        self.output_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.LayerNorm(self.embed_dim),
        )

        self.register_buffer(
            "vocab_token_ids",
            torch.zeros(0, self.max_variants, self.max_tokens, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "vocab_token_mask",
            torch.zeros(0, self.max_variants, self.max_tokens, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "vocab_variant_mask",
            torch.zeros(0, self.max_variants, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "direct_text_embeddings",
            torch.zeros(0, self.embed_dim, dtype=torch.float32),
            persistent=False,
        )

        self.text_data = OrderedDict()
        self.vocab_names = []
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        if self.token_embedding.padding_idx is not None:
            with torch.no_grad():
                self.token_embedding.weight[self.token_embedding.padding_idx].zero_()
        nn.init.normal_(self.position_embedding, std=0.01)
        if self.context_prompt is not None:
            nn.init.normal_(self.context_prompt, std=0.02)
        for module in self.output_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _stable_hash(self, token):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return (int.from_bytes(digest, "little") % (self.vocab_size - 1)) + 1

    def _tokenize(self, text):
        tokens = _TOKEN_RE.findall(str(text).lower())
        if not tokens:
            tokens = ["object"]
        tokens = tokens[: self.max_tokens]

        token_ids = torch.zeros(self.max_tokens, dtype=torch.long)
        token_mask = torch.zeros(self.max_tokens, dtype=torch.bool)
        for idx, token in enumerate(tokens):
            token_ids[idx] = self._stable_hash(token)
            token_mask[idx] = True
        return token_ids, token_mask

    def set_vocabulary(self, text_data, embeddings=None):
        self.text_data = OrderedDict()
        self.vocab_names = []

        if not text_data:
            self.vocab_token_ids = torch.zeros(
                0, self.max_variants, self.max_tokens, dtype=torch.long
            )
            self.vocab_token_mask = torch.zeros(
                0, self.max_variants, self.max_tokens, dtype=torch.bool
            )
            self.vocab_variant_mask = torch.zeros(0, self.max_variants, dtype=torch.bool)
            self.direct_text_embeddings = torch.zeros(0, self.embed_dim, dtype=torch.float32)
            return

        items = text_data.items() if isinstance(text_data, dict) else enumerate(text_data)
        normalized_items = []
        for raw_key, raw_value in items:
            key = str(raw_key).strip() if str(raw_key).strip() else f"class_{len(normalized_items)}"
            variants = _normalize_vocab_variants(raw_value, key)[: self.max_variants]
            self.text_data[key] = {"caption": list(variants)}
            self.vocab_names.append(key)
            normalized_items.append((key, variants))

        num_classes = len(normalized_items)
        token_ids = torch.zeros(num_classes, self.max_variants, self.max_tokens, dtype=torch.long)
        token_mask = torch.zeros(num_classes, self.max_variants, self.max_tokens, dtype=torch.bool)
        variant_mask = torch.zeros(num_classes, self.max_variants, dtype=torch.bool)

        for cls_idx, (_, variants) in enumerate(normalized_items):
            for var_idx, caption in enumerate(variants):
                ids, mask = self._tokenize(caption)
                token_ids[cls_idx, var_idx] = ids
                token_mask[cls_idx, var_idx] = mask
                variant_mask[cls_idx, var_idx] = True

        self.vocab_token_ids = token_ids
        self.vocab_token_mask = token_mask
        self.vocab_variant_mask = variant_mask

        if embeddings is None:
            self.direct_text_embeddings = torch.zeros(0, self.embed_dim, dtype=torch.float32)
            return

        direct = torch.as_tensor(embeddings, dtype=torch.float32)
        if direct.ndim == 1:
            direct = direct.unsqueeze(0)
        if direct.size(-1) > self.embed_dim:
            direct = direct[:, : self.embed_dim]
        elif direct.size(-1) < self.embed_dim:
            pad = torch.zeros(direct.size(0), self.embed_dim - direct.size(-1), dtype=direct.dtype)
            direct = torch.cat([direct, pad], dim=-1)
        self.direct_text_embeddings = F.normalize(direct, dim=-1)

    def _encode_sequences(self, token_ids, token_mask):
        x = self.token_embedding(token_ids)
        mask = token_mask

        if self.context_prompt is not None:
            prompt = self.context_prompt.unsqueeze(0).expand(x.size(0), -1, -1)
            prompt_mask = torch.ones(
                x.size(0),
                self.context_tokens,
                dtype=torch.bool,
                device=token_ids.device,
            )
            x = torch.cat([prompt, x], dim=1)
            mask = torch.cat([prompt_mask, token_mask], dim=1)

        pos = self.position_embedding[: x.size(1)].unsqueeze(0).to(device=x.device, dtype=x.dtype)
        x = self.input_norm(x + pos)
        x = self.input_dropout(x)

        if self.transformer is not None:
            x = self.transformer(x, src_key_padding_mask=~mask)

        weights = mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        pooled = self.output_proj(pooled)
        return F.normalize(pooled, dim=-1)

    def forward(self, device=None, cond=None):
        return self.forward_text(device=device, cond=cond)

    def forward_text(self, device=None, cond=None):
        if device is None:
            device = self.position_embedding.device

        if self.direct_text_embeddings.numel() > 0:
            return self.direct_text_embeddings.to(device=device)

        if self.vocab_token_ids.numel() == 0:
            return torch.zeros(0, self.embed_dim, device=device, dtype=self.position_embedding.dtype)

        token_ids = self.vocab_token_ids.to(device=device)
        token_mask = self.vocab_token_mask.to(device=device)
        variant_mask = self.vocab_variant_mask.to(device=device)

        flat_ids = token_ids.view(-1, self.max_tokens)
        flat_mask = token_mask.view(-1, self.max_tokens)
        active = variant_mask.reshape(-1)
        if not bool(active.any().item()):
            return torch.zeros(token_ids.size(0), self.embed_dim, device=device, dtype=self.position_embedding.dtype)

        encoded = self._encode_sequences(flat_ids[active], flat_mask[active])
        all_encoded = torch.zeros(flat_ids.size(0), self.embed_dim, device=device, dtype=encoded.dtype)
        all_encoded[active] = encoded
        all_encoded = all_encoded.view(token_ids.size(0), self.max_variants, self.embed_dim)

        variant_weights = variant_mask.unsqueeze(-1).to(all_encoded.dtype)
        text_features = (all_encoded * variant_weights).sum(dim=1)
        text_features = text_features / variant_weights.sum(dim=1).clamp(min=1.0)
        return F.normalize(text_features, dim=-1)


class ImageResNetLiteText(ImageResNetLite):
    """AeroLite image+text backbone with an integrated prompt encoder."""

    def __init__(self, cfg):
        variant_name, variant_spec = _resolve_aerolite_variant(cfg)
        super().__init__(
            cfg,
            stem_channels=variant_spec["stem_channels"],
            stage_channels=variant_spec["stage_channels"],
            block_depths=variant_spec["block_depths"],
        )
        self.text_encoder = LiteTextEncoder(cfg)
        self.text_dim = self.text_encoder.embed_dim
        self.variant_name = variant_name

    def forward_text(self, device=None, cond=None):
        return self.text_encoder.forward_text(device=device, cond=cond)
