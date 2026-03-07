import unittest

import torch
import torch.nn as nn

from alphaction.config import cfg as default_cfg
from alphaction.modeling.encoders.openai_clip.clip_encoder import CLIPVisualEncoder


class _FakeResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.ln = nn.LayerNorm(width)
        self.fc = nn.Linear(width, width)

    def forward(self, x):
        return x + 0.1 * self.fc(self.ln(x))


class _FakeTransformer(nn.Module):
    def __init__(self, width: int, depth: int):
        super().__init__()
        self.width = width
        self.resblocks = nn.ModuleList(
            [_FakeResidualBlock(width) for _ in range(depth)]
        )


class _FakeVisual(nn.Module):
    def __init__(self, width: int = 8, depth: int = 4, input_resolution: int = 32, patch_size: int = 16):
        super().__init__()
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(
            3, width, kernel_size=patch_size, stride=patch_size, bias=False
        )
        self.class_embedding = nn.Parameter(torch.zeros(width))
        num_tokens = (input_resolution // patch_size) ** 2 + 1
        self.positional_embedding = nn.Parameter(torch.zeros(num_tokens, width))
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = _FakeTransformer(width, depth)
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(torch.eye(width))


class _FakeClipModel:
    def __init__(self):
        self.visual = _FakeVisual()


class TestCLIPMultiscaleEncoder(unittest.TestCase):
    def test_forward_multiscale_returns_three_feature_levels(self):
        cfg = default_cfg.clone()
        cfg.defrost()
        cfg.MODEL.CLIP.USE_CHECKPOINT = False
        cfg.MODEL.STM.USE_CLS_FEAT = False
        cfg.freeze()

        encoder = CLIPVisualEncoder(cfg, _FakeClipModel(), dtype=torch.float32)
        feats, cls_feat = encoder.forward_multiscale([torch.randn(2, 3, 32, 32)])

        self.assertEqual(len(feats), 3)
        self.assertIsNone(cls_feat)
        for feat in feats:
            self.assertEqual(tuple(feat.shape), (2, 8, 1, 2, 2))
            self.assertTrue(torch.isfinite(feat).all().item())


if __name__ == "__main__":
    unittest.main()
