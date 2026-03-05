import unittest

import torch
from torch import nn

from alphaction.config import cfg as base_cfg
from alphaction.modeling.stm_decoder.util.loss import SetCriterion


class _IdentityMatcher(nn.Module):
    @torch.no_grad()
    def forward(self, outputs, targets):
        _ = outputs
        indices = []
        for target in targets:
            n = int(target["labels"].numel())
            src = torch.arange(n, dtype=torch.int64)
            tgt = torch.arange(n, dtype=torch.int64)
            indices.append((src, tgt))
        return indices


class TestClassWeightedLoss(unittest.TestCase):
    def test_inverse_freq_boosts_rare_class_weight(self):
        cfg = base_cfg.clone()
        cfg.defrost()
        cfg.MODEL.STM.CLASS_WEIGHTING = "inverse_freq"
        cfg.MODEL.STM.CLASS_WEIGHT_POWER = 1.0
        cfg.MODEL.STM.CLASS_WEIGHT_MIN = 0.25
        cfg.MODEL.STM.CLASS_WEIGHT_MAX = 4.0
        cfg.MODEL.STM.CLASS_WEIGHT_EMA = 0.0
        cfg.freeze()

        criterion = SetCriterion(
            cfg=cfg,
            num_classes=3,
            matcher=_IdentityMatcher(),
            eos_coef=0.1,
            losses=["labels"],
        )

        outputs = {
            "pred_logits": torch.randn(1, 4, 4),
            "pred_boxes": torch.zeros(1, 4, 4),
        }
        targets = [
            {
                "labels": torch.tensor([0, 0, 0, 2], dtype=torch.int64),
            }
        ]
        _ = criterion(outputs, targets)
        weights = criterion.last_class_weights.detach().cpu()
        self.assertGreater(float(weights[2]), float(weights[0]))
        self.assertAlmostEqual(float(weights[3]), 0.1, places=6)

    def test_none_mode_keeps_uniform_fg_weights(self):
        cfg = base_cfg.clone()
        cfg.defrost()
        cfg.MODEL.STM.CLASS_WEIGHTING = "none"
        cfg.freeze()

        criterion = SetCriterion(
            cfg=cfg,
            num_classes=3,
            matcher=_IdentityMatcher(),
            eos_coef=0.1,
            losses=["labels"],
        )

        outputs = {
            "pred_logits": torch.randn(1, 2, 4),
            "pred_boxes": torch.zeros(1, 2, 4),
        }
        targets = [{"labels": torch.tensor([0, 1], dtype=torch.int64)}]
        _ = criterion(outputs, targets)
        weights = criterion.last_class_weights.detach().cpu()
        self.assertAlmostEqual(float(weights[0]), 1.0, places=6)
        self.assertAlmostEqual(float(weights[1]), 1.0, places=6)
        self.assertAlmostEqual(float(weights[2]), 1.0, places=6)
        self.assertAlmostEqual(float(weights[3]), 0.1, places=6)


if __name__ == "__main__":
    unittest.main()
