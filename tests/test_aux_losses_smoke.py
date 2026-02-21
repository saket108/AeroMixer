import unittest

import torch

from alphaction.modeling.stm_decoder.util.loss import HungarianMatcher, SetCriterion


def _make_boxes(batch_size, num_queries):
    x1y1 = torch.rand(batch_size, num_queries, 2) * 32.0
    wh = torch.rand(batch_size, num_queries, 2) * 16.0 + 1.0
    x2y2 = x1y1 + wh
    return torch.cat([x1y1, x2y2], dim=-1)


class TestAuxLossesSmoke(unittest.TestCase):
    def test_setcriterion_returns_aux_stage_losses(self):
        num_classes = 2
        matcher = HungarianMatcher(cost_class=2.0, cost_bbox=2.0, cost_giou=2.0)
        criterion = SetCriterion(
            num_classes=num_classes,
            matcher=matcher,
            eos_coef=0.1,
            losses=["labels", "boxes"],
        )

        outputs = {
            "pred_logits": torch.randn(1, 4, num_classes + 1),
            "pred_boxes": _make_boxes(1, 4),
            "aux_outputs": [
                {
                    "pred_logits": torch.randn(1, 4, num_classes + 1),
                    "pred_boxes": _make_boxes(1, 4),
                },
                {
                    "pred_logits": torch.randn(1, 4, num_classes + 1),
                    "pred_boxes": _make_boxes(1, 4),
                },
            ],
        }
        targets = [
            {
                "labels": torch.tensor([0, 1], dtype=torch.int64),
                "boxes_xyxy": torch.tensor(
                    [
                        [2.0, 3.0, 12.0, 15.0],
                        [10.0, 8.0, 22.0, 25.0],
                    ],
                    dtype=torch.float32,
                ),
            }
        ]

        losses = criterion(outputs, targets)

        self.assertIn("loss_ce", losses)
        self.assertIn("loss_bbox", losses)
        self.assertIn("loss_giou", losses)
        self.assertIn("loss_ce_0", losses)
        self.assertIn("loss_bbox_0", losses)
        self.assertIn("loss_giou_0", losses)
        self.assertIn("loss_ce_1", losses)
        self.assertIn("loss_bbox_1", losses)
        self.assertIn("loss_giou_1", losses)

    def test_setcriterion_returns_severity_losses(self):
        matcher = HungarianMatcher(cost_class=2.0, cost_bbox=2.0, cost_giou=2.0)
        criterion = SetCriterion(
            num_classes=2,
            matcher=matcher,
            eos_coef=0.1,
            losses=["labels", "boxes", "severity"],
        )

        outputs = {
            "pred_logits": torch.randn(1, 3, 3),
            "pred_boxes": _make_boxes(1, 3),
            "pred_severity": torch.randn(1, 3),
            "aux_outputs": [
                {
                    "pred_logits": torch.randn(1, 3, 3),
                    "pred_boxes": _make_boxes(1, 3),
                    "pred_severity": torch.randn(1, 3),
                }
            ],
        }
        targets = [
            {
                "labels": torch.tensor([0, 1], dtype=torch.int64),
                "boxes_xyxy": torch.tensor(
                    [
                        [2.0, 3.0, 12.0, 15.0],
                        [10.0, 8.0, 22.0, 25.0],
                    ],
                    dtype=torch.float32,
                ),
                "severity": torch.tensor([0.2, 0.9], dtype=torch.float32),
            }
        ]

        losses = criterion(outputs, targets)
        self.assertIn("loss_severity", losses)
        self.assertIn("loss_severity_0", losses)


if __name__ == "__main__":
    unittest.main()
