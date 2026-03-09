import unittest

import numpy as np
import torch

from alphaction.dataset.datasets.evaluation.images.image_eval import (
    _compute_detection_precision_recall,
    _format_console_eval_summary,
    _prepare_for_image_ap,
)


class _DummyDataset:
    def __init__(self, multilabel_action: bool):
        self.multilabel_action = multilabel_action
        self.num_classes = 2
        self.closed_set_classes = ["a", "b"]
        self.test_iou_thresh = 0.5
        self.samples = [
            {
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "labels": np.zeros((0,), dtype=np.int64),
            }
        ]

    def get_sample_info(self, _index):
        return {
            "image_id": "img0.jpg",
            "width": 100,
            "height": 100,
            "resolution": (100, 100),
        }


class TestImageEvalPostprocess(unittest.TestCase):
    def test_single_label_mode_emits_one_class_per_query(self):
        ds = _DummyDataset(multilabel_action=False)
        boxes = torch.tensor(
            [
                [0.10, 0.10, 0.20, 0.20],
                [0.30, 0.30, 0.40, 0.40],
            ],
            dtype=torch.float32,
        )
        # logits over [class0, class1, background]
        logits = torch.tensor(
            [
                [
                    2.0,
                    1.9,
                    1.0,
                ],  # top-1 = class0; class1 also high but should NOT become another detection
                [0.2, 0.1, 3.0],  # mostly background
            ],
            dtype=torch.float32,
        )

        predictions = [(boxes, logits)]
        results, _targets = _prepare_for_image_ap(predictions, ds, score_thresh=0.30)
        det = results["img0.jpg"]
        self.assertEqual(det["boxes"].shape[0], 1)
        self.assertEqual(det["action_ids"].tolist(), [1])  # class0 + 1 for evaluator

    def test_multilabel_mode_keeps_multi_class_hits(self):
        ds = _DummyDataset(multilabel_action=True)
        boxes = torch.tensor([[0.10, 0.10, 0.20, 0.20]], dtype=torch.float32)
        # sigmoid scores on [class0, class1, background_like]
        logits = torch.tensor([[3.0, 2.0, -2.0]], dtype=torch.float32)

        predictions = [(boxes, logits)]
        results, _targets = _prepare_for_image_ap(predictions, ds, score_thresh=0.50)
        det = results["img0.jpg"]
        self.assertEqual(det["boxes"].shape[0], 2)
        self.assertEqual(sorted(det["action_ids"].tolist()), [1, 2])

    def test_detection_precision_recall_counts_tp_fp_fn(self):
        results = {
            "img0.jpg": {
                "boxes": np.asarray(
                    [[0.10, 0.10, 0.30, 0.30], [0.60, 0.60, 0.80, 0.80]],
                    dtype=np.float32,
                ),
                "scores": np.asarray([0.95, 0.40], dtype=np.float32),
                "action_ids": np.asarray([1, 1], dtype=np.int64),
            }
        }
        targets = {
            "img0.jpg": {
                "bbox": np.asarray([[0.10, 0.10, 0.30, 0.30]], dtype=np.float32),
                "labels": [1],
                "resolution": (100, 100),
            }
        }

        metrics = _compute_detection_precision_recall(results, targets, iou_thresh=0.5)
        self.assertAlmostEqual(metrics["Detection/Precision@0.5IOU"], 0.5, places=6)
        self.assertAlmostEqual(metrics["Detection/Recall@0.5IOU"], 1.0, places=6)
        self.assertEqual(metrics["Detection/TP@0.5IOU"], 1)
        self.assertEqual(metrics["Detection/FP@0.5IOU"], 1)
        self.assertEqual(metrics["Detection/FN@0.5IOU"], 0)

    def test_console_eval_summary_uses_clean_metric_names(self):
        summary = _format_console_eval_summary(
            {
                "Detection/Precision@0.5IOU": 0.5,
                "Detection/Recall@0.5IOU": 0.25,
                "PascalBoxes_Precision/mAP@0.5IOU": 0.1,
                "PascalBoxes_Precision/mAP@0.5:0.95IOU": 0.04,
                "SmallObject/AP@0.5IOU": -1.0,
            }
        )
        self.assertIn("precision=0.5000", summary)
        self.assertIn("recall=0.2500", summary)
        self.assertIn("mAP50=0.1000", summary)
        self.assertIn("mAP50-95=0.0400", summary)
        self.assertIn("smallAP=n/a", summary)
        self.assertNotIn("PascalBoxes", summary)


if __name__ == "__main__":
    unittest.main()
