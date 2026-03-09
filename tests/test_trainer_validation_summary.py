import unittest

from alphaction.engine.trainer import (
    _coerce_eval_metrics,
    _extract_validation_summary,
    _format_epoch_train_summary,
    _format_validation_table_row,
    _resolve_iteration_logging,
)
from alphaction.config import cfg as default_cfg


class TestTrainerValidationSummary(unittest.TestCase):
    def test_iteration_logging_defaults_to_epoch_first(self):
        cfg = default_cfg.clone()
        cfg.freeze()

        class DummyModel:
            def __init__(self, cfg):
                self.cfg = cfg

        enabled, every = _resolve_iteration_logging(DummyModel(cfg))
        self.assertFalse(enabled)
        self.assertEqual(every, 20)

    def test_coerce_eval_metrics_accepts_tuple_payload(self):
        metrics = {"PascalBoxes_Precision/mAP@0.5IOU": 0.25}
        out = _coerce_eval_metrics((metrics, {"unused": True}))
        self.assertEqual(out, metrics)

    def test_extract_validation_summary_picks_expected_metrics(self):
        metrics = {
            "PascalBoxes_Precision/mAP@0.5IOU": 0.11,
            "PascalBoxes_Precision/mAP@0.5:0.95IOU": 0.05,
            "Detection/Precision@0.5IOU": 0.8,
            "Detection/Recall@0.5IOU": 0.6,
            "SmallObject/AP@0.5IOU": 0.02,
            "ignored": 123,
        }
        summary = _extract_validation_summary(metrics)
        self.assertEqual(
            set(summary.keys()),
            {
                "PascalBoxes_Precision/mAP@0.5IOU",
                "PascalBoxes_Precision/mAP@0.5:0.95IOU",
                "Detection/Precision@0.5IOU",
                "Detection/Recall@0.5IOU",
                "SmallObject/AP@0.5IOU",
            },
        )
        self.assertAlmostEqual(summary["Detection/Precision@0.5IOU"], 0.8)
        self.assertNotIn("ignored", summary)

    def test_format_epoch_train_summary_uses_expected_columns(self):
        epoch_summary = {
            "num_batches": 2,
            "loss_sums": {
                "loss_bbox": 4.0,
                "loss_ce": 6.0,
                "loss_giou": 2.0,
            },
            "instance_sum": 10,
            "size_label": "640",
            "gpu_mem_gb": 6.25,
        }
        row = "".join(_format_epoch_train_summary(3, 50, epoch_summary))
        self.assertIn("3/50", row)
        self.assertIn("6.25G", row)
        self.assertIn("2.0000", row)
        self.assertIn("3.0000", row)
        self.assertIn("1.0000", row)
        self.assertIn("640", row)

    def test_format_validation_table_row_uses_dataset_counts(self):
        class DummyDataset:
            samples = [
                {"labels": [0, 1]},
                {"labels": [1]},
            ]

        class DummyLoader:
            dataset = DummyDataset()

        summary = {
            "Detection/Precision@0.5IOU": 0.5,
            "Detection/Recall@0.5IOU": 0.25,
            "PascalBoxes_Precision/mAP@0.5IOU": 0.1,
            "PascalBoxes_Precision/mAP@0.5:0.95IOU": 0.04,
            "SmallObject/AP@0.5IOU": 0.01,
        }
        row = "".join(_format_validation_table_row(summary, DummyLoader()))
        self.assertIn("all", row)
        self.assertIn("2", row)
        self.assertIn("3", row)
        self.assertIn("0.5000", row)
        self.assertIn("0.2500", row)
        self.assertIn("0.1000", row)
        self.assertIn("0.0400", row)
        self.assertIn("0.0100", row)


if __name__ == "__main__":
    unittest.main()
