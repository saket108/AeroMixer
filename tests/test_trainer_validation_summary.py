import unittest

from alphaction.engine.trainer import _coerce_eval_metrics, _extract_validation_summary


class TestTrainerValidationSummary(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
