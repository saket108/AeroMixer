import logging
import unittest
from unittest import mock

from alphaction.dataset.datasets.evaluation.images import image_eval


class _DummyTestCfg:
    REPORT_AP5095 = True
    AP5095_MIN = 0.5
    AP5095_MAX = 0.95
    AP5095_STEP = 0.05


class _DummyCfg:
    TEST = _DummyTestCfg()


class _DummyDataset:
    cfg = _DummyCfg()
    closed_set_classes = ["dent"]


class TestImageEvalAP5095(unittest.TestCase):
    def test_adds_map5095_from_iou_sweep(self):
        dataset = _DummyDataset()
        logger = logging.getLogger("test_image_eval_ap5095")
        base = {"PascalBoxes_Precision/mAP@0.5IOU": 0.123}

        def _fake_frame_map_pascal(_results, _targets, _vocab, _logger, iou_list):
            out = {}
            for idx, iou in enumerate(iou_list):
                key = image_eval._ap_key_for_iou(iou)
                out[key] = 0.10 + (0.01 * idx)
            return out

        with mock.patch.object(
            image_eval, "frame_mAP_pascal", side_effect=_fake_frame_map_pascal
        ) as patched:
            out = image_eval._maybe_add_ap5095_metrics(
                dataset=dataset,
                results={},
                targets={},
                eval_res=dict(base),
                logger=logger,
            )

        self.assertTrue(patched.called)
        self.assertIn("PascalBoxes_Precision/mAP@0.5:0.95IOU", out)
        # Mean of [0.10, 0.11, ..., 0.19] = 0.145
        self.assertAlmostEqual(
            float(out["PascalBoxes_Precision/mAP@0.5:0.95IOU"]), 0.145, places=6
        )

    def test_skip_when_disabled(self):
        dataset = _DummyDataset()
        dataset.cfg.TEST.REPORT_AP5095 = False
        logger = logging.getLogger("test_image_eval_ap5095")
        base = {"PascalBoxes_Precision/mAP@0.5IOU": 0.123}
        with mock.patch.object(image_eval, "frame_mAP_pascal") as patched:
            out = image_eval._maybe_add_ap5095_metrics(
                dataset=dataset,
                results={},
                targets={},
                eval_res=dict(base),
                logger=logger,
            )
        self.assertEqual(out, base)
        patched.assert_not_called()
        dataset.cfg.TEST.REPORT_AP5095 = True


if __name__ == "__main__":
    unittest.main()
