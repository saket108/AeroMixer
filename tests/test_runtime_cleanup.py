import unittest

from alphaction.config import cfg as default_cfg, uses_text_branch
from alphaction.modeling.detector.action_detector import (
    ActionDetector,
    MultimodalActionDetector,
)
from alphaction.modeling.detector.stm_detector import AeroLiteDetector, STMDetector
from alphaction.modeling.runtime import (
    configure_text_encoder,
    get_backbone,
    has_text_encoder,
)


class TestRuntimeCleanup(unittest.TestCase):
    def _build_cfg(self):
        cfg = default_cfg.clone()
        cfg.defrost()
        cfg.DATA.INPUT_TYPE = "image"
        cfg.DATA.IMAGE_MODE = True
        cfg.DATA.MULTIMODAL = True
        cfg.MODEL.TEXT_ENCODER = "LITE_TEXT"
        cfg.MODEL.BACKBONE.CONV_BODY = "AeroLite-Det-S"
        cfg.freeze()
        return cfg

    def test_action_detector_aliases_point_to_active_detector(self):
        self.assertIs(ActionDetector, AeroLiteDetector)
        self.assertIs(MultimodalActionDetector, AeroLiteDetector)
        self.assertIs(STMDetector, AeroLiteDetector)

    def test_runtime_helpers_configure_backbone_text_encoder(self):
        cfg = self._build_cfg()
        model = AeroLiteDetector(cfg)
        vocab = {
            "dent": {"caption": "a photo of dent"},
            "scratch": {"caption": "a photo of scratch"},
        }

        self.assertTrue(uses_text_branch(cfg))
        self.assertTrue(has_text_encoder(model))
        self.assertIsNotNone(get_backbone(model))
        self.assertTrue(configure_text_encoder(model, vocab))
        self.assertEqual(
            list(model.backbone.text_encoder.text_data.keys()), ["dent", "scratch"]
        )


if __name__ == "__main__":
    unittest.main()
