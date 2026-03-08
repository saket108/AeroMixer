import unittest

import torch

from alphaction.config import cfg as default_cfg
from alphaction.modeling.detector.stm_detector import AeroLiteDetector
from alphaction.modeling.stm_decoder.stm_decoder import AMStage, STMDecoder


class TestMultimodalLiteDetector(unittest.TestCase):
    def _build_cfg(self):
        cfg = default_cfg.clone()
        cfg.defrost()
        cfg.DATA.INPUT_TYPE = "image"
        cfg.DATA.IMAGE_MODE = True
        cfg.DATA.MULTIMODAL = True
        cfg.MODEL.TEXT_ENCODER = "LITE_TEXT"
        cfg.MODEL.BACKBONE.CONV_BODY = "AeroLite-Det-S"
        cfg.MODEL.BACKBONE.PATHWAYS = 1
        cfg.MODEL.STM.OBJECT_CLASSES = 3
        cfg.MODEL.STM.ACTION_CLASSES = 3
        cfg.MODEL.STM.NUM_ACT = 3
        cfg.MODEL.STM.NUM_CLS = 3
        cfg.MODEL.STM.TEXT_SCORE_FUSION = True
        cfg.MODEL.STM.TEXT_QUERY_COND = True
        cfg.freeze()
        return cfg

    def test_backbone_text_encoder_emits_normalized_class_features(self):
        cfg = self._build_cfg()
        model = AeroLiteDetector(cfg)
        model.backbone.text_encoder.set_vocabulary(
            {
                "dent": {"caption": "metal dent on fuselage"},
                "scratch": {"caption": ["surface scratch", "thin paint scratch"]},
                "corrosion": {"caption": "corrosion patch on panel"},
            }
        )

        x = torch.randn(2, 3, 224, 224)
        multiscale_feats, cls_feat = model.backbone.forward_multiscale([x])
        text_features = model.backbone.forward_text(device=torch.device("cpu"))

        self.assertEqual(len(multiscale_feats), 3)
        self.assertEqual(tuple(cls_feat.shape), (2, 384))
        self.assertEqual(tuple(text_features.shape), (3, cfg.MODEL.LITE_TEXT.EMBED_DIM))
        self.assertTrue(
            torch.allclose(text_features.norm(dim=-1), torch.ones(3), atol=1e-4)
        )

    def test_amstage_text_fusion_changes_foreground_logits(self):
        stage = AMStage(
            query_dim=16,
            feat_channels=16,
            num_heads=4,
            feedforward_channels=32,
            spatial_points=4,
            temporal_points=1,
            out_multiplier=2,
            n_groups=1,
            num_cls_fcs=1,
            num_reg_fcs=1,
            num_classes_object=3,
            text_dim=8,
            text_score_fusion=True,
            text_score_fusion_alpha=0.5,
            text_logit_scale=8.0,
            image_mode=True,
        )

        cls_score = torch.randn(1, 5, 4)
        cls_feat = torch.randn(1, 5, 16)
        text_a = torch.nn.functional.pad(torch.eye(3), (0, 5))
        text_b = text_a.roll(shifts=1, dims=0)

        fused_a = stage._apply_text_score_fusion(cls_score.clone(), cls_feat, text_a)
        fused_b = stage._apply_text_score_fusion(cls_score.clone(), cls_feat, text_b)

        self.assertFalse(torch.allclose(fused_a[..., :-1], fused_b[..., :-1]))
        self.assertTrue(torch.allclose(fused_a[..., -1], cls_score[..., -1]))

    def test_decoder_builds_label_aware_text_context(self):
        cfg = self._build_cfg()
        decoder = STMDecoder(cfg, image_mode=True)

        text_features = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        labels = [
            torch.tensor([0, 2], dtype=torch.int64),
            torch.tensor([1], dtype=torch.int64),
        ]
        vis_cls_feat = torch.randn(2, 512)

        context = decoder.get_image_text_context(
            text_features, labels=labels, vis_cls_feat=vis_cls_feat
        )

        self.assertEqual(tuple(context.shape), (2, 4))
        self.assertTrue(torch.allclose(context[0], torch.tensor([0.5, 0.0, 0.5, 0.0])))
        self.assertTrue(torch.allclose(context[1], torch.tensor([0.0, 1.0, 0.0, 0.0])))

    def test_aerolite_family_scales_increase_capacity(self):
        dims = {}
        for name in ["AeroLite-Det-T", "AeroLite-Det-S", "AeroLite-Det-B"]:
            cfg = self._build_cfg()
            cfg.defrost()
            cfg.MODEL.BACKBONE.CONV_BODY = name
            cfg.freeze()
            model = AeroLiteDetector(cfg)
            dims[name] = model.backbone.dim_out

        self.assertEqual(dims["AeroLite-Det-T"], 256)
        self.assertEqual(dims["AeroLite-Det-S"], 384)
        self.assertEqual(dims["AeroLite-Det-B"], 512)
        self.assertLess(dims["AeroLite-Det-T"], dims["AeroLite-Det-S"])
        self.assertLess(dims["AeroLite-Det-S"], dims["AeroLite-Det-B"])


if __name__ == "__main__":
    unittest.main()
