import unittest

import torch

from alphaction.config import cfg as default_cfg
from alphaction.modeling.detector.stm_detector import LayerNorm, STMDetector
from alphaction.modeling.stm_decoder.stm_decoder import STMDecoder
from alphaction.modeling.stm_decoder.util.head_utils import decode_box


class TestModelForwardSmoke(unittest.TestCase):
    def test_layernorm_forward(self):
        module = LayerNorm(8)
        x = torch.randn(2, 8, 2, 8, 8)
        y = module(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertTrue(torch.isfinite(y).all().item())

    def test_resnet_lite_backbone_fpn_shapes(self):
        cfg = default_cfg.clone()
        cfg.defrost()
        cfg.DATA.INPUT_TYPE = "image"
        cfg.DATA.IMAGE_MODE = True
        cfg.MODEL.BACKBONE.CONV_BODY = "ImageResNet-Lite"
        cfg.MODEL.BACKBONE.PATHWAYS = 1
        cfg.freeze()

        model = STMDetector(cfg)
        self.assertTrue(model.use_backbone_fpn)

        x = torch.randn(2, 3, 224, 224)
        multiscale_feats, _ = model.backbone.forward_multiscale([x])
        pyramid = model._build_backbone_fpn(multiscale_feats)

        self.assertEqual(len(pyramid), 4)
        self.assertEqual(tuple(pyramid[0].shape), (2, cfg.MODEL.STM.HIDDEN_DIM, 1, 56, 56))
        self.assertEqual(tuple(pyramid[1].shape), (2, cfg.MODEL.STM.HIDDEN_DIM, 1, 28, 28))
        self.assertEqual(tuple(pyramid[2].shape), (2, cfg.MODEL.STM.HIDDEN_DIM, 1, 14, 14))
        self.assertEqual(tuple(pyramid[3].shape), (2, cfg.MODEL.STM.HIDDEN_DIM, 1, 7, 7))

    def test_query_init_learnable_anchors(self):
        cfg = default_cfg.clone()
        cfg.defrost()
        cfg.MODEL.STM.QUERY_INIT_MODE = "learnable_anchors"
        cfg.MODEL.STM.QUERY_INIT_SMALL_OBJECT_BIAS = True
        cfg.freeze()

        decoder = STMDecoder(cfg, image_mode=True)
        whwh = torch.tensor([[640.0, 640.0, 640.0, 640.0]], dtype=torch.float32)
        xyzr = decoder._box_init(whwh, extras={})
        boxes = decode_box(xyzr)

        self.assertEqual(tuple(boxes.shape[:2]), (1, cfg.MODEL.STM.NUM_QUERIES))
        widths = boxes[0, :, 2] - boxes[0, :, 0]
        heights = boxes[0, :, 3] - boxes[0, :, 1]
        self.assertTrue(torch.all(widths > 0).item())
        self.assertTrue(torch.all(heights > 0).item())
        self.assertLess(float(widths.mean()), 640.0 * 0.6)
        centers = torch.stack(
            [(boxes[0, :, 0] + boxes[0, :, 2]) * 0.5, (boxes[0, :, 1] + boxes[0, :, 3]) * 0.5],
            dim=-1,
        )
        self.assertGreater(int(torch.unique(centers.round(), dim=0).shape[0]), 4)

    def test_attention_telemetry_outputs_metrics(self):
        cfg = default_cfg.clone()
        cfg.defrost()
        cfg.DATA.INPUT_TYPE = "image"
        cfg.DATA.IMAGE_MODE = True
        cfg.MODEL.STM.NUM_QUERIES = 16
        cfg.MODEL.STM.NUM_STAGES = 2
        cfg.MODEL.STM.OBJECT_CLASSES = 3
        cfg.MODEL.STM.ATTN_TELEMETRY = True
        cfg.MODEL.STM.ATTN_TELEMETRY_STAGEWISE = True
        cfg.MODEL.STM.ATTN_TELEMETRY_COMPARE_NOMASK = True
        cfg.MODEL.STM.IOF_TAU_MODE = "zero"
        cfg.freeze()

        decoder = STMDecoder(cfg, image_mode=True)
        decoder.train()

        features = [
            torch.randn(1, cfg.MODEL.STM.HIDDEN_DIM, 1, 8, 8),
            torch.randn(1, cfg.MODEL.STM.HIDDEN_DIM, 1, 4, 4),
            torch.randn(1, cfg.MODEL.STM.HIDDEN_DIM, 1, 2, 2),
            torch.randn(1, cfg.MODEL.STM.HIDDEN_DIM, 1, 1, 1),
        ]
        whwh = torch.tensor([[64.0, 64.0, 64.0, 64.0]], dtype=torch.float32)
        gt_boxes = [torch.tensor([[10.0, 10.0, 30.0, 30.0]], dtype=torch.float32)]
        labels = [torch.tensor([1], dtype=torch.int64)]

        losses = decoder(
            features,
            whwh,
            gt_boxes=gt_boxes,
            labels=labels,
            extras={},
        )

        self.assertIn("loss_ce", losses)
        self.assertIn("loss_bbox", losses)
        self.assertIn("attn_entropy_avg", losses)
        self.assertIn("attn_diag_avg", losses)
        self.assertIn("attn_s0_entropy", losses)
        self.assertIn("attn_s1_entropy", losses)
        self.assertIn("attn_entropy_nomask_avg", losses)
        self.assertTrue(torch.isfinite(losses["attn_entropy_avg"]).item())
        self.assertAlmostEqual(float(losses["attn_tau_mean_avg"].item()), 0.0, places=5)
        self.assertGreater(len(decoder.last_attn_metrics), 0)


if __name__ == "__main__":
    unittest.main()
