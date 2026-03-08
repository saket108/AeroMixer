import unittest

import torch

from alphaction.config import cfg as default_cfg
from alphaction.modeling.detector.stm_detector import (
    AeroLiteDetector,
    DefectPrototypeMemory,
    ScaleTextRouter,
    TileGlobalContextFusion,
)
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

    def test_scale_text_router_changes_level_weights_for_different_text(self):
        torch.manual_seed(0)
        router = ScaleTextRouter(
            feature_dim=16,
            text_dim=8,
            num_levels=4,
            hidden_dim=12,
            gain=0.50,
        )
        features = [
            torch.randn(2, 16, 1, 32, 32),
            torch.randn(2, 16, 1, 16, 16),
            torch.randn(2, 16, 1, 8, 8),
            torch.randn(2, 16, 1, 4, 4),
        ]
        text_a = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        text_b = torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        routed_a, summary_a = router(features, text_a)
        routed_b, summary_b = router(features, text_b)

        self.assertEqual(len(routed_a), 4)
        self.assertEqual(tuple(summary_a["level_weights"].shape), (2, 4))
        self.assertEqual(tuple(summary_a["object_scale"].shape), (2,))
        self.assertFalse(
            torch.allclose(summary_a["level_weights"], summary_b["level_weights"])
        )
        self.assertFalse(torch.allclose(routed_a[0], routed_b[0]))

    def test_tile_global_context_fusion_routes_features_from_tile_metadata(self):
        torch.manual_seed(0)
        fusion = TileGlobalContextFusion(
            feature_dim=16,
            text_dim=8,
            num_levels=4,
            hidden_dim=12,
            blend=0.25,
        )
        features = [
            torch.randn(2, 16, 1, 32, 32),
            torch.randn(2, 16, 1, 16, 16),
            torch.randn(2, 16, 1, 8, 8),
            torch.randn(2, 16, 1, 4, 4),
        ]
        tile_meta_a = torch.tensor(
            [
                [0.25, 0.50, 0.50, 1.0, 0.50, 0.0, 0.0, 0.50, 0.0, 1.0, 0.0, 0.0],
                [0.75, 0.50, 0.50, 1.0, 0.50, 0.50, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        tile_meta_b = torch.tensor(
            [
                [0.50, 0.25, 1.0, 0.50, 0.50, 0.0, 0.0, 0.0, 0.50, 1.0, 0.0, 0.0],
                [0.50, 0.75, 1.0, 0.50, 0.50, 0.0, 0.50, 0.0, 0.0, 1.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        text_context = torch.randn(2, 8)

        routed_a, summary_a = fusion(features, tile_meta_a, text_context=text_context)
        routed_b, summary_b = fusion(features, tile_meta_b, text_context=text_context)

        self.assertEqual(len(routed_a), 4)
        self.assertEqual(tuple(summary_a["context"].shape), (2, 8))
        self.assertEqual(tuple(summary_a["level_weights"].shape), (2, 4))
        self.assertFalse(torch.allclose(summary_a["context"], summary_b["context"]))
        self.assertFalse(torch.allclose(routed_a[0], routed_b[0]))

    def test_prompt_adaptive_queries_change_routed_prefix_only(self):
        cfg = self._build_cfg()
        cfg.defrost()
        cfg.MODEL.STM.NUM_QUERIES = 10
        cfg.MODEL.STM.PROMPT_ADAPTIVE_QUERIES = True
        cfg.MODEL.STM.PROMPT_ADAPTIVE_QUERY_RATIO = 0.4
        cfg.MODEL.STM.PROMPT_ADAPTIVE_QUERY_SCALE = 0.35
        cfg.freeze()
        decoder = STMDecoder(cfg, image_mode=True)

        whwh = torch.tensor([[640.0, 640.0, 640.0, 640.0]], dtype=torch.float32)
        text_context = torch.linspace(
            0.0, 1.0, cfg.MODEL.LITE_TEXT.EMBED_DIM
        ).unsqueeze(0)
        routed_extras = {
            "scale_routing": {
                "level_weights": torch.tensor(
                    [[0.90, 0.06, 0.03, 0.01]], dtype=torch.float32
                ),
                "object_scale": torch.tensor([0.70], dtype=torch.float32),
            }
        }

        xyzr_base, query_base, _ = decoder._decode_init_queries(
            whwh, text_context=text_context, extras={}
        )
        xyzr_routed, query_routed, _ = decoder._decode_init_queries(
            whwh,
            text_context=text_context,
            extras=routed_extras,
        )

        adaptive_count = 4
        self.assertFalse(
            torch.allclose(
                xyzr_base[:, :adaptive_count], xyzr_routed[:, :adaptive_count]
            )
        )
        self.assertFalse(
            torch.allclose(
                query_base[:, :adaptive_count], query_routed[:, :adaptive_count]
            )
        )
        self.assertTrue(
            torch.allclose(
                xyzr_base[:, adaptive_count:], xyzr_routed[:, adaptive_count:]
            )
        )
        self.assertTrue(
            torch.allclose(
                query_base[:, adaptive_count:], query_routed[:, adaptive_count:]
            )
        )

    def test_class_partition_queries_use_reserved_slice_after_prompt_prefix(self):
        cfg = self._build_cfg()
        cfg.defrost()
        cfg.MODEL.STM.NUM_QUERIES = 10
        cfg.MODEL.STM.PROMPT_ADAPTIVE_QUERIES = True
        cfg.MODEL.STM.PROMPT_ADAPTIVE_QUERY_RATIO = 0.4
        cfg.MODEL.STM.PROMPT_ADAPTIVE_QUERY_SCALE = 0.35
        cfg.MODEL.STM.CLASS_PARTITION_QUERIES = True
        cfg.MODEL.STM.CLASS_PARTITION_QUERY_RATIO = 0.4
        cfg.MODEL.STM.CLASS_PARTITION_TOPK = 2
        cfg.MODEL.STM.CLASS_PARTITION_SCALE = 0.5
        cfg.freeze()
        decoder = STMDecoder(cfg, image_mode=True)

        whwh = torch.tensor([[640.0, 640.0, 640.0, 640.0]], dtype=torch.float32)
        text_context = torch.tensor(
            [[1.0, 0.2, 0.0] + [0.0] * (cfg.MODEL.LITE_TEXT.EMBED_DIM - 3)],
            dtype=torch.float32,
        )
        text_features = torch.zeros(3, cfg.MODEL.LITE_TEXT.EMBED_DIM)
        text_features[0, 0] = 1.0
        text_features[1, 1] = 1.0
        text_features[2, 2] = 1.0
        routed_extras = {
            "scale_routing": {
                "level_weights": torch.tensor(
                    [[0.90, 0.06, 0.03, 0.01]], dtype=torch.float32
                ),
                "object_scale": torch.tensor([0.70], dtype=torch.float32),
            }
        }

        original_flag = decoder.class_partition_queries
        decoder.class_partition_queries = False
        query_base = decoder._decode_init_queries(
            whwh,
            text_context=text_context,
            text_features=text_features,
            extras=routed_extras,
        )[1]
        decoder.class_partition_queries = original_flag

        query_partitioned = decoder._decode_init_queries(
            whwh,
            text_context=text_context,
            text_features=text_features,
            extras=routed_extras,
        )[1]

        self.assertTrue(torch.allclose(query_base[:, :4], query_partitioned[:, :4]))
        self.assertFalse(torch.allclose(query_base[:, 4:8], query_partitioned[:, 4:8]))
        self.assertTrue(torch.allclose(query_base[:, 8:], query_partitioned[:, 8:]))

    def test_defect_prototype_memory_updates_bank_and_fuses_text(self):
        memory = DefectPrototypeMemory(
            feature_dim=4,
            text_dim=4,
            num_classes=3,
            momentum=0.5,
            blend=0.4,
            context_blend=0.25,
        )
        memory.train()
        with torch.no_grad():
            memory.visual_proj.weight.copy_(torch.eye(4))

        feature_map = torch.zeros(2, 4, 4, 4)
        feature_map[0, :, :2, :2] = torch.tensor([4.0, 0.0, 0.0, 0.0]).view(4, 1, 1)
        feature_map[0, :, 2:, 2:] = torch.tensor([0.0, 4.0, 0.0, 0.0]).view(4, 1, 1)
        feature_map[1, :, :, :] = torch.tensor([0.0, 0.0, 4.0, 0.0]).view(4, 1, 1)

        boxes = [
            torch.tensor([[0.0, 0.0, 2.0, 2.0], [2.0, 2.0, 4.0, 4.0]]),
            torch.tensor([[0.0, 0.0, 4.0, 4.0]]),
        ]
        labels = [
            torch.tensor([0, 1], dtype=torch.int64),
            torch.tensor([2], dtype=torch.int64),
        ]
        whwh = torch.tensor(
            [[4.0, 4.0, 4.0, 4.0], [4.0, 4.0, 4.0, 4.0]], dtype=torch.float32
        )
        text_features = torch.eye(3, 4, dtype=torch.float32)

        fused_text, summary = memory(
            text_features=text_features,
            feature_map=feature_map,
            whwh=whwh,
            boxes=boxes,
            labels=labels,
        )

        self.assertTrue(memory.prototype_initialized.all())
        self.assertTrue(torch.all(memory.prototype_counts > 0))
        self.assertEqual(tuple(fused_text.shape), (3, 4))
        self.assertEqual(tuple(summary["image_context"].shape), (2, 4))
        self.assertFalse(torch.allclose(fused_text, text_features))

    def test_decoder_context_blends_prototype_image_context(self):
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
        vis_cls_feat = torch.randn(2, 512)
        base_context = decoder.get_image_text_context(
            text_features,
            labels=None,
            vis_cls_feat=vis_cls_feat,
            extras={},
        )
        proto_context = torch.tensor(
            [[0.0, 0.5, 0.5, 0.0], [0.5, 0.0, 0.5, 0.0]], dtype=torch.float32
        )
        fused_context = decoder.get_image_text_context(
            text_features,
            labels=None,
            vis_cls_feat=vis_cls_feat,
            extras={
                "prototype_memory": {
                    "image_context": proto_context,
                    "context_blend": 0.5,
                }
            },
        )

        self.assertFalse(torch.allclose(base_context, fused_context))

    def test_decoder_blends_tile_global_context_and_biases_queries(self):
        cfg = self._build_cfg()
        cfg.defrost()
        cfg.MODEL.STM.NUM_QUERIES = 8
        cfg.MODEL.STM.TILE_GLOBAL_CONTEXT = True
        cfg.MODEL.STM.TILE_GLOBAL_CONTEXT_BLEND = 0.4
        cfg.MODEL.STM.TILE_GLOBAL_CONTEXT_QUERY_SCALE = 0.3
        cfg.freeze()
        decoder = STMDecoder(cfg, image_mode=True)

        text_features = torch.zeros(3, cfg.MODEL.LITE_TEXT.EMBED_DIM)
        text_features[0, 0] = 1.0
        text_features[1, 1] = 1.0
        text_features[2, 2] = 1.0
        vis_cls_feat = torch.randn(1, 512)
        whwh = torch.tensor([[640.0, 640.0, 640.0, 640.0]], dtype=torch.float32)

        base_context = decoder.get_image_text_context(
            text_features,
            labels=None,
            vis_cls_feat=vis_cls_feat,
            extras={},
        )
        base_queries = decoder._decode_init_queries(
            whwh,
            text_context=base_context,
            text_features=text_features,
            extras={},
        )[1]

        tile_extras = {
            "tile_global_context": {
                "context": torch.tensor(
                    [
                        [0.0, 0.5, 0.5]
                        + [0.0] * (cfg.MODEL.LITE_TEXT.EMBED_DIM - 3)
                    ],
                    dtype=torch.float32,
                ),
                "blend": 0.4,
                "coverage": torch.tensor([0.25], dtype=torch.float32),
                "edge_proximity": torch.tensor([0.75], dtype=torch.float32),
            }
        }
        fused_context = decoder.get_image_text_context(
            text_features,
            labels=None,
            vis_cls_feat=vis_cls_feat,
            extras=tile_extras,
        )
        fused_queries = decoder._decode_init_queries(
            whwh,
            text_context=fused_context,
            text_features=text_features,
            extras=tile_extras,
        )[1]

        self.assertFalse(torch.allclose(base_context, fused_context))
        self.assertFalse(torch.allclose(base_queries, fused_queries))

    def test_cross_tile_consistency_loss_prefers_consistent_overlapping_tiles(self):
        cfg = self._build_cfg()
        cfg.defrost()
        cfg.MODEL.STM.CROSS_TILE_CONSISTENCY = True
        cfg.MODEL.STM.CROSS_TILE_CONSISTENCY_TOPK = 1
        cfg.MODEL.STM.CROSS_TILE_CONSISTENCY_MIN_SCORE = 0.0
        cfg.MODEL.STM.CROSS_TILE_CONSISTENCY_IOU = 0.1
        cfg.freeze()
        decoder = STMDecoder(cfg, image_mode=True)

        whwh = torch.tensor(
            [
                [100.0, 100.0, 100.0, 100.0],
                [100.0, 100.0, 100.0, 100.0],
            ],
            dtype=torch.float32,
        )
        extras = {
            "tile_meta": [
                {
                    "is_tiled": True,
                    "base_image_id": "panel",
                    "offset_norm": [0.0, 0.0],
                    "size_norm": [0.75, 1.0],
                },
                {
                    "is_tiled": True,
                    "base_image_id": "panel",
                    "offset_norm": [0.25, 0.0],
                    "size_norm": [0.75, 1.0],
                },
            ]
        }

        pred_logits = torch.full((2, 4, 4), -4.0, dtype=torch.float32)
        pred_logits[:, :, -1] = -2.0
        pred_logits[0, 0, 1] = 6.0
        pred_logits[1, 0, 1] = 6.0

        pred_boxes_consistent = torch.tensor(
            [
                [
                    [40.0, 20.0, 80.0, 60.0],
                    [5.0, 5.0, 20.0, 20.0],
                    [10.0, 10.0, 18.0, 18.0],
                    [20.0, 20.0, 30.0, 30.0],
                ],
                [
                    [6.6667, 20.0, 46.6667, 60.0],
                    [70.0, 5.0, 90.0, 20.0],
                    [75.0, 10.0, 85.0, 18.0],
                    [60.0, 20.0, 70.0, 30.0],
                ],
            ],
            dtype=torch.float32,
        )
        pred_boxes_inconsistent = pred_boxes_consistent.clone()
        pred_boxes_inconsistent[1, 0] = torch.tensor([20.0, 20.0, 50.0, 60.0])

        query_consistent = torch.zeros(2, 4, cfg.MODEL.STM.HIDDEN_DIM)
        query_consistent[0, 0, 0] = 1.0
        query_consistent[1, 0, 0] = 1.0
        query_inconsistent = query_consistent.clone()
        query_inconsistent[1, 0, 0] = 0.0
        query_inconsistent[1, 0, 1] = 1.0

        loss_consistent = decoder._compute_cross_tile_consistency_loss(
            pred_logits=pred_logits,
            pred_boxes=pred_boxes_consistent,
            query_feats=query_consistent,
            whwh=whwh,
            extras=extras,
        )["loss_tile_consistency"]
        loss_inconsistent = decoder._compute_cross_tile_consistency_loss(
            pred_logits=pred_logits,
            pred_boxes=pred_boxes_inconsistent,
            query_feats=query_inconsistent,
            whwh=whwh,
            extras=extras,
        )["loss_tile_consistency"]

        self.assertTrue(torch.isfinite(loss_consistent).item())
        self.assertTrue(torch.isfinite(loss_inconsistent).item())
        self.assertLess(float(loss_consistent), float(loss_inconsistent))


if __name__ == "__main__":
    unittest.main()
