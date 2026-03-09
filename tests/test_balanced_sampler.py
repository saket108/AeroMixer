import unittest

import numpy as np
import torch

from alphaction.config import cfg as base_cfg
from alphaction.dataset.build import (
    _compute_tile_group_ids,
    make_batch_data_sampler,
    make_data_sampler,
)


class _DummyDataset:
    def __init__(self):
        self.num_classes = 2
        self.samples = []
        for _ in range(100):
            self.samples.append({"labels": np.array([0], dtype=np.int64)})
        for _ in range(5):
            self.samples.append({"labels": np.array([1], dtype=np.int64)})

    def __len__(self):
        return len(self.samples)


class _DummyTiledDataset:
    def __init__(self):
        self.samples = [
            {
                "labels": np.array([0], dtype=np.int64),
                "tile_meta": {"is_tiled": True, "base_image_id": "img_a"},
            },
            {
                "labels": np.array([0], dtype=np.int64),
                "tile_meta": {"is_tiled": True, "base_image_id": "img_a"},
            },
            {
                "labels": np.array([1], dtype=np.int64),
                "tile_meta": {"is_tiled": True, "base_image_id": "img_b"},
            },
            {
                "labels": np.array([1], dtype=np.int64),
                "tile_meta": {"is_tiled": True, "base_image_id": "img_b"},
            },
            {
                "labels": np.array([0], dtype=np.int64),
                "tile_meta": None,
            },
        ]

    def __len__(self):
        return len(self.samples)


class TestBalancedSampler(unittest.TestCase):
    def test_weighted_sampler_prioritizes_rare_class(self):
        cfg = base_cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.BALANCED_SAMPLING = True
        cfg.DATALOADER.BALANCED_SAMPLING_POWER = 1.0
        cfg.freeze()

        ds = _DummyDataset()
        sampler = make_data_sampler(
            ds,
            shuffle=True,
            distributed=False,
            cfg=cfg,
            is_train=True,
        )
        self.assertIsInstance(sampler, torch.utils.data.WeightedRandomSampler)
        weights = sampler.weights
        self.assertGreater(float(weights[-1]), float(weights[0]))

    def test_balanced_sampler_off_uses_random_sampler(self):
        cfg = base_cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.BALANCED_SAMPLING = False
        cfg.freeze()

        ds = _DummyDataset()
        sampler = make_data_sampler(
            ds,
            shuffle=True,
            distributed=False,
            cfg=cfg,
            is_train=True,
        )
        self.assertIsInstance(sampler, torch.utils.data.RandomSampler)

    def test_compute_tile_group_ids_groups_sibling_tiles(self):
        ds = _DummyTiledDataset()
        group_ids, tiled_samples = _compute_tile_group_ids(ds)
        self.assertEqual(tiled_samples, 4)
        self.assertIsNotNone(group_ids)
        self.assertEqual(group_ids[0], group_ids[1])
        self.assertEqual(group_ids[2], group_ids[3])
        self.assertNotEqual(group_ids[1], group_ids[2])
        self.assertNotEqual(group_ids[3], group_ids[4])

    def test_tile_group_batching_keeps_sibling_tiles_together(self):
        cfg = base_cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.TILE_GROUP_BATCHING = True
        cfg.freeze()

        ds = _DummyTiledDataset()
        sampler = torch.utils.data.SequentialSampler(ds)
        batch_sampler = make_batch_data_sampler(
            ds,
            sampler,
            samples_per_gpu=2,
            num_iters=None,
            start_iter=0,
            drop_last=True,
            cfg=cfg,
        )

        batches = list(batch_sampler)
        self.assertGreaterEqual(len(batches), 2)
        batch_groups = []
        for batch in batches:
            group = {
                ds.samples[idx]["tile_meta"]["base_image_id"]
                for idx in batch
                if isinstance(ds.samples[idx].get("tile_meta"), dict)
            }
            batch_groups.append(group)
        self.assertIn({"img_a"}, batch_groups)
        self.assertIn({"img_b"}, batch_groups)


if __name__ == "__main__":
    unittest.main()
