import unittest

import numpy as np
import torch

from alphaction.config import cfg as base_cfg
from alphaction.dataset.build import make_data_sampler


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


if __name__ == "__main__":
    unittest.main()
