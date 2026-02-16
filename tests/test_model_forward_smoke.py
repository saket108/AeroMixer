import unittest

import torch

from alphaction.modeling.detector.stm_detector import LayerNorm


class TestModelForwardSmoke(unittest.TestCase):
    def test_layernorm_forward(self):
        module = LayerNorm(8)
        x = torch.randn(2, 8, 2, 8, 8)
        y = module(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertTrue(torch.isfinite(y).all().item())


if __name__ == "__main__":
    unittest.main()
