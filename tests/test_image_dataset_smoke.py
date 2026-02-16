import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from alphaction.config import cfg as global_cfg
from alphaction.dataset.datasets.image_dataset import ImageDataset


class TestImageDatasetSmoke(unittest.TestCase):
    def test_load_single_image_txt_annotation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_dir = root / "images"
            image_dir.mkdir(parents=True, exist_ok=True)

            image = np.zeros((64, 64, 3), dtype=np.uint8)
            image_path = image_dir / "sample.jpg"
            ok = cv2.imwrite(str(image_path), image)
            self.assertTrue(ok)

            (root / "train.txt").write_text("sample.jpg 5 6 40 50 0\n", encoding="utf-8")

            cfg = global_cfg.clone()
            cfg.defrost()
            cfg.DATA.INPUT_TYPE = "image"
            cfg.DATA.PATH_TO_DATA_DIR = str(root)
            cfg.DATA.FRAME_DIR = "images"
            cfg.DATA.NUM_FRAMES = 1
            cfg.DATA.SAMPLING_RATE = 1
            cfg.DATA.ANNOTATION_FORMAT = "txt"
            cfg.DATA.OPEN_VOCABULARY = False
            cfg.MODEL.BACKBONE.PATHWAYS = 1
            cfg.TEST.EVAL_OPEN = False
            cfg.freeze()

            dataset = ImageDataset(cfg, split="train")
            self.assertEqual(len(dataset), 1)

            primary_input, secondary_input, whwh, boxes, labels, extras, index = dataset[0]
            self.assertIsNone(secondary_input)
            self.assertEqual(primary_input.ndim, 4)
            self.assertEqual(tuple(whwh.shape), (4,))
            self.assertEqual(index, 0)
            self.assertEqual(boxes.shape[1], 4)
            self.assertEqual(labels.shape[0], 1)
            self.assertIn("image_rel", extras)


if __name__ == "__main__":
    unittest.main()
