import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import demo_image
from alphaction.config import cfg as global_cfg


class TestDemoImageHelpers(unittest.TestCase):
    def test_load_dataset_context_matches_split_image_and_descriptions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_dir = root / "train" / "images"
            label_dir = root / "train" / "labels"
            train_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)

            image_path = train_dir / "image_00001.jpg"
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(image_path), image))

            train_json = {
                "images": [
                    {
                        "image_id": "image_00001",
                        "file_name": "image_00001.jpg",
                        "split": "train",
                        "annotations": [
                            {
                                "annotation_id": "image_00001_0",
                                "category_id": 1,
                                "category_name": "dent",
                                "bounding_box_normalized": {
                                    "x_center": 0.5,
                                    "y_center": 0.5,
                                    "width": 0.25,
                                    "height": 0.25,
                                },
                                "description": "Low severity dent on the panel.",
                            }
                        ],
                    }
                ]
            }
            (root / "train.json").write_text(
                json.dumps(train_json, indent=2), encoding="utf-8"
            )
            (root / "data.yaml").write_text(
                "\n".join(
                    [
                        "train: ../train/images",
                        "val: ../valid/images",
                        "test: ../test/images",
                        "names:",
                        "  0: crack",
                        "  1: dent",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            cfg = global_cfg.clone()
            cfg.defrost()
            cfg.DATA.INPUT_TYPE = "image"
            cfg.DATA.PATH_TO_DATA_DIR = str(root)
            cfg.DATA.FRAME_DIR = ""
            cfg.DATA.NUM_FRAMES = 1
            cfg.DATA.SAMPLING_RATE = 1
            cfg.DATA.ANNOTATION_FORMAT = "custom_json"
            cfg.DATA.MULTIMODAL = True
            cfg.TEST.EVAL_OPEN = False
            cfg.freeze()

            context = demo_image._load_dataset_context(cfg, str(image_path))
            self.assertIsNotNone(context)
            self.assertEqual(context["split"], "train")
            self.assertEqual(context["image_rel"], "train/images/image_00001.jpg")
            self.assertEqual(context["class_names"], ["dent"])
            self.assertEqual(len(context["annotation_extras"]), 1)
            self.assertEqual(
                context["annotation_extras"][0]["description"],
                "Low severity dent on the panel.",
            )


if __name__ == "__main__":
    unittest.main()
