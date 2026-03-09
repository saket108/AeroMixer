import tempfile
import unittest
from pathlib import Path
import json

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

            (root / "train.txt").write_text(
                "sample.jpg 5 6 40 50 0\n", encoding="utf-8"
            )

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

            primary_input, secondary_input, whwh, boxes, labels, extras, index = (
                dataset[0]
            )
            self.assertIsNone(secondary_input)
            self.assertEqual(primary_input.ndim, 4)
            self.assertEqual(tuple(whwh.shape), (4,))
            self.assertEqual(index, 0)
            self.assertEqual(boxes.shape[1], 4)
            self.assertEqual(labels.shape[0], 1)
            self.assertIn("image_rel", extras)

    def test_load_single_image_txt_annotation_with_severity(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_dir = root / "images"
            image_dir.mkdir(parents=True, exist_ok=True)

            image = np.zeros((64, 64, 3), dtype=np.uint8)
            image_path = image_dir / "sample.jpg"
            ok = cv2.imwrite(str(image_path), image)
            self.assertTrue(ok)

            (root / "train.txt").write_text(
                "sample.jpg 5 6 40 50 0 0.75\n", encoding="utf-8"
            )

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
            _, _, _, _, _, extras, _ = dataset[0]
            self.assertIn("severity", extras)
            self.assertAlmostEqual(float(extras["severity"][0]), 0.75, places=6)

    def test_load_custom_nested_json_with_severity_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_dir = root / "images"
            image_dir.mkdir(parents=True, exist_ok=True)

            image = np.zeros((64, 64, 3), dtype=np.uint8)
            image_path = image_dir / "sample.jpg"
            ok = cv2.imwrite(str(image_path), image)
            self.assertTrue(ok)

            custom_data = {
                "images": [
                    {
                        "image_id": "img_0001",
                        "file_name": "sample.jpg",
                        "split": "train",
                        "annotations": [
                            {
                                "annotation_id": "img_0001_1",
                                "category_id": 1,
                                "category_name": "dent",
                                "bounding_box_normalized": {
                                    "x_center": 0.5,
                                    "y_center": 0.5,
                                    "width": 0.5,
                                    "height": 0.5,
                                },
                                "damage_metrics": {
                                    "raw_severity_score": 0.33,
                                },
                                "risk_assessment": {
                                    "severity_level": "low",
                                    "requires_manual_validation": True,
                                },
                                "description": "Sample defect description",
                            }
                        ],
                    }
                ]
            }
            (root / "train.json").write_text(
                json.dumps(custom_data, indent=2), encoding="utf-8"
            )

            cfg = global_cfg.clone()
            cfg.defrost()
            cfg.DATA.INPUT_TYPE = "image"
            cfg.DATA.PATH_TO_DATA_DIR = str(root)
            cfg.DATA.FRAME_DIR = "images"
            cfg.DATA.NUM_FRAMES = 1
            cfg.DATA.SAMPLING_RATE = 1
            cfg.DATA.ANNOTATION_FORMAT = "custom_json"
            cfg.DATA.OPEN_VOCABULARY = False
            cfg.MODEL.BACKBONE.PATHWAYS = 1
            cfg.TEST.EVAL_OPEN = False
            cfg.freeze()

            dataset = ImageDataset(cfg, split="train")
            self.assertEqual(len(dataset), 1)

            _, _, _, boxes, labels, extras, _ = dataset[0]
            self.assertEqual(boxes.shape[1], 4)
            self.assertEqual(labels.shape[0], 1)
            self.assertIn("severity", extras)
            self.assertAlmostEqual(float(extras["severity"][0]), 0.33, places=6)
            self.assertIn("annotation_extras", extras)
            self.assertEqual(len(extras["annotation_extras"]), 1)
            self.assertEqual(
                extras["annotation_extras"][0]["description"],
                "Sample defect description",
            )
            self.assertEqual(extras["annotation_extras"][0]["category_name"], "dent")

    def test_load_custom_json_split_layout_with_basename_file_names(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for split_name, file_name in [
                ("train", "image_00001.jpg"),
                ("valid", "image_00002.jpg"),
                ("test", "image_00003.jpg"),
            ]:
                image_dir = root / split_name / "images"
                label_dir = root / split_name / "labels"
                image_dir.mkdir(parents=True, exist_ok=True)
                label_dir.mkdir(parents=True, exist_ok=True)
                image = np.zeros((64, 64, 3), dtype=np.uint8)
                self.assertTrue(cv2.imwrite(str(image_dir / file_name), image))

            train_json = {
                "images": [
                    {
                        "image_id": "image_00001",
                        "file_name": "image_00001.jpg",
                        "split": "train",
                        "annotations": [
                            {
                                "annotation_id": "image_00001_0",
                                "category_id": 0,
                                "category_name": "crack",
                                "bounding_box_normalized": {
                                    "x_center": 0.5,
                                    "y_center": 0.5,
                                    "width": 0.25,
                                    "height": 0.25,
                                },
                            }
                        ],
                    }
                ]
            }
            (root / "train.json").write_text(
                json.dumps(train_json, indent=2), encoding="utf-8"
            )

            cfg = global_cfg.clone()
            cfg.defrost()
            cfg.DATA.INPUT_TYPE = "image"
            cfg.DATA.PATH_TO_DATA_DIR = str(root)
            cfg.DATA.FRAME_DIR = ""
            cfg.DATA.NUM_FRAMES = 1
            cfg.DATA.SAMPLING_RATE = 1
            cfg.DATA.ANNOTATION_FORMAT = "custom_json"
            cfg.DATA.OPEN_VOCABULARY = False
            cfg.MODEL.BACKBONE.PATHWAYS = 1
            cfg.TEST.EVAL_OPEN = False
            cfg.freeze()

            dataset = ImageDataset(cfg, split="train")
            self.assertEqual(len(dataset), 1)

            _, _, _, boxes, labels, extras, _ = dataset[0]
            self.assertEqual(boxes.shape[0], 1)
            self.assertEqual(labels.shape[0], 1)
            self.assertEqual(extras["image_rel"], "train/images/image_00001.jpg")

    def test_custom_json_val_and_test_splits_stay_separate(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for split_name, file_name in [
                ("valid", "image_valid.jpg"),
                ("test", "image_test.jpg"),
            ]:
                image_dir = root / split_name / "images"
                label_dir = root / split_name / "labels"
                image_dir.mkdir(parents=True, exist_ok=True)
                label_dir.mkdir(parents=True, exist_ok=True)
                image = np.zeros((64, 64, 3), dtype=np.uint8)
                self.assertTrue(cv2.imwrite(str(image_dir / file_name), image))

            valid_json = {
                "images": [
                    {
                        "image_id": "image_valid",
                        "file_name": "image_valid.jpg",
                        "split": "valid",
                        "annotations": [
                            {
                                "annotation_id": "image_valid_0",
                                "category_id": 0,
                                "category_name": "crack",
                                "bounding_box_normalized": {
                                    "x_center": 0.5,
                                    "y_center": 0.5,
                                    "width": 0.25,
                                    "height": 0.25,
                                },
                            }
                        ],
                    }
                ]
            }
            test_json = {
                "images": [
                    {
                        "image_id": "image_test",
                        "file_name": "image_test.jpg",
                        "split": "test",
                        "annotations": [
                            {
                                "annotation_id": "image_test_0",
                                "category_id": 1,
                                "category_name": "dent",
                                "bounding_box_normalized": {
                                    "x_center": 0.5,
                                    "y_center": 0.5,
                                    "width": 0.25,
                                    "height": 0.25,
                                },
                            }
                        ],
                    }
                ]
            }
            (root / "valid.json").write_text(
                json.dumps(valid_json, indent=2), encoding="utf-8"
            )
            (root / "test.json").write_text(
                json.dumps(test_json, indent=2), encoding="utf-8"
            )

            cfg = global_cfg.clone()
            cfg.defrost()
            cfg.DATA.INPUT_TYPE = "image"
            cfg.DATA.PATH_TO_DATA_DIR = str(root)
            cfg.DATA.FRAME_DIR = ""
            cfg.DATA.NUM_FRAMES = 1
            cfg.DATA.SAMPLING_RATE = 1
            cfg.DATA.ANNOTATION_FORMAT = "custom_json"
            cfg.DATA.OPEN_VOCABULARY = False
            cfg.MODEL.BACKBONE.PATHWAYS = 1
            cfg.TEST.EVAL_OPEN = False
            cfg.freeze()

            valid_dataset = ImageDataset(cfg, split="val")
            test_dataset = ImageDataset(cfg, split="test")

            self.assertEqual(valid_dataset.samples[0]["image_rel"], "valid/images/image_valid.jpg")
            self.assertEqual(test_dataset.samples[0]["image_rel"], "test/images/image_test.jpg")
            self.assertEqual(valid_dataset.class_names[0], "crack")
            self.assertEqual(test_dataset.class_names[0], "dent")

    def test_infers_tile_metadata_from_tiled_yolo_filenames(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_dir = root / "images" / "train"
            label_dir = root / "labels" / "train"
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)

            left = np.zeros((32, 32, 3), dtype=np.uint8)
            right = np.zeros((32, 32, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(image_dir / "panel__x0_y0.jpg"), left))
            self.assertTrue(cv2.imwrite(str(image_dir / "panel__x32_y0.jpg"), right))
            (label_dir / "panel__x0_y0.txt").write_text(
                "0 0.5 0.5 0.25 0.25\n", encoding="utf-8"
            )
            (label_dir / "panel__x32_y0.txt").write_text(
                "0 0.5 0.5 0.25 0.25\n", encoding="utf-8"
            )

            cfg = global_cfg.clone()
            cfg.defrost()
            cfg.DATA.INPUT_TYPE = "image"
            cfg.DATA.PATH_TO_DATA_DIR = str(root)
            cfg.DATA.FRAME_DIR = "images"
            cfg.DATA.NUM_FRAMES = 1
            cfg.DATA.SAMPLING_RATE = 1
            cfg.DATA.ANNOTATION_FORMAT = "yolo"
            cfg.DATA.OPEN_VOCABULARY = False
            cfg.MODEL.BACKBONE.PATHWAYS = 1
            cfg.TEST.EVAL_OPEN = False
            cfg.freeze()

            dataset = ImageDataset(cfg, split="train")
            self.assertEqual(len(dataset), 2)

            _, _, _, _, _, extras_left, _ = dataset[0]
            tile_meta = extras_left["tile_meta"]
            self.assertTrue(tile_meta["is_tiled"])
            self.assertTrue(str(tile_meta["base_image_id"]).endswith("panel"))
            self.assertEqual(tile_meta["full_size_wh"], [64, 32])
            self.assertAlmostEqual(tile_meta["position_norm"][0], 0.25, places=5)
            self.assertAlmostEqual(tile_meta["position_norm"][1], 0.50, places=5)
            self.assertAlmostEqual(tile_meta["coverage_ratio"], 0.50, places=5)


if __name__ == "__main__":
    unittest.main()
