import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from alphaction.config import auto_sync_dataset_class_counts, get_config


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((48, 48, 3), dtype=np.uint8)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def _write_label(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(line + "\n", encoding="utf-8")


def _write_data_yaml(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: dent",
                "  1: scratch",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _build_tiny_yolo_dataset(root: Path) -> Path:
    data = root / "tiny_yolo"
    _write_data_yaml(data / "data.yaml")
    for split in ("train", "val", "test"):
        _write_image(data / "images" / split / f"{split}_0.jpg")
        class_id = 0 if split != "test" else 1
        _write_label(
            data / "labels" / split / f"{split}_0.txt",
            f"{class_id} 0.5 0.5 0.25 0.25",
        )
    return data


class TestConfigDatasetAutosync(unittest.TestCase):
    def test_auto_sync_dataset_class_counts_from_dataset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = _build_tiny_yolo_dataset(Path(tmp_dir))
            config = get_config()
            config.DATA.PATH_TO_DATA_DIR = str(data_dir)
            config.DATA.FRAME_DIR = ""
            config.DATA.ANNOTATION_FORMAT = "auto"
            config.DATA.INPUT_TYPE = "image"
            config.DATA.NUM_FRAMES = 1
            config.DATA.SAMPLING_RATE = 1
            config.MODEL.STM.ACTION_CLASSES = 9
            config.MODEL.STM.OBJECT_CLASSES = 9
            config.MODEL.STM.NUM_ACT = 9
            config.MODEL.STM.NUM_CLS = 9

            metadata = auto_sync_dataset_class_counts(config)

            self.assertIsNotNone(metadata)
            self.assertEqual(metadata["num_classes"], 2)
            self.assertEqual(metadata["class_names"], ["dent", "scratch"])
            self.assertTrue(metadata["applied"])
            self.assertEqual(config.MODEL.STM.ACTION_CLASSES, 2)
            self.assertEqual(config.MODEL.STM.OBJECT_CLASSES, 2)
            self.assertEqual(config.MODEL.STM.NUM_ACT, 2)
            self.assertEqual(config.MODEL.STM.NUM_CLS, 2)

    def test_auto_sync_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = _build_tiny_yolo_dataset(Path(tmp_dir))
            config = get_config()
            config.DATA.PATH_TO_DATA_DIR = str(data_dir)
            config.DATA.AUTO_SYNC_CLASS_COUNTS = False
            config.MODEL.STM.ACTION_CLASSES = 7
            config.MODEL.STM.OBJECT_CLASSES = 7
            config.MODEL.STM.NUM_ACT = 7
            config.MODEL.STM.NUM_CLS = 7

            metadata = auto_sync_dataset_class_counts(config)

            self.assertIsNone(metadata)
            self.assertEqual(config.MODEL.STM.ACTION_CLASSES, 7)
            self.assertEqual(config.MODEL.STM.OBJECT_CLASSES, 7)
            self.assertEqual(config.MODEL.STM.NUM_ACT, 7)
            self.assertEqual(config.MODEL.STM.NUM_CLS, 7)


if __name__ == "__main__":
    unittest.main()
