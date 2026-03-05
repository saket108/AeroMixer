import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def _write_label(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(line + "\n", encoding="utf-8")


def _build_tiny_dataset(root: Path, invalid_train_box: bool = False) -> Path:
    data = root / "tiny_ds"
    for split in ["train", "val", "test"]:
        _write_image(data / "images" / split / f"{split}_a.jpg")
        label_line = "0 0.5 0.5 0.2 0.2"
        if split == "train" and invalid_train_box:
            label_line = "0 0.5 0.5 1.2 0.2"
        _write_label(data / "labels" / split / f"{split}_a.txt", label_line)
    return data


class TestPipelineSmoke(unittest.TestCase):
    def test_pipeline_dry_run_writes_manifest_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            data_dir = _build_tiny_dataset(tmp, invalid_train_box=False)
            out_dir = tmp / "run_ok"

            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "pipeline.py"),
                "--mode",
                "run",
                "--data",
                str(data_dir),
                "--preset",
                "lite",
                "--output-dir",
                str(out_dir),
                "--dry-run",
            ]
            result = subprocess.run(
                cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=180
            )
            self.assertEqual(
                result.returncode,
                0,
                msg=f"pipeline dry-run failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )

            manifest = out_dir / "pipeline_manifest.json"
            validation = out_dir / "dataset_validation.json"
            self.assertTrue(manifest.exists(), f"missing {manifest}")
            self.assertTrue(validation.exists(), f"missing {validation}")

            manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertIn("commands", manifest_data)
            self.assertIn("train", manifest_data["commands"])
            self.assertIn("eval", manifest_data["commands"])
            self.assertIn("dataset_fingerprint", manifest_data)
            self.assertTrue(manifest_data["validation"]["enabled"])
            self.assertTrue(manifest_data["validation"]["ok"])

    def test_pipeline_fails_on_invalid_dataset_by_default(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            data_dir = _build_tiny_dataset(tmp, invalid_train_box=True)
            out_dir = tmp / "run_bad"

            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "pipeline.py"),
                "--mode",
                "run",
                "--data",
                str(data_dir),
                "--preset",
                "lite",
                "--output-dir",
                str(out_dir),
                "--dry-run",
            ]
            result = subprocess.run(
                cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=180
            )
            self.assertEqual(
                result.returncode,
                2,
                msg=f"expected validation failure (rc=2)\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )

            validation = out_dir / "dataset_validation.json"
            self.assertTrue(validation.exists(), f"missing {validation}")
            validation_data = json.loads(validation.read_text(encoding="utf-8"))
            self.assertFalse(validation_data["ok"])
            self.assertGreater(len(validation_data["errors"]), 0)

    def test_pipeline_can_continue_with_allow_validation_errors(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            data_dir = _build_tiny_dataset(tmp, invalid_train_box=True)
            out_dir = tmp / "run_allow"

            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "pipeline.py"),
                "--mode",
                "run",
                "--data",
                str(data_dir),
                "--preset",
                "lite",
                "--output-dir",
                str(out_dir),
                "--dry-run",
                "--allow-validation-errors",
            ]
            result = subprocess.run(
                cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=180
            )
            self.assertEqual(
                result.returncode,
                0,
                msg=f"pipeline should continue with --allow-validation-errors\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )


if __name__ == "__main__":
    unittest.main()
