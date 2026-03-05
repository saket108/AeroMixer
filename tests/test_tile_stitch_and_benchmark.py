import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from alphaction.dataset.datasets.evaluation.images.image_eval import _stitch_tiled_predictions


class TestTileStitchAndBenchmark(unittest.TestCase):
    def test_tile_stitch_remap_and_global_nms(self):
        # Two overlapping tiles covering the same object on a larger image.
        targets = {
            "img1__x0_y0.jpg": {
                "bbox": np.array([[0.50, 0.20, 0.90, 0.60]], dtype=np.float32),
                "labels": [1],
                "resolution": (100, 100),
            },
            "img1__x40_y0.jpg": {
                "bbox": np.array([[0.10, 0.20, 0.50, 0.60]], dtype=np.float32),
                "labels": [1],
                "resolution": (100, 100),
            },
        }
        results = {
            "img1__x0_y0.jpg": {
                "boxes": np.array([[0.50, 0.20, 0.90, 0.60]], dtype=np.float32),
                "scores": np.array([0.90], dtype=np.float32),
                "action_ids": np.array([1], dtype=np.int64),
            },
            "img1__x40_y0.jpg": {
                "boxes": np.array([[0.10, 0.20, 0.50, 0.60]], dtype=np.float32),
                "scores": np.array([0.80], dtype=np.float32),
                "action_ids": np.array([1], dtype=np.int64),
            },
        }

        stitched_results, stitched_targets, tile_keys, stitched_images = _stitch_tiled_predictions(
            results,
            targets,
            nms_iou=0.5,
            gt_dedup_iou=0.8,
        )

        self.assertEqual(tile_keys, 2)
        self.assertEqual(stitched_images, 1)
        self.assertIn("img1", stitched_results)
        self.assertIn("img1", stitched_targets)
        # Duplicate tile predictions for the same object should collapse to one.
        self.assertEqual(stitched_results["img1"]["boxes"].shape[0], 1)
        # Duplicate GT from overlapping tiles should dedupe to one.
        self.assertEqual(stitched_targets["img1"]["bbox"].shape[0], 1)

    def test_pipeline_benchmark_csv_append_helpers(self):
        # Import pipeline as module from scripts/ for helper-level unit test.
        import sys

        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "scripts"))
        import pipeline as pl  # type: ignore

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            out_dir = tmp / "run"
            log_dir = out_dir / "inference" / "aircraft"
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "result_image.log").write_text(
                "{ 'PascalBoxes_Precision/mAP@0.5IOU': np.float64(0.1234), "
                "'SmallObject/AP@0.5IOU': 0.0456 }",
                encoding="utf-8",
            )

            manifest = {
                "created_at_utc": "2026-01-01T00:00:00+00:00",
                "git_commit": "deadbeef",
                "preset": "lite",
                "epochs": 1,
                "batch_size": 2,
                "dataset_plan": {"data_dir": str(tmp / "ds")},
                "commands": {"eval": "python test_net.py ..."},
            }

            eval_metrics = pl._extract_eval_metrics(out_dir)
            self.assertIn("aircraft", eval_metrics)
            self.assertAlmostEqual(float(eval_metrics["aircraft"]["map50"]), 0.1234, places=6)

            rows = pl._append_benchmark_rows(tmp, manifest, eval_metrics)
            self.assertEqual(len(rows), 1)

            csv_path = tmp / "benchmarks" / "summary.csv"
            self.assertTrue(csv_path.exists())
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                data_rows = list(csv.DictReader(f))
            self.assertEqual(len(data_rows), 1)
            self.assertEqual(data_rows[0]["git_commit"], "deadbeef")
            self.assertEqual(data_rows[0]["dataset"], "aircraft")


if __name__ == "__main__":
    unittest.main()

