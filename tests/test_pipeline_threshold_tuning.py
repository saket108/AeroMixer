import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def _load_pipeline_module():
    script_path = ROOT / "scripts" / "pipeline.py"
    spec = importlib.util.spec_from_file_location("pipeline_under_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load pipeline module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


PIPELINE = _load_pipeline_module()


class TestPipelineThresholdTuning(unittest.TestCase):
    def test_threshold_sweeps_reuse_base_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            output_dir = tmp / "run"
            plan = PIPELINE.ds.DatasetPlan(
                data_dir=tmp / "dataset",
                annotation_format="yolo",
                frame_dir="",
                num_classes=5,
            )
            issued_cmds = []

            def _fake_run(cmd, cwd, dry_run):
                issued_cmds.append(cmd)
                return 0

            fake_metrics = {
                "images": {
                    "map50": 0.12,
                    "map5095": 0.03,
                    "small_ap": 0.01,
                    "metrics": {"PascalBoxes_Precision/mAP@0.5IOU": 0.12},
                }
            }

            with mock.patch.object(PIPELINE, "_run_command", side_effect=_fake_run):
                with mock.patch.object(
                    PIPELINE, "_extract_eval_metrics", return_value=fake_metrics
                ):
                    summary = PIPELINE._run_threshold_tuning(
                        root=ROOT,
                        python_exe=sys.executable,
                        config_file="config_files/presets/full.yaml",
                        output_dir=str(output_dir),
                        plan=plan,
                        preset="full",
                        num_workers=2,
                        disable_guardrails=False,
                        model_weight=None,
                        tile_stitch_eval=True,
                        tile_stitch_nms_iou=0.5,
                        tile_stitch_gt_dedup_iou=0.9,
                        base_extra_opts=[],
                        thresholds=[0.05],
                        dry_run=False,
                    )

            self.assertEqual(len(issued_cmds), 1)
            cmd = issued_cmds[0]
            weight_idx = cmd.index("MODEL.WEIGHT") + 1
            self.assertEqual(
                cmd[weight_idx],
                str(output_dir / "checkpoints" / "model_final.pth"),
            )
            self.assertEqual(summary["best"]["score_threshold"], 0.05)
            self.assertAlmostEqual(summary["best"]["map50_avg"], 0.12, places=6)


if __name__ == "__main__":
    unittest.main()
