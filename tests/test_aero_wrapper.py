import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "aero.py"
SPEC = importlib.util.spec_from_file_location("aero_wrapper", MODULE_PATH)
aero_wrapper = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(aero_wrapper)


class TestAeroWrapper(unittest.TestCase):
    def test_train_command_skips_final_test(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = Path(tmp_dir)
            args = SimpleNamespace(
                data=str(dataset),
                preset="prod",
                output_dir="output/aero_train",
                epochs=30,
                batch_size=2,
                num_workers=0,
                annotation_format="custom_json",
                tile_size=0,
                tile_overlap=0.25,
                tile_min_cover=0.35,
                tune_thresholds=False,
                skip_val_in_train=False,
                resume=False,
                extra_opts=[],
            )
            cmd = aero_wrapper._build_pipeline_command(args, mode="train")
            self.assertIn("--skip-final-test", cmd)
            self.assertIn("--epochs", cmd)
            self.assertNotIn("--resume", cmd)
            self.assertNotIn("--tile-size", cmd)
            self.assertIn("DATA.ANNOTATION_FORMAT", cmd)
            self.assertIn("custom_json", cmd)

    def test_train_command_only_resumes_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = Path(tmp_dir)
            args = SimpleNamespace(
                data=str(dataset),
                preset="prod",
                output_dir="output/aero_train",
                epochs=30,
                batch_size=2,
                num_workers=0,
                annotation_format="custom_json",
                tile_size=0,
                tile_overlap=0.25,
                tile_min_cover=0.35,
                tune_thresholds=False,
                skip_val_in_train=False,
                resume=True,
                extra_opts=[],
            )
            cmd = aero_wrapper._build_pipeline_command(args, mode="train")
            self.assertIn("--resume", cmd)

    def test_eval_command_does_not_include_epochs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = Path(tmp_dir)
            args = SimpleNamespace(
                data=str(dataset),
                preset="prod",
                output_dir="output/aero_train",
                batch_size=2,
                num_workers=0,
                annotation_format="custom_json",
                tile_size=0,
                tile_overlap=0.25,
                tile_min_cover=0.35,
                tune_thresholds=True,
                threshold_grid="0.05,0.1,0.2,0.3",
                resume=False,
                extra_opts=[],
            )
            cmd = aero_wrapper._build_pipeline_command(args, mode="eval")
            self.assertIn("--mode", cmd)
            self.assertIn("eval", cmd)
            self.assertNotIn("--epochs", cmd)
            self.assertNotIn("--skip-final-test", cmd)
            self.assertNotIn("--tile-size", cmd)
            self.assertIn("DATA.ANNOTATION_FORMAT", cmd)
            self.assertIn("custom_json", cmd)


if __name__ == "__main__":
    unittest.main()
