import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestCliHelpSmoke(unittest.TestCase):
    def _run_help(self, relative_script):
        script = ROOT / relative_script
        self.assertTrue(script.exists(), f"Missing script: {script}")

        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"{relative_script} --help failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )

    def test_demo_image_help(self):
        self._run_help("demo_image.py")

    def test_demo_video_help(self):
        self._run_help("demo.py")

    def test_train_help(self):
        self._run_help("train_net.py")

    def test_test_help(self):
        self._run_help("test_net.py")

    def test_prepare_generic_video_dataset_help(self):
        self._run_help("preprocess/prepare_generic_video_dataset.py")

    def test_build_open_vocab_help(self):
        self._run_help("preprocess/build_open_vocab.py")


if __name__ == "__main__":
    unittest.main()
