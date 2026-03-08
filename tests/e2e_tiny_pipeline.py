#!/usr/bin/env python3
"""Tiny end-to-end pipeline run for CI.

Runs a real (non-dry-run) prepare -> train -> eval flow on a synthetic dataset.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((96, 96, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (70, 70), (255, 255, 255), thickness=-1)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def _write_label(path: Path, cls_id: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # centered box
    path.write_text(f"{cls_id} 0.5 0.5 0.5 0.5\n", encoding="utf-8")


def _build_dataset(root: Path) -> Path:
    data = root / "tiny_e2e_ds"
    counts = {"train": 4, "val": 2, "test": 2}
    for split, n in counts.items():
        for i in range(n):
            stem = f"{split}_{i:03d}"
            _write_image(data / split / "images" / f"{stem}.jpg")
            _write_label(data / split / "labels" / f"{stem}.txt", cls_id=(i % 2))

    data_yaml = data / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {data.as_posix()}",
                "train: train/images",
                "val: val/images",
                "test: test/images",
                "nc: 2",
                "names:",
                "  0: class0",
                "  1: class1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        data_dir = _build_dataset(tmp)
        out_dir = tmp / "run"

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
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--skip-val-in-train",
            "--extra-opts",
            "TEST.IMAGES_PER_BATCH",
            "2",
            "DATA.TRAIN_MIN_SCALES",
            "[128]",
            "DATA.TEST_MIN_SCALES",
            "[128]",
            "SOLVER.EVAL_AFTER",
            "999",
            "SOLVER.EVAL_PERIOD",
            "999",
        ]

        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr)
            return int(proc.returncode)

        manifest = out_dir / "pipeline_manifest.json"
        if not manifest.exists():
            print(f"Missing manifest: {manifest}")
            return 2
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if "commands" not in payload:
            print("Manifest missing commands.")
            return 2

        ckpt = out_dir / "checkpoints" / "model_final.pth"
        if not ckpt.exists():
            print(f"Missing final checkpoint: {ckpt}")
            return 2

        logs = sorted(out_dir.glob("inference/*/result_image.log"))
        if not logs:
            print(f"Missing inference log under: {out_dir / 'inference'}")
            return 2

        # copy artifacts for debugging if needed
        ci_artifacts = ROOT / "output" / "ci_tiny_e2e_last"
        if ci_artifacts.exists():
            shutil.rmtree(ci_artifacts, ignore_errors=True)
        ci_artifacts.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(out_dir, ci_artifacts)
        print(f"Tiny E2E pipeline passed. Artifacts: {ci_artifacts}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
