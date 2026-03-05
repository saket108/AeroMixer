# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added
- Stable inference entrypoint: `scripts/inference_pipeline.py`
- Dataset version freezing: `scripts/freeze_dataset_version.py`
- Pre-commit configuration: `.pre-commit-config.yaml`
- Benchmark tracking scaffold: `benchmarks/summary.csv`, `benchmarks/README.md`
- YOLO tiling utility for small-object workflows: `scripts/build_tiled_yolo_dataset.py`
- Pipeline tiling flags (`--tile-size`, `--tile-overlap`, `--tile-min-cover`, `--tile-splits`)
- Vendored OpenAI CLIP runtime module under `clip/` for install-independent imports.
- Minimal Colab dependency profile: `requirements_lock_colab_minimal.txt`.
- One-click Colab helper: `scripts/colab_oneclick_train.sh`.
- Colab quickstart guide: `docs/COLAB_QUICKSTART.md`.

### Changed
- `scripts/colab_bootstrap.sh` now defaults to `AEROMIXER_COLAB_PROFILE=minimal` and verifies vendored CLIP module (no git-wheel build path).
- `requirements*.txt` remove git-based CLIP dependency and use `ftfy` runtime dependency.


## [0.5.0] - 2026-03-06

### Added
- Tile-stitch full-image evaluation with global NMS and GT dedup controls.
- Canonical config layout (`base.yaml`, `presets/*`, `archive/*`) and config policy docs.
- Release discipline tooling (`scripts/release_tools.py`) and checklist.
- Deployment contract docs (`docs/INFERENCE_CONTRACT.md`) and Docker packaging files.
- CI hardening with tiny end-to-end pipeline smoke and new benchmark/tile tests.
- Unified baseline benchmark runner (`scripts/run_baseline_benchmarks.py`) for AeroMixer/YOLOv8/DETR with canonical row format.

### Changed
- `scripts/validate_dataset.py` now supports safe YOLO auto-fix mode (`--fix`) and post-fix revalidation.
- README benchmark and release sections now point to active baseline runner and release prepare/check/tag flow.

## [0.4.0] - 2026-03-05

### Added
- Dataset fail-fast quality gate: `scripts/validate_dataset.py`
- Pipeline integration for automatic validation and `dataset_validation.json`
- Production preset + guardrails in `scripts/pipeline.py`
- `config_files/images/aeromixer_images_prod.yaml`
- Reproducible lock files: `requirements_lock.txt`, `requirements_lock_colab.txt`
- New pipeline smoke test: `tests/test_pipeline_smoke.py`

### Changed
- CI now uses lockfile dependency installation path and enforces lint/format gates.
- Colab bootstrap prefers lockfile install.

## [0.3.0] - 2026-03-05

### Added
- One-command any-dataset training launcher: `scripts/train_any_dataset.py`
- One-command pipeline orchestrator: `scripts/pipeline.py`
- Colab bootstrap script: `scripts/colab_bootstrap.sh`
- Script/config organization docs and archived research utilities structure.

## [0.2.0] - 2026-03-04

### Fixed
- Label/box alignment bug in preprocessing path (Hungarian matcher shape mismatch).
- Checkpoint retention parser crash with `model_final.pth`.

## [0.1.0] - 2026-03-03

### Added
- Condensed startup logging mode.
- Lite image config and initial AeroMixer image-only training workflow hardening.
