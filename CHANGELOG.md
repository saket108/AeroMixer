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
