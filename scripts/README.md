# Scripts Layout

Most users should use only one wrapper:
- `aero.py`
  - `python scripts/aero.py smoke`
  - `python scripts/aero.py train`
  - `python scripts/aero.py vis --image ...`

## Public entrypoints (use these)
- `aero.py`: simplified wrapper around the active workflow
  - `smoke`: 1-epoch health check
  - `train`: main AeroMixer training command
  - `vis`: one-image prediction with JSON descriptions
- `pipeline.py`: one-command professional pipeline (`prepare -> train -> eval`)
  - Presets: `lite`, `full`, `prod`
  - `prod` enables detector-only guardrails by default
  - Auto-appends eval metrics to `benchmarks/summary.csv`
  - Supports AP50 and optional AP50:95 reporting (`TEST.REPORT_AP5095`)
  - Supports tiled runs with stitched full-image evaluation (`--tile-stitch-eval`)
  - Supports integrated threshold sweep output (`--tune-thresholds`)
- `validate_dataset.py`: fail-fast dataset validation (`bad labels`, `split leakage`, `class stats`)
- `freeze_dataset_version.py`: dataset fingerprint + validation manifest for reproducibility
- `release_tools.py`: release readiness checks and annotated tag helper
- `run_baseline_benchmarks.py`: unified baseline runner (AeroMixer/YOLOv8/DETR) with one comparable CSV format
- `validate_docker_inference.py`: container build/run contract validator with JSON report
- `colab_bootstrap.sh`: Colab dependency bootstrap

## Internal helpers
- `internal/train_any_dataset.py`: dataset preparation and low-level train launcher used by `pipeline.py`
- `internal/build_tiled_yolo_dataset.py`: tiling helper used by `pipeline.py`

## Archived research utilities
- `archive/run_iof_tau_ablation.py`
- `archive/run_baseline_benchmarks.py`
