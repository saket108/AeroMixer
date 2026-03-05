# Scripts Layout

## Active entrypoints (use these)
- `pipeline.py`: one-command professional pipeline (`prepare -> train -> eval`)
  - Presets: `lite`, `full`, `prod`
  - `prod` enables detector-only guardrails by default
  - Auto-appends eval metrics to `benchmarks/summary.csv`
  - Supports AP50 and optional AP50:95 reporting (`TEST.REPORT_AP5095`)
  - Supports tiled runs with stitched full-image evaluation (`--tile-stitch-eval`)
  - Supports integrated threshold sweep output (`--tune-thresholds`)
- `train_any_dataset.py`: dataset-aware training launcher
- `validate_dataset.py`: fail-fast dataset validation (`bad labels`, `split leakage`, `class stats`)
- `inference_pipeline.py`: stable inference/eval entrypoint with JSON summary output
- `freeze_dataset_version.py`: dataset fingerprint + validation manifest for reproducibility
- `build_tiled_yolo_dataset.py`: tile YOLO datasets for small-object runs
- `release_tools.py`: release readiness checks and annotated tag helper
- `run_baseline_benchmarks.py`: unified baseline runner (AeroMixer/YOLOv8/DETR) with one comparable CSV format
- `validate_docker_inference.py`: container build/run contract validator with JSON report
- `colab_bootstrap.sh`: Colab dependency bootstrap
- `colab_oneclick_train.sh`: one-command Colab bootstrap + pipeline run

## Archived research utilities
- `archive/run_iof_tau_ablation.py`
- `archive/run_baseline_benchmarks.py`

Some top-level wrappers remain for backward compatibility (e.g. IOF tau ablation).
