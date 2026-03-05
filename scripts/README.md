# Scripts Layout

## Active entrypoints (use these)
- `pipeline.py`: one-command professional pipeline (`prepare -> train -> eval`)
  - Presets: `lite`, `full`, `prod`
  - `prod` enables detector-only guardrails by default
  - Auto-appends eval metrics to `benchmarks/summary.csv`
  - Supports tiled runs with stitched full-image evaluation (`--tile-stitch-eval`)
  - Supports integrated threshold sweep output (`--tune-thresholds`)
- `train_any_dataset.py`: dataset-aware training launcher
- `validate_dataset.py`: fail-fast dataset validation (`bad labels`, `split leakage`, `class stats`)
- `inference_pipeline.py`: stable inference/eval entrypoint with JSON summary output
- `freeze_dataset_version.py`: dataset fingerprint + validation manifest for reproducibility
- `build_tiled_yolo_dataset.py`: tile YOLO datasets for small-object runs
- `release_tools.py`: release readiness checks and annotated tag helper
- `colab_bootstrap.sh`: Colab dependency bootstrap

## Archived research utilities
- `archive/run_iof_tau_ablation.py`
- `archive/run_baseline_benchmarks.py`

Top-level wrappers remain for backward compatibility and forward to the archived scripts.
