# Scripts Layout

## Active entrypoints (use these)
- `pipeline.py`: one-command professional pipeline (`prepare -> train -> eval`)
  - Presets: `lite`, `full`, `prod`
  - `prod` enables detector-only guardrails by default
- `train_any_dataset.py`: dataset-aware training launcher
- `colab_bootstrap.sh`: Colab dependency bootstrap

## Archived research utilities
- `archive/run_iof_tau_ablation.py`
- `archive/run_baseline_benchmarks.py`

Top-level wrappers remain for backward compatibility and forward to the archived scripts.
