# Config Layout

## User-facing presets
- `presets/lite.yaml`: `AeroLite-Det-T` for quick smoke tests
- `presets/full.yaml`: `AeroLite-Det-S` as the main balanced training preset
- `presets/prod.yaml`: `AeroLite-Det-B` for the heavier production-oriented recipe

## Archived configs
- `archive/*`: historical notes only; active runs should use `presets/*`

## Policy
- Day-to-day runs should start from `presets/full.yaml` unless you explicitly want a lighter or more conservative recipe.
- Prefer CLI overrides from `scripts/pipeline.py` over creating many YAML variants.
- Active presets are pinned to `C:/Users/tsake/OneDrive/Desktop/datasets/Aero_dataset` as the canonical external dataset root.
- Active presets do not hard-code class counts; `train_net.py`, `test_net.py`, and `scripts/pipeline.py` auto-sync detector class counts from the dataset.
- Move one-off experiment configs to `config_files/archive/`.
