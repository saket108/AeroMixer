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
- Do not commit machine-specific absolute dataset paths.
- Move one-off experiment configs to `config_files/archive/`.
