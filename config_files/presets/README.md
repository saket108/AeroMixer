# Active Presets

These are the only preset files meant for normal train/eval use:
- `lite.yaml` -> `AeroLite-Det-T`
- `full.yaml` -> `AeroLite-Det-S`
- `prod.yaml` -> `AeroLite-Det-B`

`scripts/pipeline.py` defaults to `full`.

Canonical dataset root for the active presets:
- `C:/Users/tsake/OneDrive/Desktop/datasets/Aero_dataset`

Class counts are auto-synced from the dataset at runtime, so these presets are scale recipes, not dataset-specific class configs.
