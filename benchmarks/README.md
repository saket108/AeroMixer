# Benchmarks

This folder tracks reproducible benchmark outcomes across model presets and baselines.

## Files
- `summary.csv`: canonical benchmark table (append-only)

## Policy
- Record one row per completed run.
- Keep dataset/split fixed when comparing model changes.
- Always include `git_commit` and key run settings.
- For cross-model comparisons, use:
  - `python scripts/run_baseline_benchmarks.py ...`
  which appends AeroMixer / YOLOv8 / DETR rows into the same canonical format.

## Suggested columns
See `summary.csv` header for required fields.
