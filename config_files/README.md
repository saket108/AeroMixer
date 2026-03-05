# Config Layout

## Active configs (use these only)
- `base.yaml`: shared baseline reference
- `presets/lite.yaml`: fast sanity/Colab baseline
- `presets/full.yaml`: stronger training recipe
- `presets/prod.yaml`: stable detector-only production profile

## Policy
- Keep day-to-day runs on one of the three preset configs above.
- Prefer CLI overrides from `scripts/pipeline.py` over creating many new YAML variants.
- Treat extra YAML variants as experiment snapshots and move them to `config_files/archive/`.
