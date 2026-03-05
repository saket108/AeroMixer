# Config Layout

## Recommended configs
- `images/aeromixer_images_lite.yaml`: fast sanity/Colab baseline
- `images/aeromixer_images.yaml`: stronger training recipe

## Policy
- Keep day-to-day runs on one of the two configs above.
- Prefer CLI overrides from `scripts/pipeline.py` over creating many new YAML variants.
- Treat extra YAML variants as experiment snapshots and archive them when no longer active.
