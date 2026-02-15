# AeroMixer

AeroMixer is our adapted branch of OpenMixer focused on practical training and evaluation for both **video** and **image** detection pipelines.

## What We Built

- Unified support for **video mode** and **image mode** in one codebase.
- Added an image-first dataset/evaluation path for detection workflows.
- Kept video path compatibility (JHMDB/UCF24 configs and scripts).
- Improved CPU/GPU portability across training, inference, and evaluation.
- Added flexible text/vocab integration hooks for multimodal experiments.

## Current Capabilities

- Train/evaluate on video datasets (original OpenMixer flow).
- Train/evaluate on image detection datasets.
- Run inference on a single image with `demo_image.py`.
- Run on either CUDA or CPU (CPU is slower).

## Project Layout

- `alphaction/` - core model, dataset, engine, utils.
- `config_files/` - training/eval configs.
- `config_files/images/aeromixer_images.yaml` - image mode config.
- `config_files/jhmdb/`, `config_files/ucf24/` - video mode configs.
- `third_party/` - helper scripts (evaluation, video IO, maskrcnn helpers).

## Setup

Use Python 3.13 (your current environment) or a compatible version.

```bash
pip install -r requirements.txt
pip install tensorboardX supervision imageio
```

## Training / Eval

### Image mode

```bash
python train_net.py --config-file config_files/images/aeromixer_images.yaml
python test_net.py --config-file config_files/images/aeromixer_images.yaml
```

### Video mode (bash script)

```bash
bash trainval.sh train jhmdb
bash trainval.sh eval jhmdb checkpoints/model_final.pth
```

(Use `ucf24` instead of `jhmdb` when needed.)

## Dataset Notes

### Video mode

- Uses dataset structures expected by JHMDB/UCF24 preprocess/eval tools.

### Image mode

- Designed for detection datasets.
- Supports practical annotation setups used in this repo flow (including YOLO-style and JSON-based metadata usage through dataset adapters).
- Optional text metadata can be connected for multimodal runs.

## Troubleshooting

If VS Code/Pylance shows unresolved imports:

1. Select the correct interpreter (Python used for this repo).
2. Reinstall requirements:
   - `pip install -r requirements.txt`
3. For common warnings, ensure these are installed:
   - `tensorboardX`, `supervision`, `imageio`

If folder rename/move fails on Windows (`EPERM`/"folder in use"):

- Close VS Code windows using the repo.
- Pause/exit OneDrive sync temporarily.
- Retry rename from parent directory.

## Acknowledgements

Built on top of OpenMixer/STMixer and related open-source dependencies used in this repository.

## License

See `LICENSE`.
