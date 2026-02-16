# AeroMixer

AeroMixer is a unified action/detection training pipeline for **images** and **videos**, with optional **open-vocabulary semantic labels** (text prompts per class).

## Key Capabilities

- Unified training and evaluation stack for image and video inputs.
- Generic dataset support (no hard dependency on AVA/JHMDB/UCF24 structure).
- Open-vocabulary mode with closed/open class vocabularies.
- CPU and GPU execution support for training and demo inference.

## Repository Structure

- `alphaction/` - core model, dataset loaders, training, inference, utilities.
- `config_files/images/aeromixer_images.yaml` - default image config.
- `config_files/videos/aeromixer_videos.yaml` - default video config.
- `preprocess/prepare_generic_video_dataset.py` - video-to-frames + split generation.
- `preprocess/build_open_vocab.py` - build closed/open semantic vocab files.
- `demo_image.py` / `demo.py` - single-image and video demo inference.

## Installation

1. Create and activate your Python environment.
2. Install PyTorch for your CUDA/CPU setup.
3. Install project dependencies:

```bash
pip install -r requirements.txt
```

## Configuration Entry Points

- Image pipeline: `config_files/images/aeromixer_images.yaml`
- Video pipeline: `config_files/videos/aeromixer_videos.yaml`

Core fields you typically update:

- `DATA.PATH_TO_DATA_DIR`
- `DATA.FRAME_DIR`
- `MODEL.WEIGHT` (for evaluation/demo)
- `SOLVER.MAX_EPOCH`
- `OUTPUT_DIR`

## Dataset Support

### Image Mode (`DATA.INPUT_TYPE: "image"`)

`ImageDataset` supports:

- Plain TXT: `<image_rel_path> <x1> <y1> <x2> <y2> <class_id>`
- YOLO-style folder layout (`images/...`, `labels/...`)
- COCO JSON
- Pascal VOC XML

### Video Mode (`DATA.INPUT_TYPE: "video"`)

Expected frame layout:

```text
<PATH_TO_DATA_DIR>/<FRAME_DIR>/<video_id>/<frame_files>
```

Split TXT format:

```text
<video_id> <frame_id> <x1> <y1> <x2> <y2> <class_id_or_name>
```

## Prepare a Generic Video Dataset

### A) Quick smoke-test data (placeholder boxes)

```bash
python preprocess/prepare_generic_video_dataset.py \
  --videos-dir data/raw_videos \
  --output-root data/Videos \
  --split all \
  --train-ratio 0.8 \
  --recursive
```

### B) Real annotations from JSON

Single JSON with split labels:

```bash
python preprocess/prepare_generic_video_dataset.py \
  --videos-dir data/raw_videos \
  --output-root data/Videos \
  --split all \
  --annotation-json data/annotations/all.json \
  --bbox-format auto \
  --recursive
```

Per-split JSON:

```bash
python preprocess/prepare_generic_video_dataset.py \
  --videos-dir data/raw_videos \
  --output-root data/Videos \
  --split all \
  --annotation-json data/annotations/{split}.json \
  --bbox-format auto \
  --recursive
```

Generated outputs:

- `data/Videos/frames/<video_id>/<frame_id>.jpg`
- `data/Videos/train.txt` and/or `data/Videos/test.txt`
- `data/Videos/annotations/video_id_map.json`

## Open-Vocabulary Semantic Labels

Build closed/open vocab files from your existing annotations:

```bash
python preprocess/build_open_vocab.py \
  --annotations data/Videos/train.txt data/Videos/test.txt \
  --out-closed data/Videos/annotations/vocab_closed.json \
  --out-open data/Videos/annotations/vocab_open.json \
  --closed-ratio 0.8 \
  --prompt-template "a person is {label}"
```

Enable in config:

```yaml
DATA:
  OPEN_VOCABULARY: True
  VOCAB_FILE: "annotations/vocab_closed.json"
  VOCAB_OPEN_FILE: "annotations/vocab_open.json"
TEST:
  EVAL_OPEN: True
```

- `VOCAB_FILE`: classes used for training (closed set)
- `VOCAB_OPEN_FILE`: classes used during open-set evaluation

## Train and Evaluate

### Image

```bash
python train_net.py --config-file config_files/images/aeromixer_images.yaml
python test_net.py --config-file config_files/images/aeromixer_images.yaml
```

### Video

```bash
python train_net.py --config-file config_files/videos/aeromixer_videos.yaml
python test_net.py --config-file config_files/videos/aeromixer_videos.yaml
```

### Optional helper script (Bash)

```bash
bash trainval.sh train images
bash trainval.sh eval images checkpoints/model_final.pth
bash trainval.sh train videos
bash trainval.sh eval videos checkpoints/model_final.pth
```

## Demo Inference

### Image

```bash
python demo_image.py \
  --config-file config_files/images/aeromixer_images.yaml \
  --image path/to/image.jpg \
  --output output/image_demo.png \
  --device cpu
```

### Video

```bash
python demo.py \
  --config-file config_files/videos/aeromixer_videos.yaml \
  --video path/to/input.mp4 \
  --output output/video_demo.mp4 \
  --device cpu
```

Use `--device cuda` when GPU is available.

## Troubleshooting

- Missing import warnings in IDE (for example `yacs`, `fvcore`, `einops`, `supervision`, `imageio`, `tensorboardX`) usually mean the current Python environment does not have all dependencies installed.
- If running on Windows without Bash, use direct `python ...` commands instead of `trainval.sh`.
- If open-vocabulary evaluation fails, verify class labels in your annotations match labels in vocab files.

## License

This project is released under the terms in `LICENSE`.
