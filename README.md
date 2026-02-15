# AeroMixer

AeroMixer supports both image and generic video detection pipelines.

## What Is Active

- Image pipeline via `ImageDataset`.
- Generic video pipeline via `VideoDataset` (no AVA/JHMDB/UCF24 required).
- Same detector/trainer stack for both modes.

## Setup

Use your current Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Configs

- Image: `config_files/images/aeromixer_images.yaml`
- Video: `config_files/videos/aeromixer_videos.yaml`

## Run

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

Or use script:

```bash
bash trainval.sh train images
bash trainval.sh eval images checkpoints/model_final.pth
bash trainval.sh train videos
bash trainval.sh eval videos checkpoints/model_final.pth
```

## Generic Video Dataset Format

Set in config:

- `DATA.PATH_TO_DATA_DIR`: dataset root
- `DATA.FRAME_DIR`: frame root under dataset root
- `DATA.DATASETS: ['videos']`
- `DATA.INPUT_TYPE: 'video'`

## Video Preprocess Script

Use the generic script that replaces benchmark-specific preprocess scripts:

Quick test split (placeholders):

```bash
python preprocess/prepare_generic_video_dataset.py \
  --videos-dir data/raw_videos \
  --output-root data/Videos \
  --split test \
  --recursive
```

Generate both `train.txt` and `test.txt` in one command:

```bash
python preprocess/prepare_generic_video_dataset.py \
  --videos-dir data/raw_videos \
  --output-root data/Videos \
  --split all \
  --train-ratio 0.8 \
  --recursive
```

With real labels JSON (single file with `split` field):

```bash
python preprocess/prepare_generic_video_dataset.py \
  --videos-dir data/raw_videos \
  --output-root data/Videos \
  --split all \
  --annotation-json data/annotations/all.json \
  --bbox-format auto \
  --recursive
```

With per-split JSON files:

```bash
python preprocess/prepare_generic_video_dataset.py \
  --videos-dir data/raw_videos \
  --output-root data/Videos \
  --split all \
  --annotation-json data/annotations/{split}.json \
  --bbox-format auto \
  --recursive
```

Outputs:
- `data/Videos/frames/<video_id>/<frame_id>.jpg`
- `data/Videos/train.txt` and/or `data/Videos/test.txt`
- `data/Videos/annotations/video_id_map.json`


Frame layout:

```text
<PATH_TO_DATA_DIR>/<FRAME_DIR>/<video_id>/<frame_files>
```

Annotations (choose one):

1) TXT (`train.txt`, `test.txt` or `annotations/train.txt`, `annotations/test.txt`)

```text
<video_id> <frame_id> <x1> <y1> <x2> <y2> <class_id_or_name>
```

2) JSON (`<split>.json` or `annotations/<split>.json`)

Each record needs:
- `video` / `video_id` / `vid`
- `frame` / `frame_id`
- `bbox` / `box` / `xyxy` (xyxy)
- `label` / `class_id` / `category_id` / `class`

## Notes

- Legacy benchmark-specific dataset adapters were removed; use generic `images`/`videos` modes.
- For video, preprocess your raw videos into frame folders first.

## License

See `LICENSE`.
