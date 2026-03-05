# AeroMixer

AeroMixer is currently an image-first detection training stack built around an STM-style query decoder.

This README documents what the code does today.

## Current Status

- The active data path is image-only (`alphaction/dataset/build.py` always builds `ImageDataset`).
- Training and evaluation are end-to-end: dataset -> model -> losses -> optimizer/scheduler -> checkpoint -> inference -> mAP.
- Open-vocabulary plumbing exists (vocab loading, text encoder integration hooks), but the active STM path is detection-only (class + box losses).
- Primary runtime entrypoints are image-only (`train_net.py`, `test_net.py`, `demo_image.py`, `trainval.sh`).

## Model Overview

Default detector (`MODEL.DET: STMDetector`) is:

1. Backbone (`alphaction/modeling/backbone/backbone.py`)
2. STM decoder head (`alphaction/modeling/stm_decoder/stm_decoder.py`)
3. DETR-style Hungarian assignment + set losses

### High-level Forward (Image Mode)

1. Input image batch is preprocessed and padded to `[B, C, H, W]`.
2. Backbone outputs patch features (and optionally class/global features).
3. `STMDetector` projects image features into `HIDDEN_DIM` using a lightweight conv projection block.
4. Feature levels are built as:
   - real `C3/C4/C5 -> FPN (P3/P4/P5/P6)` when backbone exposes intermediate maps
   - resized single-map pyramid fallback for backbones without intermediate maps
5. STM decoder runs multi-stage query refinement:
   - query attention + IoF bias
   - adaptive sampling/mixing
   - class logits + box delta prediction
6. Training returns weighted losses; eval returns normalized boxes + scores + labels.

## Architecture Details

### 1) Backbone Options

Registered backbones include:

- `ImageResNet-50`, `ImageResNet-101`, `ImageResNet-152`
- `ImageResNet-Lite`
- `ImageViT-B`, `ImageViT-L`, `ImageViT-H`
- CLIP-family wrappers: `ViT-B/16`, `ViT-B/32`, `ViT-L/14`
- CLIP-ViP: `ViP-B/16`, `ViP-B/32`
- ViCLIP: `ViCLIP-L/14`

The provided image config (`config_files/images/aeromixer_images.yaml`) uses `ViT-B/16` (better spatial detail than `ViT-B/32` for fine defects).

### 2) Detector (`STMDetector`)

Core behavior in image mode:

- `is_image` when `DATA.INPUT_TYPE == "image"` (or `DATA.IMAGE_MODE`).
- Backbone output is normalized into a feature tensor.
- If temporal dim exists (`[B,C,T,H,W]`), it is averaged over `T`.
- `img_proj`:
  - `Conv2d(in_dim -> hidden_dim, 1x1)`
  - `ReLU`
  - `Conv2d(hidden_dim -> hidden_dim, 3x3, padding=1)`
- Feature pyramid passed to decoder as 4 levels:
  - preferred path (ResNet backbones): `C3/C4/C5 -> lateral 1x1 -> top-down fusion -> 3x3 output convs -> P3/P4/P5 + stride-2 P6`
  - fallback path (other backbones): bilinear resized maps at `1x`, `1/2x`, `1/4x`, `1/8x` with per-level adapters
  - each level is reshaped to `[B, C, 1, H, W]` for decoder compatibility
- Optional text features resolved from:
  - `extras["text_features"]`, or
  - backbone `forward_text()` when `DATA.OPEN_VOCABULARY=True`

### 3) STM Decoder (`STMDecoder`)

Default STM hyperparameters (from `alphaction/config/defaults.py`):

- `NUM_QUERIES=100`
- `HIDDEN_DIM=256`
- `NUM_STAGES=6`
- `NUM_HEADS=8`
- `SPATIAL_POINTS=32`
- `TEMPORAL_POINTS=4` (temporal branch is disabled in image mode)

Decoder internals:

- Query init:
  - learned spatial query embedding (`num_queries x hidden_dim`)
  - temporal query embedding only for non-image branches
- Box init:
  - uses `extras["prior_boxes"]` if provided
  - else CAM-based init if `extras["cams"]` exists
  - else learnable anchor-like priors on a query grid (default)
  - optional full-image fallback via `MODEL.STM.QUERY_INIT_MODE: "full_image"`
  - optional small-object bias via `MODEL.STM.QUERY_INIT_SMALL_OBJECT_BIAS`
- Stage update loop:
  - self-attention with IoF-based attention bias
  - adaptive sampling/mixing (via `SAMPLE4D` + `AdaptiveMixing`)
  - classification head (`OBJECT_CLASSES + 1` with background)
  - regression head (`4`-dim delta in `xyzr`)
  - iterative box refinement (`refine_xyzr`)
  - optional attention telemetry for stage-level quality analysis

### 4) Active Heads and Outputs

- Active supervised heads:
  - object classification logits
  - box regression
- Optional supervised head:
  - severity regression (`MODEL.STM.PREDICT_SEVERITY=True`)
- Action/open-vocab classification in STM stage is not the active supervised path in current image flow.

Train-time output:

- loss dict with keys:
  - `loss_ce`
  - `loss_bbox`
  - `loss_giou`
  - plus stage-wise auxiliary losses when `MODEL.STM.INTERMEDIATE_SUPERVISION=True`:
    - `loss_ce_0 ... loss_ce_{N-2}`
    - `loss_bbox_0 ... loss_bbox_{N-2}`
    - `loss_giou_0 ... loss_giou_{N-2}`
  - when `MODEL.STM.ATTN_TELEMETRY=True`, extra logged metrics are also returned (not used for optimization), e.g.:
    - `attn_entropy_avg`, `attn_diag_avg`, `attn_top1_avg`
    - `attn_tau_mean_avg`, `attn_bias_mean_avg`
    - `refine_l1_avg` and `refine_l1_s*` (stage-to-stage box delta magnitude)
    - optional no-mask comparisons when `MODEL.STM.ATTN_TELEMETRY_COMPARE_NOMASK=True`
    - stagewise keys like `attn_s0_entropy` when `MODEL.STM.ATTN_TELEMETRY_STAGEWISE=True`

Eval-time output per image:

- dict:
  - `boxes`: normalized `xyxy` in `[0, 1]`
  - `scores`: foreground score
  - `labels`: predicted class index
  - `severity`: optional per-box severity score in `[0, 1]` when severity head is enabled

## Loss and Matching

Matching is DETR-style Hungarian assignment:

- cost = `cost_class * CE_term + cost_bbox * L1 + cost_giou * GIoU`
- defaults:
  - `OBJECT_WEIGHT=2.0`
  - `L1_WEIGHT=2.0`
  - `GIOU_WEIGHT=2.0`
  - `BACKGROUND_WEIGHT=0.1`

Criterion:

- Classification: cross-entropy with background class.
- Box loss: L1 + GIoU.
- Stage-wise auxiliary supervision is enabled through `aux_outputs` (all decoder stages except the last).
- Final scalar loss = weighted sum of all active keys in `weight_dict` (final stage + auxiliary stages).

## Data Pipeline Details

### 1) Supported Annotation Formats (`ImageDataset`)

- TXT: `<image_rel> <x1> <y1> <x2> <y2> <class_id> [severity]`
- YOLO folder layout (`images/...`, `labels/...`)
- COCO JSON
- Pascal VOC XML
- Custom nested JSON (`images[].annotations[]`) with normalized boxes:
  - `bounding_box_normalized.{x_center,y_center,width,height}`
  - optional `damage_metrics.raw_severity_score`
  - optional passthrough fields in metadata (`risk_assessment`, `damage_metrics`, `description`)

Selection is via:

- `DATA.ANNOTATION_FORMAT` in `{auto, txt, yolo, coco, voc, custom_json}`

### 2) Preprocessing

`PreprocessWithBoxes` does:

- resize using train/test scale config
- optional horizontal flip
- channel conversion (BGR/RGB by config)
- mean/std normalization
- outputs tensor as `C x T x H x W` (`T=1` in image mode)

### 3) Batch Contract

Collated batch from `BatchCollator`:

- `primary_inputs`: padded tensor `[B, C, H, W]` (or `[B, C, T, H, W]` if needed)
- `secondary_inputs`: `None` in current image path
- `whwh`: tensor `[B, 4]`
- `boxes`: list of tensors `[Ni, 4]`
- `labels`: list of tensors
- `metadata`: list of dicts
- `indices`: list of sample ids

## Training Loop Details

`train_net.py` + `alphaction/engine/trainer.py`:

- builds model, optimizer, LR scheduler, checkpoint manager
- optional checkpoint load/transfer
- forward -> summed losses -> backward
- gradient clipping: max norm `5.0`
- optimizer step every iteration
- scheduler step every 10 iterations
- periodic checkpoint save + final checkpoint
- optional validation/test inference pass

Checkpoints are stored under:

- `<OUTPUT_DIR>/checkpoints/*.pth`

## Inference and Evaluation

`test_net.py` + `alphaction/engine/inference.py`:

- runs model in eval mode
- normalizes outputs to a unified `(boxes, scores_matrix)` form
- calls dataset evaluation (`image_ap` / `frame_ap` alias path)
- Pascal-style frame mAP backend (`IOU_THRESH` from config)

Output artifacts:

- inference logs/results under `<OUTPUT_DIR>/inference/...`

## Open-Vocabulary and Text Path

Available hooks:

- vocab loading from `IMAGES.VOCAB_FILE` / `IMAGES.VOCAB_OPEN_FILE`
- text prompt materialization in `ImageDataset`
- text encoder vocabulary injection via:
  - `model.backbone.text_encoder.set_vocabulary(...)`
- optional text feature passing through `extras`

Utility script:

```bash
python preprocess/build_open_vocab.py \
  --annotations data/aircraft/train.txt data/aircraft/test.txt \
  --out-closed data/aircraft/annotations/vocab_closed.json \
  --out-open data/aircraft/annotations/vocab_open.json \
  --closed-ratio 0.8 \
  --prompt-template "a photo of {label}"
```

## Running the Project

Install:

```bash
pip install -r requirements.txt
```

Reproducible install (recommended):

```bash
pip install -r requirements_lock.txt
```

Colab quickstart (2-command setup):

```bash
git clone https://github.com/saket108/AeroMixer.git /content/AeroMixer
cd /content/AeroMixer && bash scripts/colab_bootstrap.sh
```

Then run full pipeline (prepare + train + eval) from uploaded dataset:

```bash
python scripts/pipeline.py \
  --mode run \
  --data /content/model_dataset_zipped.zip \
  --preset prod \
  --output-dir output/colab_model_dataset \
  --epochs 3 \
  --batch-size 4 \
  --num-workers 2 \
  --split-ratio 80,10,10 \
  --skip-val-in-train
```

Professional local workflow (single command):

```bash
python scripts/pipeline.py \
  --mode run \
  --data "C:/path/to/dataset_or_zip" \
  --preset prod \
  --output-dir output/my_run \
  --epochs 30 \
  --batch-size 4 \
  --num-workers 2
```

The pipeline writes a run manifest:
- `<OUTPUT_DIR>/pipeline_manifest.json`
- `<OUTPUT_DIR>/dataset_validation.json` (automatic dataset quality report)

Preset guide:
- `lite`: fastest sanity/debug loop
- `full`: research-heavy default
- `prod`: detector-only guarded profile (open-vocab/severity/telemetry disabled by default)

Validation behavior:
- `scripts/pipeline.py` validates dataset integrity before training by default.
- If validation fails, pipeline stops early.
- Override only when needed: `--allow-validation-errors` or `--skip-validation`.

Direct low-level train command (advanced/debug):

```bash
python train_net.py --config-file config_files/images/aeromixer_images.yaml
```

Dataset-aware train-only command (without eval):

```bash
python scripts/train_any_dataset.py \
  --data /content/my_dataset.zip \
  --config-file config_files/images/aeromixer_images_lite.yaml \
  --output-dir output/colab_any \
  --epochs 3 \
  --batch-size 4 \
  --num-workers 2 \
  --split-ratio 80,10,10 \
  --skip-val-in-train
```

Notes:
- `--data` supports zip, folder, `data.yaml`, or `.json`.
- Flat YOLO folders (`images/` + `labels/`) need `--split-ratio`.
- Script auto-detects annotation format and sets class-count overrides for STM.

Project script policy:
- Active: `scripts/pipeline.py`, `scripts/train_any_dataset.py`, `scripts/colab_bootstrap.sh`
- Archived research tools: `scripts/archive/*` (top-level wrappers kept for compatibility)

Eval:

```bash
python test_net.py --config-file config_files/images/aeromixer_images.yaml
```

Image demo:

```bash
python demo_image.py \
  --config-file config_files/images/aeromixer_images.yaml \
  --image path/to/image.jpg \
  --output output/image_demo.png \
  --device cpu
```

## Configuration Cheat Sheet

Most important fields to tune:

- `DATA.PATH_TO_DATA_DIR`
- `DATA.FRAME_DIR`
- `DATA.ANNOTATION_FORMAT`
- `DATA.OPEN_VOCABULARY`
- `MODEL.BACKBONE.CONV_BODY`
- `MODEL.STM.OBJECT_CLASSES`
- `MODEL.STM.NUM_QUERIES`
- `MODEL.STM.NUM_STAGES`
- `MODEL.STM.QUERY_INIT_MODE`
- `MODEL.STM.QUERY_INIT_BASE_SCALE`
- `MODEL.STM.QUERY_INIT_SMALL_OBJECT_BIAS`
- `MODEL.STM.ATTN_TELEMETRY`
- `MODEL.STM.ATTN_TELEMETRY_STAGEWISE`
- `MODEL.STM.ATTN_TELEMETRY_COMPARE_NOMASK`
- `MODEL.STM.IOF_TAU_MODE`
- `MODEL.STM.IOF_TAU_FIXED`
- `MODEL.STM.IOF_TAU_CLAMP_MIN`
- `MODEL.STM.IOF_TAU_CLAMP_MAX`
- `MODEL.STM.PREDICT_SEVERITY`
- `MODEL.STM.SEVERITY_WEIGHT`
- `SOLVER.BASE_LR`
- `SOLVER.MAX_EPOCH`
- `SOLVER.IMAGES_PER_BATCH`
- `TEST.IOU_THRESH`
- `OUTPUT_DIR`

## Known Limitations

- Active training/eval data path is image-only.
- `demo_image.py` is a utility path and may need adaptation depending on detector output shape changes.

## Quality Checks

```bash
python -m compileall alphaction preprocess demo_image.py train_net.py test_net.py
python -m unittest discover -s tests -v
```

## Attention Ablation Runner (Archived Research Tool)

Run baseline/zero/clamp IoF tau experiments on a fixed 5% subset:

```bash
python scripts/archive/run_iof_tau_ablation.py \
  --config-file config_files/images/aeromixer_images.yaml \
  --subset-ratio 0.05 \
  --epochs 20 \
  --seed 2 \
  --clamp-values 0.5,1.0,2.0
```

Notes:
- Source annotations can be `txt` split files or YOLO folder layout (`images/` + `labels/`).
- For ablation consistency, the script materializes a subset as `train.txt/test.txt` and runs with `DATA.ANNOTATION_FORMAT=txt`.

The script writes `ablation_summary.csv` under `outputs/iof_tau_ablation/` with:

- `attn_entropy_avg`, `attn_diag_avg`, `attn_tau_mean_avg`, `refine_l1_avg`
- `mAP@0.5`
- `SmallObject/AP@0.5`

## Baseline Benchmark Runner (Archived Research Tool)

Run AeroMixer / YOLO / DETR commands and aggregate one benchmark CSV:

```bash
python scripts/archive/run_baseline_benchmarks.py \
  --output-root outputs/baseline_benchmarks \
  --tag merged_dataset_v1 \
  --aeromixer-cmd "python train_net.py --config-file config_files/images/aeromixer_images.yaml --skip-final-test" \
  --aeromixer-metrics "output/aircraft_run/inference/train_metrics_final.json" \
  --yolo-cmd "python path/to/yolo_train.py ..." \
  --yolo-metrics "runs/detect/train/results.csv" \
  --detr-cmd "python path/to/detr_train.py ..." \
  --detr-metrics "outputs/detr/metrics.json"
```

Output:
- `outputs/baseline_benchmarks/benchmark_summary.csv`
- `outputs/baseline_benchmarks/benchmark_summary.json`

## License

See `LICENSE`.
