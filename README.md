# AeroMixer

AeroMixer is an image-only detection stack built around a lightweight query detector with optional image+text class prompting.

This README documents the active production path, not the older video research history.

## Current Status

- The active data path is image-only (`alphaction/dataset/build.py` always builds `ImageDataset`).
- The supported public model family is `AeroLite-Det-T`, `AeroLite-Det-S`, and `AeroLite-Det-B`.
- The active runtime supports one detector entrypoint: `AeroLiteDetector` (legacy alias: `STMDetector`).
- Text prompts are active in the main configs through lightweight class-prototype fusion, not only as offline metadata.
- Primary runtime entrypoints are `scripts/pipeline.py`, `scripts/validate_dataset.py`, `train_net.py`, `test_net.py`, and `demo_image.py`.
- Detector class counts are auto-synced from the dataset path at runtime; the active presets are not locked to a fixed class list.

## Simplest Workflow

If you want the least confusing path, use only `scripts/aero.py`:

- Health check:
  - `python scripts/aero.py smoke`
- Main training run:
  - `python scripts/aero.py train`
- Final test evaluation:
  - `python scripts/aero.py eval`
- Visualize one image:
  - `python scripts/aero.py vis --image "C:\path\to\image.jpg"`

For the local Windows setup, this wrapper defaults to:
- dataset: `C:/Users/tsake/OneDrive/Desktop/datasets/Aero_dataset`
- smoke output: `output/aero_smoke`
- train output: `output/aero_train`

## Model Overview

Supported detector (`MODEL.DET: AeroLiteDetector`) is:

1. Backbone (`alphaction/modeling/backbone/backbone.py`)
2. STM decoder head (`alphaction/modeling/stm_decoder/stm_decoder.py`)
3. DETR-style Hungarian assignment + set losses

In practical terms, `AeroLite-Det` is a lightweight ResNet-like CNN backbone with an FPN-style pyramid, a prompt-conditioned scale router, a defect prototype memory, class-partitioned query allocation, tile-global context fusion for tiled defect runs, a cross-tile consistency loss for overlapping tiles, a query-based decoder head, and a native LiteText prompt encoder for image+text detection.

### High-level Forward (Image Mode)

1. Input image batch is preprocessed and padded to `[B, C, H, W]`.
2. Backbone outputs patch features (and optionally class/global features).
3. `AeroLiteDetector` projects image features into `HIDDEN_DIM` using a lightweight conv projection block.
4. Feature levels are built as:
   - real `C3/C4/C5 -> FPN (P3/P4/P5/P6)` when backbone exposes intermediate maps
   - resized single-map pyramid fallback for backbones without intermediate maps
   - optional `DefectPrototypeMemory` pools boxed regions into an EMA class bank and fuses them back into the prompt space
   - optional `ScaleTextRouter` reweights the pyramid using prompt context before decoding
   - optional `TileGlobalContextFusion` infers tile position/full-image coverage context from tiled sample names and fuses that summary back into the multiscale path
5. STM decoder runs multi-stage query refinement:
   - query attention + IoF bias
   - adaptive sampling/mixing
   - class-partitioned query slots reserve part of the query bank for the most likely defect prompts
   - prompt-adaptive query priors for the routed query subset
   - tile-global query bias blends surface-position context into text/query initialization for tiled inference
   - class logits + box delta prediction
   - optional cross-tile consistency loss aligns overlapping sibling-tile predictions in full-image normalized space during training
6. Training returns weighted losses; eval returns normalized boxes + scores + labels.

## Architecture Details

### 1) Backbone Options

Supported backbones in the active runtime:

- `AeroLite-Det-T`
- `AeroLite-Det-S`
- `AeroLite-Det-B`

The default presets map directly to that family:

- `lite`: `AeroLite-Det-T`
- `full`: `AeroLite-Det-S`
- `prod`: `AeroLite-Det-B`

This matches the way modern detector repos present one architecture family with a few scale variants instead of a flat list of unrelated backbones. Older research backbones are not part of the supported runtime surface.

### 2) Detector (`AeroLiteDetector`)

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
  - backbone `forward_text()` when `DATA.MULTIMODAL=True` or `DATA.OPEN_VOCABULARY=True`

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
  - optional lightweight text-conditioned query bias in the active multimodal presets
  - optional prompt-adaptive routing that scales the first query block by text-driven multiscale context
- Box init:
  - uses `extras["prior_boxes"]` if provided
  - else CAM-based init if `extras["cams"]` exists
  - else learnable anchor-like priors on a query grid (default)
  - optional full-image fallback via `MODEL.STM.QUERY_INIT_MODE: "full_image"`
  - optional small-object bias via `MODEL.STM.QUERY_INIT_SMALL_OBJECT_BIAS`
  - optional prompt-adaptive prior scaling driven by routed FPN level weights
- Stage update loop:
  - self-attention with IoF-based attention bias
  - adaptive sampling/mixing (via `SAMPLE4D` + `AdaptiveMixing`)
  - classification head (`OBJECT_CLASSES + 1` with background)
  - optional query-to-text logit fusion against class prompt prototypes
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
  --annotations "C:/Users/tsake/OneDrive/Desktop/datasets/Aero_dataset/train.txt" "C:/Users/tsake/OneDrive/Desktop/datasets/Aero_dataset/test.txt" \
  --out-closed "C:/Users/tsake/OneDrive/Desktop/datasets/Aero_dataset/annotations/vocab_closed.json" \
  --out-open "C:/Users/tsake/OneDrive/Desktop/datasets/Aero_dataset/annotations/vocab_open.json" \
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

Optional local quality hook setup:

```bash
pip install pre-commit
pre-commit install
```

Colab quickstart (2-command setup):

```bash
git clone https://github.com/saket108/AeroMixer.git /content/AeroMixer
cd /content/AeroMixer && AEROMIXER_COLAB_PROFILE=minimal bash scripts/colab_bootstrap.sh
```

Notes:
- No external CLIP bootstrap step is required for the active AeroLite runtime.
- `AEROMIXER_COLAB_PROFILE=minimal` is the recommended stable profile.
- Full walkthrough: `docs/COLAB_QUICKSTART.md`.

Then run full pipeline (prepare + train + eval) from uploaded dataset:

```bash
python scripts/pipeline.py \
  --mode run \
  --data "/content/drive/MyDrive/Aero_dataset.zip" \
  --preset prod \
  --output-dir output/colab_aero_dataset \
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
  --data "C:/Users/tsake/OneDrive/Desktop/datasets/Aero_dataset" \
  --preset prod \
  --output-dir output/aero_dataset_run \
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

Small-object tiling (train + eval in one run):

```bash
python scripts/pipeline.py \
  --mode run \
  --data "C:/Users/tsake/OneDrive/Desktop/datasets/Aero_dataset" \
  --preset prod \
  --output-dir output/aero_dataset_run_tiled \
  --epochs 30 \
  --batch-size 4 \
  --tile-size 640 \
  --tile-overlap 0.2 \
  --tile-min-cover 0.35 \
  --tile-splits train,val,test
```

This writes:
- `<OUTPUT_DIR>/tiling_report.json`
- tiled working dataset under `output/_auto_data_prep/tiled_datasets/...`

Validation behavior:
- `scripts/pipeline.py` validates dataset integrity before training by default.
- If validation fails, pipeline stops early.
- Override only when needed: `--allow-validation-errors` or `--skip-validation`.
- Auto-fix label/split issues (YOLO):  
  `python scripts/validate_dataset.py --data <dataset> --fix --report-out output/quality/report.json`

Direct low-level train command (advanced/debug):

```bash
python train_net.py --config-file config_files/presets/full.yaml
```

Notes:
- `--data` supports zip, folder, `data.yaml`, or `.json`.
- Flat YOLO folders (`images/` + `labels/`) need `--split-ratio`.
- `scripts/pipeline.py` handles dataset preparation, class-count overrides, train, and eval in one public workflow.

Project script policy:
- Public entrypoints: `scripts/pipeline.py`, `scripts/validate_dataset.py`, `scripts/freeze_dataset_version.py`, `scripts/run_baseline_benchmarks.py`, `scripts/release_tools.py`, `scripts/validate_docker_inference.py`, `scripts/colab_bootstrap.sh`
- Internal helpers: `scripts/internal/*`
- Archived research tools: `scripts/archive/*`

Stable inference/eval pipeline:

```bash
python scripts/pipeline.py \
  --mode eval \
  --data "C:/Users/tsake/OneDrive/Desktop/datasets/Aero_dataset" \
  --preset prod \
  --output-dir output/aero_dataset_inference \
  --model-weight checkpoints/model_final.pth
```

Dataset version freeze (for reproducibility records):

```bash
python scripts/freeze_dataset_version.py \
  --data "C:/Users/tsake/OneDrive/Desktop/datasets/Aero_dataset" \
  --out output/dataset_version.json
```

Eval:

```bash
python test_net.py --config-file config_files/presets/full.yaml
```

Image demo:

```bash
python demo_image.py \
  --config-file config_files/presets/full.yaml \
  --image path/to/image.jpg \
  --output output/image_demo.png \
  --device cpu
```

Integrated threshold tuning after eval:

```bash
python scripts/pipeline.py \
  --mode eval \
  --data "C:/Users/tsake/OneDrive/Desktop/datasets/Aero_dataset" \
  --preset prod \
  --output-dir output/aero_dataset_inference \
  --model-weight checkpoints/model_final.pth \
  --tune-thresholds \
  --threshold-grid 0.0,0.05,0.1,0.2,0.3
```

Outputs:
- `<OUTPUT_DIR>/threshold_tuning.json`
- `<OUTPUT_DIR>/pipeline_manifest.json` updated with best threshold summary
- benchmark row append to `benchmarks/summary.csv`

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
  --config-file config_files/presets/full.yaml \
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

## Baseline Benchmark Runner (Unified)

Run AeroMixer / YOLO / DETR commands and append one comparable benchmark format:

```bash
python scripts/run_baseline_benchmarks.py \
  --dataset aero_dataset \
  --preset benchmark \
  --aeromixer-cmd "python train_net.py --config-file config_files/presets/full.yaml --skip-final-test" \
  --aeromixer-metrics "output/aero_dataset_run/inference/aero_dataset/result_image.log" \
  --yolo-cmd "python path/to/yolo_train.py ..." \
  --yolo-metrics "runs/detect/train/results.csv" \
  --detr-cmd "python path/to/detr_train.py ..." \
  --detr-metrics "outputs/detr/metrics.json"
```

Output:
- `benchmarks/summary.csv` (canonical shared table)
- `output/benchmarks/baseline_run.json` (run detail dump)

## License

See `LICENSE`.

## Release & Benchmark Tracking

- Changelog: `CHANGELOG.md`
- Benchmark table: `benchmarks/summary.csv`
- Release checklist: `RELEASE_CHECKLIST.md`
- Release tooling:
  - `python scripts/release_tools.py prepare --version 0.5.0`
  - `python scripts/release_tools.py check --version 0.5.0`
  - `python scripts/release_tools.py tag --version 0.5.0 --dry-run`

## Docker Inference

Build:

```bash
docker build -t aeromixer:latest .
```

Validate docker contract (build + container command + JSON report):

```bash
python scripts/validate_docker_inference.py \
  --image aeromixer:latest \
  --container-cmd "python scripts/pipeline.py --help"
```

Run inference (contract):

```bash
docker run --rm -it \
  -v /host/datasets:/data \
  -v /host/runs:/work/output \
  aeromixer:latest \
  --data /data/Aero_dataset \
  --preset prod \
  --output-dir /work/output/aero_dataset_infer_run \
  --model-weight /work/output/aero_dataset_train_run/checkpoints/model_final.pth
```

See `docs/INFERENCE_CONTRACT.md` for full details.
