# Colab Quickstart (GPU)

This flow uses the trimmed AeroLite runtime and avoids extra legacy encoder setup.

## 1) Enable GPU
- Runtime -> Change runtime type -> `T4 GPU` (or better).

## 2) Fresh clone
```bash
%cd /content
!rm -rf AeroMixer CLIP
!git clone https://github.com/saket108/AeroMixer.git
%cd /content/AeroMixer
!git rev-parse --short HEAD
```

## 3) Bootstrap (minimal profile, recommended)
```bash
!AEROMIXER_COLAB_PROFILE=minimal bash scripts/colab_bootstrap.sh
```

## 4) Upload dataset zip
```python
from google.colab import files
files.upload()  # e.g. aero_data.zip
```

## 5) Train + eval
```bash
!python scripts/pipeline.py \
  --mode run \
  --data /content/aero_data.zip \
  --preset full \
  --output-dir /content/output/aero_colab_full \
  --epochs 30 \
  --batch-size 4 \
  --num-workers 2 \
  --split-ratio 80,10,10
```
