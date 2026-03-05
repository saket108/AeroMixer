#!/usr/bin/env bash
set -euo pipefail

# One-command helper for Colab after cloning AeroMixer.
# Usage:
#   bash scripts/colab_oneclick_train.sh /content/aero_data.zip [output_dir] [preset]

DATA_PATH="${1:-}"
OUTPUT_DIR="${2:-/content/output/aeromixer_colab_run}"
PRESET="${3:-full}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SPLIT_RATIO="${SPLIT_RATIO:-80,10,10}"
PROFILE="${AEROMIXER_COLAB_PROFILE:-minimal}"

if [[ -z "${DATA_PATH}" ]]; then
  echo "Usage: bash scripts/colab_oneclick_train.sh <data_path> [output_dir] [preset]"
  exit 2
fi

echo "[oneclick] repo: $(pwd)"
echo "[oneclick] data: ${DATA_PATH}"
echo "[oneclick] output: ${OUTPUT_DIR}"
echo "[oneclick] preset: ${PRESET}"
echo "[oneclick] profile: ${PROFILE}"

AEROMIXER_COLAB_PROFILE="${PROFILE}" bash scripts/colab_bootstrap.sh

python scripts/pipeline.py \
  --mode run \
  --data "${DATA_PATH}" \
  --preset "${PRESET}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --split-ratio "${SPLIT_RATIO}"
