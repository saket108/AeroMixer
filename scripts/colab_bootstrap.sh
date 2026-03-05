#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "[1/4] Upgrading pip tooling..."
python3 -m pip install --upgrade pip setuptools wheel

PROFILE="${AEROMIXER_COLAB_PROFILE:-minimal}"   # minimal|full
echo "[2/4] Installing Colab requirements profile: ${PROFILE}"
if [[ "${PROFILE}" == "minimal" ]]; then
  REQ_FILE="requirements_lock_colab_minimal.txt"
  if [[ ! -f "${REQ_FILE}" ]]; then
    REQ_FILE="requirements_colab_minimal.txt"
  fi
else
  REQ_FILE="requirements_lock_colab.txt"
  if [[ ! -f "${REQ_FILE}" ]]; then
    REQ_FILE="requirements_colab.txt"
  fi
fi
python3 -m pip install --no-cache-dir -r "${REQ_FILE}"

echo "[3/4] Verifying vendored CLIP module..."
if [[ ! -f "clip/clip.py" ]]; then
  echo "[clip] Vendored module missing at clip/clip.py"
  exit 2
fi

echo "[4/4] Verifying core imports..."
python3 - <<'PY'
import torch
import numpy
import yacs
import clip

print("torch:", torch.__version__)
print("numpy:", numpy.__version__)
print("yacs: ok")
print("clip: ok ->", clip.__file__)
PY

echo "Bootstrap complete."
