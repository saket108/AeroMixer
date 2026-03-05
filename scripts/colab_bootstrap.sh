#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "[1/4] Upgrading pip tooling..."
python3 -m pip install --upgrade pip setuptools wheel

echo "[2/4] Installing Colab-pinned requirements..."
REQ_FILE="requirements_lock_colab.txt"
if [[ ! -f "${REQ_FILE}" ]]; then
  REQ_FILE="requirements_colab.txt"
fi
python3 -m pip install --no-cache-dir -r "${REQ_FILE}"

echo "[3/4] Installing OpenAI CLIP..."
python3 -m pip install --no-cache-dir ftfy
if ! python3 -m pip install --no-cache-dir "git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1"; then
  echo "[clip] Direct pip git install failed, trying fallback install..."
  TMP_CLIP_DIR="$(mktemp -d)"
  git clone https://github.com/openai/CLIP.git "${TMP_CLIP_DIR}/CLIP"
  git -C "${TMP_CLIP_DIR}/CLIP" checkout dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1
  if ! python3 -m pip install --no-cache-dir --no-build-isolation "${TMP_CLIP_DIR}/CLIP"; then
    echo "[clip] No-build-isolation install failed, trying setup.py install..."
    (cd "${TMP_CLIP_DIR}/CLIP" && python3 setup.py install)
  fi
  rm -rf "${TMP_CLIP_DIR}"
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
print("clip: ok")
PY

echo "Bootstrap complete."
