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
python3 -m pip install --no-cache-dir "git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1"

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
