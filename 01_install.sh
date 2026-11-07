#!/usr/bin/env bash
# Step 1 - install everything into the existing conda env "minimax-h3".
# Run:  bash 01_install.sh
set -eo pipefail            # (no -u: conda's activate scripts use unset variables)

ENV_NAME="ltx25"
BASE="${LTX_BASE:-/media/avidmech/data/LTX 2.5 Video Genetaion}"
REPO="$BASE/LTX-2"

echo "== Base folder: $BASE"
if [ ! -d "$(dirname "$BASE")" ]; then
  echo "ERROR: $(dirname "$BASE") does not exist. Is the data drive mounted?"; exit 1
fi
mkdir -p "$BASE"/{models,outputs,inputs,logs,hf_cache}

# ---- activate the conda env -------------------------------------------------
CONDA_SH=""
for c in "$(conda info --base 2>/dev/null || true)/etc/profile.d/conda.sh" \
         "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh" \
         "$HOME/miniforge3/etc/profile.d/conda.sh" "/opt/conda/etc/profile.d/conda.sh"; do
  if [ -f "$c" ]; then CONDA_SH="$c"; break; fi
done
if [ -z "$CONDA_SH" ]; then
  echo "ERROR: could not find conda. Activate '$ENV_NAME' yourself and re-run."; exit 1
fi
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME"
echo "== Using python: $(which python)"

python - << 'PY'
import sys
if sys.version_info < (3, 12):
    sys.exit(f"ERROR: Python {sys.version.split()[0]} found, LTX-2.5 docs recommend >= 3.12. "
             "Create a new env with python=3.12 or tell me and I'll adapt the script.")
print("Python OK:", sys.version.split()[0])
PY

echo "== GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || \
  echo "WARNING: nvidia-smi failed"

# ---- LTX-2 code -------------------------------------------------------------
if [ -d "$REPO/.git" ]; then
  echo "== Updating LTX-2 repo"; git -C "$REPO" pull --ff-only
else
  echo "== Cloning LTX-2 repo"; git clone https://github.com/Lightricks/LTX-2.git "$REPO"
fi

python -m pip install --upgrade pip
# Install from source (the PyPI release predates LTX-2.5's Gemma-4 encoder / duration head).
python -m pip install -e "$REPO/packages/ltx-core" -e "$REPO/packages/ltx-pipelines"
python -m pip install streamlit huggingface_hub hf_xet imageio-ffmpeg safetensors pillow

echo "== Verifying imports"
python - << 'PY'
import importlib, torch
for m in ("ltx_core", "ltx_pipelines", "streamlit", "huggingface_hub", "imageio_ffmpeg", "safetensors"):
    importlib.import_module(m); print("  import ok:", m)
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

cat << MSG

Install finished.
Next:
  1. Accept the license at https://huggingface.co/Lightricks/LTX-2.5  (button: "Agree and Access")
  2. hf auth login          (paste a read token)
  3. python 02_download.py
MSG
