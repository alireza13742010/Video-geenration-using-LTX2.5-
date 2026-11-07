#!/usr/bin/env bash
# Step 4 - launch the Streamlit app inside the minimax-h3 env.
set -eo pipefail
ENV_NAME="ltx25"
BASE="${LTX_BASE:-/media/avidmech/data/LTX 2.5 Video Genetaion}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for c in "$(conda info --base 2>/dev/null || true)/etc/profile.d/conda.sh" \
         "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh" \
         "$HOME/miniforge3/etc/profile.d/conda.sh" "/opt/conda/etc/profile.d/conda.sh"; do
  if [ -f "$c" ]; then source "$c"; break; fi
done
conda activate "$ENV_NAME"

export HF_HOME="$BASE/hf_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$HERE"
exec streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.maxUploadSize 500
