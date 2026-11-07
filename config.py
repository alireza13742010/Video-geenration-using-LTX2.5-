"""Shared settings for the LTX-2.5 setup scripts and the Streamlit app.

Everything (weights, code, outputs, HF cache) lives under BASE.
Override with:  export LTX_BASE="/some/other/path"
"""
import os
from pathlib import Path

BASE = Path(os.environ.get("LTX_BASE", "/media/avidmech/data/LTX 2.5 Video Genetaion"))

MODELS = BASE / "models" / "ltx-2.5"
LORA_DIR = BASE / "models" / "ic-lora-ingredients"
REPO_DIR = BASE / "LTX-2"
OUT_DIR = BASE / "outputs"
IN_DIR = BASE / "inputs"
LOG_DIR = BASE / "logs"
HF_HOME = BASE / "hf_cache"

HF_REPO = "Lightricks/LTX-2.5"
LORA_REPO = "Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients"

# key -> (path inside the HF repo, approx. size in GB from the HF file listing; None = unknown)
FILES = {
    "transformer":    ("diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors", 42.0),
    "text_encoder":   ("text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors", 26.3),
    "video_vae":      ("vae/ltx-2.5-video-vae-bf16.safetensors", 1.47),
    "video_vae_conv": ("vae/ltx-2.5-video-vae-conv-bf16.safetensors", 1.45),
    "audio_vae":      ("vae/ltx-2.5-audio-vae-bf16.safetensors", 0.365),
    "duration_head":  ("model_patches/ltx-2.5-duration-head-bf16.safetensors", None),
    "spatial_up":     ("latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors", 0.996),
    "temporal_up":    ("latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors", 0.262),
}

REQUIRED = ["transformer", "text_encoder", "video_vae", "audio_vae", "duration_head", "spatial_up"]
DEFAULT_EXTRA = ["video_vae_conv"]      # small, lets the app offer a faster VAE
OPTIONAL = ["temporal_up"]              # only with: 02_download.py --with-temporal


def model_path(key: str) -> Path:
    return MODELS / FILES[key][0]


def file_ok(key: str) -> bool:
    """Cheap check: file exists and is at least ~95% of the expected size."""
    p = model_path(key)
    if not p.exists():
        return False
    gb = FILES[key][1]
    if gb is None:
        return p.stat().st_size > 1_000_000
    return p.stat().st_size >= 0.95 * gb * 1e9
