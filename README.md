# Video-geenration-using-LTX2.5-


A local, self-hosted video **and audio** generation studio built on Lightricks' **LTX-2.5 (22B distilled)** model. It is a set of setup scripts plus a Streamlit web app that runs on a single consumer GPU (developed on an **RTX 3090, 24 GB**).

The app wraps the official `ltx_pipelines` command-line tools. Every generation runs as a separate `python -m ltx_pipelines.<name>` process, so the app does not depend on any internal Python API of LTX (the trade-off: weights are re-loaded from disk for every run).

## Features

| Tab | What it does | Pipeline used |
|---|---|---|
| **1 · Text → Video** | Generate a clip (with audio) from a text prompt | `ltx_pipelines.distilled` |
| **2 · Image → Video** | Animate a still image used as the first frame | `ltx_pipelines.distilled` (`--image`) |
| **3 · Revise video / character** | (a) Regenerate a time window of an existing clip. (b) Create a new clip with a character reference (IC-LoRA "Ingredients", experimental) | `ltx_pipelines.retake`, `ltx_pipelines.ic_lora` |

Other highlights:

- Clip lengths from 1 s up to **20 s** (481 frames at 24 fps), or a custom number of seconds.
- Clips longer than ~10 s are generated in **overlapping windows** (`--chunk-pixel-frames`, `--chunk-carry-frames`, `--chunk-blend-frames`).
- **Auto length** mode: the duration head picks the clip length from your prompt.
- FP8 weight cast and CPU offload so the 22B model fits a 24 GB GPU.
- Pre-flight checks (resolution, frame count, window settings) *before* a ~70 GB model load starts.
- Real error messages are extracted from the log (harmless `nanobind` shutdown noise is hidden).
- Every output is saved as `.mp4` with a `.json` file next to it (prompt, seed, command, run time) and a full log.

## Requirements

| | Requirement |
|---|---|
| OS | Linux (the scripts are bash and the default paths are Linux paths) |
| GPU | NVIDIA GPU with CUDA; 24 GB VRAM (RTX 3090 class) is the target. The check script warns below 23 GB |
| System RAM | **64 GB or more advised** (CPU offload keeps weights in RAM) |
| Disk | **About 75 GB free** on the target drive for the core weights (more if you add the optional files) |
| Python | **3.12 or newer** (the install script refuses older versions) |
| Conda | Miniconda / Anaconda / Miniforge (the scripts look for `conda.sh` in the usual places) |
| Git | To clone the LTX-2 repository |
| Hugging Face account | Free account with an access token (read) and the model licence accepted |

## What is inside this repo

```
.
├── 01_install.sh     # Step 1: clone LTX-2, install everything into the conda env
├── 02_download.py    # Step 2: download the model weights from Hugging Face
├── 03_check.py       # Step 3: verify environment, weights and pipeline CLI flags
├── run_app.sh        # Step 4: launch the Streamlit app
├── config.py         # Shared settings: paths, repo names, list of weight files
└── app.py            # The Streamlit app (see note below)
```

> **Note on `app.py`:** `run_app.sh` starts `app.py`. If your copy of the app is named `app_long.py` (the version with 15–20 s long-clip support), rename it:
> ```bash
> mv app_long.py app.py
> ```
> or change the last line of `run_app.sh` to `streamlit run app_long.py ...`.

Put all of these files together in one folder (for example `~/ltx25-studio`). The weights, outputs and the LTX-2 code are **not** stored in this folder; they go to the base folder described below.

## Where everything is stored

All heavy data lives under one **base folder** (`BASE`).

Default:

```
/media/avidmech/data/LTX 2.5 Video Genetaion
```

Change it for every script at once with an environment variable:

```bash
export LTX_BASE="/path/to/your/big/drive/ltx"
```

(Set it in the same shell before running the scripts, or add it to `~/.bashrc`.)

Layout created under `BASE`:

```
BASE/
├── LTX-2/                         # cloned Lightricks/LTX-2 source code
├── models/
│   ├── ltx-2.5/                   # core weights (from Lightricks/LTX-2.5)
│   │   ├── diffusion_models/
│   │   ├── text_encoders/
│   │   ├── vae/
│   │   ├── model_patches/
│   │   └── latent_upscale_models/
│   └── ic-lora-ingredients/       # optional IC-LoRA (character re-cast)
├── outputs/                       # generated videos (.mp4 + .json)
├── inputs/                        # uploaded images/videos saved by the app
├── logs/                          # per-run logs and help_*.txt from 03_check.py
└── hf_cache/                      # Hugging Face cache (kept off your home drive)
```

> The default path contains spaces. The scripts quote it correctly. If you use your own path, spaces are fine too, but a path without spaces is simpler.

## Installation

### 0. Create the conda environment

`01_install.sh` **activates** an environment named `ltx25` but does not create it, so create it first with Python 3.12 or newer:

```bash
conda create -n ltx25 python=3.12 -y
```

The environment name is set by `ENV_NAME="ltx25"` at the top of `01_install.sh` and `run_app.sh`. If you prefer another name, edit it in both files.

Make sure the data drive is mounted, because the install script stops if the parent folder of `BASE` does not exist.

### 1. Run the install script

```bash
bash 01_install.sh
```

What it does, in order:

1. Creates `BASE/{models,outputs,inputs,logs,hf_cache}`.
2. Finds `conda.sh` and activates the `ltx25` environment.
3. Checks that Python is 3.12 or newer.
4. Prints your GPU name, VRAM and driver version (`nvidia-smi`).
5. Clones `https://github.com/Lightricks/LTX-2.git` into `BASE/LTX-2` (or runs `git pull --ff-only` if it already exists).
6. Upgrades `pip`, then installs **from source**, in editable mode:
   - `packages/ltx-core`
   - `packages/ltx-pipelines`

   (Installed from source because the PyPI release predates the Gemma-4 text encoder and duration head used by LTX-2.5.)
7. Installs the extras: `streamlit`, `huggingface_hub`, `hf_xet`, `imageio-ffmpeg`, `safetensors`, `pillow`.
8. Verifies that all modules import and prints the PyTorch version and whether CUDA is available.

The script can be re-run safely.

### 2. Accept the model licence and log in to Hugging Face

The weights are **gated**. Before downloading:

1. Open <https://huggingface.co/Lightricks/LTX-2.5> while logged in and click **"Agree and Access"**.
2. Create a **read** token at <https://huggingface.co/settings/tokens>.
3. Log in inside the environment:

```bash
conda activate ltx25
hf auth login          # paste your read token
```

(Alternatively, set the token in the environment: `export HF_TOKEN=hf_xxx`.)

If you also want the character re-cast mode, open the IC-LoRA model page and accept its licence too: <https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients>.

## Downloading the weights

Run from the folder containing the scripts, inside the `ltx25` environment:

```bash
conda activate ltx25
python 02_download.py
```

Options:

```bash
python 02_download.py                      # core files (~72 GB)
python 02_download.py --with-ingredients   # + IC-LoRA for character consistency
python 02_download.py --with-temporal      # + x2 temporal upscaler (not used by the app)
python 02_download.py --with-ingredients --with-temporal   # everything
```

The script:

- checks you are logged in,
- skips files that are already complete,
- resumes partial downloads,
- compares the needed size against free disk space before starting,
- tells you exactly what to do if access is denied (licence not accepted) or a download fails.

### Files downloaded

All files come from the Hugging Face repo `Lightricks/LTX-2.5` and are saved under `BASE/models/ltx-2.5/`.

| Purpose | File (inside the repo) | Size | Downloaded |
|---|---|---|---|
| Transformer (22B distilled) | `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors` | ~42 GB | always |
| Text encoder (Gemma 4 12B + projection) | `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` | ~26.3 GB | always |
| Video VAE (DiffVAE, higher quality) | `vae/ltx-2.5-video-vae-bf16.safetensors` | ~1.47 GB | always |
| Video VAE (Conv, faster and lighter) | `vae/ltx-2.5-video-vae-conv-bf16.safetensors` | ~1.45 GB | always (optional in the app) |
| Audio VAE | `vae/ltx-2.5-audio-vae-bf16.safetensors` | ~0.37 GB | always |
| Duration head (auto clip length) | `model_patches/ltx-2.5-duration-head-bf16.safetensors` | small | always |
| Spatial upscaler x2 | `latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | ~1.0 GB | always |
| Temporal upscaler x2 | `latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors` | ~0.26 GB | only with `--with-temporal` |
| IC-LoRA "Ingredients" | repo `Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients` → `BASE/models/ic-lora-ingredients/` | see HF page | only with `--with-ingredients` |

The file list and expected sizes are defined in `config.py` (`FILES`). The check script treats a file as complete when it is at least 95% of its expected size.

### Downloading manually (alternative)

If you prefer the CLI, download a single file into the same layout:

```bash
export HF_HOME="$LTX_BASE/hf_cache"
hf download Lightricks/LTX-2.5 \
  diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
  --local-dir "$LTX_BASE/models/ltx-2.5"
```

The relative folder names (`diffusion_models/`, `vae/`, ...) must be preserved, because `config.py` expects them.

## Verify the setup

```bash
python 03_check.py
```

It checks, and prints `OK`, `WARN` or `FAIL` for each item:

1. **Environment**: Python version, PyTorch and CUDA, GPU name and VRAM, system RAM (warning only), required modules, base folder and free space.
2. **Weights**: each file exists, has a plausible size, and its safetensors header can be read (detects truncated or corrupt downloads). The IC-LoRA is reported as a warning if missing.
3. **Pipeline CLI options**: runs `--help` for `ltx_pipelines.distilled`, `.retake` and `.ic_lora` and checks that the flags the app uses are present. The help text is saved to `BASE/logs/help_<module>.txt`.

Exit code `0` means you are ready. If it reports blocking problems, fix them and run it again.

## Run the app

```bash
bash run_app.sh
```

Then open <http://127.0.0.1:8501> in your browser.

`run_app.sh` activates the environment, sets `HF_HOME` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and starts Streamlit bound to `127.0.0.1` only (not reachable from other machines) with an upload limit of 500 MB.

To reach it from another computer on your network, change `--server.address` in `run_app.sh` (be aware there is no authentication).

## Using the app

### Sidebar settings (apply to all tabs)

| Setting | Meaning |
|---|---|
| **Resolution** | Presets, all multiples of 64: 768×512 (default, recommended for long clips), 960×512, 1024×576, portrait 576×1024, portrait 512×768, square 704×704, preview 640×384. Lower it first if you run out of VRAM |
| **Length** | Auto (the duration head decides), or 25 / 49 / 81 / 121 / 161 / 241 / 361 / 481 frames (≈1 s to ≈20 s), or a custom number of seconds (1–20 s). Frame counts are always snapped to the **8k+1** grid |
| **Windowing** | `Auto` (turns on above 10 s), `Always on`, or `Off`. Generates long clips in overlapping windows |
| **Window size** | 73, 97 (default), 121 or 145 pixel frames |
| **Overlap carried between windows** | 9, 17, 25 (default), 33 or 41 frames. Must be smaller than the window size. Larger overlap means smoother seams but slower |
| **Seam blend** | Default (full carry) or Hard cut (0) |
| **Random seed / Seed** | Random per run, or a fixed seed for reproducibility |
| **Video VAE** | DiffVAE (higher quality) or Conv VAE (faster, lighter). Falls back to DiffVAE if the Conv file is not downloaded |
| **FP8 weight cast** | On by default; needed to fit the 22B model in 24 GB (`--quantization fp8-cast`) |
| **CPU offload** | On by default; moves weights to system RAM between steps (`--offload cpu`) |
| **Extra CLI arguments** | Free text appended to every command, for flags listed in `logs/help_*.txt` |

The sidebar also shows the frame count, duration and an estimated number of video tokens for the current settings.

### Tab 1: Text → Video

Write a detailed prompt and press **Generate video**. Good prompts describe the subject, action, camera, lighting, mood and sound. For 15–20 s clips the whole clip receives one prompt, so describe a slow progression (start → middle → end) with a single camera move rather than several unrelated actions.

Example:

```
A close-up macro shot of a pink rose bud at sunrise, the petals slowly unfurl,
dewdrops glisten, the camera slowly pushes in, warm golden backlight, soft birdsong.
```

Tip: test a prompt at 121 frames (~5 s) before launching a long run.

### Tab 2: Image → Video

Upload a first-frame image (PNG, JPG, JPEG or WebP), set **Image strength** (1.0 = the first frame matches your image exactly; lower lets the model drift), describe the motion and press **Animate image**. Pick a resolution preset with the same orientation as your image to avoid heavy cropping.

### Tab 3: Revise video / character

Choose a source: one of your generated videos, or an uploaded MP4 / MOV / WebM.

**(a) Retake a time window.** Select the seconds to regenerate with the slider, describe what should happen in that window, and press **Regenerate this part**. The source must have an **8k+1 frame count** (for example 97, 121, 361, 481); videos from tabs 1–2 using the lengths above qualify. The retake pipeline has no windowing flags, so very long sources may need a lot of memory; use a shorter clip if it fails.

**(b) Re-cast with a character reference (IC-LoRA, experimental).** Requires `python 02_download.py --with-ingredients`. Use the selected video, or upload a character image (it is converted to a tiny reference clip). Adjust LoRA strength and reference strength, write the new scene prompt and press **Generate with reference**. This creates a *new* video; it does not edit the old one.

## Output files

For each successful run, in `BASE/outputs/`:

```
t2v_20261107_153000_s123456.mp4      # the video (with audio)
t2v_20261107_153000_s123456.json     # title, prompt, seed, size, frames, windowing, full command, minutes
```

File name prefixes: `t2v` (text to video), `i2v` (image to video), `retake`, `recast`. The complete log of every run, including failures, is in `BASE/logs/<name>.log`. The `.json` file makes any result reproducible.

## Troubleshooting

| Problem | What to do |
|---|---|
| `ERROR: ... does not exist. Is the data drive mounted?` | Mount the drive or set `LTX_BASE` to a path that exists |
| `could not find conda` | Activate the environment yourself and run the script again, or install conda in one of the standard locations |
| `Python x.y found, LTX-2.5 docs recommend >= 3.12` | Recreate the environment with `python=3.12` |
| `Not logged in` when downloading | Run `hf auth login` (or set `HF_TOKEN`) and retry |
| `Access denied` when downloading | Open the model page, click **Agree and Access** with the same account as your token, and retry |
| `Not enough free space on the target drive` | Free disk space or choose another `LTX_BASE` |
| `looks incomplete` / `header unreadable` in `03_check.py` | Delete that file and run `02_download.py` again |
| Missing model files error in the app | Run `python 02_download.py` and then `python 03_check.py` |
| Out of memory / `CUDA error` | Use 768×512 or 640×384, set windowing to Auto or Always on, choose the Conv VAE, pick a shorter length, keep FP8 and CPU offload enabled |
| `libcupti` or other CUDA library errors | The app already removes `LD_LIBRARY_PATH` and ignores `~/.local` packages for its subprocesses. If you run the pipelines by hand, do the same (`env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 ...`) |
| "unknown flag" or no video produced | Look at `logs/help_ltx_pipelines.*.txt` for the real flag names and pass them in **Extra CLI arguments**. The app skips any flag a pipeline does not list |
| Retake says the source needs 8k+1 frames | Use a source with 97, 121, 361 or 481 frames, or re-generate it from tabs 1–2 |
| Resolution/frame error before the run starts | Resolution must be multiples of 64; frame count must be 8k+1; clips are capped at 481 frames (≈20 s) |

Every run also prints the exact command it used, so you can copy it and run it in a terminal to debug.

## Notes and limits

- Every generation reloads the model from disk (about 70 GB), so the first minutes of each run are loading time.
- The 20 s cap is a conservative app limit. Lightricks lists roughly 20 s per clip for LTX-2; the LTX-2.5 model card does not state a limit, and longer clips are untested.
- Frame rate is assumed to be 24 fps for the seconds display.
- The IC-LoRA character mode is experimental.
- The temporal upscaler can be downloaded but is not used by the app.

## Credits and licence

- Model and pipelines: [Lightricks](https://github.com/Lightricks/LTX-2) (LTX-2 / LTX-2.5). Model weights are subject to the licence you accept on <https://huggingface.co/Lightricks/LTX-2.5>.
- App UI: [Streamlit](https://streamlit.io).

Add your own licence for the scripts and app in this repo (for example MIT) as a `LICENSE` file.
