"""LTX-2.5 Studio - Streamlit front-end for Lightricks' LTX-2.5 (22B distilled).

Three sections
  1. Text  -> Video       (ltx_pipelines.distilled)
  2. Image -> Video       (ltx_pipelines.distilled, first-frame conditioning)
  3. Revise a video       (a) retake a time window  (ltx_pipelines.retake)
                          (b) re-cast with a character reference (ltx_pipelines.ic_lora + Ingredients IC-LoRA)

Each generation runs as a separate `python -m ltx_pipelines.<name>` process (the documented CLI), so the
model is re-loaded from disk every time. That is slower than keeping it in memory but keeps the app
independent of internal Python APIs.

Revision notes
  * Two-stage pipelines need width and height to be multiples of 64 (544 was not). Presets fixed.
  * Preflight checks catch bad resolution / frame counts before a 70 GB model load starts.
  * Failed runs now show the real error line instead of the nanobind shutdown noise.
  * The subprocess environment drops LD_LIBRARY_PATH (system CUDA 12.2) and user site-packages,
    so only the packages inside this venv are used.
"""
import hashlib
import json
import os
import random
import re
import shlex
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import streamlit as st

import config as C

st.set_page_config(page_title="LTX-2.5 Studio", page_icon="🎬", layout="wide")

# ----------------------------------------------------------------------------- setup
try:
    for d in (C.OUT_DIR, C.IN_DIR, C.LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
except OSError as e:
    st.error(f"Cannot use the base folder `{C.BASE}`: {e}. Is the drive mounted?")
    st.stop()

ENV = os.environ.copy()
ENV["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # recommended with FP8 quantization
ENV["HF_HOME"] = str(C.HF_HOME)
ENV.pop("LD_LIBRARY_PATH", None)      # never pick up the system CUDA 12.2 libraries (caused libcupti errors)
ENV["PYTHONNOUSERSITE"] = "1"         # ignore ~/.local packages; use only this venv

DISTILLED = "ltx_pipelines.distilled"
RETAKE = "ltx_pipelines.retake"
IC_LORA = "ltx_pipelines.ic_lora"

# Flags documented on the LTX-2.5 model card; assumed available if --help could not be read.
KNOWN_FLAGS = {"--transformer-path", "--text-encoder-path", "--video-vae-path", "--audio-vae-path",
               "--spatial-upsampler-path", "--duration-head-path", "--num-frames", "--prompt", "--seed",
               "--output-path", "--image", "--quantization", "--offload"}

SIZE_MULTIPLE = 64   # two-stage pipelines: height and width must both be multiples of 64

PRESETS = {  # every value below is a multiple of 64
    "Landscape 768×512 (test)": (768, 512),
    "Landscape 960×512": (960, 512),
    "Landscape 16:9 1024×576": (1024, 576),
    "Portrait 9:16 576×1024": (576, 1024),
    "Portrait 512×768": (512, 768),
    "Square 704×704": (704, 704),
    "Small preview 640×384": (640, 384),
}
FRAMES = {  # frames must satisfy frames % 8 == 1
    "Auto (model picks the length)": None,
    "25 frames (≈1 s)": 25,
    "49 frames (≈2 s)": 49,
    "81 frames (≈3.4 s)": 81,
    "121 frames (≈5 s)": 121,
    "161 frames (≈6.7 s)": 161,
    "241 frames (≈10 s, heavy)": 241,
}

missing = [k for k in C.REQUIRED if not C.model_path(k).exists()]
if missing:
    st.error("Model files are missing: " + ", ".join(C.FILES[k][0] for k in missing)
             + ". Run `python 02_download.py` and then `python 03_check.py`.")
    st.stop()


# ----------------------------------------------------------------------------- helpers
@st.cache_resource(show_spinner="Reading pipeline options…")
def pipeline_help(module: str) -> str:
    """Text of `python -m <module> --help`, or '' if it could not be read."""
    try:
        r = subprocess.run([sys.executable, "-m", module, "--help"], capture_output=True, text=True,
                           timeout=300, env=ENV)
        return ((r.stdout or "") + (r.stderr or "")) if r.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


def supports(module: str, flag: str) -> bool:
    text = pipeline_help(module)
    return (flag in text) if text else (flag in KNOWN_FLAGS)


def missing_flags(module: str, flags) -> list:
    return [f for f in flags if not supports(module, f)]


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def out_path(kind: str, seed: int) -> Path:
    return C.OUT_DIR / f"{kind}_{stamp()}_s{seed}.mp4"


def save_upload(up, prefix: str) -> Path:
    data = up.getvalue()
    digest = hashlib.md5(data).hexdigest()[:10]
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", up.name)
    path = C.IN_DIR / f"{prefix}_{digest}_{safe}"
    if not path.exists():
        path.write_bytes(data)
    return path


def round_to(x: float, m: int = SIZE_MULTIPLE) -> int:
    return max(m, int(round(x / float(m))) * m)


def preflight(size, frames) -> bool:
    """Catch settings the pipeline would reject, before loading ~70 GB of weights."""
    problems = []
    if size:
        w, h = size
        if w % SIZE_MULTIPLE or h % SIZE_MULTIPLE:
            problems.append(f"Resolution {w}×{h} must be multiples of {SIZE_MULTIPLE} "
                            f"(nearest valid: {round_to(w)}×{round_to(h)}).")
    if frames and frames % 8 != 1:
        problems.append(f"Frame count {frames} must be 8k+1 (e.g. 97, 121, 161).")
    for p in problems:
        st.error(p)
    return not problems


@st.cache_data(show_spinner=False)
def probe(path: str, mtime: float):
    """(frame_count, duration_seconds) or (None, None)."""
    try:
        import imageio_ffmpeg
        n, secs = imageio_ffmpeg.count_frames_and_secs(path)
        return int(n), float(secs)
    except Exception:  # noqa: BLE001
        return None, None


def image_to_reference_clip(img_path: Path, out: Path, width: int, height: int, frames: int = 9, fps: int = 24):
    """Turn a still image into a tiny MP4 (frames % 8 == 1) so it can be used as video conditioning."""
    import imageio_ffmpeg
    vf = (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
          f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p")
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loop", "1", "-framerate", str(fps), "-i", str(img_path),
           "-vf", vf, "-frames:v", str(frames), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(r.stderr[-600:])


def show_last(key: str):
    p = st.session_state.get(key)
    if p and Path(p).exists():
        st.video(p)
        st.caption(f"Saved to `{p}`")
        st.download_button("Download MP4", data=Path(p).read_bytes(), file_name=Path(p).name,
                           mime="video/mp4", key=f"dl_{key}")


NOISE = re.compile(r"^\s*(nanobind:|- leaked|- \.\.\. skipped|See https://nanobind)")


def is_noise(line: str) -> bool:
    """nanobind prints harmless leak reports at interpreter shutdown; they hide the real error."""
    return bool(NOISE.match(line))


def last_error(log_file: Path):
    """Most recent real error line in the log (ignoring nanobind noise), or None."""
    try:
        lines = log_file.read_text(errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        s = line.strip()
        if not s or is_noise(s) or s.startswith("File "):
            continue
        if re.search(r"(Error|Exception)\b", s):
            return s
    return None


def run_job(cmd, out: Path, title: str, params: dict, state_key: str) -> bool:
    """Run one pipeline process, stream its log, and remember the result."""
    log_file = C.LOG_DIR / f"{out.stem}.log"
    tail = deque(maxlen=30)
    t0 = time.time()
    ok = False
    with st.status(f"{title}: running — the first minutes are spent loading ~70 GB of weights",
                   expanded=True) as status:
        st.code(" ".join(shlex.quote(c) for c in cmd), language="bash")
        box = st.empty()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                bufsize=1, env=ENV, cwd=str(C.BASE))
        try:
            with open(log_file, "w") as lf:
                for line in proc.stdout:
                    lf.write(line)                      # the log file keeps everything
                    if not is_noise(line):              # the live view hides shutdown noise
                        tail.append(line.rstrip())
                        box.code("\n".join(tail), language=None)
            rc = proc.wait()
        finally:
            if proc.poll() is None:      # e.g. the page was re-run: don't leave a GPU job orphaned
                proc.terminate()
        ok = rc == 0 and out.exists()
        mins = (time.time() - t0) / 60
        if ok:
            out.with_suffix(".json").write_text(json.dumps(
                {"title": title, "command": cmd, **params, "minutes": round(mins, 1)}, indent=2, default=str))
            st.session_state[state_key] = str(out)
            status.update(label=f"{title}: done in {mins:.1f} min", state="complete", expanded=False)
        else:
            status.update(label=f"{title}: failed (exit code {rc})", state="error", expanded=True)
    if not ok:
        err = last_error(log_file)
        if err:
            st.error(f"The pipeline did not produce a video.\n\n**Error:** `{err}`")
        else:
            st.error("The pipeline did not produce a video. Check the last lines above for an unknown flag, "
                     "or lower the resolution/length.")
        st.caption(f"Full log: `{log_file}`")
    return ok


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    preset = st.selectbox("Resolution", list(PRESETS), index=0,
                          help="Width and height must be multiples of 64 (two-stage pipeline). "
                               "Lower this first if you run out of VRAM.")
    frames_label = st.selectbox("Length", list(FRAMES), index=4,
                                help="Frame counts must be 8k+1. Seconds are approximate (≈24 fps).")
    random_seed = st.checkbox("Random seed", value=True)
    seed_in = st.number_input("Seed", min_value=0, max_value=2**31 - 1, value=42, disabled=random_seed)
    vae_label = st.radio("Video VAE", ["DiffVAE (higher quality)", "Conv VAE (faster, lighter)"])
    use_fp8 = st.checkbox("FP8 weight cast", value=True, help="Needed to fit the 22B model on a 24 GB GPU.")
    use_offload = st.checkbox("CPU offload", value=True, help="Moves weights to system RAM between steps.")
    extra_args = st.text_input("Extra CLI arguments (advanced)", value="",
                               help="Appended to every command, e.g. a flag revealed by logs/help_*.txt")
    if not pipeline_help(DISTILLED):
        st.warning("Couldn't read `ltx_pipelines.distilled --help`. Is `ltx_pipelines` installed in this "
                   "environment? Falling back to the flags documented on the model card.")

VAE_KEY = "video_vae" if vae_label.startswith("DiffVAE") or not C.file_ok("video_vae_conv") else "video_vae_conv"
if vae_label.startswith("Conv") and VAE_KEY == "video_vae":
    st.sidebar.warning("Conv VAE not downloaded; using DiffVAE.")
SIZE = PRESETS[preset]
FRAME_COUNT = FRAMES[frames_label]


def next_seed() -> int:
    return random.randint(0, 2**31 - 1) if random_seed else int(seed_in)


def build_cmd(module: str, prompt: str, out: Path, seed: int, frames, size) -> list:
    """Model arguments shared by all pipelines; flags a pipeline doesn't list are skipped."""
    cmd = [sys.executable, "-m", module]

    def add(flag, *vals):
        if supports(module, flag):
            cmd.extend([flag, *map(str, vals)])

    add("--transformer-path", C.model_path("transformer"))
    add("--text-encoder-path", C.model_path("text_encoder"))
    add("--video-vae-path", C.model_path(VAE_KEY))
    add("--audio-vae-path", C.model_path("audio_vae"))
    add("--spatial-upsampler-path", C.model_path("spatial_up"))
    if C.model_path("duration_head").exists():
        add("--duration-head-path", C.model_path("duration_head"))
    add("--prompt", prompt)
    add("--seed", seed)
    add("--output-path", out)
    if frames:
        add("--num-frames", frames)
    if size and supports(module, "--height") and supports(module, "--width"):
        add("--height", size[1])
        add("--width", size[0])
    if use_fp8:
        add("--quantization", "fp8-cast")
    if use_offload:
        add("--offload", "cpu")
    if extra_args.strip():
        cmd.extend(shlex.split(extra_args))
    return cmd


# ----------------------------------------------------------------------------- UI
st.title("🎬 LTX-2.5 Studio")
st.caption("Local video + audio generation on your RTX 3090 · distilled 22B model · "
           "detailed prompts (subject, action, camera, lighting, mood, sound) give the best results.")

tab1, tab2, tab3 = st.tabs(["1 · Text → Video", "2 · Image → Video", "3 · Revise video / character"])

# ---- 1. text to video ---------------------------------------------------------
with tab1:
    p1 = st.text_area("Prompt", height=170, key="p1",
                      placeholder="A close-up macro shot of a pink rose bud at sunrise, the petals slowly unfurl, "
                                  "dewdrops glisten, the camera slowly pushes in, warm golden backlight, "
                                  "soft birdsong…")
    if st.button("Generate video", type="primary", key="go1", disabled=not p1.strip()):
        if preflight(SIZE, FRAME_COUNT):
            s = next_seed()
            out = out_path("t2v", s)
            run_job(build_cmd(DISTILLED, p1.strip(), out, s, FRAME_COUNT, SIZE), out, "Text → Video",
                    {"prompt": p1, "seed": s, "frames": FRAME_COUNT, "size": SIZE}, "last_t2v")
    show_last("last_t2v")

# ---- 2. image to video --------------------------------------------------------
with tab2:
    left, right = st.columns([1, 1])
    with left:
        up = st.file_uploader("First-frame image", type=["png", "jpg", "jpeg", "webp"], key="img2v")
        strength = st.slider("Image strength", 0.3, 1.0, 1.0, 0.05,
                             help="1.0 = the first frame matches your image exactly; lower lets the model drift.")
    with right:
        if up is not None:
            st.image(up, caption=up.name, use_container_width=True)
            try:
                from PIL import Image
                w, h = Image.open(up).size
                if (w > h) != (SIZE[0] > SIZE[1]) and w != h:
                    st.info("This image's orientation differs from the selected resolution preset — "
                            "pick the matching preset in the sidebar to avoid heavy cropping/stretching.")
            except Exception:  # noqa: BLE001
                pass
    p2 = st.text_area("Motion / scene prompt", height=130, key="p2",
                      placeholder="The camera slowly dollies out as wind moves through the grass…")
    if st.button("Animate image", type="primary", key="go2", disabled=(up is None or not p2.strip())):
        if preflight(SIZE, FRAME_COUNT):
            img_path = save_upload(up, "img")
            s = next_seed()
            out = out_path("i2v", s)
            cmd = build_cmd(DISTILLED, p2.strip(), out, s, FRAME_COUNT, SIZE)
            cmd += ["--image", str(img_path), "0", f"{strength:.2f}"]
            run_job(cmd, out, "Image → Video",
                    {"prompt": p2, "seed": s, "frames": FRAME_COUNT, "size": SIZE, "image": str(img_path)},
                    "last_i2v")
    show_last("last_i2v")

# ---- 3. revise ----------------------------------------------------------------
with tab3:
    st.subheader("Revise an existing video")
    how = st.radio("Source video", ["One of my generated videos", "Upload a video"], horizontal=True, key="src_how")
    src = None
    if how.startswith("One"):
        vids = sorted(C.OUT_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if vids:
            src = st.selectbox("Video", vids, format_func=lambda p: p.name, key="src_pick")
        else:
            st.info("No generated videos yet — create one in tab 1 or 2 first.")
    else:
        vup = st.file_uploader("MP4 / MOV / WebM", type=["mp4", "mov", "webm"], key="src_up")
        if vup is not None:
            src = save_upload(vup, "src")

    if src is not None:
        st.video(str(src))
        n_frames, secs = probe(str(src), src.stat().st_mtime)
        if n_frames:
            st.caption(f"{n_frames} frames · {secs:.1f} s · {n_frames / secs:.1f} fps")
            if n_frames % 8 != 1:
                st.warning("Retake needs a source with 8k+1 frames (e.g. 97, 121). This one has "
                           f"{n_frames}; videos made in tabs 1–2 with the lengths above are fine.")

        mode = st.radio("What do you want to change?",
                        ["Retake a time window (change the action or a character in part of the clip)",
                         "Re-cast with a character reference (IC-LoRA — experimental)"], key="rev_mode")

        # ---- 3a retake
        if mode.startswith("Retake"):
            lo, hi = 0.0, float(secs or 5.0)
            win = st.slider("Seconds to regenerate", lo, hi, (lo, min(hi, 2.0)), 0.1, key="win")
            p3 = st.text_area("What should happen in that window?", height=130, key="p3",
                              placeholder="Same scene, but the woman now wears a red coat and turns to smile at the camera…")
            need = ["--video-path", "--start-time", "--end-time", "--prompt", "--output-path",
                    "--transformer-path", "--text-encoder-path"]
            miss = missing_flags(RETAKE, need)
            if miss:
                st.warning(f"`ltx_pipelines.retake` doesn't list: {', '.join(miss)}. See `logs/help_{RETAKE}.txt` "
                           "(created by 03_check.py) and pass the equivalent flags in the sidebar's extra arguments.")
            if st.button("Regenerate this part", type="primary", key="go3a", disabled=not p3.strip()):
                s = next_seed()
                out = out_path("retake", s)
                cmd = build_cmd(RETAKE, p3.strip(), out, s, None, None)
                cmd += ["--video-path", str(src), "--start-time", f"{win[0]:.2f}", "--end-time", f"{win[1]:.2f}"]
                run_job(cmd, out, "Retake", {"prompt": p3, "seed": s, "window": win, "source": str(src)}, "last_rev")

        # ---- 3b IC-LoRA character re-cast
        else:
            lora = next(C.LORA_DIR.rglob("*.safetensors"), None) if C.LORA_DIR.exists() else None
            if lora is None:
                st.info("The Ingredients IC-LoRA isn't downloaded yet. Run "
                        "`python 02_download.py --with-ingredients` and reload this page.")
            else:
                st.caption(f"LoRA: `{lora.name}`. This mode runs the IC-LoRA pipeline with a reference clip "
                           "and a new prompt; it creates a *new* video rather than editing the old one.")
            ref_mode = st.radio("Character reference", ["The selected video", "A character image"],
                                horizontal=True, key="ref_mode")
            ref_img = None
            if ref_mode == "A character image":
                ref_img = st.file_uploader("Character / reference-sheet image", type=["png", "jpg", "jpeg", "webp"],
                                           key="refimg")
            c1, c2 = st.columns(2)
            lora_strength = c1.slider("LoRA strength", 0.0, 1.5, 1.0, 0.05)
            cond_strength = c2.slider("Reference strength", 0.0, 1.0, 1.0, 0.05)
            p4 = st.text_area("New scene prompt", height=130, key="p4",
                              placeholder="The same character walks through a rainy neon-lit street at night…")
            ready = lora is not None and p4.strip() and (ref_mode == "The selected video" or ref_img is not None)
            if st.button("Generate with reference", type="primary", key="go3b", disabled=not ready):
                if preflight(SIZE, FRAME_COUNT):
                    s = next_seed()
                    out = out_path("recast", s)
                    try:
                        if ref_mode == "A character image":
                            ref = C.IN_DIR / f"{save_upload(ref_img, 'ref').stem}_ref.mp4"
                            image_to_reference_clip(save_upload(ref_img, "ref"), ref,
                                                    round_to(SIZE[0]), round_to(SIZE[1]))
                        else:
                            ref = src
                        cmd = build_cmd(IC_LORA, p4.strip(), out, s, FRAME_COUNT, SIZE)
                        cmd += ["--lora", str(lora), f"{lora_strength:.2f}",
                                "--video-conditioning", str(ref), f"{cond_strength:.2f}"]
                        run_job(cmd, out, "Character re-cast",
                                {"prompt": p4, "seed": s, "reference": str(ref), "lora": str(lora)}, "last_rev")
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Could not prepare the reference clip: {e}")
    show_last("last_rev")
