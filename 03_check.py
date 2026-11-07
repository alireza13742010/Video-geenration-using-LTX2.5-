"""Step 3 - verify the environment, the downloaded weights and the pipeline CLI options.

    python 03_check.py
Exit code 0 = ready to run the app.
"""
import importlib.util
import shutil
import subprocess
import sys

import config as C

problems = 0


def say(ok, msg, fatal=True):
    global problems
    sym = "OK  " if ok else ("FAIL" if fatal else "WARN")
    if not ok and fatal:
        problems += 1
    print(f"[{sym}] {msg}")


def has_module(name):
    return importlib.util.find_spec(name) is not None


print("=== 1. Environment ===")
say(sys.version_info >= (3, 12), f"Python {sys.version.split()[0]} (>= 3.12 recommended)")
try:
    import torch
    cuda = torch.cuda.is_available()
    say(cuda, f"torch {torch.__version__}, CUDA available: {cuda}")
    if cuda:
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        say(vram >= 23, f"GPU: {torch.cuda.get_device_name(0)} ({vram:.0f} GB VRAM)")
        if vram < 30:
            print("       -> The 22B model needs FP8 cast + CPU offload on this card (the app enables both).")
except Exception as e:  # noqa: BLE001
    say(False, f"torch import failed: {e}")
try:
    kb = int(next(l for l in open("/proc/meminfo") if l.startswith("MemTotal")).split()[1])
    say(kb / 1e6 >= 64, f"System RAM: {kb / 1e6:.0f} GB (>= 64 GB advised for CPU offload)", fatal=False)
except Exception:  # noqa: BLE001
    pass
for mod in ("ltx_core", "ltx_pipelines", "streamlit", "huggingface_hub", "imageio_ffmpeg", "safetensors", "PIL"):
    say(has_module(mod), f"module {mod}")
if C.BASE.exists():
    say(True, f"Base folder: {C.BASE} ({shutil.disk_usage(C.BASE).free / 1e9:.0f} GB free)")
else:
    say(False, f"Base folder missing: {C.BASE}")

print("\n=== 2. Downloaded weights ===")
try:
    from safetensors import safe_open
except Exception:  # noqa: BLE001
    safe_open = None
for key in C.REQUIRED + C.DEFAULT_EXTRA + C.OPTIONAL:
    p = C.model_path(key)
    required = key in C.REQUIRED
    if not p.exists():
        say(False, f"{C.FILES[key][0]} -> missing", fatal=required)
        continue
    size_gb = p.stat().st_size / 1e9
    exp = C.FILES[key][1]
    size_ok = C.file_ok(key)
    detail = f"{size_gb:.2f} GB" + (f" (expected ~{exp} GB)" if exp else "")
    if not size_ok:
        say(False, f"{C.FILES[key][0]} -> {detail}: looks incomplete, re-run 02_download.py", fatal=required)
        continue
    if safe_open is None:
        say(True, f"{C.FILES[key][0]} -> {detail}")
        continue
    try:  # reads only the header; fails if the file is truncated/corrupt
        with safe_open(str(p), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            dtype = f.get_slice(keys[0]).get_dtype() if keys else "?"
        say(True, f"{C.FILES[key][0]} -> {detail}, {len(keys)} tensors, first dtype {dtype}")
    except Exception as e:  # noqa: BLE001
        say(False, f"{C.FILES[key][0]} -> header unreadable ({e}); delete it and re-run 02_download.py", fatal=required)

lora = next(C.LORA_DIR.rglob("*.safetensors"), None) if C.LORA_DIR.exists() else None
say(lora is not None, f"IC-LoRA Ingredients: {lora if lora else 'not downloaded (only needed for the character re-cast mode)'}",
    fatal=False)

print("\n=== 3. Pipeline command-line options ===")
EXPECT = {
    "ltx_pipelines.distilled": (
        ["--transformer-path", "--text-encoder-path", "--video-vae-path", "--audio-vae-path",
         "--spatial-upsampler-path", "--prompt", "--output-path", "--seed", "--image"],
        ["--duration-head-path", "--num-frames", "--height", "--width", "--quantization", "--offload"]),
    "ltx_pipelines.retake": (
        ["--video-path", "--start-time", "--end-time", "--prompt", "--output-path"],
        ["--transformer-path", "--text-encoder-path", "--seed"]),
    "ltx_pipelines.ic_lora": (
        ["--lora", "--video-conditioning", "--prompt", "--output-path"],
        ["--transformer-path", "--spatial-upsampler-path"]),
}
C.LOG_DIR.mkdir(parents=True, exist_ok=True)
for module, (must, nice) in EXPECT.items():
    try:
        r = subprocess.run([sys.executable, "-m", module, "--help"], capture_output=True, text=True, timeout=300)
        text = (r.stdout or "") + (r.stderr or "")
        (C.LOG_DIR / f"help_{module}.txt").write_text(text)
        if r.returncode != 0:
            say(False, f"{module} --help failed (see logs/help_{module}.txt)")
            continue
        miss = [f for f in must if f not in text]
        warn = [f for f in nice if f not in text]
        say(not miss, f"{module}: required flags {'all present' if not miss else 'MISSING ' + ', '.join(miss)}",
            fatal=(module == "ltx_pipelines.distilled"))
        if warn:
            say(False, f"{module}: optional flags not listed: {', '.join(warn)} (app will skip them)", fatal=False)
    except Exception as e:  # noqa: BLE001
        say(False, f"{module}: could not run ({e})")

print()
if problems:
    print(f"{problems} blocking problem(s). Fix them, then re-run: python 03_check.py")
    sys.exit(1)
print("All checks passed. Start the app with:  bash run_app.sh")
