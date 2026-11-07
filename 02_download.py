"""Step 2 - download the LTX-2.5 weights into BASE/models.

Run inside the minimax-h3 env:
    python 02_download.py                      # core files (~72 GB)
    python 02_download.py --with-ingredients   # + IC-LoRA for character consistency
    python 02_download.py --with-temporal      # + x2 temporal upscaler (not used by the app)
Re-running is safe: finished files are skipped, partial downloads resume.
"""
import argparse
import os
import shutil
import sys

import config as C

os.environ["HF_HOME"] = str(C.HF_HOME)          # keep the HF cache off your home drive

from huggingface_hub import hf_hub_download, snapshot_download, whoami   # noqa: E402
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError          # noqa: E402

LICENSE_URL = f"https://huggingface.co/{C.HF_REPO}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-ingredients", action="store_true")
    ap.add_argument("--with-temporal", action="store_true")
    args = ap.parse_args()

    C.BASE.mkdir(parents=True, exist_ok=True)
    C.MODELS.mkdir(parents=True, exist_ok=True)

    try:
        who = whoami()
        print(f"Logged in to Hugging Face as: {who.get('name')}")
    except Exception:
        print("Not logged in. Run `hf auth login` (or set HF_TOKEN), then re-run this script.")
        return 1

    keys = C.REQUIRED + C.DEFAULT_EXTRA + (C.OPTIONAL if args.with_temporal else [])
    todo = [k for k in keys if not C.file_ok(k)]
    done = [k for k in keys if k not in todo]
    for k in done:
        print(f"  skip (already complete): {C.FILES[k][0]}")

    need_gb = sum((C.FILES[k][1] or 0.5) for k in todo) * 1.05
    free_gb = shutil.disk_usage(C.BASE).free / 1e9
    print(f"Still to download: ~{need_gb:.0f} GB | free on target drive: {free_gb:.0f} GB")
    if need_gb > free_gb:
        print("Not enough free space on the target drive."); return 1

    for k in todo:
        rel = C.FILES[k][0]
        print(f"\n>>> Downloading {rel}")
        try:
            hf_hub_download(repo_id=C.HF_REPO, filename=rel, local_dir=str(C.MODELS))
        except GatedRepoError:
            print(f"\nAccess denied: open {LICENSE_URL}, click 'Agree and Access' while logged in "
                  "as the same account, then re-run.")
            return 2
        except HfHubHTTPError as e:
            print(f"\nDownload failed for {rel}: {e}")
            return 3

    if args.with_ingredients:
        print(f"\n>>> Downloading IC-LoRA repo {C.LORA_REPO}")
        try:
            snapshot_download(repo_id=C.LORA_REPO, local_dir=str(C.LORA_DIR))
        except (GatedRepoError, HfHubHTTPError) as e:
            print(f"Could not download the IC-LoRA ({e}). You may need to accept its license on its model page.")
            return 4

    print("\nDownload step finished. Next: python 03_check.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
