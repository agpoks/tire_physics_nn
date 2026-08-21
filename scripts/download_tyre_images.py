#!/usr/bin/env python3
"""Download real tyre photographs from the Hugging Face Hub.

Type: **real measurement** (photographs). Licence: **CC BY 4.0** — openly licensed and
needing no credentials, which is what makes these usable here when the Kaggle mirrors
need an account and the Mendeley TyreNet release is a single 1.9 GB RAR.

    python -m pip install huggingface_hub
    python scripts/download_tyre_images.py

Two repositories, together roughly 1 850 images:

    NMiriams/Good_Tires        good / serviceable tyres
    NMiriams/Defective_Tires   defective tyres

What the labels mean, and do not mean
-------------------------------------
These are **binary condition** labels — good versus defective — where "defective" mixes
tread wear with cracking, bulges and punctures. They are *not* a measured tread depth,
and they are not a graded wear scale. Treat them as a two-level ordinal (good < defective
in severity) and nothing more; see the imaging chapter for what that supports.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPOS = {
    "good": "NMiriams/Good_Tires",
    "defective": "NMiriams/Defective_Tires",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--subdir", default="tyre_quality")
    ap.add_argument("--limit", type=int, default=250,
                    help="images per class (default 250; the full set is ~900 each)")
    ap.add_argument("--max-size", type=int, default=256,
                    help="downscale the long edge to this many pixels on save. The "
                         "originals are ~2 MB photographs and the models here work at "
                         "64 px, so storing them full size wastes ~2 GB for nothing.")
    args = ap.parse_args()

    try:
        from huggingface_hub import list_repo_files, hf_hub_download
    except ImportError:
        print("huggingface_hub is not installed.\n\n  python -m pip install huggingface_hub\n")
        print("Without it, the imaging notebook falls back to synthetic tread textures,")
        print("which are generated in-repo and labelled as synthetic throughout.")
        return 1

    out_root = Path(args.root) / args.subdir
    total = 0
    for label, repo in REPOS.items():
        target = out_root / label
        target.mkdir(parents=True, exist_ok=True)
        files = [f for f in list_repo_files(repo, repo_type="dataset")
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        files.sort()
        if args.limit:
            files = files[:args.limit]
        print(f"[{label}] {len(files)} images from {repo}")
        for i, name in enumerate(files):
            destination = target / Path(name).name
            if destination.exists():
                continue
            cached = hf_hub_download(repo, name, repo_type="dataset")
            if args.max_size:
                try:
                    from PIL import Image

                    with Image.open(cached) as img:
                        img.thumbnail((args.max_size, args.max_size))
                        img.convert("RGB").save(destination, quality=90)
                except Exception:                     # noqa: BLE001 - fall back to a copy
                    shutil.copyfile(cached, destination)
            else:
                shutil.copyfile(cached, destination)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(files)}")
        total += len(files)
        print(f"[{label}] -> {target}")

    print(f"\n{total} images under {out_root}")
    print("Licence: CC BY 4.0 — attribute the source repositories if you republish.")
    print("Load with:  from tire_nn.data.tread_images import load_tyre_quality_images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
