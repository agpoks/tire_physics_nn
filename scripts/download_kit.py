#!/usr/bin/env python3
"""Download helper for: KIT inner-drum tire force transmission characteristic on dry asphalt

Type: real measurement
Status: verified source; licence acceptance required, so the download is manual

Nothing large is downloaded automatically (PLAN.md 4.4). This script prints the exact
manual steps and exits non-zero so a missing dataset can never be mistaken for an empty
one.
"""

import argparse
import sys
from pathlib import Path

DATASET = 'KIT inner-drum tire force transmission characteristic on dry asphalt'
DATA_TYPE = 'real measurement'
TARGET = 'kit'
URL = 'https://radar.kit.edu/radar/en/dataset/p0rr2jc5wmf0drf8'
STEPS = """  1. open the RADAR4KIT landing page above
  2. accept the CC BY-NC-SA 4.0 licence and download the archive
  3. extract it into the target directory (keep the folder structure)
  4. the simulated slalom driving-cycle folder is SIMULATION, not measurement —
     the adapter excludes it by default (include_simulation=True to keep it)"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/raw", help="dataset root (default: data/raw)")
    args = ap.parse_args()

    target = Path(args.root) / TARGET
    print(f"Dataset : {DATASET}")
    print(f"Type    : {DATA_TYPE}")
    print(f"Source  : {URL}")
    print(f"Target  : {target}")
    if target.exists() and any(target.rglob("*")):
        print("\nAlready present — nothing to do.")
        return 0
    print("\nManual steps:")
    print(STEPS)
    print(f"\nThen:  python -c \"from tire_nn.data import adapters; "
          f"print(adapters.load('kit', root={args.root!r}).head())\"")
    return 1


if __name__ == "__main__":
    sys.exit(main())
