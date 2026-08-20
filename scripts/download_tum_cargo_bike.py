#!/usr/bin/env python3
"""Download helper for: TUM cargo bicycle tire characteristics

Type: real measurement
Status: SOURCE UNVERIFIED — locate and confirm before use

Nothing large is downloaded automatically (PLAN.md 4.4). This script prints the exact
manual steps and exits non-zero so a missing dataset can never be mistaken for an empty
one.
"""

import argparse
import sys
from pathlib import Path

DATASET = 'TUM cargo bicycle tire characteristics'
DATA_TYPE = 'real measurement'
TARGET = 'tum_cargo_bike'
URL = '(no confirmed source at the time of writing)'
STEPS = """  1. locate the actual release (mediaTUM, TUM library, or the publication's
     supplementary material)
  2. record the URL and licence in PLAN.md 4.4 and papers/references.bib
  3. place the tables in the target directory
  4. adjust TUM_COLUMNS in tire_nn/data/tum_cargo_bike.py to the real headers

Until step 2 is done, do not report any quantitative result from this source.
Consider the verified VeTyT cargo-bicycle measurements instead."""


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
          f"print(adapters.load('tum_cargo_bike', root={args.root!r}).head())\"")
    return 1


if __name__ == "__main__":
    sys.exit(main())
