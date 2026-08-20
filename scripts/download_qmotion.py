#!/usr/bin/env python3
"""Download helper for: Q-Motion tire dataset with inflation-pressure variation

Type: real measurement
Status: SOURCE UNVERIFIED — confirm the release and licence before use

Nothing large is downloaded automatically (PLAN.md 4.4). This script prints the exact
manual steps and exits non-zero so a missing dataset can never be mistaken for an empty
one.
"""

import argparse
import sys
from pathlib import Path

DATASET = 'Q-Motion tire dataset with inflation-pressure variation'
DATA_TYPE = 'real measurement'
TARGET = 'qmotion'
URL = '(no confirmed source at the time of writing)'
STEPS = """  1. confirm the exact release and licence; record them in PLAN.md 4.4
  2. place the tables in the target directory
  3. CHECK THE PRESSURE UNIT and pass it explicitly:
       adapters.load("qmotion", root=..., pressure_unit="kPa" | "bar" | "psi" | "Pa")
     A wrong pressure unit is silent — the model simply learns nothing from a context
     channel that is off by a factor of 100."""


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
          f"print(adapters.load('qmotion', root={args.root!r}).head())\"")
    return 1


if __name__ == "__main__":
    sys.exit(main())
