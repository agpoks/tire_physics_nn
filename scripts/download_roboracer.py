#!/usr/bin/env python3
"""Download helper for: RoboRacer / F1TENTH model-structured NN dataset (tire-set and mass-change runs)

Type: real measurement (small scale)
Status: SOURCE UNVERIFIED — confirm the release and licence before use

Nothing large is downloaded automatically (PLAN.md 4.4). This script prints the exact
manual steps and exits non-zero so a missing dataset can never be mistaken for an empty
one.
"""

import argparse
import sys
from pathlib import Path

DATASET = 'RoboRacer / F1TENTH model-structured NN dataset (tire-set and mass-change runs)'
DATA_TYPE = 'real measurement (small scale)'
TARGET = 'roboracer'
URL = '(no confirmed source at the time of writing)'
STEPS = """  1. confirm the exact release and licence; record them in PLAN.md 4.4
  2. place the logs as  <root>/roboracer/<experiment_name>/*.csv
     — one subdirectory per experiment (tire set, mass); the directory name becomes
     tire_id, which is what makes the tire-set comparison possible
  3. note the wheel radius and vehicle geometry: Experiment 3 needs exact values"""


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
          f"print(adapters.load('roboracer', root={args.root!r}).head())\"")
    return 1


if __name__ == "__main__":
    sys.exit(main())
