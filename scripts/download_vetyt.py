#!/usr/bin/env python3
"""Download helper for: VeTyT bicycle tyre measurements (Politecnico di Milano)

Type: real measurement
Status: verified publications; data available on request from the authors

Nothing large is downloaded automatically (PLAN.md 4.4). This script prints the exact
manual steps and exits non-zero so a missing dataset can never be mistaken for an empty
one.
"""

import argparse
import sys
from pathlib import Path

DATASET = 'VeTyT bicycle tyre measurements (Politecnico di Milano)'
DATA_TYPE = 'real measurement'
TARGET = 'vetyt'
URL = 'https://doi.org/10.1080/00423114.2024.2338143'
STEPS = """  1. read the paper (Dell'Orto et al., Vehicle System Dynamics 2024) and the
     test-rig paper (Measurement 2022, doi:10.1016/j.measurement.2022.111813)
  2. request the measurement tables from the authors, or extract them from the
     supplementary material
  3. place them in the target directory, one file per tyre if possible
  4. confirm the sign convention on one pure-lateral sweep before trusting a run"""


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
          f"print(adapters.load('vetyt', root={args.root!r}).head())\"")
    return 1


if __name__ == "__main__":
    sys.exit(main())
