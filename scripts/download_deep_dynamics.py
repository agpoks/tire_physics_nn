#!/usr/bin/env python3
"""Download helper for: Deep Dynamics datasets (BayesRace simulation + Indy Autonomous Challenge logs)

Type: MIXED: BayesRace is SIMULATED, the IAC logs are REAL measurement
Status: public git repository

Nothing large is downloaded automatically (PLAN.md 4.4). This script prints the exact
manual steps and exits non-zero so a missing dataset can never be mistaken for an empty
one.
"""

import argparse
import sys
from pathlib import Path

DATASET = 'Deep Dynamics datasets (BayesRace simulation + Indy Autonomous Challenge logs)'
DATA_TYPE = 'MIXED: BayesRace is SIMULATED, the IAC logs are REAL measurement'
TARGET = 'deep_dynamics'
URL = 'https://github.com/linklab-uva/deep-dynamics'
STEPS = """  1. git clone https://github.com/linklab-uva/deep-dynamics
  2. copy the dataset files into  <root>/deep_dynamics/bayesrace/  and
     <root>/deep_dynamics/iac/  respectively
  3. keep the two subsets separate — they carry different type labels and must never
     be pooled into one results row
  4. cite Chrosniak, Ning, Behl, IEEE RA-L 2024 (arXiv:2312.04374)"""


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
          f"print(adapters.load('deep_dynamics', root={args.root!r}).head())\"")
    return 1


if __name__ == "__main__":
    sys.exit(main())
