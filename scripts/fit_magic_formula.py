#!/usr/bin/env python3
"""Fit the analytical Magic Formula baseline to a dataset with scipy.

The learned models in this project must beat a *properly fitted* Magic Formula, not a
hand-guessed one — so this script exists to produce that baseline honestly.

    python scripts/fit_magic_formula.py --source synthetic
    python scripts/fit_magic_formula.py --source kit --root data/raw --out results/mf_kit.yaml

Residuals are load-normalised (F/Fz): without that, high-load samples dominate the fit
and the resulting mu is biased toward the heavy end of the load range.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tire_nn.data import adapters, make_synthetic  # noqa: E402
from tire_nn.evaluation import evaluate_on  # noqa: E402
from tire_nn.physics.fitting import fit_magic_formula  # noqa: E402
from tire_nn.physics.pacejka import MagicFormulaTire  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="synthetic")
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--out", default=None, help="write the fitted parameters to this YAML file")
    ap.add_argument("--n", type=int, default=6000, help="samples, synthetic source only")
    args = ap.parse_args()

    if args.source == "synthetic":
        df = make_synthetic(args.n, seed=0)
    else:
        df = adapters.load(args.source, root=args.root)
    print(f"[data] source={args.source}  n={len(df)}  type={df['source'].iloc[0]}")

    px, py = fit_magic_formula(df)
    model = MagicFormulaTire(px, py)
    metrics = evaluate_on(model, df, targets=("Fx", "Fy"))

    result = {
        "source": args.source,
        "n_samples": int(len(df)),
        "longitudinal": {k: float(v) for k, v in px.as_dict().items()},
        "lateral": {k: float(v) for k, v in py.as_dict().items()},
        "fit_metrics": {k: float(v) for k, v in metrics.items()},
    }
    print(yaml.safe_dump(result, sort_keys=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(yaml.safe_dump(result, sort_keys=False))
        print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
