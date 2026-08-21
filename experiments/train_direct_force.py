#!/usr/bin/env python3
"""Experiment 1 — direct tire-force modelling on test-bench data (PLAN.md §5).

Trains the whole ablation ladder on one dataset and reports, for every rung:
accuracy (RMSE/MAE/R2), interpolation vs extrapolation, physical violations
(symmetry, zero-slip, friction envelope, dissipativity) and — optionally — the
learning curve over training-set size.

    python experiments/train_direct_force.py --config configs/exp1_direct_force.yaml
    python experiments/train_direct_force.py --config ... --set data.source=kit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tire_nn.data import Normalizer, TireDataset, make_synthetic, split_by_group  # noqa: E402
from tire_nn.evaluation import audit, evaluate_on, learning_curve_sizes, load_range_holdout, slip_range_holdout  # noqa: E402
from tire_nn.evaluation import plots  # noqa: E402
from tire_nn.models import build_model  # noqa: E402
from tire_nn.physics.fitting import fit_magic_formula  # noqa: E402
from tire_nn.training import TrainConfig, append_summary, set_seed, train_model  # noqa: E402


def load_config(path: str, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for item in overrides:
        key, _, value = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def load_dataframe(dcfg: dict) -> pd.DataFrame:
    """Load the configured source, falling back to synthetic with a loud message.

    Real datasets require a manual download (PLAN.md §4.4); the experiment must still
    be runnable and verifiable without them, so the fallback is explicit and recorded
    in the summary CSV rather than silent.
    """
    source = dcfg.get("source", "synthetic")
    if source == "synthetic":
        return make_synthetic(
            n=dcfg.get("n_samples", 6000),
            noise=dcfg.get("noise", 0.01),
            pressure_range=dcfg.get("pressure_range"),
            seed=dcfg.get("seed", 0),
        )
    from tire_nn.data import adapters

    try:
        return adapters.load(source, root=dcfg.get("root", "data/raw"))
    except FileNotFoundError as exc:
        print(f"[data] {exc}\n[data] falling back to synthetic data — results are labelled accordingly.")
        return make_synthetic(n=dcfg.get("n_samples", 6000), noise=dcfg.get("noise", 0.01))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/exp1_direct_force.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. training.epochs=10")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    set_seed(cfg.get("seed", 0))
    out_root = ROOT / cfg.get("out_dir", "results/exp1")
    out_root.mkdir(parents=True, exist_ok=True)

    dcfg = cfg["data"]
    df = load_dataframe(dcfg)
    source_label = df["source"].iloc[0]
    targets = tuple(dcfg.get("targets", ["Fy"]))
    context_keys = tuple(dcfg.get("context_keys", []))
    print(f"[data] source={source_label}  n={len(df)}  targets={targets}  context={context_keys}")

    train_df, val_df, test_df = split_by_group(df, fractions=tuple(dcfg.get("split", [0.7, 0.15, 0.15])),
                                               seed=cfg.get("seed", 0))
    norm = Normalizer.fit(train_df, ("alpha", "kappa", "Fz") + targets)
    norm.save(out_root / "norm.json")

    extrap = cfg["evaluation"].get("extrapolation", "slip")
    if extrap == "slip":
        inner_df, outer_df = slip_range_holdout(df)
    elif extrap == "load":
        inner_df, outer_df = load_range_holdout(df)
    else:
        inner_df, outer_df = train_df, test_df

    tire_index = {name: i for i, name in enumerate(sorted(df["tire_id"].unique()))}
    make_ds = lambda d: TireDataset(d, targets=targets, context_keys=context_keys, tire_index=tire_index)

    tcfg_base = cfg["training"]
    summary_path = out_root / "summary.csv"
    trained: dict[str, torch.nn.Module] = {}

    for name, mkw in cfg["models"].items():
        print(f"\n=== {name} ===")
        run_dir = out_root / name
        mkw = dict(mkw)
        penalty = mkw.pop("friction_penalty", False)

        if name == "magic_formula":
            px, py = fit_magic_formula(train_df)
            model = build_model(name, px=px, py=py, **mkw)
            print(f"  fitted mu_y={float(py.mu):.3f} B_y={float(py.B):.2f} C_y={float(py.C):.2f} "
                  f"E_y={float(py.E):.2f} k_mu={float(py.k_mu):.3f}")
            history = {"best_epoch": -1, "best_val_loss": float("nan")}
        else:
            kwargs = dict(mkw)
            if name != "mlp" and name != "mlp_penalty":
                kwargs.setdefault("context_keys", context_keys)
                kwargs.setdefault("n_tires", len(tire_index) if len(tire_index) > 1 else 0)
            else:
                kwargs.setdefault("context_keys", context_keys)
                kwargs.setdefault("n_tires", len(tire_index) if len(tire_index) > 1 else 0)
            if name == "residual":
                px, py = fit_magic_formula(train_df)
                kwargs.update(px=px, py=py)
            model = build_model(name, **kwargs)
            tcfg = TrainConfig(
                **{k: v for k, v in tcfg_base.items() if k in TrainConfig.__dataclass_fields__},
                targets=targets,
                seed=cfg.get("seed", 0),
                device=cfg.get("device", "cpu"),
            )
            if penalty:
                tcfg.friction_penalty_weight = max(tcfg_base.get("friction_penalty_weight", 0.0), 1.0)
            history = train_model(model, make_ds(train_df), make_ds(val_df), tcfg, run_dir)

        trained[name] = model.eval()

        row = {"experiment": cfg["experiment"], "source": source_label, "model": name,
               "n_params": sum(p.numel() for p in model.parameters()),
               "n_train": len(train_df), "best_epoch": history["best_epoch"]}
        row.update({f"test_{k}": v for k, v in
                    evaluate_on(model, test_df, targets, context_keys, tire_index).items()})
        row.update({f"extrap_{k}": v for k, v in
                    evaluate_on(model, outer_df, targets, context_keys, tire_index).items()})
        # Audit against two friction limits: a deliberately generous one (a violation
        # of it is unambiguous) and the limit a controller would actually plan
        # against. A model can look clean against the generous limit and still promise
        # more grip than the tire has.
        audit_kw = dict(alpha_max=cfg["evaluation"].get("audit_alpha_max", 0.6),
                        kappa_max=cfg["evaluation"].get("audit_kappa_max", 0.6))
        generous = audit(model, mu_ref=cfg["evaluation"].get("audit_mu_generous", 1.5), **audit_kw)
        row.update(generous)
        tight = cfg["evaluation"].get("audit_mu_tight")
        if tight is not None:
            row["envelope_violation_tight"] = audit(model, mu_ref=float(tight), **audit_kw)["envelope_violation"]
            row["audit_mu_tight"] = float(tight)
        append_summary(summary_path, row)
        print("  " + "  ".join(f"{k}={v:.4g}" for k, v in row.items() if isinstance(v, float)))

    # --- learning curve: accuracy vs amount of training data -------------------
    if cfg["evaluation"].get("learning_curve", False):
        print("\n=== learning curve ===")
        lc_path = out_root / "learning_curve.csv"
        for n in learning_curve_sizes(len(train_df), cfg["evaluation"].get("learning_curve_points", 4)):
            subset = train_df.iloc[:n]
            for name, mkw in cfg["models"].items():
                if name == "magic_formula":
                    px, py = fit_magic_formula(subset)
                    model = build_model(name, px=px, py=py)
                else:
                    mkw = {k: v for k, v in mkw.items() if k != "friction_penalty"}
                    kwargs = dict(mkw, context_keys=context_keys,
                                  n_tires=len(tire_index) if len(tire_index) > 1 else 0)
                    if name == "residual":
                        px, py = fit_magic_formula(subset)
                        kwargs.update(px=px, py=py)
                    model = build_model(name, **kwargs)
                    tcfg = TrainConfig(
                        **{k: v for k, v in tcfg_base.items() if k in TrainConfig.__dataclass_fields__},
                        targets=targets, seed=cfg.get("seed", 0), device=cfg.get("device", "cpu"))
                    train_model(model, make_ds(subset), make_ds(val_df), tcfg, verbose=False)
                m = evaluate_on(model.eval(), test_df, targets, context_keys, tire_index)
                append_summary(lc_path, {"n_train": n, "model": name, **m})
            print(f"  n_train={n} done")

    # --- plots -----------------------------------------------------------------
    if cfg["evaluation"].get("plots", True):
        import matplotlib
        matplotlib.use("Agg")
        pdir = out_root / "plots"
        Fz_plot = float(train_df["Fz"].median())
        plots.save(plots.plot_lateral_curve(trained, Fz=Fz_plot,
                                            data=test_df[test_df["kappa"].abs() < 0.02]), pdir / "Fy_vs_alpha.png")
        plots.save(plots.plot_longitudinal_curve(trained, Fz=Fz_plot,
                                                 data=test_df[test_df["alpha"].abs() < 0.02]), pdir / "Fx_vs_kappa.png")
        plots.save(plots.plot_friction_ellipse(trained, Fz=Fz_plot), pdir / "friction_ellipse.png")
        plots.save(plots.plot_learned_mu(trained), pdir / "learned_mu.png")
        print(f"\n[plots] written to {pdir}")

    print(f"\n[summary] {summary_path}")
    print(pd.read_csv(summary_path).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
