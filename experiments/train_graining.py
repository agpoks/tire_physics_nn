#!/usr/bin/env python3
"""Experiment 4 — thermal / wear / graining demonstrator (PLAN.md §5).

SYNTHETIC AND WEAKLY SUPERVISED. Demonstrates that the encoded condition structure
reproduces cold-start graining formation, growth under high slip energy, clean-up on
warm-up and irreversible wear accumulation — and that the structural guarantees
(``dwear/dt >= 0``, ``g`` in ``[0, 1]``) hold throughout training.

This experiment does **not** claim to represent validated real motorsport graining.

    python experiments/train_graining.py --config configs/exp4_graining.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tire_nn.data.graining import make_synthetic_graining  # noqa: E402
from tire_nn.evaluation import plots  # noqa: E402
from tire_nn.models import build_model  # noqa: E402
from tire_nn.models.thermo_graining_tire import ThermoGrainingTire  # noqa: E402
from tire_nn.training import append_summary, set_seed  # noqa: E402


def windows(df, size: int, stride: int):
    for start in range(0, len(df) - size + 1, stride):
        yield df.iloc[start:start + size]


def to_tensors(chunk):
    t = lambda c: torch.as_tensor(chunk[c].to_numpy(), dtype=torch.float32).unsqueeze(0)
    return t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/exp4_graining.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    for item in args.set:
        key, _, value = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)

    set_seed(cfg["seed"])
    out_root = ROOT / cfg["out_dir"]
    out_root.mkdir(parents=True, exist_ok=True)
    d, tr = cfg["data"], cfg["training"]

    df = make_synthetic_graining(T=d["T"], dt=d["dt"], vx=d["vx"], Fz=d["Fz"], mu=d["mu"],
                                 noise_state=d["noise_state"], seed=cfg["seed"])
    print(f"[data] SYNTHETIC weakly-supervised stint: {len(df)} samples, {len(df)*d['dt']:.0f} s")
    print(f"[data] reference graining peaks at {df['graining'].max():.2f}, "
          f"ends at {df['graining'].iloc[-1]:.2f}; wear reaches {df['wear'].iloc[-1]:.4f}")

    mcfg = dict(cfg["model"])
    steady = build_model(mcfg.pop("steady", "encoded"), context_keys=("vx",))
    model = ThermoGrainingTire(steady, **mcfg)

    opt = torch.optim.Adam(model.parameters(), lr=tr["lr"])
    w = tr["weights"]
    size = d["window"]
    chunks = list(windows(df, size, size))
    print(f"[train] {len(chunks)} windows of {size} samples")

    history = []
    for epoch in range(tr["epochs"]):
        total = 0.0
        for chunk in chunks:
            t = to_tensors(chunk)
            opt.zero_grad(set_to_none=True)
            F, z, _ = model.rollout_condition(
                t("alpha"), t("kappa"), t("Fz"), t("vx"), d["dt"],
                T_road=float(chunk["T_road"].iloc[0]), T_air=float(chunk["T_air"].iloc[0]),
                z0=torch.tensor([[float(chunk["Ts"].iloc[0]), float(chunk["Tc"].iloc[0]),
                                  float(chunk["wear"].iloc[0]), float(chunk["graining"].iloc[0])]]))
            loss = w["force"] * (((F[..., 0] - t("Fx")) ** 2 + (F[..., 1] - t("Fy")) ** 2).mean()
                                 / float(d["Fz"]) ** 2)
            loss = loss + w["Ts"] * ((z[..., 0] - t("Ts_meas")) ** 2).mean() / 100.0
            loss = loss + w["graining"] * ((z[..., 3] - t("graining_meas")) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            total += float(loss.detach())
        history.append(total / len(chunks))
        if epoch % 10 == 0 or epoch == tr["epochs"] - 1:
            print(f"  epoch {epoch:3d}  loss {history[-1]:.6g}")

    # --- full-stint rollout from a cold, unworn, clean tire --------------------
    model.eval()
    with torch.no_grad():
        t = to_tensors(df)
        F, z, extra = model.rollout_condition(
            t("alpha"), t("kappa"), t("Fz"), t("vx"), d["dt"],
            T_road=float(df["T_road"].iloc[0]), T_air=float(df["T_air"].iloc[0]))
    Ts, Tc, wear, g = (z[0, :, i].numpy() for i in range(4))
    time = df["t"].to_numpy()

    # --- structural guarantees, checked on the learned model -------------------
    checks = {
        "wear_monotone": bool(np.all(np.diff(wear) >= -1e-12)),
        "graining_in_unit_interval": bool(g.min() >= 0.0 and g.max() <= 1.0),
        "temperatures_finite": bool(np.isfinite(Ts).all() and np.isfinite(Tc).all()),
        "slip_power_non_negative": bool(float(extra["P_slip"].min()) >= 0.0),
    }
    print("\n[checks] " + "  ".join(f"{k}={v}" for k, v in checks.items()))
    assert all(checks.values()), "a structural guarantee failed — this must never happen"

    # --- demonstrator numbers --------------------------------------------------
    q = len(df) // 4
    row = {
        "experiment": cfg["experiment"], "data": "SYNTHETIC (not validated real graining)",
        "final_loss": history[-1],
        "graining_peak": float(g.max()), "graining_peak_time_s": float(time[int(g.argmax())]),
        "graining_after_warmup": float(g[2 * q - 1]), "graining_final": float(g[-1]),
        "Ts_start_K": float(Ts[0]), "Ts_peak_K": float(Ts.max()),
        "wear_final": float(wear[-1]), "wear_monotone": checks["wear_monotone"],
        "graining_in_unit_interval": checks["graining_in_unit_interval"],
        "ref_graining_peak": float(df["graining"].max()),
        "ref_graining_final": float(df["graining"].iloc[-1]),
        "learned_T_opt_K": float(model.T_opt()), "learned_k_grain": float(model.k_grain()),
        "learned_k_wear": float(model.k_wear()),
    }
    append_summary(out_root / "summary.csv", row)

    import matplotlib
    matplotlib.use("Agg")
    pdir = out_root / "plots"
    plots.save(plots.plot_time_series(time, {"Ts (learned)": Ts, "Tc (learned)": Tc,
                                             "Ts (reference)": df["Ts"], "Tc (reference)": df["Tc"]},
                                      "temperature [K]", "Tire temperature over the stint"),
               pdir / "temperature.png")
    plots.save(plots.plot_time_series(time, {"graining (learned)": g,
                                             "graining (reference)": df["graining"]},
                                      "graining [-]", "Graining (SYNTHETIC demonstrator)"),
               pdir / "graining.png")
    plots.save(plots.plot_time_series(time, {"wear (learned)": wear, "wear (reference)": df["wear"]},
                                      "wear [-]", "Irreversible wear accumulation"),
               pdir / "wear.png")
    plots.save(plots.plot_time_series(time, {"P_slip [kW]": extra["P_slip"][0].numpy() / 1000.0},
                                      "slip power [kW]", "Dissipated slip power"),
               pdir / "slip_power.png")

    print(f"\n[plots] {pdir}")
    print(f"[summary] {out_root / 'summary.csv'}")
    for k, v in row.items():
        print(f"  {k}: {v}")
    print("\nNOTE: synthetic demonstrator — not validated real motorsport graining.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
