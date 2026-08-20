"""Deep Dynamics datasets: BayesRace (simulated) and Indy Autonomous Challenge (real).

**Type: mixed — BayesRace is SIMULATED, the IAC logs are REAL measurement.**
The distinction is preserved in the ``source`` column (``deep_dynamics_bayesrace`` vs
``deep_dynamics_iac``) so a simulated result can never be reported as a measured one.

Reference: Chrosniak, Ning, Behl, *Deep Dynamics: Vehicle Dynamics Modeling with a
Physics-Constrained Neural Network for Autonomous Racing*, IEEE RA-L 2024
(arXiv:2312.04374). Code and data: https://github.com/linklab-uva/deep-dynamics

These are **vehicle-level** logs, so they load into the vehicle schema of
:py:mod:`tire_nn.data.vehicle`, not the tire schema — there are no measured tire
forces. They are the input to Experiment 3.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tire_nn.data.adapters import DEG2RAD, ColumnSpec, DatasetNotAvailable, map_columns, read_any
from tire_nn.data.vehicle import validate_vehicle_schema

__all__ = ["DEEP_DYNAMICS_COLUMNS", "load_deep_dynamics"]

INSTRUCTIONS = """
Manual download (see scripts/download_deep_dynamics.py):

  git clone https://github.com/linklab-uva/deep-dynamics
  copy the dataset files into  data/raw/deep_dynamics/{bayesrace,iac}/

Label which subset you loaded: BayesRace is SIMULATED, the IAC logs are REAL.
""".strip()

DEEP_DYNAMICS_COLUMNS: dict[str, ColumnSpec] = {
    "t": ColumnSpec(("t", "time", "timestamp"), 1.0),
    "vx": ColumnSpec(("vx", "v_x", "vel_x", "u"), 1.0, required=True),
    "vy": ColumnSpec(("vy", "v_y", "vel_y", "v"), 1.0, required=True),
    "r": ColumnSpec(("r", "yaw_rate", "omega_z", "w"), 1.0, required=True),
    "ax": ColumnSpec(("ax", "a_x", "accel_x"), 1.0),
    "ay": ColumnSpec(("ay", "a_y", "accel_y"), 1.0),
    "r_dot": ColumnSpec(("r_dot", "yaw_accel", "dr"), 1.0),
    "delta": ColumnSpec(("delta", "steering", "steering_angle", "steer"), 1.0, required=True),
    "throttle": ColumnSpec(("throttle", "T", "drive"), 1.0),
}


def load_deep_dynamics(
    root: str | Path = "data/raw",
    subdir: str = "deep_dynamics",
    subset: str = "bayesrace",
    columns: dict[str, ColumnSpec] | None = None,
    R_e: float = 0.3,
    steering_in_degrees: bool = False,
) -> pd.DataFrame:
    """Load Deep Dynamics logs into the vehicle-level schema.

    Wheel speeds are frequently absent from these logs. When they are, they are
    **reconstructed** as free-rolling (``omega_i = v_{x,i}/R_e``) and a warning is
    printed: with free-rolling wheel speeds the longitudinal slip is identically zero,
    so only the lateral part of the tire model is identifiable from that subset. This
    is stated rather than hidden, because a silently zero ``kappa`` would look like a
    successful fit with no longitudinal information in it at all.
    """
    if subset not in ("bayesrace", "iac"):
        raise ValueError("subset must be 'bayesrace' (simulated) or 'iac' (real)")
    path = Path(root) / subdir / subset
    raw = read_any(path, ("*.csv", "*.npz", "*.parquet"), f"Deep Dynamics [{subset}]", INSTRUCTIONS)
    cols = dict(columns or DEEP_DYNAMICS_COLUMNS)
    if steering_in_degrees:
        cols["delta"] = ColumnSpec(cols["delta"].candidates, DEG2RAD, required=True)
    df = map_columns(raw, cols)

    if "t" not in df.columns:
        df["t"] = np.arange(len(df)) * 0.04
    dt = float(np.median(np.diff(df["t"]))) if len(df) > 1 else 0.04
    for name, series in (("ax", "vx"), ("ay", "vy"), ("r_dot", "r")):
        if name not in df.columns:
            df[name] = np.gradient(df[series].to_numpy(), dt)
            print(f"[deep_dynamics] {name} not in the log — differentiated {series} at dt={dt:.4g}s")

    wheel_cols = [c for c in raw.columns if c.lower().startswith("omega")]
    if len(wheel_cols) >= 4:
        for target, source in zip(("omega_FL", "omega_FR", "omega_RL", "omega_RR"), wheel_cols[:4]):
            df[target] = pd.to_numeric(raw[source], errors="coerce")
    else:
        print("[deep_dynamics] no wheel speeds in this log — reconstructing free-rolling "
              f"omega = vx/R_e (R_e={R_e} m). Longitudinal slip will be ~0, so only the "
              "lateral tire behaviour is identifiable from this subset.")
        for target in ("omega_FL", "omega_FR", "omega_RL", "omega_RR"):
            df[target] = df["vx"] / R_e

    df["sequence_id"] = pd.factorize(raw["__file"])[0]
    df["tire_id"] = subset
    df["source"] = f"deep_dynamics_{subset}"
    df["data_type"] = "simulated" if subset == "bayesrace" else "real measurement"
    return validate_vehicle_schema(df.dropna(subset=["vx", "vy", "r", "delta"]).reset_index(drop=True))
