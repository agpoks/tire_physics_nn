"""RoboRacer / F1TENTH model-structured NN dataset (PLAN.md §4.4, source 5).

**Type: real measurement (small scale) — SOURCE UNVERIFIED.**

The interesting property of this source for this project is that it includes
**tire-set** and **mass-change** experiments: the same car, the same track, a different
tire compound or a different mass. That is exactly the generalisation axis Experiment 3
tests, and it is what the ``tire_id`` context embedding exists for.

Small-scale vehicles have short relaxation lengths (centimetres rather than decimetres),
so this is also a useful check that the relaxation model of
:py:mod:`tire_nn.models.relaxation_tire` identifies a *scale*, not a memorised constant.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tire_nn.data.adapters import ColumnSpec, map_columns, read_any
from tire_nn.data.vehicle import validate_vehicle_schema

__all__ = ["ROBORACER_COLUMNS", "load_roboracer"]

INSTRUCTIONS = """
UNVERIFIED SOURCE. Before using this adapter, confirm the exact release and licence,
record them in PLAN.md 4.4 and papers/references.bib, then place the logs under

    data/raw/roboracer/<experiment_name>/*.csv

One subdirectory per experiment (tire set, mass) — the directory name becomes tire_id,
which is what makes the tire-set comparison possible.
""".strip()

ROBORACER_COLUMNS: dict[str, ColumnSpec] = {
    "t": ColumnSpec(("t", "time", "stamp", "timestamp"), 1.0),
    "vx": ColumnSpec(("vx", "v_x", "linear_x", "speed"), 1.0, required=True),
    "vy": ColumnSpec(("vy", "v_y", "linear_y"), 1.0, default=0.0),
    "r": ColumnSpec(("r", "yaw_rate", "angular_z", "omega"), 1.0, required=True),
    "ax": ColumnSpec(("ax", "accel_x", "imu_ax", "linear_acceleration_x"), 1.0),
    "ay": ColumnSpec(("ay", "accel_y", "imu_ay", "linear_acceleration_y"), 1.0),
    "delta": ColumnSpec(("delta", "steering_angle", "steer", "servo"), 1.0, required=True),
}


def load_roboracer(
    root: str | Path = "data/raw",
    subdir: str = "roboracer",
    columns: dict[str, ColumnSpec] | None = None,
    R_e: float = 0.05,
    dt_default: float = 0.02,
) -> pd.DataFrame:
    """Load RoboRacer logs into the vehicle-level schema, one ``tire_id`` per subdirectory."""
    root = Path(root) / subdir
    raw = read_any(root, ("*.csv", "*.parquet"), "RoboRacer", INSTRUCTIONS)
    df = map_columns(raw, columns or ROBORACER_COLUMNS)

    if "t" not in df.columns:
        df["t"] = np.arange(len(df)) * dt_default
    dt = float(np.median(np.diff(df["t"]))) if len(df) > 1 else dt_default
    for name, series in (("ax", "vx"), ("ay", "vy")):
        if name not in df.columns:
            df[name] = np.gradient(df[series].to_numpy(), dt)
    df["r_dot"] = np.gradient(df["r"].to_numpy(), dt)

    wheel_cols = [c for c in raw.columns if c.lower().startswith(("omega", "wheel"))]
    if len(wheel_cols) >= 4:
        for target, source in zip(("omega_FL", "omega_FR", "omega_RL", "omega_RR"), wheel_cols[:4]):
            df[target] = pd.to_numeric(raw[source], errors="coerce")
    else:
        print("[roboracer] no per-wheel speeds — reconstructing free-rolling omega = vx/R_e; "
              "longitudinal slip is then ~0 and only lateral behaviour is identifiable.")
        for target in ("omega_FL", "omega_FR", "omega_RL", "omega_RR"):
            df[target] = df["vx"] / R_e

    # The directory name identifies the experiment (tire set / mass) -> tire_id.
    files = raw["__file"]
    df["sequence_id"] = pd.factorize(files)[0]
    df["tire_id"] = [Path(f).parent.name or "roboracer" for f in files]
    df["source"] = "roboracer"
    return validate_vehicle_schema(df.dropna(subset=["vx", "r", "delta"]).reset_index(drop=True))
