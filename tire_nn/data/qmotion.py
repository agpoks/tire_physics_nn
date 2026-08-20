"""Q-Motion tire dataset with inflation-pressure variation (PLAN.md §4.4, source 6).

**Type: real measurement — SOURCE UNVERIFIED.**

Used here for the **context / generalisation** study: the pressure column becomes a
context input, entire pressure levels are held out with
:py:func:`tire_nn.data.common.split_by_condition`, and the question is whether the
encoded models transfer to a pressure they never saw.

Inflation pressure has a well-documented, structured effect — higher pressure stiffens
the carcass (raising cornering stiffness) while shrinking the contact patch (slightly
lowering peak friction) {cite}`besselink2010magicformula`. A model that has the right
structure should need very little data to pick that up; a black box needs a sweep.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tire_nn.data.adapters import BAR2PA, DEG2RAD, KPA2PA, PSI2PA, ColumnSpec, finalise, map_columns, read_any
from tire_nn.data.common import flip_sign_convention

__all__ = ["QMOTION_COLUMNS", "PRESSURE_UNITS", "load_qmotion"]

INSTRUCTIONS = """
UNVERIFIED SOURCE. Confirm the exact release and licence, record them in PLAN.md 4.4
and papers/references.bib, then place the tables under  data/raw/qmotion/  and re-run.

Check the pressure unit before loading: pass pressure_unit="kPa" | "bar" | "psi" | "Pa".
A wrong pressure unit is a silent error — the model will simply learn nothing from a
context channel that is off by a factor of 100.
""".strip()

PRESSURE_UNITS = {"Pa": 1.0, "kPa": KPA2PA, "bar": BAR2PA, "psi": PSI2PA}

QMOTION_COLUMNS: dict[str, ColumnSpec] = {
    "alpha": ColumnSpec(("alpha", "slip_angle", "SA"), DEG2RAD, required=True),
    "kappa": ColumnSpec(("kappa", "slip_ratio", "SR"), 1.0, default=0.0),
    "Fz": ColumnSpec(("Fz", "vertical_load", "normal_load"), 1.0, required=True),
    "Fx": ColumnSpec(("Fx", "longitudinal_force"), 1.0),
    "Fy": ColumnSpec(("Fy", "lateral_force"), 1.0),
    "Mz": ColumnSpec(("Mz", "aligning_moment"), 1.0),
    "gamma": ColumnSpec(("gamma", "camber", "IA"), DEG2RAD, default=0.0),
    "vx": ColumnSpec(("v", "vx", "speed"), 1.0),
    "Ts": ColumnSpec(("T", "temperature", "tire_temp"), 1.0, 273.15),
}


def load_qmotion(
    root: str | Path = "data/raw",
    subdir: str = "qmotion",
    columns: dict[str, ColumnSpec] | None = None,
    targets: tuple[str, ...] = ("Fy",),
    pressure_unit: str = "kPa",
    flip_signs: bool = False,
) -> pd.DataFrame:
    """Load Q-Motion measurements, converting pressure explicitly."""
    if pressure_unit not in PRESSURE_UNITS:
        raise ValueError(f"pressure_unit must be one of {sorted(PRESSURE_UNITS)}")
    root = Path(root) / subdir
    raw = read_any(root, ("*.csv", "*.txt", "*.xlsx", "*.parquet"), "Q-Motion", INSTRUCTIONS)
    cols = dict(columns or QMOTION_COLUMNS)
    cols["p"] = ColumnSpec(("p", "pressure", "inflation_pressure"), PRESSURE_UNITS[pressure_unit],
                           required=True)
    df = map_columns(raw, cols)
    if flip_signs:
        df = flip_sign_convention(df)
    # One tire_id per pressure level makes a per-condition holdout trivial.
    df["tire_id"] = "qmotion_p" + (df["p"] / 1000.0).round(0).astype(int).astype(str) + "kPa"
    return finalise(df, "qmotion", targets)
