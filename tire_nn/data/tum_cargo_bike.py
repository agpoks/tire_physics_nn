"""TUM cargo bicycle tire dataset (PLAN.md §4.4, source 3).

**Type: real measurement — SOURCE UNVERIFIED.**

At the time of writing no primary source for this dataset was confirmed. The adapter is
provided so that, once the release is located, only the column map has to be filled in.
Until then it raises with instructions rather than silently loading something else.

Note that the closely related cargo-bicycle *lateral* measurements published with the
VeTyT rig (see :py:mod:`tire_nn.data.vetyt`) are verified and may be the better
starting point.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tire_nn.data.adapters import DEG2RAD, BAR2PA, ColumnSpec, finalise, map_columns, read_any

__all__ = ["TUM_COLUMNS", "load_tum_cargo_bike"]

INSTRUCTIONS = """
UNVERIFIED SOURCE. Before using this adapter:

  1. locate the actual release (mediaTUM / TUM library / the publication's
     supplementary material) and record the URL and licence in PLAN.md 4.4
     and papers/references.bib
  2. place the tables under  data/raw/tum_cargo_bike/
  3. check the sign convention on one pure-lateral sweep and set flip_signs accordingly
  4. adjust TUM_COLUMNS to the real headers

Do not report any quantitative result from this source until step 1 is done.
""".strip()

TUM_COLUMNS: dict[str, ColumnSpec] = {
    "alpha": ColumnSpec(("alpha", "slip_angle", "sideslip_angle"), DEG2RAD, required=True),
    "kappa": ColumnSpec(("kappa", "slip_ratio", "longitudinal_slip"), 1.0, default=0.0),
    "Fz": ColumnSpec(("Fz", "vertical_load", "normal_load"), 1.0, required=True),
    "Fx": ColumnSpec(("Fx", "longitudinal_force"), 1.0),
    "Fy": ColumnSpec(("Fy", "lateral_force"), 1.0),
    "Mz": ColumnSpec(("Mz", "aligning_moment"), 1.0),
    "p": ColumnSpec(("p", "pressure", "inflation_pressure_bar"), BAR2PA),
    "gamma": ColumnSpec(("gamma", "camber"), DEG2RAD, default=0.0),
    "vx": ColumnSpec(("v", "vx", "speed"), 1.0),
}


def load_tum_cargo_bike(
    root: str | Path = "data/raw",
    subdir: str = "tum_cargo_bike",
    columns: dict[str, ColumnSpec] | None = None,
    targets: tuple[str, ...] = ("Fy",),
    flip_signs: bool = False,
) -> pd.DataFrame:
    root = Path(root) / subdir
    raw = read_any(root, ("*.csv", "*.txt", "*.xlsx", "*.parquet"), "TUM cargo bike", INSTRUCTIONS)
    df = map_columns(raw, columns or TUM_COLUMNS)
    if flip_signs:
        from tire_nn.data.common import flip_sign_convention
        df = flip_sign_convention(df)
    return finalise(df, "tum_cargo_bike", targets, tire_id="tum_cargo")
