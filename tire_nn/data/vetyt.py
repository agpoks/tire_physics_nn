"""VeTyT bicycle tyre dataset (PLAN.md §4.4, source 2).

**Type: real measurement.**

VeTyT (Velo Tyre Testing) is an indoor flat-track test-rig for bicycle tyres at the
Department of Mechanical Engineering, Politecnico di Milano. Published measurements
cover lateral force and self-aligning moment at vertical loads of roughly 343–526 N,
camber angles of -5, 0 and +5 degrees and inflation pressures of 300–500 kPa, with an
air-cooling system holding the rolling-surface temperature constant.

References
----------
- Dell'Orto, Ballo, Gobbi, Mastinu, *Measurement of the lateral characteristics and
  identification of the Magic Formula parameters of city and cargo bicycle tyres*,
  Vehicle System Dynamics, 2024. doi:10.1080/00423114.2024.2338143
- Dell'Orto et al., *Bicycle tyres — Development of a new test-rig to measure
  mechanical characteristics*, Measurement, 2022. doi:10.1016/j.measurement.2022.111813

Why this dataset is interesting here
------------------------------------
It is a **pure-lateral** rig with genuine camber and pressure variation, which makes it
the natural test of the context-encoding path ({py:class}`tire_nn.models.base.ContextEncoder`)
and of whether the encoded priors still help when there is no longitudinal data at all
(``kappa`` is set to 0 and only ``Fy``/``Mz`` are supervised).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tire_nn.data.adapters import DEG2RAD, KPA2PA, ColumnSpec, finalise, map_columns, read_any
from tire_nn.data.common import flip_sign_convention

__all__ = ["VETYT_COLUMNS", "load_vetyt"]

INSTRUCTIONS = """
The VeTyT measurements are published with the papers rather than as a self-service
download. Request the data from the authors (Politecnico di Milano) or extract it from
the supplementary material, then place the tables under  data/raw/vetyt/  and re-run.

See scripts/download_vetyt.py and PLAN.md 4.4.
""".strip()

VETYT_COLUMNS: dict[str, ColumnSpec] = {
    "alpha": ColumnSpec(("alpha", "slip_angle", "sideslip", "sideslip_angle", "alpha_deg"),
                        DEG2RAD, required=True),
    "kappa": ColumnSpec(("kappa", "slip_ratio"), 1.0, default=0.0),
    "Fz": ColumnSpec(("Fz", "F_z", "vertical_load", "normal_load"), 1.0, required=True),
    "Fy": ColumnSpec(("Fy", "F_y", "lateral_force"), 1.0),
    "Mz": ColumnSpec(("Mz", "M_z", "self_aligning_moment", "aligning_torque", "SAT"), 1.0),
    "gamma": ColumnSpec(("gamma", "camber", "camber_angle", "camber_deg"), DEG2RAD, default=0.0),
    "p": ColumnSpec(("p", "pressure", "inflation_pressure", "p_kPa"), KPA2PA),
    "vx": ColumnSpec(("v", "vx", "speed", "rolling_speed"), 1.0),
}


def load_vetyt(
    root: str | Path = "data/raw",
    subdir: str = "vetyt",
    columns: dict[str, ColumnSpec] | None = None,
    targets: tuple[str, ...] = ("Fy",),
    flip_signs: bool = False,
    tire_column: str | None = "tyre",
) -> pd.DataFrame:
    """Load VeTyT measurements into the canonical schema.

    ``tire_column`` names the source column holding the tyre identifier (city / cargo /
    model name). It becomes ``tire_id`` and hence the context embedding index, which is
    what makes a per-tyre comparison possible.
    """
    root = Path(root) / subdir
    raw = read_any(root, ("*.csv", "*.txt", "*.xlsx", "*.parquet"), "VeTyT", INSTRUCTIONS)
    df = map_columns(raw, columns or VETYT_COLUMNS)
    if tire_column and tire_column in raw.columns:
        df["tire_id"] = raw[tire_column].astype(str)
    else:
        df["tire_id"] = raw["__file"].str.replace(r"\.[^.]+$", "", regex=True)
    if flip_signs:
        df = flip_sign_convention(df)
    return finalise(df, "vetyt", targets)
