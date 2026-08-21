"""FSAE Tire Test Consortium / Calspan tire data (PLAN.md §4.4).

**Type: real measurement.** The most directly useful dataset for this project: raw
force-and-moment measurements from the Calspan Tire Research Facility, covering slip
angle, slip ratio, inclination angle, vertical load, inflation pressure, speed and tire
temperature — which is exactly the input space Experiment 1 is built around. The pool
holds 430+ tests across 40+ tire constructions.

Access is restricted rather than open: membership of the consortium is required
(registration plus a fee, aimed at Formula SAE teams), and the data may not be
redistributed. Nothing here downloads it; the loader reads what a member has already
obtained.

- Consortium: https://www.fsaettc.org/
- Test facility: https://calspan.com/automotive/fsae-ttc
- Method paper: Kasprzak & Gentz, *The Formula SAE Tire Test Consortium — Tire Testing
  and Data Handling*, SAE 2006-01-3606.

Channel names and units
-----------------------
TTC files use a fixed channel set, which is what :data:`TTC_COLUMNS` maps. The units are
**not** SI and are the usual source of error:

``SA``
    slip angle, **degrees**.
``SR``
    slip ratio, dimensionless.
``IA``
    inclination (camber) angle, **degrees**.
``FZ``
    vertical load in N — **negative in several rounds**.
``FX``, ``FY``
    longitudinal and lateral force, N.
``MZ``
    aligning moment, N·m.
``P``
    inflation pressure, **kPa**.
``V``
    road speed, **km/h**.
``TSTC``
    tire surface temperature at the centre, **degrees Celsius**.
``RST``
    road surface temperature, **degrees Celsius**.

.. warning::

   Two conventions must be checked against your own copy before trusting a fit. TTC runs
   often report ``FZ`` as a **negative** number (load pressing down in their axis
   system), and the lateral-force sign follows the axis system of the particular
   processing. This loader takes ``abs(FZ)`` and exposes ``flip_signs``; verify on one
   pure-lateral sweep that positive slip angle gives negative ``Fy`` before running a
   whole campaign.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tire_nn.data.adapters import (
    C2K,
    DEG2RAD,
    KPA2PA,
    ColumnSpec,
    finalise,
    map_columns,
    read_any,
)
from tire_nn.data.common import flip_sign_convention

__all__ = ["TTC_COLUMNS", "load_fsae_ttc"]

KPH2MS = 1.0 / 3.6

INSTRUCTIONS = """
FSAE TTC data is available to consortium members only and may not be redistributed.

  1. Register at https://www.fsaettc.org/ (a fee applies; the consortium is aimed at
     Formula SAE teams, so university affiliation is the usual route).
  2. Download the round data you need from the members' server.
  3. Export the runs to CSV, or convert the supplied .mat files, and place them under
     data/raw/fsae_ttc/ — one file per run is fine, they are concatenated.
  4. Check the sign conventions on one pure-lateral sweep before trusting a campaign:
     positive slip angle must give negative Fy in this project's convention.

Channel names are mapped by TTC_COLUMNS; pass your own map if your export renames them.
""".strip()

#: TTC channel -> canonical column, with the unit conversion into SI.
TTC_COLUMNS: dict[str, ColumnSpec] = {
    "alpha": ColumnSpec(("SA", "slip_angle", "slipangle"), DEG2RAD, required=True),
    "kappa": ColumnSpec(("SR", "slip_ratio", "slipratio"), 1.0, default=0.0),
    "Fz": ColumnSpec(("FZ", "Fz", "vertical_load"), 1.0, required=True),
    "Fx": ColumnSpec(("FX", "Fx"), 1.0),
    "Fy": ColumnSpec(("FY", "Fy"), 1.0),
    "Mz": ColumnSpec(("MZ", "Mz"), 1.0),
    "gamma": ColumnSpec(("IA", "inclination_angle", "camber"), DEG2RAD, default=0.0),
    "p": ColumnSpec(("P", "pressure", "inflation_pressure"), KPA2PA),
    "vx": ColumnSpec(("V", "road_speed", "speed"), KPH2MS),
    "Ts": ColumnSpec(("TSTC", "tire_surface_temp_centre", "TSTI"), 1.0, C2K),
    "T_road": ColumnSpec(("RST", "road_surface_temp"), 1.0, C2K),
    "t": ColumnSpec(("ET", "elapsed_time", "time"), 1.0),
}


def load_fsae_ttc(
    root: str | Path = "data/raw",
    subdir: str = "fsae_ttc",
    columns: dict[str, ColumnSpec] | None = None,
    targets: tuple[str, ...] = ("Fx", "Fy"),
    flip_signs: bool = False,
    tire_id: str | None = None,
) -> pd.DataFrame:
    """Load TTC runs into the canonical schema.

    ``tire_id`` defaults to the source file stem, so a directory of runs across several
    tire constructions becomes a multi-tire dataset automatically — which is what the
    context embedding and the tire-set generalisation studies want.
    """
    root = Path(root) / subdir
    raw = read_any(root, ("*.csv", "*.txt", "*.dat", "*.parquet"), "FSAE TTC", INSTRUCTIONS)
    df = map_columns(raw, columns or TTC_COLUMNS)

    # TTC reports FZ negative in several rounds; the canonical schema is compression
    # positive, and validate_schema would otherwise reject the whole file.
    df["Fz"] = df["Fz"].abs()

    if flip_signs:
        df = flip_sign_convention(df)
    df["tire_id"] = (tire_id if tire_id is not None
                     else raw["__file"].str.replace(r"\.[^.]+$", "", regex=True))
    have = tuple(t for t in targets if t in df.columns and df[t].notna().any())
    if not have:
        raise ValueError(f"none of the target columns {targets} were found in the TTC files")
    return finalise(df, "fsae_ttc", have)
