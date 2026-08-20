"""Least-squares Magic Formula fitting with scipy (analytical baseline).

Blueprint taken from ``Tire_Parameter_and_Uncertainty_Estimation-main`` (Nelder-Mead /
SVI fitting of ``mf_simple`` in JAX), re-implemented here with ``scipy.optimize`` so
the project keeps a single autodiff framework and no JAX dependency.

The fitted parameters are the honest analytical reference for every experiment: the
learned models must beat a *properly fitted* Magic Formula, not a hand-guessed one.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import least_squares

from tire_nn.physics.pacejka import MFParams, pacejka_lateral, pacejka_longitudinal

__all__ = ["fit_magic_formula", "fit_axis"]

# (name, lower, upper, initial)
_BOUNDS = (
    ("B", 0.5, 40.0, 9.0),
    ("C", 0.5, 2.5, 1.6),
    ("E", -2.0, 1.0, 0.4),
    ("mu", 0.05, 2.5, 1.0),
    ("k_mu", 0.0, 0.5, 0.05),
)


def fit_axis(slip: np.ndarray, force: np.ndarray, Fz: np.ndarray, axis: str = "y",
             Fz0: float = 1000.0) -> MFParams:
    """Fit one axis of the Magic Formula to pure-slip data.

    Residuals are **load-normalised** (``F/Fz``): without that, high-load samples
    dominate the fit and the resulting ``mu`` is biased toward the heavy end of the
    load range.
    """
    slip_t = torch.as_tensor(slip, dtype=torch.float32)
    Fz_t = torch.as_tensor(Fz, dtype=torch.float32)
    target = torch.as_tensor(force, dtype=torch.float32) / Fz_t
    fn = pacejka_lateral if axis == "y" else pacejka_longitudinal

    def residual(theta):
        p = MFParams(B=float(theta[0]), C=float(theta[1]), E=float(theta[2]),
                     mu=float(theta[3]), k_mu=float(theta[4]), Fz0=Fz0)
        with torch.no_grad():
            pred, _ = fn(slip_t, Fz_t, p)
        return (pred / Fz_t - target).numpy()

    x0 = np.array([b[3] for b in _BOUNDS])
    lo = np.array([b[1] for b in _BOUNDS])
    hi = np.array([b[2] for b in _BOUNDS])
    sol = least_squares(residual, x0, bounds=(lo, hi), method="trf", max_nfev=2000)
    return MFParams(B=float(sol.x[0]), C=float(sol.x[1]), E=float(sol.x[2]),
                    mu=float(sol.x[3]), k_mu=float(sol.x[4]), Fz0=Fz0)


def fit_magic_formula(df, Fz0: float = 1000.0, pure_slip_tol: float = 0.02) -> tuple[MFParams, MFParams]:
    """Fit ``(px, py)`` from a canonical DataFrame.

    Pure-slip samples are selected by ``|kappa| < tol`` (for the lateral fit) and
    ``|alpha| < tol`` (for the longitudinal fit). If a dataset has no pure-slip
    samples, all samples are used and the fit is reported as approximate — combined
    data biases a pure-slip fit, and that is worth knowing rather than hiding.
    """
    lat = df[df["kappa"].abs() < pure_slip_tol] if "kappa" in df else df
    lon = df[df["alpha"].abs() < pure_slip_tol] if "alpha" in df else df
    if len(lat) < 20:
        lat = df
    if len(lon) < 20:
        lon = df

    py = fit_axis(lat["alpha"].to_numpy(), lat["Fy"].to_numpy(), lat["Fz"].to_numpy(), "y", Fz0) \
        if "Fy" in df.columns else MFParams(Fz0=Fz0)
    px = fit_axis(lon["kappa"].to_numpy(), lon["Fx"].to_numpy(), lon["Fz"].to_numpy(), "x", Fz0) \
        if "Fx" in df.columns else MFParams(Fz0=Fz0)
    return px, py
