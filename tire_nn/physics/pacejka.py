"""Differentiable Magic Formula (Pacejka) tire model.

Pure physics: no learnable parameters live in this module. Parameters are passed
in explicitly, which is what lets ``models/parameter_tire.py`` predict them with a
network and ``scripts/fit_magic_formula.py`` fit them with ``scipy`` while both go
through *the same* equations (PLAN.md §2.1).

Sign convention (PLAN.md §4.1, SAE-style):
    positive slip angle  ``alpha``  -> negative lateral force ``Fy``
    positive slip ratio  ``kappa``  -> positive (driving) longitudinal force ``Fx``

Reference implementation cross-checked against the numba full-Pacejka model in
``scuderia_gymnasium/gym/scuderia_gym/envs/dynamic_models.py`` (not imported: it is
numba-jitted NumPy and not autograd-compatible).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
from torch import Tensor

from tire_nn.types import TireForces

__all__ = [
    "MFParams",
    "magic_formula",
    "load_sensitive_mu",
    "pacejka_lateral",
    "pacejka_longitudinal",
    "pacejka_combined",
    "cornering_stiffness",
    "MagicFormulaTire",
]

EPS = 1e-9


@dataclass
class MFParams:
    """Simplified Magic Formula parameter set (per axis).

    ``D`` is not stored directly: the peak force is ``mu(Fz) * Fz`` so that load
    sensitivity stays physical and ``mu`` remains readable/comparable across models.
    """

    B: float | Tensor = 10.0        # stiffness factor       [-]
    C: float | Tensor = 1.6         # shape factor           [-]
    E: float | Tensor = 0.5         # curvature factor       [-]
    mu: float | Tensor = 1.0        # peak friction at Fz0   [-]
    k_mu: float | Tensor = 0.05     # load sensitivity       [-]
    Fz0: float | Tensor = 1000.0    # nominal load           [N]
    Sh: float | Tensor = 0.0        # horizontal shift       [rad] or [-]
    Sv: float | Tensor = 0.0        # vertical shift         [N]

    def as_dict(self) -> dict:
        return asdict(self)


def magic_formula(
    x: Tensor,
    B: Tensor | float,
    C: Tensor | float,
    D: Tensor | float,
    E: Tensor | float,
    Sh: Tensor | float = 0.0,
    Sv: Tensor | float = 0.0,
) -> Tensor:
    """Sine-form Magic Formula ``D sin(C atan(Bx - E(Bx - atan(Bx)))) + Sv``."""
    xs = x + Sh
    Bx = B * xs
    return D * torch.sin(C * torch.atan(Bx - E * (Bx - torch.atan(Bx)))) + Sv


def load_sensitive_mu(
    Fz: Tensor,
    mu: Tensor | float,
    k_mu: Tensor | float = 0.05,
    Fz0: Tensor | float = 1000.0,
    mu_min: float = 1e-3,
) -> Tensor:
    """Decreasing peak friction with load: ``mu(Fz) = mu0 (1 - k_mu (Fz/Fz0 - 1))``.

    Load sensitivity is a first-order fact about real tires (the friction
    coefficient drops as the contact patch is loaded up). Encoding it here keeps
    ``D`` from floating freely and keeps ``mu`` interpretable.
    """
    ratio = Fz / Fz0
    return torch.clamp(mu * (1.0 - k_mu * (ratio - 1.0)), min=mu_min)


def pacejka_lateral(alpha: Tensor, Fz: Tensor, p: MFParams) -> tuple[Tensor, Tensor]:
    """Pure-slip lateral force. Returns ``(Fy, mu_y)``; ``Fy(0) = Sv``, negative for ``alpha > 0``."""
    mu_y = load_sensitive_mu(Fz, p.mu, p.k_mu, p.Fz0)
    D = mu_y * Fz
    return -magic_formula(alpha, p.B, p.C, D, p.E, p.Sh, p.Sv), mu_y


def pacejka_longitudinal(kappa: Tensor, Fz: Tensor, p: MFParams) -> tuple[Tensor, Tensor]:
    """Pure-slip longitudinal force. Returns ``(Fx, mu_x)``; positive for ``kappa > 0``."""
    mu_x = load_sensitive_mu(Fz, p.mu, p.k_mu, p.Fz0)
    D = mu_x * Fz
    return magic_formula(kappa, p.B, p.C, D, p.E, p.Sh, p.Sv), mu_x


def pacejka_combined(
    alpha: Tensor,
    kappa: Tensor,
    Fz: Tensor,
    px: MFParams,
    py: MFParams,
    theoretical_slip: bool = False,
) -> tuple[Tensor, Tensor]:
    """Combined slip by the *similarity* (normalised slip vector) method.

    The slip vector ``(sigma_x, sigma_y)`` is what the brush model says the
    contact-patch shear follows; the resulting force is aligned with it and has
    magnitude given by the pure-slip curve evaluated at ``|sigma|``. This gives a
    friction-ellipse-shaped combined response without the ~20 extra weighting
    coefficients of full MF 6.x, and it keeps the force direction opposite to the slip
    direction (dissipativity).

    ``theoretical_slip`` selects the normalisation:

    * ``False`` (default) — ``(kappa, tan alpha)``. **Exactly odd** in ``(alpha, kappa)``,
      so the P2 symmetry guarantee also holds for the analytical and ParameterNet
      rungs of the ladder, making the ablation comparison clean.
    * ``True`` — ``(kappa, tan alpha) / (1 + kappa)``, the textbook theoretical slip.
      More accurate at large *braking* slip, but note that it is **not** odd in
      ``kappa``: the practical slip ratio is itself an asymmetric definition (driving
      slip is unbounded above, braking slip is bounded by -1). That asymmetry is
      kinematic, not constitutive — it lives in the definition of ``kappa``, not in
      the tire's constitutive law — which is why the symmetric form is the default
      here and the asymmetric one is opt-in for high-fidelity braking work.
    """
    denom = 1.0 + torch.clamp(kappa, min=-0.99) if theoretical_slip else 1.0
    sx = kappa / denom
    sy = torch.tan(alpha) / denom
    s = torch.sqrt(sx * sx + sy * sy + EPS)

    # Pure-slip magnitudes evaluated at the combined slip magnitude |sigma| >= 0.
    Fx_mag = pacejka_longitudinal(s, Fz, px)[0].abs()
    Fy_mag = pacejka_lateral(s, Fz, py)[0].abs()

    # Project onto the slip direction. The minus sign on Fy is the SAE convention:
    # positive slip angle -> sy > 0 -> Fy < 0.
    Fx = (sx / s) * Fx_mag
    Fy = -(sy / s) * Fy_mag
    return Fx, Fy


def cornering_stiffness(p: MFParams, Fz: Tensor) -> Tensor:
    """``C_alpha = dFy/dalpha`` at zero slip ``= B*C*D`` [N/rad]."""
    mu = load_sensitive_mu(Fz, p.mu, p.k_mu, p.Fz0)
    return p.B * p.C * mu * Fz


class MagicFormulaTire(torch.nn.Module):
    """Analytical baseline with the same call signature as the learned models.

    Parameters are registered as **buffers**, not ``Parameter``s: this model is the
    fixed analytical reference for the ablation ladder. Fit it with
    ``scripts/fit_magic_formula.py`` (scipy) and load the result.
    """

    def __init__(
        self,
        px: MFParams | None = None,
        py: MFParams | None = None,
        combined: bool = True,
        theoretical_slip: bool = False,
    ):
        super().__init__()
        px = px or MFParams()
        py = py or MFParams()
        self.combined = combined
        self.theoretical_slip = theoretical_slip
        for tag, p in (("x", px), ("y", py)):
            for k, v in p.as_dict().items():
                self.register_buffer(f"{tag}_{k}", torch.as_tensor(float(v)))

    def _params(self, tag: str) -> MFParams:
        return MFParams(**{k: getattr(self, f"{tag}_{k}") for k in MFParams().as_dict()})

    def forward(self, alpha: Tensor, kappa: Tensor, Fz: Tensor, context: dict | None = None) -> TireForces:
        """Returns :class:`~tire_nn.types.TireForces`, which also unpacks as ``(Fx, Fy)``."""
        px, py = self._params("x"), self._params("y")
        if self.combined:
            Fx, Fy = pacejka_combined(alpha, kappa, Fz, px, py, self.theoretical_slip)
        else:
            Fx, _ = pacejka_longitudinal(kappa, Fz, px)
            Fy, _ = pacejka_lateral(alpha, Fz, py)
        mu_x = load_sensitive_mu(Fz, px.mu, px.k_mu, px.Fz0)
        mu_y = load_sensitive_mu(Fz, py.mu, py.k_mu, py.Fz0)
        return TireForces(Fx=Fx, Fy=Fy, params={"mu_x": mu_x, "mu_y": mu_y,
                                                "C_alpha": cornering_stiffness(py, Fz),
                                                "C_kappa": cornering_stiffness(px, Fz)})
