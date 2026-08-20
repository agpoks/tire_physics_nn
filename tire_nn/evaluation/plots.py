"""Standard plot set (PLAN.md §7). Matplotlib only, no styling dependencies."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

__all__ = [
    "plot_lateral_curve",
    "plot_longitudinal_curve",
    "plot_friction_ellipse",
    "plot_residuals",
    "plot_learned_mu",
    "plot_time_series",
    "save",
]


def _fig(figsize=(5.0, 3.6)):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(alpha=0.3)
    return fig, ax


def save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    return path


@torch.no_grad()
def plot_lateral_curve(models: dict, Fz: float = 1000.0, alpha_max: float = 0.35, data=None, context=None):
    """``Fy`` vs ``alpha`` at fixed load, all models overlaid, optional scatter of data."""
    fig, ax = _fig()
    a = torch.linspace(-alpha_max, alpha_max, 401)
    z = torch.zeros_like(a)
    fz = torch.full_like(a, Fz)
    if data is not None:
        ax.scatter(data["alpha"], data["Fy"], s=4, alpha=0.25, color="0.5", label="data")
    for name, model in models.items():
        out = model(a, z, fz, context)
        ax.plot(a, out.Fy.detach(), label=name)
    ax.set_xlabel(r"slip angle $\alpha$ [rad]")
    ax.set_ylabel(r"$F_y$ [N]")
    ax.set_title(f"Lateral force at $F_z$={Fz:.0f} N")
    ax.legend(fontsize=7)
    return fig


@torch.no_grad()
def plot_longitudinal_curve(models: dict, Fz: float = 1000.0, kappa_max: float = 0.35, data=None, context=None):
    fig, ax = _fig()
    k = torch.linspace(-kappa_max, kappa_max, 401)
    z = torch.zeros_like(k)
    fz = torch.full_like(k, Fz)
    if data is not None:
        ax.scatter(data["kappa"], data["Fx"], s=4, alpha=0.25, color="0.5", label="data")
    for name, model in models.items():
        ax.plot(k, model(z, k, fz, context).Fx.detach(), label=name)
    ax.set_xlabel(r"slip ratio $\kappa$ [-]")
    ax.set_ylabel(r"$F_x$ [N]")
    ax.set_title(f"Longitudinal force at $F_z$={Fz:.0f} N")
    ax.legend(fontsize=7)
    return fig


@torch.no_grad()
def plot_friction_ellipse(models: dict, Fz: float = 1000.0, mu: float = 1.1, n: int = 60, context=None):
    """Combined-slip force locus. The reference ellipse is the physical bound."""
    fig, ax = _fig(figsize=(4.4, 4.2))
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(mu * Fz * np.cos(th), mu * Fz * np.sin(th), "k--", lw=1, label=f"$\\mu F_z$ ($\\mu$={mu})")
    a = torch.linspace(-0.4, 0.4, n).repeat_interleave(n)
    k = torch.linspace(-0.4, 0.4, n).repeat(n)
    fz = torch.full_like(a, Fz)
    for name, model in models.items():
        out = model(a, k, fz, context)
        ax.scatter(out.Fx.detach(), out.Fy.detach(), s=2, alpha=0.4, label=name)
    ax.set_xlabel("$F_x$ [N]")
    ax.set_ylabel("$F_y$ [N]")
    ax.set_aspect("equal")
    ax.set_title("Combined-slip locus")
    ax.legend(fontsize=7)
    return fig


def plot_residuals(pred, target, x=None, xlabel="index", ylabel="residual [N]"):
    fig, ax = _fig()
    res = np.asarray(pred) - np.asarray(target)
    ax.scatter(np.arange(len(res)) if x is None else np.asarray(x), res, s=4, alpha=0.4)
    ax.axhline(0.0, color="k", lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Residuals (RMSE={np.sqrt((res**2).mean()):.1f} N)")
    return fig


@torch.no_grad()
def plot_learned_mu(models: dict, Fz_range=(200.0, 3000.0), context=None):
    """Learned peak friction vs load — the load-sensitivity curve, directly readable."""
    fig, ax = _fig()
    fz = torch.linspace(*Fz_range, 200)
    z = torch.zeros_like(fz)
    for name, model in models.items():
        out = model(z + 0.05, z, fz, context)
        if "mu_y" in out.params:
            ax.plot(fz, out.params["mu_y"].detach(), label=f"{name} $\\mu_y$")
        if "mu_x" in out.params:
            ax.plot(fz, out.params["mu_x"].detach(), "--", label=f"{name} $\\mu_x$")
    ax.set_xlabel("$F_z$ [N]")
    ax.set_ylabel(r"learned $\mu$ [-]")
    ax.set_title("Learned friction vs load")
    ax.legend(fontsize=7)
    return fig


def plot_time_series(t, series: dict, ylabel="", title=""):
    """Generic time-series plot — used for relaxation, temperature, wear and graining."""
    fig, ax = _fig(figsize=(6.0, 3.4))
    for name, y in series.items():
        ax.plot(np.asarray(t), np.asarray(y), label=name)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7)
    return fig
