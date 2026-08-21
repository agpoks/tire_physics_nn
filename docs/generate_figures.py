#!/usr/bin/env python3
"""Regenerate every figure used by the documentation.

    python docs/generate_figures.py

Figures are written to ``docs/source/_static/figures/`` and committed, so the
ReadTheDocs build never has to import torch or run training. Re-run this script
whenever a model or a default changes.

Everything here uses *untrained* or analytically-parameterised models unless a
figure explicitly says otherwise: the point of most of these plots is that the
structural guarantees hold before any fitting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIG = ROOT / "docs" / "source" / "_static" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

from tire_nn.data.graining import make_synthetic_graining  # noqa: E402
from tire_nn.evaluation.physical_consistency import audit  # noqa: E402
from tire_nn.layers.friction_envelope import ellipse_radius, project_into_ellipse  # noqa: E402
from tire_nn.models import EncodedTireNet, MLPTireModel, SymmetryTireNet  # noqa: E402
from tire_nn.models.relaxation_tire import RelaxationTireCell  # noqa: E402
from tire_nn.physics import MFParams, MagicFormulaTire, VehicleParams  # noqa: E402
from tire_nn.physics.brush import brush_combined  # noqa: E402
from tire_nn.physics.pacejka import pacejka_lateral  # noqa: E402
from tire_nn.physics.vehicle_dynamics import quasi_static_loads  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "legend.fontsize": 8, "figure.autolayout": True})


def randomize(model, std=1.5, seed=0):
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, std)
    return model


def save(fig, name):
    path = FIG / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.relative_to(ROOT)}")
    return path


# --------------------------------------------------------------------- P2
def fig_symmetry():
    """Untrained MLP vs untrained symmetry-encoded net near zero slip."""
    a = torch.linspace(-0.3, 0.3, 401)
    z = torch.zeros_like(a)
    Fz = torch.full_like(a, 1000.0)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    for seed in range(4):
        mlp = randomize(MLPTireModel(hidden=(64, 64)), 1.2, seed)
        sym = randomize(SymmetryTireNet(), 1.2, seed)
        axes[0].plot(a, mlp(a, z, Fz).Fy.detach(), lw=1.2, label=f"seed {seed}")
        axes[1].plot(a, sym(a, z, Fz).Fy.detach(), lw=1.2, label=f"seed {seed}")
    for ax, title in zip(axes, ["plain MLP (no prior)", "symmetry-encoded (P2)"]):
        ax.axvline(0, color="k", lw=0.8)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel(r"slip angle $\alpha$ [rad]")
        ax.set_ylabel("$F_y$ [N]")
        ax.set_title(title)
        ax.legend()
    axes[0].annotate("force at zero slip", xy=(0, float(mlp(z, z, Fz).Fy[0])),
                     xytext=(0.08, float(mlp(z, z, Fz).Fy[0]) * 1.6),
                     arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=8)
    return save(fig, "symmetry_zero_slip.png")


# --------------------------------------------------------------------- P3
def fig_envelope_scaling():
    rho = torch.linspace(0, 4, 500)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    axes[0].plot(rho, torch.tanh(rho) / rho.clamp(min=1e-9), label=r"$\tanh(\rho)/\rho$ (default)")
    axes[0].plot(rho, 1 / torch.sqrt(1 + rho**2), label=r"$1/\sqrt{1+\rho^2}$ (algebraic)")
    axes[0].plot(rho, torch.clamp(1 / rho.clamp(min=1e-9), max=1.0), "--",
                 label=r"hard clip $\min(1,1/\rho)$")
    axes[0].set_xlabel(r"raw utilisation $\rho$")
    axes[0].set_ylabel(r"scaling $s(\rho)$")
    axes[0].set_title("Envelope scaling functions")
    axes[0].legend()

    axes[1].plot(rho, rho * torch.tanh(rho) / rho.clamp(min=1e-9), label="tanh projection")
    axes[1].plot(rho, rho / torch.sqrt(1 + rho**2), label="algebraic projection")
    axes[1].plot(rho, torch.minimum(rho, torch.ones_like(rho)), "--", label="hard clip")
    axes[1].axhline(1.0, color="k", lw=0.8, ls=":")
    axes[1].set_xlabel(r"raw utilisation $\rho$")
    axes[1].set_ylabel(r"projected utilisation $\hat\rho$")
    axes[1].set_title(r"Output can never exceed $\hat\rho=1$")
    axes[1].legend()
    return save(fig, "envelope_scaling.png")


def fig_friction_ellipse():
    n = 70
    a = torch.linspace(-0.4, 0.4, n).repeat_interleave(n)
    k = torch.linspace(-0.4, 0.4, n).repeat(n)
    Fz = torch.full_like(a, 1000.0)
    sym = randomize(SymmetryTireNet(), 1.0, 3)
    enc = randomize(EncodedTireNet(), 1.0, 3)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.1))
    th = np.linspace(0, 2 * np.pi, 400)
    for ax, model, title in ((axes[0], sym, "symmetry only (P2)"),
                             (axes[1], enc, "symmetry + hard envelope (P2+P3)")):
        out = model(a, k, Fz)
        mu = float(out.params["mu_x"].mean().detach()) if "mu_x" in out.params else 1.0
        ax.scatter(out.Fx.detach(), out.Fy.detach(), s=1.5, alpha=0.35)
        ax.plot(mu * 1000 * np.cos(th), mu * 1000 * np.sin(th), "k--", lw=1.2,
                label=fr"$\mu F_z$ ($\mu$={mu:.2f})")
        ax.set_aspect("equal")
        ax.set_xlabel("$F_x$ [N]")
        ax.set_ylabel("$F_y$ [N]")
        ax.set_title(title)
        ax.legend(loc="upper right")
    # Note the axis scales: the unconstrained model leaves the ellipse by a factor of
    # ~40, so a shared scale would make the constrained panel invisible.
    axes[0].text(0.03, 0.03, "note the axis scale", transform=axes[0].transAxes, fontsize=7)
    return save(fig, "friction_ellipse.png")


# --------------------------------------------------------------------- P4
def fig_magic_formula_parameters():
    a = torch.linspace(-0.4, 0.4, 501)
    Fz = torch.full_like(a, 1000.0)
    base = dict(B=9.0, C=1.6, E=0.4, mu=1.0, k_mu=0.0)
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.9), sharey=True)
    sweeps = [("B", [4.0, 9.0, 18.0]), ("C", [1.1, 1.6, 2.2]),
              ("E", [-1.0, 0.4, 0.95]), ("mu", [0.6, 1.0, 1.4])]
    for ax, (name, values) in zip(axes, sweeps):
        for v in values:
            p = MFParams(**{**base, name: v})
            Fy, _ = pacejka_lateral(a, Fz, p)
            ax.plot(a, Fy.detach(), label=f"{name}={v}")
        ax.set_xlabel(r"$\alpha$ [rad]")
        ax.set_title(f"effect of ${name}$" if name != "mu" else r"effect of $\mu$")
        ax.legend()
    axes[0].set_ylabel("$F_y$ [N]")
    return save(fig, "magic_formula_parameters.png")


def fig_steady_state_laws():
    """The four classical steady-state laws, matched at the same stiffness and mu."""
    from tire_nn.physics import dugoff_tire, linear_tire

    Fz = torch.full((601,), 1000.0)
    a = torch.linspace(-0.35, 0.35, 601)
    z = torch.zeros_like(a)
    mu, C_alpha = 1.0, 45000.0

    # Match the Magic Formula's initial slope to C_alpha, since C_alpha = B*C*D.
    C_mf, E_mf = 1.6, 0.4
    B_mf = C_alpha / (C_mf * mu * 1000.0)
    p = MFParams(B=B_mf, C=C_mf, E=E_mf, mu=mu, k_mu=0.0)

    curves = {
        "linear": linear_tire(a, z, Fz, C_alpha, C_alpha)[1],
        "brush": brush_combined(a, z, Fz, C_alpha, C_alpha, mu)[1],
        "Dugoff": dugoff_tire(a, z, Fz, C_alpha, C_alpha, mu)[1],
        "Magic Formula": pacejka_lateral(a, Fz, p)[0],
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    for name, Fy in curves.items():
        axes[0].plot(a, Fy.detach(), label=name, lw=1.5)
        axes[1].plot(a, Fy.detach(), label=name, lw=1.5)
    for ax in axes:
        ax.axhline(-mu * 1000, color="k", ls=":", lw=1)
        ax.axhline(mu * 1000, color="k", ls=":", lw=1)
        ax.set_xlabel(r"slip angle $\alpha$ [rad]")
        ax.set_ylabel("$F_y$ [N]")
    axes[0].set_ylim(-2600, 2600)
    axes[0].set_title(r"All four, matched $C_\alpha$ and $\mu$")
    axes[0].text(-0.33, -mu * 1000 - 300, r"$\pm\mu F_z$", fontsize=7)
    axes[0].legend(loc="lower left", fontsize=7.5)
    axes[1].set_xlim(-0.06, 0.06)
    axes[1].set_ylim(-1500, 1500)
    axes[1].set_title("Zoom: they agree only near zero slip")
    return save(fig, "steady_state_laws.png")


def fig_combined_slip_methods():
    """Combined-slip force locus of the three bounded analytical laws."""
    from tire_nn.physics import dugoff_tire

    # Slip range chosen so the interior of the locus is visible: with a wider range
    # every model is fully sliding everywhere and all three collapse onto the circle,
    # which hides exactly the difference this figure is meant to show.
    n = 61
    a = torch.linspace(-0.14, 0.14, n).repeat_interleave(n)
    k = torch.linspace(-0.14, 0.14, n).repeat(n)
    Fz = torch.full_like(a, 1000.0)
    mu, C = 1.0, 45000.0
    p = MFParams(B=C / (1.6 * mu * 1000.0), C=1.6, E=0.4, mu=mu, k_mu=0.0)

    loci = {
        "brush": brush_combined(a, k, Fz, C, C, mu),
        "Dugoff": dugoff_tire(a, k, Fz, C, C, mu),
        "Magic Formula (similarity)": tuple(MagicFormulaTire(p, p)(a, k, Fz)),
    }
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6), sharex=True, sharey=True)
    th = np.linspace(0, 2 * np.pi, 400)
    for ax, (name, (Fx, Fy)) in zip(axes, loci.items()):
        ax.plot(mu * 1000 * np.cos(th), mu * 1000 * np.sin(th), "k--", lw=1.1, label=r"$\mu F_z$")
        ax.scatter(Fx.detach(), Fy.detach(), s=1.6, alpha=0.35)
        ax.set_aspect("equal")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("$F_x$ [N]")
        util = float((torch.sqrt(Fx ** 2 + Fy ** 2) / (mu * Fz)).max().detach())
        ax.text(0.5, 0.03, f"peak utilisation {util:.2f}", transform=ax.transAxes,
                ha="center", fontsize=7.5)
    axes[0].set_ylabel("$F_y$ [N]")
    axes[0].legend(fontsize=7.5, loc="upper right")
    return save(fig, "combined_slip_methods.png")


def fig_thermal_two_time_scales():
    """A step in slip power: the surface responds in seconds, the core over a stint."""
    from tire_nn.physics import ThermalParams, thermal_rates

    p = ThermalParams()
    dt, T = 0.05, 24000
    t = np.arange(T) * dt
    P = np.where((t > 60) & (t < 900), 3000.0, 200.0)
    Ts, Tc = 300.0, 300.0
    hist = np.empty((T, 2))
    for i in range(T):
        hist[i] = (Ts, Tc)
        dTs, dTc = thermal_rates(*(torch.tensor([float(v)]) for v in (Ts, Tc, P[i], 305.0, 298.0)), p)
        Ts += dt * float(dTs)
        Tc += dt * float(dTc)

    fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.0), sharex=True,
                             gridspec_kw={"height_ratios": [1, 2]})
    axes[0].plot(t, P / 1000, color="tab:red")
    axes[0].set_ylabel("$P_{slip}$ [kW]")
    axes[1].plot(t, hist[:, 0], label=r"$T_s$ surface, $\tau \approx 32$ s")
    axes[1].plot(t, hist[:, 1], label=r"$T_c$ core, $\tau \approx 242$ s")
    axes[1].axhline(305.0, color="k", ls=":", lw=1, label="road")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("temperature [K]")
    axes[1].legend(fontsize=7.5)
    axes[0].set_title("Two thermal time scales from one energy input")
    return save(fig, "thermal_two_time_scales.png")


def fig_effective_friction():
    """How tire condition modulates the friction limit."""
    from tire_nn.models import EncodedTireNet
    from tire_nn.models.thermo_graining_tire import ThermoGrainingTire

    m = ThermoGrainingTire(EncodedTireNet())
    Ts = torch.linspace(290, 430, 300)
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 2.9))
    axes[0].plot(Ts, m.temperature_factor(Ts).detach())
    axes[0].axvline(float(m.T_opt().detach()), color="k", ls=":", lw=1)
    axes[0].set_xlabel("$T_s$ [K]")
    axes[0].set_title("Grip window (not monotone)")

    w = torch.linspace(0, 3, 200)
    for kw in (0.05, 0.1, 0.3):
        axes[1].plot(w, torch.exp(-kw * w), label=f"$k_w$={kw}")
    axes[1].set_xlabel("wear [-]")
    axes[1].set_title(r"Wear: $e^{-k_w w}$, irreversible")
    axes[1].legend(fontsize=7.5)

    g = torch.linspace(0, 1, 200)
    for kg in (0.1, 0.3, 0.6):
        axes[2].plot(g, 1 - kg * g, label=f"$k_g$={kg}")
    axes[2].set_xlabel("graining $g$ [-]")
    axes[2].set_title(r"Graining: $1-k_g g$, reversible")
    axes[2].legend(fontsize=7.5)
    for ax in axes:
        ax.set_ylabel(r"$\mu / \mu_{ref}$")
    return save(fig, "effective_friction.png")


def fig_learning_curve():
    """Measured data efficiency (Experiment 1, full-budget run)."""
    # From results/exp1/learning_curve.csv (6000-sample synthetic set, 300 epochs,
    # seed 0). Hard-coded because results/ is not committed.
    n = [210, 570, 1547, 4200]
    series = {
        "Magic Formula (scipy fit)": [196.9, 17.6, 17.6, 17.7],
        "plain MLP": [95.2, 20.3, 18.4, 18.1],
        "symmetry only": [75.3, 42.8, 26.8, 17.7],
        "residual grey box": [25.2, 19.0, 18.1, 17.5],
        "encoded (P2+P3)": [20.6, 19.0, 18.8, 18.6],
        "ParameterNet + MF": [17.4, 17.3, 17.3, 17.4],
    }
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for name, values in series.items():
        ax.plot(n, values, "o-", lw=1.5, ms=4, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training samples")
    ax.set_ylabel("test $F_y$ RMSE [N]")
    ax.set_title("Data efficiency (Experiment 1, synthetic)")
    ax.legend(fontsize=7.5)
    ax.set_xticks(n)
    ax.set_xticklabels([str(v) for v in n])
    return save(fig, "learning_curve.png")


def fig_transient_ratio():
    """Measured rise-distance ratio: distance-parameterised or time-parameterised?"""
    # run 1: results/exp2 (experiment script); run 2: notebook 02, shorter budget.
    models = ["static", "GRU", "Neural ODE", "relaxation\ncell", "relaxation +\nParameterNet"]
    run1 = [0.0, 2.25, 3.00, 1.25, 0.92]
    run2 = [0.0, 0.53, 3.00, 1.15, np.nan]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.bar(x - 0.19, run1, 0.36, label="run 1 (experiment script)")
    ax.bar(x + 0.19, run2, 0.36, label="run 2 (notebook, shorter budget)")
    ax.axhline(1.0, color="tab:green", ls="--", lw=1.4)
    ax.text(len(models) - 0.45, 1.08, "distance-parameterised (correct)",
            fontsize=7.5, ha="right", color="tab:green")
    ax.axhline(3.0, color="tab:red", ls=":", lw=1.4)
    ax.text(len(models) - 0.45, 3.08, "fixed time constant", fontsize=7.5, ha="right", color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8)
    ax.set_ylabel("rise distance at 30 m/s / at 10 m/s")
    ax.set_title("Does the transient scale with distance or with time?")
    ax.legend(fontsize=7.5, loc="upper left")
    return save(fig, "transient_ratio.png")


def fig_load_sensitivity():
    Fz = torch.linspace(200, 4000, 300)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    for k_mu in (0.0, 0.05, 0.12):
        from tire_nn.physics.pacejka import load_sensitive_mu
        axes[0].plot(Fz, load_sensitive_mu(Fz, 1.1, k_mu, 1000.0), label=f"$k_\\mu$={k_mu}")
        axes[1].plot(Fz, load_sensitive_mu(Fz, 1.1, k_mu, 1000.0) * Fz, label=f"$k_\\mu$={k_mu}")
    axes[0].set_ylabel(r"$\mu(F_z)$ [-]")
    axes[1].set_ylabel(r"peak force $D=\mu(F_z)F_z$ [N]")
    for ax in axes:
        ax.set_xlabel("$F_z$ [N]")
        ax.legend()
    axes[0].set_title("Load sensitivity of friction")
    axes[1].set_title("Peak force is sub-linear in load")
    return save(fig, "load_sensitivity.png")


# --------------------------------------------------------------------- P5
def fig_relaxation():
    cell = RelaxationTireCell(MagicFormulaTire(MFParams(mu=1.0), MFParams(mu=1.0)))
    with torch.no_grad():
        cell.sigma_y.raw.fill_(float(cell.sigma_y.spec.inverse(0.30)))
        cell.sigma_x.raw.fill_(float(cell.sigma_x.spec.inverse(0.15)))
    dt, T = 5e-4, 4000
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for vx in (10.0, 20.0, 40.0):
        a = torch.zeros(1, T)
        a[:, 200:] = 0.08
        k = torch.zeros(1, T)
        Fz = torch.full((1, T), 1000.0)
        v = torch.full((1, T), vx)
        F = cell.rollout(a, k, Fz, v, dt, {"vx": v}, method="exact").detach()
        t = np.arange(T) * dt
        axes[0].plot(t - 200 * dt, F[0, :, 1], label=f"$v_x$={vx:.0f} m/s")
        axes[1].plot((t - 200 * dt) * vx, F[0, :, 1], label=f"$v_x$={vx:.0f} m/s")
    axes[0].set_xlabel("time since step [s]")
    axes[0].set_xlim(-0.02, 0.25)
    axes[0].set_title("Step response vs TIME (speed dependent)")
    axes[1].set_xlabel("travelled distance since step [m]")
    axes[1].set_xlim(-0.2, 2.0)
    axes[1].axvline(0.30, color="k", ls=":", lw=1)
    axes[1].annotate(r"$\sigma_y=0.30$ m", xy=(0.30, -300), xytext=(0.55, -250), fontsize=8,
                     arrowprops=dict(arrowstyle="->", lw=0.8))
    axes[1].set_title("Step response vs DISTANCE (curves collapse)")
    for ax in axes:
        ax.set_ylabel("$F_y$ [N]")
        ax.legend()
    return save(fig, "relaxation_step.png")


# --------------------------------------------------------------------- P6
def fig_load_transfer():
    vp = VehicleParams(m=1200, Iz=1500, lf=1.3, lr=1.4, t_f=1.6, t_r=1.6, h_cg=0.45)
    t = np.linspace(0, 6, 600)
    ax_ = -6.0 * np.exp(-((t - 1.5) ** 2) / 0.6)
    ay_ = 8.0 * np.sin(2 * np.pi * 0.25 * t) * (t > 2.0)
    Fz = quasi_static_loads(torch.tensor(ax_, dtype=torch.float32),
                            torch.tensor(ay_, dtype=torch.float32), vp).numpy()
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.4), sharex=True)
    axes[0].plot(t, ax_, label="$a_x$")
    axes[0].plot(t, ay_, label="$a_y$")
    axes[0].set_ylabel("acceleration [m/s$^2$]")
    axes[0].legend()
    for i, name in enumerate(("FL", "FR", "RL", "RR")):
        axes[1].plot(t, Fz[:, i], label=name)
    axes[1].axhline(vp.m * 9.81 / 4, color="k", ls=":", lw=1, label="static/4")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("$F_{z,i}$ [N]")
    axes[1].legend(ncol=5)
    axes[0].set_title(r"Quasi-static load transfer; $\sum_i F_{z,i}=mg$ exactly")
    return save(fig, "load_transfer.png")


# --------------------------------------------------------------------- P7
def fig_graining_stint():
    df = make_synthetic_graining(T=8000, dt=0.05, seed=0)
    t = df["t"].to_numpy()
    fig, axes = plt.subplots(4, 1, figsize=(7.0, 7.4), sharex=True)
    axes[0].plot(t, df["P_slip"] / 1000.0, color="tab:red")
    axes[0].set_ylabel("$P_{slip}$ [kW]")
    axes[1].plot(t, df["Ts"], label="$T_s$ (surface)")
    axes[1].plot(t, df["Tc"], label="$T_c$ (core)")
    axes[1].set_ylabel("temperature [K]")
    axes[1].legend()
    axes[2].plot(t, df["graining"], color="tab:orange")
    axes[2].set_ylabel("graining $g$ [-]")
    axes[2].set_ylim(-0.05, 1.05)
    axes[3].plot(t, df["wear"], color="tab:brown")
    axes[3].set_ylabel("wear [-]")
    axes[3].set_xlabel("time [s]")
    phases = ["cold +\naggressive", "hot +\naggressive", "cool-down", "aggressive\nagain"]
    for i, name in enumerate(phases):
        for ax in axes:
            ax.axvline(t[len(t) // 4 * i], color="0.6", lw=0.8, ls="--")
        axes[0].text(t[len(t) // 4 * i] + 5, axes[0].get_ylim()[1] * 0.55, name, fontsize=7)
    axes[0].set_title("SYNTHETIC demonstrator — not validated real graining")
    return save(fig, "graining_stint.png")


# --------------------------------------------------------- ablation summary
def fig_violations():
    models = {
        "Magic Formula": MagicFormulaTire(),
        "plain MLP": randomize(MLPTireModel(), 1.0, 1),
        "symmetry": randomize(SymmetryTireNet(), 1.0, 1),
        "symmetry+envelope": randomize(EncodedTireNet(), 1.0, 1),
    }
    keys = ("zero_slip_force", "sym_violation_y", "envelope_violation", "dissipativity_violation")
    labels = ("force at\nzero slip", "odd-symmetry\nviolation", "friction-envelope\nviolation",
              "dissipativity\nviolation")
    floor = 1e-8
    data = {name: [max(audit(m, n=2048)[k], floor) for k in keys] for name, m in models.items()}
    x = np.arange(len(keys))
    width = 0.2
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    for i, (name, values) in enumerate(data.items()):
        ax.bar(x + (i - 1.5) * width, values, width, label=name)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("violation [-], load-normalised")
    ax.axhline(floor, color="k", lw=0.8, ls=":")
    ax.text(len(keys) - 0.5, floor * 1.4, "exactly zero", fontsize=7, ha="right")
    ax.set_title("Physical violations of untrained models on a wide audit grid")
    ax.legend(ncol=2)
    return save(fig, "violations.png")


def fig_learned_mu_bounds():
    """Bounded parameter transforms: the range holds for any raw network output."""
    z = torch.linspace(-30, 30, 1000)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0))
    from tire_nn.layers.bounded_parameters import to_bounded, to_positive
    axes[0].plot(z, to_bounded(z, 0.05, 2.5))
    axes[0].axhline(0.05, color="k", ls=":", lw=1)
    axes[0].axhline(2.5, color="k", ls=":", lw=1)
    axes[0].set_title(r"$\mu = \mu_{\min}+(\mu_{\max}-\mu_{\min})\,\sigma(z)$")
    axes[0].set_ylabel(r"$\mu$ [-]")
    axes[1].plot(z, to_positive(z, 0.01))
    axes[1].axhline(0.01, color="k", ls=":", lw=1)
    axes[1].set_yscale("log")
    axes[1].set_title(r"$\sigma_{rel} = \sigma_{\min}+\mathrm{softplus}(z)$")
    axes[1].set_ylabel("relaxation length [m]")
    for ax in axes:
        ax.set_xlabel("raw network output $z$")
    return save(fig, "bounded_parameters.png")


def main():
    print("Generating documentation figures:")
    fig_symmetry()
    fig_envelope_scaling()
    fig_friction_ellipse()
    fig_magic_formula_parameters()
    fig_steady_state_laws()
    fig_combined_slip_methods()
    fig_load_sensitivity()
    fig_relaxation()
    fig_load_transfer()
    fig_thermal_two_time_scales()
    fig_effective_friction()
    fig_graining_stint()
    fig_violations()
    fig_learning_curve()
    fig_transient_ratio()
    fig_learned_mu_bounds()
    print("done.")


if __name__ == "__main__":
    main()
