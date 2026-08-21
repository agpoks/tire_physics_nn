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


# ----------------------------------------------------------- contact patch
def fig_patch_mechanics():
    """What the discretised patch is doing internally: adhesion giving way to sliding."""
    from tire_nn.physics.brush_patch import parabolic_pressure, patch_coordinates, patch_forces

    A, KB, MU, NE = 0.06, 6.0e6, 1.0, 200
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.0), sharey=True)
    for ax, slip in zip(axes, (0.015, 0.035, 0.07)):
        a = torch.tensor([A])
        Fz = torch.tensor([1000.0])
        xi, _ = patch_coordinates(NE, a)
        p = parabolic_pressure(xi, a, Fz)
        out = patch_forces(torch.zeros(1), torch.tan(torch.tensor([slip])), Fz, a,
                           torch.tensor([KB]), torch.tensor([MU]), n_elements=NE)
        x = xi[0].numpy() * 1000
        ax.plot(x, (MU * p[0]).numpy(), "k--", lw=1.3, label=r"bound $\mu p(\xi)$")
        # tau_y is already the positive magnitude along the slip direction; the sign
        # flip to SAE convention happens when it is integrated into Fy.
        ax.plot(x, out["tau_y"][0].numpy(), color="tab:orange", lw=2.2, label=r"shear $|\tau|$")
        ax.fill_between(x, 0, out["tau_y"][0].numpy(), color="tab:orange", alpha=0.22)
        ax.set_xlabel("distance from leading edge [mm]")
        ax.set_title(fr"$\alpha$={slip:.3f} rad, sliding {float(out['sliding_fraction']):.0%}",
                     fontsize=9)
    axes[0].set_ylabel("line load [N/m]")
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle("The sliding region eats forward from the trailing edge — that is the force curve",
                 fontsize=9.5)
    return save(fig, "patch_mechanics.png")


def fig_patch_pressure_recovery():
    """A tyre the parabolic assumption cannot represent, and what each model does."""
    from tire_nn.models.patch_brush_net import PatchBrushNet
    from tire_nn.physics.brush_patch import patch_coordinates, patch_forces

    torch.manual_seed(0)
    A, KB, MU, NE, n = 0.06, 6.0e6, 1.0, 48, 300
    alpha = torch.linspace(-0.09, 0.09, n)
    Fz = torch.full((n,), 1000.0)
    a = torch.full((n,), A)
    xi, _ = patch_coordinates(NE, a)
    u = xi / (2 * a.unsqueeze(-1))
    shape = torch.clamp(u, min=1e-6) ** 2.2 * torch.clamp(1 - u, min=1e-6) ** 0.9
    dxi = (2 * a / NE).unsqueeze(-1)
    p_true = shape / (shape * dxi).sum(-1, keepdim=True) * Fz.unsqueeze(-1)
    truth = patch_forces(torch.zeros(n), torch.tan(alpha), Fz, a,
                         torch.full((n,), KB), torch.full((n,), MU),
                         pressure=p_true, n_elements=NE)["Fy"]
    data = truth + torch.randn(n) * 3.0

    fitted = {}
    for label, learn in (("learned", True), ("parabolic", False)):
        model = PatchBrushNet(n_elements=NE, learn_pressure=learn)
        opt = torch.optim.Adam(model.parameters(), lr=0.02)
        for _ in range(900):
            opt.zero_grad(set_to_none=True)
            loss = ((model(alpha, torch.zeros(n), Fz).Fy - data) ** 2).mean() / 1e6
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            fitted[label] = (model.pressure_profile(torch.tensor([1000.0]))[0],
                             model(alpha, torch.zeros(n), Fz).Fy,
                             float(torch.sqrt(((model(alpha, torch.zeros(n), Fz).Fy - data) ** 2).mean())))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.3))
    x = xi[0].numpy() * 1000
    axes[0].plot(x, p_true[0].numpy(), "k--", lw=1.7, label="true profile")
    axes[0].plot(x, fitted["learned"][0].numpy(), color="tab:orange", lw=2.2, label="learned")
    axes[0].plot(x, fitted["parabolic"][0].numpy(), color="tab:blue", lw=1.7, label="parabolic (fixed)")
    axes[0].set_xlabel("distance from leading edge [mm]")
    axes[0].set_ylabel("line load [N/m]")
    axes[0].set_title("Pressure distribution")
    axes[0].legend(fontsize=7.5)

    axes[1].scatter(alpha, data, s=3, alpha=0.25, color="0.6", label="data (3 N noise)")
    axes[1].plot(alpha, fitted["learned"][1], color="tab:orange", lw=2.2,
                 label=f"learned, RMSE {fitted['learned'][2]:.1f} N")
    axes[1].plot(alpha, fitted["parabolic"][1], color="tab:blue", lw=1.7,
                 label=f"parabolic, RMSE {fitted['parabolic'][2]:.1f} N")
    axes[1].set_xlabel(r"$\alpha$ [rad]")
    axes[1].set_ylabel("$F_y$ [N]")
    axes[1].set_title("Force curve")
    axes[1].legend(fontsize=7.5)
    return save(fig, "patch_pressure_recovery.png")


# ---------------------------------------------------------------- imaging
def fig_tread_images():
    """The synthetic tread imagery: what wear and graining look like to a camera."""
    from tire_nn.data.tread_images import make_tread_image

    rng = np.random.default_rng(0)
    values = (0.0, 0.25, 0.5, 0.75, 1.0)
    fig, axes = plt.subplots(2, 5, figsize=(9.6, 4.3))
    for j, v in enumerate(values):
        axes[0, j].imshow(make_tread_image(v, 0.0, rng=rng), cmap="gray", vmin=0, vmax=1)
        axes[0, j].set_title(f"wear {v:.2f}", fontsize=8.5)
        axes[1, j].imshow(make_tread_image(0.3, v, rng=rng), cmap="gray", vmin=0, vmax=1)
        axes[1, j].set_title(f"graining {v:.2f}", fontsize=8.5)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].set_ylabel("wear sweep", fontsize=9)
    axes[1, 0].set_ylabel("graining sweep\n(wear fixed at 0.3)", fontsize=9)
    fig.suptitle("SYNTHETIC tread imagery — no public dataset of tread depth or graining exists",
                 fontsize=9.5)
    return save(fig, "tread_images.png")


def fig_identifiability_gain():
    """The measured effect of adding an image channel to the degradation UDE."""
    # From notebook 06 / the seed sweeps: four seeds per condition.
    lap_only = [0.283, 0.552, 0.794, 0.795]
    with_image = [0.934, 0.942, 0.936, 0.952]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3))

    for i, (label, values) in enumerate((("lap time\nonly", lap_only),
                                         ("lap time\n+ image", with_image))):
        axes[0].scatter([i] * len(values), values, s=55, alpha=0.8, zorder=3)
        axes[0].plot([i - 0.17, i + 0.17], [np.mean(values)] * 2, "k-", lw=2.2, zorder=4)
        axes[0].vlines(i, min(values), max(values), lw=10, alpha=0.18)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["lap time only", "lap time + image"])
    axes[0].set_ylabel("wear correlation with hidden truth")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("One photograph per pit stop")

    axes[1].bar(["lap time\nonly", "lap time\n+ image"],
                [max(lap_only) - min(lap_only), max(with_image) - min(with_image)],
                color=["tab:red", "tab:green"], alpha=0.75)
    axes[1].set_ylabel("spread across seeds")
    axes[1].set_title("Spread collapses 28x")
    for i, v in enumerate([max(lap_only) - min(lap_only), max(with_image) - min(with_image)]):
        axes[1].text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=8.5)
    return save(fig, "identifiability_gain.png")


# ------------------------------------------------------------ degradation
def fig_degradation_signal():
    """The real degradation signal in F1 lap times, if the data has been downloaded."""
    try:
        from tire_nn.data.lap_degradation import load_fastf1_stints
        df = load_fastf1_stints(ROOT / "data" / "raw")
        source = f"{df['session_id'].nunique()} races, {len(df)} dry laps (real F1 timing)"
    except Exception:
        from tire_nn.data.lap_degradation import make_synthetic_stints
        df = make_synthetic_stints(n_sessions=8, n_drivers=6, seed=0)
        source = "SYNTHETIC stints (real F1 data not downloaded)"

    d = df.copy()
    d["rel"] = d["lap_time"] - d.groupby(["session_id", "driver", "stint"])["lap_time"].transform("median")
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    for compound in ("soft", "medium", "hard"):
        subset = d[(d["compound"] == compound) & (d["tyre_age"] <= 20)]
        if subset.empty:
            continue
        curve = subset.groupby("tyre_age")["rel"].median()
        ax.plot(curve.index, curve.values, "o-", ms=3.5, lw=1.6, label=compound)
    ax.axhline(0.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("tyre age [laps]")
    ax.set_ylabel("lap time relative to stint median [s]")
    ax.set_title(f"Degradation seen through lap time\n{source}", fontsize=9)
    ax.legend(fontsize=8)
    return save(fig, "degradation_signal.png")

def fig_real_tyre_images():
    """Real tyre photographs, and what the encoded model does with them."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from tire_nn.data.tread_images import load_tyre_quality_images, make_tread_dataset
    from tire_nn.models.condition_vision import TreadConditionNet, ordinal_loss

    try:
        real = load_tyre_quality_images(ROOT / "data" / "raw", size=64)
    except FileNotFoundError:
        print("  (real tyre photographs not downloaded — skipping; run "
              "scripts/download_tyre_images.py)")
        return None

    torch.manual_seed(0)
    X, y = real["images"], real["label"]
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(0))
    X, y = X[perm], y[perm]
    n_train = int(0.75 * len(X))
    Xtr, ytr, Xte, yte = X[:n_train], y[:n_train], X[n_train:], y[n_train:]

    def train_on(images, labels, epochs=30):
        model = TreadConditionNet(n_classes=2, predict_graining=False, width=16)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        loader = DataLoader(TensorDataset(images, labels), batch_size=32, shuffle=True)
        for _ in range(epochs):
            for img, lab in loader:
                opt.zero_grad(set_to_none=True)
                ordinal_loss(model(img)["cumulative"], lab).backward()
                opt.step()
        return model.eval()

    trained_real = train_on(Xtr, ytr)
    synthetic = make_tread_dataset(n=1200, size=64, seed=0)
    trained_syn = train_on(synthetic["images"], (synthetic["wear"] > 0.5).long())

    with torch.no_grad():
        out_real = trained_real(Xte)
        out_transfer = trained_syn(Xte)

    fig = plt.figure(figsize=(10.2, 5.6))
    grid = fig.add_gridspec(2, 4, height_ratios=[1.15, 1.0], hspace=0.35, wspace=0.3)

    # Row 1: the photographs themselves.
    good_idx = (y == 0).nonzero().flatten()[:2]
    bad_idx = (y == 1).nonzero().flatten()[:2]
    for column, (index, label) in enumerate(
            [(good_idx[0], "good"), (good_idx[1], "good"),
             (bad_idx[0], "defective"), (bad_idx[1], "defective")]):
        ax = fig.add_subplot(grid[0, column])
        ax.imshow(X[index, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(label, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    # Row 2, left: the learned wear index separates the real classes.
    ax = fig.add_subplot(grid[1, :2])
    bins = np.linspace(0, 1, 22)
    ax.hist(out_real["wear"][yte == 0].numpy(), bins=bins, alpha=0.65, label="good")
    ax.hist(out_real["wear"][yte == 1].numpy(), bins=bins, alpha=0.65, label="defective")
    ax.axvline(float(trained_real.thresholds()[0].detach()), color="k", ls="--", lw=1.2,
               label="learned threshold")
    accuracy = float((out_real["class_prob"].argmax(-1) == yte).float().mean())
    ax.set_xlabel("wear index")
    ax.set_ylabel("count")
    ax.set_title(f"Trained on real photographs — accuracy {accuracy:.2f}", fontsize=9.5)
    ax.legend(fontsize=7.5)

    # Row 2, right: the synthetic-trained model fails to transfer.
    ax = fig.add_subplot(grid[1, 2:])
    ax.hist(out_transfer["wear"][yte == 0].numpy(), bins=bins, alpha=0.65, label="good")
    ax.hist(out_transfer["wear"][yte == 1].numpy(), bins=bins, alpha=0.65, label="defective")
    transfer_acc = float((out_transfer["class_prob"].argmax(-1) == yte).float().mean())
    ax.set_xlabel("wear index")
    ax.set_title(f"Synthetic-trained, tested on real — accuracy {transfer_acc:.2f}\n"
                 f"(chance = 0.50)", fontsize=9.5)
    ax.legend(fontsize=7.5)

    fig.suptitle("Real tyre photographs (NMiriams/Good_Tires, NMiriams/Defective_Tires, "
                 "CC BY 4.0)", fontsize=9.5)
    return save(fig, "real_tyre_images.png")


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
    fig_patch_mechanics()
    fig_patch_pressure_recovery()
    fig_tread_images()
    fig_identifiability_gain()
    fig_degradation_signal()
    fig_real_tyre_images()
    fig_learned_mu_bounds()
    print("done.")


if __name__ == "__main__":
    main()
