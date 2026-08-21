#!/usr/bin/env python3
"""Generate the schematic diagrams used by the documentation.

    python docs/diagrams.py

These are block-and-arrow schematics (matplotlib, no extra dependency) written to
``docs/source/_static/diagrams/``. They are committed, so the ReadTheDocs build never
imports torch or runs anything.

Keep them monochrome-friendly: fills are light, meaning is carried by outline style and
text, so the diagrams stay readable when printed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "source" / "_static" / "diagrams"
OUT.mkdir(parents=True, exist_ok=True)

# Semantic palette: what kind of thing a box is.
PHYSICS = dict(fc="#dfeaf7", ec="#2c5f9e")     # fixed equations, no parameters
LEARNED = dict(fc="#fde8d8", ec="#c96f1f")     # neural network
DATA = dict(fc="#e8e8e8", ec="#555555")        # inputs / measurements
LOSS = dict(fc="#f6dede", ec="#b03a3a")        # loss terms
GUARANTEE = dict(fc="#dff0e2", ec="#2f7d43")   # structural guarantee

plt.rcParams.update({"font.size": 8.5, "figure.dpi": 150})


def box(ax, x, y, w, h, text, style, fontsize=8.5, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.3, **style))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, weight=weight, linespacing=1.35)
    return (x, y, w, h)


def arrow(ax, p0, p1, style="-|>", ls="-", color="#333333", text=None, dx=0.0, dy=0.02):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=11,
                                 linewidth=1.1, linestyle=ls, color=color,
                                 shrinkA=1, shrinkB=1))
    if text:
        ax.text((p0[0] + p1[0]) / 2 + dx, (p0[1] + p1[1]) / 2 + dy, text,
                ha="center", va="bottom", fontsize=7.5, color=color)


def canvas(w=9.0, h=4.0, title=None):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10.5, weight="bold", pad=6)
    return fig, ax


def legend(ax, items, y=0.02):
    x = 0.02
    for label, style in items:
        ax.add_patch(FancyBboxPatch((x, y), 0.022, 0.035,
                                    boxstyle="round,pad=0.004,rounding_size=0.01",
                                    linewidth=1.1, **style))
        ax.text(x + 0.03, y + 0.018, label, va="center", fontsize=7.5)
        x += 0.028 + 0.011 * len(label)


def save(fig, name):
    path = OUT / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {path.relative_to(OUT.parents[3])}")


# ---------------------------------------------------------------- taxonomy
def diagram_taxonomy():
    """Where the physics enters: the input, the loss, or the computation itself."""
    fig, ax = canvas(10.2, 4.4, "Three ways to put physics into a neural tire model")
    col_w, x0 = 0.29, 0.025
    titles = ["PHYSICS-GUIDED\nphysics shapes the INPUT",
              "PHYSICS-INFORMED\nphysics enters the LOSS",
              "PHYSICS-ENCODED\nphysics is the COMPUTATION"]
    verdicts = ["no property is guaranteed;\nthe physics is a hint",
                "holds in expectation on the\ntraining set; violated off-distribution",
                "holds for every weight vector,\neverywhere, always"]

    for i, (title, verdict) in enumerate(zip(titles, verdicts)):
        x = x0 + i * (col_w + 0.035)
        ax.text(x + col_w / 2, 0.93, title, ha="center", va="center",
                fontsize=9, weight="bold", linespacing=1.4)
        box(ax, x, 0.70, col_w, 0.12,
            "engineered features\n$\\kappa^2,\\ \\alpha^2,\\ F_z/F_{z0}$" if i == 0 else "raw inputs $\\alpha,\\kappa,F_z$",
            PHYSICS if i == 0 else DATA)
        if i == 2:
            box(ax, x, 0.46, col_w, 0.18,
                "network output passed through\nfixed physical structure\n"
                "$F=\\kappa\\,g(\\cdot)$, ellipse projection", LEARNED)
        else:
            box(ax, x, 0.46, col_w, 0.18, "black-box network\n(MLP / GRU)", LEARNED)
        box(ax, x, 0.30, col_w, 0.10, "prediction $F_x, F_y$", DATA)
        loss_text = "data loss" if i != 1 else "data loss\n$+\\ \\lambda\\,\\|$physics residual$\\|^2$"
        box(ax, x, 0.13, col_w, 0.11, loss_text, LOSS if i == 1 else DATA)
        for y0, y1 in ((0.70, 0.64), (0.46, 0.40), (0.30, 0.24)):
            arrow(ax, (x + col_w / 2, y0), (x + col_w / 2, y1))
        ax.text(x + col_w / 2, 0.055, verdict, ha="center", va="center", fontsize=7.5,
                style="italic", linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec=GUARANTEE["ec"] if i == 2 else "#999999", lw=1.0))
    save(fig, "taxonomy.png")


def diagram_integration_patterns():
    """The concrete ways a physical law can be wired into a learned model."""
    fig, ax = canvas(10.2, 5.6, "How a physical law gets wired into a learned tire model")
    patterns = [
        ("1. Feature engineering", "guided",
         "$x \\rightarrow \\phi_{phys}(x) \\rightarrow$ NN", "cheap; guarantees nothing"),
        ("2. Penalty / PINN", "informed",
         "$\\mathcal{L} = \\mathcal{L}_{data} + \\lambda\\,\\mathcal{L}_{phys}$",
         "any law expressible; soft, weight-dependent"),
        ("3. Residual (grey box)", "mixed",
         "$F = F_{phys}(x) + r_\\theta(x)$", "small NN, needs a good baseline"),
        ("4. Parameter network", "encoded",
         "$\\theta_{phys} = \\Pi_{bounds}(f_\\theta(c)),\\ F = \\mathrm{law}(x;\\theta_{phys})$",
         "fully interpretable; limited by the law"),
        ("5. Structural output map", "encoded",
         "$F = \\kappa\\,g_\\theta(\\kappa^2,\\alpha^2) \\cdot s(\\rho)$",
         "exact symmetry + bound; needs the law to be expressible"),
        ("6. Differentiable ODE", "encoded",
         "$\\dot z = f_{phys}(z, u, g_\\theta(\\cdot))$",
         "correct dynamics and stability; sequential cost"),
    ]
    styles = {"guided": PHYSICS, "informed": LOSS, "mixed": LEARNED, "encoded": GUARANTEE}
    for i, (name, kind, eq, note) in enumerate(patterns):
        col, row = i % 2, i // 2
        x = 0.03 + col * 0.495
        y = 0.72 - row * 0.245
        box(ax, x, y, 0.45, 0.17, "", styles[kind])
        ax.text(x + 0.018, y + 0.128, name, fontsize=9, weight="bold", va="center")
        ax.text(x + 0.018, y + 0.078, eq, fontsize=8.5, va="center")
        ax.text(x + 0.018, y + 0.030, note, fontsize=7.5, va="center", style="italic",
                color="#444444")
        ax.text(x + 0.432, y + 0.128, kind, fontsize=7, ha="right", va="center",
                color="#666666", weight="bold")
    ax.text(0.5, 0.045,
            "This project uses 3, 4, 5 and 6. Pattern 2 is implemented only as the control "
            "experiment;\npattern 1 is used for the invariants $\\kappa^2,\\alpha^2$ that make pattern 5 work.",
            ha="center", fontsize=8, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="#f7f7f7", ec="#999999"))
    save(fig, "integration_patterns.png")


# ------------------------------------------------------------- model shapes
def diagram_encoded_tire():
    fig, ax = canvas(10.0, 3.9, "EncodedTireNet — symmetry and the friction envelope as computation")
    box(ax, 0.02, 0.60, 0.135, 0.20, "wheel state\n$\\omega, v_x, v_y, \\delta$", DATA)
    box(ax, 0.185, 0.60, 0.15, 0.20, "slip kinematics\n$\\alpha,\\ \\kappa$\n(analytic, P1)", PHYSICS, 8)
    box(ax, 0.365, 0.62, 0.15, 0.16, "even invariants\n$\\kappa^2,\\ \\alpha^2,\\ F_z$", PHYSICS, 8)
    box(ax, 0.545, 0.60, 0.14, 0.20, "MLP\n$g_x, g_y > 0$\n(softplus)", LEARNED, 8)
    box(ax, 0.545, 0.24, 0.14, 0.20, "MLP head\n$\\mu_x, \\mu_y$\nbounded (P4)", LEARNED, 8)
    box(ax, 0.715, 0.60, 0.145, 0.20,
        "odd factors (P2)\n$q_x=\\kappa g_x$\n$q_y=-\\alpha g_y$", PHYSICS, 8)
    box(ax, 0.715, 0.24, 0.145, 0.20,
        "envelope (P3)\n$s(\\rho)=\\tanh\\rho/\\rho$", PHYSICS, 8)
    box(ax, 0.885, 0.42, 0.10, 0.20, "$F_x, F_y$", DATA)

    for p0, p1 in (((0.155, 0.70), (0.185, 0.70)), ((0.335, 0.70), (0.365, 0.70)),
                   ((0.515, 0.70), (0.545, 0.70)), ((0.685, 0.70), (0.715, 0.70)),
                   ((0.685, 0.34), (0.715, 0.34)), ((0.860, 0.70), (0.885, 0.56)),
                   ((0.860, 0.34), (0.885, 0.48))):
        arrow(ax, p0, p1)
    arrow(ax, (0.7875, 0.60), (0.7875, 0.44), text="$q_x,q_y$", dy=-0.01)
    arrow(ax, (0.44, 0.62), (0.50, 0.34), ls="--")
    ax.text(0.435, 0.46, "$F_z$, context", fontsize=7, rotation=-55, color="#555555")

    ax.text(0.5, 0.10,
            "Guaranteed for ANY weights:   $F_y(\\alpha{=}0)=0$   ·   $F(-\\alpha,-\\kappa) = -F(\\alpha,\\kappa)$   ·   "
            "$(F_x/\\mu_x F_z)^2+(F_y/\\mu_y F_z)^2 < 1$",
            ha="center", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.45", **{"fc": GUARANTEE["fc"], "ec": GUARANTEE["ec"]}))
    legend(ax, [("fixed physics", PHYSICS), ("learned", LEARNED), ("signals", DATA)], y=0.0)
    save(fig, "encoded_tire.png")


def diagram_parameter_net():
    fig, ax = canvas(9.4, 3.5, "ParameterTireNet — the network predicts a tire, not a force")
    box(ax, 0.03, 0.52, 0.16, 0.22, "operating condition\n$F_z$, $p$, $T_s$, tire id\n(NOT the slip)", DATA, 8)
    box(ax, 0.23, 0.52, 0.15, 0.22, "small MLP\n$z \\in \\mathbb{R}^{11}$", LEARNED)
    box(ax, 0.42, 0.52, 0.19, 0.22,
        "bounded transforms\n$p_{\\min}+(p_{\\max}-p_{\\min})\\sigma(z)$\n$p_{\\min}+\\mathrm{softplus}(z)$", PHYSICS, 8)
    box(ax, 0.65, 0.52, 0.15, 0.22, "$\\mu, B, C, E,$\n$k_\\mu, \\sigma_{rel}$", PHYSICS, 8.5)
    box(ax, 0.84, 0.52, 0.13, 0.22, "Magic Formula\n$F(\\alpha,\\kappa,F_z)$", PHYSICS, 8)
    box(ax, 0.42, 0.16, 0.19, 0.16, "slip $\\alpha, \\kappa$", DATA)
    for p0, p1 in (((0.19, 0.63), (0.23, 0.63)), ((0.38, 0.63), (0.42, 0.63)),
                   ((0.61, 0.63), (0.65, 0.63)), ((0.80, 0.63), (0.84, 0.63))):
        arrow(ax, p0, p1)
    arrow(ax, (0.61, 0.24), (0.90, 0.52))
    ax.text(0.5, 0.04,
            "Every coefficient is inside its declared physical range at every training step, so the model "
            "never\nholds a non-physical tire ($C<0$ flips the curve, $D<0$ is a negative peak force).",
            ha="center", fontsize=8, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="#f7f7f7", ec="#999999"))
    save(fig, "parameter_net.png")


def diagram_relaxation_cell():
    fig, ax = canvas(9.4, 3.4, "RelaxationTireCell — first-order lag in travelled distance")
    box(ax, 0.03, 0.55, 0.17, 0.20, "any static tire model\n$F_{ss}(\\alpha,\\kappa,F_z)$", LEARNED, 8)
    box(ax, 0.25, 0.55, 0.16, 0.20, "$\\sigma_x,\\sigma_y>0$\nsoftplus", LEARNED, 8.5)
    box(ax, 0.46, 0.55, 0.17, 0.20, "$\\tau_i = \\dfrac{\\sigma_i}{|v_x|+\\varepsilon}$", PHYSICS, 9)
    box(ax, 0.68, 0.55, 0.19, 0.20, "$\\dot F_i = \\dfrac{F_{i,ss}-F_i}{\\tau_i}$", PHYSICS, 9)
    box(ax, 0.68, 0.20, 0.19, 0.16, "integrator\nexact / RK4 / Euler", PHYSICS, 8)
    box(ax, 0.90, 0.45, 0.08, 0.20, "$F_x,F_y$", DATA)
    for p0, p1 in (((0.20, 0.65), (0.68, 0.65)), ((0.41, 0.65), (0.46, 0.65)),
                   ((0.63, 0.65), (0.68, 0.65)), ((0.775, 0.55), (0.775, 0.36))):
        arrow(ax, p0, p1)
    arrow(ax, (0.87, 0.28), (0.94, 0.45))
    arrow(ax, (0.68, 0.28), (0.60, 0.28), ls="--")
    arrow(ax, (0.60, 0.28), (0.60, 0.55), ls="--", text="state feedback", dx=-0.06, dy=0.05)
    ax.text(0.5, 0.045,
            "$\\sigma>0 \\Rightarrow \\tau>0 \\Rightarrow$ contractive: the model CANNOT learn a divergent transient. "
            "At $v_x\\to 0$ the force freezes,\nwhich is right — relaxation is a rolling-distance phenomenon.",
            ha="center", fontsize=8, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.4", **{"fc": GUARANTEE["fc"], "ec": GUARANTEE["ec"]}))
    save(fig, "relaxation_cell.png")


def diagram_four_wheel():
    fig, ax = canvas(9.6, 4.2, "FourWheelVehicle — one shared tire model, exact rigid-body aggregation")
    box(ax, 0.03, 0.52, 0.15, 0.22, "vehicle state\n$v_x,v_y,r,\\delta,\\omega_i$", DATA, 8)
    box(ax, 0.21, 0.52, 0.16, 0.22,
        "corner kinematics\n$v_i = v_{cog}+\\omega\\times r_i$\nload transfer", PHYSICS, 7.5)
    box(ax, 0.42, 0.44, 0.15, 0.30,
        "ONE shared\nTireNet\n\nevaluated 4x", LEARNED, 9, "bold")
    for i, name in enumerate(("FL", "FR", "RL", "RR")):
        box(ax, 0.61, 0.72 - i * 0.09, 0.075, 0.075, name, PHYSICS, 7.5)
        arrow(ax, (0.57, 0.59), (0.61, 0.755 - i * 0.09))
        arrow(ax, (0.685, 0.755 - i * 0.09), (0.73, 0.59))
    box(ax, 0.73, 0.44, 0.16, 0.30,
        "Newton–Euler\n$m(\\dot v_x - r v_y)=\\sum F_x$\n$m(\\dot v_y + r v_x)=\\sum F_y$\n"
        "$I_z\\dot r=\\sum(x_iF_{y,i}-y_iF_{x,i})$", PHYSICS, 7)
    box(ax, 0.905, 0.52, 0.075, 0.22, "$a_x,a_y$\n$\\dot r$", DATA, 8)
    for p0, p1 in (((0.18, 0.63), (0.21, 0.63)), ((0.37, 0.60), (0.42, 0.60)),
                   ((0.89, 0.60), (0.905, 0.60))):
        arrow(ax, p0, p1)
    arrow(ax, (0.81, 0.44), (0.81, 0.30), color="#b03a3a")
    arrow(ax, (0.81, 0.30), (0.495, 0.30), color="#b03a3a")
    arrow(ax, (0.495, 0.30), (0.495, 0.44), color="#b03a3a",
          text="IMU error gradient", dx=0.0, dy=0.02)
    ax.text(0.5, 0.10,
            "No chassis quantity is learnable, so a good fit cannot be bought with a wrong $I_z$. "
            "Sharing turns\nevery vehicle sample into four constraints on one constitutive law.",
            ha="center", fontsize=8, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="#f7f7f7", ec="#999999"))
    save(fig, "four_wheel.png")


def diagram_condition_states():
    fig, ax = canvas(9.6, 4.3, "ThermoGrainingTire — latent condition states with structural guarantees")
    box(ax, 0.03, 0.62, 0.15, 0.16, "$F_x,F_y$ and\nslip velocity", DATA, 8)
    box(ax, 0.21, 0.62, 0.16, 0.16, "$P_{slip}=-F\\cdot v_{slip}$\n$\\geq 0$", PHYSICS, 8.5)
    box(ax, 0.41, 0.76, 0.24, 0.13,
        "$C_s\\dot T_s = \\eta P - h_{sc}(T_s{-}T_c) - h_{sa}(T_s{-}T_{road})$", PHYSICS, 7.5)
    box(ax, 0.41, 0.60, 0.24, 0.13,
        "$C_c\\dot T_c = h_{sc}(T_s{-}T_c) - h_{ca}(T_c{-}T_{air})$", PHYSICS, 7.5)
    box(ax, 0.41, 0.44, 0.24, 0.12, "$\\dot w = \\mathrm{softplus}(f_\\theta) \\geq 0$", PHYSICS, 8)
    box(ax, 0.41, 0.28, 0.24, 0.12,
        "$\\dot g = (1{-}g)R_{form} - g\\,R_{clean}$", PHYSICS, 8)
    box(ax, 0.70, 0.44, 0.13, 0.20, "rate nets\n$R \\geq 0$\n(softplus)", LEARNED, 8)
    box(ax, 0.70, 0.72, 0.13, 0.16,
        "$\\mu_{eff}=\\mu(T_s)e^{-k_w w}(1{-}k_g g)$", PHYSICS, 7)
    box(ax, 0.87, 0.60, 0.10, 0.16, "friction\nellipse", GUARANTEE, 8)
    for p0, p1 in (((0.18, 0.70), (0.21, 0.70)), ((0.37, 0.70), (0.41, 0.82)),
                   ((0.37, 0.70), (0.41, 0.66)), ((0.37, 0.68), (0.41, 0.50)),
                   ((0.65, 0.50), (0.70, 0.50)), ((0.83, 0.80), (0.87, 0.72))):
        arrow(ax, p0, p1)
    arrow(ax, (0.53, 0.28), (0.53, 0.20), ls="--")
    arrow(ax, (0.53, 0.20), (0.765, 0.20), ls="--")
    arrow(ax, (0.765, 0.20), (0.765, 0.72), ls="--", text="condition", dx=0.05, dy=0.15)
    ax.text(0.5, 0.075,
            "$g=0 \\Rightarrow \\dot g \\geq 0$ and $g=1 \\Rightarrow \\dot g \\leq 0$: the interval $[0,1]$ is invariant, with no clamp. "
            "$\\dot w \\geq 0$ makes wear\nirreversible for every weight. The condition only SCALES the ellipse, so P2/P3 still hold.",
            ha="center", fontsize=8, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.4", **{"fc": GUARANTEE["fc"], "ec": GUARANTEE["ec"]}))
    save(fig, "condition_states.png")


def diagram_model_map():
    """Which physics each model contains, as a build-up."""
    fig, ax = canvas(10.0, 4.6, "Model catalogue — what each one contains")
    rows = [
        ("plain MLP", ["", "", "", "", ""], "black box"),
        ("MLP + penalty", ["", "", "~", "", ""], "physics-informed"),
        ("SymmetryTireNet", ["X", "X", "", "", ""], "physics-encoded"),
        ("EncodedTireNet", ["X", "X", "X", "", ""], "physics-encoded"),
        ("ParameterTireNet", ["X", "X", "X", "X", ""], "physics-encoded"),
        ("ResidualTireNet", ["X", "X", "X", "X", ""], "grey box (guided+encoded)"),
        ("RelaxationTireCell", ["X", "X", "X", "~", "X"], "physics-encoded"),
        ("FourWheelVehicle", ["X", "X", "X", "~", "~"], "physics-encoded"),
        ("ThermoGrainingTire", ["X", "X", "X", "~", "X"], "physics-encoded"),
    ]
    cols = ["slip\nkinematics", "odd\nsymmetry", "friction\nenvelope", "tire law\n(MF/brush)", "ODE\ndynamics"]
    x0, w, y0, h = 0.30, 0.093, 0.80, 0.075
    for j, c in enumerate(cols):
        ax.text(x0 + w * (j + 0.5), y0 + 0.075, c, ha="center", va="center",
                fontsize=7.5, weight="bold", linespacing=1.3)
    for i, (name, marks, kind) in enumerate(rows):
        y = y0 - (i + 1) * h
        ax.text(0.285, y + h / 2, name, ha="right", va="center", fontsize=8.5)
        for j, m in enumerate(marks):
            style = GUARANTEE if m == "X" else (LOSS if m == "~" else DATA)
            box(ax, x0 + j * w + 0.004, y + 0.006, w - 0.008, h - 0.012, m, style, 8)
        ax.text(x0 + 5 * w + 0.015, y + h / 2, kind, va="center", fontsize=7.5, color="#555555")
    ax.text(0.5, 0.055, "X = encoded structurally   ~ = present but optional or soft   "
                        "blank = absent",
            ha="center", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.4", fc="#f7f7f7", ec="#999999"))
    save(fig, "model_map.png")


def main():
    print("Generating documentation diagrams:")
    diagram_taxonomy()
    diagram_integration_patterns()
    diagram_encoded_tire()
    diagram_parameter_net()
    diagram_relaxation_cell()
    diagram_four_wheel()
    diagram_condition_states()
    diagram_model_map()
    print("done.")


if __name__ == "__main__":
    main()
