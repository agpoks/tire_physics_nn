# tire_physics_nn

**Physics-encoded neural tire models for autonomous racing and motorsport.**

This is a research framework in which the physics lives in the *architecture*, not in
the loss. Slip kinematics are computed analytically; odd symmetry and zero force at
zero slip hold exactly for any weights; the friction ellipse is enforced by a
differentiable radial projection rather than a penalty; relaxation is a first-order
ODE in travelled distance; and one shared `TireNet` serves all four corners inside
exact Newton–Euler vehicle equations.

The **Theory** chapters below are a tutorial: each one states the physical problem,
derives the equation, explains why it is encoded structurally rather than penalised,
shows the resulting behaviour in a figure, and points at the code and the test that
guarantees it.

```{figure} _static/figures/violations.png
:alt: physical violations by model
:width: 100%

What the priors buy. Four *untrained* models audited on a grid wider than any
training range. The plain MLP produces force at zero slip, breaks odd symmetry and
pushes force against the slip direction; the symmetry-encoded model fixes the shape
but not the magnitude; only symmetry **plus** the hard envelope satisfies everything.
Bars at the dotted line are exactly zero. Reproduce with
`python docs/generate_figures.py`.
```

```{toctree}
:maxdepth: 2
:caption: Project

readme
plan
datasets
bibliography
```

```{toctree}
:maxdepth: 2
:caption: Theory tutorial

theory/00_overview
theory/01_slip_kinematics
theory/02_symmetry
theory/03_friction_envelope
theory/04_parameter_network
theory/05_relaxation
theory/06_four_wheel
theory/07_thermal_wear_graining
```

```{toctree}
:maxdepth: 2
:caption: Using the framework

tutorials/quickstart
tutorials/experiments
```

```{toctree}
:maxdepth: 1
:caption: Notebooks

notebooks/01_encoded_tire_force
notebooks/02_relaxation_graining_tire_cell
notebooks/03_four_wheel_physics_supervision
```

## Conventions used throughout

All equations use SI units and the **SAE sign convention** (see
{doc}`theory/01_slip_kinematics`):

| symbol | meaning | unit |
|---|---|---|
| $\alpha$ | slip angle (positive $\Rightarrow$ $F_y < 0$) | rad |
| $\kappa$ | slip ratio (positive $\Rightarrow$ driving) | – |
| $F_z$ | vertical load, compression positive | N |
| $F_x, F_y$ | longitudinal / lateral tire force, wheel frame | N |
| $\mu_x, \mu_y$ | friction-ellipse semi-axes | – |
| $\sigma_x, \sigma_y$ | relaxation lengths | m |
| $T_s, T_c$ | surface / core tire temperature | K |
| $g$ | graining state, $g\in[0,1]$ | – |

## Indices

* {ref}`genindex`
* {ref}`modindex`
