# tire_physics_nn

```{image} _static/logo.svg
:alt: tire_physics_nn
:width: 430px
:align: center
:class: only-light
```

```{image} _static/logo-dark.svg
:alt: tire_physics_nn
:width: 430px
:align: center
:class: only-dark
```

**Physics-encoded neural tire models for autonomous racing and motorsport.**

A research framework for building tire models in which the physics lives in the
*architecture* rather than in the loss — so properties such as "no force at zero slip"
and "never exceed the friction ellipse" hold for every weight vector, everywhere,
including outside the training data.

```{figure} _static/diagrams/taxonomy.png
:alt: physics-guided, physics-informed and physics-encoded
:width: 100%

The three ways physics can enter a learned tire model. This project is built on the
third, and implements the second only as a control experiment.
```

## How to read this documentation

| | |
|---|---|
| **1. The physics** | [Tire physics for modelling](physics/index) — every physical effect available to a tire model, with its equations, its assumptions, plots of what it does, and an honest account of what it gets right and wrong. |
| **2. The method** | [Guided, informed, encoded](methods/taxonomy) — what those three terms mean precisely, and [six concrete integration patterns](methods/integration) for wiring a physical law into a network. |
| **3. The models** | [Model catalogue](models/index) — every model in the framework, its equations, which physics it contains, which category it belongs to, and an architecture diagram. |
| **4. The evidence** | [Comparisons](comparison/dynamics) — measured dynamic behaviour, benchmark tables and a trade-off matrix showing where each approach wins and loses. |

New to the framework? Start with [Getting started](getting-started), then the three
[notebooks](notebooks/01_encoded_tire_force).

## Conventions

SI units throughout, and the **SAE sign convention** ([details](physics/conventions)):

| symbol | meaning | unit |
|---|---|---|
| $\alpha$ | slip angle (positive $\Rightarrow F_y < 0$) | rad |
| $\kappa$ | slip ratio (positive $\Rightarrow$ driving) | – |
| $F_z$ | vertical load, compression positive | N |
| $F_x, F_y$ | longitudinal / lateral force, wheel frame | N |
| $\mu_x, \mu_y$ | friction-ellipse semi-axes | – |
| $\sigma_x, \sigma_y$ | relaxation lengths | m |
| $T_s, T_c$ | surface / core tire temperature | K |
| $w$, $g$ | wear, graining ($g \in [0,1]$) | – |

```{toctree}
:maxdepth: 1
:hidden:
:caption: Start here

getting-started
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Tire physics

physics/index
physics/conventions
physics/steady-state
physics/combined-slip
physics/transient
physics/thermal-wear
physics/degradation
physics/vehicle
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Method

methods/taxonomy
methods/integration
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Models

models/index
models/black-box
models/encoded
models/parameter
models/residual
models/dynamic
models/vehicle
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Comparisons

comparison/dynamics
comparison/benchmarks
comparison/tradeoffs
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Notebooks

notebooks/01_encoded_tire_force
notebooks/02_relaxation_graining_tire_cell
notebooks/03_four_wheel_physics_supervision
notebooks/04_tyre_degradation_ude
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Guides

guides/experiments
guides/datasets
guides/extending
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Reference

bibliography
```
