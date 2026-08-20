# 0. Why encode physics in the architecture?

## The problem with a tire model that is only "accurate"

A tire model for a racing controller is not judged by its RMSE on a test set. It is
judged by what it does at the operating point the controller drives it to — the
friction limit — and by whether the optimiser inside an MPC can trust its shape.

Three failure modes matter, and none of them show up in an interpolation RMSE:

1. **Force at zero slip.** A black-box network fitted to tire data will generally
   predict $F_y(\alpha=0)\neq 0$. Linearise an MPC about straight-line running and
   that offset becomes a phantom steering command.
2. **Broken symmetry.** $F_y(-\alpha) \neq -F_y(\alpha)$ means the model has learned a
   left-turn tire and a different right-turn tire from the same physical object,
   usually because one direction had more data.
3. **Forces outside the friction ellipse.** A network can happily predict
   $\sqrt{F_x^2+F_y^2} > \mu F_z$. An optimiser *searches for the best achievable
   force*, so it will find exactly the region where the model is unphysically
   optimistic and plan a lap that the car cannot drive.

All three get worse where data is sparse: near zero slip, at the extremes, and outside
the training envelope.

## Three ways to bring physics in

:::{list-table}
:header-rows: 1
:widths: 18 42 40

* - Approach
  - How the physics enters
  - Failure mode
* - **Physics-guided**
  - Engineered features, physical baselines, physics-informed data splits. The
    network is still a black box.
  - Nothing is guaranteed; the guidance is a hint.
* - **Physics-informed** (PINN-style {cite}`raissi2019pinn`)
  - A residual of the governing equation is added to the loss.
  - Satisfied *in expectation on the training distribution*. Silently violated
    off-distribution — which is exactly where a racing controller operates. The
    trade-off against RMSE is set by an arbitrary weight.
* - **Physics-encoded** (this project)
  - The constraint is built into the computation, so it holds for *every* weight
    vector, before, during and after training.
  - Only expresses constraints that can be written structurally; the rest still needs
    data.
:::

The project rule follows directly:

> **Never hide a physical constraint in the loss if it can be encoded exactly in the
> architecture.**

The penalty version is still implemented — as the `mlp_penalty` rung of the ablation
ladder — precisely so the difference can be measured rather than asserted.

## The ablation ladder

Each rung adds one prior. Everything else (data, optimiser, seed, budget) is held
fixed, so a difference in the results is attributable to the prior.

| Rung | Module | Encoded |
|---|---|---|
| Magic Formula (fitted with `scipy`) | {py:mod}`tire_nn.physics.pacejka` | all — analytical reference |
| plain MLP | {py:mod}`tire_nn.models.mlp_tire` | nothing |
| MLP + friction **penalty** | same architecture, extra loss term | envelope, softly |
| symmetry-encoded net | {py:mod}`tire_nn.models.encoded_tire` | P1, P2 |
| symmetry + **hard** envelope | {py:mod}`tire_nn.models.encoded_tire` | P1, P2, P3 |
| ParameterNet + Magic Formula | {py:mod}`tire_nn.models.parameter_tire` | P1, P2, P4 |
| residual grey-box | {py:mod}`tire_nn.models.residual_tire` | analytical prior + bounded residual |
| relaxation cell | {py:mod}`tire_nn.models.relaxation_tire` | P5 |
| four-wheel vehicle | {py:mod}`tire_nn.models.four_wheel_vehicle` | P6 |
| thermal / wear / graining | {py:mod}`tire_nn.models.thermo_graining_tire` | P7 |

## The seven priors

| | Prior | Chapter |
|---|---|---|
| P1 | Slip kinematics are computed, never learned | {doc}`01_slip_kinematics` |
| P2 | Odd symmetry and zero force at zero slip, by construction | {doc}`02_symmetry` |
| P3 | Hard friction envelope by differentiable radial projection | {doc}`03_friction_envelope` |
| P4 | Bounded physical parameters through a differentiable tire law | {doc}`04_parameter_network` |
| P5 | Relaxation as a first-order ODE in travelled distance | {doc}`05_relaxation` |
| P6 | One shared tire model, exact Newton–Euler aggregation | {doc}`06_four_wheel` |
| P7 | Thermal / wear / graining states with structural irreversibility and bounds | {doc}`07_thermal_wear_graining` |

## Strict layering

```text
tire_nn/physics/     pure, differentiable, NO learnable parameters
tire_nn/layers/      constrained building blocks
tire_nn/models/      compositions of layers + small MLPs
tire_nn/data/        adapters -> one canonical schema
tire_nn/training/    losses, metrics, deterministic trainer
tire_nn/evaluation/  plots, extrapolation protocol, consistency audit
```

`physics/` never imports from `models/`, and `models/` never re-implements an equation
that already exists in `physics/`. This is what makes the ablations honest: every rung
of the ladder evaluates the *same* analytical core.
