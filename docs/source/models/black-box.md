# Black-box and analytical models

The two ends of the spectrum: models with no physics at all, and models that are nothing
but physics. Both exist so the encoded models have something honest to be compared
against.

## `MagicFormulaTire` — the analytical reference

Parameters are registered as **buffers**, not `Parameter`s: this is the fixed reference,
fitted with `scipy`, not trained by gradient descent.

```python
from tire_nn.physics import MagicFormulaTire, MFParams
from tire_nn.physics.fitting import fit_magic_formula

px, py = fit_magic_formula(train_df)       # bounded least squares on F/Fz residuals
model = MagicFormulaTire(px, py)
assert list(model.parameters()) == []
```

Equations and coefficient roles: [Steady state](../physics/steady-state).
Combined slip: [Combined slip](../physics/combined-slip).

**Strengths.** Zero parameters to train, every guarantee by construction, the format
every vehicle-dynamics tool speaks.
**Weaknesses.** The fit is only as good as the data covers, it cannot use context
(pressure, temperature) without a hand-built extension, and the fit itself is
ill-conditioned on sparse data — on 210 samples it is the *worst* model in the
[benchmark](../comparison/benchmarks).

## `MLPTireModel` — the black-box control

```{math}
(F_x, F_y) = F_z \cdot \mathrm{MLP}\big(\alpha/\alpha_{ref},\ \kappa/\kappa_{ref},\ F_z/F_{z,ref},\ c\big)
```

Deliberately encodes nothing. It exists to quantify what each prior buys and to
demonstrate the failure modes the priors remove.

```python
from tire_nn.models import MLPTireModel
model = MLPTireModel(hidden=(64, 64), context_keys=("p",))
```

`friction_penalty=True` does not change the architecture; it marks the model so the
trainer adds a soft envelope penalty. That is the `mlp_penalty` rung — the
physics-informed control described in [taxonomy](../methods/taxonomy).

```{figure} ../_static/figures/symmetry_zero_slip.png
:alt: untrained MLP vs encoded network near zero slip
:width: 100%

Four random initialisations of each, **untrained**. The MLP predicts a large lateral
force at zero slip and has no symmetry; the encoded network passes through the origin
and is exactly odd for every seed. Training shrinks the MLP's offset but never removes
it, and it returns off-distribution.
```

**Strengths.** Maximum flexibility; a fine interpolator given plenty of clean data —
see the [benchmarks](../comparison/benchmarks), where a well-trained MLP is within 1 N
of the analytical model.
**Weaknesses.** Every guarantee is absent, its violations are worst where data is
thinnest, and it needs roughly an order of magnitude more data to reach the encoded
models' accuracy.

## `GRUTireModel` — sequential black box

A generic recurrent baseline: hidden state $\to$ force, no physical structure. Given
exactly the same inputs and the same `rollout` signature as the relaxation cell, so the
comparison isolates one thing — whether $\tau = \sigma/v$ is encoded or has to be
learned {cite}`cho2014gru`.

## `NeuralODETireModel` — continuous-time black box

```{math}
\dot F = f_\theta(F, \alpha, \kappa, F_z, v_x)
```

Continuous-time like the relaxation cell, but with an unconstrained right-hand side:
nothing forces the time constant to scale with speed, nothing forces the dynamics to be
contractive, and the steady state is only implicit {cite}`chen2018neuralode`.

**Measured outcome** ([dynamics comparison](../comparison/dynamics)): both learn the
training speeds and neither learns a reliable speed law. The Neural ODE consistently
behaves as a fixed *time* constant (rise-distance ratio 3.0); the GRU produced 2.25 in
one run and 0.53 in another — no stable scaling at all.
