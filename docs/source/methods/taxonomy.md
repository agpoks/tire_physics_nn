# Physics-guided, physics-informed, physics-encoded

These three terms are used loosely in the literature. This page fixes precise
definitions, because the whole design of this framework rests on the distinction.

```{figure} ../_static/diagrams/taxonomy.png
:alt: three ways to put physics into a neural model
:width: 100%

Where the physics enters: the input, the loss, or the computation itself.
```

## Definitions

**Physics-guided**
: Physical knowledge shapes the *inputs*, the *training signal* or the *evaluation
  protocol*, but no physical equation constrains the network's internal computation and
  none appears in the loss.
: Examples: feeding engineered features $\kappa^2, \alpha^2, F_z/F_{z0}$; normalising by
  load; splitting train/test by operating condition rather than randomly; initialising
  from an analytical fit.
: **Guarantee: none.** The physics is a hint. A guided model can still produce force at
  zero slip.

**Physics-informed** (PINN-style {cite}`raissi2019pinn`)
: The architecture is generic, but a residual of a governing equation is added to the
  loss: $\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda\,\mathcal{L}_{\text{physics}}$.
: **Guarantee: in expectation, on the training distribution, subject to $\lambda$.** The
  constraint is a soft trade-off against the data term. Off-distribution — which is
  where a racing controller operates — nothing holds.

**Physics-encoded**
: The physical property is a consequence of how the output is *computed*, so it holds
  identically for every parameter value.
: Examples: $F_y = -\alpha\,g(\alpha^2,\ldots)$ is odd and vanishes at $\alpha = 0$ for
  any $g$; $\mathrm{softplus}$ makes a rate non-negative for any input; a radial
  projection makes the friction ellipse inviolable.
: **Guarantee: always.** Before training, after training, out of distribution, at any
  operating point, for any weights — including diverged ones.

## The decisive difference, measured

The penalty is not a weak version of the constraint; it is a different kind of object.
From the full-budget Experiment 1 run (identical architecture, seed and budget; only the
loss differs):

| | test $F_y$ RMSE [N] | zero-slip force | odd-symmetry violation | envelope violation ($\mu = 1.1$) |
|---|---|---|---|---|
| plain MLP | 18.61 | 0.016 | 0.076 | 0.065 |
| MLP + friction **penalty** | 18.14 | 0.014 | **0.100** | 0.057 |
| symmetry + hard envelope | 18.55 | **0** | **0** | **0** |

The penalty reduced the violation it targeted by 13 %, and made the *symmetry* violation
worse — because a soft term trades against the data term and against the other soft
terms. The structural version is exactly zero at no cost in RMSE.

## When each is the right choice

Physics-encoding is not always available or appropriate:

| Situation | Use | Why |
|---|---|---|
| The property is an algebraic identity in the output ($F(0)=0$, oddness, a bound, a sign) | **encoded** | free, exact, no tuning |
| The property is a differential relation you can integrate ($\dot F = (F_{ss}-F)/\tau$) | **encoded**, as an ODE layer — a [universal differential equation](integration.md#this-pattern-has-a-name-ude) {cite}`rackauckas2020ude` | exact dynamics and stability, with the unknown kinetics left to a network |
| The physics is a PDE with no closed-form solution, or you need it only on a domain | **informed** | encoding is not available; a residual loss is |
| You have a good analytical model and only need a correction | **encoded + guided** (residual) | small network, interpretable decomposition |
| The property is statistical or approximate ("usually smooth") | **informed** or regularisation | it is not a hard fact, so do not encode it as one |
| You have no physical knowledge you trust | **black box** | be honest about it and report the violations |

The rule this project follows:

> **Never express a constraint in the loss if the architecture can guarantee it.**
> Express it in the loss when the architecture *cannot* — and then report how much it
> is actually violated.

## Where each model in this framework sits

```{figure} ../_static/diagrams/model_map.png
:alt: which physics each model contains
:width: 100%

Every model, and the physics it contains. See the [catalogue](../models/index) for the
equations behind each row.
```

Note the two mixed rows. `ResidualTireNet` is guided (it is initialised from and built
around an analytical fit) *and* encoded (its residual is symmetric and the sum is
projected into the ellipse). `MLP + penalty` is the informed control experiment, kept
deliberately so the comparison above can be made rather than asserted.
