# Benchmark results

All numbers below are measured, reproducible with the commands shown, and come from
**synthetic** data unless stated. Read the caveat at the bottom of this page before
quoting any of them.

## Experiment 1 — steady-state force modelling

`python experiments/train_direct_force.py` — 6 000 synthetic samples, 300 epochs,
identical optimiser, seed and budget for every rung. RMSE in newtons on a tire loaded to
~1.4 kN; violations are load-normalised and measured on a grid wider than the training
range.

| model | params | test $F_y$ RMSE | extrap. RMSE | zero-slip force | symmetry | envelope ($\mu{=}1.1$) |
|---|---|---|---|---|---|---|
| Magic Formula (fitted) | 0 | 17.71 | 17.87 | 0 | 0 | 0 |
| plain MLP | 4 546 | 18.61 | 18.32 | 0.016 | 0.076 | 0.065 |
| MLP + friction **penalty** | 4 546 | 18.14 | 18.10 | 0.014 | 0.100 | 0.057 |
| symmetry-encoded | 1 250 | 17.61 | 17.60 | **0** | **0** | 0.426 |
| symmetry + hard envelope | 1 254 | 18.55 | 19.21 | **0** | **0** | **0** |
| ParameterNet + Magic Formula | 1 483 | **17.39** | **17.49** | **0** | **0** | **0** |
| residual grey box | 1 254 | 17.47 | 17.48 | **0** | **0** | **0** |

Three readings, one of which is a warning:

1. **Accuracy is close to a wash** — 17.4 to 18.6 N across every rung. On clean,
   plentiful, in-distribution data a well-trained MLP is a perfectly good interpolator.
   What separates the rungs is the guarantee columns, not the RMSE column. If a paper
   reports a large accuracy win from physics encoding on data like this, check whether
   the baseline was trained to convergence — an earlier version of *this* table was
   wrong for exactly that reason.
2. **The penalty does not deliver the constraint.** It cut the violation it targeted by
   13 % and made the odd-symmetry violation *worse*, because a soft term trades against
   the data term. The structural version is exactly zero at no cost in RMSE.
3. **Symmetry without the envelope is the worst case for the friction limit** (0.426),
   worse than the plain MLP. Correct shape, unbounded magnitude.

## Data efficiency

```{figure} ../_static/figures/learning_curve.png
:alt: test RMSE vs training set size
:width: 70%
```

| training samples | Magic Formula (fitted) | plain MLP | symmetry | residual | encoded | ParameterNet |
|---|---|---|---|---|---|---|
| 210 | 196.9 | 95.2 | 75.3 | 25.2 | 20.6 | **17.4** |
| 570 | 17.6 | 20.3 | 42.8 | 19.0 | 19.0 | **17.3** |
| 1 547 | 17.6 | 18.4 | 26.8 | 18.1 | 18.8 | **17.3** |
| 4 200 | 17.7 | 18.1 | 17.7 | 17.5 | 18.6 | **17.4** |

**This is where the priors win, and it is the only place the gap is large.** The encoded
models are near their final accuracy from 210 samples; the plain MLP needs roughly an
order of magnitude more data.

The surprise is the first column: the *analytical* baseline is not automatically
data-efficient. A direct `scipy` fit of the Magic Formula is the **worst** model at 210
samples (196.9 N) — five free parameters on sparse noisy data is ill-conditioned, and
$B$ and $C$ are only jointly identifiable — and it recovers only by 570.
`ParameterTireNet` contains the *same* Magic Formula but reaches its coefficients through
bounded transforms, and is stable at every size. The bounded parameterisation is doing
real regularisation work, not just keeping the parameters legal.

## Experiment 2 — transient

`python experiments/train_relaxation.py`

| model | rise-distance ratio (run 1) | (run 2) | rollout $F_y$ RMSE [N] |
|---|---|---|---|
| static tire net | – | – | 293 |
| generic GRU | 2.25 | 0.53 | 203 |
| generic Neural ODE | 3.00 | 3.00 | 252 |
| **relaxation cell** | **1.25** | **1.15** | **110** |
| **relaxation + ParameterNet** | **0.92** | – | – |

The encoded cell is both the most accurate and the only one with a stable speed law. It
recovers $\sigma_y \approx 0.25$ m against a true $0.30$ m from a few epochs of step
tests. Discussion: [Transient](../physics/transient).

## Experiment 3 — vehicle-supervised

`python experiments/train_vehicle_supervised.py` — trained on IMU signals only at
$\mu \in \{1.0, 0.85\}$, tested on an unseen $\mu = 0.65$ and an unseen tire set.

| model | params | val $a_y$ RMSE | unseen-$\mu$ $a_y$ RMSE |
|---|---|---|---|
| fitted Magic Formula (never sees vehicle data) | 0 | 1.04 | 1.59 |
| shared TireNet | 1 323 | 0.59 | 1.84 |
| per-wheel nets (ablation) | 5 292 | 0.62 | 1.92 |
| shared ParameterNet | 1 548 | **0.38** | **1.12** |

The per-wheel ablation uses four times the parameters and is worse on both. The most
constrained model generalises best to the friction level it never saw.

## Experiment 4 — condition states

`python experiments/train_graining.py` — a synthetic demonstrator. The trained model
reproduces cold-start graining formation, clean-up on warm-up and monotone wear, and the
structural guarantees are asserted on the trained model:

```text
[checks] wear_monotone=True  graining_in_unit_interval=True
         temperatures_finite=True  slip_power_non_negative=True
```

:::{warning}
**All of the above is synthetic.** The steady-state data is Magic-Formula generated with
clean Gaussian noise on a dense slip grid — kinder than any real rig, and structurally
matched to `ParameterTireNet`, so its win is expected rather than evidence. The graining
experiment is weakly supervised synthetic data and says nothing about a real tire. Re-run
on real measurements (`--set data.source=kit`, see [Datasets](../guides/datasets)) before
drawing any conclusion about relative accuracy.
:::
