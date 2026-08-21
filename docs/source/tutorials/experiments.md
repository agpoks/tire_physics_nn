# Running the experiments

Every experiment is fully described by a YAML config in `configs/` and runs on
in-repo synthetic data by default, so all four are verifiable before any dataset
download succeeds. Override any key from the command line:

```bash
python experiments/train_direct_force.py --config configs/exp1_direct_force.yaml \
       --set training.epochs=50 data.source=kit
```

Outputs land in `results/<experiment>/`: per-model checkpoints, `summary.csv`,
`learning_curve.csv` and `plots/`.

## Experiment 1 — direct tire-force modelling

```bash
python experiments/train_direct_force.py
```

Trains the whole ablation ladder on one dataset and reports accuracy **and** physical
violations for each rung: RMSE, MAE, $R^2$, extrapolation RMSE, odd-symmetry
violation, zero-slip force, envelope violation, dissipativity violation, and the
learning curve over training-set size.

Full-budget run on synthetic data (6 000 samples, 300 epochs, identical
optimiser/seed/budget per rung). RMSE in newtons on a tire loaded to ~1.4 kN;
violations are load-normalised, and `envelope (μ=1.1)` is measured against the **true**
friction limit of the data — what a controller would actually plan against, as opposed
to the deliberately generous μ=1.5 limit a violation of which is unambiguous:

| model | params | test $F_y$ RMSE | extrap. RMSE | zero-slip force | symmetry | envelope (μ=1.1) |
|---|---|---|---|---|---|---|
| Magic Formula (fitted) | 0 | 17.71 | 17.87 | 0 | 0 | 0 |
| plain MLP | 4 546 | 18.61 | 18.32 | 0.016 | 0.076 | 0.065 |
| MLP + friction **penalty** | 4 546 | 18.14 | 18.10 | 0.014 | 0.100 | 0.057 |
| symmetry-encoded | 1 250 | 17.61 | 17.60 | **0** | **0** | 0.426 |
| symmetry + hard envelope | 1 254 | 18.55 | 19.21 | **0** | **0** | **0** |
| ParameterNet + Magic Formula | 1 483 | **17.39** | **17.49** | **0** | **0** | **0** |
| residual grey-box | 1 254 | 17.47 | 17.48 | **0** | **0** | **0** |

Four things to read from this table, one of which is a warning:

1. **Accuracy is close to a wash here** — 17.4 to 18.6 N across every rung. On clean,
   plentiful, in-distribution data a well-trained MLP is a perfectly good interpolator.
   What separates the rungs is the guarantee columns, not the RMSE column. If a paper
   reports a large accuracy win from physics encoding on data like this, check whether
   the baseline was trained to convergence.
2. **The penalty does not deliver the constraint.** It reduced the violation of the true
   friction limit by 13 % and made the odd-symmetry violation *worse* (0.076 → 0.100),
   because a soft term trades against the data term. The structural projection is
   exactly zero for every weight vector, at no cost in RMSE.
3. **Symmetry without the envelope is the worst case for the friction limit** (0.426).
   A correct shape with an unconstrained magnitude is not a safe model.
4. **Data efficiency is where the priors win decisively.** Test $F_y$ RMSE at 210
   training samples: ParameterNet 17.4, encoded 20.6, residual 25.2, symmetry-only 75.3,
   plain MLP 95.2. That is the regime a real tire programme lives in — rig time is
   expensive.

:::{warning}
This data is Magic-Formula generated with clean Gaussian noise on a dense slip grid.
That is kinder than any real rig, and it is structurally matched to `ParameterTireNet`,
so its win is expected. Re-run on real measurements (`--set data.source=kit`) before
drawing any conclusion about relative accuracy.
:::

## Experiment 2 — transient behaviour

```bash
python experiments/train_relaxation.py
```

Compares a static tire net, a generic GRU, a generic Neural ODE and the encoded
relaxation cell on step tests in $\alpha$, $\kappa$, $F_z$ and $\mu$, with a **speed
holdout**: sequences above `data.vx_train_max` are never trained on.

The headline metric is the rise-distance ratio described in
{doc}`../theory/05_relaxation` — see that chapter for the result table.

## Experiment 3 — vehicle-supervised learning

```bash
python experiments/train_vehicle_supervised.py
```

No tire force is ever observed. Supervision is $a_x$, $a_y$ and $\dot r$ only,
connected to a single shared tire model by the exact Newton–Euler equations. A whole
friction level and tire set are held out. Optionally warm-started from direct tire data
(`evaluation.pretrain_with_tire_data: true`), which is the realistic workflow: rig data
for one compound, on-track data for the rest.

Results are in {doc}`../theory/06_four_wheel`.

## Experiment 4 — thermal / wear / graining demonstrator

```bash
python experiments/train_graining.py
```

:::{warning}
Synthetic and weakly supervised. Demonstrates the model structure; says nothing about
real motorsport graining.
:::

Generates a four-phase stint, trains the neural rate networks on noisy weak labels for
$T_s$ and $g$, rolls the trained model out from a cold, unworn, clean tire, and asserts
the structural guarantees. See {doc}`../theory/07_thermal_wear_graining`.

## Regenerating the documentation figures

```bash
cd docs && make figures && make html
```

`generate_figures.py` uses untrained or analytically parameterised models, so it runs
in seconds and needs no results directory.
