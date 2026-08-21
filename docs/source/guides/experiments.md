# Running the experiments

Each experiment is fully described by a YAML config in `configs/` and runs on in-repo
synthetic data by default, so all four are verifiable before any dataset download
succeeds. Override any key from the command line:

```bash
python experiments/train_direct_force.py --config configs/exp1_direct_force.yaml \
       --set training.epochs=50 data.source=kit
```

Outputs land in `results/<experiment>/`: per-model checkpoints, `norm.json`,
`summary.csv`, `learning_curve.csv` and `plots/`.

| experiment | script | question | results |
|---|---|---|---|
| 1 | `train_direct_force.py` | What does each prior buy on steady-state rig data? | [benchmarks](../comparison/benchmarks) |
| 2 | `train_relaxation.py` | Does encoding $\tau = \sigma/v$ beat learning the transient? | [transient](../physics/transient) |
| 3 | `train_vehicle_supervised.py` | Can a tire be identified from IMU signals alone? | [benchmarks](../comparison/benchmarks) |
| 4 | `train_graining.py` | Can the condition structure be identified at all? | [thermal](../physics/thermal-wear) |
| 5 | `train_degradation_ude.py` | Can degradation dynamics be identified from lap times, on real data? | [degradation](../physics/degradation) |
| 6 | `train_vision_condition.py` | Does the encoded ordinal model work on real photographs, and do synthetic textures transfer? | [imaging](../physics/imaging) |

## Experiment 1 — steady-state ablation

Trains the whole ladder and reports accuracy **and** physical violations for each rung,
plus a learning curve over training-set size.

Two friction limits are audited: a deliberately generous $\mu = 1.5$, a violation of
which is unambiguous, and the data's true $\mu = 1.1$, which is what a controller plans
against. A model can look clean against the generous limit while still promising more
grip than the tire has.

```yaml
evaluation:
  audit_alpha_max: 0.6      # audit grid is wider than training on purpose
  audit_mu_generous: 1.5
  audit_mu_tight: 1.1
  learning_curve: true
  extrapolation: slip       # slip | load | none
```

## Experiment 2 — transient

Compares a static net, a GRU, a Neural ODE and the encoded relaxation cell on step tests
in $\alpha$, $\kappa$, $F_z$ and $\mu$, with a **speed holdout**: sequences above
`data.vx_train_max` are never trained on. The headline metric is the rise-distance ratio.

Sequence training rolls the model out over a window and supervises the whole trajectory —
a one-step-ahead loss is dominated by the steady-state map and barely constrains $\tau$.

## Experiment 3 — vehicle-supervised

No tire force is ever observed. Supervision is $a_x$, $a_y$, $\dot r$ only, connected to
one shared tire model by exact Newton–Euler. A whole friction level and tire set are held
out. Optionally warm-started from direct tire data
(`evaluation.pretrain_with_tire_data: true`), which is the realistic workflow: rig data
for one compound, on-track data for the rest.

## Experiment 4 — condition demonstrator

:::{warning}
Synthetic and weakly supervised. Demonstrates the model structure; says nothing about
real motorsport graining.
:::

Generates a four-phase stint, trains the neural rate networks on noisy weak labels for
$T_s$ and $g$, rolls the trained model out from a cold, unworn, clean tire, and **asserts
the structural guarantees on the trained model**.

## Experiment 5 — degradation as a UDE

The only experiment with a **real** dataset attached: Formula 1 stint data via FastF1.

```bash
python -m pip install fastf1
python scripts/download_f1_stints.py --seasons 2023 --rounds 1 2 3 4 5 9 13 17
python experiments/train_degradation_ude.py
python experiments/train_degradation_ude.py --set data.source=synthetic   # known truth
python experiments/train_degradation_ude.py --set data.source=synthetic --seeds 6
```

The `--seeds` sweep is worth running before believing any single result: the lap-time fit
is stable across seeds while the split of degradation between the wear and graining
channels is not.

Compares a linear-in-tyre-age baseline, a black-box MLP and the UDE, holding out whole
races. On synthetic stints, where the latent truth is known, the recovered states are
additionally scored against it. Full discussion, including two modelling decisions the
real data forced, is in [Degradation from stint data](../physics/degradation).

## Experiment 6 — condition from imagery, real and synthetic

```bash
python -m pip install huggingface_hub
python scripts/download_tyre_images.py --limit 250   # ~1850 photos, CC BY 4.0, no account
python experiments/train_vision_condition.py
```

Trains the ordinal vision model on real photographs and on synthetic textures, and
cross-tests. The headline is a negative result worth knowing: the synthetic-trained model
performs at **chance** on real photographs. See [Imaging tyre condition](../physics/imaging).

## Runtime notes

The sequential models dominate cost — a 400-step rollout is 400 forward passes with
autograd. Budgets used for the published numbers:

| experiment | data | epochs | approximate runtime (16-core CPU) |
|---|---|---|---|
| 1 | 6 000 samples | 300 (early stop) | ~25 min including the learning curve |
| 2 | 60 sequences × 400 | 60 | ~20 min |
| 3 | 8 sequences × 1 200 | 150 | ~15 min |
| 4 | 24 000 samples | 40 | ~15 min |
| 5 | 7 400 real laps | 600 | ~4 min |

## Regenerating the documentation assets

```bash
cd docs
python generate_figures.py    # data plots
python diagrams.py            # schematics
make html
```

Both scripts use untrained or analytically parameterised models, so they run in seconds
and need no `results/` directory.
