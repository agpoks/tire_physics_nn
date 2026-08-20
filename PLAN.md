# PLAN.md — tire_physics_nn

**Physics-Encoded Neural Tire Models for Autonomous Racing and Motorsport**

Status: living document. Written 2026-08-20, before any modeling code was committed. Update it whenever a module interface changes, a dataset turns out to be unavailable, or an architectural decision is taken — it is the single source of truth for *why* the framework is structured the way it is.

---

## 0. Repository inspection result

`/home/poxx/github` was inspected before writing this plan. It contains ~46 unrelated robotics/racing projects. **No existing physics-encoded tire-learning project was found**, so this is a greenfield repository. Nothing outside `tire_physics_nn/` is modified by this project.

Relevant neighbours found (inspected, *not* imported as runtime dependencies — see §1):

| Path | What it contains | How it is used here |
|---|---|---|
| `scuderia_gymnasium/gym/scuderia_gym/envs/dynamic_models.py` | Numba/NumPy **full Pacejka (MF 32-param) combined slip**, simplified MF (14-param), **brush**, **Dugoff** tire models; `vehicle_dynamics_std4w` with exact 4-corner slip kinematics, load transfer, Newton–Euler aggregation | **Reference implementation** for the analytical baselines. Formulas and sign conventions are ported to differentiable PyTorch in `tire_nn/physics/`. Not imported (numba-jitted, NumPy scalars, not autograd-compatible). |
| `scuderia_gym/…/dynamic_models.py` | Older variant of the same | Cross-check only |
| `On-Track-SysID/params/pacejka_params.yaml`, `src/on_track_sys_id.py` | Fitted 4-param Pacejka `[B,C,D,E]` front/rear for an F1TENTH-class car; on-track NN system-ID loop | Realistic **parameter bounds** and initialisation for `ParameterTireNet`; residual-learning baseline pattern |
| `Tire_Parameter_and_Uncertainty_Estimation-main/src/{utils/tiremodels.py,param_fitting/param_fitting.py}` | JAX `mf_simple` + Nelder–Mead and SVI Magic-Formula fitting, `calc_force_shift` | Blueprint for `scripts/fit_magic_formula.py` (re-done with `scipy.optimize`, no JAX dependency) |
| `greyid/`, `nlgreyfast_py/` | Grey-box system-ID (multiple-shooting, Gauss–Newton, homotopy), `models/vehicle.py`, `models/bicycle.py` | Reference for the vehicle-level training loop (Experiment 3); optional future integration |
| `nnodely-applications/vehicle/*` | `nnodely` model-structured NN examples (lateral/longitudinal vehicle dynamics, friction-aware ABS) | Comparison baseline for Experiment 1/2 (optional dependency, see §1) |
| `neurelux/` | Same author's physics-encoded-NN research template (PLAN.md → notebooks → `scripts/download_*.py` → tests) | **Repository conventions** are deliberately mirrored |
| `liquid-nn-playground/liquid_playground/data/datasets.py` | Cached auto-download loaders returning plain tensors | Pattern for `tire_nn/data/` adapters + `data_cache` semantics |
| `diplomarbeit_schuster-gymnasium_branch/env/env_parts/pacejka_tire.py` | Simple Pacejka env part | Cross-check |

## 1. Dependencies — what is already available

Verified in the active environment (Python **3.10.12**):

| Package | Version | Status |
|---|---|---|
| torch | 2.13.0+cu130 | present — primary framework |
| numpy | 1.26.4 | present |
| scipy | 1.15.3 | present — Magic-Formula fitting |
| pandas | 2.2.3 | present — dataset adapters |
| matplotlib | 3.10.9 | present |
| scikit-learn | 1.7.2 | present — splits/metrics |
| pyyaml | ok | present — configs |
| **torchdiffeq** | **missing** | *optional*. Neural-ODE path guarded by `try: import torchdiffeq`; fixed-step Euler/RK4 fallback is always available and is the default. |
| **nnodely** | **missing** | *optional*. Only used by the comparison baseline in Experiment 1/2; skipped with a clear message when absent. |

Design rule: **the framework must run fully with the packages already installed.** Optional imports never appear at module import time in `tire_nn/` core code.

---

## 2. Architecture

### 2.1 Layering (strict)

```
physics/     pure, differentiable, NO learnable parameters
             (Pacejka, brush, thermal, wear, Newton-Euler, slip kinematics)
layers/      reusable constrained building blocks (symmetry, envelope, bounded params)
models/      compositions of layers + small MLPs  -> the learned models
data/        dataset adapters -> one canonical schema
training/    losses, trainer, metrics
evaluation/  plots, extrapolation protocol, physical-consistency checks
```

`physics/` must never import from `models/`. `models/` never re-implements an equation that exists in `physics/`. This is what makes the ablations honest: every model in §2.3 sees the *same* analytical core.

### 2.2 Core interface (all models implement it)

```python
class BaseTireModel(nn.Module):
    def forward(
        self,
        alpha: Tensor,           # (B,) slip angle [rad]
        kappa: Tensor,           # (B,) slip ratio [-]
        Fz:    Tensor,           # (B,) vertical load [N], > 0
        context: dict[str, Tensor] | None = None,
    ) -> TireForces: ...
```

`TireForces` is a small dataclass: `Fx, Fy` (required, N), optional `Mz` (Nm), `mu_x`, `mu_y`, `params` (dict of the physically-meaningful quantities the model exposes — always populated by `ParameterTireNet`, possibly empty for the plain MLP).

`context` is an **open dict** with a fixed, documented key vocabulary; every model declares which keys it consumes and ignores the rest:

| key | unit | meaning |
|---|---|---|
| `vx` | m/s | wheel-centre longitudinal velocity (needed for relaxation `tau`) |
| `Ts`, `Tc` | K | tire surface / core temperature |
| `p` | Pa | inflation pressure |
| `gamma` | rad | camber (inclination) angle |
| `mu_est` | - | externally estimated road friction |
| `wear`, `graining` | - | latent condition states (0..1 for graining) |
| `tire_id` | int64 | index into a learned tire/context embedding |

Rationale for a dict instead of a fixed tensor: the six target datasets expose *different* subsets (VeTyT has camber+pressure but no temperature; KIT has no camber; Q-Motion is pressure-centric). A fixed input vector would force fabricated fill values into the model. Missing keys are handled by an explicit `ContextEncoder` with per-key learned "missing" embeddings, never by silent zero-filling.

### 2.3 Model zoo (= the ablation ladder)

| Module | Class | Encoded physics | Role |
|---|---|---|---|
| `physics/pacejka.py` | `MagicFormula` (fitted, no NN) | all | analytical baseline |
| `models/mlp_tire.py` | `MLPTireModel` | none | black-box baseline |
| `models/mlp_tire.py` | `MLPTireModel(friction_penalty=True)` | envelope **as loss only** | shows why a penalty is not enough |
| `models/encoded_tire.py` | `SymmetryTireNet` | P1 + P2 | odd symmetry + zero-slip exact |
| `models/encoded_tire.py` | `EncodedTireNet` | P1 + P2 + P3 | + **hard** friction envelope |
| `models/parameter_tire.py` | `ParameterTireNet` | P1 + P4 (+P3) | NN predicts `mu, C_alpha, C_kappa, B,C,D,E, sigma` → differentiable MF |
| `models/residual_tire.py` | `ResidualTireNet` | analytical + bounded residual | grey-box, residual capped by envelope |
| `models/relaxation_tire.py` | `RelaxationTireCell` | P5 | first-order transient dynamics |
| `models/thermo_graining_tire.py` | `ThermoGrainingTire` | P7 | latent `[Ts, Tc, wear, graining]` |
| `models/four_wheel_vehicle.py` | `FourWheelVehicle` | P6 | one shared `TireNet`, exact Newton–Euler |

Sequence baselines for Experiment 2 (`models/baselines_seq.py`): `GRUTireModel`, `NeuralODETireModel`.

---

## 3. Physics priors — decision and justification

Every prior below is stated with *why it is encoded structurally rather than penalised*. This section is the required "document why before implementing".

### P1 — Slip kinematics are computed, not learned (`layers/slip_kinematics.py`)

```
kappa = (R_e * omega - v_x) / max(|v_x|, v_eps)          (SAE driving-positive)
alpha = atan2(v_y, |v_x|) - delta                         (wheel frame)
```

**Why:** slip is a *definition*, not a physical uncertainty — it follows exactly from rigid-body kinematics and measured wheel speed. Learning it would spend capacity on an identity and destroy interpretability of everything downstream. The only modelling choice here is the low-speed regularisation `v_eps`, which is a documented numerical constant, and the sign convention (§4.1). A `SlipKinematics` module also gives Experiment 3 a differentiable path from raw wheel speeds/steering to slip, so vehicle-level gradients reach `TireNet`.

### P2 — Odd symmetry by construction (`layers/symmetry.py`)

```
Fx = kappa * gx(kappa^2, alpha^2, Fz, c)
Fy = -alpha * gy(alpha^2, kappa^2, Fz, c)
```

**Why:** an isotropic, symmetric tire on a symmetric road has `Fy(-alpha) = -Fy(alpha)` and `Fx(-kappa) = -Fx(kappa)`, and, crucially, `F(0) = 0` — a tire at zero slip transmits no shear force. A plain MLP violates both, and it violates them *worst where data is sparsest* (near zero and at the extremes), producing the notorious non-zero force at zero slip that destabilises an MPC solved around a straight-line operating point. Multiplying by the odd factor `kappa` / `-alpha` and feeding only the **even** invariants `kappa^2, alpha^2` into the network makes both properties exact to machine precision for *any* network weights — it cannot be trained away, cannot degrade off-distribution, and costs nothing. `gx, gy` are made positive with `softplus` so the force never points against the slip direction (a passivity/dissipativity requirement: `Fx*kappa >= 0`, `Fy*alpha <= 0`).

Asymmetric effects that *do* exist (ply steer, conicity, camber thrust) are deliberately **not** folded into `g`. They enter as an explicit, optional, additive offset term `S_v(Fz, gamma, c)` that is (a) separable, (b) individually reportable, and (c) switchable off — so the symmetry test still passes for the base term, and the asymmetry stays physically named instead of hidden in the weights.

### P3 — Hard friction envelope (`layers/friction_envelope.py`)

Given a raw force `(qx, qy)` and limits `mu_x*Fz, mu_y*Fz`, define the normalised radius
`rho = sqrt((qx/(mu_x Fz))^2 + (qy/(mu_y Fz))^2)` and apply a **smooth radial projection**

```
s(rho) = tanh(rho) / (rho + eps)        (default, C-infinity, s(0)=1)
Fx = qx * s(rho),  Fy = qy * s(rho)
```

**Why `tanh` and not a hard clip:** `s` must be (i) *strictly* bounding — `rho * tanh(rho)/rho = tanh(rho) < 1` guarantees the ellipse constraint for **all** inputs and all weights, (ii) differentiable everywhere (a `min(1, 1/rho)` clip has zero gradient outside the ellipse, which kills learning exactly where the saturating data lives, and a kink on the ellipse that a Newton-based MPC will find), and (iii) the identity near the origin so the linear-range cornering stiffness is untouched (`s -> 1 - rho^2/3`). A softer alternative `rho/sqrt(1+rho^2)` is provided as `mode="algebraic"`.
The scaling is **radial**, i.e. it preserves the force *direction* — this is the standard friction-circle assumption (the shear stress vector opposes the slip velocity vector), so combined-slip behaviour degrades gracefully instead of clipping one component and leaving the other.
**A penalty loss cannot deliver this.** A penalty is (a) satisfied only in expectation over the training distribution, (b) silently violated on extrapolation — which is exactly the operating point an aggressive racing controller drives to — and (c) a soft constraint traded off against RMSE by an arbitrary weight. The framework still *measures* violation for the penalty-only ablation, to quantify the gap.

`mu_x, mu_y` come from a `BoundedParameter` head (P4), so the envelope itself is learnable but always inside `[mu_min, mu_max]`, and is modulated by tire condition (P7).

### P4 — Parameter network (`layers/bounded_parameters.py`, `models/parameter_tire.py`)

The NN outputs *unconstrained* reals; a fixed monotone map sends them to a physical range:

```
positive: p = p_min + softplus(z)              (mu, B, D, C_alpha, C_kappa, sigma)
bounded:  p = p_min + (p_max - p_min)*sigmoid(z)   (C, E, mu)
```

They are then passed through the **differentiable Magic Formula** in `physics/pacejka.py` (never re-implemented in the model). Load dependence uses the standard `D = mu(Fz) * Fz` with a decreasing `mu(Fz)` (load sensitivity), rather than letting `D` float freely.

**Why:** this is the interpretable end of the ladder. The output is a *tire*, not just a fit: `mu`, cornering stiffness `C_alpha = B*C*D`, and relaxation length `sigma` can be read off, plotted, compared with the fitted analytical baseline, and handed to an MPC that expects Pacejka coefficients. Bounded transforms make the parameters valid **by construction** at every training step, so the model never passes through a non-physical region (e.g. `C < 0`, which flips the curve, or `D < 0`, negative peak force) where the loss landscape is meaningless and gradients are misleading. Clipping in the loss would leave the raw parameters free and only mask the symptom.

### P5 — Relaxation dynamics (`models/relaxation_tire.py`)

```
tau_i = sigma_i / (|v_x| + eps),   dF_i/dt = (F_i,ss - F_i)/tau_i,  i in {x, y}
```
with `sigma_i = sigma_min + softplus(z_i)` (positive by construction, so `tau > 0` and the ODE is unconditionally stable/contractive).

**Why:** tire force does not respond instantaneously to a slip step; the contact patch must deform, and the transient is *travelled-distance*-parameterised, not time-parameterised — hence `tau = sigma/v`. This single structural fact is what a GRU has to discover from data (and typically discovers only for the speeds it saw). Encoding it means the learned quantity is one interpretable length per axis (typically 0.1–0.6 m for a passenger tire, shorter for small-scale racing tires) instead of a gate matrix, and the model extrapolates across speed for free. `F_ss` is any static `TireNet` from §2.3, so relaxation composes with every other prior.
Integration: fixed-step **Euler** and **RK4** written explicitly (`training/` uses these by default; step size comes from the data's sample time and is checked against `tau_min`), plus an optional `torchdiffeq.odeint_adjoint` path for stiff/irregular sampling. Both must agree on a smooth test trajectory — that is a unit test.

### P6 — Four-wheel structure (`models/four_wheel_vehicle.py`)

**One** `TireNet` instance is evaluated four times (batched over the wheel axis). Per-corner differences enter only through *physical* inputs: `Fz_i` (static + load transfer), `alpha_i, kappa_i` from the exact corner kinematics `v_i = v_cog + omega x r_i`, and an optional per-axle/per-corner **embedding** of small dimension when the config explicitly asks for it.

**Why:** four independent networks have 4x the parameters, cannot share the (scarce) evidence about the friction limit that only one corner visits at a time, and can absorb chassis errors (wrong `Iz`, wrong load transfer) into per-wheel weights — producing a model that fits the training laps and is wrong about the tire. Sharing turns each vehicle-level sample into four constraints on the *same* constitutive law. It is also physically right: the four tires are (nominally) the same product.

Aggregation is the exact rigid-body model, no learned correction:

```
Fx_body_i = Fx_i*cos(delta_i) - Fy_i*sin(delta_i)
Fy_body_i = Fx_i*sin(delta_i) + Fy_i*cos(delta_i)
m(dvx/dt - r*vy) = sum_i Fx_body_i (+ F_drag, F_roll if configured)
m(dvy/dt + r*vx) = sum_i Fy_body_i
Iz*dr/dt         = sum_i (x_i*Fy_body_i - y_i*Fx_body_i)
```

with `(x_i, y_i)` the exact geometric corner positions (`x = +lf/-lr`, `y = +t/2 / -t/2`). Training supervises `(ax, ay, dr/dt)` — i.e. **IMU-level** supervision — so the tire model is identified without ever measuring a tire force. Load transfer (quasi-static, longitudinal + lateral) lives in `physics/vehicle_dynamics.py` and is shared with the analytical baseline.

### P7 — Thermal / wear / graining (`models/thermo_graining_tire.py`, `physics/thermal.py`, `physics/wear.py`)

Latent state `z = [Ts, Tc, wear, graining]`, with the **structure fixed and only the rates learned**:

```
P_slip = -(Fx*vsx + Fy*vsy)                     >= 0 for a dissipative tire
Cs*dTs/dt = eta*P_slip - h_sc*(Ts-Tc) - h_sa*(Ts-T_road)
Cc*dTc/dt = h_sc*(Ts-Tc) - h_ca*(Tc-T_air)
dwear/dt  = softplus(f_w(...))                  >= 0, irreversible
dg/dt     = (1-g)*R_form(...) - g*R_clean(...)  R_* = softplus(.) >= 0
mu_eff    = mu_base(Ts, Fz, p) * exp(-kw*wear) * (1 - kg*g)
```

**Why structural:** (i) the two-node surface/core split is the minimal model that reproduces the observed fact that surface temperature drives grip while core temperature drives the slow drift over a stint; (ii) `P_slip = -F·v_slip` is the *only* correct energy input — a network that learns "heating from speed" gets braking and cornering wrong; (iii) wear irreversibility is a thermodynamic one-way street, and `softplus` makes `dwear/dt >= 0` exact rather than probable, so `wear` can never "heal" during a bad gradient step; (iv) the graining ODE has the algebraic property that `g in [0,1]` is **invariant**: at `g = 0` the sink term vanishes and at `g = 1` the source term vanishes, so with non-negative rates the state can never leave the interval — no clamping, no penalty, and the gradient stays informative at the boundary. Gating (cold + high slip energy → formation; hot + in-window → cleaning) is imposed as *monotone* inputs to the rate networks, not hard-coded curves, so the shape stays learnable while the sign of the effect does not.

This whole block is **optional** (`enable_thermal`, `enable_wear`, `enable_graining` flags). Experiment 4 uses **synthetic** graining states, and the README/notebook state explicitly that it is a demonstrator, not validated real motorsport graining.

---

## 4. Canonical data format

### 4.1 Conventions (fixed project-wide, converted at adapter level)

- **SAE-style sign convention**, matching the target parameterisation `Fy = -alpha*gy(...)`: positive slip angle → negative lateral force; positive slip ratio → driving/positive `Fx`. Every adapter declares the source convention in its metadata and converts; `tire_nn/data/common.py` has the single conversion helper. (`scuderia_gymnasium` uses CommonRoad braking-positive slip — flagged when its data is used.)
- SI units everywhere: rad, N, Nm, m/s, Pa, K. Adapters convert deg→rad, bar→Pa, °C→K.
- `Fz > 0` (compression positive).

### 4.2 Sample schema (one row = one measurement instant)

Stored as **Parquet** (fallback CSV), one file per source + condition, plus a sidecar `meta.yaml`.

| column | unit | required | note |
|---|---|---|---|
| `alpha` | rad | yes | slip angle |
| `kappa` | - | yes | slip ratio (0 if pure lateral rig) |
| `Fz` | N | yes | vertical load |
| `Fx`, `Fy` | N | target | at least one required for direct-force data |
| `Mz` | Nm | no | aligning moment (VeTyT) |
| `vx` | m/s | no | required for relaxation/transient data |
| `p` | Pa | no | inflation pressure (VeTyT, Q-Motion) |
| `gamma` | rad | no | camber |
| `Ts`, `Tc`, `T_road`, `T_air` | K | no | temperatures |
| `mu_ref` | - | no | reference/estimated friction |
| `t` | s | seq | time stamp — required for sequence datasets |
| `sequence_id` | int | seq | groups rows into a trajectory; splits never cut inside one |
| `tire_id` | str | yes | tire/compound/condition identifier → embedding index |
| `source` | str | yes | dataset tag (`kit`, `vetyt`, …) |

Vehicle-level (Experiment 3) adds: `vy, r, ax, ay, r_dot, delta_FL..RR, omega_FL..RR`, and vehicle geometry lives in the sidecar `meta.yaml` (`m, Iz, lf, lr, t_f, t_r, R_e, h_cg`).

### 4.3 Python surface

```python
@dataclass
class TireSample:                 # batched tensors, all (B,) or (B,T)
    alpha, kappa, Fz: Tensor
    context: dict[str, Tensor]
    Fx, Fy: Tensor | None
    mask: Tensor | None

class TireDataset(torch.utils.data.Dataset):
    """Built from a canonical DataFrame; static (B,) or windowed (B,T,...) mode."""

def load_kit(root, **kw) -> pd.DataFrame        # one adapter per source,
def load_vetyt(root, **kw) -> pd.DataFrame      # each returning the canonical schema
def load_roboracer(root, **kw) -> pd.DataFrame
def load_qmotion(root, **kw) -> pd.DataFrame
def load_deep_dynamics(root, **kw) -> pd.DataFrame
def load_tum_cargo_bike(root, **kw) -> pd.DataFrame
def make_synthetic(...) -> pd.DataFrame         # always available, MF-generated ground truth
```

Normalisation statistics (`mean/std` per column, computed on the **train split only**) are saved to `checkpoints/<run>/norm.json` and reloaded at eval — never recomputed on test data.

### 4.4 Datasets — status and acquisition

No dataset is committed. Nothing large is downloaded automatically; each source gets `scripts/download_<name>.py`, which either fetches a small subset or prints exact manual instructions and exits non-zero (pattern taken from `neurelux/scripts/`).

| # | Dataset | Content | Type | Auto-download | Used by |
|---|---|---|---|---|---|
| 1 | KIT Tire Force Transmission | Fx, Fy vs slip, load | **real** test bench | manual (script prints steps) | Exp 1 |
| 2 | VeTyT (bicycle tyre) | Fy, Mz, Fz, pressure, camber | **real** test bench | manual | Exp 1 |
| 3 | TUM Cargo Bicycle Tire | Fx, Fy characteristics | **real** test bench | manual | Exp 1 |
| 4 | Deep Dynamics (BayesRace, IAC) | vehicle-level states/inputs | **simulated** (BayesRace) + **real** (IAC logs) | manual | Exp 3 |
| 5 | RoboRacer model-structured NN | vehicle-level, tire-set + mass-change experiments | **real** small-scale | manual | Exp 2/3 |
| 6 | Q-Motion | tire-pressure variation | **real** | manual | Exp 1 context/generalisation |
| — | synthetic MF + relaxation + graining | generated in-repo | **synthetic** | n/a (always available) | Exp 1–4 smoke tests |

Kaggle candidates are documented in `data/README.md` with a mandatory type label — **real measurement / simulated / game telemetry / synthetic degradation** — and game telemetry (e.g. F1-series UDP logs) is explicitly excluded from any quantitative claim.

**Consequence for scheduling:** all four experiments must be runnable end-to-end on `make_synthetic` from day one, so that the framework is verifiable before any download succeeds.

---

## 5. Experiments

| Exp | Script | Data | Compares | Primary metrics |
|---|---|---|---|---|
| 1 | `experiments/train_direct_force.py` | KIT / VeTyT / synthetic | MF, MLP, MLP+penalty, symmetry, symmetry+envelope, ParameterNet+MF | RMSE, MAE, extrapolation RMSE, symmetry violation, zero-slip force, envelope violation, RMSE vs #train samples |
| 2 | `experiments/train_relaxation.py` | synthetic step tests + RoboRacer | static TireNet, GRU, Neural ODE, RelaxationTireCell | trajectory RMSE, step-response rise distance, behaviour under rapid `alpha, kappa, Fz, mu` changes |
| 3 | `experiments/train_vehicle_supervised.py` | Deep Dynamics / RoboRacer | shared-TireNet Newton–Euler vs per-wheel nets vs analytical | `ax, ay, r_dot` RMSE; generalisation to unseen tire set / friction |
| 4 | `experiments/train_graining.py` | synthetic (weakly supervised) | with/without thermal-graining states | qualitative demonstrator + invariant checks |

Every run writes `results/<exp>/<run_id>/{config.yaml, norm.json, best.pt, metrics.csv, plots/}` and appends one row to `results/<exp>/summary.csv`.

## 6. Tests (`tests/`, pytest)

1. `Fy(alpha=0) == 0` and 2. `Fx(kappa=0) == 0` — exactly, for random weights.
3./4. Odd symmetry in `alpha` / `kappa` — exactly, for random weights.
5. Friction envelope never violated — random and adversarially large inputs, random weights.
6. `mu` inside configured bounds for extreme network outputs.
7. Relaxation lengths strictly positive; `tau > 0`.
8. Wear monotone non-decreasing over a rollout.
9. Graining stays in `[0,1]` over a long rollout with adversarial rates.
10. All four wheels share **the same parameter objects** (identity check, not value check).
11. Force/moment aggregation matches a hand-computed reference case; pure-`Fy` front-only case yields the expected yaw moment sign.
Plus: Euler vs RK4 vs torchdiffeq agreement (skipped if absent); adapter schema validation; MF fit round-trip on synthetic data.

## 7. Reproducibility

Deterministic seeds (`torch`, `numpy`, `random`, `cudnn.deterministic`), YAML configs in `configs/` with a documented override syntax, saved normalisation stats, explicit train/val/test splits (grouped by `sequence_id`, and by *condition* for extrapolation holdouts), checkpoints, CSV summaries. Plot set: `Fy(alpha)`, `Fx(kappa)`, combined-slip friction ellipse, learned `mu`, relaxation step response, residuals, extrapolation, and graining/temperature/wear time series.

## 8. Milestones

- [x] M0 — inspection, PLAN.md, repo skeleton, git
- [x] M1 — `physics/` (pacejka, brush, vehicle_dynamics, thermal, wear) + `layers/` + tests 1–7
- [x] M2 — `models/` static ladder + `training/` + synthetic Experiment 1
- [ ] M3 — `data/` adapters + `scripts/download_*.py` + real-data Experiment 1
- [x] M4 — relaxation + sequence baselines + Experiment 2
- [x] M5 — four-wheel + Experiment 3 (tests 10–11)
- [ ] M6 — thermal/wear/graining + Experiment 4 (tests 8–9)
- [ ] M7 — notebooks, evaluation plots, README results table

## 9. Non-goals / risks

- No claim that the synthetic graining experiment reflects validated real motorsport graining.
- No transient thermal FEM, no tread-block-level physics, no full MF 6.x with all scaling factors (a documented subset only).
- Real-data risk: several of the six datasets may be behind manual registration. Mitigation: synthetic-first (§4.4), and each experiment reports which data it actually ran on.
- Small networks by default (2 hidden layers, 32–64 units). Larger models require an explicit config change and are reported as such.
