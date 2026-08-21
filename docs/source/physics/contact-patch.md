# The contact patch as a PDE

Every model in [Steady state](steady-state.md) is *algebraic*: slip in, force out. The
brush model is not — it is the **integral of a local boundary-value problem along the
contact patch**, and collapsing it to a closed form costs assumptions that are often
wrong. This page keeps the local problem explicit.

## The local problem

Take a patch of length $2a$ with a coordinate $\xi$ from the leading edge ($\xi=0$) to
the trailing edge ($\xi=2a$). Tread material enters undeflected and is carried through,
so in the **adhesion** region a bristle's deflection grows with distance travelled:

```{math}
:label: eq-bristle-ode
\frac{\mathrm{d}\boldsymbol\delta}{\mathrm{d}\xi} = \boldsymbol\sigma,
\qquad \boldsymbol\delta(0) = \mathbf 0,
\qquad \boldsymbol\tau = k_b\,\boldsymbol\delta
```

and the local shear cannot exceed what friction carries:

```{math}
:label: eq-local-bound
\|\boldsymbol\tau(\xi)\| \le \mu\,p(\xi),
\qquad
\mathbf F = \int_0^{2a}\boldsymbol\tau(\xi)\,\mathrm{d}\xi
```

Discretising $\xi$ into $n$ elements turns {eq}`eq-bristle-ode` into a cumulative sum
along a **chain** — a 1-D graph in which each element talks only to its predecessor —
and the integral into a quadrature.

:::{admonition} Units: everything here is a line quantity
:class: important
$p$ is a line load in N/m with $\int p\,\mathrm{d}\xi = F_z$, and $\tau$ is likewise N/m.
Patch width never appears. Normalising $p$ to $F_z$ *and then* multiplying the force by
a width silently scales the whole model by that width — a bug this codebase had, which
made the model saturate at 200 N instead of 1000 N.
:::

## Does the discretisation reproduce the closed form?

A discretisation that does not converge to the known answer is a bug, not a model.
Measured against {py:func}`tire_nn.physics.brush.brush_combined`:

| elements | max error [N] on a 1 kN tire |
|---|---|
| 8 | 9.71 |
| 32 | 0.624 |
| 128 | 0.035 |
| 512 | 0.002 |

Second order, as a midpoint rule should be: 4× the elements cut the error 17.9×.

## What the extra structure buys

The closed form assumes a **parabolic** pressure distribution and uniform bristle
stiffness. Real patches are neither: load, inflation pressure, camber and wear all
reshape them. The discretised form can represent that; the closed form cannot.

It also exposes the mechanism. At small slip the shear stays under the bound everywhere
(pure adhesion); as slip grows the shear meets the bound at the trailing edge first,
where pressure has fallen, and the sliding region eats forward. **That migration is the
force curve** — its saturation is not a fitted shape but the point where the bound
binds.

## The encoded model

{py:class}`tire_nn.models.patch_brush_net.PatchBrushNet` learns the pressure
distribution and the material parameters while keeping every constraint exact:

| guarantee | mechanism |
|---|---|
| $p_i \ge 0$ | `softmax` over the patch elements |
| $\sum_i p_i \Delta\xi = F_z$ | the same `softmax` — it sums to one by definition |
| $\|\tau_i\| \le \mu p_i$ | elementwise `min` against the bound |
| force opposes slip | shear built along $-\boldsymbol\sigma$ |
| $F(0)=0$, odd symmetry | zero slip gives zero deflection |
| $\mu$, $k_b$, $a$ physical | bounded parameter transforms |

### The `softmax` is the distribution-valued analogue of a bounded parameter

Elsewhere in this project a scalar with known bounds gets a monotone transform into that
range ([parameter networks](../models/parameter.md)). A *distribution* with a known
integral gets a `softmax`: positivity and the exact integral in one operation, for any
weights.

Written as a penalty $\lambda\big(\sum_i p_i\Delta\xi - F_z\big)^2$ the load balance
holds only on average — and a model that can quietly invent or destroy vertical load can
fit almost anything. Measured on a fit to a non-parabolic tire:

| load-balance treatment | force RMSE [N] | load-balance error |
|---|---|---|
| penalty, $\lambda = 0$ | 13.5 | 62.6 N (6.3 % of $F_z$) |
| penalty, $\lambda = 1$ | 8.1 | 1.9 N (0.2 %) |
| penalty, $\lambda = 100$ | 68.0 | 0.4 N (0.04 %) |
| **`softmax` (encoded)** | **2.9** | **0 N, exactly** |

The penalty trades the constraint against the fit, and the trade is set by a weight
nobody can choose from first principles: too low and the model invents vertical load,
too high and it stops fitting the force. The encoded version gives the exact constraint
*and* the best fit, with nothing to tune.

Also note the initialisation trap. A penalised model whose pressure starts orders of
magnitude away from $F_z/2a$ never recovers — the first run of this comparison produced
a 997 N load error at every penalty weight, which measured the initialisation rather
than the constraint. The `softmax` has no such failure mode: it is correctly normalised
from the first step by construction.

## Measured: a tyre the parabolic assumption cannot represent

Force data generated from a skewed, flat-topped profile with 3 N noise:

| model | force RMSE [N] |
|---|---|
| learned pressure | **2.86** (the noise floor is 3.0) |
| parabolic only | 25.75 — **9× worse** |

The learned model also recovers the physical parameters: half-length 0.0600 m against a
true 0.060, and $\mu = 1.001$ against a true 1.0.

The pressure *shape* correlates 0.94 with the truth — recovered in character but
not in every detail. A scalar force curve does not pin down all 48 elements of a
distribution, which is the same identifiability limit that appears in
[degradation](degradation.md). A pressure-mapping test bench would resolve it; force
alone does not.

## Code

```python
from tire_nn.physics.brush_patch import patch_forces, pressure_from_logits
from tire_nn.models.patch_brush_net import PatchBrushNet

model = PatchBrushNet(n_elements=48, learn_pressure=True)
out = model(alpha, kappa, Fz)
out.params["pressure"]            # the learned line load along the patch
out.params["sliding_fraction"]    # how much of the patch is sliding
```

Notebook: [The contact patch as a PDE](../notebooks/05_brush_patch_pde).

## Tests

`tests/test_patch_brush.py`: `test_discretised_patch_reproduces_the_closed_form_brush_model`,
`test_quadrature_converges_at_second_order`,
`test_softmax_pressure_integrates_to_the_load_for_any_logits`,
`test_learned_patch_keeps_the_load_balance_for_adversarial_weights`,
`test_sliding_fraction_grows_from_zero_to_one_with_slip`.

## References

Brush-model derivation and the adhesion/sliding structure: {cite}`svendenius2007tire`,
{cite}`pacejka2012tire`. Rubber friction theory behind the local bound:
{cite}`maglione2026rubberwear`.
