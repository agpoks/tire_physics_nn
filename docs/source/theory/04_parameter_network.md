# 4. Parameter networks and the differentiable tire law (P4)

## The analytical laws

### Magic Formula

The sine-form Magic Formula {cite}`pacejka2012tire`:

```{math}
:label: eq-mf
F(x) = D \sin\Big( C \arctan\big( Bx' - E\,(Bx' - \arctan Bx')\big)\Big) + S_v,
\qquad x' = x + S_h
```

with $x = \alpha$ (lateral) or $x=\kappa$ (longitudinal), and

```{math}
:label: eq-mf-roles
\underbrace{B}_{\text{stiffness}},\quad
\underbrace{C}_{\text{shape}},\quad
\underbrace{D = \mu(F_z)\,F_z}_{\text{peak}},\quad
\underbrace{E}_{\text{curvature}} .
```

The initial slope — the cornering stiffness — is

```{math}
:label: eq-cornering-stiffness
C_\alpha = \left.\frac{\partial F_y}{\partial \alpha}\right|_{\alpha=0} = B\,C\,D .
```

```{figure} ../_static/figures/magic_formula_parameters.png
:alt: effect of B, C, E and mu on the Magic Formula curve
:width: 100%

What each coefficient does. $B$ sets the initial slope, $C$ how far the curve falls
back after the peak, $E$ the curvature around the peak, and $\mu$ the peak height. The
peak is always $D = \mu F_z$ — which is why $D$ is stored as $\mu$ here, not as a free
force.
```

### Load sensitivity

Peak friction falls as the tire is loaded up, so $D$ is *not* a free parameter:

```{math}
:label: eq-load-sensitivity
\mu(F_z) = \mu_0 \left[1 - k_\mu\left(\frac{F_z}{F_{z0}} - 1\right)\right],
\qquad D = \mu(F_z)\,F_z .
```

```{figure} ../_static/figures/load_sensitivity.png
:alt: load sensitivity of friction and peak force
:width: 100%

Left: $\mu(F_z)$. Right: the peak force is sub-linear in load. A model with a free $D$
would have to learn this from the two or three load levels a rig usually provides;
encoding it leaves only $k_\mu$ to identify.
```

### Combined slip by the similarity method

Rather than the twenty-odd weighting coefficients of full MF 6.x
{cite}`besselink2010magicformula`, this project uses the *similarity* (normalised slip
vector) construction, which is what the brush model implies
{cite}`svendenius2007tire`:

```{math}
:label: eq-combined
\sigma_x = \kappa,\quad \sigma_y = \tan\alpha,\quad \sigma = \sqrt{\sigma_x^2+\sigma_y^2},
\qquad
F_x = \frac{\sigma_x}{\sigma}\,\big|F_{x0}(\sigma)\big|, \quad
F_y = -\frac{\sigma_y}{\sigma}\,\big|F_{y0}(\sigma)\big| .
```

:::{admonition} A subtlety worth stating: theoretical slip breaks odd symmetry
:class: important

The textbook definition normalises by $1+\kappa$:
$\;(\sigma_x,\sigma_y) = (\kappa, \tan\alpha)/(1+\kappa)$. That factor is more accurate
at large braking slip, but it is **not odd in $\kappa$** — the practical slip ratio is
itself an asymmetric definition (driving slip is unbounded above, braking slip is
bounded below by $-1$).

That asymmetry is *kinematic*, living in the definition of $\kappa$, not *constitutive*,
living in the tire. Since the whole ablation rests on comparing models under the same
symmetry claim, the symmetric form is the **default** here and the textbook form is
opt-in:

```python
pacejka_combined(alpha, kappa, Fz, px, py, theoretical_slip=True)
```

Auditing the two side by side shows the effect directly: the symmetric form reports an
odd-symmetry violation of exactly 0, the theoretical-slip form reports 0.22 (in
load-normalised force units).
:::

### The brush model

The physical cross-check: the tread is a row of elastic bristles that stick until the
local shear exceeds $\mu p(x)$, then slide. With a parabolic pressure distribution and
$\theta = C_{\text{stiff}}/(3\mu F_z)$, $z = \theta\sigma$:

```{math}
:label: eq-brush
F(\sigma) = \begin{cases}
\mu F_z\,(3z - 3z^2 + z^3), & z < 1 \quad \text{(partial sliding)}\\[2pt]
\mu F_z, & z \ge 1 \quad \text{(full sliding)}
\end{cases}
```

It gives odd symmetry and the bound $|F| \le \mu F_z$ *for free* — which is the whole
argument for encoding them.

```{figure} ../_static/figures/brush_vs_magic_formula.png
:alt: brush model vs magic formula
:width: 65%

The two analytical baselines. The brush model saturates exactly at $\mu F_z$ and never
falls back; the Magic Formula can reproduce the post-peak decay that real tires show.
```

## The parameter network

Instead of predicting a force, the network predicts the *coefficients* of
{eq}`eq-mf` — and only from the **operating condition**, never from the slip:

```{math}
:label: eq-paramnet
\big(\mu, B, C, E, k_\mu, \sigma_x, \sigma_y\big) = \Pi_{\text{bounds}}\Big(f_\theta(F_z, c)\Big),
\qquad
(F_x, F_y) = \text{MF}\big(\alpha, \kappa, F_z;\ \text{those coefficients}\big).
```

The bounding map $\Pi$ is a fixed monotone transform:

```{math}
:label: eq-bounds
p = p_{\min} + \text{softplus}(z) \quad\text{(positive)},
\qquad
p = p_{\min} + (p_{\max}-p_{\min})\,\text{sigmoid}(z) \quad\text{(bounded)} .
```

```{figure} ../_static/figures/bounded_parameters.png
:alt: bounded parameter transforms
:width: 100%

The parameter is valid for *any* raw network output, including $z = \pm 30$. There is
no training step at which the model holds a non-physical tire.
```

### Why bounded transforms rather than clipping in the loss

A negative $C$ flips the curve; a negative $D$ is a negative peak force. In those
regions the loss landscape is meaningless and the gradients are actively misleading.
Clipping in the loss leaves the *raw* parameter free to wander there and only masks the
symptom. The monotone transform makes the parameter valid at **every step**, and the
bias of each head is initialised so an untrained model already starts at a sensible
tire ($\mu = 1.0$, $B = 10$, $C = 1.6$, $\sigma_y = 0.30$ m).

Default ranges bracket published passenger-car values and the fitted small-scale
racing values in `On-Track-SysID` ($B \approx 7\text{–}8$, $C \approx 1.6\text{–}2.1$,
$E \approx 0.4\text{–}0.5$) {cite}`ontracksysid` with margin.

### Why the coefficients must not depend on slip

At a given operating point a tire has *one* set of coefficients. If the network were
allowed to see $\alpha$ and $\kappa$, it could produce a different "tire" at every slip
value and the result would be a curve fit wearing the vocabulary of a tire model. This
is enforced structurally — `parameters_at(Fz, context)` never receives the slip — and
tested (`test_parameter_model_coefficients_depend_only_on_condition_not_on_slip`).

The reward is interpretability: $\mu$, $C_\alpha = BCD$ and $\sigma$ can be read off,
plotted against operating conditions, compared with a `scipy` fit, and handed to an MPC
that expects Pacejka coefficients.

## Fitting the analytical baseline

The learned models must beat a *properly fitted* Magic Formula, not a guessed one.
{py:func}`tire_nn.physics.fitting.fit_magic_formula` does bounded least squares with
`scipy.optimize.least_squares` on **load-normalised** residuals $F/F_z$ — without that
normalisation, high-load samples dominate and $\mu$ is biased toward the heavy end of
the range.

```python
from tire_nn.physics.fitting import fit_magic_formula
px, py = fit_magic_formula(train_df)     # -> two MFParams
```

On synthetic data generated with $\mu = 1.1$, $k_\mu = 0.08$, the fit recovers
$\mu = 1.10$ and $k_\mu = 0.08$; $B$ and $C$ trade off against each other, as expected,
because only the product $BCD$ is well identified near the origin.

## Code

```python
from tire_nn.models import ParameterTireNet

model = ParameterTireNet(context_keys=("p", "Ts"), hidden=(32, 32))
out = model(alpha, kappa, Fz, ctx)
out.params["mu_y"]      # effective (load-corrected) friction
out.params["C_alpha"]   # cornering stiffness [N/rad]
out.params["sigma_y"]   # relaxation length [m]
```

:::{note}
`params["mu_x"]`/`["mu_y"]` are the **load-corrected** values $\mu(F_z)$; the nominal
values at $F_{z0}$ are `params["mu_x0"]`/`["mu_y0"]`. Reporting the nominal value would
understate the friction ellipse at light load and overstate it at high load.
:::

## Guaranteed by tests

- `test_parameter_head_stays_in_bounds_under_adversarial_features` ($\sigma = 20$ weights, features $\times 50$)
- `test_mu_stays_within_configured_bounds`
- `test_relaxation_length_is_strictly_positive`
- `test_parameter_model_exposes_interpretable_quantities`
- `test_cornering_stiffness_matches_numerical_derivative` — {eq}`eq-cornering-stiffness` checked against finite differences
