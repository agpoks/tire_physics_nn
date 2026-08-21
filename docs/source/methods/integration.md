# Six ways to integrate a physical law

The [taxonomy](taxonomy) says *where* physics enters. This page is the practical
counterpart: the concrete mechanisms, what each costs, and what each buys.

```{figure} ../_static/diagrams/integration_patterns.png
:alt: six integration patterns
:width: 100%
```

## 1. Feature engineering — *guided*

```{math}
x \;\longrightarrow\; \phi_{\text{phys}}(x) \;\longrightarrow\; \text{NN}
```

Present the network with physically meaningful coordinates: even invariants
$\kappa^2, \alpha^2$, load-normalised force $F/F_z$, dimensionless slip.

**Buys** faster convergence and better conditioning, for free.
**Costs** nothing.
**Guarantees** nothing on its own — but it is what makes pattern 5 possible, because the
even invariants are exactly what an odd output map needs.

Used in this project by every encoded model
({py:class}`tire_nn.layers.symmetry.OddSymmetricForceField`).

## 2. Penalty / PINN — *informed*

```{math}
\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda\,\big\|\mathcal{R}_{\text{phys}}(f_\theta)\big\|^2
```

**Buys** the ability to express *any* differentiable property, including PDE residuals
with no closed-form solution.
**Costs** a weight $\lambda$ to tune, a slower optimisation, and competition with the
data term.
**Guarantees** satisfaction only in expectation on the sampled points.

In this framework it appears only as the control:
{py:func}`tire_nn.training.losses.friction_penalty`, used by the `mlp_penalty` rung.

## 3. Residual / grey box — *guided + encoded*

```{math}
F = \underbrace{F_{\text{phys}}(x)}_{\text{fitted analytical model}} \;+\; r_\theta(x)
```

**Buys** a small network — it only has to explain the mismatch — and a directly
reportable diagnostic (how large is the correction?).
**Costs** a dependence on the baseline being decent; and if the residual is
unconstrained, all the baseline's guarantees are lost.
**Guarantees** whatever the *sum* is constructed to guarantee. Here the residual uses the
same odd parameterisation and the sum is projected into the ellipse, so symmetry, the
zero-slip property and the bound all survive.

{py:class}`tire_nn.models.residual_tire.ResidualTireNet`.

## 4. Parameter network — *encoded*

```{math}
\theta_{\text{phys}} = \Pi_{\text{bounds}}\big(f_\theta(c)\big), \qquad
F = \mathrm{law}\big(x;\ \theta_{\text{phys}}\big)
```

The network never outputs a force. It outputs *coefficients* of a physical law, through
a monotone map into a valid range, and the law does the rest.

```{math}
p = p_{\min} + \mathrm{softplus}(z) \quad\text{(positive)}, \qquad
p = p_{\min} + (p_{\max}-p_{\min})\,\sigma(z) \quad\text{(bounded)}
```

**Buys** full interpretability — the output is a *tire*, with a readable $\mu$,
$C_\alpha = BCD$ and $\sigma_{rel}$ — plus every guarantee the underlying law has, plus
strong regularisation (see the [benchmarks](../comparison/benchmarks): the same Magic
Formula fitted directly by `scipy` is the *worst* model on 210 samples, while the
parameter network is the best).
**Costs** expressiveness: you inherit the law's blind spots. If the real tire does
something the Magic Formula cannot represent, this model cannot either.

{py:class}`tire_nn.models.parameter_tire.ParameterTireNet`.

```{figure} ../_static/diagrams/parameter_net.png
:alt: parameter network architecture
:width: 100%
```

:::{admonition} The coefficients must not depend on the slip
:class: important
At a given operating point a tire has *one* set of coefficients. If the network sees
$\alpha$ and $\kappa$ it can emit a different "tire" at every slip value, and the result
is a curve fit wearing the vocabulary of a tire model. `parameters_at(Fz, context)`
therefore never receives the slip — enforced structurally and tested.
:::

## 5. Structural output map — *encoded*

```{math}
F_x = \kappa\; g_x(\kappa^2, \alpha^2, F_z, c)\cdot s(\rho), \qquad
F_y = -\alpha\; g_y(\alpha^2, \kappa^2, F_z, c)\cdot s(\rho)
```

Compose the network output with fixed algebra chosen so the desired property is an
identity. Three techniques cover most cases:

| goal | construction |
|---|---|
| $f(0) = 0$ and oddness | multiply by the odd variable; feed only even invariants |
| positivity / non-negativity | `softplus`, `exp`, squaring |
| a bound | a saturating radial map, $\tanh\rho/\rho$ |
| an invariant interval $[0,1]$ | write the ODE so both boundaries are absorbing |
| a **distribution** with a known integral | `softmax` — positivity and $\sum_i p_i \Delta\xi = F_z$ in one operation ([contact patch](../physics/contact-patch)) |
| an **ordered** set of thresholds | cumulative `softplus`, $t_{k+1} = t_k + \mathrm{softplus}(b_k)$ ([imaging](../physics/imaging)) |
| a monotone latent behind ordinal labels | one sigmoid output, thresholded — the classes cannot disagree with the ordering |

**Buys** exactness with no tuning parameter and no runtime cost.
**Costs** design effort, and it only works for properties expressible as algebra on the
output. Asymmetries that genuinely exist (ply steer, conicity, camber thrust) have to be
added back as an explicit, separable, switchable term — which is arguably a feature,
since they then stay individually reportable.

{py:class}`tire_nn.models.encoded_tire.EncodedTireNet`.

```{figure} ../_static/diagrams/encoded_tire.png
:alt: encoded tire architecture
:width: 100%
```

## 6. Universal differential equation — *encoded structure, learned closure*

```{math}
\dot z = f_{\text{phys}}\big(z,\; u,\; g_\theta(\cdot)\big)
```

Fix the *structure* of the differential equation and learn only the rates or
coefficients inside it. The state stays physical, stability properties are inherited,
and the invariants of the continuous system carry over if the discretisation is chosen
to preserve them.

**Buys** correct dynamics, guaranteed stability ($\tau > 0 \Rightarrow$ contractive),
conserved invariants ($w$ monotone, $g \in [0,1]$), and interpretable time constants.
**Costs** sequential integration — the main runtime expense in this framework — and care
with the discrete step (see {eq}`eq-graining-step` in
[Thermal, wear, graining](../physics/thermal-wear)).

{py:class}`tire_nn.models.relaxation_tire.RelaxationTireCell`,
{py:class}`tire_nn.models.thermo_graining_tire.ThermoGrainingTire`,
{py:class}`tire_nn.models.four_wheel_vehicle.FourWheelVehicle`.

### This pattern has a name: UDE

A differential equation whose right-hand side is *partly* a known mechanism and *partly*
a universal approximator is a **universal differential equation**
{cite}`rackauckas2020ude`:

```{math}
\dot z = \underbrace{f_{\text{known}}(z, u)}_{\text{conservation, balance, structure}}
        \;+\; \underbrace{g_\theta(z, u)}_{\text{unknown closure}}
```

The two dynamic models in this framework are UDEs, and it is worth naming them as such
because the UDE literature is exactly where their justification lives:

| model | known part $f_{\text{known}}$ | learned closure $g_\theta$ |
|---|---|---|
| `RelaxationTireCell` | the first-order lag $\dot F = (F_{ss}-F)/\tau$, and $\tau = \sigma/|v_x|$ | the steady-state map $F_{ss}$ and the relaxation length $\sigma$ |
| `ThermoGrainingTire` | the two-node energy balance, $P_{slip} = -F\cdot v_{slip}$, the $(1-g)R_f - gR_c$ form, $\dot w \ge 0$ | the rate functions $R_{\text{form}}, R_{\text{clean}}, f_w$ |

The **Neural ODE baseline is the degenerate case** in which $f_{\text{known}} = 0$ and
the whole right-hand side is learned. So the comparison in
[Transient](../physics/transient) — encoded relaxation cell versus generic Neural ODE —
is a direct UDE-versus-black-box test on tire data: same integrator, same inputs, same
budget, and the only difference is whether the known part is supplied. The UDE wins on
accuracy (110 N against 252 N rollout RMSE), on extrapolation across speed, and on
returning a parameter an engineer can check.

### Degradation is the textbook UDE case

Tire degradation is close to an ideal application, because the split between known and
unknown is unusually clean:

**Known with confidence** — energy balance (heat in equals heat stored plus heat lost),
that dissipation is $-F \cdot v_{slip}$, that wear is monotone, that a fractional
surface state lives in $[0,1]$, and that grip degrades multiplicatively.

**Genuinely unknown** — the *kinetics*. How fast graining forms at a given temperature
and slip energy, how the cleaning rate depends on the operating point, how wear rate
scales with compound and load. These are compound-specific, empirical, and exactly the
sort of thing a small network should absorb.

Encoding the first group and learning the second is what
{py:class}`~tire_nn.models.thermo_graining_tire.ThermoGrainingTire` does, and it is why
the guarantees survive: the closure terms are wrapped in `softplus`, so no weight vector
can make wear reverse or push graining out of $[0,1]$.

:::{admonition} What this framework does *not* do with its UDEs
:class: caution
The usual next step in the UDE workflow is to recover a *symbolic* form for the learned
closure — fit $g_\theta$ with sparse regression (SINDy {cite}`brunton2016sindy`) and read
off an interpretable rate law, e.g. an Arrhenius-type temperature dependence for the
wear rate. That is not implemented here. The closure stays a small network, which means
you get a usable model but not a discovered equation. It is the most natural extension
of this work.
:::

## Combining them

The patterns compose, and the models in this framework are mostly combinations:

```python
tire = ParameterTireNet(context_keys=("vx", "p"))     # 1 + 4 + 5
tire = ThermoGrainingTire(tire)                        # + 6 (condition states)
cell = RelaxationTireCell(tire.steady)                 # + 6 (relaxation)
vehicle = FourWheelVehicle(tire, vp)                   # + 6 (rigid body), one shared tire
```

Each wrapper preserves the guarantees of what it wraps, which is the property that makes
the stack safe to build: the condition model only scales the friction ellipse, and the
relaxation cell only contracts toward a force that is already inside it.
