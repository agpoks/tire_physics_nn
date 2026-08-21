# `ResidualTireNet` — grey box

```{math}
F = \Pi_{\text{ellipse}}\Big( \underbrace{F_{\text{MF}}(\alpha,\kappa,F_z)}_{\text{fitted, fixed}}
    \;+\; \lambda_r\,\underbrace{r_\theta(\alpha,\kappa,F_z,c)}_{\text{odd-symmetric residual}} \Big)
```

The analytical model carries the structure; the network only explains the mismatch.

## Why this rung exists

Most real projects start here. You have a supplier's Magic Formula coefficients, or a fit
from a rig session, and the car does not quite behave like them — different surface,
different temperature, a year of wear on the compound. The interesting question is not
"what is the tire?" but "how does this tire differ from the one I was given?".

The decomposition is also a diagnostic: `params["residual_fraction"]` reports how much of
the force the network is contributing. A residual that grows to dominate the baseline is
telling you the analytical model is wrong, not that the network is clever.

## What is preserved

The residual uses the **same odd parameterisation** as
[`EncodedTireNet`](encoded), so the *sum* is still exactly odd and still vanishes at zero
slip; and the envelope is applied to the sum, so the correction can reshape the curve but
cannot push the tire past its friction limit.

```python
from tire_nn.models import ResidualTireNet
from tire_nn.physics.fitting import fit_magic_formula

px, py = fit_magic_formula(train_df)
model = ResidualTireNet(px=px, py=py, residual_scale=0.3, hard_envelope=True)
```

`residual_scale` bounds how much authority the network has relative to the baseline —
set it small when you trust the analytical model and only expect a correction.

## Category

This is the genuine hybrid in the framework: **physics-guided** (initialised from and
built around an analytical fit) *and* **physics-encoded** (symmetric residual, projected
sum). See [taxonomy](../methods/taxonomy).

## Measured

Third-best model in the full-budget [benchmark](../comparison/benchmarks) (17.47 N,
against 17.39 N for `ParameterTireNet` and 17.71 N for the fitted Magic Formula alone),
with all violations at zero — and notably better than either component on its own in the
mid-data regime (19.0 N at 570 samples).
