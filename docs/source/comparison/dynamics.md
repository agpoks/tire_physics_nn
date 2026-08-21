# Comparing the dynamics

Side-by-side behaviour of the modelling choices, measured rather than asserted. Every
figure is reproducible with `python docs/generate_figures.py`.

## Steady-state shape

```{figure} ../_static/figures/steady_state_laws.png
:alt: four steady-state laws compared
:width: 100%

Four laws matched to the same $C_\alpha$ and $\mu$. **What to take from this:** near
zero slip they are indistinguishable, so low-slip data cannot tell you which one is
right. They differ entirely at the limit — which is the only place a racing controller
operates.
```

| | near zero slip | at the peak | past the peak | outside the data |
|---|---|---|---|---|
| linear | exact | badly wrong | badly wrong | diverges |
| brush | exact | good | flat, no decay | stays bounded |
| Dugoff | exact | good | flat, no decay | stays bounded |
| Magic Formula | exact | excellent (fitted) | excellent (fitted) | unreliable — it is a fit |
| encoded NN | learned | learned | learned | **bounded, symmetric, but shape unknown** |

The last row is the point of the project: the encoded network's *shape* outside the data
is as unknown as any other learned model's, but its *properties* — sign, symmetry,
bound, dissipativity — are guaranteed there.

## Combined slip

```{figure} ../_static/figures/combined_slip_methods.png
:alt: combined slip loci
:width: 100%

All three analytical laws respect the friction circle but fill it differently. Dugoff's
radial spokes come from its piecewise $f(\lambda)$; the Magic Formula's locus pulls
*inside* the circle at large combined slip because of its post-peak decay.
```

```{figure} ../_static/figures/friction_ellipse.png
:alt: encoded vs unbounded network locus
:width: 100%

The same comparison for learned models. Symmetry alone (left) reaches 40 kN on a 1 kN
tire. The projection (right) makes that impossible.
```

## Transient response

```{figure} ../_static/figures/relaxation_step.png
:alt: step response vs time and distance
:width: 100%

Why the relaxation length, not a time constant, is the physical parameter: plotted
against travelled distance, the response at 10, 20 and 40 m/s collapses onto one curve.
```

```{figure} ../_static/figures/transient_ratio.png
:alt: rise distance ratio
:width: 78%

The diagnostic across models and two independent runs. A ratio near 1 is
distance-parameterised (correct); near 3 is a fixed time constant. The unstructured
baselines do not have a *stable* answer — the GRU gave 2.25 in one run and 0.53 in
another. The encoded cell cannot give anything other than ≈1.
```

| | speed extrapolation | stability | interpretable | parameters |
|---|---|---|---|---|
| quasi-static | n/a | n/a | n/a | 0 |
| GRU | not guaranteed, empirically unstable across runs | not guaranteed | no | ~2 600 |
| Neural ODE | learned a fixed time constant | not guaranteed | no | ~1 400 |
| relaxation cell | exact by construction | contractive by construction | $\sigma$ in metres | 2 |

## Thermal response

```{figure} ../_static/figures/thermal_two_time_scales.png
:alt: two thermal time scales
:width: 75%

One energy input, two responses: the surface reacts within a corner, the core over a
stint. A single-node model cannot be both.
```

```{figure} ../_static/figures/effective_friction.png
:alt: condition effects on friction
:width: 100%

How condition modulates grip. Note the temperature factor is **not monotone** — a
monotone model would predict a tire that keeps gaining grip as it heats.
```

```{figure} ../_static/figures/graining_stint.png
:alt: synthetic graining stint
:width: 80%

The four-phase synthetic stint. Wear accumulates monotonically; graining forms cold,
cleans up warm, and re-forms during the cool-down. **Synthetic demonstrator — not
validated real graining.**
```

## Vehicle level

```{figure} ../_static/figures/load_transfer.png
:alt: load transfer during braking and cornering
:width: 78%

Per-corner loads during a braking event and a cornering sequence. The total stays at
$mg$ exactly, by construction.
```

## Physical consistency

```{figure} ../_static/figures/violations.png
:alt: violations by model
:width: 100%

Four *untrained* models audited on a grid wider than any training range. Bars at the
dotted line are exactly zero. Symmetry alone is the worst case for the envelope; the
plain MLP fails every column it can.
```
