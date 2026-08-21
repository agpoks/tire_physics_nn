# Degradation from stint data

A worked example of the whole method on a problem where the physics is partly known,
the state is completely unobserved, and the data is real.

## The problem

Tyre condition is latent. Nobody publishes a tyre's wear depth or graining fraction lap
by lap. What *is* published, for every lap of every Formula 1 race, is the consequence:
the car gets slower.

This is the same shape of problem as [vehicle-level identification](vehicle.md), one
level further removed. There, tyre *forces* were unobserved and their effect on
acceleration was measured. Here the tyre *state* is unobserved and its effect on lap
time is measured — through an even noisier channel.

## Why a UDE fits

The split between known and unknown is unusually clean, which is exactly the setting a
**universal differential equation** targets {cite}`rackauckas2020ude`.

**Known — the observation structure.** Lap time is additive in its contributions, and
the non-tyre terms are understood:

```{math}
:label: eq-lap-time
t_{\text{lap}} = \underbrace{t_{\text{ref}}(\text{compound}) + p_{\text{car,circuit}}}_{\text{pace}}
    + \underbrace{c_{\text{fuel}}\,\phi_{\text{fuel}}}_{\text{fuel load}}
    + \underbrace{w}_{\text{wear}} + \underbrace{a_g\,g}_{\text{graining}}
```

A full fuel tank costs a few seconds a lap and burns off linearly with distance; a pit
stop resets the tyre state to zero; wear is monotone; graining is a bounded fraction of
the tread surface.

**Unknown — the kinetics.** How fast wear and graining accumulate as a function of
compound, track temperature and tyre age:

```{math}
:label: eq-degradation-ude
\frac{\mathrm{d}w}{\mathrm{d}\lambda} = \mathrm{softplus}\big(f_w(u)\big) \ge 0,
\qquad
\frac{\mathrm{d}g}{\mathrm{d}\lambda} = (1-g)R_{\text{form}}(u) - g R_{\text{clean}}(u)
```

per lap $\lambda$, with $u$ the gating features (tyre age, how far the track is below
and above a reference temperature, air temperature, compound embedding). The
`softplus` wrappers make wear irreversible and keep $g \in [0,1]$ for any weights, and
the discrete graining update uses the exact zero-order-hold solution so the bound
survives a one-lap step. Same machinery as
[Thermal, wear, graining](thermal-wear.md); different observation.

## An identifiability result worth knowing

Wear *amplitude* and wear *rate* are not separately identifiable from lap time: only
their product enters {eq}`eq-lap-time`, and $w$ never saturates. The state is therefore
defined directly in **seconds of lap time** — the amplitude is fixed at 1 by definition
and the learned quantity is the rate.

Graining is different, because it saturates at $g = 1$. Its amplitude $a_g$ — the lap
time lost at fully developed graining — *is* identifiable, and is learned.

That asymmetry is a property of the problem, not of the implementation, and it is the
kind of thing an encoded model makes visible.

## The data

**Real**: official F1 timing data via the MIT-licensed FastF1 package
{cite}`fastf1`, which needs no API key.

```bash
python -m pip install fastf1
python scripts/download_f1_stints.py --seasons 2023 --rounds 1 2 3 4 5 9 13 17
```

The canonical stint schema is in {py:mod}`tire_nn.data.lap_degradation`: tyre age,
compound, lap time, track and air temperature, remaining fuel fraction. The tyre state
itself is deliberately *not* a column, because nobody measures it.

**Synthetic**: {py:func}`~tire_nn.data.lap_degradation.make_synthetic_stints` generates
stints from a known degradation model, with graining that forms on a cold track early in
a stint and cleans up once the track is warm. Because the latent truth is known there,
the recovered states can be scored against it — which is the only way to check that the
model has identified the *dynamics* rather than merely fitted the lap times.

:::{warning}
Lap time is an **indirect, aggregate and confounded** observation of tyre state: fuel
load, traffic, driver management, track evolution, wind and safety cars all move it.
It is the right data for asking whether degradation *dynamics* can be identified from
their consequence, and the wrong data for asking what a tyre's friction coefficient is.
Tyre temperature is not published, so the thermal gating uses track and air temperature
as proxies.
:::

```{figure} ../_static/figures/degradation_signal.png
:alt: lap time versus tyre age by compound
:width: 68%

The signal the model has to work with: lap time relative to the stint median, against
tyre age, from real 2023 Formula 1 timing data. The soft compound loses roughly a second
over a dozen laps and the hard degrades more slowly — but the scatter is large, because
lap time also carries fuel load, traffic, driver management and track evolution.
```

## What the data forced

Two modelling decisions were not in the original design and were forced by the real
data. Both are worth recording, because both are the kind of thing that silently
invalidates a result.

**Car and circuit pace dominate everything.** A front-running car is over a second a lap
faster than a backmarker, and circuits differ by thirty seconds. Degradation is about a
second over a whole stint. Without a per-(session, driver) offset, the model fits the
pace difference and reports nonsense kinetics — in the first run the linear baseline
fitted *negative* degradation for two of three compounds. With the offset, the same
baseline recovers a sensible ordering (soft degrading fastest).

**The offsets must be solved, not descended.** They are nuisance parameters spanning
tens of seconds while the kinetics live in hundredths of a second per lap. Learning both
with one gradient step means either the offsets converge far too slowly or the kinetics
are swamped. Since the offsets enter additively, they have an exact least-squares
solution for any frozen kinetics — the mean residual per group — so the training loop
alternates: solve the offsets, then take a gradient step on the kinetics. This is the
standard profile-likelihood treatment of a nuisance parameter, and it is also what makes
the held-out evaluation meaningful: on a new circuit only that scalar is re-estimated,
so the reported error measures degradation rather than pace.

## Results

### Synthetic — does it recover known dynamics?

Ten sessions, six drivers, ground-truth graining that forms cold and cleans up warm.
A single representative run:

| model | params | test RMSE [s] | wear corr. with truth | graining corr. |
|---|---|---|---|---|
| linear in tyre age | 19 | 0.290 | – | – |
| black-box MLP | 1 398 | 0.296 | – | – |
| UDE, wear only | 1 278 | 0.262 | 0.75 | – |
| **UDE, wear + graining** | 1 278 | **0.258** | 0.86 | 0.93 |

The UDE is the most accurate, and it recovers the **fuel coefficient** as 3.47 s against
a true 3.50 s — a physical quantity identified from lap times alone.

:::{admonition} But the wear/graining decomposition is NOT reliably identifiable
:class: warning

Repeating that run across seeds on *identical data* gives, in two independent sweeps
(six seeds at 800 epochs via the experiment script, three seeds at 400 epochs in the
notebook):

| quantity | sweep A range | sweep B range |
|---|---|---|
| lap-time RMSE [s] | 0.237 – 0.241 | 0.243 – 0.244 |
| fuel coefficient [s] (true 3.50) | **3.47 – 3.48** | **3.471 – 3.474** |
| wear correlation with truth | **0.29 – 0.63** | **0.49 – 0.91** |
| graining correlation with truth | 0.65 – 0.83 | 0.81 – 0.96 |

In both sweeps the **fit is stable to three decimal places and the known parameter is
recovered every time**, while the split of degradation between the two latent channels
varies by a factor of two or more from seed to seed. The single-run numbers above
(0.86 / 0.93) are a favourable seed and must not be quoted alone.

This is a genuine identifiability limit, not an optimisation bug: early in a stint wear
and graining both rise, so they are partially exchangeable, and only the *late*-stint
cleaning of graining distinguishes them. The observation channel — one scalar per lap —
simply does not contain enough information to separate two latent states robustly.

The practical readings:

* trust the UDE's **total** degradation and its **known** parameters;
* do not trust an individual latent channel without a second observation that sees it
  (a tyre-temperature or wear sensor, or deliberately varied conditions);
* the wear-only model is the honest choice when only lap time is available.

This is the sort of thing an encoded model makes *visible*: because the states are named
and physically meaningful, their unreliability can be measured. In a black-box model the
same ambiguity exists and is simply invisible.
:::

The wear-only ablation is nonetheless informative: with nowhere to put the non-monotone
graining signal it absorbs part of it into wear. The linear model cannot represent
non-monotone degradation at all — that is asserted as a test, not just claimed.

### Real F1 data — an honest, more modest result

Eight 2023 races, 7 223 valid dry laps, 20 drivers, track temperature 299–313 K. Whole
sessions held out. (Wet and intermediate running is excluded by default — it is a
different regime, and leaving the one wet race in moved the held-out error by a factor
of four.)

| model | params | train RMSE [s] | test RMSE [s] |
|---|---|---|---|
| linear in tyre age | 45 | 1.32 | **0.82** |
| black-box MLP | 1 424 | **0.49** | 1.68 |
| UDE, wear only | 1 304 | 1.27 | 1.07 |
| UDE, wear + graining | 1 304 | 1.27 | 1.10 |

Read this carefully, because it does **not** say the UDE wins:

1. **The linear model wins, and generalises best by a clear margin.** Over a typical
   14-lap stint real degradation is close to linear, so the simplest model with 45
   parameters is hard to beat. The UDE is not better here, and this page does not
   pretend otherwise.
2. **The black box overfits by more than a factor of three** (best train by far, worst
   test). With eight circuits and a strong pace confound, unconstrained flexibility is a
   liability, not an advantage.
3. **The UDE's value here is that it returns a state.** It reports wear in seconds and
   graining in $[0,1]$, guaranteed monotone and bounded, from data containing neither —
   and it sits much closer to the constrained baseline than to the black box on
   held-out races.
4. **This dataset cannot identify graining.** The learned graining is ~0 throughout.
   Track temperature spans 14 K across eight races and barely varies within one, so the
   thermal gating has almost nothing to learn from, and graining is confounded with the
   early-stint fuel-heavy laps. Reporting "no identifiable graining" is the correct
   answer for this data, not a failure of the model.
5. **The learned fuel coefficient (~5 s) is higher than the physical ~3.5 s**, because
   fuel fraction and tyre age are strongly correlated within a stint. With eight races
   the two are only weakly separable. This is a caveat on the fitted parameters, not
   just on the RMSE.

The synthetic and real results together are the useful pair: the method demonstrably
recovers degradation dynamics when they are present and observable, and on this
particular real dataset the honest conclusion is that **a linear model is sufficient**
and the value of the UDE is interpretability rather than accuracy.

### A third thing the data forced: bound the rate

The first real-data run produced a tyre losing **9 seconds** of lap time to wear over a
stint, with a fuel effect of 5.8 s for a full tank. Both are physically absurd; in
sample they were absorbed by the pace offsets, and out of sample they were badly wrong.
The fix was to apply this framework's own bounded-parameter principle to the kinetics —
the wear rate is now ``sigmoid`` into $[0, r_{\max}]$ with $r_{\max}$ itself a bounded
learnable parameter capped at 0.3 s/lap — rather than an unbounded ``softplus``. Held-out
error improved from 1.74 s to 1.10 s and peak wear fell from 9.1 s to 2.3 s.

That is the general lesson of the whole project, arriving again in a new context: a
quantity with known physical bounds should be given those bounds by construction.

## Running it

```bash
python experiments/train_degradation_ude.py                          # real data if present
python experiments/train_degradation_ude.py --set data.source=synthetic

# reproduce the identifiability study above
python experiments/train_degradation_ude.py --set data.source=synthetic --seeds 6
```

Notebook: [Tyre degradation as a UDE](../notebooks/04_tyre_degradation_ude).

## Further reading

- {cite}`cappello2025f1degradation` — a Bayesian state-space model of exactly this
  observation (lap time as fuel plus latent tyre pace, pit stops as state resets), on
  the same FastF1 data. The direct precedent for the observation model here.
- {cite}`kuzhiyil2025batteryude` — the closest methodological analogue: lithium-ion
  battery degradation as a UDE, with a known structure and neural closure terms
  identified from indirect measurements across widely separated time scales.
- {cite}`maglione2026rubberwear` — a survey of tyre-wear models from empirical to
  data-driven, useful for choosing what to encode if you have wear measurements rather
  than lap times.
- {cite}`rackauckas2020ude` — the UDE formalism itself.
- {cite}`brunton2016sindy` — sparse regression for recovering a *symbolic* rate law
  from a trained closure. Not implemented here; the natural next step.
