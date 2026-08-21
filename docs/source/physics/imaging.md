# Imaging tyre condition

[Degradation from stint data](degradation.md) ended on a negative result: wear and
graining are **not separately identifiable from lap time**. Across seeds the lap-time fit
was stable to three decimals while the recovered wear wandered between 0.29 and 0.63
correlation with the hidden truth. One scalar per lap cannot resolve two latent states.

That is not an optimiser problem, and no amount of architecture fixes it. It is a
shortage of information, and the fix is a **second observation channel**. A camera
pointed at the tread is exactly that: tread depth sees wear directly, surface texture
sees graining directly.

## What real datasets actually label

Public tyre imagery is plentiful and almost entirely *condition classification*:

| dataset | labels | useful for |
|---|---|---|
| Kaggle *Tyre Quality Classification* | cracked / normal | damage, **not** wear |
| Kaggle *Tyre Condition Classification* | new / serviceable / unusable, real workshop photos | **ordinal wear** |
| Kaggle *Tire Texture Image Recognition* | cracked / normal | damage |
| Mendeley *TyreNet* (1 698 images) | defective / good, expert annotated | damage |

**No public dataset of measured tread depth exists, and none of graining.** That fact
shapes what is honest to build:

* the classes *new < serviceable < unusable* are a genuine wear ordering, so real data
  can supervise a **monotone wear index**;
* graining has no public imagery at all, so it is demonstrated on **synthetic textures**
  and labelled as such everywhere.

{py:data}`tire_nn.data.tread_images.REAL_IMAGE_DATASETS` records what each set labels,
and a test asserts that none of them claims to provide depth — a guard against silently
treating an ordinal class as a measurement.

## The encoded vision model

{py:class}`tire_nn.models.condition_vision.TreadConditionNet` builds in three things.

**A single monotone wear index.** Wear is one number that only increases. The network
emits one latent $w \in [0,1]$ and every downstream quantity is a function of it. A plain
three-way classifier can place a photograph in *unusable* while ranking it less worn than
a *serviceable* one; here that is unrepresentable.

**Ordered class boundaries**, built cumulatively so the ordering holds for any weights:

```{math}
:label: eq-ordinal-thresholds
t_1 = \sigma(b_1), \qquad t_2 = t_1 + \mathrm{softplus}(b_2), \qquad
P(y > k) = \sigma\big(s\,(w - t_k)\big)
```

$P(y>k)$ is then monotone in $w$ and decreasing in $k$ by construction, so the class
probabilities are automatically a valid distribution over an *ordered* set.

**A bounded graining fraction**, a sigmoid output in $[0,1]$ matching the state variable
in the [condition model](thermal-wear.md).

### Measured: continuous wear from ordinal labels

Trained on synthetic imagery with only the three ordered classes as supervision (plus a
graining term, which real data cannot provide):

| quantity | result |
|---|---|
| ordinal accuracy | 0.978 |
| **wear index vs the continuous truth** | **r = 0.983** |
| graining vs truth | r = 0.982, MAE 0.068 |
| learned thresholds | 0.358, 0.857 (true 0.35, 0.75) |

The monotone latent recovers a continuous wear scale from three classes, with thresholds
landing near the true boundaries. Nothing in the loss ever saw a continuous wear value —
that is the encoded structure doing the work, and it is what makes ordinal real data
useful to a physical model.

## Closing the identifiability gap

The experiment this page exists for. Take the degradation UDE and give it **one
photograph per pit stop**: a single noisy graining reading at the end of each stint, with
noise matching the vision model above (MAE 0.066).

| supervision | wear corr. (mean) | spread across seeds | lap-time RMSE [s] |
|---|---|---|---|
| lap time only | 0.606 | **0.283 – 0.794** | 0.242 |
| **lap time + image** | **0.941** | **0.934 – 0.952** | 0.247 |

The seed-to-seed spread collapses from 0.511 to 0.018 — **28× narrower** — while the
lap-time fit is essentially unchanged. That is the signature of **added information rather than
added flexibility**: the model does not fit better, it stops being ambiguous.

This is the practical answer to "which latent channel do I trust?" — do not trust a
channel no measurement sees. Add the measurement, or report only the total.

## Closing the loop

The estimated state is not just a diagnostic: it is the input
{py:class}`~tire_nn.models.thermo_graining_tire.ThermoGrainingTire` already accepts, so
image → state → friction closes the loop. And because the condition only *scales* the
friction ellipse, odd symmetry, zero force at zero slip and the hard envelope all still
hold, now against a condition-dependent limit.

```python
from tire_nn.models.condition_vision import TreadConditionNet
from tire_nn.data.tread_images import make_tread_dataset

model = TreadConditionNet()
out = model(images)
out["wear"], out["graining"], out["class_prob"], out["thresholds"]
```

Notebook: [Imaging tyre condition](../notebooks/06_condition_from_images).

:::{warning}
Every number on this page comes from **synthetic imagery**, because no public
tread-depth or graining dataset exists. The vision model is deliberately built to consume
what real data *does* provide — ordered classes — but none of this is evidence about real
tyres. Using it on real photographs requires your own calibrated measurements to fix the
scale.
:::

## Tests

`tests/test_condition_vision.py`: `test_thresholds_are_ordered_for_adversarial_weights`,
`test_class_ranking_is_monotone_in_the_wear_index`,
`test_vision_model_learns_wear_from_ordinal_labels_only`,
`test_real_dataset_registry_states_what_each_one_labels`,
`test_groove_contrast_falls_as_wear_increases`.
