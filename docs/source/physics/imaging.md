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

TyreNet is directly downloadable without credentials, but it is a **1.9 GB `.rar`**, so
this project does not fetch it automatically (PLAN.md §4.4: nothing large downloads on
its own). The verified URL and size are in the registry; fetch it deliberately if you
want to work with real photographs.

## The imagery used here

```{figure} ../_static/figures/tread_images.png
:alt: synthetic tread images across wear and graining
:width: 100%

Top: wear shallows and narrows the grooves, so image contrast carries the depth
information — at `wear = 1` the grooves have closed up entirely. Bottom: graining adds
mottled, torn-looking texture between the grooves at fixed wear. Two visually separable
channels, which is what makes the identifiability experiment below meaningful.
```

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

```{figure} ../_static/figures/identifiability_gain.png
:alt: wear correlation with and without the image channel
:width: 100%

Four seeds per condition. Left: each point is one run; the bar is the mean. Right: the
spread the runs disagree over.
```

| supervision | wear corr. (mean) | spread across seeds | lap-time RMSE [s] |
|---|---|---|---|
| lap time only | 0.606 | **0.283 – 0.794** | 0.242 |
| **lap time + image** | **0.941** | **0.934 – 0.952** | 0.247 |

The seed-to-seed spread collapses from 0.511 to 0.018 — **28× narrower** — while the
lap-time fit is essentially unchanged. That is the signature of **added information rather than
added flexibility**: the model does not fit better, it stops being ambiguous.

This is the practical answer to "which latent channel do I trust?" — do not trust a
channel no measurement sees. Add the measurement, or report only the total.


## On real photographs

Everything above uses synthetic textures. That is a real limitation, and it can now be
checked, because one openly-licensed source of real tyre imagery needs no account:
**NMiriams/Good_Tires** and **NMiriams/Defective_Tires** on the Hugging Face Hub,
~1 850 photographs under **CC BY 4.0**.

```bash
python -m pip install huggingface_hub
python scripts/download_tyre_images.py --limit 250
python experiments/train_vision_condition.py
```

:::{warning}
The labels are **binary condition** — good versus defective — and "defective" mixes
tread wear with cracking, bulges and punctures. This is **not** a graded wear scale and
contains **no measured tread depth**. It supports one question: does a monotone latent
order real photographs sensibly? It does not support estimating how worn a tyre is.
:::

```{figure} ../_static/figures/real_tyre_images.png
:alt: real tyre photographs and the model's wear index on them
:width: 100%

Top: real photographs from the dataset — close-ups of tread and sidewall. Bottom left:
trained on real images, the wear index separates the two classes cleanly either side of
the learned threshold. Bottom right: the same architecture trained on synthetic textures
and applied to those photographs. Images CC BY 4.0.
```

| setting | accuracy | wear index, low class | high class | ordered correctly |
|---|---|---|---|---|
| synthetic → synthetic | 1.000 | 0.043 | 0.953 | ✅ |
| **real → real** | **0.816** | 0.297 | 0.630 | ✅ |
| **synthetic → real** (transfer) | **0.464** | 0.470 | 0.400 | ❌ **inverted** |

Chance level is 0.500.

Two conclusions, one encouraging and one not:

**The encoded model works on real data.** Trained on photographs, the monotone wear index
orders good below defective with a clean separation, at 0.82 accuracy from 375 training
images and a network of about 5 000 parameters. The ordinal structure is doing real work
here: the classes are all the supervision available, and the latent behind them is still
a usable continuous scale.

**The synthetic textures do not transfer at all.** A model trained on them performs at
chance on real photographs, and its wear index is *inverted*. This is worth stating
plainly: the synthetic generator is a stand-in for the **structure** of the problem —
two visually separable channels driving a latent state — and not a renderer of real
tyres. Look at the photographs above and the reason is obvious: they are close-ups at
varying scale, lighting and framing, with sidewall lettering in frame, nothing like the
clean groove textures the generator produces.

**What this means for the identifiability result.** The experiment showing that one
photograph per pit stop collapses the wear/graining ambiguity was run entirely in the
synthetic domain, and it remains a statement about *information*, not about any
particular camera: it says that a second channel observing the state resolves what
lap time alone cannot. Reproducing it with real imagery would need something no public
dataset provides — photographs of the *same tyre* through a stint, with the lap times
alongside.

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
