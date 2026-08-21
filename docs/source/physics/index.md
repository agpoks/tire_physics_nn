# Tire physics for modelling

This section is a survey of the physics available for building a tire model: what each
effect is, what equation describes it, what it costs, and what it gets wrong. It is
written to be usable independently of the neural-network parts of this project — the
[method](../methods/taxonomy) and [model](../models/index) sections then show how each
of these pieces is integrated into a learned model.

## The modelling stack

A tire model is a stack of largely separable questions. You can pick a different answer
at each level, and this framework lets you.

| Level | Question | Options | Page |
|---|---|---|---|
| 0 | What *is* slip? | kinematic definitions, sign conventions | [Conventions](conventions) |
| 1 | Force from slip, one axis | linear · brush · Dugoff · Magic Formula | [Steady state](steady-state) |
| 1b | ...or keep the contact patch explicit | discretised brush PDE on a chain | [The contact patch as a PDE](contact-patch) |
| 2 | Force under simultaneous $\alpha$ and $\kappa$ | friction ellipse · similarity method · MF weighting functions | [Combined slip](combined-slip) |
| 3 | How force depends on load | linear in $F_z$ · load-sensitive $\mu(F_z)$ | [Steady state](steady-state) |
| 4 | How force responds in time | instantaneous · relaxation length · full carcass dynamics | [Transient](transient) |
| 5 | How condition changes grip | none · thermal · + wear · + graining | [Thermal, wear, graining](thermal-wear) |
| 5b | How condition is identified when it is never measured | UDE from lap-time data | [Degradation from stint data](degradation) |
| 5c | How a second observation channel resolves what one cannot | encoded vision model, ordinal labels | [Imaging tyre condition](imaging) |
| 6 | How four tires make a vehicle move | bicycle · four-wheel Newton–Euler | [Vehicle](vehicle.md) |

Levels 1–3 are the *constitutive* model — the tire's own force law. Level 4 makes it
dynamic, level 5 makes it non-stationary, level 6 embeds it in a vehicle.

## What is actually uncertain

Not all of this is equally unknown, and that distinction drives every design decision in
this project:

**Definitions** (level 0) are exact. Slip follows from rigid-body kinematics and a wheel
speed measurement. Nothing here should ever be learned.

**Conservation laws** (level 6) are exact. Newton–Euler with known geometry is not an
approximation. Learning a correction to it means learning to compensate for a
measurement error somewhere else, and it destroys the identifiability of the tire.

**Structural facts** are certain even when their coefficients are not. A tire produces
no force at zero slip; force cannot exceed available friction; the transient is
parameterised by rolling distance; wear does not reverse. These are the properties this
project *encodes*: the shape is certain, the numbers are not.

**Constitutive detail** is genuinely uncertain — the exact shape of $F_y(\alpha)$ for a
particular compound at a particular temperature is what you need data for. This is what
the network is for.

The mistake this framework is designed to avoid is treating all four the same way: a
black-box model spends capacity re-learning the definitions and the conservation laws,
and still gets the structural facts wrong where data is thin.

## Reading order

Each page states the physics first, then shows it, then evaluates it. If you only read
one, read [Steady state](steady-state): it contains the comparison of the four classical
force laws that everything else builds on.
