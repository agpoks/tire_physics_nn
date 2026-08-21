"""Estimating tyre condition from imagery, with the ordinal structure encoded.

This module exists to close a gap the rest of the project measured. Experiment 5 showed
that wear and graining are **not separately identifiable from lap time**: one scalar per
lap cannot resolve two latent states, so the fit is stable while the decomposition
wanders between seeds. A camera is a second observation channel that sees the states
*directly* — tread depth is wear, surface texture is graining — and this model turns an
image into the condition state that
:py:class:`tire_nn.models.thermo_graining_tire.ThermoGrainingTire` consumes.

What is encoded
---------------

**A single monotone wear index.** Wear is one number that only increases. The network
outputs one latent :math:`w \\in [0,1]` through a sigmoid, and every downstream quantity
is a function of it. A plain three-way classifier can put a photograph in *unusable*
while ranking it as less worn than a *serviceable* one; here that is unrepresentable.

**Ordered class boundaries.** The ordinal thresholds are built cumulatively,

.. math:: t_1 = \\sigma(b_1), \\qquad t_2 = t_1 + \\mathrm{softplus}(b_2),

so :math:`t_1 < t_2` holds for any weights. The class probabilities come from the
cumulative form :math:`P(y > k) = \\sigma\\big(s\\,(w - t_k)\\big)`, which is monotone in
:math:`w` by construction — the model cannot produce a non-monotone ranking.

**A bounded graining fraction.** Graining is the *share* of the surface affected, so it
is a sigmoid output in :math:`[0,1]`, matching the state variable in the condition model.

Why this matters practically: real public tyre imagery is labelled *ordinally*
(new / serviceable / unusable) and never with measured depth, so the ordinal head is the
only thing real data can supervise — while the latent it thresholds is exactly the
continuous quantity the physics wants.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

__all__ = ["TreadConditionNet", "ordinal_loss"]


class TreadConditionNet(nn.Module):
    """Small CNN mapping a tread image to a condition state.

    Args:
        n_classes: number of ordinal wear classes (3 for new/serviceable/unusable).
        predict_graining: also emit a bounded graining fraction. Requires supervision
            that actually sees graining — no public dataset does, so this is exercised
            on synthetic imagery.
        width: base channel count. Deliberately small: this is a texture-statistics
            problem, not ImageNet.
    """

    def __init__(self, n_classes: int = 3, predict_graining: bool = True,
                 width: int = 16, sharpness: float = 12.0):
        super().__init__()
        self.n_classes = int(n_classes)
        self.predict_graining = bool(predict_graining)
        self.sharpness = float(sharpness)

        self.features = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 2, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(width * 2, 2 if predict_graining else 1)
        # Raw thresholds; the ordering is imposed in `thresholds`, not learned.
        self.raw_thresholds = nn.Parameter(torch.zeros(n_classes - 1))

    def thresholds(self) -> Tensor:
        """Class boundaries on the wear index, strictly increasing for any weights."""
        first = torch.sigmoid(self.raw_thresholds[:1])
        steps = F.softplus(self.raw_thresholds[1:])
        return torch.cat([first, first + torch.cumsum(steps, dim=0)])

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        z = self.head(self.features(images).flatten(1))
        wear = torch.sigmoid(z[:, 0])
        out = {"wear": wear}
        if self.predict_graining:
            out["graining"] = torch.sigmoid(z[:, 1])

        # Cumulative ordinal probabilities: P(class > k), monotone in the wear index.
        t = self.thresholds()
        out["cumulative"] = torch.sigmoid(self.sharpness * (wear.unsqueeze(-1) - t))
        # Turn them into class probabilities, P(class = k).
        ones = torch.ones_like(out["cumulative"][:, :1])
        upper = torch.cat([ones, out["cumulative"]], dim=-1)
        lower = torch.cat([out["cumulative"], torch.zeros_like(ones)], dim=-1)
        out["class_prob"] = torch.clamp(upper - lower, min=1e-8)
        out["thresholds"] = t
        return out


def ordinal_loss(cumulative: Tensor, labels: Tensor) -> Tensor:
    """Binary cross-entropy on the cumulative targets ``1[y > k]``.

    The standard ordinal (Frank & Hall style) formulation: each threshold is a binary
    problem, and because the thresholds are ordered by construction the resulting class
    probabilities are automatically a valid distribution over an ordered set. Training a
    plain softmax classifier instead would throw the ordering away and let the model
    treat *new* and *unusable* as equally different from *serviceable*.
    """
    k = torch.arange(cumulative.shape[-1], device=labels.device)
    targets = (labels.unsqueeze(-1) > k).to(cumulative.dtype)
    return F.binary_cross_entropy(cumulative, targets)
