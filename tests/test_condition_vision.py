"""Tread imagery and the vision condition model.

The images are synthetic (no public tread-depth or graining dataset exists — see
:py:data:`tire_nn.data.tread_images.REAL_IMAGE_DATASETS`), but the encoded properties
of the model must hold regardless of where the pixels come from.
"""

import numpy as np
import pytest
import torch

from tire_nn.data.tread_images import (
    ORDINAL_CLASSES,
    REAL_IMAGE_DATASETS,
    load_condition_images,
    make_tread_dataset,
    make_tread_image,
)
from tire_nn.models.condition_vision import TreadConditionNet, ordinal_loss
from conftest import randomize_


# --- the imagery -----------------------------------------------------------

def test_tread_image_shape_and_range():
    img = make_tread_image(0.3, 0.2, size=48)
    assert img.shape == (48, 48)
    assert 0.0 <= img.min() and img.max() <= 1.0


def test_groove_contrast_falls_as_wear_increases():
    """The physical content of the image: worn grooves are shallower, so flatter."""
    rng = np.random.default_rng(0)
    contrasts = [make_tread_image(w, 0.0, size=64, rng=rng, noise=0.0).std()
                 for w in (0.0, 0.35, 0.7, 1.0)]
    assert all(a > b for a, b in zip(contrasts, contrasts[1:])), contrasts


def test_graining_adds_texture_at_fixed_wear():
    rng = np.random.default_rng(1)
    plain = make_tread_image(0.3, 0.0, size=64, rng=rng, noise=0.0)
    grained = make_tread_image(0.3, 0.9, size=64, rng=rng, noise=0.0)
    # Graining shows up as extra high-frequency structure across the tread.
    assert np.abs(np.diff(grained, axis=0)).mean() > np.abs(np.diff(plain, axis=0)).mean()


def test_dataset_labels_are_ordered_by_wear():
    d = make_tread_dataset(n=300, size=32, seed=0)
    for lower, upper in ((0, 1), (1, 2)):
        a = d["wear"][d["label"] == lower]
        b = d["wear"][d["label"] == upper]
        if len(a) and len(b):
            assert float(a.max()) <= float(b.min()) + 1e-6


def test_real_dataset_registry_states_what_each_one_labels():
    """Guards against silently treating an ordinal class as a depth measurement."""
    for name, info in REAL_IMAGE_DATASETS.items():
        assert info["source"] and info["labels"] and info["target"]
        assert "depth" not in info["target"], (
            f"{name} claims to provide depth; no public dataset does")


def test_missing_real_images_fail_with_instructions(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        load_condition_images(tmp_path)
    message = str(exc.value)
    assert "Kaggle" in message and "Mendeley" in message


# --- the encoded model -----------------------------------------------------

def test_thresholds_are_ordered_for_adversarial_weights():
    model = randomize_(TreadConditionNet(n_classes=3), std=20.0)
    t = model.thresholds()
    assert torch.all(torch.diff(t) > 0), t


def test_more_classes_still_gives_ordered_thresholds():
    model = randomize_(TreadConditionNet(n_classes=5), std=10.0)
    assert torch.all(torch.diff(model.thresholds()) > 0)


def test_outputs_are_bounded_and_probabilities_are_valid():
    model = randomize_(TreadConditionNet(), std=8.0)
    d = make_tread_dataset(n=24, size=32, seed=2)
    out = model(d["images"])
    assert torch.all(out["wear"] >= 0) and torch.all(out["wear"] <= 1)
    assert torch.all(out["graining"] >= 0) and torch.all(out["graining"] <= 1)
    assert torch.allclose(out["class_prob"].sum(-1), torch.ones(24), atol=1e-5)
    assert torch.all(out["class_prob"] >= 0)


def test_class_ranking_is_monotone_in_the_wear_index():
    """A plain classifier can rank an 'unusable' tyre as less worn than a 'serviceable'
    one. Here that is unrepresentable, which is the whole point of the ordinal head."""
    model = randomize_(TreadConditionNet(), std=5.0)
    wear = torch.linspace(0.0, 1.0, 50)
    t = model.thresholds()
    cumulative = torch.sigmoid(model.sharpness * (wear.unsqueeze(-1) - t))
    # P(class > k) rises with the wear index, and falls with k.
    assert torch.all(torch.diff(cumulative, dim=0) >= -1e-6)
    assert torch.all(torch.diff(cumulative, dim=-1) <= 1e-6)
    predicted = cumulative.sum(-1)                    # expected class index
    assert torch.all(torch.diff(predicted) >= -1e-6)


def test_ordinal_loss_prefers_the_correct_ordering():
    """Predicting a neighbouring class must cost less than predicting the far one."""
    labels = torch.tensor([2, 2, 2])
    near = torch.tensor([[0.9, 0.6], [0.9, 0.6], [0.9, 0.6]])    # says "≈ class 1-2"
    far = torch.tensor([[0.1, 0.02], [0.1, 0.02], [0.1, 0.02]])  # says "class 0"
    assert float(ordinal_loss(near, labels)) < float(ordinal_loss(far, labels))


def test_vision_model_learns_wear_from_ordinal_labels_only():
    """The encoded monotone latent should recover a *continuous* quantity from three
    ordered classes — which is all any real tyre dataset provides."""
    torch.manual_seed(0)
    train = make_tread_dataset(n=400, size=32, seed=0)
    test = make_tread_dataset(n=150, size=32, seed=1)
    model = TreadConditionNet(width=12, predict_graining=False)
    opt = torch.optim.Adam(model.parameters(), lr=4e-3)
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        loss = ordinal_loss(model(train["images"])["cumulative"], train["label"])
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        wear = model(test["images"])["wear"]
    corr = float(np.corrcoef(wear.numpy(), test["wear"].numpy())[0, 1])
    assert corr > 0.7, f"monotone latent only reached r={corr:.2f} against the true wear"


# --- real photographs (skipped unless downloaded) ---------------------------

def _real_images():
    from tire_nn.data.tread_images import load_tyre_quality_images

    try:
        return load_tyre_quality_images("data/raw", size=64)
    except FileNotFoundError:
        pytest.skip("real tyre photographs not downloaded "
                    "(scripts/download_tyre_images.py)")


def test_real_images_load_in_the_expected_shape():
    real = _real_images()
    assert real["images"].ndim == 4 and real["images"].shape[1] == 1
    assert real["label"].max() == 1
    assert real["label_type"] == "binary condition", (
        "the label type must say what it is: these are not tread-depth measurements")


def test_real_images_are_normalised():
    real = _real_images()
    assert float(real["images"].min()) >= 0.0
    assert float(real["images"].max()) <= 1.0


def test_model_separates_real_classes_better_than_chance():
    """The encoded ordinal model must work on real photographs, not just synthetic ones."""
    from torch.utils.data import DataLoader, TensorDataset

    real = _real_images()
    torch.manual_seed(0)
    order = torch.randperm(len(real["images"]), generator=torch.Generator().manual_seed(0))
    images, labels = real["images"][order], real["label"][order]
    split = int(0.75 * len(images))

    model = TreadConditionNet(n_classes=2, predict_graining=False, width=12)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loader = DataLoader(TensorDataset(images[:split], labels[:split]),
                        batch_size=32, shuffle=True)
    for _ in range(12):
        for img, lab in loader:
            opt.zero_grad(set_to_none=True)
            ordinal_loss(model(img)["cumulative"], lab).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        out = model(images[split:])
    accuracy = float((out["class_prob"].argmax(-1) == labels[split:]).float().mean())
    assert accuracy > 0.6, f"only {accuracy:.2f} on real photographs (chance is 0.5)"

    low = out["wear"][labels[split:] == 0].mean()
    high = out["wear"][labels[split:] == 1].mean()
    assert float(high) > float(low), "the wear index must order good below defective"
