"""Tread and graining imagery: synthetic generator, and adapters for real datasets.

Why images belong in this project
---------------------------------
Experiment 5 found that tyre wear and graining are **not separately identifiable** from
lap time: one scalar per lap cannot resolve two latent states, so the fit is stable while
the decomposition wanders. The fix is not a better optimiser, it is a **second
observation channel** — and a camera looking at the tread is exactly that. Tread depth
sees wear directly; surface texture sees graining directly.

What real data exists (and what does not)
-----------------------------------------
Public tyre imagery is plentiful but it is nearly all **condition classification**, not
the regression targets a physical model wants:

=================================================  ==========================================
dataset                                            what it labels
=================================================  ==========================================
Kaggle *Tyre Quality Classification*               cracked / normal
Kaggle *Tyre Condition Classification*             NEW / SERVICEABLE / UNUSABLE (real workshop
                                                   photos)
Kaggle *Tire Texture Image Recognition*            cracked / normal
Mendeley *TyreNet* (1 698 images)                  defective / good, expert annotated
=================================================  ==========================================

No public dataset of **measured tread depth** and none of **graining** could be found.
That shapes what is honest to do here:

* the ordinal classes NEW < SERVICEABLE < UNUSABLE *are* a monotone wear scale, so real
  data can supervise a monotone wear index — see
  :py:class:`tire_nn.models.condition_vision.TreadConditionNet`;
* graining has no public imagery at all, so it is demonstrated on **synthetic textures**
  generated here and labelled as such everywhere.

Every adapter reports the label type it provides so a downstream experiment cannot
silently treat an ordinal class as a depth measurement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "ORDINAL_CLASSES",
    "load_tyre_quality_images",
    "REAL_IMAGE_DATASETS",
    "make_tread_image",
    "make_tread_dataset",
    "load_condition_images",
]

#: Ordinal wear classes, ordered. The ordering is the physical content: wear only
#: increases, so a model's class boundaries must respect this sequence.
ORDINAL_CLASSES = ("new", "serviceable", "unusable")

#: Public datasets, with what they actually label. None provides measured tread depth.
REAL_IMAGE_DATASETS = {
    "tyre_quality": dict(
        source="Hugging Face: NMiriams/Good_Tires + NMiriams/Defective_Tires "
               "(CC BY 4.0, ~1850 images, no credentials); also mirrored on Kaggle as "
               "warcoder/tyre-quality-classification",
        labels="binary (good / defective)", target="binary condition",
        note="The only real tyre imagery here that downloads without an account. "
             "'Defective' mixes tread wear with cracking, bulges and punctures, so it "
             "is a condition label, NOT a graded wear scale."),
    "tyre_condition": dict(
        source="Kaggle: sameersambhare1/tyre-condition-classification-dataset",
        labels="ordinal (new / serviceable / unusable)", target="ordinal wear",
        note="Real workshop photographs. The most useful of the four here, because the "
             "classes are genuinely ordered by wear."),
    "tire_texture": dict(
        source="Kaggle: jehanbhathena/tire-texture-image-recognition",
        labels="binary (cracked / normal)", target="classification", note=""),
    "tyrenet": dict(
        source="Mendeley Data 32b5vfj6tc: TyreNet, 1698 images",
        labels="binary (defective / good), expert annotated", target="classification",
        note="Collected from six service stations and two showrooms. Downloadable "
             "without credentials, but it is a single 1.9 GB .rar archive, so this "
             "project does not fetch it automatically (see PLAN.md 4.4). Get the "
             "current link from https://data.mendeley.com/datasets/32b5vfj6tc/1 and "
             "extract into data/raw/tyre_condition/<class>/."),
}


def make_tread_image(
    wear: float,
    graining: float = 0.0,
    size: int = 64,
    n_grooves: int = 4,
    rng: np.random.Generator | None = None,
    noise: float = 0.04,
) -> np.ndarray:
    """Render one **synthetic** tread patch as a grayscale image in ``[0, 1]``.

    The generator encodes what a camera would actually see:

    * **wear** shallows and narrows the grooves — at ``wear = 1`` they have closed up,
      so the image contrast carries the depth information;
    * **graining** adds a mottled, torn-looking texture to the rubber between the
      grooves, which is roughly what graining looks like: many small tears that fuse.

    This is a *stand-in for imagery that does not exist publicly*, not a rendering
    model. It is used to demonstrate that a second observation channel restores the
    identifiability that lap time alone lacks; it is not evidence about real tyres.
    """
    rng = rng or np.random.default_rng(0)
    wear = float(np.clip(wear, 0.0, 1.0))
    graining = float(np.clip(graining, 0.0, 1.0))

    y, x = np.mgrid[0:size, 0:size] / size
    image = np.full((size, size), 0.55)

    # Grooves: dark bands whose depth (contrast) and width shrink with wear.
    depth = (1.0 - wear) ** 1.2
    width = 0.055 * (1.0 - 0.55 * wear)
    for i in range(n_grooves):
        centre = (i + 0.5) / n_grooves
        band = np.exp(-0.5 * ((x - centre) / max(width, 1e-3)) ** 2)
        image -= 0.42 * depth * band

    # Graining: correlated mottling on the rubber, strongest away from the grooves.
    if graining > 0:
        coarse = rng.normal(size=(size // 4, size // 4))
        texture = np.kron(coarse, np.ones((4, 4)))[:size, :size]
        texture = (texture - texture.mean()) / (texture.std() + 1e-9)
        image += 0.16 * graining * texture

    image += rng.normal(scale=noise, size=image.shape)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def make_tread_dataset(
    n: int = 600,
    size: int = 64,
    seed: int = 0,
    graining_prob: float = 0.5,
) -> dict[str, Tensor]:
    """A **synthetic** image dataset with known wear and graining.

    Returns tensors ``images (n,1,H,W)``, ``wear (n,)``, ``graining (n,)`` and the
    derived ordinal ``label (n,)`` using the same thresholds a workshop would apply.
    Having both the continuous truth and the ordinal label is what lets the ordinal
    model be scored against the quantity it is really estimating.
    """
    rng = np.random.default_rng(seed)
    wear = rng.uniform(0.0, 1.0, n)
    graining = np.where(rng.random(n) < graining_prob, rng.uniform(0.0, 1.0, n), 0.0)
    images = np.stack([make_tread_image(w, g, size=size, rng=rng)
                       for w, g in zip(wear, graining)])
    label = np.digitize(wear, [0.35, 0.75])          # new < serviceable < unusable
    return {
        "images": torch.as_tensor(images).unsqueeze(1),
        "wear": torch.as_tensor(wear, dtype=torch.float32),
        "graining": torch.as_tensor(graining, dtype=torch.float32),
        "label": torch.as_tensor(label, dtype=torch.long),
    }


def load_tyre_quality_images(
    root: str | Path = "data/raw",
    subdir: str = "tyre_quality",
    size: int = 64,
    grayscale: bool = True,
):
    """Load the real good/defective tyre photographs fetched by
    ``scripts/download_tyre_images.py``.

    **Type: real measurement (photographs), CC BY 4.0.** Labels are binary condition,
    ordered ``good`` (0) < ``defective`` (1) by severity, which is the weakest possible
    ordinal scale but a genuine one — enough to ask whether a monotone latent orders
    real images sensibly.

    What this data cannot do: it has no measured tread depth, and "defective" includes
    damage modes (cracks, bulges, punctures) that are not wear at all. A model trained
    on it estimates *condition*, not wear, and reporting it as wear would be exactly the
    substitution this project's dataset registry exists to prevent.
    """
    import numpy as np

    root = Path(root) / subdir
    classes = ("good", "defective")
    if not root.exists():
        raise FileNotFoundError(
            f"no tyre photographs under {root}.\n"
            "Fetch them (CC BY 4.0, no account needed) with:\n"
            "    python -m pip install huggingface_hub\n"
            "    python scripts/download_tyre_images.py\n"
        )
    try:
        from PIL import Image
    except ImportError as exc:                        # pragma: no cover
        raise ImportError("reading images needs pillow: pip install pillow") from exc

    images, labels = [], []
    for index, name in enumerate(classes):
        folder = root / name
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*")):
            if path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            img = Image.open(path)
            img = img.convert("L" if grayscale else "RGB").resize((size, size))
            images.append(np.asarray(img, dtype=np.float32) / 255.0)
            labels.append(index)
    if not images:
        raise FileNotFoundError(f"found {root} but no images inside it")

    array = np.stack(images)
    if grayscale:
        array = array[:, None]
    else:
        array = array.transpose(0, 3, 1, 2)
    counts = {c: labels.count(i) for i, c in enumerate(classes)}
    print(f"[tyre_quality] {len(images)} real photographs {counts}, "
          f"binary condition labels (NOT measured tread depth)")
    return {
        "images": torch.as_tensor(array),
        "label": torch.as_tensor(labels, dtype=torch.long),
        "classes": classes,
        "label_type": "binary condition",
    }


def load_condition_images(root: str | Path = "data/raw", subdir: str = "tyre_condition",
                          size: int = 64) -> dict:
    """Load a real ordinal tyre-condition dataset laid out as ``<root>/<subdir>/<class>/*.jpg``.

    **Type: real measurement (photographs), ordinal labels only.** There is no measured
    tread depth in any public set, so what this supervises is a *monotone wear index*,
    not a depth in millimetres. Anything quantitative requires your own calibrated
    measurements.
    """
    root = Path(root) / subdir
    if not root.exists():
        raise FileNotFoundError(
            f"no tyre-condition images under {root}.\n"
            "Manual download (no automatic fetch — these are Kaggle/Mendeley sets with\n"
            "their own terms):\n"
            "  Kaggle: sameersambhare1/tyre-condition-classification-dataset\n"
            "  or Mendeley Data 32b5vfj6tc (TyreNet)\n"
            f"Extract so that {root}/<class>/*.jpg exists, with class folders named "
            f"{list(ORDINAL_CLASSES)}.\n"
            "See tire_nn/data/tread_images.REAL_IMAGE_DATASETS for what each set labels."
        )
    try:
        from PIL import Image
    except ImportError as exc:                        # pragma: no cover
        raise ImportError("reading real images needs pillow: pip install pillow") from exc

    images, labels = [], []
    for index, name in enumerate(ORDINAL_CLASSES):
        folder = root / name
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*")):
            if path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            img = Image.open(path).convert("L").resize((size, size))
            images.append(np.asarray(img, dtype=np.float32) / 255.0)
            labels.append(index)
    if not images:
        raise FileNotFoundError(f"found {root} but no images inside it")
    print(f"[tread_images] {len(images)} real photographs, ordinal labels only "
          f"(no measured tread depth exists publicly)")
    return {
        "images": torch.as_tensor(np.stack(images)).unsqueeze(1),
        "label": torch.as_tensor(labels, dtype=torch.long),
        "label_type": "ordinal",
    }
