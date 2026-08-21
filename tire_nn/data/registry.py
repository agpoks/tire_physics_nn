"""One source of truth for every dataset this project can use.

The registry exists so that three things cannot drift apart: what the documentation
says a dataset is, what the download helper fetches, and what the loader expects. The
table in ``docs/source/guides/datasets.md`` is generated from this file.

Each entry records what you need in order to decide whether to use it:

``kind``
    ``real`` measurement, ``simulated``, ``game`` telemetry or ``synthetic``. This label
    travels with the data into any results table.
``auto``
    whether ``fetch()`` can get it without human involvement. Most cannot: they sit
    behind a licence click, a Kaggle account, or an author request.
``url``
    where to get it. ``None`` means no primary source has been confirmed — those
    entries are explicitly unverified rather than guessed.
``size``
    approximate download size, so nothing large surprises you.

Usage::

    from tire_nn.data import registry
    registry.describe()                 # print the whole table
    registry.describe("kit")            # one entry, with the exact steps
    df = registry.get("f1_stints")      # fetch if possible, then load
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Dataset", "DATASETS", "describe", "fetch", "get", "available", "as_markdown"]


@dataclass
class Dataset:
    key: str
    title: str
    kind: str                       # real | simulated | game | synthetic
    provides: str                   # what quantities it actually contains
    url: str | None
    licence: str
    size: str
    auto: bool                      # can fetch() get it unattended?
    used_by: str                    # which experiment
    loader: str | None = None       # dotted path to the loader function
    subdir: str | None = None       # where it lives under data/raw
    steps: tuple[str, ...] = field(default_factory=tuple)
    verified: bool = True           # has the primary source been confirmed?
    note: str = ""


DATASETS: dict[str, Dataset] = {
    # ---------------------------------------------------------------- real, tire-level
    "kit": Dataset(
        key="kit",
        title="KIT inner-drum tire force transmission (dry asphalt)",
        kind="real",
        provides="Fx, Fy vs slip and load; TYDEX C/H/W frames; plus a simulated slalom cycle",
        url="https://radar.kit.edu/radar/en/dataset/p0rr2jc5wmf0drf8",
        licence="CC BY-NC-SA 4.0 (non-commercial)",
        size="~hundreds of MB",
        auto=False,
        used_by="Experiment 1",
        loader="tire_nn.data.kit.load_kit",
        subdir="kit",
        steps=(
            "Open the RADAR4KIT page above.",
            "Accept the CC BY-NC-SA 4.0 licence and download the archive.",
            "Extract into data/raw/kit/ , keeping the folder structure.",
            "The slalom folder is SIMULATION, not measurement — the adapter excludes it "
            "by default.",
        ),
        note="The most directly useful real dataset for this project: it is exactly the "
             "steady-state force data Experiment 1 is built around.",
    ),
    "vetyt": Dataset(
        key="vetyt",
        title="VeTyT bicycle tyre measurements (Politecnico di Milano)",
        kind="real",
        provides="Fy, Mz vs slip; Fz 343–526 N, camber ±5°, pressure 300–500 kPa",
        url="https://doi.org/10.1080/00423114.2024.2338143",
        licence="with the publication; request from the authors",
        size="small",
        auto=False,
        used_by="Experiment 1 (context / camber / pressure)",
        loader="tire_nn.data.vetyt.load_vetyt",
        subdir="vetyt",
        steps=(
            "Read the paper (Dell'Orto et al., Vehicle System Dynamics 2024) and the "
            "test-rig paper (Measurement 2022, doi:10.1016/j.measurement.2022.111813).",
            "Request the measurement tables from the authors, or extract them from the "
            "supplementary material.",
            "Place them under data/raw/vetyt/ , one file per tyre if possible.",
            "Check the sign convention on one pure-lateral sweep before trusting a run.",
        ),
    ),
    "tum_cargo_bike": Dataset(
        key="tum_cargo_bike",
        title="TUM cargo bicycle tire characteristics",
        kind="real",
        provides="longitudinal and lateral characteristics",
        url=None,
        licence="unknown",
        size="unknown",
        auto=False,
        used_by="Experiment 1",
        loader="tire_nn.data.tum_cargo_bike.load_tum_cargo_bike",
        subdir="tum_cargo_bike",
        verified=False,
        steps=(
            "No primary source confirmed. Locate the actual release (mediaTUM, TUM "
            "library, or the publication's supplementary material).",
            "Record the URL and licence here and in papers/references.bib.",
            "Consider the verified VeTyT cargo-bicycle measurements instead.",
        ),
        note="Do not report any quantitative result from this source until the release "
             "is confirmed.",
    ),
    "qmotion": Dataset(
        key="qmotion",
        title="Q-Motion tire data with inflation-pressure variation",
        kind="real",
        provides="forces vs slip across inflation pressures",
        url=None,
        licence="unknown",
        size="unknown",
        auto=False,
        used_by="Experiment 1 (pressure generalisation)",
        loader="tire_nn.data.qmotion.load_qmotion",
        subdir="qmotion",
        verified=False,
        steps=(
            "No primary source confirmed; locate the release and record it here.",
            "CHECK THE PRESSURE UNIT and pass it explicitly: "
            'load_qmotion(..., pressure_unit="kPa"|"bar"|"psi"|"Pa"). A wrong unit is '
            "silent — the model simply learns nothing from that channel.",
        ),
    ),
    # ------------------------------------------------------------- vehicle-level
    "deep_dynamics": Dataset(
        key="deep_dynamics",
        title="Deep Dynamics: BayesRace simulation + Indy Autonomous Challenge logs",
        kind="real",              # the IAC half; BayesRace is simulated, kept separate
        provides="vehicle-level states and inputs (no measured tire forces)",
        url="https://github.com/linklab-uva/deep-dynamics",
        licence="see repository",
        size="~tens of MB",
        auto=False,
        used_by="Experiment 3",
        loader="tire_nn.data.deep_dynamics.load_deep_dynamics",
        subdir="deep_dynamics",
        steps=(
            "git clone https://github.com/linklab-uva/deep-dynamics",
            "Copy the dataset files into data/raw/deep_dynamics/bayesrace/ and "
            "data/raw/deep_dynamics/iac/ respectively.",
            "Keep the two subsets separate: BayesRace is SIMULATED, IAC is REAL, and "
            "they must never be pooled into one results row.",
            "Cite Chrosniak, Ning & Behl, IEEE RA-L 2024 (arXiv:2312.04374).",
        ),
    ),
    "roboracer": Dataset(
        key="roboracer",
        title="RoboRacer / F1TENTH model-structured NN dataset",
        kind="real",
        provides="vehicle-level logs with tire-set and mass-change experiments",
        url=None,
        licence="unknown",
        size="unknown",
        auto=False,
        used_by="Experiments 2 and 3",
        loader="tire_nn.data.roboracer.load_roboracer",
        subdir="roboracer",
        verified=False,
        steps=(
            "No primary source confirmed; locate the release and record it here.",
            "Lay it out as data/raw/roboracer/<experiment_name>/*.csv — one directory "
            "per experiment, since the directory name becomes tire_id and is what makes "
            "the tire-set comparison possible.",
            "Note the wheel radius and vehicle geometry: Experiment 3 needs exact values.",
        ),
    ),
    "f1_stints": Dataset(
        key="f1_stints",
        title="Formula 1 stint data (lap time, compound, tyre age, weather)",
        kind="real",
        provides="lap time, compound, tyre age, track/air temperature, fuel fraction",
        url="https://github.com/theOehrly/Fast-F1",
        licence="FastF1 is MIT; the underlying timing data is subject to F1's terms",
        size="~50 MB for 8 races, cached locally",
        auto=True,
        used_by="Experiment 5",
        loader="tire_nn.data.lap_degradation.load_fastf1_stints",
        subdir="f1_stints",
        steps=(
            "python -m pip install fastf1",
            "python scripts/download_f1_stints.py --seasons 2023 --rounds 1 2 3 4 5 9 13 17",
        ),
        note="The only dataset here that needs no manual step: no API key, no licence "
             "click. Start with this one if you want real data quickly.",
    ),
    # -------------------------------------------------------------------- imagery
    "tyre_condition_images": Dataset(
        key="tyre_condition_images",
        title="Tyre condition photographs (ordinal: new / serviceable / unusable)",
        kind="real",
        provides="photographs with ORDINAL wear labels — no measured tread depth",
        url="https://www.kaggle.com/datasets/sameersambhare1/tyre-condition-classification-dataset",
        licence="see the Kaggle page",
        size="~100 MB",
        auto=False,
        used_by="Notebook 6 (imaging)",
        loader="tire_nn.data.tread_images.load_condition_images",
        subdir="tyre_condition",
        steps=(
            "Needs a Kaggle account. Download and extract so that "
            "data/raw/tyre_condition/<class>/*.jpg exists, with class folders named "
            "new, serviceable, unusable.",
            "Alternative: Mendeley TyreNet, https://data.mendeley.com/datasets/32b5vfj6tc/1 "
            "— no credentials needed, but a single 1.9 GB .rar, and its labels are "
            "defective/good rather than ordinal wear.",
        ),
        note="No public dataset of measured tread depth or of graining exists. These "
             "ordinal classes are a genuine wear ordering, which is what the monotone "
             "wear index in TreadConditionNet is built to consume.",
    ),
    "fsae_ttc": Dataset(
        key="fsae_ttc",
        title="FSAE Tire Test Consortium / Calspan tire force and moment data",
        kind="real",
        provides="Fx, Fy, Fz, Mz vs slip angle, slip ratio, camber, load, pressure, "
                 "speed and tire temperature; 430+ tests over 40+ constructions",
        url="https://www.fsaettc.org/",
        licence="consortium members only; redistribution not permitted",
        size="GBs across all rounds; a single round is manageable",
        auto=False,
        used_by="Experiment 1 (the best available fit)",
        loader="tire_nn.data.fsae_ttc.load_fsae_ttc",
        subdir="fsae_ttc",
        steps=(
            "Register at https://www.fsaettc.org/ — a fee applies and the consortium is "
            "aimed at Formula SAE teams, so university affiliation is the usual route.",
            "Download the round data from the members' server "
            "(test facility: https://calspan.com/automotive/fsae-ttc).",
            "Export runs to CSV or convert the .mat files into data/raw/fsae_ttc/.",
            "CHECK THE SIGNS on one pure-lateral sweep: TTC often reports FZ negative "
            "(handled) and the Fy sign depends on the processing axis system (pass "
            "flip_signs=True if needed).",
        ),
        note="The single most valuable dataset for this project: it is exactly the "
             "steady-state force data Experiment 1 is built around, across many tire "
             "constructions, which is what the tire-id context embedding is for. "
             "Method paper: SAE 2006-01-3606.",
    ),
    "tudelft_bicycle_mf": Dataset(
        key="tudelft_bicycle_mf",
        title="TU Delft Magic Formula parameters — bicycle tyres",
        kind="real",
        provides="published Magic Formula coefficients vs load, pressure and camber "
                 "(PARAMETERS, not raw time series)",
        url="https://research.tudelft.nl/en/datasets/magic-formula-parameters-bicycle-tyres/",
        licence="see the TU Delft research portal entry",
        size="tiny",
        auto=False,
        used_by="validating ParameterTireNet",
        subdir="tudelft_bicycle_mf",
        steps=(
            "Download the parameter tables from the TU Delft research portal.",
            "These are fitted coefficients, not measurements: use them as a validation "
            "target (does ParameterTireNet recover comparable B, C, D, E?) or to "
            "generate synthetic data with realistic coefficients.",
        ),
        note="Companion to the VeTyT measurements (Dell'Orto et al., Vehicle System "
             "Dynamics 2024). A published coefficient set is a rare and useful thing: "
             "it lets the parameter network be checked against numbers someone else "
             "fitted, rather than only against its own training data.",
    ),
    "racecar": Dataset(
        key="racecar",
        title="RACECAR — high-speed autonomous racing (Indy Autonomous Challenge)",
        kind="real",
        provides="6.5 h over 27 sessions, six teams, speeds to 273 km/h; LiDAR, radar, "
                 "camera, IMU, GPS, vehicle states — no measured tire forces",
        url="https://github.com/linklab-uva/RACECAR_DATA",
        licence="open (also on the AWS Open Data Registry and Hugging Face)",
        size="large; download selected scenarios",
        auto=False,
        used_by="Experiment 3 (vehicle-supervised)",
        subdir="racecar",
        steps=(
            "Clone https://github.com/linklab-uva/RACECAR_DATA or pull selected "
            "scenarios from the AWS Open Data Registry / Hugging Face mirror.",
            "Data ships in ROS2 and nuScenes formats; extract the vehicle-state and "
            "wheel-speed channels into the vehicle schema (tire_nn/data/vehicle.py).",
            "Cite the RACECAR paper (arXiv:2306.03252).",
        ),
        note="Unusually valuable for vehicle-level tire identification because the cars "
             "actually operate near the friction limit — road-car datasets rarely do, "
             "and a vehicle that never approaches the limit carries no information "
             "about where it is.",
    ),
    "tartandrive": Dataset(
        key="tartandrive",
        title="TartanDrive 2.0 — off-road driving with proprioception",
        kind="real",
        provides="7 h to 15 m/s; camera, IMU, GPS, wheel encoders, LiDAR, changing "
                 "terrain — no measured tire forces",
        url="https://theairlab.org/TartanDrive2/",
        licence="see the project page",
        size="large",
        auto=False,
        used_by="surface / friction adaptation (condition model)",
        subdir="tartandrive",
        steps=(
            "Follow the download instructions at https://theairlab.org/TartanDrive2/ .",
            "Map the proprioceptive and IMU channels onto the vehicle schema.",
            "Cite TartanDrive 2.0 (arXiv:2402.01913).",
        ),
        note="The interesting property here is deliberate terrain change, which makes it "
             "a candidate for identifying a surface-dependent friction state rather than "
             "a single mu — the same second-channel argument as the tread imagery.",
    ),
    "road_surface_images": Dataset(
        key="road_surface_images",
        title="Road surface condition imagery (dry / wet / snow / ice / rough)",
        kind="real",
        provides="road-surface photographs with condition classes — no vehicle states",
        url=None,
        licence="unknown",
        size="unknown",
        auto=False,
        used_by="vision prior on friction (analogous to the tread-imagery channel)",
        verified=False,
        steps=(
            "No primary source confirmed for the RSCD / extreme-road-image sets; locate "
            "the release and record it here before use.",
            "The intended role is a second observation channel constraining mu, exactly "
            "as tread imagery constrains wear and graining — see the imaging chapter.",
        ),
    ),
    "simulator_ground_truth": Dataset(
        key="simulator_ground_truth",
        title="Simulator ground truth (F1TENTH / CARLA / scuderia_gymnasium)",
        kind="simulated",
        provides="full state including tire forces, slip, mu — labels no rig provides",
        url="https://github.com/f1tenth/f1tenth_gym",
        licence="see the individual simulator",
        size="generated on demand",
        auto=False,
        used_by="pretraining and ablation ground truth",
        steps=(
            "Pick a simulator that exposes per-wheel forces: f1tenth_gym, "
            "scuderia_gymnasium (this author's, with a full Pacejka/brush/Dugoff stack), "
            "CARLA or Isaac Lab.",
            "Log slip, load and the resulting tire forces together with the vehicle "
            "states — the point is having both, which no rig or vehicle log provides.",
            "Label every result SIMULATED and never pool it with measurements: a model "
            "that fits a simulator has learned that simulator's tire model.",
        ),
        note="A simulator is the only source that gives per-wheel tire forces AND "
             "vehicle states together, which makes it the right place to check that "
             "vehicle-level identification recovers what it should. Its tire model is a "
             "design artefact, though, so a model that fits it has only learned that "
             "simulator: results must be labelled SIMULATED and never pooled with "
             "measurements.",
    ),
    # ------------------------------------------------------------------ synthetic
    "synthetic_force": Dataset(
        key="synthetic_force",
        title="Synthetic steady-state force data (Magic Formula + noise)",
        kind="synthetic",
        provides="alpha, kappa, Fz, Fx, Fy, optional pressure context",
        url=None, licence="generated in-repo", size="none", auto=True,
        used_by="Experiment 1",
        loader="tire_nn.data.common.make_synthetic",
        note="Always available. Clean Gaussian noise on a dense slip grid — kinder than "
             "any rig, and structurally matched to ParameterTireNet.",
    ),
    "synthetic_transient": Dataset(
        key="synthetic_transient",
        title="Synthetic step tests with known relaxation lengths",
        kind="synthetic",
        provides="slip/force sequences with a known sigma", url=None,
        licence="generated in-repo", size="none", auto=True,
        used_by="Experiment 2",
        loader="tire_nn.data.common.make_synthetic_transient",
    ),
    "synthetic_vehicle": Dataset(
        key="synthetic_vehicle",
        title="Synthetic vehicle logs from a known tire",
        kind="synthetic",
        provides="IMU, yaw rate, wheel speeds, steering — no tire forces", url=None,
        licence="generated in-repo", size="none", auto=True,
        used_by="Experiment 3",
        loader="tire_nn.data.vehicle.make_synthetic_vehicle",
    ),
    "synthetic_stints": Dataset(
        key="synthetic_stints",
        title="Synthetic stints with known wear and graining",
        kind="synthetic",
        provides="lap times with latent wear/graining ground truth", url=None,
        licence="generated in-repo", size="none", auto=True,
        used_by="Experiment 5",
        loader="tire_nn.data.lap_degradation.make_synthetic_stints",
    ),
    "synthetic_images": Dataset(
        key="synthetic_images",
        title="Synthetic tread imagery with known wear and graining",
        kind="synthetic",
        provides="grayscale tread patches, continuous wear/graining truth", url=None,
        licence="generated in-repo", size="none", auto=True,
        used_by="Notebook 6",
        loader="tire_nn.data.tread_images.make_tread_dataset",
    ),
}


def _resolve(dotted: str):
    module_name, _, attribute = dotted.rpartition(".")
    import importlib

    return getattr(importlib.import_module(module_name), attribute)


def available(kind: str | None = None, real_only: bool = False) -> list[str]:
    """Dataset keys, optionally filtered by kind."""
    keys = list(DATASETS)
    if kind:
        keys = [k for k in keys if DATASETS[k].kind == kind]
    if real_only:
        keys = [k for k in keys if DATASETS[k].kind == "real"]
    return keys


def describe(key: str | None = None) -> None:
    """Print the registry, or one entry with its exact acquisition steps."""
    if key is None:
        print(f"{'key':24s} {'kind':10s} {'auto':5s} {'verified':9s} title")
        print("-" * 100)
        for entry in DATASETS.values():
            print(f"{entry.key:24s} {entry.kind:10s} "
                  f"{'yes' if entry.auto else 'no':5s} "
                  f"{'yes' if entry.verified else 'NO':9s} {entry.title}")
        print("\nregistry.describe('<key>') for the acquisition steps of one dataset.")
        return

    entry = DATASETS[key]
    print(f"{entry.title}\n{'=' * len(entry.title)}")
    print(f"key       : {entry.key}")
    print(f"type      : {entry.kind.upper()}"
          f"{'' if entry.verified else '   *** SOURCE NOT VERIFIED ***'}")
    print(f"provides  : {entry.provides}")
    print(f"url       : {entry.url or '(no confirmed primary source)'}")
    print(f"licence   : {entry.licence}")
    print(f"size      : {entry.size}")
    print(f"automatic : {'yes' if entry.auto else 'no — manual steps below'}")
    print(f"used by   : {entry.used_by}")
    if entry.subdir:
        print(f"expects   : data/raw/{entry.subdir}/")
    if entry.steps:
        print("\nsteps:")
        for i, step in enumerate(entry.steps, 1):
            print(f"  {i}. {step}")
    if entry.note:
        print(f"\nnote: {entry.note}")


def fetch(key: str, root: str | Path = "data/raw", **kwargs):
    """Acquire a dataset if that can be done unattended, else explain how.

    Deliberately conservative: it will not click a licence, use your Kaggle credentials
    or pull a multi-gigabyte archive on your behalf. When it cannot proceed it raises
    with the exact steps rather than failing obscurely or silently substituting
    something else.
    """
    entry = DATASETS[key]
    if entry.kind == "synthetic":
        return _resolve(entry.loader)(**kwargs)

    if entry.auto and key == "f1_stints":
        import subprocess
        import sys

        seasons = kwargs.pop("seasons", [2023])
        rounds = kwargs.pop("rounds", [1, 2, 3])
        command = [sys.executable, "scripts/download_f1_stints.py",
                   "--root", str(root),
                   "--seasons", *map(str, seasons),
                   "--rounds", *map(str, rounds)]
        print("$ " + " ".join(command))
        if subprocess.call(command) != 0:
            raise RuntimeError("the FastF1 download did not complete; see the output above")
        return _resolve(entry.loader)(root, **kwargs)

    raise NotImplementedError(
        f"{entry.title} cannot be fetched unattended.\n"
        + "\n".join(f"  {i}. {s}" for i, s in enumerate(entry.steps, 1))
        + (f"\n  url: {entry.url}" if entry.url else "")
        + f"\n\nThen: registry.get({key!r})"
    )


def get(key: str, root: str | Path = "data/raw", **kwargs):
    """Load a dataset, fetching it first if that is possible unattended."""
    entry = DATASETS[key]
    if entry.loader is None:
        raise KeyError(f"{key} has no loader")
    if entry.kind == "synthetic":
        return _resolve(entry.loader)(**kwargs)
    try:
        return _resolve(entry.loader)(root, **kwargs)
    except FileNotFoundError:
        return fetch(key, root, **kwargs)


def as_markdown() -> str:
    """The documentation table, generated so it cannot drift from the code."""
    lines = ["| dataset | type | provides | acquisition | used by |",
             "|---|---|---|---|---|"]
    for e in DATASETS.values():
        if e.url:
            acquisition = f"[link]({e.url})"
            acquisition += ", **automatic**" if e.auto else ", manual"
            acquisition += f" · {e.size}" if e.size not in ("none", "unknown") else ""
        elif e.kind == "synthetic":
            acquisition = "generated in-repo"
        else:
            acquisition = "**source unverified**"
        kind = e.kind if e.verified else f"{e.kind} ⚠"
        lines.append(f"| `{e.key}` — {e.title} | {kind} | {e.provides} | {acquisition} | {e.used_by} |")
    return "\n".join(lines)
