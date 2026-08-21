"""The dataset registry and the benchmark harness.

The registry is the single source of truth for datasets, so the things most worth
testing are the ones that would silently mislead: a source claiming to be verified
without a URL, a loader path that no longer resolves, or a fetch that quietly
substitutes something else when it cannot get what was asked for.
"""

import pandas as pd
import pytest
import torch

from tire_nn.benchmark import DEFAULT_MODELS, compare
from tire_nn.data import registry


# --- registry integrity ----------------------------------------------------

def test_every_entry_has_the_fields_a_user_needs():
    for key, entry in registry.DATASETS.items():
        assert entry.key == key
        assert entry.title and entry.provides and entry.licence and entry.used_by
        assert entry.kind in ("real", "simulated", "game", "synthetic")


def test_verified_real_datasets_have_a_url():
    """A source cannot be called verified without somewhere to get it."""
    for entry in registry.DATASETS.values():
        if entry.kind != "synthetic" and entry.verified:
            assert entry.url, f"{entry.key} is marked verified but has no URL"


def test_unverified_entries_are_flagged_rather_than_guessed():
    unverified = [e for e in registry.DATASETS.values() if not e.verified]
    assert unverified, "the registry should still record which sources are unconfirmed"
    for entry in unverified:
        assert entry.url is None, (
            f"{entry.key} is unverified but carries a URL — either confirm it or drop it")
        assert entry.steps, f"{entry.key} gives no instructions for confirming it"


def test_every_loader_path_resolves():
    for entry in registry.DATASETS.values():
        if entry.loader:
            assert callable(registry._resolve(entry.loader)), entry.key


def test_non_automatic_datasets_explain_themselves():
    for entry in registry.DATASETS.values():
        if not entry.auto:
            assert entry.steps, f"{entry.key} is manual but has no steps"


def test_synthetic_datasets_are_all_reachable():
    for key in registry.available(kind="synthetic"):
        assert registry.DATASETS[key].auto


def test_available_filters():
    assert "kit" in registry.available(real_only=True)
    assert "kit" not in registry.available(kind="synthetic")
    assert set(registry.available()) == set(registry.DATASETS)


# --- fetching --------------------------------------------------------------

def test_get_returns_synthetic_data_directly():
    df = registry.get("synthetic_force", n=120, seed=0)
    assert len(df) == 120
    assert {"alpha", "kappa", "Fz", "Fy"} <= set(df.columns)


def test_fetch_refuses_manual_datasets_with_actionable_steps(tmp_path):
    with pytest.raises(NotImplementedError) as exc:
        registry.fetch("kit", root=tmp_path)
    message = str(exc.value)
    assert "radar.kit.edu" in message
    assert "1." in message, "the error should list the numbered steps"


def test_get_never_silently_substitutes_another_dataset(tmp_path):
    """Missing data must raise, not quietly hand back something else."""
    with pytest.raises((NotImplementedError, FileNotFoundError)):
        registry.get("kit", root=tmp_path)


def test_markdown_table_covers_every_entry():
    table = registry.as_markdown()
    for key in registry.DATASETS:
        assert f"`{key}`" in table
    assert table.count("\n") >= len(registry.DATASETS)


def test_describe_runs_for_all_entries(capsys):
    registry.describe()
    for key in registry.DATASETS:
        registry.describe(key)
    assert "kit" in capsys.readouterr().out


# --- benchmark harness -----------------------------------------------------

def test_compare_returns_accuracy_and_violation_columns():
    data = registry.get("synthetic_force", n=400, seed=0)
    table = compare({"encoded": DEFAULT_MODELS["encoded"]}, data, epochs=3, verbose=False)
    assert isinstance(table, pd.DataFrame) and len(table) == 1
    for column in ("model", "n_params", "test_Fy_rmse",
                   "zero_slip_force", "sym_violation_y", "envelope_violation"):
        assert column in table.columns, f"{column} missing — a comparison must report it"


def test_compare_accepts_a_user_model():
    from tire_nn.models import EncodedTireNet

    data = registry.get("synthetic_force", n=300, seed=1)
    table = compare({"mine": lambda: EncodedTireNet(hidden=(8, 8))}, data,
                    epochs=3, verbose=False)
    assert table.iloc[0]["model"] == "mine"
    assert table.iloc[0]["n_params"] > 0


def test_compare_uses_a_fresh_model_per_run():
    """Factories, not instances: two runs must not share trained weights."""
    from tire_nn.models import EncodedTireNet

    data = registry.get("synthetic_force", n=300, seed=2)
    first = compare({"a": lambda: EncodedTireNet(hidden=(8, 8))}, data, epochs=3, verbose=False)
    second = compare({"a": lambda: EncodedTireNet(hidden=(8, 8))}, data, epochs=3, verbose=False)
    assert abs(float(first.iloc[0]["test_Fy_rmse"]) - float(second.iloc[0]["test_Fy_rmse"])) < 1e-6


@pytest.mark.parametrize("holdout", ["slip", "load", "none"])
def test_all_extrapolation_modes_run(holdout):
    data = registry.get("synthetic_force", n=400, seed=0)
    table = compare({"encoded": DEFAULT_MODELS["encoded"]}, data, epochs=2,
                    extrapolation=holdout, verbose=False)
    assert "extrap_Fy_rmse" in table.columns


def test_analytical_baseline_needs_no_training():
    data = registry.get("synthetic_force", n=300, seed=0)
    table = compare({"magic_formula": DEFAULT_MODELS["magic_formula"]}, data,
                    epochs=2, verbose=False)
    assert int(table.iloc[0]["n_params"]) == 0
    assert float(table.iloc[0]["envelope_violation"]) == 0.0
