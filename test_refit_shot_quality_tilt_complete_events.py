from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ARCHIVE = Path(__file__).resolve().parent / "archive"
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

import refit_shot_quality_tilt_complete_events as refit


def _panel() -> pd.DataFrame:
    rows = []
    betas = [-0.1, 0.0, 0.1]
    for season, event, shots in (
        (2020, "a", 1000),
        (2021, "b", 1000),
        (2021, "broken", 4),
        (2022, "c", 1000),
    ):
        for beta in betas:
            delta = 0.0 if beta == 0 else (-beta if event != "broken" else 100 * beta)
            rows.append(
                {
                    "scenario": "all_valid",
                    "donor_normalization": "event_normalized",
                    "category": "ott",
                    "event_key": event,
                    "event_name": event,
                    "season": season,
                    "beta": beta,
                    "shots": shots,
                    "logloss_delta_vs_beta0": delta,
                    "crps_delta_vs_beta0": delta / 10,
                }
            )
    return pd.DataFrame(rows)


def test_gate_removes_incomplete_panel_before_selection() -> None:
    panel = _panel()
    betas = refit._validate_panel(panel)
    filtered, excluded = refit.apply_completeness_gate(panel, 500)
    assert set(excluded["event_key"]) == {"broken"}
    assert "broken" not in set(filtered["event_key"])
    folds, selected = refit.rolling_selection(
        filtered,
        betas,
        start_season=2020,
        end_season=2022,
        minimum_tune_events=2,
        beta_prior_events=0,
    )
    assert len(folds) == 1
    assert folds.iloc[0]["selected_beta"] == pytest.approx(0.1)
    assert set(selected["event_key"]) == {"c"}


def test_validation_requires_exact_beta_zero_parity() -> None:
    panel = _panel()
    panel.loc[np.isclose(panel["beta"], 0.0), "logloss_delta_vs_beta0"] = 1e-9
    with pytest.raises(ValueError, match="beta-zero log-score parity"):
        refit._validate_panel(panel)


def test_shot_count_must_be_constant_across_beta() -> None:
    panel = _panel()
    panel.loc[0, "shots"] += 1
    with pytest.raises(ValueError, match="shot count varies"):
        refit._validate_panel(panel)


def test_validation_requires_one_exact_beta_grid() -> None:
    panel = _panel()
    mask = panel["event_key"].eq("a") & np.isclose(panel["beta"], -0.1)
    panel.loc[mask, "beta"] = -0.2
    with pytest.raises(ValueError, match="one exact beta grid"):
        refit._validate_panel(panel)
