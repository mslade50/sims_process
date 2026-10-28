from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from archive.calibrate_shot_quality_normalization import (
    _distance_state,
    _fit_slope,
    expanding_folds,
    residualize_states,
)


def test_distance_states_use_category_specific_buckets() -> None:
    category = pd.Series(["ott", "app", "app", "app", "app", "arg", "putt"])
    distance = pd.Series([276.0, 99.0, 125.0, 190.0, 230.0, 32.0, 4.0])
    result = _distance_state(category, distance).tolist()
    assert result[:6] == [
        "275yd",
        "<=100",
        "100-150",
        "150-215",
        ">215",
        "30yd",
    ]
    assert result[6].startswith("logft_")


def _synthetic_panel() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(17)
    for season in range(2020, 2026):
        for event_no in range(2):
            for shot in range(800):
                # Vary ability within each round/hole state rather than tying
                # it to the round-number modulo.
                ability = (-1.0, -0.25, 0.25, 1.0)[(shot // 72) % 4]
                rows.append(
                    {
                        "event_key": f"e{season}_{event_no}",
                        "season": season,
                        "round_no": 1 + shot % 4,
                        "hole_no": 1 + shot % 18,
                        "sg_category": "app",
                        "start_lie_state": "fairway",
                        "distance_state": "150-215",
                        "sg_raw": 0.04 * ability + rng.normal(0, 0.25),
                        "ability_sg_round": ability,
                    }
                )
    return pd.DataFrame(rows)


def test_strict_state_residual_slope_and_expanding_folds() -> None:
    panel = residualize_states(_synthetic_panel())
    fit = _fit_slope(panel)
    assert fit["slope_sg_shot_per_sg_round"] == pytest.approx(0.04, abs=0.006)
    folds = expanding_folds(panel)
    assert folds["test_year"].tolist() == [2022, 2023, 2024, 2025]
    assert folds["test_mse_delta"].lt(0).all()
