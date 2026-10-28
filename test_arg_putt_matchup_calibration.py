from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from archive.analyze_arg_putt_matchup_calibration import (
    american_implied,
    conditional_probability,
    holm_adjust,
    orient_pairs,
    weighted_ema,
)


def test_american_implied_and_conditional_no_vig() -> None:
    odds = pd.Series([-110.0, 120.0])
    implied = american_implied(odds)

    assert implied[0] == pytest.approx(110.0 / 210.0)
    assert implied[1] == pytest.approx(100.0 / 220.0)
    assert conditional_probability(pd.Series([0.45]), pd.Series([0.50]))[0] == pytest.approx(
        0.45 / 0.95
    )


def test_weighted_ema_uses_recent_values_more_heavily() -> None:
    rising = weighted_ema(np.array([-1.0, 0.0, 1.0]), span=3)
    falling = weighted_ema(np.array([1.0, 0.0, -1.0]), span=3)

    assert rising > 0.0
    assert falling < 0.0
    assert rising == pytest.approx(-falling)


def test_holm_adjustment_is_monotone_in_rank_order() -> None:
    adjusted = holm_adjust(pd.Series([0.01, 0.04, 0.03]))

    assert adjusted.tolist() == pytest.approx([0.03, 0.06, 0.06])


def test_pair_orientation_selects_the_model_over_market_side() -> None:
    pair = pd.DataFrame(
        {
            "event_key": ["pga|2026|1|2026-01-01"],
            "market": ["72h"],
            "pair_key": ["1|2"],
            "model_p1": [0.55],
            "market_p1_close": [0.60],
            "market_p1_open": [0.60],
            "outcome_p1": [0.0],
            "p1_short_dev_z": [1.0],
            "p2_short_dev_z": [-1.0],
            "p1_putt_dev_z": [0.5],
            "p2_putt_dev_z": [-0.5],
            "p1_arg_dev_z": [0.5],
            "p2_arg_dev_z": [-0.5],
            "p1_ott_dev_z": [-0.5],
            "p2_ott_dev_z": [0.5],
            "p1_app_dev_z": [-0.5],
            "p2_app_dev_z": [0.5],
            "p1_category_total_z": [0.0],
            "p2_category_total_z": [0.0],
            "p1_short_level_z": [1.0],
            "p2_short_level_z": [-1.0],
            "p1_sg_putt_z": [1.0],
            "p2_sg_putt_z": [-1.0],
            "p1_sg_arg_z": [1.0],
            "p2_sg_arg_z": [-1.0],
            "p1_sg_ott_z": [-1.0],
            "p2_sg_ott_z": [1.0],
            "p1_sg_app_z": [-1.0],
            "p2_sg_app_z": [1.0],
        }
    )

    oriented = orient_pairs(pair, "close").iloc[0]

    assert oriented["edge_side"] == "p2"
    assert oriented["model_probability"] == pytest.approx(0.45)
    assert oriented["market_probability"] == pytest.approx(0.40)
    assert oriented["outcome"] == pytest.approx(1.0)
    assert oriented["profile_short_diff"] == pytest.approx(-2.0)

