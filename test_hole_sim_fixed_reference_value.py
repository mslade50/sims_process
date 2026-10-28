from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from archive import hole_sim_fixed_reference_value as reference


def _source_frame() -> pd.DataFrame:
    rows = []
    lies = ("tee", "fairway", "rough", "sand", "recovery")
    for index, lie in enumerate(lies):
        rows.append(
            {
                "start_lie": lie,
                "start_distance_yd": 100.0,
                "expected_strokes_start": 3.0 + index / 10.0,
                "end_lie": lie,
                "end_distance_yd": 200.0,
                "expected_strokes_end": 4.0 + index / 10.0,
                "sg_valid": 1,
            }
        )
    rows.append(
        {
            "start_lie": "hole",
            "start_distance_yd": 0.0,
            "expected_strokes_start": 0.0,
            "end_lie": "hole",
            "end_distance_yd": 0.0,
            "expected_strokes_end": 0.0,
            "sg_valid": 1,
        }
    )
    return pd.DataFrame(rows)


def _model() -> reference.ReferenceValueModel:
    non_green = reference.collect_non_green_knots([_source_frame()])
    green = pd.DataFrame(
        {
            "lie": ["green", "green"],
            "distance_yd": [0.0, 10.0],
            "expected_strokes": [1.0, 2.0],
            "source_kind": ["frozen_prior_putting_curve"] * 2,
        }
    )
    table = pd.concat([non_green, green], ignore_index=True)
    putting_sha = "a" * 64
    model_sha = reference.reference_model_sha256(table, putting_sha)
    return reference.ReferenceValueModel(table, putting_sha, model_sha)


def test_collect_non_green_knots_rejects_changed_reference() -> None:
    first = _source_frame()
    second = _source_frame()
    second.loc[second["start_lie"].eq("fairway"), "expected_strokes_start"] += 0.01
    with pytest.raises(reference.ReferenceValueUnavailable, match="Broadie reference changes"):
        reference.collect_non_green_knots([first, second])


def test_value_interpolation_terminal_and_broadie_style_extrapolation() -> None:
    model = _model()
    assert model.value("fairway", 150.0) == pytest.approx(3.6)
    assert model.value("green", 5.0) == pytest.approx(1.5)
    assert model.value("green", 100.0) == pytest.approx(2.0)
    assert model.value("hole", 0.0) == 0.0
    assert model.state_key("hole", 0.0) == "terminal"
    assert model.value("fairway", 250.0) == pytest.approx(4.6)
    assert model.value("fairway", -5.0) == pytest.approx(2.1)
    with pytest.raises(reference.ReferenceValueUnavailable, match="zero distance"):
        model.value("hole", 1.0)


def test_transition_rewards_telescope_with_linked_penalty() -> None:
    model = _model()
    first = model.transition_reward(
        start_lie="tee",
        start_distance_yd=150.0,
        end_lie="rough",
        end_distance_yd=150.0,
        stroke_cost=3,
    )
    second = model.transition_reward(
        start_lie="rough",
        start_distance_yd=150.0,
        end_lie="green",
        end_distance_yd=5.0,
        stroke_cost=1,
    )
    third = model.transition_reward(
        start_lie="green",
        start_distance_yd=5.0,
        end_lie="hole",
        end_distance_yd=0.0,
        stroke_cost=1,
    )
    assert first + second + third == pytest.approx(model.value("tee", 150.0) - 5.0)
    assert model.state_key("rough", 150.0) == "rough|150.000000yd"
