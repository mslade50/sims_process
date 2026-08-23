import numpy as np
import pytest

from round_sim import (
    _active_round_expected_scores,
    build_round_score_probs,
    build_score_card,
)
from score_reprice import (
    fractional_settlement_pmf,
    score_est_requires_live_refresh,
    strict_under_probability,
    uniformly_shift_score_tape,
)


BASE_DRAWS = np.array([69] * 25 + [70] * 50 + [71] * 25)


def _pmf(draws):
    scores, probabilities = fractional_settlement_pmf(draws)
    return dict(zip(scores.tolist(), probabilities.tolist()))


def test_plus_three_tenths_moves_boundary_mass_and_score_fair():
    shifted = BASE_DRAWS.astype(float) + 0.3

    assert _pmf(shifted) == pytest.approx(
        {69: 0.175, 70: 0.425, 71: 0.325, 72: 0.075}
    )
    assert strict_under_probability(shifted, 69.5) == pytest.approx(0.175)


def test_minus_three_tenths_moves_adjacent_mass_and_score_fair():
    shifted = BASE_DRAWS.astype(float) - 0.3

    assert _pmf(shifted) == pytest.approx(
        {68: 0.075, 69: 0.325, 70: 0.425, 71: 0.175}
    )
    assert strict_under_probability(shifted, 69.5) == pytest.approx(0.40)


@pytest.mark.parametrize("delta", [-2.0, -1.0, 0.0, 1.0, 2.0])
def test_whole_stroke_shift_has_exact_point_mass_parity(delta):
    shifted = BASE_DRAWS.astype(float) + delta
    expected = {
        int(score + delta): probability
        for score, probability in {69: 0.25, 70: 0.50, 71: 0.25}.items()
    }
    assert _pmf(shifted) == expected
    for line in (68.5, 69.5, 70.5, 71.5, 72.5):
        direct = float(np.mean(shifted < line))
        assert strict_under_probability(shifted, line) == direct


@pytest.mark.parametrize("delta", [-1.7, -0.3, 0.0, 0.3, 2.4])
def test_fractional_pmf_preserves_probability_mass_and_shifted_mean(delta):
    shifted = np.array([67, 68, 68, 70, 73], dtype=float) + delta
    scores, probabilities = fractional_settlement_pmf(shifted)

    assert probabilities.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.dot(scores, probabilities) == pytest.approx(shifted.mean(), abs=1e-12)
    assert np.all(scores == scores.astype(int))


def test_published_pmf_uses_fractional_mass_and_active_expected_average():
    shifted = BASE_DRAWS.astype(float) + 0.3
    probs = build_round_score_probs({"player": shifted}, 70.3)

    actual = dict(zip(probs["score"], probs["prob"]))
    assert actual == pytest.approx({69: 0.175, 70: 0.425, 71: 0.325, 72: 0.075})
    assert probs["expected_avg"].unique().tolist() == [70.3]


def test_score_card_fair_price_moves_for_both_decimal_directions():
    pred = {"player": 1.0}
    base = build_score_card({"player": BASE_DRAWS}, 70.0, pred).iloc[0]
    worse = build_score_card(
        {"player": BASE_DRAWS.astype(float) + 0.3}, 70.3, pred
    ).iloc[0]
    better = build_score_card(
        {"player": BASE_DRAWS.astype(float) - 0.3}, 69.7, pred
    ).iloc[0]

    # At under 69.5: 25% -> 17.5% when scoring worsens, 40% when it improves.
    assert worse["69.5"] > base["69.5"]
    assert better["69.5"] < base["69.5"]


def test_uniform_shift_preserves_every_joint_matchup_and_tie():
    tape = {
        "a": np.array([68, 70, 71, 69]),
        "b": np.array([69, 70, 70, 72]),
        "c": np.array([67, 71, 72, 72]),
    }
    shifted = uniformly_shift_score_tape(tape, 0.3)


    for left in tape:
        for right in tape:
            assert np.array_equal(tape[left] < tape[right], shifted[left] < shifted[right])
            assert np.array_equal(tape[left] == tape[right], shifted[left] == shifted[right])


def test_live_decimal_reprice_requires_remaining_tournament_refresh():
    assert score_est_requires_live_refresh(
        price_only=True, score_shift_delta=0.3, completed_round=3
    )
    assert not score_est_requires_live_refresh(
        price_only=True, score_shift_delta=0.0, completed_round=3
    )
    assert not score_est_requires_live_refresh(
        price_only=False, score_shift_delta=0.3, completed_round=3
    )
    assert not score_est_requires_live_refresh(
        price_only=True, score_shift_delta=0.3, completed_round=0
    )


def test_remaining_tournament_uses_active_decimal_course_baselines():
    import pandas as pd

    predictions = pd.DataFrame({
        "player_name": ["a", "b", "c"],
        "course_score_adj": [68.7, 69.2, 68.7],
    })
    expected = _active_round_expected_scores(
        ["c", "b", "a"],
        predictions,
        {"default_expected": 70.0},
    )
    assert expected.tolist() == [68.7, 69.2, 68.7]
