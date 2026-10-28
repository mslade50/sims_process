from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from archive import analyze_base_skill_discovery_search_correction as search


def _logical_fixture() -> pd.DataFrame:
    rows = []
    for event_index, outcome in enumerate((1.0, 0.0)):
        event = f"event-{event_index}"
        date = pd.Timestamp("2021-01-07") + pd.Timedelta(days=7 * event_index)
        for market in search.MARKETS:
            for candidate, probability in (
                (search.BASELINE, 0.5),
                ("good", 0.8 if outcome == 1.0 else 0.2),
                ("bad", 0.2 if outcome == 1.0 else 0.8),
            ):
                rows.append(
                    {
                        "candidate": candidate,
                        "mapped_event_key": event,
                        "event_start_date": date,
                        "market": market,
                        "pair_key": "1|2",
                        "canonical_outcome": outcome,
                        "canonical_probability": probability,
                        "canonical_push_probability": 0.0,
                    }
                )
    return pd.DataFrame(rows)


def test_event_loss_matrix_is_event_and_market_equal() -> None:
    event_market, pooled = search.build_event_loss_matrix(
        _logical_fixture(), ["good", "bad"]
    )
    assert len(event_market) == 8
    assert len(pooled) == 4
    means = pooled.groupby("candidate")["loss_delta"].mean()
    expected_good = -np.log(0.8) + np.log(0.5)
    expected_bad = -np.log(0.2) + np.log(0.5)
    assert np.isclose(means["good"], expected_good)
    assert np.isclose(means["bad"], expected_bad)


def test_search_correction_is_deterministic_and_familywise_fail_closed() -> None:
    rows = []
    for event_index in range(20):
        for candidate, delta in (("strong", -0.1), ("null", 0.0)):
            rows.append(
                {
                    "candidate": candidate,
                    "mapped_event_key": f"event-{event_index:02d}",
                    "event_start_date": pd.Timestamp("2021-01-01")
                    + pd.Timedelta(days=event_index),
                    "loss_delta": delta,
                }
            )
    pooled = pd.DataFrame(rows)
    first, first_audit = search.search_corrected_inference(
        pooled, draws=250, seed=7, block_lengths=(1, 2, 4)
    )
    second, second_audit = search.search_corrected_inference(
        pooled, draws=250, seed=7, block_lengths=(1, 2, 4)
    )
    pd.testing.assert_frame_equal(first, second)
    assert first_audit == second_audit
    indexed = first.set_index("candidate")
    assert indexed.loc["strong", "familywise_improvement_certified"]
    assert not indexed.loc["null", "familywise_improvement_certified"]
    assert first_audit["best_candidate"] == "strong"
    assert (
        first_audit["white_reality_check_null_centered_one_sided_p"] == 1 / 251
    )
    assert "iid_null_centered_one_sided_p_improvement" in first.columns
    for block in first_audit["moving_event_block_sensitivity"].values():
        assert "white_reality_check_null_centered_one_sided_p" in block
        assert "simultaneous_95pct_upper_increment" in block
    assert first_audit["calendar_week_clusters"] == 4
    assert first_audit["same_calendar_week_events_resampled_as_indivisible_clusters"]
    for block in first_audit["calendar_week_block_sensitivity"].values():
        assert block["status"] == "complete"
        assert "white_reality_check_null_centered_one_sided_p" in block
        assert "simultaneous_95pct_upper_increment" in block


def test_event_loss_matrix_rejects_challenger_only_canonical_key() -> None:
    logical = _logical_fixture()
    extra = logical.loc[logical["candidate"].eq("good")].iloc[[0]].copy()
    extra["mapped_event_key"] = "challenger-only-event"
    extra["event_start_date"] = pd.Timestamp("2021-03-01")
    logical = pd.concat([logical, extra], ignore_index=True)
    with pytest.raises(ValueError, match="challenger_only=1"):
        search.build_event_loss_matrix(logical, ["good", "bad"])


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("canonical_probability", np.nan),
        ("canonical_probability", np.inf),
        ("canonical_push_probability", np.nan),
        ("canonical_outcome", np.nan),
    ],
)
def test_event_loss_matrix_rejects_nonfinite_selected_values(
    column: str, value: float
) -> None:
    logical = _logical_fixture()
    logical.loc[logical["candidate"].eq("good").idxmax(), column] = value
    with pytest.raises(ValueError, match="non-finite"):
        search.build_event_loss_matrix(logical, ["good", "bad"])


def test_event_loss_matrix_rejects_candidate_outcome_disagreement() -> None:
    logical = _logical_fixture()
    row = logical["candidate"].eq("good").idxmax()
    logical.loc[row, "canonical_outcome"] = 1.0 - logical.loc[row, "canonical_outcome"]
    with pytest.raises(ValueError, match="outcomes disagree"):
        search.build_event_loss_matrix(logical, ["good", "bad"])


def _pooled_fixture() -> pd.DataFrame:
    rows = []
    for event_index in range(4):
        for candidate, delta in (("a", -0.02), ("b", 0.01)):
            rows.append(
                {
                    "candidate": candidate,
                    "mapped_event_key": f"event-{event_index}",
                    "event_start_date": pd.Timestamp("2021-02-01")
                    + pd.Timedelta(days=7 * event_index),
                    "loss_delta": delta,
                }
            )
    return pd.DataFrame(rows)


def test_search_correction_rejects_challenger_only_event_key() -> None:
    pooled = _pooled_fixture()
    pooled = pd.concat(
        [
            pooled,
            pd.DataFrame(
                [
                    {
                        "candidate": "a",
                        "mapped_event_key": "a-only-event",
                        "event_start_date": pd.Timestamp("2021-04-01"),
                        "loss_delta": -0.5,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="not exactly rectangular"):
        search.search_corrected_inference(pooled, draws=20, block_lengths=(1,))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_search_correction_rejects_nonfinite_event_delta(bad_value: float) -> None:
    pooled = _pooled_fixture()
    pooled.loc[0, "loss_delta"] = bad_value
    with pytest.raises(ValueError, match="must all be finite"):
        search.search_corrected_inference(pooled, draws=20, block_lengths=(1,))


def test_search_correction_rejects_conflicting_event_dates() -> None:
    pooled = _pooled_fixture()
    pooled.loc[
        pooled["candidate"].eq("b") & pooled["mapped_event_key"].eq("event-0"),
        "event_start_date",
    ] = pd.Timestamp("2021-02-02")
    with pytest.raises(ValueError, match="multiple event dates"):
        search.search_corrected_inference(pooled, draws=20, block_lengths=(1,))


def test_null_centered_tests_and_upper_band_use_correct_tail_and_sign() -> None:
    observed = np.array([-0.2, -0.1])
    centered = np.array(
        [
            [-2.0, 0.1],
            [-1.0, 0.2],
            [0.2, 0.3],
            [0.3, 0.4],
        ]
    )
    stats = search._null_centered_search_statistics(centered + observed, observed)
    np.testing.assert_allclose(
        stats["individual_null_centered_one_sided_p"], [3 / 5, 1 / 5]
    )
    assert stats["white_reality_check_null_centered_one_sided_p"] == 3 / 5
    # q95(max(-centered)) is 2.0; q95(max(centered)) would be only 0.4.
    assert stats["simultaneous_upper_increment"] == 2.0
    np.testing.assert_allclose(stats["simultaneous_upper_bounds"], [1.8, 1.9])
    assert not np.any(stats["familywise_improvement_certified"])


def test_calendar_week_cluster_bootstrap_never_splits_same_week_events() -> None:
    matrix = np.array([[0.0], [2.0], [10.0]])
    labels = ["2021-01-04", "2021-01-04", "2021-01-11"]
    samples = search._cluster_block_bootstrap_means(
        matrix,
        labels,
        500,
        np.random.default_rng(19),
        block_length=1,
    )
    # Two cluster draws can only select week1 twice (mean 1), one of each
    # complete week (mean 4), or week2 twice (mean 10).
    assert set(np.round(samples[:, 0], 12)).issubset({1.0, 4.0, 10.0})
    assert {1.0, 4.0, 10.0}.issubset(set(np.round(samples[:, 0], 12)))
