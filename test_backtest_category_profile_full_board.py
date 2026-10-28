from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from archive.backtest_category_profile_full_board import (
    ARMS,
    CAT_ORDER,
    KEY_COLS,
    calibrate_profiles,
    compare_discovery,
    dedupe_matchup_direction,
    event_field_shot_history,
    file_signature,
    invariant_round_probabilities,
    normalize_cache_board,
    normalize_round_authority,
    profile_eligibility,
    round_invariance_audit,
)


def test_file_signature_tracks_content_not_rewrite_time(tmp_path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"alpha")
    first = file_signature(artifact)
    artifact.write_bytes(b"bravo")
    second = file_signature(artifact)

    assert first["size"] == second["size"] == 5
    assert first["sha256"] != second["sha256"]
    assert "mtime_ns" not in first


def test_event_shot_history_excludes_future_rows_and_other_event_players() -> None:
    stats = pd.DataFrame(
        {
            "source_player_id": ["a", "a", "b", "c"],
            "event_date": pd.to_datetime(
                ["2026-01-01", "2026-02-01", "2026-01-01", "2026-01-01"]
            ),
            "value": [1, 2, 3, 4],
        }
    )
    field = pd.DataFrame({"source_player_id": ["a", "b"]})

    result = event_field_shot_history(stats, field, pd.Timestamp("2026-01-15"))

    assert result[["source_player_id", "value"]].to_dict("records") == [
        {"source_player_id": "a", "value": 1},
        {"source_player_id": "b", "value": 3},
    ]


def test_blanket_and_targeted_calibrations_preserve_total_and_use_common_cap() -> None:
    # CAT_ORDER is OTT, APP, ARG, PUTT in the production kernel.
    assert tuple(CAT_ORDER) == ("sg_ott", "sg_app", "sg_arg", "sg_putt")
    mu = np.array(
        [
            [1.2, 0.6, -0.8, -1.0],
            [0.4, 0.3, 0.2, 0.1],
            [1.0, -1.0, 0.5, -0.5],
        ]
    )
    eligible = np.array([True, True, False])

    blanket, blanket_audit = calibrate_profiles(mu, eligible, "blanket_065")
    targeted, targeted_audit = calibrate_profiles(mu, eligible, "targeted_arg_putt")

    assert blanket.sum(axis=1) == pytest.approx(mu.sum(axis=1))
    assert targeted.sum(axis=1) == pytest.approx(mu.sum(axis=1))
    assert np.max(np.abs(blanket - mu)) <= 0.15 + 1e-12
    assert np.max(np.abs(targeted - mu)) <= 0.15 + 1e-12
    assert blanket_audit["cap_bound_players"] == 1
    assert targeted_audit["cap_bound_players"] == 1
    assert blanket[2] == pytest.approx(mu[2])
    assert targeted[2] == pytest.approx(mu[2])

    # The uncapped second row has mean 0.25. ARG and PUTT move exactly 35%
    # toward it; OTT/APP receive equal zero-sum compensation.
    expected_shift = np.array([-0.035, -0.035, 0.0175, 0.0525])
    assert targeted[1] - mu[1] == pytest.approx(expected_shift)


def test_profile_eligibility_is_strict_prior_and_requires_age_rounds_and_each_category() -> None:
    field = pd.DataFrame({"dg_id": [1, 2, 3, 4]})
    cutoff = pd.Timestamp("2026-05-01")

    def history(count: int, missing_putt: int = 0) -> pd.DataFrame:
        dates = pd.date_range("2025-01-01", periods=count, freq="7D")
        frame = pd.DataFrame(
            {
                "round_date": dates,
                "sg_ott_adj": 1.0,
                "sg_app_adj": 1.0,
                "sg_arg_adj": 1.0,
                "sg_putt_adj": 1.0,
            }
        )
        if missing_putt:
            frame.loc[: missing_putt - 1, "sg_putt_adj"] = np.nan
        return frame

    histories = {
        1: history(20),
        2: history(19),
        3: history(20, missing_putt=9),
        4: history(20),
    }
    ages = {
        1: (np.array(["2026-01-01"], dtype="datetime64[ns]"), np.array([40.0])),
        2: (np.array(["2026-01-01"], dtype="datetime64[ns]"), np.array([40.0])),
        3: (np.array(["2026-01-01"], dtype="datetime64[ns]"), np.array([40.0])),
        4: (np.array(["2026-01-01"], dtype="datetime64[ns]"), np.array([55.0])),
    }
    result = profile_eligibility(field, histories, ages, cutoff)
    assert result["eligible"].tolist() == [True, False, False, False]
    assert result["eligibility_reason"].tolist() == [
        "eligible",
        "prior_rounds_lt20",
        "category_obs_lt12",
        "age_ge55",
    ]


def test_cache_normalization_supports_open_and_finish_markets() -> None:
    matchups = [
        {
            "season": 2026,
            "event_id": 9,
            "event_completed": "2026-03-08",
            "bet_type": "72-hole Match",
            "book": "Pinnacle",
            "open_time": "2026-03-03",
            "close_time": "2026-03-05",
            "tie_rule": "void",
            "p1_dg_id": 1,
            "p1_player_name": "One",
            "p1_open": 120,
            "p1_close": -105,
            "p1_outcome": 1,
            "p1_outcome_text": "win",
            "p2_dg_id": 2,
            "p2_player_name": "Two",
            "p2_open": -130,
            "p2_close": -115,
            "p2_outcome": 0,
            "p2_outcome_text": "loss",
        }
    ]
    outrights = [
        {
            "season": 2026,
            "event_id": 9,
            "event_completed": "2026-03-08",
            "market": "top20",
            "book": "BetOnline",
            "dg_id": 1,
            "player_name": "One",
            "open_odds": 250,
            "close_odds": 200,
            "open_time": "2026-03-03",
            "close_time": "2026-03-05",
            "bet_outcome_numeric": 0.5,
        }
    ]
    opened = normalize_cache_board(matchups, outrights, "open")
    closed = normalize_cache_board(matchups, outrights, "close")
    assert len(opened) == len(closed) == 3
    assert set(opened["selection_id"]) == {"1", "2"}
    assert set(opened["market"]) == {"tournament_matchup", "top_20"}
    assert opened.loc[opened["selection_id"].eq("1") & opened["market"].eq("tournament_matchup"), "close_odds"].item() == 120
    assert closed.loc[closed["selection_id"].eq("1") & closed["market"].eq("tournament_matchup"), "close_odds"].item() == -105


def test_round_authority_is_expanded_and_identical_across_all_arms() -> None:
    source = pd.DataFrame(
        [
            {
                "candidate": "baseline_current_prod",
                "market": "r2",
                "event_key": "pga|2026|9|2026-03-05",
                "event_start_date": "2026-03-05",
                "book": "betcris",
                "p1_id": 1,
                "p2_id": 2,
                "p1_open": -110,
                "p2_open": 100,
                "p1_close": -105,
                "p2_close": -105,
                "p1_probability": 0.48,
                "p2_probability": 0.47,
                "push_probability": 0.05,
                "settlement_p1_outcome": 1.0,
                "tie_rule": "void",
                "quote_id": "q1",
            }
        ]
    )
    lines, baseline = normalize_round_authority(source, "open")
    assert len(lines) == len(baseline) == 2
    assert set(lines["board"]) == {"r2"}
    assert set(lines["outcome_numeric"]) == {0.0, 1.0}
    all_arms = invariant_round_probabilities(baseline)
    assert set(all_arms["arm"]) == set(ARMS)
    audit = round_invariance_audit(all_arms)
    assert audit["structurally_invariant"].all()
    assert (audit["max_abs_probability_delta"] == 0.0).all()


def _bet(selection: str, opponent: str, stake: float = 1.0) -> dict[str, object]:
    return {
        "event_id": "pga:R2026009",
        "board": "r1",
        "market": "tournament_matchup",
        "selection_id": selection,
        "opponent_id": opponent,
        "stake": stake,
        "pnl": 0.5,
        "price_point": "open",
        "sizing_name": "flat_1u",
    }


def test_discovery_comparison_labels_overlap_new_dropped_and_side_flip_once() -> None:
    baseline = pd.DataFrame([_bet("1", "2"), _bet("3", "4"), _bet("5", "6")])
    candidate = pd.DataFrame([_bet("1", "2"), _bet("4", "3"), _bet("7", "8")])
    result = compare_discovery(baseline, candidate, "targeted_arg_putt")
    assert result["attribution"].value_counts().to_dict() == {
        "overlap": 1,
        "side_flip": 1,
        "dropped": 1,
        "new": 1,
    }
    flip = result[result["attribution"].eq("side_flip")].iloc[0]
    assert flip["baseline_selection_id"] == "3"
    assert flip["challenger_selection_id"] == "4"
    assert set(KEY_COLS).issubset(result.columns)


def test_matchup_selection_keeps_only_highest_ev_direction_per_unordered_pair() -> None:
    selected = pd.DataFrame(
        [
            {
                **_bet("1", "2"),
                "ev_per_unit": 0.04,
                "close_odds": 110,
                "book": "pinnacle",
            },
            {
                **_bet("2", "1"),
                "ev_per_unit": 0.06,
                "close_odds": -105,
                "book": "betcris",
            },
        ]
    )
    output = dedupe_matchup_direction(selected)
    assert len(output) == 1
    assert output.iloc[0]["selection_id"] == "2"
