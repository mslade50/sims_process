from __future__ import annotations

import numpy as np
import pandas as pd

from archive.analyze_course_profile_matchup_archetypes import (
    ARM_FULL,
    ARM_HALF,
    ARM_NONE,
    BootstrapWeights,
    add_full_reference_edge,
    attach_profile_attribution,
    bootstrap_table,
    build_driving_profiles,
    normalize_bets,
)


def _bet_row(arm: str, challenger: str, selection: int, pnl: float) -> dict:
    return {
        "event_id": "pga:R2025001",
        "event_start": "2025-01-02",
        "market": "tournament_matchup",
        "market_group": "matchup",
        "selection_id": selection,
        "selection_name": f"Player {selection}",
        "opponent_id": 99,
        "opponent_name": "Opponent",
        "arm": arm,
        "challenger_name": challenger,
        "line_policy_name": "best_of_sharp",
        "sizing_name": "production_style",
        "stake": 1.0,
        "pnl": pnl,
        "ev_per_unit": 0.05,
        "probability": 0.55,
        "push_probability": 0.0,
        "close_odds": -110,
        "settlement_status": "win" if pnl > 0 else "loss",
        "is_graded": True,
    }


def test_bet_normalization_maps_comparison_arms_and_dedupes_full() -> None:
    rows = [
        _bet_row("baseline", "profile_half", 1, 1.0),
        _bet_row("baseline", "profile_none", 1, 1.0),
        _bet_row("challenger", "profile_half", 2, -1.0),
        _bet_row("challenger", "profile_none", 3, 1.0),
    ]
    normalized, audit = normalize_bets(
        pd.DataFrame(rows), "best_of_sharp", "production_style"
    )

    assert normalized["arm"].value_counts().to_dict() == {
        ARM_FULL: 1,
        ARM_HALF: 1,
        ARM_NONE: 1,
    }
    assert audit["deduped"] == 1


def test_driving_profiles_use_strict_prior_rows_and_event_field_percentiles() -> None:
    roster = pd.DataFrame(
        {
            "event_id": ["event"] * 4,
            "event_start": pd.to_datetime(["2025-01-10"] * 4),
            "dg_id": ["1", "2", "3", "4"],
            "player_name": ["one", "two", "three", "four"],
        }
    )
    rows = []
    # Player 1 is shortest and most accurate; player 4 has only ten prior
    # rounds but still contributes his available values to percentile ranks.
    for dg_id, rounds, distance, accuracy in (
        ("1", 20, -10.0, 0.80),
        ("2", 20, 0.0, 0.60),
        ("3", 20, 10.0, 0.40),
        ("4", 10, 20.0, 0.20),
    ):
        for day in range(rounds):
            rows.append(
                {
                    "dg_id": dg_id,
                    "round_date": pd.Timestamp("2024-12-01") + pd.Timedelta(days=day),
                    "sg_ott_adj": 0.0,
                    "rel_dd": distance,
                    "driving_acc": accuracy,
                }
            )
        # This future row must not enter the profile.
        rows.append(
            {
                "dg_id": dg_id,
                "round_date": pd.Timestamp("2025-01-11"),
                "sg_ott_adj": 0.0,
                "rel_dd": -100.0,
                "driving_acc": 1.0,
            }
        )
    profiles = build_driving_profiles(roster, pd.DataFrame(rows))
    indexed = profiles.set_index("dg_id")

    assert indexed.loc["1", "strict_archetype"] == "short_accurate"
    assert indexed.loc["4", "strict_archetype"] == "unknown"
    assert indexed.loc["1", "rel_dd_rolling"] == -10.0
    assert indexed.loc["1", "distance_percentile"] == 25.0


def test_bootstrap_retains_zero_bet_events() -> None:
    frame = pd.DataFrame(
        {
            "event_id": ["e1", "e1", "e1"],
            "arm": [ARM_FULL, ARM_HALF, ARM_NONE],
            "stake": [1.0, 1.0, 1.0],
            "pnl": [1.0, 0.0, -1.0],
            "ev_per_unit": [0.05, 0.05, 0.05],
        }
    )
    result = bootstrap_table(
        frame,
        "overall",
        None,
        ["all"],
        ["e1", "e2", "e3"],
        BootstrapWeights(1_000, 7),
    )

    assert set(result["event_clusters"]) == {3}
    none_full = result[
        result["comparison"].eq("profile_none_minus_profile_full")
        & result["metric"].eq("pnl_delta")
    ].iloc[0]
    assert none_full["estimate"] == -2.0
    assert np.isfinite(none_full["ci_low"])


def test_profile_attribution_gap_uses_selection_minus_opponent() -> None:
    bets = pd.DataFrame(
        {
            "event_id": ["e1", "e1"],
            "selection_id": ["1", "2"],
            "opponent_id": ["2", "1"],
        }
    )
    attribution = pd.DataFrame(
        {
            "event_id": ["e1", "e1"],
            "dg_id": ["1", "2"],
            "profile_in_baseline": [0.06, -0.01],
        }
    )
    result = attach_profile_attribution(bets, attribution)

    assert np.allclose(result["profile_in_baseline_gap"], [0.07, -0.07])
    assert result["profile_in_baseline_gap_bucket"].astype(str).tolist() == [
        "selection_favored_ge_0p05_sg",
        "selection_disfavored_ge_0p05_sg",
    ]


def test_full_reference_ev_matches_sealed_push_aware_scorer() -> None:
    bets = pd.DataFrame(
        {
            "event_id": ["e1", "e1"],
            "selection_id": ["1", "2"],
            "opponent_id": ["2", "1"],
            "close_odds": [-110.0, 120.0],
            "tie_rule": ["Bets are void in event of tie", "ties lose"],
        }
    )
    probabilities = pd.DataFrame(
        {
            "event_id": ["e1", "e1"],
            "arm": [ARM_FULL, ARM_FULL],
            "selection_id": ["1", "2"],
            "opponent_id": ["2", "1"],
            "probability": [0.55, 0.50],
            "push_probability": [0.05, 0.05],
        }
    )

    result = add_full_reference_edge(bets, probabilities)
    expected = [
        0.55 * (100.0 / 110.0) - (1.0 - 0.55 - 0.05),
        0.50 * 1.20 - (1.0 - 0.50),
    ]
    assert np.allclose(result["full_reference_ev"], expected)
