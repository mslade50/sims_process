import numpy as np
import pandas as pd

from archive.build_all_round_tournament_outcome_authority import (
    add_tournament_score_tie_authority,
)


def test_half_half_requires_equal_score_and_equal_completion_round():
    quotes = pd.DataFrame(
        {
            "event_key": ["e", "e", "e"],
            "event_start_date": ["2024-01-01"] * 3,
            "p1_id": ["a", "a", "a"],
            "p2_id": ["b", "c", "d"],
            "p1_outcome": [0.5, 0.5, 0.5],
            "p2_outcome": [0.5, 0.5, 0.5],
            "pair_key": ["a|b", "a|c", "a|d"],
        }
    )
    rounds = pd.DataFrame(
        {
            "event_key": ["e"] * 7,
            "dg_id": ["a", "a", "b", "b", "c", "c", "d"],
            "round_num": [1, 2, 1, 2, 1, 2, 1],
            "round_date": pd.date_range("2024-01-01", periods=7),
            "cumulative_score": [70, 140, 71, 140, 72, 141, 70],
        }
    )

    enriched, audit = add_tournament_score_tie_authority(quotes, rounds)

    assert enriched["physical_p1_outcome"].tolist()[:1] == [0.5]
    assert np.isnan(enriched.loc[1, "physical_p1_outcome"])
    assert np.isnan(enriched.loc[2, "physical_p1_outcome"])
    assert enriched["other_void_with_scores"].tolist() == [False, True, True]
    assert enriched["no_score_void"].tolist() == [False, False, False]
    assert audit["verified_physical_tie_logical_pairs"] == 1
    assert audit["other_void_with_scores_logical_pairs"] == 2

