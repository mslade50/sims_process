"""Regression coverage for book-complete matchup and 3-ball emails."""

import pandas as pd

from reprice_core import build_matchup_outputs, retain_unique_actionable_quotes


def _matchup_row(p1, p2, book, o1, o2, edge1, edge2):
    return {
        "Player 1": p1,
        "Player 2": p2,
        "Bookmaker": book,
        "Ties": "push",
        "P1 Odds": o1,
        "P2 Odds": o2,
        # This fixture represents an inspected straight-H2H quote. BetCRIS
        # rows without this provenance are intentionally quarantined because
        # its feed can carry a hidden +/-0.5-stroke contract.
        "P1 Line": None,
        "P2 Line": None,
        "line_verified": True,
        "Fair_p1": -120,
        "Fair_p2": 120,
        "edge_p1": edge1,
        "edge_p2": edge2,
        "half_shot_p1": 0.0,
        "half_shot_p2": 0.0,
    }


def test_sharp_email_keeps_each_qualifying_book_for_same_pair():
    rows = [
        _matchup_row("akshay", "thomas", "betcris", 110, -120, 12.0, -5.0),
        _matchup_row("akshay", "thomas", "betonline", 105, -115, 10.0, -4.0),
    ]
    _, sharp = build_matchup_outputs(
        pd.DataFrame(rows),
        4,
        {"akshay": 1.2, "thomas": 0.4},
        {"akshay": 50, "thomas": 50},
    )

    assert set(sharp["Bookmaker"]) == {"betcris", "betonline"}


def test_reversed_feed_duplicate_collapses_but_preserves_distinct_side():
    rows = pd.DataFrame([
        {
            **_matchup_row("rai", "straka", "betonline", 115, -125, 9.0, -3.0),
            "edge_on": 9.0,
            "bet_on": "rai",
        },
        {
            **_matchup_row("straka", "rai", "betonline", -125, 115, -3.0, 8.5),
            "edge_on": 8.5,
            "bet_on": "rai",
        },
        {
            **_matchup_row("rai", "straka", "betonline", 115, -125, -3.0, 8.0),
            "edge_on": 8.0,
            "bet_on": "straka",
        },
    ])

    unique = retain_unique_actionable_quotes(rows, player_count=2)

    assert len(unique) == 2
    assert set(unique["bet_on"]) == {"rai", "straka"}
    assert unique.loc[unique["bet_on"] == "rai", "edge_on"].iloc[0] == 9.0


def test_threeball_email_keeps_books_and_collapses_reversed_duplicate():
    base = {
        "Ties": "dead heat",
        "edge_on": 11.0,
        "bet_on": "a",
    }
    rows = pd.DataFrame([
        {**base, "Player 1": "a", "Player 2": "b", "Player 3": "c",
         "P1 Odds": 250, "P2 Odds": 275, "P3 Odds": 300,
         "Bookmaker": "betcris"},
        {**base, "Player 1": "c", "Player 2": "a", "Player 3": "b",
         "P1 Odds": 300, "P2 Odds": 250, "P3 Odds": 275,
         "Bookmaker": "betcris", "edge_on": 10.0},
        {**base, "Player 1": "a", "Player 2": "b", "Player 3": "c",
         "P1 Odds": 240, "P2 Odds": 275, "P3 Odds": 300,
         "Bookmaker": "betonline"},
    ])

    unique = retain_unique_actionable_quotes(rows, player_count=3)

    assert len(unique) == 2
    assert set(unique["Bookmaker"]) == {"betcris", "betonline"}
