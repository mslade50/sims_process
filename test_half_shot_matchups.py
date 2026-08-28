import os
from pathlib import Path

import pandas as pd
import pytest

os.environ.setdefault("COEFFS_FROM_CACHE", "1")

import clv
import grade_bets
import odds_loader
import reprice_core
from dashboard import data_layer
from round_matchup_coverage import is_actionable_matchup_price


def _priced_rows():
    return pd.DataFrame({
        "Player 1": ["alpha", "alpha", "alpha", "alpha"],
        "Player 2": ["beta", "beta", "beta", "beta"],
        "Bookmaker": ["pinnacle", "betcris", "betcris", "fanduel"],
        "Ties": ["void", "void", "void", "void"],
        "P1 Odds": [100, 100, 100, 100],
        "P2 Odds": [-110, -110, -110, -110],
        "P1 Line": [None, -0.5, 0.5, -0.5],
        "P2 Line": [None, 0.5, -0.5, 0.5],
        "line_verified": [False, True, True, False],
        "my_odds_p1": [0.4772727] * 4,
        "my_odds_p2": [0.5227273] * 4,
        "my_odds_p1_tl": [0.42] * 4,
        "my_odds_p2_tl": [0.46] * 4,
    })


def test_settlement_probabilities_price_cover_and_fail_unverified_spread_closed():
    priced = _priced_rows()
    p1, p2, kind = reprice_core.matchup_settlement_probabilities(priced)

    assert p1.iloc[0] == pytest.approx(0.4772727)
    assert p2.iloc[0] == pytest.approx(0.5227273)
    assert p1.iloc[1] == pytest.approx(0.42)
    assert p2.iloc[1] == pytest.approx(0.58)
    assert p1.iloc[2] == pytest.approx(0.54)
    assert p2.iloc[2] == pytest.approx(0.46)
    assert kind.tolist() == [
        "straight", "half_shot", "half_shot", "unsupported_spread"
    ]
    assert pd.isna(p1.iloc[3]) and pd.isna(p2.iloc[3])


def test_no_line_columns_remain_backward_compatible_for_non_betcris():
    legacy = _priced_rows().iloc[[0]].drop(
        columns=["P1 Line", "P2 Line", "line_verified"]
    )
    p1, p2, kind = reprice_core.matchup_settlement_probabilities(legacy)

    assert kind.iloc[0] == "straight"
    assert p1.iloc[0] == pytest.approx(0.4772727)
    assert p2.iloc[0] == pytest.approx(0.5227273)


def test_calculate_edges_uses_settlement_fair_and_quarantines_unsupported():
    calculated = reprice_core.calculate_edges(_priced_rows())

    assert calculated.loc[1, "Fair_p1"] == reprice_core.implied_to_american(0.42)
    assert calculated.loc[1, "Fair_p2"] == reprice_core.implied_to_american(0.58)
    assert pd.isna(calculated.loc[3, "edge_p1"])
    assert reprice_core.actionable_matchup_mask(calculated).tolist() == [
        True, True, True, False
    ]


def test_scraped_json_keeps_contracts_separate_and_drops_legacy_betcris():
    data = {"match_list": [
        {
            "p1_player_name": "Alpha", "p2_player_name": "Beta", "ties": "void",
            "odds": {"betcris": {"p1": "+100", "p2": "-110"}},
        },
        {
            "p1_player_name": "Alpha", "p2_player_name": "Beta", "ties": "void",
            "odds": {"betcris": {
                "p1": "+100", "p2": "-110", "p1_line": None,
                "p2_line": None, "line_verified": True,
            }},
        },
        {
            "p1_player_name": "Alpha", "p2_player_name": "Beta", "ties": "void",
            "odds": {
                "betcris": {
                    "p1": "+130", "p2": "-150", "p1_line": -0.5,
                    "p2_line": 0.5, "line_verified": True,
                },
                "pinnacle": {"p1": "+105", "p2": "-115"},
            },
        },
    ]}

    parsed = odds_loader._parse_datagolf_json(data)

    assert len(parsed) == 3
    assert len(parsed[parsed["Bookmaker"] == "betcris"]) == 2
    assert set(parsed["P1 Line"].dropna()) == {-0.5}
    assert len(parsed[parsed["Bookmaker"] == "pinnacle"]) == 1


def test_scraped_one_sided_line_metadata_is_marked_unsupported():
    parsed = odds_loader._parse_datagolf_json({"match_list": [{
        "p1_player_name": "Alpha", "p2_player_name": "Beta", "ties": "void",
        "odds": {"pinnacle": {
            "p1": "+100", "p2": "-110", "p1_line": -0.5,
            "line_verified": True,
        }},
    }]})

    assert parsed["line_parse_error"].tolist() == [True]
    parsed["my_odds_p1"] = 0.48
    parsed["my_odds_p2"] = 0.52
    parsed["my_odds_p1_tl"] = 0.42
    parsed["my_odds_p2_tl"] = 0.46
    p1, p2, kind = reprice_core.matchup_settlement_probabilities(parsed)
    assert kind.tolist() == ["unsupported_spread"]
    assert p1.isna().all() and p2.isna().all()


def test_load_matchup_odds_keeps_straight_and_half_contracts(monkeypatch):
    scraped = {"match_list": [
        {
            "p1_player_name": "Alpha", "p2_player_name": "Beta", "ties": "void",
            "odds": {"betcris": {
                "p1": -110, "p2": -110, "p1_line": None,
                "p2_line": None, "line_verified": True,
            }},
        },
        {
            "p1_player_name": "Alpha", "p2_player_name": "Beta", "ties": "void",
            "odds": {"betcris": {
                "p1": 130, "p2": -150, "p1_line": -0.5,
                "p2_line": 0.5, "line_verified": True,
            }},
        },
    ]}
    monkeypatch.setattr(odds_loader, "_fetch_scraped_json", lambda _market: scraped)
    monkeypatch.setattr(
        odds_loader, "guard_scraped_data", lambda data, *_args, **_kwargs: data
    )
    monkeypatch.setattr(
        odds_loader, "_fetch_datagolf_api", lambda *_args, **_kwargs: pd.DataFrame()
    )

    loaded = odds_loader.load_matchup_odds(api_key="test-key")

    assert len(loaded) == 2
    assert {
        tuple(row) for row in loaded[["P1 Line", "P2 Line"]].fillna(0).to_numpy()
    } == {(0.0, 0.0), (-0.5, 0.5)}


def test_new_sim_uses_contract_aware_dedup_at_both_write_boundaries():
    source = Path("new_sim.py").read_text(encoding="utf-8")
    assert "df_match = deduplicate_matchup_contracts(pd.DataFrame(rows_mu))" in source
    assert "deduplicate_matchup_contracts(dfb).to_csv" in source


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({}, True),
        ({"P1 Line": None, "P2 Line": None}, True),
        ({"P1 Line": 0, "P2 Line": 0}, True),
        ({"P1 Line": -0.5, "P2 Line": 0.5}, False),
        ({"P1 Line": 0.5, "P2 Line": 0.5}, False),
        ({"P1 Line": -0.5, "P2 Line": None}, False),
        ({"P1 Line": "bad", "P2 Line": "bad"}, False),
    ],
)
def test_exchange_substitution_is_limited_to_straight_contracts(row, expected):
    assert reprice_core.is_straight_matchup_contract(row) is expected


def test_matchup_emails_label_spread_separately_from_odds():
    for filename in ("round_sim.py", "new_sim.py"):
        source = Path(filename).read_text(encoding="utf-8")
        assert ">Spread</th>" in source
        assert ">Odds</th>" in source
        assert "selected_spread_line(row)" in source

    tournament_source = Path("new_sim.py").read_text(encoding="utf-8")
    assert "if _exch and is_straight_matchup_contract(row):" in tournament_source


def test_matchup_email_attachments_filter_and_label_contracts():
    priced = reprice_core.calculate_edges(_priced_rows())
    priced["bet_on"] = priced["Player 1"]

    attachment = reprice_core.prepare_matchup_attachment_rows(priced)

    assert attachment.index.tolist() == [0, 1, 2]
    assert attachment["Spread"].tolist() == [0.0, -0.5, 0.5]
    source = Path("round_sim.py").read_text(encoding="utf-8")
    for function_name in (
        "build_betonline_all_matchups_csv",
        "build_all_books_fair_csv",
    ):
        block = source.split(f"def {function_name}", 1)[1].split("\ndef ", 1)[0]
        assert "prepare_matchup_attachment_rows" in block
        for column in (
            '"P1 Line"', '"P2 Line"', '"Spread"',
            '"line_verified"', '"market_kind"',
        ):
            assert column in block


@pytest.mark.parametrize(
    ("price", "expected"),
    [
        ({"p1": -110, "p2": -110}, False),
        ({"p1": -110, "p2": -110, "p1_line": 0, "p2_line": 0}, False),
        ({"p1": -110, "p2": -110, "p1_line": 0, "p2_line": 0,
          "line_verified": True}, True),
        ({"p1": -110, "p2": -110, "p1_line": -0.5, "p2_line": 0.5}, False),
        ({"p1": -110, "p2": -110, "p1_line": -0.5, "p2_line": 0.5,
          "line_verified": True}, True),
        ({"p1": -110, "p2": -110, "p1_line": -0.5}, False),
        ({"p1": -110, "p2": -110, "p1_line": -1.5, "p2_line": 1.5,
          "line_verified": True}, False),
    ],
)
def test_betcris_coverage_requires_supported_contract(price, expected):
    assert is_actionable_matchup_price("betcris", price) is expected


def test_non_betcris_legacy_straight_is_coverage_eligible_but_bad_spread_is_not():
    assert is_actionable_matchup_price("pinnacle", {"p1": -110, "p2": -110})
    assert not is_actionable_matchup_price(
        "pinnacle",
        {"p1": -110, "p2": -110, "p1_line": -0.5, "p2_line": 0.5},
    )


def _tie_results():
    return pd.DataFrame({
        "player_name": ["alpha", "beta"],
        "round_2": [70, 70],
        "fin_num": [5, 5],
        "fin_text": ["T5", "T5"],
    })


@pytest.mark.parametrize(
    ("line", "expected", "units"),
    [(0.5, "win", 1.0), (-0.5, "loss", -1.1), (0.0, "push", 0.0)],
)
def test_round_half_shot_ties_settle_by_selected_line(line, expected, units):
    row = {
        "player_1": "alpha", "player_2": "beta", "bet_on": "alpha",
        "round": 2, "bookmaker": "betcris", "p1_odds": -110,
        "p1_line": line, "p2_line": -line, "line_verified": True,
    }

    grade = grade_bets.grade_round_matchup(row, _tie_results())

    assert grade["result"] == expected
    assert grade["units_won"] == pytest.approx(units)
    assert grade["spread_line"] == line


def test_legacy_betcris_is_left_ungraded_for_audit():
    row = {
        "player_1": "alpha", "player_2": "beta", "bet_on": "alpha",
        "round": 2, "bookmaker": "betcris", "p1_odds": -110,
    }
    grade = grade_bets.grade_round_matchup(row, _tie_results())

    assert grade["result"] == "no_data"
    assert grade["skip_write"] is True
    assert "unknown BetCRIS handicap" in grade["notes"]


def test_tournament_plus_half_wins_a_tied_finish():
    row = {
        "player_1": "alpha", "player_2": "beta", "bet_on": "alpha",
        "bookmaker": "betcris", "p1_odds": 120,
        "p1_line": 0.5, "p2_line": -0.5, "line_verified": True,
    }
    grade = grade_bets.grade_tournament_matchup(row, _tie_results())

    assert grade["result"] == "win"
    assert grade["spread_line"] == 0.5


def test_clv_excludes_spreads_and_ambiguous_betcris_but_keeps_verified_straight():
    frame = pd.DataFrame([
        {"bookmaker": "betcris", "player_1": "alpha", "bet_on": "alpha",
         "p1_line": -0.5, "p2_line": 0.5, "line_verified": True},
        {"bookmaker": "betcris", "player_1": "alpha", "bet_on": "alpha"},
        {"bookmaker": "betcris", "player_1": "alpha", "bet_on": "alpha",
         "p1_line": None, "p2_line": None, "line_verified": True},
        {"bookmaker": "pinnacle", "player_1": "alpha", "bet_on": "alpha"},
    ])

    eligible = clv._drop_line_ineligible_matchups(frame)

    assert eligible.index.tolist() == [2, 3]


def test_alert_identity_and_message_include_selected_line():
    straight = reprice_core.alerted_key("alpha", "beta", "alpha", 0.0)
    half = reprice_core.alerted_key("alpha", "beta", "alpha", -0.5)
    assert straight != half

    row = pd.DataFrame([{
        "Player 1": "alpha", "Player 2": "beta", "bet_on": "alpha",
        "Bookmaker": "betcris", "P1 Odds": 120, "P2 Odds": -140,
        "Fair_p1": 110, "Fair_p2": -130, "edge_on": 8.0,
        "P1 Line": -0.5, "P2 Line": 0.5,
    }])
    text = reprice_core._matchup_alert_messages(row, 2, "event")[0]
    assert "alpha -0.5 vs beta" in text


def test_dashboard_primary_hides_legacy_and_all_spreads_half_tab_is_explicit(
    monkeypatch,
):
    raw = pd.DataFrame([
        {"run_timestamp": "2026-08-28T01:00:00", "round": "2",
         "player_1": "alpha", "player_2": "beta", "bet_on": "alpha",
         "bookmaker": "betcris", "p1_odds": -110, "p2_odds": -110,
         "fair_p1": -110, "fair_p2": -110, "line_verified": ""},
        {"run_timestamp": "2026-08-28T01:00:00", "round": "2",
         "player_1": "gamma", "player_2": "delta", "bet_on": "gamma",
         "bookmaker": "pinnacle", "p1_odds": -110, "p2_odds": -110,
         "fair_p1": -110, "fair_p2": -110},
        {"run_timestamp": "2026-08-28T01:00:00", "round": "2",
         "player_1": "echo", "player_2": "foxtrot", "bet_on": "echo",
         "bookmaker": "betcris", "p1_odds": 120, "p2_odds": -140,
         "fair_p1": 110, "fair_p2": -130, "p1_line": -0.5,
         "p2_line": 0.5, "line_verified": True},
        {"run_timestamp": "2026-08-28T01:00:00", "round": "2",
         "player_1": "hotel", "player_2": "india", "bet_on": "hotel",
         "bookmaker": "fanduel", "p1_odds": 120, "p2_odds": -140,
         "fair_p1": 110, "fair_p2": -130, "p1_line": -1.5,
         "p2_line": 1.5, "line_verified": True},
        {"run_timestamp": "2026-08-28T01:00:00", "round": "2",
         "player_1": "juliet", "player_2": "kilo", "bet_on": "juliet",
         "bookmaker": "fanduel", "p1_odds": 120, "p2_odds": -140,
         "fair_p1": 110, "fair_p2": -130, "p1_line": 0.5,
         "p2_line": 0.5, "line_verified": True},
        {"run_timestamp": "2026-08-28T01:00:00", "round": "2",
         "player_1": "lima", "player_2": "mike", "bet_on": "lima",
         "bookmaker": "fanduel", "p1_odds": 120, "p2_odds": -140,
         "fair_p1": 110, "fair_p2": -130, "p1_line": "bad",
         "p2_line": "bad", "line_verified": True},
        {"run_timestamp": "2026-08-28T01:00:00", "round": "2",
         "player_1": "november", "player_2": "oscar", "bet_on": "november",
         "bookmaker": "fanduel", "p1_odds": 120, "p2_odds": -140,
         "fair_p1": 110, "fair_p2": -130, "p1_line": -0.5,
         "p2_line": None, "line_verified": True},
    ])
    monkeypatch.setattr(
        data_layer, "_load_matchups_from_sheets", lambda *_args, **_kwargs: raw
    )

    primary = data_layer.get_matchups(2, line_mode="straight")
    half = data_layer.get_matchups(2, line_mode="half_shot")

    assert primary["Bookmaker"].tolist() == ["pinnacle"]
    assert half["Bookmaker"].tolist() == ["betcris"]
    assert half["spread_line"].tolist() == [-0.5]


def test_dashboard_legacy_tab_without_book_or_line_columns_fails_soft(monkeypatch):
    raw = pd.DataFrame([{
        "run_timestamp": "2026-08-28T01:00:00", "round": "2",
        "player_1": "alpha", "player_2": "beta", "bet_on": "alpha",
        "p1_odds": -110, "p2_odds": -110,
        "fair_p1": -110, "fair_p2": -110,
    }])
    monkeypatch.setattr(
        data_layer, "_load_matchups_from_sheets", lambda *_args, **_kwargs: raw
    )

    primary = data_layer.get_matchups(2, line_mode="straight")

    assert len(primary) == 1
    assert primary["spread_line"].tolist() == [0.0]
