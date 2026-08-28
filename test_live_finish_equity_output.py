"""Regression tests for the live finish-position dashboard snapshot."""

import ast
from pathlib import Path

import pandas as pd


ROUND_SIM = Path(__file__).with_name("round_sim.py")


def _load_round_sim_functions(*names):
    """Load pure helpers without importing round_sim's live sheet config."""
    tree = ast.parse(ROUND_SIM.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in names
    ]
    module = ast.Module(body=functions, type_ignores=[])
    namespace = {"pd": pd}
    exec(compile(ast.fix_missing_locations(module), str(ROUND_SIM), "exec"), namespace)
    return tuple(namespace[name] for name in names)


def test_live_finish_snapshot_keeps_kalshi_topn_rows(tmp_path, monkeypatch):
    (write_live_finish_equity,) = _load_round_sim_functions(
        "write_live_finish_equity"
    )
    monkeypatch.chdir(tmp_path)
    rows = pd.DataFrame(
        [
            {
                "player_name": "player, retail",
                "bookmaker": "fanduel",
                "market_type": "top_10",
                "edge": 2.5,
            },
            {
                "player_name": "player, five",
                "bookmaker": "kalshi",
                "market_type": "top_5",
                "edge": 3.0,
            },
            {
                "player_name": "player, ten",
                "bookmaker": "kalshi",
                "market_type": "top_10",
                "edge": 4.0,
            },
            {
                "player_name": "player, twenty",
                "bookmaker": "kalshi",
                "market_type": "top_20",
                "edge": 5.0,
            },
        ]
    )

    path = write_live_finish_equity(rows, "test_event")

    assert path == "finish_equity_live_test_event.csv"
    written = pd.read_csv(tmp_path / path)
    kalshi = written[written["bookmaker"] == "kalshi"]
    assert set(kalshi["market_type"]) == {"top_5", "top_10", "top_20"}
    assert len(written) == len(rows)


def test_live_finish_snapshot_is_finalized_after_full_sim_exchange_pricing():
    tree = ast.parse(ROUND_SIM.read_text(encoding="utf-8"))
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    calls = [
        (node.lineno, node.func.id)
        for node in ast.walk(main)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    kalshi_lines = [line for line, name in calls if name == "price_kalshi_outrights"]
    writer_lines = [line for line, name in calls if name == "write_live_finish_equity"]

    assert len(kalshi_lines) == 2  # full simulation and --price-only
    assert len(writer_lines) == 1
    # The full simulation writes after Kalshi has been merged.  The separate
    # --price-only branch intentionally does not overwrite the complete
    # DataGolf + exchange dashboard snapshot with exchange-only rows.
    assert kalshi_lines[0] < writer_lines[0] < kalshi_lines[1]


def test_live_finish_snapshot_skips_empty_output(tmp_path, monkeypatch):
    (write_live_finish_equity,) = _load_round_sim_functions(
        "write_live_finish_equity"
    )
    monkeypatch.chdir(tmp_path)

    assert write_live_finish_equity(pd.DataFrame(), "test_event") is None
    assert not (tmp_path / "finish_equity_live_test_event.csv").exists()


def test_sims_kalshi_parser_accepts_current_topn_title_and_rules_metadata():
    player_parser, tournament_parser = _load_round_sim_functions(
        "_kalshi_outright_player",
        "_kalshi_outright_tournament",
    )
    for top_n in (5, 10, 20):
        market = {
            "title": f"Xander Schauffele finishes top {top_n}",
            "rules_primary": (
                f"If Xander Schauffele finishes in the top {top_n} "
                "(including ties) in the 2026 TOUR Championship, then the "
                "market resolves to Yes."
            ),
        }

        assert player_parser(market["title"]) == "Xander Schauffele"
        assert tournament_parser(market) == "TOUR Championship"


def test_sims_kalshi_scoping_joins_blank_titles_by_event_ticker():
    tournament_parser, scope_markets = _load_round_sim_functions(
        "_kalshi_outright_tournament",
        "_scope_kalshi_outright_markets",
    )
    # Inject the dependency referenced by the scope helper.
    scope_markets.__globals__["_kalshi_outright_tournament"] = tournament_parser
    markets = [
        {
            "title": "Xander Schauffele finishes top 5",
            "event_ticker": "KXPGATOP5-TOC26",
        },
        {
            "title": "Will Xander Schauffele win the TOUR Championship?",
            "event_ticker": "KXPGATOUR-TOC26",
        },
        {
            "title": "Rory McIlroy finishes top 5",
            "event_ticker": "KXPGATOP5-MAST26",
        },
        {
            "title": "Will Rory McIlroy win the Masters?",
            "event_ticker": "KXPGATOUR-MAST26",
        },
        {
            "title": "Unknown Golfer finishes top 5",
            "event_ticker": "KXPGATOP5-OTHER26",
        },
    ]

    scoped, matched, fallback = scope_markets(markets, "tourchamp")

    assert matched == ["TOUR Championship"]
    assert fallback == ""
    assert {market["event_ticker"] for market in scoped} == {
        "KXPGATOP5-TOC26",
        "KXPGATOUR-TOC26",
    }


def test_sims_kalshi_scoping_rejects_positively_wrong_configured_event():
    tournament_parser, scope_markets = _load_round_sim_functions(
        "_kalshi_outright_tournament",
        "_scope_kalshi_outright_markets",
    )
    scope_markets.__globals__["_kalshi_outright_tournament"] = tournament_parser
    markets = [
        {
            "title": "Will Xander Schauffele win the TOUR Championship?",
            "event_ticker": "KXPGATOUR-TOC26",
        },
        {
            "title": "Unknown Golfer finishes top 5",
            "event_ticker": "KXPGATOP5-OTHER26",
        },
    ]

    scoped, matched, rejection = scope_markets(markets, "masters")

    assert scoped == []
    assert matched == []
    assert "does not match" in rejection
    assert "TOUR Championship" in rejection


def test_sims_kalshi_unresolved_suffix_must_have_one_proven_event():
    tournament_parser, scope_markets = _load_round_sim_functions(
        "_kalshi_outright_tournament",
        "_scope_kalshi_outright_markets",
    )
    scope_markets.__globals__["_kalshi_outright_tournament"] = tournament_parser
    markets = [
        {
            "title": "Will Alpha Golfer win the TOUR Championship?",
            "event_ticker": "KXPGATOUR-SHARED26",
        },
        {
            "title": "Masters: Will Beta Golfer finish top 5?",
            "event_ticker": "KXPGATOP10-SHARED26",
        },
        {
            "title": "Unknown Golfer finishes top 20",
            "event_ticker": "KXPGATOP20-SHARED26",
        },
    ]

    scoped, matched, rejection = scope_markets(markets, "tourchamp")

    assert matched == ["TOUR Championship"]
    assert rejection == ""
    assert [market["title"] for market in scoped] == [
        "Will Alpha Golfer win the TOUR Championship?"
    ]


def test_full_sim_isolates_kalshi_failure_before_novig_and_snapshot_write():
    tree = ast.parse(ROUND_SIM.read_text(encoding="utf-8"))
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )

    def called_names(node):
        return {
            child.func.id
            for child in ast.walk(node)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
        }

    kalshi_tries = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Try)
        and "price_kalshi_outrights" in called_names(node)
        and "price_novig_outrights" not in called_names(node)
    ]
    # Pick the first/full-simulation isolation block (the other Kalshi call is
    # in the separate --price-only path and intentionally outside this guard).
    isolated = min(kalshi_tries, key=lambda node: node.lineno)
    assert any(
        isinstance(handler.type, ast.Name) and handler.type.id == "Exception"
        for handler in isolated.handlers
    )

    calls = [
        (node.lineno, node.func.id)
        for node in ast.walk(main)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    first_novig = min(line for line, name in calls if name == "price_novig_outrights")
    snapshot = min(line for line, name in calls if name == "write_live_finish_equity")
    assert isolated.end_lineno < first_novig < snapshot
