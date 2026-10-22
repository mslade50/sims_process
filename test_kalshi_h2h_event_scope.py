"""Kalshi H2H and the round-sim NoVig matcher must not price the busiest event."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIELD = [
    "scheffler, scottie",
    "mcilroy, rory",
    "rahm, jon",
    "morikawa, collin",
]


def _load(filename, name):
    path = ROOT / filename
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


def _h2h(player, opponent, event, tournament, suffix):
    return {
        "ticker": f"KXPGAH2H-{event}-{suffix}",
        "event_ticker": f"KXPGAH2H-{event}",
        "title": f"Will {player} beat {opponent} in the {tournament}?",
    }


def test_title_match_ignores_a_busier_open_event():
    scope = _load("new_sim.py", "scope_kalshi_h2h_markets")
    ours = _h2h("Scottie Scheffler", "Rory McIlroy", "VAL26", "Valspar Championship", "A")
    untitled = {"ticker": "KXPGAH2H-VAL26-BLANK", "event_ticker": "KXPGAH2H-VAL26", "title": ""}
    busy = [
        _h2h(f"Ghost {i}", f"Other {i}", "BUSY26", "Busier Open", str(i))
        for i in range(8)
    ]
    kept, report = scope([*busy, ours, untitled], "valspar", FIELD)
    assert report["path"] == "title"
    assert report["reason"] == ""
    tickers = {market["ticker"] for market in kept}
    assert "KXPGAH2H-VAL26-A" in tickers
    assert "KXPGAH2H-VAL26-BLANK" in tickers
    assert not any(ticker.startswith("KXPGAH2H-BUSY26") for ticker in tickers)


def test_overlap_replaces_the_busiest_event_when_the_title_misses():
    scope = _load("new_sim.py", "scope_kalshi_h2h_markets")
    pairs = [
        ("Scottie Scheffler", "Rory McIlroy", "A"),
        ("Jon Rahm", "Collin Morikawa", "B"),
        ("Rory McIlroy", "Jon Rahm", "C"),
        ("Collin Morikawa", "Scottie Scheffler", "D"),
    ]
    ours = [
        _h2h(player, opponent, "OUR26", "Renamed Championship", suffix)
        for player, opponent, suffix in pairs
    ]
    busy = [
        _h2h(f"Ghost {i}", f"Other {i}", "BUSY26", "Busier Open", str(i))
        for i in range(10)
    ]
    kept, report = scope([*busy, *ours], "unrelated_slug", FIELD)
    assert report["path"] == "overlap"
    assert report["target"] == "OUR26"
    assert report["reason"] == ""
    assert {market["ticker"] for market in kept} == {market["ticker"] for market in ours}


def test_no_overlap_prices_no_rows_instead_of_the_busiest_event():
    scope = _load("new_sim.py", "scope_kalshi_h2h_markets")
    busy = [
        _h2h(f"Ghost {i}", f"Other {i}", "BUSY26", "Busier Open", str(i))
        for i in range(6)
    ]
    kept, report = scope(busy, "valspar", FIELD)
    assert kept == []
    assert report["matched"] == 0
    assert report["reason"]


def test_thin_overlap_aborts_instead_of_keeping_one_pair():
    scope = _load("new_sim.py", "scope_kalshi_h2h_markets")
    pair = [
        _h2h("Scottie Scheffler", "Rory McIlroy", "PAIR26", "Other Championship", "A"),
        _h2h("Rory McIlroy", "Scottie Scheffler", "PAIR26", "Other Championship", "B"),
    ]
    kept, report = scope(pair, "unrelated_slug", FIELD)
    assert kept == []
    assert "only" in report["reason"]


def test_sources_no_longer_fall_back_to_the_busiest_event():
    new_sim = (ROOT / "new_sim.py").read_text(encoding="utf-8")
    round_sim = (ROOT / "round_sim.py").read_text(encoding="utf-8")
    assert "most_common" not in new_sim
    assert "falling back to most-common" not in new_sim
    assert "scope_kalshi_h2h_markets(" in new_sim
    assert "max(events, key=lambda e: e.get(\"child_events_aggregate\"" not in round_sim
    assert "select_novig_tournament(" in round_sim


def test_novig_matcher_returns_no_event_when_the_name_misses():
    select = _load("round_sim.py", "select_novig_tournament")
    events = [
        {"id": "busy", "description": "Busier Open", "child_events_aggregate": {"aggregate": {"count": 40}}},
        {"id": "ours", "description": "Valspar Championship", "child_events_aggregate": {"aggregate": {"count": 2}}},
    ]
    chosen, note = select(events, "valspar")
    assert chosen["id"] == "ours"
    assert "Valspar" in note

    missed, miss_note = select(events, "unrelated_slug")
    assert missed is None
    assert "pricing no rows" in miss_note
    assert "Busier" in miss_note

    blank, blank_note = select(events, "")
    assert blank is None
    assert "pricing no rows" in blank_note
