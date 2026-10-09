"""Offline contracts: no operational imports, credentials, or API requests."""
import ast
from collections import Counter, defaultdict
from pathlib import Path
import re

import pandas as pd
import pytest

import kalshi_match as km
from kalshi_winner import winner_player, winner_title_parts


def market(player="Nick Hardy", event="BAOUC26", tournament="Bank of Utah Championship", **changes):
    value = {
        "ticker": f"KXPGATOUR-{event}-NHAR",
        "event_ticker": f"KXPGATOUR-{event}",
        "title": f"{tournament}: {player} wins",
        "yes_sub_title": player,
        "market_type": "binary",
        "rules_primary": f"If {player} wins the {tournament}, then the market resolves to Yes.",
    }
    return dict(value, **changes)


def load_functions(filename, *names, **namespace):
    # Pipeline/trader/maker imports can read live state. Compile only pure
    # functions or a pricer with every external read mocked by its caller.
    path = Path(__file__).with_name(filename)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = {"re": re, "pd": pd, "Counter": Counter, "defaultdict": defaultdict, **namespace}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return [scope[n] for n in names]


@pytest.mark.parametrize("title", [
    "Bank of Utah Championship: Nick Hardy wins",
    "Bank of Utah Championship: Nick Hardy wins?",
    "Will Nick Hardy win the Bank of Utah Championship?",
    "Bank of Utah Championship: Will Nick Hardy win?",
])
def test_old_new_titles_and_consumers(title):
    value = market(title=title)
    assert winner_title_parts(title) == ("Nick Hardy", "Bank of Utah Championship")
    assert winner_player(value) == "Nick Hardy"
    assert km._player_from_title(title) == "Nick Hardy"
    for kind in ("title", "subtitle"):
        assert km.player_from_market(value, kind) == "Nick Hardy"
    assert load_functions("round_sim.py", "_kalshi_outright_player")[0](title) == "Nick Hardy"


@pytest.mark.parametrize("changes", [
    {"yes_sub_title": "Harry Higgs"},
    {"rules_primary": "If Harry Higgs wins the Bank of Utah Championship, then the market resolves to Yes."},
    {"rules_primary": "If Nick Hardy wins the Masters, then the market resolves to Yes."},
    {"event_ticker": "KXPGATOUR-MAST27"},
    {"ticker": "KXPGAR2LEAD-BAOUC26-NHAR"},
    {"market_type": "scalar"},
    {"primary_participant_key": "winning_score"},
    {"_market_type": "top_5"},
    {"ticker": "KXPGATOUR-BAOUC26R2-NHAR", "event_ticker": "KXPGATOUR-BAOUC26R2"},
    {"title": "Bank of Utah Championship: Nick Hardy wins round 2"},
    {"title": "Bank of Utah Championship Round 2: Nick Hardy wins"},
    {"title": "Bank of Utah Championship: Nick Hardy wins the 3-ball"},
    {"title": "Bank of Utah Championship: Winning score under 270"},
    {"title": "Will Nick Hardy win the 1st round 3-ball matchup?"},
    {"title": "Bank of Utah Championship: Nick Hardy makes a hole in one"},
])
def test_conflicting_metadata_and_nonwinner_outcomes_fail_closed(changes):
    assert winner_player(market(**changes)) == ""
    # For winner tickers even subtitle mode must not override a conflict.
    if "ticker" not in changes:
        assert km.player_from_market(market(**changes), "subtitle") == ""


def test_subtitle_requires_winner_rules_if_title_missing():
    assert winner_player(market(title="")) == "Nick Hardy"
    assert winner_player(market(title="", rules_primary="")) == ""
    assert winner_player(market(title="Unknown proposition")) == ""
    assert winner_player(market(title="", yes_sub_title="")) == ""
    assert winner_player(market(yes_sub_title="")) == "Nick Hardy"
    assert winner_player(market(yes_sub_title="Yes")) == "Nick Hardy"


def test_name_normalization_and_simultaneous_events():
    field = {"hardy, nick", "higgs, harry", "van rooyen, erik"}
    utah = [market(p) for p in ("Nick Hardy", "Harry Higgs", "Erik van Rooyen")]
    masters = [market("Nick Hardy", "MAST27", "Masters")]
    matched, report = km.match_markets(utah + masters, field, kind="title")
    assert report["counts"] == {"BAOUC26": 3, "MAST27": 1}
    assert report["target"] == "BAOUC26"
    assert matched == utah
    assert km.norm_name(km.player_from_market(utah[2], "title"), field_set=field) == "van rooyen, erik"
    conflict = market(yes_sub_title="Harry Higgs")
    assert km.event_overlap_counts([conflict], field, kind="title") == {}
    assert km.match_markets([conflict], field, kind="title")[0] == []


def test_existing_ancillary_subtitle_and_threeballs():
    value = {"ticker": "KXPGA3BALL-BAOUC26R2-A", "yes_sub_title": "Nick Hardy beats Harry Higgs and Erik van Rooyen"}
    assert km.player_from_market(value) == "Nick Hardy"
    assert km.three_ball_members(value) == ["Nick Hardy", "Harry Higgs", "Erik van Rooyen"]
    assert km.player_from_market(market(ticker="KXPGAR2LEAD-BAOUC26-NHAR"), "title") == ""


def test_live_tournament_scope_accepts_utah_rejects_masters_and_blank_config():
    parser, scope = load_functions("round_sim.py", "_kalshi_outright_tournament", "_scope_kalshi_outright_markets")
    scope.__globals__["_kalshi_outright_tournament"] = parser
    values = [market(), market("Harry Higgs", "MAST27", "Masters")]
    selected, names, reason = scope(values, "utah")
    assert len(selected) == 1 and selected[0]["event_ticker"] == "KXPGATOUR-BAOUC26"
    assert names == ["Bank of Utah Championship"] and not reason
    assert scope(values, "valspar")[0] == []
    assert scope(values, "")[0] == []


@pytest.mark.parametrize("filename,function", [
    ("round_sim.py", "price_kalshi_outrights"),
    ("new_sim.py", "price_kalshi_outrights_tourney"),
])
@pytest.mark.parametrize("configured", ["utah", "valspar", "", "american_express"])
def test_pricers_use_winner_probability_and_correct_event(monkeypatch, tmp_path, filename, function, configured):
    import httpx
    import os
    import sys
    import time
    from types import SimpleNamespace

    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "sim_inputs", SimpleNamespace(name_replacements={}))
    monkeypatch.setattr(time, "sleep", lambda _: None)
    current_event = "Bank of Utah Championship" if configured != "american_express" else "American Express"
    other_event = "Masters" if configured != "american_express" else "American Century Championship"
    values = [
        market(tournament=current_event, yes_bid_dollars="0.18", yes_ask_dollars="0.20"),
        market("Nick Hardy", "MAST27", other_event, yes_bid_dollars="0.18", yes_ask_dollars="0.20"),
        market(tournament=current_event, yes_sub_title="Harry Higgs", yes_bid_dollars="0.18", yes_ask_dollars="0.20"),
        market(tournament=current_event, yes_bid_dollars="0.00", yes_ask_dollars="0.01"),
    ]
    books = []

    class Client:
        def __init__(self, **kwargs):
            pass

        def get(self, url, params=None):
            if url.endswith("/markets"):
                payload = {"markets": values if params["series_ticker"] == "KXPGATOUR" else [], "cursor": ""}
            else:
                assert url.endswith("KXPGATOUR-BAOUC26-NHAR/orderbook")
                books.append(url)
                payload = {"orderbook_fp": {"yes_dollars": [["0.18", "1000"]], "no_dollars": [["0.80", "1000"]]}}
            return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "Client", Client)
    names = (function, "_kalshi_taker_fee")
    if filename == "round_sim.py":
        names += ("_kalshi_outright_player", "_kalshi_outright_tournament", "_scope_kalshi_outright_markets")
    pricer = load_functions(
        filename, *names, tourney=configured, os=os, name_replacements={},
        implied_to_american=lambda p: p, implied_prob_to_american_odds=lambda p: p,
        prob_to_american=lambda p: p,
    )[0]
    frame = pd.DataFrame([{"player_name": "hardy, nick", "simulated_win_prob": 0.30, "top_5_nodh": 0.85}])
    result = pricer(frame, {}, {})
    if configured not in {"utah", "american_express"}:
        assert result.empty and books == []
    else:
        assert len(books) == 1
        assert set(result["player_name"]) == {"hardy, nick"}
        assert set(result["market_type"]) == {"winner"}
        assert result.loc[result["side"] == "yes", "sim_prob"].eq(0.30).all()


def test_compound_surname_miss_stays_unmatched_without_forced_alias():
    # The existing particle splitter does not include 'dumont' in the surname.
    # This task changes title parsing only, so preserve the conservative miss.
    display = "Adrien Dumont De Chassart"
    field = {"dumont de chassart, adrien"}
    assert km.norm_name(display, field_set=field) == "de chassart, adrien dumont"
    assert km.event_overlap_counts([market(display)], field, kind="title") == {}


def test_execution_consumers_and_shared_normalizer_stay_at_head():
    import subprocess
    repo = Path(__file__).resolve().parent

    def original(name):
        return subprocess.check_output([
            "git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo), "show", f"HEAD:{name}"
        ], text=True, encoding="utf-8")

    for name in ("kalshi_maker.py", "kalshi_trader.py"):
        assert (repo / name).read_text(encoding="utf-8") == original(name)
    old = ast.parse(original("kalshi_match.py"))
    current = ast.parse((repo / "kalshi_match.py").read_text(encoding="utf-8"))
    for name in ("norm_name", "_split_first_last"):
        before = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == name)
        after = next(n for n in current.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(before) == ast.dump(after)

    maker = ast.parse(original("kalshi_maker.py"))
    calls = {n.func.attr for n in ast.walk(maker) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
             and n.func.value.id == "_km"}
    assert calls == {"norm_name"}
    # Maker's ancillary H2H imports are unchanged and do not use the repaired
    # shared title/overlap helpers. Do not import operational modules here.
    ancillary = ast.parse(original("kalshi_ancillary.py"))
    for name in ("_match_tournament", "h2h_prob"):
        function = next(n for n in ancillary.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert not any(isinstance(n, ast.Name) and n.id == "km" for n in ast.walk(function))
    trader = ast.parse(original("kalshi_trader.py"))
    assert not any(isinstance(n, ast.ImportFrom) and n.module in {
        "kalshi_match", "kalshi_winner", "new_sim", "round_sim"
    } for n in ast.walk(trader))
