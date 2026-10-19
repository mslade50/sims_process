"""Kalshi preflight resolution: missing listing vs a broken lookup.

Synthetic markets only. No network, no credentials, no licensed odds.
The Oct 17-18 2026 runs failed with raw=52, overlap=0, event=None against a
120-player live field. Those 52 markets were the Masters futures book (the
only open KXPGATOUR markets); Bermuda's winner markets were created later,
at 2026-10-19T02:39Z. This file locks that shape as "not listed yet".
"""
import os
import sys

os.environ.setdefault("COEFFS_FROM_CACHE", "1")

import kalshi_preflight as kp


def _market(player, event, tournament, suffix):
    return {
        "ticker": f"KXPGATOUR-{event}-{suffix}",
        "event_ticker": f"KXPGATOUR-{event}",
        "title": f"{tournament}: {player} wins",
        "yes_sub_title": player,
        "market_type": "binary",
        "rules_primary": f"If {player} wins the {tournament}, then the market resolves to Yes.",
    }


def _field_player(i):
    return f"Test Player{i}", f"player{i}, test"


def _field(n):
    return {key for _, key in (_field_player(i) for i in range(n))}


def _weekly(n, event="BAC26", tournament="Weekly Open"):
    markets = []
    for i in range(n):
        display, _ = _field_player(i)
        markets.append(_market(display, event, tournament, f"P{i:03d}"))
    return markets


def _futures(n, event="MAST27", tournament="Masters Tournament"):
    return [
        _market(f"Other Future{i}", event, tournament, f"F{i:03d}")
        for i in range(n)
    ]


def test_oct17_readable_futures_with_zero_overlap_is_not_listed():
    """The failed runs: 120-player field, 52 readable markets, overlap {}."""
    field = _field(120)
    markets = _futures(52)
    result = kp.assess_winner_resolution(markets, field, min_overlap=10)

    assert result["raw"] == 52
    assert result["named"] == 52
    assert result["leader"] == 0
    assert result["counts"] == {}
    assert result["target"] is None
    # Before the fix, target is None so main appended this and exited 1.
    assert result["status"] == "not_listed"
    assert "not listed yet" in result["message"]
    assert kp.resolution_failure_issue(result) is None
    assert kp.resolution_failure_issue(result, strict=True).startswith(
        "RESOLUTION FAILED: no Kalshi event overlaps our field")


def test_unreadable_winner_markets_still_fail():
    """Oct 1 2026 shape: a full book was open and zero titles named a player."""
    field = _field(120)
    markets = []
    for i in range(173):
        markets.append({
            "ticker": f"KXPGATOUR-BAOUC26-X{i:03d}",
            "event_ticker": "KXPGATOUR-BAOUC26",
            "title": "Prop format the winner parser does not recognize",
            "yes_sub_title": "Yes",
            "market_type": "binary",
            "rules_primary": "Resolves some other way.",
        })
    result = kp.assess_winner_resolution(markets, field, min_overlap=10)
    assert result["raw"] == 173
    assert result["named"] == 0
    assert result["leader"] == 0
    assert result["status"] == "failed"
    issue = kp.resolution_failure_issue(result)
    assert issue.startswith("RESOLUTION FAILED: no Kalshi event overlaps our field")
    assert "title-format break" in issue


def test_full_field_book_still_resolves():
    field = _field(72)
    result = kp.assess_winner_resolution(_weekly(72), field, min_overlap=10)
    assert result["status"] == "resolved"
    assert result["target"] == "BAC26"
    assert result["leader"] == 72
    assert kp.resolution_failure_issue(result) is None


def test_futures_sliver_is_not_a_successful_resolution():
    """Oct 13-16 passed ALL CLEAR: 19-20 of 72 local names sat in MAST27."""
    field = _field(72)
    sliver = _weekly(20, event="MAST27", tournament="Masters Tournament")
    rest = _futures(32)
    result = kp.assess_winner_resolution(sliver + rest, field, min_overlap=10)
    assert result["leader"] == 20
    assert result["leader"] >= 10
    assert result["counts"] == {"MAST27": 20}
    assert result["coverage"] < kp.FIELD_COVERAGE_MIN
    assert result["status"] == "not_listed"
    assert "not listed yet" in result["message"]
    assert kp.resolution_failure_issue(result) is None


def test_small_overlap_stays_a_loud_name_drift_failure():
    field = _field(72)
    result = kp.assess_winner_resolution(_weekly(5), field, min_overlap=10)
    assert result["status"] == "low_overlap"
    assert result["leader"] == 5
    issue = kp.resolution_failure_issue(result)
    assert issue.startswith("LOW OVERLAP:")
    assert "BAC26" in issue


def test_empty_winner_book_is_not_listed():
    result = kp.assess_winner_resolution([], _field(120), min_overlap=10)
    assert result["status"] == "not_listed"
    assert result["raw"] == 0
    assert "not listed yet" in result["message"]


def test_near_complete_listing_still_resolves():
    # Bank of Utah week: 117 of 120 field players had an open winner market.
    field = _field(120)
    result = kp.assess_winner_resolution(_weekly(117), field, min_overlap=10)
    assert result["status"] == "resolved"
    assert result["leader"] == 117


def _not_listed_assessment():
    return kp.assess_winner_resolution(_futures(52), _field(120), min_overlap=10)


def test_stale_settled_tourney_fails_instead_of_not_listed():
    """Oct 17-18: config still baycurrent after KXPGATOUR-BAC26 settled.

    The live field did not overlap the only open book (Masters futures). That
    shape is 'not listed' only when sim_inputs is current. A settled configured
    event is STALE CONFIG and fails.
    """
    assessment = _not_listed_assessment()
    assert assessment["status"] == "not_listed"
    settled = [{
        "event_ticker": "KXPGATOUR-BAC26",
        "title": "Baycurrent Classic Winner",
        "sub_title": "2026 Baycurrent Classic",
        "book": "settled",
    }]
    verdict = kp.classify_preflight(
        assessment, configured_event_ids=[527], week_events=[],
        field_event_id=None, tourney="baycurrent", winner_events=settled)
    assert verdict["warning"] is None
    assert len(verdict["issues"]) == 1
    assert verdict["issues"][0].startswith("STALE CONFIG:")
    assert "KXPGATOUR-BAC26" in verdict["issues"][0]
    assert "not a missing listing" in verdict["issues"][0]
    # Strict mode does not hide a forgotten config.
    strict = kp.classify_preflight(
        assessment, configured_event_ids=[527], week_events=[],
        field_event_id=None, tourney="baycurrent", winner_events=settled, strict=True)
    assert strict["issues"][0].startswith("STALE CONFIG:")


def test_stale_event_id_mismatch_fails_even_if_the_live_field_resolves():
    """DataGolf's current field is 528 while sim_inputs is still event 527.

    The winner book can resolve and the previous event can be settled. That
    is a simple lag: NOT READY YET, a green warning, and orders stay blocked.
    Overlap success must not print ALL CLEAR.
    """
    assessment = kp.assess_winner_resolution(
        _weekly(120, event="BUBC26", tournament="Weekly Open"), _field(120), min_overlap=10)
    assert assessment["status"] == "resolved"
    week = [{"event_id": 528, "event_name": "Butterfield Bermuda Championship"}]
    verdict = kp.classify_preflight(
        assessment, configured_event_ids=[527], week_events=week,
        field_event_id=528, field_event_name="Butterfield Bermuda Championship",
        tourney="baycurrent", winner_events=[{
            "event_ticker": "KXPGATOUR-BAC26",
            "title": "Baycurrent Classic Winner",
            "sub_title": "2026 Baycurrent Classic",
            "book": "settled",
        }])
    ready = ("NOT READY YET: sim_inputs is event 527, live event is 528 "
             "(Butterfield Bermuda Championship); advance sim_inputs")
    assert verdict["issues"] == []
    assert verdict["warning"] == ready
    assert "528" in verdict["warning"]
    assert "527" in verdict["warning"]
    assert "STALE CONFIG" not in verdict["warning"]
    ok, reason = kp.inputs_ready_to_quote(
        [527], 528, "Butterfield Bermuda Championship", week_events=week)
    assert ok is False
    assert reason == ready


def test_current_config_not_listed_stays_a_warning():
    assessment = _not_listed_assessment()
    verdict = kp.classify_preflight(
        assessment, configured_event_ids=[528],
        week_events=[{"event_id": 528}], field_event_id=528, tourney="bermuda",
        winner_events=[{
            "event_ticker": "KXPGATOUR-BUBC26",
            "title": "Butterfield Bermuda Championship Winner",
            "sub_title": "2026 Butterfield Bermuda Championship",
            "book": "open",
        }])
    assert verdict["issues"] == []
    assert verdict["warning"]
    assert "not listed yet" in verdict["warning"]


def test_open_book_for_the_configured_tourney_is_not_stale():
    reason = kp.settled_tourney_reason("baycurrent", [
        {"event_ticker": "KXPGATOUR-BAC26", "title": "Baycurrent Classic Winner",
         "sub_title": "2026 Baycurrent Classic", "book": "open"},
    ])
    assert reason is None


def _unreadable(n, event="BAC26"):
    return [{
        "ticker": f"KXPGATOUR-{event}-X{i:03d}",
        "event_ticker": f"KXPGATOUR-{event}",
        "title": "Prop format the winner parser does not recognize",
        "yes_sub_title": "Yes",
        "market_type": "binary",
        "rules_primary": "Resolves some other way.",
    } for i in range(n)]


def _current_baycurrent():
    """sim_inputs matches this week's event, so 'not listed' would be green."""
    return dict(
        configured_event_ids=[527],
        week_events=[{"event_id": 527, "event_name": "Baycurrent Classic"}],
        field_event_id=527,
        tourney="baycurrent",
        winner_events=[{
            "event_ticker": "KXPGATOUR-BAC26",
            "title": "Baycurrent Classic Winner",
            "sub_title": "2026 Baycurrent Classic",
            "book": "open",
        }],
    )


def test_unreadable_majority_fails_even_when_futures_overlap_clears_ten():
    """72 markets that name nobody plus 19 readable Masters names.

    The 19 match the field, so the old '10 or more' shortcut called this
    not listed and exited green. Under half the open winner markets name a
    player, so it stays RESOLUTION FAILED and is an issue main Telegrams.
    """
    field = _field(72)
    markets = _unreadable(72) + _weekly(19, event="MAST27", tournament="Masters Tournament")
    result = kp.assess_winner_resolution(markets, field, min_overlap=10)
    assert result["raw"] == 91
    assert result["named"] == 19
    assert result["named"] * 2 < result["raw"]
    assert result["leader"] == 19
    assert result["leader"] >= 10
    assert result["counts"] == {"MAST27": 19}
    assert result["parse_rate"] < 0.5
    assert result["status"] == "failed"
    assert "under half" in result["message"]
    assert "not a missing listing" in result["message"]
    verdict = kp.classify_preflight(result, **_current_baycurrent())
    assert verdict["warning"] is None
    assert len(verdict["issues"]) == 1
    assert verdict["issues"][0].startswith("RESOLUTION FAILED:")
    assert "not listed yet" not in verdict["issues"][0]
    assert not any(item.startswith("STALE CONFIG:") for item in verdict["issues"])


class _Response:
    def __init__(self, status_code, payload=None, json_error=None):
        self.status_code = status_code
        self._payload = {} if payload is None else payload
        self._json_error = json_error

    def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._payload


class _Client:
    """Scripted read-only markets client. No order or portfolio calls."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def get(self, url, params=None):
        assert "/markets" in url
        assert "/orders" not in url and "/portfolio" not in url
        self.calls.append(params or {})
        nxt = self._responses.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt


def _assert_fetch_failure(error, detail):
    assert error is not None
    assert detail in error
    assessment = kp.assess_winner_resolution(
        [], _field(72), min_overlap=10, fetch_error=error)
    assert assessment["status"] == "fetch_failed"
    assert assessment["raw"] == 0
    assert "not an empty book" in assessment["message"]
    verdict = kp.classify_preflight(assessment, **_current_baycurrent())
    assert verdict["warning"] is None
    assert verdict["issues"]
    assert verdict["issues"][0].startswith("RESOLUTION FAILED:")
    assert "winner market fetch failed" in verdict["issues"][0]
    assert "not an empty book" in verdict["issues"][0]
    assert "not listed yet" not in verdict["issues"][0]


def test_winner_fetch_http_500_fails_instead_of_not_listed():
    """A 500 used to come back as [] and exit green as 'not listed yet'."""
    client = _Client([_Response(500, {"error": "upstream"})])
    markets, error = kp.load_open_winner_markets(client)
    assert markets is None
    assert client.calls == [{"limit": 200, "status": "open", "series_ticker": "KXPGATOUR"}]
    _assert_fetch_failure(error, "HTTP 500")


def test_winner_fetch_timeout_bad_json_and_pagination_fail(monkeypatch):
    monkeypatch.setattr(kp.time, "sleep", lambda *_a, **_k: None)

    timed = _Client([TimeoutError("timed out")])
    markets, error = kp.load_open_winner_markets(timed)
    assert markets is None
    _assert_fetch_failure(error, "TimeoutError")

    broken = _Client([_Response(200, json_error=ValueError("not json"))])
    markets, error = kp.load_open_winner_markets(broken)
    assert markets is None
    _assert_fetch_failure(error, "bad JSON")

    # A full page plus a cursor, then a 500. The first page must not count
    # as a successful book.
    paged = _Client([
        _Response(200, {"markets": [{}] * 200, "cursor": "more"}),
        _Response(500, {}),
    ])
    markets, error = kp.load_open_winner_markets(paged)
    assert markets is None
    assert len(paged.calls) == 2
    _assert_fetch_failure(error, "HTTP 500")

    endless = _Client([
        _Response(200, {"markets": [{}] * 200, "cursor": f"page-{i}"})
        for i in range(30)
    ])
    markets, error = kp.load_open_winner_markets(endless)
    assert markets is None
    assert len(endless.calls) == 25
    _assert_fetch_failure(error, "pagination did not finish")


def test_successful_empty_winner_book_with_current_config_stays_a_warning():
    """Green only when both winner GETs succeed and the book is actually empty."""
    empty = _Response(200, {"markets": []})
    client = _Client([empty, empty])
    markets, error = kp.load_open_winner_markets(client)
    assert markets == []
    assert error is None
    assert [call["series_ticker"] for call in client.calls] == ["KXPGATOUR", "KXPGAWIN"]
    assessment = kp.assess_winner_resolution(
        markets, _field(72), min_overlap=10, fetch_error=error)
    assert assessment["status"] == "not_listed"
    verdict = kp.classify_preflight(assessment, **_current_baycurrent())
    assert verdict["issues"] == []
    assert "not listed yet" in verdict["warning"]


def _lag_week():
    return [{"event_id": 528, "event_name": "Butterfield Bermuda Championship"}]


def _resolved_live_book():
    return kp.assess_winner_resolution(
        _weekly(120, event="BUBC26", tournament="Weekly Open"), _field(120), min_overlap=10)


def test_matching_live_event_keeps_the_resolved_path_and_allows_quotes():
    assessment = _resolved_live_book()
    assert assessment["status"] == "resolved"
    week = _lag_week()
    verdict = kp.classify_preflight(
        assessment, configured_event_ids=[528], week_events=week,
        field_event_id=528, field_event_name="Butterfield Bermuda Championship",
        tourney="bermuda")
    assert verdict["issues"] == []
    assert verdict["warning"] is None
    ok, reason = kp.inputs_ready_to_quote(
        [528], 528, "Butterfield Bermuda Championship", week_events=week)
    assert ok is True
    assert reason == ""


def test_field_and_schedule_disagreement_stays_stale_config():
    """Two live sources that name different events are not a simple lag."""
    assessment = _resolved_live_book()
    verdict = kp.classify_preflight(
        assessment, configured_event_ids=[527],
        week_events=[{"event_id": 529, "event_name": "Other Championship"}],
        field_event_id=528, field_event_name="Butterfield Bermuda Championship",
        tourney="baycurrent")
    assert verdict["warning"] is None
    assert verdict["issues"][0].startswith("STALE CONFIG:")
    assert "528" in verdict["issues"][0]
    assert "529" in verdict["issues"][0]
    ok, reason = kp.inputs_ready_to_quote(
        [527], 528, "Butterfield Bermuda Championship",
        week_events=[{"event_id": 529, "event_name": "Other Championship"}])
    assert ok is True
    assert reason == ""


def test_lag_without_a_name_still_says_not_ready():
    verdict = kp.classify_preflight(
        _resolved_live_book(), configured_event_ids=[527],
        week_events=[{"event_id": 528}], field_event_id=528)
    assert verdict["issues"] == []
    assert verdict["warning"] == (
        "NOT READY YET: sim_inputs is event 527, live event is 528 "
        "(name unavailable); advance sim_inputs")


def test_strict_turns_not_ready_red_but_does_not_release_orders(monkeypatch):
    monkeypatch.setenv("KALSHI_PREFLIGHT_STRICT", "1")
    week = _lag_week()
    verdict = kp.classify_preflight(
        _resolved_live_book(), configured_event_ids=[527], week_events=week,
        field_event_id=528, field_event_name="Butterfield Bermuda Championship",
        strict=True)
    assert verdict["warning"] is None
    assert verdict["issues"][0].startswith("NOT READY YET:")
    assert "advance sim_inputs" in verdict["issues"][0]
    ok, reason = kp.inputs_ready_to_quote(
        [527], 528, "Butterfield Bermuda Championship", week_events=week)
    assert ok is False
    assert reason.startswith("NOT READY YET:")


def test_unreadable_majority_stays_red_when_inputs_also_lag():
    field = _field(72)
    markets = _unreadable(72) + _weekly(19, event="MAST27", tournament="Masters Tournament")
    result = kp.assess_winner_resolution(markets, field, min_overlap=10)
    assert result["status"] == "failed"
    verdict = kp.classify_preflight(
        result, configured_event_ids=[527], week_events=_lag_week(),
        field_event_id=528, field_event_name="Butterfield Bermuda Championship")
    assert verdict["warning"] is None
    assert verdict["issues"][0].startswith("RESOLUTION FAILED:")
    assert "under half" in verdict["issues"][0]
    assert any(item.startswith("NOT READY YET:") for item in verdict["issues"])


def test_fetch_failure_stays_red_when_inputs_also_lag():
    assessment = kp.assess_winner_resolution(
        [], _field(72), min_overlap=10, fetch_error="KXPGATOUR: HTTP 500")
    verdict = kp.classify_preflight(
        assessment, configured_event_ids=[527], week_events=_lag_week(),
        field_event_id=528, field_event_name="Butterfield Bermuda Championship")
    assert verdict["warning"] is None
    assert verdict["issues"][0].startswith("RESOLUTION FAILED:")
    assert "winner market fetch failed" in verdict["issues"][0]
    assert "not an empty book" in verdict["issues"][0]
    assert any(item.startswith("NOT READY YET:") for item in verdict["issues"][1:])


def test_low_overlap_stays_red_when_inputs_also_lag():
    assessment = kp.assess_winner_resolution(_weekly(5), _field(72), min_overlap=10)
    assert assessment["status"] == "low_overlap"
    verdict = kp.classify_preflight(
        assessment, configured_event_ids=[527], week_events=_lag_week(),
        field_event_id=528, field_event_name="Butterfield Bermuda Championship")
    assert verdict["warning"] is None
    assert verdict["issues"][0].startswith("LOW OVERLAP:")


def test_strict_turns_a_current_not_listed_warning_red():
    verdict = kp.classify_preflight(
        _not_listed_assessment(), configured_event_ids=[528],
        week_events=[{"event_id": 528}], field_event_id=528, tourney="bermuda",
        winner_events=[{
            "event_ticker": "KXPGATOUR-BUBC26",
            "title": "Butterfield Bermuda Championship Winner",
            "sub_title": "2026 Butterfield Bermuda Championship",
            "book": "open",
        }], strict=True)
    assert verdict["warning"] is None
    assert verdict["issues"][0].startswith("RESOLUTION FAILED:")
    assert "not listed yet" in verdict["issues"][0]


def test_schedule_only_lag_is_not_ready_and_blocks_quotes():
    """A confirmed schedule with no field id is still one live source."""
    verdict = kp.classify_preflight(
        _resolved_live_book(), configured_event_ids=[527], week_events=_lag_week(),
        field_event_id=None)
    assert verdict["issues"] == []
    assert verdict["warning"].startswith("NOT READY YET:")
    assert "Butterfield Bermuda Championship" in verdict["warning"]
    ok, reason = kp.inputs_ready_to_quote([527], None, week_events=_lag_week())
    assert ok is False
    assert reason == verdict["warning"]


def test_unconfirmed_live_event_does_not_block_quotes():
    ok, reason = kp.inputs_ready_to_quote([527], None, week_events=None)
    assert ok is True
    assert reason == ""
    ok, reason = kp.inputs_ready_to_quote([527], None, week_events=[])
    assert ok is True
    assert reason == ""


def _patch_preflight_io(monkeypatch, *, markets, fetch_error, field_event_id, event_ids,
                        tourney="baycurrent", winner_events=None):
    import sim_inputs
    if winner_events is None:
        winner_events = [{
            "event_ticker": "KXPGATOUR-BAC26",
            "title": "Baycurrent Classic Winner",
            "sub_title": "",
            "book": "settled",
        }]
    monkeypatch.setattr(kp, "_client", lambda: object())
    monkeypatch.setattr(kp, "load_field", lambda tourney: {
        "names": _field(120), "source": "datagolf_live",
        "event_id": field_event_id,
        "event_name": "Butterfield Bermuda Championship",
        "detail": "synthetic",
    })
    monkeypatch.setattr(kp, "_week_events", lambda: _lag_week())
    monkeypatch.setattr(kp, "_tourney_winner_books", lambda client, tourney: winner_events)
    monkeypatch.setattr(kp, "load_open_winner_markets", lambda client: (markets, fetch_error))
    monkeypatch.setattr(kp, "discover_golf_series", lambda client: {})
    monkeypatch.setattr(kp, "_get_open", lambda client, series: [])
    monkeypatch.setattr(sim_inputs, "event_ids", event_ids, raising=False)
    monkeypatch.setattr(sim_inputs, "tourney", tourney, raising=False)
    monkeypatch.setattr(sim_inputs, "name_replacements", {}, raising=False)
    monkeypatch.delenv("KALSHI_PREFLIGHT_STRICT", raising=False)
    monkeypatch.setattr(sys, "argv", ["kalshi_preflight.py"])
    sent = []
    monkeypatch.setattr(kp, "_telegram", lambda text: sent.append(text))
    return sent


def test_main_lag_exits_green_without_telegram(monkeypatch, capsys):
    sent = _patch_preflight_io(
        monkeypatch, markets=_weekly(120, event="BUBC26", tournament="Weekly Open"),
        fetch_error=None, field_event_id=528, event_ids=[527])
    code = kp.main()
    out = capsys.readouterr().out
    assert code == 0
    assert sent == []
    assert "::warning title=Kalshi not ready yet::" in out
    assert ("NOT READY YET: sim_inputs is event 527, live event is 528 "
            "(Butterfield Bermuda Championship); advance sim_inputs") in out
    assert "::error" not in out
    assert "ALL CLEAR" not in out


def test_main_fetch_error_still_exits_red(monkeypatch, capsys):
    sent = _patch_preflight_io(
        monkeypatch, markets=None, fetch_error="KXPGATOUR: HTTP 500",
        field_event_id=528, event_ids=[528], tourney="bermuda",
        winner_events=[{
            "event_ticker": "KXPGATOUR-BUBC26",
            "title": "Butterfield Bermuda Championship Winner",
            "sub_title": "",
            "book": "open",
        }])
    code = kp.main()
    out = capsys.readouterr().out
    assert code == 1
    assert sent
    assert "RESOLUTION FAILED:" in out
    assert "winner market fetch failed" in out
    assert "NOT READY YET:" not in out


def test_field_source_names_live_datagolf_not_the_configured_file():
    text = kp.format_field_source({
        "names": _field(120), "source": "datagolf_live",
        "event_id": 528, "event_name": "Butterfield Bermuda Championship",
        "detail": "DataGolf field-updates, the current event",
    }, "baycurrent")
    assert "DataGolf live current event" in text
    assert "event_id=528" in text
    assert "baycurrent" in text
    local = kp.format_field_source({
        "names": _field(72), "source": "local", "event_id": None,
        "event_name": None, "detail": "player_names_live_baycurrent.json",
    }, "baycurrent")
    assert "local file" in local
    assert "not available" in local


def _load_maker(monkeypatch):
    import kalshi_maker as maker
    import sim_inputs
    return maker, sim_inputs


def test_lagging_inputs_halt_before_scan_or_post(monkeypatch):
    """A confirmed lag blocks --live before any other gate or order post.

    KALSHI_PREFLIGHT_STRICT stays on so a red CI flag cannot release the halt.
    """
    maker, sim_inputs = _load_maker(monkeypatch)
    monkeypatch.setattr(sim_inputs, "event_ids", [527])
    monkeypatch.setenv("KALSHI_PREFLIGHT_STRICT", "1")
    monkeypatch.setattr(maker, "_live_event_for_orders", lambda: (
        528, "Butterfield Bermuda Championship",
        [{"event_id": 528, "event_name": "Butterfield Bermuda Championship"}],
    ))
    called = {"should_trade": 0, "post": 0, "scan": 0}

    def should_trade(*_a, **_k):
        called["should_trade"] += 1
        return True, ""

    def post(cands):
        called["post"] += 1
        return {}

    def scan():
        called["scan"] += 1
        return [], {}

    monkeypatch.setattr(maker.maker_guard, "should_trade", should_trade)
    monkeypatch.setattr(maker, "reconcile_and_post", post)
    monkeypatch.setattr(maker, "scan", scan)
    ok, reason = maker._check_live_preconditions()
    assert ok is False
    assert reason == (
        "NOT READY YET: sim_inputs is event 527, live event is 528 "
        "(Butterfield Bermuda Championship); advance sim_inputs")
    assert maker._abort_live_post(ok) is True
    assert called == {"should_trade": 0, "post": 0, "scan": 0}


def test_matching_inputs_do_not_take_the_lag_halt(monkeypatch):
    maker, sim_inputs = _load_maker(monkeypatch)
    monkeypatch.setattr(sim_inputs, "event_ids", [528])
    monkeypatch.setattr(maker, "_live_event_for_orders", lambda: (
        528, "Butterfield Bermuda Championship",
        [{"event_id": 528, "event_name": "Butterfield Bermuda Championship"}],
    ))

    def should_trade(*_a, **_k):
        return False, "kill switch test"

    monkeypatch.setattr(maker.maker_guard, "should_trade", should_trade)
    ok, reason = maker._check_live_preconditions()
    assert ok is False
    assert reason == "kill switch test"
    assert "NOT READY" not in reason


def test_unconfirmed_live_event_does_not_newly_halt_the_maker(monkeypatch):
    maker, sim_inputs = _load_maker(monkeypatch)
    monkeypatch.setattr(sim_inputs, "event_ids", [527])
    monkeypatch.setattr(maker, "_live_event_for_orders", lambda: (None, None, None))

    def should_trade(*_a, **_k):
        return False, "kill switch test"

    monkeypatch.setattr(maker.maker_guard, "should_trade", should_trade)
    ok, reason = maker._check_live_preconditions()
    assert ok is False
    assert reason == "kill switch test"


def test_live_event_lookup_stays_read_only_when_datagolf_fails(monkeypatch):
    maker, _sim_inputs = _load_maker(monkeypatch)
    calls = []

    def field(*_a, **_k):
        calls.append("field")
        raise RuntimeError("field down")

    def week(*_a, **_k):
        calls.append("week")
        raise RuntimeError("schedule down")

    import api_utils
    monkeypatch.setattr(api_utils, "fetch_field_updates", field)
    monkeypatch.setattr(api_utils, "fetch_pga_events_this_week", week)
    monkeypatch.setenv("DATAGOLF_API_KEY", "synthetic")
    field_id, name, week_events = maker._live_event_for_orders()
    assert field_id is None
    assert name is None
    assert week_events is None
    assert calls == ["field", "week"]
