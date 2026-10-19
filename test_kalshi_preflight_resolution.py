"""Kalshi preflight resolution: missing listing vs a broken lookup.

Synthetic markets only. No network, no credentials, no licensed odds.
The Oct 17-18 2026 runs failed with raw=52, overlap=0, event=None against a
120-player live field. Those 52 markets were the Masters futures book (the
only open KXPGATOUR markets); Bermuda's winner markets were created later,
at 2026-10-19T02:39Z. This file locks that shape as "not listed yet".
"""
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
    """Today's trap: DataGolf's current field overlaps Bermuda while sim_inputs
    is still event 527. Overlap success must not print ALL CLEAR."""
    assessment = kp.assess_winner_resolution(
        _weekly(120, event="BUBC26", tournament="Weekly Open"), _field(120), min_overlap=10)
    assert assessment["status"] == "resolved"
    verdict = kp.classify_preflight(
        assessment, configured_event_ids=[527],
        week_events=[{"event_id": 528, "event_name": "Butterfield Bermuda Championship"}],
        field_event_id=528, tourney="baycurrent", winner_events=[{
            "event_ticker": "KXPGATOUR-BAC26",
            "title": "Baycurrent Classic Winner",
            "sub_title": "2026 Baycurrent Classic",
            "book": "settled",
        }])
    assert verdict["warning"] is None
    assert verdict["issues"][0].startswith("STALE CONFIG:")
    assert "528" in verdict["issues"][0]
    assert "527" in verdict["issues"][0]


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
