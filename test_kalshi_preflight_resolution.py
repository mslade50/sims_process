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
