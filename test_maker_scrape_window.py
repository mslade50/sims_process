"""Hard scrape rule for the maker's dry-run / preview / shadow reads. No network, no orders.
Run: python -m pytest test_maker_scrape_window.py -q"""
import datetime as dt
from zoneinfo import ZoneInfo

import maker_guard as mg

ET = ZoneInfo("America/New_York")


def ts(y, mo, d, h, mi):
    return dt.datetime(y, mo, d, h, mi, tzinfo=ET).timestamp()


def utc_ts(y, mo, d, h, mi):
    return dt.datetime(y, mo, d, h, mi, tzinfo=dt.timezone.utc).timestamp()


def not_live():
    return (True, "round 2: datagolf live=False")


def live():
    return (False, "round 2: datagolf live=True")


def test_boundaries_on_a_friday():
    assert mg.weekday_blackout(ts(2026, 10, 30, 7, 29)) is False
    assert mg.weekday_blackout(ts(2026, 10, 30, 7, 30)) is True
    assert mg.weekday_blackout(ts(2026, 10, 30, 11, 29)) is True
    assert mg.weekday_blackout(ts(2026, 10, 30, 11, 30)) is False


def test_weekend_is_not_in_the_window():
    assert mg.weekday_blackout(ts(2026, 10, 31, 9, 0)) is False
    assert mg.weekday_blackout(ts(2026, 11, 1, 9, 0)) is False


def test_dst_edt_and_est():
    assert mg.weekday_blackout(utc_ts(2026, 10, 30, 11, 29)) is False      # EDT: 07:29 ET
    assert mg.weekday_blackout(utc_ts(2026, 10, 30, 11, 30)) is True
    assert mg.weekday_blackout(utc_ts(2026, 11, 2, 12, 29)) is False       # EST after fall-back: 07:29 ET
    assert mg.weekday_blackout(utc_ts(2026, 11, 2, 12, 30)) is True
    assert mg.weekday_blackout(utc_ts(2026, 11, 2, 16, 29)) is True
    assert mg.weekday_blackout(utc_ts(2026, 11, 2, 16, 30)) is False
    assert mg.weekday_blackout(utc_ts(2026, 3, 9, 11, 29)) is False        # spring-forward Monday, EDT
    assert mg.weekday_blackout(utc_ts(2026, 3, 9, 11, 30)) is True


def test_blackout_refuses_without_reading_live_status():
    def boom():
        raise AssertionError("must not be called inside the window")
    assert "blackout" in mg.scrape_refusal(ts(2026, 10, 30, 9, 0), boom)


def test_round_in_play_refuses_outside_the_window():
    reason = mg.scrape_refusal(ts(2026, 10, 30, 15, 0), live)
    assert reason.startswith("scrape refused: round in play")


def test_not_live_outside_the_window_is_allowed():
    assert mg.scrape_refusal(ts(2026, 10, 30, 18, 0), not_live) is None
    assert mg.scrape_refusal(ts(2026, 10, 31, 9, 0), not_live) is None


def test_unreadable_live_status_refuses():
    def broken():
        raise OSError("datagolf down")
    assert "live status check failed" in mg.scrape_refusal(ts(2026, 10, 30, 15, 0), broken)


def test_both_sources_blind_is_treated_as_live_by_resolve_live():
    ok = not mg.resolve_live(None, None)[0]
    assert ok is False
    assert mg.scrape_refusal(ts(2026, 10, 30, 15, 0), lambda: (ok, "blind")) is not None
