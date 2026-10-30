"""Hard scrape rule for the maker's dry-run / preview / shadow reads. No network, no orders.
Run: python -m pytest test_maker_scrape_window.py -q"""
import datetime as dt
import os
import runpy
import socket
import sys
import time
import types
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

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


# ── conservative resolution (item: a stale feed must not override a live schedule) ──
def test_live_schedule_with_stale_feed_is_refused():
    # DataGolf confidently stale (False) while the tee-time schedule says live (True)
    assert mg.resolve_live(True, False)[0] is False          # the --live resolver is unchanged: stale feed wins
    live_now, _ = mg.resolve_scrape_live(True, False)
    assert live_now is True


@pytest.mark.parametrize("sched,dg,refused", [
    (True, True, True), (True, False, True), (True, None, True),
    (False, True, True), (None, True, True), (None, False, True), (None, None, True),
    (False, None, True), (False, False, False),
])
def test_resolve_scrape_live_table(sched, dg, refused):
    assert mg.resolve_scrape_live(sched, dg)[0] is refused


# ── pre-import window check ──
def test_pre_import_refusal_modes():
    fri9 = ts(2026, 10, 30, 9, 0)
    assert "blackout" in mg.pre_import_refusal([], fri9)
    assert "blackout" in mg.pre_import_refusal(["--preview"], fri9)
    for flag in ("--live", "--cancel-all", "--cancel-script", "--list-orders"):
        assert mg.pre_import_refusal([flag], fri9) is None
    assert mg.pre_import_refusal([], ts(2026, 10, 30, 18, 0)) is None


# ── entrypoint-level: run kalshi_maker.py as __main__ ──
KM = str(Path(__file__).resolve().parent / "kalshi_maker.py")


class _Poison(types.ModuleType):
    def __getattr__(self, name):
        raise AssertionError(f"network-capable module touched during the window: {self.__name__}.{name}")


def _no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network boundary hit")
    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.setattr(socket, "getaddrinfo", boom)


def test_entrypoint_in_window_exits_zero_before_any_network_import(monkeypatch, capsys):
    _no_network(monkeypatch)
    for name in ("sim_inputs", "coeff_loader", "api_utils", "httpx", "dotenv"):
        monkeypatch.setitem(sys.modules, name, _Poison(name))
    monkeypatch.setattr(time, "time", lambda: ts(2026, 10, 30, 9, 0))
    monkeypatch.setattr(sys, "argv", ["kalshi_maker.py"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(KM, run_name="__main__")
    assert exc.value.code == 0
    assert "weekday 07:30-11:30 ET blackout" in capsys.readouterr().out


def test_entrypoint_recheck_before_scan_exits_zero_without_fetching(monkeypatch, capsys):
    monkeypatch.setenv("COEFFS_FROM_CACHE", "1")
    _no_network(monkeypatch)
    monkeypatch.setattr(time, "time", lambda: ts(2026, 10, 30, 18, 0))     # outside the window
    monkeypatch.setattr(sys, "argv", ["kalshi_maker.py", "--no-matchups"])
    answers = [None, "scrape refused: round in play (test: started during preconditions)"]
    calls = []
    monkeypatch.setattr(mg, "scrape_refusal", lambda now, check: calls.append(1) or answers[len(calls) - 1])
    import httpx
    fetched = []

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def close(self):
            pass

        def __getattr__(self, name):
            def rec(*a, **k):
                fetched.append(name)
                raise AssertionError(f"book fetch via client.{name}")
            return rec
    monkeypatch.setattr(httpx, "Client", FakeClient)
    import api_utils
    monkeypatch.setattr(api_utils, "fetch_live_stats", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(KM, run_name="__main__")
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert len(calls) == 2, "the refusal must be rechecked right before scan()"
    assert "started during preconditions" in out
    assert "[maker] tourney=" not in out, "scan() must not be reached"
    assert fetched == []


# ── round 2 review ──
def test_a_slow_check_that_crosses_0730_refuses():
    clock = {"t": ts(2026, 10, 30, 7, 29) + 59}

    def slow_but_clear():
        clock["t"] += 3           # the status I/O takes 3 s and ends at 07:30:02
        return (True, "not live")
    reason = mg.scrape_refusal(clock["t"], slow_but_clear, clock=lambda: clock["t"])
    assert reason and "blackout" in reason and "during the status check" in reason


def test_a_check_that_stays_outside_the_window_is_allowed():
    t = ts(2026, 10, 30, 18, 0)
    assert mg.scrape_refusal(t, lambda: (True, "ok"), clock=lambda: t + 5) is None


def _km(monkeypatch):
    monkeypatch.setenv("COEFFS_FROM_CACHE", "1")
    import kalshi_maker
    return kalshi_maker


def test_unknown_round_with_a_stale_feed_is_refused_for_scrape_but_not_for_live(monkeypatch):
    import pandas as pd
    km = _km(monkeypatch)
    sheet = types.ModuleType("sheet_config")

    def down(*a, **k):
        raise OSError("sheet unavailable")
    sheet.get_param = down
    monkeypatch.setitem(sys.modules, "sheet_config", sheet)
    stale = pd.DataFrame({"last_updated": ["2020-01-01 00:00:00"]})
    import api_utils
    monkeypatch.setattr(api_utils, "fetch_live_stats", lambda *a, **k: stale)
    monkeypatch.setattr(km, "_round_teetimes", lambda rnd: [])
    ok, reason = km._scrape_not_live()
    assert ok is False and "unknown" in reason.lower() or "refuse" in reason.lower()
    assert km._current_play_round_strict() is None
    assert km._current_play_round() == 1               # the --live path keeps its old fallback


def test_blank_or_missing_round_cell_is_unknown_for_scrape(monkeypatch):
    km = _km(monkeypatch)
    for value in (None, "", "   "):
        sheet = types.ModuleType("sheet_config")
        sheet.get_param = lambda name, default=None, _v=value: default if _v is None else _v
        monkeypatch.setitem(sys.modules, "sheet_config", sheet)
        assert km._current_play_round_strict() is None, repr(value)


def test_scrape_signals_use_a_clock_read_after_the_io(monkeypatch):
    import pandas as pd
    km = _km(monkeypatch)
    sheet = types.ModuleType("sheet_config")
    sheet.get_param = lambda *a, **k: "1"
    monkeypatch.setitem(sys.modules, "sheet_config", sheet)
    import api_utils
    now = {"t": 1_000_000.0}
    monkeypatch.setattr(km.time, "time", lambda: now["t"])

    def slow_feed(*a, **k):
        now["t"] += 3600                                  # the read takes an hour
        return pd.DataFrame({"last_updated": ["2026-10-30 12:00:00"]})
    monkeypatch.setattr(api_utils, "fetch_live_stats", slow_feed)
    seen = []
    monkeypatch.setattr(km, "_round_teetimes", lambda rnd: [now["t"] + 100, now["t"] + 200])
    monkeypatch.setattr(mg, "schedule_live", lambda a, b, n, round_hours=None: seen.append(n) or True)
    km._live_signals(strict=True)
    assert seen == [now["t"]] and seen[0] >= 1_000_000.0 + 3600


def test_pre_import_refusal_handles_argparse_abbreviations():
    fri9 = ts(2026, 10, 30, 9, 0)
    for flag in ("--liv", "--live", "--cancel-a", "--list", "--cancel-script=1"):
        assert mg.pre_import_refusal([flag], fri9) is None, flag
    for flag in ("--preview", "--prev", "--no-matchups", "--limit=3", "-x", "--"):
        assert mg.pre_import_refusal([flag], fri9) is not None, flag
