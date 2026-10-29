"""Offline tests for the prior-round matchup recap (round_recap.py) and the
killed-edge capture that feeds it. No network, Sheets, or Telegram."""

import sys
from pathlib import Path
from unittest.mock import patch

import gspread
import numpy as np
import pandas as pd
import pytest

import reprice
import reprice_core as rc
import round_recap
import sheets_storage as ss


# ── fakes ────────────────────────────────────────────────────────────────────
class FakeWS:
    def __init__(self, headers=None, records=None):
        self.headers = list(headers or [])
        self.rows = []
        for rec in records or []:
            self.rows.append([rec.get(h, "") for h in self.headers])

    def row_values(self, n):
        return list(self.headers)

    def append_row(self, row, value_input_option=None):
        self.headers = list(row)

    def append_rows(self, rows, value_input_option=None):
        self.rows.extend([list(r) for r in rows])

    def get_all_values(self):
        return [self.headers] + [list(r) for r in self.rows]

    def get_all_records(self):
        return [dict(zip(self.headers, r)) for r in self.rows]


class FakeSpreadsheet:
    def __init__(self, tabs=None):
        self.tabs = dict(tabs or {})

    def worksheet(self, name):
        if name not in self.tabs:
            raise gspread.exceptions.WorksheetNotFound(name)
        return self.tabs[name]

    def add_worksheet(self, title, rows, cols):
        self.tabs[title] = FakeWS()
        return self.tabs[title]


def _round_mu_row(p1, p2, book, o1, o2, bet_on, ts, edge=8.0):
    return {
        "run_timestamp": ts, "event_id": "100", "round": 2,
        "player_1": p1, "player_2": p2, "bookmaker": book, "ties_rule": "",
        "p1_odds": o1, "p2_odds": o2, "bet_on": bet_on, "edge_on": edge,
        "p1_line": "", "p2_line": "", "line_verified": "", "result": "",
    }


def _killed_row(p1, p2, book, o1, o2, bet_on, reason, edge=6.0):
    return {
        "event_id": "100", "round": 2, "player_1": p1, "player_2": p2,
        "bookmaker": book, "ties_rule": "", "p1_odds": o1, "p2_odds": o2,
        "bet_on": bet_on, "edge_on": edge, "kill_reason": reason,
        "p1_line": "", "p2_line": "", "line_verified": "",
    }


def _scores(live=(), terminal=(), **by_player):
    """Frame shaped like fetch_round_scores output."""
    names = list(by_player) + list(live) + list(terminal)
    status = ["final"] * len(by_player) + ["live"] * len(live) + ["terminal"] * len(terminal)
    values = [float(v) for v in by_player.values()] + [np.nan] * (len(live) + len(terminal))
    return pd.DataFrame({"player_name": names, "round_2": values, "status": status})


def _spreadsheet():
    mu = FakeWS(ss.ROUND_MU_HEADERS, [
        _round_mu_row("alpha", "bravo", "pinnacle", 120, -140, "alpha", "2026-10-23 10:00"),
        # same pairing+side at a later price: still one alerted bet
        _round_mu_row("alpha", "bravo", "betonline", 125, -145, "alpha", "2026-10-23 11:00"),
        _round_mu_row("charlie", "delta", "pinnacle", -110, -110, "charlie", "2026-10-23 10:05"),
        # soft book: stored but never alerted
        _round_mu_row("echo", "foxtrot", "fanduel", 100, -120, "echo", "2026-10-23 10:10"),
    ])
    killed = FakeWS(ss.KILLED_ROUND_MU_HEADERS, [
        _killed_row("golf", "hotel", "pinnacle", 150, -180, "golf", "pred_on gate"),
        _killed_row("india", "juliet", "pinnacle", 105, -125, "india", "sample_on<20"),
    ])
    return FakeSpreadsheet({ss.TAB_ROUND_MU: mu, ss.TAB_KILLED_ROUND_MU: killed})


FULL_SCORES = _scores(alpha=68, bravo=70, charlie=72, delta=70,
                      golf=71, hotel=69, india=70, juliet=70)


# ── kill capture ─────────────────────────────────────────────────────────────
def _matchups():
    def row(p1, p2, book, e1, e2, **extra):
        return {"Player 1": p1, "Player 2": p2, "Bookmaker": book, "Ties": "",
                "P1 Odds": 110, "P2 Odds": -130, "Fair_p1": -105, "Fair_p2": -115,
                "edge_p1": e1, "edge_p2": e2, "P1 Line": np.nan, "P2 Line": np.nan,
                **extra}
    return pd.DataFrame([
        row("a", "b", "pinnacle", 9.0, -9.0),                       # survives
        row("c", "d", "pinnacle", 8.0, -8.0),                       # sample too small
        row("e", "f", "betonline", 6.0, -6.0),                      # pred gate
        row("g", "h", "fanduel", 9.0, -9.0),                        # non-sharp book
        row("k", "l", "pinnacle", -4.0, 4.0),                       # edge below 5: not a kill
        row("i", "j", "pinnacle", 12.0, -12.0, **{"P1 Line": 3.5, "P2 Line": -3.5}),
    ])


PRED = {"a": 1.5, "b": 0.0, "c": 1.5, "d": 0.0, "e": 0.2, "f": 0.0,
        "g": 1.5, "h": 0.0, "k": 0.0, "l": 1.5, "i": 1.5, "j": 0.0}
SAMPLE = {name: 30.0 for name in PRED} | {"c": 10.0}


def test_killed_capture_records_reasons_and_leaves_outputs_unchanged():
    killed = []
    combined, sharp = rc.build_matchup_outputs(
        _matchups(), 2, PRED, SAMPLE, killed_out=killed)
    plain_combined, plain_sharp = rc.build_matchup_outputs(_matchups(), 2, PRED, SAMPLE)
    assert combined.equals(plain_combined) and sharp.equals(plain_sharp)

    frame = killed[0]
    reasons = dict(zip(frame["bet_on"], frame["kill_reason"]))
    # g (soft-book quote of a passing bet) and i (quarantined) are not kills
    assert reasons == {"c": "sample_on<20", "e": "pred_on gate"}
    assert (frame["edge_on"] > 5).all()


def test_capture_failure_does_not_affect_pricing_outputs():
    killed = []
    with patch.object(rc, "collect_killed_matchups", side_effect=RuntimeError("boom")):
        combined, sharp = rc.build_matchup_outputs(
            _matchups(), 2, PRED, SAMPLE, killed_out=killed)
    assert killed == []
    assert set(combined["bet_on"]) == {"a", "g", "l"}
    assert set(sharp["bet_on"]) == {"a", "l"}


def test_store_killed_dedups_per_pair_side_book_across_runs():
    killed = []
    rc.build_matchup_outputs(_matchups(), 2, PRED, SAMPLE, killed_out=killed)
    book = FakeSpreadsheet()
    first = ss.store_killed_round_matchups(killed[0], 2, "test", "100", spreadsheet=book)
    again = ss.store_killed_round_matchups(killed[0], 2, "test", "100", spreadsheet=book)
    assert first == 2 and again == 0
    moved = killed[0].copy()
    moved["P1 Odds"] = 130  # repriced quote of the same pair/side/book: first quote wins
    assert ss.store_killed_round_matchups(moved, 2, "test", "100", spreadsheet=book) == 0
    assert len(book.tabs[ss.TAB_KILLED_ROUND_MU].rows) == 2


# ── grading / aggregation ────────────────────────────────────────────────────
def test_grade_records_and_summary_units():
    recs = [
        _round_mu_row("alpha", "bravo", "pinnacle", 120, -140, "alpha", "t"),    # win +1.2
        _round_mu_row("charlie", "delta", "pinnacle", -150, 130, "charlie", "t"),  # loss -1.5
        _round_mu_row("echo", "foxtrot", "pinnacle", 100, -120, "echo", "t"),    # push
    ]
    results = _scores(alpha=68, bravo=70, charlie=72, delta=70, echo=69, foxtrot=69)
    graded = round_recap.grade_records(recs, results, 2)
    assert [g["result"] for g in graded] == ["win", "loss", "push"]
    assert (graded[0]["bet_score"], graded[0]["opp_score"]) == (68, 70)
    assert round_recap.summarize(graded) == (1, 1, 1, pytest.approx(-0.3))


def test_grade_records_marks_missing_scores_void():
    recs = [_round_mu_row("alpha", "ghost", "pinnacle", 120, -140, "alpha", "t")]
    graded = round_recap.grade_records(recs, _scores(alpha=68), 2)
    assert graded[0]["result"] == "void"
    assert round_recap.summarize(graded) == (0, 0, 0, 0.0)


def test_load_recommended_keeps_one_sharp_row_per_pairing_side():
    recs = round_recap.load_recommended("100", 2, _spreadsheet())
    assert [(r["player_1"], r["bookmaker"]) for r in recs] == [
        ("alpha", "pinnacle"), ("charlie", "pinnacle")]


# ── formatting ───────────────────────────────────────────────────────────────
def test_message_has_groups_per_reason_subtotals_and_escapes():
    sheet = _spreadsheet()
    recs = round_recap.grade_records(round_recap.load_recommended("100", 2, sheet),
                                     FULL_SCORES, 2)
    killed = round_recap.grade_records(
        sheet.worksheet(ss.TAB_KILLED_ROUND_MU).get_all_records(), FULL_SCORES, 2)
    text = "\n".join(round_recap.format_recap_messages("some_open & co", 2, recs, killed))
    assert "R2 Recap - Some Open &amp; Co" in text
    assert "Recommended (alerted): 1-1-0  +0.10u" in text
    assert "Killed edges >5% (hypothetical): 0-1-1  -1.00u" in text
    assert "pred_on gate: 0-1-0  -1.00u" in text
    assert "sample_on&lt;20: 0-0-1  +0.00u" in text
    assert "alpha v bravo | pinnacle +120 | edge 8.0% | 68 v 70" in text


def test_long_recap_is_split_into_telegram_sized_messages():
    bets = [{"bet_on": f"player {i}", "opponent": "x", "book": "pinnacle", "odds": 110,
             "edge": 6.0, "bet_score": 68, "opp_score": 70, "result": "win",
             "units": 1.1, "reason": "pred_on gate"} for i in range(80)]
    messages = round_recap.format_recap_messages("big", 2, bets, bets, max_chars=1000)
    assert len(messages) > 1 and all(len(m) <= 1000 for m in messages)


# ── send-once marker, incomplete skip, failure isolation ─────────────────────
def _run(sheet, fetch, sent, **kwargs):
    return round_recap.run_recap(
        "100", "test_open", 2, spreadsheet=sheet, api_key="k",
        fetch_scores=lambda rnd, key, expected_event_name=None: fetch,
        sender=lambda text, chat_id=None, required=False: sent.append(text) or True,
        **kwargs)


def test_recap_sends_once_and_marks():
    sheet, sent = _spreadsheet(), []
    assert _run(sheet, FULL_SCORES, sent) == "sent"
    assert _run(sheet, FULL_SCORES, sent) == "already_sent"
    assert len(sent) == 1
    log = sheet.worksheet(ss.TAB_ROUND_RECAP_LOG).get_all_records()
    assert [(r["event_id"], r["round"], r["n_recommended"], r["n_killed"]) for r in log] \
        == [("100", 2, 2, 2)]


def test_player_still_on_course_skips_without_marker_then_retry_succeeds():
    sheet, sent = _spreadsheet(), []
    partial = _scores(alpha=68, bravo=70, charlie=72, delta=70, golf=71, hotel=69,
                      india=70, live=["juliet"])
    assert _run(sheet, partial, sent) == "incomplete"
    assert _run(sheet, None, sent) == "incomplete"
    assert sent == [] and ss.TAB_ROUND_RECAP_LOG not in sheet.tabs
    assert _run(sheet, FULL_SCORES, sent) == "sent"


def test_terminal_wd_and_absent_players_are_graded_void_and_sent():
    sheet, sent = _spreadsheet(), []
    scores = _scores(alpha=68, bravo=70, charlie=72, golf=71, hotel=69,
                     terminal=["delta"])  # india/juliet absent from the feed entirely
    assert _run(sheet, scores, sent) == "sent"
    assert "no score" in sent[0]
    assert "Recommended (alerted): 1-0-0  +1.20u" in sent[0]


def test_dry_run_prints_and_neither_sends_nor_marks(capsys):
    sheet, sent = _spreadsheet(), []
    assert _run(sheet, FULL_SCORES, sent, dry_run=True) == "dry_run"
    assert sent == [] and ss.TAB_ROUND_RECAP_LOG not in sheet.tabs
    assert "R2 Recap" in capsys.readouterr().out


def test_failed_telegram_leaves_no_marker():
    sheet = _spreadsheet()

    def failing(text, chat_id=None, required=False):
        raise rc.TelegramDeliveryError("down")

    with pytest.raises(rc.TelegramDeliveryError):
        round_recap.run_recap("100", "t", 2, spreadsheet=sheet, api_key="k",
                              fetch_scores=lambda *a, **k: FULL_SCORES, sender=failing)
    assert ss.TAB_ROUND_RECAP_LOG not in sheet.tabs


def test_nothing_to_recap_marks_without_sending():
    sheet, sent = FakeSpreadsheet({ss.TAB_ROUND_MU: FakeWS(ss.ROUND_MU_HEADERS)}), []
    assert _run(sheet, FULL_SCORES, sent) == "nothing_to_recap"
    assert sent == []


def test_recap_and_persist_failures_are_swallowed_by_reprice_hooks():
    def boom(*args, **kwargs):
        raise RuntimeError("sheets down")

    assert reprice._send_prior_round_recap(
        event_id="100", tourney="t", round_num=2, recap_func=boom) is None
    assert reprice._send_prior_round_recap(
        event_id="100", tourney="t", round_num=0, recap_func=boom) is None
    frame = pd.DataFrame([{"Player 1": "a", "Player 2": "b", "edge_on": 6.0}])
    with patch.object(ss, "store_killed_round_matchups", side_effect=boom):
        assert reprice._persist_killed_edges(
            [frame], sim_round=2, tourney="t", event_id="100", spreadsheet=None) == 0


def test_first_chunk_failure_leaves_no_marker_and_later_chunk_failure_keeps_it():
    sheet = _spreadsheet()
    with patch.object(round_recap, "format_recap_messages", return_value=["one", "two"]):
        def fail_first(text, chat_id=None, required=False):
            raise rc.TelegramDeliveryError("down")

        with pytest.raises(rc.TelegramDeliveryError):
            round_recap.run_recap("100", "t", 2, spreadsheet=sheet, api_key="k",
                                  fetch_scores=lambda *a, **k: FULL_SCORES, sender=fail_first)
        assert ss.TAB_ROUND_RECAP_LOG not in sheet.tabs

        sent = []

        def fail_second(text, chat_id=None, required=False):
            if text == "two":
                raise rc.TelegramDeliveryError("down")
            sent.append(text)

        assert round_recap.run_recap(
            "100", "t", 2, spreadsheet=sheet, api_key="k",
            fetch_scores=lambda *a, **k: FULL_SCORES, sender=fail_second) == "sent"
        assert sent == ["one"] and len(sheet.worksheet(ss.TAB_ROUND_RECAP_LOG).rows) == 1
        assert round_recap.run_recap(
            "100", "t", 2, spreadsheet=sheet, api_key="k",
            fetch_scores=lambda *a, **k: FULL_SCORES, sender=fail_second) == "already_sent"


def test_killed_edge_later_alerted_is_not_double_counted():
    sheet, sent = _spreadsheet(), []
    killed = _killed_row("bravo", "alpha", "pinnacle", -140, 120, "alpha", "pred_on gate")
    sheet.tabs[ss.TAB_KILLED_ROUND_MU].rows.append(
        [killed.get(h, "") for h in ss.KILLED_ROUND_MU_HEADERS])
    assert _run(sheet, FULL_SCORES, sent) == "sent"
    log = sheet.worksheet(ss.TAB_ROUND_RECAP_LOG).get_all_records()
    assert log[0]["n_killed"] == 2  # alpha/bravo is already counted as recommended


def test_unverified_betcris_rows_are_not_recommended():
    sheet = _spreadsheet()
    legacy = _round_mu_row("kilo", "lima", "betcris", 110, -130, "kilo", "2026-10-23 12:00")
    verified = _round_mu_row("mike", "november", "betcris", 110, -130, "mike", "2026-10-23 12:01")
    verified["line_verified"] = True
    for row in (legacy, verified):
        sheet.tabs[ss.TAB_ROUND_MU].rows.append([row.get(h, "") for h in ss.ROUND_MU_HEADERS])
    players = {r["player_1"] for r in round_recap.load_recommended("100", 2, sheet)}
    assert "kilo" not in players and "mike" in players


# ── official-score fields, timeout, ETR guard ────────────────────────────────
def _live_feed():
    rows = [{"player_name": f"p{i}", "position": "T1", "thru": 18, "round": -1 - (i % 3),
             "score": 140 + i, "event_name": "Test Open"} for i in range(12)]
    rows += [
        {"player_name": "still", "position": "T9", "thru": 11, "round": -2, "score": 150,
         "event_name": "Test Open"},
        {"player_name": "wd guy", "position": "WD", "thru": 5, "round": 2, "score": 160,
         "event_name": "Test Open"},
        {"player_name": "cutguy", "position": "CUT", "thru": np.nan, "round": np.nan,
         "score": np.nan, "event_name": "Test Open"},
    ]
    return pd.DataFrame(rows)


def test_fetch_round_scores_uses_round_and_thru_not_total_score():
    with patch("api_utils.fetch_live_stats", return_value=_live_feed()) as fetch:
        frame = round_recap.fetch_round_scores(2, "k", expected_event_name="test open")
    assert fetch.call_args.kwargs["timeout"] == round_recap.DATAGOLF_TIMEOUT == 20
    by_name = frame.set_index("player_name")
    assert by_name.loc["p0", "round_2"] == -1  # 'round', not the 2-round 'score' total
    assert by_name.loc["still", "status"] == "live" and np.isnan(by_name.loc["still", "round_2"])
    assert by_name.loc["wd guy", "status"] == "terminal"
    assert by_name.loc["cutguy", "status"] == "terminal"


def test_fetch_round_scores_rejects_unusable_feeds():
    feed = _live_feed()
    with patch("api_utils.fetch_live_stats", return_value=feed.drop(columns=["thru"])):
        assert round_recap.fetch_round_scores(2, "k") is None
    with patch("api_utils.fetch_live_stats", return_value=feed):
        assert round_recap.fetch_round_scores(2, "k", expected_event_name="Other Open") is None
    with patch("api_utils.fetch_live_stats", return_value=feed.head(4)):
        assert round_recap.fetch_round_scores(2, "k") is None
    with patch("api_utils.fetch_live_stats", return_value=None):
        assert round_recap.fetch_round_scores(2, "k") is None


def test_fetch_live_stats_passes_timeout_and_defaults_to_none():
    import api_utils

    class Resp:
        status_code = 500
        text = "x"

    with patch.object(api_utils.requests, "get", return_value=Resp()) as get:
        assert api_utils.fetch_live_stats(2, "k", timeout=20) is None
        assert get.call_args.kwargs["timeout"] == 20
        api_utils.fetch_live_stats(2, "k")
        assert get.call_args.kwargs["timeout"] is None


def test_reprice_job_has_timeout_minutes():
    path = Path(__file__).parent / ".github" / "workflows" / "reprice.yml"
    assert "timeout-minutes:" in path.read_text(encoding="utf-8")


def test_recap_marker_write_honours_etr_only_mode(monkeypatch):
    monkeypatch.setenv("ETR_ONLY_MODE", "1")
    sheet = FakeSpreadsheet({ss.TAB_ROUND_MU: FakeWS(ss.ROUND_MU_HEADERS)})
    with pytest.raises(RuntimeError, match="ETR-only"):
        round_recap.run_recap("100", "t", 2, spreadsheet=sheet)
    assert ss.TAB_ROUND_RECAP_LOG not in sheet.tabs


# ── persist only after successful delivery ───────────────────────────────────
def _delivery(after, alert):
    new_mu = pd.DataFrame([{
        "Player 1": "alpha", "Player 2": "bravo", "Bookmaker": "pinnacle",
        "bet_on": "alpha", "edge_on": 8.2, "P1 Odds": 120, "P2 Odds": -140,
    }])
    return reprice._deliver_then_store_matchups(
        new_mu, set(), sim_round=2, tourney="t", event_id="100", dg_id_lookup={},
        spreadsheet=None, rc_module=rc, store_func=lambda *a, **k: None,
        alert_func=alert, after_delivery=after)


def test_after_delivery_runs_only_when_delivery_succeeds():
    calls = []
    _delivery(lambda: calls.append("persist"), lambda rows, *a, **k: len(rows))
    assert calls == ["persist"]

    def failing_alert(rows, *a, **k):
        raise rc.TelegramDeliveryError("down")

    calls.clear()
    with pytest.raises(rc.TelegramDeliveryError):
        _delivery(lambda: calls.append("persist"), failing_alert)
    assert calls == []

    reprice._deliver_then_store_matchups(
        pd.DataFrame(), set(), sim_round=2, tourney="t", event_id="100", dg_id_lookup={},
        spreadsheet=None, rc_module=rc, store_func=None,
        after_delivery=lambda: calls.append("persist"))
    assert calls == ["persist"]


def test_rejected_health_gate_never_reaches_persist():
    calls = []

    def rejecting_health():
        raise RuntimeError("bundle rejected")

    new_mu = pd.DataFrame([{
        "Player 1": "alpha", "Player 2": "bravo", "Bookmaker": "pinnacle",
        "bet_on": "alpha", "edge_on": 8.2, "P1 Odds": 120, "P2 Odds": -140,
    }])
    with pytest.raises(RuntimeError):
        reprice._deliver_then_store_matchups(
            new_mu, set(), sim_round=2, tourney="t", event_id="100", dg_id_lookup={},
            spreadsheet=None, rc_module=rc, store_func=lambda *a, **k: None,
            alert_func=lambda rows, *a, **k: len(rows), health_check=rejecting_health,
            after_delivery=lambda: calls.append("persist"))
    assert calls == []


# ── recap attempted on early-exit paths of main() ────────────────────────────
@pytest.mark.parametrize("path", ["stand_down", "not_ready"])
def test_main_attempts_recap_on_early_exit_paths(path, monkeypatch):
    import types
    cfg = {"tourney": "test_open", "event_id": "100", "round_num": 2}
    calls = []
    monkeypatch.setattr(sys, "argv", ["reprice.py", "--check-ready"])
    monkeypatch.setattr(reprice, "_send_prior_round_recap", lambda **kw: calls.append(kw))
    monkeypatch.setattr(reprice, "_write_readiness_output", lambda ready: None)
    monkeypatch.setattr(reprice, "_setup_env", lambda: str(Path(__file__).parent))
    own = types.SimpleNamespace(
        legacy_mode=lambda: path != "stand_down",
        resolve_ownership=lambda *a, **k: types.SimpleNamespace(
            status="owned", stand_down=True, detail="", log_line=lambda: "owned"))
    monkeypatch.setitem(sys.modules, "sim_fairs_ownership", own)
    monkeypatch.setattr(reprice, "reprice_not_ready_reason", lambda *a, **k: "no sim")
    with patch("sheet_config.load_config", return_value=cfg):
        assert reprice.main() == 0
    assert calls == [{"event_id": "100", "tourney": "test_open", "round_num": 2}]
