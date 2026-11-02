"""Offline tests for the Monday grading matchup recap (round_recap.py).
No network, Sheets, or Telegram."""

from pathlib import Path
from unittest.mock import patch

import gspread
import numpy as np
import pandas as pd
import pytest

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


def test_monday_grading_runs_r4_recap_without_failing_the_job():
    root = Path(__file__).parent
    source = (root / "monday_grading.py").read_text(encoding="utf-8")
    block = source[source.index("Step 4b"):source.index("Step 5")]
    assert "from round_recap import run_recap" in block
    assert "run_recap(event_id, event_name, 4" in block
    assert "except Exception" in block
    workflow = (root / ".github" / "workflows" / "monday-grading.yml").read_text(encoding="utf-8")
    assert "runs-on: ubuntu-latest" in workflow
    assert "TELEGRAM_ROUND_BETS_CHAT_ID: ${{ secrets.TELEGRAM_ROUND_BETS_CHAT_ID }}" in workflow


def test_recap_marker_write_honours_etr_only_mode(monkeypatch):
    monkeypatch.setenv("ETR_ONLY_MODE", "1")
    sheet = FakeSpreadsheet({ss.TAB_ROUND_MU: FakeWS(ss.ROUND_MU_HEADERS)})
    with pytest.raises(RuntimeError, match="ETR-only"):
        round_recap.run_recap("100", "t", 2, spreadsheet=sheet)
    assert ss.TAB_ROUND_RECAP_LOG not in sheet.tabs
