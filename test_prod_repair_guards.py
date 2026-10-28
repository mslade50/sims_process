"""Tests for the 2026-10-18 production repair: ownership stand-down, telegram state,
kalshi preflight skip, mkt_regress / final-pass guards. Secret-free, no network."""
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import mkt_regress_guards as g
import sim_fairs_ownership as own
import telegram_policy

NOW = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)


def _root(tmp_path, marker=None):
    """A directory with no .git so read_marker falls back to the local file."""
    if marker is not None:
        (tmp_path / own.MARKER).write_text(json.dumps(marker), encoding="utf-8")
    return tmp_path


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("SIM_FAIRS_FORCE_PRODUCTION", "SIM_FAIRS_OWNERSHIP_LEGACY",
              "SIM_FAIRS_OWNERSHIP_RESUME_ON_STALE", "SIM_FAIRS_EXTERNAL_MAX_AGE_HOURS",
              "MKT_REGRESS_ALLOW_STALE", "NEW_SIM_ALLOW_UNVERIFIED_FINAL",
              "KALSHI_PREFLIGHT_STRICT"):
        monkeypatch.delenv(k, raising=False)


# ---- SC-02 -----------------------------------------------------------------
def test_empty_or_corrupt_telegram_state_is_no_state(tmp_path, capsys):
    p = tmp_path / "state.json"
    p.write_text("", encoding="utf-8")
    assert telegram_policy.load_state(p) == {}
    p.write_text("{not json", encoding="utf-8")
    assert telegram_policy.load_state(p) == {}
    p.write_text("[1, 2]", encoding="utf-8")
    assert telegram_policy.load_state(p) == {}
    p.write_text('{"sent_at": 5}', encoding="utf-8")
    assert telegram_policy.load_state(p) == {"sent_at": 5}
    assert telegram_policy.should_alert({}, 200000)


def test_main_survives_empty_state_object(monkeypatch, capsys):
    import subprocess
    import sys

    def fake_run(cmd, **kw):
        if "get" in cmd:  # wrangler writes an EMPTY state file, exit 0
            Path([a for a in cmd if a.startswith("--file=")][0][7:]).write_text("", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "test")
    monkeypatch.delenv("TELEGRAM_DISABLED", raising=False)
    monkeypatch.setattr(subprocess, "run", fake_run)
    sent = []

    class _Resp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(telegram_policy.urllib.request, "urlopen", lambda r, timeout=0: sent.append(r) or _Resp())
    monkeypatch.setattr(sys, "argv", ["telegram_policy.py", "--key", "k", "--message", "m"])
    telegram_policy.main()  # must not raise JSONDecodeError
    assert len(sent) == 1


# ---- SC-01/03/08 -----------------------------------------------------------
def test_no_marker_means_production_owns(tmp_path):
    d = own.resolve_ownership("527", root=_root(tmp_path), now=NOW)
    assert (d.stand_down, d.status) == (False, "none")


def test_fresh_marker_for_same_event_stands_down(tmp_path):
    root = _root(tmp_path, {"owner": "golfprice", "event_id": "527", "since": "2026-10-17 00:00:00 UTC"})
    d = own.resolve_ownership(527, root=root, now=NOW)
    assert (d.stand_down, d.status, d.owner) == (True, "owned", "golfprice")


def test_event_mismatch_is_loud_but_still_stands_down(tmp_path):
    root = _root(tmp_path, {"owner": "golfprice", "event_id": "527", "since": "2026-10-17 00:00:00 UTC"})
    d = own.resolve_ownership("528", root=root, now=NOW)
    assert (d.stand_down, d.status) == (True, "stale_event")
    assert "528" in d.detail and "STALE MARKER" in d.detail


def test_expired_marker_flagged_and_resume_flag_resumes(tmp_path, monkeypatch):
    root = _root(tmp_path, {"owner": "golfprice", "event_id": "527", "since": "2026-10-01 00:00:00 UTC"})
    d = own.resolve_ownership("527", root=root, now=NOW)
    assert (d.stand_down, d.status) == (True, "expired")
    monkeypatch.setenv("SIM_FAIRS_OWNERSHIP_RESUME_ON_STALE", "1")
    d = own.resolve_ownership("527", root=root, now=NOW)
    assert (d.stand_down, d.status) == (False, "expired")


def test_explicit_expires_at_and_event_ids_list(tmp_path):
    root = _root(tmp_path, {"owner": "golfprice", "event_ids": [527, 528],
                            "expires_at": "2026-10-20T00:00:00Z"})
    assert own.resolve_ownership("528", root=root, now=NOW).status == "owned"
    assert own.resolve_ownership("528", root=root, now=NOW + timedelta(days=3)).status == "expired"


def test_corrupt_marker_fails_closed(tmp_path):
    (tmp_path / own.MARKER).write_text("{oops", encoding="utf-8")
    d = own.resolve_ownership("527", root=tmp_path, now=NOW)
    assert (d.stand_down, d.status) == (True, "unreadable")


def test_git_error_fails_closed_not_open(tmp_path):
    (tmp_path / ".git").mkdir()  # looks like a repo, but git cannot read it and no local marker
    d = own.resolve_ownership("527", root=tmp_path, now=NOW)
    assert (d.stand_down, d.status) == (True, "unreadable")


def test_force_production_and_legacy_switches(tmp_path, monkeypatch):
    root = _root(tmp_path, {"owner": "golfprice", "event_id": "527", "since": "2026-10-17 00:00:00 UTC"})
    monkeypatch.setenv("SIM_FAIRS_FORCE_PRODUCTION", "1")
    assert own.resolve_ownership("527", root=root, now=NOW).stand_down is False
    monkeypatch.delenv("SIM_FAIRS_FORCE_PRODUCTION")
    monkeypatch.setenv("SIM_FAIRS_OWNERSHIP_LEGACY", "1")
    assert own.legacy_mode() is True


def test_reprice_stands_down_cleanly_when_golfprice_owns(tmp_path, monkeypatch, capsys):
    import reprice
    out = tmp_path / "gh_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.setattr(reprice, "_setup_env", lambda: str(_root(tmp_path, {
        "owner": "golfprice", "event_id": "527",
        "since": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")})))
    monkeypatch.setitem(__import__("sys").modules, "sheet_config", type("M", (), {
        "load_config": staticmethod(lambda: {"tourney": "baycurrent", "event_id": "527", "round_num": 3})}))
    monkeypatch.setattr("sys.argv", ["reprice.py", "--check-ready"])
    assert reprice.main() == 0
    assert "ready=false" in out.read_text()
    assert "legacy reprice stood down" in capsys.readouterr().out


# ---- SC-07 -----------------------------------------------------------------
def test_kalshi_preflight_skips_only_when_config_is_stale():
    import kalshi_preflight as kp
    assert "off week" in kp.stale_config_reason([527], [])
    assert "not this week's" in kp.stale_config_reason([527], [{"event_id": 528}])
    assert kp.stale_config_reason([528], [{"event_id": 528}]) is None   # real event: stay loud
    assert kp.stale_config_reason([527], None) is None                  # lookup failed: stay loud


def test_kalshi_preflight_strict_flag(monkeypatch):
    import kalshi_preflight as kp
    monkeypatch.setenv("KALSHI_PREFLIGHT_STRICT", "1")
    assert kp.stale_config_reason([527], []) is None


# ---- SC-05/06 --------------------------------------------------------------
def test_stale_inputs_fail_loudly(tmp_path):
    f = tmp_path / "betonline_odds_with_my_odds_tu.csv"
    f.write_text("x")
    old = time.time() - 80 * 86400
    os.utime(f, (old, old))
    with pytest.raises(g.StaleInputError, match="80\\.0 days old"):
        g.check_inputs_fresh([("matchup edges", str(f))])
    g.check_inputs_fresh([("matchup edges", str(f))], now=old + 3600)  # fresh relative to now


def test_stale_override_flag(tmp_path, monkeypatch):
    f = tmp_path / "a.csv"
    f.write_text("x")
    old = time.time() - 80 * 86400
    os.utime(f, (old, old))
    monkeypatch.setenv("MKT_REGRESS_ALLOW_STALE", "1")
    g.check_inputs_fresh([("a", str(f))])


def test_odds_coverage_guard():
    g.check_odds_coverage(100, 95)
    with pytest.raises(g.StaleInputError, match="wrong event"):
        g.check_odds_coverage(100, 30)


def test_final_pass_requires_matching_sidecar(tmp_path):
    final = tmp_path / "final_predictions_x.csv"
    skill = tmp_path / "init_sim_skill_x.csv"
    final.write_text("player_name,my_pred\na,1\n")
    skill.write_text("player_name,pred\na,1\n")
    with pytest.raises(g.StaleInputError, match="no lineage sidecar"):
        g.verify_final_meta(final, tourney="x", event_ids=[1], init_skill_path=skill)
    g.write_final_meta(final, tourney="x", event_ids=[1], n_players=1, input_paths=[("init_sim_skill", skill)])
    assert g.verify_final_meta(final, tourney="x", event_ids=[1], init_skill_path=skill)["tourney"] == "x"
    with pytest.raises(g.StaleInputError, match="event_ids"):
        g.verify_final_meta(final, tourney="x", event_ids=[2], init_skill_path=skill)
    skill.write_text("player_name,pred\na,2\n")
    with pytest.raises(g.StaleInputError, match="init_sim_skill differs"):
        g.verify_final_meta(final, tourney="x", event_ids=[1], init_skill_path=skill)


def test_final_pass_unverified_switch(tmp_path, monkeypatch):
    monkeypatch.setenv("NEW_SIM_ALLOW_UNVERIFIED_FINAL", "1")
    assert g.verify_final_meta(tmp_path / "f.csv", tourney="x", event_ids=[1], init_skill_path="n") is None


def test_hung_git_times_out_and_fails_closed(tmp_path, monkeypatch):
    import subprocess
    (tmp_path / ".git").mkdir()
    seen = []

    def hung(cmd, **kw):
        seen.append(kw.get("timeout"))
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))

    monkeypatch.setattr(own.subprocess, "run", hung)
    d = own.resolve_ownership("527", root=tmp_path, now=NOW)
    assert seen and all(t == 60 for t in seen)
    assert (d.stand_down, d.status) == (True, "unreadable")
