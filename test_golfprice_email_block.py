"""Unit tests for golfprice_email_block (no network: a fake S3 client stands in for R2)."""

import datetime as dt
import io
import json
from pathlib import Path

import golfprice_email_block as gb

NOW = dt.datetime(2026, 10, 12, 15, 0, tzinfo=dt.timezone.utc)


def _doc(generated_at="2026-10-12T14:40:00Z"):
    arm = lambda a, label, rmse, d, ll, lc: {"arm": a, "label": label, "rmse": rmse, "rmse_vs_champion": d, "logloss_vs_champion_mean": ll, "logloss_vs_close_mean": lc}  # noqa: E731
    return {
        "schema": "golfprice.scorecard.v1",
        "generated_at": generated_at,
        "headlines": ["Bank of Utah (PGA): challenger 1.827 vs 1.848 <b>better</b>.", "Forward record: 1 of the 40-event futility check."],
        "latest_week": {"event_start": "2026-10-01", "event_uids": ["euro:4:2026-10-01", "pga:2:2026-10-01"]},
        "events": [
            {"event_uid": "euro:4:2026-10-01", "name": "Euro Open", "tour": "euro", "event_start": "2026-10-01", "arms": []},
            {"event_uid": "pga:2:2026-10-01", "name": "Bank of Utah", "tour": "pga", "event_start": "2026-10-01",
             "arms": [arm("champion", "Production model (champion)", 1.848, None, None, 0.001), arm("challenger", "Challenger v2.2", 1.827, -0.0202, 0.00106, -0.0006)]},
        ],
        "forward": {"events_counted": 1, "futility_check_at": 40},
    }


class FakeClient:
    def __init__(self, doc=None, exc=None, raw=None):
        self.doc, self.exc, self.raw, self.calls = doc, exc, raw, []

    def get_object(self, Bucket, Key):  # noqa: N803 - boto3 signature
        self.calls.append((Bucket, Key))
        if self.exc:
            raise self.exc
        body = self.raw if self.raw is not None else json.dumps(self.doc).encode()
        return {"Body": io.BytesIO(body)}


def _logs():
    lines = []
    return lines, lines.append


def test_reads_the_scorecard_key_from_the_dashboard_bucket():
    c = FakeClient(_doc())
    gb.build_block(client=c, now=NOW)
    assert c.calls == [("golf-model-dashboard-data", "golfprice/scorecard/latest.json")]


def test_renders_headlines_and_table_with_escaping_and_pga_preferred():
    block = gb.build_block(client=FakeClient(_doc()), now=NOW)
    assert "golfprice model scorecard" in block and "Challenger v2.2" in block and "Bank of Utah" in block and "Euro Open" not in block
    assert "&lt;b&gt;better&lt;/b&gt;" in block and "<b>better</b>" not in block        # text from the file is escaped
    assert "-0.020" in block and "baseline" in block and "1 of 40 events" in block
    assert "#28a745" in block                                                           # negative delta is green


def test_stale_file_skips_with_one_log_line():
    lines, log = _logs()
    assert gb.build_block(client=FakeClient(_doc("2026-10-03T14:00:00Z")), now=NOW, log=log) == ""     # 9.04 days old
    assert len(lines) == 1 and "9.0 days" in lines[0]
    lines, log = _logs()
    assert gb.build_block(client=FakeClient(_doc("2026-10-05T16:00:00Z")), now=NOW, log=log) != "" and lines == []   # 6.96 days: fresh


def test_every_failure_mode_yields_empty_block_and_one_log_line_never_raises():
    cases = [
        FakeClient(exc=RuntimeError("NoSuchKey")),
        FakeClient(raw=b"not json"),
        FakeClient(doc={"schema": "other", "generated_at": "2026-10-12T14:00:00Z"}),
        FakeClient(doc=[1, 2]),
        FakeClient(doc={"schema": gb.SCHEMA}),                                      # no generated_at
        FakeClient(doc={**_doc(), "generated_at": "garbage"}),
    ]
    for c in cases:
        lines, log = _logs()
        assert gb.build_block(client=c, now=NOW, log=log) == ""
        assert len(lines) == 1 and lines[0].startswith("skipped")


def test_malformed_but_valid_schema_never_raises():
    weird = _doc()
    weird["events"] = [None, 5, {"event_uid": "pga:2:2026-10-01", "arms": [None, {"arm": "x", "rmse": "n/a"}]}]
    weird["headlines"] = [None, 3, "ok"]
    weird["forward"] = None
    lines, log = _logs()
    out = gb.build_block(client=FakeClient(weird), now=NOW, log=log)
    assert isinstance(out, str) and len(lines) <= 1


def test_missing_credentials_are_contained_and_not_printed(monkeypatch):
    for n in ("CF_ACCOUNT_ID", "DASHBOARD_R2_ACCESS_KEY_ID", "DASHBOARD_R2_SECRET_ACCESS_KEY", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(n, raising=False)
    lines, log = _logs()
    assert gb.build_block(now=NOW, log=log) == "" and len(lines) == 1 and lines[0] == "skipped: KeyError"


def test_prepare_env_sets_variable_only_with_a_block():
    env = {}
    lines, log = _logs()
    assert gb.prepare_env(environ=env, client=FakeClient(_doc()), now=NOW, log=log) is True
    assert "golfprice model scorecard" in env[gb.ENV_VAR]
    env2 = {}
    assert gb.prepare_env(environ=env2, client=FakeClient(exc=RuntimeError("x")), now=NOW, log=lambda m: None) is False and env2 == {}
    assert gb.prepare_env(environ=None, client=FakeClient(exc=RuntimeError("x")), now=NOW, log=lambda m: (_ for _ in ()).throw(OSError("log broke"))) is False   # even a broken logger is contained


def test_inject_places_block_before_the_footer_and_is_a_noop_without_a_block():
    page = '<html><body><div><p style="a">row</p><p style="color:#999;">Generated by Golf Sim Bet Tracker | 3 total bets graded</p></div></body></html>'
    out = gb.inject(page, "<BLOCK/>")
    assert out.index("<BLOCK/>") < out.index("Generated by Golf Sim Bet Tracker") and out.index("<BLOCK/>") > out.index("row")
    assert out.count("Generated by Golf Sim Bet Tracker") == 1
    assert gb.inject(page, "") == page
    assert gb.inject("<body>x</body>", "<B/>") == "<body>x<B/></body>"
    assert gb.inject("plain", "<B/>") == "plain<B/>"
    assert gb.with_block(page, environ={}) == page and "<BLOCK/>" in gb.with_block(page, environ={gb.ENV_VAR: "<BLOCK/>"})
    assert gb.with_block(None, environ={gb.ENV_VAR: "<B/>"}) is None             # contained, input returned unchanged


def test_footer_marker_matches_the_real_results_email():
    src = (Path(__file__).parent / "grade_bets.py").read_text(encoding="utf-8")
    assert gb.FOOTER_MARKER in src


def test_hooks_are_wrapped_so_they_cannot_affect_grading_or_the_email():
    root = Path(__file__).parent
    mg = (root / "monday_grading.py").read_text(encoding="utf-8")
    i = mg.index("golfprice_email_block.prepare_env()")
    assert "try:" in mg[i - 200:i] and "except Exception" in mg[i:i + 200] and mg.index("grade_bets.py") > 0
    assert i < mg.index('cmd = [python, "grade_bets.py"]')                    # prepared before the grading subprocess starts
    gbt = (root / "grade_bets.py").read_text(encoding="utf-8")
    j = gbt.index("from golfprice_email_block import with_block")
    assert "try:" in gbt[j - 150:j] and "except Exception" in gbt[j:j + 150] and "if not filter_label:" in gbt[j - 300:j]
