"""Tests for TEMPORARY legacy ETR-only mode (owner decision 2026-10-19).

With ETR_ONLY_MODE=1 every production side-effect path must raise BEFORE any
network / git / ownership / sheet call.  With the flag unset the guarded
functions must be reached exactly as before.  No real network, git, SMTP or
Sheets call is ever made: every primitive is replaced by a recorder that raises
a BaseException (so ``except Exception`` blocks cannot swallow it).
"""
from __future__ import annotations

import ast
import importlib
import inspect
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

os.environ.setdefault("COEFFS_FROM_CACHE", "1")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import etr_only_mode  # noqa: E402

MSG = re.compile(r"^ETR-only mode: .+ is disabled \(temporary legacy ETR mode, owner 2026-10-19\)$")

# (file, qualified name). Every one of these must refuse first.
GUARDED = [
    # emails
    ("new_sim.py", "send_tournament_email"), ("new_sim.py", "send_sportsbook_priority_email"),
    ("new_sim.py", "send_exchange_email"), ("new_sim.py", "send_outrights_email"),
    ("round_sim.py", "send_round_sim_email"), ("live_stats_engine.py", "send_summary_email"),
    ("grade_bets.py", "send_results_email"), ("grade_bets.py", "send_filtered_results_email"),
    ("sg_diagnostic.py", "send_diagnostic_email"),
    # telegram
    ("telegram_policy.py", "main"), ("reprice_core.py", "send_telegram"),
    ("round_sim.py", "_send_telegram"), ("round_sim.py", "_send_reprice_alert"),
    ("clv_alert.py", "send_telegram"), ("maker_alerts.py", "send_telegram"),
    ("kalshi_preflight.py", "_telegram"),
    # sim_fairs / board
    ("new_sim.py", "_publish_sim_fairs_required"), ("publish_sim_fairs.py", "publish"),
    ("publish_sim_fairs.py", "_git_push"), ("publish_sim_fairs.py", "_dispatch_board_build"),
    ("publish_sim_fairs.py", "_upload_release_asset"),
    ("publish_sim_fairs.py", "_upload_full_tape_release"),
    ("publish_sim_fairs.py", "_upload_matchup_tape_release"),
    ("publish_sim_fairs.py", "_upload_release_tape_family"),
    ("push_odds_screen.py", "main"), ("push_odds_screen.py", "_upload_json"),
    ("push_odds_screen.py", "_upload_atomic_payload_bundle"),
    ("push_maker_state.py", "main"), ("kalshi_exec/push_proposals.py", "main"),
    # dashboard data
    ("push_dashboard_data.py", "copy_files"), ("push_dashboard_data.py", "git_push"),
    ("push_dashboard_data.py", "publish_cloudflare"), ("push_dashboard_data.py", "main"),
    # bet storage
    ("sheets_storage.py", "store_tournament_matchups"), ("sheets_storage.py", "store_finish_positions"),
    ("sheets_storage.py", "store_round_matchups"), ("sheets_storage.py", "store_round_3balls"),
    ("sheets_storage.py", "store_score_edges"), ("sheets_storage.py", "_append_to_ledger"),
    ("sheets_storage.py", "update_ledger_grades"), ("sheets_storage.py", "update_ledger_clv"),
    # exchange orders / quotes
    ("kalshi_client.py", "KalshiClient.place_order"), ("kalshi_client.py", "KalshiClient.cancel_order"),
    ("kalshi_client.py", "KalshiClient.cancel_all_orders"), ("kalshi_trader.py", "submit_orders"),
    ("kalshi_maker.py", "post_limit"), ("kalshi_maker.py", "cancel_order"),
    ("kalshi_maker.py", "cancel_all"), ("kalshi_maker.py", "reconcile_and_post"),
    ("novig_client.py", "NovigClient.place_order"), ("novig_client.py", "NovigClient.place_orders"),
    ("novig_client.py", "NovigClient.cancel_order"), ("novig_client.py", "NovigClient.cancel_orders"),
    ("novig_client.py", "NovigClient.cancel_all_orders"),
    ("novig_client.py", "NovigClient.cancel_orders_by_event"),
    ("novig_client.py", "NovigClient.cancel_orders_by_market"),
    # shot archive
    ("hole_baselines.py", "main"),
]


def _find(tree, qualname):
    nodes, node = tree.body, None
    for part in qualname.split("."):
        node = next(n for n in nodes
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == part)
        nodes = node.body
    return node


# --------------------------------------------------------------------------
# module basics
# --------------------------------------------------------------------------
def test_etr_only_flag_parsing(monkeypatch):
    monkeypatch.delenv("ETR_ONLY_MODE", raising=False)
    assert etr_only_mode.etr_only() is False
    for val in ("1", "true", "YES", " 1 "):
        monkeypatch.setenv("ETR_ONLY_MODE", val)
        assert etr_only_mode.etr_only() is True
    for val in ("", "0", "false", "no"):
        monkeypatch.setenv("ETR_ONLY_MODE", val)
        assert etr_only_mode.etr_only() is False


def test_refuse_message_and_banner(monkeypatch, capsys):
    monkeypatch.setenv("ETR_ONLY_MODE", "1")
    with pytest.raises(RuntimeError) as exc:
        etr_only_mode.refuse("thing")
    assert str(exc.value) == ("ETR-only mode: thing is disabled "
                              "(temporary legacy ETR mode, owner 2026-10-19)")
    assert etr_only_mode.BANNER == ("TEMPORARY LEGACY ETR-ONLY MODE (until post-season ETR refactor) -- "
                                    "owner decision 2026-10-19 -- produces ETR inputs only; NOT production")
    etr_only_mode.print_banner()
    assert etr_only_mode.BANNER in capsys.readouterr().out
    monkeypatch.delenv("ETR_ONLY_MODE")
    etr_only_mode.refuse("thing")  # no-op
    etr_only_mode.print_banner()
    assert capsys.readouterr().out == ""


def test_refuse_not_overridable_by_force_production(monkeypatch):
    monkeypatch.setenv("ETR_ONLY_MODE", "1")
    monkeypatch.setenv("SIM_FAIRS_FORCE_PRODUCTION", "1")
    with pytest.raises(RuntimeError):
        etr_only_mode.refuse("sim_fairs publish")


# --------------------------------------------------------------------------
# structural: the guard is the FIRST statement of every guarded function
# --------------------------------------------------------------------------
@pytest.mark.parametrize("fname,qualname", GUARDED)
def test_guard_is_first_statement(fname, qualname):
    tree = ast.parse((ROOT / fname).read_text(encoding="utf-8"))
    node = _find(tree, qualname)
    body = list(node.body)
    if (isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]  # docstring
    imp, call = body[0], body[1]
    assert isinstance(imp, ast.ImportFrom) and imp.module == "etr_only_mode"
    assert isinstance(call, ast.Expr) and isinstance(call.value, ast.Call)
    assert getattr(call.value.func, "id", None) == "_etr_refuse"


def test_non_get_requests_refused_in_exchange_transports():
    for fname, needle in [("kalshi_client.py", 'method.upper() != "GET"'),
                          ("kalshi_maker.py", 'str(method).upper() != "GET"'),
                          ("novig_client.py", "Novig DELETE request")]:
        src = (ROOT / fname).read_text(encoding="utf-8")
        assert needle in src and "_etr_refuse" in src, fname


def _new_sim_src():
    return (ROOT / "new_sim.py").read_text(encoding="utf-8")


def test_new_sim_final_pass_refused_before_everything_else():
    src = _new_sim_src()
    parse = src.index("args = _parser.parse_args()")
    refuse_at = src.index('_etr_refuse("new_sim.py --final-pass/--price-only/--reprice")')
    assert refuse_at > parse
    between = src[parse:refuse_at]
    assert "requests." not in between and "subprocess" not in between and "get_spreadsheet" not in between
    # defence in depth on the late blocks; the dashboard refusal sits outside the try
    storage = src.index('_etr_refuse("final-pass bet storage block")')
    assert storage < re.search(r"from sheets_storage import \(\s+is_valid_run_time", src).start()
    dash = src.index('_etr_refuse("final-pass dashboard data push")')
    assert dash < re.search(r"try:\s+from push_dashboard_data import copy_files, git_push", src).start()


def test_new_sim_final_pass_refuses_in_subprocess():
    """Real process: `new_sim.py --final-pass` must die on the guard under the flag."""
    env = dict(os.environ, ETR_ONLY_MODE="1", COEFFS_FROM_CACHE="1", PYTHONUTF8="1")
    for k in ("SIM_FAIRS_FORCE_PRODUCTION", "DATAGOLF_API_KEY"):
        env.pop(k, None)
    proc = subprocess.run([sys.executable, str(ROOT / "new_sim.py"), "--final-pass"],
                          cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode != 0
    assert "TEMPORARY LEGACY ETR-ONLY MODE" in proc.stdout
    assert "ETR-only mode: new_sim.py --final-pass" in proc.stderr
    assert "is disabled (temporary legacy ETR mode, owner 2026-10-19)" in proc.stderr


def test_new_sim_calibration_pass_not_refused_by_guard():
    """The kept path: --calibration-pass is not matched by the final-pass guard."""
    src = _new_sim_src()
    guard = re.search(r"if args\.final_pass or args\.price_only or args\.reprice:\s+_etr_refuse", src)
    assert guard and "calibration" not in guard.group(0)


# --------------------------------------------------------------------------
# behavioural: raises under the flag with ZERO primitive calls; reached when unset
# --------------------------------------------------------------------------
class _Touched(BaseException):
    """A network/git/smtp/sheet primitive was reached."""


@pytest.fixture
def primitives(monkeypatch, tmp_path):
    """Replace every outward primitive with a recorder; work in an empty cwd."""
    import asyncio  # noqa: F401 - import before Popen/httpx are patched
    import smtplib
    import urllib.request
    import requests
    import httpx
    calls: list[str] = []

    def boom(name):
        def _f(*a, **k):
            calls.append(name)
            raise _Touched(name)
        return _f

    for mod, names in [(subprocess, ["run", "check_output", "check_call", "call"]),
                       (subprocess.Popen, ["__init__"]),
                       (requests, ["get", "post", "put", "delete", "request", "patch"]),
                       (requests.Session, ["request", "send"]),
                       (httpx, ["get", "post", "put", "delete", "request"]),
                       (httpx.Client, ["request", "send"]),
                       (smtplib, ["SMTP_SSL", "SMTP"]),
                       (urllib.request, ["urlopen"])]:
        for n in names:
            if hasattr(mod, n):
                monkeypatch.setattr(mod, n, boom(f"{getattr(mod, '__name__', mod)}.{n}"))
    try:
        import gspread
        monkeypatch.setattr(gspread, "authorize", boom("gspread.authorize"))
        monkeypatch.setattr(gspread, "service_account", boom("gspread.service_account"))
    except ImportError:
        pass
    monkeypatch.chdir(tmp_path)
    return calls


def _load(fname):
    if "/" in fname:
        return None
    if fname in ("new_sim.py", "round_sim.py"):  # whole program runs at import
        return None
    old_argv = sys.argv
    sys.argv = [fname]
    try:
        return importlib.import_module(fname[:-3])
    except BaseException:  # needs live credentials/sheets at import: structural test covers it
        return None
    finally:
        sys.argv = old_argv


def _resolve(mod, qualname):
    parts = qualname.split(".")
    if len(parts) == 2:
        inst = object.__new__(getattr(mod, parts[0]))  # skip __init__ (credentials)
        return getattr(inst, parts[1])
    return getattr(mod, parts[0])


def _args_for(fn):
    sig = inspect.signature(fn)
    return [None for p in sig.parameters.values()
            if p.default is inspect._empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]


BEHAVIOUR = [t for t in GUARDED if "/" not in t[0]]


@pytest.mark.parametrize("fname,qualname", BEHAVIOUR)
def test_guard_raises_first_under_flag_and_is_reached_when_unset(fname, qualname, monkeypatch, request):
    mod = _load(fname)  # import BEFORE the primitives are replaced
    if mod is None:
        pytest.skip(f"{fname} runs/needs live sheets at import; structural test covers it")
    primitives = request.getfixturevalue("primitives")
    monkeypatch.setenv("SIM_FAIRS_FORCE_PRODUCTION", "1")  # must NOT override
    for k in ("GH_TOKEN", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        monkeypatch.setenv(k, "x")
    fn = _resolve(mod, qualname)
    args = _args_for(fn)

    monkeypatch.setenv("ETR_ONLY_MODE", "1")
    with pytest.raises(RuntimeError) as exc:
        fn(*args)
    assert MSG.match(str(exc.value)), str(exc.value)
    assert primitives == [], f"primitive reached before guard: {primitives}"

    # flag unset: behaviour unchanged, i.e. the guard is not what stops it
    monkeypatch.delenv("ETR_ONLY_MODE")
    monkeypatch.delenv("SIM_FAIRS_FORCE_PRODUCTION")
    try:
        fn(*args)
    except BaseException as e:  # noqa: BLE001 - any outcome except the ETR refusal is fine
        assert not (isinstance(e, RuntimeError) and "ETR-only mode" in str(e)), "guard fired with flag unset"


# --------------------------------------------------------------------------
# sim_fairs ownership: stand down, fail closed without .git, not overridable
# --------------------------------------------------------------------------
def test_ownership_stands_down_under_flag_even_if_forced(monkeypatch, tmp_path):
    import sim_fairs_ownership as own
    monkeypatch.setenv("ETR_ONLY_MODE", "1")
    monkeypatch.setenv("SIM_FAIRS_FORCE_PRODUCTION", "1")
    monkeypatch.setenv("SIM_FAIRS_OWNERSHIP_RESUME_ON_STALE", "1")

    def no_git(*a, **k):
        raise _Touched("git")
    monkeypatch.setattr(own, "_git", no_git)
    assert not (tmp_path / ".git").exists()
    assert own.read_marker(tmp_path) == ("error", "")  # fail closed, no git call
    d = own.resolve_ownership("527", root=tmp_path)
    assert d.stand_down is True and d.status == "etr_only"


def test_ownership_unchanged_when_unset(monkeypatch, tmp_path):
    import sim_fairs_ownership as own
    monkeypatch.delenv("ETR_ONLY_MODE", raising=False)
    monkeypatch.delenv("SIM_FAIRS_FORCE_PRODUCTION", raising=False)

    class R:
        returncode, stdout = 1, ""
    monkeypatch.setattr(own, "_git", lambda *a, **k: R())
    assert not (tmp_path / ".git").exists()
    assert own.read_marker(tmp_path) == ("absent", "")  # pre-existing git-less behaviour
    assert own.resolve_ownership("527", root=tmp_path).stand_down is False
    monkeypatch.setenv("SIM_FAIRS_FORCE_PRODUCTION", "1")
    assert own.resolve_ownership("527", root=tmp_path).status == "forced_production"


def test_legacy_external_owner_stands_down_under_flag(monkeypatch, primitives):
    import publish_sim_fairs as p
    monkeypatch.setenv("ETR_ONLY_MODE", "1")
    monkeypatch.setenv("SIM_FAIRS_FORCE_PRODUCTION", "1")
    assert p._external_owner_legacy() == "etr_only_mode"
    assert primitives == []


# --------------------------------------------------------------------------
# C4: DG_HISTORICAL_DB override, evaluated from the real assignment lines
# --------------------------------------------------------------------------
def _eval_assignment(fname, target, env):
    tree = ast.parse((ROOT / fname).read_text(encoding="utf-8"))
    for n in tree.body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == target:
            code = compile(ast.Expression(n.value), fname, "eval")
            old = dict(os.environ)
            try:
                os.environ.clear()
                os.environ.update(env)
                return eval(code, {"os": os})
            finally:
                os.environ.clear()
                os.environ.update(old)
    raise AssertionError(f"{target} not found in {fname}")


@pytest.mark.parametrize("fname,target", [("scoring_baseline.py", "DG_HISTORICAL_DB"),
                                          ("write_base_rates.py", "DB_PATH"),
                                          ("cat_dists_team.py", "DB_PATH")])
def test_dg_historical_db_env_override(fname, target):
    default = _eval_assignment(fname, target, {})
    assert default.endswith(os.path.join("OneDrive", "dg_historical.db"))  # unchanged default
    fresh = "D:/fresh/dg_historical.db"
    assert _eval_assignment(fname, target, {"DG_HISTORICAL_DB": fresh}) == fresh
    assert _eval_assignment(fname, target, {"DG_HISTORICAL_DB": ""}) == default


# --------------------------------------------------------------------------
# kept steps: scoring_baseline prompt is non-interactive; init_weekly confined
# --------------------------------------------------------------------------
def _scoring_prompt_fn():
    tree = ast.parse((ROOT / "scoring_baseline.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "prompt_coefficient_approval")
    ns = {"sys": sys, "baseline_wind": 0.3}
    exec(compile(ast.Module([fn], []), "scoring_baseline.py", "exec"), ns)
    return ns["prompt_coefficient_approval"]


def test_scoring_baseline_prompt_never_blocks_under_flag(monkeypatch):
    fn = _scoring_prompt_fn()
    monkeypatch.setenv("ETR_ONLY_MODE", "1")

    def no_input(*a, **k):
        raise AssertionError("input() called in ETR-only mode")
    monkeypatch.setattr("builtins.input", no_input)
    assert fn(0.5, 0.4) == 0.5


def test_scoring_baseline_prompt_unchanged_when_unset(monkeypatch):
    fn = _scoring_prompt_fn()
    monkeypatch.delenv("ETR_ONLY_MODE", raising=False)

    def eof(*a):
        raise EOFError()
    monkeypatch.setattr("builtins.input", eof)
    assert fn(0.5, 0.4) == 0.5  # EOF => default y, as before
    monkeypatch.setattr("builtins.input", lambda *a: "n")
    with pytest.raises(SystemExit):
        fn(0.5, 0.4)


def test_init_weekly_sweep_confined_to_cwd(monkeypatch, tmp_path):
    import init_weekly
    work = tmp_path / "iso" / "sims_process"
    work.mkdir(parents=True)
    (work / "sim_inputs.py").write_text("tourney = 'newevent'\n", encoding="utf-8")
    stale = work / "h2h_matrix_oldevent.parquet"
    stale.write_text("x")
    monkeypatch.setattr(init_weekly, "ROOT", str(work))
    monkeypatch.setenv("ETR_ONLY_MODE", "1")
    monkeypatch.setenv("ETR_LEGACY_ROOT", str(tmp_path / "iso"))
    monkeypatch.chdir(tmp_path)  # cwd != ROOT -> refuse
    with pytest.raises(RuntimeError, match="ETR-only mode: init_weekly sweep refused"):
        init_weekly.sweep_stale_transients()
    assert stale.exists()
    monkeypatch.chdir(work)
    monkeypatch.setenv("ETR_LEGACY_ROOT", str(tmp_path / "elsewhere"))  # root not under ETR_LEGACY_ROOT
    with pytest.raises(RuntimeError):
        init_weekly.sweep_stale_transients()
    assert stale.exists()
    monkeypatch.setenv("ETR_LEGACY_ROOT", str(tmp_path / "iso"))
    init_weekly.sweep_stale_transients()
    assert not stale.exists()


def test_init_weekly_unchanged_when_unset(monkeypatch, tmp_path):
    import init_weekly
    (tmp_path / "sim_inputs.py").write_text("tourney = 'newevent'\n", encoding="utf-8")
    stale = tmp_path / "h2h_matrix_oldevent.parquet"
    stale.write_text("x")
    monkeypatch.setattr(init_weekly, "ROOT", str(tmp_path))
    monkeypatch.delenv("ETR_ONLY_MODE", raising=False)
    monkeypatch.chdir(tmp_path.parent)  # legacy behaviour has no cwd check
    init_weekly.sweep_stale_transients()
    assert not stale.exists()


def test_kept_steps_do_not_import_hole_baselines():
    """Nothing in the kept ETR-only steps invokes hole_baselines (it reads the shot archive)."""
    for fname in ("init_weekly.py", "humidity.py", "update_sheet_courses.py", "scoring_baseline.py",
                  "write_base_rates.py", "mkt_regress.py", "skill_merge.py", "dists_thiswk_team.py",
                  "new_sim.py"):
        src = (ROOT / fname).read_text(encoding="utf-8")
        assert not re.search(r"^\s*(import|from)\s+hole_baselines\b", src, re.M), fname
        assert not re.search(r"subprocess[^\n]*hole_baselines", src), fname


def test_banner_wired_into_kept_steps():
    for fname in ("init_weekly.py", "humidity.py", "update_sheet_courses.py", "scoring_baseline.py",
                  "write_base_rates.py", "mkt_regress.py", "skill_merge.py", "dists_thiswk_team.py",
                  "new_sim.py"):
        assert "print_banner" in (ROOT / fname).read_text(encoding="utf-8"), fname
