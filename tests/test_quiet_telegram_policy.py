import ast
import os
from pathlib import Path
from types import SimpleNamespace
from telegram_policy import should_alert


def isolated_function(file, name):
    tree = ast.parse(Path(file).read_text(encoding="utf-8"))
    fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name)
    namespace = {"os": os, "requests": SimpleNamespace(post=lambda *a, **k: (_ for _ in ()).throw(AssertionError("unexpected send")))}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), file, "exec"), namespace)
    return namespace[name]


def test_routine_clv_and_round_diagnostics_default_quiet(monkeypatch):
    monkeypatch.delenv("TELEGRAM_CLV_ENABLED", raising=False)
    monkeypatch.delenv("TELEGRAM_DIAGNOSTICS", raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "test")
    isolated_function("clv_alert.py", "send_telegram")("report")
    isolated_function("round_sim.py", "_send_telegram")("diagnostic")


def test_repeated_workflow_failures_are_bounded():
    assert should_alert({}, 200000)
    assert not should_alert({"sent_at": 199999}, 200000)
    assert not should_alert({"reserved_at": 199999}, 200000)
    assert should_alert({"sent_at": 100000}, 200000)
