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


def test_notifier_decodes_subprocess_output_as_utf8_and_tolerates_none(monkeypatch, capsys):
    import subprocess
    import sys
    import telegram_policy

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(kwargs)
        # reader-thread decode failure leaves stdout/stderr as None on Windows
        return subprocess.CompletedProcess(cmd, 1, stdout=None, stderr=None)

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "test")
    monkeypatch.delenv("TELEGRAM_DISABLED", raising=False)
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["telegram_policy.py", "--key", "k", "--message", "m"])
    telegram_policy.main()  # must not raise TypeError on None + str
    assert calls and calls[0]["encoding"] == "utf-8" and calls[0]["errors"] == "replace"
    assert "unavailable" in capsys.readouterr().out


def test_notifier_subprocess_kwargs_survive_cp1252_undefined_bytes():
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'abc\\x8f\\xe2\\x9c\\x93')"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
    )
    assert out.returncode == 0 and out.stdout.startswith("abc")
