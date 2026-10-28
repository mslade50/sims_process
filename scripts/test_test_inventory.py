"""Inventory guard and SCRIPT_MODULES runner rules.

The throwaway trees live under tmp_path. Nothing here commits to this repo.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.check_test_inventory import (
    collect_tree,
    compare_inventory,
    counts,
    evaluate,
    main,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_test_inventory.py"

RUNNER = '''\
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTEST_COLLECT_ARGS = [
    "--collect-only",
    "-q",
    "--tb=short",
    "-p",
    "no:cacheprovider",
    {extra}
]


def _env():
    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return env


def parse_script_summary(text):
    passed = failed = 0
    for line in reversed([ln.strip() for ln in text.splitlines() if ln.strip()]):
        if "passed" in line and "failed" in line:
            import re
            pm = re.search(r"(\\d+) passed", line)
            fm = re.search(r"(\\d+) failed", line)
            if pm:
                passed = int(pm.group(1))
            if fm:
                failed = int(fm.group(1))
            break
    return {{"passed": passed, "failed": failed}}


def _run_script(name, env):
    path = Path(name)
    if not path.is_absolute():
        path = ROOT / name
    proc = subprocess.run(
        [sys.executable, str(path)],
        cwd=path.parent,
        env=env,
        capture_output=True,
        text=True,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
'''

CONFTEST = '''\
import os
from pathlib import Path

SCRIPT_MODULES = {{
    "test_script.py": "script program",
}}
SKIP_TESTS = {skip_tests}
EXTRA_IGNORE = {extra_ignore}


def excluded_modules():
    return {excluded}


def pytest_ignore_collect(collection_path, config):
    name = Path(str(collection_path)).name
    if name in EXTRA_IGNORE:
        return True
    if name in SCRIPT_MODULES and os.environ.get("SECRET_FREE_COLLECT_SCRIPTS") != "1":
        return True
    return None


def pytest_collection_modifyitems(config, items):
    import pytest
    for item in items:
        reason = SKIP_TESTS.get(item.name)
        if reason:
            item.add_marker(getattr(pytest.mark, "skip")(reason=reason))
'''


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _tree(
    root: Path,
    *,
    drop: bool = False,
    extra_test: bool = False,
    skip: bool = False,
    checks: int = 2,
    extra_ignore: set[str] | None = None,
    skip_tests: dict[str, str] | None = None,
    ignore_arg: str | None = None,
    workflow: str | None = None,
    swallow: bool = False,
    excluded: list[tuple[str, str]] | None = None,
) -> None:
    tests = ["def test_keep():\n    assert 1 == 1\n"]
    if not drop:
        tests.append("def test_drop():\n    assert 2 == 2\n")
    if extra_test:
        tests.append("def test_added():\n    assert 3 == 3\n")
    body = "\n".join(tests)
    if skip:
        body = "import pytest\n\n@pytest.mark.skip(reason='later')\n" + body
    _write(root / "test_keep.py", body + "\n")
    _write(root / "test_script.py", f'print("{checks} passed, 0 failed")\nraise SystemExit(0)\n')
    extra = ""
    if ignore_arg:
        extra = f'"{ignore_arg}",'
    _write(root / "secret_free_tests.py", RUNNER.format(extra=extra))
    ignored = set(extra_ignore or set())
    scripts = {"test_script.py": "script program"}
    if swallow:
        ignored = set()
        _write(root / "test_hidden.py", "def test_hidden():\n    assert 1 == 1\n")
        scripts["test_hidden.py"] = "hidden pytest module"
    conftest = CONFTEST.format(
        skip_tests=repr(skip_tests or {}),
        extra_ignore=repr(ignored),
        excluded=repr(list(excluded or [])),
    ).replace(
        'SCRIPT_MODULES = {\n    "test_script.py": "script program",\n}',
        "SCRIPT_MODULES = " + repr(scripts),
    )
    _write(root / "conftest.py", conftest)
    if workflow is not None:
        _write(root / ".github" / "workflows" / "review-check.yml", workflow)


def _ids(inv: dict, status: str | None = None) -> set[str]:
    rows = inv["tests"]
    if status is None:
        return set(rows)
    return {tid for tid, row in rows.items() if row["status"] == status}


def test_removed_test_fails():
    base = {"tests": {"test_keep.py::test_keep": {"status": "collected", "reason": ""}}, "scripts": {}}
    head = {"tests": {}, "scripts": {}}
    problems = compare_inventory(base, head)
    assert problems == ["disappeared: test_keep.py::test_keep (was collected)"]
    code, lines = evaluate(base, head, None)
    assert code == 1
    assert any(line.startswith("- disappeared:") for line in lines)
    assert any("placeholder" in line for line in lines)


def test_newly_skipped_test_fails():
    base = {"tests": {"test_keep.py::test_keep": {"status": "collected", "reason": ""}}, "scripts": {}}
    head = {
        "tests": {"test_keep.py::test_keep": {"status": "skipped", "reason": "later"}},
        "scripts": {},
    }
    assert compare_inventory(base, head) == ["now skipped: test_keep.py::test_keep — later"]
    assert evaluate(base, head, None)[0] == 1


def test_added_test_passes():
    base = {"tests": {"test_keep.py::test_keep": {"status": "collected", "reason": ""}}, "scripts": {}}
    head = {
        "tests": {
            "test_keep.py::test_keep": {"status": "collected", "reason": ""},
            "test_keep.py::test_new": {"status": "collected", "reason": ""},
        },
        "scripts": {},
    }
    assert compare_inventory(base, head) == []
    code, lines = evaluate(base, head, None)
    assert code == 0
    assert any("added 1 test ids" in line for line in lines)


def test_script_check_count_drop_fails_and_a_raise_does_not():
    base = {"tests": {}, "scripts": {"test_script.py": {"checks": 4}}}
    dropped = {"tests": {}, "scripts": {"test_script.py": {"checks": 3}}}
    raised = {"tests": {}, "scripts": {"test_script.py": {"checks": 5}}}
    gone = {"tests": {}, "scripts": {}}
    assert compare_inventory(base, dropped) == ["script checks dropped: test_script.py 4 -> 3"]
    assert compare_inventory(base, raised) == []
    assert compare_inventory(base, gone) == ["script disappeared: test_script.py (4 checks)"]


def test_test_changes_section_passes_and_a_placeholder_does_not():
    base = {"tests": {"test_keep.py::test_keep": {"status": "collected", "reason": ""}}, "scripts": {}}
    head = {"tests": {}, "scripts": {}}
    explained = evaluate(base, head, "## Test changes\nRemoved test_keep; it duplicated coverage.\n")
    assert explained[0] == 0
    assert any("explained in ## Test changes" in line for line in explained[1])
    placeholder = evaluate(base, head, "## Test changes\nN/A\n")
    assert placeholder[0] == 1
    mapping = evaluate(base, head, "## Test changes\ntest_keep.py::test_keep -> test_new.py::test_keep\n")
    assert mapping[0] == 0


def test_throwaway_collection_sees_a_removed_test_a_skip_and_a_script_drop(tmp_path: Path):
    base = tmp_path / "base"
    removed = tmp_path / "removed"
    skipped = tmp_path / "skipped"
    added = tmp_path / "added"
    fewer = tmp_path / "fewer"
    _tree(base)
    _tree(removed, drop=True)
    _tree(skipped, skip=True)
    _tree(added, extra_test=True)
    _tree(fewer, checks=1)

    base_inv = collect_tree(base)
    assert "test_keep.py::test_keep" in _ids(base_inv, "collected")
    assert "test_keep.py::test_drop" in _ids(base_inv, "collected")
    assert base_inv["scripts"]["test_script.py"]["checks"] == 2
    assert counts(base_inv)["scripts"] == 1

    removed_problems = compare_inventory(base_inv, collect_tree(removed))
    assert "disappeared: test_keep.py::test_drop (was collected)" in removed_problems

    skip_problems = compare_inventory(base_inv, collect_tree(skipped))
    assert "now skipped: test_keep.py::test_keep — later" in skip_problems
    assert not any("test_drop" in line for line in skip_problems)

    assert compare_inventory(base_inv, collect_tree(added)) == []
    assert compare_inventory(base_inv, collect_tree(fewer)) == [
        "script checks dropped: test_script.py 2 -> 1"
    ]


def test_throwaway_cli_fails_without_the_section_and_passes_with_it(tmp_path: Path):
    base = tmp_path / "base"
    head = tmp_path / "head"
    _tree(base)
    _tree(head, drop=True)
    missing = _cli(base, head, None)
    assert missing.returncode == 1
    assert "disappeared: test_keep.py::test_drop" in missing.stdout
    assert "Test changes" in missing.stdout
    secret = "SECRET_SENTINEL_do_not_print"
    explained = _cli(
        base,
        head,
        f"## Test changes\nDeleted test_drop because it duplicated test_keep. {secret}\n",
    )
    assert explained.returncode == 0
    assert "explained in ## Test changes" in explained.stdout
    assert secret not in explained.stdout
    assert secret not in explained.stderr


def test_conftest_hook_runner_flag_workflow_list_and_script_swallow(tmp_path: Path):
    base = tmp_path / "base"
    _tree(base)
    base_inv = collect_tree(base)

    hooked = tmp_path / "hook"
    _tree(hooked, extra_ignore={"test_keep.py"})
    hook_problems = compare_inventory(base_inv, collect_tree(hooked))
    assert "disappeared: test_keep.py::test_keep (was collected)" in hook_problems
    assert "disappeared: test_keep.py::test_drop (was collected)" in hook_problems

    flagged = tmp_path / "flag"
    _tree(flagged, ignore_arg="--ignore=test_keep.py")
    flag_problems = compare_inventory(base_inv, collect_tree(flagged))
    assert "disappeared: test_keep.py::test_keep (was collected)" in flag_problems

    narrowed = tmp_path / "workflow"
    _tree(
        narrowed,
        workflow=(
            "name: review-check\n"
            "on: [pull_request]\n"
            "jobs:\n"
            "  tests:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: python -m pytest test_keep.py::test_keep -q\n"
        ),
    )
    narrow_inv = collect_tree(narrowed)
    assert _ids(narrow_inv, "collected") == {"test_keep.py::test_keep"}
    assert narrow_inv["scripts"] == {}
    narrow_problems = compare_inventory(base_inv, narrow_inv)
    assert "disappeared: test_keep.py::test_drop (was collected)" in narrow_problems
    assert "script disappeared: test_script.py (2 checks)" in narrow_problems

    hidden = tmp_path / "swallow"
    _tree(hidden, swallow=True)
    hidden_inv = collect_tree(hidden)
    assert "test_hidden.py::test_hidden" not in hidden_inv["tests"]
    assert "test_hidden.py" in hidden_inv["scripts"]


def test_excluded_modules_are_not_collected_entries(tmp_path: Path):
    root = tmp_path / "excluded"
    _tree(
        root,
        excluded=[
            ("test_round_score_reprice.py", "needs Google credentials"),
            ("test_cascade_parity.py", "needs sims_kernel"),
        ],
    )
    inv = collect_tree(root)
    assert inv["tests"]["not-collected:test_round_score_reprice.py"] == {
        "status": "not_collected",
        "reason": "needs Google credentials",
    }
    assert inv["tests"]["not-collected:test_cascade_parity.py"]["status"] == "not_collected"
    assert counts(inv)["not_collected"] == 2


def test_skip_tests_dict_is_a_skipped_id(tmp_path: Path):
    root = tmp_path / "skipdict"
    _tree(root, skip_tests={"test_drop": "stale grep"})
    inv = collect_tree(root)
    assert inv["tests"]["test_keep.py::test_drop"]["status"] == "skipped"
    assert "stale grep" in inv["tests"]["test_keep.py::test_drop"]["reason"]
    assert inv["tests"]["test_keep.py::test_keep"]["status"] == "collected"


def test_sha_mode_on_a_throwaway_repo_does_not_use_this_checkout(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "inventory@example.com")
    _git(repo, "config", "user.name", "Inventory Test")
    _tree(repo)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _tree(repo, drop=True)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "drop a test")
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    env = os.environ.copy()
    env["BASE_SHA"] = base
    env["HEAD_SHA"] = head
    env.pop("PR_BODY", None)
    env.pop("PR_AUTHOR", None)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--repo", str(repo)],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stderr
    assert "disappeared: test_keep.py::test_drop" in result.stdout
    assert "test-inventory: base collected=" in result.stdout
    assert "test-inventory: head collected=" in result.stdout
    assert str(ROOT) not in result.stdout


def test_dependency_bot_skips_without_collecting(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PR_AUTHOR", "dependabot[bot]")
    monkeypatch.delenv("BASE_SHA", raising=False)
    called: list[object] = []
    monkeypatch.setattr(
        "scripts.check_test_inventory.collect_tree",
        lambda *args, **kwargs: called.append(args),
    )
    assert main([]) == 0
    assert called == []


def test_bad_sha_fails_without_a_worktree(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    monkeypatch.delenv("PR_AUTHOR", raising=False)
    monkeypatch.setenv("BASE_SHA", "not-a-sha")
    monkeypatch.setenv("HEAD_SHA", "b" * 40)
    called: list[object] = []
    monkeypatch.setattr(
        "scripts.check_test_inventory.subprocess.run",
        lambda *args, **kwargs: called.append(args),
    )
    assert main([]) == 1
    assert called == []
    assert "BASE_SHA and HEAD_SHA must be git SHAs" in capsys.readouterr().err


def test_script_module_with_pytest_tests_fails(tmp_path: Path):
    import secret_free_tests as runner

    hidden = tmp_path / "test_hidden_probe.py"
    hidden.write_text("def test_hidden():\n    assert True\n", encoding="utf-8")
    code, output = runner.evaluate_script(str(hidden), runner._env())
    assert code == 1
    assert "pytest tests" in output


def test_script_with_zero_passes_fails_and_a_reporting_script_passes(tmp_path: Path):
    import secret_free_tests as runner

    empty = tmp_path / "test_empty_probe.py"
    empty.write_text('print("0 passed, 0 failed")\nraise SystemExit(0)\n', encoding="utf-8")
    code, output = runner.evaluate_script(str(empty), runner._env())
    assert code == 1
    assert "nonzero check count" in output

    good = tmp_path / "test_good_probe.py"
    good.write_text('print("2 passed, 0 failed")\nraise SystemExit(0)\n', encoding="utf-8")
    code, output = runner.evaluate_script(str(good), runner._env())
    assert code == 0
    assert "2 passed, 0 failed" in output


def test_script_problem_states_both_rules():
    import secret_free_tests as runner

    hidden = runner.script_problem("test_hidden.py", 2, 5)
    assert hidden is not None and "2 pytest tests" in hidden
    zero = runner.script_problem("test_script.py", None, 0)
    assert zero is not None and "nonzero check count" in zero
    assert runner.script_problem("test_script.py", None, 3) is None
    assert runner.script_problem("test_script.py", 0, 3) is None


def test_ignore_hook_skips_script_modules_until_the_probe(monkeypatch: pytest.MonkeyPatch):
    import conftest

    monkeypatch.setitem(conftest.SCRIPT_MODULES, "test_fake_probe.py", "probe")
    monkeypatch.delenv("SECRET_FREE_COLLECT_SCRIPTS", raising=False)
    assert conftest.pytest_ignore_collect(Path("test_fake_probe.py"), None) is True
    monkeypatch.setenv("SECRET_FREE_COLLECT_SCRIPTS", "1")
    assert conftest.pytest_ignore_collect(Path("test_fake_probe.py"), None) is None


def test_collect_command_is_only_the_shared_args(monkeypatch: pytest.MonkeyPatch):
    import secret_free_tests as runner

    seen: dict[str, list[str]] = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = list(cmd)

        class Result:
            returncode = 0
            stdout = "scripts/test_review_check.py::test_x\n"
            stderr = ""

        return Result()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    code, files, _output = runner._collect({"COEFFS_FROM_CACHE": "1"})
    assert code == 0
    assert files == ["scripts/test_review_check.py"]
    assert seen["cmd"] == [sys.executable, "-m", "pytest", *runner.PYTEST_COLLECT_ARGS]
    assert "--ignore" not in seen["cmd"]


def _cli(base: Path, head: Path, body: str | None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("BASE_SHA", None)
    env.pop("HEAD_SHA", None)
    env.pop("PR_AUTHOR", None)
    env.pop("GITHUB_STEP_SUMMARY", None)
    if body is None:
        env.pop("PR_BODY", None)
    else:
        env["PR_BODY"] = body
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--base-root",
            str(base),
            "--head-root",
            str(head),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
