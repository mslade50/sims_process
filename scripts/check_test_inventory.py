"""Fail a PR whose collected tests get weaker unless the body explains why.

This is the inventory guard. It does not read the diff. It runs the same
secret-free collection the tests job runs, once on the base commit and once
on the head commit, and compares the results.

For each commit it records:

- every pytest node id the runner collects, with status ``collected`` or
  ``skipped`` (skip / skipif, including conftest ``SKIP_TESTS``)
- each ``CREDENTIAL_IMPORTS`` / ``KERNEL_IMPORTS`` module the runner did not
  collect, as ``not_collected`` plus the reason
- each script-style program's reported check count (passed + failed)

It fails, unless the PR body has a real ``## Test changes`` section, when:

- a test id disappears
- a collected test becomes skipped or not collected
- a skipped test becomes not collected
- a script program disappears or its check count drops

Added tests are allowed. A rename is one disappeared id and one new id, so it
needs the section. ``is_placeholder`` in ``scripts/check_pr_body.py`` decides
whether that section is real.

The collector follows the tree's own conftest, ``PYTEST_COLLECT_ARGS`` (or the
runner's historical collect command), script runner, and the tests job's
``run`` steps. A workflow that stops calling ``secret_free_tests.py`` is
collected from the pytest command it runs instead. CI installs that commit's
``requirements.txt`` before collecting, so a dependency change that stops
collection fails this job.

Usage:
    python3 scripts/check_test_inventory.py --collect ROOT --out inventory.json
    python3 scripts/check_test_inventory.py --compare base.json head.json
    BASE_SHA=... HEAD_SHA=... python3 scripts/check_test_inventory.py --base-root A --head-root B
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SHA = re.compile(r"^[0-9a-fA-F]{7,64}$")
DEFAULT_COLLECT_ARGS = [
    "--collect-only",
    "-q",
    "--tb=short",
    "-p",
    "no:cacheprovider",
]


class InventoryError(Exception):
    """Collection failed. The message is safe to print."""


class _InventoryPlugin:
    """Record node ids and skip markers during collection."""

    def __init__(self) -> None:
        self.rows: list[dict[str, str]] = []

    def pytest_collection_finish(self, session) -> None:
        for item in session.items:
            skip = item.get_closest_marker("skip")
            skipif = item.get_closest_marker("skipif")
            marker = skip or skipif
            reason = ""
            if marker is not None:
                if marker.kwargs.get("reason"):
                    reason = str(marker.kwargs["reason"])
                elif marker.args:
                    reason = str(marker.args[0])
            self.rows.append(
                {
                    "id": item.nodeid.replace("\\", "/"),
                    "status": "skipped" if marker is not None else "collected",
                    "reason": reason,
                }
            )


def _body_api():
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from scripts.check_pr_body import is_dependency_bot, is_placeholder, sections

    return is_dependency_bot, is_placeholder, sections


def explained(body: str | None) -> bool:
    """True when ## Test changes is present and not placeholder text."""
    _bots, is_placeholder, sections = _body_api()
    if body is None:
        return False
    text = sections(body).get("test")
    if text is None:
        return False
    return not is_placeholder(text)


def counts(inv: dict) -> dict[str, int]:
    tests = inv.get("tests") or {}
    scripts = inv.get("scripts") or {}
    return {
        "collected": sum(1 for row in tests.values() if row.get("status") == "collected"),
        "skipped": sum(1 for row in tests.values() if row.get("status") == "skipped"),
        "not_collected": sum(1 for row in tests.values() if row.get("status") == "not_collected"),
        "scripts": len(scripts),
        "checks": sum(_checks(row) for row in scripts.values()),
    }


def format_counts(label: str, inv: dict) -> str:
    row = counts(inv)
    return (
        "test-inventory: {label} collected={collected} skipped={skipped} "
        "not_collected={not_collected} scripts={scripts} checks={checks}"
    ).format(label=label, **row)


def _checks(row: object) -> int:
    if isinstance(row, int):
        return row
    if isinstance(row, dict):
        return int(row.get("checks") or 0)
    return 0


def compare_inventory(base: dict, head: dict) -> list[str]:
    """Return human-readable weakenings. Additions are not weakenings."""
    problems: list[str] = []
    base_tests = base.get("tests") or {}
    head_tests = head.get("tests") or {}
    for tid in sorted(base_tests):
        old = base_tests[tid]
        if tid not in head_tests:
            problems.append(f"disappeared: {tid} (was {old.get('status', 'collected')})")
            continue
        new = head_tests[tid]
        old_status = old.get("status")
        new_status = new.get("status")
        if old_status == "collected" and new_status in {"skipped", "not_collected"}:
            reason = new.get("reason") or ""
            suffix = f" — {reason}" if reason else ""
            problems.append(f"now {new_status}: {tid}{suffix}")
        elif old_status == "skipped" and new_status == "not_collected":
            reason = new.get("reason") or ""
            suffix = f" — {reason}" if reason else ""
            problems.append(f"now not_collected: {tid}{suffix}")
    base_scripts = base.get("scripts") or {}
    head_scripts = head.get("scripts") or {}
    for name in sorted(base_scripts):
        before = _checks(base_scripts[name])
        if name not in head_scripts:
            problems.append(f"script disappeared: {name} ({before} checks)")
            continue
        after = _checks(head_scripts[name])
        if after < before:
            problems.append(f"script checks dropped: {name} {before} -> {after}")
    return problems


def evaluate(base: dict, head: dict, body: str | None, author: str | None = None) -> tuple[int, list[str]]:
    """Return an exit code and summary lines. Never includes the body."""
    is_dependency_bot, _placeholder, _sections = _body_api()
    if is_dependency_bot(author):
        return 0, ["test-inventory: dependency-bot PR, check skipped"]
    problems = compare_inventory(base, head)
    added = sorted(set((head.get("tests") or {})) - set(base.get("tests") or {}))
    lines = [
        format_counts("base", base),
        format_counts("head", head),
        f"test-inventory: added {len(added)} test ids",
    ]
    if not problems:
        lines.append("test-inventory: no tests disappeared, skipped, or lost script checks")
        return 0, lines
    noun = "change" if len(problems) == 1 else "changes"
    lines.append(f"test-inventory: {len(problems)} inventory {noun}:")
    lines.extend(f"- {item}" for item in problems)
    if explained(body):
        lines.append("test-inventory: explained in ## Test changes")
        return 0, lines
    lines.append(
        "test-inventory: ## Test changes is missing or still placeholder text. "
        "A renamed test is one disappeared id and one added id; name it as old -> new."
    )
    return 1, lines


def _workflow_text(root: Path) -> str:
    path = root / ".github" / "workflows" / "review-check.yml"
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def _tests_job(root: Path) -> dict | None:
    text = _workflow_text(root)
    if not text:
        return None
    import yaml

    loaded = yaml.safe_load(text) or {}
    job = (loaded.get("jobs") or {}).get("tests")
    return job if isinstance(job, dict) else None


def workflow_uses_runner(root: Path) -> bool:
    """True when the tests job runs secret_free_tests.py, or no workflow is present."""
    job = _tests_job(root)
    if job is None:
        return (root / "secret_free_tests.py").is_file()
    runs = []
    for step in job.get("steps") or []:
        if isinstance(step, dict) and step.get("run"):
            runs.append(str(step["run"]))
    return "secret_free_tests.py" in "\n".join(runs)


def pytest_args_from_workflow(root: Path) -> list[str]:
    """Arguments of the pytest command in the tests job, excluding the pytest token."""
    job = _tests_job(root)
    if job is None:
        return []
    found: list[str] = []
    for step in job.get("steps") or []:
        if not isinstance(step, dict) or not step.get("run"):
            continue
        for raw in str(step["run"]).splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                tokens = shlex.split(line, posix=True)
            except ValueError:
                continue
            if "pytest" not in tokens:
                continue
            found = tokens[tokens.index("pytest") + 1 :]
    return found


def _collect_args(extra: list[str]) -> list[str]:
    skip = {"--collect-only", "-q", "--tb=short", "--tb=line", "--tb=no"}
    kept = [tok for tok in extra if tok not in skip]
    return list(DEFAULT_COLLECT_ARGS) + kept


def _pytest_collect(args: list[str]) -> tuple[int, list[dict[str, str]], str]:
    import pytest

    plugin = _InventoryPlugin()
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = int(pytest.main(args, plugins=[plugin]))
    output = stdout.getvalue() + stderr.getvalue()
    return code, plugin.rows, output


def _not_collected(conftest) -> dict[str, dict[str, str]]:
    rows = []
    if hasattr(conftest, "excluded_modules"):
        rows = list(conftest.excluded_modules())
    found: dict[str, dict[str, str]] = {}
    for name, reason in rows:
        found[f"not-collected:{name}"] = {"status": "not_collected", "reason": str(reason)}
    return found


def _script_inventory(root: Path, runner, env: dict[str, str]) -> dict[str, dict[str, int]]:
    import conftest

    scripts: dict[str, dict[str, int]] = {}
    names = sorted(getattr(conftest, "SCRIPT_MODULES", {}))
    for name in names:
        if hasattr(runner, "_run_script") and hasattr(runner, "parse_script_summary"):
            status, output = runner._run_script(name, env)
            parsed = runner.parse_script_summary(output)
            passed = int(parsed.get("passed") or 0)
            failed = int(parsed.get("failed") or 0)
        else:
            proc = subprocess.run(
                [sys.executable, str(root / name)],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            status = proc.returncode
            text = (proc.stdout or "") + (proc.stderr or "")
            passed_match = re.search(r"(\d+) passed", text)
            failed_match = re.search(r"(\d+) failed", text)
            passed = int(passed_match.group(1)) if passed_match else 0
            failed = int(failed_match.group(1)) if failed_match else 0
        scripts[name] = {
            "checks": passed + failed,
            "passed": passed,
            "failed": failed,
            "returncode": int(status),
        }
    return scripts


def _worker(root: Path) -> None:
    """Collect one tree. stdout is a single JSON document."""
    root = root.resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ.setdefault("COEFFS_FROM_CACHE", "1")
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ.pop("SECRET_FREE_COLLECT_SCRIPTS", None)
    mode = os.environ.get("INVENTORY_MODE", "runner")
    if mode == "workflow":
        extra = json.loads(os.environ.get("INVENTORY_PYTEST_ARGS") or "[]")
        code, rows, output = _pytest_collect(_collect_args(extra))
        if code not in (0, 5):
            sys.stderr.write(output)
            raise InventoryError(f"pytest collection failed with exit {code}")
        tests = {row["id"]: {"status": row["status"], "reason": row["reason"]} for row in rows}
        payload = {"tests": tests, "scripts": {}}
    else:
        import conftest
        import secret_free_tests as runner

        args = list(getattr(runner, "PYTEST_COLLECT_ARGS", DEFAULT_COLLECT_ARGS))
        code, rows, output = _pytest_collect(args)
        if code not in (0, 5):
            sys.stderr.write(output)
            raise InventoryError(f"pytest collection failed with exit {code}")
        tests = {row["id"]: {"status": row["status"], "reason": row["reason"]} for row in rows}
        tests.update(_not_collected(conftest))
        env = runner._env() if hasattr(runner, "_env") else os.environ.copy()
        scripts = _script_inventory(root, runner, env)
        payload = {"tests": tests, "scripts": scripts}
    json.dump(payload, sys.stdout)
    sys.stdout.write("\n")


def collect_tree(root: Path) -> dict:
    """Collect ``root`` in a subprocess so its conftest and runner are used."""
    root = root.resolve()
    env = os.environ.copy()
    env["COEFFS_FROM_CACHE"] = "1"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env.pop("SECRET_FREE_COLLECT_SCRIPTS", None)
    if workflow_uses_runner(root):
        env["INVENTORY_MODE"] = "runner"
        env.pop("INVENTORY_PYTEST_ARGS", None)
    else:
        env["INVENTORY_MODE"] = "workflow"
        env["INVENTORY_PYTEST_ARGS"] = json.dumps(pytest_args_from_workflow(root))
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", str(root)],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = ((proc.stderr or "") + "\n" + (proc.stdout or "")).strip()
        raise InventoryError(detail[-4000:] or "inventory collection failed")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise InventoryError("inventory worker returned invalid JSON: " + proc.stdout[-500:]) from exc


def _emit(lines: list[str]) -> None:
    text = "\n".join(lines)
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary:
        return
    with open(summary, "a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def _write_inventory(path: Path, inv: dict) -> None:
    path.write_text(json.dumps(inv, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_inventory(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_git(repo: Path, cmd: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    return subprocess.run(cmd, cwd=repo, capture_output=True, text=True, check=False, env=env)


def _have_commit(repo: Path, sha: str) -> bool:
    proc = _run_git(repo, ["git", "cat-file", "-e", f"{sha}^{{commit}}"])
    return proc.returncode == 0


def _worktree(repo: Path, sha: str) -> Path:
    if not _have_commit(repo, sha):
        raise InventoryError(f"could not find commit {sha}")
    parent = Path(tempfile.mkdtemp(prefix="inv-"))
    dest = parent / "tree"
    proc = _run_git(repo, ["git", "worktree", "add", "--detach", str(dest), sha])
    if proc.returncode != 0:
        raise InventoryError("git worktree add failed")
    return dest


def _remove_worktree(repo: Path, dest: Path) -> None:
    _run_git(repo, ["git", "worktree", "remove", "--force", str(dest)])
    parent = dest.parent
    if parent.name.startswith("inv-"):
        try:
            parent.rmdir()
        except OSError:
            pass


def _compare_paths(base_path: Path, head_path: Path, author: str | None, body: str | None) -> int:
    try:
        base = _read_inventory(base_path)
        head = _read_inventory(head_path)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"test-inventory: {exc}", file=sys.stderr)
        return 1
    code, lines = evaluate(base, head, body, author)
    _emit(lines)
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare secret-free test inventories.")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--collect", type=Path, help="Collect one tree and write JSON.")
    parser.add_argument("--out", type=Path, help="Where --collect writes JSON.")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BASE_JSON", "HEAD_JSON"))
    parser.add_argument("--base-root", type=Path, help="Collect this tree as the base side.")
    parser.add_argument("--head-root", type=Path, help="Collect this tree as the head side.")
    parser.add_argument("--repo", type=Path, help="Git repo used with BASE_SHA and HEAD_SHA.")
    args = parser.parse_args(argv)
    if args.worker:
        try:
            _worker(args.worker)
        except InventoryError as exc:
            print(f"test-inventory: {exc}", file=sys.stderr)
            return 1
        return 0

    author = os.environ.get("PR_AUTHOR")
    is_dependency_bot, _placeholder, _sections = _body_api()
    if is_dependency_bot(author) and not args.collect and not args.worker:
        _emit(["test-inventory: dependency-bot PR, check skipped"])
        return 0

    body = os.environ.get("PR_BODY")
    try:
        if args.collect:
            inv = collect_tree(args.collect)
            if args.out:
                _write_inventory(args.out, inv)
            print(format_counts(str(args.collect), inv), file=sys.stderr)
            if not args.out:
                json.dump(inv, sys.stdout)
                sys.stdout.write("\n")
            return 0
        if args.compare:
            return _compare_paths(args.compare[0], args.compare[1], author, body)
        if args.base_root or args.head_root:
            if not args.base_root or not args.head_root:
                print("test-inventory: pass both --base-root and --head-root", file=sys.stderr)
                return 1
            base = collect_tree(args.base_root)
            head = collect_tree(args.head_root)
            code, lines = evaluate(base, head, body, author)
            _emit(lines)
            return code
        repo = (args.repo or ROOT).resolve()
        base_sha = os.environ.get("BASE_SHA", "")
        head_sha = os.environ.get("HEAD_SHA", "")
        if not _SHA.fullmatch(base_sha) or not _SHA.fullmatch(head_sha):
            print("test-inventory: BASE_SHA and HEAD_SHA must be git SHAs", file=sys.stderr)
            return 1
        base_tree: Path | None = None
        head_tree: Path | None = None
        try:
            base_tree = _worktree(repo, base_sha)
            head_tree = _worktree(repo, head_sha)
            base = collect_tree(base_tree)
            head = collect_tree(head_tree)
        finally:
            if base_tree is not None:
                _remove_worktree(repo, base_tree)
            if head_tree is not None:
                _remove_worktree(repo, head_tree)
    except InventoryError as exc:
        print(f"test-inventory: {exc}", file=sys.stderr)
        return 1
    code, lines = evaluate(base, head, body, author)
    _emit(lines)
    return code


if __name__ == "__main__":
    sys.exit(main())
