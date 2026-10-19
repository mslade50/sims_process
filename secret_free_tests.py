"""Run the offline suite used by the review-check tests job.

One pytest process per module. In a shared process, importing an earlier
module loads ``sim_inputs``, and
``test_provisional_round.py::ProvisionalTests::test_manifest_hash_identity_and_group_contract``
then fails because its fixture event does not match that live configuration.
The test passes when its module is the only one loaded. A second
same-process failure was not reproduced across shuffled full-suite orders;
the source-grep skip below is the test that fails on its own.

Credential and kernel modules are not collected when that dependency is
missing (see ``conftest.py``). They still collect when the dependency is
present. Script-style modules are executed as programs.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import conftest

ROOT = Path(__file__).resolve().parent

# Collected on a credential-free checkout. If one of these disappears from
# collection, the ignore rules have become too broad and the run fails.
REQUIRED_MODULES = (
    "scripts/test_review_check.py",
    "test_nightly_complete_live.py",
    "test_provisional_round.py",
    "test_score_centering.py",
)


def parse_pytest_summary(text: str) -> dict[str, int]:
    """Pull passed/failed/skipped/error counts from pytest's summary line."""
    summary = ""
    for line in reversed([ln.strip() for ln in text.splitlines() if ln.strip()]):
        if " in " in line and re.search(r"\d+ (passed|failed|skipped|error)", line):
            summary = line
            break
    summary = re.sub(r"\d+ subtests passed", "", summary)

    def grab(pattern: str) -> int:
        match = re.search(pattern, summary)
        return int(match.group(1)) if match else 0

    return {
        "passed": grab(r"(\d+) passed"),
        "failed": grab(r"(\d+) failed"),
        "skipped": grab(r"(\d+) skipped"),
        "errors": grab(r"(\d+) errors?"),
    }


def parse_script_summary(text: str) -> dict[str, int]:
    """Pull the ``N passed, M failed`` line printed by the script-style modules."""
    summary = ""
    for line in reversed([ln.strip() for ln in text.splitlines() if ln.strip()]):
        if re.search(r"\d+ passed", line) and re.search(r"\d+ failed", line):
            summary = line
            break

    def grab(pattern: str) -> int:
        match = re.search(pattern, summary)
        return int(match.group(1)) if match else 0

    return {"passed": grab(r"(\d+) passed"), "failed": grab(r"(\d+) failed")}


def _env() -> dict[str, str]:
    """Offline coefficient load, no third-party pytest plugins."""
    env = os.environ.copy()
    env["COEFFS_FROM_CACHE"] = "1"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return env


def _print_plan() -> None:
    creds = "present" if conftest.credentials_present() else "absent"
    kernel = "installed" if conftest.kernel_present() else "not installed"
    print(f"credentials: {creds}")
    print(f"sims_kernel: {kernel}")
    print("COEFFS_FROM_CACHE=1")
    print("pytest: one process per module")
    print("not collected:")
    for name, reason in conftest.excluded_modules():
        print(f"  {name}: {reason}")
    print("skipped:")
    for name, reason in sorted(conftest.SKIP_TESTS.items()):
        print(f"  {name}: {reason}")
    print("scripts:")
    for name, reason in sorted(conftest.SCRIPT_MODULES.items()):
        print(f"  {name}: {reason}")


def _collect(env: dict[str, str]) -> tuple[int, list[str], str]:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "--tb=short",
            "-p",
            "no:cacheprovider",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    files: list[str] = []
    for line in combined.splitlines():
        stripped = line.strip()
        if "::" not in stripped:
            continue
        path = stripped.split("::", 1)[0].strip()
        if not path.endswith(".py"):
            continue
        if path not in files:
            files.append(path)
    return proc.returncode, files, combined


def _run_pytest_file(path: str, env: dict[str, str]) -> tuple[int, str]:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--tb=line",
            "-p",
            "no:cacheprovider",
            path,
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _run_script(name: str, env: dict[str, str]) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(ROOT / name)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def main(argv: list[str]) -> int:
    if "--list" in argv:
        _print_plan()
        return 0

    env = _env()
    _print_plan()
    print()
    code, files, collected = _collect(env)
    if code != 0:
        sys.stdout.write(collected)
        print("collection failed")
        return code
    missing = [name for name in REQUIRED_MODULES if name not in files]
    if missing or not files:
        sys.stdout.write(collected)
        print("collection did not include: " + ", ".join(missing or ["(no files)"]))
        return 1

    pytest_totals = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    failed_runs: list[str] = []
    for path in files:
        status, output = _run_pytest_file(path, env)
        counts = parse_pytest_summary(output)
        for key in pytest_totals:
            pytest_totals[key] += counts[key]
        if status != 0 or counts["failed"] or counts["errors"]:
            failed_runs.append(path)
            print(f"FAIL {path}")
            sys.stdout.write(output)
            if not output.endswith("\n"):
                print()
        else:
            summary = ""
            for line in reversed([ln.strip() for ln in output.splitlines() if ln.strip()]):
                if " in " in line and ("passed" in line or "skipped" in line):
                    summary = line
                    break
            print(f"ok {path}: {summary or counts}")

    script_totals = {"passed": 0, "failed": 0}
    for name in sorted(conftest.SCRIPT_MODULES):
        status, output = _run_script(name, env)
        counts = parse_script_summary(output)
        script_totals["passed"] += counts["passed"]
        script_totals["failed"] += counts["failed"]
        if status != 0 or counts["failed"]:
            failed_runs.append(name)
            print(f"FAIL {name}")
            sys.stdout.write(output)
            if not output.endswith("\n"):
                print()
        else:
            print(f"ok {name}: {counts['passed']} passed, {counts['failed']} failed")

    print()
    print("=== secret-free summary ===")
    print(
        "pytest: {passed} passed, {skipped} skipped, {failed} failed, {errors} errors "
        "across {n} modules".format(n=len(files), **pytest_totals)
    )
    print(
        "scripts: {programs} programs, {passed} passed, {failed} failed".format(
            programs=len(conftest.SCRIPT_MODULES), **script_totals
        )
    )
    print(f"not collected: {len(conftest.excluded_modules())} modules")
    for name, reason in conftest.excluded_modules():
        print(f"  {name}: {reason}")
    print(f"skipped tests: {len(conftest.SKIP_TESTS)}")
    for name, reason in sorted(conftest.SKIP_TESTS.items()):
        print(f"  {name}: {reason}")
    if failed_runs:
        print("failed runs: " + ", ".join(failed_runs))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
