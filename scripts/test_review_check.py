"""The review-check workflow parses, stays read-only, and rejects empty PR bodies."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "review-check.yml"
SCRIPT = ROOT / "scripts" / "check_pr_body.py"
TEMPLATE = ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md"

FILLED = """## Risk tier
LOW

## Proof
```
python -m pytest scripts/test_review_check.py -q
12 passed
```

## Rollback steps
Revert this PR.
"""


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _workflow() -> dict:
    return yaml.safe_load(_workflow_text())


def _run(body: str | None, author: str = "agent") -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PR_AUTHOR"] = author
    if body is None:
        env.pop("PR_BODY", None)
    else:
        env["PR_BODY"] = body
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_workflow_parses_and_is_read_only():
    text = _workflow_text()
    wf = _workflow()
    assert wf["permissions"] == {"contents": "read", "pull-requests": "read"}
    assert "secrets." not in text
    assert "pull-requests: write" not in text
    assert "contents: write" not in text
    assert "pull_request_target" not in text
    assert "self-hosted" not in text
    on = wf[True]
    assert list(on) == ["pull_request"]
    assert on["pull_request"]["types"] == [
        "opened",
        "edited",
        "synchronize",
        "reopened",
        "ready_for_review",
    ]
    for job in wf["jobs"].values():
        assert job["runs-on"] == "ubuntu-latest"


def test_body_is_passed_by_env_not_interpolated_into_the_script():
    wf = _workflow()
    step = next(s for s in wf["jobs"]["body"]["steps"] if s.get("name") == "Check Proof and Rollback")
    assert step["env"]["PR_BODY"] == "${{ github.event.pull_request.body }}"
    assert step["env"]["PR_AUTHOR"] == "${{ github.event.pull_request.user.login }}"
    assert step["run"].strip() == "python3 scripts/check_pr_body.py"
    for job in wf["jobs"].values():
        for item in job["steps"]:
            run = item.get("run")
            if run:
                assert "${{" not in run


def test_body_job_only_checks_the_pr_body():
    wf = _workflow()
    assert list(wf["jobs"]) == ["body", "test-weakening", "test-inventory", "tests"]
    body = wf["jobs"]["body"]
    dumped = yaml.safe_dump(body)
    assert "pip install" not in dumped
    assert "pytest" not in dumped
    assert "ruff" not in dumped
    runs = [step.get("run", "") for step in body["steps"] if "run" in step]
    assert runs == ["python3 scripts/check_pr_body.py"]
    assert body["runs-on"] == "ubuntu-latest"
    assert "permissions" not in body


def test_unfilled_template_does_not_require_test_changes():
    result = _run(TEMPLATE.read_text(encoding="utf-8"))
    assert "Test changes" not in result.stderr


def test_test_weakening_job_is_read_only_and_fetches_history():
    text = _workflow_text()
    job = _workflow()["jobs"]["test-weakening"]
    assert job["runs-on"] == "ubuntu-latest"
    assert "permissions" not in job
    checkout = job["steps"][0]
    assert checkout["uses"] == "actions/checkout@v4"
    assert checkout["with"]["fetch-depth"] == 0
    step = next(item for item in job["steps"] if item.get("name") == "Check test weakening")
    assert step["env"]["PR_BODY"] == "${{ github.event.pull_request.body }}"
    assert step["env"]["PR_AUTHOR"] == "${{ github.event.pull_request.user.login }}"
    assert step["env"]["PR_NUMBER"] == "${{ github.event.pull_request.number }}"
    assert step["env"]["BASE_SHA"] == "${{ github.event.pull_request.base.sha }}"
    assert step["env"]["HEAD_SHA"] == "${{ github.event.pull_request.head.sha }}"
    assert step["run"].strip() == "python3 scripts/check_test_weakening.py"
    assert "pull_request_target" not in text
    assert "secrets." not in text


def test_inventory_job_collects_base_and_head_without_shell_interpolation():
    job = _workflow()["jobs"]["test-inventory"]
    assert job["runs-on"] == "ubuntu-latest"
    assert job["permissions"] == {"contents": "read"}
    assert job["env"]["COEFFS_FROM_CACHE"] == "1"
    checkout = job["steps"][0]
    assert checkout["with"]["fetch-depth"] == 0
    runs = [step.get("run", "") for step in job["steps"] if "run" in step]
    joined = "\n".join(runs)
    assert "${{" not in joined
    assert 'python -m pip install -r /tmp/inv-base/requirements.txt "pytest>=8.1,<10"' in joined
    assert 'python -m pip install -r /tmp/inv-head/requirements.txt "pytest>=8.1,<10"' in joined
    assert 'git fetch --no-tags origin "$BASE_REF"' in joined
    assert "git worktree add --detach /tmp/inv-base" in joined
    assert "git worktree add --detach /tmp/inv-head" in joined
    assert "python3 scripts/check_test_inventory.py --collect /tmp/inv-base --out /tmp/inventory-base.json" in joined
    assert "python3 scripts/check_test_inventory.py --collect /tmp/inv-head --out /tmp/inventory-head.json" in joined
    assert "python3 scripts/check_test_inventory.py --compare /tmp/inventory-base.json /tmp/inventory-head.json" in joined
    checkout = job["steps"][0]
    assert checkout["with"]["ref"] == "${{ github.event.pull_request.head.sha }}"
    base = next(step for step in job["steps"] if step.get("name") == "Collect base inventory")
    assert base["env"]["BASE_SHA"] == "${{ github.event.pull_request.base.sha }}"
    assert base["env"]["BASE_REF"] == "${{ github.base_ref }}"
    head = next(step for step in job["steps"] if step.get("name") == "Collect head inventory and compare")
    assert head["env"]["HEAD_SHA"] == "${{ github.event.pull_request.head.sha }}"
    assert head["env"]["PR_BODY"] == "${{ github.event.pull_request.body }}"
    assert head["env"]["PR_AUTHOR"] == "${{ github.event.pull_request.user.login }}"


def test_guard_checkouts_do_not_persist_credentials():
    jobs = _workflow()["jobs"]
    assert list(jobs) == ["body", "test-weakening", "test-inventory", "tests"]
    for name, job in jobs.items():
        checkouts = [
            step for step in job["steps"] if str(step.get("uses", "")).startswith("actions/checkout@")
        ]
        assert checkouts, name
        for step in checkouts:
            assert step["with"]["persist-credentials"] is False, name


def test_secret_free_job_reads_contents_only_and_runs_the_suite():
    text = _workflow_text()
    job = _workflow()["jobs"]["tests"]
    assert job["runs-on"] == "ubuntu-latest"
    assert job["permissions"] == {"contents": "read"}
    assert job["env"]["COEFFS_FROM_CACHE"] == "1"
    assert "ruff" not in text
    runs = [step.get("run", "") for step in job["steps"] if "run" in step]
    assert 'python -m pip install -r requirements.txt "pytest>=8.1,<10"' in runs
    assert "python secret_free_tests.py" in runs


def test_pytest_summary_parser_ignores_subtests():
    import secret_free_tests

    text = "1 failed, 606 passed, 11 warnings, 434 subtests passed in 13.31s\n"
    assert secret_free_tests.parse_pytest_summary(text) == {
        "passed": 606,
        "failed": 1,
        "skipped": 0,
        "errors": 0,
    }


def test_exclusion_list_is_explicit():
    import conftest

    assert "test_round_score_reprice.py" in conftest.CREDENTIAL_IMPORTS
    assert "test_reprice_alert_delivery.py" in conftest.CREDENTIAL_IMPORTS
    assert "test_finish_probs_real.py" in conftest.CREDENTIAL_IMPORTS
    assert "test_env_setup.py" in conftest.CREDENTIAL_IMPORTS
    assert "test_cascade_parity.py" in conftest.KERNEL_IMPORTS
    assert "test_maker_guard.py" in conftest.SCRIPT_MODULES
    assert "test_live_stats_engine_exposes_no_sheet_write_mode" in conftest.SKIP_TESTS


def test_empty_body_fails():
    result = _run("")
    assert result.returncode == 1
    assert "PR body is empty" in result.stderr
    assert result.stdout == ""


def test_unfilled_template_fails():
    result = _run(TEMPLATE.read_text(encoding="utf-8"))
    assert result.returncode == 1
    assert "Proof section is empty or still placeholder text" in result.stderr
    assert "Rollback section is empty or still placeholder text" in result.stderr
    assert "Test changes" not in result.stderr


def test_placeholder_words_fail():
    body = "## Proof\nTODO\n\n## Rollback steps\nTBD\n"
    result = _run(body)
    assert result.returncode == 1
    assert "Proof section is empty or still placeholder text" in result.stderr
    assert "Rollback section is empty or still placeholder text" in result.stderr


def test_visible_template_instructions_fail():
    body = (
        "## Proof\n"
        "Paste the real checker-test output here. If the change touches a page or an alert, "
        "also attach sample alert text, a screenshot, or a JSON sample.\n\n"
        "## Rollback steps\n"
        "Plain words: what to revert, which flag to turn off, or what to redeploy.\n"
    )
    result = _run(body)
    assert result.returncode == 1


def test_fenced_heading_does_not_end_the_section():
    body = """## Proof
```
## not a heading
1 passed
```

## Rollback steps
Revert this PR.
"""
    result = _run(body)
    assert result.returncode == 0


REJECTED_SECTIONS = [
    "N/A.",
    "```\n```",
    "TODO fill this in",
    "Tests passed",
    "Revert if needed",
]
REAL_PYTEST = """python -m pytest scripts/test_review_check.py -q
........................ [100%]
12 passed in 0.2s
"""
FENCED_LOG = """```
python -m pytest scripts/test_review_check.py -q
12 passed in 0.2s
```
"""


@pytest.mark.parametrize("section", REJECTED_SECTIONS)
def test_vague_phrase_is_rejected_as_the_whole_proof_section(section: str):
    body = f"## Proof\n{section}\n\n## Rollback steps\nRevert this PR\n"
    result = _run(body)
    assert result.returncode == 1
    assert "Proof section is empty or still placeholder text" in result.stderr
    assert "Rollback" not in result.stderr


@pytest.mark.parametrize("section", REJECTED_SECTIONS)
def test_vague_phrase_is_rejected_as_the_whole_rollback_section(section: str):
    body = f"## Proof\n{REAL_PYTEST}\n## Rollback steps\n{section}\n"
    result = _run(body)
    assert result.returncode == 1
    assert "Rollback section is empty or still placeholder text" in result.stderr
    assert "Proof section" not in result.stderr


def test_real_pytest_output_is_accepted():
    body = f"## Proof\n{REAL_PYTEST}\n## Rollback steps\nRevert this PR\n"
    result = _run(body)
    assert result.returncode == 0
    assert "Proof and Rollback sections are filled" in result.stdout


def test_fenced_test_log_is_accepted():
    body = f"## Proof\n{FENCED_LOG}\n## Rollback steps\nRevert this PR\n"
    result = _run(body)
    assert result.returncode == 0


def test_revert_this_pr_is_accepted():
    body = f"## Proof\n{FENCED_LOG}\n## Rollback steps\nRevert this PR\n"
    result = _run(body)
    assert result.returncode == 0
    assert "Rollback" not in result.stderr


def test_missing_rollback_fails():
    body = "## Proof\npython -m pytest scripts/test_review_check.py -q\n1 passed\n"
    result = _run(body)
    assert result.returncode == 1
    assert "Rollback section is missing" in result.stderr
    assert "Proof" not in result.stderr


def test_filled_body_passes_and_does_not_print_the_body():
    secret = "SECRET_SENTINEL_do_not_print"
    body = FILLED.replace("12 passed", f"12 passed {secret}")
    result = _run(body, author="cursor[bot]")
    assert result.returncode == 0
    assert "Proof and Rollback sections are filled" in result.stdout
    assert secret not in result.stdout
    assert secret not in result.stderr


def test_dependency_bots_skip_the_template_other_bots_do_not():
    template = TEMPLATE.read_text(encoding="utf-8")
    for author in ("dependabot[bot]", "dependabot", "renovate[bot]"):
        result = _run(template, author=author)
        assert result.returncode == 0, author
        assert "dependency-bot" in result.stdout
        assert "<!--" not in result.stdout
    agent = _run("", author="cursor[bot]")
    assert agent.returncode == 1
    assert "PR body is empty" in agent.stderr
