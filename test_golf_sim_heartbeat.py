"""The golf-sim heartbeat fails closed without calling GitHub."""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

import scripts.golf_sim_heartbeat as hb

NOW = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)
WORKFLOW = Path(".github/workflows/golf-sim-heartbeat.yml")
SAMPLE_ALERT = (
    "FAILED golf-sim heartbeat: no successful job on the self-hosted "
    "golf-sim runner within the configured window."
)


def _job(labels, conclusion, completed_at, name="round-sim"):
    return {
        "name": name,
        "labels": labels,
        "conclusion": conclusion,
        "completed_at": completed_at,
    }


def test_recent_golf_sim_success_passes():
    completed = (NOW - timedelta(hours=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
    ok, message = hb.evaluate(
        [_job(["self-hosted", "Windows", "X64", "golf-sim"], "success", completed)],
        now=NOW,
        max_hours=36,
    )
    assert ok is True
    assert "round-sim" in message


def test_stale_or_other_runner_fails_and_the_window_is_configurable():
    stale = (NOW - timedelta(hours=40)).strftime("%Y-%m-%dT%H:%M:%SZ")
    jobs = [
        _job(["self-hosted", "Windows", "X64", "golf-sim"], "success", stale),
        _job(["ubuntu-latest"], "success", NOW.strftime("%Y-%m-%dT%H:%M:%SZ")),
        _job(
            ["self-hosted", "Windows", "X64", "golf-sim"],
            "failure",
            NOW.strftime("%Y-%m-%dT%H:%M:%SZ"),
        ),
    ]
    ok, message = hb.evaluate(jobs, now=NOW, max_hours=36)
    assert ok is False
    assert "36" in message
    ok, _ = hb.evaluate(jobs, now=NOW, max_hours=48)
    assert ok is True


def test_blank_hours_default_and_bad_hours_are_rejected():
    assert hb.parse_max_hours("") == 36
    assert hb.parse_max_hours(None) == 36
    with pytest.raises(ValueError):
        hb.parse_max_hours("0")
    with pytest.raises(ValueError):
        hb.parse_max_hours("nope")


def test_run_paging_stops_once_a_page_is_entirely_old():
    cutoff = datetime(2026, 10, 21, tzinfo=timezone.utc)
    calls = []

    def fetch(url):
        calls.append(url)
        if "page=1" in url:
            return {"workflow_runs": [{
                "id": 1, "conclusion": "success", "updated_at": "2026-10-21T12:00:00Z",
                "jobs_url": "https://example.test/jobs/1",
            }]}
        if "page=2" in url:
            return {"workflow_runs": [{
                "id": 2, "conclusion": "success", "updated_at": "2026-10-19T00:00:00Z",
            }]}
        raise AssertionError(url)

    runs = hb.list_success_runs(fetch, "mslade50/sims_process", cutoff)
    assert [run["id"] for run in runs] == [1, 2]
    assert len(calls) == 2
    assert "status=success" in calls[0]

    def jobs(url):
        assert url == "https://example.test/jobs/1"
        return {"jobs": [{"name": "kept"}]}

    assert hb.collect_jobs(runs, jobs) == [{"name": "kept"}]


def test_main_does_not_call_the_network_without_a_token(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("network")

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    monkeypatch.setattr(hb.urllib.request, "urlopen", boom)
    assert hb.main(["--repo", "mslade50/sims_process"]) == 1


def test_workflow_runs_on_ubuntu_and_alerts_with_the_helper():
    text = WORKFLOW.read_text(encoding="utf-8")
    wf = yaml.safe_load(text)
    assert wf[True]["schedule"][0]["cron"] == "17 */6 * * *"
    assert wf[True]["workflow_dispatch"]["inputs"]["max_hours"]["default"] == "36"
    assert wf["permissions"] == {"actions": "read", "contents": "read"}
    job = wf["jobs"]["heartbeat"]
    assert job["runs-on"] == "ubuntu-latest"
    assert "runs-on: [self-hosted" not in text
    check = next(step for step in job["steps"] if step["name"].startswith("Fail if"))
    assert check["env"]["GITHUB_TOKEN"] == "${{ secrets.GITHUB_TOKEN }}"
    assert "vars.HEARTBEAT_MAX_HOURS" in check["env"]["HEARTBEAT_MAX_HOURS"]
    assert check["run"] == "python scripts/golf_sim_heartbeat.py"
    alert = next(step for step in job["steps"] if step["name"] == "Telegram on failure")
    assert alert["if"] == "failure()"
    assert "telegram_policy.py" in alert["run"]
    assert "--key legacy-golf-sim-heartbeat" in alert["run"]
    assert SAMPLE_ALERT in alert["run"]
