"""Fail when the self-hosted golf-sim runner has no recent successful job.

Reads the GitHub Actions runs API with GITHUB_TOKEN. It does not dispatch
workflows or call any trading endpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.request
from datetime import datetime, timedelta, timezone

RUNNER_LABEL = "golf-sim"
DEFAULT_MAX_HOURS = 36.0
_ANCIENT = datetime.min.replace(tzinfo=timezone.utc)


def parse_max_hours(raw, default=DEFAULT_MAX_HOURS):
    """Return a positive hour window. Blank uses the default."""
    text = str(default if raw is None or str(raw).strip() == "" else raw).strip()
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"HEARTBEAT_MAX_HOURS={raw!r} is not a number") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"HEARTBEAT_MAX_HOURS={raw!r} must be a positive number of hours"
        )
    return value


def parse_github_time(value):
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def job_is_recent_success(job, cutoff):
    labels = [str(label) for label in (job.get("labels") or [])]
    if RUNNER_LABEL not in labels:
        return False
    if job.get("conclusion") != "success":
        return False
    completed = parse_github_time(job.get("completed_at"))
    return completed is not None and completed >= cutoff


def evaluate(jobs, *, now, max_hours):
    """Return (ok, message) for jobs already fetched from the Actions API."""
    cutoff = now - timedelta(hours=max_hours)
    hits = [job for job in jobs if job_is_recent_success(job, cutoff)]
    if not hits:
        return False, (
            f"No successful {RUNNER_LABEL} job in the last {max_hours:g} hours "
            f"(cutoff {cutoff.strftime('%Y-%m-%d %H:%M UTC')})"
        )
    newest = max(hits, key=lambda job: parse_github_time(job.get("completed_at")))
    completed = parse_github_time(newest.get("completed_at"))
    name = newest.get("name") or newest.get("id")
    return True, (
        f"{RUNNER_LABEL} job {name} succeeded at "
        f"{completed.strftime('%Y-%m-%d %H:%M UTC')}"
    )


def list_success_runs(fetch_json, repo, cutoff, max_pages=10):
    """Page successful workflow runs until a page is entirely before cutoff."""
    runs = []
    for page in range(1, max_pages + 1):
        url = (
            f"https://api.github.com/repos/{repo}/actions/runs"
            f"?status=success&per_page=50&page={page}"
        )
        batch = fetch_json(url).get("workflow_runs") or []
        if not batch:
            break
        runs.extend(batch)
        if all(
            (parse_github_time(run.get("updated_at")) or _ANCIENT) < cutoff
            for run in batch
        ):
            break
    return runs


def collect_jobs(runs, fetch_json):
    jobs = []
    for run in runs:
        if run.get("conclusion") != "success":
            continue
        url = run.get("jobs_url")
        if not url:
            continue
        jobs.extend(fetch_json(url).get("jobs") or [])
    return jobs


def github_get(url, token):
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "golf-sim-heartbeat",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-hours",
        default=os.getenv("HEARTBEAT_MAX_HOURS", str(int(DEFAULT_MAX_HOURS))),
    )
    parser.add_argument("--repo", default=os.getenv("GITHUB_REPOSITORY", ""))
    args = parser.parse_args(argv)
    try:
        hours = parse_max_hours(args.max_hours)
    except ValueError as exc:
        print(exc)
        return 1
    token = (os.getenv("GITHUB_TOKEN") or "").strip()
    repo = str(args.repo or "").strip()
    if not token or "/" not in repo:
        print("GITHUB_TOKEN and GITHUB_REPOSITORY (owner/name) are required")
        return 1
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=hours)

    def fetch(url):
        return github_get(url, token)

    try:
        runs = list_success_runs(fetch, repo, cutoff)
        jobs = collect_jobs(runs, fetch)
    except Exception as exc:
        print("Actions API failed:", type(exc).__name__)
        return 1
    ok, message = evaluate(jobs, now=now, max_hours=hours)
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
