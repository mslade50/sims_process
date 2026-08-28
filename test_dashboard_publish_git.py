import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import push_dashboard_data as dashboard_publish


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", "-c", "core.hooksPath=", "-C", str(repo), *args],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_git_push_builds_on_atomic_remote_commit_without_rebasing_checkout(
    tmp_path, monkeypatch
):
    """The strict fairs publisher advances origin/main but not the local branch.

    Generated fairs files in the worktree already match that remote commit.  The
    dashboard publish must layer its own scoped commit on the remote tip without
    rebasing/autostashing those generated files or disturbing unrelated staging.
    """
    origin = tmp_path / "origin.git"
    work = tmp_path / "work"
    publisher = tmp_path / "publisher"

    subprocess.run(
        ["git", "init", "--bare", "--initial-branch=main", str(origin)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "init", "--initial-branch=main", str(work)],
        check=True,
        capture_output=True,
        text=True,
    )
    _git(work, "config", "user.name", "test")
    _git(work, "config", "user.email", "test@example.com")
    _git(work, "remote", "add", "origin", str(origin))

    _write(work / "sim_fairs.json", '{"generation":"old"}\n')
    _write(work / "dashboard_data" / "finish_equity_live.csv", "book,market\nold,winner\n")
    _write(work / "sim_inputs.py", 'tourney = "test"\n')
    _write(work / "permanent_data" / "historical_dists" / "fixture.txt", "seed\n")
    _write(work / "notes.txt", "seed\n")
    _git(work, "add", ".")
    _git(work, "commit", "-m", "seed")
    _git(work, "push", "-u", "origin", "main")
    local_head = _git(work, "rev-parse", "HEAD").stdout.strip()

    subprocess.run(
        ["git", "clone", str(origin), str(publisher)],
        check=True,
        capture_output=True,
        text=True,
    )
    _git(publisher, "config", "user.name", "atomic publisher")
    _git(publisher, "config", "user.email", "publisher@example.com")
    _write(publisher / "sim_fairs.json", '{"generation":"fresh"}\n')
    _write(publisher / "round_h2h_r2.parquet", "new atomic artifact\n")
    _git(publisher, "add", "sim_fairs.json", "round_h2h_r2.parquet")
    _git(publisher, "commit", "-m", "sim_fairs: atomic publish")
    _git(publisher, "push", "origin", "main")
    atomic_head = _git(publisher, "rev-parse", "HEAD").stdout.strip()

    # publish_sim_fairs mirrors its pushed blobs into the worktree, but deliberately
    # does not move this checkout's branch or index. Newly introduced package files
    # are therefore untracked relative to local HEAD; pull --rebase --autostash
    # cannot stash them and used to fail rather than overwrite them.
    _write(work / "sim_fairs.json", '{"generation":"fresh"}\n')
    _write(work / "round_h2h_r2.parquet", "new atomic artifact\n")
    _write(
        work / "dashboard_data" / "finish_equity_live.csv",
        "book,market\nkalshi,top_5\n",
    )
    _write(work / "notes.txt", "unrelated staged work\n")
    _git(work, "add", "notes.txt")

    monkeypatch.setattr(dashboard_publish, "PROJECT_ROOT", str(work))
    monkeypatch.setattr(dashboard_publish, "get_tourney", lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "maker_alerts",
        SimpleNamespace(send_telegram=lambda _message: False),
    )

    dashboard_publish.git_push()

    _git(work, "fetch", "origin", "main")
    remote_head = _git(work, "rev-parse", "origin/main").stdout.strip()
    assert remote_head != atomic_head
    assert _git(work, "rev-parse", "origin/main^").stdout.strip() == atomic_head
    assert (
        _git(work, "show", "origin/main:sim_fairs.json").stdout
        == '{"generation":"fresh"}\n'
    )
    assert (
        _git(work, "show", "origin/main:round_h2h_r2.parquet").stdout
        == "new atomic artifact\n"
    )
    assert "kalshi,top_5" in _git(
        work, "show", "origin/main:dashboard_data/finish_equity_live.csv"
    ).stdout

    # The plumbing publish must preserve the invariant shared with
    # publish_sim_fairs: no branch, worktree, or ordinary index mutation.
    assert _git(work, "rev-parse", "HEAD").stdout.strip() == local_head
    assert _git(work, "diff", "--cached", "--name-only").stdout.strip() == "notes.txt"
    assert _git(work, "diff", "--name-only").stdout.splitlines() == [
        "dashboard_data/finish_equity_live.csv",
        "sim_fairs.json",
    ]
    assert _git(work, "ls-files", "--others", "--exclude-standard").stdout.strip() == (
        "round_h2h_r2.parquet"
    )


def test_git_push_rebuilds_after_remote_advance_and_preserves_remote_only_file(
    tmp_path, monkeypatch
):
    origin = tmp_path / "origin.git"
    work = tmp_path / "work"
    publisher = tmp_path / "publisher"
    real_run = subprocess.run

    real_run(
        ["git", "init", "--bare", "--initial-branch=main", str(origin)],
        check=True,
        capture_output=True,
        text=True,
    )
    real_run(
        ["git", "init", "--initial-branch=main", str(work)],
        check=True,
        capture_output=True,
        text=True,
    )
    _git(work, "config", "user.name", "test")
    _git(work, "config", "user.email", "test@example.com")
    _git(work, "remote", "add", "origin", str(origin))
    _write(work / "dashboard_data" / "finish_equity_live.csv", "old\n")
    _write(work / "sim_inputs.py", 'tourney = "test"\n')
    _write(work / "permanent_data" / "historical_dists" / "fixture.txt", "seed\n")
    _write(work / "notes.txt", "seed\n")
    _git(work, "add", ".")
    _git(work, "commit", "-m", "seed")
    _git(work, "push", "-u", "origin", "main")
    local_head = _git(work, "rev-parse", "HEAD").stdout.strip()

    real_run(
        ["git", "clone", str(origin), str(publisher)],
        check=True,
        capture_output=True,
        text=True,
    )
    _git(publisher, "config", "user.name", "racing publisher")
    _git(publisher, "config", "user.email", "publisher@example.com")
    _write(work / "dashboard_data" / "finish_equity_live.csv", "fresh local\n")
    _write(work / "notes.txt", "unrelated staged work\n")
    _git(work, "add", "notes.txt")

    race = {"injected": False, "head": None}

    def run_with_remote_race(command, *args, **kwargs):
        if (
            not race["injected"]
            and command[:3] == ["git", "push", "origin"]
            and Path(kwargs.get("cwd", "")) == work
        ):
            race["injected"] = True
            _write(
                publisher / "dashboard_data" / "remote_only.csv",
                "must survive retry\n",
            )
            for git_args in (
                ("add", "dashboard_data/remote_only.csv"),
                ("commit", "-m", "concurrent dashboard update"),
                ("push", "origin", "main"),
            ):
                result = real_run(
                    [
                        "git", "-c", "core.hooksPath=", "-C", str(publisher),
                        *git_args,
                    ],
                    capture_output=True,
                    text=True,
                )
                assert result.returncode == 0, result.stderr or result.stdout
            race["head"] = real_run(
                ["git", "-C", str(publisher), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(dashboard_publish, "PROJECT_ROOT", str(work))
    monkeypatch.setattr(dashboard_publish, "get_tourney", lambda: None)
    monkeypatch.setattr(subprocess, "run", run_with_remote_race)

    dashboard_publish.git_push()

    assert race["injected"] is True
    _git(work, "fetch", "origin", "main")
    assert _git(work, "rev-parse", "origin/main^").stdout.strip() == race["head"]
    assert _git(
        work, "show", "origin/main:dashboard_data/remote_only.csv"
    ).stdout == "must survive retry\n"
    assert _git(
        work, "show", "origin/main:dashboard_data/finish_equity_live.csv"
    ).stdout == "fresh local\n"
    assert _git(work, "rev-parse", "HEAD").stdout.strip() == local_head
    assert _git(work, "diff", "--cached", "--name-only").stdout.strip() == "notes.txt"
