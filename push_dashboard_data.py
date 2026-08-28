"""Copy pipeline outputs, sync GitHub, and publish dashboard JSON to Cloudflare.

Usage:
    python push_dashboard_data.py             # Copy, commit, push
    python push_dashboard_data.py --dry-run   # Preview only
"""

import os
import sys
import json
import shutil
import argparse
from datetime import date

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DATA = os.path.join(PROJECT_ROOT, "dashboard_data")
SYNC_MANIFEST = os.path.join(DASHBOARD_DATA, ".sync_manifest.json")
CLOUDFLARE_PUBLISHER = os.path.join(
    PROJECT_ROOT, "cloudflare_dashboard", "scripts", "publish_dashboard_data.py"
)

# Files to copy: (source_pattern, dest_name_or_None)
# source_pattern is relative to PROJECT_ROOT; {tourney} gets substituted
# dest_name is the name in dashboard_data/ (None = keep original name)

ROOT_FILES = [
    "model_predictions_r1.csv",
    "model_predictions_r2.csv",
    "model_predictions_r3.csv",
    "model_predictions_r4.csv",
    "r1_live_model.csv",
    "r2_live_model.csv",
    "r3_live_model.csv",
    "r4_live_model.csv",
    "simulated_probs_live.csv",
    "simulated_probs.csv",
    "sg_dist_player.csv",
    "this_week_dists_v2.csv",
]

# Files only produced DURING an event (Thu-Sun). A copy whose mtime predates
# the current event's pre-event sim is a previous week's leftover: copying it
# would attribute it to the new event in the sync manifest, and the post-merge
# hook would then reinstall last week's live state as this week's on every
# consuming machine (2026-08-12: wyndham live models stamped as st_jude).
LIVE_ROUND_PREFIXES = (
    "model_predictions_r",
    "r1_live_model", "r2_live_model", "r3_live_model", "r4_live_model",
    "simulated_probs_live",
)

# Files in the tournament folder → flattened into dashboard_data/
TOURNEY_FILES = [
    "outright_win_edges.csv",
    "betonline_devig_fades.csv",
    "fair_card_r2.csv",
    "fair_card_r3.csv",
    "fair_card_r4.csv",
]

# Files with tourney-specific names → renamed for dashboard_data/
TOURNEY_RENAMED = [
    # (source_template, dest_name) — {tourney} substituted in source
    ("finish_equity_live_{tourney}.csv", "finish_equity_live.csv"),
    ("finish_equity_{tourney}.csv", "finish_equity_pre.csv"),
    ("rank_probs_updated_{tourney}.parquet", "rank_probs_pre.parquet"),
    ("rank_probs_live_{tourney}.parquet", "rank_probs_live.parquet"),
    ("h2h_matrix_{tourney}.parquet", "h2h_matrix.parquet"),
    ("weather_impact_{tourney}.csv", "weather_impact.csv"),
]

# Round score distribution files (pre-aggregated, ~200KB each)
ROUND_SCORE_PROBS = [
    "round_score_probs_r1.parquet",
    "round_score_probs_r2.parquet",
    "round_score_probs_r3.parquet",
    "round_score_probs_r4.parquet",
]

# Parquet from permanent_data/
PERMANENT_FILES = [
    ("permanent_data/sg_diagnostic.parquet", "sg_diagnostic.parquet"),
]


HISTORICAL_DISTS = os.path.join(PROJECT_ROOT, "permanent_data", "historical_dists")


def get_tourney():
    """Import tourney name from sim_inputs.py."""
    sys.path.insert(0, PROJECT_ROOT)
    try:
        import sim_inputs
        return getattr(sim_inputs, "tourney", None)
    except ImportError:
        return None


def get_event_id():
    """Import first event_id from sim_inputs.py."""
    sys.path.insert(0, PROJECT_ROOT)
    try:
        import sim_inputs
        ids = getattr(sim_inputs, "event_ids", [])
        return ids[0] if ids else None
    except (ImportError, IndexError):
        return None


def _write_sync_manifest(root_files, tourney, event_id, dry_run=False):
    """Record which ROOT_FILES vintages dashboard_data/ holds for which event.

    sync_event_files.py (the post-merge hook) only resyncs a root pred file
    from dashboard_data/ when this manifest attributes that exact file to the
    current event — so a pull can never reinstall a previous week's files
    (2026-08-10: a stale wyndham model_predictions_r1.csv resurrected by the
    hook silently aborted the sim-fairs publish). Same-event pushes union
    their file lists — later pushes add files as they start existing; an
    event change resets the manifest.
    """
    if dry_run or not tourney or not root_files:
        return
    manifest = {
        "event": str(tourney).lower(),
        "event_id": event_id,
        "year": date.today().year,
        "files": [],
    }
    try:
        with open(SYNC_MANIFEST) as f:
            prior = json.load(f)
        if (
            str(prior.get("event", "")).strip().lower() == manifest["event"]
            and prior.get("year") == manifest["year"]
        ):
            manifest["files"] = [str(name) for name in prior.get("files", [])]
    except (OSError, ValueError):
        pass
    manifest["files"] = sorted(set(manifest["files"]) | set(root_files))
    with open(SYNC_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)


def copy_files(dry_run=False):
    """Copy pipeline outputs into dashboard_data/."""
    tourney = get_tourney()
    copied = []
    skipped = []
    root_copied = []

    os.makedirs(DASHBOARD_DATA, exist_ok=True)

    # Staleness anchor for live-round files: the current event's pre-event sim
    # output. Fail-open (no guard) if it doesn't exist yet.
    anchor = os.path.join(PROJECT_ROOT, f"final_predictions_{tourney}.csv") if tourney else None
    anchor_mtime = os.path.getmtime(anchor) if anchor and os.path.exists(anchor) else None

    # Root-level files (prefer v2 output if available)
    for fname in ROOT_FILES:
        v2_src = os.path.join(PROJECT_ROOT, tourney, "v2", fname) if tourney else None
        root_src = os.path.join(PROJECT_ROOT, fname)
        if v2_src and os.path.exists(v2_src):
            src = v2_src
            label = f"{tourney}/v2/{fname}"
        elif os.path.exists(root_src):
            src = root_src
            label = fname
        else:
            skipped.append(fname)
            continue
        if (
            anchor_mtime is not None
            and fname.startswith(LIVE_ROUND_PREFIXES)
            and os.path.getmtime(src) < anchor_mtime
        ):
            skipped.append(f"{fname} (stale: predates {tourney} pre-event sim)")
            continue
        dst = os.path.join(DASHBOARD_DATA, fname)
        if not dry_run:
            shutil.copy2(src, dst)
        copied.append(label)
        root_copied.append(fname)

    # Tournament folder files
    if tourney:
        for fname in TOURNEY_FILES:
            src = os.path.join(PROJECT_ROOT, tourney, fname)
            dst = os.path.join(DASHBOARD_DATA, fname)
            if os.path.exists(src):
                if not dry_run:
                    shutil.copy2(src, dst)
                copied.append(f"{tourney}/{fname}")
            else:
                skipped.append(f"{tourney}/{fname}")

        for src_template, dst_name in TOURNEY_RENAMED:
            src_fname = src_template.format(tourney=tourney)
            # Prefer v2 output if available
            v2_src = os.path.join(PROJECT_ROOT, tourney, "v2", src_fname)
            root_src = os.path.join(PROJECT_ROOT, src_fname)
            if os.path.exists(v2_src):
                src = v2_src
                label = f"{tourney}/v2/{src_fname} -> {dst_name}"
            elif os.path.exists(root_src):
                src = root_src
                label = f"{src_fname} -> {dst_name}"
            else:
                skipped.append(src_fname)
                continue
            dst = os.path.join(DASHBOARD_DATA, dst_name)
            if not dry_run:
                shutil.copy2(src, dst)
            copied.append(label)
    else:
        print("  Warning: Could not import tourney from sim_inputs.py — skipping tournament folder files")

    # Round score distribution files (from tournament folder)
    if tourney:
        found_any = False
        for fname in ROUND_SCORE_PROBS:
            src = os.path.join(PROJECT_ROOT, tourney, fname)
            if os.path.exists(src):
                found_any = True
                dst = os.path.join(DASHBOARD_DATA, fname)
                if not dry_run:
                    shutil.copy2(src, dst)
                copied.append(f"{tourney}/{fname}")
        if not found_any:
            skipped.append("round_score_probs_r*.parquet (none found)")

    # Archive rank_probs into permanent_data/historical_dists/
    event_id = get_event_id()
    if tourney and event_id:
        archive_dir = os.path.join(HISTORICAL_DISTS, f"{event_id}_{tourney}")
        for src_template, archive_name in [
            ("rank_probs_updated_{tourney}.parquet", "rank_probs_pre.parquet"),
            ("rank_probs_live_{tourney}.parquet", "rank_probs_live.parquet"),
        ]:
            src_fname = src_template.format(tourney=tourney)
            # Prefer v2 output
            v2_src = os.path.join(PROJECT_ROOT, tourney, "v2", src_fname)
            root_src = os.path.join(PROJECT_ROOT, src_fname)
            if os.path.exists(v2_src):
                src = v2_src
            elif os.path.exists(root_src):
                src = root_src
            else:
                skipped.append(f"archive: {src_fname}")
                continue
            if not dry_run:
                os.makedirs(archive_dir, exist_ok=True)
                shutil.copy2(src, os.path.join(archive_dir, archive_name))
            copied.append(f"archive: {src_fname} -> historical_dists/{event_id}_{tourney}/{archive_name}")
    else:
        print("  Warning: Could not determine event_id — skipping rank_probs archive")

    # Permanent data files
    for src_rel, dst_name in PERMANENT_FILES:
        src = os.path.join(PROJECT_ROOT, src_rel)
        dst = os.path.join(DASHBOARD_DATA, dst_name)
        if os.path.exists(src):
            if not dry_run:
                shutil.copy2(src, dst)
            copied.append(f"{src_rel} -> {dst_name}")
        else:
            skipped.append(src_rel)

    _write_sync_manifest(root_copied, tourney, event_id, dry_run=dry_run)

    return copied, skipped


def git_push(dry_run=False):
    """Publish ONLY dashboard-deploy files on top of the latest ``origin/main``.

    The commit is built with temporary Git indexes, so a preceding plumbing
    publisher may advance the remote without advancing this checkout.  No pull,
    rebase, branch move, or mutation of the ordinary index is required.  Only
    changes to the explicit dashboard pathspec are overlaid on the remote tip;
    unrelated local staging and remote-only dashboard changes are preserved.
    """
    import subprocess
    import tempfile
    import time

    if dry_run:
        return

    def _run(cmd, *, env=None):
        return subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env=env,
        )

    # The exact set of paths this deploy commit is allowed to touch.
    # historical_dists is included so the per-event rank_probs archives written
    # above actually publish — they're the cross-machine/dashboard copy; without
    # this they stay local to whichever machine ran the deploy that week.
    paths = ["dashboard_data", "sim_inputs.py", "permanent_data/historical_dists"]
    tourney = get_tourney()
    if tourney:
        prep_files = [
            f"pre_course_fit_{tourney}.csv",
            f"pre_sim_summary_{tourney}.csv",
            f"final_predictions_{tourney}.csv",
            "this_week_dists_v2.csv",
            "sg_dist_player.csv",
            f"permanent_data/avg_expected_cat_sg_{tourney}.csv",
        ]
        paths += [f for f in prep_files
                  if os.path.exists(os.path.join(PROJECT_ROOT, f))]

    # Commit/push failures alert loudly: this commit carries the cross-machine
    # skill-update state (model_predictions_r{N} / r{N}_live_model) — if it
    # silently doesn't land, another machine or nightly CI runs the next round
    # on stale predictions.
    def _fail(msg):
        print(f"  ERROR: {msg}")
        try:
            from maker_alerts import send_telegram
            if not send_telegram(f"[push_dashboard_data] {msg} — dashboard_data on origin "
                                 f"is STALE; next-round sims elsewhere will use old state"):
                print("  (telegram alert not sent: TELEGRAM_* env missing or send failed)")
        except Exception as e:
            print(f"  (telegram alert failed: {e})")
        # A swallowed push failure kept monday-grading green while two weeks of
        # diagnostics were silently lost (2026-08). Fail the process.
        sys.exit(1)

    def _result_error(result):
        return (result.stderr or result.stdout or "unknown git error").strip()[:240]

    # Snapshot exactly the worktree changes under our pathspec relative to the
    # checked-out commit.  A separate temporary index means unrelated staged work
    # is neither read nor changed.  The resulting blobs remain stable if a push
    # race requires rebuilding the commit on a newer remote parent.
    local_head = _run(["git", "rev-parse", "HEAD"])
    if local_head.returncode != 0 or not local_head.stdout.strip():
        _fail(f"git rev-parse HEAD failed: {_result_error(local_head)}")
    local_head_sha = local_head.stdout.strip()
    snapshot_index = os.path.join(
        tempfile.gettempdir(),
        f"dashboard_snapshot_index_{os.getpid()}_{time.time_ns()}",
    )
    snapshot_env = {**os.environ, "GIT_INDEX_FILE": snapshot_index}
    overlay = {}
    try:
        read = _run(["git", "read-tree", local_head_sha], env=snapshot_env)
        if read.returncode != 0:
            _fail(f"git read-tree failed: {_result_error(read)}")
        stage = _run(["git", "add", "-A", "-f", "--", *paths], env=snapshot_env)
        if stage.returncode != 0:
            _fail(f"git add failed: {_result_error(stage)}")
        changed = _run(
            [
                "git", "diff", "--cached", "--name-only", "-z",
                local_head_sha, "--", *paths,
            ],
            env=snapshot_env,
        )
        if changed.returncode != 0:
            _fail(f"git diff failed: {_result_error(changed)}")
        for path in (name for name in changed.stdout.split("\0") if name):
            entry = _run(
                ["git", "ls-files", "--stage", "--", path],
                env=snapshot_env,
            )
            if entry.returncode != 0:
                _fail(f"git ls-files failed for {path}: {_result_error(entry)}")
            if not entry.stdout.strip():
                overlay[path] = None
                continue
            fields = entry.stdout.split("\t", 1)[0].split()
            if len(fields) != 3 or fields[2] != "0":
                _fail(f"git index entry is invalid for {path}")
            overlay[path] = (fields[0], fields[1])
    finally:
        try:
            if os.path.exists(snapshot_index):
                os.remove(snapshot_index)
        except OSError:
            pass

    if not overlay:
        print("\n  No dashboard changes to commit.")
        return

    def _commit_on(base):
        publish_index = os.path.join(
            tempfile.gettempdir(),
            f"dashboard_publish_index_{os.getpid()}_{time.time_ns()}",
        )
        publish_env = {**os.environ, "GIT_INDEX_FILE": publish_index}
        try:
            read = _run(["git", "read-tree", base], env=publish_env)
            if read.returncode != 0:
                _fail(f"git read-tree origin failed: {_result_error(read)}")
            for path, entry in overlay.items():
                if entry is None:
                    update = _run(
                        ["git", "update-index", "--force-remove", "--", path],
                        env=publish_env,
                    )
                else:
                    mode, blob = entry
                    update = _run(
                        [
                            "git", "update-index", "--add", "--cacheinfo",
                            f"{mode},{blob},{path}",
                        ],
                        env=publish_env,
                    )
                if update.returncode != 0:
                    _fail(f"git update-index failed for {path}: {_result_error(update)}")
            diff = _run(
                ["git", "diff", "--cached", "--quiet", base, "--", *overlay],
                env=publish_env,
            )
            if diff.returncode == 0:
                return None
            if diff.returncode != 1:
                _fail(f"git diff against origin failed: {_result_error(diff)}")
            tree = _run(["git", "write-tree"], env=publish_env)
            if tree.returncode != 0 or not tree.stdout.strip():
                _fail(f"git write-tree failed: {_result_error(tree)}")
            commit = _run(
                [
                    "git", "commit-tree", tree.stdout.strip(), "-p", base,
                    "-m", "Update dashboard data for Cloudflare",
                ]
            )
            if commit.returncode != 0 or not commit.stdout.strip():
                # On CI this usually means the workflow omitted GIT_* identity.
                _fail(f"git commit-tree failed: {_result_error(commit)}")
            return commit.stdout.strip()
        finally:
            try:
                if os.path.exists(publish_index):
                    os.remove(publish_index)
            except OSError:
                pass

    last_error = ""
    for attempt in range(1, 4):
        fetch = _run(["git", "fetch", "origin", "main"])
        if fetch.returncode != 0:
            last_error = _result_error(fetch)
            if attempt < 3:
                continue
            _fail(f"git fetch failed after 3 attempts: {last_error}")
        base = _run(["git", "rev-parse", "FETCH_HEAD"])
        if base.returncode != 0 or not base.stdout.strip():
            _fail(f"git rev-parse FETCH_HEAD failed: {_result_error(base)}")
        commit_sha = _commit_on(base.stdout.strip())
        if commit_sha is None:
            print("\n  Dashboard changes are already current on origin/main.")
            return
        result = _run(["git", "push", "origin", f"{commit_sha}:main"])
        if result.returncode == 0:
            return

        last_error = _result_error(result)
        verify = _run(["git", "fetch", "origin", "main"])
        if verify.returncode == 0:
            remote = _run(["git", "rev-parse", "FETCH_HEAD"])
            remote_sha = remote.stdout.strip() if remote.returncode == 0 else ""
            if remote_sha and (
                remote_sha == commit_sha
                or _run(
                    ["git", "merge-base", "--is-ancestor", commit_sha, remote_sha]
                ).returncode == 0
            ):
                print("  Dashboard push was accepted despite a client error.")
                return

        rejection = (result.stderr or result.stdout or "").lower()
        if "rejected" in rejection or "non-fast-forward" in rejection:
            print(
                f"  Push rejected (attempt {attempt}/3); rebuilding the scoped "
                "commit on latest origin/main..."
            )
            continue
        _fail(f"git push failed: {last_error}")

    _fail(f"git push failed after 3 attempts: {last_error}")


def publish_cloudflare(dry_run=False):
    """Export the browser JSON contract and publish it to the dashboard's R2 bucket."""
    import subprocess

    command = [sys.executable, CLOUDFLARE_PUBLISHER]
    if dry_run:
        command.append("--dry-run")
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    if result.returncode == 0:
        return

    msg = f"Cloudflare dashboard publish failed with exit code {result.returncode}"
    print(f"  ERROR: {msg}")
    try:
        from maker_alerts import send_telegram
        if not send_telegram(
            f"[push_dashboard_data] {msg} — the private Cloudflare dashboard is STALE"
        ):
            print("  (telegram alert not sent: TELEGRAM_* env missing or send failed)")
    except Exception as exc:
        print(f"  (telegram alert failed: {exc})")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Publish dashboard data to GitHub and Cloudflare")
    parser.add_argument("--dry-run", action="store_true", help="Preview files without copying or pushing")
    args = parser.parse_args()

    mode = "DRY RUN" if args.dry_run else "LIVE"
    print(f"\n  push_dashboard_data.py [{mode}]")
    print("  " + "=" * 50)

    copied, skipped = copy_files(dry_run=args.dry_run)

    if copied:
        print(f"\n  Copied ({len(copied)}):")
        for f in copied:
            print(f"    + {f}")

    if skipped:
        print(f"\n  Skipped (not found) ({len(skipped)}):")
        for f in skipped:
            print(f"    - {f}")

    if copied and not args.dry_run:
        print("\n  Staging and pushing to GitHub...")
        git_push(dry_run=args.dry_run)
    elif not copied:
        print("\n  No pipeline files to copy; refreshing Cloudflare from current sources.")
    else:
        print(f"\n  [DRY RUN] Would copy {len(copied)} files and push to GitHub.")

    print("\n  Publishing browser JSON to Cloudflare...")
    publish_cloudflare(dry_run=args.dry_run)
    if not args.dry_run:
        print("  Done! The private Cloudflare dashboard data is current.")


if __name__ == "__main__":
    main()
