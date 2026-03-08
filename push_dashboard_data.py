"""Copy pipeline outputs into dashboard_data/ and push to trigger Render deploy.

Usage:
    python push_dashboard_data.py             # Copy, commit, push
    python push_dashboard_data.py --dry-run   # Preview only
"""

import os
import sys
import shutil
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DATA = os.path.join(PROJECT_ROOT, "dashboard_data")

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
    "this_week_dists_adjusted.csv",
]

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
]

# Parquet from permanent_data/
PERMANENT_FILES = [
    ("permanent_data/sg_diagnostic.parquet", "sg_diagnostic.parquet"),
    ("permanent_data/bet_ledger.parquet", "luck_attribution.parquet"),
]


def get_tourney():
    """Import tourney name from sim_inputs.py."""
    sys.path.insert(0, PROJECT_ROOT)
    try:
        import sim_inputs
        return getattr(sim_inputs, "tourney", None)
    except ImportError:
        return None


def copy_files(dry_run=False):
    """Copy pipeline outputs into dashboard_data/."""
    tourney = get_tourney()
    copied = []
    skipped = []

    os.makedirs(DASHBOARD_DATA, exist_ok=True)

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
        dst = os.path.join(DASHBOARD_DATA, fname)
        if not dry_run:
            shutil.copy2(src, dst)
        copied.append(label)

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

    return copied, skipped


def git_push(dry_run=False):
    """Stage dashboard_data/, commit, and push."""
    if dry_run:
        return

    os.system(f'cd "{PROJECT_ROOT}" && git add dashboard_data/')
    os.system(f'cd "{PROJECT_ROOT}" && git add sim_inputs.py')

    # Check if there are staged changes
    ret = os.system(f'cd "{PROJECT_ROOT}" && git diff --staged --quiet')
    if ret == 0:
        print("\n  No changes to commit.")
        return

    os.system(f'cd "{PROJECT_ROOT}" && git commit -m "Update dashboard data for Render deploy"')
    os.system(f'cd "{PROJECT_ROOT}" && git push')


def main():
    parser = argparse.ArgumentParser(description="Push dashboard data to trigger Render deploy")
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

    if not copied:
        print("\n  No files to copy. Nothing to push.")
        return

    if not args.dry_run:
        print("\n  Staging and pushing to GitHub...")
        git_push(dry_run=args.dry_run)
        print("  Done! Render will auto-deploy from the push.")
    else:
        print(f"\n  [DRY RUN] Would copy {len(copied)} files and push to GitHub.")


if __name__ == "__main__":
    main()
