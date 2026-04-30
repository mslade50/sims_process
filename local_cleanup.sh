#!/bin/bash
# Local cleanup script — mirrors the GitHub Actions weekly-cleanup.yml
# but for the local OneDrive working directory.
#
# Usage:
#   bash local_cleanup.sh          # dry-run (preview only)
#   bash local_cleanup.sh --delete  # actually delete files

set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DRY_RUN=true
if [[ "${1:-}" == "--delete" ]]; then
    DRY_RUN=false
fi

deleted=0

remove() {
    local path="$1"
    if $DRY_RUN; then
        echo "  [dry-run] would delete: $path"
    else
        rm -rf "$path"
        echo "  deleted: $path"
    fi
    deleted=$((deleted + 1))
}

echo "=========================================="
if $DRY_RUN; then
    echo "  LOCAL CLEANUP — DRY RUN (pass --delete to execute)"
else
    echo "  LOCAL CLEANUP — DELETING FILES"
fi
echo "=========================================="
echo ""

# --- Root CSV files ---
echo "CSV files:"
for f in *.csv; do
    remove "$f"
done
echo ""

# --- Root Excel files ---
echo "Excel files:"
for f in *.xlsx; do
    remove "$f"
done
echo ""

# --- Root text files (protect requirements.txt) ---
echo "Text files:"
for f in *.txt; do
    [[ "$f" == "requirements.txt" ]] && continue
    remove "$f"
done
echo ""

# --- Root parquet files ---
echo "Parquet files:"
for f in *.parquet; do
    remove "$f"
done
echo ""

# --- Root PDF files ---
echo "PDF files:"
for f in *.pdf; do
    remove "$f"
done
echo ""

# --- Root JSON files (model coefficients etc, but not .claude/) ---
echo "JSON files:"
for f in *.json; do
    remove "$f"
done
echo ""

# --- Tournament folders (same exclusions as GH Actions cleanup) ---
PROTECTED_DIRS=".git .github .claude permanent_data backups __pycache__ dashboard dashboard_data archive"

echo "Tournament folders:"
for d in */; do
    [ -d "$d" ] || continue
    dirname="${d%/}"
    skip=false
    for p in $PROTECTED_DIRS; do
        if [[ "$dirname" == "$p" ]]; then
            skip=true
            break
        fi
    done
    $skip && continue
    remove "$d"
done
echo ""

# --- Stale Windows artifacts ---
[ -e "nul" ] && remove "nul"

echo "=========================================="
echo "  Total: $deleted items"
if $DRY_RUN; then
    echo "  Re-run with --delete to execute"
fi
echo "=========================================="
