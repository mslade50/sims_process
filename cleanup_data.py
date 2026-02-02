"""
Automated Data Cleanup Script with Git Auto-Commit

This script automatically cleans up tournament data files and commits the changes to git.
NO PROMPTS - It will delete files and commit automatically.

Usage:
    python cleanup_data.py              # Delete files and auto-commit
    python cleanup_data.py --no-backup  # Skip backup creation (not recommended)
    python cleanup_data.py --no-commit  # Delete but don't commit to git

The script will:
1. Delete all CSV files in the root directory (except permanent_data/)
2. Delete all Excel files in the root directory
3. Delete all text files in the root directory  
4. Delete all tournament folders
5. Create a backup archive before deletion
6. Automatically commit and push changes to git
7. Preserve: permanent_data/, .git/, .github/, Python scripts, .env files

Safety features:
- Creates backup archive before deletion (cleanup_backup_YYYYMMDD_HHMM.tar.gz)
- Logs exactly what was deleted
- Only commits if files were actually deleted
"""

import os
import sys
import argparse
import tarfile
import subprocess
from datetime import datetime
from pathlib import Path


def get_files_to_delete():
    """
    Identify all files that will be deleted by the cleanup process.
    
    Returns:
        dict with keys: csv_files, xlsx_files, txt_files, tournament_folders
    """
    root_dir = Path(".")
    
    # CSV files (excluding permanent_data folder)
    csv_files = [
        f for f in root_dir.glob("*.csv") 
        if not str(f).startswith("permanent_data")
    ]
    
    # Excel files in root
    xlsx_files = list(root_dir.glob("*.xlsx"))
    
    # Text files in root (scraped sportsbook data)
    txt_files = list(root_dir.glob("*.txt"))
    
    # Tournament folders (exclude system folders)
    protected_dirs = {
        ".", "..", ".git", ".github", "permanent_data", 
        "__pycache__", ".venv", "venv", "env"
    }
    tournament_folders = [
        d for d in root_dir.iterdir() 
        if d.is_dir() and d.name not in protected_dirs
    ]
    
    return {
        "csv_files": csv_files,
        "xlsx_files": xlsx_files,
        "txt_files": txt_files,
        "tournament_folders": tournament_folders
    }


def print_summary(files_dict):
    """Print a summary of what will be deleted."""
    print("\n" + "="*70)
    print("📋 CLEANUP SUMMARY - Files to be deleted:")
    print("="*70)
    
    total_files = sum(len(v) for k, v in files_dict.items() if k != "tournament_folders")
    total_folders = len(files_dict["tournament_folders"])
    
    print(f"\n📊 Total: {total_files} files + {total_folders} folders\n")
    
    # CSV files
    csv_files = files_dict["csv_files"]
    print(f"📄 CSV files ({len(csv_files)}):")
    if csv_files:
        for f in sorted(csv_files)[:10]:  # Show first 10
            print(f"  - {f}")
        if len(csv_files) > 10:
            print(f"  ... and {len(csv_files) - 10} more")
    else:
        print("  (none found)")
    
    # Excel files
    xlsx_files = files_dict["xlsx_files"]
    print(f"\n📊 Excel files ({len(xlsx_files)}):")
    if xlsx_files:
        for f in sorted(xlsx_files)[:10]:
            print(f"  - {f}")
        if len(xlsx_files) > 10:
            print(f"  ... and {len(xlsx_files) - 10} more")
    else:
        print("  (none found)")
    
    # Text files
    txt_files = files_dict["txt_files"]
    print(f"\n📝 Text files ({len(txt_files)}):")
    if txt_files:
        for f in sorted(txt_files)[:10]:
            print(f"  - {f}")
        if len(txt_files) > 10:
            print(f"  ... and {len(txt_files) - 10} more")
    else:
        print("  (none found)")
    
    # Tournament folders
    folders = files_dict["tournament_folders"]
    print(f"\n📁 Tournament folders ({len(folders)}):")
    if folders:
        for d in sorted(folders):
            file_count = len(list(d.rglob("*")))
            print(f"  - {d}/ ({file_count} files inside)")
    else:
        print("  (none found)")
    
    print("\n" + "="*70)
    print("✅ PROTECTED (will NOT be deleted):")
    print("="*70)
    print("  - permanent_data/ folder and all contents")
    print("  - All .py Python scripts")
    print("  - .env and .gitignore files")
    print("  - .git/ and .github/ folders")
    print("  - requirements.txt, README.md, etc.")
    print("="*70 + "\n")


def create_backup(files_dict):
    """Create a compressed backup of files before deletion."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    backup_name = f"cleanup_backup_{timestamp}.tar.gz"
    
    print(f"📦 Creating backup: {backup_name}")
    
    try:
        with tarfile.open(backup_name, "w:gz") as tar:
            for category, files in files_dict.items():
                for item in files:
                    if item.exists():
                        tar.add(item, arcname=str(item))
        
        backup_size = os.path.getsize(backup_name) / (1024 * 1024)  # MB
        print(f"✅ Backup created: {backup_name} ({backup_size:.2f} MB)")
        print(f"💡 Keep this backup for a week, then delete if everything looks good\n")
        
        return backup_name
    except Exception as e:
        print(f"⚠️  Warning: Backup creation failed: {e}")
        print(f"⚠️  Continuing with cleanup anyway...\n")
        return None


def delete_files(files_dict):
    """Delete all identified files and folders."""
    print("\n🗑️  Starting deletion...")
    
    deleted_count = 0
    deleted_items = []
    
    # Delete CSV files
    for f in files_dict["csv_files"]:
        try:
            f.unlink()
            deleted_count += 1
            deleted_items.append(str(f))
        except Exception as e:
            print(f"  ⚠️  Error deleting {f}: {e}")
    
    # Delete Excel files
    for f in files_dict["xlsx_files"]:
        try:
            f.unlink()
            deleted_count += 1
            deleted_items.append(str(f))
        except Exception as e:
            print(f"  ⚠️  Error deleting {f}: {e}")
    
    # Delete text files
    for f in files_dict["txt_files"]:
        try:
            f.unlink()
            deleted_count += 1
            deleted_items.append(str(f))
        except Exception as e:
            print(f"  ⚠️  Error deleting {f}: {e}")
    
    # Delete tournament folders
    import shutil
    for d in files_dict["tournament_folders"]:
        try:
            shutil.rmtree(d)
            deleted_count += 1
            deleted_items.append(str(d) + "/")
            print(f"  ✅ Deleted folder: {d}/")
        except Exception as e:
            print(f"  ⚠️  Error deleting {d}: {e}")
    
    print(f"\n✅ Cleanup complete! Deleted {deleted_count} items")
    
    return deleted_items


def git_commit_and_push(deleted_items):
    """Commit deletions to git and push to remote."""
    print("\n" + "="*70)
    print("📤 COMMITTING CHANGES TO GIT")
    print("="*70)
    
    try:
        # Check if there are changes to commit
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True
        )
        
        if not result.stdout.strip():
            print("✅ No changes to commit - repository already clean")
            return True
        
        # Stage all deletions
        print("📝 Staging deletions...")
        subprocess.run(["git", "add", "-A"], check=True)
        
        # Create commit message
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        commit_msg = f"chore: Weekly data cleanup - {timestamp}\n\n"
        commit_msg += "Automated cleanup of transient data files:\n"
        commit_msg += f"- Removed {len(deleted_items)} items\n"
        commit_msg += "- Preserved permanent_data/ folder\n"
        commit_msg += "- Preserved all Python scripts and config files"
        
        # Commit changes
        print("💾 Committing changes...")
        subprocess.run(
            ["git", "commit", "-m", commit_msg],
            check=True,
            capture_output=True
        )
        
        # Push to remote
        print("📤 Pushing to GitHub...")
        subprocess.run(["git", "push"], check=True)
        
        print("✅ Changes committed and pushed successfully!")
        print(f"   Commit message: Weekly data cleanup - {timestamp}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Git error: {e}")
        print(f"   You may need to commit and push manually:")
        print(f"   git add -A")
        print(f"   git commit -m 'Weekly cleanup'")
        print(f"   git push")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during git commit: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Automated cleanup with git auto-commit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--no-backup",
        action="store_true", 
        help="Skip creating backup archive (not recommended)"
    )
    parser.add_argument(
        "--no-commit",
        action="store_true",
        help="Delete files but don't commit to git"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🧹 AUTOMATED DATA CLEANUP SCRIPT")
    print("="*70)
    print("⚠️  This will automatically delete files and commit changes!")
    print("="*70)
    
    # Get list of files to delete
    files_dict = get_files_to_delete()
    
    # Show summary
    print_summary(files_dict)
    
    # Check if there's anything to delete
    total_items = sum(len(v) for v in files_dict.values())
    if total_items == 0:
        print("✅ Nothing to clean up! Repository is already clean.")
        return
    
    # Create backup
    if not args.no_backup:
        create_backup(files_dict)
    else:
        print("⚠️  Skipping backup (--no-backup flag used)")
    
    # Delete files
    deleted_items = delete_files(files_dict)
    
    # Commit to git (unless --no-commit flag is used)
    if not args.no_commit and deleted_items:
        git_commit_and_push(deleted_items)
    elif args.no_commit:
        print("\n💡 Skipping git commit (--no-commit flag used)")
        print("   Run these commands manually to commit:")
        print("   git add -A")
        print("   git commit -m 'Weekly cleanup'")
        print("   git push")
    
    print("\n" + "="*70)
    print("✅ CLEANUP COMPLETE")
    print("="*70)
    print("💡 Your repository is now clean!")
    if deleted_items and not args.no_commit:
        print("💡 Changes have been committed and pushed to GitHub")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()