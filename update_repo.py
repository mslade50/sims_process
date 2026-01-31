import os
import subprocess
from datetime import datetime

# configuration
REPO_PATH = r"C:\Users\mckin\OneDrive\sims_process"

def run_git_command(commands, cwd):
    """Runs a git command and returns the output/success status."""
    try:
        result = subprocess.run(
            commands,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        return True, result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip()

def update_repo():
    print("\n" + "="*50)
    print(f"GIT REPO UPDATER")
    print(f"Target: {REPO_PATH}")
    print("="*50)

    # 1. Verify it is a valid repo
    if not os.path.exists(os.path.join(REPO_PATH, ".git")):
        print(f"[error] No .git folder found in {REPO_PATH}")
        print("       Did you run 'git init' yet?")
        return

    # 2. Stage all files
    print("[1/3] Staging changes...")
    success, output = run_git_command(["git", "add", "."], REPO_PATH)
    if not success:
        print(f"[error] git add failed: {output}")
        return
    print("      Done.")

    # 3. Commit changes
    print("[2/3] Committing...")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    commit_msg = f"Repo update: {timestamp}"
    
    # We don't use check=True here because 'nothing to commit' returns a non-zero exit code
    result = subprocess.run(
        ["git", "commit", "-m", commit_msg],
        cwd=REPO_PATH,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"[success] Committed: {commit_msg}")
        print(f"          {result.stdout.splitlines()[0]}") # Print first line of git output
    elif "nothing to commit" in result.stdout:
        print("[info] No new changes to commit.")
    else:
        print(f"[error] Commit failed: {result.stderr}")

    # 4. Push to remote (Optional)
    print("[3/3] Pushing to remote...")
    success, output = run_git_command(["git", "push"], REPO_PATH)
    if success:
        print("[success] Push complete.")
    else:
        if "No configured push destination" in output:
            print("[info] No remote server (GitHub) configured. Changes are saved LOCALLY only.")
        else:
            print(f"[warn] Push failed (you might need to pull first?):")
            print(f"       {output}")

    print("\n" + "="*50)
    print("Update Complete.")

if __name__ == "__main__":
    update_repo()
    input("\nPress Enter to close...")