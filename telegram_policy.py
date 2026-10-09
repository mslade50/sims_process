"""One Telegram page per failed workflow incident, persisted in an R2 object.

Failure-only CLI. A 24-hour cooldown bounds repeated CI failures; ordinary success
messages are never sent. State read/auth failures stay in Actions logs.
"""
import argparse
import json
import os
import re
import subprocess
import shutil
import tempfile
import time
import urllib.request
from pathlib import Path


def should_alert(previous, now):
    return now - float(previous.get("sent_at", 0)) >= 24 * 3600 and now - float(previous.get("reserved_at", 0)) >= 3600


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--message", required=True)
    args = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.key):
        raise SystemExit("invalid notification key")
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat or os.getenv("TELEGRAM_DISABLED") == "1":
        return
    key = "golf-odds-board/board/notifications/" + args.key + ".json"
    with tempfile.TemporaryDirectory(prefix="golf-alert-") as tmp:
        path = Path(tmp) / "state.json"
        def r2(verb):
            return subprocess.run([shutil.which("npx") or "npx", "--yes", "wrangler@4.92.0", "r2", "object", verb, key, "--remote", "--file=" + str(path)], capture_output=True, text=True, timeout=90)
        result = r2("get")
        if result.returncode and not any(s in (result.stdout + result.stderr).lower() for s in ("404", "does not exist", "not found")):
            print("Notification state unavailable; inspect workflow logs")
            return
        previous = json.loads(path.read_text()) if path.exists() else {}
        now = time.time()
        if not should_alert(previous, now):
            print("Unchanged/recent workflow failure: Telegram suppressed")
            return
        path.write_text(json.dumps({**previous, "reserved_at": now}))
        if r2("put").returncode:
            print("Notification reservation unavailable; inspect workflow logs")
            return
        try:
            request = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage", data=json.dumps({"chat_id": chat, "text": args.message[:3900], "disable_web_page_preview": True}).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=15) as response:
                if response.status != 200:
                    return
        except Exception as ex:
            print("Telegram transport failed:", type(ex).__name__)
            return
        path.write_text(json.dumps({"sent_at": now}))
        if r2("put").returncode:
            print("Notification state write failed; inspect Actions logs")


if __name__ == "__main__":
    main()
