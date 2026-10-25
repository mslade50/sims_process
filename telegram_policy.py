"""One Telegram page per failed workflow incident, persisted in an R2 object.

Failure-only CLI. A 24-hour cooldown bounds repeated CI failures; ordinary success
messages are never sent. State read/auth failures stay in Actions logs and exit 0.
A failed send exits non-zero. Stdio is forced to UTF-8 so the Windows runner can
carry non-ASCII alert text.
"""
import argparse
import json
import os
import re
import subprocess
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path


def force_utf8_stdio():
    """Make stdin/stdout/stderr UTF-8. Safe to call more than once."""
    os.environ["PYTHONIOENCODING"] = "utf-8"
    for name in ("stdin", "stdout", "stderr"):
        stream = getattr(sys, name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:
            pass


def needs_windows_utf8_reexec():
    """True when this Windows process was not started in UTF-8 mode."""
    return os.name == "nt" and os.environ.get("PYTHONUTF8") != "1"


def telegram_body(chat, message):
    """UTF-8 JSON body for sendMessage. Non-ASCII is kept, not escaped."""
    payload = {
        "chat_id": chat,
        "text": str(message)[:3900],
        "disable_web_page_preview": True,
    }
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def load_state(path):
    """Read the notification state; empty/corrupt/non-dict state counts as no state (SC-02)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        print("Notification state empty or unreadable; treating as no prior alert")
        return {}
    return data if isinstance(data, dict) else {}


def should_alert(previous, now):
    return now - float(previous.get("sent_at", 0)) >= 24 * 3600 and now - float(previous.get("reserved_at", 0)) >= 3600


def main():
    from etr_only_mode import refuse as _etr_refuse  # ETR-only mode guard
    _etr_refuse('Telegram telegram_policy.main')
    force_utf8_stdio()
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--message", required=True)
    args = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.key):
        raise SystemExit("invalid notification key")
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat or os.getenv("TELEGRAM_DISABLED") == "1":
        return 0
    key = "golf-odds-board/board/notifications/" + args.key + ".json"
    with tempfile.TemporaryDirectory(prefix="golf-alert-") as tmp:
        path = Path(tmp) / "state.json"
        def r2(verb):
            return subprocess.run([shutil.which("npx") or "npx", "--yes", "wrangler@4.92.0", "r2", "object", verb, key, "--remote", "--file=" + str(path)], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)
        result = r2("get")
        if result.returncode and not any(s in ((result.stdout or "") + (result.stderr or "")).lower() for s in ("404", "does not exist", "not found")):
            print("Notification state unavailable; inspect workflow logs")
            return 0
        previous = load_state(path) if path.exists() else {}
        now = time.time()
        if not should_alert(previous, now):
            print("Unchanged/recent workflow failure: Telegram suppressed")
            return 0
        path.write_text(json.dumps({**previous, "reserved_at": now}), encoding="utf-8")
        if r2("put").returncode:
            print("Notification reservation unavailable; inspect workflow logs")
            return 0
        try:
            request = urllib.request.Request(
                f"https://api.telegram.org/bot{token}/sendMessage",
                data=telegram_body(chat, args.message),
                headers={"Content-Type": "application/json; charset=utf-8"},
            )
            with urllib.request.urlopen(request, timeout=15) as response:
                if response.status != 200:
                    print("Telegram transport failed: HTTP", response.status)
                    return 1
        except Exception as ex:
            print("Telegram transport failed:", type(ex).__name__)
            return 1
        path.write_text(json.dumps({"sent_at": now}), encoding="utf-8")
        if r2("put").returncode:
            print("Notification state write failed; inspect Actions logs")
    return 0


if __name__ == "__main__":
    if needs_windows_utf8_reexec():
        os.environ["PYTHONUTF8"] = "1"
        os.execv(sys.executable, [sys.executable, "-X", "utf8", *sys.argv])
    sys.exit(main() or 0)
