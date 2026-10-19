"""golfprice ownership of sim_fairs.json / the odds board (SC-01/03/08).

Stdlib only, so cheap CI steps can call it before any heavy import or full sim.

SIM_FAIRS_EXTERNAL.json (committed by `python -m golfprice.board publish`) says another
publisher owns the board.  Optional fields (all backward compatible):

    owner       publisher name                      (required to mean anything)
    event_id    event the owner currently publishes (string)
    event_ids   extra event ids it covers           (list)
    since       "YYYY-MM-DD HH:MM:SS UTC"
    expires_at  "YYYY-MM-DD HH:MM:SS UTC" or ISO8601; ownership is stale after this.
                Without it, age is measured from `since` against
                SIM_FAIRS_EXTERNAL_MAX_AGE_HOURS (default 240).

Decision (resolve_ownership):
    none            no marker -> production owns the file, nothing stands down
    owned           marker for this event and fresh -> stand down
    stale_event     marker names a different event -> still stand down (never overwrite
                    golfprice's board by accident) but LOUD
    expired         marker older than its expiry -> still stand down, LOUD
    unreadable      marker/ref could not be read -> fail CLOSED (stand down), LOUD
Stale/expired/unreadable resume production only with SIM_FAIRS_OWNERSHIP_RESUME_ON_STALE=1.

Kill switches:
    SIM_FAIRS_FORCE_PRODUCTION=1     production owns everything (manual rollback, pre-existing)
    SIM_FAIRS_OWNERSHIP_LEGACY=1     old presence-only behaviour (callers fall back to it)
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

MARKER = "SIM_FAIRS_EXTERNAL.json"
ROOT = Path(__file__).resolve().parent
DEFAULT_MAX_AGE_HOURS = 240.0
_TRUE = ("1", "true", "yes")


def _flag(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in _TRUE


def legacy_mode() -> bool:
    return _flag("SIM_FAIRS_OWNERSHIP_LEGACY")


@dataclass
class Ownership:
    stand_down: bool
    status: str          # none | owned | stale_event | expired | unreadable | forced_production
    owner: str | None
    detail: str

    def log_line(self) -> str:
        return f"[ownership] status={self.status} stand_down={self.stand_down} owner={self.owner} :: {self.detail}"


def _parse_ts(value):
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S UTC", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            dt = datetime.strptime(text, fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _git(root: Path, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=60)


def read_marker(root: Path = ROOT):
    """Return (state, payload_text). state: 'absent' | 'present' | 'error'.
    origin/main is authoritative when reachable; the local file is the fallback. A git
    failure is an error, never 'absent' (the old guard failed open on any git error)."""
    root = Path(root)
    try:
        _git(root, "fetch", "origin", "main")
        listed = _git(root, "ls-tree", "--name-only", "origin/main", "--", MARKER)
        if listed.returncode == 0:
            if not listed.stdout.strip():
                return "absent", ""
            shown = _git(root, "show", f"origin/main:{MARKER}")
            if shown.returncode == 0:
                return "present", shown.stdout
    except (OSError, subprocess.TimeoutExpired):
        pass
    local = root / MARKER
    try:
        if local.exists():
            return "present", local.read_text(encoding="utf-8")
    except OSError:
        pass
    # origin/main unreadable and no local copy: cannot prove the marker is absent.
    # A checkout that simply has no git metadata and no marker is treated as absent only
    # when git itself is not usable at all AND no marker file exists on disk.
    if not (root / ".git").exists():
        return "absent", ""
    return "error", ""


def resolve_ownership(event_id=None, *, root: Path = ROOT, now: datetime | None = None) -> Ownership:
    if _flag("SIM_FAIRS_FORCE_PRODUCTION"):
        return Ownership(False, "forced_production", None, "SIM_FAIRS_FORCE_PRODUCTION set")
    resume = _flag("SIM_FAIRS_OWNERSHIP_RESUME_ON_STALE")
    state, text = read_marker(root)
    if state == "absent":
        return Ownership(False, "none", None, f"no {MARKER}")
    if state == "error":
        return Ownership(not resume, "unreadable", None,
                         f"could not read {MARKER} from origin/main or disk; failing closed")
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("not an object")
    except ValueError:
        return Ownership(not resume, "unreadable", "external", f"{MARKER} is corrupt; failing closed")
    owner = str(data.get("owner") or "external")
    now = now or datetime.now(timezone.utc)

    problems = []
    status = "owned"
    marker_events = {str(data["event_id"])} if data.get("event_id") not in (None, "") else set()
    marker_events |= {str(x) for x in (data.get("event_ids") or [])}
    if event_id not in (None, "") and marker_events and str(event_id) not in marker_events:
        status = "stale_event"
        problems.append(f"marker event(s) {sorted(marker_events)} != current event {event_id}")
    expires = _parse_ts(data.get("expires_at"))
    if expires is None:
        since = _parse_ts(data.get("since"))
        try:
            max_h = float(os.environ.get("SIM_FAIRS_EXTERNAL_MAX_AGE_HOURS") or DEFAULT_MAX_AGE_HOURS)
        except ValueError:
            max_h = DEFAULT_MAX_AGE_HOURS
        if since is not None:
            expires = since + timedelta(hours=max_h)
    if expires is not None and now > expires:
        status = "expired" if status == "owned" else status
        problems.append(f"ownership expired {expires.strftime('%Y-%m-%d %H:%M UTC')}")
    if status == "owned":
        return Ownership(True, "owned", owner, f"owner '{owner}' owns event {event_id}")
    return Ownership(not resume, status, owner,
                     "; ".join(problems) + (" - STALE MARKER, standing down anyway to protect the board"
                                            if not resume else " - resuming production (RESUME_ON_STALE=1)"))
