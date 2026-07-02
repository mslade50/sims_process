"""maker_alerts.py — dead-man alerting for the automated maker (Telegram).

The rolling TTL means a dead maker stops bleeding, but SILENTLY. This module
makes the states an operator must know about NOW loud:
  * TRADE <-> HALT transitions (with the guard reason)
  * a post-fail streak (>= MAKER_ALERT_POSTFAIL_STREAK consecutive live runs
    with at least one failed POST — Kalshi is rejecting our re-quotes)
  * new fills since the last run (inventory just changed)

State (last status, streak, last-fill watermark) lives in
permanent_data/maker_alert_state.json so transitions survive process restarts.
The decision core (diff_alerts) is pure and unit-tested in test_maker_guard.py;
process_run wraps it with file + Telegram I/O and is best-effort: alerting can
NEVER break a maker run.

Configure TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID (same env as reprice/preflight
alerts); without them alerts are printed only. MAKER_ALERTS=1 forces alerting on
in dry/shadow runs (default: --live only).
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import maker_guard

STATE_FILE = Path("permanent_data") / "maker_alert_state.json"
_TRUTHY = {"1", "true", "yes", "on", "y", "t"}


def _env_int(name, default):
    try:
        v = os.getenv(name)
        return int(v) if v not in (None, "") else default
    except (TypeError, ValueError):
        return default


def alerts_forced():
    """MAKER_ALERTS truthy forces alerting on outside --live (dry/shadow)."""
    return (os.getenv("MAKER_ALERTS") or "").strip().lower() in _TRUTHY


def send_telegram(text):
    """Best-effort Telegram send. Returns True if a send was attempted OK."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat:
        return False
    try:
        import httpx
        r = httpx.post(f"https://api.telegram.org/bot{token}/sendMessage",
                       json={"chat_id": chat, "text": text, "parse_mode": "HTML"},
                       timeout=10)
        return 200 <= r.status_code < 300
    except Exception as e:
        print(f"  [alerts] Telegram send failed: {e}")
        return False


def _load_state(path=None):
    p = Path(path) if path else STATE_FILE
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_state(state, path=None):
    p = Path(path) if path else STATE_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, p)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _fill_ts(f):
    return maker_guard.parse_fairs_ts(f.get("created_time") or f.get("ts")) or 0.0


def _fill_line(f):
    side = f.get("side", "?")
    raw = f.get(f"{side}_price_dollars") or f.get(f"{side}_price") or 0
    try:
        px = float(raw)
        px = px / 100.0 if px > 1 else px
    except (TypeError, ValueError):
        px = 0.0
    try:
        n = int(round(float(f.get("count_fp") or f.get("count") or 0)))
    except (TypeError, ValueError):
        n = 0
    return f"{f.get('ticker', '?')} {side.upper()} {n} @ {px * 100:.1f}c"


def diff_alerts(prev, status, reason, post_report=None, fills=None,
                postfail_streak_n=None):
    """Pure. (alerts, new_state) from the previous state + this run's outcome.

    First run (empty prev) sets baselines without alerting — a fresh state file
    must not replay history as news.
    """
    n_streak = (_env_int("MAKER_ALERT_POSTFAIL_STREAK", 3)
                if postfail_streak_n is None else postfail_streak_n)
    alerts = []
    new_state = dict(prev or {})

    # 1. TRADE <-> HALT transitions.
    prev_status = (prev or {}).get("status")
    if prev_status and status and status != prev_status:
        icon = "🛑" if status == "HALT" else "✅"
        alerts.append(f"{icon} <b>Maker {prev_status} → {status}</b>\n{reason}")
    new_state["status"] = status
    new_state["reason"] = reason

    # 2. Post-fail streaks (live runs that actually posted).
    streak = int((prev or {}).get("postfail_streak") or 0)
    if post_report is not None:
        streak = streak + 1 if int(post_report.get("failed") or 0) > 0 else 0
        if streak == n_streak:
            alerts.append(f"⚠️ <b>Maker: {streak} consecutive runs with failed "
                          f"POSTs</b>\nlast run: posted={post_report.get('posted')} "
                          f"failed={post_report.get('failed')} "
                          f"re-armed={post_report.get('rearmed')} — Kalshi may be "
                          f"rejecting re-quotes; check [post-fail] lines")
    new_state["postfail_streak"] = streak

    # 3. New fills since the watermark. fills=None means "not fetched this run"
    # (dry/shadow or fetch failure) — leave the watermark alone so a credless run
    # can't reset it and replay history later. [] is real data (no fills exist).
    # No watermark yet -> baseline at the latest fill silently.
    if fills is not None:
        latest = max((_fill_ts(f) for f in fills), default=0.0)
        wm = (prev or {}).get("last_fill_ts")
        if wm is None:
            new_state["last_fill_ts"] = latest
        else:
            fresh = [f for f in fills if _fill_ts(f) > float(wm)]
            if fresh:
                lines = [_fill_line(f) for f in sorted(fresh, key=_fill_ts, reverse=True)]
                shown = "\n".join(lines[:8]) + ("" if len(lines) <= 8
                                                else f"\n… +{len(lines) - 8} more")
                alerts.append(f"💰 <b>Maker: {len(fresh)} new fill(s)</b>\n{shown}")
            new_state["last_fill_ts"] = max(float(wm), latest)

    return alerts, new_state


def process_run(status, reason, post_report=None, fills=None, enabled=True,
                state_path=None):
    """I/O wrapper: load state, diff, persist, send. Returns the alert texts."""
    prev = _load_state(state_path)
    alerts, new_state = diff_alerts(prev, status, reason,
                                    post_report=post_report, fills=fills)
    try:
        _save_state(new_state, state_path)
    except Exception as e:
        print(f"  [alerts] state save failed: {e}")
    for a in alerts:
        print(f"  [alerts] {' '.join(a.splitlines())}")
        if enabled:
            send_telegram(a)
    if alerts and not enabled:
        print("  [alerts] (not sent — alerts active under --live or MAKER_ALERTS=1)")
    return alerts
