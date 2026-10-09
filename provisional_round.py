"""Explicit, event-bound provisional weekend inputs; never used by normal runs."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from zoneinfo import ZoneInfo

ENV = "PROVISIONAL_ROUND_INPUT"


def build_inputs(stats, official, *, event_id, tourney, course_id, par, cut,
                 date, start="09:20", end="11:30", tz="America/Denver"):
    """Freeze original observations and rank total-to-par, including cut ties.

    Tied players sort by canonical name then DG ID. The odd player goes in the
    better half. Each half is chunked into threes, with a final one/two left over.
    Times span the requested window at whole-minute resolution.
    """
    ZoneInfo(tz)  # Fail before writing if timezone support is unavailable.
    if int(cut) < 1:
        raise ValueError("Provisional cut must be positive")
    frame = stats.copy()
    if frame.player_name.duplicated().any():
        raise ValueError("Provisional stats have duplicate players")
    eligible = ~frame.position.astype(str).str.contains("WD|DQ|CUT", case=False)
    ranked = frame.loc[eligible].copy()
    for col in ("total", "round", "thru"):
        ranked[col] = pd.to_numeric(ranked[col], errors="raise")
        if ranked[col].isna().any():
            raise ValueError(f"Provisional stats have missing {col}")
    if ranked.empty or not ranked.thru.between(0, 18).all():
        raise ValueError("Invalid provisional field/holes completed")
    ranked = ranked.sort_values(["total", "player_name"], kind="stable")
    cutoff = ranked.total.iloc[min(int(cut), len(ranked)) - 1]
    ranked = ranked.loc[ranked.total <= cutoff].copy()
    ranked["assumed_r2_strokes"] = float(par) + ranked["round"]
    ranked["provisional_rank"] = range(1, len(ranked) + 1)
    midpoint = (len(ranked) + 1) // 2
    begin = datetime.fromisoformat(f"{date}T{start}")
    finish = datetime.fromisoformat(f"{date}T{end}")
    minutes = int((finish - begin).total_seconds() / 60)
    if minutes <= 0:
        raise ValueError("Provisional tee window must increase")
    rows = []
    for tee, half in ((1, ranked.iloc[:midpoint].iloc[::-1]),
                      (10, ranked.iloc[midpoint:])):
        groups = (len(half) + 2) // 3
        if groups > minutes + 1:
            raise ValueError("Too many tee groups for requested window")
        for index, (_, player) in enumerate(half.iterrows()):
            group = index // 3
            offset = round(minutes * group / max(1, groups - 1))
            rows.append({"player_name": player.player_name,
                         "r3_teetime": (begin + timedelta(minutes=offset)).strftime("%Y-%m-%d %H:%M"),
                         "starting_tee": tee, "course": None,
                         "tee_time_source": "provisional"})
    schedule = pd.DataFrame(rows)
    # Preserve every posted individual time; fallback fills only missing times.
    if official is not None and "r3_teetime" in official:
        if official.player_name.duplicated().any():
            raise ValueError("Official field has duplicate players")
        lookup = official.set_index("player_name")
        for i, row in schedule.iterrows():
            if row.player_name not in lookup.index:
                continue
            posted = lookup.loc[row.player_name]
            if pd.notna(posted.r3_teetime) and str(posted.r3_teetime).strip():
                schedule.loc[i, "r3_teetime"] = posted.r3_teetime
                schedule.loc[i, "tee_time_source"] = "official"
                tee = posted.get("starting_tee", 1)
                schedule.loc[i, "starting_tee"] = tee if pd.notna(tee) else 1
            schedule.loc[i, "course"] = posted.get("course")
    courses = frame.get("course", pd.Series(dtype=str)).dropna().unique()
    if len(courses) != 1:
        raise ValueError("Provisional workflow requires one physical course")
    schedule["course"] = schedule.course.fillna(courses[0])
    summary = ("PROVISIONAL R3: current R2 scores assumed final; low "
               f"{cut} and ties (cut {cutoff:+g}). Actual holes completed preserved. "
               f"Missing R3 times generated {date} {start}-{end} {tz}; "
               "better half tee 1 leaders last, lower half tee 10 worst last. "
               "Official individual times take precedence. Market coverage may be incomplete.")
    return {"version": 1, "event_id": int(event_id), "tourney": tourney,
            "course_id": int(course_id), "target_round": 3,
            "created_at": datetime.now(timezone.utc).isoformat(), "timezone": tz,
            "label": summary, "cut_total": float(cutoff),
            "stats": json.loads(frame.to_json(orient="records")),
            "standings": json.loads(ranked.to_json(orient="records")),
            "field": json.loads(schedule.to_json(orient="records"))}


def write_inputs(payload, path):
    payload = dict(payload)
    payload["sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path.resolve()


def current_inputs(*, event_id=None, target_round=None):
    path = os.environ.get(ENV)
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = {k: v for k, v in data.items() if k != "sha256"}
    digest = hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest()
    if data.get("version") != 1 or data.get("sha256") != digest:
        raise ValueError("Invalid provisional input manifest/hash")
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(data["created_at"])).total_seconds()
    if not -900 <= age <= 18 * 3600:
        raise ValueError("Provisional input snapshot is stale")
    if event_id is not None and str(data["event_id"]) != str(event_id):
        raise ValueError("Provisional input event mismatch")
    if target_round is not None and int(data["target_round"]) != int(target_round):
        raise ValueError("Provisional input round mismatch")
    import sys
    config = sys.modules.get("sim_inputs")
    if config and hasattr(config, "event_ids"):
        if (int(data["event_id"]) not in config.event_ids or
                str(data["tourney"]) != str(config.tourney) or
                int(data["course_id"]) != int(config.course_id)):
            raise ValueError("Provisional input does not match active event/course configuration")
    return data


def record_email_delivery(subject, recipient_count):
    data = current_inputs()
    if data:
        path = Path(os.environ[ENV]).with_suffix(".email.json")
        receipts = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        receipts.append({"subject": subject, "recipient_count": recipient_count,
                         "accepted_at": datetime.now(timezone.utc).isoformat(),
                         "input_sha256": data["sha256"], "transport": "smtp_accepted"})
        path.write_text(json.dumps(receipts, indent=2), encoding="utf-8")


def email_notice():
    from html import escape
    data = current_inputs()
    return f"<p><strong>{escape(data['label'])}</strong></p>" if data else ""


def tee_contract(event_id, rnd):
    data = current_inputs(event_id=event_id, target_round=rnd)
    if not data:
        return None
    frame = pd.DataFrame(data["field"])
    grouped = list(frame.groupby(["r3_teetime", "starting_tee"]).player_name.apply(list))
    if any(len(g) > 3 for g in grouped):
        raise ValueError("Provisional/official schedule has colliding tee groups")
    names = sorted(frame.player_name)
    return {"status": "groups", "groups": [sorted(g) for g in grouped if len(g) == 3],
            "group_sizes": sorted(map(len, grouped)), "field_names": names,
            "tee_time_names": names, "field_players": len(names),
            "tee_time_players": len(names), "round": rnd, "event_id": str(event_id),
            "event_identity_basis": "provisional_event_snapshot", "provisional": data["label"],
            "input_sha256": data["sha256"]}
