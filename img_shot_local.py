"""Read the local rich-shot archive without touching the live writer.

The collector owns the WAL database. Sims open it with SQLite ``mode=ro`` and
select the newest versioned accounting and predictive models for a round.
"""

from __future__ import annotations

import csv
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping


ARCHIVE_ADAPTER = "rich_numbered_strokes_course_aware_v2"
CROSSWALK_FIELDS = (
    "source_player_id", "source_player_name", "datagolf_player_name",
    "canonical_name", "event_key", "first_seen_at", "last_seen_at",
)

# Explicit signal-version policy: never select a model merely because it was
# recomputed most recently. Priority is ordered best-first; statuses outside
# the allowlist are rejected (v1's "provisional_calibration_required" DPWT
# signal never reaches the sim unless deliberately enabled via env).
SIGNAL_METHOD_PRIORITY = tuple(
    part.strip()
    for part in (
        os.getenv("IMG_SHOT_SIGNAL_METHODS")
        or "predictive_skill_pga_validated_v2"
    ).split(",")
    if part.strip()
)
SIGNAL_STATUS_ALLOWLIST = frozenset(
    part.strip()
    for part in (
        os.getenv("IMG_SHOT_SIGNAL_STATUSES")
        or "validated_pga_2020_2024_plus_2026"
    ).split(",")
    if part.strip()
)


def default_db_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "golf_scraping" / "output" / "imgarena_shots_lean.sqlite3"
    )


def resolve_db_path(value: str | Path | None = None) -> Path:
    configured = value or os.getenv("IMG_SHOT_DB_PATH")
    return Path(configured).expanduser().resolve() if configured else default_db_path()


def _connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = resolve_db_path(db_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def _resolve_event_key(
    connection: sqlite3.Connection,
    event_id: int | str,
    *,
    tour: str = "pga",
    season: int | None = None,
    event_key: str | None = None,
) -> str:
    explicit = str(event_key or "").strip()
    raw = str(event_id).strip()
    if not explicit and ":" in raw:
        explicit = raw
    if explicit:
        row = connection.execute(
            "SELECT event_key FROM archive_events WHERE event_key=?", (explicit,)
        ).fetchone()
        if row:
            return str(row[0])
        raise LookupError(f"archive event not found: {explicit}")

    source_ids = [raw]
    if raw.isdigit():
        numeric = int(raw)
        if str(tour).lower() == "pga" and season is not None:
            source_ids.insert(0, f"R{int(season)}{numeric:03d}")
        elif str(tour).lower() == "dpwt" and season is not None:
            source_ids.insert(0, f"{int(season)}{numeric:03d}")
    placeholders = ",".join("?" for _ in source_ids)
    values = [str(tour).lower(), *source_ids]
    season_sql = ""
    if season is not None:
        season_sql = "AND e.season=?"
        values.append(int(season))
    row = connection.execute(
        f"""
        SELECT e.event_key
        FROM archive_event_sources s
        JOIN archive_events e ON e.event_key=s.event_key
        WHERE e.tour=? AND s.source_event_id IN ({placeholders}) {season_sql}
        ORDER BY e.season DESC, e.event_key DESC LIMIT 1
        """,
        values,
    ).fetchone()
    if row:
        return str(row[0])
    raise LookupError(
        f"archive event not found for tour={tour} event_id={event_id} season={season}"
    )


def _current_raw_model(
    connection: sqlite3.Connection, event_key: str, round_no: int
) -> str | None:
    row = connection.execute(
        """
        SELECT p.model_id
        FROM archive_player_round_strokes_gained p
        JOIN sg_models m ON m.model_id=p.model_id
        WHERE p.event_key=? AND p.round_no=?
          AND json_extract(m.metadata_json, '$.archive_adapter')=?
        ORDER BY p.calculated_at DESC, p.model_id DESC LIMIT 1
        """,
        (event_key, int(round_no), ARCHIVE_ADAPTER),
    ).fetchone()
    return None if row is None else str(row[0])


def _current_signal_model(
    connection: sqlite3.Connection, event_key: str, round_no: int,
    raw_model_id: str,
) -> str | None:
    candidates = connection.execute(
        """
        SELECT DISTINCT s.signal_model_id,
               json_extract(m.metadata_json, '$.method') AS method,
               json_extract(m.metadata_json, '$.status') AS status,
               MAX(s.calculated_at) AS calculated_at
        FROM archive_player_round_skill_signal s
        JOIN sg_models m ON m.model_id=s.signal_model_id
        WHERE s.event_key=? AND s.round_no=? AND s.raw_model_id=?
        GROUP BY s.signal_model_id
        """,
        (event_key, int(round_no), raw_model_id),
    ).fetchall()
    eligible = [
        row for row in candidates
        if str(row["method"] or "") in SIGNAL_METHOD_PRIORITY
        and str(row["status"] or "") in SIGNAL_STATUS_ALLOWLIST
    ]
    if not eligible:
        return None
    eligible.sort(
        key=lambda row: (
            SIGNAL_METHOD_PRIORITY.index(str(row["method"])),
            str(row["calculated_at"] or ""),
        )
    )
    best_method = str(eligible[0]["method"])
    same_method = [r for r in eligible if str(r["method"]) == best_method]
    # Within the pinned method, prefer the most recent recompute.
    same_method.sort(key=lambda row: str(row["calculated_at"] or ""), reverse=True)
    return str(same_method[0]["signal_model_id"])


def read_player_rounds(
    event_id: int | str, round_no: int, *, db_path: str | Path | None = None,
    tour: str = "pga", season: int | None = None,
    event_key: str | None = None, with_meta: bool = False,
) -> tuple[str, list[dict]] | tuple[str, list[dict], dict]:
    with _connect(db_path) as connection:
        resolved = _resolve_event_key(
            connection, event_id, tour=tour, season=season, event_key=event_key
        )
        raw_model = _current_raw_model(connection, resolved, round_no)
        if raw_model is None:
            if with_meta:
                return resolved, [], {"raw_model_id": None, "signal_model_id": None}
            return resolved, []
        signal_model = _current_signal_model(connection, resolved, round_no, raw_model)
        rows = connection.execute(
            """
            SELECT p.event_key, p.round_no,
                   p.source_player_id AS img_player_id,
                   p.source_team_id AS img_team_id,
                   COALESCE((SELECT ap.player_name FROM archive_players ap
                             WHERE ap.event_key=p.event_key
                               AND ap.source_player_id=p.source_player_id
                             LIMIT 1), p.player_name) AS source_player_name,
                   p.player_name, p.model_id AS raw_model_id,
                   p.round_strokes, p.total_strokes, p.valid_sg_strokes,
                   p.complete, p.sg_ott_raw, p.sg_app_raw, p.sg_arg_raw,
                   p.sg_putt_raw, p.sg_total_raw, p.sg_ott, p.sg_app,
                   p.sg_arg, p.sg_putt, p.sg_bs, p.sg_t2g, p.sg_total,
                   p.adjustment_method, s.course_id, s.signal_model_id,
                   s.sg_ott_signal, s.sg_app_signal, s.sg_arg_signal,
                   s.sg_putt_signal, s.sg_bs_signal, s.sg_t2g_signal,
                   s.sg_total_signal,
                   json_extract(sm.metadata_json, '$.status') AS signal_status
            FROM archive_player_round_strokes_gained p
            LEFT JOIN archive_player_round_skill_signal s
              ON s.event_key=p.event_key AND s.round_no=p.round_no
             AND s.source_player_id=p.source_player_id
             AND s.source_team_id=p.source_team_id
             AND s.raw_model_id=p.model_id AND s.signal_model_id=?
            LEFT JOIN sg_models sm ON sm.model_id=s.signal_model_id
            WHERE p.event_key=? AND p.round_no=? AND p.model_id=?
            ORDER BY p.player_name, p.source_player_id
            """,
            (signal_model or "", resolved, int(round_no), raw_model),
        ).fetchall()
        records = [dict(row) for row in rows]
        if not with_meta:
            return resolved, records
        freshness = connection.execute(
            """
            SELECT MAX(calculated_at) FROM archive_player_round_strokes_gained
            WHERE event_key=? AND round_no=? AND model_id=?
            """,
            (resolved, int(round_no), raw_model),
        ).fetchone()
        meta = {
            "raw_model_id": raw_model,
            "signal_model_id": signal_model,
            "raw_calculated_at": None if freshness is None else freshness[0],
            "signal_method_priority": list(SIGNAL_METHOD_PRIORITY),
            "row_count": len(records),
            "complete_count": sum(1 for row in records if row.get("complete")),
        }
        return resolved, records, meta


def read_player_shots(
    event_id: int | str, round_no: int, *, db_path: str | Path | None = None,
    tour: str = "pga", season: int | None = None,
    event_key: str | None = None,
) -> tuple[str, list[dict]]:
    with _connect(db_path) as connection:
        resolved = _resolve_event_key(
            connection, event_id, tour=tour, season=season, event_key=event_key
        )
        raw_model = _current_raw_model(connection, resolved, round_no)
        if raw_model is None:
            return resolved, []
        signal_model = _current_signal_model(connection, resolved, round_no, raw_model)
        rows = connection.execute(
            """
            SELECT sg.shot_key, sg.event_key, sg.round_no, sg.hole_no,
                   sg.source_player_id AS img_player_id,
                   sg.source_team_id AS img_team_id,
                   sg.player_name, sg.stroke_no AS shot_no,
                   sg.start_lie, sg.start_distance_yd,
                   sg.end_lie, sg.end_distance_yd,
                   sg.sg_category, sg.sg_raw, sg.sg_valid, sg.sg_issue,
                   sg.complete_hole, sg.state_inferred,
                   a.course_id, a.event_type, a.shot_distance_yd,
                   a.start_x, a.start_y, a.start_z,
                   a.end_x, a.end_y, a.end_z,
                   ss.signal_model_id, ss.is_penalty,
                   ss.sg_signal_uncentered, ss.transform_reason
            FROM archive_shot_strokes_gained sg
            JOIN archive_shots a ON a.shot_key=sg.shot_key
            LEFT JOIN archive_shot_skill_signal ss
              ON ss.shot_key=sg.shot_key AND ss.raw_model_id=sg.model_id
             AND ss.signal_model_id=?
            WHERE sg.event_key=? AND sg.round_no=? AND sg.model_id=?
            ORDER BY sg.source_player_id, sg.hole_no, sg.stroke_no
            """,
            (signal_model or "", resolved, int(round_no), raw_model),
        ).fetchall()
        return resolved, [dict(row) for row in rows]


def write_player_crosswalk(
    matches: Iterable[Mapping], path: str | Path | None = None
) -> Path | None:
    incoming = [dict(row) for row in matches if row.get("source_player_id")]
    if not incoming:
        return None
    target = Path(
        path or Path(__file__).parent / "permanent_data" / "img_player_crosswalk.csv"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if target.exists():
        with target.open(newline="", encoding="utf-8-sig") as handle:
            existing = {
                row["source_player_id"]: row
                for row in csv.DictReader(handle)
                if row.get("source_player_id")
            }
    now = datetime.now(timezone.utc).isoformat()
    for row in incoming:
        key = str(row["source_player_id"])
        prior = existing.get(key, {})
        existing[key] = {
            field: str(row.get(field) or prior.get(field) or "")
            for field in CROSSWALK_FIELDS
        }
        existing[key]["source_player_id"] = key
        existing[key]["first_seen_at"] = prior.get("first_seen_at") or now
        existing[key]["last_seen_at"] = now
    handle = tempfile.NamedTemporaryFile(
        mode="w", newline="", encoding="utf-8", suffix=".tmp",
        prefix="img_player_crosswalk_", dir=target.parent, delete=False,
    )
    with handle:
        writer = csv.DictWriter(handle, fieldnames=CROSSWALK_FIELDS)
        writer.writeheader()
        writer.writerows(existing[key] for key in sorted(existing))
    os.replace(handle.name, target)
    return target


def available() -> bool:
    return resolve_db_path().is_file()
