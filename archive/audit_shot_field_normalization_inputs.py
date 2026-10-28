"""Audit leakage-safe inputs for historical shot-donor normalization.

This is a research-only gate for the spatial shot simulator.  It verifies that
ShotLink players can be joined to a strictly prior DataGolf skill snapshot,
measures how much recurring-course donor fields differ across editions, and
quantifies the R3/R4 survivor-composition shift.  It does not alter production
inputs or simulation outputs.

The three default events are deliberate: Travelers' 2024-25 fields are much
stronger than its earlier editions, while Sony and Wyndham provide lower-skill
recurring-course controls.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_EXPORT = Path(
    r"C:\Users\McKinley Slade\dev\golf_scraping\output\export"
)
DEFAULT_HISTORY_DB = Path(
    r"C:\Users\McKinley Slade\dev\sim_prep\work\dg_historical.db"
)
DEFAULT_FIELD_DB = Path(
    r"C:\Users\McKinley Slade\dev\sim_prep\work\field_strength.db"
)
DEFAULT_OUTPUT = Path("permanent_data/shot_field_normalization_input_audit")
DEFAULT_EVENTS = (
    "Sony Open in Hawaii",
    "Travelers Championship",
    "Wyndham Championship",
)
SKILL_COLUMNS = (
    "skill_est",
    "sg_ott",
    "sg_app",
    "ag_arg",
    "ag_putt",
    "rel_dd",
)


def _load_name_replacements(path: Path) -> dict[str, str]:
    """Read the literal canonical-name map without importing weekly config."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "name_replacements"
            for target in node.targets
        ):
            continue
        raw = ast.literal_eval(node.value)
        return {
            str(key).casefold().strip(): str(value).casefold().strip()
            for key, value in raw.items()
        }
    return {}


NAME_REPLACEMENTS = _load_name_replacements(
    Path(__file__).resolve().parents[1] / "sim_inputs.py"
)


def _name_key(value: object) -> str:
    raw = str(value).casefold().strip()
    raw = NAME_REPLACEMENTS.get(raw, raw)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", raw).split())


def _frame_hash(frame: pd.DataFrame) -> str:
    """Stable content hash after deterministic row/column ordering."""
    ordered = frame.reindex(sorted(frame.columns), axis=1).copy()
    if len(ordered):
        ordered = ordered.sort_values(list(ordered.columns), kind="mergesort")
    payload = ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_meta(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def load_event_calendar(history_db: Path) -> pd.DataFrame:
    with sqlite3.connect(f"file:{history_db.as_posix()}?mode=ro", uri=True) as db:
        frame = pd.read_sql_query(
            """
            SELECT year AS season, event_id, event_name,
                   MIN(round_date) AS event_start
            FROM player_rounds
            WHERE tour = 'pga' AND year BETWEEN 2020 AND 2026
            GROUP BY year, event_id, event_name
            """,
            db,
        )
    frame["event_start"] = pd.to_datetime(frame["event_start"], errors="coerce")
    frame["event_name_key"] = frame["event_name"].map(_name_key)
    frame["event_key"] = [
        f"pga:R{int(year)}{int(event_id):03d}"
        for year, event_id in zip(frame["season"], frame["event_id"])
    ]
    return frame.dropna(subset=["event_start"]).drop_duplicates("event_key")


def load_skill_history(history_db: Path) -> pd.DataFrame:
    columns = ", ".join(SKILL_COLUMNS)
    with sqlite3.connect(f"file:{history_db.as_posix()}?mode=ro", uri=True) as db:
        frame = pd.read_sql_query(
            f"""
            SELECT Date AS skill_date, dg_id, player_name, {columns}
            FROM model_results2
            WHERE Date >= '2018-01-01' AND Date < '2027-01-01'
            """,
            db,
        )
    frame["skill_date"] = pd.to_datetime(frame["skill_date"], errors="coerce")
    frame["name_key"] = frame["player_name"].map(_name_key)
    numeric = list(SKILL_COLUMNS)
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    return frame.dropna(subset=["skill_date", "name_key", "dg_id", "skill_est"])


def load_stored_field_strength(field_db: Path) -> pd.DataFrame:
    if not field_db.exists():
        return pd.DataFrame(columns=["season", "event_name_key", "stored_field_skill"])
    with sqlite3.connect(f"file:{field_db.as_posix()}?mode=ro", uri=True) as db:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", db
        )["name"].tolist()
        if not tables:
            return pd.DataFrame(columns=["season", "event_name_key", "stored_field_skill"])
        selected = None
        for table in tables:
            info = pd.read_sql_query(f'PRAGMA table_info("{table}")', db)
            names = set(info["name"])
            if {"Tournament Name", "Mean Skill", "Year"}.issubset(names):
                selected = table
                break
        if selected is None:
            return pd.DataFrame(columns=["season", "event_name_key", "stored_field_skill"])
        frame = pd.read_sql_query(
            f'SELECT "Tournament Name" AS event_name, "Mean Skill" AS '
            f'stored_field_skill, "Year" AS season FROM "{selected}"',
            db,
        )
    frame["event_name_key"] = frame["event_name"].map(_name_key)
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    frame["stored_field_skill"] = pd.to_numeric(
        frame["stored_field_skill"], errors="coerce"
    )
    return frame.dropna(subset=["season", "event_name_key"])


def load_round_rosters(export_root: Path, seasons: list[int]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    columns = [
        "event_key",
        "season",
        "event_name",
        "round_no",
        "source_player_id",
        "player_name",
        "complete",
    ]
    for season in seasons:
        path = export_root / f"pga_{season}_player_rounds.parquet"
        frame = pd.read_parquet(path, columns=columns)
        frame["source_player_id"] = frame["source_player_id"].astype(str)
        frame["name_key"] = frame["player_name"].map(_name_key)
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def latest_strict_snapshot(
    roster: pd.DataFrame,
    skill_history: pd.DataFrame,
    event_start: pd.Timestamp,
    aliases: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return the latest unambiguous alias-matched strictly-prior snapshot.

    ShotLink's source ID is not DataGolf's ``dg_id``.  Names for the same
    source ID can also change within a season (for example initials versus a
    full first name), so every archived alias for that source ID is considered.
    A source ID resolving to multiple DataGolf IDs fails closed as ambiguous.
    """
    query = roster[["source_player_id", "player_name", "name_key"]].drop_duplicates()
    query["source_player_id"] = query["source_player_id"].astype(str)
    if aliases is None:
        aliases = query[["source_player_id", "name_key"]].copy()
    else:
        aliases = aliases[["source_player_id", "name_key"]].drop_duplicates().copy()
        aliases["source_player_id"] = aliases["source_player_id"].astype(str)
        aliases = pd.concat(
            [aliases, query[["source_player_id", "name_key"]]], ignore_index=True
        ).drop_duplicates()
    aliases = aliases[aliases["source_player_id"].isin(query["source_player_id"])]

    eligible = skill_history[skill_history["skill_date"].lt(event_start)].copy()
    eligible["dg_id"] = eligible["dg_id"].astype(str)
    latest = (
        eligible.sort_values("skill_date")
        .drop_duplicates(["name_key", "dg_id"], keep="last")
        .rename(columns={"name_key": "matched_name_key"})
    )
    candidates = aliases.rename(columns={"name_key": "matched_name_key"}).merge(
        latest[["matched_name_key", "skill_date", "dg_id", *SKILL_COLUMNS]],
        on="matched_name_key",
        how="left",
    )
    candidate_counts = candidates.groupby("source_player_id", observed=True)[
        "dg_id"
    ].nunique()
    ambiguous_ids = set(candidate_counts[candidate_counts.gt(1)].index.astype(str))
    unique_ids = candidate_counts[candidate_counts.eq(1)].index.astype(str)
    picked = (
        candidates[candidates["source_player_id"].isin(unique_ids)]
        .dropna(subset=["dg_id"])
        .sort_values("skill_date")
        .drop_duplicates("source_player_id", keep="last")
    )
    result = query.merge(
        picked[
            [
                "source_player_id",
                "matched_name_key",
                "skill_date",
                "dg_id",
                *SKILL_COLUMNS,
            ]
        ],
        on="source_player_id",
        how="left",
        validate="many_to_one",
    )
    result["ambiguous_name"] = result["source_player_id"].isin(ambiguous_ids)
    result["snapshot_age_days"] = (
        event_start - result["skill_date"]
    ).dt.total_seconds() / 86400.0
    return result


def build_audit(
    rosters: pd.DataFrame,
    calendar: pd.DataFrame,
    skills: pd.DataFrame,
    events: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    desired = {_name_key(name) for name in events}
    calendar = calendar[calendar["event_name_key"].isin(desired)].copy()
    event_rows: list[dict[str, object]] = []
    player_rows: list[pd.DataFrame] = []
    for event in calendar.sort_values(["season", "event_name"]).itertuples(index=False):
        rounds = rosters[rosters["event_key"].eq(event.event_key)].copy()
        if rounds.empty:
            continue
        r1 = rounds[rounds["round_no"].eq(1)].copy()
        season_aliases = rosters[
            rosters["season"].eq(event.season)
        ][["source_player_id", "name_key"]].drop_duplicates()
        snapshot = latest_strict_snapshot(
            r1, skills, event.event_start, aliases=season_aliases
        )
        snapshot["event_key"] = event.event_key
        snapshot["event_name"] = event.event_name
        snapshot["season"] = int(event.season)
        snapshot["event_start"] = event.event_start
        player_rows.append(snapshot)

        matched = snapshot[snapshot["skill_est"].notna()].copy()
        late_ids = set(
            rounds.loc[rounds["round_no"].ge(3), "source_player_id"].astype(str)
        )
        early_ids = set(
            rounds.loc[rounds["round_no"].le(2), "source_player_id"].astype(str)
        )
        late = matched[matched["source_player_id"].isin(late_ids)]
        early = matched[matched["source_player_id"].isin(early_ids)]
        row: dict[str, object] = {
            "event_key": event.event_key,
            "event_name": event.event_name,
            "season": int(event.season),
            "event_start": event.event_start,
            "r1_players": int(r1["source_player_id"].nunique()),
            "matched_players": int(matched["source_player_id"].nunique()),
            "skill_coverage": float(len(matched) / max(len(snapshot), 1)),
            "ambiguous_players": int(snapshot["ambiguous_name"].sum()),
            "median_snapshot_age_days": float(matched["snapshot_age_days"].median()),
            "r3_r4_players": int(len(late_ids)),
            "field_skill": float(matched["skill_est"].mean()),
            "post_cut_skill": float(late["skill_est"].mean()),
            "post_cut_minus_field_skill": float(
                late["skill_est"].mean() - early["skill_est"].mean()
            ),
        }
        for column in SKILL_COLUMNS[1:]:
            row[f"field_{column}"] = float(matched[column].mean())
        event_rows.append(row)
    event_audit = pd.DataFrame(event_rows)
    player_audit = pd.concat(player_rows, ignore_index=True)
    return event_audit, player_audit


def write_report(event_audit: pd.DataFrame, output_dir: Path) -> Path:
    rows = []
    for row in event_audit.sort_values(["event_name", "season"]).itertuples(index=False):
        stored = "" if pd.isna(row.stored_field_skill) else f"{row.stored_field_skill:+.3f}"
        rows.append(
            f"| {row.event_name} | {int(row.season)} | {row.field_skill:+.3f} | "
            f"{stored} | {row.skill_coverage:.1%} | "
            f"{row.post_cut_minus_field_skill:+.3f} |"
        )
    report = f"""# Shot donor field-normalization input audit

All model snapshots are the latest observation strictly before the event start.
ShotLink IDs are not treated as DataGolf IDs; the bridge is normalized player
    name with ambiguous matches rejected.  Season-wide aliases are treated as
    stable identity metadata, not as predictive features; skill values themselves
    remain strictly prior.  The final normalizer must retain this distinction or
    replace it with a validated canonical crosswalk.

| Event | Year | Snapshot field SG | Stored field SG | Skill coverage | R3/R4 survivor shift |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

`R3/R4 survivor shift` is the mean pre-event skill of players observed in
rounds 3-4 minus the mean for the R1/R2 roster.  It quantifies why simply
pooling late-round course shots can change the donor distribution even on the
same physical course.

This file audits input availability and selection effects only.  It does not
validate a coordinate transform, transition-quality model, or production use.
"""
    path = output_dir / "INPUT_AUDIT.md"
    path.write_text(report, encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--history-db", type=Path, default=DEFAULT_HISTORY_DB)
    parser.add_argument("--field-db", type=Path, default=DEFAULT_FIELD_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--event", action="append", dest="events")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = tuple(args.events) if args.events else DEFAULT_EVENTS
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    calendar = load_event_calendar(args.history_db)
    event_keys = calendar[
        calendar["event_name_key"].isin({_name_key(name) for name in events})
    ]
    seasons = sorted(event_keys["season"].astype(int).unique())
    rosters = load_round_rosters(args.export_root, seasons)
    skills = load_skill_history(args.history_db)
    stored = load_stored_field_strength(args.field_db)
    event_audit, player_audit = build_audit(rosters, calendar, skills, events)
    event_audit["event_name_key"] = event_audit["event_name"].map(_name_key)
    event_audit = event_audit.merge(
        stored[["season", "event_name_key", "stored_field_skill"]],
        on=["season", "event_name_key"],
        how="left",
    ).drop(columns="event_name_key")

    event_path = output_dir / "event_field_strength.csv"
    player_path = output_dir / "player_identity_snapshot.csv"
    event_audit.to_csv(event_path, index=False)
    player_audit.to_csv(player_path, index=False)
    report_path = write_report(event_audit, output_dir)
    manifest = {
        "contract": "strict_prior_skill_global_identity_alias_v2",
        "skill_snapshot_rule": "latest model_results2.skill_date < event_start",
        "identity_alias_scope": (
            "full archived season; stable identity metadata only, never a predictor"
        ),
        "events": list(events),
        "event_rows": int(len(event_audit)),
        "player_rows": int(len(player_audit)),
        "event_audit_sha256": _frame_hash(event_audit),
        "player_audit_sha256": _frame_hash(player_audit),
        "event_artifact_sha256": _file_sha256(event_path),
        "player_artifact_sha256": _file_sha256(player_path),
        "history_db": _file_meta(args.history_db),
        "field_db": _file_meta(args.field_db),
        "events_parquet": _file_meta(args.export_root / "events.parquet"),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "name_replacements_sha256": hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "sim_inputs.py").read_bytes()
        ).hexdigest(),
        "strict_cutoff_violations": int(
            (pd.to_datetime(player_audit["skill_date"]) >= pd.to_datetime(player_audit["event_start"]))
            .fillna(False)
            .sum()
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    if manifest["strict_cutoff_violations"]:
        raise RuntimeError("Strict-prior snapshot audit failed")
    if event_audit["skill_coverage"].min() < 0.85:
        raise RuntimeError("Player identity/skill coverage fell below 85%")
    print(event_audit.to_string(index=False))
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
