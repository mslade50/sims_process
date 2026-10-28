"""Supplement the authoritative normalization-input gate with donor coverage.

The authoritative event/roster identity contract is
``audit_shot_field_normalization_inputs.py``.  This research-only script uses
the same strict ``model Date < event_start`` rule and adds shot-weighted donor
coverage plus static-profile provenance; it is not a second authority.  It
answers three integration questions without importing the production weekly
pipeline:

* what the two static spatial-profile CSVs actually contain;
* whether ShotLink ``source_player_id`` is the same identifier as DataGolf
  ``dg_id`` (it is not); and
* how much of the Sony/Travelers/Wyndham 2020-2025 donor support can be joined
  to an event-start ``model_results2`` snapshot through the archived yearly
  ShotLink player-name crosswalk.

The APP counts are deliberately labelled as a raw ``sg_category == 'app'``
coverage proxy.  The full-sequence spatial foundation applies additional
state/coherence filters, so this audit must not be interpreted as an exact
count of final APP donor atoms.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT = Path(
    r"C:\Users\McKinley Slade\dev\golf_scraping\output\export"
)
DEFAULT_DB = Path(
    r"C:\Users\McKinley Slade\dev\sim_prep\work\dg_historical.db"
)
DEFAULT_TEE_PROFILES = (
    REPO / "permanent_data" / "hole_sim_prototype" / "player_tee_profiles.csv"
)
DEFAULT_APP_PROFILES = (
    REPO
    / "permanent_data"
    / "hole_sim_app_transition_strict"
    / "player_app_profiles_structured.csv"
)
DEFAULT_OUTPUT = REPO / "permanent_data" / "hole_sim_normalized_donor_design"
PRIMARY_INPUT_AUDIT = REPO / "archive" / "audit_shot_field_normalization_inputs.py"
PRIMARY_INPUT_AUDIT_OUTPUT = (
    REPO / "permanent_data" / "shot_field_normalization_input_audit"
)
EVENTS = (
    "Sony Open in Hawaii",
    "Travelers Championship",
    "Wyndham Championship",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _literal_name_replacements(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "name_replacements"
            for target in node.targets
        ):
            continue
        value = ast.literal_eval(node.value)
        return {
            str(key).casefold().strip(): str(replacement).casefold().strip()
            for key, replacement in value.items()
        }
    raise ValueError(f"No literal name_replacements mapping found in {path}")


def _normalise_name(value: object, replacements: dict[str, str]) -> str:
    key = str(value).casefold().strip()
    canonical = replacements.get(key, key)
    # Match the authoritative field-normalization audit's punctuation-safe
    # name key.  The replacement pass handles known semantic aliases first.
    return " ".join(re.sub(r"[^a-z0-9]+", " ", canonical).split())


def _read_only_connection(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)


def _event_catalog(export_root: Path, connection: sqlite3.Connection) -> pd.DataFrame:
    catalog = pd.read_parquet(
        export_root / "events.parquet",
        columns=["event_key", "tour", "season", "event_name"],
    )
    catalog = catalog[
        catalog["tour"].eq("pga")
        & catalog["event_name"].isin(EVENTS)
        & catalog["season"].between(2020, 2025)
    ].copy()
    starts = pd.read_sql_query(
        """
        SELECT year AS season, event_name, MIN(round_date) AS event_start
        FROM player_rounds
        WHERE tour = 'pga'
          AND year BETWEEN 2020 AND 2025
          AND event_name IN (?, ?, ?)
        GROUP BY year, event_name
        """,
        connection,
        params=EVENTS,
    )
    result = catalog.merge(
        starts,
        on=["season", "event_name"],
        how="inner",
        validate="one_to_one",
    )
    if len(result) != 18:
        raise ValueError(f"Expected 18 recurring-event editions; found {len(result)}")
    result["event_start"] = pd.to_datetime(result["event_start"]).dt.normalize()
    return result.sort_values(["event_name", "season"]).reset_index(drop=True)


def _skill_snapshot(
    connection: sqlite3.Connection,
    event_start: pd.Timestamp,
    replacements: dict[str, str],
) -> pd.DataFrame:
    # The integration gate requires the *stored snapshot timestamp* itself to
    # precede the event.  This is intentionally stricter than relying on
    # golf_bt.py's internal ``round_date < Date`` producer rule.
    frame = pd.read_sql_query(
        """
        WITH ranked AS (
            SELECT rowid, Date, dg_id, player_name, sg_ott, sg_app, rel_dd,
                   skill_est, std_dev,
                   ROW_NUMBER() OVER (
                       PARTITION BY dg_id
                       ORDER BY datetime(Date) DESC, rowid DESC
                   ) AS rank_no
            FROM model_results2
            WHERE datetime(Date) < datetime(?)
              AND dg_id IS NOT NULL
        )
        SELECT rowid, Date, dg_id, player_name, sg_ott, sg_app, rel_dd,
               skill_est, std_dev
        FROM ranked
        WHERE rank_no = 1
        """,
        connection,
        params=[event_start.strftime("%Y-%m-%d")],
    )
    if frame.empty:
        raise ValueError(f"No model_results2 snapshot before {event_start.date()}")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise").dt.normalize()
    if not frame["Date"].lt(event_start).all():
        raise AssertionError("Strictly-prior model_results2 join failed")
    frame["player_key"] = frame["player_name"].map(
        lambda value: _normalise_name(value, replacements)
    )
    frame["dg_id"] = pd.to_numeric(frame["dg_id"], errors="raise").astype(int)
    frame = (
        frame.sort_values(["player_key", "dg_id", "rowid"])
        .drop_duplicates(["player_key", "dg_id"], keep="last")
        .reset_index(drop=True)
    )
    return frame


def _source_aliases(
    export_root: Path,
    season: int,
    replacements: dict[str, str],
) -> dict[str, set[str]]:
    frame = pd.read_parquet(
        export_root / f"pga_{season}_player_rounds.parquet",
        columns=["source_player_id", "player_name"],
    ).drop_duplicates()
    aliases: dict[str, set[str]] = defaultdict(set)
    for source_id, player_name in frame.itertuples(index=False):
        aliases[str(source_id)].add(_normalise_name(player_name, replacements))
    return dict(aliases)


def _identity_bridge(
    aliases: dict[str, set[str]],
    snapshot: pd.DataFrame,
) -> pd.DataFrame:
    ids_by_name = snapshot.groupby("player_key", observed=True)["dg_id"].agg(
        lambda values: frozenset(int(value) for value in values)
    )
    rows: list[dict[str, object]] = []
    for source_id, names in sorted(aliases.items()):
        candidates: set[int] = set()
        for name in names:
            if name in ids_by_name.index:
                candidates.update(ids_by_name.loc[name])
        if len(candidates) == 1:
            status = "unique"
            dg_id: int | float = next(iter(candidates))
        elif len(candidates) > 1:
            status = "ambiguous"
            dg_id = np.nan
        else:
            status = "unmatched"
            dg_id = np.nan
        rows.append(
            {
                "source_player_id": source_id,
                "source_alias_count": len(names),
                "source_aliases": "|".join(sorted(names)),
                "dg_candidate_count": len(candidates),
                "dg_candidates": "|".join(str(value) for value in sorted(candidates)),
                "identity_status": status,
                "dg_id": dg_id,
            }
        )
    return pd.DataFrame(rows)


def _tee_rows(export_root: Path, season: int, event_key: str) -> pd.DataFrame:
    shots = pd.read_parquet(
        export_root / f"pga_{season}_shots.parquet",
        columns=[
            "event_key",
            "source_player_id",
            "round_no",
            "hole_no",
            "course_id",
            "stroke_no",
            "event_type",
            "start_lie",
            "ball_holed",
            "shot_distance_yd",
        ],
        filters=[
            ("event_key", "==", event_key),
            ("stroke_no", "==", 1),
            ("start_lie", "==", "tee"),
        ],
    )
    geometry = pd.read_parquet(
        export_root / f"pga_{season}_hole_geometry.parquet",
        columns=["event_key", "round_no", "hole_no", "course_id", "par"],
        filters=[("event_key", "==", event_key)],
    ).drop_duplicates(["event_key", "round_no", "hole_no", "course_id"])
    shots = shots[
        shots["event_type"].eq("STROKE")
        & shots["ball_holed"].fillna(0).eq(0)
        & shots["shot_distance_yd"].between(100, 420)
    ].copy()
    shots = shots.merge(
        geometry,
        on=["event_key", "round_no", "hole_no", "course_id"],
        how="inner",
        validate="many_to_one",
    )
    return shots.loc[shots["par"].ge(4), ["source_player_id"]].copy()


def _app_rows(export_root: Path, season: int, event_key: str) -> pd.DataFrame:
    frame = pd.read_parquet(
        export_root / f"pga_{season}_shot_sg.parquet",
        columns=["event_key", "source_player_id", "sg_category"],
        filters=[
            ("event_key", "==", event_key),
            ("sg_category", "==", "app"),
        ],
    )
    return frame[["source_player_id"]].copy()


def _coverage_row(
    donor_rows: pd.DataFrame,
    bridge: pd.DataFrame,
    snapshot: pd.DataFrame,
    *,
    event_name: str,
    season: int,
    event_key: str,
    event_start: pd.Timestamp,
    category: str,
) -> dict[str, object]:
    work = donor_rows.copy()
    work["source_player_id"] = work["source_player_id"].astype(str)
    work = work.merge(
        bridge[["source_player_id", "identity_status", "dg_id"]],
        on="source_player_id",
        how="left",
        validate="many_to_one",
    )
    skill = snapshot.drop_duplicates("dg_id").copy()
    work = work.merge(
        skill[["dg_id", "sg_ott", "sg_app", "rel_dd", "skill_est", "std_dev"]],
        on="dg_id",
        how="left",
        validate="many_to_one",
    )
    players = work.drop_duplicates("source_player_id")
    row: dict[str, object] = {
        "event": event_name,
        "season": int(season),
        "event_key": event_key,
        "event_start": event_start.strftime("%Y-%m-%d"),
        "category": category,
        "donor_count": len(work),
        "donor_players": int(players["source_player_id"].nunique()),
        "snapshot_players": int(snapshot["dg_id"].nunique()),
        "identity_unique_shot_rate": float(work["identity_status"].eq("unique").mean()),
        "identity_unique_player_rate": float(players["identity_status"].eq("unique").mean()),
        "identity_ambiguous_shot_rate": float(
            work["identity_status"].eq("ambiguous").mean()
        ),
        "median_skill_snapshot_age_days": float(
            (event_start - snapshot["Date"]).dt.days.median()
        ),
        "skill_est_shot_coverage": float(work["skill_est"].notna().mean()),
        "sg_ott_shot_coverage": float(work["sg_ott"].notna().mean()),
        "sg_app_shot_coverage": float(work["sg_app"].notna().mean()),
        "rel_dd_shot_coverage": float(work["rel_dd"].notna().mean()),
        "skill_est_player_coverage": float(players["skill_est"].notna().mean()),
        "sg_ott_player_coverage": float(players["sg_ott"].notna().mean()),
        "sg_app_player_coverage": float(players["sg_app"].notna().mean()),
        "rel_dd_player_coverage": float(players["rel_dd"].notna().mean()),
        "id_contract": "ShotLink source_player_id -> yearly player_rounds canonical name -> model_results2 dg_id",
        "skill_snapshot_semantics": "latest model Date < event_start",
        "contract_role": "supplemental donor-shot-weighted coverage; primary gate is audit_shot_field_normalization_inputs.py",
    }
    return row


def _profile_inventory(path: Path, kind: str) -> dict[str, object]:
    frame = pd.read_csv(path, dtype={"source_player_id": str})
    return {
        "profile_kind": kind,
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "rows": len(frame),
        "unique_source_player_ids": int(frame["source_player_id"].nunique()),
        "unique_player_names": int(frame["player_name"].nunique()),
        "columns": "|".join(frame.columns),
        "materialized_cutoff_column_present": "profile_cutoff" in frame.columns,
        "materialized_source_hash_column_present": any(
            "sha256" in column or "hash" in column for column in frame.columns
        ),
        "training_window_inferred_from_producer": "2020-01-01 through 2025-12-31",
        "safe_role": "2026 target-player transform",
        "unsafe_role": "as-of donor-player transform for 2020-2025 donor events",
    }


def _weighted_summary(coverage: pd.DataFrame) -> list[dict[str, object]]:
    columns = [
        "identity_unique_shot_rate",
        "skill_est_shot_coverage",
        "sg_ott_shot_coverage",
        "sg_app_shot_coverage",
        "rel_dd_shot_coverage",
    ]
    result: list[dict[str, object]] = []
    for category, group in coverage.groupby("category", observed=True):
        weights = group["donor_count"].to_numpy(float)
        row: dict[str, object] = {
            "category": category,
            "events": len(group),
            "donor_count": int(group["donor_count"].sum()),
        }
        for column in columns:
            row[f"weighted_{column}"] = float(
                np.average(group[column].to_numpy(float), weights=weights)
            )
        result.append(row)
    return result


def run_audit(args: argparse.Namespace) -> dict[str, object]:
    replacements = _literal_name_replacements(args.repo / "sim_inputs.py")
    with _read_only_connection(args.db) as connection:
        events = _event_catalog(args.export_root, connection)
        database = pd.read_sql_query(
            """
            SELECT MIN(Date) AS min_date, MAX(Date) AS max_date,
                   COUNT(*) AS rows, COUNT(DISTINCT dg_id) AS players
            FROM model_results2
            """,
            connection,
        ).iloc[0]
        coverage_rows: list[dict[str, object]] = []
        for record in events.itertuples(index=False):
            snapshot = _skill_snapshot(connection, record.event_start, replacements)
            aliases = _source_aliases(args.export_root, int(record.season), replacements)
            bridge = _identity_bridge(aliases, snapshot)
            for category, donors in (
                ("tee", _tee_rows(args.export_root, int(record.season), record.event_key)),
                ("app_raw_category_proxy", _app_rows(args.export_root, int(record.season), record.event_key)),
            ):
                coverage_rows.append(
                    _coverage_row(
                        donors,
                        bridge,
                        snapshot,
                        event_name=record.event_name,
                        season=int(record.season),
                        event_key=record.event_key,
                        event_start=record.event_start,
                        category=category,
                    )
                )
    coverage = pd.DataFrame(coverage_rows)
    profiles = pd.DataFrame(
        [
            _profile_inventory(args.tee_profiles, "tee"),
            _profile_inventory(args.app_profiles, "app"),
        ]
    )
    args.output.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(args.output / "input_coverage.csv", index=False)
    profiles.to_csv(args.output / "profile_inventory.csv", index=False)
    payload: dict[str, object] = {
        "audit_version": "normalization_donor_support_supplement_v1",
        "contract_role": "supplemental_to_strictly_prior_name_bridge_v1",
        "authoritative_input_audit": {
            "script": str(PRIMARY_INPUT_AUDIT.resolve()),
            "output": str(PRIMARY_INPUT_AUDIT_OUTPUT.resolve()),
        },
        "events": list(EVENTS),
        "event_editions": len(events),
        "coverage_summary": _weighted_summary(coverage),
        "database": {
            "path": str(args.db.resolve()),
            "model_results2_min_date": str(database["min_date"]),
            "model_results2_max_date": str(database["max_date"]),
            "model_results2_rows": int(database["rows"]),
            "model_results2_players": int(database["players"]),
            "fields": [
                "dg_id",
                "Date",
                "sg_ott",
                "sg_app",
                "rel_dd",
                "skill_est",
                "std_dev",
            ],
            "producer": str((args.sim_prep / "golf_bt.py").resolve()),
            "producer_input_rule": "player_rounds.round_date < model_results2.Date",
            "consumer_cutoff_rule": "model_results2.Date < donor event_start",
        },
        "profiles": profiles.to_dict("records"),
        "findings": [
            "ShotLink source_player_id and DataGolf dg_id are different identifiers.",
            "The yearly pga_<year>_player_rounds name crosswalk is required before joining model_results2.",
            "Daily pre-event SG and relative-driving-distance snapshots exist locally and cover most, not all, recurring-course donor support.",
            "The current tee and APP profile CSVs are undated aggregate 2020-2025 snapshots; they are safe target transforms for a 2026 event but leak future spatial observations if used to neutralize a 2020-2025 donor.",
            "No materialized dated APP lie-by-distance spatial-profile table existed at audit time.",
            "The local ShotLink export starts in 2020, so the first 2020 donor editions cannot have a locally estimated strictly-prior spatial profile and require an explicit identity/population fallback.",
        ],
        "output_files": [
            "input_coverage.csv",
            "profile_inventory.csv",
            "donor_support_supplement.json",
        ],
    }
    (args.output / "donor_support_supplement.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    # An earlier exploratory run used this authoritative-sounding filename.
    # Keep a small in-place redirect rather than leaving two competing audits.
    (args.output / "input_audit.json").write_text(
        json.dumps(
            {
                "status": "superseded_non_authoritative_redirect",
                "authoritative_input_audit": str(PRIMARY_INPUT_AUDIT_OUTPUT.resolve()),
                "supplemental_donor_coverage": str(
                    (args.output / "donor_support_supplement.json").resolve()
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--sim-prep",
        type=Path,
        default=Path(r"C:\Users\McKinley Slade\dev\sim_prep"),
    )
    parser.add_argument("--tee-profiles", type=Path, default=DEFAULT_TEE_PROFILES)
    parser.add_argument("--app-profiles", type=Path, default=DEFAULT_APP_PROFILES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    payload = run_audit(args)
    print(json.dumps(payload["coverage_summary"], indent=2))
    print(f"Wrote normalization input audit to {args.output.resolve()}")


if __name__ == "__main__":
    main()
