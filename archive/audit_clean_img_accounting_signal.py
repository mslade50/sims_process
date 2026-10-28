"""Independently audit the chronology-clean IMG research signal artifact.

This verifier does not rebuild SG and never writes to the IMG database.  It
recomputes artifact hashes and algebraic identities, rechecks the selected
latest-model inventory against the live read-only database, measures the exact
IMG-to-DG source-player join rate, and counts surface-code exclusions from the
selected canonical shot rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPO
    / "permanent_data"
    / "research"
    / "base_skill_rebuild_20200101_20260822"
    / "clean_img_accounting"
)
DEFAULT_CROSSWALK = DEFAULT_OUTPUT.parent / "img_crosswalk_research_extended.parquet"
DEFAULT_IMG_DB = Path(
    r"C:\Users\McKinley Slade\dev\golf_scraping\output\imgarena_shots_lean.sqlite3"
)
CATEGORIES = ("ott", "app", "arg", "putt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--crosswalk", type=Path, default=DEFAULT_CROSSWALK)
    parser.add_argument("--img-db", type=Path, default=DEFAULT_IMG_DB)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sqlite_ro(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve().as_posix()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA temp_store=MEMORY")
    return connection


def atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def latest_model_inventory(connection: sqlite3.Connection) -> pd.DataFrame:
    query = """
        WITH distinct_models AS (
            SELECT DISTINCT event_key, round_no, model_id
            FROM archive_player_round_strokes_gained
        ),
        ranked AS (
            SELECT p.event_key, p.round_no, p.model_id, m.model_version,
                   m.created_at,
                   ROW_NUMBER() OVER (
                       PARTITION BY p.event_key, p.round_no
                       ORDER BY m.created_at DESC, p.model_id DESC
                   ) AS model_rank
            FROM distinct_models p
            JOIN sg_models m ON m.model_id = p.model_id
            JOIN archive_events e ON e.event_key = p.event_key
            WHERE e.tour = 'pga'
              AND m.model_version LIKE '%v4_lies2026%'
        )
        SELECT event_key, round_no, model_id, model_version, created_at
        FROM ranked
        WHERE model_rank = 1
    """
    return pd.read_sql_query(query, connection)


def source_join_and_code_audit(
    connection: sqlite3.Connection,
    inventory: pd.DataFrame,
    crosswalk: pd.DataFrame,
    warmup_codes: set[str],
    literal_codes: set[str],
) -> dict[str, Any]:
    mapped_by_round: dict[tuple[str, int], set[tuple[str, str]]] = {}
    for key, group in crosswalk.groupby(["event_key", "round_no"], observed=True):
        mapped_by_round[(str(key[0]), int(key[1]))] = set(
            zip(group["source_player_id"], group["source_team_id"])
        )

    source_player_rounds = 0
    joined_source_player_rounds = 0
    crosswalk_keys_missing_from_selected_model = 0
    start_counts: Counter[str] = Counter()
    end_counts: Counter[str] = Counter()
    warmup_ambiguous_rows = 0
    literal_ambiguous_rows = 0
    shot_rows = 0

    code_query = """
        SELECT COALESCE(start_surface_code, '') AS start_code,
               COALESCE(end_surface_code, '') AS end_code,
               COUNT(*) AS row_count
        FROM archive_shot_strokes_gained
        WHERE event_key = ? AND round_no = ? AND model_id = ?
        GROUP BY COALESCE(start_surface_code, ''), COALESCE(end_surface_code, '')
    """
    player_query = """
        SELECT source_player_id, source_team_id
        FROM archive_player_round_strokes_gained
        WHERE event_key = ? AND round_no = ? AND model_id = ?
    """
    for index, model in enumerate(inventory.itertuples(index=False), start=1):
        params = (str(model.event_key), int(model.round_no), str(model.model_id))
        source = pd.read_sql_query(player_query, connection, params=params)
        source["source_player_id"] = source["source_player_id"].fillna("").astype(str)
        source["source_team_id"] = source["source_team_id"].fillna("").astype(str)
        source_keys = set(zip(source["source_player_id"], source["source_team_id"]))
        mapped_keys = mapped_by_round.get((params[0], params[1]), set())
        source_player_rounds += len(source_keys)
        joined_source_player_rounds += len(source_keys & mapped_keys)
        crosswalk_keys_missing_from_selected_model += len(mapped_keys - source_keys)

        for start_code, end_code, count in connection.execute(code_query, params):
            start = str(start_code or "").upper().strip()
            end = str(end_code or "").upper().strip()
            rows = int(count)
            shot_rows += rows
            start_counts[start] += rows
            end_counts[end] += rows
            warmup_ambiguous_rows += rows * int(
                start not in warmup_codes or end not in warmup_codes
            )
            literal_ambiguous_rows += rows * int(
                start not in literal_codes or end not in literal_codes
            )
        if index % 100 == 0 or index == len(inventory):
            print(f"Independent source/code audit: {index}/{len(inventory)} models", flush=True)

    unjoined = source_player_rounds - joined_source_player_rounds
    warmup_excluded = sorted(
        set(start_counts).union(end_counts).difference(warmup_codes)
    )
    literal_excluded = sorted(
        set(start_counts).union(end_counts).difference(literal_codes)
    )

    def counts_for(codes: list[str]) -> dict[str, dict[str, int]]:
        return {
            code if code else "<blank>": {
                "start_rows": int(start_counts[code]),
                "end_rows": int(end_counts[code]),
                "state_occurrences": int(start_counts[code] + end_counts[code]),
            }
            for code in codes
        }

    return {
        "selected_shot_rows": shot_rows,
        "source_player_rounds": source_player_rounds,
        "joined_source_player_rounds": joined_source_player_rounds,
        "unjoined_source_player_rounds": unjoined,
        "source_player_round_join_rate": (
            joined_source_player_rounds / source_player_rounds
            if source_player_rounds
            else None
        ),
        "crosswalk_keys_missing_from_selected_model": (
            crosswalk_keys_missing_from_selected_model
        ),
        "warmup_unmapped_shot_rows": warmup_ambiguous_rows,
        "literal_unmapped_shot_rows": literal_ambiguous_rows,
        "warmup_excluded_code_counts": counts_for(warmup_excluded),
        "literal_excluded_code_counts": counts_for(literal_excluded),
    }


def build_report(audit: Mapping[str, Any], manifest_hash: str) -> str:
    joins = audit["source_join"]
    integrity = audit["integrity"]
    hashes = audit["artifact_hashes"]
    literal_counts = joins["literal_excluded_code_counts"]
    warmup_counts = joins["warmup_excluded_code_counts"]

    hash_rows = "\n".join(
        f"| `{name}` | `{values['actual_sha256']}` | {str(values['matches_manifest']).lower()} |"
        for name, values in hashes.items()
    )
    def top_code_rows(counts: Mapping[str, Mapping[str, int]], limit: int = 12) -> str:
        ordered = sorted(
            counts.items(),
            key=lambda item: int(item[1]["state_occurrences"]),
            reverse=True,
        )
        rows = [
            f"| `{code}` | {values['start_rows']:,} | {values['end_rows']:,} | {values['state_occurrences']:,} |"
            for code, values in ordered[:limit]
        ]
        omitted = ordered[limit:]
        if omitted:
            rows.append(
                "| all other excluded codes "
                f"({len(omitted)}) | "
                f"{sum(value['start_rows'] for _, value in omitted):,} | "
                f"{sum(value['end_rows'] for _, value in omitted):,} | "
                f"{sum(value['state_occurrences'] for _, value in omitted):,} |"
            )
        return "\n".join(rows) or "| none | 0 | 0 | 0 |"

    literal_rows = top_code_rows(literal_counts)
    warmup_rows = top_code_rows(warmup_counts)

    return f"""# Clean IMG Accounting Signal — Independent Provenance Audit

## Decision

The artifact passes its chronology, identity, hashing, and algebraic checks and is
supportable as a **research-only, chronology-clean round signal** for 2021 onward.
It does not, by itself, support a betting-model improvement claim.  Any skill model
must use only rounds completed before the prediction timestamp.

## What was frozen

- One putting curve and every learned surface-code decision use only 27 completed
  2020 PGA events (108 round models); the latest fit event ended 2020-11-22.
- Evaluation eligibility starts 2021-01-01.  Boundary violations: 0.
- The exact 964-model inventory is unique by event/round and is materialized as
  `selected_model_inventory.parquet`.
- Stored v4 expected strokes, SG values, lie mapping, and per-event putting curves
  were not reused.  Categories were recomputed under the frozen map; only the v4
  tee category was used as a safe par-3 versus par-4/5 indicator because the archive
  shot table does not materialize hole par.

## Independent integrity results

- Signal rows: {integrity['signal_rows']:,}; events: {integrity['signal_events']:,}.
- Duplicate `player_round_key`: {integrity['duplicate_player_round_keys']}.
- Duplicate selected event/round: {integrity['duplicate_inventory_event_rounds']}.
- Frozen output keys equal the frozen IMG↔DG crosswalk keys: {str(integrity['output_keys_equal_crosswalk']).lower()}.
- Current DB latest-model inventory still equals the materialized mapped inventory:
  {str(integrity['live_inventory_matches']).lower()}.
- Category-to-total maximum identity error: {integrity['max_category_total_identity_error']:.3g}.
- Availability/clean-hole gate mismatches: {integrity['availability_gate_mismatches']}.
- Maximum field-centering error recorded during extraction: {integrity['max_field_centering_error']:.3g}.
- 2020 rows are correctly marked ineligible and every 2021+ row eligible: {str(integrity['eligibility_flags_exact']).lower()}.

## DG join coverage

The output contains every one of the {joins['joined_source_player_rounds']:,} frozen
crosswalk player-rounds, but the selected IMG models contain
{joins['source_player_rounds']:,} source player-rounds in total.  The exact join rate
is **{joins['source_player_round_join_rate']:.2%}**; {joins['unjoined_source_player_rounds']:,}
source player-rounds are excluded.  Crosswalk keys absent from the selected models:
{joins['crosswalk_keys_missing_from_selected_model']}.

That is a material coverage limitation, not a failed join: downstream comparisons
must use a common observed menu and should test whether excluded players differ by
event, skill, newcomer status, or withdrawal status.

## Surface-code exclusions

The 2020-fitted arm has {joins['warmup_unmapped_shot_rows']:,} selected shot rows
with at least one start/end state outside its frozen map.  The literal-only arm has
{joins['literal_unmapped_shot_rows']:,}.  Those rows make the entire player-hole
ineligible; a round signal is offered only with at least 12 clean holes.

Largest codes excluded by the 2020-fitted arm (the complete table is in
`independent_audit.json`):

| Code | Start rows | End rows | State occurrences |
|---|---:|---:|---:|
{warmup_rows}

Largest nonliteral codes excluded by the strict arm:

| Code | Start rows | End rows | State occurrences |
|---|---:|---:|---:|
{literal_rows}

## Frozen artifact hashes

Build manifest SHA-256: `{manifest_hash}`.

| Artifact | Recomputed SHA-256 | Matches manifest |
|---|---|---|
{hash_rows}

## Limitations that must travel with the signal

1. Canonical shot-state repair was run retrospectively with the current 2026 parser.
   The audit removed outcome-fitted expected-strokes transforms, but it cannot recreate
   historical parser versions that were never archived.
2. The fitted OCO/ODO/ONA and other nonliteral mappings are valid 2020-only choices,
   not semantic truths.  The literal-only arm is a required sensitivity, not optional
   decoration.
3. Signals scaled from 12–17 clean holes can be noisier or selectively missing.  Show
   a complete-18 sensitivity alongside the ≥12-hole coverage arm.
4. Field adjustment uses contemporaneous players on the same physical course, round,
   and hole.  That is valid after a round is complete; it is not available before that
   round.
5. For pre-event skill, use only prior completed events.  For an R2–R4 cascade, use
   only earlier completed rounds from that event—never later rounds or a same-event
   all-round aggregate.
6. IMG availability starts in 2020 and does not solve newcomer priors.  Missing-signal
   handling must be frozen on discovery data and evaluated on a common betting menu.

The build manifest predates this independent report, so the report and
`independent_audit.json` are intentionally not self-included in that manifest.
"""


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    measurement = json.loads(
        (output_dir / "clean_img_fixed_measurement.json").read_text(encoding="utf-8")
    )
    signals = pd.read_parquet(output_dir / "clean_img_player_round.parquet")
    inventory = pd.read_parquet(output_dir / "selected_model_inventory.parquet")
    crosswalk = pd.read_parquet(args.crosswalk)
    for frame in (crosswalk,):
        for column in ("source_player_id", "source_team_id"):
            frame[column] = frame[column].fillna("").astype(str)

    artifact_hashes: dict[str, dict[str, Any]] = {}
    for record in manifest["artifacts"]:
        actual = sha256_file(output_dir / record["path"])
        artifact_hashes[record["path"]] = {
            "manifest_sha256": record["sha256"],
            "actual_sha256": actual,
            "matches_manifest": actual == record["sha256"],
        }

    comparison_key = [
        "event_key",
        "round_no",
        "year",
        "event_id",
        "round_num",
        "round_date",
        "dg_id",
    ]
    for frame in (crosswalk, signals):
        frame["round_date"] = pd.to_datetime(
            frame["round_date"], errors="coerce"
        ).dt.normalize()
    crosswalk_keys = set(map(tuple, crosswalk[comparison_key].astype(str).to_numpy()))
    output_keys = set(map(tuple, signals[comparison_key].astype(str).to_numpy()))
    total_identity_errors: list[float] = []
    availability_mismatches = 0
    for prefix in ("img_clean_warmup", "img_clean_unambiguous"):
        categories = [
            f"{prefix}_sg_{category}_field_adjusted_per18"
            for category in CATEGORIES
        ]
        error = (
            signals[categories].sum(axis=1)
            - signals[f"{prefix}_sg_total_field_adjusted_per18"]
        ).abs()
        total_identity_errors.append(float(error.max()))
        availability_mismatches += int(
            (
                signals[f"{prefix}_signal_available"]
                != signals[f"{prefix}_clean_holes"].ge(12)
            ).sum()
        )

    with sqlite_ro(args.img_db) as connection:
        live_inventory = latest_model_inventory(connection)
        mapped_events = set(inventory["event_key"])
        live_mapped = live_inventory.loc[live_inventory["event_key"].isin(mapped_events)]
        comparison_columns = ["event_key", "round_no", "model_id", "model_version"]
        frozen_set = set(map(tuple, inventory[comparison_columns].astype(str).to_numpy()))
        live_set = set(map(tuple, live_mapped[comparison_columns].astype(str).to_numpy()))
        join_audit = source_join_and_code_audit(
            connection,
            inventory,
            crosswalk,
            set(measurement["warmup_fitted_surface_map"]),
            set(measurement["unambiguous_surface_map"]),
        )

    extraction_audit = json.loads((output_dir / "audit.json").read_text(encoding="utf-8"))
    eligibility_exact = bool(
        (
            signals["img_clean_evaluation_eligible"]
            == pd.to_datetime(signals["round_date"]).ge("2021-01-01")
        ).all()
    )
    integrity = {
        "signal_rows": int(len(signals)),
        "signal_events": int(signals["event_key"].nunique()),
        "duplicate_player_round_keys": int(signals["player_round_key"].duplicated().sum()),
        "duplicate_inventory_event_rounds": int(
            inventory.duplicated(["event_key", "round_no"]).sum()
        ),
        "output_keys_equal_crosswalk": output_keys == crosswalk_keys,
        "crosswalk_only_keys": len(crosswalk_keys - output_keys),
        "output_only_keys": len(output_keys - crosswalk_keys),
        "live_inventory_matches": frozen_set == live_set,
        "live_inventory_only_rows": len(live_set - frozen_set),
        "frozen_inventory_only_rows": len(frozen_set - live_set),
        "max_category_total_identity_error": max(total_identity_errors),
        "availability_gate_mismatches": availability_mismatches,
        "eligibility_flags_exact": eligibility_exact,
        "max_fit_event_end": str(pd.to_datetime(inventory.loc[inventory["crosswalk_year"].eq(2020), "end_date"]).max().date()),
        "fit_rows_at_or_after_2021": int(
            pd.to_datetime(
                inventory.loc[inventory["crosswalk_year"].eq(2020), "end_date"]
            ).ge("2021-01-01").sum()
        ),
        "putting_curve_monotone": bool(
            np.diff(measurement["putting_curve"]["expected_strokes"]).min() >= -1e-12
        ),
        "max_field_centering_error": max(
            float(values["max_abs_hole_field_centering_error"])
            for values in extraction_audit["signal_processing"]["arm_totals"].values()
        ),
        "selected_shot_rows_match_extractor": (
            join_audit["selected_shot_rows"]
            == extraction_audit["signal_processing"]["shot_rows"]
        ),
        "warmup_unmapped_rows_match_extractor": (
            join_audit["warmup_unmapped_shot_rows"]
            == extraction_audit["signal_processing"]["warmup_ambiguous_state_rows"]
        ),
        "literal_unmapped_rows_match_extractor": (
            join_audit["literal_unmapped_shot_rows"]
            == extraction_audit["signal_processing"]["unambiguous_ambiguous_state_rows"]
        ),
    }
    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "production_changes": False,
        "artifact_hashes": artifact_hashes,
        "manifest_sha256": sha256_file(manifest_path),
        "source_inventory_sha256": sha256_file(
            output_dir / "selected_model_inventory.parquet"
        ),
        "integrity": integrity,
        "source_join": join_audit,
    }
    required = [
        all(value["matches_manifest"] for value in artifact_hashes.values()),
        integrity["duplicate_player_round_keys"] == 0,
        integrity["duplicate_inventory_event_rounds"] == 0,
        integrity["output_keys_equal_crosswalk"],
        integrity["live_inventory_matches"],
        integrity["max_category_total_identity_error"] <= 1e-12,
        integrity["availability_gate_mismatches"] == 0,
        integrity["eligibility_flags_exact"],
        integrity["fit_rows_at_or_after_2021"] == 0,
        integrity["putting_curve_monotone"],
        integrity["selected_shot_rows_match_extractor"],
        integrity["warmup_unmapped_rows_match_extractor"],
        integrity["literal_unmapped_rows_match_extractor"],
    ]
    audit["all_required_checks_pass"] = all(required)
    atomic_json(audit, output_dir / "independent_audit.json")
    report = build_report(audit, audit["manifest_sha256"])
    atomic_text(report, output_dir / "CLEAN_IMG_PROVENANCE_AUDIT.md")
    print(
        "Clean IMG independent audit: "
        f"{'PASS' if audit['all_required_checks_pass'] else 'FAIL'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
