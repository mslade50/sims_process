"""Build event-dated TEE/APP player spatial profiles from prior ShotLink only.

This research builder closes the historical-profile gap in the spatial hole
simulator.  It creates a fresh player profile at every requested event start,
using only shots from PGA events that had already completed.  Same-date and
overlapping events cannot enter one another's snapshot.

The default manifest is the audited Sony/Travelers/Wyndham panel: 18 donor
editions from 2020-2025 and three 2026 targets.  Static pre-2026 profile CSVs
are never read.  The only identity input is the separately audited,
strictly-prior ShotLink-name to DataGolf bridge.

Player-state taxonomy is shared with the integrated transition evaluator:

* TEE: ``tee|par=4`` and ``tee|par=5``;
* APP: ``app|lie=<lie>|band=<audited four-band value>``.

Profiles are setup-independent.  ``materialize_normalization_profiles`` joins
them to a strict-prior setup/reference-center manifest immediately before atom
normalization.  That avoids a large player x hole x round Cartesian artifact.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import threading
import time
from typing import Iterable

import numpy as np
import pandas as pd
import psutil


ARCHIVE = Path(__file__).resolve().parent
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

import hole_sim_app_transition as app  # noqa: E402
import hole_sim_tee_transition as tee  # noqa: E402


DEFAULT_EXPORT = Path(
    r"C:\Users\McKinley Slade\dev\golf_scraping\output\export"
)
DEFAULT_HISTORY_DB = Path(
    r"C:\Users\McKinley Slade\dev\sim_prep\work\dg_historical.db"
)
DEFAULT_BRIDGE = Path(
    "permanent_data/shot_field_normalization_input_audit/player_identity_snapshot.csv"
)
DEFAULT_BRIDGE_MANIFEST = DEFAULT_BRIDGE.with_name("manifest.json")
DEFAULT_OUTPUT = Path("permanent_data/hole_sim_asof_spatial_profiles")

EXPECTED_EVENTS = (
    "Sony Open in Hawaii",
    "Travelers Championship",
    "Wyndham Championship",
)
APP_LIES = tuple(app.START_LIES)
APP_BANDS = ("<=100", "100-150", "150-215", ">215")
TEE_STATES = ((4, "tee|par=4"), (5, "tee|par=5"))
PRIOR_ORIGIN = pd.Timestamp("2019-12-31", tz="UTC")
PROFILE_VERSION = "strict_prior_spatial_profiles_v1"


@dataclass(frozen=True)
class BuilderConfig:
    start_year: int = 2020
    end_year: int = 2026
    tee_mean_prior: float = 50.0
    tee_variance_prior: float = 120.0
    app_cross_mean_prior: float = 200.0
    app_along_mean_prior: float = 200.0
    app_variance_prior: float = 450.0


def _utc(series: pd.Series, label: str) -> pd.Series:
    values = pd.to_datetime(series, errors="coerce", utc=True)
    if values.isna().any():
        sample = series[values.isna()].astype(str).head(5).tolist()
        raise ValueError(f"{label} contains invalid dates: {sample}")
    return values


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_sha256(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> str:
    work = frame[list(columns) if columns is not None else sorted(frame.columns)].copy()
    schema = [(column, str(work[column].dtype)) for column in work.columns]
    hashes = pd.util.hash_pandas_object(work, index=False).to_numpy("uint64")
    hashes.sort()
    digest = sha256(json.dumps(schema, separators=(",", ":")).encode("utf-8"))
    digest.update(hashes.tobytes())
    return digest.hexdigest()


def _audited_bridge_frame_hash(frame: pd.DataFrame) -> str:
    """Reproduce the bridge producer's deterministic content hash."""

    ordered = frame.reindex(sorted(frame.columns), axis=1).copy()
    if len(ordered):
        ordered = ordered.sort_values(list(ordered.columns), kind="mergesort")
    payload = ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return sha256(payload).hexdigest()


def _name_key(value: object) -> str:
    text = re.sub(r"\s*#\d+\s*$", "", str(value).casefold())
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def load_event_calendar(history_db: Path, events_parquet: Path) -> pd.DataFrame:
    """Load physical event start/end dates used as the leakage boundary."""

    # `events.parquet` has null PGA dates in the current R2 export and
    # `player_tourney` ends in 2024.  The round table is complete through 2026.
    # MAX(round_date) is treated as the completion boundary; a candidate event
    # must have a strictly earlier final-round date than the target start.
    with sqlite3.connect(f"file:{history_db.as_posix()}?mode=ro", uri=True) as db:
        rounds = pd.read_sql_query(
            """
            SELECT year AS actual_year, event_id, event_name,
                   MIN(round_date) AS event_start,
                   MAX(round_date) AS event_end
            FROM player_rounds
            WHERE tour = 'pga' AND year BETWEEN 2019 AND 2026
            GROUP BY year, event_id, event_name
            """,
            db,
        )
        tourneys = pd.read_sql_query(
            """
            SELECT year AS actual_year, event_id, event_name,
                   MIN(start_date) AS event_start,
                   MAX(end_date) AS event_end
            FROM player_tourney
            WHERE tour = 'pga' AND year BETWEEN 2019 AND 2026
            GROUP BY year, event_id, event_name
            """,
            db,
        )
    frame = pd.concat([rounds, tourneys], ignore_index=True)
    frame = frame.drop_duplicates(
        ["actual_year", "event_id", "event_name"], keep="first"
    )
    frame["event_start"] = _utc(frame["event_start"], "calendar event_start")
    frame["event_end"] = _utc(frame["event_end"], "calendar event_end")
    frame["event_name_key"] = frame["event_name"].map(_name_key)

    r2 = pd.read_parquet(
        events_parquet,
        columns=["event_key", "tour", "season", "event_name", "start_date", "end_date"],
    )
    r2 = r2[r2["tour"].eq("pga") & r2["season"].between(2020, 2026)].copy()
    parsed = r2["event_key"].str.extract(r"^pga:R(\d{4})(\d+)$")
    r2["event_id"] = pd.to_numeric(parsed[1], errors="coerce")
    r2["event_name_key"] = r2["event_name"].map(_name_key)
    r2["r2_start"] = pd.to_datetime(r2["start_date"], errors="coerce", utc=True)
    r2["r2_end"] = pd.to_datetime(r2["end_date"], errors="coerce", utc=True)

    rows: list[dict[str, object]] = []
    unresolved: list[str] = []
    for event in r2.itertuples(index=False):
        candidates = frame[
            frame["event_id"].eq(event.event_id)
            & frame["actual_year"].isin([int(event.season) - 1, int(event.season)])
        ].copy()
        exact = candidates[candidates["event_name_key"].eq(event.event_name_key)]
        if len(exact) > 1:
            if event.event_name_key == "tour championship":
                # The season-ending TOUR Championship can extend into early
                # September but still belongs to the ending calendar season.
                exact = exact["actual_year"].eq(int(event.season)).pipe(
                    lambda mask: exact[mask]
                )
            else:
                # Calendar-year fall events belong to the following PGA season;
                # January-August events belong to their calendar-year season.
                inferred_season = exact["actual_year"] + np.where(
                    exact["event_start"].dt.month.ge(9), 1, 0
                )
                exact = exact[inferred_season.eq(int(event.season))]
            if exact.empty:
                exact = candidates[
                    candidates["event_name_key"].eq(event.event_name_key)
                    & candidates["actual_year"].eq(int(event.season))
                ]
        if len(exact) == 1:
            selected = exact.iloc[0]
        elif len(candidates) == 1:
            selected = candidates.iloc[0]
        elif pd.notna(event.r2_start) and pd.notna(event.r2_end):
            selected = pd.Series(
                {
                    "actual_year": pd.Timestamp(event.r2_start).year,
                    "event_start": event.r2_start,
                    "event_end": event.r2_end,
                }
            )
        else:
            unresolved.append(str(event.event_key))
            continue
        rows.append(
            {
                "event_key": event.event_key,
                "season": int(event.season),
                "actual_year": int(selected["actual_year"]),
                "event_id": int(event.event_id),
                "event_name": event.event_name,
                "event_start": selected["event_start"],
                "event_end": selected["event_end"],
                "calendar_source": (
                    "player_rounds_season_mapped"
                    if "event_name_key" in selected.index
                    else "r2_metadata_date"
                ),
            }
        )
    out = pd.DataFrame(rows)
    if out.duplicated("event_key").any():
        raise ValueError("historical calendar has duplicate R2 event keys")
    out.attrs["unresolved_event_keys"] = unresolved
    out.attrs["source_files"] = {
        "history_db": {
            "path": str(history_db.resolve()),
            "size_bytes": int(history_db.stat().st_size),
            "sha256": _file_sha256(history_db),
        },
        "events_parquet": {
            "path": str(events_parquet.resolve()),
            "size_bytes": int(events_parquet.stat().st_size),
            "sha256": _file_sha256(events_parquet),
        },
    }
    return out


def load_strict_bridge(
    path: Path,
    manifest_path: Path | None = None,
    *,
    expected_events: tuple[str, ...] = EXPECTED_EVENTS,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Validate the audited name/DG bridge and its strict-prior cutoffs."""

    manifest_path = manifest_path or path.with_name("manifest.json")
    if not path.exists() or not manifest_path.exists():
        raise FileNotFoundError("audited strict-prior bridge or manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    accepted_contracts = {
        "strictly_prior_name_bridge_v1",
        "strict_prior_skill_global_identity_alias_v2",
    }
    if manifest.get("contract") not in accepted_contracts:
        raise ValueError("unsupported or unaudited identity bridge contract")
    if int(manifest.get("strict_cutoff_violations", -1)) != 0:
        raise ValueError("identity bridge manifest reports cutoff violations")
    frame = pd.read_csv(path, dtype={"source_player_id": str})
    expected_file_hash = manifest.get("player_artifact_sha256")
    observed_file_hash = _file_sha256(path)
    if expected_file_hash and observed_file_hash != expected_file_hash:
        raise ValueError(
            "identity bridge file does not match its audited manifest: "
            f"expected={expected_file_hash}, observed={observed_file_hash}"
        )
    required = {
        "event_key", "event_name", "season", "event_start",
        "source_player_id", "player_name", "name_key", "skill_date", "dg_id",
        "skill_est", "sg_ott", "sg_app", "ambiguous_name",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"identity bridge is missing columns: {missing}")
    frame["event_start"] = _utc(frame["event_start"], "bridge event_start")
    frame["skill_date"] = pd.to_datetime(frame["skill_date"], errors="coerce", utc=True)
    invalid_cutoff = frame["skill_date"].notna() & (
        frame["skill_date"] >= frame["event_start"]
    )
    if invalid_cutoff.any():
        raise ValueError("identity bridge contains a non-prior skill snapshot")
    if frame.duplicated(["event_key", "source_player_id"]).any():
        raise ValueError("identity bridge has duplicate event/player rows")
    if frame["ambiguous_name"].fillna(False).astype(bool).any():
        raise ValueError("identity bridge contains ambiguous name matches")
    observed_events = set(frame["event_name"].dropna().astype(str))
    missing_events = sorted(set(expected_events) - observed_events)
    if missing_events:
        raise ValueError(f"identity bridge is missing events: {missing_events}")
    event_rows = frame[["event_key", "event_name", "season", "event_start"]].drop_duplicates()
    # The default audit should contain exactly 18 historical donor editions
    # and three 2026 target editions.  Custom bridges can still use the loader
    # by passing a different event tuple.
    if set(expected_events) == set(EXPECTED_EVENTS):
        donors = event_rows["season"].between(2020, 2025).sum()
        targets = event_rows["season"].eq(2026).sum()
        if donors < 18 or targets < 3:
            raise ValueError(
                f"default bridge is incomplete: donor editions={donors}, targets={targets}"
            )
    metadata = {
        "bridge_path": str(path.resolve()),
        "bridge_sha256": _file_sha256(path),
        "bridge_logical_sha256_from_manifest": manifest.get("player_audit_sha256"),
        "bridge_serialized_hash_verified": bool(expected_file_hash),
        "bridge_manifest_path": str(manifest_path.resolve()),
        "bridge_manifest_sha256": _file_sha256(manifest_path),
        "bridge_contract": manifest["contract"],
    }
    return frame, metadata


def _attach_completed_event_dates(
    frame: pd.DataFrame, calendar: pd.DataFrame, label: str
) -> pd.DataFrame:
    out = frame.merge(
        calendar[["event_key", "event_start", "event_end"]],
        on="event_key",
        how="left",
        validate="many_to_one",
    )
    missing = out["event_end"].isna()
    if missing.any():
        keys = out.loc[missing, "event_key"].drop_duplicates().head(10).tolist()
        raise ValueError(f"{label} contains events absent from calendar: {keys}")
    return out


def load_spatial_history(
    export_root: Path,
    calendar: pd.DataFrame,
    config: BuilderConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Read TEE and APP spatial history once across all configured years."""

    tee_parts: list[pd.DataFrame] = []
    app_parts: list[pd.DataFrame] = []
    source_files: dict[str, dict[str, object]] = {}
    for year in range(config.start_year, config.end_year + 1):
        print(f"Loading strict-prior spatial history: {year}")
        paths = [
            export_root / f"pga_{year}_shots.parquet",
            export_root / f"pga_{year}_shot_sg.parquet",
            export_root / f"pga_{year}_hole_geometry.parquet",
        ]
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(path)
            stat = path.stat()
            source_files[path.name] = {
                "path": str(path.resolve()),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "sha256": _file_sha256(path),
            }

        tee_geometry = tee._read_year_geometry(export_root, year)
        drives = tee._add_local_coordinates(
            tee._read_drives(export_root, year), tee_geometry
        )
        drives["source_player_id"] = drives["source_player_id"].astype(str)
        tee_parts.append(
            drives[
                [
                    "event_key", "round_no", "hole_no", "source_player_id",
                    "player_name", "long_yd", "lat_yd", "shot_key",
                ]
            ].copy()
        )

        app_geometry = app._read_geometry(export_root, year)
        app_rows = app._read_approach_rows(export_root, year)
        prepared = app._apply_reference_pins(
            app._attach_geometry(app_rows, app_geometry)
        )
        continuous = prepared["spatial_valid"] & ~prepared["is_holeout"]
        app_parts.append(
            prepared.loc[
                continuous,
                [
                    "event_key", "round_no", "hole_no", "source_player_id",
                    "player_name", "shot_key", "start_lie_class",
                    "start_distance_yd", "miss_long_yd", "miss_lat_yd",
                    "spatial_valid",
                ],
            ].copy()
        )

    tee_history = _attach_completed_event_dates(
        pd.concat(tee_parts, ignore_index=True), calendar, "TEE history"
    )
    app_history = _attach_completed_event_dates(
        pd.concat(app_parts, ignore_index=True), calendar, "APP history"
    )
    return tee_history, app_history, {
        "source_files": source_files,
        "tee_rows": int(len(tee_history)),
        "app_rows": int(len(app_history)),
    }


def strict_prior_rows(history: pd.DataFrame, event_start: pd.Timestamp) -> pd.DataFrame:
    """Return completed-event history only; fail if the boundary is violated."""

    cutoff = pd.Timestamp(event_start)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")
    eligible = history[history["event_end"] < cutoff].copy()
    if not eligible.empty and not (eligible["event_end"] < cutoff).all():
        raise AssertionError("strict-prior completed-event boundary failed")
    return eligible


def _event_cutoff(history: pd.DataFrame) -> pd.Timestamp:
    return PRIOR_ORIGIN if history.empty else pd.Timestamp(history["event_end"].max())


def _player_last_dates(history: pd.DataFrame) -> pd.Series:
    if history.empty:
        return pd.Series(dtype="datetime64[ns, UTC]")
    return history.groupby("source_player_id", observed=True)["event_end"].max()


def _tee_event_profiles(
    roster: pd.DataFrame,
    history: pd.DataFrame,
    event: pd.Series,
    config: BuilderConfig,
) -> pd.DataFrame:
    if history.empty:
        fitted = pd.DataFrame(
            columns=[
                "source_player_id", "n_drives", "distance_ratio",
                "lateral_bias_yd", "long_scale", "lat_scale",
            ]
        )
        globals_: dict[str, float] = {"global_long_sd": np.nan, "global_lat_sd": np.nan}
    else:
        fitted, globals_ = tee.fit_player_profiles(
            history,
            mean_prior=config.tee_mean_prior,
            variance_prior=config.tee_variance_prior,
        )
    fitted["source_player_id"] = fitted["source_player_id"].astype(str)
    base = roster.merge(
        fitted[
            [
                "source_player_id", "n_drives", "distance_ratio",
                "lateral_bias_yd", "long_scale", "lat_scale",
            ]
        ],
        on="source_player_id",
        how="left",
        validate="one_to_one",
    )
    base["spatial_profile_available"] = base["n_drives"].notna()
    fills = {
        "n_drives": 0, "distance_ratio": 1.0, "lateral_bias_yd": 0.0,
        "long_scale": 1.0, "lat_scale": 1.0,
    }
    base = base.fillna(value=fills)
    parts = []
    for par, state in TEE_STATES:
        part = base.copy()
        part["sg_category"] = "ott"
        part["state_key"] = state
        part["par"] = par
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out["long_bias_yd"] = 0.0
    out["lat_bias_yd"] = out["lateral_bias_yd"]
    out["long_lat_corr"] = 0.0
    out["n_spatial"] = out["n_drives"].astype(int)
    out["n_state"] = out["n_drives"].astype(int)
    out["spatial_profile_source"] = np.where(
        out["spatial_profile_available"], "tee_expanding_prior_shotlink", "neutral_prior_no_tee_history"
    )
    out["global_long_sd"] = globals_["global_long_sd"]
    out["global_lat_sd"] = globals_["global_lat_sd"]
    return out.drop(columns="lateral_bias_yd")


def _app_event_profiles(
    roster: pd.DataFrame,
    history: pd.DataFrame,
    event: pd.Series,
    config: BuilderConfig,
) -> pd.DataFrame:
    if history.empty:
        fitted = pd.DataFrame(
            columns=[
                "source_player_id", "start_lie_class", "profile_distance_band",
                "n_band", "long_bias_yd", "n_app", "lat_bias_yd",
                "long_scale", "lat_scale",
            ]
        )
        globals_: dict[str, float] = {"global_long_sd": np.nan, "global_lat_sd": np.nan}
    else:
        fitted, globals_ = app.fit_player_profiles(
            history,
            cross_mean_prior=config.app_cross_mean_prior,
            along_mean_prior=config.app_along_mean_prior,
            variance_prior=config.app_variance_prior,
        )
    fitted["source_player_id"] = fitted["source_player_id"].astype(str)
    player_base = (
        fitted.groupby("source_player_id", observed=True, as_index=False)
        .agg(
            n_app=("n_app", "max"),
            lat_bias_yd=("lat_bias_yd", "first"),
            long_scale=("long_scale", "first"),
            lat_scale=("lat_scale", "first"),
        )
        if not fitted.empty
        else pd.DataFrame(
            columns=["source_player_id", "n_app", "lat_bias_yd", "long_scale", "lat_scale"]
        )
    )
    state_grid = pd.MultiIndex.from_product(
        [roster["source_player_id"].astype(str), APP_LIES, APP_BANDS],
        names=["source_player_id", "start_lie_class", "profile_distance_band"],
    ).to_frame(index=False)
    out = state_grid.merge(roster, on="source_player_id", how="left", validate="many_to_one")
    out = out.merge(player_base, on="source_player_id", how="left", validate="many_to_one")
    out = out.merge(
        fitted[
            [
                "source_player_id", "start_lie_class", "profile_distance_band",
                "n_band", "long_bias_yd",
            ]
        ],
        on=["source_player_id", "start_lie_class", "profile_distance_band"],
        how="left",
        validate="one_to_one",
    )
    out["spatial_profile_available"] = out["n_app"].notna()
    out["state_profile_available"] = out["n_band"].notna()
    out = out.fillna(
        value={
            "n_app": 0, "n_band": 0, "long_bias_yd": 0.0,
            "lat_bias_yd": 0.0, "long_scale": 1.0, "lat_scale": 1.0,
        }
    )
    out["sg_category"] = "app"
    out["state_key"] = [
        f"app|lie={lie}|band={band}"
        for lie, band in zip(out["start_lie_class"], out["profile_distance_band"])
    ]
    out["distance_ratio"] = 1.0
    out["long_lat_corr"] = 0.0
    out["n_spatial"] = out["n_app"].astype(int)
    out["n_state"] = out["n_band"].astype(int)
    out["spatial_profile_source"] = np.select(
        [out["state_profile_available"], out["spatial_profile_available"]],
        ["app_state_expanding_prior_shotlink", "app_player_prior_neutral_unseen_state"],
        default="neutral_prior_no_app_history",
    )
    out["global_long_sd"] = globals_["global_long_sd"]
    out["global_lat_sd"] = globals_["global_lat_sd"]
    return out


def _merge_snapshot_provenance(
    profiles: pd.DataFrame,
    roster: pd.DataFrame,
    event: pd.Series,
    history: pd.DataFrame,
) -> pd.DataFrame:
    event_start = pd.Timestamp(event["event_start"])
    cutoff = _event_cutoff(history)
    if cutoff >= event_start:
        raise AssertionError("profile cutoff must strictly predate target event")
    player_last = _player_last_dates(history)
    out = profiles.copy()
    out["event_key"] = event["event_key"]
    out["event_name"] = event["event_name"]
    out["season"] = int(event["season"])
    out["event_start"] = event_start
    skill_dates = pd.to_datetime(out["skill_date"], errors="coerce", utc=True)
    out["profile_cutoff"] = pd.concat(
        [
            pd.Series([cutoff] * len(out), index=out.index),
            skill_dates,
        ],
        axis=1,
    ).max(axis=1)
    out["max_included_event_end"] = cutoff
    out["player_last_spatial_event_end"] = out["source_player_id"].map(player_last)
    out["snapshot_kind"] = "strict_prior_asof"
    out["profile_version"] = PROFILE_VERSION
    out["ability_weighting_status"] = "disabled_unestimated_beta"
    out["ability_log_weight_per_sg2"] = 0.0
    out["ability_sg"] = np.where(
        out["sg_category"].eq("ott"), out["sg_ott"], out["sg_app"]
    )
    out["ability_normalization_eligible"] = (
        out["dg_id"].notna() & out["skill_date"].notna() & out["ability_sg"].notna()
    )
    out["spatial_normalization_eligible"] = True
    out["normalization_eligible"] = out["spatial_normalization_eligible"]
    out["normalization_source"] = out["spatial_profile_source"]
    out["exposure_weight"] = 1.0
    bad = out["skill_date"].notna() & (out["skill_date"] >= out["event_start"])
    if bad.any() or not (out["profile_cutoff"] < out["event_start"]).all():
        raise ValueError("profile provenance violates strict event cutoff")
    # Ability is disabled.  Numeric zero is used for unmatched bridge rows so
    # the downstream normalizer cannot accidentally propagate NaN weights;
    # eligibility and status columns retain the fail-closed distinction.
    out["ability_sg"] = pd.to_numeric(out["ability_sg"], errors="coerce").fillna(0.0)
    return out


def build_asof_spatial_profiles(
    bridge: pd.DataFrame,
    tee_history: pd.DataFrame,
    app_history: pd.DataFrame,
    config: BuilderConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build expanding strict-prior snapshots for every bridge event."""

    config = config or BuilderConfig()
    bridge = bridge.copy()
    bridge["event_start"] = _utc(bridge["event_start"], "bridge event_start")
    bridge["skill_date"] = pd.to_datetime(bridge["skill_date"], errors="coerce", utc=True)
    events = bridge[["event_key", "event_name", "season", "event_start"]].drop_duplicates()
    tee_parts: list[pd.DataFrame] = []
    app_parts: list[pd.DataFrame] = []
    for event in events.sort_values(["event_start", "event_key"]).to_dict("records"):
        event_series = pd.Series(event)
        roster = bridge[bridge["event_key"].eq(event["event_key"])].copy()
        roster["source_player_id"] = roster["source_player_id"].astype(str)
        if roster.empty:
            raise ValueError(f"empty roster for {event['event_key']}")
        prior_tee = strict_prior_rows(tee_history, event["event_start"])
        prior_app = strict_prior_rows(app_history, event["event_start"])
        tee_profile = _tee_event_profiles(roster, prior_tee, event_series, config)
        app_profile = _app_event_profiles(roster, prior_app, event_series, config)
        tee_parts.append(
            _merge_snapshot_provenance(tee_profile, roster, event_series, prior_tee)
        )
        app_parts.append(
            _merge_snapshot_provenance(app_profile, roster, event_series, prior_app)
        )
        print(
            f"Built {event['event_key']}: tee prior={len(prior_tee):,}, "
            f"APP prior={len(prior_app):,}, roster={len(roster):,}"
        )
    tee_profiles = pd.concat(tee_parts, ignore_index=True)
    app_profiles = pd.concat(app_parts, ignore_index=True)
    modifier_hash_cols = [
        "event_key", "source_player_id", "sg_category", "state_key",
        "profile_cutoff", "distance_ratio", "long_bias_yd", "lat_bias_yd",
        "long_scale", "lat_scale", "long_lat_corr", "n_spatial", "n_state",
        "skill_date", "dg_id", "ability_sg", "spatial_profile_source",
    ]
    for frame in (tee_profiles, app_profiles):
        if not (frame["profile_cutoff"] < frame["event_start"]).all():
            raise AssertionError("strict-prior output invariant failed")
        frame["profile_sha256"] = ""
        for _, indices in frame.groupby(
            ["event_key", "sg_category"], observed=True, sort=False
        ).groups.items():
            group_hash = _frame_sha256(frame.loc[indices], modifier_hash_cols)
            frame.loc[indices, "profile_sha256"] = group_hash
    return tee_profiles, app_profiles


def materialize_normalization_profiles(
    spatial_profiles: pd.DataFrame,
    atom_conditions: pd.DataFrame,
) -> pd.DataFrame:
    """Expand player-state profiles to strict-prior setup/reference conditions.

    ``atom_conditions`` must be produced without held-out transition outcomes.
    Its reference cutoff is validated independently.  Reference centers are
    typically the center of the prior-course donor distribution; APP centers
    are naturally pin-relative and normally equal ``(0, 0)``.
    """

    required_conditions = {
        "event_key", "sg_category", "state_key", "setup_key",
        "reference_long_yd", "reference_lat_yd", "reference_cutoff",
    }
    missing = sorted(required_conditions - set(atom_conditions.columns))
    if missing:
        raise ValueError(f"atom_conditions is missing columns: {missing}")
    conditions = atom_conditions.copy()
    conditions["reference_cutoff"] = _utc(
        conditions["reference_cutoff"], "reference_cutoff"
    )
    if conditions.duplicated(
        ["event_key", "sg_category", "state_key", "setup_key"]
    ).any():
        raise ValueError("atom_conditions contains duplicate setup states")
    event_starts = spatial_profiles[["event_key", "event_start"]].drop_duplicates()
    conditions = conditions.merge(
        event_starts, on="event_key", how="left", validate="many_to_one"
    )
    if conditions["event_start"].isna().any():
        raise ValueError("atom_conditions contains an unknown event")
    if not (conditions["reference_cutoff"] < conditions["event_start"]).all():
        raise ValueError("setup reference cutoff must strictly predate event start")
    if not np.isfinite(
        conditions[["reference_long_yd", "reference_lat_yd"]].apply(
            pd.to_numeric, errors="coerce"
        )
    ).all().all():
        raise ValueError("setup reference centers must be finite")
    result = conditions.drop(columns="event_start").merge(
        spatial_profiles,
        on=["event_key", "sg_category", "state_key"],
        how="left",
        validate="many_to_many",
    )
    if result["source_player_id"].isna().any():
        raise ValueError("a setup state has no strict-prior player profiles")
    result["profile_cutoff"] = result[
        ["profile_cutoff", "reference_cutoff"]
    ].max(axis=1)
    if not (result["profile_cutoff"] < result["event_start"]).all():
        raise AssertionError("materialized profile cutoff invariant failed")
    result["materialization_sha256"] = _frame_sha256(
        result,
        [
            "event_key", "source_player_id", "sg_category", "state_key",
            "setup_key", "profile_cutoff", "reference_long_yd",
            "reference_lat_yd", "distance_ratio", "long_bias_yd",
            "lat_bias_yd", "long_scale", "lat_scale", "long_lat_corr",
        ],
    )
    return result


def write_artifacts(
    tee_profiles: pd.DataFrame,
    app_profiles: pd.DataFrame,
    output_dir: Path,
    *,
    bridge_metadata: dict[str, object],
    history_metadata: dict[str, object],
    config: BuilderConfig,
    build_metrics: dict[str, object] | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tee_path = output_dir / "tee_player_spatial_profiles_asof.parquet"
    app_path = output_dir / "app_player_spatial_profiles_asof.parquet"
    tee_profiles.to_parquet(tee_path, index=False)
    app_profiles.to_parquet(app_path, index=False)
    manifest: dict[str, object] = {
        "contract": PROFILE_VERSION,
        "snapshot_kind": "strict_prior_asof",
        "ability_weighting_status": "disabled_unestimated_beta",
        "events": int(
            pd.concat([tee_profiles[["event_key"]], app_profiles[["event_key"]]])[
                "event_key"
            ].nunique()
        ),
        "donor_editions_2020_2025": int(
            tee_profiles.loc[tee_profiles["season"].between(2020, 2025), "event_key"].nunique()
        ),
        "target_editions_2026": int(
            tee_profiles.loc[tee_profiles["season"].eq(2026), "event_key"].nunique()
        ),
        "tee_profile_rows": int(len(tee_profiles)),
        "app_profile_rows": int(len(app_profiles)),
        "tee_profiles_sha256": _frame_sha256(tee_profiles),
        "app_profiles_sha256": _frame_sha256(app_profiles),
        "tee_artifact_sha256": _file_sha256(tee_path),
        "app_artifact_sha256": _file_sha256(app_path),
        "strict_cutoff_violations": int(
            (~(tee_profiles["profile_cutoff"] < tee_profiles["event_start"])).sum()
            + (~(app_profiles["profile_cutoff"] < app_profiles["event_start"])).sum()
        ),
        "builder_config": asdict(config),
        "bridge": bridge_metadata,
        "history": history_metadata,
        "coverage": {
            "tee": _coverage_summary(tee_profiles),
            "app": _coverage_summary(app_profiles),
        },
        "build_metrics": build_metrics or {},
        "code_sha256": _file_sha256(Path(__file__)),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    if manifest["strict_cutoff_violations"]:
        raise RuntimeError("strict-prior artifact audit failed")
    return manifest


def _coverage_summary(frame: pd.DataFrame) -> dict[str, object]:
    """Emit compact availability/fallback audit statistics for an artifact."""

    total = len(frame)
    spatial = frame["spatial_profile_available"].fillna(False).astype(bool)
    eligible = frame["normalization_eligible"].fillna(False).astype(bool)
    ability_eligible = frame["ability_normalization_eligible"].fillna(False).astype(bool)
    source_counts = {
        str(key): int(value)
        for key, value in frame["normalization_source"].fillna("missing").value_counts().items()
    }
    by_event = []
    for event_key, group in frame.groupby("event_key", sort=True, observed=True):
        event_spatial = group["spatial_profile_available"].fillna(False).astype(bool)
        event_eligible = group["normalization_eligible"].fillna(False).astype(bool)
        event_ability_eligible = (
            group["ability_normalization_eligible"].fillna(False).astype(bool)
        )
        by_event.append(
            {
                "event_key": str(event_key),
                "event_name": str(group["event_name"].iloc[0]),
                "season": int(group["season"].iloc[0]),
                "rows": int(len(group)),
                "spatial_available_rows": int(event_spatial.sum()),
                "spatial_available_rate": float(event_spatial.mean()),
                "normalization_eligible_rows": int(event_eligible.sum()),
                "normalization_eligible_rate": float(event_eligible.mean()),
                "ability_normalization_eligible_rows": int(
                    event_ability_eligible.sum()
                ),
                "ability_normalization_eligible_rate": float(
                    event_ability_eligible.mean()
                ),
            }
        )
    return {
        "rows": int(total),
        "spatial_available_rows": int(spatial.sum()),
        "spatial_available_rate": float(spatial.mean()) if total else 0.0,
        "normalization_eligible_rows": int(eligible.sum()),
        "normalization_eligible_rate": float(eligible.mean()) if total else 0.0,
        "ability_normalization_eligible_rows": int(ability_eligible.sum()),
        "ability_normalization_eligible_rate": (
            float(ability_eligible.mean()) if total else 0.0
        ),
        "normalization_source_counts": source_counts,
        "by_event": by_event,
    }


class _PeakRSSMonitor:
    """Sample process RSS while the expensive read/fit stage is active."""

    def __init__(self, interval_seconds: float = 0.1) -> None:
        self.interval_seconds = interval_seconds
        self.peak_rss_bytes = int(psutil.Process(os.getpid()).memory_info().rss)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        process = psutil.Process(os.getpid())
        while not self._stop.wait(self.interval_seconds):
            self.peak_rss_bytes = max(
                self.peak_rss_bytes, int(process.memory_info().rss)
            )

    def __enter__(self) -> "_PeakRSSMonitor":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()
        self.peak_rss_bytes = max(
            self.peak_rss_bytes, int(psutil.Process(os.getpid()).memory_info().rss)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--history-db", type=Path, default=DEFAULT_HISTORY_DB)
    parser.add_argument("--bridge", type=Path, default=DEFAULT_BRIDGE)
    parser.add_argument("--bridge-manifest", type=Path, default=DEFAULT_BRIDGE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--event", action="append", dest="events")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = tuple(args.events) if args.events else EXPECTED_EVENTS
    config = BuilderConfig(start_year=args.start_year, end_year=args.end_year)
    build_start = time.perf_counter()
    with _PeakRSSMonitor() as memory_monitor:
        bridge, bridge_metadata = load_strict_bridge(
            args.bridge, args.bridge_manifest, expected_events=events
        )
        bridge = bridge[bridge["event_name"].isin(events)].copy()
        calendar = load_event_calendar(
            args.history_db, args.export_root / "events.parquet"
        )
        tee_history, app_history, history_metadata = load_spatial_history(
            args.export_root, calendar, config
        )
        history_metadata["calendar_source_files"] = calendar.attrs["source_files"]
        history_metadata["calendar_unresolved_event_keys"] = calendar.attrs[
            "unresolved_event_keys"
        ]
        tee_profiles, app_profiles = build_asof_spatial_profiles(
            bridge, tee_history, app_history, config
        )
    build_metrics = {
        "scope": "bridge_calendar_hash_history_load_and_profile_fit",
        "elapsed_seconds": float(time.perf_counter() - build_start),
        "peak_rss_bytes": int(memory_monitor.peak_rss_bytes),
        "rss_sample_interval_seconds": float(memory_monitor.interval_seconds),
    }
    manifest = write_artifacts(
        tee_profiles,
        app_profiles,
        args.output_dir,
        bridge_metadata=bridge_metadata,
        history_metadata=history_metadata,
        config=config,
        build_metrics=build_metrics,
    )
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()


__all__ = [
    "BuilderConfig",
    "build_asof_spatial_profiles",
    "load_event_calendar",
    "load_spatial_history",
    "load_strict_bridge",
    "materialize_normalization_profiles",
    "strict_prior_rows",
]
