"""Build the frozen, research-only player-round panel for the base-skill rebuild.

This script deliberately does not fit a model and does not write any production
input.  It assembles the outcome history and benchmark columns needed by the
walk-forward research process, while keeping backfilled IMG Arena signals and
post-round labels visibly separated from point-in-time inputs.

The most important leakage rule is not something a parquet file can enforce:
when constructing a pre-tournament feature row, use only observations whose
``round_date < event_start_date``.  Events sharing an event-start date must be
processed as one batch.  The manifest and audit repeat that rule so downstream
scripts can fail closed rather than silently train on the target week.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import sqlite3
import tempfile
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_DG_DB = Path(
    r"C:\Users\McKinley Slade\dev\sim_prep\work\dg_historical.db"
)
DEFAULT_CASCADE = Path(
    r"C:\Users\McKinley Slade\dev\sim_prep\work\sg_preds_resids.csv"
)
DEFAULT_CROSSWALK = Path(
    r"C:\Users\McKinley Slade\dev\sim_prep\shot_bridge\crosswalk.parquet"
)
DEFAULT_IMG_DB = Path(
    r"C:\Users\McKinley Slade\dev\golf_scraping\output\imgarena_shots_lean.sqlite3"
)
DEFAULT_WEATHER_EXCLUSIONS = Path(
    r"C:\Users\McKinley Slade\dev\sim_prep\weather_exclusion_rounds.csv"
)
DEFAULT_IMG_PLAYER_CROSSWALK = Path("permanent_data/img_player_crosswalk.csv")
DEFAULT_ODDS_CACHE = Path(
    "permanent_data/research/dg_matchup_cache_20200101_20260822"
)
DEFAULT_OUTPUT = Path(
    "permanent_data/research/base_skill_rebuild_20200101_20260822"
)

DG_CATEGORIES = ("ott", "app", "arg", "putt", "total")
IMG_CATEGORIES = ("ott", "app", "arg", "putt", "total")
WEATHER_FIELDS = ("wind", "dew", "temp", "rain")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dg-db", type=Path, default=DEFAULT_DG_DB)
    parser.add_argument("--cascade", type=Path, default=DEFAULT_CASCADE)
    parser.add_argument("--crosswalk", type=Path, default=DEFAULT_CROSSWALK)
    parser.add_argument("--img-db", type=Path, default=DEFAULT_IMG_DB)
    parser.add_argument(
        "--weather-exclusions", type=Path, default=DEFAULT_WEATHER_EXCLUSIONS
    )
    parser.add_argument(
        "--img-player-crosswalk", type=Path, default=DEFAULT_IMG_PLAYER_CROSSWALK
    )
    parser.add_argument("--odds-cache", type=Path, default=DEFAULT_ODDS_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--history-start",
        default="2015-04-01",
        help=(
            "Earliest outcome history retained. The current DG database begins "
            "later, but this must precede the market window so early tests have "
            "the same lookback opportunity as later tests."
        ),
    )
    parser.add_argument(
        "--market-start",
        default="2020-01-01",
        help="First event-start date eligible for market evaluation.",
    )
    parser.add_argument("--end", default="2026-08-22")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_name(value: Any) -> str:
    """Conservative accent/punctuation-insensitive join key, without reordering."""

    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value).casefold())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", text)


def canonical_bridge_player_name(value: Any) -> str:
    """Order-insensitive key used by sim_prep's validated IMG bridge."""

    if value is None or pd.isna(value):
        return ""
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii").lower()
    suffixes = {"jr", "sr", "ii", "iii", "iv"}
    tokens = [
        token for token in re.findall(r"[a-z0-9]+", ascii_value) if token not in suffixes
    ]
    return " ".join(sorted(tokens))


def sqlite_numeric(value: Any) -> Any:
    """Decode numpy integers that older SQLite writes sometimes return as blobs."""

    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        if len(raw) in (1, 2, 4, 8):
            return int.from_bytes(raw, byteorder="little", signed=True)
        return np.nan
    return value


def canonical_event_name(value: Any) -> str:
    """Normalize event names for identity scoring, not as a standalone key."""

    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value).casefold())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    # Sponsors and articles move often; retain words but standardize punctuation.
    tokens = re.findall(r"[a-z0-9]+", text)
    stop = {"the", "presented", "by"}
    return " ".join(token for token in tokens if token not in stop)


def event_name_similarity(left: Any, right: Any) -> float:
    a = canonical_event_name(left)
    b = canonical_event_name(right)
    if not a or not b:
        return 0.0
    sequence = difflib.SequenceMatcher(a=a, b=b).ratio()
    aset, bset = set(a.split()), set(b.split())
    jaccard = len(aset & bset) / len(aset | bset) if aset | bset else 0.0
    return float(max(sequence, jaccard))


def sqlite_ro(path: Path, *, immutable: bool = False) -> sqlite3.Connection:
    suffix = "?mode=ro"
    if immutable:
        suffix += "&immutable=1"
    connection = sqlite3.connect(f"file:{path}{suffix}", uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required research inputs are missing: {missing}")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_metadata(path: Path, *, hash_limit_bytes: int = 128 * 1024 * 1024) -> dict:
    stat = path.stat()
    result = {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }
    if stat.st_size <= hash_limit_bytes:
        result["sha256"] = sha256_file(path)
    else:
        result["sha256"] = None
        result["sha256_note"] = "Skipped for large database; table snapshots are audited."
    return result


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".parquet.tmp", dir=path.parent
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        frame.to_parquet(temp_path, index=False)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".csv.tmp", dir=path.parent
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        frame.to_csv(temp_path, index=False)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".json.tmp", dir=path.parent
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=json_default)
            handle.write("\n")
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Cannot JSON-serialize {type(value)!r}")


def load_player_rounds(
    connection: sqlite3.Connection, start: str, end: str
) -> tuple[pd.DataFrame, dict]:
    query = """
        SELECT
            rowid AS source_rowid,
            player_name, dg_id, event_id, event_name, lower(tour) AS tour,
            year, round_num, date(round_date) AS round_date, fin_text,
            course_name, course_num, course_par, score,
            sg_app, sg_arg, sg_ott, sg_putt, sg_t2g, sg_total,
            driving_acc, driving_dist, gir, scrambling,
            great_shots, poor_shots, prox_fw, prox_rgh, rel_dd,
            start_hole, teetime, age, pro_time, skill, cut, field_size,
            rd_num, sg_putt_adj, sg_ott_adj, sg_app_adj, sg_arg_adj,
            sg_total_adj, cumulative_score, finish_pos, start_pos, Lead,
            post_cut
        FROM player_rounds
        WHERE lower(tour) = 'pga'
          AND date(round_date) >= date(?)
          AND date(round_date) <= date(?)
    """
    frame = pd.read_sql_query(query, connection, params=(start, end))
    source_row_count = len(frame)
    frame["round_date"] = pd.to_datetime(frame["round_date"]).dt.normalize()
    for column in ("dg_id", "event_id", "year", "round_num", "course_num"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("int64")
    numeric_columns = [
        "course_par", "score", "sg_app", "sg_arg", "sg_ott", "sg_putt",
        "sg_t2g", "sg_total", "driving_acc", "driving_dist", "gir",
        "scrambling", "great_shots", "poor_shots", "prox_fw", "prox_rgh",
        "rel_dd", "start_hole", "age", "pro_time", "skill", "field_size",
        "rd_num", "sg_putt_adj", "sg_ott_adj", "sg_app_adj", "sg_arg_adj",
        "sg_total_adj", "cumulative_score", "finish_pos", "start_pos", "Lead",
        "post_cut",
    ]
    blob_conversions = {}
    for column in numeric_columns:
        if column not in frame:
            continue
        blob_count = int(frame[column].map(lambda value: isinstance(value, (bytes, bytearray, memoryview))).sum())
        if blob_count:
            blob_conversions[column] = blob_count
        frame[column] = pd.to_numeric(frame[column].map(sqlite_numeric), errors="coerce")

    natural_key = [
        "tour",
        "year",
        "event_id",
        "round_num",
        "round_date",
        "course_num",
        "dg_id",
    ]
    duplicate_mask = frame.duplicated(natural_key, keep=False)
    duplicate_rows = int(duplicate_mask.sum())
    if duplicate_rows:
        source_columns = [column for column in frame.columns if column != "source_rowid"]
        frame["_nonnull"] = frame[source_columns].notna().sum(axis=1)
        frame = (
            frame.sort_values(natural_key + ["_nonnull", "source_rowid"])
            .drop_duplicates(natural_key, keep="last")
            .drop(columns="_nonnull")
        )

    event_group = ["tour", "year", "event_id"]
    frame["event_start_date"] = frame.groupby(event_group)["round_date"].transform(
        "min"
    )
    frame["event_end_date"] = frame.groupby(event_group)["round_date"].transform(
        "max"
    )
    frame["event_name_normalized"] = frame["event_name"].map(canonical_event_name)
    frame["player_name_normalized"] = frame["player_name"].map(canonical_name)
    frame["event_key"] = (
        frame["tour"]
        + "|"
        + frame["year"].astype(str)
        + "|"
        + frame["event_id"].astype(str)
        + "|"
        + frame["event_start_date"].dt.strftime("%Y-%m-%d")
    )
    frame["player_event_key"] = frame["event_key"] + "|" + frame["dg_id"].astype(str)
    frame["player_round_key"] = (
        frame["player_event_key"]
        + "|"
        + frame["round_num"].astype(str)
        + "|"
        + frame["round_date"].dt.strftime("%Y-%m-%d")
        + "|"
        + frame["course_num"].astype(str)
    )
    if frame["player_round_key"].duplicated().any():
        raise AssertionError("Canonical player_round_key is not unique after deduplication")

    rename = {
        "sg_ott": "dg_sg_ott_raw",
        "sg_app": "dg_sg_app_raw",
        "sg_arg": "dg_sg_arg_raw",
        "sg_putt": "dg_sg_putt_raw",
        "sg_total": "dg_sg_total_raw",
        "sg_t2g": "dg_sg_t2g_raw",
        "sg_ott_adj": "dg_sg_ott_field_adjusted",
        "sg_app_adj": "dg_sg_app_field_adjusted",
        "sg_arg_adj": "dg_sg_arg_field_adjusted",
        "sg_putt_adj": "dg_sg_putt_field_adjusted",
        "sg_total_adj": "dg_sg_total_field_adjusted",
        "skill": "dg_source_skill_same_round",
        "Lead": "lead",
    }
    frame = frame.rename(columns=rename)
    for category in DG_CATEGORIES:
        raw = f"dg_sg_{category}_raw"
        adjusted = f"dg_sg_{category}_field_adjusted"
        if raw in frame and adjusted in frame:
            frame[f"dg_sg_{category}_field_adjustment"] = frame[adjusted] - frame[raw]

    audit = {
        "source_rows": int(source_row_count),
        "source_duplicate_rows_detected": duplicate_rows,
        "output_rows_after_dedup": int(len(frame)),
        "events": int(frame["event_key"].nunique()),
        "players": int(frame["dg_id"].nunique()),
        "sqlite_numeric_blob_conversions": blob_conversions,
        "min_round_date": frame["round_date"].min(),
        "max_round_date": frame["round_date"].max(),
    }
    return frame, audit


def two_pass_name_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    exact_keys: list[str],
    normalized_keys: list[str],
    right_value_columns: list[str],
    method_column: str,
) -> tuple[pd.DataFrame, dict]:
    """Exact-name join first; normalized fallback only when the key is unique."""

    right_exact = right[exact_keys + right_value_columns].copy()
    if right_exact.duplicated(exact_keys).any():
        raise AssertionError(f"Right side is not unique on exact keys: {exact_keys}")
    merged = left.merge(
        right_exact,
        on=exact_keys,
        how="left",
        validate="many_to_one",
        indicator="_exact_join",
    )
    merged[method_column] = np.where(
        merged["_exact_join"].eq("both"), "exact_name", "unmatched"
    )
    merged = merged.drop(columns="_exact_join")

    available_probe = right_value_columns[0]
    unmatched_mask = merged[method_column].eq("unmatched")
    fallback_right = right[normalized_keys + right_value_columns].copy()
    ambiguous_mask = fallback_right.duplicated(normalized_keys, keep=False)
    ambiguous_keys = int(fallback_right.loc[ambiguous_mask, normalized_keys].drop_duplicates().shape[0])
    fallback_right = fallback_right.loc[~ambiguous_mask]
    if unmatched_mask.any() and not fallback_right.empty:
        fallback_left = merged.loc[unmatched_mask, normalized_keys].copy()
        fallback_left["_target_index"] = fallback_left.index
        fallback = fallback_left.merge(
            fallback_right,
            on=normalized_keys,
            how="left",
            validate="many_to_one",
            suffixes=("", "_fallback"),
        ).set_index("_target_index")
        for column in right_value_columns:
            fallback_column = (
                f"{column}_fallback" if f"{column}_fallback" in fallback else column
            )
            merged.loc[fallback.index, column] = fallback[fallback_column]
        matched_fallback = fallback[available_probe].notna()
        merged.loc[fallback.index[matched_fallback], method_column] = "normalized_unique"

    stats = {
        "exact_rows": int(merged[method_column].eq("exact_name").sum()),
        "normalized_unique_rows": int(
            merged[method_column].eq("normalized_unique").sum()
        ),
        "unmatched_rows": int(merged[method_column].eq("unmatched").sum()),
        "ambiguous_normalized_source_keys_excluded": ambiguous_keys,
    }
    return merged, stats


def attach_weather(
    panel: pd.DataFrame,
    connection: sqlite3.Connection,
    start: str,
    end: str,
    exclusions_path: Path,
) -> tuple[pd.DataFrame, dict]:
    query = """
        SELECT
            player_name, event_id, date(round_date) AS round_date,
            course_name, course_num, score, teetime, wind, temp, rain, dew
        FROM player_weather_data_test
        WHERE date(round_date) >= date(?) AND date(round_date) <= date(?)
    """
    weather = pd.read_sql_query(query, connection, params=(start, end))
    weather["round_date"] = pd.to_datetime(weather["round_date"]).dt.normalize()
    weather["event_id"] = pd.to_numeric(weather["event_id"], errors="raise").astype(
        "int64"
    )
    weather["course_num"] = pd.to_numeric(
        weather["course_num"], errors="raise"
    ).astype("int64")
    weather["player_name_normalized"] = weather["player_name"].map(canonical_name)
    weather = weather.rename(
        columns={
            "player_name": "weather_source_player_name",
            "course_name": "weather_source_course_name",
            "score": "weather_source_score",
            "teetime": "weather_source_teetime",
            **{field: f"weather_{field}" for field in WEATHER_FIELDS},
        }
    )
    exact_keys = ["event_id", "round_date", "course_num", "player_name"]
    weather = weather.rename(columns={"weather_source_player_name": "player_name"})
    value_columns = [
        "weather_source_course_name",
        "weather_source_score",
        "weather_source_teetime",
        *[f"weather_{field}" for field in WEATHER_FIELDS],
    ]
    normalized_keys = [
        "event_id",
        "round_date",
        "course_num",
        "player_name_normalized",
    ]
    panel, join_stats = two_pass_name_join(
        panel,
        weather,
        exact_keys=exact_keys,
        normalized_keys=normalized_keys,
        right_value_columns=value_columns,
        method_column="weather_join_method",
    )

    group_columns = [
        "year",
        "event_id",
        "round_num",
        "round_date",
        "course_num",
    ]
    for field in WEATHER_FIELDS:
        source = f"weather_{field}"
        mean = f"weather_field_mean_{field}"
        delta = f"weather_delta_{field}"
        panel[mean] = panel.groupby(group_columns, dropna=False)[source].transform("mean")
        panel[delta] = panel[source] - panel[mean]
    panel["weather_field_n"] = panel.groupby(group_columns, dropna=False)[
        "weather_wind"
    ].transform("count")

    panel["weather_exclusion_tier"] = pd.NA
    panel["weather_exclusion_flags"] = pd.NA
    panel["weather_excluded"] = False
    panel["weather_review"] = False
    exclusion_rows = 0
    if exclusions_path.exists():
        exclusions = pd.read_csv(exclusions_path)
        exclusions["round_date"] = pd.to_datetime(exclusions["round_date"]).dt.normalize()
        exclusion_keys = ["year", "event_id", "round_num", "round_date"]
        exclusions = exclusions.rename(
            columns={
                "exclusion_tier": "weather_exclusion_tier_source",
                "flags": "weather_exclusion_flags_source",
            }
        )
        exclusions = exclusions.drop_duplicates(exclusion_keys, keep="last")
        panel = panel.merge(
            exclusions[
                exclusion_keys
                + ["weather_exclusion_tier_source", "weather_exclusion_flags_source"]
            ],
            on=exclusion_keys,
            how="left",
            validate="many_to_one",
        )
        panel["weather_exclusion_tier"] = panel[
            "weather_exclusion_tier_source"
        ]
        panel["weather_exclusion_flags"] = panel[
            "weather_exclusion_flags_source"
        ]
        panel = panel.drop(
            columns=["weather_exclusion_tier_source", "weather_exclusion_flags_source"]
        )
        panel["weather_excluded"] = panel["weather_exclusion_tier"].eq("exclude")
        panel["weather_review"] = panel["weather_exclusion_tier"].eq("review")
        exclusion_rows = int(panel["weather_excluded"].sum())

    centering_error = {}
    for field in WEATHER_FIELDS:
        delta = f"weather_delta_{field}"
        grouped = panel.loc[panel[delta].notna()].groupby(group_columns)[delta].mean()
        maximum = float(grouped.abs().max()) if len(grouped) else np.nan
        centering_error[field] = maximum
        if len(grouped) and maximum > 1e-10:
            raise AssertionError(
                f"Weather {field} is not field centered; max group mean={maximum}"
            )

    audit = {
        "source_rows": int(len(weather)),
        "join": join_stats,
        "coverage_rows": int(panel["weather_wind"].notna().sum()),
        "coverage_rate": float(panel["weather_wind"].notna().mean()),
        "field_centering_max_abs_group_mean": centering_error,
        "excluded_player_round_rows": exclusion_rows,
        "review_player_round_rows": int(panel["weather_review"].sum()),
    }
    return panel, audit


def attach_legacy_asof(
    panel: pd.DataFrame, connection: sqlite3.Connection, start: str, end: str
) -> tuple[pd.DataFrame, dict]:
    player_ids = sorted(int(value) for value in panel["dg_id"].dropna().unique())
    player_id_sql = ",".join(map(str, player_ids))
    # Pull an ample pre-window so early-2020 entrants can still find a prior value.
    query = f"""
        SELECT
            date(Date) AS legacy_asof_date, dg_id,
            sg_ott AS legacy_sg_ott_asof,
            sg_app AS legacy_sg_app_asof,
            ag_putt AS legacy_sg_putt_source_asof,
            ag_arg AS legacy_sg_arg_source_asof,
            rel_dd AS legacy_rel_dd_asof,
            sg_total_adj AS legacy_sg_total_adj_asof,
            composite_sg AS legacy_composite_sg_asof,
            seq_indx AS legacy_seq_indx_asof,
            std_dev AS legacy_std_dev_asof,
            time_weighted_sg AS legacy_time_weighted_sg_asof,
            timewt_allocation AS legacy_timewt_allocation_asof,
            skill_est AS legacy_skill_est_asof
        FROM model_results2
        WHERE dg_id IN ({player_id_sql})
          AND date(Date) >= date(?, '-730 days')
          AND date(Date) <= date(?)
    """
    legacy = pd.read_sql_query(query, connection, params=(start, end))
    legacy["legacy_asof_date"] = pd.to_datetime(
        legacy["legacy_asof_date"]
    ).dt.normalize()
    legacy["dg_id"] = pd.to_numeric(legacy["dg_id"], errors="raise").astype("int64")
    duplicate_rows = int(legacy.duplicated(["legacy_asof_date", "dg_id"], False).sum())
    if duplicate_rows:
        raise AssertionError("model_results2 is not unique by (Date, dg_id)")

    left = panel.reset_index(drop=True).copy()
    left["_panel_order"] = np.arange(len(left), dtype=np.int64)
    left = left.sort_values(["round_date", "dg_id", "_panel_order"])
    right = legacy.sort_values(["legacy_asof_date", "dg_id"])
    merged = pd.merge_asof(
        left,
        right,
        left_on="round_date",
        right_on="legacy_asof_date",
        by="dg_id",
        direction="backward",
        allow_exact_matches=False,
    )
    merged = merged.sort_values("_panel_order").drop(columns="_panel_order").reset_index(
        drop=True
    )
    merged["legacy_asof_lag_days"] = (
        merged["round_date"] - merged["legacy_asof_date"]
    ).dt.days
    invalid = merged["legacy_asof_date"].notna() & ~(
        merged["legacy_asof_date"] < merged["round_date"]
    )
    if invalid.any():
        raise AssertionError("Legacy as-of join leaked same-day or future rows")
    audit = {
        "source_rows_loaded": int(len(legacy)),
        "source_duplicate_rows": duplicate_rows,
        "coverage_rows": int(merged["legacy_skill_est_asof"].notna().sum()),
        "coverage_rate": float(merged["legacy_skill_est_asof"].notna().mean()),
        "strictly_prior_date_violations": int(invalid.sum()),
        "median_lag_days": float(merged["legacy_asof_lag_days"].median()),
        "max_lag_days": float(merged["legacy_asof_lag_days"].max()),
    }
    return merged, audit


def load_img_accounting(connection: sqlite3.Connection) -> tuple[pd.DataFrame, dict]:
    query = """
        WITH current_models AS (
            SELECT event_key, round_no, model_id, model_version
            FROM (
                SELECT
                    p.event_key, p.round_no, p.model_id, m.model_version,
                    ROW_NUMBER() OVER (
                        PARTITION BY p.event_key, p.round_no
                        ORDER BY m.created_at DESC, p.model_id DESC
                    ) AS model_rank
                FROM (
                    SELECT DISTINCT event_key, round_no, model_id
                    FROM archive_player_round_strokes_gained
                ) p
                JOIN sg_models m ON m.model_id = p.model_id
                WHERE m.model_version LIKE '%v4_lies2026%'
            )
            WHERE model_rank = 1
        )
        SELECT
            s.event_key, s.round_no, s.source_player_id, s.source_team_id,
            s.player_name AS img_accounting_player_name,
            s.model_id AS img_accounting_model_id,
            c.model_version AS img_accounting_model_version,
            s.round_strokes AS img_accounting_round_strokes,
            s.total_strokes AS img_accounting_total_strokes,
            s.valid_sg_strokes AS img_accounting_valid_sg_strokes,
            s.complete AS img_accounting_complete,
            s.sg_ott_raw AS img_accounting_sg_ott_raw,
            s.sg_app_raw AS img_accounting_sg_app_raw,
            s.sg_arg_raw AS img_accounting_sg_arg_raw,
            s.sg_putt_raw AS img_accounting_sg_putt_raw,
            s.sg_total_raw AS img_accounting_sg_total_raw,
            s.sg_ott AS img_accounting_sg_ott_field_adjusted,
            s.sg_app AS img_accounting_sg_app_field_adjusted,
            s.sg_arg AS img_accounting_sg_arg_field_adjusted,
            s.sg_putt AS img_accounting_sg_putt_field_adjusted,
            s.sg_total AS img_accounting_sg_total_field_adjusted,
            s.sg_bs AS img_accounting_sg_bs_field_adjusted,
            s.sg_t2g AS img_accounting_sg_t2g_field_adjusted,
            s.adjustment_method AS img_accounting_adjustment_method,
            s.calculated_at AS img_accounting_calculated_at,
            1 AS img_accounting_generation_count
        FROM archive_player_round_strokes_gained s
        JOIN current_models c
          ON c.event_key = s.event_key
         AND c.round_no = s.round_no
         AND c.model_id = s.model_id
        JOIN archive_events e ON e.event_key = s.event_key
        WHERE e.tour = 'pga' AND s.complete = 1
    """
    frame = pd.read_sql_query(query, connection)
    key = ["event_key", "round_no", "source_player_id", "source_team_id"]
    if frame.duplicated(key).any():
        raise AssertionError("IMG accounting canonical generation is not unique")
    audit = {
        "canonical_rows": int(len(frame)),
        "events": int(frame["event_key"].nunique()),
        "source_keys_with_multiple_generations": int(
            frame["img_accounting_generation_count"].gt(1).sum()
        ),
        "incomplete_rows": int(frame["img_accounting_complete"].ne(1).sum()),
        "model_versions": frame["img_accounting_model_version"]
        .value_counts(dropna=False)
        .to_dict(),
    }
    return frame, audit


def load_img_v2(connection: sqlite3.Connection) -> tuple[pd.DataFrame, dict]:
    query = """
        WITH ranked AS (
            SELECT
                s.*,
                sm.model_version,
                ROW_NUMBER() OVER (
                    PARTITION BY s.event_key, s.course_id, s.round_no,
                                 s.source_player_id, s.source_team_id
                    ORDER BY s.complete DESC, s.calculated_at DESC,
                             s.signal_model_id DESC
                ) AS generation_rank,
                COUNT(*) OVER (
                    PARTITION BY s.event_key, s.course_id, s.round_no,
                                 s.source_player_id, s.source_team_id
                ) AS generation_count
            FROM archive_player_round_skill_signal s
            JOIN sg_models sm ON sm.model_id = s.signal_model_id
            WHERE s.event_key LIKE 'pga:%'
              AND sm.model_version = 'predictive_skill_pga_validated_v2'
        )
        SELECT
            event_key, course_id AS img_v2_course_id, round_no,
            source_player_id, source_team_id,
            player_name AS img_v2_player_name,
            raw_model_id AS img_v2_raw_model_id,
            signal_model_id AS img_v2_signal_model_id,
            model_version AS img_v2_model_version,
            complete AS img_v2_complete,
            sg_ott_reference AS img_v2_sg_ott_reference,
            sg_app_reference AS img_v2_sg_app_reference,
            sg_arg_reference AS img_v2_sg_arg_reference,
            sg_putt_reference AS img_v2_sg_putt_reference,
            sg_total_reference AS img_v2_sg_total_reference,
            sg_ott_signal AS img_v2_sg_ott_signal,
            sg_app_signal AS img_v2_sg_app_signal,
            sg_arg_signal AS img_v2_sg_arg_signal,
            sg_putt_signal AS img_v2_sg_putt_signal,
            sg_bs_signal AS img_v2_sg_bs_signal,
            sg_t2g_signal AS img_v2_sg_t2g_signal,
            sg_total_signal AS img_v2_sg_total_signal,
            centering_json AS img_v2_centering_json,
            adjustment_method AS img_v2_adjustment_method,
            calculated_at AS img_v2_calculated_at,
            generation_count AS img_v2_generation_count
        FROM ranked
        WHERE generation_rank = 1
    """
    frame = pd.read_sql_query(query, connection)
    source_key = ["event_key", "round_no", "source_player_id", "source_team_id"]
    multiple_courses = frame.duplicated(source_key, keep=False)
    if multiple_courses.any():
        raise AssertionError(
            "Current PGA v2 has multiple course rows for one source player-round"
        )
    audit = {
        "canonical_rows": int(len(frame)),
        "events": int(frame["event_key"].nunique()),
        "source_keys_with_multiple_generations": int(
            frame["img_v2_generation_count"].gt(1).sum()
        ),
        "incomplete_rows": int(frame["img_v2_complete"].ne(1).sum()),
        "model_versions": frame["img_v2_model_version"]
        .value_counts(dropna=False)
        .to_dict(),
    }
    return frame, audit


def load_archive_current_player_round_keys(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    """Load current v4 accounting keys exactly as the production bridge resolves them."""

    query = """
        WITH current_models AS (
            SELECT event_key, round_no, model_id
            FROM (
                SELECT
                    p.event_key, p.round_no, p.model_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY p.event_key, p.round_no
                        ORDER BY m.created_at DESC, p.model_id DESC
                    ) AS model_rank
                FROM (
                    SELECT DISTINCT event_key, round_no, model_id
                    FROM archive_player_round_strokes_gained
                ) p
                JOIN sg_models m ON m.model_id = p.model_id
                WHERE m.model_version LIKE '%v4_lies2026%'
            )
            WHERE model_rank = 1
        )
        SELECT DISTINCT
            p.event_key, p.round_no, p.source_player_id, p.source_team_id,
            p.player_name, e.season AS year, e.event_name,
            p.complete, p.calculated_at
        FROM archive_player_round_strokes_gained p
        JOIN current_models c
          ON c.event_key = p.event_key
         AND c.round_no = p.round_no
         AND c.model_id = p.model_id
        JOIN archive_events e ON e.event_key = p.event_key
        WHERE e.tour = 'pga' AND p.complete = 1
    """
    frame = pd.read_sql_query(query, connection)
    for column in ("source_player_id", "source_team_id"):
        frame[column] = frame[column].fillna("").astype(str)
    frame["round_no"] = pd.to_numeric(frame["round_no"], errors="raise").astype(
        "int64"
    )
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype("int64")

    def parse_event_id(value: Any) -> float:
        match = re.search(r"R\d{4}(\d+)$", str(value))
        return float(match.group(1)) if match else np.nan

    frame["event_id"] = frame["event_key"].map(parse_event_id)
    frame["player_key"] = frame["player_name"].map(canonical_bridge_player_name)
    source_key = ["event_key", "source_player_id", "source_team_id", "round_no"]
    if frame.duplicated(source_key).any():
        raise AssertionError("Current IMG v4 archive keys are not unique")
    return frame


def load_img_player_override_map(path: Path) -> tuple[dict[str, str], dict]:
    if not path.exists():
        return {}, {"available": False, "rows": 0, "ambiguous_source_ids": 0}
    frame = pd.read_csv(path)
    required = {"source_player_id", "datagolf_player_name"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"IMG player override is missing columns: {sorted(missing)}")
    frame["source_player_id"] = frame["source_player_id"].fillna("").astype(str)
    frame["override_player_key"] = frame["datagolf_player_name"].map(
        canonical_bridge_player_name
    )
    grouped = frame.loc[frame["override_player_key"].ne("")].groupby(
        "source_player_id"
    )["override_player_key"]
    unique_counts = grouped.nunique()
    ambiguous = set(unique_counts[unique_counts.gt(1)].index)
    mapping = {
        str(source_id): str(values.iloc[-1])
        for source_id, values in grouped
        if source_id not in ambiguous
    }
    audit = {
        "available": True,
        "rows": int(len(frame)),
        "usable_source_ids": int(len(mapping)),
        "ambiguous_source_ids": int(len(ambiguous)),
    }
    return mapping, audit


def build_research_img_crosswalk(
    panel: pd.DataFrame,
    original_path: Path,
    img_connection: sqlite3.Connection,
    override_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Extend the stale bridge in-memory and persist only a research artifact.

    The tracked sim_prep crosswalk is never changed. New matches follow the
    same event/year/round plus canonical-player logic as shot_features_bridge,
    with the permanent IMG player override taking precedence.
    """

    original = pd.read_parquet(original_path)
    original["round_date"] = pd.to_datetime(original["round_date"]).dt.normalize()
    for column in ("source_player_id", "source_team_id"):
        original[column] = original[column].fillna("").astype(str)
    original["round_no"] = pd.to_numeric(
        original["round_no"], errors="raise"
    ).astype("int64")
    for column in ("dg_id", "event_id", "year"):
        original[column] = pd.to_numeric(original[column], errors="raise").astype(
            "int64"
        )
    original["round_num"] = pd.to_numeric(
        original["round_num"], errors="coerce"
    ).astype("Int64")
    original["crosswalk_origin"] = "original_sim_prep"
    original["crosswalk_extended_at_utc"] = pd.NA

    archive = load_archive_current_player_round_keys(img_connection)
    override_map, override_audit = load_img_player_override_map(override_path)
    override = archive["source_player_id"].map(override_map)
    archive["player_key"] = override.fillna(archive["player_key"])
    archive["match_method"] = np.where(
        override.notna(), "crosswalk_csv", "canonical_name"
    )

    dg_rounds = panel[
        [
            "player_name",
            "dg_id",
            "event_id",
            "year",
            "round_num",
            "round_date",
        ]
    ].copy()
    dg_rounds["player_key"] = dg_rounds["player_name"].map(
        canonical_bridge_player_name
    )
    dg_match_key = ["year", "event_id", "round_num", "player_key"]
    ambiguous_dg_mask = dg_rounds.duplicated(dg_match_key, keep=False)
    # Repeated rows with the same dg_id are harmless; collisions across dg_id are not.
    ambiguous_keys = set()
    for key, group in dg_rounds.loc[ambiguous_dg_mask].groupby(dg_match_key):
        if group["dg_id"].nunique() > 1:
            ambiguous_keys.add(tuple(key))
    dg_rounds = dg_rounds.sort_values("round_date").drop_duplicates(
        dg_match_key, keep="last"
    )
    dg_rounds["_ambiguous_dg_key"] = [
        tuple(row) in ambiguous_keys
        for row in dg_rounds[dg_match_key].itertuples(index=False, name=None)
    ]

    source_key = ["event_key", "source_player_id", "source_team_id", "round_no"]
    original_source_keys = set(
        original[source_key].itertuples(index=False, name=None)
    )
    archive["already_in_original"] = [
        tuple(row) in original_source_keys
        for row in archive[source_key].itertuples(index=False, name=None)
    ]
    candidates = archive.loc[~archive["already_in_original"]].copy()
    candidates["round_num"] = candidates["round_no"]
    candidates = candidates.merge(
        dg_rounds.rename(columns={"player_name": "dg_player_name"}),
        on=dg_match_key,
        how="left",
        validate="many_to_one",
        suffixes=("", "_dg"),
    )
    event_pairs = set(
        panel[["year", "event_id"]].drop_duplicates().itertuples(index=False, name=None)
    )
    candidates["dg_event_available"] = [
        (int(year), int(event_id)) in event_pairs if pd.notna(event_id) else False
        for year, event_id in zip(candidates["year"], candidates["event_id"])
    ]
    candidates["unmatched_reason"] = np.select(
        [
            candidates["event_id"].isna(),
            ~candidates["dg_event_available"],
            candidates["_ambiguous_dg_key"].eq(True),
            candidates["dg_id"].isna(),
        ],
        [
            "unparseable_archive_event_id",
            "event_missing_in_dg_history",
            "ambiguous_dg_player_key",
            "player_not_matched",
        ],
        default="matched",
    )
    matched = candidates.loc[candidates["unmatched_reason"].eq("matched")].copy()
    matched["dg_id"] = pd.to_numeric(matched["dg_id"], errors="raise").astype(
        "int64"
    )
    matched["event_id"] = pd.to_numeric(
        matched["event_id"], errors="raise"
    ).astype("int64")
    matched["round_num"] = pd.to_numeric(
        matched["round_num"], errors="raise"
    ).astype("int64")
    matched["crosswalk_origin"] = "research_extension"
    matched["crosswalk_extended_at_utc"] = utc_now()
    extension_columns = [
        "event_key",
        "source_player_id",
        "source_team_id",
        "round_no",
        "player_name",
        "match_method",
        "dg_player_name",
        "dg_id",
        "event_id",
        "year",
        "round_num",
        "round_date",
        "crosswalk_origin",
        "crosswalk_extended_at_utc",
    ]
    extension = matched[extension_columns].copy()
    combined = pd.concat(
        [original[extension_columns], extension], ignore_index=True, sort=False
    )
    if combined.duplicated(source_key).any():
        raise AssertionError("Research IMG crosswalk is not unique by archive source key")
    dg_key = ["dg_id", "event_id", "year", "round_no", "round_date"]
    if combined.duplicated(dg_key).any():
        raise AssertionError("Research IMG crosswalk is not unique by DG player-round key")

    unmatched_columns = [
        "event_key",
        "event_name",
        "year",
        "event_id",
        "round_no",
        "source_player_id",
        "source_team_id",
        "player_name",
        "player_key",
        "match_method",
        "dg_event_available",
        "unmatched_reason",
    ]
    unmatched = candidates.loc[
        ~candidates["unmatched_reason"].eq("matched"), unmatched_columns
    ].copy()

    original_counts = original.groupby("event_key").size()
    extension_counts = extension.groupby("event_key").size()
    unmatched_counts = unmatched.groupby("event_key").size()
    event_rows = []
    for event_key, group in archive.groupby("event_key", sort=True):
        event_unmatched = unmatched.loc[unmatched["event_key"].eq(event_key)]
        event_extension = extension.loc[extension["event_key"].eq(event_key)]
        archive_count = len(group)
        original_count = int(original_counts.get(event_key, 0))
        extension_count = int(extension_counts.get(event_key, 0))
        unmatched_count = int(unmatched_counts.get(event_key, 0))
        combined_count = original_count + extension_count
        event_rows.append(
            {
                "event_key": event_key,
                "year": int(group["year"].iloc[0]),
                "event_id": (
                    int(group["event_id"].iloc[0])
                    if pd.notna(group["event_id"].iloc[0])
                    else np.nan
                ),
                "archive_event_name": group["event_name"].iloc[0],
                "archive_player_rounds": int(archive_count),
                "original_crosswalk_rows": original_count,
                "research_extension_rows": extension_count,
                "unmatched_rows": unmatched_count,
                "combined_crosswalk_rows": combined_count,
                "combined_match_rate": combined_count / archive_count,
                "has_dg_event": bool(
                    (int(group["year"].iloc[0]), int(group["event_id"].iloc[0]))
                    in event_pairs
                )
                if pd.notna(group["event_id"].iloc[0])
                else False,
                "extension_max_round_date": (
                    event_extension["round_date"].max()
                    if len(event_extension)
                    else pd.NaT
                ),
                "unmatched_reasons": "|".join(
                    sorted(str(value) for value in event_unmatched["unmatched_reason"].unique())
                ),
            }
        )
    event_audit = pd.DataFrame(event_rows)
    audit = {
        "original_rows": int(len(original)),
        "original_max_round_date": original["round_date"].max(),
        "archive_current_rows": int(len(archive)),
        "research_extension_rows": int(len(extension)),
        "research_extension_max_round_date": extension["round_date"].max(),
        "combined_rows": int(len(combined)),
        "combined_max_round_date": combined["round_date"].max(),
        "unmatched_new_archive_rows": int(len(unmatched)),
        "unmatched_reason_counts": unmatched["unmatched_reason"]
        .value_counts(dropna=False)
        .to_dict(),
        "override_map": override_audit,
        "ambiguous_dg_player_keys": int(len(ambiguous_keys)),
        "tracked_crosswalk_mutated": False,
    }
    return combined, event_audit, unmatched, audit


def attach_img(
    panel: pd.DataFrame,
    crosswalk_source: Path | pd.DataFrame,
    img_connection: sqlite3.Connection,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, dict]:
    crosswalk = (
        pd.read_parquet(crosswalk_source)
        if isinstance(crosswalk_source, Path)
        else crosswalk_source.copy()
    )
    crosswalk["round_date"] = pd.to_datetime(crosswalk["round_date"]).dt.normalize()
    crosswalk = crosswalk.loc[
        crosswalk["round_date"].between(pd.Timestamp(start), pd.Timestamp(end))
        & crosswalk["event_key"].astype(str).str.startswith("pga:")
    ].copy()
    crosswalk["source_player_id"] = crosswalk["source_player_id"].fillna("").astype(str)
    crosswalk["source_team_id"] = crosswalk["source_team_id"].fillna("").astype(str)
    crosswalk["round_no"] = pd.to_numeric(
        crosswalk["round_no"], errors="raise"
    ).astype("int64")
    # The bridge carries a redundant float round_num copied from DataGolf. Keep
    # the authoritative integer source round_no and rename it only once below.
    crosswalk = crosswalk.drop(columns=["round_num"], errors="ignore")
    for column in ("dg_id", "event_id", "year"):
        crosswalk[column] = pd.to_numeric(crosswalk[column], errors="raise").astype(
            "int64"
        )
    source_key = ["event_key", "round_no", "source_player_id", "source_team_id"]
    dg_key = ["dg_id", "event_id", "year", "round_no", "round_date"]
    if crosswalk.duplicated(source_key).any():
        raise AssertionError("IMG crosswalk source key is not unique")
    if crosswalk.duplicated(dg_key).any():
        raise AssertionError("IMG crosswalk DataGolf player-round key is not unique")

    accounting, accounting_audit = load_img_accounting(img_connection)
    v2, v2_audit = load_img_v2(img_connection)
    for frame in (accounting, v2):
        frame["source_player_id"] = frame["source_player_id"].fillna("").astype(str)
        frame["source_team_id"] = frame["source_team_id"].fillna("").astype(str)
        frame["round_no"] = pd.to_numeric(frame["round_no"], errors="raise").astype(
            "int64"
        )

    bridge = crosswalk.merge(
        accounting, on=source_key, how="left", validate="one_to_one"
    )
    bridge = bridge.merge(v2, on=source_key, how="left", validate="one_to_one")
    bridge = bridge.rename(
        columns={
            "event_key": "img_event_key",
            "round_no": "round_num",
            "player_name": "img_crosswalk_player_name",
            "dg_player_name": "img_crosswalk_dg_player_name",
            "match_method": "img_crosswalk_match_method",
        }
    )
    bridge_dg_key = ["dg_id", "event_id", "year", "round_num", "round_date"]
    if bridge.duplicated(bridge_dg_key).any():
        raise AssertionError("IMG bridge is not unique on DataGolf player-round key")
    keep_columns = bridge_dg_key + [
        column for column in bridge.columns if column not in bridge_dg_key
    ]
    panel = panel.merge(
        bridge[keep_columns],
        on=bridge_dg_key,
        how="left",
        validate="one_to_one",
    )
    panel["img_crosswalk_available"] = panel["img_event_key"].notna()
    panel["img_accounting_available"] = panel[
        "img_accounting_model_id"
    ].notna()
    panel["img_v2_available"] = panel["img_v2_signal_model_id"].notna()
    # These were rebuilt/backfilled in 2026; they are useful research outcomes,
    # but neither is evidence of what the live model had stored at bet time.
    panel["img_accounting_backfilled_research_only"] = panel[
        "img_accounting_available"
    ]
    panel["img_v2_retrospective_only"] = panel["img_v2_available"]
    panel["img_v2_future_validated_transform"] = panel["img_v2_available"]

    panel["img_accounting_calculated_at"] = pd.to_datetime(
        panel["img_accounting_calculated_at"], errors="coerce", utc=True
    )
    panel["img_v2_calculated_at"] = pd.to_datetime(
        panel["img_v2_calculated_at"], errors="coerce", utc=True
    )
    round_date_utc = panel["round_date"].dt.tz_localize("UTC")
    panel["img_accounting_calculated_after_round"] = panel[
        "img_accounting_calculated_at"
    ].notna() & (panel["img_accounting_calculated_at"] >= round_date_utc)
    panel["img_v2_calculated_after_round"] = panel["img_v2_calculated_at"].notna() & (
        panel["img_v2_calculated_at"] >= round_date_utc
    )

    audit = {
        "crosswalk_rows": int(len(crosswalk)),
        "crosswalk_match_methods": crosswalk["match_method"]
        .value_counts(dropna=False)
        .to_dict(),
        "accounting_source": accounting_audit,
        "v2_source": v2_audit,
        "panel_crosswalk_coverage_rate": float(panel["img_crosswalk_available"].mean()),
        "panel_accounting_coverage_rate": float(
            panel["img_accounting_available"].mean()
        ),
        "panel_v2_coverage_rate": float(panel["img_v2_available"].mean()),
        "accounting_calculated_after_round_rate_when_available": float(
            panel.loc[
                panel["img_accounting_available"],
                "img_accounting_calculated_after_round",
            ].mean()
        ),
        "v2_calculated_after_round_rate_when_available": float(
            panel.loc[panel["img_v2_available"], "img_v2_calculated_after_round"].mean()
        ),
    }
    return panel, audit


def attach_cascade(
    panel: pd.DataFrame, cascade_path: Path, start: str, end: str
) -> tuple[pd.DataFrame, dict]:
    cascade = pd.read_csv(cascade_path)
    cascade["round_date"] = pd.to_datetime(cascade["round_date"]).dt.normalize()
    cascade = cascade.loc[
        cascade["round_date"].between(pd.Timestamp(start), pd.Timestamp(end))
    ].copy()
    cascade["player_name_normalized"] = cascade["player_name"].map(canonical_name)
    cascade = cascade.rename(
        columns={
            "sample": "cascade_sample",
            "start_pos": "cascade_start_pos",
            "actual": "cascade_actual",
            "prediction": "cascade_prediction",
            "std_dev_50": "cascade_std_dev_50",
            "residual": "cascade_residual",
            "majors_history": "cascade_majors_history",
            "links_adj": "cascade_links_adj",
        }
    )
    exact_keys = [
        "event_id",
        "year",
        "round_num",
        "round_date",
        "player_name",
    ]
    normalized_keys = [
        "event_id",
        "year",
        "round_num",
        "round_date",
        "player_name_normalized",
    ]
    values = [
        "cascade_sample",
        "cascade_start_pos",
        "cascade_actual",
        "cascade_prediction",
        "cascade_std_dev_50",
        "cascade_residual",
        "cascade_majors_history",
        "cascade_links_adj",
    ]
    panel, join_stats = two_pass_name_join(
        panel,
        cascade,
        exact_keys=exact_keys,
        normalized_keys=normalized_keys,
        right_value_columns=values,
        method_column="cascade_join_method",
    )
    actual_overlap = panel["cascade_actual"].notna() & panel[
        "dg_sg_total_field_adjusted"
    ].notna()
    differences = (
        panel.loc[actual_overlap, "cascade_actual"]
        - panel.loc[actual_overlap, "dg_sg_total_field_adjusted"]
    )
    audit = {
        "source_rows": int(len(cascade)),
        "source_events": int(
            cascade[["year", "event_id"]].drop_duplicates().shape[0]
        ),
        "join": join_stats,
        "prediction_coverage_rate": float(panel["cascade_prediction"].notna().mean()),
        "actual_vs_dg_adjusted_overlap_rows": int(actual_overlap.sum()),
        "actual_vs_dg_adjusted_mean_abs_diff": float(differences.abs().mean()),
        "actual_vs_dg_adjusted_max_abs_diff": float(differences.abs().max()),
    }
    return panel, audit


def read_matchup_event_inventory(odds_cache: Path) -> pd.DataFrame:
    records: list[dict] = []
    matchup_dir = odds_cache / "matchups"
    if not matchup_dir.exists():
        return pd.DataFrame()
    for path in sorted(matchup_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                records.append(
                    {
                        "odds_calendar_year": row.get(
                            "query_year", row.get("season")
                        ),
                        "odds_event_id": str(row.get("event_id", "")),
                        "odds_event_name": row.get("event_name"),
                        "odds_event_completed_date": row.get("event_completed"),
                        "book": row.get("book"),
                        "bet_type": row.get("bet_type"),
                        "p1_dg_id": row.get("p1_dg_id"),
                        "p2_dg_id": row.get("p2_dg_id"),
                    }
                )
    if not records:
        return pd.DataFrame()
    raw = pd.DataFrame.from_records(records)
    raw["odds_calendar_year"] = pd.to_numeric(
        raw["odds_calendar_year"], errors="coerce"
    ).astype("Int64")
    raw["odds_event_completed_date"] = pd.to_datetime(
        raw["odds_event_completed_date"], errors="coerce"
    ).dt.normalize()
    event_key = [
        "odds_calendar_year",
        "odds_event_id",
        "odds_event_name",
        "odds_event_completed_date",
    ]
    rows = []
    for key, group in raw.groupby(event_key, dropna=False, sort=True):
        ids = pd.concat([group["p1_dg_id"], group["p2_dg_id"]], ignore_index=True)
        ids = sorted(
            int(value) for value in pd.to_numeric(ids, errors="coerce").dropna().unique()
        )
        rows.append(
            {
                **dict(zip(event_key, key)),
                "odds_event_name_normalized": canonical_event_name(key[2]),
                "odds_matchup_rows": int(len(group)),
                "odds_books": "|".join(
                    sorted(str(value) for value in group["book"].dropna().unique())
                ),
                "odds_bet_types": "|".join(
                    sorted(str(value) for value in group["bet_type"].dropna().unique())
                ),
                "odds_unique_golfer_count": len(ids),
                "odds_dg_ids_private": ids,
            }
        )
    return pd.DataFrame(rows)


def build_odds_event_crosswalk(
    panel: pd.DataFrame, odds_cache: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    local_columns = [
        "event_key",
        "tour",
        "year",
        "event_id",
        "event_name",
        "event_name_normalized",
        "event_start_date",
        "event_end_date",
    ]
    local = panel[local_columns].drop_duplicates("event_key").copy()
    participant_sets = panel.groupby("event_key")["dg_id"].agg(
        lambda values: set(int(value) for value in values)
    )
    local["local_unique_golfer_count"] = local["event_key"].map(
        participant_sets.map(len)
    )
    odds = read_matchup_event_inventory(odds_cache)
    if odds.empty:
        empty = pd.DataFrame(
            columns=[
                "odds_calendar_year",
                "odds_event_id",
                "odds_event_name",
                "odds_event_completed_date",
                "local_event_key",
                "event_identity_status",
            ]
        )
        local["odds_event_match_count"] = 0
        return empty, local, {"odds_cache_available": False, "odds_events": 0}

    output_rows = []
    for odds_row in odds.itertuples(index=False):
        year = int(odds_row.odds_calendar_year)
        candidates = local.loc[local["year"].eq(year)].copy()
        if pd.notna(odds_row.odds_event_completed_date):
            candidates["date_delta_days"] = (
                candidates["event_end_date"] - odds_row.odds_event_completed_date
            ).dt.days.abs()
            near = candidates["date_delta_days"].le(3)
            # Keep exact-name candidates even if an unusual Monday/delay moves the date.
            exact_name = candidates["event_name_normalized"].eq(
                odds_row.odds_event_name_normalized
            )
            candidates = candidates.loc[near | exact_name].copy()
        else:
            candidates["date_delta_days"] = np.nan
            candidates = candidates.loc[
                candidates["event_name_normalized"].eq(
                    odds_row.odds_event_name_normalized
                )
            ].copy()

        odds_ids = set(odds_row.odds_dg_ids_private)
        scored: list[dict] = []
        for candidate in candidates.itertuples(index=False):
            local_ids = participant_sets[candidate.event_key]
            overlap = len(odds_ids & local_ids)
            overlap_rate = overlap / min(len(odds_ids), len(local_ids)) if odds_ids and local_ids else 0.0
            name_similarity = event_name_similarity(
                odds_row.odds_event_name, candidate.event_name
            )
            date_delta = (
                int(candidate.date_delta_days)
                if pd.notna(candidate.date_delta_days)
                else 999
            )
            # Golfer overlap is the strongest identity evidence, then date and name.
            identity_score = (
                min(overlap, 25) * 10.0
                + overlap_rate * 20.0
                + name_similarity * 5.0
                + max(0, 3 - date_delta)
            )
            scored.append(
                {
                    "local_event_key": candidate.event_key,
                    "local_event_id": int(candidate.event_id),
                    "local_event_name": candidate.event_name,
                    "local_event_start_date": candidate.event_start_date,
                    "local_event_end_date": candidate.event_end_date,
                    "date_delta_days": date_delta if date_delta != 999 else np.nan,
                    "event_name_similarity": name_similarity,
                    "golfer_overlap_count": overlap,
                    "golfer_overlap_rate": overlap_rate,
                    "identity_score": identity_score,
                }
            )
        scored.sort(
            key=lambda row: (
                row["identity_score"],
                row["golfer_overlap_count"],
                row["event_name_similarity"],
                -float(row["date_delta_days"])
                if pd.notna(row["date_delta_days"])
                else -999,
            ),
            reverse=True,
        )
        best = scored[0] if scored else None
        runner_up = scored[1] if len(scored) > 1 else None
        accepted = False
        method = "unmatched"
        ambiguous = False
        if best is not None:
            accepted = (
                best["golfer_overlap_count"] >= 2
                or (
                    best["event_name_similarity"] >= 0.80
                    and pd.notna(best["date_delta_days"])
                    and best["date_delta_days"] <= 3
                )
                or (
                    best["event_name_similarity"] == 1.0
                    and best["local_event_start_date"].year == year
                )
            )
            if runner_up is not None:
                ambiguous = (
                    abs(best["identity_score"] - runner_up["identity_score"]) < 1e-9
                    and best["golfer_overlap_count"]
                    == runner_up["golfer_overlap_count"]
                )
            if accepted and not ambiguous:
                if best["golfer_overlap_count"] >= 2:
                    method = "calendar_date_name_and_golfer_overlap"
                else:
                    method = "calendar_date_and_normalized_name"
            else:
                accepted = False
                method = "ambiguous" if ambiguous else "unmatched"

        selected = best if accepted else {}
        output_rows.append(
            {
                "odds_calendar_year": year,
                "odds_event_id": odds_row.odds_event_id,
                "odds_event_name": odds_row.odds_event_name,
                "odds_event_name_normalized": odds_row.odds_event_name_normalized,
                "odds_event_completed_date": odds_row.odds_event_completed_date,
                "odds_matchup_rows": int(odds_row.odds_matchup_rows),
                "odds_books": odds_row.odds_books,
                "odds_bet_types": odds_row.odds_bet_types,
                "odds_unique_golfer_count": int(odds_row.odds_unique_golfer_count),
                **{
                    key: selected.get(key, np.nan)
                    for key in [
                        "local_event_key",
                        "local_event_id",
                        "local_event_name",
                        "local_event_start_date",
                        "local_event_end_date",
                        "date_delta_days",
                        "event_name_similarity",
                        "golfer_overlap_count",
                        "golfer_overlap_rate",
                        "identity_score",
                    ]
                },
                "event_identity_method": method,
                "event_identity_status": "matched" if accepted else method,
                "event_identity_ambiguous": ambiguous,
                "event_id_drift": bool(
                    accepted and str(odds_row.odds_event_id) != str(best["local_event_id"])
                ),
                "runner_up_local_event_key": (
                    runner_up["local_event_key"] if runner_up else pd.NA
                ),
                "runner_up_identity_score": (
                    runner_up["identity_score"] if runner_up else np.nan
                ),
            }
        )

    crosswalk = pd.DataFrame(output_rows)
    matched = crosswalk.loc[crosswalk["event_identity_status"].eq("matched")]
    duplicate_local = matched.duplicated("local_event_key", keep=False)
    # A duplicate local match means the odds archive exposes multiple IDs for one
    # real event. Keep it visible, and do not flatten it into the local inventory.
    duplicate_local_keys = set(matched.loc[duplicate_local, "local_event_key"].dropna())
    crosswalk["local_event_has_multiple_odds_ids"] = crosswalk[
        "local_event_key"
    ].isin(duplicate_local_keys)

    unique_mapping = matched.loc[~duplicate_local].copy()
    identity = local.merge(
        unique_mapping[
            [
                "local_event_key",
                "odds_calendar_year",
                "odds_event_id",
                "odds_event_name",
                "odds_event_completed_date",
                "event_identity_method",
                "event_id_drift",
                "golfer_overlap_count",
                "golfer_overlap_rate",
                "event_name_similarity",
            ]
        ],
        left_on="event_key",
        right_on="local_event_key",
        how="left",
        validate="one_to_one",
    ).drop(columns="local_event_key")
    match_counts = matched["local_event_key"].value_counts()
    identity["odds_event_match_count"] = identity["event_key"].map(match_counts).fillna(0).astype(int)

    audit = {
        "odds_cache_available": True,
        "odds_events": int(len(crosswalk)),
        "matched_odds_events": int(crosswalk["event_identity_status"].eq("matched").sum()),
        "unmatched_odds_events": int(crosswalk["event_identity_status"].eq("unmatched").sum()),
        "ambiguous_odds_events": int(crosswalk["event_identity_status"].eq("ambiguous").sum()),
        "matched_with_event_id_drift": int(
            crosswalk.loc[
                crosswalk["event_identity_status"].eq("matched"), "event_id_drift"
            ].sum()
        ),
        "local_events_with_multiple_odds_ids": int(len(duplicate_local_keys)),
        "local_events": int(len(local)),
        "local_events_with_unique_odds_match": int(
            identity["odds_event_id"].notna().sum()
        ),
    }
    return crosswalk, identity, audit


def build_player_event_outcomes(panel: pd.DataFrame) -> pd.DataFrame:
    group_key = ["event_key", "player_event_key"]
    metadata_columns = [
        "tour",
        "year",
        "event_id",
        "event_name",
        "event_name_normalized",
        "event_start_date",
        "event_end_date",
        "market_evaluation_eligible",
        "dg_id",
        "player_name",
        "player_name_normalized",
    ]
    metadata = panel[group_key + metadata_columns].drop_duplicates(group_key)
    grouped = panel.groupby(group_key, sort=False)
    summary = grouped.agg(
        rounds_played=("round_num", "nunique"),
        max_round_num=("round_num", "max"),
        label_dg_sg_adj_full_mean=("dg_sg_total_field_adjusted", "mean"),
        label_dg_sg_raw_full_mean=("dg_sg_total_raw", "mean"),
        label_dg_sg_adj_full_count=("dg_sg_total_field_adjusted", "count"),
        label_img_accounting_full_mean=(
            "img_accounting_sg_total_field_adjusted",
            "mean",
        ),
        label_img_accounting_full_count=(
            "img_accounting_sg_total_field_adjusted",
            "count",
        ),
        label_img_v2_signal_full_mean=("img_v2_sg_total_signal", "mean"),
        label_img_v2_signal_full_count=("img_v2_sg_total_signal", "count"),
    ).reset_index()

    first_two = panel.loc[panel["round_num"].isin([1, 2])]
    first_two_summary = (
        first_two.groupby(group_key, sort=False)
        .agg(
            label_dg_sg_adj_r1_r2_mean=("dg_sg_total_field_adjusted", "mean"),
            label_dg_sg_adj_r1_r2_count=("dg_sg_total_field_adjusted", "count"),
            label_dg_sg_raw_r1_r2_mean=("dg_sg_total_raw", "mean"),
            label_img_accounting_r1_r2_mean=(
                "img_accounting_sg_total_field_adjusted",
                "mean",
            ),
            label_img_accounting_r1_r2_count=(
                "img_accounting_sg_total_field_adjusted",
                "count",
            ),
            label_img_v2_signal_r1_r2_mean=("img_v2_sg_total_signal", "mean"),
            label_img_v2_signal_r1_r2_count=("img_v2_sg_total_signal", "count"),
        )
        .reset_index()
    )
    summary = summary.merge(first_two_summary, on=group_key, how="left", validate="one_to_one")

    round_value_columns = {
        "dg_sg_total_field_adjusted": "label_dg_sg_adj",
        "dg_sg_total_raw": "label_dg_sg_raw",
        "img_accounting_sg_total_field_adjusted": "label_img_accounting",
        "img_v2_sg_total_signal": "label_img_v2_signal",
        "cascade_prediction": "cascade_prediction",
    }
    for source, prefix in round_value_columns.items():
        pivot = panel.pivot_table(
            index=group_key,
            columns="round_num",
            values=source,
            aggfunc="first",
        )
        pivot.columns = [f"{prefix}_r{int(round_num)}" for round_num in pivot.columns]
        summary = summary.merge(pivot.reset_index(), on=group_key, how="left", validate="one_to_one")

    r1_benchmark_columns = [
        "player_event_key",
        "legacy_asof_date",
        "legacy_asof_lag_days",
        "legacy_skill_est_asof",
        "legacy_time_weighted_sg_asof",
        "legacy_seq_indx_asof",
        "legacy_std_dev_asof",
        "cascade_prediction",
        "cascade_std_dev_50",
    ]
    r1 = (
        panel.loc[panel["round_num"].eq(1), r1_benchmark_columns]
        .drop_duplicates("player_event_key")
        .rename(
            columns={
                "cascade_prediction": "benchmark_preevent_cascade_prediction",
                "cascade_std_dev_50": "benchmark_preevent_cascade_std_dev_50",
            }
        )
    )
    output = metadata.merge(summary, on=group_key, how="left", validate="one_to_one")
    output = output.merge(r1, on="player_event_key", how="left", validate="one_to_one")
    if output["player_event_key"].duplicated().any():
        raise AssertionError("player_event_outcomes is not unique")
    return output.sort_values(
        ["event_start_date", "event_id", "dg_id"]
    ).reset_index(drop=True)


def build_coverage(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, frame in panel.groupby("year", sort=True):
        rows.append(
            {
                "year": int(year),
                "player_round_rows": int(len(frame)),
                "events": int(frame["event_key"].nunique()),
                "players": int(frame["dg_id"].nunique()),
                "dg_adjusted_total_coverage": float(
                    frame["dg_sg_total_field_adjusted"].notna().mean()
                ),
                "weather_coverage": float(frame["weather_wind"].notna().mean()),
                "legacy_asof_coverage": float(
                    frame["legacy_skill_est_asof"].notna().mean()
                ),
                "cascade_prediction_coverage": float(
                    frame["cascade_prediction"].notna().mean()
                ),
                "img_crosswalk_coverage": float(
                    frame["img_crosswalk_available"].mean()
                ),
                "img_accounting_coverage": float(
                    frame["img_accounting_available"].mean()
                ),
                "img_v2_coverage": float(frame["img_v2_available"].mean()),
            }
        )
    return pd.DataFrame(rows)


def make_unmatched_samples(panel: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "player_round_key",
        "event_key",
        "year",
        "event_id",
        "event_name",
        "round_num",
        "round_date",
        "course_num",
        "dg_id",
        "player_name",
        "weather_join_method",
        "cascade_join_method",
        "img_crosswalk_available",
        "img_accounting_available",
        "img_v2_available",
    ]
    problem = (
        panel["weather_join_method"].eq("unmatched")
        | panel["cascade_join_method"].eq("unmatched")
        | ~panel["img_crosswalk_available"]
    )
    return (
        panel.loc[problem, columns]
        .sort_values(["round_date", "event_id", "round_num", "player_name"])
        .head(1000)
        .reset_index(drop=True)
    )


def schema_contract(panel: pd.DataFrame) -> dict:
    def select(prefix: str) -> list[str]:
        return [column for column in panel.columns if column.startswith(prefix)]

    return {
        "keys": [
            "event_key",
            "player_event_key",
            "player_round_key",
            "event_start_date",
            "round_date",
            "dg_id",
            "event_id",
            "year",
            "round_num",
            "course_num",
        ],
        "dg_round_outcomes": select("dg_sg_"),
        "weather_realized_round_outcomes": select("weather_"),
        "legacy_strictly_prior_asof": select("legacy_"),
        "cascade_retrospective_benchmark": select("cascade_"),
        "img_accounting_backfilled_research": select("img_accounting_"),
        "img_v2_retrospective_only": select("img_v2_"),
        "join_provenance": [
            "weather_join_method",
            "cascade_join_method",
            "img_crosswalk_available",
            "img_crosswalk_match_method",
        ],
        "leakage_rules": [
            "For a pre-event row, use historical outcome rows only when round_date < event_start_date.",
            "Process all events sharing event_start_date as one batch before updating history.",
            "Do not use same-event labels, same-round DG source skill, cascade actuals, or realized target-round weather as pre-round inputs.",
            "IMG v2 is retrospective-only: its transform was selected with future-period outcomes and then backfilled.",
            "IMG accounting was backfilled in 2026; it is not an exact historical-production snapshot.",
            "Closing odds are evaluation labels only and are not included in this player-round panel.",
        ],
    }


def final_audit(panel: pd.DataFrame, event_outcomes: pd.DataFrame) -> dict:
    key_duplicates = int(panel["player_round_key"].duplicated().sum())
    event_key_duplicates = int(event_outcomes["player_event_key"].duplicated().sum())
    event_date_violations = int((panel["round_date"] < panel["event_start_date"]).sum())
    legacy_violations = int(
        (
            panel["legacy_asof_date"].notna()
            & ~(panel["legacy_asof_date"] < panel["round_date"])
        ).sum()
    )
    v2_versions = sorted(
        str(value) for value in panel["img_v2_model_version"].dropna().unique()
    )
    result = {
        "player_round_key_duplicates": key_duplicates,
        "player_event_key_duplicates": event_key_duplicates,
        "round_before_event_start_violations": event_date_violations,
        "legacy_asof_strictly_prior_violations": legacy_violations,
        "img_v2_model_versions": v2_versions,
        "all_rows_pga": bool(panel["tour"].eq("pga").all()),
        "all_v2_rows_flagged_retrospective_only": bool(
            panel.loc[panel["img_v2_available"], "img_v2_retrospective_only"].all()
        ),
        "all_accounting_rows_flagged_backfilled": bool(
            panel.loc[
                panel["img_accounting_available"],
                "img_accounting_backfilled_research_only",
            ].all()
        ),
    }
    if key_duplicates or event_key_duplicates or event_date_violations or legacy_violations:
        raise AssertionError(f"Final integrity audit failed: {result}")
    if v2_versions not in ([], ["predictive_skill_pga_validated_v2"]):
        raise AssertionError(f"Unexpected IMG v2 model versions: {v2_versions}")
    return result


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    history_start = pd.Timestamp(args.history_start).normalize()
    market_start = pd.Timestamp(args.market_start).normalize()
    end = pd.Timestamp(args.end).normalize()
    if history_start > market_start:
        raise ValueError("--history-start must be on or before --market-start")
    if market_start > end:
        raise ValueError("--market-start must be on or before --end")
    require_files([args.dg_db, args.cascade, args.crosswalk, args.img_db])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading canonical PGA rounds from {args.dg_db}")
    with sqlite_ro(args.dg_db) as dg_connection:
        panel, rounds_audit = load_player_rounds(
            dg_connection, args.history_start, args.end
        )
        print(f"Loaded {len(panel):,} canonical player-round rows")
        panel, weather_audit = attach_weather(
            panel,
            dg_connection,
            args.history_start,
            args.end,
            args.weather_exclusions,
        )
        print(f"Weather coverage: {weather_audit['coverage_rate']:.1%}")
        panel, legacy_audit = attach_legacy_asof(
            panel, dg_connection, args.history_start, args.end
        )
        print(f"Strict-prior legacy coverage: {legacy_audit['coverage_rate']:.1%}")

    print(f"Loading read-only IMG archive from {args.img_db}")
    with sqlite_ro(args.img_db, immutable=True) as img_connection:
        (
            research_img_crosswalk,
            img_crosswalk_event_audit,
            img_crosswalk_unmatched,
            img_crosswalk_audit,
        ) = build_research_img_crosswalk(
            panel,
            args.crosswalk,
            img_connection,
            args.img_player_crosswalk,
        )
        print(
            "Research-only IMG crosswalk extension: "
            f"{img_crosswalk_audit['research_extension_rows']:,} rows through "
            f"{img_crosswalk_audit['combined_max_round_date']}"
        )
        panel, img_audit = attach_img(
            panel,
            research_img_crosswalk,
            img_connection,
            args.history_start,
            args.end,
        )
    print(f"IMG accounting coverage: {img_audit['panel_accounting_coverage_rate']:.1%}")
    print(f"IMG v2 coverage: {img_audit['panel_v2_coverage_rate']:.1%}")

    panel, cascade_audit = attach_cascade(
        panel, args.cascade, args.history_start, args.end
    )
    print(f"Cascade benchmark coverage: {cascade_audit['prediction_coverage_rate']:.1%}")

    # A stable presentation order keeps keys/provenance visible in previews.
    leading_columns = [
        "event_key",
        "player_event_key",
        "player_round_key",
        "tour",
        "year",
        "event_id",
        "event_name",
        "event_name_normalized",
        "event_start_date",
        "event_end_date",
        "round_num",
        "round_date",
        "course_num",
        "course_name",
        "dg_id",
        "player_name",
        "player_name_normalized",
    ]
    panel = panel[
        leading_columns + [column for column in panel.columns if column not in leading_columns]
    ]
    panel["market_evaluation_eligible"] = panel["event_start_date"].ge(market_start)
    panel = panel.sort_values(
        ["event_start_date", "event_id", "round_num", "dg_id", "course_num"]
    ).reset_index(drop=True)

    event_outcomes = build_player_event_outcomes(panel)
    odds_crosswalk, event_identity, odds_audit = build_odds_event_crosswalk(
        panel, args.odds_cache
    )
    if not event_identity.empty and "odds_event_id" in event_identity:
        event_odds_columns = [
            "event_key",
            "odds_calendar_year",
            "odds_event_id",
            "odds_event_name",
            "odds_event_completed_date",
            "event_identity_method",
            "event_id_drift",
            "odds_event_match_count",
        ]
        event_outcomes = event_outcomes.merge(
            event_identity[event_odds_columns],
            on="event_key",
            how="left",
            validate="many_to_one",
        )
    coverage = build_coverage(panel)
    unmatched = make_unmatched_samples(panel)
    integrity_audit = final_audit(panel, event_outcomes)

    output_paths = {
        "player_round_panel": args.output_dir / "player_round_panel.parquet",
        "player_event_outcomes": args.output_dir / "player_event_outcomes.parquet",
        "odds_event_crosswalk": args.output_dir / "odds_event_crosswalk.parquet",
        "event_identity_panel": args.output_dir / "event_identity_panel.parquet",
        "img_crosswalk_research_extended": args.output_dir
        / "img_crosswalk_research_extended.parquet",
        "img_crosswalk_extension_by_event": args.output_dir
        / "img_crosswalk_extension_by_event.csv",
        "img_crosswalk_extension_unmatched": args.output_dir
        / "img_crosswalk_extension_unmatched.csv",
        "coverage_by_year": args.output_dir / "coverage_by_year.csv",
        "unmatched_join_samples": args.output_dir / "unmatched_join_samples.csv",
        "audit": args.output_dir / "audit.json",
        "manifest": args.output_dir / "manifest.json",
    }
    print(f"Writing research outputs to {args.output_dir}")
    atomic_parquet(panel, output_paths["player_round_panel"])
    atomic_parquet(event_outcomes, output_paths["player_event_outcomes"])
    atomic_parquet(odds_crosswalk, output_paths["odds_event_crosswalk"])
    atomic_parquet(event_identity, output_paths["event_identity_panel"])
    atomic_parquet(
        research_img_crosswalk,
        output_paths["img_crosswalk_research_extended"],
    )
    atomic_csv(
        img_crosswalk_event_audit,
        output_paths["img_crosswalk_extension_by_event"],
    )
    atomic_csv(
        img_crosswalk_unmatched,
        output_paths["img_crosswalk_extension_unmatched"],
    )
    atomic_csv(coverage, output_paths["coverage_by_year"])
    atomic_csv(unmatched, output_paths["unmatched_join_samples"])

    audit = {
        "created_at_utc": utc_now(),
        "requested_window": {
            "history_start": args.history_start,
            "market_start": args.market_start,
            "end": args.end,
        },
        "player_rounds": rounds_audit,
        "weather": weather_audit,
        "legacy_asof": legacy_audit,
        "img": {**img_audit, "research_crosswalk_extension": img_crosswalk_audit},
        "cascade": cascade_audit,
        "odds_event_identity": odds_audit,
        "integrity": integrity_audit,
        "output_counts": {
            "player_round_rows": int(len(panel)),
            "player_event_rows": int(len(event_outcomes)),
            "local_events": int(panel["event_key"].nunique()),
            "odds_event_crosswalk_rows": int(len(odds_crosswalk)),
            "img_crosswalk_rows": int(len(research_img_crosswalk)),
            "img_crosswalk_extension_rows": int(
                img_crosswalk_audit["research_extension_rows"]
            ),
        },
    }
    atomic_json(audit, output_paths["audit"])

    inputs = {
        "dg_historical_db": file_metadata(args.dg_db),
        "cascade_benchmark": file_metadata(args.cascade),
        "img_crosswalk": file_metadata(args.crosswalk),
        "img_archive_db": file_metadata(args.img_db),
    }
    if args.weather_exclusions.exists():
        inputs["weather_exclusions"] = file_metadata(args.weather_exclusions)
    if args.img_player_crosswalk.exists():
        inputs["img_player_crosswalk"] = file_metadata(args.img_player_crosswalk)
    odds_manifest = args.odds_cache / "manifest.json"
    if odds_manifest.exists():
        inputs["odds_cache_manifest"] = file_metadata(odds_manifest)

    manifest = {
        "created_at_utc": utc_now(),
        "purpose": "Research-only frozen outcome panel for walk-forward base-skill and market evaluation.",
        "production_changes": False,
        "script": str(Path(__file__).resolve()),
        "parameters": {
            "history_start": args.history_start,
            "market_start": args.market_start,
            "end": args.end,
            "tour": "pga",
            "img_signal_model_version": "predictive_skill_pga_validated_v2",
        },
        "inputs": inputs,
        "outputs": {
            name: {
                **file_metadata(path),
                "rows": int(
                    {
                        "player_round_panel": len(panel),
                        "player_event_outcomes": len(event_outcomes),
                        "odds_event_crosswalk": len(odds_crosswalk),
                        "event_identity_panel": len(event_identity),
                        "img_crosswalk_research_extended": len(
                            research_img_crosswalk
                        ),
                        "img_crosswalk_extension_by_event": len(
                            img_crosswalk_event_audit
                        ),
                        "img_crosswalk_extension_unmatched": len(
                            img_crosswalk_unmatched
                        ),
                        "coverage_by_year": len(coverage),
                        "unmatched_join_samples": len(unmatched),
                    }.get(name, 1)
                ),
            }
            for name, path in output_paths.items()
            if name not in {"manifest"} and path.exists()
        },
        "schema_contract": schema_contract(panel),
        "odds_event_identity_rule": (
            "Map odds events to local events using calendar year, event-completion/end-date "
            "proximity, normalized event-name similarity, and DataGolf golfer-ID overlap. "
            "Never equate odds event_id with player_rounds event_id."
        ),
        "known_limitations": [
            "sg_preds_resids.csv is a current-code historical reconstruction, not an exact archived production snapshot.",
            "model_results2 has dates but no intraday publication timestamps; using only Date < round_date is intentionally conservative.",
            "Realized weather can normalize prior observed rounds, but target-round realized weather is oracle-only for a pre-round betting backtest.",
            "IMG accounting and v2 data were backfilled/recomputed in 2026 and were not live snapshots at historical bet time.",
            "The current PGA v2 transform was selected using future-period outcomes; it must remain a retrospective sensitivity, not the primary untouched result.",
            "Weather name fallback refuses ambiguous normalized keys rather than choosing a spelling variant silently.",
        ],
    }
    atomic_json(manifest, output_paths["manifest"])
    print("Research panel build complete")
    print(json.dumps(audit["output_counts"], indent=2))


if __name__ == "__main__":
    main()
