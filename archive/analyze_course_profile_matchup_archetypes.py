"""Postprocess the three-arm course-profile matchup sensitivity.

Research only.  This script does not modify production inputs or outputs.  It
consumes a course-profile sensitivity run with the canonical probability arms
``profile_full``, ``profile_half``, and ``profile_none`` and produces the
pre-registered matchup breakdowns:

* overall production-style close-price performance;
* strict and broad pre-event driving distance/accuracy archetypes;
* theoretical-EV buckets;
* calendar-year slices;
* recurring-player attribution; and
* paired 10,000-draw whole-event bootstrap intervals which retain every event,
  including events with zero bets in a slice.

The market scorer may encode FULL rows as ``arm=baseline`` and HALF/NONE rows
as ``arm=challenger`` plus ``challenger_name``.  Those rows are normalized and
the duplicated FULL comparison rows are checked before they are deduplicated.

Driving profiles intentionally use the frozen research definition rather than
the newer tour-population dashboard archetype:

* rolling values are built only from rounds strictly before the event;
* 100+ observations use the average of the latest-50 and latest-100 means;
* 50-99 use the latest-50 mean; fewer than 50 use all available observations;
* players with fewer than 20 prior SG rounds remain ``unknown`` but their
  available driving values still participate in the event-field percentile
  ranks; and
* ``short_accurate`` means distance percentile < 35 and accuracy percentile
  >= 70.

Example
-------
python archive/analyze_course_profile_matchup_archetypes.py \
  --input-dir permanent_data/research/course_profile_sensitivity_20230822_20260822 \
  --output-dir permanent_data/research/course_profile_sensitivity_20230822_20260822/matchup_archetype_audit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
V4_ROOT = (
    REPO
    / "permanent_data"
    / "research"
    / "shot_mean_closing_backtest_20230822_20260822_v4"
)
DEFAULT_ROSTER = (
    V4_ROOT
    / "matchup_skill_bucket_audit"
    / "event_player_projected_skill.parquet"
)
DEFAULT_REFERENCE_PROBABILITIES = V4_ROOT / "probabilities.parquet"
DEFAULT_DB = Path(r"C:\Users\McKinley Slade\dev\sim_prep\work\dg_historical.db")

ARM_FULL = "profile_full"
ARM_HALF = "profile_half"
ARM_NONE = "profile_none"
ARM_ORDER = (ARM_FULL, ARM_HALF, ARM_NONE)
ARM_PAIRS = (
    (ARM_HALF, ARM_FULL),
    (ARM_NONE, ARM_FULL),
    (ARM_HALF, ARM_NONE),
)

STRICT_ORDER = (
    "short_accurate",
    "long_wild",
    "long_accurate",
    "other_known",
    "unknown",
)
DISTANCE_ORDER = ("short", "middle", "long")
ACCURACY_ORDER = ("wild", "average_accuracy", "accurate")
BROAD_ORDER = tuple(
    f"{distance}__{accuracy}"
    for distance in DISTANCE_ORDER
    for accuracy in ACCURACY_ORDER
) + ("unknown",)
EV_ORDER = ("0_to_lt3", "3_to_lt5", "5_to_lt10", "ge10", "negative", "missing")
PROFILE_GAP_ORDER = (
    "selection_disfavored_ge_0p05_sg",
    "selection_disfavored_0p02_to_0p05_sg",
    "near_zero_pm_0p02_sg",
    "selection_favored_0p02_to_0p05_sg",
    "selection_favored_ge_0p05_sg",
    "missing",
)

PRIOR_FIVE = {
    "14704",  # Andrew Putnam
    "17606",  # Daniel Berger
    "18554",  # Aaron Rai
    "8825",   # Brian Harman
    "14578",  # Russell Henley
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--bets", type=Path)
    parser.add_argument("--candidate-bets", type=Path)
    parser.add_argument("--probabilities", type=Path)
    parser.add_argument(
        "--profile-attribution",
        type=Path,
        help=(
            "Optional event/player course-profile attribution. If omitted, "
            "<input-dir>/event_field_profile_attribution.parquet is used when present."
        ),
    )
    parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--line-policy", default="best_of_sharp")
    parser.add_argument("--sizing", default="production_style")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--min-recurring-bets", type=int, default=10)
    parser.add_argument("--min-recurring-events", type=int, default=5)
    parser.add_argument(
        "--reference-probabilities",
        type=Path,
        default=DEFAULT_REFERENCE_PROBABILITIES,
    )
    parser.add_argument("--skip-full-reference-check", action="store_true")
    return parser.parse_args(argv)


def _normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _canonical_player_id(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError, OverflowError):
        return _normalize_text(value)


def _event_key(frame: pd.DataFrame) -> pd.Series:
    if "event_key" in frame.columns:
        return frame["event_key"].astype(str).str.strip()
    if "event_id" in frame.columns:
        return frame["event_id"].astype(str).str.strip()
    raise ValueError("input is missing event_key/event_id")


def _arm_alias(value: object) -> str:
    normalized = _normalize_text(value).replace("-", "_").replace(" ", "_")
    aliases = {
        "profile_full": ARM_FULL,
        "course_profile_full": ARM_FULL,
        "full": ARM_FULL,
        "old_full_strength": ARM_FULL,
        "baseline": ARM_FULL,
        "profile_half": ARM_HALF,
        "course_profile_half": ARM_HALF,
        "half": ARM_HALF,
        "profile_none": ARM_NONE,
        "course_profile_none": ARM_NONE,
        "none": ARM_NONE,
        "no": ARM_NONE,
        "no_mean_course_fit": ARM_NONE,
    }
    return aliases.get(normalized, normalized)


def _canonical_bet_arm(row: pd.Series) -> str:
    raw = _normalize_text(row.get("arm")).replace("-", "_").replace(" ", "_")
    if raw == "baseline":
        return ARM_FULL
    if raw == "challenger":
        challenger = row.get("challenger_name", row.get("comparison_arm", ""))
        return _arm_alias(challenger)
    return _arm_alias(raw)


def _american_to_decimal(odds: pd.Series) -> pd.Series:
    values = pd.to_numeric(odds, errors="coerce")
    return pd.Series(
        np.where(values > 0, 1.0 + values / 100.0, 1.0 + 100.0 / values.abs()),
        index=values.index,
        dtype=float,
    ).where(values.abs().ge(100.0))


def _ties_lose(value: object) -> bool:
    """Mirror the sealed scorer's matchup tie-settlement classification."""
    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    return (
        "separate" in text
        or "ties lose" in text
        or "tie loses" in text
        or "three way" in text
        or "3 way" in text
    )


def _ev_bucket(values: pd.Series) -> pd.Categorical:
    numeric = pd.to_numeric(values, errors="coerce")
    labels = np.select(
        [
            numeric.ge(0.0) & numeric.lt(0.03),
            numeric.ge(0.03) & numeric.lt(0.05),
            numeric.ge(0.05) & numeric.lt(0.10),
            numeric.ge(0.10),
            numeric.lt(0.0),
        ],
        list(EV_ORDER[:-1]),
        default="missing",
    )
    return pd.Categorical(labels, categories=EV_ORDER, ordered=True)


def normalize_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["event_id"] = _event_key(output)
    if "market" in output.columns:
        output = output[
            output["market"].astype(str).str.casefold().eq("tournament_matchup")
        ].copy()
    output["arm"] = output["arm"].map(_arm_alias)
    output = output[output["arm"].isin(ARM_ORDER)].copy()
    output["selection_id"] = output["selection_id"].map(_canonical_player_id)
    output["opponent_id"] = output["opponent_id"].map(_canonical_player_id)
    output["event_start"] = pd.to_datetime(output.get("event_start"), errors="coerce")
    for column in ("probability", "push_probability"):
        output[column] = pd.to_numeric(output.get(column), errors="coerce")
    required = ["event_id", "arm", "selection_id", "opponent_id", "probability"]
    if output[required].isna().any().any() or output["selection_id"].eq("").any():
        raise ValueError("probability input contains null/blank required matchup keys")
    key = ["event_id", "arm", "selection_id", "opponent_id"]
    duplicates = output.duplicated(key, keep=False)
    if duplicates.any():
        raise ValueError(f"probability input contains {int(duplicates.sum())} duplicate keys")
    missing_arms = sorted(set(ARM_ORDER) - set(output["arm"]))
    if missing_arms:
        raise ValueError(f"probability input is missing arms: {missing_arms}")
    return output.sort_values(key, kind="mergesort").reset_index(drop=True)


def validate_probability_arm_menus(probabilities: pd.DataFrame) -> dict[str, Any]:
    key = ["event_id", "selection_id", "opponent_id"]
    reference = pd.MultiIndex.from_frame(
        probabilities.loc[probabilities["arm"].eq(ARM_FULL), key]
    )
    audit: dict[str, Any] = {ARM_FULL: {"rows": int(len(reference))}}
    for arm in (ARM_HALF, ARM_NONE):
        menu = pd.MultiIndex.from_frame(
            probabilities.loc[probabilities["arm"].eq(arm), key]
        )
        full_only = reference.difference(menu)
        arm_only = menu.difference(reference)
        audit[arm] = {
            "rows": int(len(menu)),
            "full_only_rows": int(len(full_only)),
            "arm_only_rows": int(len(arm_only)),
        }
        if len(full_only) or len(arm_only):
            raise ValueError(f"{arm} probability menu differs from FULL: {audit[arm]}")
    return audit


def _filter_variant(
    frame: pd.DataFrame,
    column_candidates: Sequence[str],
    requested: str,
) -> pd.DataFrame:
    for column in column_candidates:
        if column not in frame.columns:
            continue
        normalized = frame[column].astype(str).map(_normalize_text)
        target = _normalize_text(requested)
        if normalized.eq(target).any():
            return frame.loc[normalized.eq(target)].copy()
        raise ValueError(
            f"requested {requested!r} not present in {column}; "
            f"found {sorted(normalized.dropna().unique().tolist())}"
        )
    return frame


def _verify_and_dedupe_full_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    key = ["event_id", "arm", "selection_id", "opponent_id"]
    duplicates = frame.duplicated(key, keep=False)
    if not duplicates.any():
        return frame, 0
    duplicated = frame.loc[duplicates].copy()
    check_columns = [
        column
        for column in (
            "stake",
            "pnl",
            "ev_per_unit",
            "probability",
            "push_probability",
            "close_odds",
            "source_price",
            "book",
            "bookmaker",
            "settlement_status",
        )
        if column in duplicated.columns
    ]
    conflicts: list[tuple[str, str, str, str]] = []
    for values, group in duplicated.groupby(key, dropna=False, sort=False):
        for column in check_columns:
            series = group[column]
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().sum() == len(series):
                if not np.allclose(numeric, numeric.iloc[0], atol=1e-12, rtol=0.0):
                    conflicts.append(values)
                    break
            elif series.astype(str).nunique(dropna=False) > 1:
                conflicts.append(values)
                break
    if conflicts:
        raise ValueError(
            "duplicate logical bet keys contain conflicting values; "
            f"first conflicts: {conflicts[:5]}"
        )
    before = len(frame)
    output = frame.sort_values(key, kind="mergesort").drop_duplicates(key, keep="first")
    return output.reset_index(drop=True), before - len(output)


def normalize_bets(
    frame: pd.DataFrame,
    line_policy: str,
    sizing: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    output = frame.copy()
    if "market" in output.columns:
        output = output[
            output["market"].astype(str).str.casefold().eq("tournament_matchup")
        ].copy()
    if "market_group" in output.columns:
        output = output[
            output["market_group"].astype(str).str.casefold().eq("matchup")
        ].copy()
    if "price_type" in output.columns:
        output = output[output["price_type"].astype(str).str.casefold().eq("close")].copy()
    if "is_graded" in output.columns:
        output = output[output["is_graded"].fillna(False).astype(bool)].copy()
    output = _filter_variant(output, ("line_policy_name", "line_policy"), line_policy)
    output = _filter_variant(output, ("sizing_name",), sizing)
    output["event_id"] = _event_key(output)
    output["arm"] = output.apply(_canonical_bet_arm, axis=1)
    output = output[output["arm"].isin(ARM_ORDER)].copy()
    output["selection_id"] = output["selection_id"].map(_canonical_player_id)
    output["opponent_id"] = output["opponent_id"].map(_canonical_player_id)
    output["selection_name"] = output.get("selection_name", "").map(_normalize_text)
    output["opponent_name"] = output.get("opponent_name", "").map(_normalize_text)
    output["event_start"] = pd.to_datetime(output.get("event_start"), errors="coerce")
    if "year" in output.columns:
        output["year"] = pd.to_numeric(output["year"], errors="coerce")
    else:
        output["year"] = output["event_start"].dt.year
    missing_year = output["year"].isna()
    if missing_year.any():
        extracted = output.loc[missing_year, "event_id"].str.extract(r"R(\d{4})")[0]
        output.loc[missing_year, "year"] = pd.to_numeric(extracted, errors="coerce")
    for column in ("stake", "pnl", "ev_per_unit", "probability", "push_probability"):
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    required = ["event_id", "arm", "selection_id", "opponent_id", "stake", "pnl"]
    missing_columns = [column for column in required if column not in output.columns]
    if missing_columns:
        raise ValueError(f"bet input is missing required columns: {missing_columns}")
    if output[required].isna().any().any() or output["selection_id"].eq("").any():
        raise ValueError("bet input contains null/blank required matchup fields")
    if (output["stake"] <= 0).any():
        raise ValueError("bet input contains nonpositive stakes")
    if "ev_per_unit" not in output.columns:
        raise ValueError("bet input must contain ev_per_unit for pre-registered buckets")
    output, deduped = _verify_and_dedupe_full_rows(output)
    missing_arms = sorted(set(ARM_ORDER) - set(output["arm"]))
    if missing_arms:
        raise ValueError(f"filtered bet input is missing arms: {missing_arms}")
    output["ev_bucket_native"] = _ev_bucket(output["ev_per_unit"])
    audit = {"input_rows": int(len(frame)), "normalized_rows": int(len(output)), "deduped": deduped}
    return output, audit


def load_roster(path: Path | None, probabilities: pd.DataFrame) -> pd.DataFrame:
    if path is not None and path.is_file():
        roster = pd.read_parquet(path)
        roster = roster.rename(
            columns={"selection_id": "dg_id", "player_name": "player_name"}
        )
        roster["event_id"] = _event_key(roster)
        roster["dg_id"] = roster["dg_id"].map(_canonical_player_id)
        roster["player_name"] = roster["player_name"].map(_normalize_text)
        roster["event_start"] = pd.to_datetime(roster.get("event_start"), errors="coerce")
        roster = roster[["event_id", "event_start", "dg_id", "player_name"]]
    else:
        roster = probabilities[
            ["event_id", "event_start", "selection_id", "selection_name"]
        ].rename(columns={"selection_id": "dg_id", "selection_name": "player_name"})
        roster["player_name"] = roster["player_name"].map(_normalize_text)
    roster = roster[roster["event_id"].isin(probabilities["event_id"].unique())].copy()
    roster = roster.dropna(subset=["event_start"])
    conflicts = (
        roster.groupby(["event_id", "dg_id"], dropna=False)["player_name"]
        .nunique(dropna=False)
        .gt(1)
    )
    if conflicts.any():
        raise ValueError("roster has conflicting names for an event/player ID")
    roster = roster.sort_values(["event_id", "dg_id"], kind="mergesort")
    roster = roster.drop_duplicates(["event_id", "dg_id"], keep="last")
    if roster["dg_id"].eq("").any():
        raise ValueError("roster contains blank player IDs")
    return roster.reset_index(drop=True)


def _read_history(db_path: Path, roster: pd.DataFrame) -> pd.DataFrame:
    if not db_path.is_file():
        raise FileNotFoundError(f"historical database not found: {db_path}")
    ids = sorted({_canonical_player_id(value) for value in roster["dg_id"]} - {""})
    numeric_ids = [int(value) for value in ids if value.isdigit()]
    cutoff = roster["event_start"].max() + pd.Timedelta(days=1)
    pieces: list[pd.DataFrame] = []
    connection = sqlite3.connect(str(db_path))
    try:
        for start in range(0, len(numeric_ids), 800):
            chunk = numeric_ids[start : start + 800]
            placeholders = ",".join("?" for _ in chunk)
            query = f"""
                SELECT dg_id, round_date, sg_ott_adj, rel_dd, driving_acc
                FROM player_rounds
                WHERE sg_ott_adj IS NOT NULL
                  AND round_date < ?
                  AND dg_id IN ({placeholders})
                ORDER BY dg_id, round_date DESC
            """
            params: list[Any] = [cutoff.strftime("%Y-%m-%d"), *chunk]
            pieces.append(pd.read_sql_query(query, connection, params=params))
    finally:
        connection.close()
    if not pieces:
        return pd.DataFrame(columns=["dg_id", "round_date", "sg_ott_adj", "rel_dd", "driving_acc"])
    history = pd.concat(pieces, ignore_index=True)
    history["dg_id"] = history["dg_id"].map(_canonical_player_id)
    history["round_date"] = pd.to_datetime(history["round_date"], errors="coerce")
    history = history.dropna(subset=["round_date"])
    return history.sort_values(["dg_id", "round_date"], ascending=[True, False], kind="mergesort")


def _tiered_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    count = len(numeric)
    if count == 0:
        return np.nan
    if count >= 100:
        return float(0.5 * numeric.iloc[:50].mean() + 0.5 * numeric.iloc[:100].mean())
    if count >= 50:
        return float(numeric.iloc[:50].mean())
    return float(numeric.mean())


def build_driving_profiles(roster: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    histories = {
        dg_id: group.reset_index(drop=True)
        for dg_id, group in history.groupby("dg_id", sort=False)
    }
    event_frames: list[pd.DataFrame] = []
    for (event_id, event_start), field in roster.groupby(
        ["event_id", "event_start"], sort=True, observed=True
    ):
        records: list[dict[str, Any]] = []
        for row in field.itertuples(index=False):
            player_history = histories.get(row.dg_id)
            if player_history is None:
                prior = player_history
            else:
                prior = player_history[player_history["round_date"].lt(event_start)]
            if prior is None or prior.empty:
                records.append(
                    {
                        "event_id": event_id,
                        "event_start": event_start,
                        "dg_id": row.dg_id,
                        "player_name": row.player_name,
                        "prior_rounds": 0,
                        "rel_dd_rolling": np.nan,
                        "driving_acc_rolling": np.nan,
                    }
                )
                continue
            records.append(
                {
                    "event_id": event_id,
                    "event_start": event_start,
                    "dg_id": row.dg_id,
                    "player_name": row.player_name,
                    "prior_rounds": int(len(prior)),
                    "rel_dd_rolling": _tiered_mean(prior["rel_dd"]),
                    "driving_acc_rolling": _tiered_mean(prior["driving_acc"]),
                }
            )
        profiles = pd.DataFrame(records)
        # The percentile pool includes partial-data players with a valid metric;
        # they remain unknown below after contributing to the field reference.
        profiles["distance_percentile"] = (
            profiles["rel_dd_rolling"].rank(method="average", pct=True) * 100.0
        )
        profiles["accuracy_percentile"] = (
            profiles["driving_acc_rolling"].rank(method="average", pct=True) * 100.0
        )
        known = (
            profiles["prior_rounds"].ge(20)
            & profiles["distance_percentile"].notna()
            & profiles["accuracy_percentile"].notna()
        )
        profiles["strict_archetype"] = "unknown"
        profiles.loc[known, "strict_archetype"] = "other_known"
        profiles.loc[
            known
            & profiles["distance_percentile"].lt(35.0)
            & profiles["accuracy_percentile"].ge(70.0),
            "strict_archetype",
        ] = "short_accurate"
        profiles.loc[
            known
            & profiles["distance_percentile"].ge(80.0)
            & profiles["accuracy_percentile"].lt(40.0),
            "strict_archetype",
        ] = "long_wild"
        profiles.loc[
            known
            & profiles["distance_percentile"].ge(80.0)
            & profiles["accuracy_percentile"].ge(40.0),
            "strict_archetype",
        ] = "long_accurate"

        distance = np.select(
            [
                profiles["distance_percentile"].lt(100.0 / 3.0),
                profiles["distance_percentile"].lt(200.0 / 3.0),
            ],
            ["short", "middle"],
            default="long",
        )
        accuracy = np.select(
            [
                profiles["accuracy_percentile"].lt(100.0 / 3.0),
                profiles["accuracy_percentile"].lt(200.0 / 3.0),
            ],
            ["wild", "average_accuracy"],
            default="accurate",
        )
        profiles["broad_archetype"] = pd.Series(distance, index=profiles.index) + "__" + pd.Series(
            accuracy, index=profiles.index
        )
        profiles.loc[~known, "broad_archetype"] = "unknown"
        profiles["strict_archetype"] = pd.Categorical(
            profiles["strict_archetype"], categories=STRICT_ORDER, ordered=True
        )
        profiles["broad_archetype"] = pd.Categorical(
            profiles["broad_archetype"], categories=BROAD_ORDER, ordered=True
        )
        event_frames.append(profiles)
    result = pd.concat(event_frames, ignore_index=True)
    if result.duplicated(["event_id", "dg_id"]).any():
        raise ValueError("driving profile reconstruction produced duplicate event/player keys")
    return result


def attach_profiles(bets: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    selection = profiles.rename(
        columns={
            column: f"selection_{column}"
            for column in profiles.columns
            if column not in {"event_id", "dg_id"}
        }
    ).rename(columns={"dg_id": "selection_id"})
    opponent = profiles.rename(
        columns={
            column: f"opponent_{column}"
            for column in profiles.columns
            if column not in {"event_id", "dg_id"}
        }
    ).rename(columns={"dg_id": "opponent_id"})
    output = bets.merge(
        selection,
        on=["event_id", "selection_id"],
        how="left",
        validate="many_to_one",
    )
    output = output.merge(
        opponent,
        on=["event_id", "opponent_id"],
        how="left",
        validate="many_to_one",
    )
    output["strict_archetype"] = output["selection_strict_archetype"].astype(object).fillna("unknown")
    output["broad_archetype"] = output["selection_broad_archetype"].astype(object).fillna("unknown")
    return output


def add_full_reference_edge(bets: pd.DataFrame, probabilities: pd.DataFrame) -> pd.DataFrame:
    full = probabilities[probabilities["arm"].eq(ARM_FULL)][
        ["event_id", "selection_id", "opponent_id", "probability", "push_probability"]
    ].rename(
        columns={
            "probability": "full_probability",
            "push_probability": "full_push_probability",
        }
    )
    output = bets.merge(
        full,
        on=["event_id", "selection_id", "opponent_id"],
        how="left",
        validate="many_to_one",
    )
    odds_column = next(
        (column for column in ("source_price", "close_odds", "odds") if column in output.columns),
        None,
    )
    if odds_column is None:
        output["full_reference_ev"] = np.nan
    else:
        decimal = _american_to_decimal(output[odds_column])
        p_win = output["full_probability"].astype(float)
        push = output["full_push_probability"].fillna(0.0).astype(float)
        tie_rule = output.get("tie_rule", pd.Series("", index=output.index))
        effective_push = push.where(~tie_rule.map(_ties_lose), 0.0)
        loss_probability = 1.0 - p_win - effective_push
        output["full_reference_ev"] = p_win * (decimal - 1.0) - loss_probability
    output["ev_bucket_full_reference"] = _ev_bucket(output["full_reference_ev"])
    return output


def load_profile_attribution(path: Path, event_universe: Sequence[str]) -> pd.DataFrame:
    """Load the exact profile contribution actually applied in the FULL arm."""
    frame = pd.read_parquet(path).copy()
    required = {"dg_id", "profile_in_baseline"}
    missing = sorted(required - set(frame.columns))
    if missing or not ({"event_id", "event_key"} & set(frame.columns)):
        raise ValueError(
            "profile-attribution input is missing event key/dg_id/profile_in_baseline; "
            f"missing columns: {missing}"
        )
    frame["event_id"] = _event_key(frame)
    frame["dg_id"] = frame["dg_id"].map(_canonical_player_id)
    frame["profile_in_baseline"] = pd.to_numeric(
        frame["profile_in_baseline"], errors="coerce"
    )
    frame = frame[frame["event_id"].isin(event_universe)].copy()
    if frame.duplicated(["event_id", "dg_id"]).any():
        raise ValueError("profile-attribution input has duplicate event/player keys")
    return frame[["event_id", "dg_id", "profile_in_baseline"]].reset_index(drop=True)


def attach_profile_attribution(
    bets: pd.DataFrame, attribution: pd.DataFrame
) -> pd.DataFrame:
    """Attach the applied FULL profile term and selection-minus-opponent gap."""
    selection = attribution.rename(
        columns={
            "dg_id": "selection_id",
            "profile_in_baseline": "selection_profile_in_baseline",
        }
    )
    opponent = attribution.rename(
        columns={
            "dg_id": "opponent_id",
            "profile_in_baseline": "opponent_profile_in_baseline",
        }
    )
    output = bets.merge(
        selection,
        on=["event_id", "selection_id"],
        how="left",
        validate="many_to_one",
    ).merge(
        opponent,
        on=["event_id", "opponent_id"],
        how="left",
        validate="many_to_one",
    )
    output["profile_in_baseline_gap"] = (
        output["selection_profile_in_baseline"]
        - output["opponent_profile_in_baseline"]
    )
    gap = output["profile_in_baseline_gap"]
    labels = np.select(
        [
            gap.lt(-0.05),
            gap.ge(-0.05) & gap.lt(-0.02),
            gap.ge(-0.02) & gap.lt(0.02),
            gap.ge(0.02) & gap.lt(0.05),
            gap.ge(0.05),
        ],
        list(PROFILE_GAP_ORDER[:-1]),
        default="missing",
    )
    output["profile_in_baseline_gap_bucket"] = pd.Categorical(
        labels, categories=PROFILE_GAP_ORDER, ordered=True
    )
    return output


def _settlement_counts(group: pd.DataFrame) -> tuple[int, int, int]:
    if "settlement_status" in group.columns:
        status = group["settlement_status"].astype(str).str.casefold()
        wins = int(status.str.contains("win").sum())
        losses = int(status.str.contains("loss").sum())
        voids = int(status.str.contains("void|push").sum())
        return wins, losses, voids
    pnl = pd.to_numeric(group["pnl"], errors="coerce")
    return int(pnl.gt(0).sum()), int(pnl.lt(0).sum()), int(pnl.eq(0).sum())


def _max_drawdown(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    event = (
        frame.groupby(["event_start", "event_id"], dropna=False, observed=True)["pnl"]
        .sum()
        .sort_index()
    )
    cumulative = event.cumsum().to_numpy(float)
    path = np.concatenate([[0.0], cumulative])
    running_peak = np.maximum.accumulate(path)
    return float(np.max(running_peak - path))


def performance_table(
    frame: pd.DataFrame,
    dimension: str,
    bucket_column: str | None,
    bucket_order: Sequence[object],
) -> pd.DataFrame:
    columns = [
        "dimension",
        "bucket",
        "arm",
        "bets",
        "events_with_bets",
        "stake",
        "pnl",
        "roi",
        "wins",
        "losses",
        "voids",
        "mean_ev",
        "median_ev",
        "positive_events",
        "max_drawdown",
    ]
    rows: list[dict[str, Any]] = []
    for bucket in bucket_order:
        sliced = frame if bucket_column is None else frame[frame[bucket_column].astype(str).eq(str(bucket))]
        for arm in ARM_ORDER:
            group = sliced[sliced["arm"].eq(arm)]
            stake = float(group["stake"].sum())
            pnl = float(group["pnl"].sum())
            wins, losses, voids = _settlement_counts(group)
            rows.append(
                {
                    "dimension": dimension,
                    "bucket": bucket,
                    "arm": arm,
                    "bets": int(len(group)),
                    "events_with_bets": int(group["event_id"].nunique()),
                    "stake": stake,
                    "pnl": pnl,
                    "roi": pnl / stake if stake > 0 else np.nan,
                    "wins": wins,
                    "losses": losses,
                    "voids": voids,
                    "mean_ev": float(group["ev_per_unit"].mean()) if len(group) else np.nan,
                    "median_ev": float(group["ev_per_unit"].median()) if len(group) else np.nan,
                    "positive_events": int((group.groupby("event_id")["pnl"].sum() > 0).sum()),
                    "max_drawdown": _max_drawdown(group),
                }
            )
    return pd.DataFrame(rows, columns=columns)


class BootstrapWeights:
    def __init__(self, draws: int, seed: int):
        if draws <= 0:
            raise ValueError("bootstrap draws must be positive")
        self.draws = draws
        self.seed = seed
        self._cache: dict[tuple[str, ...], np.ndarray] = {}

    def get(self, events: Sequence[str]) -> np.ndarray:
        key = tuple(events)
        if key not in self._cache:
            if not key:
                self._cache[key] = np.zeros((self.draws, 0), dtype=np.int16)
            else:
                digest = hashlib.sha256("\x1f".join(key).encode("utf-8")).digest()
                derived = self.seed ^ int.from_bytes(digest[:8], "little")
                rng = np.random.default_rng(derived)
                probability = np.full(len(key), 1.0 / len(key))
                self._cache[key] = rng.multinomial(
                    len(key), probability, size=self.draws
                ).astype(np.int16)
        return self._cache[key]


def _bootstrap_metric_row(
    dimension: str,
    bucket: object,
    arm_a: str,
    arm_b: str,
    metric: str,
    point: float,
    samples: np.ndarray,
    event_count: int,
    draws: int,
) -> dict[str, Any]:
    finite = samples[np.isfinite(samples)]
    return {
        "dimension": dimension,
        "bucket": bucket,
        "comparison": f"{arm_a}_minus_{arm_b}",
        "arm_a": arm_a,
        "arm_b": arm_b,
        "metric": metric,
        "estimate": point,
        "ci_low": float(np.quantile(finite, 0.025)) if len(finite) else np.nan,
        "ci_high": float(np.quantile(finite, 0.975)) if len(finite) else np.nan,
        "probability_above_zero": float(np.mean(finite > 0)) if len(finite) else np.nan,
        "draws": draws,
        "event_clusters": event_count,
    }


def bootstrap_table(
    frame: pd.DataFrame,
    dimension: str,
    bucket_column: str | None,
    bucket_order: Sequence[object],
    global_events: Sequence[str],
    weights: BootstrapWeights,
    events_by_bucket: dict[object, Sequence[str]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bucket in bucket_order:
        events = list(events_by_bucket[bucket]) if events_by_bucket else list(global_events)
        event_index = pd.Index(events, name="event_id")
        sliced = frame if bucket_column is None else frame[frame[bucket_column].astype(str).eq(str(bucket))]
        aggregate = sliced.groupby(["event_id", "arm"], observed=True)[["stake", "pnl"]].sum()
        matrix = weights.get(events)

        arm_values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for arm in ARM_ORDER:
            if arm in aggregate.index.get_level_values("arm"):
                arm_group = aggregate.xs(arm, level="arm")
            else:
                arm_group = pd.DataFrame(columns=["stake", "pnl"])
            stake = arm_group["stake"].reindex(event_index, fill_value=0.0).to_numpy(float)
            pnl = arm_group["pnl"].reindex(event_index, fill_value=0.0).to_numpy(float)
            arm_values[arm] = (stake, pnl)

        for arm_a, arm_b in ARM_PAIRS:
            stake_a, pnl_a = arm_values[arm_a]
            stake_b, pnl_b = arm_values[arm_b]
            draw_pnl_a = matrix @ pnl_a
            draw_pnl_b = matrix @ pnl_b
            draw_stake_a = matrix @ stake_a
            draw_stake_b = matrix @ stake_b
            pnl_samples = draw_pnl_a - draw_pnl_b
            with np.errstate(divide="ignore", invalid="ignore"):
                roi_a = np.divide(
                    draw_pnl_a,
                    draw_stake_a,
                    out=np.full_like(draw_pnl_a, np.nan),
                    where=draw_stake_a > 0,
                )
                roi_b = np.divide(
                    draw_pnl_b,
                    draw_stake_b,
                    out=np.full_like(draw_pnl_b, np.nan),
                    where=draw_stake_b > 0,
                )
            roi_samples = roi_a - roi_b
            point_pnl = float(pnl_a.sum() - pnl_b.sum())
            roi_a_point = pnl_a.sum() / stake_a.sum() if stake_a.sum() > 0 else np.nan
            roi_b_point = pnl_b.sum() / stake_b.sum() if stake_b.sum() > 0 else np.nan
            rows.append(
                _bootstrap_metric_row(
                    dimension,
                    bucket,
                    arm_a,
                    arm_b,
                    "pnl_delta",
                    point_pnl,
                    pnl_samples,
                    len(events),
                    weights.draws,
                )
            )
            rows.append(
                _bootstrap_metric_row(
                    dimension,
                    bucket,
                    arm_a,
                    arm_b,
                    "roi_delta",
                    float(roi_a_point - roi_b_point),
                    roi_samples,
                    len(events),
                    weights.draws,
                )
            )
    return pd.DataFrame(rows)


def player_attribution(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    players = (
        frame[["selection_id", "selection_name"]]
        .drop_duplicates("selection_id", keep="last")
        .sort_values("selection_id")
    )
    for player in players.itertuples(index=False):
        player_frame = frame[frame["selection_id"].eq(player.selection_id)]
        for arm in ARM_ORDER:
            group = player_frame[player_frame["arm"].eq(arm)]
            stake = float(group["stake"].sum())
            pnl = float(group["pnl"].sum())
            rows.append(
                {
                    "selection_id": player.selection_id,
                    "selection_name": player.selection_name,
                    "arm": arm,
                    "bets": int(len(group)),
                    "events": int(group["event_id"].nunique()),
                    "stake": stake,
                    "pnl": pnl,
                    "roi": pnl / stake if stake > 0 else np.nan,
                    "mean_ev": float(group["ev_per_unit"].mean()) if len(group) else np.nan,
                    "short_accurate_bet_share": float(
                        group["strict_archetype"].eq("short_accurate").mean()
                    )
                    if len(group)
                    else np.nan,
                }
            )
    output = pd.DataFrame(rows)
    pivot = output.pivot(index="selection_id", columns="arm", values="pnl")
    for arm_a, arm_b in ARM_PAIRS:
        delta = pivot.get(arm_a, 0.0) - pivot.get(arm_b, 0.0)
        output = output.merge(
            delta.rename(f"pnl_delta__{arm_a}_minus_{arm_b}"),
            left_on="selection_id",
            right_index=True,
            how="left",
        )
    return output


def recurring_players(
    attribution: pd.DataFrame,
    min_bets: int,
    min_events: int,
) -> list[str]:
    reference = attribution[attribution["arm"].eq(ARM_FULL)]
    return sorted(
        reference.loc[
            reference["bets"].ge(min_bets) & reference["events"].ge(min_events),
            "selection_id",
        ].tolist()
    )


def _input_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".parquet", dir=path.parent)
    os.close(fd)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        Path(temporary).write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = frame[list(columns)].copy()
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    rows = []
    for row in selected.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append("" if not np.isfinite(value) else f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *rows])


def _compare_full_reference(
    probabilities: pd.DataFrame,
    reference_path: Path | None,
) -> dict[str, Any]:
    if reference_path is None or not reference_path.is_file():
        return {"status": "reference_unavailable"}
    reference = pd.read_parquet(reference_path)
    reference = reference[
        reference["market"].astype(str).str.casefold().eq("tournament_matchup")
        & reference["arm"].astype(str).str.casefold().eq("baseline")
    ].copy()
    reference["event_id"] = _event_key(reference)
    reference["selection_id"] = reference["selection_id"].map(_canonical_player_id)
    reference["opponent_id"] = reference["opponent_id"].map(_canonical_player_id)
    full = probabilities[probabilities["arm"].eq(ARM_FULL)].copy()
    reference = reference[reference["event_id"].isin(full["event_id"].unique())].copy()
    key = ["event_id", "selection_id", "opponent_id"]
    joined = full.merge(
        reference[key + ["probability", "push_probability"]],
        on=key,
        how="outer",
        suffixes=("_new", "_reference"),
        indicator=True,
    )
    matched = joined["_merge"].eq("both")
    probability_error = (
        joined.loc[matched, "probability_new"] - joined.loc[matched, "probability_reference"]
    ).abs()
    push_error = (
        joined.loc[matched, "push_probability_new"]
        - joined.loc[matched, "push_probability_reference"]
    ).abs()
    return {
        "status": "compared",
        "new_rows": int(len(full)),
        "reference_rows_in_new_events": int(len(reference)),
        "matched_rows": int(matched.sum()),
        "new_only_rows": int(joined["_merge"].eq("left_only").sum()),
        "reference_only_rows": int(joined["_merge"].eq("right_only").sum()),
        "max_probability_error": float(probability_error.max()) if len(probability_error) else np.nan,
        "max_push_error": float(push_error.max()) if len(push_error) else np.nan,
    }


def _manifest(output_dir: Path, paths: Iterable[Path]) -> dict[str, Any]:
    files = {}
    for path in sorted(paths, key=lambda item: item.name):
        files[path.name] = {"bytes": path.stat().st_size, "sha256": _input_sha256(path)}
    return {"contract": "course_profile_matchup_archetype_audit_v1", "files": files}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir = args.input_dir.resolve()
    bets_path = (args.bets or input_dir / "bets.parquet").resolve()
    probabilities_path = (args.probabilities or input_dir / "probabilities.parquet").resolve()
    candidate_path = args.candidate_bets.resolve() if args.candidate_bets else None
    default_profile_attribution = input_dir / "event_field_profile_attribution.parquet"
    profile_attribution_path = (
        args.profile_attribution.resolve()
        if args.profile_attribution
        else default_profile_attribution.resolve()
        if default_profile_attribution.is_file()
        else None
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else input_dir / "matchup_archetype_audit"
    )
    if not bets_path.is_file():
        raise FileNotFoundError(f"bets file not found: {bets_path}")
    if not probabilities_path.is_file():
        raise FileNotFoundError(f"probability file not found: {probabilities_path}")

    probabilities = normalize_probabilities(pd.read_parquet(probabilities_path))
    probability_menu_audit = validate_probability_arm_menus(probabilities)
    bets, bet_audit = normalize_bets(
        pd.read_parquet(bets_path), args.line_policy, args.sizing
    )
    candidate_bets = None
    candidate_audit = None
    if candidate_path is not None:
        candidate_bets, candidate_audit = normalize_bets(
            pd.read_parquet(candidate_path), args.line_policy, args.sizing
        )

    event_counts = probabilities.groupby("arm")["event_id"].nunique().to_dict()
    if len(set(event_counts.values())) != 1:
        raise ValueError(f"probability arms have unequal event counts: {event_counts}")
    event_universe = sorted(probabilities["event_id"].unique().tolist())
    event_meta = (
        probabilities[["event_id", "event_start"]]
        .drop_duplicates("event_id", keep="last")
        .set_index("event_id")
    )
    event_meta["year"] = event_meta["event_start"].dt.year

    roster = load_roster(args.roster, probabilities)
    history = _read_history(args.db, roster)
    profiles = build_driving_profiles(roster, history)
    bets = add_full_reference_edge(attach_profiles(bets, profiles), probabilities)
    if candidate_bets is not None:
        candidate_bets = add_full_reference_edge(
            attach_profiles(candidate_bets, profiles), probabilities
        )
    profile_attribution = None
    profile_gap_performance = None
    if profile_attribution_path is not None:
        if not profile_attribution_path.is_file():
            raise FileNotFoundError(
                f"profile-attribution file not found: {profile_attribution_path}"
            )
        profile_attribution = load_profile_attribution(
            profile_attribution_path, event_universe
        )
        bets = attach_profile_attribution(bets, profile_attribution)
        if candidate_bets is not None:
            candidate_bets = attach_profile_attribution(
                candidate_bets, profile_attribution
            )

    profile_source = candidate_bets if candidate_bets is not None else bets
    overall = performance_table(bets, "overall", None, ["all"])
    opportunities = probabilities.groupby("arm").size().rename("probability_opportunities")
    overall = overall.merge(opportunities, left_on="arm", right_index=True, how="left")
    strict = performance_table(
        bets, "strict_archetype", "strict_archetype", STRICT_ORDER
    )
    broad = performance_table(
        bets, "broad_archetype", "broad_archetype", BROAD_ORDER
    )
    native_edge = performance_table(
        profile_source, "ev_bucket_native", "ev_bucket_native", EV_ORDER
    )
    reference_edge = performance_table(
        profile_source,
        "ev_bucket_full_reference",
        "ev_bucket_full_reference",
        EV_ORDER,
    )
    if profile_attribution is not None:
        profile_gap_performance = performance_table(
            bets,
            "modeled_profile_gap",
            "profile_in_baseline_gap_bucket",
            PROFILE_GAP_ORDER,
        )
    years = sorted(int(year) for year in event_meta["year"].dropna().unique())
    bets["analysis_year"] = bets["event_id"].map(event_meta["year"]).astype("Int64")
    year_performance = performance_table(
        bets, "year", "analysis_year", years
    )

    attribution = player_attribution(bets)
    recurring_ids = recurring_players(
        attribution, args.min_recurring_bets, args.min_recurring_events
    )
    recurring_frame = bets[bets["selection_id"].isin(recurring_ids)].copy()
    recurring_performance = performance_table(
        recurring_frame,
        "recurring_player",
        "selection_id",
        recurring_ids,
    )
    recurring_names = (
        bets[["selection_id", "selection_name"]]
        .drop_duplicates("selection_id", keep="last")
        .set_index("selection_id")["selection_name"]
    )
    recurring_performance.insert(
        2,
        "selection_name",
        recurring_performance["bucket"].map(recurring_names),
    )
    named = bets.copy()
    named["prior_five_sensitivity"] = np.where(
        named["selection_id"].isin(PRIOR_FIVE), "prior_five_only", "exclude_prior_five"
    )
    named_performance = performance_table(
        named,
        "prior_five_sensitivity",
        "prior_five_sensitivity",
        ["prior_five_only", "exclude_prior_five"],
    )

    weights = BootstrapWeights(args.bootstrap, args.seed)
    bootstrap_parts = [
        bootstrap_table(bets, "overall", None, ["all"], event_universe, weights),
        bootstrap_table(
            bets,
            "strict_archetype",
            "strict_archetype",
            STRICT_ORDER,
            event_universe,
            weights,
        ),
        bootstrap_table(
            bets,
            "broad_archetype",
            "broad_archetype",
            BROAD_ORDER,
            event_universe,
            weights,
        ),
        bootstrap_table(
            profile_source,
            "ev_bucket_native",
            "ev_bucket_native",
            EV_ORDER,
            event_universe,
            weights,
        ),
        bootstrap_table(
            profile_source,
            "ev_bucket_full_reference",
            "ev_bucket_full_reference",
            EV_ORDER,
            event_universe,
            weights,
        ),
        bootstrap_table(
            recurring_frame,
            "recurring_player",
            "selection_id",
            recurring_ids,
            event_universe,
            weights,
        ),
        bootstrap_table(
            named,
            "prior_five_sensitivity",
            "prior_five_sensitivity",
            ["prior_five_only", "exclude_prior_five"],
            event_universe,
            weights,
        ),
    ]
    if profile_attribution is not None:
        bootstrap_parts.append(
            bootstrap_table(
                bets,
                "modeled_profile_gap",
                "profile_in_baseline_gap_bucket",
                PROFILE_GAP_ORDER,
                event_universe,
                weights,
            )
        )
    events_by_year = {
        year: event_meta.index[event_meta["year"].eq(year)].tolist() for year in years
    }
    bootstrap_parts.append(
        bootstrap_table(
            bets,
            "year",
            "analysis_year",
            years,
            event_universe,
            weights,
            events_by_bucket=events_by_year,
        )
    )
    bootstrap = pd.concat(bootstrap_parts, ignore_index=True)

    full_reference = (
        {"status": "skipped"}
        if args.skip_full_reference_check
        else _compare_full_reference(probabilities, args.reference_probabilities)
    )
    if full_reference.get("status") == "compared":
        if full_reference["new_only_rows"] or full_reference["reference_only_rows"]:
            raise ValueError(f"FULL/reference probability menus differ: {full_reference}")
        if max(
            full_reference.get("max_probability_error", math.inf),
            full_reference.get("max_push_error", math.inf),
        ) > 1e-12:
            raise ValueError(f"FULL arm does not reproduce v4 probabilities: {full_reference}")

    selection_coverage = float(bets["selection_prior_rounds"].notna().mean())
    opponent_coverage = float(bets["opponent_prior_rounds"].notna().mean())
    audit = {
        "contract": "course_profile_matchup_archetype_audit_v1",
        "research_only": True,
        "input_dir": str(input_dir),
        "bets": str(bets_path),
        "probabilities": str(probabilities_path),
        "candidate_bets": str(candidate_path) if candidate_path else None,
        "profile_attribution": (
            str(profile_attribution_path) if profile_attribution_path else None
        ),
        "roster": str(args.roster),
        "database": str(args.db),
        "line_policy": args.line_policy,
        "sizing": args.sizing,
        "event_count": len(event_universe),
        "event_universe": event_universe,
        "probability_event_counts_by_arm": event_counts,
        "probability_menu_audit": probability_menu_audit,
        "bet_normalization": bet_audit,
        "candidate_bet_normalization": candidate_audit,
        "bet_rows_by_arm": {key: int(value) for key, value in bets["arm"].value_counts().items()},
        "profile_rows": len(profiles),
        "profile_known_rows": int(profiles["prior_rounds"].ge(20).sum()),
        "selection_profile_join_coverage": selection_coverage,
        "opponent_profile_join_coverage": opponent_coverage,
        "modeled_profile_gap": {
            "available": profile_attribution is not None,
            "selection_join_coverage": (
                float(bets["selection_profile_in_baseline"].notna().mean())
                if profile_attribution is not None
                else None
            ),
            "opponent_join_coverage": (
                float(bets["opponent_profile_in_baseline"].notna().mean())
                if profile_attribution is not None
                else None
            ),
            "definition": (
                "selection profile_in_baseline minus opponent profile_in_baseline; "
                "fixed symmetric bins; exploratory diagnostic"
            ),
        },
        "strict_definition": {
            "short_accurate": "distance percentile <35 and accuracy percentile >=70",
            "long_wild": "distance percentile >=80 and accuracy percentile <40",
            "long_accurate": "distance percentile >=80 and accuracy percentile >=40",
            "minimum_prior_rounds": 20,
            "percentile_reference": "event field, including partial-data values",
        },
        "broad_definition": "event-field distance tercile x accuracy tercile",
        "bootstrap": {
            "draws": args.bootstrap,
            "seed": args.seed,
            "unit": "whole event",
            "zero_bet_events_retained": True,
            "comparison_order": [
                f"{arm_a}_minus_{arm_b}" for arm_a, arm_b in ARM_PAIRS
            ],
            "positive_delta_favors": "first named arm",
        },
        "recurring_definition": {
            "reference_arm": ARM_FULL,
            "minimum_bets": args.min_recurring_bets,
            "minimum_events": args.min_recurring_events,
            "players": len(recurring_ids),
        },
        "zero_to_three_complete": candidate_bets is not None,
        "ev_bucket_opportunity_set": (
            "candidate_bets" if candidate_bets is not None else "selected_bets_only"
        ),
        "full_reference_check": full_reference,
        "input_hashes": {
            "bets": _input_sha256(bets_path),
            "probabilities": _input_sha256(probabilities_path),
            "roster": _input_sha256(args.roster) if args.roster.is_file() else None,
            "database": _input_sha256(args.db),
            "candidate_bets": _input_sha256(candidate_path) if candidate_path else None,
            "profile_attribution": (
                _input_sha256(profile_attribution_path)
                if profile_attribution_path
                else None
            ),
        },
        "interpretation_warning": (
            "The short/accurate hypothesis was discovered on this evaluation sample; "
            "archetype slices are exploratory even though this postprocessor contract is frozen."
        ),
    }

    report = "\n".join(
        [
            "# Course-profile matchup archetype audit",
            "",
            "Research-only postprocessing of the FULL/HALF/NONE course-profile sensitivity.",
            "Whole-event bootstrap samples retain all eligible events, including zero-bet events.",
            (
                "Paired deltas are decision-oriented: HALF minus FULL, NONE minus FULL, "
                "and HALF minus NONE; positive values favor the first named arm."
            ),
            (
                "The 0-3% native-EV bucket is unavailable from the selected-bets file "
                "unless a separate all-candidate bet artifact is supplied."
            ),
            "",
            "## Overall production-style close performance",
            "",
            _markdown_table(
                overall,
                ["arm", "bets", "stake", "pnl", "roi", "mean_ev", "max_drawdown"],
            ),
            "",
            "## Strict archetype performance",
            "",
            _markdown_table(
                strict,
                ["bucket", "arm", "bets", "stake", "pnl", "roi", "mean_ev"],
            ),
            "",
            "## Audit caveat",
            "",
            audit["interpretation_warning"],
            "",
        ]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "event_player_driving_profiles.parquet": profiles,
        "matchup_bets_with_profiles.parquet": bets,
        "overall_performance.csv": overall,
        "strict_archetype_performance.csv": strict,
        "broad_archetype_performance.csv": broad,
        "ev_bucket_native_performance.csv": native_edge,
        "ev_bucket_full_reference_performance.csv": reference_edge,
        "year_performance.csv": year_performance,
        "player_attribution.csv": attribution,
        "recurring_player_performance.csv": recurring_performance,
        "prior_five_sensitivity.csv": named_performance,
        "paired_bootstrap.csv": bootstrap,
    }
    if profile_gap_performance is not None:
        outputs["modeled_profile_gap_performance.csv"] = profile_gap_performance
    written: list[Path] = []
    for name, frame in outputs.items():
        path = output_dir / name
        if name.endswith(".parquet"):
            _atomic_parquet(frame, path)
        else:
            _atomic_csv(frame, path)
        written.append(path)
    audit_path = output_dir / "audit.json"
    _atomic_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", audit_path)
    written.append(audit_path)
    report_path = output_dir / "MATCHUP_ARCHETYPE_AUDIT.md"
    _atomic_text(report, report_path)
    written.append(report_path)
    manifest_path = output_dir / "manifest.json"
    _atomic_text(json.dumps(_manifest(output_dir, written), indent=2, sort_keys=True) + "\n", manifest_path)

    print(f"[ok] {len(event_universe)} events; {len(bets):,} normalized bets")
    print(f"[ok] {len(profiles):,} event-player driving profiles")
    print(f"[ok] {args.bootstrap:,} paired whole-event bootstrap draws")
    print(f"[ok] outputs -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
