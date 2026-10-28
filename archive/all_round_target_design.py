"""Research-only all-round target construction for the base-skill rebuild.

This module deliberately does not participate in the weekly production pipeline.
It materializes two player-event labels from every observed R1-R4 round and a
round-level table suitable for the proposed MAR-RE4 likelihood:

``RAW-OBS4``
    The mean of all observed field-adjusted SG rounds in a player-event.

``SURVCOMP-OBS4``
    The same mean after subtracting, round by round, the strictly pre-event
    skill-composition difference between that round's participants and the R1
    entrant field.

The composition anchor is intentionally simple and candidate-independent.  It
uses only the player's earlier R1-R2 field-adjusted SG, an observation-recency
EWM with a 40-round half-life, and 20-round-equivalent shrinkage to zero.  No
legacy skill, market price, current-event result, or post-cut outcome enters it.

The public entry points are :func:`build_all_round_targets` and
:func:`build_mar_round_rows`.  Both accept an already-loaded frame so callers
can freeze their own sample before constructing targets.  The loader below
reads only the columns named in this research contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAYER_ROUND_PANEL = (
    REPO_ROOT
    / "permanent_data"
    / "research"
    / "base_skill_rebuild_20200101_20260822"
    / "player_round_panel.parquet"
)

OUTCOME_COLUMN = "dg_sg_total_field_adjusted"
EWM_HALF_LIFE_ROUNDS = 40.0
PRIOR_ROUND_EQUIVALENT = 20.0
PRIOR_MEAN = 0.0
TARGET_ROUNDS = (1, 2, 3, 4)
ANCHOR_SOURCE_ROUNDS = (1, 2)

REQUIRED_COLUMNS = (
    "event_key",
    "event_start_date",
    "dg_id",
    "round_num",
    "round_date",
    OUTCOME_COLUMN,
)
OPTIONAL_COLUMNS = ("tour", "year", "player_event_key")


def _stable_id(value: object) -> str | None:
    """Return a stable player-id string without turning nulls into ``"nan"``."""

    if pd.isna(value):
        return None
    text = str(value).strip()
    try:
        numeric = float(text)
    except (TypeError, ValueError):
        return text or None
    if np.isfinite(numeric) and numeric.is_integer():
        return str(int(numeric))
    return text or None


def load_player_round_panel(
    path: Path | str = DEFAULT_PLAYER_ROUND_PANEL,
    *,
    tour: str | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
) -> pd.DataFrame:
    """Load only the columns used by the all-round target contract.

    Filtering occurs after the narrow parquet read.  In particular, no
    ``legacy_*`` or market column is loaded.
    """

    panel_path = Path(path)
    available = set(pq.ParquetFile(panel_path).schema.names)
    missing = sorted(set(REQUIRED_COLUMNS) - available)
    if missing:
        raise ValueError(f"player-round panel missing required columns: {missing}")
    columns = list(REQUIRED_COLUMNS) + [
        column for column in OPTIONAL_COLUMNS if column in available
    ]
    frame = pd.read_parquet(panel_path, columns=columns)
    if tour is not None:
        if "tour" not in frame:
            raise ValueError("tour filter requested but panel has no tour column")
        frame = frame.loc[
            frame["tour"].astype("string").str.casefold().eq(str(tour).casefold())
        ]
    if start_year is not None or end_year is not None:
        if "year" in frame:
            years = pd.to_numeric(frame["year"], errors="coerce")
        else:
            years = pd.to_datetime(frame["event_start_date"], errors="coerce").dt.year
        if start_year is not None:
            frame = frame.loc[years.ge(int(start_year))]
            years = years.loc[frame.index]
        if end_year is not None:
            frame = frame.loc[years.le(int(end_year))]
    return frame.reset_index(drop=True)


def _prepare_rounds(rounds: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    missing = sorted(set(REQUIRED_COLUMNS) - set(rounds.columns))
    if missing:
        raise ValueError(f"round frame missing required columns: {missing}")

    # Copy only contract columns.  This is also a mechanical guarantee that a
    # caller-supplied legacy or market column cannot influence either target.
    keep = list(REQUIRED_COLUMNS) + [
        column for column in OPTIONAL_COLUMNS if column in rounds.columns
    ]
    frame = rounds.loc[:, keep].copy()
    input_rows = int(len(frame))
    frame["event_key"] = frame["event_key"].astype("string").str.strip()
    frame["dg_id_key"] = frame["dg_id"].map(_stable_id).astype("string")
    frame["event_start_date"] = pd.to_datetime(
        frame["event_start_date"], errors="coerce"
    ).dt.normalize()
    frame["round_date"] = pd.to_datetime(
        frame["round_date"], errors="coerce"
    ).dt.normalize()
    frame["round_num"] = pd.to_numeric(frame["round_num"], errors="coerce")
    frame["raw_y"] = pd.to_numeric(frame[OUTCOME_COLUMN], errors="coerce")

    identity_ok = (
        frame["event_key"].notna()
        & frame["event_key"].ne("")
        & frame["dg_id_key"].notna()
        & frame["event_start_date"].notna()
        & frame["round_date"].notna()
    )
    round_ok = frame["round_num"].isin(TARGET_ROUNDS)
    finite_outcome = np.isfinite(frame["raw_y"].to_numpy(float))

    identified = frame.loc[identity_ok & round_ok].copy()
    duplicate_mask = identified.duplicated(
        ["event_key", "dg_id_key", "round_num"], keep=False
    )
    if duplicate_mask.any():
        sample = identified.loc[
            duplicate_mask, ["event_key", "dg_id_key", "round_num"]
        ].head(10)
        raise AssertionError(
            "duplicate player-event-round rows are not admissible:\n"
            + sample.to_string(index=False)
        )
    before_start = identified["round_date"].lt(identified["event_start_date"])
    if before_start.any():
        sample = identified.loc[
            before_start,
            ["event_key", "dg_id_key", "round_num", "event_start_date", "round_date"],
        ].head(10)
        raise AssertionError(
            "round dates before their own event start would violate the anchor boundary:\n"
            + sample.to_string(index=False)
        )

    frame = frame.loc[identity_ok & round_ok & finite_outcome].copy()
    frame["round_num"] = frame["round_num"].astype(int)
    frame = frame.sort_values(
        ["event_start_date", "event_key", "dg_id_key", "round_num"],
        kind="mergesort",
    ).reset_index(drop=True)

    audit = {
        "input_rows": input_rows,
        "valid_round_rows": int(len(frame)),
        "excluded_bad_identity_rows": int((~identity_ok).sum()),
        "excluded_non_r1_r4_rows": int((identity_ok & ~round_ok).sum()),
        "excluded_missing_outcome_rows": int(
            (identity_ok & round_ok & ~finite_outcome).sum()
        ),
        "duplicate_player_event_round_rows": 0,
        "own_event_rounds_before_event_start": 0,
    }
    return frame, audit


def _unique_player_events(frame: pd.DataFrame) -> pd.DataFrame:
    consistency = frame.groupby(
        ["event_key", "dg_id_key"], sort=False, observed=True
    )["event_start_date"].nunique(dropna=False)
    if consistency.gt(1).any():
        raise AssertionError("player-event rows disagree on event_start_date")
    columns = ["event_key", "event_start_date", "dg_id_key", "dg_id"]
    return (
        frame[columns]
        .drop_duplicates(["event_key", "dg_id_key"])
        .sort_values(
            ["event_start_date", "event_key", "dg_id_key"], kind="mergesort"
        )
        .reset_index(drop=True)
    )


def _build_strict_prior_anchors(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the frozen EWM40 + 20-round-shrink anchor for each player-event."""

    queries = _unique_player_events(frame)
    sources = frame.loc[frame["round_num"].isin(ANCHOR_SOURCE_ROUNDS)].copy()
    sources = sources.sort_values(
        ["dg_id_key", "round_date", "event_key", "round_num"], kind="mergesort"
    )
    source_groups = {
        str(key): positions
        for key, positions in sources.groupby(
            "dg_id_key", sort=False, observed=True
        ).indices.items()
    }
    query_groups = queries.groupby("dg_id_key", sort=False, observed=True).indices
    decay = float(np.exp2(-1.0 / EWM_HALF_LIFE_ROUNDS))
    output: list[dict[str, Any]] = [dict() for _ in range(len(queries))]

    for dg_id_key, query_positions_raw in query_groups.items():
        query_positions = np.asarray(query_positions_raw, dtype=int)
        source_positions = source_groups.get(str(dg_id_key))
        history = (
            sources.iloc[np.asarray(source_positions, dtype=int)]
            if source_positions is not None
            else sources.iloc[:0]
        )
        history = history.sort_values(
            ["round_date", "event_key", "round_num"], kind="mergesort"
        )
        dates = history["round_date"].to_numpy(dtype="datetime64[ns]")
        values = history["raw_y"].to_numpy(float)

        # Prefix recurrences reproduce weights 1, decay, decay**2, ... from
        # newest to oldest while making every as-of query O(log n).
        weighted_sum = np.zeros(len(values) + 1, dtype=float)
        weight_sum = np.zeros(len(values) + 1, dtype=float)
        squared_weight_sum = np.zeros(len(values) + 1, dtype=float)
        for index, value in enumerate(values, start=1):
            weighted_sum[index] = value + decay * weighted_sum[index - 1]
            weight_sum[index] = 1.0 + decay * weight_sum[index - 1]
            squared_weight_sum[index] = 1.0 + decay * decay * squared_weight_sum[index - 1]

        for position in query_positions:
            target = np.datetime64(queries.at[position, "event_start_date"])
            stop = int(np.searchsorted(dates, target, side="left"))
            if stop:
                ewm = float(weighted_sum[stop] / weight_sum[stop])
                ess = float(weight_sum[stop] ** 2 / squared_weight_sum[stop])
                shrink_weight = float(ess / (ess + PRIOR_ROUND_EQUIVALENT))
                anchor = float(
                    shrink_weight * ewm + (1.0 - shrink_weight) * PRIOR_MEAN
                )
                max_source = pd.Timestamp(dates[stop - 1])
            else:
                ewm = np.nan
                ess = 0.0
                shrink_weight = 0.0
                anchor = PRIOR_MEAN
                max_source = pd.NaT
            output[position] = {
                "strict_prior_anchor": anchor,
                "anchor_history_count": stop,
                "anchor_ess": ess,
                "anchor_ewm40": ewm,
                "anchor_shrink_weight": shrink_weight,
                "anchor_max_source_round_date": max_source,
            }

    anchors = pd.concat([queries, pd.DataFrame(output)], axis=1)
    source_date = pd.to_datetime(anchors["anchor_max_source_round_date"], errors="coerce")
    target_date = pd.to_datetime(anchors["event_start_date"], errors="coerce")
    violations = source_date.notna() & source_date.ge(target_date)
    if violations.any():
        raise AssertionError("strict-prior anchor source reaches target event start")
    audit = {
        "anchor_method": "r1_r2_observation_ewm40_eb20_to_zero",
        "anchor_source_rounds": list(ANCHOR_SOURCE_ROUNDS),
        "anchor_half_life_rounds": EWM_HALF_LIFE_ROUNDS,
        "anchor_prior_round_equivalent": PRIOR_ROUND_EQUIVALENT,
        "anchor_prior_mean": PRIOR_MEAN,
        "anchor_player_event_rows": int(len(anchors)),
        "anchor_cold_start_rows": int(anchors["anchor_history_count"].eq(0).sum()),
        "anchor_strict_prior_violations": int(violations.sum()),
        "anchor_max_history_count": int(anchors["anchor_history_count"].max()),
        "legacy_columns_read": [],
        "market_columns_read": [],
    }
    return anchors, audit


def _build_round_design(
    rounds: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame, preparation_audit = _prepare_rounds(rounds)
    if frame.empty:
        raise ValueError("no finite R1-R4 field-adjusted SG rows remain")
    anchors, anchor_audit = _build_strict_prior_anchors(frame)
    anchor_columns = [
        "event_key",
        "dg_id_key",
        "strict_prior_anchor",
        "anchor_history_count",
        "anchor_ess",
        "anchor_ewm40",
        "anchor_shrink_weight",
        "anchor_max_source_round_date",
    ]
    frame = frame.merge(
        anchors[anchor_columns],
        on=["event_key", "dg_id_key"],
        how="left",
        validate="many_to_one",
    )
    if frame["strict_prior_anchor"].isna().any():
        raise AssertionError("strict-prior anchor merge left target rows unmatched")

    event_columns = ["event_key", "event_start_date"]
    entrants = frame.loc[frame["round_num"].eq(1)].copy()
    entrant_summary = (
        entrants.groupby(event_columns, as_index=False, observed=True)
        .agg(
            entrant_anchor_mean=("strict_prior_anchor", "mean"),
            entrant_count=("dg_id_key", "nunique"),
        )
    )
    participant_summary = (
        frame.groupby(event_columns + ["round_num"], as_index=False, observed=True)
        .agg(
            participant_anchor_mean=("strict_prior_anchor", "mean"),
            participant_count=("dg_id_key", "nunique"),
        )
    )
    frame = frame.merge(
        entrant_summary, on=event_columns, how="left", validate="many_to_one"
    ).merge(
        participant_summary,
        on=event_columns + ["round_num"],
        how="left",
        validate="many_to_one",
    )
    frame["composition_shift"] = (
        frame["participant_anchor_mean"] - frame["entrant_anchor_mean"]
    )
    # The two participant sets are identical in R1.  Force the algebraic zero
    # so serialized targets do not carry floating-point noise.
    frame.loc[frame["round_num"].eq(1), "composition_shift"] = 0.0
    frame["survcomp_y"] = frame["raw_y"] - frame["composition_shift"]

    missing_entrant_events = entrant_summary["event_key"].nunique()
    all_valid_events = frame["event_key"].nunique()
    shift_by_round: dict[str, dict[str, float | int | None]] = {}
    for round_num in TARGET_ROUNDS:
        values = frame.loc[
            frame["round_num"].eq(round_num), "composition_shift"
        ].dropna()
        shift_by_round[str(round_num)] = {
            "rows": int(len(values)),
            "mean": float(values.mean()) if len(values) else None,
            "min": float(values.min()) if len(values) else None,
            "max": float(values.max()) if len(values) else None,
        }
    r1_shift = frame.loc[frame["round_num"].eq(1), "composition_shift"].abs()
    audit = {
        **preparation_audit,
        **anchor_audit,
        "valid_event_count": int(all_valid_events),
        "events_with_valid_r1_entrant_field": int(missing_entrant_events),
        "events_without_valid_r1_entrant_field": int(
            all_valid_events - missing_entrant_events
        ),
        "round_rows_missing_composition_shift": int(
            frame["composition_shift"].isna().sum()
        ),
        "max_abs_r1_composition_shift": (
            float(r1_shift.max()) if len(r1_shift) else None
        ),
        "composition_shift_by_round": shift_by_round,
        "composition_reference": "valid_r1_entrant_anchor_mean",
        "composition_participants": "valid_adjusted_sg_participants_by_event_round",
    }
    return frame, audit


def _count_distribution(values: pd.Series) -> dict[str, int]:
    counts = pd.to_numeric(values, errors="coerce").fillna(0).astype(int)
    return {str(key): int(value) for key, value in counts.value_counts().sort_index().items()}


def build_all_round_targets(
    rounds: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the frozen RAW-OBS4 and SURVCOMP-OBS4 player-event labels.

    ``*_weight`` is the label precision weight ``observed_count / 4``.  The
    output retains one-round events so callers can run the declared WD/DQ
    sensitivity; the headline model should filter ``*_count >= 2``.
    """

    frame, audit = _build_round_design(rounds)
    keys = ["event_key", "event_start_date", "dg_id_key"]
    targets = (
        frame.groupby(keys, as_index=False, observed=True)
        .agg(
            dg_id=("dg_id", "first"),
            raw_obs4_target=("raw_y", "mean"),
            raw_obs4_count=("raw_y", "count"),
            survcomp_obs4_target=("survcomp_y", "mean"),
            survcomp_obs4_count=("survcomp_y", "count"),
            strict_prior_anchor=("strict_prior_anchor", "first"),
            anchor_history_count=("anchor_history_count", "first"),
            anchor_ess=("anchor_ess", "first"),
            anchor_max_source_round_date=("anchor_max_source_round_date", "first"),
        )
        .sort_values(keys, kind="mergesort")
        .reset_index(drop=True)
    )
    targets["raw_obs4_weight"] = targets["raw_obs4_count"] / 4.0
    targets["survcomp_obs4_weight"] = targets["survcomp_obs4_count"] / 4.0
    if targets[["raw_obs4_count", "survcomp_obs4_count"]].gt(4).any().any():
        raise AssertionError("an all-round target contains more than four observations")

    target_audit = {
        **audit,
        "target_schema_version": "all_round_targets_v1",
        "target_player_event_rows": int(len(targets)),
        "raw_obs4_count_distribution": _count_distribution(targets["raw_obs4_count"]),
        "survcomp_obs4_count_distribution": _count_distribution(
            targets["survcomp_obs4_count"]
        ),
        "raw_obs4_headline_rows_n2_plus": int(targets["raw_obs4_count"].ge(2).sum()),
        "survcomp_obs4_headline_rows_n2_plus": int(
            targets["survcomp_obs4_count"].ge(2).sum()
        ),
        "raw_obs4_nonstandard_n1_or_n3_rows": int(
            targets["raw_obs4_count"].isin([1, 3]).sum()
        ),
        "survcomp_obs4_nonstandard_n1_or_n3_rows": int(
            targets["survcomp_obs4_count"].isin([1, 3]).sum()
        ),
        "target_weight_formula": "observed_count / 4",
    }
    return targets, target_audit


def build_mar_round_rows(
    rounds: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return observed R1-R4 rows for a MAR-RE4 mixed/GLS likelihood.

    The caller should repeat its strict pre-event feature vector on these rows
    and use ``event_round_key`` as the nuisance fixed-effect group and
    ``mar_event_player_key`` as the random-effect/covariance block.  Both raw
    and composition-corrected outcomes are retained.  Centered columns are
    supplied for a fixed-effect-free within-event-round implementation.
    """

    frame, audit = _build_round_design(rounds)
    event_round_keys = ["event_key", "event_start_date", "round_num"]
    frame["event_round_raw_mean"] = frame.groupby(
        event_round_keys, observed=True
    )["raw_y"].transform("mean")
    frame["event_round_survcomp_mean"] = frame.groupby(
        event_round_keys, observed=True
    )["survcomp_y"].transform("mean")
    frame["raw_y_event_round_centered"] = (
        frame["raw_y"] - frame["event_round_raw_mean"]
    )
    frame["survcomp_y_event_round_centered"] = (
        frame["survcomp_y"] - frame["event_round_survcomp_mean"]
    )
    player_keys = ["event_key", "dg_id_key"]
    frame["player_event_round_count"] = frame.groupby(
        player_keys, observed=True
    )["raw_y"].transform("count")
    frame["player_event_weight"] = frame["player_event_round_count"] / 4.0
    frame["nonstandard_completion"] = ~frame["player_event_round_count"].isin([2, 4])
    frame["event_round_key"] = (
        frame["event_key"].astype(str) + "|r" + frame["round_num"].astype(str)
    )
    frame["mar_event_player_key"] = (
        frame["event_key"].astype(str) + "|" + frame["dg_id_key"].astype(str)
    )

    columns = [
        "event_key",
        "event_start_date",
        "event_round_key",
        "mar_event_player_key",
        "dg_id_key",
        "dg_id",
        "round_num",
        "round_date",
        "raw_y",
        "survcomp_y",
        "raw_y_event_round_centered",
        "survcomp_y_event_round_centered",
        "event_round_raw_mean",
        "event_round_survcomp_mean",
        "strict_prior_anchor",
        "anchor_history_count",
        "anchor_ess",
        "anchor_max_source_round_date",
        "entrant_anchor_mean",
        "participant_anchor_mean",
        "entrant_count",
        "participant_count",
        "composition_shift",
        "player_event_round_count",
        "player_event_weight",
        "nonstandard_completion",
    ]
    output = frame[columns].sort_values(
        ["event_start_date", "event_key", "dg_id_key", "round_num"],
        kind="mergesort",
    ).reset_index(drop=True)
    centered_means = output.groupby("event_round_key", observed=True)[
        ["raw_y_event_round_centered", "survcomp_y_event_round_centered"]
    ].mean()
    max_center_error = float(centered_means.abs().to_numpy().max())
    mar_audit = {
        **audit,
        "mar_schema_version": "mar_re4_round_rows_v1",
        "mar_round_rows": int(len(output)),
        "mar_player_event_blocks": int(output["mar_event_player_key"].nunique()),
        "mar_event_round_groups": int(output["event_round_key"].nunique()),
        "mar_nonstandard_completion_rows": int(output["nonstandard_completion"].sum()),
        "mar_max_abs_event_round_centered_mean": max_center_error,
        "mar_random_effect_group": "event_key + dg_id_key",
        "mar_nuisance_group": "event_key + round_num",
    }
    return output, mar_audit


__all__ = [
    "DEFAULT_PLAYER_ROUND_PANEL",
    "EWM_HALF_LIFE_ROUNDS",
    "PRIOR_ROUND_EQUIVALENT",
    "PRIOR_MEAN",
    "build_all_round_targets",
    "build_mar_round_rows",
    "load_player_round_panel",
]
