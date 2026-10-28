"""Research-only strict-prior normalization for shot-transition donor atoms.

The eventual hole simulator needs historical transition atoms without silently
embedding the strength and spatial style of the historical field.  This module
implements that boundary.  It deliberately does not fit a production model or
write any weekly-pipeline files.

The normalization is conditional on ``category x state x setup`` and has three
modes:

``raw``
    Keep the observed local endpoint and its labels.
``event_centered``
    Remove the strictly-prior average ability/style of the donor event field.
``player_state_residualized``
    Remove the event field and the donor's remaining player-specific
    ability/style.  This is algebraically the same as removing the player's
    complete pre-event profile, but the two components remain explicit for
    auditability.

Spatial transforms never copy an old lie/hazard label onto a moved endpoint.
A caller-provided geometry classifier can atomically reclassify the moved
point.  Otherwise transformed neutral atoms are marked as having invalidated
labels.  Target TEE reapplication always requires a classifier.  APP may use
an explicit deferred mode because its miss vector cannot become a physical
endpoint until the recursive simulator supplies the generated start and shot
direction; such rows fail the physical-transition consumption guard.

Ability is removed and reapplied through coherent whole-atom importance
weights based on a fixed-reference transition reward.  The reward is never
subtracted from an endpoint while leaving its state unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd


GeometryClassifier = Callable[[pd.DataFrame], pd.DataFrame]
RewardFunction = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class NormalizationConfig:
    """Column contract and numerical settings for normalization."""

    event_col: str = "event_key"
    event_start_col: str = "event_start"
    player_col: str = "source_player_id"
    category_col: str = "sg_category"
    state_col: str = "state_key"
    setup_col: str = "setup_key"
    long_col: str = "long_yd"
    lat_col: str = "lat_yd"
    lie_col: str = "end_lie"
    hazard_col: str = "end_hazard"
    reward_col: str = "sg_raw"

    cutoff_col: str = "profile_cutoff"
    reference_long_col: str = "reference_long_yd"
    reference_lat_col: str = "reference_lat_yd"
    ability_col: str = "ability_sg"
    distance_ratio_col: str = "distance_ratio"
    long_bias_col: str = "long_bias_yd"
    lat_bias_col: str = "lat_bias_yd"
    long_scale_col: str = "long_scale"
    lat_scale_col: str = "lat_scale"
    correlation_col: str = "long_lat_corr"
    exposure_weight_col: str = "exposure_weight"
    source_col: str = "normalization_source"
    snapshot_kind_col: str = "snapshot_kind"

    model_long_col: str = "model_long_yd"
    model_lat_col: str = "model_lat_yd"
    model_lie_col: str = "model_end_lie"
    model_hazard_col: str = "model_end_hazard"
    model_reward_col: str = "model_sg"

    # Maps an ability estimate expressed in SG/round to the natural parameter
    # used for atom reweighting.  This is a research hyperparameter and must be
    # tuned out of sample; making it explicit prevents a hidden unit mismatch.
    # Disabled by default.  A chronological calibration must estimate this
    # SG/round -> per-transition natural-parameter map before real-data
    # ability weighting is defensible.
    ability_log_weight_per_sg2: float = 0.0
    max_abs_log_weight: float = 30.0
    reference_tolerance_yd: float = 1e-8

    @property
    def condition_cols(self) -> list[str]:
        return [self.category_col, self.state_col, self.setup_col]

    @property
    def field_group_cols(self) -> list[str]:
        return [self.event_col, *self.condition_cols]

    @property
    def profile_key_cols(self) -> list[str]:
        return [self.event_col, self.player_col, *self.condition_cols]


PROFILE_NUMERIC_DEFAULTS = {
    "ability": 0.0,
    "distance_ratio": 1.0,
    "long_bias": 0.0,
    "lat_bias": 0.0,
    "long_scale": 1.0,
    "lat_scale": 1.0,
    "correlation": 0.0,
    "exposure_weight": 1.0,
}


def _required(columns: Iterable[str], frame: pd.DataFrame, label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _parse_dates(series: pd.Series, label: str) -> pd.Series:
    values = pd.to_datetime(series, errors="coerce", utc=True)
    if values.isna().any():
        examples = series[values.isna()].astype(str).head(3).tolist()
        raise ValueError(f"{label} contains invalid dates: {examples}")
    return values


def _stable_frame_hash(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    selected = list(dict.fromkeys(columns))
    _required(selected, frame, "hash input")
    work = frame[selected].copy()
    schema = [(column, str(work[column].dtype)) for column in selected]
    row_hashes = pd.util.hash_pandas_object(work, index=False).to_numpy("uint64")
    row_hashes.sort()
    digest = sha256()
    digest.update(json.dumps(schema, separators=(",", ":")).encode("utf-8"))
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def _code_hash() -> str:
    try:
        return sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        return "unavailable"


def _metadata_hash(metadata: dict[str, object]) -> str:
    payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(payload.encode("utf-8")).hexdigest()


def latest_strict_prior_profiles(
    roster: pd.DataFrame,
    history: pd.DataFrame,
    *,
    event_col: str = "event_key",
    event_start_col: str = "event_start",
    player_col: str = "source_player_id",
    history_player_col: str | None = None,
    history_date_col: str = "Date",
    output_cutoff_col: str = "profile_cutoff",
    output_snapshot_kind_col: str = "snapshot_kind",
) -> pd.DataFrame:
    """Attach the latest player record strictly before each event start.

    The function is suitable for ``model_results2`` after the caller maps its
    player identifier (normally ``dg_id``) to ``player_col``.  Exact-date
    records are intentionally ineligible, so concurrent/held-out event results
    can never enter their own snapshot.
    """

    history_player_col = history_player_col or player_col
    _required([event_col, event_start_col, player_col], roster, "roster")
    _required([history_player_col, history_date_col], history, "history")
    if roster.duplicated([event_col, player_col]).any():
        raise ValueError("roster has duplicate event/player rows")

    left = roster.copy()
    left["__event_start"] = _parse_dates(left[event_start_col], event_start_col)
    left["__player_join"] = left[player_col].astype(str)
    left["__order"] = np.arange(len(left))
    right = history.copy()
    right["__profile_date"] = _parse_dates(right[history_date_col], history_date_col)
    right["__player_join"] = right[history_player_col].astype(str)
    right = right.drop(columns=[history_player_col], errors="ignore")

    # pandas requires global sort by the merge key, even when using ``by``.
    left_sorted = left.sort_values(["__event_start", "__player_join"])
    right_sorted = right.sort_values(["__profile_date", "__player_join"])
    merged = pd.merge_asof(
        left_sorted,
        right_sorted,
        left_on="__event_start",
        right_on="__profile_date",
        by="__player_join",
        direction="backward",
        allow_exact_matches=False,
    )
    if merged["__profile_date"].isna().any():
        sample = merged.loc[
            merged["__profile_date"].isna(), [event_col, player_col]
        ].head(5)
        raise ValueError(
            "No strictly-prior profile exists for roster rows: "
            f"{sample.to_dict('records')}"
        )
    if not (merged["__profile_date"] < merged["__event_start"]).all():
        raise AssertionError("strict-prior merge invariant failed")
    merged[output_cutoff_col] = merged["__profile_date"]
    merged[output_snapshot_kind_col] = "strict_prior_asof"
    return (
        merged.sort_values("__order")
        .drop(columns=["__event_start", "__player_join", "__order", "__profile_date"])
        .reset_index(drop=True)
    )


def _profile_columns(config: NormalizationConfig) -> list[str]:
    return [
        *config.profile_key_cols,
        config.cutoff_col,
        config.reference_long_col,
        config.reference_lat_col,
        config.ability_col,
        config.distance_ratio_col,
        config.long_bias_col,
        config.lat_bias_col,
        config.long_scale_col,
        config.lat_scale_col,
        config.correlation_col,
        config.exposure_weight_col,
        config.source_col,
        config.snapshot_kind_col,
    ]


def _prepare_profiles(
    profiles: pd.DataFrame,
    event_starts: pd.DataFrame,
    config: NormalizationConfig,
    *,
    allowed_snapshot_kinds: set[str],
) -> pd.DataFrame:
    required = [
        *config.profile_key_cols,
        config.cutoff_col,
        config.reference_long_col,
        config.reference_lat_col,
        config.snapshot_kind_col,
    ]
    _required(required, profiles, "profile_snapshot")
    if profiles.duplicated(config.profile_key_cols).any():
        sample = profiles.loc[
            profiles.duplicated(config.profile_key_cols, keep=False),
            config.profile_key_cols,
        ].head(5)
        raise ValueError(f"profile_snapshot has duplicate keys: {sample.to_dict('records')}")

    # The event-start authority is the donor/target event contract, not a
    # possibly stale copy bundled into a profile export.
    out = profiles.drop(columns=[config.event_start_col], errors="ignore").copy()
    optional = {
        config.ability_col: PROFILE_NUMERIC_DEFAULTS["ability"],
        config.distance_ratio_col: PROFILE_NUMERIC_DEFAULTS["distance_ratio"],
        config.long_bias_col: PROFILE_NUMERIC_DEFAULTS["long_bias"],
        config.lat_bias_col: PROFILE_NUMERIC_DEFAULTS["lat_bias"],
        config.long_scale_col: PROFILE_NUMERIC_DEFAULTS["long_scale"],
        config.lat_scale_col: PROFILE_NUMERIC_DEFAULTS["lat_scale"],
        config.correlation_col: PROFILE_NUMERIC_DEFAULTS["correlation"],
        config.exposure_weight_col: PROFILE_NUMERIC_DEFAULTS["exposure_weight"],
        config.source_col: "unspecified_strict_prior_snapshot",
    }
    for column, default in optional.items():
        if column not in out:
            out[column] = default
    kinds = out[config.snapshot_kind_col].astype(str)
    invalid_kinds = sorted(set(kinds) - allowed_snapshot_kinds)
    if invalid_kinds:
        raise ValueError(
            f"snapshot kinds {invalid_kinds} are not allowed here; allowed="
            f"{sorted(allowed_snapshot_kinds)}"
        )

    numeric = [
        config.reference_long_col,
        config.reference_lat_col,
        config.ability_col,
        config.distance_ratio_col,
        config.long_bias_col,
        config.lat_bias_col,
        config.long_scale_col,
        config.lat_scale_col,
        config.correlation_col,
        config.exposure_weight_col,
    ]
    for column in numeric:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if out[numeric].isna().any().any() or not np.isfinite(out[numeric]).all().all():
        raise ValueError("profile_snapshot contains non-finite numeric modifiers")
    if (out[[config.distance_ratio_col, config.long_scale_col, config.lat_scale_col]] <= 0).any().any():
        raise ValueError("distance ratios and spatial scales must be positive")
    if (out[config.correlation_col].abs() >= 0.999).any():
        raise ValueError("absolute long/lat correlation must be below 0.999")
    if (out[config.exposure_weight_col] <= 0).any():
        raise ValueError("pre-event exposure weights must be positive")

    starts = event_starts[[config.event_col, config.event_start_col]].drop_duplicates()
    if starts.duplicated(config.event_col).any():
        raise ValueError("an event has multiple event_start values")
    out = out.merge(starts, on=config.event_col, how="left", validate="many_to_one")
    if out[config.event_start_col].isna().any():
        raise ValueError("profile_snapshot contains events absent from event starts")
    out[config.cutoff_col] = _parse_dates(out[config.cutoff_col], config.cutoff_col)
    out[config.event_start_col] = _parse_dates(
        out[config.event_start_col], config.event_start_col
    )
    invalid = out[config.cutoff_col] >= out[config.event_start_col]
    if invalid.any():
        sample = out.loc[
            invalid, [config.event_col, config.player_col, config.cutoff_col, config.event_start_col]
        ].head(5)
        raise ValueError(
            "Profile snapshots must be strictly before event start; invalid rows: "
            f"{sample.to_dict('records')}"
        )
    return out


def _weighted_group_profiles(
    profiles: pd.DataFrame, config: NormalizationConfig
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    weight_col = config.exposure_weight_col
    for keys, group in profiles.groupby(config.field_group_cols, sort=False, dropna=False):
        weights = group[weight_col].to_numpy(float)
        weights = weights / weights.sum()
        ref_long = group[config.reference_long_col].to_numpy(float)
        ref_lat = group[config.reference_lat_col].to_numpy(float)
        if np.ptp(ref_long) > config.reference_tolerance_yd or np.ptp(ref_lat) > config.reference_tolerance_yd:
            raise ValueError(
                "Reference centers differ within event/category/state/setup; "
                f"group={keys}"
            )

        def avg(column: str) -> float:
            return float(np.dot(weights, group[column].to_numpy(float)))

        # Positive quantities combine on the log scale.  Correlation uses a
        # Fisher transform so the aggregate remains inside (-1, 1).
        field_distance_ratio = float(
            np.exp(np.dot(weights, np.log(group[config.distance_ratio_col].to_numpy(float))))
        )
        field_long_scale = float(
            np.exp(np.dot(weights, np.log(group[config.long_scale_col].to_numpy(float))))
        )
        field_lat_scale = float(
            np.exp(np.dot(weights, np.log(group[config.lat_scale_col].to_numpy(float))))
        )
        field_corr = float(
            np.tanh(np.dot(weights, np.arctanh(group[config.correlation_col].to_numpy(float))))
        )
        row = dict(zip(config.field_group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update(
            {
                "field_ability_sg": avg(config.ability_col),
                "field_distance_ratio": field_distance_ratio,
                "field_long_bias_yd": avg(config.long_bias_col),
                "field_lat_bias_yd": avg(config.lat_bias_col),
                "field_long_scale": field_long_scale,
                "field_lat_scale": field_lat_scale,
                "field_long_lat_corr": field_corr,
                "field_profile_players": int(group[config.player_col].nunique()),
                "field_profile_weight": float(group[weight_col].sum()),
                "field_profile_max_cutoff": group[config.cutoff_col].max(),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _cholesky(long_scale: float, lat_scale: float, corr: float) -> np.ndarray:
    return np.array(
        [
            [long_scale, 0.0],
            [corr * lat_scale, lat_scale * np.sqrt(max(1.0 - corr * corr, 1e-12))],
        ],
        dtype=float,
    )


def _player_shift(row: pd.Series, config: NormalizationConfig) -> np.ndarray:
    reference_long = float(row[config.reference_long_col])
    return np.array(
        [
            reference_long * (float(row[config.distance_ratio_col]) - 1.0)
            + float(row[config.long_bias_col]),
            float(row[config.lat_bias_col]),
        ]
    )


def _field_shift(row: pd.Series, config: NormalizationConfig) -> np.ndarray:
    reference_long = float(row[config.reference_long_col])
    return np.array(
        [
            reference_long * (float(row["field_distance_ratio"]) - 1.0)
            + float(row["field_long_bias_yd"]),
            float(row["field_lat_bias_yd"]),
        ]
    )


def _transform_row_to_neutral(
    row: pd.Series, mode: str, config: NormalizationConfig
) -> tuple[np.ndarray, np.ndarray]:
    center = np.array(
        [float(row[config.reference_long_col]), float(row[config.reference_lat_col])]
    )
    point = np.array([float(row[config.long_col]), float(row[config.lat_col])])
    if mode == "raw":
        transformed = point
    else:
        player_matrix = _cholesky(
            float(row[config.long_scale_col]),
            float(row[config.lat_scale_col]),
            float(row[config.correlation_col]),
        )
        field_matrix = _cholesky(
            float(row["field_long_scale"]),
            float(row["field_lat_scale"]),
            float(row["field_long_lat_corr"]),
        )
        field_shift = _field_shift(row, config)
        event_residual = np.linalg.solve(field_matrix, point - center - field_shift)
        if mode == "event_centered":
            transformed = center + event_residual
        else:
            player_shift = _player_shift(row, config)
            relative_matrix = np.linalg.solve(field_matrix, player_matrix)
            relative_shift = np.linalg.solve(field_matrix, player_shift - field_shift)
            neutral_residual = np.linalg.solve(
                relative_matrix, event_residual - relative_shift
            )
            transformed = center + neutral_residual
    return transformed, transformed - center


def _inverse_lower_2d(
    x: np.ndarray,
    y: np.ndarray,
    long_scale: np.ndarray,
    lat_scale: np.ndarray,
    corr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized inverse of the lower-triangular spatial transform."""

    diag_y = lat_scale * np.sqrt(np.maximum(1.0 - corr * corr, 1e-12))
    solved_x = x / long_scale
    solved_y = (y - corr * lat_scale * solved_x) / diag_y
    return solved_x, solved_y


def _apply_lower_2d(
    x: np.ndarray,
    y: np.ndarray,
    long_scale: np.ndarray,
    lat_scale: np.ndarray,
    corr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized application of the lower-triangular spatial transform."""

    transformed_x = long_scale * x
    transformed_y = (
        corr * lat_scale * x
        + lat_scale * np.sqrt(np.maximum(1.0 - corr * corr, 1e-12)) * y
    )
    return transformed_x, transformed_y


def _transform_rows_to_neutral_vectorized(
    frame: pd.DataFrame, mode: str, config: NormalizationConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Batched equivalent of :func:`_transform_row_to_neutral`."""

    center_long = frame[config.reference_long_col].to_numpy(float)
    center_lat = frame[config.reference_lat_col].to_numpy(float)
    raw_long = frame[config.long_col].to_numpy(float)
    raw_lat = frame[config.lat_col].to_numpy(float)
    if mode == "raw":
        model_long, model_lat = raw_long, raw_lat
    elif mode == "event_centered":
        field_shift_long = (
            center_long * (frame["field_distance_ratio"].to_numpy(float) - 1.0)
            + frame["field_long_bias_yd"].to_numpy(float)
        )
        field_shift_lat = frame["field_lat_bias_yd"].to_numpy(float)
        resid_long, resid_lat = _inverse_lower_2d(
            raw_long - center_long - field_shift_long,
            raw_lat - center_lat - field_shift_lat,
            frame["field_long_scale"].to_numpy(float),
            frame["field_lat_scale"].to_numpy(float),
            frame["field_long_lat_corr"].to_numpy(float),
        )
        model_long = center_long + resid_long
        model_lat = center_lat + resid_lat
    else:
        # Removing field then player-relative style is exactly the inverse of
        # the player's complete affine transform.  This closed form preserves
        # that decomposition in the audit columns without millions of 2x2
        # solves or Python row iteration.
        player_shift_long = (
            center_long
            * (frame[config.distance_ratio_col].to_numpy(float) - 1.0)
            + frame[config.long_bias_col].to_numpy(float)
        )
        player_shift_lat = frame[config.lat_bias_col].to_numpy(float)
        resid_long, resid_lat = _inverse_lower_2d(
            raw_long - center_long - player_shift_long,
            raw_lat - center_lat - player_shift_lat,
            frame[config.long_scale_col].to_numpy(float),
            frame[config.lat_scale_col].to_numpy(float),
            frame[config.correlation_col].to_numpy(float),
        )
        model_long = center_long + resid_long
        model_lat = center_lat + resid_lat
    points = np.column_stack([model_long, model_lat])
    residuals = np.column_stack(
        [model_long - center_long, model_lat - center_lat]
    )
    return points, residuals


def _apply_classifier(
    frame: pd.DataFrame,
    classifier: GeometryClassifier,
    config: NormalizationConfig,
) -> pd.DataFrame:
    classified = classifier(frame.copy())
    if not isinstance(classified, pd.DataFrame):
        raise TypeError("geometry_classifier must return a pandas DataFrame")
    _required(
        [config.model_lie_col, config.model_hazard_col],
        classified,
        "geometry_classifier output",
    )
    if len(classified) != len(frame):
        raise ValueError("geometry_classifier changed the row count")
    out = frame.copy()
    out[config.model_lie_col] = classified[config.model_lie_col].to_numpy()
    out[config.model_hazard_col] = classified[config.model_hazard_col].to_numpy()
    out["coherence_status"] = "geometry_reclassified"
    out["geometry_classification_status"] = "classified_in_supplied_frame"
    out["physical_transition_ready"] = True
    return out


def assert_physical_transition_ready(
    transitions: pd.DataFrame,
    *,
    config: NormalizationConfig | None = None,
) -> None:
    """Fail closed unless every row is a classified physical transition.

    Deferred APP rows are shot-relative cache atoms, not consumable transition
    states.  The recursive simulator must rotate/translate their miss vectors
    from each generated start, classify the resulting target-course endpoint,
    and only then set ``physical_transition_ready`` true.
    """

    config = config or NormalizationConfig()
    _required(
        [
            "geometry_classification_status",
            "physical_transition_ready",
            config.model_lie_col,
            config.model_hazard_col,
        ],
        transitions,
        "transitions",
    )
    ready = transitions["physical_transition_ready"].fillna(False).astype(bool)
    labels_present = transitions[
        [config.model_lie_col, config.model_hazard_col]
    ].notna().all(axis=1)
    classified = transitions["geometry_classification_status"].eq(
        "classified_in_target_frame"
    )
    invalid = ~(ready & labels_present & classified)
    if invalid.any():
        statuses = (
            transitions.loc[invalid, "geometry_classification_status"]
            .fillna("missing")
            .astype(str)
            .value_counts()
            .to_dict()
        )
        raise ValueError(
            "transition consumption requires batched target-frame endpoint "
            "classification; unresolved rows="
            f"{int(invalid.sum())}, statuses={statuses}"
        )


def _normalized_weights(
    log_weights: pd.Series, groups: pd.DataFrame
) -> np.ndarray:
    temp = groups.copy()
    temp["__log_weight"] = pd.to_numeric(log_weights, errors="raise").to_numpy(float)
    maxima = temp.groupby(list(groups.columns), dropna=False)["__log_weight"].transform("max")
    shifted = np.exp(temp["__log_weight"] - maxima)
    means = shifted.groupby([temp[column] for column in groups.columns], dropna=False).transform("mean")
    return (shifted / means).to_numpy(float)


def normalize_donor_points(
    donors: pd.DataFrame,
    profile_snapshot: pd.DataFrame,
    *,
    mode: str = "player_state_residualized",
    geometry_classifier: GeometryClassifier | None = None,
    reward_function: RewardFunction | None = None,
    config: NormalizationConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Normalize historical donor transitions with strict-prior profiles.

    ``profile_snapshot`` must describe the full pre-event roster for every
    event/category/state/setup represented in donors.  Field composition is
    calculated from those roster rows, never from realized held-out shot
    counts.  ``exposure_weight`` is allowed only because it is part of the
    dated snapshot; callers should use equal weights unless they have a
    separately frozen pre-event opportunity model.
    """

    config = config or NormalizationConfig()
    modes = {"raw", "event_centered", "player_state_residualized"}
    if mode not in modes:
        raise ValueError(f"mode must be one of {sorted(modes)}")
    _required(
        [
            config.event_col,
            config.event_start_col,
            config.player_col,
            *config.condition_cols,
            config.long_col,
            config.lat_col,
            config.lie_col,
            config.hazard_col,
            config.reward_col,
        ],
        donors,
        "donors",
    )
    if donors.empty:
        raise ValueError("donors must not be empty")
    work = donors.copy()
    work[config.event_start_col] = _parse_dates(
        work[config.event_start_col], config.event_start_col
    )
    if work.groupby(config.event_col)[config.event_start_col].nunique().gt(1).any():
        raise ValueError("a donor event has multiple event_start values")
    for column in [config.long_col, config.lat_col, config.reward_col]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    if work[[config.long_col, config.lat_col, config.reward_col]].isna().any().any():
        raise ValueError("donor endpoint coordinates and rewards must be finite")

    starts = work[[config.event_col, config.event_start_col]].drop_duplicates()
    profiles = _prepare_profiles(
        profile_snapshot,
        starts,
        config,
        allowed_snapshot_kinds={"strict_prior_asof"},
    )
    field = _weighted_group_profiles(profiles, config)
    profile_payload = profiles.drop(columns=config.event_start_col)
    profile_modifiers = set(profile_payload.columns) - set(config.profile_key_cols)
    overlap = sorted(set(work.columns) & profile_modifiers)
    if overlap:
        raise ValueError(
            "donors must not override strict-prior profile columns; overlapping "
            f"columns: {overlap}"
        )
    joined = work.merge(
        profile_payload,
        on=config.profile_key_cols,
        how="left",
        validate="many_to_one",
        suffixes=("", "_profile"),
    )
    if joined[config.cutoff_col].isna().any():
        sample = joined.loc[joined[config.cutoff_col].isna(), config.profile_key_cols].head(5)
        raise ValueError(f"donors lack exact strict-prior profiles: {sample.to_dict('records')}")
    joined = joined.merge(
        field,
        on=config.field_group_cols,
        how="left",
        validate="many_to_one",
    )

    points, residuals = _transform_rows_to_neutral_vectorized(joined, mode, config)
    out = joined.copy()
    out[config.model_long_col] = points[:, 0]
    out[config.model_lat_col] = points[:, 1]
    out["neutral_resid_long_yd"] = residuals[:, 0]
    out["neutral_resid_lat_yd"] = residuals[:, 1]
    out["raw_end_lie"] = out[config.lie_col]
    out["raw_end_hazard"] = out[config.hazard_col]
    out["donor_distance_ratio"] = out[config.distance_ratio_col]
    out["donor_ability_sg"] = out[config.ability_col]
    out["donor_player_relative_ability_sg"] = (
        out[config.ability_col] - out["field_ability_sg"]
    )
    out["profile_cutoff"] = out[config.cutoff_col]
    out["normalization_mode"] = mode
    out["normalization_source"] = out[config.source_col].astype(str)

    if mode == "raw":
        out[config.model_lie_col] = out[config.lie_col]
        out[config.model_hazard_col] = out[config.hazard_col]
        out[config.model_reward_col] = out[config.reward_col]
        out["coherence_status"] = "raw_labels_preserved"
    elif geometry_classifier is not None:
        out = _apply_classifier(out, geometry_classifier, config)
        if reward_function is not None:
            rewards = pd.to_numeric(reward_function(out.copy()), errors="coerce")
            if rewards.isna().any() or len(rewards) != len(out):
                raise ValueError("reward_function returned invalid rewards")
            out[config.model_reward_col] = rewards.to_numpy(float)
        else:
            out[config.model_reward_col] = np.nan
    else:
        out[config.model_lie_col] = pd.NA
        out[config.model_hazard_col] = pd.NA
        out[config.model_reward_col] = np.nan
        out["coherence_status"] = "labels_invalidated_neutral_latent"

    beta_scale = config.ability_log_weight_per_sg2
    if mode == "raw":
        field_log = np.zeros(len(out))
        player_log = np.zeros(len(out))
    else:
        reward = out[config.reward_col].to_numpy(float)
        field_log = -beta_scale * out["field_ability_sg"].to_numpy(float) * reward
        if mode == "event_centered":
            player_log = np.zeros(len(out))
        else:
            player_log = (
                -beta_scale
                * out["donor_player_relative_ability_sg"].to_numpy(float)
                * reward
            )
    out["neutral_field_log_weight"] = np.clip(
        field_log, -config.max_abs_log_weight, config.max_abs_log_weight
    )
    out["neutral_player_log_weight"] = np.clip(
        player_log, -config.max_abs_log_weight, config.max_abs_log_weight
    )
    out["neutral_log_weight"] = np.clip(
        out["neutral_field_log_weight"] + out["neutral_player_log_weight"],
        -config.max_abs_log_weight,
        config.max_abs_log_weight,
    )
    out["neutral_weight"] = _normalized_weights(
        out["neutral_log_weight"], out[config.field_group_cols]
    )
    out["ability_weighting_status"] = (
        "disabled_unestimated_beta"
        if abs(config.ability_log_weight_per_sg2) <= 1e-12
        else "enabled_explicit_research_beta"
    )

    donor_hash_cols = [
        config.event_col,
        config.event_start_col,
        config.player_col,
        *config.condition_cols,
        config.long_col,
        config.lat_col,
        config.lie_col,
        config.hazard_col,
        config.reward_col,
    ]
    profile_hash_cols = _profile_columns(config)
    metadata: dict[str, object] = {
        "artifact": "hole_sim_field_normalization",
        "version": 1,
        "mode": mode,
        "strict_prior": True,
        "donor_rows": int(len(out)),
        "donor_events": int(out[config.event_col].nunique()),
        "field_condition_groups": int(len(field)),
        "max_donor_event_start": out[config.event_start_col].max().isoformat(),
        "max_profile_cutoff": out[config.cutoff_col].max().isoformat(),
        "min_profile_cutoff": out[config.cutoff_col].min().isoformat(),
        "classifier_applied": geometry_classifier is not None,
        "reward_function_applied": reward_function is not None,
        "ability_weighting_status": out["ability_weighting_status"].iloc[0],
        "spatial_transform_backend": "vectorized_closed_form_2d_v1",
        "donor_input_sha256": _stable_frame_hash(work, donor_hash_cols),
        "profile_input_sha256": _stable_frame_hash(profiles, profile_hash_cols),
        "config_sha256": sha256(
            json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "code_sha256": _code_hash(),
    }
    metadata["provenance_sha256"] = _metadata_hash(metadata)
    out["normalization_provenance_sha256"] = metadata["provenance_sha256"]
    return out, metadata


def reapply_target_profile(
    neutral_atoms: pd.DataFrame,
    target_profile_snapshot: pd.DataFrame,
    *,
    target_player_id: object,
    target_event_start: object,
    geometry_classifier: GeometryClassifier | None = None,
    geometry_classification_mode: str = "physical",
    reward_function: RewardFunction | None = None,
    config: NormalizationConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply one target player's strict-prior profile to neutral donor atoms.

    The snapshot should contain the full target field, even though only
    ``target_player_id`` is applied.  That preserves the explicit field versus
    player-relative decomposition and makes the output auditable.

    ``geometry_classification_mode='physical'`` requires a classifier and is
    the only permitted TEE path.  ``'deferred_shot_relative'`` is APP-only and
    returns transformed miss vectors with lie/hazard/reward invalidated.  Those
    rows cannot be consumed as transitions until a later batched operation
    rotates/translates each vector from its generated particle start and
    classifies the target-course surface.
    """

    config = config or NormalizationConfig()
    classification_modes = {"physical", "deferred_shot_relative"}
    if geometry_classification_mode not in classification_modes:
        raise ValueError(
            "geometry_classification_mode must be one of "
            f"{sorted(classification_modes)}"
        )
    if geometry_classification_mode == "physical" and geometry_classifier is None:
        raise ValueError("physical target reapplication requires geometry_classifier")
    if geometry_classification_mode == "deferred_shot_relative":
        if geometry_classifier is not None:
            raise ValueError(
                "deferred_shot_relative must not run a geometry classifier before "
                "the generated APP start/direction is known"
            )
        categories = (
            neutral_atoms[config.category_col].dropna().astype(str).str.casefold().unique()
            if config.category_col in neutral_atoms
            else np.array([], dtype=object)
        )
        if len(categories) != 1 or categories[0] != "app":
            raise ValueError("deferred_shot_relative is permitted only for APP atoms")
        if reward_function is not None:
            raise ValueError(
                "deferred_shot_relative cannot compute reward before physical "
                "endpoint classification"
            )
    _required(
        [
            *config.condition_cols,
            "neutral_resid_long_yd",
            "neutral_resid_lat_yd",
            "neutral_log_weight",
        ],
        neutral_atoms,
        "neutral_atoms",
    )
    if neutral_atoms.empty:
        raise ValueError("neutral_atoms must not be empty")
    _required([config.event_col], target_profile_snapshot, "target_profile_snapshot")
    target_events = target_profile_snapshot[config.event_col].drop_duplicates()
    if len(target_events) != 1:
        raise ValueError("target_profile_snapshot must contain exactly one target event")
    target_event = target_events.iloc[0]
    starts = pd.DataFrame(
        {config.event_col: [target_event], config.event_start_col: [target_event_start]}
    )
    profiles = _prepare_profiles(
        target_profile_snapshot,
        starts,
        config,
        allowed_snapshot_kinds={"strict_prior_asof", "static_target_snapshot"},
    )
    field = _weighted_group_profiles(profiles, config)
    selected = profiles[profiles[config.player_col].astype(str) == str(target_player_id)].copy()
    if selected.empty:
        raise ValueError(f"target player {target_player_id!r} is absent from profile snapshot")
    if selected.duplicated(config.condition_cols).any():
        raise ValueError("target player has duplicate category/state/setup profiles")
    selected = selected.merge(
        field, on=config.field_group_cols, how="left", validate="one_to_one"
    )

    # Donor event/player columns remain as provenance.  Target identity is
    # added explicitly rather than overwriting them.
    target_payload_cols = [
        *config.condition_cols,
        config.cutoff_col,
        config.reference_long_col,
        config.reference_lat_col,
        config.ability_col,
        config.distance_ratio_col,
        config.long_bias_col,
        config.lat_bias_col,
        config.long_scale_col,
        config.lat_scale_col,
        config.correlation_col,
        config.source_col,
        config.snapshot_kind_col,
        "field_ability_sg",
        "field_distance_ratio",
        "field_long_bias_yd",
        "field_lat_bias_yd",
        "field_long_scale",
        "field_lat_scale",
        "field_long_lat_corr",
        "field_profile_players",
        "field_profile_max_cutoff",
    ]
    payload = selected[target_payload_cols].rename(
        columns={column: f"__target_{column}" for column in target_payload_cols if column not in config.condition_cols}
    )
    out = neutral_atoms.merge(
        payload, on=config.condition_cols, how="left", validate="many_to_one"
    )
    if out[f"__target_{config.cutoff_col}"].isna().any():
        sample = out.loc[
            out[f"__target_{config.cutoff_col}"].isna(), config.condition_cols
        ].drop_duplicates().head(5)
        raise ValueError(f"target profile lacks neutral atom conditions: {sample.to_dict('records')}")

    center_long = out[f"__target_{config.reference_long_col}"].to_numpy(float)
    center_lat = out[f"__target_{config.reference_lat_col}"].to_numpy(float)
    shift_long = (
        center_long
        * (out[f"__target_{config.distance_ratio_col}"].to_numpy(float) - 1.0)
        + out[f"__target_{config.long_bias_col}"].to_numpy(float)
    )
    shift_lat = out[f"__target_{config.lat_bias_col}"].to_numpy(float)
    transformed_long, transformed_lat = _apply_lower_2d(
        out["neutral_resid_long_yd"].to_numpy(float),
        out["neutral_resid_lat_yd"].to_numpy(float),
        out[f"__target_{config.long_scale_col}"].to_numpy(float),
        out[f"__target_{config.lat_scale_col}"].to_numpy(float),
        out[f"__target_{config.correlation_col}"].to_numpy(float),
    )
    model_points = np.column_stack(
        [center_long + shift_long + transformed_long, center_lat + shift_lat + transformed_lat]
    )
    out[config.model_long_col] = model_points[:, 0]
    out[config.model_lat_col] = model_points[:, 1]
    out["target_event_key"] = target_event
    out["target_player_id"] = target_player_id
    out["target_event_start"] = _parse_dates(
        pd.Series([target_event_start] * len(out)), "target_event_start"
    ).to_numpy()
    out["target_profile_cutoff"] = out[f"__target_{config.cutoff_col}"]
    out["target_normalization_source"] = out[f"__target_{config.source_col}"].astype(str)
    out["target_snapshot_kind"] = out[f"__target_{config.snapshot_kind_col}"].astype(str)
    out["target_field_ability_sg"] = out["__target_field_ability_sg"]
    out["target_player_relative_ability_sg"] = (
        out[f"__target_{config.ability_col}"] - out["__target_field_ability_sg"]
    )
    if geometry_classification_mode == "physical":
        # Checked above; the assertion keeps the callable type explicit here.
        assert geometry_classifier is not None
        out = _apply_classifier(out, geometry_classifier, config)
        out["geometry_classification_status"] = "classified_in_target_frame"
    else:
        out["deferred_miss_long_yd"] = out[config.model_long_col]
        out["deferred_miss_lat_yd"] = out[config.model_lat_col]
        out[config.model_lie_col] = pd.NA
        out[config.model_hazard_col] = pd.NA
        out[config.model_reward_col] = np.nan
        out["coherence_status"] = "shot_relative_labels_deferred"
        out["geometry_classification_status"] = "deferred_until_generated_start"
        out["physical_transition_ready"] = False

    target_ability = out[f"__target_{config.ability_col}"].to_numpy(float)
    ability_weighting_active = abs(config.ability_log_weight_per_sg2) > 1e-12
    if reward_function is None:
        if ability_weighting_active and np.any(np.abs(target_ability) > 1e-12):
            raise ValueError(
                "A reward_function is required to reapply non-zero target ability "
                "without breaking endpoint/SG coherence"
            )
        target_rewards = np.zeros(len(out), dtype=float)
        out[config.model_reward_col] = np.nan
    else:
        rewards = pd.to_numeric(reward_function(out.copy()), errors="coerce")
        if rewards.isna().any() or len(rewards) != len(out):
            raise ValueError("reward_function returned invalid target rewards")
        target_rewards = rewards.to_numpy(float)
        out[config.model_reward_col] = target_rewards

    out["target_log_weight"] = np.clip(
        config.ability_log_weight_per_sg2 * target_ability * target_rewards,
        -config.max_abs_log_weight,
        config.max_abs_log_weight,
    )
    out["combined_log_weight"] = np.clip(
        out["neutral_log_weight"] + out["target_log_weight"],
        -config.max_abs_log_weight,
        config.max_abs_log_weight,
    )
    target_groups = out[config.condition_cols].copy()
    target_groups["target_player_id"] = str(target_player_id)
    out["model_weight"] = _normalized_weights(out["combined_log_weight"], target_groups)
    out["ability_weighting_status"] = (
        "disabled_unestimated_beta"
        if not ability_weighting_active
        else "enabled_explicit_research_beta"
    )

    profile_hash_cols = _profile_columns(config)
    parent_hashes = sorted(
        neutral_atoms.get("normalization_provenance_sha256", pd.Series(dtype=str))
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    metadata: dict[str, object] = {
        "artifact": "hole_sim_target_reapplication",
        "version": 1,
        "strict_prior": True,
        "target_event": str(target_event),
        "target_player": str(target_player_id),
        "target_event_start": pd.Timestamp(target_event_start).isoformat(),
        "target_profile_max_cutoff": profiles[config.cutoff_col].max().isoformat(),
        "neutral_atom_rows": int(len(out)),
        "geometry_reclassified": geometry_classification_mode == "physical",
        "geometry_classification_mode": geometry_classification_mode,
        "geometry_classification_status": out[
            "geometry_classification_status"
        ].iloc[0],
        "physical_transition_ready": bool(out["physical_transition_ready"].all()),
        "reward_function_applied": reward_function is not None,
        "ability_weighting_status": out["ability_weighting_status"].iloc[0],
        "spatial_transform_backend": "vectorized_closed_form_2d_v1",
        "parent_provenance_sha256": parent_hashes,
        "target_profile_input_sha256": _stable_frame_hash(profiles, profile_hash_cols),
        "config_sha256": sha256(
            json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "code_sha256": _code_hash(),
    }
    metadata["provenance_sha256"] = _metadata_hash(metadata)
    out["target_reapplication_provenance_sha256"] = metadata["provenance_sha256"]
    return out, metadata


__all__ = [
    "NormalizationConfig",
    "assert_physical_transition_ready",
    "latest_strict_prior_profiles",
    "normalize_donor_points",
    "reapply_target_profile",
]
