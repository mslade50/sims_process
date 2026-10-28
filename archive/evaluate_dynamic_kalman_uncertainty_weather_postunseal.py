"""Post-unseal dynamic-Kalman uncertainty and historical-weather diagnostic.

This is a research-only extension of the frozen base-skill study.  It does not
edit or overwrite the builder, discovery, sealed, cascade, finish-market, or
production artifacts.  See ``DYNAMIC_KALMAN_POST_UNSEAL_DIAGNOSTIC_DESIGN_2026_08.md``.

The two predeclared extensions are:

* preserve the selected q=0.0005/prior=20 W0 state mean while carrying its
  target-date-aged posterior covariance into a pair-specific matchup mapper;
* build an otherwise identical W1 state that weather-neutralizes only exact,
  strictly prior PGA round matches and leaves every other observation W0.

Every fitted probability parameter uses 2020-2022 only.  All 2023-2026 output
is explicitly post-unseal exploratory and cannot be treated as fresh sealed
validation or as a promotion decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from archive import backtest_base_skill_market_models as market_eval  # noqa: E402
from archive import build_complete_base_skill_candidates as builder  # noqa: E402
from archive import evaluate_frozen_base_skill_sealed_matchups as sealed_eval  # noqa: E402
from archive import research_execution_provenance as execution_provenance  # noqa: E402


ROOT = (
    REPO / "permanent_data" / "research" / "base_skill_rebuild_20200101_20260822"
)
COMPLETION_DIR = ROOT / "model_completion_v3_provenance"
SEALED_DIR = ROOT / "model_eval_complete_sealed_v5_diagnostic_full_20230101_20260822"
DEFAULT_OUTPUT = ROOT / "dynamic_kalman_uncertainty_weather_postunseal_v1"
DEFAULT_ALL_TOUR_DB = (
    REPO.parent / "sim_prep" / "work" / "dg_historical.db"
)

SELECTED = "complete__dynamic_kalman__xtour__w0__event_batch__q0.0005"
CONTROL = "current_architecture_pt_v2_approx_control"

ARM_CONTROL = "postunseal__strict_pt_control__existing_global"
ARM_W0_EXISTING = "postunseal__kalman_w0__existing_global"
ARM_W0_GLOBAL = "postunseal__kalman_w0__global_refit_2020_2022"
ARM_W0_UNCERTAINTY = "postunseal__kalman_w0__posterior_uncertainty_2020_2022"
ARM_W1_GLOBAL = "postunseal__kalman_w1__global_refit_2020_2022"
ARM_W1_UNCERTAINTY = "postunseal__kalman_w1__posterior_uncertainty_2020_2022"
ARMS = (
    ARM_CONTROL,
    ARM_W0_EXISTING,
    ARM_W0_GLOBAL,
    ARM_W0_UNCERTAINTY,
    ARM_W1_GLOBAL,
    ARM_W1_UNCERTAINTY,
)

CALIBRATION_START = pd.Timestamp("2020-01-01")
CALIBRATION_END = pd.Timestamp("2022-12-31")
SEALED_START = pd.Timestamp("2023-01-01")
EXACT_THREE_YEAR_START = pd.Timestamp("2023-08-22")
STUDY_END = pd.Timestamp("2026-08-22")
PROCESS_VARIANCE_PER_DAY = 0.0005
PRIOR_ROUNDS = 20.0
BOOTSTRAP_SEED = 847_231
WEATHER_JOIN_KEYS = (
    "dg_id",
    "year",
    "event_id",
    "round_num",
    "round_date",
    "course_num",
)

EXPECTED = {
    "completion_manifest": (COMPLETION_DIR / "manifest.json", "53a6446d683d6a4709acd24cdf7843bf2db7393c38e7f09eb3cc803fa416e8e0", 177_511),
    "predictions": (COMPLETION_DIR / "player_event_predictions.parquet", "c5ad98f85ce19a750922735c067a14cde0bd1b2d002e35df1103a8392c5aaae2", 149_135_818),
    "augmented_snapshots": (COMPLETION_DIR / "augmented_feature_snapshots.parquet", "47d4d7580599ba0452ef53a4893802ee69f5690677c3179c9dde5ef0adf44a96", 195_986_301),
    "quotes": (ROOT / "model_eval_clean_dg_plumbing_v4" / "normalized_matchup_quotes.parquet", "a1ea5b39786148f471cf0786f2e17a8681fe0fd38374b583f8e3f8cab1aba894", 7_481_034),
    "round_panel": (ROOT / "player_round_panel.parquet", "37df323c7e43681e02aa90bb12d84406bade24586315082e2e553ff323485f2a", 44_577_853),
    "weather_coefficients": (ROOT / "model_eval_clean_dg_plumbing_v2" / "weather_coefficients.csv", "f5bd742ad81a35cf87a7e8c27254863ffbfc081afece629cd110b6d9b84fa4ec", 99_059),
    "all_tour_db": (DEFAULT_ALL_TOUR_DB, "a932187976398d6976a1d366194f03a8df7a5efa5c56c6b11b7b0e13d6b29841", 1_003_442_176),
    "sealed_logical": (SEALED_DIR / "sealed_logical_probabilities.parquet", "b3de33d9a2282e5c25aef4c0f7952da799f41e9f2551779ff8b865e97fd9bb8e", 1_815_450),
    "sealed_performance": (SEALED_DIR / "betting_performance.csv", "4a3180a63656fe6ee53460749938583c933e9addd3e479da291a862a44d4a36d", 372_637),
    "sealed_mapper": (SEALED_DIR / "frozen_probability_mapper.csv", "a10154abbeb9e159ce92f3787a1663d8c8ea7e9747155f52b05d6397bc9d9899", 1_886),
    "sealed_manifest": (SEALED_DIR / "manifest.json", "fc6f329e2369c9171f2e47d284a33ba3b83e925c7d6746141011bafc05c1d7ed", 14_637),
}

SAMPLES = {
    "calibration_2020_2022_in_sample": (CALIBRATION_START, CALIBRATION_END),
    "sealed_2023_2026_postunseal": (SEALED_START, STUDY_END),
    "exact_three_year_postunseal": (EXACT_THREE_YEAR_START, STUDY_END),
}

COMPARISONS = (
    (ARM_W0_UNCERTAINTY, ARM_W0_GLOBAL, "posterior_uncertainty_within_w0"),
    (ARM_W1_GLOBAL, ARM_W0_GLOBAL, "historical_weather_global_mapper"),
    (ARM_W1_UNCERTAINTY, ARM_W1_GLOBAL, "posterior_uncertainty_within_w1"),
    (ARM_W1_UNCERTAINTY, ARM_W0_GLOBAL, "combined_corrections"),
    (ARM_W0_UNCERTAINTY, ARM_CONTROL, "w0_uncertainty_vs_strict_pt_control"),
    (ARM_W1_UNCERTAINTY, ARM_CONTROL, "w1_uncertainty_vs_strict_pt_control"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(path: Path) -> dict[str, Any]:
    path = path.resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _validate_authorities() -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for name, (path, expected_hash, expected_bytes) in EXPECTED.items():
        record = _fingerprint(path)
        if expected_hash and record["sha256"] != expected_hash:
            raise ValueError(
                f"pinned {name} hash mismatch: {record['sha256']} != {expected_hash}"
            )
        if expected_bytes and record["bytes"] != expected_bytes:
            raise ValueError(
                f"pinned {name} byte-size mismatch: {record['bytes']} != {expected_bytes}"
            )
        output[name] = record
    source_expected = {
        Path(builder.__file__).resolve(): "69d442b7c91a6191861842290fa69ecee437d35830510bcf0fd3edd9d28cc019",
        Path(market_eval.__file__).resolve(): "0870d1eb7dd9364b3e7090b2f61643ddca59e536641f6fdb18971b004f22529d",
        Path(sealed_eval.__file__).resolve(): "b9d5d74621568497b85ab73d641617ab17877f9107d2d7c3eaaf1097198c0500",
        Path(execution_provenance.__file__).resolve(): "b5ed954a1c7578891614d319915848ca659ad7f2dd40fde77e146cda7eeb0034",
    }
    for path, expected_hash in source_expected.items():
        actual = _sha256(path)
        if actual != expected_hash:
            raise ValueError(f"imported research source changed: {path}: {actual}")
    manifest = json.loads(EXPECTED["completion_manifest"][0].read_text(encoding="utf-8"))
    if not manifest.get("complete") or not manifest.get("research_only"):
        raise ValueError("completion authority is not a complete research-only run")
    if manifest.get("production_files_changed") is not False:
        raise ValueError("completion authority does not attest zero production changes")
    return output


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _normalize_join_types(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in ("dg_id", "year", "event_id", "round_num", "course_num"):
        output[column] = pd.to_numeric(output[column], errors="coerce").astype("Int64")
    output["round_date"] = pd.to_datetime(
        output["round_date"], errors="coerce"
    ).dt.normalize()
    return output


def load_all_tour_rounds_with_course(path: Path) -> pd.DataFrame:
    """Pinned builder load plus physical course identity for the W1 join."""

    columns = (*builder.ALL_TOUR_QUERY_COLUMNS, "course_num")
    uri = f"file:{path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        info = pd.read_sql_query("PRAGMA table_info(player_rounds)", connection)
        available = set(info["name"].astype(str))
        missing = sorted(set(columns) - available)
        if missing:
            raise ValueError(f"all-tour database lacks W1 identity columns: {missing}")
        query = (
            "SELECT "
            + ", ".join(f'"{column}"' for column in columns)
            + " FROM player_rounds WHERE dg_id IS NOT NULL "
            + "AND round_date IS NOT NULL AND sg_total_adj IS NOT NULL"
        )
        frame = pd.read_sql_query(query, connection)
    frame["dg_id"] = pd.to_numeric(frame["dg_id"], errors="coerce").astype("Int64")
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    frame["event_id"] = pd.to_numeric(frame["event_id"], errors="coerce").astype("Int64")
    frame["round_num"] = pd.to_numeric(frame["round_num"], errors="coerce")
    frame["course_num"] = pd.to_numeric(frame["course_num"], errors="coerce").astype("Int64")
    frame["round_date"] = pd.to_datetime(
        frame["round_date"], errors="coerce"
    ).dt.normalize()
    frame["sg_total_adj"] = pd.to_numeric(frame["sg_total_adj"], errors="coerce")
    frame["tour"] = builder._normalize_tour(frame["tour"])
    for column in ("age", "pro_time"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[frame["tour"].isin(builder.TOURS)].dropna(
        subset=["dg_id", "round_date", "sg_total_adj"]
    )
    return frame.sort_values(
        ["dg_id", "round_date", "tour", "event_id", "round_num"],
        kind="mergesort",
    ).reset_index(drop=True)


def attach_exact_pga_weather(
    rounds: pd.DataFrame,
    round_panel_path: Path,
    *,
    target_dates: Sequence[pd.Timestamp],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach weather only through an exact physical-round identity.

    The round panel is loaded as one pinned authority.  Audit its literal I/O
    separately from feature use: rows on or after the first target can be
    loaded here, while :func:`build_dynamic_states` must still prove that no
    such row entered a target's strictly-prior history.
    """

    output = _normalize_join_types(rounds)
    output["weather_delta_wind"] = np.nan
    output["weather_delta_dew"] = np.nan
    output["weather_exact_match"] = False
    output["weather_available"] = False
    pga_positions = output.index[output["tour"].eq("pga")].to_numpy(int)
    pga = output.loc[pga_positions, list(WEATHER_JOIN_KEYS) + ["sg_total_adj"]].copy()
    if pga.duplicated(list(WEATHER_JOIN_KEYS)).any():
        examples = pga.loc[
            pga.duplicated(list(WEATHER_JOIN_KEYS), keep=False), WEATHER_JOIN_KEYS
        ].head(10)
        raise ValueError(f"all-tour PGA weather identities are not unique: {examples}")

    weather_columns = list(WEATHER_JOIN_KEYS) + [
        "dg_sg_total_field_adjusted",
        "weather_delta_wind",
        "weather_delta_dew",
    ]
    weather = _normalize_join_types(
        pd.read_parquet(round_panel_path, columns=weather_columns)
    )
    normalized_targets = pd.to_datetime(
        pd.Series(target_dates, dtype="object"), errors="coerce"
    ).dropna().dt.normalize()
    if normalized_targets.empty:
        raise ValueError("weather audit requires at least one finite target date")
    first_target = pd.Timestamp(normalized_targets.min()).normalize()
    last_target = pd.Timestamp(normalized_targets.max()).normalize()
    loaded_target_or_upcoming = weather["round_date"].ge(first_target)
    if weather.duplicated(list(WEATHER_JOIN_KEYS)).any():
        examples = weather.loc[
            weather.duplicated(list(WEATHER_JOIN_KEYS), keep=False), WEATHER_JOIN_KEYS
        ].head(10)
        raise ValueError(f"round-panel weather identities are not unique: {examples}")

    joined = pga.merge(
        weather,
        on=list(WEATHER_JOIN_KEYS),
        how="left",
        validate="one_to_one",
        indicator=True,
        suffixes=("_db", "_panel"),
    )
    joined.index = pga_positions
    matched = joined["_merge"].eq("both")
    available = matched & joined[["weather_delta_wind", "weather_delta_dew"]].notna().all(axis=1)
    output.loc[pga_positions, "weather_exact_match"] = matched.to_numpy(bool)
    output.loc[pga_positions, "weather_available"] = available.to_numpy(bool)
    output.loc[pga_positions, "weather_delta_wind"] = joined[
        "weather_delta_wind"
    ].to_numpy(float)
    output.loc[pga_positions, "weather_delta_dew"] = joined[
        "weather_delta_dew"
    ].to_numpy(float)

    comparable = matched & joined["dg_sg_total_field_adjusted"].notna()
    difference = (
        pd.to_numeric(joined.loc[comparable, "sg_total_adj"], errors="coerce")
        - pd.to_numeric(
            joined.loc[comparable, "dg_sg_total_field_adjusted"], errors="coerce"
        )
    ).abs()
    audit = {
        "join_keys": list(WEATHER_JOIN_KEYS),
        "all_tour_rows": int(len(output)),
        "pga_rows": int(len(pga)),
        "non_pga_rows_left_w0": int(len(output) - len(pga)),
        "exact_weather_matches": int(matched.sum()),
        "exact_weather_match_rate": float(matched.mean()) if len(matched) else 0.0,
        "finite_wind_dew_rows": int(available.sum()),
        "finite_wind_dew_rate_of_pga": float(available.mean()) if len(available) else 0.0,
        "unmatched_pga_rows_left_w0": int((~matched).sum()),
        "matched_incomplete_weather_rows_left_w0": int((matched & ~available).sum()),
        "base_sg_comparable_rows": int(comparable.sum()),
        "base_sg_max_abs_difference": float(difference.max()) if len(difference) else None,
        "weather_panel_rows_loaded": int(len(weather)),
        "weather_panel_min_round_date": weather["round_date"].min(),
        "weather_panel_max_round_date": weather["round_date"].max(),
        "target_date_min": first_target,
        "target_date_max": last_target,
        "target_or_upcoming_weather_rows_loaded": int(
            loaded_target_or_upcoming.sum()
        ),
        "target_or_upcoming_weather_loaded": bool(
            loaded_target_or_upcoming.any()
        ),
        "weather_excluded_column_read": False,
    }
    return output.sort_values(
        ["dg_id", "round_date", "tour", "event_id", "round_num"],
        kind="mergesort",
    ).reset_index(drop=True), audit


def w1_values(history: pd.DataFrame, coefficient: Mapping[str, Any]) -> np.ndarray:
    """Return W1 values; unmatched/non-PGA/incomplete rows are exact W0."""

    raw = pd.to_numeric(history["sg_total_adj"], errors="coerce").to_numpy(float)
    available = (
        history["tour"].eq("pga")
        & history["weather_exact_match"].fillna(False).astype(bool)
        & history["weather_available"].fillna(False).astype(bool)
    ).to_numpy(bool)
    wind = pd.to_numeric(history["weather_delta_wind"], errors="coerce").to_numpy(float)
    dew = pd.to_numeric(history["weather_delta_dew"], errors="coerce").to_numpy(float)
    correction = (
        float(coefficient["total_beta_wind"]) * np.nan_to_num(wind, nan=0.0)
        + float(coefficient["total_beta_dew"]) * np.nan_to_num(dew, nan=0.0)
    )
    result = raw.copy()
    result[available] += correction[available]
    return result


def kalman_state_with_target_variance(
    history: pd.DataFrame,
    offsets: Mapping[str, float],
    variances: Mapping[str, float],
    prior_mean: float,
    target: pd.Timestamp,
    *,
    prior_rounds: float = PRIOR_ROUNDS,
    process_variance_per_day: float = PROCESS_VARIANCE_PER_DAY,
    weather_coefficient: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay the selected state and carry covariance to the target date."""

    target = pd.Timestamp(target).normalize()
    use = history.loc[
        history["round_num"].isin([1, 2]) & history["sg_total_adj"].notna()
    ].sort_values("round_date", kind="mergesort")
    dates = pd.to_datetime(use["round_date"], errors="coerce").dt.normalize()
    if dates.isna().any() or dates.ge(target).any():
        raise ValueError("Kalman history contains a missing or non-prior round date")
    pga_variance = max(float(variances.get("pga", 5.0)), 1e-8)
    state = float(prior_mean)
    variance = pga_variance / max(float(prior_rounds), 1e-8)
    if use.empty:
        return {
            "mean": state,
            "posterior_variance_last_observation": variance,
            "target_variance": variance,
            "last_observation_date": pd.NaT,
            "target_age_days": 0,
            "history_rows": 0,
            "weather_adjusted_rows": 0,
        }
    observations = (
        w1_values(use, weather_coefficient)
        if weather_coefficient is not None
        else use["sg_total_adj"].to_numpy(float)
    )
    previous_date: pd.Timestamp | None = None
    for position, row in enumerate(use.itertuples(index=False)):
        date = pd.Timestamp(row.round_date).normalize()
        if previous_date is not None:
            variance += process_variance_per_day * max((date - previous_date).days, 0)
        tour = str(row.tour)
        observation = float(observations[position]) - float(offsets.get(tour, 0.0))
        observation_variance = max(float(variances.get(tour, 5.0)), 1e-8)
        gain = variance / (variance + observation_variance)
        state += gain * (observation - state)
        variance = max((1.0 - gain) * variance, 1e-8)
        previous_date = date
    assert previous_date is not None
    age_days = max((target - previous_date).days, 0)
    target_variance = variance + process_variance_per_day * age_days
    weather_adjusted = 0
    if weather_coefficient is not None:
        weather_adjusted = int(
            (
                use["tour"].eq("pga")
                & use["weather_exact_match"].fillna(False).astype(bool)
                & use["weather_available"].fillna(False).astype(bool)
            ).sum()
        )
    return {
        "mean": float(state),
        "posterior_variance_last_observation": float(variance),
        "target_variance": float(target_variance),
        "last_observation_date": previous_date,
        "target_age_days": int(age_days),
        "history_rows": int(len(use)),
        "weather_adjusted_rows": weather_adjusted,
    }


def _w1_pga_variance(
    pga_r12: pd.DataFrame,
    cutoff: pd.Timestamp,
    coefficient: Mapping[str, Any],
) -> float:
    use = pga_r12.loc[pga_r12["round_date"].lt(cutoff)]
    if len(use) < 100:
        return 5.0
    values = w1_values(use, coefficient)
    return float(np.clip(np.var(values), 1.0, 12.0))


def _smoke_subset(
    predictions: pd.DataFrame, events_per_year: int
) -> pd.DataFrame:
    if events_per_year <= 0:
        return predictions
    event_dates = (
        predictions[["event_key", "event_start_date"]]
        .drop_duplicates()
        .assign(year=lambda frame: frame["event_start_date"].dt.year)
        .sort_values(["year", "event_start_date", "event_key"], kind="mergesort")
    )
    keep = set(
        event_dates.groupby("year", observed=True).head(events_per_year)["event_key"]
    )
    return predictions.loc[predictions["event_key"].isin(keep)].copy()


def build_dynamic_states(
    snapshots: pd.DataFrame,
    rounds: pd.DataFrame,
    frozen_selected: pd.DataFrame,
    weather_coefficients: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build W0 covariance replay and the exact W1 sibling."""

    lookup = builder._coefficient_lookup(weather_coefficients)
    history_map = {
        int(dg_id): rounds.iloc[np.asarray(positions, dtype=int)].sort_values(
            "round_date", kind="mergesort"
        )
        for dg_id, positions in rounds.groupby("dg_id", sort=False, observed=True).indices.items()
    }
    empty_history = rounds.iloc[:0]
    w0_moments = builder.ExpandingTourMoments(rounds)
    pga_r12 = rounds.loc[
        rounds["tour"].eq("pga") & rounds["round_num"].isin([1, 2])
    ].copy()
    pieces: list[pd.DataFrame] = []
    coefficient_fit_violations = 0
    source_date_violations = 0
    target_or_upcoming_weather_feature_rows = 0
    for cutoff, test in frozen_selected.groupby(
        "event_start_date", sort=True, observed=True
    ):
        cutoff = pd.Timestamp(cutoff).normalize()
        coefficient = builder._weather_row_for_target(lookup, cutoff)
        fit_max = pd.to_datetime(coefficient.get("fit_max_round_date"), errors="coerce")
        if pd.notna(fit_max) and pd.Timestamp(fit_max).normalize() >= cutoff:
            coefficient_fit_violations += 1
        offsets_w0, _ = builder.estimate_tour_translations(snapshots, cutoff)
        offsets_w1, _ = builder.estimate_tour_translations(
            snapshots, cutoff, pga_weather_arm="w1"
        )
        variances_w0 = {
            tour: w0_moments.variance(tour, cutoff) for tour in builder.TOURS
        }
        variances_w1 = dict(variances_w0)
        variances_w1["pga"] = _w1_pga_variance(pga_r12, cutoff, coefficient)
        prior_w0 = builder._dynamic_prior_mean(snapshots, cutoff)
        prior_w1 = builder._dynamic_prior_mean(
            snapshots,
            cutoff,
            target_column=builder.NEUTRAL_LABEL,
            count_column="label_dg_sg_neutral_r1_r2_count",
        )
        rows: list[dict[str, Any]] = []
        for row in test.itertuples(index=False):
            dg_id = int(row.dg_id)
            history = history_map.get(dg_id, empty_history)
            prior = builder._dynamic_player_history(history, cutoff)
            target_or_upcoming_weather_feature_rows += int(
                (
                    pd.to_datetime(prior["round_date"], errors="coerce")
                    .dt.normalize()
                    .ge(cutoff)
                    & prior["weather_available"].fillna(False).astype(bool)
                ).sum()
            )
            w0 = kalman_state_with_target_variance(
                prior, offsets_w0, variances_w0, prior_w0, cutoff
            )
            w1 = kalman_state_with_target_variance(
                prior,
                offsets_w1,
                variances_w1,
                prior_w1,
                cutoff,
                weather_coefficient=coefficient,
            )
            for state in (w0, w1):
                if pd.notna(state["last_observation_date"]) and pd.Timestamp(
                    state["last_observation_date"]
                ) >= cutoff:
                    source_date_violations += 1
            rows.append(
                {
                    "event_key": row.event_key,
                    "event_start_date": cutoff,
                    "dg_id_key": market_eval._stable_id(row.dg_id_key),
                    "dg_id": dg_id,
                    "frozen_w0_mean": float(row.prediction),
                    "w0_mean": w0["mean"],
                    "w0_variance_last": w0["posterior_variance_last_observation"],
                    "w0_target_variance": w0["target_variance"],
                    "w0_last_observation_date": w0["last_observation_date"],
                    "w0_target_age_days": w0["target_age_days"],
                    "w1_mean": w1["mean"],
                    "w1_variance_last": w1["posterior_variance_last_observation"],
                    "w1_target_variance": w1["target_variance"],
                    "w1_last_observation_date": w1["last_observation_date"],
                    "w1_target_age_days": w1["target_age_days"],
                    "history_rows": w0["history_rows"],
                    "w1_weather_adjusted_rows": w1["weather_adjusted_rows"],
                    "w1_total_beta_wind": float(coefficient["total_beta_wind"]),
                    "w1_total_beta_dew": float(coefficient["total_beta_dew"]),
                    "w1_coefficient_fit_max_round_date": fit_max,
                    "w1_coefficient_source": coefficient.get("coefficient_source", "unknown"),
                    "w0_pga_observation_variance": variances_w0["pga"],
                    "w1_pga_observation_variance": variances_w1["pga"],
                }
            )
        pieces.append(pd.DataFrame(rows))
    states = pd.concat(pieces, ignore_index=True)
    if states.duplicated(["event_key", "dg_id_key"]).any():
        raise ValueError("dynamic state keys are not unique")
    replay_error = (states["w0_mean"] - states["frozen_w0_mean"]).abs()
    if not np.isfinite(replay_error).all() or replay_error.max() > 1e-12:
        raise ValueError(
            "W0 posterior mean replay does not preserve the frozen selected mean: "
            f"max_abs_error={replay_error.max()}"
        )
    if (
        coefficient_fit_violations
        or source_date_violations
        or target_or_upcoming_weather_feature_rows
    ):
        raise ValueError(
            "dynamic state chronology failed: "
            f"coefficient={coefficient_fit_violations}, "
            f"source={source_date_violations}, "
            f"target_or_upcoming_weather_feature_rows="
            f"{target_or_upcoming_weather_feature_rows}"
        )
    if (states[["w0_target_variance", "w1_target_variance"]] <= 0).any().any():
        raise ValueError("dynamic state emitted nonpositive target covariance")
    audit = {
        "rows": int(len(states)),
        "events": int(states["event_key"].nunique()),
        "players": int(states["dg_id_key"].nunique()),
        "w0_replay_max_abs_error": float(replay_error.max()),
        "w0_replay_exact_within_1e_12": True,
        "source_date_violations": int(source_date_violations),
        "coefficient_fit_violations": int(coefficient_fit_violations),
        "w0_target_variance_mean": float(states["w0_target_variance"].mean()),
        "w1_target_variance_mean": float(states["w1_target_variance"].mean()),
        "target_aging_rows": int(states["w0_target_age_days"].gt(0).sum()),
        "target_aging_mean_days": float(states["w0_target_age_days"].mean()),
        "w1_rows_with_weather_history": int(states["w1_weather_adjusted_rows"].gt(0).sum()),
        "w1_weather_history_coverage_rate": float(states["w1_weather_adjusted_rows"].gt(0).mean()),
        "w1_total_historical_weather_observations_used": int(states["w1_weather_adjusted_rows"].sum()),
        "w1_mean_minus_w0_mean_average": float((states["w1_mean"] - states["w0_mean"]).mean()),
        "w1_mean_minus_w0_mean_abs_average": float((states["w1_mean"] - states["w0_mean"]).abs().mean()),
        "target_or_upcoming_weather_feature_rows": int(
            target_or_upcoming_weather_feature_rows
        ),
        "target_or_upcoming_realized_weather_feature_used": bool(
            target_or_upcoming_weather_feature_rows
        ),
        "weather_excluded_or_outcome_selected_flag_read": False,
    }
    return states, audit


def build_prediction_arms(
    states: pd.DataFrame,
    frozen_selected: pd.DataFrame,
    frozen_control: pd.DataFrame,
) -> pd.DataFrame:
    metadata = frozen_selected[
        [
            "event_key",
            "dg_id_key",
            "reliability_ok",
            "history_rounds",
            "days_since_last_round",
            "prior_event_withdrawal",
        ]
    ].drop_duplicates(["event_key", "dg_id_key"])
    state_base = states.merge(
        metadata,
        on=["event_key", "dg_id_key"],
        how="left",
        validate="one_to_one",
    )
    pieces: list[pd.DataFrame] = []
    for arm, mean, variance in (
        (ARM_W0_EXISTING, "w0_mean", "w0_target_variance"),
        (ARM_W0_GLOBAL, "w0_mean", "w0_target_variance"),
        (ARM_W0_UNCERTAINTY, "w0_mean", "w0_target_variance"),
        (ARM_W1_GLOBAL, "w1_mean", "w1_target_variance"),
        (ARM_W1_UNCERTAINTY, "w1_mean", "w1_target_variance"),
    ):
        piece = state_base.copy()
        piece["candidate"] = arm
        piece["prediction"] = piece[mean]
        piece["target_posterior_variance"] = piece[variance]
        pieces.append(piece)
    control_columns = [
        "event_key",
        "dg_id_key",
        "dg_id",
        "event_start_date",
        "prediction",
        "reliability_ok",
        "history_rounds",
        "days_since_last_round",
        "prior_event_withdrawal",
    ]
    control = frozen_control[control_columns].copy()
    control["candidate"] = ARM_CONTROL
    control["target_posterior_variance"] = np.nan
    pieces.append(control)
    output = pd.concat(pieces, ignore_index=True, sort=False)
    keys = ["candidate", "event_key", "dg_id_key"]
    if output.duplicated(keys).any():
        raise ValueError("prediction-arm keys are not unique")
    counts = output.groupby("candidate", observed=True).size()
    if counts.nunique() != 1 or set(counts.index) != set(ARMS):
        raise ValueError(f"prediction-arm coverage differs: {counts.to_dict()}")
    return output


def attach_pair_variances(
    quotes: pd.DataFrame, predictions: pd.DataFrame
) -> pd.DataFrame:
    paired = market_eval.attach_skill_differences(quotes, predictions)
    lookup = predictions[
        ["candidate", "event_key", "dg_id_key", "target_posterior_variance"]
    ].drop_duplicates(["candidate", "event_key", "dg_id_key"])
    p1 = lookup.rename(
        columns={
            "event_key": "mapped_event_key",
            "dg_id_key": "p1_id",
            "target_posterior_variance": "p1_target_variance",
        }
    )
    p2 = lookup.rename(
        columns={
            "event_key": "mapped_event_key",
            "dg_id_key": "p2_id",
            "target_posterior_variance": "p2_target_variance",
        }
    )
    paired = paired.merge(
        p1,
        on=["candidate", "mapped_event_key", "p1_id"],
        how="left",
        validate="many_to_one",
    ).merge(
        p2,
        on=["candidate", "mapped_event_key", "p2_id"],
        how="left",
        validate="many_to_one",
    )
    paired["pair_target_variance"] = (
        paired["p1_target_variance"] + paired["p2_target_variance"]
    )
    counts = paired.groupby("candidate", observed=True)["quote_id"].nunique()
    if counts.nunique() != 1 or set(counts.index) != set(ARMS):
        raise ValueError(f"quote menus differ across arms: {counts.to_dict()}")
    menu = None
    for candidate in ARMS:
        values = set(paired.loc[paired["candidate"].eq(candidate), "quote_id"])
        if menu is None:
            menu = values
        elif values != menu:
            raise ValueError(f"quote menu mismatch for {candidate}")
    dynamic = paired.loc[~paired["candidate"].eq(ARM_CONTROL)]
    if dynamic["pair_target_variance"].isna().any() or (
        dynamic["pair_target_variance"] <= 0
    ).any():
        raise ValueError("dynamic matchup pairs lack positive posterior variance")
    return paired


def _event_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("mapped_event_key", observed=True)[
        "pair_key"
    ].transform("count")
    return 1.0 / counts.to_numpy(float)


def _canonical_fit_frame(frame: pd.DataFrame) -> pd.DataFrame:
    logical = market_eval._canonical_logical_pairs(frame)
    logical = logical.loc[
        logical["canonical_outcome"].isin([0.0, 0.5, 1.0])
    ].copy()
    if len(logical) < 100:
        raise ValueError(f"insufficient canonical calibration pairs: {len(logical)}")
    return logical


def _mapper_loss(
    difference: np.ndarray,
    pair_variance: np.ndarray,
    outcome: np.ndarray,
    weights: np.ndarray,
    base_sigma: float,
    tie_threshold: float,
    alpha: float,
) -> float:
    effective = np.sqrt(
        np.square(float(base_sigma))
        + max(float(alpha), 0.0) * np.clip(pair_variance, 0.0, None)
    )
    win = norm.cdf((difference - float(tie_threshold)) / effective)
    loss = norm.cdf((-difference - float(tie_threshold)) / effective)
    push = np.maximum(1.0 - win - loss, 0.0)
    probability = np.where(outcome == 1.0, win, np.where(outcome == 0.5, push, loss))
    return float(
        -np.sum(weights * np.log(np.clip(probability, market_eval.EPS, 1.0)))
        / np.sum(weights)
    )


def fit_uncertainty_mapper(frame: pd.DataFrame) -> dict[str, Any]:
    logical = _canonical_fit_frame(frame)
    difference = logical["canonical_skill_difference"].to_numpy(float)
    variance = logical["pair_target_variance"].to_numpy(float)
    outcome = logical["canonical_outcome"].to_numpy(float)
    weights = _event_equal_weights(logical)
    if not np.isfinite(variance).all() or (variance <= 0).any():
        raise ValueError("uncertainty calibration received invalid pair variance")

    global_sigma, global_tie = market_eval.fit_probability_params(frame)
    if not np.isfinite([global_sigma, global_tie]).all():
        raise RuntimeError("nested global mapper failed")
    global_loss = _mapper_loss(
        difference, variance, outcome, weights, global_sigma, global_tie, 0.0
    )

    best = None
    for alpha_start in (1e-4, 0.1, 1.0, 10.0, 100.0):
        result = minimize(
            lambda x: _mapper_loss(
                difference,
                variance,
                outcome,
                weights,
                math.exp(float(x[0])),
                math.exp(float(x[1])),
                math.exp(float(x[2])),
            ),
            np.log([global_sigma, global_tie, alpha_start]),
            method="L-BFGS-B",
            bounds=[
                (math.log(0.01), math.log(100.0)),
                (math.log(1e-5), math.log(5.0)),
                (math.log(1e-8), math.log(1e5)),
            ],
        )
        if result.success and (best is None or result.fun < best.fun):
            best = result
    if best is None:
        raise RuntimeError("posterior-uncertainty mapper failed every deterministic start")
    sigma, tie, alpha = np.exp(best.x)
    hetero_loss = float(best.fun)
    selected_boundary = False
    # Alpha=0 is part of the predeclared nested family.  Preserve that exact
    # boundary rather than accepting a numerically inferior tiny-positive
    # alpha when the data do not support pair-specific uncertainty.
    if hetero_loss >= global_loss:
        sigma = float(global_sigma)
        tie = float(global_tie)
        alpha = 0.0
        hetero_loss = float(global_loss)
        selected_boundary = True
    return {
        "global_sigma": float(global_sigma),
        "global_tie_threshold": float(global_tie),
        "global_event_equal_multinomial_log_loss": float(global_loss),
        "uncertainty_base_sigma": float(sigma),
        "uncertainty_tie_threshold": float(tie),
        "uncertainty_alpha": float(alpha),
        "uncertainty_event_equal_multinomial_log_loss": hetero_loss,
        "uncertainty_fit_improvement": float(global_loss - hetero_loss),
        "uncertainty_alpha_zero_boundary_selected": selected_boundary,
        "calibration_pairs": int(len(logical)),
        "calibration_events": int(logical["mapped_event_key"].nunique()),
        "pair_variance_mean": float(variance.mean()),
        "pair_variance_p10": float(np.quantile(variance, 0.10)),
        "pair_variance_p90": float(np.quantile(variance, 0.90)),
        "optimizer_success": True,
    }


def fit_mappers(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_arm in (ARM_W0_GLOBAL, ARM_W1_GLOBAL):
        state = "w0" if source_arm == ARM_W0_GLOBAL else "w1"
        for market in ("72h", "r1"):
            use = paired.loc[
                paired["candidate"].eq(source_arm)
                & paired["market"].eq(market)
                & paired["event_start_date"].between(
                    CALIBRATION_START, CALIBRATION_END
                )
            ].copy()
            fit = fit_uncertainty_mapper(use)
            rows.append(
                {
                    "state": state,
                    "source_arm": source_arm,
                    "market": market,
                    "fit_start": CALIBRATION_START,
                    "fit_end": CALIBRATION_END,
                    "sealed_outcomes_used": False,
                    "post_unseal_exploratory": True,
                    **fit,
                }
            )
    return pd.DataFrame(rows)


def _score_with_parameters(
    frame: pd.DataFrame,
    arm: str,
    *,
    base_sigma: float,
    tie_threshold: float,
    alpha: float,
    fit_start: pd.Timestamp,
    fit_end: pd.Timestamp,
    mapper_type: str,
) -> pd.DataFrame:
    output = frame.copy()
    output["candidate"] = arm
    pair_variance = pd.to_numeric(
        output["pair_target_variance"], errors="coerce"
    ).fillna(0.0).to_numpy(float)
    effective = np.sqrt(base_sigma**2 + alpha * np.clip(pair_variance, 0.0, None))
    difference = output["skill_difference"].to_numpy(float)
    win = norm.cdf((difference - tie_threshold) / effective)
    loss = norm.cdf((-difference - tie_threshold) / effective)
    push = np.maximum(1.0 - win - loss, 0.0)
    output["base_sigma"] = float(base_sigma)
    output["uncertainty_alpha"] = float(alpha)
    output["sigma"] = effective
    output["tie_threshold"] = float(tie_threshold)
    output["p1_probability"] = np.clip(win, market_eval.EPS, 1.0 - market_eval.EPS)
    output["push_probability"] = push
    output["p1_pricing_probability"] = (
        output["p1_probability"] / (1.0 - output["push_probability"])
    ).clip(market_eval.EPS, 1.0 - market_eval.EPS)
    output["probability_fit_start"] = fit_start
    output["probability_fit_end"] = fit_end
    output["mapper_type"] = mapper_type
    return output


def score_all_arms(
    paired: pd.DataFrame, fitted: pd.DataFrame, frozen_mapper: pd.DataFrame
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    frozen_mapper = frozen_mapper.copy()
    for source_candidate, source_arm, output_arm in (
        (CONTROL, ARM_CONTROL, ARM_CONTROL),
        (SELECTED, ARM_W0_EXISTING, ARM_W0_EXISTING),
    ):
        for market in ("72h", "r1"):
            params = frozen_mapper.loc[
                frozen_mapper["candidate"].eq(source_candidate)
                & frozen_mapper["market"].eq(market)
            ]
            if len(params) != 1:
                raise ValueError(f"frozen mapper missing {source_candidate}/{market}")
            row = params.iloc[0]
            source = paired.loc[
                paired["candidate"].eq(source_arm) & paired["market"].eq(market)
            ]
            pieces.append(
                _score_with_parameters(
                    source,
                    output_arm,
                    base_sigma=float(row["sigma"]),
                    tie_threshold=float(row["tie_threshold"]),
                    alpha=0.0,
                    fit_start=pd.Timestamp(row["fit_start"]),
                    fit_end=pd.Timestamp(row["fit_end"]),
                    mapper_type="existing_frozen_global_2021_2022",
                )
            )

    for state, global_arm, uncertainty_arm in (
        ("w0", ARM_W0_GLOBAL, ARM_W0_UNCERTAINTY),
        ("w1", ARM_W1_GLOBAL, ARM_W1_UNCERTAINTY),
    ):
        for market in ("72h", "r1"):
            params = fitted.loc[
                fitted["state"].eq(state) & fitted["market"].eq(market)
            ]
            if len(params) != 1:
                raise ValueError(f"fitted mapper missing {state}/{market}")
            row = params.iloc[0]
            source = paired.loc[
                paired["candidate"].eq(global_arm) & paired["market"].eq(market)
            ]
            pieces.append(
                _score_with_parameters(
                    source,
                    global_arm,
                    base_sigma=float(row["global_sigma"]),
                    tie_threshold=float(row["global_tie_threshold"]),
                    alpha=0.0,
                    fit_start=CALIBRATION_START,
                    fit_end=CALIBRATION_END,
                    mapper_type="nested_global_refit_2020_2022",
                )
            )
            pieces.append(
                _score_with_parameters(
                    source,
                    uncertainty_arm,
                    base_sigma=float(row["uncertainty_base_sigma"]),
                    tie_threshold=float(row["uncertainty_tie_threshold"]),
                    alpha=float(row["uncertainty_alpha"]),
                    fit_start=CALIBRATION_START,
                    fit_end=CALIBRATION_END,
                    mapper_type="target_aged_posterior_variance_2020_2022",
                )
            )
    output = pd.concat(pieces, ignore_index=True)
    counts = output.groupby("candidate", observed=True)["quote_id"].nunique()
    if counts.nunique() != 1 or set(counts.index) != set(ARMS):
        raise ValueError(f"scored quote menus differ: {counts.to_dict()}")
    return output


def canonical_probabilities(scored: pd.DataFrame) -> pd.DataFrame:
    logical = market_eval._canonical_logical_pairs(scored)
    keys = ["candidate", "mapped_event_key", "market", "pair_key"]
    if logical.duplicated(keys).any():
        raise ValueError("canonical logical probabilities are not unique")
    counts = logical.groupby("candidate", observed=True).size()
    if counts.nunique() != 1:
        raise ValueError(f"canonical menus differ: {counts.to_dict()}")
    return logical


def audit_original_probability_reproduction(
    logical: pd.DataFrame, original_path: Path
) -> dict[str, Any]:
    original = pd.read_parquet(original_path)
    mappings = {ARM_CONTROL: CONTROL, ARM_W0_EXISTING: SELECTED}
    rows: dict[str, Any] = {}
    key = ["mapped_event_key", "market", "pair_key"]
    for arm, candidate in mappings.items():
        ours = logical.loc[
            logical["candidate"].eq(arm)
            & logical["event_start_date"].between(SEALED_START, STUDY_END)
        ].copy()
        reference = original.loc[
            original["candidate"].eq(candidate)
            & original["mapped_event_key"].isin(set(ours["mapped_event_key"]))
        ].copy()
        merged = ours.merge(
            reference,
            on=key,
            how="outer",
            validate="one_to_one",
            indicator=True,
            suffixes=("_ours", "_reference"),
        )
        if not merged["_merge"].eq("both").all():
            raise ValueError(f"original logical menu reproduction failed for {arm}")
        probability_error = (
            merged["canonical_probability_ours"]
            - merged["canonical_probability_reference"]
        ).abs()
        push_error = (
            merged["canonical_push_probability_ours"]
            - merged["canonical_push_probability_reference"]
        ).abs()
        outcome_error = (
            pd.to_numeric(merged["canonical_outcome_ours"], errors="coerce")
            - pd.to_numeric(merged["canonical_outcome_reference"], errors="coerce")
        ).abs()
        if probability_error.max() > 1e-12 or push_error.max() > 1e-12:
            raise ValueError(f"original probability reproduction failed for {arm}")
        if outcome_error.dropna().max() > 0 or (
            merged["canonical_outcome_ours"].isna()
            != merged["canonical_outcome_reference"].isna()
        ).any():
            raise ValueError(f"original outcome reproduction failed for {arm}")
        rows[arm] = {
            "rows": int(len(merged)),
            "max_probability_abs_error": float(probability_error.max()),
            "max_push_probability_abs_error": float(push_error.max()),
            "max_outcome_abs_error": float(outcome_error.dropna().max()),
        }
    return rows


def _proper_event_scores(logical: pd.DataFrame, sample: str) -> pd.DataFrame:
    start, end = SAMPLES[sample]
    use = logical.loc[
        logical["event_start_date"].between(start, end)
        & logical["canonical_outcome"].isin([0.0, 1.0])
    ].copy()
    conditional = (
        use["canonical_probability"]
        / (1.0 - use["canonical_push_probability"])
    ).clip(market_eval.EPS, 1.0 - market_eval.EPS)
    y = use["canonical_outcome"].to_numpy(float)
    use["log_loss"] = -(y * np.log(conditional) + (1.0 - y) * np.log(1.0 - conditional))
    use["brier"] = np.square(conditional.to_numpy(float) - y)
    use["conditional_probability"] = conditional
    event = (
        use.groupby(
            ["candidate", "market", "mapped_event_key", "event_start_date"],
            observed=True,
        )
        .agg(
            log_loss=("log_loss", "mean"),
            brier=("brier", "mean"),
            resolved_pairs=("pair_key", "count"),
            mean_probability=("conditional_probability", "mean"),
            observed_win_rate=("canonical_outcome", "mean"),
        )
        .reset_index()
    )
    event["sample"] = sample
    return event


def proper_summaries(event: pd.DataFrame) -> pd.DataFrame:
    return (
        event.groupby(["sample", "candidate", "market"], observed=True)
        .agg(
            events=("mapped_event_key", "nunique"),
            resolved_pairs=("resolved_pairs", "sum"),
            event_equal_log_loss=("log_loss", "mean"),
            event_equal_brier=("brier", "mean"),
            mean_event_probability=("mean_probability", "mean"),
            mean_event_observed_win_rate=("observed_win_rate", "mean"),
        )
        .reset_index()
    )


def _bootstrap_mean_or_total(
    values: np.ndarray,
    draws: int,
    seed: int,
    *,
    total: bool,
) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_favorable": np.nan}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    sampled = values[indices]
    statistic = sampled.sum(axis=1) if total else sampled.mean(axis=1)
    estimate = float(values.sum() if total else values.mean())
    return {
        "estimate": estimate,
        "ci_low": float(np.quantile(statistic, 0.025)),
        "ci_high": float(np.quantile(statistic, 0.975)),
        "p_favorable": float(np.mean(statistic < 0)) if not total else float(np.mean(statistic > 0)),
    }


def proper_bootstrap(event: pd.DataFrame, draws: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sample in ("sealed_2023_2026_postunseal", "exact_three_year_postunseal"):
        for market in ("72h", "r1"):
            subset = event.loc[event["sample"].eq(sample) & event["market"].eq(market)]
            for challenger, reference, label in COMPARISONS:
                left = subset.loc[subset["candidate"].eq(challenger), ["mapped_event_key", "log_loss", "brier"]]
                right = subset.loc[subset["candidate"].eq(reference), ["mapped_event_key", "log_loss", "brier"]]
                merged = left.merge(
                    right,
                    on="mapped_event_key",
                    how="inner",
                    validate="one_to_one",
                    suffixes=("_challenger", "_reference"),
                )
                expected_left = set(left["mapped_event_key"])
                expected_right = set(right["mapped_event_key"])
                if expected_left != expected_right or len(merged) != len(expected_left):
                    raise ValueError(f"proper-score event menu mismatch for {label}/{sample}/{market}")
                for metric in ("log_loss", "brier"):
                    delta = (
                        merged[f"{metric}_challenger"]
                        - merged[f"{metric}_reference"]
                    ).to_numpy(float)
                    result = _bootstrap_mean_or_total(
                        delta, draws, seed + len(rows) * 101, total=False
                    )
                    rows.append(
                        {
                            "sample": sample,
                            "market": market,
                            "comparison": label,
                            "challenger": challenger,
                            "reference": reference,
                            "metric": metric,
                            "events": int(len(delta)),
                            "delta_definition": "challenger_minus_reference_lower_is_better",
                            **result,
                        }
                    )
    return pd.DataFrame(rows)


def _settlement_counts(bets: pd.DataFrame) -> dict[str, int]:
    """Return explicit settlement/refund counts for a materialized bet slice.

    DataGolf labels tied two-way matchup refunds as ``void`` rather than
    ``push``.  Keep the technical statuses separate and expose ``tie_refunds``
    so a zero push-status count cannot be mistaken for zero refunded ties.
    """

    status = bets.get(
        "settlement_status", pd.Series("", index=bets.index, dtype="object")
    ).fillna("").astype(str).str.lower()
    outcome = pd.to_numeric(
        bets.get("outcome_numeric", pd.Series(np.nan, index=bets.index)),
        errors="coerce",
    )
    refunded = status.isin(["push", "void"])
    tie_refund = refunded & np.isclose(outcome, 0.5, rtol=0.0, atol=1e-12)
    return {
        "pushes": int(status.eq("push").sum()),
        "voids": int(status.eq("void").sum()),
        "refunded_bets": int(refunded.sum()),
        "tie_refunds": int(tie_refund.sum()),
    }


def _summarize_bets(
    bets: pd.DataFrame,
    *,
    sample: str,
    price_basis: str,
    line_policy: str,
    sizing: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (candidate, market), group in bets.groupby(
        ["candidate", "round_market"], observed=True
    ):
        stake = float(group["stake"].sum())
        pnl = float(group["pnl"].sum())
        settlements = _settlement_counts(group)
        rows.append(
            {
                "sample": sample,
                "candidate": candidate,
                "market": market,
                "price_basis": price_basis,
                "line_policy": line_policy,
                "sizing": sizing,
                "reliability_policy": "ungated",
                "events_with_bets": int(group["event_key"].nunique()),
                "bets": int(len(group)),
                "stake": stake,
                "pnl": pnl,
                "roi": pnl / stake if stake else np.nan,
                "max_drawdown": market_eval._max_drawdown(group),
                "mean_nominal_ev": float(group["ev_per_unit"].mean()),
                "mean_open_to_close_clv_pp": float(
                    group["open_to_close_clv_pp"].mean()
                ),
                **settlements,
            }
        )
    return pd.DataFrame(rows)


def materialize_money(
    scored: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    summaries: list[pd.DataFrame] = []
    bets_out: list[pd.DataFrame] = []
    audit: dict[str, Any] = {"selected_bets": 0, "ungraded_selected_bets": 0}
    for sample, (start, end) in SAMPLES.items():
        sample_scored = scored.loc[scored["event_start_date"].between(start, end)]
        for price_basis in sealed_eval.PRICE_BASES:
            for line_policy in sealed_eval.LINE_POLICIES:
                for sizing in sealed_eval.SIZINGS:
                    bets, suppressed = sealed_eval.select_bet_slice(
                        sample_scored,
                        price_basis=price_basis,
                        line_policy=line_policy,
                        sizing=sizing,
                        reliability_policy="ungated",
                    )
                    if not suppressed.empty:
                        raise ValueError("ungated portfolio unexpectedly suppressed bets")
                    if not bets.empty:
                        ungraded = bets["outcome_numeric"].isna() | bets["pnl"].isna()
                        if ungraded.any():
                            raise ValueError(
                                f"selected portfolio has {int(ungraded.sum())} ungraded bets"
                            )
                        bets = bets.copy()
                        bets["sample"] = sample
                        bets["price_basis"] = price_basis
                        bets["line_policy"] = line_policy
                        bets["sizing"] = sizing
                        bets_out.append(bets)
                    summaries.append(
                        _summarize_bets(
                            bets,
                            sample=sample,
                            price_basis=price_basis,
                            line_policy=line_policy,
                            sizing=sizing,
                        )
                    )
    all_bets = pd.concat(bets_out, ignore_index=True) if bets_out else pd.DataFrame()
    performance = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    audit["selected_bets"] = int(len(all_bets))
    audit["ungraded_selected_bets"] = 0
    audit.update(_settlement_counts(all_bets))
    status_counts = (
        all_bets["settlement_status"].value_counts(dropna=False).sort_index()
        if "settlement_status" in all_bets
        else pd.Series(dtype=int)
    )
    audit["settlement_status_counts"] = {
        str(status): int(count) for status, count in status_counts.items()
    }
    audit["price_bases"] = list(sealed_eval.PRICE_BASES)
    audit["line_policies"] = list(sealed_eval.LINE_POLICIES)
    audit["sizing_rules"] = list(sealed_eval.SIZINGS)
    audit["ev_gate"] = float(market_eval.EV_GATE)
    audit["books"] = list(market_eval.BOOKS)
    return all_bets, performance, audit


def audit_original_money_reproduction(
    performance: pd.DataFrame, original_path: Path
) -> dict[str, Any]:
    original = pd.read_csv(original_path)
    reference = original.loc[
        original["opportunity_mode"].eq("all_available")
        & original["reliability_policy"].eq("ungated")
        & original["candidate"].isin([CONTROL, SELECTED])
    ].copy()
    mappings = {CONTROL: ARM_CONTROL, SELECTED: ARM_W0_EXISTING}
    reference["candidate"] = reference["candidate"].map(mappings)
    ours = performance.loc[
        performance["sample"].eq("sealed_2023_2026_postunseal")
        & performance["candidate"].isin(mappings.values())
    ].copy()
    keys = ["candidate", "market", "price_basis", "line_policy", "sizing"]
    merged = ours.merge(
        reference,
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_ours", "_reference"),
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("original portfolio summary menus were not reproduced")
    errors: dict[str, float] = {}
    for column in ("bets", "stake", "pnl", "roi", "max_drawdown"):
        error = (
            pd.to_numeric(merged[f"{column}_ours"], errors="coerce")
            - pd.to_numeric(merged[f"{column}_reference"], errors="coerce")
        ).abs()
        errors[column] = float(error.max())
    if errors["bets"] > 0 or max(errors[column] for column in errors if column != "bets") > 1e-9:
        raise ValueError(f"original portfolio summary reproduction failed: {errors}")
    return {"rows": int(len(merged)), "max_abs_errors": errors}


def money_bootstrap(
    bets: pd.DataFrame,
    logical: pd.DataFrame,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    samples = ("sealed_2023_2026_postunseal", "exact_three_year_postunseal")
    for sample in samples:
        start, end = SAMPLES[sample]
        for market in ("72h", "r1"):
            event_universe = sorted(
                logical.loc[
                    logical["candidate"].eq(ARM_W0_EXISTING)
                    & logical["market"].eq(market)
                    & logical["event_start_date"].between(start, end),
                    "mapped_event_key",
                ].unique()
            )
            for price_basis in sealed_eval.PRICE_BASES:
                for line_policy in sealed_eval.LINE_POLICIES:
                    for sizing in sealed_eval.SIZINGS:
                        use = bets.loc[
                            bets["sample"].eq(sample)
                            & bets["round_market"].eq(market)
                            & bets["price_basis"].eq(price_basis)
                            & bets["line_policy"].eq(line_policy)
                            & bets["sizing"].eq(sizing)
                        ]
                        event_pnl = (
                            use.groupby(["candidate", "event_key"], observed=True)["pnl"]
                            .sum()
                            .unstack("candidate", fill_value=0.0)
                            .reindex(event_universe, fill_value=0.0)
                        )
                        for arm in ARMS:
                            if arm not in event_pnl:
                                event_pnl[arm] = 0.0
                        for challenger, reference, label in COMPARISONS:
                            delta = (
                                event_pnl[challenger] - event_pnl[reference]
                            ).to_numpy(float)
                            result = _bootstrap_mean_or_total(
                                delta, draws, seed + len(rows) * 103, total=True
                            )
                            rows.append(
                                {
                                    "sample": sample,
                                    "market": market,
                                    "price_basis": price_basis,
                                    "line_policy": line_policy,
                                    "sizing": sizing,
                                    "comparison": label,
                                    "challenger": challenger,
                                    "reference": reference,
                                    "events": int(len(event_universe)),
                                    "delta_definition": "challenger_minus_reference_higher_is_better",
                                    **result,
                                }
                            )
    return pd.DataFrame(rows)


def _sample_summary_table(
    proper: pd.DataFrame, performance: pd.DataFrame, sample: str
) -> str:
    proper_use = proper.loc[proper["sample"].eq(sample)].pivot(
        index="candidate", columns="market", values="event_equal_log_loss"
    )
    money = performance.loc[
        performance["sample"].eq(sample)
        & performance["price_basis"].eq("close")
        & performance["line_policy"].eq("best_of_three")
        & performance["sizing"].eq("production_risk_to_win")
    ].pivot(index="candidate", columns="market", values=["pnl", "roi"])
    lines = [
        "| Arm | 72h log loss | R1 log loss | 72h P&L / ROI | R1 P&L / ROI |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        def value(frame: pd.DataFrame, column: Any) -> float:
            try:
                return float(frame.loc[arm, column])
            except (KeyError, TypeError, ValueError):
                return np.nan
        ll72 = value(proper_use, "72h")
        llr1 = value(proper_use, "r1")
        p72 = value(money, ("pnl", "72h"))
        r72 = value(money, ("roi", "72h"))
        pr1 = value(money, ("pnl", "r1"))
        rr1 = value(money, ("roi", "r1"))
        lines.append(
            f"| `{arm}` | {ll72:.6f} | {llr1:.6f} | {p72:+.2f}u / {r72:+.2%} | {pr1:+.2f}u / {rr1:+.2%} |"
        )
    return "\n".join(lines)


def write_report(
    path: Path,
    proper: pd.DataFrame,
    performance: pd.DataFrame,
    proper_boot: pd.DataFrame,
    money_boot: pd.DataFrame,
    fitted: pd.DataFrame,
    state_audit: Mapping[str, Any],
    weather_audit: Mapping[str, Any],
    money_audit: Mapping[str, Any],
) -> None:
    exact = proper.loc[proper["sample"].eq("exact_three_year_postunseal")]
    pivot = exact.pivot(index="candidate", columns="market", values="event_equal_log_loss")
    uncertainty_better_both = all(
        float(pivot.loc[ARM_W0_UNCERTAINTY, market])
        < float(pivot.loc[ARM_W0_GLOBAL, market])
        for market in ("72h", "r1")
    )
    weather_better_both = all(
        float(pivot.loc[ARM_W1_GLOBAL, market])
        < float(pivot.loc[ARM_W0_GLOBAL, market])
        for market in ("72h", "r1")
    )
    combined_beats_control_both = all(
        float(pivot.loc[ARM_W1_UNCERTAINTY, market])
        < float(pivot.loc[ARM_CONTROL, market])
        for market in ("72h", "r1")
    )
    material_change = combined_beats_control_both
    conclusion = (
        "The corrections do not overturn the negative conclusion."
        if not material_change
        else "The combined arm beats the strict-PT control on both exact-three-year proper scores, but this remains post-unseal exploratory and requires a new prospective shadow protocol."
    )
    fit_lines = []
    for row in fitted.to_dict(orient="records"):
        fit_lines.append(
            f"- {row['state'].upper()} {row['market']}: nested global sigma {row['global_sigma']:.4f}; uncertainty base sigma {row['uncertainty_base_sigma']:.4f}, alpha {row['uncertainty_alpha']:.6f}; in-sample event-equal multinomial improvement {row['uncertainty_fit_improvement']:.8f}."
        )
    primary_boot = proper_boot.loc[
        proper_boot["sample"].eq("exact_three_year_postunseal")
        & proper_boot["metric"].eq("log_loss")
        & proper_boot["comparison"].isin(
            ["posterior_uncertainty_within_w0", "historical_weather_global_mapper", "combined_corrections", "w1_uncertainty_vs_strict_pt_control"]
        )
    ]
    boot_lines = []
    for row in primary_boot.to_dict(orient="records"):
        boot_lines.append(
            f"- {row['comparison']} / {row['market']}: delta {row['estimate']:+.8f}, 95% whole-event CI [{row['ci_low']:+.8f}, {row['ci_high']:+.8f}]."
        )
    target_weather_loaded = (
        "yes" if weather_audit["target_or_upcoming_weather_loaded"] else "no"
    )
    target_weather_used = (
        "yes"
        if state_audit["target_or_upcoming_realized_weather_feature_used"]
        else "no"
    )
    report = f"""# Post-unseal dynamic Kalman uncertainty/weather diagnostic

{conclusion}

This is a research-only, post-unseal exploratory analysis. It is not fresh
validation, cannot reselect the frozen study, and made no production changes.

## Exact three-year headline

{_sample_summary_table(proper, performance, 'exact_three_year_postunseal')}

## Full sealed-period sensitivity

{_sample_summary_table(proper, performance, 'sealed_2023_2026_postunseal')}

## Frozen 2020-2022 mapper fits

{os.linesep.join(fit_lines)}

The heteroskedastic form was predeclared as
`sqrt(base_sigma^2 + alpha * (v1 + v2))`. Both the nested global fit and the
uncertainty fit use exactly the same 2020-2022 canonical menu. No 2023+ outcome
or price selected the functional form or parameters.

## Whole-event uncertainty

{os.linesep.join(boot_lines)}

## State and weather audits

- W0 posterior mean replay max absolute error: {state_audit['w0_replay_max_abs_error']:.3g}.
- Mean W0 target posterior variance: {state_audit['w0_target_variance_mean']:.6f}; mean W1: {state_audit['w1_target_variance_mean']:.6f}.
- Player-event rows with at least one weather-adjusted prior round: {state_audit['w1_rows_with_weather_history']:,} / {state_audit['rows']:,} ({state_audit['w1_weather_history_coverage_rate']:.1%}).
- Exact all-tour/PGA weather matches: {weather_audit['exact_weather_matches']:,} / {weather_audit['pga_rows']:,}; finite wind+dew: {weather_audit['finite_wind_dew_rows']:,}.
- Non-PGA histories left W0: {weather_audit['non_pga_rows_left_w0']:,}; unmatched PGA rows left W0: {weather_audit['unmatched_pga_rows_left_w0']:,}.
- Weather-panel rows literally loaded: {weather_audit['weather_panel_rows_loaded']:,}; rows on/after the first target date also loaded: {weather_audit['target_or_upcoming_weather_rows_loaded']:,}. Target/upcoming rows used by a state: {state_audit['target_or_upcoming_weather_feature_rows']:,}.
- Positive wind delta means worse-than-field conditions. W1 uses `observed SG + beta * centered exposure`, so it removes the expected penalty.
- Target/upcoming realized weather loaded into the pinned panel: {target_weather_loaded}; used as a feature: {target_weather_used}. `weather_excluded` or another outcome-selected weather flag read: no.
- Materialized selected-bet rows include {money_audit['tie_refunds']:,} tied refunds ({money_audit['voids']:,} `void` statuses and {money_audit['pushes']:,} `push` statuses). These are row occurrences across the declared sample, price, line-policy, and sizing slices.

## Interpretation

- Posterior covariance is real model state, but the original implementation
  discarded it and then priced every pair through one global sigma. This run
  tests that omission directly without changing the selected state mean.
- Pair variance uses the scalar filters' independence approximation
  `v1 + v2`. Common field-adjustment and tour-translation uncertainty is not
  modeled as covariance, so this is a deliberately simple diagnostic rather
  than a complete hierarchical state-space model.
- W1 is a same-family sibling, not a regression-family proxy: q, prior,
  R1-R2 history, tour translations, event batching, and downstream menus are
  unchanged. Only exact historical PGA observations with available centered
  weather are normalized.
- A favorable post-unseal result would be hypothesis generation only. The
  appropriate next step would be frozen prospective shadow deployment with a
  predeclared calibration-refresh schedule—not rewriting the sealed result.
- Archived openers across the three books are asynchronous and the frozen
  model has no exact at-open timestamp. Open-price results are sensitivities,
  not a claim that every displayed opener was executable with that model state.

Diagnostic booleans: W0 uncertainty lower log loss in both exact-three-year
markets = `{uncertainty_better_both}`; W1 global lower than W0 global in both =
`{weather_better_both}`; combined W1 uncertainty lower than strict-PT control
in both = `{combined_beats_control_both}`.

Complete machine-readable results are in `proper_scores.csv`,
`proper_score_bootstrap.csv`, `betting_performance.csv`,
`money_bootstrap.csv`, `fitted_probability_mappers.csv`, `audit.json`, and
`manifest.json` beside this report.
"""
    path.write_text(report, encoding="utf-8")


def _select_frozen_predictions(
    path: Path, smoke_events_per_year: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "candidate",
        "event_key",
        "dg_id_key",
        "dg_id",
        "event_start_date",
        "prediction",
        "reliability_ok",
        "history_rounds",
        "days_since_last_round",
        "prior_event_withdrawal",
    ]
    predictions = pd.read_parquet(
        path,
        columns=columns,
        filters=[("candidate", "in", [SELECTED, CONTROL])],
    )
    predictions["event_start_date"] = pd.to_datetime(
        predictions["event_start_date"], errors="coerce"
    ).dt.normalize()
    predictions["dg_id_key"] = predictions["dg_id_key"].map(market_eval._stable_id)
    predictions = predictions.loc[
        predictions["event_start_date"].between(CALIBRATION_START, STUDY_END)
    ]
    selected = predictions.loc[predictions["candidate"].eq(SELECTED)].copy()
    selected = _smoke_subset(selected, smoke_events_per_year)
    keys = set(selected["event_key"])
    control = predictions.loc[
        predictions["candidate"].eq(CONTROL) & predictions["event_key"].isin(keys)
    ].copy()
    if len(selected) != len(control):
        raise ValueError("selected/control prediction coverage differs")
    return selected, control


def _filter_quotes(path: Path, event_keys: set[str]) -> pd.DataFrame:
    quotes = pd.read_parquet(path)
    quotes["event_start_date"] = pd.to_datetime(
        quotes["event_start_date"], errors="coerce"
    ).dt.normalize()
    quotes = quotes.loc[
        quotes["mapped_event_key"].isin(event_keys)
        & quotes["event_start_date"].between(CALIBRATION_START, STUDY_END)
        & quotes["market"].isin(["72h", "r1"])
        & quotes["book"].isin(market_eval.BOOKS)
    ].copy()
    market_eval.validate_normalized_matchup_quotes(quotes)
    if quotes.empty:
        raise ValueError("filtered exact matchup menu is empty")
    return quotes


def _output_manifest(output_dir: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            outputs[path.name] = _fingerprint(path)
    manifest = {
        "contract": "dynamic_kalman_uncertainty_weather_postunseal_v1",
        "complete": True,
        "research_only": True,
        "post_unseal_exploratory": True,
        "fresh_validation_claimed": False,
        "can_reselect_frozen_study": False,
        "production_files_changed": False,
        "outputs": outputs,
        **payload,
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    manifest["manifest_sha256"] = _sha256(output_dir / "manifest.json")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--all-tour-db", type=Path, default=DEFAULT_ALL_TOUR_DB)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--smoke-events-per-year",
        type=int,
        default=0,
        help="First N events per calendar year; zero runs the complete panel.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory must be fresh: {output_dir}")
    output_dir.mkdir(parents=True)
    if args.all_tour_db.resolve() != DEFAULT_ALL_TOUR_DB.resolve():
        raise ValueError("this diagnostic accepts only the pinned all-tour database")
    authorities = _validate_authorities()
    input_paths = [record[0] for record in EXPECTED.values()]
    provenance = execution_provenance.begin(
        output_dir=output_dir,
        primary_source=Path(__file__).resolve(),
        input_paths=input_paths,
        code_paths=(
            Path(builder.__file__).resolve(),
            Path(market_eval.__file__).resolve(),
            Path(sealed_eval.__file__).resolve(),
            Path(execution_provenance.__file__).resolve(),
            REPO / "DYNAMIC_KALMAN_POST_UNSEAL_DIAGNOSTIC_DESIGN_2026_08.md",
        ),
    )

    snapshots = pd.read_parquet(EXPECTED["augmented_snapshots"][0])
    snapshots["event_start_date"] = pd.to_datetime(
        snapshots["event_start_date"], errors="coerce"
    ).dt.normalize()
    snapshots["event_end_date"] = pd.to_datetime(
        snapshots["event_end_date"], errors="coerce"
    ).dt.normalize()
    selected, control = _select_frozen_predictions(
        EXPECTED["predictions"][0], args.smoke_events_per_year
    )
    rounds = load_all_tour_rounds_with_course(args.all_tour_db)
    rounds, weather_join_audit = attach_exact_pga_weather(
        rounds,
        EXPECTED["round_panel"][0],
        target_dates=selected["event_start_date"],
    )
    coefficients = pd.read_csv(EXPECTED["weather_coefficients"][0])
    states, state_audit = build_dynamic_states(
        snapshots, rounds, selected, coefficients
    )
    predictions = build_prediction_arms(states, selected, control)
    quotes = _filter_quotes(EXPECTED["quotes"][0], set(selected["event_key"]))
    paired = attach_pair_variances(quotes, predictions)
    fitted = fit_mappers(paired)
    frozen_mapper = pd.read_csv(EXPECTED["sealed_mapper"][0])
    scored = score_all_arms(paired, fitted, frozen_mapper)
    logical = canonical_probabilities(scored)
    probability_reproduction = audit_original_probability_reproduction(
        logical, EXPECTED["sealed_logical"][0]
    )

    event_parts = [
        _proper_event_scores(logical, sample) for sample in SAMPLES
    ]
    proper_event = pd.concat(event_parts, ignore_index=True)
    proper = proper_summaries(proper_event)
    proper_boot = proper_bootstrap(
        proper_event, args.bootstrap_draws, args.seed
    )
    bets, performance, money_audit = materialize_money(scored)
    money_reproduction = (
        audit_original_money_reproduction(
            performance, EXPECTED["sealed_performance"][0]
        )
        if args.smoke_events_per_year == 0
        else {"skipped_for_bounded_smoke": True}
    )
    money_boot = money_bootstrap(
        bets, logical, args.bootstrap_draws, args.seed + 1_000_003
    )

    execution_provenance.assert_current(provenance)
    _atomic_parquet(states, output_dir / "player_event_dynamic_states.parquet")
    _atomic_parquet(scored, output_dir / "scored_quote_probabilities.parquet")
    _atomic_parquet(logical, output_dir / "logical_probabilities.parquet")
    _atomic_parquet(bets, output_dir / "selected_bets.parquet")
    _atomic_csv(fitted, output_dir / "fitted_probability_mappers.csv")
    _atomic_csv(proper_event, output_dir / "proper_score_by_event.csv")
    _atomic_csv(proper, output_dir / "proper_scores.csv")
    _atomic_csv(proper_boot, output_dir / "proper_score_bootstrap.csv")
    _atomic_csv(performance, output_dir / "betting_performance.csv")
    _atomic_csv(money_boot, output_dir / "money_bootstrap.csv")
    audit = {
        "contract": "dynamic_kalman_uncertainty_weather_audit_v1",
        "complete": True,
        "research_only": True,
        "post_unseal_exploratory": True,
        "fresh_validation_claimed": False,
        "production_files_changed": False,
        "smoke_events_per_year": int(args.smoke_events_per_year),
        "authorities": authorities,
        "state": state_audit,
        "weather_join": weather_join_audit,
        "probability_reproduction": probability_reproduction,
        "money_reproduction": money_reproduction,
        "money": money_audit,
        "quote_rows": int(len(quotes)),
        "canonical_logical_rows": int(len(logical)),
        "quote_menu_equal_across_arms": True,
        "settlement_ungraded_selected_bets": 0,
        "closing_price_used_as_model_feature": False,
        "target_or_upcoming_realized_weather_used_as_feature": bool(
            state_audit["target_or_upcoming_realized_weather_feature_used"]
        ),
        "weather_excluded_read": False,
        "calibration_fit_start": CALIBRATION_START,
        "calibration_fit_end": CALIBRATION_END,
        "sealed_outcomes_used_in_parameter_fit": False,
        "bootstrap_draws": int(args.bootstrap_draws),
        "bootstrap_unit": "whole_event",
    }
    _atomic_json(output_dir / "audit.json", audit)
    write_report(
        output_dir / "POST_UNSEAL_DYNAMIC_KALMAN_REPORT.md",
        proper,
        performance,
        proper_boot,
        money_boot,
        fitted,
        state_audit,
        weather_join_audit,
        money_audit,
    )
    provenance_payload = execution_provenance.finalize(provenance)
    manifest = _output_manifest(
        output_dir,
        {
            "created_at_utc": pd.Timestamp.utcnow().isoformat(),
            "inputs": authorities,
            "execution_provenance": provenance_payload,
            "configuration": {
                "q_per_day": PROCESS_VARIANCE_PER_DAY,
                "prior_rounds": PRIOR_ROUNDS,
                "calibration_start": CALIBRATION_START,
                "calibration_end": CALIBRATION_END,
                "sealed_start": SEALED_START,
                "exact_three_year_start": EXACT_THREE_YEAR_START,
                "study_end": STUDY_END,
                "bootstrap_draws": int(args.bootstrap_draws),
                "seed": int(args.seed),
                "smoke_events_per_year": int(args.smoke_events_per_year),
            },
            "audits": {
                "state": state_audit,
                "weather_join": weather_join_audit,
                "probability_reproduction": probability_reproduction,
                "money_reproduction": money_reproduction,
                "money": money_audit,
            },
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "manifest_sha256": manifest["manifest_sha256"],
                "state_rows": int(len(states)),
                "logical_rows": int(len(logical)),
                "selected_bets": int(len(bets)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
