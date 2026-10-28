"""Whole-event and multiple-search inference for the base-skill discovery screen.

Research only.  This consumes the completed 2021-22 logical probability panel;
it never reads the sealed period and cannot choose or promote a new candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from archive import research_execution_provenance as execution_provenance


ROOT = REPO / "permanent_data" / "research" / "base_skill_rebuild_20200101_20260822"
DISCOVERY = ROOT / "model_eval_complete_discovery_v5_provenance"
DEFAULT_LOGICAL = DISCOVERY / "discovery_crossfit_logical_probabilities.parquet"
DEFAULT_SCORECARD = DISCOVERY / "candidate_gate_scorecard.csv"
DEFAULT_SELECTION = DISCOVERY / "selection_manifest.json"
DEFAULT_OUTPUT = ROOT / "model_eval_complete_discovery_v5_search_correction_v2"
BASELINE = "current_architecture_pt_v2_approx_control"
DISCOVERY_START = pd.Timestamp("2021-01-01")
DISCOVERY_END = pd.Timestamp("2022-12-31")
MARKETS = ("72h", "r1")
EPS = 1e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def validate_completed_discovery(
    logical_path: Path,
    scorecard_path: Path,
    selection_path: Path,
) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("complete") is not True:
        raise ValueError("selection manifest is incomplete")
    if selection.get("selection_uses_sealed") is not False:
        raise ValueError("selection manifest does not exclude sealed selection")
    if selection.get("sealed_data_used_for_selection") is not False:
        raise ValueError("selection manifest reports sealed selection data")
    if selection.get("selection_window_end") != DISCOVERY_END.date().isoformat():
        raise ValueError("selection manifest has the wrong discovery cutoff")
    artifacts = {
        str(record.get("path")): record
        for record in selection.get("artifacts", [])
        if isinstance(record, Mapping)
    }
    for path in (logical_path, scorecard_path):
        try:
            relative = path.resolve().relative_to(selection_path.resolve().parent).as_posix()
        except ValueError as exc:
            raise ValueError(f"input is outside the completed discovery directory: {path}") from exc
        record = artifacts.get(relative)
        if not isinstance(record, Mapping):
            raise ValueError(f"selection manifest does not authorize {relative}")
        if int(record.get("bytes", -1)) != path.stat().st_size:
            raise ValueError(f"authorized bytes differ for {relative}")
        if str(record.get("sha256", "")).lower() != _sha256(path):
            raise ValueError(f"authorized hash differs for {relative}")
    return selection


def build_event_loss_matrix(
    logical: pd.DataFrame,
    candidate_names: Sequence[str],
    baseline: str = BASELINE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return event-market and event-pooled conditional log-loss deltas."""

    required = {
        "candidate",
        "mapped_event_key",
        "event_start_date",
        "market",
        "pair_key",
        "canonical_outcome",
        "canonical_probability",
        "canonical_push_probability",
    }
    _require_columns(logical, required, "logical probabilities")
    work = logical.copy()
    key_columns = [
        "candidate",
        "mapped_event_key",
        "event_start_date",
        "market",
        "pair_key",
    ]
    if work[key_columns].isna().any().any():
        raise ValueError("logical panel contains null canonical keys")
    for column in ("candidate", "mapped_event_key", "market", "pair_key"):
        work[column] = work[column].astype(str)
        if work[column].str.strip().eq("").any():
            raise ValueError(f"logical panel contains blank {column} values")
    work["event_start_date"] = pd.to_datetime(
        work["event_start_date"], errors="raise"
    ).dt.normalize()
    if work["event_start_date"].isna().any():
        raise ValueError("logical panel contains null event dates")
    if not work["event_start_date"].between(DISCOVERY_START, DISCOVERY_END).all():
        raise ValueError("logical panel contains a non-discovery event")
    event_dates = work.groupby("mapped_event_key", observed=True)[
        "event_start_date"
    ].nunique()
    if not event_dates.eq(1).all():
        raise ValueError("an event key maps to multiple event dates")
    if work.duplicated(
        ["candidate", "mapped_event_key", "market", "pair_key"]
    ).any():
        raise ValueError("logical panel contains duplicate candidate/pair keys")
    requested = sorted(set(map(str, candidate_names)))
    if not requested:
        raise ValueError("at least one challenger is required")
    if baseline in requested:
        raise ValueError("the baseline cannot also be a challenger")
    available = set(work["candidate"].astype(str))
    missing = sorted((set(requested) | {baseline}) - available)
    if missing:
        raise ValueError(f"logical panel is missing candidates: {missing}")

    keys = ["mapped_event_key", "event_start_date", "market", "pair_key"]
    panel = work.loc[work["candidate"].isin([baseline, *requested])].copy()
    numeric_columns = [
        "canonical_outcome",
        "canonical_probability",
        "canonical_push_probability",
    ]
    for column in numeric_columns:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    numeric = panel[numeric_columns].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("logical panel contains non-finite outcomes or probabilities")
    valid_outcome = panel["canonical_outcome"].isin([0.0, 0.5, 1.0])
    probability_bad = (
        ~panel["canonical_probability"].between(0.0, 1.0, inclusive="both")
        | ~panel["canonical_push_probability"].between(
            0.0, 1.0, inclusive="both"
        )
        | (
            panel["canonical_probability"]
            + panel["canonical_push_probability"]
        ).gt(1.0 + 1e-10)
    )
    if (~valid_outcome).any() or probability_bad.any():
        raise ValueError("logical panel contains out-of-domain outcomes or probabilities")
    if (
        panel["canonical_outcome"].isin([0.0, 1.0])
        & (1.0 - panel["canonical_push_probability"]).le(EPS)
    ).any():
        raise ValueError("logical panel has zero resolved mass for a resolved outcome")

    baseline_frame = panel.loc[panel["candidate"].eq(baseline), keys + [
        "canonical_outcome",
        "canonical_probability",
        "canonical_push_probability",
    ]].rename(columns={
        "canonical_outcome": "baseline_outcome",
        "canonical_probability": "baseline_probability",
        "canonical_push_probability": "baseline_push_probability",
    })
    if baseline_frame.empty:
        raise ValueError("baseline canonical menu is empty")
    baseline_menu = pd.MultiIndex.from_frame(baseline_frame[keys])
    if baseline_menu.has_duplicates:
        raise ValueError("baseline canonical menu contains duplicate keys")
    event_market_parts: list[pd.DataFrame] = []
    for candidate in requested:
        challenger = panel.loc[panel["candidate"].eq(candidate), keys + [
            "canonical_outcome",
            "canonical_probability",
            "canonical_push_probability",
        ]].rename(columns={
            "canonical_outcome": "candidate_outcome",
            "canonical_probability": "candidate_probability",
            "canonical_push_probability": "candidate_push_probability",
        })
        challenger_menu = pd.MultiIndex.from_frame(challenger[keys])
        baseline_only = baseline_menu.difference(challenger_menu)
        challenger_only = challenger_menu.difference(baseline_menu)
        if len(baseline_only) or len(challenger_only):
            raise ValueError(
                "candidate does not share the exact baseline menu: "
                f"{candidate}; baseline_only={len(baseline_only)}; "
                f"challenger_only={len(challenger_only)}"
            )
        common = baseline_frame.merge(
            challenger, on=keys, how="outer", validate="one_to_one", indicator=True
        )
        if not common["_merge"].eq("both").all():
            raise ValueError(f"candidate menu outer merge is not exact: {candidate}")
        common = common.drop(columns="_merge")
        if not np.array_equal(
            common["baseline_outcome"].to_numpy(float),
            common["candidate_outcome"].to_numpy(float),
        ):
            raise ValueError(f"outcomes disagree for {candidate}")
        common = common.loc[common["candidate_outcome"].isin([0.0, 1.0])].copy()
        if common.empty:
            raise ValueError(f"candidate has no resolved outcomes: {candidate}")
        y = common["candidate_outcome"].to_numpy(float)
        candidate_probability = np.clip(
            common["candidate_probability"].to_numpy(float)
            / (1.0 - common["candidate_push_probability"].to_numpy(float)),
            EPS,
            1.0 - EPS,
        )
        baseline_probability = np.clip(
            common["baseline_probability"].to_numpy(float)
            / (1.0 - common["baseline_push_probability"].to_numpy(float)),
            EPS,
            1.0 - EPS,
        )
        common["loss_delta"] = -(
            y * np.log(candidate_probability)
            + (1.0 - y) * np.log(1.0 - candidate_probability)
        ) + (
            y * np.log(baseline_probability)
            + (1.0 - y) * np.log(1.0 - baseline_probability)
        )
        if not np.isfinite(common["loss_delta"].to_numpy(float)).all():
            raise ValueError(f"candidate produces non-finite loss deltas: {candidate}")
        part = common.groupby(
            ["mapped_event_key", "event_start_date", "market"],
            observed=True,
            as_index=False,
        )["loss_delta"].mean()
        part["candidate"] = candidate
        event_market_parts.append(part)
    event_market = pd.concat(event_market_parts, ignore_index=True)
    if set(event_market["market"].astype(str)) != set(MARKETS):
        raise ValueError("discovery market set differs from the frozen two-market set")
    market_counts = event_market.groupby(
        ["candidate", "mapped_event_key"], observed=True
    )["market"].nunique()
    if not market_counts.eq(len(MARKETS)).all():
        raise ValueError("an event/candidate does not contain both frozen markets")
    pooled = event_market.groupby(
        ["candidate", "mapped_event_key", "event_start_date"],
        observed=True,
        as_index=False,
    )["loss_delta"].mean()
    if not np.isfinite(pooled["loss_delta"].to_numpy(float)).all():
        raise ValueError("pooled event loss deltas are non-finite")
    return event_market, pooled


def _bootstrap_means(
    matrix: np.ndarray,
    draws: int,
    rng: np.random.Generator,
    *,
    block_length: int = 1,
    batch_size: int = 500,
) -> np.ndarray:
    events, candidates = matrix.shape
    if events < 2 or candidates < 1 or draws < 1:
        raise ValueError("bootstrap dimensions must be positive")
    if block_length < 1 or block_length > events:
        raise ValueError("invalid block length")
    output = np.empty((draws, candidates), dtype=float)
    offsets = np.arange(block_length, dtype=int)
    for start in range(0, draws, batch_size):
        stop = min(start + batch_size, draws)
        size = stop - start
        if block_length == 1:
            weights = rng.multinomial(
                events, np.full(events, 1.0 / events), size=size
            )
            output[start:stop] = weights @ matrix / events
            continue
        blocks = int(math.ceil(events / block_length))
        starts = rng.integers(0, events, size=(size, blocks))
        indices = (starts[..., None] + offsets) % events
        indices = indices.reshape(size, -1)[:, :events]
        output[start:stop] = matrix[indices].mean(axis=1)
    return output


def _cluster_block_bootstrap_means(
    matrix: np.ndarray,
    cluster_labels: Sequence[Any],
    draws: int,
    rng: np.random.Generator,
    *,
    block_length: int = 1,
    batch_size: int = 500,
) -> np.ndarray:
    """Resample complete ordered clusters, optionally in circular cluster blocks.

    Every occurrence of a sampled cluster contributes all of its event rows with
    a common multiplicity.  Thus simultaneous events in the same calendar week
    can never be split across bootstrap samples.
    """

    events, candidates = matrix.shape
    labels = pd.Series(list(cluster_labels), dtype="object")
    if events < 2 or candidates < 1 or draws < 1:
        raise ValueError("cluster bootstrap dimensions must be positive")
    if len(labels) != events or labels.isna().any():
        raise ValueError("cluster labels must be complete and event-aligned")
    if not np.isfinite(matrix).all():
        raise ValueError("cluster bootstrap matrix must be finite")
    unique_labels = list(dict.fromkeys(labels.tolist()))
    cluster_count = len(unique_labels)
    if cluster_count < 2:
        raise ValueError("cluster bootstrap requires at least two clusters")
    if block_length < 1 or block_length > cluster_count:
        raise ValueError("invalid cluster block length")
    positions = {label: index for index, label in enumerate(unique_labels)}
    numeric_labels = labels.map(positions).to_numpy(int)
    # Chronological callers order by event date.  Reject a label that disappears
    # and later reappears, because circular blocks require contiguous clusters.
    transitions = np.r_[True, numeric_labels[1:] != numeric_labels[:-1]]
    if len(numeric_labels[transitions]) != cluster_count:
        raise ValueError("cluster labels must form contiguous ordered groups")
    cluster_sums = np.vstack(
        [matrix[numeric_labels == index].sum(axis=0) for index in range(cluster_count)]
    )
    cluster_sizes = np.asarray(
        [np.count_nonzero(numeric_labels == index) for index in range(cluster_count)],
        dtype=float,
    )
    output = np.empty((draws, candidates), dtype=float)
    offsets = np.arange(block_length, dtype=int)
    blocks = int(math.ceil(cluster_count / block_length))
    for start in range(0, draws, batch_size):
        stop = min(start + batch_size, draws)
        size = stop - start
        starts = rng.integers(0, cluster_count, size=(size, blocks))
        selected = (starts[..., None] + offsets) % cluster_count
        selected = selected.reshape(size, -1)[:, :cluster_count]
        numerator = cluster_sums[selected].sum(axis=1)
        denominator = cluster_sizes[selected].sum(axis=1)[:, None]
        output[start:stop] = numerator / denominator
    if not np.isfinite(output).all():
        raise ValueError("cluster bootstrap produced non-finite means")
    return output


def _null_centered_search_statistics(
    bootstrap: np.ndarray,
    observed: np.ndarray,
    *,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Compute lower-tail tests and a simultaneous upper confidence band.

    Loss deltas are challenger minus baseline, so negative is improvement.  The
    bootstrap null distribution is formed by centering each tried candidate at
    zero.  The simultaneous upper increment uses max(-centered), the bootstrap
    analogue of max(true - estimate), rather than max(centered).
    """

    sample = np.asarray(bootstrap, dtype=float)
    estimate = np.asarray(observed, dtype=float)
    if sample.ndim != 2 or estimate.ndim != 1 or sample.shape[1] != len(estimate):
        raise ValueError("bootstrap and observed dimensions do not conform")
    if sample.shape[0] < 1 or sample.shape[1] < 1:
        raise ValueError("search correction requires nonempty bootstrap draws")
    if not np.isfinite(sample).all() or not np.isfinite(estimate).all():
        raise ValueError("search correction inputs must be finite")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    centered = sample - estimate[None, :]
    draws = sample.shape[0]
    best_index = int(np.argmin(estimate))
    individual_p = (
        1.0 + np.count_nonzero(centered <= estimate[None, :], axis=0)
    ) / (draws + 1.0)
    observed_best = float(estimate[best_index])
    white_p = float(
        (1.0 + np.count_nonzero(centered.min(axis=1) <= observed_best))
        / (draws + 1.0)
    )
    upper_error = (-centered).max(axis=1)
    upper_increment = float(
        np.quantile(upper_error, confidence, method="higher")
    )
    upper_bounds = estimate + upper_increment
    return {
        "best_index": best_index,
        "centered": centered,
        "individual_null_centered_one_sided_p": individual_p,
        "white_reality_check_null_centered_one_sided_p": white_p,
        "simultaneous_upper_increment": upper_increment,
        "simultaneous_upper_bounds": upper_bounds,
        "familywise_improvement_certified": upper_bounds < 0.0,
    }


def _validated_event_matrix(
    pooled: pd.DataFrame,
) -> tuple[list[str], pd.DataFrame, np.ndarray]:
    required = {
        "candidate",
        "mapped_event_key",
        "event_start_date",
        "loss_delta",
    }
    _require_columns(pooled, required, "pooled event losses")
    work = pooled[list(required)].copy()
    if work[["candidate", "mapped_event_key", "event_start_date"]].isna().any().any():
        raise ValueError("pooled event losses contain null keys")
    for column in ("candidate", "mapped_event_key"):
        work[column] = work[column].astype(str)
        if work[column].str.strip().eq("").any():
            raise ValueError(f"pooled event losses contain blank {column} values")
    work["event_start_date"] = pd.to_datetime(
        work["event_start_date"], errors="raise"
    ).dt.normalize()
    if work["event_start_date"].isna().any():
        raise ValueError("pooled event losses contain null event dates")
    if not work["event_start_date"].between(DISCOVERY_START, DISCOVERY_END).all():
        raise ValueError("pooled event losses contain a non-discovery event")
    work["loss_delta"] = pd.to_numeric(work["loss_delta"], errors="coerce")
    if not np.isfinite(work["loss_delta"].to_numpy(float)).all():
        raise ValueError("pooled event loss deltas must all be finite")
    if work.duplicated(["candidate", "mapped_event_key"]).any():
        raise ValueError("pooled event losses contain duplicate candidate/event keys")
    event_dates = work.groupby("mapped_event_key", observed=True)[
        "event_start_date"
    ].nunique()
    if not event_dates.eq(1).all():
        raise ValueError("a pooled event key maps to multiple event dates")
    candidates = sorted(work["candidate"].unique().tolist())
    if not candidates:
        raise ValueError("pooled event losses contain no candidates")
    event_meta = work[["mapped_event_key", "event_start_date"]].drop_duplicates()
    event_meta = event_meta.sort_values(
        ["event_start_date", "mapped_event_key"], kind="mergesort"
    ).reset_index(drop=True)
    if len(event_meta) < 2:
        raise ValueError("search correction requires at least two events")
    expected_menu = pd.MultiIndex.from_frame(event_meta)
    for candidate in candidates:
        candidate_menu = pd.MultiIndex.from_frame(
            work.loc[
                work["candidate"].eq(candidate),
                ["mapped_event_key", "event_start_date"],
            ]
        )
        expected_only = expected_menu.difference(candidate_menu)
        challenger_only = candidate_menu.difference(expected_menu)
        if len(expected_only) or len(challenger_only):
            raise ValueError(
                "event/candidate loss matrix is not exactly rectangular: "
                f"{candidate}; expected_only={len(expected_only)}; "
                f"challenger_only={len(challenger_only)}"
            )
    events = event_meta["mapped_event_key"].tolist()
    pivot = work.pivot(
        index="mapped_event_key", columns="candidate", values="loss_delta"
    ).reindex(index=events, columns=candidates)
    if pivot.isna().any().any():
        raise ValueError("event/candidate loss matrix is not rectangular")
    matrix = pivot.to_numpy(float)
    if not np.isfinite(matrix).all():
        raise ValueError("event/candidate loss matrix contains non-finite values")
    return candidates, event_meta, matrix


def _resampling_search_audit(
    bootstrap: np.ndarray,
    observed: np.ndarray,
    candidates: Sequence[str],
    *,
    method: str,
    block_length: int,
    block_unit: str,
) -> dict[str, Any]:
    stats = _null_centered_search_statistics(bootstrap, observed)
    best_index = int(stats["best_index"])
    upper_bounds = np.asarray(stats["simultaneous_upper_bounds"], dtype=float)
    individual_p = np.asarray(
        stats["individual_null_centered_one_sided_p"], dtype=float
    )
    return {
        "status": "complete",
        "bootstrap_method": method,
        "block_length": int(block_length),
        "block_unit": block_unit,
        "candidate": str(candidates[best_index]),
        "ci_low": float(np.quantile(bootstrap[:, best_index], 0.025)),
        "ci_high": float(np.quantile(bootstrap[:, best_index], 0.975)),
        "best_candidate_null_centered_one_sided_p_improvement": float(
            individual_p[best_index]
        ),
        "white_reality_check_null_centered_one_sided_p": float(
            stats["white_reality_check_null_centered_one_sided_p"]
        ),
        "simultaneous_95pct_upper_increment": float(
            stats["simultaneous_upper_increment"]
        ),
        "best_candidate_simultaneous_95pct_upper_bound": float(
            upper_bounds[best_index]
        ),
        "any_familywise_improvement_certified": bool(
            np.any(stats["familywise_improvement_certified"])
        ),
        "candidate_null_centered_one_sided_p_improvement": {
            str(candidate): float(individual_p[index])
            for index, candidate in enumerate(candidates)
        },
        "candidate_simultaneous_95pct_upper_bound": {
            str(candidate): float(upper_bounds[index])
            for index, candidate in enumerate(candidates)
        },
    }


def search_corrected_inference(
    pooled: pd.DataFrame,
    *,
    draws: int = 10_000,
    seed: int = 20260823,
    block_lengths: Sequence[int] = (1, 2, 3, 4),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidates, event_meta, matrix = _validated_event_matrix(pooled)
    events = event_meta["mapped_event_key"].tolist()
    observed = matrix.mean(axis=0)
    rng = np.random.default_rng(seed)
    iid = _bootstrap_means(matrix, draws, rng, block_length=1)
    iid_stats = _null_centered_search_statistics(iid, observed)
    best_index = int(iid_stats["best_index"])
    observed_best = float(observed[best_index])
    individual_p = np.asarray(
        iid_stats["individual_null_centered_one_sided_p"], dtype=float
    )
    simultaneous_upper_bounds = np.asarray(
        iid_stats["simultaneous_upper_bounds"], dtype=float
    )
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        sample = iid[:, index]
        rows.append(
            {
                "candidate": candidate,
                "events": len(events),
                "observed_pooled_log_loss_delta": float(observed[index]),
                "iid_ci_low": float(np.quantile(sample, 0.025)),
                "iid_ci_high": float(np.quantile(sample, 0.975)),
                "iid_null_centered_one_sided_p_improvement": float(
                    individual_p[index]
                ),
                "simultaneous_95pct_upper_bound": float(
                    simultaneous_upper_bounds[index]
                ),
                "familywise_improvement_certified": bool(
                    iid_stats["familywise_improvement_certified"][index]
                ),
            }
        )
    event_block_audit: dict[str, Any] = {}
    for block_length in block_lengths:
        if int(block_length) < 1 or int(block_length) > len(events):
            raise ValueError("invalid event block length")
        block_rng = np.random.default_rng(seed + 1009 * int(block_length))
        boot = _bootstrap_means(
            matrix, draws, block_rng, block_length=int(block_length)
        )
        event_block_audit[str(block_length)] = _resampling_search_audit(
            boot,
            observed,
            candidates,
            method=("whole_event_iid" if int(block_length) == 1 else "circular_moving_block"),
            block_length=int(block_length),
            block_unit="event",
        )

    week_start = event_meta["event_start_date"] - pd.to_timedelta(
        event_meta["event_start_date"].dt.weekday, unit="D"
    )
    week_labels = week_start.dt.strftime("%Y-%m-%d").tolist()
    week_sizes = pd.Series(week_labels).value_counts(sort=False)
    week_count = int(len(week_sizes))
    calendar_week_audit: dict[str, Any] = {}
    for block_length in block_lengths:
        length = int(block_length)
        if length < 1:
            raise ValueError("invalid calendar-week block length")
        if length > week_count:
            calendar_week_audit[str(length)] = {
                "status": "block_larger_than_calendar_week_count",
                "bootstrap_method": "calendar_week_circular_block",
                "block_length": length,
                "block_unit": "calendar_week",
            }
            continue
        week_rng = np.random.default_rng(seed + 1_000_003 + 2003 * length)
        week_boot = _cluster_block_bootstrap_means(
            matrix,
            week_labels,
            draws,
            week_rng,
            block_length=length,
        )
        calendar_week_audit[str(length)] = _resampling_search_audit(
            week_boot,
            observed,
            candidates,
            method=(
                "calendar_week_cluster"
                if length == 1
                else "calendar_week_circular_block"
            ),
            block_length=length,
            block_unit="calendar_week",
        )
    audit = {
        "events": len(events),
        "candidates": len(candidates),
        "draws": int(draws),
        "seed": int(seed),
        "best_candidate": candidates[best_index],
        "best_observed_pooled_log_loss_delta": observed_best,
        "white_reality_check_null_centered_one_sided_p": float(
            iid_stats["white_reality_check_null_centered_one_sided_p"]
        ),
        "simultaneous_95pct_upper_increment": float(
            iid_stats["simultaneous_upper_increment"]
        ),
        "any_familywise_improvement_certified": bool(
            any(row["familywise_improvement_certified"] for row in rows)
        ),
        "moving_event_block_sensitivity": event_block_audit,
        "calendar_week_clusters": week_count,
        "calendar_weeks_with_multiple_events": int(week_sizes.gt(1).sum()),
        "maximum_events_in_calendar_week": int(week_sizes.max()),
        "same_calendar_week_events_resampled_as_indivisible_clusters": True,
        "calendar_week_block_sensitivity": calendar_week_audit,
    }
    return pd.DataFrame(rows), audit


def run(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    logical_path = args.logical.resolve()
    scorecard_path = args.scorecard.resolve()
    selection_path = args.selection.resolve()
    validate_completed_discovery(logical_path, scorecard_path, selection_path)
    provenance = execution_provenance.begin(
        output_dir=output,
        primary_source=Path(__file__),
        input_paths=[logical_path, scorecard_path, selection_path],
        code_paths=[Path(execution_provenance.__file__).resolve()],
    )
    logical = pd.read_parquet(logical_path)
    scorecard = pd.read_csv(scorecard_path)
    candidate_names = sorted(scorecard["candidate"].astype(str).unique())
    event_market, pooled = build_event_loss_matrix(logical, candidate_names)
    inference, audit = search_corrected_inference(
        pooled,
        draws=args.draws,
        seed=args.seed,
        block_lengths=args.block_lengths,
    )
    market_summary = event_market.groupby(
        ["candidate", "market"], observed=True, as_index=False
    )["loss_delta"].mean().pivot(
        index="candidate", columns="market", values="loss_delta"
    ).reset_index()
    market_summary.columns.name = None
    inference = inference.merge(
        market_summary.rename(
            columns={market: f"observed_{market}_log_loss_delta" for market in MARKETS}
        ),
        on="candidate",
        how="left",
        validate="one_to_one",
    ).sort_values("observed_pooled_log_loss_delta", kind="mergesort")
    inference.to_csv(output / "candidate_search_corrected_inference.csv", index=False)
    event_market.to_parquet(output / "event_market_loss_deltas.parquet", index=False)
    _atomic_json(output / "search_correction_audit.json", audit)
    lines = [
        "# Discovery candidate-search correction",
        "",
        "Research only. This analysis uses 2021-22 discovery events only and cannot promote a model.",
        "",
        f"Best raw candidate: `{audit['best_candidate']}`",
        f"Raw pooled log-loss delta: {audit['best_observed_pooled_log_loss_delta']:.8f}",
        f"Null-centered White reality-check one-sided p-value across {audit['candidates']} tried candidates: {audit['white_reality_check_null_centered_one_sided_p']:.4f}",
        f"Any familywise 95% improvement certified: {audit['any_familywise_improvement_certified']}",
        "",
        "Negative log-loss delta favors the challenger. Whole canonical events, including both frozen markets, are the primary resampling unit. Calendar-week cluster and circular week-block sensitivities keep simultaneous events together. All one-sided tests use null-centered bootstrap distributions; simultaneous upper bands use q95(max(-centered)).",
        "",
        "```text",
        inference.head(15).to_string(index=False, float_format=lambda value: f"{value:.8f}"),
        "```",
        "",
    ]
    (output / "DISCOVERY_SEARCH_CORRECTION.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    execution_audit = execution_provenance.finalize(provenance)
    artifacts = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            artifacts.append(
                {
                    "path": path.relative_to(output).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    manifest = {
        "contract": "base_skill_discovery_search_correction_v2",
        "complete": True,
        "research_only": True,
        "selection_period": "2021-01-01/2022-12-31",
        "sealed_data_used": False,
        "cannot_promote_or_reselect": True,
        "draws": int(args.draws),
        "seed": int(args.seed),
        "block_lengths": list(map(int, args.block_lengths)),
        "audit": audit,
        "execution_provenance": {
            "path": "execution_provenance/execution_provenance.json",
            "verified_unchanged_before_complete_manifest": execution_audit[
                "verified_unchanged_before_complete_manifest"
            ],
        },
        "inputs": {
            str(path.name): {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in (logical_path, scorecard_path, selection_path)
        },
        "artifacts": artifacts,
    }
    execution_provenance.assert_current(provenance)
    _atomic_json(output / "manifest.json", manifest)
    print(f"Completed discovery search correction: {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logical", type=Path, default=DEFAULT_LOGICAL)
    parser.add_argument("--scorecard", type=Path, default=DEFAULT_SCORECARD)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--block-lengths",
        type=lambda value: tuple(int(item) for item in value.split(",") if item),
        default=(1, 2, 3, 4),
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
