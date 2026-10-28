"""Freeze strict pre-event production inputs for transition-level mean anchoring.

This research-only exporter reconstructs the deterministic category means used
by the frozen category-first backtest, then keeps them separate from the
strict-prior category-ability vector used by the learned transition-quality
tilt.  No held-out scores or realized field scoring centers are read.

The two vectors have different jobs:

* ``target_*_sg`` is the frozen backtest mean a research chain must reproduce;
* ``ability_*_sg`` controls the transferred beta quality tilt before the joint
  lambda correction is solved.

The category target follows the production rule exactly: calculate the
50-round EMA category means from completed rounds strictly before the event,
apply the established-player field fallback for cells with fewer than twelve
rounds, and shift all four categories equally so they sum to the frozen total
skill prediction.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import sys
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
ARCHIVE = Path(__file__).resolve().parent
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

import hole_sim_transition_mean_chain_adapter as chain_adapter  # noqa: E402


CONTRACT = "frozen_backtest_category_anchor_inputs_v1"
CATEGORY_RECONSTRUCTION = "category_first_ema50_field_prior_equal_recenter_v1"
ABILITY_SNAPSHOT = "model_results2_strict_prior_four_category_v1"
CATEGORIES = ("ott", "app", "arg", "putt")
ROUND_SOURCE_COLUMNS = {
    "ott": "sg_ott_adj",
    "app": "sg_app_adj",
    "arg": "sg_arg_adj",
    "putt": "sg_putt_adj",
}
ABILITY_SOURCE_COLUMNS = {
    "ott": "sg_ott",
    "app": "sg_app",
    "arg": "ag_arg",
    "putt": "ag_putt",
}
EMA_SPAN = 50
MINIMUM_CATEGORY_ROUNDS = 12
CATEGORY_CLIP = (-8.0, 8.0)
DEFAULT_DB = Path(r"C:\Users\McKinley Slade\dev\sim_prep\work\dg_historical.db")
DEFAULT_BASELINE = REPO / "permanent_data" / "spatial_chain_common_baselines" / "round_case_pmfs.parquet"
DEFAULT_OUTPUT = REPO / "permanent_data" / "hole_sim_backtest_anchor_inputs_travelers_v2"


class ProductionAnchorInputUnavailable(ValueError):
    """Raised when a target cannot be frozen without leakage or imputation."""


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_sha(payload: object) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return sha256(raw).hexdigest()


def _number(value: object) -> str:
    number = float(value)
    if not np.isfinite(number):
        raise ProductionAnchorInputUnavailable("Canonical input contains non-finite numeric data")
    if number == 0.0:
        number = 0.0
    return format(number, ".17g")


def ema_mean(values: Sequence[float], span: int = EMA_SPAN) -> float:
    """Production-compatible exponentially weighted category mean."""

    array = np.clip(np.asarray(values, dtype=float), *CATEGORY_CLIP)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan")
    alpha = 2.0 / (float(span) + 1.0)
    weights = (1.0 - alpha) ** np.arange(len(array) - 1, -1, -1, dtype=float)
    weights /= weights.sum()
    return float(weights @ array)


def canonical_event_key(event_start: object, source_event_key: str) -> str:
    """Convert the backtest composite key to the ShotLink event-key contract."""

    pieces = str(source_event_key).split("|", 2)
    if len(pieces) != 3:
        raise ProductionAnchorInputUnavailable("Baseline event key is not date|event_id|name")
    try:
        event_id = int(pieces[1])
    except ValueError as exc:
        raise ProductionAnchorInputUnavailable("Baseline event ID is not an integer") from exc
    year = pd.Timestamp(event_start).year
    return f"pga:R{year}{event_id:03d}"


def _frozen_event_cases(
    baseline_path: Path,
    event_name: str,
    event_start: object | None = None,
) -> pd.DataFrame:
    # Deliberately omit actual_strokes, actual_relative and
    # field_scoring_center.  They are held-out evaluation data, not inputs.
    columns = [
        "event_key",
        "event",
        "event_start",
        "dg_id",
        "source_player_id",
        "player_name",
        "round_num",
        "arm",
        "target_sg",
        "skill_date",
    ]
    frame = pd.read_parquet(baseline_path, columns=columns)
    mask = frame["event"].astype(str).str.casefold().eq(str(event_name).casefold())
    if event_start is not None:
        mask &= pd.to_datetime(frame["event_start"]).eq(pd.Timestamp(event_start))
    frame = frame.loc[mask].copy()
    if frame.empty:
        raise ProductionAnchorInputUnavailable("No frozen baseline cases match the event")
    keys = ["event_key", "dg_id", "source_player_id", "round_num"]
    for _, group in frame.groupby(keys, observed=True, sort=False):
        if group["target_sg"].nunique(dropna=False) != 1:
            raise ProductionAnchorInputUnavailable("Frozen total skill changes across PMF arms")
        if group["event_start"].nunique(dropna=False) != 1:
            raise ProductionAnchorInputUnavailable("Event start changes across PMF arms")
    cases = (
        frame.sort_values([*keys, "arm"], kind="mergesort")
        .drop_duplicates(keys)
        .drop(columns="arm")
        .reset_index(drop=True)
    )
    if cases.duplicated(["dg_id", "round_num"]).any():
        raise ProductionAnchorInputUnavailable("Event has duplicate player-round targets")
    rounds = sorted(cases["round_num"].astype(int).unique().tolist())
    if rounds != [1, 2, 3, 4]:
        raise ProductionAnchorInputUnavailable(f"Expected rounds 1-4; got {rounds}")
    return cases


def _load_prior_rounds(
    db_path: Path,
    dg_ids: Sequence[int],
    event_start: object,
) -> pd.DataFrame:
    placeholders = ",".join("?" for _ in dg_ids)
    columns = ", ".join(ROUND_SOURCE_COLUMNS.values())
    query = f"""
        SELECT dg_id, round_date, {columns}
        FROM player_rounds
        WHERE tour='pga'
          AND sg_total_adj IS NOT NULL
          AND round_date >= '2019-01-01'
          AND round_date < ?
          AND dg_id IN ({placeholders})
    """
    params = [pd.Timestamp(event_start).strftime("%Y-%m-%d"), *map(int, dg_ids)]
    uri = f"file:{db_path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as database:
        frame = pd.read_sql_query(query, database, params=params)
    frame["round_date"] = pd.to_datetime(frame["round_date"], errors="coerce")
    if frame["round_date"].ge(pd.Timestamp(event_start)).fillna(False).any():
        raise ProductionAnchorInputUnavailable("Category history includes event-date/future rounds")
    return frame.sort_values(["dg_id", "round_date"], kind="mergesort")


def reconstruct_field_category_means(
    cases: pd.DataFrame,
    prior_rounds: pd.DataFrame,
) -> pd.DataFrame:
    """Reproduce the frozen category-first EMA means before total recentering."""

    players = cases[["dg_id"]].drop_duplicates().sort_values("dg_id")
    rows: list[dict[str, object]] = []
    for dg_id in players["dg_id"].astype(int):
        history = prior_rounds.loc[prior_rounds["dg_id"].eq(dg_id)]
        row: dict[str, object] = {"dg_id": dg_id}
        for category, column in ROUND_SOURCE_COLUMNS.items():
            values = pd.to_numeric(history[column], errors="coerce").dropna().to_numpy(float)
            row[f"raw_{category}_sg"] = ema_mean(values)
            row[f"{category}_rounds"] = int(len(values))
        rows.append(row)
    estimates = pd.DataFrame(rows).set_index("dg_id")
    for category in CATEGORIES:
        mean_column = f"raw_{category}_sg"
        n_column = f"{category}_rounds"
        field_prior = float(estimates[mean_column].mean())
        if not np.isfinite(field_prior):
            raise ProductionAnchorInputUnavailable(f"No field category prior for {category}")
        reliability = (estimates[n_column].astype(float) / MINIMUM_CATEGORY_ROUNDS).clip(0.0, 1.0)
        observed = estimates[mean_column].fillna(field_prior)
        estimates[mean_column] = reliability * observed + (1.0 - reliability) * field_prior
        estimates[f"{category}_field_prior_sg"] = field_prior
        estimates[f"{category}_reliability"] = reliability
    return estimates.reset_index()


def _load_strict_abilities(
    db_path: Path,
    dg_ids: Sequence[int],
    event_start: object,
) -> pd.DataFrame:
    placeholders = ",".join("?" for _ in dg_ids)
    columns = ", ".join(ABILITY_SOURCE_COLUMNS.values())
    query = f"""
        SELECT rowid, Date AS ability_skill_date, dg_id, player_name, skill_est, {columns}
        FROM model_results2
        WHERE Date < ? AND dg_id IN ({placeholders})
    """
    params = [pd.Timestamp(event_start).strftime("%Y-%m-%d"), *map(int, dg_ids)]
    uri = f"file:{db_path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as database:
        frame = pd.read_sql_query(query, database, params=params)
    frame["ability_skill_date"] = pd.to_datetime(frame["ability_skill_date"], errors="coerce")
    frame = (
        frame.sort_values(["ability_skill_date", "dg_id", "rowid"], kind="mergesort")
        .drop_duplicates("dg_id", keep="last")
        .drop(columns="rowid")
    )
    if frame["ability_skill_date"].ge(pd.Timestamp(event_start)).fillna(True).any():
        raise ProductionAnchorInputUnavailable("Quality ability snapshot is not strictly prior")
    required = [*ABILITY_SOURCE_COLUMNS.values(), "skill_est"]
    if frame[required].apply(pd.to_numeric, errors="coerce").isna().any().any():
        raise ProductionAnchorInputUnavailable("Quality ability snapshot has missing categories")
    if set(map(int, dg_ids)) != set(frame["dg_id"].astype(int)):
        raise ProductionAnchorInputUnavailable("Quality ability snapshot misses event players")
    rename = {source: f"ability_{category}_sg" for category, source in ABILITY_SOURCE_COLUMNS.items()}
    return frame.rename(columns=rename)


def _row_digest(row: Mapping[str, object]) -> str:
    text_columns = [
        "event_key",
        "source_event_key",
        "player_id",
        "player_name",
        "category_reconstruction_contract",
        "ability_snapshot_contract",
        "round_target_authority",
        "target_context",
        "production_equivalence_status",
        "baseline_source_sha256",
        "database_source_sha256",
        "category_reconstruction_source_sha256",
    ]
    numeric_columns = [
        "dg_id",
        "round_no",
        *[f"raw_{category}_sg" for category in CATEGORIES],
        *[f"target_{category}_sg" for category in CATEGORIES],
        "target_total_sg",
        "equal_recenter_shift_sg",
        *[f"ability_{category}_sg" for category in CATEGORIES],
    ]
    payload: dict[str, object] = {"contract": CONTRACT}
    for column in text_columns:
        payload[column] = str(row[column]).strip().casefold()
    for column in numeric_columns:
        payload[column] = _number(row[column])
    payload["event_start"] = pd.Timestamp(row["event_start"]).isoformat()
    payload["ability_skill_date"] = pd.Timestamp(row["ability_skill_date"]).isoformat()
    return _stable_sha(payload)


def build_production_anchor_inputs(
    *,
    baseline_path: Path,
    db_path: Path,
    event_name: str,
    event_start: object | None = None,
    category_source_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build hash-bound production targets and strict-prior quality abilities."""

    cases = _frozen_event_cases(baseline_path, event_name, event_start)
    starts = pd.to_datetime(cases["event_start"]).drop_duplicates()
    if len(starts) != 1:
        raise ProductionAnchorInputUnavailable("Selected cases span multiple event starts")
    start = pd.Timestamp(starts.iloc[0])
    dg_ids = sorted(cases["dg_id"].astype(int).unique().tolist())
    histories = _load_prior_rounds(db_path, dg_ids, start)
    category = reconstruct_field_category_means(cases, histories)
    ability = _load_strict_abilities(db_path, dg_ids, start)

    source_path = category_source_path or Path(__file__)
    sources = {
        "exporter_source_sha256": _sha256(Path(__file__)),
        "baseline_source_sha256": _sha256(baseline_path),
        "database_source_sha256": _sha256(db_path),
        "category_reconstruction_source_sha256": _sha256(source_path),
    }
    frame = cases.merge(category, on="dg_id", how="left", validate="many_to_one")
    ability_columns = [
        "dg_id",
        "ability_skill_date",
        "skill_est",
        *[f"ability_{category}_sg" for category in CATEGORIES],
    ]
    frame = frame.merge(ability[ability_columns], on="dg_id", how="left", validate="many_to_one")
    frame["event_start"] = pd.to_datetime(frame["event_start"])
    frame["event_key"] = [
        canonical_event_key(start_value, source_key)
        for start_value, source_key in zip(frame["event_start"], frame["event_key"])
    ]
    frame = frame.rename(
        columns={
            "source_player_id": "player_id",
            "round_num": "round_no",
            "target_sg": "target_total_sg",
        }
    )
    # Preserve the original composite key separately from the canonical key.
    source_event_key = cases["event_key"].astype(str).to_numpy()
    frame["source_event_key"] = source_event_key
    raw_sum = sum(frame[f"raw_{category}_sg"] for category in CATEGORIES)
    frame["equal_recenter_shift_sg"] = (frame["target_total_sg"] - raw_sum) / 4.0
    for category_name in CATEGORIES:
        frame[f"target_{category_name}_sg"] = (
            frame[f"raw_{category_name}_sg"] + frame["equal_recenter_shift_sg"]
        )
    target_sum = sum(frame[f"target_{category_name}_sg"] for category_name in CATEGORIES)
    if not np.allclose(target_sum, frame["target_total_sg"], atol=1e-12, rtol=0.0):
        raise ProductionAnchorInputUnavailable("Category targets do not sum to frozen total")
    frame["category_reconstruction_contract"] = CATEGORY_RECONSTRUCTION
    frame["ability_snapshot_contract"] = ABILITY_SNAPSHOT
    frame["round_target_authority"] = np.where(
        frame["round_no"].eq(1),
        "deterministic_frozen_backtest_r1_category_mean",
        "static_pre_event_backtest_mean_not_live_cascade",
    )
    frame["target_context"] = "heldout_backtest_static_pre_event"
    frame["production_equivalence_status"] = "not_exact_historical_production_reconstruction"
    for column, value in sources.items():
        frame[column] = value
    frame["production_anchor_input_sha256"] = [
        _row_digest(row) for row in frame.to_dict("records")
    ]

    adapter_columns = [
        "event_key",
        "player_id",
        "round_no",
        "target_ott_sg",
        "target_app_sg",
        "target_arg_sg",
        "target_putt_sg",
        "target_total_sg",
    ]
    frame["production_input_sha256"] = [
        chain_adapter.production_input_sha256(row)
        for row in frame[adapter_columns].to_dict("records")
    ]
    # The adapter independently canonicalizes identities and recomputes every
    # digest.  This catches stale or positionally attached hashes here, before
    # a bundle can be written.
    chain_adapter.validate_production_target_frame(
        frame[[*adapter_columns, "production_input_sha256"]]
    )
    frame = frame.sort_values(["player_name", "round_no"], kind="mergesort").reset_index(drop=True)
    final_target_sum = frame[
        [f"target_{category_name}_sg" for category_name in CATEGORIES]
    ].sum(axis=1).to_numpy(float)
    final_target_total = frame["target_total_sg"].to_numpy(float)

    bundle_columns = [
        "event_key",
        "player_id",
        "round_no",
        "production_anchor_input_sha256",
        "production_input_sha256",
    ]
    bundle_records = frame[bundle_columns].astype(str).to_dict("records")
    bundle_sha = _stable_sha({"contract": CONTRACT, "records": bundle_records, "sources": sources})
    diagnostics = {
        "contract": CONTRACT,
        "event_key": str(frame["event_key"].iloc[0]),
        "source_event_key": str(frame["source_event_key"].iloc[0]),
        "event_name": event_name,
        "event_start": start.isoformat(),
        "players": int(frame["dg_id"].nunique()),
        "round_rows": int(len(frame)),
        "rounds": sorted(frame["round_no"].astype(int).unique().tolist()),
        "ability_cutoff_violations": int(
            frame["ability_skill_date"].ge(frame["event_start"]).sum()
        ),
        "category_target_sum_max_abs_error": float(
            np.max(
                np.abs(
                    final_target_sum - final_target_total
                ),
                initial=0.0,
            )
        ),
        "heldout_score_columns_read": [],
        "heldout_field_scoring_center_used": False,
        "posthoc_round_recenter": False,
        "target_context": "heldout_backtest_static_pre_event",
        "production_equivalence_status": "not_exact_historical_production_reconstruction",
        "r1_target_authority": "deterministic_frozen_backtest_category_mean",
        "r2_r4_target_authority": (
            "static_backtest_sensitivity_only; live production uses path/observation-conditioned updates"
        ),
        "bundle_sha256": bundle_sha,
        "sources": sources,
    }
    return frame, diagnostics


def write_bundle(frame: pd.DataFrame, diagnostics: Mapping[str, object], output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Output path already exists: {output}")
    output.mkdir(parents=True)
    target_path = output / "production_anchor_inputs.parquet"
    frame.to_parquet(target_path, index=False)
    manifest = dict(diagnostics)
    manifest["artifact_sha256"] = {
        target_path.name: _sha256(target_path),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    lowry = frame.loc[frame["player_name"].astype(str).str.contains("lowry", case=False)]
    example = ""
    if len(lowry):
        row = lowry.iloc[0]
        example = (
            f"\nLowry example: OTT {row.target_ott_sg:.6f}, APP {row.target_app_sg:.6f}, "
            f"ARG {row.target_arg_sg:.6f}, PUTT {row.target_putt_sg:.6f}; "
            f"total {row.target_total_sg:.6f}.\n"
        )
    report = (
        "# Frozen backtest category-mean anchor inputs\n\n"
        "Research only. This bundle contains deterministic category targets for the frozen "
        "held-out distribution-engine backtest and a separate strict-prior quality-ability "
        "vector. It is not an exact reconstruction of the historical live production cascade. "
        "It reads no held-out score or "
        "realized field scoring-center columns. R1 is the deterministic pre-event target. "
        "R2-R4 are the frozen static-backtest sensitivity, not the live production cascade, "
        "whose means depend on observed/simulated prior-round updates.\n\n"
        f"Players: {manifest['players']}; player-round rows: {manifest['round_rows']}.\n"
        f"Strict-cutoff violations: {manifest['ability_cutoff_violations']}.\n"
        f"Maximum category-sum residual: {manifest['category_target_sum_max_abs_error']:.3g}.\n"
        + example
    )
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    manifest["artifact_sha256"]["REPORT.md"] = _sha256(output / "REPORT.md")
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--event-name", default="Travelers Championship")
    parser.add_argument("--event-start", default="2026-06-25")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    frame, diagnostics = build_production_anchor_inputs(
        baseline_path=args.baseline,
        db_path=args.db,
        event_name=args.event_name,
        event_start=args.event_start,
        category_source_path=ARCHIVE / "backtest_distribution_engines_2026.py",
    )
    write_bundle(frame, diagnostics, args.output)
    print(json.dumps(diagnostics, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
