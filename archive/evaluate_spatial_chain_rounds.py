"""Frozen round evaluator for a future complete spatial shot chain.

The evaluator has two independent jobs:

1. Preserve the already-completed, leakage-safe 2026 baseline panel for Sony,
   Travelers, and Wyndham (category current, full-strength shot dispersion, and
   pure normal arms).
2. Accept complete per-player round particles from ``spatial_chain_v2``, apply
   the research-only discrete mean/latent adapter, and score them on the same
   event/round loss interface.

The current tee->APP chain is intentionally incomplete, so its particle file
will pass the contract audit but will not be promoted to round scoring until it
contains terminal 18-hole stroke totals. This script never calls production
pricing/storage/publishing code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
ARCHIVE = Path(__file__).resolve().parent
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

from hole_sim_calibration_adapter import (  # noqa: E402
    FrozenProductionSpec,
    anchor_discrete_pmf_with_week_latent,
    anchor_discrete_rounds_with_week_latent,
    audit_spec,
    empirical_integer_pmf,
    load_frozen_production_spec,
    normalise_player,
    sha256_file,
    variance_ratio,
)


TARGET_EVENTS = (
    "Sony Open in Hawaii",
    "Travelers Championship",
    "Wyndham Championship",
)
BASELINE_ARMS = (
    "category_current",
    "category_shot",
    "normal_2p8",
    "normal_player",
)
PRIMARY_BASELINE = "category_shot"
SPATIAL_ARM = "spatial_chain_v2"
# Frozen before the authoritative 12-player x 128-particle run.  CRPS is
# stable at a lower empirical count; exact-score log density needs far more
# support, while paired H2H requires the full planned 128 paths.
MIN_PARTICLES_BY_SCOPE = {
    "round_score_crps": 64,
    "round_score_logpmf": 2_000,
    "r1_h2h_brier": 128,
    "r1_h2h_logloss": 128,
    "all_round_h2h_brier": 128,
    "all_round_h2h_logloss": 128,
}
SPATIAL_PARTICLE_COLUMNS = {
    "event_name", "event_key", "source_player_id", "player_name",
    "round_no", "hole_no", "par", "particle_id", "arm",
    "chain_status", "particle_weight",
}
COMPLETE_ROUND_COLUMNS = {
    "event_key", "event", "source_player_id", "player_name", "round_num", "particle_id",
    "round_strokes", "round_par", "terminal_holes", "input_cutoff",
    "chain_version",
}
COMPLETE_PMF_COLUMNS = {
    "event_key", "event", "player_name", "round_num", "score_value",
    "probability", "actual_strokes", "target_sg", "round_par",
}
COMPLETE_ROUND_SCHEMA = {
    "format": "long parquet or csv",
    "grain": "one row per event/player/round/particle",
    "required_columns": {
        "event_key": "stable heldout event identifier; string",
        "event": "display event name; string",
        "player_name": "canonical lower-case player name; string",
        "source_player_id": "stable R2 player identifier used for the frozen join",
        "round_num": "1-4; integer",
        "particle_id": "aligned path identifier within event; integer",
        "round_strokes": "sum of 18 terminal integer hole scores; integer",
        "round_par": "physical par for strokes=par-SG anchoring; integer",
        "terminal_holes": "must equal 18 for every emitted round/path",
        "input_cutoff": "latest fitted-input date; strictly before event start",
        "chain_version": "non-empty research chain version/provenance",
    },
    "optional_columns": {
        "particle_weight": "non-negative; current joint adapter expects equal aligned paths",
        "target_sg": "producer convenience copy; independently verified if present",
        "actual_strokes": "producer convenience copy; independently verified if present",
    },
    "hard_invariants": [
        "exactly one row per event_key/player_name/round_num/particle_id",
        "all 18 holes terminal before round_strokes is emitted",
        "particle_id is aligned across rounds for the same player/event",
        "dated target_sg and actual outcomes are joined independently by the evaluator",
        "input_cutoff and every fitted feature predate the event",
        "no missing or non-finite numeric values",
    ],
    "marginal_pmf_alternative": {
        "grain": "one row per event/player/round/integer score value",
        "replace": ["particle_id", "round_strokes"],
        "with": ["score_value", "probability"],
        "limitation": "round metrics/H2H only; cannot identify 72-hole joint or finish markets",
    },
}


def _markdown_table(frame: pd.DataFrame, decimals: int = 6) -> str:
    columns = [str(column) for column in frame.columns]

    def cell(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            return "" if not np.isfinite(value) else f"{float(value):.{decimals}f}"
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def frozen_three_event_panel(source_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the exact persisted event metrics from the 2026 walk-forward test."""

    score_path = source_dir / "round_score_event_metrics.csv"
    matchup_path = source_dir / "round_matchup_event_metrics.csv"
    scores = pd.read_csv(score_path, parse_dates=["start"])
    matchups = pd.read_csv(matchup_path, parse_dates=["start"])
    scores = scores[
        scores["event"].isin(TARGET_EVENTS) & scores["arm"].isin(BASELINE_ARMS)
    ].copy()
    matchups = matchups[
        matchups["event"].isin(TARGET_EVENTS) & matchups["arm"].isin(BASELINE_ARMS)
    ].copy()
    expected = len(TARGET_EVENTS) * len(BASELINE_ARMS) * 4
    if len(scores) != expected or len(matchups) != expected:
        raise ValueError(
            f"Frozen panel is incomplete: scores={len(scores)}, matchups={len(matchups)}, "
            f"expected each={expected}"
        )
    return scores, matchups


def _pooled_loss(
    frame: pd.DataFrame,
    sum_column: str,
    *,
    rounds: Iterable[int] | None = None,
) -> pd.DataFrame:
    use = frame if rounds is None else frame[frame["round_num"].isin(list(rounds))]
    pooled = use.groupby("arm", observed=True, as_index=False).agg(
        loss_sum=(sum_column, "sum"), n=("n", "sum"), events=("event", "nunique")
    )
    pooled["value"] = pooled["loss_sum"] / pooled["n"]
    baseline = float(pooled.loc[pooled["arm"].eq(PRIMARY_BASELINE), "value"].iloc[0])
    pooled["delta_vs_full_dispersion"] = pooled["value"] - baseline
    return pooled[["arm", "value", "delta_vs_full_dispersion", "events", "n"]]


def baseline_headline(scores: pd.DataFrame, matchups: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    specs = (
        ("round_score_crps", scores, "crps_sum", None),
        ("round_score_logpmf", scores, "logpmf_sum", None),
        ("r1_h2h_brier", matchups, "brier_sum", (1,)),
        ("r1_h2h_logloss", matchups, "logloss_sum", (1,)),
        ("all_round_h2h_brier", matchups, "brier_sum", None),
        ("all_round_h2h_logloss", matchups, "logloss_sum", None),
    )
    for scope, frame, column, rounds in specs:
        pooled = _pooled_loss(frame, column, rounds=rounds)
        pooled.insert(0, "scope", scope)
        rows.append(pooled)
    return pd.concat(rows, ignore_index=True)


def inspect_spatial_particle_contract(path: Path) -> dict[str, object]:
    """Audit v2 transition particles without pretending they are round scores."""

    frame = pd.read_parquet(path)
    columns = set(frame.columns)
    missing_transition = sorted(SPATIAL_PARTICLE_COLUMNS - columns)
    has_complete_rounds = "round_strokes" in columns
    terminal_status = False
    if "chain_status" in frame:
        status = frame["chain_status"].astype(str).str.casefold()
        terminal_status = bool(status.isin({"holed", "complete", "terminal"}).all())
    audit = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "rows": int(len(frame)),
        "columns": sorted(columns),
        "missing_transition_columns": missing_transition,
        "transition_contract_valid": not missing_transition,
        "has_round_strokes": has_complete_rounds,
        "all_particles_terminal": terminal_status,
        "round_scoring_ready": bool(
            not missing_transition and has_complete_rounds and terminal_status
        ),
    }
    if {"event_key", "source_player_id", "round_no", "hole_no", "particle_id"}.issubset(columns):
        audit["events"] = int(frame["event_key"].nunique())
        audit["players"] = int(frame["source_player_id"].nunique())
        audit["rounds"] = sorted(pd.to_numeric(frame["round_no"], errors="coerce").dropna().unique().astype(int).tolist())
        audit["holes"] = sorted(pd.to_numeric(frame["hole_no"], errors="coerce").dropna().unique().astype(int).tolist())
        audit["particles"] = int(frame["particle_id"].nunique())
    return audit


def _weighted_pmf(values: np.ndarray, weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if weights is None:
        return empirical_integer_pmf(values)
    rounded = np.rint(np.asarray(values, float))
    if not np.allclose(values, rounded, atol=1e-8, rtol=0.0):
        raise ValueError("Weighted round samples must be integer strokes")
    weights = np.asarray(weights, float)
    if np.any(weights < 0.0) or not np.isfinite(weights).all() or weights.sum() <= 0.0:
        raise ValueError("Invalid particle weights")
    support = np.unique(rounded).astype(float)
    probability = np.array([weights[rounded == value].sum() for value in support])
    probability /= probability.sum()
    return support, probability


def crps_from_pmf(support: np.ndarray, probabilities: np.ndarray, actual: float) -> float:
    first = float(probabilities @ np.abs(support - actual))
    distances = np.abs(support[:, None] - support[None, :])
    second = 0.5 * float(probabilities @ distances @ probabilities)
    return first - second


def score_round_distribution(
    samples: np.ndarray,
    actual: float,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    support, probabilities = _weighted_pmf(samples, weights)
    actual_integer = int(round(float(actual)))
    probability = float(probabilities[support == actual_integer].sum())
    # Jeffreys smoothing over the full contiguous observed/actual score span.
    width = int(max(support.max(), actual_integer) - min(support.min(), actual_integer) + 1)
    sample_size = float(len(samples) if weights is None else np.asarray(weights).sum())
    smooth = (probability * sample_size + 0.5) / (sample_size + 0.5 * width)
    cdf = np.cumsum(probabilities)

    def quantile(level: float) -> float:
        return float(support[np.searchsorted(cdf, level, side="left")])

    plugin_crps = crps_from_pmf(support, probabilities, float(actual))
    # A finite ensemble's plug-in CRPS is upward biased relative to the
    # underlying predictive distribution because its self-pairs contribute
    # zero to E|X-X'|.  The standard fair-CRPS/U-statistic correction removes
    # that bias for the uniform candidate particles.  Weighted/theoretical
    # PMFs retain the ordinary proper-score formula.
    fair_crps = plugin_crps
    if weights is None and len(samples) > 1:
        sample_values = np.asarray(samples, float)
        first = float(np.mean(np.abs(sample_values - float(actual))))
        ordered = np.sort(sample_values)
        coefficients = 2.0 * np.arange(len(ordered)) - len(ordered) + 1.0
        unordered_pair_distance = float(coefficients @ ordered)
        fair_crps = first - unordered_pair_distance / (
            len(ordered) * (len(ordered) - 1)
        )

    return {
        "crps": float(fair_crps),
        "crps_empirical_plugin": float(plugin_crps),
        "crps_finite_ensemble_correction": float(fair_crps - plugin_crps),
        "logpmf": float(-np.log(smooth)),
        "pit": float(probabilities[support <= actual].sum()),
        "cover50": float(quantile(0.25) <= actual <= quantile(0.75)),
        "cover80": float(quantile(0.10) <= actual <= quantile(0.90)),
        "cover90": float(quantile(0.05) <= actual <= quantile(0.95)),
        "pred_mean": float(probabilities @ support),
        "pred_sd": float(np.sqrt(probabilities @ ((support - probabilities @ support) ** 2))),
    }


def score_pmf_distribution(
    support: np.ndarray,
    probabilities: np.ndarray,
    actual: float,
    *,
    equivalent_sample_size: float = 10_000.0,
) -> dict[str, float]:
    support = np.asarray(support, float)
    probabilities = np.asarray(probabilities, float)
    probabilities = probabilities / probabilities.sum()
    actual_integer = int(round(float(actual)))
    exact = float(probabilities[support == actual_integer].sum())
    width = int(max(support.max(), actual_integer) - min(support.min(), actual_integer) + 1)
    smooth = (exact * equivalent_sample_size + 0.5) / (
        equivalent_sample_size + 0.5 * width
    )
    cdf = np.cumsum(probabilities)

    def quantile(level: float) -> float:
        return float(support[np.searchsorted(cdf, level, side="left")])

    mean = float(probabilities @ support)
    return {
        "crps": crps_from_pmf(support, probabilities, float(actual)),
        "logpmf": float(-np.log(smooth)),
        "pit": float(probabilities[support <= actual].sum()),
        "cover50": float(quantile(0.25) <= actual <= quantile(0.75)),
        "cover80": float(quantile(0.10) <= actual <= quantile(0.90)),
        "cover90": float(quantile(0.05) <= actual <= quantile(0.95)),
        "pred_mean": mean,
        "pred_sd": float(np.sqrt(probabilities @ ((support - mean) ** 2))),
    }


def load_complete_round_particles(path: Path) -> pd.DataFrame:
    """Load the long, scorer-ready contract produced after full recursion.

    Required columns are listed in ``COMPLETE_ROUND_COLUMNS``. One row is one
    event/player/round/particle. ``round_strokes`` must be the sum of 18
    terminal holes.  Truth and dated production skill are deliberately *not*
    trusted from this file; :func:`join_frozen_case_truth` supplies them from
    the independently generated common-baseline panel.
    """

    frame = pd.read_parquet(path) if path.suffix.casefold() == ".parquet" else pd.read_csv(path)
    missing = sorted(COMPLETE_ROUND_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(
            "Complete-round particle file is not scorer-ready; missing "
            f"{missing}. Tee/APP transition particles are not a substitute for "
            "18 terminal hole scores."
        )
    frame = frame.copy()
    frame["player_name"] = frame["player_name"].map(normalise_player)
    frame["source_player_id"] = frame["source_player_id"].astype(str).str.strip()
    if frame["source_player_id"].isin(["", "nan", "none"]).any():
        raise ValueError("Complete-round source_player_id must be present")
    frame["candidate_arm"] = (
        frame["arm"].astype(str) if "arm" in frame else SPATIAL_ARM
    )
    for column in (
        "round_num", "particle_id", "round_strokes", "round_par",
        "terminal_holes",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    numeric = ["round_num", "particle_id", "round_strokes", "round_par", "terminal_holes"]
    if frame[numeric].isna().any().any() or not np.isfinite(frame[numeric].to_numpy(float)).all():
        raise ValueError("Complete-round particle file contains missing/non-finite numeric values")
    for column in ("round_num", "particle_id", "round_strokes", "round_par", "terminal_holes"):
        if not np.allclose(frame[column], np.rint(frame[column]), atol=1e-8, rtol=0.0):
            raise ValueError(f"Complete-round {column} must be integer")
        frame[column] = np.rint(frame[column]).astype(int)
    if not frame["round_num"].between(1, 4).all():
        raise ValueError("Complete-round round_num must be 1..4")
    if not frame["terminal_holes"].eq(18).all():
        raise ValueError("Complete-round file contains paths without 18 terminal holes")
    if frame["chain_version"].astype(str).str.strip().eq("").any():
        raise ValueError("Complete-round chain_version must be non-empty")
    frame["input_cutoff"] = pd.to_datetime(frame["input_cutoff"], errors="coerce")
    if frame["input_cutoff"].isna().any():
        raise ValueError("Complete-round input_cutoff is missing or invalid")
    if "particle_weight" in frame:
        frame["particle_weight"] = pd.to_numeric(frame["particle_weight"], errors="coerce")
        if (
            frame["particle_weight"].isna().any()
            or not np.isfinite(frame["particle_weight"].to_numpy(float)).all()
            or np.any(frame["particle_weight"] <= 0.0)
        ):
            raise ValueError("Complete-round particle weights must be finite and positive")
        spread = frame.groupby(
            ["event", "source_player_id", "candidate_arm", "round_num"], observed=True
        )["particle_weight"].agg(lambda values: float(values.max() - values.min()))
        if np.any(spread.to_numpy(float) > 1e-12):
            raise ValueError("Joint adapter currently requires uniform particle weights")
        totals = frame.groupby(
            ["event", "source_player_id", "candidate_arm", "round_num"], observed=True
        )["particle_weight"].sum()
        if not np.allclose(totals.to_numpy(float), 1.0, atol=1e-10, rtol=0.0):
            raise ValueError("Complete-round particle weights must sum to one per case")
    duplicated = frame.duplicated(
        ["event", "source_player_id", "candidate_arm", "round_num", "particle_id"]
    )
    if duplicated.any():
        raise ValueError("Complete-round particle file has duplicate particle keys")

    expected_events = {value.casefold() for value in TARGET_EVENTS}
    if not set(frame["event"].astype(str).str.casefold()).issubset(expected_events):
        raise ValueError("Complete-round file contains an event outside the frozen target panel")
    for keys, group in frame.groupby(
        ["event", "source_player_id", "candidate_arm"], observed=True, sort=False
    ):
        if sorted(group["round_num"].unique().tolist()) != [1, 2, 3, 4]:
            raise ValueError(f"Complete-round player does not have R1-R4: {keys}")
        particle_sets = [
            tuple(sorted(round_group["particle_id"].unique().tolist()))
            for _, round_group in group.groupby("round_num", observed=True)
        ]
        if len(set(particle_sets)) != 1:
            raise ValueError(f"Particle IDs are not aligned across rounds: {keys}")
    axis_columns = ["source_player_id", "round_num", "particle_id"]
    for event, event_frame in frame.groupby("event", observed=True, sort=False):
        arm_axes = {
            str(arm): tuple(
                map(
                    tuple,
                    arm_frame[axis_columns]
                    .sort_values(axis_columns)
                    .to_numpy(object),
                )
            )
            for arm, arm_frame in event_frame.groupby(
                "candidate_arm", observed=True, sort=False
            )
        }
        if len(set(arm_axes.values())) != 1:
            raise ValueError(
                f"Candidate arms do not share an identical case/particle axis: {event}"
            )
    return frame


FROZEN_CASE_COLUMNS = {
    "event", "event_start", "source_player_id", "player_name", "round_num", "arm",
    "score_value_relative", "probability", "actual_strokes",
    "actual_relative", "field_scoring_center", "target_sg", "skill_date",
}


def load_frozen_case_pmfs(path: Path) -> pd.DataFrame:
    """Load independently generated dated-skill baselines and heldout truth."""

    frame = pd.read_parquet(path)
    missing = sorted(FROZEN_CASE_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Frozen common-baseline PMFs are missing {missing}")
    frame = frame.copy()
    frame["player_name"] = frame["player_name"].map(normalise_player)
    frame["source_player_id"] = frame["source_player_id"].astype(str).str.strip()
    frame["event_start"] = pd.to_datetime(frame["event_start"], errors="coerce")
    frame["skill_date"] = pd.to_datetime(frame["skill_date"], errors="coerce")
    numeric = [
        "round_num", "score_value_relative", "probability", "actual_strokes",
        "actual_relative", "field_scoring_center", "target_sg",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if (
        frame[numeric].isna().any().any()
        or not np.isfinite(frame[numeric].to_numpy(float)).all()
        or frame[["event_start", "skill_date"]].isna().any().any()
    ):
        raise ValueError("Frozen common-baseline panel contains invalid values")
    if not frame["arm"].isin(BASELINE_ARMS).all():
        raise ValueError("Frozen common-baseline panel contains an unknown arm")
    # model_results2 is date-grained and the established backtest records its
    # pre-event weekly snapshot on the event's start date.  It may not be later
    # than that date; transition/input cutoffs, which retain timestamps, remain
    # strictly earlier below.
    if not (frame["skill_date"] <= frame["event_start"]).all():
        raise ValueError("Frozen target skill is dated after the event start")
    totals = frame.groupby(
        ["event", "player_name", "round_num", "arm"], observed=True
    )["probability"].sum()
    if not np.allclose(totals.to_numpy(float), 1.0, atol=1e-8, rtol=0.0):
        raise ValueError("Frozen baseline PMFs do not sum to one")
    return frame


def join_frozen_case_truth(
    candidate: pd.DataFrame,
    frozen: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join dated skill and observed truth without selecting on made cuts.

    The candidate contract always contains four simulated rounds.  Historical
    truth does not: cut-event players generally have only R1/R2.  Skill is a
    pre-event player value and is therefore joined at event/player grain to
    all four simulated rounds.  Actual scores are joined separately at the
    observed event/player/round grain and are the only rows later scored.
    """

    case_keys = ["event", "source_player_id", "round_num"]
    player_keys = ["event", "source_player_id"]
    candidate = candidate.rename(
        columns={
            column: f"{column}_producer"
            for column in ("player_name", "target_sg", "actual_strokes")
            if column in candidate
        }
    )
    player_metadata = frozen[
        [*player_keys, "player_name", "event_start", "target_sg", "skill_date"]
    ].drop_duplicates()
    if player_metadata.duplicated(player_keys).any():
        raise ValueError("Frozen dated skill is not unique by event/player")
    truth_columns = [*case_keys, "actual_strokes", "actual_relative"]
    truth = frozen[truth_columns].drop_duplicates()
    if truth.duplicated(case_keys).any():
        raise ValueError("Frozen truth/skill metadata is not unique by case")
    # Field scoring is an event-round constant and remains available for R3/R4
    # even when a selected player missed the cut.  Join it independently of
    # player truth so all four candidate rounds can be calibrated without
    # survivor conditioning.
    round_metadata = frozen[
        ["event", "round_num", "field_scoring_center"]
    ].drop_duplicates()
    if round_metadata.duplicated(["event", "round_num"]).any():
        raise ValueError("Frozen field-scoring center is not unique by event/round")
    joined = candidate.merge(
        player_metadata,
        on=player_keys,
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        missing = joined.loc[joined["_merge"].ne("both"), player_keys].drop_duplicates()
        raise ValueError(
            "Candidate roster is absent from independently frozen pre-event skill: "
            f"{missing.to_dict('records')[:8]}"
        )
    joined = joined.drop(columns="_merge")
    joined = joined.merge(truth, on=case_keys, how="left", validate="many_to_one")
    joined = joined.merge(
        round_metadata,
        on=["event", "round_num"],
        how="left",
        validate="many_to_one",
    )
    if joined["field_scoring_center"].isna().any():
        raise ValueError("Candidate round is missing its frozen field-scoring center")
    joined["observed_case"] = joined["actual_strokes"].notna()
    observed_counts = joined.groupby(player_keys, observed=True)["observed_case"].sum()
    if (observed_counts <= 0).any():
        raise ValueError("Candidate roster contains a player with no heldout round truth")
    if not (joined["input_cutoff"] < joined["event_start"]).all():
        raise ValueError("Candidate input_cutoff must strictly predate the heldout event")
    # If the producer supplied convenience copies, require exact agreement but
    # never use them as the source of truth.
    for supplied, frozen_column in (
        ("target_sg", "target_sg"),
        ("actual_strokes", "actual_strokes"),
    ):
        producer_column = f"{supplied}_producer"
        if producer_column in joined:
            comparable = joined["observed_case"] if supplied == "actual_strokes" else np.ones(len(joined), bool)
            if not np.allclose(
                joined.loc[comparable, producer_column].to_numpy(float),
                joined.loc[comparable, frozen_column].to_numpy(float),
                atol=1e-8,
                rtol=0.0,
            ):
                raise ValueError(f"Producer {supplied} disagrees with frozen truth")

    roster = joined[player_keys].drop_duplicates()
    common = frozen.merge(roster, on=player_keys, how="inner", validate="many_to_many")
    expected_arms = set(BASELINE_ARMS)
    arm_sets = common.groupby(case_keys, observed=True)["arm"].agg(lambda x: set(x))
    if any(value != expected_arms for value in arm_sets):
        raise ValueError("Common baseline roster does not contain every baseline arm")
    return joined, common


def load_complete_round_pmfs(path: Path) -> pd.DataFrame:
    """Load a one-round PMF contract for scoring without a 72-hole joint."""

    frame = pd.read_parquet(path) if path.suffix.casefold() == ".parquet" else pd.read_csv(path)
    missing = sorted(COMPLETE_PMF_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Complete-round PMF file is missing {missing}")
    frame = frame.copy()
    frame["player_name"] = frame["player_name"].map(normalise_player)
    numeric = [
        "round_num", "score_value", "probability", "actual_strokes",
        "target_sg", "round_par",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[numeric].isna().any().any():
        raise ValueError("Complete-round PMF file contains missing numeric values")
    if np.any(frame["probability"] < 0.0):
        raise ValueError("Complete-round PMF contains negative probabilities")
    return frame


def candidate_pmf_metrics(
    frame: pd.DataFrame,
    week_latent_sd: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrate/score one-round PMFs; no joint tournament claims are made."""

    case_rows: list[dict[str, object]] = []
    keys = ["event_key", "event", "player_name", "round_num"]
    for values, group in frame.groupby(keys, observed=True, sort=False):
        event_key, event, player, round_num = values
        fixed = group[["actual_strokes", "target_sg", "round_par"]].drop_duplicates()
        if len(fixed) != 1:
            raise ValueError(f"PMF metadata mismatch for {values}")
        probability = group["probability"].to_numpy(float)
        if probability.sum() <= 0.0:
            raise ValueError(f"Zero PMF mass for {values}")
        probability /= probability.sum()
        support = group["score_value"].to_numpy(float)
        row = fixed.iloc[0]
        calibrated, diagnostics = anchor_discrete_pmf_with_week_latent(
            support,
            probability,
            target_strokes=float(row["round_par"] - row["target_sg"]),
            week_latent_sd=week_latent_sd,
        )
        # The adapter first translates the complete integer support by a whole-
        # stroke location shift. Its returned probabilities therefore belong
        # to the calibrated support in diagnostics, not the raw input support.
        calibrated_support = np.asarray(diagnostics["calibrated_support"], float)
        if len(calibrated_support) != len(calibrated):
            raise AssertionError("Calibrated PMF support/probability length mismatch")
        metrics = score_pmf_distribution(
            calibrated_support, calibrated, float(row["actual_strokes"])
        )
        case_rows.append(
            {
                "event_key": event_key,
                "event": event,
                "player_name": player,
                "round_num": int(round_num),
                "actual": float(row["actual_strokes"]),
                "anchor_expected_mean_error": float(
                    diagnostics["mixture_expected_mean_strokes"]
                    - diagnostics["target_mean_strokes"]
                ),
                "anchor_expected_variance_error": float(
                    diagnostics["mixture_expected_round_variance"]
                    - diagnostics["raw_round_variance"]
                ),
                **metrics,
            }
        )
    cases = pd.DataFrame(case_rows)
    events = cases.groupby(["event_key", "event", "round_num"], observed=True, as_index=False).agg(
        crps_sum=("crps", "sum"),
        logpmf_sum=("logpmf", "sum"),
        n=("crps", "size"),
        pit_mean=("pit", "mean"),
        cover50_n=("cover50", "sum"),
        cover80_n=("cover80", "sum"),
        cover90_n=("cover90", "sum"),
    )
    events["arm"] = SPATIAL_ARM
    return cases, events


def calibrate_complete_event(frame: pd.DataFrame, week_latent_sd: float, seed: int) -> pd.DataFrame:
    """Apply the discrete adapter per player/arm, preserving the raw copula.

    Candidate arms for the same event/player must receive the same adapter
    randomness.  Otherwise the shared week latent would add avoidable Monte
    Carlo noise to an arm comparison even when the underlying particle paths
    are common-random-number aligned.
    """

    pieces: list[pd.DataFrame] = []
    for (event_key, player, candidate_arm), group in (
        frame.groupby(
            ["event_key", "player_name", "candidate_arm"],
            observed=True,
            sort=False,
        )
    ):
        rounds = sorted(group["round_num"].astype(int).unique())
        particles = sorted(group["particle_id"].astype(int).unique())
        expected_rows = len(rounds) * len(particles)
        if len(group) != expected_rows:
            raise ValueError(
                f"Incomplete round x particle grid for {event_key}/{player}/{candidate_arm}"
            )
        strokes = group.pivot(
            index="round_num", columns="particle_id", values="round_strokes"
        ).loc[rounds, particles]
        field_centers = (
            group.drop_duplicates("round_num")
            .set_index("round_num")
            .loc[rounds, "field_scoring_center"]
            .to_numpy(float)
        )
        target = group.drop_duplicates("round_num").set_index("round_num").loc[rounds, "target_sg"].to_numpy(float)
        seed_key = f"{int(seed)}|{event_key}|{player}".encode("utf-8")
        adapter_seed = int.from_bytes(
            hashlib.sha256(seed_key).digest()[:8], "little", signed=False
        ) % (2**32 - 1)
        calibrated, diagnostics = anchor_discrete_rounds_with_week_latent(
            strokes.to_numpy(float)[None, :, :],
            # The production skill target is field-relative SG.  In physical
            # score coordinates its expected stroke mean is field center minus
            # target SG. Keeping absolute integer strokes avoids the former
            # par-vs-field-center coordinate mismatch.
            round_pars=field_centers,
            target_sg=target,
            week_latent_sd=week_latent_sd,
            seed=adapter_seed,
        )
        result = group.copy()
        lookup = {
            (round_no, particle): int(calibrated[0, r_index, p_index])
            for r_index, round_no in enumerate(rounds)
            for p_index, particle in enumerate(particles)
        }
        result["calibrated_round_strokes"] = [
            lookup[(int(round_no), int(particle))]
            for round_no, particle in result[["round_num", "particle_id"]].itertuples(index=False)
        ]
        result["calibrated_round_relative"] = (
            result["calibrated_round_strokes"]
            - result["field_scoring_center"].astype(float)
        )
        result["anchor_expected_mean_error_max"] = float(diagnostics["max_abs_expected_mean_error"])
        result["anchor_expected_variance_error_max"] = float(diagnostics["max_abs_expected_variance_error"])
        result["latent_variance_ratio"] = float(variance_ratio(calibrated)[0])
        diagnostic_by_round = {
            int(round_no): {
                "raw_mean_relative": float(
                    diagnostics["raw_mean_strokes"][0, r_index]
                    - field_centers[r_index]
                ),
                "target_mean_relative": float(
                    diagnostics["target_mean_strokes"][0, r_index]
                    - field_centers[r_index]
                ),
                "raw_round_variance": float(diagnostics["raw_round_variance"][0, r_index]),
                "integer_support_shift": int(
                    diagnostics["integer_support_shift"][0, r_index]
                ),
                "post_shift_residual_mean_adjustment": float(
                    diagnostics["post_shift_residual_mean_adjustment"][0, r_index]
                ),
                "anchor_expected_mean_error": float(
                    diagnostics["mixture_expected_mean_strokes"][0, r_index]
                    - diagnostics["target_mean_strokes"][0, r_index]
                ),
                "anchor_expected_variance_error": float(
                    diagnostics["mixture_expected_round_variance"][0, r_index]
                    - diagnostics["raw_round_variance"][0, r_index]
                ),
                "anchor_sample_mean_error": float(
                    diagnostics["sample_mean_strokes"][0, r_index]
                    - diagnostics["target_mean_strokes"][0, r_index]
                ),
                "anchor_sample_variance_error": float(
                    diagnostics["sample_round_variance"][0, r_index]
                    - diagnostics["raw_round_variance"][0, r_index]
                ),
                "anchor_power_temperature": float(diagnostics["power_temperature"][0, r_index]),
                "anchor_clipped_mass": float(
                    diagnostics["conditional_mean_clipped_mass"][0, r_index]
                ),
                "calibrated_pmf_support": tuple(
                    int(value)
                    for value in diagnostics["marginal_support"][0, r_index]
                ),
                "calibrated_pmf_probability": tuple(
                    float(value)
                    for value in diagnostics["marginal_probability"][0, r_index]
                ),
            }
            for r_index, round_no in enumerate(rounds)
        }
        for column in (
            "raw_mean_relative", "target_mean_relative", "raw_round_variance",
            "integer_support_shift", "post_shift_residual_mean_adjustment",
            "anchor_expected_mean_error", "anchor_expected_variance_error",
            "anchor_sample_mean_error", "anchor_sample_variance_error",
            "anchor_power_temperature", "anchor_clipped_mass",
            "calibrated_pmf_support", "calibrated_pmf_probability",
        ):
            result[column] = [
                diagnostic_by_round[int(round_no)][column]
                for round_no in result["round_num"]
            ]
        result["mean_anchor_shift"] = (
            result["target_mean_relative"] - result["raw_mean_relative"]
        )
        pieces.append(result)
    return pd.concat(pieces, ignore_index=True)


def candidate_event_metrics(calibrated: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    calibrated = calibrated[calibrated["observed_case"]].copy()
    if calibrated.empty:
        raise ValueError("No observed candidate rounds are available for scoring")
    case_rows: list[dict[str, object]] = []
    for keys, group in calibrated.groupby(
        ["event_key", "event", "player_name", "candidate_arm", "round_num"],
        observed=True,
        sort=False,
    ):
        event_key, event, player, candidate_arm, round_num = keys
        actual_values = group["actual_strokes"].unique()
        if len(actual_values) != 1:
            raise ValueError(f"Actual stroke mismatch for {keys}")
        support_values = group["calibrated_pmf_support"].iloc[0]
        probability_values = group["calibrated_pmf_probability"].iloc[0]
        if not all(
            value == support_values
            for value in group["calibrated_pmf_support"]
        ) or not all(
            value == probability_values
            for value in group["calibrated_pmf_probability"]
        ):
            raise ValueError(f"Candidate calibrated marginal mismatch for {keys}")
        metrics = score_pmf_distribution(
            np.asarray(support_values, float),
            np.asarray(probability_values, float),
            float(actual_values[0]),
        )
        case_rows.append(
            {
                "event_key": event_key,
                "event": event,
                "player_name": player,
                "arm": f"spatial:{candidate_arm}",
                "round_num": int(round_num),
                "actual": float(actual_values[0]),
                **metrics,
            }
        )
    cases = pd.DataFrame(case_rows)
    events = cases.groupby(
        ["event_key", "event", "round_num", "arm"],
        observed=True,
        as_index=False,
    ).agg(
        crps_sum=("crps", "sum"),
        logpmf_sum=("logpmf", "sum"),
        n=("crps", "size"),
        pit_mean=("pit", "mean"),
        cover50_n=("cover50", "sum"),
        cover80_n=("cover80", "sum"),
        cover90_n=("cover90", "sum"),
    )
    return cases, events


def calibrated_common_baseline_pmfs(
    frozen: pd.DataFrame,
    week_latent_sd: float,
    *,
    latent_particles: int = 128,
) -> pd.DataFrame:
    """Put every frozen baseline arm on the candidate's physical-score lattice.

    The historical backtest stores integer score residuals but centers the
    realized outcome on a generally fractional field scoring level. Scoring
    those raw integer residuals against a fractional outcome while scoring the
    spatial model on integer physical strokes creates an arbitrary one-stroke
    lattice-phase advantage. Apply the identical location/latent adapter to
    every baseline PMF, target ``field_scoring_center - target_sg``, and retain
    absolute integer strokes for both score and H2H comparisons.
    """

    rows: list[dict[str, object]] = []
    keys = ["event", "player_name", "round_num", "arm"]
    for values, group in frozen.groupby(keys, observed=True, sort=False):
        event, player, round_num, arm = values
        fixed_columns = [
            "actual_strokes", "actual_relative",
            "field_scoring_center", "target_sg",
        ]
        fixed = group[fixed_columns].drop_duplicates()
        if len(fixed) != 1:
            raise ValueError(f"Frozen baseline metadata mismatch for {values}")
        order = np.argsort(group["score_value_relative"].to_numpy(float))
        support = group["score_value_relative"].to_numpy(float)[order]
        probability = group["probability"].to_numpy(float)[order]
        probability /= probability.sum()
        metadata = fixed.iloc[0]
        marginal, diagnostics = anchor_discrete_pmf_with_week_latent(
            support,
            probability,
            target_strokes=float(
                metadata["field_scoring_center"] - metadata["target_sg"]
            ),
            week_latent_sd=week_latent_sd,
            latent_bins=31,
            latent_particles=latent_particles,
        )
        calibrated_support = np.asarray(diagnostics["calibrated_support"], float)
        if len(calibrated_support) != len(marginal):
            raise AssertionError("Baseline calibrated support/probability mismatch")
        common = {
            "event_key": str(
                group["event_key"].iloc[0] if "event_key" in group else event
            ),
            "event": event,
            "player_name": player,
            "round_num": int(round_num),
            "arm": arm,
            "actual_strokes": float(metadata["actual_strokes"]),
            "actual_relative": float(metadata["actual_relative"]),
            "field_scoring_center": float(metadata["field_scoring_center"]),
            "target_sg": float(metadata["target_sg"]),
            "integer_support_shift": int(diagnostics["integer_support_shift"]),
            "anchor_expected_mean_error": float(
                diagnostics["mixture_expected_mean_strokes"]
                - diagnostics["target_mean_strokes"]
            ),
            "anchor_expected_variance_error": float(
                diagnostics["mixture_expected_round_variance"]
                - diagnostics["raw_round_variance"]
            ),
        }
        for score, mass in zip(calibrated_support, marginal):
            rows.append(
                {
                    **common,
                    "score_value_physical": int(score),
                    "probability": float(mass),
                }
            )
    output = pd.DataFrame(rows)
    totals = output.groupby(keys, observed=True)["probability"].sum()
    if not np.allclose(totals.to_numpy(float), 1.0, atol=1e-10, rtol=0.0):
        raise AssertionError("Calibrated baseline PMFs do not sum to one")
    return output


def baseline_common_case_metrics(calibrated: pd.DataFrame) -> pd.DataFrame:
    """Score common-lattice baseline PMFs on the exact candidate roster."""

    rows: list[dict[str, object]] = []
    keys = ["event", "player_name", "round_num", "arm"]
    for values, group in calibrated.groupby(keys, observed=True, sort=False):
        event, player, round_num, arm = values
        actual = group["actual_strokes"].unique()
        if len(actual) != 1:
            raise ValueError(f"Frozen actual-strokes mismatch for {values}")
        order = np.argsort(group["score_value_physical"].to_numpy(float))
        support = group["score_value_physical"].to_numpy(float)[order]
        probabilities = group["probability"].to_numpy(float)[order]
        probabilities /= probabilities.sum()
        rows.append(
            {
                "event": event,
                "player_name": player,
                "round_num": int(round_num),
                "arm": arm,
                "actual": float(actual[0]),
                **score_pmf_distribution(support, probabilities, float(actual[0])),
            }
        )
    return pd.DataFrame(rows)


def calibrated_candidate_pmfs(calibrated: pd.DataFrame) -> pd.DataFrame:
    """Collapse calibrated aligned particles to PMFs for common-case scoring."""

    calibrated = calibrated[calibrated["observed_case"]].copy()
    if calibrated.empty:
        raise ValueError("No observed candidate rounds are available for H2H scoring")
    rows: list[dict[str, object]] = []
    keys = ["event", "player_name", "round_num", "candidate_arm"]
    for values, group in calibrated.groupby(keys, observed=True, sort=False):
        event, player, round_num, candidate_arm = values
        support = np.asarray(group["calibrated_pmf_support"].iloc[0], int)
        probabilities = np.asarray(
            group["calibrated_pmf_probability"].iloc[0], float
        )
        probabilities /= probabilities.sum()
        metadata = group[["actual_strokes", "actual_relative"]].drop_duplicates()
        if len(metadata) != 1:
            raise ValueError(f"Candidate truth mismatch for {values}")
        for score, probability in zip(support, probabilities):
            rows.append(
                {
                    "event": event,
                    "player_name": player,
                    "round_num": int(round_num),
                    "arm": f"spatial:{candidate_arm}",
                    "score_value_physical": int(score),
                    "probability": float(probability),
                    "actual_strokes": float(metadata.iloc[0]["actual_strokes"]),
                    "actual_relative": float(metadata.iloc[0]["actual_relative"]),
                }
            )
    return pd.DataFrame(rows)


def _pair_probability(
    left_support: np.ndarray,
    left_probability: np.ndarray,
    right_support: np.ndarray,
    right_probability: np.ndarray,
) -> float:
    """Tie-removed probability that the left golfer shoots the lower score."""

    left_support = np.asarray(left_support, int)
    right_support = np.asarray(right_support, int)
    left_probability = np.asarray(left_probability, float)
    right_probability = np.asarray(right_probability, float)
    left_probability /= left_probability.sum()
    right_probability /= right_probability.sum()
    wins = float(
        np.sum(
            left_probability[:, None]
            * right_probability[None, :]
            * (left_support[:, None] < right_support[None, :])
        )
    )
    ties = float(
        np.sum(
            left_probability[:, None]
            * right_probability[None, :]
            * (left_support[:, None] == right_support[None, :])
        )
    )
    return float(wins / max(1.0 - ties, 1e-12))


def common_matchup_metrics(
    baseline_pmfs: pd.DataFrame,
    candidate_pmfs: pd.DataFrame,
    *,
    selection_pmfs: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every arm on identical frozen-current market-like pairs.

    ``selection_pmfs`` may be the original frozen residual-lattice baselines.
    Using it freezes the 35%-65% matchup roster before common-lattice
    calibration, so an arm cannot select its own evaluation pairs.
    """

    combined = pd.concat(
        [
            baseline_pmfs[[
                "event", "player_name", "round_num", "arm",
                "score_value_physical", "probability", "actual_strokes",
            ]],
            candidate_pmfs[[
                "event", "player_name", "round_num", "arm",
                "score_value_physical", "probability", "actual_strokes",
            ]],
        ],
        ignore_index=True,
    )
    pmfs: dict[tuple[str, int, str, str], tuple[np.ndarray, np.ndarray, float]] = {}
    for keys, group in combined.groupby(
        ["event", "round_num", "arm", "player_name"], observed=True, sort=False
    ):
        event, round_num, arm, player = keys
        order = np.argsort(group["score_value_physical"].to_numpy(int))
        support = group["score_value_physical"].to_numpy(int)[order]
        probability = group["probability"].to_numpy(float)[order]
        actual = group["actual_strokes"].unique()
        if len(actual) != 1:
            raise ValueError(f"H2H actual mismatch for {keys}")
        pmfs[(str(event), int(round_num), str(arm), str(player))] = (
            support,
            probability / probability.sum(),
            float(actual[0]),
        )

    arms = sorted(combined["arm"].unique().tolist())
    selection: dict[
        tuple[str, int, str], tuple[np.ndarray, np.ndarray, float]
    ] = {}
    if selection_pmfs is not None:
        current = selection_pmfs[
            selection_pmfs["arm"].eq("category_current")
        ]
        for keys, group in current.groupby(
            ["event", "round_num", "player_name"],
            observed=True,
            sort=False,
        ):
            event, round_num, player = keys
            order = np.argsort(group["score_value_relative"].to_numpy(int))
            support = group["score_value_relative"].to_numpy(int)[order]
            probability = group["probability"].to_numpy(float)[order]
            actual = group["actual_strokes"].unique()
            if len(actual) != 1:
                raise ValueError(f"H2H selection actual mismatch for {keys}")
            selection[(str(event), int(round_num), str(player))] = (
                support,
                probability / probability.sum(),
                float(actual[0]),
            )
    case_rows: list[dict[str, object]] = []
    for (event, round_num), event_round in combined.groupby(
        ["event", "round_num"], observed=True, sort=False
    ):
        player_sets = [
            set(event_round.loc[event_round["arm"].eq(arm), "player_name"])
            for arm in arms
        ]
        players = sorted(set.intersection(*player_sets)) if player_sets else []
        for left_index, left in enumerate(players):
            for right in players[left_index + 1 :]:
                if selection:
                    base_left = selection[(str(event), int(round_num), left)]
                    base_right = selection[(str(event), int(round_num), right)]
                else:
                    base_left = pmfs[
                        (str(event), int(round_num), "category_current", left)
                    ]
                    base_right = pmfs[
                        (str(event), int(round_num), "category_current", right)
                    ]
                base_probability = _pair_probability(
                    base_left[0], base_left[1], base_right[0], base_right[1]
                )
                if not 0.35 <= base_probability <= 0.65:
                    continue
                if base_left[2] == base_right[2]:
                    continue
                outcome = float(base_left[2] < base_right[2])
                for arm in arms:
                    left_pmf = pmfs[(str(event), int(round_num), arm, left)]
                    right_pmf = pmfs[(str(event), int(round_num), arm, right)]
                    probability = np.clip(
                        _pair_probability(
                            left_pmf[0], left_pmf[1], right_pmf[0], right_pmf[1]
                        ),
                        1e-9,
                        1.0 - 1e-9,
                    )
                    case_rows.append(
                        {
                            "event": event,
                            "round_num": int(round_num),
                            "left": left,
                            "right": right,
                            "arm": arm,
                            "probability": probability,
                            "outcome": outcome,
                            "brier": float((probability - outcome) ** 2),
                            "logloss": float(
                                -(outcome * np.log(probability)
                                  + (1.0 - outcome) * np.log(1.0 - probability))
                            ),
                        }
                    )
    cases = pd.DataFrame(case_rows)
    if cases.empty:
        raise ValueError("No common market-like non-tied H2H cases")
    events = cases.groupby(
        ["event", "round_num", "arm"], observed=True, as_index=False
    ).agg(
        brier_sum=("brier", "sum"),
        logloss_sum=("logloss", "sum"),
        n=("brier", "size"),
    )
    return cases, events


def common_candidate_headline(
    baseline_cases: pd.DataFrame,
    candidate_cases: pd.DataFrame,
    matchup_events: pd.DataFrame,
) -> pd.DataFrame:
    """Pool exact-common-case score and H2H losses versus full dispersion."""

    scores = pd.concat(
        [
            baseline_cases[["event", "round_num", "arm", "crps", "logpmf"]],
            candidate_cases[["event", "round_num", "arm", "crps", "logpmf"]],
        ],
        ignore_index=True,
    )
    rows: list[dict[str, object]] = []
    for scope_index, (scope, frame, metric, rounds) in enumerate((
        ("round_score_crps", scores, "crps", None),
        ("round_score_logpmf", scores, "logpmf", None),
        ("r1_h2h_brier", matchup_events, "brier_sum", [1]),
        ("r1_h2h_logloss", matchup_events, "logloss_sum", [1]),
        ("all_round_h2h_brier", matchup_events, "brier_sum", None),
        ("all_round_h2h_logloss", matchup_events, "logloss_sum", None),
    )):
        use = frame if rounds is None else frame[frame["round_num"].isin(rounds)]
        if metric.endswith("_sum"):
            grouped = use.groupby("arm", observed=True).agg(
                loss=(metric, "sum"), n=("n", "sum"), events=("event", "nunique")
            )
            grouped["value"] = grouped["loss"] / grouped["n"]
            event_totals = use.groupby(
                ["event", "arm"], observed=True, as_index=False
            ).agg(loss=(metric, "sum"), n=("n", "sum"))
        else:
            grouped = use.groupby("arm", observed=True).agg(
                value=(metric, "mean"), n=(metric, "size"), events=("event", "nunique")
            )
            event_totals = use.groupby(
                ["event", "arm"], observed=True, as_index=False
            ).agg(loss=(metric, "sum"), n=(metric, "size"))
        baseline = float(grouped.loc[PRIMARY_BASELINE, "value"])
        for arm, record in grouped.iterrows():
            if arm == PRIMARY_BASELINE:
                low = high = 0.0
                favorable = 0
            else:
                loss_pivot = event_totals.pivot(
                    index="event", columns="arm", values="loss"
                )
                n_pivot = event_totals.pivot(
                    index="event", columns="arm", values="n"
                )
                common_events = (
                    loss_pivot[[PRIMARY_BASELINE, arm]].dropna().index
                    .intersection(n_pivot[[PRIMARY_BASELINE, arm]].dropna().index)
                )
                arm_loss = loss_pivot.loc[common_events, arm].to_numpy(float)
                arm_n = n_pivot.loc[common_events, arm].to_numpy(float)
                base_loss = loss_pivot.loc[common_events, PRIMARY_BASELINE].to_numpy(float)
                base_n = n_pivot.loc[common_events, PRIMARY_BASELINE].to_numpy(float)
                favorable = int(np.sum(arm_loss / arm_n < base_loss / base_n))
                rng = np.random.default_rng(20260820 + scope_index * 1009 + len(rows))
                draws = np.empty(5000, dtype=float)
                for draw in range(len(draws)):
                    indices = rng.integers(0, len(common_events), len(common_events))
                    draws[draw] = (
                        arm_loss[indices].sum() / arm_n[indices].sum()
                        - base_loss[indices].sum() / base_n[indices].sum()
                    )
                low, high = np.quantile(draws, [0.025, 0.975])
            rows.append(
                {
                    "scope": scope,
                    "arm": arm,
                    "value": float(record["value"]),
                    "delta_vs_full_dispersion": float(record["value"] - baseline),
                    "cluster95_low": float(low),
                    "cluster95_high": float(high),
                    "events_favorable": favorable,
                    "events": int(record["events"]),
                    "n": int(record["n"]),
                }
            )
    return pd.DataFrame(rows)


def draw_static_category_rounds(spec: FrozenProductionSpec, sims: int, seed: int) -> np.ndarray:
    """Draw a static category-first benchmark from the frozen full-strength spec."""

    import sims_kernel

    output = np.empty((spec.n_players, spec.n_rounds, sims), dtype=np.int16)
    zeros = np.zeros(spec.n_players, dtype=float)
    for round_index in range(spec.n_rounds):
        scores, _ = sims_kernel.run_single_round(
            np.ascontiguousarray(spec.category_mu),
            np.ascontiguousarray(spec.std_course),
            np.ascontiguousarray(spec.effective_skew),
            np.ascontiguousarray(spec.l_corr),
            np.ascontiguousarray(spec.target_sg[:, round_index]),
            zeros,
            np.ascontiguousarray(spec.round_pars[:, round_index]),
            int(sims),
            int(seed + round_index),
        )
        output[:, round_index] = np.asarray(scores, dtype=np.int16)
    return output


def draw_normal_rounds(spec: FrozenProductionSpec, sims: int, seed: int, sd: float = 2.8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mean = spec.round_pars - spec.target_sg
    scores = np.rint(mean[:, :, None] + float(sd) * rng.standard_normal((spec.n_players, spec.n_rounds, sims)))
    return scores.astype(np.int16)


def run_adapter_self_test(
    fixture: Path,
    feature_inputs: Path,
    dists: Path,
    output: Path,
    sims: int,
    seed: int,
) -> dict[str, object]:
    spec = load_frozen_production_spec(
        fixture,
        round_pars=70.0,
        week_latent_sd=0.65,
        dispersion_feature_inputs=feature_inputs,
        dispersion_mode="oos",
        base_dists_path=dists,
    )
    # A bounded subset keeps the smoke test fast while exercising every axis.
    take = min(12, spec.n_players)
    subset = FrozenProductionSpec(
        player_names=spec.player_names[:take],
        category_mu=spec.category_mu[:take],
        std_course=spec.std_course[:take],
        effective_skew=spec.effective_skew[:take],
        l_corr=spec.l_corr,
        target_sg=spec.target_sg[:take],
        round_pars=spec.round_pars[:take],
        week_latent_sd=spec.week_latent_sd,
        metadata=spec.metadata,
    )
    production_raw = draw_static_category_rounds(subset, sims, seed)
    production, prod_diag = anchor_discrete_rounds_with_week_latent(
        production_raw,
        round_pars=subset.round_pars,
        target_sg=subset.target_sg,
        week_latent_sd=subset.week_latent_sd,
        seed=seed + 900,
    )
    candidate, candidate_diag = anchor_discrete_rounds_with_week_latent(
        production_raw.copy(),
        round_pars=subset.round_pars,
        target_sg=subset.target_sg,
        week_latent_sd=subset.week_latent_sd,
        seed=seed + 900,
    )
    normal_raw = draw_normal_rounds(subset, sims, seed + 400)
    normal, normal_diag = anchor_discrete_rounds_with_week_latent(
        normal_raw,
        round_pars=subset.round_pars,
        target_sg=subset.target_sg,
        week_latent_sd=subset.week_latent_sd,
        seed=seed + 900,
    )
    if not np.array_equal(candidate, production):
        raise AssertionError("Identical candidate/production inputs did not calibrate identically")
    if not np.issubdtype(candidate.dtype, np.integer):
        raise AssertionError("Primary adapter did not return integer strokes")
    support_preserved = True
    for player in range(take):
        for round_index in range(subset.n_rounds):
            support_shift = int(
                candidate_diag["integer_support_shift"][player, round_index]
            )
            if not set(np.unique(candidate[player, round_index])).issubset(
                set(np.unique(production_raw[player, round_index] + support_shift))
            ):
                support_preserved = False
    if not support_preserved:
        raise AssertionError(
            "Discrete adapter created score values outside translated support"
        )
    expected_mean_error = float(candidate_diag["max_abs_expected_mean_error"])
    expected_variance_error = float(candidate_diag["max_abs_expected_variance_error"])
    if expected_mean_error > 1e-6:
        raise AssertionError(f"Discrete anchor expected mean error is {expected_mean_error}")
    if expected_variance_error > 0.03:
        raise AssertionError(f"Discrete anchor expected variance error is {expected_variance_error}")

    def off_diagonal_covariance(values: np.ndarray) -> float:
        items: list[float] = []
        for player in range(values.shape[0]):
            covariance = np.cov(values[player], bias=True)
            upper = np.triu_indices(values.shape[1], k=1)
            items.extend(covariance[upper].tolist())
        return float(np.mean(items))

    cross_round_covariance = off_diagonal_covariance(candidate.astype(float))
    expected_week_covariance = subset.week_latent_sd**2
    if abs(cross_round_covariance - expected_week_covariance) > 0.12:
        raise AssertionError(
            f"Cross-round covariance {cross_round_covariance:.4f} is not close to "
            f"week-latent variance {expected_week_covariance:.4f}"
        )

    # Use an independent frozen production draw as synthetic observations only
    # to execute the score interface; these are explicitly not evidence.
    observed_draws = draw_static_category_rounds(subset, 17, seed + 700)
    actual = observed_draws[:, :, 0]
    metric_rows: list[dict[str, object]] = []
    for arm, values in (
        ("spatial_contract_identity", candidate),
        ("category_shot", production),
        ("normal_2p8", normal),
    ):
        for player in range(take):
            for round_index in range(subset.n_rounds):
                metrics = score_round_distribution(
                    values[player, round_index], actual[player, round_index]
                )
                metric_rows.append(
                    {
                        "arm": arm,
                        "player_name": subset.player_names[player],
                        "round_num": round_index + 1,
                        **metrics,
                    }
                )
    metrics = pd.DataFrame(metric_rows)
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "self_test_score_metrics.csv", index=False)
    diagnostics = pd.DataFrame(
        [
            {
                "arm": "spatial_contract_identity",
                "mean_variance_ratio": float(variance_ratio(candidate).mean()),
                "max_expected_mean_error": expected_mean_error,
                "max_expected_variance_error": expected_variance_error,
                "mean_cross_round_covariance": cross_round_covariance,
            },
            {
                "arm": "category_shot",
                "mean_variance_ratio": float(variance_ratio(production).mean()),
                "max_expected_mean_error": float(prod_diag["max_abs_expected_mean_error"]),
                "max_expected_variance_error": float(prod_diag["max_abs_expected_variance_error"]),
                "mean_cross_round_covariance": off_diagonal_covariance(production.astype(float)),
            },
            {
                "arm": "normal_2p8",
                "mean_variance_ratio": float(variance_ratio(normal).mean()),
                "max_expected_mean_error": float(normal_diag["max_abs_expected_mean_error"]),
                "max_expected_variance_error": float(normal_diag["max_abs_expected_variance_error"]),
                "mean_cross_round_covariance": off_diagonal_covariance(normal.astype(float)),
            },
        ]
    )
    diagnostics.to_csv(output / "self_test_arm_diagnostics.csv", index=False)
    result = {
        "status": "passed",
        "purpose": "mechanical adapter/scorer validation only; synthetic outcomes are not predictive evidence",
        "players": take,
        "rounds": subset.n_rounds,
        "particles": sims,
        "candidate_equals_production": True,
        "translated_integer_support_preserved": support_preserved,
        "expected_week_cross_round_covariance": expected_week_covariance,
        "observed_week_cross_round_covariance": cross_round_covariance,
        "production_spec": audit_spec(spec),
        "diagnostics": diagnostics.to_dict("records"),
    }
    (output / "self_test_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def write_report(
    output: Path,
    headline: pd.DataFrame,
    scores: pd.DataFrame,
    matchups: pd.DataFrame,
    *,
    contract_audit: dict[str, object] | None = None,
    self_test: dict[str, object] | None = None,
    common_headline: pd.DataFrame | None = None,
    common_contract: dict[str, object] | None = None,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    headline.to_csv(output / "three_event_baseline_headline.csv", index=False)
    scores.to_csv(output / "three_event_round_score_event_metrics.csv", index=False)
    matchups.to_csv(output / "three_event_round_matchup_event_metrics.csv", index=False)
    lines = [
        "# Spatial-chain production calibration benchmark", "",
        "## Frozen held-out panel", "",
        "The decision panel is Sony, Travelers, and Wyndham 2026 from the existing "
        "chronological distribution-engine backtest. Every baseline uses the same "
        "dated `model_results2.skill_est`; category and shot histories stop before "
        "the event. `category_shot` is the BMW full-strength variance policy "
        "(OTT 90%, APP 90%, ARG 80%, PUTT 95%), reconstructed separately for each "
        "held-out event. Negative deltas favor an arm over that full-strength baseline.", "",
        _markdown_table(headline, 7), "",
        "## Production mapping", "",
        "- Category means are re-centered equally so their sum equals the production total skill mean.",
        "- Course multipliers scale category standard deviations before draws; the frozen BMW pre-overlay fixture implies OTT 1.10, APP 1.03, ARG 1.11, PUTT 1.04.",
        "- The within-round category correlation is retained through the frozen Cholesky matrix.",
        "- `WEEK_LATENT_SD=0.65` is one player/path SG draw shared across R1-R4. Production shrinks idiosyncratic category variance to keep one-round variance unchanged.",
        "- There is no separate stochastic field-wide round latent in current production. Weather is a deterministic player/tee-time mean input; the live/pre-event cascade also creates cross-round dependence through skill updates.", "",
        "## Complete spatial-chain input contract", "",
        "The scorer-ready particle file has one row per event/player/round/particle and "
        "must contain: `event_key,event,source_player_id,player_name,round_num,particle_id,round_strokes,round_par,terminal_holes,input_cutoff,chain_version`. "
        "`round_strokes` must sum 18 terminal holes; partial tee/APP states are rejected. "
        "Heldout truth and `target_sg` are independently joined from the frozen dated-skill panel; candidate copies are never trusted. "
        "A marginal-PMF alternative replaces particle fields with `score_value,probability`; "
        "it can be used for round losses and H2H but not 72-hole joint/finish products.", "",
        "The primary adapter translates each empirical integer score support by "
        "the nearest whole-stroke location shift, preserving its relative support, "
        "ties, variance, tails, and aligned path ranks. It then power-tempers and "
        "exponentially tilts only the remaining half-stroke-or-less residual to the production mean, then "
        "conditionally re-tilts by a shared player-week latent. The temperature is "
        "root-solved so the latent mixture retains the raw one-round variance. This "
        "calibrates every candidate and frozen baseline PMF in absolute physical strokes to "
        "`field_scoring_center - target_sg` and scores against absolute heldout "
        "strokes. Applying the same adapter to all arms removes the arbitrary lattice-phase "
        "difference between integer physical scores and fractional field-relative truth. "
        "It preserves translated integer support, relative blow-up tails, and the upstream aligned "
        "cross-round rank copula; the continuous linear "
        "recenter is diagnostic-only.", "",
    ]
    if common_headline is not None:
        lines.extend(
            [
                "## Exact-common-roster spatial comparison", "",
                "Every arm below is scored on the identical candidate player-rounds and "
                "the identical H2H pairs frozen from the original category-current PMFs before "
                "common-lattice calibration. Heldout outcomes, dated "
                "skill, and realized field-scoring centering come from the independently "
                "frozen baseline panel, not the candidate file. The roster is selected from "
                "the pre-event R1 field; all four rounds are simulated, while only actually "
                "observed heldout rounds are scored. Negative deltas favor the arm over "
                "full-strength `category_shot` dispersion. `decision_eligible=false` means "
                "the candidate particle count is below the predeclared Monte Carlo floor; "
                "in particular sparse empirical log-PMF values are diagnostic only.", "",
                "Candidate and baseline round CRPS/H2H use the same adapter's exact latent-mixed marginal "
                "probabilities on location-translated physical-score support, not a second 128-path resample. "
                "That removes adapter-resampling noise; uncertainty from the finite raw "
                "support remains. Frozen baseline shapes come from their exact saved PMFs.", "",
                _markdown_table(common_headline, 7), "",
            ]
        )
        if common_contract is not None:
            lines.extend(
                [
                    f"- Candidate events: `{common_contract.get('events')}`",
                    f"- Candidate players: `{common_contract.get('players')}`",
                    f"- Candidate arms: `{common_contract.get('candidate_arms')}`",
                    f"- Particles per player-round: `{common_contract.get('particles_per_case')}`",
                    f"- Simulated player-rounds: `{common_contract.get('simulated_player_rounds')}`",
                    f"- Observed/scored player-rounds: `{common_contract.get('observed_scored_player_rounds')}`",
                    f"- Strict cutoff/terminal/aligned-particle checks: `passed`", "",
                ]
            )
    if contract_audit is not None:
        lines.extend(
            [
                "## Current v2 transition-file readiness", "",
                f"- Transition contract valid: `{contract_audit.get('transition_contract_valid')}`",
                f"- Complete round strokes present: `{contract_audit.get('has_round_strokes')}`",
                f"- All particles terminal: `{contract_audit.get('all_particles_terminal')}`",
                f"- Round scoring ready: `{contract_audit.get('round_scoring_ready')}`", "",
            ]
        )
    if self_test is not None:
        lines.extend(
            [
                "## Mechanical verification", "",
                f"The frozen BMW adapter/scorer self-test passed on {self_test['players']} players, "
                f"four rounds, and {self_test['particles']:,} particles. An identity candidate "
                "matched the category benchmark draw-for-draw after discrete calibration. "
                "Synthetic outcomes exercised CRPS/log-PMF/coverage code only and are not predictive evidence.", "",
            ]
        )
    lines.extend(
        [
            "## Limits before a decision", "",
            "- Partial tee/APP particle files remain transition diagnostics and are rejected by the round scorer.",
            "- The frozen historical baseline omits archived weather forecasts and live between-round skill updates; those are common mean layers, but tournament prices will need the full cascade later.",
            "- Four linked rounds now exist, but the present 12-player fields and 128 paths are too sparse for cut/top-5/win calibration; those require full fields and materially more particles.",
            "- No production sim, cache, price, or deployment is modified by this evaluator.", "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=REPO / "permanent_data" / "distribution_engine_backtest_2026",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "permanent_data" / "hole_sim_prod_calibration",
    )
    parser.add_argument("--spatial-particles", type=Path)
    parser.add_argument("--complete-round-particles", type=Path)
    parser.add_argument("--complete-round-pmfs", type=Path)
    parser.add_argument(
        "--baseline-case-pmfs",
        type=Path,
        help="Independent exact-case PMFs from backtest_distribution_engines_2026 --save-case-pmfs.",
    )
    parser.add_argument("--week-latent-sd", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--self-test-sims", type=int, default=1500)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=REPO / "permanent_data" / "shadow_logs" / "bmw_20260820" / "run_20260819_2225" / "current_bmw_fixture.npz",
    )
    parser.add_argument(
        "--dispersion-feature-inputs",
        type=Path,
        default=REPO / "permanent_data" / "shadow_logs" / "bmw_20260820" / "run_20260819_2225" / "shadow_comparison" / "bmw_feature_inputs.csv",
    )
    parser.add_argument("--dists", type=Path, default=REPO / "this_week_dists_v2.csv")
    args = parser.parse_args()

    scores, matchups = frozen_three_event_panel(args.baseline_dir)
    headline = baseline_headline(scores, matchups)
    contract_audit = None
    if args.spatial_particles:
        contract_audit = inspect_spatial_particle_contract(args.spatial_particles)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "spatial_particle_contract_audit.json").write_text(
            json.dumps(contract_audit, indent=2), encoding="utf-8"
        )

    common_headline = None
    common_contract = None
    if args.complete_round_particles:
        if args.baseline_case_pmfs is None:
            raise ValueError(
                "--complete-round-particles requires --baseline-case-pmfs for an exact common-roster comparison"
            )
        complete = load_complete_round_particles(args.complete_round_particles)
        frozen_cases = load_frozen_case_pmfs(args.baseline_case_pmfs)
        complete, common_baselines = join_frozen_case_truth(complete, frozen_cases)
        calibrated = calibrate_complete_event(
            complete, args.week_latent_sd, args.seed
        )
        anchor_cases = calibrated.drop_duplicates(
            ["event", "source_player_id", "candidate_arm", "round_num"]
        )
        anchor_diagnostics = anchor_cases.groupby(
            ["event", "candidate_arm"], observed=True, as_index=False
        ).agg(
            cases=("round_num", "size"),
            mean_anchor_shift=("mean_anchor_shift", "mean"),
            mean_abs_anchor_shift=(
                "mean_anchor_shift", lambda values: float(np.mean(np.abs(values)))
            ),
            max_abs_anchor_shift=(
                "mean_anchor_shift", lambda values: float(np.max(np.abs(values)))
            ),
            mean_integer_support_shift=("integer_support_shift", "mean"),
            mean_abs_integer_support_shift=(
                "integer_support_shift",
                lambda values: float(np.mean(np.abs(values))),
            ),
            max_abs_integer_support_shift=(
                "integer_support_shift",
                lambda values: float(np.max(np.abs(values))),
            ),
            max_abs_post_shift_residual=(
                "post_shift_residual_mean_adjustment",
                lambda values: float(np.max(np.abs(values))),
            ),
            mean_raw_sd=(
                "raw_round_variance",
                lambda values: float(np.mean(np.sqrt(np.maximum(values, 0.0)))),
            ),
            mean_power_temperature=("anchor_power_temperature", "mean"),
            min_power_temperature=("anchor_power_temperature", "min"),
            max_power_temperature=("anchor_power_temperature", "max"),
            max_clipped_mass=("anchor_clipped_mass", "max"),
            max_expected_mean_error=("anchor_expected_mean_error_max", "max"),
            max_expected_variance_error=(
                "anchor_expected_variance_error_max", "max"
            ),
            max_abs_sample_mean_error=(
                "anchor_sample_mean_error",
                lambda values: float(np.max(np.abs(values))),
            ),
            max_abs_sample_variance_error=(
                "anchor_sample_variance_error",
                lambda values: float(np.max(np.abs(values))),
            ),
        )
        cases, candidate_events = candidate_event_metrics(calibrated)
        calibrated_baselines = calibrated_common_baseline_pmfs(
            common_baselines,
            args.week_latent_sd,
            latent_particles=int(complete["particle_id"].nunique()),
        )
        baseline_cases = baseline_common_case_metrics(calibrated_baselines)
        candidate_pmfs = calibrated_candidate_pmfs(calibrated)
        matchup_cases, matchup_events = common_matchup_metrics(
            calibrated_baselines,
            candidate_pmfs,
            selection_pmfs=common_baselines,
        )
        common_headline = common_candidate_headline(
            baseline_cases, cases, matchup_events
        )
        particle_counts = complete.groupby(
            ["event", "source_player_id", "candidate_arm", "round_num"],
            observed=True,
        )["particle_id"].nunique()
        minimum_particles = int(particle_counts.min())
        common_headline["candidate_particles_per_case_min"] = minimum_particles
        common_headline["minimum_particles_required"] = common_headline["scope"].map(
            MIN_PARTICLES_BY_SCOPE
        ).astype(int)
        common_headline["decision_eligible"] = (
            common_headline["candidate_particles_per_case_min"]
            >= common_headline["minimum_particles_required"]
        )
        common_headline["eligibility_note"] = np.where(
            common_headline["decision_eligible"],
            "meets predeclared Monte Carlo floor; statistical uncertainty still applies",
            "diagnostic only: candidate particle count below predeclared Monte Carlo floor",
        )
        common_contract = {
            "events": int(complete["event"].nunique()),
            "players": int(
                complete[["event", "player_name"]].drop_duplicates().shape[0]
            ),
            "candidate_arms": sorted(complete["candidate_arm"].unique().tolist()),
            "particles_per_case": sorted(particle_counts.unique().tolist()),
            "simulated_player_rounds": int(
                complete[["event", "source_player_id", "round_num"]]
                .drop_duplicates().shape[0]
            ),
            "observed_scored_player_rounds": int(
                complete.loc[
                    complete["observed_case"],
                    ["event", "source_player_id", "round_num"],
                ].drop_duplicates().shape[0]
            ),
            "minimum_particles_by_scope": MIN_PARTICLES_BY_SCOPE,
            "score_coordinate": "common absolute integer physical strokes",
            "mean_target": "field_scoring_center - frozen target_sg",
            "adapter_symmetry": (
                f"identical integer translation, {minimum_particles}-particle latent quadrature, "
                "exact mean, and raw-variance preservation for every baseline and candidate arm"
            ),
            "h2h_pair_selection": (
                "frozen before calibration from original category_current PMFs"
            ),
            "baseline_case_pmfs": str(args.baseline_case_pmfs.resolve()),
            "baseline_case_pmfs_sha256": sha256_file(args.baseline_case_pmfs),
            "complete_round_particles": str(args.complete_round_particles.resolve()),
            "complete_round_particles_sha256": sha256_file(args.complete_round_particles),
        }
        args.output.mkdir(parents=True, exist_ok=True)
        cases.to_csv(args.output / "spatial_round_case_metrics.csv", index=False)
        candidate_events.to_csv(args.output / "spatial_round_event_metrics.csv", index=False)
        baseline_cases.to_csv(args.output / "common_baseline_round_case_metrics.csv", index=False)
        calibrated_baselines.to_parquet(
            args.output / "common_calibrated_baseline_pmfs.parquet", index=False
        )
        matchup_cases.to_csv(args.output / "common_round_h2h_case_metrics.csv", index=False)
        matchup_events.to_csv(args.output / "common_round_h2h_event_metrics.csv", index=False)
        anchor_diagnostics.to_csv(
            args.output / "spatial_anchor_diagnostics.csv", index=False
        )
        common_headline.to_csv(args.output / "common_candidate_headline.csv", index=False)
        (args.output / "common_candidate_contract.json").write_text(
            json.dumps(common_contract, indent=2), encoding="utf-8"
        )

    if args.complete_round_pmfs:
        pmfs = load_complete_round_pmfs(args.complete_round_pmfs)
        pmf_cases, pmf_events = candidate_pmf_metrics(pmfs, args.week_latent_sd)
        args.output.mkdir(parents=True, exist_ok=True)
        pmf_cases.to_csv(args.output / "spatial_round_pmf_case_metrics.csv", index=False)
        pmf_events.to_csv(args.output / "spatial_round_pmf_event_metrics.csv", index=False)

    self_test = None
    if args.self_test:
        self_test = run_adapter_self_test(
            args.fixture,
            args.dispersion_feature_inputs,
            args.dists,
            args.output,
            args.self_test_sims,
            args.seed,
        )
    write_report(
        args.output,
        headline,
        scores,
        matchups,
        contract_audit=contract_audit,
        self_test=self_test,
        common_headline=common_headline,
        common_contract=common_contract,
    )
    manifest = {
        "target_events": list(TARGET_EVENTS),
        "baseline_arms": list(BASELINE_ARMS),
        "primary_baseline": PRIMARY_BASELINE,
        "baseline_dir": str(args.baseline_dir.resolve()),
        "baseline_input_hashes": {
            name: sha256_file(args.baseline_dir / name)
            for name in ("round_score_event_metrics.csv", "round_matchup_event_metrics.csv")
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "adapter_sha256": hashlib.sha256((ARCHIVE / "hole_sim_calibration_adapter.py").read_bytes()).hexdigest(),
        "self_test": bool(args.self_test),
        "spatial_particle_contract_audit": contract_audit,
        "complete_round_particles": (
            str(args.complete_round_particles.resolve())
            if args.complete_round_particles else None
        ),
        "complete_round_particles_sha256": (
            sha256_file(args.complete_round_particles)
            if args.complete_round_particles else None
        ),
        "baseline_case_pmfs": (
            str(args.baseline_case_pmfs.resolve())
            if args.baseline_case_pmfs else None
        ),
        "baseline_case_pmfs_sha256": (
            sha256_file(args.baseline_case_pmfs)
            if args.baseline_case_pmfs else None
        ),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.output / "complete_round_input_schema.json").write_text(
        json.dumps(COMPLETE_ROUND_SCHEMA, indent=2), encoding="utf-8"
    )
    print(f"Wrote {args.output / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
