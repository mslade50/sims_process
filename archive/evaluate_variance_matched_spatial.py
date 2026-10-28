"""Variance-matched common-physical-lattice spatial-chain sensitivity.

This research-only evaluator asks a deliberately narrow question: after every
arm is mean-anchored on the same physical integer-stroke lattice, does the
spatial simulator retain predictive value when its *marginal round variance*
is forced to equal the frozen full-strength ``category_shot`` variance for the
same event/player/round?

The spatial PMF keeps its observed relative integer support.  A whole-integer
translation handles location, then power tempering plus a small exponential
tilt matches the frozen production mean and the exact case-level reference
variance.  No new tail locations are invented.  Frozen baseline arms receive
the same physical-lattice mean/week-latent adapter while retaining their own
raw variance.  H2H cases are selected once from the untouched original
``category_current`` PMFs.

The script writes only a new research output directory.  It does not import or
modify production pricing, caches, storage, publishing, or deployment code.
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
for search_path in (REPO, ARCHIVE):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from evaluate_common_lattice_shape import (  # noqa: E402
    PMF_KEYS,
    _attach_eligibility,
    _baseline_raw_pmfs,
    _candidate_raw_pmfs,
    _markdown_table,
    _normalise_pmf,
    finite_particle_latent_grid,
    frozen_market_like_pairs,
    score_common_h2h,
    score_common_pmfs,
)
from evaluate_spatial_chain_rounds import (  # noqa: E402
    BASELINE_ARMS,
    common_candidate_headline,
    join_frozen_case_truth,
    load_complete_round_particles,
    load_frozen_case_pmfs,
)
from hole_sim_calibration_adapter import (  # noqa: E402
    _fit_variance_preserving_power,
    _pmf_moments,
    sha256_file,
)


SCENARIO = "variance_matched_physical"
REFERENCE_ARM = "category_shot"
EXPECTED_PARTICLES = 128
LATENT_BINS = 5
DEFAULT_WEEK_LATENT_SD = 0.65
CASE_LOOKUP_KEYS = ["event", "source_player_id", "player_name", "round_num"]
POWER_SEARCH_LOWER = 0.03
POWER_SEARCH_UPPER = 100.0
POWER_BOUNDARY_ATOL = 1e-6
POWER_NEAR_LOWER = 0.035
POWER_NEAR_UPPER = 90.0


def _case_key(values: tuple[object, ...]) -> tuple[str, str, str, int]:
    event, source_id, player, round_num = values
    return str(event), str(source_id), str(player), int(round_num)


def category_shot_variance_lookup(
    baseline_raw: pd.DataFrame,
) -> tuple[dict[tuple[str, str, str, int], float], pd.DataFrame]:
    """Return exact raw ``category_shot`` variance for every common case."""

    reference = baseline_raw[baseline_raw["arm"].eq(REFERENCE_ARM)]
    rows: list[dict[str, object]] = []
    lookup: dict[tuple[str, str, str, int], float] = {}
    for values, group in reference.groupby(
        CASE_LOOKUP_KEYS, observed=True, sort=False
    ):
        key = _case_key(values)
        pars = group["round_par"].drop_duplicates()
        if len(pars) != 1:
            raise ValueError(f"Reference round par is not unique for {key}")
        support, probability = _normalise_pmf(
            group["score_value"].to_numpy(float) + float(pars.iloc[0]),
            group["probability"].to_numpy(float),
        )
        mean, variance = _pmf_moments(support, probability)
        if variance <= 0.0:
            raise ValueError(f"Reference variance is non-positive for {key}")
        lookup[key] = float(variance)
        rows.append(
            {
                "event": key[0],
                "source_player_id": key[1],
                "player_name": key[2],
                "round_num": key[3],
                "category_shot_raw_mean_physical": float(mean),
                "category_shot_reference_variance": float(variance),
            }
        )
    if not lookup:
        raise ValueError("No category_shot case variances were found")
    return lookup, pd.DataFrame(rows)


def calibrate_variance_matched_pmfs(
    candidate_raw: pd.DataFrame,
    baseline_raw: pd.DataFrame,
    *,
    week_latent_sd: float,
    latent_nodes: np.ndarray,
    node_weights: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Mean-anchor all arms and variance-match every spatial arm by case."""

    reference_variance, reference_cases = category_shot_variance_lookup(
        baseline_raw
    )
    pmf_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    def append_group(
        values: tuple[object, ...],
        group: pd.DataFrame,
        *,
        source: str,
    ) -> None:
        event_key, event, source_id, player, round_num, arm = values
        key = _case_key((event, source_id, player, round_num))
        if key not in reference_variance:
            raise ValueError(f"Missing common-case category_shot variance for {key}")
        fixed = group[
            [
                "round_par",
                "actual_strokes",
                "field_scoring_center",
                "target_sg",
            ]
        ].drop_duplicates()
        if len(fixed) != 1:
            raise ValueError(f"PMF metadata mismatch for {values}")
        meta = fixed.iloc[0]
        support = group["score_value"].to_numpy(float)
        if source == "baseline":
            support = support + float(meta["round_par"])
        probability = group["probability"].to_numpy(float)
        raw_support, raw_probability = _normalise_pmf(support, probability)
        raw_mean, raw_variance = _pmf_moments(raw_support, raw_probability)
        target_mean = float(meta["field_scoring_center"] - meta["target_sg"])
        integer_support_shift = int(np.rint(target_mean - raw_mean))
        calibrated_support = raw_support + integer_support_shift
        case_reference_variance = float(reference_variance[key])
        is_spatial = str(arm).startswith("spatial:")
        target_variance = (
            case_reference_variance if is_spatial else float(raw_variance)
        )
        variance_policy = (
            "exact_case_category_shot"
            if is_spatial
            else "preserve_baseline_arm_raw_variance"
        )
        (
            power,
            conditional,
            expected_mean,
            expected_variance,
            clipped_mass,
        ) = _fit_variance_preserving_power(
            calibrated_support,
            raw_probability,
            target_mean,
            np.asarray(latent_nodes, float) * float(week_latent_sd),
            np.asarray(node_weights, float),
            target_variance,
        )
        marginal = np.asarray(node_weights, float) @ conditional
        marginal /= marginal.sum()
        mean_error = float(expected_mean - target_mean)
        variance_error = float(expected_variance - target_variance)
        power_at_search_boundary = bool(
            np.isclose(
                power,
                POWER_SEARCH_LOWER,
                rtol=0.0,
                atol=POWER_BOUNDARY_ATOL,
            )
            or np.isclose(
                power,
                POWER_SEARCH_UPPER,
                rtol=0.0,
                atol=POWER_BOUNDARY_ATOL,
            )
        )
        power_near_search_boundary = bool(
            power <= POWER_NEAR_LOWER or power >= POWER_NEAR_UPPER
        )
        if abs(mean_error) > 1e-7:
            raise AssertionError(f"Variance-matched mean error is {mean_error} for {values}")
        if abs(variance_error) > 1e-6:
            raise AssertionError(
                f"Variance-matched variance error is {variance_error} for {values}"
            )
        if not np.array_equal(
            calibrated_support, raw_support + integer_support_shift
        ):
            raise AssertionError("Calibrated support is not a pure integer translation")
        for score, mass in zip(calibrated_support, marginal):
            pmf_rows.append(
                {
                    "scenario": SCENARIO,
                    "event_key": event_key,
                    "event": event,
                    "source_player_id": str(source_id),
                    "player_name": player,
                    "round_num": int(round_num),
                    "arm": arm,
                    "score_value": float(score),
                    "probability": float(mass),
                    "actual": float(meta["actual_strokes"]),
                    "target_mean": target_mean,
                    "coordinate": "physical_integer_strokes",
                    "source": source,
                    "variance_policy": variance_policy,
                }
            )
        audit_rows.append(
            {
                "scenario": SCENARIO,
                "event": event,
                "source_player_id": str(source_id),
                "player_name": player,
                "round_num": int(round_num),
                "arm": arm,
                "source": source,
                "variance_policy": variance_policy,
                "raw_mean": float(raw_mean),
                "target_mean": target_mean,
                "integer_support_shift": float(integer_support_shift),
                "post_shift_residual": float(
                    target_mean - (raw_mean + integer_support_shift)
                ),
                "raw_variance": float(raw_variance),
                "category_shot_reference_variance": case_reference_variance,
                "target_variance": target_variance,
                "raw_to_reference_variance_ratio": float(
                    raw_variance / case_reference_variance
                ),
                "power_temperature": float(power),
                "power_at_search_boundary": power_at_search_boundary,
                "power_near_search_boundary": power_near_search_boundary,
                "clipped_mass": float(clipped_mass),
                "expected_mean": float(expected_mean),
                "expected_variance": float(expected_variance),
                "expected_mean_error": mean_error,
                "expected_variance_error": variance_error,
                "support_min": float(calibrated_support.min()),
                "support_max": float(calibrated_support.max()),
                "support_size": int(len(calibrated_support)),
            }
        )

    for values, group in baseline_raw.groupby(PMF_KEYS, observed=True, sort=False):
        append_group(values, group, source="baseline")
    for values, group in candidate_raw.groupby(PMF_KEYS, observed=True, sort=False):
        append_group(values, group, source="candidate")

    pmfs = pd.DataFrame(pmf_rows)
    audits = pd.DataFrame(audit_rows)
    expected_cases = set(reference_variance)
    spatial = audits[audits["arm"].str.startswith("spatial:")]
    for arm, group in spatial.groupby("arm", observed=True):
        actual_cases = {
            _case_key(values)
            for values in group[CASE_LOOKUP_KEYS].itertuples(index=False, name=None)
        }
        if actual_cases != expected_cases:
            raise ValueError(
                f"{arm} is not on the exact category_shot case panel: "
                f"got {len(actual_cases)}, expected {len(expected_cases)}"
            )
    return pmfs, audits, reference_cases


def _audit_summary(audits: pd.DataFrame) -> pd.DataFrame:
    return audits.groupby(
        ["event", "arm", "source", "variance_policy"],
        observed=True,
        as_index=False,
    ).agg(
        cases=("round_num", "size"),
        mean_raw_to_reference_variance_ratio=(
            "raw_to_reference_variance_ratio",
            "mean",
        ),
        min_raw_to_reference_variance_ratio=(
            "raw_to_reference_variance_ratio",
            "min",
        ),
        max_raw_to_reference_variance_ratio=(
            "raw_to_reference_variance_ratio",
            "max",
        ),
        mean_power_temperature=("power_temperature", "mean"),
        min_power_temperature=("power_temperature", "min"),
        max_power_temperature=("power_temperature", "max"),
        power_at_search_boundary=("power_at_search_boundary", "sum"),
        power_near_search_boundary=("power_near_search_boundary", "sum"),
        max_clipped_mass=("clipped_mass", "max"),
        max_abs_expected_mean_error=(
            "expected_mean_error",
            lambda values: float(np.max(np.abs(values))),
        ),
        max_abs_expected_variance_error=(
            "expected_variance_error",
            lambda values: float(np.max(np.abs(values))),
        ),
    )


def _spatial_variance_summary(audits: pd.DataFrame) -> pd.DataFrame:
    spatial = audits[audits["arm"].str.startswith("spatial:")].copy()
    spatial["needed_widening"] = (
        spatial["raw_variance"] < spatial["category_shot_reference_variance"]
    )
    return spatial.groupby("arm", observed=True, as_index=False).agg(
        cases=("round_num", "size"),
        mean_raw_variance=("raw_variance", "mean"),
        mean_reference_variance=("category_shot_reference_variance", "mean"),
        mean_raw_to_reference_variance_ratio=(
            "raw_to_reference_variance_ratio", "mean"
        ),
        median_raw_to_reference_variance_ratio=(
            "raw_to_reference_variance_ratio", "median"
        ),
        min_raw_to_reference_variance_ratio=(
            "raw_to_reference_variance_ratio", "min"
        ),
        max_raw_to_reference_variance_ratio=(
            "raw_to_reference_variance_ratio", "max"
        ),
        cases_widened=("needed_widening", "sum"),
        mean_power_temperature=("power_temperature", "mean"),
        min_power_temperature=("power_temperature", "min"),
        max_power_temperature=("power_temperature", "max"),
        power_at_search_boundary=("power_at_search_boundary", "sum"),
        power_near_search_boundary=("power_near_search_boundary", "sum"),
    )


def _write_report(
    output: Path,
    headline: pd.DataFrame,
    audits: pd.DataFrame,
    audit_summary: pd.DataFrame,
    variance_summary: pd.DataFrame,
    *,
    pairs: pd.DataFrame,
    week_latent_sd: float,
    latent_nodes: np.ndarray,
    node_weights: np.ndarray,
) -> None:
    spatial_audits = audits[audits["arm"].str.startswith("spatial:")].copy()
    unattainable = int(
        (spatial_audits["expected_variance_error"].abs() > 1e-6).sum()
    )
    boundary = int(spatial_audits["power_at_search_boundary"].sum())
    near_boundary = spatial_audits[
        spatial_audits["power_near_search_boundary"]
    ].sort_values("power_temperature")
    if near_boundary.empty:
        near_boundary_note = "No solution was near either search boundary."
    else:
        descriptions = [
            f"{row.arm}, {row.player_name} R{int(row.round_num)} at {row.event} "
            f"(power {float(row.power_temperature):.6f})"
            for row in near_boundary.itertuples(index=False)
        ]
        near_boundary_note = (
            f"{len(near_boundary)} solution(s) were near a search boundary: "
            + "; ".join(descriptions)
            + ". These are effectively close to uniform or concentrated base "
            "mass on finite observed support and should be treated as support "
            "sensitivities, not strong tail evidence."
        )
    focus = headline[
        headline["arm"].str.startswith("spatial:")
        & headline["scope"].isin(
            [
                "round_score_crps",
                "r1_h2h_brier",
                "r1_h2h_logloss",
                "all_round_h2h_brier",
                "all_round_h2h_logloss",
            ]
        )
    ][
        [
            "scope",
            "arm",
            "value",
            "delta_vs_full_dispersion",
            "cluster95_low",
            "cluster95_high",
            "events_favorable",
            "n",
            "decision_eligible",
        ]
    ]
    moment_focus = audit_summary[
        audit_summary["arm"].str.startswith("spatial:")
    ][
        [
            "event",
            "arm",
            "cases",
            "max_clipped_mass",
            "max_abs_expected_mean_error",
            "max_abs_expected_variance_error",
        ]
    ]
    lines = [
        "# Variance-matched spatial-chain sensitivity",
        "",
        "## Result",
        "",
        "Negative deltas favor the spatial arm over the frozen full-strength "
        "`category_shot` reference. Every spatial event/player/round has exactly "
        "the same marginal variance as its matching reference case, so these "
        "deltas isolate the remaining discrete shape: skew, tail allocation, "
        "and probability mass across integer scores.",
        "",
        _markdown_table(focus),
        "",
        "## What was controlled",
        "",
        "All arms are expressed in physical integer strokes. Frozen baseline "
        "support is `score_value_relative + round_par`; spatial support is the "
        "terminal simulated round score. Every arm is targeted to "
        "`field_scoring_center - target_sg`.",
        "",
        "For a spatial PMF, the exact matching `category_shot` variance is the "
        "target. A nearest-integer location translation preserves every relative "
        "support point and tie. Power tempering adjusts concentration on that "
        "fixed support, and a residual exponential tilt solves the exact mean. "
        "No tail location is added or removed. Baseline arms receive the same "
        "adapter but retain their own raw marginal variance.",
        "",
        f"The marginal week mixture uses {len(latent_nodes)} finite standardized "
        f"normal-quantile nodes from 128 particles, with week SD "
        f"{week_latent_sd:.2f}. Nodes: "
        f"{np.array2string(latent_nodes, precision=6)}; weights: "
        f"{np.array2string(node_weights, precision=6)}.",
        "",
        f"H2H selection is frozen once from the untouched original "
        f"`category_current` PMFs: {len(pairs)} non-tied 35%-65% pairs. "
        "Candidate distributions do not select their own matchups.",
        "",
        "## Variance intervention",
        "",
        _markdown_table(variance_summary),
        "",
        "## Exact moment audit",
        "",
        _markdown_table(moment_focus),
        "",
        f"Of {len(spatial_audits)} spatial case/arm targets, {unattainable} were "
        f"unattainable within the declared 1e-6 variance tolerance and "
        f"{boundary} stopped at the [{POWER_SEARCH_LOWER:g}, "
        f"{POWER_SEARCH_UPPER:g}] numerical search boundaries. Maximum clipped "
        f"latent mass was {float(spatial_audits['clipped_mass'].max()):.8f}. "
        + near_boundary_note,
        "",
        "## Interpretation limits",
        "",
        "- The panel contains only three events and 12 pre-event players per "
        "event; event-cluster intervals are necessarily coarse.",
        "- Only 124 observed player-rounds are scored because missed-cut "
        "R3/R4 outcomes do not exist. Pair selection remains frozen on the same "
        "observed common panel.",
        "- The 463 market-like rows are many overlapping player pairs, not 463 "
        "independent observations. Event-cluster uncertainty is the relevant "
        "readout.",
        "- Variance matching on finite 128-particle empirical support can test "
        "the observed spatial tail topology, but cannot recover tail locations "
        "that were never sampled.",
        "- This is a one-round score/H2H shape test. It does not validate cut, "
        "top-five, or win probabilities and does not modify production.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_variance_matched_evaluation(
    complete_path: Path,
    baseline_path: Path,
    output: Path,
    *,
    week_latent_sd: float = DEFAULT_WEEK_LATENT_SD,
) -> dict[str, object]:
    complete = load_complete_round_particles(complete_path)
    frozen = load_frozen_case_pmfs(baseline_path)
    complete, baselines = join_frozen_case_truth(complete, frozen)
    candidate_raw = _candidate_raw_pmfs(complete)
    baseline_raw = _baseline_raw_pmfs(baselines, complete)
    pairs = frozen_market_like_pairs(baseline_raw)

    particles_by_case = complete.groupby(
        ["event", "source_player_id", "candidate_arm", "round_num"],
        observed=True,
    )["particle_id"].nunique()
    if particles_by_case.nunique() != 1 or int(particles_by_case.iloc[0]) != EXPECTED_PARTICLES:
        raise ValueError(
            "This frozen sensitivity requires exactly 128 particles in every "
            f"candidate case; observed {sorted(particles_by_case.unique().tolist())}"
        )
    latent_nodes, node_weights = finite_particle_latent_grid(
        EXPECTED_PARTICLES, LATENT_BINS
    )
    if len(latent_nodes) != LATENT_BINS:
        raise AssertionError(
            f"Expected exactly {LATENT_BINS} latent nodes, got {len(latent_nodes)}"
        )
    pmfs, audits, reference_cases = calibrate_variance_matched_pmfs(
        candidate_raw,
        baseline_raw,
        week_latent_sd=week_latent_sd,
        latent_nodes=latent_nodes,
        node_weights=node_weights,
    )
    score_cases = score_common_pmfs(pmfs)
    h2h_cases, h2h_events = score_common_h2h(pmfs, pairs)
    baseline_cases = score_cases[score_cases["arm"].isin(BASELINE_ARMS)]
    candidate_cases = score_cases[score_cases["arm"].str.startswith("spatial:")]
    headline = common_candidate_headline(
        baseline_cases,
        candidate_cases,
        h2h_events.drop(columns="scenario"),
    )
    headline.insert(0, "scenario", SCENARIO)
    headline = _attach_eligibility(headline, EXPECTED_PARTICLES)
    audit_summary = _audit_summary(audits)
    variance_summary = _spatial_variance_summary(audits)

    spatial = audits[audits["arm"].str.startswith("spatial:")]
    max_mean_error = float(spatial["expected_mean_error"].abs().max())
    max_variance_error = float(spatial["expected_variance_error"].abs().max())
    max_clipped_mass = float(spatial["clipped_mass"].max())
    unattainable_targets = int(
        (spatial["expected_variance_error"].abs() > 1e-6).sum()
    )
    power_boundary_fits = int(spatial["power_at_search_boundary"].sum())
    power_near_boundary = int(spatial["power_near_search_boundary"].sum())
    if max_mean_error > 1e-7 or max_variance_error > 1e-6:
        raise AssertionError("Spatial exact-moment contract failed")
    if max_clipped_mass > 0.0:
        raise AssertionError("A latent conditional mean hit finite support")
    if power_boundary_fits > 0:
        raise AssertionError("A spatial power solve stopped at its search boundary")

    output.mkdir(parents=True, exist_ok=True)
    pmfs.to_parquet(output / "variance_matched_pmfs.parquet", index=False)
    audits.to_csv(output / "variance_matched_case_audit.csv", index=False)
    audit_summary.to_csv(output / "variance_matched_audit_summary.csv", index=False)
    variance_summary.to_csv(output / "spatial_variance_intervention.csv", index=False)
    reference_cases.to_csv(output / "category_shot_case_variances.csv", index=False)
    score_cases.to_csv(output / "variance_matched_round_case_metrics.csv", index=False)
    pairs.to_csv(output / "frozen_category_current_h2h_pairs.csv", index=False)
    h2h_cases.to_csv(output / "variance_matched_h2h_case_metrics.csv", index=False)
    h2h_events.to_csv(output / "variance_matched_h2h_event_metrics.csv", index=False)
    headline.to_csv(output / "variance_matched_headline.csv", index=False)
    _write_report(
        output,
        headline,
        audits,
        audit_summary,
        variance_summary,
        pairs=pairs,
        week_latent_sd=week_latent_sd,
        latent_nodes=latent_nodes,
        node_weights=node_weights,
    )
    manifest = {
        "status": "complete",
        "research_only": True,
        "scenario": SCENARIO,
        "reference_arm": REFERENCE_ARM,
        "spatial_variance_policy": "exact_matching_category_shot_raw_variance",
        "baseline_variance_policy": "preserve_each_baseline_arm_raw_variance",
        "coordinate": "physical_integer_strokes",
        "mean_target": "field_scoring_center - target_sg",
        "support_policy": "nearest_integer_translation_no_support_extension",
        "shape_adapter": "power_tempering_plus_exponential_mean_tilt",
        "week_latent_sd": float(week_latent_sd),
        "particles_per_candidate_case": EXPECTED_PARTICLES,
        "latent_bins": LATENT_BINS,
        "latent_nodes": latent_nodes.tolist(),
        "latent_node_weights": node_weights.tolist(),
        "events": sorted(complete["event"].unique().tolist()),
        "observed_common_player_rounds": int(len(reference_cases)),
        "frozen_h2h_pairs": int(len(pairs)),
        "max_abs_spatial_expected_mean_error": max_mean_error,
        "max_abs_spatial_expected_variance_error": max_variance_error,
        "max_spatial_clipped_mass": max_clipped_mass,
        "spatial_variance_targets": int(len(spatial)),
        "unattainable_spatial_variance_targets": unattainable_targets,
        "spatial_power_boundary_fits": power_boundary_fits,
        "spatial_power_near_boundary": power_near_boundary,
        "power_search_bounds": [POWER_SEARCH_LOWER, POWER_SEARCH_UPPER],
        "power_near_boundary_thresholds": [POWER_NEAR_LOWER, POWER_NEAR_UPPER],
        "complete_round_particles": str(complete_path.resolve()),
        "complete_round_particles_sha256": sha256_file(complete_path),
        "baseline_case_pmfs": str(baseline_path.resolve()),
        "baseline_case_pmfs_sha256": sha256_file(baseline_path),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "common_lattice_helper_sha256": hashlib.sha256(
            (ARCHIVE / "evaluate_common_lattice_shape.py").read_bytes()
        ).hexdigest(),
        "calibration_adapter_sha256": hashlib.sha256(
            (ARCHIVE / "hole_sim_calibration_adapter.py").read_bytes()
        ).hexdigest(),
        "core_evaluator_modified": False,
        "production_modified": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--complete-round-particles", type=Path, required=True)
    parser.add_argument("--baseline-case-pmfs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--week-latent-sd", type=float, default=DEFAULT_WEEK_LATENT_SD
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.week_latent_sd < 0.0:
        raise ValueError("--week-latent-sd must be non-negative")
    manifest = run_variance_matched_evaluation(
        args.complete_round_particles,
        args.baseline_case_pmfs,
        args.output,
        week_latent_sd=args.week_latent_sd,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
