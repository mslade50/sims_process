"""Common-lattice round-distribution comparison for the spatial shot chain.

This research-only evaluator resolves a discrete-coordinate asymmetry in the
first complete-chain benchmark.  The frozen category/normal reference PMFs are
integer scores in a nuisance-relative coordinate, while terminal hole scores
are integers in physical-stroke coordinates.  Those two lattices have a
different fractional phase whenever the realized field scoring center is not
an integer.

Two deliberately separate comparisons are emitted:

``reference_relative``
    Keep every frozen baseline PMF exactly as saved.  Express candidate scores
    as ``round_strokes - round_par``, anchor their mean to ``-target_sg``, and
    score against ``actual_strokes - field_scoring_center``.  This is the
    primary comparison because it preserves the original backtest reference.

``symmetric_physical``
    Put every arm on the physical integer-stroke lattice and apply the same
    discrete mean/week-latent adapter to every arm.  Frozen baseline support is
    ``score_value_relative + round_par``; candidate support is the terminal
    physical score.  Every target mean is
    ``field_scoring_center - target_sg``.  This sensitivity preserves each
    arm's raw variance while neutralising adapter and lattice asymmetry.

The script only writes a new research output directory.  It does not import or
modify production pricing, caches, storage, or deployment code.
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

from hole_sim_calibration_adapter import (  # noqa: E402
    _fit_variance_preserving_power,
    _pmf_moments,
    _standard_normal_nodes,
    empirical_integer_pmf,
    sha256_file,
)
from evaluate_spatial_chain_rounds import (  # noqa: E402
    BASELINE_ARMS,
    MIN_PARTICLES_BY_SCOPE,
    PRIMARY_BASELINE,
    common_candidate_headline,
    join_frozen_case_truth,
    load_complete_round_particles,
    load_frozen_case_pmfs,
    score_pmf_distribution,
)


REFERENCE_RELATIVE = "reference_relative"
SYMMETRIC_PHYSICAL = "symmetric_physical"
SCENARIOS = (REFERENCE_RELATIVE, SYMMETRIC_PHYSICAL)

PMF_KEYS = [
    "event_key",
    "event",
    "source_player_id",
    "player_name",
    "round_num",
    "arm",
]
CASE_KEYS = ["event", "player_name", "round_num", "arm"]
META_COLUMNS = [
    "event_key",
    "event",
    "source_player_id",
    "player_name",
    "round_num",
    "round_par",
    "actual_strokes",
    "actual_relative",
    "field_scoring_center",
    "target_sg",
]


def _normalise_pmf(
    support: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted, positive, unique integer support and normalised mass."""

    x = np.asarray(support, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if x.ndim != 1 or p.ndim != 1 or len(x) != len(p) or len(x) < 2:
        raise ValueError("PMF support/probability must be equal-length vectors")
    if not np.isfinite(x).all() or not np.isfinite(p).all() or np.any(p < 0.0):
        raise ValueError("PMF support/probability contains invalid values")
    if not np.allclose(x, np.rint(x), atol=1e-8, rtol=0.0):
        raise ValueError("Common-lattice support must be integer")
    frame = pd.DataFrame({"support": np.rint(x).astype(int), "probability": p})
    frame = frame.groupby("support", observed=True, as_index=False)["probability"].sum()
    frame = frame[frame["probability"] > 0.0].sort_values("support")
    if len(frame) < 2 or float(frame["probability"].sum()) <= 0.0:
        raise ValueError("Common-lattice PMF has degenerate positive support")
    probability = frame["probability"].to_numpy(float)
    probability /= probability.sum()
    return frame["support"].to_numpy(float), probability


def calibrate_common_pmf(
    support: np.ndarray,
    probabilities: np.ndarray,
    *,
    target_mean: float,
    week_latent_sd: float,
    latent_nodes: np.ndarray,
    node_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Location-anchor one integer PMF and verify its exact moment contract."""

    raw_support, raw_probability = _normalise_pmf(support, probabilities)
    raw_mean, raw_variance = _pmf_moments(raw_support, raw_probability)
    integer_support_shift = int(np.rint(float(target_mean) - raw_mean))
    calibrated_support = raw_support + integer_support_shift
    (
        power,
        conditional,
        expected_mean,
        expected_variance,
        clipped_mass,
    ) = _fit_variance_preserving_power(
        calibrated_support,
        raw_probability,
        float(target_mean),
        np.asarray(latent_nodes, float) * float(week_latent_sd),
        np.asarray(node_weights, float),
        raw_variance,
    )
    calibrated_probability = np.asarray(node_weights, float) @ conditional
    calibrated_probability /= calibrated_probability.sum()
    mean_error = float(expected_mean - target_mean)
    variance_error = float(expected_variance - raw_variance)
    if abs(mean_error) > 1e-7:
        raise AssertionError(f"Common-lattice mean error is {mean_error}")
    if abs(variance_error) > 1e-6:
        raise AssertionError(f"Common-lattice variance error is {variance_error}")
    if not np.allclose(
        calibrated_support,
        raw_support + integer_support_shift,
        atol=1e-8,
        rtol=0.0,
    ):
        raise AssertionError("Calibrated support is not a pure integer translation")
    audit = {
        "raw_mean": float(raw_mean),
        "target_mean": float(target_mean),
        "raw_variance": float(raw_variance),
        "integer_support_shift": float(integer_support_shift),
        "post_shift_residual": float(
            target_mean - (raw_mean + integer_support_shift)
        ),
        "power_temperature": float(power),
        "clipped_mass": float(clipped_mass),
        "expected_mean_error": mean_error,
        "expected_variance_error": variance_error,
    }
    return calibrated_support, np.asarray(calibrated_probability, float), audit


def finite_particle_latent_grid(
    particles: int,
    requested_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact latent nodes/weights used by the aligned path adapter."""

    nodes, weights, _ = _standard_normal_nodes(
        int(particles), int(requested_bins), np.random.default_rng(0)
    )
    return np.asarray(nodes, float), np.asarray(weights, float)


def _observed_metadata(complete: pd.DataFrame) -> pd.DataFrame:
    metadata = complete.loc[complete["observed_case"], META_COLUMNS].drop_duplicates()
    keys = ["event", "source_player_id", "player_name", "round_num"]
    if metadata.duplicated(keys).any():
        raise ValueError("Observed candidate metadata is not unique per player-round")
    return metadata


def _candidate_raw_pmfs(complete: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    use = complete[complete["observed_case"]].copy()
    keys = [
        "event_key",
        "event",
        "source_player_id",
        "player_name",
        "candidate_arm",
        "round_num",
    ]
    for values, group in use.groupby(keys, observed=True, sort=False):
        event_key, event, source_id, player, candidate_arm, round_num = values
        support, probability = empirical_integer_pmf(group["round_strokes"].to_numpy())
        fixed = group[
            [
                "round_par",
                "actual_strokes",
                "actual_relative",
                "field_scoring_center",
                "target_sg",
            ]
        ].drop_duplicates()
        if len(fixed) != 1:
            raise ValueError(f"Candidate truth mismatch for {values}")
        meta = fixed.iloc[0]
        for score, mass in zip(support, probability):
            rows.append(
                {
                    "event_key": event_key,
                    "event": event,
                    "source_player_id": str(source_id),
                    "player_name": player,
                    "round_num": int(round_num),
                    "arm": f"spatial:{candidate_arm}",
                    "score_value": int(score),
                    "probability": float(mass),
                    **meta.to_dict(),
                }
            )
    return pd.DataFrame(rows)


def _baseline_raw_pmfs(
    baselines: pd.DataFrame,
    complete: pd.DataFrame,
) -> pd.DataFrame:
    metadata = _observed_metadata(complete)[
        ["event", "source_player_id", "player_name", "round_num", "round_par"]
    ]
    joined = baselines.merge(
        metadata,
        on=["event", "source_player_id", "player_name", "round_num"],
        how="inner",
        validate="many_to_one",
    )
    if len(joined) != len(baselines):
        raise ValueError("Baseline PMFs did not receive one candidate round par")
    joined = joined.rename(columns={"score_value_relative": "score_value"})
    return joined[
        PMF_KEYS
        + [
            "score_value",
            "probability",
            "round_par",
            "actual_strokes",
            "actual_relative",
            "field_scoring_center",
            "target_sg",
        ]
    ].copy()


def calibrate_scenario_pmfs(
    candidate_raw: pd.DataFrame,
    baseline_raw: pd.DataFrame,
    *,
    scenario: str,
    week_latent_sd: float,
    latent_nodes: np.ndarray,
    node_weights: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build all-arm PMFs for one common-coordinate comparison."""

    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown common-lattice scenario: {scenario}")
    pmf_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []

    def append_group(
        values: tuple[object, ...],
        group: pd.DataFrame,
        *,
        calibrate: bool,
        relative: bool,
        source: str,
    ) -> None:
        event_key, event, source_id, player, round_num, arm = values
        fixed = group[
            [
                "round_par",
                "actual_strokes",
                "actual_relative",
                "field_scoring_center",
                "target_sg",
            ]
        ].drop_duplicates()
        if len(fixed) != 1:
            raise ValueError(f"PMF metadata mismatch for {values}")
        meta = fixed.iloc[0]
        support = group["score_value"].to_numpy(float)
        probability = group["probability"].to_numpy(float)
        if relative:
            if source == "candidate":
                support = support - float(meta["round_par"])
            target = -float(meta["target_sg"])
            actual = float(meta["actual_relative"])
        else:
            if source == "baseline":
                support = support + float(meta["round_par"])
            target = float(meta["field_scoring_center"] - meta["target_sg"])
            actual = float(meta["actual_strokes"])

        raw_support, raw_probability = _normalise_pmf(support, probability)
        if calibrate:
            out_support, out_probability, audit = calibrate_common_pmf(
                raw_support,
                raw_probability,
                target_mean=target,
                week_latent_sd=week_latent_sd,
                latent_nodes=latent_nodes,
                node_weights=node_weights,
            )
            adapter_policy = "common_discrete_location_week_latent"
        else:
            # The primary scenario is a frozen-reference comparison. Preserve
            # its saved positive probabilities bit-for-bit rather than even
            # renormalising by a floating-point epsilon.
            order = np.argsort(np.asarray(support, float))
            out_support = np.rint(np.asarray(support, float)[order])
            out_probability = np.asarray(probability, float)[order]
            keep = out_probability > 0.0
            out_support, out_probability = out_support[keep], out_probability[keep]
            if len(out_support) != len(raw_support) or not np.isclose(
                out_probability.sum(), 1.0, atol=1e-12, rtol=0.0
            ):
                raise ValueError("Frozen reference PMF is not already normalised/unique")
            raw_mean = float(raw_probability @ raw_support)
            raw_variance = float(raw_probability @ ((raw_support - raw_mean) ** 2))
            audit = {
                "raw_mean": raw_mean,
                "target_mean": target,
                "raw_variance": raw_variance,
                "integer_support_shift": 0.0,
                "post_shift_residual": target - raw_mean,
                "power_temperature": 1.0,
                "clipped_mass": 0.0,
                "expected_mean_error": raw_mean - target,
                "expected_variance_error": 0.0,
            }
            adapter_policy = "frozen_reference_unchanged"
        for score, mass in zip(out_support, out_probability):
            pmf_rows.append(
                {
                    "scenario": scenario,
                    "event_key": event_key,
                    "event": event,
                    "source_player_id": str(source_id),
                    "player_name": player,
                    "round_num": int(round_num),
                    "arm": arm,
                    "score_value": float(score),
                    "probability": float(mass),
                    "actual": actual,
                    "target_mean": target,
                    "coordinate": (
                        "nuisance_relative_integer" if relative else "physical_integer_strokes"
                    ),
                    "source": source,
                    "adapter_policy": adapter_policy,
                }
            )
        diagnostic_rows.append(
            {
                "scenario": scenario,
                "event": event,
                "source_player_id": str(source_id),
                "player_name": player,
                "round_num": int(round_num),
                "arm": arm,
                "source": source,
                "adapter_policy": adapter_policy,
                **audit,
            }
        )

    candidate_groups = candidate_raw.groupby(PMF_KEYS, observed=True, sort=False)
    baseline_groups = baseline_raw.groupby(PMF_KEYS, observed=True, sort=False)
    if scenario == REFERENCE_RELATIVE:
        for values, group in baseline_groups:
            append_group(values, group, calibrate=False, relative=True, source="baseline")
        for values, group in candidate_groups:
            append_group(values, group, calibrate=True, relative=True, source="candidate")
    else:
        for values, group in baseline_groups:
            append_group(values, group, calibrate=True, relative=False, source="baseline")
        for values, group in candidate_groups:
            append_group(values, group, calibrate=True, relative=False, source="candidate")

    return pd.DataFrame(pmf_rows), pd.DataFrame(diagnostic_rows)


def score_common_pmfs(pmfs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for values, group in pmfs.groupby(CASE_KEYS, observed=True, sort=False):
        event, player, round_num, arm = values
        fixed = group[["actual", "scenario"]].drop_duplicates()
        if len(fixed) != 1:
            raise ValueError(f"Scoring metadata mismatch for {values}")
        order = np.argsort(group["score_value"].to_numpy(float))
        support = group["score_value"].to_numpy(float)[order]
        probability = group["probability"].to_numpy(float)[order]
        actual = float(fixed.iloc[0]["actual"])
        rows.append(
            {
                "scenario": fixed.iloc[0]["scenario"],
                "event": event,
                "player_name": player,
                "round_num": int(round_num),
                "arm": arm,
                "actual": actual,
                **score_pmf_distribution(support, probability, actual),
            }
        )
    return pd.DataFrame(rows)


def _pair_probability(
    left_support: np.ndarray,
    left_probability: np.ndarray,
    right_support: np.ndarray,
    right_probability: np.ndarray,
) -> float:
    left_probability = np.asarray(left_probability, float)
    right_probability = np.asarray(right_probability, float)
    left_probability /= left_probability.sum()
    right_probability /= right_probability.sum()
    left_support = np.asarray(left_support, float)
    right_support = np.asarray(right_support, float)
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
            * np.isclose(left_support[:, None], right_support[None, :], atol=1e-10)
        )
    )
    return float(wins / max(1.0 - ties, 1e-12))


def frozen_market_like_pairs(baseline_raw: pd.DataFrame) -> pd.DataFrame:
    """Freeze pair selection from the untouched category-current reference."""

    selector = baseline_raw[baseline_raw["arm"].eq("category_current")]
    pmfs: dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray, float]] = {}
    for (event, round_num, player), group in selector.groupby(
        ["event", "round_num", "player_name"], observed=True, sort=False
    ):
        support, probability = _normalise_pmf(
            group["score_value"].to_numpy(float),
            group["probability"].to_numpy(float),
        )
        pmfs[(str(event), int(round_num), str(player))] = (
            support,
            probability,
            float(group["actual_strokes"].iloc[0]),
        )
    rows: list[dict[str, object]] = []
    events_rounds = sorted({(key[0], key[1]) for key in pmfs})
    for event, round_num in events_rounds:
        players = sorted(key[2] for key in pmfs if key[:2] == (event, round_num))
        for left_index, left in enumerate(players):
            for right in players[left_index + 1 :]:
                left_pmf = pmfs[(event, round_num, left)]
                right_pmf = pmfs[(event, round_num, right)]
                probability = _pair_probability(
                    left_pmf[0], left_pmf[1], right_pmf[0], right_pmf[1]
                )
                if not 0.35 <= probability <= 0.65:
                    continue
                if left_pmf[2] == right_pmf[2]:
                    continue
                rows.append(
                    {
                        "event": event,
                        "round_num": int(round_num),
                        "left": left,
                        "right": right,
                        "selector_probability": probability,
                        "outcome": float(left_pmf[2] < right_pmf[2]),
                    }
                )
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        raise ValueError("Frozen category-current selector produced no H2H pairs")
    return pairs


def score_common_h2h(
    pmfs: pd.DataFrame,
    pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pmf_lookup: dict[tuple[str, int, str, str], tuple[np.ndarray, np.ndarray]] = {}
    for (event, round_num, arm, player), group in pmfs.groupby(
        ["event", "round_num", "arm", "player_name"], observed=True, sort=False
    ):
        order = np.argsort(group["score_value"].to_numpy(float))
        support = group["score_value"].to_numpy(float)[order]
        probability = group["probability"].to_numpy(float)[order]
        probability /= probability.sum()
        pmf_lookup[(str(event), int(round_num), str(arm), str(player))] = (
            support,
            probability,
        )
    arms = sorted(pmfs["arm"].unique().tolist())
    scenario_values = pmfs["scenario"].drop_duplicates().tolist()
    if len(scenario_values) != 1:
        raise ValueError("H2H scorer expects exactly one scenario")
    scenario = scenario_values[0]
    rows: list[dict[str, object]] = []
    for pair in pairs.itertuples(index=False):
        for arm in arms:
            left = pmf_lookup[(pair.event, int(pair.round_num), arm, pair.left)]
            right = pmf_lookup[(pair.event, int(pair.round_num), arm, pair.right)]
            probability = float(
                np.clip(
                    _pair_probability(left[0], left[1], right[0], right[1]),
                    1e-9,
                    1.0 - 1e-9,
                )
            )
            outcome = float(pair.outcome)
            rows.append(
                {
                    "scenario": scenario,
                    "event": pair.event,
                    "round_num": int(pair.round_num),
                    "left": pair.left,
                    "right": pair.right,
                    "arm": arm,
                    "selector_probability": float(pair.selector_probability),
                    "probability": probability,
                    "outcome": outcome,
                    "brier": float((probability - outcome) ** 2),
                    "logloss": float(
                        -(
                            outcome * np.log(probability)
                            + (1.0 - outcome) * np.log(1.0 - probability)
                        )
                    ),
                }
            )
    cases = pd.DataFrame(rows)
    events = cases.groupby(
        ["scenario", "event", "round_num", "arm"],
        observed=True,
        as_index=False,
    ).agg(
        brier_sum=("brier", "sum"),
        logloss_sum=("logloss", "sum"),
        n=("brier", "size"),
    )
    return cases, events


def _anchor_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    return diagnostics.groupby(
        ["scenario", "event", "arm", "source", "adapter_policy"],
        observed=True,
        as_index=False,
    ).agg(
        cases=("round_num", "size"),
        mean_integer_support_shift=("integer_support_shift", "mean"),
        mean_abs_integer_support_shift=(
            "integer_support_shift", lambda values: float(np.mean(np.abs(values)))
        ),
        max_abs_integer_support_shift=(
            "integer_support_shift", lambda values: float(np.max(np.abs(values)))
        ),
        max_abs_post_shift_residual=(
            "post_shift_residual", lambda values: float(np.max(np.abs(values)))
        ),
        mean_raw_sd=(
            "raw_variance", lambda values: float(np.mean(np.sqrt(np.maximum(values, 0.0))))
        ),
        mean_power_temperature=("power_temperature", "mean"),
        min_power_temperature=("power_temperature", "min"),
        max_power_temperature=("power_temperature", "max"),
        max_clipped_mass=("clipped_mass", "max"),
        max_abs_expected_mean_error=(
            "expected_mean_error", lambda values: float(np.max(np.abs(values)))
        ),
        max_abs_expected_variance_error=(
            "expected_variance_error", lambda values: float(np.max(np.abs(values)))
        ),
    )


def _attach_eligibility(headline: pd.DataFrame, particles: int) -> pd.DataFrame:
    output = headline.copy()
    output["candidate_particles_per_case_min"] = int(particles)
    output["minimum_particles_required"] = output["scope"].map(
        MIN_PARTICLES_BY_SCOPE
    ).astype(int)
    output["decision_eligible"] = (
        output["candidate_particles_per_case_min"]
        >= output["minimum_particles_required"]
    )
    output["eligibility_note"] = np.where(
        output["decision_eligible"],
        "meets predeclared Monte Carlo floor; statistical uncertainty still applies",
        "diagnostic only: candidate particle count below predeclared Monte Carlo floor",
    )
    return output


def _markdown_table(frame: pd.DataFrame, decimals: int = 7) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(
            lambda value: f"{value:.{decimals}f}" if pd.notna(value) else ""
        )
    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(map(str, row)) + " |"
        for row in display.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def _write_report(
    output: Path,
    headline: pd.DataFrame,
    anchor_summary: pd.DataFrame,
    *,
    week_latent_sd: float,
    requested_latent_bins: int,
    effective_latent_bins: int,
    pairs: pd.DataFrame,
) -> None:
    focus = headline[
        headline["arm"].str.startswith("spatial:")
        & headline["scope"].isin(
            ["round_score_crps", "r1_h2h_brier", "all_round_h2h_brier"]
        )
    ][
        [
            "scenario",
            "scope",
            "arm",
            "delta_vs_full_dispersion",
            "cluster95_low",
            "cluster95_high",
            "events_favorable",
            "decision_eligible",
        ]
    ]
    calibrated = anchor_summary[
        anchor_summary["adapter_policy"].eq("common_discrete_location_week_latent")
    ]
    lines = [
        "# Common-lattice spatial-chain sensitivity",
        "",
        "## Decision readout",
        "",
        "Negative loss deltas favor the spatial arm over frozen full-strength "
        "`category_shot` dispersion. The reference-relative scenario is primary "
        "because it leaves every historical baseline probability unchanged. The "
        "symmetric-physical scenario is a deployable-score sensitivity in which "
        "all arms receive the identical discrete adapter.",
        "",
        _markdown_table(focus),
        "",
        "## Exact coordinate contracts",
        "",
        "Primary `reference_relative`: baseline support/probability is frozen. "
        "Candidate raw support is `round_strokes - round_par`; its target mean is "
        "`-target_sg`; the realized outcome is "
        "`actual_strokes - field_scoring_center`.",
        "",
        "Sensitivity `symmetric_physical`: candidate raw support is "
        "`round_strokes`; baseline raw support is "
        "`score_value_relative + round_par`; every target mean is "
        "`field_scoring_center - target_sg`; the realized outcome is "
        "`actual_strokes`.",
        "",
        "For every calibrated PMF, `k = rint(target - raw_mean)`, translated "
        "support is `raw_support + k`, and the residual exponential/power adapter "
        "solves the exact target mean while preserving the arm's raw variance. "
        f"The shared player-week marginal uses the exact finite grid of the "
        f"authoritative 128 aligned paths: {effective_latent_bins} effective "
        f"standardized normal-quantile nodes after the path adapter's "
        f"requested-{requested_latent_bins} sparsity clamp, at SD "
        f"{week_latent_sd:.2f}.",
        "",
        f"H2H selection is frozen once from the untouched category-current PMFs: "
        f"{len(pairs)} non-tied 35%-65% pairs. No scenario selects its own cases.",
        "",
        "## Moment audit",
        "",
        _markdown_table(
            calibrated[
                [
                    "scenario",
                    "event",
                    "arm",
                    "cases",
                    "max_clipped_mass",
                    "max_abs_expected_mean_error",
                    "max_abs_expected_variance_error",
                ]
            ]
        ),
        "",
        "## Limits",
        "",
        "- Only 12 pre-event players per event and three events are represented; "
        "event-cluster intervals are necessarily coarse.",
        "- Candidate exact-score log PMF remains below its predeclared particle "
        "floor; it is diagnostic only.",
        "- Frozen reference PMFs were already formed by rounding continuous SG "
        "onto the nuisance-relative integer lattice. Their exact counterfactual "
        "`rint(field_center - SG)` physical scores cannot be recovered from the "
        "saved PMFs. Therefore `symmetric_physical` is a principled common-adapter "
        "sensitivity, not an exact replay of the category producer; an exact replay "
        "would require the original continuous draws.",
        "- The comparison validates one-round shape and H2H. The 128 aligned paths "
        "are not yet adequate for full-field cut/top-5/win calibration.",
        "- This evaluator does not modify production simulations, prices, caches, "
        "or deployments.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_common_lattice_evaluation(
    complete_path: Path,
    baseline_path: Path,
    output: Path,
    *,
    week_latent_sd: float = 0.65,
    latent_bins: int = 31,
) -> dict[str, object]:
    complete = load_complete_round_particles(complete_path)
    frozen = load_frozen_case_pmfs(baseline_path)
    complete, baselines = join_frozen_case_truth(complete, frozen)
    candidate_raw = _candidate_raw_pmfs(complete)
    baseline_raw = _baseline_raw_pmfs(baselines, complete)
    pairs = frozen_market_like_pairs(baseline_raw)

    particles = int(
        complete.groupby(
            ["event", "source_player_id", "candidate_arm", "round_num"],
            observed=True,
        )["particle_id"].nunique().min()
    )
    latent_nodes, node_weights = finite_particle_latent_grid(particles, latent_bins)
    all_pmfs: list[pd.DataFrame] = []
    all_diagnostics: list[pd.DataFrame] = []
    all_cases: list[pd.DataFrame] = []
    all_h2h_cases: list[pd.DataFrame] = []
    all_h2h_events: list[pd.DataFrame] = []
    all_headlines: list[pd.DataFrame] = []
    for scenario in SCENARIOS:
        pmfs, diagnostics = calibrate_scenario_pmfs(
            candidate_raw,
            baseline_raw,
            scenario=scenario,
            week_latent_sd=week_latent_sd,
            latent_nodes=latent_nodes,
            node_weights=node_weights,
        )
        cases = score_common_pmfs(pmfs)
        h2h_cases, h2h_events = score_common_h2h(pmfs, pairs)
        baseline_cases = cases[cases["arm"].isin(BASELINE_ARMS)]
        candidate_cases = cases[cases["arm"].str.startswith("spatial:")]
        headline = common_candidate_headline(
            baseline_cases,
            candidate_cases,
            h2h_events.drop(columns="scenario"),
        )
        headline.insert(0, "scenario", scenario)
        headline = _attach_eligibility(headline, particles)
        all_pmfs.append(pmfs)
        all_diagnostics.append(diagnostics)
        all_cases.append(cases)
        all_h2h_cases.append(h2h_cases)
        all_h2h_events.append(h2h_events)
        all_headlines.append(headline)

    pmfs = pd.concat(all_pmfs, ignore_index=True)
    diagnostics = pd.concat(all_diagnostics, ignore_index=True)
    cases = pd.concat(all_cases, ignore_index=True)
    h2h_cases = pd.concat(all_h2h_cases, ignore_index=True)
    h2h_events = pd.concat(all_h2h_events, ignore_index=True)
    headline = pd.concat(all_headlines, ignore_index=True)
    anchor_summary = _anchor_summary(diagnostics)

    output.mkdir(parents=True, exist_ok=True)
    pmfs.to_parquet(output / "common_lattice_pmfs.parquet", index=False)
    diagnostics.to_csv(output / "common_lattice_case_anchor_diagnostics.csv", index=False)
    anchor_summary.to_csv(output / "common_lattice_anchor_summary.csv", index=False)
    cases.to_csv(output / "common_lattice_round_case_metrics.csv", index=False)
    pairs.to_csv(output / "frozen_market_like_pairs.csv", index=False)
    h2h_cases.to_csv(output / "common_lattice_h2h_case_metrics.csv", index=False)
    h2h_events.to_csv(output / "common_lattice_h2h_event_metrics.csv", index=False)
    headline.to_csv(output / "common_lattice_headline.csv", index=False)
    _write_report(
        output,
        headline,
        anchor_summary,
        week_latent_sd=week_latent_sd,
        requested_latent_bins=latent_bins,
        effective_latent_bins=len(latent_nodes),
        pairs=pairs,
    )
    manifest = {
        "status": "complete",
        "research_only": True,
        "scenarios": list(SCENARIOS),
        "primary_scenario": REFERENCE_RELATIVE,
        "week_latent_sd": float(week_latent_sd),
        "requested_latent_bins": int(latent_bins),
        "effective_latent_bins": int(len(latent_nodes)),
        "latent_nodes": latent_nodes.tolist(),
        "latent_node_weights": node_weights.tolist(),
        "candidate_particles_per_case": particles,
        "events": sorted(complete["event"].unique().tolist()),
        "observed_player_rounds": int(
            complete.loc[
                complete["observed_case"],
                ["event", "source_player_id", "round_num"],
            ].drop_duplicates().shape[0]
        ),
        "frozen_h2h_pairs": int(len(pairs)),
        "complete_round_particles": str(complete_path.resolve()),
        "complete_round_particles_sha256": sha256_file(complete_path),
        "baseline_case_pmfs": str(baseline_path.resolve()),
        "baseline_case_pmfs_sha256": sha256_file(baseline_path),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "adapter_sha256": hashlib.sha256(
            (ARCHIVE / "hole_sim_calibration_adapter.py").read_bytes()
        ).hexdigest(),
        "source_baselines_unchanged_in_primary": True,
        "production_modified": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--complete-round-particles",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--baseline-case-pmfs",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--week-latent-sd", type=float, default=0.65)
    parser.add_argument("--latent-bins", type=int, default=31)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.week_latent_sd < 0.0:
        raise ValueError("--week-latent-sd must be non-negative")
    if args.latent_bins < 3:
        raise ValueError("--latent-bins must be at least 3")
    manifest = run_common_lattice_evaluation(
        args.complete_round_particles,
        args.baseline_case_pmfs,
        args.output,
        week_latent_sd=args.week_latent_sd,
        latent_bins=args.latent_bins,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
