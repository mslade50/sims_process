from __future__ import annotations

import numpy as np
import pandas as pd

from archive.evaluate_spatial_chain_rounds import (
    BASELINE_ARMS,
    baseline_common_case_metrics,
    calibrate_complete_event,
    calibrated_common_baseline_pmfs,
    calibrated_candidate_pmfs,
    candidate_pmf_metrics,
    candidate_event_metrics,
    common_candidate_headline,
    common_matchup_metrics,
    join_frozen_case_truth,
    score_round_distribution,
)


def _synthetic_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    event = "Sony Open in Hawaii"
    players = ["alpha", "bravo", "charlie"]
    source_ids = {player: str(index + 1) for index, player in enumerate(players)}
    actual = {"alpha": 68.0, "bravo": 70.0, "charlie": 72.0}
    support = np.arange(-6, 7)
    probability = np.exp(-0.5 * (support / 2.8) ** 2)
    probability /= probability.sum()
    baseline_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(42)
    for player in players:
        for round_num in range(1, 5):
            for arm in BASELINE_ARMS:
                for score, mass in zip(support, probability):
                    baseline_rows.append(
                        {
                            "event": event,
                            "event_start": pd.Timestamp("2026-01-15"),
                            "player_name": player,
                            "source_player_id": source_ids[player],
                            "round_num": round_num,
                            "arm": arm,
                            "score_value_relative": int(score),
                            "probability": float(mass),
                            "actual_strokes": actual[player],
                            "actual_relative": actual[player] - 68.75,
                            "field_scoring_center": 68.75,
                            "target_sg": 0.0,
                            "skill_date": pd.Timestamp("2026-01-15"),
                        }
                    )
            draws = np.rint(70.0 + 2.8 * rng.standard_normal(400)).astype(int)
            for particle_id, strokes in enumerate(draws):
                candidate_rows.append(
                    {
                        "event_key": "pga:R2026006",
                        "event": event,
                        "player_name": player,
                        "source_player_id": source_ids[player],
                        "candidate_arm": "population",
                        "round_num": round_num,
                        "particle_id": particle_id,
                        "round_strokes": int(strokes),
                        "round_par": 70,
                        "terminal_holes": 18,
                        "input_cutoff": pd.Timestamp("2026-01-14"),
                        "chain_version": "synthetic-v1",
                    }
                )
    return pd.DataFrame(candidate_rows), pd.DataFrame(baseline_rows)


def test_exact_common_roster_pipeline_scores_candidate_and_h2h() -> None:
    candidate, frozen = _synthetic_panels()
    joined, common = join_frozen_case_truth(candidate, frozen)
    calibrated = calibrate_complete_event(joined, 0.65, 19)
    assert {
        "raw_mean_relative", "target_mean_relative", "mean_anchor_shift",
        "integer_support_shift", "post_shift_residual_mean_adjustment",
        "anchor_expected_mean_error", "anchor_expected_variance_error",
        "anchor_sample_mean_error", "anchor_sample_variance_error",
        "anchor_power_temperature", "anchor_clipped_mass",
        "calibrated_pmf_support", "calibrated_pmf_probability",
    }.issubset(calibrated.columns)
    marginal_rows = calibrated.drop_duplicates(
        ["event", "source_player_id", "candidate_arm", "round_num"]
    )
    assert all(
        np.isclose(sum(probability), 1.0)
        for probability in marginal_rows["calibrated_pmf_probability"]
    )
    assert np.allclose(marginal_rows["target_mean_relative"], 0.0)
    assert min(min(values) for values in marginal_rows["calibrated_pmf_support"]) > 50
    candidate_cases, _ = candidate_event_metrics(calibrated)
    baseline_pmfs = calibrated_common_baseline_pmfs(common, 0.65)
    baseline_means = baseline_pmfs.assign(
        weighted_score=(
            baseline_pmfs["score_value_physical"]
            * baseline_pmfs["probability"]
        )
    ).groupby(
        ["event", "player_name", "round_num", "arm"], observed=True
    )["weighted_score"].sum()
    assert np.allclose(baseline_means.to_numpy(float), 68.75, atol=1e-8)
    assert baseline_pmfs["score_value_physical"].min() > 50
    baseline_cases = baseline_common_case_metrics(baseline_pmfs)
    candidate_pmfs = calibrated_candidate_pmfs(calibrated)
    matchup_cases, matchup_events = common_matchup_metrics(
        baseline_pmfs, candidate_pmfs, selection_pmfs=common
    )
    headline = common_candidate_headline(
        baseline_cases, candidate_cases, matchup_events
    )

    assert not candidate_cases.empty
    assert set(candidate_cases["actual"]) == {68.0, 70.0, 72.0}
    assert not matchup_cases.empty
    assert "spatial:population" in set(headline["arm"])
    assert set(BASELINE_ARMS).issubset(set(headline["arm"]))
    assert set(headline["scope"]) == {
        "round_score_crps",
        "round_score_logpmf",
        "r1_h2h_brier",
        "r1_h2h_logloss",
        "all_round_h2h_brier",
        "all_round_h2h_logloss",
    }


def test_adapter_randomness_is_shared_across_identical_candidate_arms() -> None:
    candidate, frozen = _synthetic_panels()
    duplicate = candidate.copy()
    duplicate["candidate_arm"] = "identical_copy"
    joined, _ = join_frozen_case_truth(
        pd.concat([candidate, duplicate], ignore_index=True), frozen
    )
    calibrated = calibrate_complete_event(joined, 0.65, 19)
    left = calibrated[calibrated["candidate_arm"].eq("population")].sort_values(
        ["event", "source_player_id", "round_num", "particle_id"]
    )
    right = calibrated[calibrated["candidate_arm"].eq("identical_copy")].sort_values(
        ["event", "source_player_id", "round_num", "particle_id"]
    )
    assert np.array_equal(
        left["calibrated_round_relative"].to_numpy(),
        right["calibrated_round_relative"].to_numpy(),
    )


def test_four_round_candidate_scores_only_observed_cut_event_rounds() -> None:
    candidate, frozen = _synthetic_panels()
    frozen = frozen[
        ~(
            frozen["source_player_id"].eq("1")
            & frozen["round_num"].isin([3, 4])
        )
    ].copy()
    joined, common = join_frozen_case_truth(candidate, frozen)
    alpha = joined[joined["source_player_id"].eq("1")]
    assert set(alpha.loc[alpha["observed_case"], "round_num"]) == {1, 2}
    assert set(alpha.loc[~alpha["observed_case"], "round_num"]) == {3, 4}
    assert alpha["field_scoring_center"].notna().all()

    calibrated = calibrate_complete_event(joined, 0.65, 19)
    candidate_cases, _ = candidate_event_metrics(calibrated)
    candidate_pmfs = calibrated_candidate_pmfs(calibrated)
    baseline_pmfs = calibrated_common_baseline_pmfs(common, 0.65)
    baseline_cases = baseline_common_case_metrics(baseline_pmfs)
    matchup_cases, matchup_events = common_matchup_metrics(
        baseline_pmfs, candidate_pmfs, selection_pmfs=common
    )
    headline = common_candidate_headline(
        baseline_cases, candidate_cases, matchup_events
    )

    assert len(candidate_cases) == 10
    assert len(baseline_cases) == 10 * len(BASELINE_ARMS)
    assert not matchup_cases.empty
    assert not headline.empty


def test_candidate_uses_fair_finite_ensemble_crps() -> None:
    metrics = score_round_distribution(np.asarray([0, 2]), actual=1.0)
    assert np.isclose(metrics["crps_empirical_plugin"], 0.5)
    assert np.isclose(metrics["crps"], 0.0)
    assert np.isclose(metrics["crps_finite_ensemble_correction"], -0.5)


def test_marginal_pmf_path_scores_on_translated_support() -> None:
    frame = pd.DataFrame(
        {
            "event_key": ["pga:test"] * 3,
            "event": ["Test Event"] * 3,
            "player_name": ["alpha"] * 3,
            "round_num": [1] * 3,
            "score_value": [68, 69, 70],
            "probability": [0.25, 0.50, 0.25],
            "actual_strokes": [65.0] * 3,
            "target_sg": [0.0] * 3,
            "round_par": [65.0] * 3,
        }
    )

    cases, _ = candidate_pmf_metrics(frame, week_latent_sd=0.0)

    assert len(cases) == 1
    assert np.isclose(float(cases.iloc[0]["pred_mean"]), 65.0, atol=1e-10)
    assert np.isfinite(float(cases.iloc[0]["logpmf"]))
