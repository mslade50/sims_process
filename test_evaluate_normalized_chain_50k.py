from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from archive import hole_sim_tail_policy as tail_policy

from archive.evaluate_normalized_chain_50k import (
    ARMS,
    LATTICE_MIN_STROKES,
    MECHANICS_ANCHOR_STATUS,
    NORMALIZATION_MODE,
    PHYSICAL_STROKE_LATTICE,
    REQUIRED_ANCHOR_POLICY,
    REQUIRED_ANCHOR_STATUS,
    EvaluationContractError,
    PredictiveScoringBlocked,
    audit_schema_only,
    crps_common_lattice,
    evaluate_panel,
    h2h_probabilities_from_pmfs,
    load_case_index,
    load_particle_panel,
    load_truth,
    pmf_on_common_lattice,
    sha256_file,
)


def _write_bundle(
    root: Path,
    *,
    player_id: str = "101",
    event_key: str = "pga:RTEST",
    particles: int = 5_000,
    anchored: bool = False,
    particle_axis: str = "shared",
    raw_values: np.ndarray | None = None,
    normalized_values: np.ndarray | None = None,
) -> Path:
    root.mkdir(parents=True)
    if raw_values is None:
        raw_values = np.resize(np.asarray([68, 69, 70, 71, 72], np.int16), particles)
    if normalized_values is None:
        normalized_values = np.resize(
            np.asarray([66, 68, 70, 72, 74], np.int16), particles
        )
    totals = np.stack(
        [
            np.stack([raw_values] * 4),
            np.stack([normalized_values] * 4),
        ]
    ).astype(np.int16)
    payload: dict[str, np.ndarray] = {
        "arm_names": np.asarray(ARMS),
        "round_totals": totals,
        "round_numbers": np.asarray([1, 2, 3, 4], np.int8),
        "event_key": np.asarray([event_key]),
        "player_id": np.asarray([player_id]),
    }
    if particle_axis == "shared":
        payload["particle_ids"] = np.arange(particles, dtype=np.int64)
    elif particle_axis == "by_arm_mismatch":
        payload["particle_ids_by_arm"] = np.stack(
            [np.arange(particles), np.arange(particles)[::-1]]
        )
    np.savez_compressed(root / "round_totals_int16.npz", **payload)
    totals_hash = sha256_file(root / "round_totals_int16.npz")
    anchor_status = REQUIRED_ANCHOR_STATUS if anchored else MECHANICS_ANCHOR_STATUS
    manifest = {
        "artifact": "synthetic_streaming_fixture",
        "research_only": True,
        "production_touched": False,
        "tail_policy": tail_policy.manifest_policy(),
        "event_key": event_key,
        "player_id": player_id,
        "arms": list(ARMS),
        "normalization_mode": NORMALIZATION_MODE,
        "ability_log_weight_per_sg2": 0.0,
        "round_totals_sha256": totals_hash,
        "streaming_source_start_sha256": "a" * 64,
        "streaming_source_end_sha256": "a" * 64,
        "streaming_contract": {
            "survivor_filtering": False,
            "unsupported_states_fail_closed": True,
        },
        "rng_modes": {
            "benchmark": "counter",
            "counter_version": "synthetic_counter_v1",
            "arm_in_key": False,
            "chunk_in_key": False,
        },
        "input_hashes_sha256": {"synthetic_input": "b" * 64},
        "raw_128_gate": {
            "counter_50k_prefix_of_100k": {"exact": True}
        },
        "transform_order": [
            "donor_native_asof_normalize",
            "rigid_align_neutral_endpoint_exactly_once",
            "target_profile_reapply_beta_zero",
            "runtime_physical_surface_reclassification",
        ],
        "alignment_count": 1,
        "mass_contract": "base_hierarchy_mass_times_mean_one_model_weight",
        "production_mean_anchor_status": anchor_status,
    }
    if anchored:
        manifest["mean_anchor_contract"] = {
            "production_mean_anchor_status": REQUIRED_ANCHOR_STATUS,
            "owner": "production_skill",
            "policy": REQUIRED_ANCHOR_POLICY,
            "applied_stage": "generator_before_persistence",
            "same_target_for_all_arms": True,
            "posthoc_arm_specific_adjustment": False,
            "evaluator_adjustment": False,
        }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _case_rows(
    players: tuple[str, ...] = ("101",),
    *,
    event_key: str = "pga:RTEST",
    cut_rule: str = "no_cut",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for player in players:
        for round_num in range(1, 5):
            rows.append(
                {
                    "case_id": f"{event_key}|{player}",
                    "event_key": event_key,
                    "event_name": "Synthetic Open",
                    "event_start": "2026-06-04T12:00:00Z",
                    "source_player_id": player,
                    "player_name": f"player {player}",
                    "round_num": round_num,
                    "round_par": 70,
                    "cut_rule": cut_rule,
                    "pre_event_roster": True,
                    "roster_frozen_at": "2026-06-01T12:00:00Z",
                    "skill_asof": "2026-06-01T00:00:00Z",
                    "input_cutoff": "2026-06-03T00:00:00Z",
                    "production_skill_sg": 0.5,
                    "production_field_mean_strokes": 70.5,
                    "production_mean_strokes": 70.0,
                    "generator_expected_mean_strokes_raw": 70.0,
                    "generator_expected_mean_strokes_normalized": 70.0,
                    "mean_anchor_policy": REQUIRED_ANCHOR_POLICY,
                }
            )
    return pd.DataFrame(rows)


def _truth_rows(
    players: tuple[str, ...] = ("101",),
    *,
    event_key: str = "pga:RTEST",
    rounds: tuple[int, ...] = (1, 2, 3, 4),
) -> pd.DataFrame:
    scores = {1: 69, 2: 70, 3: 72, 4: 68}
    return pd.DataFrame(
        [
            {
                "event_key": event_key,
                "source_player_id": player,
                "round_num": round_num,
                "actual_strokes": scores[round_num] + (1 if player == "202" else 0),
            }
            for player in players
            for round_num in rounds
        ]
    )


def test_fixed_common_lattice_and_crps_do_not_depend_on_sample_support() -> None:
    at_actual = pmf_on_common_lattice(np.asarray([70] * 50), alpha=0.0)
    one_stroke_high = pmf_on_common_lattice(np.asarray([71] * 50), alpha=0.0)

    assert len(at_actual) == len(PHYSICAL_STROKE_LATTICE) == 101
    assert at_actual[70 - LATTICE_MIN_STROKES] == 1.0
    assert crps_common_lattice(at_actual, 70) == 0.0
    assert crps_common_lattice(one_stroke_high, 70) == 1.0


def test_five_thousand_is_accepted_for_schema_but_predictive_scoring_is_blocked(
    tmp_path: Path,
) -> None:
    bundle = _write_bundle(tmp_path / "bundle", particles=5_000, anchored=False)
    panel = load_particle_panel([bundle])
    audit = audit_schema_only(panel)

    assert audit["schema_valid"] is True
    assert audit["engineering_particle_floor_passed"] is True
    assert audit["decision_particle_floor_passed"] is False
    assert audit["predictive_scoring_permitted"] is False
    with pytest.raises(PredictiveScoringBlocked, match="at least 50,000"):
        evaluate_panel(
            panel,
            case_index_path=tmp_path / "not_read.parquet",
            truth_path=tmp_path / "not_read_either.parquet",
        )


def test_unanchored_fifty_thousand_remains_mechanics_only(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "bundle", particles=50_000, anchored=False)
    panel = load_particle_panel([bundle])

    with pytest.raises(PredictiveScoringBlocked, match="generator-time shared"):
        evaluate_panel(
            panel,
            case_index_path=tmp_path / "not_read.parquet",
            truth_path=tmp_path / "not_read_either.parquet",
        )


def test_misaligned_particle_axes_fail_closed(tmp_path: Path) -> None:
    bundle = _write_bundle(
        tmp_path / "bundle",
        particles=5_000,
        particle_axis="by_arm_mismatch",
    )
    with pytest.raises(EvaluationContractError, match="particle axes differ"):
        load_particle_panel([bundle])


def test_missing_or_mutated_tail_policy_fails_evaluator_schema(
    tmp_path: Path,
) -> None:
    for mutation in ("missing", "score_cap"):
        bundle = _write_bundle(
            tmp_path / f"bundle_{mutation}", particles=5_000
        )
        manifest_path = bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if mutation == "missing":
            manifest.pop("tail_policy")
        else:
            manifest["tail_policy"]["physical_hole_score_cap"] = 15
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(EvaluationContractError, match="tail_policy"):
            load_particle_panel([bundle])


def test_implicit_particle_axis_is_mechanics_only_even_at_fifty_thousand(
    tmp_path: Path,
) -> None:
    bundle = _write_bundle(
        tmp_path / "bundle",
        particles=50_000,
        anchored=True,
        particle_axis="implicit",
    )
    panel = load_particle_panel([bundle])

    with pytest.raises(PredictiveScoringBlocked, match="explicit particle_ids"):
        evaluate_panel(
            panel,
            case_index_path=tmp_path / "not_read.parquet",
            truth_path=tmp_path / "not_read_either.parquet",
        )


def test_anchored_fifty_thousand_scores_unchanged_but_one_event_is_not_a_decision(
    tmp_path: Path,
) -> None:
    bundle = _write_bundle(tmp_path / "bundle", particles=50_000, anchored=True)
    cases_path = tmp_path / "cases.parquet"
    truth_path = tmp_path / "truth.parquet"
    _case_rows().to_parquet(cases_path, index=False)
    _truth_rows().to_parquet(truth_path, index=False)
    panel = load_particle_panel([bundle])

    result = evaluate_panel(
        panel,
        case_index_path=cases_path,
        truth_path=truth_path,
    )

    assert len(result.case_metrics) == 8  # 4 rounds x 2 arms
    assert len(result.paired_case_deltas) == 4
    assert result.decision_status["predictive_scoring_permitted"] is True
    assert result.decision_status["particle_precision_gate"] is True
    assert result.decision_status["sample_anchor_consistency_gate"] is True
    assert result.decision_status["broad_heldout_panel_gate"] is False
    assert result.decision_status["decision_claim_permitted"] is False
    assert set(result.case_metrics["evaluation_scope"]) == {"primary_no_survivor"}
    assert np.allclose(
        result.anchor_audit["sample_mean_error"].to_numpy(float), 0.0
    )


def test_generator_expected_means_must_equal_shared_production_anchor(
    tmp_path: Path,
) -> None:
    panel = load_particle_panel(
        [_write_bundle(tmp_path / "bundle", particles=50_000, anchored=True)]
    )
    cases = _case_rows()
    cases.loc[cases["round_num"].eq(1), "generator_expected_mean_strokes_normalized"] = 70.1
    path = tmp_path / "cases.parquet"
    cases.to_parquet(path, index=False)

    with pytest.raises(EvaluationContractError, match="does not equal"):
        load_case_index(path, panel)


def test_cut_event_primary_truth_is_r1_r2_without_survivor_selection(
    tmp_path: Path,
) -> None:
    panel = load_particle_panel(
        [_write_bundle(tmp_path / "bundle", particles=50_000, anchored=True)]
    )
    cases_path = tmp_path / "cases.parquet"
    truth_path = tmp_path / "truth.parquet"
    _case_rows(cut_rule="cut_after_r2").to_parquet(cases_path, index=False)
    _truth_rows(rounds=(1, 2)).to_parquet(truth_path, index=False)
    cases = load_case_index(cases_path, panel)
    joined = load_truth(truth_path, cases)

    assert joined.loc[joined["round_num"].le(2), "actual_strokes"].notna().all()
    assert joined.loc[joined["round_num"].gt(2), "actual_strokes"].isna().all()
    assert set(joined.loc[joined["round_num"].le(2), "evaluation_scope"]) == {
        "primary_no_survivor"
    }
    assert set(joined.loc[joined["round_num"].gt(2), "evaluation_scope"]) == {
        "survivor_conditional_secondary"
    }


def test_h2h_uses_exact_common_lattice_marginal_convolution() -> None:
    a = pmf_on_common_lattice(np.asarray([69] * 10), alpha=0.0)
    b = pmf_on_common_lattice(np.asarray([70] * 10), alpha=0.0)
    win, tie, loss = h2h_probabilities_from_pmfs(a, b)
    assert (win, tie, loss) == (1.0, 0.0, 0.0)

    tied = h2h_probabilities_from_pmfs(b, b)
    assert tied == (0.0, 1.0, 0.0)


def test_frozen_h2h_roster_is_scored_on_both_arms_as_secondary(
    tmp_path: Path,
) -> None:
    bundles = [
        _write_bundle(
            tmp_path / f"bundle_{player}",
            player_id=player,
            particles=50_000,
            anchored=True,
        )
        for player in ("101", "202")
    ]
    cases_path = tmp_path / "cases.parquet"
    truth_path = tmp_path / "truth.parquet"
    pairs_path = tmp_path / "pairs.parquet"
    _case_rows(players=("101", "202")).to_parquet(cases_path, index=False)
    _truth_rows(players=("101", "202")).to_parquet(truth_path, index=False)
    pd.DataFrame(
        [
            {
                "pair_id": "r1:101:202",
                "event_key": "pga:RTEST",
                "round_num": 1,
                "player_a_id": "101",
                "player_b_id": "202",
                "pair_frozen_at": "2026-06-03T00:00:00Z",
                "pair_source": "synthetic_pre_event_market",
            }
        ]
    ).to_parquet(pairs_path, index=False)

    result = evaluate_panel(
        load_particle_panel(bundles),
        case_index_path=cases_path,
        truth_path=truth_path,
        h2h_pairs_path=pairs_path,
    )

    assert len(result.h2h_metrics) == 2
    assert len(result.paired_h2h_deltas) == 1
    assert set(result.h2h_metrics["arm"]) == set(ARMS)
    assert set(result.h2h_metrics["h2h_dependence_contract"]) == {
        "exact_independent_marginal_convolution"
    }
    assert result.decision_status["h2h_role"] == "secondary"
