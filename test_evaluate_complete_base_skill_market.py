from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from archive import backtest_base_skill_market_models as plumbing
from archive import evaluate_complete_base_skill_market_candidates as complete_eval
from archive.market_backtest_scoring import ties_lose


def _pin_synthetic_evaluation_cache(
    tmp_path: Path,
    monkeypatch,
) -> tuple[Path, Path, dict[str, Path], dict[str, object]]:
    cache = tmp_path / "cache"
    cache.mkdir()
    quotes = cache / "normalized_matchup_quotes.parquet"
    snapshots = cache / "feature_snapshots.parquet"
    quotes.write_bytes(b"synthetic normalized quotes")
    snapshots.write_bytes(b"synthetic strict-prior feature snapshots")
    contract = tmp_path / "BASE_SKILL_MARKET_REBUILD_CONTRACT_2026_08.md"
    contract.write_text("synthetic frozen contract\n", encoding="utf-8")
    input_paths = {
        "event_panel": tmp_path / "player_event_outcomes.parquet",
        "legacy_core_bridge": tmp_path / "legacy_core_bridge.parquet",
        "odds_cache_manifest": tmp_path / "odds_cache_manifest.json",
        "odds_event_crosswalk": tmp_path / "odds_event_crosswalk.parquet",
        "round_panel": tmp_path / "player_round_panel.parquet",
    }
    for name, path in input_paths.items():
        path.write_bytes(f"synthetic {name}".encode("utf-8"))
    artifacts = []
    for path in (quotes, snapshots):
        artifacts.append(
            {
                "path": path.name,
                "sha256": complete_eval._sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    inputs = {
        name: {
            "path": str(path.resolve()),
            "sha256": complete_eval._sha256(path),
        }
        for name, path in input_paths.items()
    }
    payload: dict[str, object] = {
        "complete": True,
        "research_only": True,
        "contract": str(contract.resolve()),
        "run_type": "shortlist_money_screen",
        "artifacts": artifacts,
        "inputs": inputs,
    }
    manifest_path = cache / "manifest.json"
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(complete_eval, "DEFAULT_CACHE", cache)
    monkeypatch.setattr(
        complete_eval,
        "EXPECTED_EVALUATION_CACHE_MANIFEST_SHA256",
        complete_eval._sha256(manifest_path),
    )
    monkeypatch.setattr(
        complete_eval,
        "EXPECTED_EVALUATION_CACHE_MANIFEST_BYTES",
        manifest_path.stat().st_size,
    )
    monkeypatch.setattr(
        complete_eval, "EXPECTED_EVALUATION_CACHE_CONTRACT", contract
    )
    monkeypatch.setattr(
        complete_eval,
        "EXPECTED_EVALUATION_CACHE_ARTIFACTS",
        {
            record["path"]: {
                "sha256": record["sha256"],
                "bytes": record["bytes"],
            }
            for record in artifacts
        },
    )
    monkeypatch.setattr(
        complete_eval,
        "EXPECTED_EVALUATION_CACHE_INPUTS",
        {
            name: {
                "path": path,
                "sha256": complete_eval._sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in input_paths.items()
        },
    )
    return cache, manifest_path, input_paths, payload


def _repin_synthetic_cache_manifest(monkeypatch, manifest_path: Path) -> None:
    monkeypatch.setattr(
        complete_eval,
        "EXPECTED_EVALUATION_CACHE_MANIFEST_SHA256",
        complete_eval._sha256(manifest_path),
    )
    monkeypatch.setattr(
        complete_eval,
        "EXPECTED_EVALUATION_CACHE_MANIFEST_BYTES",
        manifest_path.stat().st_size,
    )


def _pin_synthetic_completed_study(
    tmp_path: Path,
    monkeypatch,
) -> tuple[Path, dict[str, Path], dict[str, object]]:
    study = tmp_path / "completed_study"
    study.mkdir(parents=True)
    files = {
        "predictions": study / "player_event_predictions.parquet",
        "candidate_catalog": study / "candidate_catalog.csv",
        "coverage_by_candidate_year": study / "coverage_by_candidate_year.csv",
        "frozen_candidate_catalog": study / "candidate_catalog_frozen.json",
        "fit_log": study / "fit_log.csv",
        "envelope_admission_by_candidate": (
            study / "regression_envelope_admission_by_candidate.csv"
        ),
        "envelope_admission_by_fold": (
            study / "regression_envelope_admission_by_fold.csv"
        ),
    }
    for name, path in files.items():
        path.write_bytes(f"synthetic completed study {name}".encode("utf-8"))
    frozen_hash = complete_eval._sha256(files["frozen_candidate_catalog"])
    outputs = {
        path.name: {
            "path": str(path.resolve()),
            "sha256": complete_eval._sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in files.values()
    }
    payload: dict[str, object] = {
        "complete": True,
        "research_only": True,
        "production_files_changed": False,
        "frozen_candidate_catalog_sha256": frozen_hash,
        "outputs": outputs,
    }
    manifest_path = study / "manifest.json"
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(complete_eval, "DEFAULT_SOURCE_MANIFEST", manifest_path)
    monkeypatch.setattr(
        complete_eval, "EXPECTED_FROZEN_CATALOG_SHA256", frozen_hash
    )
    monkeypatch.setattr(
        complete_eval,
        "EXPECTED_COMPLETED_STUDY_MANIFEST_SHA256",
        complete_eval._sha256(manifest_path),
    )
    monkeypatch.setattr(
        complete_eval,
        "EXPECTED_COMPLETED_STUDY_MANIFEST_BYTES",
        manifest_path.stat().st_size,
    )
    monkeypatch.setattr(
        complete_eval,
        "EXPECTED_COMPLETED_STUDY_OUTPUTS",
        {
            path.name: {
                "sha256": complete_eval._sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in files.values()
        },
    )
    bindings = {"completion_manifest": manifest_path.resolve(), **files}
    bindings = {name: path.resolve() for name, path in bindings.items()}
    return manifest_path, bindings, payload


def _paired_fixture() -> pd.DataFrame:
    rows = []
    for year in (2020, 2021, 2022):
        for candidate in ("baseline", "challenger"):
            # Exercise the predeclared strict-PT fallback for the challenger's
            # 2021 mapper while retaining direct 2021 calibration for 2022.
            if candidate == "challenger" and year == 2020:
                continue
            for index in range(120):
                left = str(1000 + index)
                right = str(5000 + index)
                outcome = (1.0, 0.0, 0.5)[index % 3]
                rows.append(
                    {
                        "candidate": candidate,
                        "mapped_event_key": f"event-{year}-{index // 20}",
                        "event_start_date": pd.Timestamp(year=year, month=6, day=1),
                        "market": "r1",
                        "book": "pinnacle",
                        "pair_key": f"{left}|{right}",
                        "p1_id": left,
                        "p2_id": right,
                        "p1_close": -110.0,
                        "p2_close": -110.0,
                        "p1_outcome": outcome,
                        "skill_difference": (index % 11 - 5) / 10.0,
                        "quote_id": f"{candidate}-{year}-{index}",
                    }
                )
    return pd.DataFrame(rows)


def test_temporal_probability_mapper_uses_only_previous_year() -> None:
    mapper = complete_eval.fit_temporal_probability_mappers(
        _paired_fixture(), "baseline", min_pairs=100
    )
    assert (mapper["calibration_year"] == mapper["evaluation_year"] - 1).all()
    challenger_2021 = mapper.loc[
        mapper["candidate"].eq("challenger")
        & mapper["evaluation_year"].eq(2021)
        & mapper["market"].eq("r1")
    ].iloc[0]
    challenger_2022 = mapper.loc[
        mapper["candidate"].eq("challenger")
        & mapper["evaluation_year"].eq(2022)
        & mapper["market"].eq("r1")
    ].iloc[0]
    assert challenger_2021["calibration_source"].startswith(
        "strict_pt_baseline_fallback"
    )
    assert challenger_2022["calibration_source"] == "candidate_previous_year"


def test_selection_gates_fail_closed_and_choose_only_full_pass() -> None:
    common = {
        "minimum_coverage_ratio": 1.0,
        "pooled_brier_delta": -0.001,
        "calibration_fit_success": True,
        "calibration_intercept_distance_delta": -0.001,
        "calibration_slope_error_delta": -0.001,
        "ece_delta": -0.001,
        "edge_violation_magnitude_delta": -0.001,
        "close_pnl_delta": 1.0,
        "close_roi_delta": 0.01,
        "candidate_mean_open_to_close_clv": 0.001,
        "open_to_close_clv_delta": 0.001,
        "max_drawdown_delta": -1.0,
        "worst_event_loss_delta": -1.0,
    }
    scorecard = pd.DataFrame(
        [
            {
                "candidate": "good",
                "pooled_log_loss_delta": -0.001,
                "worst_market_log_loss_delta": -0.0001,
                **common,
            },
            {
                "candidate": "bad",
                "pooled_log_loss_delta": 0.001,
                "worst_market_log_loss_delta": 0.001,
                **common,
            },
        ]
    )
    gated = complete_eval.apply_selection_gates(scorecard)
    assert gated.set_index("candidate").loc["good", "all_gates_pass"]
    assert not gated.set_index("candidate").loc["bad", "all_gates_pass"]
    catalog = pd.DataFrame(
        {
            "name": ["good", "bad"],
            "family": ["ridge", "ridge"],
            "selection_eligible": [True, True],
            "sensitivity_only": [False, False],
            "control": [False, False],
        }
    )
    assert complete_eval.choose_finalists(gated, catalog, 1) == ["good"]


def test_empty_guardrail_screen_continues_fail_closed_without_finalist() -> None:
    aggregate = pd.DataFrame(
        [
            {
                "candidate": "diagnostic_only",
                "minimum_coverage_ratio": 1.0,
                "pooled_log_loss_delta": -0.001,
                "worst_market_log_loss_delta": 0.001,
                "pooled_brier_delta": -0.001,
                "calibration_fit_success": True,
                "calibration_intercept_distance_delta": -0.001,
                "calibration_slope_error_delta": -0.001,
                "ece_delta": -0.001,
            }
        ]
    )
    guardrails = complete_eval.guardrail_summary_frame([])
    assert list(guardrails.columns) == ["candidate"]
    assert guardrails.empty

    scorecard = aggregate.merge(guardrails, on="candidate", how="left")
    adverse_defaults = {
        "close_pnl_delta": -np.inf,
        "close_roi_delta": -np.inf,
        "candidate_mean_open_to_close_clv": -np.inf,
        "open_to_close_clv_delta": -np.inf,
        "max_drawdown_delta": np.inf,
        "worst_event_loss_delta": np.inf,
        "edge_violation_magnitude_delta": np.inf,
    }
    for column, default in adverse_defaults.items():
        scorecard[column] = default
    gated = complete_eval.apply_selection_gates(scorecard)
    catalog = pd.DataFrame(
        {
            "name": ["diagnostic_only"],
            "family": ["ridge"],
            "selection_eligible": [True],
            "sensitivity_only": [False],
            "control": [False],
        }
    )

    assert not gated.loc[0, "proper_gate_pass"]
    assert not gated.loc[0, "money_gate_pass"]
    assert not gated.loc[0, "concentration_gate_pass"]
    assert not gated.loc[0, "all_gates_pass"]
    assert complete_eval.choose_finalists(gated, catalog, 1) == []
    diagnostic, failed = complete_eval.choose_diagnostic_challenger(gated, catalog)
    assert diagnostic == "diagnostic_only"
    assert "proper_gate" in failed
    assert "money_gate" in failed
    assert "concentration_gate" in failed


def test_elastic_cd_has_declared_complexity_rank_in_tie_break() -> None:
    scorecard = pd.DataFrame(
        {
            "candidate": ["a_elastic", "z_ridge"],
            "all_gates_pass": [True, True],
            "pooled_log_loss_delta": [-0.01, -0.01],
            "pooled_brier_delta": [-0.01, -0.01],
            "calibration_slope_error_delta": [-0.01, -0.01],
            "close_pnl_delta": [1.0, 1.0],
            "max_drawdown_delta": [-1.0, -1.0],
        }
    )
    catalog = pd.DataFrame(
        {
            "name": ["a_elastic", "z_ridge"],
            "family": ["elastic_cd", "ridge"],
            "selection_eligible": [True, True],
            "sensitivity_only": [False, False],
            "control": [False, False],
        }
    )
    assert complete_eval.choose_finalists(scorecard, catalog, 1) == ["z_ridge"]


def test_locked_ablation_set_is_same_spec_and_records_missing_roles() -> None:
    common = {
        "family": "ridge",
        "refit_schedule": "quarterly",
        "alpha": 10.0,
        "short_game_penalty": 1.0,
    }
    catalog = pd.DataFrame(
        [
            {
                "name": "model_w0",
                "source": "total_cat",
                "weather_arm": "w0",
                "target_arm": "observed",
                **common,
            },
            {
                "name": "model_w1_feature",
                "source": "total_cat",
                "weather_arm": "w1",
                "target_arm": "feature_only",
                **common,
            },
            {
                "name": "model_w1_neutral",
                "source": "total_cat",
                "weather_arm": "w1",
                "target_arm": "neutral_target",
                **common,
            },
            {
                "name": "model_robust_feature",
                "source": "total_cat",
                "weather_arm": "w1_robust_covariate",
                "target_arm": "feature_only",
                **common,
            },
            {
                "name": "model_img_w0",
                "source": "total_cat_img",
                "weather_arm": "w0",
                "target_arm": "observed",
                **common,
            },
        ]
    )
    locked = complete_eval.build_locked_diagnostic_ablation_set("model_w0", catalog)
    assert locked["locked_before_sealed_scoring"]
    assert not locked["may_reselect_or_promote_from_sealed"]
    assert locked["clean_img_no_img_sibling"]["names"] == ["model_img_w0"]
    assert set(locked["included_candidates"]) == {
        "model_w0",
        "model_w1_feature",
        "model_w1_neutral",
        "model_robust_feature",
        "model_img_w0",
    }
    assert "robust_w1_neutral_target" in locked["unavailable_requested_siblings"]


def test_only_declared_count_audited_controls_can_be_synthesized(tmp_path) -> None:
    controls = list(complete_eval.KNOWN_CONTROL_UNITS)
    predictions = pd.DataFrame(
        {"candidate": ["clean_candidate", *controls, *controls]}
    )
    catalog = pd.DataFrame(
        {
            "name": ["clean_candidate"],
            "family": ["ridge"],
            "selection_eligible": [True],
            "sensitivity_only": [False],
            "control": [False],
        }
    )
    coverage = (
        predictions.groupby("candidate", observed=True)
        .size()
        .rename("prediction_rows")
        .reset_index()
    )
    coverage_path = tmp_path / "coverage_by_candidate_year.csv"
    coverage.to_csv(coverage_path, index=False)
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "control_catalog": [
            {
                "name": name,
                "selection_eligible": False,
                "sensitivity_only": True,
                "control": True,
            }
            for name in controls
        ],
        "control_audit": {"verified": True},
        "outputs": {
            "coverage_by_candidate_year.csv": {
                "path": str(coverage_path),
                "sha256": complete_eval._sha256(coverage_path),
            }
        },
    }
    reconciled, audit = complete_eval.reconcile_known_controls(
        predictions, catalog, manifest, manifest_path
    )
    synthesized = reconciled.loc[reconciled["name"].isin(controls)]
    assert set(audit["synthesized_controls"]) == set(controls)
    assert not synthesized["selection_eligible"].any()
    assert synthesized["sensitivity_only"].all()
    assert synthesized["control"].all()

    bad = pd.concat(
        [predictions, pd.DataFrame({"candidate": ["unknown_control"]})],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="non-allowlisted"):
        complete_eval.reconcile_known_controls(bad, catalog, manifest, manifest_path)


def test_control_units_are_exactly_allowlisted() -> None:
    rows = [{"candidate": "clean", "prediction_units": "sg_per_round"}]
    rows.extend(
        {"candidate": name, "prediction_units": unit}
        for name, unit in complete_eval.KNOWN_CONTROL_UNITS.items()
    )
    accepted = complete_eval.validate_prediction_units(pd.DataFrame(rows))
    assert accepted == {
        "clean": "sg_per_round",
        **complete_eval.KNOWN_CONTROL_UNITS,
    }

    bad = pd.DataFrame(
        [
            {
                "candidate": "legacy_pred_step2_exact_retrospective_control",
                "prediction_units": "sg_per_round",
            }
        ]
    )
    with pytest.raises(ValueError, match="unit"):
        complete_eval.validate_prediction_units(bad)


def test_matchup_push_settlement_is_zero_and_ties_lose_is_loss() -> None:
    bets = pd.DataFrame(
        {
            "event_id": ["event", "event"],
            "event_key": ["event", "event"],
            "market": ["tournament_matchup", "tournament_matchup"],
            "selection_id": ["a", "b"],
            "opponent_id": ["b", "a"],
            "close_odds": [110.0, 110.0],
            "stake": [1.0, 1.0],
            "outcome_numeric": [0.5, 0.5],
            "tie_rule": ["ties push", "ties lose"],
        }
    )
    settled = plumbing.settle_bets(bets)
    assert settled.loc[0, "settlement_status"] == "push"
    assert settled.loc[0, "pnl"] == 0.0
    assert settled.loc[1, "settlement_status"] == "tie_loss"
    assert settled.loc[1, "pnl"] == -1.0


def test_canonical_probability_rows_give_one_vote_per_logical_pair() -> None:
    rows = []
    for book in ("betonline", "betcris", "pinnacle"):
        rows.append(
            {
                "candidate": "candidate",
                "mapped_event_key": "event",
                "event_start_date": pd.Timestamp("2021-06-01"),
                "market": "r1",
                "book": book,
                "pair_key": "1|2",
                "p1_id": "1",
                "p2_id": "2",
                "p1_close": -110.0,
                "p2_close": -110.0,
                "p1_outcome": 1.0,
                "skill_difference": 0.25,
                "p1_probability": 0.52,
                "push_probability": 0.04,
                "quote_id": f"event|r1|{book}|1|2",
            }
        )
    logical = complete_eval.canonical_discovery_probabilities(pd.DataFrame(rows))
    assert len(logical) == 1
    assert logical.iloc[0]["book"] == "pinnacle"
    assert not logical.duplicated(
        ["candidate", "mapped_event_key", "market", "pair_key"]
    ).any()


def test_pairwise_logistic_is_temporal_and_not_neutral_skill() -> None:
    scores, choices = complete_eval.pairwise_logistic_sensitivity(
        _paired_fixture(),
        "baseline",
        {"challenger"},
        regularization_c=0.1,
        min_pairs=100,
    )
    assert (scores["training_year"] == scores["evaluation_year"] - 1).all()
    assert set(scores["prediction_units"]) == {"market_logit_score"}
    assert not scores["neutral_base_finalist_eligible"].any()
    assert choices.loc[0, "candidate"] == "challenger"


def test_reliability_gate_suppresses_only_unreliable_large_edge() -> None:
    rows = []
    for index, (reliable, probability) in enumerate(
        ((False, 0.80), (True, 0.80), (False, 0.53)), start=1
    ):
        rows.append(
            {
                "candidate": "candidate",
                "quote_id": f"quote-{index}",
                "mapped_event_key": f"event-{index}",
                "event_start_date": pd.Timestamp("2021-06-01"),
                "market": "r1",
                "book": "pinnacle",
                "pair_key": f"{index}|{index + 10}",
                "p1_id": str(index),
                "p2_id": str(index + 10),
                "p1_player_name": "left",
                "p2_player_name": "right",
                "p1_close": 100.0,
                "p2_close": -110.0,
                "p1_outcome": 1.0,
                "p2_outcome": 0.0,
                "p1_outcome_text": "win",
                "p2_outcome_text": "loss",
                "tie_rule": "ties push",
                "pair_reliability_ok": reliable,
                "p1_probability": probability,
                "push_probability": 0.02,
            }
        )
    bets, suppressed = complete_eval.select_reliability_gated_bets(
        pd.DataFrame(rows), min_ev=0.03
    )
    assert len(bets) == 2
    assert len(suppressed) == 1
    assert set(bets["candidate"]) == {"candidate__reliability_gate"}
    assert "pair_reliability_ok" in bets
    assert (~bets["pair_reliability_ok"]).sum() == 1
    assert bets.loc[~bets["pair_reliability_ok"], "ev_per_unit"].between(
        0.03, 0.10, inclusive="left"
    ).all()
    assert suppressed.iloc[0]["ev_per_unit"] >= 0.10
    assert not bool(suppressed.iloc[0]["pair_reliability_ok"])
    assert suppressed.iloc[0]["suppression_reason"] == (
        "nominal_ev_ge_10pct_and_pair_reliability_false"
    )


def test_tie_loss_predicate_is_shared_by_pricing_and_settlement() -> None:
    assert ties_lose("separate bet offered")
    assert plumbing._ties_lose_text("SEPARATE")
    base = pd.DataFrame(
        {
            "candidate": ["candidate"],
            "quote_id": ["q1"],
            "mapped_event_key": ["event"],
            "event_start_date": [pd.Timestamp("2021-06-01")],
            "market": ["r1"],
            "book": ["pinnacle"],
            "pair_key": ["1|2"],
            "p1_id": ["1"],
            "p2_id": ["2"],
            "p1_close": [100.0],
            "p2_close": [-110.0],
            "p1_outcome": [0.5],
            "p2_outcome": [0.5],
            "p1_outcome_text": ["tie"],
            "p2_outcome_text": ["tie"],
            "p1_probability": [0.45],
            "push_probability": [0.20],
            "pair_reliability_ok": [True],
        }
    )
    push_bets = plumbing.select_and_settle_bets(
        base.assign(tie_rule="ties push"), "pinnacle", "flat_1u", min_ev=0.0
    )
    separate_bets = plumbing.select_and_settle_bets(
        base.assign(tie_rule="separate bet offered"),
        "pinnacle",
        "flat_1u",
        min_ev=0.0,
    )
    assert len(push_bets) == 1
    assert push_bets.iloc[0]["settlement_status"] == "push"
    assert separate_bets.empty


def test_normalized_quote_integrity_reconstructs_pairs_and_outcome_complements() -> None:
    valid = pd.DataFrame(
        {
            "candidate": ["candidate"],
            "mapped_event_key": ["event"],
            "market": ["r1"],
            "book": ["pinnacle"],
            "pair_key": ["1|2"],
            "p1_id": ["2"],
            "p2_id": ["1"],
            "p1_outcome": [1.0],
            "p2_outcome": [0.0],
            "p1_close": [-110.0],
            "p2_close": [-110.0],
            "skill_difference": [0.3],
            "quote_id": ["q1"],
        }
    )
    audit = plumbing.validate_normalized_matchup_quotes(valid)
    assert audit["complementary_outcome_rows"] == 1
    logical = plumbing._canonical_logical_pairs(valid)
    assert logical.iloc[0]["canonical_skill_difference"] == pytest.approx(-0.3)
    assert logical.iloc[0]["canonical_outcome"] == 0.0
    with pytest.raises(ValueError, match="pair_key"):
        plumbing.validate_normalized_matchup_quotes(valid.assign(pair_key="2|9"))
    with pytest.raises(ValueError, match="complements"):
        plumbing.validate_normalized_matchup_quotes(valid.assign(p2_outcome=1.0))


def test_discovery_prediction_menus_must_be_exactly_identical() -> None:
    frame = pd.DataFrame(
        [
            {
                "candidate": candidate,
                "event_key": event,
                "dg_id_key": player,
                "event_start_date": pd.Timestamp("2021-01-01"),
            }
            for candidate in ("a", "b")
            for event, player in (("event-1", "1"), ("event-2", "2"))
        ]
    )
    audit = complete_eval.validate_identical_prediction_menus(frame)
    assert audit["menus_identical"]
    with pytest.raises(ValueError, match="exact event/player menu"):
        complete_eval.validate_identical_prediction_menus(
            frame.loc[~((frame["candidate"] == "b") & (frame["event_key"] == "event-2"))]
        )


def test_discovery_filter_requires_every_frozen_candidate_and_control() -> None:
    clean_names = [
        f"candidate_{index:03d}"
        for index in range(complete_eval.EXPECTED_CLEAN_CANDIDATE_COUNT)
    ]
    rows = [
        {
            "candidate": name,
            # Simulate a challenger that exists in the full builder output but
            # only in the sealed period, so discovery filtering removes it.
            "event_start_date": pd.Timestamp(
                "2023-01-01" if name == clean_names[-1] else "2022-01-01"
            ),
        }
        for name in clean_names
    ]
    rows.extend(
        {
            "candidate": name,
            "event_start_date": pd.Timestamp("2022-01-01"),
        }
        for name in complete_eval.KNOWN_CONTROL_UNITS
    )
    full_panel = pd.DataFrame(rows)
    discovery = full_panel.loc[
        full_panel["event_start_date"].between(
            complete_eval.WARMUP_START,
            complete_eval.DISCOVERY_END,
        )
    ]
    with pytest.raises(ValueError, match="discovery-filtered.*exact frozen"):
        complete_eval.validate_discovery_candidate_set(discovery, clean_names)

    repaired = pd.concat(
        [
            discovery,
            pd.DataFrame(
                {
                    "candidate": [clean_names[-1]],
                    "event_start_date": [pd.Timestamp("2022-01-01")],
                }
            ),
        ],
        ignore_index=True,
    )
    audit = complete_eval.validate_discovery_candidate_set(repaired, clean_names)
    assert audit["exact_post_filter_name_set"]
    assert audit["observed_candidate_units"] == 95


def test_retrospective_controls_may_be_partial_but_never_expand_menu() -> None:
    reference = complete_eval.DEFAULT_BASELINE
    retrospective = sorted(set(complete_eval.KNOWN_CONTROL_UNITS) - {reference})
    rows = [
        {"candidate": reference, "event_key": event, "dg_id_key": player}
        for event, player in (("event-1", "1"), ("event-2", "2"))
    ]
    rows.extend(
        {
            "candidate": name,
            "event_key": "event-1",
            "dg_id_key": "1",
        }
        for name in retrospective
    )
    frame = pd.DataFrame(rows)
    audit = complete_eval.audit_retrospective_control_coverage(frame)
    assert set(audit["controls"]) == set(retrospective)
    assert all(
        control["coverage_ratio"] == pytest.approx(0.5)
        for control in audit["controls"].values()
    )
    assert all(
        control["missing_rows_imputed"] == 0
        for control in audit["controls"].values()
    )

    expanded = pd.concat(
        [
            frame,
            pd.DataFrame(
                [
                    {
                        "candidate": retrospective[0],
                        "event_key": "event-extra",
                        "dg_id_key": "999",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="outside the reference menu"):
        complete_eval.audit_retrospective_control_coverage(expanded)


def _calibration_warmup_frame(
    *,
    candidates: tuple[str, ...] | None = None,
    event_date: str = "2020-01-02",
) -> pd.DataFrame:
    names = candidates or (complete_eval.DEFAULT_BASELINE, "candidate-a", "candidate-b")
    return pd.DataFrame(
        [
            {
                "candidate": candidate,
                "event_key": "warmup-event",
                "dg_id_key": player,
                "event_start_date": pd.Timestamp(event_date),
            }
            for candidate in names
            for player in ("101", "202")
        ]
    )


def test_calibration_warmup_accepts_first_event_on_january_2() -> None:
    frame = _calibration_warmup_frame()
    required = set(frame["candidate"])
    audit = complete_eval.audit_calibration_warmup_menu(frame, required)
    assert audit["first_event_start"] == "2020-01-02"
    assert audit["exact_required_menu"] is True


def test_calibration_warmup_rejects_empty_interval() -> None:
    frame = _calibration_warmup_frame(event_date="2021-01-02")
    with pytest.raises(ValueError, match="no calibration warm-up event"):
        complete_eval.audit_calibration_warmup_menu(frame, set(frame["candidate"]))


def test_calibration_warmup_rejects_missing_required_candidate() -> None:
    frame = _calibration_warmup_frame(
        candidates=(complete_eval.DEFAULT_BASELINE, "candidate-a")
    )
    with pytest.raises(ValueError, match="missing required candidate"):
        complete_eval.audit_calibration_warmup_menu(
            frame,
            {complete_eval.DEFAULT_BASELINE, "candidate-a", "candidate-b"},
        )


def test_calibration_warmup_rejects_duplicate_or_incomplete_menu() -> None:
    frame = _calibration_warmup_frame()
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate candidate/event/player"):
        complete_eval.audit_calibration_warmup_menu(
            duplicate, set(frame["candidate"])
        )
    incomplete = frame.drop(
        frame.index[
            frame["candidate"].eq("candidate-b")
            & frame["dg_id_key"].eq("202")
        ]
    )
    with pytest.raises(ValueError, match="exact calibration warm-up menu"):
        complete_eval.audit_calibration_warmup_menu(
            incomplete, set(frame["candidate"])
        )


def test_calibration_warmup_rejects_inconsistent_event_dates() -> None:
    frame = _calibration_warmup_frame()
    frame.loc[
        frame["candidate"].eq("candidate-b"), "event_start_date"
    ] = pd.Timestamp("2020-01-03")
    with pytest.raises(ValueError, match="inconsistent dates"):
        complete_eval.audit_calibration_warmup_menu(frame, set(frame["candidate"]))


def _synthetic_envelope_admission_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path, Path]:
    names = ("candidate-a", "candidate-b", "candidate-c")
    monkeypatch.setattr(
        complete_eval, "EXPECTED_REGRESSION_CANDIDATE_COUNT", len(names)
    )
    monkeypatch.setattr(
        complete_eval, "EXPECTED_DISCOVERY_GUARD_DEPENDENT_COUNT", 1
    )
    prediction_rows = []
    fold_rows = []
    fit_rows = []
    for candidate in names:
        clips = 1 if candidate == "candidate-a" else 0
        for index in range(10):
            prediction_rows.append(
                {
                    "candidate": candidate,
                    "event_start_date": pd.Timestamp("2020-01-02"),
                    "refit_cutoff": pd.Timestamp("2019-12-31"),
                    "training_max_event_end": pd.Timestamp("2019-12-30"),
                    "prediction_envelope_clipped": index < clips,
                }
            )
        fold_rows.append(
            {
                "candidate": candidate,
                "refit_cutoff": "2019-12-31",
                "rows": 10,
                "clips": clips,
                "clip_rate": clips / 10,
            }
        )
        fit_rows.append(
            {
                "candidate": candidate,
                "refit_cutoff": "2019-12-31",
                "status": "fit",
                "training_max_event_end": "2019-12-30",
                "test_rows": 10,
                "prediction_envelope_clip_count": clips,
            }
        )
    catalog = pd.DataFrame(
        {
            "name": list(names),
            "prediction_envelope_applied": True,
            "selection_eligible": [False, True, True],
            "sensitivity_only": False,
            "control": False,
        }
    )
    candidate_audit = pd.DataFrame(
        {
            "candidate": list(names),
            "guard_dependent": [True, False, False],
            "selection_eligible": [False, True, True],
        }
    )
    candidate_path = tmp_path / "regression_envelope_admission_by_candidate.csv"
    fold_path = tmp_path / "regression_envelope_admission_by_fold.csv"
    fit_path = tmp_path / "fit_log.csv"
    candidate_audit.to_csv(candidate_path, index=False)
    pd.DataFrame(fold_rows).to_csv(fold_path, index=False)
    pd.DataFrame(fit_rows).to_csv(fit_path, index=False)
    return (
        pd.DataFrame(prediction_rows),
        catalog,
        fit_path,
        candidate_path,
        fold_path,
    )


def test_discovery_only_envelope_admission_matches_frozen_full_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predictions, catalog, fit_path, candidate_path, fold_path = (
        _synthetic_envelope_admission_inputs(tmp_path, monkeypatch)
    )
    audit = complete_eval.audit_discovery_only_envelope_admission(
        predictions,
        catalog,
        fit_log_path=fit_path,
        admission_candidate_path=candidate_path,
        admission_fold_path=fold_path,
    )
    assert audit["guard_dependent_candidates"] == ["candidate-a"]
    assert audit["sealed_prediction_rows_used"] == 0


def test_discovery_only_envelope_rejects_full_only_admission_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predictions, catalog, fit_path, candidate_path, fold_path = (
        _synthetic_envelope_admission_inputs(tmp_path, monkeypatch)
    )
    full = pd.read_csv(candidate_path)
    full["guard_dependent"] = False
    full.to_csv(candidate_path, index=False)
    with pytest.raises(ValueError, match="full-only and discovery-only"):
        complete_eval.audit_discovery_only_envelope_admission(
            predictions,
            catalog,
            fit_log_path=fit_path,
            admission_candidate_path=candidate_path,
            admission_fold_path=fold_path,
        )


def test_discovery_only_envelope_rejects_fold_audit_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predictions, catalog, fit_path, candidate_path, fold_path = (
        _synthetic_envelope_admission_inputs(tmp_path, monkeypatch)
    )
    folds = pd.read_csv(fold_path)
    folds.loc[0, "rows"] = 11
    folds.to_csv(fold_path, index=False)
    with pytest.raises(ValueError, match="prediction/fold envelope admission differs"):
        complete_eval.audit_discovery_only_envelope_admission(
            predictions,
            catalog,
            fit_log_path=fit_path,
            admission_candidate_path=candidate_path,
            admission_fold_path=fold_path,
        )


def _synthetic_frozen_specs() -> list[dict[str, object]]:
    return [
        {
            "name": f"candidate_{index:03d}",
            "family": "ridge",
            "source": "total",
            "weather_arm": "w0",
            "target_arm": "observed",
            "refit_schedule": "quarterly",
            "alpha": 10.0,
            "l1_ratio": 0.0,
            "huber_epsilon": 1.35,
            "dynamic_half_life": 80.0,
            "prior_rounds": 20.0,
            "process_variance_per_day": 0.0,
            "short_game_penalty": 1.0,
            "selection_eligible": True,
            "sensitivity_only": False,
            "control": False,
            "prediction_envelope_applied": True,
            "prediction_envelope_method": (
                "strict_prior_event_equal_weighted_fitted_prediction_median_mad"
            ),
            "prediction_envelope_scale_factor": 1.4826,
            "prediction_envelope_multiplier": 6.0,
            "raw_regression_component_selection_eligible": False,
            "selection_ineligibility_reason": "",
            "description": "synthetic frozen spec",
        }
        for index in range(complete_eval.EXPECTED_CLEAN_CANDIDATE_COUNT)
    ]


def test_frozen_catalog_is_exactly_hashed_92_specs_and_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert complete_eval.EXPECTED_FROZEN_CATALOG_SHA256 == (
        "d355481e8aaa7a23c7e43470ab6d683f2f5017d9ac89c611e04adae8a6597d23"
    )
    specs = _synthetic_frozen_specs()
    frozen = tmp_path / "candidate_catalog_frozen.json"
    frozen.write_text(
        json.dumps(
            {
                "contract": "base_skill_complete_candidate_catalog_v1",
                "frozen_before_complete_candidate_scoring": True,
                "sealed_data_used_for_catalog": False,
                "candidates": specs,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    digest = complete_eval._sha256(frozen)
    monkeypatch.setattr(complete_eval, "EXPECTED_FROZEN_CATALOG_SHA256", digest)
    manifest = {
        "frozen_candidate_catalog_sha256": digest,
        "candidates": specs,
    }
    catalog = pd.DataFrame(specs)
    audit = complete_eval.validate_frozen_catalog_contract(
        catalog, manifest, frozen
    )
    assert audit["clean_candidate_specs"] == 92
    csv_roundtrip = catalog.copy()
    csv_roundtrip["selection_ineligibility_reason"] = pd.Series(
        [np.nan] * len(csv_roundtrip), dtype=object
    )
    roundtrip_audit = complete_eval.validate_frozen_catalog_contract(
        csv_roundtrip, manifest, frozen
    )
    assert roundtrip_audit["exact_spec_match"] is True
    tampered_reason = csv_roundtrip.copy()
    tampered_reason.loc[0, "selection_ineligibility_reason"] = (
        "forged_selection_ineligibility_reason"
    )
    with pytest.raises(ValueError, match="differ"):
        complete_eval.validate_frozen_catalog_contract(
            tampered_reason, manifest, frozen
        )
    missing_required_string = catalog.copy()
    missing_required_string.loc[0, "description"] = np.nan
    with pytest.raises(ValueError, match="required string description is missing"):
        complete_eval.validate_frozen_catalog_contract(
            missing_required_string, manifest, frozen
        )
    tampered = catalog.copy()
    tampered.loc[0, "alpha"] = 100.0
    with pytest.raises(ValueError, match="differ"):
        complete_eval.validate_frozen_catalog_contract(tampered, manifest, frozen)
    missing_envelope = catalog.drop(columns=["prediction_envelope_method"])
    with pytest.raises(ValueError, match="missing required columns"):
        complete_eval.validate_frozen_catalog_contract(
            missing_envelope, manifest, frozen
        )
    tampered_envelope = catalog.copy()
    tampered_envelope.loc[0, "prediction_envelope_multiplier"] = 5.0
    with pytest.raises(ValueError, match="differ"):
        complete_eval.validate_frozen_catalog_contract(
            tampered_envelope, manifest, frozen
        )


def test_builder_manifest_prediction_catalog_and_frozen_hashes_are_mandatory(
    tmp_path: Path,
) -> None:
    files = {}
    for name in (
        "player_event_predictions.parquet",
        "candidate_catalog.csv",
        "candidate_catalog_frozen.json",
    ):
        path = tmp_path / name
        path.write_bytes(name.encode("utf-8"))
        files[name] = {
            "path": str(path),
            "sha256": complete_eval._sha256(path),
        }
    manifest_path = tmp_path / "manifest.json"
    manifest = {"outputs": files}
    for name in files:
        path, digest = complete_eval._required_manifest_output(
            manifest, manifest_path, name
        )
        assert path == Path(files[name]["path"])
        assert digest == files[name]["sha256"]
    del manifest["outputs"]["candidate_catalog.csv"]["sha256"]
    with pytest.raises(ValueError, match="required hashed output"):
        complete_eval._required_manifest_output(
            manifest, manifest_path, "candidate_catalog.csv"
        )


def test_completed_study_authority_is_byte_exact_and_path_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, bindings, _payload = _pin_synthetic_completed_study(
        tmp_path / "authority", monkeypatch
    )
    _manifest, audit, observed = complete_eval.validate_completed_study_authority(
        manifest_path, bindings["predictions"], bindings["candidate_catalog"]
    )
    assert observed == bindings
    assert audit["exact_completed_study"] is True

    alternate_manifest, alternate_bindings, _alternate_payload = (
        _pin_synthetic_completed_study(tmp_path / "alternate", monkeypatch)
    )
    monkeypatch.setattr(complete_eval, "DEFAULT_SOURCE_MANIFEST", manifest_path)
    with pytest.raises(ValueError, match="not the audited completed study authority"):
        complete_eval.validate_completed_study_authority(
            alternate_manifest,
            alternate_bindings["predictions"],
            alternate_bindings["candidate_catalog"],
        )


def test_completed_study_authority_rejects_output_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, bindings, _payload = _pin_synthetic_completed_study(
        tmp_path, monkeypatch
    )
    bindings["fit_log"].write_bytes(b"tampered fit log")
    with pytest.raises(ValueError, match="output bytes differ"):
        complete_eval.validate_completed_study_authority(
            manifest_path, bindings["predictions"], bindings["candidate_catalog"]
        )


def test_completion_manifest_requires_exact_unchanged_builder_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executed = tmp_path / "executed_builder_source.py"
    executed.write_text("# frozen builder\n", encoding="utf-8")
    digest = complete_eval._sha256(executed)
    monkeypatch.setattr(complete_eval, "EXPECTED_BUILDER_SOURCE_SHA256", digest)
    source_record = {
        "path": str(executed),
        "sha256": digest,
        "size_bytes": executed.stat().st_size,
    }
    manifest = {
        "research_only": True,
        "production_files_changed": False,
        "contract": {
            "challengers_use_legacy_skill": False,
            "closing_prices_used_as_features": False,
            "selection_uses_sealed": False,
        },
        "execution_provenance": {
            "all_unchanged": True,
            "source_unchanged": True,
            "source_copy_matches": True,
            "source_copy_unchanged": True,
            "input_unchanged": {"snapshots": True, "round_panel": True},
            "source_before": source_record,
            "source_after": source_record,
            "executed_source_copy_before": source_record,
            "executed_source_copy_after": source_record,
        },
    }
    audit = complete_eval.validate_completion_manifest_provenance(
        manifest, tmp_path / "manifest.json"
    )
    assert audit["exact_builder_source_verified"]

    changed = json.loads(json.dumps(manifest))
    changed["execution_provenance"]["source_after"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="not the frozen builder"):
        complete_eval.validate_completion_manifest_provenance(
            changed, tmp_path / "manifest.json"
        )
    changed = json.loads(json.dumps(manifest))
    changed["execution_provenance"]["input_unchanged"]["round_panel"] = False
    with pytest.raises(ValueError, match="inputs changed"):
        complete_eval.validate_completion_manifest_provenance(
            changed, tmp_path / "manifest.json"
        )


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("--baseline-candidate", "not_the_frozen_control"),
        ("--min-calibration-pairs", "99"),
        ("--finalist-count", "2"),
        ("--coverage-tolerance", "0.01"),
        ("--logloss-tolerance", "0.001"),
        ("--calibration-tolerance", "0.001"),
        ("--money-tolerance", "0.001"),
        ("--pairwise-regularization-c", "1.0"),
        ("--downstream-lambdas", "0,1"),
        ("--opener-weights", "0,1"),
        ("--skip-sensitivities", None),
    ],
)
def test_selection_critical_cli_overrides_fail_closed(
    argument: str, value: str | None
) -> None:
    parser = complete_eval.build_parser()
    defaults = parser.parse_args([])
    contract = complete_eval.validate_frozen_selection_cli(defaults)
    assert contract["frozen_before_scoring"]
    argv = [argument] if value is None else [argument, value]
    with pytest.raises(ValueError, match="selection-critical CLI"):
        complete_eval.validate_frozen_selection_cli(parser.parse_args(argv))


def test_discovery_opener_uses_best_valid_price_not_fixed_book_priority() -> None:
    rows = []
    for book, left_open in (
        ("pinnacle", 0.0),
        ("betcris", 105.0),
        ("betonline", 120.0),
        ("otherbook", 500.0),
    ):
        rows.append(
            {
                "candidate": "candidate",
                "mapped_event_key": "event",
                "event_start_date": pd.Timestamp("2021-06-01"),
                "market": "r1",
                "book": book,
                "pair_key": "1|2",
                "p1_id": "1",
                "p2_id": "2",
                "p1_open": left_open,
                "p2_open": -130.0,
                "p1_close": -110.0,
                "p2_close": -110.0,
                "p1_outcome": 1.0,
                "p2_outcome": 0.0,
                "skill_difference": 0.25,
                "p1_probability": 0.55,
                "push_probability": 0.02,
                "quote_id": f"q-{book}",
            }
        )
    best, audit = complete_eval.best_available_same_book_opening_pairs(
        pd.DataFrame(rows)
    )
    assert best.iloc[0]["book"] == "betonline"
    assert best.iloc[0]["canonical_left_open"] == 120.0
    assert audit["invalid_same_book_open_rows"] == 1
    assert audit["disallowed_book_rows"] == 1
    assert audit["strict_odds_conversion_after_validity_filter"]


def test_market_scoring_module_is_an_explicit_provenance_dependency() -> None:
    expected = (
        complete_eval.REPO / "archive" / "evaluate_complete_base_skill_market.py",
        complete_eval.REPO / "archive" / "backtest_base_skill_market_models.py",
        complete_eval.REPO / "archive" / "historical_odds_cache.py",
        complete_eval.REPO / "archive" / "market_backtest_scoring.py",
        Path(complete_eval.execution_provenance.__file__).resolve(),
    )
    assert complete_eval.DISCOVERY_PROVENANCE_SOURCE_CLOSURE == expected
    assert complete_eval.PROVENANCE_CODE_PATHS == expected
    assert len(set(map(Path.resolve, expected))) == len(expected)
    assert all(path.resolve().is_file() for path in expected)


def test_discovery_postbegin_rejects_source_manifest_path_switch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache, _cache_manifest, cache_inputs, _cache_payload = (
        _pin_synthetic_evaluation_cache(tmp_path, monkeypatch)
    )
    bridge = cache_inputs["legacy_core_bridge"]
    source_manifest_path, study_bindings, _payload = (
        _pin_synthetic_completed_study(tmp_path / "pre", monkeypatch)
    )
    predictions = study_bindings["predictions"]
    catalog = study_bindings["candidate_catalog"]
    (
        _pre_manifest,
        pre_bindings,
        _pre_study_audit,
        pre_study_bindings,
    ) = complete_eval.resolve_discovery_direct_inputs(
        predictions, catalog, source_manifest_path, cache, bridge
    )
    _cache, _audit, pre_cache_authority = (
        complete_eval.validate_evaluation_cache_authority(cache, bridge)
    )
    tracked_paths = list(
        dict.fromkeys(
            [
                *pre_bindings.values(),
                *pre_cache_authority.values(),
                *pre_study_bindings.values(),
            ]
        )
    )
    tracked = SimpleNamespace(
        input_fingerprints=tuple(
            {"path": str(path)} for path in tracked_paths
        )
    )
    alternate_manifest, alternate_bindings, _alternate_payload = (
        _pin_synthetic_completed_study(tmp_path / "alternate", monkeypatch)
    )
    with pytest.raises(ValueError, match="paths changed"):
        complete_eval.revalidate_discovery_inputs_after_begin(
            predictions_path=alternate_bindings["predictions"],
            catalog_path=alternate_bindings["candidate_catalog"],
            source_manifest_path=alternate_manifest,
            cache_dir=cache,
            core_bridge_path=bridge,
            expected_bindings=pre_bindings,
            expected_study_authority_bindings=pre_study_bindings,
            expected_cache_authority_bindings=pre_cache_authority,
            provenance_state=tracked,
        )


def test_discovery_rejects_alternate_self_consistent_evaluation_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache, manifest_path, input_paths, _payload = (
        _pin_synthetic_evaluation_cache(tmp_path, monkeypatch)
    )
    complete_eval.validate_evaluation_cache_authority(
        cache, input_paths["legacy_core_bridge"]
    )
    alternate = tmp_path / "alternate_cache"
    alternate.mkdir()
    for name in (
        "normalized_matchup_quotes.parquet",
        "feature_snapshots.parquet",
        "manifest.json",
    ):
        (alternate / name).write_bytes((cache / name).read_bytes())
    assert complete_eval._sha256(alternate / "manifest.json") == (
        complete_eval._sha256(manifest_path)
    )
    with pytest.raises(ValueError, match="not the audited V4 authority"):
        complete_eval.validate_evaluation_cache_authority(
            alternate, input_paths["legacy_core_bridge"]
        )


def test_discovery_rejects_tampered_evaluation_cache_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache, _manifest_path, input_paths, _payload = (
        _pin_synthetic_evaluation_cache(tmp_path, monkeypatch)
    )
    (cache / "normalized_matchup_quotes.parquet").write_bytes(b"tampered quotes")
    with pytest.raises(ValueError, match="artifact bytes differ"):
        complete_eval.validate_evaluation_cache_authority(
            cache, input_paths["legacy_core_bridge"]
        )


def test_discovery_postbegin_rejects_evaluation_cache_input_path_switch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache, manifest_path, input_paths, payload = (
        _pin_synthetic_evaluation_cache(tmp_path, monkeypatch)
    )
    source_manifest_path, study_bindings, _study_payload = (
        _pin_synthetic_completed_study(tmp_path, monkeypatch)
    )
    predictions = study_bindings["predictions"]
    catalog = study_bindings["candidate_catalog"]
    bridge = input_paths["legacy_core_bridge"]
    _source, direct, _study_audit, study_authority = (
        complete_eval.resolve_discovery_direct_inputs(
            predictions, catalog, source_manifest_path, cache, bridge
        )
    )
    _cache, _audit, cache_authority = (
        complete_eval.validate_evaluation_cache_authority(cache, bridge)
    )
    tracked_paths = list(
        dict.fromkeys(
            [*direct.values(), *cache_authority.values(), *study_authority.values()]
        )
    )
    tracked = SimpleNamespace(
        input_fingerprints=tuple({"path": str(path)} for path in tracked_paths)
    )

    alternate_event_panel = tmp_path / "alternate_player_event_outcomes.parquet"
    alternate_event_panel.write_bytes(input_paths["event_panel"].read_bytes())
    switched_payload = json.loads(json.dumps(payload))
    switched_payload["inputs"]["event_panel"]["path"] = str(
        alternate_event_panel.resolve()
    )
    manifest_path.write_text(
        json.dumps(switched_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _repin_synthetic_cache_manifest(monkeypatch, manifest_path)
    switched_inputs = dict(complete_eval.EXPECTED_EVALUATION_CACHE_INPUTS)
    switched_inputs["event_panel"] = {
        "path": alternate_event_panel,
        "sha256": complete_eval._sha256(alternate_event_panel),
        "bytes": alternate_event_panel.stat().st_size,
    }
    monkeypatch.setattr(
        complete_eval, "EXPECTED_EVALUATION_CACHE_INPUTS", switched_inputs
    )
    with pytest.raises(ValueError, match="authority paths changed"):
        complete_eval.revalidate_discovery_inputs_after_begin(
            predictions_path=predictions,
            catalog_path=catalog,
            source_manifest_path=source_manifest_path,
            cache_dir=cache,
            core_bridge_path=bridge,
            expected_bindings=direct,
            expected_study_authority_bindings=study_authority,
            expected_cache_authority_bindings=cache_authority,
            provenance_state=tracked,
        )


def test_discovery_complete_manifest_rejects_post_finalize_input_mutation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "evaluator.py"
    direct_input = tmp_path / "input.parquet"
    source.write_text("# synthetic evaluator\n", encoding="utf-8")
    direct_input.write_bytes(b"before")
    output = tmp_path / "output"
    state = complete_eval.execution_provenance.begin(
        output_dir=output,
        primary_source=source,
        input_paths=[direct_input],
    )
    complete_eval.execution_provenance.finalize(state)
    direct_input.write_bytes(b"mutated after finalize")
    manifest_path = output / "selection_manifest.json"
    with pytest.raises(RuntimeError, match="input changed"):
        complete_eval.write_discovery_complete_manifest(
            manifest_path, {"complete": True}, state
        )
    assert not manifest_path.exists()
