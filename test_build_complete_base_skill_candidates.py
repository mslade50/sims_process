from __future__ import annotations

import json
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from archive import build_complete_base_skill_candidates as complete


def _snapshots() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_key": ["pga|2021|1|2021-01-10", "pga|2021|2|2021-01-11"],
            "dg_id_key": ["7", "7"],
            "dg_id": pd.Series([7, 7], dtype="Int64"),
            "event_start_date": pd.to_datetime(["2021-01-10", "2021-01-11"]),
            "event_end_date": pd.to_datetime(["2021-01-13", "2021-01-14"]),
            "year": [2021, 2021],
            "event_id": [1, 2],
            "history_rounds": [1, 2],
            "days_since_last_round": [2.0, 1.0],
            "prior_event_withdrawal": [False, False],
            complete.PRIMARY_LABEL: [0.2, -0.1],
            "label_dg_sg_adj_r1_r2_count": [2, 2],
            complete.NEUTRAL_LABEL: [0.15, -0.05],
            "label_dg_sg_neutral_r1_r2_count": [2, 2],
        }
    )


def _rounds() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_name": ["Seven"] * 4,
            "dg_id": pd.Series([7] * 4, dtype="Int64"),
            "event_id": [90, 90, 91, 92],
            "tour": ["euro", "euro", "pga", "kft"],
            "year": [2021] * 4,
            "round_num": [1, 3, 1, 1],
            "round_date": pd.to_datetime(
                ["2021-01-08", "2021-01-09", "2021-01-10", "2021-01-12"]
            ),
            "age": [25.0] * 4,
            "pro_time": [3.0] * 4,
            "sg_total_adj": [1.0, 2.0, 3.0, 4.0],
            "post_cut": [0, 1, 0, 0],
        }
    )


def _valid_enveloped_prediction_row(
    candidate: complete.CandidateSpec,
    event_id: int = 1,
) -> dict[str, object]:
    return {
        "candidate": candidate.name,
        "event_key": f"pga|2021|{event_id}|2021-01-{event_id + 9:02d}",
        "canonical_event_key": f"pga:R2021{event_id:03d}",
        "dg_id_key": "7",
        "dg_id": 7,
        "event_start_date": pd.Timestamp(f"2021-01-{event_id + 9:02d}"),
        "prediction": 0.1,
        "prediction_units": "sg_per_round",
        "prediction_asof": pd.Timestamp(f"2021-01-{event_id + 9:02d}"),
        "chronology_ok": True,
        "split": "discovery",
        "training_max_event_end": pd.Timestamp("2020-12-31"),
        "refit_cutoff": pd.Timestamp("2021-01-01"),
        "raw_regression_component": 0.1,
        "guarded_regression_component": 0.1,
        "raw_regression_component_selection_eligible": False,
        "prediction_envelope_method": complete.PREDICTION_ENVELOPE_METHOD,
        "prediction_envelope_center": 0.0,
        "prediction_envelope_weighted_mad": 1.0,
        "prediction_envelope_robust_scale": 1.4826,
        "prediction_envelope_lower": -8.8956,
        "prediction_envelope_upper": 8.8956,
        "prediction_envelope_clipped": False,
        "prediction_envelope_clip_distance": 0.0,
    }


def test_cross_tour_features_exclude_same_day_and_future_rounds() -> None:
    snapshots = _snapshots()
    features, audit = complete.build_cross_tour_features(snapshots, _rounds())

    assert features.loc[0, "xtour_euro_r12__n_all"] == 1
    assert features.loc[0, "xtour_pga_r12__n_all"] == 0
    assert features.loc[1, "xtour_pga_r12__n_all"] == 1
    assert pd.Timestamp(features.loc[0, "xtour_max_source_round_date"]) < pd.Timestamp(
        snapshots.loc[0, "event_start_date"]
    )
    assert pd.Timestamp(features.loc[1, "xtour_max_source_round_date"]) < pd.Timestamp(
        snapshots.loc[1, "event_start_date"]
    )
    assert audit["strict_prior_violations"] == 0
    assert audit["postcut_nuisance_strict_prior_violations"] == 0


def test_challenger_catalog_never_selects_legacy_or_market_feature() -> None:
    columns = {
        "age_asof": [30.0],
        "legacy_skill_est_asof": [1.0],
        "market_close_price": [-110.0],
        "dg_precut_total_w0__ewm_hl20": [0.2],
        "xtour_pga_r12__mean_all": [0.1],
    }
    frame = pd.DataFrame(columns)
    candidates = complete.candidate_catalog()
    assert complete.assert_no_legacy_challenger_features(frame, candidates) == {}
    for candidate in candidates:
        if candidate.control:
            continue
        selected = complete.candidate_feature_columns(frame, candidate)
        assert "legacy_skill_est_asof" not in selected
        assert "market_close_price" not in selected


def test_prediction_validator_rejects_duplicate_keys() -> None:
    row = {
        "candidate": "complete__test",
        "event_key": "pga|2021|1|2021-01-10",
        "canonical_event_key": "pga:R2021001",
        "dg_id_key": "7",
        "dg_id": 7,
        "event_start_date": pd.Timestamp("2021-01-10"),
        "prediction": 0.1,
        "prediction_units": "sg_per_round",
        "prediction_asof": pd.Timestamp("2021-01-10"),
        "chronology_ok": True,
        "split": "discovery",
        "training_max_event_end": pd.Timestamp("2020-12-31"),
        "refit_cutoff": pd.Timestamp("2021-01-01"),
        "raw_regression_component": 0.1,
        "guarded_regression_component": 0.1,
        "raw_regression_component_selection_eligible": False,
        "prediction_envelope_method": complete.PREDICTION_ENVELOPE_METHOD,
        "prediction_envelope_center": 0.0,
        "prediction_envelope_weighted_mad": 1.0,
        "prediction_envelope_robust_scale": 1.4826,
        "prediction_envelope_lower": -8.8956,
        "prediction_envelope_upper": 8.8956,
        "prediction_envelope_clipped": False,
        "prediction_envelope_clip_distance": 0.0,
    }
    frame = pd.DataFrame([row, row])
    spec = complete._bind_regression_prediction_envelope(
        complete.CandidateSpec(
            name="complete__test",
            family="ridge",
            source="total",
            weather_arm="w0",
            target_arm="observed",
            refit_schedule="annual",
        )
    )
    with pytest.raises(AssertionError, match="duplicate prediction keys"):
        complete.validate_prediction_output(frame, [spec])


def test_prediction_validator_requires_exact_catalog_and_snapshot_keys() -> None:
    spec = complete._bind_regression_prediction_envelope(
        complete.CandidateSpec(
            name="complete__test",
            family="ridge",
            source="total",
            weather_arm="w0",
            target_arm="observed",
            refit_schedule="annual",
        )
    )
    rows = []
    for event_id in (1, 2):
        rows.append(
            {
                "candidate": spec.name,
                "event_key": f"pga|2021|{event_id}|2021-01-{event_id + 9:02d}",
                "canonical_event_key": f"pga:R2021{event_id:03d}",
                "dg_id_key": "7",
                "dg_id": 7,
                "event_start_date": pd.Timestamp(f"2021-01-{event_id + 9:02d}"),
                "prediction": 0.1,
                "prediction_units": "sg_per_round",
                "prediction_asof": pd.Timestamp(f"2021-01-{event_id + 9:02d}"),
                "chronology_ok": True,
                "split": "discovery",
                "training_max_event_end": pd.Timestamp("2020-12-31"),
                "refit_cutoff": pd.Timestamp("2021-01-01"),
                "raw_regression_component": 0.1,
                "guarded_regression_component": 0.1,
                "raw_regression_component_selection_eligible": False,
                "prediction_envelope_method": complete.PREDICTION_ENVELOPE_METHOD,
                "prediction_envelope_center": 0.0,
                "prediction_envelope_weighted_mad": 1.0,
                "prediction_envelope_robust_scale": 1.4826,
                "prediction_envelope_lower": -8.8956,
                "prediction_envelope_upper": 8.8956,
                "prediction_envelope_clipped": False,
                "prediction_envelope_clip_distance": 0.0,
            }
        )
    frame = pd.DataFrame(rows)
    expected = frame[["event_key", "dg_id_key"]].copy()
    with pytest.raises(AssertionError, match="key coverage"):
        complete.validate_prediction_output(frame.iloc[:1], [spec], expected)
    second = complete._bind_regression_prediction_envelope(
        complete.CandidateSpec(
            name="complete__missing",
            family="ridge",
            source="total",
            weather_arm="w0",
            target_arm="observed",
            refit_schedule="annual",
        )
    )
    with pytest.raises(AssertionError, match="frozen catalog"):
        complete.validate_prediction_output(frame, [spec, second], expected)
    audit = complete.validate_prediction_output(frame, [spec], expected)
    assert audit["expected_challenger_count"] == 1
    assert audit["expected_keys_per_challenger"] == 2
    assert audit["total_missing_challenger_keys"] == 0
    assert audit["total_extra_challenger_keys"] == 0


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("guarded_regression_component", 0.2, "not clip"),
        (
            "raw_regression_component_selection_eligible",
            True,
            "selection eligible",
        ),
        ("prediction_envelope_clip_distance", 0.1, "distances are inconsistent"),
    ],
)
def test_prediction_validator_rejects_invalid_envelope_rows(
    column: str, value: object, message: str
) -> None:
    spec = complete._bind_regression_prediction_envelope(
        complete.CandidateSpec(
            name="complete__envelope_validator_test",
            family="ridge",
            source="total",
            weather_arm="w0",
            target_arm="observed",
            refit_schedule="quarterly",
        )
    )
    row = _valid_enveloped_prediction_row(spec)
    row[column] = value
    with pytest.raises(AssertionError, match=message):
        complete.validate_prediction_output(pd.DataFrame([row]), [spec])


def test_prediction_validator_rejects_fold_varying_envelope() -> None:
    spec = complete._bind_regression_prediction_envelope(
        complete.CandidateSpec(
            name="complete__varying_envelope_test",
            family="ridge",
            source="total",
            weather_arm="w0",
            target_arm="observed",
            refit_schedule="quarterly",
        )
    )
    first = _valid_enveloped_prediction_row(spec, 1)
    second = _valid_enveloped_prediction_row(spec, 2)
    second["prediction_envelope_center"] = 1.0
    second["prediction_envelope_lower"] = -7.8956
    second["prediction_envelope_upper"] = 9.8956
    with pytest.raises(AssertionError, match="vary inside"):
        complete.validate_prediction_output(pd.DataFrame([first, second]), [spec])


def test_prediction_validator_rejects_img_blend_before_guard() -> None:
    spec = complete._bind_regression_prediction_envelope(
        complete.CandidateSpec(
            name="complete__img_validator_test",
            family="ridge",
            source="img_primary",
            weather_arm="w0",
            target_arm="observed",
            refit_schedule="quarterly",
        )
    )
    row = _valid_enveloped_prediction_row(spec)
    row["img_primary_weight"] = 0.5
    row["img_primary_fallback_component"] = 2.0
    row["img_primary_fallback_used"] = True
    # Correct guard-then-blend final is 1.05; leave 0.1 to force the audit failure.
    with pytest.raises(AssertionError, match="guard-then-fallback blend"):
        complete.validate_prediction_output(pd.DataFrame([row]), [spec])


def test_freeze_refuses_sealed_selection_and_emits_exact_contract(tmp_path) -> None:
    prediction_path = tmp_path / "predictions.parquet"
    candidate = "complete__ridge__total__w0__quarterly__a10"
    pd.DataFrame(
        {
            "event_key": ["pga|2023|1|2023-01-05"] * 2,
            "canonical_event_key": ["pga:R2023001"] * 2,
            "dg_id": [7, 7],
            "dg_id_key": ["7", "7"],
            "event_start": pd.to_datetime(["2023-01-05"] * 2),
            "event_start_date": pd.to_datetime(["2023-01-05"] * 2),
            "candidate": [candidate, "current_architecture_pt_v2_approx_control"],
            "finalist_core": [0.25, 0.10],
            "prediction": [0.25, 0.10],
            "split": ["sealed", "sealed"],
            "prediction_asof": pd.to_datetime(["2023-01-05"] * 2),
            "chronology_ok": [True, True],
        }
    ).to_parquet(prediction_path, index=False)

    bad_manifest = tmp_path / "bad.json"
    bad_manifest.write_text(
        json.dumps(
            {
                "finalist_name": candidate,
                "selection_window_end": "2023-01-01",
                "sealed_data_used_for_selection": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exceeds"):
        complete.freeze_finalist(
            prediction_path, candidate, bad_manifest, tmp_path / "bad_output"
        )

    good_manifest = tmp_path / "good.json"
    good_manifest.write_text(
        json.dumps(
            {
                "finalist_name": candidate,
                "selection_window_end": "2022-12-31",
                "sealed_data_used_for_selection": False,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "frozen"
    complete.freeze_finalist(prediction_path, candidate, good_manifest, output)
    payload = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    rows = pd.read_parquet(output / "player_event_predictions.parquet")
    assert payload["contract"] == "base_skill_frozen_finalist_v1"
    assert payload["frozen"] is True
    assert payload["selection_window_end"] == "2022-12-31"
    assert payload["sealed_data_used_for_selection"] is False
    assert payload["uses_closing_odds_as_feature"] is False
    assert rows.columns.tolist() == [
        "event_key",
        "dg_id",
        "event_start",
        "candidate",
        "finalist_core",
        "control_core",
        "split",
        "prediction_asof",
        "chronology_ok",
    ]
    assert rows.loc[0, "event_key"] == "pga:R2023001"
    assert rows.loc[0, "control_core"] == pytest.approx(0.10)


def test_huber_irls_is_deterministic_and_convergent() -> None:
    x = np.arange(30, dtype=float).reshape(-1, 1)
    y = 0.5 + 0.2 * x[:, 0]
    y[-1] += 20.0
    first = complete.HuberIRLS(alpha=1.0).fit(x, y)
    second = complete.HuberIRLS(alpha=1.0).fit(x, y)
    assert first.converged_ and second.converged_
    np.testing.assert_allclose(first.predict(x), second.predict(x), atol=1e-12)


def test_elastic_cd_is_stable_and_uses_frozen_solver_configuration() -> None:
    rng = np.random.default_rng(20260823)
    x = rng.normal(size=(200, 5))
    y = 0.5 * x[:, 0] - 0.25 * x[:, 1] + rng.normal(scale=0.1, size=200)
    weights = np.linspace(0.5, 1.5, len(y))
    spec = complete.CandidateSpec(
        name="complete__elastic_cd__test",
        family="elastic_cd",
        source="total",
        weather_arm="w0",
        target_arm="observed",
        refit_schedule="quarterly",
        alpha=0.001,
        l1_ratio=0.25,
    )
    model = complete._fit_regression(spec, x, y, weights)
    parameters = model.get_params()
    assert parameters["precompute"] is True
    assert parameters["selection"] == "cyclic"
    assert parameters["max_iter"] == 50_000
    assert parameters["tol"] == pytest.approx(1e-3)
    assert model.n_iter_ < parameters["max_iter"]
    assert model.converged_ is True
    assert model.convergence_warning_count_ == 0
    assert np.isfinite(model.training_wmse_)
    assert model.training_wmse_ < complete.ELASTIC_CD_MAX_TRAIN_WMSE
    assert model.max_abs_coefficient_ < complete.ELASTIC_CD_MAX_ABS_COEFFICIENT
    assert np.isfinite(model.dual_gap_max_)
    assert model.numerical_finite_ is True


@pytest.mark.parametrize(("family", "alpha"), [("ridge", 1.0), ("robust_irls", 10.0)])
def test_all_regression_families_expose_numerical_fit_telemetry(
    family: str, alpha: float
) -> None:
    rng = np.random.default_rng(17)
    x = rng.normal(size=(100, 3))
    y = 0.4 * x[:, 0] - 0.2 * x[:, 1] + rng.normal(scale=0.2, size=100)
    spec = complete.CandidateSpec(
        name=f"complete__{family}__telemetry_test",
        family=family,
        source="total",
        weather_arm="w0",
        target_arm="observed",
        refit_schedule="quarterly",
        alpha=alpha,
    )
    model = complete._fit_regression(spec, x, y, np.ones(len(y)))
    assert model.numerical_finite_ is True
    assert np.isfinite(model.training_wmse_)
    assert np.isfinite(model.max_abs_coefficient_)
    assert np.isnan(model.dual_gap_max_)
    assert model.convergence_warning_count_ == 0


def test_elastic_cd_fails_closed_on_convergence_warning(monkeypatch) -> None:
    def warning_fit(model, x, y, sample_weight=None):
        warnings.warn("forced numerical warning", complete.ConvergenceWarning)
        return model

    monkeypatch.setattr(complete.ElasticNet, "fit", warning_fit)
    spec = complete.CandidateSpec(
        name="complete__elastic_cd__warning_test",
        family="elastic_cd",
        source="total",
        weather_arm="w0",
        target_arm="observed",
        refit_schedule="quarterly",
        alpha=0.001,
        l1_ratio=0.25,
    )
    with pytest.raises(RuntimeError, match="convergence warning"):
        complete._fit_regression(
            spec,
            np.ones((10, 2), dtype=float),
            np.ones(10, dtype=float),
            np.ones(10, dtype=float),
        )


def test_regression_prediction_envelope_uses_weighted_median_and_mad() -> None:
    spec = complete._bind_regression_prediction_envelope(
        complete.CandidateSpec(
            name="complete__envelope_test",
            family="ridge",
            source="total",
            weather_arm="w0",
            target_arm="observed",
            refit_schedule="quarterly",
        )
    )
    training_prediction = np.array([-1.0, 0.0, 1.0, 100.0])
    event_equal_weights = np.array([1.0, 1.0, 1.0, 0.1])
    envelope = complete._fit_prediction_envelope(
        training_prediction, event_equal_weights, spec
    )
    assert envelope["center"] == pytest.approx(0.0)
    assert envelope["weighted_mad"] == pytest.approx(1.0)
    assert envelope["robust_scale"] == pytest.approx(1.4826)
    assert envelope["lower"] == pytest.approx(-8.8956)
    assert envelope["upper"] == pytest.approx(8.8956)

    guarded, clipped, distance = complete._apply_prediction_envelope(
        np.array([-20.0, 2.0, 20.0]), envelope
    )
    np.testing.assert_allclose(guarded, [-8.8956, 2.0, 8.8956])
    assert clipped.tolist() == [True, False, True]
    np.testing.assert_allclose(distance, [11.1044, 0.0, 11.1044])


def test_regression_prediction_envelope_fails_closed_on_zero_scale() -> None:
    spec = complete._bind_regression_prediction_envelope(
        complete.CandidateSpec(
            name="complete__zero_envelope_test",
            family="ridge",
            source="total",
            weather_arm="w0",
            target_arm="observed",
            refit_schedule="quarterly",
        )
    )
    with pytest.raises(RuntimeError, match="zero/nonfinite scale"):
        complete._fit_prediction_envelope(
            np.ones(5, dtype=float), np.ones(5, dtype=float), spec
        )


def test_img_primary_guards_component_before_fallback_blend(monkeypatch) -> None:
    spec = complete._bind_regression_prediction_envelope(
        complete.CandidateSpec(
            name="complete__img_envelope_test",
            family="ridge",
            source="img_primary",
            weather_arm="w0",
            target_arm="observed",
            refit_schedule="quarterly",
        )
    )
    test = pd.DataFrame(
        {
            "clean_img_warmup_history_rounds": [10.0],
            "clean_img_warmup_history_coverage": [0.25],
            "clean_img_warmup_complete_rounds": [4.0],
        }
    )
    monkeypatch.setattr(
        complete,
        "_cross_tour_hierarchical_fallback",
        lambda *args, **kwargs: np.array([2.0]),
    )
    final, guarded, clipped, distance, envelope, weight, fallback = (
        complete._guard_regression_prediction(
            raw_regression_component=np.array([100.0]),
            training_prediction=np.array([-1.0, 0.0, 1.0]),
            event_equal_weights=np.ones(3),
            candidate=spec,
            test=test,
            snapshots=pd.DataFrame(),
            cutoff=pd.Timestamp("2021-01-01"),
        )
    )
    assert guarded[0] == pytest.approx(envelope["upper"])
    assert clipped.tolist() == [True]
    assert distance[0] > 90.0
    assert weight[0] == pytest.approx(0.5)
    assert fallback[0] == pytest.approx(2.0)
    assert final[0] == pytest.approx(0.5 * guarded[0] + 0.5 * 2.0)
    assert final[0] != pytest.approx(0.5 * 100.0 + 0.5 * 2.0)


def test_prediction_envelope_is_bound_to_every_regression_not_dynamics() -> None:
    catalog = complete.candidate_catalog()
    regressions = [
        candidate
        for candidate in catalog
        if candidate.family in complete.REGRESSION_FAMILIES
    ]
    dynamics = [
        candidate for candidate in catalog if candidate.family.startswith("dynamic_")
    ]
    assert regressions
    assert dynamics
    assert all(candidate.prediction_envelope_applied for candidate in regressions)
    assert all(
        candidate.prediction_envelope_method
        == complete.PREDICTION_ENVELOPE_METHOD
        for candidate in regressions
    )
    assert all(
        candidate.prediction_envelope_scale_factor == pytest.approx(1.4826)
        and candidate.prediction_envelope_multiplier == pytest.approx(6.0)
        and candidate.raw_regression_component_selection_eligible is False
        for candidate in regressions
    )
    assert all(not candidate.prediction_envelope_applied for candidate in dynamics)
    assert "market" not in complete.PREDICTION_ENVELOPE_METHOD
    assert "outcome" not in complete.PREDICTION_ENVELOPE_METHOD


def test_walk_forward_fit_log_persists_numerical_and_envelope_audit() -> None:
    training_rows = 12
    rows = training_rows + 2
    starts = pd.to_datetime(
        ["2019-06-01"] * training_rows + ["2020-01-09", "2020-01-09"]
    )
    frame = pd.DataFrame(
        {
            "event_key": [
                f"pga|2019|{index // 4 + 1}|2019-06-01"
                for index in range(training_rows)
            ]
            + ["pga|2020|1|2020-01-09"] * 2,
            "canonical_event_key": [
                f"pga:R2019{index // 4 + 1:03d}" for index in range(training_rows)
            ]
            + ["pga:R2020001"] * 2,
            "dg_id_key": [str(index + 1) for index in range(rows)],
            "dg_id": pd.Series(range(1, rows + 1), dtype="Int64"),
            "event_start_date": starts,
            "event_end_date": starts + pd.Timedelta(days=3),
            "split": ["warmup"] * training_rows + ["calibration_warmup"] * 2,
            complete.PRIMARY_LABEL: np.linspace(-1.0, 1.0, rows),
            "label_dg_sg_adj_r1_r2_count": [2] * rows,
            "dg_precut_total_w0__ewm_hl20": np.linspace(-0.8, 0.8, rows),
            "history_rounds": [30] * rows,
            "xtour_combined_r12__n_all": [30] * rows,
            "xtour_days_since_last_round": [7.0] * rows,
            "prior_event_withdrawal": [False] * rows,
            "cross_tour_fallback_reason": ["pga_history_only"] * rows,
        }
    )
    spec = complete._bind_regression_prediction_envelope(
        complete.CandidateSpec(
            name="complete__ridge__fit_log_test",
            family="ridge",
            source="total",
            weather_arm="w0",
            target_arm="observed",
            refit_schedule="annual",
            alpha=1.0,
        )
    )
    predictions, fit_log, _ = complete.walk_forward_regression_predictions(
        frame, spec, minimum_training_rows=5
    )
    assert len(predictions) == 2
    assert {
        "raw_regression_component",
        "guarded_regression_component",
        "prediction_envelope_clipped",
        "prediction_envelope_clip_distance",
    }.issubset(predictions.columns)
    assert len(fit_log) == 1
    log = fit_log.iloc[0]
    assert bool(log["numerical_finite"]) is True
    assert np.isnan(log["dual_gap_max"])
    assert np.isfinite(log["training_wmse"])
    assert np.isfinite(log["max_abs_coefficient"])
    assert log["convergence_warning_count"] == 0
    assert log["prediction_envelope_method"] == complete.PREDICTION_ENVELOPE_METHOD


def test_freeze_accepts_discovery_only_diagnostic_without_promotion(tmp_path) -> None:
    candidate = "complete__ridge__total__w0__quarterly__a10"
    prediction_path = tmp_path / "predictions.parquet"
    pd.DataFrame(
        {
            "event_key": ["internal", "internal"],
            "canonical_event_key": ["pga:R2023001", "pga:R2023001"],
            "dg_id": [7, 7],
            "dg_id_key": ["7", "7"],
            "event_start": pd.to_datetime(["2023-01-05"] * 2),
            "event_start_date": pd.to_datetime(["2023-01-05"] * 2),
            "candidate": [candidate, "current_architecture_pt_v2_approx_control"],
            "finalist_core": [0.25, 0.1],
            "prediction": [0.25, 0.1],
            "split": ["sealed", "sealed"],
            "prediction_asof": pd.to_datetime(["2023-01-05"] * 2),
            "chronology_ok": [True, True],
        }
    ).to_parquet(prediction_path, index=False)
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "selection_window_end": "2022-12-31",
                "sealed_data_used_for_selection": False,
                "promotion_gate_passed": False,
                "selection_role": "diagnostic_challenger",
                "diagnostic_challenger": candidate,
                "finalists": [],
                "failed_reasons": ["ROI guardrail"],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "diagnostic"
    complete.freeze_finalist(prediction_path, candidate, selection, output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["finalist_name"] is None
    assert manifest["diagnostic_challenger"] == candidate
    assert manifest["selection_role"] == "diagnostic_challenger"
    assert manifest["promotion_gate_passed"] is False


def test_robust_weather_columns_are_preserved_without_outcome_flags() -> None:
    frame = pd.DataFrame(
        columns=[
            "dg_precut_total_w1_robust_covariate__ewm_hl20",
            "dg_precut_cat_ott_w1_robust_covariate__ewm_hl20",
            "weather_excluded",
        ]
    )
    candidates = complete.add_available_robust_weather_candidates(
        complete.candidate_catalog(), frame
    )
    robust = [candidate for candidate in candidates if "w1_robust_covariate" in candidate.name]
    assert robust
    selected = complete.candidate_feature_columns(frame, robust[0])
    assert "dg_precut_total_w1_robust_covariate__ewm_hl20" in selected
    assert "weather_excluded" not in selected


def test_w1_feature_policy_excludes_raw_pga_but_allows_raw_non_pga() -> None:
    frame = pd.DataFrame(
        {
            "xtour_pga_r12__ewm_hl80": [9.0],
            "xtour_pga_r12__ess_hl80": [20.0],
            "xtour_euro_r12__ewm_hl80": [0.5],
            "xtour_euro_r12__ess_hl80": [10.0],
            "dg_precut_total_w1__ewm_hl80": [0.2],
            "dg_precut_total_w1__ess_hl80": [15.0],
        }
    )
    spec = complete.CandidateSpec(
        name="complete__weather_policy_test",
        family="ridge",
        source="total",
        weather_arm="w1",
        target_arm="feature_only",
        refit_schedule="quarterly",
    )
    selected = complete.candidate_feature_columns(frame, spec)
    assert "xtour_pga_r12__ewm_hl80" not in selected
    assert "xtour_pga_r12__ess_hl80" not in selected
    assert "xtour_euro_r12__ewm_hl80" in selected
    assert "dg_precut_total_w1__ewm_hl80" in selected
    audit = complete.assert_weather_feature_policy(frame, [spec])
    assert audit["raw_pga_performance_feature_violations"] == 0
    assert audit["allowed_raw_non_pga_performance_feature_counts"][spec.name] == 2


def test_img_primary_w1_fallback_uses_cleaned_pga_history() -> None:
    snapshots = pd.DataFrame(
        {
            "event_start_date": pd.to_datetime(["2020-12-17"]),
            "event_end_date": pd.to_datetime(["2020-12-20"]),
            "dg_id_key": ["7"],
            complete.PRIMARY_LABEL: [0.0],
            "label_dg_sg_adj_r1_r2_count": [2],
            complete.NEUTRAL_LABEL: [0.0],
            "label_dg_sg_neutral_r1_r2_count": [2],
            "dg_precut_total_w1__mean_all": [0.0],
            "dg_precut_total_w1__n_all": [2],
            **{
                f"xtour_{tour}_r12__mean_all": [np.nan]
                for tour in complete.TOURS
            },
            **{
                f"xtour_{tour}_r12__n_all": [0]
                for tour in complete.TOURS
            },
        }
    )
    test = pd.DataFrame(
        {
            "dg_precut_total_w1__ewm_hl80": [1.0],
            "dg_precut_total_w1__ess_hl80": [20.0],
            "xtour_pga_r12__ewm_hl80": [100.0],
            "xtour_pga_r12__ess_hl80": [20.0],
        }
    )
    fallback = complete._cross_tour_hierarchical_fallback(
        test,
        snapshots,
        pd.Timestamp("2021-01-01"),
        weather_arm="w1",
        prior_rounds=20.0,
    )
    assert fallback[0] == pytest.approx(0.5)


def test_img_weather_join_uses_exact_physical_round_identity(tmp_path) -> None:
    img_path = tmp_path / "img.parquet"
    img = pd.DataFrame(
        {
            "event_key": ["pga:R2020016"],
            "year": pd.Series([2020], dtype="Int64"),
            "event_id": pd.Series([16], dtype="Int64"),
            "dg_id": pd.Series([6169], dtype="Int64"),
            "round_num": pd.Series([1], dtype="Int64"),
            "round_date": pd.to_datetime(["2020-01-02"]),
            "course_num": pd.Series([656], dtype="Int64"),
            "img_clean_warmup_signal_available": [True],
            "img_clean_warmup_sg_total_field_adjusted_per18": [0.2],
            "img_clean_warmup_sg_ott_field_adjusted_per18": [0.1],
            "img_clean_warmup_sg_app_field_adjusted_per18": [0.1],
            "img_clean_warmup_sg_arg_field_adjusted_per18": [0.0],
            "img_clean_warmup_sg_putt_field_adjusted_per18": [0.0],
        }
    )
    img.to_parquet(img_path, index=False)
    weather = pd.DataFrame(
        {
            "event_key": ["pga|2020|16|2020-01-02"],
            "year": pd.Series([2020], dtype="Int64"),
            "event_id": pd.Series([16], dtype="Int64"),
            "dg_id": pd.Series([6169], dtype="Int64"),
            "round_num": pd.Series([1], dtype="Int64"),
            "round_date": pd.to_datetime(["2020-01-02"]),
            "course_num": pd.Series([656], dtype="Int64"),
            "weather_delta_wind": [2.0],
            "weather_delta_dew": [1.0],
        }
    )
    merged, audit = complete._load_img_with_weather(
        img_path, tmp_path / "unused.parquet", weather
    )
    assert audit["weather_join_matched_rows"] == 1
    assert audit["weather_join_unmatched_rows"] == 0
    assert merged.loc[0, "weather_delta_wind"] == pytest.approx(2.0)
    coefficient = {
        "total_beta_wind": 0.1,
        "total_beta_dew": 0.02,
        "ott_beta_wind": 0.1,
        "ott_beta_dew": 0.02,
        "app_beta_wind": 0.1,
        "app_beta_dew": 0.02,
        "arg_beta_wind": 0.1,
        "arg_beta_dew": 0.02,
        "putt_beta_wind": 0.1,
        "putt_beta_dew": 0.02,
    }
    w0, _ = complete._img_signal_arrays(merged, "warmup", "w0", coefficient)
    w1, _ = complete._img_signal_arrays(merged, "warmup", "w1", coefficient)
    assert w1["total"][0] != pytest.approx(w0["total"][0])


def test_img_coverage_is_bounded_by_all_round_denominator() -> None:
    valid = pd.DataFrame(
        {
            "clean_img_warmup_history_rounds": [2],
            "clean_img_warmup_history_coverage_denominator": [4],
            "clean_img_warmup_history_coverage": [0.5],
            "clean_img_unambiguous_history_rounds": [1],
            "clean_img_unambiguous_history_coverage_denominator": [4],
            "clean_img_unambiguous_history_coverage": [0.25],
        }
    )
    audit = complete.validate_img_history_coverage(valid)
    assert audit["history_coverage_violation_count"] == 0
    invalid = valid.copy()
    invalid.loc[0, "clean_img_warmup_history_rounds"] = 5
    invalid.loc[0, "clean_img_warmup_history_coverage"] = 1.25
    with pytest.raises(AssertionError, match="coverage audit"):
        complete.validate_img_history_coverage(invalid)


def test_recency_features_include_all_prespecified_windows() -> None:
    values = np.array([1.0, 2.0])
    dates = np.array(["2021-03-01", "2021-04-20"], dtype="datetime64[ns]")
    features = complete._aggregate_values(
        values, dates, pd.Timestamp("2021-05-01"), "signal"
    )
    for days in (60, 180, 365, 730):
        assert f"signal__mean_d{days}" in features
        assert f"signal__n_d{days}" in features
        assert f"mean_d{days}" in complete.MODEL_SIGNAL_SUFFIXES
        assert f"n_d{days}" in complete.MODEL_SIGNAL_SUFFIXES


def test_broadie_group_penalty_targets_only_arg_and_putt_profiles() -> None:
    spec = complete.CandidateSpec(
        name="complete__group_test",
        family="ridge",
        source="total_cat",
        weather_arm="w0",
        target_arm="observed",
        refit_schedule="quarterly",
        short_game_penalty=4.0,
    )
    columns = [
        "dg_precut_total_w0__ewm_hl20",
        "dg_precut_cat_ott_w0__ewm_hl20",
        "dg_precut_cat_app_w0__ewm_hl20",
        "dg_precut_cat_arg_w0__ewm_hl20",
        "clean_img_warmup_w0_cat_putt__ewm_hl20",
    ]
    np.testing.assert_allclose(
        complete._feature_penalty_scales(columns, spec),
        [1.0, 1.0, 1.0, 0.5, 0.5],
    )
    # Ridge on X*s penalizes the equivalent original standardized coefficient
    # by alpha/s^2: s=.5 is exactly a 4x penalty, not 16x.
    scales = complete._feature_penalty_scales(columns, spec)
    assert 1.0 / scales[3] ** 2 == pytest.approx(4.0)


def test_execution_provenance_fails_closed_when_input_changes(tmp_path) -> None:
    source = tmp_path / "builder.py"
    input_file = tmp_path / "input.parquet"
    source.write_text("print('frozen')\n", encoding="utf-8")
    input_file.write_bytes(b"immutable-input")
    preserved = complete.preserve_execution_source(source, tmp_path)
    start = {
        "process_id": 123,
        "process_start_utc": "2026-08-23T00:00:00+00:00",
        "working_directory": str(tmp_path),
        "argv": ["builder.py"],
        "parsed_args": {},
        "runtime": complete.runtime_fingerprint(),
        **preserved,
        "inputs_before": {"panel": complete.file_fingerprint(input_file)},
    }
    unchanged = complete.verify_execution_unchanged(start)
    assert unchanged["all_unchanged"] is True
    assert unchanged["source_unchanged"] is True
    assert unchanged["source_copy_matches"] is True
    assert (tmp_path / "executed_builder_source.py").read_bytes() == source.read_bytes()
    input_file.write_bytes(b"changed-input")
    with pytest.raises(RuntimeError, match="panel"):
        complete.verify_execution_unchanged(start)


def test_clean_frozen_catalog_excludes_control_rows() -> None:
    clean = complete.candidate_catalog()
    assert len(clean) == 92
    assert sum(candidate.family == "elastic_cd" for candidate in clean) == 4
    assert not any(candidate.family == "elastic_sgd" for candidate in clean)
    assert not any(candidate.control for candidate in clean)
    assert {candidate.name for candidate in complete.CONTROL_SPECS} == {
        "current_architecture_pt_v2_approx_control",
        "legacy_skill_est_pt_current_coefficients_control",
        "legacy_pred_step2_exact_retrospective_control",
    }
    assert all(candidate.control for candidate in complete.CONTROL_SPECS)


def test_frozen_numerical_gate_marks_exact_34_diagnostic_only() -> None:
    catalog = complete.candidate_catalog()
    by_name = {candidate.name: candidate for candidate in catalog}
    excluded = {
        name
        for name, candidate in by_name.items()
        if candidate.selection_ineligibility_reason.startswith(
            complete.REGRESSION_ENVELOPE_GATE_INELIGIBILITY_REASON
        )
    }
    assert excluded == set(complete.NUMERICAL_GATE_INELIGIBLE_CANDIDATES)
    assert len(excluded) == 34
    assert all(
        by_name[name].family in complete.REGRESSION_FAMILIES
        and by_name[name].selection_eligible is False
        and by_name[name].sensitivity_only is True
        and "overall_clip_rate=" in by_name[name].selection_ineligibility_reason
        and "worst_fold=" in by_name[name].selection_ineligibility_reason
        and "worst_fold_clip_rate=" in by_name[name].selection_ineligibility_reason
        for name in excluded
    )
    regressions = [
        candidate
        for candidate in catalog
        if candidate.family in complete.REGRESSION_FAMILIES
    ]
    assert len(regressions) == 88
    assert sum(candidate.selection_eligible for candidate in regressions) == 53
    assert sum(
        candidate.name not in excluded for candidate in regressions
    ) == 54

    evidence, audit = complete._load_regression_envelope_gate_evidence()
    validated = complete._validate_regression_envelope_gate_evidence(
        catalog, evidence, audit
    )
    computed_failures = set(
        validated.index[
            validated["overall_clip_rate"].gt(
                complete.REGRESSION_ENVELOPE_GATE_OVERALL_CLIP_RATE_MAX
            )
            | validated["worst_fold_clip_rate"].gt(
                complete.REGRESSION_ENVELOPE_GATE_ANY_FOLD_CLIP_RATE_MAX
            )
        ]
    )
    assert computed_failures == excluded
    payload = complete.frozen_catalog_payload(catalog)
    gate = payload["post_preflight_regression_envelope_admission_gate"]
    assert gate["overall_clip_rate_max"] == 0.01
    assert gate["any_refit_fold_clip_rate_max"] == 0.05
    assert gate["or_gate_failure_count"] == 34
    assert gate["evidence"]["audit_sha256"] == (
        complete.REGRESSION_ENVELOPE_GATE_AUDIT_SHA256
    )
    assert gate["evidence"]["candidate_sha256"] == (
        complete.REGRESSION_ENVELOPE_GATE_CANDIDATE_SHA256
    )
    assert gate["evidence"]["fold_sha256"] == (
        complete.REGRESSION_ENVELOPE_GATE_FOLD_SHA256
    )


def test_current_run_envelope_gate_recomputes_strict_rates_and_eligibility() -> None:
    passing = complete._bind_regression_prediction_envelope(
        complete.CandidateSpec(
            name="complete__synthetic_boundary_pass",
            family="ridge",
            source="total",
            weather_arm="w0",
            target_arm="observed",
            refit_schedule="quarterly",
        )
    )
    failing_name = sorted(complete.NUMERICAL_GATE_INELIGIBLE_CANDIDATES)[0]
    failing = complete._bind_regression_prediction_envelope(
        complete.CandidateSpec(
            name=failing_name,
            family="ridge",
            source="img_primary",
            weather_arm="w0",
            target_arm="observed",
            refit_schedule="quarterly",
            selection_eligible=False,
            sensitivity_only=True,
            selection_ineligibility_reason=(
                complete.REGRESSION_ENVELOPE_GATE_INELIGIBILITY_REASON
            ),
        )
    )
    prediction_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []

    def add_fold(
        candidate: complete.CandidateSpec,
        cutoff: str,
        rows: int,
        clips: int,
    ) -> None:
        for index in range(rows):
            clipped = index < clips
            prediction_rows.append(
                {
                    "candidate": candidate.name,
                    "refit_cutoff": pd.Timestamp(cutoff),
                    "prediction": 1.0 if clipped else 0.1,
                    "raw_regression_component": 2.0 if clipped else 0.1,
                    "guarded_regression_component": 1.0 if clipped else 0.1,
                    "prediction_envelope_clipped": clipped,
                    "prediction_envelope_clip_distance": 1.0 if clipped else 0.0,
                }
            )
        fit_rows.append(
            {
                "candidate": candidate.name,
                "refit_cutoff": pd.Timestamp(cutoff),
                "test_rows": rows,
                "prediction_envelope_clip_count": clips,
                "prediction_envelope_clip_rate": clips / rows,
            }
        )

    # Equality to both frozen thresholds passes because the protocol uses strict >.
    add_fold(passing, "2021-01-01", 100, 5)
    for quarter in ("2021-04-01", "2021-07-01", "2021-10-01", "2022-01-01"):
        add_fold(passing, quarter, 100, 0)
    add_fold(failing, "2021-01-01", 100, 6)
    predictions = pd.DataFrame(prediction_rows)
    fit_log = pd.DataFrame(fit_rows)
    summary, by_candidate, by_fold = (
        complete.audit_current_regression_envelope_admission(
            predictions,
            fit_log,
            [passing, failing],
            require_exact_frozen_failure_set=False,
        )
    )
    assert summary["guard_dependent_candidates"] == [failing.name]
    assert summary["eligible_gate_failure_count"] == 0
    assert summary["prediction_fit_log_fold_match"] is True
    passing_row = by_candidate.set_index("candidate").loc[passing.name]
    assert passing_row["overall_clip_rate"] == pytest.approx(0.01)
    assert passing_row["worst_fold_clip_rate"] == pytest.approx(0.05)
    assert bool(passing_row["guard_dependent"]) is False
    assert len(by_fold) == 6

    with pytest.raises(
        AssertionError, match="selection-eligible regression fails"
    ):
        complete.audit_current_regression_envelope_admission(
            predictions,
            fit_log,
            [passing, replace(failing, selection_eligible=True)],
            require_exact_frozen_failure_set=False,
        )


def test_amended_frozen_catalog_hash_is_exact(tmp_path) -> None:
    path, digest = complete.write_frozen_catalog(
        complete.candidate_catalog(), tmp_path
    )
    assert path.exists()
    assert digest == complete.EXPECTED_FULL_CATALOG_SHA256
    assert digest == "d355481e8aaa7a23c7e43470ab6d683f2f5017d9ac89c611e04adae8a6597d23"
