import json

import numpy as np
import pandas as pd
import pytest

from archive.run_all_round_base_skill_rebuild import (
    CandidateSpec,
    EXPECTED_CONTRACT_SHA256,
    HuberRidgeIRLS,
    _assert_contract,
    _candidate_fit_rows,
    _fit_candidate,
    assert_prediction_artifact_roles,
    build_candidate_catalog,
    build_market_context,
    chronological_batches,
    load_selected_spec,
    predict_candidate,
    split_market_selection_oof,
)


def _spec(target="T2", weather="W0", family="T", alpha=0.0, selectable=True):
    return CandidateSpec(
        candidate=f"test__{target}__{weather}__{family}__a{alpha:g}",
        target_arm=target,
        weather_arm=weather,
        feature_family=family,
        estimator="ridge",
        alpha=alpha,
        selectable=selectable,
        diagnostic_only=not selectable,
    )


def test_frozen_catalog_has_34_selectable_and_two_raw_diagnostics():
    catalog = build_candidate_catalog()
    assert len(catalog) == 36
    assert sum(spec.selectable for spec in catalog) == 34
    assert sum(spec.target_arm == "T0" for spec in catalog) == 2
    assert all(not spec.selectable for spec in catalog if spec.target_arm == "T0")
    assert {spec.alpha for spec in catalog if spec.estimator == "ridge"} == {
        100.0,
        1000.0,
        3000.0,
    }
    assert _assert_contract() == EXPECTED_CONTRACT_SHA256


def test_market_oof_admits_exactly_34_selectable_candidates():
    specs = build_candidate_catalog()
    catalog = pd.DataFrame(
        [
            {
                "candidate": spec.candidate,
                "selectable": spec.selectable,
                "run_status": "ok",
            }
            for spec in specs
        ]
    )
    oof = pd.DataFrame(
        {
            "candidate": [spec.candidate for spec in specs],
            "event_key": "event",
            "dg_id_key": "player",
            "prediction": 0.0,
        }
    )
    market, diagnostics = split_market_selection_oof(oof, catalog)
    assert market["candidate"].nunique() == 34
    assert diagnostics["candidate"].nunique() == 2
    assert set(diagnostics["candidate"]) == {
        spec.candidate for spec in specs if spec.target_arm == "T0"
    }


def test_event_start_batches_are_whole_and_chronological():
    events = pd.DataFrame(
        {
            "event_key": ["warm", "a", "b", "c"],
            "event_start_date": pd.to_datetime(
                ["2023-03-01", "2023-04-06", "2023-04-06", "2023-04-13"]
            ),
            "event_end_date": pd.to_datetime(
                ["2023-03-04", "2023-04-09", "2023-04-09", "2023-04-16"]
            ),
        }
    )
    batches = chronological_batches(events)
    assert batches == [
        (pd.Timestamp("2023-04-06"), ("a", "b")),
        (pd.Timestamp("2023-04-13"), ("c",)),
    ]


def test_2023_context_predicates_cascade_at_parquet_io_boundary(monkeypatch):
    features = pd.DataFrame(
        {
            "event_key": ["pga|2023|16|2023-01-05"],
            "event_start_date": pd.to_datetime(["2023-01-05"]),
            "year": [2023],
            "event_id": [16],
        }
    )
    cascade_filter_calls = []

    def fake_read_parquet(path, *, columns, filters):
        if str(path) == "bridge":
            assert ("year", "in", [2023]) in filters
            return pd.DataFrame(
                {
                    "year": [2023],
                    "event_id": [16],
                    "event_start": pd.to_datetime(["2023-01-05"]),
                    "dg_id": [1],
                    "downstream_preserved_total": [0.1],
                    "legacy_core_pred_step2": [0.2],
                    "baseline_r1_prediction": [0.3],
                }
            )[columns]
        cascade_filter_calls.append(filters)
        assert ("event_start_date", ">=", pd.Timestamp("2023-01-01")) in filters
        assert ("event_start_date", "<", pd.Timestamp("2024-01-01")) in filters
        # A sentinel 2024 row would be returned by this spy if either physical
        # date predicate were missing, so the assertion fires before it can be
        # materialized into the training process.
        common = {
            "event_key": ["pga|2023|16|2023-01-05"],
            "event_start_date": pd.to_datetime(["2023-01-05"]),
            "dg_id_key": ["1"],
            "round_num": [1],
        }
        if "model" in columns:
            common.update(
                {
                    "prediction": [0.3],
                    "model": ["C0_current_cascade_retrospective"],
                }
            )
        else:
            common.update(
                {
                    "legacy_stage_adjustment": [0.0],
                    "variant": ["C1"],
                    "candidate": ["current_architecture_pt_v2_approx_control"],
                }
            )
        return pd.DataFrame(common)[columns]

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    context = build_market_context(
        features, years=(2023,), bridge_path="bridge", cascade_path="cascade"
    )
    assert len(cascade_filter_calls) == 2
    assert context["event_start_date"].dt.year.eq(2023).all()
    assert len(context) == 1


def test_t0_precision_is_normalized_to_equal_event_mass():
    spec = _spec(target="T0")
    features = pd.DataFrame(
        {
            "event_key": ["big", "big", "small"],
            "event_start_date": pd.to_datetime(["2023-01-01"] * 3),
            "dg_id_key": ["1", "2", "3"],
            "x": [0.0, 1.0, 2.0],
        }
    )
    targets = pd.DataFrame(
        {
            "event_key": ["big", "big", "small"],
            "event_start_date": pd.to_datetime(["2023-01-01"] * 3),
            "dg_id_key": ["1", "2", "3"],
            "raw_obs4_target": [0.0, 1.0, 2.0],
            "raw_obs4_count": [4, 2, 2],
            "raw_obs4_weight": [1.0, 0.5, 0.5],
        }
    )
    fit = _candidate_fit_rows(
        features,
        spec,
        {"W0": targets},
        {"W0": pd.DataFrame()},
        {"big", "small"},
    )
    event_mass = fit.groupby("event_key")["_sample_weight"].sum()
    assert event_mass["big"] == pytest.approx(event_mass["small"])
    assert fit["_sample_weight"].mean() == pytest.approx(1.0)


def test_t2_retains_one_round_cases_and_recovers_absolute_intercept():
    spec = _spec(target="T2")
    features = pd.DataFrame(
        {
            "event_key": ["e1", "e1", "e2", "e2", "e3"],
            "event_start_date": pd.to_datetime(
                ["2023-01-01", "2023-01-01", "2023-01-08", "2023-01-08", "2023-01-15"]
            ),
            "dg_id_key": ["1", "2", "1", "2", "9"],
            "x": [-1.0, 1.0, -1.0, 1.0, 0.0],
        }
    )
    mar = features[["event_key", "event_start_date", "dg_id_key"]].copy()
    mar["event_round_key"] = ["e1|r1", "e1|r1", "e2|r1", "e2|r1", "e3|r1"]
    mar["raw_y"] = [1.0, 5.0, 11.0, 15.0, 21.0]
    mar["player_event_round_count"] = [2, 2, 2, 2, 1]
    fit = _candidate_fit_rows(
        features,
        spec,
        {"W0": pd.DataFrame()},
        {"W0": mar},
        {"e1", "e2", "e3"},
    )
    assert len(fit) == 5  # T2 explicitly retains the n=1 WD/DQ row.
    round_mass = fit.groupby("event_round_key")["_sample_weight"].sum()
    assert round_mass.nunique() == 1
    assert fit["_sample_weight"].mean() == pytest.approx(1.0)

    # Use the two balanced groups for the exact fixed-effect/intercept check;
    # their nuisance intercepts are 3 and 13, so the identifying level is 8.
    model = _fit_candidate(spec, fit.iloc[:4].copy(), ["x"])
    predicted = predict_candidate(model, pd.DataFrame({"x": [0.0]}))[0]
    assert predicted == pytest.approx(8.0, abs=1e-6)


def test_selected_spec_enforces_frozen_menus_and_accepts_live_alias(tmp_path):
    catalog = [_spec(alpha=100.0)]
    payload = {
        "candidate": catalog[0].candidate,
        "downstream_lambda": 0.5,
        "live_lambda": 1.0,
        "catalog_sha256": "catalog",
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
    }
    path = tmp_path / "selected.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    selected, normalized = load_selected_spec(
        path, catalog, expected_catalog_sha256="catalog"
    )
    assert selected == catalog[0]
    assert normalized["live_update_lambda"] == 1.0

    payload["downstream_lambda"] = 0.25
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="downstream_lambda"):
        load_selected_spec(path, catalog, expected_catalog_sha256="catalog")


def test_raw_and_composed_prediction_artifact_roles_are_unambiguous():
    raw = pd.DataFrame(
        {
            "prediction_stage": ["raw_base_core_before_downstream_and_live"],
            "downstream_lambda_applied": [False],
            "live_update_lambda_applied": [False],
        }
    )
    pre = pd.DataFrame(
        {
            "prediction_stage": ["composed_pre_event_after_downstream"],
            "downstream_lambda_applied": [True],
            "live_update_lambda_applied": [False],
        }
    )
    rounds = pd.DataFrame(
        {
            "prediction_stage": [
                "composed_round_after_downstream_and_live",
                "composed_round_after_downstream_and_live",
            ],
            "downstream_lambda_applied": [True, True],
            "live_update_lambda_applied": [False, True],
        }
    )
    assert_prediction_artifact_roles(raw, pre, rounds)
    bad = raw.copy()
    bad["downstream_lambda_applied"] = True
    with pytest.raises(AssertionError, match="downstream-applied"):
        assert_prediction_artifact_roles(bad, pre, rounds)


def test_huber_irls_is_deterministic_and_downweights_outlier():
    x = np.arange(10, dtype=float)[:, None]
    y = 2.0 * x[:, 0] + 1.0
    y[-1] += 100.0
    first = HuberRidgeIRLS(alpha=0.01, fit_intercept=True).fit(x, y)
    second = HuberRidgeIRLS(alpha=0.01, fit_intercept=True).fit(x, y)
    assert np.allclose(first.coef_, second.coef_)
    assert first.intercept_ == pytest.approx(second.intercept_)
    assert first.coef_[0] < 4.0
