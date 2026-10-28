from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from archive import explore_base_skill_discovery_extensions as extension


def _prediction_panel() -> pd.DataFrame:
    rows = []
    for candidate in ("a", "b"):
        for index, event in enumerate(("e1", "e2"), start=1):
            start = pd.Timestamp(f"2021-0{index}-10")
            rows.append(
                {
                    "candidate": candidate,
                    "event_key": event,
                    "dg_id_key": f"p{index}",
                    "event_start_date": start,
                    "prediction": 0.1 * index,
                    "prediction_asof": start - pd.Timedelta(days=1),
                    "chronology_ok": True,
                    "training_max_event_end": start - pd.Timedelta(days=7),
                }
            )
    return pd.DataFrame(rows)


def _logical_panel() -> pd.DataFrame:
    rows = []
    for candidate, probability in (("base", 0.50), ("challenger", 0.60)):
        for market in ("72h", "r1"):
            rows.append(
                {
                    "candidate": candidate,
                    "mapped_event_key": "event-1",
                    "event_start_date": pd.Timestamp("2021-01-07"),
                    "market": market,
                    "pair_key": "1|2",
                    "canonical_outcome": 1.0,
                    "canonical_probability": probability,
                    "canonical_push_probability": 0.02,
                }
            )
    return pd.DataFrame(rows)


def _proper_gate_frame() -> pd.DataFrame:
    rows = []
    for market in ("72h", "r1"):
        rows.append(
            {
                "candidate": "candidate",
                "baseline": "baseline",
                "market": market,
                "coverage_ratio_vs_baseline": 1.0,
                "log_loss_delta": -0.01,
                "brier_delta": -0.01,
                "candidate_calibration_intercept": 0.01,
                "baseline_calibration_intercept": 0.02,
                "candidate_calibration_slope": 0.95,
                "baseline_calibration_slope": 0.90,
                "candidate_ece_decile": 0.01,
                "baseline_ece_decile": 0.02,
                "candidate_calibration_fit_success": True,
                "baseline_calibration_fit_success": True,
            }
        )
    return pd.DataFrame(rows)


def test_ridge_identity_fit_matches_closed_form_and_shrinks() -> None:
    x = np.asarray([[1.0], [2.0], [3.0]])
    y = np.asarray([2.0, 4.0, 6.0])
    events = ["a", "b", "c"]
    unpenalized = extension.ridge_identity_fit(
        x, y, events, alpha=0.0, prior=np.asarray([0.0])
    )
    heavily_penalized = extension.ridge_identity_fit(
        x, y, events, alpha=1_000_000.0
    )
    assert unpenalized[0] == pytest.approx(2.0)
    assert heavily_penalized[0] == pytest.approx(1.0, abs=2e-5)
    with pytest.raises(ValueError, match="non-finite"):
        extension.ridge_identity_fit(x, np.asarray([2.0, np.nan, 6.0]), events, 1.0)


def test_blend_weight_recovers_and_clips() -> None:
    anchor = np.zeros(4)
    challenger = np.asarray([1.0, 2.0, 3.0, 4.0])
    events = ["a", "b", "c", "d"]
    recovered = extension.fit_blend_weight(
        anchor,
        challenger,
        0.3 * challenger,
        events,
        prior_weight=0.0,
        alpha=0.0,
    )
    assert recovered == pytest.approx(0.3)
    assert extension.fit_blend_weight(
        anchor, challenger, 2.0 * challenger, events, prior_weight=0.0, alpha=0.0
    ) == pytest.approx(1.0)
    assert extension.fit_blend_weight(
        anchor, challenger, -challenger, events, prior_weight=0.0, alpha=0.0
    ) == pytest.approx(0.0)


def test_alpha_selection_weights_events_not_start_dates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(extension, "MIN_TRAINING_ROWS", 1)
    monkeypatch.setattr(extension, "MIN_TRAINING_EVENTS", 1)

    def fake_beta(_train: pd.DataFrame, alpha: float) -> np.ndarray:
        value = {1.0: 0.0, 10.0: 2.0, 100.0: 10.0}[float(alpha)]
        return np.asarray([value, 0.0, 0.0, 0.0, 0.0, 0.0])

    monkeypatch.setattr(extension, "_fit_bridge_beta", fake_beta)
    rows = []
    specifications = [
        ("train", "2020-01-01", "2020-01-04", 0.0),
        ("same-date-a", "2020-02-01", "2020-02-04", 0.0),
        ("same-date-b", "2020-02-01", "2020-02-04", 0.0),
        ("solo-date", "2020-03-01", "2020-03-04", np.sqrt(6.0)),
    ]
    for event, start, end, label in specifications:
        row = {
            "event_key": event,
            "event_start_date": pd.Timestamp(start),
            "event_end_date": pd.Timestamp(end),
            "label_dg_sg_adj_r1_r2_count": 2,
            "label_dg_sg_adj_r1_r2_mean": label,
            "base_prediction": 0.0,
        }
        row.update({component: 0.0 for component in extension.COMPONENTS})
        row[extension.COMPONENTS[0]] = 1.0
        rows.append(row)
    selected, folds, losses = extension.choose_bridge_alpha(pd.DataFrame(rows))
    assert folds == 2
    assert selected == 1.0
    assert losses["1.0"] < losses["10.0"]


@pytest.mark.parametrize(
    "mutation,pattern",
    [
        (lambda frame: frame.assign(prediction=np.inf), "non-finite"),
        (lambda frame: frame.assign(chronology_ok="True"), "exact boolean"),
        (
            lambda frame: frame.assign(
                prediction_asof=frame["event_start_date"] + pd.Timedelta(days=1)
            ),
            "as-of chronology",
        ),
    ],
)
def test_prediction_panel_rejects_bad_domains(mutation, pattern: str) -> None:
    with pytest.raises(ValueError, match=pattern):
        extension.validate_exact_prediction_panel(
            mutation(_prediction_panel()), ["a", "b"], label="test"
        )


def test_prediction_panel_rejects_equal_size_different_menu() -> None:
    frame = _prediction_panel()
    frame.loc[(frame["candidate"] == "b") & (frame["event_key"] == "e2"), "event_key"] = "e3"
    with pytest.raises(ValueError, match="menu differs"):
        extension.validate_exact_prediction_panel(frame, ["a", "b"], label="test")


def test_event_deltas_reject_challenger_only_key_and_nonfinite() -> None:
    logical = _logical_panel()
    extra = logical.loc[
        logical["candidate"].eq("challenger") & logical["market"].eq("72h")
    ].copy()
    extra["mapped_event_key"] = "event-extra"
    with pytest.raises(ValueError, match="extra=1"):
        extension.event_loss_deltas(
            pd.concat([logical, extra], ignore_index=True), ["challenger"], "base"
        )
    logical.loc[logical["candidate"].eq("challenger"), "canonical_probability"] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        extension.event_loss_deltas(logical, ["challenger"], "base")


def test_year_stratified_search_is_deterministic_and_familywise() -> None:
    rows = []
    for candidate, delta in (("good", -0.1), ("bad", 0.1)):
        for event, date in (("e1", "2021-01-07"), ("e2", "2022-01-06")):
            rows.append(
                {
                    "candidate": candidate,
                    "mapped_event_key": event,
                    "event_start_date": pd.Timestamp(date),
                    "loss_delta": delta,
                }
            )
    frame = pd.DataFrame(rows)
    first, first_audit = extension.year_stratified_search_inference(
        frame, draws=500, seed=123
    )
    second, second_audit = extension.year_stratified_search_inference(
        frame, draws=500, seed=123
    )
    pd.testing.assert_frame_equal(first, second)
    assert first_audit == second_audit
    certified = first.set_index("candidate")["familywise_improvement_certified"]
    assert bool(certified["good"])
    assert not bool(certified["bad"])


def test_inference_certification_requires_every_event_and_week_block() -> None:
    hypothesis = "candidate___versus___baseline"
    inference = pd.DataFrame(
        {"candidate": [hypothesis], "familywise_improvement_certified": [True]}
    )
    block = {
        "status": "complete",
        "candidate_simultaneous_95pct_upper_bound": {hypothesis: -0.01},
    }
    audit = {
        "moving_event_block_sensitivity": {str(i): copy.deepcopy(block) for i in range(1, 5)},
        "calendar_week_block_sensitivity": {str(i): copy.deepcopy(block) for i in range(1, 5)},
    }
    assert extension.inference_certified_under_all_resamples(
        inference, audit, hypothesis
    )
    audit["calendar_week_block_sensitivity"]["3"][
        "candidate_simultaneous_95pct_upper_bound"
    ][hypothesis] = 0.001
    assert not extension.inference_certified_under_all_resamples(
        inference, audit, hypothesis
    )


def test_market_gate_requires_successful_calibration() -> None:
    frame = _proper_gate_frame()
    assert extension._market_gate(frame, "candidate")
    frame.loc[frame.index[0], "candidate_calibration_fit_success"] = False
    assert not extension._market_gate(frame, "candidate")


def test_each_money_guardrail_is_fail_closed() -> None:
    passing = {
        "close_pnl_delta": 0.0,
        "close_roi_delta": 0.0,
        "candidate_mean_open_to_close_clv": 0.0,
        "open_to_close_clv_delta": 0.0,
        "edge_violation_magnitude_delta": 0.0,
        "max_drawdown_delta": 0.0,
        "worst_event_loss_delta": 0.0,
    }
    assert extension._guardrail_gate(passing)
    for key in passing:
        failed = dict(passing)
        failed[key] = -0.001 if key in {
            "close_pnl_delta",
            "close_roi_delta",
            "candidate_mean_open_to_close_clv",
            "open_to_close_clv_delta",
        } else 0.001
        assert not extension._guardrail_gate(failed), key


def test_dependency_hash_drift_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    altered = dict(extension.EXPECTED_DEPENDENCY_SHA256)
    altered["analyze_base_skill_discovery_search_correction.py"] = "0" * 64
    monkeypatch.setattr(extension, "EXPECTED_DEPENDENCY_SHA256", altered)
    with pytest.raises(ValueError, match="dependency hash changed"):
        extension.validate_dependency_authority()

