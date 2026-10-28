from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from archive import all_round_target_design as design


def _rounds() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add_event(
        event_key: str,
        start: str,
        player: int,
        values: list[float],
    ) -> None:
        start_date = pd.Timestamp(start)
        for round_num, value in enumerate(values, start=1):
            rows.append(
                {
                    "event_key": event_key,
                    "event_start_date": start_date,
                    "dg_id": player,
                    "round_num": round_num,
                    "round_date": start_date + pd.Timedelta(days=round_num - 1),
                    "dg_sg_total_field_adjusted": value,
                }
            )

    # Only R1-R2 from these earlier events may enter the target-event anchors.
    add_event("pga|2022|1|2022-12-01", "2022-12-01", 1, [2.0, 0.0])
    add_event("pga|2022|1|2022-12-01", "2022-12-01", 2, [-1.0, -1.0])

    # Player 2 misses the cut; players 1 and 3 play all four rounds.
    add_event("pga|2023|1|2023-01-05", "2023-01-05", 1, [1.0, 1.0, 0.5, 0.5])
    add_event("pga|2023|1|2023-01-05", "2023-01-05", 2, [-1.0, -1.0])
    add_event("pga|2023|1|2023-01-05", "2023-01-05", 3, [0.0, 0.0, 0.0, 0.0])
    return pd.DataFrame(rows)


def _expected_player_one_anchor() -> float:
    decay = np.exp2(-1.0 / design.EWM_HALF_LIFE_ROUNDS)
    ewm = (2.0 * decay + 0.0) / (decay + 1.0)
    ess = (decay + 1.0) ** 2 / (decay**2 + 1.0)
    return float(ewm * ess / (ess + design.PRIOR_ROUND_EQUIVALENT))


def test_all_round_targets_use_every_observed_round_and_exact_weights() -> None:
    targets, audit = design.build_all_round_targets(_rounds())
    event = targets.loc[targets["event_key"].str.contains("2023\\|1")].set_index(
        "dg_id_key"
    )

    assert event.loc["1", "raw_obs4_target"] == pytest.approx(0.75)
    assert event.loc["1", "raw_obs4_count"] == 4
    assert event.loc["1", "raw_obs4_weight"] == pytest.approx(1.0)
    assert event.loc["2", "raw_obs4_target"] == pytest.approx(-1.0)
    assert event.loc["2", "raw_obs4_count"] == 2
    assert event.loc["2", "raw_obs4_weight"] == pytest.approx(0.5)
    assert event.loc["3", "raw_obs4_count"] == 4

    assert audit["target_weight_formula"] == "observed_count / 4"
    assert audit["anchor_source_rounds"] == [1, 2]
    assert audit["anchor_strict_prior_violations"] == 0
    assert audit["legacy_columns_read"] == []


def test_survivor_composition_uses_strict_prior_candidate_independent_anchor() -> None:
    rounds = _rounds()
    targets, _ = design.build_all_round_targets(rounds)
    event = targets.loc[targets["event_key"].str.contains("2023\\|1")].set_index(
        "dg_id_key"
    )
    anchor_one = _expected_player_one_anchor()
    decay = np.exp2(-1.0 / design.EWM_HALF_LIFE_ROUNDS)
    ess = (decay + 1.0) ** 2 / (decay**2 + 1.0)
    anchor_two = -ess / (ess + design.PRIOR_ROUND_EQUIVALENT)
    field_anchor = (anchor_one + anchor_two + 0.0) / 3.0
    post_anchor = anchor_one / 2.0
    post_shift = post_anchor - field_anchor

    assert event.loc["1", "strict_prior_anchor"] == pytest.approx(anchor_one)
    assert event.loc["2", "strict_prior_anchor"] == pytest.approx(anchor_two)
    assert event.loc["3", "strict_prior_anchor"] == pytest.approx(0.0)
    assert event.loc["1", "survcomp_obs4_target"] == pytest.approx(
        (1.0 + 1.0 + (0.5 - post_shift) + (0.5 - post_shift)) / 4.0
    )
    assert event.loc["2", "survcomp_obs4_target"] == pytest.approx(-1.0)
    assert event.loc["3", "survcomp_obs4_target"] == pytest.approx(
        (0.0 + 0.0 - post_shift - post_shift) / 4.0
    )

    # Caller-supplied legacy/market data are mechanically ignored.
    contaminated = rounds.assign(
        legacy_skill_est_asof=np.linspace(-1000, 1000, len(rounds)),
        market_close_price=np.linspace(1, 999, len(rounds)),
    )
    rebuilt, _ = design.build_all_round_targets(contaminated)
    pd.testing.assert_frame_equal(targets, rebuilt)


def test_anchor_does_not_read_same_event_or_future_rounds() -> None:
    rounds = _rounds()
    targets, _ = design.build_all_round_targets(rounds)
    before = targets.loc[
        (targets["event_key"].str.contains("2023\\|1"))
        & targets["dg_id_key"].eq("1"),
        "strict_prior_anchor",
    ].iloc[0]

    future = pd.DataFrame(
        {
            "event_key": ["pga|2024|1|2024-01-04"] * 2,
            "event_start_date": pd.to_datetime(["2024-01-04"] * 2),
            "dg_id": [1, 1],
            "round_num": [1, 2],
            "round_date": pd.to_datetime(["2024-01-04", "2024-01-05"]),
            "dg_sg_total_field_adjusted": [1000.0, 1000.0],
        }
    )
    expanded = pd.concat([rounds, future], ignore_index=True)
    rebuilt, audit = design.build_all_round_targets(expanded)
    after = rebuilt.loc[
        (rebuilt["event_key"].str.contains("2023\\|1"))
        & rebuilt["dg_id_key"].eq("1"),
        "strict_prior_anchor",
    ].iloc[0]
    assert after == pytest.approx(before)
    assert audit["anchor_strict_prior_violations"] == 0


def test_mar_rows_expose_raw_corrected_and_centered_outcomes() -> None:
    rows, audit = design.build_mar_round_rows(_rounds())
    assert {
        "raw_y",
        "survcomp_y",
        "raw_y_event_round_centered",
        "survcomp_y_event_round_centered",
        "event_round_key",
        "mar_event_player_key",
        "composition_shift",
    }.issubset(rows.columns)
    centered = rows.groupby("event_round_key")[[
        "raw_y_event_round_centered",
        "survcomp_y_event_round_centered",
    ]].mean()
    assert centered.abs().to_numpy().max() < 1e-12
    assert audit["mar_max_abs_event_round_centered_mean"] < 1e-12

    target_rows = rows.loc[rows["event_key"].str.contains("2023\\|1")]
    r1 = target_rows.loc[target_rows["round_num"].eq(1)]
    assert (r1["composition_shift"] == 0.0).all()
    missed = target_rows.loc[target_rows["dg_id_key"].eq("2")]
    assert (missed["player_event_round_count"] == 2).all()
    assert not missed["nonstandard_completion"].any()


def test_duplicate_player_event_round_is_rejected() -> None:
    rounds = _rounds()
    duplicated = pd.concat([rounds, rounds.iloc[[0]]], ignore_index=True)
    with pytest.raises(AssertionError, match="duplicate player-event-round"):
        design.build_all_round_targets(duplicated)


def test_loader_reads_narrow_contract_and_filters(tmp_path) -> None:
    rounds = _rounds().assign(
        tour="pga",
        year=lambda frame: pd.to_datetime(frame["event_start_date"]).dt.year,
        legacy_skill_est_asof=999.0,
    )
    path = tmp_path / "panel.parquet"
    rounds.to_parquet(path, index=False)
    loaded = design.load_player_round_panel(
        path, tour="pga", start_year=2023, end_year=2023
    )
    assert set(design.REQUIRED_COLUMNS).issubset(loaded.columns)
    assert "legacy_skill_est_asof" not in loaded.columns
    assert loaded["year"].eq(2023).all()
