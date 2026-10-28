from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from archive.build_shot_mean_factor_panel import (
    BASE_COLUMNS,
    FEATURE_COLUMN,
    OUTPUT_COLUMNS,
    build_cross_track_factor_panel,
    build_event_safe_app_history,
    standardize_distance_lie_panel,
)


def _round_rows(
    dg_id: int,
    year: int,
    event_id: int,
    start: str,
    values: list[float],
) -> list[dict[str, object]]:
    first = pd.Timestamp(start)
    return [
        {
            "dg_id": dg_id,
            "player_name": "test, player",
            "event_id": event_id,
            "year": year,
            "round_num": round_num,
            "round_date": first + pd.Timedelta(days=round_num - 1),
            "sg_app_adj": value,
        }
        for round_num, value in enumerate(values, start=1)
    ]


def test_event_safe_history_snapshots_concurrent_events_before_updates() -> None:
    rows = []
    rows += _round_rows(1, 2023, 1, "2023-01-01", [1.0] * 4)
    rows += _round_rows(1, 2023, 2, "2023-01-08", [2.0] * 4)
    rows += _round_rows(1, 2023, 3, "2023-01-15", [100.0] * 4)
    rows += _round_rows(1, 2023, 4, "2023-01-15", [-100.0] * 4)
    rows += _round_rows(1, 2023, 5, "2023-01-22", [3.0] * 4)

    history = build_event_safe_app_history(pd.DataFrame(rows))
    concurrent = history[history["event_id"].isin([3, 4])]

    assert len(concurrent) == 2
    assert concurrent["prior_app_mean50"].tolist() == pytest.approx([1.5, 1.5])
    assert concurrent["prior_app_sd50"].isna().all()
    next_event = history.loc[history["event_id"].eq(5)].iloc[0]
    assert next_event["prior_app_mean50"] == pytest.approx(0.75)
    assert np.isfinite(next_event["prior_app_sd50"])


def _training_panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    events = [
        ("pga:R2023001", 1, "2023-01-01"),
        ("pga:R2023002", 2, "2023-01-08"),
        ("pga:R2023003", 3, "2023-01-15"),
        ("pga:R2023004", 4, "2023-02-01"),
        ("pga:R2023005", 5, "2023-02-01"),
        ("pga:R2023006", 6, "2023-03-01"),
    ]
    for event_key, event_id, start in events:
        for player_offset, raw in enumerate([8.0, 9.0, 10.0, 11.0], start=1):
            centered_on_median = raw - 9.5
            rows.append(
                {
                    "event_key": event_key,
                    "year": 2023,
                    "event_id": event_id,
                    "start": pd.Timestamp(start),
                    "dg_id": 100 + player_offset,
                    "player_name": f"player {player_offset}",
                    FEATURE_COLUMN: raw,
                    "target_app": -0.2 * centered_on_median,
                    "prior_app_mean50": 0.0,
                    "prior_app_sd50": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_walk_forward_factor_uses_strictly_earlier_events_only() -> None:
    panel = _training_panel()
    output, audit = build_cross_track_factor_panel(
        panel,
        start="2023-02-01",
        end="2023-02-01",
        ridge_alpha=1e-9,
        min_fit_rows=8,
        min_fit_events=2,
    )

    assert set(OUTPUT_COLUMNS).issubset(output.columns)
    assert output["event_key"].nunique() == 2
    assert output["coefficient_fit_status"].eq("fitted").all()
    assert output["coefficient_sg_per_unit"].unique()[0] == pytest.approx(
        -0.2, abs=1e-7
    )
    assert output["coefficient_fit_max_start"].max() == pd.Timestamp("2023-01-15")
    assert output["coefficient_fit_strictly_prior"].all()
    assert audit["leakage_violations"] == 0
    assert audit["events"] == 2
    event_sums = output.groupby("event_key")["factor_delta_sg_per_round"].sum()
    assert event_sums.abs().max() < 1e-12


def test_future_and_concurrent_outcomes_cannot_change_target_coefficient() -> None:
    panel = _training_panel()
    kwargs = dict(
        start="2023-02-01",
        end="2023-02-01",
        ridge_alpha=1e-9,
        min_fit_rows=8,
        min_fit_events=2,
    )
    original, _ = build_cross_track_factor_panel(panel, **kwargs)

    changed = panel.copy()
    changed.loc[changed["start"].ge("2023-02-01"), "target_app"] = 1_000_000.0
    replay, _ = build_cross_track_factor_panel(changed, **kwargs)

    assert replay["coefficient_sg_per_unit"].to_numpy() == pytest.approx(
        original["coefficient_sg_per_unit"].to_numpy()
    )
    assert replay["factor_delta_sg_per_round"].to_numpy() == pytest.approx(
        original["factor_delta_sg_per_round"].to_numpy()
    )


def test_missing_player_feature_is_neutral_not_field_shifting() -> None:
    panel = _training_panel()
    missing_index = panel[
        panel["event_key"].eq("pga:R2023004") & panel["dg_id"].eq(101)
    ].index
    panel.loc[missing_index, FEATURE_COLUMN] = np.nan

    output, _ = build_cross_track_factor_panel(
        panel,
        start="2023-02-01",
        end="2023-02-01",
        ridge_alpha=1e-9,
        min_fit_rows=8,
        min_fit_events=2,
    )
    event = output[output["event_key"].eq("pga:R2023004")]
    missing = event[event["dg_id"].eq(101)].iloc[0]

    assert not missing["factor_available"]
    assert missing["factor_centered"] == 0.0
    assert missing["factor_delta_sg_per_round"] == 0.0
    assert event["factor_centered"].sum() == pytest.approx(0.0, abs=1e-12)


def test_insufficient_fit_is_unavailable_instead_of_using_future_data() -> None:
    panel = _training_panel()
    output, audit = build_cross_track_factor_panel(
        panel,
        start="2023-01-01",
        end="2023-01-01",
        min_fit_rows=1,
        min_fit_events=1,
    )

    assert output["coefficient_fit_status"].eq("insufficient_history").all()
    assert output["factor_delta_sg_per_round"].isna().all()
    assert audit["fitted_events"] == 0


def test_distance_lie_adapter_rejects_aggregate_results() -> None:
    aggregate = pd.DataFrame(
        [{"section": "r1_physical", "delta_rmse": -0.0012}]
    )
    roster = _training_panel().head(1)

    with pytest.raises(ValueError, match="genuine player/event panel"):
        standardize_distance_lie_panel(aggregate, roster)


def test_distance_lie_adapter_preserves_explicit_player_event_rows() -> None:
    roster = _training_panel().query("event_id == 4").copy()
    explicit = pd.DataFrame(
        {
            "event_key": roster["event_key"],
            "source_player_id": ["a", "b", "c", "d"],
            "player_name": roster["player_name"],
            "phys__exact8_full_k400": [-0.3, -0.1, 0.1, 0.3],
        }
    )
    output = standardize_distance_lie_panel(explicit, roster)

    assert len(output) == 4
    assert output["factor_available"].all()
    assert output["factor_centered"].sum() == pytest.approx(0.0)
    assert output["factor_delta_sg_per_round"].tolist() == pytest.approx(
        output["factor_centered"].tolist()
    )
    assert output["coefficient_fit_strictly_prior"].all()
