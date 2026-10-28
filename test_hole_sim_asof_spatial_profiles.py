from __future__ import annotations

import pandas as pd
import pytest

from archive.hole_sim_asof_spatial_profiles import (
    materialize_normalization_profiles,
    strict_prior_rows,
)


def test_strict_prior_rows_requires_completed_events() -> None:
    history = pd.DataFrame(
        {
            "event_key": ["before", "overlap", "after"],
            "event_end": pd.to_datetime(
                ["2025-06-15", "2025-06-20", "2025-06-25"], utc=True
            ),
            "value": [1, 2, 3],
        }
    )
    result = strict_prior_rows(history, pd.Timestamp("2025-06-20", tz="UTC"))
    assert result["event_key"].tolist() == ["before"]


def _spatial_profiles() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_key": "target",
                "event_start": pd.Timestamp("2026-06-25", tz="UTC"),
                "source_player_id": player,
                "sg_category": "ott",
                "state_key": "tee|par=4",
                "profile_cutoff": pd.Timestamp("2026-06-18", tz="UTC"),
                "distance_ratio": ratio,
                "long_bias_yd": 0.0,
                "lat_bias_yd": 0.0,
                "long_scale": 1.0,
                "lat_scale": 1.0,
                "long_lat_corr": 0.0,
                "ability_sg": 0.0,
                "exposure_weight": 1.0,
                "normalization_source": "strict_prior_test",
                "snapshot_kind": "strict_prior_asof",
                "ability_weighting_status": "disabled_unestimated_beta",
            }
            for player, ratio in [("p1", 1.05), ("p2", 0.95)]
        ]
    )


def test_materializer_expands_only_observed_setup_conditions() -> None:
    conditions = pd.DataFrame(
        [
            {
                "event_key": "target",
                "sg_category": "ott",
                "state_key": "tee|par=4",
                "setup_key": setup,
                "reference_long_yd": center,
                "reference_lat_yd": 0.0,
                "reference_cutoff": pd.Timestamp("2026-06-17", tz="UTC"),
            }
            for setup, center in [("target|course=1|r=1|h=1", 290.0), ("target|course=1|r=2|h=1", 285.0)]
        ]
    )
    result = materialize_normalization_profiles(_spatial_profiles(), conditions)
    assert len(result) == 4
    assert result["source_player_id"].nunique() == 2
    assert result["setup_key"].nunique() == 2
    assert result["event_start"].notna().all()
    assert (result["event_start"] == pd.Timestamp("2026-06-25", tz="UTC")).all()
    assert (result["profile_cutoff"] == pd.Timestamp("2026-06-18", tz="UTC")).all()
    assert result["materialization_sha256"].str.len().eq(64).all()
    assert result["ability_weighting_status"].eq("disabled_unestimated_beta").all()


def test_materializer_rejects_heldout_setup_reference() -> None:
    conditions = pd.DataFrame(
        [
            {
                "event_key": "target",
                "sg_category": "ott",
                "state_key": "tee|par=4",
                "setup_key": "target|course=1|r=1|h=1",
                "reference_long_yd": 290.0,
                "reference_lat_yd": 0.0,
                "reference_cutoff": pd.Timestamp("2026-06-25", tz="UTC"),
            }
        ]
    )
    with pytest.raises(ValueError, match="strictly predate event start"):
        materialize_normalization_profiles(_spatial_profiles(), conditions)
