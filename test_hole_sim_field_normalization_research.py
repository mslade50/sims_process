from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import archive.hole_sim_field_normalization as normalization_module
from archive.hole_sim_field_normalization import (
    NormalizationConfig,
    assert_physical_transition_ready,
    latest_strict_prior_profiles,
    normalize_donor_points,
    reapply_target_profile,
)


def _classifier(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    lateral = out["model_lat_yd"].abs()
    out["model_end_hazard"] = np.where(lateral > 6.0, "water", "none")
    out["model_end_lie"] = np.select(
        [lateral > 6.0, lateral > 2.0], ["water", "rough"], default="fairway"
    )
    return out


def _profiles(event: str, cutoff: str, *, target: bool = False) -> pd.DataFrame:
    players = ["target", "peer"] if target else ["p1", "p2", "roster_only"]
    rows = []
    for player in players:
        if target:
            traits = {
                "target": (0.5, 1.05, 0.0, 0.0, 1.5, 2.0),
                "peer": (-0.5, 0.95, 0.0, 0.0, 1.0, 1.0),
            }[player]
        else:
            traits = {
                "p1": (1.0, 1.10, 2.0, 1.0, 2.0, 1.0),
                "p2": (-1.0, 0.90, -1.0, -1.0, 1.0, 1.0),
                # This player has no donor shot.  Its ability must nevertheless
                # enter the pre-event field-composition calculation.
                "roster_only": (10.0, 1.0, 0.0, 0.0, 1.0, 1.0),
            }[player]
        ability, ratio, long_bias, lat_bias, long_scale, lat_scale = traits
        rows.append(
            {
                "event_key": event,
                "source_player_id": player,
                "sg_category": "ott",
                "state_key": "tee_par4",
                "setup_key": "hole_1_r1",
                "profile_cutoff": cutoff,
                "reference_long_yd": 100.0,
                "reference_lat_yd": 0.0,
                "ability_sg": ability,
                "distance_ratio": ratio,
                "long_bias_yd": long_bias,
                "lat_bias_yd": lat_bias,
                "long_scale": long_scale,
                "lat_scale": lat_scale,
                "long_lat_corr": 0.0,
                "exposure_weight": 1.0,
                "normalization_source": "unit_test_strict_prior",
                "snapshot_kind": "strict_prior_asof",
            }
        )
    return pd.DataFrame(rows)


def _donors() -> pd.DataFrame:
    # Both raw points were generated from the corresponding profile transform.
    # p1 neutral residual=(3, 4): [100,0] + [12,1] + diag(2,1)*[3,4]
    # p2 neutral residual=(-2, 1): [100,0] + [-11,-1] + I*[-2,1]
    return pd.DataFrame(
        [
            {
                "event_key": "donor",
                "event_start": "2025-06-20",
                "source_player_id": "p1",
                "sg_category": "ott",
                "state_key": "tee_par4",
                "setup_key": "hole_1_r1",
                "long_yd": 118.0,
                "lat_yd": 5.0,
                "end_lie": "fairway",
                "end_hazard": "none",
                "sg_raw": 0.4,
            },
            {
                "event_key": "donor",
                "event_start": "2025-06-20",
                "source_player_id": "p2",
                "sg_category": "ott",
                "state_key": "tee_par4",
                "setup_key": "hole_1_r1",
                "long_yd": 87.0,
                "lat_yd": 0.0,
                "end_lie": "rough",
                "end_hazard": "none",
                "sg_raw": -0.2,
            },
        ]
    )


def test_latest_profiles_are_strict_prior_across_field_strength_jump() -> None:
    roster = pd.DataFrame(
        [
            {"event_key": "travelers_2024", "event_start": "2024-06-20", "source_player_id": "7"},
            {"event_key": "travelers_2025", "event_start": "2025-06-19", "source_player_id": "7"},
        ]
    )
    history = pd.DataFrame(
        [
            {"dg_id": 7, "Date": "2024-06-19", "skill_est": 0.25},
            # Equal-date data are ineligible for the 2024 event.
            {"dg_id": 7, "Date": "2024-06-20", "skill_est": 9.0},
            {"dg_id": 7, "Date": "2025-06-18", "skill_est": 1.10},
            # Held-out/post-start data are ineligible for the 2025 event.
            {"dg_id": 7, "Date": "2025-06-19", "skill_est": 8.0},
        ]
    )
    result = latest_strict_prior_profiles(
        roster, history, history_player_col="dg_id"
    )
    assert result["skill_est"].tolist() == [0.25, 1.10]
    assert result["snapshot_kind"].tolist() == ["strict_prior_asof"] * 2
    assert pd.to_datetime(result.loc[0, "profile_cutoff"], utc=True) < pd.Timestamp(
        "2024-06-20", tz="UTC"
    )
    assert pd.to_datetime(result.loc[1, "profile_cutoff"], utc=True) < pd.Timestamp(
        "2025-06-19", tz="UTC"
    )


def test_player_state_normalization_and_target_reapplication_are_coherent() -> None:
    neutral, metadata = normalize_donor_points(
        _donors(),
        _profiles("donor", "2025-06-19"),
        mode="player_state_residualized",
        geometry_classifier=_classifier,
    )
    assert neutral["model_long_yd"].tolist() == pytest.approx([103.0, 98.0])
    assert neutral["model_lat_yd"].tolist() == pytest.approx([4.0, 1.0])
    assert neutral["neutral_resid_long_yd"].tolist() == pytest.approx([3.0, -2.0])
    assert neutral["neutral_resid_lat_yd"].tolist() == pytest.approx([4.0, 1.0])
    # Field composition is from all three pre-event roster profiles, not just
    # the two players who happened to produce these donor atoms.
    assert neutral["field_ability_sg"].tolist() == pytest.approx([10.0 / 3.0] * 2)
    assert (neutral["coherence_status"] == "geometry_reclassified").all()
    assert metadata["strict_prior"] is True
    assert len(metadata["provenance_sha256"]) == 64

    def reward(frame: pd.DataFrame) -> pd.Series:
        return pd.Series(
            np.where(frame["model_end_hazard"].eq("water"), -1.0, 0.25),
            index=frame.index,
        )

    reapplied, target_metadata = reapply_target_profile(
        neutral,
        _profiles("target_event", "2026-08-19", target=True),
        target_player_id="target",
        target_event_start="2026-08-20",
        geometry_classifier=_classifier,
        reward_function=reward,
    )
    # target point = [100,0] + [5,0] + diag(1.5,2.0)*neutral residual
    assert reapplied["model_long_yd"].tolist() == pytest.approx([109.5, 102.0])
    assert reapplied["model_lat_yd"].tolist() == pytest.approx([8.0, 2.0])
    assert reapplied["model_end_hazard"].tolist() == ["water", "none"]
    assert reapplied["model_end_lie"].tolist() == ["water", "fairway"]
    assert reapplied["model_sg"].tolist() == pytest.approx([-1.0, 0.25])
    assert reapplied["model_weight"].mean() == pytest.approx(1.0)
    assert target_metadata["strict_prior"] is True
    assert target_metadata["parent_provenance_sha256"] == [metadata["provenance_sha256"]]
    assert_physical_transition_ready(reapplied)


def test_app_target_reapplication_can_defer_physical_classification() -> None:
    donors = _donors()
    donor_profiles = _profiles("donor", "2025-06-19")
    target_profiles = _profiles("target_event", "2026-08-19", target=True)
    for frame in (donors, donor_profiles, target_profiles):
        frame["sg_category"] = "app"
        frame["state_key"] = "app|lie=fairway|band=150-215"

    neutral, _ = normalize_donor_points(
        donors,
        donor_profiles,
        mode="player_state_residualized",
    )
    deferred, metadata = reapply_target_profile(
        neutral,
        target_profiles,
        target_player_id="target",
        target_event_start="2026-08-20",
        geometry_classification_mode="deferred_shot_relative",
    )
    assert deferred["deferred_miss_long_yd"].tolist() == pytest.approx([109.5, 102.0])
    assert deferred["deferred_miss_lat_yd"].tolist() == pytest.approx([8.0, 2.0])
    assert deferred["model_end_lie"].isna().all()
    assert deferred["model_end_hazard"].isna().all()
    assert deferred["model_sg"].isna().all()
    assert (
        deferred["geometry_classification_status"]
        == "deferred_until_generated_start"
    ).all()
    assert not deferred["physical_transition_ready"].any()
    assert metadata["geometry_reclassified"] is False
    assert metadata["physical_transition_ready"] is False
    with pytest.raises(ValueError, match="requires batched target-frame"):
        assert_physical_transition_ready(deferred)


def test_tee_target_reapplication_cannot_defer_geometry() -> None:
    neutral, _ = normalize_donor_points(
        _donors(),
        _profiles("donor", "2025-06-19"),
        mode="player_state_residualized",
    )
    with pytest.raises(ValueError, match="permitted only for APP"):
        reapply_target_profile(
            neutral,
            _profiles("target_event", "2026-08-19", target=True),
            target_player_id="target",
            target_event_start="2026-08-20",
            geometry_classification_mode="deferred_shot_relative",
        )

    with pytest.raises(ValueError, match="requires geometry_classifier"):
        reapply_target_profile(
            neutral,
            _profiles("target_event", "2026-08-19", target=True),
            target_player_id="target",
            target_event_start="2026-08-20",
        )


def test_transformed_labels_are_invalidated_without_geometry() -> None:
    neutral, _ = normalize_donor_points(
        _donors(),
        _profiles("donor", "2025-06-19"),
        mode="event_centered",
    )
    assert neutral["model_end_lie"].isna().all()
    assert neutral["model_end_hazard"].isna().all()
    assert (neutral["coherence_status"] == "labels_invalidated_neutral_latent").all()
    assert neutral["raw_end_lie"].tolist() == ["fairway", "rough"]


def test_equal_or_future_profile_cutoff_fails_closed() -> None:
    bad = _profiles("donor", "2025-06-20")
    with pytest.raises(ValueError, match="strictly before event start"):
        normalize_donor_points(_donors(), bad)


def test_static_snapshot_is_target_only_and_cannot_normalize_donors() -> None:
    static = _profiles("donor", "2025-06-19")
    static["snapshot_kind"] = "static_target_snapshot"
    with pytest.raises(ValueError, match="not allowed here"):
        normalize_donor_points(_donors(), static)


def test_nonzero_target_ability_requires_recomputed_reward() -> None:
    config = NormalizationConfig(ability_log_weight_per_sg2=1.0)
    neutral, _ = normalize_donor_points(
        _donors(),
        _profiles("donor", "2025-06-19"),
        geometry_classifier=_classifier,
        config=config,
    )
    with pytest.raises(ValueError, match="reward_function is required"):
        reapply_target_profile(
            neutral,
            _profiles("target_event", "2026-08-19", target=True),
            target_player_id="target",
            target_event_start="2026-08-20",
            geometry_classifier=_classifier,
            config=config,
        )


@pytest.mark.parametrize(
    "mode", ["raw", "event_centered", "player_state_residualized"]
)
def test_vectorized_transform_matches_scalar_reference(mode: str) -> None:
    output, metadata = normalize_donor_points(
        _donors(),
        _profiles("donor", "2025-06-19"),
        mode=mode,
        geometry_classifier=_classifier,
    )
    config = NormalizationConfig()
    scalar = [
        normalization_module._transform_row_to_neutral(row, mode, config)
        for _, row in output.iterrows()
    ]
    scalar_points = np.vstack([item[0] for item in scalar])
    scalar_residuals = np.vstack([item[1] for item in scalar])
    assert output[["model_long_yd", "model_lat_yd"]].to_numpy() == pytest.approx(
        scalar_points, abs=1e-12
    )
    assert output[["neutral_resid_long_yd", "neutral_resid_lat_yd"]].to_numpy() == pytest.approx(
        scalar_residuals, abs=1e-12
    )
    assert metadata["spatial_transform_backend"] == "vectorized_closed_form_2d_v1"
