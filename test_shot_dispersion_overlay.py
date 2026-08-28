import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shot_dispersion_overlay import apply_shot_dispersion_overlay
from portable_hash import LF_NORMALIZED_HASH_MODE, lf_normalized_sha256


CATS = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]
REPO_ROOT = Path(__file__).resolve().parent


def _sha256(path):
    return lf_normalized_sha256(path)


def test_overlay_matches_variance_formula_and_preserves_field_mean(tmp_path):
    players = ["alpha, a", "beta, b"]
    base = pd.DataFrame(
        {
            "sg_ott": [1.0, 2.0],
            "sg_app": [2.0, 1.0],
            "sg_arg": [1.5, 0.5],
            "sg_putt": [0.8, 1.6],
        },
        index=players,
    )
    original = base.copy()
    features = pd.DataFrame(
        {
            "player_name": players,
            "ott_shot_indep_var50_shrunk": [4.0, 1.0],
            "app_shot_indep_var50_shrunk": [1.0, 4.0],
            "arg_shot_indep_var50_shrunk": [3.0, 1.0],
            "putt_shot_indep_var50_shrunk": [1.0, 3.0],
        }
    )
    feature_path = tmp_path / "features.csv"
    dists_path = tmp_path / "dists.csv"
    config_path = tmp_path / "config.json"
    features.to_csv(feature_path, index=False)
    dists_path.write_text("hash-locked input", encoding="utf-8")
    weights = {"sg_ott": 0.90, "sg_app": 0.90, "sg_arg": 0.80, "sg_putt": 0.95}
    config_path.write_text(
        json.dumps(
            {
                "enabled": True,
                "tourney": "bmw",
                "event_id": 28,
                "expected_field_size": 2,
                "feature_file": str(feature_path),
                "feature_sha256": _sha256(feature_path),
                "distribution_sha256": _sha256(dists_path),
                "weights": weights,
            }
        ),
        encoding="utf-8",
    )

    actual = apply_shot_dispersion_overlay(
        base,
        players,
        CATS,
        tourney="bmw",
        event_id=28,
        dists_path=dists_path,
        config_path=config_path,
    )

    pd.testing.assert_frame_equal(base, original)
    for cat in CATS:
        shot_col = f"{cat.removeprefix('sg_')}_shot_indep_var50_shrunk"
        base_var = original[cat].to_numpy() ** 2
        shot_var = features[shot_col].to_numpy()
        scaled_shot_var = shot_var * base_var.mean() / shot_var.mean()
        expected_var = (1.0 - weights[cat]) * base_var + weights[cat] * scaled_shot_var
        np.testing.assert_allclose(actual[cat].to_numpy() ** 2, expected_var)
        np.testing.assert_allclose(expected_var.mean(), base_var.mean())


def test_live_overlay_accepts_a_subset_of_the_frozen_roster(tmp_path):
    players = ["alpha, a", "beta, b"]
    active_players = ["alpha, a"]
    base = pd.DataFrame({cat: [1.0] for cat in CATS}, index=active_players)
    features = pd.DataFrame(
        {
            "player_name": players,
            "ott_shot_indep_var50_shrunk": [2.0, 1.0],
            "app_shot_indep_var50_shrunk": [2.0, 1.0],
            "arg_shot_indep_var50_shrunk": [2.0, 1.0],
            "putt_shot_indep_var50_shrunk": [2.0, 1.0],
        }
    )
    feature_path = tmp_path / "features.csv"
    dists_path = tmp_path / "dists.csv"
    config_path = tmp_path / "config.json"
    features.to_csv(feature_path, index=False)
    dists_path.write_text("hash-locked input", encoding="utf-8")
    config_path.write_text(
        json.dumps(
            {
                "enabled": True,
                "tourney": "bmw",
                "event_id": 28,
                "expected_field_size": 2,
                "feature_file": str(feature_path),
                "feature_sha256": _sha256(feature_path),
                "distribution_sha256": _sha256(dists_path),
                "weights": {cat: 0.9 for cat in CATS},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected 2 players, got 1"):
        apply_shot_dispersion_overlay(
            base,
            active_players,
            CATS,
            tourney="bmw",
            event_id=28,
            dists_path=dists_path,
            config_path=config_path,
        )

    actual = apply_shot_dispersion_overlay(
        base,
        active_players,
        CATS,
        tourney="bmw",
        event_id=28,
        dists_path=dists_path,
        config_path=config_path,
        allow_active_subset=True,
    )

    assert list(actual.index) == active_players
    assert np.isfinite(actual.loc[active_players, CATS].to_numpy()).all()

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["expected_field_size"] = 3
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen feature roster"):
        apply_shot_dispersion_overlay(
            base,
            active_players,
            CATS,
            tourney="bmw",
            event_id=28,
            dists_path=dists_path,
            config_path=config_path,
            allow_active_subset=True,
        )


def test_overlay_is_noop_outside_configured_event(tmp_path):
    base = pd.DataFrame({cat: [1.0] for cat in CATS}, index=["alpha, a"])
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"enabled": True, "tourney": "bmw", "event_id": 28}),
        encoding="utf-8",
    )

    actual = apply_shot_dispersion_overlay(
        base,
        ["alpha, a"],
        CATS,
        tourney="next_week",
        event_id=29,
        dists_path=tmp_path / "not-needed.csv",
        config_path=config_path,
    )

    assert actual is base


def test_required_weekly_config_rejects_event_mismatch(tmp_path):
    base = pd.DataFrame({cat: [1.0] for cat in CATS}, index=["alpha, a"])
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "enabled": True,
                "required_current_event": True,
                "tourney": "bmw",
                "event_id": 28,
                "feature_file": "shot_dispersion_features.csv",
                "feature_sha256": "feature",
                "distribution_sha256": "dists",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="does not match the active event"):
        apply_shot_dispersion_overlay(
            base,
            ["alpha, a"],
            CATS,
            tourney="next_week",
            event_id=29,
            dists_path=tmp_path / "not-needed.csv",
            config_path=config_path,
        )


def test_required_weekly_config_cannot_be_disabled_by_environment(
    tmp_path, monkeypatch
):
    base = pd.DataFrame({cat: [1.0] for cat in CATS}, index=["alpha, a"])
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "enabled": True,
                "required_current_event": True,
                "tourney": "bmw",
                "event_id": 28,
                "feature_file": "shot_dispersion_features.csv",
                "feature_sha256": "feature",
                "distribution_sha256": "dists",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SHOT_DISPERSION_DISABLE", "1")

    with pytest.raises(RuntimeError, match="cannot bypass"):
        apply_shot_dispersion_overlay(
            base,
            ["alpha, a"],
            CATS,
            tourney="bmw",
            event_id=28,
            dists_path=tmp_path / "not-needed.csv",
            config_path=config_path,
        )


def test_portable_hash_ignores_platform_newlines(tmp_path):
    lf_path = tmp_path / "lf.csv"
    crlf_path = tmp_path / "crlf.csv"
    lf_path.write_bytes(b"player,value\nalpha,1\n")
    crlf_path.write_bytes(b"player,value\r\nalpha,1\r\n")

    assert lf_normalized_sha256(lf_path) == lf_normalized_sha256(crlf_path)
    assert LF_NORMALIZED_HASH_MODE == "sha256_lf_normalized_v1"


@pytest.mark.parametrize(
    "contents, error_type, pattern",
    [
        (None, FileNotFoundError, "required"),
        ("not-json", ValueError, "invalid JSON"),
        ('{"weights": {}}', ValueError, "explicit boolean enabled"),
        ('{"enabled": "false"}', ValueError, "explicit boolean enabled"),
        (
            '{"enabled": true, "required_current_event": "yes"}',
            ValueError,
            "required_current_event must be boolean",
        ),
    ],
)
def test_missing_or_invalid_config_is_fatal(tmp_path, contents, error_type, pattern):
    base = pd.DataFrame({cat: [1.0] for cat in CATS}, index=["alpha"])
    config_path = tmp_path / "shot_dispersion_config.json"
    if contents is not None:
        config_path.write_text(contents, encoding="utf-8")

    with pytest.raises(error_type, match=pattern):
        apply_shot_dispersion_overlay(
            base,
            ["alpha"],
            CATS,
            tourney="test",
            event_id=1,
            dists_path=tmp_path / "not-needed.csv",
            config_path=config_path,
        )


def test_explicit_disabled_config_is_a_valid_opt_out(tmp_path):
    base = pd.DataFrame({cat: [1.0] for cat in CATS}, index=["alpha"])
    config_path = tmp_path / "shot_dispersion_config.json"
    config_path.write_text('{"enabled": false}', encoding="utf-8")

    actual = apply_shot_dispersion_overlay(
        base,
        ["alpha"],
        CATS,
        tourney="test",
        event_id=1,
        dists_path=tmp_path / "not-needed.csv",
        config_path=config_path,
    )

    assert actual is base


def test_production_snapshot_is_present_and_hash_locked():
    config_path = REPO_ROOT / "shot_dispersion_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    feature_path = REPO_ROOT / config["feature_file"]
    dists_path = REPO_ROOT / "this_week_dists_v2.csv"

    assert config["enabled"] is True
    assert config["required_current_event"] is True
    assert config["feature_file"] == "shot_dispersion_features.csv"
    assert feature_path == REPO_ROOT / "shot_dispersion_features.csv"
    assert feature_path.is_file()
    assert dists_path.is_file()
    assert _sha256(feature_path) == config["feature_sha256"]
    assert _sha256(dists_path) == config["distribution_sha256"]
