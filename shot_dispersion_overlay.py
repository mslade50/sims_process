"""Apply the frozen BMW shot-level dispersion feature to category SG stds.

The feature changes only player/category variance. Category means, course
multipliers, skew, correlations, weather, and the shared week latent remain in
their existing simulator paths. The config is event-scoped and hash-locked so
it cannot silently carry into another week or consume regenerated inputs.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "shot_dispersion_config.json"


def _normalise_name(value: object) -> str:
    return str(value).casefold().strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_config(config_path: str | os.PathLike[str] | None) -> tuple[dict, Path]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(
            "shot_dispersion_config.json is required; set enabled=false in a "
            "readable config for a deliberate opt-out"
        )
    try:
        with path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except Exception as exc:
        raise ValueError(
            f"Shot-dispersion config is unreadable or invalid JSON: {path}"
        ) from exc
    if not isinstance(config, dict) or not isinstance(config.get("enabled"), bool):
        raise ValueError(
            "shot_dispersion_config.json must contain an explicit boolean enabled"
        )
    return config, path


def apply_shot_dispersion_overlay(
    std_w: pd.DataFrame,
    player_names: list[str],
    cat_order: list[str],
    *,
    tourney: str,
    event_id: int,
    dists_path: str | os.PathLike[str],
    config_path: str | os.PathLike[str] | None = None,
) -> pd.DataFrame:
    """Return category stds with the event-scoped shot variance overlay.

    Shot variances are rescaled separately by category so their field-average
    variance equals the current production field average. The configured OOS
    weight is then applied in variance space:

        new_var = (1 - weight) * production_var + weight * scaled_shot_var

    An explicit ``enabled: false`` or a config for another event returns
    ``std_w`` unchanged. Missing, unreadable, or structurally invalid config is
    always fatal so production cannot silently lose this tracked model input.
    """
    config, resolved_config_path = _load_config(config_path)
    disabled_by_env = os.getenv("SHOT_DISPERSION_DISABLE", "").strip().lower()
    if disabled_by_env in {"1", "true", "yes", "on"}:
        print("[shot-dispersion] Disabled by SHOT_DISPERSION_DISABLE")
        return std_w
    if not config["enabled"]:
        print(f"[shot-dispersion] Disabled ({resolved_config_path.name})")
        return std_w

    configured_tourney = str(config.get("tourney", "")).casefold().strip()
    active_tourney = str(tourney).casefold().strip()
    configured_event_id = int(config.get("event_id", -1))
    if active_tourney != configured_tourney or int(event_id) != configured_event_id:
        print(
            "[shot-dispersion] Not active for "
            f"{active_tourney}/event {event_id}; configured for "
            f"{configured_tourney}/event {configured_event_id}"
        )
        return std_w

    expected_cats = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]
    if list(cat_order) != expected_cats:
        raise ValueError(
            f"Shot-dispersion category order must be {expected_cats}; got {list(cat_order)}"
        )

    dists = Path(dists_path)
    if not dists.is_absolute():
        dists = REPO_ROOT / dists
    feature_path = Path(config["feature_file"])
    if not feature_path.is_absolute():
        feature_path = REPO_ROOT / feature_path
    for label, path in (("distribution", dists), ("feature", feature_path)):
        if not path.exists():
            raise FileNotFoundError(f"Shot-dispersion {label} file not found: {path}")

    expected_dists_hash = str(config.get("distribution_sha256", "")).casefold()
    expected_feature_hash = str(config.get("feature_sha256", "")).casefold()
    actual_dists_hash = _sha256(dists)
    actual_feature_hash = _sha256(feature_path)
    if expected_dists_hash and actual_dists_hash != expected_dists_hash:
        raise ValueError(
            "Shot-dispersion distribution hash mismatch: "
            f"expected {expected_dists_hash}, got {actual_dists_hash}"
        )
    if expected_feature_hash and actual_feature_hash != expected_feature_hash:
        raise ValueError(
            "Shot-dispersion feature hash mismatch: "
            f"expected {expected_feature_hash}, got {actual_feature_hash}"
        )

    players = [_normalise_name(player) for player in player_names]
    if len(players) != len(set(players)):
        raise ValueError("Shot-dispersion active player names are not unique")
    expected_field_size = int(config.get("expected_field_size", len(players)))
    if len(players) != expected_field_size:
        raise ValueError(
            f"Shot-dispersion expected {expected_field_size} players, got {len(players)}"
        )

    features = pd.read_csv(feature_path)
    if "player_name" not in features.columns:
        raise ValueError("Shot-dispersion feature file lacks player_name")
    features["player_name"] = features["player_name"].map(_normalise_name)
    if features["player_name"].duplicated().any():
        raise ValueError("Shot-dispersion feature file has duplicate player names")
    features = features.set_index("player_name")
    missing_features = sorted(set(players) - set(features.index))
    if missing_features:
        raise ValueError(
            f"Shot-dispersion feature file is missing {len(missing_features)} players: "
            f"{missing_features}"
        )

    effective = std_w.copy()
    missing_stds = sorted(set(players) - set(effective.index))
    if missing_stds:
        raise ValueError(
            f"Production category dists are missing {len(missing_stds)} players: {missing_stds}"
        )

    weights = config.get("weights", {})
    diagnostics = []
    for cat in cat_order:
        short_cat = cat.removeprefix("sg_")
        shot_col = f"{short_cat}_shot_indep_var50_shrunk"
        if cat not in effective.columns:
            raise ValueError(f"Production category dists lack {cat}")
        if shot_col not in features.columns:
            raise ValueError(f"Shot-dispersion feature file lacks {shot_col}")
        if cat not in weights:
            raise ValueError(f"Shot-dispersion config lacks a weight for {cat}")

        weight = float(weights[cat])
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"Shot-dispersion weight for {cat} is outside [0, 1]: {weight}")

        base_std = effective.loc[players, cat].to_numpy(dtype=float)
        shot_var = features.loc[players, shot_col].to_numpy(dtype=float)
        if not np.isfinite(base_std).all() or np.any(base_std <= 0):
            raise ValueError(f"Production category dists have invalid {cat} standard deviations")
        if not np.isfinite(shot_var).all() or np.any(shot_var <= 0):
            raise ValueError(f"Shot-dispersion features have invalid {shot_col}")

        base_var = base_std**2
        scale = float(base_var.mean() / shot_var.mean())
        scaled_shot_var = shot_var * scale
        new_var = (1.0 - weight) * base_var + weight * scaled_shot_var
        if not np.isfinite(new_var).all() or np.any(new_var <= 0):
            raise ValueError(f"Shot-dispersion produced invalid {cat} variance")
        effective.loc[players, cat] = np.sqrt(new_var)

        preserved_error = float(abs(new_var.mean() - base_var.mean()))
        diagnostics.append(
            f"{short_cat.upper()} w={weight:.2f} scale={scale:.6f} "
            f"std-ratio={np.sqrt(new_var / base_var).min():.3f}-"
            f"{np.sqrt(new_var / base_var).max():.3f} "
            f"mean-var-error={preserved_error:.2e}"
        )

    print(
        "[shot-dispersion] ACTIVE: frozen pre-event 50-round shot variance; "
        "means/correlation/skew unchanged"
    )
    print(f"[shot-dispersion] {' | '.join(diagnostics)}")
    print(
        f"[shot-dispersion] feature={feature_path.name} sha256={actual_feature_hash[:12]}... "
        f"dists={dists.name} sha256={actual_dists_hash[:12]}..."
    )
    return effective
