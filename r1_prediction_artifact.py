"""Validation and identity helpers for the complete R1 prediction artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


REQUIRED_R1_COLUMNS = ("player_name", "my_pred", "wind_adj1", "dew_adj1")


def canonical_player_name(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def canonical_player_set(values):
    return {
        name
        for name in (canonical_player_name(value) for value in values)
        if name
    }


def player_field_signature(values):
    """Stable digest that detects same-size player swaps."""
    players = sorted(canonical_player_set(values))
    return hashlib.sha256("\n".join(players).encode("utf-8")).hexdigest()


def manifest_path_for(csv_path):
    return Path(csv_path).with_suffix(".meta.json")


def validate_r1_prediction_frame(frame, *, active_players=None):
    """Require a complete, unique artifact covering every active player.

    Extra prediction rows are allowed because a late withdrawal can remain in
    the pre-event snapshot. Missing active players are never allowed: that is
    the dangerous late-add/same-size-swap case that previously failed later in
    score centering with no useful explanation.
    """
    missing_columns = [column for column in REQUIRED_R1_COLUMNS if column not in frame]
    if missing_columns:
        raise ValueError(f"missing required column(s): {', '.join(missing_columns)}")
    if len(frame) < 10:
        raise ValueError(f"artifact is too small ({len(frame)} rows)")

    names = frame["player_name"].map(canonical_player_name)
    blank_names = int((names == "").sum())
    if blank_names:
        raise ValueError(f"artifact contains {blank_names} blank player name(s)")
    duplicates = sorted(names[names.duplicated(keep=False)].unique())
    if duplicates:
        raise ValueError(f"artifact contains duplicate player(s): {', '.join(duplicates)}")

    incomplete = frame[list(REQUIRED_R1_COLUMNS)].isna().any(axis=1)
    if incomplete.any():
        players = sorted(names[incomplete].tolist())
        raise ValueError(
            "artifact contains missing prediction/weather values for: "
            + ", ".join(players)
        )

    prediction_players = set(names)
    active_values = [] if active_players is None else active_players
    active = canonical_player_set(active_values)
    missing_active = sorted(active - prediction_players)
    if missing_active:
        raise ValueError(
            "artifact is missing active player(s): " + ", ".join(missing_active)
        )
    return {
        "prediction_players": prediction_players,
        "active_players": active,
        "extra_players": sorted(prediction_players - active) if active else [],
    }


def build_r1_prediction_manifest(frame, payload):
    """Build the sidecar committed atomically with the R1 prediction CSV."""
    active_field = payload.get("field", [])
    details = validate_r1_prediction_frame(frame, active_players=active_field)
    return {
        "schema_version": 1,
        "event_id": str(payload.get("event_id", "")).strip(),
        "tourney": str(payload.get("tourney", "")).strip().lower(),
        "sim_run_at": payload.get("sim_run_at"),
        "prediction_count": len(details["prediction_players"]),
        "active_field_count": len(details["active_players"]),
        "prediction_field_sha256": player_field_signature(frame["player_name"]),
        "active_field_sha256": player_field_signature(active_field),
        "withdrawn_or_extra_players": details["extra_players"],
    }


def validate_r1_prediction_manifest(
    frame,
    manifest,
    *,
    expected_event_ids,
    expected_tourney,
):
    """Confirm the published CSV and its event identity still match."""
    expected_ids = {str(event_id).strip() for event_id in expected_event_ids}
    event_id = str(manifest.get("event_id", "")).strip()
    artifact_tourney = str(manifest.get("tourney", "")).strip().lower()
    if event_id not in expected_ids:
        raise ValueError(
            f"manifest event {event_id!r} does not match {sorted(expected_ids)}"
        )
    if artifact_tourney != str(expected_tourney).strip().lower():
        raise ValueError(
            f"manifest tournament {artifact_tourney!r} does not match "
            f"{expected_tourney!r}"
        )
    actual_signature = player_field_signature(frame["player_name"])
    if manifest.get("prediction_field_sha256") != actual_signature:
        raise ValueError("CSV player field does not match its published manifest")
    if int(manifest.get("prediction_count", -1)) != len(frame):
        raise ValueError("CSV row count does not match its published manifest")


def load_matching_r1_predictions(
    candidates,
    *,
    active_players,
    expected_event_ids,
    expected_tourney,
):
    """Load the first complete local/published snapshot covering the live field."""
    failures = []
    for candidate in (Path(path) for path in candidates):
        if not candidate.is_file():
            failures.append(f"{candidate}: not found")
            continue
        try:
            frame = pd.read_csv(candidate)
            details = validate_r1_prediction_frame(
                frame,
                active_players=active_players,
            )
            manifest_path = manifest_path_for(candidate)
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                validate_r1_prediction_manifest(
                    frame,
                    manifest,
                    expected_event_ids=expected_event_ids,
                    expected_tourney=expected_tourney,
                )
            return frame, candidate, details
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            failures.append(f"{candidate}: {exc}")

    raise ValueError(
        "No complete R1 prediction snapshot covers the active field. "
        "This usually means a late field change was not included in the last "
        "published pre-event run. " + " | ".join(failures)
    )
