"""Safe prediction recovery from the last completed simulation artifact."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _canonical_name(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def backfill_missing_predictions(
    frame,
    *,
    player_col,
    prediction_col,
    artifact_path,
    expected_event_ids,
    expected_tourney,
):
    """Fill missing predictions from a matching ``sim_fairs.json`` artifact.

    The event and tournament checks keep a stale artifact from another weekly
    slate from contaminating the active field. Existing predictions are never
    overwritten. Missing or malformed optional artifacts fail open.
    """
    result = frame.copy()
    if player_col not in result.columns or prediction_col not in result.columns:
        return result, []

    current = pd.to_numeric(result[prediction_col], errors="coerce")
    missing = current.isna()
    if not missing.any():
        result[prediction_col] = current
        return result, []

    try:
        payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return result, []

    expected_ids = {str(event_id).strip() for event_id in expected_event_ids}
    artifact_event_id = str(payload.get("event_id", "")).strip()
    artifact_tourney = str(payload.get("tourney", "")).strip().lower()
    if (
        artifact_event_id not in expected_ids
        or artifact_tourney != str(expected_tourney).strip().lower()
    ):
        return result, []

    raw_predictions = payload.get("pred")
    if not isinstance(raw_predictions, dict):
        return result, []

    fallback = pd.Series(
        {
            _canonical_name(player): pd.to_numeric(value, errors="coerce")
            for player, value in raw_predictions.items()
        },
        dtype=float,
    )
    names = result[player_col].map(_canonical_name)
    fallback_values = names.map(fallback)
    fillable = missing & fallback_values.notna()
    if not fillable.any():
        return result, []

    result[prediction_col] = current
    result.loc[fillable, prediction_col] = fallback_values.loc[fillable]
    filled_players = result.loc[fillable, player_col].astype(str).tolist()
    return result, filled_players
