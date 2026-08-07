"""Shared field-relative score centering for live round simulations.

The Sheet stores an absolute expected score for the active field.  Player
prediction files therefore store only deviations from that field average:
centered skill plus centered tee-time weather.  Keeping those two contracts
separate prevents field strength or common weather from being counted twice.
"""

from __future__ import annotations

import pandas as pd


CENTERING_VERSION = "field_relative_v1"


def expected_field_score(
    base_score,
    field_mean_skill,
    wind_effect=0.0,
    dew_effect=0.0,
    difficulty_feedback=0.0,
):
    """Return the active field's absolute expected scoring average."""
    return (
        float(base_score)
        - float(field_mean_skill)
        + float(wind_effect)
        + float(dew_effect)
        + float(difficulty_feedback)
    )


def _mean_aligned(frame, value_col, group_col=None):
    values = pd.to_numeric(frame[value_col], errors="coerce")
    if group_col:
        if frame.loc[values.notna(), group_col].isna().any():
            raise ValueError(
                f"{group_col} is missing for players with {value_col}"
            )
        return values.groupby(frame[group_col], dropna=False).transform("mean")
    return pd.Series(values.mean(), index=frame.index, dtype=float)


def validate_centered_scores(
    frame,
    score_col,
    group_col=None,
    tolerance=1e-9,
):
    """Raise when a field-relative score column is not zero-mean."""
    if score_col not in frame.columns:
        raise ValueError(f"Missing centered score column: {score_col}")

    scores = pd.to_numeric(frame[score_col], errors="coerce")
    valid = scores.notna()
    if not valid.any():
        raise ValueError(f"{score_col} has no numeric values")

    if group_col:
        if group_col not in frame.columns:
            raise ValueError(f"Missing centering group column: {group_col}")
        if frame.loc[valid, group_col].isna().any():
            raise ValueError(f"{group_col} is missing for centered players")
        means = scores[valid].groupby(
            frame.loc[valid, group_col], dropna=False
        ).mean()
    else:
        means = pd.Series({"field": scores[valid].mean()})

    failures = means[means.abs() > float(tolerance)]
    if not failures.empty:
        detail = ", ".join(
            f"{key}={value:+.6f}" for key, value in failures.items()
        )
        raise ValueError(
            f"{score_col} is not field-centered within tolerance "
            f"{tolerance}: {detail}"
        )
    return means


def validate_field_relative_predictions(
    frame,
    *,
    skill_col,
    score_col,
    weather_col,
    group_col=None,
    tolerance=1e-9,
):
    """Validate the full field-average/player-deviation data contract."""
    required = [skill_col, score_col, weather_col, "field_skill_mean"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing field-relative columns: {missing}")

    validate_centered_scores(
        frame, score_col, group_col=group_col, tolerance=tolerance
    )
    validate_centered_scores(
        frame, weather_col, group_col=group_col, tolerance=tolerance
    )

    skill = pd.to_numeric(frame[skill_col], errors="coerce")
    field_skill = pd.to_numeric(frame["field_skill_mean"], errors="coerce")
    weather = pd.to_numeric(frame[weather_col], errors="coerce")
    score = pd.to_numeric(frame[score_col], errors="coerce")
    if pd.concat([skill, field_skill, weather, score], axis=1).isna().any().any():
        raise ValueError("Field-relative prediction components contain missing values")

    expected_field_skill = _mean_aligned(frame, skill_col, group_col)
    field_error = (field_skill - expected_field_skill).abs().max()
    formula_error = (score - (skill - field_skill + weather)).abs().max()
    if field_error > float(tolerance):
        raise ValueError(
            f"field_skill_mean does not match the active field ({field_error:.6g})"
        )
    if formula_error > float(tolerance):
        raise ValueError(
            f"{score_col} does not equal centered skill plus weather "
            f"({formula_error:.6g})"
        )
    return True


def center_player_advantages(
    frame,
    *,
    skill_col,
    score_col,
    weather_col,
    wind_cost_col=None,
    dew_cost_col=None,
    group_col=None,
):
    """Center skill and weather to deviations from the active field/course."""
    if skill_col not in frame.columns:
        raise ValueError(f"Missing skill column: {skill_col}")

    result = frame.copy()
    use_group = (
        group_col
        if group_col and group_col in result.columns
        and result[group_col].nunique(dropna=True) > 1
        else None
    )

    result[skill_col] = pd.to_numeric(result[skill_col], errors="coerce")
    result["field_skill_mean"] = _mean_aligned(
        result, skill_col, use_group
    )

    if wind_cost_col and wind_cost_col in result.columns:
        result[wind_cost_col] = pd.to_numeric(
            result[wind_cost_col], errors="coerce"
        )
        result["field_wind_cost"] = _mean_aligned(
            result, wind_cost_col, use_group
        )
        wind_relative = result["field_wind_cost"] - result[wind_cost_col]
    else:
        result["field_wind_cost"] = 0.0
        wind_relative = pd.Series(0.0, index=result.index)

    if dew_cost_col and dew_cost_col in result.columns:
        result[dew_cost_col] = pd.to_numeric(
            result[dew_cost_col], errors="coerce"
        )
        result["field_dew_cost"] = _mean_aligned(
            result, dew_cost_col, use_group
        )
        dew_relative = result["field_dew_cost"] - result[dew_cost_col]
    else:
        result["field_dew_cost"] = 0.0
        dew_relative = pd.Series(0.0, index=result.index)

    result[weather_col] = wind_relative + dew_relative
    result[score_col] = (
        result[skill_col] - result["field_skill_mean"]
        + result[weather_col]
    )
    result["centering_version"] = CENTERING_VERSION
    result["centering_group"] = use_group or "field"

    validate_field_relative_predictions(
        result,
        skill_col=skill_col,
        score_col=score_col,
        weather_col=weather_col,
        group_col=use_group,
        tolerance=1e-9,
    )
    return result
