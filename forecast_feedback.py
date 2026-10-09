"""Forecast-error feedback for live round scoring expectations."""

from __future__ import annotations

from collections.abc import Iterable
import math


FEEDBACK_WEIGHTS = {1: 0.5, 2: 0.6, 3: 0.7}


def single_published_forecast(value):
    """Return one absolute published score, or ``None`` when ambiguous."""
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            return None
        value = value[0]
    try:
        forecast = float(value)
    except (TypeError, ValueError):
        return None
    return forecast if math.isfinite(forecast) and forecast > 50 else None


def forecast_feedback(misses: Iterable[float]) -> tuple[float, float]:
    """Return ``(feedback, weight)`` from published forecast misses."""
    values = [float(value) for value in misses]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Forecast misses must be finite")
    if not values:
        return 0.0, 0.0
    weight = FEEDBACK_WEIGHTS.get(len(values), 0.7)
    return round(weight * (sum(values) / len(values)), 3), weight


def round_scoring_result(
    *,
    published_forecast,
    actual_score,
    base_score,
    field_adjustment,
    wind_impact,
    dew_impact,
):
    """Calculate forecast accuracy plus the realized-weather diagnostic."""
    diagnostics = [base_score, field_adjustment, wind_impact, dew_impact]
    structural_baseline = (
        sum(float(value) for value in diagnostics)
        if all(value is not None for value in diagnostics) else None
    )
    published = single_published_forecast(published_forecast)
    actual = None if actual_score is None else float(actual_score)
    return {
        "published_forecast": published,
        "actual_score": actual,
        "forecast_miss": (
            actual - published
            if actual is not None and published is not None
            else None
        ),
        "structural_baseline": structural_baseline,
        "structural_residual": (
            actual - structural_baseline
            if actual is not None and structural_baseline is not None else None
        ),
    }
