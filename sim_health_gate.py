"""Fail-closed provenance and health gate for betting simulation artifacts.

The betting paths deliberately do not infer trust from a filename or a recent
mtime.  A round simulation writes a content-addressed health manifest; derived
artifacts bind that manifest to the exact bytes they publish.  Every path that
emails, stores, or Telegram-alerts bets must call one of the ``require_*``
functions immediately before the side effect.

This module is intentionally independent of Google Sheets and ``round_sim`` so
the invariant checks are cheap to exercise in CI.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from portable_hash import (
    LF_NORMALIZED_HASH_MODE,
    lf_normalized_bytes,
    lf_normalized_sha256,
)


SCHEMA_VERSION = 1
GATE_VERSION = "round-bet-health/v1"
DEFAULT_MAX_AGE_HOURS = 18.0
MIN_SIMULATIONS = 10_000
MIN_FIELD_PLAYERS = 20
EXPECTED_AVG_RANGE = (55.0, 90.0)
SCORE_RANGE = (40.0, 110.0)
CENTERING_TOLERANCE_STROKES = 0.12
PLAYER_CENTERING_MAX_TOLERANCE = 0.16
PLAYER_CENTERING_RMSE_TOLERANCE = 0.08
SAFE_OVERLAY_STATUSES = frozenset({
    "active",
    "not_applicable_to_event",
    "disabled_in_config",
})


def _overlay_status_is_unsafe(overlay: Mapping[str, Any]) -> bool:
    return overlay.get("status") not in SAFE_OVERLAY_STATUSES


class SimulationHealthError(RuntimeError):
    """Raised before any betting side effect when an artifact is not trusted."""


@dataclass
class HealthReport:
    ok: bool
    manifest_id: str | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        state = "PASS" if self.ok else "BLOCKED"
        ident = f" manifest={self.manifest_id[:12]}" if self.manifest_id else ""
        detail = "; ".join(self.errors or self.warnings[:3])
        return f"{state}{ident}" + (f" — {detail}" if detail else "")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_stamp(value: datetime | None = None) -> str:
    value = value or utc_now()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def parse_utc(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    for fmt in (
        "%Y-%m-%d %H:%M:%S UTC",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            parsed = datetime.strptime(text, fmt)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            continue
    return None


def _normalise_name(value: Any) -> str:
    return str(value).casefold().strip()


def configured_round_expected_avg(config: Mapping[str, Any]) -> float:
    """Resolve the Sheet's legacy adjustment-or-full-score convention."""
    raw = config.get("expected_score_1")
    if raw is None:
        raise SimulationHealthError("active config has no expected_score_1")
    value = float(raw)
    if abs(value) > 50:
        return value
    pars = config.get("course_pars") or []
    if not pars:
        raise SimulationHealthError(
            "cannot resolve expected_score_1 adjustment without course_pars"
        )
    return float(pars[0]) + value


def configured_round_scoring_baselines(config: Mapping[str, Any]) -> dict[str, float]:
    """Return the Sheet-authoritative absolute baseline for every active course.

    ``expected_score_1`` is the single-course field average on ordinary weeks.
    Multi-course weeks must provide a one-to-one ``course_codes`` mapping; an
    incomplete/stale mapping is unsafe because a prediction file could otherwise
    carry last week's per-course baseline while the headline average looks current.
    """
    raw_values = [config.get(f"expected_score_{idx}") for idx in range(1, 4)]
    populated = [idx for idx, value in enumerate(raw_values) if value is not None]
    if populated and populated != list(range(max(populated) + 1)):
        raise SimulationHealthError(
            "multi-course expected_score_N values contain a gap"
        )
    values = [value for value in raw_values if value is not None]
    if not values:
        raise SimulationHealthError("active config has no course scoring baselines")

    pars = list(config.get("course_pars") or [])

    def _absolute(value: Any, index: int) -> float:
        number = float(value)
        if abs(number) <= 50:
            if not pars:
                raise SimulationHealthError(
                    "cannot resolve course scoring adjustment without course_pars"
                )
            par = float(pars[index] if index < len(pars) else pars[0])
            number = par + number
        if not math.isfinite(number) or not EXPECTED_AVG_RANGE[0] <= number <= EXPECTED_AVG_RANGE[1]:
            raise SimulationHealthError(
                f"configured course scoring baseline {number!r} is outside "
                f"{EXPECTED_AVG_RANGE[0]:g}..{EXPECTED_AVG_RANGE[1]:g}"
            )
        return number

    absolute = [_absolute(value, idx) for idx, value in enumerate(values)]
    course_codes = [str(code).casefold().strip() for code in (config.get("course_codes") or [])]
    if len(absolute) == 1:
        return {"field": absolute[0]}
    if len(course_codes) != len(absolute) or len(set(course_codes)) != len(course_codes):
        raise SimulationHealthError(
            "multi-course scoring config requires unique course_codes matching every "
            "expected_score_N value"
        )
    return dict(zip(course_codes, absolute))


def derive_authoritative_scoring_targets(
    frame: Any,
    *,
    sim_round: int,
    skill_col: str,
    configured_expected_avg: float,
    course_averages: Mapping[str, Any],
) -> tuple[float, dict[str, float]]:
    """Derive per-player score means from current Sheet baselines, not CSV labels.

    This is deliberately independent of ``course_score_adj``. That column is an
    output of ``live_stats_engine`` and is checked against the Sheet, but is never
    allowed to define its own health target. Otherwise a stale 69.7 column could
    generate a 69.7 tape, label it 68.7, and approve itself.
    """
    import pandas as pd
    from score_centering import CENTERING_VERSION, validate_field_relative_predictions

    if frame is None or getattr(frame, "empty", True):
        raise SimulationHealthError("model prediction field is empty")
    score_col = f"scores_r{int(sim_round)}"
    weather_col = f"weather_sg_r{int(sim_round)}"
    required = {"player_name", skill_col, score_col, weather_col, "centering_version", "centering_group"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SimulationHealthError(
            f"prediction artifact lacks required field-relative columns: {missing}"
        )
    if frame[["centering_version", "centering_group"]].isna().any().any():
        raise SimulationHealthError(
            f"prediction artifact is not exclusively {CENTERING_VERSION}: "
            "centering metadata is partially missing"
        )
    versions = set(frame["centering_version"].dropna().astype(str))
    if versions != {CENTERING_VERSION}:
        raise SimulationHealthError(
            f"prediction artifact is not exclusively {CENTERING_VERSION}: {sorted(versions)}"
        )
    groups = set(frame["centering_group"].dropna().astype(str))
    if len(groups) != 1:
        raise SimulationHealthError(f"prediction artifact has inconsistent centering groups: {groups}")
    group_col = next(iter(groups))
    if group_col == "field":
        group_col = None
    try:
        validate_field_relative_predictions(
            frame,
            skill_col=skill_col,
            score_col=score_col,
            weather_col=weather_col,
            group_col=group_col,
        )
    except ValueError as exc:
        raise SimulationHealthError(f"invalid field-relative prediction artifact: {exc}") from exc

    normalised_courses = {
        str(code).casefold().strip(): float(value)
        for code, value in course_averages.items()
    }
    if not normalised_courses:
        raise SimulationHealthError("current Sheet has no authoritative scoring baseline")
    for code, value in normalised_courses.items():
        if not math.isfinite(value) or not EXPECTED_AVG_RANGE[0] <= value <= EXPECTED_AVG_RANGE[1]:
            raise SimulationHealthError(f"invalid scoring baseline for {code!r}: {value!r}")

    if set(normalised_courses) == {"field"}:
        baseline = pd.Series(normalised_courses["field"], index=frame.index, dtype=float)
        if abs(float(configured_expected_avg) - normalised_courses["field"]) > 1e-6:
            raise SimulationHealthError(
                "primary expected average disagrees with the single-course Sheet baseline"
            )
        if "course_score_adj" in frame.columns:
            labelled = pd.to_numeric(frame["course_score_adj"], errors="coerce")
            labelled = labelled[labelled.notna()]
            if not labelled.empty and float((labelled - baseline.loc[labelled.index]).abs().max()) > 1e-6:
                raise SimulationHealthError(
                    "prediction course_score_adj disagrees with the current Sheet baseline"
                )
    else:
        course_col = "course" if "course" in frame.columns else "course_x" if "course_x" in frame.columns else None
        if course_col is None:
            raise SimulationHealthError(
                "multi-course prediction artifact has no player course assignment"
            )
        course_keys = frame[course_col].astype(str).str.casefold().str.strip()
        baseline = course_keys.map(normalised_courses)
        if baseline.isna().any():
            unknown = sorted(set(course_keys[baseline.isna()]))
            raise SimulationHealthError(
                f"prediction artifact contains unmapped course codes: {unknown}"
            )
        if "course_score_adj" not in frame.columns:
            raise SimulationHealthError("multi-course prediction artifact lacks course_score_adj")
        labelled = pd.to_numeric(frame["course_score_adj"], errors="coerce")
        if labelled.isna().any() or float((labelled - baseline).abs().max()) > 1e-6:
            raise SimulationHealthError(
                "prediction course_score_adj disagrees with current per-course Sheet baselines"
            )

    advantages = pd.to_numeric(frame[score_col], errors="coerce")
    if advantages.isna().any():
        raise SimulationHealthError(f"{score_col} contains missing/non-numeric values")
    player_means = baseline.astype(float) - advantages.astype(float)
    names = frame["player_name"].map(_normalise_name)
    if names.duplicated().any():
        raise SimulationHealthError("prediction artifact has duplicate normalised players")
    return float(player_means.mean()), dict(zip(names, player_means.astype(float)))


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _content_id(value: Mapping[str, Any]) -> str:
    unsigned = {k: v for k, v in value.items() if k != "manifest_sha256"}
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def seal_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    sealed = copy.deepcopy(dict(manifest))
    sealed.pop("manifest_sha256", None)
    sealed["manifest_sha256"] = _content_id(sealed)
    return sealed


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _legacy_crlf_text_sha256(path: str | os.PathLike[str]) -> str:
    """Return the old Windows checkout hash for a normalized text file."""
    crlf_bytes = lf_normalized_bytes(path).replace(b"\n", b"\r\n")
    return hashlib.sha256(crlf_bytes).hexdigest()


def names_sha256(names: Iterable[Any]) -> str:
    normalised = sorted(_normalise_name(name) for name in names)
    return hashlib.sha256(_canonical_json(normalised)).hexdigest()


def tape_sha256(sim_dict: Mapping[Any, Any]) -> str:
    """Stable semantic hash for a player x simulation score tape.

    Values are normalised to little-endian float64 so a parquet integer-width
    round-trip cannot produce a false mismatch, while even a fractional
    score-est shift necessarily changes the digest.
    """
    digest = hashlib.sha256()
    for raw_name in sorted(sim_dict, key=_normalise_name):
        name = _normalise_name(raw_name)
        values = np.asarray(sim_dict[raw_name], dtype="<f8")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def sealed_cache_expected_avg(cache_meta: Mapping[str, Any] | None) -> float:
    """Resolve a price-only shift anchor exclusively from the sealed manifest."""
    meta = dict(cache_meta or {})
    manifest = meta.get("health_manifest") or {}
    if not manifest or manifest.get("manifest_sha256") != _content_id(manifest):
        raise SimulationHealthError(
            "sim cache has no valid sealed health manifest for expected_avg"
        )
    try:
        sealed = float((manifest.get("scoring") or {}).get("expected_avg"))
    except (TypeError, ValueError):
        raise SimulationHealthError("sealed sim cache expected_avg is missing or invalid")
    if not math.isfinite(sealed):
        raise SimulationHealthError("sealed sim cache expected_avg is missing or invalid")
    if meta.get("expected_avg") is not None:
        try:
            unsealed = float(meta["expected_avg"])
        except (TypeError, ValueError):
            raise SimulationHealthError(
                "cache expected_avg metadata disagrees with its sealed manifest"
            )
        if not math.isfinite(unsealed) or abs(unsealed - sealed) > 1e-6:
            raise SimulationHealthError(
                "cache expected_avg metadata disagrees with its sealed manifest"
            )
    return sealed


def _source_file(path: str | os.PathLike[str] | None) -> dict[str, Any] | None:
    if not path:
        return None
    resolved = Path(path)
    if not resolved.is_file():
        return {"path": resolved.name, "exists": False}
    return {
        "path": resolved.name,
        "exists": True,
        "sha256": file_sha256(resolved),
        "modified_at": utc_stamp(datetime.fromtimestamp(resolved.stat().st_mtime, timezone.utc)),
    }


def collect_overlay_provenance(
    *,
    tourney: str,
    event_id: int | str,
    dists_path: str | os.PathLike[str] | None,
    selected_model: str,
    config_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Describe the overlay policy that produced the selected score tape.

    The overlay implementation itself is fail-closed when it is in scope.  This
    record makes that choice durable so a cache cannot later be repriced under a
    different config or with ``SHOT_DISPERSION_DISABLE`` silently set.
    """
    root = Path(__file__).resolve().parent
    config = Path(config_path) if config_path else root / "shot_dispersion_config.json"
    if not config.is_absolute():
        config = root / config
    payload: dict[str, Any] = {}
    if config.is_file():
        try:
            payload = json.loads(config.read_text(encoding="utf-8"))
        except Exception as exc:
            return {
                "status": "invalid_config",
                "used_by_selected_tape": False,
                "config_path": config.name,
                "error": str(exc),
            }
        if not isinstance(payload, dict) or not isinstance(payload.get("enabled"), bool):
            return {
                "status": "invalid_config",
                "used_by_selected_tape": False,
                "config_path": config.name,
                "error": "config must contain an explicit boolean enabled",
            }
        if (
            "required_current_event" in payload
            and not isinstance(payload["required_current_event"], bool)
        ):
            return {
                "status": "invalid_config",
                "used_by_selected_tape": False,
                "config_path": config.name,
                "error": "required_current_event must be boolean",
            }

    required_current_event = bool(
        payload.get("enabled", False)
        and payload.get("required_current_event", False)
    )
    if required_current_event:
        missing = [
            key
            for key in (
                "tourney",
                "event_id",
                "feature_file",
                "feature_sha256",
                "distribution_sha256",
            )
            if payload.get(key) in (None, "")
        ]
        if missing:
            return {
                "status": "invalid_config",
                "used_by_selected_tape": False,
                "config_path": config.name,
                "required_current_event": True,
                "error": "required weekly config lacks: " + ", ".join(missing),
            }

    active_event = (
        bool(payload.get("enabled", False))
        and _normalise_name(payload.get("tourney")) == _normalise_name(tourney)
        and str(payload.get("event_id")) == str(event_id)
    )
    env_disabled = os.getenv("SHOT_DISPERSION_DISABLE", "").strip().casefold() in {
        "1", "true", "yes", "on",
    }
    category_first = selected_model == "category_first"
    used = bool(active_event and not env_disabled and category_first)
    if not payload:
        status = "config_absent"
    elif required_current_event and not active_event:
        status = "required_event_mismatch"
    elif env_disabled and active_event and required_current_event:
        status = "required_disabled_by_environment"
    elif env_disabled and active_event:
        status = "disabled_by_environment"
    elif active_event and not category_first:
        status = "configured_but_not_selected"
    elif active_event:
        status = "active"
    elif payload.get("enabled", False):
        status = "not_applicable_to_event"
    else:
        status = "disabled_in_config"

    feature = Path(str(payload.get("feature_file", ""))) if payload.get("feature_file") else None
    if feature is not None and not feature.is_absolute():
        feature = root / feature
    dists = Path(dists_path) if dists_path else None
    if dists is not None and not dists.is_absolute():
        dists = root / dists
    return {
        "status": status,
        "required_current_event": required_current_event,
        "configured_for_active_event": active_event,
        "used_by_selected_tape": used,
        "config_path": config.name,
        "config_sha256": (
            lf_normalized_sha256(config) if config.is_file() else None
        ),
        # Manifests written before the config hash joined the portable text
        # contract used raw checkout bytes. Keep the deterministic CRLF variant
        # so a Windows-produced tape can be safely repriced on Linux.
        "config_sha256_crlf_legacy": (
            _legacy_crlf_text_sha256(config) if config.is_file() else None
        ),
        "feature_file": feature.name if feature else None,
        "feature_sha256": (
            lf_normalized_sha256(feature)
            if feature and feature.is_file()
            else None
        ),
        "distribution_file": dists.name if dists else None,
        "distribution_sha256": (
            lf_normalized_sha256(dists)
            if dists and dists.is_file()
            else None
        ),
        "text_hash_mode": LF_NORMALIZED_HASH_MODE,
        "configured_feature_sha256": payload.get("feature_sha256"),
        "configured_distribution_sha256": payload.get("distribution_sha256"),
        "weights": payload.get("weights") or {},
    }


def _inspect_tape(sim_dict: Mapping[Any, Any], model_players: Iterable[Any]) -> tuple[dict, list[str]]:
    errors: list[str] = []
    tape_names = [_normalise_name(name) for name in sim_dict]
    model_names = [_normalise_name(name) for name in model_players]
    if len(tape_names) != len(set(tape_names)):
        errors.append("simulation tape has duplicate normalised player names")
    if len(model_names) != len(set(model_names)):
        errors.append("model field has duplicate normalised player names")
    tape_set, model_set = set(tape_names), set(model_names)
    missing = sorted(model_set - tape_set)
    extra = sorted(tape_set - model_set)
    if missing or extra:
        errors.append(
            f"tape/model field mismatch (missing={len(missing)}, extra={len(extra)})"
        )

    lengths: set[int] = set()
    finite = True
    score_min = math.inf
    score_max = -math.inf
    score_sum = 0.0
    score_count = 0
    for values in sim_dict.values():
        arr = np.asarray(values)
        if arr.ndim != 1:
            errors.append("simulation tape arrays must be one-dimensional")
            continue
        lengths.add(int(arr.size))
        try:
            numeric = arr.astype(float)
        except (TypeError, ValueError):
            finite = False
            continue
        if numeric.size:
            finite = finite and bool(np.isfinite(numeric).all())
            if np.isfinite(numeric).any():
                score_min = min(score_min, float(np.nanmin(numeric)))
                score_max = max(score_max, float(np.nanmax(numeric)))
                score_sum += float(np.nansum(numeric))
                score_count += int(np.isfinite(numeric).sum())
    if len(lengths) != 1:
        errors.append(f"simulation arrays are not aligned (lengths={sorted(lengths)})")
    num_sims = next(iter(lengths), 0)
    if num_sims < MIN_SIMULATIONS:
        errors.append(f"simulation count {num_sims} is below {MIN_SIMULATIONS}")
    if len(tape_names) < MIN_FIELD_PLAYERS:
        errors.append(f"field contains only {len(tape_names)} players")
    if not finite:
        errors.append("simulation tape contains non-finite/non-numeric scores")
    if math.isinf(score_min) or score_min < SCORE_RANGE[0] or score_max > SCORE_RANGE[1]:
        errors.append(
            f"simulated score range {score_min:g}..{score_max:g} is outside "
            f"{SCORE_RANGE[0]:g}..{SCORE_RANGE[1]:g}"
        )
    coverage = (len(tape_set & model_set) / len(model_set)) if model_set else 0.0
    facts = {
        "num_players": len(tape_names),
        "num_model_players": len(model_names),
        "num_sims": num_sims,
        "field_coverage": round(float(coverage), 8),
        "missing_players": missing,
        "extra_players": extra,
        "player_set_sha256": names_sha256(tape_names),
        "tape_sha256": tape_sha256(sim_dict),
        "score_min": None if math.isinf(score_min) else score_min,
        "score_max": None if math.isinf(score_max) else score_max,
        "empirical_field_score_mean": (score_sum / score_count) if score_count else None,
        # Every empirical player PMF contains exactly num_sims / num_sims mass.
        "player_probability_mass_min": 1.0 if num_sims else 0.0,
        "player_probability_mass_max": 1.0 if num_sims else 0.0,
    }
    return facts, errors


def build_simulation_manifest(
    sim_dict: Mapping[Any, Any],
    *,
    tourney: str,
    event_id: int | str,
    sim_round: int,
    expected_avg: float,
    model_players: Iterable[Any],
    expected_avg_authority: str,
    expected_field_mean: float | None = None,
    expected_player_means: Mapping[Any, Any] | None = None,
    configured_course_averages: Mapping[str, Any] | None = None,
    selected_model: str = "category_first",
    skew_calibrated: bool = True,
    overlay: Mapping[str, Any] | None = None,
    prediction_path: str | os.PathLike[str] | None = None,
    generated_at: datetime | None = None,
    manual_approved_by: str | None = None,
    manual_approval_required: bool | None = None,
    parent_manifest: Mapping[str, Any] | None = None,
    derivation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and content-seal the health record for an exact score tape."""
    generated_at = generated_at or utc_now()
    parent_source = (parent_manifest or {}).get("source") or {}
    root_generated_at = (
        parent_source.get("root_generated_at")
        or parent_source.get("generated_at")
        or utc_stamp(generated_at)
    )
    facts, errors = _inspect_tape(sim_dict, model_players)
    try:
        expected = float(expected_avg)
    except (TypeError, ValueError):
        expected = math.nan
    if not math.isfinite(expected) or not EXPECTED_AVG_RANGE[0] <= expected <= EXPECTED_AVG_RANGE[1]:
        errors.append(
            f"expected scoring average {expected_avg!r} is outside "
            f"{EXPECTED_AVG_RANGE[0]:g}..{EXPECTED_AVG_RANGE[1]:g}"
        )
    if parent_manifest:
        derivation_record = dict(derivation or {})
        try:
            derivation_from = float(derivation_record.get("from_expected_avg"))
            derivation_to = float(derivation_record.get("to_expected_avg"))
            derivation_delta = float(derivation_record.get("delta"))
            if derivation_record.get("method") != "uniform_rounding_bin_v1":
                errors.append("derived simulation uses an unsupported fractional method")
            if (
                not all(math.isfinite(value) for value in (
                    derivation_from, derivation_to, derivation_delta
                ))
                or abs(derivation_to - expected) > 1e-6
                or abs((derivation_to - derivation_from) - derivation_delta) > 1e-6
            ):
                errors.append("derived simulation baseline/delta provenance is inconsistent")
        except (TypeError, ValueError):
            errors.append("derived simulation provenance is missing or invalid")
    course_averages: dict[str, float] = {}
    for raw_code, raw_value in (
        configured_course_averages or {"field": expected}
    ).items():
        code = str(raw_code).casefold().strip()
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            errors.append(f"course scoring baseline for {code!r} is non-numeric")
            continue
        if not code or not math.isfinite(value) or not EXPECTED_AVG_RANGE[0] <= value <= EXPECTED_AVG_RANGE[1]:
            errors.append(f"course scoring baseline for {code!r} is invalid: {raw_value!r}")
            continue
        course_averages[code] = value
    if not course_averages:
        errors.append("authoritative course scoring baselines are missing")
    if set(course_averages) == {"field"} and math.isfinite(expected):
        if abs(course_averages["field"] - expected) > 1e-6:
            errors.append("single-course baseline disagrees with expected scoring average")
    try:
        centering_target = float(expected if expected_field_mean is None else expected_field_mean)
    except (TypeError, ValueError):
        centering_target = math.nan
    empirical_mean = facts.get("empirical_field_score_mean")
    centering_delta = (
        float(empirical_mean) - centering_target
        if empirical_mean is not None and math.isfinite(centering_target)
        else math.nan
    )
    if not math.isfinite(centering_target):
        errors.append("implied field scoring mean is missing or invalid")
    elif not math.isfinite(centering_delta) or abs(centering_delta) > CENTERING_TOLERANCE_STROKES:
        errors.append(
            f"empirical tape mean {empirical_mean!r} is not centered on implied "
            f"field mean {centering_target:.4f} (delta={centering_delta:+.4f}, "
            f"tolerance={CENTERING_TOLERANCE_STROKES:.2f})"
        )

    normalised_expected = {
        _normalise_name(name): round(float(value), 8)
        for name, value in (expected_player_means or {}).items()
    }
    if not normalised_expected:
        normalised_expected = {
            _normalise_name(name): centering_target for name in sim_dict
        }
    tape_names = {_normalise_name(name) for name in sim_dict}
    if set(normalised_expected) != tape_names:
        errors.append("per-player centering inputs do not align with simulation field")
    player_deltas: list[float] = []
    for raw_name, values in sim_dict.items():
        name = _normalise_name(raw_name)
        expected_player = normalised_expected.get(name)
        if expected_player is None or not math.isfinite(expected_player):
            continue
        empirical_player = float(np.asarray(values, dtype=float).mean())
        player_deltas.append(empirical_player - expected_player)
    player_rmse = (
        float(np.sqrt(np.mean(np.square(player_deltas)))) if player_deltas else math.nan
    )
    player_max_error = max((abs(delta) for delta in player_deltas), default=math.nan)
    expected_player_field_mean = (
        float(np.mean(list(normalised_expected.values())))
        if normalised_expected else math.nan
    )
    if math.isfinite(expected_player_field_mean) and abs(
        expected_player_field_mean - centering_target
    ) > 1e-6:
        errors.append("aggregate centering target disagrees with per-player inputs")
    if set(course_averages) == {"field"} and math.isfinite(centering_target):
        if abs(centering_target - course_averages["field"]) > 1e-6:
            errors.append(
                "single-course behavioral target is not anchored to the current Sheet baseline"
            )
    if not math.isfinite(player_rmse) or player_rmse > PLAYER_CENTERING_RMSE_TOLERANCE:
        errors.append(
            f"per-player tape centering RMSE {player_rmse:.4f} exceeds "
            f"{PLAYER_CENTERING_RMSE_TOLERANCE:.2f} strokes"
        )
    if not math.isfinite(player_max_error) or player_max_error > PLAYER_CENTERING_MAX_TOLERANCE:
        errors.append(
            f"per-player tape centering max error {player_max_error:.4f} exceeds "
            f"{PLAYER_CENTERING_MAX_TOLERANCE:.2f} strokes"
        )
    if not 1 <= int(sim_round) <= 4:
        errors.append(f"invalid simulation round {sim_round!r}")
    if not str(tourney).strip() or str(event_id).strip() in {"", "0", "None"}:
        errors.append("event identity is incomplete")
    if selected_model != "category_first":
        errors.append(f"non-production score model selected: {selected_model}")
    if not skew_calibrated:
        errors.append("production round-score skew calibration was disabled")

    overlay_record = dict(overlay or {})
    if _overlay_status_is_unsafe(overlay_record):
        errors.append(f"shot-dispersion overlay provenance is unsafe: {overlay_record.get('status')}")
    if overlay_record.get("configured_for_active_event") and not overlay_record.get("used_by_selected_tape"):
        errors.append("active-event shot-dispersion overlay was not used by the selected tape")
    if overlay_record.get("configured_for_active_event"):
        for actual_key, expected_key in (
            ("feature_sha256", "configured_feature_sha256"),
            ("distribution_sha256", "configured_distribution_sha256"),
        ):
            configured = overlay_record.get(expected_key)
            if configured and overlay_record.get(actual_key) != configured:
                errors.append(f"shot-dispersion {actual_key} does not match frozen config")

    valid_authorities = {"sheet", "sheet_score_est", "cli"}
    if expected_avg_authority not in valid_authorities:
        errors.append(f"untrusted expected-average authority: {expected_avg_authority!r}")
    requires_manual = (
        expected_avg_authority == "cli"
        if manual_approval_required is None
        else bool(manual_approval_required)
    )
    if requires_manual and not str(manual_approved_by or "").strip():
        errors.append("CLI-configured simulation lacks an explicit health approver")
    approval_mode = "manual_cli" if requires_manual else "automatic_sheet"
    approval = {
        "status": "approved" if not errors else "rejected",
        "mode": approval_mode,
        "approved_by": (str(manual_approved_by).strip() if manual_approved_by else GATE_VERSION),
        "approved_at": utc_stamp(generated_at) if not errors else None,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "gate_version": GATE_VERSION,
        "kind": "round_simulation",
        "event": {
            "tourney": _normalise_name(tourney),
            "event_id": str(event_id),
            "round": int(sim_round),
        },
        "source": {
            "generated_at": utc_stamp(generated_at),
            # Derived decimal reprices receive a fresh publish generation so
            # consumers refresh, while retaining the root tape clock so a chain
            # of reprices can never extend an old Monte Carlo cache indefinitely.
            "root_generated_at": root_generated_at,
            "expected_avg_authority": expected_avg_authority,
            "prediction_artifact": _source_file(prediction_path),
            "parent_manifest_sha256": (parent_manifest or {}).get("manifest_sha256"),
            "derivation": copy.deepcopy(dict(derivation or {})),
        },
        "scoring": {
            "expected_avg": expected,
            "configured_course_averages": {
                code: round(value, 8) for code, value in sorted(course_averages.items())
            },
            "implied_field_mean": centering_target,
            "empirical_field_mean": empirical_mean,
            "centering_delta": centering_delta,
            "centering_tolerance": CENTERING_TOLERANCE_STROKES,
            "player_expected_means": {
                name: round(value, 8) for name, value in sorted(normalised_expected.items())
            },
            "player_expected_means_sha256": hashlib.sha256(
                _canonical_json({
                    name: round(value, 8) for name, value in sorted(normalised_expected.items())
                })
            ).hexdigest(),
            "player_centering_rmse": player_rmse,
            "player_centering_max_error": player_max_error,
            "player_centering_rmse_tolerance": PLAYER_CENTERING_RMSE_TOLERANCE,
            "player_centering_max_tolerance": PLAYER_CENTERING_MAX_TOLERANCE,
        },
        "simulation": facts,
        "model": {
            "selected": selected_model,
            "skew_calibrated": bool(skew_calibrated),
            "shot_dispersion_overlay": overlay_record,
        },
        "checks": {"errors": errors, "passed": not errors},
        "approval": approval,
    }
    return seal_manifest(manifest)


def validate_simulation_manifest(
    manifest: Mapping[str, Any] | None,
    *,
    tourney: str,
    event_id: int | str,
    sim_round: int,
    configured_expected_avg: float,
    configured_course_averages: Mapping[str, Any] | None = None,
    sim_dict: Mapping[Any, Any] | None = None,
    model_players: Iterable[Any] | None = None,
    now: datetime | None = None,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
    current_overlay: Mapping[str, Any] | None = None,
) -> HealthReport:
    errors: list[str] = []
    warnings: list[str] = []
    payload = dict(manifest or {})
    manifest_id = payload.get("manifest_sha256")
    if not payload:
        return HealthReport(False, errors=["simulation health manifest is missing"])
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"unsupported health schema {payload.get('schema_version')!r}")
    if not manifest_id or manifest_id != _content_id(payload):
        errors.append("simulation health manifest content hash is invalid")
    if payload.get("kind") != "round_simulation":
        errors.append(f"unexpected health manifest kind {payload.get('kind')!r}")
    event = payload.get("event") or {}
    if _normalise_name(event.get("tourney")) != _normalise_name(tourney):
        errors.append(f"manifest tourney {event.get('tourney')!r} != active {tourney!r}")
    if str(event.get("event_id")) != str(event_id):
        errors.append(f"manifest event {event.get('event_id')!r} != active {event_id!r}")
    if str(event.get("round")) != str(sim_round):
        errors.append(f"manifest round {event.get('round')!r} != active R{sim_round}")
    manifest_avg = math.nan
    try:
        manifest_avg = float((payload.get("scoring") or {}).get("expected_avg"))
        active_avg = float(configured_expected_avg)
        if not math.isfinite(active_avg) or abs(manifest_avg - active_avg) > 1e-6:
            errors.append(
                f"manifest expected average {manifest_avg:g} != active {active_avg:g}"
            )
    except (TypeError, ValueError):
        errors.append("expected scoring average is missing or invalid")
    scoring = payload.get("scoring") or {}
    source = payload.get("source") or {}
    stored_course_averages = scoring.get("configured_course_averages") or {}
    try:
        stored_course_averages = {
            str(code).casefold().strip(): float(value)
            for code, value in stored_course_averages.items()
        }
        if not stored_course_averages or any(
            not code
            or not math.isfinite(value)
            or not EXPECTED_AVG_RANGE[0] <= value <= EXPECTED_AVG_RANGE[1]
            for code, value in stored_course_averages.items()
        ):
            errors.append("manifest authoritative course scoring baselines are invalid")
        if set(stored_course_averages) == {"field"}:
            if abs(stored_course_averages["field"] - manifest_avg) > 1e-6:
                errors.append("manifest single-course baseline disagrees with expected average")
        if configured_course_averages is not None:
            active_course_averages = {
                str(code).casefold().strip(): float(value)
                for code, value in configured_course_averages.items()
            }
            if set(active_course_averages) != set(stored_course_averages) or any(
                abs(active_course_averages[code] - stored_course_averages.get(code, math.inf)) > 1e-6
                for code in active_course_averages
            ):
                errors.append(
                    "manifest per-course scoring baselines differ from the active Sheet"
                )
    except (AttributeError, TypeError, ValueError):
        errors.append("manifest authoritative course scoring baselines are missing or invalid")
    if (
        source.get("expected_avg_authority") == "sheet_score_est"
        or source.get("parent_manifest_sha256")
    ):
        derivation = source.get("derivation") or {}
        try:
            derivation_from = float(derivation.get("from_expected_avg"))
            derivation_to = float(derivation.get("to_expected_avg"))
            derivation_delta = float(derivation.get("delta"))
            if source.get("parent_manifest_sha256") in {None, ""}:
                errors.append("score-est derivation has no parent simulation manifest")
            if derivation.get("method") != "uniform_rounding_bin_v1":
                errors.append("score-est derivation uses an unsupported fractional method")
            if (
                not all(math.isfinite(v) for v in (
                    derivation_from, derivation_to, derivation_delta
                ))
                or abs(derivation_to - manifest_avg) > 1e-6
                or abs((derivation_to - derivation_from) - derivation_delta) > 1e-6
            ):
                errors.append("score-est derivation baseline/delta provenance is inconsistent")
        except (TypeError, ValueError):
            errors.append("score-est derivation provenance is missing or invalid")
    try:
        implied_mean = float(scoring.get("implied_field_mean"))
        recorded_empirical = float(scoring.get("empirical_field_mean"))
        recorded_delta = float(scoring.get("centering_delta"))
        tolerance = float(scoring.get("centering_tolerance"))
        if tolerance > CENTERING_TOLERANCE_STROKES + 1e-12:
            errors.append("manifest weakened the production centering tolerance")
        if abs((recorded_empirical - implied_mean) - recorded_delta) > 1e-9:
            errors.append("manifest centering delta is internally inconsistent")
        if abs(recorded_empirical - implied_mean) > CENTERING_TOLERANCE_STROKES:
            errors.append(
                f"empirical tape mean {recorded_empirical:.4f} is miscentered from "
                f"implied mean {implied_mean:.4f}"
            )
        player_expected = scoring.get("player_expected_means") or {}
        expected_hash = hashlib.sha256(_canonical_json(player_expected)).hexdigest()
        if expected_hash != scoring.get("player_expected_means_sha256"):
            errors.append("per-player centering input hash is invalid")
        if float(scoring.get("player_centering_rmse_tolerance")) > (
            PLAYER_CENTERING_RMSE_TOLERANCE + 1e-12
        ):
            errors.append("manifest weakened the per-player centering RMSE tolerance")
        if float(scoring.get("player_centering_max_tolerance")) > (
            PLAYER_CENTERING_MAX_TOLERANCE + 1e-12
        ):
            errors.append("manifest weakened the per-player centering max tolerance")
        if float(scoring.get("player_centering_rmse")) > PLAYER_CENTERING_RMSE_TOLERANCE:
            errors.append("manifest records excessive per-player centering RMSE")
        if float(scoring.get("player_centering_max_error")) > PLAYER_CENTERING_MAX_TOLERANCE:
            errors.append("manifest records excessive per-player centering error")
    except (TypeError, ValueError):
        errors.append("manifest behavioral centering evidence is missing or invalid")

    generated = parse_utc((payload.get("source") or {}).get("generated_at"))
    root_generated = parse_utc(
        (payload.get("source") or {}).get("root_generated_at")
        or (payload.get("source") or {}).get("generated_at")
    )
    now = now or utc_now()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    if generated is None:
        errors.append("manifest source timestamp is missing or invalid")
    else:
        age = now.astimezone(timezone.utc) - generated
        if age < -timedelta(minutes=5):
            errors.append(f"manifest timestamp is {abs(age.total_seconds()) / 60:.1f}m in the future")
        if age > timedelta(hours=float(max_age_hours)):
            errors.append(
                f"simulation is {age.total_seconds() / 3600:.1f}h old; "
                f"maximum is {max_age_hours:g}h"
            )
    if root_generated is None:
        errors.append("manifest root tape timestamp is missing or invalid")
    else:
        root_age = now.astimezone(timezone.utc) - root_generated
        if root_age < -timedelta(minutes=5):
            errors.append(
                f"root tape timestamp is {abs(root_age.total_seconds()) / 60:.1f}m "
                "in the future"
            )
        if root_age > timedelta(hours=float(max_age_hours)):
            errors.append(
                f"root simulation tape is {root_age.total_seconds() / 3600:.1f}h old; "
                f"maximum is {max_age_hours:g}h"
            )

    if not (payload.get("checks") or {}).get("passed"):
        prior = (payload.get("checks") or {}).get("errors") or []
        errors.append("simulation producer rejected the tape" + (f": {prior[0]}" if prior else ""))
    if (payload.get("approval") or {}).get("status") != "approved":
        errors.append("simulation health manifest is not approved")

    stored_sim = payload.get("simulation") or {}
    if sim_dict is not None:
        comparison_players = list(model_players) if model_players is not None else list(sim_dict)
        facts, tape_errors = _inspect_tape(sim_dict, comparison_players)
        errors.extend(tape_errors)
        for key in ("num_players", "num_sims", "player_set_sha256", "tape_sha256"):
            if facts.get(key) != stored_sim.get(key):
                errors.append(
                    f"exact simulation tape mismatch for {key}: "
                    f"{facts.get(key)!r} != {stored_sim.get(key)!r}"
                )
        if abs(
            float(facts.get("empirical_field_score_mean"))
            - float(stored_sim.get("empirical_field_score_mean"))
        ) > 1e-10:
            errors.append("exact simulation tape empirical field mean differs from manifest")
        stored_player_expected = scoring.get("player_expected_means") or {}
        if set(stored_player_expected) != {_normalise_name(name) for name in sim_dict}:
            errors.append("stored per-player centering field does not match exact tape")
        else:
            deltas = []
            for raw_name, values in sim_dict.items():
                expected_player = float(stored_player_expected[_normalise_name(raw_name)])
                deltas.append(float(np.asarray(values, dtype=float).mean()) - expected_player)
            actual_rmse = float(np.sqrt(np.mean(np.square(deltas))))
            actual_max = max(abs(delta) for delta in deltas)
            if abs(actual_rmse - float(scoring.get("player_centering_rmse"))) > 1e-9:
                errors.append("exact tape per-player centering RMSE differs from manifest")
            if abs(actual_max - float(scoring.get("player_centering_max_error"))) > 1e-9:
                errors.append("exact tape per-player centering max differs from manifest")

    stored_overlay = (payload.get("model") or {}).get("shot_dispersion_overlay") or {}
    if _overlay_status_is_unsafe(stored_overlay):
        errors.append(
            "stored shot-dispersion overlay provenance is unsafe: "
            f"{stored_overlay.get('status')}"
        )
    if current_overlay is not None:
        if _overlay_status_is_unsafe(current_overlay):
            errors.append(
                "current shot-dispersion overlay provenance is unsafe: "
                f"{current_overlay.get('status')}"
            )
        for key in (
            "status", "required_current_event", "used_by_selected_tape",
            "feature_sha256", "distribution_sha256", "text_hash_mode",
        ):
            if stored_overlay.get(key) != current_overlay.get(key):
                errors.append(f"shot-dispersion provenance changed for {key}")
        stored_config_hash = stored_overlay.get("config_sha256")
        current_config_hashes = {
            current_overlay.get("config_sha256"),
            current_overlay.get("config_sha256_crlf_legacy"),
        } - {None}
        if stored_config_hash not in current_config_hashes:
            errors.append("shot-dispersion provenance changed for config_sha256")

    return HealthReport(
        not errors,
        manifest_id=manifest_id,
        errors=errors,
        warnings=warnings,
        facts={"generated_at": utc_stamp(generated) if generated else None},
    )


def require_simulation_healthy(*args: Any, **kwargs: Any) -> HealthReport:
    report = validate_simulation_manifest(*args, **kwargs)
    print(f"  [sim-health] {report.summary()}")
    if not report.ok:
        raise SimulationHealthError(report.summary())
    return report


def write_bound_artifact_manifest(
    destination: str | os.PathLike[str],
    *,
    kind: str,
    simulation_manifest: Mapping[str, Any],
    files: Mapping[str, str | os.PathLike[str]],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a detached manifest binding a simulation approval to exact files."""
    if (
        not simulation_manifest
        or simulation_manifest.get("manifest_sha256") != _content_id(simulation_manifest)
        or (simulation_manifest.get("approval") or {}).get("status") != "approved"
        or not (simulation_manifest.get("checks") or {}).get("passed")
    ):
        raise SimulationHealthError(
            f"cannot bind {kind}: source simulation manifest is unapproved or invalid"
        )
    bindings = {
        label: {"path": Path(path).name, "sha256": file_sha256(path)}
        for label, path in sorted(files.items())
    }
    payload = seal_manifest({
        "schema_version": SCHEMA_VERSION,
        "gate_version": GATE_VERSION,
        "kind": kind,
        "generated_at": utc_stamp(),
        "simulation_manifest": copy.deepcopy(dict(simulation_manifest)),
        "files": bindings,
        "extra": copy.deepcopy(dict(extra or {})),
    })
    Path(destination).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def validate_bound_artifact(
    manifest: Mapping[str, Any] | None,
    *,
    kind: str,
    files: Mapping[str, str | os.PathLike[str]],
    tourney: str,
    event_id: int | str,
    sim_round: int,
    configured_expected_avg: float,
    configured_course_averages: Mapping[str, Any] | None = None,
    now: datetime | None = None,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
    current_overlay: Mapping[str, Any] | None = None,
) -> HealthReport:
    errors: list[str] = []
    payload = dict(manifest or {})
    outer_id = payload.get("manifest_sha256")
    if not payload:
        return HealthReport(False, errors=[f"{kind} health manifest is missing"])
    if outer_id != _content_id(payload):
        errors.append(f"{kind} manifest content hash is invalid")
    if payload.get("kind") != kind:
        errors.append(f"artifact manifest kind {payload.get('kind')!r} != {kind!r}")
    stored_files = payload.get("files") or {}
    for label, path in files.items():
        binding = stored_files.get(label) or {}
        target = Path(path)
        if not target.is_file():
            errors.append(f"bound artifact file is missing: {target}")
            continue
        actual = file_sha256(target)
        if actual != binding.get("sha256"):
            errors.append(f"exact {label} file hash does not match published manifest")
        if target.name != binding.get("path"):
            errors.append(f"{label} filename does not match published manifest")

    sim_report = validate_simulation_manifest(
        payload.get("simulation_manifest"),
        tourney=tourney,
        event_id=event_id,
        sim_round=sim_round,
        configured_expected_avg=configured_expected_avg,
        configured_course_averages=configured_course_averages,
        now=now,
        max_age_hours=max_age_hours,
        current_overlay=current_overlay,
    )
    errors.extend(sim_report.errors)
    return HealthReport(
        not errors,
        manifest_id=outer_id,
        errors=errors,
        warnings=sim_report.warnings,
        facts={"simulation_manifest_id": sim_report.manifest_id},
    )


def require_bound_artifact(*args: Any, **kwargs: Any) -> HealthReport:
    report = validate_bound_artifact(*args, **kwargs)
    print(f"  [sim-health] {report.summary()}")
    if not report.ok:
        raise SimulationHealthError(report.summary())
    return report


def require_exact_simulation_source(
    artifact_manifest: Mapping[str, Any] | None,
    active_simulation_manifest: Mapping[str, Any] | None,
    *,
    artifact_label: str,
) -> None:
    """Reject artifacts bound to a parent or any other simulation generation.

    Parent reuse is unsafe for derived score-est manifests: it can make stale
    outright fairs look healthy merely because their old round cache is the
    active cache's ancestor. Exact manifest identity is cheap and unambiguous.
    """
    artifact_source = (artifact_manifest or {}).get("simulation_manifest") or {}
    artifact_id = artifact_source.get("manifest_sha256")
    active_id = (active_simulation_manifest or {}).get("manifest_sha256")
    if not artifact_id or not active_id or artifact_id != active_id:
        raise SimulationHealthError(
            f"BLOCKED — {artifact_label} does not match the exact active simulation "
            "manifest; rebuild it before pricing or publishing"
        )


def validate_h2h_probability_table(h2h_df: Any, simulation_manifest: Mapping[str, Any]) -> list[str]:
    """Check all-pairs coverage, uniqueness, field identity, and probability mass."""
    errors: list[str] = []
    required = {"player_a", "player_b", "p_a_lt_b", "p_tie"}
    if h2h_df is None or not required <= set(getattr(h2h_df, "columns", [])):
        return ["round H2H table lacks required probability columns"]
    expected_n = int((simulation_manifest.get("simulation") or {}).get("num_players") or 0)
    expected_pairs = expected_n * (expected_n - 1) // 2
    if len(h2h_df) != expected_pairs:
        errors.append(f"H2H pair coverage is {len(h2h_df)}/{expected_pairs}")
    a = h2h_df["player_a"].map(_normalise_name)
    b = h2h_df["player_b"].map(_normalise_name)
    if bool((a >= b).any()):
        errors.append("H2H pairs are not in unique canonical player order")
    pairs = list(zip(a, b))
    if len(pairs) != len(set(pairs)):
        errors.append("H2H table contains duplicate player pairs")
    players = set(a) | set(b)
    if len(players) != expected_n:
        errors.append(f"H2H player coverage is {len(players)}/{expected_n}")
    elif names_sha256(players) != (simulation_manifest.get("simulation") or {}).get("player_set_sha256"):
        errors.append("H2H player set does not match source simulation tape")
    try:
        p_lt = np.asarray(h2h_df["p_a_lt_b"], dtype=float)
        p_tie = np.asarray(h2h_df["p_tie"], dtype=float)
        if not np.isfinite(p_lt).all() or not np.isfinite(p_tie).all():
            errors.append("H2H probabilities contain non-finite values")
        elif (
            np.any(p_lt < 0) or np.any(p_lt > 1)
            or np.any(p_tie < 0) or np.any(p_tie > 1)
            # Publisher rounds each component to 5dp independently; allow the
            # worst-case 0.00001 rounding overshoot, but nothing material.
            or np.any(p_lt + p_tie > 1.00002)
        ):
            errors.append("H2H win/tie probability mass is outside [0, 1]")
    except (TypeError, ValueError):
        errors.append("H2H probabilities are not numeric")
    return errors


def require_h2h_probability_table(h2h_df: Any, simulation_manifest: Mapping[str, Any]) -> None:
    errors = validate_h2h_probability_table(h2h_df, simulation_manifest)
    if errors:
        report = HealthReport(False, errors=errors)
        print(f"  [sim-health] {report.summary()}")
        raise SimulationHealthError(report.summary())
    print("  [sim-health] PASS — complete H2H field and probability mass")


def validate_round_score_probability_table(
    score_df: Any, simulation_manifest: Mapping[str, Any]
) -> list[str]:
    """Validate a published integer-score PMF against its exact source tape."""
    errors: list[str] = []
    required = {"player_name", "score", "prob"}
    if score_df is None or getattr(score_df, "empty", True):
        return ["round score PMF is empty"]
    if not required <= set(getattr(score_df, "columns", [])):
        return ["round score PMF lacks player_name/score/prob columns"]

    names = score_df["player_name"].map(_normalise_name)
    if bool((names == "").any()):
        errors.append("round score PMF contains blank player names")
    if bool(score_df.assign(_name=names).duplicated(["_name", "score"]).any()):
        errors.append("round score PMF contains duplicate player/score rows")
    expected_n = int((simulation_manifest.get("simulation") or {}).get("num_players") or 0)
    players = set(names)
    if len(players) != expected_n:
        errors.append(f"round score PMF player coverage is {len(players)}/{expected_n}")
    elif names_sha256(players) != (
        simulation_manifest.get("simulation") or {}
    ).get("player_set_sha256"):
        errors.append("round score PMF player set does not match source simulation tape")

    try:
        scores = np.asarray(score_df["score"], dtype=float)
        probs = np.asarray(score_df["prob"], dtype=float)
        if not np.isfinite(scores).all() or not np.isfinite(probs).all():
            errors.append("round score PMF contains non-finite values")
        elif np.any(np.abs(scores - np.rint(scores)) > 1e-9):
            errors.append("round score PMF contains non-integer settlement scores")
        elif np.any(probs < 0.0) or np.any(probs > 1.0):
            errors.append("round score PMF probability mass is outside [0, 1]")
        else:
            work = score_df.assign(_name=names, _score=scores, _prob=probs)
            totals = work.groupby("_name")["_prob"].sum()
            if not np.allclose(totals.to_numpy(dtype=float), 1.0, atol=1e-8):
                errors.append("round score PMF probabilities do not sum to one per player")
            pmf_field_mean = float(
                work.assign(_weighted=work["_score"] * work["_prob"])
                .groupby("_name")["_weighted"]
                .sum()
                .mean()
            )
            tape_field_mean = float(
                (simulation_manifest.get("simulation") or {}).get(
                    "empirical_field_score_mean"
                )
            )
            if not math.isfinite(tape_field_mean) or abs(
                pmf_field_mean - tape_field_mean
            ) > 1e-8:
                errors.append(
                    "round score PMF mean does not match its source simulation tape"
                )
    except (KeyError, TypeError, ValueError):
        errors.append("round score PMF values are not numeric")
    return errors


def require_round_score_probability_table(
    score_df: Any, simulation_manifest: Mapping[str, Any]
) -> None:
    errors = validate_round_score_probability_table(score_df, simulation_manifest)
    if errors:
        report = HealthReport(False, errors=errors)
        print(f"  [sim-health] {report.summary()}")
        raise SimulationHealthError(report.summary())
    print("  [sim-health] PASS — complete round-score PMF probability mass")


def validate_finish_probability_mass(finish_probs: Any, expected_players: Iterable[Any]) -> list[str]:
    """Validate the market-level probability mass before email/storage."""
    errors: list[str] = []
    if finish_probs is None or getattr(finish_probs, "empty", True):
        return errors
    required = {"player_name", "simulated_win_prob"}
    if not required <= set(finish_probs.columns):
        return ["finish probabilities lack player_name/simulated_win_prob"]
    names = finish_probs["player_name"].map(_normalise_name)
    if names.duplicated().any():
        errors.append("finish probability table has duplicate players")
    expected_set = {_normalise_name(p) for p in expected_players}
    if set(names) != expected_set:
        errors.append("finish probability field does not align with round simulation field")
    win = np.asarray(finish_probs["simulated_win_prob"], dtype=float)
    if not np.isfinite(win).all() or np.any((win < 0) | (win > 1)):
        errors.append("winner probabilities contain invalid values")
    elif abs(float(win.sum()) - 1.0) > 1e-6:
        errors.append(f"winner probability mass is {float(win.sum()):.8f}, expected 1.0")
    for col, target in (("top_5", 5), ("top_10", 10), ("top_20", 20)):
        if col not in finish_probs.columns:
            continue
        values = np.asarray(finish_probs[col], dtype=float)
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            errors.append(f"{col} probabilities contain invalid values")
        elif len(values) >= target and abs(float(values.sum()) - target) > 1e-4:
            errors.append(
                f"dead-heat {col} probability mass is {float(values.sum()):.6f}, expected {target}"
            )
    return errors


def validate_live_tournament_alignment(
    *,
    final_scores_path: str | os.PathLike[str],
    player_names_path: str | os.PathLike[str],
    made_cut_path: str | os.PathLike[str],
    finish_probs: Any,
    artifact_manifest: Mapping[str, Any],
) -> list[str]:
    """Verify every live outright artifact uses the same row/draw axes."""
    errors: list[str] = []
    try:
        names = json.loads(Path(player_names_path).read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"live tournament player-name tape is unreadable: {exc}"]
    normalised = [_normalise_name(name) for name in names]
    if len(normalised) != len(set(normalised)):
        errors.append("live tournament player-name tape contains duplicates")
    extra = artifact_manifest.get("extra") or {}
    try:
        final_scores = np.load(final_scores_path, mmap_mode="r", allow_pickle=False)
        made_cut = np.load(made_cut_path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        return errors + [f"live tournament array tape is unreadable: {exc}"]
    expected_shape = (int(extra.get("num_players", -1)), int(extra.get("num_sims", -1)))
    if tuple(final_scores.shape) != expected_shape:
        errors.append(
            f"live final-score shape {tuple(final_scores.shape)} != manifest {expected_shape}"
        )
    if tuple(made_cut.shape) != tuple(final_scores.shape):
        errors.append(
            f"made-cut shape {tuple(made_cut.shape)} != final-score shape {tuple(final_scores.shape)}"
        )
    if len(names) != final_scores.shape[0]:
        errors.append("live player-name count does not match final-score rows")
    if names_sha256(names) != extra.get("field_player_set_sha256"):
        errors.append("live tournament field hash does not match manifest")
    errors.extend(validate_finish_probability_mass(finish_probs, names))
    return errors


def require_live_tournament_alignment(**kwargs: Any) -> None:
    errors = validate_live_tournament_alignment(**kwargs)
    if errors:
        report = HealthReport(False, errors=errors)
        print(f"  [sim-health] {report.summary()}")
        raise SimulationHealthError(report.summary())
    print("  [sim-health] PASS — live outright tape row/draw alignment")


def require_market_outputs_healthy(
    *,
    finish_probs: Any,
    expected_players: Iterable[Any],
    bound_artifact_report: HealthReport | None = None,
) -> None:
    errors = validate_finish_probability_mass(finish_probs, expected_players)
    if finish_probs is not None and not getattr(finish_probs, "empty", True):
        if bound_artifact_report is not None and not bound_artifact_report.ok:
            errors.extend(bound_artifact_report.errors)
    if errors:
        report = HealthReport(False, errors=errors)
        print(f"  [sim-health] {report.summary()}")
        raise SimulationHealthError(report.summary())
    print("  [sim-health] PASS — market probability mass and field alignment")
