"""Pure paired-round shadow model for live field scoring expectations.

The shadow is deliberately isolated from the authoritative Sheet expectation.
It consumes completed-round context and returns diagnostics only; callers decide
where to log the result.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import median


CALIBRATION_PATH = (
    Path(__file__).resolve().parent
    / "permanent_data"
    / "live_scoring_shadow_calibration.json"
)


class ShadowUnavailable(ValueError):
    """A shadow forecast cannot be computed safely from the available inputs."""


def _calibration_hash(calibration):
    canonical_calibration = {
        key: value
        for key, value in calibration.items()
        if key != "_calibration_sha256"
    }
    canonical = json.dumps(
        canonical_calibration, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_shadow_calibration(path=CALIBRATION_PATH):
    """Load and minimally validate the frozen shadow calibration."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        calibration = json.load(handle)
    if not calibration.get("calibration_version"):
        raise ShadowUnavailable(
            f"Shadow calibration has no calibration_version: {path}"
        )
    if not isinstance(calibration.get("rounds"), dict):
        raise ShadowUnavailable(f"Shadow calibration has no round settings: {path}")
    if calibration.get("schema_version") != "live-scoring-paired-calibration/v1":
        raise ShadowUnavailable(f"Unsupported shadow calibration schema: {path}")
    if calibration.get("status") != "shadow_only" or not calibration.get("frozen"):
        raise ShadowUnavailable(f"Calibration is not frozen shadow-only: {path}")
    calibration["_calibration_sha256"] = _calibration_hash(calibration)
    return calibration


def classify_cut_format(cut_line, opening_field_size):
    """Return cut or no_cut using the simulator's field-size contract."""
    try:
        cut_line = int(cut_line)
        opening_field_size = int(opening_field_size)
    except (TypeError, ValueError) as exc:
        raise ShadowUnavailable("Cut format needs integer cut/field sizes") from exc
    if opening_field_size < 1:
        raise ShadowUnavailable("Opening field size must be positive")
    return (
        "no_cut"
        if cut_line <= 0 or cut_line >= opening_field_size
        else "cut"
    )


def _finite(name, value):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ShadowUnavailable(f"{name} is missing or non-numeric") from exc
    if not math.isfinite(number):
        raise ShadowUnavailable(f"{name} is not finite")
    return number


def _transition_prior(calibration, target_round, course_id, cut_format):
    round_cfg = calibration["rounds"].get(str(target_round))
    if not isinstance(round_cfg, dict):
        raise ShadowUnavailable(f"No calibration for R{target_round}")
    transitions = round_cfg.get("global_transition_prior") or {}
    global_entry = transitions.get(cut_format)
    if global_entry is None:
        raise ShadowUnavailable(
            f"No {cut_format} transition calibration for R{target_round}"
        )
    global_mean = _finite("global transition", global_entry)
    pseudo_count = _finite(
        "course transition pseudo-count",
        round_cfg.get("course_transition_eb_pseudocount"),
    )
    if pseudo_count < 0:
        raise ShadowUnavailable("Course transition pseudo-count cannot be negative")

    course_entry = (
        (calibration.get("course_transitions") or {})
        .get(str(target_round), {})
        .get(cut_format, {})
        .get(str(course_id))
    )
    if not isinstance(course_entry, dict):
        return {
            "global_mean": global_mean,
            "course_mean": None,
            "course_n": 0,
            "pseudo_count": pseudo_count,
            "used": global_mean,
            "source": "global",
        }

    course_n = int(_finite("course transition n", course_entry.get("n")))
    course_mean = _finite(
        "course transition mean", course_entry.get("mean_delta")
    )
    if course_n < 1:
        raise ShadowUnavailable("Course transition n must be positive")
    used = (
        course_n * course_mean + pseudo_count * global_mean
    ) / (course_n + pseudo_count)
    return {
        "global_mean": global_mean,
        "course_mean": course_mean,
        "course_n": course_n,
        "pseudo_count": pseudo_count,
        "used": used,
        "source": f"course_eb:{course_id}",
    }


def compute_shadow_forecast(
    calibration,
    *,
    target_round,
    course_id,
    cut_format,
    active_players,
    completed_rounds,
    target_baseline,
    target_field_skill,
    target_weather_effect,
    production_candidate=None,
    sheet_before=None,
    published_after=None,
    cohort_members=None,
):
    """Compute a context-aware paired scoring forecast without side effects.

    completed_rounds must contain one mapping for every completed round with
    round, target-cohort score, realized weather_effect, the durable full-field
    structural_residual recorded when that round completed, and target-cohort
    coverage.
    """
    try:
        target_round = int(target_round)
        course_id = int(course_id)
        active_players = int(active_players)
    except (TypeError, ValueError) as exc:
        raise ShadowUnavailable("Round/course/player counts must be integers") from exc
    if target_round not in (2, 3, 4):
        raise ShadowUnavailable(f"Shadow target must be R2-R4, got R{target_round}")
    if cut_format not in ("cut", "no_cut"):
        raise ShadowUnavailable(f"Unknown cut format: {cut_format!r}")
    if active_players < 1:
        raise ShadowUnavailable("Active target cohort is empty")
    cohort_members = sorted({
        str(name).strip().lower()
        for name in (cohort_members or [])
        if str(name).strip()
    })
    if cohort_members and len(cohort_members) != active_players:
        raise ShadowUnavailable(
            "Target cohort membership does not match active player count"
        )

    round_cfg = calibration["rounds"].get(str(target_round)) or {}
    robust_weight = _finite(
        "robust residual weight", round_cfg.get("robust_residual_lambda")
    )
    paired_weight = _finite(
        "paired blend weight", round_cfg.get("paired_weight_beta")
    )
    if not 0 <= robust_weight <= 1:
        raise ShadowUnavailable("Robust residual weight must be in [0, 1]")
    if not 0 <= paired_weight <= 1:
        raise ShadowUnavailable("Paired blend weight must be in [0, 1]")
    minimum_coverage = _finite(
        "minimum cohort coverage",
        calibration.get("minimum_cohort_coverage", 0.8),
    )
    if not 0 <= minimum_coverage <= 1:
        raise ShadowUnavailable("Minimum cohort coverage must be in [0, 1]")

    by_round = {}
    for item in completed_rounds or []:
        if not isinstance(item, dict):
            raise ShadowUnavailable("Completed-round inputs must be mappings")
        try:
            rnd = int(item.get("round"))
        except (TypeError, ValueError) as exc:
            raise ShadowUnavailable("Completed round number is invalid") from exc
        if rnd in by_round:
            raise ShadowUnavailable(f"Duplicate completed-round input for R{rnd}")
        by_round[rnd] = {
            "score": _finite(f"R{rnd} score", item.get("score")),
            "weather_effect": _finite(
                f"R{rnd} weather effect", item.get("weather_effect")
            ),
            "structural_residual": _finite(
                f"R{rnd} structural residual",
                item.get("structural_residual"),
            ),
            "coverage": _finite(f"R{rnd} cohort coverage", item.get("coverage")),
        }
        if not 50 < by_round[rnd]["score"] < 100:
            raise ShadowUnavailable(
                f"R{rnd} score must be an absolute field score"
            )
        if abs(by_round[rnd]["structural_residual"]) > 5:
            raise ShadowUnavailable(
                f"R{rnd} structural residual is implausible"
            )

    expected_rounds = set(range(1, target_round))
    if set(by_round) != expected_rounds:
        raise ShadowUnavailable(
            f"R{target_round} needs completed rounds "
            f"{sorted(expected_rounds)}, got {sorted(by_round)}"
        )
    for rnd, item in by_round.items():
        if not 0 <= item["coverage"] <= 1:
            raise ShadowUnavailable(f"R{rnd} cohort coverage is outside [0, 1]")
        if item["coverage"] < minimum_coverage:
            raise ShadowUnavailable(
                f"R{rnd} cohort coverage {item['coverage']:.1%} is below "
                f"{minimum_coverage:.1%}"
            )

    target_baseline = _finite("target baseline", target_baseline)
    if not 50 < target_baseline < 100:
        raise ShadowUnavailable(
            "Target baseline must be an absolute field score"
        )
    target_field_skill = _finite("target field skill", target_field_skill)
    target_weather_effect = _finite(
        "target weather effect", target_weather_effect
    )
    production_candidate = (
        None
        if production_candidate is None
        else _finite("production candidate", production_candidate)
    )
    sheet_before = (
        None
        if sheet_before is None
        else _finite("pre-run Sheet value", sheet_before)
    )
    published_after = (
        None
        if published_after is None
        else _finite("published post-run value", published_after)
    )

    ordered = [by_round[rnd] for rnd in sorted(by_round)]
    prior_score_average = sum(item["score"] for item in ordered) / len(ordered)
    prior_weather_average = (
        sum(item["weather_effect"] for item in ordered) / len(ordered)
    )
    weather_delta = target_weather_effect - prior_weather_average
    transition = _transition_prior(
        calibration, target_round, course_id, cut_format
    )
    raw_paired = prior_score_average + transition["used"]
    weather_paired = raw_paired + weather_delta

    residuals = [item["structural_residual"] for item in ordered]
    median_residual = median(residuals)
    structural_no_feedback = (
        target_baseline - target_field_skill + target_weather_effect
    )
    robust_structural = structural_no_feedback + robust_weight * median_residual
    shadow_unrounded = (
        (1 - paired_weight) * robust_structural
        + paired_weight * weather_paired
    )

    calibration_hash = _calibration_hash(calibration)
    model_hash_payload = {
        "model_version": calibration["calibration_version"],
        "calibration_hash": calibration_hash,
        "target_round": target_round,
        "course_id": course_id,
        "cut_format": cut_format,
        "active_players": active_players,
        "cohort_members": cohort_members,
        "completed_rounds": by_round,
        "target_baseline": target_baseline,
        "target_field_skill": target_field_skill,
        "target_weather_effect": target_weather_effect,
    }
    model_input_hash = hashlib.sha256(
        json.dumps(
            model_hash_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    observation_hash_payload = {
        "model_input_hash": model_input_hash,
        "production_candidate": production_candidate,
        "sheet_before": sheet_before,
        "published_after": published_after,
    }
    input_hash = hashlib.sha256(
        json.dumps(
            observation_hash_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    cohort_hash = hashlib.sha256(
        "\n".join(cohort_members).encode("utf-8")
    ).hexdigest() if cohort_members else None

    return {
        "status": "ok",
        "model_version": calibration["calibration_version"],
        "calibration_hash": calibration_hash,
        "input_hash": input_hash,
        "model_input_hash": model_input_hash,
        "target_round": target_round,
        "course_id": course_id,
        "cut_format": cut_format,
        "active_players": active_players,
        "cohort_hash": cohort_hash,
        "minimum_coverage": minimum_coverage,
        "round_scores": {
            str(rnd): by_round[rnd]["score"] for rnd in sorted(by_round)
        },
        "round_coverages": {
            str(rnd): by_round[rnd]["coverage"] for rnd in sorted(by_round)
        },
        "round_weather_effects": {
            str(rnd): by_round[rnd]["weather_effect"] for rnd in sorted(by_round)
        },
        "round_structural_residuals": {
            str(rnd): residual
            for rnd, residual in zip(sorted(by_round), residuals)
        },
        "prior_score_average": prior_score_average,
        "prior_weather_average": prior_weather_average,
        "target_weather_effect": target_weather_effect,
        "weather_delta": weather_delta,
        "transition_global": transition["global_mean"],
        "transition_course_mean": transition["course_mean"],
        "transition_course_n": transition["course_n"],
        "transition_pseudo_count": transition["pseudo_count"],
        "transition_used": transition["used"],
        "transition_source": transition["source"],
        "raw_paired": raw_paired,
        "weather_paired": weather_paired,
        "target_baseline": target_baseline,
        "target_field_skill": target_field_skill,
        "structural_no_feedback": structural_no_feedback,
        "median_structural_residual": median_residual,
        "robust_residual_weight": robust_weight,
        "robust_structural": robust_structural,
        "paired_blend_weight": paired_weight,
        "production_candidate": production_candidate,
        "sheet_before": sheet_before,
        "published_after": published_after,
        "shadow_unrounded": shadow_unrounded,
        "shadow_display": round(shadow_unrounded, 1),
        "shadow_minus_production": (
            None
            if production_candidate is None
            else shadow_unrounded - production_candidate
        ),
    }
