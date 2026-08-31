"""Pure paired-round shadow model for live field scoring expectations.

The shadow is deliberately isolated from the authoritative Sheet expectation.
It consumes completed-round context and returns diagnostics only; callers decide
where to log the result.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from statistics import median


CALIBRATION_PATH = (
    Path(__file__).resolve().parent
    / "permanent_data"
    / "live_scoring_shadow_calibration.json"
)
SETUP_YARDAGE_CALIBRATION_PATH = (
    Path(__file__).resolve().parent
    / "permanent_data"
    / "live_scoring_setup_yardage_shadow.json"
)

# PGA TOUR benchmark tee expectations from Broadie (2012), Table B.1.  Keeping
# the frozen curve here makes the scoring shadow independent of the collector
# checkout and lets its input hash fully describe the prospective prediction.
_TEE_DISTANCES_YD = (
    100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340,
    360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600,
)
_TEE_EXPECTED_STROKES = (
    2.92, 2.99, 2.97, 2.99, 3.05, 3.12, 3.17, 3.25, 3.45,
    3.65, 3.71, 3.79, 3.86, 3.92, 3.96, 3.99, 4.02, 4.08,
    4.17, 4.28, 4.41, 4.54, 4.65, 4.74, 4.79, 4.82,
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


def load_setup_yardage_calibration(path=SETUP_YARDAGE_CALIBRATION_PATH):
    """Load the independent, frozen daily-setup shadow calibration."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        calibration = json.load(handle)
    if calibration.get("schema_version") != "live-scoring-setup-yardage/v2":
        raise ShadowUnavailable(f"Unsupported setup calibration schema: {path}")
    if calibration.get("status") != "shadow_only" or not calibration.get("frozen"):
        raise ShadowUnavailable(f"Setup calibration is not frozen shadow-only: {path}")
    if not isinstance(calibration.get("round_reference_mean"), dict):
        raise ShadowUnavailable(f"Setup calibration has no round references: {path}")
    if not isinstance(calibration.get("round_reference_mode"), dict):
        raise ShadowUnavailable(
            f"Setup calibration has no round reference modes: {path}"
        )
    if not isinstance(calibration.get("round_course_eb_pseudocount"), dict):
        raise ShadowUnavailable(
            f"Setup calibration has no round course EB settings: {path}"
        )
    if not isinstance(calibration.get("course_references"), dict):
        raise ShadowUnavailable(f"Setup calibration has no course references: {path}")
    calibration["_calibration_sha256"] = _calibration_hash(calibration)
    return calibration


def _optional_finite(value):
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _tee_expected_strokes(distance_yd):
    """Linearly interpolate/extrapolate the frozen Broadie tee curve."""
    distance = _finite("hole yardage", distance_yd)
    upper = next(
        (
            index for index, threshold in enumerate(_TEE_DISTANCES_YD)
            if distance <= threshold
        ),
        len(_TEE_DISTANCES_YD),
    )
    if upper == 0:
        lower, upper = 0, 1
    elif upper == len(_TEE_DISTANCES_YD):
        lower, upper = upper - 2, upper - 1
    else:
        lower = upper - 1
    x0, x1 = _TEE_DISTANCES_YD[lower], _TEE_DISTANCES_YD[upper]
    y0, y1 = _TEE_EXPECTED_STROKES[lower], _TEE_EXPECTED_STROKES[upper]
    return y0 + (distance - x0) * (y1 - y0) / (x1 - x0)


def unavailable_setup_yardage_signal(calibration, reason):
    """Return an auditable zero-impact signal when setup data is unavailable."""
    calibration = calibration if isinstance(calibration, Mapping) else {}
    return {
        "status": "unavailable",
        "reason": str(reason),
        "model_version": (
            calibration.get("calibration_version")
            or "setup-yardage-unavailable"
        ),
        "calibration_hash": (
            _calibration_hash(calibration) if calibration else None
        ),
        "reference_source": "unavailable",
        "reference_course_id": None,
        "reference_course_n": None,
        "reference_pseudocount": None,
        "adjustment": 0.0,
    }


def _setup_round_reference(calibration, target_round, course_id):
    """Return the exact-physical-course EB reference or the global fallback."""
    global_reference = _finite(
        f"R{target_round} global setup reference",
        (calibration.get("round_reference_mean") or {}).get(str(target_round)),
    )
    mode = str(
        (calibration.get("round_reference_mode") or {}).get(str(target_round))
        or ""
    ).strip()
    if mode not in {"global", "course_eb"}:
        raise ShadowUnavailable(
            f"R{target_round} setup reference mode is invalid"
        )

    physical_course_id = None
    if course_id not in (None, ""):
        try:
            physical_course_id = int(course_id)
        except (TypeError, ValueError):
            # A non-DataGolf identifier (including a TOURCAST layout ID) is not
            # eligible for course centering and deliberately falls back global.
            physical_course_id = None

    if mode == "global":
        return {
            "reference": global_reference,
            "global_reference": global_reference,
            "course_mean": None,
            "source": "global",
            "course_id": physical_course_id,
            "course_n": 0,
            "pseudocount": None,
        }

    pseudo = _finite(
        f"R{target_round} setup EB pseudocount",
        (calibration.get("round_course_eb_pseudocount") or {}).get(
            str(target_round)
        ),
    )
    if pseudo < 0:
        raise ShadowUnavailable("Setup EB pseudocount must be non-negative")

    course_entry = (
        (calibration.get("course_references") or {}).get(
            str(physical_course_id)
        )
        if physical_course_id is not None
        else None
    )
    if course_entry is None:
        return {
            "reference": global_reference,
            "global_reference": global_reference,
            "course_mean": None,
            "source": "global",
            "course_id": physical_course_id,
            "course_n": 0,
            "pseudocount": pseudo,
        }
    if not isinstance(course_entry, Mapping):
        raise ShadowUnavailable(
            f"Setup course reference {physical_course_id} is malformed"
        )
    try:
        course_n = int(course_entry.get("n"))
    except (TypeError, ValueError) as exc:
        raise ShadowUnavailable(
            f"Setup course reference {physical_course_id} has invalid n"
        ) from exc
    round_entry = (course_entry.get("rounds") or {}).get(str(target_round))
    if course_n < 1 or not isinstance(round_entry, Mapping):
        raise ShadowUnavailable(
            f"Setup course reference {physical_course_id} has no R{target_round} data"
        )
    course_sum = _finite(
        f"R{target_round} course setup sum", round_entry.get("sum_delta")
    )
    course_mean = course_sum / course_n
    stored_mean = _finite(
        f"R{target_round} stored course setup mean",
        round_entry.get("mean_delta"),
    )
    if not math.isclose(course_mean, stored_mean, rel_tol=0.0, abs_tol=1e-12):
        raise ShadowUnavailable(
            f"Setup course reference {physical_course_id} mean is inconsistent"
        )
    reference = (course_sum + pseudo * global_reference) / (course_n + pseudo)
    stored_reference = _finite(
        f"R{target_round} stored course EB setup reference",
        round_entry.get("eb_reference"),
    )
    if not math.isclose(reference, stored_reference, rel_tol=0.0, abs_tol=1e-12):
        raise ShadowUnavailable(
            f"Setup course reference {physical_course_id} EB value is inconsistent"
        )
    return {
        "reference": reference,
        "global_reference": global_reference,
        "course_mean": course_mean,
        "source": f"course_eb:{physical_course_id}",
        "course_id": physical_course_id,
        "course_n": course_n,
        "pseudocount": pseudo,
    }


def compute_setup_yardage_signal(
    calibration, geometry_rows, target_round, course_id=None
):
    """Convert complete daily hole setups into a guarded score adjustment.

    The signal compares the target round's sum of expected strokes from the tee
    with the mean of every strictly prior round.  It then subtracts the frozen
    historical mean for that transition so a normal R2/R3/R4 setup is already
    represented by the round baseline and paired transition calibration.
    """
    try:
        target_round = int(target_round)
    except (TypeError, ValueError) as exc:
        raise ShadowUnavailable("Setup target round is invalid") from exc
    if target_round not in (2, 3, 4):
        raise ShadowUnavailable(f"Setup target must be R2-R4, got R{target_round}")

    required_holes = int(_finite(
        "required setup holes", calibration.get("required_holes")
    ))
    response_weight = _finite(
        "setup response weight", calibration.get("response_weight")
    )
    max_adjustment = _finite(
        "maximum setup adjustment", calibration.get("max_abs_adjustment")
    )
    max_round_delta = _finite(
        "maximum round yardage delta",
        calibration.get("max_abs_round_yardage_delta"),
    )
    max_actual_official = _finite(
        "maximum actual/official hole delta",
        calibration.get("max_abs_actual_official_hole_delta"),
    )
    reference_details = _setup_round_reference(
        calibration, target_round, course_id
    )
    reference = reference_details["reference"]
    if not 0 <= response_weight <= 1:
        raise ShadowUnavailable("Setup response weight must be in [0, 1]")
    if max_adjustment <= 0 or max_round_delta <= 0 or max_actual_official <= 0:
        raise ShadowUnavailable("Setup guards must be positive")

    guard_ranges = calibration.get("guards") or {}
    par_ranges = {}
    for par in (3, 4, 5, 6):
        values = guard_ranges.get(f"par_{par}_yardage")
        if not isinstance(values, list) or len(values) != 2:
            raise ShadowUnavailable(f"Missing par-{par} setup guard")
        par_ranges[par] = tuple(_finite(f"par-{par} guard", value) for value in values)

    by_round = {rnd: {} for rnd in range(1, target_round + 1)}
    round_course_ids = {rnd: set() for rnd in range(1, target_round + 1)}
    for source in geometry_rows or []:
        if not isinstance(source, Mapping):
            raise ShadowUnavailable("Hole geometry rows must be mappings")
        try:
            rnd = int(source.get("round_no"))
            hole = int(source.get("hole_no"))
            par = int(source.get("par"))
        except (TypeError, ValueError) as exc:
            raise ShadowUnavailable("Hole geometry identity is invalid") from exc
        if rnd not in by_round:
            continue
        if hole < 1 or hole > required_holes:
            raise ShadowUnavailable(f"R{rnd} has invalid hole number {hole}")
        if par not in par_ranges:
            raise ShadowUnavailable(f"R{rnd}H{hole} has unsupported par {par}")
        if hole in by_round[rnd]:
            raise ShadowUnavailable(
                f"R{rnd}H{hole} has duplicate geometry (multi-course or stale rows)"
            )

        official = _optional_finite(source.get("official_yardage"))
        actual = _optional_finite(source.get("actual_yardage"))
        canonical = _optional_finite(source.get("yardage"))
        if actual is not None and official is not None and abs(actual - official) > max_actual_official:
            raise ShadowUnavailable(
                f"R{rnd}H{hole} actual/official yardage differs by "
                f"{abs(actual - official):.1f} yards"
            )
        yardage = next(
            (value for value in (actual, canonical, official) if value is not None),
            None,
        )
        if yardage is None:
            raise ShadowUnavailable(f"R{rnd}H{hole} has no usable yardage")
        lower, upper = par_ranges[par]
        if not lower <= yardage <= upper:
            raise ShadowUnavailable(
                f"R{rnd}H{hole} par-{par} yardage {yardage:.1f} is outside "
                f"[{lower:.0f}, {upper:.0f}]"
            )
        by_round[rnd][hole] = {
            "par": par,
            "yardage": yardage,
            "expected_strokes": _tee_expected_strokes(yardage),
        }
        course_value = str(source.get("course_id") or "").strip()
        if course_value:
            round_course_ids[rnd].add(course_value)

    expected_holes = set(range(1, required_holes + 1))
    for rnd, holes in by_round.items():
        if set(holes) != expected_holes:
            missing = sorted(expected_holes - set(holes))
            raise ShadowUnavailable(
                f"R{rnd} setup needs {required_holes} unique holes; "
                f"missing {missing}"
            )
        if len(round_course_ids[rnd]) > 1:
            raise ShadowUnavailable(f"R{rnd} setup contains multiple courses")

    for hole in expected_holes:
        pars = {by_round[rnd][hole]["par"] for rnd in by_round}
        if len(pars) != 1:
            raise ShadowUnavailable(
                f"H{hole} par changes across rounds: {sorted(pars)}"
            )

    round_yardages = {
        rnd: sum(item["yardage"] for item in holes.values())
        for rnd, holes in by_round.items()
    }
    round_indices = {
        rnd: sum(item["expected_strokes"] for item in holes.values())
        for rnd, holes in by_round.items()
    }
    prior_rounds = list(range(1, target_round))
    prior_yardage_average = sum(round_yardages[rnd] for rnd in prior_rounds) / len(prior_rounds)
    yardage_delta = round_yardages[target_round] - prior_yardage_average
    if abs(yardage_delta) > max_round_delta:
        raise ShadowUnavailable(
            f"R{target_round} setup yardage delta {yardage_delta:+.1f} exceeds "
            f"the {max_round_delta:.0f}-yard guard"
        )
    prior_index_average = sum(round_indices[rnd] for rnd in prior_rounds) / len(prior_rounds)
    raw_delta = round_indices[target_round] - prior_index_average
    centered_delta = raw_delta - reference
    uncapped_adjustment = response_weight * centered_delta
    adjustment = max(-max_adjustment, min(max_adjustment, uncapped_adjustment))

    return {
        "status": "ok",
        "reason": "",
        "model_version": calibration.get("calibration_version"),
        "calibration_hash": _calibration_hash(calibration),
        "prior_rounds": prior_rounds,
        "round_yardages": {str(key): value for key, value in round_yardages.items()},
        "prior_yardage_average": prior_yardage_average,
        "target_yardage": round_yardages[target_round],
        "yardage_delta": yardage_delta,
        "round_expected_strokes": {str(key): value for key, value in round_indices.items()},
        "prior_expected_strokes_average": prior_index_average,
        "target_expected_strokes": round_indices[target_round],
        "raw_expected_strokes_delta": raw_delta,
        "historical_round_reference_global": reference_details[
            "global_reference"
        ],
        "historical_round_reference_course_mean": reference_details[
            "course_mean"
        ],
        "historical_round_reference": reference,
        "reference_source": reference_details["source"],
        "reference_course_id": reference_details["course_id"],
        "reference_course_n": reference_details["course_n"],
        "reference_pseudocount": reference_details["pseudocount"],
        "centered_expected_strokes_delta": centered_delta,
        "response_weight": response_weight,
        "uncapped_adjustment": uncapped_adjustment,
        "max_abs_adjustment": max_adjustment,
        "was_capped": not math.isclose(adjustment, uncapped_adjustment),
        "adjustment": adjustment,
    }


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
    setup_yardage=None,
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
    if setup_yardage is None:
        setup_yardage = {
            "status": "unavailable",
            "reason": "setup signal was not supplied",
            "model_version": None,
            "calibration_hash": None,
            "adjustment": 0.0,
        }
    if not isinstance(setup_yardage, Mapping):
        raise ShadowUnavailable("Setup yardage signal must be a mapping")
    setup_yardage = dict(setup_yardage)
    setup_status = str(setup_yardage.get("status") or "").strip().lower()
    if setup_status not in {"ok", "unavailable"}:
        raise ShadowUnavailable(f"Unknown setup yardage status: {setup_status!r}")
    setup_adjustment = _finite(
        "setup yardage adjustment", setup_yardage.get("adjustment", 0.0)
    )
    if setup_status != "ok" and not math.isclose(setup_adjustment, 0.0):
        raise ShadowUnavailable("Unavailable setup yardage cannot move the shadow")

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
    shadow_before_setup = (
        (1 - paired_weight) * robust_structural
        + paired_weight * weather_paired
    )
    shadow_unrounded = shadow_before_setup + setup_adjustment

    setup_model_version = setup_yardage.get("model_version")
    model_version = calibration["calibration_version"]
    if setup_model_version:
        model_version = f"{model_version}+{setup_model_version}"

    calibration_hash = _calibration_hash(calibration)
    model_hash_payload = {
        "model_version": model_version,
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
        "setup_yardage": setup_yardage,
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
        "model_version": model_version,
        "calibration_hash": calibration_hash,
        "setup_calibration_hash": setup_yardage.get("calibration_hash"),
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
        "setup_yardage": setup_yardage,
        "setup_status": setup_status,
        "setup_reason": setup_yardage.get("reason", ""),
        "setup_adjustment": setup_adjustment,
        "shadow_before_setup": shadow_before_setup,
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
