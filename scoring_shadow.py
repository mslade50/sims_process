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
    100,
    120,
    140,
    160,
    180,
    200,
    220,
    240,
    260,
    280,
    300,
    320,
    340,
    360,
    380,
    400,
    420,
    440,
    460,
    480,
    500,
    520,
    540,
    560,
    580,
    600,
)
_TEE_EXPECTED_STROKES = (
    2.92,
    2.99,
    2.97,
    2.99,
    3.05,
    3.12,
    3.17,
    3.25,
    3.45,
    3.65,
    3.71,
    3.79,
    3.86,
    3.92,
    3.96,
    3.99,
    4.02,
    4.08,
    4.17,
    4.28,
    4.41,
    4.54,
    4.65,
    4.74,
    4.79,
    4.82,
)

_SETUP_SCHEMA_V2 = "live-scoring-setup-yardage/v2"
_SETUP_SCHEMA_V3 = "live-scoring-setup-yardage/v3"
_SETUP_ADJUSTMENT_MODES = {
    "broadie",
    "empirical_global",
    "empirical_course_eb",
}


class ShadowUnavailable(ValueError):
    """A shadow forecast cannot be computed safely from the available inputs."""


def _calibration_hash(calibration):
    canonical_calibration = {
        key: value for key, value in calibration.items() if key != "_calibration_sha256"
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
    """Load and validate the independent daily-setup shadow calibration."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        calibration = json.load(handle)
    _validate_setup_yardage_calibration(calibration, path)
    calibration["_calibration_sha256"] = _calibration_hash(calibration)
    return calibration


def _integer(name, value, *, minimum=None):
    """Return a strict integer, rejecting bools and JSON floats."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ShadowUnavailable(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ShadowUnavailable(f"{name} must be at least {minimum}")
    return value


def _mapping(name, value):
    if not isinstance(value, Mapping):
        raise ShadowUnavailable(f"{name} must be a mapping")
    return value


def _canonical_string_list(name, value, *, allow_empty=False):
    if not isinstance(value, list):
        raise ShadowUnavailable(f"{name} must be a list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ShadowUnavailable(f"{name} must contain non-empty strings")
    if not allow_empty and not value:
        raise ShadowUnavailable(f"{name} cannot be empty")
    if value != sorted(set(value)):
        raise ShadowUnavailable(f"{name} must be sorted and unique")
    return value


def _canonical_json_sha256(value):
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _canonical_course_id(value):
    """Return a positive, canonical DataGolf course number or None.

    Runtime identifiers such as TOURCAST layout hashes, bools, and floats are
    intentionally ineligible.  Numeric strings are accepted because config and
    JSON object keys naturally serialize course numbers that way.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or not text.isascii() or not text.isdecimal():
        return None
    number = int(text)
    if number <= 0 or str(number) != text:
        return None
    return number


def _validate_reference_bundle(container, label):
    """Validate a three-transition reference block and stored EB arithmetic."""
    means = _mapping(
        f"{label} round_reference_mean",
        container.get("round_reference_mean"),
    )
    modes = _mapping(
        f"{label} round_reference_mode",
        container.get("round_reference_mode"),
    )
    pseudos = _mapping(
        f"{label} round_course_eb_pseudocount",
        container.get("round_course_eb_pseudocount"),
    )
    references = _mapping(
        f"{label} course_references",
        container.get("course_references"),
    )
    for rnd in (2, 3, 4):
        key = str(rnd)
        global_reference = _finite(f"{label} R{rnd} global reference", means.get(key))
        mode = str(modes.get(key) or "").strip()
        if mode not in {"global", "course_eb"}:
            raise ShadowUnavailable(
                f"{label} R{rnd} reference mode must be global or course_eb"
            )
        pseudo_value = pseudos.get(key)
        if mode == "global":
            if pseudo_value is not None:
                raise ShadowUnavailable(
                    f"{label} R{rnd} global reference must have null pseudocount"
                )
        else:
            pseudo = _finite(f"{label} R{rnd} course EB pseudocount", pseudo_value)
            if pseudo < 0:
                raise ShadowUnavailable(
                    f"{label} R{rnd} course EB pseudocount cannot be negative"
                )

        # Accessing the value above is deliberate: every transition must exist.
        del global_reference

    for course_key, course_value in references.items():
        if not isinstance(course_key, str) or _canonical_course_id(course_key) is None:
            raise ShadowUnavailable(
                f"{label} course reference key {course_key!r} is not a "
                "canonical DataGolf course_id"
            )
        course = _mapping(f"{label} course reference {course_key}", course_value)
        n = _integer(
            f"{label} course reference {course_key} n",
            course.get("n"),
            minimum=1,
        )
        if "course_names" in course:
            _canonical_string_list(
                f"{label} course reference {course_key} course_names",
                course.get("course_names"),
            )
        rounds = _mapping(
            f"{label} course reference {course_key} rounds",
            course.get("rounds"),
        )
        for round_key, round_value in rounds.items():
            if round_key not in {"2", "3", "4"}:
                raise ShadowUnavailable(
                    f"{label} course reference {course_key} has invalid "
                    f"round {round_key!r}"
                )
            entry = _mapping(
                f"{label} course reference {course_key} R{round_key}",
                round_value,
            )
            total = _finite(
                f"{label} course reference {course_key} R{round_key} sum",
                entry.get("sum_delta"),
            )
            mean = _finite(
                f"{label} course reference {course_key} R{round_key} mean",
                entry.get("mean_delta"),
            )
            if not math.isclose(total / n, mean, rel_tol=0.0, abs_tol=1e-12):
                raise ShadowUnavailable(
                    f"{label} course reference {course_key} R{round_key} "
                    "mean is inconsistent"
                )
            mode = modes[round_key]
            stored_eb = entry.get("eb_reference")
            if mode == "global":
                if stored_eb is not None:
                    raise ShadowUnavailable(
                        f"{label} course reference {course_key} R{round_key} "
                        "must not store a course EB value"
                    )
                continue
            global_reference = _finite(
                f"{label} R{round_key} global reference", means[round_key]
            )
            pseudo = _finite(
                f"{label} R{round_key} course EB pseudocount",
                pseudos[round_key],
            )
            expected_eb = (total + pseudo * global_reference) / (n + pseudo)
            eb = _finite(
                f"{label} course reference {course_key} R{round_key} EB",
                stored_eb,
            )
            if not math.isclose(expected_eb, eb, rel_tol=0.0, abs_tol=1e-12):
                raise ShadowUnavailable(
                    f"{label} course reference {course_key} R{round_key} "
                    "EB value is inconsistent"
                )


def _validate_yardage_coefficient_model(model):
    model = _mapping("yardage_coefficient_model", model)
    if model.get("schema_version") != "course-yardage-coefficient/v1":
        raise ShadowUnavailable("Unsupported yardage coefficient model schema")
    if not str(model.get("model_version") or "").strip():
        raise ShadowUnavailable("Yardage coefficient model has no model_version")
    if model.get("units") != "strokes_per_10_yards":
        raise ShadowUnavailable(
            "Yardage coefficient model units must be strokes_per_10_yards"
        )
    design = _mapping("yardage coefficient design", model.get("design"))
    for field in (
        "observation",
        "formula",
        "within_transform",
        "standard_errors",
        "course_identity",
        "informative_edition",
        "heterogeneity",
        "shrinkage",
    ):
        if not isinstance(design.get(field), str) or not design[field].strip():
            raise ShadowUnavailable(f"Yardage coefficient design {field} is missing")
    _canonical_string_list(
        "yardage coefficient design fixed_effects", design.get("fixed_effects")
    )
    _integer(
        "yardage coefficient training cutoff calendar year",
        model.get("training_cutoff_calendar_year"),
        minimum=1900,
    )
    training_keys = _canonical_string_list(
        "yardage coefficient training_event_keys",
        model.get("training_event_keys"),
    )
    expected_hash = _canonical_json_sha256(training_keys)
    if model.get("training_event_keys_sha256") != expected_hash:
        raise ShadowUnavailable(
            "Yardage coefficient training_event_keys_sha256 is inconsistent"
        )

    eligibility = _mapping("yardage coefficient eligibility", model.get("eligibility"))
    minimum_years = _integer(
        "minimum informative years",
        eligibility.get("min_informative_years"),
        minimum=3,
    )
    _integer(
        "minimum varying holes per edition",
        eligibility.get("min_varying_holes_per_edition"),
        minimum=1,
    )

    global_entry = _mapping("global yardage coefficient", model.get("global"))
    global_beta = _finite("global yardage beta", global_entry.get("beta"))
    if _finite("global yardage cluster SE", global_entry.get("cluster_se")) < 0:
        raise ShadowUnavailable("Global yardage cluster SE cannot be negative")
    global_n_events = _integer(
        "global yardage coefficient event count",
        global_entry.get("n_events"),
        minimum=1,
    )
    if global_n_events != len(training_keys):
        raise ShadowUnavailable(
            "Global yardage coefficient event count is inconsistent"
        )
    _integer(
        "global yardage coefficient hole-round count",
        global_entry.get("n_hole_rounds"),
        minimum=1,
    )
    if (
        _finite(
            "empirical maximum setup adjustment",
            model.get("max_abs_adjustment"),
        )
        <= 0
    ):
        raise ShadowUnavailable("Empirical maximum setup adjustment must be positive")

    heterogeneity = _mapping(
        "yardage coefficient heterogeneity", model.get("heterogeneity")
    )
    if heterogeneity.get("method") != "fixed_center_marginal_ml":
        raise ShadowUnavailable(
            "Yardage heterogeneity method must be fixed_center_marginal_ml"
        )
    heterogeneity_mean = _finite("yardage heterogeneity mean", heterogeneity.get("mu"))
    tau2 = _finite("yardage heterogeneity tau2", heterogeneity.get("tau2"))
    tau = _finite("yardage heterogeneity tau", heterogeneity.get("tau"))
    if tau2 < 0 or tau < 0:
        raise ShadowUnavailable("Yardage heterogeneity must be non-negative")
    if not math.isclose(tau * tau, tau2, rel_tol=1e-9, abs_tol=1e-12):
        raise ShadowUnavailable("Yardage heterogeneity tau and tau2 disagree")
    if not math.isclose(heterogeneity_mean, global_beta, rel_tol=0.0, abs_tol=1e-12):
        raise ShadowUnavailable("Yardage heterogeneity mean must equal the global beta")

    if model.get("fallback") != "global":
        raise ShadowUnavailable("Yardage coefficient fallback must be global")
    courses = _mapping("yardage coefficient courses", model.get("courses"))
    training_key_set = set(training_keys)
    for course_key, course_value in courses.items():
        if not isinstance(course_key, str) or _canonical_course_id(course_key) is None:
            raise ShadowUnavailable(
                f"Yardage coefficient course key {course_key!r} is not a "
                "canonical DataGolf course_id"
            )
        course = _mapping(f"yardage coefficient course {course_key}", course_value)
        _canonical_string_list(
            f"yardage coefficient course {course_key} course_names",
            course.get("course_names"),
        )
        years = course.get("informative_years")
        if not isinstance(years, list) or any(
            isinstance(year, bool) or not isinstance(year, int) for year in years
        ):
            raise ShadowUnavailable(
                f"Yardage coefficient course {course_key} informative_years "
                "must be integers"
            )
        if years != sorted(set(years)):
            raise ShadowUnavailable(
                f"Yardage coefficient course {course_key} informative_years "
                "must be sorted and unique"
            )
        n_years = _integer(
            f"yardage coefficient course {course_key} n_informative_years",
            course.get("n_informative_years"),
            minimum=0,
        )
        if n_years != len(years):
            raise ShadowUnavailable(
                f"Yardage coefficient course {course_key} informative-year "
                "count is inconsistent"
            )
        if n_years < minimum_years:
            raise ShadowUnavailable(
                f"Yardage coefficient course {course_key} does not meet the "
                "minimum informative-year requirement"
            )
        n_events = _integer(
            f"yardage coefficient course {course_key} n_events",
            course.get("n_events"),
            minimum=1,
        )
        event_keys = _canonical_string_list(
            f"yardage coefficient course {course_key} event_keys",
            course.get("event_keys"),
        )
        if not set(event_keys).issubset(training_key_set):
            raise ShadowUnavailable(
                f"Yardage coefficient course {course_key} has event keys "
                "outside the training panel"
            )
        if n_events != len(event_keys):
            raise ShadowUnavailable(
                f"Yardage coefficient course {course_key} event count is inconsistent"
            )
        raw_beta = _finite(
            f"yardage coefficient course {course_key} raw beta",
            course.get("raw_beta"),
        )
        cluster_se = _finite(
            f"yardage coefficient course {course_key} cluster SE",
            course.get("cluster_se"),
        )
        if cluster_se < 0:
            raise ShadowUnavailable(
                f"Yardage coefficient course {course_key} cluster SE cannot be negative"
            )
        reliability = _finite(
            f"yardage coefficient course {course_key} reliability",
            course.get("reliability"),
        )
        if not 0 <= reliability <= 1:
            raise ShadowUnavailable(
                f"Yardage coefficient course {course_key} reliability must be in [0, 1]"
            )
        denominator = tau2 + cluster_se * cluster_se
        expected_reliability = tau2 / denominator if denominator else 0.0
        if not math.isclose(
            reliability,
            expected_reliability,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ShadowUnavailable(
                f"Yardage coefficient course {course_key} reliability is inconsistent"
            )
        shrunk_beta = _finite(
            f"yardage coefficient course {course_key} shrunk beta",
            course.get("shrunk_beta"),
        )
        expected_shrunk = reliability * raw_beta + (1.0 - reliability) * global_beta
        if not math.isclose(shrunk_beta, expected_shrunk, rel_tol=0.0, abs_tol=1e-12):
            raise ShadowUnavailable(
                f"Yardage coefficient course {course_key} shrunk beta is inconsistent"
            )


def _validate_setup_yardage_calibration(calibration, source="calibration"):
    calibration = _mapping("setup calibration", calibration)
    schema = calibration.get("schema_version")
    if schema not in {_SETUP_SCHEMA_V2, _SETUP_SCHEMA_V3}:
        raise ShadowUnavailable(f"Unsupported setup calibration schema: {source}")
    if not str(calibration.get("calibration_version") or "").strip():
        raise ShadowUnavailable(
            f"Setup calibration has no calibration_version: {source}"
        )
    if (
        calibration.get("status") != "shadow_only"
        or calibration.get("frozen") is not True
    ):
        raise ShadowUnavailable(
            f"Setup calibration is not frozen shadow-only: {source}"
        )
    required_holes = _integer(
        "required setup holes", calibration.get("required_holes"), minimum=1
    )
    if required_holes != 18:
        raise ShadowUnavailable("Setup calibration must require 18 holes")
    response_weight = _finite(
        "setup response weight", calibration.get("response_weight")
    )
    if not 0 <= response_weight <= 1:
        raise ShadowUnavailable("Setup response weight must be in [0, 1]")
    for field, label in (
        ("max_abs_adjustment", "maximum setup adjustment"),
        ("max_abs_round_yardage_delta", "maximum round yardage delta"),
        (
            "max_abs_actual_official_hole_delta",
            "maximum actual/official hole delta",
        ),
    ):
        if _finite(label, calibration.get(field)) <= 0:
            raise ShadowUnavailable(f"{label} must be positive")
    guards = _mapping("setup guards", calibration.get("guards"))
    for par in (3, 4, 5, 6):
        values = guards.get(f"par_{par}_yardage")
        if not isinstance(values, list) or len(values) != 2:
            raise ShadowUnavailable(f"Missing par-{par} setup guard")
        lower, upper = (_finite(f"par-{par} guard", value) for value in values)
        if lower >= upper:
            raise ShadowUnavailable(f"Par-{par} setup guard is not increasing")

    _validate_reference_bundle(calibration, "Broadie setup")
    if schema == _SETUP_SCHEMA_V2:
        return

    mode = str(calibration.get("selected_adjustment_mode") or "broadie").strip()
    if mode not in _SETUP_ADJUSTMENT_MODES:
        raise ShadowUnavailable(f"Unknown selected setup adjustment mode: {mode!r}")
    _validate_yardage_coefficient_model(calibration.get("yardage_coefficient_model"))
    yardage_references = _mapping(
        "yardage_delta_references",
        calibration.get("yardage_delta_references"),
    )
    if not str(yardage_references.get("definition") or "").strip():
        raise ShadowUnavailable("Yardage delta reference definition is missing")
    _validate_reference_bundle(yardage_references, "Yardage-delta setup")


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
            index
            for index, threshold in enumerate(_TEE_DISTANCES_YD)
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
    selected_mode = str(
        calibration.get("selected_adjustment_mode") or "broadie"
    ).strip()
    if selected_mode not in _SETUP_ADJUSTMENT_MODES:
        selected_mode = "broadie"
    return {
        "status": "unavailable",
        "reason": str(reason),
        "model_version": (
            calibration.get("calibration_version") or "setup-yardage-unavailable"
        ),
        "calibration_hash": (_calibration_hash(calibration) if calibration else None),
        "reference_source": "unavailable",
        "reference_course_id": None,
        "reference_course_n": None,
        "reference_pseudocount": None,
        "selected_adjustment_mode": selected_mode,
        "selected_adjustment": 0.0,
        "broadie_adjustment": 0.0,
        "empirical_global_adjustment": 0.0,
        "empirical_course_adjustment": 0.0,
        "empirical_course_eb_adjustment": 0.0,
        "empirical_global_coefficient": None,
        "empirical_course_coefficient": None,
        "empirical_course_coefficient_source": None,
        "empirical_course_coefficient_fallback_reason": None,
        "empirical_course_n_informative_years": None,
        "empirical_course_cluster_se": None,
        "empirical_course_n_events": None,
        "yardage_delta_reference_global": None,
        "yardage_delta_reference_global_source": None,
        "yardage_delta_reference_course": None,
        "yardage_delta_reference_source": None,
        "centered_yardage_delta_global": None,
        "centered_yardage_delta_course": None,
        "adjustment": 0.0,
    }


def _round_reference(
    references,
    target_round,
    course_id,
    *,
    label,
    allow_course=True,
    missing_course_fallback=True,
):
    """Resolve a global/course-EB reference and verify stored arithmetic."""
    round_key = str(target_round)
    global_reference = _finite(
        f"R{target_round} global {label} reference",
        references["round_reference_mean"].get(round_key),
    )
    mode = str(references["round_reference_mode"].get(round_key) or "").strip()
    if mode not in {"global", "course_eb"}:
        raise ShadowUnavailable(f"R{target_round} {label} reference mode is invalid")

    base = {
        "reference": global_reference,
        "global_reference": global_reference,
        "course_mean": None,
        "source": "global",
        "course_id": course_id,
        "course_n": 0,
        "pseudocount": None,
        "fallback_reason": None,
    }
    if mode == "global":
        return base
    pseudo = _finite(
        f"R{target_round} {label} EB pseudocount",
        references["round_course_eb_pseudocount"].get(round_key),
    )
    if pseudo < 0:
        raise ShadowUnavailable(f"{label} EB pseudocount must be non-negative")
    if not allow_course:
        base["fallback_reason"] = "global_arm" if course_id else None
        return base

    course_entry = references["course_references"].get(str(course_id))
    if course_entry is None:
        base["pseudocount"] = pseudo
        base["fallback_reason"] = "course_reference_unavailable"
        return base
    if not isinstance(course_entry, Mapping):
        raise ShadowUnavailable(f"{label} course reference {course_id} is malformed")
    round_entry = (course_entry.get("rounds") or {}).get(round_key)
    if not isinstance(round_entry, Mapping):
        if missing_course_fallback:
            base["fallback_reason"] = "course_round_reference_unavailable"
            return base
        raise ShadowUnavailable(
            f"{label} course reference {course_id} has no R{target_round} data"
        )
    course_n = _integer(
        f"R{target_round} {label} course reference n",
        course_entry.get("n"),
        minimum=1,
    )
    course_sum = _finite(
        f"R{target_round} {label} course sum", round_entry.get("sum_delta")
    )
    course_mean = course_sum / course_n
    stored_mean = _finite(
        f"R{target_round} stored {label} course mean", round_entry.get("mean_delta")
    )
    if not math.isclose(course_mean, stored_mean, rel_tol=0.0, abs_tol=1e-12):
        raise ShadowUnavailable(
            f"{label} course reference {course_id} mean is inconsistent"
        )
    reference = (course_sum + pseudo * global_reference) / (course_n + pseudo)
    stored_reference = _finite(
        f"R{target_round} stored {label} course EB reference",
        round_entry.get("eb_reference"),
    )
    if not math.isclose(reference, stored_reference, rel_tol=0.0, abs_tol=1e-12):
        raise ShadowUnavailable(
            f"{label} course reference {course_id} EB value is inconsistent"
        )
    return {
        **base,
        "reference": reference,
        "course_mean": course_mean,
        "source": f"course_eb:{course_id}",
        "course_n": course_n,
        "pseudocount": pseudo,
    }


def _setup_round_reference(calibration, target_round, course_id):
    """Return the existing Broadie reference without tightening v2 identity."""
    try:
        physical_course_id = int(course_id) if course_id not in (None, "") else None
    except (TypeError, ValueError):
        physical_course_id = None
    return _round_reference(
        calibration,
        target_round,
        physical_course_id,
        label="setup",
        missing_course_fallback=False,
    )


def _yardage_coefficient_details(calibration, course_id):
    """Return global and eligible course-EB coefficients with provenance."""
    model = calibration["yardage_coefficient_model"]
    global_entry = model["global"]
    global_beta = _finite(
        "global empirical yardage coefficient", global_entry.get("beta")
    )
    global_cluster_se = _finite(
        "global empirical yardage coefficient cluster SE",
        global_entry.get("cluster_se"),
    )
    global_n_events = _integer(
        "global empirical yardage coefficient event count",
        global_entry.get("n_events"),
        minimum=1,
    )
    minimum_years = _integer(
        "minimum informative years",
        model["eligibility"].get("min_informative_years"),
        minimum=3,
    )
    physical_course_id = _canonical_course_id(course_id)
    course_entry = (
        model["courses"].get(str(physical_course_id))
        if physical_course_id is not None
        else None
    )
    details = {
        "model_version": model["model_version"],
        "units": model["units"],
        "global_beta": global_beta,
        "global_cluster_se": global_cluster_se,
        "global_n_events": global_n_events,
        "course_beta": global_beta,
        "course_source": "global",
        "course_id": physical_course_id,
        "course_n_informative_years": 0,
        "course_min_informative_years": minimum_years,
        "course_raw_beta": None,
        "course_cluster_se": global_cluster_se,
        "course_reliability": None,
        "course_n_events": global_n_events,
        "course_fallback_reason": None,
    }
    if physical_course_id is None:
        details["course_fallback_reason"] = "course_id_not_exact_numeric_datagolf"
    elif not isinstance(course_entry, Mapping):
        details["course_fallback_reason"] = "course_not_calibrated"
    else:
        n_years = _integer(
            f"Course {physical_course_id} informative-year count",
            course_entry.get("n_informative_years"),
            minimum=0,
        )
        if n_years < minimum_years:
            details["course_fallback_reason"] = "insufficient_informative_years"
        else:
            details.update(
                course_beta=_finite(
                    f"Course {physical_course_id} shrunk yardage coefficient",
                    course_entry.get("shrunk_beta"),
                ),
                course_source=f"course_eb:{physical_course_id}",
                course_n_informative_years=n_years,
                course_raw_beta=_finite(
                    f"Course {physical_course_id} raw yardage coefficient",
                    course_entry.get("raw_beta"),
                ),
                course_cluster_se=_finite(
                    f"Course {physical_course_id} yardage cluster SE",
                    course_entry.get("cluster_se"),
                ),
                course_reliability=_finite(
                    f"Course {physical_course_id} yardage reliability",
                    course_entry.get("reliability"),
                ),
                course_n_events=int(course_entry["n_events"]),
            )
    return details


def _yardage_round_reference(calibration, target_round, course_id, *, allow_course):
    return _round_reference(
        calibration["yardage_delta_references"],
        target_round,
        course_id,
        label="yardage-delta",
        allow_course=allow_course,
    )


def _capped_adjustment(value, maximum):
    adjustment = max(-maximum, min(maximum, value))
    return adjustment, not math.isclose(adjustment, value)


def _empirical_adjustment_arm(
    coefficient_details,
    reference_details,
    yardage_delta,
    maximum,
    *,
    use_course,
):
    prefix = "course" if use_course else "global"
    coefficient = coefficient_details[f"{prefix}_beta"]
    centered_yardage = yardage_delta - reference_details["reference"]
    uncapped = coefficient * centered_yardage / 10.0
    adjustment, was_capped = _capped_adjustment(uncapped, maximum)
    provenance = {
        "model_version": coefficient_details["model_version"],
        "units": coefficient_details["units"],
        "source": coefficient_details.get(f"{prefix}_source", "global"),
        "course_id": coefficient_details["course_id"] if use_course else None,
        "n_informative_years": (
            coefficient_details["course_n_informative_years"] if use_course else None
        ),
        "min_informative_years": (
            coefficient_details["course_min_informative_years"] if use_course else None
        ),
        "n_events": (
            coefficient_details["course_n_events"]
            if use_course
            else coefficient_details["global_n_events"]
        ),
        "raw_beta": coefficient_details["course_raw_beta"] if use_course else None,
        "cluster_se": coefficient_details[f"{prefix}_cluster_se"],
        "reliability": (
            coefficient_details["course_reliability"] if use_course else None
        ),
        "fallback_reason": (
            coefficient_details["course_fallback_reason"] if use_course else None
        ),
    }
    return {
        "status": "ok",
        "method": "empirical_yardage_coefficient",
        "coefficient": coefficient,
        "coefficient_source": provenance["source"],
        "coefficient_provenance": provenance,
        "yardage_delta": yardage_delta,
        "yardage_reference": reference_details["reference"],
        "yardage_reference_source": reference_details["source"],
        "yardage_reference_provenance": dict(reference_details),
        "centered_yardage_delta": centered_yardage,
        "uncapped_adjustment": uncapped,
        "max_abs_adjustment": maximum,
        "was_capped": was_capped,
        "adjustment": adjustment,
    }


def compute_setup_yardage_signal(
    calibration,
    geometry_rows,
    target_round,
    course_id=None,
    event_key=None,
):
    """Convert complete daily hole setups into a guarded score adjustment.

    The signal compares the target round's sum of expected strokes from the tee
    with the mean of every strictly prior round.  It then subtracts the frozen
    historical mean for that transition so a normal R2/R3/R4 setup is already
    represented by the round baseline and paired transition calibration.
    """
    schema = calibration.get("schema_version")
    if schema not in {_SETUP_SCHEMA_V2, _SETUP_SCHEMA_V3}:
        raise ShadowUnavailable(f"Unsupported setup calibration schema: {schema!r}")
    current_event_key = None
    if schema == _SETUP_SCHEMA_V3:
        # Revalidation is intentional.  It protects callers that pass an
        # in-memory artifact without using the JSON loader first.
        _validate_setup_yardage_calibration(calibration)
        if not isinstance(event_key, str) or not event_key.strip():
            raise ShadowUnavailable(
                "Setup calibration v3 requires current event_key for the "
                "training-overlap guard"
            )
        current_event_key = event_key.strip()
        training_keys = calibration["yardage_coefficient_model"]["training_event_keys"]
        if current_event_key in training_keys:
            raise ShadowUnavailable(
                f"Current event_key {current_event_key!r} overlaps the "
                "yardage-coefficient training panel"
            )

    try:
        target_round = int(target_round)
    except (TypeError, ValueError) as exc:
        raise ShadowUnavailable("Setup target round is invalid") from exc
    if target_round not in (2, 3, 4):
        raise ShadowUnavailable(f"Setup target must be R2-R4, got R{target_round}")

    required_holes = int(
        _finite("required setup holes", calibration.get("required_holes"))
    )
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
    reference_details = _setup_round_reference(calibration, target_round, course_id)
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
        if (
            actual is not None
            and official is not None
            and abs(actual - official) > max_actual_official
        ):
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
                f"R{rnd} setup needs {required_holes} unique holes; missing {missing}"
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
    prior_yardage_average = sum(round_yardages[rnd] for rnd in prior_rounds) / len(
        prior_rounds
    )
    yardage_delta = round_yardages[target_round] - prior_yardage_average
    if abs(yardage_delta) > max_round_delta:
        raise ShadowUnavailable(
            f"R{target_round} setup yardage delta {yardage_delta:+.1f} exceeds "
            f"the {max_round_delta:.0f}-yard guard"
        )
    prior_index_average = sum(round_indices[rnd] for rnd in prior_rounds) / len(
        prior_rounds
    )
    raw_delta = round_indices[target_round] - prior_index_average
    centered_delta = raw_delta - reference
    broadie_uncapped_adjustment = response_weight * centered_delta
    broadie_adjustment, broadie_was_capped = _capped_adjustment(
        broadie_uncapped_adjustment, max_adjustment
    )
    adjustment_arms = {
        "broadie": {
            "status": "ok",
            "method": "broadie_tee_expected_strokes",
            "raw_expected_strokes_delta": raw_delta,
            "reference": reference,
            "reference_source": reference_details["source"],
            "centered_expected_strokes_delta": centered_delta,
            "response_weight": response_weight,
            "uncapped_adjustment": broadie_uncapped_adjustment,
            "max_abs_adjustment": max_adjustment,
            "was_capped": broadie_was_capped,
            "adjustment": broadie_adjustment,
        }
    }

    global_arm = None
    course_arm = None
    empirical_max_adjustment = None
    if schema == _SETUP_SCHEMA_V3:
        coefficient_details = _yardage_coefficient_details(calibration, course_id)
        empirical_max_adjustment = _finite(
            "empirical maximum setup adjustment",
            calibration["yardage_coefficient_model"].get("max_abs_adjustment"),
        )
        global_reference = _yardage_round_reference(
            calibration,
            target_round,
            coefficient_details["course_id"],
            allow_course=False,
        )
        course_reference = _yardage_round_reference(
            calibration,
            target_round,
            coefficient_details["course_id"],
            allow_course=(
                coefficient_details["course_source"].startswith("course_eb:")
            ),
        )
        global_arm = _empirical_adjustment_arm(
            coefficient_details,
            global_reference,
            yardage_delta,
            empirical_max_adjustment,
            use_course=False,
        )
        course_arm = _empirical_adjustment_arm(
            coefficient_details,
            course_reference,
            yardage_delta,
            empirical_max_adjustment,
            use_course=True,
        )
        adjustment_arms.update(
            {
                "empirical_global": global_arm,
                "empirical_course_eb": course_arm,
            }
        )

    selected_mode = str(
        calibration.get("selected_adjustment_mode") or "broadie"
    ).strip()
    if schema == _SETUP_SCHEMA_V2:
        selected_mode = "broadie"
    if selected_mode not in adjustment_arms:
        raise ShadowUnavailable(
            f"Selected setup adjustment mode {selected_mode!r} is unavailable"
        )
    selected_arm = adjustment_arms[selected_mode]
    adjustment = selected_arm["adjustment"]
    uncapped_adjustment = selected_arm["uncapped_adjustment"]
    was_capped = selected_arm["was_capped"]
    selected_max_adjustment = selected_arm["max_abs_adjustment"]

    return {
        "status": "ok",
        "reason": "",
        "schema_version": schema,
        "model_version": calibration.get("calibration_version"),
        "calibration_hash": _calibration_hash(calibration),
        "event_key": current_event_key,
        "training_event_overlap_checked": schema == _SETUP_SCHEMA_V3,
        "training_event_keys_sha256": (
            calibration["yardage_coefficient_model"].get("training_event_keys_sha256")
            if schema == _SETUP_SCHEMA_V3
            else None
        ),
        "prior_rounds": prior_rounds,
        "round_yardages": {str(key): value for key, value in round_yardages.items()},
        "prior_yardage_average": prior_yardage_average,
        "target_yardage": round_yardages[target_round],
        "yardage_delta": yardage_delta,
        "round_expected_strokes": {
            str(key): value for key, value in round_indices.items()
        },
        "prior_expected_strokes_average": prior_index_average,
        "target_expected_strokes": round_indices[target_round],
        "raw_expected_strokes_delta": raw_delta,
        "historical_round_reference_global": reference_details["global_reference"],
        "historical_round_reference_course_mean": reference_details["course_mean"],
        "historical_round_reference": reference,
        "reference_source": reference_details["source"],
        "reference_course_id": reference_details["course_id"],
        "reference_course_n": reference_details["course_n"],
        "reference_pseudocount": reference_details["pseudocount"],
        "centered_expected_strokes_delta": centered_delta,
        "response_weight": response_weight,
        "uncapped_adjustment": uncapped_adjustment,
        "max_abs_adjustment": selected_max_adjustment,
        "was_capped": was_capped,
        "selected_adjustment_mode": selected_mode,
        "selected_adjustment": adjustment,
        "adjustment_arms": adjustment_arms,
        "broadie_uncapped_adjustment": broadie_uncapped_adjustment,
        "broadie_max_abs_adjustment": max_adjustment,
        "broadie_was_capped": broadie_was_capped,
        "broadie_adjustment": broadie_adjustment,
        "empirical_global_uncapped_adjustment": (
            global_arm["uncapped_adjustment"] if global_arm else None
        ),
        "empirical_global_was_capped": (
            global_arm["was_capped"] if global_arm else None
        ),
        "empirical_max_abs_adjustment": empirical_max_adjustment,
        "empirical_global_adjustment": (
            global_arm["adjustment"] if global_arm else None
        ),
        "empirical_course_uncapped_adjustment": (
            course_arm["uncapped_adjustment"] if course_arm else None
        ),
        "empirical_course_was_capped": (
            course_arm["was_capped"] if course_arm else None
        ),
        "empirical_course_adjustment": (
            course_arm["adjustment"] if course_arm else None
        ),
        "empirical_course_eb_adjustment": (
            course_arm["adjustment"] if course_arm else None
        ),
        "empirical_global_coefficient": (
            global_arm["coefficient"] if global_arm else None
        ),
        "empirical_course_coefficient": (
            course_arm["coefficient"] if course_arm else None
        ),
        "empirical_course_coefficient_source": (
            course_arm["coefficient_source"] if course_arm else None
        ),
        "empirical_course_coefficient_fallback_reason": (
            course_arm["coefficient_provenance"]["fallback_reason"]
            if course_arm
            else None
        ),
        "empirical_course_n_informative_years": (
            course_arm["coefficient_provenance"]["n_informative_years"]
            if course_arm
            else None
        ),
        "empirical_course_cluster_se": (
            course_arm["coefficient_provenance"]["cluster_se"] if course_arm else None
        ),
        "empirical_course_n_events": (
            course_arm["coefficient_provenance"]["n_events"] if course_arm else None
        ),
        "yardage_delta_reference_global": (
            global_arm["yardage_reference"] if global_arm else None
        ),
        "yardage_delta_reference_global_source": (
            global_arm["yardage_reference_source"] if global_arm else None
        ),
        "yardage_delta_reference_course": (
            course_arm["yardage_reference"] if course_arm else None
        ),
        "yardage_delta_reference_source": (
            course_arm["yardage_reference_source"] if course_arm else None
        ),
        "centered_yardage_delta_global": (
            global_arm["centered_yardage_delta"] if global_arm else None
        ),
        "centered_yardage_delta_course": (
            course_arm["centered_yardage_delta"] if course_arm else None
        ),
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
    return "no_cut" if cut_line <= 0 or cut_line >= opening_field_size else "cut"


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
    course_mean = _finite("course transition mean", course_entry.get("mean_delta"))
    if course_n < 1:
        raise ShadowUnavailable("Course transition n must be positive")
    used = (course_n * course_mean + pseudo_count * global_mean) / (
        course_n + pseudo_count
    )
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
    cohort_members = sorted(
        {
            str(name).strip().lower()
            for name in (cohort_members or [])
            if str(name).strip()
        }
    )
    if cohort_members and len(cohort_members) != active_players:
        raise ShadowUnavailable(
            "Target cohort membership does not match active player count"
        )

    round_cfg = calibration["rounds"].get(str(target_round)) or {}
    robust_weight = _finite(
        "robust residual weight", round_cfg.get("robust_residual_lambda")
    )
    paired_weight = _finite("paired blend weight", round_cfg.get("paired_weight_beta"))
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
            raise ShadowUnavailable(f"R{rnd} score must be an absolute field score")
        if abs(by_round[rnd]["structural_residual"]) > 5:
            raise ShadowUnavailable(f"R{rnd} structural residual is implausible")

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
        raise ShadowUnavailable("Target baseline must be an absolute field score")
    target_field_skill = _finite("target field skill", target_field_skill)
    target_weather_effect = _finite("target weather effect", target_weather_effect)
    production_candidate = (
        None
        if production_candidate is None
        else _finite("production candidate", production_candidate)
    )
    sheet_before = (
        None if sheet_before is None else _finite("pre-run Sheet value", sheet_before)
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
    prior_weather_average = sum(item["weather_effect"] for item in ordered) / len(
        ordered
    )
    weather_delta = target_weather_effect - prior_weather_average
    transition = _transition_prior(calibration, target_round, course_id, cut_format)
    raw_paired = prior_score_average + transition["used"]
    weather_paired = raw_paired + weather_delta

    residuals = [item["structural_residual"] for item in ordered]
    median_residual = median(residuals)
    structural_no_feedback = (
        target_baseline - target_field_skill + target_weather_effect
    )
    robust_structural = structural_no_feedback + robust_weight * median_residual
    shadow_before_setup = (
        1 - paired_weight
    ) * robust_structural + paired_weight * weather_paired
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
        json.dumps(model_hash_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
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
    cohort_hash = (
        hashlib.sha256("\n".join(cohort_members).encode("utf-8")).hexdigest()
        if cohort_members
        else None
    )

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
        "round_scores": {str(rnd): by_round[rnd]["score"] for rnd in sorted(by_round)},
        "round_coverages": {
            str(rnd): by_round[rnd]["coverage"] for rnd in sorted(by_round)
        },
        "round_weather_effects": {
            str(rnd): by_round[rnd]["weather_effect"] for rnd in sorted(by_round)
        },
        "round_structural_residuals": {
            str(rnd): residual for rnd, residual in zip(sorted(by_round), residuals)
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
