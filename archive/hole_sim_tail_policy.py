"""Shared fail-closed tail policy for archive-only shot-chain research.

The transition count is a computational guard, not a physical score cap.
Every path that holes within the guarded transition lattice keeps its exact
physical score.  A path still active after the final transition is invalid
for the whole run; it is never clipped, force-holed, rerolled, or filtered.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


TAIL_POLICY_VERSION = "unbounded_hole_score_18_transition_fail_closed_v1"
MAX_CONTINUATION_TRANSITIONS = 18
HIGH_SCORE_DIAGNOSTIC_STROKES = 15
ROUND_LATTICE_MIN_STROKES = 40
ROUND_LATTICE_MAX_STROKES = 140
TRANSITION_LOOP_STATUS = "transition_loop_unresolved"
LEGACY_SCORE_CAP_STATUS = "max_hole_strokes_unresolved"


def manifest_policy() -> dict[str, object]:
    """Return the closed, JSON-safe policy persisted by every producer."""

    return {
        "version": TAIL_POLICY_VERSION,
        "max_continuation_transitions": MAX_CONTINUATION_TRANSITIONS,
        "transition_indices": list(range(MAX_CONTINUATION_TRANSITIONS)),
        "physical_hole_score_cap": None,
        "terminal_by_score_cap": False,
        "hole_score_clipping": False,
        "high_score_diagnostic_strokes": HIGH_SCORE_DIAGNOSTIC_STROKES,
        "high_score_diagnostic_only": True,
        "still_active_after_transition_limit": TRANSITION_LOOP_STATUS,
        "legacy_score_cap_status": LEGACY_SCORE_CAP_STATUS,
        "legacy_score_cap_status_emitted": False,
        "survivor_filtering": False,
        "round_totals_dtype": "int16",
        "round_lattice_strokes": [
            ROUND_LATTICE_MIN_STROKES,
            ROUND_LATTICE_MAX_STROKES,
        ],
        "numeric_overflow_or_lattice_violation": "fail_closed",
    }


def validate_manifest_policy(value: object, *, label: str = "tail_policy") -> None:
    """Require exact policy identity; missing/extra/changed fields fail closed."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    observed = dict(value)
    expected = manifest_policy()
    if observed != expected:
        missing = sorted(set(expected).difference(observed))
        extra = sorted(set(observed).difference(expected))
        changed = sorted(
            key
            for key in set(expected).intersection(observed)
            if observed[key] != expected[key]
        )
        raise ValueError(
            f"{label} differs from {TAIL_POLICY_VERSION}: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )


def int16_exact(values: object, *, label: str) -> np.ndarray:
    """Convert integral values to int16 without rounding, clipping, or wrap."""

    array = np.asarray(values)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{label} must be numeric")
    if array.dtype.kind == "f":
        if not np.isfinite(array).all() or not np.equal(array, np.rint(array)).all():
            raise ValueError(f"{label} must contain finite integer strokes")
    as_int64 = array.astype(np.int64, casting="unsafe", copy=False)
    if array.size:
        lower = int(as_int64.min())
        upper = int(as_int64.max())
        limits = np.iinfo(np.int16)
        if lower < limits.min or upper > limits.max:
            raise OverflowError(
                f"{label} outside int16 range [{limits.min}, {limits.max}]: "
                f"observed [{lower}, {upper}]"
            )
    return np.asarray(as_int64, dtype=np.int16)


def add_hole_strokes_to_round(
    round_totals: object,
    hole_strokes: object,
    *,
    label: str,
) -> np.ndarray:
    """Add one hole on a wide accumulator and reject eventual int16 wrap."""

    current = np.asarray(round_totals)
    hole = np.asarray(hole_strokes)
    if current.shape != hole.shape:
        raise ValueError(
            f"{label} round/hole shapes differ: {current.shape} != {hole.shape}"
        )
    if current.dtype.kind not in "iu" or hole.dtype.kind not in "iu":
        raise ValueError(f"{label} round/hole strokes must be integers")
    combined = current.astype(np.int64) + hole.astype(np.int64)
    # Validate now, but retain a wide accumulator until all 18 holes are in.
    int16_exact(combined, label=label)
    return combined.astype(np.int32)


def round_totals_int16(values: object, *, label: str) -> np.ndarray:
    """Validate the common physical 40..140 lattice and return exact int16."""

    exact = int16_exact(values, label=label)
    if exact.size:
        lower = int(exact.min())
        upper = int(exact.max())
        if (
            lower < ROUND_LATTICE_MIN_STROKES
            or upper > ROUND_LATTICE_MAX_STROKES
        ):
            raise ValueError(
                f"{label} outside frozen physical round lattice "
                f"[{ROUND_LATTICE_MIN_STROKES}, {ROUND_LATTICE_MAX_STROKES}]: "
                f"observed [{lower}, {upper}]"
            )
    return exact


def high_score_diagnostic_mask(hole_strokes: object) -> np.ndarray:
    """Flag tails for reporting only; this never changes terminal state/score."""

    values = np.asarray(hole_strokes)
    return values >= HIGH_SCORE_DIAGNOSTIC_STROKES
