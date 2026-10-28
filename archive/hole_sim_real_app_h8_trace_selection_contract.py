"""Source-only contract for a future Lowry R1 H8 APP trace overlay.

This module deliberately does not import the admitted selector session or any
chain, runner, solver, simulation, or production module.  It freezes the one
reviewed selection scope and the semantic common-random-number mapping, but it
does not authorize a callback.  The current selector session exposes an atom
identity result rather than the complete physical row required by the traced
state machine, so the adapter below always fails before invoking a session.
"""

from __future__ import annotations

from dataclasses import dataclass as _dataclass, fields as _fields
import math as _math
from types import MappingProxyType as _MappingProxyType
from typing import NoReturn as _NoReturn


globals().pop("annotations", None)


TRACE_SELECTION_CONTRACT = "fixed_lowry_r1h8_trace_selection_source_contract_v1"
PHYSICAL_SELECTION_CONTRACT = "complete_lowry_r1h8_physical_app_selection_v1"
INTEGRATION_STATUS = "HOLD_physical_projection_and_chain_callback_not_admitted"
COUNTER_RNG_CONTRACT = "blake2b_splitmix64_counter_v1"

TARGET_EVENT_KEY = "pga:R2026034"
TARGET_PLAYER_ID = "33204"
TARGET_ARM = "raw"
TARGET_ROUND_NO = 1
TARGET_HOLE_NO = 8
TARGET_START_LIE = "tee"
TARGET_START_DISTANCE_YD = 204.54171527805522
TARGET_DISTANCE_BUCKET = 14
TARGET_TRANSITION_INDEX = -1
TARGET_PARTICLE_COUNT = 128
TARGET_SUPPORT_KEY = "app|raw|R1|H8|tee|B14|33204"
TARGET_START_STATE_KEY = "tee|204.541715yd"
REWARD_REFERENCE_SHA256 = (
    "35ffdc1aa21ccc209ceb56a8b8e78b3118aa1cc8bd4c7372a92a0dc01b0bc68c"
)
ATOM_REWARD_TOLERANCE = 5e-6
TERMINAL_STATE_KEY = "terminal"
MAX_ABS_LOG_WEIGHT = 30.0

INITIAL_COUNTER_SEED = 20260824
CONTINUATION_COUNTER_SEED = 20261324

PINNED_HASHES = _MappingProxyType(
    {
        "authority_id": (
            "1a9d6fd9133811799a74cc61fcdc0ccbeec47617da11ef4a993ce3b2dcd529ce"
        ),
        "authority_lock_sha256": (
            "6b3e75849a44c4e6b880af09f7bac86e609bffb5bf64401c9ddf89fe25c30da5"
        ),
        "admission_session_source_sha256": (
            "55595e3ef2dfb58db64a0ecbf0563b9174c2b0e67cf9d689ce39d06b59771941"
        ),
        "selector_runtime_sha256": (
            "ab4607085ca3dfd75ee38deab7fddd5932be644aaac1dad5688622b4a452660c"
        ),
        "selector_support_sha256": (
            "f5fd780ea6c2a3e2795c8197f45f3486cd4a03606bac8e8c17b0548d7a07dc4c"
        ),
        "parent_trace_output_index_sha256": (
            "fc8073bef16bf4c710bb7c270fb3105e10f214779d3f6ce978c414ce60c242c9"
        ),
        "parent_trace_manifest_sha256": (
            "000bb368b0694eee0669cb03849fb97bf4307ccdfb94e63992298bce775dcb91"
        ),
        "parent_trace_ledger_sha256": (
            "8147e74cc7e859cbea12f9f73344d84cf2452392ce2e768bef0363c3db2f1e33"
        ),
        "traced_chain_source_sha256": (
            "11b14109b938cda5da4b4eaae4d00eb43f9d44074fdbd9e48328a4f889d7ab97"
        ),
        "all18_runner_source_sha256": (
            "b71e4b801ad68d98bc43b67a4f6d96d777fe9a32f089d902fe4a999f9bcfa164"
        ),
        "population_v5_output_index_sha256": (
            "0b965fb10e3842d825582c1fb6786d486b0cb9c2eb471eb7dee634f99e1b43ed"
        ),
        "population_v5_manifest_sha256": (
            "655d21edb07830d87af2f2e084d284381c649582dace729d6bd34f2a488524ff"
        ),
        "population_v5_runner_source_sha256": (
            "41aa690f5c6eb0f9efe79c8cf6d56cd88d7670bb6071e3240da6dbb34b65ee44"
        ),
    }
)

PINNED_CAPABILITIES = _MappingProxyType(
    {
        "support_admission": True,
        "weighting": True,
        "selection": True,
        "callback": False,
        "chain": False,
        "anchor": False,
        "mean_solve": False,
        "simulation": False,
        "production": False,
        "research_only": True,
    }
)


class H8TraceSelectionUnavailable(ValueError):
    """Raised whenever the frozen selection-only boundary is exceeded."""


def _exact_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise H8TraceSelectionUnavailable(f"{label} must be an exact int")
    return value


def _exact_float(value: object, label: str) -> float:
    if type(value) is not float or not _math.isfinite(value):
        raise H8TraceSelectionUnavailable(f"{label} must be an exact finite float")
    return value


def _exact_text(value: object, label: str) -> str:
    text = _exact_string(value, label)
    if not text:
        raise H8TraceSelectionUnavailable(f"{label} must be a non-empty exact str")
    return text


def _exact_string(value: object, label: str) -> str:
    if type(value) is not str:
        raise H8TraceSelectionUnavailable(f"{label} must be an exact str")
    return value


def _sha256(value: object, label: str) -> str:
    text = _exact_text(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise H8TraceSelectionUnavailable(f"{label} must be a lowercase SHA-256")
    return text


@_dataclass(frozen=True)
class H8TraceSelectionPins:
    authority_id: str
    authority_lock_sha256: str
    admission_session_source_sha256: str
    selector_runtime_sha256: str
    selector_support_sha256: str
    parent_trace_output_index_sha256: str
    parent_trace_manifest_sha256: str
    parent_trace_ledger_sha256: str
    traced_chain_source_sha256: str
    all18_runner_source_sha256: str
    population_v5_output_index_sha256: str
    population_v5_manifest_sha256: str
    population_v5_runner_source_sha256: str

    def __post_init__(self) -> None:
        for field in _fields(self):
            observed = _sha256(getattr(self, field.name), field.name)
            expected = PINNED_HASHES[field.name]
            if not str.__eq__(observed, expected):
                raise H8TraceSelectionUnavailable(
                    f"Pinned selection dependency changed: {field.name}"
                )


EXPECTED_PINS = H8TraceSelectionPins(**dict(PINNED_HASHES))


@_dataclass(frozen=True)
class H8TraceSelectionContext:
    event_key: str
    player_id: str
    arm: str
    round_no: int
    hole_no: int
    start_lie: str
    distance_bucket: int
    transition_index: int
    particle_id: int
    support_key: str

    def __post_init__(self) -> None:
        expected = {
            "event_key": TARGET_EVENT_KEY,
            "player_id": TARGET_PLAYER_ID,
            "arm": TARGET_ARM,
            "round_no": TARGET_ROUND_NO,
            "hole_no": TARGET_HOLE_NO,
            "start_lie": TARGET_START_LIE,
            "distance_bucket": TARGET_DISTANCE_BUCKET,
            "transition_index": TARGET_TRANSITION_INDEX,
            "support_key": TARGET_SUPPORT_KEY,
        }
        for name, target in expected.items():
            value = getattr(self, name)
            if type(value) is not type(target) or value != target:
                raise H8TraceSelectionUnavailable(
                    f"Selection context escaped fixed raw Lowry R1 H8 scope: {name}"
                )
        particle_id = _exact_int(self.particle_id, "particle_id")
        if not 0 <= particle_id < TARGET_PARTICLE_COUNT:
            raise H8TraceSelectionUnavailable("particle_id is outside frozen 0..127 axis")


@_dataclass(frozen=True)
class SemanticCounterSpec:
    semantic_name: str
    seed: int
    transition_index: int
    stream: str
    counter_contract: str = COUNTER_RNG_CONTRACT


SEMANTIC_COUNTER_SPECS = (
    SemanticCounterSpec("holeout", INITIAL_COUNTER_SEED, -1, "initial_app_holeout"),
    SemanticCounterSpec("proposal", INITIAL_COUNTER_SEED, -1, "initial_app_index"),
    SemanticCounterSpec("surface", INITIAL_COUNTER_SEED, -1, "initial_app_lie"),
    SemanticCounterSpec("hazard", CONTINUATION_COUNTER_SEED, 0, "hazard"),
)
SEMANTIC_UNIFORM_ORDER = tuple(spec.semantic_name for spec in SEMANTIC_COUNTER_SPECS)


@_dataclass(frozen=True)
class H8SemanticUniforms:
    holeout: float
    proposal: float
    surface: float
    hazard: float

    def __post_init__(self) -> None:
        for name in SEMANTIC_UNIFORM_ORDER:
            value = _exact_float(getattr(self, name), f"{name} uniform")
            if not 0.0 <= value < 1.0:
                raise H8TraceSelectionUnavailable(
                    f"{name} uniform must satisfy 0 <= u < 1"
                )

    def as_tuple(self) -> tuple[float, float, float, float]:
        return tuple(getattr(self, name) for name in SEMANTIC_UNIFORM_ORDER)  # type: ignore[return-value]


@_dataclass(frozen=True)
class H8TraceWeightingParameters:
    transfer_beta: tuple[float, float, float, float]
    target_ability_sg: tuple[float, float, float, float]
    lambda_correction: tuple[float, float, float, float]
    reward_reference_sha256: str = REWARD_REFERENCE_SHA256
    max_abs_log_weight: float = MAX_ABS_LOG_WEIGHT

    def __post_init__(self) -> None:
        for name in ("transfer_beta", "target_ability_sg", "lambda_correction"):
            value = getattr(self, name)
            if type(value) is not tuple or len(value) != 4:
                raise H8TraceSelectionUnavailable(f"{name} must be an immutable four-vector")
            for item in value:
                _exact_float(item, name)
        if any(item < 0.0 for item in self.transfer_beta):
            raise H8TraceSelectionUnavailable("transfer_beta cannot be negative")
        if self.reward_reference_sha256 != REWARD_REFERENCE_SHA256:
            raise H8TraceSelectionUnavailable("reward reference changed")
        if type(self.max_abs_log_weight) is not float or self.max_abs_log_weight != 30.0:
            raise H8TraceSelectionUnavailable("maximum log weight changed")


@_dataclass(frozen=True)
class BoundSemanticUniform:
    semantic_name: str
    value: float
    seed: int
    transition_index: int
    stream: str
    counter_contract: str


def bind_semantic_uniforms(
    uniforms: H8SemanticUniforms,
) -> tuple[BoundSemanticUniform, BoundSemanticUniform, BoundSemanticUniform, BoundSemanticUniform]:
    if type(uniforms) is not H8SemanticUniforms:
        raise H8TraceSelectionUnavailable("uniforms must be exact H8SemanticUniforms")
    return tuple(
        BoundSemanticUniform(
            semantic_name=spec.semantic_name,
            value=getattr(uniforms, spec.semantic_name),
            seed=spec.seed,
            transition_index=spec.transition_index,
            stream=spec.stream,
            counter_contract=spec.counter_contract,
        )
        for spec in SEMANTIC_COUNTER_SPECS
    )  # type: ignore[return-value]


@_dataclass(frozen=True)
class H8TraceSelectionRequest:
    context: H8TraceSelectionContext
    parameters: H8TraceWeightingParameters
    uniforms: H8SemanticUniforms
    pins: H8TraceSelectionPins = EXPECTED_PINS

    def __post_init__(self) -> None:
        if type(self.context) is not H8TraceSelectionContext:
            raise H8TraceSelectionUnavailable("context type changed")
        if type(self.parameters) is not H8TraceWeightingParameters:
            raise H8TraceSelectionUnavailable("weighting parameter type changed")
        if type(self.uniforms) is not H8SemanticUniforms:
            raise H8TraceSelectionUnavailable("uniform type changed")
        if type(self.pins) is not H8TraceSelectionPins or self.pins != EXPECTED_PINS:
            raise H8TraceSelectionUnavailable("selection dependency pins changed")


@_dataclass(frozen=True)
class CompletePhysicalAppSelection:
    contract: str
    support_key: str
    atom_index: int
    complete_row_order: int
    atom_id: str
    proposal_row_order: int
    surface_class_index: int
    hazard_donor_order: int
    start_lie: str
    start_distance_yd: float
    start_long_yd: float
    start_lat_yd: float
    start_state_key: str
    end_lie: str
    end_distance_yd: float
    end_long_yd: float
    end_lat_yd: float
    causal_strokes: int
    penalty_strokes: int
    stroke_cost: int
    hazard_resolution_status: str
    hazard_donor_key: str
    donor_event_key: str
    donor_player_id: str
    donor_player_name: str
    donor_round_no: int
    donor_hole_no: int
    donor_stroke_no: int
    start_reference_value: float
    end_reference_value: float
    fixed_reference_reward: float
    reward_reference_descriptor: str
    reference_model_sha256: str
    reward_reference_putting_sha256: str
    terminal: bool
    holed: bool
    end_state_key: str
    next_support_key: str

    def __post_init__(self) -> None:
        if type(self.contract) is not str or self.contract != PHYSICAL_SELECTION_CONTRACT:
            raise H8TraceSelectionUnavailable("physical selection contract changed")
        for name in (
            "atom_index",
            "complete_row_order",
            "proposal_row_order",
            "surface_class_index",
            "hazard_donor_order",
            "causal_strokes",
            "penalty_strokes",
            "stroke_cost",
            "donor_round_no",
            "donor_hole_no",
            "donor_stroke_no",
        ):
            _exact_int(getattr(self, name), name)
        for name in (
            "start_distance_yd",
            "start_long_yd",
            "start_lat_yd",
            "end_distance_yd",
            "end_long_yd",
            "end_lat_yd",
            "start_reference_value",
            "end_reference_value",
            "fixed_reference_reward",
        ):
            _exact_float(getattr(self, name), name)
        for name in (
            "support_key",
            "start_lie",
            "start_state_key",
            "end_lie",
            "hazard_resolution_status",
            "hazard_donor_key",
            "donor_event_key",
            "donor_player_id",
            "donor_player_name",
            "reward_reference_descriptor",
            "end_state_key",
        ):
            _exact_text(getattr(self, name), name)
        _exact_string(self.next_support_key, "next_support_key")
        _sha256(self.atom_id, "atom_id")
        reference_model = _sha256(
            self.reference_model_sha256, "reference_model_sha256"
        )
        if not str.__eq__(reference_model, REWARD_REFERENCE_SHA256):
            raise H8TraceSelectionUnavailable("reference model changed")
        _sha256(self.reward_reference_putting_sha256, "reward reference putting")
        if type(self.terminal) is not bool or type(self.holed) is not bool:
            raise H8TraceSelectionUnavailable("terminal and holed must be exact bools")
        if self.terminal != self.holed:
            raise H8TraceSelectionUnavailable("terminal and holed identity changed")
        if self.causal_strokes < 1 or self.penalty_strokes < 0:
            raise H8TraceSelectionUnavailable("physical stroke components are invalid")
        if self.stroke_cost != self.causal_strokes + self.penalty_strokes:
            raise H8TraceSelectionUnavailable("physical stroke-cost identity changed")
        if self.support_key != TARGET_SUPPORT_KEY:
            raise H8TraceSelectionUnavailable("physical support escaped fixed B14 scope")
        if self.start_lie != TARGET_START_LIE:
            raise H8TraceSelectionUnavailable("physical start lie must remain tee")
        if self.start_distance_yd != TARGET_START_DISTANCE_YD:
            raise H8TraceSelectionUnavailable("physical start distance escaped fixed B14 scope")
        if self.start_state_key != TARGET_START_STATE_KEY:
            raise H8TraceSelectionUnavailable("physical start-state key changed")
        if self.start_state_key != f"{self.start_lie}|{self.start_distance_yd:.6f}yd":
            raise H8TraceSelectionUnavailable("physical start-state identity changed")
        if self.end_distance_yd < 0.0:
            raise H8TraceSelectionUnavailable("end distance cannot be negative")
        expected_reward = (
            self.start_reference_value - float(self.stroke_cost) - self.end_reference_value
        )
        if not _math.isclose(
            self.fixed_reference_reward,
            expected_reward,
            rel_tol=0.0,
            abs_tol=ATOM_REWARD_TOLERANCE,
        ):
            raise H8TraceSelectionUnavailable(
                "fixed reference reward fails exact stroke/value accounting"
            )
        if self.terminal:
            if (
                self.end_lie != "hole"
                or self.end_distance_yd != 0.0
                or self.end_state_key != TERMINAL_STATE_KEY
                or self.end_reference_value != 0.0
                or self.penalty_strokes != 0
            ):
                raise H8TraceSelectionUnavailable("terminal physical endpoint changed")
            if self.next_support_key != "":
                raise H8TraceSelectionUnavailable("terminal atom cannot expose next support")
        else:
            if self.end_lie == "hole" or self.end_state_key == TERMINAL_STATE_KEY:
                raise H8TraceSelectionUnavailable("nonterminal physical endpoint changed")
            if self.end_state_key != f"{self.end_lie}|{self.end_distance_yd:.6f}yd":
                raise H8TraceSelectionUnavailable("nonterminal end-state identity changed")
            if self.next_support_key == "":
                raise H8TraceSelectionUnavailable("nonterminal atom requires next support")
            _sha256(self.next_support_key, "next_support_key")


REQUIRED_PHYSICAL_PROJECTION_FIELDS = tuple(
    field.name for field in _fields(CompletePhysicalAppSelection)
)


def validate_complete_physical_projection(value: object) -> CompletePhysicalAppSelection:
    if type(value) is not CompletePhysicalAppSelection:
        missing = tuple(
            name for name in REQUIRED_PHYSICAL_PROJECTION_FIELDS if not hasattr(value, name)
        )
        suffix = ",".join(missing) if missing else "exact_complete_physical_type"
        raise H8TraceSelectionUnavailable(
            f"Current public selection cannot satisfy physical projection: {suffix}"
        )
    return value


def select_complete_physical_app(
    request: H8TraceSelectionRequest, session: object
) -> _NoReturn:
    """Fail closed before touching ``session``; no physical callback is admitted."""

    if type(request) is not H8TraceSelectionRequest:
        raise H8TraceSelectionUnavailable("request type changed")
    if request.pins != EXPECTED_PINS:
        raise H8TraceSelectionUnavailable("selection dependency pins changed")
    bind_semantic_uniforms(request.uniforms)
    del session
    raise H8TraceSelectionUnavailable(
        "HOLD: callback_ready=False and chain_ready=False; current public "
        "selection omits the complete physical projection; session.select was not called"
    )


__all__ = (
    "TRACE_SELECTION_CONTRACT",
    "PHYSICAL_SELECTION_CONTRACT",
    "INTEGRATION_STATUS",
    "COUNTER_RNG_CONTRACT",
    "TARGET_EVENT_KEY",
    "TARGET_PLAYER_ID",
    "TARGET_ARM",
    "TARGET_ROUND_NO",
    "TARGET_HOLE_NO",
    "TARGET_START_LIE",
    "TARGET_START_DISTANCE_YD",
    "TARGET_DISTANCE_BUCKET",
    "TARGET_TRANSITION_INDEX",
    "TARGET_PARTICLE_COUNT",
    "TARGET_SUPPORT_KEY",
    "TARGET_START_STATE_KEY",
    "REWARD_REFERENCE_SHA256",
    "ATOM_REWARD_TOLERANCE",
    "TERMINAL_STATE_KEY",
    "MAX_ABS_LOG_WEIGHT",
    "INITIAL_COUNTER_SEED",
    "CONTINUATION_COUNTER_SEED",
    "PINNED_HASHES",
    "PINNED_CAPABILITIES",
    "SEMANTIC_COUNTER_SPECS",
    "SEMANTIC_UNIFORM_ORDER",
    "REQUIRED_PHYSICAL_PROJECTION_FIELDS",
    "H8TraceSelectionUnavailable",
    "H8TraceSelectionPins",
    "H8TraceSelectionContext",
    "SemanticCounterSpec",
    "H8SemanticUniforms",
    "H8TraceWeightingParameters",
    "BoundSemanticUniform",
    "H8TraceSelectionRequest",
    "CompletePhysicalAppSelection",
    "EXPECTED_PINS",
    "bind_semantic_uniforms",
    "validate_complete_physical_projection",
    "select_complete_physical_app",
)
