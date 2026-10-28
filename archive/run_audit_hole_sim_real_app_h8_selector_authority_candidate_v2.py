"""Launch the fixed v2 real-APP selector audit from one captured source image.

The launcher is deliberately separate from the detached auditor.  It reads the
reviewed auditor bytes once, verifies the frozen auditor/source-test pair, and
executes exactly those captured bytes.  Neither mode grants selector authority
or enables weighting, sampling, callbacks, simulation, anchoring, or production.

The two source pins below bind the independently reviewed detached auditor and
its focused test.  Any source drift makes the launcher fail closed.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import types
from typing import Mapping


REPO = Path(__file__).resolve().parents[1]
ARCHIVE = Path(__file__).resolve().parent

LAUNCHER_RELATIVE_PATH = (
    "archive/run_audit_hole_sim_real_app_h8_selector_authority_candidate_v2.py"
)
LAUNCHER_TEST_RELATIVE_PATH = (
    "test_run_audit_hole_sim_real_app_h8_selector_authority_candidate_v2.py"
)
AUDITOR_RELATIVE_PATH = (
    "archive/audit_hole_sim_real_app_h8_selector_authority_candidate_v2.py"
)
AUDITOR_TEST_RELATIVE_PATH = (
    "test_audit_hole_sim_real_app_h8_selector_authority_candidate_v2.py"
)
AUDIT_OUTPUT_RELATIVE_PATH = (
    "permanent_data/"
    "hole_sim_real_app_h8_selector_authority_candidate_audit_v2"
)

LAUNCHER_SOURCE = REPO / LAUNCHER_RELATIVE_PATH
LAUNCHER_TEST = REPO / LAUNCHER_TEST_RELATIVE_PATH
AUDITOR_SOURCE = REPO / AUDITOR_RELATIVE_PATH
AUDITOR_TEST = REPO / AUDITOR_TEST_RELATIVE_PATH
_FIXED_AUDIT_ROOT = REPO / AUDIT_OUTPUT_RELATIVE_PATH

LAUNCH_ATTESTATION_CONTRACT = "fixed_one_read_launcher_attestation_v2"
EXPECTED_SCOPE = "independent_byte_and_vectorized_invariant_audit_v2"
EXPECTED_SEMANTIC_ROOT = (
    "b8312616d491bdce5e97fa438e95a8b9ead36ce1cb0fb9d7274605cfe3675dae"
)
EXPECTED_VALIDATE_STATUS = "HOLD_source_reviews_required_no_audit_emitted"
EXPECTED_AUDIT_DECISION = (
    "GO_for_separate_data_only_authority_record_review_only"
)
EXPECTED_AUDIT_STATUS = (
    "candidate_byte_and_vectorized_audit_passed_record_review_required"
)

EXPECTED_AUDITOR_SOURCE_SHA256 = (
    "93b790ba2adc34df5c6ea2e65ae88c989c4f55a2747c922e11cc4f786b0e35f0"
)
EXPECTED_AUDITOR_TEST_SHA256 = (
    "11f9a55e89a22e1bb77addf09dc9fb395fd645cae6c6337c0d4a242147017eb9"
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ATTESTATION_FIELDS = {
    "contract",
    "launcher_relative_path",
    "launcher_test_relative_path",
    "launcher_sha256",
    "launcher_test_sha256",
}
_VALIDATE_RESULT_FIELDS = {
    "mode",
    "scope",
    "audit_only_semantic_root_sha256",
    "status",
}
_AUDIT_RESULT_FIELDS = {
    "mode",
    "scope",
    "decision",
    "audit_status",
    "audit_id",
    "output",
    "final_audit_tree",
}
_FINAL_TREE_FIELDS = {
    "files",
    "directories",
    "bytes",
    "inventory_sha256",
}
_CAPTURED_MODULE_NAME = (
    "_captured_audit_hole_sim_real_app_h8_selector_authority_candidate_v2"
)
_PATH_TYPE = type(Path())


class FixedAuditLauncherError(RuntimeError):
    """Raised when a fixed launcher invariant is not exact."""


def _digest(payload: bytes) -> str:
    if type(payload) is not bytes:
        raise TypeError("SHA-256 payload must be exact built-in bytes")
    return sha256(payload).hexdigest()


def _require_hex64(value: object, label: str) -> str:
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        raise FixedAuditLauncherError(f"{label} must be exact lowercase SHA-256")
    return value


def _require_equal(actual: object, expected: object, label: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise FixedAuditLauncherError(f"{label} changed")


def _require_fixed_paths() -> None:
    expected = {
        "launcher source": REPO / LAUNCHER_RELATIVE_PATH,
        "launcher test": REPO / LAUNCHER_TEST_RELATIVE_PATH,
        "auditor source": REPO / AUDITOR_RELATIVE_PATH,
        "auditor test": REPO / AUDITOR_TEST_RELATIVE_PATH,
        "audit output": REPO / AUDIT_OUTPUT_RELATIVE_PATH,
    }
    actual = {
        "launcher source": LAUNCHER_SOURCE,
        "launcher test": LAUNCHER_TEST,
        "auditor source": AUDITOR_SOURCE,
        "auditor test": AUDITOR_TEST,
        "audit output": _FIXED_AUDIT_ROOT,
    }
    for label, expected_path in expected.items():
        path = actual[label]
        if type(path) is not _PATH_TYPE or path != expected_path:
            raise FixedAuditLauncherError(f"Fixed {label} path changed")


def _read_regular_file_once(path: Path, label: str) -> bytes:
    if type(path) is not _PATH_TYPE:
        raise TypeError(f"{label} path must be exact pathlib.Path")
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise FixedAuditLauncherError(f"{label} must be a regular file")
    with path.open("rb") as handle:
        payload = handle.read()
    if type(payload) is not bytes:
        raise TypeError(f"{label} read did not return exact bytes")
    return payload


def _file_state(path: Path, label: str) -> dict[str, object]:
    payload = _read_regular_file_once(path, label)
    return {"bytes": len(payload), "sha256": _digest(payload)}


def _require_frozen_pin(value: object, label: str) -> str:
    try:
        return _require_hex64(value, label)
    except FixedAuditLauncherError as exc:
        raise FixedAuditLauncherError(
            f"{label} is provisional; launcher remains fail-closed"
        ) from exc


def _capture_fixed_sources() -> tuple[bytes, dict[str, dict[str, object]]]:
    """Capture the auditor once and freeze all four launch-time source states."""

    expected_auditor = _require_frozen_pin(
        EXPECTED_AUDITOR_SOURCE_SHA256,
        "expected auditor source SHA-256",
    )
    expected_auditor_test = _require_frozen_pin(
        EXPECTED_AUDITOR_TEST_SHA256,
        "expected auditor test SHA-256",
    )

    auditor_bytes = _read_regular_file_once(AUDITOR_SOURCE, "auditor source")
    auditor_state = {
        "bytes": len(auditor_bytes),
        "sha256": _digest(auditor_bytes),
    }
    _require_equal(
        auditor_state["sha256"],
        expected_auditor,
        "frozen auditor source SHA-256",
    )

    states = {
        "auditor_source": auditor_state,
        "auditor_test": _file_state(AUDITOR_TEST, "auditor test"),
        "launcher_source": _file_state(LAUNCHER_SOURCE, "launcher source"),
        "launcher_test": _file_state(LAUNCHER_TEST, "launcher test"),
    }
    _require_equal(
        states["auditor_test"]["sha256"],
        expected_auditor_test,
        "frozen auditor test SHA-256",
    )
    return auditor_bytes, states


def _launcher_attestation(
    states: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    if type(states) is not dict or set(states) != {
        "auditor_source",
        "auditor_test",
        "launcher_source",
        "launcher_test",
    }:
        raise TypeError("Captured source states must be an exact built-in dict")
    launcher_state = states["launcher_source"]
    launcher_test_state = states["launcher_test"]
    if type(launcher_state) is not dict or type(launcher_test_state) is not dict:
        raise TypeError("Launcher source states must be exact built-in dicts")
    attestation = {
        "contract": LAUNCH_ATTESTATION_CONTRACT,
        "launcher_relative_path": LAUNCHER_RELATIVE_PATH,
        "launcher_test_relative_path": LAUNCHER_TEST_RELATIVE_PATH,
        "launcher_sha256": launcher_state.get("sha256"),
        "launcher_test_sha256": launcher_test_state.get("sha256"),
    }
    _validate_attestation(attestation)
    return attestation


def _validate_attestation(attestation: object) -> None:
    if type(attestation) is not dict or set(attestation) != _ATTESTATION_FIELDS:
        raise TypeError("Launcher attestation must be an exact closed built-in dict")
    expected_text = {
        "contract": LAUNCH_ATTESTATION_CONTRACT,
        "launcher_relative_path": LAUNCHER_RELATIVE_PATH,
        "launcher_test_relative_path": LAUNCHER_TEST_RELATIVE_PATH,
    }
    for field, expected in expected_text.items():
        _require_equal(attestation.get(field), expected, f"attestation {field}")
    _require_hex64(attestation.get("launcher_sha256"), "attestation launcher SHA-256")
    _require_hex64(
        attestation.get("launcher_test_sha256"),
        "attestation launcher-test SHA-256",
    )


def _exec_captured_auditor(auditor_bytes: bytes) -> dict[str, object]:
    """Compile and execute only the already captured auditor byte string."""

    if type(auditor_bytes) is not bytes:
        raise TypeError("Captured auditor must be exact built-in bytes")
    namespace: dict[str, object] = {
        "__name__": _CAPTURED_MODULE_NAME,
        "__file__": str(AUDITOR_SOURCE),
        "__package__": None,
        "__loader__": None,
        "__spec__": None,
        "__cached__": None,
        "__builtins__": __builtins__,
    }
    code = compile(
        auditor_bytes,
        str(AUDITOR_SOURCE),
        "exec",
        flags=0,
        dont_inherit=True,
        optimize=0,
    )
    exec(code, namespace, namespace)
    for name in ("run_fixed_audit_v2", "final_audit_tree_v2"):
        value = namespace.get(name)
        if type(value) is not types.FunctionType:
            raise FixedAuditLauncherError(f"Captured auditor {name} is not an exact function")
        if value.__globals__ is not namespace:
            raise FixedAuditLauncherError(f"Captured auditor {name} escaped its namespace")
    return namespace


def _validate_final_tree(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != _FINAL_TREE_FIELDS:
        raise FixedAuditLauncherError(f"{label} must be an exact closed tree state")
    for field in ("files", "directories", "bytes"):
        number = value.get(field)
        if type(number) is not int or number < 0:
            raise FixedAuditLauncherError(f"{label} {field} must be a nonnegative int")
    _require_hex64(value.get("inventory_sha256"), f"{label} inventory SHA-256")
    return value


def _validate_result(mode: str, value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise FixedAuditLauncherError("Captured auditor result must be exact built-in dict")
    if mode == "validate-only":
        if set(value) != _VALIDATE_RESULT_FIELDS:
            raise FixedAuditLauncherError("Validate-only result field set changed")
        _require_equal(value.get("mode"), "validate-only", "validate-only mode")
        _require_equal(value.get("scope"), EXPECTED_SCOPE, "validate-only scope")
        _require_equal(
            value.get("audit_only_semantic_root_sha256"),
            EXPECTED_SEMANTIC_ROOT,
            "validate-only semantic root",
        )
        _require_equal(
            value.get("status"),
            EXPECTED_VALIDATE_STATUS,
            "validate-only status",
        )
        return value

    if mode != "audit":
        raise FixedAuditLauncherError("Internal mode changed")
    if set(value) != _AUDIT_RESULT_FIELDS:
        raise FixedAuditLauncherError("Audit result field set changed")
    _require_equal(value.get("mode"), "audit", "audit result mode")
    _require_equal(value.get("scope"), EXPECTED_SCOPE, "audit result scope")
    _require_equal(value.get("decision"), EXPECTED_AUDIT_DECISION, "audit decision")
    _require_equal(value.get("audit_status"), EXPECTED_AUDIT_STATUS, "audit status")
    _require_hex64(value.get("audit_id"), "audit ID")
    _require_equal(value.get("output"), str(_FIXED_AUDIT_ROOT), "audit output")
    _validate_final_tree(value.get("final_audit_tree"), "returned final audit tree")
    return value


def _rehash_fixed_sources(states: Mapping[str, Mapping[str, object]]) -> None:
    if type(states) is not dict:
        raise TypeError("Captured source states must remain exact built-in dict")
    paths = (
        ("auditor_source", AUDITOR_SOURCE, "auditor source postcheck"),
        ("auditor_test", AUDITOR_TEST, "auditor test postcheck"),
        ("launcher_source", LAUNCHER_SOURCE, "launcher source postcheck"),
        ("launcher_test", LAUNCHER_TEST, "launcher test postcheck"),
    )
    for key, path, label in paths:
        expected = states.get(key)
        if type(expected) is not dict:
            raise TypeError(f"Captured {key} state changed type")
        current = _file_state(path, label)
        if current != expected:
            raise FixedAuditLauncherError(f"{label} changed after capture")


def _path_lexists(path: Path) -> bool:
    if type(path) is not _PATH_TYPE:
        raise TypeError("Output path must be exact pathlib.Path")
    return os.path.lexists(os.fspath(path))


def run_fixed(mode: str) -> dict[str, object]:
    """Run exactly one fixed validation or audit launch with no overrides."""

    if type(mode) is not str or mode not in {"validate-only", "audit"}:
        raise TypeError("mode must be exact 'validate-only' or 'audit'")
    _require_fixed_paths()
    if _path_lexists(_FIXED_AUDIT_ROOT):
        raise FileExistsError(f"Fixed v2 audit output must be absent: {_FIXED_AUDIT_ROOT}")

    auditor_bytes, states = _capture_fixed_sources()
    attestation = _launcher_attestation(states)
    attestation_before = dict(attestation)
    namespace = _exec_captured_auditor(auditor_bytes)
    run_function = namespace["run_fixed_audit_v2"]
    result = run_function(mode, attestation)
    if attestation != attestation_before:
        raise FixedAuditLauncherError("Captured auditor mutated launcher attestation")
    result = _validate_result(mode, result)

    # These are the last source reads.  In audit mode no filesystem operation may
    # follow except the one fixed output-tree snapshot below.
    _rehash_fixed_sources(states)

    if mode == "validate-only":
        # The absence check is the final filesystem operation in validation mode.
        if _path_lexists(_FIXED_AUDIT_ROOT):
            raise FixedAuditLauncherError("Validate-only emitted an audit output")
        return result

    final_tree_function = namespace["final_audit_tree_v2"]
    # This call must remain the final filesystem operation in audit mode.  All
    # following work is an in-memory type check and equality comparison.
    observed_final_tree = final_tree_function()
    _validate_final_tree(observed_final_tree, "launcher final audit tree")
    if observed_final_tree != result["final_audit_tree"]:
        raise FixedAuditLauncherError("Audit tree changed after auditor return")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch the frozen v2 Lowry R1 H8 real-APP candidate audit. "
            "No path, digest, cap, parameter, or runtime override exists."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--audit", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    mode = "validate-only" if args.validate_only else "audit"
    result = run_fixed(mode)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if mode == "validate-only":
        print("HOLD: fixed inputs validate; no v2 audit artifact was emitted.")
    else:
        print(
            "GO only for separate data-only authority-record review; "
            "all runtime paths remain disabled."
        )


if __name__ == "__main__":
    main()
