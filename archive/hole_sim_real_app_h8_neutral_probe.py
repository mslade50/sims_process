"""Fixed, research-only neutral probe for the admitted Lowry R1 H8 APP support.

Only the public admission, readiness, weighting, and selection surface is used.
The committed receipt is intentionally summary-only: it contains no atom vector,
callback, chain, anchor, solve, simulation, or production capability.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import stat
import tempfile

import hole_sim_real_app_h8_admission_session as admission


PROBE_CONTRACT = "fixed_lowry_r1h8_real_app_neutral_probe_v1"
MANIFEST_CONTRACT = "fixed_lowry_r1h8_real_app_neutral_probe_manifest_v1"
OUTPUT_CONTRACT = "non_circular_output_sha256_v1"

_REPO = Path(__file__).resolve().parent.parent
_OUTPUT_RELATIVE = "permanent_data/hole_sim_real_app_h8_neutral_probe_v1"
_OUTPUT_ROOT = (
    _REPO
    / Path(_OUTPUT_RELATIVE)
)
_OUTPUT_FILES = frozenset(("probe.json", "manifest.json", "OUTPUT_SHA256.json"))

_AUTHORITY_ID = "1a9d6fd9133811799a74cc61fcdc0ccbeec47617da11ef4a993ce3b2dcd529ce"
_AUTHORITY_RECORD_SHA256 = (
    "4022bdd3413b94054cdca0ea32e2dd015666c2502af9ad21113d1584f3a439a7"
)
_AUTHORITY_LOCK_SHA256 = (
    "6b3e75849a44c4e6b880af09f7bac86e609bffb5bf64401c9ddf89fe25c30da5"
)
_WRAPPER_SHA256 = "55595e3ef2dfb58db64a0ecbf0563b9174c2b0e67cf9d689ce39d06b59771941"
_SELECTOR_RUNTIME_SHA256 = (
    "ab4607085ca3dfd75ee38deab7fddd5932be644aaac1dad5688622b4a452660c"
)
_SELECTOR_SUPPORT_SHA256 = (
    "f5fd780ea6c2a3e2795c8197f45f3486cd4a03606bac8e8c17b0548d7a07dc4c"
)
_PARAMETER_SHA256 = "74ce4c51233f644bae88af173beccb86b4bba0a97970a0c2842fb7d0c45e42c5"
_REWARD_REFERENCE_SHA256 = (
    "35ffdc1aa21ccc209ceb56a8b8e78b3118aa1cc8bd4c7372a92a0dc01b0bc68c"
)
_QUERY_KEY = "943f67faa5e345d395bf706e29c1a526e69879f80e14e81f55e255fd81dbf43e"
_SUPPORT_KEY = "app|raw|R1|H8|tee|B14|33204"
_SELECTION_MODE = "frozen_real_app_four_uniform_zero_path_v1"

_ZERO = "0x0.0p+0"
_HALF = "0x1.0000000000000p-1"
_HIGH = "0x1.fffffffffffffp-1"
_ONE_FSUM = "0x1.0000000000001p+0"
_TERMINAL = "0x1.0624dd2f1a9fcp-9"
_GROUP_MASS = (
    "0x1.c28f5c28f5c48p-3",
    "0x1.c6a7ef9db22c0p-3",
    "0x1.402bb0cf87d9fp-3",
    "0x1.a3d70a3d70a40p-3",
    "0x1.9af72015d866ep-4",
    "0x1.8a94d242e6bcep-4",
)
_DOWNSTREAM = {
    "callback_ready": False,
    "chain_ready": False,
    "anchor_ready": False,
    "mean_solve_ready": False,
    "simulation_ready": False,
    "production_ready": False,
    "research_only": True,
}
_UNIFORM_HEX = (
    (_ZERO, _ZERO, _ZERO, _ZERO),
    (_TERMINAL, _ZERO, _ZERO, _ZERO),
    (_TERMINAL, _HALF, _HALF, _HALF),
    (_TERMINAL, _HIGH, _HIGH, _HIGH),
)
_EXPECTED_SELECTIONS = (
    {
        "atom_id": "62118466441c40a23026610a15dcdb7106931319fe5409fd930ee89dd716666a",
        "complete_row_order": 35_217,
        "terminal": True,
        "proposal_row_order": 903,
        "surface_class_index": -1,
        "hazard_donor_order": -1,
        "stroke_cost": 1,
        "end_state_key": "terminal",
        "next_support_key": "",
        "terminal_residual_uniform_hex": _ZERO,
    },
    {
        "atom_id": "9523bab88b881a9b89d3e24388992bd3fb8957e8a4995f77910c0ff4d63b3531",
        "complete_row_order": 0,
        "terminal": False,
        "proposal_row_order": 0,
        "surface_class_index": 0,
        "hazard_donor_order": -1,
        "stroke_cost": 1,
        "end_state_key": "green|28.397756yd",
        "next_support_key": "709a42b8e03f77d149874397625f63254d0e39cc520d0388b7d1f8044a77ed39",
        "terminal_residual_uniform_hex": None,
    },
    {
        "atom_id": "7a9a50a13f9ccbe8e82a5d5e641a32bc2732be23699e396b4f6e95e6ad681490",
        "complete_row_order": 18_096,
        "terminal": False,
        "proposal_row_order": 464,
        "surface_class_index": 0,
        "hazard_donor_order": -1,
        "stroke_cost": 1,
        "end_state_key": "green|17.222403yd",
        "next_support_key": "e45bc87fab4c783197931ef776b5935c8a8a6a87cbb950e4cd3ce8a80a453f93",
        "terminal_residual_uniform_hex": None,
    },
    {
        "atom_id": "c2ad3416da08278211d7e11203c5226a8b1db7393f0675b59a6356cea278e0ec",
        "complete_row_order": 35_216,
        "terminal": False,
        "proposal_row_order": 902,
        "surface_class_index": 7,
        "hazard_donor_order": 33,
        "stroke_cost": 2,
        "end_state_key": "recovery|138.783789yd",
        "next_support_key": "9c191938eed83baf2be52dac8ad731fcd8de5f3b1bb423b065c944eadd9a27ca",
        "terminal_residual_uniform_hex": None,
    },
)


class NeutralProbeError(RuntimeError):
    """The fixed probe or its committed summary receipt changed."""


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _file_identity(raw: bytes) -> dict[str, object]:
    return {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NeutralProbeError(message)


def _exact_tree_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(
            _exact_tree_equal(left[key], right[key]) for key in left
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            _exact_tree_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _lstat_no_follow(path: Path, label: str):
    try:
        return os.lstat(path)
    except OSError as exc:
        raise NeutralProbeError(f"{label} cannot be inspected without following links") from exc


def _is_reparse(info: object) -> bool:
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    attributes = getattr(info, "st_file_attributes", 0)
    return bool(attributes & flag)


def _require_directory_no_follow(path: Path, label: str) -> None:
    info = _lstat_no_follow(path, label)
    _require(
        stat.S_ISDIR(info.st_mode)
        and not stat.S_ISLNK(info.st_mode)
        and not _is_reparse(info),
        f"{label} must be an exact non-symlink, non-reparse directory",
    )


def _require_regular_no_follow(path: Path, label: str) -> None:
    info = _lstat_no_follow(path, label)
    _require(
        stat.S_ISREG(info.st_mode)
        and not stat.S_ISLNK(info.st_mode)
        and not _is_reparse(info),
        f"{label} must be an exact regular non-symlink, non-reparse file",
    )


def _strict_json(raw: bytes, label: str) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value: str):
        raise ValueError(f"non-finite JSON value: {value}")

    try:
        text = raw.decode("utf-8")
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise NeutralProbeError(f"{label} must be strict finite duplicate-free JSON") from exc


def _selection_summary(result: object, uniforms_hex: tuple[str, ...]) -> dict[str, object]:
    residual = result.terminal_residual_uniform
    return {
        "semantic_uniforms_hex": list(uniforms_hex),
        "atom_id": result.atom_id,
        "complete_row_order": result.complete_row_order,
        "terminal": result.terminal,
        "proposal_row_order": result.proposal_row_order,
        "surface_class_index": result.surface_class_index,
        "hazard_donor_order": result.hazard_donor_order,
        "stroke_cost": result.stroke_cost,
        "end_state_key": result.end_state_key,
        "next_support_key": result.next_support_key,
        "terminal_probability_hex": result.terminal_probability.hex(),
        "terminal_residual_uniform_hex": None if residual is None else residual.hex(),
        "selection_mode": result.selection_mode,
        "hierarchical_atom_id": result.hierarchical_atom_id,
        "flat_atom_id": result.flat_atom_id,
        "flat_hierarchical_equal": result.flat_hierarchical_equal,
    }


def _assert_readiness(readiness: object) -> None:
    expected = {
        "authority_id": _AUTHORITY_ID,
        "authority_record_sha256": _AUTHORITY_RECORD_SHA256,
        "authority_lock_sha256": _AUTHORITY_LOCK_SHA256,
        "selector_runtime_sha256": _SELECTOR_RUNTIME_SHA256,
        "selector_support_sha256": _SELECTOR_SUPPORT_SHA256,
        "complete_atoms": 35_218,
        "proposals": 904,
        "spatial_proposals": 903,
        "terminal_atoms": 1,
        "real_support_admitted": True,
        "weighting_ready": True,
        "selection_ready": True,
    }
    for field, value in expected.items():
        _require(getattr(readiness, field) == value, f"readiness {field} changed")
    for field, value in _DOWNSTREAM.items():
        _require(getattr(readiness, field) is value, f"readiness {field} changed")


def _assert_weighted(weighted: object) -> None:
    _require(weighted.runtime_binding_sha256 == _SELECTOR_RUNTIME_SHA256, "runtime changed")
    _require(weighted.support_sha256 == _SELECTOR_SUPPORT_SHA256, "support changed")
    _require(weighted.parameter_sha256 == _PARAMETER_SHA256, "parameters changed")
    _require(math.fsum(weighted.probability).hex() == _ONE_FSUM, "probability mass changed")
    _require(
        math.fsum(weighted.neutral_probability_mass).hex() == _ONE_FSUM,
        "neutral mass changed",
    )
    _require(weighted.terminal_probability.hex() == _TERMINAL, "terminal mass changed")
    before = tuple(value.hex() for value in weighted.donor_group_mass_before)
    after = tuple(value.hex() for value in weighted.donor_group_mass_after)
    _require(before == _GROUP_MASS, "donor group mass before changed")
    _require(after == _GROUP_MASS, "donor group mass after changed")
    _require(max(abs(a - b) for a, b in zip(
        weighted.probability, weighted.neutral_probability_mass, strict=True
    )).hex() == _ZERO, "atom mass delta changed")
    _require(max(abs(a - b) for a, b in zip(
        weighted.donor_group_mass_before, weighted.donor_group_mass_after, strict=True
    )).hex() == _ZERO, "group mass delta changed")
    _require(weighted.clipped_donor_logs == 0, "donor clipping changed")
    _require(weighted.clipped_target_logs == 0, "target clipping changed")
    _require(weighted.zero_fast_path is True, "neutral zero path changed")
    _require(weighted.selection_mode == _SELECTION_MODE, "selection mode changed")


def _assert_selections(selections: list[dict[str, object]]) -> None:
    _require(len(selections) == 4, "selection count changed")
    for observed, expected, uniforms in zip(
        selections, _EXPECTED_SELECTIONS, _UNIFORM_HEX, strict=True
    ):
        for field, value in expected.items():
            _require(observed[field] == value, f"selection {field} changed")
        _require(observed["semantic_uniforms_hex"] == list(uniforms), "uniforms changed")
        _require(observed["terminal_probability_hex"] == _TERMINAL, "selection terminal changed")
        _require(observed["selection_mode"] == _SELECTION_MODE, "selection mode changed")
        _require(observed["hierarchical_atom_id"] == observed["atom_id"], "hierarchy changed")
        _require(observed["flat_atom_id"] == observed["atom_id"], "flat selector changed")
        _require(observed["flat_hierarchical_equal"] is True, "selector equality changed")


def _probe_payload(weighted: object, selections: list[dict[str, object]]) -> dict[str, object]:
    probability_delta = max(abs(a - b) for a, b in zip(
        weighted.probability, weighted.neutral_probability_mass, strict=True
    ))
    group_delta = max(abs(a - b) for a, b in zip(
        weighted.donor_group_mass_before, weighted.donor_group_mass_after, strict=True
    ))
    return {
        "contract": PROBE_CONTRACT,
        "authority": {
            "authority_id": _AUTHORITY_ID,
            "authority_record_sha256": _AUTHORITY_RECORD_SHA256,
            "authority_lock_sha256": _AUTHORITY_LOCK_SHA256,
            "wrapper_sha256": _WRAPPER_SHA256,
            "selector_runtime_sha256": _SELECTOR_RUNTIME_SHA256,
            "selector_support_sha256": _SELECTOR_SUPPORT_SHA256,
        },
        "scope": {
            "query_key": _QUERY_KEY,
            "support_key": _SUPPORT_KEY,
            "target_player_id": "33204",
            "complete_atoms": 35_218,
            "proposals": 904,
            "spatial_proposals": 903,
            "terminal_atoms": 1,
        },
        "parameters": {
            "parameter_sha256": _PARAMETER_SHA256,
            "mode": "neutral_zero_fast_path",
        },
        "masses": {
            "probability_fsum_hex": math.fsum(weighted.probability).hex(),
            "neutral_probability_fsum_hex": math.fsum(
                weighted.neutral_probability_mass
            ).hex(),
            "terminal_probability_hex": weighted.terminal_probability.hex(),
            "donor_group_mass_before_hex": [
                value.hex() for value in weighted.donor_group_mass_before
            ],
            "donor_group_mass_after_hex": [
                value.hex() for value in weighted.donor_group_mass_after
            ],
            "max_atom_probability_delta_hex": probability_delta.hex(),
            "max_donor_group_delta_hex": group_delta.hex(),
            "clipped_donor_logs": weighted.clipped_donor_logs,
            "clipped_target_logs": weighted.clipped_target_logs,
        },
        "selections": selections,
        "evaluation_counts": {
            "admission": 1,
            "readiness": 1,
            "explicit_weight": 1,
            "select": 4,
            "underlying_weight_evaluations": 5,
        },
        "downstream": dict(_DOWNSTREAM),
    }


def _manifest(probe_raw: bytes) -> dict[str, object]:
    return {
        "contract": MANIFEST_CONTRACT,
        "artifact": "hole_sim_real_app_h8_neutral_probe_v1",
        "status": "verified_research_only_no_downstream_readiness",
        "payload_files": {"probe.json": _file_identity(probe_raw)},
        **_DOWNSTREAM,
    }


def _validate_payload(payload: object) -> dict[str, object]:
    _require(type(payload) is dict, "probe payload type changed")
    expected_fields = {
        "contract",
        "authority",
        "scope",
        "parameters",
        "masses",
        "selections",
        "evaluation_counts",
        "downstream",
    }
    _require(set(payload) == expected_fields, "probe payload fields changed")
    _require(_exact_tree_equal(payload["contract"], PROBE_CONTRACT), "probe contract changed")
    _require(
        _exact_tree_equal(payload["authority"], {
            "authority_id": _AUTHORITY_ID,
            "authority_record_sha256": _AUTHORITY_RECORD_SHA256,
            "authority_lock_sha256": _AUTHORITY_LOCK_SHA256,
            "wrapper_sha256": _WRAPPER_SHA256,
            "selector_runtime_sha256": _SELECTOR_RUNTIME_SHA256,
            "selector_support_sha256": _SELECTOR_SUPPORT_SHA256,
        }),
        "probe authority changed",
    )
    _require(
        _exact_tree_equal(payload["scope"], {
            "query_key": _QUERY_KEY,
            "support_key": _SUPPORT_KEY,
            "target_player_id": "33204",
            "complete_atoms": 35_218,
            "proposals": 904,
            "spatial_proposals": 903,
            "terminal_atoms": 1,
        }),
        "probe scope changed",
    )
    _require(
        _exact_tree_equal(payload["parameters"], {
            "parameter_sha256": _PARAMETER_SHA256,
            "mode": "neutral_zero_fast_path",
        }),
        "probe parameters changed",
    )
    _require(
        _exact_tree_equal(payload["masses"], {
            "probability_fsum_hex": _ONE_FSUM,
            "neutral_probability_fsum_hex": _ONE_FSUM,
            "terminal_probability_hex": _TERMINAL,
            "donor_group_mass_before_hex": list(_GROUP_MASS),
            "donor_group_mass_after_hex": list(_GROUP_MASS),
            "max_atom_probability_delta_hex": _ZERO,
            "max_donor_group_delta_hex": _ZERO,
            "clipped_donor_logs": 0,
            "clipped_target_logs": 0,
        }),
        "probe masses changed",
    )
    _require(
        _exact_tree_equal(payload["evaluation_counts"], {
            "admission": 1,
            "readiness": 1,
            "explicit_weight": 1,
            "select": 4,
            "underlying_weight_evaluations": 5,
        }),
        "probe evaluation counts changed",
    )
    _require(
        _exact_tree_equal(payload["downstream"], _DOWNSTREAM),
        "probe downstream scope changed",
    )
    selections = payload["selections"]
    _require(type(selections) is list and len(selections) == 4, "probe selections changed")
    _assert_selections(selections)
    return payload


def _bundle_bytes(payload: dict[str, object]) -> dict[str, bytes]:
    probe_raw = _json_bytes(payload)
    manifest_raw = _json_bytes(_manifest(probe_raw))
    output_raw = _json_bytes(
        {
            "contract": OUTPUT_CONTRACT,
            "files": {
                "probe.json": _file_identity(probe_raw),
                "manifest.json": _file_identity(manifest_raw),
            },
        }
    )
    return {
        "probe.json": probe_raw,
        "manifest.json": manifest_raw,
        "OUTPUT_SHA256.json": output_raw,
    }


def _validate_bundle(root: Path) -> dict[str, object]:
    _require_directory_no_follow(root, "neutral probe output root")
    try:
        observed = set(os.listdir(root))
    except OSError as exc:
        raise NeutralProbeError("neutral probe output root cannot be listed") from exc
    _require(observed == _OUTPUT_FILES, "neutral probe output closure changed")
    for name in observed:
        _require_regular_no_follow(root / name, f"neutral probe output {name}")
    try:
        raw = {name: (root / name).read_bytes() for name in observed}
    except OSError as exc:
        raise NeutralProbeError("neutral probe output cannot be read") from exc
    for name in observed:
        _require_regular_no_follow(root / name, f"neutral probe output {name}")
    payload = _strict_json(raw["probe.json"], "probe.json")
    manifest = _strict_json(raw["manifest.json"], "manifest.json")
    output = _strict_json(raw["OUTPUT_SHA256.json"], "OUTPUT_SHA256.json")
    _require(raw["probe.json"] == _json_bytes(payload), "probe encoding changed")
    _require(raw["manifest.json"] == _json_bytes(manifest), "manifest encoding changed")
    _require(raw["OUTPUT_SHA256.json"] == _json_bytes(output), "output encoding changed")
    _require(
        _exact_tree_equal(manifest, _manifest(raw["probe.json"])),
        "manifest changed",
    )
    expected_output = {
        "contract": OUTPUT_CONTRACT,
        "files": {
            name: _file_identity(raw[name]) for name in ("probe.json", "manifest.json")
        },
    }
    _require(_exact_tree_equal(output, expected_output), "output seal changed")
    return _validate_payload(payload)


def validate_only() -> dict[str, object]:
    """Validate the fixed output without admitting support; absence is ready."""

    if not _lexists(_OUTPUT_ROOT):
        return {
            "contract": PROBE_CONTRACT,
            "status": "ready_to_materialize",
            "output": _OUTPUT_RELATIVE,
        }
    payload = _validate_bundle(_OUTPUT_ROOT)
    return {"contract": PROBE_CONTRACT, "status": "validated", "probe": payload}


def _write_bundle_atomically(bundle: dict[str, bytes]) -> None:
    _OUTPUT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{_OUTPUT_ROOT.name}.", dir=_OUTPUT_ROOT.parent)
    )
    for name in ("probe.json", "manifest.json", "OUTPUT_SHA256.json"):
        with (temporary / name).open("xb") as handle:
            handle.write(bundle[name])
            handle.flush()
            os.fsync(handle.fileno())
    _validate_bundle(temporary)
    if _lexists(_OUTPUT_ROOT):
        raise NeutralProbeError(
            f"neutral probe output already exists; failed temporary evidence left at {temporary}"
        )
    temporary.rename(_OUTPUT_ROOT)


def materialize() -> dict[str, object]:
    """Run the one fixed neutral probe and atomically create its receipt."""

    if _lexists(_OUTPUT_ROOT):
        raise NeutralProbeError("neutral probe output already exists")

    parameters = admission.RealAppWeightingParameters(
        transfer_beta=(0.0, 0.0, 0.0, 0.0),
        target_ability_sg=(0.0, 0.0, 0.0, 0.0),
        lambda_correction=(0.0, 0.0, 0.0, 0.0),
        reward_reference_sha256=_REWARD_REFERENCE_SHA256,
    )
    uniforms = tuple(
        admission.RealAppSemanticUniforms(*(float.fromhex(value) for value in row))
        for row in _UNIFORM_HEX
    )
    session = admission.admit_lowry_r1h8_app_support()
    readiness = session.readiness
    weighted = session.weight(parameters)
    selected = [session.select(parameters, value) for value in uniforms]

    _assert_readiness(readiness)
    _assert_weighted(weighted)
    selections = [
        _selection_summary(result, uniform_hex)
        for result, uniform_hex in zip(selected, _UNIFORM_HEX, strict=True)
    ]
    _assert_selections(selections)
    payload = _probe_payload(weighted, selections)
    _write_bundle_atomically(_bundle_bytes(payload))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--materialize", action="store_true")
    arguments = parser.parse_args()
    result = validate_only() if arguments.validate_only else materialize()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


__all__ = ("NeutralProbeError", "validate_only", "materialize", "main")


if __name__ == "__main__":
    raise SystemExit(main())
