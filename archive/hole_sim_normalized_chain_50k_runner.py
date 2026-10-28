"""Research-only, admission-gated 50,000-particle shot-chain runner.

This is deliberately a thin orchestration layer around the frozen streaming
kernel.  It cannot score predictions, calibrate a production mean, resume a
partial run, or run any particle count other than 50,000.  Before simulation
it verifies a complete authoritative 5k artifact and an externally supplied
hash lock that binds the artifact, this runner, the executable evaluator
contract, the streaming source, and every dependency.

The process is continuously RSS-monitored from admission through final output
hashing.  Crossing four GiB interrupts the main thread and fails the run.
The shared tail policy permits every exact physical score that holes within
18 continuation transitions; no score cap, clipping, or terminal-by-cap is
allowed.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import _thread
import json
from pathlib import Path
import re
import sys
import threading
import time
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import psutil


ARCHIVE = Path(__file__).resolve().parent
REPO = ARCHIVE.parent
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

import hole_sim_normalized_chain_streaming as stream  # noqa: E402
import evaluate_normalized_chain_50k as evaluator  # noqa: E402
import hole_sim_tail_policy as tail_policy  # noqa: E402


RUNNER_VERSION = "normalized_chain_50k_engineering_v3_unbounded_score_tail"
LOCK_SCHEMA_VERSION = "normalized_chain_50k_approval_lock_v4"
PARTICLES = 50_000
PREFIX_WIDE_PARTICLES = 100_000
DEFAULT_CHUNK_SIZE = 1_000
MIN_CHUNK_SIZE = 64
MAX_CHUNK_SIZE = 5_000
RSS_CAP_BYTES = 4 * 2**30
RSS_SAMPLE_SECONDS = 0.02
EXPECTED_ARMS = ("raw", "normalized")
EXPECTED_ROUNDS = (1, 2, 3, 4)
EXPECTED_HOLES = tuple(range(1, 19))
EXPECTED_TRANSITION_INDICES = tuple(
    range(tail_policy.MAX_CONTINUATION_TRANSITIONS)
)
EXPECTED_STREAMS = (
    ("initial", "tee_index"),
    ("initial", "tee_lie"),
    ("initial", "app_index"),
    ("initial", "app_holeout"),
    ("initial", "app_lie"),
    ("transition", "hazard"),
    ("transition", "tee_index"),
    ("transition", "tee_lie"),
    ("transition", "app_index"),
    ("transition", "app_holeout"),
    ("transition", "app_lie"),
    ("transition", "state_arg"),
    ("transition", "state_putt"),
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _require_sha(value: object, label: str) -> str:
    text = str(value)
    if not HEX64.fullmatch(text):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return text


def _as_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be an integer") from error
    if isinstance(value, float) and value != result:
        raise ValueError(f"{label} must be an integer")
    return result


def validate_cli_contract(args: argparse.Namespace) -> None:
    """Reject any scale, chunk, or output mode outside the approved contract."""

    if _as_int(args.particles, "particles") != PARTICLES:
        raise ValueError(
            f"This runner accepts exactly {PARTICLES:,} particles; "
            f"received {args.particles!r}"
        )
    chunk_size = _as_int(args.chunk_size, "chunk_size")
    if not MIN_CHUNK_SIZE <= chunk_size <= MAX_CHUNK_SIZE:
        raise ValueError(
            f"chunk_size must be in [{MIN_CHUNK_SIZE}, {MAX_CHUNK_SIZE}]"
        )
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise FileExistsError(
                f"Refusing non-empty or non-directory output path: {output_dir}"
            )
    if not Path(args.authoritative_5k_dir).is_dir():
        raise FileNotFoundError(args.authoritative_5k_dir)
    if not Path(args.approval_lock).is_file():
        raise FileNotFoundError(args.approval_lock)
    resolved_output = output_dir.resolve()
    resolved_5k = Path(args.authoritative_5k_dir).resolve()
    _require(
        resolved_output != resolved_5k
        and resolved_5k not in resolved_output.parents,
        "Output path must not be the authoritative 5k artifact or its child",
    )


@dataclass(frozen=True)
class ApprovalLock:
    path: Path
    sha256: str
    authoritative_5k_manifest_sha256: str
    authoritative_5k_output_index_sha256: str
    runner_source_sha256: str
    evaluator_source_sha256: str
    streaming_source_sha256: str
    dependency_sha256: dict[str, str]
    tail_policy: dict[str, object]


def load_approval_lock(path: Path) -> ApprovalLock:
    """Load a strict, closed-schema hash lock supplied after 5k review."""

    path = Path(path)
    payload = _read_json(path)
    expected_keys = {
        "schema_version",
        "authoritative_5k_manifest_sha256",
        "authoritative_5k_output_index_sha256",
        "runner_source_sha256",
        "evaluator_source_sha256",
        "streaming_source_sha256",
        "dependency_sha256",
        "tail_policy",
    }
    _require(
        set(payload) == expected_keys,
        "Approval lock fields differ from the closed approval-lock schema",
    )
    _require(
        payload["schema_version"] == LOCK_SCHEMA_VERSION,
        "Approval lock schema version mismatch",
    )
    dependencies = payload["dependency_sha256"]
    _require(isinstance(dependencies, dict), "dependency_sha256 must be an object")
    expected_dependency_names = set(stream.dependency_paths())
    _require(
        set(dependencies) == expected_dependency_names,
        "Approval lock dependency names do not exactly match the streaming graph",
    )
    checked_dependencies = {
        str(name): _require_sha(value, f"dependency_sha256.{name}")
        for name, value in dependencies.items()
    }
    tail_policy.validate_manifest_policy(
        payload["tail_policy"], label="approval_lock.tail_policy"
    )
    return ApprovalLock(
        path=path.resolve(),
        sha256=stream.sha256_file(path),
        authoritative_5k_manifest_sha256=_require_sha(
            payload["authoritative_5k_manifest_sha256"],
            "authoritative_5k_manifest_sha256",
        ),
        authoritative_5k_output_index_sha256=_require_sha(
            payload["authoritative_5k_output_index_sha256"],
            "authoritative_5k_output_index_sha256",
        ),
        runner_source_sha256=_require_sha(
            payload["runner_source_sha256"], "runner_source_sha256"
        ),
        evaluator_source_sha256=_require_sha(
            payload["evaluator_source_sha256"], "evaluator_source_sha256"
        ),
        streaming_source_sha256=_require_sha(
            payload["streaming_source_sha256"], "streaming_source_sha256"
        ),
        dependency_sha256=checked_dependencies,
        tail_policy=dict(payload["tail_policy"]),
    )


def _validate_complete_checksum_index(root: Path, expected_sha256: str) -> int:
    index_path = root / "OUTPUT_SHA256.json"
    _require(index_path.is_file(), "Authoritative 5k checksum index is missing")
    _require(
        stream.sha256_file(index_path) == expected_sha256,
        "Authoritative 5k checksum index is not the approved index",
    )
    index = _read_json(index_path)
    actual = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and path != index_path
    }
    _require(
        set(index) == set(actual),
        "Authoritative 5k checksum index is not complete or has stale entries",
    )
    for relative, path in actual.items():
        record = index[relative]
        _require(isinstance(record, dict), f"Bad checksum record: {relative}")
        _require(
            set(record) == {"sha256", "bytes"},
            f"Bad checksum fields: {relative}",
        )
        _require(
            _as_int(record["bytes"], f"{relative}.bytes") == path.stat().st_size,
            f"Authoritative 5k byte-size mismatch: {relative}",
        )
        _require(
            stream.sha256_file(path)
            == _require_sha(record["sha256"], f"{relative}.sha256"),
            f"Authoritative 5k hash mismatch: {relative}",
        )
    return len(actual)


def _all_zero_mismatches(value: object, label: str) -> None:
    _require(isinstance(value, dict), f"{label} is not an object")
    mismatches = value.get("mismatches")
    _require(isinstance(mismatches, dict) and mismatches, f"{label} has no mismatch map")
    _require(
        all(_as_int(count, f"{label}.{name}") == 0 for name, count in mismatches.items()),
        f"{label} did not pass exact parity",
    )


def _validate_5k_gate(gate: Mapping[str, object]) -> None:
    for name in (
        "legacy_terminal", "legacy_initializer_atom_ids",
        "legacy_initializer_physical_state",
    ):
        _all_zero_mismatches(gate.get(name), name)
        _require(
            _as_int(gate[name].get("rows"), f"{name}.rows") == 2_304,
            f"{name} did not cover 18 holes x 128 particles",
        )
    chunks = gate.get("counter_chunk_invariance")
    _require(isinstance(chunks, dict), "5k gate has no counter chunk audit")
    _require(set(chunks) == set(EXPECTED_ARMS), "5k gate chunk arms differ")
    for arm in EXPECTED_ARMS:
        _all_zero_mismatches(chunks[arm], f"counter_chunk_invariance.{arm}")
        _require(
            _as_int(chunks[arm].get("rows"), f"counter_chunk_invariance.{arm}.rows")
            == 9_216,
            f"5k {arm} counter gate did not cover R1-R4 x 18 x 128",
        )
    prefix = gate.get("counter_50k_prefix_of_100k")
    _require(isinstance(prefix, dict), "5k gate has no counter prefix audit")
    _require(prefix.get("exact") is True, "5k counter prefix audit is not exact")
    _require(
        tuple(prefix.get("transition_indices", ()))
        == EXPECTED_TRANSITION_INDICES,
        "5k prefix audit does not cover transitions 0 through 17",
    )
    _require(
        _as_int(prefix.get("max_transition_index"), "5k prefix max transition")
        == tail_policy.MAX_CONTINUATION_TRANSITIONS - 1,
        "5k prefix audit does not reach transition 17",
    )
    hashes = prefix.get("streams")
    _require(isinstance(hashes, dict), "5k counter prefix stream hashes missing")
    expected_names = {f"{family}:{name}" for family, name in EXPECTED_STREAMS}
    _require(set(hashes) == expected_names, "5k prefix stream set differs")
    for name, value in hashes.items():
        _require_sha(value, f"5k prefix stream {name}")
    _require(
        _as_int(gate.get("unsupported_state_rows"), "5k unsupported rows") == 0,
        "5k gate has unsupported states",
    )
    _require(gate.get("survivor_filtering") is False, "5k gate filtered survivors")
    tail_policy.validate_manifest_policy(
        gate.get("tail_policy"), label="5k gate tail_policy"
    )
    regression = gate.get("known_tail_regression")
    _require(isinstance(regression, dict), "5k known-tail regression missing")
    _require(regression.get("exact") is True, "5k known-tail regression failed")
    _require(
        regression.get("particle_ids")
        == list(sorted(stream.KNOWN_TAIL_REGRESSION)),
        "5k known-tail regression particle IDs differ",
    )
    _require(
        regression.get("observed_hole_strokes")
        == [
            stream.KNOWN_TAIL_REGRESSION[particle_id]
            for particle_id in sorted(stream.KNOWN_TAIL_REGRESSION)
        ],
        "5k known-tail regression scores differ",
    )
    _require(
        regression.get("singleton_chunk_exact") is True,
        "5k known-tail regression is not chunk invariant",
    )
    _require(
        regression.get("legacy_score_cap_status_emitted") is False,
        "5k known-tail regression emitted the legacy cap status",
    )
    tail_policy.validate_manifest_policy(
        regression.get("tail_policy"),
        label="5k known-tail regression tail_policy",
    )
    inventory = gate.get("counter_support_inventory")
    _require(isinstance(inventory, dict), "5k counter support inventory missing")
    _require(set(inventory) == set(EXPECTED_ARMS), "5k counter support arms differ")
    for arm in EXPECTED_ARMS:
        _require(
            isinstance(inventory[arm], dict),
            f"5k {arm} counter support inventory is not an object",
        )
        _require(
            _as_int(inventory[arm].get("second_replay_new_calls"), f"5k {arm} replay calls") == 0,
            f"5k {arm} counter replay materialized supports",
        )
        _require(
            _as_int(inventory[arm].get("second_replay_new_misses"), f"5k {arm} replay misses") == 0,
            f"5k {arm} counter replay missed supports",
        )


def _validate_hash_mapping(value: object, label: str) -> dict[str, str]:
    _require(isinstance(value, dict) and value, f"{label} must be a non-empty object")
    return {
        str(name): _require_sha(digest, f"{label}.{name}")
        for name, digest in value.items()
    }


def _validate_producer_contract(manifest: Mapping[str, object]) -> dict[str, object]:
    """Validate and normalize evaluator-facing fields from the reviewed 5k."""

    tail_policy.validate_manifest_policy(
        manifest.get("tail_policy"), label="5k manifest tail_policy"
    )
    reference_provenance = manifest.get("reference_precompute_provenance")
    expected_reference_provenance = stream.reference_precompute_provenance(
        stream.FROZEN_DEPENDENCY_SHA256["precompute_source"]
    )
    _require(
        reference_provenance == expected_reference_provenance,
        "5k reference precompute provenance differs from the frozen graph",
    )

    normalization_mode = str(manifest.get("normalization_mode"))
    _require(
        normalization_mode == evaluator.NORMALIZATION_MODE,
        "5k normalization_mode is not evaluator-compatible",
    )
    try:
        raw_beta = manifest.get("ability_log_weight_per_sg2")
        _require(not isinstance(raw_beta, bool), "5k beta cannot be boolean")
        beta = float(raw_beta)
    except (TypeError, ValueError) as error:
        raise ValueError("5k ability_log_weight_per_sg2 is not numeric") from error
    _require(beta == 0.0, "5k ability log weight must be exactly beta zero")

    streaming_contract = manifest.get("streaming_contract")
    _require(isinstance(streaming_contract, dict), "5k streaming contract missing")
    _require(
        streaming_contract.get("survivor_filtering") is False,
        "5k streamed survivor filtering",
    )
    _require(
        streaming_contract.get("unsupported_states_fail_closed") is True,
        "5k did not fail closed",
    )
    required_tail_streaming = {
        "max_continuation_transitions": (
            tail_policy.MAX_CONTINUATION_TRANSITIONS
        ),
        "physical_hole_score_cap": None,
        "terminal_by_score_cap": False,
        "hole_score_clipping": False,
        "high_score_diagnostic_strokes": (
            tail_policy.HIGH_SCORE_DIAGNOSTIC_STROKES
        ),
        "high_score_diagnostic_only": True,
        "legacy_score_cap_status_emitted": False,
    }
    for name, expected in required_tail_streaming.items():
        _require(
            name in streaming_contract
            and streaming_contract[name] == expected,
            f"5k streaming tail contract differs at {name}",
        )
    carried_streaming_contract = {
        "survivor_filtering": streaming_contract["survivor_filtering"],
        "unsupported_states_fail_closed": streaming_contract[
            "unsupported_states_fail_closed"
        ],
        **required_tail_streaming,
    }

    rng_modes = manifest.get("rng_modes")
    _require(isinstance(rng_modes, dict), "5k RNG manifest missing")
    _require(rng_modes.get("benchmark") == "counter", "5k benchmark did not use counter RNG")
    _require(rng_modes.get("arm_in_key") is False, "5k RNG keyed on arm")
    _require(rng_modes.get("chunk_in_key") is False, "5k RNG keyed on chunk")
    counter_version = rng_modes.get("counter_version")
    _require(
        isinstance(counter_version, str) and counter_version.strip(),
        "5k counter RNG version is absent",
    )
    _require(
        counter_version == stream.hook.RNG_VERSION,
        "5k counter RNG version differs from the frozen streaming dependency",
    )
    carried_rng_modes = {
        "benchmark": rng_modes["benchmark"],
        "counter_version": counter_version,
        "arm_in_key": rng_modes["arm_in_key"],
        "chunk_in_key": rng_modes["chunk_in_key"],
    }

    transform_order = tuple(map(str, manifest.get("transform_order", ())))
    _require(
        transform_order == evaluator.REQUIRED_TRANSFORM_ORDER,
        "5k transform order is not evaluator-compatible",
    )
    alignment_count = _as_int(manifest.get("alignment_count"), "5k alignment_count")
    _require(alignment_count == 1, "5k alignment count must equal one")
    mass_contract = str(manifest.get("mass_contract"))
    _require(
        mass_contract == "base_hierarchy_mass_times_mean_one_model_weight",
        "5k mass contract is not evaluator-compatible",
    )
    return {
        "normalization_mode": normalization_mode,
        "ability_log_weight_per_sg2": beta,
        "streaming_contract": carried_streaming_contract,
        "rng_modes": carried_rng_modes,
        "transform_order": list(transform_order),
        "alignment_count": alignment_count,
        "mass_contract": mass_contract,
        "tail_policy": tail_policy.manifest_policy(),
        "reference_precompute_provenance": expected_reference_provenance,
    }


def _validate_5k_totals(
    path: Path, *, expected_event_key: str, expected_player_id: str
) -> None:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "arm_names", "round_totals", "round_numbers", "particle_ids",
            "player_id", "event_key",
        }
        _require(set(payload.files) == required, "5k totals NPZ schema differs")
        arms = tuple(map(str, payload["arm_names"].tolist()))
        totals = payload["round_totals"]
        _require(arms == EXPECTED_ARMS, "5k totals arm order differs")
        _require(totals.dtype == np.int16, "5k totals are not int16")
        _require(totals.shape == (2, 4, 5_000), "5k totals axes differ")
        np.testing.assert_array_equal(payload["round_numbers"], EXPECTED_ROUNDS)
        np.testing.assert_array_equal(payload["particle_ids"], np.arange(5_000))
        tail_policy.round_totals_int16(totals, label="5k round totals")
        _require(payload["player_id"].shape == (1,), "5k player axis differs")
        _require(payload["event_key"].shape == (1,), "5k event axis differs")
        _require(
            str(payload["player_id"][0]) == str(expected_player_id),
            "5k NPZ player_id disagrees with its manifest",
        )
        _require(
            str(payload["event_key"][0]) == str(expected_event_key),
            "5k NPZ event_key disagrees with its manifest",
        )


def _validate_round_summary(
    frame: pd.DataFrame, *, particles: int, label: str
) -> None:
    required = {"arm", "round_no", "particles"}
    _require(required.issubset(frame), f"{label} round summary columns are incomplete")
    _require(len(frame) == 8, f"{label} round summary must contain exactly eight rows")
    _require(
        not frame[["arm", "round_no"]].duplicated().any(),
        f"{label} round summary axes duplicate",
    )
    observed = {
        (str(row.arm), int(row.round_no))
        for row in frame.itertuples(index=False)
    }
    expected = {
        (arm, round_no)
        for arm in EXPECTED_ARMS
        for round_no in EXPECTED_ROUNDS
    }
    _require(observed == expected, f"{label} round summary axes differ")
    _require(
        frame["particles"].map(int).eq(int(particles)).all(),
        f"{label} round summary particle count differs",
    )


def _validate_hole_diagnostics(frame: pd.DataFrame, particles: int, label: str) -> None:
    required = {
        "arm", "round_no", "hole_no", "particles",
        "unsupported_state_rows", "max_transition_count",
        "high_score_diagnostic_particles",
    }
    _require(required.issubset(frame), f"{label} diagnostics columns are incomplete")
    keys = frame[["arm", "round_no", "hole_no"]].copy()
    _require(not keys.duplicated().any(), f"{label} diagnostics axes are duplicated")
    observed = {
        (str(row.arm), int(row.round_no), int(row.hole_no))
        for row in keys.itertuples(index=False)
    }
    expected = {
        (arm, round_no, hole_no)
        for arm in EXPECTED_ARMS
        for round_no in EXPECTED_ROUNDS
        for hole_no in EXPECTED_HOLES
    }
    _require(observed == expected, f"{label} diagnostics do not cover both 4x18 axes")
    _require(
        frame["particles"].map(int).eq(int(particles)).all(),
        f"{label} diagnostics particle count differs",
    )
    _require(
        frame["unsupported_state_rows"].map(int).eq(0).all(),
        f"{label} diagnostics contain unsupported states",
    )
    _require(
        frame["max_transition_count"].map(int).between(
            0, tail_policy.MAX_CONTINUATION_TRANSITIONS
        ).all(),
        f"{label} diagnostics exceed the transition guard",
    )
    _require(
        frame["high_score_diagnostic_particles"].map(int).between(
            0, int(particles)
        ).all(),
        f"{label} high-score diagnostic counts are invalid",
    )


def _validate_5k_support_cache(
    root: Path, benchmark: Mapping[str, object]
) -> None:
    cache_metadata = benchmark.get("numeric_support_cache")
    _require(isinstance(cache_metadata, dict), "5k numeric support cache metadata missing")
    _require(set(cache_metadata) == set(EXPECTED_ARMS), "5k support cache arms differ")
    for arm in EXPECTED_ARMS:
        arm_root = root / "support_cache" / arm
        manifest_path = arm_root / "manifest.json"
        inventory_path = arm_root / "inventory.csv"
        _require(manifest_path.is_file(), f"5k {arm} support manifest missing")
        _require(inventory_path.is_file(), f"5k {arm} support inventory missing")
        sidecar = _read_json(manifest_path)
        _require(sidecar == cache_metadata[arm], f"5k {arm} support metadata differs")
        _require(sidecar.get("arm") == arm, f"5k {arm} support arm differs")
        _require(_as_int(sidecar.get("supports"), f"5k {arm} supports") > 0, f"5k {arm} cache is empty")
        _require(_as_int(sidecar.get("atoms"), f"5k {arm} atoms") > 0, f"5k {arm} cache has no atoms")
        _require(
            stream.sha256_file(inventory_path)
            == _require_sha(sidecar.get("inventory_sha256"), f"5k {arm} inventory hash"),
            f"5k {arm} inventory hash differs",
        )
        inventory = pd.read_csv(inventory_path)
        required = {
            "key", "file", "atoms", "source_cache_sha256",
            "payload_sha256", "file_sha256", "file_bytes",
        }
        _require(set(inventory) == required, f"5k {arm} support inventory schema differs")
        _require(not inventory["key"].duplicated().any(), f"5k {arm} support keys duplicate")
        _require(not inventory["file"].duplicated().any(), f"5k {arm} support files duplicate")
        _require(len(inventory) == _as_int(sidecar["supports"], f"5k {arm} supports"), f"5k {arm} support count differs")
        _require(inventory["atoms"].map(int).sum() == _as_int(sidecar["atoms"], f"5k {arm} atoms"), f"5k {arm} atom count differs")
        _require(inventory["file_bytes"].map(int).sum() == _as_int(sidecar["serialized_bytes"], f"5k {arm} bytes"), f"5k {arm} byte count differs")
        for row in inventory.itertuples(index=False):
            support_path = arm_root / str(row.file)
            _require(support_path.is_file(), f"5k {arm} support payload missing: {row.file}")
            _require(support_path.name == str(row.file), f"5k {arm} support file escapes cache root")
            _require(support_path.stat().st_size == int(row.file_bytes), f"5k {arm} support byte size differs: {row.file}")
            _require(stream.sha256_file(support_path) == _require_sha(row.file_sha256, f"5k {arm} {row.file} hash"), f"5k {arm} support hash differs: {row.file}")


def validate_authoritative_5k(
    root: Path,
    approval: ApprovalLock,
    *,
    verify_all_files: bool = True,
) -> dict[str, object]:
    """Fail closed unless ``root`` is a complete, approved 5k artifact."""

    root = Path(root)
    manifest_path = root / "manifest.json"
    _require(manifest_path.is_file(), "Authoritative 5k manifest is missing")
    _require(
        stream.sha256_file(manifest_path)
        == approval.authoritative_5k_manifest_sha256,
        "Authoritative 5k manifest is not the approved manifest",
    )
    files_verified = (
        _validate_complete_checksum_index(
            root, approval.authoritative_5k_output_index_sha256
        )
        if verify_all_files else 0
    )
    manifest = _read_json(manifest_path)
    required_files = {
        "OUTPUT_SHA256.json", "manifest.json", "raw_128_gate.json",
        "hazard_scalar_oracle.json", "raw_128_parity_particles.parquet",
        "round_totals_int16.npz", "round_summary.csv",
        "hole_diagnostics.csv", "benchmark.json", "REPORT.md",
    }
    _require(
        all((root / name).is_file() for name in required_files),
        "Authoritative 5k artifact is incomplete",
    )
    _require(manifest.get("artifact") == stream.STREAMING_VERSION, "5k artifact type differs")
    _require(manifest.get("research_only") is True, "5k artifact is not research-only")
    _require(manifest.get("production_touched") is False, "5k artifact touched production")
    _require(
        manifest.get("production_mean_anchor_status")
        == "not_calibrated_mechanics_only",
        "5k artifact claims a calibrated production mean",
    )
    _require(manifest.get("production_skill_embedded") is False, "5k embedded production skill")
    _require(manifest.get("posthoc_recentering_applied") is False, "5k used post-hoc recentering")
    tail_policy.validate_manifest_policy(
        manifest.get("tail_policy"), label="5k manifest tail_policy"
    )
    _require(
        manifest.get("tail_policy") == approval.tail_policy,
        "5k tail policy differs from approval lock",
    )
    _require(manifest.get("arms") == list(EXPECTED_ARMS), "5k manifest arm order differs")
    _require(manifest.get("benchmark_rounds") == list(EXPECTED_ROUNDS), "5k rounds differ")
    _require(_as_int(manifest.get("holes_per_round"), "5k holes") == 18, "5k is not 18 holes/round")
    producer_contract = _validate_producer_contract(manifest)
    source_start = _require_sha(
        manifest.get("streaming_source_start_sha256"), "5k source start"
    )
    source_end = _require_sha(
        manifest.get("streaming_source_end_sha256"), "5k source end"
    )
    _require(source_start == source_end == approval.streaming_source_sha256, "5k streaming source differs from approval")
    for field in ("dependency_start_hashes_sha256", "dependency_end_hashes_sha256"):
        hashes = manifest.get(field)
        _require(hashes == approval.dependency_sha256, f"5k {field} differs from approval")
    input_start = _validate_hash_mapping(
        manifest.get("input_hashes_sha256"), "5k input_hashes_sha256"
    )
    input_end = _validate_hash_mapping(
        manifest.get("input_end_hashes_sha256"), "5k input_end_hashes_sha256"
    )
    _require(input_start == input_end, "5k inputs changed during run")
    input_logical = _validate_hash_mapping(
        manifest.get("input_logical_hashes_sha256"),
        "5k input_logical_hashes_sha256",
    )

    gate = _read_json(root / "raw_128_gate.json")
    _validate_5k_gate(gate)
    _require(manifest.get("raw_128_gate") == gate, "5k gate sidecar and manifest differ")
    hazard = _read_json(root / "hazard_scalar_oracle.json")
    _require(_as_int(hazard.get("scalar_batch_exact_rows"), "5k hazard rows") > 0, "5k hazard oracle is empty")
    _require(manifest.get("hazard_scalar_oracle") == hazard, "5k hazard sidecar and manifest differ")

    benchmark = _read_json(root / "benchmark.json")
    _require(manifest.get("benchmark") == benchmark, "5k benchmark sidecar and manifest differ")
    _require(_as_int(benchmark.get("particles"), "5k particles") == 5_000, "Artifact is not 5k")
    _require(tuple(benchmark.get("arms_order", ())) == EXPECTED_ARMS, "5k benchmark arms differ")
    _require(benchmark.get("rng_mode") == "counter", "5k benchmark RNG differs")
    tail_policy.validate_manifest_policy(
        benchmark.get("tail_policy"), label="5k benchmark tail_policy"
    )
    _require(
        benchmark.get("timed_materialization_calls") == {"raw": 0, "normalized": 0},
        "5k timed kernel materialized supports",
    )
    _require(
        benchmark.get("timed_support_misses") == {"raw": 0, "normalized": 0},
        "5k timed kernel missed supports",
    )
    benchmark_arms = benchmark.get("arms")
    _require(isinstance(benchmark_arms, dict), "5k benchmark arm diagnostics missing")
    _require(set(benchmark_arms) == set(EXPECTED_ARMS), "5k benchmark diagnostic arms differ")
    for arm in EXPECTED_ARMS:
        arm_diagnostics = benchmark_arms[arm]
        _require(isinstance(arm_diagnostics, dict), f"5k {arm} diagnostics are not an object")
        _require(
            _as_int(arm_diagnostics.get("unsupported_state_rows"), f"5k {arm} unsupported") == 0,
            f"5k {arm} contains unsupported states",
        )
    for field in (
        "preparation_peak_rss_bytes", "prewarm_peak_rss_bytes",
        "timed_kernel_peak_rss_bytes", "global_peak_rss_bytes",
    ):
        _require(_as_int(benchmark.get(field), f"5k {field}") <= RSS_CAP_BYTES, f"5k {field} exceeded 4 GiB")
    _require(_as_int(benchmark.get("safe_rss_cap_bytes"), "5k RSS cap") == RSS_CAP_BYTES, "5k RSS cap differs")

    _validate_5k_support_cache(root, benchmark)

    event_key = str(manifest.get("event_key"))
    player_id = str(manifest.get("player_id"))
    _require(event_key not in {"", "None"}, "5k manifest event_key is absent")
    _require(player_id not in {"", "None"}, "5k manifest player_id is absent")
    _validate_5k_totals(
        root / "round_totals_int16.npz",
        expected_event_key=event_key,
        expected_player_id=player_id,
    )
    diagnostics = pd.read_csv(root / "hole_diagnostics.csv")
    _validate_hole_diagnostics(diagnostics, 5_000, "5k")
    parity = pd.read_parquet(root / "raw_128_parity_particles.parquet")
    _require(
        {"case_id", "particle_id"}.issubset(parity),
        "5k raw parity particle axes are missing",
    )
    _require(len(parity) == 2_304, "5k raw parity artifact is not 18 x 128")
    _require(
        not parity[["case_id", "particle_id"]].duplicated().any(),
        "5k raw parity axes duplicate",
    )
    _require(
        set(parity["case_id"].map(int)) == set(range(18))
        and set(parity["particle_id"].map(int)) == set(range(128)),
        "5k raw parity axes differ from 18 holes x particles 0:127",
    )
    summary = pd.read_csv(root / "round_summary.csv")
    _validate_round_summary(summary, particles=5_000, label="5k")
    _require((root / "REPORT.md").stat().st_size > 0, "5k report is empty")
    try:
        evaluator_panel = evaluator.load_particle_panel([root])
        evaluator_schema = evaluator.audit_schema_only(evaluator_panel)
    except (evaluator.EvaluationContractError, ValueError) as error:
        raise ValueError(
            f"Authoritative 5k fails the current evaluator contract: {error}"
        ) from error
    _require(evaluator_panel.particles == 5_000, "Evaluator did not load exactly 5k particles")
    _require(evaluator_panel.cases == 1, "Evaluator did not load one 5k event/player case")
    _require(
        evaluator_schema.get("schema_valid") is True,
        "Authoritative 5k evaluator schema audit did not pass",
    )
    return {
        "root": str(root.resolve()),
        "manifest_sha256": approval.authoritative_5k_manifest_sha256,
        "output_index_sha256": approval.authoritative_5k_output_index_sha256,
        "files_verified": int(files_verified),
        "event": manifest.get("event"),
        "event_key": event_key,
        "player_id": player_id,
        "input_hashes_sha256": input_start,
        "input_logical_hashes_sha256": input_logical,
        "producer_contract": producer_contract,
        "evaluator_schema_audit": evaluator_schema,
        "benchmark_sha256": stream.sha256_file(root / "benchmark.json"),
        "round_totals_sha256": stream.sha256_file(root / "round_totals_int16.npz"),
    }


def validate_current_frozen_graph(approval: ApprovalLock) -> dict[str, str]:
    tail_policy.validate_manifest_policy(
        approval.tail_policy, label="approval_lock.tail_policy"
    )
    current_runner = stream.sha256_file(Path(__file__))
    _require(
        current_runner == approval.runner_source_sha256,
        "Current 50k runner source differs from the approved frozen source",
    )
    current_evaluator = stream.sha256_file(Path(evaluator.__file__))
    _require(
        current_evaluator == approval.evaluator_source_sha256,
        "Current evaluator source differs from the approved frozen source",
    )
    current_streaming = stream.sha256_file(Path(stream.__file__))
    _require(
        current_streaming == approval.streaming_source_sha256,
        "Current streaming source differs from the approved frozen source",
    )
    observed = {
        name: stream.sha256_file(path)
        for name, path in stream.dependency_paths().items()
    }
    _require(
        observed == approval.dependency_sha256,
        "Current dependency graph differs from the approved frozen graph",
    )
    return observed


def validate_approval_lock_unchanged(approval: ApprovalLock) -> str:
    observed = stream.sha256_file(approval.path)
    _require(observed == approval.sha256, "Approval lock changed during execution")
    return observed


def validate_prepared_against_admission(
    prepared: stream.PreparedStreamingEvent,
    admission: Mapping[str, object],
) -> None:
    _require(
        str(prepared.atoms.target_event_key) == str(admission["event_key"]),
        "Prepared event key differs from authoritative 5k artifact",
    )
    _require(
        str(prepared.grid.iloc[0]["source_player_id"])
        == str(admission["player_id"]),
        "Prepared player differs from authoritative 5k artifact",
    )
    _require(
        prepared.input_hashes == admission["input_hashes_sha256"],
        "Current simulation inputs differ from authoritative 5k inputs",
    )
    _require(
        prepared.logical_hashes == admission["input_logical_hashes_sha256"],
        "Current logical simulation inputs differ from authoritative 5k inputs",
    )


def validate_independent_prefix_matches_gate(
    independent: Mapping[str, object], gate: Mapping[str, object]
) -> None:
    gate_prefix = gate.get("counter_50k_prefix_of_100k")
    _require(isinstance(gate_prefix, dict), "Fresh parity gate prefix audit missing")
    _require(gate_prefix.get("exact") is True, "Fresh parity gate prefix is not exact")
    _require(independent.get("exact") is True, "Independent prefix audit is not exact")
    for label, value in (
        ("fresh parity gate", gate_prefix),
        ("independent", independent),
    ):
        _require(
            tuple(value.get("transition_indices", ()))
            == EXPECTED_TRANSITION_INDICES,
            f"{label} prefix does not cover transitions 0 through 17",
        )
        _require(
            _as_int(value.get("max_transition_index"), f"{label} max transition")
            == tail_policy.MAX_CONTINUATION_TRANSITIONS - 1,
            f"{label} prefix does not reach transition 17",
        )
    gate_streams = gate_prefix.get("streams")
    independent_streams = independent.get("streams")
    _require(isinstance(gate_streams, dict), "Fresh parity gate stream hashes missing")
    _require(isinstance(independent_streams, dict), "Independent stream hashes missing")
    expected = {f"{family}:{name}" for family, name in EXPECTED_STREAMS}
    _require(set(gate_streams) == expected, "Fresh parity gate stream set differs")
    _require(set(independent_streams) == expected, "Independent stream set differs")
    _require(
        gate_streams == independent_streams,
        "Independent 50k prefix hashes differ from fresh parity-gate hashes",
    )


@dataclass
class _PhaseStats:
    started_monotonic: float
    start_rss_bytes: int
    peak_rss_bytes: int
    end_rss_bytes: int | None = None
    elapsed_seconds: float | None = None


class LifetimeRssMonitor:
    """Continuous phase-aware RSS monitor with main-thread interruption."""

    def __init__(
        self,
        *,
        cap_bytes: int = RSS_CAP_BYTES,
        interval_seconds: float = RSS_SAMPLE_SECONDS,
    ) -> None:
        self.cap_bytes = int(cap_bytes)
        self.interval_seconds = float(interval_seconds)
        self.process = psutil.Process()
        initial = int(self.process.memory_info().rss)
        self._peak_bytes = initial
        self._phase: str | None = None
        self._phases: dict[str, _PhaseStats] = {}
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.exceeded_bytes: int | None = None
        self.exceeded_phase: str | None = None

    @property
    def peak_bytes(self) -> int:
        with self._lock:
            return int(self._peak_bytes)

    def _sample(self) -> int:
        rss = int(self.process.memory_info().rss)
        interrupt = False
        with self._lock:
            self._peak_bytes = max(self._peak_bytes, rss)
            if self._phase is not None:
                stats = self._phases[self._phase]
                stats.peak_rss_bytes = max(stats.peak_rss_bytes, rss)
            if rss > self.cap_bytes and self.exceeded_bytes is None:
                self.exceeded_bytes = rss
                self.exceeded_phase = self._phase
                interrupt = True
        if interrupt:
            _thread.interrupt_main()
        return rss

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Lifetime RSS monitor already started")

        def sample_forever() -> None:
            while not self._stop.wait(self.interval_seconds):
                try:
                    self._sample()
                except (psutil.Error, OSError):
                    # A failed monitor is not silently safe: interrupt the run.
                    with self._lock:
                        if self.exceeded_bytes is None:
                            self.exceeded_bytes = self.cap_bytes + 1
                            self.exceeded_phase = self._phase or "monitor_failure"
                    _thread.interrupt_main()
                    return

        self._thread = threading.Thread(
            target=sample_forever, name="50k-rss-watchdog", daemon=True
        )
        self._thread.start()

    def assert_below(self, cap_bytes: int | None = None, phase: str = "") -> None:
        if cap_bytes is not None and int(cap_bytes) != self.cap_bytes:
            raise ValueError("RSS assertion cap differs from lifetime watchdog cap")
        self._sample()
        if self.exceeded_bytes is not None:
            raise MemoryError(
                f"RSS cap exceeded during {self.exceeded_phase or phase}: "
                f"{self.exceeded_bytes / 2**30:.3f} GiB > "
                f"{self.cap_bytes / 2**30:.3f} GiB"
            )

    @contextmanager
    def phase(self, name: str):
        with self._lock:
            if self._phase is not None:
                raise RuntimeError(f"Nested RSS phases are forbidden: {self._phase} -> {name}")
            if name in self._phases:
                raise RuntimeError(f"RSS phase already completed: {name}")
            rss = int(self.process.memory_info().rss)
            self._phase = str(name)
            self._phases[str(name)] = _PhaseStats(
                started_monotonic=time.perf_counter(),
                start_rss_bytes=rss,
                peak_rss_bytes=rss,
            )
        try:
            self.assert_below(self.cap_bytes, str(name))
            yield
            self.assert_below(self.cap_bytes, str(name))
        finally:
            rss = int(self.process.memory_info().rss)
            with self._lock:
                stats = self._phases[str(name)]
                stats.peak_rss_bytes = max(stats.peak_rss_bytes, rss)
                stats.end_rss_bytes = rss
                stats.elapsed_seconds = time.perf_counter() - stats.started_monotonic
                self._phase = None

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._sample()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            phases = {
                name: {
                    "start_rss_bytes": int(stats.start_rss_bytes),
                    "peak_rss_bytes": int(stats.peak_rss_bytes),
                    "end_rss_bytes": (
                        None if stats.end_rss_bytes is None else int(stats.end_rss_bytes)
                    ),
                    "elapsed_seconds": stats.elapsed_seconds,
                }
                for name, stats in self._phases.items()
            }
            return {
                "monitor": "continuous_process_rss_watchdog",
                "sample_interval_seconds": self.interval_seconds,
                "safe_rss_cap_bytes": self.cap_bytes,
                "lifetime_peak_rss_bytes": int(self._peak_bytes),
                "cap_exceeded": self.exceeded_bytes is not None,
                "cap_exceeded_bytes": self.exceeded_bytes,
                "cap_exceeded_phase": self.exceeded_phase,
                "phases": phases,
            }


def attest_counter_prefix(record: object) -> dict[str, object]:
    """Recompute every semantic 50k prefix against its 100k stream."""

    common = {
        "mode": "counter",
        "seed": stream.DEFAULT_SEED,
        "continuation_seed": stream.DEFAULT_CONTINUATION_SEED,
        "event_key": str(record.event_key),
        "player_id": str(record.source_player_id),
        "round_no": int(record.round_no),
        "hole_no": int(record.hole_no),
    }
    narrow = stream.UniformFactory(total_particles=PARTICLES, **common)
    wide = stream.UniformFactory(total_particles=PREFIX_WIDE_PARTICLES, **common)
    ids_50 = np.arange(PARTICLES, dtype=np.int64)
    ids_100 = np.arange(PREFIX_WIDE_PARTICLES, dtype=np.int64)
    hashes: dict[str, str] = {}
    for family, name in EXPECTED_STREAMS:
        if family == "initial":
            values = narrow.initial(name, ids_50)
            reference = wide.initial(name, ids_100)[:PARTICLES]
            if not np.array_equal(values, reference):
                raise AssertionError(
                    f"50k/100k counter prefix failed: {family}:{name}"
                )
        else:
            transition_values: list[np.ndarray] = []
            for transition_index in EXPECTED_TRANSITION_INDICES:
                values_at = narrow.transition(
                    transition_index, name, ids_50
                )
                reference_at = wide.transition(
                    transition_index, name, ids_100
                )[:PARTICLES]
                if not np.array_equal(values_at, reference_at):
                    raise AssertionError(
                        "50k/100k counter prefix failed: "
                        f"{family}:{name} at transition {transition_index}"
                    )
                transition_values.append(values_at)
            values = np.stack(transition_values)
        hashes[f"{family}:{name}"] = stream.array_sha256(values)
    return {
        "exact": True,
        "narrow_particles": PARTICLES,
        "wide_particles": PREFIX_WIDE_PARTICLES,
        "transition_indices": list(EXPECTED_TRANSITION_INDICES),
        "max_transition_index": tail_policy.MAX_CONTINUATION_TRANSITIONS - 1,
        "streams": hashes,
    }


def validate_simulation_result(
    totals: Mapping[str, np.ndarray],
    diagnostics: Mapping[str, object],
    hole_frame: pd.DataFrame,
) -> None:
    _require(tuple(totals) == EXPECTED_ARMS, "Simulation arms or order differ")
    for arm in EXPECTED_ARMS:
        values = np.asarray(totals[arm])
        _require(values.dtype == np.int16, f"{arm} totals are not int16")
        _require(values.shape == (4, PARTICLES), f"{arm} totals axes differ")
        tail_policy.round_totals_int16(
            values, label=f"50k {arm} round totals"
        )
        arm_diagnostics = diagnostics.get("arms", {}).get(arm, {})
        _require(
            _as_int(arm_diagnostics.get("unsupported_state_rows"), f"{arm} unsupported") == 0,
            f"{arm} contains unsupported states",
        )
    _validate_hole_diagnostics(hole_frame, PARTICLES, "50k")


def _write_round_totals(
    output_dir: Path,
    totals: Mapping[str, np.ndarray],
    prepared: stream.PreparedStreamingEvent,
) -> Path:
    path = output_dir / "round_totals_int16.npz"
    stacked = np.stack(
        [
            tail_policy.round_totals_int16(
                totals[arm], label=f"50k {arm} persisted round totals"
            )
            for arm in EXPECTED_ARMS
        ]
    )
    np.savez_compressed(
        path,
        arm_names=np.asarray(EXPECTED_ARMS),
        round_totals=stacked,
        round_numbers=np.asarray(EXPECTED_ROUNDS, dtype=np.int8),
        particle_ids=np.arange(PARTICLES, dtype=np.int64),
        player_id=np.asarray([str(prepared.grid.iloc[0]["source_player_id"])]),
        event_key=np.asarray([prepared.atoms.target_event_key]),
    )
    return path


def _jsonable(value: object) -> object:
    return stream._jsonable(value)


def _write_output_index(output_dir: Path) -> Path:
    return stream.write_complete_output_hashes(output_dir)


def validate_output_evaluator_compatibility(output_dir: Path) -> dict[str, object]:
    """Invoke the current executable evaluator contract on the emitted bundle."""

    try:
        panel = evaluator.load_particle_panel([Path(output_dir)])
        schema = evaluator.audit_schema_only(panel)
    except (evaluator.EvaluationContractError, ValueError) as error:
        raise ValueError(f"Emitted 50k bundle fails evaluator contract: {error}") from error
    _require(panel.particles == PARTICLES, "Evaluator did not load exactly 50k particles")
    _require(panel.cases == 1, "Evaluator did not load exactly one event/player case")
    _require(schema.get("schema_valid") is True, "Evaluator schema audit did not pass")
    _require(
        schema.get("anchor_statuses") == [evaluator.MECHANICS_ANCHOR_STATUS],
        "Engineering output did not remain mechanics-only in evaluator",
    )
    _require(
        schema.get("predictive_scoring_permitted") is False,
        "Evaluator unexpectedly permitted predictive scoring",
    )
    return schema


def seal_finalization_rss(
    output_dir: Path, sealed_snapshot: Mapping[str, object]
) -> None:
    """Persist a completed lifetime snapshot, then reseal every output hash."""

    output_dir = Path(output_dir)
    snapshot = _jsonable(dict(sealed_snapshot))
    (output_dir / "RSS_AUDIT.json").write_text(
        json.dumps(snapshot, indent=2), encoding="utf-8"
    )
    manifest_path = output_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest["rss_monitor_sealed_snapshot"] = snapshot
    manifest_path.write_text(
        json.dumps(_jsonable(manifest), indent=2), encoding="utf-8"
    )
    _write_output_index(output_dir)


def write_postseal_evaluator_audit(output_dir: Path) -> dict[str, object]:
    """Audit the stable final manifest and close the detached audit in the index.

    The evaluator records the byte hash of ``manifest.json``.  It must therefore
    run only after the RSS reseal has stopped mutating that manifest.  The audit
    itself is detached from the manifest and is covered by ``OUTPUT_SHA256`` so
    there is no circular manifest/audit hash dependency.
    """

    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    final_manifest_sha256 = stream.sha256_file(manifest_path)
    schema = validate_output_evaluator_compatibility(output_dir)
    bundle_audits = schema.get("bundle_audits")
    _require(
        isinstance(bundle_audits, list) and len(bundle_audits) == 1,
        "Post-seal evaluator audit must contain exactly one bundle audit",
    )
    _require(
        bundle_audits[0].get("manifest_sha256") == final_manifest_sha256,
        "Post-seal evaluator audit did not bind the stable final manifest",
    )
    manifest_input_hashes = {
        str(Path(path).resolve()): value
        for path, value in dict(schema.get("input_hashes_sha256", {})).items()
    }
    _require(
        manifest_input_hashes.get(str(manifest_path.resolve()))
        == final_manifest_sha256,
        "Post-seal evaluator input hashes did not bind the final manifest",
    )
    audit_path = output_dir / "evaluator_schema_audit.json"
    audit_path.write_text(
        json.dumps(_jsonable(schema), indent=2), encoding="utf-8"
    )
    _require(
        stream.sha256_file(manifest_path) == final_manifest_sha256,
        "Writing the detached evaluator audit changed the final manifest",
    )
    _write_output_index(output_dir)
    output_index = _read_json(output_dir / "OUTPUT_SHA256.json")
    _require(
        output_index.get("evaluator_schema_audit.json", {}).get("sha256")
        == stream.sha256_file(audit_path),
        "Final checksum index did not bind the post-seal evaluator audit",
    )
    return schema


def write_outputs(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    approval: ApprovalLock,
    admission: Mapping[str, object],
    prepared: stream.PreparedStreamingEvent,
    gate: Mapping[str, object],
    hazard_oracle: Mapping[str, object],
    prefix: Mapping[str, object],
    parity_rows: pd.DataFrame,
    totals: Mapping[str, np.ndarray],
    diagnostics: pd.DataFrame,
    execution: Mapping[str, object],
    source_start_sha256: str,
    streaming_source_start_sha256: str,
    dependency_start_sha256: Mapping[str, str],
    input_end_sha256: Mapping[str, str],
    monitor_snapshot_before_final_hash: Mapping[str, object],
) -> Path:
    output_dir = Path(output_dir)
    validate_independent_prefix_matches_gate(prefix, gate)
    producer_contract = _validate_producer_contract(
        admission.get("producer_contract", {})
    )
    _require(
        source_start_sha256 == approval.runner_source_sha256,
        "Runner start hash differs from the approved runner hash",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    parity_rows.to_parquet(output_dir / "raw_128_parity_particles.parquet", index=False)
    (output_dir / "raw_128_gate.json").write_text(
        json.dumps(_jsonable(dict(gate)), indent=2), encoding="utf-8"
    )
    (output_dir / "hazard_scalar_oracle.json").write_text(
        json.dumps(_jsonable(dict(hazard_oracle)), indent=2), encoding="utf-8"
    )
    (output_dir / "counter_50k_prefix_of_100k.json").write_text(
        json.dumps(_jsonable(dict(prefix)), indent=2), encoding="utf-8"
    )
    diagnostics.to_csv(output_dir / "hole_diagnostics.csv", index=False)
    round_totals_path = _write_round_totals(output_dir, totals, prepared)
    round_summary = stream._round_summary(totals)
    _validate_round_summary(round_summary, particles=PARTICLES, label="50k")
    round_summary.to_csv(output_dir / "round_summary.csv", index=False)
    (output_dir / "execution.json").write_text(
        json.dumps(_jsonable(dict(execution)), indent=2), encoding="utf-8"
    )

    runner_source_end = stream.sha256_file(Path(__file__))
    evaluator_source_end = stream.sha256_file(Path(evaluator.__file__))
    streaming_source_end = stream.sha256_file(Path(stream.__file__))
    dependency_end = {
        name: stream.sha256_file(path)
        for name, path in stream.dependency_paths().items()
    }
    if runner_source_end != source_start_sha256:
        raise AssertionError("50k runner source changed during execution")
    if runner_source_end != approval.runner_source_sha256:
        raise AssertionError("50k runner source differs from approval lock")
    if evaluator_source_end != approval.evaluator_source_sha256:
        raise AssertionError("Evaluator source differs from approval lock")
    if streaming_source_end != streaming_source_start_sha256:
        raise AssertionError("Frozen streaming source changed during execution")
    if dependency_end != dict(dependency_start_sha256):
        raise AssertionError("Frozen dependency graph changed during execution")
    if dict(input_end_sha256) != prepared.input_hashes:
        raise AssertionError("Streaming inputs changed during execution")
    if prepared.input_hashes != admission.get("input_hashes_sha256"):
        raise AssertionError("Final inputs differ from authoritative 5k admission")
    if prepared.logical_hashes != admission.get("input_logical_hashes_sha256"):
        raise AssertionError("Final logical inputs differ from authoritative 5k admission")
    approval_lock_end_sha256 = validate_approval_lock_unchanged(approval)

    payload_hashes = {
        path.relative_to(output_dir).as_posix(): stream.sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "artifact": RUNNER_VERSION,
        "research_only": True,
        "engineering_mechanics_only": True,
        "production_touched": False,
        "predictive_scoring_permitted": False,
        "predictive_scores_emitted": False,
        "production_mean_anchor_status": "not_calibrated_mechanics_only",
        "production_skill_embedded": False,
        "posthoc_recentering_applied": False,
        "evaluator_schema_audit_contract": {
            "role": "post_rss_seal_detached_attestation",
            "generated_after_manifest_stabilization": True,
            "covered_by_output_index": True,
            "excluded_from_rss_sealed_snapshot": True,
        },
        "tail_policy": tail_policy.manifest_policy(),
        "reference_precompute_provenance": producer_contract[
            "reference_precompute_provenance"
        ],
        "particles": PARTICLES,
        "chunk_size": int(args.chunk_size),
        "arms": list(EXPECTED_ARMS),
        "normalization_mode": producer_contract["normalization_mode"],
        "ability_log_weight_per_sg2": producer_contract[
            "ability_log_weight_per_sg2"
        ],
        "rounds": list(EXPECTED_ROUNDS),
        "holes_per_round": 18,
        "particle_axis": {"start": 0, "stop_exclusive": PARTICLES, "shared_by_arms": True},
        "rng_mode": "counter",
        "counter_version": producer_contract["rng_modes"]["counter_version"],
        "rng_modes": producer_contract["rng_modes"],
        "counter_50k_prefix_of_100k": dict(prefix),
        "survivor_filtering": False,
        "unsupported_states_fail_closed": True,
        "streaming_contract": producer_contract["streaming_contract"],
        "transform_order": producer_contract["transform_order"],
        "alignment_count": producer_contract["alignment_count"],
        "mass_contract": producer_contract["mass_contract"],
        "checkpoint_resume_supported": False,
        "checkpoint_resume_reason": (
            "omitted until independently proven to preserve exact counter prefix, "
            "chunk invariance, and hash-complete shard semantics"
        ),
        "event": prepared.atoms.event_name,
        "event_key": prepared.atoms.target_event_key,
        "holdout_year": int(prepared.atoms.holdout_year),
        "player_id": str(prepared.grid.iloc[0]["source_player_id"]),
        "authoritative_5k_admission": dict(admission),
        "approval_lock_path": str(approval.path),
        "approval_lock_sha256": approval.sha256,
        "approval_lock_start_sha256": approval.sha256,
        "approval_lock_end_sha256": approval_lock_end_sha256,
        "runner_source_start_sha256": source_start_sha256,
        "runner_source_end_sha256": runner_source_end,
        "evaluator_source_start_sha256": approval.evaluator_source_sha256,
        "evaluator_source_end_sha256": evaluator_source_end,
        "streaming_source_start_sha256": streaming_source_start_sha256,
        "streaming_source_end_sha256": streaming_source_end,
        "dependency_start_hashes_sha256": dict(dependency_start_sha256),
        "dependency_end_hashes_sha256": dependency_end,
        "input_hashes_sha256": prepared.input_hashes,
        "input_start_hashes_sha256": prepared.input_hashes,
        "input_end_hashes_sha256": dict(input_end_sha256),
        "input_logical_hashes_sha256": prepared.logical_hashes,
        "raw_128_gate": dict(gate),
        "hazard_scalar_oracle": dict(hazard_oracle),
        "execution": dict(execution),
        "rss_monitor_before_final_hash": dict(monitor_snapshot_before_final_hash),
        "round_totals_sha256": stream.sha256_file(round_totals_path),
        "payload_output_hashes_sha256": payload_hashes,
        "cli": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(_jsonable(manifest), indent=2), encoding="utf-8"
    )
    report = [
        "# 50k normalized shot-chain engineering run",
        "",
        "Research mechanics only. This artifact is not eligible for predictive scoring.",
        "",
        f"- Particles: {PARTICLES:,} per arm, round, and hole",
        "- Axes: raw + normalized, R1-R4, 18 holes, identical particle IDs",
        "- RNG: counter; exact 50k prefix of 100k across all 13 semantic streams",
        (
            "- Tail policy: 18 continuation transitions; no physical score "
            "cap/clipping; 15+ strokes diagnostic only"
        ),
        "- Survivor filtering: forbidden; unresolved/unsupported states: fail closed",
        "- Production mean anchor: not calibrated (mechanics only)",
        "- Checkpoint/resume: omitted",
        f"- Lifetime RSS before final hash: {int(monitor_snapshot_before_final_hash['lifetime_peak_rss_bytes']) / 2**30:.3f} GiB",
        "",
        "## Round summary",
        "",
        stream._deterministic_markdown_table(round_summary),
        "",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    _write_output_index(output_dir)
    return output_dir


def run_50k(args: argparse.Namespace) -> Path | dict[str, object]:
    """Validate admission and, unless dry-run, execute the fixed 50k run."""

    validate_cli_contract(args)
    monitor = LifetimeRssMonitor()
    monitor.start()
    runner_start = stream.sha256_file(Path(__file__))
    try:
        with monitor.phase("setup_admission"):
            approval = load_approval_lock(Path(args.approval_lock))
            graph = validate_current_frozen_graph(approval)
            admission = validate_authoritative_5k(
                Path(args.authoritative_5k_dir), approval, verify_all_files=True
            )
            if args.dry_run:
                return {
                    "dry_run": True,
                    "admission_valid": True,
                    "particles": PARTICLES,
                    "chunk_size": int(args.chunk_size),
                    "authoritative_5k": admission,
                    "approval_lock_sha256": approval.sha256,
                    "runner_source_sha256": approval.runner_source_sha256,
                    "evaluator_source_sha256": approval.evaluator_source_sha256,
                    "streaming_source_sha256": approval.streaming_source_sha256,
                    "dependency_sha256": graph,
                    "tail_policy": tail_policy.manifest_policy(),
                    "rss": monitor.snapshot(),
                    "simulation_launched": False,
                }
            prepared = stream.prepare_streaming_event(
                export_root=Path(args.export_root),
                history_db=Path(args.history_db),
                profile_dir=Path(args.profile_dir),
                baseline_cases=Path(args.baseline_cases),
                reference_dir=Path(args.reference_dir),
                event_name=str(args.event_name),
                holdout_year=int(args.holdout_year),
                course_start_year=int(args.course_start_year),
                roster_seed=int(args.roster_seed),
            )
            prepared.preparation_peak_rss_bytes = monitor.peak_bytes
            validate_prepared_against_admission(prepared, admission)

        with monitor.phase("parity_gate"):
            hazard_oracle = stream.audit_batched_hazard_oracle(prepared)
            gate, parity_rows = stream.run_raw_128_gate(
                prepared, reference_dir=Path(args.reference_dir)
            )
            _validate_5k_gate(gate)
            sample_record = next(prepared.grid.itertuples(index=False))
            prefix = attest_counter_prefix(sample_record)
            validate_independent_prefix_matches_gate(prefix, gate)

        with monitor.phase("support_prewarm"):
            repositories = stream._new_benchmark_repositories(
                prepared, Path(args.reference_dir)
            )
            engine = stream.StreamingShotChain(
                repositories=repositories,
                course_drives=prepared.atoms.raw_course_drives,
                course_app=prepared.atoms.raw_course_app,
                state_sampler=stream.batched_population.BatchedPopulationTransitionSampler(
                    prepared.state_model, chunk_size=int(args.chunk_size)
                ),
                hazard_resolver=prepared.batched_hazard,
            )
            prewarm_totals, _, prewarm_details = stream._simulate_grid(
                engine,
                prepared.grid,
                arms=EXPECTED_ARMS,
                particles=PARTICLES,
                chunk_size=int(args.chunk_size),
                rng_mode="counter",
                collect_holes=False,
                rss_monitor=monitor,
                rss_cap_bytes=RSS_CAP_BYTES,
            )

        with monitor.phase("support_serialization"):
            cache_manifests = {
                arm: stream.persist_numeric_support_repository(
                    repository,
                    Path(args.output_dir) / "support_cache" / arm,
                    rss_monitor=monitor,
                    rss_cap_bytes=RSS_CAP_BYTES,
                )
                for arm, repository in repositories.items()
            }

        with monitor.phase("pre_timed_release"):
            post_release_rss = stream.release_heavy_precompute_payload(
                prepared, repositories
            )
            monitor.assert_below(RSS_CAP_BYTES, "pre_timed_release")
            calls_before = {
                arm: repo.materialization_calls for arm, repo in repositories.items()
            }
            misses_before = {
                arm: repo.miss_count for arm, repo in repositories.items()
            }

        with monitor.phase("timed_kernel"):
            timed_started = time.perf_counter()
            totals, _, timed_details = stream._simulate_grid(
                engine,
                prepared.grid,
                arms=EXPECTED_ARMS,
                particles=PARTICLES,
                chunk_size=int(args.chunk_size),
                rng_mode="counter",
                collect_holes=False,
                rss_monitor=monitor,
                rss_cap_bytes=RSS_CAP_BYTES,
            )
            timed_seconds = time.perf_counter() - timed_started

        with monitor.phase("post_timed_validation"):
            hole_frame = pd.DataFrame(timed_details.pop("hole_summary"))
            validate_simulation_result(totals, timed_details, hole_frame)
            prewarm_holes = pd.DataFrame(prewarm_details.pop("hole_summary"))
            validate_simulation_result(
                prewarm_totals, prewarm_details, prewarm_holes
            )
            for arm in EXPECTED_ARMS:
                np.testing.assert_array_equal(prewarm_totals[arm], totals[arm])
            pd.testing.assert_frame_equal(
                prewarm_holes.reset_index(drop=True),
                hole_frame.reset_index(drop=True),
                check_exact=True,
            )
            new_calls = {
                arm: repo.materialization_calls - calls_before[arm]
                for arm, repo in repositories.items()
            }
            new_misses = {
                arm: repo.miss_count - misses_before[arm]
                for arm, repo in repositories.items()
            }
            _require(
                not any(new_calls.values()), "Timed kernel materialized supports"
            )
            _require(not any(new_misses.values()), "Timed kernel missed supports")

        with monitor.phase("finalization"):
            input_end = stream.verify_input_hashes_unchanged(prepared)
            execution = {
                "particles": PARTICLES,
                "chunk_size": int(args.chunk_size),
                "arms_order": list(EXPECTED_ARMS),
                "rounds": list(EXPECTED_ROUNDS),
                "holes_per_round": 18,
                "rng_mode": "counter",
                "tail_policy": tail_policy.manifest_policy(),
                "prewarm_replay_exact": True,
                "prewarm_round_totals_sha256": {
                    arm: stream.array_sha256(prewarm_totals[arm])
                    for arm in EXPECTED_ARMS
                },
                "timed_round_totals_sha256": {
                    arm: stream.array_sha256(totals[arm])
                    for arm in EXPECTED_ARMS
                },
                "timed_seconds": timed_seconds,
                "prewarm_kernel": prewarm_details,
                "timed_kernel": timed_details,
                "timed_materialization_calls": new_calls,
                "timed_support_misses": new_misses,
                "post_release_rss_bytes": int(post_release_rss),
                "numeric_support_cache": cache_manifests,
                "support_repository": {
                    arm: {
                        "supports": len(repo.supports),
                        "cache_bytes": repo.cache_nbytes,
                        "materialization_calls": repo.materialization_calls,
                        "materialized_atoms": repo.materialized_atoms,
                    }
                    for arm, repo in repositories.items()
                },
                "production_mean_anchor_status": "not_calibrated_mechanics_only",
                "predictive_scoring_permitted": False,
            }
            output = write_outputs(
                Path(args.output_dir),
                args=args,
                approval=approval,
                admission=admission,
                prepared=prepared,
                gate=gate,
                hazard_oracle=hazard_oracle,
                prefix=prefix,
                parity_rows=parity_rows,
                totals=totals,
                diagnostics=hole_frame,
                execution=execution,
                source_start_sha256=runner_start,
                streaming_source_start_sha256=approval.streaming_source_sha256,
                dependency_start_sha256=graph,
                input_end_sha256=input_end,
                monitor_snapshot_before_final_hash=monitor.snapshot(),
            )
            # Fail early on evaluator incompatibility, but do not persist this
            # pre-seal result: the RSS seal still has to mutate manifest.json.
            validate_output_evaluator_compatibility(output)
        finalized_snapshot = monitor.snapshot()
        with monitor.phase("provenance_seal_pass1"):
            seal_finalization_rss(output, finalized_snapshot)
        # Seal a second time so the persisted lifetime peak includes a full
        # completed manifest/report/checksum pass rather than stopping just
        # before the work whose memory it is meant to audit.
        post_hash_snapshot = monitor.snapshot()
        with monitor.phase("provenance_seal_pass2"):
            seal_finalization_rss(output, post_hash_snapshot)
        write_postseal_evaluator_audit(output)
        validate_approval_lock_unchanged(approval)
        _require(
            stream.sha256_file(Path(__file__)) == approval.runner_source_sha256,
            "Approved runner source changed during provenance sealing",
        )
        _require(
            stream.sha256_file(Path(evaluator.__file__))
            == approval.evaluator_source_sha256,
            "Approved evaluator source changed during provenance sealing",
        )
        monitor.assert_below(RSS_CAP_BYTES, "finalization")
        return output
    except KeyboardInterrupt as error:
        if monitor.exceeded_bytes is not None:
            raise MemoryError(
                f"50k process interrupted at RSS hard cap during "
                f"{monitor.exceeded_phase}: {monitor.exceeded_bytes} bytes"
            ) from error
        raise
    finally:
        monitor.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authoritative-5k-dir", type=Path, required=True)
    parser.add_argument("--approval-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--particles", type=int, default=PARTICLES)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--export-root", type=Path, default=stream.precompute.DEFAULT_EXPORT)
    parser.add_argument("--history-db", type=Path, default=stream.precompute.asof.DEFAULT_HISTORY_DB)
    parser.add_argument("--profile-dir", type=Path, default=stream.precompute.DEFAULT_PROFILES)
    parser.add_argument("--baseline-cases", type=Path, default=stream.precompute.DEFAULT_BASELINES)
    parser.add_argument("--reference-dir", type=Path, default=stream.DEFAULT_REFERENCE)
    parser.add_argument("--event-name", default=stream.precompute.DEFAULT_EVENT)
    parser.add_argument("--holdout-year", type=int, default=stream.precompute.DEFAULT_HOLDOUT)
    parser.add_argument("--course-start-year", type=int, default=2020)
    parser.add_argument("--roster-seed", type=int, default=stream.DEFAULT_SEED)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        result = run_50k(args)
    except (AssertionError, FileExistsError, FileNotFoundError, MemoryError, ValueError) as error:
        raise SystemExit(f"REFUSED: {error}") from error
    if args.dry_run:
        print(json.dumps(_jsonable(result), indent=2))
    else:
        print(f"Wrote {result}")


if __name__ == "__main__":
    main()
