"""Partitioned ARG/PUTT support contract for the frozen Lowry R1 all-18 trace.

This research-only runner consumes the already sealed all-hole transition
ledger and reconstructs the exact population-transition neighbourhood for
every ARG/PUTT query.  It deliberately writes one arm/hole/category partition
at a time so the full support table is never resident in memory.

The structural support and the donor-ability policy are persisted separately:

* ``support_atoms.parquet`` is the exact, identity-preserving KNN measure from
  :mod:`hole_sim_population_transition_support_contract`;
* ``donor_ability_policy.parquet`` is a keyed overlay from the frozen v2
  donor-ability policy.  Missing observations remain explicitly unavailable
  and receive the policy's exact neutral value; they are never rewritten as
  observed data.

No production module imports this file and this runner does not solve or apply
the transition mean anchor.
"""

from __future__ import annotations

import argparse
import gc
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
ARCHIVE = Path(__file__).resolve().parent
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

import hole_sim_donor_ability_policy as ability_policy  # noqa: E402
import hole_sim_normalized_chain_streaming as streaming  # noqa: E402
import hole_sim_population_transition_support_contract as population_contract  # noqa: E402
import hole_sim_streaming_transition_ledger as ledger_v6  # noqa: E402
import hole_sim_streaming_transition_ledger_r1_all18 as all18  # noqa: E402


CONTRACT = "partitioned_complete_population_neighbourhood_r1_all18_v1"
POLICY_OVERLAY_CONTRACT = "frozen_donor_ability_policy_overlay_v2"
PARENT_OUTPUT_INDEX_CONTRACT = "non_circular_output_sha256_v1"
OUTPUT_INDEX_CONTRACT = "recursive_non_circular_output_sha256_v2"
RSS_CAP_BYTES = 4 * 2**30

DEFAULT_PARENT = (
    REPO
    / "permanent_data"
    / "hole_sim_streaming_transition_ledger_lowry_r1_all18_128_v1"
)
DEFAULT_OUTPUT = (
    REPO
    / "permanent_data"
    / "hole_sim_population_transition_support_lowry_r1_all18_128_v1"
)

EXPECTED_PARENT_MANIFEST_SHA256 = (
    "000bb368b0694eee0669cb03849fb97bf4307ccdfb94e63992298bce775dcb91"
)
EXPECTED_PARENT_INDEX_SHA256 = (
    "fc8073bef16bf4c710bb7c270fb3105e10f214779d3f6ce978c414ce60c242c9"
)
EXPECTED_PARENT_RUNNER_SHA256 = (
    "b71e4b801ad68d98bc43b67a4f6d96d777fe9a32f089d902fe4a999f9bcfa164"
)
EXPECTED_V6_SOURCE_SHA256 = (
    "11b14109b938cda5da4b4eaae4d00eb43f9d44074fdbd9e48328a4f889d7ab97"
)
EXPECTED_POPULATION_CONTRACT_SHA256 = (
    "7a4d5d5d02cd4b5cde7f216d742a63754c555253f89f00a8bcd75cf288be2ec8"
)
EXPECTED_ABILITY_POLICY_SHA256 = (
    "3f9593333a79ffde258ec1efb149294bd1b036721c30cfcf5d47cf5ea2aa7339"
)
EXPECTED_ABILITY_POLICY_IDENTITY = (
    "f30e0161a1767d42f01dc0a33f617d93327286e3f49c59287c9b751686ba5bc7"
)

PARTITION_COLUMNS = ("arm", "round_no", "hole_no", "category")
POPULATION_CATEGORIES = ("arg", "putt")
ABILITY_SOURCE_COLUMN = {"arg": "ag_arg", "putt": "ag_putt"}

POLICY_INPUT_COLUMNS = tuple(ability_policy.INPUT_COLUMNS)
POLICY_OUTPUT_COLUMNS = tuple(ability_policy.OUTPUT_COLUMNS)
POLICY_PERSISTED_COLUMNS = (*POLICY_INPUT_COLUMNS, *POLICY_OUTPUT_COLUMNS)


class PopulationAll18Error(AssertionError):
    """Raised when the parent artifact or a derived partition is not exact."""


def sha256_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _stable_sha256(value: object) -> str:
    return sha256(_stable_json_bytes(value)).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_stable_json_bytes(value))


def _require_sha(path: Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != str(expected):
        raise PopulationAll18Error(
            f"{label} SHA changed: {observed} != {expected}"
        )
    return observed


def _indexed_files(root: Path, payload: Mapping[str, object]) -> dict[str, dict[str, object]]:
    contract_name = payload.get("contract")
    if contract_name not in {
        PARENT_OUTPUT_INDEX_CONTRACT,
        OUTPUT_INDEX_CONTRACT,
    }:
        raise PopulationAll18Error("Parent output-index contract changed")
    raw = payload.get("files")
    if not isinstance(raw, Mapping):
        raise PopulationAll18Error("Parent output index lacks a file mapping")
    output: dict[str, dict[str, object]] = {}
    for name, metadata in raw.items():
        relative = str(name).replace("\\", "/")
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise PopulationAll18Error(f"Unsafe indexed path: {relative}")
        if not isinstance(metadata, Mapping):
            raise PopulationAll18Error(f"Invalid indexed metadata: {relative}")
        output[relative] = dict(metadata)
    actual = {
        str(path.relative_to(root)).replace("\\", "/")
        for path in root.rglob("*")
        if path.is_file()
        and str(path.relative_to(root)).replace("\\", "/")
        != "OUTPUT_SHA256.json"
    }
    if actual != set(output):
        raise PopulationAll18Error("Parent output index is not an exact file closure")
    actual_directories = {
        str(path.relative_to(root)).replace("\\", "/")
        for path in root.rglob("*")
        if path.is_dir() and not path.is_symlink()
    }
    if contract_name == OUTPUT_INDEX_CONTRACT:
        declared_directories = payload.get("directories")
        if not isinstance(declared_directories, list):
            raise PopulationAll18Error("Recursive output index lacks directories")
        if actual_directories != {str(value) for value in declared_directories}:
            raise PopulationAll18Error(
                "Recursive output index is not an exact directory closure"
            )
    elif actual_directories:
        raise PopulationAll18Error(
            "Legacy parent index cannot attest unexpected nested directories"
        )
    for path in root.rglob("*"):
        if path.is_symlink():
            raise PopulationAll18Error("Parent artifact contains a symlink")
    return output


def verify_output_index(root: Path) -> dict[str, object]:
    root = Path(root)
    index_path = root / "OUTPUT_SHA256.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    files = _indexed_files(root, payload)
    total_bytes = 0
    for relative, metadata in sorted(files.items()):
        path = root / relative
        expected_sha = str(metadata.get("sha256", ""))
        expected_bytes = int(metadata.get("bytes", -1))
        observed_bytes = int(path.stat().st_size)
        if observed_bytes != expected_bytes or sha256_file(path) != expected_sha:
            raise PopulationAll18Error(f"Indexed payload changed: {relative}")
        total_bytes += observed_bytes
    return {
        "files_verified": int(len(files)),
        "bytes_verified": int(total_bytes),
        "output_index_sha256": sha256_file(index_path),
        "all_exact": True,
    }


def admit_parent(parent_dir: Path = DEFAULT_PARENT) -> tuple[dict[str, object], pd.DataFrame, dict[str, object]]:
    """Validate the sealed all-18 artifact and return its population ledger."""

    parent_dir = Path(parent_dir)
    if not parent_dir.is_dir():
        raise FileNotFoundError(f"Parent all-18 artifact is absent: {parent_dir}")
    _require_sha(
        parent_dir / "manifest.json",
        EXPECTED_PARENT_MANIFEST_SHA256,
        "all-18 parent manifest",
    )
    _require_sha(
        parent_dir / "OUTPUT_SHA256.json",
        EXPECTED_PARENT_INDEX_SHA256,
        "all-18 parent output index",
    )
    closure = verify_output_index(parent_dir)
    manifest = json.loads((parent_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("contract") != all18.CONTRACT:
        raise PopulationAll18Error("Parent all-18 contract changed")
    if manifest.get("axes", {}).get("arms") != ["raw", "normalized"]:
        raise PopulationAll18Error("Parent arm axis changed")
    if manifest.get("axes", {}).get("holes") != list(range(1, 19)):
        raise PopulationAll18Error("Parent hole axis changed")
    if manifest.get("axes", {}).get("particle_ids") != list(range(128)):
        raise PopulationAll18Error("Parent particle axis changed")
    gates = manifest.get("gates", {})
    required_true = (
        "all_particles_terminal",
        "complete_axes_exact",
        "counter_rng_uniforms_exact",
        "physical_state_seams_exact",
        "plain_parent_result_exact",
        "proposal_app_terminal_mass_exact",
        "proposal_support_mass_exact",
        "zero_stroke_identity_residual",
    )
    if any(gates.get(name) is not True for name in required_true):
        raise PopulationAll18Error("Parent mechanics gates are not all exact")
    if gates.get("anchor_enabled") is not False or gates.get("anchor_authorized") is not False:
        raise PopulationAll18Error("Parent artifact unexpectedly enables the anchor")
    ledger_path = parent_dir / "complete_transition_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    expected_logical = manifest.get("output_logical_sha256", {}).get("ledger")
    observed_logical = streaming.frame_logical_sha256(ledger)
    if observed_logical != expected_logical:
        raise PopulationAll18Error("Parent ledger logical hash changed")
    population = ledger.loc[
        ledger["category"].astype(str).isin(POPULATION_CATEGORIES)
    ].copy()
    if len(population) != 9562:
        raise PopulationAll18Error(
            f"Parent population row count changed: {len(population)}"
        )
    if population.duplicated(
        ["arm", "round_no", "hole_no", "particle_id", "atom_sequence"]
    ).any():
        raise PopulationAll18Error("Parent population ledger identities duplicated")
    admission = {
        **closure,
        "parent_manifest_sha256": EXPECTED_PARENT_MANIFEST_SHA256,
        "parent_index_sha256": EXPECTED_PARENT_INDEX_SHA256,
        "parent_ledger_sha256": sha256_file(ledger_path),
        "parent_ledger_logical_sha256": observed_logical,
        "population_rows": int(len(population)),
        "source_contract": str(manifest["contract"]),
    }
    return manifest, population.reset_index(drop=True), admission


def dependency_paths() -> dict[str, Path]:
    paths = dict(all18._dependency_paths())
    paths.update(
        {
            "all18_runner_source": Path(all18.__file__),
            "population_contract_source": Path(population_contract.__file__),
            "ability_policy_source": Path(ability_policy.__file__),
            "population_all18_runner_source": Path(__file__),
        }
    )
    return paths


def hash_paths(paths: Mapping[str, Path]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in sorted(paths.items())}


def validate_rss_cap(cap_bytes: int) -> int:
    cap = int(cap_bytes)
    if cap <= 0:
        raise ValueError("RSS cap must be positive")
    if cap > RSS_CAP_BYTES:
        raise ValueError(
            "Population support RSS cap cannot exceed the fixed 4 GiB ceiling"
        )
    return cap


def validate_frozen_sources(
    parent_manifest: Mapping[str, object],
) -> dict[str, object]:
    observed = hash_paths(dependency_paths())
    try:
        expected_parent = dict(
            parent_manifest["inputs"]["dependency_sha256_end"]  # type: ignore[index]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PopulationAll18Error(
            "Parent manifest lacks its closed dependency hash map"
        ) from exc
    parent_paths = all18._dependency_paths()
    if set(expected_parent) != set(parent_paths):
        raise PopulationAll18Error(
            "Parent dependency key set changed from the executable graph"
        )
    for name, expected_digest in sorted(expected_parent.items()):
        if observed.get(name) != str(expected_digest):
            raise PopulationAll18Error(
                f"Parent-bound dependency changed for {name}: "
                f"{observed.get(name)} != {expected_digest}"
            )
    expected = {
        "all18_runner_source": EXPECTED_PARENT_RUNNER_SHA256,
        "frozen_v6_ledger_source": EXPECTED_V6_SOURCE_SHA256,
        "population_contract_source": EXPECTED_POPULATION_CONTRACT_SHA256,
        "ability_policy_source": EXPECTED_ABILITY_POLICY_SHA256,
    }
    for name, digest in expected.items():
        if observed.get(name) != digest:
            raise PopulationAll18Error(
                f"Frozen source changed for {name}: {observed.get(name)} != {digest}"
            )
    policy_identity = ability_policy.policy_payload().get("policy_identity_sha256")
    if policy_identity != EXPECTED_ABILITY_POLICY_IDENTITY:
        raise PopulationAll18Error("Frozen donor-ability policy identity changed")
    streaming_attestation = streaming.validate_frozen_dependencies()
    return {
        "parent_dependency_keys": sorted(expected_parent),
        "parent_dependency_sha256": expected_parent,
        "current_dependency_sha256": {
            name: observed[name] for name in sorted(expected_parent)
        },
        "parent_dependencies_all_exact": True,
        "derived_source_sha256": {
            name: observed[name] for name in sorted(expected)
        },
        "derived_sources_all_exact": True,
        "ability_policy_identity_sha256": str(policy_identity),
        "streaming_frozen_dependency_attestation": streaming_attestation,
    }


def partition_population_ledger(population: pd.DataFrame) -> list[tuple[tuple[str, int, int, str], pd.DataFrame]]:
    required = set(PARTITION_COLUMNS).union({"particle_id", "atom_sequence"})
    missing = sorted(required.difference(population.columns))
    if missing:
        raise ValueError(f"Population ledger misses partition columns={missing}")
    if population.empty:
        raise ValueError("Population ledger cannot be empty")
    work = population.copy()
    work["category"] = work["category"].astype(str)
    if not work["category"].isin(POPULATION_CATEGORIES).all():
        raise ValueError("Population ledger contains a non-ARG/PUTT row")
    partitions: list[tuple[tuple[str, int, int, str], pd.DataFrame]] = []
    for raw_key, group in work.groupby(list(PARTITION_COLUMNS), observed=True, sort=True):
        arm, round_no, hole_no, category = raw_key
        key = (str(arm), int(round_no), int(hole_no), str(category))
        ordered = group.sort_values(
            ["particle_id", "atom_sequence"], kind="mergesort"
        ).reset_index(drop=True)
        partitions.append((key, ordered))
    flattened = pd.concat([group for _, group in partitions], ignore_index=True)
    identity = ["arm", "round_no", "hole_no", "particle_id", "atom_sequence"]
    left = work.sort_values(identity, kind="mergesort")[identity].reset_index(drop=True)
    right = flattened.sort_values(identity, kind="mergesort")[identity].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=True)
    return partitions


def build_policy_overlay(atoms: pd.DataFrame) -> pd.DataFrame:
    """Create and validate a policy-v2 overlay keyed by support and atom ID."""

    required = {
        "support_key",
        "atom_id",
        "category",
        "donor_player_id",
        "donor_event_key",
        "donor_event_start",
        "target_event_key",
        "target_event_start",
        "donor_skill_date",
        "donor_ability_available",
        "donor_ability_sg",
    }
    missing = sorted(required.difference(atoms.columns))
    if missing:
        raise ValueError(f"Population atoms miss policy inputs={missing}")
    category = atoms["category"].astype(str)
    if not category.isin(ABILITY_SOURCE_COLUMN).all():
        raise ValueError("Policy overlay received a non-ARG/PUTT category")
    raw = pd.DataFrame(
        {
            "bridge_row_id": (
                atoms["support_key"].astype(str) + "|" + atoms["atom_id"].astype(str)
            ),
            "category": category,
            "donor_player_id": atoms["donor_player_id"].astype(str),
            "donor_event_key": atoms["donor_event_key"].astype(str),
            "donor_event_start": atoms["donor_event_start"],
            "target_event_key": atoms["target_event_key"].astype(str),
            "target_event_start": atoms["target_event_start"],
            "donor_skill_date": atoms["donor_skill_date"],
            "donor_ability_source_column": category.map(ABILITY_SOURCE_COLUMN),
            "donor_ability_available": atoms["donor_ability_available"].astype(bool),
            "donor_ability_raw_strict_prior": pd.to_numeric(
                atoms["donor_ability_sg"], errors="coerce"
            ),
        }
    )
    if raw["bridge_row_id"].duplicated().any():
        raise PopulationAll18Error("Policy bridge_row_id is not unique")
    evaluated = ability_policy.apply_frozen_policy(raw)
    ability_policy.validate_applied_frame(evaluated)
    return evaluated.loc[:, list(POLICY_PERSISTED_COLUMNS)].reset_index(drop=True)


def validate_policy_overlay(
    overlay: pd.DataFrame,
    atoms: pd.DataFrame,
) -> dict[str, object]:
    ability_policy.validate_applied_frame(overlay)
    expected = build_policy_overlay(atoms)
    try:
        pd.testing.assert_frame_equal(
            overlay.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
            check_exact=True,
            check_categorical=False,
        )
    except AssertionError as exc:
        raise PopulationAll18Error("Persisted policy overlay changed") from exc
    kind = overlay["donor_ability_value_kind"].astype(str)
    observed = kind.eq(ability_policy.OBSERVED_VALUE_KIND)
    imputed = kind.eq(ability_policy.IMPUTED_VALUE_KIND)
    if not (observed | imputed).all():
        raise PopulationAll18Error("Policy overlay contains an unknown value kind")
    if not overlay.loc[imputed, "donor_ability_sg"].eq(0.0).all():
        raise PopulationAll18Error("Imputed donor abilities are not exact zero")
    if overlay.loc[imputed, "donor_ability_available"].astype(bool).any():
        raise PopulationAll18Error("Imputed donor abilities were marked observed")
    return {
        "rows": int(len(overlay)),
        "observed_rows": int(observed.sum()),
        "imputed_rows": int(imputed.sum()),
        "clipped_rows": int(
            overlay["donor_ability_was_clipped"].astype(bool).sum()
        ),
        "dated_missing_rows": int(
            (imputed & overlay["donor_skill_date"].notna()).sum()
        ),
        "undated_missing_rows": int(
            (imputed & overlay["donor_skill_date"].isna()).sum()
        ),
        "policy_identity_sha256": str(
            overlay["donor_ability_policy_sha256"].astype(str).iloc[0]
        ),
        "all_exact": True,
    }


def partition_relative_dir(key: tuple[str, int, int, str]) -> Path:
    arm, round_no, hole_no, category = key
    if arm not in {"raw", "normalized"} or category not in POPULATION_CATEGORIES:
        raise ValueError(f"Invalid partition key={key}")
    if round_no != 1 or not 1 <= hole_no <= 18:
        raise ValueError(f"Invalid R1 all-18 partition key={key}")
    return Path("partitions") / arm / f"R{round_no}H{hole_no:02d}" / category


def write_verified_parquet(frame: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    persisted = pd.read_parquet(path)
    try:
        pd.testing.assert_frame_equal(
            persisted,
            frame.reset_index(drop=True),
            check_dtype=False,
            check_exact=True,
            check_categorical=False,
        )
    except AssertionError as exc:
        raise PopulationAll18Error(
            f"Parquet payload differs from the in-memory frame: {path}"
        ) from exc
    try:
        pd.testing.assert_frame_equal(
            persisted,
            pd.read_parquet(path),
            check_exact=True,
            check_categorical=False,
        )
    except AssertionError as exc:
        raise PopulationAll18Error(f"Parquet readback is unstable: {path}") from exc
    return persisted, streaming.frame_logical_sha256(persisted)


def _prepare_authorities(
    manifest: Mapping[str, object],
    *,
    export_root: Path,
    history_db: Path,
    profile_dir: Path,
    baseline_cases: Path,
    streaming_reference_dir: Path,
    reference_value_dir: Path,
    bridge_path: Path,
):
    prepared = streaming.prepare_streaming_event(
        export_root=Path(export_root),
        history_db=Path(history_db),
        profile_dir=Path(profile_dir),
        baseline_cases=Path(baseline_cases),
        reference_dir=Path(streaming_reference_dir),
        event_name=ledger_v6.precompute.DEFAULT_EVENT,
        holdout_year=ledger_v6.precompute.DEFAULT_HOLDOUT,
        course_start_year=2020,
        roster_seed=streaming.DEFAULT_SEED,
    )
    expected_inputs = manifest["inputs"]["prepared_input_sha256"]
    expected_logical = manifest["inputs"]["prepared_logical_sha256"]
    if prepared.input_hashes != expected_inputs or prepared.logical_hashes != expected_logical:
        raise PopulationAll18Error("Prepared event inputs differ from the parent trace")
    prior = pd.DataFrame(
        [
            {"year": int(year), "event_key": str(event_key)}
            for year, event_key in sorted(prepared.atoms.prepared["event_keys"].items())
            if int(year) < int(prepared.atoms.holdout_year)
        ]
    )
    bundle = ledger_v6.traced_population.build_strict_prior_course_trace_bundle(
        Path(export_root),
        prior,
        prepared.atoms.target_event_key,
        categories=POPULATION_CATEGORIES,
        min_exact=12,
        knn=64,
    )
    transition_hash = streaming.frame_logical_sha256(bundle.transitions)
    expected_transition_hash = manifest["inputs"][
        "population_transition_frame_logical_sha256"
    ]
    if transition_hash != expected_transition_hash:
        raise PopulationAll18Error("Strict-prior population transition frame changed")
    sampler = bundle.traced_sampler(chunk_size=128)
    reference_model = ledger_v6.fixed_reference.ReferenceValueModel.load(
        Path(reference_value_dir)
    )
    if reference_model.model_sha256 != manifest["reference"]["function_sha256"]:
        raise PopulationAll18Error("Fixed reference function changed")
    if reference_model.putting_curve_sha256 != manifest["reference"]["putting_curve_sha256"]:
        raise PopulationAll18Error("Fixed putting reference changed")
    bridge, bridge_meta = ledger_v6.asof.load_strict_bridge(Path(bridge_path))
    if bridge_meta != manifest["bridge"]:
        raise PopulationAll18Error("Strict-prior bridge metadata changed")
    authority = {
        "target_event_key": str(prepared.atoms.target_event_key),
        "target_event_start": prepared.atoms.target_event_start.isoformat(),
        "prior_events": prior.to_dict("records"),
        "transition_frame_logical_sha256": transition_hash,
        "transition_rows": int(len(bundle.transitions)),
        "reference_function_sha256": str(reference_model.model_sha256),
        "reference_putting_sha256": str(reference_model.putting_curve_sha256),
        "bridge": bridge_meta,
    }
    # The population bundle and sampler are self-contained after construction.
    del prepared
    gc.collect()
    return sampler, reference_model, bridge, authority


def _output_index(output_dir: Path) -> dict[str, object]:
    files: dict[str, dict[str, object]] = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_symlink():
            raise PopulationAll18Error("Output contains a symlink")
        if not path.is_file():
            continue
        relative = str(path.relative_to(output_dir)).replace("\\", "/")
        if relative == "OUTPUT_SHA256.json":
            continue
        files[relative] = {
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    directories = sorted(
        str(path.relative_to(output_dir)).replace("\\", "/")
        for path in output_dir.rglob("*")
        if path.is_dir() and not path.is_symlink()
    )
    return {
        "contract": OUTPUT_INDEX_CONTRACT,
        "directories": directories,
        "files": files,
    }


def _phase_totals(rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
    rows = list(rows)
    return {
        "partitions": int(len(rows)),
        "selected_ledger_rows": int(sum(int(row["selected_ledger_rows"]) for row in rows)),
        "supports": int(sum(int(row["supports"]) for row in rows)),
        "atoms": int(sum(int(row["atoms"]) for row in rows)),
        "policy_observed_rows": int(sum(int(row["policy"]["observed_rows"]) for row in rows)),
        "policy_imputed_rows": int(sum(int(row["policy"]["imputed_rows"]) for row in rows)),
        "policy_clipped_rows": int(sum(int(row["policy"]["clipped_rows"]) for row in rows)),
    }


def build_all18_population_support(
    *,
    parent_dir: Path = DEFAULT_PARENT,
    output_dir: Path = DEFAULT_OUTPUT,
    export_root: Path = ledger_v6.precompute.DEFAULT_EXPORT,
    history_db: Path = ledger_v6.asof.DEFAULT_HISTORY_DB,
    profile_dir: Path = ledger_v6.precompute.DEFAULT_PROFILES,
    baseline_cases: Path = ledger_v6.precompute.DEFAULT_BASELINES,
    streaming_reference_dir: Path = streaming.DEFAULT_REFERENCE,
    reference_value_dir: Path = ledger_v6.DEFAULT_REFERENCE_DIR,
    bridge_path: Path = ledger_v6.asof.DEFAULT_BRIDGE,
    rss_cap_bytes: int = RSS_CAP_BYTES,
) -> Path:
    """Build, validate, and seal the partitioned all-18 support artifact."""

    output_dir = Path(output_dir)
    rss_cap_bytes = validate_rss_cap(rss_cap_bytes)
    if output_dir.exists():
        raise FileExistsError(f"Output path already exists: {output_dir}")
    manifest, population, parent_admission = admit_parent(Path(parent_dir))
    source_admission = validate_frozen_sources(manifest)
    source_path = Path(__file__)
    source_start = sha256_file(source_path)
    dependency_start = hash_paths(dependency_paths())
    monitor = streaming.RssPeakMonitor()
    monitor.start()
    started = time.perf_counter()
    try:
        sampler, reference_model, bridge, authority = _prepare_authorities(
            manifest,
            export_root=Path(export_root),
            history_db=Path(history_db),
            profile_dir=Path(profile_dir),
            baseline_cases=Path(baseline_cases),
            streaming_reference_dir=Path(streaming_reference_dir),
            reference_value_dir=Path(reference_value_dir),
            bridge_path=Path(bridge_path),
        )
        monitor.assert_below(rss_cap_bytes, "authority preparation")
        partitions = partition_population_ledger(population)
        output_dir.mkdir(parents=True)
        partition_rows: list[dict[str, object]] = []
        inventory_parts: list[pd.DataFrame] = []
        for sequence, (key, ledger_part) in enumerate(partitions):
            relative = partition_relative_dir(key)
            partition_dir = output_dir / relative
            atoms, inventory, metadata = (
                population_contract.materialize_population_support_contract(
                    ledger_part, sampler, reference_model, bridge
                )
            )
            overlay = build_policy_overlay(atoms)
            atoms_path = partition_dir / "support_atoms.parquet"
            inventory_path = partition_dir / "support_inventory.parquet"
            overlay_path = partition_dir / "donor_ability_policy.parquet"
            atoms, atoms_logical = write_verified_parquet(atoms, atoms_path)
            inventory, inventory_logical = write_verified_parquet(
                inventory, inventory_path
            )
            overlay, overlay_logical = write_verified_parquet(overlay, overlay_path)
            structural_audit = population_contract.validate_population_support_contract(
                atoms,
                inventory,
                ledger_part,
                sampler,
                reference_model,
                bridge,
                expected_metadata=metadata,
            )
            policy_audit = validate_policy_overlay(overlay, atoms)
            partition_manifest = {
                "contract": CONTRACT,
                "partition_sequence": int(sequence),
                "key": {
                    name: value for name, value in zip(PARTITION_COLUMNS, key)
                },
                "relative_dir": str(relative).replace("\\", "/"),
                "selected_ledger_rows": int(len(ledger_part)),
                "supports": int(len(inventory)),
                "atoms": int(len(atoms)),
                "structural_audit": structural_audit,
                "structural_metadata": metadata,
                "policy": policy_audit,
                "logical_sha256": {
                    "support_atoms": atoms_logical,
                    "support_inventory": inventory_logical,
                    "donor_ability_policy": overlay_logical,
                },
                "file_sha256": {
                    "support_atoms": sha256_file(atoms_path),
                    "support_inventory": sha256_file(inventory_path),
                    "donor_ability_policy": sha256_file(overlay_path),
                },
            }
            _write_json(partition_dir / "partition_manifest.json", partition_manifest)
            partition_manifest["partition_manifest_sha256"] = sha256_file(
                partition_dir / "partition_manifest.json"
            )
            partition_rows.append(partition_manifest)
            inventory_parts.append(
                inventory.assign(
                    partition_sequence=int(sequence),
                    partition_relative_dir=str(relative).replace("\\", "/"),
                )
            )
            del atoms, inventory, overlay, metadata, structural_audit
            gc.collect()
            monitor.assert_below(rss_cap_bytes, f"partition {sequence}")

        global_inventory = pd.concat(inventory_parts, ignore_index=True, sort=False)
        global_inventory = global_inventory.sort_values(
            ["partition_sequence", "support_key"], kind="mergesort"
        ).reset_index(drop=True)
        global_inventory, global_inventory_logical = write_verified_parquet(
            global_inventory, output_dir / "support_inventory.parquet"
        )
        totals = _phase_totals(partition_rows)
        if totals["selected_ledger_rows"] != len(population):
            raise PopulationAll18Error("Partitioned selected-row total changed")
        if totals["supports"] != len(global_inventory):
            raise PopulationAll18Error("Partitioned support total changed")
        if global_inventory["support_key"].duplicated().any():
            raise PopulationAll18Error("A support appears in multiple partitions")
        dependency_end = hash_paths(dependency_paths())
        source_end = sha256_file(source_path)
        if source_start != source_end or dependency_start != dependency_end:
            raise RuntimeError("Population support source/dependency changed during run")
        peak = monitor.stop()
        if peak > rss_cap_bytes:
            raise MemoryError(
                f"Population support RSS exceeded cap: {peak / 2**30:.3f} GiB"
            )
        elapsed = float(time.perf_counter() - started)
        manifest_out = {
            "contract": CONTRACT,
            "research_only": True,
            "production_untouched": True,
            "anchor_solved": False,
            "anchor_authorized": False,
            "complete_population_neighbourhoods": True,
            "donor_ability_policy_overlay": {
                "contract": POLICY_OVERLAY_CONTRACT,
                "policy_contract": ability_policy.POLICY_CONTRACT,
                "policy_identity_sha256": EXPECTED_ABILITY_POLICY_IDENTITY,
                "missing_rows_remain_unavailable": True,
                "missing_numeric_value": 0.0,
                "current_chain_adapter_v3_accepts_policy_missing": False,
            },
            "parent_admission": parent_admission,
            "source_admission": source_admission,
            "authority": authority,
            "axes": {
                "event_key": manifest["axes"]["event_key"],
                "player_id": manifest["axes"]["player_id"],
                "rounds": [1],
                "holes": list(range(1, 19)),
                "arms": ["raw", "normalized"],
                "categories": list(POPULATION_CATEGORIES),
            },
            "totals": totals,
            "partitions": partition_rows,
            "support_inventory_logical_sha256": global_inventory_logical,
            "rss": {
                "cap_bytes": rss_cap_bytes,
                "peak_bytes": int(peak),
                "under_cap": bool(peak <= rss_cap_bytes),
                "continuous_sampled_heavy_phases": True,
                "scope": (
                    "authority_preparation_through_partition_materialization_"
                    "validation_and_global_inventory; excludes tiny final JSON/Markdown seal"
                ),
            },
            "elapsed_seconds": elapsed,
            "source_sha256_start": source_start,
            "source_sha256_end": source_end,
            "dependency_sha256_start": dependency_start,
            "dependency_sha256_end": dependency_end,
        }
        _write_json(output_dir / "manifest.json", manifest_out)
        report = "\n".join(
            [
                "# Partitioned R1 all-18 population supports",
                "",
                f"- Selected ARG/PUTT ledger rows: {totals['selected_ledger_rows']:,}",
                f"- Exact query supports: {totals['supports']:,}",
                f"- Complete population atoms: {totals['atoms']:,}",
                f"- Observed donor abilities: {totals['policy_observed_rows']:,}",
                f"- Explicit neutral-imputed donor abilities: {totals['policy_imputed_rows']:,}",
                f"- Peak RSS: {peak / 2**30:.3f} GiB (cap {rss_cap_bytes / 2**30:.3f} GiB)",
                "",
                "Mechanics and the frozen donor-ability policy are sealed separately. "
                "No mean anchor was solved and the current adapter contract does not yet "
                "admit policy-imputed rows.",
                "",
            ]
        )
        (output_dir / "REPORT.md").write_text(report, encoding="utf-8", newline="\n")
        index = _output_index(output_dir)
        _write_json(output_dir / "OUTPUT_SHA256.json", index)
        verify_output_index(output_dir)
        if source_start != sha256_file(source_path):
            raise RuntimeError("Population runner source changed during final sealing")
        if dependency_start != hash_paths(dependency_paths()):
            raise RuntimeError("Population dependency changed during final sealing")
        return output_dir
    except BaseException:
        # The monitor is stopped on every error; the partial versioned output is
        # intentionally left in place for forensic inspection, never reused.
        try:
            monitor.stop()
        except BaseException:
            pass
        raise


def dry_run(parent_dir: Path = DEFAULT_PARENT, output_dir: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    if Path(output_dir).exists():
        raise FileExistsError(f"Output path already exists: {output_dir}")
    manifest, population, admission = admit_parent(Path(parent_dir))
    sources = validate_frozen_sources(manifest)
    partitions = partition_population_ledger(population)
    return {
        "contract": CONTRACT,
        "admission_valid": True,
        "simulation_launched": False,
        "parent": admission,
        "expected_partitions": int(len(partitions)),
        "selected_population_rows": int(len(population)),
        "unique_queries": int(
            population.loc[:, list(population_contract.QUERY_COLUMNS)]
            .drop_duplicates()
            .shape[0]
        ),
        "maximum_atoms": int(
            population.loc[:, list(population_contract.QUERY_COLUMNS)]
            .drop_duplicates()
            .shape[0]
            * 64
        ),
        "event_key": manifest["axes"]["event_key"],
        "player_id": manifest["axes"]["player_id"],
        "source_sha256": sources,
        "output_absent": not Path(output_dir).exists(),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--export-root", type=Path, default=ledger_v6.precompute.DEFAULT_EXPORT)
    parser.add_argument("--history-db", type=Path, default=ledger_v6.asof.DEFAULT_HISTORY_DB)
    parser.add_argument("--profile-dir", type=Path, default=ledger_v6.precompute.DEFAULT_PROFILES)
    parser.add_argument("--baseline-cases", type=Path, default=ledger_v6.precompute.DEFAULT_BASELINES)
    parser.add_argument("--streaming-reference-dir", type=Path, default=streaming.DEFAULT_REFERENCE)
    parser.add_argument("--reference-value-dir", type=Path, default=ledger_v6.DEFAULT_REFERENCE_DIR)
    parser.add_argument("--bridge", type=Path, default=ledger_v6.asof.DEFAULT_BRIDGE)
    parser.add_argument("--rss-cap-gib", type=float, default=4.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        print(json.dumps(dry_run(args.parent_dir, args.output_dir), indent=2, default=str))
        return 0
    output = build_all18_population_support(
        parent_dir=args.parent_dir,
        output_dir=args.output_dir,
        export_root=args.export_root,
        history_db=args.history_db,
        profile_dir=args.profile_dir,
        baseline_cases=args.baseline_cases,
        streaming_reference_dir=args.streaming_reference_dir,
        reference_value_dir=args.reference_value_dir,
        bridge_path=args.bridge,
        rss_cap_bytes=int(float(args.rss_cap_gib) * 2**30),
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONTRACT",
    "DEFAULT_OUTPUT",
    "DEFAULT_PARENT",
    "PopulationAll18Error",
    "admit_parent",
    "build_all18_population_support",
    "build_policy_overlay",
    "dry_run",
    "partition_population_ledger",
    "validate_rss_cap",
    "validate_policy_overlay",
    "verify_output_index",
]
