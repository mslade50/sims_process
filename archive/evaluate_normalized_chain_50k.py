"""Decision gate for beta-zero normalized shot-chain round distributions.

This module is research-only.  It consumes persisted physical integer round
totals; it never calls a simulator, production pricing, storage, or publishing
code.  Its central comparison is deliberately narrow:

``raw``
    The frozen strict-prior donor chain.

``normalized``
    The beta-zero ``player_state_residualized`` donor chain with the target
    spatial profile reapplied.

Both arms must already have the *same production skill mean* inside the
generator.  This evaluator neither recentres a sample nor applies a latent,
temperature, variance, or lattice adapter.  An unanchored 5k/50k mechanical
artifact can pass the input audit, but it cannot be scored predictively.

The one-round scores use one fixed physical-stroke lattice for every arm and
case.  This prevents the old 128-path failure modes: no support is inferred
from a candidate, no arm is translated onto another coordinate system, and no
low-particle quadrature is passed off as a predictive distribution.

The public API is intentionally small:

``load_particle_panel``
    Validate and combine one-player streaming bundles.

``audit_schema_only``
    Accept a 5k engineering artifact without producing predictive metrics.

``evaluate_panel``
    Score an anchored panel with at least 50,000 particles per player-round.

The CLI writes only a new versioned research directory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:  # Works both as ``archive.evaluate...`` and as an archive-path script.
    from . import hole_sim_tail_policy as tail_policy
except ImportError:  # pragma: no cover - exercised by direct CLI invocation
    import hole_sim_tail_policy as tail_policy


CONTRACT_VERSION = "normalized_chain_round_evaluation_v2_tail_policy"
RAW_ARM = "raw"
NORMALIZED_ARM = "normalized"
ARMS = (RAW_ARM, NORMALIZED_ARM)
NORMALIZATION_MODE = "player_state_residualized"
MECHANICS_ANCHOR_STATUS = "not_calibrated_mechanics_only"
REQUIRED_ANCHOR_STATUS = "generator_time_shared_production_skill_anchor_v1"
REQUIRED_ANCHOR_POLICY = "shared_production_skill_mean_pre_persistence_v1"
REQUIRED_TRANSFORM_ORDER = (
    "donor_native_asof_normalize",
    "rigid_align_neutral_endpoint_exactly_once",
    "target_profile_reapply_beta_zero",
    "runtime_physical_surface_reclassification",
)

# 5k exists to validate vectorization/runtime/schema.  It is never evidence
# about predictive quality.  50k is a necessary particle floor, not proof that
# a one-event or one-player panel is representative.
MIN_SCHEMA_PARTICLES = 5_000
MIN_DECISION_PARTICLES = 50_000
MIN_DECISION_EVENTS = 5
MIN_DECISION_PRIMARY_CASES = 100

# One universal, predeclared, unit-spaced, physical 18-hole stroke lattice.
# Values outside it fail closed; they are never clipped or used to expand an
# arm-specific support after observing the result.
LATTICE_MIN_STROKES = tail_policy.ROUND_LATTICE_MIN_STROKES
LATTICE_MAX_STROKES = tail_policy.ROUND_LATTICE_MAX_STROKES
PHYSICAL_STROKE_LATTICE = np.arange(
    LATTICE_MIN_STROKES, LATTICE_MAX_STROKES + 1, dtype=np.int16
)
JEFFREYS_ALPHA = 0.5
PIT_SALT = "normalized_chain_pit_v1"

CUT_RULE_NO_CUT = "no_cut"
CUT_RULE_AFTER_R2 = "cut_after_r2"
CUT_RULES = (CUT_RULE_NO_CUT, CUT_RULE_AFTER_R2)

CASE_REQUIRED_COLUMNS = {
    "case_id",
    "event_key",
    "event_name",
    "event_start",
    "source_player_id",
    "player_name",
    "round_num",
    "round_par",
    "cut_rule",
    "pre_event_roster",
    "roster_frozen_at",
    "skill_asof",
    "input_cutoff",
    "production_skill_sg",
    "production_field_mean_strokes",
    "production_mean_strokes",
    "generator_expected_mean_strokes_raw",
    "generator_expected_mean_strokes_normalized",
    "mean_anchor_policy",
}
TRUTH_REQUIRED_COLUMNS = {
    "event_key",
    "source_player_id",
    "round_num",
    "actual_strokes",
}
PAIR_REQUIRED_COLUMNS = {
    "pair_id",
    "event_key",
    "round_num",
    "player_a_id",
    "player_b_id",
    "pair_frozen_at",
    "pair_source",
}


class EvaluationContractError(ValueError):
    """The artifact violates the frozen evaluation contract."""


class PredictiveScoringBlocked(EvaluationContractError):
    """The file is valid engineering output but cannot be scored predictively."""


def sha256_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array))
    digest = sha256()
    digest.update(str(value.shape).encode("utf-8"))
    digest.update(value.dtype.str.encode("utf-8"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def _nested(mapping: Mapping[str, object], *keys: str, default: object = None) -> object:
    value: object = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


@dataclass(frozen=True)
class BundleAudit:
    bundle_dir: Path
    manifest_path: Path
    totals_path: Path
    manifest_sha256: str
    totals_sha256: str
    event_key: str
    player_id: str
    particles: int
    anchor_status: str
    counter_version: str
    streaming_source_sha256: str
    strict_particle_axis: bool
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ParticlePanel:
    """Aligned raw/normalized round totals at [arm, case, round, particle]."""

    totals: np.ndarray
    case_ids: tuple[str, ...]
    round_numbers: tuple[int, ...]
    particle_ids: np.ndarray
    audits: tuple[BundleAudit, ...]
    input_hashes_start: Mapping[str, str]

    @property
    def particles(self) -> int:
        return int(self.totals.shape[-1])

    @property
    def cases(self) -> int:
        return int(self.totals.shape[1])

    def arm(self, name: str) -> np.ndarray:
        if name not in ARMS:
            raise KeyError(name)
        return self.totals[ARMS.index(name)]


@dataclass(frozen=True)
class EvaluationResult:
    case_metrics: pd.DataFrame
    paired_case_deltas: pd.DataFrame
    aggregate_metrics: pd.DataFrame
    anchor_audit: pd.DataFrame
    pit_histogram: pd.DataFrame
    h2h_metrics: pd.DataFrame
    paired_h2h_deltas: pd.DataFrame
    decision_status: Mapping[str, object]
    input_hashes: Mapping[str, str]


def _validate_manifest(
    manifest: Mapping[str, object],
    *,
    totals_hash: str,
) -> tuple[str, list[str]]:
    warnings: list[str] = []
    if manifest.get("research_only") is not True:
        raise EvaluationContractError("Streaming manifest must be research_only=true")
    if manifest.get("production_touched") is not False:
        raise EvaluationContractError("Streaming manifest must say production_touched=false")
    try:
        tail_policy.validate_manifest_policy(
            manifest.get("tail_policy"), label="streaming manifest tail_policy"
        )
    except ValueError as error:
        raise EvaluationContractError(str(error)) from error
    if set(map(str, manifest.get("arms", []))) != set(ARMS):
        raise EvaluationContractError("Streaming bundle must contain exactly raw and normalized")
    if str(manifest.get("normalization_mode")) != NORMALIZATION_MODE:
        raise EvaluationContractError(
            f"normalization_mode must be {NORMALIZATION_MODE!r}"
        )
    if float(manifest.get("ability_log_weight_per_sg2", math.nan)) != 0.0:
        raise EvaluationContractError("Decision comparison requires beta exactly zero")
    if str(manifest.get("round_totals_sha256")) != totals_hash:
        raise EvaluationContractError("round_totals_int16.npz hash disagrees with manifest")
    source_start = manifest.get("streaming_source_start_sha256")
    source_end = manifest.get("streaming_source_end_sha256")
    if not source_start or source_start != source_end:
        raise EvaluationContractError("Streaming source start/end hashes must match")
    if _nested(manifest, "streaming_contract", "survivor_filtering") is not False:
        raise EvaluationContractError("Survivor filtering must be explicitly false")
    if _nested(manifest, "streaming_contract", "unsupported_states_fail_closed") is not True:
        raise EvaluationContractError("Unsupported states must fail closed")
    if _nested(manifest, "rng_modes", "benchmark") != "counter":
        raise EvaluationContractError("Persisted arm comparison must use counter RNG")
    if _nested(manifest, "rng_modes", "arm_in_key") is not False:
        raise EvaluationContractError("Arm may not be part of the CRN counter key")
    if _nested(manifest, "rng_modes", "chunk_in_key") is not False:
        raise EvaluationContractError("Chunk may not be part of the CRN counter key")
    counter_version = _nested(manifest, "rng_modes", "counter_version")
    if not isinstance(counter_version, str) or not counter_version.strip():
        raise EvaluationContractError("Counter RNG version must be explicit")
    prefix_exact = _nested(
        manifest,
        "raw_128_gate",
        "counter_50k_prefix_of_100k",
        "exact",
    )
    if prefix_exact is not True:
        warnings.append("producer did not attest exact 50k prefix of 100k")

    anchor = manifest.get("production_mean_anchor_status")
    if anchor is None:
        anchor = _nested(
            manifest, "mean_anchor_contract", "production_mean_anchor_status"
        )
    if anchor is None:
        anchor = MECHANICS_ANCHOR_STATUS
        warnings.append("mean anchor status absent; treated as mechanics-only")
    anchor = str(anchor)
    return anchor, warnings


def _load_one_bundle(bundle_dir: Path) -> tuple[np.ndarray, str, BundleAudit, dict[str, str]]:
    bundle_dir = Path(bundle_dir).resolve()
    manifest_path = bundle_dir / "manifest.json"
    totals_path = bundle_dir / "round_totals_int16.npz"
    if not manifest_path.exists() or not totals_path.exists():
        raise FileNotFoundError(
            f"Bundle requires manifest.json and round_totals_int16.npz: {bundle_dir}"
        )
    hashes = {
        str(manifest_path): sha256_file(manifest_path),
        str(totals_path): sha256_file(totals_path),
    }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    anchor_status, warnings = _validate_manifest(
        manifest, totals_hash=hashes[str(totals_path)]
    )
    with np.load(totals_path, allow_pickle=False) as payload:
        required = {"arm_names", "round_totals", "round_numbers"}
        missing = sorted(required - set(payload.files))
        if missing:
            raise EvaluationContractError(f"Particle NPZ is missing {missing}")
        arm_names = tuple(payload["arm_names"].astype(str).tolist())
        if len(arm_names) != 2 or set(arm_names) != set(ARMS):
            raise EvaluationContractError("NPZ arm_names must be raw and normalized")
        totals = np.asarray(payload["round_totals"])
        if totals.dtype != np.int16:
            raise EvaluationContractError("Persisted round totals must be int16")
        if totals.ndim == 4:
            if totals.shape[1] != 1:
                raise EvaluationContractError(
                    "Each streaming bundle must represent one player case"
                )
            totals = totals[:, 0]
        if totals.ndim != 3 or totals.shape[0] != 2 or totals.shape[1] != 4:
            raise EvaluationContractError(
                "round_totals must have shape [2 arms, 4 rounds, particles]"
            )
        order = np.asarray([arm_names.index(name) for name in ARMS], dtype=int)
        totals = np.ascontiguousarray(totals[order])
        rounds = tuple(map(int, payload["round_numbers"].tolist()))
        if rounds != (1, 2, 3, 4):
            raise EvaluationContractError("round_numbers must be exactly [1,2,3,4]")
        event_values = payload["event_key"].astype(str) if "event_key" in payload else np.asarray([])
        player_values = payload["player_id"].astype(str) if "player_id" in payload else np.asarray([])
        if len(event_values) != 1 or len(player_values) != 1:
            raise EvaluationContractError("NPZ needs exactly one event_key and player_id")
        event_key, player_id = str(event_values[0]), str(player_values[0])
        expected_event = str(manifest.get("event_key"))
        expected_player = str(manifest.get("player_id"))
        if event_key != expected_event or player_id != expected_player:
            raise EvaluationContractError("NPZ event/player identity disagrees with manifest")
        case_id = f"{event_key}|{player_id}"

        strict_axis = False
        if "particle_ids_by_arm" in payload:
            axes = np.asarray(payload["particle_ids_by_arm"])
            if axes.shape != (2, totals.shape[-1]):
                raise EvaluationContractError("particle_ids_by_arm has the wrong shape")
            axes = axes[order]
            if not np.array_equal(axes[0], axes[1]):
                raise EvaluationContractError("Raw and normalized particle axes differ")
            expected_axis = np.arange(totals.shape[-1], dtype=axes.dtype)
            if not np.array_equal(axes[0], expected_axis):
                raise EvaluationContractError("Particle IDs must be contiguous zero-based")
            strict_axis = True
        elif "particle_ids" in payload:
            axis = np.asarray(payload["particle_ids"])
            if not np.array_equal(axis, np.arange(totals.shape[-1], dtype=axis.dtype)):
                raise EvaluationContractError("Particle IDs must be contiguous zero-based")
            strict_axis = True
        else:
            warnings.append(
                "particle IDs are implicit zero-based; alignment rests on producer CRN attestation"
            )

    if int(totals.min()) < LATTICE_MIN_STROKES or int(totals.max()) > LATTICE_MAX_STROKES:
        raise EvaluationContractError(
            f"Round total outside frozen [{LATTICE_MIN_STROKES},{LATTICE_MAX_STROKES}] lattice"
        )
    audit = BundleAudit(
        bundle_dir=bundle_dir,
        manifest_path=manifest_path,
        totals_path=totals_path,
        manifest_sha256=hashes[str(manifest_path)],
        totals_sha256=hashes[str(totals_path)],
        event_key=event_key,
        player_id=player_id,
        particles=int(totals.shape[-1]),
        anchor_status=anchor_status,
        counter_version=str(_nested(manifest, "rng_modes", "counter_version")),
        streaming_source_sha256=str(manifest["streaming_source_start_sha256"]),
        strict_particle_axis=strict_axis,
        warnings=tuple(warnings),
    )
    return totals, case_id, audit, hashes


def load_particle_panel(bundle_dirs: Sequence[Path]) -> ParticlePanel:
    """Load one or more single-player bundles without simulating anything."""

    if not bundle_dirs:
        raise EvaluationContractError("At least one streaming bundle is required")
    pieces: list[np.ndarray] = []
    case_ids: list[str] = []
    audits: list[BundleAudit] = []
    hashes: dict[str, str] = {}
    particle_counts: set[int] = set()
    for path in bundle_dirs:
        totals, case_id, audit, file_hashes = _load_one_bundle(Path(path))
        pieces.append(totals)
        case_ids.append(case_id)
        audits.append(audit)
        hashes.update(file_hashes)
        particle_counts.add(audit.particles)
    if len(set(case_ids)) != len(case_ids):
        raise EvaluationContractError("Duplicate event/player case across bundles")
    if len(particle_counts) != 1:
        raise EvaluationContractError("Every bundle must use the same particle count")
    if len({audit.counter_version for audit in audits}) != 1:
        raise EvaluationContractError("Streaming bundles use different counter RNG versions")
    if len({audit.streaming_source_sha256 for audit in audits}) != 1:
        raise EvaluationContractError("Streaming bundles use different source-code versions")
    particles = particle_counts.pop()
    panel = np.stack(pieces, axis=1)  # [arm, case, round, particle]
    if panel.shape != (2, len(case_ids), 4, particles):
        raise AssertionError("Combined particle panel has an unexpected shape")
    return ParticlePanel(
        totals=np.ascontiguousarray(panel),
        case_ids=tuple(case_ids),
        round_numbers=(1, 2, 3, 4),
        particle_ids=np.arange(particles, dtype=np.int64),
        audits=tuple(audits),
        input_hashes_start=hashes,
    )


def audit_schema_only(panel: ParticlePanel) -> dict[str, object]:
    """Return mechanics/schema status and explicitly withhold predictive claims."""

    particles_ok = panel.particles >= MIN_SCHEMA_PARTICLES
    statuses = sorted(set(audit.anchor_status for audit in panel.audits))
    return {
        "contract_version": CONTRACT_VERSION,
        "schema_valid": True,
        "engineering_particle_floor_passed": particles_ok,
        "particles_per_player_round": panel.particles,
        "player_cases": panel.cases,
        "arms": list(ARMS),
        "rounds": list(panel.round_numbers),
        "physical_stroke_lattice": [LATTICE_MIN_STROKES, LATTICE_MAX_STROKES],
        "tail_policy": tail_policy.manifest_policy(),
        "anchor_statuses": statuses,
        "all_bundles_generator_anchored": statuses == [REQUIRED_ANCHOR_STATUS],
        "decision_particle_floor_passed": panel.particles >= MIN_DECISION_PARTICLES,
        "predictive_scoring_permitted": False,
        "decision_claim_permitted": False,
        "status": "schema_and_mechanics_only",
        "reason": (
            "Schema-only mode never computes predictive performance; 5k is a "
            "runtime/mechanics gate and an unanchored artifact remains mechanics-only "
            "at any particle count."
        ),
        "bundle_audits": [_jsonable(audit.__dict__) for audit in panel.audits],
        "input_hashes_sha256": dict(panel.input_hashes_start),
    }


def _load_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.casefold() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.casefold() == ".csv":
        return pd.read_csv(path)
    raise EvaluationContractError(f"Unsupported table format: {path}")


def load_case_index(path: Path, panel: ParticlePanel) -> pd.DataFrame:
    frame = _load_table(path).copy()
    missing = sorted(CASE_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise EvaluationContractError(f"Case index is missing {missing}")
    frame["case_id"] = frame["case_id"].astype(str).str.strip()
    frame["event_key"] = frame["event_key"].astype(str).str.strip()
    frame["source_player_id"] = frame["source_player_id"].astype(str).str.strip()
    frame["player_name"] = frame["player_name"].astype(str).str.strip()
    if not frame["player_name"].eq(frame["player_name"].str.casefold()).all():
        raise EvaluationContractError("Player names must already be canonical lower-case")
    for column in ("event_start", "roster_frozen_at", "skill_asof", "input_cutoff"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)
    numeric = [
        "round_num",
        "round_par",
        "production_skill_sg",
        "production_field_mean_strokes",
        "production_mean_strokes",
        "generator_expected_mean_strokes_raw",
        "generator_expected_mean_strokes_normalized",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[numeric].isna().any().any() or frame[
        ["event_start", "roster_frozen_at", "skill_asof", "input_cutoff"]
    ].isna().any().any():
        raise EvaluationContractError("Case index contains missing/invalid values")
    if not np.isfinite(frame[numeric].to_numpy(float)).all():
        raise EvaluationContractError("Case index numeric values must be finite")
    for column in ("round_num", "round_par"):
        if not np.allclose(frame[column], np.rint(frame[column]), atol=0.0, rtol=0.0):
            raise EvaluationContractError(f"{column} must be integer")
        frame[column] = np.rint(frame[column]).astype(int)
    if not frame["round_num"].between(1, 4).all():
        raise EvaluationContractError("round_num must be 1..4")
    if not frame["cut_rule"].isin(CUT_RULES).all():
        raise EvaluationContractError(f"cut_rule must be one of {CUT_RULES}")
    roster = frame["pre_event_roster"]
    if not roster.map(lambda value: bool(value) if isinstance(value, (bool, np.bool_)) else False).all():
        raise EvaluationContractError("Every simulated case must be in the frozen pre-event roster")
    if not (frame["roster_frozen_at"] < frame["event_start"]).all():
        raise EvaluationContractError("Roster must be frozen strictly before event start")
    if not (frame["skill_asof"] < frame["event_start"]).all():
        raise EvaluationContractError("Production skill snapshot must strictly predate event start")
    if not (frame["input_cutoff"] < frame["event_start"]).all():
        raise EvaluationContractError("Every simulation input cutoff must predate event start")
    expected_mean = frame["production_field_mean_strokes"] - frame["production_skill_sg"]
    if not np.allclose(
        frame["production_mean_strokes"], expected_mean, atol=1e-10, rtol=0.0
    ):
        raise EvaluationContractError(
            "production_mean_strokes must equal pre-event field mean minus production skill SG"
        )
    if not frame["mean_anchor_policy"].eq(REQUIRED_ANCHOR_POLICY).all():
        raise EvaluationContractError(
            f"mean_anchor_policy must be {REQUIRED_ANCHOR_POLICY!r}"
        )
    for column in (
        "generator_expected_mean_strokes_raw",
        "generator_expected_mean_strokes_normalized",
    ):
        if not np.allclose(
            frame[column], frame["production_mean_strokes"], atol=1e-10, rtol=0.0
        ):
            raise EvaluationContractError(
                f"{column} does not equal the shared production mean anchor"
            )
    duplicate_key = ["case_id", "round_num"]
    if frame.duplicated(duplicate_key).any():
        raise EvaluationContractError("Case index has duplicate player-round rows")
    for case_id, group in frame.groupby("case_id", observed=True, sort=False):
        if tuple(sorted(group["round_num"].tolist())) != (1, 2, 3, 4):
            raise EvaluationContractError(f"Case {case_id} does not contain R1-R4")
        fixed = group[
            ["event_key", "source_player_id", "event_name", "event_start", "cut_rule"]
        ].drop_duplicates()
        if len(fixed) != 1:
            raise EvaluationContractError(f"Case identity changes across rounds: {case_id}")
        identity = fixed.iloc[0]
        expected_case_id = f"{identity['event_key']}|{identity['source_player_id']}"
        if case_id != expected_case_id:
            raise EvaluationContractError(
                f"case_id must be event_key|source_player_id; got {case_id!r}"
            )
    if set(frame["case_id"]) != set(panel.case_ids):
        raise EvaluationContractError("Case index and particle bundles have different rosters")
    order = {case_id: index for index, case_id in enumerate(panel.case_ids)}
    frame["_case_order"] = frame["case_id"].map(order)
    return frame.sort_values(["_case_order", "round_num"]).drop(columns="_case_order").reset_index(drop=True)


def load_truth(path: Path, cases: pd.DataFrame) -> pd.DataFrame:
    truth = _load_table(path).copy()
    missing = sorted(TRUTH_REQUIRED_COLUMNS - set(truth.columns))
    if missing:
        raise EvaluationContractError(f"Truth table is missing {missing}")
    truth["event_key"] = truth["event_key"].astype(str).str.strip()
    truth["source_player_id"] = truth["source_player_id"].astype(str).str.strip()
    truth["round_num"] = pd.to_numeric(truth["round_num"], errors="coerce")
    truth["actual_strokes"] = pd.to_numeric(truth["actual_strokes"], errors="coerce")
    if truth[["round_num", "actual_strokes"]].isna().any().any():
        raise EvaluationContractError("Truth contains missing/invalid scores")
    if not np.allclose(truth["round_num"], np.rint(truth["round_num"]), atol=0.0, rtol=0.0):
        raise EvaluationContractError("Truth round_num must be integer")
    if not np.allclose(truth["actual_strokes"], np.rint(truth["actual_strokes"]), atol=0.0, rtol=0.0):
        raise EvaluationContractError("Truth actual_strokes must be integer physical strokes")
    truth["round_num"] = np.rint(truth["round_num"]).astype(int)
    truth["actual_strokes"] = np.rint(truth["actual_strokes"]).astype(int)
    if not truth["actual_strokes"].between(
        LATTICE_MIN_STROKES, LATTICE_MAX_STROKES
    ).all():
        raise EvaluationContractError("Truth falls outside the frozen physical stroke lattice")
    keys = ["event_key", "source_player_id", "round_num"]
    if truth.duplicated(keys).any():
        raise EvaluationContractError("Truth has duplicate player-round rows")
    joined = cases.merge(truth[keys + ["actual_strokes"]], on=keys, how="left", validate="one_to_one")
    joined["evaluation_scope"] = np.where(
        joined["cut_rule"].eq(CUT_RULE_NO_CUT) | joined["round_num"].le(2),
        "primary_no_survivor",
        "survivor_conditional_secondary",
    )
    primary = joined["evaluation_scope"].eq("primary_no_survivor")
    if joined.loc[primary, "actual_strokes"].isna().any():
        missing_cases = joined.loc[
            primary & joined["actual_strokes"].isna(),
            ["case_id", "round_num"],
        ]
        raise EvaluationContractError(
            "Primary no-survivor truth is incomplete: "
            f"{missing_cases.to_dict('records')[:8]}"
        )
    return joined


def pmf_on_common_lattice(
    samples: np.ndarray,
    *,
    alpha: float = JEFFREYS_ALPHA,
) -> np.ndarray:
    """Return one smoothed PMF on the frozen physical integer lattice."""

    values = np.asarray(samples)
    if values.ndim != 1 or len(values) == 0:
        raise EvaluationContractError("Round samples must be a non-empty vector")
    if values.dtype.kind not in "iu":
        numeric = values.astype(float)
        if not np.isfinite(numeric).all() or not np.allclose(
            numeric, np.rint(numeric), atol=0.0, rtol=0.0
        ):
            raise EvaluationContractError("Round samples must be integer physical strokes")
        values = np.rint(numeric).astype(int)
    if int(values.min()) < LATTICE_MIN_STROKES or int(values.max()) > LATTICE_MAX_STROKES:
        raise EvaluationContractError("Round sample is outside the frozen lattice")
    if not np.isfinite(alpha) or alpha < 0.0:
        raise EvaluationContractError("PMF smoothing alpha must be finite and non-negative")
    indices = values.astype(np.int64) - LATTICE_MIN_STROKES
    counts = np.bincount(indices, minlength=len(PHYSICAL_STROKE_LATTICE)).astype(float)
    probability = counts + float(alpha)
    probability /= probability.sum()
    return probability


def crps_common_lattice(probability: np.ndarray, actual_strokes: int) -> float:
    p = np.asarray(probability, dtype=float)
    if p.shape != PHYSICAL_STROKE_LATTICE.shape or np.any(p < 0.0):
        raise EvaluationContractError("Probability is not on the frozen lattice")
    if not np.isclose(p.sum(), 1.0, atol=1e-12, rtol=0.0):
        raise EvaluationContractError("Probability does not sum to one")
    actual = int(actual_strokes)
    if actual < LATTICE_MIN_STROKES or actual > LATTICE_MAX_STROKES:
        raise EvaluationContractError("Actual score is outside the frozen lattice")
    cdf = np.cumsum(p)
    observed_cdf = (PHYSICAL_STROKE_LATTICE >= actual).astype(float)
    # The final CDF term is identically zero in the squared difference.
    return float(np.sum((cdf[:-1] - observed_cdf[:-1]) ** 2))


def _case_uniform(event_key: str, player_id: str, round_num: int) -> float:
    key = f"{PIT_SALT}|{event_key}|{player_id}|{int(round_num)}".encode("utf-8")
    integer = int.from_bytes(sha256(key).digest()[:8], "little", signed=False)
    return (integer + 0.5) / float(2**64)


def _quantile(probability: np.ndarray, q: float) -> int:
    index = int(np.searchsorted(np.cumsum(probability), float(q), side="left"))
    return int(PHYSICAL_STROKE_LATTICE[min(index, len(probability) - 1)])


def score_round_pmf(
    probability: np.ndarray,
    *,
    actual_strokes: int,
    production_mean_strokes: float,
    event_key: str,
    player_id: str,
    round_num: int,
) -> dict[str, float | int]:
    """Proper one-round scores plus predeclared calibration/tail diagnostics."""

    p = np.asarray(probability, dtype=float)
    actual = int(actual_strokes)
    actual_index = actual - LATTICE_MIN_STROKES
    cdf = np.cumsum(p)
    lower = float(cdf[actual_index - 1]) if actual_index > 0 else 0.0
    mass = float(p[actual_index])
    upper = lower + mass
    randomised = lower + _case_uniform(event_key, player_id, round_num) * mass
    lattice = PHYSICAL_STROKE_LATTICE.astype(float)
    predictive_mean = float(p @ lattice)
    predictive_variance = float(p @ ((lattice - predictive_mean) ** 2))
    anchor_mse = float(p @ ((lattice - float(production_mean_strokes)) ** 2))
    q05, q10, q25, q50, q75, q90, q95 = (
        _quantile(p, q) for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
    )
    low3 = math.floor(float(production_mean_strokes) - 3.0)
    high3 = math.ceil(float(production_mean_strokes) + 3.0)
    low5 = math.floor(float(production_mean_strokes) - 5.0)
    high5 = math.ceil(float(production_mean_strokes) + 5.0)

    def p_le(threshold: int) -> float:
        if threshold < LATTICE_MIN_STROKES:
            return 0.0
        if threshold >= LATTICE_MAX_STROKES:
            return 1.0
        return float(cdf[threshold - LATTICE_MIN_STROKES])

    def p_ge(threshold: int) -> float:
        if threshold <= LATTICE_MIN_STROKES:
            return 1.0
        if threshold > LATTICE_MAX_STROKES:
            return 0.0
        return float(1.0 - cdf[threshold - LATTICE_MIN_STROKES - 1])

    return {
        "crps": crps_common_lattice(p, actual),
        "neg_log_score": float(-math.log(max(mass, np.finfo(float).tiny))),
        "pit_lower": lower,
        "pit_upper": upper,
        "pit_mid": lower + 0.5 * mass,
        "pit_randomized": randomised,
        "predictive_mean_strokes": predictive_mean,
        "predictive_variance": predictive_variance,
        "predictive_sd": math.sqrt(max(predictive_variance, 0.0)),
        "predictive_mse_about_production_anchor": anchor_mse,
        "mean_error_vs_production": predictive_mean - float(production_mean_strokes),
        "actual_sq_error_vs_production": (actual - float(production_mean_strokes)) ** 2,
        "q05": q05,
        "q10": q10,
        "q25": q25,
        "q50": q50,
        "q75": q75,
        "q90": q90,
        "q95": q95,
        "cover50": int(q25 <= actual <= q75),
        "cover80": int(q10 <= actual <= q90),
        "cover90": int(q05 <= actual <= q95),
        "width50": q75 - q25,
        "width80": q90 - q10,
        "width90": q95 - q05,
        "p_score_le_anchor_minus_3": p_le(low3),
        "p_score_ge_anchor_plus_3": p_ge(high3),
        "p_score_le_anchor_minus_5": p_le(low5),
        "p_score_ge_anchor_plus_5": p_ge(high5),
        "actual_score_le_anchor_minus_3": int(actual <= low3),
        "actual_score_ge_anchor_plus_3": int(actual >= high3),
        "actual_score_le_anchor_minus_5": int(actual <= low5),
        "actual_score_ge_anchor_plus_5": int(actual >= high5),
    }


def _require_predictive_ready(panel: ParticlePanel) -> None:
    if panel.particles < MIN_DECISION_PARTICLES:
        raise PredictiveScoringBlocked(
            f"{panel.particles:,} particles are schema/runtime-only; predictive "
            f"scoring requires at least {MIN_DECISION_PARTICLES:,} per player-round"
        )
    statuses = {audit.anchor_status for audit in panel.audits}
    if statuses != {REQUIRED_ANCHOR_STATUS}:
        raise PredictiveScoringBlocked(
            "Predictive scoring requires generator-time shared production mean "
            f"anchor status {REQUIRED_ANCHOR_STATUS!r}; found {sorted(statuses)}"
        )
    for audit in panel.audits:
        if not audit.strict_particle_axis:
            raise PredictiveScoringBlocked(
                "Predictive scoring requires an explicit particle_ids or "
                "particle_ids_by_arm array; an implicit axis is mechanics-only"
            )
        manifest = json.loads(audit.manifest_path.read_text(encoding="utf-8"))
        contract = manifest.get("mean_anchor_contract")
        if not isinstance(contract, Mapping):
            raise PredictiveScoringBlocked("mean_anchor_contract is absent")
        required = {
            "owner": "production_skill",
            "policy": REQUIRED_ANCHOR_POLICY,
            "applied_stage": "generator_before_persistence",
            "same_target_for_all_arms": True,
            "posthoc_arm_specific_adjustment": False,
            "evaluator_adjustment": False,
        }
        for key, expected in required.items():
            if contract.get(key) != expected:
                raise PredictiveScoringBlocked(
                    f"mean_anchor_contract.{key} must be {expected!r}"
                )
        prefix_exact = _nested(
            manifest,
            "raw_128_gate",
            "counter_50k_prefix_of_100k",
            "exact",
        )
        if prefix_exact is not True:
            raise PredictiveScoringBlocked("Exact 50k-prefix-of-100k RNG gate is absent")
        if tuple(map(str, manifest.get("transform_order", []))) != REQUIRED_TRANSFORM_ORDER:
            raise PredictiveScoringBlocked("Normalized donor transform order is not attested")
        if int(manifest.get("alignment_count", -1)) != 1:
            raise PredictiveScoringBlocked("Normalized donor endpoints must be aligned exactly once")
        if manifest.get("mass_contract") != "base_hierarchy_mass_times_mean_one_model_weight":
            raise PredictiveScoringBlocked("Normalized donor mass contract is not attested")
        frozen_inputs = manifest.get("input_hashes_sha256")
        if not isinstance(frozen_inputs, Mapping) or not frozen_inputs:
            raise PredictiveScoringBlocked("Frozen simulator input hashes are absent")
        if any(
            not isinstance(value, str) or len(value) != 64
            for value in frozen_inputs.values()
        ):
            raise PredictiveScoringBlocked("Frozen simulator input hashes are malformed")


def _anchor_sample_audit(panel: ParticlePanel, cases: pd.DataFrame) -> pd.DataFrame:
    metadata = cases.set_index(["case_id", "round_num"])
    rows: list[dict[str, object]] = []
    for case_index, case_id in enumerate(panel.case_ids):
        for round_index, round_num in enumerate(panel.round_numbers):
            row = metadata.loc[(case_id, round_num)]
            arm_values: dict[str, np.ndarray] = {}
            for arm_index, arm in enumerate(ARMS):
                values = panel.totals[arm_index, case_index, round_index].astype(float)
                arm_values[arm] = values
                mean = float(values.mean())
                sd = float(values.std(ddof=0))
                se = sd / math.sqrt(len(values))
                target = float(row["production_mean_strokes"])
                tolerance = max(0.05, 5.0 * se)
                rows.append(
                    {
                        "case_id": case_id,
                        "event_key": row["event_key"],
                        "source_player_id": row["source_player_id"],
                        "round_num": int(round_num),
                        "arm": arm,
                        "particles": len(values),
                        "sample_mean_strokes": mean,
                        "sample_sd_strokes": sd,
                        "sample_mean_se": se,
                        "production_mean_strokes": target,
                        "sample_mean_error": mean - target,
                        "sample_mean_tolerance": tolerance,
                        "sample_mean_consistent": abs(mean - target) <= tolerance,
                    }
                )
            difference = arm_values[NORMALIZED_ARM] - arm_values[RAW_ARM]
            paired_se = float(difference.std(ddof=0) / math.sqrt(len(difference)))
            paired_mean = float(difference.mean())
            paired_tolerance = max(0.025, 5.0 * paired_se)
            rows.append(
                {
                    "case_id": case_id,
                    "event_key": row["event_key"],
                    "source_player_id": row["source_player_id"],
                    "round_num": int(round_num),
                    "arm": "normalized_minus_raw_paired",
                    "particles": len(difference),
                    "sample_mean_strokes": paired_mean,
                    "sample_sd_strokes": float(difference.std(ddof=0)),
                    "sample_mean_se": paired_se,
                    "production_mean_strokes": 0.0,
                    "sample_mean_error": paired_mean,
                    "sample_mean_tolerance": paired_tolerance,
                    "sample_mean_consistent": abs(paired_mean) <= paired_tolerance,
                }
            )
    return pd.DataFrame(rows)


def _score_cases(panel: ParticlePanel, joined: pd.DataFrame) -> pd.DataFrame:
    metadata = joined.set_index(["case_id", "round_num"])
    rows: list[dict[str, object]] = []
    for case_index, case_id in enumerate(panel.case_ids):
        for round_index, round_num in enumerate(panel.round_numbers):
            row = metadata.loc[(case_id, round_num)]
            if pd.isna(row["actual_strokes"]):
                continue
            for arm_index, arm in enumerate(ARMS):
                samples = panel.totals[arm_index, case_index, round_index]
                probability = pmf_on_common_lattice(samples)
                metrics = score_round_pmf(
                    probability,
                    actual_strokes=int(row["actual_strokes"]),
                    production_mean_strokes=float(row["production_mean_strokes"]),
                    event_key=str(row["event_key"]),
                    player_id=str(row["source_player_id"]),
                    round_num=int(round_num),
                )
                rows.append(
                    {
                        "case_id": case_id,
                        "event_key": row["event_key"],
                        "event_name": row["event_name"],
                        "source_player_id": row["source_player_id"],
                        "player_name": row["player_name"],
                        "round_num": int(round_num),
                        "cut_rule": row["cut_rule"],
                        "evaluation_scope": row["evaluation_scope"],
                        "arm": arm,
                        "particles": panel.particles,
                        "actual_strokes": int(row["actual_strokes"]),
                        "production_mean_strokes": float(row["production_mean_strokes"]),
                        **metrics,
                    }
                )
    result = pd.DataFrame(rows)
    if result.empty:
        raise EvaluationContractError("No observed round cases are available to score")
    return result


def paired_case_deltas(case_metrics: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "case_id",
        "event_key",
        "event_name",
        "source_player_id",
        "player_name",
        "round_num",
        "cut_rule",
        "evaluation_scope",
    ]
    value_columns = [
        "crps",
        "neg_log_score",
        "predictive_mean_strokes",
        "predictive_variance",
        "predictive_sd",
        "mean_error_vs_production",
        "p_score_le_anchor_minus_3",
        "p_score_ge_anchor_plus_3",
        "p_score_le_anchor_minus_5",
        "p_score_ge_anchor_plus_5",
    ]
    wide = case_metrics.pivot(index=keys, columns="arm", values=value_columns)
    if set(wide.columns.get_level_values("arm")) != set(ARMS):
        raise EvaluationContractError("Scored cases are not aligned across arms")
    output = wide.index.to_frame(index=False)
    for column in value_columns:
        output[f"{column}_raw"] = wide[(column, RAW_ARM)].to_numpy()
        output[f"{column}_normalized"] = wide[(column, NORMALIZED_ARM)].to_numpy()
        output[f"delta_{column}_normalized_minus_raw"] = (
            wide[(column, NORMALIZED_ARM)].to_numpy()
            - wide[(column, RAW_ARM)].to_numpy()
        )
    return output


def _aggregate(case_metrics: pd.DataFrame, deltas: pd.DataFrame) -> pd.DataFrame:
    primary = case_metrics[case_metrics["evaluation_scope"].eq("primary_no_survivor")]
    rows: list[dict[str, object]] = []
    metrics = [
        "crps",
        "neg_log_score",
        "predictive_variance",
        "predictive_sd",
        "mean_error_vs_production",
        "cover50",
        "cover80",
        "cover90",
        "width50",
        "width80",
        "width90",
    ]
    for arm, group in primary.groupby("arm", observed=True):
        for metric in metrics:
            rows.append(
                {
                    "scope": "primary_no_survivor_case_weighted",
                    "arm": arm,
                    "metric": metric,
                    "value": float(group[metric].mean()),
                    "events": int(group["event_key"].nunique()),
                    "cases": int(len(group)),
                }
            )
        event_means = group.groupby("event_key", observed=True)[metrics].mean()
        for metric in metrics:
            rows.append(
                {
                    "scope": "primary_no_survivor_event_weighted",
                    "arm": arm,
                    "metric": metric,
                    "value": float(event_means[metric].mean()),
                    "events": int(len(event_means)),
                    "cases": int(len(group)),
                }
            )
    primary_delta = deltas[deltas["evaluation_scope"].eq("primary_no_survivor")]
    for metric in ("crps", "neg_log_score", "predictive_variance", "predictive_sd"):
        column = f"delta_{metric}_normalized_minus_raw"
        rows.append(
            {
                "scope": "primary_no_survivor_paired_case_weighted",
                "arm": "normalized_minus_raw",
                "metric": metric,
                "value": float(primary_delta[column].mean()),
                "events": int(primary_delta["event_key"].nunique()),
                "cases": int(len(primary_delta)),
            }
        )
        event_delta = primary_delta.groupby("event_key", observed=True)[column].mean()
        rows.append(
            {
                "scope": "primary_no_survivor_paired_event_weighted",
                "arm": "normalized_minus_raw",
                "metric": metric,
                "value": float(event_delta.mean()),
                "events": int(len(event_delta)),
                "cases": int(len(primary_delta)),
            }
        )
    return pd.DataFrame(rows)


def pit_histogram(case_metrics: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    rows: list[dict[str, object]] = []
    for (scope, arm), group in case_metrics.groupby(
        ["evaluation_scope", "arm"], observed=True
    ):
        counts, _ = np.histogram(group["pit_randomized"].to_numpy(float), bins=edges)
        for index, count in enumerate(counts):
            rows.append(
                {
                    "evaluation_scope": scope,
                    "arm": arm,
                    "bin": index,
                    "lower": edges[index],
                    "upper": edges[index + 1],
                    "count": int(count),
                    "fraction": float(count / len(group)),
                }
            )
    return pd.DataFrame(rows)


def h2h_probabilities_from_pmfs(
    probability_a: np.ndarray, probability_b: np.ndarray
) -> tuple[float, float, float]:
    """Exact independent-marginal P(A wins), P(tie), P(B wins)."""

    a = np.asarray(probability_a, dtype=float)
    b = np.asarray(probability_b, dtype=float)
    if a.shape != PHYSICAL_STROKE_LATTICE.shape or b.shape != a.shape:
        raise EvaluationContractError("H2H PMFs must share the frozen lattice")
    p_tie = float(a @ b)
    # Lower strokes win.  For each A score, B must be strictly higher.
    b_above = 1.0 - np.cumsum(b)
    p_a_win = float(a @ b_above)
    p_b_win = float(max(0.0, 1.0 - p_a_win - p_tie))
    total = p_a_win + p_tie + p_b_win
    return p_a_win / total, p_tie / total, p_b_win / total


def load_h2h_pairs(path: Path, cases: pd.DataFrame) -> pd.DataFrame:
    pairs = _load_table(path).copy()
    missing = sorted(PAIR_REQUIRED_COLUMNS - set(pairs.columns))
    if missing:
        raise EvaluationContractError(f"H2H pair table is missing {missing}")
    for column in ("pair_id", "event_key", "player_a_id", "player_b_id", "pair_source"):
        pairs[column] = pairs[column].astype(str).str.strip()
    pairs["round_num"] = pd.to_numeric(pairs["round_num"], errors="coerce")
    pairs["pair_frozen_at"] = pd.to_datetime(pairs["pair_frozen_at"], errors="coerce", utc=True)
    if pairs[["round_num", "pair_frozen_at"]].isna().any().any():
        raise EvaluationContractError("H2H pair table contains invalid values")
    if not np.allclose(pairs["round_num"], np.rint(pairs["round_num"]), atol=0.0, rtol=0.0):
        raise EvaluationContractError("H2H round_num must be integer")
    pairs["round_num"] = np.rint(pairs["round_num"]).astype(int)
    if pairs.duplicated(["pair_id", "round_num"]).any():
        raise EvaluationContractError("H2H pair IDs are duplicated")
    if pairs["player_a_id"].eq(pairs["player_b_id"]).any():
        raise EvaluationContractError("H2H pair cannot contain the same player twice")
    lookup = cases[
        [
            "event_key",
            "event_start",
            "source_player_id",
            "case_id",
            "round_num",
            "cut_rule",
            "actual_strokes",
        ]
    ]
    left = lookup.rename(
        columns={
            "source_player_id": "player_a_id",
            "case_id": "case_id_a",
            "actual_strokes": "actual_a",
        }
    )
    right = lookup.rename(
        columns={
            "source_player_id": "player_b_id",
            "case_id": "case_id_b",
            "actual_strokes": "actual_b",
            "event_start": "event_start_b",
            "cut_rule": "cut_rule_b",
        }
    )
    joined = pairs.merge(
        left,
        on=["event_key", "round_num", "player_a_id"],
        how="left",
        validate="many_to_one",
    ).merge(
        right,
        on=["event_key", "round_num", "player_b_id"],
        how="left",
        validate="many_to_one",
    )
    if joined[["case_id_a", "case_id_b", "event_start", "event_start_b"]].isna().any().any():
        raise EvaluationContractError("H2H pair references a player-round outside the frozen roster")
    if not (joined["pair_frozen_at"] < joined["event_start"]).all():
        raise EvaluationContractError("H2H roster must be frozen before event start")
    if not joined["event_start"].eq(joined["event_start_b"]).all():
        raise EvaluationContractError("H2H pair player metadata disagrees")
    if not joined["cut_rule"].eq(joined["cut_rule_b"]).all():
        raise EvaluationContractError("H2H pair cut rules disagree")
    joined["evaluation_scope"] = np.where(
        joined["cut_rule"].eq(CUT_RULE_NO_CUT) | joined["round_num"].le(2),
        "primary_no_survivor",
        "survivor_conditional_secondary",
    )
    primary = joined["evaluation_scope"].eq("primary_no_survivor")
    if joined.loc[primary, ["actual_a", "actual_b"]].isna().any().any():
        raise EvaluationContractError("Primary H2H truth is incomplete")
    return joined


def _score_h2h(
    panel: ParticlePanel,
    cases: pd.DataFrame,
    pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    case_position = {case_id: index for index, case_id in enumerate(panel.case_ids)}
    rows: list[dict[str, object]] = []
    for pair in pairs.itertuples(index=False):
        if pd.isna(pair.actual_a) or pd.isna(pair.actual_b):
            continue
        round_index = int(pair.round_num) - 1
        outcome_class = (
            "a_win" if pair.actual_a < pair.actual_b else
            "b_win" if pair.actual_a > pair.actual_b else "tie"
        )
        outcome_index = {"a_win": 0, "tie": 1, "b_win": 2}[outcome_class]
        for arm_index, arm in enumerate(ARMS):
            a_samples = panel.totals[
                arm_index, case_position[str(pair.case_id_a)], round_index
            ]
            b_samples = panel.totals[
                arm_index, case_position[str(pair.case_id_b)], round_index
            ]
            pa = pmf_on_common_lattice(a_samples)
            pb = pmf_on_common_lattice(b_samples)
            pwin, ptie, ploss = h2h_probabilities_from_pmfs(pa, pb)
            probabilities = np.asarray([pwin, ptie, ploss])
            one_hot = np.zeros(3)
            one_hot[outcome_index] = 1.0
            no_tie_denominator = pwin + ploss
            two_way = pwin / no_tie_denominator
            two_way_outcome = (
                1.0 if outcome_class == "a_win" else
                0.0 if outcome_class == "b_win" else np.nan
            )
            two_way_brier = (
                (two_way - two_way_outcome) ** 2
                if np.isfinite(two_way_outcome) else np.nan
            )
            two_way_log = (
                -(
                    two_way_outcome * math.log(max(two_way, np.finfo(float).tiny))
                    + (1.0 - two_way_outcome)
                    * math.log(max(1.0 - two_way, np.finfo(float).tiny))
                )
                if np.isfinite(two_way_outcome) else np.nan
            )
            rows.append(
                {
                    "pair_id": pair.pair_id,
                    "event_key": pair.event_key,
                    "round_num": int(pair.round_num),
                    "player_a_id": pair.player_a_id,
                    "player_b_id": pair.player_b_id,
                    "evaluation_scope": pair.evaluation_scope,
                    "arm": arm,
                    "outcome_class": outcome_class,
                    "p_a_win": pwin,
                    "p_tie": ptie,
                    "p_b_win": ploss,
                    "three_way_brier": float(np.sum((probabilities - one_hot) ** 2)),
                    "three_way_neg_log_score": float(
                        -math.log(max(probabilities[outcome_index], np.finfo(float).tiny))
                    ),
                    "two_way_p_a_win_no_tie": two_way,
                    "two_way_brier_nonpush": two_way_brier,
                    "two_way_neg_log_nonpush": two_way_log,
                    "push": outcome_class == "tie",
                    "h2h_dependence_contract": "exact_independent_marginal_convolution",
                }
            )
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        return metrics, pd.DataFrame()
    keys = [
        "pair_id",
        "event_key",
        "round_num",
        "player_a_id",
        "player_b_id",
        "evaluation_scope",
        "outcome_class",
    ]
    values = [
        "p_a_win",
        "p_tie",
        "three_way_brier",
        "three_way_neg_log_score",
        "two_way_p_a_win_no_tie",
        "two_way_brier_nonpush",
        "two_way_neg_log_nonpush",
    ]
    wide = metrics.pivot(index=keys, columns="arm", values=values)
    deltas = wide.index.to_frame(index=False)
    for column in values:
        deltas[f"{column}_raw"] = wide[(column, RAW_ARM)].to_numpy()
        deltas[f"{column}_normalized"] = wide[(column, NORMALIZED_ARM)].to_numpy()
        deltas[f"delta_{column}_normalized_minus_raw"] = (
            wide[(column, NORMALIZED_ARM)].to_numpy()
            - wide[(column, RAW_ARM)].to_numpy()
        )
    return metrics, deltas


def _input_hashes(paths: Iterable[Path]) -> dict[str, str]:
    return {str(Path(path).resolve()): sha256_file(Path(path)) for path in paths}


def evaluate_panel(
    panel: ParticlePanel,
    *,
    case_index_path: Path,
    truth_path: Path,
    h2h_pairs_path: Path | None = None,
) -> EvaluationResult:
    """Score an unchanged, generator-anchored >=50k particle panel."""

    _require_predictive_ready(panel)
    auxiliary_paths = [Path(case_index_path).resolve(), Path(truth_path).resolve()]
    if h2h_pairs_path is not None:
        auxiliary_paths.append(Path(h2h_pairs_path).resolve())
    auxiliary_hashes_start = _input_hashes(auxiliary_paths)
    evaluator_path = Path(__file__).resolve()
    auxiliary_hashes_start[str(evaluator_path)] = sha256_file(evaluator_path)
    cases = load_case_index(case_index_path, panel)
    joined = load_truth(truth_path, cases)
    anchor = _anchor_sample_audit(panel, cases)
    case_metrics = _score_cases(panel, joined)
    deltas = paired_case_deltas(case_metrics)
    aggregate = _aggregate(case_metrics, deltas)
    pits = pit_histogram(case_metrics)
    h2h_metrics = pd.DataFrame()
    h2h_deltas = pd.DataFrame()
    if h2h_pairs_path is not None:
        pairs = load_h2h_pairs(h2h_pairs_path, joined)
        h2h_metrics, h2h_deltas = _score_h2h(panel, joined, pairs)

    primary = case_metrics[
        case_metrics["evaluation_scope"].eq("primary_no_survivor")
    ][["case_id", "event_key", "round_num"]].drop_duplicates()
    particle_gate = panel.particles >= MIN_DECISION_PARTICLES
    anchor_consistency = bool(anchor["sample_mean_consistent"].all())
    broad_panel = (
        primary["event_key"].nunique() >= MIN_DECISION_EVENTS
        and len(primary) >= MIN_DECISION_PRIMARY_CASES
    )
    status = {
        "contract_version": CONTRACT_VERSION,
        "predictive_scoring_permitted": True,
        "samples_scored_without_location_or_variance_adjustment": True,
        "particle_precision_gate": particle_gate,
        "particles_per_player_round": panel.particles,
        "generator_anchor_attested": True,
        "sample_anchor_consistency_gate": anchor_consistency,
        "primary_panel_no_survivor_selection": True,
        "tail_policy": tail_policy.manifest_policy(),
        "primary_events": int(primary["event_key"].nunique()),
        "primary_cases": int(len(primary)),
        "broad_heldout_panel_gate": bool(broad_panel),
        "minimum_events_for_decision_claim": MIN_DECISION_EVENTS,
        "minimum_primary_cases_for_decision_claim": MIN_DECISION_PRIMARY_CASES,
        "decision_claim_permitted": bool(
            particle_gate and anchor_consistency and broad_panel
        ),
        "decision_interpretation": (
            "decision-grade heldout comparison"
            if particle_gate and anchor_consistency and broad_panel
            else "anchored high-particle pilot; insufficient for a production decision"
        ),
        "lattice": {
            "coordinate": "physical_integer_round_strokes",
            "minimum": LATTICE_MIN_STROKES,
            "maximum": LATTICE_MAX_STROKES,
            "spacing": 1,
            "support_selection": "fixed_before_predictions_and_truth",
            "jeffreys_alpha_per_cell": JEFFREYS_ALPHA,
        },
        "h2h_role": "secondary",
        "h2h_dependence_contract": (
            "exact_independent_marginal_convolution"
            if h2h_pairs_path is not None else None
        ),
    }
    combined_hashes_start = {
        **dict(panel.input_hashes_start),
        **auxiliary_hashes_start,
    }
    hashes_end = {
        path: sha256_file(Path(path)) for path in combined_hashes_start
    }
    if hashes_end != combined_hashes_start:
        raise EvaluationContractError("An input artifact changed during evaluation")
    return EvaluationResult(
        case_metrics=case_metrics,
        paired_case_deltas=deltas,
        aggregate_metrics=aggregate,
        anchor_audit=anchor,
        pit_histogram=pits,
        h2h_metrics=h2h_metrics,
        paired_h2h_deltas=h2h_deltas,
        decision_status=status,
        input_hashes=combined_hashes_start,
    )


def contract_schema() -> dict[str, object]:
    """Machine-readable frozen contract for producer and reviewer handoff."""

    return {
        "contract_version": CONTRACT_VERSION,
        "particle_bundle": {
            "files": ["manifest.json", "round_totals_int16.npz"],
            "npz_required_arrays": [
                "arm_names",
                "round_totals",
                "round_numbers",
                "event_key",
                "player_id",
            ],
            "shape": "[2 arms, 4 rounds, N aligned particles] per player bundle",
            "arms": list(ARMS),
            "dtype": "int16 physical strokes",
            "preferred_particle_axis": "particle_ids_by_arm identical and 0..N-1",
            "required_manifest_tail_policy": tail_policy.manifest_policy(),
        },
        "case_index": {
            "grain": "one pre-event row per event/player/round; always R1-R4",
            "required_columns": sorted(CASE_REQUIRED_COLUMNS),
        },
        "truth": {
            "grain": "one independently sourced row per observed event/player/round",
            "required_columns": sorted(TRUTH_REQUIRED_COLUMNS),
        },
        "h2h_pairs_optional": {
            "grain": "one pre-event frozen matchup per event/round",
            "required_columns": sorted(PAIR_REQUIRED_COLUMNS),
        },
        "hard_rules": [
            "beta equals zero",
            "normalization mode is player_state_residualized",
            "raw and normalized use the exact same counter particle axis",
            "50k is the exact prefix of 100k",
            "all R1-R4 simulation rows from the frozen pre-event roster are retained",
            "generator owns one shared production skill mean; evaluator never recentres",
            "unanchored mechanics output is never scored predictively",
            "fixed physical integer-stroke lattice 40..140 for every arm and case",
            (
                "18 continuation transitions; no physical hole-score cap, "
                "score clipping, or terminal-by-cap"
            ),
            (
                "still-active after transition 17 fails the whole run as "
                "transition_loop_unresolved; survivor filtering is forbidden"
            ),
            "cut-event primary metrics use R1-R2; no-cut primary metrics use R1-R4",
            "H2H roster is frozen independently and is always secondary",
            "fewer than 50k particles can never produce a predictive verdict",
        ],
    }


def _write_result(output_dir: Path, result: EvaluationResult) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Evaluation output must be a new empty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    result.case_metrics.to_parquet(output_dir / "round_case_metrics.parquet", index=False)
    result.paired_case_deltas.to_csv(output_dir / "paired_round_deltas.csv", index=False)
    result.aggregate_metrics.to_csv(output_dir / "aggregate_metrics.csv", index=False)
    result.anchor_audit.to_csv(output_dir / "mean_anchor_audit.csv", index=False)
    result.pit_histogram.to_csv(output_dir / "pit_histogram.csv", index=False)
    if not result.h2h_metrics.empty:
        result.h2h_metrics.to_parquet(output_dir / "h2h_metrics.parquet", index=False)
        result.paired_h2h_deltas.to_csv(output_dir / "paired_h2h_deltas.csv", index=False)
    manifest = {
        "artifact": CONTRACT_VERSION,
        "research_only": True,
        "production_touched": False,
        "decision_status": result.decision_status,
        "input_hashes_sha256": result.input_hashes,
        "contract": contract_schema(),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(_jsonable(manifest), indent=2), encoding="utf-8"
    )
    report = [
        "# Normalized chain evaluation",
        "",
        f"Decision interpretation: **{result.decision_status['decision_interpretation']}**.",
        "",
        "The persisted raw and normalized particles were scored unchanged. No mean, variance, latent, or lattice adapter was applied by this evaluator.",
        "",
        f"Particles per player-round: {result.decision_status['particles_per_player_round']:,}.",
        f"Primary no-survivor cases/events: {result.decision_status['primary_cases']} / {result.decision_status['primary_events']}.",
        f"Shared production mean sample-consistency gate: {result.decision_status['sample_anchor_consistency_gate']}.",
        f"Decision claim permitted: {result.decision_status['decision_claim_permitted']}.",
        "",
        "H2H, when present, is secondary and uses exact independent marginal convolution on the same frozen pair roster.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", action="append", type=Path, required=True)
    parser.add_argument("--schema-only", action="store_true")
    parser.add_argument("--case-index", type=Path)
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--h2h-pairs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    panel = load_particle_panel(args.bundle)
    if args.schema_only:
        output = Path(args.output)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Schema output must be a new empty directory: {output}")
        output.mkdir(parents=True, exist_ok=True)
        audit = audit_schema_only(panel)
        (output / "schema_audit.json").write_text(
            json.dumps(_jsonable(audit), indent=2), encoding="utf-8"
        )
        (output / "contract_schema.json").write_text(
            json.dumps(contract_schema(), indent=2), encoding="utf-8"
        )
        return 0
    if args.case_index is None or args.truth is None:
        raise EvaluationContractError(
            "Predictive evaluation requires --case-index and --truth"
        )
    result = evaluate_panel(
        panel,
        case_index_path=args.case_index,
        truth_path=args.truth,
        h2h_pairs_path=args.h2h_pairs,
    )
    _write_result(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
