"""Research-chain contracts for normalized TEE/APP donor atoms.

This module is intentionally a hook, not a second simulator.  It imports the
shared field-normalization and as-of profile contracts and supplies the small
pieces the full-hole chain will need when normalization is integrated:

* canonical state/setup keys;
* a fail-closed, versioned quality-coefficient registry (beta is zero unless
  an explicit research registry passes every provenance check);
* stable cache identities for precomputed neutral/target atom tables; and
* a vectorized counter RNG whose particle prefix is invariant to chunking and
  whose key deliberately excludes the research arm.

No production module imports this file.  It does not read current-week inputs,
modify the existing chain, or run a large simulation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from hashlib import blake2b, sha256
import json
import math
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd

import hole_sim_asof_spatial_profiles as asof
import hole_sim_field_normalization as normalization


HOOK_VERSION = "normalized_chain_hook_v1"
RNG_VERSION = "blake2b_splitmix64_counter_v1"
QUALITY_REGISTRY_CONTRACT = "normalized_chain_quality_coefficients_v1"
ALLOWED_CALIBRATION_CONTRACTS = frozenset(
    {"strict_prior_transition_quality_v1"}
)
DISABLED_COEFFICIENT_VERSION = "quality_beta_zero_default_v1"
SETUP_RE = re.compile(
    r"^(?P<event>.+)\|course=(?P<course>.+)\|r=(?P<round>[1-4])\|h=(?P<hole>(?:[1-9]|1[0-8]))$"
)
APP_LIES = ("tee", "fairway", "rough", "sand", "recovery", "other")
APP_BANDS = tuple(asof.APP_BANDS)
QUALITY_REQUIRED_COLUMNS = (
    "coefficient_version",
    "calibration_contract",
    "sg_category",
    "state_key",
    "beta_natural_per_sg_round",
    "training_shots",
    "training_cutoff",
    "source_sha256",
    "enabled",
    "shrinkage_status",
)
NEUTRAL_ATOM_REQUIRED_COLUMNS = (
    "atom_id",
    "shot_key",
    "donor_event_key",
    "donor_event_start",
    "donor_setup_key",
    "sg_category",
    "state_key",
    "setup_key",
    "neutral_resid_long_yd",
    "neutral_resid_lat_yd",
    "neutral_weight",
    "profile_cutoff",
    "normalization_mode",
    "normalization_provenance_sha256",
    "alignment_count",
    "coordinate_frame_version",
)
TARGET_ATOM_REQUIRED_COLUMNS = (
    *NEUTRAL_ATOM_REQUIRED_COLUMNS,
    "target_event_key",
    "target_event_start",
    "target_player_id",
    "target_profile_cutoff",
    "model_long_yd",
    "model_lat_yd",
    "model_end_lie",
    "model_end_hazard",
    "model_weight",
    "target_reapplication_provenance_sha256",
)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_app_lie(value: object) -> str:
    """Map a generated APP start lie to the shared profile taxonomy."""

    text = str(value).strip().casefold()
    if "bunker" in text:
        return "sand"
    return text if text in APP_LIES else "other"


def app_distance_band(distance_yd: float) -> str:
    """Return the audited four-band APP profile state."""

    distance = float(distance_yd)
    if not math.isfinite(distance) or distance < 0.0:
        raise ValueError("APP distance must be finite and non-negative")
    if distance <= 100.0:
        return "<=100"
    if distance <= 150.0:
        return "100-150"
    if distance <= 215.0:
        return "150-215"
    return ">215"


def tee_state_key(par: int) -> str:
    """Return the shared TEE state key; par threes start in APP."""

    value = int(par)
    if value not in {4, 5}:
        raise ValueError("TEE normalization is defined only for par 4/5")
    return f"tee|par={value}"


def app_state_key(start_lie: object, distance_yd: float) -> str:
    """Return the shared APP lie x four-band state key."""

    return (
        f"app|lie={canonical_app_lie(start_lie)}|"
        f"band={app_distance_band(distance_yd)}"
    )


def setup_key(
    event_key: object,
    course_id: object,
    round_no: int,
    hole_no: int,
) -> str:
    """Return the architecture-owned setup identifier exactly."""

    round_value = int(round_no)
    hole_value = int(hole_no)
    if round_value not in {1, 2, 3, 4}:
        raise ValueError("round_no must be 1..4")
    if not 1 <= hole_value <= 18:
        raise ValueError("hole_no must be 1..18")
    event = str(event_key).strip()
    course = str(course_id).strip()
    if not event or not course or "|" in course:
        raise ValueError("event_key/course_id must be non-empty canonical values")
    return f"{event}|course={course}|r={round_value}|h={hole_value}"


def validate_setup_key(value: object) -> None:
    if SETUP_RE.fullmatch(str(value)) is None:
        raise ValueError(f"Invalid normalized-chain setup_key: {value!r}")


@dataclass(frozen=True)
class NeutralAtomCacheKey:
    """Content-addressed target-support cache identity.

    The key includes every input that can change atom coordinates, labels or
    probabilities.  Particle count and chunk size are intentionally absent:
    they are consumers of the same cached distribution.
    """

    target_event_key: str
    target_player_id: str
    sg_category: str
    state_key: str
    setup_key: str
    normalization_mode: str
    donor_atom_sha256: str
    donor_profile_sha256: str
    target_profile_sha256: str
    normalization_config_sha256: str
    coefficient_registry_sha256: str
    coordinate_frame_version: str
    geometry_classifier_version: str
    hook_version: str = HOOK_VERSION

    def digest(self) -> str:
        validate_setup_key(self.setup_key)
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return sha256(payload).hexdigest()


@dataclass(frozen=True)
class QualityCoefficientRegistry:
    """Validated category/state natural parameters for research only."""

    rows: pd.DataFrame
    registry_sha256: str
    coefficient_version: str
    calibration_contract: str
    enabled: bool
    status: str

    def coefficient(self, category: str, state_key_value: str) -> float:
        """Resolve exact state, then explicit category wildcard, then zero."""

        if not self.enabled:
            return 0.0
        category_rows = self.rows[
            self.rows["sg_category"].eq(str(category).casefold())
            & self.rows["enabled"]
        ]
        exact = category_rows[category_rows["state_key"].eq(state_key_value)]
        selected = exact
        if selected.empty:
            selected = category_rows[category_rows["state_key"].eq("*")]
        if selected.empty:
            return 0.0
        if len(selected) != 1:
            raise ValueError("Quality coefficient resolution is not unique")
        return float(selected["beta_natural_per_sg_round"].iloc[0])

    def config_for(
        self,
        category: str,
        state_key_value: str,
        base: normalization.NormalizationConfig | None = None,
    ) -> normalization.NormalizationConfig:
        """Create a core config for one precomputed condition partition."""

        base = base or normalization.NormalizationConfig()
        return replace(
            base,
            ability_log_weight_per_sg2=self.coefficient(
                category, state_key_value
            ),
        )


def disabled_quality_registry() -> QualityCoefficientRegistry:
    """The required default: no implicit quality weighting of any kind."""

    rows = pd.DataFrame(
        {
            "coefficient_version": [DISABLED_COEFFICIENT_VERSION] * 2,
            "calibration_contract": ["disabled"] * 2,
            "sg_category": ["ott", "app"],
            "state_key": ["*", "*"],
            "beta_natural_per_sg_round": [0.0, 0.0],
            "training_shots": [0, 0],
            "training_cutoff": [pd.NaT, pd.NaT],
            "source_sha256": ["disabled", "disabled"],
            "enabled": [False, False],
            "shrinkage_status": ["disabled_beta_zero", "disabled_beta_zero"],
        }
    )
    return QualityCoefficientRegistry(
        rows=rows,
        registry_sha256=sha256(DISABLED_COEFFICIENT_VERSION.encode()).hexdigest(),
        coefficient_version=DISABLED_COEFFICIENT_VERSION,
        calibration_contract="disabled",
        enabled=False,
        status="disabled_beta_zero_default",
    )


def load_quality_registry(
    table_path: Path | None,
    manifest_path: Path | None,
    *,
    target_event_start: object,
    enable: bool = False,
    minimum_training_shots: int = 750,
) -> QualityCoefficientRegistry:
    """Load an explicit versioned beta registry or return the zero default.

    The exploratory calibration CSV is deliberately not accepted directly.
    A promoted registry must freeze its own coefficient version, source hash,
    training cutoff and shrinkage decision.  This prevents the core module's
    former placeholder value ``1.0`` (or any other ad-hoc scalar) from becoming
    a simulation coefficient.
    """

    if not enable:
        return disabled_quality_registry()
    if table_path is None or manifest_path is None:
        raise ValueError("Quality weighting requires both table and manifest")
    table_path = Path(table_path)
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("contract") != QUALITY_REGISTRY_CONTRACT:
        raise ValueError("Unsupported quality coefficient registry contract")
    calibration_contract = str(manifest.get("calibration_contract", ""))
    if calibration_contract not in ALLOWED_CALIBRATION_CONTRACTS:
        raise ValueError("Quality registry references an unapproved calibration")
    observed_hash = _sha256_file(table_path)
    if observed_hash != manifest.get("coefficient_table_sha256"):
        raise ValueError("Quality coefficient table hash does not match manifest")
    if not bool(manifest.get("enabled_for_research", False)):
        raise ValueError("Quality coefficient registry is not research-enabled")

    rows = pd.read_csv(table_path, dtype={"state_key": str})
    missing = sorted(set(QUALITY_REQUIRED_COLUMNS) - set(rows.columns))
    if missing:
        raise ValueError(f"Quality coefficient table misses columns: {missing}")
    if rows.empty:
        raise ValueError("Quality coefficient table is empty")
    versions = rows["coefficient_version"].astype(str).unique()
    contracts = rows["calibration_contract"].astype(str).unique()
    if len(versions) != 1 or versions[0] != manifest.get("coefficient_version"):
        raise ValueError("Coefficient version is not unique/manifest-matched")
    if len(contracts) != 1 or contracts[0] != calibration_contract:
        raise ValueError("Calibration contract is not unique/manifest-matched")
    rows["sg_category"] = rows["sg_category"].astype(str).str.casefold()
    if not set(rows["sg_category"]).issubset({"ott", "app"}):
        raise ValueError("Normalized spatial hook accepts only OTT/APP betas")
    enabled_text = rows["enabled"].astype(str).str.strip().str.casefold()
    if not enabled_text.isin({"true", "false", "1", "0"}).all():
        raise ValueError("Quality enabled must contain only true/false or 1/0")
    rows["enabled"] = enabled_text.isin({"true", "1"})
    rows["training_shots"] = pd.to_numeric(
        rows["training_shots"], errors="raise"
    ).astype(int)
    rows["beta_natural_per_sg_round"] = pd.to_numeric(
        rows["beta_natural_per_sg_round"], errors="raise"
    )
    if not np.isfinite(rows["beta_natural_per_sg_round"]).all():
        raise ValueError("Quality beta contains a non-finite value")
    cutoffs = pd.to_datetime(rows["training_cutoff"], errors="coerce", utc=True)
    target = pd.Timestamp(target_event_start)
    target = target.tz_localize("UTC") if target.tzinfo is None else target.tz_convert("UTC")
    enabled_rows = rows["enabled"]
    if cutoffs[enabled_rows].isna().any() or not (
        cutoffs[enabled_rows] < target
    ).all():
        raise ValueError("Enabled quality coefficients are not strictly prior")
    if (
        rows.loc[enabled_rows, "training_shots"] < int(minimum_training_shots)
    ).any():
        raise ValueError("Enabled quality coefficient lacks minimum support")
    if rows.loc[enabled_rows, "shrinkage_status"].astype(str).isin(
        {"raw_unshrunk", "unreviewed", ""}
    ).any():
        raise ValueError("Enabled quality coefficient lacks an accepted shrinkage status")
    if rows.duplicated(["sg_category", "state_key"]).any():
        raise ValueError("Quality coefficient category/state keys are duplicated")
    for record in rows.itertuples(index=False):
        state = str(record.state_key)
        if state == "*":
            continue
        if record.sg_category == "ott" and state not in {
            "tee|par=4",
            "tee|par=5",
        }:
            raise ValueError(f"Invalid OTT quality state_key: {state}")
        if record.sg_category == "app":
            valid_app_states = {
                f"app|lie={lie}|band={band}"
                for lie in APP_LIES
                for band in APP_BANDS
            }
            if state not in valid_app_states:
                raise ValueError(f"Invalid APP quality state_key: {state}")
    if not rows["source_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("Quality coefficient source_sha256 is invalid")
    return QualityCoefficientRegistry(
        rows=rows.assign(training_cutoff=cutoffs),
        registry_sha256=observed_hash,
        coefficient_version=str(versions[0]),
        calibration_contract=calibration_contract,
        enabled=True,
        status="enabled_explicit_versioned_research_registry",
    )


def _splitmix64(values: np.ndarray) -> np.ndarray:
    """Vectorized SplitMix64 finalizer with defined uint64 wraparound."""

    with np.errstate(over="ignore"):
        values = values + np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(
            0xBF58476D1CE4E5B9
        )
        values = (values ^ (values >> np.uint64(27))) * np.uint64(
            0x94D049BB133111EB
        )
    return values ^ (values >> np.uint64(31))


def counter_uniforms(
    *,
    seed: int,
    event_key: object,
    player_id: object,
    round_no: int,
    hole_no: int,
    transition_index: int,
    stream: str,
    particle_ids: Iterable[int] | np.ndarray,
    draw_index: int = 0,
) -> np.ndarray:
    """Generate arm- and chunk-independent uniforms for particle IDs.

    ``arm``, normalization mode, worker and chunk are intentionally not
    accepted arguments.  Adding a new branch therefore cannot shift later
    streams, and particles ``0..49_999`` in a 100k run are exactly the 50k
    run, provided all other semantic keys are equal.
    """

    ids_signed = np.asarray(particle_ids, dtype=np.int64)
    if ids_signed.ndim != 1 or (ids_signed < 0).any():
        raise ValueError("particle_ids must be a one-dimensional non-negative array")
    payload = json.dumps(
        {
            "seed": int(seed),
            "event_key": str(event_key),
            "player_id": str(player_id),
            "round_no": int(round_no),
            "hole_no": int(hole_no),
            "transition_index": int(transition_index),
            "stream": str(stream),
            "draw_index": int(draw_index),
            "rng_version": RNG_VERSION,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    prefix = np.uint64(int.from_bytes(blake2b(payload, digest_size=8).digest(), "little"))
    with np.errstate(over="ignore"):
        counters = ids_signed.astype(np.uint64) * np.uint64(0xD2B74407B1CE6E93)
        bits = _splitmix64(counters ^ prefix)
    # Use the high 53 bits and midpoint convention, so values are strictly in
    # (0, 1) and deterministic under every chunking strategy.
    return ((bits >> np.uint64(11)).astype(np.float64) + 0.5) * (1.0 / 2**53)


def validate_atom_table(
    frame: pd.DataFrame,
    *,
    target_reapplied: bool,
) -> dict[str, object]:
    """Fail closed on the precomputed atom-cache handoff."""

    required = (
        TARGET_ATOM_REQUIRED_COLUMNS
        if target_reapplied
        else NEUTRAL_ATOM_REQUIRED_COLUMNS
    )
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Atom cache misses columns: {missing}")
    if frame.empty:
        raise ValueError("Atom cache is empty")
    if frame["atom_id"].isna().any() or frame["shot_key"].isna().any():
        raise ValueError("Atom identifiers must be complete")
    if not frame["alignment_count"].eq(1).all():
        raise ValueError("Every atom must be aligned exactly once")
    for value in frame["setup_key"].drop_duplicates():
        validate_setup_key(value)
    profile_cutoff = pd.to_datetime(frame["profile_cutoff"], errors="coerce", utc=True)
    if profile_cutoff.isna().any():
        raise ValueError("Atom profile cutoffs are incomplete")
    donor_start = pd.to_datetime(
        frame["donor_event_start"], errors="coerce", utc=True
    )
    if donor_start.isna().any() or not (profile_cutoff < donor_start).all():
        raise ValueError("Atom profiles are not strictly prior to donor events")
    if target_reapplied:
        target_start = pd.to_datetime(
            frame["target_event_start"], errors="coerce", utc=True
        )
        target_cutoff = pd.to_datetime(
            frame["target_profile_cutoff"], errors="coerce", utc=True
        )
        if (
            target_start.isna().any()
            or target_cutoff.isna().any()
            or not (target_cutoff < target_start).all()
        ):
            raise ValueError("Target profiles are not strictly prior to target event")
    weights = pd.to_numeric(
        frame["model_weight" if target_reapplied else "neutral_weight"],
        errors="coerce",
    )
    if weights.isna().any() or not np.isfinite(weights).all() or (weights < 0.0).any():
        raise ValueError("Atom weights must be finite and non-negative")
    # Stable sort order is part of the CRN contract: identical U values can
    # select identical atom IDs only if support ordering does not drift.
    duplicate_keys = frame.duplicated(
        ["sg_category", "state_key", "setup_key", "atom_id"]
    )
    if duplicate_keys.any():
        raise ValueError("Atom IDs are duplicated inside a condition support")
    return {
        "rows": int(len(frame)),
        "atoms": int(frame["atom_id"].nunique()),
        "setups": int(frame["setup_key"].nunique()),
        "states": int(frame["state_key"].nunique()),
        "target_reapplied": bool(target_reapplied),
        "alignment_count_min": int(frame["alignment_count"].min()),
        "alignment_count_max": int(frame["alignment_count"].max()),
    }


def _self_test() -> dict[str, object]:
    assert tee_state_key(4) == "tee|par=4"
    assert app_state_key("bunker", 150.0) == "app|lie=sand|band=100-150"
    key = setup_key("pga:R2026006", "7", 4, 18)
    validate_setup_key(key)
    particles = np.arange(100_000, dtype=np.int64)
    full = counter_uniforms(
        seed=17,
        event_key="pga:R2026006",
        player_id="123",
        round_no=2,
        hole_no=7,
        transition_index=3,
        stream="app_index",
        particle_ids=particles,
    )
    first = counter_uniforms(
        seed=17,
        event_key="pga:R2026006",
        player_id="123",
        round_no=2,
        hole_no=7,
        transition_index=3,
        stream="app_index",
        particle_ids=particles[:50_000],
    )
    chunks = np.concatenate(
        [
            counter_uniforms(
                seed=17,
                event_key="pga:R2026006",
                player_id="123",
                round_no=2,
                hole_no=7,
                transition_index=3,
                stream="app_index",
                particle_ids=chunk,
            )
            for chunk in np.array_split(particles, 17)
        ]
    )
    assert np.array_equal(full[:50_000], first)
    assert np.array_equal(full, chunks)
    disabled = disabled_quality_registry()
    assert disabled.coefficient("ott", "tee|par=4") == 0.0
    assert disabled.coefficient("app", "app|lie=fairway|band=150-215") == 0.0
    return {
        "hook_version": HOOK_VERSION,
        "rng_version": RNG_VERSION,
        "rng_particles_tested": int(len(particles)),
        "prefix_50k_matches_100k": True,
        "seventeen_chunk_reassembly_matches": True,
        "default_quality_beta": 0.0,
        "core_normalizer": str(Path(normalization.__file__).resolve()),
        "asof_profile_builder": str(Path(asof.__file__).resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--print-contract", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        print(json.dumps(_self_test(), indent=2))
    if args.print_contract:
        print(
            json.dumps(
                {
                    "hook_version": HOOK_VERSION,
                    "rng_version": RNG_VERSION,
                    "profile_version": asof.PROFILE_VERSION,
                    "condition_columns": normalization.NormalizationConfig().condition_cols,
                    "app_lies": APP_LIES,
                    "app_bands": APP_BANDS,
                    "setup_format": "{event_key}|course={course_id}|r={round_no}|h={hole_no}",
                    "quality_default": "beta_zero",
                    "quality_registry_contract": QUALITY_REGISTRY_CONTRACT,
                },
                indent=2,
            )
        )
    if not args.self_test and not args.print_contract:
        print("Use --self-test and/or --print-contract")


if __name__ == "__main__":
    main()


__all__ = [
    "APP_BANDS",
    "APP_LIES",
    "HOOK_VERSION",
    "NeutralAtomCacheKey",
    "QualityCoefficientRegistry",
    "RNG_VERSION",
    "app_distance_band",
    "app_state_key",
    "canonical_app_lie",
    "counter_uniforms",
    "disabled_quality_registry",
    "load_quality_registry",
    "setup_key",
    "tee_state_key",
    "validate_atom_table",
    "validate_setup_key",
]
