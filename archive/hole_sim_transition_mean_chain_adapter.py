"""Research-only bridge from complete physical atoms to the mean anchor.

This module defines the smallest support contract that can honestly accumulate
OTT/APP/ARG/PUTT strokes-gained inside a recursive physical hole chain.  It is
deliberately separate from the production and streaming simulators.

The central accounting identity for one complete atom is::

    reward = reference_value(start) - stroke_cost - reference_value(end)

``stroke_cost`` includes exactly one causal shot and every immediately linked
penalty stroke.  Because a path's end state must exactly equal the following
atom's start state, category rewards telescope to starting reference value
minus physical strokes at absorption.  APP hole-outs are ordinary terminal
atoms; an independent hole-out probability is forbidden.

The current research caches do not satisfy this schema.  The readiness helpers
make that gap executable and fail closed rather than treating missing reward or
donor ability as zero.  Persisted v3 atom contracts are authoritative in
Parquet and must be read through :func:`load_complete_atom_contract`; generic
CSV parsing is intentionally rejected because it can alter hash-bound floats.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence
import unicodedata
import warnings

import numpy as np
import pandas as pd


ARCHIVE = Path(__file__).resolve().parent
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

import hole_sim_transition_mean_anchor as anchor  # noqa: E402


ATOM_SCHEMA_CONTRACT = "complete_physical_category_sg_atom_v3"
PHYSICAL_ATOM_CONTRACT = "canonical_physical_atom_content_v3"
EXPECTATION_CONTRACT = "exact_absorbing_complete_atom_expectation_v3"
PRODUCTION_TARGET_CONTRACT = "frozen_production_category_mean_input_v2"
REWARD_REFERENCE_CONTRACT = "canonical_full_reference_value_table_v1"
APP_HOLEOUT_CONTRACT = "explicit_complete_atom_terminal_holed_iff_expected_mass_v3"
APP_HOLEOUT_EQUIVALENCE_CONTRACT = "per_support_app_holeout_equivalence_v2"
DONOR_SOURCE_CONTRACT = "canonical_strict_prior_donor_source_v1"
DONOR_SNAPSHOT_CONTRACT = "strict_prior_asof_content_verified_v2"
CATEGORIES = anchor.CATEGORIES
CATEGORY_INDEX = anchor.CATEGORY_INDEX
TERMINAL_STATE_KEY = "terminal"
ATOM_REWARD_TOLERANCE = 5e-6
APP_HOLEOUT_MASS_TOLERANCE = 1e-12


REQUIRED_ATOM_COLUMNS = (
    "support_key",
    "atom_id",
    "physical_atom_sha256",
    "category",
    "start_state_key",
    "end_state_key",
    "next_support_key",
    "terminal",
    "holed",
    "atom_kind",
    "complete_atom",
    "causal_strokes",
    "penalty_strokes",
    "stroke_cost",
    "hazard_resolution_status",
    "start_reference_value",
    "end_reference_value",
    "fixed_reference_reward",
    "quality_reward_available",
    "reward_reference_descriptor",
    "reward_reference_version",
    "reward_reference_sha256",
    "donor_ability_sg",
    "donor_ability_available",
    "donor_quality_group_id",
    "donor_event_key",
    "donor_player_id",
    "donor_skill_date",
    "donor_event_start",
    "donor_ability_snapshot_kind",
    "donor_source_descriptor",
    "donor_source_version",
    "donor_source_sha256",
    "donor_snapshot_sha256",
    "target_event_key",
    "target_event_start",
    "base_hierarchy_mass",
    "spatial_model_weight",
    "quality_base_mass",
    "holeout_representation",
    "declared_app_base_holeout_probability",
    "app_holeout_equivalence_sha256",
    "implicit_holeout_probability",
    "quality_input_sha256",
)

QUALITY_HASH_COLUMNS = (
    "atom_id",
    "physical_atom_sha256",
    "category",
    "fixed_reference_reward",
    "quality_reward_available",
    "reward_reference_sha256",
    "donor_ability_sg",
    "donor_ability_available",
    "donor_quality_group_id",
    "donor_event_key",
    "donor_player_id",
    "donor_skill_date",
    "donor_event_start",
    "donor_ability_snapshot_kind",
    "donor_source_descriptor",
    "donor_source_version",
    "donor_source_sha256",
    "donor_snapshot_sha256",
    "target_event_key",
    "target_event_start",
    "base_hierarchy_mass",
    "spatial_model_weight",
    "quality_base_mass",
    "declared_app_base_holeout_probability",
    "app_holeout_equivalence_sha256",
)

PHYSICAL_ATOM_HASH_COLUMNS = (
    "support_key",
    "atom_id",
    "category",
    "start_state_key",
    "end_state_key",
    "next_support_key",
    "terminal",
    "holed",
    "atom_kind",
    "complete_atom",
    "causal_strokes",
    "penalty_strokes",
    "stroke_cost",
    "hazard_resolution_status",
    "donor_event_key",
    "donor_player_id",
)

SUPPORT_HASH_COLUMNS = tuple(
    column for column in REQUIRED_ATOM_COLUMNS if column != "quality_input_sha256"
) + ("quality_input_sha256",)

PRODUCTION_TARGET_REQUIRED_COLUMNS = (
    "event_key",
    "player_id",
    "round_no",
    "target_ott_sg",
    "target_app_sg",
    "target_arg_sg",
    "target_putt_sg",
    "target_total_sg",
    "production_input_sha256",
)
PRODUCTION_TARGET_CONTENT_COLUMNS = tuple(
    column
    for column in PRODUCTION_TARGET_REQUIRED_COLUMNS
    if column != "production_input_sha256"
)

CURRENT_SUPPORT_FIELDS = {
    "precompute.SupportArrays": {
        "key",
        "atom_ids",
        "points",
        "weights",
        "cdf",
        "source",
        "n_setup",
        "n_hole",
        "n_broad",
        "holeout_probability",
        "cache_sha256",
        "metadata",
    },
    "streaming.NumericSupport": {
        "key",
        "atom_ids",
        "points",
        "probability",
        "cdf",
        "holeout_probability",
        "cache_sha256",
        "payload_sha256",
    },
    "markov.TransitionPool": {
        "start_distance",
        "next_distance",
        "next_lie",
        "sg",
        "cost",
        "holed",
        "sg_weight_valid",
    },
}


class QualitySupportUnavailable(ValueError):
    """Raised when a physical support cannot be quality tilted coherently."""


class UnsafeContractSerializationWarning(UserWarning):
    """Warn that CSV is permitted only through the exact round-trip loader."""


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _stable_sha(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _canonical_text(value: object, label: str, *, casefold: bool = True) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        raise QualitySupportUnavailable(f"{label} cannot be missing")
    text = unicodedata.normalize("NFKC", str(value)).strip()
    if casefold:
        text = text.casefold()
    if not text:
        raise QualitySupportUnavailable(f"{label} cannot be empty")
    return text


def _canonical_timestamp(value: object, label: str) -> str:
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        raise QualitySupportUnavailable(f"{label} is invalid")
    return timestamp.isoformat().replace("+00:00", "Z")


def _canonical_number(value: object, label: str) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise QualitySupportUnavailable(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise QualitySupportUnavailable(f"{label} must be finite")
    if number == 0.0:
        number = 0.0
    return format(number, ".17g")


def _canonical_bool(value: object, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    if isinstance(value, (float, np.floating)) and float(value) in (0.0, 1.0):
        return bool(value)
    raise QualitySupportUnavailable(f"{label} must be boolean")


_TIMESTAMP_COLUMNS = {"donor_skill_date", "donor_event_start", "target_event_start"}
_BOOLEAN_COLUMNS = {
    "terminal",
    "holed",
    "complete_atom",
    "quality_reward_available",
    "donor_ability_available",
}
_INTEGER_COLUMNS = {"causal_strokes", "penalty_strokes", "stroke_cost", "round_no"}
_NUMERIC_COLUMNS = {
    "start_reference_value",
    "end_reference_value",
    "fixed_reference_reward",
    "donor_ability_sg",
    "base_hierarchy_mass",
    "spatial_model_weight",
    "quality_base_mass",
    "implicit_holeout_probability",
    "declared_app_base_holeout_probability",
    "target_ott_sg",
    "target_app_sg",
    "target_arg_sg",
    "target_putt_sg",
    "target_total_sg",
}


def _canonical_hash_value(column: str, value: object) -> object:
    if column in _TIMESTAMP_COLUMNS:
        return _canonical_timestamp(value, column)
    if column in _BOOLEAN_COLUMNS:
        return _canonical_bool(value, column)
    if column in _INTEGER_COLUMNS:
        numeric = float(value)
        if not math.isfinite(numeric) or numeric != round(numeric):
            raise QualitySupportUnavailable(f"{column} must be an integer")
        return int(round(numeric))
    if column in _NUMERIC_COLUMNS:
        return _canonical_number(value, column)
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    return unicodedata.normalize("NFKC", str(value)).strip().casefold()


def _records_sha(frame: pd.DataFrame, columns: Sequence[str], contract: str) -> str:
    ordered = frame.sort_values(["support_key", "atom_id"]).reset_index(drop=True)
    records: list[dict[str, object]] = []
    for row in ordered[list(columns)].itertuples(index=False, name=None):
        record: dict[str, object] = {}
        for column, value in zip(columns, row):
            record[str(column)] = _canonical_hash_value(str(column), value)
        records.append(record)
    return _stable_sha({"contract": contract, "records": records})


def _physical_atom_digest(row: Mapping[str, object]) -> str:
    payload = {
        column: _canonical_hash_value(column, row[column])
        for column in PHYSICAL_ATOM_HASH_COLUMNS
    }
    return _stable_sha({"contract": PHYSICAL_ATOM_CONTRACT, "atom": payload})


def seal_physical_atom_hashes(frame: pd.DataFrame) -> pd.DataFrame:
    """Seal every atom to its canonical physical and donor identity content."""

    missing = set(PHYSICAL_ATOM_HASH_COLUMNS).difference(frame.columns)
    if missing:
        raise QualitySupportUnavailable(
            f"Cannot seal physical atoms; missing fields={sorted(missing)}"
        )
    out = frame.copy()
    out["physical_atom_sha256"] = [
        _physical_atom_digest(row)
        for row in out[list(PHYSICAL_ATOM_HASH_COLUMNS)].to_dict("records")
    ]
    return out


def _reference_definition_payload(frame: pd.DataFrame) -> dict[str, object]:
    required = {
        "reward_reference_descriptor",
        "reward_reference_version",
        "start_state_key",
        "start_reference_value",
        "end_state_key",
        "end_reference_value",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise QualitySupportUnavailable(
            f"Cannot bind reward reference; missing fields={sorted(missing)}"
        )
    descriptors = {
        _canonical_text(value, "reward_reference_descriptor")
        for value in frame["reward_reference_descriptor"]
    }
    versions = {
        _canonical_text(value, "reward_reference_version")
        for value in frame["reward_reference_version"]
    }
    if len(descriptors) != 1 or len(versions) != 1:
        raise QualitySupportUnavailable(
            "Reward reference must have one canonical descriptor and version"
        )
    state_values: dict[str, str] = {}
    for row in frame.itertuples(index=False):
        for key_value, reference_value in (
            (row.start_state_key, row.start_reference_value),
            (row.end_state_key, row.end_reference_value),
        ):
            key = _canonical_text(key_value, "reference state key")
            value = _canonical_number(reference_value, "reference state value")
            if key in state_values and state_values[key] != value:
                raise QualitySupportUnavailable(
                    f"Reference value changes for state {key!r}"
                )
            state_values[key] = value
    return {
        "contract": REWARD_REFERENCE_CONTRACT,
        "descriptor": next(iter(descriptors)),
        "version": next(iter(versions)),
        "state_value_table": [
            {"state_key": key, "reference_value": state_values[key]}
            for key in sorted(state_values)
        ],
    }


def reward_reference_sha256(frame: pd.DataFrame) -> str:
    """Digest the full canonical state-value table plus descriptor/version."""

    return _stable_sha(_reference_definition_payload(frame))


def seal_reward_reference_hash(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["reward_reference_sha256"] = reward_reference_sha256(out)
    return out


def _canonical_event_start_map(frame: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}
    pairs = (
        ("donor_event_key", "donor_event_start"),
        ("target_event_key", "target_event_start"),
    )
    for key_column, start_column in pairs:
        if key_column not in frame or start_column not in frame:
            raise QualitySupportUnavailable(
                f"Event mapping misses {key_column!r} or {start_column!r}"
            )
        for key_value, start_value in frame[[key_column, start_column]].itertuples(
            index=False, name=None
        ):
            key = _canonical_text(key_value, key_column)
            start = _canonical_timestamp(start_value, start_column)
            if key in mapping and mapping[key] != start:
                raise QualitySupportUnavailable(
                    f"Event {key!r} maps to multiple canonical start timestamps"
                )
            mapping[key] = start
    return mapping


def _donor_source_payload(frame: pd.DataFrame) -> dict[str, object]:
    required = {
        "donor_source_descriptor",
        "donor_source_version",
        "donor_event_key",
        "donor_event_start",
        "donor_player_id",
        "category",
        "donor_skill_date",
        "donor_ability_sg",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise QualitySupportUnavailable(
            f"Cannot bind donor source; missing fields={sorted(missing)}"
        )
    descriptors = {
        _canonical_text(value, "donor_source_descriptor")
        for value in frame["donor_source_descriptor"]
    }
    versions = {
        _canonical_text(value, "donor_source_version")
        for value in frame["donor_source_version"]
    }
    if len(descriptors) != 1 or len(versions) != 1:
        raise QualitySupportUnavailable(
            "Donor source must have one canonical descriptor and version"
        )
    event_starts = _canonical_event_start_map(frame)
    records_by_identity: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in frame.itertuples(index=False):
        event_key = _canonical_text(row.donor_event_key, "donor_event_key")
        player_id = _canonical_text(row.donor_player_id, "donor_player_id")
        category = _canonical_text(row.category, "category")
        identity = (event_key, player_id, category)
        record = {
            "donor_event_key": event_key,
            "donor_event_start": event_starts[event_key],
            "donor_player_id": player_id,
            "category": category,
            "donor_skill_date": _canonical_timestamp(
                row.donor_skill_date, "donor_skill_date"
            ),
            "donor_ability_sg": _canonical_number(
                row.donor_ability_sg, "donor_ability_sg"
            ),
        }
        if identity in records_by_identity and records_by_identity[identity] != record:
            raise QualitySupportUnavailable(
                "One donor event/player/category has conflicting snapshot content"
            )
        records_by_identity[identity] = record
    return {
        "contract": DONOR_SOURCE_CONTRACT,
        "descriptor": next(iter(descriptors)),
        "version": next(iter(versions)),
        "event_start_mapping": [
            {"event_key": key, "event_start": event_starts[key]}
            for key in sorted(
                {
                    _canonical_text(value, "donor_event_key")
                    for value in frame["donor_event_key"]
                }
            )
        ],
        "snapshot_records": [
            records_by_identity[key] for key in sorted(records_by_identity)
        ],
    }


def donor_source_sha256(frame: pd.DataFrame) -> str:
    """Digest the canonical strict-prior donor source actually used by atoms."""

    return _stable_sha(_donor_source_payload(frame))


def _donor_snapshot_digest(row: Mapping[str, object], source_sha256: str) -> str:
    payload = {
        "contract": DONOR_SNAPSHOT_CONTRACT,
        "donor_source_sha256": source_sha256,
        "donor_event_key": _canonical_text(row["donor_event_key"], "donor_event_key"),
        "donor_event_start": _canonical_timestamp(
            row["donor_event_start"], "donor_event_start"
        ),
        "donor_player_id": _canonical_text(row["donor_player_id"], "donor_player_id"),
        "category": _canonical_text(row["category"], "category"),
        "donor_skill_date": _canonical_timestamp(
            row["donor_skill_date"], "donor_skill_date"
        ),
        "donor_ability_sg": _canonical_number(
            row["donor_ability_sg"], "donor_ability_sg"
        ),
    }
    return _stable_sha(payload)


def canonical_donor_quality_group_id(row: Mapping[str, object]) -> str:
    """Derive event/category/state normalization; donor player is deliberately excluded."""

    return _stable_sha(
        {
            "contract": "canonical_donor_quality_group_v2",
            "donor_event_key": _canonical_text(
                row["donor_event_key"], "donor_event_key"
            ),
            "category": _canonical_text(row["category"], "category"),
            "start_state_key": _canonical_text(
                row["start_state_key"], "start_state_key"
            ),
        }
    )


def seal_donor_provenance(frame: pd.DataFrame) -> pd.DataFrame:
    """Seal full-source, per-snapshot, and internally derived group provenance."""

    out = frame.copy()
    source_sha = donor_source_sha256(out)
    out["donor_source_sha256"] = source_sha
    records = out.to_dict("records")
    out["donor_snapshot_sha256"] = [
        _donor_snapshot_digest(row, source_sha) for row in records
    ]
    out["donor_quality_group_id"] = [
        canonical_donor_quality_group_id(row) for row in records
    ]
    return out


def _app_holeout_equivalence_digest(group: pd.DataFrame) -> str:
    support_keys = {
        _canonical_text(value, "support_key") for value in group["support_key"]
    }
    if len(support_keys) != 1:
        raise QualitySupportUnavailable("APP equivalence accepts exactly one support")
    records = []
    for row in group.sort_values("atom_id").itertuples(index=False):
        records.append(
            {
                "atom_id": _canonical_text(row.atom_id, "atom_id"),
                "physical_atom_sha256": str(row.physical_atom_sha256),
                "terminal": _canonical_bool(row.terminal, "terminal"),
                "holed": _canonical_bool(row.holed, "holed"),
                "atom_kind": _canonical_text(row.atom_kind, "atom_kind"),
                "next_support_key": unicodedata.normalize(
                    "NFKC", str(row.next_support_key)
                ).strip().casefold(),
                "quality_base_mass": _canonical_number(
                    row.quality_base_mass, "quality_base_mass"
                ),
                "declared_app_base_holeout_probability": _canonical_number(
                    row.declared_app_base_holeout_probability,
                    "declared_app_base_holeout_probability",
                ),
            }
        )
    return _stable_sha(
        {
            "contract": APP_HOLEOUT_EQUIVALENCE_CONTRACT,
            "semantics": APP_HOLEOUT_CONTRACT,
            "support_key": next(iter(support_keys)),
            "atoms": records,
        }
    )


def seal_app_holeout_equivalence(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["app_holeout_equivalence_sha256"] = "not_applicable"
    app = out["category"].astype(str).str.strip().str.casefold().eq("app")
    for _, index in out.loc[app].groupby("support_key", sort=False).groups.items():
        positions = list(index)
        out.loc[positions, "app_holeout_equivalence_sha256"] = (
            _app_holeout_equivalence_digest(out.loc[positions])
        )
    return out


def seal_complete_atom_contracts(frame: pd.DataFrame) -> pd.DataFrame:
    """Seal every content-derived research contract in dependency order."""

    out = seal_reward_reference_hash(frame)
    out = seal_donor_provenance(out)
    out = seal_physical_atom_hashes(out)
    out = seal_app_holeout_equivalence(out)
    return seal_quality_input_hashes(out)


def quality_input_sha256(frame: pd.DataFrame) -> str:
    """Hash one conditional support's quality-critical fields."""

    missing = set(QUALITY_HASH_COLUMNS).difference(frame.columns)
    if missing:
        raise QualitySupportUnavailable(
            f"Cannot hash quality support; missing fields={sorted(missing)}"
        )
    if frame["support_key"].astype(str).nunique() != 1:
        raise ValueError("quality_input_sha256 accepts exactly one support")
    return _records_sha(frame, QUALITY_HASH_COLUMNS, ATOM_SCHEMA_CONTRACT)


def seal_quality_input_hashes(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a content hash repeated on every support atom."""

    out = frame.copy()
    out["quality_input_sha256"] = ""
    for _, index in out.groupby("support_key", observed=True, sort=False).groups.items():
        positions = list(index)
        digest = quality_input_sha256(out.loc[positions])
        out.loc[positions, "quality_input_sha256"] = digest
    return out


def _canonical_production_target_content(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(PRODUCTION_TARGET_CONTENT_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Production target misses fields={sorted(missing)}")
    out = frame[list(PRODUCTION_TARGET_CONTENT_COLUMNS)].copy()
    if out.empty:
        raise ValueError("Production target input cannot be empty")
    out["event_key"] = [
        _canonical_text(value, "production event_key") for value in out["event_key"]
    ]
    out["player_id"] = [
        _canonical_text(value, "production player_id") for value in out["player_id"]
    ]
    rounds = pd.to_numeric(out["round_no"], errors="coerce")
    if (
        rounds.isna().any()
        or not np.equal(rounds.to_numpy(float), np.rint(rounds.to_numpy(float))).all()
        or not rounds.isin([1, 2, 3, 4]).all()
    ):
        raise ValueError("Production round_no must be in 1..4")
    out["round_no"] = rounds.astype(np.int8)
    if out.duplicated(["event_key", "player_id", "round_no"]).any():
        raise ValueError(
            "Production target player-round keys are duplicated after canonicalization"
        )
    category_columns = [f"target_{category}_sg" for category in CATEGORIES]
    numeric = out[category_columns + ["target_total_sg"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("Production category/total means must be finite")
    residual = numeric[category_columns].sum(axis=1) - numeric["target_total_sg"]
    if not np.allclose(residual, 0.0, atol=1e-10, rtol=0.0):
        raise ValueError("Production category means do not sum to target_total_sg")
    out[category_columns + ["target_total_sg"]] = numeric
    return out


def production_input_sha256(row: Mapping[str, object]) -> str:
    payload = {
        column: _canonical_hash_value(column, row[column])
        for column in PRODUCTION_TARGET_CONTENT_COLUMNS
    }
    return _stable_sha({"contract": PRODUCTION_TARGET_CONTRACT, "player_round": payload})


def seal_production_target_hashes(frame: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize player-round keys and seal every target to its own values."""

    out = _canonical_production_target_content(frame)
    out["production_input_sha256"] = [
        production_input_sha256(row) for row in out.to_dict("records")
    ]
    return out.sort_values(["event_key", "player_id", "round_no"]).reset_index(drop=True)


def validate_production_target_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate, but never create, deterministic production mean targets."""

    missing = set(PRODUCTION_TARGET_REQUIRED_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Production target misses fields={sorted(missing)}")
    out = _canonical_production_target_content(frame)
    declared = frame["production_input_sha256"].astype(str).reset_index(drop=True)
    if not declared.map(_is_sha256).all():
        raise ValueError("production_input_sha256 must be lowercase SHA-256")
    observed = pd.Series(
        [production_input_sha256(row) for row in out.to_dict("records")]
    )
    if not declared.equals(observed):
        raise ValueError(
            "production_input_sha256 is stale or not bound to canonical player-round content"
        )
    out["production_input_sha256"] = observed.to_numpy()
    return out.sort_values(["event_key", "player_id", "round_no"]).reset_index(drop=True)


@dataclass(frozen=True)
class SupportReadiness:
    source_contract: str
    category: str
    anchor_ready: bool
    missing_fields: tuple[str, ...]
    blockers: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def current_support_readiness(
    source_contract: str,
    *,
    category: str,
    available_fields: Iterable[str] | None = None,
) -> SupportReadiness:
    """Describe why an existing cache/pool cannot enter the anchor today."""

    canonical = str(category).strip().casefold()
    if canonical not in CATEGORIES:
        raise ValueError(f"Unknown category: {category!r}")
    if available_fields is None:
        if source_contract not in CURRENT_SUPPORT_FIELDS:
            raise ValueError(f"Unknown current support contract: {source_contract!r}")
        fields = set(CURRENT_SUPPORT_FIELDS[source_contract])
    else:
        fields = {str(value) for value in available_fields}
    missing = tuple(sorted(set(REQUIRED_ATOM_COLUMNS).difference(fields)))
    blockers: list[str] = []
    if missing:
        blockers.append("complete_atom_schema_missing")
    quality_fields = {
        "fixed_reference_reward",
        "quality_reward_available",
        "reward_reference_sha256",
        "donor_ability_sg",
        "donor_ability_available",
        "donor_quality_group_id",
        "donor_event_key",
        "donor_player_id",
        "donor_skill_date",
        "donor_event_start",
        "donor_ability_snapshot_kind",
        "donor_source_descriptor",
        "donor_source_version",
        "donor_source_sha256",
        "donor_snapshot_sha256",
        "target_event_key",
        "target_event_start",
        "declared_app_base_holeout_probability",
        "app_holeout_equivalence_sha256",
        "quality_input_sha256",
    }
    if quality_fields.difference(fields):
        blockers.append("reward_or_strict_prior_donor_ability_provenance_missing")
    if {
        "start_state_key",
        "end_state_key",
        "stroke_cost",
        "penalty_strokes",
        "physical_atom_sha256",
        "holed",
    }.difference(fields):
        blockers.append("complete_physical_state_or_penalty_accounting_missing")
    if canonical == "app" and "holeout_probability" in fields:
        blockers.append("app_holeout_is_separate_scalar_not_complete_atom")
    if source_contract == "markov.TransitionPool":
        blockers.append("selected_donor_identity_is_discarded_by_pool")
        blockers.append("sg_field_is_not_bound_to_frozen_reward_reference_hash")
    return SupportReadiness(
        source_contract=str(source_contract),
        category=canonical,
        anchor_ready=not blockers,
        missing_fields=missing,
        blockers=tuple(dict.fromkeys(blockers)),
    )


def assert_current_support_anchor_ready(
    source_contract: str,
    *,
    category: str,
    available_fields: Iterable[str] | None = None,
) -> None:
    readiness = current_support_readiness(
        source_contract,
        category=category,
        available_fields=available_fields,
    )
    if not readiness.anchor_ready:
        raise QualitySupportUnavailable(json.dumps(readiness.as_dict(), sort_keys=True))


def _coerce_bool(series: pd.Series, label: str) -> np.ndarray:
    values = series.to_numpy(object)
    valid = np.asarray(
        [isinstance(value, (bool, np.bool_)) or value in (0, 1) for value in values],
        bool,
    )
    if not valid.all():
        raise QualitySupportUnavailable(f"{label} must contain booleans only")
    return np.asarray(values, bool)


def validate_complete_atom_frame(
    frame: pd.DataFrame,
    *,
    expected_reward_reference_sha256: str,
    expected_donor_source_sha256: str,
    expected_target_event_key: str,
    expected_target_event_start: object,
) -> pd.DataFrame:
    """Validate an immutable complete-atom graph and return canonical rows."""

    missing = set(REQUIRED_ATOM_COLUMNS).difference(frame.columns)
    if missing:
        raise QualitySupportUnavailable(f"Complete atoms miss fields={sorted(missing)}")
    out = frame[list(REQUIRED_ATOM_COLUMNS)].copy()
    if out.empty:
        raise QualitySupportUnavailable("Complete atom graph cannot be empty")
    for column in (
        "support_key",
        "atom_id",
        "physical_atom_sha256",
        "category",
        "start_state_key",
        "end_state_key",
        "atom_kind",
        "hazard_resolution_status",
        "reward_reference_sha256",
        "donor_quality_group_id",
        "donor_event_key",
        "donor_player_id",
        "donor_ability_snapshot_kind",
        "donor_source_descriptor",
        "donor_source_version",
        "donor_source_sha256",
        "donor_snapshot_sha256",
        "target_event_key",
        "holeout_representation",
        "app_holeout_equivalence_sha256",
        "quality_input_sha256",
        "reward_reference_descriptor",
        "reward_reference_version",
    ):
        out[column] = [_canonical_text(value, column) for value in out[column]]
    out["next_support_key"] = [
        ""
        if value is None or (not isinstance(value, str) and pd.isna(value))
        else unicodedata.normalize("NFKC", str(value)).strip().casefold()
        for value in out["next_support_key"]
    ]
    if out.duplicated(["support_key", "atom_id"]).any():
        raise QualitySupportUnavailable("Atom IDs are duplicated inside a support")
    if not out["category"].isin(CATEGORIES).all():
        raise QualitySupportUnavailable("Atom category is outside OTT/APP/ARG/PUTT")
    if not out["physical_atom_sha256"].map(_is_sha256).all():
        raise QualitySupportUnavailable("physical_atom_sha256 is invalid")
    if not out["reward_reference_sha256"].map(_is_sha256).all():
        raise QualitySupportUnavailable("reward_reference_sha256 is invalid")
    reward_hashes = set(out["reward_reference_sha256"])
    if len(reward_hashes) != 1:
        raise QualitySupportUnavailable("Complete atoms use multiple reward references")
    reward_sha = next(iter(reward_hashes))
    if not _is_sha256(expected_reward_reference_sha256):
        raise QualitySupportUnavailable("Expected reward reference digest is invalid")
    if reward_sha != str(expected_reward_reference_sha256):
        raise QualitySupportUnavailable("Complete atom reward reference changed")
    if not out["donor_source_sha256"].map(_is_sha256).all():
        raise QualitySupportUnavailable("donor_source_sha256 is invalid")
    if not out["donor_snapshot_sha256"].map(_is_sha256).all():
        raise QualitySupportUnavailable("donor_snapshot_sha256 is invalid")
    if not _is_sha256(expected_donor_source_sha256):
        raise QualitySupportUnavailable("Expected donor source digest is invalid")

    terminal = _coerce_bool(out["terminal"], "terminal")
    holed = _coerce_bool(out["holed"], "holed")
    complete = _coerce_bool(out["complete_atom"], "complete_atom")
    reward_available = _coerce_bool(
        out["quality_reward_available"], "quality_reward_available"
    )
    ability_available = _coerce_bool(
        out["donor_ability_available"], "donor_ability_available"
    )
    if not complete.all():
        raise QualitySupportUnavailable("Every quality atom must be complete")
    if not reward_available.all():
        raise QualitySupportUnavailable(
            "Missing fixed-reference reward cannot be treated as zero"
        )
    if not ability_available.all():
        raise QualitySupportUnavailable(
            "Missing donor ability cannot be treated as neutral zero"
        )
    out["terminal"] = terminal
    out["holed"] = holed
    out["complete_atom"] = complete
    out["quality_reward_available"] = reward_available
    out["donor_ability_available"] = ability_available

    numeric_columns = (
        "causal_strokes",
        "penalty_strokes",
        "stroke_cost",
        "start_reference_value",
        "end_reference_value",
        "fixed_reference_reward",
        "donor_ability_sg",
        "base_hierarchy_mass",
        "spatial_model_weight",
        "quality_base_mass",
        "implicit_holeout_probability",
        "declared_app_base_holeout_probability",
    )
    numeric = out[list(numeric_columns)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise QualitySupportUnavailable("Complete atom numeric fields must be finite")
    out[list(numeric_columns)] = numeric
    integer_columns = ("causal_strokes", "penalty_strokes", "stroke_cost")
    for column in integer_columns:
        if not np.array_equal(
            numeric[column].to_numpy(float), np.rint(numeric[column].to_numpy(float))
        ):
            raise QualitySupportUnavailable(f"{column} must be integer physical strokes")
        out[column] = np.rint(numeric[column]).astype(np.int16)
    if not out["causal_strokes"].eq(1).all():
        raise QualitySupportUnavailable("One complete atom must own exactly one causal shot")
    if out["penalty_strokes"].lt(0).any():
        raise QualitySupportUnavailable("Penalty strokes cannot be negative")
    if not out["stroke_cost"].eq(
        out["causal_strokes"] + out["penalty_strokes"]
    ).all():
        raise QualitySupportUnavailable(
            "stroke_cost must equal causal shot plus linked penalty strokes"
        )
    observed_physical = pd.Series(
        [
            _physical_atom_digest(row)
            for row in out[list(PHYSICAL_ATOM_HASH_COLUMNS)].to_dict("records")
        ],
        index=out.index,
    )
    if not out["physical_atom_sha256"].equals(observed_physical):
        raise QualitySupportUnavailable(
            "physical_atom_sha256 is stale or not bound to canonical physical content"
        )
    penalty = out["penalty_strokes"].gt(0)
    if not out.loc[penalty, "hazard_resolution_status"].isin(
        ["resolved_complete_atom", "collapsed_exact_transition"]
    ).all():
        raise QualitySupportUnavailable("Penalty atoms are not coherently resolved")
    if not out.loc[~penalty, "hazard_resolution_status"].eq("none").all():
        raise QualitySupportUnavailable("Non-penalty atoms have an invalid hazard status")

    if out["base_hierarchy_mass"].le(0.0).any():
        raise QualitySupportUnavailable("Every structural atom needs positive hierarchy mass")
    if out["spatial_model_weight"].le(0.0).any():
        raise QualitySupportUnavailable("Spatial model weights must be positive")
    expected_mass = (
        out["base_hierarchy_mass"].to_numpy(float)
        * out["spatial_model_weight"].to_numpy(float)
    )
    if not np.allclose(
        out["quality_base_mass"].to_numpy(float),
        expected_mass,
        atol=1e-15,
        rtol=1e-12,
    ):
        raise QualitySupportUnavailable(
            "quality_base_mass must be hierarchy mass times spatial model weight"
        )

    observed_reward_sha = reward_reference_sha256(out)
    if not out["reward_reference_sha256"].eq(observed_reward_sha).all():
        raise QualitySupportUnavailable(
            "reward_reference_sha256 is not bound to the canonical full value table"
        )
    if observed_reward_sha != str(expected_reward_reference_sha256):
        raise QualitySupportUnavailable("Canonical reward reference changed")

    expected_reward = (
        out["start_reference_value"].to_numpy(float)
        - out["stroke_cost"].to_numpy(float)
        - out["end_reference_value"].to_numpy(float)
    )
    reward_residual = out["fixed_reference_reward"].to_numpy(float) - expected_reward
    if np.max(np.abs(reward_residual), initial=0.0) > ATOM_REWARD_TOLERANCE:
        raise QualitySupportUnavailable(
            "Fixed-reference atom reward fails exact stroke/value accounting"
        )
    terminal_rows = out["terminal"]
    if not np.array_equal(
        terminal_rows.to_numpy(bool), out["holed"].to_numpy(bool)
    ):
        raise QualitySupportUnavailable(
            "Terminal and holed semantics must be exactly equivalent per atom"
        )
    if not out.loc[terminal_rows, "end_state_key"].eq(TERMINAL_STATE_KEY).all():
        raise QualitySupportUnavailable("Terminal atoms must end at terminal state")
    if not np.allclose(
        out.loc[terminal_rows, "end_reference_value"], 0.0, atol=0.0, rtol=0.0
    ):
        raise QualitySupportUnavailable("Terminal reference value must be exactly zero")
    if not out.loc[terminal_rows, "next_support_key"].eq("").all():
        raise QualitySupportUnavailable("Terminal atoms cannot point to another support")
    if out.loc[~terminal_rows, "next_support_key"].eq("").any():
        raise QualitySupportUnavailable("Nonterminal atoms need a next support")
    if out.loc[terminal_rows, "penalty_strokes"].gt(0).any():
        raise QualitySupportUnavailable(
            "A linked penalty/drop sequence must continue to a playable state"
        )

    app_rows = out["category"].eq("app")
    if not out.loc[app_rows, "holeout_representation"].eq(
        APP_HOLEOUT_CONTRACT
    ).all():
        raise QualitySupportUnavailable("APP support does not declare explicit holeout atoms")
    if not np.allclose(
        out.loc[app_rows, "implicit_holeout_probability"],
        0.0,
        atol=0.0,
        rtol=0.0,
    ):
        raise QualitySupportUnavailable("Independent APP holeout probability is forbidden")
    app_terminal = app_rows & terminal_rows
    app_holeout = out["atom_kind"].eq("app_holeout")
    if not np.array_equal(
        app_holeout.to_numpy(bool), app_terminal.to_numpy(bool)
    ):
        raise QualitySupportUnavailable(
            "APP holeout atom must be present iff category is APP and atom is terminal/holed"
        )
    if out.loc[~app_rows, "atom_kind"].eq("app_holeout").any():
        raise QualitySupportUnavailable("APP holeout atom is assigned to another category")
    if not out.loc[~app_rows, "holeout_representation"].eq("not_applicable").all():
        raise QualitySupportUnavailable("Non-APP holeout representation must be not_applicable")
    if not np.allclose(
        out.loc[~app_rows, "declared_app_base_holeout_probability"],
        0.0,
        atol=0.0,
        rtol=0.0,
    ):
        raise QualitySupportUnavailable(
            "Non-APP declared holeout probability must be exactly zero"
        )
    if not out.loc[~app_rows, "app_holeout_equivalence_sha256"].eq(
        "not_applicable"
    ).all():
        raise QualitySupportUnavailable(
            "Non-APP holeout equivalence must be not_applicable"
        )
    for support_key, group in out.loc[app_rows].groupby(
        "support_key", observed=True, sort=False
    ):
        if group["declared_app_base_holeout_probability"].nunique(
            dropna=False
        ) != 1:
            raise QualitySupportUnavailable(
                f"APP support {support_key!r} has multiple declared holeout probabilities"
            )
        declared_holeout = float(
            group["declared_app_base_holeout_probability"].iloc[0]
        )
        if not 0.0 <= declared_holeout <= 1.0:
            raise QualitySupportUnavailable(
                f"APP support {support_key!r} declared holeout probability is outside [0,1]"
            )
        terminal_mask = group["terminal"].to_numpy(bool)
        has_terminal_holeout = bool(terminal_mask.any())
        if (declared_holeout > 0.0) != has_terminal_holeout:
            raise QualitySupportUnavailable(
                f"APP support {support_key!r} terminal holeout coverage must exist iff "
                "declared expected mass is positive"
            )
        base_mass = group["quality_base_mass"].to_numpy(float)
        actual_holeout = float(base_mass[terminal_mask].sum() / base_mass.sum())
        if not math.isclose(
            actual_holeout,
            declared_holeout,
            rel_tol=0.0,
            abs_tol=APP_HOLEOUT_MASS_TOLERANCE,
        ):
            raise QualitySupportUnavailable(
                f"APP support {support_key!r} actual base holeout mass "
                f"{actual_holeout:.17g} does not equal declared "
                f"{declared_holeout:.17g}"
            )
        declared_equivalence = set(group["app_holeout_equivalence_sha256"])
        observed_equivalence = _app_holeout_equivalence_digest(group)
        if declared_equivalence != {observed_equivalence}:
            raise QualitySupportUnavailable(
                f"APP support {support_key!r} holeout semantics/equivalence digest is stale"
            )

    snapshot_kind = out["donor_ability_snapshot_kind"]
    if not snapshot_kind.eq(DONOR_SNAPSHOT_CONTRACT).all():
        raise QualitySupportUnavailable("Donor ability snapshot is not strict-prior as-of")
    skill_date = pd.to_datetime(out["donor_skill_date"], errors="coerce", utc=True)
    event_start = pd.to_datetime(out["donor_event_start"], errors="coerce", utc=True)
    target_start = pd.to_datetime(out["target_event_start"], errors="coerce", utc=True)
    if skill_date.isna().any() or event_start.isna().any() or target_start.isna().any():
        raise QualitySupportUnavailable("Donor skill/event/target timestamps are invalid")
    if not (skill_date < event_start).all():
        raise QualitySupportUnavailable("Donor ability is not strictly prior to donor event")
    if not (event_start < target_start).all():
        raise QualitySupportUnavailable(
            "Donor event is not strictly prior to the target-event cutoff"
        )
    out["donor_skill_date"] = skill_date
    out["donor_event_start"] = event_start
    out["target_event_start"] = target_start

    expected_target_key = _canonical_text(
        expected_target_event_key, "expected_target_event_key"
    )
    expected_target_start_text = _canonical_timestamp(
        expected_target_event_start, "expected_target_event_start"
    )
    if not out["target_event_key"].eq(expected_target_key).all():
        raise QualitySupportUnavailable("Atom target event identity changed")
    if not all(
        _canonical_timestamp(value, "target_event_start")
        == expected_target_start_text
        for value in out["target_event_start"]
    ):
        raise QualitySupportUnavailable("Atom target event start changed")
    _canonical_event_start_map(out)

    observed_source_sha = donor_source_sha256(out)
    declared_source_hashes = set(out["donor_source_sha256"])
    if declared_source_hashes != {observed_source_sha}:
        raise QualitySupportUnavailable(
            "donor_source_sha256 is stale or not bound to canonical source content"
        )
    if observed_source_sha != str(expected_donor_source_sha256):
        raise QualitySupportUnavailable("Canonical donor source changed")
    observed_snapshots = pd.Series(
        [
            _donor_snapshot_digest(row, observed_source_sha)
            for row in out.to_dict("records")
        ],
        index=out.index,
    )
    if not out["donor_snapshot_sha256"].equals(observed_snapshots):
        raise QualitySupportUnavailable(
            "donor_snapshot_sha256 is stale or not bound to strict-prior snapshot content"
        )
    observed_groups = pd.Series(
        [canonical_donor_quality_group_id(row) for row in out.to_dict("records")],
        index=out.index,
    )
    if not out["donor_quality_group_id"].equals(observed_groups):
        raise QualitySupportUnavailable(
            "donor_quality_group_id must be internally derived from event/category/conditioning-state"
        )
    out["donor_quality_group_id"] = observed_groups

    for support_key, group in out.groupby("support_key", observed=True, sort=False):
        for column in ("category", "start_state_key", "start_reference_value"):
            if group[column].nunique(dropna=False) != 1:
                raise QualitySupportUnavailable(
                    f"Support {support_key!r} does not have one {column}"
                )
        declared = set(group["quality_input_sha256"])
        observed = quality_input_sha256(group)
        if declared != {observed}:
            raise QualitySupportUnavailable(
                f"Support {support_key!r} quality_input_sha256 is stale"
            )

    state_values: dict[str, float] = {TERMINAL_STATE_KEY: 0.0}
    for row in out.itertuples(index=False):
        for state_key, value in (
            (str(row.start_state_key), float(row.start_reference_value)),
            (str(row.end_state_key), float(row.end_reference_value)),
        ):
            if state_key in state_values and not math.isclose(
                state_values[state_key], value, rel_tol=0.0, abs_tol=1e-12
            ):
                raise QualitySupportUnavailable(
                    f"Reference value changes for state {state_key!r}"
                )
            state_values[state_key] = value
    return out.sort_values(["support_key", "atom_id"]).reset_index(drop=True)


def load_complete_atom_contract(
    path: Path,
    *,
    expected_reward_reference_sha256: str,
    expected_donor_source_sha256: str,
    expected_target_event_key: str,
    expected_target_event_start: object,
    allow_round_trip_csv: bool = False,
) -> pd.DataFrame:
    """Load and validate a persisted v3 atom contract without float corruption.

    Parquet is the authoritative serialization.  Generic ``pandas.read_csv`` is
    unsafe for hash-bound IEEE-754 values because its default parser need not
    round-trip the final bit.  CSV therefore fails closed unless the caller
    explicitly opts into this loader's ``float_precision='round_trip'`` path;
    that compatibility path always emits a warning.
    """

    contract_path = Path(path)
    suffix = contract_path.suffix.casefold()
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(contract_path)
    elif suffix == ".csv":
        if not allow_round_trip_csv:
            raise QualitySupportUnavailable(
                "Generic CSV atom loading is unsafe and disabled; use Parquet or "
                "allow_round_trip_csv=True through load_complete_atom_contract"
            )
        warnings.warn(
            "Loading a hash-bound atom CSV through the explicit round-trip parser; "
            "Parquet is authoritative",
            UnsafeContractSerializationWarning,
            stacklevel=2,
        )
        frame = pd.read_csv(
            contract_path,
            keep_default_na=False,
            float_precision="round_trip",
        )
    else:
        raise QualitySupportUnavailable(
            f"Unsupported atom-contract serialization: {contract_path.suffix!r}"
        )
    return validate_complete_atom_frame(
        frame,
        expected_reward_reference_sha256=expected_reward_reference_sha256,
        expected_donor_source_sha256=expected_donor_source_sha256,
        expected_target_event_key=expected_target_event_key,
        expected_target_event_start=expected_target_event_start,
    )


@dataclass(frozen=True)
class _NodeMoment:
    reward: np.ndarray
    opportunities: np.ndarray
    structural_lower: np.ndarray
    structural_upper: np.ndarray
    expected_strokes: float
    expected_penalties: float
    expected_causal_shots: float
    maximum_path_depth: int


class PhysicalChainExpectationAdapter:
    """Exact expectation callback over a finite unrolled absorbing atom graph."""

    def __init__(
        self,
        atoms: pd.DataFrame,
        *,
        start_support_keys: Sequence[str],
        crn: anchor.CommonRandomContext,
        expected_reward_reference_sha256: str,
        expected_donor_source_sha256: str,
        expected_target_event_key: str,
        expected_target_event_start: object,
        expected_round_holes: int | None = 18,
    ) -> None:
        self.atoms = validate_complete_atom_frame(
            atoms,
            expected_reward_reference_sha256=expected_reward_reference_sha256,
            expected_donor_source_sha256=expected_donor_source_sha256,
            expected_target_event_key=expected_target_event_key,
            expected_target_event_start=expected_target_event_start,
        )
        self.start_support_keys = tuple(map(str, start_support_keys))
        if not self.start_support_keys:
            raise QualitySupportUnavailable("At least one hole-start support is required")
        if expected_round_holes is not None and len(self.start_support_keys) != int(
            expected_round_holes
        ):
            raise QualitySupportUnavailable(
                f"Expected {int(expected_round_holes)} hole starts; got {len(self.start_support_keys)}"
            )
        if not isinstance(crn, anchor.CommonRandomContext):
            raise TypeError("crn must be CommonRandomContext")
        self.crn = crn
        self.reward_reference_sha256 = str(expected_reward_reference_sha256)
        self.donor_source_sha256 = str(expected_donor_source_sha256)
        self.target_event_key = _canonical_text(
            expected_target_event_key, "expected_target_event_key"
        )
        self.target_event_start = _canonical_timestamp(
            expected_target_event_start, "expected_target_event_start"
        )
        self.supports = {
            str(key): group.reset_index(drop=True)
            for key, group in self.atoms.groupby("support_key", observed=True, sort=True)
        }
        missing_roots = sorted(set(self.start_support_keys).difference(self.supports))
        if missing_roots:
            raise QualitySupportUnavailable(f"Start supports are missing: {missing_roots}")
        self._validate_edges()
        self.topological_postorder = self._topological_postorder()
        payload = {
            "contract": EXPECTATION_CONTRACT,
            "start_support_keys": list(self.start_support_keys),
            "atom_frame_sha256": _records_sha(
                self.atoms, SUPPORT_HASH_COLUMNS, ATOM_SCHEMA_CONTRACT
            ),
        }
        self.support_sha256 = _stable_sha(payload)
        self.max_atom_telescoping_residual = float(
            np.max(
                np.abs(
                    self.atoms["fixed_reference_reward"].to_numpy(float)
                    - (
                        self.atoms["start_reference_value"].to_numpy(float)
                        - self.atoms["stroke_cost"].to_numpy(float)
                        - self.atoms["end_reference_value"].to_numpy(float)
                    )
                ),
                initial=0.0,
            )
        )

    def _validate_edges(self) -> None:
        start_state = {
            key: str(group["start_state_key"].iloc[0]) for key, group in self.supports.items()
        }
        for row in self.atoms.itertuples(index=False):
            if bool(row.terminal):
                continue
            next_key = str(row.next_support_key)
            if next_key not in self.supports:
                raise QualitySupportUnavailable(
                    f"Atom {row.atom_id!r} points to missing support {next_key!r}"
                )
            if str(row.end_state_key) != start_state[next_key]:
                raise QualitySupportUnavailable(
                    f"Atom {row.atom_id!r} end state does not match next support start"
                )

    def _topological_postorder(self) -> tuple[str, ...]:
        visiting: set[str] = set()
        visited: set[str] = set()
        postorder: list[str] = []

        def visit(key: str) -> None:
            if key in visiting:
                raise QualitySupportUnavailable(
                    "Support graph contains a cycle; unroll by transition/stroke budget first"
                )
            if key in visited:
                return
            visiting.add(key)
            group = self.supports[key]
            for child in sorted(set(group.loc[~group["terminal"], "next_support_key"])):
                visit(str(child))
            visiting.remove(key)
            visited.add(key)
            postorder.append(key)

        for root in self.start_support_keys:
            visit(root)
        unreachable = sorted(set(self.supports).difference(visited))
        if unreachable:
            raise QualitySupportUnavailable(
                f"Atom graph contains unreachable supports: {unreachable[:8]}"
            )
        return tuple(postorder)

    def __call__(
        self,
        parameters: anchor.TransitionQualityParameters,
        crn: anchor.CommonRandomContext,
    ) -> anchor.ChainExpectation:
        if not isinstance(parameters, anchor.TransitionQualityParameters):
            raise TypeError("parameters must be TransitionQualityParameters")
        if crn.digest != self.crn.digest:
            raise QualitySupportUnavailable("Expectation callback CRN identity changed")
        if parameters.reward_reference_sha256 != self.reward_reference_sha256:
            raise QualitySupportUnavailable("Quality parameters use a different reward reference")

        moments: dict[str, _NodeMoment] = {}
        clipped_donor_logs = 0
        clipped_target_logs = 0
        for support_key in self.topological_postorder:
            group = self.supports[support_key]
            category = str(group["category"].iloc[0])
            category_index = CATEGORY_INDEX[category]
            mass = anchor.whole_atom_probabilities(
                base_mass=group["quality_base_mass"].to_numpy(float),
                fixed_reference_reward=group["fixed_reference_reward"].to_numpy(float),
                donor_ability_sg=group["donor_ability_sg"].to_numpy(float),
                donor_group_id=group["donor_quality_group_id"].astype(str).to_numpy(),
                category=category,
                parameters=parameters,
            )
            clipped_donor_logs += int(mass.clipped_donor_logs)
            clipped_target_logs += int(mass.clipped_target_logs)
            branch_reward: list[np.ndarray] = []
            branch_opportunities: list[np.ndarray] = []
            branch_lower: list[np.ndarray] = []
            branch_upper: list[np.ndarray] = []
            branch_strokes: list[float] = []
            branch_penalties: list[float] = []
            branch_causal: list[float] = []
            branch_depth: list[int] = []
            for row in group.itertuples(index=False):
                if bool(row.terminal):
                    child_reward = np.zeros(len(CATEGORIES), dtype=float)
                    child_opportunities = np.zeros(len(CATEGORIES), dtype=float)
                    child_lower = np.zeros(len(CATEGORIES), dtype=float)
                    child_upper = np.zeros(len(CATEGORIES), dtype=float)
                    child_strokes = 0.0
                    child_penalties = 0.0
                    child_causal = 0.0
                    child_depth = 0
                else:
                    child = moments[str(row.next_support_key)]
                    child_reward = child.reward
                    child_opportunities = child.opportunities
                    child_lower = child.structural_lower
                    child_upper = child.structural_upper
                    child_strokes = child.expected_strokes
                    child_penalties = child.expected_penalties
                    child_causal = child.expected_causal_shots
                    child_depth = child.maximum_path_depth
                immediate_reward = np.zeros(len(CATEGORIES), dtype=float)
                immediate_reward[category_index] = float(row.fixed_reference_reward)
                immediate_opportunity = np.zeros(len(CATEGORIES), dtype=float)
                immediate_opportunity[category_index] = 1.0
                branch_reward.append(immediate_reward + child_reward)
                branch_opportunities.append(immediate_opportunity + child_opportunities)
                branch_lower.append(immediate_reward + child_lower)
                branch_upper.append(immediate_reward + child_upper)
                branch_strokes.append(float(row.stroke_cost) + child_strokes)
                branch_penalties.append(float(row.penalty_strokes) + child_penalties)
                branch_causal.append(float(row.causal_strokes) + child_causal)
                branch_depth.append(1 + child_depth)
            probability = mass.probability
            reward_matrix = np.stack(branch_reward)
            opportunity_matrix = np.stack(branch_opportunities)
            lower_matrix = np.stack(branch_lower)
            upper_matrix = np.stack(branch_upper)
            moment = _NodeMoment(
                reward=np.ascontiguousarray(probability @ reward_matrix),
                opportunities=np.ascontiguousarray(probability @ opportunity_matrix),
                structural_lower=np.min(lower_matrix, axis=0),
                structural_upper=np.max(upper_matrix, axis=0),
                expected_strokes=float(probability @ np.asarray(branch_strokes, float)),
                expected_penalties=float(probability @ np.asarray(branch_penalties, float)),
                expected_causal_shots=float(probability @ np.asarray(branch_causal, float)),
                maximum_path_depth=max(branch_depth),
            )
            start_value = float(group["start_reference_value"].iloc[0])
            telescoping_residual = float(moment.reward.sum() - (start_value - moment.expected_strokes))
            allowed = max(
                1e-10,
                self.max_atom_telescoping_residual * moment.maximum_path_depth + 1e-10,
            )
            if abs(telescoping_residual) > allowed:
                raise AssertionError(
                    f"Support {support_key!r} reward/stroke telescoping failed"
                )
            stroke_identity = (
                moment.expected_causal_shots
                + moment.expected_penalties
                - moment.expected_strokes
            )
            if abs(stroke_identity) > 1e-10:
                raise AssertionError(f"Support {support_key!r} stroke accounting failed")
            moments[support_key] = moment

        round_reward = np.sum(
            [moments[key].reward for key in self.start_support_keys], axis=0
        )
        round_opportunities = np.sum(
            [moments[key].opportunities for key in self.start_support_keys], axis=0
        )
        structural_lower = np.sum(
            [moments[key].structural_lower for key in self.start_support_keys], axis=0
        )
        structural_upper = np.sum(
            [moments[key].structural_upper for key in self.start_support_keys], axis=0
        )
        expected_strokes = float(
            sum(moments[key].expected_strokes for key in self.start_support_keys)
        )
        expected_penalties = float(
            sum(moments[key].expected_penalties for key in self.start_support_keys)
        )
        expected_causal = float(
            sum(moments[key].expected_causal_shots for key in self.start_support_keys)
        )
        starting_reference_value = float(
            sum(
                float(self.supports[key]["start_reference_value"].iloc[0])
                for key in self.start_support_keys
            )
        )
        round_telescoping_residual = float(
            round_reward.sum() - (starting_reference_value - expected_strokes)
        )
        maximum_round_depth = int(
            sum(moments[key].maximum_path_depth for key in self.start_support_keys)
        )
        allowed_round = max(
            1e-9,
            self.max_atom_telescoping_residual * maximum_round_depth + 1e-9,
        )
        if abs(round_telescoping_residual) > allowed_round:
            raise AssertionError("Round category SG does not telescope to physical strokes")
        stroke_identity_residual = expected_causal + expected_penalties - expected_strokes
        if abs(stroke_identity_residual) > 1e-9:
            raise AssertionError("Round causal-shot plus penalty accounting failed")
        explicit_app_holeouts = int(
            (
                self.atoms["category"].eq("app")
                & self.atoms["terminal"]
                & self.atoms["atom_kind"].eq("app_holeout")
            ).sum()
        )
        app_holeout_mass_residuals: list[float] = []
        for _, group in self.atoms.loc[self.atoms["category"].eq("app")].groupby(
            "support_key", observed=True, sort=False
        ):
            mass = group["quality_base_mass"].to_numpy(float)
            actual = float(mass[group["terminal"].to_numpy(bool)].sum() / mass.sum())
            declared = float(group["declared_app_base_holeout_probability"].iloc[0])
            app_holeout_mass_residuals.append(actual - declared)
        return anchor.ChainExpectation(
            mean_category_sg=tuple(map(float, round_reward)),
            mean_opportunities=tuple(map(float, round_opportunities)),
            support_lower_sg=tuple(map(float, structural_lower)),
            support_upper_sg=tuple(map(float, structural_upper)),
            simulations=int(crn.simulations),
            crn_sha256=crn.digest,
            reward_reference_sha256=self.reward_reference_sha256,
            support_sha256=self.support_sha256,
            quality_weight_config_sha256=parameters.quality_weight_config_sha256,
            quality_parameter_sha256=parameters.parameter_sha256,
            clipped_donor_logs=clipped_donor_logs,
            clipped_target_logs=clipped_target_logs,
            diagnostics={
                "contract": EXPECTATION_CONTRACT,
                "analytic_complete_support_integration": True,
                "support_states": int(len(self.supports)),
                "complete_atoms": int(len(self.atoms)),
                "hole_starts": int(len(self.start_support_keys)),
                "donor_source_sha256": self.donor_source_sha256,
                "target_event_key": self.target_event_key,
                "target_event_start": self.target_event_start,
                "explicit_app_holeout_atoms": explicit_app_holeouts,
                "app_supports_with_declared_holeout_mass": len(
                    app_holeout_mass_residuals
                ),
                "maximum_app_base_holeout_mass_residual": float(
                    np.max(np.abs(app_holeout_mass_residuals), initial=0.0)
                ),
                "independent_app_holeout_probability": 0.0,
                "donor_quality_group_contract": (
                    "donor_event_plus_category_plus_conditioning_state_excludes_player"
                ),
                "expected_physical_strokes": expected_strokes,
                "expected_causal_shots": expected_causal,
                "expected_penalty_strokes": expected_penalties,
                "stroke_identity_residual": float(stroke_identity_residual),
                "starting_reference_value": starting_reference_value,
                "round_telescoping_residual": round_telescoping_residual,
                "max_atom_telescoping_residual": self.max_atom_telescoping_residual,
                "maximum_round_path_depth": maximum_round_depth,
                "post_round_recenter": "forbidden",
            },
        )


def build_synthetic_128_state_atoms() -> tuple[pd.DataFrame, tuple[str, ...], str]:
    """Build a tiny unrolled physical graph for executable contract testing.

    This is synthetic mechanics data only.  It is not a production mean export.
    """

    reward_descriptor = "synthetic 128-state fixed reference value table"
    reward_version = "synthetic-2026-08-21-v2"
    donor_source_descriptor = "synthetic strict-prior donor ability source"
    donor_source_version = "synthetic-2026-08-21-v2"
    target_event_key = "synthetic_target_2024_03"
    target_event_start = "2024-03-07T00:00:00Z"
    depth_values = (4.30, 3.05, 2.15, 1.72, 1.30, 0.91, 0.51)
    values: dict[int, float] = {}
    for depth, base in enumerate(depth_values):
        for hole in range(18):
            values[depth * 18 + hole] = float(base + (hole - 8.5) * 0.005)
    values[126] = 2.30
    values[127] = 2.25

    def support_key(index: int) -> str:
        return f"synthetic128|support={index:03d}"

    def state_key(index: int) -> str:
        return f"synthetic128|state={index:03d}"

    def category_for(index: int) -> str:
        if index >= 126:
            return "arg"
        depth = index // 18
        if depth == 0:
            return "ott"
        if depth == 1:
            return "app"
        if depth == 2:
            return "arg"
        return "putt"

    def default_child(index: int) -> int | None:
        if index >= 126:
            return 54 + (index - 126)
        depth, hole = divmod(index, 18)
        return (depth + 1) * 18 + hole if depth < 6 else None

    donor_meta = {
        "event_a": ("2024-01-02T00:00:00Z", "2024-01-11T00:00:00Z"),
        "event_b": ("2024-02-02T00:00:00Z", "2024-02-15T00:00:00Z"),
    }
    rows: list[dict[str, object]] = []
    for index in range(128):
        category = category_for(index)
        primary_child = default_child(index)
        alternate_child = primary_child
        if index == 18:
            alternate_child = 126
        elif index == 19:
            alternate_child = 127
        specs = (
            ("a_progress", "event_a", -0.35, 0.20, primary_child, 0),
            ("a_terminal", "event_a", 0.25, 0.30, None, 0),
            ("b_progress", "event_b", 0.55, 0.28, alternate_child, 0),
            ("b_penalty_progress", "event_b", -0.15, 0.22, alternate_child, 1),
        )
        if primary_child is None:
            specs = tuple(
                (name, event, ability, mass, None, 0)
                for atom_index, (name, event, ability, mass, _, _) in enumerate(specs)
            )
        elif category == "app":
            # The fourth APP atom is a second empirical hole-out, not a
            # penalty side channel.  A penalty/drop outcome must recurse.
            name, event, ability, mass, _, _ = specs[3]
            specs = (*specs[:3], ("b_terminal", event, ability, mass, None, 0))
        for atom_index, (name, event, ability, base_mass, child, penalty) in enumerate(specs):
            terminal = child is None
            if category == "app" and terminal:
                penalty = 0
                atom_kind = "app_holeout"
            elif penalty:
                atom_kind = "collapsed_penalty"
            else:
                atom_kind = "standard"
            causal = 1
            cost = causal + int(penalty)
            end_value = 0.0 if terminal else values[int(child)]
            end_state = TERMINAL_STATE_KEY if terminal else state_key(int(child))
            reward = values[index] - cost - end_value
            atom_id = f"s{index:03d}|{atom_index}|{name}"
            skill_date, event_start = donor_meta[event]
            rows.append(
                {
                    "support_key": support_key(index),
                    "atom_id": atom_id,
                    "physical_atom_sha256": "",
                    "category": category,
                    "start_state_key": state_key(index),
                    "end_state_key": end_state,
                    "next_support_key": "" if terminal else support_key(int(child)),
                    "terminal": terminal,
                    "holed": terminal,
                    "atom_kind": atom_kind,
                    "complete_atom": True,
                    "causal_strokes": causal,
                    "penalty_strokes": int(penalty),
                    "stroke_cost": cost,
                    "hazard_resolution_status": (
                        "collapsed_exact_transition" if penalty else "none"
                    ),
                    "start_reference_value": values[index],
                    "end_reference_value": end_value,
                    "fixed_reference_reward": reward,
                    "quality_reward_available": True,
                    "reward_reference_descriptor": reward_descriptor,
                    "reward_reference_version": reward_version,
                    "reward_reference_sha256": "",
                    "donor_ability_sg": ability,
                    "donor_ability_available": True,
                    "donor_quality_group_id": "",
                    "donor_event_key": event,
                    "donor_player_id": f"{event}|player={atom_index}",
                    "donor_skill_date": skill_date,
                    "donor_event_start": event_start,
                    "donor_ability_snapshot_kind": DONOR_SNAPSHOT_CONTRACT,
                    "donor_source_descriptor": donor_source_descriptor,
                    "donor_source_version": donor_source_version,
                    "donor_source_sha256": "",
                    "donor_snapshot_sha256": "",
                    "target_event_key": target_event_key,
                    "target_event_start": target_event_start,
                    "base_hierarchy_mass": base_mass,
                    "spatial_model_weight": 1.0,
                    "quality_base_mass": base_mass,
                    "holeout_representation": (
                        APP_HOLEOUT_CONTRACT if category == "app" else "not_applicable"
                    ),
                    "declared_app_base_holeout_probability": (
                        0.52 if category == "app" else 0.0
                    ),
                    "app_holeout_equivalence_sha256": "",
                    "implicit_holeout_probability": 0.0,
                }
            )
    atoms = seal_complete_atom_contracts(pd.DataFrame(rows))
    reward_sha = reward_reference_sha256(atoms)
    roots = tuple(support_key(hole) for hole in range(18))
    return atoms, roots, reward_sha


def run_128_state_smoke(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Smoke output directory is not fresh: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    repository_root = Path(__file__).resolve().parents[1]
    source_paths = {
        "adapter": Path(__file__).resolve(),
        "adapter_test": repository_root / "test_hole_sim_transition_mean_chain_adapter.py",
        "anchor": Path(anchor.__file__).resolve(),
    }

    def file_sha256(path: Path) -> str:
        digest = sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    source_start = {name: file_sha256(path) for name, path in source_paths.items()}
    atoms, roots, reward_sha = build_synthetic_128_state_atoms()
    donor_sha = str(atoms["donor_source_sha256"].iloc[0])
    target_event_key = str(atoms["target_event_key"].iloc[0])
    target_event_start = atoms["target_event_start"].iloc[0]
    crn = anchor.CommonRandomContext(
        seed=128,
        simulations=128,
        semantic_key="synthetic128|18_hole_exact_support_lattice",
        payload_sha256=_stable_sha({"roots": roots, "states": 128}),
    )
    adapter = PhysicalChainExpectationAdapter(
        atoms,
        start_support_keys=roots,
        crn=crn,
        expected_reward_reference_sha256=reward_sha,
        expected_donor_source_sha256=donor_sha,
        expected_target_event_key=target_event_key,
        expected_target_event_start=target_event_start,
    )
    ability = (0.42, 0.18, -0.11, 0.07)
    hidden_lambda = (0.16, -0.21, 0.19, 0.12)
    template = anchor.MeanAnchorProblem(
        target_category_sg=(0.0, 0.0, 0.0, 0.0),
        target_ability_sg=ability,
        reward_reference_sha256=reward_sha,
        support_sha256=adapter.support_sha256,
        context_key="synthetic128|target_builder",
    )
    expected = adapter(template.parameters(hidden_lambda), crn)
    repeated = adapter(template.parameters(hidden_lambda), crn)
    if asdict(expected) != asdict(repeated):
        raise AssertionError("128-state exact callback is not deterministic")
    problem = anchor.MeanAnchorProblem(
        target_category_sg=expected.mean_category_sg,
        target_ability_sg=ability,
        reward_reference_sha256=reward_sha,
        support_sha256=adapter.support_sha256,
        context_key="synthetic128|solver_smoke",
        production_total_sg=float(sum(expected.mean_category_sg)),
        category_tolerance_sg=2e-9,
        finite_difference_step=0.02,
    )
    solved = anchor.solve_production_category_mean_anchor(problem, adapter, crn)
    if not solved.converged:
        raise AssertionError(f"128-state mean-anchor smoke failed: {solved.status}")
    synthetic_target = seal_production_target_hashes(
        pd.DataFrame(
            [
                {
                    "event_key": target_event_key,
                    "player_id": "synthetic-player-001",
                    "round_no": 1,
                    "target_ott_sg": expected.mean_category_sg[0],
                    "target_app_sg": expected.mean_category_sg[1],
                    "target_arg_sg": expected.mean_category_sg[2],
                    "target_putt_sg": expected.mean_category_sg[3],
                    "target_total_sg": float(sum(expected.mean_category_sg)),
                }
            ]
        )
    )
    validate_production_target_frame(synthetic_target)
    gaps = {
        name: current_support_readiness(
            name, category="app" if "Support" in name else "arg"
        ).as_dict()
        for name in CURRENT_SUPPORT_FIELDS
    }

    def input_hashes(atom_frame: pd.DataFrame) -> dict[str, str]:
        return {
            "canonical_complete_atom_frame": _records_sha(
                atom_frame, SUPPORT_HASH_COLUMNS, ATOM_SCHEMA_CONTRACT
            ),
            "canonical_physical_atom_set": _stable_sha(
                {
                    "contract": PHYSICAL_ATOM_CONTRACT,
                    "physical_atom_sha256": sorted(
                        atom_frame["physical_atom_sha256"]
                    ),
                }
            ),
            "canonical_full_reward_reference": reward_reference_sha256(atom_frame),
            "canonical_strict_prior_donor_source": donor_source_sha256(atom_frame),
            "canonical_production_player_round": str(
                synthetic_target["production_input_sha256"].iloc[0]
            ),
            "hole_start_supports": _stable_sha(
                {"contract": "synthetic_hole_start_supports_v1", "roots": roots}
            ),
            "crn_payload": crn.payload_sha256,
        }

    input_start = input_hashes(atoms)
    atoms_path = output_dir / "synthetic_complete_atoms.parquet"
    reference_path = output_dir / "synthetic_reward_reference.json"
    donor_path = output_dir / "synthetic_donor_source.json"
    target_path = output_dir / "synthetic_production_target.csv"
    result_path = output_dir / "expectation_and_anchor_result.json"
    report_path = output_dir / "REPORT.md"
    atoms.to_parquet(
        atoms_path,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    persisted_atoms = load_complete_atom_contract(
        atoms_path,
        expected_reward_reference_sha256=reward_sha,
        expected_donor_source_sha256=donor_sha,
        expected_target_event_key=target_event_key,
        expected_target_event_start=target_event_start,
    )
    input_end = input_hashes(persisted_atoms)
    if input_start != input_end:
        raise AssertionError("Persisted/reloaded smoke inputs changed")
    reference_path.write_text(
        json.dumps(
            _reference_definition_payload(atoms),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    donor_path.write_text(
        json.dumps(
            _donor_source_payload(atoms),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    synthetic_target.to_csv(target_path, index=False, lineterminator="\n")
    result_payload = {
        "expected": asdict(expected),
        "repeated_exact": asdict(repeated) == asdict(expected),
        "parquet_contract_reload_validated": True,
        "solved": solved.as_dict(),
    }
    result_path.write_text(
        json.dumps(result_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "contract": EXPECTATION_CONTRACT,
        "manifest_contract": "self_attested_research_smoke_manifest_v3",
        "status": "synthetic_mechanics_only_not_production",
        "support_states": len(adapter.supports),
        "complete_atoms": len(adapter.atoms),
        "hole_starts": len(roots),
        "reward_reference_sha256": reward_sha,
        "donor_source_sha256": donor_sha,
        "support_sha256": adapter.support_sha256,
        "crn_sha256": crn.digest,
        "hidden_lambda": list(hidden_lambda),
        "recovered_lambda": list(solved.lambda_correction),
        "maximum_abs_lambda_error": float(
            np.max(np.abs(np.asarray(solved.lambda_correction) - hidden_lambda))
        ),
        "maximum_abs_category_residual": float(
            np.max(np.abs(solved.residual_category_sg))
        ),
        "expectation": {
            "mean_category_sg": list(expected.mean_category_sg),
            "mean_opportunities": list(expected.mean_opportunities),
            "diagnostics": dict(expected.diagnostics),
        },
        "current_support_readiness": gaps,
        "production_target_policy": (
            "input_validator_only; synthetic regression row only; no live production target created"
        ),
        "input_start": input_start,
        "input_end": input_end,
        "input_start_equals_end": True,
    }
    report = [
        "# Complete-atom category SG adapter smoke",
        "",
        "Status: synthetic mechanics only; not a production target or decision simulation.",
        "",
        f"- Exact unrolled support states: {manifest['support_states']}",
        f"- Complete physical atoms: {manifest['complete_atoms']}",
        f"- Explicit APP holeout atoms: {expected.diagnostics['explicit_app_holeout_atoms']}",
        f"- Maximum declared-vs-atom APP base-holeout residual: {expected.diagnostics['maximum_app_base_holeout_mass_residual']:.3g}",
        f"- Maximum atom telescoping residual: {expected.diagnostics['max_atom_telescoping_residual']:.3g}",
        f"- Round telescoping residual: {expected.diagnostics['round_telescoping_residual']:.3g}",
        f"- Stroke identity residual: {expected.diagnostics['stroke_identity_residual']:.3g}",
        f"- Maximum recovered-lambda error: {manifest['maximum_abs_lambda_error']:.3g}",
        f"- Maximum category-mean residual: {manifest['maximum_abs_category_residual']:.3g}",
        f"- Canonical reward-reference digest: `{reward_sha}`",
        f"- Canonical donor-source digest: `{donor_sha}`",
        "",
        "## Persistence contract",
        "",
        "`synthetic_complete_atoms.parquet` is the authoritative exact serialization and ",
        "was reloaded and fully revalidated through `load_complete_atom_contract`. Generic ",
        "CSV loading fails closed; its compatibility path requires explicit opt-in and uses ",
        "`float_precision='round_trip'` with a warning.",
        "",
        "## Current-cache gate",
        "",
        "All three existing support contracts fail closed. TEE/APP numeric caches omit ",
        "fixed-reference reward and strictly-prior donor-ability provenance; APP holeout ",
        "mass is separate from the selected atom; ARG/PUTT pools discard donor identity ",
        "and do not bind `sg` to the frozen reward-reference hash.",
        "",
        "The production category mean appears only as a validated external input schema. ",
        "This smoke does not create or infer a live player target.",
    ]
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    artifact_paths = {
        path.name: path
        for path in (
            atoms_path,
            reference_path,
            donor_path,
            target_path,
            result_path,
            report_path,
        )
    }
    artifact_start = {
        name: file_sha256(path) for name, path in sorted(artifact_paths.items())
    }
    source_end = {name: file_sha256(path) for name, path in source_paths.items()}
    if source_start != source_end:
        raise AssertionError("Adapter/test/anchor source changed during smoke generation")
    manifest["source_sha256_start"] = source_start
    manifest["source_sha256_end"] = source_end
    manifest["source_start_equals_end"] = True
    artifact_end = {
        name: file_sha256(path) for name, path in sorted(artifact_paths.items())
    }
    if artifact_start != artifact_end:
        raise AssertionError("Smoke artifacts changed between start/end attestations")
    manifest["artifact_start"] = artifact_start
    manifest["artifact_end"] = artifact_end
    manifest["artifact_start_equals_end"] = True
    manifest["manifest_self_hash_policy"] = (
        "manifest_payload_sha256 hashes this object with that field omitted; "
        "manifest.sha256 is the detached checksum of serialized manifest.json"
    )
    manifest["detached_manifest_self_check_passed"] = True
    manifest["manifest_payload_sha256"] = _stable_sha(manifest)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    persisted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    persisted_payload_sha = str(persisted_manifest.pop("manifest_payload_sha256"))
    if _stable_sha(persisted_manifest) != persisted_payload_sha:
        raise AssertionError("Serialized manifest payload self-check failed")
    manifest_file_sha = file_sha256(manifest_path)
    detached_path = output_dir / "manifest.sha256"
    detached_path.write_text(
        f"{manifest_file_sha}  manifest.json\n", encoding="ascii"
    )
    if detached_path.read_text(encoding="ascii") != (
        f"{file_sha256(manifest_path)}  manifest.json\n"
    ):
        raise AssertionError("Detached manifest checksum self-check failed")
    return manifest_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "permanent_data"
            / "hole_sim_transition_mean_chain_adapter_smoke_v3"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    path = run_128_state_smoke(args.output_dir)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
