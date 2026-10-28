"""Research-only tournament-market evaluation for a frozen base-skill finalist.

This module replaces only the legacy neutral core in the audited retrospective
R1 prediction bridge.  It deliberately reuses the v4 category-first tournament
simulation, physical-course shape, shot-variance overlay, cut rules, archived
odds, settlement, and event-cluster scoring code.  Closing odds are evaluation
data only and are never visible to the finalist loader or simulator.

The replacement is exact on every covered player-event row::

    candidate_full = baseline_r1_prediction
                     + finalist_core
                     - legacy_core_pred_step2

The resulting total delta is spread equally across OTT/APP/ARG/PUTT.  This is
the production category re-centering convention; the prior shot-mean study's
APP-only mean tilt is intentionally not used here.

The command has a guarded compute ladder:

* ``preflight``: no simulation (an identity bridge may be used before a
  finalist is frozen);
* ``one-event``: exactly one event at 2,000 draws;
* ``screen-5k``: all eligible events at 5,000 draws;
* ``final-50k``: all eligible events at 50,000 draws.

No production file or consumer is written by this script.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import json
import platform
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


REPO = Path(__file__).resolve().parents[1]
ARCHIVE = Path(__file__).resolve().parent
SIM_PREP = Path(r"C:\Users\McKinley Slade\dev\sim_prep")
EXPORT = Path(r"C:\Users\McKinley Slade\dev\golf_scraping\output\export")
RESEARCH = (
    REPO
    / "permanent_data"
    / "research"
    / "base_skill_rebuild_20200101_20260822"
)
V4 = (
    REPO
    / "permanent_data"
    / "research"
    / "shot_mean_closing_backtest_20230822_20260822_v4"
)
EXPECTED_V4_MANIFEST_SHA256 = (
    "a8bce381f07f085d511a0db2e6ced2ba481c67e37e6183c438403632c12e1c50"
)
EXPECTED_V4_MANIFEST_BYTES = 7_926
EXPECTED_BRIDGE_MANIFEST_SHA256 = (
    "8020ef51cc83efdaa89061aedeeb19b9c264a06a260c16c9ad645c311818b4dc"
)
EXPECTED_BRIDGE_MANIFEST_BYTES = 2_597
EXPECTED_BRIDGE_OUTPUT_SHA256 = {
    "legacy_core_bridge.parquet": (
        "1820586323c8fcbb66e696e8400c2adf628be6a2e758b81987eb524247dee121"
    ),
    "local_event_identity.parquet": (
        "f7ba5ba99ebb04717786f8b23f19b535d4c782ea260672b355eefad349ac7fa4"
    ),
    "odds_event_crosswalk.parquet": (
        "bb3a930447d1f3ca91aa10f45cc2b276315e2932df8a7fd3813973aac13c6659"
    ),
}
DEFAULT_DB = SIM_PREP / "work" / "dg_historical.db"
DEFAULT_PREDS = SIM_PREP / "work" / "sg_preds_resids.csv"
DEFAULT_SOURCE_CROSSWALK = SIM_PREP / "shot_bridge" / "crosswalk.parquet"
DEFAULT_ODDS_CACHE = (
    REPO
    / "permanent_data"
    / "research"
    / "dg_closing_cache_20230822_20260822"
)
DEFAULT_BRIDGE = RESEARCH / "core_bridge" / "legacy_core_bridge.parquet"
DEFAULT_BRIDGE_MANIFEST = RESEARCH / "core_bridge" / "manifest.json"
DEFAULT_EVENT_CROSSWALK = RESEARCH / "core_bridge" / "odds_event_crosswalk.parquet"
DEFAULT_FINALIST_MANIFEST = (
    RESEARCH
    / "model_completion_v3_provenance"
    / "frozen_finalist"
    / "manifest.json"
)
DEFAULT_ROUND_PANEL = RESEARCH / "player_round_panel.parquet"
DEFAULT_OUTPUT = RESEARCH / "finish_market_eval_v3_provenance"
DEFAULT_RUNTIME_LOCK = RESEARCH / "runtime_locks" / "finish_market_runtime_v1.json"

for path in (str(REPO), str(ARCHIVE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import backtest_shot_mean_closing_markets as v4bt  # noqa: E402
from base_skill_frozen_handoff import validate_frozen_handoff_chain  # noqa: E402
from archive import evaluate_complete_base_skill_market_candidates as discovery  # noqa: E402
from backtest_distribution_engines_2026 import (  # noqa: E402
    build_category_params,
    load_correlation,
)
from build_bmw_shot_features_shadow import (  # noqa: E402
    build_dispersion_features,
    load_event_dates,
    load_shot_round_stats,
)
from historical_odds_cache import HistoricalOddsCacheReader  # noqa: E402
from market_backtest_scoring import (  # noqa: E402
    GateConfig,
    SizingConfig,
    american_implied_probability,
    compare_arm_bets,
    event_cluster_bootstrap,
    performance_summary,
    select_arm_bets,
    settle_bets,
    summarize_attribution,
    ties_lose,
)
import research_execution_provenance as execution_provenance  # noqa: E402


CONTRACT = "base_skill_frozen_finalist_finish_market_v1"
FINALIST_CONTRACT = "base_skill_frozen_finalist_v1"
RUNTIME_LOCK_CONTRACT = "base_skill_finish_research_runtime_lock_v1"
SIMS_KERNEL_NATIVE_MODULE = "sims_kernel.sims_kernel"
SIMS_KERNEL_NATIVE_SOURCE_LABEL = (
    "native_extension::sims_kernel.sims_kernel"
)
START = "2023-08-22"
END = "2026-08-22"
DISCOVERY_END = pd.Timestamp("2022-12-31")
STAGE_SIMULATIONS = {
    "preflight": 0,
    "one-event": 2_000,
    "screen-5k": 5_000,
    "final-50k": 50_000,
}
PRICE_POINTS = ("open", "close")
SHARP_BOOKS = ("betonline", "betcris", "pinnacle")
LINE_POLICIES = (*SHARP_BOOKS, "best_of_three")
OUTRIGHT_MARKET_CUTOFF = {"win": 1, "top_5": 5, "top_10": 10, "top_20": 20}
TOP_FINISH_MARKETS = frozenset(("top_5", "top_10", "top_20"))
STANDARD_DEAD_HEAT_PROBABILITY = "standard_dead_heat_probability"
FULL_TIE_QUALIFICATION_PROBABILITY = "full_tie_qualification_probability"
KNOWN_NONFINISH_RESULTS = frozenset({"CUT", "DQ", "WD"})
# The exact frozen odds-cache/DB/crosswalk authorities contain 50 fractional
# Pinnacle top-finish settlements that conflict with Pinnacle's full-ties rule.
# Thirty-seven of those also conflict with universal standard dead-heat math.
# Both identity inventories are checked before any simulation or P&L can start.
EXPECTED_PINNACLE_FULL_PAYOUT_OVERRIDES = 50
EXPECTED_PINNACLE_FULL_PAYOUT_RECORD_ID_SHA256 = (
    "d4c7158a7f5ca1fb97513288d3bfee960ce3806700b015a85f2c5ee6139e8e3f"
)
EXPECTED_STANDARD_DEAD_HEAT_DISCREPANCIES = 37
EXPECTED_STANDARD_DEAD_HEAT_RECORD_ID_SHA256 = (
    "22e7f2908464e9e4281e2131f03e934f546b2ada862b93b974c95dcf3021014d"
)
PINNACLE_TOP_FINISH_RULE_URL = "https://www.pinnacle.com/en/future/betting-rules"
PINNACLE_TOP_FINISH_RULE = (
    "Top Finish bets are paid at full odds when the player ties for that finish, "
    "regardless of how many players are tied."
)
SETTLEMENT_SCENARIO_COLUMNS = {
    "book_rule_primary": "book_rule_bet_outcome_numeric",
    "raw_archive_sensitivity": "archived_bet_outcome_numeric",
    "universal_standard_dead_heat_sensitivity": (
        "standard_dead_heat_bet_outcome_numeric"
    ),
}
PROBABILITY_KEYS = ["event_id", "market", "selection_id", "opponent_id"]
FINALIST_REQUIRED_COLUMNS = {
    "event_key",
    "dg_id",
    "event_start",
    "candidate",
    "finalist_core",
    "control_core",
    "split",
    "prediction_asof",
    "chronology_ok",
}
ATTRIBUTION_COLUMNS = (
    "history_rounds",
    "days_since_last_round",
    "reliability_ok",
    "cross_tour_fallback_reason",
    "img_history_rounds",
    "img_history_coverage",
    "prior_event_withdrawal",
)
ODDS_EVENT_KEY = re.compile(r"^pga:R\d{7}$")
HANDOFF_PROVENANCE_SOURCE_CLOSURE = tuple(
    dict.fromkeys(
        (
            ARCHIVE / "base_skill_frozen_handoff.py",
            Path(discovery.__file__).resolve(),
            *(
                Path(path).resolve()
                for path in discovery.DISCOVERY_PROVENANCE_SOURCE_CLOSURE
            ),
        )
    )
)


def _runtime_file_fingerprint(path: Path, *, source_label: str) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"runtime file is missing: {resolved}")
    return {
        "source_label": source_label,
        "path": str(resolved),
        "sha256": v4bt.sha256_file(resolved),
        "bytes": int(resolved.stat().st_size),
    }


def _compiled_extension_suffix(path: Path) -> str:
    """Return the importable native suffix or reject a Python/source artifact."""

    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"sims_kernel native extension is missing: {resolved}")
    lowered = str(resolved).casefold()
    for suffix in sorted(importlib.machinery.EXTENSION_SUFFIXES, key=len, reverse=True):
        if lowered.endswith(suffix.casefold()):
            return suffix
    raise ValueError(
        "sims_kernel native path is not a compiled extension: "
        f"{resolved}; accepted suffixes={importlib.machinery.EXTENSION_SUFFIXES}"
    )


def active_research_runtime_lock_payload() -> dict[str, Any]:
    """Fingerprint the exact interpreter and native finish-simulation runtime."""

    pyarrow = importlib.import_module("pyarrow")
    scipy = importlib.import_module("scipy")
    sklearn = importlib.import_module("sklearn")
    kernel_package = importlib.import_module("sims_kernel")
    kernel_native = importlib.import_module(SIMS_KERNEL_NATIVE_MODULE)

    package_file = getattr(kernel_package, "__file__", None)
    native_file = getattr(kernel_native, "__file__", None)
    if not package_file:
        raise ValueError("sims_kernel package has no resolved __file__")
    if not native_file:
        raise ValueError(f"{SIMS_KERNEL_NATIVE_MODULE} has no resolved __file__")
    package_path = Path(str(package_file)).resolve()
    native_path = Path(str(native_file)).resolve()
    native_suffix = _compiled_extension_suffix(native_path)

    spec = getattr(kernel_native, "__spec__", None)
    if spec is None or not isinstance(
        getattr(spec, "loader", None), importlib.machinery.ExtensionFileLoader
    ):
        raise ValueError(
            f"{SIMS_KERNEL_NATIVE_MODULE} was not loaded by ExtensionFileLoader"
        )
    spec_origin = getattr(spec, "origin", None)
    if not spec_origin or Path(str(spec_origin)).resolve() != native_path:
        raise ValueError("sims_kernel native module spec origin disagrees with __file__")

    package_run = getattr(kernel_package, "run_pretournament", None)
    native_run = getattr(kernel_native, "run_pretournament", None)
    if package_run is None or native_run is None or package_run is not native_run:
        raise ValueError(
            "sims_kernel.run_pretournament is not bound to the resolved native extension"
        )
    if getattr(package_run, "__module__", None) != SIMS_KERNEL_NATIVE_MODULE:
        raise ValueError("sims_kernel.run_pretournament has an unexpected source module")

    api_version = str(kernel_package.version()).strip()
    distribution_version = str(importlib.metadata.version("sims-kernel")).strip()
    if not api_version or not distribution_version:
        raise ValueError("sims_kernel runtime version is empty")

    return {
        "contract": RUNTIME_LOCK_CONTRACT,
        "research_only": True,
        "runtime_role": "presealed_finish_market_simulation_authority",
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "cache_tag": str(sys.implementation.cache_tag),
            "architecture": platform.architecture()[0],
            "machine": platform.machine(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "packages": {
            "numpy": str(np.__version__),
            "pandas": str(pd.__version__),
            "pyarrow": str(pyarrow.__version__),
            "scipy": str(scipy.__version__),
            "sklearn": str(sklearn.__version__),
            "sims_kernel_api": api_version,
            "sims_kernel_distribution": distribution_version,
        },
        "sims_kernel": {
            "package_init": _runtime_file_fingerprint(
                package_path, source_label="python_package_init::sims_kernel"
            ),
            "native_extension": {
                **_runtime_file_fingerprint(
                    native_path, source_label=SIMS_KERNEL_NATIVE_SOURCE_LABEL
                ),
                "module": SIMS_KERNEL_NATIVE_MODULE,
                "extension_suffix": native_suffix,
            },
            "api_distribution_version_match": api_version == distribution_version,
            "run_pretournament_bound_to_native": True,
        },
    }


def validate_research_runtime_lock(lock_path: Path) -> dict[str, Any]:
    """Require a byte-bound lock that exactly describes the active runtime."""

    resolved = lock_path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"research runtime lock is missing: {resolved}")
    try:
        locked = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"research runtime lock is unreadable: {resolved}") from exc
    if not isinstance(locked, dict):
        raise ValueError("research runtime lock root must be an object")
    active = active_research_runtime_lock_payload()
    if locked != active:
        locked_hash = hashlib.sha256(
            json.dumps(locked, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        active_hash = hashlib.sha256(
            json.dumps(active, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        raise ValueError(
            "research runtime lock does not exactly match the active runtime "
            f"(locked_payload_sha256={locked_hash}, active_payload_sha256={active_hash})"
        )
    return {
        "lock": _runtime_file_fingerprint(
            resolved, source_label="research_runtime_lock::finish_market_runtime_v1"
        ),
        "active_runtime": active,
    }


def runtime_provenance_source_paths(runtime_audit: Mapping[str, Any]) -> tuple[Path, Path]:
    runtime = runtime_audit.get("active_runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("runtime audit lacks active_runtime")
    kernel = runtime.get("sims_kernel")
    if not isinstance(kernel, Mapping):
        raise ValueError("runtime audit lacks sims_kernel")
    package = kernel.get("package_init")
    native = kernel.get("native_extension")
    if not isinstance(package, Mapping) or not isinstance(native, Mapping):
        raise ValueError("runtime audit lacks exact sims_kernel package/native files")
    package_path = Path(str(package.get("path", ""))).resolve()
    native_path = Path(str(native.get("path", ""))).resolve()
    _compiled_extension_suffix(native_path)
    return package_path, native_path


def assert_research_runtime_current(runtime_audit: Mapping[str, Any]) -> None:
    """Fail if the interpreter, dependencies, binding, or lock changed mid-run."""

    expected = runtime_audit.get("active_runtime")
    if not isinstance(expected, Mapping) or active_research_runtime_lock_payload() != expected:
        raise RuntimeError("active research runtime changed during finish evaluation")
    lock = runtime_audit.get("lock")
    if not isinstance(lock, Mapping):
        raise RuntimeError("runtime audit lacks its lock fingerprint")
    lock_path = Path(str(lock.get("path", ""))).resolve()
    if (
        not lock_path.is_file()
        or v4bt.sha256_file(lock_path) != str(lock.get("sha256", ""))
        or lock_path.stat().st_size != int(lock.get("bytes", -1))
    ):
        raise RuntimeError("research runtime lock changed during finish evaluation")


def _serialized_cli_settings(args: argparse.Namespace) -> list[str]:
    payload = {
        str(key): str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    return [
        "python",
        str(Path(__file__).resolve()),
        "--parsed-args-json",
        json.dumps(payload, sort_keys=True, default=str),
    ]


def _odds_cache_provenance_paths(cache_dir: Path) -> list[Path]:
    cache_dir = cache_dir.resolve()
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise ValueError("odds cache provenance manifest is incomplete")
    paths = [manifest_path]
    for entry in manifest.get("active_shards", []):
        if not isinstance(entry, Mapping) or not entry.get("data_path"):
            raise ValueError("odds cache has malformed active shard provenance")
        path = (cache_dir / str(entry["data_path"])).resolve()
        try:
            path.relative_to(cache_dir)
        except ValueError as exc:
            raise ValueError("odds cache shard escapes cache directory") from exc
        if not path.is_file() or v4bt.sha256_file(path) != str(
            entry.get("sha256", "")
        ).lower():
            raise ValueError(f"odds cache shard hash mismatch: {path}")
        paths.append(path)
    return paths


def _require_locked_hash(
    input_hashes: Mapping[str, Any], key: str, path: Path
) -> dict[str, str]:
    resolved = path.resolve()
    expected = str(input_hashes.get(key, "")).lower()
    if len(expected) != 64:
        raise ValueError(f"V4 run config lacks locked input hash {key!r}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    actual = v4bt.sha256_file(resolved)
    if actual != expected:
        raise ValueError(
            f"V4-authoritative input hash mismatch for {key}: "
            f"expected {expected}, got {actual}"
        )
    return {"path": str(resolved), "sha256": actual}


def validate_v4_authoritative_inputs(
    *,
    v4_dir: Path,
    db: Path,
    predictions: Path,
    source_crosswalk: Path,
    correlation: Path,
    coefficients: Path,
    odds_cache: Path,
    shot_export: Path,
) -> dict[str, Any]:
    """Bind finish simulation inputs to the exact audited V4 authority."""

    v4_dir = v4_dir.resolve()
    manifest_path = v4_dir / "manifest.json"
    if manifest_path.stat().st_size != EXPECTED_V4_MANIFEST_BYTES:
        raise ValueError("V4 authority manifest byte size differs from the frozen audit")
    manifest_hash = v4bt.sha256_file(manifest_path)
    if manifest_hash != EXPECTED_V4_MANIFEST_SHA256:
        raise ValueError("V4 authority manifest hash differs from the frozen audit")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("contract") != "shot_mean_closing_market_backtest_v2"
        or manifest.get("completed") is not True
    ):
        raise ValueError("V4 authority manifest contract is incomplete or unexpected")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("V4 authority manifest lacks outputs")

    bound_outputs: dict[str, dict[str, str]] = {}
    for name in ("run_config.json", "event_audit.parquet"):
        path = (v4_dir / name).resolve()
        entry = outputs.get(name)
        if not isinstance(entry, Mapping) or not entry.get("sha256"):
            raise ValueError(f"V4 authority manifest lacks output {name!r}")
        if path.parent != v4_dir or v4bt.sha256_file(path) != str(
            entry["sha256"]
        ).lower():
            raise ValueError(f"V4 authority output hash mismatch: {name}")
        if entry.get("bytes") is not None and path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"V4 authority output byte size mismatch: {name}")
        bound_outputs[name] = {
            "path": str(path),
            "sha256": v4bt.sha256_file(path),
        }

    run_config = json.loads(
        (v4_dir / "run_config.json").read_text(encoding="utf-8")
    )
    if run_config.get("contract") != manifest["contract"]:
        raise ValueError("V4 run config contract disagrees with its manifest")
    input_hashes = run_config.get("input_hashes")
    if not isinstance(input_hashes, Mapping):
        raise ValueError("V4 run config lacks locked input hashes")
    bindings = {
        "historical_database": db,
        "predictions": predictions,
        "crosswalk": source_crosswalk,
        "correlation": correlation,
        "coefficients": coefficients,
        "odds_cache_manifest": odds_cache.resolve() / "manifest.json",
        "shot_export::manifest": shot_export.resolve() / "manifest.json",
        **{
            f"shot_export::shot_sg_{year}": (
                shot_export.resolve() / f"pga_{year}_shot_sg.parquet"
            )
            for year in range(2020, 2027)
        },
    }
    verified_inputs = {
        key: _require_locked_hash(input_hashes, key, path)
        for key, path in bindings.items()
    }
    return {
        "manifest": {"path": str(manifest_path), "sha256": manifest_hash},
        "outputs": bound_outputs,
        "inputs": verified_inputs,
        "contract": manifest["contract"],
        "complete": True,
    }


def _v4_authority_path_set(audit: Mapping[str, Any]) -> frozenset[Path]:
    records = [audit.get("manifest")]
    for section in ("outputs", "inputs"):
        values = audit.get(section)
        if not isinstance(values, Mapping):
            raise ValueError(f"V4 authority audit lacks {section}")
        records.extend(values.values())
    if any(not isinstance(record, Mapping) or not record.get("path") for record in records):
        raise ValueError("V4 authority audit contains a malformed path record")
    return frozenset(Path(str(record["path"])).resolve() for record in records)


def revalidate_v4_authority_after_begin(
    *,
    expected_audit: Mapping[str, Any],
    provenance_state: execution_provenance.ExecutionProvenance,
    v4_dir: Path,
    db: Path,
    predictions: Path,
    source_crosswalk: Path,
    correlation: Path,
    coefficients: Path,
    odds_cache: Path,
    shot_export: Path,
) -> dict[str, Any]:
    """Close the validation-to-provenance gap for every V4-bound input."""

    expected_paths = _v4_authority_path_set(expected_audit)
    tracked = frozenset(
        Path(str(record["path"])).resolve()
        for record in provenance_state.input_fingerprints
    )
    if not expected_paths.issubset(tracked):
        raise ValueError("V4 authority paths were not all tracked before consumption")
    current = validate_v4_authoritative_inputs(
        v4_dir=v4_dir,
        db=db,
        predictions=predictions,
        source_crosswalk=source_crosswalk,
        correlation=correlation,
        coefficients=coefficients,
        odds_cache=odds_cache,
        shot_export=shot_export,
    )
    if _v4_authority_path_set(current) != expected_paths:
        raise ValueError("V4 authority resolved path set changed across provenance begin")
    if current != dict(expected_audit):
        raise ValueError("V4 authority hashes changed across provenance begin")
    return current


def _resolved_path_set(paths: Iterable[Path]) -> frozenset[Path]:
    return frozenset(Path(path).resolve() for path in paths)


def revalidate_finish_dynamic_inputs_after_begin(
    *,
    finalist_manifest_path: Path,
    identity_preflight: bool,
    expected_handoff_paths: Iterable[Path],
    odds_cache: Path,
    expected_odds_paths: Iterable[Path],
    provenance_state: execution_provenance.ExecutionProvenance,
) -> tuple[tuple[dict[str, Any], Path] | None, list[Path]]:
    """Re-enumerate mutable manifests after provenance begins, before reads."""

    tracked = frozenset(
        Path(str(record["path"])).resolve()
        for record in provenance_state.input_fingerprints
    )
    expected_odds = _resolved_path_set(expected_odds_paths)
    if not expected_odds.issubset(tracked):
        raise ValueError("pre-begin odds-cache paths were not all provenance-tracked")
    current_odds_paths = _odds_cache_provenance_paths(odds_cache)
    if _resolved_path_set(current_odds_paths) != expected_odds:
        raise ValueError("odds-cache resolved path set changed across provenance begin")

    if identity_preflight:
        if _resolved_path_set(expected_handoff_paths):
            raise ValueError("identity preflight unexpectedly declared frozen handoff paths")
        return None, current_odds_paths

    expected_handoff = _resolved_path_set(expected_handoff_paths)
    if not expected_handoff or not expected_handoff.issubset(tracked):
        raise ValueError("pre-begin frozen handoff paths were not exactly provenance-tracked")
    current_finalist = validate_finalist_manifest(finalist_manifest_path)
    current_handoff = _resolved_path_set(
        Path(value)
        for value in current_finalist[0].get("_verified_handoff_paths", [])
    )
    if current_handoff != expected_handoff:
        raise ValueError("frozen handoff resolved path set changed across provenance begin")
    return current_finalist, current_odds_paths


def begin_finish_execution_provenance(
    *,
    output_dir: Path,
    input_paths: Iterable[Path],
    args: argparse.Namespace,
    runtime_audit: Mapping[str, Any],
    extra_code_paths: Iterable[Path] = (),
) -> execution_provenance.ExecutionProvenance:
    assert_research_runtime_current(runtime_audit)
    runtime_code_paths = runtime_provenance_source_paths(runtime_audit)
    runtime_lock = runtime_audit.get("lock")
    if not isinstance(runtime_lock, Mapping) or not runtime_lock.get("path"):
        raise ValueError("runtime audit lacks its exact lock path")
    code_paths = [
        *HANDOFF_PROVENANCE_SOURCE_CLOSURE,
        Path(v4bt.__file__).resolve(),
        ARCHIVE / "market_backtest_scoring.py",
        ARCHIVE / "historical_odds_cache.py",
        ARCHIVE / "backtest_distribution_engines_2026.py",
        ARCHIVE / "build_bmw_shot_features_shadow.py",
        ARCHIVE / "backtest_shot_dispersion_recent.py",
        Path(execution_provenance.__file__).resolve(),
        REPO / "sim_inputs.py",
        *runtime_code_paths,
        *[Path(path).resolve() for path in extra_code_paths],
    ]
    return execution_provenance.begin(
        output_dir=output_dir,
        primary_source=Path(__file__).resolve(),
        input_paths=[
            *[Path(path).resolve() for path in input_paths],
            Path(str(runtime_lock["path"])).resolve(),
        ],
        code_paths=code_paths,
        cli=_serialized_cli_settings(args),
    )


def finalize_finish_execution_provenance(
    state: execution_provenance.ExecutionProvenance,
) -> dict[str, Any]:
    return execution_provenance.finalize(state)


def write_finish_complete_manifest(
    path: Path,
    payload: Mapping[str, Any],
    provenance_state: execution_provenance.ExecutionProvenance,
) -> None:
    """Write a terminal finish/preflight manifest only on current inputs."""

    execution_provenance.assert_current(provenance_state)
    v4bt.atomic_json(path, payload)


def require_tracked_resume_checkpoint(
    provenance_state: execution_provenance.ExecutionProvenance,
    *,
    run_config_path: Path,
    checkpoint_path: Path,
    checkpoint_meta_path: Path,
) -> None:
    """Reject checkpoints created after provenance input enumeration."""

    required = {
        run_config_path.resolve(),
        checkpoint_path.resolve(),
        checkpoint_meta_path.resolve(),
    }
    tracked = {
        Path(str(record["path"])).resolve()
        for record in provenance_state.input_fingerprints
    }
    missing = required - tracked
    if missing:
        raise ValueError(
            "resume checkpoint/run_config was not tracked at provenance begin: "
            + ", ".join(sorted(str(path) for path in missing))
        )
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("a provenance-tracked resume artifact is missing")


def finish_event_format_exclusion(meta: v4bt.EventMeta) -> str:
    """Fail closed on tournament formats the four-round stroke kernel cannot price."""

    name = v4bt.normalize_text(meta.event_name)
    inherited = v4bt.event_format_exclusion(meta)
    if inherited:
        return inherited
    if "zurich classic" in name:
        return "team_format_unmodeled"
    # The 2024 Pebble Beach event was declared complete after 54 holes.  A
    # four-round simulation would grade a tournament that was never played.
    if meta.year == 2024 and "pebble beach pro am" in name:
        return "known_54_hole_weather_shortening_unmodeled"
    return ""


def cut_line_for_event(meta: v4bt.EventMeta, field_size: int) -> tuple[int, str]:
    """Event-year-safe cut contract for the supported 72-hole stroke events.

    Event names alone are not enough: several formerly full-field tournaments
    became no-cut signature events in 2024.  Field size and event year are
    therefore part of the rule, and ambiguous non-standard formats are
    excluded before this function is called.
    """

    if field_size <= 0:
        raise ValueError("field_size must be positive")
    exclusion = finish_event_format_exclusion(meta)
    if exclusion:
        raise ValueError(f"unsupported event format for cut rule: {exclusion}")
    name = v4bt.normalize_text(meta.event_name)
    if "masters" in name:
        return min(50, field_size), "masters_top50"
    if "pga championship" in name:
        return min(70, field_size), "pga_championship_top70"
    if "u.s. open" in name or "u s open" in name or "us open" in name:
        return min(60, field_size), "us_open_top60"
    if "open championship" in name:
        return min(70, field_size), "open_championship_top70"
    invitational = any(
        token in name
        for token in (
            "genesis invitational",
            "arnold palmer",
            "memorial tournament",
        )
    )
    if meta.year >= 2024 and invitational and field_size <= 90:
        return min(50, field_size), "signature_invitational_top50"
    if field_size <= 80:
        return field_size, "limited_field_no_cut"
    return min(65, field_size), "standard_top65"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=tuple(STAGE_SIMULATIONS), default="preflight")
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDS)
    parser.add_argument("--source-crosswalk", type=Path, default=DEFAULT_SOURCE_CROSSWALK)
    parser.add_argument("--odds-cache", type=Path, default=DEFAULT_ODDS_CACHE)
    parser.add_argument("--shot-export", type=Path, default=EXPORT)
    parser.add_argument("--bridge", type=Path, default=DEFAULT_BRIDGE)
    parser.add_argument("--bridge-manifest", type=Path, default=DEFAULT_BRIDGE_MANIFEST)
    parser.add_argument("--event-crosswalk", type=Path, default=DEFAULT_EVENT_CROSSWALK)
    parser.add_argument("--finalist-manifest", type=Path, default=DEFAULT_FINALIST_MANIFEST)
    parser.add_argument("--round-panel", type=Path, default=DEFAULT_ROUND_PANEL)
    parser.add_argument(
        "--runtime-lock",
        type=Path,
        default=DEFAULT_RUNTIME_LOCK,
        help=(
            "Research-only exact Python/dependency/sims_kernel native runtime lock; "
            "the active runtime must match it byte-for-byte semantically."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--event-key", action="append")
    parser.add_argument("--no-course-shape", action="store_true")
    parser.add_argument("--no-shot-variance", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume only an immutable run in this directory. Without this flag, "
            "a nonempty output directory is rejected."
        ),
    )
    parser.add_argument(
        "--identity-preflight",
        action="store_true",
        help="Use legacy_core as finalist_core for coverage checks; legal only at preflight.",
    )
    return parser.parse_args(argv)


def _manifest_bool(manifest: dict[str, Any], key: str) -> bool:
    value = manifest.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"Finalist manifest requires boolean {key!r}")
    return value


def _resolve_manifest_path(manifest_path: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else manifest_path.parent / path


def validate_finalist_manifest(manifest_path: Path) -> tuple[dict[str, Any], Path]:
    """Reject an unfrozen, market-unit, hash-mismatched, or sealed-selected finalist."""
    handoff = validate_frozen_handoff_chain(manifest_path)
    manifest = dict(handoff.compact)
    manifest["_resolved_selected_name"] = handoff.selected_name
    manifest["_selection_role"] = handoff.selection_role
    manifest["_promotion_gate_passed"] = handoff.promotion_gate_passed
    manifest["_verified_selection_manifest"] = handoff.selection
    manifest["_discovery_selected_downstream_lambda"] = handoff.downstream_lambda
    manifest["_provenance_chain_audit"] = handoff.audit
    manifest["_verified_completion_inputs"] = handoff.audit.get(
        "completion_input_bindings", {}
    )
    manifest["_verified_source_prediction_path"] = str(
        handoff.source_prediction_path
    )
    manifest["_verified_source_manifest_path"] = str(handoff.source_manifest_path)
    manifest["_verified_candidate_catalog_path"] = str(
        handoff.candidate_catalog_path
    )
    manifest["_verified_frozen_catalog_path"] = str(handoff.frozen_catalog_path)
    manifest["_verified_core_bridge_path"] = str(handoff.core_bridge_path)
    manifest["_verified_core_bridge_sha256"] = handoff.core_bridge_sha256
    manifest["_verified_handoff_paths"] = [str(path) for path in handoff.verified_paths]
    forbidden = ("close_odds", "closing_odds", "closing_price", "close_price")
    features = [str(value).casefold() for value in manifest.get("feature_names", [])]
    if any(any(token in feature for token in forbidden) for feature in features):
        raise ValueError("Finalist feature_names contains a closing-price feature")
    return manifest, handoff.compact_prediction_path


def validate_finish_completion_authorities(
    manifest: Mapping[str, Any], *, round_panel: Path, bridge: Path
) -> dict[str, dict[str, str]]:
    """Bind material local model inputs to the frozen completion provenance."""

    records = manifest.get("_verified_completion_inputs")
    if not isinstance(records, Mapping):
        raise ValueError("frozen handoff lacks completion input authority bindings")
    requested = {"round_panel": round_panel.resolve(), "core_bridge": bridge.resolve()}
    audit: dict[str, dict[str, str]] = {}
    for name, supplied in requested.items():
        record = records.get(name)
        if not isinstance(record, Mapping) or not record.get("path") or not record.get(
            "sha256"
        ):
            raise ValueError(f"completion authority lacks input {name!r}")
        authoritative = Path(str(record["path"])).resolve()
        if supplied != authoritative:
            raise ValueError(f"--{name.replace('_', '-')} is not the frozen completion input")
        actual = v4bt.sha256_file(supplied)
        if actual != str(record["sha256"]).lower():
            raise ValueError(f"frozen completion input hash mismatch: {name}")
        audit[name] = {"path": str(supplied), "sha256": actual}
    return audit


def selected_downstream_lambdas(
    manifest: dict[str, Any], identity_preflight: bool = False
) -> tuple[float, ...]:
    """Read a discovery-frozen downstream sensitivity set from the finalist manifest."""

    if identity_preflight:
        return (1.0,)
    frozen_choice = manifest.get("_discovery_selected_downstream_lambda")
    declared = manifest.get("downstream_lambdas")
    if frozen_choice is None:
        if declared is None:
            return (1.0,)
        if not isinstance(declared, list) or not declared:
            raise ValueError("downstream_lambdas must be a non-empty list")
        lambdas = tuple(sorted({float(value) for value in declared}, reverse=True))
    else:
        lambdas = tuple(sorted({1.0, float(frozen_choice)}, reverse=True))
        if declared is not None:
            if not isinstance(declared, list) or not declared:
                raise ValueError("downstream_lambdas must be a non-empty list")
            compact_lambdas = tuple(
                sorted({float(value) for value in declared}, reverse=True)
            )
            if compact_lambdas != lambdas:
                raise ValueError(
                    "compact downstream_lambdas disagrees with the verified discovery choice"
                )
    allowed = {0.0, 0.5, 1.0}
    if set(lambdas) - allowed:
        raise ValueError("downstream_lambdas may contain only 0, 0.5, and 1")
    if 1.0 not in lambdas:
        raise ValueError("Primary isolation lambda=1 must always be present")
    if any(value != 1.0 for value in lambdas):
        selection = manifest.get("_verified_selection_manifest", {})
        cutoff = pd.to_datetime(
            manifest.get(
                "downstream_lambda_selection_window_end",
                selection.get("selection_window_end")
                if isinstance(selection, Mapping)
                else None,
            ),
            errors="coerce",
        )
        if not pd.isna(cutoff):
            cutoff = pd.Timestamp(cutoff).normalize()
        if pd.isna(cutoff) or cutoff > DISCOVERY_END:
            raise ValueError(
                "Alternate downstream lambdas require a selection cutoff <=2022-12-31"
            )
        sealed_flag = manifest.get(
            "sealed_data_used_for_downstream_lambda_selection",
            selection.get("sealed_data_used_for_selection")
            if isinstance(selection, Mapping)
            else None,
        )
        if sealed_flag is not False:
            raise ValueError("Alternate downstream lambda was not selected without sealed data")
        frozen_flag = manifest.get(
            "downstream_lambda_frozen_before_sealed_scoring",
            bool(selection) and manifest.get("frozen_before_sealed_scoring") is True,
        )
        if frozen_flag is not True:
            raise ValueError("Alternate downstream lambda was not frozen before sealed scoring")
    return lambdas


def lambda_arm_name(value: float) -> str:
    return f"finalist_lambda_{str(float(value)).replace('.', 'p')}"


def normalize_finalist_predictions(
    frame: pd.DataFrame,
    finalist_name: str,
) -> pd.DataFrame:
    missing = sorted(FINALIST_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Finalist prediction file is missing columns: {missing}")
    output = frame[frame["candidate"].astype(str).eq(finalist_name)].copy()
    if output.empty:
        raise ValueError(f"No prediction rows for finalist {finalist_name!r}")
    output["event_key"] = output["event_key"].astype(str).str.strip()
    invalid_keys = ~output["event_key"].str.fullmatch(ODDS_EVENT_KEY)
    if invalid_keys.any():
        raise ValueError(
            "Finalist event_key must use canonical odds identity pga:RYYYYEEE; "
            f"examples={output.loc[invalid_keys, 'event_key'].head(5).tolist()}"
        )
    output["dg_id"] = pd.to_numeric(output["dg_id"], errors="coerce")
    output["finalist_core"] = pd.to_numeric(output["finalist_core"], errors="coerce")
    output["control_core"] = pd.to_numeric(output["control_core"], errors="coerce")
    output["event_start"] = pd.to_datetime(output["event_start"], errors="coerce").dt.normalize()
    output["prediction_asof"] = pd.to_datetime(
        output["prediction_asof"], errors="coerce", utc=True
    ).dt.tz_localize(None)
    chronology = output["chronology_ok"].map(
        lambda value: value if isinstance(value, (bool, np.bool_)) else str(value).casefold() == "true"
    )
    invalid = (
        output["dg_id"].isna()
        | output["finalist_core"].isna()
        | ~np.isfinite(output["finalist_core"])
        | output["control_core"].isna()
        | ~np.isfinite(output["control_core"])
        | output["event_start"].isna()
        | output["prediction_asof"].isna()
        | ~chronology
        | (output["prediction_asof"] > output["event_start"])
    )
    if invalid.any():
        raise ValueError(f"Finalist has {int(invalid.sum())} invalid/non-chronological rows")
    units = output.get("prediction_units")
    if units is not None and not units.astype(str).str.casefold().eq("sg_per_round").all():
        raise ValueError("Pairwise market-logit scores cannot enter the tournament simulator")
    control_units = output.get("control_units")
    if control_units is not None and not control_units.astype(str).str.casefold().eq(
        "sg_per_round"
    ).all():
        raise ValueError("control_core must be expressed in SG per round")
    if "training_max_event_end" in output:
        training_end = pd.to_datetime(output["training_max_event_end"], errors="coerce").dt.normalize()
        leaked = training_end.notna() & training_end.ge(output["event_start"])
        if leaked.any():
            raise ValueError(f"Finalist contains {int(leaked.sum())} training chronology leaks")
    output["dg_id"] = output["dg_id"].astype(int)
    if output.duplicated(["event_key", "dg_id"]).any():
        raise ValueError("Finalist contains duplicate event_key/dg_id predictions")
    return output.sort_values(["event_start", "event_key", "dg_id"], kind="mergesort")


def attach_verified_source_attribution(
    finalist: pd.DataFrame,
    manifest: Mapping[str, Any],
    selected_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach non-feature diagnostics from the exact source panel in the hash chain."""

    source_path = Path(str(manifest.get("_verified_source_prediction_path", "")))
    if not source_path.exists():
        raise FileNotFoundError(
            "verified source prediction panel is required for coverage/WD attribution"
        )
    schema = set(pq.ParquetFile(source_path).schema.names)
    required = {
        "candidate",
        "canonical_event_key",
        "dg_id_key",
        "prediction",
        *ATTRIBUTION_COLUMNS,
    }
    missing = sorted(required - schema)
    if missing:
        raise ValueError(
            f"verified source prediction panel lacks attribution columns: {missing}"
        )
    columns = [
        "candidate",
        "canonical_event_key",
        "dg_id_key",
        "prediction",
        *ATTRIBUTION_COLUMNS,
    ]
    control_name = str(manifest.get("control_name", ""))
    if not control_name:
        raise ValueError("compact finalist manifest lacks control_name")
    source = pd.read_parquet(
        source_path,
        columns=columns,
        filters=[("candidate", "in", [selected_name, control_name])],
    ).copy()
    source = source.loc[
        source["candidate"].astype(str).isin([selected_name, control_name])
    ].copy()
    source["event_key"] = source["canonical_event_key"].astype(str).str.strip()
    source = source.loc[
        source["event_key"].isin(set(finalist["event_key"].astype(str)))
    ].copy()
    source["dg_id"] = pd.to_numeric(source["dg_id_key"], errors="coerce").astype(
        "Int64"
    )
    source["source_prediction_check"] = pd.to_numeric(
        source["prediction"], errors="coerce"
    )
    if (
        source[["event_key", "dg_id"]].isna().any(axis=None)
        or source["event_key"].eq("").any()
    ):
        raise ValueError("verified source attribution has invalid event/player keys")
    if source.duplicated(["candidate", "event_key", "dg_id"]).any():
        raise ValueError(
            "verified source attribution is ambiguous by candidate/event/player"
        )
    finalist_detail = source.loc[
        source["candidate"].eq(selected_name),
        ["event_key", "dg_id", "source_prediction_check", *ATTRIBUTION_COLUMNS],
    ].copy()
    finalist_detail = finalist_detail.rename(
        columns={"source_prediction_check": "source_finalist_core_check"}
    )
    control_detail = source.loc[source["candidate"].eq(control_name), [
        "event_key", "dg_id", "source_prediction_check"
    ]].copy().rename(
        columns={"source_prediction_check": "source_control_core_check"}
    )
    finalist_detail["dg_id"] = finalist_detail["dg_id"].astype(int)
    control_detail["dg_id"] = control_detail["dg_id"].astype(int)
    output = finalist.merge(
        finalist_detail,
        on=["event_key", "dg_id"],
        how="left",
        validate="one_to_one",
    ).merge(
        control_detail,
        on=["event_key", "dg_id"],
        how="left",
        validate="one_to_one",
    )
    missing_source = output[
        ["source_finalist_core_check", "source_control_core_check"]
    ].isna().any(axis=1)
    if missing_source.any():
        raise ValueError(
            "compact finalist rows are missing from verified source panel: "
            f"{int(missing_source.sum())}"
        )
    finalist_error = (
        output["finalist_core"] - output["source_finalist_core_check"]
    ).abs()
    control_error = (
        output["control_core"] - output["source_control_core_check"]
    ).abs()
    if finalist_error.max() > 1e-10:
        raise ValueError("compact finalist core differs from verified source prediction")
    if control_error.max() > 1e-10:
        raise ValueError("compact control core differs from verified source prediction")
    output = output.drop(
        columns=["source_finalist_core_check", "source_control_core_check"]
    )
    return output, {
        "source_prediction_path": str(source_path.resolve()),
        "source_rows_attached": int(len(output)),
        "source_coverage_rate": float((~missing_source).mean()),
        "max_compact_vs_source_prediction_error": float(finalist_error.max()),
        "max_compact_control_vs_source_prediction_error": float(
            control_error.max()
        ),
        "attribution_columns": list(ATTRIBUTION_COLUMNS),
    }


def _finish_has_third_participant(value: object) -> bool:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in {"{", "["}:
            try:
                return _finish_has_third_participant(json.loads(stripped))
            except json.JSONDecodeError:
                return False
        return False
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).casefold())
            if normalized.startswith("p3"):
                scalar_missing = nested is None or (
                    not isinstance(nested, (Mapping, list, tuple))
                    and bool(pd.isna(nested))
                )
                if not scalar_missing and nested not in ("", [], {}):
                    return True
            if normalized in {
                "participants",
                "players",
                "competitors",
                "entrants",
            } and isinstance(nested, (list, tuple)) and len(nested) > 2:
                return True
            if _finish_has_third_participant(nested):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_finish_has_third_participant(nested) for nested in value)
    return False


def normalize_finish_market_prices(
    matchup_rows: Sequence[dict[str, Any]],
    outright_rows: Sequence[dict[str, Any]],
) -> pd.DataFrame:
    """Keep both archived endpoints and the selected book's opposite H2H quote."""

    pieces: list[pd.DataFrame] = []
    matchup_input = pd.DataFrame(matchup_rows)
    raw_matchups = (
        v4bt.add_event_key(matchup_input) if not matchup_input.empty else matchup_input
    )
    if not raw_matchups.empty:
        bet_type = (
            raw_matchups.get(
                "bet_type", pd.Series("", index=raw_matchups.index, dtype=str)
            )
            .astype(str)
            .str.casefold()
            .str.replace(r"[_-]+", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        tournament = bet_type.str.contains("tournament|72", regex=True, na=False)
        unsupported_variant = bet_type.str.contains(
            r"(?:^|\s)(?:3|three)\s*ball(?:\s|$)",
            regex=True,
            na=False,
        )
        if (tournament & unsupported_variant).any():
            raise ValueError(
                "tournament/72-hole three-ball prices cannot be "
                "normalized as two-way matchups"
            )
        p3_columns = [
            column
            for column in raw_matchups.columns
            if re.sub(r"[^a-z0-9]", "", str(column).casefold()).startswith("p3")
        ]
        populated_p3 = pd.Series(False, index=raw_matchups.index)
        for column in p3_columns:
            values = raw_matchups[column]
            populated_p3 |= values.notna() & values.astype(str).str.strip().ne("")
        populated_p3 |= raw_matchups.apply(
            lambda row: _finish_has_third_participant(row.to_dict()), axis=1
        )
        if (tournament & populated_p3).any():
            raise ValueError(
                "tournament/72-hole matchup has a populated third participant"
            )
        raw_matchups = raw_matchups.loc[tournament].copy()
        p1_ids = raw_matchups.get("p1_dg_id", pd.Series(index=raw_matchups.index)).map(
            v4bt.canonical_player_id
        )
        p2_ids = raw_matchups.get("p2_dg_id", pd.Series(index=raw_matchups.index)).map(
            v4bt.canonical_player_id
        )
        invalid_pair = p1_ids.eq("") | p2_ids.eq("") | p1_ids.eq(p2_ids)
        if invalid_pair.any():
            raise ValueError(
                "tournament matchup requires distinct nonempty p1/p2 participants"
            )
        expanded: list[dict[str, Any]] = []
        for row in raw_matchups.to_dict(orient="records"):
            for side, other in (("p1", "p2"), ("p2", "p1")):
                expanded.append(
                    {
                        "event_id": row["event_key"],
                        "event_key": row["event_key"],
                        "year": row["year"],
                        "dg_event_id": row["dg_event_id"],
                        "event_name": row.get("event_name"),
                        "market": "tournament_matchup",
                        "selection_id": v4bt.canonical_player_id(
                            row.get(f"{side}_dg_id")
                        ),
                        "selection_name": row.get(f"{side}_player_name"),
                        "opponent_id": v4bt.canonical_player_id(
                            row.get(f"{other}_dg_id")
                        ),
                        "opponent_name": row.get(f"{other}_player_name"),
                        "book": v4bt.normalize_text(row.get("book")),
                        "open_odds": row.get(f"{side}_open"),
                        "close_odds_archived": row.get(f"{side}_close"),
                        "opposite_open_odds": row.get(f"{other}_open"),
                        "opposite_close_odds": row.get(f"{other}_close"),
                        "open_time": row.get("open_time"),
                        "close_time_archived": row.get("close_time"),
                        "tie_rule": row.get("tie_rule"),
                        "outcome": row.get(
                            f"{side}_outcome_text", row.get(f"{side}_outcome")
                        ),
                        "outcome_numeric": row.get(f"{side}_outcome"),
                        "bet_outcome_numeric": np.nan,
                        "record_id": row.get("record_id"),
                    }
                )
        matchup = pd.DataFrame(expanded)
        if not matchup.empty:
            matchup = matchup.loc[
                matchup["selection_id"].ne("")
                & matchup["opponent_id"].ne("")
                & matchup["book"].isin(SHARP_BOOKS)
            ].copy()
            pieces.append(matchup)

    outright_input = pd.DataFrame(outright_rows)
    raw_outrights = (
        v4bt.add_event_key(outright_input) if not outright_input.empty else outright_input
    )
    if not raw_outrights.empty:
        raw_outrights["market"] = raw_outrights.get("market", "").map(
            v4bt.canonical_market
        )
        raw_outrights = raw_outrights.loc[
            raw_outrights["market"].isin(v4bt.MARKETS[1:])
        ].copy()
        outright = pd.DataFrame(
            {
                "event_id": raw_outrights["event_key"],
                "event_key": raw_outrights["event_key"],
                "year": raw_outrights["year"],
                "dg_event_id": raw_outrights["dg_event_id"],
                "event_name": raw_outrights.get("event_name"),
                "market": raw_outrights["market"],
                "selection_id": raw_outrights.get("dg_id").map(
                    v4bt.canonical_player_id
                ),
                "selection_name": raw_outrights.get("player_name"),
                "opponent_id": "",
                "opponent_name": "",
                "book": raw_outrights.get("book", "").map(v4bt.normalize_text),
                "open_odds": raw_outrights.get("open_odds"),
                "close_odds_archived": raw_outrights.get("close_odds"),
                "opposite_open_odds": np.nan,
                "opposite_close_odds": np.nan,
                "open_time": raw_outrights.get("open_time"),
                "close_time_archived": raw_outrights.get("close_time"),
                "tie_rule": "",
                "outcome": raw_outrights.get(
                    "outcome", raw_outrights.get("bet_outcome_text")
                ),
                "outcome_numeric": np.nan,
                "bet_outcome_numeric": raw_outrights.get("bet_outcome_numeric"),
                "bet_outcome_text": raw_outrights.get("bet_outcome_text"),
                "record_id": raw_outrights.get("record_id"),
            }
        )
        outright = outright.loc[
            outright["selection_id"].ne("") & outright["book"].isin(SHARP_BOOKS)
        ].copy()
        pieces.append(outright)
    if not pieces:
        return pd.DataFrame()
    output = pd.concat(pieces, ignore_index=True, sort=False)
    for column in (
        "open_odds",
        "close_odds_archived",
        "opposite_open_odds",
        "opposite_close_odds",
        "outcome_numeric",
        "bet_outcome_numeric",
    ):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def _unique_archived_numeric(values: pd.Series, label: str) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    unique = np.unique(np.round(numeric.to_numpy(float), 12))
    if len(unique) > 1:
        raise ValueError(f"inconsistent archived settlement for {label}: {unique.tolist()}")
    return float(unique[0]) if len(unique) else np.nan


def _canonical_finish_result(value: object) -> str:
    return str(value or "").strip().upper()


def _finish_rank(value: object) -> int | None:
    match = re.fullmatch(r"T?(\d+)", _canonical_finish_result(value))
    return int(match.group(1)) if match else None


def load_official_outright_results(
    db_path: Path,
    event_crosswalk: pd.DataFrame,
    lines: pd.DataFrame,
) -> pd.DataFrame:
    """Load exact player finishes and full tie-group sizes from the locked DB.

    ``player_rounds.fin_text`` is the authoritative completed-event result used
    here.  A distinct player-event row is formed before tie groups are counted;
    round-state ``finish_pos`` is intentionally never used.
    """

    outright = lines.loc[~lines["market"].eq("tournament_matchup")].copy()
    if outright.empty:
        return pd.DataFrame(
            columns=(
                "event_id",
                "selection_id",
                "official_finish",
                "official_tie_count",
                "local_year",
                "local_event_id",
            )
        )
    required_crosswalk = {
        "year",
        "event_id",
        "odds_query_year",
        "odds_event_id",
        "odds_match_found",
        "crosswalk_ambiguous",
    }
    missing_crosswalk = required_crosswalk - set(event_crosswalk.columns)
    if missing_crosswalk:
        raise ValueError(
            f"event crosswalk lacks official-result columns: {sorted(missing_crosswalk)}"
        )
    mapping = event_crosswalk.copy()
    mapping["odds_event_key"] = [
        v4bt.canonical_event_key(year, event_id)
        for year, event_id in zip(
            mapping["odds_query_year"], mapping["odds_event_id"]
        )
    ]
    requested_events = set(outright["event_id"].dropna().astype(str))
    mapping = mapping.loc[mapping["odds_event_key"].isin(requested_events)].copy()
    if mapping["odds_event_key"].duplicated().any():
        raise ValueError("official-result crosswalk maps an odds event more than once")
    mapped_events = set(mapping["odds_event_key"].astype(str))
    if mapped_events != requested_events:
        raise ValueError(
            "official-result crosswalk does not cover every outright event: "
            f"{sorted(requested_events - mapped_events)[:10]}"
        )
    if not mapping["odds_match_found"].eq(True).all():
        raise ValueError("official-result crosswalk contains an unmatched outright event")
    if mapping["crosswalk_ambiguous"].fillna(True).astype(bool).any():
        raise ValueError("official-result crosswalk contains an ambiguous outright event")
    mapping = mapping[
        ["odds_event_key", "year", "event_id"]
    ].rename(columns={"year": "local_year", "event_id": "local_event_id"})
    local_years = pd.to_numeric(mapping["local_year"], errors="raise").astype(int)
    with sqlite3.connect(str(db_path)) as connection:
        round_results = pd.read_sql_query(
            """
            SELECT year, event_id, dg_id, fin_text
            FROM player_rounds
            WHERE lower(tour) = 'pga' AND year BETWEEN ? AND ?
            """,
            connection,
            params=(int(local_years.min()), int(local_years.max())),
        )
    round_results["year"] = pd.to_numeric(
        round_results["year"], errors="raise"
    ).astype(int)
    round_results["event_id"] = pd.to_numeric(
        round_results["event_id"], errors="raise"
    ).astype(int)
    round_results["selection_id"] = round_results["dg_id"].map(
        v4bt.canonical_player_id
    )
    round_results["official_finish"] = round_results["fin_text"].map(
        _canonical_finish_result
    )
    joined = round_results.merge(
        mapping,
        left_on=["year", "event_id"],
        right_on=["local_year", "local_event_id"],
        how="inner",
        validate="many_to_one",
    )
    player_finish_counts = (
        joined.groupby(
            ["odds_event_key", "selection_id"], observed=True
        )["official_finish"]
        .nunique(dropna=False)
    )
    if not player_finish_counts.eq(1).all():
        raise ValueError("official results contain ambiguous player-event finishes")
    player_results = joined.drop_duplicates(
        ["odds_event_key", "selection_id", "official_finish"]
    ).copy()
    if player_results["official_finish"].eq("").any():
        raise ValueError("official results contain a blank finish")
    tie_counts = (
        player_results.groupby(
            ["odds_event_key", "official_finish"], observed=True
        )["selection_id"]
        .nunique()
        .rename("official_tie_count")
        .reset_index()
    )
    player_results = player_results.merge(
        tie_counts,
        on=["odds_event_key", "official_finish"],
        how="left",
        validate="many_to_one",
    )
    return player_results[
        [
            "odds_event_key",
            "selection_id",
            "official_finish",
            "official_tie_count",
            "local_year",
            "local_event_id",
        ]
    ].rename(columns={"odds_event_key": "event_id"})


def _correction_record_id_digest(record_ids: Iterable[object]) -> str:
    normalized = sorted(str(value).strip() for value in record_ids)
    if any(not value for value in normalized) or len(normalized) != len(set(normalized)):
        raise ValueError("settlement corrections require unique nonempty record_id values")
    payload = "".join(f"{value}\n" for value in normalized).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _unchanged_settlement_payload_digest(
    frame: pd.DataFrame, value_column: str
) -> str:
    """Hash unchanged row identities plus one canonical settlement value."""

    columns = [
        "record_id",
        "event_id",
        "market",
        "selection_id",
        "opponent_id",
        "book",
    ]
    payload = frame[columns].copy()
    payload["settlement_value"] = pd.to_numeric(
        frame[value_column], errors="raise"
    ).map(lambda value: f"{float(value):.12f}")
    payload = payload.sort_values(columns, kind="mergesort")
    encoded = payload.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def reconcile_outright_settlements(
    lines: pd.DataFrame,
    official_results: pd.DataFrame,
    *,
    enforce_frozen_authority: bool = False,
    expected_primary_override_count: int = EXPECTED_PINNACLE_FULL_PAYOUT_OVERRIDES,
    expected_primary_record_id_sha256: str = (
        EXPECTED_PINNACLE_FULL_PAYOUT_RECORD_ID_SHA256
    ),
    expected_standard_discrepancy_count: int = (
        EXPECTED_STANDARD_DEAD_HEAT_DISCREPANCIES
    ),
    expected_standard_record_id_sha256: str = (
        EXPECTED_STANDARD_DEAD_HEAT_RECORD_ID_SHA256
    ),
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Reconstruct outrights from official results plus explicit book rules.

    BetOnline and Betcris are required to agree exactly with standard dead-heat
    math. Pinnacle top-finish qualifiers are paid in full under its published
    rule. Raw, universal-standard, and primary book-rule settlements are all
    retained. The frozen run pins both the 50-row primary override inventory
    and the 37-row universal-standard discrepancy inventory.
    """

    output = lines.copy()
    raw_numeric = pd.to_numeric(output["bet_outcome_numeric"], errors="coerce")
    output["archived_bet_outcome_numeric"] = raw_numeric
    output["standard_dead_heat_bet_outcome_numeric"] = raw_numeric
    output["book_rule_bet_outcome_numeric"] = raw_numeric
    output["settlement_reconstructed"] = False
    output["settlement_reconstruction_delta"] = 0.0
    output["standard_dead_heat_delta_vs_archive"] = 0.0
    output["settlement_source"] = np.where(
        output["market"].eq("tournament_matchup"),
        "archived_matchup_outcome",
        "pending_official_outright_reconciliation",
    )
    outright_mask = ~output["market"].eq("tournament_matchup")
    if not outright_mask.any():
        audit = {
            "outright_rows": 0,
            "primary_override_count": 0,
            "primary_override_record_id_sha256": _correction_record_id_digest([]),
            "standard_discrepancy_count": 0,
            "standard_discrepancy_record_id_sha256": _correction_record_id_digest([]),
            "nonconflicting_rows_unchanged": True,
        }
        return output, pd.DataFrame(), audit
    required_official = {
        "event_id",
        "selection_id",
        "official_finish",
        "official_tie_count",
        "local_year",
        "local_event_id",
    }
    missing_official = required_official - set(official_results.columns)
    if missing_official:
        raise ValueError(
            f"official results lack settlement columns: {sorted(missing_official)}"
        )
    if official_results.duplicated(["event_id", "selection_id"]).any():
        raise ValueError("official results duplicate event/player identity")
    detail = output.loc[outright_mask].copy()
    detail["_source_index"] = detail.index
    detail = detail.merge(
        official_results,
        on=["event_id", "selection_id"],
        how="left",
        validate="many_to_one",
    )
    if detail["official_finish"].isna().any():
        missing = detail.loc[
            detail["official_finish"].isna(),
            ["event_id", "market", "selection_id", "book"],
        ]
        raise ValueError(
            "official result is missing for an outright row: "
            + missing.head(10).to_json(orient="records")
        )
    detail["official_finish"] = detail["official_finish"].map(
        _canonical_finish_result
    )
    detail["archived_finish"] = detail["outcome"].map(_canonical_finish_result)
    finish_mismatch = ~detail["archived_finish"].eq(detail["official_finish"])
    if finish_mismatch.any():
        raise ValueError(
            "archived outright finish disagrees with official player_rounds.fin_text"
        )
    detail["official_rank"] = detail["official_finish"].map(_finish_rank)
    unknown = detail["official_rank"].isna() & ~detail["official_finish"].isin(
        KNOWN_NONFINISH_RESULTS
    )
    if unknown.any():
        raise ValueError(
            "unknown official outright finish status: "
            f"{sorted(detail.loc[unknown, 'official_finish'].unique())}"
        )
    detail["official_tie_count"] = pd.to_numeric(
        detail["official_tie_count"], errors="coerce"
    )
    ranked = detail["official_rank"].notna()
    invalid_tie_count = ranked & (
        ~np.isfinite(detail["official_tie_count"])
        | detail["official_tie_count"].le(0)
        | ~detail["official_tie_count"].mod(1).eq(0)
    )
    tied_label = detail["official_finish"].str.startswith("T")
    invalid_tie_count |= ranked & tied_label & detail["official_tie_count"].lt(2)
    invalid_tie_count |= ranked & ~tied_label & detail["official_tie_count"].ne(1)
    if invalid_tie_count.any():
        raise ValueError("official result has an ambiguous or impossible tie group")
    detail["cutoff"] = detail["market"].map(OUTRIGHT_MARKET_CUTOFF)
    if detail["cutoff"].isna().any():
        raise ValueError("unknown outright market in settlement reconstruction")
    archived = pd.to_numeric(detail["bet_outcome_numeric"], errors="coerce")
    if archived.isna().any() or (~archived.between(0.0, 1.0)).any():
        raise ValueError("archived outright settlement must be finite and in [0, 1]")
    rank_value = pd.to_numeric(detail["official_rank"], errors="coerce")
    places_remaining = detail["cutoff"] - rank_value + 1.0
    standard_dead_heat = np.where(
        ~ranked,
        0.0,
        np.where(
            rank_value > detail["cutoff"],
            0.0,
            np.minimum(1.0, places_remaining / detail["official_tie_count"]),
        ),
    )
    detail["standard_dead_heat_bet_outcome_numeric"] = np.asarray(
        standard_dead_heat, dtype=float
    )
    nonfinish_nonzero = ~ranked & archived.ne(0.0)
    if nonfinish_nonzero.any():
        raise ValueError("CUT/WD/DQ outright settlement is not a loss")
    standard_difference = (
        detail["standard_dead_heat_bet_outcome_numeric"] - archived
    )
    detail["standard_dead_heat_delta_vs_archive"] = standard_difference
    detail["standard_dead_heat_differs_from_archive"] = standard_difference.abs().gt(
        1e-12
    )
    non_pinnacle = detail["book"].isin(("betonline", "betcris"))
    if (non_pinnacle & detail["standard_dead_heat_differs_from_archive"]).any():
        raise ValueError(
            "BetOnline/Betcris archive disagrees with official standard dead-heat math"
        )
    pinnacle_top_finish = detail["book"].eq("pinnacle") & detail["market"].isin(
        ("top_5", "top_10", "top_20")
    )
    physically_qualifies = ranked & rank_value.le(detail["cutoff"])
    detail["book_rule_bet_outcome_numeric"] = np.where(
        pinnacle_top_finish & physically_qualifies,
        1.0,
        detail["standard_dead_heat_bet_outcome_numeric"],
    )
    detail["settlement_reconstruction_delta"] = (
        detail["book_rule_bet_outcome_numeric"] - archived
    )
    detail["settlement_reconstructed"] = detail[
        "settlement_reconstruction_delta"
    ].abs().gt(1e-12)
    identity = ["event_id", "market", "selection_id", "opponent_id"]
    qualifying_books = {
        tuple(str(value) for value in key): sorted(set(group["book"].astype(str)))
        for key, group in detail.loc[physically_qualifies].groupby(
            identity, dropna=False, observed=True
        )
    }
    detail["cross_book_qualification_corroboration"] = [
        "|".join(
            book
            for book in qualifying_books.get(
                tuple(str(value) for value in key), []
            )
            if book != str(current_book)
        )
        for (*key, current_book) in detail[[*identity, "book"]].itertuples(
            index=False, name=None
        )
    ]
    corrections = detail.loc[detail["settlement_reconstructed"]].copy()
    if corrections["cross_book_qualification_corroboration"].eq("").any():
        raise ValueError(
            "a Pinnacle full-payout override lacks cross-book qualification corroboration"
        )
    if not corrections.empty:
        if not (
            corrections["book"].eq("pinnacle")
            & corrections["market"].isin(("top_5", "top_10", "top_20"))
            & pd.to_numeric(
                corrections["bet_outcome_numeric"], errors="coerce"
            ).between(0.0, 1.0, inclusive="neither")
            & corrections["book_rule_bet_outcome_numeric"].eq(1.0)
        ).all():
            raise ValueError(
                "primary settlement override is not a fractional qualifying "
                "Pinnacle top-finish row"
            )
    standard_discrepancies = detail.loc[
        detail["standard_dead_heat_differs_from_archive"]
    ].copy()
    if not standard_discrepancies.empty and not (
        standard_discrepancies["book"].eq("pinnacle")
        & pd.to_numeric(
            standard_discrepancies["bet_outcome_numeric"], errors="coerce"
        ).between(0.0, 1.0, inclusive="neither")
        & standard_discrepancies[
            "standard_dead_heat_bet_outcome_numeric"
        ].gt(0.0)
    ).all():
        raise ValueError(
            "universal-standard discrepancy is not a fractional qualifying "
            "Pinnacle top-finish row"
        )
    correction_record_ids = set(corrections["record_id"].astype(str))
    standard_record_ids = set(standard_discrepancies["record_id"].astype(str))
    if not standard_record_ids.issubset(correction_record_ids):
        raise ValueError(
            "universal-standard discrepancies are not fully represented in the "
            "emitted primary-override inventory"
        )
    primary_record_id_sha256 = _correction_record_id_digest(
        corrections["record_id"]
    )
    standard_record_id_sha256 = _correction_record_id_digest(
        standard_discrepancies["record_id"]
    )
    if enforce_frozen_authority:
        if len(corrections) != int(expected_primary_override_count):
            raise ValueError(
                "frozen Pinnacle full-payout override count differs from the audit: "
                f"expected {expected_primary_override_count}, got {len(corrections)}"
            )
        if primary_record_id_sha256 != str(
            expected_primary_record_id_sha256
        ).lower():
            raise ValueError(
                "frozen Pinnacle full-payout record-id digest differs "
                "from the independent audit"
            )
        if len(standard_discrepancies) != int(expected_standard_discrepancy_count):
            raise ValueError(
                "frozen universal-standard discrepancy count differs from the audit: "
                f"expected {expected_standard_discrepancy_count}, "
                f"got {len(standard_discrepancies)}"
            )
        if standard_record_id_sha256 != str(
            expected_standard_record_id_sha256
        ).lower():
            raise ValueError(
                "frozen universal-standard record-id digest differs from the "
                "independent audit"
            )
    unchanged = detail.loc[~detail["settlement_reconstructed"]]
    if not np.array_equal(
        unchanged["book_rule_bet_outcome_numeric"].to_numpy(float),
        pd.to_numeric(unchanged["bet_outcome_numeric"], errors="raise").to_numpy(float),
    ):
        raise AssertionError("nonconflicting outright settlement changed")
    unchanged_raw_payload_sha256 = _unchanged_settlement_payload_digest(
        unchanged, "bet_outcome_numeric"
    )
    unchanged_primary_payload_sha256 = _unchanged_settlement_payload_digest(
        unchanged, "book_rule_bet_outcome_numeric"
    )
    if unchanged_raw_payload_sha256 != unchanged_primary_payload_sha256:
        raise AssertionError("nonconflicting outright settlement payload changed")
    detail = detail.sort_values("_source_index", kind="mergesort")
    output.loc[detail["_source_index"], "bet_outcome_numeric"] = detail[
        "book_rule_bet_outcome_numeric"
    ].to_numpy(float)
    output.loc[
        detail["_source_index"], "standard_dead_heat_bet_outcome_numeric"
    ] = detail["standard_dead_heat_bet_outcome_numeric"].to_numpy(float)
    output.loc[
        detail["_source_index"], "book_rule_bet_outcome_numeric"
    ] = detail["book_rule_bet_outcome_numeric"].to_numpy(float)
    output.loc[detail["_source_index"], "settlement_reconstructed"] = detail[
        "settlement_reconstructed"
    ].to_numpy(bool)
    output.loc[
        detail["_source_index"], "settlement_reconstruction_delta"
    ] = detail["settlement_reconstruction_delta"].to_numpy(float)
    output.loc[
        detail["_source_index"], "standard_dead_heat_delta_vs_archive"
    ] = detail["standard_dead_heat_delta_vs_archive"].to_numpy(float)
    output.loc[detail["_source_index"], "settlement_source"] = np.where(
        detail["book"].eq("pinnacle")
        & detail["market"].isin(("top_5", "top_10", "top_20")),
        "pinnacle_full_ties_rule_plus_locked_official_finish",
        "archive_verified_against_locked_official_standard_dead_heat",
    )
    correction_columns = [
        "record_id",
        "event_id",
        "local_year",
        "local_event_id",
        "event_name",
        "market",
        "selection_id",
        "selection_name",
        "book",
        "archived_finish",
        "official_finish",
        "official_tie_count",
        "bet_outcome_text",
        "bet_outcome_numeric",
        "standard_dead_heat_bet_outcome_numeric",
        "book_rule_bet_outcome_numeric",
        "standard_dead_heat_delta_vs_archive",
        "settlement_reconstruction_delta",
        "standard_dead_heat_differs_from_archive",
        "settlement_reconstructed",
        "cross_book_qualification_corroboration",
    ]
    corrections = corrections[correction_columns].rename(
        columns={"bet_outcome_numeric": "archived_bet_outcome_numeric"}
    )
    corrections = corrections.sort_values(
        ["event_id", "market", "selection_id", "book", "record_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    correction_csv = corrections.to_csv(
        index=False, float_format="%.12f", lineterminator="\n"
    ).encode("utf-8")
    audit = {
        "settlement_claim": "reconstructed_from_locked_official_results_and_book_rules",
        "primary_settlement_scenario": "book_rule_primary",
        "official_result_source": "dg_historical.db::player_rounds.fin_text",
        "standard_dead_heat_formula": (
            "min(1,(market_cutoff-finish_rank+1)/full_official_tie_group_size)"
        ),
        "pinnacle_top_finish_rule_url": PINNACLE_TOP_FINISH_RULE_URL,
        "pinnacle_top_finish_rule": PINNACLE_TOP_FINISH_RULE,
        "pinnacle_top_finish_rule_provenance": (
            "embedded_in_hash-tracked_evaluator_source; current official rule plus "
            "Internet Archive snapshots and pre-window public corroboration"
        ),
        "settlement_scenarios": [
            "book_rule_primary",
            "raw_archive_sensitivity",
            "universal_standard_dead_heat_sensitivity",
        ],
        "outright_rows": int(len(detail)),
        "primary_override_count": int(len(corrections)),
        "primary_override_events": int(corrections["event_id"].nunique()),
        "primary_override_record_id_sha256": primary_record_id_sha256,
        "standard_discrepancy_count": int(len(standard_discrepancies)),
        "standard_discrepancy_events": int(
            standard_discrepancies["event_id"].nunique()
        ),
        "standard_discrepancy_record_id_sha256": standard_record_id_sha256,
        "correction_csv_payload_sha256": hashlib.sha256(correction_csv).hexdigest(),
        "primary_overrides_all_pinnacle": bool(
            corrections.empty or corrections["book"].eq("pinnacle").all()
        ),
        "primary_overrides_all_cross_book_qualification_corroborated": bool(
            corrections.empty
            or corrections[
                "cross_book_qualification_corroboration"
            ].ne("").all()
        ),
        "nonconflicting_rows": int(len(detail) - len(corrections)),
        "nonconflicting_rows_unchanged": True,
        "nonconflicting_raw_payload_sha256": unchanged_raw_payload_sha256,
        "nonconflicting_primary_payload_sha256": (
            unchanged_primary_payload_sha256
        ),
        "standard_discrepancies_explicitly_emitted": bool(
            standard_record_ids.issubset(correction_record_ids)
        ),
        "known_nonfinish_status_rows": {
            str(status): int(detail["official_finish"].eq(status).sum())
            for status in sorted(KNOWN_NONFINISH_RESULTS)
        },
        "unknown_finish_status_rows": 0,
        "missing_official_result_rows": 0,
        "archived_finish_disagreement_rows": 0,
        "frozen_authority_enforced": bool(enforce_frozen_authority),
    }
    return output, corrections, audit


def _validate_finish_settlement_coherence(lines: pd.DataFrame) -> dict[str, Any]:
    """Validate one physical result globally, allowing book-specific tie rules."""

    matchups = lines.loc[lines["market"].eq("tournament_matchup")].copy()
    physical_pairs = 0
    if not matchups.empty:
        left = matchups["selection_id"].astype(str)
        right = matchups["opponent_id"].astype(str)
        if left.eq("").any() or right.eq("").any() or left.eq(right).any():
            raise ValueError("matchup settlement requires distinct nonempty participants")
        matchups["_pair_lo"] = np.where(left <= right, left, right)
        matchups["_pair_hi"] = np.where(left <= right, right, left)
        physical_by_pair: dict[tuple[str, str, str, str], list[str | None]] = {}
        pair_book_keys = ["event_id", "market", "_pair_lo", "_pair_hi", "book"]
        for group_key, group in matchups.groupby(
            pair_book_keys, dropna=False, observed=True
        ):
            lo = str(group_key[2])
            hi = str(group_key[3])
            directional: dict[str, float] = {}
            rule_classes: set[bool] = set()
            for selection_id, direction in group.groupby(
                "selection_id", dropna=False, observed=True
            ):
                selection = str(selection_id)
                if selection not in {lo, hi}:
                    raise ValueError(f"impossible reverse matchup participant: {group_key}")
                directional[selection] = _unique_archived_numeric(
                    direction["outcome_numeric"], f"{group_key}/{selection}"
                )
                rules = direction["tie_rule"].dropna().astype(str).str.strip()
                rules = rules.loc[rules.ne("")]
                classes = set(rules.map(ties_lose).astype(bool))
                if len(classes) > 1:
                    raise ValueError(
                        f"inconsistent tie-rule class within book/direction: {group_key}"
                    )
                if classes:
                    rule_classes.update(classes)
            if set(directional) != {lo, hi}:
                raise ValueError(f"matchup lacks coherent reverse legs: {group_key}")
            if len(rule_classes) > 1:
                raise ValueError(f"reverse legs disagree on tie rule: {group_key}")
            lo_value, hi_value = directional[lo], directional[hi]
            if np.isnan(lo_value) and np.isnan(hi_value):
                physical: str | None = None
            elif np.isnan(lo_value) or np.isnan(hi_value):
                raise ValueError(f"only one reverse matchup leg is graded: {group_key}")
            else:
                if lo_value not in {0.0, 0.5, 1.0} or hi_value not in {
                    0.0,
                    0.5,
                    1.0,
                }:
                    raise ValueError(f"invalid reverse matchup outcome: {group_key}")
                if (lo_value, hi_value) == (1.0, 0.0):
                    physical = "lo_win"
                elif (lo_value, hi_value) == (0.0, 1.0):
                    physical = "hi_win"
                elif (lo_value, hi_value) == (0.5, 0.5) and not any(rule_classes):
                    physical = "tie"
                elif (lo_value, hi_value) == (0.0, 0.0) and any(rule_classes):
                    physical = "tie"
                else:
                    raise ValueError(
                        "impossible reverse matchup settlement tuple "
                        f"{(lo_value, hi_value)} for {group_key}"
                    )
            pair_key = tuple(str(value) for value in group_key[:4])
            physical_by_pair.setdefault(pair_key, []).append(physical)
        for pair_key, results in physical_by_pair.items():
            graded = {result for result in results if result is not None}
            if len(graded) > 1:
                raise ValueError(
                    f"cross-book physical settlement conflict for {pair_key}: {sorted(graded)}"
                )
            if graded and any(result is None for result in results):
                raise ValueError(
                    f"cross-book settlement is graded for only some books: {pair_key}"
                )
        physical_pairs = len(physical_by_pair)

    outright = lines.loc[~lines["market"].eq("tournament_matchup")].copy()
    outright_groups = 0
    if not outright.empty:
        keys = ["event_id", "market", "selection_id", "opponent_id"]
        for group_key, group in outright.groupby(keys, dropna=False, observed=True):
            by_book = [
                _unique_archived_numeric(
                    book_group["bet_outcome_numeric"],
                    f"outright {group_key}/{book}",
                )
                for book, book_group in group.groupby(
                    "book", dropna=False, observed=True
                )
            ]
            finite = np.asarray([value for value in by_book if np.isfinite(value)])
            if len(finite) not in {0, len(by_book)}:
                raise ValueError(
                    f"cross-book outright settlement is only partially graded: {group_key}"
                )
            if len(finite) and ((finite < 0.0).any() or (finite > 1.0).any()):
                raise ValueError(f"invalid outright settlement fraction: {group_key}")
            if len(set((finite > 0.0).tolist())) > 1:
                raise ValueError(
                    f"cross-book outright physical qualification conflict: {group_key}"
                )
            if len(np.unique(np.round(finite, 12))) > 1:
                sources = group["settlement_source"].dropna().astype(str)
                if sources.eq("archived_unreconstructed").any() or sources.str.startswith(
                    "pending_"
                ).any():
                    raise ValueError(
                        "cross-book positive settlement differences require locked "
                        f"official-result/book-rule reconciliation: {group_key}"
                    )
            finishes = group["outcome"].dropna().map(_canonical_finish_result)
            finishes = finishes.loc[finishes.ne("")]
            if finishes.nunique() > 1:
                raise ValueError(
                    f"cross-book outright physical finish conflict: {group_key}"
                )
            outright_groups += 1
    return {
        "physical_matchup_pairs_verified": physical_pairs,
        "outright_settlements_verified": outright_groups,
    }


def _physical_tie_pair_books(lines: pd.DataFrame) -> set[tuple[str, str, str, str, str]]:
    """Return validated event/market/pair/book keys whose result was a tie."""

    matchups = lines.loc[lines["market"].eq("tournament_matchup")].copy()
    if matchups.empty:
        return set()
    left = matchups["selection_id"].astype(str)
    right = matchups["opponent_id"].astype(str)
    matchups["_pair_lo"] = np.where(left <= right, left, right)
    matchups["_pair_hi"] = np.where(left <= right, right, left)
    keys = ["event_id", "market", "_pair_lo", "_pair_hi", "book"]
    result: set[tuple[str, str, str, str, str]] = set()
    for group_key, group in matchups.groupby(keys, dropna=False, observed=True):
        outcomes = {
            str(selection): _unique_archived_numeric(
                direction["outcome_numeric"], f"tie marker {group_key}/{selection}"
            )
            for selection, direction in group.groupby(
                "selection_id", dropna=False, observed=True
            )
        }
        if len(outcomes) != 2 or any(np.isnan(value) for value in outcomes.values()):
            continue
        values = tuple(outcomes[player] for player in sorted(outcomes))
        rules = group["tie_rule"].dropna().astype(str).str.strip()
        rules = rules.loc[rules.ne("")]
        rule_is_ties_lose = bool(rules.map(ties_lose).iloc[0]) if len(rules) else False
        if values == (0.5, 0.5) or (values == (0.0, 0.0) and rule_is_ties_lose):
            result.add(tuple(str(value) for value in group_key))
    return result


def select_finish_line_sets(
    lines: pd.DataFrame,
) -> dict[tuple[str, str], pd.DataFrame]:
    """Build true per-book and independently shopped side-price menus."""

    lines = lines.copy()
    if "archived_bet_outcome_numeric" not in lines:
        lines["archived_bet_outcome_numeric"] = pd.to_numeric(
            lines.get("bet_outcome_numeric"), errors="coerce"
        )
    if "standard_dead_heat_bet_outcome_numeric" not in lines:
        lines["standard_dead_heat_bet_outcome_numeric"] = lines[
            "archived_bet_outcome_numeric"
        ]
    if "book_rule_bet_outcome_numeric" not in lines:
        lines["book_rule_bet_outcome_numeric"] = lines[
            "archived_bet_outcome_numeric"
        ]
    if "settlement_reconstruction_delta" not in lines:
        lines["settlement_reconstruction_delta"] = 0.0
    if "standard_dead_heat_delta_vs_archive" not in lines:
        lines["standard_dead_heat_delta_vs_archive"] = 0.0
    if "settlement_reconstructed" not in lines:
        lines["settlement_reconstructed"] = False
    if "settlement_source" not in lines:
        lines["settlement_source"] = "archived_unreconstructed"
    if "bet_outcome_text" not in lines:
        lines["bet_outcome_text"] = ""
    _validate_finish_settlement_coherence(lines)
    physical_tie_pairs = _physical_tie_pair_books(lines)
    keys = list(v4bt.KEY_COLS)
    settlement_keys = [*keys, "book"]
    settlement_rows: list[dict[str, Any]] = []
    for group_key, group in lines.groupby(
        settlement_keys, dropna=False, observed=True
    ):
        record = dict(zip(settlement_keys, group_key))
        for column in ("outcome_numeric", "bet_outcome_numeric"):
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            unique = np.unique(np.round(values.to_numpy(float), 12))
            if len(unique) > 1:
                raise ValueError(
                    f"inconsistent {column} across archived captures for {group_key}"
                )
            record[column] = float(unique[0]) if len(unique) else np.nan
        outcome_text = group["outcome"].dropna().astype(str).str.strip()
        outcome_text = outcome_text.loc[outcome_text.ne("")]
        record["outcome"] = outcome_text.iloc[-1] if len(outcome_text) else np.nan
        for column in (
            "archived_bet_outcome_numeric",
            "standard_dead_heat_bet_outcome_numeric",
            "book_rule_bet_outcome_numeric",
            "settlement_reconstruction_delta",
            "standard_dead_heat_delta_vs_archive",
        ):
            record[column] = _unique_archived_numeric(
                group[column], f"{column} {group_key}"
            )
        reconstructed_values = group["settlement_reconstructed"].dropna().astype(bool)
        if reconstructed_values.nunique() > 1:
            raise ValueError(
                f"inconsistent settlement reconstruction marker for {group_key}"
            )
        record["settlement_reconstructed"] = bool(
            reconstructed_values.iloc[0]
        ) if len(reconstructed_values) else False
        settlement_sources = group["settlement_source"].dropna().astype(str).str.strip()
        settlement_sources = settlement_sources.loc[settlement_sources.ne("")]
        if settlement_sources.nunique() > 1:
            raise ValueError(f"inconsistent settlement source for {group_key}")
        record["settlement_source"] = (
            settlement_sources.iloc[0] if len(settlement_sources) else ""
        )
        archived_text = group.get(
            "bet_outcome_text", pd.Series(index=group.index, dtype=object)
        ).dropna().astype(str).str.strip()
        archived_text = archived_text.loc[archived_text.ne("")]
        if archived_text.nunique() > 1:
            raise ValueError(f"inconsistent archived bet outcome text for {group_key}")
        record["archived_bet_outcome_text"] = (
            archived_text.iloc[0] if len(archived_text) else ""
        )
        tie_rules = group["tie_rule"].dropna().astype(str).str.strip()
        tie_rules = tie_rules.loc[tie_rules.ne("")]
        if len(tie_rules) and tie_rules.map(ties_lose).nunique() > 1:
            raise ValueError(
                f"inconsistent tie-rule class across archived captures for {group_key}"
            )
        record["tie_rule"] = tie_rules.iloc[-1] if len(tie_rules) else ""
        if str(record["market"]) == "tournament_matchup":
            selection_id = str(record["selection_id"])
            opponent_id = str(record["opponent_id"])
            pair_lo, pair_hi = sorted((selection_id, opponent_id))
            pair_book_key = (
                str(record["event_id"]),
                str(record["market"]),
                pair_lo,
                pair_hi,
                str(record["book"]),
            )
            record["is_tie"] = pair_book_key in physical_tie_pairs
        else:
            record["is_tie"] = False
        settlement_rows.append(record)
    settlement = pd.DataFrame(settlement_rows)
    outputs: dict[tuple[str, str], pd.DataFrame] = {}
    book_rank = {book: index for index, book in enumerate(SHARP_BOOKS)}
    for price_point in PRICE_POINTS:
        price_column = "open_odds" if price_point == "open" else "close_odds_archived"
        opposite_column = (
            "opposite_open_odds"
            if price_point == "open"
            else "opposite_close_odds"
        )
        time_column = "open_time" if price_point == "open" else "close_time_archived"
        valid = v4bt._valid_american(lines[price_column])
        matchup = lines["market"].eq("tournament_matchup")
        valid &= ~matchup | v4bt._valid_american(lines[opposite_column])
        endpoint = lines.loc[valid].copy()
        endpoint["execution_odds"] = pd.to_numeric(
            endpoint[price_column], errors="raise"
        ).astype(float)
        endpoint["same_book_opposite_odds"] = pd.to_numeric(
            endpoint[opposite_column], errors="coerce"
        )
        endpoint["execution_time"] = pd.to_datetime(
            endpoint[time_column], errors="coerce", utc=True
        )
        endpoint["price_point"] = price_point
        endpoint["archived_open_odds"] = endpoint["open_odds"]
        endpoint["archived_close_odds"] = endpoint["close_odds_archived"]
        endpoint["_book_rank"] = endpoint["book"].map(book_rank).astype(int)
        endpoint["_record_sort"] = endpoint["record_id"].fillna("").astype(str)
        endpoint = endpoint.sort_values(
            keys + ["book", "execution_time", "_record_sort"],
            ascending=[True] * len(keys)
            + [True, price_point == "open", True],
            kind="mergesort",
            na_position="last",
        ).drop_duplicates(keys + ["book"], keep="first")
        endpoint = endpoint.drop(
            columns=[
                "outcome",
                "outcome_numeric",
                "bet_outcome_numeric",
                "tie_rule",
                "archived_bet_outcome_numeric",
                "standard_dead_heat_bet_outcome_numeric",
                "book_rule_bet_outcome_numeric",
                "settlement_reconstruction_delta",
                "standard_dead_heat_delta_vs_archive",
                "settlement_reconstructed",
                "settlement_source",
                "archived_bet_outcome_text",
            ],
            errors="ignore",
        ).merge(
            settlement,
            on=settlement_keys,
            how="left",
            validate="one_to_one",
        )
        availability = (
            endpoint.groupby(keys, observed=True)["book"]
            .agg(
                available_book_count="nunique",
                available_books=lambda values: "|".join(sorted(set(values))),
            )
            .reset_index()
        )
        for policy in LINE_POLICIES:
            if policy in SHARP_BOOKS:
                chosen = endpoint.loc[endpoint["book"].eq(policy)].copy()
            else:
                chosen = endpoint.sort_values(
                    keys + ["execution_odds", "_book_rank", "execution_time"],
                    ascending=[True] * len(keys)
                    + [False, True, price_point == "open"],
                    kind="mergesort",
                    na_position="last",
                ).drop_duplicates(keys, keep="first")
            chosen = chosen.merge(
                availability, on=keys, how="left", validate="one_to_one"
            )
            chosen["line_policy"] = policy
            # The reusable scoring code calls the execution endpoint close_odds.
            chosen["close_odds"] = chosen["execution_odds"]
            chosen["close_time"] = chosen["execution_time"]
            implied = np.asarray(
                american_implied_probability(chosen["execution_odds"]), dtype=float
            )
            opposite_valid = v4bt._valid_american(
                chosen["same_book_opposite_odds"]
            )
            opposite_implied = np.full(len(chosen), np.nan)
            opposite_implied[opposite_valid.to_numpy()] = np.asarray(
                american_implied_probability(
                    chosen.loc[opposite_valid, "same_book_opposite_odds"]
                ),
                dtype=float,
            )
            chosen["same_book_devig_probability"] = np.where(
                chosen["market"].eq("tournament_matchup") & opposite_valid,
                implied / (implied + opposite_implied),
                np.nan,
            )
            chosen["same_book_devig_probability_role"] = np.where(
                ~chosen["market"].eq("tournament_matchup"),
                "not_applicable_outright",
                np.where(
                    chosen["tie_rule"].map(ties_lose),
                    "conditional_on_either_player_winning_attribution_only_no_draw_quote",
                    "resolved_two_way_attribution_only",
                ),
            )
            chosen["same_book_devig_proper_score_eligible"] = False
            archived_close_valid = v4bt._valid_american(
                chosen["archived_close_odds"]
            )
            chosen["execution_to_archived_close_clv_pp"] = np.nan
            if archived_close_valid.any():
                chosen.loc[
                    archived_close_valid, "execution_to_archived_close_clv_pp"
                ] = 100.0 * (
                    np.asarray(
                        american_implied_probability(
                            chosen.loc[archived_close_valid, "archived_close_odds"]
                        ),
                        dtype=float,
                    )
                    - np.asarray(
                        american_implied_probability(
                            chosen.loc[archived_close_valid, "execution_odds"]
                        ),
                        dtype=float,
                    )
                )
            chosen["cross_book_two_side_arbitrage"] = False
            chosen["cross_book_arbitrage_both_legs_push_on_tie"] = False
            chosen["cross_book_arbitrage_tie_rule_classes"] = ""
            if policy == "best_of_three":
                matchup_rows = chosen["market"].eq("tournament_matchup")
                directional = chosen.loc[
                    matchup_rows,
                    [*keys, "execution_odds", "book", "tie_rule"],
                ].copy()
                left = directional["selection_id"].astype(str)
                right = directional["opponent_id"].astype(str)
                directional["_pair_lo"] = np.where(left <= right, left, right)
                directional["_pair_hi"] = np.where(left <= right, right, left)
                directional["_push_on_tie"] = ~directional["tie_rule"].map(
                    ties_lose
                )
                directional["_tie_rule_class"] = np.where(
                    directional["_push_on_tie"], "push_on_tie", "ties_lose"
                )
                pair_hold = (
                    directional.assign(
                        _raw_implied=american_implied_probability(
                            directional["execution_odds"]
                        )
                    )
                    .groupby(
                        ["event_id", "market", "_pair_lo", "_pair_hi"],
                        observed=True,
                    )
                    .agg(
                        _direction_count=("selection_id", "size"),
                        _raw_implied_sum=("_raw_implied", "sum"),
                        _execution_book_count=("book", "nunique"),
                        _both_legs_push_on_tie=("_push_on_tie", "all"),
                        _tie_rule_classes=(
                            "_tie_rule_class",
                            lambda values: "|".join(values.astype(str)),
                        ),
                    )
                    .reset_index()
                )
                pair_hold["cross_book_two_side_arbitrage"] = (
                    pair_hold["_direction_count"].eq(2)
                    & pair_hold["_execution_book_count"].gt(1)
                    & pair_hold["_raw_implied_sum"].lt(1.0 - 1e-12)
                    & pair_hold["_both_legs_push_on_tie"].eq(True)
                )
                chosen_left = chosen["selection_id"].astype(str)
                chosen_right = chosen["opponent_id"].astype(str)
                chosen["_pair_lo"] = np.where(
                    chosen_left <= chosen_right, chosen_left, chosen_right
                )
                chosen["_pair_hi"] = np.where(
                    chosen_left <= chosen_right, chosen_right, chosen_left
                )
                chosen = chosen.merge(
                    pair_hold[
                        [
                            "event_id",
                            "market",
                            "_pair_lo",
                            "_pair_hi",
                            "cross_book_two_side_arbitrage",
                            "_both_legs_push_on_tie",
                            "_tie_rule_classes",
                        ]
                    ].rename(
                        columns={
                            "cross_book_two_side_arbitrage": "_pair_arbitrage"
                        }
                    ),
                    on=["event_id", "market", "_pair_lo", "_pair_hi"],
                    how="left",
                    validate="many_to_one",
                )
                chosen["cross_book_two_side_arbitrage"] = chosen[
                    "_pair_arbitrage"
                ].fillna(False).astype(bool)
                chosen["cross_book_arbitrage_both_legs_push_on_tie"] = chosen[
                    "_both_legs_push_on_tie"
                ].fillna(False).astype(bool)
                chosen["cross_book_arbitrage_tie_rule_classes"] = chosen[
                    "_tie_rule_classes"
                ].fillna("").astype(str)
                chosen = chosen.drop(
                    columns=[
                        "_pair_lo",
                        "_pair_hi",
                        "_pair_arbitrage",
                        "_both_legs_push_on_tie",
                        "_tie_rule_classes",
                    ],
                    errors="ignore",
                )
            chosen["opportunity_class"] = np.where(
                chosen["cross_book_two_side_arbitrage"],
                "cross_book_two_side_arbitrage_pair",
                "model_directional_opportunity",
            )
            if chosen.duplicated(keys).any():
                raise ValueError(
                    f"{price_point}/{policy} did not produce one line per logical side"
                )
            outputs[(price_point, policy)] = chosen.drop(
                columns=["_book_rank", "_record_sort"], errors="ignore"
            ).reset_index(drop=True)
    return outputs


def load_bridge(
    bridge_path: Path,
    bridge_manifest_path: Path,
    crosswalk_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load and hash-check the exact core bridge, mapped to archived odds IDs."""

    bridge_path = bridge_path.resolve()
    bridge_manifest_path = bridge_manifest_path.resolve()
    crosswalk_path = crosswalk_path.resolve()
    if bridge_manifest_path.stat().st_size != EXPECTED_BRIDGE_MANIFEST_BYTES:
        raise ValueError("Core bridge manifest byte size differs from audited authority")
    bridge_manifest_hash = v4bt.sha256_file(bridge_manifest_path)
    if bridge_manifest_hash != EXPECTED_BRIDGE_MANIFEST_SHA256:
        raise ValueError("Core bridge manifest hash differs from audited authority")
    bridge_manifest = json.loads(bridge_manifest_path.read_text(encoding="utf-8"))
    if (
        bridge_manifest.get("artifact_label")
        != "CURRENT-CODE RETROSPECTIVE LEGACY DECOMPOSITION"
        or bridge_manifest.get("purpose")
        != "research-only exact neutral-core bridge for the current-code retrospective baseline"
    ):
        raise ValueError("Core bridge manifest purpose/label is unexpected")
    reconstruction_bound = pd.to_numeric(
        bridge_manifest.get("max_abs_reconstruction_error"), errors="coerce"
    )
    if not np.isfinite(reconstruction_bound) or float(reconstruction_bound) > 1e-10:
        raise ValueError("Core bridge manifest reconstruction bound is not admissible")
    if bridge_manifest_path.name != "manifest.json":
        raise ValueError("Core bridge manifest must be the bridge directory manifest.json")
    if bridge_path.name != "legacy_core_bridge.parquet":
        raise ValueError("Core bridge must use the manifest's exact legacy_core_bridge output")
    if crosswalk_path.name != "odds_event_crosswalk.parquet":
        raise ValueError(
            "Event crosswalk must use the manifest's exact odds_event_crosswalk output"
        )
    if (
        bridge_path.parent != bridge_manifest_path.parent
        or crosswalk_path.parent != bridge_manifest_path.parent
    ):
        raise ValueError(
            "Core bridge, event crosswalk, and manifest must share one resolved parent"
        )
    outputs = bridge_manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("Core bridge manifest lacks outputs")
    if set(outputs) != set(EXPECTED_BRIDGE_OUTPUT_SHA256):
        raise ValueError("Core bridge manifest output inventory differs from audited authority")
    for name, expected_hash in EXPECTED_BRIDGE_OUTPUT_SHA256.items():
        entry = outputs.get(name)
        if not isinstance(entry, Mapping) or str(entry.get("sha256", "")).lower() != (
            expected_hash
        ):
            raise ValueError(f"Core bridge audited output hash mismatch: {name}")

    def bound_output(path: Path, name: str) -> str:
        entry = outputs.get(name)
        if not isinstance(entry, Mapping) or not entry.get("sha256"):
            raise ValueError(f"Core bridge manifest lacks exact output {name!r}")
        if entry.get("path"):
            recorded = _resolve_manifest_path(
                bridge_manifest_path, entry["path"]
            ).resolve()
            if recorded != path:
                raise ValueError(f"Core bridge manifest output path mismatch: {name}")
        expected = str(entry["sha256"]).lower()
        actual_hash = v4bt.sha256_file(path)
        if actual_hash != expected:
            raise ValueError(
                f"Core bridge manifest output hash mismatch for {name}: "
                f"expected {expected}, got {actual_hash}"
            )
        return actual_hash

    actual = bound_output(bridge_path, bridge_path.name)
    crosswalk_hash = bound_output(crosswalk_path, crosswalk_path.name)
    bridge = pd.read_parquet(bridge_path)
    required = {
        "event_key",
        "event_start",
        "year",
        "event_id",
        "dg_id",
        "legacy_core_pred_step2",
        "baseline_r1_prediction",
        "downstream_preserved_total",
        "reconstruction_error",
    }
    missing = sorted(required - set(bridge.columns))
    if missing:
        raise ValueError(f"Core bridge missing columns: {missing}")
    if bridge.duplicated(["event_key", "dg_id"]).any():
        raise ValueError("Core bridge has duplicate local event/player rows")
    reconstruction_columns = [
        "legacy_core_pred_step2",
        "baseline_r1_prediction",
        "downstream_preserved_total",
        "reconstruction_error",
    ]
    reconstruction = bridge[reconstruction_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(reconstruction.to_numpy(float)).all():
        raise ValueError("Core bridge reconstruction fields must all be finite")
    identity_error = (
        reconstruction["baseline_r1_prediction"]
        - reconstruction["legacy_core_pred_step2"]
        - reconstruction["downstream_preserved_total"]
    ).abs()
    recorded_error = reconstruction["reconstruction_error"].abs()
    if identity_error.max() > 1e-10 or recorded_error.max() > 1e-10:
        raise ValueError(
            "Core bridge downstream reconstruction mismatch: "
            f"identity={identity_error.max():.3g}, recorded={recorded_error.max():.3g}"
        )
    crosswalk = pd.read_parquet(crosswalk_path)
    crosswalk = crosswalk[
        crosswalk["odds_match_found"].fillna(False).astype(bool)
        & ~crosswalk["crosswalk_ambiguous"].fillna(True).astype(bool)
    ].copy()
    if crosswalk.duplicated("event_key").any():
        raise ValueError("Event crosswalk is not unique by local event key")
    crosswalk["odds_event_key"] = [
        v4bt.canonical_event_key(year, event_id)
        for year, event_id in crosswalk[["odds_query_year", "odds_event_id"]].itertuples(
            index=False, name=None
        )
    ]
    if crosswalk.duplicated("odds_event_key").any():
        raise ValueError("Event crosswalk is not unique by odds event key")
    bridge = bridge.rename(columns={"event_key": "local_event_key"}).merge(
        crosswalk[
            [
                "event_key",
                "odds_event_key",
                "odds_query_year",
                "odds_event_id",
                "event_id_drift",
                "crosswalk_score",
            ]
        ].rename(columns={"event_key": "local_event_key"}),
        on="local_event_key",
        how="inner",
        validate="many_to_one",
    )
    bridge["event_key"] = bridge["odds_event_key"]
    bridge["event_start"] = pd.to_datetime(bridge["event_start"]).dt.normalize()
    bridge["dg_id"] = pd.to_numeric(bridge["dg_id"], errors="raise").astype(int)
    if bridge.duplicated(["event_key", "dg_id"]).any():
        raise ValueError("Odds-mapped core bridge has duplicate event/player rows")
    audit = {
        "sha256": actual,
        "crosswalk_sha256": crosswalk_hash,
        "bridge_manifest_sha256": v4bt.sha256_file(bridge_manifest_path),
        "rows": int(len(bridge)),
        "events": int(bridge["event_key"].nunique()),
        "event_id_drift_events": int(
            crosswalk.loc[crosswalk["event_id_drift"].fillna(False), "odds_event_key"].nunique()
        ),
        "max_reconstruction_error": float(recorded_error.max()),
        "max_downstream_identity_error": float(identity_error.max()),
    }
    return bridge, crosswalk, audit


def identity_finalist_from_bridge(bridge: pd.DataFrame) -> pd.DataFrame:
    output = bridge[
        ["event_key", "event_start", "dg_id", "legacy_core_pred_step2"]
    ].rename(columns={"legacy_core_pred_step2": "finalist_core"})
    output["candidate"] = "identity_preflight_only"
    output["control_core"] = output["finalist_core"]
    output["split"] = np.where(output["event_start"].dt.year >= 2023, "sealed", "discovery")
    output["prediction_asof"] = output["event_start"]
    output["chronology_ok"] = True
    return output


def read_crosswalked_event_metadata(
    db_path: Path,
    crosswalk: pd.DataFrame,
    odds_event_keys: Iterable[str],
) -> dict[str, v4bt.EventMeta]:
    """Read physical-course metadata by local identity, then relabel to odds identity."""

    requested = set(odds_event_keys)
    mapping = crosswalk[crosswalk["odds_event_key"].isin(requested)].copy()
    if mapping.empty:
        return {}
    years = pd.to_numeric(mapping["year"], errors="raise").astype(int)
    uri = f"file:{db_path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rounds = pd.read_sql_query(
            "SELECT year, event_id, event_name, round_num, round_date, course_num "
            "FROM player_rounds WHERE tour='pga' AND year BETWEEN ? AND ?",
            connection,
            params=[int(years.min()), int(years.max())],
        )
    rounds["round_date"] = pd.to_datetime(rounds["round_date"], errors="coerce").dt.normalize()
    output: dict[str, v4bt.EventMeta] = {}
    for row in mapping.to_dict(orient="records"):
        start = pd.Timestamp(row["event_start"]).normalize()
        local = rounds[
            pd.to_numeric(rounds["year"], errors="coerce").eq(int(row["year"]))
            & pd.to_numeric(rounds["event_id"], errors="coerce").eq(int(row["event_id"]))
        ]
        r1 = local[local["round_num"].eq(1) & local["round_date"].eq(start)]
        if r1.empty:
            continue
        counts = r1["course_num"].dropna().astype(int).value_counts()
        names = r1["event_name"].dropna()
        odds_key = str(row["odds_event_key"])
        output[odds_key] = v4bt.EventMeta(
            event_key=odds_key,
            year=int(row["year"]),
            event_id=int(row["event_id"]),
            event_name=str(names.mode().iloc[0] if len(names) else ""),
            start=start,
            course_counts=tuple((int(course), int(count)) for course, count in counts.items()),
        )
    return output


def realized_weather_summaries(
    round_panel_path: Path,
    event_crosswalk: pd.DataFrame,
    odds_event_keys: Iterable[str],
) -> pd.DataFrame:
    """Build post-hoc/oracle weather descriptors; never return model features."""

    columns = [
        "year",
        "event_id",
        "round_num",
        "round_date",
        "dg_id",
        "weather_wind",
        "weather_dew",
        "weather_delta_wind",
        "weather_delta_dew",
        "weather_excluded",
    ]
    rounds = pd.read_parquet(round_panel_path, columns=columns)
    mapping = event_crosswalk[
        event_crosswalk["odds_event_key"].isin(set(odds_event_keys))
    ][["year", "event_id", "odds_event_key"]].copy()
    rounds = rounds.merge(mapping, on=["year", "event_id"], how="inner", validate="many_to_one")
    for column in ("weather_wind", "weather_dew", "weather_delta_wind", "weather_delta_dew"):
        rounds[column] = pd.to_numeric(rounds[column], errors="coerce")
    rows: list[dict[str, Any]] = []
    for event_key, event in rounds.groupby("odds_event_key", sort=False):
        wind = event["weather_wind"].dropna()
        dew = event["weather_dew"].dropna()
        wind_delta = event["weather_delta_wind"].dropna()
        dew_delta = event["weather_delta_dew"].dropna()
        round_spreads = (
            event.groupby(["round_num", "round_date"], dropna=False)
            .agg(
                wind_spread=("weather_wind", lambda values: values.max() - values.min()),
                dew_spread=("weather_dew", lambda values: values.max() - values.min()),
            )
            .reset_index()
        )

        def metric(values: pd.Series, operation: str) -> float:
            if values.empty:
                return np.nan
            if operation == "mean":
                return float(values.mean())
            if operation == "sd":
                return float(values.std(ddof=0))
            if operation == "p90_abs":
                return float(values.abs().quantile(0.90))
            if operation == "max_abs":
                return float(values.abs().max())
            raise ValueError(operation)

        rows.append(
            {
                "event_key": event_key,
                "realized_weather_rows": int(len(event)),
                "realized_weather_coverage": float(
                    event[["weather_wind", "weather_dew"]].notna().all(axis=1).mean()
                ),
                "realized_weather_excluded_rows": int(
                    event["weather_excluded"].fillna(False).astype(bool).sum()
                ),
                "realized_wind_mean": metric(wind, "mean"),
                "realized_wind_delta_sd": metric(wind_delta, "sd"),
                "realized_wind_delta_p90_abs": metric(wind_delta, "p90_abs"),
                "realized_wind_delta_max_abs": metric(wind_delta, "max_abs"),
                "realized_dew_mean": metric(dew, "mean"),
                "realized_dew_delta_sd": metric(dew_delta, "sd"),
                "realized_dew_delta_p90_abs": metric(dew_delta, "p90_abs"),
                "realized_dew_delta_max_abs": metric(dew_delta, "max_abs"),
                "realized_round_wind_spread_max": float(round_spreads["wind_spread"].max()),
                "realized_round_dew_spread_max": float(round_spreads["dew_spread"].max()),
            }
        )
    return pd.DataFrame(rows)


def actual_starter_keys(
    round_panel_path: Path,
    event_crosswalk: pd.DataFrame,
    odds_event_keys: Iterable[str],
) -> set[tuple[str, int]]:
    rounds = pd.read_parquet(
        round_panel_path, columns=["year", "event_id", "round_num", "dg_id"]
    )
    mapping = event_crosswalk[
        event_crosswalk["odds_event_key"].isin(set(odds_event_keys))
    ][["year", "event_id", "odds_event_key"]]
    rounds = rounds[rounds["round_num"].eq(1)].merge(
        mapping, on=["year", "event_id"], how="inner", validate="many_to_one"
    )
    return set(
        zip(
            rounds["odds_event_key"].astype(str),
            pd.to_numeric(rounds["dg_id"], errors="raise").astype(int),
        )
    )


def build_baseline_field(
    roster: pd.DataFrame,
    meta: v4bt.EventMeta,
    r1_predictions: pd.DataFrame,
    model_histories: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    source_crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    """Reproduce the v4 baseline roster/skill fallback without its shot-mean factor."""

    field = roster[["dg_id", "player_name"]].drop_duplicates("dg_id").copy()
    exact = r1_predictions[
        r1_predictions["year"].eq(meta.year)
        & r1_predictions["event_id"].eq(meta.event_id)
    ][["player_name", "prediction"]]
    field = field.merge(exact, on="player_name", how="left", validate="many_to_one")
    field["skill_est"] = field["prediction"]
    field["skill_source"] = np.where(field["prediction"].notna(), "r1_prediction", "")
    missing = field["skill_est"].isna()
    field.loc[missing, "skill_est"] = [
        v4bt.latest_model_skill(model_histories, dg_id, meta.start)
        for dg_id in field.loc[missing, "dg_id"]
    ]
    model_filled = missing & field["skill_est"].notna()
    field.loc[model_filled, "skill_source"] = "dated_model_fallback"
    still_missing = field["skill_est"].isna()
    finite = field.loc[~still_missing, "skill_est"]
    floor = float(finite.quantile(0.10) - 0.50) if len(finite) else -2.0
    field.loc[still_missing, "skill_est"] = floor
    field.loc[still_missing, "skill_source"] = "field_floor_fallback"
    field = field.merge(source_crosswalk, on="dg_id", how="left", validate="many_to_one")
    field["source_player_id"] = field["source_player_id"].fillna("").astype(str)
    return field.sort_values(["player_name", "dg_id"], kind="mergesort").reset_index(drop=True)


def bridge_candidate_prediction(
    baseline_r1_prediction: np.ndarray,
    legacy_core: np.ndarray,
    finalist_core: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    baseline = np.asarray(baseline_r1_prediction, dtype=float)
    legacy = np.asarray(legacy_core, dtype=float)
    finalist = np.asarray(finalist_core, dtype=float)
    delta = finalist - legacy
    return baseline + delta, delta


def bridge_candidate_prediction_with_lambda(
    baseline_r1_prediction: np.ndarray,
    legacy_core: np.ndarray,
    finalist_core: np.ndarray,
    downstream_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply ``finalist_core + lambda * legacy_downstream`` and return its delta."""

    baseline = np.asarray(baseline_r1_prediction, dtype=float)
    legacy = np.asarray(legacy_core, dtype=float)
    finalist = np.asarray(finalist_core, dtype=float)
    downstream = baseline - legacy
    full = finalist + float(downstream_lambda) * downstream
    return full, full - baseline


def attach_finalist(
    field: pd.DataFrame,
    event_bridge: pd.DataFrame,
    event_finalist: pd.DataFrame,
    downstream_lambdas: Sequence[float] = (1.0,),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach an exact replacement; uncovered roster rows remain neutral and visible."""

    finalist_columns = [
        "dg_id",
        "finalist_core",
        "control_core",
        *[column for column in ATTRIBUTION_COLUMNS if column in event_finalist],
    ]
    output = field.merge(
        event_bridge[
            [
                "dg_id",
                "legacy_core_pred_step2",
                "baseline_r1_prediction",
                "local_event_key",
                "event_id_drift",
            ]
        ],
        on="dg_id",
        how="left",
        validate="one_to_one",
    ).merge(
        event_finalist[finalist_columns],
        on="dg_id",
        how="left",
        validate="one_to_one",
    )
    output["bridge_available"] = output["legacy_core_pred_step2"].notna()
    output["finalist_available"] = output["finalist_core"].notna()
    output["control_available"] = output["control_core"].notna()
    output["exact_bridge_replacement"] = (
        output["bridge_available"] & output["finalist_available"]
    )
    output["strict_pt_control_fallback"] = (
        ~output["bridge_available"]
        & output["finalist_available"]
        & output["control_available"]
    )
    output["core_replacement_available"] = (
        output["exact_bridge_replacement"] | output["strict_pt_control_fallback"]
    )
    exact = output["exact_bridge_replacement"]
    fallback = output["strict_pt_control_fallback"]
    neutral = ~output["core_replacement_available"]
    # The archived odds roster is joined to sg_preds_resids by normalized name in
    # the v4 loader.  A handful of aliases fall through to its dated/floor fallback
    # even though the exact ID-keyed bridge exists.  Use the ID-keyed bridge as the
    # baseline authority for those rows; otherwise delta=0 would not be an identity.
    output["pre_bridge_skill_est"] = output["skill_est"]
    mismatch = (
        output.loc[output["bridge_available"], "skill_est"]
        - output.loc[output["bridge_available"], "baseline_r1_prediction"]
    ).abs()
    bridge_override = output["bridge_available"] & (
        output["skill_est"] - output["baseline_r1_prediction"]
    ).abs().gt(1e-12)
    output.loc[output["bridge_available"], "skill_est"] = output.loc[
        output["bridge_available"], "baseline_r1_prediction"
    ]
    output.loc[bridge_override, "skill_source"] = "exact_core_bridge_baseline_override"
    control_prior = pd.to_numeric(output["control_core"], errors="coerce").median()
    if not np.isfinite(control_prior):
        control_prior = float(output["skill_est"].median())
    output["comparison_control_core"] = output["control_core"]
    output["comparison_finalist_core"] = output["finalist_core"]
    output.loc[neutral, "comparison_control_core"] = control_prior
    output.loc[neutral, "comparison_finalist_core"] = control_prior
    output["replacement_path"] = np.select(
        [exact, fallback],
        ["exact_legacy_bridge", "strict_pt_v2_control_fallback"],
        default="shared_neutral_field_tour_prior",
    )
    output["candidate_full_prediction"] = output["skill_est"]
    output["core_delta"] = 0.0
    output["legacy_downstream"] = (
        output["baseline_r1_prediction"] - output["legacy_core_pred_step2"]
    )
    if exact.any():
        full, delta = bridge_candidate_prediction(
            output.loc[exact, "baseline_r1_prediction"].to_numpy(float),
            output.loc[exact, "legacy_core_pred_step2"].to_numpy(float),
            output.loc[exact, "finalist_core"].to_numpy(float),
        )
        output.loc[exact, "candidate_full_prediction"] = full
        output.loc[exact, "core_delta"] = delta
    if fallback.any():
        delta = (
            output.loc[fallback, "finalist_core"] - output.loc[fallback, "control_core"]
        ).to_numpy(float)
        output.loc[fallback, "candidate_full_prediction"] = (
            output.loc[fallback, "skill_est"].to_numpy(float) + delta
        )
        output.loc[fallback, "core_delta"] = delta
    for downstream_lambda in downstream_lambdas:
        arm = lambda_arm_name(float(downstream_lambda))
        delta_column = f"{arm}_delta"
        full_column = f"{arm}_full_prediction"
        output[delta_column] = 0.0
        output[full_column] = output["skill_est"]
        if exact.any():
            full, delta = bridge_candidate_prediction_with_lambda(
                output.loc[exact, "baseline_r1_prediction"].to_numpy(float),
                output.loc[exact, "legacy_core_pred_step2"].to_numpy(float),
                output.loc[exact, "finalist_core"].to_numpy(float),
                downstream_lambda,
            )
            output.loc[exact, full_column] = full
            output.loc[exact, delta_column] = delta
        if fallback.any():
            comparison_control = output.loc[
                fallback, "comparison_control_core"
            ].to_numpy(float)
            comparison_finalist = output.loc[
                fallback, "comparison_finalist_core"
            ].to_numpy(float)
            baseline = output.loc[fallback, "skill_est"].to_numpy(float)
            downstream = baseline - comparison_control
            full = comparison_finalist + float(downstream_lambda) * downstream
            output.loc[fallback, full_column] = full
            output.loc[fallback, delta_column] = full - baseline
        # Fully neutral rows have neither a selected nor control core.  Moving
        # them toward an event prior at lambda<1 would manufacture a challenger
        # effect unrelated to the frozen finalist, so they remain baseline in
        # every sensitivity arm.
        output.loc[neutral, full_column] = output.loc[neutral, "skill_est"]
        output.loc[neutral, delta_column] = 0.0
    lambda_one_delta = output[f"{lambda_arm_name(1.0)}_delta"]
    if not np.allclose(lambda_one_delta, output["core_delta"], atol=1e-12, rtol=0.0):
        raise AssertionError("Lambda=1 sensitivity is not the primary isolation formula")
    audit = {
        "roster_players": int(len(output)),
        "r1_prediction_players": int(output["skill_source"].eq("r1_prediction").sum()),
        "dated_model_fallback_players": int(
            output["skill_source"].eq("dated_model_fallback").sum()
        ),
        "field_floor_fallback_players": int(
            output["skill_source"].eq("field_floor_fallback").sum()
        ),
        "bridge_players": int(output["bridge_available"].sum()),
        "finalist_players": int(output["finalist_available"].sum()),
        "control_players": int(output["control_available"].sum()),
        "exact_bridge_replacement_players": int(exact.sum()),
        "strict_pt_control_fallback_players": int(fallback.sum()),
        "replacement_players": int(output["core_replacement_available"].sum()),
        "neutral_fallback_players": int(neutral.sum()),
        "replacement_coverage": float(output["core_replacement_available"].mean())
        if len(output)
        else np.nan,
        "pre_override_baseline_bridge_max_abs_error": float(mismatch.max())
        if len(mismatch)
        else 0.0,
        "exact_bridge_baseline_override_players": int(bridge_override.sum()),
        "post_override_baseline_bridge_max_abs_error": float(
            (
                output.loc[output["bridge_available"], "skill_est"]
                - output.loc[output["bridge_available"], "baseline_r1_prediction"]
            )
            .abs()
            .max()
        )
        if output["bridge_available"].any()
        else 0.0,
        "core_delta_min": float(output["core_delta"].min()),
        "core_delta_max": float(output["core_delta"].max()),
        "core_delta_mean": float(output["core_delta"].mean()),
        "legacy_downstream_min": float(output.loc[exact, "legacy_downstream"].min())
        if exact.any()
        else np.nan,
        "legacy_downstream_max": float(output.loc[exact, "legacy_downstream"].max())
        if exact.any()
        else np.nan,
        "shared_neutral_prior_core": float(control_prior),
    }
    return output, audit


def apply_core_delta_to_category_mu(
    params: dict[str, np.ndarray], delta: np.ndarray
) -> np.ndarray:
    """Shift all four category means equally and verify the row-sum identity."""

    mu = np.asarray(params["mu"], dtype=float).copy()
    shift = np.asarray(delta, dtype=float)
    if mu.ndim != 2 or mu.shape[1] != 4 or len(shift) != len(mu):
        raise ValueError("Expected an n-player by four-category mean matrix")
    before = mu.sum(axis=1)
    mu += shift[:, None] / 4.0
    if not np.allclose(mu.sum(axis=1), before + shift, atol=1e-12, rtol=0.0):
        raise AssertionError("Category means do not sum to the candidate total shift")
    return mu


def simulate_scores(
    params: dict[str, np.ndarray],
    field: pd.DataFrame,
    chol: np.ndarray,
    coefficients: dict[str, Any],
    delta: np.ndarray,
    simulations: int,
    seed: int,
    cut_line: int,
    week_latent_sd: float,
) -> np.ndarray:
    """Run the v4 kernel with common seeds and an equal four-category core shift."""

    import sims_kernel

    n_players = len(field)
    shift = np.asarray(delta, dtype=float)
    total_mean = np.ascontiguousarray(field["skill_est"].to_numpy(float) + shift)
    mu = apply_core_delta_to_category_mu(params, shift)
    common = (
        total_mean,
        total_mean,
        total_mean,
        total_mean,
        np.zeros(n_players, dtype=float),
        np.zeros(n_players, dtype=float),
        v4bt._r1_coefficients(coefficients, "coefficients_r1_high"),
        v4bt._r1_coefficients(coefficients, "coefficients_r1_midh"),
        v4bt._r1_coefficients(coefficients, "coefficients_r1_midl"),
        v4bt._r1_coefficients(coefficients, "coefficients_r1_low"),
        v4bt._r2_coefficients(coefficients, "coefficients_r2"),
        v4bt._r2_coefficients(coefficients, "coefficients_r2_6_30"),
        v4bt._r2_coefficients(coefficients, "coefficients_r2_30_up"),
        v4bt._r3_coefficients(coefficients, "coefficients_r3"),
        v4bt._r3_coefficients(coefficients, "coefficients_r3_mid"),
        v4bt._r3_coefficients(coefficients, "coefficients_r3_high"),
        int(cut_line),
        False,
        int(simulations),
        int(seed),
        float(week_latent_sd),
    )
    result = sims_kernel.run_pretournament(
        np.ascontiguousarray(mu),
        np.ascontiguousarray(params["std_shot"]),
        np.ascontiguousarray(params["skew"]),
        np.ascontiguousarray(chol),
        *common,
    )
    return np.ascontiguousarray(result[0]).astype(np.int64, copy=False)


def finish_probability_variants(
    scores: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    """Return finish probabilities under each supported outright payout rule.

    ``standard_dead_heat`` is expected paid share: qualification probability is
    reduced when a boundary tie spans paid and unpaid positions.
    ``full_tie_qualification`` counts the player in full whenever the player's
    minimum rank is inside the threshold.  Pinnacle Top 5/10/20 uses the latter;
    BetOnline and Betcris use the former.  Win keeps its existing split-first
    dead-heat expectation under both labels.
    """

    import sims_kernel

    final_scores = np.ascontiguousarray(scores, dtype=np.int64)
    if final_scores.ndim != 2 or final_scores.shape[0] == 0 or final_scores.shape[1] == 0:
        raise ValueError("scores must be a nonempty player-by-simulation matrix")
    minimum = final_scores.min(axis=0)
    tied = final_scores == minimum[None, :]
    win = (tied / tied.sum(axis=0, keepdims=True)).mean(axis=1)
    _, top_dead_heat, top_no_dead_heat = sims_kernel.aggregate_round(final_scores)
    standard_top = np.asarray(top_dead_heat, dtype=float)
    full_tie_top = np.asarray(top_no_dead_heat, dtype=float)
    expected_shape = (final_scores.shape[0], 3)
    if standard_top.shape != expected_shape or full_tie_top.shape != expected_shape:
        raise ValueError(
            "aggregate_round returned unexpected Top 5/10/20 shapes: "
            f"standard={standard_top.shape}, full_tie={full_tie_top.shape}, "
            f"expected={expected_shape}"
        )
    for label, values in (
        ("win", win),
        ("standard_dead_heat", standard_top),
        ("full_tie_qualification", full_tie_top),
    ):
        if not np.isfinite(values).all() or (values < -1e-12).any() or (values > 1.0 + 1e-12).any():
            raise ValueError(f"{label} finish probabilities fall outside [0, 1]")
    if (standard_top > full_tie_top + 1e-12).any():
        raise ValueError("standard dead-heat share exceeds full-tie qualification probability")
    standard = {
        "win": win,
        "top_5": standard_top[:, 0],
        "top_10": standard_top[:, 1],
        "top_20": standard_top[:, 2],
    }
    full_tie = {
        "win": win,
        "top_5": full_tie_top[:, 0],
        "top_10": full_tie_top[:, 1],
        "top_20": full_tie_top[:, 2],
    }
    return {
        "standard_dead_heat": standard,
        "full_tie_qualification": full_tie,
    }


def probabilities_for_finish_lines(
    scores: np.ndarray,
    field: pd.DataFrame,
    event_lines: pd.DataFrame,
    event_key: str,
    event_start: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build the logical market menu while retaining both top-finish variants."""

    player_ids = field["dg_id"].map(v4bt.canonical_player_id).tolist()
    player_names = field["player_name"].tolist()
    index = {player_id: offset for offset, player_id in enumerate(player_ids)}
    finishes = finish_probability_variants(scores)
    rows: list[dict[str, Any]] = []
    missing = 0
    opportunities = event_lines[v4bt.KEY_COLS].drop_duplicates()
    for line in opportunities.to_dict(orient="records"):
        selection = v4bt.canonical_player_id(line["selection_id"])
        opponent = v4bt.canonical_player_id(line.get("opponent_id"))
        if selection not in index:
            missing += 1
            continue
        market = str(line["market"])
        if market == "tournament_matchup":
            if opponent not in index:
                missing += 1
                continue
            left = scores[index[selection]]
            right = scores[index[opponent]]
            probability = float(np.mean(left < right))
            push = float(np.mean(left == right))
            standard_probability = probability
            full_tie_probability = probability
        else:
            if market not in OUTRIGHT_MARKET_CUTOFF:
                raise ValueError(f"Unsupported outright market: {market}")
            standard_probability = float(
                finishes["standard_dead_heat"][market][index[selection]]
            )
            full_tie_probability = float(
                finishes["full_tie_qualification"][market][index[selection]]
            )
            # The persisted logical menu is book-neutral.  Keep the historical
            # expected-paid-share value as its default and resolve the executable
            # book-specific value only after a fixed/best-three line is selected.
            probability = standard_probability
            push = 0.0
        rows.append(
            {
                "event_id": event_key,
                "event_key": event_key,
                "event_start": event_start,
                "market": market,
                "selection_id": selection,
                "selection_name": player_names[index[selection]],
                "opponent_id": opponent,
                "probability": probability,
                STANDARD_DEAD_HEAT_PROBABILITY: standard_probability,
                FULL_TIE_QUALIFICATION_PROBABILITY: full_tie_probability,
                "push_probability": push,
            }
        )
    output = pd.DataFrame(rows)
    top = output[output["market"].isin(TOP_FINISH_MARKETS)] if not output.empty else output
    return output, {
        "probability_opportunities": int(len(opportunities)),
        "probability_rows": int(len(rows)),
        "market_players_missing_from_roster": int(missing),
        "top_finish_probability_rows": int(len(top)),
        "top_finish_rule_variant_rows": int(
            (
                top[FULL_TIE_QUALIFICATION_PROBABILITY]
                - top[STANDARD_DEAD_HEAT_PROBABILITY]
            )
            .abs()
            .gt(1e-12)
            .sum()
        )
        if not top.empty
        else 0,
    }


def assert_common_market_menus(probabilities: pd.DataFrame) -> None:
    reference: pd.MultiIndex | None = None
    for arm, frame in probabilities.groupby("arm", sort=False, observed=True):
        if frame.duplicated(PROBABILITY_KEYS).any():
            raise ValueError(f"{arm} probabilities contain duplicate logical keys")
        menu = pd.MultiIndex.from_frame(frame[PROBABILITY_KEYS]).sort_values()
        if reference is None:
            reference = menu
        elif not menu.equals(reference):
            raise ValueError(f"Probability market menu differs for arm {arm}")


def assert_matchup_probability_complements(probabilities: pd.DataFrame) -> None:
    matchup = probabilities[probabilities["market"].eq("tournament_matchup")].copy()
    if matchup.empty:
        return
    reverse = matchup.rename(
        columns={
            "selection_id": "opponent_id",
            "opponent_id": "selection_id",
            "probability": "reverse_probability",
            "push_probability": "reverse_push_probability",
        }
    )
    keys = ["arm", "event_id", "market", "selection_id", "opponent_id"]
    joined = matchup.merge(
        reverse[keys + ["reverse_probability", "reverse_push_probability"]],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if joined.empty:
        return
    if not np.allclose(
        joined["push_probability"], joined["reverse_push_probability"], atol=1e-12, rtol=0.0
    ):
        raise ValueError("Reverse matchup push probabilities disagree")
    total = (
        joined["probability"]
        + joined["reverse_probability"]
        + joined["push_probability"]
    )
    if not np.allclose(total, 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("Reverse matchup probabilities do not complement with pushes")


def resolve_book_rule_probabilities(
    lines: pd.DataFrame,
    probabilities: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve book-neutral probabilities to the selected executable book rule."""

    annotated_lines = lines.copy()
    resolved = probabilities.copy()
    if annotated_lines.empty:
        annotated_lines["model_probability_rule"] = pd.Series(dtype=str)
        return annotated_lines, resolved
    keys = list(v4bt.KEY_COLS)
    missing_lines = [column for column in keys + ["book"] if column not in annotated_lines]
    if missing_lines:
        raise ValueError(f"executable lines lack probability-rule columns: {missing_lines}")
    missing_probabilities = [column for column in keys + ["probability"] if column not in resolved]
    if missing_probabilities:
        raise ValueError(f"logical probabilities lack required columns: {missing_probabilities}")
    if annotated_lines.duplicated(keys).any():
        raise ValueError("executable line menu contains duplicate logical probability keys")
    if resolved.duplicated(keys).any():
        raise ValueError("logical probability panel contains duplicate keys")

    annotated_lines["book"] = annotated_lines["book"].astype(str).str.strip().str.lower()
    top_line = annotated_lines["market"].isin(TOP_FINISH_MARKETS)
    unsupported_top_book = top_line & ~annotated_lines["book"].isin(SHARP_BOOKS)
    if unsupported_top_book.any():
        books = sorted(annotated_lines.loc[unsupported_top_book, "book"].unique())
        raise ValueError(f"Top-finish line has unsupported book probability rule: {books}")
    pinnacle_top = top_line & annotated_lines["book"].eq("pinnacle")
    annotated_lines["model_probability_rule"] = np.select(
        [
            pinnacle_top,
            top_line,
            annotated_lines["market"].eq("win"),
            annotated_lines["market"].eq("tournament_matchup"),
        ],
        [
            "pinnacle_full_tie_qualification_top_finish",
            "standard_dead_heat_expected_paid_share_top_finish",
            "win_split_first_place_dead_heat_share",
            "matchup_win_with_explicit_push_mass",
        ],
        default="unsupported",
    )
    if annotated_lines["model_probability_rule"].eq("unsupported").any():
        markets = sorted(
            annotated_lines.loc[
                annotated_lines["model_probability_rule"].eq("unsupported"), "market"
            ].astype(str).unique()
        )
        raise ValueError(f"Unsupported market probability rule: {markets}")

    if top_line.any():
        required_variants = [
            STANDARD_DEAD_HEAT_PROBABILITY,
            FULL_TIE_QUALIFICATION_PROBABILITY,
        ]
        missing_variants = [column for column in required_variants if column not in resolved]
        if missing_variants:
            raise ValueError(
                "Top-finish probabilities lack native book-rule variants: "
                f"{missing_variants}"
            )
    menu = annotated_lines[keys + ["book", "model_probability_rule"]]
    resolved = resolved.merge(menu, on=keys, how="inner", validate="one_to_one")
    if len(resolved) != len(annotated_lines):
        raise ValueError(
            "Executable line menu and logical probability panel do not close: "
            f"lines={len(annotated_lines)}, matched={len(resolved)}"
        )
    resolved_top = resolved["market"].isin(TOP_FINISH_MARKETS)
    resolved_pinnacle_top = resolved_top & resolved["book"].eq("pinnacle")
    if resolved_top.any():
        standard = pd.to_numeric(
            resolved[STANDARD_DEAD_HEAT_PROBABILITY], errors="raise"
        ).to_numpy(float)
        full_tie = pd.to_numeric(
            resolved[FULL_TIE_QUALIFICATION_PROBABILITY], errors="raise"
        ).to_numpy(float)
        chosen = np.where(resolved_pinnacle_top.to_numpy(), full_tie, standard)
        resolved["probability"] = np.where(
            resolved_top.to_numpy(), chosen, resolved["probability"].to_numpy(float)
        )
        if (
            resolved.loc[resolved_top, STANDARD_DEAD_HEAT_PROBABILITY]
            .gt(
                resolved.loc[
                    resolved_top, FULL_TIE_QUALIFICATION_PROBABILITY
                ]
                + 1e-12
            )
            .any()
        ):
            raise ValueError("standard dead-heat probability exceeds full qualification")

    diagnostic_columns = [
        column
        for column in (
            STANDARD_DEAD_HEAT_PROBABILITY,
            FULL_TIE_QUALIFICATION_PROBABILITY,
        )
        if column in resolved
    ]
    if diagnostic_columns:
        annotated_lines = annotated_lines.merge(
            resolved[keys + diagnostic_columns],
            on=keys,
            how="left",
            validate="one_to_one",
        )
    return annotated_lines, resolved.drop(columns=["book", "model_probability_rule"])


def book_rule_probability_selection_audit(frame: pd.DataFrame) -> dict[str, Any]:
    """Fail closed unless persisted executable probabilities match their book."""

    required = {
        "market",
        "book",
        "probability",
        "model_probability_rule",
        STANDARD_DEAD_HEAT_PROBABILITY,
        FULL_TIE_QUALIFICATION_PROBABILITY,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"book-rule probability audit lacks columns: {missing}")
    logical_columns = [
        column
        for column in (
            "challenger_name",
            "line_policy_name",
            "price_point",
            "sizing_name",
            "arm",
            *v4bt.KEY_COLS,
            "book",
            "probability",
            "model_probability_rule",
            STANDARD_DEAD_HEAT_PROBABILITY,
            FULL_TIE_QUALIFICATION_PROBABILITY,
        )
        if column in frame.columns
    ]
    unique = frame[logical_columns].drop_duplicates().copy()
    unique["book"] = unique["book"].astype(str).str.strip().str.lower()
    top = unique["market"].isin(TOP_FINISH_MARKETS)
    pinnacle = unique["book"].eq("pinnacle")
    expected = np.where(
        (top & pinnacle).to_numpy(),
        unique[FULL_TIE_QUALIFICATION_PROBABILITY].to_numpy(float),
        np.where(
            top.to_numpy(),
            unique[STANDARD_DEAD_HEAT_PROBABILITY].to_numpy(float),
            unique["probability"].to_numpy(float),
        ),
    )
    error = np.abs(unique["probability"].to_numpy(float) - expected)
    expected_rule = np.select(
        [
            top & pinnacle,
            top,
            unique["market"].eq("win"),
            unique["market"].eq("tournament_matchup"),
        ],
        [
            "pinnacle_full_tie_qualification_top_finish",
            "standard_dead_heat_expected_paid_share_top_finish",
            "win_split_first_place_dead_heat_share",
            "matchup_win_with_explicit_push_mass",
        ],
        default="unsupported",
    )
    label_mismatch = unique["model_probability_rule"].astype(str).ne(expected_rule)
    if (error > 1e-12).any() or label_mismatch.any():
        raise ValueError(
            "Persisted probability does not match executable book rule: "
            f"numeric_mismatches={int((error > 1e-12).sum())}, "
            f"label_mismatches={int(label_mismatch.sum())}"
        )
    top_unique = unique.loc[top].copy()
    uplift = (
        top_unique[FULL_TIE_QUALIFICATION_PROBABILITY]
        - top_unique[STANDARD_DEAD_HEAT_PROBABILITY]
    )
    return {
        "status": "pass",
        "unique_executable_arm_menu_rows": int(len(unique)),
        "top_finish_rows": int(top.sum()),
        "pinnacle_top_finish_rows": int((top & pinnacle).sum()),
        "standard_dead_heat_top_finish_rows": int((top & ~pinnacle).sum()),
        "top_finish_variant_difference_rows": int(uplift.abs().gt(1e-12).sum()),
        "top_finish_full_tie_uplift_mean": float(uplift.mean()) if len(uplift) else 0.0,
        "top_finish_full_tie_uplift_max": float(uplift.max()) if len(uplift) else 0.0,
        "max_selected_probability_rule_abs_error": float(error.max()) if len(error) else 0.0,
        "probability_rule_counts": unique["model_probability_rule"]
        .value_counts()
        .sort_index()
        .to_dict(),
    }


def annotate_line_replacement_paths(
    lines: pd.DataFrame,
    prepared_fields: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Label every offered line without removing players or opportunities."""

    path_map = {
        (event_key, v4bt.canonical_player_id(dg_id)): str(path)
        for event_key, field in prepared_fields.items()
        for dg_id, path in field[["dg_id", "replacement_path"]].itertuples(
            index=False, name=None
        )
    }
    output = lines.copy()
    output["selection_replacement_path"] = [
        path_map.get((str(event), v4bt.canonical_player_id(player)), "missing_roster")
        for event, player in output[["event_key", "selection_id"]].itertuples(
            index=False, name=None
        )
    ]
    output["opponent_replacement_path"] = [
        path_map.get((str(event), v4bt.canonical_player_id(player)), "missing_roster")
        if v4bt.canonical_player_id(player)
        else "not_applicable"
        for event, player in output[["event_key", "opponent_id"]].itertuples(
            index=False, name=None
        )
    ]
    attribute_targets = {
        "skill_est": "base_skill",
        "core_delta": "core_delta",
        "history_rounds": "history_rounds",
        "days_since_last_round": "days_since_last_round",
        "reliability_ok": "reliability_ok",
        "cross_tour_fallback_reason": "cross_tour_fallback_reason",
        "img_history_rounds": "img_history_rounds",
        "img_history_coverage": "img_history_coverage",
        "prior_event_withdrawal": "prior_event_withdrawal",
    }
    for source, target in attribute_targets.items():
        values = {
            (event_key, v4bt.canonical_player_id(row["dg_id"])): row.get(source, np.nan)
            for event_key, field in prepared_fields.items()
            for row in field.to_dict(orient="records")
        }
        output[f"selection_{target}"] = [
            values.get((str(event), v4bt.canonical_player_id(player)), np.nan)
            for event, player in output[["event_key", "selection_id"]].itertuples(
                index=False, name=None
            )
        ]
        output[f"opponent_{target}"] = [
            values.get((str(event), v4bt.canonical_player_id(player)), np.nan)
            if v4bt.canonical_player_id(player)
            else np.nan
            for event, player in output[["event_key", "opponent_id"]].itertuples(
                index=False, name=None
            )
        ]

    def combined(row: pd.Series) -> str:
        values = {row["selection_replacement_path"]}
        if row["market"] == "tournament_matchup":
            values.add(row["opponent_replacement_path"])
        if "missing_roster" in values:
            return "missing_roster"
        if "shared_neutral_field_tour_prior" in values:
            return "shared_neutral_field_tour_prior"
        if "strict_pt_v2_control_fallback" in values:
            return "strict_pt_v2_control_fallback"
        return "exact_legacy_bridge"

    output["replacement_path"] = output.apply(combined, axis=1)
    return output


def replacement_path_performance(bets: pd.DataFrame) -> pd.DataFrame:
    if bets.empty or "replacement_path" not in bets:
        return pd.DataFrame()
    output = (
        bets[bets["pnl"].notna()]
        .groupby(
            [
                "challenger_name",
                "line_policy_name",
                "sizing_name",
                "arm",
                "market_group",
                "replacement_path",
            ],
            dropna=False,
            observed=True,
        )
        .agg(
            bet_count=("pnl", "size"),
            event_count=("event_id", "nunique"),
            stake=("stake", "sum"),
            pnl=("pnl", "sum"),
        )
        .reset_index()
    )
    output["roi"] = np.divide(
        output["pnl"],
        output["stake"],
        out=np.full(len(output), np.nan),
        where=output["stake"].to_numpy(float) > 0,
    )
    return output


def _bool_bucket(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes", "y"}:
            return "yes"
        if normalized in {"false", "0", "no", "n", ""}:
            return "no"
        return "unknown"
    return "yes" if bool(value) else "no"


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column] if column in frame else pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(values, errors="coerce")


def finish_bet_bucket_performance(bets: pd.DataFrame) -> pd.DataFrame:
    """Year/book/skill/coverage/weather/WD/edge attribution for settled bets."""

    if bets.empty:
        return pd.DataFrame()
    frame = bets.loc[bets["pnl"].notna()].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["year_bucket"] = pd.to_datetime(
        frame["event_start"], errors="coerce"
    ).dt.year.astype("Int64").astype(str)
    frame["edge_bucket"] = pd.cut(
        pd.to_numeric(frame["ev_per_unit"], errors="coerce"),
        bins=[-np.inf, 0.0, 0.03, 0.05, 0.10, np.inf],
        labels=["<0%", "0-3%", "3-5%", "5-10%", ">10%"],
        right=False,
    ).astype(str)
    frame["skill_bucket"] = pd.cut(
        _numeric_series(frame, "selection_base_skill"),
        bins=[-np.inf, 0.0, 0.5, 1.0, 1.5, np.inf],
        labels=["<0", "0-0.5", "0.5-1.0", "1.0-1.5", ">=1.5"],
        right=False,
    ).astype(str)
    skill_gap = _numeric_series(frame, "selection_base_skill") - _numeric_series(
        frame, "opponent_base_skill"
    )
    frame["skill_gap_bucket"] = pd.cut(
        skill_gap,
        bins=[-np.inf, -0.5, -0.2, 0.0, 0.2, 0.5, np.inf],
        labels=["<-0.5", "-0.5--0.2", "-0.2-0", "0-0.2", "0.2-0.5", ">=0.5"],
        right=False,
    ).astype(str)
    frame["history_bucket"] = pd.cut(
        _numeric_series(frame, "selection_history_rounds"),
        bins=[-np.inf, 5, 10, 20, 50, np.inf],
        labels=["<5", "5-9", "10-19", "20-49", ">=50"],
        right=False,
    ).astype(str)
    frame["img_coverage_bucket"] = pd.cut(
        _numeric_series(frame, "selection_img_history_coverage"),
        bins=[-np.inf, 0.01, 0.25, 0.50, 0.75, np.inf],
        labels=["none", "<25%", "25-50%", "50-75%", ">=75%"],
        right=False,
    ).astype(str)
    frame["reliability_bucket"] = frame.get(
        "selection_reliability_ok", pd.Series(np.nan, index=frame.index)
    ).map(_bool_bucket)
    frame["prior_withdrawal_bucket"] = frame.get(
        "selection_prior_event_withdrawal", pd.Series(np.nan, index=frame.index)
    ).map(_bool_bucket)
    frame["wind_level_bucket"] = pd.cut(
        _numeric_series(frame, "realized_wind_mean"),
        bins=[-np.inf, 5, 10, 15, np.inf],
        labels=["<5mph", "5-10mph", "10-15mph", ">=15mph"],
        right=False,
    ).astype(str)
    dimensions = {
        "edge": "edge_bucket",
        "year": "year_bucket",
        "selected_book": "book",
        "projected_skill": "skill_bucket",
        "projected_skill_gap": "skill_gap_bucket",
        "replacement_coverage": "replacement_path",
        "history_round_coverage": "history_bucket",
        "img_history_coverage": "img_coverage_bucket",
        "reliability": "reliability_bucket",
        "cross_tour_fallback_reason": "selection_cross_tour_fallback_reason",
        "prior_event_withdrawal": "prior_withdrawal_bucket",
        "posthoc_realized_wind_level": "wind_level_bucket",
        "opportunity_class": "opportunity_class",
        "market": "market",
    }
    base_keys = [
        "challenger_name",
        "line_policy_name",
        "price_point",
        "sizing_name",
        "arm",
        "market_group",
    ]
    rows: list[dict[str, Any]] = []
    for dimension, column in dimensions.items():
        if column not in frame:
            continue
        for keys, group in frame.groupby(
            [*base_keys, column], dropna=False, observed=True
        ):
            *base_values, bucket = keys
            stake = float(group["stake"].sum())
            pnl = float(group["pnl"].sum())
            rows.append(
                {
                    **dict(zip(base_keys, base_values)),
                    "bucket_dimension": dimension,
                    "bucket": str(bucket),
                    "bets": int(len(group)),
                    "events": int(group["event_id"].nunique()),
                    "stake": stake,
                    "pnl": pnl,
                    "roi": pnl / stake if stake else np.nan,
                    "mean_ev": float(group["ev_per_unit"].mean()),
                }
            )
    return pd.DataFrame(rows)


def year_book_performance(bets: pd.DataFrame) -> pd.DataFrame:
    if bets.empty:
        return pd.DataFrame()
    frame = bets.loc[bets["pnl"].notna()].copy()
    frame["year"] = pd.to_datetime(frame["event_start"], errors="coerce").dt.year
    keys = [
        "challenger_name",
        "line_policy_name",
        "price_point",
        "sizing_name",
        "arm",
        "market_group",
        "year",
        "book",
        "opportunity_class",
    ]
    output = (
        frame.groupby(keys, dropna=False, observed=True)
        .agg(
            bets=("pnl", "size"),
            events=("event_id", "nunique"),
            stake=("stake", "sum"),
            pnl=("pnl", "sum"),
            mean_ev=("ev_per_unit", "mean"),
        )
        .reset_index()
    )
    output["roi"] = np.divide(
        output["pnl"],
        output["stake"],
        out=np.full(len(output), np.nan),
        where=output["stake"].to_numpy(float) > 0,
    )
    return output


def cross_book_matchup_arbitrage_audit(best_lines: pd.DataFrame) -> pd.DataFrame:
    """Report only guaranteed two-leg arbitrage in shopped matchup prices.

    A raw implied sum below one is not a guaranteed arbitrage when either leg
    loses on a physical tie.  Both archived directions must therefore carry an
    explicit push-on-tie rule before the pair is classified as arbitrage.
    """

    matchup = best_lines[best_lines["market"].eq("tournament_matchup")].copy()
    if matchup.empty:
        return pd.DataFrame()
    matchup["selection_id"] = matchup["selection_id"].astype(str)
    matchup["opponent_id"] = matchup["opponent_id"].astype(str)
    matchup["pair_lo"] = np.where(
        matchup["selection_id"] <= matchup["opponent_id"],
        matchup["selection_id"],
        matchup["opponent_id"],
    )
    matchup["pair_hi"] = np.where(
        matchup["selection_id"] <= matchup["opponent_id"],
        matchup["opponent_id"],
        matchup["selection_id"],
    )
    if "price_point" not in matchup:
        matchup["price_point"] = "unspecified"
    left = matchup[matchup["selection_id"].eq(matchup["pair_lo"])].copy()
    right = matchup[matchup["selection_id"].eq(matchup["pair_hi"])].copy()
    keys = ["event_key", "price_point", "pair_lo", "pair_hi"]
    pairs = left.merge(
        right,
        on=keys,
        how="inner",
        suffixes=("_lo", "_hi"),
        validate="one_to_one",
    )
    pairs["raw_implied_sum"] = american_implied_probability(
        pairs["close_odds_lo"]
    ) + american_implied_probability(pairs["close_odds_hi"])
    for suffix in ("lo", "hi"):
        tie_column = f"tie_rule_{suffix}"
        if tie_column in pairs:
            tie_rule = pairs[tie_column].fillna("").astype(str).str.strip()
        else:
            tie_rule = pd.Series("", index=pairs.index, dtype=str)
        pairs[f"tie_rule_class_{suffix}"] = np.where(
            tie_rule.eq(""),
            "unknown",
            np.where(tie_rule.map(ties_lose), "ties_lose", "push_on_tie"),
        )
    pairs["both_legs_push_on_tie"] = (
        pairs["tie_rule_class_lo"].eq("push_on_tie")
        & pairs["tie_rule_class_hi"].eq("push_on_tie")
    )
    pairs["tie_rule_classes"] = (
        pairs["tie_rule_class_lo"] + "|" + pairs["tie_rule_class_hi"]
    )
    pairs["arbitrage"] = (
        pairs["book_lo"].ne(pairs["book_hi"])
        & pairs["raw_implied_sum"].lt(1.0 - 1e-12)
        & pairs["both_legs_push_on_tie"]
    )
    pairs["theoretical_hold"] = 1.0 - pairs["raw_implied_sum"]
    pairs["opportunity_class"] = np.where(
        pairs["arbitrage"],
        "cross_book_two_side_arbitrage_pair",
        "model_directional_opportunity",
    )
    columns = [
        "event_key",
        "price_point",
        "pair_lo",
        "pair_hi",
        "selection_name_lo",
        "selection_name_hi",
        "book_lo",
        "book_hi",
        "close_odds_lo",
        "close_odds_hi",
        "same_book_opposite_odds_lo",
        "same_book_opposite_odds_hi",
        "raw_implied_sum",
        "theoretical_hold",
        "tie_rule_class_lo",
        "tie_rule_class_hi",
        "tie_rule_classes",
        "both_legs_push_on_tie",
        "arbitrage",
        "opportunity_class",
    ]
    return pairs[[column for column in columns if column in pairs]].sort_values(
        ["arbitrage", "theoretical_hold"], ascending=[False, False], kind="mergesort"
    )


def _one_matchup_direction(bets: pd.DataFrame) -> pd.DataFrame:
    if bets.empty:
        return bets
    output = bets.copy()
    left = output["selection_id"].astype(str)
    right = output["opponent_id"].astype(str)
    output["_pair_lo"] = np.where(left <= right, left, right)
    output["_pair_hi"] = np.where(left <= right, right, left)
    output = output.sort_values(
        ["event_id", "_pair_lo", "_pair_hi", "ev_per_unit", "selection_id"],
        ascending=[True, True, True, False, True],
        kind="mergesort",
    )
    output = output.drop_duplicates(["event_id", "_pair_lo", "_pair_hi"], keep="first")
    return output.drop(columns=["_pair_lo", "_pair_hi"]).reset_index(drop=True)


def assert_one_matchup_direction_per_pair(bets: pd.DataFrame) -> None:
    """Reject any portfolio that credits both H2H directions as model wagers."""

    if bets.empty:
        return
    matchup = bets.loc[bets["market"].eq("tournament_matchup")].copy()
    if matchup.empty:
        return
    left = matchup["selection_id"].astype(str)
    right = matchup["opponent_id"].astype(str)
    matchup["_pair_lo"] = np.where(left <= right, left, right)
    matchup["_pair_hi"] = np.where(left <= right, right, left)
    keys = [
        column
        for column in (
            "challenger_name",
            "line_policy_name",
            "price_point",
            "sizing_name",
            "arm",
            "event_id",
            "_pair_lo",
            "_pair_hi",
        )
        if column in matchup
    ]
    if matchup.duplicated(keys).any():
        raise AssertionError("portfolio contains both directions of a canonical H2H pair")


def select_family_bets(
    lines: pd.DataFrame,
    probabilities: pd.DataFrame,
    arm: str,
    market_group: str,
    sizing_name: str,
) -> pd.DataFrame:
    rule_lines, rule_probabilities = resolve_book_rule_probabilities(lines, probabilities)
    selected = v4bt.select_family_bets(
        rule_lines, rule_probabilities, arm, market_group, sizing_name
    )
    return _one_matchup_direction(selected) if market_group == "matchup" else selected


def select_all_offered_family_bets(
    lines: pd.DataFrame,
    probabilities: pd.DataFrame,
    arm: str,
    market_group: str,
) -> pd.DataFrame:
    """Settle one canonical direction at every offered EV, for edge diagnostics."""

    matchup = market_group == "matchup"
    rule_lines, rule_probabilities = resolve_book_rule_probabilities(lines, probabilities)
    gate = GateConfig(
        min_ev=-np.inf,
        min_probability_edge=-1.0,
        allowed_markets=("tournament_matchup",) if matchup else v4bt.MARKETS[1:],
    )
    selected = select_arm_bets(
        rule_lines,
        rule_probabilities,
        arm,
        gate,
        SizingConfig(method="flat", flat_stake=1.0),
        v4bt.KEY_COLS,
    )
    return _one_matchup_direction(selected) if matchup else selected


def add_complete_zero_bet_event_clusters(
    bets: pd.DataFrame,
    lines: pd.DataFrame,
    arms: Sequence[str] = ("baseline", "challenger"),
) -> pd.DataFrame:
    """Represent every eligible event for every arm, including a zero-bet arm."""

    if lines.empty:
        return bets
    universe = lines[["event_id", "event_start"]].drop_duplicates("event_id")
    present = set(
        zip(
            bets.get("event_id", pd.Series(dtype=str)).astype(str),
            bets.get("arm", pd.Series(dtype=str)).astype(str),
        )
    )
    rows: list[dict[str, Any]] = []
    for event_id, event_start in universe.itertuples(index=False, name=None):
        for arm in arms:
            if (str(event_id), str(arm)) not in present:
                rows.append(
                    {
                        "event_id": event_id,
                        "event_start": event_start,
                        "arm": str(arm),
                        "stake": 0.0,
                        "pnl": 0.0,
                        "is_zero_bet_event_cluster": True,
                    }
                )
    output = pd.concat([bets, pd.DataFrame(rows)], ignore_index=True, sort=False)
    if "is_zero_bet_event_cluster" not in output:
        output["is_zero_bet_event_cluster"] = False
    else:
        values = output["is_zero_bet_event_cluster"]
        output["is_zero_bet_event_cluster"] = np.where(
            values.isna(), False, values
        ).astype(bool)
    return output


def score_comparison(
    lines: pd.DataFrame,
    baseline_probabilities: pd.DataFrame,
    finalist_probabilities: pd.DataFrame,
    finalist_name: str,
    line_policy: str,
    price_point: str,
    sizing_name: str,
    settlement_scenario: str,
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> dict[str, pd.DataFrame]:
    bet_pieces: list[pd.DataFrame] = []
    all_offered_pieces: list[pd.DataFrame] = []
    attribution_pieces: list[pd.DataFrame] = []
    for market_group, mask in (
        ("matchup", lines["market"].eq("tournament_matchup")),
        ("outright", ~lines["market"].eq("tournament_matchup")),
    ):
        family_lines = lines.loc[mask].copy()
        baseline = select_family_bets(
            family_lines, baseline_probabilities, "baseline", market_group, sizing_name
        )
        finalist = select_family_bets(
            family_lines, finalist_probabilities, "challenger", market_group, sizing_name
        )
        all_offered_baseline = select_all_offered_family_bets(
            family_lines, baseline_probabilities, "baseline", market_group
        )
        all_offered_finalist = select_all_offered_family_bets(
            family_lines, finalist_probabilities, "challenger", market_group
        )
        for frame in (baseline, finalist):
            frame["market_group"] = market_group
        for frame in (all_offered_baseline, all_offered_finalist):
            frame["market_group"] = market_group
        attribution = compare_arm_bets(baseline, finalist, v4bt.KEY_COLS)
        attribution["market_group"] = market_group
        bet_pieces.extend([baseline, finalist])
        all_offered_pieces.extend([all_offered_baseline, all_offered_finalist])
        attribution_pieces.append(attribution)
    bets = pd.concat(bet_pieces, ignore_index=True, sort=False)
    all_offered = pd.concat(all_offered_pieces, ignore_index=True, sort=False)
    attribution = pd.concat(attribution_pieces, ignore_index=True, sort=False)
    for frame in (bets, all_offered, attribution):
        frame["challenger_name"] = finalist_name
        frame["line_policy_name"] = line_policy
        frame["price_point"] = price_point
        frame["sizing_name"] = sizing_name
        frame["settlement_scenario"] = settlement_scenario
    assert_one_matchup_direction_per_pair(bets)
    assert_one_matchup_direction_per_pair(all_offered)
    performance_parts: list[pd.DataFrame] = []
    bootstrap_parts: list[pd.DataFrame] = []
    arbitrage = bets.get(
        "cross_book_two_side_arbitrage", pd.Series(False, index=bets.index)
    ).fillna(False).astype(bool)
    line_arbitrage = lines.get(
        "cross_book_two_side_arbitrage", pd.Series(False, index=lines.index)
    ).fillna(False).astype(bool)
    for group_name, group, group_lines in (
        ("all", bets, lines),
        ("all_model_alpha_ex_arbitrage", bets.loc[~arbitrage], lines.loc[~line_arbitrage]),
        (
            "matchup",
            bets[bets["market_group"].eq("matchup")],
            lines[lines["market"].eq("tournament_matchup")],
        ),
        (
            "matchup_model_alpha_ex_arbitrage",
            bets[bets["market_group"].eq("matchup") & ~arbitrage],
            lines[lines["market"].eq("tournament_matchup") & ~line_arbitrage],
        ),
        (
            "arbitrage_contaminated_selected_leg",
            bets.loc[arbitrage],
            lines.loc[line_arbitrage],
        ),
        (
            "outright",
            bets[bets["market_group"].eq("outright")],
            lines[~lines["market"].eq("tournament_matchup")],
        ),
    ):
        if group_lines.empty:
            continue
        clustered = add_complete_zero_bet_event_clusters(group, group_lines)
        perf = performance_summary(
            clustered,
            event_col="event_id",
            event_order_col="event_start",
            starting_bankroll=100.0,
        )
        selected_counts = group.groupby("arm", observed=True).size()
        selected_events = group.groupby("arm", observed=True)["event_id"].nunique()
        perf["bet_count"] = perf["arm"].map(selected_counts).fillna(0).astype(int)
        perf["bet_event_count"] = (
            perf["arm"].map(selected_events).fillna(0).astype(int)
        )
        perf["eligible_event_count"] = int(group_lines["event_id"].nunique())
        perf["market_group"] = group_name
        performance_parts.append(perf)
        boot = event_cluster_bootstrap(
            clustered,
            event_col="event_id",
            event_order_col="event_start",
            draws=bootstrap_draws,
            seed=bootstrap_seed,
        )
        boot["market_group"] = group_name
        bootstrap_parts.append(boot)
    performance = pd.concat(performance_parts, ignore_index=True)
    bootstrap = pd.concat(bootstrap_parts, ignore_index=True)
    for frame in (performance, bootstrap):
        frame["challenger_name"] = finalist_name
        frame["line_policy_name"] = line_policy
        frame["price_point"] = price_point
        frame["sizing_name"] = sizing_name
        frame["settlement_scenario"] = settlement_scenario
    attribution_summary = summarize_attribution(attribution)
    attribution_summary["challenger_name"] = finalist_name
    attribution_summary["line_policy_name"] = line_policy
    attribution_summary["price_point"] = price_point
    attribution_summary["sizing_name"] = sizing_name
    attribution_summary["settlement_scenario"] = settlement_scenario
    return {
        "bets": bets,
        "all_offered_bets": all_offered,
        "attribution": attribution,
        "attribution_summary": attribution_summary,
        "performance": performance,
        "bootstrap": bootstrap,
    }


def estimate_compute_runtime(events: int, arms: int, simulations: int) -> dict[str, float]:
    """Scale the audited v4 per-event timings to this arm/draw count."""

    audit_path = V4 / "event_audit.parquet"
    if not audit_path.exists() or simulations <= 0:
        return {"event_kernel_minutes": 0.0, "median_event_seconds": 0.0}
    audit = pd.read_parquet(audit_path, columns=["runtime_seconds"])
    v4_arms = 3  # baseline plus two shot-mean challengers
    scale = (arms / v4_arms) * (simulations / 50_000.0)
    return {
        "event_kernel_minutes": float(audit["runtime_seconds"].mean() * events * scale / 60.0),
        "median_event_seconds": float(audit["runtime_seconds"].median() * scale),
        "source_v4_events": int(len(audit)),
        "source_v4_total_minutes": float(audit["runtime_seconds"].sum() / 60.0),
    }


def write_preflight_report(
    output_dir: Path,
    config: dict[str, Any],
    coverage: pd.DataFrame,
    field_coverage: pd.DataFrame,
) -> None:
    eligible = coverage[coverage["eligible"]]
    replacement = (
        int(field_coverage["replacement_players"].sum()) if not field_coverage.empty else 0
    )
    roster = int(field_coverage["roster_players"].sum()) if not field_coverage.empty else 0
    market_players = (
        int(field_coverage["market_players"].sum()) if not field_coverage.empty else 0
    )
    market_exact = (
        int(field_coverage["market_exact_bridge_players"].sum())
        if not field_coverage.empty
        else 0
    )
    market_fallback = (
        int(field_coverage["market_strict_pt_control_fallback_players"].sum())
        if not field_coverage.empty
        else 0
    )
    market_neutral = (
        int(field_coverage["market_shared_neutral_prior_players"].sum())
        if not field_coverage.empty
        else 0
    )
    lines = [
        "# Base-Skill Tournament-Market Evaluator Preflight",
        "",
        "Research only. No production consumer was changed and no P&L was scored.",
        "",
        f"- Finalist: `{config['finalist_name']}`",
        f"- Stage: `{config['stage']}` ({config['simulations']:,} simulations per arm/event)",
        f"- Eligible events: {len(eligible)}",
        f"- Roster replacement coverage: {replacement:,}/{roster:,} "
        f"({replacement / roster:.2%})" if roster else "- Roster replacement coverage: unavailable",
        f"- Offered-player paths: exact bridge {market_exact:,}; strict-PT control fallback "
        f"{market_fallback:,}; shared neutral prior {market_neutral:,}; total {market_players:,}",
        f"- Event-kernel estimate: {config['compute_estimate']['event_kernel_minutes']:.1f} minutes",
        f"- 5k ladder estimate: {config['compute_ladder_estimates']['screen-5k']['event_kernel_minutes']:.1f} minutes",
        f"- 50k ladder estimate: {config['compute_ladder_estimates']['final-50k']['event_kernel_minutes']:.1f} minutes",
        "- Markets: 72-hole matchup, win, top 5, top 10, top 20",
        "- Price points: archived open and close, evaluated as separate counterfactuals",
        "- Books: BetOnline, Betcris, Pinnacle; true per-book menus and best-of-three",
        "- Top-finish probability rules: Pinnacle uses native full-tie qualification "
        "(`top_nodh`); BetOnline/Betcris use native expected standard dead-heat paid "
        "share (`top_dh`). Win and matchup semantics are unchanged",
        "- Candidate delta allocation: 25% each to OTT/APP/ARG/PUTT",
        "- Weather: neutral upcoming-round state in both arms",
        "- Market prices: evaluation/settlement only, never model features",
        "- Finish probabilities: current-code/full-sample coefficient and correlation "
        "sensitivity; retrospective and ineligible for finalist selection or promotion",
        "- Outright settlement primary: locked official fin_text plus book rules; all 50 "
        "fractional Pinnacle full-tie overrides are emitted and identity-digest pinned",
        "- Settlement sensitivities: untouched raw archive and universal standard "
        "dead-heat math (37 independently pinned raw discrepancies)",
        "- Unsupported formats: Zurich team play, staggered starts, Stableford, and the "
        "known 2024 Pebble Beach 54-hole finish are excluded fail-closed",
        "",
        "No offered player is dropped. Exact bridge rows use the frozen legacy decomposition; "
        "unbridged rows use finalist minus strict-point-in-time v2 control when available; "
        "otherwise both cores receive the same event/tour prior and the primary delta is zero.",
        "",
        "The identity-preflight arm, when used, is only a plumbing check. It is not a finalist and cannot be simulated or scored.",
        "",
    ]
    (output_dir / "PREFLIGHT.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.bootstrap <= 0:
        raise ValueError("bootstrap must be positive")
    simulations = STAGE_SIMULATIONS[args.stage]
    if args.identity_preflight and args.stage != "preflight":
        raise ValueError("--identity-preflight is legal only with --stage preflight")
    start = pd.Timestamp(args.start).normalize()
    end = pd.Timestamp(args.end).normalize()
    required_paths = [
        args.db,
        args.predictions,
        args.source_crosswalk,
        args.odds_cache / "manifest.json",
        args.bridge,
        args.bridge_manifest,
        args.event_crosswalk,
        args.round_panel,
        args.runtime_lock,
        v4bt.CORR_PATH,
        v4bt.COEFF_PATH,
    ]
    if not args.identity_preflight:
        required_paths.append(args.finalist_manifest)
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    if not args.no_shot_variance and not (args.shot_export / "manifest.json").exists():
        raise FileNotFoundError(args.shot_export / "manifest.json")
    runtime_audit = validate_research_runtime_lock(args.runtime_lock)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.resume:
            raise FileExistsError(
                f"Finish-market output directory is nonempty; choose a fresh path: "
                f"{args.output_dir}"
            )
        if not (args.output_dir / "run_config.json").exists():
            raise FileExistsError(
                "--resume requires an existing immutable run_config.json; refusing "
                f"to mix artifacts in {args.output_dir}"
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    v4_authority_audit = validate_v4_authoritative_inputs(
        v4_dir=V4,
        db=args.db,
        predictions=args.predictions,
        source_crosswalk=args.source_crosswalk,
        correlation=v4bt.CORR_PATH,
        coefficients=v4bt.COEFF_PATH,
        odds_cache=args.odds_cache,
        shot_export=args.shot_export,
    )
    v4_authority_paths = [
        Path(record["path"]).resolve()
        for section in ("manifest", "outputs", "inputs")
        for record in (
            [v4_authority_audit[section]]
            if section == "manifest"
            else v4_authority_audit[section].values()
        )
    ]

    prevalidated_finalist: tuple[dict[str, Any], Path] | None = None
    handoff_paths: list[Path] = []
    if not args.identity_preflight:
        prevalidated_finalist = validate_finalist_manifest(args.finalist_manifest)
        handoff_paths = [
            Path(value).resolve()
            for value in prevalidated_finalist[0]["_verified_handoff_paths"]
        ]
    pre_begin_odds_paths = _odds_cache_provenance_paths(args.odds_cache)
    provenance_inputs = [
        *required_paths,
        *pre_begin_odds_paths,
        *handoff_paths,
        *v4_authority_paths,
    ]
    if not args.no_shot_variance:
        provenance_inputs.append((args.shot_export / "manifest.json").resolve())
        provenance_inputs.extend(
            (args.shot_export / f"pga_{year}_shot_sg.parquet").resolve()
            for year in range(2020, end.year + 1)
        )
    if args.resume:
        for existing in (
            [args.output_dir / "run_config.json"]
            + list((args.output_dir / "event_probabilities").glob("*.parquet"))
            + list((args.output_dir / "event_probabilities").glob("*.meta.json"))
        ):
            if existing.is_file():
                provenance_inputs.append(existing.resolve())
    provenance_state = begin_finish_execution_provenance(
        output_dir=args.output_dir,
        input_paths=provenance_inputs,
        args=args,
        runtime_audit=runtime_audit,
    )
    assert_research_runtime_current(runtime_audit)
    v4_authority_audit = revalidate_v4_authority_after_begin(
        expected_audit=v4_authority_audit,
        provenance_state=provenance_state,
        v4_dir=V4,
        db=args.db,
        predictions=args.predictions,
        source_crosswalk=args.source_crosswalk,
        correlation=v4bt.CORR_PATH,
        coefficients=v4bt.COEFF_PATH,
        odds_cache=args.odds_cache,
        shot_export=args.shot_export,
    )
    postvalidated_finalist, _post_begin_odds_paths = (
        revalidate_finish_dynamic_inputs_after_begin(
            finalist_manifest_path=args.finalist_manifest,
            identity_preflight=args.identity_preflight,
            expected_handoff_paths=handoff_paths,
            odds_cache=args.odds_cache,
            expected_odds_paths=pre_begin_odds_paths,
            provenance_state=provenance_state,
        )
    )
    if not args.identity_preflight:
        if postvalidated_finalist is None:
            raise AssertionError("post-begin frozen finalist validation was skipped")
        prevalidated_finalist = postvalidated_finalist

    completion_authority_audit: dict[str, dict[str, str]] = {}
    if not args.identity_preflight:
        if prevalidated_finalist is None:
            raise AssertionError("frozen finalist was not post-begin validated")
        completion_authority_audit = validate_finish_completion_authorities(
            prevalidated_finalist[0],
            round_panel=args.round_panel,
            bridge=args.bridge,
        )

    bridge, event_crosswalk, bridge_audit = load_bridge(
        args.bridge, args.bridge_manifest, args.event_crosswalk
    )
    if args.identity_preflight:
        finalist_manifest: dict[str, Any] = {
            "contract": "identity_preflight_only",
            "finalist_name": "identity_preflight_only",
            "_resolved_selected_name": "identity_preflight_only",
            "_selection_role": "identity_preflight_only",
            "_promotion_gate_passed": False,
            "prediction_units": "sg_per_round",
            "selection_window_end": "not_applicable",
        }
        finalist = identity_finalist_from_bridge(bridge)
        finalist_hash = bridge_audit["sha256"]
        source_attribution_audit: dict[str, Any] = {
            "status": "not_applicable_identity_preflight"
        }
    else:
        if prevalidated_finalist is None:
            raise AssertionError("frozen finalist was not provenance-prevalidated")
        finalist_manifest, finalist_path = prevalidated_finalist
        verified_bridge_path = Path(
            str(finalist_manifest["_verified_core_bridge_path"])
        ).resolve()
        if args.bridge.resolve() != verified_bridge_path:
            raise ValueError(
                "--bridge differs from the discovery-provenance-bound core_bridge"
            )
        if (
            v4bt.sha256_file(args.bridge.resolve())
            != finalist_manifest["_verified_core_bridge_sha256"]
        ):
            raise ValueError("discovery-provenance-bound core_bridge hash mismatch")
        selected_model_name = str(finalist_manifest["_resolved_selected_name"])
        finalist = normalize_finalist_predictions(
            pd.read_parquet(finalist_path)
            if finalist_path.suffix.casefold() in {".parquet", ".pq"}
            else pd.read_csv(finalist_path),
            selected_model_name,
        )
        finalist, source_attribution_audit = attach_verified_source_attribution(
            finalist, finalist_manifest, selected_model_name
        )
        finalist_hash = v4bt.sha256_file(finalist_path)
    selected_model_name = str(finalist_manifest["_resolved_selected_name"])
    downstream_lambdas = selected_downstream_lambdas(
        finalist_manifest, identity_preflight=args.identity_preflight
    )
    finalist = finalist[finalist["event_start"].between(start, end, inclusive="both")].copy()

    replacements = v4bt.load_name_replacements()
    print("Reading verified historical odds cache ...", flush=True)
    reader = HistoricalOddsCacheReader(args.odds_cache)
    raw_lines = normalize_finish_market_prices(reader.matchups(), reader.outrights())
    rosters = v4bt.normalize_prediction_rosters(reader.predictions(), replacements)
    official_outright_results = load_official_outright_results(
        args.db, event_crosswalk, raw_lines
    )
    raw_lines, outright_settlement_corrections, outright_settlement_audit = (
        reconcile_outright_settlements(
            raw_lines,
            official_outright_results,
            enforce_frozen_authority=True,
        )
    )
    settlement_source_authority_sha256 = {
        "historical_database": v4bt.sha256_file(args.db),
        "event_crosswalk": v4bt.sha256_file(args.event_crosswalk),
        "odds_cache_manifest": v4bt.sha256_file(
            args.odds_cache / "manifest.json"
        ),
        "reconciliation_rule_source": v4bt.sha256_file(Path(__file__)),
    }
    outright_settlement_audit["source_authority_sha256"] = (
        settlement_source_authority_sha256
    )
    outright_settlement_corrections.to_csv(
        args.output_dir / "outright_settlement_reconciliation.csv",
        index=False,
        float_format="%.12f",
        lineterminator="\n",
    )
    v4bt.atomic_json(
        args.output_dir / "outright_settlement_reconciliation_audit.json",
        outright_settlement_audit,
    )
    line_sets = select_finish_line_sets(raw_lines)

    finalist_events = set(finalist["event_key"])
    bridge_events = set(bridge["event_key"])
    fixed_events = {
        str(event)
        for (price_point, policy), menu in line_sets.items()
        if price_point == "close" and policy in SHARP_BOOKS
        for event in menu["event_key"].dropna().astype(str)
    }
    roster_events = set(rosters["event_key"])
    intersection = finalist_events & bridge_events & fixed_events & roster_events
    metadata = read_crosswalked_event_metadata(args.db, event_crosswalk, intersection)
    format_exclusions = {
        event: finish_event_format_exclusion(metadata[event])
        for event in intersection
        if event in metadata and finish_event_format_exclusion(metadata[event])
    }
    candidates = [event for event in intersection if event in metadata and event not in format_exclusions]
    if args.event_key:
        requested = set(args.event_key)
        candidates = [event for event in candidates if event in requested]
    candidates.sort(key=lambda event: (metadata[event].start, event))
    if args.stage == "one-event":
        candidates = candidates[:1]
    selected = set(candidates)
    weather_audit = realized_weather_summaries(
        args.round_panel, event_crosswalk, candidates
    )
    starter_keys = actual_starter_keys(args.round_panel, event_crosswalk, candidates)
    weather_by_event = {
        str(row["event_key"]): {
            key: value for key, value in row.items() if key != "event_key"
        }
        for row in weather_audit.to_dict(orient="records")
    }
    all_events = sorted(finalist_events | bridge_events | fixed_events | roster_events)
    coverage_rows: list[dict[str, Any]] = []
    for event in all_events:
        reasons: list[str] = []
        if event not in finalist_events:
            reasons.append("no_finalist_prediction")
        if event not in bridge_events:
            reasons.append("no_exact_core_bridge")
        if event not in fixed_events:
            reasons.append("no_fixed_book_closing_lines")
        if event not in roster_events:
            reasons.append("no_archived_roster")
        if event in intersection and event not in metadata:
            reasons.append("no_crosswalked_local_metadata")
        if event in format_exclusions:
            reasons.append(format_exclusions[event])
        if args.event_key and event not in set(args.event_key):
            reasons.append("event_filter")
        if args.stage == "one-event" and event not in selected and not reasons:
            reasons.append("one_event_stage_limit")
        if event in selected:
            reasons = []
        coverage_rows.append(
            {
                "event_key": event,
                "start": metadata[event].start if event in metadata else pd.NaT,
                "has_finalist": event in finalist_events,
                "has_bridge": event in bridge_events,
                "has_fixed_lines": event in fixed_events,
                "has_roster": event in roster_events,
                "has_crosswalked_metadata": event in metadata,
                "eligible": event in selected,
                "simulated": False,
                "exclusion_reason": ";".join(reasons),
            }
        )
    coverage = pd.DataFrame(coverage_rows)

    if not candidates:
        raise ValueError("No eligible events after finalist/bridge/odds/roster/metadata intersection")
    union_ids = sorted(
        rosters[rosters["event_key"].isin(candidates)]["dg_id"].dropna().astype(int).unique()
    )
    histories, model_histories = v4bt.load_player_histories(
        args.db, union_ids, max(metadata[event].start for event in candidates) + pd.Timedelta(days=1)
    )
    r1_predictions = v4bt.load_r1_predictions(args.predictions, replacements)
    source_crosswalk = v4bt.load_source_crosswalk(args.source_crosswalk)
    prepared_fields: dict[str, pd.DataFrame] = {}
    field_audits: dict[str, dict[str, Any]] = {}
    bridge_gap_rows: list[dict[str, Any]] = []
    for event_key in candidates:
        base_field = build_baseline_field(
            rosters[rosters["event_key"].eq(event_key)],
            metadata[event_key],
            r1_predictions,
            model_histories,
            source_crosswalk,
        )
        field, audit = attach_finalist(
            base_field,
            bridge[bridge["event_key"].eq(event_key)],
            finalist[finalist["event_key"].eq(event_key)],
            downstream_lambdas,
        )
        event_lines = raw_lines[raw_lines["event_key"].eq(event_key)]
        market_ids = set(event_lines["selection_id"].map(v4bt.canonical_player_id))
        market_ids.update(
            value
            for value in event_lines["opponent_id"].map(v4bt.canonical_player_id)
            if value
        )
        field_ids = set(field["dg_id"].map(v4bt.canonical_player_id))
        missing_market_ids = sorted(market_ids - field_ids)
        if missing_market_ids:
            raise ValueError(
                f"{event_key} has {len(missing_market_ids)} market players absent from roster"
            )
        market_mask = field["dg_id"].map(v4bt.canonical_player_id).isin(market_ids)
        market_paths = field.loc[market_mask, "replacement_path"].value_counts()
        audit.update(
            {
                "market_players": int(market_mask.sum()),
                "market_replacement_players": int(
                    (market_mask & field["core_replacement_available"]).sum()
                ),
                "market_neutral_fallback_players": int(
                    (market_mask & ~field["core_replacement_available"]).sum()
                ),
                "market_replacement_coverage": float(
                    field.loc[market_mask, "core_replacement_available"].mean()
                )
                if market_mask.any()
                else np.nan,
                "market_exact_bridge_players": int(
                    market_paths.get("exact_legacy_bridge", 0)
                ),
                "market_strict_pt_control_fallback_players": int(
                    market_paths.get("strict_pt_v2_control_fallback", 0)
                ),
                "market_shared_neutral_prior_players": int(
                    market_paths.get("shared_neutral_field_tour_prior", 0)
                ),
            }
        )
        offered_markets = (
            event_lines.groupby("selection_id")["market"]
            .agg(lambda values: "|".join(sorted(set(values.astype(str)))))
            .to_dict()
        )
        missing_bridge = field[~field["bridge_available"]]
        for row in missing_bridge.to_dict(orient="records"):
            player_id = v4bt.canonical_player_id(row["dg_id"])
            bridge_gap_rows.append(
                {
                    "event_key": event_key,
                    "event_start": metadata[event_key].start,
                    "dg_id": int(row["dg_id"]),
                    "player_name": row["player_name"],
                    "actual_r1_starter": (event_key, int(row["dg_id"])) in starter_keys,
                    "market_offered": player_id in market_ids,
                    "offered_markets_as_selection": offered_markets.get(player_id, ""),
                    "baseline_skill_source": row["skill_source"],
                    "finalist_available": bool(row["finalist_available"]),
                    "strict_pt_control_available": bool(row["control_available"]),
                    "replacement_path": row["replacement_path"],
                    "cause": (
                        "absent_from_exact_weekly_model_capture;"
                        + (
                            "actual_starter_low_data_or_feature_incomplete"
                            if (event_key, int(row["dg_id"])) in starter_keys
                            else "pre_event_roster_nonstarter_or_no_local_r1"
                        )
                        + f";baseline_{row['skill_source']}"
                    ),
                }
            )
        prepared_fields[event_key] = field
        field_audits[event_key] = {"event_key": event_key, **audit}
    field_coverage = pd.DataFrame(field_audits.values())
    bridge_gap = pd.DataFrame(bridge_gap_rows)
    line_sets = {
        key: annotate_line_replacement_paths(menu, prepared_fields)
        for key, menu in line_sets.items()
    }
    menu_rows: list[pd.DataFrame] = []
    for (price_point, line_policy), menu in line_sets.items():
        use = menu[menu["event_key"].isin(candidates)].copy()
        summary = (
            use.groupby(["market", "replacement_path"], observed=True)
            .agg(
                logical_prices=("selection_id", "size"),
                events=("event_key", "nunique"),
                books=("book", lambda values: "|".join(sorted(set(values.astype(str))))),
            )
            .reset_index()
        )
        summary["line_policy"] = line_policy
        summary["price_point"] = price_point
        menu_rows.append(summary)
    market_menu_preflight = pd.concat(menu_rows, ignore_index=True)
    arbitrage_preflight = pd.concat(
        [
            cross_book_matchup_arbitrage_audit(
                line_sets[(price_point, "best_of_three")]
            )
            for price_point in PRICE_POINTS
        ],
        ignore_index=True,
        sort=False,
    )

    arm_names = [lambda_arm_name(value) for value in downstream_lambdas]
    compute_estimate = estimate_compute_runtime(
        len(candidates), 1 + len(arm_names), simulations
    )
    compute_ladder_estimates = {
        stage: estimate_compute_runtime(
            1 if stage == "one-event" else len(candidates),
            1 + len(arm_names),
            draws,
        )
        for stage, draws in STAGE_SIMULATIONS.items()
    }
    source_files = {
        "frozen_handoff_validator": ARCHIVE / "base_skill_frozen_handoff.py",
        "category_parameter_builder": ARCHIVE / "backtest_distribution_engines_2026.py",
        "course_shape_builder": ARCHIVE / "backtest_shot_dispersion_recent.py",
        "shot_feature_builder": ARCHIVE / "build_bmw_shot_features_shadow.py",
        "sim_inputs": REPO / "sim_inputs.py",
    }
    shot_hashes: dict[str, str] = {}
    if not args.no_shot_variance:
        shot_hashes["shot_export::manifest"] = v4bt.sha256_file(
            args.shot_export / "manifest.json"
        )
        for year in range(2020, end.year + 1):
            shot_path = args.shot_export / f"pga_{year}_shot_sg.parquet"
            if not shot_path.exists():
                raise FileNotFoundError(shot_path)
            shot_hashes[f"shot_export::pga_{year}_shot_sg"] = v4bt.sha256_file(
                shot_path
            )
    runtime_payload = runtime_audit["active_runtime"]
    runtime_versions = dict(runtime_payload["packages"])
    runtime_versions["python"] = runtime_payload["python"]["version"]
    runtime_versions["sims_kernel"] = runtime_versions["sims_kernel_api"]
    config: dict[str, Any] = {
        "contract": CONTRACT,
        "stage": args.stage,
        "start": start.date().isoformat(),
        "end": end.date().isoformat(),
        "simulations": simulations,
        "bootstrap": int(args.bootstrap),
        "seed": int(args.seed),
        "finalist_name": selected_model_name,
        "selection_role": finalist_manifest.get("_selection_role"),
        "promotion_gate_passed": finalist_manifest.get("_promotion_gate_passed"),
        "downstream_lambdas": list(downstream_lambdas),
        "primary_downstream_lambda": 1.0,
        "challenger_arms": arm_names,
        "identity_preflight": bool(args.identity_preflight),
        "event_count": len(candidates),
        "event_keys": candidates,
        "format_exclusions": format_exclusions,
        "v4_authoritative_input_audit": v4_authority_audit,
        "completion_authority_audit": completion_authority_audit,
        "replacement_formula": (
            "baseline_r1_prediction + finalist_core - legacy_core_pred_step2"
        ),
        "category_delta_allocation": {category: 0.25 for category in v4bt.CAT_ORDER},
        "weather": "neutral_pre_event_common_both_arms",
        "realized_weather": "posthoc_oracle_event_audit_only_not_probability_input",
        "market_regression": "disabled_close_is_evaluation_only",
        "shot_variance": (
            "disabled_diagnostic" if args.no_shot_variance else "current_v4_overlay_common"
        ),
        "course_shape": (
            "disabled_diagnostic" if args.no_course_shape else "strict_prior_physical_course"
        ),
        "cut_rules": (
            "event-year-and-field-size-safe local contract; majors explicit, "
            "2024+ signature invitationals top-50, <=80 no-cut, otherwise top-65"
        ),
        "unsupported_formats_fail_closed": True,
        "zurich_team_format_excluded": True,
        "common_random_numbers": "same_stable_event_seed_for_baseline_and_finalist",
        "market_menus": "asserted_identical_across_arms",
        "matchup_books": list(v4bt.MATCHUP_BOOKS),
        "outright_books": list(v4bt.OUTRIGHT_BOOKS),
        "price_points": list(PRICE_POINTS),
        "line_policies": list(LINE_POLICIES),
        "fixed_book_semantics": "one separate menu per named book; no priority fallback",
        "best_of_three_semantics": (
            "best independently executable side price; selected row retains the same-book "
            "opposite quote for no-vig/CLV attribution"
        ),
        "same_book_devig_probability_role": (
            "attribution_only_never_used_for_finish_proper_score; ties-lose values "
            "are conditional on either player winning because no draw quote exists"
        ),
        "same_book_devig_proper_score_eligible": False,
        "book_rule_probability_selection": {
            "top_finish_pinnacle": (
                "native_top_nodh_full_tie_qualification_probability"
            ),
            "top_finish_betonline_betcris": (
                "native_top_dh_standard_dead_heat_expected_paid_share"
            ),
            "win": "split_first_place_dead_heat_expected_share_unchanged",
            "matchup": "win_probability_plus_explicit_push_mass_unchanged",
            "logical_probability_panel_default": (
                "standard_dead_heat_expected_share_with_both_native_variants_persisted"
            ),
            "resolution_time": "after_executable_fixed_or_best_three_book_selection",
            "alternative_settlement_scenarios": (
                "outcome_only_sensitivities_keep_primary_book_rule_pricing_and_selection"
            ),
        },
        "primary_pricing_settlement_book_rule_coherent": True,
        "outright_settlement_reconciliation": outright_settlement_audit,
        "outright_settlement_result_role": (
            "book-rule reconstructed-settlement retrospective sensitivity; 50 "
            "Pinnacle full-tie overrides and 37 universal-standard discrepancies "
            "are separately provenance-pinned; raw and universal-standard scenarios "
            "are scored alongside the primary"
        ),
        "sizing": ["flat_1u", "production_style"],
        "matchup_ev_gate": v4bt.MATCHUP_GATE,
        "outright_ev_gate": v4bt.OUTRIGHT_GATE,
        "bootstrap_event_universe": "all_eligible_events_including_zero_bet_events",
        "one_direction_per_canonical_matchup": True,
        "cross_book_arbitrage_reporting": (
            "pair flagged only when both selected reverse legs push on tie and raw "
            "implied sum is below one; ties-lose three-way pairs are not guaranteed "
            "arbitrage; flagged single legs are excluded from model-alpha"
        ),
        "finish_probability_claim": (
            "retrospective_current_code_sensitivity_selection_ineligible"
        ),
        "finish_kernel_coefficients_point_in_time": False,
        "finish_correlation_point_in_time": False,
        "finish_results_may_select_or_promote_finalist": False,
        "full_sample_coefficients_and_correlation_role": (
            "common-current-code-retrospective-sensitivity-only"
        ),
        "bridge_audit": bridge_audit,
        "source_attribution_audit": source_attribution_audit,
        "full_finalist_provenance_chain": finalist_manifest.get(
            "_provenance_chain_audit", {}
        ),
        "compute_estimate": compute_estimate,
        "compute_ladder_estimates": compute_ladder_estimates,
        "runtime_versions": runtime_versions,
        "runtime_lock_audit": runtime_audit,
        "input_hashes": {
            "bridge": bridge_audit["sha256"],
            "bridge_manifest": v4bt.sha256_file(args.bridge_manifest),
            "event_crosswalk": settlement_source_authority_sha256[
                "event_crosswalk"
            ],
            "round_panel_for_posthoc_weather_only": v4bt.sha256_file(args.round_panel),
            "historical_database": settlement_source_authority_sha256[
                "historical_database"
            ],
            "finalist_predictions": finalist_hash,
            "finalist_manifest": v4bt.sha256_file(
                args.bridge_manifest if args.identity_preflight else args.finalist_manifest
            ),
            "odds_cache_manifest": settlement_source_authority_sha256[
                "odds_cache_manifest"
            ],
            "predictions": v4bt.sha256_file(args.predictions),
            "source_crosswalk": v4bt.sha256_file(args.source_crosswalk),
            "correlation": v4bt.sha256_file(v4bt.CORR_PATH),
            "coefficients": v4bt.sha256_file(v4bt.COEFF_PATH),
            "source::evaluator": v4bt.sha256_file(Path(__file__)),
            "source::v4_backtest": v4bt.sha256_file(Path(v4bt.__file__)),
            "source::market_scoring": v4bt.sha256_file(ARCHIVE / "market_backtest_scoring.py"),
            "research_runtime_lock::finish_market_runtime_v1": runtime_audit["lock"][
                "sha256"
            ],
            "python_package_init::sims_kernel": runtime_payload["sims_kernel"][
                "package_init"
            ]["sha256"],
            SIMS_KERNEL_NATIVE_SOURCE_LABEL: runtime_payload["sims_kernel"][
                "native_extension"
            ]["sha256"],
            **{
                f"source::{name}": v4bt.sha256_file(path)
                for name, path in source_files.items()
            },
            **shot_hashes,
        },
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, default=v4bt._json_default).encode("utf-8")
    ).hexdigest()
    config["config_hash"] = config_hash
    config_path = args.output_dir / "run_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("config_hash") != config_hash:
            raise ValueError(
                f"Output directory contains a different immutable run_config: {config_path}. "
                "Choose a new output directory; existing research artifacts were preserved."
            )
    else:
        v4bt.atomic_json(config_path, config)
    coverage.to_csv(args.output_dir / "event_coverage_preflight.csv", index=False)
    field_coverage.to_csv(args.output_dir / "field_coverage_preflight.csv", index=False)
    weather_audit.to_csv(args.output_dir / "realized_weather_oracle_audit.csv", index=False)
    bridge_gap.to_csv(args.output_dir / "bridge_gap_player_audit.csv", index=False)
    market_menu_preflight.to_csv(
        args.output_dir / "market_menu_preflight.csv", index=False
    )
    arbitrage_preflight.to_csv(
        args.output_dir / "cross_book_matchup_arbitrage_preflight.csv", index=False
    )
    write_preflight_report(args.output_dir, config, coverage, field_coverage)
    print(
        f"Eligible events: {len(candidates)}; replacement coverage "
        f"{field_coverage['replacement_players'].sum():,}/"
        f"{field_coverage['roster_players'].sum():,}; estimated event-kernel time "
        f"{compute_estimate['event_kernel_minutes']:.1f} minutes.",
        flush=True,
    )
    if args.stage == "preflight":
        assert_research_runtime_current(runtime_audit)
        execution_audit = finalize_finish_execution_provenance(provenance_state)
        preflight_manifest = {
            **config,
            "completed": True,
            "simulation_started": False,
            "pnl_scored": False,
            "execution_provenance": execution_audit,
            "outputs": {
                path.relative_to(args.output_dir).as_posix(): {
                    "sha256": v4bt.sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                for path in sorted(args.output_dir.rglob("*"))
                if path.is_file() and path.name != "preflight_manifest.json"
            },
        }
        write_finish_complete_manifest(
            args.output_dir / "preflight_manifest.json",
            preflight_manifest,
            provenance_state,
        )
        print(f"Preflight complete; no simulations or P&L: {args.output_dir}", flush=True)
        return 0

    import sims_kernel

    week_latent_sd = float(v4bt.load_sim_input_literal("WEEK_LATENT_SD"))
    corr, chol = load_correlation(v4bt.CORR_PATH)
    coefficients = v4bt.load_coefficients()
    union_source_ids = sorted(
        {
            source_id
            for field in prepared_fields.values()
            for source_id in field.loc[field["source_player_id"].ne(""), "source_player_id"]
        }
    )
    if args.no_shot_variance:
        shot_stats = pd.DataFrame()
        valid_shots = 0
    else:
        event_dates = load_event_dates(args.db)
        shot_stats, valid_shots = load_shot_round_stats(
            args.shot_export,
            union_source_ids,
            event_dates,
            max(metadata[event].start for event in candidates) + pd.Timedelta(days=1),
        )
    checkpoint_dir = args.output_dir / "event_probabilities"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    probability_pieces: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for index, event_key in enumerate(candidates, start=1):
        checkpoint = checkpoint_dir / f"{event_key.replace(':', '_')}.parquet"
        checkpoint_meta = checkpoint.with_suffix(".meta.json")
        if checkpoint.exists() or checkpoint_meta.exists():
            if not checkpoint.exists() or not checkpoint_meta.exists():
                raise ValueError(
                    f"Incomplete resume checkpoint pair for {event_key}: "
                    f"{checkpoint}, {checkpoint_meta}"
                )
            if not args.resume:
                raise ValueError(
                    f"Unexpected checkpoint appeared during a fresh run: {checkpoint}"
                )
            require_tracked_resume_checkpoint(
                provenance_state,
                run_config_path=args.output_dir / "run_config.json",
                checkpoint_path=checkpoint,
                checkpoint_meta_path=checkpoint_meta,
            )
            saved = json.loads(checkpoint_meta.read_text(encoding="utf-8"))
            if saved.get("config_hash") == config_hash and saved.get("complete") is True:
                if v4bt.sha256_file(checkpoint) != saved.get("checkpoint_sha256"):
                    raise ValueError(f"Checkpoint hash mismatch: {checkpoint}")
                probability_pieces.append(pd.read_parquet(checkpoint))
                audit_rows.append(saved["event_audit"])
                print(f"[{index}/{len(candidates)}] resume {event_key}", flush=True)
                continue
        started = time.perf_counter()
        meta = metadata[event_key]
        field = prepared_fields[event_key]
        if args.no_shot_variance:
            features = pd.DataFrame({"source_player_id": field["source_player_id"].unique()})
            feature_prior: dict[str, float] = {}
        else:
            shot_input = shot_stats[shot_stats["event_date"].lt(meta.start)]
            feature_ids = field.loc[
                field["source_player_id"].ne(""), ["source_player_id"]
            ].drop_duplicates("source_player_id")
            features, feature_prior = build_dispersion_features(shot_input, feature_ids)
        mults, skews, course_audit = v4bt.course_shape(args.db, meta, args.no_course_shape)
        params, distribution_audit = build_category_params(
            histories, field, meta.start, features, mults, skews
        )
        cut_line, cut_rule = cut_line_for_event(meta, len(field))
        event_seed = v4bt.stable_seed(event_key, args.seed)
        event_lines = raw_lines[raw_lines["event_key"].eq(event_key)]
        arm_probabilities: list[pd.DataFrame] = []
        probability_audit: dict[str, int] = {}
        simulation_arms = [("baseline", np.zeros(len(field), dtype=float))] + [
            (arm, field[f"{arm}_delta"].to_numpy(float)) for arm in arm_names
        ]
        for arm, delta in simulation_arms:
            scores = simulate_scores(
                params,
                field,
                chol,
                coefficients,
                delta,
                simulations,
                event_seed,
                cut_line,
                week_latent_sd,
            )
            probabilities, probability_audit = probabilities_for_finish_lines(
                scores, field, event_lines, event_key, meta.start
            )
            probabilities["arm"] = arm
            arm_probabilities.append(probabilities)
            del scores
            gc.collect()
        event_probabilities = pd.concat(arm_probabilities, ignore_index=True, sort=False)
        assert_common_market_menus(event_probabilities)
        assert_matchup_probability_complements(event_probabilities)
        v4bt.atomic_parquet(event_probabilities, checkpoint)
        event_audit = {
            "event_key": event_key,
            "year": meta.year,
            "local_event_id": meta.event_id,
            "event_name": meta.event_name,
            "start": meta.start,
            "seed": event_seed,
            "simulations": simulations,
            "cut_line": cut_line,
            "cut_rule": cut_rule,
            "field": field_audits[event_key],
            "realized_weather_oracle": weather_by_event.get(event_key, {}),
            "course_shape": course_audit,
            "distribution": distribution_audit,
            "shot_variance_category_priors": feature_prior,
            **probability_audit,
            "runtime_seconds": time.perf_counter() - started,
        }
        v4bt.atomic_json(
            checkpoint_meta,
            {
                "complete": True,
                "config_hash": config_hash,
                "checkpoint_sha256": v4bt.sha256_file(checkpoint),
                "event_audit": event_audit,
            },
        )
        probability_pieces.append(event_probabilities)
        audit_rows.append(event_audit)
        print(f"[{index}/{len(candidates)}] completed {event_key}", flush=True)

    probabilities = pd.concat(probability_pieces, ignore_index=True, sort=False)
    assert_common_market_menus(probabilities)
    assert_matchup_probability_complements(probabilities)
    event_audit = v4bt.normalize_event_audit_rows(audit_rows)
    simulated_events = set(probabilities["event_key"])
    coverage["simulated"] = coverage["event_key"].isin(simulated_events)
    coverage.to_csv(args.output_dir / "event_coverage.csv", index=False)
    v4bt.atomic_parquet(probabilities, args.output_dir / "probabilities.parquet")
    v4bt.atomic_parquet(event_audit, args.output_dir / "event_audit.parquet")

    scored_outputs: dict[str, list[pd.DataFrame]] = {
        "bets": [],
        "all_offered_bets": [],
        "attribution": [],
        "attribution_summary": [],
        "performance": [],
        "bootstrap": [],
    }
    base_probs = probabilities[probabilities["arm"].eq("baseline")]
    for arm_index, arm_name in enumerate(arm_names):
        finalist_probs = probabilities[probabilities["arm"].eq(arm_name)]
        challenger_label = (
            f"{selected_model_name}|downstream_lambda="
            f"{downstream_lambdas[arm_index]:g}"
        )
        for (price_point, line_policy), chosen_lines in line_sets.items():
            chosen_lines = chosen_lines[chosen_lines["event_key"].isin(simulated_events)].copy()
            if chosen_lines.empty:
                continue
            chosen_lines = chosen_lines.merge(
                coverage[["event_key", "start"]].rename(columns={"start": "event_start"}),
                on="event_key",
                how="left",
                validate="many_to_one",
            )
            for settlement_scenario, outcome_column in (
                SETTLEMENT_SCENARIO_COLUMNS.items()
            ):
                scenario_lines = chosen_lines.copy()
                scenario_lines["bet_outcome_numeric"] = pd.to_numeric(
                    scenario_lines[outcome_column], errors="raise"
                )
                scenario_lines["settlement_scenario"] = settlement_scenario
                scenario_lines["settlement_scenario_outcome_column"] = outcome_column
                scenario_challenger_label = (
                    f"{challenger_label}|settlement={settlement_scenario}"
                )
                for sizing_name in ("flat_1u", "production_style"):
                    scored = score_comparison(
                        scenario_lines,
                        base_probs,
                        finalist_probs,
                        scenario_challenger_label,
                        line_policy,
                        price_point,
                        sizing_name,
                        settlement_scenario,
                        args.bootstrap,
                        args.seed + 10_000 * arm_index,
                    )
                    for key, frame in scored.items():
                        scored_outputs[key].append(frame)
    combined = {
        key: pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
        for key, frames in scored_outputs.items()
    }
    weather_detail = weather_audit.rename(columns={"event_key": "event_id"})
    if not weather_detail.empty:
        for key in ("bets", "all_offered_bets"):
            combined[key] = combined[key].merge(
                weather_detail,
                on="event_id",
                how="left",
                validate="many_to_one",
            )
    probability_rule_audit = book_rule_probability_selection_audit(
        combined["all_offered_bets"]
    )
    v4bt.atomic_json(
        args.output_dir / "book_rule_probability_selection_audit.json",
        probability_rule_audit,
    )
    assert_one_matchup_direction_per_pair(combined["bets"])
    assert_one_matchup_direction_per_pair(combined["all_offered_bets"])
    v4bt.atomic_parquet(combined["bets"], args.output_dir / "bets.parquet")
    v4bt.atomic_parquet(
        combined["all_offered_bets"], args.output_dir / "all_offered_bets.parquet"
    )
    primary_selected = combined["bets"].loc[
        combined["bets"]["settlement_scenario"].eq("book_rule_primary")
    ]
    settlement_completeness = {
        "all_scenario_selected_rows": int(len(combined["bets"])),
        "all_scenario_selected_ungraded_rows": int(
            (~combined["bets"]["is_graded"].fillna(False).astype(bool)).sum()
        ),
        "primary_selected_rows": int(len(primary_selected)),
        "primary_ungraded_rows": int(
            (~primary_selected["is_graded"].fillna(False).astype(bool)).sum()
        ),
        "all_offered_rows": int(len(combined["all_offered_bets"])),
        "all_offered_ungraded_rows": int(
            (
                ~combined["all_offered_bets"]["is_graded"]
                .fillna(False)
                .astype(bool)
            ).sum()
        ),
        "primary_settlement_status": primary_selected["settlement_status"]
        .fillna("missing")
        .value_counts()
        .sort_index()
        .to_dict(),
        "selected_rows_by_settlement_scenario": combined["bets"]
        .groupby("settlement_scenario", observed=True)
        .size()
        .sort_index()
        .to_dict(),
        "all_offered_settlement_status": combined["all_offered_bets"][
            "settlement_status"
        ]
        .fillna("missing")
        .value_counts()
        .sort_index()
        .to_dict(),
    }
    v4bt.atomic_json(
        args.output_dir / "settlement_completeness_audit.json",
        settlement_completeness,
    )
    if (
        settlement_completeness["all_scenario_selected_ungraded_rows"]
        or settlement_completeness["all_offered_ungraded_rows"]
    ):
        raise ValueError(
            "Finish-market opportunities lack settlement; refusing biased P&L: "
            + json.dumps(settlement_completeness, sort_keys=True)
        )
    v4bt.atomic_parquet(combined["attribution"], args.output_dir / "attribution.parquet")
    combined["attribution_summary"].to_csv(
        args.output_dir / "attribution_summary.csv", index=False
    )
    combined["performance"].to_csv(args.output_dir / "performance.csv", index=False)
    combined["bootstrap"].to_csv(args.output_dir / "bootstrap.csv", index=False)
    replacement_path_performance(combined["bets"]).to_csv(
        args.output_dir / "replacement_path_performance.csv", index=False
    )
    selected_buckets = finish_bet_bucket_performance(combined["bets"])
    selected_buckets.to_csv(
        args.output_dir / "bet_bucket_performance.csv", index=False
    )
    all_offered_buckets = finish_bet_bucket_performance(
        combined["all_offered_bets"]
    )
    all_offered_buckets.to_csv(
        args.output_dir / "all_offered_bucket_performance.csv", index=False
    )
    all_offered_buckets.loc[
        all_offered_buckets.get(
            "bucket_dimension", pd.Series(dtype=str)
        ).eq("edge")
    ].to_csv(
        args.output_dir / "all_offered_edge_bucket_performance.csv", index=False
    )
    year_book_performance(combined["bets"]).to_csv(
        args.output_dir / "year_book_performance.csv", index=False
    )
    v4bt.build_year_market_summary(combined["bets"]).to_csv(
        args.output_dir / "year_market_summary.csv", index=False
    )
    assert_research_runtime_current(runtime_audit)
    execution_audit = finalize_finish_execution_provenance(provenance_state)
    manifest = {
        **config,
        "completed": True,
        "execution_provenance": execution_audit,
        "simulated_events": len(simulated_events),
        "valid_shots_loaded": int(valid_shots),
        "runtime_versions": runtime_versions,
        "outputs": {
            path.relative_to(args.output_dir).as_posix(): {
                "sha256": v4bt.sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(args.output_dir.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        },
    }
    write_finish_complete_manifest(
        args.output_dir / "manifest.json", manifest, provenance_state
    )
    print(f"Evaluation complete: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
