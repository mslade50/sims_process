"""Discovery-only market selection for externally built base-skill predictions.

This is the bounded second half of ``backtest_base_skill_market_models.py``.
It consumes a point-in-time prediction panel that was built elsewhere, together
with the audited matchup quotes/snapshots from a completed plumbing run.  It
does not fit golfer-skill models.  A source panel may contain sealed 2023+ rows,
but they are physically filtered out before calibration, scoring, or selection.

Probability calibration is itself out of sample for selection:

* 2021 matchup probabilities use a mapper fitted on 2020 only;
* 2022 matchup probabilities use a mapper fitted on 2021 only; and
* the mapper intended for the later sealed evaluation is fitted on all of
  2021-22 only after selection scores are complete.

All broad candidates receive canonical-pair proper scores.  Betting rows are
written only for the mechanically selected finalist (if one passes every
gate) and declared controls.  Closing prices are never prediction features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from archive import backtest_base_skill_market_models as market_eval  # noqa: E402
from archive import research_execution_provenance as execution_provenance  # noqa: E402
from archive.market_backtest_scoring import (  # noqa: E402
    american_implied_probability,
    valid_american_odds_mask,
)


DEFAULT_ROOT = (
    REPO / "permanent_data" / "research" / "base_skill_rebuild_20200101_20260822"
)
DEFAULT_CACHE = DEFAULT_ROOT / "model_eval_clean_dg_plumbing_v4"
DEFAULT_PREDICTIONS = (
    DEFAULT_ROOT
    / "model_completion_v3_provenance"
    / "player_event_predictions.parquet"
)
DEFAULT_CATALOG = (
    DEFAULT_ROOT / "model_completion_v3_provenance" / "candidate_catalog.csv"
)
DEFAULT_SOURCE_MANIFEST = (
    DEFAULT_ROOT / "model_completion_v3_provenance" / "manifest.json"
)
DEFAULT_OUTPUT = DEFAULT_ROOT / "model_eval_complete_discovery_v3_provenance"
DEFAULT_CORE_BRIDGE = DEFAULT_ROOT / "core_bridge" / "legacy_core_bridge.parquet"
EXPECTED_COMPLETED_STUDY_MANIFEST_SHA256 = (
    "53a6446d683d6a4709acd24cdf7843bf2db7393c38e7f09eb3cc803fa416e8e0"
)
EXPECTED_COMPLETED_STUDY_MANIFEST_BYTES = 177_511
EXPECTED_COMPLETED_STUDY_OUTPUTS = {
    "player_event_predictions.parquet": {
        "sha256": "c5ad98f85ce19a750922735c067a14cde0bd1b2d002e35df1103a8392c5aaae2",
        "bytes": 149_135_818,
    },
    "candidate_catalog.csv": {
        "sha256": "1ab117a5f4aec78c2885f0db0d7ce6e60541177f7b288c61b10227e96c0feaae",
        "bytes": 537_929,
    },
    "coverage_by_candidate_year.csv": {
        "sha256": "960e15978c18cf59f11630c87aef2c8a06dee2f700e274805aa6c4c3d806ace4",
        "bytes": 62_355,
    },
    "candidate_catalog_frozen.json": {
        "sha256": "d355481e8aaa7a23c7e43470ab6d683f2f5017d9ac89c611e04adae8a6597d23",
        "bytes": 116_032,
    },
    "fit_log.csv": {
        "sha256": "e19c542752ca6ffaee3d558248fe9dcf31bf868b9ef5d95df5512bcf3309db0d",
        "bytes": 1_559_479,
    },
    "regression_envelope_admission_by_candidate.csv": {
        "sha256": "e0d470e190c4754820f6d0ade286e5353ce7e31bec46ef2359a4b77d90b6e86d",
        "bytes": 36_475,
    },
    "regression_envelope_admission_by_fold.csv": {
        "sha256": "0c248aed7c373d949cd38adff27371ae082760178e6d76592298acccf146a86a",
        "bytes": 526_400,
    },
}
COMPLETED_STUDY_AUTHORITY_KEYS = (
    "completion_manifest",
    "predictions",
    "candidate_catalog",
    "coverage_by_candidate_year",
    "frozen_candidate_catalog",
    "fit_log",
    "envelope_admission_by_candidate",
    "envelope_admission_by_fold",
)

# Independently audited authority for the only plumbing cache discovery may
# consume.  A self-consistent replacement cache is not selection-equivalent:
# it could change quotes, point-in-time feature membership, or either outcome
# authority while retaining the same filenames and ``complete=true`` label.
EXPECTED_EVALUATION_CACHE_MANIFEST_SHA256 = (
    "b71541f3eac354b75b28875893db9e053fa6b11b113629082e0e25f183ef0a68"
)
EXPECTED_EVALUATION_CACHE_MANIFEST_BYTES = 5_522
EXPECTED_EVALUATION_CACHE_CONTRACT = (
    REPO / "BASE_SKILL_MARKET_REBUILD_CONTRACT_2026_08.md"
)
EXPECTED_EVALUATION_CACHE_RUN_TYPE = "shortlist_money_screen"
EXPECTED_EVALUATION_CACHE_ARTIFACTS = {
    "normalized_matchup_quotes.parquet": {
        "sha256": "a1ea5b39786148f471cf0786f2e17a8681fe0fd38374b583f8e3f8cab1aba894",
        "bytes": 7_481_034,
    },
    "feature_snapshots.parquet": {
        "sha256": "dd27daf7f8be5449399d142343f259ac8bbf4c3c0b87a13001da607651af2bdd",
        "bytes": 159_146_373,
    },
}
EXPECTED_EVALUATION_CACHE_INPUTS = {
    "event_panel": {
        "path": DEFAULT_ROOT / "player_event_outcomes.parquet",
        "sha256": "5840e76073a5bd7e63798f53496d89e17caa3d80c3576a90a50ea81902c6668e",
        "bytes": 6_939_233,
    },
    "legacy_core_bridge": {
        "path": DEFAULT_CORE_BRIDGE,
        "sha256": "1820586323c8fcbb66e696e8400c2adf628be6a2e758b81987eb524247dee121",
        "bytes": 6_406_412,
    },
    "odds_cache_manifest": {
        "path": (
            REPO
            / "permanent_data"
            / "research"
            / "dg_matchup_cache_20200101_20260822"
            / "manifest.json"
        ),
        "sha256": "5661a5b08dd5fa748b067c0eaf15bdee4a12e29a69623872be726fb1c1836335",
        "bytes": 6_922,
    },
    "odds_event_crosswalk": {
        "path": DEFAULT_ROOT / "odds_event_crosswalk.parquet",
        "sha256": "60959cfb194f27769c914d367a8bf3697a74966ef5ffa48a839722109d24267d",
        "bytes": 39_196,
    },
    "round_panel": {
        "path": DEFAULT_ROOT / "player_round_panel.parquet",
        "sha256": "37df323c7e43681e02aa90bb12d84406bade24586315082e2e553ff323485f2a",
        "bytes": 44_577_853,
    },
}
EVALUATION_CACHE_AUTHORITY_KEYS = (
    "evaluation_cache_manifest",
    "normalized_quotes",
    "feature_snapshots",
    "cache_input_event_panel",
    "cache_input_legacy_core_bridge",
    "cache_input_odds_cache_manifest",
    "cache_input_odds_event_crosswalk",
    "cache_input_round_panel",
)

DISCOVERY_START = pd.Timestamp("2021-01-01")
DISCOVERY_END = pd.Timestamp("2022-12-31")
WARMUP_START = pd.Timestamp("2020-01-01")
EVALUATION_YEARS = (2021, 2022)
DEFAULT_BASELINE = "current_architecture_pt_v2_approx_control"
EXPECTED_FROZEN_CATALOG_SHA256 = (
    "d355481e8aaa7a23c7e43470ab6d683f2f5017d9ac89c611e04adae8a6597d23"
)
EXPECTED_BUILDER_SOURCE_SHA256 = (
    "69d442b7c91a6191861842290fa69ecee437d35830510bcf0fd3edd9d28cc019"
)
EXPECTED_CLEAN_CANDIDATE_COUNT = 92
FROZEN_MIN_CALIBRATION_PAIRS = 100
FROZEN_FINALIST_COUNT = 1
FROZEN_GATE_TOLERANCES = {
    "coverage": 0.0,
    "logloss": 0.0,
    "calibration": 0.0,
    "money": 0.0,
}
FROZEN_PAIRWISE_REGULARIZATION_C = 0.1
FROZEN_DOWNSTREAM_LAMBDAS = (0.0, 0.5, 1.0)
FROZEN_OPENER_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
FROZEN_ENVELOPE_OVERALL_CLIP_RATE_MAX = 0.01
FROZEN_ENVELOPE_FOLD_CLIP_RATE_MAX = 0.05
EXPECTED_REGRESSION_CANDIDATE_COUNT = 88
EXPECTED_DISCOVERY_GUARD_DEPENDENT_COUNT = 34
# Canonical transitive research-source closure for discovery and every
# downstream consumer that calls its frozen-handoff validators.  Keep this
# exact: additions/removals are a provenance-contract change, not a best-effort
# list.  ``backtest_base_skill_market_models`` imports historical_odds_cache,
# so that module must be preserved even though this file does not name its
# reader directly.
DISCOVERY_PROVENANCE_SOURCE_CLOSURE = (
    REPO / "archive" / "evaluate_complete_base_skill_market.py",
    REPO / "archive" / "backtest_base_skill_market_models.py",
    REPO / "archive" / "historical_odds_cache.py",
    REPO / "archive" / "market_backtest_scoring.py",
    Path(execution_provenance.__file__),
)
PROVENANCE_CODE_PATHS = DISCOVERY_PROVENANCE_SOURCE_CLOSURE
DISCOVERY_DIRECT_INPUT_KEYS = (
    "predictions",
    "candidate_catalog",
    "frozen_candidate_catalog",
    "candidate_manifest",
    "normalized_quotes",
    "feature_snapshots",
    "evaluation_cache_manifest",
    "coverage_by_candidate_year",
    "core_bridge",
)
KNOWN_CONTROL_UNITS = {
    "current_architecture_pt_v2_approx_control": "sg_per_round_control",
    "legacy_skill_est_pt_current_coefficients_control": (
        "sg_per_round_retrospective_control"
    ),
    "legacy_pred_step2_exact_retrospective_control": (
        "sg_per_round_retrospective_control"
    ),
}
KNOWN_CONTROL_METADATA = {
    "current_architecture_pt_v2_approx_control": {
        "family": "control",
        "source": "current_skill_v2_architecture_proxy",
        "weather_arm": "none",
        "target_arm": "control",
        "refit_schedule": "event_batch",
    },
    "legacy_skill_est_pt_current_coefficients_control": {
        "family": "control",
        "source": "legacy_skill_est",
        "weather_arm": "none",
        "target_arm": "control",
        "refit_schedule": "event_batch",
    },
    "legacy_pred_step2_exact_retrospective_control": {
        "family": "control",
        "source": "legacy_pred_step2_bridge",
        "weather_arm": "none",
        "target_arm": "control",
        "refit_schedule": "event_batch",
    },
}
DEFAULT_PRIORS = {
    "72h": (12.0, 0.70),
    "r1": (6.0, 0.80),
}
EDGE_BINS = [-np.inf, 0.0, 0.03, 0.05, 0.10, np.inf]
EDGE_LABELS = ["<0%", "0-3%", "3-5%", "5-10%", ">10%"]
POSITIVE_EDGE_LABELS = EDGE_LABELS[1:]

REQUIRED_PREDICTION_COLUMNS = {
    "candidate",
    "event_key",
    "dg_id_key",
    "event_start_date",
    "prediction",
    "prediction_units",
    "chronology_ok",
    "prediction_asof",
    "training_max_event_end",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_discovery_complete_manifest(
    path: Path,
    payload: Mapping[str, Any],
    provenance_state: execution_provenance.ExecutionProvenance,
) -> None:
    """Write the complete manifest only while the execution snapshot is current."""

    execution_provenance.assert_current(provenance_state)
    _atomic_json(path, payload)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".parquet", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


FROZEN_SPEC_FIELDS = (
    "name",
    "family",
    "source",
    "weather_arm",
    "target_arm",
    "refit_schedule",
    "alpha",
    "l1_ratio",
    "huber_epsilon",
    "dynamic_half_life",
    "prior_rounds",
    "process_variance_per_day",
    "short_game_penalty",
    "selection_eligible",
    "sensitivity_only",
    "control",
    "prediction_envelope_applied",
    "prediction_envelope_method",
    "prediction_envelope_scale_factor",
    "prediction_envelope_multiplier",
    "raw_regression_component_selection_eligible",
    "selection_ineligibility_reason",
    "description",
)
FROZEN_BOOL_FIELDS = {
    "selection_eligible",
    "sensitivity_only",
    "control",
    "prediction_envelope_applied",
    "raw_regression_component_selection_eligible",
}
FROZEN_FLOAT_FIELDS = {
    "alpha",
    "l1_ratio",
    "huber_epsilon",
    "dynamic_half_life",
    "prior_rounds",
    "process_variance_per_day",
    "short_game_penalty",
    "prediction_envelope_scale_factor",
    "prediction_envelope_multiplier",
}


def _required_manifest_output(
    payload: Mapping[str, Any],
    manifest_path: Path,
    name: str,
    *,
    expected_path: Path | None = None,
) -> tuple[Path, str]:
    """Resolve and verify one mandatory builder-manifest output."""

    outputs = payload.get("outputs")
    entry = outputs.get(name) if isinstance(outputs, Mapping) else None
    if not isinstance(entry, Mapping) or not entry.get("path") or not entry.get(
        "sha256"
    ):
        raise ValueError(f"source manifest lacks required hashed output {name!r}")
    path = Path(str(entry["path"]))
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    if expected_path is not None and path != expected_path.resolve():
        raise ValueError(f"source manifest output path disagrees for {name!r}")
    digest = str(entry["sha256"]).lower()
    if len(digest) != 64 or not path.is_file() or _sha256(path) != digest:
        raise ValueError(f"source manifest output hash mismatch for {name!r}")
    return path, digest


def validate_completion_manifest_provenance(
    source_manifest: Mapping[str, Any],
    source_manifest_path: Path,
) -> dict[str, Any]:
    """Bind market scoring to the byte-exact, unchanged completion builder."""

    if source_manifest.get("research_only") is not True:
        raise ValueError("completion manifest is not explicitly research-only")
    if source_manifest.get("production_files_changed") is not False:
        raise ValueError("completion manifest does not prove production stayed unchanged")
    contract = source_manifest.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("completion manifest lacks its modeling contract")
    for key, expected in (
        ("challengers_use_legacy_skill", False),
        ("closing_prices_used_as_features", False),
        ("selection_uses_sealed", False),
    ):
        if contract.get(key) is not expected:
            raise ValueError(f"completion contract requires {key}={expected}")

    execution = source_manifest.get("execution_provenance")
    if not isinstance(execution, Mapping):
        raise ValueError("completion manifest lacks execution provenance")
    for key in (
        "all_unchanged",
        "source_unchanged",
        "source_copy_matches",
        "source_copy_unchanged",
    ):
        if execution.get(key) is not True:
            raise ValueError(f"completion execution provenance requires {key}=true")
    input_unchanged = execution.get("input_unchanged")
    if not isinstance(input_unchanged, Mapping) or not input_unchanged:
        raise ValueError("completion execution provenance lacks input-unchanged checks")
    bad_inputs = sorted(
        str(name) for name, unchanged in input_unchanged.items() if unchanged is not True
    )
    if bad_inputs:
        raise ValueError(f"completion inputs changed during execution: {bad_inputs}")

    source_records = (
        "source_before",
        "source_after",
        "executed_source_copy_before",
        "executed_source_copy_after",
    )
    observed_hashes: dict[str, str] = {}
    for key in source_records:
        record = execution.get(key)
        if not isinstance(record, Mapping):
            raise ValueError(f"completion execution provenance lacks {key}")
        digest = str(record.get("sha256", "")).lower()
        if digest != EXPECTED_BUILDER_SOURCE_SHA256:
            raise ValueError(
                f"completion {key} is not the frozen builder source: {digest}"
            )
        observed_hashes[key] = digest

    executed_copy = execution["executed_source_copy_after"]
    copy_path = Path(str(executed_copy.get("path", "")))
    if not copy_path.is_absolute():
        copy_path = source_manifest_path.parent / copy_path
    copy_path = copy_path.resolve()
    if not copy_path.is_file():
        raise FileNotFoundError(copy_path)
    if _sha256(copy_path) != EXPECTED_BUILDER_SOURCE_SHA256:
        raise ValueError("preserved executed builder source bytes changed after completion")
    if executed_copy.get("size_bytes") is not None and copy_path.stat().st_size != int(
        executed_copy["size_bytes"]
    ):
        raise ValueError("preserved executed builder source size changed")
    return {
        "expected_builder_source_sha256": EXPECTED_BUILDER_SOURCE_SHA256,
        "source_hashes": observed_hashes,
        "executed_source_copy": str(copy_path),
        "executed_source_copy_sha256": _sha256(copy_path),
        "all_inputs_unchanged": True,
        "production_files_changed": False,
        "exact_builder_source_verified": True,
    }


def _normalized_frozen_spec(record: Mapping[str, Any]) -> dict[str, Any]:
    missing = sorted(set(FROZEN_SPEC_FIELDS) - set(record))
    if missing:
        raise ValueError(f"frozen candidate spec is incomplete: {missing}")
    output: dict[str, Any] = {}
    for field in FROZEN_SPEC_FIELDS:
        value = record[field]
        if field in FROZEN_BOOL_FIELDS:
            output[field] = _as_bool(value)
        elif field in FROZEN_FLOAT_FIELDS:
            numeric = float(value)
            if not np.isfinite(numeric):
                raise ValueError(f"frozen candidate {field} is non-finite")
            output[field] = numeric
        elif field == "selection_ineligibility_reason":
            # pandas.read_csv materializes the frozen JSON's empty string as
            # NaN.  This is the sole optional string in the exact spec.
            output[field] = "" if pd.isna(value) else str(value)
        else:
            if pd.isna(value):
                raise ValueError(f"frozen candidate required string {field} is missing")
            output[field] = str(value)
    return output


def validate_frozen_catalog_contract(
    catalog: pd.DataFrame,
    source_manifest: Mapping[str, Any],
    frozen_catalog_path: Path,
    prediction_names: set[str] | None = None,
) -> dict[str, Any]:
    """Bind scoring to the byte-exact pre-sealed 92-spec catalog."""

    observed_sha = _sha256(frozen_catalog_path)
    if observed_sha != EXPECTED_FROZEN_CATALOG_SHA256:
        raise ValueError(
            "frozen candidate catalog hash differs from the launch contract: "
            f"expected={EXPECTED_FROZEN_CATALOG_SHA256}, observed={observed_sha}"
        )
    if str(source_manifest.get("frozen_candidate_catalog_sha256", "")).lower() != (
        EXPECTED_FROZEN_CATALOG_SHA256
    ):
        raise ValueError("source manifest does not bind the exact frozen catalog hash")
    payload = json.loads(frozen_catalog_path.read_text(encoding="utf-8"))
    if (
        payload.get("contract") != "base_skill_complete_candidate_catalog_v1"
        or payload.get("frozen_before_complete_candidate_scoring") is not True
        or payload.get("sealed_data_used_for_catalog") is not False
    ):
        raise ValueError("frozen candidate catalog provenance contract is unsafe")
    frozen_raw = payload.get("candidates")
    manifest_raw = source_manifest.get("candidates")
    if not isinstance(frozen_raw, list) or not isinstance(manifest_raw, list):
        raise ValueError("frozen/source manifests lack candidate spec lists")
    if len(frozen_raw) != EXPECTED_CLEAN_CANDIDATE_COUNT:
        raise ValueError(
            f"frozen catalog must contain exactly {EXPECTED_CLEAN_CANDIDATE_COUNT} specs"
        )
    frozen_specs = [_normalized_frozen_spec(record) for record in frozen_raw]
    manifest_specs = [_normalized_frozen_spec(record) for record in manifest_raw]
    frozen_names = [record["name"] for record in frozen_specs]
    if len(set(frozen_names)) != EXPECTED_CLEAN_CANDIDATE_COUNT:
        raise ValueError("frozen catalog candidate names are not exactly unique")
    by_name = lambda rows: sorted(rows, key=lambda record: record["name"])
    if by_name(manifest_specs) != by_name(frozen_specs):
        raise ValueError("source manifest candidate specs differ from frozen catalog")

    required_catalog_columns = set(FROZEN_SPEC_FIELDS)
    _require_columns(catalog, required_catalog_columns, "candidate catalog")
    clean_catalog = catalog.loc[~catalog["control"].map(_as_bool)].copy()
    if len(clean_catalog) != EXPECTED_CLEAN_CANDIDATE_COUNT:
        raise ValueError("CSV catalog does not contain exactly 92 clean candidate rows")
    csv_specs = [
        _normalized_frozen_spec(record)
        for record in clean_catalog[list(FROZEN_SPEC_FIELDS)].to_dict(orient="records")
    ]
    if by_name(csv_specs) != by_name(frozen_specs):
        raise ValueError("CSV candidate specs differ from the frozen JSON catalog")
    if prediction_names is not None:
        clean_prediction_names = prediction_names - set(KNOWN_CONTROL_UNITS)
        if clean_prediction_names != set(frozen_names):
            raise ValueError("prediction candidate names differ from the frozen 92-name set")
        if prediction_names & set(KNOWN_CONTROL_UNITS) != set(KNOWN_CONTROL_UNITS):
            raise ValueError("prediction panel does not contain all three frozen controls")
        unknown_controls = prediction_names - set(frozen_names) - set(KNOWN_CONTROL_UNITS)
        if unknown_controls:
            raise ValueError(f"prediction panel contains unknown names: {sorted(unknown_controls)}")
    return {
        "frozen_catalog_path": str(frozen_catalog_path.resolve()),
        "frozen_catalog_sha256": observed_sha,
        "clean_candidate_specs": int(len(frozen_specs)),
        "clean_candidate_names": sorted(frozen_names),
        "exact_spec_match": True,
    }


def validate_frozen_selection_cli(args: argparse.Namespace) -> dict[str, Any]:
    """Reject any runtime override that could change discovery selection."""

    expected = {
        "baseline_candidate": DEFAULT_BASELINE,
        "min_calibration_pairs": FROZEN_MIN_CALIBRATION_PAIRS,
        "finalist_count": FROZEN_FINALIST_COUNT,
        "coverage_tolerance": FROZEN_GATE_TOLERANCES["coverage"],
        "logloss_tolerance": FROZEN_GATE_TOLERANCES["logloss"],
        "calibration_tolerance": FROZEN_GATE_TOLERANCES["calibration"],
        "money_tolerance": FROZEN_GATE_TOLERANCES["money"],
        "pairwise_regularization_c": FROZEN_PAIRWISE_REGULARIZATION_C,
        "downstream_lambdas": FROZEN_DOWNSTREAM_LAMBDAS,
        "opener_weights": FROZEN_OPENER_WEIGHTS,
        "skip_sensitivities": False,
    }
    observed = {
        key: getattr(args, key)
        for key in expected
    }
    observed["downstream_lambdas"] = tuple(
        float(value) for value in observed["downstream_lambdas"]
    )
    observed["opener_weights"] = tuple(
        float(value) for value in observed["opener_weights"]
    )
    mismatches = {
        key: {"expected": expected[key], "observed": observed[key]}
        for key in expected
        if observed[key] != expected[key]
    }
    if mismatches:
        raise ValueError(f"selection-critical CLI differs from frozen contract: {mismatches}")
    contract = {
        "contract": "base_skill_discovery_selection_parameters_v1",
        "frozen_before_scoring": True,
        "sealed_data_used_to_set_parameters": False,
        **expected,
    }
    contract["downstream_lambdas"] = list(FROZEN_DOWNSTREAM_LAMBDAS)
    contract["opener_weights"] = list(FROZEN_OPENER_WEIGHTS)
    return contract


def validate_identical_prediction_menus(predictions: pd.DataFrame) -> dict[str, Any]:
    """Require every candidate/control to score the exact event/player universe."""

    keys = ["event_key", "dg_id_key"]
    candidates = sorted(predictions["candidate"].astype(str).unique())
    if not candidates:
        raise ValueError("prediction panel contains no candidates")
    if predictions.duplicated(["candidate", *keys]).any():
        raise ValueError("prediction panel has duplicate candidate/event/player keys")
    membership = predictions.groupby(keys, observed=True)["candidate"].nunique()
    bad = membership.ne(len(candidates))
    if bad.any():
        examples = [list(index) for index in membership.index[bad][:10]]
        raise ValueError(
            "discovery candidates do not share an exact event/player menu: "
            f"{examples}"
        )
    event_date_counts = predictions.groupby("event_key", observed=True)[
        "event_start_date"
    ].nunique(dropna=False)
    if event_date_counts.gt(1).any():
        raise ValueError("prediction event keys map to multiple start dates")
    return {
        "candidates": int(len(candidates)),
        "event_player_keys": int(len(membership)),
        "rows": int(len(predictions)),
        "menus_identical": True,
    }


def audit_calibration_warmup_menu(
    predictions: pd.DataFrame,
    required_candidates: Iterable[str],
) -> dict[str, Any]:
    """Require an exact, nonempty pre-discovery calibration menu.

    The first PGA event is normally after January 1.  Warm-up coverage means
    at least one event in the half-open 2020 calibration interval, not an
    observation on the interval's first calendar day.  Every selectable
    challenger and the strict point-in-time baseline must share the exact
    event/player menu; partial retrospective controls are audited separately.
    """

    _require_columns(
        predictions,
        {"candidate", "event_key", "dg_id_key", "event_start_date"},
        "calibration warm-up predictions",
    )
    required = {str(candidate) for candidate in required_candidates}
    if not required or DEFAULT_BASELINE not in required:
        raise ValueError(
            "calibration warm-up contract requires challengers and strict-PT baseline"
        )
    event_start = pd.to_datetime(predictions["event_start_date"], errors="coerce")
    warmup = predictions.loc[
        event_start.ge(WARMUP_START) & event_start.lt(DISCOVERY_START)
    ].copy()
    if warmup.empty or warmup["event_key"].nunique() == 0:
        raise ValueError(
            "prediction panel has no calibration warm-up event in "
            "[2020-01-01, 2021-01-01)"
        )
    observed = set(warmup["candidate"].astype(str))
    missing = sorted(required - observed)
    if missing:
        raise ValueError(
            "calibration warm-up is missing required candidate/control menus: "
            f"{missing}"
        )
    required_warmup = warmup.loc[
        warmup["candidate"].astype(str).isin(required)
    ].copy()
    keys = ["event_key", "dg_id_key"]
    if required_warmup.duplicated(["candidate", *keys]).any():
        raise ValueError("calibration warm-up has duplicate candidate/event/player keys")
    membership = required_warmup.groupby(keys, observed=True)["candidate"].nunique()
    incomplete = membership.ne(len(required))
    if incomplete.any():
        examples = [list(index) for index in membership.index[incomplete][:10]]
        raise ValueError(
            "required candidates do not share an exact calibration warm-up menu: "
            f"{examples}"
        )
    event_date_counts = required_warmup.groupby("event_key", observed=True)[
        "event_start_date"
    ].nunique(dropna=False)
    if event_date_counts.ne(1).any():
        raise ValueError("calibration warm-up event keys map to inconsistent dates")
    rows_by_candidate = (
        required_warmup.groupby("candidate", observed=True).size().astype(int)
    )
    events_by_candidate = (
        required_warmup.groupby("candidate", observed=True)["event_key"]
        .nunique()
        .astype(int)
    )
    return {
        "interval_start": WARMUP_START.date().isoformat(),
        "interval_end_exclusive": DISCOVERY_START.date().isoformat(),
        "first_event_start": pd.Timestamp(event_start.loc[warmup.index].min())
        .date()
        .isoformat(),
        "last_event_start": pd.Timestamp(event_start.loc[warmup.index].max())
        .date()
        .isoformat(),
        "required_candidates": int(len(required)),
        "event_player_keys": int(len(membership)),
        "events": int(required_warmup["event_key"].nunique()),
        "rows_by_candidate": {
            str(name): int(value) for name, value in rows_by_candidate.items()
        },
        "events_by_candidate": {
            str(name): int(value) for name, value in events_by_candidate.items()
        },
        "exact_required_menu": True,
        "retrospective_controls_scope": (
            "selection_ineligible_partial_coverage_audited_separately"
        ),
    }


def audit_discovery_only_envelope_admission(
    predictions: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    fit_log_path: Path,
    admission_candidate_path: Path,
    admission_fold_path: Path,
) -> dict[str, Any]:
    """Recompute envelope admission using only 2020-22 prediction folds."""

    _require_columns(
        catalog,
        {
            "name",
            "prediction_envelope_applied",
            "selection_eligible",
            "sensitivity_only",
            "control",
        },
        "envelope admission catalog",
    )
    clean_catalog = catalog.loc[~catalog["control"].map(_as_bool)].copy()
    regression_catalog = clean_catalog.loc[
        clean_catalog["prediction_envelope_applied"].map(_as_bool)
    ].copy()
    regression_names = set(regression_catalog["name"].astype(str))
    if len(regression_names) != EXPECTED_REGRESSION_CANDIDATE_COUNT:
        raise ValueError(
            "frozen envelope admission must contain "
            f"{EXPECTED_REGRESSION_CANDIDATE_COUNT} regression candidates"
        )
    required_prediction_columns = {
        "candidate",
        "event_start_date",
        "refit_cutoff",
        "training_max_event_end",
        "prediction_envelope_clipped",
    }
    _require_columns(
        predictions,
        required_prediction_columns,
        "discovery-only envelope predictions",
    )
    regression = predictions.loc[
        predictions["candidate"].astype(str).isin(regression_names)
    ].copy()
    if set(regression["candidate"].astype(str)) != regression_names:
        missing = sorted(regression_names - set(regression["candidate"].astype(str)))
        raise ValueError(f"discovery-only envelope predictions omit candidates: {missing}")
    event_start = pd.to_datetime(regression["event_start_date"], errors="coerce")
    refit_cutoff = pd.to_datetime(regression["refit_cutoff"], errors="coerce")
    training_max = pd.to_datetime(
        regression["training_max_event_end"], errors="coerce"
    )
    boundary_bad = (
        event_start.isna()
        | event_start.lt(WARMUP_START)
        | event_start.gt(DISCOVERY_END)
        | refit_cutoff.isna()
        | refit_cutoff.gt(DISCOVERY_END)
        | training_max.isna()
        | training_max.ge(refit_cutoff)
    )
    if boundary_bad.any():
        raise ValueError(
            "discovery-only envelope admission crosses event/refit/training cutoff"
        )
    if regression["prediction_envelope_clipped"].isna().any():
        raise ValueError("discovery-only envelope clipped flag is missing")
    regression["refit_cutoff"] = refit_cutoff.dt.normalize()
    regression["_clipped"] = regression["prediction_envelope_clipped"].map(_as_bool)
    direct_folds = (
        regression.groupby(["candidate", "refit_cutoff"], observed=True)
        .agg(rows=("candidate", "size"), clips=("_clipped", "sum"))
        .reset_index()
    )
    direct_folds["clip_rate"] = direct_folds["clips"] / direct_folds["rows"]
    direct_candidates = (
        direct_folds.groupby("candidate", observed=True)
        .agg(
            rows=("rows", "sum"),
            clips=("clips", "sum"),
            worst_fold_clip_rate=("clip_rate", "max"),
        )
        .reset_index()
    )
    direct_candidates["overall_clip_rate"] = (
        direct_candidates["clips"] / direct_candidates["rows"]
    )
    direct_candidates["fails_overall_gate"] = direct_candidates[
        "overall_clip_rate"
    ].gt(FROZEN_ENVELOPE_OVERALL_CLIP_RATE_MAX)
    direct_candidates["fails_fold_gate"] = direct_candidates[
        "worst_fold_clip_rate"
    ].gt(FROZEN_ENVELOPE_FOLD_CLIP_RATE_MAX)
    direct_candidates["guard_dependent"] = (
        direct_candidates["fails_overall_gate"]
        | direct_candidates["fails_fold_gate"]
    )
    discovery_guard = set(
        direct_candidates.loc[direct_candidates["guard_dependent"], "candidate"]
        .astype(str)
    )
    if len(discovery_guard) != EXPECTED_DISCOVERY_GUARD_DEPENDENT_COUNT:
        raise ValueError(
            "discovery-only envelope guard-dependent count differs from frozen "
            f"{EXPECTED_DISCOVERY_GUARD_DEPENDENT_COUNT}"
        )

    full_candidate = pd.read_csv(admission_candidate_path)
    _require_columns(
        full_candidate,
        {"candidate", "guard_dependent", "selection_eligible"},
        "full-sample envelope admission candidate audit",
    )
    if set(full_candidate["candidate"].astype(str)) != regression_names:
        raise ValueError("full-sample envelope candidate inventory is not exact")
    full_guard = set(
        full_candidate.loc[
            full_candidate["guard_dependent"].map(_as_bool), "candidate"
        ].astype(str)
    )
    if full_guard != discovery_guard:
        raise ValueError(
            "full-only and discovery-only envelope guard-dependent sets differ"
        )

    archived_folds = pd.read_csv(admission_fold_path)
    _require_columns(
        archived_folds,
        {"candidate", "refit_cutoff", "rows", "clips", "clip_rate"},
        "archived envelope fold audit",
    )
    archived_folds = archived_folds.loc[
        archived_folds["candidate"].astype(str).isin(regression_names)
    ].copy()
    archived_folds["refit_cutoff"] = pd.to_datetime(
        archived_folds["refit_cutoff"], errors="coerce"
    ).dt.normalize()
    archived_folds = archived_folds.loc[
        archived_folds["refit_cutoff"].le(DISCOVERY_END)
    ].copy()
    fold_compare = direct_folds.merge(
        archived_folds[["candidate", "refit_cutoff", "rows", "clips"]],
        on=["candidate", "refit_cutoff"],
        how="outer",
        suffixes=("_direct", "_archived"),
        indicator=True,
        validate="one_to_one",
    )
    fold_mismatch = (
        ~fold_compare["_merge"].eq("both")
        | fold_compare["rows_direct"].ne(fold_compare["rows_archived"])
        | fold_compare["clips_direct"].ne(fold_compare["clips_archived"])
    )
    if fold_mismatch.any():
        raise ValueError("discovery-only prediction/fold envelope admission differs")

    fit_log = pd.read_csv(fit_log_path)
    _require_columns(
        fit_log,
        {
            "candidate",
            "refit_cutoff",
            "status",
            "training_max_event_end",
            "test_rows",
            "prediction_envelope_clip_count",
        },
        "envelope fit log",
    )
    fit_log = fit_log.loc[fit_log["candidate"].astype(str).isin(regression_names)].copy()
    fit_log["refit_cutoff"] = pd.to_datetime(
        fit_log["refit_cutoff"], errors="coerce"
    ).dt.normalize()
    fit_log["training_max_event_end"] = pd.to_datetime(
        fit_log["training_max_event_end"], errors="coerce"
    ).dt.normalize()
    fit_log = fit_log.loc[fit_log["refit_cutoff"].le(DISCOVERY_END)].copy()
    if (
        ~fit_log["status"].astype(str).eq("fit")
        | fit_log["refit_cutoff"].isna()
        | fit_log["training_max_event_end"].isna()
        | fit_log["training_max_event_end"].ge(fit_log["refit_cutoff"])
    ).any():
        raise ValueError("discovery-only envelope fit chronology/status failed")
    fit_compare = direct_folds.merge(
        fit_log[
            [
                "candidate",
                "refit_cutoff",
                "test_rows",
                "prediction_envelope_clip_count",
            ]
        ],
        on=["candidate", "refit_cutoff"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    fit_mismatch = (
        ~fit_compare["_merge"].eq("both")
        | fit_compare["rows"].ne(fit_compare["test_rows"])
        | fit_compare["clips"].ne(
            fit_compare["prediction_envelope_clip_count"]
        )
    )
    if fit_mismatch.any():
        raise ValueError("discovery-only prediction/fit-log envelope admission differs")

    catalog_by_name = regression_catalog.set_index(
        regression_catalog["name"].astype(str)
    )
    expected_eligible = pd.Series(
        {
            name: name not in discovery_guard
            and not _as_bool(catalog_by_name.loc[name, "sensitivity_only"])
            for name in regression_names
        }
    )
    actual_eligible = pd.Series(
        {
            name: _as_bool(catalog_by_name.loc[name, "selection_eligible"])
            for name in regression_names
        }
    )
    if not actual_eligible.sort_index().equals(expected_eligible.sort_index()):
        raise ValueError(
            "discovery-only envelope admission differs from frozen selection eligibility"
        )
    full_eligible = full_candidate.set_index(
        full_candidate["candidate"].astype(str)
    )["selection_eligible"].map(_as_bool)
    if not full_eligible.sort_index().equals(actual_eligible.sort_index()):
        raise ValueError("archived envelope eligibility differs from frozen catalog")
    return {
        "contract": "warmup_discovery_only_regression_envelope_admission_v1",
        "event_start_max": str(pd.Timestamp(event_start.max()).date()),
        "refit_cutoff_max": str(pd.Timestamp(refit_cutoff.max()).date()),
        "regression_candidates": int(len(regression_names)),
        "prediction_rows": int(len(regression)),
        "folds": int(len(direct_folds)),
        "guard_dependent_count": int(len(discovery_guard)),
        "guard_dependent_candidates": sorted(discovery_guard),
        "overall_clip_rate_max": FROZEN_ENVELOPE_OVERALL_CLIP_RATE_MAX,
        "fold_clip_rate_max": FROZEN_ENVELOPE_FOLD_CLIP_RATE_MAX,
        "full_and_discovery_guard_sets_equal": True,
        "frozen_selection_eligibility_exact": True,
        "sealed_prediction_rows_used": 0,
        "sealed_fit_rows_used": 0,
    }


def validate_discovery_candidate_set(
    predictions: pd.DataFrame,
    frozen_clean_candidate_names: Iterable[str],
) -> dict[str, Any]:
    """Require the discovery slice to retain every frozen challenger/control.

    The builder emits discovery and sealed predictions in one panel.  Validating
    the full panel alone is insufficient: a candidate could otherwise exist only
    after the discovery cutoff and silently disappear before model selection.
    """

    expected_clean = {str(name) for name in frozen_clean_candidate_names}
    if len(expected_clean) != EXPECTED_CLEAN_CANDIDATE_COUNT:
        raise ValueError(
            "discovery candidate-set audit did not receive the exact frozen "
            f"{EXPECTED_CLEAN_CANDIDATE_COUNT}-name clean set"
        )
    expected = expected_clean | set(KNOWN_CONTROL_UNITS)
    observed = set(predictions["candidate"].astype(str))
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing or unexpected:
        raise ValueError(
            "discovery-filtered prediction panel must retain the exact frozen "
            "candidate/control set: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {
        "expected_clean_candidates": int(len(expected_clean)),
        "expected_controls": int(len(KNOWN_CONTROL_UNITS)),
        "observed_candidate_units": int(len(observed)),
        "exact_post_filter_name_set": True,
    }


def audit_retrospective_control_coverage(
    predictions: pd.DataFrame,
    reference_candidate: str = DEFAULT_BASELINE,
) -> dict[str, Any]:
    """Report, but never impute, partial retrospective-control coverage.

    The two retrospective controls exist only where their historical source
    values were archived.  They are descriptive and selection-ineligible, so
    their missing rows must not shrink the complete challenger/baseline menu.
    Any row outside the strict point-in-time reference menu is still invalid.
    """

    keys = ["event_key", "dg_id_key"]
    reference = predictions.loc[
        predictions["candidate"].astype(str).eq(reference_candidate), keys
    ].drop_duplicates()
    if reference.empty:
        raise ValueError("retrospective-control audit has no reference menu")
    reference_index = pd.MultiIndex.from_frame(reference)
    rows: dict[str, dict[str, Any]] = {}
    retrospective_names = sorted(set(KNOWN_CONTROL_UNITS) - {reference_candidate})
    for name in retrospective_names:
        control = predictions.loc[
            predictions["candidate"].astype(str).eq(name), keys
        ].drop_duplicates()
        if control.empty:
            raise ValueError(f"retrospective control is absent after discovery filter: {name}")
        control_index = pd.MultiIndex.from_frame(control)
        extra = control_index.difference(reference_index)
        if len(extra):
            raise ValueError(
                f"retrospective control contains keys outside the reference menu: "
                f"{name}, examples={[list(value) for value in extra[:10]]}"
            )
        missing = reference_index.difference(control_index)
        rows[name] = {
            "prediction_rows": int(len(control_index)),
            "reference_rows": int(len(reference_index)),
            "missing_reference_rows": int(len(missing)),
            "coverage_ratio": float(len(control_index) / len(reference_index)),
            "partial_coverage_allowed": True,
            "selection_eligible": False,
            "missing_rows_imputed": 0,
        }
    return {
        "reference_candidate": reference_candidate,
        "reference_rows": int(len(reference_index)),
        "controls": rows,
        "retrospective_controls_excluded_from_exact_menu_gate": retrospective_names,
    }


def reconcile_known_controls(
    predictions: pd.DataFrame,
    catalog: pd.DataFrame,
    source_manifest: Mapping[str, Any],
    source_manifest_path: Path,
    *,
    expected_coverage_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Allow only the builder's three declared, count-audited controls."""

    prediction_names = set(predictions["candidate"].astype(str))
    catalog_names = set(catalog["name"].astype(str))
    missing = prediction_names - catalog_names
    unknown = sorted(missing - set(KNOWN_CONTROL_UNITS))
    if unknown:
        raise ValueError(
            "predictions contain non-catalog, non-allowlisted candidates: "
            f"{unknown}"
        )

    declared_raw = source_manifest.get("control_catalog")
    if not isinstance(declared_raw, list):
        raise ValueError("source manifest does not declare its control catalog")
    declared = {
        str(entry.get("name")): entry
        for entry in declared_raw
        if isinstance(entry, Mapping) and entry.get("name")
    }
    controls_present = sorted(prediction_names & set(KNOWN_CONTROL_UNITS))
    undeclared = sorted(set(controls_present) - set(declared))
    if undeclared:
        raise ValueError(f"prediction controls are undeclared in manifest: {undeclared}")
    if not isinstance(source_manifest.get("control_audit"), Mapping):
        raise ValueError("source manifest lacks the builder control audit")
    for name in controls_present:
        entry = declared[name]
        if (
            _as_bool(entry.get("selection_eligible"))
            or not _as_bool(entry.get("sensitivity_only"))
            or not _as_bool(entry.get("control"))
        ):
            raise ValueError(f"manifest control flags are unsafe for {name}")

    coverage_entry = source_manifest.get("outputs", {}).get(
        "coverage_by_candidate_year.csv"
    )
    if not isinstance(coverage_entry, Mapping):
        raise ValueError("source manifest lacks hashed candidate/year coverage")
    coverage_path = Path(str(coverage_entry.get("path", "")))
    if not coverage_path.is_absolute():
        coverage_path = source_manifest_path.parent / coverage_path
    coverage_path = coverage_path.resolve()
    if (
        expected_coverage_path is not None
        and coverage_path != expected_coverage_path.resolve()
    ):
        raise ValueError("source control count-audit path changed after provenance begin")
    expected_hash = str(coverage_entry.get("sha256", ""))
    if not coverage_path.exists() or not expected_hash:
        raise ValueError("source control count-audit artifact is unavailable")
    if _sha256(coverage_path) != expected_hash:
        raise ValueError("source control count-audit hash mismatch")
    coverage = pd.read_csv(coverage_path)
    _require_columns(
        coverage,
        {"candidate", "prediction_rows"},
        "candidate/year coverage audit",
    )
    audited_counts = (
        coverage.groupby("candidate", observed=True)["prediction_rows"].sum().to_dict()
    )
    actual_counts = predictions.groupby("candidate", observed=True).size().to_dict()
    for name in controls_present:
        if int(audited_counts.get(name, -1)) != int(actual_counts[name]):
            raise ValueError(
                f"manifest-hashed control count disagrees for {name}: "
                f"audit={audited_counts.get(name)}, actual={actual_counts[name]}"
            )

    additions: list[dict[str, Any]] = []
    for name in sorted(missing):
        additions.append(
            {
                "name": name,
                **KNOWN_CONTROL_METADATA[name],
                "selection_eligible": False,
                "sensitivity_only": True,
                "control": True,
                "description": declared[name].get(
                    "description", "Builder-declared non-selectable control."
                ),
            }
        )
    if additions:
        catalog = pd.concat([catalog, pd.DataFrame(additions)], ignore_index=True, sort=False)
    for column in ("selection_eligible", "sensitivity_only", "control"):
        catalog[column] = catalog[column].map(_as_bool)
    for name in controls_present:
        row = catalog.loc[catalog["name"].astype(str).eq(name)]
        if len(row) != 1:
            raise ValueError(f"control catalog row is missing or nonunique: {name}")
        if (
            bool(row.iloc[0]["selection_eligible"])
            or not bool(row.iloc[0]["sensitivity_only"])
            or not bool(row.iloc[0]["control"])
        ):
            raise ValueError(f"control catalog flags are unsafe for {name}")
    return catalog, {
        "allowlisted_controls_present": controls_present,
        "synthesized_controls": sorted(missing),
        "count_audit_path": str(coverage_path.resolve()),
        "count_audit_sha256": expected_hash,
        "prediction_counts": {
            name: int(actual_counts[name]) for name in controls_present
        },
    }


def validate_prediction_units(predictions: pd.DataFrame) -> dict[str, str]:
    """Enforce SG units, with exact labels only for allowlisted controls."""

    units = predictions.groupby("candidate", observed=True)["prediction_units"].agg(
        lambda values: sorted(set(values.astype(str)))
    )
    bad_units: dict[str, list[str]] = {}
    accepted: dict[str, str] = {}
    for candidate, values in units.items():
        name = str(candidate)
        expected_unit = KNOWN_CONTROL_UNITS.get(name, "sg_per_round")
        if values != [expected_unit]:
            bad_units[name] = values
        else:
            accepted[name] = expected_unit
    if bad_units:
        raise ValueError(
            "external candidate/control units violate the allowlist: "
            f"{bad_units}"
        )
    return accepted


def validate_completed_study_authority(
    source_manifest_path: Path,
    predictions_path: Path,
    catalog_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path]]:
    """Bind discovery to one byte-exact completed raw candidate study."""

    source_manifest_path = source_manifest_path.resolve()
    predictions_path = predictions_path.resolve()
    catalog_path = catalog_path.resolve()
    if source_manifest_path != DEFAULT_SOURCE_MANIFEST.resolve():
        raise ValueError("candidate manifest is not the audited completed study authority")
    if not source_manifest_path.is_file():
        raise FileNotFoundError(source_manifest_path)
    manifest_stat = source_manifest_path.stat()
    if manifest_stat.st_size != EXPECTED_COMPLETED_STUDY_MANIFEST_BYTES:
        raise ValueError("completed study manifest byte size differs from authority")
    manifest_sha = _sha256(source_manifest_path)
    if manifest_sha != EXPECTED_COMPLETED_STUDY_MANIFEST_SHA256:
        raise ValueError("completed study manifest hash differs from authority")
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("complete") is not True
        or manifest.get("research_only") is not True
        or manifest.get("production_files_changed") is not False
    ):
        raise ValueError("completed study authority is not complete research-only output")
    if str(manifest.get("frozen_candidate_catalog_sha256", "")).lower() != (
        EXPECTED_FROZEN_CATALOG_SHA256
    ):
        raise ValueError("completed study authority frozen-catalog hash differs")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("completed study authority lacks output bindings")
    binding_names = {
        "player_event_predictions.parquet": "predictions",
        "candidate_catalog.csv": "candidate_catalog",
        "coverage_by_candidate_year.csv": "coverage_by_candidate_year",
        "candidate_catalog_frozen.json": "frozen_candidate_catalog",
        "fit_log.csv": "fit_log",
        "regression_envelope_admission_by_candidate.csv": (
            "envelope_admission_by_candidate"
        ),
        "regression_envelope_admission_by_fold.csv": "envelope_admission_by_fold",
    }
    bindings: dict[str, Path] = {"completion_manifest": source_manifest_path}
    output_audit: dict[str, Any] = {}
    for name, expected in EXPECTED_COMPLETED_STUDY_OUTPUTS.items():
        record = outputs.get(name)
        if not isinstance(record, Mapping) or not record.get("path"):
            raise ValueError(f"completed study authority omits output {name!r}")
        path = Path(str(record["path"])).resolve()
        expected_path = (source_manifest_path.parent / name).resolve()
        recorded_sha = str(record.get("sha256", "")).lower()
        expected_sha = str(expected["sha256"]).lower()
        if path != expected_path or recorded_sha != expected_sha:
            raise ValueError(f"completed study output authority differs: {name!r}")
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        if stat.st_size != int(expected["bytes"]) or _sha256(path) != expected_sha:
            raise ValueError(f"completed study output bytes differ: {name!r}")
        bindings[binding_names[name]] = path
        output_audit[name] = {
            "path": str(path),
            "sha256": expected_sha,
            "bytes": int(expected["bytes"]),
        }
    if predictions_path != bindings["predictions"]:
        raise ValueError("prediction CLI path differs from completed study authority")
    if catalog_path != bindings["candidate_catalog"]:
        raise ValueError("candidate-catalog CLI path differs from completed study authority")
    if tuple(bindings) != COMPLETED_STUDY_AUTHORITY_KEYS:
        raise AssertionError("completed study authority binding order changed")
    return manifest, {
        "manifest": {
            "path": str(source_manifest_path),
            "sha256": manifest_sha,
            "bytes": int(manifest_stat.st_size),
        },
        "outputs": output_audit,
        "exact_completed_study": True,
    }, bindings


def validate_evaluation_cache_authority(
    cache_dir: Path,
    core_bridge_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path]]:
    """Bind discovery to the exact audited V4 cache and its source inputs."""

    cache_dir = cache_dir.resolve()
    core_bridge_path = core_bridge_path.resolve()
    if cache_dir != DEFAULT_CACHE.resolve():
        raise ValueError("evaluation cache directory is not the audited V4 authority")
    manifest_path = (cache_dir / "manifest.json").resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest_stat = manifest_path.stat()
    if manifest_stat.st_size != EXPECTED_EVALUATION_CACHE_MANIFEST_BYTES:
        raise ValueError("evaluation cache manifest byte size is not the audited value")
    manifest_sha256 = _sha256(manifest_path)
    if manifest_sha256 != EXPECTED_EVALUATION_CACHE_MANIFEST_SHA256:
        raise ValueError("evaluation cache manifest hash is not the audited V4 authority")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True or manifest.get("research_only") is not True:
        raise ValueError("evaluation cache authority is not complete research-only output")
    if manifest.get("run_type") != EXPECTED_EVALUATION_CACHE_RUN_TYPE:
        raise ValueError("evaluation cache authority has an unexpected run_type")
    declared_contract = manifest.get("contract")
    if not declared_contract or Path(str(declared_contract)).resolve() != (
        EXPECTED_EVALUATION_CACHE_CONTRACT.resolve()
    ):
        raise ValueError("evaluation cache authority has an unexpected contract")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("evaluation cache authority lacks its artifact inventory")
    artifact_by_name: dict[str, Mapping[str, Any]] = {}
    for record in artifacts:
        if not isinstance(record, Mapping) or not record.get("path"):
            raise ValueError("evaluation cache artifact inventory contains an invalid row")
        name = str(record["path"])
        if name in artifact_by_name:
            raise ValueError(f"evaluation cache artifact inventory duplicates {name!r}")
        artifact_by_name[name] = record

    bindings: dict[str, Path] = {"evaluation_cache_manifest": manifest_path}
    artifact_audit: dict[str, Any] = {}
    artifact_binding_names = {
        "normalized_matchup_quotes.parquet": "normalized_quotes",
        "feature_snapshots.parquet": "feature_snapshots",
    }
    for name, expected in EXPECTED_EVALUATION_CACHE_ARTIFACTS.items():
        record = artifact_by_name.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"evaluation cache authority omits artifact {name!r}")
        path = (manifest_path.parent / name).resolve()
        if path.parent != cache_dir:
            raise ValueError(f"evaluation cache artifact escapes authority root: {name!r}")
        recorded_sha = str(record.get("sha256", "")).lower()
        recorded_bytes = int(record.get("bytes", -1))
        if (
            recorded_sha != str(expected["sha256"]).lower()
            or recorded_bytes != int(expected["bytes"])
        ):
            raise ValueError(f"evaluation cache artifact record is not frozen: {name!r}")
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        if stat.st_size != int(expected["bytes"]) or _sha256(path) != recorded_sha:
            raise ValueError(f"evaluation cache artifact bytes differ: {name!r}")
        bindings[artifact_binding_names[name]] = path
        artifact_audit[name] = {
            "path": str(path),
            "sha256": recorded_sha,
            "bytes": recorded_bytes,
        }

    declared_inputs = manifest.get("inputs")
    if not isinstance(declared_inputs, Mapping) or set(declared_inputs) != set(
        EXPECTED_EVALUATION_CACHE_INPUTS
    ):
        raise ValueError("evaluation cache authority input inventory is not exact")
    input_audit: dict[str, Any] = {}
    for name, expected in EXPECTED_EVALUATION_CACHE_INPUTS.items():
        record = declared_inputs.get(name)
        if not isinstance(record, Mapping) or not record.get("path"):
            raise ValueError(f"evaluation cache authority omits input {name!r}")
        path = Path(str(record["path"])).resolve()
        expected_path = Path(expected["path"]).resolve()
        recorded_sha = str(record.get("sha256", "")).lower()
        expected_sha = str(expected["sha256"]).lower()
        if path != expected_path or recorded_sha != expected_sha:
            raise ValueError(f"evaluation cache input authority differs: {name!r}")
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        if stat.st_size != int(expected["bytes"]) or _sha256(path) != expected_sha:
            raise ValueError(f"evaluation cache input bytes differ: {name!r}")
        bindings[f"cache_input_{name}"] = path
        input_audit[name] = {
            "path": str(path),
            "sha256": expected_sha,
            "bytes": int(expected["bytes"]),
        }
    if core_bridge_path != bindings["cache_input_legacy_core_bridge"]:
        raise ValueError("discovery core bridge differs from V4 cache authority")
    if tuple(bindings) != EVALUATION_CACHE_AUTHORITY_KEYS:
        raise AssertionError("evaluation cache authority binding order changed")
    return manifest, {
        "manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha256,
            "bytes": int(manifest_stat.st_size),
            "contract": str(EXPECTED_EVALUATION_CACHE_CONTRACT.resolve()),
            "run_type": EXPECTED_EVALUATION_CACHE_RUN_TYPE,
        },
        "artifacts": artifact_audit,
        "inputs": input_audit,
    }, bindings


def resolve_discovery_direct_inputs(
    predictions_path: Path,
    catalog_path: Path,
    source_manifest_path: Path,
    cache_dir: Path,
    core_bridge_path: Path,
) -> tuple[dict[str, Any], dict[str, Path], dict[str, Any], dict[str, Path]]:
    """Resolve and hash-check the exact nine discovery inputs from authority."""

    predictions_path = predictions_path.resolve()
    catalog_path = catalog_path.resolve()
    source_manifest_path = source_manifest_path.resolve()
    cache_dir = cache_dir.resolve()
    core_bridge_path = core_bridge_path.resolve()
    source_manifest, study_audit, study_bindings = validate_completed_study_authority(
        source_manifest_path, predictions_path, catalog_path
    )
    _required_manifest_output(
        source_manifest,
        source_manifest_path,
        "player_event_predictions.parquet",
        expected_path=predictions_path,
    )
    _required_manifest_output(
        source_manifest,
        source_manifest_path,
        "candidate_catalog.csv",
        expected_path=catalog_path,
    )
    frozen_catalog_path, _ = _required_manifest_output(
        source_manifest,
        source_manifest_path,
        "candidate_catalog_frozen.json",
    )
    coverage_path, _ = _required_manifest_output(
        source_manifest,
        source_manifest_path,
        "coverage_by_candidate_year.csv",
    )
    bindings = {
        "predictions": predictions_path,
        "candidate_catalog": catalog_path,
        "frozen_candidate_catalog": frozen_catalog_path.resolve(),
        "candidate_manifest": source_manifest_path,
        "normalized_quotes": (cache_dir / "normalized_matchup_quotes.parquet").resolve(),
        "feature_snapshots": (cache_dir / "feature_snapshots.parquet").resolve(),
        "evaluation_cache_manifest": (cache_dir / "manifest.json").resolve(),
        "coverage_by_candidate_year": coverage_path.resolve(),
        "core_bridge": core_bridge_path,
    }
    if tuple(bindings) != DISCOVERY_DIRECT_INPUT_KEYS:
        raise AssertionError("discovery direct-input contract ordering changed")
    for name, path in bindings.items():
        if not path.is_file():
            raise FileNotFoundError(f"discovery input {name}: {path}")
    return source_manifest, bindings, study_audit, study_bindings


def revalidate_discovery_inputs_after_begin(
    *,
    predictions_path: Path,
    catalog_path: Path,
    source_manifest_path: Path,
    cache_dir: Path,
    core_bridge_path: Path,
    expected_bindings: Mapping[str, Path],
    expected_study_authority_bindings: Mapping[str, Path],
    expected_cache_authority_bindings: Mapping[str, Path],
    provenance_state: execution_provenance.ExecutionProvenance,
) -> tuple[
    dict[str, Any],
    dict[str, Path],
    dict[str, Any],
    dict[str, Any],
    dict[str, Path],
    dict[str, Any],
    dict[str, Path],
]:
    """Close source/cache-manifest path switching before any discovery read."""

    if tuple(expected_bindings) != DISCOVERY_DIRECT_INPUT_KEYS:
        raise ValueError("pre-begin discovery input names are not the exact frozen nine")
    if tuple(expected_study_authority_bindings) != COMPLETED_STUDY_AUTHORITY_KEYS:
        raise ValueError("pre-begin completed-study authority bindings are not exact")
    if tuple(expected_cache_authority_bindings) != EVALUATION_CACHE_AUTHORITY_KEYS:
        raise ValueError("pre-begin evaluation-cache authority bindings are not exact")
    source_manifest, current, study_audit, current_study_authority = resolve_discovery_direct_inputs(
        predictions_path,
        catalog_path,
        source_manifest_path,
        cache_dir,
        core_bridge_path,
    )
    expected = {name: Path(path).resolve() for name, path in expected_bindings.items()}
    if current != expected:
        raise ValueError("discovery direct-input paths changed across provenance begin")
    expected_study = {
        name: Path(path).resolve()
        for name, path in expected_study_authority_bindings.items()
    }
    if current_study_authority != expected_study:
        raise ValueError("completed-study authority paths changed across provenance begin")
    cache_manifest, cache_audit, current_cache_authority = (
        validate_evaluation_cache_authority(cache_dir, core_bridge_path)
    )
    expected_cache = {
        name: Path(path).resolve()
        for name, path in expected_cache_authority_bindings.items()
    }
    if current_cache_authority != expected_cache:
        raise ValueError(
            "evaluation-cache authority paths changed across provenance begin"
        )
    for direct_name, authority_name in (
        ("evaluation_cache_manifest", "evaluation_cache_manifest"),
        ("normalized_quotes", "normalized_quotes"),
        ("feature_snapshots", "feature_snapshots"),
        ("core_bridge", "cache_input_legacy_core_bridge"),
    ):
        if current[direct_name] != current_cache_authority[authority_name]:
            raise ValueError(
                f"discovery direct input is not cache-authorized: {direct_name}"
            )
    direct_paths = set(expected.values())
    if len(direct_paths) != len(DISCOVERY_DIRECT_INPUT_KEYS):
        raise ValueError("discovery direct inputs do not resolve to nine distinct files")
    expected_paths = direct_paths | set(expected_cache.values()) | set(expected_study.values())
    if len(expected_paths) != 16:
        raise ValueError(
            "discovery/cache/completed-study authority must resolve to 16 exact files"
        )
    tracked = {
        Path(str(record["path"])).resolve()
        for record in provenance_state.input_fingerprints
    }
    if tracked != expected_paths:
        raise ValueError(
            "execution provenance paths differ from discovery/cache authority"
        )
    return (
        source_manifest,
        current,
        cache_manifest,
        cache_audit,
        current_cache_authority,
        study_audit,
        current_study_authority,
    )


def load_and_validate_inputs(
    predictions_path: Path,
    catalog_path: Path,
    source_manifest_path: Path,
    cache_dir: Path,
    *,
    source_manifest_payload: Mapping[str, Any] | None = None,
    cache_manifest_payload: Mapping[str, Any] | None = None,
    cache_authority_audit: Mapping[str, Any] | None = None,
    completed_study_authority_audit: Mapping[str, Any] | None = None,
    expected_frozen_catalog_path: Path | None = None,
    expected_coverage_path: Path | None = None,
    expected_normalized_quotes_path: Path | None = None,
    expected_feature_snapshots_path: Path | None = None,
    expected_fit_log_path: Path | None = None,
    expected_envelope_candidate_path: Path | None = None,
    expected_envelope_fold_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    for path in (predictions_path, catalog_path, source_manifest_path):
        if not path.exists():
            raise FileNotFoundError(path)
    for name in (
        "normalized_matchup_quotes.parquet",
        "feature_snapshots.parquet",
        "manifest.json",
    ):
        if not (cache_dir / name).exists():
            raise FileNotFoundError(cache_dir / name)

    source_manifest = (
        dict(source_manifest_payload)
        if source_manifest_payload is not None
        else json.loads(source_manifest_path.read_text(encoding="utf-8"))
    )
    if source_manifest.get("complete") is not True:
        raise ValueError("external prediction manifest is not complete")
    completion_provenance_audit = validate_completion_manifest_provenance(
        source_manifest,
        source_manifest_path,
    )
    contract = source_manifest.get("contract", {})
    if isinstance(contract, Mapping):
        if contract.get("selection_uses_sealed") is True:
            raise ValueError("prediction source declares sealed selection")
        discovery_end = pd.to_datetime(contract.get("discovery_end"), errors="coerce")
        if pd.notna(discovery_end) and discovery_end > DISCOVERY_END:
            raise ValueError("prediction source discovery cutoff exceeds 2022-12-31")

    _, expected_prediction_hash = _required_manifest_output(
        source_manifest,
        source_manifest_path,
        "player_event_predictions.parquet",
        expected_path=predictions_path,
    )
    _, expected_catalog_hash = _required_manifest_output(
        source_manifest,
        source_manifest_path,
        "candidate_catalog.csv",
        expected_path=catalog_path,
    )
    frozen_catalog_path, frozen_catalog_hash = _required_manifest_output(
        source_manifest,
        source_manifest_path,
        "candidate_catalog_frozen.json",
        expected_path=expected_frozen_catalog_path,
    )
    if frozen_catalog_hash != EXPECTED_FROZEN_CATALOG_SHA256:
        raise ValueError("source manifest frozen-catalog output hash is not approved")

    predictions = pd.read_parquet(predictions_path)
    _require_columns(predictions, REQUIRED_PREDICTION_COLUMNS, "external predictions")
    predictions = predictions.copy()
    predictions["event_start_date"] = pd.to_datetime(
        predictions["event_start_date"], errors="raise"
    ).dt.normalize()
    predictions["prediction_asof"] = pd.to_datetime(
        predictions["prediction_asof"], errors="coerce"
    ).dt.normalize()
    predictions["training_max_event_end"] = pd.to_datetime(
        predictions["training_max_event_end"], errors="coerce"
    ).dt.normalize()
    predictions["dg_id_key"] = predictions["dg_id_key"].map(market_eval._stable_id)
    predictions["prediction"] = pd.to_numeric(predictions["prediction"], errors="coerce")
    full_predictions_for_control_audit = predictions[["candidate"]].copy()
    total_prediction_rows = int(len(predictions))
    sealed_rows_excluded = int((predictions["event_start_date"] > DISCOVERY_END).sum())
    # The completion builder may emit the whole walk-forward panel in one file.
    # Selection is physically restricted before any score, aggregation, or
    # candidate comparison is computed.
    predictions = predictions.loc[
        predictions["event_start_date"].between(WARMUP_START, DISCOVERY_END)
    ].copy()
    source_manifest = dict(source_manifest)
    source_manifest["_completion_provenance_audit"] = completion_provenance_audit
    source_manifest["_selection_filter_audit"] = {
        "source_prediction_rows": total_prediction_rows,
        "sealed_prediction_rows_excluded_before_scoring": sealed_rows_excluded,
        "selection_prediction_rows": int(len(predictions)),
    }
    if predictions.duplicated(["candidate", "event_key", "dg_id_key"]).any():
        raise AssertionError("external predictions have duplicate candidate/event/player keys")
    if not predictions["chronology_ok"].map(_as_bool).all():
        raise AssertionError("external prediction chronology_ok is not universally true")
    asof_bad = (
        predictions["prediction_asof"].notna()
        & (predictions["prediction_asof"] > predictions["event_start_date"])
    )
    training_bad = (
        predictions["training_max_event_end"].notna()
        & (predictions["training_max_event_end"] >= predictions["event_start_date"])
    )
    if asof_bad.any() or training_bad.any():
        raise AssertionError(
            f"prediction chronology failed: asof={int(asof_bad.sum())}, "
            f"training={int(training_bad.sum())}"
        )
    if (~np.isfinite(predictions["prediction"])).any():
        raise ValueError("external prediction panel contains nonfinite predictions")

    catalog = pd.read_csv(catalog_path)
    _require_columns(
        catalog,
        {"name", "selection_eligible", "sensitivity_only", "control", "family"},
        "candidate catalog",
    )
    catalog = catalog.copy()
    for column in ("selection_eligible", "sensitivity_only", "control"):
        catalog[column] = catalog[column].map(_as_bool)
    catalog, control_integration_audit = reconcile_known_controls(
        full_predictions_for_control_audit,
        catalog,
        source_manifest,
        source_manifest_path,
        expected_coverage_path=expected_coverage_path,
    )
    source_manifest["_control_integration_audit"] = control_integration_audit
    frozen_catalog_audit = validate_frozen_catalog_contract(
        catalog,
        source_manifest,
        frozen_catalog_path,
        set(full_predictions_for_control_audit["candidate"].astype(str)),
    )
    source_manifest["_frozen_catalog_audit"] = frozen_catalog_audit
    discovery_candidate_set_audit = validate_discovery_candidate_set(
        predictions,
        frozen_catalog_audit["clean_candidate_names"],
    )
    exact_menu_names = set(frozen_catalog_audit["clean_candidate_names"]) | {
        DEFAULT_BASELINE
    }
    warmup_audit = audit_calibration_warmup_menu(predictions, exact_menu_names)
    full_event_date_counts = predictions.groupby("event_key", observed=True)[
        "event_start_date"
    ].nunique(dropna=False)
    if full_event_date_counts.gt(1).any():
        raise ValueError(
            "prediction event keys map to multiple start dates when retrospective "
            "controls are included"
        )
    exact_menu_predictions = predictions.loc[
        predictions["candidate"].astype(str).isin(exact_menu_names)
    ].copy()
    menu_audit = validate_identical_prediction_menus(exact_menu_predictions)
    menu_audit["scope"] = "frozen_92_clean_candidates_plus_strict_pt_baseline"
    menu_audit["selection_candidate_units"] = int(len(exact_menu_names))
    menu_audit["all_candidate_event_dates_consistent"] = True
    retrospective_control_audit = audit_retrospective_control_coverage(predictions)
    if completed_study_authority_audit is None:
        raise ValueError("post-begin completed-study authority validation is required")
    authority_paths = {
        "fit_log": expected_fit_log_path,
        "envelope_admission_by_candidate": expected_envelope_candidate_path,
        "envelope_admission_by_fold": expected_envelope_fold_path,
    }
    if any(path is None for path in authority_paths.values()):
        raise ValueError("completed-study envelope authority paths are required")
    envelope_admission_audit = audit_discovery_only_envelope_admission(
        predictions,
        catalog,
        fit_log_path=Path(expected_fit_log_path).resolve(),
        admission_candidate_path=Path(expected_envelope_candidate_path).resolve(),
        admission_fold_path=Path(expected_envelope_fold_path).resolve(),
    )
    source_manifest["_completed_study_authority_audit"] = dict(
        completed_study_authority_audit
    )
    source_manifest["_discovery_envelope_admission_audit"] = (
        envelope_admission_audit
    )
    source_manifest["_discovery_candidate_set_audit"] = (
        discovery_candidate_set_audit
    )
    source_manifest["_identical_menu_audit"] = menu_audit
    source_manifest["_calibration_warmup_audit"] = warmup_audit
    source_manifest["_retrospective_control_menu_audit"] = (
        retrospective_control_audit
    )
    catalog_names = set(catalog["name"].astype(str))
    prediction_names = set(predictions["candidate"].astype(str))
    if prediction_names - catalog_names:
        raise ValueError(
            "predictions contain candidates missing from catalog: "
            f"{sorted(prediction_names - catalog_names)}"
        )
    validate_prediction_units(predictions)

    if cache_manifest_payload is None or cache_authority_audit is None:
        raise ValueError("post-begin evaluation-cache authority validation is required")
    quotes_path = (cache_dir / "normalized_matchup_quotes.parquet").resolve()
    snapshots_path = (cache_dir / "feature_snapshots.parquet").resolve()
    if (
        expected_normalized_quotes_path is None
        or quotes_path != expected_normalized_quotes_path.resolve()
        or expected_feature_snapshots_path is None
        or snapshots_path != expected_feature_snapshots_path.resolve()
    ):
        raise ValueError("post-begin evaluation-cache artifact paths are not bound")
    cache_manifest = dict(cache_manifest_payload)
    if cache_manifest.get("complete") is not True:
        raise ValueError("evaluation cache manifest is not complete")
    cache_manifest["_authority_audit"] = dict(cache_authority_audit)
    quotes = pd.read_parquet(quotes_path)
    _require_columns(
        quotes,
        {
            "quote_id",
            "mapped_event_key",
            "event_start_date",
            "market",
            "book",
            "pair_key",
            "p1_id",
            "p2_id",
            "p1_open",
            "p2_open",
            "p1_close",
            "p2_close",
            "p1_outcome",
            "p2_outcome",
        },
        "normalized matchup cache",
    )
    quotes["event_start_date"] = pd.to_datetime(quotes["event_start_date"]).dt.normalize()
    quotes = quotes.loc[
        quotes["event_start_date"].between(WARMUP_START, DISCOVERY_END)
    ].copy()
    quote_integrity_audit = market_eval.validate_normalized_matchup_quotes(quotes)
    cache_manifest = dict(cache_manifest)
    cache_manifest["_quote_integrity_audit"] = quote_integrity_audit
    valid_open = valid_american_odds_mask(
        pd.to_numeric(quotes["p1_open"], errors="coerce")
    ) & valid_american_odds_mask(pd.to_numeric(quotes["p2_open"], errors="coerce"))
    valid_close = valid_american_odds_mask(
        pd.to_numeric(quotes["p1_close"], errors="coerce")
    ) & valid_american_odds_mask(pd.to_numeric(quotes["p2_close"], errors="coerce"))
    cache_manifest["_same_book_price_coverage_audit"] = {
        "quote_rows": int(len(quotes)),
        "valid_two_sided_open_rows": int(valid_open.sum()),
        "invalid_two_sided_open_rows": int((~valid_open).sum()),
        "valid_two_sided_close_rows": int(valid_close.sum()),
        "invalid_two_sided_close_rows": int((~valid_close).sum()),
        "strict_conversion_after_filter": True,
    }
    if quotes.duplicated("quote_id").any():
        raise AssertionError("cached normalized quote IDs are not unique")
    if quotes.duplicated(["mapped_event_key", "market", "book", "pair_key"]).any():
        raise AssertionError("cached quote logical book keys are not unique")

    snapshot_columns = [
        "event_key",
        "dg_id_key",
        "event_start_date",
        "max_source_round_date",
    ]
    snapshots = pd.read_parquet(
        snapshots_path, columns=snapshot_columns
    )
    snapshots["event_start_date"] = pd.to_datetime(
        snapshots["event_start_date"]
    ).dt.normalize()
    snapshots["max_source_round_date"] = pd.to_datetime(
        snapshots["max_source_round_date"], errors="coerce"
    ).dt.normalize()
    snapshots["dg_id_key"] = snapshots["dg_id_key"].map(market_eval._stable_id)
    if snapshots.duplicated(["event_key", "dg_id_key"]).any():
        raise AssertionError("cached snapshot event/player keys are not unique")
    snapshot_bad = (
        snapshots["max_source_round_date"].notna()
        & (snapshots["max_source_round_date"] >= snapshots["event_start_date"])
    )
    if snapshot_bad.any():
        raise AssertionError("cached strict-prior snapshots fail chronology")
    membership = predictions[["event_key", "dg_id_key"]].drop_duplicates().merge(
        snapshots[["event_key", "dg_id_key"]].assign(_snapshot=True),
        on=["event_key", "dg_id_key"],
        how="left",
        validate="one_to_one",
    )
    if membership["_snapshot"].isna().any():
        raise AssertionError(
            f"{int(membership['_snapshot'].isna().sum())} prediction keys are absent "
            "from the frozen snapshot panel"
        )
    return predictions, catalog, source_manifest, quotes, snapshots, cache_manifest


def _canonical_pair_count(frame: pd.DataFrame) -> tuple[int, int]:
    if frame.empty:
        return 0, 0
    logical = market_eval._canonical_logical_pairs(frame)
    logical = logical.loc[logical["canonical_outcome"].isin([0.0, 0.5, 1.0])]
    return int(len(logical)), int(logical["mapped_event_key"].nunique())


def _fit_mapper_or_none(frame: pd.DataFrame, min_pairs: int) -> tuple[float, float, int, int]:
    pairs, events = _canonical_pair_count(frame)
    if pairs < min_pairs:
        return np.nan, np.nan, pairs, events
    sigma, tie = market_eval.fit_probability_params(frame)
    return sigma, tie, pairs, events


def fit_temporal_probability_mappers(
    paired: pd.DataFrame,
    baseline_candidate: str,
    *,
    evaluation_years: Sequence[int] = EVALUATION_YEARS,
    min_pairs: int = 100,
) -> pd.DataFrame:
    """Fit previous-calendar-year probability mappers for discovery scoring."""

    paired = paired.copy()
    paired["evaluation_year"] = pd.to_datetime(paired["event_start_date"]).dt.year
    candidates = sorted(paired["candidate"].unique())
    if baseline_candidate not in candidates:
        raise ValueError(f"strict-PT baseline {baseline_candidate!r} has no quoted pairs")
    markets = sorted(set(paired["market"]) & set(DEFAULT_PRIORS))
    rows: list[dict[str, Any]] = []
    baseline_mappers: dict[tuple[int, str], tuple[float, float, str]] = {}

    for year in evaluation_years:
        calibration_year = int(year) - 1
        for market in markets:
            train = paired.loc[
                paired["candidate"].eq(baseline_candidate)
                & paired["market"].eq(market)
                & paired["evaluation_year"].eq(calibration_year)
            ]
            sigma, tie, pairs, events = _fit_mapper_or_none(train, min_pairs)
            if np.isfinite(sigma) and np.isfinite(tie):
                source = "strict_pt_baseline_previous_year"
            else:
                sigma, tie = DEFAULT_PRIORS[market]
                source = "fixed_prespecified_prior"
            baseline_mappers[(int(year), market)] = (float(sigma), float(tie), source)
            rows.append(
                {
                    "candidate": baseline_candidate,
                    "market": market,
                    "evaluation_year": int(year),
                    "calibration_year": calibration_year,
                    "calibration_start": f"{calibration_year}-01-01",
                    "calibration_end": f"{calibration_year}-12-31",
                    "calibration_source_candidate": baseline_candidate,
                    "calibration_source": source,
                    "sigma": float(sigma),
                    "tie_threshold": float(tie),
                    "calibration_pairs": pairs,
                    "calibration_events": events,
                    "sigma_near_upper_bound": bool(sigma >= 99.0),
                }
            )

    for candidate in candidates:
        if candidate == baseline_candidate:
            continue
        for year in evaluation_years:
            calibration_year = int(year) - 1
            for market in markets:
                train = paired.loc[
                    paired["candidate"].eq(candidate)
                    & paired["market"].eq(market)
                    & paired["evaluation_year"].eq(calibration_year)
                ]
                sigma, tie, pairs, events = _fit_mapper_or_none(train, min_pairs)
                if np.isfinite(sigma) and np.isfinite(tie):
                    source_candidate = candidate
                    source = "candidate_previous_year"
                else:
                    sigma, tie, baseline_source = baseline_mappers[(int(year), market)]
                    source_candidate = baseline_candidate
                    source = f"strict_pt_baseline_fallback:{baseline_source}"
                rows.append(
                    {
                        "candidate": candidate,
                        "market": market,
                        "evaluation_year": int(year),
                        "calibration_year": calibration_year,
                        "calibration_start": f"{calibration_year}-01-01",
                        "calibration_end": f"{calibration_year}-12-31",
                        "calibration_source_candidate": source_candidate,
                        "calibration_source": source,
                        "sigma": float(sigma),
                        "tie_threshold": float(tie),
                        "calibration_pairs": pairs,
                        "calibration_events": events,
                        "sigma_near_upper_bound": bool(sigma >= 99.0),
                    }
                )
    output = pd.DataFrame(rows)
    if output.duplicated(["candidate", "market", "evaluation_year"]).any():
        raise AssertionError("temporal probability mapper keys are not unique")
    if (output["calibration_year"] >= output["evaluation_year"]).any():
        raise AssertionError("probability mapper chronology is not strict")
    return output.sort_values(["candidate", "market", "evaluation_year"]).reset_index(drop=True)


SCORED_KEEP_COLUMNS = [
    "candidate",
    "quote_id",
    "mapped_event_key",
    "event_start_date",
    "market",
    "book",
    "pair_key",
    "p1_id",
    "p2_id",
    "p1_dg_id",
    "p2_dg_id",
    "p1_player_name",
    "p2_player_name",
    "p1_close",
    "p2_close",
    "p1_open",
    "p2_open",
    "p1_outcome",
    "p2_outcome",
    "p1_outcome_text",
    "p2_outcome_text",
    "tie_rule",
    "skill_difference",
    "pair_reliability_ok",
    "sigma",
    "tie_threshold",
    "p1_probability",
    "push_probability",
    "p1_pricing_probability",
]


def score_temporal_discovery(
    paired: pd.DataFrame, mapper: pd.DataFrame
) -> pd.DataFrame:
    work = paired.copy()
    work["evaluation_year"] = pd.to_datetime(work["event_start_date"]).dt.year
    pieces: list[pd.DataFrame] = []
    for row in mapper.to_dict(orient="records"):
        group = work.loc[
            work["candidate"].eq(row["candidate"])
            & work["market"].eq(row["market"])
            & work["evaluation_year"].eq(int(row["evaluation_year"]))
        ]
        if group.empty:
            continue
        scored = market_eval.score_probabilities(
            group, float(row["sigma"]), float(row["tie_threshold"])
        )
        scored["probability_calibration_year"] = int(row["calibration_year"])
        scored["probability_calibration_source"] = row["calibration_source"]
        keep = [column for column in SCORED_KEEP_COLUMNS if column in scored]
        keep += ["probability_calibration_year", "probability_calibration_source"]
        pieces.append(scored[keep].copy())
    if not pieces:
        raise ValueError("temporal probability calibration produced no discovery scores")
    output = pd.concat(pieces, ignore_index=True)
    if output.duplicated(["candidate", "quote_id"]).any():
        raise AssertionError("cross-fitted candidate/quote keys are not unique")
    if (output["probability_calibration_year"] >= output["event_start_date"].dt.year).any():
        raise AssertionError("cross-fitted probabilities use a non-prior calibration year")
    return output


def canonical_discovery_probabilities(scored: pd.DataFrame) -> pd.DataFrame:
    logical = market_eval._canonical_logical_pairs(scored)
    keys = ["candidate", "mapped_event_key", "market", "pair_key"]
    if logical.duplicated(keys).any():
        raise AssertionError("canonical cross-fitted probability keys are not unique")
    return logical


def _weighted_calibration(
    event_ids: pd.Series, outcome: np.ndarray, probability: np.ndarray
) -> tuple[float, float, float, bool]:
    probability = np.clip(np.asarray(probability, dtype=float), market_eval.EPS, 1.0 - market_eval.EPS)
    outcome = np.asarray(outcome, dtype=float)
    event_frame = pd.DataFrame({"event": event_ids.to_numpy()})
    weights = 1.0 / event_frame.groupby("event", observed=True)["event"].transform("count").to_numpy(float)
    x = logit(probability)

    def objective(parameters: np.ndarray) -> float:
        calibrated = np.clip(
            expit(float(parameters[0]) + float(parameters[1]) * x),
            market_eval.EPS,
            1.0 - market_eval.EPS,
        )
        loss = -(outcome * np.log(calibrated) + (1.0 - outcome) * np.log(1.0 - calibrated))
        return float(np.sum(weights * loss) / np.sum(weights))

    fit = minimize(objective, np.array([0.0, 1.0]), method="BFGS")
    intercept, slope = fit.x if fit.success else (np.nan, np.nan)
    try:
        bins = pd.qcut(pd.Series(probability), q=10, duplicates="drop")
        calibration = pd.DataFrame(
            {"bin": bins, "predicted": probability, "actual": outcome}
        ).groupby("bin", observed=True).agg(
            predicted=("predicted", "mean"),
            actual=("actual", "mean"),
            n=("actual", "size"),
        )
        ece = float(
            np.average(
                np.abs(calibration["predicted"] - calibration["actual"]),
                weights=calibration["n"],
            )
        )
    except ValueError:
        ece = np.nan
    return float(intercept), float(slope), ece, bool(fit.success)


def paired_common_proper_scores(
    logical: pd.DataFrame, baseline_candidate: str
) -> pd.DataFrame:
    resolved = logical.loc[logical["canonical_outcome"].isin([0.0, 1.0])].copy()
    baseline = resolved.loc[
        resolved["candidate"].eq(baseline_candidate),
        [
            "mapped_event_key",
            "market",
            "pair_key",
            "canonical_outcome",
            "canonical_probability",
            "canonical_push_probability",
        ],
    ].rename(
        columns={
            "canonical_outcome": "baseline_outcome",
            "canonical_probability": "baseline_probability",
            "canonical_push_probability": "baseline_push_probability",
        }
    )
    if baseline.empty:
        raise ValueError("strict-PT baseline has no resolved discovery pairs")
    baseline_counts = baseline.groupby("market", observed=True).size().to_dict()
    rows: list[dict[str, Any]] = []
    for candidate in sorted(resolved["candidate"].unique()):
        challenger = resolved.loc[
            resolved["candidate"].eq(candidate),
            [
                "mapped_event_key",
                "market",
                "pair_key",
                "canonical_outcome",
                "canonical_probability",
                "canonical_push_probability",
            ],
        ].rename(
            columns={
                "canonical_outcome": "candidate_outcome",
                "canonical_probability": "candidate_probability",
                "canonical_push_probability": "candidate_push_probability",
            }
        )
        common = baseline.merge(
            challenger,
            on=["mapped_event_key", "market", "pair_key"],
            how="inner",
            validate="one_to_one",
        )
        if not np.allclose(common["baseline_outcome"], common["candidate_outcome"]):
            raise AssertionError(f"outcomes disagree on common pairs for {candidate}")
        for market, group in common.groupby("market", observed=True):
            y = group["candidate_outcome"].to_numpy(float)
            candidate_p = (
                group["candidate_probability"]
                / (1.0 - group["candidate_push_probability"])
            ).clip(market_eval.EPS, 1.0 - market_eval.EPS).to_numpy(float)
            baseline_p = (
                group["baseline_probability"]
                / (1.0 - group["baseline_push_probability"])
            ).clip(market_eval.EPS, 1.0 - market_eval.EPS).to_numpy(float)
            metrics = group[["mapped_event_key"]].copy()
            metrics["candidate_log_loss"] = -(
                y * np.log(candidate_p) + (1.0 - y) * np.log(1.0 - candidate_p)
            )
            metrics["baseline_log_loss"] = -(
                y * np.log(baseline_p) + (1.0 - y) * np.log(1.0 - baseline_p)
            )
            metrics["candidate_brier"] = np.square(candidate_p - y)
            metrics["baseline_brier"] = np.square(baseline_p - y)
            event_scores = metrics.groupby("mapped_event_key", observed=True).mean()
            ci, cs, ce, csuccess = _weighted_calibration(
                group["mapped_event_key"], y, candidate_p
            )
            bi, bs, be, bsuccess = _weighted_calibration(
                group["mapped_event_key"], y, baseline_p
            )
            baseline_count = int(baseline_counts.get(market, 0))
            rows.append(
                {
                    "candidate": candidate,
                    "baseline": baseline_candidate,
                    "market": market,
                    "events": int(event_scores.index.nunique()),
                    "common_resolved_pairs": int(len(group)),
                    "baseline_resolved_pairs": baseline_count,
                    "coverage_ratio_vs_baseline": (
                        float(len(group) / baseline_count) if baseline_count else np.nan
                    ),
                    "candidate_event_equal_log_loss": float(
                        event_scores["candidate_log_loss"].mean()
                    ),
                    "baseline_event_equal_log_loss": float(
                        event_scores["baseline_log_loss"].mean()
                    ),
                    "log_loss_delta": float(
                        event_scores["candidate_log_loss"].mean()
                        - event_scores["baseline_log_loss"].mean()
                    ),
                    "candidate_event_equal_brier": float(
                        event_scores["candidate_brier"].mean()
                    ),
                    "baseline_event_equal_brier": float(
                        event_scores["baseline_brier"].mean()
                    ),
                    "brier_delta": float(
                        event_scores["candidate_brier"].mean()
                        - event_scores["baseline_brier"].mean()
                    ),
                    "candidate_calibration_intercept": ci,
                    "baseline_calibration_intercept": bi,
                    "candidate_calibration_slope": cs,
                    "baseline_calibration_slope": bs,
                    "candidate_ece_decile": ce,
                    "baseline_ece_decile": be,
                    "candidate_calibration_fit_success": csuccess,
                    "baseline_calibration_fit_success": bsuccess,
                }
            )
    return pd.DataFrame(rows)


def _settled_summary(bets: pd.DataFrame, candidate: str) -> dict[str, float | int]:
    selected = bets.loc[bets["candidate"].eq(candidate)].copy()
    if selected.empty:
        return {
            "bets": 0,
            "stake": 0.0,
            "pnl": 0.0,
            "roi": 0.0,
            "max_drawdown": 0.0,
            "worst_event_loss": 0.0,
        }
    event_pnl = selected.groupby("event_key", observed=True)["pnl"].sum()
    stake = float(selected["stake"].sum())
    pnl = float(selected["pnl"].sum())
    return {
        "bets": int(len(selected)),
        "stake": stake,
        "pnl": pnl,
        "roi": pnl / stake if stake else 0.0,
        "max_drawdown": float(market_eval._max_drawdown(selected)),
        "worst_event_loss": float(max(-float(event_pnl.min()), 0.0)),
    }


def _edge_monotonicity(bets: pd.DataFrame, candidate: str) -> dict[str, float | int]:
    selected = bets.loc[bets["candidate"].eq(candidate)].copy()
    if selected.empty:
        return {"edge_violations": 0, "edge_violation_magnitude": 0.0}
    selected["edge_bucket"] = pd.cut(
        selected["ev_per_unit"],
        bins=EDGE_BINS,
        labels=EDGE_LABELS,
        right=False,
    )
    bucket = selected.loc[selected["edge_bucket"].isin(POSITIVE_EDGE_LABELS)].groupby(
        "edge_bucket", observed=True
    ).agg(pnl=("pnl", "sum"), stake=("stake", "sum"))
    bucket["roi"] = bucket["pnl"] / bucket["stake"]
    ordered = [label for label in POSITIVE_EDGE_LABELS if label in bucket.index]
    violations = 0
    magnitude = 0.0
    for left, right in zip(ordered, ordered[1:]):
        deterioration = float(bucket.loc[left, "roi"] - bucket.loc[right, "roi"])
        if deterioration > 0.0:
            violations += 1
            magnitude += deterioration
    return {
        "edge_violations": int(violations),
        "edge_violation_magnitude": float(magnitude),
    }


def select_reliability_gated_bets(
    scored: pd.DataFrame,
    *,
    min_ev: float = market_eval.EV_GATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the conservative reliability rule after nominal EV is known.

    Only nominal edges of at least 10% are suppressed, and only when either
    golfer fails the upstream point-in-time reliability audit.  Ordinary
    3-10% bets remain eligible even when reliability is false.
    """

    nominal = market_eval.select_and_settle_bets(
        scored, "best_of_three", "flat_1u", min_ev=min_ev
    )
    if nominal.empty:
        return nominal.copy(), nominal.copy()
    if "pair_reliability_ok" not in nominal:
        raise ValueError("reliability sensitivity requires pair_reliability_ok")
    suppress = (
        (nominal["ev_per_unit"] >= 0.10)
        & (~nominal["pair_reliability_ok"].fillna(False).astype(bool))
    )
    suppressed = nominal.loc[suppress].copy()
    suppressed["suppression_reason"] = "nominal_ev_ge_10pct_and_pair_reliability_false"
    kept = nominal.loc[~suppress].copy()
    kept["candidate"] = kept["candidate"] + "__reliability_gate"
    kept["reliability_gate_applied"] = True
    if "pair_reliability_ok" not in kept:
        raise AssertionError("settled reliability bets lost pair_reliability_ok")
    return kept, suppressed


def reliability_edge_bucket_performance(
    scored: pd.DataFrame, sample: str
) -> pd.DataFrame:
    all_directional, _ = select_reliability_gated_bets(scored, min_ev=-np.inf)
    if all_directional.empty:
        return pd.DataFrame()
    all_directional["edge_bucket"] = pd.cut(
        all_directional["ev_per_unit"],
        bins=EDGE_BINS,
        labels=EDGE_LABELS,
        right=False,
    )
    rows: list[dict[str, Any]] = []
    for keys, group in all_directional.groupby(
        ["candidate", "round_market", "edge_bucket"], observed=True
    ):
        candidate, market, bucket = keys
        rows.append(
            {
                "sample": sample,
                "line_policy": "best_of_three",
                "candidate": candidate,
                "market": market,
                "edge_bucket": str(bucket),
                "bets": int(len(group)),
                "events": int(group["event_key"].nunique()),
                "pnl": float(group["pnl"].sum()),
                "roi": float(group["pnl"].sum() / group["stake"].sum()),
                "mean_model_ev": float(group["ev_per_unit"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _open_selection_rows(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    line_rows: list[dict[str, Any]] = []
    probability_rows: list[dict[str, Any]] = []
    for row in scored.to_dict(orient="records"):
        for side, other, probability in (
            ("p1", "p2", row["p1_probability"]),
            (
                "p2",
                "p1",
                1.0 - row["p1_probability"] - row.get("push_probability", 0.0),
            ),
        ):
            key = f"{row['candidate']}|{row['quote_id']}|{side}|open"
            line_rows.append(
                {
                    "selection_key": key,
                    "candidate": row["candidate"],
                    "event_id": row["mapped_event_key"],
                    "event_key": row["mapped_event_key"],
                    "event_start_date": row["event_start_date"],
                    "market": "tournament_matchup",
                    "round_market": row["market"],
                    "selection_id": row[f"{side}_id"],
                    "opponent_id": row[f"{other}_id"],
                    "selection_name": row.get(f"{side}_player_name"),
                    "opponent_name": row.get(f"{other}_player_name"),
                    "book": row["book"],
                    "open_odds": row[f"{side}_open"],
                    "opponent_open_odds": row[f"{other}_open"],
                    "market_close_odds": row[f"{side}_close"],
                    "opponent_close_odds": row[f"{other}_close"],
                    "close_odds": row[f"{side}_open"],
                    "outcome_numeric": row.get(f"{side}_outcome"),
                    "outcome": row.get(f"{side}_outcome_text"),
                    "tie_rule": row.get("tie_rule"),
                    "pair_key": row["pair_key"],
                }
            )
            probability_rows.append(
                {
                    "selection_key": key,
                    "probability": float(probability),
                    "push_probability": float(row.get("push_probability", 0.0)),
                }
            )
    return pd.DataFrame(line_rows), pd.DataFrame(probability_rows)


def select_open_bets(scored: pd.DataFrame, min_ev: float = market_eval.EV_GATE) -> pd.DataFrame:
    lines, probabilities = _open_selection_rows(scored)
    if lines.empty:
        return pd.DataFrame()
    for column in (
        "open_odds",
        "opponent_open_odds",
        "market_close_odds",
        "opponent_close_odds",
    ):
        lines[column] = pd.to_numeric(lines[column], errors="coerce")
    lines = lines.loc[
        valid_american_odds_mask(lines["open_odds"])
        & valid_american_odds_mask(lines["opponent_open_odds"])
        & valid_american_odds_mask(lines["market_close_odds"])
        & valid_american_odds_mask(lines["opponent_close_odds"])
    ].copy()
    keys = ["candidate", "event_key", "round_market", "selection_id", "opponent_id"]
    lines = lines.sort_values(
        keys + ["open_odds", "book"],
        ascending=[True] * len(keys) + [False, True],
        kind="mergesort",
    ).drop_duplicates(keys)
    joined = lines.merge(probabilities, on="selection_key", how="inner", validate="one_to_one")
    decimal = np.asarray(market_eval.american_decimal_odds(joined["open_odds"]), dtype=float)
    effective_push = np.where(
        joined["tie_rule"].map(market_eval._ties_lose_text).to_numpy(),
        0.0,
        joined["push_probability"].to_numpy(float),
    )
    loss_probability = 1.0 - joined["probability"].to_numpy(float) - effective_push
    joined["ev_per_unit"] = joined["probability"] * (decimal - 1.0) - loss_probability
    joined = joined.loc[joined["ev_per_unit"] >= min_ev].copy()
    direction_keys = ["candidate", "event_key", "round_market", "pair_key"]
    joined = joined.sort_values(
        direction_keys + ["ev_per_unit", "open_odds", "book"],
        ascending=[True] * len(direction_keys) + [False, False, True],
        kind="mergesort",
    ).drop_duplicates(direction_keys)
    if joined.empty:
        return joined
    open_implied = np.asarray(
        market_eval.american_implied_probability(joined["open_odds"]), dtype=float
    )
    opponent_open_implied = np.asarray(
        market_eval.american_implied_probability(joined["opponent_open_odds"]), dtype=float
    )
    close_implied = np.asarray(
        market_eval.american_implied_probability(joined["market_close_odds"]), dtype=float
    )
    opponent_close_implied = np.asarray(
        market_eval.american_implied_probability(joined["opponent_close_odds"]), dtype=float
    )
    joined["open_novig_probability"] = open_implied / (open_implied + opponent_open_implied)
    joined["close_novig_probability"] = close_implied / (close_implied + opponent_close_implied)
    joined["clv_probability"] = (
        joined["close_novig_probability"] - joined["open_novig_probability"]
    )
    joined["stake"] = 1.0
    return market_eval.settle_bets(joined)


def discovery_guardrails(
    scored: pd.DataFrame,
    candidate: str,
    baseline_candidate: str,
) -> dict[str, Any]:
    subset = scored.loc[scored["candidate"].isin([baseline_candidate, candidate])].copy()
    coverage = subset.groupby("quote_id", observed=True)["candidate"].nunique()
    common_ids = coverage.loc[coverage == 2].index
    subset = subset.loc[subset["quote_id"].isin(common_ids)].copy()
    if subset.empty:
        raise ValueError(f"no common discovery quotes for {candidate}")
    close_bets = market_eval.select_and_settle_bets(
        subset, "best_of_three", "flat_1u", min_ev=market_eval.EV_GATE
    )
    all_directions = market_eval.select_and_settle_bets(
        subset, "best_of_three", "flat_1u", min_ev=-np.inf
    )
    open_bets = select_open_bets(subset, min_ev=market_eval.EV_GATE)
    candidate_close = _settled_summary(close_bets, candidate)
    baseline_close = _settled_summary(close_bets, baseline_candidate)
    candidate_edge = _edge_monotonicity(all_directions, candidate)
    baseline_edge = _edge_monotonicity(all_directions, baseline_candidate)

    def clv_summary(name: str) -> tuple[int, float]:
        group = open_bets.loc[open_bets["candidate"].eq(name)]
        return int(len(group)), float(group["clv_probability"].mean()) if len(group) else np.nan

    candidate_open_bets, candidate_clv = clv_summary(candidate)
    baseline_open_bets, baseline_clv = clv_summary(baseline_candidate)
    return {
        "candidate": candidate,
        "baseline": baseline_candidate,
        "common_book_quotes": int(len(common_ids)),
        "candidate_close_bets": candidate_close["bets"],
        "baseline_close_bets": baseline_close["bets"],
        "candidate_close_stake": candidate_close["stake"],
        "baseline_close_stake": baseline_close["stake"],
        "candidate_close_pnl": candidate_close["pnl"],
        "baseline_close_pnl": baseline_close["pnl"],
        "close_pnl_delta": float(candidate_close["pnl"] - baseline_close["pnl"]),
        "candidate_close_roi": candidate_close["roi"],
        "baseline_close_roi": baseline_close["roi"],
        "close_roi_delta": float(candidate_close["roi"] - baseline_close["roi"]),
        "candidate_max_drawdown": candidate_close["max_drawdown"],
        "baseline_max_drawdown": baseline_close["max_drawdown"],
        "max_drawdown_delta": float(
            candidate_close["max_drawdown"] - baseline_close["max_drawdown"]
        ),
        "candidate_worst_event_loss": candidate_close["worst_event_loss"],
        "baseline_worst_event_loss": baseline_close["worst_event_loss"],
        "worst_event_loss_delta": float(
            candidate_close["worst_event_loss"] - baseline_close["worst_event_loss"]
        ),
        "candidate_edge_violations": candidate_edge["edge_violations"],
        "baseline_edge_violations": baseline_edge["edge_violations"],
        "candidate_edge_violation_magnitude": candidate_edge["edge_violation_magnitude"],
        "baseline_edge_violation_magnitude": baseline_edge["edge_violation_magnitude"],
        "edge_violation_magnitude_delta": float(
            candidate_edge["edge_violation_magnitude"]
            - baseline_edge["edge_violation_magnitude"]
        ),
        "candidate_open_bets": candidate_open_bets,
        "baseline_open_bets": baseline_open_bets,
        "candidate_mean_open_to_close_clv": candidate_clv,
        "baseline_mean_open_to_close_clv": baseline_clv,
        "open_to_close_clv_delta": float(candidate_clv - baseline_clv)
        if np.isfinite(candidate_clv) and np.isfinite(baseline_clv)
        else np.nan,
    }


def guardrail_summary_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Return a merge-safe table when no model clears the proper-score screen.

    An empty result is a valid, fail-closed discovery outcome: it means no
    challenger earned evaluation at the later money/CLV gates. Preserve the
    join key so the existing adverse defaults record that result instead of
    raising during scorecard assembly.
    """

    if not rows:
        return pd.DataFrame({"candidate": pd.Series(dtype="string")})
    output = pd.DataFrame(rows)
    if "candidate" not in output.columns:
        raise ValueError("non-empty guardrail rows are missing candidate")
    return output


def aggregate_candidate_scores(proper: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate, group in proper.groupby("candidate", observed=True):
        rows.append(
            {
                "candidate": candidate,
                "markets": int(group["market"].nunique()),
                "minimum_coverage_ratio": float(group["coverage_ratio_vs_baseline"].min()),
                "pooled_candidate_log_loss": float(
                    group["candidate_event_equal_log_loss"].mean()
                ),
                "pooled_baseline_log_loss": float(
                    group["baseline_event_equal_log_loss"].mean()
                ),
                "pooled_log_loss_delta": float(group["log_loss_delta"].mean()),
                "worst_market_log_loss_delta": float(group["log_loss_delta"].max()),
                "pooled_brier_delta": float(group["brier_delta"].mean()),
                "candidate_abs_calibration_intercept": float(
                    group["candidate_calibration_intercept"].abs().mean()
                ),
                "baseline_abs_calibration_intercept": float(
                    group["baseline_calibration_intercept"].abs().mean()
                ),
                "calibration_intercept_distance_delta": float(
                    group["candidate_calibration_intercept"].abs().mean()
                    - group["baseline_calibration_intercept"].abs().mean()
                ),
                "candidate_abs_calibration_slope_error": float(
                    (group["candidate_calibration_slope"] - 1.0).abs().mean()
                ),
                "baseline_abs_calibration_slope_error": float(
                    (group["baseline_calibration_slope"] - 1.0).abs().mean()
                ),
                "calibration_slope_error_delta": float(
                    (group["candidate_calibration_slope"] - 1.0).abs().mean()
                    - (group["baseline_calibration_slope"] - 1.0).abs().mean()
                ),
                "candidate_ece": float(group["candidate_ece_decile"].mean()),
                "baseline_ece": float(group["baseline_ece_decile"].mean()),
                "ece_delta": float(
                    group["candidate_ece_decile"].mean()
                    - group["baseline_ece_decile"].mean()
                ),
                "calibration_fit_success": bool(
                    group["candidate_calibration_fit_success"].all()
                    and group["baseline_calibration_fit_success"].all()
                ),
            }
        )
    return pd.DataFrame(rows)


def apply_selection_gates(
    scorecard: pd.DataFrame,
    *,
    coverage_tolerance: float = 0.0,
    logloss_tolerance: float = 0.0,
    calibration_tolerance: float = 0.0,
    money_tolerance: float = 0.0,
) -> pd.DataFrame:
    """Apply frozen sequential no-worse gates; no fallback winner is invented."""

    output = scorecard.copy()
    output["coverage_gate_pass"] = (
        output["minimum_coverage_ratio"] >= 1.0 - coverage_tolerance
    )
    output["proper_gate_pass"] = (
        (output["pooled_log_loss_delta"] < 0.0)
        & (output["worst_market_log_loss_delta"] <= logloss_tolerance)
    )
    output["calibration_gate_pass"] = (
        output["calibration_fit_success"].map(_as_bool)
        & (output["calibration_intercept_distance_delta"] <= calibration_tolerance)
        & (output["calibration_slope_error_delta"] <= calibration_tolerance)
        & (output["ece_delta"] <= calibration_tolerance)
        & (output["edge_violation_magnitude_delta"] <= calibration_tolerance)
    )
    output["money_gate_pass"] = (
        (output["close_pnl_delta"] >= -money_tolerance)
        & (output["close_roi_delta"] >= -money_tolerance)
        & (output["candidate_mean_open_to_close_clv"] >= -money_tolerance)
        & (output["open_to_close_clv_delta"] >= -money_tolerance)
    )
    output["concentration_gate_pass"] = (
        (output["max_drawdown_delta"] <= money_tolerance)
        & (output["worst_event_loss_delta"] <= money_tolerance)
    )
    ordered_gates = [
        "coverage_gate_pass",
        "proper_gate_pass",
        "calibration_gate_pass",
        "money_gate_pass",
        "concentration_gate_pass",
    ]
    output["gate_stage_reached"] = 0
    active = pd.Series(True, index=output.index)
    for stage, column in enumerate(ordered_gates, start=1):
        active &= output[column]
        output.loc[active, "gate_stage_reached"] = stage
    output["all_gates_pass"] = output[ordered_gates].all(axis=1)
    return output


def choose_finalists(
    scorecard: pd.DataFrame,
    catalog: pd.DataFrame,
    finalist_count: int,
) -> list[str]:
    eligible_names = set(
        catalog.loc[
            catalog["selection_eligible"]
            & ~catalog["sensitivity_only"]
            & ~catalog["control"],
            "name",
        ].astype(str)
    )
    passed = scorecard.loc[
        scorecard["candidate"].isin(eligible_names) & scorecard["all_gates_pass"]
    ].copy()
    if passed.empty:
        return []
    complexity = {
        "ridge": 0,
        "robust_irls": 1,
        "elastic_net": 1,
        "elastic_cd": 1,
        "dynamic_eb": 2,
        "dynamic_kalman": 3,
    }
    metadata = catalog[["name", "family"]].rename(columns={"name": "candidate"})
    passed = passed.merge(metadata, on="candidate", how="left", validate="one_to_one")
    passed["complexity_rank"] = passed["family"].map(complexity).fillna(99)
    passed = passed.sort_values(
        [
            "pooled_log_loss_delta",
            "pooled_brier_delta",
            "calibration_slope_error_delta",
            "close_pnl_delta",
            "max_drawdown_delta",
            "complexity_rank",
            "candidate",
        ],
        ascending=[True, True, True, False, True, True, True],
        kind="mergesort",
    )
    return passed["candidate"].head(max(int(finalist_count), 0)).tolist()


def choose_diagnostic_challenger(
    scorecard: pd.DataFrame, catalog: pd.DataFrame
) -> tuple[str | None, list[str]]:
    eligible_names = set(
        catalog.loc[
            catalog["selection_eligible"]
            & ~catalog["sensitivity_only"]
            & ~catalog["control"],
            "name",
        ].astype(str)
    )
    eligible = scorecard.loc[scorecard["candidate"].isin(eligible_names)].copy()
    if eligible.empty:
        return None, ["no_selection_eligible_candidate_was_scored"]
    ranked = eligible.sort_values(
        [
            "gate_stage_reached",
            "pooled_log_loss_delta",
            "pooled_brier_delta",
            "calibration_slope_error_delta",
            "close_pnl_delta",
            "candidate",
        ],
        ascending=[False, True, True, True, False, True],
        kind="mergesort",
    )
    winner = ranked.iloc[0]
    gate_columns = [
        "coverage_gate_pass",
        "proper_gate_pass",
        "calibration_gate_pass",
        "money_gate_pass",
        "concentration_gate_pass",
    ]
    failed = [column.removesuffix("_pass") for column in gate_columns if not _as_bool(winner[column])]
    return str(winner["candidate"]), failed


def build_locked_diagnostic_ablation_set(
    selected_candidate: str | None,
    catalog: pd.DataFrame,
) -> dict[str, Any]:
    """Lock same-spec weather and IMG siblings before sealed scoring.

    These names are descriptive ablations only.  They cannot replace the
    discovery-selected neutral-base model, even if one later looks better in
    the sealed period.
    """

    weather_roles = (
        ("w0", "w0", "observed"),
        ("w1_feature_only", "w1", "feature_only"),
        ("w1_neutral_target", "w1", "neutral_target"),
        (
            "robust_w1_feature_only",
            "w1_robust_covariate",
            "feature_only",
        ),
        (
            "robust_w1_neutral_target",
            "w1_robust_covariate",
            "neutral_target",
        ),
    )
    required_match_fields = (
        "family",
        "source",
        "refit_schedule",
        "alpha",
        "short_game_penalty",
    )
    additional_hyperparameter_fields = tuple(
        column
        for column in (
            "dynamic_half_life",
            "huber_epsilon",
            "l1_ratio",
            "prior_rounds",
            "process_variance_per_day",
            "prediction_envelope_applied",
            "prediction_envelope_method",
            "prediction_envelope_scale_factor",
            "prediction_envelope_multiplier",
            "raw_regression_component_selection_eligible",
        )
        if column in catalog.columns
    )
    payload: dict[str, Any] = {
        "selected_candidate": selected_candidate,
        "locked_before_sealed_scoring": True,
        "selection_uses_sealed": False,
        "purpose": "descriptive_weather_and_clean_img_attribution_only",
        "may_reselect_or_promote_from_sealed": False,
        "matching_rule": {
            "weather": (
                "exact family/source/refit_schedule/alpha/short_game_penalty "
                "and all catalogued family hyperparameters; vary only "
                "weather_arm/target_arm"
            ),
            "clean_img": (
                "exact model and weather specification; map total<->total_img "
                "or total_cat<->total_cat_img when that catalog sibling exists"
            ),
            "required_match_fields": list(required_match_fields),
            "additional_hyperparameter_fields": list(
                additional_hyperparameter_fields
            ),
        },
        "weather_siblings": [],
        "clean_img_no_img_sibling": {
            "counterpart_source": None,
            "names": [],
            "available": False,
        },
        "included_candidates": [],
        "unavailable_requested_siblings": [],
    }
    if selected_candidate is None:
        payload["unavailable_requested_siblings"] = [
            role for role, _, _ in weather_roles
        ] + ["clean_img_no_img"]
        return payload

    selected = catalog.loc[catalog["name"].astype(str).eq(selected_candidate)]
    if len(selected) != 1:
        payload["unavailable_requested_siblings"] = [
            "selected_candidate_missing_or_nonunique_in_catalog"
        ]
        return payload
    selected_row = selected.iloc[0]
    missing_match_fields = [
        column for column in required_match_fields if column not in catalog.columns
    ]
    if missing_match_fields:
        payload["unavailable_requested_siblings"] = [
            f"catalog_missing_match_fields:{','.join(missing_match_fields)}"
        ]
        return payload

    match_fields = required_match_fields + additional_hyperparameter_fields
    same_spec = pd.Series(True, index=catalog.index)
    for column in match_fields:
        value = selected_row[column]
        if pd.isna(value):
            same_spec &= catalog[column].isna()
        elif isinstance(value, (float, np.floating, int, np.integer)):
            numeric = pd.to_numeric(catalog[column], errors="coerce")
            same_spec &= np.isclose(
                numeric.astype(float), float(value), rtol=0.0, atol=1e-12
            )
        else:
            same_spec &= catalog[column].astype(str).eq(str(value))

    included = {selected_candidate}
    for role, weather_arm, target_arm in weather_roles:
        role_mask = same_spec.copy()
        if "weather_arm" not in catalog or "target_arm" not in catalog:
            names: list[str] = []
        else:
            role_mask &= catalog["weather_arm"].astype(str).eq(weather_arm)
            role_mask &= catalog["target_arm"].astype(str).eq(target_arm)
            names = sorted(catalog.loc[role_mask, "name"].astype(str).unique())
        payload["weather_siblings"].append(
            {
                "role": role,
                "weather_arm": weather_arm,
                "target_arm": target_arm,
                "names": names,
                "available": bool(names),
            }
        )
        included.update(names)
        if not names:
            payload["unavailable_requested_siblings"].append(role)

    source = str(selected_row["source"])
    counterpart_source = {
        "total": "total_img",
        "total_img": "total",
        "total_cat": "total_cat_img",
        "total_cat_img": "total_cat",
    }.get(source)
    img_payload = payload["clean_img_no_img_sibling"]
    img_payload["selected_source"] = source
    img_payload["counterpart_source"] = counterpart_source
    if counterpart_source is not None:
        img_mask = pd.Series(True, index=catalog.index)
        for column in match_fields:
            if column == "source":
                img_mask &= catalog[column].astype(str).eq(counterpart_source)
                continue
            value = selected_row[column]
            if pd.isna(value):
                img_mask &= catalog[column].isna()
            elif isinstance(value, (float, np.floating, int, np.integer)):
                numeric = pd.to_numeric(catalog[column], errors="coerce")
                img_mask &= np.isclose(
                    numeric.astype(float), float(value), rtol=0.0, atol=1e-12
                )
            else:
                img_mask &= catalog[column].astype(str).eq(str(value))
        for column in ("weather_arm", "target_arm"):
            if column in catalog:
                img_mask &= catalog[column].astype(str).eq(str(selected_row[column]))
        img_names = sorted(catalog.loc[img_mask, "name"].astype(str).unique())
        img_payload["names"] = img_names
        img_payload["available"] = bool(img_names)
        included.update(img_names)
    if not img_payload["available"]:
        payload["unavailable_requested_siblings"].append("clean_img_no_img")
    payload["included_candidates"] = sorted(included)
    return payload


def fit_final_discovery_mappers(
    paired: pd.DataFrame,
    baseline_candidate: str,
    min_pairs: int,
) -> pd.DataFrame:
    discovery = paired.loc[
        pd.to_datetime(paired["event_start_date"]).between(DISCOVERY_START, DISCOVERY_END)
    ].copy()
    candidates = sorted(discovery["candidate"].unique())
    markets = sorted(set(discovery["market"]) & set(DEFAULT_PRIORS))
    baseline_params: dict[str, tuple[float, float, str]] = {}
    rows: list[dict[str, Any]] = []
    for market in markets:
        group = discovery.loc[
            discovery["candidate"].eq(baseline_candidate)
            & discovery["market"].eq(market)
        ]
        sigma, tie, pairs, events = _fit_mapper_or_none(group, min_pairs)
        if np.isfinite(sigma) and np.isfinite(tie):
            source = "strict_pt_baseline_discovery"
        else:
            sigma, tie = DEFAULT_PRIORS[market]
            source = "fixed_prespecified_prior"
        baseline_params[market] = (float(sigma), float(tie), source)
        rows.append(
            {
                "candidate": baseline_candidate,
                "market": market,
                "fit_start": DISCOVERY_START.date().isoformat(),
                "fit_end": DISCOVERY_END.date().isoformat(),
                "calibration_source_candidate": baseline_candidate,
                "calibration_source": source,
                "sigma": float(sigma),
                "tie_threshold": float(tie),
                "calibration_pairs": pairs,
                "calibration_events": events,
                "sigma_near_upper_bound": bool(sigma >= 99.0),
                "frozen_before_sealed_scoring": True,
            }
        )
    for candidate in candidates:
        if candidate == baseline_candidate:
            continue
        for market in markets:
            group = discovery.loc[
                discovery["candidate"].eq(candidate)
                & discovery["market"].eq(market)
            ]
            sigma, tie, pairs, events = _fit_mapper_or_none(group, min_pairs)
            if np.isfinite(sigma) and np.isfinite(tie):
                source_candidate = candidate
                source = "candidate_discovery"
            else:
                sigma, tie, baseline_source = baseline_params[market]
                source_candidate = baseline_candidate
                source = f"strict_pt_baseline_fallback:{baseline_source}"
            rows.append(
                {
                    "candidate": candidate,
                    "market": market,
                    "fit_start": DISCOVERY_START.date().isoformat(),
                    "fit_end": DISCOVERY_END.date().isoformat(),
                    "calibration_source_candidate": source_candidate,
                    "calibration_source": source,
                    "sigma": float(sigma),
                    "tie_threshold": float(tie),
                    "calibration_pairs": pairs,
                    "calibration_events": events,
                    "sigma_near_upper_bound": bool(sigma >= 99.0),
                    "frozen_before_sealed_scoring": True,
                }
            )
    output = pd.DataFrame(rows)
    if output.duplicated(["candidate", "market"]).any():
        raise AssertionError("final discovery mapper keys are not unique")
    return output.sort_values(["candidate", "market"]).reset_index(drop=True)


def downstream_lambda_sensitivity(
    predictions: pd.DataFrame,
    quotes: pd.DataFrame,
    bridge_path: Path,
    baseline_candidate: str,
    candidate_names: set[str],
    lambdas: Sequence[float],
    min_pairs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Discovery-only exact-bridge downstream preservation sensitivity.

    ``lambda=1`` preserves every reconstructable downstream term and is the
    isolation default.  ``lambda=0`` prices the neutral core alone.  This
    sensitivity is never substituted for the neutral-base candidate score.
    """

    if not bridge_path.exists():
        raise FileNotFoundError(bridge_path)
    bridge = pd.read_parquet(
        bridge_path, columns=["event_start", "dg_id", "downstream_preserved_total"]
    )
    bridge["event_start_date"] = pd.to_datetime(bridge["event_start"]).dt.normalize()
    bridge["dg_id_key"] = bridge["dg_id"].map(market_eval._stable_id)
    bridge["downstream_preserved_total"] = pd.to_numeric(
        bridge["downstream_preserved_total"], errors="coerce"
    )
    bridge = bridge.loc[bridge["downstream_preserved_total"].notna(), [
        "event_start_date", "dg_id_key", "downstream_preserved_total"
    ]].drop_duplicates(["event_start_date", "dg_id_key"])
    exact = predictions.loc[
        predictions["candidate"].isin(candidate_names | {baseline_candidate})
    ].merge(
        bridge,
        on=["event_start_date", "dg_id_key"],
        how="inner",
        validate="many_to_one",
    )
    if exact.empty:
        raise ValueError("downstream lambda sensitivity has no exact bridge rows")

    score_pieces: list[pd.DataFrame] = []
    parameter_pieces: list[pd.DataFrame] = []
    for value in lambdas:
        adjusted = exact.copy()
        adjusted["prediction"] = (
            adjusted["prediction"]
            + float(value) * adjusted["downstream_preserved_total"]
        )
        paired = market_eval.attach_skill_differences(quotes, adjusted)
        mapper = fit_temporal_probability_mappers(
            paired,
            baseline_candidate,
            evaluation_years=EVALUATION_YEARS,
            min_pairs=min_pairs,
        )
        mapper["downstream_lambda"] = float(value)
        parameter_pieces.append(mapper)
        scored = score_temporal_discovery(paired, mapper)
        logical = canonical_discovery_probabilities(scored)
        proper = paired_common_proper_scores(logical, baseline_candidate)
        aggregate = aggregate_candidate_scores(proper)
        aggregate["downstream_lambda"] = float(value)
        score_pieces.append(aggregate)
    scores = pd.concat(score_pieces, ignore_index=True)
    parameters = pd.concat(parameter_pieces, ignore_index=True)
    choices: list[dict[str, Any]] = []
    for candidate, group in scores.loc[
        scores["candidate"].isin(candidate_names)
    ].groupby("candidate", observed=True):
        ranked = group.assign(
            _default_distance=(group["downstream_lambda"] - 1.0).abs()
        ).sort_values(
            ["pooled_candidate_log_loss", "_default_distance", "downstream_lambda"],
            ascending=[True, True, False],
            kind="mergesort",
        )
        selected = ranked.iloc[0]
        choices.append(
            {
                "candidate": candidate,
                "selected_downstream_lambda": float(selected["downstream_lambda"]),
                "selection_metric": "pooled 2021-22 temporally cross-fitted event-equal log loss",
                "selected_pooled_log_loss": float(selected["pooled_candidate_log_loss"]),
                "is_isolation_default": bool(
                    math.isclose(float(selected["downstream_lambda"]), 1.0)
                ),
                "selection_window_end": DISCOVERY_END.date().isoformat(),
                "sealed_data_used_for_selection": False,
            }
        )
    choices_frame = pd.DataFrame(choices)
    audit = {
        "exact_bridge_prediction_rows": int(len(exact)),
        "exact_bridge_players": int(exact["dg_id_key"].nunique()),
        "lambdas": [float(value) for value in lambdas],
        "lambda_one_is_isolation_default": True,
        "neutral_base_redefined": False,
        "selection_period": "2021-01-01/2022-12-31",
        "sealed_data_used_for_selection": False,
        "probability_parameter_rows": int(len(parameters)),
    }
    return scores, choices_frame, {"audit": audit, "parameters": parameters}


def best_available_same_book_opening_pairs(
    scored: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Choose the best canonical-left opener, retaining its same-book other side."""

    market_eval.validate_normalized_matchup_quotes(scored)
    work = scored.copy()
    work["book"] = work["book"].astype(str).str.strip().str.lower()
    allowed_book = work["book"].isin(market_eval.BOOKS)
    allowed = work.loc[allowed_book].copy()
    if allowed.empty:
        raise ValueError("opener sensitivity has no quotes at the frozen three books")
    canonical_left = allowed["pair_key"].str.split("|", regex=False).str[0]
    flip = allowed["p1_id"].astype(str).ne(canonical_left)
    allowed["canonical_skill_difference"] = np.where(
        flip, -allowed["skill_difference"], allowed["skill_difference"]
    )
    outcome = pd.to_numeric(allowed["p1_outcome"], errors="coerce")
    allowed["canonical_outcome"] = np.where(
        flip & outcome.isin([0.0, 1.0]), 1.0 - outcome, outcome
    )
    push = pd.to_numeric(
        allowed["push_probability"], errors="coerce"
    ).fillna(0.0)
    allowed["canonical_probability"] = np.where(
        flip,
        1.0 - allowed["p1_probability"] - push,
        allowed["p1_probability"],
    )
    allowed["canonical_push_probability"] = push
    for suffix in ("open", "close"):
        allowed[f"canonical_left_{suffix}"] = pd.to_numeric(
            np.where(flip, allowed[f"p2_{suffix}"], allowed[f"p1_{suffix}"]),
            errors="coerce",
        )
        allowed[f"canonical_right_{suffix}"] = pd.to_numeric(
            np.where(flip, allowed[f"p1_{suffix}"], allowed[f"p2_{suffix}"]),
            errors="coerce",
        )
    valid_open = valid_american_odds_mask(allowed["canonical_left_open"]) & (
        valid_american_odds_mask(allowed["canonical_right_open"])
    )
    valid_close = valid_american_odds_mask(allowed["canonical_left_close"]) & (
        valid_american_odds_mask(allowed["canonical_right_close"])
    )
    eligible = allowed.loc[valid_open & valid_close].copy()
    if eligible.empty:
        raise ValueError("opener sensitivity has no valid same-book open/close pairs")

    # This call checks cross-book outcome/probability invariants.  Its fixed
    # canonical row is intentionally discarded: opener selection below is by
    # best American opening price, never fixed book priority.
    market_eval._canonical_logical_pairs(eligible)
    keys = ["candidate", "mapped_event_key", "market", "pair_key"]
    book_rank = {book: index for index, book in enumerate(market_eval.BOOKS)}
    eligible["_book_rank"] = eligible["book"].map(book_rank).astype(int)
    best = eligible.sort_values(
        keys + ["canonical_left_open", "_book_rank"],
        ascending=[True] * len(keys) + [False, True],
        kind="mergesort",
    ).drop_duplicates(keys, keep="first")
    if best.duplicated(keys).any():
        raise AssertionError("best-opener logical keys are not unique")
    offered_pairs = allowed[keys].drop_duplicates()
    eligible_pairs = eligible[keys].drop_duplicates()
    audit = {
        "books": list(market_eval.BOOKS),
        "selection_rule": (
            "maximum canonical-left American opening price; frozen book order "
            "breaks exact price ties only"
        ),
        "source_quote_rows": int(len(work)),
        "three_book_quote_rows": int(len(allowed)),
        "disallowed_book_rows": int((~allowed_book).sum()),
        "invalid_same_book_open_rows": int((~valid_open).sum()),
        "invalid_same_book_close_rows": int((~valid_close).sum()),
        "offered_logical_candidate_pairs": int(len(offered_pairs)),
        "eligible_logical_candidate_pairs": int(len(eligible_pairs)),
        "selected_logical_candidate_pairs": int(len(best)),
        "logical_pair_coverage": float(len(eligible_pairs) / len(offered_pairs)),
        "selected_rows_by_book": {
            str(key): int(value)
            for key, value in best.groupby("book", observed=True).size().items()
        },
        "strict_odds_conversion_after_validity_filter": True,
    }
    return best.drop(columns="_book_rank"), audit


def opener_blend_sensitivity(
    scored: pd.DataFrame,
    weights: Sequence[float],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Choose one discovery-only opener blend weight per candidate.

    The blend operates on conditional-on-no-push probability.  One canonical
    same-book row per logical pair receives weight, so a pair quoted at three
    books does not receive three votes.
    """

    logical, opener_audit = best_available_same_book_opening_pairs(scored)
    logical = logical.loc[logical["canonical_outcome"].isin([0.0, 1.0])].copy()
    nonpush = 1.0 - logical["canonical_push_probability"]
    valid_model = (
        np.isfinite(pd.to_numeric(logical["canonical_probability"], errors="coerce"))
        & np.isfinite(pd.to_numeric(nonpush, errors="coerce"))
        & nonpush.gt(market_eval.EPS)
    )
    invalid_model_rows = int((~valid_model).sum())
    logical = logical.loc[valid_model].copy()
    if logical.empty:
        raise ValueError("opener sensitivity has no resolved valid model probabilities")
    left_open = np.asarray(
        american_implied_probability(logical["canonical_left_open"]), dtype=float
    )
    right_open = np.asarray(
        american_implied_probability(logical["canonical_right_open"]), dtype=float
    )
    left_close = np.asarray(
        american_implied_probability(logical["canonical_left_close"]), dtype=float
    )
    right_close = np.asarray(
        american_implied_probability(logical["canonical_right_close"]), dtype=float
    )
    logical["open_novig_probability"] = left_open / (left_open + right_open)
    logical["close_novig_probability"] = left_close / (left_close + right_close)
    logical["model_conditional_probability"] = (
        logical["canonical_probability"]
        / (1.0 - logical["canonical_push_probability"])
    ).clip(market_eval.EPS, 1.0 - market_eval.EPS)
    rows: list[dict[str, Any]] = []
    for (candidate, market), group in logical.groupby(
        ["candidate", "market"], observed=True
    ):
        y = group["canonical_outcome"].to_numpy(float)
        model_logit = logit(group["model_conditional_probability"].to_numpy(float))
        open_logit = logit(
            group["open_novig_probability"].clip(
                market_eval.EPS, 1.0 - market_eval.EPS
            ).to_numpy(float)
        )
        close_p = group["close_novig_probability"].clip(
            market_eval.EPS, 1.0 - market_eval.EPS
        ).to_numpy(float)
        close_loss = -(y * np.log(close_p) + (1.0 - y) * np.log(1.0 - close_p))
        for weight in weights:
            probability = np.clip(
                expit((1.0 - float(weight)) * model_logit + float(weight) * open_logit),
                market_eval.EPS,
                1.0 - market_eval.EPS,
            )
            loss = -(y * np.log(probability) + (1.0 - y) * np.log(1.0 - probability))
            event = pd.DataFrame(
                {
                    "event": group["mapped_event_key"].to_numpy(),
                    "loss": loss,
                    "close_loss": close_loss,
                }
            ).groupby("event", observed=True).mean()
            rows.append(
                {
                    "candidate": candidate,
                    "market": market,
                    "opener_weight": float(weight),
                    "events": int(event.index.nunique()),
                    "resolved_pairs": int(len(group)),
                    "event_equal_blend_log_loss": float(event["loss"].mean()),
                    "event_equal_same_book_close_log_loss": float(
                        event["close_loss"].mean()
                    ),
                    "mean_open_to_close_probability_move": float(
                        (
                            group["close_novig_probability"]
                            - group["open_novig_probability"]
                        ).mean()
                    ),
                }
            )
    scores = pd.DataFrame(rows)
    pooled = scores.groupby(["candidate", "opener_weight"], observed=True)[
        "event_equal_blend_log_loss"
    ].mean().reset_index(name="pooled_event_equal_blend_log_loss")
    choices: list[dict[str, Any]] = []
    for candidate, group in pooled.groupby("candidate", observed=True):
        selected = group.sort_values(
            ["pooled_event_equal_blend_log_loss", "opener_weight"],
            ascending=[True, True],
            kind="mergesort",
        ).iloc[0]
        choices.append(
            {
                "candidate": candidate,
                "selected_opener_weight": float(selected["opener_weight"]),
                "selected_pooled_log_loss": float(
                    selected["pooled_event_equal_blend_log_loss"]
                ),
                "selection_window_end": DISCOVERY_END.date().isoformat(),
                "sealed_data_used_for_selection": False,
                "neutral_base_redefined": False,
                "same_book_open_close_required": True,
                "opening_line_policy": "best_of_three",
                "opening_books": list(market_eval.BOOKS),
            }
        )
    opener_audit["resolved_model_rows"] = int(len(logical))
    opener_audit["invalid_model_probability_rows"] = invalid_model_rows
    opener_audit["weights"] = [float(value) for value in weights]
    return scores, pd.DataFrame(choices), opener_audit


def pairwise_logistic_sensitivity(
    paired: pd.DataFrame,
    baseline_candidate: str,
    candidate_names: set[str],
    *,
    regularization_c: float = 0.1,
    min_pairs: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strict-temporal pairwise market-ranking sensitivity.

    This arm uses only the difference in an already strict-prior candidate
    prediction.  It fits without an intercept so reversing the player order
    reverses the score.  Its units are logit/ranking units, never SG/round, and
    it is categorically ineligible to replace the neutral base.
    """

    logical = market_eval._canonical_logical_pairs(paired)
    logical["year"] = pd.to_datetime(logical["event_start_date"]).dt.year
    resolved = logical.loc[logical["canonical_outcome"].isin([0.0, 1.0])].copy()
    candidates = sorted((candidate_names | {baseline_candidate}) & set(resolved["candidate"]))
    markets = sorted(set(resolved["market"]) & set(DEFAULT_PRIORS))

    def fit_coefficient(group: pd.DataFrame) -> tuple[float, int, int, bool]:
        if len(group) < min_pairs or group["canonical_outcome"].nunique() < 2:
            return np.nan, int(len(group)), int(group["mapped_event_key"].nunique()), False
        x = group[["canonical_skill_difference"]].to_numpy(float)
        y = group["canonical_outcome"].to_numpy(int)
        counts = group.groupby("mapped_event_key", observed=True)["pair_key"].transform("count")
        weights = 1.0 / counts.to_numpy(float)
        model = LogisticRegression(
            C=float(regularization_c),
            fit_intercept=False,
            solver="lbfgs",
            max_iter=500,
        )
        model.fit(x, y, sample_weight=weights)
        return (
            float(model.coef_[0, 0]),
            int(len(group)),
            int(group["mapped_event_key"].nunique()),
            True,
        )

    baseline_coefficients: dict[tuple[int, str], tuple[float, str]] = {}
    fit_rows: list[dict[str, Any]] = []
    for year in EVALUATION_YEARS:
        prior_year = year - 1
        for market in markets:
            train = resolved.loc[
                resolved["candidate"].eq(baseline_candidate)
                & resolved["market"].eq(market)
                & resolved["year"].eq(prior_year)
            ]
            coefficient, pairs, events, success = fit_coefficient(train)
            if success:
                source = "strict_pt_baseline_previous_year"
            else:
                coefficient = 1.0
                source = "fixed_unit_logit_prior"
            baseline_coefficients[(year, market)] = (float(coefficient), source)
            fit_rows.append(
                {
                    "candidate": baseline_candidate,
                    "market": market,
                    "evaluation_year": year,
                    "training_year": prior_year,
                    "regularization_c": float(regularization_c),
                    "coefficient": float(coefficient),
                    "fit_pairs": pairs,
                    "fit_events": events,
                    "fit_source": source,
                    "prediction_units": "market_logit_score",
                    "neutral_base_finalist_eligible": False,
                }
            )
    for candidate in candidates:
        if candidate == baseline_candidate:
            continue
        for year in EVALUATION_YEARS:
            prior_year = year - 1
            for market in markets:
                train = resolved.loc[
                    resolved["candidate"].eq(candidate)
                    & resolved["market"].eq(market)
                    & resolved["year"].eq(prior_year)
                ]
                coefficient, pairs, events, success = fit_coefficient(train)
                if success:
                    source = "candidate_previous_year"
                else:
                    coefficient, baseline_source = baseline_coefficients[(year, market)]
                    source = f"strict_pt_baseline_fallback:{baseline_source}"
                fit_rows.append(
                    {
                        "candidate": candidate,
                        "market": market,
                        "evaluation_year": year,
                        "training_year": prior_year,
                        "regularization_c": float(regularization_c),
                        "coefficient": float(coefficient),
                        "fit_pairs": pairs,
                        "fit_events": events,
                        "fit_source": source,
                        "prediction_units": "market_logit_score",
                        "neutral_base_finalist_eligible": False,
                    }
                )
    fits = pd.DataFrame(fit_rows)
    score_rows: list[dict[str, Any]] = []
    for row in fits.to_dict(orient="records"):
        test = resolved.loc[
            resolved["candidate"].eq(row["candidate"])
            & resolved["market"].eq(row["market"])
            & resolved["year"].eq(int(row["evaluation_year"]))
        ].copy()
        if test.empty:
            continue
        probability = np.clip(
            expit(float(row["coefficient"]) * test["canonical_skill_difference"].to_numpy(float)),
            market_eval.EPS,
            1.0 - market_eval.EPS,
        )
        y = test["canonical_outcome"].to_numpy(float)
        loss = -(y * np.log(probability) + (1.0 - y) * np.log(1.0 - probability))
        event_loss = pd.DataFrame(
            {"event": test["mapped_event_key"].to_numpy(), "loss": loss}
        ).groupby("event", observed=True)["loss"].mean()
        score_rows.append(
            {
                **row,
                "events": int(event_loss.index.nunique()),
                "resolved_pairs": int(len(test)),
                "event_equal_log_loss": float(event_loss.mean()),
            }
        )
    scores = pd.DataFrame(score_rows)
    baseline_scores = scores.loc[scores["candidate"].eq(baseline_candidate), [
        "market", "evaluation_year", "event_equal_log_loss"
    ]].rename(columns={"event_equal_log_loss": "baseline_event_equal_log_loss"})
    scores = scores.merge(
        baseline_scores,
        on=["market", "evaluation_year"],
        how="left",
        validate="many_to_one",
    )
    scores["log_loss_delta"] = (
        scores["event_equal_log_loss"] - scores["baseline_event_equal_log_loss"]
    )
    pooled = scores.loc[scores["candidate"].isin(candidate_names)].groupby(
        "candidate", observed=True
    ).agg(
        pooled_event_equal_log_loss=("event_equal_log_loss", "mean"),
        pooled_log_loss_delta=("log_loss_delta", "mean"),
        temporal_fit_rows=("evaluation_year", "size"),
    ).reset_index()
    if not pooled.empty:
        pooled = pooled.sort_values(
            ["pooled_event_equal_log_loss", "candidate"], kind="mergesort"
        )
        pooled["selected_market_ranking_sensitivity"] = False
        pooled.loc[pooled.index[0], "selected_market_ranking_sensitivity"] = True
    pooled["prediction_units"] = "market_logit_score"
    pooled["neutral_base_finalist_eligible"] = False
    pooled["selection_window_end"] = DISCOVERY_END.date().isoformat()
    pooled["sealed_data_used_for_selection"] = False
    return scores, pooled


def _write_report(
    output: Path,
    scorecard: pd.DataFrame,
    finalists: Sequence[str],
    audit: Mapping[str, Any],
) -> None:
    columns = [
        "candidate",
        "minimum_coverage_ratio",
        "pooled_log_loss_delta",
        "worst_market_log_loss_delta",
        "calibration_slope_error_delta",
        "ece_delta",
        "close_pnl_delta",
        "open_to_close_clv_delta",
        "max_drawdown_delta",
        "gate_stage_reached",
        "all_gates_pass",
    ]
    table = scorecard[[column for column in columns if column in scorecard]].sort_values(
        ["all_gates_pass", "gate_stage_reached", "pooled_log_loss_delta"],
        ascending=[False, False, True],
    )
    lines = [
        "# Complete base-skill candidate discovery selection",
        "",
        "Research only. No production file or consumer was changed.",
        "",
        "Probability calibration is temporally cross-fitted: 2021 uses 2020 only and 2022 uses 2021 only. The final sealed mapper is separately fit on all 2021-22 after scoring and is not reused to score discovery.",
        "",
        "Selection is fail-closed. A challenger must clear coverage, proper-score, calibration/edge-monotonicity, money/CLV, and concentration gates versus the strict point-in-time current-architecture control. No candidate is selected merely because it is least bad.",
        "",
        f"Selected finalists: {', '.join(finalists) if finalists else 'none'}",
        "",
        "```text",
        table.to_string(index=False, float_format=lambda value: f"{value:.6f}"),
        "```",
        "",
        "Opening-price CLV is an asynchronous archive sensitivity; it is a discovery guardrail, not proof those openers were simultaneously executable.",
        "",
        "## Audit",
        "",
        "```json",
        json.dumps(dict(audit), indent=2, sort_keys=True, default=str),
        "```",
        "",
    ]
    (output / "DISCOVERY_SELECTION_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def run(args: argparse.Namespace) -> None:
    frozen_selection_contract = validate_frozen_selection_cli(args)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"discovery output directory is not empty; use a new versioned path: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = args.predictions.resolve()
    catalog_path = args.candidate_catalog.resolve()
    source_manifest_path = args.candidate_manifest.resolve()
    cache_dir = args.evaluation_cache.resolve()
    core_bridge_path = args.core_bridge.resolve()
    (
        _source_manifest_preflight,
        pre_bindings,
        _study_authority_audit_preflight,
        pre_study_authority_bindings,
    ) = resolve_discovery_direct_inputs(
        predictions_path,
        catalog_path,
        source_manifest_path,
        cache_dir,
        core_bridge_path,
    )
    (
        _cache_manifest_preflight,
        _cache_authority_audit_preflight,
        pre_cache_authority_bindings,
    ) = validate_evaluation_cache_authority(cache_dir, core_bridge_path)
    for direct_name, authority_name in (
        ("evaluation_cache_manifest", "evaluation_cache_manifest"),
        ("normalized_quotes", "normalized_quotes"),
        ("feature_snapshots", "feature_snapshots"),
        ("core_bridge", "cache_input_legacy_core_bridge"),
    ):
        if pre_bindings[direct_name] != pre_cache_authority_bindings[authority_name]:
            raise ValueError(
                f"pre-begin discovery input is not cache-authorized: {direct_name}"
            )
    direct_inputs = list(
        dict.fromkeys(
            [*pre_bindings.values(), *pre_cache_authority_bindings.values()]
            + list(pre_study_authority_bindings.values())
        )
    )
    provenance_state = execution_provenance.begin(
        output_dir=output,
        primary_source=Path(__file__),
        input_paths=direct_inputs,
        code_paths=PROVENANCE_CODE_PATHS,
    )
    (
        post_source_manifest,
        post_bindings,
        post_cache_manifest,
        post_cache_authority_audit,
        post_cache_authority_bindings,
        post_study_authority_audit,
        post_study_authority_bindings,
    ) = revalidate_discovery_inputs_after_begin(
        predictions_path=predictions_path,
        catalog_path=catalog_path,
        source_manifest_path=source_manifest_path,
        cache_dir=cache_dir,
        core_bridge_path=core_bridge_path,
        expected_bindings=pre_bindings,
        expected_study_authority_bindings=pre_study_authority_bindings,
        expected_cache_authority_bindings=pre_cache_authority_bindings,
        provenance_state=provenance_state,
    )
    frozen_catalog_path = post_bindings["frozen_candidate_catalog"]
    coverage_path = post_bindings["coverage_by_candidate_year"]
    core_bridge_path = post_bindings["core_bridge"]
    (
        predictions,
        catalog,
        source_manifest,
        quotes,
        snapshots,
        cache_manifest,
    ) = load_and_validate_inputs(
        post_bindings["predictions"],
        post_bindings["candidate_catalog"],
        post_bindings["candidate_manifest"],
        cache_dir,
        source_manifest_payload=post_source_manifest,
        cache_manifest_payload=post_cache_manifest,
        cache_authority_audit=post_cache_authority_audit,
        completed_study_authority_audit=post_study_authority_audit,
        expected_frozen_catalog_path=frozen_catalog_path,
        expected_coverage_path=coverage_path,
        expected_normalized_quotes_path=post_cache_authority_bindings[
            "normalized_quotes"
        ],
        expected_feature_snapshots_path=post_cache_authority_bindings[
            "feature_snapshots"
        ],
        expected_fit_log_path=post_study_authority_bindings["fit_log"],
        expected_envelope_candidate_path=post_study_authority_bindings[
            "envelope_admission_by_candidate"
        ],
        expected_envelope_fold_path=post_study_authority_bindings[
            "envelope_admission_by_fold"
        ],
    )
    if args.baseline_candidate not in set(predictions["candidate"]):
        raise ValueError(f"baseline candidate {args.baseline_candidate!r} is unavailable")

    prediction_candidates = set(predictions["candidate"].astype(str))
    catalog = catalog.loc[catalog["name"].astype(str).isin(prediction_candidates)].copy()
    eligible_names = set(
        catalog.loc[
            catalog["selection_eligible"]
            & ~catalog["sensitivity_only"]
            & ~catalog["control"],
            "name",
        ].astype(str)
    )
    control_names = set(catalog.loc[catalog["control"], "name"].astype(str))
    screen_names = eligible_names | control_names | {args.baseline_candidate}
    predictions = predictions.loc[predictions["candidate"].isin(screen_names)].copy()
    paired = market_eval.attach_skill_differences(quotes, predictions)
    if paired.empty:
        raise ValueError("no external predictions mapped to the frozen matchup board")
    mapper = fit_temporal_probability_mappers(
        paired,
        args.baseline_candidate,
        evaluation_years=EVALUATION_YEARS,
        min_pairs=args.min_calibration_pairs,
    )
    mapper.to_csv(output / "crossfit_probability_parameters.csv", index=False)
    scored = score_temporal_discovery(paired, mapper)
    logical = canonical_discovery_probabilities(scored)
    logical_columns = [
        "candidate",
        "mapped_event_key",
        "event_start_date",
        "market",
        "pair_key",
        "canonical_outcome",
        "canonical_probability",
        "canonical_push_probability",
        "probability_calibration_year",
        "probability_calibration_source",
    ]
    _atomic_parquet(
        logical[[column for column in logical_columns if column in logical]],
        output / "discovery_crossfit_logical_probabilities.parquet",
    )
    proper = paired_common_proper_scores(logical, args.baseline_candidate)
    proper.to_csv(output / "discovery_crossfit_proper_scores.csv", index=False)
    aggregate = aggregate_candidate_scores(proper)

    # Money/edge/CLV summaries are streamed one candidate at a time.  They are
    # needed for frozen discovery gates but broad candidate ticket frames are
    # never retained or written.
    guardrail_rows: list[dict[str, Any]] = []
    for candidate in sorted(eligible_names):
        candidate_score = aggregate.loc[aggregate["candidate"].eq(candidate)]
        if candidate_score.empty:
            continue
        coverage = float(candidate_score["minimum_coverage_ratio"].iloc[0])
        proper_delta = float(candidate_score["pooled_log_loss_delta"].iloc[0])
        worst_market = float(candidate_score["worst_market_log_loss_delta"].iloc[0])
        if coverage < 1.0 - args.coverage_tolerance:
            continue
        if proper_delta >= 0.0 or worst_market > args.logloss_tolerance:
            continue
        guardrail_rows.append(
            discovery_guardrails(scored, candidate, args.baseline_candidate)
        )
    guardrails = guardrail_summary_frame(guardrail_rows)
    guardrails.to_csv(output / "discovery_guardrail_summaries.csv", index=False)
    scorecard = aggregate.loc[aggregate["candidate"].isin(eligible_names)].copy()
    scorecard = scorecard.merge(guardrails, on="candidate", how="left")
    guardrail_defaults = {
        "close_pnl_delta": -np.inf,
        "close_roi_delta": -np.inf,
        "candidate_mean_open_to_close_clv": -np.inf,
        "open_to_close_clv_delta": -np.inf,
        "max_drawdown_delta": np.inf,
        "worst_event_loss_delta": np.inf,
        "edge_violation_magnitude_delta": np.inf,
        "candidate_edge_violations": np.inf,
        "baseline_edge_violations": 0,
    }
    for column, default in guardrail_defaults.items():
        if column not in scorecard:
            scorecard[column] = default
        else:
            scorecard[column] = scorecard[column].fillna(default)
    scorecard = apply_selection_gates(
        scorecard,
        coverage_tolerance=args.coverage_tolerance,
        logloss_tolerance=args.logloss_tolerance,
        calibration_tolerance=args.calibration_tolerance,
        money_tolerance=args.money_tolerance,
    )
    finalists = choose_finalists(scorecard, catalog, args.finalist_count)
    diagnostic_challenger, diagnostic_failed_gates = choose_diagnostic_challenger(
        scorecard, catalog
    )
    frozen_selected_candidate = finalists[0] if finalists else diagnostic_challenger
    locked_diagnostic_ablation_set = build_locked_diagnostic_ablation_set(
        frozen_selected_candidate, catalog
    )
    scorecard["selected_finalist"] = scorecard["candidate"].isin(finalists)
    scorecard["selected_diagnostic_challenger"] = scorecard["candidate"].eq(
        diagnostic_challenger
    )
    scorecard.to_csv(output / "candidate_gate_scorecard.csv", index=False)

    final_mapper = fit_final_discovery_mappers(
        paired, args.baseline_candidate, args.min_calibration_pairs
    )
    final_mapper.to_csv(output / "final_frozen_probability_parameters.csv", index=False)

    downstream_choices = pd.DataFrame()
    opener_choices = pd.DataFrame()
    pairwise_choices = pd.DataFrame()
    sensitivity_audit: dict[str, Any] = {"skipped": bool(args.skip_sensitivities)}
    if not args.skip_sensitivities:
        lambda_scores, downstream_choices, downstream_payload = (
            downstream_lambda_sensitivity(
                predictions,
                quotes,
                core_bridge_path,
                args.baseline_candidate,
                eligible_names,
                args.downstream_lambdas,
                args.min_calibration_pairs,
            )
        )
        lambda_scores.to_csv(
            output / "downstream_lambda_discovery_scores.csv", index=False
        )
        downstream_choices.to_csv(
            output / "downstream_lambda_selection.csv", index=False
        )
        downstream_payload["parameters"].to_csv(
            output / "downstream_lambda_crossfit_parameters.csv", index=False
        )
        opener_scores, opener_choices, opener_audit = opener_blend_sensitivity(
            scored, args.opener_weights
        )
        opener_scores.to_csv(output / "opener_blend_discovery_scores.csv", index=False)
        opener_choices.to_csv(output / "opener_blend_selection.csv", index=False)
        _atomic_json(output / "opener_blend_audit.json", opener_audit)
        pairwise_scores, pairwise_choices = pairwise_logistic_sensitivity(
            paired,
            args.baseline_candidate,
            eligible_names,
            regularization_c=args.pairwise_regularization_c,
            min_pairs=args.min_calibration_pairs,
        )
        pairwise_scores.to_csv(
            output / "pairwise_logistic_discovery_scores.csv", index=False
        )
        pairwise_choices.to_csv(
            output / "pairwise_logistic_selection.csv", index=False
        )
        sensitivity_audit = {
            "skipped": False,
            "downstream": downstream_payload["audit"],
            "opener_weights": [float(value) for value in args.opener_weights],
            "opener_candidates": int(opener_choices["candidate"].nunique()),
            "opener": opener_audit,
            "pairwise_regularization_c": float(args.pairwise_regularization_c),
            "pairwise_candidates": int(pairwise_choices["candidate"].nunique()),
            "pairwise_prediction_units": "market_logit_score",
            "pairwise_neutral_base_finalist_eligible": False,
            "neutral_base_redefined": False,
        }

    money_names = (
        set(finalists)
        | ({diagnostic_challenger} if diagnostic_challenger else set())
        | control_names
        | {args.baseline_candidate}
    )
    money_scored = scored.loc[scored["candidate"].isin(money_names)].copy()
    money_bets = market_eval.select_and_settle_bets(
        money_scored, "best_of_three", "flat_1u", min_ev=market_eval.EV_GATE
    )
    if not money_bets.empty:
        money_bets["sample"] = "discovery_crossfit"
        money_bets["line_policy"] = "best_of_three"
        money_bets["opportunity_mode"] = "all_available"
        _atomic_parquet(money_bets, output / "discovery_money_bets.parquet")
        performance = market_eval.betting_performance(
            money_bets, "discovery_crossfit"
        )
    else:
        performance = pd.DataFrame()
    performance.to_csv(output / "discovery_money_performance.csv", index=False)
    edge = market_eval.edge_bucket_performance(
        money_scored, "discovery_crossfit", "best_of_three"
    )
    edge.to_csv(output / "discovery_edge_bucket_performance.csv", index=False)
    reliability_bets, reliability_suppressed = select_reliability_gated_bets(
        money_scored,
        min_ev=market_eval.EV_GATE,
    )
    if not reliability_bets.empty:
        reliability_bets["sample"] = "discovery_crossfit"
        reliability_bets["line_policy"] = "best_of_three"
        reliability_bets["opportunity_mode"] = "conservative_reliability_gate"
        _atomic_parquet(
            reliability_bets, output / "discovery_reliability_gate_bets.parquet"
        )
        reliability_performance = market_eval.betting_performance(
            reliability_bets, "discovery_crossfit"
        )
    else:
        reliability_performance = pd.DataFrame()
    reliability_performance.to_csv(
        output / "discovery_reliability_gate_performance.csv", index=False
    )
    if not reliability_suppressed.empty:
        _atomic_parquet(
            reliability_suppressed,
            output / "discovery_reliability_gate_suppressed.parquet",
        )
    reliability_edge = reliability_edge_bucket_performance(
        money_scored, "discovery_crossfit"
    )
    reliability_edge.to_csv(
        output / "discovery_reliability_gate_edge_buckets.csv", index=False
    )

    prediction_dates = predictions.groupby("candidate", observed=True)["event_start_date"].agg(
        ["min", "max"]
    )
    warmup_candidates = sorted(
        prediction_dates.loc[prediction_dates["min"].dt.year <= 2020].index.astype(str)
    )
    pair_counts = paired.groupby("candidate", observed=True)["quote_id"].nunique().to_dict()
    audit = {
        "research_only": True,
        "production_files_changed": False,
        "selection_period": "2021-01-01/2022-12-31",
        "calibration_warmup_period": "2020-01-01/2020-12-31",
        "sealed_data_used_for_selection": False,
        "selection_uses_sealed": False,
        "closing_prices_used_as_features": False,
        "closing_prices_used_for_evaluation": True,
        "opening_price_clv_is_asynchronous_sensitivity": True,
        "prediction_rows": int(len(predictions)),
        "prediction_candidates": int(predictions["candidate"].nunique()),
        "eligible_candidates": int(len(eligible_names)),
        "control_candidates": sorted(control_names),
        "warmup_candidates": warmup_candidates,
        "prediction_key_duplicates": int(
            predictions.duplicated(["candidate", "event_key", "dg_id_key"]).sum()
        ),
        "prediction_chronology_violations": 0,
        "snapshot_chronology_violations": 0,
        "quote_rows_2020_2022": int(len(quotes)),
        "quote_key_duplicates": int(quotes.duplicated("quote_id").sum()),
        "mapped_quote_rows_by_candidate": {str(key): int(value) for key, value in pair_counts.items()},
        "crossfit_mapper_rows": int(len(mapper)),
        "crossfit_mapper_future_violations": int(
            (mapper["calibration_year"] >= mapper["evaluation_year"]).sum()
        ),
        "crossfit_baseline_fallback_rows": int(
            mapper["calibration_source"].astype(str).str.startswith("strict_pt_baseline_fallback").sum()
        ),
        "crossfit_fixed_prior_rows": int(
            mapper["calibration_source"].eq("fixed_prespecified_prior").sum()
        ),
        "crossfit_sigma_near_upper_bound_rows": int(mapper["sigma_near_upper_bound"].sum()),
        "canonical_probability_rows": int(len(logical)),
        "canonical_probability_key_duplicates": int(
            logical.duplicated(["candidate", "mapped_event_key", "market", "pair_key"]).sum()
        ),
        "proper_score_rows": int(len(proper)),
        "guardrail_candidates_evaluated": int(len(guardrails)),
        "finalists": finalists,
        "diagnostic_challenger": diagnostic_challenger,
        "diagnostic_challenger_failed_gates": diagnostic_failed_gates,
        "locked_diagnostic_ablation_set": locked_diagnostic_ablation_set,
        "promotion_gate_passed": bool(finalists),
        "selection_status": "promotion_winner" if finalists else "diagnostic_only_no_candidate_passed_all_gates",
        "money_materialized_candidates": sorted(money_names),
        "money_bet_rows": int(len(money_bets)),
        "reliability_gate_materialized_candidates": sorted(
            reliability_bets["candidate"].astype(str).unique()
        ) if not reliability_bets.empty else [],
        "reliability_gate_bet_rows": int(len(reliability_bets)),
        "reliability_gate_suppressed_rows": int(len(reliability_suppressed)),
        "reliability_gate_suppressed_by_candidate": {
            str(key): int(value)
            for key, value in reliability_suppressed.groupby(
                "candidate", observed=True
            ).size().items()
        } if not reliability_suppressed.empty else {},
        "reliability_gate_suppression_reasons": {
            str(key): int(value)
            for key, value in reliability_suppressed.groupby(
                "suppression_reason", observed=True
            ).size().items()
        } if not reliability_suppressed.empty else {},
        "reliability_gate_rows_retain_pair_reliability": bool(
            reliability_bets.empty or "pair_reliability_ok" in reliability_bets
        ),
        "reliability_gate_rule": (
            "suppress nominal EV >=10% when either player fails the completion "
            "panel's thin/missing/stale/prior-WD/identity reliability audit"
        ),
        "gate_tolerances": {
            "coverage": float(args.coverage_tolerance),
            "logloss": float(args.logloss_tolerance),
            "calibration": float(args.calibration_tolerance),
            "money": float(args.money_tolerance),
        },
        "baseline_candidate": args.baseline_candidate,
        "min_calibration_pairs": int(args.min_calibration_pairs),
        "frozen_selection_parameter_contract": frozen_selection_contract,
        "sensitivities": sensitivity_audit,
        "completion_builder_provenance": source_manifest.get(
            "_completion_provenance_audit", {}
        ),
        "source_prediction_filter": source_manifest.get("_selection_filter_audit", {}),
        "control_integration": source_manifest.get("_control_integration_audit", {}),
        "frozen_catalog": source_manifest.get("_frozen_catalog_audit", {}),
        "discovery_candidate_set": source_manifest.get(
            "_discovery_candidate_set_audit", {}
        ),
        "identical_candidate_menus": source_manifest.get(
            "_identical_menu_audit", {}
        ),
        "calibration_warmup_menu": source_manifest.get(
            "_calibration_warmup_audit", {}
        ),
        "retrospective_control_partial_coverage": source_manifest.get(
            "_retrospective_control_menu_audit", {}
        ),
        "quote_identity_and_outcome_integrity": cache_manifest.get(
            "_quote_integrity_audit", {}
        ),
        "same_book_price_coverage": cache_manifest.get(
            "_same_book_price_coverage_audit", {}
        ),
        "evaluation_cache_authority": cache_manifest.get(
            "_authority_audit", {}
        ),
        "completed_study_authority": source_manifest.get(
            "_completed_study_authority_audit", {}
        ),
        "discovery_only_envelope_admission": source_manifest.get(
            "_discovery_envelope_admission_audit", {}
        ),
    }
    _atomic_json(output / "audit.json", audit)
    _write_report(output, scorecard, finalists, audit)

    diagnostic_manifest: dict[str, Any] = {
        "contract": "base_skill_discovery_selection_v1",
        "complete": True,
        "research_only": True,
        "selection_role": "diagnostic_challenger",
        "diagnostic_challenger": diagnostic_challenger,
        "selected_candidate": diagnostic_challenger,
        "selection_window_end": DISCOVERY_END.date().isoformat(),
        "selection_uses_sealed": False,
        "sealed_data_used_for_selection": False,
        "promotion_gate_passed": bool(finalists),
        "failed_gates": diagnostic_failed_gates,
        "uses_closing_odds_as_feature": False,
        "source_predictions": str(predictions_path),
        "source_predictions_sha256": _sha256(predictions_path),
        "locked_diagnostic_ablation_set": locked_diagnostic_ablation_set,
        "frozen_selection_parameter_contract": frozen_selection_contract,
        "frozen_candidate_catalog_sha256": EXPECTED_FROZEN_CATALOG_SHA256,
    }
    _atomic_json(output / "diagnostic_selection_manifest.json", diagnostic_manifest)

    execution_audit = execution_provenance.finalize(provenance_state)

    artifacts: list[dict[str, Any]] = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "selection_manifest.json":
            artifacts.append(
                {
                    "path": path.relative_to(output).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    selection_manifest: dict[str, Any] = {
        "contract": "base_skill_discovery_selection_v1",
        "complete": True,
        "research_only": True,
        "selection_window_end": DISCOVERY_END.date().isoformat(),
        "selection_uses_sealed": False,
        "sealed_data_used_for_selection": False,
        "uses_closing_odds_as_feature": False,
        "baseline_candidate": args.baseline_candidate,
        "frozen_selection_parameter_contract": frozen_selection_contract,
        "frozen_candidate_catalog_sha256": EXPECTED_FROZEN_CATALOG_SHA256,
        "finalists": finalists,
        "selection_status": audit["selection_status"],
        "diagnostic_challenger": diagnostic_challenger,
        "diagnostic_challenger_failed_gates": diagnostic_failed_gates,
        "locked_diagnostic_ablation_set": locked_diagnostic_ablation_set,
        "promotion_gate_passed": bool(finalists),
        "crossfit_probability_rule": {
            "2021": "fit on 2020 only",
            "2022": "fit on 2021 only",
            "fallback": "same-period strict-PT baseline mapper; fixed prespecified prior only if baseline is insufficient",
        },
        "final_frozen_probability_parameters": final_mapper.to_dict(orient="records"),
        "downstream_preservation_sensitivity": {
            "neutral_base_redefined": False,
            "lambda_one_is_isolation_default": True,
            "choices": downstream_choices.to_dict(orient="records"),
        },
        "opener_blend_sensitivity": {
            "neutral_base_redefined": False,
            "asynchronous_openers_not_execution_proof": True,
            "formula": "logit(p_blend)=(1-w)*logit(p_model)+w*logit(p_open_novig)",
            "line_policy": "best available opening American price across betonline/betcris/pinnacle; same-book opponent retained",
            "coverage_audit": sensitivity_audit.get("opener", {}),
            "choices": opener_choices.to_dict(orient="records"),
        },
        "pairwise_logistic_sensitivity": {
            "neutral_base_redefined": False,
            "prediction_units": "market_logit_score",
            "neutral_base_finalist_eligible": False,
            "regularization_c": float(args.pairwise_regularization_c),
            "choices": pairwise_choices.to_dict(orient="records"),
        },
        "gate_tolerances": audit["gate_tolerances"],
        "execution_provenance": {
            "path": "execution_provenance/execution_provenance.json",
            "start_utc": execution_audit["start_utc"],
            "end_utc": execution_audit["end_utc"],
            "pid": execution_audit["pid"],
            "verified_unchanged_before_complete_manifest": True,
        },
        "inputs": {
            "predictions": {
                "path": str(predictions_path),
                "sha256": _sha256(predictions_path),
            },
            "candidate_catalog": {
                "path": str(catalog_path),
                "sha256": _sha256(catalog_path),
            },
            "frozen_candidate_catalog": {
                "path": str(frozen_catalog_path),
                "sha256": _sha256(frozen_catalog_path),
            },
            "candidate_manifest": {
                "path": str(source_manifest_path),
                "sha256": _sha256(source_manifest_path),
            },
            "normalized_quotes": {
                "path": str(cache_dir / "normalized_matchup_quotes.parquet"),
                "sha256": _sha256(cache_dir / "normalized_matchup_quotes.parquet"),
            },
            "feature_snapshots": {
                "path": str(cache_dir / "feature_snapshots.parquet"),
                "sha256": _sha256(cache_dir / "feature_snapshots.parquet"),
            },
            "evaluation_cache_manifest": {
                "path": str(cache_dir / "manifest.json"),
                "sha256": _sha256(cache_dir / "manifest.json"),
            },
            "coverage_by_candidate_year": {
                "path": str(coverage_path.resolve()),
                "sha256": _sha256(coverage_path.resolve()),
            },
            "core_bridge": {
                "path": str(core_bridge_path),
                "sha256": _sha256(core_bridge_path),
            },
        },
        "artifacts": artifacts,
    }
    if finalists:
        selection_manifest["selected_candidate"] = finalists[0]
        selection_manifest["finalist_name"] = finalists[0]
    write_discovery_complete_manifest(
        output / "selection_manifest.json", selection_manifest, provenance_state
    )
    print(f"Completed discovery-only external selection: {output}")
    print(f"Finalists: {finalists if finalists else 'none'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--candidate-catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--candidate-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--evaluation-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--core-bridge", type=Path, default=DEFAULT_CORE_BRIDGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline-candidate", default=DEFAULT_BASELINE)
    parser.add_argument(
        "--min-calibration-pairs", type=int, default=FROZEN_MIN_CALIBRATION_PAIRS
    )
    parser.add_argument("--finalist-count", type=int, default=FROZEN_FINALIST_COUNT)
    parser.add_argument(
        "--coverage-tolerance",
        type=float,
        default=FROZEN_GATE_TOLERANCES["coverage"],
    )
    parser.add_argument(
        "--logloss-tolerance",
        type=float,
        default=FROZEN_GATE_TOLERANCES["logloss"],
    )
    parser.add_argument(
        "--calibration-tolerance",
        type=float,
        default=FROZEN_GATE_TOLERANCES["calibration"],
    )
    parser.add_argument(
        "--money-tolerance", type=float, default=FROZEN_GATE_TOLERANCES["money"]
    )
    parser.add_argument(
        "--pairwise-regularization-c",
        type=float,
        default=FROZEN_PAIRWISE_REGULARIZATION_C,
    )
    parser.add_argument(
        "--downstream-lambdas",
        type=lambda value: tuple(float(item.strip()) for item in value.split(",") if item.strip()),
        default=FROZEN_DOWNSTREAM_LAMBDAS,
    )
    parser.add_argument(
        "--opener-weights",
        type=lambda value: tuple(float(item.strip()) for item in value.split(",") if item.strip()),
        default=FROZEN_OPENER_WEIGHTS,
    )
    parser.add_argument("--skip-sensitivities", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
