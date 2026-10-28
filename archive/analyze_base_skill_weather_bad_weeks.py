"""Post-sealed weather and bad-week attribution for the base-skill rebuild.

This is a research-only consumer of already-frozen evaluation artifacts.  It
does not select a model, refit a probability mapper, or change any weekly
pipeline input.  It answers two deliberately separate questions:

* W1: did point-in-time normalization of *historical* rounds improve the
  frozen pre-event/R1 base-skill result relative to an otherwise matched W0
  arm?
* W2: did point-in-time normalization of completed-round observations improve
  the frozen R1-R4 cascade result relative to its otherwise matched W0 arm?

Realized weather from the event/round being priced is never treated as an
available feature.  It is attached only after settlement as an explicitly
labelled oracle attribution.  W2 may consume realized weather from rounds
that were already complete at the target-round boundary; R1 therefore has no
current-event completed-round weather.

The script fails closed on artifact hashes, provenance flags, schemas,
ambiguous W0/W1 or W0/W2 pairings, mismatched quote menus, settlement gaps,
and event/weather identity gaps.  Use a fresh output directory for every run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
ARCHIVE = Path(__file__).resolve().parent
for _path in (str(REPO), str(ARCHIVE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from archive import base_skill_frozen_handoff as frozen_handoff  # noqa: E402
from archive import research_execution_provenance as execution_provenance  # noqa: E402


RESEARCH_ROOT = (
    REPO / "permanent_data" / "research" / "base_skill_rebuild_20200101_20260822"
)
DEFAULT_SEALED_DIR = RESEARCH_ROOT / "model_eval_complete_sealed_v3_provenance"
DEFAULT_CASCADE_DIR = RESEARCH_ROOT / "cascade_eval_v3_provenance"
DEFAULT_PANEL_DIR = RESEARCH_ROOT
DEFAULT_OUTPUT_DIR = RESEARCH_ROOT / "weather_bad_week_audit_v1"

SEALED_CONTRACT = "base_skill_frozen_sealed_matchup_v1"
CASCADE_TERMINAL_CONTRACT = "base_skill_round_cascade_terminalization_v1"
CASCADE_TERMINAL_MANIFEST_SHA256 = (
    "6f36026612d99853b941e38696baa81a866f822c8ef1ea922d7bf5444816c139"
)
CASCADE_TERMINALIZER_PRIMARY_SOURCE = (
    ARCHIVE / "terminalize_base_skill_round_cascade.py"
)
CASCADE_TERMINALIZER_SOURCE_SHA256 = (
    "c47819a36b7aff4a8446b300692b7001ba1b128ed6d19d226ca162c5f8ecae0e"
)
CASCADE_TERMINAL_NUMERIC_ARTIFACTS = frozenset(
    {
        "all_offered_edge_bets.parquet",
        "all_offered_edge_bucket_performance.csv",
        "audit.json",
        "bad_week_golfers.csv",
        "bad_weeks.csv",
        "bet_bucket_performance.csv",
        "bets.parquet",
        "c3_coefficients.csv",
        "c3_discovery_hyperparameter_scores.csv",
        "calibration.csv",
        "candidate_round_coverage.csv",
        "candidate_selection.csv",
        "cascade_player_round_predictions.parquet",
        "counterfactual_zero_gate_bets.parquet",
        "counterfactual_zero_gate_bucket_performance.csv",
        "discovery_probability_sigmas.csv",
        "event_cluster_bootstrap.csv",
        "model_quote_coverage.csv",
        "normalized_round_matchup_quotes.parquet",
        "performance.csv",
        "proper_scores.csv",
        "round_matchup_probabilities.parquet",
        "settlement_completeness_audit.json",
    }
)
SEALED_START = pd.Timestamp("2023-01-01")
SEALED_END = pd.Timestamp("2026-08-22")
MIN_BOOTSTRAP_DRAWS = 10_000
EPS = 1e-12
CASCADE_CONTROL = "C0_current_cascade_retrospective"
EXPECTED_BOOKS = ("betonline", "betcris", "pinnacle")
EXPECTED_LINE_POLICIES = (*EXPECTED_BOOKS, "best_of_three")
EXPECTED_SEALED_MARKETS = ("72h", "r1")
EXPECTED_SEALED_PRICE_BASES = ("close", "open")
EXPECTED_SEALED_SIZINGS = ("flat_1u", "production_risk_to_win")
EXPECTED_SEALED_RELIABILITY_POLICIES = (
    "ungated",
    "suppress_unreliable_ge10pct",
)
EXPECTED_SEALED_OPPORTUNITY_MODES = (
    "all_available",
    "pairwise_common_with_strict_pt_baseline",
)
EXPECTED_CASCADE_PRICE_POINTS = ("open", "close")
EXPECTED_CASCADE_SIZINGS = ("flat_1u", "production_risk_to_win")
EXPECTED_CASCADE_OPPORTUNITY_MODES = ("ev_ge_3pct",)
STRICT_LEGACY_CONTROL = "legacy_skill_est_pt_current_coefficients_control"
RETROSPECTIVE_CONTROL = "legacy_pred_step2_exact_retrospective_control"
FROZEN_OPENER_SUFFIX = "__frozen_opener_blend"
SEALED_CATALOG_SIGNATURE_COLUMNS = (
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
    "prediction_envelope_applied",
    "prediction_envelope_method",
    "prediction_envelope_scale_factor",
    "prediction_envelope_multiplier",
    "raw_regression_component_selection_eligible",
)
SEALED_METADATA_CATALOG_COLUMNS = (
    "family",
    "source",
    "weather_arm",
    "target_arm",
    "refit_schedule",
    "alpha",
    "short_game_penalty",
    "control",
    "sensitivity_only",
    "selection_eligible",
)
W2_W0_VARIANTS = {
    "w0": "C3_pt_continuous_partial_pool_w0",
    "w2": "C3_pt_continuous_partial_pool_w2",
}
SLICE_COLUMNS = (
    "opportunity_mode",
    "price_basis",
    "line_policy",
    "sizing",
    "reliability_policy",
)
CASCADE_SLICE_COLUMNS = (
    "opportunity_mode",
    "price_point",
    "line_policy",
    "sizing",
)
EXECUTION_CONFIG_CONTRACT = "base_skill_weather_bad_week_execution_config_v1"
PANEL_MANIFEST_SHA256 = (
    "cd8dea51658a7899fa0c2d2dd534d52f456485e1528290c1fba27b33c471d3db"
)
PANEL_MANIFEST_BYTES = 12_767
PANEL_MANIFEST_AUTHORITY_PATH = RESEARCH_ROOT / "manifest.json"
PANEL_ROUND_AUTHORITY_PATH = RESEARCH_ROOT / "player_round_panel.parquet"
PANEL_OUTPUT_PINS = {
    "player_round_panel": {
        "sha256": "37df323c7e43681e02aa90bb12d84406bade24586315082e2e553ff323485f2a",
        "bytes": 44_577_853,
    },
    "player_event_outcomes": {
        "sha256": "5840e76073a5bd7e63798f53496d89e17caa3d80c3576a90a50ea81902c6668e",
        "bytes": 6_939_233,
    },
    "odds_event_crosswalk": {
        "sha256": "60959cfb194f27769c914d367a8bf3697a74966ef5ffa48a839722109d24267d",
        "bytes": 39_196,
    },
}
C3_WEATHER_LINEAGE_CONTRACT = "completed_prior_round_weather_state_v1"
C3_WEATHER_BOUNDARY = "round_num_strictly_less_than_target_round"
C3_CURRENT_WEATHER_ROLE = "posthoc_outcome_attribution_only_not_model_features"
C3_WEATHER_SOURCE_COLUMNS = {
    "w0": "target_total_w0|orth_ott_w0|orth_app_w0|orth_arg_w0|orth_putt_w0",
    "w2": "target_total_w2|orth_ott_w2|orth_app_w2|orth_arg_w2|orth_putt_w2",
}
C3_WEATHER_LINEAGE_FAILURE_KEYS = (
    "available_not_round_minus_one",
    "used_exceeds_available",
    "r1_uses_completed_state",
    "weather_available_exceeds_completed_rounds",
    "valid_weather_state_exceeds_available",
    "compat_weather_count_mismatch",
    "w2_weather_used_count_mismatch",
    "w2_weather_used_flag_mismatch",
    "prior_weather_coverage_rate_mismatch",
    "r1_claims_weather_use",
    "target_column_mismatch",
    "source_columns_mismatch",
    "coefficient_cutoff_not_strict_prior",
    "coefficient_source_missing",
    "coefficient_path_mismatch",
    "coefficient_hash_mismatch",
    "target_or_upcoming_exclusion_false",
    "completed_prior_only_false",
    "boundary_label_mismatch",
    "strict_pt_label_mismatch",
    "current_weather_role_mismatch",
)
CASCADE_PRICING_AUDIT_FIELDS = (
    "rows",
    "ties_lose_rows",
    "push_on_tie_rows",
    "zero_push_rows",
    "simplex_max_abs_error",
    "pricing_identity_max_abs_error",
    "zero_push_identity_max_abs_error",
    "ties_lose_identity",
    "push_on_tie_identity",
)
CASCADE_PRICING_IDENTITIES = {
    "ties_lose_identity": (
        "pricing_p1_probability_equals_unconditional_p1_win"
    ),
    "push_on_tie_identity": (
        "pricing_p1_probability_equals_p1_win_divided_by_one_minus_push"
    ),
}
CASCADE_SCORE_VOID_AUDIT_FIELDS = (
    "quotes",
    "archived_voids",
    "score_tie_voids",
    "score_tie_losses",
    "no_score_voids",
    "other_voids_with_two_non_tied_scores",
    "quotes_with_two_valid_round_scores",
    "score_model_eligible_quotes",
    "market_devig_proper_score_eligible_quotes",
    "ties_lose_quotes_excluded_from_market_proper_score",
    "archived_vs_score_outcome_mismatches_when_both_scores_valid",
    "p1_archived_vs_score_mismatches",
    "p2_archived_vs_score_mismatches",
    "impossible_archived_outcome_pairs",
    "rows_by_round",
)
# These hashes bind future sealed/cascade artifacts to the reviewed evaluator
# bytes.  They are intentionally updated only after those evaluators and their
# focused tests are frozen; an arbitrary shared-provenance payload is not an
# authority merely because its internal hashes are self-consistent.
SEALED_EVALUATOR_PRIMARY_SOURCE = (
    ARCHIVE / "evaluate_frozen_base_skill_sealed_matchups.py"
)
CASCADE_EVALUATOR_PRIMARY_SOURCE = ARCHIVE / "evaluate_base_skill_round_cascade.py"
SEALED_EVALUATOR_SOURCE_PINS = {
    SEALED_EVALUATOR_PRIMARY_SOURCE: (
        "b9d5d74621568497b85ab73d641617ab17877f9107d2d7c3eaaf1097198c0500"
    ),
    ARCHIVE / "base_skill_frozen_handoff.py": (
        "b072b2fdf18769db99e65f633ea24c28c5491fa1c659b0da300b59d7c9fc3a4f"
    ),
    ARCHIVE / "evaluate_complete_base_skill_market_candidates.py": (
        "1818f08cd03a8bde71058223686798c5eaf4aa68f714664671e7f42eb7442c49"
    ),
    ARCHIVE / "evaluate_complete_base_skill_market.py": (
        "92d298661c5f189703c960fcab54e2c7cda321e8d8bbc8351f3d615467374f47"
    ),
    ARCHIVE / "backtest_base_skill_market_models.py": (
        "0870d1eb7dd9364b3e7090b2f61643ddca59e536641f6fdb18971b004f22529d"
    ),
    ARCHIVE / "historical_odds_cache.py": (
        "58a7b774edc9a7e23aab772023846402c02f37396c1b8c7a0d3a4863feeb56c7"
    ),
    ARCHIVE / "market_backtest_scoring.py": (
        "53681535e0d1da7bde9cccede5ac09d7943e7493160c1ac2a248a0c0bfae2608"
    ),
    ARCHIVE / "research_execution_provenance.py": (
        "b5ed954a1c7578891614d319915848ca659ad7f2dd40fde77e146cda7eeb0034"
    ),
}
CASCADE_EVALUATOR_SOURCE_PINS = {
    CASCADE_EVALUATOR_PRIMARY_SOURCE: (
        "8544ad2c0076c95b737e4dc214c1bd5d2e05a5c9068f6242a9403f8794938202"
    ),
    ARCHIVE / "base_skill_frozen_handoff.py": (
        "b072b2fdf18769db99e65f633ea24c28c5491fa1c659b0da300b59d7c9fc3a4f"
    ),
    ARCHIVE / "evaluate_complete_base_skill_market_candidates.py": (
        "1818f08cd03a8bde71058223686798c5eaf4aa68f714664671e7f42eb7442c49"
    ),
    ARCHIVE / "evaluate_complete_base_skill_market.py": (
        "92d298661c5f189703c960fcab54e2c7cda321e8d8bbc8351f3d615467374f47"
    ),
    ARCHIVE / "backtest_base_skill_market_models.py": (
        "0870d1eb7dd9364b3e7090b2f61643ddca59e536641f6fdb18971b004f22529d"
    ),
    ARCHIVE / "historical_odds_cache.py": (
        "58a7b774edc9a7e23aab772023846402c02f37396c1b8c7a0d3a4863feeb56c7"
    ),
    ARCHIVE / "market_backtest_scoring.py": (
        "53681535e0d1da7bde9cccede5ac09d7943e7493160c1ac2a248a0c0bfae2608"
    ),
    ARCHIVE / "research_execution_provenance.py": (
        "b5ed954a1c7578891614d319915848ca659ad7f2dd40fde77e146cda7eeb0034"
    ),
}
WEATHER_PROVENANCE_CODE_PATHS = tuple(
    sorted(
        {
            *(Path(path).resolve() for path in SEALED_EVALUATOR_SOURCE_PINS),
            *(Path(path).resolve() for path in CASCADE_EVALUATOR_SOURCE_PINS),
            CASCADE_TERMINALIZER_PRIMARY_SOURCE.resolve(),
        },
        key=str,
    )
)


def _require(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _require_exact_bool(value: Any, expected: bool, label: str) -> None:
    if not isinstance(value, (bool, np.bool_)) or bool(value) is not expected:
        raise ValueError(f"{label} must be exactly {expected}")


def _require_exact_boolean_series(series: pd.Series, label: str) -> pd.Series:
    valid = series.map(lambda value: isinstance(value, (bool, np.bool_)))
    if not valid.all():
        examples = series.loc[~valid].astype(str).unique()[:5].tolist()
        raise ValueError(f"{label} must contain exact booleans; examples={examples}")
    return series.astype(bool)


def _require_nullable_exact_boolean_series(
    series: pd.Series, label: str
) -> pd.Series:
    missing = series.isna()
    valid = missing | series.map(lambda value: isinstance(value, (bool, np.bool_)))
    if not valid.all():
        examples = series.loc[~valid].astype(str).unique()[:5].tolist()
        raise ValueError(
            f"{label} must contain exact booleans or missing values; "
            f"examples={examples}"
        )
    return series.where(~missing, pd.NA).astype("boolean")


CASCADE_LEGACY_NULLABLE_BINARY_COLUMNS = frozenset(
    {
        "selected_prior_event_withdrawal",
        "opponent_prior_event_withdrawal",
        "selected_reliability_ok",
        "opponent_reliability_ok",
    }
)


def _decode_cascade_legacy_nullable_binary(
    series: pd.Series, *, column: str
) -> tuple[pd.Series, bool]:
    """Decode only the pinned cascade parquet's four legacy float flags."""

    if column not in CASCADE_LEGACY_NULLABLE_BINARY_COLUMNS:
        raise ValueError(f"undeclared cascade nullable-binary column: {column}")
    missing = series.isna()
    exact_bool = missing | series.map(
        lambda value: isinstance(value, (bool, np.bool_))
    )
    if exact_bool.all():
        # An all-null float column is still the legacy numeric parquet
        # serialization; without this dtype check one all-null flag would make
        # an otherwise uniform four-column repair look spuriously partial.
        repaired_all_null_numeric = bool(
            missing.all()
            and pd.api.types.is_numeric_dtype(series.dtype)
            and not pd.api.types.is_bool_dtype(series.dtype)
        )
        return (
            series.where(~missing, pd.NA).astype("boolean"),
            repaired_all_null_numeric,
        )
    numeric_type = missing | series.map(
        lambda value: (
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
        )
    )
    numeric = pd.to_numeric(series, errors="coerce")
    invalid = (
        ~numeric_type
        | (~missing & ~np.isfinite(numeric))
        | (~missing & ~numeric.isin([0.0, 1.0]))
    )
    if invalid.any():
        examples = series.loc[invalid].astype(str).unique()[:5].tolist()
        raise ValueError(
            f"cascade {column} must contain exact booleans or the pinned legacy "
            f"numeric 0/1/NA serialization; examples={examples}"
        )
    decoded = numeric.map({0.0: False, 1.0: True}).where(~missing, pd.NA)
    return decoded.astype("boolean"), True


def _require_nullable_nonnegative_integer_series(
    series: pd.Series, label: str
) -> pd.Series:
    missing = series.isna()
    numeric_type = series.map(
        lambda value: pd.isna(value)
        or (
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
        )
    )
    numeric = pd.to_numeric(series, errors="coerce")
    invalid = (
        ~numeric_type
        | (~missing & ~np.isfinite(numeric))
        | (~missing & numeric.lt(0.0))
        | (~missing & (numeric % 1.0).ne(0.0))
    )
    if invalid.any():
        examples = series.loc[invalid].astype(str).unique()[:5].tolist()
        raise ValueError(
            f"{label} must contain finite nonnegative integer counts or missing "
            f"values; examples={examples}"
        )
    return numeric


def _require_nonnegative_integer_series(
    series: pd.Series, label: str
) -> pd.Series:
    if series.isna().any():
        raise ValueError(f"{label} must not contain missing counts")
    return _require_nullable_nonnegative_integer_series(series, label).astype(int)


def _require_nullable_unit_interval_series(
    series: pd.Series, label: str
) -> pd.Series:
    missing = series.isna()
    numeric_type = series.map(
        lambda value: pd.isna(value)
        or (
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
        )
    )
    numeric = pd.to_numeric(series, errors="coerce")
    invalid = (
        ~numeric_type
        | (~missing & ~np.isfinite(numeric))
        | (~missing & ~numeric.between(0.0, 1.0, inclusive="both"))
    )
    if invalid.any():
        examples = series.loc[invalid].astype(str).unique()[:5].tolist()
        raise ValueError(
            f"{label} must contain finite coverage values in [0,1] or missing "
            f"values; examples={examples}"
        )
    return numeric


def _require_exact_nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{label} must be an exact nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{label} must be an exact nonnegative integer")
    return result


def _require_exact_string_menu(
    value: Any, expected: Sequence[str], label: str
) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be an explicit string list")
    observed = tuple(value)
    expected_tuple = tuple(expected)
    if observed != expected_tuple:
        raise ValueError(
            f"{label} differs from the code-frozen menu: "
            f"observed={observed}, expected={expected_tuple}"
        )
    return observed


def _require_finite_numeric_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    label: str,
    nonnegative: Sequence[str] = (),
) -> None:
    values = frame[list(columns)].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite numeric values")
    for column in nonnegative:
        if frame[column].lt(0.0).any():
            raise ValueError(f"{label} contains negative {column}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=path.name, suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=path.name, suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_csv(temp_name, index=False)
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=path.name, suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_parquet(temp_name, index=False)
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def validate_fresh_output_path(
    output: Path, *, research_root: Path = RESEARCH_ROOT
) -> Path:
    """Require a nonexistent strict descendant of the frozen research root."""

    resolved_root = research_root.resolve()
    resolved_output = output.resolve()
    if resolved_output == resolved_root or not _inside(resolved_root, resolved_output):
        raise ValueError(
            "weather audit output must be a strict descendant of the frozen "
            f"base-skill research root: {resolved_root}"
        )
    if resolved_output.exists():
        raise FileExistsError(
            "weather audit output must be a fresh, nonexistent directory: "
            f"{resolved_output}"
        )
    return resolved_output


def _execution_configuration(
    args: argparse.Namespace,
    *,
    output: Path,
    block_sizes: Sequence[int],
    argv: Sequence[str],
) -> dict[str, Any]:
    """Serialize every score-affecting CLI setting before any scoring starts."""

    parsed_args = {
        "sealed_dir": str(args.sealed_dir.resolve()),
        "cascade_dir": str(args.cascade_dir.resolve()),
        "round_panel": str(args.round_panel.resolve()),
        "panel_manifest": str(args.panel_manifest.resolve()),
        "output_dir": str(output),
        "bootstrap_draws": int(args.bootstrap_draws),
        "block_sizes": [int(value) for value in block_sizes],
        "seed": int(args.seed),
    }
    return {
        "contract": EXECUTION_CONFIG_CONTRACT,
        "complete": True,
        "research_only": True,
        "research_root": str(RESEARCH_ROOT.resolve()),
        "raw_argv": list(map(str, argv)),
        "parsed_args": parsed_args,
        "score_affecting_settings_bound_before_scoring": {
            "bootstrap_draws": int(args.bootstrap_draws),
            "block_sizes_events": [int(value) for value in block_sizes],
            "seed": int(args.seed),
        },
    }


def bind_execution_configuration(
    output: Path,
    args: argparse.Namespace,
    *,
    block_sizes: Sequence[int],
    argv: Sequence[str],
) -> tuple[Path, dict[str, Any]]:
    """Create the output and persist the immutable parsed execution contract."""

    output.mkdir(parents=True, exist_ok=False)
    payload = _execution_configuration(
        args, output=output, block_sizes=block_sizes, argv=argv
    )
    path = output / "execution_configuration.json"
    _atomic_json(path, payload)
    return path, payload


def _fingerprint_paths(records: Iterable[Mapping[str, Any]]) -> set[Path]:
    return {Path(str(record["path"])).resolve() for record in records}


def _require_provenance_record(
    record: Mapping[str, Any], *, label: str
) -> tuple[Path, str]:
    path = Path(str(record.get("path", ""))).resolve()
    expected_hash = str(record.get("sha256", "")).lower()
    if not path.is_file():
        raise FileNotFoundError(path)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError(f"{label} has malformed sha256")
    if _sha256(path) != expected_hash:
        raise ValueError(f"{label} hash mismatch")
    if "bytes" in record and int(record["bytes"]) != path.stat().st_size:
        raise ValueError(f"{label} byte count mismatch")
    if "mtime_ns" in record and int(record["mtime_ns"]) != path.stat().st_mtime_ns:
        raise ValueError(f"{label} mtime differs from completed provenance")
    return path, expected_hash


def validate_shared_execution_provenance(
    manifest: Mapping[str, Any],
    artifacts: Mapping[str, Path],
    *,
    label: str,
) -> tuple[dict[str, Any], tuple[Path, ...]]:
    """Verify a shared begin/finalize record and every file it hash-binds."""

    declared = manifest.get("execution_provenance")
    if not isinstance(declared, Mapping):
        raise ValueError(f"{label} manifest lacks execution provenance")
    relative = str(
        declared.get(
            "path", "execution_provenance/execution_provenance.json"
        )
    ).replace("\\", "/")
    provenance_path = _require_artifact(artifacts, relative, label)
    payload = _json(provenance_path)
    _require_exact_bool(payload.get("complete"), True, f"{label} execution complete")
    _require_exact_bool(
        payload.get("verified_unchanged_before_complete_manifest"),
        True,
        f"{label} execution verified unchanged",
    )
    _require_exact_bool(
        declared.get("verified_unchanged_before_complete_manifest"),
        True,
        f"{label} manifest execution verified unchanged",
    )
    # Cascade manifests embed the complete payload; sealed manifests retain a
    # compact pointer.  Any overlapping values must agree byte-for-byte.
    for field, value in declared.items():
        if field == "path":
            continue
        if field not in payload or payload[field] != value:
            raise ValueError(
                f"{label} execution provenance field {field!r} disagrees with artifact"
            )
    if "complete" in declared and dict(declared) != payload:
        raise ValueError(f"{label} embedded execution provenance is not exact")

    cli = payload.get("cli")
    runtime = payload.get("runtime")
    sources = payload.get("sources")
    preserved = payload.get("preserved_sources")
    inputs = payload.get("inputs")
    if not isinstance(cli, list) or not cli or not all(
        isinstance(value, str) for value in cli
    ):
        raise ValueError(f"{label} execution provenance lacks argv")
    for field in ("start_utc", "end_utc", "pid"):
        if payload.get(field) in (None, ""):
            raise ValueError(f"{label} execution provenance lacks {field}")
    if not isinstance(runtime, Mapping) or any(
        not runtime.get(key)
        for key in (
            "python",
            "python_executable",
            "platform",
            "numpy",
            "pandas",
            "scipy",
            "sklearn",
            "pyarrow",
        )
    ):
        raise ValueError(f"{label} execution provenance lacks runtime fingerprint")
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"{label} execution provenance lacks sources")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError(f"{label} execution provenance lacks inputs")
    if not isinstance(preserved, list) or len(preserved) != len(sources):
        raise ValueError(f"{label} execution provenance lacks preserved sources")

    verified: set[Path] = {provenance_path.resolve()}
    primary = Path(str(payload.get("primary_source", ""))).resolve()
    primary_matches = 0
    for index, (source_record, copy_record) in enumerate(zip(sources, preserved)):
        if not isinstance(source_record, Mapping) or not isinstance(copy_record, Mapping):
            raise ValueError(f"{label} execution source record is malformed")
        source, source_hash = _require_provenance_record(
            source_record, label=f"{label} execution source[{index}]"
        )
        copy, copy_hash = _require_provenance_record(
            copy_record, label=f"{label} preserved source[{index}]"
        )
        if source_hash != copy_hash or source.stat().st_size != copy.stat().st_size:
            raise ValueError(f"{label} preserved source copy differs from source")
        relative_copy = None
        for relative_path, artifact_path in artifacts.items():
            if artifact_path.resolve() == copy:
                relative_copy = relative_path
                break
        if relative_copy is None:
            raise ValueError(f"{label} preserved source is not a hashed artifact")
        verified.update((source, copy))
        primary_matches += int(source == primary)
    if primary_matches != 1:
        raise ValueError(f"{label} primary evaluator source is not uniquely bound")
    for index, input_record in enumerate(inputs):
        if not isinstance(input_record, Mapping):
            raise ValueError(f"{label} execution input record is malformed")
        path, _ = _require_provenance_record(
            input_record, label=f"{label} execution input[{index}]"
        )
        verified.add(path)
    return payload, tuple(sorted(verified, key=str))


def validate_execution_source_authority(
    execution: Mapping[str, Any],
    *,
    label: str,
    expected_sources: Mapping[Path, str],
    expected_primary: Path,
) -> dict[str, Any]:
    """Pin the exact reviewed primary and dependency source set."""

    primary = Path(str(execution.get("primary_source", ""))).resolve()
    expected_primary = expected_primary.resolve()
    if primary != expected_primary:
        raise ValueError(f"{label} execution primary source is not authoritative")
    records = execution.get("sources")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{label} execution source inventory is malformed")
    observed: dict[Path, str] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"{label} execution source[{index}] is malformed")
        path = Path(str(record.get("path", ""))).resolve()
        digest = str(record.get("sha256", "")).lower()
        if path in observed:
            raise ValueError(f"{label} execution source path is duplicated: {path}")
        observed[path] = digest
    expected = {Path(path).resolve(): str(digest).lower() for path, digest in expected_sources.items()}
    if observed != expected:
        missing = sorted(map(str, set(expected) - set(observed)))
        extra = sorted(map(str, set(observed) - set(expected)))
        mismatched = sorted(
            str(path)
            for path in set(observed).intersection(expected)
            if observed[path] != expected[path]
        )
        raise ValueError(
            f"{label} execution source authority differs; missing={missing}, "
            f"extra={extra}, hash_mismatch={mismatched}"
        )
    if expected_primary not in expected:
        raise ValueError(f"{label} source pins omit the primary evaluator")
    return {
        "primary_source": str(expected_primary),
        "primary_source_sha256": expected[expected_primary],
        "dependency_source_count": int(len(expected) - 1),
        "exact_source_set_verified": True,
    }


def _verify_manifest_inputs_in_execution(
    manifest: Mapping[str, Any], execution: Mapping[str, Any], *, label: str
) -> tuple[Path, ...]:
    records = execution.get("inputs")
    if not isinstance(records, list):
        raise ValueError(f"{label} execution inputs are malformed")
    execution_hashes = {
        Path(str(record.get("path", ""))).resolve(): str(record.get("sha256", ""))
        for record in records
        if isinstance(record, Mapping)
    }
    paths: list[Path] = []
    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping) or not inputs:
        raise ValueError(f"{label} manifest lacks hashed inputs")
    for name, entry in inputs.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"{label} manifest input {name!r} is malformed")
        path = _validate_input_entry(entry, str(name))
        if execution_hashes.get(path) != str(entry.get("sha256", "")):
            raise ValueError(
                f"{label} manifest input {name!r} is absent from execution provenance"
            )
        paths.append(path)
    return tuple(paths)


def validate_artifact_manifest(
    root: Path, *, label: str, contract: str | None = None
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Verify every declared output byte and reject undeclared additions."""

    root = root.resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = _json(manifest_path)
    _require_exact_bool(manifest.get("complete"), True, f"{label}.complete")
    _require_exact_bool(
        manifest.get("research_only"), True, f"{label}.research_only"
    )
    if contract is not None and manifest.get("contract") != contract:
        raise ValueError(
            f"{label} contract mismatch: {manifest.get('contract')!r} != {contract!r}"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"{label} manifest lacks a nonempty artifact inventory")
    mapping: dict[str, Path] = {}
    for entry in artifacts:
        if not isinstance(entry, Mapping):
            raise ValueError(f"{label} artifact entry is not an object")
        relative = str(entry.get("path", "")).replace("\\", "/")
        candidate = (root / relative).resolve()
        if (
            not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not _inside(root, candidate)
        ):
            raise ValueError(f"{label} artifact path escapes output root: {relative!r}")
        if relative in mapping:
            raise ValueError(f"{label} artifact path is duplicated: {relative}")
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        expected_bytes = entry.get("bytes")
        if not isinstance(expected_bytes, int) or candidate.stat().st_size != expected_bytes:
            raise ValueError(f"{label} artifact byte count changed: {relative}")
        expected_hash = str(entry.get("sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise ValueError(f"{label} artifact hash is malformed: {relative}")
        if _sha256(candidate) != expected_hash:
            raise ValueError(f"{label} artifact hash mismatch: {relative}")
        mapping[relative] = candidate
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    declared = set(mapping)
    if actual != declared:
        raise ValueError(
            f"{label} artifact inventory is not exact; "
            f"undeclared={sorted(actual - declared)[:10]}, "
            f"missing={sorted(declared - actual)[:10]}"
        )
    return manifest, mapping


def _require_artifact(
    artifacts: Mapping[str, Path], relative: str, label: str
) -> Path:
    if relative not in artifacts:
        raise ValueError(f"{label} manifest does not declare {relative}")
    return artifacts[relative]


def validate_panel_manifest(
    panel_path: Path, panel_manifest_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    panel_path = panel_path.resolve()
    panel_manifest_path = panel_manifest_path.resolve()
    if not panel_path.is_file() or not panel_manifest_path.is_file():
        raise FileNotFoundError(panel_path if not panel_path.is_file() else panel_manifest_path)
    if panel_manifest_path != PANEL_MANIFEST_AUTHORITY_PATH.resolve():
        raise ValueError("round-panel manifest path is not the frozen research authority")
    if panel_path != PANEL_ROUND_AUTHORITY_PATH.resolve():
        raise ValueError("round-panel path is not the frozen research authority")
    if (
        panel_manifest_path.stat().st_size != PANEL_MANIFEST_BYTES
        or _sha256(panel_manifest_path) != PANEL_MANIFEST_SHA256
    ):
        raise ValueError("round-panel manifest is not the frozen research authority")
    manifest = _json(panel_manifest_path)
    _require_exact_bool(
        manifest.get("production_changes"), False, "panel.production_changes"
    )
    purpose = str(manifest.get("purpose", ""))
    if "research-only" not in purpose.casefold():
        raise ValueError("panel manifest is not explicitly research-only")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("panel manifest lacks outputs")
    for name, pin in PANEL_OUTPUT_PINS.items():
        entry = outputs.get(name)
        if not isinstance(entry, Mapping):
            raise ValueError(f"panel manifest lacks frozen output {name!r}")
        if (
            str(entry.get("sha256", "")) != pin["sha256"]
            or int(entry.get("size_bytes", entry.get("bytes", -1))) != pin["bytes"]
        ):
            raise ValueError(f"panel manifest output pin differs for {name!r}")
        declared = Path(str(entry.get("path", ""))).resolve()
        if not declared.is_file():
            raise FileNotFoundError(declared)
        if declared.stat().st_size != pin["bytes"] or _sha256(declared) != pin["sha256"]:
            raise ValueError(f"frozen panel output changed for {name!r}")
    output = outputs.get("player_round_panel")
    if not isinstance(output, Mapping):
        raise ValueError("panel manifest lacks outputs.player_round_panel")
    declared_path = Path(str(output.get("path", ""))).resolve()
    if declared_path != panel_path:
        raise ValueError("round-panel path differs from its manifest")
    panel_hash = _sha256(panel_path)
    if panel_hash != str(output.get("sha256", "")):
        raise ValueError("round-panel hash differs from its manifest")
    if "bytes" in output and int(output["bytes"]) != panel_path.stat().st_size:
        raise ValueError("round-panel byte count differs from its manifest")
    if "size_bytes" in output and int(output["size_bytes"]) != panel_path.stat().st_size:
        raise ValueError("round-panel byte count differs from its manifest")
    return manifest, {
        "panel_path": str(panel_path),
        "panel_sha256": panel_hash,
        "panel_manifest_path": str(panel_manifest_path),
        "panel_manifest_sha256": _sha256(panel_manifest_path),
    }


def validate_sealed_selection_cutoff(manifest: Mapping[str, Any]) -> pd.Timestamp:
    cutoff = pd.to_datetime(
        manifest.get("selection_window_end"), errors="coerce", utc=True
    )
    if pd.isna(cutoff) or cutoff >= SEALED_START.tz_localize("UTC"):
        raise ValueError(
            "sealed.selection_window_end must parse and precede 2023-01-01"
        )
    return pd.Timestamp(cutoff)


def validate_sealed_provenance(
    root: Path, panel_sha256: str
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Path],
    tuple[Path, ...],
]:
    manifest, artifacts = validate_artifact_manifest(
        root, label="sealed", contract=SEALED_CONTRACT
    )
    for key in (
        "sealed_data_used_for_selection",
        "sealed_results_used_for_selection",
        "uses_closing_odds_as_feature",
    ):
        _require_exact_bool(manifest.get(key), False, f"sealed.{key}")
    _require_exact_bool(
        manifest.get("frozen_before_sealed_scoring"),
        True,
        "sealed.frozen_before_sealed_scoring",
    )
    validate_sealed_selection_cutoff(manifest)
    if manifest.get("root_panel_manifest_sha256") != PANEL_MANIFEST_SHA256:
        raise ValueError("sealed root-panel authority differs from the frozen manifest")
    if manifest.get("round_panel_sha256") != panel_sha256:
        raise ValueError("sealed round-panel authority differs from weather attribution")
    if manifest.get("event_crosswalk_sha256") != PANEL_OUTPUT_PINS[
        "odds_event_crosswalk"
    ]["sha256"]:
        raise ValueError("sealed event-crosswalk authority differs from the frozen panel")
    execution, execution_paths = validate_shared_execution_provenance(
        manifest, artifacts, label="sealed"
    )
    validate_execution_source_authority(
        execution,
        label="sealed",
        expected_sources=SEALED_EVALUATOR_SOURCE_PINS,
        expected_primary=SEALED_EVALUATOR_PRIMARY_SOURCE,
    )
    audit_path = _require_artifact(artifacts, "audit.json", "sealed")
    audit = _json(audit_path)
    _require_exact_bool(audit.get("research_only"), True, "sealed audit research_only")
    _require_exact_bool(
        audit.get("production_files_changed"),
        False,
        "sealed audit production_files_changed",
    )
    for key in (
        "sealed_candidate_selection_performed",
        "sealed_results_can_promote_or_reselect",
        "closing_odds_used_as_features",
        "probability_mapper_refit_on_sealed",
    ):
        _require_exact_bool(audit.get(key), False, f"sealed audit {key}")
    weather = audit.get("weather_attribution")
    if not isinstance(weather, Mapping):
        raise ValueError("sealed audit lacks weather attribution provenance")
    if str(weather.get("round_panel_sha256", "")) != panel_sha256:
        raise ValueError("sealed weather attribution used a different round panel")
    _require_exact_bool(weather.get("uses_outcomes"), False, "sealed weather uses_outcomes")
    _require_exact_bool(
        weather.get("uses_weather_excluded"),
        False,
        "sealed weather uses_weather_excluded",
    )
    panel_authority = audit.get("panel_authority")
    if not isinstance(panel_authority, Mapping):
        raise ValueError("sealed audit lacks root-panel authority")
    for key, expected in (
        ("round_panel_sha256", panel_sha256),
        (
            "event_crosswalk_sha256",
            PANEL_OUTPUT_PINS["odds_event_crosswalk"]["sha256"],
        ),
        ("root_panel_manifest_sha256", PANEL_MANIFEST_SHA256),
    ):
        if panel_authority.get(key) != expected:
            raise ValueError(f"sealed panel authority differs at {key!r}")
    return manifest, audit, artifacts, execution_paths


def _validate_input_entry(entry: Mapping[str, Any], label: str) -> Path:
    path = Path(str(entry.get("path", ""))).resolve()
    expected = str(entry.get("sha256", ""))
    if not path.is_file():
        raise FileNotFoundError(path)
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or _sha256(path) != expected:
        raise ValueError(f"cascade input hash mismatch: {label}")
    return path


def validate_cascade_completion_authority(
    manifest: Mapping[str, Any], handoff: frozen_handoff.FrozenHandoff
) -> dict[str, dict[str, str]]:
    """Require cascade state inputs to equal the frozen completion authorities."""

    manifest_inputs = manifest.get("inputs")
    authority = manifest.get("completion_authority")
    handoff_bindings = handoff.audit.get("completion_input_bindings")
    if not isinstance(manifest_inputs, Mapping):
        raise ValueError("cascade manifest inputs are malformed")
    if not isinstance(authority, Mapping) or set(authority) != {
        "round_panel",
        "weather_coefficients",
        "core_bridge",
    }:
        raise ValueError("cascade completion-authority menu is not exact")
    if not isinstance(handoff_bindings, Mapping):
        raise ValueError("frozen handoff lacks completion input bindings")
    manifest_names = {
        "round_panel": "player_round_panel",
        "weather_coefficients": "weather_coefficients",
        "core_bridge": "core_bridge",
    }
    verified: dict[str, dict[str, str]] = {}
    for name, manifest_name in manifest_names.items():
        records = (
            authority.get(name),
            handoff_bindings.get(name),
            manifest_inputs.get(manifest_name),
        )
        if not all(isinstance(record, Mapping) for record in records):
            raise ValueError(f"cascade completion authority lacks {name!r}")
        normalized = []
        for index, record in enumerate(records):
            assert isinstance(record, Mapping)
            path = Path(str(record.get("path", ""))).resolve()
            digest = str(record.get("sha256", "")).lower()
            if not path.is_file() or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(
                    f"cascade completion authority {name!r}[{index}] is invalid"
                )
            if _sha256(path) != digest:
                raise ValueError(f"cascade completion authority hash changed: {name}")
            normalized.append((path, digest))
        if len(set(normalized)) != 1:
            raise ValueError(
                f"cascade manifest/handoff completion authority differs for {name!r}"
            )
        path, digest = normalized[0]
        verified[name] = {"path": str(path), "sha256": digest}
    return verified


def _parse_declared_dates(
    series: pd.Series,
    *,
    label: str,
    required: pd.Series | None = None,
) -> pd.Series:
    present = series.notna() & series.astype(str).str.strip().ne("")
    parsed = pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(
        None
    ).dt.normalize()
    if (present & parsed.isna()).any():
        examples = series.loc[present & parsed.isna()].astype(str).unique()[:5].tolist()
        raise ValueError(f"{label} contains unparseable dates: {examples}")
    if required is not None:
        required_bool = required.fillna(False).astype(bool)
        if parsed.loc[required_bool].isna().any():
            raise ValueError(f"{label} is missing on rows where it is required")
    return parsed


def _validated_cascade_pricing_audit(
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    payload = audit.get("pricing_probability_algebra")
    if not isinstance(payload, Mapping) or set(payload) != set(
        CASCADE_PRICING_AUDIT_FIELDS
    ):
        raise ValueError(
            "cascade pricing_probability_algebra field menu is not producer-exact"
        )
    result = dict(payload)
    rows = _require_exact_nonnegative_integer(
        result["rows"], "cascade pricing_probability_algebra.rows"
    )
    ties_lose = _require_exact_nonnegative_integer(
        result["ties_lose_rows"],
        "cascade pricing_probability_algebra.ties_lose_rows",
    )
    push_on_tie = _require_exact_nonnegative_integer(
        result["push_on_tie_rows"],
        "cascade pricing_probability_algebra.push_on_tie_rows",
    )
    zero_push = _require_exact_nonnegative_integer(
        result["zero_push_rows"],
        "cascade pricing_probability_algebra.zero_push_rows",
    )
    if rows < 1 or ties_lose + push_on_tie != rows or zero_push > rows:
        raise ValueError("cascade pricing_probability_algebra row counts are false")
    for key in (
        "simplex_max_abs_error",
        "pricing_identity_max_abs_error",
        "zero_push_identity_max_abs_error",
    ):
        value = result[key]
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"cascade pricing_probability_algebra.{key} is invalid")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"cascade pricing_probability_algebra.{key} is invalid"
            ) from exc
        if not np.isfinite(numeric) or numeric < 0.0 or numeric > 1e-12:
            raise ValueError(
                f"cascade pricing_probability_algebra.{key} must be finite and <= 1e-12"
            )
        result[key] = numeric
    for key, expected in CASCADE_PRICING_IDENTITIES.items():
        if result.get(key) != expected:
            raise ValueError(
                f"cascade pricing_probability_algebra.{key} is not authoritative"
            )
    return result


def validate_cascade_critical_audit_gate(
    audit: Mapping[str, Any],
    artifacts: Mapping[str, Path],
    *,
    discovery_cutoff: pd.Timestamp | str = frozen_handoff.DISCOVERY_END,
) -> dict[str, Any]:
    """Reproduce chronology/critical audit claims before probabilities are read."""

    cutoff = pd.to_datetime(discovery_cutoff, errors="coerce")
    if pd.isna(cutoff):
        raise ValueError("cascade discovery cutoff is invalid")
    cutoff = pd.Timestamp(cutoff).normalize()
    if cutoff >= SEALED_START:
        raise ValueError("cascade discovery cutoff reaches the sealed window")

    declared_zero: dict[str, int] = {}
    for key in (
        "c3_training_boundary_violations",
        "live_lambda_selection_after_discovery_rows",
        "compatibility_quote_menu_drops",
        "probability_sigma_upper_boundary_hits",
    ):
        value = _require_exact_nonnegative_integer(
            audit.get(key), f"cascade audit {key}"
        )
        if value != 0:
            raise ValueError(f"cascade audit {key} must be zero")
        declared_zero[key] = value

    pricing = _validated_cascade_pricing_audit(audit)
    for key in (
        "probability_sum_max_abs_error",
        "reverse_pair_symmetry_max_abs_error",
    ):
        value = audit.get(key)
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"cascade audit {key} is invalid")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"cascade audit {key} is invalid") from exc
        if not np.isfinite(numeric) or numeric < 0.0 or numeric > 1e-12:
            raise ValueError(f"cascade audit {key} must be finite and <= 1e-12")

    player_path = _require_artifact(
        artifacts, "cascade_player_round_predictions.parquet", "cascade"
    )
    player = pd.read_parquet(player_path)
    _require(
        player,
        [
            "model",
            "event_key",
            "dg_id_key",
            "round_num",
            "variant",
            "event_start_date",
            "training_max_event_end",
            "prediction_year",
        ],
        "cascade player prediction chronology",
    )
    duplicate_count = int(
        player.duplicated(["model", "event_key", "dg_id_key", "round_num"]).sum()
    )
    declared_duplicates = _require_exact_nonnegative_integer(
        audit.get("duplicate_player_prediction_keys"),
        "cascade audit duplicate_player_prediction_keys",
    )
    if declared_duplicates != duplicate_count or duplicate_count != 0:
        raise ValueError(
            "cascade duplicate-player-prediction audit differs from its artifact"
        )
    c3 = player["variant"].astype(str).str.startswith("C3_")
    if not c3.any():
        raise ValueError("cascade player predictions contain no C3 rows")
    event_dates = _parse_declared_dates(
        player["event_start_date"],
        label="cascade event_start_date",
        required=c3,
    )
    training_dates = _parse_declared_dates(
        player["training_max_event_end"],
        label="cascade training_max_event_end",
        required=c3,
    )
    prediction_year = pd.to_numeric(player["prediction_year"], errors="coerce")
    if prediction_year.loc[c3].isna().any() or (
        prediction_year.loc[c3] % 1.0
    ).ne(0.0).any():
        raise ValueError("cascade C3 prediction_year is missing or nonintegral")
    if not prediction_year.loc[c3].astype(int).eq(
        event_dates.loc[c3].dt.year.astype(int)
    ).all():
        raise ValueError(
            "cascade C3 prediction_year does not equal event_start_date year"
        )
    prediction_cutoff = pd.to_datetime(
        prediction_year.astype("Int64").astype(str) + "-01-01", errors="coerce"
    )
    boundary_bad = c3 & training_dates.notna() & (
        prediction_cutoff.isna() | training_dates.ge(prediction_cutoff)
    )
    boundary_count = int(boundary_bad.sum())
    if (
        boundary_count != declared_zero["c3_training_boundary_violations"]
        or boundary_count != 0
    ):
        raise ValueError(
            "cascade c3_training_boundary_violations label differs from its artifact"
        )

    tuning = pd.read_csv(
        _require_artifact(
            artifacts, "c3_discovery_hyperparameter_scores.csv", "cascade"
        )
    )
    _require(
        tuning,
        ["live_update_lambda", "selection_max_event_date"],
        "cascade discovery hyperparameter chronology",
    )
    lambda_raw = tuning["live_update_lambda"]
    lambda_present = lambda_raw.notna() & lambda_raw.astype(str).str.strip().ne("")
    lambda_numeric = pd.to_numeric(lambda_raw, errors="coerce")
    if (lambda_present & (~np.isfinite(lambda_numeric))).any() or not lambda_present.any():
        raise ValueError("cascade live-update tuning rows are missing or malformed")
    selection_dates = _parse_declared_dates(
        tuning["selection_max_event_date"],
        label="cascade selection_max_event_date",
        required=lambda_present,
    )
    selection_after = int(selection_dates.loc[lambda_present].gt(cutoff).sum())
    if (
        selection_after
        != declared_zero["live_lambda_selection_after_discovery_rows"]
        or selection_after != 0
    ):
        raise ValueError(
            "cascade live_lambda_selection_after_discovery_rows label differs "
            "from its artifact"
        )

    coverage = pd.read_csv(
        _require_artifact(artifacts, "model_quote_coverage.csv", "cascade")
    )
    _require(
        coverage,
        ["variant", "c0_quotes_dropped"],
        "cascade compatibility quote coverage",
    )
    drops = pd.to_numeric(coverage["c0_quotes_dropped"], errors="coerce")
    if (
        not np.isfinite(drops).all()
        or drops.lt(0).any()
        or (drops % 1.0).ne(0.0).any()
    ):
        raise ValueError("cascade c0_quotes_dropped is malformed")
    primary = coverage["variant"].astype(str).isin(
        ["C1", "C2_NO_LIVE_UPDATE", "C2_DISCOVERY_LIVE_UPDATE_SHRINK"]
    ) | coverage["variant"].astype(str).str.startswith(
        "C3_pt_continuous_partial_pool_"
    )
    if not primary.any():
        raise ValueError("cascade compatibility quote audit has no primary arms")
    compatibility_drops = int(drops.loc[primary].sum())
    if (
        compatibility_drops
        != declared_zero["compatibility_quote_menu_drops"]
        or compatibility_drops != 0
    ):
        raise ValueError(
            "cascade compatibility_quote_menu_drops label differs from its artifact"
        )

    sigmas = pd.read_csv(
        _require_artifact(artifacts, "discovery_probability_sigmas.csv", "cascade")
    )
    _require(
        sigmas,
        ["sigma", "discovery_max_event_date"],
        "cascade discovery probability chronology",
    )
    sigma_raw = sigmas["sigma"]
    sigma_present = sigma_raw.notna() & sigma_raw.astype(str).str.strip().ne("")
    sigma = pd.to_numeric(sigma_raw, errors="coerce")
    finite_sigma = sigma_present & np.isfinite(sigma)
    if (
        (sigma_present & ~finite_sigma).any()
        or not finite_sigma.any()
        or sigma.loc[finite_sigma].le(0.0).any()
    ):
        raise ValueError("cascade discovery probability sigma is malformed")
    discovery_dates = _parse_declared_dates(
        sigmas["discovery_max_event_date"],
        label="cascade discovery_max_event_date",
        required=finite_sigma,
    )
    if discovery_dates.dropna().gt(cutoff).any():
        raise ValueError("cascade discovery_max_event_date exceeds discovery cutoff")
    sigma_upper_hits = int(sigma.loc[finite_sigma].ge(12.0 - 1e-6).sum())
    if (
        sigma_upper_hits
        != declared_zero["probability_sigma_upper_boundary_hits"]
        or sigma_upper_hits != 0
    ):
        raise ValueError(
            "cascade probability_sigma_upper_boundary_hits label differs from its artifact"
        )

    return {
        "contract": "cascade_critical_chronology_audit_v1",
        "discovery_cutoff": cutoff.date().isoformat(),
        "c3_training_boundary_violations": boundary_count,
        "live_lambda_selection_after_discovery_rows": selection_after,
        "compatibility_quote_menu_drops": compatibility_drops,
        "probability_sigma_upper_boundary_hits": sigma_upper_hits,
        "pricing_probability_algebra": pricing,
        "selection_max_event_date": selection_dates.loc[lambda_present]
        .max()
        .date()
        .isoformat(),
        "discovery_max_event_date": discovery_dates.dropna()
        .max()
        .date()
        .isoformat(),
        "artifact_claims_independently_reproduced": True,
    }


def validate_cascade_probability_audit_consistency(
    frame: pd.DataFrame, audit: Mapping[str, Any]
) -> dict[str, Any]:
    """Match the producer pricing audit to the hash-bound probability rows."""

    declared = _validated_cascade_pricing_audit(audit)
    required = [
        "p1_probability",
        "p2_probability",
        "push_probability",
        "pricing_p1_probability",
        "tie_rule_ties_lose",
    ]
    _require(frame, required, "cascade probability audit consistency")
    if frame.empty:
        raise ValueError("cascade probability audit consistency has no rows")
    numeric = frame[required[:4]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("cascade probability audit consistency is nonfinite")
    ties_lose = _require_exact_boolean_series(
        frame["tie_rule_ties_lose"],
        "cascade probability audit consistency tie_rule_ties_lose",
    )
    simplex = (
        numeric["p1_probability"]
        + numeric["p2_probability"]
        + numeric["push_probability"]
        - 1.0
    ).abs()
    nonpush = 1.0 - numeric["push_probability"]
    if ((~ties_lose) & nonpush.le(0.0)).any():
        raise ValueError("cascade probability audit has zero resolved mass")
    expected = numeric["p1_probability"].where(
        ties_lose, numeric["p1_probability"] / nonpush
    )
    pricing_error = (numeric["pricing_p1_probability"] - expected).abs()
    zero_push = numeric["push_probability"].abs().le(1e-15)
    zero_push_error = (
        numeric.loc[zero_push, "pricing_p1_probability"]
        - numeric.loc[zero_push, "p1_probability"]
    ).abs()
    observed = {
        "rows": int(len(frame)),
        "ties_lose_rows": int(ties_lose.sum()),
        "push_on_tie_rows": int((~ties_lose).sum()),
        "zero_push_rows": int(zero_push.sum()),
        "simplex_max_abs_error": float(simplex.max()),
        "pricing_identity_max_abs_error": float(pricing_error.max()),
        "zero_push_identity_max_abs_error": float(
            zero_push_error.max() if len(zero_push_error) else 0.0
        ),
    }
    for key in ("rows", "ties_lose_rows", "push_on_tie_rows", "zero_push_rows"):
        if observed[key] != declared[key]:
            raise ValueError(
                f"cascade pricing_probability_algebra.{key} differs from probability artifact"
            )
    for key in (
        "simplex_max_abs_error",
        "pricing_identity_max_abs_error",
        "zero_push_identity_max_abs_error",
    ):
        if abs(float(observed[key]) - float(declared[key])) > 1e-15:
            raise ValueError(
                f"cascade pricing_probability_algebra.{key} differs from probability artifact"
            )
    declared_sum = float(audit.get("probability_sum_max_abs_error", np.nan))
    if not np.isfinite(declared_sum) or abs(
        declared_sum - observed["simplex_max_abs_error"]
    ) > 1e-15:
        raise ValueError(
            "cascade probability_sum_max_abs_error differs from probability artifact"
        )
    return {**observed, "producer_audit_exact": True}


def _validate_cascade_score_void_classification_subset_legacy(
    raw_probabilities: pd.DataFrame,
    normalized_quotes: pd.DataFrame,
    audit: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Reproduce quote-level score/void eligibility and its producer audit."""

    identity = [
        "quote_id",
        "event_key",
        "event_start_date",
        "round_num",
        "pair_key",
        "book",
        "p1_id",
        "p2_id",
        "p1_outcome",
        "p2_outcome",
        "tie_rule",
    ]
    classification_bool = [
        "both_target_round_scores_valid",
        "tie_rule_ties_lose",
        "archived_outcome_pair_possible",
        "p1_archive_outcome_matches_score",
        "p2_archive_outcome_matches_score",
        "archive_outcome_matches_score",
        "actual_score_tie",
        "score_tie_void",
        "score_tie_loss",
        "no_score_void",
        "other_void_with_scores",
        "score_model_eligible",
        "market_devig_proper_score_eligible",
        "binary_model_eligible",
    ]
    classification_numeric = [
        "p1_round_score",
        "p2_round_score",
        "derived_score_outcome",
        "p1_derived_score_outcome",
        "p2_derived_score_outcome",
        "settlement_score_outcome",
        "p1_settlement_score_outcome",
        "p2_settlement_score_outcome",
    ]
    required = [
        "model",
        *identity,
        *classification_bool,
        *classification_numeric,
    ]
    _require(raw_probabilities, required, "cascade score/void classification")
    _require(normalized_quotes, identity, "cascade normalized score/void authority")

    raw = raw_probabilities[required].copy()
    quotes = normalized_quotes[identity].copy()
    for frame in (raw, quotes):
        frame["event_start_date"] = pd.to_datetime(
            frame["event_start_date"], errors="coerce"
        ).dt.normalize()
        frame["round_num"] = pd.to_numeric(frame["round_num"], errors="coerce")
        frame["quote_id"] = frame["quote_id"].astype(str)
        frame["book"] = frame["book"].astype(str).str.strip().str.casefold()
        frame["event_key"] = frame["event_key"].astype(str)
        frame["pair_key"] = frame["pair_key"].astype(str)
        frame["p1_id"] = frame["p1_id"].astype(str)
        frame["p2_id"] = frame["p2_id"].astype(str)
    if (
        quotes.empty
        or quotes[identity[:-3]].isna().any().any()
        or quotes["quote_id"].duplicated().any()
    ):
        raise ValueError("cascade normalized score/void quote authority is ambiguous")
    control = raw.loc[raw["model"].astype(str).eq(CASCADE_CONTROL)].copy()
    if control.empty or control["quote_id"].duplicated().any():
        raise ValueError("cascade C0 score/void quote authority is incomplete or ambiguous")
    quote_ids = set(quotes["quote_id"])
    if set(control["quote_id"]) != quote_ids:
        raise ValueError(
            "cascade C0 score/void menu differs from normalized quote authority"
        )

    compare_columns = [
        "event_key",
        "event_start_date",
        "round_num",
        "pair_key",
        "book",
        "p1_id",
        "p2_id",
        "p1_outcome",
        "p2_outcome",
        "tie_rule",
    ]
    authority_check = quotes.merge(
        control[["quote_id", *compare_columns]],
        on="quote_id",
        how="outer",
        validate="one_to_one",
        suffixes=("_quote", "_probability"),
        indicator=True,
    )
    if not authority_check["_merge"].eq("both").all():
        raise ValueError("cascade score/void quote identities are incomplete")
    for column in compare_columns:
        left = authority_check[f"{column}_quote"]
        right = authority_check[f"{column}_probability"]
        if column in {"p1_outcome", "p2_outcome", "round_num"}:
            equal = np.isclose(
                pd.to_numeric(left, errors="coerce"),
                pd.to_numeric(right, errors="coerce"),
                rtol=0.0,
                atol=1e-10,
                equal_nan=True,
            )
            if not bool(np.asarray(equal).all()):
                raise ValueError(
                    f"cascade score/void quote authority differs at {column!r}"
                )
        elif not left.fillna("__missing__").astype(str).eq(
            right.fillna("__missing__").astype(str)
        ).all():
            raise ValueError(
                f"cascade score/void quote authority differs at {column!r}"
            )

    # Every model copy must preserve the quote classification emitted before
    # model probabilities were attached.
    invariant = [*identity[1:], *classification_bool, *classification_numeric]
    disagreement = (
        raw.groupby("quote_id", observed=True, dropna=False)[invariant]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    if disagreement.any():
        raise ValueError("cascade score/void classification differs across model copies")

    quote = control.copy()
    for column in classification_bool:
        quote[column] = _require_exact_boolean_series(
            quote[column], f"cascade score/void {column}"
        )
    for column in classification_numeric:
        raw_present = quote[column].notna() & quote[column].astype(str).str.strip().ne("")
        numeric = pd.to_numeric(quote[column], errors="coerce")
        if (raw_present & numeric.isna()).any() or np.isinf(
            numeric.fillna(0.0).to_numpy(float)
        ).any():
            raise ValueError(f"cascade score/void {column} is malformed")
        quote[column] = numeric
    for column in ("p1_outcome", "p2_outcome"):
        quote[column] = pd.to_numeric(quote[column], errors="coerce")
        if not quote[column].isin([0.0, 0.5, 1.0]).all():
            raise ValueError(f"cascade score/void {column} is outside {{0,.5,1}}")

    p1_score = quote["p1_round_score"]
    p2_score = quote["p2_round_score"]
    p1_archived = quote["p1_outcome"]
    p2_archived = quote["p2_outcome"]
    both_valid = p1_score.notna() & p2_score.notna()
    p1_derived = pd.Series(
        np.where(
            p1_score.lt(p2_score),
            1.0,
            np.where(p1_score.gt(p2_score), 0.0, 0.5),
        ),
        index=quote.index,
    )
    p2_derived = pd.Series(
        np.where(
            p2_score.lt(p1_score),
            1.0,
            np.where(p2_score.gt(p1_score), 0.0, 0.5),
        ),
        index=quote.index,
    )
    normalized_tie = quote["tie_rule"].fillna("").astype(str).str.strip()
    normalized_tie = normalized_tie.str.casefold().str.replace(
        r"[_-]", " ", regex=True
    )
    ties_lose = (
        normalized_tie.str.contains("separate", regex=False)
        | normalized_tie.str.contains("ties lose", regex=False)
        | normalized_tie.str.contains("tie loses", regex=False)
        | normalized_tie.str.contains("three way", regex=False)
        | normalized_tie.str.contains("3 way", regex=False)
    )
    physical_tie = both_valid & p1_score.eq(p2_score)
    p1_settlement = p1_derived.where(~(physical_tie & ties_lose), 0.0)
    p2_settlement = p2_derived.where(~(physical_tie & ties_lose), 0.0)
    p1_matches = pd.Series(
        np.isclose(
            p1_archived,
            p1_settlement,
            rtol=0.0,
            atol=1e-10,
            equal_nan=False,
        ),
        index=quote.index,
    )
    p2_matches = pd.Series(
        np.isclose(
            p2_archived,
            p2_settlement,
            rtol=0.0,
            atol=1e-10,
            equal_nan=False,
        ),
        index=quote.index,
    )
    possible = (
        (p1_archived.eq(1.0) & p2_archived.eq(0.0))
        | (p1_archived.eq(0.0) & p2_archived.eq(1.0))
        | (p1_archived.eq(0.5) & p2_archived.eq(0.5) & ~ties_lose)
        | (p1_archived.eq(0.0) & p2_archived.eq(0.0) & ties_lose)
    )
    archive_match = both_valid & possible & p1_matches & p2_matches
    expected_bool = {
        "both_target_round_scores_valid": both_valid,
        "tie_rule_ties_lose": ties_lose,
        "archived_outcome_pair_possible": possible,
        "p1_archive_outcome_matches_score": both_valid & p1_matches,
        "p2_archive_outcome_matches_score": both_valid & p2_matches,
        "archive_outcome_matches_score": archive_match,
        "actual_score_tie": physical_tie,
        "score_tie_void": (
            p1_archived.eq(0.5)
            & p2_archived.eq(0.5)
            & physical_tie
            & ~ties_lose
        ),
        "score_tie_loss": (
            p1_archived.eq(0.0)
            & p2_archived.eq(0.0)
            & physical_tie
            & ties_lose
        ),
        "no_score_void": (
            p1_archived.eq(0.5) & p2_archived.eq(0.5) & ~both_valid
        ),
        "other_void_with_scores": (
            (p1_archived.eq(0.5) | p2_archived.eq(0.5))
            & both_valid
            & ~physical_tie
        ),
        "score_model_eligible": archive_match,
        "market_devig_proper_score_eligible": (
            archive_match & ~physical_tie & ~ties_lose
        ),
        "binary_model_eligible": archive_match & ~physical_tie & ~ties_lose,
    }
    for column, expected in expected_bool.items():
        if not quote[column].eq(expected).all():
            raise ValueError(
                f"cascade score/void classification is false at {column!r}"
            )
    expected_numeric = {
        "derived_score_outcome": p1_derived.where(both_valid, np.nan),
        "p1_derived_score_outcome": p1_derived.where(both_valid, np.nan),
        "p2_derived_score_outcome": p2_derived.where(both_valid, np.nan),
        "settlement_score_outcome": p1_settlement.where(both_valid, np.nan),
        "p1_settlement_score_outcome": p1_settlement.where(both_valid, np.nan),
        "p2_settlement_score_outcome": p2_settlement.where(both_valid, np.nan),
    }
    for column, expected in expected_numeric.items():
        if not np.isclose(
            quote[column].to_numpy(float),
            expected.to_numpy(float),
            rtol=0.0,
            atol=1e-10,
            equal_nan=True,
        ).all():
            raise ValueError(
                f"cascade score/void classification is false at {column!r}"
            )

    archived_void = p1_archived.eq(0.5) & p2_archived.eq(0.5)
    graded_with_scores = (
        p1_archived.isin([0.0, 0.5, 1.0])
        & p2_archived.isin([0.0, 0.5, 1.0])
        & both_valid
    )
    mismatch = graded_with_scores & ~archive_match
    quote["round_market"] = "r" + quote["round_num"].astype(int).astype(str)
    rows_by_round = (
        quote.groupby("round_market", observed=True)
        .agg(
            quotes=("quote_id", "size"),
            score_tie_voids=("score_tie_void", "sum"),
            score_tie_losses=("score_tie_loss", "sum"),
            no_score_voids=("no_score_void", "sum"),
            score_model_eligible=("score_model_eligible", "sum"),
            market_devig_proper_score_eligible=(
                "market_devig_proper_score_eligible",
                "sum",
            ),
            ties_lose_market_benchmark_exclusions=("tie_rule_ties_lose", "sum"),
        )
        .astype(int)
        .to_dict(orient="index")
    )
    reproduced = {
        "quotes": int(len(quote)),
        "archived_voids": int(archived_void.sum()),
        "score_tie_voids": int(quote["score_tie_void"].sum()),
        "score_tie_losses": int(quote["score_tie_loss"].sum()),
        "no_score_voids": int(quote["no_score_void"].sum()),
        "other_voids_with_two_non_tied_scores": int(
            quote["other_void_with_scores"].sum()
        ),
        "quotes_with_two_valid_round_scores": int(both_valid.sum()),
        "score_model_eligible_quotes": int(archive_match.sum()),
        "market_devig_proper_score_eligible_quotes": int(
            quote["market_devig_proper_score_eligible"].sum()
        ),
        "ties_lose_quotes_excluded_from_market_proper_score": int(
            (archive_match & ties_lose).sum()
        ),
        "archived_vs_score_outcome_mismatches_when_both_scores_valid": int(
            mismatch.sum()
        ),
        "p1_archived_vs_score_mismatches": int(
            (graded_with_scores & ~p1_matches).sum()
        ),
        "p2_archived_vs_score_mismatches": int(
            (graded_with_scores & ~p2_matches).sum()
        ),
        "impossible_archived_outcome_pairs": int(
            (graded_with_scores & ~possible).sum()
        ),
        "rows_by_round": rows_by_round,
    }
    declared = audit.get("score_void_classification")
    if not isinstance(declared, Mapping) or set(declared) != set(
        CASCADE_SCORE_VOID_AUDIT_FIELDS
    ):
        raise ValueError("cascade score_void_classification field menu is not exact")
    if dict(declared) != reproduced:
        raise ValueError(
            "cascade score_void_classification audit differs from reproduced quote classes"
        )

    bookless_keys = [
        "event_key",
        "event_start_date",
        "round_num",
        "pair_key",
    ]
    if quote.groupby(bookless_keys, observed=True, dropna=False)[
        "score_model_eligible"
    ].nunique(dropna=False).gt(1).any():
        raise ValueError(
            "cascade bookless score-model eligibility differs across book quotes"
        )
    exclusions = (
        quote.loc[~quote["score_model_eligible"]]
        .groupby(bookless_keys, observed=True, dropna=False)
        .agg(
            excluded_book_quotes=("quote_id", "size"),
            no_score_void=("no_score_void", "all"),
            other_void_with_scores=("other_void_with_scores", "any"),
            both_target_round_scores_valid=(
                "both_target_round_scores_valid",
                "all",
            ),
        )
        .reset_index()
    )
    if not exclusions.empty:
        exclusions["proper_exclusion_reason"] = np.select(
            [
                exclusions["no_score_void"],
                exclusions["other_void_with_scores"],
                exclusions["both_target_round_scores_valid"],
            ],
            [
                "no_target_round_scores_archived_void",
                "archived_void_with_non_tied_scores",
                "archive_outcome_mismatch_with_valid_scores",
            ],
            default="missing_or_ungraded_target_round_score",
        )
    else:
        exclusions["proper_exclusion_reason"] = pd.Series(dtype=str)
    return (
        {
            **reproduced,
            "bookless_three_way_proper_exclusion_rows": int(len(exclusions)),
            "classification_reproduced_from_scores_and_archived_outcomes": True,
            "w2_w0_control_proper_menu_parity_checked_downstream": True,
        },
        exclusions,
    )


def _recompute_cascade_score_void_authority(frame: pd.DataFrame) -> pd.DataFrame:
    """Classify frozen quotes from target-round scores and book settlement rules."""

    required = [
        "quote_id",
        "event_key",
        "event_start_date",
        "round_num",
        "pair_key",
        "book",
        "p1_id",
        "p2_id",
        "p1_outcome",
        "p2_outcome",
        "tie_rule",
        "p1_round_score",
        "p2_round_score",
    ]
    _require(frame, required, "cascade full score/void authority")
    output = frame.copy()
    p1_score = pd.to_numeric(output["p1_round_score"], errors="coerce")
    p2_score = pd.to_numeric(output["p2_round_score"], errors="coerce")
    p1_archived = pd.to_numeric(output["p1_outcome"], errors="coerce")
    p2_archived = pd.to_numeric(output["p2_outcome"], errors="coerce")
    if not p1_archived.isin([0.0, 0.5, 1.0]).all() or not p2_archived.isin(
        [0.0, 0.5, 1.0]
    ).all():
        raise ValueError("cascade score/void archived outcome is outside {0,.5,1}")
    both_valid = p1_score.notna() & p2_score.notna()
    p1_derived = pd.Series(
        np.where(
            p1_score.lt(p2_score),
            1.0,
            np.where(p1_score.gt(p2_score), 0.0, 0.5),
        ),
        index=output.index,
    )
    p2_derived = pd.Series(
        np.where(
            p2_score.lt(p1_score),
            1.0,
            np.where(p2_score.gt(p1_score), 0.0, 0.5),
        ),
        index=output.index,
    )
    normalized_tie = output["tie_rule"].fillna("").astype(str).str.strip()
    normalized_tie = normalized_tie.str.casefold().str.replace(
        r"[_-]", " ", regex=True
    )
    ties_lose = (
        normalized_tie.str.contains("separate", regex=False)
        | normalized_tie.str.contains("ties lose", regex=False)
        | normalized_tie.str.contains("tie loses", regex=False)
        | normalized_tie.str.contains("three way", regex=False)
        | normalized_tie.str.contains("3 way", regex=False)
    )
    physical_tie = both_valid & p1_score.eq(p2_score)
    p1_settlement = p1_derived.where(~(physical_tie & ties_lose), 0.0)
    p2_settlement = p2_derived.where(~(physical_tie & ties_lose), 0.0)
    p1_matches = pd.Series(
        np.isclose(
            p1_archived,
            p1_settlement,
            rtol=0.0,
            atol=1e-10,
            equal_nan=False,
        ),
        index=output.index,
    )
    p2_matches = pd.Series(
        np.isclose(
            p2_archived,
            p2_settlement,
            rtol=0.0,
            atol=1e-10,
            equal_nan=False,
        ),
        index=output.index,
    )
    possible = (
        (p1_archived.eq(1.0) & p2_archived.eq(0.0))
        | (p1_archived.eq(0.0) & p2_archived.eq(1.0))
        | (p1_archived.eq(0.5) & p2_archived.eq(0.5) & ~ties_lose)
        | (p1_archived.eq(0.0) & p2_archived.eq(0.0) & ties_lose)
    )
    archive_match = both_valid & possible & p1_matches & p2_matches
    output["p1_round_score"] = p1_score
    output["p2_round_score"] = p2_score
    output["both_target_round_scores_valid"] = both_valid
    output["tie_rule_ties_lose"] = ties_lose
    output["archived_outcome_pair_possible"] = possible
    output["p1_archive_outcome_matches_score"] = both_valid & p1_matches
    output["p2_archive_outcome_matches_score"] = both_valid & p2_matches
    output["archive_outcome_matches_score"] = archive_match
    output["actual_score_tie"] = physical_tie
    output["score_tie_void"] = (
        p1_archived.eq(0.5)
        & p2_archived.eq(0.5)
        & physical_tie
        & ~ties_lose
    )
    output["score_tie_loss"] = (
        p1_archived.eq(0.0)
        & p2_archived.eq(0.0)
        & physical_tie
        & ties_lose
    )
    output["no_score_void"] = (
        p1_archived.eq(0.5) & p2_archived.eq(0.5) & ~both_valid
    )
    output["other_void_with_scores"] = (
        (p1_archived.eq(0.5) | p2_archived.eq(0.5))
        & both_valid
        & ~physical_tie
    )
    output["score_model_eligible"] = archive_match
    output["market_devig_proper_score_eligible"] = (
        archive_match & ~physical_tie & ~ties_lose
    )
    output["binary_model_eligible"] = output[
        "market_devig_proper_score_eligible"
    ]
    output["derived_score_outcome"] = p1_derived.where(both_valid, np.nan)
    output["p1_derived_score_outcome"] = p1_derived.where(both_valid, np.nan)
    output["p2_derived_score_outcome"] = p2_derived.where(both_valid, np.nan)
    output["settlement_score_outcome"] = p1_settlement.where(both_valid, np.nan)
    output["p1_settlement_score_outcome"] = p1_settlement.where(
        both_valid, np.nan
    )
    output["p2_settlement_score_outcome"] = p2_settlement.where(
        both_valid, np.nan
    )
    return output


def _cascade_score_void_audit_from_authority(
    quote: pd.DataFrame,
) -> dict[str, Any]:
    p1_archived = pd.to_numeric(quote["p1_outcome"], errors="coerce")
    p2_archived = pd.to_numeric(quote["p2_outcome"], errors="coerce")
    both_valid = quote["both_target_round_scores_valid"]
    archive_match = quote["archive_outcome_matches_score"]
    ties_lose = quote["tie_rule_ties_lose"]
    possible = quote["archived_outcome_pair_possible"]
    p1_matches = quote["p1_archive_outcome_matches_score"]
    p2_matches = quote["p2_archive_outcome_matches_score"]
    archived_void = p1_archived.eq(0.5) & p2_archived.eq(0.5)
    graded_with_scores = (
        p1_archived.isin([0.0, 0.5, 1.0])
        & p2_archived.isin([0.0, 0.5, 1.0])
        & both_valid
    )
    mismatch = graded_with_scores & ~archive_match
    use = quote.copy()
    use["round_market"] = "r" + use["round_num"].astype(int).astype(str)
    rows_by_round = (
        use.groupby("round_market", observed=True)
        .agg(
            quotes=("quote_id", "size"),
            score_tie_voids=("score_tie_void", "sum"),
            score_tie_losses=("score_tie_loss", "sum"),
            no_score_voids=("no_score_void", "sum"),
            score_model_eligible=("score_model_eligible", "sum"),
            market_devig_proper_score_eligible=(
                "market_devig_proper_score_eligible",
                "sum",
            ),
            ties_lose_market_benchmark_exclusions=(
                "tie_rule_ties_lose",
                "sum",
            ),
        )
        .astype(int)
        .to_dict(orient="index")
    )
    return {
        "quotes": int(len(use)),
        "archived_voids": int(archived_void.sum()),
        "score_tie_voids": int(use["score_tie_void"].sum()),
        "score_tie_losses": int(use["score_tie_loss"].sum()),
        "no_score_voids": int(use["no_score_void"].sum()),
        "other_voids_with_two_non_tied_scores": int(
            use["other_void_with_scores"].sum()
        ),
        "quotes_with_two_valid_round_scores": int(both_valid.sum()),
        "score_model_eligible_quotes": int(archive_match.sum()),
        "market_devig_proper_score_eligible_quotes": int(
            use["market_devig_proper_score_eligible"].sum()
        ),
        "ties_lose_quotes_excluded_from_market_proper_score": int(
            (archive_match & ties_lose).sum()
        ),
        "archived_vs_score_outcome_mismatches_when_both_scores_valid": int(
            mismatch.sum()
        ),
        "p1_archived_vs_score_mismatches": int(
            (graded_with_scores & ~p1_matches).sum()
        ),
        "p2_archived_vs_score_mismatches": int(
            (graded_with_scores & ~p2_matches).sum()
        ),
        "impossible_archived_outcome_pairs": int(
            (graded_with_scores & ~possible).sum()
        ),
        "rows_by_round": rows_by_round,
    }


def validate_cascade_score_void_classification(
    raw_probabilities: pd.DataFrame,
    normalized_quotes: pd.DataFrame,
    player_predictions: pd.DataFrame,
    round_panel: pd.DataFrame,
    audit: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Bind model copies to prediction-covered quotes and full score authority."""

    identity = [
        "quote_id",
        "event_key",
        "event_start_date",
        "round_num",
        "pair_key",
        "book",
        "p1_id",
        "p2_id",
        "p1_outcome",
        "p2_outcome",
        "tie_rule",
    ]
    classification_bool = [
        "both_target_round_scores_valid",
        "tie_rule_ties_lose",
        "archived_outcome_pair_possible",
        "p1_archive_outcome_matches_score",
        "p2_archive_outcome_matches_score",
        "archive_outcome_matches_score",
        "actual_score_tie",
        "score_tie_void",
        "score_tie_loss",
        "no_score_void",
        "other_void_with_scores",
        "score_model_eligible",
        "market_devig_proper_score_eligible",
        "binary_model_eligible",
    ]
    classification_numeric = [
        "p1_round_score",
        "p2_round_score",
        "derived_score_outcome",
        "p1_derived_score_outcome",
        "p2_derived_score_outcome",
        "settlement_score_outcome",
        "p1_settlement_score_outcome",
        "p2_settlement_score_outcome",
    ]
    required_raw = [
        "model",
        *identity,
        *classification_bool,
        *classification_numeric,
    ]
    _require(raw_probabilities, required_raw, "cascade score/void classification")
    _require(normalized_quotes, identity, "cascade normalized score/void authority")
    _require(
        player_predictions,
        ["model", "event_key", "round_num", "dg_id_key", "prediction"],
        "cascade prediction quote-menu authority",
    )
    _require(
        round_panel,
        ["event_key", "round_num", "dg_id", "score"],
        "cascade target-round score authority",
    )

    quotes = normalized_quotes[identity].copy()
    raw = raw_probabilities[required_raw].copy()
    for frame in (quotes, raw):
        frame["event_start_date"] = pd.to_datetime(
            frame["event_start_date"], errors="coerce"
        ).dt.normalize()
        frame["round_num"] = pd.to_numeric(frame["round_num"], errors="coerce")
        frame["quote_id"] = frame["quote_id"].astype(str)
        frame["book"] = frame["book"].astype(str).str.strip().str.casefold()
        frame["event_key"] = frame["event_key"].astype(str)
        frame["pair_key"] = frame["pair_key"].astype(str)
        frame["p1_id"] = frame["p1_id"].map(_stable_bet_id)
        frame["p2_id"] = frame["p2_id"].map(_stable_bet_id)
    if (
        quotes.empty
        or quotes[["quote_id", "event_key", "event_start_date", "round_num"]]
        .isna()
        .any()
        .any()
        or quotes["quote_id"].duplicated().any()
    ):
        raise ValueError("cascade normalized score/void quote authority is ambiguous")
    if raw.empty or raw.duplicated(["model", "quote_id"]).any():
        raise ValueError("cascade model score/void copies are empty or ambiguous")

    scores = round_panel[["event_key", "round_num", "dg_id", "score"]].copy()
    scores["event_key"] = scores["event_key"].astype(str)
    scores["round_num"] = pd.to_numeric(scores["round_num"], errors="coerce")
    scores["dg_id_key"] = scores["dg_id"].map(_stable_bet_id)
    score_present = scores["score"].notna() & scores["score"].astype(str).str.strip().ne("")
    scores["round_score"] = pd.to_numeric(scores["score"], errors="coerce")
    if (score_present & scores["round_score"].isna()).any() or np.isinf(
        scores["round_score"].fillna(0.0).to_numpy(float)
    ).any():
        raise ValueError("cascade target-round scores are malformed")
    scores = scores.loc[
        scores["event_key"].isin(set(quotes["event_key"]))
        & scores["round_num"].isin([1, 2, 3, 4])
    ].copy()
    score_keys = ["event_key", "round_num", "dg_id_key"]
    if scores[score_keys].isna().any().any() or scores.duplicated(score_keys).any():
        raise ValueError("cascade target-round score authority is ambiguous")
    left_scores = scores[score_keys + ["round_score"]].rename(
        columns={"dg_id_key": "p1_id", "round_score": "p1_round_score"}
    )
    right_scores = scores[score_keys + ["round_score"]].rename(
        columns={"dg_id_key": "p2_id", "round_score": "p2_round_score"}
    )
    authority = quotes.merge(
        left_scores,
        on=["event_key", "round_num", "p1_id"],
        how="left",
        validate="many_to_one",
    ).merge(
        right_scores,
        on=["event_key", "round_num", "p2_id"],
        how="left",
        validate="many_to_one",
    )
    authority = _recompute_cascade_score_void_authority(authority)
    reproduced = _cascade_score_void_audit_from_authority(authority)
    declared = audit.get("score_void_classification")
    if not isinstance(declared, Mapping) or set(declared) != set(
        CASCADE_SCORE_VOID_AUDIT_FIELDS
    ):
        raise ValueError("cascade score_void_classification field menu is not exact")
    if dict(declared) != reproduced:
        raise ValueError(
            "cascade score_void_classification audit differs from full normalized-quote/panel authority"
        )

    predictions = player_predictions[
        ["model", "event_key", "round_num", "dg_id_key", "prediction"]
    ].copy()
    predictions["model"] = predictions["model"].astype(str)
    predictions["event_key"] = predictions["event_key"].astype(str)
    predictions["round_num"] = pd.to_numeric(
        predictions["round_num"], errors="coerce"
    )
    predictions["dg_id_key"] = predictions["dg_id_key"].map(_stable_bet_id)
    prediction_present = predictions["prediction"].notna() & predictions[
        "prediction"
    ].astype(str).str.strip().ne("")
    predictions["_prediction"] = pd.to_numeric(
        predictions["prediction"], errors="coerce"
    )
    if (
        (prediction_present & predictions["_prediction"].isna()).any()
        or np.isinf(predictions["_prediction"].fillna(0.0).to_numpy(float)).any()
        or predictions.duplicated(
            ["model", "event_key", "round_num", "dg_id_key"]
        ).any()
    ):
        raise ValueError("cascade player-prediction quote-menu authority is malformed")
    raw_models = set(raw["model"].astype(str))
    prediction_models = set(predictions["model"])
    if raw_models != prediction_models or CASCADE_CONTROL not in raw_models:
        raise ValueError("cascade probability and player-prediction model menus differ")
    quote_menu = quotes[
        ["quote_id", "event_key", "round_num", "p1_id", "p2_id"]
    ].copy()
    menu_rows: list[dict[str, Any]] = []
    for model in sorted(raw_models):
        available = predictions.loc[
            predictions["model"].eq(model) & predictions["_prediction"].notna(),
            ["event_key", "round_num", "dg_id_key"],
        ].drop_duplicates()
        p1 = available.rename(columns={"dg_id_key": "p1_id"})
        p2 = available.rename(columns={"dg_id_key": "p2_id"})
        expected_ids = set(
            quote_menu.merge(
                p1,
                on=["event_key", "round_num", "p1_id"],
                how="inner",
                validate="many_to_one",
            )
            .merge(
                p2,
                on=["event_key", "round_num", "p2_id"],
                how="inner",
                validate="many_to_one",
            )["quote_id"]
            .astype(str)
        )
        observed_ids = set(
            raw.loc[raw["model"].astype(str).eq(model), "quote_id"].astype(str)
        )
        if observed_ids != expected_ids:
            raise ValueError(
                "cascade model score/void menu differs from its finite player-prediction quote subset: "
                f"model={model}, expected={len(expected_ids)}, observed={len(observed_ids)}, "
                f"expected_only={len(expected_ids - observed_ids)}, observed_only={len(observed_ids - expected_ids)}"
            )
        menu_rows.append(
            {
                "model": model,
                "prediction_covered_quote_ids": len(expected_ids),
                "probability_quote_ids": len(observed_ids),
                "normalized_quote_ids_outside_model_coverage": len(quotes)
                - len(expected_ids),
            }
        )

    authority_columns = [
        "quote_id",
        *identity[1:],
        *classification_bool,
        *classification_numeric,
    ]
    joined = raw.merge(
        authority[authority_columns].rename(
            columns={column: f"_authority_{column}" for column in authority_columns[1:]}
        ),
        on="quote_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("cascade probability row is absent from normalized quote authority")
    for column in identity[1:]:
        observed = joined[column]
        expected = joined[f"_authority_{column}"]
        if column in {"event_start_date"}:
            equal = pd.to_datetime(observed, errors="coerce").dt.normalize().eq(
                pd.to_datetime(expected, errors="coerce").dt.normalize()
            )
        elif column in {"round_num", "p1_outcome", "p2_outcome"}:
            equal = pd.Series(
                np.isclose(
                    pd.to_numeric(observed, errors="coerce"),
                    pd.to_numeric(expected, errors="coerce"),
                    rtol=0.0,
                    atol=1e-10,
                    equal_nan=True,
                ),
                index=joined.index,
            )
        else:
            equal = observed.fillna("__missing__").astype(str).eq(
                expected.fillna("__missing__").astype(str)
            )
        if not equal.all():
            raise ValueError(
                f"cascade score/void quote authority differs at {column!r}"
            )
    for column in classification_bool:
        observed = _require_exact_boolean_series(
            joined[column], f"cascade score/void {column}"
        )
        expected = _require_exact_boolean_series(
            joined[f"_authority_{column}"],
            f"cascade score/void authority {column}",
        )
        if not observed.eq(expected).all():
            raise ValueError(
                f"cascade score/void classification is false at {column!r}"
            )
    for column in classification_numeric:
        observed_raw = joined[column]
        observed_present = observed_raw.notna() & observed_raw.astype(str).str.strip().ne("")
        observed = pd.to_numeric(observed_raw, errors="coerce")
        expected = pd.to_numeric(
            joined[f"_authority_{column}"], errors="coerce"
        )
        if (observed_present & observed.isna()).any() or not np.isclose(
            observed.to_numpy(float),
            expected.to_numpy(float),
            rtol=0.0,
            atol=1e-10,
            equal_nan=True,
        ).all():
            raise ValueError(
                f"cascade score/void classification is false at {column!r}"
            )

    bookless_keys = ["event_key", "event_start_date", "round_num", "pair_key"]
    if authority.groupby(bookless_keys, observed=True, dropna=False)[
        "score_model_eligible"
    ].nunique(dropna=False).gt(1).any():
        raise ValueError(
            "cascade bookless score-model eligibility differs across book quotes"
        )
    exclusions = (
        authority.loc[~authority["score_model_eligible"]]
        .groupby(bookless_keys, observed=True, dropna=False)
        .agg(
            excluded_book_quotes=("quote_id", "size"),
            no_score_void=("no_score_void", "all"),
            other_void_with_scores=("other_void_with_scores", "any"),
            both_target_round_scores_valid=(
                "both_target_round_scores_valid",
                "all",
            ),
        )
        .reset_index()
    )
    if not exclusions.empty:
        exclusions["proper_exclusion_reason"] = np.select(
            [
                exclusions["no_score_void"],
                exclusions["other_void_with_scores"],
                exclusions["both_target_round_scores_valid"],
            ],
            [
                "no_target_round_scores_archived_void",
                "archived_void_with_non_tied_scores",
                "archive_outcome_mismatch_with_valid_scores",
            ],
            default="missing_or_ungraded_target_round_score",
        )
    else:
        exclusions["proper_exclusion_reason"] = pd.Series(dtype=str)
    c0_menu = next(row for row in menu_rows if row["model"] == CASCADE_CONTROL)
    return (
        {
            **reproduced,
            "bookless_three_way_proper_exclusion_rows": int(len(exclusions)),
            "classification_reproduced_from_scores_and_archived_outcomes": True,
            "full_normalized_quote_panel_authority_rows": int(len(authority)),
            "all_model_prediction_covered_quote_menus_exact": True,
            "model_quote_menu_rows": menu_rows,
            "c0_prediction_covered_quote_ids": int(
                c0_menu["prediction_covered_quote_ids"]
            ),
            "c0_normalized_quotes_outside_prediction_coverage": int(
                c0_menu["normalized_quote_ids_outside_model_coverage"]
            ),
            "raw_model_copy_rows_checked": int(len(raw)),
            "w2_w0_control_proper_menu_parity_checked_downstream": True,
        },
        exclusions,
    )


def _validate_cascade_audit_and_selection(
    manifest: Mapping[str, Any],
    audit: Mapping[str, Any],
    artifacts: Mapping[str, Path],
    handoff: frozen_handoff.FrozenHandoff,
) -> dict[str, Any]:
    """Apply the common semantic gate after provenance bytes are authoritative."""

    _require_exact_bool(audit.get("research_only"), True, "cascade audit research_only")
    _require_exact_bool(
        audit.get("production_changed"), False, "cascade audit production_changed"
    )
    for key in (
        "closing_price_used_as_skill_or_cascade_feature",
        "opening_price_used_as_skill_or_cascade_feature",
        "market_prices_used_for_cascade_probability_calibration",
        "sealed_results_used_for_selection",
        "target_or_upcoming_realized_weather_feature_used",
    ):
        _require_exact_bool(audit.get(key), False, f"cascade audit {key}")
    _require_exact_bool(
        audit.get("realized_weather_used_for_posthoc_attribution"),
        True,
        "cascade audit realized_weather_used_for_posthoc_attribution",
    )
    for key in (
        "ungraded_bets",
        "zero_gate_ungraded_bets",
        "all_offered_ungraded_bets",
        "nonzero_pnl_on_void_pushes",
    ):
        if int(audit.get(key, -1)) != 0:
            raise ValueError(f"cascade audit {key} must be zero")
    critical_audit = validate_cascade_critical_audit_gate(audit, artifacts)
    if str(audit.get("selection_role", "")) != handoff.selection_role:
        raise ValueError("cascade selection role differs from frozen handoff")
    if audit.get("promotion_gate_passed") is not handoff.promotion_gate_passed:
        raise ValueError("cascade promotion gate differs from frozen handoff")
    selected_candidates = audit.get("selected_candidates")
    expected_candidates = {
        handoff.selected_name,
        frozen_handoff.STRICT_PT_CONTROL,
    }
    if not isinstance(selected_candidates, list) or set(
        map(str, selected_candidates)
    ) != expected_candidates:
        raise ValueError("cascade selected candidate differs from frozen handoff")
    selection_path = _require_artifact(
        artifacts, "candidate_selection.csv", "cascade"
    )
    selection = pd.read_csv(selection_path)
    _require(
        selection,
        [
            "candidate",
            "candidate_type",
            "selected",
            "finalist_role",
            "promotion_gate_passed",
        ],
        "cascade candidate selection",
    )
    selected_mask = _require_exact_boolean_series(
        selection["selected"], "cascade candidate selection.selected"
    )
    if set(selection.loc[selected_mask, "candidate"].astype(str)) != expected_candidates:
        raise ValueError("cascade candidate-selection artifact differs from frozen handoff")
    if selection["candidate"].astype(str).duplicated().any():
        raise ValueError("cascade candidate-selection artifact has duplicate candidates")
    finalist_row = selection.loc[
        selection["candidate"].astype(str).eq(handoff.selected_name)
    ]
    control_row = selection.loc[
        selection["candidate"].astype(str).eq(frozen_handoff.STRICT_PT_CONTROL)
    ]
    if len(finalist_row) != 1 or len(control_row) != 1:
        raise ValueError("cascade candidate-selection finalist/control rows are ambiguous")
    if str(finalist_row["finalist_role"].iloc[0]) != handoff.selection_role:
        raise ValueError("cascade candidate-selection role differs from frozen handoff")
    finalist_gate = _require_exact_boolean_series(
        finalist_row["promotion_gate_passed"],
        "cascade finalist promotion_gate_passed",
    )
    control_gate = _require_exact_boolean_series(
        control_row["promotion_gate_passed"],
        "cascade control promotion_gate_passed",
    )
    if bool(finalist_gate.iloc[0]) is not handoff.promotion_gate_passed or bool(
        control_gate.iloc[0]
    ):
        raise ValueError("cascade candidate-selection promotion gate is inconsistent")
    frozen_audit = audit.get("frozen_finalist")
    if not isinstance(frozen_audit, Mapping):
        raise ValueError("cascade audit lacks frozen finalist validation")
    for key, expected in (
        ("manifest_path", str(handoff.compact_manifest_path)),
        ("manifest_sha256", handoff.audit["compact_manifest_sha256"]),
        ("prediction_path", str(handoff.compact_prediction_path)),
        ("prediction_sha256", handoff.audit["compact_prediction_sha256"]),
        ("finalist_name", handoff.selected_name),
        ("selection_role", handoff.selection_role),
        ("promotion_gate_passed", handoff.promotion_gate_passed),
        ("control_name", frozen_handoff.STRICT_PT_CONTROL),
    ):
        if frozen_audit.get(key) != expected:
            raise ValueError(f"cascade frozen finalist wrapper differs at {key!r}")
    full_chain = frozen_audit.get("full_provenance_chain")
    if not isinstance(full_chain, Mapping):
        raise ValueError("cascade frozen finalist lacks its full provenance chain")
    for key, expected in handoff.audit.items():
        if full_chain.get(key) != expected:
            raise ValueError(f"cascade frozen finalist audit differs at {key!r}")
    return critical_audit


def _validate_standard_cascade_provenance(
    root: Path, panel_manifest_path: Path
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Path],
    dict[str, Any],
]:
    manifest, artifacts = validate_artifact_manifest(root, label="cascade")
    execution, execution_paths = validate_shared_execution_provenance(
        manifest, artifacts, label="cascade"
    )
    source_authority = validate_execution_source_authority(
        execution,
        label="cascade",
        expected_sources=CASCADE_EVALUATOR_SOURCE_PINS,
        expected_primary=CASCADE_EVALUATOR_PRIMARY_SOURCE,
    )
    manifest_input_paths = _verify_manifest_inputs_in_execution(
        manifest, execution, label="cascade"
    )
    inputs = manifest["inputs"]
    panel_entry = inputs.get("panel_manifest")
    if not isinstance(panel_entry, Mapping):
        raise ValueError("cascade manifest lacks panel_manifest input")
    if Path(str(panel_entry.get("path", ""))).resolve() != panel_manifest_path.resolve():
        raise ValueError("cascade and weather audit use different panel manifests")
    frozen_entry = inputs.get("frozen_finalist_manifest")
    if not isinstance(frozen_entry, Mapping):
        raise ValueError("cascade manifest lacks frozen_finalist_manifest input")
    handoff = frozen_handoff.validate_frozen_handoff_chain(
        Path(str(frozen_entry.get("path", ""))).resolve()
    )
    completion_authority = validate_cascade_completion_authority(manifest, handoff)
    if _sha256(handoff.compact_manifest_path) != str(
        frozen_entry.get("sha256", "")
    ):
        raise ValueError("cascade frozen handoff manifest hash mismatch")
    frozen_prediction_entry = inputs.get("frozen_finalist_predictions")
    if not isinstance(frozen_prediction_entry, Mapping):
        raise ValueError("cascade manifest lacks frozen_finalist_predictions input")
    if (
        Path(str(frozen_prediction_entry.get("path", ""))).resolve()
        != handoff.compact_prediction_path.resolve()
        or str(frozen_prediction_entry.get("sha256", ""))
        != _sha256(handoff.compact_prediction_path)
    ):
        raise ValueError("cascade compact finalist prediction differs from frozen chain")
    core_entry = inputs.get("core_bridge")
    if not isinstance(core_entry, Mapping):
        raise ValueError("cascade manifest lacks core_bridge input")
    if (
        Path(str(core_entry.get("path", ""))).resolve()
        != handoff.core_bridge_path.resolve()
        or str(core_entry.get("sha256", "")) != handoff.core_bridge_sha256
    ):
        raise ValueError("cascade core bridge differs from frozen handoff chain")
    audit_path = _require_artifact(artifacts, "audit.json", "cascade")
    if str(manifest.get("audit_sha256", "")) != _sha256(audit_path):
        raise ValueError("cascade audit hash differs from manifest")
    audit = _json(audit_path)
    _require_exact_bool(audit.get("research_only"), True, "cascade audit research_only")
    _require_exact_bool(
        audit.get("production_changed"), False, "cascade audit production_changed"
    )
    for key in (
        "closing_price_used_as_skill_or_cascade_feature",
        "opening_price_used_as_skill_or_cascade_feature",
        "market_prices_used_for_cascade_probability_calibration",
        "sealed_results_used_for_selection",
        "target_or_upcoming_realized_weather_feature_used",
    ):
        _require_exact_bool(audit.get(key), False, f"cascade audit {key}")
    _require_exact_bool(
        audit.get("realized_weather_used_for_posthoc_attribution"),
        True,
        "cascade audit realized_weather_used_for_posthoc_attribution",
    )
    for key in (
        "ungraded_bets",
        "zero_gate_ungraded_bets",
        "all_offered_ungraded_bets",
        "nonzero_pnl_on_void_pushes",
    ):
        if int(audit.get(key, -1)) != 0:
            raise ValueError(f"cascade audit {key} must be zero")
    critical_audit = validate_cascade_critical_audit_gate(audit, artifacts)
    if str(audit.get("selection_role", "")) != handoff.selection_role:
        raise ValueError("cascade selection role differs from frozen handoff")
    if audit.get("promotion_gate_passed") is not handoff.promotion_gate_passed:
        raise ValueError("cascade promotion gate differs from frozen handoff")
    selected_candidates = audit.get("selected_candidates")
    expected_candidates = {
        handoff.selected_name,
        frozen_handoff.STRICT_PT_CONTROL,
    }
    if not isinstance(selected_candidates, list) or set(
        map(str, selected_candidates)
    ) != expected_candidates:
        raise ValueError("cascade selected candidate differs from frozen handoff")
    selection_path = _require_artifact(
        artifacts, "candidate_selection.csv", "cascade"
    )
    selection = pd.read_csv(selection_path)
    _require(
        selection,
        [
            "candidate",
            "candidate_type",
            "selected",
            "finalist_role",
            "promotion_gate_passed",
        ],
        "cascade candidate selection",
    )
    selected_mask = _require_exact_boolean_series(
        selection["selected"], "cascade candidate selection.selected"
    )
    if set(selection.loc[selected_mask, "candidate"].astype(str)) != expected_candidates:
        raise ValueError("cascade candidate-selection artifact differs from frozen handoff")
    if selection["candidate"].astype(str).duplicated().any():
        raise ValueError("cascade candidate-selection artifact has duplicate candidates")
    finalist_row = selection.loc[
        selection["candidate"].astype(str).eq(handoff.selected_name)
    ]
    control_row = selection.loc[
        selection["candidate"].astype(str).eq(frozen_handoff.STRICT_PT_CONTROL)
    ]
    if len(finalist_row) != 1 or len(control_row) != 1:
        raise ValueError("cascade candidate-selection finalist/control rows are ambiguous")
    if str(finalist_row["finalist_role"].iloc[0]) != handoff.selection_role:
        raise ValueError("cascade candidate-selection role differs from frozen handoff")
    finalist_gate = _require_exact_boolean_series(
        finalist_row["promotion_gate_passed"],
        "cascade finalist promotion_gate_passed",
    )
    control_gate = _require_exact_boolean_series(
        control_row["promotion_gate_passed"],
        "cascade control promotion_gate_passed",
    )
    if bool(finalist_gate.iloc[0]) is not handoff.promotion_gate_passed or bool(
        control_gate.iloc[0]
    ):
        raise ValueError("cascade candidate-selection promotion gate is inconsistent")
    frozen_audit = audit.get("frozen_finalist")
    if not isinstance(frozen_audit, Mapping):
        raise ValueError("cascade audit lacks frozen finalist validation")
    for key, expected in (
        ("manifest_path", str(handoff.compact_manifest_path)),
        ("manifest_sha256", handoff.audit["compact_manifest_sha256"]),
        ("prediction_path", str(handoff.compact_prediction_path)),
        ("prediction_sha256", handoff.audit["compact_prediction_sha256"]),
        ("finalist_name", handoff.selected_name),
        ("selection_role", handoff.selection_role),
        ("promotion_gate_passed", handoff.promotion_gate_passed),
        ("control_name", frozen_handoff.STRICT_PT_CONTROL),
    ):
        if frozen_audit.get(key) != expected:
            raise ValueError(f"cascade frozen finalist wrapper differs at {key!r}")
    full_chain = frozen_audit.get("full_provenance_chain")
    if not isinstance(full_chain, Mapping):
        raise ValueError("cascade frozen finalist lacks its full provenance chain")
    for key, expected in handoff.audit.items():
        if full_chain.get(key) != expected:
            raise ValueError(f"cascade frozen finalist audit differs at {key!r}")
    return manifest, audit, artifacts, {
        "handoff": handoff,
        "completion_authority": completion_authority,
        "execution": execution,
        "source_authority": source_authority,
        "critical_audit": critical_audit,
        "direct_paths": tuple(
            sorted(
                {
                    *execution_paths,
                    *manifest_input_paths,
                    *handoff.verified_paths,
                },
                key=str,
            )
        ),
    }


def _terminal_numeric_artifact_mapping(
    manifest: Mapping[str, Any],
    terminal_audit: Mapping[str, Any],
    source_dir: Path,
) -> tuple[dict[str, Path], tuple[Path, ...]]:
    """Verify and expose hash-bound external numeric artifacts read-only."""

    records = manifest.get("numeric_artifacts")
    if not isinstance(records, list) or not records:
        raise ValueError("terminal cascade manifest lacks numeric artifacts")
    if terminal_audit.get("numeric_artifacts") != records:
        raise ValueError("terminal cascade numeric inventories disagree")
    mapping: dict[str, Path] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"terminal cascade numeric record {index} is malformed")
        relative = str(record.get("path", "")).replace("\\", "/")
        if (
            not relative
            or Path(relative).is_absolute()
            or len(Path(relative).parts) != 1
            or relative in mapping
        ):
            raise ValueError(f"terminal cascade numeric path is invalid: {relative!r}")
        declared = Path(str(record.get("absolute_path", ""))).resolve()
        expected = (source_dir / relative).resolve()
        if declared != expected or declared.parent != source_dir:
            raise ValueError(
                f"terminal cascade numeric path escapes interrupted source: {relative}"
            )
        check_record = dict(record)
        check_record["path"] = str(declared)
        _require_provenance_record(
            check_record, label=f"terminal cascade numeric artifact[{index}]"
        )
        mapping[relative] = declared
    if set(mapping) != set(CASCADE_TERMINAL_NUMERIC_ARTIFACTS):
        raise ValueError(
            "terminal cascade numeric artifact menu differs from authority; "
            f"missing={sorted(set(CASCADE_TERMINAL_NUMERIC_ARTIFACTS) - set(mapping))}, "
            f"extra={sorted(set(mapping) - set(CASCADE_TERMINAL_NUMERIC_ARTIFACTS))}"
        )
    actual = {path.name for path in source_dir.iterdir() if path.is_file()}
    if actual != set(CASCADE_TERMINAL_NUMERIC_ARTIFACTS):
        raise ValueError("interrupted cascade top-level numeric inventory is not exact")
    if (source_dir / "manifest.json").exists() or (
        source_dir / "CASCADE_MARKET_EVAL_REPORT.md"
    ).exists():
        raise ValueError("interrupted cascade was improperly relabelled terminal")
    return mapping, tuple(mapping[name] for name in sorted(mapping))


def _parse_interrupted_cascade_args(
    original_execution: Mapping[str, Any], source_dir: Path
) -> dict[str, Any]:
    cli = original_execution.get("cli")
    if not isinstance(cli, list) or cli.count("--parsed-args-json") != 1:
        raise ValueError("interrupted cascade lacks one parsed-args authority")
    index = cli.index("--parsed-args-json")
    if index + 1 >= len(cli):
        raise ValueError("interrupted cascade parsed-args payload is missing")
    try:
        parsed = json.loads(str(cli[index + 1]))
    except json.JSONDecodeError as exc:
        raise ValueError("interrupted cascade parsed-args payload is malformed") from exc
    if not isinstance(parsed, dict):
        raise ValueError("interrupted cascade parsed-args payload is not an object")
    required = {
        "bootstrap_draws",
        "core_bridge",
        "finalist_dir",
        "odds_cache",
        "output_dir",
        "panel_dir",
        "preflight_only",
        "seed",
        "weather_coefficients",
    }
    if not required.issubset(parsed):
        raise ValueError("interrupted cascade parsed-args menu is incomplete")

    def resolve(value: Any) -> Path:
        path = Path(str(value))
        return (path if path.is_absolute() else REPO / path).resolve()

    if resolve(parsed["output_dir"]) != source_dir:
        raise ValueError("interrupted cascade CLI output differs from source directory")
    _require_exact_bool(
        parsed.get("preflight_only"), False, "interrupted cascade preflight_only"
    )
    parsed["_panel_dir"] = resolve(parsed["panel_dir"])
    parsed["_finalist_dir"] = resolve(parsed["finalist_dir"])
    parsed["_weather_coefficients"] = resolve(parsed["weather_coefficients"])
    parsed["_core_bridge"] = resolve(parsed["core_bridge"])
    parsed["_odds_cache"] = resolve(parsed["odds_cache"])
    return parsed


def _original_input_entry(
    input_records: Mapping[Path, Mapping[str, Any]], path: Path, label: str
) -> dict[str, str]:
    resolved = path.resolve()
    record = input_records.get(resolved)
    if not isinstance(record, Mapping):
        raise ValueError(f"interrupted cascade did not hash-bind {label}: {resolved}")
    _require_provenance_record(record, label=f"interrupted cascade input {label}")
    return {"path": str(resolved), "sha256": str(record["sha256"]).lower()}


def validate_terminalized_cascade_provenance(
    root: Path, panel_manifest_path: Path
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Path],
    dict[str, Any],
]:
    """Resolve a terminal wrapper without declaring its interrupted run complete."""

    root = root.resolve()
    manifest_path = root / "manifest.json"
    if _sha256(manifest_path) != CASCADE_TERMINAL_MANIFEST_SHA256:
        raise ValueError("terminal cascade manifest is not the pinned authority")
    wrapper, terminal_artifacts = validate_artifact_manifest(
        root, label="terminal cascade", contract=CASCADE_TERMINAL_CONTRACT
    )
    for key, expected in (
        ("production_changed", False),
        ("numeric_outputs_recomputed", False),
        ("interrupted_directory_relabelled_complete", False),
    ):
        _require_exact_bool(wrapper.get(key), expected, f"terminal cascade {key}")
    terminal_audit_path = _require_artifact(
        terminal_artifacts, "terminalization_audit.json", "terminal cascade"
    )
    if str(wrapper.get("terminalization_audit_sha256", "")) != _sha256(
        terminal_audit_path
    ):
        raise ValueError("terminal cascade terminalization-audit hash differs")
    terminal_audit = _json(terminal_audit_path)
    if terminal_audit.get("contract") != CASCADE_TERMINAL_CONTRACT:
        raise ValueError("terminal cascade audit contract differs")
    for key, expected in (
        ("complete", True),
        ("research_only", True),
        ("production_changed", False),
        ("numeric_outputs_recomputed", False),
        ("interrupted_directory_relabelled_complete", False),
    ):
        _require_exact_bool(terminal_audit.get(key), expected, f"terminal audit {key}")
    source_dir = Path(str(wrapper.get("interrupted_source_dir", ""))).resolve()
    if (
        not source_dir.is_dir()
        or Path(str(terminal_audit.get("interrupted_source_dir", ""))).resolve()
        != source_dir
    ):
        raise ValueError("terminal cascade interrupted source identity differs")
    if Path(str(terminal_audit.get("terminal_output_dir", ""))).resolve() != root:
        raise ValueError("terminal cascade audit output identity differs")
    numeric_artifacts, numeric_paths = _terminal_numeric_artifact_mapping(
        wrapper, terminal_audit, source_dir
    )

    incomplete_path = (
        source_dir / "execution_provenance" / "execution_provenance.incomplete.json"
    ).resolve()
    completed_path = (
        source_dir / "execution_provenance" / "execution_provenance.json"
    ).resolve()
    if completed_path.exists():
        raise ValueError("interrupted cascade unexpectedly has complete provenance")
    original_execution = _json(incomplete_path)
    _require_exact_bool(
        original_execution.get("complete"), False, "interrupted cascade complete"
    )
    original_sources = original_execution.get("sources")
    original_preserved = original_execution.get("preserved_sources")
    original_inputs = original_execution.get("inputs")
    if not all(
        isinstance(records, list)
        for records in (original_sources, original_preserved, original_inputs)
    ):
        raise ValueError("interrupted cascade provenance inventories are malformed")
    assert isinstance(original_sources, list)
    assert isinstance(original_preserved, list)
    assert isinstance(original_inputs, list)
    if not original_sources or len(original_sources) != len(original_preserved):
        raise ValueError("interrupted cascade source-copy inventory is not exact")
    original_paths: set[Path] = {incomplete_path}
    for label, records in (
        ("source", original_sources),
        ("preserved source", original_preserved),
        ("input", original_inputs),
    ):
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise ValueError(f"interrupted cascade {label}[{index}] is malformed")
            path, _ = _require_provenance_record(
                record, label=f"interrupted cascade {label}[{index}]"
            )
            original_paths.add(path)
    for source_record, copy_record in zip(original_sources, original_preserved):
        if (
            source_record.get("sha256") != copy_record.get("sha256")
            or source_record.get("bytes") != copy_record.get("bytes")
        ):
            raise ValueError("interrupted cascade preserved source differs")
        if Path(str(copy_record.get("path", ""))).resolve().parent != (
            source_dir / "execution_provenance"
        ).resolve():
            raise ValueError("interrupted cascade preserved source escapes provenance")
    original_source_authority = validate_execution_source_authority(
        original_execution,
        label="interrupted cascade",
        expected_sources=CASCADE_EVALUATOR_SOURCE_PINS,
        expected_primary=CASCADE_EVALUATOR_PRIMARY_SOURCE,
    )
    original_summary = terminal_audit.get("original_execution_provenance")
    if not isinstance(original_summary, Mapping):
        raise ValueError("terminal audit lacks original execution summary")
    expected_original_summary = {
        "incomplete_provenance_path": str(incomplete_path),
        "incomplete_provenance_sha256": _sha256(incomplete_path),
        "sources_verified": len(original_sources),
        "preserved_sources_verified": len(original_preserved),
        "inputs_verified": len(original_inputs),
        "executed_evaluator_source_sha256": CASCADE_EVALUATOR_SOURCE_PINS[
            CASCADE_EVALUATOR_PRIMARY_SOURCE
        ],
        "executed_evaluator_snapshot": str(
            Path(str(original_preserved[0]["path"])).resolve()
        ),
        "source_copy_pairs_exact": True,
    }
    if dict(original_summary) != expected_original_summary:
        raise ValueError("terminal audit original execution summary differs")

    terminal_execution, terminal_execution_paths = validate_shared_execution_provenance(
        wrapper, terminal_artifacts, label="terminal cascade"
    )
    expected_terminal_sources = {
        CASCADE_TERMINALIZER_PRIMARY_SOURCE.resolve(): (
            CASCADE_TERMINALIZER_SOURCE_SHA256
        ),
        CASCADE_EVALUATOR_PRIMARY_SOURCE.resolve(): CASCADE_EVALUATOR_SOURCE_PINS[
            CASCADE_EVALUATOR_PRIMARY_SOURCE
        ],
        Path(str(original_preserved[0]["path"])).resolve(): (
            CASCADE_EVALUATOR_SOURCE_PINS[CASCADE_EVALUATOR_PRIMARY_SOURCE]
        ),
        (ARCHIVE / "research_execution_provenance.py").resolve(): (
            CASCADE_EVALUATOR_SOURCE_PINS[
                ARCHIVE / "research_execution_provenance.py"
            ]
        ),
    }
    terminal_source_authority = validate_execution_source_authority(
        terminal_execution,
        label="terminal cascade",
        expected_sources=expected_terminal_sources,
        expected_primary=CASCADE_TERMINALIZER_PRIMARY_SOURCE,
    )
    terminal_input_paths = _fingerprint_paths(terminal_execution["inputs"])
    expected_terminal_inputs = {
        *numeric_paths,
        incomplete_path,
        *(Path(str(record["path"])).resolve() for record in original_preserved),
    }
    if terminal_input_paths != expected_terminal_inputs:
        raise ValueError("terminalizer input set differs from numeric/provenance authority")

    numeric_validation = terminal_audit.get("numeric_validation")
    if not isinstance(numeric_validation, Mapping):
        raise ValueError("terminal audit lacks numeric validation")
    numeric_audit_hash = _sha256(numeric_artifacts["audit.json"])
    if (
        str(numeric_validation.get("numeric_audit_sha256", "")) != numeric_audit_hash
        or str(wrapper.get("numeric_audit_sha256", "")) != numeric_audit_hash
    ):
        raise ValueError("terminal cascade numeric-audit hash differs")
    for key in ("bootstrap_draws", "bootstrap_seed", "selection_role"):
        if wrapper.get(key) != numeric_validation.get(key):
            raise ValueError(f"terminal cascade numeric validation differs at {key}")
    for key in (
        "primary_ungraded_rows",
        "zero_gate_ungraded_rows",
        "all_offered_ungraded_rows",
        "c3_annual_point_in_time_boundary_violations",
    ):
        if int(numeric_validation.get(key, -1)) != 0:
            raise ValueError(f"terminal cascade numeric validation {key} must be zero")
    for key, expected in (
        ("research_only", True),
        ("production_changed", False),
        ("promotion_gate_passed", False),
        ("target_or_upcoming_realized_weather_feature_used", False),
        ("performance_logical_keys_unique", True),
        ("bootstrap_matches_sealed_and_comparable_performance_keys", True),
    ):
        _require_exact_bool(
            numeric_validation.get(key), expected, f"terminal numeric validation {key}"
        )
    report = terminal_audit.get("report")
    if not isinstance(report, Mapping):
        raise ValueError("terminal audit lacks report fingerprint")
    report_path = _require_artifact(
        terminal_artifacts, "CASCADE_MARKET_EVAL_REPORT.md", "terminal cascade"
    )
    if (
        Path(str(report.get("path", ""))).resolve() != report_path
        or int(report.get("bytes", -1)) != report_path.stat().st_size
        or str(report.get("sha256", "")) != _sha256(report_path)
        or str(wrapper.get("report_sha256", "")) != _sha256(report_path)
    ):
        raise ValueError("terminal cascade report fingerprint differs")
    readme_path = _require_artifact(
        terminal_artifacts, "TERMINALIZATION_README.md", "terminal cascade"
    )
    if str(terminal_audit.get("readme_sha256", "")) != _sha256(readme_path):
        raise ValueError("terminal cascade README fingerprint differs")

    parsed = _parse_interrupted_cascade_args(original_execution, source_dir)
    if (
        int(parsed["bootstrap_draws"]) != int(wrapper["bootstrap_draws"])
        or int(parsed["seed"]) != int(wrapper["bootstrap_seed"])
    ):
        raise ValueError("terminal wrapper bootstrap settings differ from original CLI")
    input_records: dict[Path, Mapping[str, Any]] = {}
    for record in original_inputs:
        assert isinstance(record, Mapping)
        path = Path(str(record.get("path", ""))).resolve()
        if path in input_records:
            raise ValueError("interrupted cascade has duplicate input paths")
        input_records[path] = record
    panel_dir = parsed["_panel_dir"]
    finalist_dir = parsed["_finalist_dir"]
    canonical_inputs = {
        "panel_manifest": _original_input_entry(
            input_records, panel_dir / "manifest.json", "panel manifest"
        ),
        "player_event_outcomes": _original_input_entry(
            input_records,
            panel_dir / "player_event_outcomes.parquet",
            "player event outcomes",
        ),
        "player_round_panel": _original_input_entry(
            input_records, panel_dir / "player_round_panel.parquet", "round panel"
        ),
        "odds_event_crosswalk": _original_input_entry(
            input_records,
            panel_dir / "odds_event_crosswalk.parquet",
            "odds event crosswalk",
        ),
        "weather_coefficients": _original_input_entry(
            input_records, parsed["_weather_coefficients"], "weather coefficients"
        ),
        "frozen_finalist_manifest": _original_input_entry(
            input_records, finalist_dir / "manifest.json", "frozen finalist manifest"
        ),
        "core_bridge": _original_input_entry(
            input_records, parsed["_core_bridge"], "core bridge"
        ),
        "odds_cache_manifest": _original_input_entry(
            input_records, parsed["_odds_cache"] / "manifest.json", "odds cache manifest"
        ),
    }
    audit = _json(numeric_artifacts["audit.json"])
    frozen_audit = audit.get("frozen_finalist")
    if not isinstance(frozen_audit, Mapping):
        raise ValueError("terminal cascade numeric audit lacks frozen finalist")
    frozen_prediction = Path(str(frozen_audit.get("prediction_path", ""))).resolve()
    canonical_inputs["frozen_finalist_predictions"] = _original_input_entry(
        input_records, frozen_prediction, "frozen finalist predictions"
    )
    if Path(canonical_inputs["panel_manifest"]["path"]).resolve() != (
        panel_manifest_path.resolve()
    ):
        raise ValueError("terminal cascade and weather audit use different panel manifests")
    handoff = frozen_handoff.validate_frozen_handoff_chain(
        Path(canonical_inputs["frozen_finalist_manifest"]["path"])
    )
    handoff_bindings = handoff.audit.get("completion_input_bindings")
    if not isinstance(handoff_bindings, Mapping):
        raise ValueError("terminal cascade frozen handoff lacks completion bindings")
    completion_bindings = {
        name: handoff_bindings.get(name)
        for name in ("round_panel", "weather_coefficients", "core_bridge")
    }
    canonical_manifest = dict(wrapper)
    canonical_manifest.update(
        {
            "inputs": canonical_inputs,
            "completion_authority": completion_bindings,
            "audit_sha256": numeric_audit_hash,
            "terminal_wrapper_complete": True,
            "original_evaluator_complete": False,
            "numeric_artifacts_external_read_only": True,
        }
    )
    completion_authority = validate_cascade_completion_authority(
        canonical_manifest, handoff
    )
    if (
        Path(canonical_inputs["frozen_finalist_predictions"]["path"]).resolve()
        != handoff.compact_prediction_path.resolve()
        or canonical_inputs["frozen_finalist_predictions"]["sha256"]
        != _sha256(handoff.compact_prediction_path)
        or Path(canonical_inputs["core_bridge"]["path"]).resolve()
        != handoff.core_bridge_path.resolve()
        or canonical_inputs["core_bridge"]["sha256"] != handoff.core_bridge_sha256
    ):
        raise ValueError("terminal cascade frozen handoff inputs differ")
    critical_audit = _validate_cascade_audit_and_selection(
        canonical_manifest, audit, numeric_artifacts, handoff
    )
    direct_paths = tuple(
        sorted(
            {
                *terminal_execution_paths,
                *original_paths,
                *numeric_paths,
                *handoff.verified_paths,
                *(Path(record["path"]).resolve() for record in canonical_inputs.values()),
            },
            key=str,
        )
    )
    return canonical_manifest, audit, numeric_artifacts, {
        "handoff": handoff,
        "completion_authority": completion_authority,
        "execution": original_execution,
        "source_authority": original_source_authority,
        "terminal_source_authority": terminal_source_authority,
        "critical_audit": critical_audit,
        "direct_paths": direct_paths,
        "provenance_mode": "terminal_wrapper_external_numeric_v1",
        "terminal_wrapper_complete": True,
        "original_evaluator_complete": False,
        "numeric_artifacts_external_read_only": True,
        "terminal_manifest_sha256": _sha256(manifest_path),
        "interrupted_incomplete_provenance_sha256": _sha256(incomplete_path),
    }


def validate_cascade_provenance(
    root: Path, panel_manifest_path: Path
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Path],
    dict[str, Any],
]:
    """Validate either a standard complete cascade or the pinned terminal wrapper."""

    root = root.resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    contract = _json(manifest_path).get("contract")
    if contract == CASCADE_TERMINAL_CONTRACT:
        return validate_terminalized_cascade_provenance(root, panel_manifest_path)
    return _validate_standard_cascade_provenance(root, panel_manifest_path)


def validate_sealed_frozen_chain_identity(
    sealed_manifest: Mapping[str, Any],
    sealed_audit: Mapping[str, Any],
    sealed_execution_paths: Sequence[Path],
    handoff: frozen_handoff.FrozenHandoff,
) -> None:
    """Require sealed scoring and cascade scoring to use one frozen handoff."""

    expected = {
        "selected_model": handoff.selected_name,
        "selected_model_role": handoff.selection_role,
        "selection_manifest_sha256": handoff.audit["selection_manifest_sha256"],
        "compact_manifest_sha256": handoff.audit["compact_manifest_sha256"],
        "compact_prediction_sha256": handoff.audit["compact_prediction_sha256"],
        "source_prediction_sha256": handoff.audit["source_prediction_sha256"],
    }
    for key, value in expected.items():
        if sealed_manifest.get(key) != value:
            raise ValueError(f"sealed manifest differs from frozen handoff at {key!r}")
    if sealed_audit.get("selected_model") != handoff.selected_name:
        raise ValueError("sealed audit selected model differs from frozen handoff")
    if sealed_audit.get("selected_model_role") != handoff.selection_role:
        raise ValueError("sealed audit selected role differs from frozen handoff")
    frozen_audit = sealed_audit.get("frozen_handoff")
    if not isinstance(frozen_audit, Mapping):
        raise ValueError("sealed audit lacks frozen-handoff validation")
    for key, expected_value in handoff.audit.items():
        if frozen_audit.get(key) != expected_value:
            raise ValueError(f"sealed frozen-handoff audit differs at {key!r}")
    _require_exact_bool(
        sealed_audit.get("promotion_gate_passed_on_discovery"),
        handoff.promotion_gate_passed,
        "sealed audit promotion_gate_passed_on_discovery",
    )
    if sealed_manifest.get("locked_diagnostic_ablation_set") != handoff.selection.get(
        "locked_diagnostic_ablation_set"
    ):
        raise ValueError("sealed locked diagnostic menu differs from frozen selection")
    execution_path_set = {Path(path).resolve() for path in sealed_execution_paths}
    required_paths = {Path(path).resolve() for path in handoff.verified_paths}
    if not required_paths.issubset(execution_path_set):
        raise ValueError(
            "sealed execution provenance does not bind every required frozen-chain input: "
            f"{sorted(map(str, required_paths - execution_path_set))}"
        )


def validate_output_disjoint_from_inputs(
    output: Path, input_paths: Iterable[Path]
) -> None:
    output = output.resolve()
    conflicts: list[str] = []
    for raw_path in input_paths:
        path = Path(raw_path).resolve()
        if path == output or _inside(output, path):
            conflicts.append(str(path))
        elif path.is_dir() and _inside(path, output):
            conflicts.append(str(path))
    if conflicts:
        raise ValueError(
            "weather audit output overlaps an upstream input: "
            + repr(sorted(conflicts)[:10])
        )


def _direct_upstream_paths(
    *,
    panel_path: Path,
    panel_manifest_path: Path,
    sealed_dir: Path,
    sealed_artifacts: Mapping[str, Path],
    sealed_execution_paths: Sequence[Path],
    cascade_dir: Path,
    cascade_artifacts: Mapping[str, Path],
    cascade_context: Mapping[str, Any],
) -> tuple[Path, ...]:
    paths = {
        panel_path.resolve(),
        panel_manifest_path.resolve(),
        (sealed_dir / "manifest.json").resolve(),
        (cascade_dir / "manifest.json").resolve(),
        *[path.resolve() for path in sealed_artifacts.values()],
        *[Path(path).resolve() for path in sealed_execution_paths],
        *[path.resolve() for path in cascade_artifacts.values()],
        *[Path(path).resolve() for path in cascade_context["direct_paths"]],
    }
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return tuple(sorted(paths, key=str))


def _assert_provenance_input_set(
    state: execution_provenance.ExecutionProvenance,
    expected_paths: Iterable[Path],
) -> None:
    expected = {Path(path).resolve() for path in expected_paths}
    observed = _fingerprint_paths(state.input_fingerprints)
    if observed != expected:
        raise RuntimeError(
            "weather audit execution provenance input set differs from validated "
            f"inputs; missing={sorted(map(str, expected - observed))}, "
            f"extra={sorted(map(str, observed - expected))}"
        )


def _assert_execution_snapshot_current(
    state: execution_provenance.ExecutionProvenance,
) -> None:
    for label, records in (
        ("source", state.source_fingerprints),
        ("preserved source", state.source_copy_fingerprints),
        ("input", state.input_fingerprints),
    ):
        for record in records:
            current = execution_provenance.fingerprint(
                Path(str(record["path"])).resolve()
            )
            if any(
                current[field] != record[field]
                for field in ("sha256", "bytes", "mtime_ns")
            ):
                raise RuntimeError(
                    f"weather audit {label} changed between provenance snapshot "
                    "and semantic validation"
                )


def _write_complete_manifest_after_final_snapshot(
    output: Path,
    manifest: Mapping[str, Any],
    provenance_state: execution_provenance.ExecutionProvenance,
) -> None:
    """Perform the last snapshot check and the sole complete-manifest write."""

    _require_exact_bool(manifest.get("complete"), True, "weather manifest complete")
    _require_exact_bool(
        manifest.get("research_only"), True, "weather manifest research_only"
    )
    _assert_execution_snapshot_current(provenance_state)
    _atomic_json(output.resolve() / "manifest.json", manifest)


def _numeric(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")


def _tee_minutes(value: Any) -> float:
    if value is None or pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    match = re.search(r"(?<!\d)(\d{1,2}):(\d{2})\s*([ap]m)?", text, re.I)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        marker = (match.group(3) or "").casefold()
        if minute > 59 or hour > 23 or (marker and not 1 <= hour <= 12):
            return np.nan
        if marker == "pm" and hour != 12:
            hour += 12
        if marker == "am" and hour == 12:
            hour = 0
        return float(hour * 60 + minute)
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return np.nan
    return float(parsed.hour * 60 + parsed.minute)


def _spread(values: pd.Series) -> float:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if not len(array):
        return np.nan
    return float(np.quantile(array, 0.9) - np.quantile(array, 0.1))


def _withdrawal_flag(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.casefold().str.contains(
        r"(?:^|[^a-z])(?:wd|w/d|withdrawn|withdrew)(?:[^a-z]|$)", regex=True
    )


def build_weather_profiles(
    panel_path: Path, *, required_event_keys: Iterable[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Create physical-course-stratified weather profiles and tee-wave gaps."""

    desired = [
        "event_key",
        "event_name",
        "event_start_date",
        "year",
        "round_num",
        "round_date",
        "course_num",
        "dg_id",
        "fin_text",
        "weather_source_teetime",
        "weather_wind",
        "weather_dew",
        "weather_temp",
        "weather_rain",
        "weather_field_mean_wind",
        "weather_field_mean_dew",
        "weather_field_mean_temp",
        "weather_field_mean_rain",
        "weather_delta_wind",
        "weather_delta_dew",
        "weather_delta_temp",
        "weather_delta_rain",
        "dg_sg_total_field_adjusted",
    ]
    try:
        rounds = pd.read_parquet(panel_path, columns=desired)
    except Exception as exc:
        raise ValueError(f"round panel cannot satisfy weather schema: {exc}") from exc
    _require(rounds, desired, "round weather panel")
    rounds["event_start_date"] = pd.to_datetime(
        rounds["event_start_date"], errors="coerce"
    ).dt.normalize()
    rounds["round_date"] = pd.to_datetime(rounds["round_date"], errors="coerce").dt.normalize()
    rounds["round_num"] = pd.to_numeric(rounds["round_num"], errors="coerce")
    if rounds[
        ["event_key", "event_start_date", "year", "round_num", "course_num", "dg_id"]
    ].isna().any().any():
        raise ValueError("round weather panel has null identity fields")
    rounds = rounds.loc[
        rounds["event_start_date"].between(SEALED_START, SEALED_END)
        & rounds["round_num"].isin([1, 2, 3, 4])
    ].copy()
    rounds["round_num"] = rounds["round_num"].astype(int)
    rounds["year"] = pd.to_numeric(rounds["year"], errors="raise").astype(int)
    rounds["course_num"] = rounds["course_num"].astype(str).str.strip()
    if rounds["course_num"].eq("").any():
        raise ValueError("round weather panel has blank physical-course identity")
    if rounds["event_start_date"].dt.year.ne(rounds["year"]).any():
        raise ValueError("round weather panel event-year/date identity is inconsistent")
    identity = [
        "event_key",
        "event_start_date",
        "year",
        "round_num",
        "course_num",
        "dg_id",
    ]
    if rounds.duplicated(identity).any():
        raise ValueError("round weather panel has duplicate event/player/round keys")
    required = set(str(value) for value in required_event_keys)
    present = set(rounds["event_key"].astype(str))
    if not required.issubset(present):
        raise ValueError(
            "weather panel is missing evaluated events: "
            + repr(sorted(required - present)[:10])
        )
    rounds = rounds.loc[rounds["event_key"].astype(str).isin(required)].copy()
    numeric_columns = [
        column
        for column in desired
        if column.startswith("weather_") and column != "weather_source_teetime"
    ]
    for column in numeric_columns:
        raw_values = rounds[column]
        declared = raw_values.notna() & raw_values.astype(str).str.strip().ne("")
        numeric = pd.to_numeric(raw_values, errors="coerce")
        if (declared & (~np.isfinite(numeric))).any():
            raise ValueError(
                f"round weather panel {column} contains a declared non-finite value"
            )
        rounds[column] = numeric
    rounds["dg_sg_total_field_adjusted"] = pd.to_numeric(
        rounds["dg_sg_total_field_adjusted"], errors="coerce"
    )
    weather_value_bounds: dict[str, dict[str, float | int | None | str | bool]] = {}
    for field in ("wind", "rain"):
        for role, column in (
            ("raw", f"weather_{field}"),
            ("physical_course_field_mean", f"weather_field_mean_{field}"),
        ):
            declared = rounds[column].notna()
            values = rounds.loc[declared, column]
            if (values < 0.0).any():
                raise ValueError(
                    f"round weather panel {column} must be nonnegative when declared"
                )
            weather_value_bounds[f"{field}_{role}"] = {
                "declared_rows": int(declared.sum()),
                "minimum": float(values.min()) if len(values) else None,
                "maximum": float(values.max()) if len(values) else None,
                "finite_when_declared": True,
                "nonnegative_when_declared": True,
                "units": "source_native_round_weather_units_no_conversion",
            }
    weather_group_identity = [
        "event_key",
        "event_start_date",
        "year",
        "round_num",
        "course_num",
    ]
    weather_delta_max_errors: dict[str, float | None] = {}
    weather_declared_mean_max_errors: dict[str, float | None] = {}
    weather_centered_delta_mean_max_errors: dict[str, float | None] = {}
    weather_mean_only_unmatched_rows: dict[str, int] = {}
    for field in ("wind", "dew", "temp", "rain"):
        raw = rounds[f"weather_{field}"]
        mean = rounds[f"weather_field_mean_{field}"]
        delta = rounds[f"weather_delta_{field}"]
        complete = raw.notna() & mean.notna() & delta.notna()
        # The frozen panel intentionally retains the physical-course/round
        # field mean on unmatched player rows while leaving that player's raw
        # weather and centered delta missing.  That is group context, not a
        # player observation.  Raw and delta must still be jointly present,
        # and every raw observation must carry its declared mean.
        raw_delta_mismatch = raw.notna().ne(delta.notna())
        raw_without_mean = raw.notna() & mean.isna()
        if raw_delta_mismatch.any() or raw_without_mean.any():
            raise ValueError(
                f"round weather panel has incomplete {field} delta identity rows"
            )
        mean_only = raw.isna() & delta.isna() & mean.notna()
        weather_mean_only_unmatched_rows[field] = int(mean_only.sum())
        error = (delta.loc[complete] - (raw.loc[complete] - mean.loc[complete])).abs()
        max_error = float(error.max()) if len(error) else None
        weather_delta_max_errors[field] = max_error
        if max_error is not None and (not math.isfinite(max_error) or max_error > 1e-8):
            raise ValueError(
                f"round weather panel {field} delta algebra identity failed: "
                f"max_abs_error={max_error}"
            )
        recomputed_mean = rounds.groupby(
            weather_group_identity,
            observed=True,
            dropna=False,
        )[f"weather_{field}"].transform("mean")
        declared_mean = mean.notna()
        if (declared_mean & recomputed_mean.isna()).any():
            raise ValueError(
                f"round weather panel {field} declares a physical-course mean "
                "without any raw observations in its group"
            )
        mean_error = (
            mean.loc[declared_mean] - recomputed_mean.loc[declared_mean]
        ).abs()
        max_mean_error = float(mean_error.max()) if len(mean_error) else None
        weather_declared_mean_max_errors[field] = max_mean_error
        if max_mean_error is not None and (
            not math.isfinite(max_mean_error) or max_mean_error > 1e-8
        ):
            raise ValueError(
                f"round weather panel {field} declared physical-course mean failed: "
                f"max_abs_error={max_mean_error}"
            )
        centered_means = (
            rounds.loc[complete]
            .groupby(weather_group_identity, observed=True, dropna=False)[
                f"weather_delta_{field}"
            ]
            .mean()
            .abs()
        )
        max_center_error = (
            float(centered_means.max()) if len(centered_means) else None
        )
        weather_centered_delta_mean_max_errors[field] = max_center_error
        if max_center_error is not None and (
            not math.isfinite(max_center_error) or max_center_error > 1e-8
        ):
            raise ValueError(
                f"round weather panel {field} deltas are not centered within "
                f"physical course/round: max_abs_error={max_center_error}"
            )
    rounds["weather_observed"] = (
        rounds[
            [
                "weather_wind",
                "weather_dew",
                "weather_field_mean_wind",
                "weather_field_mean_dew",
                "weather_delta_wind",
                "weather_delta_dew",
            ]
        ]
        .notna()
        .all(axis=1)
    )
    rounds["tee_minute"] = rounds["weather_source_teetime"].map(_tee_minutes)
    rounds["event_withdrawal"] = _withdrawal_flag(rounds["fin_text"])

    wave_parts: list[pd.DataFrame] = []
    for _, group in rounds.groupby(
        ["event_key", "event_start_date", "year", "round_num", "course_num"],
        sort=False,
        observed=True,
    ):
        local = group.copy()
        valid = local["tee_minute"].notna()
        local["tee_wave"] = "unavailable"
        unique_times = int(local.loc[valid, "tee_minute"].nunique())
        if int(valid.sum()) >= 8 and unique_times >= 2:
            median = float(local.loc[valid, "tee_minute"].median())
            early = valid & local["tee_minute"].le(median)
            late = valid & local["tee_minute"].gt(median)
            if early.any() and late.any():
                local.loc[early, "tee_wave"] = "early"
                local.loc[late, "tee_wave"] = "late"
        wave_parts.append(local)
    rounds = pd.concat(wave_parts, ignore_index=True) if wave_parts else rounds

    rows: list[dict[str, Any]] = []
    for (event_key, event_start_date, year, round_num, course_num), group in rounds.groupby(
        ["event_key", "event_start_date", "year", "round_num", "course_num"],
        sort=True,
        observed=True,
    ):
        row: dict[str, Any] = {
            "event_key": str(event_key),
            "event_name": str(group["event_name"].iloc[0]),
            "event_start_date": event_start_date,
            "year": int(year),
            "round_num": int(round_num),
            "round_date": group["round_date"].iloc[0],
            "course_num": str(course_num),
            "player_round_rows": int(len(group)),
            "weather_observed_rows": int(group["weather_observed"].sum()),
            "weather_coverage_rate": float(group["weather_observed"].mean()),
            "tee_time_observed_rows": int(group["tee_minute"].notna().sum()),
            "tee_time_coverage_rate": float(group["tee_minute"].notna().mean()),
            "tee_wave_available": bool(group["tee_wave"].isin(["early", "late"]).any()),
            "event_withdrawal_players": int(group.loc[group["event_withdrawal"], "dg_id"].nunique()),
        }
        for field in ("wind", "dew", "temp", "rain"):
            row[f"field_mean_{field}"] = float(group[f"weather_field_mean_{field}"].mean())
            row[f"{field}_delta_p90_p10"] = _spread(group[f"weather_delta_{field}"])
            # Deltas were centered within the exact physical course/round in
            # the frozen panel.  Using them here prevents a multi-course event
            # from masquerading as a tee-wave effect.
            early = group.loc[
                group["tee_wave"].eq("early"), f"weather_delta_{field}"
            ]
            late = group.loc[
                group["tee_wave"].eq("late"), f"weather_delta_{field}"
            ]
            row[f"late_minus_early_{field}"] = (
                float(late.mean() - early.mean()) if len(early) and len(late) else np.nan
            )
        early_sg = group.loc[
            group["tee_wave"].eq("early"), "dg_sg_total_field_adjusted"
        ]
        late_sg = group.loc[
            group["tee_wave"].eq("late"), "dg_sg_total_field_adjusted"
        ]
        row["late_minus_early_field_adjusted_sg"] = (
            float(late_sg.mean() - early_sg.mean())
            if len(early_sg) and len(late_sg)
            else np.nan
        )
        rows.append(row)
    course_profiles = pd.DataFrame(rows)
    if course_profiles.empty:
        raise ValueError("weather profile contains no evaluated event-rounds")
    severity_sources = {
        "field_wind": course_profiles["field_mean_wind"],
        "wind_spread": course_profiles["wind_delta_p90_p10"],
        "dew_spread": course_profiles["dew_delta_p90_p10"],
        "temp_spread": course_profiles["temp_delta_p90_p10"],
        "rain_level": course_profiles["field_mean_rain"],
        "wave_wind_gap": course_profiles["late_minus_early_wind"].abs(),
    }
    percentile_columns: list[str] = []
    for label, values in severity_sources.items():
        column = f"severity_component_{label}_pct"
        course_profiles[column] = pd.to_numeric(values, errors="coerce").rank(
            method="average", pct=True
        )
        percentile_columns.append(column)
    course_profiles["weather_severity_percentile"] = course_profiles[
        percentile_columns
    ].max(
        axis=1, skipna=True
    )
    course_profiles["weather_severity_bucket"] = pd.cut(
        course_profiles["weather_severity_percentile"],
        bins=[-np.inf, 0.50, 0.75, 0.90, np.inf],
        labels=["low", "elevated", "high", "extreme"],
        right=False,
    ).astype("string").fillna("missing")
    course_profiles["target_round_realized_weather_timing"] = (
        "oracle_post_round_attribution_only"
    )
    course_profiles["target_round_realized_weather_used_as_feature"] = False
    course_profiles["physical_course_join_preserved"] = True

    def weighted_mean(group: pd.DataFrame, column: str) -> float:
        values = pd.to_numeric(group[column], errors="coerce")
        weights = pd.to_numeric(group["player_round_rows"], errors="coerce")
        valid = values.notna() & weights.gt(0)
        return (
            float(np.average(values.loc[valid], weights=weights.loc[valid]))
            if valid.any()
            else np.nan
        )

    event_round_rows: list[dict[str, Any]] = []
    event_round_keys = [
        "event_key",
        "event_name",
        "event_start_date",
        "year",
        "round_num",
        "round_date",
    ]
    for keys, group in course_profiles.groupby(
        event_round_keys, observed=True, sort=True, dropna=False
    ):
        row = dict(zip(event_round_keys, keys if isinstance(keys, tuple) else (keys,)))
        physical_courses = sorted(group["course_num"].astype(str).unique())
        player_rows = int(group["player_round_rows"].sum())
        weather_rows = int(group["weather_observed_rows"].sum())
        tee_rows = int(group["tee_time_observed_rows"].sum())
        row.update(
            {
                "physical_course_count": int(len(physical_courses)),
                "physical_courses": "|".join(physical_courses),
                "multi_course_event_round": bool(len(physical_courses) > 1),
                "course_aggregation_method": (
                    "physical_course_stratified_then_player_round_weighted"
                ),
                "player_round_rows": player_rows,
                "weather_observed_rows": weather_rows,
                "weather_coverage_rate": weather_rows / player_rows if player_rows else np.nan,
                "tee_time_observed_rows": tee_rows,
                "tee_time_coverage_rate": tee_rows / player_rows if player_rows else np.nan,
                "tee_wave_available": bool(group["tee_wave_available"].any()),
                "event_withdrawal_players": int(group["event_withdrawal_players"].sum()),
                "weather_severity_percentile": float(
                    group["weather_severity_percentile"].max()
                ),
                "course_weighted_weather_severity_mean": weighted_mean(
                    group, "weather_severity_percentile"
                ),
                "target_round_realized_weather_timing": (
                    "oracle_post_round_attribution_only"
                ),
                "target_round_realized_weather_used_as_feature": False,
                "physical_course_join_preserved": True,
            }
        )
        for field in ("wind", "dew", "temp", "rain"):
            row[f"field_mean_{field}"] = weighted_mean(group, f"field_mean_{field}")
            row[f"{field}_delta_p90_p10"] = float(
                group[f"{field}_delta_p90_p10"].max()
            )
            row[f"late_minus_early_{field}"] = weighted_mean(
                group, f"late_minus_early_{field}"
            )
            row[f"max_abs_course_late_minus_early_{field}"] = float(
                group[f"late_minus_early_{field}"].abs().max()
            )
        row["late_minus_early_field_adjusted_sg"] = weighted_mean(
            group, "late_minus_early_field_adjusted_sg"
        )
        row["max_abs_course_late_minus_early_field_adjusted_sg"] = float(
            group["late_minus_early_field_adjusted_sg"].abs().max()
        )
        event_round_rows.append(row)
    profiles = pd.DataFrame(event_round_rows)
    profiles["weather_severity_bucket"] = pd.cut(
        profiles["weather_severity_percentile"],
        bins=[-np.inf, 0.50, 0.75, 0.90, np.inf],
        labels=["low", "elevated", "high", "extreme"],
        right=False,
    ).astype("string").fillna("missing")

    event_profiles = profiles.groupby(
        ["event_key", "event_name", "event_start_date", "year"], observed=True
    ).agg(
        event_rounds=("round_num", "nunique"),
        event_weather_coverage_rate=("weather_coverage_rate", "mean"),
        event_weather_severity_mean=("weather_severity_percentile", "mean"),
        event_weather_severity_max=("weather_severity_percentile", "max"),
        event_max_abs_wave_wind_gap=("late_minus_early_wind", lambda x: x.abs().max()),
        event_max_abs_course_wave_wind_gap=(
            "max_abs_course_late_minus_early_wind",
            "max",
        ),
        event_max_wind_spread=("wind_delta_p90_p10", "max"),
        event_withdrawal_players=("event_withdrawal_players", "max"),
        event_max_physical_course_count=("physical_course_count", "max"),
        event_multi_course=("multi_course_event_round", "max"),
    ).reset_index()
    event_profiles["event_weather_severity_bucket"] = pd.cut(
        event_profiles["event_weather_severity_max"],
        bins=[-np.inf, 0.50, 0.75, 0.90, np.inf],
        labels=["low", "elevated", "high", "extreme"],
        right=False,
    ).astype("string").fillna("missing")
    event_profiles["target_event_realized_weather_timing"] = (
        "oracle_post_event_attribution_only"
    )
    event_profiles["target_event_realized_weather_used_as_feature"] = False
    event_profiles["course_join_scope"] = (
        "event_portfolio_physical_course_stratified_summary_no_bet_course_identity"
    )
    audit = {
        "events": int(event_profiles["event_key"].nunique()),
        "event_rounds": int(len(profiles)),
        "player_rounds": int(len(rounds)),
        "physical_course_round_profiles": int(len(course_profiles)),
        "physical_course_identity_column": "course_num",
        "weather_group_identity": (
            "event_key|event_start_date|year|round_num|course_num"
        ),
        "multi_course_event_rounds": int(profiles["multi_course_event_round"].sum()),
        "missing_physical_course_rows": 0,
        "physical_course_join_preserved_before_event_aggregation": True,
        "weather_coverage_rate": float(rounds["weather_observed"].mean()),
        "weather_coverage_requires_wind_and_dew": True,
        "weather_delta_identity_max_abs_error": weather_delta_max_errors,
        "weather_declared_physical_course_mean_max_abs_error": (
            weather_declared_mean_max_errors
        ),
        "weather_centered_delta_group_mean_max_abs_error": (
            weather_centered_delta_mean_max_errors
        ),
        "weather_mean_only_unmatched_rows_by_field": (
            weather_mean_only_unmatched_rows
        ),
        "weather_mean_only_unmatched_rows_semantics": (
            "physical_course_round_group_context_only_not_player_observation"
        ),
        "weather_delta_identity_tolerance": 1e-8,
        "weather_delta_algebra_verified": True,
        "weather_physical_course_means_recomputed": True,
        "weather_centered_deltas_recomputed": True,
        "weather_value_domain_contract": {
            "wind_and_rain": weather_value_bounds,
            "wind_and_rain_rule": (
                "raw_and_physical_course_field_mean_finite_nonnegative_when_declared"
            ),
            "dew_and_temperature_units": (
                "source_native_units_not_declared_by_frozen_panel_manifest"
            ),
            "dew_and_temperature_range_gate": (
                "none_no_arbitrary_threshold_without_authoritative_unit_schema"
            ),
        },
        "tee_time_coverage_rate": float(rounds["tee_minute"].notna().mean()),
        "severity_definition": (
            "outcome-blind maximum empirical percentile of field wind, wind/dew/temp "
            "spread, rain level, and absolute early/late wind gap"
        ),
        "target_or_upcoming_realized_weather_feature_used": False,
        "target_weather_is_oracle_posthoc_attribution_only": True,
        "outcome_selected_weather_exclusion_read": False,
    }
    return course_profiles, profiles, event_profiles, audit


def _catalog_scalar_equal(observed: Any, expected: Any) -> bool:
    observed_missing = bool(pd.isna(observed))
    expected_missing = bool(pd.isna(expected))
    if observed_missing or expected_missing:
        return observed_missing and expected_missing
    if isinstance(expected, (bool, np.bool_)):
        return isinstance(observed, (bool, np.bool_)) and bool(observed) is bool(expected)
    if isinstance(expected, (int, float, np.integer, np.floating)) and not isinstance(
        expected, (bool, np.bool_)
    ):
        try:
            return bool(
                np.isclose(float(observed), float(expected), rtol=0.0, atol=1e-12)
            )
        except (TypeError, ValueError):
            return False
    return isinstance(observed, str) and observed == str(expected)


def reconstruct_and_validate_sealed_candidate_authority(
    handoff: frozen_handoff.FrozenHandoff,
    sealed_manifest: Mapping[str, Any],
    sealed_audit: Mapping[str, Any],
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[str, ...], dict[str, str]]:
    """Rebuild the producer-prescribed candidate menu from frozen inputs.

    The sealed manifest and metadata are consumers, not candidate authorities.
    Names, roles, and pairing fields are independently reconstructed from the
    validated handoff selection and its hash-bound frozen candidate catalog.
    """

    selected = str(handoff.selected_name)
    if selected == frozen_handoff.STRICT_PT_CONTROL:
        raise ValueError("sealed selected candidate cannot relabel the strict PT control")
    locked = handoff.selection.get("locked_diagnostic_ablation_set")
    if not isinstance(locked, Mapping):
        raise ValueError("frozen handoff lacks locked diagnostic candidates")
    included = locked.get("included_candidates")
    if (
        not isinstance(included, list)
        or not all(isinstance(value, str) and value for value in included)
        or len(included) != len(set(included))
        or included != sorted(included)
    ):
        raise ValueError("frozen included-candidate menu is malformed or noncanonical")
    if str(locked.get("selected_candidate", "")) != selected:
        raise ValueError("frozen diagnostic menu names a different selected candidate")
    _require_exact_bool(
        locked.get("locked_before_sealed_scoring"),
        True,
        "frozen locked_diagnostic_ablation_set.locked_before_sealed_scoring",
    )
    _require_exact_bool(
        locked.get("may_reselect_or_promote_from_sealed"),
        False,
        "frozen locked_diagnostic_ablation_set.may_reselect_or_promote_from_sealed",
    )
    for payload, label in (
        (sealed_manifest, "sealed manifest"),
        (sealed_audit, "sealed audit"),
    ):
        if payload.get("locked_diagnostic_ablation_set") != locked:
            raise ValueError(
                f"{label} locked diagnostic candidate authority differs from handoff"
            )

    base_names = tuple(
        dict.fromkeys(
            (
                selected,
                frozen_handoff.STRICT_PT_CONTROL,
                STRICT_LEGACY_CONTROL,
                RETROSPECTIVE_CONTROL,
                *included,
            )
        )
    )
    opener_name = f"{selected}{FROZEN_OPENER_SUFFIX}"
    expected_names = (*base_names, opener_name)
    if len(expected_names) != len(set(expected_names)):
        raise ValueError("reconstructed sealed candidate names are not unique")

    expected_roles = {name: "locked_descriptive_ablation" for name in included}
    expected_roles[selected] = str(handoff.selection_role)
    expected_roles[frozen_handoff.STRICT_PT_CONTROL] = (
        "strict_point_in_time_architecture_control"
    )
    expected_roles[STRICT_LEGACY_CONTROL] = (
        "strict_prior_legacy_input_current_coefficients_control"
    )
    expected_roles[RETROSPECTIVE_CONTROL] = (
        "exact_current_code_retrospective_control"
    )
    expected_roles[opener_name] = "frozen_discovery_opener_blend_sensitivity"

    prediction_input = sealed_audit.get("prediction_input")
    if not isinstance(prediction_input, Mapping):
        raise ValueError("sealed audit lacks frozen prediction candidate authority")
    if prediction_input.get("candidate_names") != list(base_names):
        raise ValueError("sealed prediction-input candidate menu is not frozen exact")
    declared_input_roles = prediction_input.get("candidate_roles")
    expected_input_roles = {name: expected_roles[name] for name in base_names}
    if declared_input_roles != expected_input_roles:
        raise ValueError("sealed prediction-input candidate roles are not frozen exact")
    if (
        str(sealed_manifest.get("selected_model", "")) != selected
        or str(sealed_manifest.get("selected_model_role", ""))
        != expected_roles[selected]
        or str(sealed_audit.get("selected_model", "")) != selected
        or str(sealed_audit.get("selected_model_role", ""))
        != expected_roles[selected]
    ):
        raise ValueError("sealed selected candidate name/role is not frozen exact")

    for payload, label in (
        (sealed_manifest.get("opener_blend_sensitivity"), "sealed manifest"),
        (sealed_audit.get("opener_blend_sensitivity"), "sealed audit"),
    ):
        if not isinstance(payload, Mapping):
            raise ValueError(f"{label} lacks the frozen opener sensitivity")
        if (
            str(payload.get("candidate", "")) != opener_name
            or str(payload.get("source_candidate", "")) != selected
        ):
            raise ValueError(f"{label} opener sensitivity identity is false")
        cutoff = pd.to_datetime(payload.get("selection_window_end"), errors="coerce")
        if pd.isna(cutoff) or pd.Timestamp(cutoff).normalize() >= SEALED_START:
            raise ValueError(f"{label} opener sensitivity is not discovery-frozen")
        _require_exact_bool(
            payload.get("sealed_data_used_for_selection"),
            False,
            f"{label} opener sealed_data_used_for_selection",
        )
        _require_exact_bool(
            payload.get("neutral_base_redefined"),
            False,
            f"{label} opener neutral_base_redefined",
        )

    for payload, key, label in (
        (sealed_manifest, "evaluated_candidate_names", "sealed manifest"),
        (sealed_audit, "evaluated_candidate_names", "sealed audit"),
    ):
        names = payload.get(key)
        if not isinstance(names, list) or tuple(map(str, names)) != expected_names:
            raise ValueError(f"{label} evaluated candidate menu is not frozen exact")

    if metadata["candidate"].astype(str).duplicated().any():
        raise ValueError("sealed candidate metadata has duplicate candidates")
    if set(metadata["candidate"].astype(str)) != set(expected_names):
        raise ValueError("sealed metadata candidate menu is not frozen exact")
    _require(
        metadata,
        [
            "candidate",
            "candidate_role",
            "sealed_selection_eligible",
            "sealed_result_can_reselect",
            *SEALED_METADATA_CATALOG_COLUMNS,
        ],
        "sealed candidate metadata",
    )
    for column in (
        "sealed_selection_eligible",
        "sealed_result_can_reselect",
        "control",
        "sensitivity_only",
        "selection_eligible",
    ):
        _require_exact_boolean_series(metadata[column], f"sealed metadata {column}")
    if metadata["sealed_selection_eligible"].any() or metadata[
        "sealed_result_can_reselect"
    ].any():
        raise ValueError("sealed candidate metadata permits sealed selection")

    catalog = pd.read_csv(Path(handoff.candidate_catalog_path).resolve())
    _require(
        catalog,
        ["name", *SEALED_CATALOG_SIGNATURE_COLUMNS, *SEALED_METADATA_CATALOG_COLUMNS],
        "hash-bound frozen candidate catalog",
    )
    if catalog["name"].astype(str).duplicated().any():
        raise ValueError("hash-bound frozen candidate catalog names are ambiguous")
    catalog = catalog.loc[catalog["name"].astype(str).isin(base_names)].copy()
    if set(catalog["name"].astype(str)) != set(base_names):
        missing = sorted(set(base_names) - set(catalog["name"].astype(str)))
        raise ValueError(f"prescribed sealed candidates absent from frozen catalog: {missing}")
    by_name = catalog.set_index(catalog["name"].astype(str), drop=False)

    authoritative_rows: list[dict[str, Any]] = []
    for candidate in base_names:
        catalog_row = by_name.loc[candidate]
        metadata_row = metadata.loc[metadata["candidate"].astype(str).eq(candidate)].iloc[0]
        if str(metadata_row["candidate_role"]) != expected_roles[candidate]:
            raise ValueError(f"sealed candidate role is forged for {candidate!r}")
        for column in SEALED_METADATA_CATALOG_COLUMNS:
            if not _catalog_scalar_equal(metadata_row[column], catalog_row[column]):
                raise ValueError(
                    f"sealed candidate metadata differs from frozen catalog: "
                    f"candidate={candidate!r}, column={column!r}"
                )
        record = catalog_row.to_dict()
        record["candidate"] = candidate
        record["candidate_role"] = expected_roles[candidate]
        authoritative_rows.append(record)

    opener = metadata.loc[metadata["candidate"].astype(str).eq(opener_name)].iloc[0]
    selected_catalog = by_name.loc[selected]
    if str(opener["candidate_role"]) != expected_roles[opener_name]:
        raise ValueError("sealed opener sensitivity role is forged")
    if str(opener.get("source_candidate", "")) != selected or str(
        opener.get("sealed_sensitivity_type", "")
    ) != "frozen_opener_market_blend":
        raise ValueError("sealed opener sensitivity metadata has false lineage")
    for column in SEALED_METADATA_CATALOG_COLUMNS:
        expected = selected_catalog[column]
        if column == "control":
            expected = False
        elif column == "sensitivity_only":
            expected = True
        elif column == "selection_eligible":
            expected = False
        if not _catalog_scalar_equal(opener[column], expected):
            raise ValueError(
                f"sealed opener metadata differs from its frozen source: {column!r}"
            )
    opener_record = selected_catalog.to_dict()
    opener_record.update(
        {
            "candidate": opener_name,
            "candidate_role": expected_roles[opener_name],
            "source_candidate": selected,
            "sealed_sensitivity_type": "frozen_opener_market_blend",
            "control": False,
            "sensitivity_only": True,
            "selection_eligible": False,
        }
    )
    authoritative_rows.append(opener_record)
    return pd.DataFrame(authoritative_rows), expected_names, expected_roles


def validate_sealed_prediction_and_mapper_chronology(
    predictions: pd.DataFrame,
    mapper: pd.DataFrame,
    logical: pd.DataFrame,
    sealed_audit: Mapping[str, Any],
    *,
    expected_names: Sequence[str],
    expected_roles: Mapping[str, str],
    selected_name: str,
    control_names: Iterable[str] = (),
) -> dict[str, Any]:
    """Recompute strict pre-event prediction and pre-sealed mapper boundaries."""

    opener_name = f"{selected_name}{FROZEN_OPENER_SUFFIX}"
    base_names = tuple(name for name in expected_names if name != opener_name)
    _require(
        predictions,
        [
            "candidate",
            "event_start_date",
            "prediction_asof",
            "training_max_event_end",
            "chronology_ok",
        ],
        "hash-bound source predictions",
    )
    source = predictions.copy()
    for column in ("event_start_date", "prediction_asof", "training_max_event_end"):
        source[column] = pd.to_datetime(source[column], errors="coerce").dt.normalize()
    source = source.loc[
        source["candidate"].astype(str).isin(base_names)
        & source["event_start_date"].between(SEALED_START, SEALED_END)
    ].copy()
    if source.empty or set(source["candidate"].astype(str)) != set(base_names):
        raise ValueError("hash-bound source prediction candidate menu is incomplete")
    chronology_ok = _require_exact_boolean_series(
        source["chronology_ok"], "hash-bound source predictions chronology_ok"
    )
    control_set = set(map(str, control_names))
    if not control_set.issubset(set(base_names)):
        raise ValueError("sealed chronology control menu is not a candidate subset")
    control_rows = source["candidate"].astype(str).isin(control_set)
    training_present = source["training_max_event_end"].notna()
    chronology_bad = (
        ~chronology_ok
        | source[["event_start_date", "prediction_asof"]].isna().any(axis=1)
        # Direct control rows have no fitted training sample, so their
        # training_max_event_end is intentionally null.  Fitted candidate
        # rows must always carry a strict pre-event training boundary.
        | (~control_rows & ~training_present)
        | source["prediction_asof"].gt(source["event_start_date"])
        | (
            training_present
            & source["training_max_event_end"].ge(source["event_start_date"])
        )
        | (
            training_present
            & source["training_max_event_end"].gt(source["prediction_asof"])
        )
    )
    if chronology_bad.any():
        raise ValueError(
            "sealed prediction max training dates are not strictly pre-event"
        )

    _require(
        mapper,
        [
            "candidate",
            "market",
            "fit_end",
            "sigma",
            "tie_threshold",
            "frozen_before_sealed_scoring",
        ],
        "frozen probability mapper",
    )
    frozen_mapper = mapper.copy()
    frozen_mapper["fit_end"] = pd.to_datetime(
        frozen_mapper["fit_end"], errors="coerce"
    ).dt.normalize()
    frozen_flag = _require_exact_boolean_series(
        frozen_mapper["frozen_before_sealed_scoring"],
        "frozen probability mapper frozen_before_sealed_scoring",
    )
    frozen_mapper["sigma"] = pd.to_numeric(frozen_mapper["sigma"], errors="coerce")
    frozen_mapper["tie_threshold"] = pd.to_numeric(
        frozen_mapper["tie_threshold"], errors="coerce"
    )
    expected_mapper_keys = pd.MultiIndex.from_product(
        [base_names, EXPECTED_SEALED_MARKETS], names=["candidate", "market"]
    )
    actual_mapper_keys = pd.MultiIndex.from_frame(
        frozen_mapper[["candidate", "market"]].astype(str)
    )
    if (
        len(frozen_mapper) != len(expected_mapper_keys)
        or frozen_mapper.duplicated(["candidate", "market"]).any()
        or set(actual_mapper_keys) != set(expected_mapper_keys)
    ):
        raise ValueError("frozen probability mapper candidate/market menu is not exact")
    mapper_bad = (
        ~frozen_flag
        | frozen_mapper["fit_end"].isna()
        | frozen_mapper["fit_end"].ge(SEALED_START)
        | ~np.isfinite(frozen_mapper["sigma"])
        | frozen_mapper["sigma"].le(0.0)
        | ~np.isfinite(frozen_mapper["tie_threshold"])
        | frozen_mapper["tie_threshold"].lt(0.0)
    )
    if mapper_bad.any():
        raise ValueError("frozen probability mapper contains a future/invalid fit")
    mapper_fit_max = pd.Timestamp(frozen_mapper["fit_end"].max()).normalize()
    declared_fit_max = pd.to_datetime(
        sealed_audit.get("probability_mapper_fit_end_max"), errors="coerce"
    )
    if (
        pd.isna(declared_fit_max)
        or pd.Timestamp(declared_fit_max).normalize() >= SEALED_START
        or pd.Timestamp(declared_fit_max).normalize() != mapper_fit_max
    ):
        raise ValueError(
            "sealed audit probability_mapper_fit_end_max is false or post-discovery"
        )

    if set(logical["candidate"].astype(str)) != set(expected_names):
        raise ValueError("sealed logical candidate menu is not frozen exact")
    role_map = logical[["candidate", "candidate_role"]].drop_duplicates()
    if role_map["candidate"].duplicated().any() or set(
        role_map["candidate"].astype(str)
    ) != set(expected_names):
        raise ValueError("sealed logical candidate roles are missing or ambiguous")
    for row in role_map.to_dict(orient="records"):
        candidate = str(row["candidate"])
        if str(row["candidate_role"]) != str(expected_roles[candidate]):
            raise ValueError(f"sealed logical candidate role is forged for {candidate!r}")

    expected_fit = frozen_mapper[["candidate", "market", "fit_end"]].copy()
    opener_fit = expected_fit.loc[
        expected_fit["candidate"].astype(str).eq(selected_name)
    ].copy()
    opener_fit["candidate"] = opener_name
    expected_fit = pd.concat([expected_fit, opener_fit], ignore_index=True)
    checked = logical.merge(
        expected_fit.rename(columns={"fit_end": "expected_probability_fit_end"}),
        on=["candidate", "market"],
        how="left",
        validate="many_to_one",
    )
    if (
        checked["expected_probability_fit_end"].isna().any()
        or checked["probability_fit_end"].isna().any()
        or not checked["probability_fit_end"].eq(
            checked["expected_probability_fit_end"]
        ).all()
    ):
        raise ValueError("sealed logical probability mapper lineage is false")
    return {
        "source_prediction_rows": int(len(source)),
        "prediction_training_max": source.loc[
            training_present, "training_max_event_end"
        ].max().date().isoformat(),
        "fitted_candidate_prediction_rows": int((~control_rows).sum()),
        "control_prediction_rows": int(control_rows.sum()),
        "control_rows_without_fitted_training_boundary": int(
            (control_rows & ~training_present).sum()
        ),
        "probability_mapper_fit_end_max": mapper_fit_max.date().isoformat(),
        "prediction_training_strictly_pre_event": True,
        "direct_controls_may_omit_fitted_training_boundary": True,
        "logical_mapper_lineage_exact": True,
    }


def validate_sealed_candidate_chronology_bundle(
    handoff: frozen_handoff.FrozenHandoff,
    sealed_manifest: Mapping[str, Any],
    sealed_audit: Mapping[str, Any],
    metadata: pd.DataFrame,
    predictions: pd.DataFrame,
    mapper: pd.DataFrame,
    logical: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[str, ...], dict[str, str], dict[str, Any]]:
    """One fail-closed pre-analysis gate for candidate identity and chronology."""

    authoritative, names, roles = reconstruct_and_validate_sealed_candidate_authority(
        handoff, sealed_manifest, sealed_audit, metadata
    )
    control_flags = _require_exact_boolean_series(
        authoritative["control"], "authoritative sealed candidate control"
    )
    chronology = validate_sealed_prediction_and_mapper_chronology(
        predictions,
        mapper,
        logical,
        sealed_audit,
        expected_names=names,
        expected_roles=roles,
        selected_name=str(handoff.selected_name),
        control_names=authoritative.loc[control_flags, "candidate"].astype(str),
    )
    return authoritative, names, roles, chronology


def build_w1_pairing_audit(
    metadata: pd.DataFrame, *, selected_model: str | None = None
) -> pd.DataFrame:
    """Match each W1 arm to a unique structurally identical W0 arm.

    Weather and target labels are deliberately excluded from the structural
    signature.  All other persisted catalog fields are held fixed.  A missing
    or ambiguous reference is reported and never guessed.
    """

    required = [
        "candidate",
        "family",
        "source",
        "weather_arm",
        "target_arm",
        "refit_schedule",
        "alpha",
        "short_game_penalty",
    ]
    _require(metadata, required, "sealed candidate metadata")
    frame = metadata.copy()
    if frame["candidate"].astype(str).duplicated().any():
        raise ValueError("sealed candidate metadata has duplicate candidates")
    if "sealed_sensitivity_type" in frame:
        frame = frame.loc[frame["sealed_sensitivity_type"].isna()].copy()
    # When called from run(), ``metadata`` is the independently reconstructed
    # frozen catalog frame, so all catalogued hyperparameters participate in
    # the signature.  The five core fields remain the focused-test minimum.
    signature_columns = [
        column
        for column in SEALED_CATALOG_SIGNATURE_COLUMNS
        if column in frame and column not in {"weather_arm", "target_arm"}
    ]
    for column in signature_columns:
        frame[f"_sig_{column}"] = frame[column].fillna("__missing__").astype(str)
    signature = [f"_sig_{column}" for column in signature_columns]
    w0 = frame.loc[
        frame["weather_arm"].astype(str).str.casefold().eq("w0")
        & frame["target_arm"].astype(str).str.casefold().eq("observed")
    ].copy()
    w1 = frame.loc[
        frame["weather_arm"].astype(str).str.casefold().str.startswith("w1")
    ].copy()
    if w1.empty:
        return pd.DataFrame(
            columns=[
                "w1_candidate",
                "w1_weather_arm",
                "w1_target_arm",
                "w0_candidate",
                "pairing_status",
                "w0_match_count",
                "selected_model_in_pair",
                *signature_columns,
            ]
        )
    rows: list[dict[str, Any]] = []
    for row in w1.to_dict(orient="records"):
        matches = w0.copy()
        for column in signature:
            matches = matches.loc[matches[column].eq(row[column])]
        status = "paired" if len(matches) == 1 else (
            "missing_w0_reference" if matches.empty else "ambiguous_w0_reference"
        )
        rows.append(
            {
                "w1_candidate": str(row["candidate"]),
                "w1_weather_arm": str(row["weather_arm"]),
                "w1_target_arm": str(row["target_arm"]),
                "w0_candidate": (
                    str(matches["candidate"].iloc[0]) if len(matches) == 1 else pd.NA
                ),
                "pairing_status": status,
                "w0_match_count": int(len(matches)),
                "selected_model_in_pair": bool(
                    selected_model is not None
                    and str(row["candidate"]) == str(selected_model)
                ),
                **{column: row[column] for column in signature_columns},
            }
        )
    output = pd.DataFrame(rows)
    if not output["pairing_status"].eq("paired").all():
        failures = output.loc[
            ~output["pairing_status"].eq("paired"),
            ["w1_candidate", "pairing_status", "w0_match_count"],
        ].to_dict(orient="records")
        raise ValueError(
            "every W1 candidate requires exactly one structurally matched W0 arm; "
            f"failures={failures}"
        )
    selected_rows = output.loc[output["selected_model_in_pair"]]
    if not selected_rows.empty and not selected_rows["pairing_status"].eq("paired").all():
        raise ValueError("selected W1 model lacks an unambiguous W0 reference")
    return output


def normalize_sealed_event_keys(
    frame: pd.DataFrame, event_identity: pd.DataFrame, *, label: str
) -> pd.DataFrame:
    """Map either archived canonical or local sealed keys to the local panel key."""

    _require(
        event_identity,
        ["local_event_key", "canonical_event_key"],
        "sealed event weather identity",
    )
    identity = event_identity[["local_event_key", "canonical_event_key"]].drop_duplicates()
    if identity.isna().any().any():
        raise ValueError("sealed event identity contains null keys")
    if identity["local_event_key"].duplicated().any() or identity[
        "canonical_event_key"
    ].duplicated().any():
        raise ValueError("sealed event identity is not one-to-one")
    lookup = pd.concat(
        [
            identity[["local_event_key"]]
            .rename(columns={"local_event_key": "source_event_key"})
            .assign(local_event_key=identity["local_event_key"].to_numpy()),
            identity[["canonical_event_key"]]
            .rename(columns={"canonical_event_key": "source_event_key"})
            .assign(local_event_key=identity["local_event_key"].to_numpy()),
        ],
        ignore_index=True,
    )
    if lookup["source_event_key"].duplicated().any():
        raise ValueError("local and canonical event-key namespaces collide")
    source_event_key = frame["event_key"].astype(str)
    local_lookup = lookup.set_index("source_event_key")["local_event_key"]
    mapped_local_event_key = source_event_key.map(local_lookup)
    if mapped_local_event_key.isna().any():
        examples = sorted(source_event_key.loc[mapped_local_event_key.isna()].unique())[:10]
        raise ValueError(f"{label} contains unmapped sealed event keys: {examples}")
    mapped_local_event_key = mapped_local_event_key.astype(str)
    output = frame.copy()
    if "local_event_key" in output:
        if output["local_event_key"].isna().any() or not output[
            "local_event_key"
        ].astype(str).eq(mapped_local_event_key).all():
            raise ValueError(f"{label} producer local_event_key disagrees with identity")
        # Preserve the historical output schema: this key was formerly the
        # private merge result and was consumed into event_key.
        output = output.drop(columns="local_event_key")
    if "canonical_event_key" in output:
        canonical_by_local = identity.set_index("local_event_key")[
            "canonical_event_key"
        ]
        expected_canonical = mapped_local_event_key.map(canonical_by_local).astype(str)
        if output["canonical_event_key"].isna().any() or not output[
            "canonical_event_key"
        ].astype(str).eq(expected_canonical).all():
            raise ValueError(
                f"{label} producer canonical_event_key disagrees with identity"
            )
    if "source_sealed_event_key" in output:
        preserved_source = output["source_sealed_event_key"].astype(str)
        if preserved_source.map(local_lookup).isna().any() or not preserved_source.map(
            local_lookup
        ).astype(str).eq(mapped_local_event_key).all():
            raise ValueError(
                f"{label} prior source_sealed_event_key disagrees with identity"
            )
    else:
        output["source_sealed_event_key"] = source_event_key
    output["event_key"] = mapped_local_event_key
    return output


def _unique_hash_bound_input(
    paths: Iterable[Path], *, filename: str, label: str
) -> Path:
    matches = {
        Path(path).resolve()
        for path in paths
        if Path(path).name.casefold() == filename.casefold()
    }
    if len(matches) != 1:
        raise ValueError(
            f"{label} requires exactly one execution-provenance-bound {filename}; "
            f"found={sorted(map(str, matches))}"
        )
    return next(iter(matches))


def build_sealed_price_quote_universe(
    quotes: pd.DataFrame,
    event_identity: pd.DataFrame,
    *,
    expected_price_bases: Sequence[str],
    expected_line_policies: Sequence[str],
    expected_books: Sequence[str],
) -> pd.DataFrame:
    """Build price-specific eligible events from the sealed hash-bound quote cache."""

    price_bases = tuple(map(str, expected_price_bases))
    line_policies = tuple(map(str, expected_line_policies))
    books = tuple(map(str, expected_books))
    if (
        not price_bases
        or not line_policies
        or not books
        or set(line_policies) != {*books, "best_of_three"}
        or len(set(price_bases)) != len(price_bases)
        or len(set(line_policies)) != len(line_policies)
    ):
        raise ValueError("sealed declared price/book/line-policy menu is malformed")
    required = [
        "mapped_event_key",
        "event_start_date",
        "market",
        "book",
        "pair_key",
        *[
            column
            for price in price_bases
            for column in (f"p1_{price}", f"p2_{price}")
        ],
    ]
    # Open-price evaluation additionally requires an executable same-book
    # close, matching the sealed evaluator's CLV/executability contract.
    if "open" in price_bases:
        required.extend(["p1_close", "p2_close"])
    required = list(dict.fromkeys(required))
    _require(quotes, required, "sealed normalized quote cache")
    frame = quotes[required].copy()
    frame["event_start_date"] = pd.to_datetime(
        frame["event_start_date"], errors="coerce"
    ).dt.normalize()
    if frame[
        ["mapped_event_key", "event_start_date", "market", "book", "pair_key"]
    ].isna().any().any():
        raise ValueError("sealed normalized quote cache has invalid canonical keys")
    frame = frame.loc[
        frame["event_start_date"].between(SEALED_START, SEALED_END)
        & frame["book"].astype(str).isin(books)
    ].copy()
    identity = ["mapped_event_key", "market", "book", "pair_key"]
    if frame.empty or frame.duplicated(identity).any():
        raise ValueError("sealed normalized quote cache is empty or ambiguous")
    observed_books = set(frame["book"].astype(str))
    if observed_books != set(books):
        raise ValueError("sealed normalized quote cache book menu is incomplete")
    rows: list[pd.DataFrame] = []
    for price in price_bases:
        valid = _valid_american_odds(frame[f"p1_{price}"]) & _valid_american_odds(
            frame[f"p2_{price}"]
        )
        if price == "open":
            valid &= _valid_american_odds(frame["p1_close"]) & _valid_american_odds(
                frame["p2_close"]
            )
        for policy in line_policies:
            eligible = frame.loc[valid].copy()
            if policy != "best_of_three":
                eligible = eligible.loc[eligible["book"].astype(str).eq(policy)]
            events = eligible[
                ["mapped_event_key", "event_start_date", "market"]
            ].drop_duplicates()
            if events.empty:
                raise ValueError(
                    "sealed declared price/line-policy has no executable events: "
                    f"price={price}, policy={policy}"
                )
            events = events.rename(columns={"mapped_event_key": "event_key"})
            events["price_basis"] = price
            events["line_policy"] = policy
            rows.append(events)
    universe = pd.concat(rows, ignore_index=True)
    universe = normalize_sealed_event_keys(
        universe, event_identity, label="sealed price-specific quote universe"
    )
    identity = [
        "event_key",
        "event_start_date",
        "market",
        "price_basis",
        "line_policy",
    ]
    if universe[identity].isna().any().any() or universe.duplicated(identity).any():
        raise ValueError("sealed price-specific quote universe is ambiguous")
    return universe


def validate_sealed_logical_quote_authority(
    quotes: pd.DataFrame,
    exclusions: pd.DataFrame,
    predictions: pd.DataFrame,
    logical: pd.DataFrame,
    *,
    expected_candidates: Sequence[str],
    selected_name: str,
    expected_books: Sequence[str] = EXPECTED_BOOKS,
    expected_markets: Sequence[str] = EXPECTED_SEALED_MARKETS,
) -> dict[str, Any]:
    """Require logical probabilities to equal offered executable quotes minus exclusions."""

    quote_columns = [
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
    ]
    _require(quotes, quote_columns, "sealed normalized logical quote authority")
    offered = quotes[quote_columns].copy()
    offered["event_start_date"] = pd.to_datetime(
        offered["event_start_date"], errors="coerce"
    ).dt.normalize()
    offered["book"] = offered["book"].astype(str).str.strip().str.casefold()
    offered["market"] = offered["market"].astype(str).str.strip().str.casefold()
    offered = offered.loc[
        offered["event_start_date"].between(SEALED_START, SEALED_END)
        & offered["book"].isin(tuple(expected_books))
        & offered["market"].isin(tuple(expected_markets))
    ].copy()
    key_columns = ["mapped_event_key", "event_start_date", "market", "book", "pair_key"]
    if (
        offered.empty
        or offered[["quote_id", *key_columns, "p1_id", "p2_id"]].isna().any().any()
        or offered["quote_id"].astype(str).duplicated().any()
        or offered.duplicated(key_columns).any()
        or set(offered["book"].astype(str)) != set(map(str, expected_books))
        or set(offered["market"].astype(str)) != set(map(str, expected_markets))
    ):
        raise ValueError("sealed normalized logical quote authority is incomplete or ambiguous")
    executable = _valid_american_odds(offered["p1_close"]) & _valid_american_odds(
        offered["p2_close"]
    )
    # Open is a sensitivity only when the same-book close also exists, so every
    # logical row admitted to money must have an executable close somewhere.
    executable |= (
        _valid_american_odds(offered["p1_open"])
        & _valid_american_odds(offered["p2_open"])
        & _valid_american_odds(offered["p1_close"])
        & _valid_american_odds(offered["p2_close"])
    )
    executable_offered = offered.loc[executable].copy()
    if executable_offered.empty:
        raise ValueError("sealed normalized quote authority has no executable quotes")

    _require(
        predictions,
        ["candidate", "event_key", "dg_id_key", "event_start_date"],
        "hash-bound prediction availability",
    )
    availability = predictions.copy()
    availability["event_start_date"] = pd.to_datetime(
        availability["event_start_date"], errors="coerce"
    ).dt.normalize()
    availability = availability.loc[
        availability["event_start_date"].between(SEALED_START, SEALED_END)
    ].copy()
    opener_name = f"{selected_name}{FROZEN_OPENER_SUFFIX}"
    selected_availability = availability.loc[
        availability["candidate"].astype(str).eq(selected_name)
    ].copy()
    selected_availability["candidate"] = opener_name
    availability = pd.concat(
        [availability, selected_availability], ignore_index=True, sort=False
    )
    availability = availability.loc[
        availability["candidate"].astype(str).isin(expected_candidates)
    ].copy()
    availability["event_key"] = availability["event_key"].astype(str)
    availability["dg_id_key"] = availability["dg_id_key"].astype(str)
    if (
        set(availability["candidate"].astype(str)) != set(expected_candidates)
        or availability.duplicated(["candidate", "event_key", "dg_id_key"]).any()
    ):
        raise ValueError("sealed prediction availability menu is incomplete or ambiguous")
    quote_event_dates = offered[
        ["mapped_event_key", "event_start_date"]
    ].drop_duplicates()
    if quote_event_dates["mapped_event_key"].astype(str).duplicated().any():
        raise ValueError("sealed normalized quotes map an event to multiple dates")
    prediction_event_dates = availability[
        ["candidate", "event_key", "event_start_date"]
    ].drop_duplicates()
    if prediction_event_dates.duplicated(["candidate", "event_key"]).any():
        raise ValueError("sealed source predictions map an event to multiple dates")
    date_check = prediction_event_dates.merge(
        quote_event_dates.rename(columns={"mapped_event_key": "event_key"}),
        on="event_key",
        how="inner",
        validate="many_to_one",
        suffixes=("_prediction", "_quote"),
    )
    if date_check.empty or not date_check["event_start_date_prediction"].eq(
        date_check["event_start_date_quote"]
    ).all():
        raise ValueError(
            "sealed source-prediction event dates differ from normalized quotes"
        )

    expected_missing_parts: list[pd.DataFrame] = []
    expected_logical_parts: list[pd.DataFrame] = []
    logical_keys = ["mapped_event_key", "event_start_date", "market", "pair_key"]
    for candidate in expected_candidates:
        keys = availability.loc[
            availability["candidate"].astype(str).eq(str(candidate)),
            ["event_key", "dg_id_key"],
        ].drop_duplicates()
        left = keys.rename(
            columns={"event_key": "mapped_event_key", "dg_id_key": "p1_id"}
        ).assign(p1_available=True)
        right = keys.rename(
            columns={"event_key": "mapped_event_key", "dg_id_key": "p2_id"}
        ).assign(p2_available=True)
        checked = offered.merge(
            left,
            on=["mapped_event_key", "p1_id"],
            how="left",
            validate="many_to_one",
        ).merge(
            right,
            on=["mapped_event_key", "p2_id"],
            how="left",
            validate="many_to_one",
        )
        checked["p1_available"] = checked["p1_available"].eq(True)
        checked["p2_available"] = checked["p2_available"].eq(True)
        missing = checked.loc[
            ~(checked["p1_available"] & checked["p2_available"])
        ].copy()
        if not missing.empty:
            missing["candidate"] = str(candidate)
            missing["missing_reason"] = np.select(
                [
                    ~missing["p1_available"] & ~missing["p2_available"],
                    ~missing["p1_available"],
                    ~missing["p2_available"],
                ],
                ["both_players_absent", "p1_absent", "p2_absent"],
                default="unknown",
            )
            expected_missing_parts.append(
                missing[
                    [
                        "candidate",
                        "quote_id",
                        *logical_keys,
                        "book",
                        "p1_id",
                        "p2_id",
                        "missing_reason",
                    ]
                ]
            )

        executable_ids = set(executable_offered["quote_id"].astype(str))
        covered = checked.loc[
            checked["quote_id"].astype(str).isin(executable_ids)
            & checked["p1_available"]
            & checked["p2_available"],
            logical_keys,
        ].drop_duplicates()
        covered["candidate"] = str(candidate)
        expected_logical_parts.append(covered[["candidate", *logical_keys]])

    exclusion_columns = [
        "candidate",
        "quote_id",
        *logical_keys,
        "book",
        "p1_id",
        "p2_id",
        "missing_reason",
    ]
    _require(exclusions, exclusion_columns, "sealed hash-bound exclusion ledger")
    declared_exclusions = exclusions[exclusion_columns].copy()
    declared_exclusions["event_start_date"] = pd.to_datetime(
        declared_exclusions["event_start_date"], errors="coerce"
    ).dt.normalize()
    expected_exclusions = (
        pd.concat(expected_missing_parts, ignore_index=True)
        if expected_missing_parts
        else pd.DataFrame(columns=exclusion_columns)
    )
    exclusion_identity = ["candidate", "quote_id"]
    if (
        declared_exclusions.duplicated(exclusion_identity).any()
        or expected_exclusions.duplicated(exclusion_identity).any()
    ):
        raise ValueError("sealed exclusion ledger is ambiguous")
    compared_exclusions = expected_exclusions.merge(
        declared_exclusions,
        on=exclusion_identity,
        how="outer",
        validate="one_to_one",
        suffixes=("_expected", "_declared"),
        indicator=True,
    )
    if not compared_exclusions["_merge"].eq("both").all():
        raise ValueError("sealed exclusion ledger does not equal prediction-derived exclusions")
    for column in exclusion_columns[2:]:
        expected_column = compared_exclusions[f"{column}_expected"]
        declared_column = compared_exclusions[f"{column}_declared"]
        if not expected_column.fillna("__missing__").astype(str).eq(
            declared_column.fillna("__missing__").astype(str)
        ).all():
            raise ValueError(f"sealed exclusion ledger has false {column!r} values")

    expected_logical = pd.concat(expected_logical_parts, ignore_index=True).rename(
        columns={"mapped_event_key": "event_key"}
    )
    canonical_logical_keys = ["event_key", "event_start_date", "market", "pair_key"]
    actual_logical = logical[["candidate", *canonical_logical_keys]].copy()
    logical_identity = ["candidate", *canonical_logical_keys]
    if (
        expected_logical.duplicated(logical_identity).any()
        or actual_logical.duplicated(logical_identity).any()
    ):
        raise ValueError("sealed logical quote authority is ambiguous")
    compared_logical = expected_logical.merge(
        actual_logical,
        on=logical_identity,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not compared_logical["_merge"].eq("both").all():
        raise ValueError(
            "sealed canonical logical menu does not equal executable offered-minus-excluded"
        )
    return {
        "normalized_offered_quote_rows": int(len(offered)),
        "executable_offered_quote_rows": int(len(executable_offered)),
        "validated_exclusion_rows": int(len(expected_exclusions)),
        "authoritative_candidate_logical_rows": int(len(expected_logical)),
        "offered_minus_excluded_exact": True,
    }


def _canonical_logical(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    required = [
        "candidate",
        "candidate_role",
        "mapped_event_key",
        "event_start_date",
        "market",
        "pair_key",
        "canonical_outcome",
        "canonical_probability",
        "canonical_push_probability",
        "probability_fit_end",
    ]
    _require(frame, required, label)
    output = frame[required].copy().rename(columns={"mapped_event_key": "event_key"})
    output["event_start_date"] = pd.to_datetime(
        output["event_start_date"], errors="coerce"
    ).dt.normalize()
    output["probability_fit_end"] = pd.to_datetime(
        output["probability_fit_end"], errors="coerce"
    ).dt.normalize()
    _numeric(
        output,
        [
            "canonical_outcome",
            "canonical_probability",
            "canonical_push_probability",
        ],
    )
    key = ["candidate", "event_key", "market", "pair_key"]
    if (
        output[key].isna().any().any()
        or output[["event_start_date", "probability_fit_end"]].isna().any().any()
        or output.duplicated(key).any()
    ):
        raise ValueError(f"{label} has null or duplicate canonical keys")
    numeric = output[
        [
            "canonical_outcome",
            "canonical_probability",
            "canonical_push_probability",
        ]
    ].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{label} contains non-finite outcomes or probabilities")
    valid_outcome = output["canonical_outcome"].isin([0.0, 0.5, 1.0])
    probability_bad = (
        ~output["canonical_probability"].between(0.0, 1.0, inclusive="both")
        | ~output["canonical_push_probability"].between(
            0.0, 1.0, inclusive="both"
        )
        | (
            output["canonical_probability"]
            + output["canonical_push_probability"]
        ).gt(1.0 + 1e-10)
    )
    if (~valid_outcome).any() or probability_bad.any():
        raise ValueError(f"{label} contains out-of-domain outcomes or probabilities")
    if (
        output["canonical_outcome"].isin([0.0, 1.0])
        & (1.0 - output["canonical_push_probability"]).le(EPS)
    ).any():
        raise ValueError(f"{label} has zero resolved mass for a resolved outcome")
    if output["probability_fit_end"].ge(SEALED_START).any():
        raise ValueError(f"{label} contains a post-discovery probability mapper")
    return output


def _exact_menu(
    frame: pd.DataFrame,
    left_name: str,
    right_name: str,
    *,
    model_column: str,
    keys: Sequence[str],
) -> tuple[bool, int, int, int]:
    left = set(
        map(
            tuple,
            frame.loc[frame[model_column].eq(left_name), list(keys)]
            .astype(str)
            .itertuples(index=False, name=None),
        )
    )
    right = set(
        map(
            tuple,
            frame.loc[frame[model_column].eq(right_name), list(keys)]
            .astype(str)
            .itertuples(index=False, name=None),
        )
    )
    return left == right and bool(left), len(left), len(left - right), len(right - left)


def audit_w1_quote_menus(
    pairing: pd.DataFrame, logical: pd.DataFrame
) -> pd.DataFrame:
    if pairing.empty:
        output = pairing.copy()
        output["quote_menu_status"] = pd.Series(dtype=str)
        output["w1_quote_keys"] = pd.Series(dtype="int64")
        output["w1_only_quote_keys"] = pd.Series(dtype="int64")
        output["w0_only_quote_keys"] = pd.Series(dtype="int64")
        return output
    keys = ["event_key", "event_start_date", "market", "pair_key"]
    rows: list[dict[str, Any]] = []
    for pair in pairing.to_dict(orient="records"):
        record = dict(pair)
        if pair["pairing_status"] != "paired":
            record.update(
                {
                    "quote_menu_status": "not_checked_unpaired",
                    "w1_quote_keys": 0,
                    "w1_only_quote_keys": 0,
                    "w0_only_quote_keys": 0,
                }
            )
        else:
            exact, count, left_only, right_only = _exact_menu(
                logical,
                str(pair["w1_candidate"]),
                str(pair["w0_candidate"]),
                model_column="candidate",
                keys=keys,
            )
            record.update(
                {
                    "quote_menu_status": "exact" if exact else "mismatch",
                    "w1_quote_keys": count,
                    "w1_only_quote_keys": left_only,
                    "w0_only_quote_keys": right_only,
                }
            )
        rows.append(record)
    output = pd.DataFrame(rows)
    if not output["quote_menu_status"].eq("exact").all():
        failures = output.loc[
            ~output["quote_menu_status"].eq("exact"),
            [
                "w1_candidate",
                "w0_candidate",
                "pairing_status",
                "quote_menu_status",
                "w1_only_quote_keys",
                "w0_only_quote_keys",
            ],
        ].to_dict(orient="records")
        raise ValueError(
            "every W1/W0 arm requires the identical complete quote menu; "
            f"failures={failures}"
        )
    selected = output.loc[output["selected_model_in_pair"]]
    if not selected.empty and not selected["quote_menu_status"].eq("exact").all():
        raise ValueError("selected W1 model and its W0 reference have different quote menus")
    return output


def pairwise_base_proper_deltas(
    logical: pd.DataFrame,
    *,
    challenger: str,
    reference: str,
    analysis_pair: str,
    comparison_kind: str,
) -> pd.DataFrame:
    """Event-equal binary log-loss delta on an exact common canonical menu."""

    keys = ["event_key", "event_start_date", "market", "pair_key"]
    exact, _, _, _ = _exact_menu(
        logical,
        challenger,
        reference,
        model_column="candidate",
        keys=keys,
    )
    if not exact:
        raise ValueError(f"proper-score menu mismatch: {challenger} vs {reference}")

    left_identity = logical.loc[
        logical["candidate"].eq(challenger),
        [*keys, "canonical_outcome"],
    ]
    right_identity = logical.loc[
        logical["candidate"].eq(reference),
        [*keys, "canonical_outcome"],
    ]
    identity_check = left_identity.merge(
        right_identity,
        on=keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_challenger", "_reference"),
        indicator=True,
    )
    if (
        not identity_check["_merge"].eq("both").all()
        or not np.allclose(
            identity_check["canonical_outcome_challenger"],
            identity_check["canonical_outcome_reference"],
            rtol=0.0,
            atol=1e-10,
        )
    ):
        raise ValueError("canonical outcomes or quote keys differ across proper-score pair")

    def prepared(candidate: str, prefix: str) -> pd.DataFrame:
        use = logical.loc[logical["candidate"].eq(candidate)].copy()
        resolved_mass = 1.0 - use["canonical_push_probability"]
        eligible = use["canonical_outcome"].isin([0.0, 1.0]) & resolved_mass.gt(EPS)
        use = use.loc[eligible].copy()
        probability = np.clip(
            use["canonical_probability"] / resolved_mass.loc[eligible],
            EPS,
            1.0 - EPS,
        )
        outcome = use["canonical_outcome"].to_numpy(float)
        use[f"{prefix}_loss"] = -(
            outcome * np.log(probability)
            + (1.0 - outcome) * np.log(1.0 - probability)
        )
        return use[[*keys, "canonical_outcome", f"{prefix}_loss"]]

    left = prepared(challenger, "challenger")
    right = prepared(reference, "reference")
    merged = left.merge(
        right,
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("", "_reference"),
    )
    if not np.allclose(
        merged["canonical_outcome"],
        merged["canonical_outcome_reference"],
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError("canonical outcomes differ across proper-score pair")
    event = merged.groupby(
        ["event_key", "event_start_date", "market"], observed=True
    ).agg(
        common_pairs=("pair_key", "size"),
        challenger_score=("challenger_loss", "mean"),
        reference_score=("reference_loss", "mean"),
    ).reset_index()
    event["event_delta"] = event["challenger_score"] - event["reference_score"]
    event["analysis_block"] = "base_w1_historical_weather_normalization"
    event["analysis_pair"] = analysis_pair
    event["comparison_kind"] = comparison_kind
    event["challenger"] = challenger
    event["reference"] = reference
    event["metric"] = "binary_logloss_delta"
    event["higher_is_better"] = False
    event["round_market"] = event["market"]
    event["weather_information_scope"] = "historical_rounds_only"
    return event


def build_base_proper_deltas(
    logical: pd.DataFrame,
    pairing: pd.DataFrame,
    *,
    control: str,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    eligible = pairing.loc[pairing["quote_menu_status"].eq("exact")]
    for row in eligible.to_dict(orient="records"):
        w1 = str(row["w1_candidate"])
        w0 = str(row["w0_candidate"])
        pair_id = f"{w1}__vs__{w0}"
        pieces.extend(
            [
                pairwise_base_proper_deltas(
                    logical,
                    challenger=w1,
                    reference=w0,
                    analysis_pair=pair_id,
                    comparison_kind="w1_vs_matched_w0",
                ),
                pairwise_base_proper_deltas(
                    logical,
                    challenger=w1,
                    reference=control,
                    analysis_pair=pair_id,
                    comparison_kind="w1_vs_strict_pt_control",
                ),
                pairwise_base_proper_deltas(
                    logical,
                    challenger=w0,
                    reference=control,
                    analysis_pair=pair_id,
                    comparison_kind="w0_vs_strict_pt_control",
                ),
            ]
        )
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _validate_money_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    required = [
        "event_key",
        "event_start_date",
        "candidate",
        "baseline",
        "market",
        *SLICE_COLUMNS,
        "metric",
        "candidate_pnl",
        "baseline_pnl",
        "event_delta",
        "zero_bet_event",
    ]
    _require(frame, required, "sealed money event deltas")
    output = frame[required].copy()
    output["event_start_date"] = pd.to_datetime(
        output["event_start_date"], errors="coerce"
    ).dt.normalize()
    _numeric(output, ["candidate_pnl", "baseline_pnl", "event_delta"])
    _require_finite_numeric_columns(
        output,
        ["candidate_pnl", "baseline_pnl", "event_delta"],
        label="sealed money event deltas",
    )
    if not output["metric"].astype(str).eq("pnl_delta").all():
        raise ValueError("sealed money delta artifact has an unknown metric")
    identity = ["candidate", "event_key", "market", *SLICE_COLUMNS]
    if (
        output[identity].isna().any().any()
        or output["event_start_date"].isna().any()
        or output.duplicated(identity).any()
    ):
        raise ValueError("sealed money deltas have null or duplicate event keys")
    _require_exact_boolean_series(
        output["zero_bet_event"], "sealed money zero_bet_event"
    )
    error = (
        output["candidate_pnl"] - output["baseline_pnl"] - output["event_delta"]
    ).abs()
    if error.max() > 1e-10:
        raise ValueError("sealed money event-delta identity failed")
    return output


def validate_sealed_money_slice_coverage(
    money: pd.DataFrame,
    quote_universe: pd.DataFrame,
    *,
    expected_candidates: Iterable[str],
    baseline_candidate: str,
    expected_price_bases: Sequence[str],
    expected_line_policies: Sequence[str],
    expected_sizings: Sequence[str],
    expected_reliability_policies: Sequence[str],
    expected_markets: Sequence[str] = EXPECTED_SEALED_MARKETS,
    expected_opportunity_modes: Sequence[str] = EXPECTED_SEALED_OPPORTUNITY_MODES,
) -> None:
    """Reject missing slices and event-catalog zero fills without valid prices."""

    candidates = set(map(str, expected_candidates)) - {str(baseline_candidate)}
    observed_candidates = set(money["candidate"].astype(str))
    if not candidates or observed_candidates != candidates:
        raise ValueError("sealed money candidate menu differs from declared menu")
    if not money["baseline"].astype(str).eq(str(baseline_candidate)).all():
        raise ValueError("sealed money baseline is not the exact frozen strict-PT control")
    expected_opportunities = set(map(str, expected_opportunity_modes))
    if not expected_opportunities:
        raise ValueError("sealed expected opportunity-mode menu is empty")
    if set(money["opportunity_mode"].astype(str)) != expected_opportunities:
        raise ValueError("sealed money opportunity-mode menu is incomplete")
    declared = {
        "price_basis": set(map(str, expected_price_bases)),
        "line_policy": set(map(str, expected_line_policies)),
        "sizing": set(map(str, expected_sizings)),
        "reliability_policy": set(map(str, expected_reliability_policies)),
    }
    for column, values in declared.items():
        if not values or set(money[column].astype(str)) != values:
            raise ValueError(f"sealed money {column} menu is incomplete")
    markets = set(quote_universe["market"].astype(str))
    expected_market_set = set(map(str, expected_markets))
    if (
        not markets
        or markets != expected_market_set
        or set(money["market"].astype(str)) != expected_market_set
    ):
        raise ValueError("sealed money market menu differs from executable quotes")

    for candidate in sorted(candidates):
        for opportunity in sorted(expected_opportunities):
            for market in sorted(markets):
                for price in sorted(declared["price_basis"]):
                    for line_policy in sorted(declared["line_policy"]):
                        expected = quote_universe.loc[
                            quote_universe["market"].astype(str).eq(market)
                            & quote_universe["price_basis"].astype(str).eq(price)
                            & quote_universe["line_policy"].astype(str).eq(
                                line_policy
                            ),
                            ["event_key", "event_start_date"],
                        ].drop_duplicates()
                        if expected.empty:
                            raise ValueError("sealed executable quote slice is empty")
                        for sizing in sorted(declared["sizing"]):
                            for reliability in sorted(
                                declared["reliability_policy"]
                            ):
                                actual = money.loc[
                                    money["candidate"].astype(str).eq(candidate)
                                    & money["opportunity_mode"].astype(str).eq(
                                        opportunity
                                    )
                                    & money["market"].astype(str).eq(market)
                                    & money["price_basis"].astype(str).eq(price)
                                    & money["line_policy"].astype(str).eq(
                                        line_policy
                                    )
                                    & money["sizing"].astype(str).eq(sizing)
                                    & money["reliability_policy"].astype(str).eq(
                                        reliability
                                    ),
                                    ["event_key", "event_start_date"],
                                ].drop_duplicates()
                                compared = expected.merge(
                                    actual,
                                    on=["event_key", "event_start_date"],
                                    how="outer",
                                    indicator=True,
                                    validate="one_to_one",
                                )
                                if not compared["_merge"].eq("both").all():
                                    raise ValueError(
                                        "sealed money slice/event universe differs from "
                                        "its executable price-specific quote universe: "
                                        f"candidate={candidate}, opportunity={opportunity}, "
                                        f"market={market}, price={price}, "
                                        f"line_policy={line_policy}, sizing={sizing}, "
                                        f"reliability={reliability}"
                                    )


def filter_sealed_money_to_executable_price_universe(
    money: pd.DataFrame, quote_universe: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Remove only producer-padded, exact-zero rows lacking executable prices."""

    keys = [
        "event_key",
        "event_start_date",
        "market",
        "price_basis",
        "line_policy",
    ]
    _require(
        money,
        [
            *keys,
            "candidate_pnl",
            "baseline_pnl",
            "event_delta",
            "zero_bet_event",
        ],
        "sealed money executable-universe filter",
    )
    _require(quote_universe, keys, "sealed executable quote universe")
    universe = quote_universe[keys].drop_duplicates()
    if universe.empty or universe.duplicated(keys).any():
        raise ValueError("sealed executable quote universe is empty or ambiguous")
    joined = money.merge(
        universe.assign(_executable_price_event=True),
        on=keys,
        how="left",
        validate="many_to_one",
    )
    excluded = joined["_executable_price_event"].isna()
    zero_flag = _require_exact_boolean_series(
        joined["zero_bet_event"], "sealed money zero_bet_event before price filter"
    )
    excluded_values = joined.loc[
        excluded, ["candidate_pnl", "baseline_pnl", "event_delta"]
    ].to_numpy(float)
    if excluded.any() and (
        not np.isfinite(excluded_values).all()
        or not np.isclose(excluded_values, 0.0, atol=1e-12, rtol=0.0).all()
        or not zero_flag.loc[excluded].all()
    ):
        raise ValueError(
            "sealed money row without an executable price is not an exact zero-bet pad"
        )
    filtered = joined.loc[~excluded, money.columns].copy()
    excluded_frame = joined.loc[excluded].copy()
    return filtered, {
        "source_rows": int(len(money)),
        "retained_executable_price_rows": int(len(filtered)),
        "excluded_nonexecutable_zero_pad_rows": int(excluded.sum()),
        "excluded_nonexecutable_event_price_policy_keys": int(
            excluded_frame[keys].drop_duplicates().shape[0]
        ),
        "excluded_nonexecutable_events": int(
            excluded_frame[["event_key", "event_start_date"]].drop_duplicates().shape[0]
        ),
        "all_exclusions_exact_zero_bet_pads": True,
        "excluded_before_money_attribution_and_bootstrap": True,
    }


def build_base_money_deltas(
    money: pd.DataFrame, pairing: pd.DataFrame
) -> pd.DataFrame:
    keys = ["event_key", "event_start_date", "market", *SLICE_COLUMNS]
    pieces: list[pd.DataFrame] = []
    eligible = pairing.loc[pairing["quote_menu_status"].eq("exact")]
    for row in eligible.to_dict(orient="records"):
        w1 = str(row["w1_candidate"])
        w0 = str(row["w0_candidate"])
        left = money.loc[
            money["candidate"].eq(w1)
            & money["opportunity_mode"].eq(
                "pairwise_common_with_strict_pt_baseline"
            )
        ].copy()
        right = money.loc[
            money["candidate"].eq(w0)
            & money["opportunity_mode"].eq(
                "pairwise_common_with_strict_pt_baseline"
            )
        ].copy()
        if left.empty or right.empty:
            raise ValueError(f"missing paired money rows for {w1} / {w0}")
        merged = left.merge(
            right,
            on=keys,
            how="outer",
            validate="one_to_one",
            suffixes=("_w1", "_w0"),
            indicator=True,
        )
        if not merged["_merge"].eq("both").all():
            raise ValueError(f"W1/W0 money event universe differs for {w1} / {w0}")
        if not merged["baseline_w1"].astype(str).eq(
            merged["baseline_w0"].astype(str)
        ).all():
            raise ValueError("W1/W0 money rows use different controls")
        if (merged["baseline_pnl_w1"] - merged["baseline_pnl_w0"]).abs().max() > 1e-10:
            raise ValueError("control P&L differs across exact W1/W0 menus")
        output = merged[keys].copy()
        output["analysis_block"] = "base_w1_historical_weather_normalization"
        output["analysis_pair"] = f"{w1}__vs__{w0}"
        output["model"] = w1
        output["reference"] = w0
        output["control"] = merged["baseline_w1"].astype(str)
        output["model_pnl"] = merged["candidate_pnl_w1"]
        output["reference_pnl"] = merged["candidate_pnl_w0"]
        output["control_pnl"] = merged["baseline_pnl_w1"]
        output["pnl_delta_vs_reference"] = output["model_pnl"] - output["reference_pnl"]
        output["pnl_delta_vs_control"] = output["model_pnl"] - output["control_pnl"]
        output["reference_pnl_delta_vs_control"] = (
            output["reference_pnl"] - output["control_pnl"]
        )
        output["model_zero_bet_event"] = merged["candidate_pnl_w1"].eq(0.0)
        output["reference_zero_bet_event"] = merged["candidate_pnl_w0"].eq(0.0)
        output["control_zero_bet_event"] = merged["baseline_pnl_w1"].eq(0.0)
        output["round_market"] = output["market"].astype(str)
        output["weather_information_scope"] = "historical_rounds_only"
        pieces.append(output)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def build_w2_pairing_audit(
    player_predictions: pd.DataFrame,
    probabilities: pd.DataFrame,
    *,
    expected_candidates: Iterable[str] | None = None,
) -> pd.DataFrame:
    _require(
        player_predictions,
        ["model", "candidate", "variant", "weather_arm"],
        "cascade player predictions",
    )
    mapping = player_predictions[
        ["model", "candidate", "variant", "weather_arm"]
    ].drop_duplicates()
    if mapping["model"].duplicated().any():
        raise ValueError("cascade model maps to multiple candidate/variant/weather arms")
    probability_mapping = probabilities[
        ["model", "candidate", "variant"]
    ].drop_duplicates()
    prediction_mapping = mapping[["model", "candidate", "variant"]]
    mapping_check = prediction_mapping.merge(
        probability_mapping,
        on=["model", "candidate", "variant"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not mapping_check["_merge"].eq("both").all():
        raise ValueError(
            "cascade player-prediction and probability model menus differ"
        )
    rows: list[dict[str, Any]] = []
    for candidate, group in mapping.groupby("candidate", observed=True):
        w0 = group.loc[group["variant"].eq(W2_W0_VARIANTS["w0"])]
        w2 = group.loc[group["variant"].eq(W2_W0_VARIANTS["w2"])]
        if w0.empty and w2.empty:
            continue
        status = "paired" if len(w0) == 1 and len(w2) == 1 else "missing_or_ambiguous_arm"
        record: dict[str, Any] = {
            "candidate": str(candidate),
            "w0_model": str(w0["model"].iloc[0]) if len(w0) == 1 else pd.NA,
            "w2_model": str(w2["model"].iloc[0]) if len(w2) == 1 else pd.NA,
            "w0_count": int(len(w0)),
            "w2_count": int(len(w2)),
            "pairing_status": status,
        }
        if status == "paired":
            if (
                not w0["weather_arm"].astype(str).str.casefold().eq("w0").all()
                or not w2["weather_arm"].astype(str).str.casefold().eq("w2").all()
            ):
                status = "variant_weather_arm_mismatch"
                record["pairing_status"] = status
        if status == "paired":
            exact, count, left_only, right_only = _exact_menu(
                probabilities,
                str(record["w2_model"]),
                str(record["w0_model"]),
                model_column="model",
                keys=[
                    "event_key",
                    "event_start_date",
                    "round_num",
                    "pair_key",
                ],
            )
            record.update(
                {
                    "quote_menu_status": "exact" if exact else "mismatch",
                    "quote_keys": int(count),
                    "w2_only_quote_keys": int(left_only),
                    "w0_only_quote_keys": int(right_only),
                }
            )
        else:
            record.update(
                {
                    "quote_menu_status": "not_checked_unpaired",
                    "quote_keys": 0,
                    "w2_only_quote_keys": 0,
                    "w0_only_quote_keys": 0,
                }
            )
        rows.append(record)
    output = pd.DataFrame(rows)
    if output.empty:
        raise ValueError("cascade output contains no strict-PT W2/W0 candidate")
    if expected_candidates is not None:
        expected = set(map(str, expected_candidates))
        observed = set(output["candidate"].astype(str))
        if not expected or observed != expected:
            raise ValueError(
                "cascade W2/W0 candidate set differs from the frozen declared set; "
                f"missing={sorted(expected - observed)}, "
                f"undeclared={sorted(observed - expected)}"
            )
    if not output["quote_menu_status"].eq("exact").all():
        failures = output.loc[
            ~output["quote_menu_status"].eq("exact"),
            [
                "candidate",
                "pairing_status",
                "quote_menu_status",
                "w0_count",
                "w2_count",
                "w2_only_quote_keys",
                "w0_only_quote_keys",
            ],
        ].to_dict(orient="records")
        raise ValueError(
            "every W2 arm requires exactly one W0 arm on the identical complete "
            f"quote menu; failures={failures}"
        )
    return output


def _canonical_cascade_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    required = [
        "model",
        "candidate",
        "variant",
        "event_key",
        "event_start_date",
        "round_num",
        "pair_key",
        "p1_probability",
        "p2_probability",
        "push_probability",
        "pricing_p1_probability",
        "tie_rule",
        "tie_rule_ties_lose",
        "derived_score_outcome",
        "score_model_eligible",
        "market_devig_proper_score_eligible",
        "market_devig_proper_score_exclusion_reason",
        "binary_model_eligible",
        "market_devig_probability_scope",
        "p1_open",
        "p2_open",
        "p1_close",
        "p2_close",
        "p1_open_devig_probability",
        "p1_close_devig_probability",
        "model_minus_devig_open_pp",
        "model_minus_devig_close_pp",
    ]
    _require(frame, required, "cascade round matchup probabilities")
    output = frame[required].copy()
    output["event_start_date"] = pd.to_datetime(
        output["event_start_date"], errors="coerce"
    ).dt.normalize()
    output["round_num"] = pd.to_numeric(output["round_num"], errors="coerce")
    _numeric(
        output,
        [
            "p1_probability",
            "p2_probability",
            "push_probability",
            "score_model_eligible",
            "pricing_p1_probability",
            "derived_score_outcome",
        ],
    )
    probability_columns = [
        "p1_probability",
        "p2_probability",
        "push_probability",
        "pricing_p1_probability",
    ]
    if output["event_start_date"].isna().any() or output["round_num"].isna().any():
        raise ValueError("cascade contains invalid event dates or rounds")
    if not np.isfinite(output[probability_columns].to_numpy(float)).all():
        raise ValueError("cascade contains non-finite probabilities")
    for column in (
        "p1_probability",
        "p2_probability",
        "push_probability",
        "pricing_p1_probability",
    ):
        if not output[column].between(0.0, 1.0, inclusive="both").all():
            raise ValueError(f"cascade {column} is outside [0,1]")
    for column in (
        "score_model_eligible",
        "market_devig_proper_score_eligible",
        "binary_model_eligible",
    ):
        _require_exact_boolean_series(output[column], f"cascade {column}")
    score_eligible = output["score_model_eligible"].astype(bool)
    if np.isinf(output["derived_score_outcome"].to_numpy(float)).any():
        raise ValueError("cascade derived_score_outcome contains an infinity")
    finite_outcome = np.isfinite(output["derived_score_outcome"])
    valid_outcome = output["derived_score_outcome"].isin([0.0, 0.5, 1.0])
    if (finite_outcome & ~valid_outcome).any() or (
        score_eligible & ~valid_outcome
    ).any():
        raise ValueError(
            "cascade derived_score_outcome is invalid for its score-model eligibility"
        )
    tie_rule_loses = _require_exact_boolean_series(
        output["tie_rule_ties_lose"], "cascade tie_rule_ties_lose"
    )
    tie_rule = output["tie_rule"].fillna("").astype(str).str.strip()
    if tie_rule.eq("").any():
        raise ValueError("cascade tie_rule is missing")
    normalized_tie_rule = tie_rule.str.casefold().str.replace(
        r"[_-]", " ", regex=True
    )
    expected_tie_rule_loses = (
        normalized_tie_rule.str.contains("separate", regex=False)
        | normalized_tie_rule.str.contains("ties lose", regex=False)
        | normalized_tie_rule.str.contains("tie loses", regex=False)
        | normalized_tie_rule.str.contains("three way", regex=False)
        | normalized_tie_rule.str.contains("3 way", regex=False)
    )
    if not tie_rule_loses.eq(expected_tie_rule_loses).all():
        raise ValueError("cascade tie_rule_ties_lose differs from tie_rule")
    resolved_mass = 1.0 - output["push_probability"]
    if ((~tie_rule_loses) & resolved_mass.le(0.0)).any():
        raise ValueError("cascade push-on-tie pricing has no resolved probability mass")
    expected_pricing = output["p1_probability"].where(
        tie_rule_loses,
        output["p1_probability"] / resolved_mass,
    )
    pricing_error = (output["pricing_p1_probability"] - expected_pricing).abs()
    if (
        not np.isfinite(expected_pricing.to_numpy(float)).all()
        or not np.isfinite(pricing_error.to_numpy(float)).all()
        or pricing_error.max() > 1e-12
    ):
        raise ValueError(
            "cascade pricing probability violates tie-rule/push algebra"
        )
    market_eligible = output["market_devig_proper_score_eligible"].astype(bool)
    binary_eligible = output["binary_model_eligible"].astype(bool)
    expected_market_eligible = (
        score_eligible
        & ~tie_rule_loses
        & ~output["derived_score_outcome"].eq(0.5)
    )
    if (
        not market_eligible.eq(expected_market_eligible).all()
        or not binary_eligible.eq(market_eligible).all()
    ):
        raise ValueError("cascade market proper-score eligibility contract is false")
    expected_reason = np.select(
        [
            ~score_eligible,
            tie_rule_loses,
            output["derived_score_outcome"].eq(0.5),
        ],
        [
            "ungraded_or_score_settlement_mismatch",
            "ties_lose_without_draw_quote_conditional_devig_not_comparable",
            "physical_tie_excluded_from_resolved_two_way_benchmark",
        ],
        default="eligible_resolved_push_on_tie_two_way_benchmark",
    )
    if not output["market_devig_proper_score_exclusion_reason"].astype(str).eq(
        pd.Series(expected_reason, index=output.index).astype(str)
    ).all():
        raise ValueError("cascade market proper-score exclusion reason is false")
    expected_scope = np.where(
        tie_rule_loses,
        "conditional_on_either_player_winning_no_draw_quote_attribution_only",
        "resolved_two_way_push_on_tie_market_benchmark",
    )
    if not output["market_devig_probability_scope"].astype(str).eq(
        pd.Series(expected_scope, index=output.index).astype(str)
    ).all():
        raise ValueError("cascade market devig probability scope is false")
    for price in EXPECTED_CASCADE_PRICE_POINTS:
        p1_odds = f"p1_{price}"
        p2_odds = f"p2_{price}"
        devig_column = f"p1_{price}_devig_probability"
        gap_column = f"model_minus_devig_{price}_pp"
        output[devig_column] = pd.to_numeric(output[devig_column], errors="coerce")
        output[gap_column] = pd.to_numeric(output[gap_column], errors="coerce")
        valid_price = _valid_american_odds(output[p1_odds]) & _valid_american_odds(
            output[p2_odds]
        )
        valid_devig = (
            np.isfinite(output[devig_column])
            & output[devig_column].between(0.0, 1.0, inclusive="both")
        )
        if not valid_devig.eq(valid_price).all():
            raise ValueError(f"cascade {price} devig probability coverage is false")
        expected_gap_valid = valid_price & market_eligible
        if (
            not np.isfinite(output.loc[expected_gap_valid, gap_column]).all()
            or output.loc[~expected_gap_valid, gap_column].notna().any()
        ):
            raise ValueError(f"cascade {price} market benchmark gap eligibility is false")
        gap_error = (
            output.loc[expected_gap_valid, gap_column]
            - 100.0
            * (
                output.loc[expected_gap_valid, "pricing_p1_probability"]
                - output.loc[expected_gap_valid, devig_column]
            )
        ).abs()
        if not gap_error.empty and gap_error.max() > 1e-10:
            raise ValueError(f"cascade {price} market benchmark gap algebra is false")
    output = output.loc[
        output["event_start_date"].between(SEALED_START, SEALED_END)
        & output["round_num"].isin([1, 2, 3, 4])
    ].copy()
    if output.empty:
        raise ValueError("cascade probabilities contain no sealed-period rows")
    output["round_num"] = output["round_num"].astype(int)
    identity = [
        "model",
        "event_key",
        "event_start_date",
        "round_num",
        "pair_key",
    ]
    invariant_columns = [
        "p1_probability",
        "p2_probability",
        "push_probability",
        "derived_score_outcome",
        "score_model_eligible",
        "candidate",
        "variant",
    ]
    # Tie settlement and devig comparison are quote/book attributes.  A single
    # physical pair can legitimately be push-on-tie at one book and ties-lose
    # at another.  Collapse only the invariant three-way model probability and
    # physical score outcome; pricing/market-benchmark fields stay in the raw
    # quote-scoped panel validated above.
    grouped = output.groupby(identity, observed=True, dropna=False)
    disagreement = grouped[invariant_columns].nunique(dropna=False).gt(1).any(axis=1)
    if disagreement.any():
        raise ValueError(
            "cascade invariant probability/outcome differs across book copies"
        )
    canonical = (
        output[[*identity, *invariant_columns]]
        .sort_values(identity, kind="mergesort")
        .drop_duplicates(identity)
    )
    if (
        canonical[identity].isna().any().any()
        or canonical["event_start_date"].isna().any()
        or canonical.duplicated(identity).any()
    ):
        raise ValueError("cascade canonical probability keys are invalid")
    model_mapping = canonical[["model", "candidate", "variant"]].drop_duplicates()
    if model_mapping["model"].duplicated().any():
        raise ValueError("cascade model maps to multiple candidate/variant identities")
    total_error = (
        canonical["p1_probability"]
        + canonical["p2_probability"]
        + canonical["push_probability"]
        - 1.0
    ).abs()
    if total_error.max() > 1e-10:
        raise ValueError("cascade probability simplex identity failed")
    return canonical


def _valid_american_odds(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return np.isfinite(numeric) & (numeric.ge(100.0) | numeric.le(-100.0))


def _finite_numeric_authority_series(
    series: pd.Series, *, label: str
) -> tuple[pd.Series, pd.Series]:
    """Coerce numeric authority without confusing forgery with raw missingness."""

    raw_missing = series.isna()
    coerced = pd.to_numeric(series, errors="coerce")
    values = pd.Series(coerced, index=series.index).to_numpy(
        dtype=float, na_value=np.nan
    )
    malformed = (~raw_missing.to_numpy(dtype=bool)) & ~np.isfinite(values)
    if malformed.any():
        examples = series.loc[malformed].astype(str).unique()[:5].tolist()
        raise ValueError(
            f"{label} contains nonmissing non-numeric or non-finite values; "
            f"examples={examples}"
        )
    return pd.Series(values, index=series.index), raw_missing.astype(bool)


def _cascade_verified_model_quote_menu(raw_probabilities: pd.DataFrame) -> pd.DataFrame:
    """Scope the score-verified producer menu to this sealed analysis window."""

    required = ["model", "quote_id", "event_start_date", "round_num"]
    _require(raw_probabilities, required, "cascade verified model quote menu source")
    frame = raw_probabilities[required].copy()
    declared_date = (
        frame["event_start_date"].notna()
        & frame["event_start_date"].astype(str).str.strip().ne("")
    )
    frame["event_start_date"] = pd.to_datetime(
        frame["event_start_date"], errors="coerce"
    ).dt.normalize()
    frame["round_num"] = pd.to_numeric(frame["round_num"], errors="coerce")
    if (
        (declared_date & frame["event_start_date"].isna()).any()
        or frame[["model", "quote_id", "event_start_date", "round_num"]]
        .isna()
        .any()
        .any()
    ):
        raise ValueError("cascade verified model quote menu source has null identity")
    frame = frame.loc[
        frame["event_start_date"].between(SEALED_START, SEALED_END)
        & frame["round_num"].isin([1, 2, 3, 4])
    ].copy()
    menu = frame[["model", "quote_id"]].drop_duplicates()
    if menu.empty or menu.duplicated(["model", "quote_id"]).any():
        raise ValueError("cascade verified in-window model quote menu is empty or ambiguous")
    return menu


def audit_cascade_quote_slices(
    normalized_quotes: pd.DataFrame,
    raw_probabilities: pd.DataFrame,
    pairing: pd.DataFrame,
    *,
    expected_price_points: Sequence[str],
    expected_books: Sequence[str],
    verified_model_quote_menu: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bind each prediction-covered arm to normalized executable quotes."""

    price_points = tuple(map(str, expected_price_points))
    books = tuple(map(str, expected_books))
    if not price_points or not books or len(set(price_points)) != len(price_points):
        raise ValueError("cascade declared price/book menu is empty or duplicated")
    if len(set(books)) != len(books):
        raise ValueError("cascade declared book menu is duplicated")
    quote_required = [
        "quote_id",
        "event_key",
        "event_start_date",
        "round_num",
        "pair_key",
        "book",
        "p1_id",
        "p2_id",
        "p1_outcome",
        "p2_outcome",
        "tie_rule",
        *[
            column
            for price in price_points
            for column in (f"p1_{price}", f"p2_{price}")
        ],
    ]
    _require(normalized_quotes, quote_required, "cascade normalized quote authority")
    authority = normalized_quotes[quote_required].copy()
    authority["event_start_date"] = pd.to_datetime(
        authority["event_start_date"], errors="coerce"
    ).dt.normalize()
    authority["round_num"] = pd.to_numeric(authority["round_num"], errors="coerce")
    authority["book"] = authority["book"].astype(str).str.strip().str.casefold()
    authority = authority.loc[
        authority["event_start_date"].between(SEALED_START, SEALED_END)
        & authority["round_num"].isin([1, 2, 3, 4])
    ].copy()
    authority["round_num"] = authority["round_num"].astype(int)
    keys = ["event_key", "event_start_date", "round_num", "pair_key"]
    authority_identity = [*keys, "book"]
    if (
        authority.empty
        or authority[["quote_id", *authority_identity, "p1_id", "p2_id"]]
        .isna()
        .any()
        .any()
        or authority["quote_id"].astype(str).duplicated().any()
        or authority.duplicated(authority_identity).any()
        or set(authority["book"].astype(str)) != set(books)
        or authority["tie_rule"].fillna("").astype(str).str.strip().eq("").any()
    ):
        raise ValueError("cascade normalized quote authority is incomplete or ambiguous")

    probability_required = [
        "model",
        "quote_id",
        "event_key",
        "event_start_date",
        "round_num",
        "pair_key",
        "book",
        "p1_id",
        "p2_id",
        "p1_outcome",
        "p2_outcome",
        "tie_rule",
        *[
            column
            for price in price_points
            for column in (f"p1_{price}", f"p2_{price}")
        ],
    ]
    _require(raw_probabilities, probability_required, "cascade quoted probability panel")
    frame = raw_probabilities[probability_required].copy()
    frame["event_start_date"] = pd.to_datetime(
        frame["event_start_date"], errors="coerce"
    ).dt.normalize()
    frame["round_num"] = pd.to_numeric(frame["round_num"], errors="coerce")
    frame = frame.loc[
        frame["event_start_date"].between(SEALED_START, SEALED_END)
        & frame["round_num"].isin([1, 2, 3, 4])
    ].copy()
    frame["round_num"] = frame["round_num"].astype(int)
    relevant_models = {
        CASCADE_CONTROL,
        *pairing["w0_model"].astype(str),
        *pairing["w2_model"].astype(str),
    }
    frame = frame.loc[frame["model"].astype(str).isin(relevant_models)].copy()
    identity = ["model", *keys, "book"]
    if (
        frame[[*identity]].isna().any().any()
        or frame.duplicated(identity).any()
    ):
        raise ValueError("cascade quote endpoints have null or ambiguous keys")
    frame["book"] = frame["book"].astype(str).str.strip().str.casefold()
    observed_books = set(frame["book"].astype(str))
    if observed_books != set(books):
        raise ValueError(
            "cascade probability quote panel book menu differs from authority: "
            f"observed={sorted(observed_books)}, expected={sorted(books)}"
        )
    model_quote_ids: dict[str, set[str]] = {}
    if verified_model_quote_menu is not None:
        _require(
            verified_model_quote_menu,
            ["model", "quote_id"],
            "verified cascade model quote menu",
        )
        menu = verified_model_quote_menu[["model", "quote_id"]].copy()
        menu["model"] = menu["model"].astype(str)
        menu["quote_id"] = menu["quote_id"].astype(str)
        menu = menu.loc[menu["model"].isin(relevant_models)].copy()
        if (
            menu.empty
            or menu.isna().any().any()
            or menu.duplicated(["model", "quote_id"]).any()
            or set(menu["model"]) != relevant_models
            or not set(menu["quote_id"]).issubset(
                set(authority["quote_id"].astype(str))
            )
        ):
            raise ValueError("verified cascade model quote menu is incomplete")
        model_quote_ids = {
            model: set(group["quote_id"])
            for model, group in menu.groupby("model", observed=True)
        }
        observed_menu = frame[["model", "quote_id"]].copy()
        observed_menu["model"] = observed_menu["model"].astype(str)
        observed_menu["quote_id"] = observed_menu["quote_id"].astype(str)
        expected_menu = menu.merge(
            observed_menu,
            on=["model", "quote_id"],
            how="outer",
            indicator=True,
            validate="one_to_one",
        )
        if not expected_menu["_merge"].eq("both").all():
            raise ValueError(
                "cascade probability menu differs from verified prediction-covered quote menu"
            )
    else:
        # Standalone callers that have not already verified prediction
        # coverage retain the stricter all-normalized-quote expectation.
        all_ids = set(authority["quote_id"].astype(str))
        model_quote_ids = {model: all_ids for model in relevant_models}
    policies = (*books, "best_of_three")
    quote_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []

    def authority_detail(price: str, policy: str) -> pd.DataFrame:
        valid = _valid_american_odds(
            authority[f"p1_{price}"]
        ) & _valid_american_odds(authority[f"p2_{price}"])
        use = authority.loc[valid].copy()
        if policy != "best_of_three":
            use = use.loc[use["book"].astype(str).eq(policy)].copy()
        return use[
            [
                *keys,
                "book",
                "quote_id",
                "p1_id",
                "p2_id",
                "p1_outcome",
                "p2_outcome",
                "tie_rule",
                f"p1_{price}",
                f"p2_{price}",
            ]
        ].sort_values([*keys, "book"], kind="mergesort")

    def detail(model: str, price: str, policy: str) -> pd.DataFrame:
        use = frame.loc[frame["model"].eq(model)].copy()
        valid = _valid_american_odds(use[f"p1_{price}"]) & _valid_american_odds(
            use[f"p2_{price}"]
        )
        use = use.loc[valid & use["book"].astype(str).isin(books)].copy()
        if policy != "best_of_three":
            use = use.loc[use["book"].astype(str).eq(policy)].copy()
        return use[
            [
                *keys,
                "book",
                "quote_id",
                "p1_id",
                "p2_id",
                "p1_outcome",
                "p2_outcome",
                "tie_rule",
                f"p1_{price}",
                f"p2_{price}",
            ]
        ].sort_values([*keys, "book"], kind="mergesort")

    for row in pairing.to_dict(orient="records"):
        w2 = str(row["w2_model"])
        w0 = str(row["w0_model"])
        for price in price_points:
            for policy in policies:
                full_authoritative = authority_detail(price, policy)
                if full_authoritative.empty:
                    raise ValueError(
                        "cascade code-frozen price/policy has no normalized executable quotes: "
                        f"price={price}, policy={policy}"
                    )
                arm_details = {
                    model: detail(model, price, policy)
                    for model in (w2, w0, CASCADE_CONTROL)
                }
                if any(value.empty for value in arm_details.values()):
                    raise ValueError(
                        "cascade declared price/policy has no executable quote menu "
                        f"for every arm: pair={w2}__{w0}, price={price}, "
                        f"policy={policy}"
                    )
                arm_authorities: dict[str, pd.DataFrame] = {}
                for model, candidate_detail in arm_details.items():
                    authoritative = full_authoritative.loc[
                        full_authoritative["quote_id"].astype(str).isin(
                            model_quote_ids[model]
                        )
                    ].copy()
                    if authoritative.empty:
                        raise ValueError(
                            "cascade model has no executable normalized quotes for "
                            f"model={model}, price={price}, policy={policy}"
                        )
                    arm_authorities[model] = authoritative
                    candidate = candidate_detail.rename(
                        columns={
                            f"p1_{price}": "p1_candidate",
                            f"p2_{price}": "p2_candidate",
                            "tie_rule": "tie_rule_candidate",
                            "quote_id": "quote_id_candidate",
                            "p1_id": "p1_id_candidate",
                            "p2_id": "p2_id_candidate",
                            "p1_outcome": "p1_outcome_candidate",
                            "p2_outcome": "p2_outcome_candidate",
                        }
                    )
                    compared = authoritative.rename(
                        columns={
                            f"p1_{price}": "p1_authority",
                            f"p2_{price}": "p2_authority",
                            "tie_rule": "tie_rule_authority",
                            "quote_id": "quote_id_authority",
                            "p1_id": "p1_id_authority",
                            "p2_id": "p2_id_authority",
                            "p1_outcome": "p1_outcome_authority",
                            "p2_outcome": "p2_outcome_authority",
                        }
                    ).merge(
                        candidate,
                        on=[*keys, "book"],
                        how="outer",
                        validate="one_to_one",
                        indicator=True,
                    )
                    p1_authority, _ = _finite_numeric_authority_series(
                        compared["p1_authority"],
                        label="cascade normalized p1 price authority",
                    )
                    p1_candidate, _ = _finite_numeric_authority_series(
                        compared["p1_candidate"],
                        label="cascade probability p1 price copy",
                    )
                    p2_authority, _ = _finite_numeric_authority_series(
                        compared["p2_authority"],
                        label="cascade normalized p2 price authority",
                    )
                    p2_candidate, _ = _finite_numeric_authority_series(
                        compared["p2_candidate"],
                        label="cascade probability p2 price copy",
                    )
                    p1_outcome_authority, _ = _finite_numeric_authority_series(
                        compared["p1_outcome_authority"],
                        label="cascade normalized p1 outcome authority",
                    )
                    p1_outcome_candidate, _ = _finite_numeric_authority_series(
                        compared["p1_outcome_candidate"],
                        label="cascade probability p1 outcome copy",
                    )
                    p2_outcome_authority, _ = _finite_numeric_authority_series(
                        compared["p2_outcome_authority"],
                        label="cascade normalized p2 outcome authority",
                    )
                    p2_outcome_candidate, _ = _finite_numeric_authority_series(
                        compared["p2_outcome_candidate"],
                        label="cascade probability p2 outcome copy",
                    )
                    exact_values = (
                        compared["_merge"].eq("both").all()
                        and np.allclose(
                            p1_authority,
                            p1_candidate,
                            rtol=0.0,
                            atol=1e-10,
                        )
                        and np.allclose(
                            p2_authority,
                            p2_candidate,
                            rtol=0.0,
                            atol=1e-10,
                        )
                        and compared["tie_rule_authority"].astype(str).eq(
                            compared["tie_rule_candidate"].astype(str)
                        ).all()
                        and compared["quote_id_authority"].astype(str).eq(
                            compared["quote_id_candidate"].astype(str)
                        ).all()
                        and compared["p1_id_authority"].astype(str).eq(
                            compared["p1_id_candidate"].astype(str)
                        ).all()
                        and compared["p2_id_authority"].astype(str).eq(
                            compared["p2_id_candidate"].astype(str)
                        ).all()
                        and np.allclose(
                            p1_outcome_authority,
                            p1_outcome_candidate,
                            rtol=0.0,
                            atol=1e-10,
                            equal_nan=True,
                        )
                        and np.allclose(
                            p2_outcome_authority,
                            p2_outcome_candidate,
                            rtol=0.0,
                            atol=1e-10,
                            equal_nan=True,
                        )
                    )
                    if not exact_values:
                        raise ValueError(
                            "cascade probability arm differs from normalized executable "
                            f"quote authority: model={model}, price={price}, policy={policy}"
                        )
                arm_key_sets = {
                    model: set(
                        map(tuple, values[keys].drop_duplicates().to_numpy())
                    )
                    for model, values in arm_authorities.items()
                }
                if arm_key_sets[w2] != arm_key_sets[w0]:
                    raise ValueError(
                        "cascade W2/W0 executable quote menus differ after "
                        f"prediction coverage: pair={w2}__{w0}, price={price}, "
                        f"policy={policy}"
                    )
                common_three_arm = set.intersection(*arm_key_sets.values())
                if not common_three_arm:
                    raise ValueError("cascade W2/W0/C0 common executable menu is empty")
                for model, values in arm_authorities.items():
                    eligible = values[keys].drop_duplicates().assign(
                        model=model,
                        price_point=price,
                        line_policy=policy,
                    )
                    quote_parts.append(eligible)
                audit_rows.append(
                    {
                        "analysis_pair": f"{w2}__vs__{w0}",
                        "price_point": price,
                        "line_policy": policy,
                        "w2_model": w2,
                        "w0_model": w0,
                        "control_model": CASCADE_CONTROL,
                        "w2_quote_keys": int(len(arm_key_sets[w2])),
                        "w0_quote_keys": int(len(arm_key_sets[w0])),
                        "control_quote_keys": int(len(arm_key_sets[CASCADE_CONTROL])),
                        "three_arm_common_quote_keys": int(len(common_three_arm)),
                        "w2_quote_keys_outside_control_common": int(
                            len(arm_key_sets[w2] - common_three_arm)
                        ),
                        "control_quote_keys_outside_w2_w0_common": int(
                            len(arm_key_sets[CASCADE_CONTROL] - common_three_arm)
                        ),
                        "w2_w0_complete_exact_menu": True,
                        "control_comparisons_require_pairwise_common_menu": True,
                        "normalized_quote_authority_exact": True,
                        "three_arm_complete_exact_menu": bool(
                            all(
                                keys_set == arm_key_sets[w2]
                                for keys_set in arm_key_sets.values()
                            )
                        ),
                        # Backward-readable alias, now truthfully scoped to all
                        # three arms rather than only the W2/W0 matched pair.
                        "complete_exact_menu": bool(
                            all(
                                keys_set == arm_key_sets[w2]
                                for keys_set in arm_key_sets.values()
                            )
                        ),
                    }
                )
    quote_keys = pd.concat(quote_parts, ignore_index=True).drop_duplicates()
    quote_identity = ["model", *keys, "price_point", "line_policy"]
    if quote_keys.duplicated(quote_identity).any():
        raise ValueError("cascade eligible quote universe is ambiguous")
    return pd.DataFrame(audit_rows), quote_keys


def validate_cascade_bets_against_quote_slices(
    bets: pd.DataFrame, quote_keys: pd.DataFrame
) -> None:
    keys = [
        "model",
        "event_key",
        "event_start_date",
        "round_num",
        "pair_key",
        "price_point",
        "line_policy",
    ]
    comparison = bets[keys].drop_duplicates().merge(
        quote_keys[keys],
        on=keys,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not comparison["_merge"].eq("both").all():
        raise ValueError(
            "cascade bet references a missing/undeclared executable quote slice"
        )


def select_cascade_analysis_bets(
    bets: pd.DataFrame, pairing: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Explicitly isolate paired W2/W0+C0 bets after full-ledger settlement audit."""

    _require(bets, ["model", "stake", "pnl"], "full cascade bet ledger")
    _require(
        pairing,
        ["w2_model", "w0_model", "quote_menu_status"],
        "cascade analysis pairing",
    )
    if pairing.empty or not pairing["quote_menu_status"].astype(str).eq("exact").all():
        raise ValueError("cascade analysis pairing contains a non-exact quote menu")
    required_models = {
        CASCADE_CONTROL,
        *pairing["w2_model"].astype(str),
        *pairing["w0_model"].astype(str),
    }
    observed_models = set(bets["model"].astype(str))
    missing = required_models - observed_models
    if missing:
        raise ValueError(
            "cascade bet ledger is missing a required paired analysis model: "
            + repr(sorted(missing))
        )
    analysis = bets.loc[bets["model"].astype(str).isin(required_models)].copy()
    if analysis.empty or set(analysis["model"].astype(str)) != required_models:
        raise ValueError("cascade analysis bet model menu is not pairing+C0 exact")
    excluded = bets.loc[~bets["model"].astype(str).isin(required_models)].copy()
    excluded_by_model = {
        str(model): int(len(group))
        for model, group in excluded.groupby("model", observed=True)
    }
    audit = {
        "full_settled_ledger_rows": int(len(bets)),
        "full_settled_ledger_models": sorted(observed_models),
        "analysis_rows": int(len(analysis)),
        "analysis_models": sorted(required_models),
        "analysis_model_menu_pairing_plus_c0_exact": True,
        "excluded_nonanalysis_rows": int(len(excluded)),
        "excluded_nonanalysis_models": sorted(observed_models - required_models),
        "excluded_nonanalysis_rows_by_model": excluded_by_model,
        "excluded_nonanalysis_stake": float(excluded["stake"].sum()),
        "excluded_nonanalysis_pnl": float(excluded["pnl"].sum()),
        "full_ledger_used_for_settlement_validation": True,
        "analysis_subset_used_for_quote_menu_money_and_attribution": True,
    }
    return analysis, audit


def validate_c3_coefficient_feature_lineage(
    coefficients: pd.DataFrame, pairing: pd.DataFrame
) -> dict[str, Any]:
    """Prove W0/W2 C3 fits use only the declared completed-state feature arm."""

    required = [
        "model",
        "candidate",
        "mode",
        "weather_arm",
        "round_num",
        "prediction_year",
        "training_max_event_end",
        "feature",
        "standardized_coefficient",
    ]
    _require(coefficients, required, "cascade C3 coefficients")
    frame = coefficients[required].copy()
    frame["round_num"] = pd.to_numeric(frame["round_num"], errors="coerce")
    frame["prediction_year"] = pd.to_numeric(
        frame["prediction_year"], errors="coerce"
    )
    frame["training_max_event_end"] = pd.to_datetime(
        frame["training_max_event_end"], errors="coerce"
    ).dt.normalize()
    frame["standardized_coefficient"] = pd.to_numeric(
        frame["standardized_coefficient"], errors="coerce"
    )
    relevant: dict[str, tuple[str, str]] = {}
    for row in pairing.to_dict(orient="records"):
        relevant[str(row["w0_model"])] = (str(row["candidate"]), "w0")
        relevant[str(row["w2_model"])] = (str(row["candidate"]), "w2")
    frame = frame.loc[frame["model"].astype(str).isin(relevant)].copy()
    if not relevant or set(frame["model"].astype(str)) != set(relevant):
        raise ValueError("cascade coefficients lack a complete W0/W2 model menu")
    identity = ["model", "round_num", "prediction_year", "feature"]
    if (
        frame[
            [
                "model",
                "candidate",
                "mode",
                "weather_arm",
                "round_num",
                "prediction_year",
                "training_max_event_end",
                "feature",
            ]
        ].isna().any().any()
        or frame.duplicated(identity).any()
        or not np.isfinite(frame["standardized_coefficient"].to_numpy(float)).all()
    ):
        raise ValueError("cascade C3 coefficient identities/values are invalid")
    if not frame["round_num"].isin([1, 2, 3, 4]).all():
        raise ValueError("cascade C3 coefficients contain an invalid round")
    cutoff = pd.to_datetime(
        frame["prediction_year"].astype(int).astype(str) + "-01-01"
    )
    if frame["training_max_event_end"].ge(cutoff).any():
        raise ValueError("cascade C3 coefficient training reaches prediction year")

    categories = ("ott", "app", "arg", "putt")
    rows_by_arm: dict[str, int] = {"w0": 0, "w2": 0}
    for model, (candidate, arm) in relevant.items():
        use = frame.loc[frame["model"].astype(str).eq(model)].copy()
        if (
            not use["candidate"].astype(str).eq(candidate).all()
            or not use["mode"].astype(str).eq("pt").all()
            or not use["weather_arm"].astype(str).eq(arm).all()
        ):
            raise ValueError(f"cascade coefficient identity is false for {model}")
        if set(use["round_num"].astype(int)) != {1, 2, 3, 4}:
            raise ValueError(f"cascade coefficient round menu is incomplete for {model}")
        opposite = "w2" if arm == "w0" else "w0"
        for round_num, group in use.groupby("round_num", observed=True):
            features = set(group["feature"].astype(str))
            allowed = {"base_event_hybrid", "__intercept_scaled__"}
            if int(round_num) >= 2:
                allowed.update(
                    {
                        f"prev_mean_resid_{arm}_pt",
                        "start_pos_scaled",
                        "prev_weather_observed_rate",
                        *{
                            f"prev_mean_orth_{category}_{arm}"
                            for category in categories
                        },
                    }
                )
            if int(round_num) >= 3:
                allowed.update(
                    {
                        f"prev_last_resid_{arm}_pt",
                        f"prev_std_total_{arm}",
                        f"prev_trend_total_{arm}",
                        *{
                            f"prev_last_orth_{category}_{arm}"
                            for category in categories
                        },
                        *{
                            f"prev_trend_orth_{category}_{arm}"
                            for category in categories
                        },
                    }
                )
            if not features.issubset(allowed):
                raise ValueError(
                    f"cascade {arm} coefficient uses undeclared/current-weather "
                    f"features: {sorted(features - allowed)}"
                )
            if "base_event_hybrid" not in features:
                raise ValueError(f"cascade {arm} coefficient lacks its base anchor")
            if int(round_num) >= 2 and f"prev_mean_resid_{arm}_pt" not in features:
                raise ValueError(
                    f"cascade {arm} coefficient lacks completed-state lineage"
                )
            if any(opposite in feature for feature in features):
                raise ValueError(f"cascade {arm} coefficient crosses weather arms")
        rows_by_arm[arm] += int(len(use))
    return {
        "coefficient_rows": int(len(frame)),
        "w0_coefficient_rows": rows_by_arm["w0"],
        "w2_coefficient_rows": rows_by_arm["w2"],
        "coefficient_training_boundary_violations": 0,
        "current_or_target_weather_feature_columns": 0,
        "completed_state_feature_arm_isolated": True,
    }


def validate_c3_weather_state_lineage(
    predictions: pd.DataFrame,
    cascade_manifest: Mapping[str, Any],
    cascade_audit: Mapping[str, Any],
    completion_authority: Mapping[str, Mapping[str, str]],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Independently prove C3 uses only weather from strictly completed rounds."""

    required = [
        "model",
        "candidate",
        "event_key",
        "event_start_date",
        "round_num",
        "variant",
        "weather_arm",
        "weather_state_target_column",
        "weather_state_source_columns",
        "completed_rounds_available",
        "completed_rounds_used",
        "completed_weather_observations_available",
        "completed_weather_state_rows_with_valid_coverage",
        "completed_weather_observations_used",
        "completed_weather_state_rows_used_by_w2",
        "completed_weather_used_by_w2",
        "completed_prior_weather_coverage_rate",
        "completed_prior_round_boundary",
        "completed_prior_rounds_only",
        "weather_coefficient_fit_cutoff",
        "weather_coefficient_source",
        "weather_coefficients_path",
        "weather_coefficients_sha256",
        "target_or_upcoming_realized_weather_excluded",
        "strict_point_in_time",
        "current_round_weather_columns_role",
    ]
    _require(predictions, required, "cascade C3 weather-state lineage")
    c3 = predictions.loc[
        predictions["variant"].astype("string").str.startswith("C3_", na=False)
    ].copy()
    if c3.empty:
        raise ValueError("cascade C3 weather-state lineage is empty")
    if c3[["model", "candidate", "event_key", "variant"]].isna().any().any():
        raise ValueError("cascade C3 weather-state lineage has null identities")
    for column in ("model", "candidate", "event_key", "variant"):
        if c3[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"cascade C3 weather-state lineage has blank {column}")

    def exact_integers(column: str) -> pd.Series:
        return _require_nonnegative_integer_series(
            c3[column], f"cascade C3 weather-state lineage {column}"
        )

    round_num = exact_integers("round_num")
    available = exact_integers("completed_rounds_available")
    used = exact_integers("completed_rounds_used")
    weather_available = exact_integers("completed_weather_observations_available")
    weather_valid = exact_integers(
        "completed_weather_state_rows_with_valid_coverage"
    )
    weather_compat_used = exact_integers("completed_weather_observations_used")
    weather_used_by_w2 = exact_integers(
        "completed_weather_state_rows_used_by_w2"
    )
    if not round_num.isin([1, 2, 3, 4]).all():
        raise ValueError("cascade C3 weather-state lineage has an invalid target round")
    if not available.eq(round_num - 1).all():
        raise ValueError("cascade C3 available-round lineage is not round minus one")
    if used.lt(0).any() or used.gt(available).any():
        raise ValueError("cascade C3 completed-state count crosses target-round boundary")
    if weather_available.lt(0).any() or weather_available.gt(available).any():
        raise ValueError(
            "cascade C3 available completed-weather count crosses target-round boundary"
        )
    if (
        weather_valid.lt(0).any()
        or weather_valid.gt(weather_available).any()
        or weather_valid.gt(used).any()
    ):
        raise ValueError(
            "cascade C3 valid completed-weather state crosses target-round boundary"
        )
    if not weather_compat_used.eq(weather_valid).all():
        raise ValueError("cascade C3 compatibility weather count differs from valid state")
    if (
        used.loc[round_num.eq(1)].ne(0).any()
        or weather_available.loc[round_num.eq(1)].ne(0).any()
        or weather_valid.loc[round_num.eq(1)].ne(0).any()
        or weather_used_by_w2.loc[round_num.eq(1)].ne(0).any()
    ):
        raise ValueError("cascade C3 R1 claims current-event completed state/weather")

    arm = c3["weather_arm"].astype(str)
    if not arm.isin(C3_WEATHER_SOURCE_COLUMNS).all() or set(arm) != {"w0", "w2"}:
        raise ValueError("cascade C3 weather-state arm menu is not exact")
    expected_target = arm.map({"w0": "target_total_w0", "w2": "target_total_w2"})
    expected_sources = arm.map(C3_WEATHER_SOURCE_COLUMNS)
    if not c3["weather_state_target_column"].astype(str).eq(expected_target).all():
        raise ValueError("cascade C3 weather-state target column differs from its arm")
    if not c3["weather_state_source_columns"].astype(str).eq(expected_sources).all():
        raise ValueError("cascade C3 weather-state source columns differ from their arm")

    expected_weather_used_by_w2 = weather_valid.where(arm.eq("w2"), 0)
    if not weather_used_by_w2.eq(expected_weather_used_by_w2).all():
        raise ValueError("cascade C3 W2 completed-weather use count is false")
    weather_use_flag = _require_exact_boolean_series(
        c3["completed_weather_used_by_w2"],
        "cascade C3 completed_weather_used_by_w2",
    )
    if not weather_use_flag.eq(expected_weather_used_by_w2.gt(0)).all():
        raise ValueError("cascade C3 W2 completed-weather use flag is false")
    coverage_rate = pd.to_numeric(
        c3["completed_prior_weather_coverage_rate"], errors="coerce"
    )
    expected_coverage_rate = pd.Series(
        np.where(available.gt(0), weather_valid / available, 0.0),
        index=c3.index,
    )
    if (
        not np.isfinite(coverage_rate.to_numpy(float)).all()
        or not np.isclose(
            coverage_rate.to_numpy(float),
            expected_coverage_rate.to_numpy(float),
            atol=1e-12,
            rtol=0.0,
        ).all()
    ):
        raise ValueError("cascade C3 completed-weather coverage rate identity is false")

    prior_only = _require_exact_boolean_series(
        c3["completed_prior_rounds_only"],
        "cascade C3 completed_prior_rounds_only",
    )
    excluded = _require_exact_boolean_series(
        c3["target_or_upcoming_realized_weather_excluded"],
        "cascade C3 target_or_upcoming_realized_weather_excluded",
    )
    strict = _require_exact_boolean_series(
        c3["strict_point_in_time"], "cascade C3 strict_point_in_time"
    )
    expected_strict = c3["variant"].astype(str).str.startswith("C3_pt_")
    if not prior_only.all() or not excluded.all() or not strict.eq(expected_strict).all():
        raise ValueError("cascade C3 completed-weather boolean lineage is false")
    if not c3["completed_prior_round_boundary"].astype(str).eq(
        C3_WEATHER_BOUNDARY
    ).all():
        raise ValueError("cascade C3 completed-weather boundary label is false")
    if not c3["current_round_weather_columns_role"].astype(str).eq(
        C3_CURRENT_WEATHER_ROLE
    ).all():
        raise ValueError("cascade C3 current-round weather role is false")

    event_start = pd.to_datetime(c3["event_start_date"], errors="coerce", utc=True)
    coefficient_cutoff = pd.to_datetime(
        c3["weather_coefficient_fit_cutoff"], errors="coerce", utc=True
    )
    if (
        event_start.isna().any()
        or coefficient_cutoff.isna().any()
        or coefficient_cutoff.ge(event_start).any()
    ):
        raise ValueError("cascade C3 weather coefficient cutoff is not strictly prior")
    if c3["weather_coefficient_source"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("cascade C3 weather coefficient source is missing")

    coefficient_record = completion_authority.get("weather_coefficients")
    if not isinstance(coefficient_record, Mapping):
        raise ValueError("cascade completion authority lacks weather coefficients")
    coefficient_path = Path(str(coefficient_record.get("path", ""))).resolve()
    coefficient_hash = str(coefficient_record.get("sha256", "")).lower()
    if (
        not coefficient_path.is_file()
        or not re.fullmatch(r"[0-9a-f]{64}", coefficient_hash)
        or _sha256(coefficient_path) != coefficient_hash
    ):
        raise ValueError("cascade authoritative weather coefficient bytes changed")
    row_paths = c3["weather_coefficients_path"].map(
        lambda value: Path(str(value)).resolve()
    )
    if not row_paths.eq(coefficient_path).all():
        raise ValueError("cascade C3 rows cite a non-authoritative coefficient path")
    if not c3["weather_coefficients_sha256"].astype(str).str.lower().eq(
        coefficient_hash
    ).all():
        raise ValueError("cascade C3 rows cite a non-authoritative coefficient hash")

    manifest_weather = cascade_manifest.get("inputs", {}).get("weather_coefficients")
    if not isinstance(manifest_weather, Mapping) or (
        Path(str(manifest_weather.get("path", ""))).resolve() != coefficient_path
        or str(manifest_weather.get("sha256", "")).lower() != coefficient_hash
    ):
        raise ValueError("cascade weather coefficient input differs from C3 authority")

    producer = cascade_audit.get("c3_weather_lineage")
    if not isinstance(producer, Mapping):
        raise ValueError("cascade audit lacks C3 weather lineage")
    for key, expected in (
        ("contract", C3_WEATHER_LINEAGE_CONTRACT),
        ("rows", int(len(c3))),
        ("arms", ["w0", "w2"]),
        ("weather_coefficients_path", str(coefficient_path)),
        ("weather_coefficients_sha256", coefficient_hash),
        ("completed_rounds_available_definition", "round_num_minus_one"),
        (
            "completed_rounds_used_definition",
            "nonmissing_strictly_shifted_total_state_rows",
        ),
        ("completed_prior_round_boundary", C3_WEATHER_BOUNDARY),
        ("current_round_weather_columns_role", C3_CURRENT_WEATHER_ROLE),
        (
            "completed_weather_observations_available_definition",
            "strictly_prior_rows_with_finite_wind_and_dew_deltas",
        ),
        (
            "completed_weather_state_rows_with_valid_coverage_definition",
            "strictly_prior_rows_with_finite_wind_dew_and_all_w2_state_components",
        ),
        (
            "completed_weather_used_by_w2_definition",
            "weather_arm_is_w2_and_strictly_prior_valid_wind_dew_w2_state_count_gt_zero",
        ),
        (
            "completed_prior_weather_coverage_rate_definition",
            "valid_strictly_prior_w2_weather_state_rows_divided_by_completed_rounds_available",
        ),
    ):
        if producer.get(key) != expected:
            raise ValueError(f"cascade producer C3 weather audit differs at {key!r}")
    _require_exact_bool(
        producer.get("target_or_upcoming_realized_weather_excluded"),
        True,
        "cascade producer C3 weather exclusion",
    )
    failures = producer.get("failures")
    if not isinstance(failures, Mapping) or not set(
        C3_WEATHER_LINEAGE_FAILURE_KEYS
    ).issubset(failures):
        raise ValueError("cascade producer C3 weather failure menu is incomplete")
    if any(
        not isinstance(value, (int, np.integer)) or int(value) != 0
        for value in failures.values()
    ):
        raise ValueError("cascade producer C3 weather audit has failures")

    state = cascade_audit.get("state")
    if not isinstance(state, Mapping):
        raise ValueError("cascade audit lacks completed-round state provenance")
    for key in (
        "weather_coefficient_boundary_violations",
        "completed_prior_round_boundary_violations",
    ):
        if not isinstance(state.get(key), (int, np.integer)) or int(state[key]) != 0:
            raise ValueError(f"cascade state audit {key} must be exact zero")
    state_expected = {
        "weather_coefficients_path": str(coefficient_path),
        "weather_coefficients_sha256": coefficient_hash,
        "weather_coefficient_fit_cutoff_column": "fit_max_round_date",
        "weather_coefficient_source_column": "coefficient_source",
        "completed_rounds_available_column": "completed_rounds_available",
        "completed_rounds_used_columns": {
            "w0": "completed_rounds_used_w0",
            "w2": "completed_rounds_used_w2",
        },
        "completed_weather_observations_available_column": (
            "completed_weather_observations_available"
        ),
        "completed_weather_state_rows_with_valid_coverage_column": (
            "completed_weather_state_rows_with_valid_coverage"
        ),
        "completed_weather_used_by_w2_definition": (
            "weather_arm_is_w2_and_strictly_prior_valid_wind_dew_w2_state_count_gt_zero"
        ),
        "completed_prior_weather_coverage_rate_column": (
            "completed_prior_weather_coverage_rate"
        ),
        "completed_prior_round_boundary": C3_WEATHER_BOUNDARY,
        "weather_state_source_columns": C3_WEATHER_SOURCE_COLUMNS,
    }
    for key, expected in state_expected.items():
        if state.get(key) != expected:
            raise ValueError(f"cascade state weather lineage differs at {key!r}")
    _require_exact_bool(
        state.get("target_or_upcoming_realized_weather_excluded"),
        True,
        "cascade state target/upcoming exclusion",
    )
    _require_exact_bool(
        state.get("target_or_upcoming_weather_feature_used"),
        False,
        "cascade state target/upcoming feature use",
    )

    c3["round_num"] = round_num
    c3["completed_weather_state_rows_used_by_w2"] = weather_used_by_w2
    c3["completed_weather_used_by_w2"] = weather_use_flag
    lineage_summary = (
        c3.loc[c3["weather_arm"].astype(str).eq("w2")]
        .groupby(
            ["model", "candidate", "event_key", "event_start_date", "round_num"],
            observed=True,
            dropna=False,
        )
        .agg(
            c3_player_prediction_rows=("model", "size"),
            lineage_completed_weather_observations=(
                "completed_weather_state_rows_used_by_w2",
                "sum",
            ),
            lineage_players_with_completed_weather=(
                "completed_weather_state_rows_used_by_w2",
                lambda values: int(pd.Series(values).gt(0).sum()),
            ),
        )
        .reset_index()
    )
    lineage_summary["lineage_completed_weather_any"] = lineage_summary[
        "lineage_completed_weather_observations"
    ].gt(0)
    return (
        {
            "contract": C3_WEATHER_LINEAGE_CONTRACT,
            "c3_rows": int(len(c3)),
            "w2_event_round_model_rows": int(len(lineage_summary)),
            "w2_event_round_models_with_completed_weather": int(
                lineage_summary["lineage_completed_weather_any"].sum()
            ),
            "weather_coefficients_path": str(coefficient_path),
            "weather_coefficients_sha256": coefficient_hash,
            "completed_prior_round_boundary": C3_WEATHER_BOUNDARY,
            "target_or_upcoming_realized_weather_excluded": True,
            "producer_lineage_audit_verified": True,
            "state_lineage_audit_verified": True,
        },
        lineage_summary,
    )


def pairwise_cascade_proper_deltas(
    probabilities: pd.DataFrame,
    *,
    challenger: str,
    reference: str,
    analysis_pair: str,
    comparison_kind: str,
    menu_scope: str = "exact",
) -> pd.DataFrame:
    keys = ["event_key", "event_start_date", "round_num", "pair_key"]
    _require(
        probabilities,
        [
            "model",
            *keys,
            "derived_score_outcome",
            "p1_probability",
            "p2_probability",
            "push_probability",
            "score_model_eligible",
        ],
        "bookless cascade three-way proper probabilities",
    )
    if probabilities.duplicated(["model", *keys]).any():
        raise ValueError(
            "bookless cascade three-way proper probabilities are ambiguous"
        )
    _require_exact_boolean_series(
        probabilities["score_model_eligible"],
        "bookless cascade score_model_eligible",
    )
    if menu_scope not in {"exact", "pairwise_common"}:
        raise ValueError(f"unknown cascade proper-score menu scope: {menu_scope}")
    exact, _, challenger_only, reference_only = _exact_menu(
        probabilities,
        challenger,
        reference,
        model_column="model",
        keys=keys,
    )
    if menu_scope == "exact" and not exact:
        raise ValueError(f"cascade proper-score menu mismatch: {challenger} vs {reference}")

    challenger_menu = probabilities.loc[
        probabilities["model"].eq(challenger), keys
    ].drop_duplicates()
    reference_menu = probabilities.loc[
        probabilities["model"].eq(reference), keys
    ].drop_duplicates()
    common_menu = challenger_menu.merge(
        reference_menu,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if common_menu.empty:
        raise ValueError(
            f"cascade proper-score common menu is empty: {challenger} vs {reference}"
        )
    if menu_scope == "pairwise_common":
        relevant = probabilities.loc[
            probabilities["model"].isin([challenger, reference])
        ].merge(common_menu, on=keys, how="inner", validate="many_to_one")
        probabilities = pd.concat(
            [
                probabilities.loc[
                    ~probabilities["model"].isin([challenger, reference])
                ],
                relevant,
            ],
            ignore_index=True,
            sort=False,
        )

    identity_columns = [*keys, "derived_score_outcome", "score_model_eligible"]
    identity_check = probabilities.loc[
        probabilities["model"].eq(challenger), identity_columns
    ].merge(
        probabilities.loc[
            probabilities["model"].eq(reference), identity_columns
        ],
        on=keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_challenger", "_reference"),
        indicator=True,
    )
    if not identity_check["_merge"].eq("both").all():
        raise ValueError("cascade quote menu shrank during proper-score comparison")
    if not np.allclose(
        identity_check["derived_score_outcome_challenger"],
        identity_check["derived_score_outcome_reference"],
        rtol=0.0,
        atol=1e-10,
        equal_nan=True,
    ):
        raise ValueError("cascade outcomes differ across proper-score comparison")
    if not identity_check["score_model_eligible_challenger"].eq(
        identity_check["score_model_eligible_reference"]
    ).all():
        raise ValueError(
            "cascade score-model proper menu differs across comparison"
        )
    def scored(model: str, prefix: str) -> pd.DataFrame:
        use = probabilities.loc[probabilities["model"].eq(model)].copy()
        eligible = _require_exact_boolean_series(
            use["score_model_eligible"],
            f"{model} score_model_eligible",
        )
        use[f"{prefix}_three_way_logloss"] = np.nan
        use[f"{prefix}_three_way_brier"] = np.nan
        if eligible.any():
            scoreable = use.loc[eligible]
            outcome = scoreable["derived_score_outcome"].to_numpy(float)
            if not np.isin(outcome, [0.0, 0.5, 1.0]).all():
                raise ValueError(
                    f"{model} has an invalid eligible three-way score outcome"
                )
            p1 = np.clip(scoreable["p1_probability"].to_numpy(float), EPS, 1.0)
            p2 = np.clip(scoreable["p2_probability"].to_numpy(float), EPS, 1.0)
            push = np.clip(
                scoreable["push_probability"].to_numpy(float), EPS, 1.0
            )
            chosen = np.where(
                np.isclose(outcome, 1.0),
                p1,
                np.where(
                    np.isclose(outcome, 0.0),
                    p2,
                    push,
                ),
            )
            use.loc[eligible, f"{prefix}_three_way_logloss"] = -np.log(chosen)
            use.loc[eligible, f"{prefix}_three_way_brier"] = (
                np.square(p1 - np.isclose(outcome, 1.0).astype(float))
                + np.square(p2 - np.isclose(outcome, 0.0).astype(float))
                + np.square(push - np.isclose(outcome, 0.5).astype(float))
            )
        return use[
            [
                *keys,
                "derived_score_outcome",
                "score_model_eligible",
                f"{prefix}_three_way_logloss",
                f"{prefix}_three_way_brier",
            ]
        ]

    left = scored(challenger, "challenger")
    right = scored(reference, "reference")
    merged = left.merge(
        right,
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("", "_reference"),
    )
    if not np.allclose(
        merged["derived_score_outcome"],
        merged["derived_score_outcome_reference"],
        rtol=0.0,
        atol=1e-10,
        equal_nan=True,
    ):
        raise ValueError("cascade outcomes differ across proper-score comparison")
    rows: list[pd.DataFrame] = []
    for metric in ("three_way_logloss", "three_way_brier"):
        eligible = merged["score_model_eligible"].fillna(False).astype(bool) & merged[
            "score_model_eligible_reference"
        ].fillna(False).astype(bool)
        use = merged.loc[eligible].copy()
        if use.empty:
            continue
        event = use.groupby(
            ["event_key", "event_start_date", "round_num"], observed=True
        ).agg(
            common_pairs=("pair_key", "size"),
            challenger_score=(f"challenger_{metric}", "mean"),
            reference_score=(f"reference_{metric}", "mean"),
        ).reset_index()
        event["event_delta"] = event["challenger_score"] - event["reference_score"]
        event["analysis_block"] = "cascade_w2_completed_round_weather_normalization"
        event["analysis_pair"] = analysis_pair
        event["comparison_kind"] = comparison_kind
        event["challenger"] = challenger
        event["reference"] = reference
        event["metric"] = f"{metric}_delta"
        event["higher_is_better"] = False
        event["round_market"] = "r" + event["round_num"].astype(str)
        event["weather_information_scope"] = np.where(
            event["round_num"].eq(1),
            "historical_training_only_no_current_completed_round",
            "historical_training_plus_completed_rounds",
        )
        event["proper_quote_menu_scope"] = menu_scope
        event["challenger_full_quote_keys"] = int(len(challenger_menu))
        event["reference_full_quote_keys"] = int(len(reference_menu))
        event["common_quote_keys"] = int(len(common_menu))
        event["challenger_quote_keys_excluded_for_common"] = int(challenger_only)
        event["reference_quote_keys_excluded_for_common"] = int(reference_only)
        rows.append(event)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_cascade_proper_deltas(
    probabilities: pd.DataFrame, pairing: pd.DataFrame
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    control_models = probabilities.loc[
        probabilities["model"].eq(CASCADE_CONTROL), "model"
    ].unique()
    if len(control_models) != 1:
        raise ValueError("cascade probabilities lack the unique C0 control")
    for row in pairing.loc[pairing["quote_menu_status"].eq("exact")].to_dict(
        orient="records"
    ):
        w2 = str(row["w2_model"])
        w0 = str(row["w0_model"])
        pair_id = f"{w2}__vs__{w0}"
        for challenger, reference, kind in (
            (w2, w0, "w2_vs_matched_w0"),
            (w2, CASCADE_CONTROL, "w2_vs_current_cascade_control"),
            (w0, CASCADE_CONTROL, "w0_vs_current_cascade_control"),
        ):
            piece = pairwise_cascade_proper_deltas(
                probabilities,
                challenger=challenger,
                reference=reference,
                analysis_pair=pair_id,
                comparison_kind=kind,
                menu_scope=(
                    "exact"
                    if reference == w0
                    else "pairwise_common"
                ),
            )
            if not piece.empty:
                # Weather lineage belongs to the W2 arm governing this
                # analysis pair, not necessarily to the proper-score
                # challenger.  The W0-vs-C0 decomposition intentionally keeps
                # challenger=W0 while using W2 only as the post-sealed weather
                # lineage/attribution model.
                piece["model"] = w2
                piece["weather_lineage_model_role"] = (
                    "w2_arm_for_analysis_pair_not_score_challenger"
                )
                pieces.append(piece)
    output = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if not output.empty:
        expected_model = output["analysis_pair"].astype(str).str.split(
            "__vs__", n=1
        ).str[0]
        if not output["model"].astype(str).eq(expected_model).all():
            raise ValueError("cascade proper-score W2 weather-lineage model is false")
    return output


SETTLEMENT_NUMERIC_TOLERANCE = 1e-10
_WIN_OUTCOME_WORDS = frozenset({"w", "win", "won", "winner", "1", "1.0"})
_LOSS_OUTCOME_WORDS = frozenset({"l", "loss", "lost", "lose", "0", "0.0"})
_TIE_OUTCOME_WORDS = frozenset({"t", "tie", "tied", "push", "draw", "0.5"})
_VOID_OUTCOME_WORDS = frozenset(
    {"void", "voided", "refund", "refunded", "cancelled", "canceled", "no action"}
)


def _stable_bet_id(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def _ties_lose_authority(value: Any) -> bool:
    text = str(value).strip().casefold().replace("_", " ").replace("-", " ")
    return any(
        marker in text
        for marker in ("separate", "ties lose", "tie loses", "three way", "3 way")
    )


def _numeric_match_error(
    observed: pd.Series,
    expected: pd.Series,
    *,
    label: str,
    allow_both_missing: bool = False,
    tolerance: float = SETTLEMENT_NUMERIC_TOLERANCE,
) -> float:
    observed_numeric, observed_missing = _finite_numeric_authority_series(
        observed, label=f"{label} observed"
    )
    expected_numeric, expected_missing = _finite_numeric_authority_series(
        expected, label=f"{label} expected authority"
    )
    if allow_both_missing:
        missing_bad = observed_missing.ne(expected_missing)
        compare = ~observed_missing & ~expected_missing
    else:
        missing_bad = observed_missing | expected_missing
        compare = ~missing_bad
    if missing_bad.any():
        raise ValueError(f"{label} has a missing/non-numeric authority mismatch")
    error = (observed_numeric.loc[compare] - expected_numeric.loc[compare]).abs()
    max_error = float(error.max()) if len(error) else 0.0
    if not math.isfinite(max_error) or max_error > tolerance:
        raise ValueError(
            f"{label} differs from normalized quote authority: "
            f"max_abs_error={max_error}"
        )
    return max_error


def _exact_text_match(
    observed: pd.Series,
    expected: pd.Series,
    *,
    label: str,
    casefold: bool = False,
    allow_both_missing: bool = False,
) -> None:
    observed_missing = observed.isna()
    expected_missing = expected.isna()
    if allow_both_missing:
        if observed_missing.ne(expected_missing).any():
            raise ValueError(f"{label} has a missing authority mismatch")
        compare = ~observed_missing & ~expected_missing
    else:
        if observed_missing.any() or expected_missing.any():
            raise ValueError(f"{label} may not be missing")
        compare = pd.Series(True, index=observed.index)
    left = observed.loc[compare].astype(str).str.strip()
    right = expected.loc[compare].astype(str).str.strip()
    if casefold:
        left = left.str.casefold()
        right = right.str.casefold()
    if not left.eq(right).all():
        raise ValueError(f"{label} differs from normalized quote authority")


def _semantic_outcome(
    numeric: Any,
    text: Any,
    *,
    tie_rule_loses: bool,
    force_void: bool = False,
) -> str:
    if force_void:
        return "void"
    normalized = "" if text is None or pd.isna(text) else str(text).strip().casefold()
    text_class = (
        "void"
        if normalized in _VOID_OUTCOME_WORDS
        else "tie"
        if normalized in _TIE_OUTCOME_WORDS
        else "win"
        if normalized in _WIN_OUTCOME_WORDS
        else "loss"
        if normalized in _LOSS_OUTCOME_WORDS
        else None
    )
    numeric_series, numeric_missing = _finite_numeric_authority_series(
        pd.Series([numeric]), label="normalized quote outcome"
    )
    numeric_value = numeric_series.iloc[0]
    numeric_class = None
    if not bool(numeric_missing.iloc[0]):
        if np.isclose(float(numeric_value), 1.0, rtol=0.0, atol=1e-12):
            numeric_class = "win"
        elif np.isclose(float(numeric_value), 0.0, rtol=0.0, atol=1e-12):
            numeric_class = "loss"
        elif np.isclose(float(numeric_value), 0.5, rtol=0.0, atol=1e-12):
            numeric_class = "tie"
        else:
            raise ValueError("normalized quote outcome is outside {0,0.5,1}")
    # The frozen base-market producer intentionally preserves a raw 0.5
    # matchup result while writing ``void`` as the settlement text when the
    # book's tie rule voids the wager.  This is the sole valid cross-class
    # representation: the settlement engine gives explicit void text
    # precedence, while the numeric field still proves the underlying tie.
    numeric_tie_text_void = text_class == "void" and numeric_class == "tie"
    if (
        text_class is not None
        and numeric_class is not None
        and text_class != numeric_class
        and not numeric_tie_text_void
    ):
        raise ValueError("normalized quote numeric/text outcomes disagree")
    if numeric_tie_text_void:
        return "void"
    result = numeric_class or text_class
    if result is None:
        raise ValueError("selected bet has no authoritative graded/void outcome")
    if result == "tie":
        return "tie_loss" if tie_rule_loses else "push"
    return result


def _validate_settlement_money(
    joined: pd.DataFrame,
    *,
    label: str,
) -> dict[str, Any]:
    required = [
        "stake",
        "pnl",
        "sizing",
        "execution_odds",
        "close_odds",
        "settlement_status",
        "is_graded",
        "_authority_execution_odds",
        "_authority_status",
    ]
    _require(joined, required, label)
    graded = _require_exact_boolean_series(joined["is_graded"], f"{label} is_graded")
    if not graded.all():
        raise ValueError(f"{label} contains an ungraded selected bet")
    numeric = joined[["stake", "pnl", "execution_odds", "close_odds"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError(f"{label} contains non-finite money or execution odds")
    if numeric["stake"].le(0.0).any():
        raise ValueError(f"{label} stake must be finite and strictly positive")
    valid_odds = (numeric["execution_odds"].le(-100.0)) | numeric[
        "execution_odds"
    ].ge(100.0)
    if not valid_odds.all():
        raise ValueError(f"{label} execution odds are not valid American prices")
    execution_error = _numeric_match_error(
        numeric["execution_odds"],
        joined["_authority_execution_odds"],
        label=f"{label} execution_odds",
    )
    close_error = _numeric_match_error(
        numeric["close_odds"],
        joined["_authority_execution_odds"],
        label=f"{label} settlement close_odds",
    )
    sizing = joined["sizing"].astype(str)
    if not sizing.isin(EXPECTED_SEALED_SIZINGS).all():
        raise ValueError(f"{label} has an unknown sizing")
    expected_stake = pd.Series(
        np.where(
            sizing.eq("flat_1u"),
            1.0,
            np.where(
                numeric["execution_odds"].lt(0.0),
                numeric["execution_odds"].abs() / 100.0,
                1.0,
            ),
        ),
        index=joined.index,
    )
    stake_error = _numeric_match_error(
        numeric["stake"], expected_stake, label=f"{label} stake"
    )
    expected_status = joined["_authority_status"].astype(str)
    if not joined["settlement_status"].astype(str).eq(expected_status).all():
        raise ValueError(f"{label} settlement_status differs from quote outcome")
    profit_multiple = np.where(
        numeric["execution_odds"].gt(0.0),
        numeric["execution_odds"] / 100.0,
        100.0 / numeric["execution_odds"].abs(),
    )
    expected_pnl = np.select(
        [
            expected_status.eq("win"),
            expected_status.isin(["loss", "tie_loss"]),
            expected_status.isin(["push", "void"]),
        ],
        [numeric["stake"] * profit_multiple, -numeric["stake"], 0.0],
        default=np.nan,
    )
    if not np.isfinite(expected_pnl).all():
        raise ValueError(f"{label} contains an unsupported settlement status")
    pnl_error = _numeric_match_error(
        numeric["pnl"],
        pd.Series(expected_pnl, index=joined.index),
        label=f"{label} P&L",
    )
    return_error = 0.0
    if "return_amount" in joined:
        expected_return = numeric["stake"] + pd.Series(
            expected_pnl, index=joined.index
        )
        return_error = _numeric_match_error(
            joined["return_amount"],
            expected_return,
            label=f"{label} return_amount",
        )
    paid_fraction_error = 0.0
    if "paid_fraction" in joined:
        expected_paid_fraction = pd.Series(
            np.select(
                [
                    expected_status.eq("win"),
                    expected_status.isin(["loss", "tie_loss"]),
                ],
                [1.0, 0.0],
                default=np.nan,
            ),
            index=joined.index,
        )
        paid_fraction_error = _numeric_match_error(
            joined["paid_fraction"],
            expected_paid_fraction,
            label=f"{label} paid_fraction",
            allow_both_missing=True,
        )
    return {
        "rows": int(len(joined)),
        "wins": int(expected_status.eq("win").sum()),
        "losses": int(expected_status.eq("loss").sum()),
        "tie_losses": int(expected_status.eq("tie_loss").sum()),
        "pushes": int(expected_status.eq("push").sum()),
        "voids": int(expected_status.eq("void").sum()),
        "execution_odds_max_abs_error": execution_error,
        "settlement_close_odds_max_abs_error": close_error,
        "stake_max_abs_error": stake_error,
        "pnl_max_abs_error": pnl_error,
        "return_amount_max_abs_error": return_error,
        "paid_fraction_max_abs_error": paid_fraction_error,
        "normalized_quote_book_side_outcome_tie_rule_exact": True,
    }


def validate_sealed_bet_settlement_authority(
    bets: pd.DataFrame, normalized_quotes: pd.DataFrame
) -> dict[str, Any]:
    """Rebuild every sealed selected-bet settlement from its frozen quote."""

    _require(
        bets,
        [
            "candidate",
            "selection_key",
            "event_key",
            "event_start_date",
            "market",
            "pair_key",
            "selection_id",
            "opponent_id",
            "book",
            "price_basis",
            "line_policy",
            "execution_selection_odds",
            "execution_opponent_odds",
            "same_book_close_selection_odds",
            "same_book_close_opponent_odds",
            "close_odds",
            "outcome_numeric",
            "outcome",
            "tie_rule",
            "settlement_status",
            "is_graded",
            "stake",
            "pnl",
            "sizing",
        ],
        "sealed bet settlement authority",
    )
    quote_columns = [
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
        "p1_outcome_text",
        "p2_outcome_text",
        "tie_rule",
    ]
    _require(normalized_quotes, quote_columns, "sealed normalized quote settlement")
    quotes = normalized_quotes[quote_columns].copy()
    if quotes["quote_id"].astype(str).duplicated().any():
        raise ValueError("sealed normalized quote settlement IDs are ambiguous")
    work = bets.copy()
    parsed_quote_ids: list[str] = []
    parsed_sides: list[str] = []
    for candidate, selection_key in zip(work["candidate"], work["selection_key"]):
        prefix = f"{candidate}|"
        key = str(selection_key)
        if not key.startswith(prefix) or "|" not in key[len(prefix) :]:
            raise ValueError("sealed bet selection_key has a forged candidate/quote ID")
        quote_id, side = key[len(prefix) :].rsplit("|", 1)
        if side not in {"p1", "p2"} or not quote_id:
            raise ValueError("sealed bet selection_key has a forged quote side")
        parsed_quote_ids.append(quote_id)
        parsed_sides.append(side)
    work["_quote_id"] = parsed_quote_ids
    work["_side"] = parsed_sides
    authority = quotes.rename(columns={column: f"_q_{column}" for column in quote_columns})
    joined = work.merge(
        authority,
        left_on="_quote_id",
        right_on="_q_quote_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("sealed bet references a quote ID absent from authority")
    side_p1 = joined["_side"].eq("p1")
    selected_id = joined["_q_p1_id"].where(side_p1, joined["_q_p2_id"])
    opponent_id = joined["_q_p2_id"].where(side_p1, joined["_q_p1_id"])
    for observed, expected, label in (
        (joined["event_key"], joined["_q_mapped_event_key"], "event_key"),
        (joined["market"], joined["_q_market"], "market"),
        (joined["book"], joined["_q_book"], "book"),
        (joined["pair_key"], joined["_q_pair_key"], "pair_key"),
        (
            joined["selection_id"].map(_stable_bet_id),
            selected_id.map(_stable_bet_id),
            "selection_id",
        ),
        (
            joined["opponent_id"].map(_stable_bet_id),
            opponent_id.map(_stable_bet_id),
            "opponent_id",
        ),
    ):
        _exact_text_match(observed, expected, label=f"sealed bet {label}")
    bet_dates = pd.to_datetime(joined["event_start_date"], errors="coerce").dt.normalize()
    quote_dates = pd.to_datetime(
        joined["_q_event_start_date"], errors="coerce"
    ).dt.normalize()
    if bet_dates.isna().any() or quote_dates.isna().any() or not bet_dates.eq(
        quote_dates
    ).all():
        raise ValueError("sealed bet event_start_date differs from quote authority")
    price = joined["price_basis"].astype(str)
    if not price.isin(EXPECTED_SEALED_PRICE_BASES).all():
        raise ValueError("sealed bet has an unknown price basis")
    fixed = joined["line_policy"].astype(str).isin(EXPECTED_BOOKS)
    if not joined.loc[fixed, "line_policy"].astype(str).eq(
        joined.loc[fixed, "book"].astype(str)
    ).all():
        raise ValueError("sealed fixed-book bet uses another book")
    selection_open = joined["_q_p1_open"].where(side_p1, joined["_q_p2_open"])
    opponent_open = joined["_q_p2_open"].where(side_p1, joined["_q_p1_open"])
    selection_close = joined["_q_p1_close"].where(side_p1, joined["_q_p2_close"])
    opponent_close = joined["_q_p2_close"].where(side_p1, joined["_q_p1_close"])
    joined["_authority_execution_odds"] = selection_close.where(
        price.eq("close"), selection_open
    )
    joined["_authority_opponent_odds"] = opponent_close.where(
        price.eq("close"), opponent_open
    )
    _numeric_match_error(
        joined["execution_selection_odds"],
        joined["_authority_execution_odds"],
        label="sealed bet execution_selection_odds",
    )
    _numeric_match_error(
        joined["execution_opponent_odds"],
        joined["_authority_opponent_odds"],
        label="sealed bet execution_opponent_odds",
    )
    _numeric_match_error(
        joined["same_book_close_selection_odds"],
        selection_close,
        label="sealed bet same-book close selection odds",
    )
    _numeric_match_error(
        joined["same_book_close_opponent_odds"],
        opponent_close,
        label="sealed bet same-book close opponent odds",
    )
    joined["execution_odds"] = joined["execution_selection_odds"]
    _exact_text_match(
        joined["tie_rule"], joined["_q_tie_rule"], label="sealed bet tie_rule"
    )
    selected_outcome = joined["_q_p1_outcome"].where(
        side_p1, joined["_q_p2_outcome"]
    )
    selected_outcome_text = joined["_q_p1_outcome_text"].where(
        side_p1, joined["_q_p2_outcome_text"]
    )
    _numeric_match_error(
        joined["outcome_numeric"],
        selected_outcome,
        label="sealed bet outcome_numeric",
        allow_both_missing=True,
    )
    _exact_text_match(
        joined["outcome"],
        selected_outcome_text,
        label="sealed bet outcome text",
        allow_both_missing=True,
    )
    joined["_authority_status"] = [
        _semantic_outcome(
            outcome,
            outcome_text,
            tie_rule_loses=_ties_lose_authority(tie_rule),
        )
        for outcome, outcome_text, tie_rule in zip(
            selected_outcome, selected_outcome_text, joined["_q_tie_rule"]
        )
    ]
    numeric_outcome = pd.to_numeric(selected_outcome, errors="coerce")
    outcome_text = selected_outcome_text.fillna("").astype(str).str.strip().str.casefold()
    numeric_tie_text_void_rows = int(
        (
            np.isclose(numeric_outcome, 0.5, rtol=0.0, atol=1e-12)
            & outcome_text.isin(_VOID_OUTCOME_WORDS)
        ).sum()
    )
    return {
        **_validate_settlement_money(joined, label="sealed bet settlement"),
        "numeric_tie_text_void_rows": numeric_tie_text_void_rows,
        "numeric_tie_text_void_is_producer_settlement_encoding": True,
        "selection_key_quote_id_side_reconstructed": True,
    }


def validate_cascade_bet_settlement_authority(
    bets: pd.DataFrame,
    normalized_quotes: pd.DataFrame,
    raw_probabilities: pd.DataFrame,
) -> dict[str, Any]:
    """Rebuild every cascade selected-bet settlement from quote and score state."""

    required_bets = [
        "model",
        "event_key",
        "event_start_date",
        "round_num",
        "round_market",
        "pair_key",
        "selection_id",
        "opponent_id",
        "book",
        "price_point",
        "line_policy",
        "execution_odds",
        "close_odds",
        "archived_open_odds",
        "archived_close_odds",
        "outcome_numeric",
        "outcome",
        "tie_rule",
        "tie_rule_ties_lose",
        "settlement_status",
        "is_graded",
        "stake",
        "pnl",
        "sizing",
    ]
    _require(bets, required_bets, "cascade bet settlement authority")
    quote_columns = [
        "quote_id",
        "event_key",
        "event_start_date",
        "round_num",
        "round_market",
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
        "tie_rule",
    ]
    _require(normalized_quotes, quote_columns, "cascade normalized quote settlement")
    quotes = normalized_quotes[quote_columns].copy()
    if quotes["quote_id"].astype(str).duplicated().any():
        raise ValueError("cascade normalized quote settlement IDs are ambiguous")
    classification_columns = [
        "model",
        "quote_id",
        "no_score_void",
        "other_void_with_scores",
        "actual_score_tie",
    ]
    _require(raw_probabilities, classification_columns, "cascade settlement classification")
    classifications = raw_probabilities[classification_columns].drop_duplicates()
    if classifications.duplicated(["model", "quote_id"]).any():
        raise ValueError("cascade settlement classification keys are ambiguous")
    for column in classification_columns[2:]:
        classifications[column] = _require_exact_boolean_series(
            classifications[column], f"cascade settlement classification {column}"
        )
    work = bets.copy()
    work["_quote_id"] = (
        work["event_key"].astype(str)
        + "|"
        + work["round_market"].astype(str)
        + "|"
        + work["book"].astype(str)
        + "|"
        + work["pair_key"].astype(str)
    )
    if "quote_id" in work and not work["quote_id"].astype(str).eq(
        work["_quote_id"]
    ).all():
        raise ValueError("cascade bet carries a forged quote_id")
    authority = quotes.rename(columns={column: f"_q_{column}" for column in quote_columns})
    joined = work.merge(
        authority,
        left_on="_quote_id",
        right_on="_q_quote_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("cascade bet references a quote ID absent from authority")
    joined = joined.merge(
        classifications.rename(columns={"quote_id": "_quote_id"}),
        on=["model", "_quote_id"],
        how="left",
        validate="many_to_one",
    )
    if joined[classification_columns[2:]].isna().any().any():
        raise ValueError("cascade bet lacks a model/quote settlement classification")
    side_p1 = joined["selection_id"].map(_stable_bet_id).eq(
        joined["_q_p1_id"].map(_stable_bet_id)
    ) & joined["opponent_id"].map(_stable_bet_id).eq(
        joined["_q_p2_id"].map(_stable_bet_id)
    )
    side_p2 = joined["selection_id"].map(_stable_bet_id).eq(
        joined["_q_p2_id"].map(_stable_bet_id)
    ) & joined["opponent_id"].map(_stable_bet_id).eq(
        joined["_q_p1_id"].map(_stable_bet_id)
    )
    if not (side_p1 ^ side_p2).all():
        raise ValueError("cascade bet selection/opponent side is not authoritative")
    for observed, expected, label in (
        (joined["event_key"], joined["_q_event_key"], "event_key"),
        (joined["round_market"], joined["_q_round_market"], "round_market"),
        (joined["book"], joined["_q_book"], "book"),
        (joined["pair_key"], joined["_q_pair_key"], "pair_key"),
    ):
        _exact_text_match(observed, expected, label=f"cascade bet {label}")
    observed_round, observed_round_missing = _finite_numeric_authority_series(
        joined["round_num"], label="cascade bet round_num"
    )
    authority_round, authority_round_missing = _finite_numeric_authority_series(
        joined["_q_round_num"], label="cascade quote round_num authority"
    )
    if (
        observed_round_missing.any()
        or authority_round_missing.any()
        or not observed_round.eq(authority_round).all()
    ):
        raise ValueError("cascade bet round_num differs from quote authority")
    bet_dates = pd.to_datetime(joined["event_start_date"], errors="coerce").dt.normalize()
    quote_dates = pd.to_datetime(
        joined["_q_event_start_date"], errors="coerce"
    ).dt.normalize()
    if bet_dates.isna().any() or quote_dates.isna().any() or not bet_dates.eq(
        quote_dates
    ).all():
        raise ValueError("cascade bet event_start_date differs from quote authority")
    price = joined["price_point"].astype(str)
    if not price.isin(EXPECTED_CASCADE_PRICE_POINTS).all():
        raise ValueError("cascade bet has an unknown price point")
    fixed = joined["line_policy"].astype(str).isin(EXPECTED_BOOKS)
    if not joined.loc[fixed, "line_policy"].astype(str).eq(
        joined.loc[fixed, "book"].astype(str)
    ).all():
        raise ValueError("cascade fixed-book bet uses another book")
    selection_open = joined["_q_p1_open"].where(side_p1, joined["_q_p2_open"])
    selection_close = joined["_q_p1_close"].where(side_p1, joined["_q_p2_close"])
    joined["_authority_execution_odds"] = selection_close.where(
        price.eq("close"), selection_open
    )
    _numeric_match_error(
        joined["archived_open_odds"],
        selection_open,
        label="cascade bet archived_open_odds",
        allow_both_missing=True,
    )
    _numeric_match_error(
        joined["archived_close_odds"],
        selection_close,
        label="cascade bet archived_close_odds",
        allow_both_missing=True,
    )
    _exact_text_match(
        joined["tie_rule"], joined["_q_tie_rule"], label="cascade bet tie_rule"
    )
    expected_ties_lose = joined["_q_tie_rule"].map(_ties_lose_authority)
    observed_ties_lose = _require_exact_boolean_series(
        joined["tie_rule_ties_lose"], "cascade bet tie_rule_ties_lose"
    )
    if not observed_ties_lose.eq(expected_ties_lose).all():
        raise ValueError("cascade bet tie_rule_ties_lose differs from quote authority")
    selected_outcome = joined["_q_p1_outcome"].where(
        side_p1, joined["_q_p2_outcome"]
    )
    _numeric_match_error(
        joined["outcome_numeric"],
        selected_outcome,
        label="cascade bet outcome_numeric",
        allow_both_missing=True,
    )
    force_void = joined["no_score_void"].astype(bool) | joined[
        "other_void_with_scores"
    ].astype(bool)
    expected_status: list[str] = []
    expected_outcome_text: list[str] = []
    for outcome, tie_loses, void, actual_tie in zip(
        selected_outcome,
        expected_ties_lose,
        force_void,
        joined["actual_score_tie"].astype(bool),
    ):
        if void:
            status = "void"
            outcome_text = "void"
        elif actual_tie:
            status = "tie_loss" if tie_loses else "push"
            outcome_text = "tie"
        else:
            status = _semantic_outcome(
                outcome, "", tie_rule_loses=bool(tie_loses)
            )
            outcome_text = "win" if status == "win" else "loss"
        expected_status.append(status)
        expected_outcome_text.append(outcome_text)
    joined["_authority_status"] = expected_status
    if not joined["outcome"].astype(str).str.strip().str.casefold().eq(
        pd.Series(expected_outcome_text, index=joined.index)
    ).all():
        raise ValueError("cascade bet outcome text differs from quote/score authority")
    return {
        **_validate_settlement_money(joined, label="cascade bet settlement"),
        "quote_id_reconstructed_from_event_round_book_pair": True,
        "void_status_bound_to_validated_score_classification": True,
    }


def _validate_cascade_bets(frame: pd.DataFrame) -> pd.DataFrame:
    required = [
        "model",
        "candidate",
        "variant",
        "event_key",
        "event_start_date",
        "round_num",
        "round_market",
        "pair_key",
        "selection_id",
        "opponent_id",
        "price_point",
        "line_policy",
        "sizing",
        "opportunity_mode",
        "stake",
        "pnl",
        "is_graded",
    ]
    _require(frame, required, "cascade bets")
    output = frame.copy()
    output["event_start_date"] = pd.to_datetime(
        output["event_start_date"], errors="coerce"
    ).dt.normalize()
    output["round_num"] = pd.to_numeric(output["round_num"], errors="coerce")
    if output["event_start_date"].isna().any() or output["round_num"].isna().any():
        raise ValueError("cascade bets contain invalid event dates or rounds")
    graded_all = _require_exact_boolean_series(
        output["is_graded"], "cascade bets.is_graded"
    )
    output = output.loc[
        output["event_start_date"].between(SEALED_START, SEALED_END)
        & output["round_num"].isin([1, 2, 3, 4])
    ].copy()
    output["round_num"] = output["round_num"].astype(int)
    _numeric(output, ["stake", "pnl"])
    graded = graded_all.loc[output.index]
    if not graded.all():
        raise ValueError("cascade primary bets contain ungraded rows")
    _require_finite_numeric_columns(
        output,
        ["stake", "pnl"],
        label="cascade bets",
        nonnegative=["stake"],
    )
    identity = [
        "model",
        "event_key",
        "round_num",
        "pair_key",
        "price_point",
        "line_policy",
        "sizing",
        "opportunity_mode",
    ]
    if output[identity].isna().any().any() or output.duplicated(identity).any():
        raise ValueError("cascade bets contain duplicate model/pair/slice rows")
    return output


CASCADE_QUOTE_MENU_KEYS = (
    "event_key",
    "event_start_date",
    "round_num",
    "pair_key",
)


def _cascade_three_arm_common_quote_keys(
    quote_keys: pd.DataFrame,
    *,
    w2_model: str,
    w0_model: str,
    price_point: str,
    line_policy: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return the exact pair-specific W2/W0/C0 common executable menu."""

    required = [
        "model",
        *CASCADE_QUOTE_MENU_KEYS,
        "price_point",
        "line_policy",
    ]
    _require(quote_keys, required, "cascade executable quote keys")
    models = (str(w2_model), str(w0_model), CASCADE_CONTROL)
    quote_slice = quote_keys.loc[
        quote_keys["price_point"].astype(str).eq(str(price_point))
        & quote_keys["line_policy"].astype(str).eq(str(line_policy))
        & quote_keys["model"].astype(str).isin(models),
        required,
    ].copy()
    quote_slice["model"] = quote_slice["model"].astype(str)
    quote_slice["event_start_date"] = pd.to_datetime(
        quote_slice["event_start_date"], errors="coerce"
    ).dt.normalize()
    if (
        quote_slice.empty
        or quote_slice[required].isna().any().any()
        or quote_slice.duplicated(
            ["model", *CASCADE_QUOTE_MENU_KEYS, "price_point", "line_policy"]
        ).any()
    ):
        raise ValueError("cascade money quote menu is empty or ambiguous")
    model_menus = {
        model: quote_slice.loc[
            quote_slice["model"].eq(model), list(CASCADE_QUOTE_MENU_KEYS)
        ].drop_duplicates()
        for model in models
    }
    if any(menu.empty for menu in model_menus.values()):
        raise ValueError("cascade money quote menu is empty for a required arm")
    if len(model_menus[w2_model]) != len(model_menus[w0_model]) or not model_menus[
        w2_model
    ].merge(
        model_menus[w0_model],
        on=list(CASCADE_QUOTE_MENU_KEYS),
        how="outer",
        indicator=True,
        validate="one_to_one",
    )["_merge"].eq("both").all():
        raise ValueError("cascade W2/W0 executable quote menus differ")
    common_menu = model_menus[w2_model]
    for model in (w0_model, CASCADE_CONTROL):
        common_menu = common_menu.merge(
            model_menus[model],
            on=list(CASCADE_QUOTE_MENU_KEYS),
            how="inner",
            validate="one_to_one",
        )
    if common_menu.empty:
        raise ValueError("cascade W2/W0/C0 common money quote menu is empty")
    common_quote_keys = quote_slice.merge(
        common_menu,
        on=list(CASCADE_QUOTE_MENU_KEYS),
        how="inner",
        validate="many_to_one",
    )
    counts = {
        "w2_full_quote_keys": int(len(model_menus[w2_model])),
        "w0_full_quote_keys": int(len(model_menus[w0_model])),
        "control_full_quote_keys": int(len(model_menus[CASCADE_CONTROL])),
        "three_arm_common_quote_keys": int(len(common_menu)),
        "w2_quote_keys_excluded_for_control_common": int(
            len(model_menus[w2_model]) - len(common_menu)
        ),
        "control_quote_keys_excluded_for_common": int(
            len(model_menus[CASCADE_CONTROL]) - len(common_menu)
        ),
    }
    return common_quote_keys, counts


def _cascade_event_portfolio(
    bets: pd.DataFrame,
    quote_keys: pd.DataFrame,
    *,
    model: str,
    slice_values: Mapping[str, str],
) -> pd.DataFrame:
    quote_filter = quote_keys["model"].eq(model)
    for column in ("price_point", "line_policy"):
        quote_filter &= quote_keys[column].astype(str).eq(str(slice_values[column]))
    allowed_columns = [
        "model",
        *CASCADE_QUOTE_MENU_KEYS,
        "price_point",
        "line_policy",
    ]
    allowed = quote_keys.loc[quote_filter, allowed_columns].drop_duplicates()
    if allowed.duplicated(allowed_columns).any():
        raise ValueError("cascade common quote menu contains duplicate keys")
    universe = allowed[
        ["event_key", "event_start_date", "round_num"]
    ].drop_duplicates()
    if universe.empty:
        raise ValueError(
            f"cascade quote universe is empty for {model}: {dict(slice_values)}"
        )
    use = bets.loc[bets["model"].eq(model)].copy()
    universe["event_start_date"] = pd.to_datetime(
        universe["event_start_date"], errors="coerce"
    ).dt.normalize()
    use["event_start_date"] = pd.to_datetime(
        use["event_start_date"], errors="coerce"
    ).dt.normalize()
    allowed["event_start_date"] = pd.to_datetime(
        allowed["event_start_date"], errors="coerce"
    ).dt.normalize()
    for column, value in slice_values.items():
        use = use.loc[use[column].astype(str).eq(str(value))]
    # A zero-filled common universe is insufficient: the wager ledger itself
    # must be semijoined to the exact three-arm pair menu.  Otherwise bets on
    # prediction-covered W2/W0 quotes absent from C0 contaminate control-relative
    # P&L while the output merely claims a common-menu scope.
    use = use.merge(
        allowed,
        on=allowed_columns,
        how="inner",
        validate="many_to_one",
    )
    event = use.groupby(
        ["event_key", "event_start_date", "round_num"], observed=True
    ).agg(
        bets=("pnl", "size"),
        stake=("stake", "sum"),
        pnl=("pnl", "sum"),
    ).reset_index()
    output = universe.merge(
        event,
        on=["event_key", "event_start_date", "round_num"],
        how="left",
        validate="one_to_one",
    )
    output[["bets", "stake", "pnl"]] = output[["bets", "stake", "pnl"]].fillna(0.0)
    return output


def build_cascade_money_deltas(
    bets: pd.DataFrame,
    quote_keys: pd.DataFrame,
    pairing: pd.DataFrame,
    *,
    expected_price_points: Sequence[str],
    expected_line_policies: Sequence[str],
    expected_sizings: Sequence[str] = ("flat_1u", "production_risk_to_win"),
    expected_opportunity_modes: Sequence[str] = ("ev_ge_3pct",),
) -> pd.DataFrame:
    sizings = tuple(map(str, expected_sizings))
    opportunity_modes = tuple(map(str, expected_opportunity_modes))
    if not sizings or not opportunity_modes:
        raise ValueError("cascade expected sizing/opportunity menu is empty")
    if set(bets["sizing"].dropna().astype(str)) - set(sizings):
        raise ValueError("cascade bets contain an undeclared sizing")
    if set(bets["opportunity_mode"].dropna().astype(str)) - set(opportunity_modes):
        raise ValueError("cascade bets contain an undeclared opportunity mode")
    pieces: list[pd.DataFrame] = []
    for pair in pairing.loc[pairing["quote_menu_status"].eq("exact")].to_dict(
        orient="records"
    ):
        w2 = str(pair["w2_model"])
        w0 = str(pair["w0_model"])
        for opportunity_mode in opportunity_modes:
            for price_point in expected_price_points:
                for line_policy in expected_line_policies:
                    for sizing in sizings:
                        slice_values = {
                            "opportunity_mode": opportunity_mode,
                            "price_point": price_point,
                            "line_policy": line_policy,
                            "sizing": sizing,
                        }
                        common_quote_keys, common_counts = (
                            _cascade_three_arm_common_quote_keys(
                                quote_keys,
                                w2_model=w2,
                                w0_model=w0,
                                price_point=price_point,
                                line_policy=line_policy,
                            )
                        )
                        model = _cascade_event_portfolio(
                            bets,
                            common_quote_keys,
                            model=w2,
                            slice_values=slice_values,
                        )
                        reference = _cascade_event_portfolio(
                            bets,
                            common_quote_keys,
                            model=w0,
                            slice_values=slice_values,
                        )
                        control = _cascade_event_portfolio(
                            bets,
                            common_quote_keys,
                            model=CASCADE_CONTROL,
                            slice_values=slice_values,
                        )
                        keys = ["event_key", "event_start_date", "round_num"]
                        merged = model.merge(
                            reference,
                            on=keys,
                            how="outer",
                            validate="one_to_one",
                            suffixes=("_model", "_reference"),
                            indicator=True,
                        )
                        if not merged["_merge"].eq("both").all():
                            raise ValueError("W2/W0 eligible event universes differ")
                        merged = merged.drop(columns="_merge").merge(
                            control,
                            on=keys,
                            how="outer",
                            validate="one_to_one",
                            indicator=True,
                        )
                        if not merged["_merge"].eq("both").all():
                            raise ValueError("W2/W0/C0 eligible event universes differ")
                        output = merged[keys].copy()
                        output["analysis_block"] = (
                            "cascade_w2_completed_round_weather_normalization"
                        )
                        output["analysis_pair"] = f"{w2}__vs__{w0}"
                        output["model"] = w2
                        output["reference"] = w0
                        output["control"] = CASCADE_CONTROL
                        output["model_bets"] = merged["bets_model"].astype(int)
                        output["reference_bets"] = merged["bets_reference"].astype(int)
                        output["control_bets"] = merged["bets"].astype(int)
                        output["model_stake"] = merged["stake_model"]
                        output["reference_stake"] = merged["stake_reference"]
                        output["control_stake"] = merged["stake"]
                        output["model_pnl"] = merged["pnl_model"]
                        output["reference_pnl"] = merged["pnl_reference"]
                        output["control_pnl"] = merged["pnl"]
                        output["pnl_delta_vs_reference"] = (
                            output["model_pnl"] - output["reference_pnl"]
                        )
                        output["pnl_delta_vs_control"] = (
                            output["model_pnl"] - output["control_pnl"]
                        )
                        output["reference_pnl_delta_vs_control"] = (
                            output["reference_pnl"] - output["control_pnl"]
                        )
                        for column, value in slice_values.items():
                            output[column] = value
                        output["round_market"] = "r" + output["round_num"].astype(str)
                        output["weather_information_scope"] = np.where(
                            output["round_num"].eq(1),
                            "historical_training_only_no_current_completed_round",
                            "historical_training_plus_completed_rounds",
                        )
                        output["money_quote_menu_scope"] = (
                            "w2_w0_c0_pairwise_common"
                        )
                        for column, value in common_counts.items():
                            output[column] = value
                        pieces.append(output)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _standardize_bet_attribution(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    output = frame.copy()
    if source == "sealed":
        _require(
            output,
            ["current_withdrawal_state", "prior_withdrawal_state"],
            "sealed bet withdrawal attribution",
        )
        aliases = {
            "selection_history_rounds": "selected_history_rounds",
            "selection_img_history_rounds": "selected_img_history_rounds",
            "selection_img_history_coverage": "selected_img_history_coverage",
            "selection_reliability_ok": "selected_reliability_ok",
            "selection_cross_tour_fallback_reason": "selected_cross_tour_fallback_reason",
        }
        for source_column, target in aliases.items():
            if source_column in output and target not in output:
                output[target] = output[source_column]
        current_states = {
            "neither",
            "selection_withdrew",
            "opponent_withdrew",
            "both_withdrew",
        }
        prior_states = {
            "neither",
            "selection_prior_wd",
            "opponent_prior_wd",
            "both_prior_wd",
        }
        if "current_withdrawal_state" in output:
            state = output["current_withdrawal_state"]
            if state.isna().any() or not state.map(
                lambda value: isinstance(value, str) and value in current_states
            ).all():
                raise ValueError(
                    "sealed current_withdrawal_state has a malformed withdrawal value"
                )
            if {"selection_outcome_text", "opponent_outcome_text"}.issubset(
                output.columns
            ):
                selection_wd = output["selection_outcome_text"].fillna("").astype(
                    str
                ).str.casefold().str.contains(
                    r"\b(?:wd|withdraw|withdrawn)\b", regex=True
                )
                opponent_wd = output["opponent_outcome_text"].fillna("").astype(
                    str
                ).str.casefold().str.contains(
                    r"\b(?:wd|withdraw|withdrawn)\b", regex=True
                )
                reproduced = pd.Series(
                    np.select(
                        [
                            selection_wd & opponent_wd,
                            selection_wd,
                            opponent_wd,
                        ],
                        [
                            "both_withdrew",
                            "selection_withdrew",
                            "opponent_withdrew",
                        ],
                        default="neither",
                    ),
                    index=output.index,
                )
                if not state.eq(reproduced).all():
                    raise ValueError(
                        "sealed current withdrawal state differs from outcome text"
                    )
            output["target_withdrawal_any"] = ~state.eq("neither")
        if "prior_withdrawal_state" in output:
            state = output["prior_withdrawal_state"]
            if state.isna().any() or not state.map(
                lambda value: isinstance(value, str) and value in prior_states
            ).all():
                raise ValueError(
                    "sealed prior_withdrawal_state has a malformed withdrawal value"
                )
            raw_prior_columns = {
                "selection_prior_event_withdrawal",
                "opponent_prior_event_withdrawal",
            }
            if raw_prior_columns.issubset(output.columns):
                selection_prior = _require_nullable_exact_boolean_series(
                    output["selection_prior_event_withdrawal"],
                    "sealed selection_prior_event_withdrawal",
                ).fillna(False)
                opponent_prior = _require_nullable_exact_boolean_series(
                    output["opponent_prior_event_withdrawal"],
                    "sealed opponent_prior_event_withdrawal",
                ).fillna(False)
                reproduced = pd.Series(
                    np.select(
                        [
                            selection_prior & opponent_prior,
                            selection_prior,
                            opponent_prior,
                        ],
                        [
                            "both_prior_wd",
                            "selection_prior_wd",
                            "opponent_prior_wd",
                        ],
                        default="neither",
                    ),
                    index=output.index,
                )
                if not state.eq(reproduced).all():
                    raise ValueError(
                        "sealed prior withdrawal state differs from source flags"
                    )
            output["prior_withdrawal_any"] = ~state.eq("neither")
    elif source == "cascade":
        _require(
            output,
            [
                "selected_event_withdrawal",
                "opponent_event_withdrawal",
                "selected_prior_event_withdrawal",
                "opponent_prior_event_withdrawal",
                "selected_reliability_ok",
                "opponent_reliability_ok",
            ],
            "cascade bet withdrawal attribution",
        )
        repair_columns: list[str] = []
        for column in sorted(CASCADE_LEGACY_NULLABLE_BINARY_COLUMNS):
            decoded, repaired = _decode_cascade_legacy_nullable_binary(
                output[column], column=column
            )
            output[column] = decoded
            if repaired:
                repair_columns.append(column)
        if repair_columns and set(repair_columns) != set(
            CASCADE_LEGACY_NULLABLE_BINARY_COLUMNS
        ):
            raise ValueError(
                "cascade legacy nullable-binary serialization is only partially present"
            )
        output["legacy_nullable_binary_serialization_repair_bet"] = bool(
            repair_columns
        )
        selected_target = _require_nullable_exact_boolean_series(
            output["selected_event_withdrawal"],
            "cascade selected_event_withdrawal",
        ).fillna(False)
        opponent_target = _require_nullable_exact_boolean_series(
            output["opponent_event_withdrawal"],
            "cascade opponent_event_withdrawal",
        ).fillna(False)
        output["target_withdrawal_any"] = selected_target | opponent_target
        selected_prior = _require_nullable_exact_boolean_series(
            output["selected_prior_event_withdrawal"],
            "cascade selected_prior_event_withdrawal",
        ).fillna(False)
        opponent_prior = _require_nullable_exact_boolean_series(
            output["opponent_prior_event_withdrawal"],
            "cascade opponent_prior_event_withdrawal",
        ).fillna(False)
        output["prior_withdrawal_any"] = selected_prior | opponent_prior
        output["missing_prior_withdrawal_bet"] = (
            output["selected_prior_event_withdrawal"].isna()
            | output["opponent_prior_event_withdrawal"].isna()
        )
    else:
        raise ValueError(source)
    required = [
        "event_key",
        "event_start_date",
        "pnl",
        "stake",
        "selected_history_rounds",
        "opponent_history_rounds",
        "selected_img_history_rounds",
        "opponent_img_history_rounds",
        "selected_img_history_coverage",
        "opponent_img_history_coverage",
        "selected_reliability_ok",
        "opponent_reliability_ok",
        "selected_cross_tour_fallback_reason",
        "target_withdrawal_any",
        "prior_withdrawal_any",
    ]
    _require(output, required, f"{source} bet attribution")
    for column in (
        "selected_history_rounds",
        "opponent_history_rounds",
        "selected_img_history_rounds",
        "opponent_img_history_rounds",
    ):
        output[column] = _require_nullable_nonnegative_integer_series(
            output[column], f"{source} bet attribution {column}"
        )
    for column in (
        "selected_img_history_coverage",
        "opponent_img_history_coverage",
    ):
        output[column] = _require_nullable_unit_interval_series(
            output[column], f"{source} bet attribution {column}"
        )
    _numeric(output, ["pnl", "stake"])
    _require_finite_numeric_columns(
        output,
        ["pnl", "stake"],
        label=f"{source} bet attribution",
        nonnegative=["stake"],
    )
    output["newcomer_bet"] = (
        output["selected_history_rounds"].lt(5)
        | output["opponent_history_rounds"].lt(5)
    )
    output["missing_history_bet"] = (
        output["selected_history_rounds"].isna()
        | output["opponent_history_rounds"].isna()
    )
    selected_reliable = _require_nullable_exact_boolean_series(
        output["selected_reliability_ok"],
        f"{source} bet attribution selected_reliability_ok",
    )
    opponent_reliable = _require_nullable_exact_boolean_series(
        output["opponent_reliability_ok"],
        f"{source} bet attribution opponent_reliability_ok",
    )
    output["target_withdrawal_any"] = _require_exact_boolean_series(
        output["target_withdrawal_any"],
        f"{source} bet attribution target_withdrawal_any",
    )
    output["prior_withdrawal_any"] = _require_exact_boolean_series(
        output["prior_withdrawal_any"],
        f"{source} bet attribution prior_withdrawal_any",
    )
    output["low_coverage_bet"] = (
        selected_reliable.eq(False)
        | opponent_reliable.eq(False)
        | output["missing_history_bet"]
    )
    output["missing_reliability_bet"] = (
        selected_reliable.isna() | opponent_reliable.isna()
    )
    output["low_img_coverage_bet"] = (
        output["selected_img_history_coverage"].lt(0.25)
        | output["opponent_img_history_coverage"].lt(0.25)
    )
    output["missing_img_coverage_bet"] = (
        output["selected_img_history_coverage"].isna()
        | output["opponent_img_history_coverage"].isna()
    )
    fallback = output["selected_cross_tour_fallback_reason"].fillna("").astype(str)
    output["cross_tour_fallback_bet"] = ~fallback.str.casefold().isin(
        {"", "none", "weekly_pga_core", "missing"}
    )
    return output


def _attribute_event_bets(
    bets: pd.DataFrame,
    *,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    if bets.empty:
        return pd.DataFrame(columns=[*group_columns, "attribution_bets"])
    rows: list[dict[str, Any]] = []
    flag_columns = [
        "target_withdrawal_any",
        "prior_withdrawal_any",
        "newcomer_bet",
        "missing_history_bet",
        "low_coverage_bet",
        "missing_reliability_bet",
        "missing_prior_withdrawal_bet",
        "legacy_nullable_binary_serialization_repair_bet",
        "low_img_coverage_bet",
        "missing_img_coverage_bet",
        "cross_tour_fallback_bet",
    ]
    source_specific_optional_flags = {
        "missing_prior_withdrawal_bet",
        "legacy_nullable_binary_serialization_repair_bet",
    }
    for keys, group in bets.groupby(list(group_columns), observed=True, dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        row: dict[str, Any] = dict(zip(group_columns, key_tuple))
        row.update(
            {
                "attribution_bets": int(len(group)),
                "attribution_stake": float(group["stake"].sum()),
                "attribution_pnl": float(group["pnl"].sum()),
            }
        )
        for flag in flag_columns:
            if flag in group:
                mask = group[flag].fillna(False).astype(bool)
            elif flag in source_specific_optional_flags:
                # These cascade-only diagnostics are structurally absent from
                # the sealed base ledger and therefore exactly false there.
                mask = pd.Series(False, index=group.index)
            else:
                raise ValueError(f"bet attribution is missing required flag: {flag}")
            label = flag.removesuffix("_bet").removesuffix("_any")
            row[f"{label}_bets"] = int(mask.sum())
            row[f"{label}_pnl"] = float(group.loc[mask, "pnl"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def load_sealed_pairwise_bets(artifacts: Mapping[str, Path]) -> pd.DataFrame:
    paths = sorted(
        path
        for name, path in artifacts.items()
        if name.startswith("bet_partitions/pairwise_common__")
        and name.endswith(".parquet")
    )
    if not paths:
        raise ValueError("sealed output declares no pairwise-common bet partitions")
    pieces: list[pd.DataFrame] = []
    for path in paths:
        piece = pd.read_parquet(path)
        if piece.empty:
            continue
        piece["source_partition"] = path.name
        pieces.append(piece)
    if not pieces:
        raise ValueError("all sealed pairwise-common bet partitions are empty")
    bets = pd.concat(pieces, ignore_index=True, sort=False)
    required = [
        "candidate",
        "comparison_candidate",
        "opportunity_mode",
        "event_key",
        "event_start_date",
        "round_market",
        *SLICE_COLUMNS[1:],
        "stake",
        "pnl",
    ]
    _require(bets, required, "sealed pairwise bets")
    if not bets["opportunity_mode"].eq(
        "pairwise_common_with_strict_pt_baseline"
    ).all():
        raise ValueError("sealed pairwise partition contains another opportunity mode")
    if "market" in bets:
        bets["settlement_market"] = bets["market"].astype(str)
    bets["market"] = bets["round_market"].astype(str)
    bets["event_start_date"] = pd.to_datetime(
        bets["event_start_date"], errors="coerce"
    ).dt.normalize()
    return _standardize_bet_attribution(bets, source="sealed")


def attach_base_bet_attribution(
    money: pd.DataFrame, bets: pd.DataFrame
) -> pd.DataFrame:
    group_columns = [
        "candidate",
        "comparison_candidate",
        "event_key",
        "event_start_date",
        "market",
        *SLICE_COLUMNS,
    ]
    attrs = _attribute_event_bets(bets, group_columns=group_columns)
    key_columns = ["event_key", "event_start_date", "market", *SLICE_COLUMNS]
    pieces: list[pd.DataFrame] = []
    attribute_columns = [
        column for column in attrs.columns if column not in group_columns
    ]
    for pair, group in money.groupby("analysis_pair", observed=True):
        model_name = str(group["model"].iloc[0])
        reference_name = str(group["reference"].iloc[0])
        control_name = str(group["control"].iloc[0])
        model_attrs = attrs.loc[
            attrs["candidate"].eq(model_name)
            & attrs["comparison_candidate"].eq(model_name)
        ][[*key_columns, *attribute_columns]].copy()
        # Each pairwise-common partition uses its own challenger as
        # comparison_candidate.  Reproduce all three portfolio legs rather
        # than trusting the producer's event-level W0 reference total.
        reference_attrs = attrs.loc[
            attrs["candidate"].eq(reference_name)
            & attrs["comparison_candidate"].eq(reference_name)
        ][[*key_columns, *attribute_columns]].copy()
        control_attrs = attrs.loc[
            attrs["candidate"].eq(control_name)
            & attrs["comparison_candidate"].eq(model_name)
        ][[*key_columns, *attribute_columns]].copy()
        model_attrs = model_attrs.rename(
            columns={column: f"model_{column}" for column in attribute_columns}
        )
        reference_attrs = reference_attrs.rename(
            columns={column: f"reference_{column}" for column in attribute_columns}
        )
        control_attrs = control_attrs.rename(
            columns={column: f"control_{column}" for column in attribute_columns}
        )
        output = group.merge(
            model_attrs, on=key_columns, how="left", validate="one_to_one"
        ).merge(
            reference_attrs, on=key_columns, how="left", validate="one_to_one"
        ).merge(control_attrs, on=key_columns, how="left", validate="one_to_one")
        for prefix in ("model", "reference", "control"):
            for column in attribute_columns:
                target = f"{prefix}_{column}"
                if target not in output:
                    output[target] = 0.0
                output[target] = output[target].fillna(0.0)
        if (output["model_attribution_pnl"] - output["model_pnl"]).abs().max() > 1e-10:
            raise ValueError(f"sealed model bet attribution does not reproduce P&L: {pair}")
        if (
            output["reference_attribution_pnl"] - output["reference_pnl"]
        ).abs().max() > 1e-10:
            raise ValueError(
                f"sealed reference bet attribution does not reproduce P&L: {pair}"
            )
        if (output["control_attribution_pnl"] - output["control_pnl"]).abs().max() > 1e-10:
            raise ValueError(f"sealed control bet attribution does not reproduce P&L: {pair}")
        pieces.append(output)
    return pd.concat(pieces, ignore_index=True) if pieces else money


def attach_cascade_bet_attribution(
    money: pd.DataFrame,
    bets: pd.DataFrame,
    quote_keys: pd.DataFrame,
) -> pd.DataFrame:
    normalized = _standardize_bet_attribution(bets, source="cascade")
    normalized["event_start_date"] = pd.to_datetime(
        normalized["event_start_date"], errors="coerce"
    ).dt.normalize()
    group_columns = [
        "model",
        "event_key",
        "event_start_date",
        "round_num",
        *CASCADE_SLICE_COLUMNS,
    ]
    key_columns = ["event_key", "event_start_date", "round_num", *CASCADE_SLICE_COLUMNS]
    pieces: list[pd.DataFrame] = []
    for pair, group in money.groupby("analysis_pair", observed=True):
        model_name = str(group["model"].iloc[0])
        reference_name = str(group["reference"].iloc[0])
        control_name = str(group["control"].iloc[0])
        if (
            group["model"].astype(str).nunique() != 1
            or group["reference"].astype(str).nunique() != 1
            or group["control"].astype(str).nunique() != 1
            or control_name != CASCADE_CONTROL
        ):
            raise ValueError(f"cascade attribution pair identity is ambiguous: {pair}")
        allowed_parts: list[pd.DataFrame] = []
        for menu_slice in group[["price_point", "line_policy"]].drop_duplicates().to_dict(
            orient="records"
        ):
            common_quote_keys, _ = _cascade_three_arm_common_quote_keys(
                quote_keys,
                w2_model=model_name,
                w0_model=reference_name,
                price_point=str(menu_slice["price_point"]),
                line_policy=str(menu_slice["line_policy"]),
            )
            allowed_parts.append(common_quote_keys)
        if not allowed_parts:
            raise ValueError(f"cascade attribution has no common quote menu: {pair}")
        allowed = pd.concat(allowed_parts, ignore_index=True).drop_duplicates()
        allowed_keys = [
            "model",
            *CASCADE_QUOTE_MENU_KEYS,
            "price_point",
            "line_policy",
        ]
        allowed = allowed[allowed_keys].drop_duplicates()
        if allowed.duplicated(allowed_keys).any():
            raise ValueError(f"cascade attribution common menu is ambiguous: {pair}")
        pair_bets = normalized.merge(
            allowed,
            on=allowed_keys,
            how="inner",
            validate="many_to_one",
        )
        attrs = _attribute_event_bets(pair_bets, group_columns=group_columns)
        attribute_columns = [
            column for column in attrs.columns if column not in group_columns
        ]
        model_attrs = attrs.loc[attrs["model"].eq(model_name)][
            [*key_columns, *attribute_columns]
        ].rename(columns={column: f"model_{column}" for column in attribute_columns})
        reference_attrs = attrs.loc[attrs["model"].eq(reference_name)][
            [*key_columns, *attribute_columns]
        ].rename(
            columns={column: f"reference_{column}" for column in attribute_columns}
        )
        control_attrs = attrs.loc[attrs["model"].eq(control_name)][
            [*key_columns, *attribute_columns]
        ].rename(columns={column: f"control_{column}" for column in attribute_columns})
        output = group.merge(
            model_attrs, on=key_columns, how="left", validate="one_to_one"
        ).merge(
            reference_attrs, on=key_columns, how="left", validate="one_to_one"
        ).merge(control_attrs, on=key_columns, how="left", validate="one_to_one")
        for prefix in ("model", "reference", "control"):
            for column in attribute_columns:
                target = f"{prefix}_{column}"
                output[target] = output.get(target, 0.0)
                if isinstance(output[target], pd.Series):
                    output[target] = output[target].fillna(0.0)
        if (output["model_attribution_pnl"] - output["model_pnl"]).abs().max() > 1e-10:
            raise ValueError(f"cascade model attribution does not reproduce P&L: {pair}")
        if (
            output["reference_attribution_pnl"] - output["reference_pnl"]
        ).abs().max() > 1e-10:
            raise ValueError(
                f"cascade reference attribution does not reproduce P&L: {pair}"
            )
        if (output["control_attribution_pnl"] - output["control_pnl"]).abs().max() > 1e-10:
            raise ValueError(f"cascade control attribution does not reproduce P&L: {pair}")
        pieces.append(output)
    return pd.concat(pieces, ignore_index=True) if pieces else money


def attach_weather_to_base(
    frame: pd.DataFrame, event_weather: pd.DataFrame
) -> pd.DataFrame:
    columns = [
        "event_key",
        "event_start_date",
        "event_rounds",
        "event_weather_coverage_rate",
        "event_weather_severity_mean",
        "event_weather_severity_max",
        "event_weather_severity_bucket",
        "event_max_abs_wave_wind_gap",
        "event_max_abs_course_wave_wind_gap",
        "event_max_wind_spread",
        "event_withdrawal_players",
        "event_max_physical_course_count",
        "event_multi_course",
        "course_join_scope",
        "target_event_realized_weather_timing",
        "target_event_realized_weather_used_as_feature",
    ]
    output = frame.merge(
        event_weather[columns],
        on=["event_key", "event_start_date"],
        how="left",
        validate="many_to_one",
    )
    if output["event_rounds"].isna().any():
        raise ValueError("base event attribution lacks a course-stratified weather profile")
    output["target_event_weather_missing"] = output[
        "event_weather_severity_max"
    ].isna()
    output["analysis_weather_severity"] = output["event_weather_severity_max"]
    output["analysis_weather_severity_bucket"] = output[
        "event_weather_severity_bucket"
    ]
    output["analysis_tee_wave_wind_gap"] = output[
        "event_max_abs_course_wave_wind_gap"
    ]
    output["analysis_weather_missing"] = output["target_event_weather_missing"]
    output["analysis_weather_coverage_rate"] = output[
        "event_weather_coverage_rate"
    ]
    output["year"] = pd.to_datetime(output["event_start_date"]).dt.year.astype(int)
    return output


def attach_weather_to_cascade(
    frame: pd.DataFrame,
    weather_rounds: pd.DataFrame,
    c3_weather_lineage_by_event_round: pd.DataFrame,
) -> pd.DataFrame:
    _require(frame, ["model", "analysis_pair"], "cascade weather attribution")
    pair_parts = frame["analysis_pair"].astype(str).str.split(
        "__vs__", n=1, expand=True
    )
    if pair_parts.shape[1] != 2 or pair_parts.isna().any().any():
        raise ValueError("cascade analysis-pair identity is invalid")
    w2 = pair_parts[0]
    w0 = pair_parts[1]
    if not frame["model"].astype(str).eq(w2).all():
        raise ValueError("cascade weather-lineage model differs from its W2 pair arm")
    if "challenger" in frame.columns or "comparison_kind" in frame.columns:
        _require(
            frame,
            [
                "model",
                "challenger",
                "reference",
                "analysis_pair",
                "comparison_kind",
                "weather_lineage_model_role",
            ],
            "cascade proper-score weather attribution",
        )
        if (
            not frame["weather_lineage_model_role"].astype(str).eq(
                "w2_arm_for_analysis_pair_not_score_challenger"
            ).all()
        ):
            raise ValueError("cascade proper-score weather-lineage model is false")
        semantics = {
            "w2_vs_matched_w0": (
                frame["challenger"].astype(str).eq(w2)
                & frame["reference"].astype(str).eq(w0)
            ),
            "w2_vs_current_cascade_control": (
                frame["challenger"].astype(str).eq(w2)
                & frame["reference"].astype(str).eq(CASCADE_CONTROL)
            ),
            "w0_vs_current_cascade_control": (
                frame["challenger"].astype(str).eq(w0)
                & frame["reference"].astype(str).eq(CASCADE_CONTROL)
            ),
        }
        valid_semantics = pd.Series(False, index=frame.index)
        for kind, valid in semantics.items():
            valid_semantics |= frame["comparison_kind"].astype(str).eq(kind) & valid
        if not valid_semantics.all():
            raise ValueError("cascade proper-score comparison semantics are false")
    _require(
        frame,
        ["event_key", "event_start_date", "round_num"],
        "cascade weather attribution",
    )
    frame = frame.copy()
    frame["event_start_date"] = pd.to_datetime(
        frame["event_start_date"], errors="coerce"
    ).dt.normalize()
    numeric_round = pd.to_numeric(frame["round_num"], errors="coerce")
    if (
        frame["event_start_date"].isna().any()
        or not np.isfinite(numeric_round.to_numpy(float)).all()
        or not numeric_round.isin([1, 2, 3, 4]).all()
        or not numeric_round.eq(np.floor(numeric_round)).all()
    ):
        raise ValueError("cascade weather attribution event/round identity is invalid")
    frame["round_num"] = numeric_round.astype(int)
    target_columns = [
        "event_key",
        "event_start_date",
        "year",
        "round_num",
        "physical_course_count",
        "physical_courses",
        "multi_course_event_round",
        "course_aggregation_method",
        "weather_coverage_rate",
        "weather_severity_percentile",
        "weather_severity_bucket",
        "late_minus_early_wind",
        "max_abs_course_late_minus_early_wind",
        "late_minus_early_dew",
        "late_minus_early_field_adjusted_sg",
        "wind_delta_p90_p10",
        "event_withdrawal_players",
        "target_round_realized_weather_timing",
        "target_round_realized_weather_used_as_feature",
    ]
    target = weather_rounds[target_columns].rename(
        columns={
            column: f"target_oracle_{column}"
            for column in target_columns
            if column not in {"event_key", "event_start_date", "round_num"}
        }
    )
    completed_rows: list[dict[str, Any]] = []
    for (event_key, event_start_date, year), group in weather_rounds.groupby(
        ["event_key", "event_start_date", "year"], observed=True
    ):
        for round_num in range(1, 5):
            prior = group.loc[group["round_num"].lt(round_num)]
            valid_weather = prior["weather_coverage_rate"].gt(0) & prior[
                "weather_severity_percentile"
            ].notna()
            completed_rows.append(
                {
                    "event_key": event_key,
                    "event_start_date": event_start_date,
                    "year": year,
                    "round_num": round_num,
                    "completed_rounds_in_profile": int(len(prior)),
                    "completed_rounds_with_valid_wind_dew_weather": int(
                        valid_weather.sum()
                    ),
                    "completed_valid_wind_dew_player_rounds": int(
                        prior.loc[valid_weather, "weather_observed_rows"].sum()
                    ),
                    "completed_weather_coverage_rate": (
                        float(prior["weather_coverage_rate"].mean()) if len(prior) else np.nan
                    ),
                    "completed_weather_severity_mean": (
                        float(prior["weather_severity_percentile"].mean()) if len(prior) else np.nan
                    ),
                    "completed_weather_severity_max": (
                        float(prior["weather_severity_percentile"].max()) if len(prior) else np.nan
                    ),
                    "completed_max_abs_wave_wind_gap": (
                        float(prior["late_minus_early_wind"].abs().max()) if len(prior) else np.nan
                    ),
                }
            )
    completed = pd.DataFrame(completed_rows)
    lineage_keys = ["model", "event_key", "event_start_date", "round_num"]
    _require(
        c3_weather_lineage_by_event_round,
        [
            *lineage_keys,
            "lineage_completed_weather_observations",
            "lineage_players_with_completed_weather",
            "lineage_completed_weather_any",
        ],
        "cascade C3 event-round weather lineage",
    )
    lineage = c3_weather_lineage_by_event_round.copy()
    lineage["event_start_date"] = pd.to_datetime(
        lineage["event_start_date"], errors="coerce"
    ).dt.normalize()
    if lineage[lineage_keys].isna().any().any() or lineage.duplicated(
        lineage_keys
    ).any():
        raise ValueError("cascade C3 event-round weather lineage keys are invalid")
    lineage_any = _require_exact_boolean_series(
        lineage["lineage_completed_weather_any"],
        "cascade lineage_completed_weather_any",
    )
    lineage_observations = pd.to_numeric(
        lineage["lineage_completed_weather_observations"], errors="coerce"
    )
    lineage_players = pd.to_numeric(
        lineage["lineage_players_with_completed_weather"], errors="coerce"
    )
    if (
        not np.isfinite(lineage_observations.to_numpy(float)).all()
        or not np.isfinite(lineage_players.to_numpy(float)).all()
        or lineage_observations.lt(0).any()
        or lineage_players.lt(0).any()
        or not lineage_any.eq(lineage_observations.gt(0)).all()
    ):
        raise ValueError("cascade C3 event-round completed-weather counts are invalid")
    lineage["lineage_completed_weather_any"] = lineage_any
    output_keys = frame[lineage_keys].drop_duplicates()
    lineage_menu = output_keys.merge(
        lineage[lineage_keys],
        on=lineage_keys,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not lineage_menu["_merge"].eq("both").all():
        raise ValueError(
            "cascade W2 analysis row lacks completed-weather lineage"
        )
    output = frame.merge(
        target,
        on=["event_key", "event_start_date", "round_num"],
        how="left",
        validate="many_to_one",
    ).merge(
        completed,
        on=["event_key", "event_start_date", "round_num"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_completed_profile"),
    ).merge(
        lineage[
            [
                *lineage_keys,
                "lineage_completed_weather_observations",
                "lineage_players_with_completed_weather",
                "lineage_completed_weather_any",
            ]
        ],
        on=lineage_keys,
        how="left",
        validate="many_to_one",
    )
    if output["target_oracle_physical_course_count"].isna().any():
        raise ValueError(
            "cascade event/round attribution lacks a physical-course weather profile"
        )
    if output.loc[output["round_num"].eq(1), "completed_rounds_in_profile"].ne(0).any():
        raise AssertionError("R1 cannot have a completed current-event weather round")
    if output.loc[
        output["round_num"].eq(1), "lineage_completed_weather_any"
    ].any():
        raise ValueError("cascade W2 lineage claims completed weather at R1")
    output["completed_weather_profile_valid"] = (
        output["completed_rounds_with_valid_wind_dew_weather"].gt(0)
        & output["completed_valid_wind_dew_player_rounds"].gt(0)
        & output["completed_weather_coverage_rate"].gt(0)
        & output["completed_weather_severity_mean"].notna()
        & output["completed_weather_severity_max"].notna()
    )
    invalid_claim = output["lineage_completed_weather_any"] & ~output[
        "completed_weather_profile_valid"
    ]
    if invalid_claim.any():
        raise ValueError(
            "cascade W2 lineage claims completed weather without nonzero prior "
            "wind+dew coverage and severity"
        )
    output["completed_weather_used_by_w2"] = (
        output["lineage_completed_weather_any"]
        & output["completed_weather_profile_valid"]
    )
    output["target_oracle_weather_used_by_w2"] = False
    output["target_round_weather_missing"] = output[
        "target_oracle_weather_severity_percentile"
    ].isna()
    output["analysis_weather_severity"] = output[
        "target_oracle_weather_severity_percentile"
    ]
    output["analysis_weather_severity_bucket"] = output[
        "target_oracle_weather_severity_bucket"
    ]
    output["analysis_tee_wave_wind_gap"] = output[
        "target_oracle_max_abs_course_late_minus_early_wind"
    ]
    output["analysis_weather_missing"] = output["target_round_weather_missing"]
    output["analysis_weather_coverage_rate"] = output[
        "target_oracle_weather_coverage_rate"
    ]
    output["course_join_scope"] = (
        "event_round_portfolio_physical_course_stratified_summary_no_pair_course_identity"
    )
    output["year"] = pd.to_datetime(output["event_start_date"]).dt.year.astype(int)
    if output["target_oracle_year"].notna().any() and output.loc[
        output["target_oracle_year"].notna(), "target_oracle_year"
    ].astype(int).ne(
        output.loc[output["target_oracle_year"].notna(), "year"]
    ).any():
        raise ValueError("cascade target weather year differs from event date")
    return output


def mark_bad_weeks(frame: pd.DataFrame) -> pd.DataFrame:
    """Mark fixed bottom-decile negative weeks within each frozen money slice."""

    output = frame.copy()
    slice_columns = [
        "analysis_block",
        "analysis_pair",
        *[
            column
            for column in (*SLICE_COLUMNS, *CASCADE_SLICE_COLUMNS, "market", "round_market")
            if column in output
        ],
    ]
    slice_columns = list(dict.fromkeys(slice_columns))
    for target in ("control_pnl", "reference_pnl", "model_pnl"):
        threshold = output.groupby(slice_columns, observed=True, dropna=False)[
            target
        ].transform(lambda values: values.quantile(0.10))
        prefix = target.removesuffix("_pnl")
        output[f"{prefix}_bad_week_threshold"] = threshold
        output[f"is_{prefix}_bad_week"] = output[target].lt(0.0) & output[
            target
        ].le(threshold)
    output["model_loss_recovery_vs_control"] = np.where(
        output["control_pnl"].lt(0.0),
        output["pnl_delta_vs_control"] / output["control_pnl"].abs(),
        np.nan,
    )
    output["normalization_loss_recovery_vs_reference"] = np.where(
        output["reference_pnl"].lt(0.0),
        output["pnl_delta_vs_reference"] / output["reference_pnl"].abs(),
        np.nan,
    )
    return output


def annotate_money_execution_eligibility(
    frame: pd.DataFrame,
    *,
    price_column: str,
    model_state_timestamp_column: str | None = None,
    quote_timestamp_column: str | None = None,
    timestamp_authority_verified: bool = False,
) -> pd.DataFrame:
    """Keep open prices descriptive unless exact timestamp order is proven."""

    _require(frame, [price_column], "money execution eligibility")
    output = frame.copy()
    price = output[price_column].astype(str).str.strip().str.casefold()
    if not price.isin(EXPECTED_CASCADE_PRICE_POINTS).all():
        raise ValueError("money execution eligibility has an unknown price point")
    output["opening_price_sensitivity"] = price.eq("open")
    output["execution_eligible"] = price.eq("close")
    output["execution_eligibility_reason"] = np.where(
        price.eq("close"),
        "closing_price_primary_evaluation",
        "open_quote_and_model_state_timestamp_order_not_hash_bound",
    )
    if price.eq("open").any() and (
        model_state_timestamp_column is not None
        or quote_timestamp_column is not None
    ):
        if not model_state_timestamp_column or not quote_timestamp_column:
            raise ValueError(
                "open execution proof requires both model-state and quote timestamps"
            )
        _require(
            output,
            [model_state_timestamp_column, quote_timestamp_column],
            "open execution timestamp proof",
        )
        state = pd.to_datetime(
            output[model_state_timestamp_column], errors="coerce", utc=True
        )
        quote = pd.to_datetime(
            output[quote_timestamp_column], errors="coerce", utc=True
        )
        open_rows = price.eq("open")
        valid = open_rows & state.notna() & quote.notna()
        proven = valid & state.le(quote) & bool(timestamp_authority_verified)
        late = valid & state.gt(quote)
        output.loc[proven, "execution_eligible"] = True
        output.loc[proven, "execution_eligibility_reason"] = (
            "hash_bound_model_state_not_later_than_open_quote"
        )
        output.loc[late, "execution_eligibility_reason"] = (
            "model_state_timestamp_after_open_quote"
        )
        output.loc[
            open_rows & ~(valid | proven), "execution_eligibility_reason"
        ] = "open_quote_or_model_state_timestamp_missing"
        if valid.any() and not timestamp_authority_verified:
            output.loc[valid, "execution_eligibility_reason"] = (
                "open_timestamp_order_present_but_not_hash_bound"
            )
    cascade_late_risk = (
        price.eq("open")
        & ~output["execution_eligible"]
        & output.get("analysis_block", pd.Series("", index=output.index))
        .astype(str)
        .eq("cascade_w2_completed_round_weather_normalization")
        & pd.to_numeric(
            output.get("round_num", pd.Series(np.nan, index=output.index)),
            errors="coerce",
        ).gt(1)
    )
    output.loc[cascade_late_risk, "execution_eligibility_reason"] = (
        "completed_round_state_timestamp_not_proven_before_open_quote"
    )
    output["execution_scope"] = np.where(
        output["execution_eligible"],
        "executable_primary_close",
        "nonexecutable_open_sensitivity",
    )
    _require_exact_boolean_series(
        output["execution_eligible"], "money execution_eligible"
    )
    return output


def _money_breakdowns(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fixed = [
        "analysis_block",
        "analysis_pair",
        "model",
        "reference",
        "control",
        *[
            column
            for column in (*SLICE_COLUMNS, *CASCADE_SLICE_COLUMNS, "market")
            if column in frame
        ],
    ]
    fixed = list(dict.fromkeys(fixed))
    dimension_series: dict[str, pd.Series] = {
        "year": pd.to_datetime(frame["event_start_date"]).dt.year.astype(str),
        "round": frame.get("round_market", pd.Series("event", index=frame.index)).astype(str),
        "control_bad_week": frame["is_control_bad_week"].astype(str),
        "model_bad_week": frame["is_model_bad_week"].astype(str),
        "target_withdrawal": frame.get(
            "model_target_withdrawal_bets", pd.Series(0, index=frame.index)
        ).gt(0).astype(str),
        "newcomer": frame.get(
            "model_newcomer_bets", pd.Series(0, index=frame.index)
        ).gt(0).astype(str),
        "low_coverage": frame.get(
            "model_low_coverage_bets", pd.Series(0, index=frame.index)
        ).gt(0).astype(str),
        "weather_severity": frame["analysis_weather_severity_bucket"].astype(str),
        "tee_wave_wind_gap": pd.cut(
            frame["analysis_tee_wave_wind_gap"],
            bins=[-np.inf, 1.0, 2.0, 4.0, np.inf],
            labels=["<1", "1-2", "2-4", ">=4"],
            right=False,
        ).astype(str),
        "weather_missing": frame["analysis_weather_missing"].astype(str),
    }
    for dimension, buckets in dimension_series.items():
        local = frame.assign(_bucket=buckets.fillna("missing").astype(str))
        for keys, group in local.groupby([*fixed, "_bucket"], observed=True, dropna=False):
            key_tuple = keys if isinstance(keys, tuple) else (keys,)
            row = dict(zip([*fixed, "bucket"], key_tuple))
            row.update(
                {
                    "breakdown_dimension": dimension,
                    "events": int(group["event_key"].nunique()),
                    "model_bets": int(group.get("model_attribution_bets", pd.Series(0, index=group.index)).sum()),
                    "model_stake": float(group.get("model_attribution_stake", pd.Series(0.0, index=group.index)).sum()),
                    "model_pnl": float(group["model_pnl"].sum()),
                    "reference_pnl": float(group["reference_pnl"].sum()),
                    "control_pnl": float(group["control_pnl"].sum()),
                    "pnl_delta_vs_reference": float(group["pnl_delta_vs_reference"].sum()),
                    "pnl_delta_vs_control": float(group["pnl_delta_vs_control"].sum()),
                    "mean_normalization_recovery": float(
                        group["normalization_loss_recovery_vs_reference"].mean()
                    ),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _proper_breakdowns(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fixed = [
        "analysis_block",
        "analysis_pair",
        "comparison_kind",
        "challenger",
        "reference",
        "metric",
        "round_market",
    ]
    dimensions: dict[str, pd.Series] = {
        "year": pd.to_datetime(frame["event_start_date"]).dt.year.astype(str),
        "round": frame["round_market"].astype(str),
        "weather_severity": frame["analysis_weather_severity_bucket"].astype(str),
        "tee_wave_wind_gap": pd.cut(
            frame["analysis_tee_wave_wind_gap"],
            bins=[-np.inf, 1.0, 2.0, 4.0, np.inf],
            labels=["<1", "1-2", "2-4", ">=4"],
            right=False,
        ).astype(str),
        "weather_missing": frame["analysis_weather_missing"].astype(str),
    }
    for dimension, buckets in dimensions.items():
        local = frame.assign(_bucket=buckets.fillna("missing").astype(str))
        for keys, group in local.groupby([*fixed, "_bucket"], observed=True, dropna=False):
            key_tuple = keys if isinstance(keys, tuple) else (keys,)
            row = dict(zip([*fixed, "bucket"], key_tuple))
            row.update(
                {
                    "breakdown_dimension": dimension,
                    "events": int(group["event_key"].nunique()),
                    "common_pairs": int(group["common_pairs"].sum()),
                    "mean_event_delta": float(group["event_delta"].mean()),
                    "challenger_mean_score": float(group["challenger_score"].mean()),
                    "reference_mean_score": float(group["reference_score"].mean()),
                    "better_when": "negative",
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _spearman(left: pd.Series, right: pd.Series) -> float:
    valid = pd.to_numeric(left, errors="coerce").notna() & pd.to_numeric(
        right, errors="coerce"
    ).notna()
    if int(valid.sum()) < 3:
        return np.nan
    x = pd.to_numeric(left.loc[valid], errors="coerce")
    y = pd.to_numeric(right.loc[valid], errors="coerce")
    if x.nunique() < 2 or y.nunique() < 2:
        return np.nan
    return float(x.corr(y, method="spearman"))


def build_association_summary(frame: pd.DataFrame) -> pd.DataFrame:
    fixed = [
        "analysis_block",
        "analysis_pair",
        "model",
        "reference",
        "control",
        *[
            column
            for column in (*SLICE_COLUMNS, *CASCADE_SLICE_COLUMNS, "market", "round_market")
            if column in frame
        ],
    ]
    fixed = list(dict.fromkeys(fixed))
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(fixed, observed=True, dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        severity_column = "analysis_weather_severity"
        severe_bucket = group["analysis_weather_severity_bucket"].eq("extreme")
        bad = group["is_control_bad_week"].fillna(False).astype(bool)
        negative_control = group["control_pnl"].lt(0.0)
        total_control_loss = float(-group.loc[negative_control, "control_pnl"].sum())
        extreme_control_loss = float(
            -group.loc[negative_control & severe_bucket, "control_pnl"].sum()
        )
        row = dict(zip(fixed, key_tuple))
        row.update(
            {
                "events": int(group["event_key"].nunique()),
                "control_bad_weeks": int(bad.sum()),
                "bad_weeks_extreme_weather": int((bad & severe_bucket).sum()),
                "bad_weeks_with_target_withdrawal_bet": int(
                    (
                        bad
                        & group.get(
                            "model_target_withdrawal_bets",
                            pd.Series(0, index=group.index),
                        ).gt(0)
                    ).sum()
                ),
                "bad_weeks_with_newcomer_bet": int(
                    (
                        bad
                        & group.get(
                            "model_newcomer_bets", pd.Series(0, index=group.index)
                        ).gt(0)
                    ).sum()
                ),
                "bad_weeks_with_low_coverage_bet": int(
                    (
                        bad
                        & group.get(
                            "model_low_coverage_bets",
                            pd.Series(0, index=group.index),
                        ).gt(0)
                    ).sum()
                ),
                "control_pnl": float(group["control_pnl"].sum()),
                "model_pnl": float(group["model_pnl"].sum()),
                "reference_pnl": float(group["reference_pnl"].sum()),
                "pnl_delta_vs_control": float(group["pnl_delta_vs_control"].sum()),
                "pnl_delta_vs_reference": float(group["pnl_delta_vs_reference"].sum()),
                "bad_week_delta_vs_control": float(
                    group.loc[bad, "pnl_delta_vs_control"].sum()
                ),
                "bad_week_delta_vs_reference": float(
                    group.loc[bad, "pnl_delta_vs_reference"].sum()
                ),
                "extreme_weather_control_loss_share": (
                    extreme_control_loss / total_control_loss
                    if total_control_loss > 0
                    else np.nan
                ),
                "spearman_control_pnl_vs_weather_severity": _spearman(
                    group["control_pnl"], group[severity_column]
                ),
                "spearman_normalization_delta_vs_weather_severity": _spearman(
                    group["pnl_delta_vs_reference"], group[severity_column]
                ),
                "mean_normalization_delta_extreme_weather": float(
                    group.loc[severe_bucket, "pnl_delta_vs_reference"].mean()
                ),
                "mean_normalization_delta_non_extreme_weather": float(
                    group.loc[~severe_bucket, "pnl_delta_vs_reference"].mean()
                ),
                "descriptive_not_causal": True,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _ols_r2(y: np.ndarray, x: np.ndarray) -> float:
    if len(y) < x.shape[1] + 2 or np.nanstd(y) <= EPS:
        return np.nan
    design = np.column_stack([np.ones(len(x)), x])
    coefficient = np.linalg.pinv(design) @ y
    residual = y - design @ coefficient
    total = y - y.mean()
    denominator = float(total @ total)
    return float(1.0 - (residual @ residual) / denominator) if denominator > EPS else np.nan


def build_bad_week_explanation_models(frame: pd.DataFrame) -> pd.DataFrame:
    """Descriptive event-equal OLS; weather incremental R2 is not causal."""

    fixed = [
        "analysis_block",
        "analysis_pair",
        *[
            column
            for column in (*SLICE_COLUMNS, *CASCADE_SLICE_COLUMNS, "market", "round_market")
            if column in frame
        ],
    ]
    fixed = list(dict.fromkeys(fixed))
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(fixed, observed=True, dropna=False):
        severity = pd.to_numeric(
            group["analysis_weather_severity"],
            errors="coerce",
        )
        weather_missing = severity.isna().astype(float)
        wave = pd.to_numeric(
            group["analysis_tee_wave_wind_gap"],
            errors="coerce",
        ).abs()
        wd = group.get(
            "model_target_withdrawal_bets", pd.Series(0, index=group.index)
        ).gt(0).astype(float)
        newcomer = group.get(
            "model_newcomer_bets", pd.Series(0, index=group.index)
        ).gt(0).astype(float)
        coverage = group.get(
            "model_low_coverage_bets", pd.Series(0, index=group.index)
        ).gt(0).astype(float)
        matrix = pd.DataFrame(
            {
                "severity": severity,
                "wave": wave,
                "weather_missing": weather_missing,
                "withdrawal": wd,
                "newcomer": newcomer,
                "low_coverage": coverage,
            },
            index=group.index,
        )
        for column in ("severity", "wave"):
            median = matrix[column].median()
            matrix[column] = matrix[column].fillna(median if pd.notna(median) else 0.0)
            standard = matrix[column].std(ddof=0)
            if pd.notna(standard) and standard > EPS:
                matrix[column] = (matrix[column] - matrix[column].mean()) / standard
            else:
                matrix[column] = 0.0
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        for outcome_column in ("control_pnl", "pnl_delta_vs_reference"):
            y = pd.to_numeric(group[outcome_column], errors="coerce")
            valid = y.notna() & matrix.notna().all(axis=1)
            y_array = y.loc[valid].to_numpy(float)
            x = matrix.loc[valid].to_numpy(float)
            full = _ols_r2(y_array, x)
            without_weather = _ols_r2(y_array, x[:, 3:])
            row = dict(zip(fixed, key_tuple))
            row.update(
                {
                    "outcome": outcome_column,
                    "events": int(valid.sum()),
                    "full_r2": full,
                    "nonweather_r2": without_weather,
                    "weather_incremental_r2": (
                        full - without_weather
                        if np.isfinite(full) and np.isfinite(without_weather)
                        else np.nan
                    ),
                    "features": "weather_severity|tee_wave_wind_gap|weather_missing|withdrawal|newcomer|low_coverage",
                    "event_equal": True,
                    "descriptive_not_causal": True,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap_statistic(values: np.ndarray, metric: str) -> float:
    return float(values.sum()) if metric == "pnl_delta" else float(values.mean())


def bootstrap_event_metrics(
    event_metrics: pd.DataFrame,
    *,
    draws: int,
    seed: int,
    block_sizes: Sequence[int] = (2, 3, 4),
) -> pd.DataFrame:
    """Whole-event iid and circular moving-block uncertainty."""

    if draws < MIN_BOOTSTRAP_DRAWS:
        raise ValueError(f"weather audit requires at least {MIN_BOOTSTRAP_DRAWS} draws")
    required = [
        "analysis_block",
        "analysis_pair",
        "comparison_kind",
        "challenger",
        "reference",
        "metric",
        "higher_is_better",
        "event_key",
        "event_start_date",
        "event_delta",
    ]
    _require(event_metrics, required, "event metrics for bootstrap")
    metadata = [
        column
        for column in event_metrics.columns
        if column
        in {
            *required[:7],
            *SLICE_COLUMNS,
            *CASCADE_SLICE_COLUMNS,
            "market",
            "round_market",
        }
    ]
    rows: list[dict[str, Any]] = []
    for group_index, (keys, group) in enumerate(
        event_metrics.groupby(metadata, observed=True, dropna=False)
    ):
        group = group.sort_values(["event_start_date", "event_key"], kind="mergesort")
        if group["event_key"].duplicated().any():
            raise ValueError("bootstrap hypothesis has duplicate event rows")
        values = pd.to_numeric(group["event_delta"], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or not len(values):
            raise ValueError("bootstrap event deltas must be finite and nonempty")
        metric_name = str(group["metric"].iloc[0])
        aggregate_kind = "pnl_delta" if "pnl" in metric_name else "proper_score_delta"
        estimate = _bootstrap_statistic(values, aggregate_kind)
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(metadata, key_tuple))
        methods = [("whole_event_iid", 1), *[("circular_moving_block", int(size)) for size in block_sizes]]
        for method_index, (method, block_size) in enumerate(methods):
            if block_size < 1:
                raise ValueError("bootstrap block sizes must be positive")
            if block_size > len(values):
                rows.append(
                    {
                        **base,
                        "bootstrap_method": method,
                        "block_size_events": block_size,
                        "events": int(len(values)),
                        "zero_delta_events": int(np.isclose(values, 0.0).sum()),
                        "estimate": estimate,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "probability_better": np.nan,
                        "draws": int(draws),
                        "status": "block_larger_than_event_count",
                    }
                )
                continue
            rng = np.random.default_rng(seed + group_index * 10007 + method_index * 1009)
            samples = np.empty(draws, dtype=float)
            batch = 250
            for start in range(0, draws, batch):
                stop = min(start + batch, draws)
                if method == "whole_event_iid":
                    indices = rng.integers(0, len(values), size=(stop - start, len(values)))
                else:
                    blocks = math.ceil(len(values) / block_size)
                    starts = rng.integers(0, len(values), size=(stop - start, blocks))
                    offsets = np.arange(block_size)
                    indices = (starts[..., None] + offsets) % len(values)
                    indices = indices.reshape(stop - start, -1)[:, : len(values)]
                sampled = values[indices]
                samples[start:stop] = (
                    sampled.sum(axis=1)
                    if aggregate_kind == "pnl_delta"
                    else sampled.mean(axis=1)
                )
            higher = bool(group["higher_is_better"].iloc[0])
            rows.append(
                {
                    **base,
                    "bootstrap_method": method,
                    "block_size_events": block_size,
                    "events": int(len(values)),
                    "zero_delta_events": int(np.isclose(values, 0.0).sum()),
                    "estimate": estimate,
                    "ci_low": float(np.quantile(samples, 0.025)),
                    "ci_high": float(np.quantile(samples, 0.975)),
                    "probability_better": float(
                        np.mean(samples > 0.0) if higher else np.mean(samples < 0.0)
                    ),
                    "draws": int(draws),
                    "status": "complete",
                }
            )
    return pd.DataFrame(rows)


def money_to_bootstrap_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    common = [
        "analysis_block",
        "analysis_pair",
        "event_key",
        "event_start_date",
        *[
            column
            for column in (*SLICE_COLUMNS, *CASCADE_SLICE_COLUMNS, "market", "round_market")
            if column in frame
        ],
    ]
    common = list(dict.fromkeys(common))
    for comparison_kind, challenger_column, reference_column, delta_column in (
        ("normalization_vs_matched_w0", "model", "reference", "pnl_delta_vs_reference"),
        ("normalization_model_vs_control", "model", "control", "pnl_delta_vs_control"),
        (
            "matched_w0_vs_control",
            "reference",
            "control",
            "reference_pnl_delta_vs_control",
        ),
    ):
        use = frame[common].copy()
        use["comparison_kind"] = comparison_kind
        use["challenger"] = frame[challenger_column].astype(str)
        use["reference"] = frame[reference_column].astype(str)
        use["metric"] = "pnl_delta"
        use["higher_is_better"] = True
        use["event_delta"] = frame[delta_column].to_numpy(float)
        rows.append(use)
    return pd.concat(rows, ignore_index=True)


def _report_markdown_table(frame: pd.DataFrame, *, rows: int) -> str:
    """Render a compact table without making tabulate a runtime requirement."""

    preview = frame.head(rows)
    try:
        return preview.to_markdown(index=False, floatfmt=".4f")
    except ImportError:
        # Reporting is not a scoring dependency.  Preserve a readable,
        # deterministic table when pandas' optional ``tabulate`` extra is not
        # installed, while allowing every non-dependency error to fail closed.
        text = preview.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
        return f"```text\n{text}\n```"


def _write_report(
    output: Path,
    audit: Mapping[str, Any],
    associations: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> None:
    lines = [
        "# Base-skill weather and bad-week attribution",
        "",
        "This is a research-only post-sealed audit. It cannot promote or reselect a model.",
        "",
        "## Interpretation contract",
        "",
        "- W1 compares historical-weather-normalized base-skill arms with structurally matched W0 arms on exact frozen quote menus.",
        "- W2 compares strict point-in-time completed-round-weather-normalized cascade arms with matched W0 arms.",
        "- Target-event and target-round realized weather is oracle-only post-hoc attribution and was never an available feature.",
        "- Weather, withdrawal, coverage, and newcomer associations are descriptive, not causal.",
        "- Every bootstrap resamples whole events; moving-block rows preserve adjacent bad-week clustering.",
        "- Money, bad-week, and P&L-bootstrap headlines are close-only. Open-price rows remain labeled non-executable sensitivity unless hash-bound quote and model-state timestamps prove the state existed by the opener.",
        "- Cascade bookless proper scoring includes only independently audited score-model-eligible physical-outcome rows. No-score voids are hash-listed exclusions and are never treated as ties.",
        "- Every selected bet's quote ID, book, side, execution odds, outcome, tie rule, stake, return, and P&L are independently reproduced from the hash-bound normalized quote authority.",
        "- Withdrawal/reliability flags must be exact booleans; history counts must be finite nonnegative integers; IMG coverage must be finite in [0,1]. Missing source attribution stays missing rather than being coerced to zero.",
        "",
        "## Audit",
        "",
        f"- Sealed events with weather profiles: {audit['weather']['events']}",
        f"- Event-round weather profiles: {audit['weather']['event_rounds']}",
        f"- Executable close-only money rows: {audit['executable_primary_close_money_rows']}",
        f"- Non-executable open sensitivity rows: {audit['nonexecutable_open_sensitivity_money_rows']}",
        f"- W1/W0 sealed comparison available: {audit['base_w1_available']}",
        f"- W1/W0 sealed limitation: {audit['base_w1_unavailable_reason']}",
        f"- Sealed settlements independently reproduced: {audit['sealed_bet_settlement_authority']['rows']}",
        f"- Cascade settlements independently reproduced: {audit['cascade_bet_settlement_authority']['rows']}",
        f"- Bookless proper-score void exclusions: {audit['cascade_bookless_proper_score_void_exclusion_ledger']['rows']}",
        f"- Bootstrap draws: {audit['bootstrap_draws']}",
        f"- Block sizes: {audit['block_sizes_events']}",
        "",
        "## Descriptive association preview",
        "",
    ]
    if associations.empty:
        lines.append("No association rows were available.")
    else:
        columns = [
            column
            for column in (
                "analysis_block",
                "analysis_pair",
                "round_market",
                "events",
                "control_bad_weeks",
                "bad_weeks_extreme_weather",
                "pnl_delta_vs_reference",
                "bad_week_delta_vs_reference",
                "spearman_normalization_delta_vs_weather_severity",
            )
            if column in associations
        ]
        lines.append(_report_markdown_table(associations[columns], rows=30))
    lines.extend(["", "## Whole-event uncertainty preview", ""])
    if bootstrap.empty:
        lines.append("No bootstrap rows were available.")
    else:
        columns = [
            column
            for column in (
                "analysis_block",
                "comparison_kind",
                "metric",
                "round_market",
                "bootstrap_method",
                "block_size_events",
                "events",
                "estimate",
                "ci_low",
                "ci_high",
                "probability_better",
            )
            if column in bootstrap
        ]
        lines.append(_report_markdown_table(bootstrap[columns], rows=40))
    lines.extend(
        [
            "",
            "The full event-level artifacts, bad-week rows, breakdowns, and provenance hashes are in this directory.",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _verify_event_dates(
    frame: pd.DataFrame, event_weather: pd.DataFrame, *, label: str
) -> None:
    dates = frame[["event_key", "event_start_date"]].drop_duplicates()
    if dates["event_key"].duplicated().any():
        raise ValueError(f"{label} maps one event key to multiple dates")
    expected = event_weather[["event_key", "event_start_date"]].drop_duplicates()
    merged = dates.merge(
        expected,
        on="event_key",
        how="left",
        validate="one_to_one",
        suffixes=("_artifact", "_panel"),
    )
    if merged["event_start_date_panel"].isna().any():
        raise ValueError(f"{label} has event keys absent from the panel")
    left = pd.to_datetime(merged["event_start_date_artifact"], errors="coerce").dt.normalize()
    right = pd.to_datetime(merged["event_start_date_panel"], errors="coerce").dt.normalize()
    if left.isna().any() or right.isna().any() or left.ne(right).any():
        raise ValueError(f"{label} event/date identity differs from the panel")


def run(args: argparse.Namespace, *, argv: Sequence[str] | None = None) -> None:
    raw_argv = tuple(sys.argv if argv is None else map(str, argv))
    source_path = Path(__file__).resolve()
    sealed_dir = args.sealed_dir.resolve()
    cascade_dir = args.cascade_dir.resolve()
    panel_path = args.round_panel.resolve()
    panel_manifest_path = args.panel_manifest.resolve()
    output = validate_fresh_output_path(args.output_dir)
    if int(args.bootstrap_draws) < MIN_BOOTSTRAP_DRAWS:
        raise ValueError(f"--bootstrap-draws must be >= {MIN_BOOTSTRAP_DRAWS}")
    block_sizes = tuple(int(value) for value in args.block_sizes)
    if not block_sizes or any(value < 1 for value in block_sizes):
        raise ValueError("--block-sizes must contain positive integers")
    validate_output_disjoint_from_inputs(output, (sealed_dir, cascade_dir))
    config_path, execution_config = bind_execution_configuration(
        output,
        args,
        block_sizes=block_sizes,
        argv=raw_argv,
    )

    _, panel_provenance = validate_panel_manifest(panel_path, panel_manifest_path)
    (
        sealed_manifest,
        sealed_audit,
        sealed_artifacts,
        sealed_execution_paths,
    ) = validate_sealed_provenance(sealed_dir, panel_provenance["panel_sha256"])
    (
        cascade_manifest,
        cascade_audit,
        cascade_artifacts,
        cascade_context,
    ) = validate_cascade_provenance(cascade_dir, panel_manifest_path)
    validate_sealed_frozen_chain_identity(
        sealed_manifest,
        sealed_audit,
        sealed_execution_paths,
        cascade_context["handoff"],
    )
    upstream_paths = _direct_upstream_paths(
        panel_path=panel_path,
        panel_manifest_path=panel_manifest_path,
        sealed_dir=sealed_dir,
        sealed_artifacts=sealed_artifacts,
        sealed_execution_paths=sealed_execution_paths,
        cascade_dir=cascade_dir,
        cascade_artifacts=cascade_artifacts,
        cascade_context=cascade_context,
    )
    validate_output_disjoint_from_inputs(
        output, (*upstream_paths, sealed_dir, cascade_dir)
    )
    provenance_inputs = (*upstream_paths, config_path)
    provenance_state = execution_provenance.begin(
        output_dir=output,
        primary_source=source_path,
        input_paths=provenance_inputs,
        # Preserve the exact reviewed source closure of both upstream
        # evaluators, including their primaries.  Those files are also direct
        # hash-bound inputs; recording them as code makes the weather run's
        # executable dependency closure explicit rather than incidental.
        code_paths=WEATHER_PROVENANCE_CODE_PATHS,
        cli=raw_argv,
    )
    _assert_provenance_input_set(provenance_state, provenance_inputs)

    # Re-establish all semantic validations after the immutable byte snapshot.
    # This closes the validation/begin race; finalize closes the full-run race.
    _, panel_provenance = validate_panel_manifest(panel_path, panel_manifest_path)
    (
        sealed_manifest,
        sealed_audit,
        sealed_artifacts,
        sealed_execution_paths,
    ) = validate_sealed_provenance(sealed_dir, panel_provenance["panel_sha256"])
    (
        cascade_manifest,
        cascade_audit,
        cascade_artifacts,
        cascade_context,
    ) = validate_cascade_provenance(cascade_dir, panel_manifest_path)
    validate_sealed_frozen_chain_identity(
        sealed_manifest,
        sealed_audit,
        sealed_execution_paths,
        cascade_context["handoff"],
    )
    upstream_paths_after_snapshot = _direct_upstream_paths(
        panel_path=panel_path,
        panel_manifest_path=panel_manifest_path,
        sealed_dir=sealed_dir,
        sealed_artifacts=sealed_artifacts,
        sealed_execution_paths=sealed_execution_paths,
        cascade_dir=cascade_dir,
        cascade_artifacts=cascade_artifacts,
        cascade_context=cascade_context,
    )
    if upstream_paths_after_snapshot != upstream_paths:
        raise RuntimeError("direct upstream input set changed during validation")
    _assert_execution_snapshot_current(provenance_state)

    identity_path = _require_artifact(
        sealed_artifacts, "sealed_event_weather_profiles.csv", "sealed"
    )
    event_identity = pd.read_csv(identity_path)
    metadata = pd.read_csv(
        _require_artifact(
            sealed_artifacts, "candidate_roles_and_metadata.csv", "sealed"
        )
    )
    _require(metadata, ["candidate", "candidate_role", "weather_arm"], "sealed metadata")
    selected_model = str(sealed_manifest.get("selected_model", ""))
    if selected_model != cascade_context["handoff"].selected_name:
        raise ValueError("sealed selected model differs from frozen handoff")
    base_control = frozen_handoff.STRICT_PT_CONTROL
    sealed_price_bases = _require_exact_string_menu(
        sealed_audit.get("price_bases"),
        EXPECTED_SEALED_PRICE_BASES,
        "sealed audit price_bases",
    )
    sealed_line_policies = _require_exact_string_menu(
        sealed_audit.get("line_policies"),
        EXPECTED_LINE_POLICIES,
        "sealed audit line_policies",
    )
    sealed_books = _require_exact_string_menu(
        sealed_audit.get("books"), EXPECTED_BOOKS, "sealed audit books"
    )
    sealed_sizings = _require_exact_string_menu(
        sealed_audit.get("sizing_rules"),
        EXPECTED_SEALED_SIZINGS,
        "sealed audit sizing_rules",
    )
    sealed_reliability_policies = _require_exact_string_menu(
        sealed_audit.get("reliability_policies"),
        EXPECTED_SEALED_RELIABILITY_POLICIES,
        "sealed audit reliability_policies",
    )
    sealed_quote_path = _unique_hash_bound_input(
        sealed_execution_paths,
        filename="normalized_matchup_quotes.parquet",
        label="sealed execution provenance",
    )
    sealed_quotes = pd.read_parquet(sealed_quote_path)
    sealed_quote_universe = build_sealed_price_quote_universe(
        sealed_quotes,
        event_identity,
        expected_price_bases=sealed_price_bases,
        expected_line_policies=sealed_line_policies,
        expected_books=sealed_books,
    )

    logical_path = _require_artifact(
        sealed_artifacts, "sealed_logical_probabilities.parquet", "sealed"
    )
    logical = _canonical_logical(
        pd.read_parquet(logical_path),
        "sealed logical probabilities",
    )
    source_predictions = pd.read_parquet(
        cascade_context["handoff"].source_prediction_path
    )
    (
        authoritative_metadata,
        expected_sealed_names,
        expected_sealed_roles,
        chronology_audit,
    ) = validate_sealed_candidate_chronology_bundle(
        cascade_context["handoff"],
        sealed_manifest,
        sealed_audit,
        metadata,
        source_predictions,
        pd.read_csv(
            _require_artifact(
                sealed_artifacts, "frozen_probability_mapper.csv", "sealed"
            )
        ),
        logical,
    )
    metadata_names = set(expected_sealed_names)
    sealed_logical_quote_authority = validate_sealed_logical_quote_authority(
        sealed_quotes,
        pd.read_parquet(
            _require_artifact(
                sealed_artifacts, "coverage_missing_quotes.parquet", "sealed"
            )
        ),
        source_predictions,
        logical,
        expected_candidates=expected_sealed_names,
        selected_name=selected_model,
        expected_books=sealed_books,
        expected_markets=EXPECTED_SEALED_MARKETS,
    )
    logical = normalize_sealed_event_keys(
        logical, event_identity, label="sealed logical probabilities"
    )
    if set(logical["candidate"].astype(str)) != metadata_names:
        raise ValueError("sealed logical candidate menu differs from metadata")
    base_pairing = build_w1_pairing_audit(
        authoritative_metadata, selected_model=selected_model
    )
    base_pairing = audit_w1_quote_menus(base_pairing, logical)
    base_w1_available = not base_pairing.empty
    base_w1_unavailable_reason = (
        None
        if base_w1_available
        else "frozen sealed candidate menu contains no scored W1 arm"
    )
    base_proper = (
        build_base_proper_deltas(logical, base_pairing, control=base_control)
        if base_w1_available
        else pd.DataFrame()
    )

    sealed_money = _validate_money_deltas(
        pd.read_parquet(
            _require_artifact(sealed_artifacts, "money_event_deltas.parquet", "sealed")
        )
    )
    sealed_money = normalize_sealed_event_keys(
        sealed_money, event_identity, label="sealed money event deltas"
    )
    sealed_money, sealed_money_price_universe_audit = (
        filter_sealed_money_to_executable_price_universe(
            sealed_money, sealed_quote_universe
        )
    )
    validate_sealed_money_slice_coverage(
        sealed_money,
        sealed_quote_universe,
        expected_candidates=metadata_names,
        baseline_candidate=base_control,
        expected_price_bases=sealed_price_bases,
        expected_line_policies=sealed_line_policies,
        expected_sizings=sealed_sizings,
        expected_reliability_policies=sealed_reliability_policies,
        expected_markets=EXPECTED_SEALED_MARKETS,
        expected_opportunity_modes=EXPECTED_SEALED_OPPORTUNITY_MODES,
    )
    base_money = (
        build_base_money_deltas(sealed_money, base_pairing)
        if base_w1_available
        else pd.DataFrame()
    )
    sealed_bets = load_sealed_pairwise_bets(sealed_artifacts)
    sealed_settlement_audit = validate_sealed_bet_settlement_authority(
        sealed_bets, sealed_quotes
    )
    sealed_bets = normalize_sealed_event_keys(
        sealed_bets, event_identity, label="sealed pairwise bets"
    )
    if base_w1_available:
        base_money = attach_base_bet_attribution(base_money, sealed_bets)

    cascade_probabilities_raw = pd.read_parquet(
        _require_artifact(
            cascade_artifacts, "round_matchup_probabilities.parquet", "cascade"
        )
    )
    cascade_normalized_quotes = pd.read_parquet(
        _require_artifact(
            cascade_artifacts,
            "normalized_round_matchup_quotes.parquet",
            "cascade",
        )
    )
    cascade_player_predictions = pd.read_parquet(
        _require_artifact(
            cascade_artifacts, "cascade_player_round_predictions.parquet", "cascade"
        )
    )
    cascade_round_score_authority = pd.read_parquet(
        panel_path, columns=["event_key", "round_num", "dg_id", "score"]
    )
    cascade_score_void_audit, cascade_score_void_exclusions = (
        validate_cascade_score_void_classification(
            cascade_probabilities_raw,
            cascade_normalized_quotes,
            cascade_player_predictions,
            cascade_round_score_authority,
            cascade_audit,
        )
    )
    cascade_verified_model_quote_menu = _cascade_verified_model_quote_menu(
        cascade_probabilities_raw
    )
    cascade_probability_audit = validate_cascade_probability_audit_consistency(
        cascade_probabilities_raw, cascade_audit
    )
    cascade_probabilities = _canonical_cascade_probabilities(
        cascade_probabilities_raw
    )
    expected_price_points = _require_exact_string_menu(
        cascade_audit.get("price_points_evaluated"),
        EXPECTED_CASCADE_PRICE_POINTS,
        "cascade audit price_points_evaluated",
    )
    expected_books = _require_exact_string_menu(
        cascade_audit.get("fixed_books_evaluated"),
        EXPECTED_BOOKS,
        "cascade audit fixed_books_evaluated",
    )
    _require_exact_bool(
        cascade_audit.get("best_of_three_evaluated"),
        True,
        "cascade audit best_of_three_evaluated",
    )
    expected_line_policies = EXPECTED_LINE_POLICIES
    c3_weather_lineage, c3_weather_lineage_by_event_round = (
        validate_c3_weather_state_lineage(
            cascade_player_predictions,
            cascade_manifest,
            cascade_audit,
            cascade_context["completion_authority"],
        )
    )
    cascade_pairing = build_w2_pairing_audit(
        cascade_player_predictions,
        cascade_probabilities,
        expected_candidates=cascade_audit["selected_candidates"],
    )
    c3_coefficient_lineage = validate_c3_coefficient_feature_lineage(
        pd.read_csv(_require_artifact(cascade_artifacts, "c3_coefficients.csv", "cascade")),
        cascade_pairing,
    )
    cascade_quote_audit, cascade_quote_keys = audit_cascade_quote_slices(
        cascade_normalized_quotes,
        cascade_probabilities_raw,
        cascade_pairing,
        expected_price_points=expected_price_points,
        expected_books=expected_books,
        verified_model_quote_menu=cascade_verified_model_quote_menu,
    )
    cascade_proper = build_cascade_proper_deltas(
        cascade_probabilities, cascade_pairing
    )
    cascade_bets = _validate_cascade_bets(
        pd.read_parquet(_require_artifact(cascade_artifacts, "bets.parquet", "cascade"))
    )
    cascade_settlement_audit = validate_cascade_bet_settlement_authority(
        cascade_bets,
        cascade_normalized_quotes,
        cascade_probabilities_raw,
    )
    for column, expected in (
        ("price_point", set(EXPECTED_CASCADE_PRICE_POINTS)),
        ("line_policy", set(EXPECTED_LINE_POLICIES)),
        ("sizing", set(EXPECTED_CASCADE_SIZINGS)),
        ("opportunity_mode", set(EXPECTED_CASCADE_OPPORTUNITY_MODES)),
    ):
        if set(cascade_bets[column].astype(str)) != expected:
            raise ValueError(f"cascade bets {column} menu is not code-frozen exact")
    cascade_analysis_bets, cascade_bet_model_scope_audit = (
        select_cascade_analysis_bets(cascade_bets, cascade_pairing)
    )
    validate_cascade_bets_against_quote_slices(
        cascade_analysis_bets, cascade_quote_keys
    )
    cascade_money = build_cascade_money_deltas(
        cascade_analysis_bets,
        cascade_quote_keys,
        cascade_pairing,
        expected_price_points=expected_price_points,
        expected_line_policies=expected_line_policies,
        expected_sizings=EXPECTED_CASCADE_SIZINGS,
        expected_opportunity_modes=EXPECTED_CASCADE_OPPORTUNITY_MODES,
    )
    cascade_money = attach_cascade_bet_attribution(
        cascade_money,
        cascade_analysis_bets,
        cascade_quote_keys,
    )
    if base_w1_available:
        base_money = annotate_money_execution_eligibility(
            base_money,
            price_column="price_basis",
        )
    cascade_money = annotate_money_execution_eligibility(
        cascade_money,
        price_column="price_point",
    )

    event_keys = set(logical["event_key"].astype(str)) | set(
        cascade_probabilities["event_key"].astype(str)
    )
    weather_course_rounds, weather_rounds, event_weather, weather_audit = build_weather_profiles(
        panel_path, required_event_keys=event_keys
    )
    for frame, label in (
        (logical, "sealed logical probabilities"),
        (sealed_money, "sealed money event deltas"),
        (cascade_probabilities, "cascade probabilities"),
        (cascade_bets, "cascade bets"),
    ):
        _verify_event_dates(frame, event_weather, label=label)

    if base_w1_available:
        base_money = mark_bad_weeks(attach_weather_to_base(base_money, event_weather))
        base_proper = attach_weather_to_base(base_proper, event_weather)
    cascade_money = mark_bad_weeks(
        attach_weather_to_cascade(
            cascade_money, weather_rounds, c3_weather_lineage_by_event_round
        )
    )
    cascade_proper = attach_weather_to_cascade(
        cascade_proper, weather_rounds, c3_weather_lineage_by_event_round
    )
    all_money = pd.concat([base_money, cascade_money], ignore_index=True, sort=False)
    all_proper = pd.concat([base_proper, cascade_proper], ignore_index=True, sort=False)

    executable_money = all_money.loc[
        _require_exact_boolean_series(
            all_money["execution_eligible"], "all money execution_eligible"
        )
    ].copy()
    if executable_money.empty or not executable_money["opening_price_sensitivity"].eq(
        False
    ).all():
        raise ValueError("executable money headline is empty or contains open prices")
    money_breakdowns = _money_breakdowns(executable_money)
    money_sensitivity_breakdowns = _money_breakdowns(all_money)
    proper_breakdowns = _proper_breakdowns(all_proper)
    associations = build_association_summary(executable_money)
    explanation = build_bad_week_explanation_models(executable_money)
    bad_weeks = executable_money.loc[
        executable_money[["is_control_bad_week", "is_reference_bad_week", "is_model_bad_week"]]
        .fillna(False)
        .any(axis=1)
    ].copy()
    bootstrap_input = pd.concat(
        [money_to_bootstrap_metrics(executable_money), all_proper],
        ignore_index=True,
        sort=False,
    )
    bootstrap = bootstrap_event_metrics(
        bootstrap_input,
        draws=int(args.bootstrap_draws),
        seed=int(args.seed),
        block_sizes=block_sizes,
    )

    _atomic_csv(base_pairing, output / "base_w1_pairing_and_menu_audit.csv")
    _atomic_csv(cascade_pairing, output / "cascade_w2_pairing_and_menu_audit.csv")
    _atomic_csv(cascade_quote_audit, output / "cascade_quote_slice_menu_audit.csv")
    cascade_void_exclusion_path = (
        output / "cascade_bookless_proper_score_void_exclusions.parquet"
    )
    _atomic_parquet(cascade_score_void_exclusions, cascade_void_exclusion_path)
    _atomic_parquet(
        weather_course_rounds, output / "weather_physical_course_round_profiles.parquet"
    )
    _atomic_parquet(weather_rounds, output / "weather_event_round_profiles.parquet")
    _atomic_csv(event_weather, output / "weather_event_profiles.csv")
    _atomic_parquet(base_money, output / "base_w1_event_money_deltas.parquet")
    _atomic_parquet(base_proper, output / "base_w1_event_proper_deltas.parquet")
    _atomic_parquet(cascade_money, output / "cascade_w2_event_money_deltas.parquet")
    _atomic_parquet(cascade_proper, output / "cascade_w2_event_proper_deltas.parquet")
    _atomic_parquet(bad_weeks, output / "bad_week_attribution.parquet")
    _atomic_csv(money_breakdowns, output / "money_attribution_breakdowns.csv")
    _atomic_csv(
        money_sensitivity_breakdowns,
        output / "money_all_price_sensitivity_breakdowns.csv",
    )
    _atomic_csv(proper_breakdowns, output / "proper_score_attribution_breakdowns.csv")
    _atomic_csv(associations, output / "bad_week_association_summary.csv")
    _atomic_csv(explanation, output / "bad_week_explanation_models.csv")
    _atomic_csv(bootstrap, output / "whole_event_and_block_bootstrap.csv")

    # The shared finalizer rehashes the executed source, its preserved copy,
    # execution configuration, round panel, every upstream manifest/artifact,
    # and every recursively verified handoff/provenance input.
    execution_audit = execution_provenance.finalize(provenance_state)

    audit = {
        "research_only": True,
        "production_changed": False,
        "model_selected_or_reselected": False,
        "probability_mapper_refit": False,
        "sealed_results_can_promote_model": False,
        "target_or_upcoming_realized_weather_feature_used": False,
        "target_realized_weather_is_oracle_posthoc_attribution_only": True,
        "w1_information_scope": "point_in_time_historical_round_weather_only",
        "w2_information_scope": (
            "point_in_time_historical_training_weather plus current-event rounds "
            "strictly completed before target round"
        ),
        "r1_w2_current_event_completed_weather_rounds": 0,
        "c3_completed_weather_state_lineage": c3_weather_lineage,
        "c3_completed_state_coefficient_lineage": c3_coefficient_lineage,
        "sealed_candidate_authority": {
            "candidate_names": list(expected_sealed_names),
            "candidate_roles": expected_sealed_roles,
            "strict_pt_control": base_control,
            "catalog_path": str(cascade_context["handoff"].candidate_catalog_path),
            "catalog_sha256": _sha256(
                cascade_context["handoff"].candidate_catalog_path
            ),
            "manifest_metadata_logical_money_menus_exact": True,
        },
        "sealed_chronology": chronology_audit,
        "sealed_logical_quote_authority": sealed_logical_quote_authority,
        "sealed_bet_settlement_authority": sealed_settlement_audit,
        "sealed_money_executable_price_universe": (
            sealed_money_price_universe_audit
        ),
        "weather": weather_audit,
        "sealed_manifest": {
            "path": str(sealed_dir / "manifest.json"),
            "sha256": _sha256(sealed_dir / "manifest.json"),
            "contract": sealed_manifest["contract"],
            "selected_model": selected_model,
            "selected_model_role": sealed_manifest.get("selected_model_role"),
        },
        "cascade_manifest": {
            "path": str(cascade_dir / "manifest.json"),
            "sha256": _sha256(cascade_dir / "manifest.json"),
            "contract": cascade_manifest.get("contract"),
            "selection_role": cascade_audit.get("selection_role"),
            "promotion_gate_passed": cascade_audit.get("promotion_gate_passed"),
            "provenance_mode": cascade_context.get(
                "provenance_mode", "standard_complete_evaluator_v1"
            ),
            "terminal_wrapper_complete": bool(
                cascade_context.get("terminal_wrapper_complete", False)
            ),
            "original_evaluator_complete": bool(
                cascade_context.get("original_evaluator_complete", True)
            ),
            "numeric_artifacts_external_read_only": bool(
                cascade_context.get("numeric_artifacts_external_read_only", False)
            ),
        },
        "cascade_critical_chronology_audit": cascade_context["critical_audit"],
        "cascade_probability_audit": cascade_probability_audit,
        "cascade_score_void_classification": cascade_score_void_audit,
        "cascade_bet_settlement_authority": cascade_settlement_audit,
        "cascade_bet_analysis_model_scope": cascade_bet_model_scope_audit,
        "bet_attribution_domain_contract": {
            "withdrawal_flags": "exact_boolean_or_explicit_missing_source_value",
            "reliability_flags": "exact_boolean_or_explicit_missing_source_value",
            "history_round_counts": "finite_nonnegative_integer_or_missing",
            "img_history_coverage": "finite_closed_unit_interval_or_missing",
            "missing_attribution_policy": "preserve_and_report_never_coerce_to_zero",
        },
        "cascade_bookless_proper_score_void_exclusion_ledger": {
            "path": cascade_void_exclusion_path.name,
            "rows": int(len(cascade_score_void_exclusions)),
            "sha256": _sha256(cascade_void_exclusion_path),
            "excluded_before_three_way_proper_scoring": True,
            "cross_arm_score_model_eligibility_parity_required": True,
        },
        "cascade_proper_score_scope": (
            "bookless_invariant_three_way_model_probabilities_and_physical_outcomes_only"
        ),
        "cascade_binary_pricing_and_market_devig_scope": (
            "validated_only_at_hash_bound_book_quote_level_not_collapsed"
        ),
        "panel": panel_provenance,
        "base_w1_available": base_w1_available,
        "base_w1_unavailable_reason": base_w1_unavailable_reason,
        "base_w1_pairs_exact_menu": int(
            base_pairing["quote_menu_status"].eq("exact").sum()
        ),
        "base_w1_pairs_unmatched_or_menu_mismatch": int(
            (~base_pairing["quote_menu_status"].eq("exact")).sum()
        ),
        "cascade_w2_pairs_exact_menu": int(
            cascade_pairing["quote_menu_status"].eq("exact").sum()
        ),
        "base_money_event_rows": int(len(base_money)),
        "base_proper_event_rows": int(len(base_proper)),
        "cascade_money_event_rows": int(len(cascade_money)),
        "cascade_proper_event_rows": int(len(cascade_proper)),
        "executable_primary_close_money_rows": int(len(executable_money)),
        "nonexecutable_open_sensitivity_money_rows": int(
            (~all_money["execution_eligible"].astype(bool)).sum()
        ),
        "money_headline_price_scope": "close_only",
        "open_price_scope": (
            "descriptive_nonexecutable_sensitivity_without_hash_bound_quote_and_model_state_order"
        ),
        "bad_week_rows": int(len(bad_weeks)),
        "bootstrap_draws": int(args.bootstrap_draws),
        "block_sizes_events": list(block_sizes),
        "whole_event_bootstrap_includes_zero_bet_events": True,
        "moving_block_bootstrap_preserves_event_adjacency": True,
        "bad_week_definition": (
            "negative P&L at or below the within-frozen-slice 10th percentile"
        ),
        "newcomer_definition": "either matchup golfer has fewer than five prior rounds",
        "low_coverage_definition": (
            "either golfer reliability false or either prior-round count missing"
        ),
        "association_claim": "descriptive_not_causal",
        "source_sha256": execution_audit["sources"][0]["sha256"],
        "execution_configuration": execution_config,
        "execution_provenance": execution_audit,
        "direct_upstream_files_rehashed_at_finalize": len(upstream_paths),
        "output_fresh_under_frozen_research_root": True,
        "output_disjoint_from_upstream_inputs": True,
    }
    _atomic_json(output / "audit.json", audit)
    _write_report(output, audit, associations, bootstrap)

    artifacts = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            artifacts.append(
                {
                    "path": path.relative_to(output).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    manifest = {
        "contract": "base_skill_weather_bad_week_postsealed_v1",
        "complete": True,
        "research_only": True,
        "production_changed": False,
        "sealed_results_used_for_model_selection": False,
        "target_or_upcoming_realized_weather_feature_used": False,
        "final_snapshot_verified_immediately_before_complete_manifest": True,
        "execution_provenance": execution_audit,
        "execution_configuration": {
            "path": "execution_configuration.json",
            "sha256": _sha256(config_path),
        },
        "inputs": {
            "sealed_manifest": audit["sealed_manifest"],
            "cascade_manifest": audit["cascade_manifest"],
            "round_panel": {
                "path": panel_provenance["panel_path"],
                "sha256": panel_provenance["panel_sha256"],
            },
            "panel_manifest": {
                "path": panel_provenance["panel_manifest_path"],
                "sha256": panel_provenance["panel_manifest_sha256"],
            },
        },
        "artifacts": artifacts,
    }
    # Finalize closes mutations through the end of scoring.  Rechecking the
    # original source/copy/input snapshot here also closes the report and
    # artifact-inventory interval before the only complete-manifest write.
    _write_complete_manifest_after_final_snapshot(output, manifest, provenance_state)
    print(f"Weather and bad-week audit complete: {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed-dir", type=Path, default=DEFAULT_SEALED_DIR)
    parser.add_argument("--cascade-dir", type=Path, default=DEFAULT_CASCADE_DIR)
    parser.add_argument(
        "--round-panel",
        type=Path,
        default=DEFAULT_PANEL_DIR / "player_round_panel.parquet",
    )
    parser.add_argument(
        "--panel-manifest", type=Path, default=DEFAULT_PANEL_DIR / "manifest.json"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--seed", type=int, default=20260823)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args(), argv=sys.argv)
