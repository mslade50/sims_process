"""Research-only R1-R4 matchup evaluation for the base-skill rebuild.

This module is deliberately outside the weekly pipeline.  It compares:

* C0 -- the exact current-code, retrospectively fitted cascade prediction;
* C1 -- the finalist core plus its discovery-frozen share of pre-event
  course/history downstream terms, plus the current completed-round update;
* C2-NO-LIVE -- the candidate's course-adjusted pre-event prediction carried
  unchanged through all four rounds;
* C2-SHRINK -- the current live update multiplied by a separate R1-R4 scalar
  selected on 2021-22 matchup outcomes only;
* C3-PT -- a clean, annual expanding cascade fitted only on earlier event
  editions from the new point-in-time core and completed-round state, with
  continuous event-equal partial pooling; and
* C3-HYBRID -- a clearly labelled sensitivity that lets ridge shrink the
  current legacy stage term, but therefore inherits C0's retrospective fit.

W0 uses DataGolf field-adjusted SG.  W2 neutralizes completed observations
with coefficients frozen before 2021.  Target/upcoming realized weather is
never a feature.  The outcome-selected production weather exclusion list is
not read.  Opening and closing prices enter only after predictions exist.
Discovery matchup outcomes choose ridge strength, live-update scalars, and
probability scale, while 2023+ results remain sealed.

Nothing in this file writes to a production input or external service.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO = Path(__file__).resolve().parents[1]
ARCHIVE = Path(__file__).resolve().parent
for _path in (str(REPO), str(ARCHIVE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import backtest_base_skill_market_models as base_market  # noqa: E402
from base_skill_frozen_handoff import validate_frozen_handoff_chain  # noqa: E402
from historical_odds_cache import HistoricalOddsCacheReader  # noqa: E402
from archive import evaluate_complete_base_skill_market_candidates as discovery  # noqa: E402
from market_backtest_scoring import (  # noqa: E402
    american_decimal_odds,
    american_implied_probability,
    settle_bets,
    ties_lose,
)
import research_execution_provenance as execution_provenance  # noqa: E402


PANEL_DIR = (
    REPO / "permanent_data" / "research" / "base_skill_rebuild_20200101_20260822"
)
MODEL_DIR = PANEL_DIR / "model_eval_clean_dg_v1"
DEFAULT_FINALIST_DIR = PANEL_DIR / "model_completion_v3_provenance" / "frozen_finalist"
DEFAULT_WEATHER_COEFFICIENTS = (
    PANEL_DIR / "model_eval_clean_dg_plumbing_v2" / "weather_coefficients.csv"
)
CORE_BRIDGE = PANEL_DIR / "core_bridge" / "legacy_core_bridge.parquet"
ODDS_CACHE = (
    REPO / "permanent_data" / "research" / "dg_matchup_cache_20200101_20260822"
)
DEFAULT_OUTPUT = PANEL_DIR / "cascade_eval_v3_provenance"
EXPECTED_PANEL_MANIFEST_SHA256 = (
    "cd8dea51658a7899fa0c2d2dd534d52f456485e1528290c1fba27b33c471d3db"
)
EXPECTED_PANEL_MANIFEST_BYTES = 12_767
EXPECTED_PANEL_ODDS_CACHE_MANIFEST_SHA256 = (
    "5661a5b08dd5fa748b067c0eaf15bdee4a12e29a69623872be726fb1c1836335"
)
EXPECTED_PANEL_ODDS_CACHE_MANIFEST_BYTES = 6_922
EXPECTED_PANEL_OUTPUT_KEYS = frozenset(
    {
        "audit",
        "coverage_by_year",
        "event_identity_panel",
        "img_crosswalk_extension_by_event",
        "img_crosswalk_extension_unmatched",
        "img_crosswalk_research_extended",
        "odds_event_crosswalk",
        "player_event_outcomes",
        "player_round_panel",
        "unmatched_join_samples",
    }
)

BOOKS = ("betonline", "betcris", "pinnacle")
BOOK_PRIORITY = {"pinnacle": 0, "betcris": 1, "betonline": 2}
CATEGORY_SHARES = {"ott": 0.20, "app": 0.33, "arg": 0.14, "putt": 0.33}
CATEGORIES = tuple(CATEGORY_SHARES)
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
LIVE_UPDATE_LAMBDAS = (0.0, 0.25, 0.50, 0.75, 1.0)
PRICE_POINTS = ("open", "close")
DISCOVERY_START = pd.Timestamp("2021-01-01")
DISCOVERY_END = pd.Timestamp("2022-12-31")
SEALED_START = pd.Timestamp("2023-01-01")
SEALED_END = pd.Timestamp("2026-08-22")
COMPARABLE_START = pd.Timestamp("2023-08-22")
EPS = 1e-8
EV_GATE = 0.03
C0_MODEL = "C0_current_cascade_retrospective"
FINALIST_CONTRACT = "base_skill_frozen_finalist_v1"
PROVENANCE_CODE_PATHS = tuple(
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

ATTRIBUTION_COLUMNS = (
    "history_rounds",
    "days_since_last_round",
    "reliability_ok",
    "cross_tour_fallback_reason",
    "img_history_rounds",
    "img_history_coverage",
    "prior_event_withdrawal",
)


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
    if manifest_path.stat().st_size != EXPECTED_PANEL_ODDS_CACHE_MANIFEST_BYTES:
        raise ValueError("cascade odds-cache manifest byte size differs from authority")
    if _sha256(manifest_path) != EXPECTED_PANEL_ODDS_CACHE_MANIFEST_SHA256:
        raise ValueError(
            "cascade odds-cache manifest differs from the frozen panel authority"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise ValueError("odds cache provenance manifest is incomplete")
    if any(
        manifest.get(key) != expected
        for key, expected in (
            ("tour", "pga"),
            ("mode", "full"),
            ("start_date", "2020-01-01"),
            ("end_date", "2026-08-22"),
        )
    ):
        raise ValueError("cascade odds-cache frozen scope is unexpected")
    active_shards = manifest.get("active_shards")
    if not isinstance(active_shards, list) or len(active_shards) != 22:
        raise ValueError("cascade odds-cache must contain 22 authoritative active shards")
    data_paths = [
        str(entry.get("data_path", "")) if isinstance(entry, Mapping) else ""
        for entry in active_shards
    ]
    if len(set(data_paths)) != 22 or "" in data_paths:
        raise ValueError("cascade odds-cache active shard paths are not unique")
    if manifest.get("row_counts", {}).get("matchups") != 146_351:
        raise ValueError("cascade odds-cache authoritative matchup row count changed")
    paths = [manifest_path]
    for entry in active_shards:
        if not isinstance(entry, Mapping) or not entry.get("data_path"):
            raise ValueError("odds cache has malformed active shard provenance")
        path = (cache_dir / str(entry["data_path"])).resolve()
        try:
            path.relative_to(cache_dir)
        except ValueError as exc:
            raise ValueError("odds cache shard escapes cache directory") from exc
        if not path.is_file() or _sha256(path) != str(entry.get("sha256", "")).lower():
            raise ValueError(f"odds cache shard hash mismatch: {path}")
        paths.append(path)
    return paths


def _verified_panel_provenance_paths(panel_dir: Path) -> list[Path]:
    panel_dir = panel_dir.resolve()
    manifest_path = panel_dir / "manifest.json"
    if manifest_path.stat().st_size != EXPECTED_PANEL_MANIFEST_BYTES:
        raise ValueError("panel manifest byte size differs from audited authority")
    if _sha256(manifest_path) != EXPECTED_PANEL_MANIFEST_SHA256:
        raise ValueError("panel manifest hash differs from audited authority")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("purpose")
        != "Research-only frozen outcome panel for walk-forward base-skill and market evaluation."
        or manifest.get("production_changes") is not False
    ):
        raise ValueError("panel manifest purpose/production contract is unexpected")
    parameters = manifest.get("parameters")
    if not isinstance(parameters, Mapping) or any(
        parameters.get(key) != expected
        for key, expected in (
            ("end", "2026-08-22"),
            ("market_start", "2020-01-01"),
            ("tour", "pga"),
        )
    ):
        raise ValueError("panel manifest frozen parameters are unexpected")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("panel manifest lacks hashed outputs")
    if set(outputs) != EXPECTED_PANEL_OUTPUT_KEYS:
        raise ValueError("panel manifest output inventory differs from audited authority")
    paths = [manifest_path]
    for key in sorted(EXPECTED_PANEL_OUTPUT_KEYS):
        entry = outputs.get(key)
        if not isinstance(entry, Mapping) or not entry.get("path") or not entry.get(
            "sha256"
        ):
            raise ValueError(f"panel manifest lacks hashed output {key}")
        declared = Path(str(entry["path"]))
        declared = (
            declared if declared.is_absolute() else manifest_path.parent / declared
        ).resolve()
        if declared.parent != panel_dir or not declared.is_file():
            raise ValueError(f"panel manifest output path mismatch: {key}")
        if _sha256(declared) != str(entry["sha256"]).lower():
            raise ValueError(f"panel manifest output hash mismatch: {key}")
        if entry.get("size_bytes") is not None and declared.stat().st_size != int(
            entry["size_bytes"]
        ):
            raise ValueError(f"panel manifest output byte size mismatch: {key}")
        paths.append(declared)
    odds_entry = manifest.get("inputs", {}).get("odds_cache_manifest")
    if (
        not isinstance(odds_entry, Mapping)
        or str(odds_entry.get("sha256", "")).lower()
        != EXPECTED_PANEL_ODDS_CACHE_MANIFEST_SHA256
    ):
        raise ValueError("panel manifest lacks the frozen cascade odds-cache authority")
    return paths


def _resolved_path_set(paths: Iterable[Path]) -> frozenset[Path]:
    return frozenset(Path(path).resolve() for path in paths)


def revalidate_cascade_dynamic_inputs_after_begin(
    *,
    panel_dir: Path,
    expected_panel_paths: Iterable[Path],
    odds_cache: Path,
    expected_odds_paths: Iterable[Path],
    finalist_dir: Path,
    expected_handoff_paths: Iterable[Path],
    provenance_state: execution_provenance.ExecutionProvenance,
) -> tuple[tuple[dict[str, Any], Path] | None, list[Path], list[Path]]:
    """Re-enumerate manifest-driven inputs after provenance begins."""

    tracked = frozenset(
        Path(str(record["path"])).resolve()
        for record in provenance_state.input_fingerprints
    )
    expected_panel = _resolved_path_set(expected_panel_paths)
    expected_odds = _resolved_path_set(expected_odds_paths)
    if not expected_panel.issubset(tracked) or not expected_odds.issubset(tracked):
        raise ValueError("pre-begin panel/odds paths were not all provenance-tracked")
    current_panel_paths = _verified_panel_provenance_paths(panel_dir)
    current_odds_paths = _odds_cache_provenance_paths(odds_cache)
    if _resolved_path_set(current_panel_paths) != expected_panel:
        raise ValueError("panel resolved path set changed across provenance begin")
    if _resolved_path_set(current_odds_paths) != expected_odds:
        raise ValueError("odds-cache resolved path set changed across provenance begin")

    expected_handoff = _resolved_path_set(expected_handoff_paths)
    if not expected_handoff:
        return None, current_panel_paths, current_odds_paths
    if not expected_handoff.issubset(tracked):
        raise ValueError("pre-begin handoff paths were not all provenance-tracked")
    current_finalist = _validate_finalist_manifest(finalist_dir)
    current_handoff = _resolved_path_set(
        Path(value)
        for value in current_finalist[0].get("_verified_handoff_paths", [])
    )
    if current_handoff != expected_handoff:
        raise ValueError("frozen handoff resolved path set changed across provenance begin")
    return current_finalist, current_panel_paths, current_odds_paths


def begin_cascade_execution_provenance(
    *,
    output_dir: Path,
    input_paths: Iterable[Path],
    args: argparse.Namespace,
) -> execution_provenance.ExecutionProvenance:
    return execution_provenance.begin(
        output_dir=output_dir,
        primary_source=Path(__file__).resolve(),
        input_paths=[Path(path).resolve() for path in input_paths],
        code_paths=PROVENANCE_CODE_PATHS,
        cli=_serialized_cli_settings(args),
    )


def finalize_cascade_execution_provenance(
    state: execution_provenance.ExecutionProvenance,
) -> dict[str, Any]:
    return execution_provenance.finalize(state)


def _require(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def write_cascade_complete_manifest(
    path: Path,
    payload: Mapping[str, Any],
    provenance_state: execution_provenance.ExecutionProvenance,
) -> None:
    """Write a terminal cascade/preflight manifest only on current inputs."""

    execution_provenance.assert_current(provenance_state)
    _atomic_json(path, payload)


def _stable_id(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value).strip().casefold()
    if math.isfinite(numeric) and numeric.is_integer():
        return str(int(numeric))
    return str(value).strip().casefold()


def _normalize_text(value: object) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


def _valid_american(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.notna() & ((numeric <= -100.0) | (numeric >= 100.0))


def _round_market(value: object) -> str | None:
    text = str(value).strip().casefold().replace("_", " ").replace("-", " ")
    compact = " ".join(text.split())
    no_space = compact.replace(" ", "")
    if (
        re.search(r"(?:^|\s)(?:3|three)\s*ball(?:\s|$)", compact)
        or "3ball" in no_space
        or "threeball" in no_space
    ):
        return None
    for round_num in range(1, 5):
        if compact.startswith(f"r{round_num} ") and "match" in compact:
            return f"r{round_num}"
    return None


def _populated_third_participant(value: object) -> bool:
    """Inspect provider payloads without mistaking a two-player ties-lose marker."""

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return False
        if stripped[:1] in {"{", "["}:
            try:
                return _populated_third_participant(json.loads(stripped))
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
            }:
                if isinstance(nested, (list, tuple)) and len(nested) > 2:
                    return True
            if _populated_third_participant(nested):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_populated_third_participant(nested) for nested in value)
    return False


def _unexpected_third_participant(frame: pd.DataFrame) -> pd.Series:
    """Flag a third participant only on markets parsed as two-way matchups.

    The authoritative archive also contains explicitly labelled 3-ball rows.
    Those rows must be excluded, not treated as corruption.  A populated third
    participant remains a hard failure whenever the market label otherwise
    passes the two-way R1-R4 parser.
    """

    _require(frame, ["bet_type"], "round matchup archive")
    parsed = (
        frame["round_market"]
        if "round_market" in frame
        else frame["bet_type"].map(_round_market)
    )
    third_participant = frame.apply(
        lambda row: _populated_third_participant(row.to_dict()), axis=1
    )
    return parsed.notna() & third_participant


def _round_settlement_physical_result(
    p1: float, p2: float, tie_rule_loses: bool, label: object
) -> str | None:
    if np.isnan(p1) and np.isnan(p2):
        return None
    if np.isnan(p1) or np.isnan(p2):
        raise ValueError(f"only one archived round leg is graded: {label}")
    if p1 not in {0.0, 0.5, 1.0} or p2 not in {0.0, 0.5, 1.0}:
        raise ValueError(f"invalid archived round outcome pair: {label}")
    if (p1, p2) == (1.0, 0.0):
        return "p1_win"
    if (p1, p2) == (0.0, 1.0):
        return "p2_win"
    if (p1, p2) == (0.5, 0.5) and not tie_rule_loses:
        return "tie"
    if (p1, p2) == (0.0, 0.0) and tie_rule_loses:
        return "tie"
    raise ValueError(f"impossible archived round outcome pair {(p1, p2)}: {label}")


def _validate_round_quote_settlements(raw: pd.DataFrame) -> dict[str, Any]:
    physical: dict[tuple[str, str, str], list[str | None]] = {}
    keys = ["event_key", "round_market", "pair_key", "book"]
    for group_key, group in raw.groupby(keys, dropna=False, observed=True):
        values: dict[str, float] = {}
        for side in ("p1", "p2"):
            numeric = pd.to_numeric(group[f"{side}_outcome"], errors="coerce").dropna()
            unique = np.unique(np.round(numeric.to_numpy(float), 12))
            if len(unique) > 1:
                raise ValueError(
                    f"inconsistent {side} outcome across archived captures: {group_key}"
                )
            values[side] = float(unique[0]) if len(unique) else np.nan
        rules = group["tie_rule"].dropna().astype(str).str.strip()
        rules = rules.loc[rules.ne("")]
        rule_classes = set(rules.map(ties_lose).astype(bool))
        if len(rule_classes) > 1:
            raise ValueError(f"inconsistent tie rule within book: {group_key}")
        result = _round_settlement_physical_result(
            values["p1"], values["p2"], any(rule_classes), group_key
        )
        physical.setdefault(tuple(str(value) for value in group_key[:3]), []).append(
            result
        )
    for pair_key, results in physical.items():
        graded = {result for result in results if result is not None}
        if len(graded) > 1:
            raise ValueError(
                f"round outcome disagreement across books for {pair_key}: {sorted(graded)}"
            )
        if graded and any(result is None for result in results):
            raise ValueError(f"round outcome graded for only some books: {pair_key}")
    return {
        "physical_pairs_verified": len(physical),
        "outcome_disagreements": 0,
    }


def _sample_mask(frame: pd.DataFrame, sample: str) -> pd.Series:
    dates = pd.to_datetime(frame["event_start_date"], errors="coerce").dt.normalize()
    if sample == "discovery":
        return dates.between(DISCOVERY_START, DISCOVERY_END)
    if sample == "sealed":
        return dates.between(SEALED_START, SEALED_END)
    if sample == "comparable_3y":
        return dates.between(COMPARABLE_START, SEALED_END)
    raise ValueError(f"Unknown sample {sample!r}")


def _canonicalize_quote_sides(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Orient every archived capture to the same stable-ID side ordering.

    Open and close endpoints may come from different cache captures.  Without
    canonical orientation, a provider-side p1/p2 reversal can assign an early
    opener to the wrong golfer when those captures are collapsed.
    """

    output = frame.copy()
    _require(output, ["p1_id", "p2_id"], "round matchup captures")
    swap = output["p1_id"].astype(str).gt(output["p2_id"].astype(str))
    side_suffixes = (
        "id",
        "dg_id",
        "player_name",
        "open",
        "close",
        "outcome",
        "outcome_text",
    )
    for suffix in side_suffixes:
        left_column = f"p1_{suffix}"
        right_column = f"p2_{suffix}"
        if left_column not in output or right_column not in output:
            continue
        left = output.loc[swap, left_column].copy()
        output.loc[swap, left_column] = output.loc[swap, right_column].to_numpy()
        output.loc[swap, right_column] = left.to_numpy()
    if output["p1_id"].astype(str).gt(output["p2_id"].astype(str)).any():
        raise AssertionError("round matchup captures remain inconsistently oriented")
    output["source_sides_swapped_to_canonical"] = swap.to_numpy(bool)
    return output, int(swap.sum())


def load_round_matchup_quotes(
    cache_dir: Path, crosswalk_path: Path, events: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize only two-way R1-R4 boards through the audited identity map."""

    raw = pd.DataFrame(HistoricalOddsCacheReader(cache_dir).matchups())
    if raw.empty:
        raise ValueError("Historical matchup cache has no rows")
    raw["round_market"] = raw["bet_type"].map(_round_market)
    if _unexpected_third_participant(raw).any():
        raise ValueError(
            "R1-R4 matchup archive contains a populated third participant"
        )
    raw = raw.loc[raw["round_market"].notna()].copy()
    raw["round_num"] = raw["round_market"].str[1:].astype(int)
    raw["book"] = raw["book"].map(_normalize_text)
    raw = raw.loc[raw["book"].isin(BOOKS)].copy()

    crosswalk = pd.read_parquet(crosswalk_path).copy()
    _require(
        crosswalk,
        [
            "odds_calendar_year",
            "odds_event_id",
            "odds_event_name",
            "odds_event_completed_date",
            "local_event_key",
            "event_identity_status",
        ],
        "odds event crosswalk",
    )
    crosswalk["query_year_key"] = pd.to_numeric(
        crosswalk["odds_calendar_year"], errors="coerce"
    ).astype("Int64")
    crosswalk["event_id_key"] = crosswalk["odds_event_id"].map(_stable_id)
    crosswalk["event_name_key"] = crosswalk["odds_event_name"].map(_normalize_text)
    crosswalk["event_completed_key"] = pd.to_datetime(
        crosswalk["odds_event_completed_date"], errors="coerce"
    ).dt.normalize()
    join_keys = [
        "query_year_key",
        "event_id_key",
        "event_name_key",
        "event_completed_key",
    ]
    if crosswalk.duplicated(join_keys).any():
        raise ValueError("Audited odds-event crosswalk is ambiguous on odds identity")
    invalid = crosswalk["event_identity_status"].ne("matched")
    for column in ("event_identity_ambiguous", "local_event_has_multiple_odds_ids"):
        if column in crosswalk:
            invalid |= crosswalk[column].fillna(False).astype(bool)
    if invalid.any():
        raise ValueError(
            "Audited odds-event crosswalk contains unmatched/ambiguous rows: "
            + crosswalk.loc[invalid, join_keys + ["local_event_key"]]
            .head(20)
            .to_json(orient="records", date_format="iso")
        )

    raw["query_year_key"] = pd.to_numeric(raw["query_year"], errors="coerce").astype("Int64")
    raw["event_id_key"] = raw["event_id"].map(_stable_id)
    raw["event_name_key"] = raw["event_name"].map(_normalize_text)
    raw["event_completed_key"] = pd.to_datetime(
        raw["event_completed"], errors="coerce"
    ).dt.normalize()
    crosswalk_columns = join_keys + [
        "local_event_key",
        "event_identity_method",
        "event_id_drift",
    ]
    raw = raw.merge(
        crosswalk[crosswalk_columns], on=join_keys, how="left", validate="many_to_one"
    )
    if raw["local_event_key"].isna().any():
        examples = raw.loc[
            raw["local_event_key"].isna(),
            ["query_year", "event_id", "event_name", "event_completed"],
        ].drop_duplicates()
        raise ValueError(
            "Round matchup odds failed the audited identity join: "
            + examples.head(20).to_json(orient="records", date_format="iso")
        )
    raw = raw.rename(columns={"local_event_key": "event_key"})
    event_dates = events[["event_key", "event_start_date", "event_name"]].drop_duplicates(
        "event_key"
    )
    raw = raw.merge(event_dates, on="event_key", how="left", validate="many_to_one")
    if raw["event_start_date"].isna().any():
        raise ValueError("Mapped odds event is absent from the frozen local event panel")

    for side in ("p1", "p2"):
        raw[f"{side}_id"] = raw[f"{side}_dg_id"].map(_stable_id)
        raw[f"{side}_close"] = pd.to_numeric(raw[f"{side}_close"], errors="coerce")
        raw[f"{side}_open"] = pd.to_numeric(raw[f"{side}_open"], errors="coerce")
        raw[f"{side}_outcome"] = pd.to_numeric(raw[f"{side}_outcome"], errors="coerce")
    raw, side_swaps = _canonicalize_quote_sides(raw)
    valid_identity = (
        raw["p1_id"].ne("")
        & raw["p2_id"].ne("")
        & raw["p1_id"].ne(raw["p2_id"])
    )
    raw["_valid_close_pair"] = _valid_american(raw["p1_close"]) & _valid_american(
        raw["p2_close"]
    )
    raw["_valid_open_pair"] = _valid_american(raw["p1_open"]) & _valid_american(
        raw["p2_open"]
    )
    raw = raw.loc[
        valid_identity & (raw["_valid_close_pair"] | raw["_valid_open_pair"])
    ].copy()
    raw["pair_key"] = raw["p1_id"] + "|" + raw["p2_id"]
    raw["close_time_sort"] = pd.to_datetime(raw["close_time"], errors="coerce", utc=True)
    raw["open_time_sort"] = pd.to_datetime(raw["open_time"], errors="coerce", utc=True)

    settlement_audit = _validate_round_quote_settlements(raw)

    quote_keys = ["event_key", "round_market", "book", "pair_key"]
    before = len(raw)
    # A cache may contain several API captures. Preserve the first executable
    # opener and the last executable close independently instead of assuming
    # both prices came from the same archived row.
    metadata = (
        raw.sort_values(
            ["_valid_close_pair", "close_time_sort", "_valid_open_pair", "open_time_sort"],
            ascending=[False, False, False, True],
            kind="mergesort",
            na_position="last",
        )
        .drop_duplicates(quote_keys, keep="first")
        .drop(
            columns=[
                "p1_open",
                "p2_open",
                "open_time",
                "p1_close",
                "p2_close",
                "close_time",
                "_valid_close_pair",
                "_valid_open_pair",
                "close_time_sort",
                "open_time_sort",
            ],
            errors="ignore",
        )
    )
    close_rows = (
        raw.loc[raw["_valid_close_pair"]]
        .sort_values("close_time_sort", ascending=False, kind="mergesort")
        .drop_duplicates(quote_keys, keep="first")
        [quote_keys + ["p1_close", "p2_close", "close_time"]]
    )
    open_rows = (
        raw.loc[raw["_valid_open_pair"]]
        .sort_values("open_time_sort", ascending=True, kind="mergesort")
        .drop_duplicates(quote_keys, keep="first")
        [quote_keys + ["p1_open", "p2_open", "open_time"]]
    )
    raw = metadata.merge(
        close_rows, on=quote_keys, how="left", validate="one_to_one"
    ).merge(open_rows, on=quote_keys, how="left", validate="one_to_one")
    raw = raw.reset_index(drop=True)
    raw["quote_id"] = (
        raw["event_key"].astype(str)
        + "|"
        + raw["round_market"]
        + "|"
        + raw["book"]
        + "|"
        + raw["pair_key"]
    )
    if raw.duplicated("quote_id").any():
        raise AssertionError("Normalized round quote IDs are not unique")
    audit = {
        "raw_round_matchup_rows": int(before),
        "normalized_quotes": int(len(raw)),
        "events": int(raw["event_key"].nunique()),
        "rows_by_round": raw["round_market"].value_counts().sort_index().to_dict(),
        "rows_by_book": raw["book"].value_counts().sort_index().to_dict(),
        "quotes_with_valid_open_pair": int(
            (_valid_american(raw["p1_open"]) & _valid_american(raw["p2_open"])).sum()
        ),
        "quotes_with_valid_close_pair": int(
            (_valid_american(raw["p1_close"]) & _valid_american(raw["p2_close"])).sum()
        ),
        "event_id_drift_rows": int(raw["event_id_drift"].fillna(False).astype(bool).sum()),
        "capture_side_orientation_swaps": int(side_swaps),
        "outcome_disagreements": settlement_audit["outcome_disagreements"],
        "physical_settlement_pairs_verified": settlement_audit[
            "physical_pairs_verified"
        ],
        "unresolved_quotes": int(
            (
                ~raw["p1_outcome"].isin([0.0, 0.5, 1.0])
                | ~raw["p2_outcome"].isin([0.0, 0.5, 1.0])
            ).sum()
        ),
    }
    keep = [
        "quote_id",
        "event_key",
        "event_name_y",
        "event_start_date",
        "round_market",
        "round_num",
        "book",
        "pair_key",
        "p1_id",
        "p2_id",
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
        "close_time",
        "open_time",
        "event_identity_method",
        "event_id_drift",
    ]
    output = raw[keep].rename(columns={"event_name_y": "event_name"})
    return output, audit


def _candidate_discovery_rank(model_dir: Path) -> pd.Series:
    path = model_dir / "proper_scores.csv"
    if not path.exists():
        return pd.Series(dtype=float)
    scores = pd.read_csv(path)
    if "sample" in scores:
        scores = scores.loc[scores["sample"].eq("discovery")]
    loss_column = next(
        (column for column in ("event_equal_log_loss", "log_loss") if column in scores),
        None,
    )
    if loss_column is None or "candidate" not in scores:
        return pd.Series(dtype=float)
    return scores.groupby("candidate", observed=True)[loss_column].mean().sort_values()


def choose_candidates(
    predictions: pd.DataFrame, model_dir: Path, max_new_candidates: int
) -> tuple[list[str], pd.DataFrame]:
    available = sorted(predictions["candidate"].dropna().astype(str).unique())
    incompatible = [name for name in available if name.startswith("pairwise_")]
    # The archived IMG accounting total/putting values use an empirical putting
    # transform fit in July/August 2026 and then backfilled.  Any saved base arm
    # that consumes an IMG group is therefore ineligible.  Inspect the actual
    # candidate specification instead of guessing from a name: the clean-DG v1
    # long-game arm uses DG OTT/APP only, while an older draft included IMG.
    candidate_catalog = {
        candidate.name: candidate for candidate in base_market.CANDIDATES
    }
    contaminated_img = [
        name
        for name in available
        if (
            any(
                group.startswith("img")
                for group in getattr(candidate_catalog.get(name), "groups", ())
            )
            or name.startswith("img_only_")
            or name.startswith("dg_total_img_")
            or name.startswith("all_sources_")
        )
    ]
    controls = [
        name
        for name in (
            base_market.RAW_LEGACY_CANDIDATE,
            base_market.BASELINE_CANDIDATE,
            base_market.RETROSPECTIVE_CORE_CANDIDATE,
        )
        if name in available
    ]
    new = [
        name
        for name in available
        if name not in controls
        and name not in incompatible
        and name not in contaminated_img
    ]
    ranking = _candidate_discovery_rank(model_dir)
    rank_map = ranking.to_dict()
    new.sort(key=lambda name: (rank_map.get(name, np.inf), name))

    selected: list[str] = []
    # Guarantee coverage of the prespecified source/robustness families, using
    # discovery ranking only to pick within a family.
    families = (
        lambda name: name in {"dg_total_w0", "dg_total_w1"},
        lambda name: name in {"dg_precut_total_w0", "dg_precut_total_w1"},
        lambda name: name.startswith("dg_precut_total_cat_"),
        lambda name: name.startswith("dg_total_cat_")
        and "huber" not in name
        and "ridge" not in name,
        lambda name: name.startswith("long_game_"),
        lambda name: name.startswith("dg_total_huber_"),
    )
    for predicate in families:
        match = next((name for name in new if predicate(name)), None)
        if match is not None and match not in selected:
            selected.append(match)
    for name in new:
        if len(selected) >= max_new_candidates:
            break
        if name not in selected:
            selected.append(name)
    selected = selected[:max_new_candidates]
    selection_rows = []
    for name in controls + selected:
        selection_rows.append(
            {
                "candidate": name,
                "candidate_type": "control" if name in controls else "new_base",
                "discovery_base_market_logloss": rank_map.get(name, np.nan),
                "selected": True,
                "incompatible_pairwise_scale": False,
            }
        )
    for name in incompatible:
        selection_rows.append(
            {
                "candidate": name,
                "candidate_type": "pairwise_score",
                "discovery_base_market_logloss": rank_map.get(name, np.nan),
                "selected": False,
                "incompatible_pairwise_scale": True,
                "contaminated_img_transform": False,
            }
        )
    for name in contaminated_img:
        selection_rows.append(
            {
                "candidate": name,
                "candidate_type": "excluded_img_transform",
                "discovery_base_market_logloss": rank_map.get(name, np.nan),
                "selected": False,
                "incompatible_pairwise_scale": False,
                "contaminated_img_transform": True,
            }
        )
    return controls + selected, pd.DataFrame(selection_rows)


def _validate_finalist_manifest(finalist_dir: Path) -> tuple[dict[str, Any], Path]:
    """Fail closed unless the base finalist was frozen on discovery only."""

    manifest_path = finalist_dir / "manifest.json"
    handoff = validate_frozen_handoff_chain(manifest_path)
    payload = dict(handoff.compact)
    payload["_resolved_selected_name"] = handoff.selected_name
    payload["_selection_role"] = handoff.selection_role
    payload["_promotion_gate_passed"] = handoff.promotion_gate_passed
    payload["_discovery_selected_downstream_lambda"] = handoff.downstream_lambda
    payload["_provenance_chain_audit"] = handoff.audit
    payload["_verified_completion_inputs"] = handoff.audit.get(
        "completion_input_bindings", {}
    )
    payload["_verified_source_prediction_path"] = str(
        handoff.source_prediction_path
    )
    payload["_verified_core_bridge_path"] = str(handoff.core_bridge_path)
    payload["_verified_core_bridge_sha256"] = handoff.core_bridge_sha256
    payload["_verified_handoff_paths"] = [str(path) for path in handoff.verified_paths]
    return payload, handoff.compact_prediction_path


def validate_cascade_completion_authorities(
    manifest: Mapping[str, Any],
    *,
    panel_dir: Path,
    weather_coefficients: Path,
    core_bridge: Path,
) -> dict[str, dict[str, str]]:
    """Bind cascade state inputs to exact frozen completion inputs."""

    records = manifest.get("_verified_completion_inputs")
    if not isinstance(records, Mapping):
        raise ValueError("frozen handoff lacks completion input authority bindings")
    supplied = {
        "round_panel": panel_dir.resolve() / "player_round_panel.parquet",
        "weather_coefficients": weather_coefficients.resolve(),
        "core_bridge": core_bridge.resolve(),
    }
    audit: dict[str, dict[str, str]] = {}
    for name, path in supplied.items():
        record = records.get(name)
        if not isinstance(record, Mapping) or not record.get("path") or not record.get(
            "sha256"
        ):
            raise ValueError(f"completion authority lacks input {name!r}")
        authoritative = Path(str(record["path"])).resolve()
        if path != authoritative:
            raise ValueError(
                f"cascade {name} path is not the frozen completion input"
            )
        actual = _sha256(path)
        if actual != str(record["sha256"]).lower():
            raise ValueError(f"frozen completion input hash mismatch: {name}")
        audit[name] = {"path": str(path), "sha256": actual}
    return audit


def _map_finalist_event_keys(
    predictions: pd.DataFrame, event_player_map: pd.DataFrame
) -> pd.DataFrame:
    """Map canonical event keys without relying on event names or ID stability."""

    mapping = event_player_map.copy()
    _require(
        mapping,
        ["event_key", "year", "event_id", "event_start_date", "dg_id_key"],
        "round-panel event/player map",
    )
    mapping["year_key"] = pd.to_numeric(mapping["year"], errors="coerce").astype("Int64")
    mapping["event_id_key"] = mapping["event_id"].map(_stable_id)
    mapping["dg_id_key"] = mapping["dg_id_key"].map(_stable_id)
    mapping["event_start_date"] = pd.to_datetime(
        mapping["event_start_date"], errors="coerce"
    ).dt.normalize()
    mapping = mapping.drop_duplicates(
        ["year_key", "event_id_key", "dg_id_key", "event_key"]
    )

    output = predictions.copy()
    output["canonical_event_key"] = output["event_key"].astype(str)
    parsed = output["canonical_event_key"].str.extract(
        r"^pga:R(?P<year_key>\d{4})(?P<event_id_key>\d{3})$"
    )
    output["year_key"] = pd.to_numeric(parsed["year_key"], errors="coerce").astype(
        "Int64"
    )
    output["event_id_key"] = parsed["event_id_key"].map(_stable_id)
    output["dg_id_key"] = output["dg_id"].map(_stable_id)
    output["event_start_date"] = pd.to_datetime(
        output.get("event_start", output.get("event_start_date")), errors="coerce"
    ).dt.normalize()

    exact_map = mapping[
        ["year_key", "event_id_key", "dg_id_key", "event_key", "event_start_date"]
    ].rename(
        columns={
            "event_key": "local_event_key",
            "event_start_date": "local_event_start_date",
        }
    )
    exact_keys = ["year_key", "event_id_key", "dg_id_key"]
    if exact_map.duplicated(exact_keys).any():
        raise ValueError("Round panel is ambiguous on canonical event/player identity")
    output = output.merge(exact_map, on=exact_keys, how="left", validate="many_to_one")

    # The canonical key parser is the primary identity. The date is an
    # independent guard against event-ID drift or an incorrectly frozen key.
    date_mismatch = output["local_event_key"].notna() & output[
        "local_event_start_date"
    ].ne(output["event_start_date"])
    if date_mismatch.any():
        raise ValueError(
            f"Frozen finalist event/date identity mismatches: {int(date_mismatch.sum())}"
        )
    if output["local_event_key"].isna().any():
        examples = output.loc[
            output["local_event_key"].isna(),
            ["canonical_event_key", "event_start_date", "dg_id"],
        ].head(20)
        raise ValueError(
            "Frozen finalist rows do not map to the audited PGA panel: "
            + examples.to_json(orient="records", date_format="iso")
        )
    output["event_key"] = output.pop("local_event_key")
    return output.drop(columns=["local_event_start_date", "year_key", "event_id_key"])


def load_frozen_finalist_predictions(
    finalist_dir: Path,
    event_player_map: pd.DataFrame,
    *,
    require_attribution_detail: bool = True,
    expected_verified_paths: Iterable[Path] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Read the compact finalist/control handoff and verified attribution detail."""

    manifest, prediction_path = _validate_finalist_manifest(finalist_dir)
    if expected_verified_paths is not None:
        expected = _resolved_path_set(expected_verified_paths)
        current = _resolved_path_set(
            Path(value)
            for value in manifest.get("_verified_handoff_paths", [])
        )
        if current != expected:
            raise ValueError(
                "frozen handoff resolved path set changed before prediction consumption"
            )
    frozen = pd.read_parquet(prediction_path).copy()
    _require(
        frozen,
        [
            "event_key",
            "dg_id",
            "event_start",
            "candidate",
            "finalist_core",
            "control_core",
            "split",
            "prediction_asof",
            "chronology_ok",
        ],
        "frozen finalist predictions",
    )
    declared_name = manifest.get("_resolved_selected_name")
    if not declared_name:
        raise ValueError("Frozen manifest declares neither finalist nor diagnostic model")
    finalist_name = str(declared_name)
    control_name = str(manifest["control_name"])
    selection_role = str(manifest["_selection_role"])
    promotion_gate_passed = bool(manifest["_promotion_gate_passed"])
    if set(frozen["candidate"].dropna().astype(str)) != {finalist_name}:
        raise ValueError("Frozen prediction rows do not contain exactly the declared finalist")
    if frozen.duplicated(["event_key", "dg_id"]).any():
        raise ValueError("Frozen finalist predictions contain duplicate player/event keys")
    numeric = frozen[["finalist_core", "control_core"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("Frozen finalist/control cores contain missing or non-finite values")
    frozen[["finalist_core", "control_core"]] = numeric
    frozen["event_start_date"] = pd.to_datetime(
        frozen["event_start"], errors="coerce"
    ).dt.normalize()
    frozen["prediction_asof"] = pd.to_datetime(
        frozen["prediction_asof"], errors="coerce"
    ).dt.normalize()
    chronology_ok = frozen["chronology_ok"].fillna(False).astype(bool)
    asof_bad = frozen["prediction_asof"].isna() | frozen["prediction_asof"].gt(
        frozen["event_start_date"]
    )
    if (~chronology_ok | asof_bad).any():
        raise ValueError(
            "Frozen finalist violates its row-level chronology contract: "
            f"{int((~chronology_ok | asof_bad).sum())} rows"
        )
    frozen = _map_finalist_event_keys(frozen, event_player_map)

    pieces = []
    for role, candidate, column in (
        (selection_role, finalist_name, "finalist_core"),
        ("point_in_time_control", control_name, "control_core"),
    ):
        piece = frozen[
            [
                "event_key",
                "canonical_event_key",
                "dg_id",
                "dg_id_key",
                "event_start_date",
                "prediction_asof",
                "chronology_ok",
                "split",
                column,
            ]
        ].copy()
        piece = piece.rename(columns={column: "prediction"})
        piece["candidate"] = candidate
        piece["finalist_role"] = role
        piece["prediction_units"] = "sg_per_round"
        pieces.append(piece)
    long = pd.concat(pieces, ignore_index=True)

    # Coverage/withdrawal fields are deliberately not part of the compact
    # frozen scoring handoff. Read them only from the exact hashed source for
    # post-hoc attribution; they never enter a prediction feature here.
    detail_audit: dict[str, Any] = {
        "source_detail_available": False,
        "source_detail_hash_verified": False,
        "detail_rows_attached": 0,
    }
    source_value = manifest.get("_verified_source_prediction_path")
    source_hash = str(manifest.get("source_prediction_sha256", ""))
    if source_value:
        source_path = Path(str(source_value))
        if source_path.exists():
            actual_source_hash = _sha256(source_path)
            if not source_hash or actual_source_hash != source_hash:
                raise ValueError("Frozen finalist source-prediction hash mismatch")
            source_schema = set(pq.ParquetFile(source_path).schema.names)
            required_source = {
                "candidate",
                "dg_id_key",
                "event_start_date",
                "prediction",
                *ATTRIBUTION_COLUMNS,
            }
            missing_source = sorted(required_source - source_schema)
            if missing_source:
                raise ValueError(
                    "frozen finalist source predictions missing columns: "
                    f"{missing_source}"
                )
            read_columns = [
                "candidate",
                "dg_id_key",
                "event_start_date",
                "prediction",
                *[
                    column
                    for column in ATTRIBUTION_COLUMNS
                    if column in source_schema
                ],
            ]
            source = pd.read_parquet(
                source_path,
                columns=read_columns,
                filters=[
                    ("candidate", "in", [finalist_name, control_name])
                ],
            ).copy()
            source["dg_id_key"] = source["dg_id_key"].map(_stable_id)
            source["event_start_date"] = pd.to_datetime(
                source["event_start_date"], errors="coerce"
            ).dt.normalize()
            detail_columns = read_columns
            detail = source.loc[
                source["candidate"].isin([finalist_name, control_name]), detail_columns
            ].copy()
            if detail.duplicated(["candidate", "event_start_date", "dg_id_key"]).any():
                raise ValueError("Source attribution detail is ambiguous by candidate/event/player")
            detail = detail.rename(columns={"prediction": "source_prediction_check"})
            long = long.merge(
                detail,
                on=["candidate", "event_start_date", "dg_id_key"],
                how="left",
                validate="one_to_one",
            )
            check = long["source_prediction_check"].notna()
            if not check.all():
                raise ValueError(
                    "Frozen finalist/control menu is not fully represented in its "
                    f"hashed source predictions: missing={int((~check).sum())}"
                )
            prediction_error = (
                long.loc[check, "prediction"]
                - long.loc[check, "source_prediction_check"]
            ).abs()
            if not prediction_error.empty and prediction_error.max() > 1e-10:
                raise ValueError("Frozen core differs from its hashed source prediction")
            detail_audit = {
                "source_detail_available": True,
                "source_detail_hash_verified": True,
                "source_prediction_file": str(source_path.resolve()),
                "detail_rows_attached": int(check.sum()),
                "detail_coverage_rate": float(check.mean()),
                "max_frozen_vs_source_prediction_error": float(
                    prediction_error.max() if not prediction_error.empty else 0.0
                ),
            }
            long = long.drop(columns="source_prediction_check")
    if require_attribution_detail and not detail_audit["source_detail_available"]:
        raise FileNotFoundError(
            "Coverage/withdrawal attribution requires the frozen manifest's "
            "hashed source_prediction_file"
        )

    selection = pd.DataFrame(
        [
            {
                "candidate": finalist_name,
                "candidate_type": (
                    "discovery_frozen_finalist"
                    if promotion_gate_passed
                    else "diagnostic_challenger"
                ),
                "selected": True,
                "finalist_role": selection_role,
                "promotion_gate_passed": promotion_gate_passed,
            },
            {
                "candidate": control_name,
                "candidate_type": "point_in_time_control",
                "selected": True,
                "finalist_role": "point_in_time_control",
                "promotion_gate_passed": False,
            },
        ]
    )
    audit = {
        "contract": manifest["contract"],
        "manifest_path": str((finalist_dir / "manifest.json").resolve()),
        "manifest_sha256": _sha256(finalist_dir / "manifest.json"),
        "prediction_path": str(prediction_path.resolve()),
        "prediction_sha256": _sha256(prediction_path),
        "finalist_name": finalist_name,
        "selection_role": selection_role,
        "promotion_gate_passed": promotion_gate_passed,
        "control_name": control_name,
        "selection_window_end": manifest["selection_window_end"],
        "discovery_selected_pre_event_downstream_lambda": float(
            manifest["_discovery_selected_downstream_lambda"]
        ),
        "pre_event_downstream_lambda_scope": (
            "frozen_pre_event_course_history_downstream_terms"
        ),
        "sealed_data_used_for_selection": False,
        "rows_per_candidate": int(len(frozen)),
        "duplicate_keys": int(
            long.duplicated(["candidate", "event_key", "dg_id_key"]).sum()
        ),
        "chronology_violations": int((~chronology_ok | asof_bad).sum()),
        "full_provenance_chain": manifest.get("_provenance_chain_audit", {}),
    }
    if audit["duplicate_keys"]:
        raise AssertionError("Long finalist/control interface contains duplicate keys")
    return long, selection, audit, detail_audit


def _generate_2020_backcast(
    snapshots: pd.DataFrame,
    final_predictions: pd.DataFrame,
    selected: Sequence[str],
    selected_hyperparameters: Mapping[str, float],
    quotes_72h_r1: pd.DataFrame,
    core_bridge_path: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Backcast one year so C3 can learn on 2020 before discovery begins."""

    pieces = [final_predictions]
    audit: list[dict[str, Any]] = []
    catalog = {candidate.name: candidate for candidate in base_market.CANDIDATES}
    for name in selected:
        try:
            if name == base_market.RAW_LEGACY_CANDIDATE:
                piece = base_market.legacy_predictions(snapshots)
                piece = piece.loc[piece["event_start_date"].dt.year.eq(2020)]
            elif name == base_market.BASELINE_CANDIDATE:
                piece = base_market.legacy_refit_predictions(snapshots, years=(2020,))
            elif name == base_market.RETROSPECTIVE_CORE_CANDIDATE:
                piece = base_market.retrospective_core_predictions(snapshots, core_bridge_path)
                piece = piece.loc[piece["event_start_date"].dt.year.eq(2020)]
            elif name in catalog:
                piece = base_market.annual_walk_forward_predictions(
                    snapshots,
                    catalog[name],
                    float(selected_hyperparameters[name]),
                    years=(2020,),
                )
            else:
                raise ValueError("candidate has no SG-scale backcast implementation")
            piece = piece.loc[piece["candidate"].eq(name)].copy()
            if piece.empty:
                raise ValueError("backcast returned no rows")
            pieces.append(piece)
            audit.append(
                {
                    "candidate": name,
                    "backcast_2020": True,
                    "rows": int(len(piece)),
                    "error": None,
                }
            )
        except Exception as exc:  # fail this arm closed, not the entire audit
            audit.append(
                {
                    "candidate": name,
                    "backcast_2020": False,
                    "rows": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    output = pd.concat(pieces, ignore_index=True, sort=False)
    output = output.loc[output["candidate"].isin(selected)].copy()
    output["dg_id_key"] = output["dg_id_key"].map(_stable_id)
    output["event_start_date"] = pd.to_datetime(
        output["event_start_date"], errors="coerce"
    ).dt.normalize()
    output = output.sort_values(
        ["candidate", "event_key", "dg_id_key", "event_start_date"], kind="mergesort"
    ).drop_duplicates(["candidate", "event_key", "dg_id_key"], keep="last")
    if output.duplicated(["candidate", "event_key", "dg_id_key"]).any():
        raise AssertionError("Base prediction keys remain duplicated after backcast")
    return output, audit


def prepare_round_state(
    rounds_path: Path, weather_coefficients_path: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rounds = pd.read_parquet(rounds_path).copy()
    required = [
        "event_key",
        "event_start_date",
        "event_end_date",
        "event_name",
        "year",
        "event_id",
        "dg_id",
        "player_name",
        "round_num",
        "round_date",
        "score",
        "field_size",
        "start_pos",
        "dg_sg_total_field_adjusted",
        "weather_delta_wind",
        "weather_delta_dew",
        "cascade_actual",
        "cascade_prediction",
        "cascade_residual",
        "fin_text",
    ]
    required.extend(f"dg_sg_{category}_field_adjusted" for category in CATEGORIES)
    _require(rounds, required, "frozen player-round panel")
    rounds["event_start_date"] = pd.to_datetime(
        rounds["event_start_date"], errors="coerce"
    ).dt.normalize()
    rounds["event_end_date"] = pd.to_datetime(
        rounds["event_end_date"], errors="coerce"
    ).dt.normalize()
    rounds["round_date"] = pd.to_datetime(rounds["round_date"], errors="coerce").dt.normalize()
    rounds["dg_id_key"] = rounds["dg_id"].map(_stable_id)
    rounds["round_num"] = pd.to_numeric(rounds["round_num"], errors="coerce").astype("Int64")
    rounds = rounds.loc[
        rounds["round_num"].isin([1, 2, 3, 4])
        & rounds["event_start_date"].dt.year.between(2020, 2026)
    ].copy()
    duplicate_keys = ["event_key", "dg_id_key", "round_num"]
    if rounds.duplicated(duplicate_keys).any():
        raise ValueError("Frozen panel is ambiguous on player/event/round")

    weather = pd.read_csv(weather_coefficients_path)
    _require(
        weather,
        [
            "event_start_date",
            "fit_max_round_date",
            "coefficient_source",
            "total_beta_wind",
            "total_beta_dew",
        ],
        "point-in-time weather coefficients",
    )
    weather["event_start_date"] = pd.to_datetime(
        weather["event_start_date"], errors="coerce"
    ).dt.normalize()
    weather["fit_max_round_date"] = pd.to_datetime(
        weather["fit_max_round_date"], errors="coerce"
    ).dt.normalize()
    if weather.duplicated("event_start_date").any():
        raise ValueError("Weather coefficient table is ambiguous by event date")
    coefficient_columns = [
        column
        for column in weather.columns
        if column.endswith("_beta_wind")
        or column.endswith("_beta_dew")
        or column in {"event_start_date", "fit_max_round_date", "coefficient_source"}
    ]
    rounds = rounds.merge(
        weather[coefficient_columns], on="event_start_date", how="left", validate="many_to_one"
    )
    if rounds["total_beta_wind"].isna().any():
        raise ValueError("Some round rows lack a frozen weather coefficient")
    weather_boundary_bad = (
        rounds["fit_max_round_date"].notna()
        & rounds["fit_max_round_date"].ge(rounds["event_start_date"])
    )
    if weather_boundary_bad.any():
        raise ValueError("Weather nuisance fit reaches the event being predicted")

    wind_raw = pd.to_numeric(rounds["weather_delta_wind"], errors="coerce")
    dew_raw = pd.to_numeric(rounds["weather_delta_dew"], errors="coerce")
    rounds["weather_observed"] = wind_raw.notna() & dew_raw.notna()
    finish_text = rounds["fin_text"].fillna("").astype(str).str.casefold()
    rounds["event_withdrawal"] = finish_text.str.contains(
        r"(?:^|[^a-z])(?:wd|w/d|withdrawn|withdrew)(?:[^a-z]|$)", regex=True
    )
    wind = wind_raw.fillna(0.0)
    dew = dew_raw.fillna(0.0)
    rounds["target_total_w0"] = pd.to_numeric(
        rounds["dg_sg_total_field_adjusted"], errors="coerce"
    )
    rounds["target_total_w2"] = (
        rounds["target_total_w0"]
        + rounds["total_beta_wind"] * wind
        + rounds["total_beta_dew"] * dew
    )
    for category, share in CATEGORY_SHARES.items():
        raw = pd.to_numeric(
            rounds[f"dg_sg_{category}_field_adjusted"], errors="coerce"
        )
        rounds[f"target_{category}_w0"] = raw
        rounds[f"target_{category}_w2"] = (
            raw
            + rounds[f"{category}_beta_wind"] * wind
            + rounds[f"{category}_beta_dew"] * dew
        )
        for arm in ("w0", "w2"):
            rounds[f"orth_{category}_{arm}"] = (
                rounds[f"target_{category}_{arm}"]
                - share * rounds[f"target_total_{arm}"]
            )

    # A W2 history row truly uses completed weather only when both realized
    # nuisance inputs and every W2 state component consumed by C3 are finite.
    # Missing weather deliberately falls back to W0 via zero-filled nuisance
    # deltas, but that fallback must never be labeled as weather use.
    w2_state_columns = ["target_total_w2"] + [
        f"orth_{category}_w2" for category in CATEGORIES
    ]
    w2_state = rounds[w2_state_columns].apply(pd.to_numeric, errors="coerce")
    rounds["weather_w2_state_valid"] = (
        rounds["weather_observed"]
        & w2_state.notna().all(axis=1)
        & np.isfinite(w2_state).all(axis=1)
    )

    rounds = rounds.sort_values(
        ["event_start_date", "event_key", "dg_id_key", "round_num"], kind="mergesort"
    ).reset_index(drop=True)
    group_keys = ["event_key", "dg_id_key"]
    grouped = rounds.groupby(group_keys, sort=False, observed=True)
    rounds["completed_rounds"] = grouped.cumcount()
    rounds["completed_rounds_available"] = (
        pd.to_numeric(rounds["round_num"], errors="coerce") - 1
    ).astype("Int64")
    field_size = pd.to_numeric(rounds["field_size"], errors="coerce")
    start_pos = pd.to_numeric(rounds["start_pos"], errors="coerce")
    rounds["start_pos_scaled"] = np.where(
        field_size.gt(1), (start_pos - 1.0) / (field_size - 1.0), np.nan
    )

    state_columns: list[str] = []
    for arm in ("w0", "w2"):
        source_columns = [f"target_total_{arm}"] + [
            f"orth_{category}_{arm}" for category in CATEGORIES
        ]
        for source in source_columns:
            label = source.replace("target_", "").replace(f"_{arm}", "")
            # A golf event has at most four rounds.  Explicit 1/2/3-round lags
            # are much faster than tens of thousands of GroupBy.expanding
            # objects and make the strict completed-round boundary obvious.
            lags = pd.concat(
                [grouped[source].shift(lag) for lag in (1, 2, 3)], axis=1
            )
            lags.columns = ["lag1", "lag2", "lag3"]
            if source == f"target_total_{arm}":
                rounds[f"completed_rounds_used_{arm}"] = lags.notna().sum(axis=1)
            shifted = lags["lag1"]
            earliest = lags["lag3"].combine_first(lags["lag2"]).combine_first(
                lags["lag1"]
            )
            rounds[f"prev_last_{label}_{arm}"] = shifted
            rounds[f"prev_mean_{label}_{arm}"] = lags.mean(axis=1, skipna=True)
            rounds[f"prev_std_{label}_{arm}"] = lags.std(axis=1, skipna=True, ddof=0)
            rounds[f"prev_trend_{label}_{arm}"] = shifted - earliest
            state_columns.extend(
                [
                    f"prev_last_{label}_{arm}",
                    f"prev_mean_{label}_{arm}",
                    f"prev_std_{label}_{arm}",
                    f"prev_trend_{label}_{arm}",
                ]
            )
    weather_observation_lags = pd.concat(
        [grouped["weather_observed"].shift(lag) for lag in (1, 2, 3)], axis=1
    )
    weather_state_lags = pd.concat(
        [grouped["weather_w2_state_valid"].shift(lag) for lag in (1, 2, 3)],
        axis=1,
    )
    rounds["completed_weather_observations_available"] = (
        weather_observation_lags.eq(True).sum(axis=1).astype(int)
    )
    rounds["completed_weather_state_rows_with_valid_coverage"] = (
        weather_state_lags.eq(True).sum(axis=1).astype(int)
    )
    # Compatibility name retained, now with the stricter meaning used by W2.
    rounds["completed_weather_observations_used"] = rounds[
        "completed_weather_state_rows_with_valid_coverage"
    ]
    available_denominator = pd.to_numeric(
        rounds["completed_rounds_available"], errors="coerce"
    )
    rounds["completed_prior_weather_coverage_rate"] = np.where(
        available_denominator.gt(0),
        rounds["completed_weather_state_rows_with_valid_coverage"]
        / available_denominator,
        0.0,
    )
    rounds["prev_weather_observed_rate"] = rounds[
        "completed_prior_weather_coverage_rate"
    ]
    rounds["weather_coefficients_path"] = str(weather_coefficients_path.resolve())
    rounds["weather_coefficients_sha256"] = _sha256(
        weather_coefficients_path.resolve()
    )
    rounds["completed_prior_round_boundary"] = (
        "round_num_strictly_less_than_target_round"
    )
    rounds["target_or_upcoming_realized_weather_excluded"] = True
    rounds["completed_prior_rounds_only"] = True
    rounds["weather_state_source_columns_w0"] = (
        "target_total_w0|orth_ott_w0|orth_app_w0|orth_arg_w0|orth_putt_w0"
    )
    rounds["weather_state_source_columns_w2"] = (
        "target_total_w2|orth_ott_w2|orth_app_w2|orth_arg_w2|orth_putt_w2"
    )
    state_boundary_bad = (
        rounds["completed_rounds"].gt(rounds["completed_rounds_available"])
        | rounds["completed_rounds_used_w0"].gt(rounds["completed_rounds_available"])
        | rounds["completed_rounds_used_w2"].gt(rounds["completed_rounds_available"])
        | rounds["completed_weather_observations_available"].gt(
            rounds["completed_rounds_available"]
        )
        | rounds["completed_weather_state_rows_with_valid_coverage"].gt(
            rounds["completed_weather_observations_available"]
        )
        | rounds["completed_weather_state_rows_with_valid_coverage"].gt(
            rounds["completed_rounds_used_w2"]
        )
        | (
            rounds["round_num"].eq(1)
            & (
                rounds["completed_rounds_used_w0"].ne(0)
                | rounds["completed_rounds_used_w2"].ne(0)
                | rounds["completed_weather_observations_available"].ne(0)
                | rounds[
                    "completed_weather_state_rows_with_valid_coverage"
                ].ne(0)
            )
        )
    )
    if state_boundary_bad.any():
        raise ValueError("Completed-round weather state crosses the target-round boundary")

    # Compatibility audit only.  The clean C3 target above never consumes
    # cascade_actual, its outcome-selected exclusions, or its [-10, 10] clip.
    cascade_overlap = rounds.loc[
        rounds["cascade_actual"].notna() & rounds["target_total_w0"].notna()
    ].copy()
    cascade_diff = (
        pd.to_numeric(cascade_overlap["cascade_actual"], errors="coerce")
        - cascade_overlap["target_total_w0"]
    )
    residual_identity = (
        pd.to_numeric(cascade_overlap["cascade_actual"], errors="coerce")
        - pd.to_numeric(cascade_overlap["cascade_prediction"], errors="coerce")
        - pd.to_numeric(cascade_overlap["cascade_residual"], errors="coerce")
    )
    audit = {
        "round_rows": int(len(rounds)),
        "events": int(rounds["event_key"].nunique()),
        "duplicate_player_event_round_keys": int(rounds.duplicated(duplicate_keys).sum()),
        "weather_coefficient_boundary_violations": int(weather_boundary_bad.sum()),
        "weather_coverage_rate": float(rounds["weather_observed"].mean()),
        "weather_w2_state_valid_rate": float(
            rounds["weather_w2_state_valid"].mean()
        ),
        "cascade_overlap_rows": int(len(cascade_overlap)),
        "cascade_actual_vs_clean_w0_mean_abs_diff": float(cascade_diff.abs().mean()),
        "cascade_actual_vs_clean_w0_max_abs_diff": float(cascade_diff.abs().max()),
        "cascade_residual_identity_mean_abs_diff": float(residual_identity.abs().mean()),
        "cascade_residual_identity_max_abs_diff": float(residual_identity.abs().max()),
        "clean_c3_uses_cascade_actual": False,
        "outcome_selected_weather_exclusion_list_read": False,
        "img_columns_used": False,
        "target_or_upcoming_weather_feature_used": False,
        "completed_prior_round_boundary_violations": int(state_boundary_bad.sum()),
        "weather_coefficients_path": str(weather_coefficients_path.resolve()),
        "weather_coefficients_sha256": _sha256(weather_coefficients_path.resolve()),
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
        "completed_prior_round_boundary": (
            "round_num_strictly_less_than_target_round"
        ),
        "target_or_upcoming_realized_weather_excluded": True,
        "weather_state_source_columns": {
            "w0": (
                "target_total_w0|orth_ott_w0|orth_app_w0|orth_arg_w0|orth_putt_w0"
            ),
            "w2": (
                "target_total_w2|orth_ott_w2|orth_app_w2|orth_arg_w2|orth_putt_w2"
            ),
        },
    }
    keep = [
        "event_key",
        "event_name",
        "event_start_date",
        "event_end_date",
        "year",
        "event_id",
        "dg_id",
        "dg_id_key",
        "player_name",
        "round_num",
        "round_date",
        "round_score",
        "field_size",
        "start_pos",
        "start_pos_scaled",
        "completed_rounds",
        "completed_rounds_available",
        "completed_rounds_used_w0",
        "completed_rounds_used_w2",
        "completed_weather_observations_available",
        "completed_weather_state_rows_with_valid_coverage",
        "completed_weather_observations_used",
        "weather_delta_wind",
        "weather_delta_dew",
        "weather_wind",
        "weather_dew",
        "weather_field_mean_wind",
        "weather_field_mean_dew",
        "weather_observed",
        "weather_w2_state_valid",
        "event_withdrawal",
        "prev_weather_observed_rate",
        "completed_prior_weather_coverage_rate",
        "target_total_w0",
        "target_total_w2",
        "cascade_actual",
        "cascade_prediction",
        "cascade_residual",
        "fit_max_round_date",
        "coefficient_source",
        "weather_coefficients_path",
        "weather_coefficients_sha256",
        "completed_prior_round_boundary",
        "target_or_upcoming_realized_weather_excluded",
        "completed_prior_rounds_only",
        "weather_state_source_columns_w0",
        "weather_state_source_columns_w2",
        *state_columns,
    ]
    rounds = rounds.rename(columns={"score": "round_score"})
    keep = list(dict.fromkeys(column for column in keep if column in rounds.columns))
    return rounds[keep].copy(), audit


def attach_core_and_candidate(
    rounds: pd.DataFrame,
    base_predictions: pd.DataFrame,
    core_bridge: pd.DataFrame,
    candidate: str,
    *,
    pre_event_downstream_lambda: float = 1.0,
    pre_event_downstream_lambda_source: str = "compatibility_default_lambda_one",
) -> pd.DataFrame:
    if float(pre_event_downstream_lambda) not in {0.0, 0.5, 1.0}:
        raise ValueError("pre-event downstream lambda must be 0, 0.5, or 1")
    prediction_columns = [
        "event_key",
        "dg_id_key",
        "event_start_date",
        "prediction",
        "training_max_event_end",
        "prediction_asof",
        "chronology_ok",
        "history_rounds",
        "days_since_last_round",
        "reliability_ok",
        "cross_tour_fallback_reason",
        "img_history_rounds",
        "img_history_coverage",
        "prior_event_withdrawal",
        "finalist_role",
    ]
    available = [column for column in prediction_columns if column in base_predictions]
    base = base_predictions.loc[
        base_predictions["candidate"].eq(candidate), available
    ].copy()
    base = base.rename(
        columns={
            "prediction": "base_core",
            "training_max_event_end": "base_training_max_event_end",
        }
    )
    base["dg_id_key"] = base["dg_id_key"].map(_stable_id)
    if base.duplicated(["event_key", "dg_id_key"]).any():
        raise ValueError(f"Duplicate base predictions for {candidate}")
    output = rounds.merge(
        base.drop(columns=["event_start_date"], errors="ignore"),
        on=["event_key", "dg_id_key"],
        how="left",
        validate="many_to_one",
    )
    output["base_prediction_available"] = output["base_core"].notna()

    bridge = core_bridge.copy()
    bridge["event_start_date"] = pd.to_datetime(
        bridge.get("event_start", bridge.get("event_start_date")), errors="coerce"
    ).dt.normalize()
    bridge["dg_id_key"] = bridge["dg_id"].map(_stable_id)
    bridge_columns = [
        "event_start_date",
        "dg_id_key",
        "legacy_core_pred_step2",
        "baseline_r1_prediction",
        "downstream_preserved_total",
        "reconstruction_error",
    ]
    if bridge.duplicated(["event_start_date", "dg_id_key"]).any():
        raise ValueError("Legacy core bridge is ambiguous by event date/player")
    numeric_bridge = bridge[
        [
            "legacy_core_pred_step2",
            "baseline_r1_prediction",
            "downstream_preserved_total",
            "reconstruction_error",
        ]
    ].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric_bridge.to_numpy(float)).all():
        raise ValueError("Legacy core bridge reconstruction fields must be finite")
    bridge_identity_error = (
        numeric_bridge["baseline_r1_prediction"]
        - numeric_bridge["legacy_core_pred_step2"]
        - numeric_bridge["downstream_preserved_total"]
    ).abs()
    if (
        bridge_identity_error.max() > 1e-10
        or numeric_bridge["reconstruction_error"].abs().max() > 1e-10
    ):
        raise ValueError("Legacy core bridge downstream reconstruction mismatch")
    output = output.merge(
        bridge[bridge_columns],
        on=["event_start_date", "dg_id_key"],
        how="left",
        validate="many_to_one",
    )
    output["base_delta"] = output["base_core"] - output["legacy_core_pred_step2"]
    output["pre_event_downstream_lambda"] = float(pre_event_downstream_lambda)
    output["pre_event_downstream_lambda_source"] = (
        pre_event_downstream_lambda_source
    )
    output["pre_event_downstream_lambda_scope"] = (
        "frozen_pre_event_course_history_downstream_terms"
    )
    output["base_event_hybrid"] = (
        output["base_core"]
        + output["pre_event_downstream_lambda"]
        * output["downstream_preserved_total"]
    )
    output["legacy_stage_adjustment"] = (
        output["cascade_prediction"] - output["baseline_r1_prediction"]
    )
    output["c1_prediction"] = (
        output["base_event_hybrid"] + output["legacy_stage_adjustment"]
    )
    output["candidate"] = candidate
    if "base_training_max_event_end" not in output:
        output["base_training_max_event_end"] = pd.NaT
    if "prediction_asof" not in output:
        output["prediction_asof"] = pd.NaT
    if "chronology_ok" not in output:
        output["chronology_ok"] = True
    training_max = pd.to_datetime(output["base_training_max_event_end"], errors="coerce")
    # NaT is expected for the raw legacy as-of control.  Every model-generated
    # dated fit must end before the event it predicts.
    boundary_bad = training_max.notna() & training_max.ge(output["event_start_date"])
    prediction_asof = pd.to_datetime(output["prediction_asof"], errors="coerce")
    boundary_bad |= prediction_asof.notna() & prediction_asof.gt(
        output["event_start_date"]
    )
    boundary_bad |= output["base_prediction_available"] & ~output[
        "chronology_ok"
    ].fillna(False).astype(bool)
    if boundary_bad.any():
        raise ValueError(f"Base prediction leakage for {candidate}: {int(boundary_bad.sum())}")
    output["base_training_boundary_violation"] = boundary_bad
    return output


def audit_c3_weather_lineage(
    predictions: pd.DataFrame,
    *,
    authoritative_coefficients_path: Path,
    authoritative_coefficients_sha256: str,
) -> dict[str, Any]:
    """Fail closed on explicit completed-round W0/W2 feature lineage."""

    c3 = predictions.loc[
        predictions["variant"].astype(str).str.startswith("C3_")
    ].copy()
    required = {
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
    }
    _require(c3, required, "C3 weather lineage")
    if c3.empty:
        raise ValueError("C3 weather lineage cannot be audited on an empty prediction set")
    arm = c3["weather_arm"].astype(str)
    if not arm.isin(["w0", "w2"]).all():
        raise ValueError("C3 weather lineage has an unknown arm")
    expected_target = arm.map({"w0": "target_total_w0", "w2": "target_total_w2"})
    expected_sources = arm.map(
        {
            "w0": (
                "target_total_w0|orth_ott_w0|orth_app_w0|orth_arg_w0|orth_putt_w0"
            ),
            "w2": (
                "target_total_w2|orth_ott_w2|orth_app_w2|orth_arg_w2|orth_putt_w2"
            ),
        }
    )
    available = pd.to_numeric(c3["completed_rounds_available"], errors="coerce")
    used = pd.to_numeric(c3["completed_rounds_used"], errors="coerce")
    weather_available = pd.to_numeric(
        c3["completed_weather_observations_available"], errors="coerce"
    )
    weather_valid = pd.to_numeric(
        c3["completed_weather_state_rows_with_valid_coverage"], errors="coerce"
    )
    weather_compat_used = pd.to_numeric(
        c3["completed_weather_observations_used"], errors="coerce"
    )
    weather_used_by_w2 = pd.to_numeric(
        c3["completed_weather_state_rows_used_by_w2"], errors="coerce"
    )
    coverage_rate = pd.to_numeric(
        c3["completed_prior_weather_coverage_rate"], errors="coerce"
    )
    round_num = pd.to_numeric(c3["round_num"], errors="coerce")
    fit_cutoff = pd.to_datetime(c3["weather_coefficient_fit_cutoff"], errors="coerce")
    event_start = pd.to_datetime(c3["event_start_date"], errors="coerce")
    expected_strict = c3["variant"].astype(str).str.startswith("C3_pt_")
    authoritative_path = authoritative_coefficients_path.resolve()
    path_ok = c3["weather_coefficients_path"].map(
        lambda value: Path(str(value)).resolve() == authoritative_path
    )
    expected_weather_used_by_w2 = weather_valid.where(arm.eq("w2"), 0.0)
    expected_weather_bool = expected_weather_used_by_w2.gt(0)
    expected_coverage_rate = pd.Series(
        np.where(available.gt(0), weather_valid / available, 0.0),
        index=c3.index,
    )
    failures = {
        "available_not_round_minus_one": int(available.ne(round_num - 1).sum()),
        "used_exceeds_available": int((used.lt(0) | used.gt(available)).sum()),
        "r1_uses_completed_state": int((round_num.eq(1) & used.ne(0)).sum()),
        "weather_available_exceeds_completed_rounds": int(
            (
                weather_available.lt(0)
                | weather_available.gt(available)
                | weather_available.isna()
            ).sum()
        ),
        "valid_weather_state_exceeds_available": int(
            (
                weather_valid.lt(0)
                | weather_valid.gt(weather_available)
                | weather_valid.gt(used)
                | weather_valid.isna()
            ).sum()
        ),
        "compat_weather_count_mismatch": int(
            weather_compat_used.ne(weather_valid).sum()
        ),
        "w2_weather_used_count_mismatch": int(
            weather_used_by_w2.ne(expected_weather_used_by_w2).sum()
        ),
        "w2_weather_used_flag_mismatch": int(
            c3["completed_weather_used_by_w2"]
            .eq(True)
            .ne(expected_weather_bool)
            .sum()
        ),
        "prior_weather_coverage_rate_mismatch": int(
            (
                coverage_rate.isna()
                | ~np.isclose(
                    coverage_rate.to_numpy(float),
                    expected_coverage_rate.to_numpy(float),
                    atol=1e-12,
                    rtol=0.0,
                )
            ).sum()
        ),
        "r1_claims_weather_use": int(
            (
                round_num.eq(1)
                & (
                    weather_available.ne(0)
                    | weather_valid.ne(0)
                    | weather_used_by_w2.ne(0)
                    | c3["completed_weather_used_by_w2"].eq(True)
                )
            ).sum()
        ),
        "target_column_mismatch": int(
            c3["weather_state_target_column"].astype(str).ne(expected_target).sum()
        ),
        "source_columns_mismatch": int(
            c3["weather_state_source_columns"].astype(str).ne(expected_sources).sum()
        ),
        "coefficient_cutoff_not_strict_prior": int(
            (fit_cutoff.isna() | event_start.isna() | fit_cutoff.ge(event_start)).sum()
        ),
        "coefficient_source_missing": int(
            c3["weather_coefficient_source"].fillna("").astype(str).str.strip().eq("").sum()
        ),
        "coefficient_path_mismatch": int((~path_ok).sum()),
        "coefficient_hash_mismatch": int(
            c3["weather_coefficients_sha256"]
            .astype(str)
            .str.lower()
            .ne(authoritative_coefficients_sha256.lower())
            .sum()
        ),
        "target_or_upcoming_exclusion_false": int(
            (~c3["target_or_upcoming_realized_weather_excluded"].eq(True)).sum()
        ),
        "completed_prior_only_false": int(
            (~c3["completed_prior_rounds_only"].eq(True)).sum()
        ),
        "boundary_label_mismatch": int(
            c3["completed_prior_round_boundary"]
            .astype(str)
            .ne("round_num_strictly_less_than_target_round")
            .sum()
        ),
        "strict_pt_label_mismatch": int(
            c3["strict_point_in_time"].astype(bool).ne(expected_strict).sum()
        ),
        "current_weather_role_mismatch": int(
            c3["current_round_weather_columns_role"]
            .astype(str)
            .ne("posthoc_outcome_attribution_only_not_model_features")
            .sum()
        ),
    }
    if any(failures.values()):
        raise ValueError(f"C3 weather lineage audit failed: {failures}")
    return {
        "contract": "completed_prior_round_weather_state_v1",
        "rows": int(len(c3)),
        "arms": sorted(arm.unique()),
        "weather_coefficients_path": str(authoritative_path),
        "weather_coefficients_sha256": authoritative_coefficients_sha256.lower(),
        "completed_rounds_available_definition": "round_num_minus_one",
        "completed_rounds_used_definition": "nonmissing_strictly_shifted_total_state_rows",
        "completed_weather_observations_available_definition": (
            "strictly_prior_rows_with_finite_wind_and_dew_deltas"
        ),
        "completed_weather_state_rows_with_valid_coverage_definition": (
            "strictly_prior_rows_with_finite_wind_dew_and_all_w2_state_components"
        ),
        "completed_weather_used_by_w2_definition": (
            "weather_arm_is_w2_and_strictly_prior_valid_wind_dew_w2_state_count_gt_zero"
        ),
        "completed_prior_weather_coverage_rate_definition": (
            "valid_strictly_prior_w2_weather_state_rows_divided_by_completed_rounds_available"
        ),
        "completed_prior_round_boundary": "round_num_strictly_less_than_target_round",
        "target_or_upcoming_realized_weather_excluded": True,
        "current_round_weather_columns_role": (
            "posthoc_outcome_attribution_only_not_model_features"
        ),
        "failures": failures,
    }


def _feature_spec(frame: pd.DataFrame, round_num: int, arm: str, mode: str) -> list[str]:
    if mode not in {"pt", "hybrid"}:
        raise ValueError(mode)
    anchor = "base_event_hybrid"
    frame[f"prev_mean_resid_{arm}_{mode}"] = frame[f"prev_mean_total_{arm}"] - frame[anchor]
    frame[f"prev_last_resid_{arm}_{mode}"] = frame[f"prev_last_total_{arm}"] - frame[anchor]
    # The trend is already a difference of completed outcomes and does not
    # need the level anchor removed.
    base_features = [anchor]
    if mode == "hybrid":
        base_features.append("legacy_stage_adjustment")
    if round_num == 1:
        return base_features
    features = base_features + [
        f"prev_mean_resid_{arm}_{mode}",
        "start_pos_scaled",
        "prev_weather_observed_rate",
    ]
    features.extend(f"prev_mean_orth_{category}_{arm}" for category in CATEGORIES)
    if round_num >= 3:
        features.extend(
            [
                f"prev_last_resid_{arm}_{mode}",
                f"prev_std_total_{arm}",
                f"prev_trend_total_{arm}",
            ]
        )
        for category in CATEGORIES:
            features.extend(
                [
                    f"prev_last_orth_{category}_{arm}",
                    f"prev_trend_orth_{category}_{arm}",
                ]
            )
    return features


@dataclass
class CascadeFit:
    prediction: pd.DataFrame
    coefficients: pd.DataFrame


def fit_cascade_year(
    candidate_frame: pd.DataFrame,
    *,
    candidate: str,
    arm: str,
    mode: str,
    round_num: int,
    prediction_year: int,
    alpha: float,
) -> CascadeFit:
    # The feature spec adds two completed-state residual columns.  Reusing the
    # candidate frame avoids copying the full player-round state for every
    # alpha/year fold; each training/test slice remains an isolated copy.
    frame = candidate_frame
    features = _feature_spec(frame, round_num, arm, mode)
    target = f"target_total_{arm}"
    cutoff = pd.Timestamp(f"{prediction_year}-01-01")
    train = frame.loc[
        frame["round_num"].eq(round_num)
        & frame["event_end_date"].lt(cutoff)
        & frame[target].notna()
    ].copy()
    test = frame.loc[
        frame["round_num"].eq(round_num)
        & frame["event_start_date"].dt.year.eq(prediction_year)
    ].copy()
    anchor = "base_event_hybrid"
    train = train.loc[train[anchor].notna()].copy()
    test = test.loc[test[anchor].notna()].copy()
    if len(train) < 250 or test.empty:
        return CascadeFit(pd.DataFrame(), pd.DataFrame())
    usable = [feature for feature in features if train[feature].notna().any()]
    if not usable or anchor not in usable:
        return CascadeFit(pd.DataFrame(), pd.DataFrame())
    event_counts = train.groupby("event_key", observed=True)[target].transform("size")
    weights = 1.0 / event_counts.to_numpy(float)
    weights *= len(weights) / weights.sum()
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )
    pipeline.fit(train[usable], train[target], ridge__sample_weight=weights)
    predicted = pipeline.predict(test[usable])
    model_name = f"C3_{mode}_continuous_partial_pool_{arm}__{candidate}"
    output_columns = [
            "event_key",
            "event_name",
            "event_start_date",
            "event_end_date",
            "dg_id_key",
            "dg_id",
            "player_name",
            "round_num",
            "round_date",
            "round_score",
            "base_core",
            "base_delta",
            "base_event_hybrid",
            "pre_event_downstream_lambda",
            "pre_event_downstream_lambda_source",
            "pre_event_downstream_lambda_scope",
            "c1_prediction",
            "weather_delta_wind",
            "weather_delta_dew",
            *[
                column
                for column in (
                    "weather_wind",
                    "weather_dew",
                    "weather_field_mean_wind",
                    "weather_field_mean_dew",
                )
                if column in test
            ],
            "weather_observed",
            "completed_rounds_available",
            f"completed_rounds_used_{arm}",
            "completed_weather_observations_available",
            "completed_weather_state_rows_with_valid_coverage",
            "completed_weather_observations_used",
            "completed_prior_weather_coverage_rate",
            "fit_max_round_date",
            "coefficient_source",
            "weather_coefficients_path",
            "weather_coefficients_sha256",
            "completed_prior_round_boundary",
            "target_or_upcoming_realized_weather_excluded",
            "completed_prior_rounds_only",
            f"weather_state_source_columns_{arm}",
            "event_withdrawal",
            "start_pos_scaled",
            target,
            *[column for column in ATTRIBUTION_COLUMNS if column in test],
        ]
    output = test[list(dict.fromkeys(output_columns))].copy()
    output["model"] = model_name
    output["candidate"] = candidate
    output["variant"] = f"C3_{mode}_continuous_partial_pool_{arm}"
    output["prediction"] = predicted
    output["ridge_alpha"] = float(alpha)
    output["training_rows"] = int(len(train))
    output["training_events"] = int(train["event_key"].nunique())
    output["training_max_event_end"] = train["event_end_date"].max()
    output["prediction_year"] = int(prediction_year)
    output["strict_point_in_time"] = mode == "pt"
    output["retrospective_cascade_anchor"] = mode == "hybrid"
    output["weather_arm"] = arm
    output["weather_state_target_column"] = target
    output["weather_state_source_columns"] = output.pop(
        f"weather_state_source_columns_{arm}"
    )
    output["completed_rounds_used"] = output.pop(
        f"completed_rounds_used_{arm}"
    ).astype(int)
    output["completed_weather_state_rows_used_by_w2"] = (
        output["completed_weather_state_rows_with_valid_coverage"].astype(int)
        if arm == "w2"
        else 0
    )
    output["completed_weather_used_by_w2"] = output[
        "completed_weather_state_rows_used_by_w2"
    ].gt(0)
    output["weather_coefficient_fit_cutoff"] = pd.to_datetime(
        output.pop("fit_max_round_date"), errors="coerce"
    )
    output["weather_coefficient_source"] = output.pop("coefficient_source")
    output["current_round_weather_columns_role"] = (
        "posthoc_outcome_attribution_only_not_model_features"
    )
    output["continuous_sg_target"] = True
    output["partial_pooling"] = "event_equal_global_ridge"
    output["live_update_lambda_scope"] = (
        "not_scalar_continuous_partial_pool_completed_prior_round_state"
    )

    scaler = pipeline.named_steps["scale"]
    ridge = pipeline.named_steps["ridge"]
    coefficient_rows = []
    for feature, coefficient in zip(usable, ridge.coef_):
        coefficient_rows.append(
            {
                "model": model_name,
                "candidate": candidate,
                "mode": mode,
                "weather_arm": arm,
                "weather_state_target_column": target,
                "weather_state_source_columns": (
                    f"target_total_{arm}|orth_ott_{arm}|orth_app_{arm}|"
                    f"orth_arg_{arm}|orth_putt_{arm}"
                ),
                "completed_prior_round_boundary": (
                    "round_num_strictly_less_than_target_round"
                ),
                "target_or_upcoming_realized_weather_excluded": True,
                "completed_weather_used_by_w2_definition": (
                    "weather_arm_is_w2_and_strictly_prior_valid_wind_dew_w2_state_count_gt_zero"
                ),
                "strict_point_in_time": mode == "pt",
                "round_num": round_num,
                "prediction_year": prediction_year,
                "ridge_alpha": float(alpha),
                "training_rows": int(len(train)),
                "training_events": int(train["event_key"].nunique()),
                "training_max_event_end": train["event_end_date"].max(),
                "feature": feature,
                "standardized_coefficient": float(coefficient),
                "raw_scale_coefficient": float(coefficient / scaler.scale_[usable.index(feature)]),
            }
        )
    coefficient_rows.append(
        {
            "model": model_name,
            "candidate": candidate,
            "mode": mode,
            "weather_arm": arm,
            "weather_state_target_column": target,
            "weather_state_source_columns": (
                f"target_total_{arm}|orth_ott_{arm}|orth_app_{arm}|"
                f"orth_arg_{arm}|orth_putt_{arm}"
            ),
            "completed_prior_round_boundary": (
                "round_num_strictly_less_than_target_round"
            ),
            "target_or_upcoming_realized_weather_excluded": True,
            "completed_weather_used_by_w2_definition": (
                "weather_arm_is_w2_and_strictly_prior_valid_wind_dew_w2_state_count_gt_zero"
            ),
            "strict_point_in_time": mode == "pt",
            "round_num": round_num,
            "prediction_year": prediction_year,
            "ridge_alpha": float(alpha),
            "training_rows": int(len(train)),
            "training_events": int(train["event_key"].nunique()),
            "training_max_event_end": train["event_end_date"].max(),
            "feature": "__intercept_scaled__",
            "standardized_coefficient": float(ridge.intercept_),
            "raw_scale_coefficient": np.nan,
        }
    )
    return CascadeFit(output, pd.DataFrame(coefficient_rows))


def player_predictions_to_pairs(
    player_predictions: pd.DataFrame, quotes: pd.DataFrame
) -> pd.DataFrame:
    columns = [
        "model",
        "candidate",
        "variant",
        "event_key",
        "dg_id_key",
        "round_num",
        "prediction",
        "base_core",
        "weather_delta_wind",
        "weather_delta_dew",
        "weather_wind",
        "weather_dew",
        "weather_field_mean_wind",
        "weather_field_mean_dew",
        "weather_observed",
        "event_withdrawal",
        "start_pos_scaled",
        "round_score",
        "strict_point_in_time",
        "retrospective_cascade_anchor",
        *ATTRIBUTION_COLUMNS,
    ]
    available = [column for column in columns if column in player_predictions]
    lookup = player_predictions[available].copy()
    if lookup.duplicated(["model", "event_key", "dg_id_key", "round_num"]).any():
        raise ValueError("Player prediction table is ambiguous by model/event/player/round")
    rename_common = {
        "prediction": "skill",
        "base_core": "base_core",
        "weather_delta_wind": "weather_delta_wind",
        "weather_delta_dew": "weather_delta_dew",
        "weather_wind": "weather_wind",
        "weather_dew": "weather_dew",
        "weather_field_mean_wind": "weather_field_mean_wind",
        "weather_field_mean_dew": "weather_field_mean_dew",
        "weather_observed": "weather_observed",
        "event_withdrawal": "event_withdrawal",
        "start_pos_scaled": "start_pos_scaled",
        "round_score": "round_score",
        **{column: column for column in ATTRIBUTION_COLUMNS},
    }
    left = lookup.rename(
        columns={
            "dg_id_key": "p1_id",
            **{key: f"p1_{value}" for key, value in rename_common.items()},
        }
    )
    right = lookup.rename(
        columns={
            "dg_id_key": "p2_id",
            **{key: f"p2_{value}" for key, value in rename_common.items()},
        }
    )
    joined = quotes.merge(
        left,
        on=["event_key", "round_num", "p1_id"],
        how="inner",
        validate="many_to_many",
    )
    joined = joined.merge(
        right,
        on=["model", "candidate", "variant", "event_key", "round_num", "p2_id"],
        how="inner",
        suffixes=("", "_right"),
        validate="many_to_one",
    )
    joined["skill_difference"] = joined["p1_skill"] - joined["p2_skill"]
    joined["abs_realized_wind_difference"] = (
        joined["p1_weather_delta_wind"] - joined["p2_weather_delta_wind"]
    ).abs()
    joined["abs_realized_dew_difference"] = (
        joined["p1_weather_delta_dew"] - joined["p2_weather_delta_dew"]
    ).abs()
    joined["pair_weather_observed"] = (
        joined.get("p1_weather_observed", False)
        & joined.get("p2_weather_observed", False)
    )
    joined["realized_round_mean_wind"] = joined[
        ["p1_weather_wind", "p2_weather_wind"]
    ].mean(axis=1)
    joined["realized_round_field_mean_wind"] = joined[
        ["p1_weather_field_mean_wind", "p2_weather_field_mean_wind"]
    ].mean(axis=1)
    joined = classify_score_outcomes(joined)
    return joined.loc[np.isfinite(joined["skill_difference"])].copy()


def audit_model_quote_coverage(
    player_predictions: pd.DataFrame, quotes: pd.DataFrame
) -> pd.DataFrame:
    """Measure every arm against the exact C0 quote menu, never a self-selected menu."""

    rows: list[dict[str, Any]] = []
    quote_use = quotes.loc[
        pd.to_datetime(quotes["event_start_date"], errors="coerce").dt.year.ge(2021)
    ].copy()
    quote_keys = ["quote_id", "event_key", "round_num", "p1_id", "p2_id"]
    quote_use = quote_use[quote_keys].drop_duplicates("quote_id")
    model_pair_ids: dict[str, set[str]] = {}
    for model, prediction in player_predictions.groupby("model", observed=True):
        available = prediction.loc[
            prediction["prediction"].notna(),
            ["event_key", "round_num", "dg_id_key"],
        ].drop_duplicates()
        p1 = available.rename(columns={"dg_id_key": "p1_id"}).assign(_p1=True)
        p2 = available.rename(columns={"dg_id_key": "p2_id"}).assign(_p2=True)
        matched = quote_use.merge(
            p1, on=["event_key", "round_num", "p1_id"], how="left"
        ).merge(p2, on=["event_key", "round_num", "p2_id"], how="left")
        matched_ids = set(
            matched.loc[
                matched["_p1"].fillna(False) & matched["_p2"].fillna(False),
                "quote_id",
            ].astype(str)
        )
        model_pair_ids[str(model)] = matched_ids
    c0_ids = model_pair_ids.get(C0_MODEL, set())
    for model, prediction in player_predictions.groupby("model", observed=True):
        model_ids = model_pair_ids[str(model)]
        common = c0_ids.intersection(model_ids)
        rows.append(
            {
                "model": model,
                "candidate": prediction["candidate"].iloc[0],
                "variant": prediction["variant"].iloc[0],
                "quotes_with_both_predictions": int(len(model_ids)),
                "c0_quote_menu": int(len(c0_ids)),
                "quotes_on_c0_common_menu": int(len(common)),
                "coverage_vs_c0_rate": float(len(common) / len(c0_ids))
                if c0_ids
                else np.nan,
                "quotes_outside_c0_menu": int(len(model_ids - c0_ids)),
                "c0_quotes_dropped": int(len(c0_ids - model_ids)),
            }
        )
    return pd.DataFrame(rows)


def classify_score_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    """Separate integer-score ties from administrative/no-score voids."""

    output = frame.copy()
    p1_score = pd.to_numeric(output["p1_round_score"], errors="coerce")
    p2_score = pd.to_numeric(output["p2_round_score"], errors="coerce")
    p1_archived = pd.to_numeric(output["p1_outcome"], errors="coerce")
    p2_archived = pd.to_numeric(output["p2_outcome"], errors="coerce")
    both_valid = p1_score.notna() & p2_score.notna()
    p1_derived = np.where(
        p1_score.lt(p2_score),
        1.0,
        np.where(p1_score.gt(p2_score), 0.0, 0.5),
    )
    p2_derived = np.where(
        p2_score.lt(p1_score),
        1.0,
        np.where(p2_score.gt(p1_score), 0.0, 0.5),
    )
    p1_archived_graded = p1_archived.isin([0.0, 0.5, 1.0])
    p2_archived_graded = p2_archived.isin([0.0, 0.5, 1.0])
    archived_pair_graded = p1_archived_graded & p2_archived_graded
    tie_rule_loses = output.get(
        "tie_rule", pd.Series("", index=output.index)
    ).map(ties_lose)
    physical_tie = both_valid & p1_score.eq(p2_score)
    p1_settlement = np.where(
        physical_tie & tie_rule_loses, 0.0, p1_derived
    )
    p2_settlement = np.where(
        physical_tie & tie_rule_loses, 0.0, p2_derived
    )
    p1_matches = pd.Series(
        np.isclose(p1_archived.to_numpy(float), p1_settlement, equal_nan=False),
        index=output.index,
    )
    p2_matches = pd.Series(
        np.isclose(p2_archived.to_numpy(float), p2_settlement, equal_nan=False),
        index=output.index,
    )
    possible_pair = (
        (p1_archived.eq(1.0) & p2_archived.eq(0.0))
        | (p1_archived.eq(0.0) & p2_archived.eq(1.0))
        | (p1_archived.eq(0.5) & p2_archived.eq(0.5) & ~tie_rule_loses)
        | (p1_archived.eq(0.0) & p2_archived.eq(0.0) & tie_rule_loses)
    )
    output["both_target_round_scores_valid"] = both_valid
    output["derived_score_outcome"] = np.where(both_valid, p1_derived, np.nan)
    output["p1_derived_score_outcome"] = np.where(
        both_valid, p1_derived, np.nan
    )
    output["p2_derived_score_outcome"] = np.where(
        both_valid, p2_derived, np.nan
    )
    output["settlement_score_outcome"] = np.where(
        both_valid, p1_settlement, np.nan
    )
    output["p1_settlement_score_outcome"] = np.where(
        both_valid, p1_settlement, np.nan
    )
    output["p2_settlement_score_outcome"] = np.where(
        both_valid, p2_settlement, np.nan
    )
    output["tie_rule_ties_lose"] = tie_rule_loses
    output["archived_outcome_pair_possible"] = possible_pair
    output["p1_archive_outcome_matches_score"] = (
        both_valid & p1_archived_graded & p1_matches
    )
    output["p2_archive_outcome_matches_score"] = (
        both_valid & p2_archived_graded & p2_matches
    )
    output["archive_outcome_matches_score"] = (
        both_valid
        & archived_pair_graded
        & possible_pair
        & p1_matches
        & p2_matches
    )
    output["actual_score_tie"] = physical_tie
    output["score_tie_void"] = (
        p1_archived.eq(0.5)
        & p2_archived.eq(0.5)
        & physical_tie
        & ~tie_rule_loses
    )
    output["score_tie_loss"] = (
        p1_archived.eq(0.0)
        & p2_archived.eq(0.0)
        & physical_tie
        & tie_rule_loses
    )
    output["no_score_void"] = (
        p1_archived.eq(0.5) & p2_archived.eq(0.5) & ~both_valid
    )
    output["other_void_with_scores"] = (
        (p1_archived.eq(0.5) | p2_archived.eq(0.5))
        & both_valid
        & ~output["actual_score_tie"]
    )
    output["score_model_eligible"] = output["archive_outcome_matches_score"]
    output["market_devig_proper_score_eligible"] = output[
        "score_model_eligible"
    ] & ~physical_tie & ~tie_rule_loses
    output["market_devig_proper_score_exclusion_reason"] = np.select(
        [
            ~output["score_model_eligible"],
            tie_rule_loses,
            physical_tie,
        ],
        [
            "ungraded_or_score_settlement_mismatch",
            "ties_lose_without_draw_quote_conditional_devig_not_comparable",
            "physical_tie_excluded_from_resolved_two_way_benchmark",
        ],
        default="eligible_resolved_push_on_tie_two_way_benchmark",
    )
    # Compatibility alias retained for older report code.  It now means the
    # market-benchmark binary score is valid, not merely that a bet settles.
    output["binary_model_eligible"] = output[
        "market_devig_proper_score_eligible"
    ]
    return output


def audit_quote_score_voids(
    quotes: pd.DataFrame, round_state: pd.DataFrame
) -> dict[str, Any]:
    score_lookup = round_state[
        ["event_key", "round_num", "dg_id_key", "round_score"]
    ].drop_duplicates(["event_key", "round_num", "dg_id_key"])
    left = score_lookup.rename(
        columns={"dg_id_key": "p1_id", "round_score": "p1_round_score"}
    )
    right = score_lookup.rename(
        columns={"dg_id_key": "p2_id", "round_score": "p2_round_score"}
    )
    audited = quotes.merge(
        left, on=["event_key", "round_num", "p1_id"], how="left", validate="many_to_one"
    ).merge(
        right, on=["event_key", "round_num", "p2_id"], how="left", validate="many_to_one"
    )
    audited = classify_score_outcomes(audited)
    void = audited["p1_outcome"].eq(0.5) & audited["p2_outcome"].eq(0.5)
    both_valid = audited["both_target_round_scores_valid"]
    graded_with_scores = (
        audited["p1_outcome"].isin([0.0, 0.5, 1.0])
        & audited["p2_outcome"].isin([0.0, 0.5, 1.0])
        & both_valid
    )
    mismatch = graded_with_scores & ~audited["archive_outcome_matches_score"]
    return {
        "quotes": int(len(audited)),
        "archived_voids": int(void.sum()),
        "score_tie_voids": int(audited["score_tie_void"].sum()),
        "score_tie_losses": int(audited["score_tie_loss"].sum()),
        "no_score_voids": int(audited["no_score_void"].sum()),
        "other_voids_with_two_non_tied_scores": int(
            audited["other_void_with_scores"].sum()
        ),
        "quotes_with_two_valid_round_scores": int(both_valid.sum()),
        "score_model_eligible_quotes": int(audited["score_model_eligible"].sum()),
        "market_devig_proper_score_eligible_quotes": int(
            audited["market_devig_proper_score_eligible"].sum()
        ),
        "ties_lose_quotes_excluded_from_market_proper_score": int(
            (
                audited["score_model_eligible"]
                & audited["tie_rule_ties_lose"]
            ).sum()
        ),
        "archived_vs_score_outcome_mismatches_when_both_scores_valid": int(
            mismatch.sum()
        ),
        "p1_archived_vs_score_mismatches": int(
            (graded_with_scores & ~audited["p1_archive_outcome_matches_score"]).sum()
        ),
        "p2_archived_vs_score_mismatches": int(
            (graded_with_scores & ~audited["p2_archive_outcome_matches_score"]).sum()
        ),
        "impossible_archived_outcome_pairs": int(
            (graded_with_scores & ~audited["archived_outcome_pair_possible"]).sum()
        ),
        "rows_by_round": audited.groupby("round_market", observed=True).agg(
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
        ).astype(int).to_dict(orient="index"),
    }


def _canonical_market_rows(frame: pd.DataFrame) -> pd.DataFrame:
    chosen = frame.copy()
    chosen["_book_priority"] = chosen["book"].map(BOOK_PRIORITY).fillna(99)
    return (
        chosen.sort_values(
            ["event_key", "round_market", "pair_key", "_book_priority"],
            kind="mergesort",
        )
        .drop_duplicates(["event_key", "round_market", "pair_key"], keep="first")
        .drop(columns="_book_priority")
    )


def _best_price_market_rows(
    frame: pd.DataFrame, price_point: str
) -> pd.DataFrame:
    """Choose the best canonical-p1 price while retaining its same-book p2 quote."""

    if price_point not in PRICE_POINTS:
        raise ValueError(price_point)
    chosen = frame.copy()
    chosen["_book_priority"] = chosen["book"].map(BOOK_PRIORITY).fillna(99)
    time_column = f"{price_point}_time"
    chosen["_execution_time"] = pd.to_datetime(
        chosen.get(time_column), errors="coerce", utc=True
    )
    return (
        chosen.sort_values(
            [
                "event_key",
                "round_market",
                "pair_key",
                f"p1_{price_point}",
                "_book_priority",
                "_execution_time",
            ],
            ascending=[True, True, True, False, True, price_point == "open"],
            kind="mergesort",
            na_position="last",
        )
        .drop_duplicates(["event_key", "round_market", "pair_key"], keep="first")
        .drop(columns=["_book_priority", "_execution_time"])
    )


def _event_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("event_key", observed=True)["pair_key"].transform("size")
    return 1.0 / counts.to_numpy(float)


def discrete_round_probabilities(
    difference: np.ndarray | pd.Series,
    sigma: float,
    tie_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Symmetric latent-normal score model with an explicit push interval.

    If latent p1-minus-p2 score is Normal(mean=difference, sd=sigma), p1 wins
    above ``+tie_threshold``, p2 wins below ``-tie_threshold``, and otherwise
    the observed integer round scores tie.  Reversing the matchup swaps win
    and loss exactly while leaving push unchanged.
    """

    difference_array = np.asarray(difference, dtype=float)
    p1_win = norm.cdf((difference_array - tie_threshold) / sigma)
    p2_win = norm.cdf((-difference_array - tie_threshold) / sigma)
    push = np.clip(1.0 - p1_win - p2_win, 0.0, 1.0)
    total = p1_win + p2_win + push
    p1_win = p1_win / total
    p2_win = p2_win / total
    push = push / total
    return p1_win, p2_win, push


def fit_sigma_and_loss(
    pair_predictions: pd.DataFrame,
) -> tuple[float, float, float, int, int]:
    graded = pair_predictions.loc[
        pair_predictions["score_model_eligible"].fillna(False).astype(bool)
    ].copy()
    graded = _canonical_market_rows(graded)
    if len(graded) < 75 or graded["event_key"].nunique() < 5:
        return (
            np.nan,
            np.nan,
            np.nan,
            int(len(graded)),
            int(graded["event_key"].nunique()),
        )
    diff = graded["skill_difference"].to_numpy(float)
    outcome = graded["derived_score_outcome"].to_numpy(float)
    weights = _event_equal_weights(graded)

    def objective(parameters: np.ndarray) -> float:
        sigma = math.exp(float(parameters[0]))
        tie_threshold = math.exp(float(parameters[1]))
        p1_win, p2_win, push = discrete_round_probabilities(
            diff, sigma, tie_threshold
        )
        probability = np.where(
            np.isclose(outcome, 1.0),
            p1_win,
            np.where(np.isclose(outcome, 0.0), p2_win, push),
        )
        loss = -np.log(np.clip(probability, EPS, 1.0))
        return float(np.sum(weights * loss) / np.sum(weights))

    fit = minimize(
        objective,
        np.log(np.array([2.8, 0.30], dtype=float)),
        method="L-BFGS-B",
        bounds=[(math.log(0.05), math.log(12.0)), (math.log(0.001), math.log(3.0))],
    )
    if not fit.success:
        return (
            np.nan,
            np.nan,
            np.nan,
            int(len(graded)),
            int(graded["event_key"].nunique()),
        )
    sigma, tie_threshold = np.exp(fit.x)
    return (
        float(sigma),
        float(tie_threshold),
        objective(fit.x),
        int(len(graded)),
        int(graded["event_key"].nunique()),
    )


def tune_and_predict_c3(
    candidate_frame: pd.DataFrame,
    quotes: pd.DataFrame,
    *,
    candidate: str,
    arm: str,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tuning_rows: list[dict[str, Any]] = []
    chosen: dict[int, float] = {}
    for round_num in range(1, 5):
        round_quotes = quotes.loc[quotes["round_num"].eq(round_num)]
        for alpha in RIDGE_ALPHAS:
            discovery_pieces = []
            for year in (2021, 2022):
                fit = fit_cascade_year(
                    candidate_frame,
                    candidate=candidate,
                    arm=arm,
                    mode=mode,
                    round_num=round_num,
                    prediction_year=year,
                    alpha=alpha,
                )
                if not fit.prediction.empty:
                    discovery_pieces.append(fit.prediction)
            if discovery_pieces:
                player = pd.concat(discovery_pieces, ignore_index=True)
                pairs = player_predictions_to_pairs(player, round_quotes)
                pairs = pairs.loc[_sample_mask(pairs, "discovery")]
                sigma, tie_threshold, loss, resolved, events = fit_sigma_and_loss(pairs)
            else:
                sigma, tie_threshold, loss, resolved, events = (
                    np.nan,
                    np.nan,
                    np.nan,
                    0,
                    0,
                )
            tuning_rows.append(
                {
                    "candidate": candidate,
                    "mode": mode,
                    "weather_arm": arm,
                    "round_num": round_num,
                    "ridge_alpha": float(alpha),
                    "discovery_sigma": sigma,
                    "discovery_tie_threshold": tie_threshold,
                    "discovery_event_equal_logloss": loss,
                    "discovery_resolved_pairs": resolved,
                    "discovery_events": events,
                }
            )
        tuning = pd.DataFrame(tuning_rows)
        eligible = tuning.loc[
            tuning["round_num"].eq(round_num)
            & tuning["discovery_event_equal_logloss"].notna()
        ].copy()
        if eligible.empty:
            continue
        best_loss = eligible["discovery_event_equal_logloss"].min()
        tied = eligible.loc[
            eligible["discovery_event_equal_logloss"].le(best_loss + 1e-10)
        ]
        # More regularization is the conservative discovery tie-break.
        chosen[round_num] = float(tied["ridge_alpha"].max())

    prediction_pieces: list[pd.DataFrame] = []
    coefficient_pieces: list[pd.DataFrame] = []
    for round_num, alpha in chosen.items():
        for year in range(2021, 2027):
            fit = fit_cascade_year(
                candidate_frame,
                candidate=candidate,
                arm=arm,
                mode=mode,
                round_num=round_num,
                prediction_year=year,
                alpha=alpha,
            )
            if not fit.prediction.empty:
                prediction_pieces.append(fit.prediction)
                coefficient_pieces.append(fit.coefficients)
    predictions = (
        pd.concat(prediction_pieces, ignore_index=True)
        if prediction_pieces
        else pd.DataFrame()
    )
    coefficients = (
        pd.concat(coefficient_pieces, ignore_index=True)
        if coefficient_pieces
        else pd.DataFrame()
    )
    return predictions, coefficients, pd.DataFrame(tuning_rows)


def build_c0_c1_player_predictions(
    rounds: pd.DataFrame,
    candidate_frames: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    common = [
        "event_key",
        "event_name",
        "event_start_date",
        "event_end_date",
        "dg_id_key",
        "dg_id",
        "player_name",
        "round_num",
        "round_date",
        "round_score",
        "weather_delta_wind",
        "weather_delta_dew",
        "weather_wind",
        "weather_dew",
        "weather_field_mean_wind",
        "weather_field_mean_dew",
        "weather_observed",
        "event_withdrawal",
        "start_pos_scaled",
    ]
    c0 = rounds.loc[
        rounds["cascade_prediction"].notna() & rounds["event_start_date"].dt.year.ge(2021),
        common + ["cascade_prediction"],
    ].copy()
    c0["model"] = C0_MODEL
    c0["candidate"] = "current_cascade"
    c0["variant"] = "C0"
    c0["prediction"] = c0.pop("cascade_prediction")
    c0["base_core"] = c0["prediction"]
    c0["base_delta"] = 0.0
    c0["c1_prediction"] = c0["prediction"]
    c0["strict_point_in_time"] = False
    c0["retrospective_cascade_anchor"] = True
    c0["weather_arm"] = "legacy_target_sensitivity"
    pieces = [c0]
    for candidate, frame in candidate_frames.items():
        candidate_detail = [
            column for column in ATTRIBUTION_COLUMNS if column in frame
        ]
        c1 = frame.loc[
            frame["c1_prediction"].notna()
            & frame["event_start_date"].dt.year.ge(2021),
            common
            + [
                "base_core",
                "base_delta",
                "base_event_hybrid",
                "legacy_stage_adjustment",
                "pre_event_downstream_lambda",
                "pre_event_downstream_lambda_source",
                "pre_event_downstream_lambda_scope",
                "c1_prediction",
                *candidate_detail,
            ],
        ].copy()
        c1["model"] = f"C1_current_plus_delta__{candidate}"
        c1["candidate"] = candidate
        c1["variant"] = "C1"
        c1["prediction"] = c1["c1_prediction"]
        c1["live_update_lambda"] = 1.0
        c1["live_update_lambda_scope"] = (
            "completed_prior_round_current_cascade_update_only"
        )
        c1["strict_point_in_time"] = False
        c1["retrospective_cascade_anchor"] = True
        c1["weather_arm"] = "legacy_target_sensitivity"
        pieces.append(c1)
        no_live = c1.copy()
        no_live["model"] = f"C2_no_live_update__{candidate}"
        no_live["variant"] = "C2_NO_LIVE_UPDATE"
        no_live["prediction"] = no_live["base_event_hybrid"]
        no_live["live_update_lambda"] = 0.0
        no_live["strict_point_in_time"] = False
        no_live["retrospective_cascade_anchor"] = True
        no_live["weather_arm"] = "no_completed_round_update"
        pieces.append(no_live)
    output = pd.concat(pieces, ignore_index=True, sort=False)
    duplicate = output.duplicated(["model", "event_key", "dg_id_key", "round_num"])
    if duplicate.any():
        raise ValueError(f"C0/C1 player predictions contain {int(duplicate.sum())} duplicate keys")
    return output


def _live_lambda_player_predictions(
    candidate_frame: pd.DataFrame,
    *,
    candidate: str,
    lambdas_by_round: Mapping[int, float],
    model: str,
    variant: str,
) -> pd.DataFrame:
    common = [
        "event_key",
        "event_name",
        "event_start_date",
        "event_end_date",
        "dg_id_key",
        "dg_id",
        "player_name",
        "round_num",
        "round_date",
        "round_score",
        "weather_delta_wind",
        "weather_delta_dew",
        "weather_observed",
        "event_withdrawal",
        "start_pos_scaled",
        "base_core",
        "base_delta",
        "base_event_hybrid",
        "legacy_stage_adjustment",
        *[
            column
            for column in (
                "pre_event_downstream_lambda",
                "pre_event_downstream_lambda_source",
                "pre_event_downstream_lambda_scope",
            )
            if column in candidate_frame
        ],
        *[
            column
            for column in (
                "weather_wind",
                "weather_dew",
                "weather_field_mean_wind",
                "weather_field_mean_dew",
            )
            if column in candidate_frame
        ],
        *[column for column in ATTRIBUTION_COLUMNS if column in candidate_frame],
    ]
    output = candidate_frame.loc[
        candidate_frame["event_start_date"].dt.year.ge(2021)
        & candidate_frame["base_event_hybrid"].notna()
        & candidate_frame["legacy_stage_adjustment"].notna(),
        list(dict.fromkeys(common)),
    ].copy()
    output["live_update_lambda"] = output["round_num"].map(lambdas_by_round)
    output["live_update_lambda_scope"] = (
        "completed_prior_round_current_cascade_update_only"
    )
    output = output.loc[output["live_update_lambda"].notna()].copy()
    output["prediction"] = (
        output["base_event_hybrid"]
        + output["live_update_lambda"] * output["legacy_stage_adjustment"]
    )
    output["model"] = model
    output["candidate"] = candidate
    output["variant"] = variant
    output["strict_point_in_time"] = False
    output["retrospective_cascade_anchor"] = True
    output["weather_arm"] = "current_live_update_discovery_scalar_shrink"
    return output


def tune_live_update_shrinkage(
    candidate_frame: pd.DataFrame,
    quotes: pd.DataFrame,
    *,
    candidate: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose one current-live-update scalar per round using 2021-22 only."""

    tuning_rows: list[dict[str, Any]] = []
    chosen: dict[int, float] = {}
    for round_num in range(1, 5):
        for value in LIVE_UPDATE_LAMBDAS:
            temp_model = f"_discovery_lambda_{round_num}_{value:g}__{candidate}"
            player = _live_lambda_player_predictions(
                candidate_frame,
                candidate=candidate,
                lambdas_by_round={round_num: value},
                model=temp_model,
                variant="_DISCOVERY_TUNING_ONLY",
            )
            pairs = player_predictions_to_pairs(
                player, quotes.loc[quotes["round_num"].eq(round_num)]
            )
            discovery = pairs.loc[_sample_mask(pairs, "discovery")].copy()
            sigma, tie_threshold, loss, resolved, events = fit_sigma_and_loss(
                discovery
            )
            tuning_rows.append(
                {
                    "candidate": candidate,
                    "round_num": round_num,
                    "live_update_lambda": float(value),
                    "discovery_sigma": sigma,
                    "discovery_tie_threshold": tie_threshold,
                    "discovery_event_equal_three_way_logloss": loss,
                    "discovery_graded_pairs": resolved,
                    "discovery_events": events,
                    "selection_min_event_date": discovery["event_start_date"].min()
                    if not discovery.empty
                    else pd.NaT,
                    "selection_max_event_date": discovery["event_start_date"].max()
                    if not discovery.empty
                    else pd.NaT,
                    "sealed_rows_used_for_selection": 0,
                }
            )
        round_tuning = pd.DataFrame(tuning_rows)
        eligible = round_tuning.loc[
            round_tuning["round_num"].eq(round_num)
            & round_tuning[
                "discovery_event_equal_three_way_logloss"
            ].notna()
        ].copy()
        if eligible.empty:
            raise ValueError(
                f"No discovery matchup sample can tune live shrinkage for R{round_num}"
            )
        best = eligible[
            "discovery_event_equal_three_way_logloss"
        ].min()
        # When the discovery score is numerically tied, choose more shrinkage.
        tied = eligible.loc[
            eligible["discovery_event_equal_three_way_logloss"].le(best + 1e-10)
        ]
        chosen[round_num] = float(tied["live_update_lambda"].min())

    tuning = pd.DataFrame(tuning_rows)
    tuning["selected"] = [
        math.isclose(float(value), chosen[int(round_num)], abs_tol=1e-12)
        for value, round_num in zip(
            tuning["live_update_lambda"], tuning["round_num"]
        )
    ]
    if pd.to_datetime(
        tuning.loc[tuning["selected"], "selection_max_event_date"], errors="coerce"
    ).gt(DISCOVERY_END).any():
        raise AssertionError("Live-update shrinkage selection reached sealed outcomes")
    model = f"C2_discovery_live_update_shrink__{candidate}"
    output = _live_lambda_player_predictions(
        candidate_frame,
        candidate=candidate,
        lambdas_by_round=chosen,
        model=model,
        variant="C2_DISCOVERY_LIVE_UPDATE_SHRINK",
    )
    return output, tuning


def calibrate_and_score_pairs(
    player_predictions: pd.DataFrame, quotes: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = player_predictions_to_pairs(player_predictions, quotes)
    sigma_rows: list[dict[str, Any]] = []
    scored_pieces: list[pd.DataFrame] = []
    for (model, round_num), group in pairs.groupby(
        ["model", "round_num"], observed=True
    ):
        discovery = group.loc[_sample_mask(group, "discovery")]
        sigma, tie_threshold, loss, resolved, events = fit_sigma_and_loss(discovery)
        sigma_rows.append(
            {
                "model": model,
                "candidate": group["candidate"].iloc[0],
                "variant": group["variant"].iloc[0],
                "round_num": int(round_num),
                "sigma": sigma,
                "tie_threshold": tie_threshold,
                "discovery_event_equal_logloss": loss,
                "discovery_resolved_pairs": resolved,
                "discovery_events": events,
                "discovery_min_event_date": discovery["event_start_date"].min(),
                "discovery_max_event_date": discovery["event_start_date"].max(),
            }
        )
        if not np.isfinite(sigma) or not np.isfinite(tie_threshold):
            continue
        piece = group.copy()
        piece["sigma"] = sigma
        piece["tie_threshold"] = tie_threshold
        p1_win, p2_win, push = discrete_round_probabilities(
            piece["skill_difference"].to_numpy(float), sigma, tie_threshold
        )
        piece["p1_probability"] = p1_win
        piece["p2_probability"] = p2_win
        piece["push_probability"] = push
        resolved_mass = np.clip(1.0 - push, EPS, 1.0)
        piece["tie_rule_ties_lose"] = piece.get(
            "tie_rule", pd.Series("", index=piece.index)
        ).map(ties_lose)
        piece["pricing_p1_probability"] = np.where(
            piece["tie_rule_ties_lose"], p1_win, p1_win / resolved_mass
        )
        piece["market_devig_probability_scope"] = np.where(
            piece["tie_rule_ties_lose"],
            "conditional_on_either_player_winning_no_draw_quote_attribution_only",
            "resolved_two_way_push_on_tie_market_benchmark",
        )
        for price_point in PRICE_POINTS:
            p1_odds = f"p1_{price_point}"
            p2_odds = f"p2_{price_point}"
            valid_price = _valid_american(piece[p1_odds]) & _valid_american(
                piece[p2_odds]
            )
            devig_column = f"p1_{price_point}_devig_probability"
            gap_column = f"model_minus_devig_{price_point}_pp"
            piece[devig_column] = np.nan
            piece[gap_column] = np.nan
            implied1 = np.asarray(
                american_implied_probability(piece.loc[valid_price, p1_odds]),
                dtype=float,
            )
            implied2 = np.asarray(
                american_implied_probability(piece.loc[valid_price, p2_odds]),
                dtype=float,
            )
            piece.loc[valid_price, devig_column] = implied1 / (implied1 + implied2)
            comparable_price = valid_price & piece[
                "market_devig_proper_score_eligible"
            ].fillna(False).astype(bool)
            piece.loc[comparable_price, gap_column] = 100.0 * (
                piece.loc[comparable_price, "pricing_p1_probability"]
                - piece.loc[comparable_price, devig_column]
            )
        scored_pieces.append(piece)
    scored = pd.concat(scored_pieces, ignore_index=True) if scored_pieces else pd.DataFrame()
    return scored, pd.DataFrame(sigma_rows)


def audit_pricing_probability_algebra(scored: pd.DataFrame) -> dict[str, Any]:
    """Fail closed on simplex, bounds, and tie-rule pricing identities."""

    required = {
        "p1_probability",
        "p2_probability",
        "push_probability",
        "pricing_p1_probability",
        "tie_rule_ties_lose",
    }
    _require(scored, required, "round matchup probability algebra")
    numeric = scored[
        [
            "p1_probability",
            "p2_probability",
            "push_probability",
            "pricing_p1_probability",
        ]
    ].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or (~np.isfinite(numeric)).any().any():
        raise ValueError("round matchup probability algebra contains nonfinite values")
    bounds_bad = ((numeric < -1e-12) | (numeric > 1.0 + 1e-12)).any(axis=1)
    if bounds_bad.any():
        raise ValueError("round matchup probability algebra violates [0,1] bounds")
    simplex_error = (
        numeric["p1_probability"]
        + numeric["p2_probability"]
        + numeric["push_probability"]
        - 1.0
    ).abs()
    ties_loss = scored["tie_rule_ties_lose"].fillna(False).astype(bool)
    nonpush_mass = 1.0 - numeric["push_probability"]
    invalid_nonpush = ~ties_loss & nonpush_mass.le(EPS)
    if invalid_nonpush.any():
        raise ValueError("push-on-tie pricing has no non-push probability mass")
    expected = numeric["p1_probability"].copy()
    push_on_tie = ~ties_loss
    expected.loc[push_on_tie] = (
        numeric.loc[push_on_tie, "p1_probability"]
        / nonpush_mass.loc[push_on_tie]
    )
    algebra_error = (numeric["pricing_p1_probability"] - expected).abs()
    if simplex_error.max() > 1e-12 or algebra_error.max() > 1e-12:
        raise AssertionError(
            "round matchup probabilities fail simplex or tie-rule pricing algebra"
        )
    zero_push = numeric["push_probability"].abs().le(1e-15)
    zero_push_error = (
        numeric.loc[zero_push, "pricing_p1_probability"]
        - numeric.loc[zero_push, "p1_probability"]
    ).abs()
    return {
        "rows": int(len(scored)),
        "ties_lose_rows": int(ties_loss.sum()),
        "push_on_tie_rows": int((~ties_loss).sum()),
        "zero_push_rows": int(zero_push.sum()),
        "simplex_max_abs_error": float(simplex_error.max()),
        "pricing_identity_max_abs_error": float(algebra_error.max()),
        "zero_push_identity_max_abs_error": float(
            zero_push_error.max() if len(zero_push_error) else 0.0
        ),
        "ties_lose_identity": "pricing_p1_probability_equals_unconditional_p1_win",
        "push_on_tie_identity": (
            "pricing_p1_probability_equals_p1_win_divided_by_one_minus_push"
        ),
    }


def _proper_group(group: pd.DataFrame, price_point: str) -> dict[str, Any]:
    graded = group.loc[
        group["score_model_eligible"].fillna(False).astype(bool)
    ].copy()
    if graded.empty:
        return {}
    resolved = graded.loc[
        graded["market_devig_proper_score_eligible"].fillna(False).astype(bool)
    ].copy()
    outcome = graded["derived_score_outcome"].to_numpy(float)
    p1 = np.clip(graded["p1_probability"].to_numpy(float), EPS, 1.0)
    p2 = np.clip(graded["p2_probability"].to_numpy(float), EPS, 1.0)
    push = np.clip(graded["push_probability"].to_numpy(float), EPS, 1.0)
    chosen_probability = np.where(
        np.isclose(outcome, 1.0),
        p1,
        np.where(np.isclose(outcome, 0.0), p2, push),
    )
    graded["three_way_logloss"] = -np.log(chosen_probability)
    actual_p1 = np.isclose(outcome, 1.0).astype(float)
    actual_p2 = np.isclose(outcome, 0.0).astype(float)
    actual_push = np.isclose(outcome, 0.5).astype(float)
    graded["three_way_brier"] = (
        np.square(p1 - actual_p1)
        + np.square(p2 - actual_p2)
        + np.square(push - actual_push)
    )
    three_way_event = graded.groupby("event_key", observed=True)[
        ["three_way_logloss", "three_way_brier"]
    ].mean()
    output = {
        "graded_quotes_including_pushes": int(len(graded)),
        "pushes": int(np.isclose(outcome, 0.5).sum()),
        "observed_push_rate": float(np.isclose(outcome, 0.5).mean()),
        "mean_predicted_push_rate": float(graded["push_probability"].mean()),
        "event_equal_three_way_logloss": float(three_way_event["three_way_logloss"].mean()),
        "event_equal_three_way_brier": float(three_way_event["three_way_brier"].mean()),
        "ties_lose_rows_excluded_from_market_benchmark": int(
            graded["tie_rule_ties_lose"].fillna(False).astype(bool).sum()
        ),
        "market_devig_probability_scope": (
            "resolved_push_on_tie_two_way_only; ties_lose_without_draw_quote_excluded"
        ),
        "resolved_quotes": int(len(resolved)),
    }
    if resolved.empty:
        return {
            **output,
            "events": 0,
            "event_equal_logloss": np.nan,
            "event_equal_devig_price_logloss": np.nan,
            "logloss_delta_vs_price": np.nan,
            "event_equal_brier": np.nan,
            "event_equal_devig_price_brier": np.nan,
            "brier_delta_vs_price": np.nan,
            "mean_abs_model_vs_devig_price_pp": np.nan,
        }
    y = resolved["settlement_score_outcome"].to_numpy(float)
    p = np.clip(
        resolved["pricing_p1_probability"].to_numpy(float), EPS, 1.0 - EPS
    )
    market_p = np.clip(
        resolved[f"p1_{price_point}_devig_probability"].to_numpy(float),
        EPS,
        1.0 - EPS,
    )
    resolved["model_logloss"] = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    resolved["market_logloss"] = -(
        y * np.log(market_p) + (1.0 - y) * np.log(1.0 - market_p)
    )
    resolved["brier"] = np.square(p - y)
    resolved["market_brier"] = np.square(market_p - y)
    event = resolved.groupby("event_key", observed=True)[
        ["model_logloss", "market_logloss", "brier", "market_brier"]
    ].mean()
    return {
        **output,
        "events": int(event.shape[0]),
        "event_equal_logloss": float(event["model_logloss"].mean()),
        "event_equal_devig_price_logloss": float(event["market_logloss"].mean()),
        "logloss_delta_vs_price": float(
            (event["model_logloss"] - event["market_logloss"]).mean()
        ),
        "event_equal_brier": float(event["brier"].mean()),
        "event_equal_devig_price_brier": float(event["market_brier"].mean()),
        "brier_delta_vs_price": float((event["brier"] - event["market_brier"]).mean()),
        "mean_abs_model_vs_devig_price_pp": float(
            resolved[f"model_minus_devig_{price_point}_pp"].abs().mean()
        ),
    }


def _calibration_group(group: pd.DataFrame) -> dict[str, Any]:
    resolved = group.loc[
        group["market_devig_proper_score_eligible"].fillna(False).astype(bool)
    ].copy()
    if len(resolved) < 50:
        return {}
    y = resolved["settlement_score_outcome"].to_numpy(float)
    p = np.clip(
        resolved["pricing_p1_probability"].to_numpy(float), EPS, 1.0 - EPS
    )
    x = logit(p)
    weights = _event_equal_weights(resolved)

    def objective(parameters: np.ndarray) -> float:
        fitted = np.clip(expit(parameters[0] + parameters[1] * x), EPS, 1.0 - EPS)
        loss = -(y * np.log(fitted) + (1.0 - y) * np.log(1.0 - fitted))
        return float(np.sum(weights * loss) / np.sum(weights))

    fit = minimize(objective, np.array([0.0, 1.0]), method="BFGS")
    bins = pd.qcut(resolved["pricing_p1_probability"], q=10, duplicates="drop")
    calibration = resolved.assign(_bin=bins).groupby("_bin", observed=True).agg(
        predicted=("pricing_p1_probability", "mean"),
        actual=("settlement_score_outcome", "mean"),
        n=("quote_id", "size"),
    )
    ece = float(
        np.average(
            np.abs(calibration["predicted"] - calibration["actual"]),
            weights=calibration["n"],
        )
    )
    return {
        "calibration_intercept": float(fit.x[0]) if fit.success else np.nan,
        "calibration_slope": float(fit.x[1]) if fit.success else np.nan,
        "ece_decile": ece,
        "fit_success": bool(fit.success),
    }


def summarize_proper_scores(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    proper_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    for sample in ("discovery", "sealed", "comparable_3y"):
        sample_frame = scored.loc[_sample_mask(scored, sample)].copy()
        for price_point in PRICE_POINTS:
            valid_price = _valid_american(
                sample_frame[f"p1_{price_point}"]
            ) & _valid_american(sample_frame[f"p2_{price_point}"])
            priced = sample_frame.loc[valid_price].copy()
            policies: list[tuple[str, pd.DataFrame]] = [
                (book, priced.loc[priced["book"].eq(book)].copy())
                for book in BOOKS
            ]
            policies.append(
                ("best_of_three", _best_price_market_rows(priced, price_point))
            )
            for line_policy, policy_frame in policies:
                if policy_frame.empty:
                    continue
                for model, model_frame in policy_frame.groupby("model", observed=True):
                    for round_label, group in [
                        *[
                            (
                                f"r{round_num}",
                                model_frame.loc[
                                    model_frame["round_num"].eq(round_num)
                                ],
                            )
                            for round_num in range(1, 5)
                        ],
                        ("pooled", model_frame),
                    ]:
                        if group.empty:
                            continue
                        common = {
                            "sample": sample,
                            "model": model,
                            "candidate": group["candidate"].iloc[0],
                            "variant": group["variant"].iloc[0],
                            "round_market": round_label,
                            "line_policy": line_policy,
                            "price_point": price_point,
                        }
                        proper = _proper_group(group, price_point)
                        if proper:
                            proper_rows.append({**common, **proper})
                        calibration = _calibration_group(group)
                        if calibration:
                            calibration_rows.append({**common, **calibration})
    return pd.DataFrame(proper_rows), pd.DataFrame(calibration_rows)


def expand_betting_lines(scored: pd.DataFrame, price_point: str) -> pd.DataFrame:
    if price_point not in PRICE_POINTS:
        raise ValueError(price_point)
    rows: list[dict[str, Any]] = []
    for row in scored.to_dict(orient="records"):
        if not (
            bool(_valid_american(pd.Series([row[f"p1_{price_point}"]])).iloc[0])
            and bool(_valid_american(pd.Series([row[f"p2_{price_point}"]])).iloc[0])
        ):
            continue
        for side, other, probability, loss_probability in (
            ("p1", "p2", row["p1_probability"], row["p2_probability"]),
            ("p2", "p1", row["p2_probability"], row["p1_probability"]),
        ):
            execution_odds = float(row[f"{side}_{price_point}"])
            decimal = float(american_decimal_odds(execution_odds))
            implied = 1.0 / decimal
            close_devig = (
                row[f"p1_{price_point}_devig_probability"]
                if side == "p1"
                else 1.0 - row[f"p1_{price_point}_devig_probability"]
            )
            push_probability = float(row["push_probability"])
            tie_is_loss = ties_lose(row.get("tie_rule", ""))
            effective_push_probability = 0.0 if tie_is_loss else push_probability
            loss_probability = float(loss_probability) + (
                push_probability if tie_is_loss else 0.0
            )
            resolved_mass = float(probability) + float(loss_probability)
            pricing_probability = (
                float(probability) / resolved_mass if resolved_mass > 0 else np.nan
            )
            market_devig_price_comparison_eligible = not tie_is_loss
            model_minus_devig_price_pp = (
                100.0 * (pricing_probability - float(close_devig))
                if market_devig_price_comparison_eligible
                else np.nan
            )
            ev_per_unit = (
                float(probability) * (decimal - 1.0) - float(loss_probability)
            )
            full_kelly = (
                ev_per_unit / ((decimal - 1.0) * resolved_mass)
                if resolved_mass > 0 and decimal > 1.0
                else np.nan
            )
            archived_numeric_outcome = row.get(f"{side}_outcome", np.nan)
            no_score_void = row.get("no_score_void", False)
            other_void = row.get("other_void_with_scores", False)
            actual_tie = row.get("actual_score_tie", False)
            if (
                pd.notna(no_score_void)
                and bool(no_score_void)
            ) or (pd.notna(other_void) and bool(other_void)):
                settlement_outcome_text = "void"
            elif pd.notna(actual_tie) and bool(actual_tie):
                settlement_outcome_text = "tie"
            elif pd.notna(archived_numeric_outcome) and np.isclose(
                float(archived_numeric_outcome), 1.0
            ):
                settlement_outcome_text = "win"
            elif pd.notna(archived_numeric_outcome) and np.isclose(
                float(archived_numeric_outcome), 0.0
            ):
                settlement_outcome_text = "loss"
            else:
                settlement_outcome_text = row.get(f"{side}_outcome_text")
            rows.append(
                {
                    "model": row["model"],
                    "candidate": row["candidate"],
                    "variant": row["variant"],
                    "strict_point_in_time": row.get("strict_point_in_time", False),
                    "retrospective_cascade_anchor": row.get(
                        "retrospective_cascade_anchor", False
                    ),
                    "event_key": row["event_key"],
                    "event_id": row["event_key"],
                    "event_name": row["event_name"],
                    "event_start_date": row["event_start_date"],
                    "round_market": row["round_market"],
                    "round_num": row["round_num"],
                    "pair_key": row["pair_key"],
                    "selection_id": row[f"{side}_id"],
                    "selection_name": row[f"{side}_player_name"],
                    "opponent_id": row[f"{other}_id"],
                    "opponent_name": row[f"{other}_player_name"],
                    "book": row["book"],
                    "price_point": price_point,
                    "execution_odds": execution_odds,
                    # settle_bets uses this historical name for the execution
                    # price. Preserve both archived endpoints below.
                    "close_odds": execution_odds,
                    "archived_open_odds": row.get(f"{side}_open", np.nan),
                    "archived_close_odds": row.get(f"{side}_close", np.nan),
                    "execution_time": row.get(f"{price_point}_time"),
                    "close_time": row.get(f"{price_point}_time"),
                    "tie_rule": row["tie_rule"],
                    "outcome_numeric": archived_numeric_outcome,
                    "outcome": settlement_outcome_text,
                    "probability": float(probability),
                    "loss_probability": float(loss_probability),
                    "push_probability": push_probability,
                    "effective_push_probability": effective_push_probability,
                    "tie_rule_ties_lose": tie_is_loss,
                    "pricing_probability": pricing_probability,
                    "implied_probability": implied,
                    "close_devig_probability": float(close_devig),
                    "close_devig_probability_role": (
                        "resolved_two_way_market_benchmark"
                        if market_devig_price_comparison_eligible
                        else "conditional_on_either_player_winning_attribution_only_no_draw_quote"
                    ),
                    "market_devig_price_comparison_eligible": (
                        market_devig_price_comparison_eligible
                    ),
                    "market_devig_price_comparison_exclusion_reason": (
                        "" if market_devig_price_comparison_eligible
                        else "ties_lose_without_draw_quote"
                    ),
                    "model_minus_devig_price_pp": model_minus_devig_price_pp,
                    "ev_per_unit": ev_per_unit,
                    "full_kelly_fraction": full_kelly,
                    "selected_skill": row[f"{side}_skill"],
                    "opponent_skill": row[f"{other}_skill"],
                    "selected_base_core": row.get(f"{side}_base_core", np.nan),
                    "opponent_base_core": row.get(f"{other}_base_core", np.nan),
                    "abs_realized_wind_difference": row.get(
                        "abs_realized_wind_difference", np.nan
                    ),
                    "abs_realized_dew_difference": row.get(
                        "abs_realized_dew_difference", np.nan
                    ),
                    "pair_weather_observed": row.get(
                        "pair_weather_observed", False
                    ),
                    "realized_round_mean_wind": row.get(
                        "realized_round_mean_wind", np.nan
                    ),
                    "realized_round_field_mean_wind": row.get(
                        "realized_round_field_mean_wind", np.nan
                    ),
                    "selected_history_rounds": row.get(
                        f"{side}_history_rounds", np.nan
                    ),
                    "opponent_history_rounds": row.get(
                        f"{other}_history_rounds", np.nan
                    ),
                    "selected_img_history_rounds": row.get(
                        f"{side}_img_history_rounds", np.nan
                    ),
                    "opponent_img_history_rounds": row.get(
                        f"{other}_img_history_rounds", np.nan
                    ),
                    "selected_img_history_coverage": row.get(
                        f"{side}_img_history_coverage", np.nan
                    ),
                    "opponent_img_history_coverage": row.get(
                        f"{other}_img_history_coverage", np.nan
                    ),
                    "selected_reliability_ok": row.get(
                        f"{side}_reliability_ok", np.nan
                    ),
                    "opponent_reliability_ok": row.get(
                        f"{other}_reliability_ok", np.nan
                    ),
                    "selected_cross_tour_fallback_reason": row.get(
                        f"{side}_cross_tour_fallback_reason", np.nan
                    ),
                    "opponent_cross_tour_fallback_reason": row.get(
                        f"{other}_cross_tour_fallback_reason", np.nan
                    ),
                    "selected_prior_event_withdrawal": row.get(
                        f"{side}_prior_event_withdrawal", np.nan
                    ),
                    "opponent_prior_event_withdrawal": row.get(
                        f"{other}_prior_event_withdrawal", np.nan
                    ),
                    "selected_event_withdrawal": row.get(
                        f"{side}_event_withdrawal", False
                    ),
                    "opponent_event_withdrawal": row.get(
                        f"{other}_event_withdrawal", False
                    ),
                    # Reuse the independently tested two-way matchup settlement
                    # contract. ``round_market`` retains the R1-R4 identity.
                    "market": "tournament_matchup",
                }
            )
    return pd.DataFrame(rows)


def select_and_settle(
    lines: pd.DataFrame, *, line_policy: str, sizing: str, min_ev: float = EV_GATE
) -> pd.DataFrame:
    keys = [
        "model",
        "event_key",
        "round_market",
        "selection_id",
        "opponent_id",
    ]
    if line_policy in BOOKS:
        chosen = lines.loc[lines["book"].eq(line_policy)].copy()
        chosen = chosen.sort_values(
            keys + ["close_time"], ascending=[True] * len(keys) + [False], kind="mergesort"
        ).drop_duplicates(keys, keep="first")
    elif line_policy == "best_of_three":
        chosen = lines.sort_values(
            keys + ["close_odds", "book"],
            ascending=[True] * len(keys) + [False, True],
            kind="mergesort",
        ).drop_duplicates(keys, keep="first")
    else:
        raise ValueError(line_policy)
    chosen = chosen.loc[chosen["ev_per_unit"].ge(min_ev)].copy()
    direction_keys = ["model", "event_key", "round_market", "pair_key"]
    chosen = chosen.sort_values(
        direction_keys + ["ev_per_unit", "close_odds", "book"],
        ascending=[True] * len(direction_keys) + [False, False, True],
        kind="mergesort",
    ).drop_duplicates(direction_keys, keep="first")
    if sizing == "flat_1u":
        chosen["stake"] = 1.0
    elif sizing == "production_risk_to_win":
        chosen["stake"] = np.where(
            chosen["close_odds"].lt(0), chosen["close_odds"].abs() / 100.0, 1.0
        )
    else:
        raise ValueError(sizing)
    chosen["line_policy"] = line_policy
    chosen["sizing"] = sizing
    return settle_bets(chosen)


def _max_drawdown(bets: pd.DataFrame) -> float:
    if bets.empty:
        return 0.0
    event = (
        bets.groupby(["event_start_date", "event_key"], observed=True)["pnl"]
        .sum()
        .sort_index()
    )
    cumulative = event.cumsum().to_numpy(float)
    peak = np.maximum.accumulate(np.r_[0.0, cumulative])[:-1]
    return float(np.max(peak - cumulative)) if len(cumulative) else 0.0


def eligible_event_universe(
    scored: pd.DataFrame,
    sample: str,
    model: str,
    line_policy: str,
    round_label: str,
    price_point: str,
) -> list[str]:
    use = scored.loc[_sample_mask(scored, sample) & scored["model"].eq(model)].copy()
    use = use.loc[
        _valid_american(use[f"p1_{price_point}"])
        & _valid_american(use[f"p2_{price_point}"])
    ]
    if line_policy in BOOKS:
        use = use.loc[use["book"].eq(line_policy)]
    if round_label != "pooled":
        use = use.loc[use["round_market"].eq(round_label)]
    return sorted(use["event_key"].dropna().astype(str).unique())


def summarize_bets(bets: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    model_metadata = (
        scored[["model", "candidate", "variant"]]
        .drop_duplicates("model")
        .set_index("model")
    )
    for sample in ("discovery", "sealed", "comparable_3y"):
        sample_bets = (
            bets.loc[_sample_mask(bets, sample)].copy()
            if not bets.empty
            else bets.copy()
        )
        for model, metadata in model_metadata.iterrows():
            for price_point in PRICE_POINTS:
                for policy in (*BOOKS, "best_of_three"):
                    for sizing in ("flat_1u", "production_risk_to_win"):
                        group = sample_bets.loc[
                            sample_bets["model"].eq(model)
                            & sample_bets["price_point"].eq(price_point)
                            & sample_bets["line_policy"].eq(policy)
                            & sample_bets["sizing"].eq(sizing)
                        ].copy()
                        for round_label, slice_frame in [
                            *[
                                (
                                    f"r{round_num}",
                                    group.loc[group["round_num"].eq(round_num)],
                                )
                                for round_num in range(1, 5)
                            ],
                            ("pooled", group),
                        ]:
                            universe = eligible_event_universe(
                                scored,
                                sample,
                                model,
                                policy,
                                round_label,
                                price_point,
                            )
                            if not universe:
                                continue
                            stake = float(slice_frame["stake"].sum())
                            pnl = float(slice_frame["pnl"].sum())
                            comparable_price = slice_frame.loc[
                                slice_frame[
                                    "market_devig_price_comparison_eligible"
                                ]
                                .fillna(False)
                                .astype(bool)
                                & slice_frame["model_minus_devig_price_pp"].notna()
                            ]
                            rows.append(
                                {
                                    "sample": sample,
                                    "model": model,
                                    "candidate": metadata["candidate"],
                                    "variant": metadata["variant"],
                                    "round_market": round_label,
                                    "line_policy": policy,
                                    "price_point": price_point,
                                    "sizing": sizing,
                                    "eligible_events_including_zero_bet": int(
                                        len(universe)
                                    ),
                                    "bet_events": int(
                                        slice_frame["event_key"].nunique()
                                    ),
                                    "bets": int(len(slice_frame)),
                                    "stake": stake,
                                    "pnl": pnl,
                                    "roi": pnl / stake if stake else np.nan,
                                    "win_rate_resolved": float(
                                        slice_frame["paid_fraction"].dropna().mean()
                                    )
                                    if len(slice_frame)
                                    else np.nan,
                                    "max_drawdown": _max_drawdown(slice_frame),
                                    "mean_ev_at_bet": float(
                                        slice_frame["ev_per_unit"].mean()
                                    )
                                    if len(slice_frame)
                                    else np.nan,
                                    "mean_model_minus_devig_price_pp": float(
                                        comparable_price[
                                            "model_minus_devig_price_pp"
                                        ].mean()
                                    )
                                    if len(comparable_price)
                                    else np.nan,
                                    "market_devig_price_comparable_bets": int(
                                        len(comparable_price)
                                    ),
                                    "market_devig_price_excluded_ties_lose_bets": int(
                                        (
                                            slice_frame[
                                                "market_devig_price_comparison_exclusion_reason"
                                            ]
                                            .fillna("")
                                            .eq("ties_lose_without_draw_quote")
                                        ).sum()
                                    ),
                                    "market_devig_price_comparison_scope": (
                                        "push_on_tie_resolved_two_way_only; ties_lose_without_draw_quote_excluded"
                                    ),
                                }
                            )
    return pd.DataFrame(rows)


def _event_portfolio(
    bets: pd.DataFrame,
    universe: Sequence[str],
    *,
    model: str,
    line_policy: str,
    sizing: str,
    round_label: str,
    sample: str,
    price_point: str,
) -> pd.DataFrame:
    use = bets.loc[
        _sample_mask(bets, sample)
        & bets["model"].eq(model)
        & bets["line_policy"].eq(line_policy)
        & bets["sizing"].eq(sizing)
        & bets["price_point"].eq(price_point)
    ].copy()
    if round_label != "pooled":
        use = use.loc[use["round_market"].eq(round_label)]
    event = use.groupby("event_key", observed=True)[["stake", "pnl"]].sum()
    return event.reindex(list(universe), fill_value=0.0)


def event_cluster_bootstrap(
    performance: pd.DataFrame,
    bets: pd.DataFrame,
    scored: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    eligible = performance.loc[
        performance["sample"].isin(["sealed", "comparable_3y"])
    ].copy()
    for row in eligible.to_dict(orient="records"):
        universe = eligible_event_universe(
            scored,
            row["sample"],
            row["model"],
            row["line_policy"],
            row["round_market"],
            row["price_point"],
        )
        if not universe:
            continue
        event = _event_portfolio(
            bets,
            universe,
            model=row["model"],
            line_policy=row["line_policy"],
            sizing=row["sizing"],
            round_label=row["round_market"],
            sample=row["sample"],
            price_point=row["price_point"],
        )
        indices = rng.integers(0, len(event), size=(draws, len(event)))
        pnl_draw = event["pnl"].to_numpy(float)[indices].sum(axis=1)
        stake_draw = event["stake"].to_numpy(float)[indices].sum(axis=1)
        roi_draw = np.divide(
            pnl_draw,
            stake_draw,
            out=np.full(draws, np.nan),
            where=stake_draw > 0,
        )
        record = {
            "sample": row["sample"],
            "model": row["model"],
            "candidate": row["candidate"],
            "variant": row["variant"],
            "round_market": row["round_market"],
            "line_policy": row["line_policy"],
            "price_point": row["price_point"],
            "sizing": row["sizing"],
            "eligible_events_including_zero_bet": int(len(event)),
            "draws": int(draws),
            "pnl_estimate": float(event["pnl"].sum()),
            "pnl_ci_low": float(np.quantile(pnl_draw, 0.025)),
            "pnl_ci_high": float(np.quantile(pnl_draw, 0.975)),
            "probability_pnl_positive": float(np.mean(pnl_draw > 0)),
            "roi_estimate": float(event["pnl"].sum() / event["stake"].sum())
            if event["stake"].sum()
            else np.nan,
            "roi_ci_low": float(np.nanquantile(roi_draw, 0.025))
            if np.isfinite(roi_draw).any()
            else np.nan,
            "roi_ci_high": float(np.nanquantile(roi_draw, 0.975))
            if np.isfinite(roi_draw).any()
            else np.nan,
        }
        baseline_universe = eligible_event_universe(
            scored,
            row["sample"],
            C0_MODEL,
            row["line_policy"],
            row["round_market"],
            row["price_point"],
        )
        common = sorted(set(universe).intersection(baseline_universe))
        if row["model"] != C0_MODEL and common:
            challenger = event.reindex(common, fill_value=0.0)
            baseline = _event_portfolio(
                bets,
                common,
                model=C0_MODEL,
                line_policy=row["line_policy"],
                sizing=row["sizing"],
                round_label=row["round_market"],
                sample=row["sample"],
                price_point=row["price_point"],
            )
            delta = challenger["pnl"].to_numpy(float) - baseline["pnl"].to_numpy(float)
            delta_indices = rng.integers(0, len(common), size=(draws, len(common)))
            delta_draw = delta[delta_indices].sum(axis=1)
            record.update(
                {
                    "paired_c0_events": int(len(common)),
                    "pnl_delta_vs_c0": float(delta.sum()),
                    "pnl_delta_vs_c0_ci_low": float(np.quantile(delta_draw, 0.025)),
                    "pnl_delta_vs_c0_ci_high": float(np.quantile(delta_draw, 0.975)),
                    "probability_pnl_delta_vs_c0_positive": float(
                        np.mean(delta_draw > 0)
                    ),
                }
            )
        rows.append(record)
    return pd.DataFrame(rows)


def bucket_bet_performance(bets: pd.DataFrame) -> pd.DataFrame:
    if bets.empty:
        return pd.DataFrame()
    frame = bets.copy()
    frame["year"] = pd.to_datetime(frame["event_start_date"]).dt.year.astype(str)
    frame["edge_bucket"] = pd.cut(
        frame["ev_per_unit"],
        bins=[-np.inf, 0.0, 0.03, 0.05, 0.10, np.inf],
        labels=["<0%", "0-3%", "3-5%", "5-10%", ">10%"],
        right=False,
    ).astype(str)
    frame["skill_bucket"] = pd.cut(
        frame["selected_base_core"],
        bins=[-np.inf, 0.0, 0.5, 1.0, 1.5, np.inf],
        labels=["<0", "0-0.5", "0.5-1.0", "1.0-1.5", ">=1.5"],
        right=False,
    ).astype(str)
    frame["skill_gap_bucket"] = pd.cut(
        frame["selected_base_core"] - frame["opponent_base_core"],
        bins=[-np.inf, -0.5, -0.2, 0.0, 0.2, 0.5, np.inf],
        labels=["<-0.5", "-0.5--0.2", "-0.2-0", "0-0.2", "0.2-0.5", ">=0.5"],
        right=False,
    ).astype(str)
    frame["wind_gap_bucket"] = pd.cut(
        frame["abs_realized_wind_difference"],
        bins=[-np.inf, 1.0, 2.0, 4.0, np.inf],
        labels=["<1mph", "1-2mph", "2-4mph", ">=4mph"],
        right=False,
    ).astype(str)
    frame["dew_gap_bucket"] = pd.cut(
        frame["abs_realized_dew_difference"],
        bins=[-np.inf, 2.0, 5.0, 10.0, np.inf],
        labels=["<2F", "2-5F", "5-10F", ">=10F"],
        right=False,
    ).astype(str)
    frame["wind_level_bucket"] = pd.cut(
        pd.to_numeric(frame["realized_round_field_mean_wind"], errors="coerce"),
        bins=[-np.inf, 5.0, 10.0, 15.0, np.inf],
        labels=["<5mph", "5-10mph", "10-15mph", ">=15mph"],
        right=False,
    ).astype(str)
    frame["history_coverage_bucket"] = pd.cut(
        pd.to_numeric(frame["selected_history_rounds"], errors="coerce"),
        bins=[-np.inf, 5.0, 10.0, 20.0, 50.0, np.inf],
        labels=["<5", "5-9", "10-19", "20-49", ">=50"],
        right=False,
    ).astype(str)
    frame["img_coverage_bucket"] = pd.cut(
        pd.to_numeric(frame["selected_img_history_coverage"], errors="coerce"),
        bins=[-np.inf, 0.01, 0.25, 0.50, 0.75, np.inf],
        labels=["none", "<25%", "25-50%", "50-75%", ">=75%"],
        right=False,
    ).astype(str)
    for source, target in (
        ("selected_reliability_ok", "reliability_bucket"),
        ("selected_prior_event_withdrawal", "prior_withdrawal_bucket"),
        ("selected_event_withdrawal", "target_event_withdrawal_bucket"),
    ):
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

        numeric_bool = frame[source].map(
            _bool_bucket
        )
        frame[target] = numeric_bool
    frame["fallback_reason_bucket"] = frame[
        "selected_cross_tour_fallback_reason"
    ].fillna("unknown").astype(str)
    dimensions = {
        "edge": "edge_bucket",
        "projected_skill": "skill_bucket",
        "projected_skill_gap": "skill_gap_bucket",
        "posthoc_realized_wind_gap": "wind_gap_bucket",
        "posthoc_realized_dew_gap": "dew_gap_bucket",
        "posthoc_realized_wind_level": "wind_level_bucket",
        "history_round_coverage": "history_coverage_bucket",
        "img_history_coverage": "img_coverage_bucket",
        "reliability": "reliability_bucket",
        "cross_tour_fallback_reason": "fallback_reason_bucket",
        "prior_event_withdrawal": "prior_withdrawal_bucket",
        "target_event_withdrawal": "target_event_withdrawal_bucket",
        "year": "year",
        "selected_book": "book",
        "round": "round_market",
    }
    rows: list[dict[str, Any]] = []
    for sample in ("discovery", "sealed", "comparable_3y"):
        sample_frame = frame.loc[_sample_mask(frame, sample)]
        for dimension, column in dimensions.items():
            for keys, group in sample_frame.groupby(
                ["model", "price_point", "line_policy", "sizing", column],
                observed=True,
            ):
                model, price_point, policy, sizing, bucket = keys
                stake = float(group["stake"].sum())
                pnl = float(group["pnl"].sum())
                comparable_price = group.loc[
                    group["market_devig_price_comparison_eligible"]
                    .fillna(False)
                    .astype(bool)
                    & group["model_minus_devig_price_pp"].notna()
                ]
                rows.append(
                    {
                        "sample": sample,
                        "model": model,
                        "candidate": group["candidate"].iloc[0],
                        "variant": group["variant"].iloc[0],
                        "line_policy": policy,
                        "price_point": price_point,
                        "sizing": sizing,
                        "bucket_dimension": dimension,
                        "bucket": str(bucket),
                        "bets": int(len(group)),
                        "events": int(group["event_key"].nunique()),
                        "stake": stake,
                        "pnl": pnl,
                        "roi": pnl / stake if stake else np.nan,
                        "mean_ev_at_bet": float(group["ev_per_unit"].mean()),
                        "mean_model_minus_devig_price_pp": float(
                            comparable_price["model_minus_devig_price_pp"].mean()
                        )
                        if len(comparable_price)
                        else np.nan,
                        "market_devig_price_comparable_bets": int(
                            len(comparable_price)
                        ),
                        "market_devig_price_excluded_ties_lose_bets": int(
                            (
                                group[
                                    "market_devig_price_comparison_exclusion_reason"
                                ]
                                .fillna("")
                                .eq("ties_lose_without_draw_quote")
                            ).sum()
                        ),
                        "market_devig_price_comparison_scope": (
                            "push_on_tie_resolved_two_way_only; ties_lose_without_draw_quote_excluded"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def all_offered_edge_bets(scored: pd.DataFrame) -> pd.DataFrame:
    """Settle one canonical direction for every offered pair, including <0 EV."""

    pieces: list[pd.DataFrame] = []
    for model, model_scored in scored.groupby("model", observed=True):
        for price_point in PRICE_POINTS:
            lines = expand_betting_lines(model_scored, price_point)
            if lines.empty:
                continue
            for policy in (*BOOKS, "best_of_three"):
                settled = select_and_settle(
                    lines,
                    line_policy=policy,
                    sizing="flat_1u",
                    min_ev=-np.inf,
                )
                if not settled.empty:
                    pieces.append(settled)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def summarize_all_edge_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    if (~frame["is_graded"].fillna(False).astype(bool)).any():
        raise ValueError("all-offered edge buckets contain ungraded settlements")
    frame = frame.copy()
    frame["edge_bucket"] = pd.cut(
        frame["ev_per_unit"],
        bins=[-np.inf, 0.0, 0.03, 0.05, 0.10, np.inf],
        labels=["<0%", "0-3%", "3-5%", "5-10%", ">10%"],
        right=False,
    )
    rows: list[dict[str, Any]] = []
    for sample in ("discovery", "sealed", "comparable_3y"):
        sample_frame = frame.loc[_sample_mask(frame, sample)]
        for model, model_frame in sample_frame.groupby("model", observed=True):
            for round_label, round_frame in [
                *[
                    (
                        f"r{round_num}",
                        model_frame.loc[model_frame["round_num"].eq(round_num)],
                    )
                    for round_num in range(1, 5)
                ],
                ("pooled", model_frame),
            ]:
                for keys, group in round_frame.groupby(
                    ["price_point", "line_policy", "edge_bucket"], observed=True
                ):
                    price_point, policy, bucket = keys
                    stake = float(group["stake"].sum())
                    pnl = float(group["pnl"].sum())
                    rows.append(
                        {
                            "sample": sample,
                            "model": model,
                            "candidate": group["candidate"].iloc[0],
                            "variant": group["variant"].iloc[0],
                            "round_market": round_label,
                            "price_point": price_point,
                            "line_policy": policy,
                            "edge_bucket": str(bucket),
                            "bets": int(len(group)),
                            "events": int(group["event_key"].nunique()),
                            "stake": stake,
                            "pnl": pnl,
                            "roi": pnl / stake if stake else np.nan,
                            "mean_model_ev": float(group["ev_per_unit"].mean()),
                            "mean_full_kelly_fraction": float(
                                group["full_kelly_fraction"].mean()
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def all_edge_bucket_performance(scored: pd.DataFrame) -> pd.DataFrame:
    """Compatibility wrapper for synthetic callers."""

    return summarize_all_edge_buckets(all_offered_edge_bets(scored))


def worst_weeks_and_golfers(bets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if bets.empty:
        return pd.DataFrame(), pd.DataFrame()
    frame = bets.loc[_sample_mask(bets, "comparable_3y")].copy()
    week_keys = [
        "model",
        "candidate",
        "variant",
        "price_point",
        "line_policy",
        "sizing",
        "event_key",
        "event_name",
        "event_start_date",
    ]
    weekly = frame.groupby(week_keys, observed=True).agg(
        bets=("pnl", "size"),
        stake=("stake", "sum"),
        pnl=("pnl", "sum"),
        rounds=("round_market", lambda values: "|".join(sorted(set(values)))),
        golfers=("selection_name", "nunique"),
        mean_abs_wind_gap=("abs_realized_wind_difference", "mean"),
        max_abs_wind_gap=("abs_realized_wind_difference", "max"),
        mean_abs_dew_gap=("abs_realized_dew_difference", "mean"),
        mean_realized_wind=("realized_round_mean_wind", "mean"),
        mean_field_wind=("realized_round_field_mean_wind", "mean"),
        prior_withdrawal_bets=("selected_prior_event_withdrawal", "sum"),
        target_withdrawal_bets=("selected_event_withdrawal", "sum"),
        mean_history_rounds=("selected_history_rounds", "mean"),
        mean_img_history_coverage=("selected_img_history_coverage", "mean"),
    ).reset_index()
    weekly["roi"] = weekly["pnl"] / weekly["stake"]
    weekly["loss_rank_within_model_policy"] = weekly.groupby(
        ["model", "price_point", "line_policy", "sizing"], observed=True
    )["pnl"].rank(method="first", ascending=True)
    weekly = weekly.sort_values(
        ["model", "price_point", "line_policy", "sizing", "pnl"], kind="mergesort"
    )
    golfer_keys = week_keys + ["selection_id", "selection_name"]
    golfers = frame.groupby(golfer_keys, observed=True).agg(
        bets=("pnl", "size"),
        stake=("stake", "sum"),
        pnl=("pnl", "sum"),
        base_core=("selected_base_core", "mean"),
        history_rounds=("selected_history_rounds", "mean"),
        img_history_coverage=("selected_img_history_coverage", "mean"),
        prior_event_withdrawal=("selected_prior_event_withdrawal", "max"),
        target_event_withdrawal=("selected_event_withdrawal", "max"),
        mean_abs_wind_gap=("abs_realized_wind_difference", "mean"),
        mean_abs_dew_gap=("abs_realized_dew_difference", "mean"),
    ).reset_index()
    golfers["roi"] = golfers["pnl"] / golfers["stake"]
    golfers["loss_rank_within_week"] = golfers.groupby(
        week_keys, observed=True
    )["pnl"].rank(method="first", ascending=True)
    return weekly, golfers.sort_values(week_keys + ["pnl"], kind="mergesort")


def write_report(
    output: Path,
    audit: Mapping[str, Any],
    selection: pd.DataFrame,
    proper: pd.DataFrame,
    performance: pd.DataFrame,
    bootstrap: pd.DataFrame,
    tuning: pd.DataFrame,
    coefficients: pd.DataFrame,
) -> None:
    lines = [
        "# Base-skill R1-R4 cascade market evaluation",
        "",
        "Research only. No production file, sheet, API, or deployment was changed.",
        "",
        "## Interpretation contract",
        "",
        "- **C0** is the exact current-code cascade reconstructed with current full-sample coefficients. It is a retrospective compatibility benchmark, not historical production.",
        "- **C1** starts from finalist_core plus the discovery-frozen share of the exact pre-event course/history downstream terms, then adds the full current completed-round update. It still inherits the retrospective C0 live term.",
        "- **C2-NO-LIVE** carries that discovery-frozen pre-event prediction through R1-R4 without reacting to the tournament.",
        "- **C2-SHRINK** multiplies the current live-stage update by a separate R1-R4 scalar selected only on 2021-22 offered-matchup outcomes. It remains retrospective because the current cascade term is retrospective.",
        "- **C3-PT** is the fair completed-state comparison: annual expanding, event-equal ridge fits a continuous SG target from only earlier event editions, the discovery-frozen pre-event base, and completed-round DG state.",
        "- **C3-HYBRID** lets ridge shrink the legacy stage term, including late OTT analogues, but remains a labelled retrospective sensitivity because that term comes from C0.",
        "- **W0** uses clean DG field-adjusted SG. **W2** uses coefficients frozen before 2021 to neutralize completed observations. Upcoming/target realized weather is never a feature.",
        "- All current IMG accounting arms are excluded: their total/putting transform and downstream APP/ARG endpoints were fitted/backfilled with future evidence.",
        "- Market hyperparameters and probability scales use only 2021-22 discovery. Results from 2023 onward are sealed; the rolling three-year view starts 2023-08-22.",
        f"- Frozen pre-event downstream lambda: {audit.get('frozen_pre_event_downstream_lambda')}. This applies only to course/history terms; live-update lambdas are separately selected completed-round scalars.",
        "- Opening and closing boards are evaluated separately. 'Model minus de-vig price' is a probability gap, not literal CLV.",
        "",
        "## Candidate eligibility",
        "",
    ]
    selected = selection.loc[selection["selected"].fillna(False)].copy()
    if not selected.empty:
        columns = [
            column
            for column in (
                "candidate",
                "candidate_type",
                "finalist_role",
                "promotion_gate_passed",
            )
            if column in selected
        ]
        lines.append(selected[columns].to_markdown(index=False))
    headline = performance.loc[
        performance["sample"].eq("comparable_3y")
        & performance["round_market"].eq("pooled")
        & performance["line_policy"].eq("best_of_three")
        & performance["sizing"].eq("production_risk_to_win")
    ].sort_values("roi", ascending=False)
    lines.extend(["", "## Rolling three-year best-price P&L", ""])
    if not headline.empty:
        columns = [
            "model",
            "price_point",
            "bets",
            "eligible_events_including_zero_bet",
            "pnl",
            "roi",
            "max_drawdown",
        ]
        lines.append(headline[columns].to_markdown(index=False, floatfmt=".4f"))
    proper_headline = proper.loc[
        proper["sample"].eq("comparable_3y")
        & proper["round_market"].eq("pooled")
        & proper["line_policy"].eq("best_of_three")
    ].sort_values("event_equal_logloss")
    lines.extend(["", "## Rolling three-year proper scores versus de-vig price", ""])
    if not proper_headline.empty:
        columns = [
            "model",
            "price_point",
            "resolved_quotes",
            "events",
            "event_equal_logloss",
            "event_equal_devig_price_logloss",
            "logloss_delta_vs_price",
            "event_equal_brier",
        ]
        lines.append(proper_headline[columns].to_markdown(index=False, floatfmt=".5f"))
    boot_headline = bootstrap.loc[
        bootstrap["sample"].eq("comparable_3y")
        & bootstrap["round_market"].eq("pooled")
        & bootstrap["line_policy"].eq("best_of_three")
        & bootstrap["sizing"].eq("production_risk_to_win")
    ].sort_values("pnl_delta_vs_c0", ascending=False, na_position="last")
    lines.extend(["", "## Whole-event uncertainty, including zero-bet events", ""])
    if not boot_headline.empty:
        columns = [
            "model",
            "price_point",
            "eligible_events_including_zero_bet",
            "pnl_estimate",
            "pnl_ci_low",
            "pnl_ci_high",
            "pnl_delta_vs_c0",
            "pnl_delta_vs_c0_ci_low",
            "pnl_delta_vs_c0_ci_high",
            "probability_pnl_delta_vs_c0_positive",
        ]
        columns = [column for column in columns if column in boot_headline]
        lines.append(boot_headline[columns].to_markdown(index=False, floatfmt=".3f"))

    lines.extend(["", "## Late OTT shrinkage audit", ""])
    if not coefficients.empty:
        ott = coefficients.loc[
            coefficients["round_num"].isin([3, 4])
            & coefficients["feature"].str.contains("ott", case=False, na=False)
            & coefficients["prediction_year"].eq(
                coefficients["prediction_year"].max()
            )
        ].copy()
        if not ott.empty:
            columns = [
                "model",
                "round_num",
                "feature",
                "standardized_coefficient",
                "raw_scale_coefficient",
            ]
            lines.append(ott[columns].to_markdown(index=False, floatfmt=".5f"))
        else:
            lines.append("No eligible late-round OTT coefficient was produced.")
    lines.extend(
        [
            "",
            "## Governing caveats",
            "",
            "1. C0/C1/C2 and C3-HYBRID use today's retrospectively fitted cascade; every challenger C1/C2/C3 arm also uses the explicitly labelled frozen pre-event bridge sensitivity and cannot be called exact historical production.",
            "2. Best-of-three opening or closing prices need not have been simultaneously executable and the archive has no limits or slippage.",
            "3. W2 has no historical forecast archive. It cleans completed state only; realized upcoming weather is used solely for post-hoc loss attribution.",
            "4. Opening/closing prices are never skill or cascade features. Offered 2021-22 matchup outcomes tune cascade hyperparameters and probability scale.",
            "5. Open and close P&L are separate execution counterfactuals. A close-price wager has zero conventional CLV by construction.",
            "",
            "## Integrity summary",
            "",
            "```json",
            json.dumps(audit, indent=2, sort_keys=True, default=str),
            "```",
        ]
    )
    (output / "CASCADE_MARKET_EVAL_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def run(args: argparse.Namespace) -> None:
    panel_dir = args.panel_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Cascade output directory is nonempty; choose a fresh path: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)

    panel_inputs = _verified_panel_provenance_paths(panel_dir)
    pre_begin_odds_paths = _odds_cache_provenance_paths(args.odds_cache.resolve())
    direct_inputs = [
        *panel_inputs,
        args.weather_coefficients.resolve(),
        args.core_bridge.resolve(),
        *pre_begin_odds_paths,
    ]
    for path in direct_inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    finalist_manifest = args.finalist_dir.resolve() / "manifest.json"
    prevalidated_finalist: tuple[dict[str, Any], Path] | None = None
    finalist_prevalidation_error: str | None = None
    if finalist_manifest.exists():
        try:
            prevalidated_finalist = _validate_finalist_manifest(
                args.finalist_dir.resolve()
            )
            direct_inputs.extend(
                Path(value).resolve()
                for value in prevalidated_finalist[0]["_verified_handoff_paths"]
            )
        except Exception as exc:
            finalist_prevalidation_error = f"{type(exc).__name__}: {exc}"
            if not args.preflight_only:
                raise
            direct_inputs.append(finalist_manifest)
    elif not args.preflight_only:
        raise FileNotFoundError(finalist_manifest)
    provenance_state = begin_cascade_execution_provenance(
        output_dir=output,
        input_paths=direct_inputs,
        args=args,
    )
    pre_begin_handoff_paths = (
        [
            Path(value).resolve()
            for value in prevalidated_finalist[0]["_verified_handoff_paths"]
        ]
        if prevalidated_finalist is not None
        else []
    )
    postvalidated_finalist, _post_panel_paths, _post_odds_paths = (
        revalidate_cascade_dynamic_inputs_after_begin(
            panel_dir=panel_dir,
            expected_panel_paths=panel_inputs,
            odds_cache=args.odds_cache.resolve(),
            expected_odds_paths=pre_begin_odds_paths,
            finalist_dir=args.finalist_dir.resolve(),
            expected_handoff_paths=pre_begin_handoff_paths,
            provenance_state=provenance_state,
        )
    )
    if postvalidated_finalist is not None:
        prevalidated_finalist = postvalidated_finalist
    if prevalidated_finalist is None:
        raise ValueError(
            "cascade state/weather authority requires a valid frozen finalist handoff"
        )
    completion_authority_audit = validate_cascade_completion_authorities(
        prevalidated_finalist[0],
        panel_dir=panel_dir,
        weather_coefficients=args.weather_coefficients,
        core_bridge=args.core_bridge,
    )

    events = pd.read_parquet(panel_dir / "player_event_outcomes.parquet")
    events["event_start_date"] = pd.to_datetime(
        events["event_start_date"], errors="coerce"
    ).dt.normalize()
    event_catalog = events[["event_key", "event_start_date", "event_name"]].drop_duplicates(
        "event_key"
    )
    quotes, odds_audit = load_round_matchup_quotes(
        args.odds_cache.resolve(), panel_dir / "odds_event_crosswalk.parquet", event_catalog
    )
    quotes.to_parquet(output / "normalized_round_matchup_quotes.parquet", index=False)
    print(
        f"Mapped {len(quotes):,} R1-R4 quotes across "
        f"{quotes['event_key'].nunique():,} audited events"
    )

    round_state, state_audit = prepare_round_state(
        panel_dir / "player_round_panel.parquet",
        args.weather_coefficients.resolve(),
    )
    if (
        state_audit["weather_coefficients_sha256"]
        != completion_authority_audit["weather_coefficients"]["sha256"]
        or Path(state_audit["weather_coefficients_path"]).resolve()
        != Path(
            completion_authority_audit["weather_coefficients"]["path"]
        ).resolve()
    ):
        raise ValueError("W2 state weather coefficients are not completion-authoritative")
    score_void_audit = audit_quote_score_voids(quotes, round_state)
    if score_void_audit[
        "archived_vs_score_outcome_mismatches_when_both_scores_valid"
    ]:
        raise ValueError(
            "Archived round outcomes disagree with local target-round scores: "
            + json.dumps(score_void_audit, default=str)
        )
    if args.preflight_only:
        readiness = {
            "research_only": True,
            "sealed_outcomes_scored": False,
            "finalist_manifest_present": finalist_manifest.exists(),
            "finalist_manifest_path": str(finalist_manifest),
            "odds": odds_audit,
            "state": state_audit,
            "completion_authority": completion_authority_audit,
            "score_void_classification": score_void_audit,
            "ready_for_final_evaluation": False,
        }
        if prevalidated_finalist is not None:
            try:
                payload, prediction_path = prevalidated_finalist
                readiness["finalist_contract_valid"] = True
                readiness["selected_model_name"] = payload[
                    "_resolved_selected_name"
                ]
                readiness["selection_role"] = payload["_selection_role"]
                readiness["promotion_gate_passed"] = payload[
                    "_promotion_gate_passed"
                ]
                readiness["full_provenance_chain"] = payload.get(
                    "_provenance_chain_audit", {}
                )
                readiness["finalist_prediction_path"] = str(prediction_path)
                readiness["ready_for_final_evaluation"] = True
            except Exception as exc:
                readiness["finalist_contract_valid"] = False
                readiness["finalist_contract_error"] = f"{type(exc).__name__}: {exc}"
        elif finalist_manifest.exists():
            readiness["finalist_contract_valid"] = False
            readiness["finalist_contract_error"] = finalist_prevalidation_error
        execution_audit = finalize_cascade_execution_provenance(provenance_state)
        readiness["execution_provenance"] = execution_audit
        write_cascade_complete_manifest(
            output / "preflight_readiness.json", readiness, provenance_state
        )
        print(f"Cascade preflight complete without scoring: {output}")
        return

    event_player_map = round_state[
        ["event_key", "year", "event_id", "event_start_date", "dg_id_key"]
    ].drop_duplicates()
    base_predictions, selection, finalist_audit, detail_audit = (
        load_frozen_finalist_predictions(
            args.finalist_dir.resolve(),
            event_player_map,
            expected_verified_paths=pre_begin_handoff_paths,
        )
    )
    selected_candidates = selection.loc[
        selection["selected"].fillna(False), "candidate"
    ].astype(str).tolist()
    selection.to_csv(output / "candidate_selection.csv", index=False)
    print("Frozen SG-scale candidates: " + ", ".join(selected_candidates))
    verified_core_bridge = Path(
        str(finalist_audit["full_provenance_chain"]["core_bridge_path"])
    ).resolve()
    if args.core_bridge.resolve() != verified_core_bridge:
        raise ValueError(
            "--core-bridge differs from the discovery-provenance-bound core_bridge"
        )
    if (
        _sha256(args.core_bridge.resolve())
        != finalist_audit["full_provenance_chain"]["core_bridge_sha256"]
    ):
        raise ValueError("discovery-provenance-bound core_bridge hash mismatch")
    core_bridge = pd.read_parquet(args.core_bridge.resolve())

    c0_piece: pd.DataFrame | None = None
    compatibility_pieces: list[pd.DataFrame] = []
    shrink_pieces: list[pd.DataFrame] = []
    c3_pieces: list[pd.DataFrame] = []
    coefficient_pieces: list[pd.DataFrame] = []
    tuning_pieces: list[pd.DataFrame] = []
    frame_audit: list[dict[str, Any]] = []
    new_candidates = selection.loc[
        selection["selected"].fillna(False)
        & selection["candidate_type"].isin(
            ["discovery_frozen_finalist", "diagnostic_challenger"]
        ),
        "candidate",
    ].astype(str).tolist()
    hybrid_candidates = set(new_candidates[: args.max_hybrid_candidates])
    frozen_selected_name = str(finalist_audit["finalist_name"])
    frozen_pre_event_lambda = float(
        finalist_audit["discovery_selected_pre_event_downstream_lambda"]
    )

    for index, candidate in enumerate(selected_candidates, start=1):
        print(f"[{index}/{len(selected_candidates)}] Building cascade arms for {candidate}")
        candidate_pre_event_lambda = (
            frozen_pre_event_lambda if candidate == frozen_selected_name else 1.0
        )
        pre_event_lambda_source = (
            "discovery_frozen_selection_manifest"
            if candidate == frozen_selected_name
            else "strict_pt_control_compatibility_lambda_one"
        )
        candidate_frame = attach_core_and_candidate(
            round_state,
            base_predictions,
            core_bridge,
            candidate,
            pre_event_downstream_lambda=candidate_pre_event_lambda,
            pre_event_downstream_lambda_source=pre_event_lambda_source,
        )
        frame_audit.append(
            {
                "candidate": candidate,
                "round_rows": int(len(candidate_frame)),
                "events": int(candidate_frame["event_key"].nunique()),
                "core_bridge_coverage_rate": float(
                    candidate_frame["legacy_core_pred_step2"].notna().mean()
                ),
                "c0_coverage_rate": float(candidate_frame["cascade_prediction"].notna().mean()),
                "base_prediction_coverage_rate": float(
                    candidate_frame["base_prediction_available"].mean()
                ),
                "base_training_boundary_violations": int(
                    candidate_frame["base_training_boundary_violation"].sum()
                ),
                "pre_event_downstream_lambda": candidate_pre_event_lambda,
                "pre_event_downstream_lambda_source": pre_event_lambda_source,
                "pre_event_downstream_lambda_scope": (
                    "frozen_pre_event_course_history_downstream_terms"
                ),
                "live_update_lambda_scope": (
                    "separate_completed_prior_round_current_cascade_update"
                ),
            }
        )
        compatibility = build_c0_c1_player_predictions(
            round_state, {candidate: candidate_frame}
        )
        if c0_piece is None:
            c0_piece = compatibility.loc[compatibility["variant"].eq("C0")].copy()
        compatibility_pieces.append(
            compatibility.loc[~compatibility["variant"].eq("C0")].copy()
        )
        shrink, live_tuning = tune_live_update_shrinkage(
            candidate_frame, quotes, candidate=candidate
        )
        shrink_pieces.append(shrink)
        tuning_pieces.append(live_tuning)

        for arm in ("w0", "w2"):
            predictions, coefficients, tuning = tune_and_predict_c3(
                candidate_frame,
                quotes,
                candidate=candidate,
                arm=arm,
                mode="pt",
            )
            if not predictions.empty:
                c3_pieces.append(predictions)
                coefficient_pieces.append(coefficients)
            tuning_pieces.append(tuning)
        if candidate in hybrid_candidates:
            for arm in ("w0", "w2"):
                predictions, coefficients, tuning = tune_and_predict_c3(
                    candidate_frame,
                    quotes,
                    candidate=candidate,
                    arm=arm,
                    mode="hybrid",
                )
                if not predictions.empty:
                    c3_pieces.append(predictions)
                    coefficient_pieces.append(coefficients)
                tuning_pieces.append(tuning)
        del candidate_frame

    if c0_piece is None:
        raise ValueError("No C0 compatibility predictions were built")
    player_predictions = pd.concat(
        [c0_piece, *compatibility_pieces, *shrink_pieces, *c3_pieces],
        ignore_index=True,
        sort=False,
    )
    duplicate_predictions = player_predictions.duplicated(
        ["model", "event_key", "dg_id_key", "round_num"]
    )
    if duplicate_predictions.any():
        raise ValueError(
            f"Final player predictions contain {int(duplicate_predictions.sum())} duplicates"
        )
    c3_boundary = pd.to_datetime(
        player_predictions.get("training_max_event_end"), errors="coerce"
    )
    prediction_year = pd.to_numeric(
        player_predictions.get("prediction_year"), errors="coerce"
    )
    cutoff = pd.to_datetime(prediction_year.astype("Int64").astype(str) + "-01-01", errors="coerce")
    c3_boundary_bad = c3_boundary.notna() & c3_boundary.ge(cutoff)
    if c3_boundary_bad.any():
        raise ValueError(f"C3 training boundary violations: {int(c3_boundary_bad.sum())}")
    c3_weather_lineage_audit = audit_c3_weather_lineage(
        player_predictions,
        authoritative_coefficients_path=Path(
            completion_authority_audit["weather_coefficients"]["path"]
        ),
        authoritative_coefficients_sha256=completion_authority_audit[
            "weather_coefficients"
        ]["sha256"],
    )
    player_predictions.to_parquet(
        output / "cascade_player_round_predictions.parquet", index=False
    )
    coefficients = (
        pd.concat(coefficient_pieces, ignore_index=True)
        if coefficient_pieces
        else pd.DataFrame()
    )
    tuning = (
        pd.concat(tuning_pieces, ignore_index=True)
        if tuning_pieces
        else pd.DataFrame()
    )
    coefficients.to_csv(output / "c3_coefficients.csv", index=False)
    tuning.to_csv(output / "c3_discovery_hyperparameter_scores.csv", index=False)
    pd.DataFrame(frame_audit).to_csv(output / "candidate_round_coverage.csv", index=False)
    quote_coverage = audit_model_quote_coverage(player_predictions, quotes)
    quote_coverage.to_csv(output / "model_quote_coverage.csv", index=False)
    primary_variant = quote_coverage["variant"].isin(
        ["C1", "C2_NO_LIVE_UPDATE", "C2_DISCOVERY_LIVE_UPDATE_SHRINK"]
    ) | quote_coverage["variant"].astype(str).str.startswith(
        "C3_pt_continuous_partial_pool_"
    )
    compatibility_drop = quote_coverage.loc[
        primary_variant
        & quote_coverage["c0_quotes_dropped"].gt(0)
    ]
    if not compatibility_drop.empty:
        raise ValueError(
            "A finalist/control compatibility arm dropped offered C0 matchups: "
            + compatibility_drop.to_json(orient="records")
        )

    print(
        f"Calibrating {player_predictions['model'].nunique():,} cascade model arms "
        "on 2021-22 only"
    )
    scored, sigmas = calibrate_and_score_pairs(player_predictions, quotes)
    if scored.empty:
        raise ValueError("No round matchup probabilities survived calibration")
    pricing_probability_audit = audit_pricing_probability_algebra(scored)
    probability_sum_error = (
        scored["p1_probability"]
        + scored["p2_probability"]
        + scored["push_probability"]
        - 1.0
    ).abs()
    reverse_win, reverse_loss, reverse_push = discrete_round_probabilities(
        -scored["skill_difference"].to_numpy(float),
        scored["sigma"].to_numpy(float),
        scored["tie_threshold"].to_numpy(float),
    )
    reverse_symmetry_error = np.maximum.reduce(
        [
            np.abs(scored["p1_probability"].to_numpy(float) - reverse_loss),
            np.abs(scored["p2_probability"].to_numpy(float) - reverse_win),
            np.abs(scored["push_probability"].to_numpy(float) - reverse_push),
        ]
    )
    if probability_sum_error.max() > 1e-12 or reverse_symmetry_error.max() > 1e-12:
        raise AssertionError("Discrete round probabilities fail normalization/reverse symmetry")
    scored.to_parquet(output / "round_matchup_probabilities.parquet", index=False)
    sigmas.to_csv(output / "discovery_probability_sigmas.csv", index=False)
    proper, calibration = summarize_proper_scores(scored)
    proper.to_csv(output / "proper_scores.csv", index=False)
    calibration.to_csv(output / "calibration.csv", index=False)

    bet_pieces: list[pd.DataFrame] = []
    zero_gate_pieces: list[pd.DataFrame] = []
    for model, model_scored in scored.groupby("model", observed=True):
        for price_point in PRICE_POINTS:
            print(f"Pricing {price_point} boards for {model}")
            lines = expand_betting_lines(model_scored, price_point)
            for policy in (*BOOKS, "best_of_three"):
                for sizing in ("flat_1u", "production_risk_to_win"):
                    selected = select_and_settle(
                        lines, line_policy=policy, sizing=sizing, min_ev=EV_GATE
                    )
                    if not selected.empty:
                        selected["opportunity_mode"] = "ev_ge_3pct"
                        bet_pieces.append(selected)
                    if policy == "best_of_three":
                        zero_gate = select_and_settle(
                            lines, line_policy=policy, sizing=sizing, min_ev=0.0
                        )
                        if not zero_gate.empty:
                            zero_gate["opportunity_mode"] = "counterfactual_ev_ge_0pct"
                            zero_gate_pieces.append(zero_gate)
            del lines
    bets = pd.concat(bet_pieces, ignore_index=True) if bet_pieces else pd.DataFrame()
    if bets.empty:
        raise ValueError("The 3% EV gate selected no round matchup bets")
    zero_gate_bets = (
        pd.concat(zero_gate_pieces, ignore_index=True)
        if zero_gate_pieces
        else pd.DataFrame()
    )
    all_offered_bets = all_offered_edge_bets(scored)
    settlement_completeness = {
        "primary_selected_rows": int(len(bets)),
        "primary_ungraded_rows": int((~bets["is_graded"]).sum()),
        "zero_gate_selected_rows": int(len(zero_gate_bets)),
        "zero_gate_ungraded_rows": int(
            (~zero_gate_bets["is_graded"]).sum()
        )
        if not zero_gate_bets.empty
        else 0,
        "all_offered_selected_rows": int(len(all_offered_bets)),
        "all_offered_ungraded_rows": int(
            (~all_offered_bets["is_graded"].fillna(False).astype(bool)).sum()
        )
        if not all_offered_bets.empty
        else 0,
    }
    _atomic_json(output / "settlement_completeness_audit.json", settlement_completeness)
    all_offered_bets.to_parquet(
        output / "all_offered_edge_bets.parquet", index=False
    )
    if (
        settlement_completeness["primary_ungraded_rows"]
        or settlement_completeness["zero_gate_ungraded_rows"]
        or settlement_completeness["all_offered_ungraded_rows"]
    ):
        raise ValueError(
            "Selected round matchup rows lack settlement; refusing biased P&L: "
            + json.dumps(settlement_completeness, sort_keys=True)
        )
    bets.to_parquet(output / "bets.parquet", index=False)
    zero_gate_bets.to_parquet(output / "counterfactual_zero_gate_bets.parquet", index=False)
    performance = summarize_bets(bets, scored)
    performance.to_csv(output / "performance.csv", index=False)
    buckets = bucket_bet_performance(bets)
    buckets.to_csv(output / "bet_bucket_performance.csv", index=False)
    zero_buckets = bucket_bet_performance(zero_gate_bets)
    zero_buckets.to_csv(
        output / "counterfactual_zero_gate_bucket_performance.csv", index=False
    )
    all_edge_buckets = summarize_all_edge_buckets(all_offered_bets)
    all_edge_buckets.to_csv(
        output / "all_offered_edge_bucket_performance.csv", index=False
    )
    weekly, golfer_week = worst_weeks_and_golfers(bets)
    weekly.to_csv(output / "bad_weeks.csv", index=False)
    golfer_week.to_csv(output / "bad_week_golfers.csv", index=False)
    bootstrap = event_cluster_bootstrap(
        performance,
        bets,
        scored,
        draws=args.bootstrap_draws,
        seed=args.seed,
    )
    bootstrap.to_csv(output / "event_cluster_bootstrap.csv", index=False)

    def _formula_error(variant: str, expected: pd.Series) -> float:
        use = player_predictions["variant"].eq(variant)
        if not use.any():
            return np.nan
        return float(
            np.nanmax(
                np.abs(
                    player_predictions.loc[use, "prediction"].to_numpy(float)
                    - expected.loc[use].to_numpy(float)
                )
            )
        )

    c1_expected = (
        player_predictions["base_event_hybrid"]
        + player_predictions["legacy_stage_adjustment"]
    )
    no_live_expected = player_predictions["base_event_hybrid"]
    shrink_expected = (
        player_predictions["base_event_hybrid"]
        + player_predictions["live_update_lambda"]
        * player_predictions["legacy_stage_adjustment"]
    )
    c1_formula_error = _formula_error("C1", c1_expected)
    no_live_formula_error = _formula_error("C2_NO_LIVE_UPDATE", no_live_expected)
    shrink_formula_error = _formula_error(
        "C2_DISCOVERY_LIVE_UPDATE_SHRINK", shrink_expected
    )
    sigma_upper_hits = int(
        pd.to_numeric(sigmas["sigma"], errors="coerce").ge(12.0 - 1e-6).sum()
    )
    lambda_tuning = tuning.loc[
        tuning.get(
            "live_update_lambda", pd.Series(np.nan, index=tuning.index)
        ).notna()
    ].copy()
    lambda_selection_after_discovery = int(
        pd.to_datetime(
            lambda_tuning.get(
                "selection_max_event_date",
                pd.Series(pd.NaT, index=lambda_tuning.index),
            ),
            errors="coerce",
        ).gt(DISCOVERY_END).sum()
    )
    audit = {
        "research_only": True,
        "production_changed": False,
        "evaluator_source_sha256": _sha256(Path(__file__).resolve()),
        "handoff_validator_source_sha256": _sha256(
            ARCHIVE / "base_skill_frozen_handoff.py"
        ),
        "created_at_utc": pd.Timestamp.now(tz="UTC"),
        "odds": odds_audit,
        "score_void_classification": score_void_audit,
        "state": state_audit,
        "c3_weather_lineage": c3_weather_lineage_audit,
        "frozen_finalist": finalist_audit,
        "attribution_detail": detail_audit,
        "selected_candidates": selected_candidates,
        "selection_role": finalist_audit["selection_role"],
        "promotion_gate_passed": finalist_audit["promotion_gate_passed"],
        "clean_img_allowed_only_through_frozen_base_candidate": True,
        "player_prediction_rows": int(len(player_predictions)),
        "models": int(player_predictions["model"].nunique()),
        "duplicate_player_prediction_keys": int(duplicate_predictions.sum()),
        "c3_training_boundary_violations": int(c3_boundary_bad.sum()),
        "strict_c3_uses_current_cascade_prediction": False,
        "hybrid_c3_retrospective_label_present": bool(
            player_predictions["variant"].astype(str).str.contains("hybrid").any()
        ),
        "closing_price_used_as_skill_or_cascade_feature": False,
        "opening_price_used_as_skill_or_cascade_feature": False,
        "market_prices_used_for_cascade_probability_calibration": False,
        "pricing_probability_algebra": pricing_probability_audit,
        "market_devig_proper_score_scope": (
            "resolved_push_on_tie_two_way_only; ties_lose_without_draw_quote_excluded"
        ),
        "ties_lose_rows_excluded_from_market_proper_score": int(
            (
                scored["score_model_eligible"].fillna(False).astype(bool)
                & scored["tie_rule_ties_lose"].fillna(False).astype(bool)
            ).sum()
        ),
        "offered_matchup_outcomes_used_for_discovery_tuning": True,
        "sealed_results_used_for_selection": False,
        "live_lambda_selection_after_discovery_rows": lambda_selection_after_discovery,
        "frozen_pre_event_downstream_lambda": frozen_pre_event_lambda,
        "pre_event_downstream_lambda_scope": (
            "frozen_pre_event_course_history_downstream_terms"
        ),
        "live_update_lambda_scope": (
            "separate_completed_prior_round_current_cascade_update"
        ),
        "target_or_upcoming_realized_weather_feature_used": False,
        "realized_weather_used_for_posthoc_attribution": True,
        "probability_sum_max_abs_error": float(probability_sum_error.max()),
        "reverse_pair_symmetry_max_abs_error": float(reverse_symmetry_error.max()),
        "push_probability_min": float(scored["push_probability"].min()),
        "push_probability_max": float(scored["push_probability"].max()),
        "probability_sigma_upper_boundary_hits": sigma_upper_hits,
        "compatibility_quote_menu_drops": int(
            compatibility_drop["c0_quotes_dropped"].sum()
        ),
        "settled_bets": int(len(bets)),
        "ungraded_bets": settlement_completeness["primary_ungraded_rows"],
        "zero_gate_ungraded_bets": settlement_completeness[
            "zero_gate_ungraded_rows"
        ],
        "all_offered_ungraded_bets": settlement_completeness[
            "all_offered_ungraded_rows"
        ],
        "settled_void_pushes": int(
            bets["settlement_status"].isin(["void", "push"]).sum()
        ),
        "nonzero_pnl_on_void_pushes": int(
            (
                bets["settlement_status"].isin(["void", "push"])
                & bets["pnl"].abs().gt(1e-12)
            ).sum()
        ),
        "c1_formula_max_abs_error": c1_formula_error,
        "no_live_formula_max_abs_error": no_live_formula_error,
        "live_shrink_formula_max_abs_error": shrink_formula_error,
        "price_points_evaluated": list(PRICE_POINTS),
        "fixed_books_evaluated": list(BOOKS),
        "best_of_three_evaluated": True,
        "bootstrap_draws": int(args.bootstrap_draws),
        "bootstrap_includes_zero_bet_eligible_events": True,
        "seed": int(args.seed),
    }
    if (
        audit["duplicate_player_prediction_keys"]
        or audit["c3_training_boundary_violations"]
        or audit["live_lambda_selection_after_discovery_rows"]
        or audit["compatibility_quote_menu_drops"]
        or audit["ungraded_bets"]
        or audit["nonzero_pnl_on_void_pushes"]
        or audit["probability_sigma_upper_boundary_hits"]
        or (
            np.isfinite(audit["c1_formula_max_abs_error"])
            and audit["c1_formula_max_abs_error"] > 1e-12
        )
        or (
            np.isfinite(audit["no_live_formula_max_abs_error"])
            and audit["no_live_formula_max_abs_error"] > 1e-12
        )
        or (
            np.isfinite(audit["live_shrink_formula_max_abs_error"])
            and audit["live_shrink_formula_max_abs_error"] > 1e-12
        )
    ):
        raise AssertionError(f"Final cascade audit failed: {audit}")
    _atomic_json(output / "audit.json", audit)
    write_report(
        output, audit, selection, proper, performance, bootstrap, tuning, coefficients
    )

    execution_audit = finalize_cascade_execution_provenance(provenance_state)
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
        "complete": True,
        "research_only": True,
        "execution_provenance": execution_audit,
        "completion_authority": completion_authority_audit,
        "inputs": {
            "panel_manifest": {
                "path": str(panel_dir / "manifest.json"),
                "sha256": _sha256(panel_dir / "manifest.json"),
            },
            "player_event_outcomes": {
                "path": str(panel_dir / "player_event_outcomes.parquet"),
                "sha256": _sha256(panel_dir / "player_event_outcomes.parquet"),
            },
            "player_round_panel": {
                "path": str(panel_dir / "player_round_panel.parquet"),
                "sha256": _sha256(panel_dir / "player_round_panel.parquet"),
            },
            "odds_event_crosswalk": {
                "path": str(panel_dir / "odds_event_crosswalk.parquet"),
                "sha256": _sha256(panel_dir / "odds_event_crosswalk.parquet"),
            },
            "weather_coefficients": {
                "path": str(args.weather_coefficients.resolve()),
                "sha256": _sha256(args.weather_coefficients.resolve()),
            },
            "frozen_finalist_manifest": {
                "path": finalist_audit["manifest_path"],
                "sha256": finalist_audit["manifest_sha256"],
            },
            "frozen_finalist_predictions": {
                "path": finalist_audit["prediction_path"],
                "sha256": finalist_audit["prediction_sha256"],
            },
            "core_bridge": {
                "path": str(args.core_bridge.resolve()),
                "sha256": _sha256(args.core_bridge.resolve()),
            },
            "odds_cache_manifest": {
                "path": str(args.odds_cache.resolve() / "manifest.json"),
                "sha256": _sha256(args.odds_cache.resolve() / "manifest.json"),
            },
        },
        "audit_sha256": _sha256(output / "audit.json"),
        "artifacts": artifacts,
    }
    write_cascade_complete_manifest(
        output / "manifest.json", manifest, provenance_state
    )
    print(f"Cascade evaluation complete: {output}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=PANEL_DIR)
    parser.add_argument("--finalist-dir", type=Path, default=DEFAULT_FINALIST_DIR)
    parser.add_argument(
        "--weather-coefficients",
        type=Path,
        default=DEFAULT_WEATHER_COEFFICIENTS,
    )
    parser.add_argument("--core-bridge", type=Path, default=CORE_BRIDGE)
    parser.add_argument("--odds-cache", type=Path, default=ODDS_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-new-candidates", type=int, default=6)
    parser.add_argument("--max-hybrid-candidates", type=int, default=2)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate inputs/odds/score ties without fitting or scoring any model.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
