"""Exact profile-only course-fit sensitivity on the frozen v4 market replay.

The three primary arms hold every v4 input fixed, including course history,
course experience, shot variance, course shape, odds, seeds, and simulation
mechanics.  They change only the later-model course-profile contribution that
was embedded in the archived R1 prediction:

    PROFILE_FULL = v4 skill
    PROFILE_HALF = v4 skill - 0.5 * legacy_profile_contribution
    PROFILE_NONE = v4 skill - 1.0 * legacy_profile_contribution

The legacy profile contribution is reconstructed from the exact weekly-model
path (model_results2-aligned category histories, event-relative accuracy, the
pre-BMW course-random-effect git blob, and the frozen high/low step-4
coefficients).  Scalar skill changes are spread equally across OTT, APP, ARG,
and PUTT so category means continue to sum to the arm's total mean.

This is a research harness.  It does not import into or modify production.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import platform
import sqlite3
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

try:  # Package import in tests; direct import when executed as a script.
    from . import backtest_shot_mean_closing_markets as bt
except ImportError:  # pragma: no cover - exercised by the CLI invocation
    import backtest_shot_mean_closing_markets as bt


REPO = Path(__file__).resolve().parents[1]
SIM_PREP = bt.SIM_PREP
V4_DIR = (
    REPO
    / "permanent_data"
    / "research"
    / "shot_mean_closing_backtest_20230822_20260822_v4"
)
DEFAULT_OUTPUT = (
    REPO
    / "permanent_data"
    / "research"
    / "course_profile_sensitivity_20230822_20260822"
)
LOW_COEFFICIENTS = SIM_PREP / "work" / "model_coeffs_low_skill.json"
HIGH_COEFFICIENTS = SIM_PREP / "work" / "model_coeffs_high_skill.json"
WEEKLY_MANIFEST = SIM_PREP / "work" / "weekly_model_manifest.json"
WEEKLY_MODEL_SOURCE = SIM_PREP / "weekly_model.py"

LEGACY_RE_GIT_OBJECT = "5cb0116^:course_random_effects.csv"
LEGACY_RE_BLOB = "5bef0543339773f11f6bf18ba96e05285d9c4907"
LEGACY_RE_SHA256 = "270fd24fcaf21e17b2a4edacff60aa12e74c4c9b083060e32cab4afb4c852b1f"

PROFILE_FULL = "profile_full"
PROFILE_HALF = "profile_half"
PROFILE_NONE = "profile_none"
PROFILE_ARMS = (PROFILE_FULL, PROFILE_HALF, PROFILE_NONE)
PROFILE_SCALE = {PROFILE_FULL: 0.0, PROFILE_HALF: -0.5, PROFILE_NONE: -1.0}
PROFILE_TERMS = (
    "ema_21_sg_putt",
    "ema_21_sg_app",
    "ema_21_rel_dd",
    "ema_21_rel_acc",
    "ema_11_sg_arg",
)
PROFILE_SOURCES = (
    ("ema_21_sg_putt", "ag_putt", 21),
    ("ema_21_sg_app", "sg_app", 21),
    ("ema_21_rel_dd", "rel_dd", 21),
    ("ema_21_rel_acc", "rel_acc", 21),
    # The historical column says 11, but the fitted weekly path used span=21.
    ("ema_11_sg_arg", "ag_arg", 21),
)
PROFILE_KEYS = ["event_key", "dg_id"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--v4-dir", type=Path, default=V4_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--event-key",
        action="append",
        help="Canonical v4 event key; repeat for multiple events. Default: all 105.",
    )
    parser.add_argument("--event-limit", type=int)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate frozen inputs and export the exact profile panel without simulating.",
    )
    return parser.parse_args(argv)


def hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_legacy_random_effects() -> tuple[pd.DataFrame, dict[str, Any], bytes]:
    command = [
        "git",
        "-c",
        f"safe.directory={SIM_PREP.as_posix()}",
        "show",
        LEGACY_RE_GIT_OBJECT,
    ]
    result = subprocess.run(
        command,
        cwd=SIM_PREP,
        check=True,
        capture_output=True,
    )
    content = result.stdout
    actual_sha = hash_bytes(content)
    if actual_sha != LEGACY_RE_SHA256:
        raise ValueError(
            "Legacy course-RE git object changed or resolved unexpectedly: "
            f"expected {LEGACY_RE_SHA256}, got {actual_sha}"
        )
    blob = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={SIM_PREP.as_posix()}",
            "rev-parse",
            LEGACY_RE_GIT_OBJECT,
        ],
        cwd=SIM_PREP,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if blob != LEGACY_RE_BLOB:
        raise ValueError(f"Legacy course-RE blob mismatch: expected {LEGACY_RE_BLOB}, got {blob}")
    frame = pd.read_csv(io.BytesIO(content))
    if frame["course_num"].duplicated().any():
        raise ValueError("Legacy course-RE snapshot contains duplicate course_num rows")
    audit = {
        "git_object": LEGACY_RE_GIT_OBJECT,
        "git_blob": blob,
        "sha256": actual_sha,
        "bytes": len(content),
        "courses": int(frame["course_num"].nunique()),
    }
    return frame, audit, content


def expected_v4_paths(end_year: int) -> dict[str, Path]:
    paths = {
        "factor_panel": bt.DEFAULT_FACTOR_PANEL,
        "odds_cache_manifest": bt.DEFAULT_ODDS_CACHE / "manifest.json",
        "predictions": bt.DEFAULT_PREDS,
        "crosswalk": bt.DEFAULT_CROSSWALK,
        "correlation": bt.CORR_PATH,
        "coefficients": bt.COEFF_PATH,
        "historical_database": bt.DEFAULT_DB,
        "source::backtest_engine": Path(bt.__file__).resolve(),
        "source::odds_cache_reader": bt.ARCHIVE / "historical_odds_cache.py",
        "source::market_scoring": bt.ARCHIVE / "market_backtest_scoring.py",
        "source::category_parameter_builder": bt.ARCHIVE / "backtest_distribution_engines_2026.py",
        "source::course_shape_builder": bt.ARCHIVE / "backtest_shot_dispersion_recent.py",
        "source::shot_feature_builder": bt.ARCHIVE / "build_bmw_shot_features_shadow.py",
        "source::sim_inputs": REPO / "sim_inputs.py",
        "shot_export::manifest": bt.EXPORT / "manifest.json",
    }
    paths.update(
        {
            f"shot_export::shot_sg_{year}": bt.EXPORT / f"pga_{year}_shot_sg.parquet"
            for year in range(2020, end_year + 1)
        }
    )
    return paths


def validate_v4_inputs(v4_config: dict[str, Any]) -> dict[str, str]:
    expected = v4_config["input_hashes"]
    paths = expected_v4_paths(pd.Timestamp(v4_config["end"]).year)
    missing_keys = sorted(set(expected) - set(paths))
    if missing_keys:
        raise ValueError(f"Harness has no path mapping for v4 inputs: {missing_keys}")
    actual: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = bt.sha256_file(path)
        actual[name] = digest
        if digest != expected[name]:
            raise ValueError(
                f"Frozen v4 input mismatch for {name}: expected {expected[name]}, got {digest} ({path})"
            )
    return actual


def selected_events(v4_config: dict[str, Any], args: argparse.Namespace) -> list[str]:
    events = list(v4_config["event_keys"])
    if args.event_key:
        requested = set(args.event_key)
        unknown = sorted(requested - set(events))
        if unknown:
            raise ValueError(f"Requested events are not in frozen v4: {unknown}")
        events = [event for event in events if event in requested]
    if args.event_limit is not None:
        if args.event_limit <= 0:
            raise ValueError("event-limit must be positive")
        events = events[: args.event_limit]
    if not events:
        raise ValueError("No v4 events selected")
    return events


def synthetic_date_from_manifest(manifest: dict[str, Any]) -> pd.Timestamp:
    started = pd.Timestamp(int(manifest["run_started_ns"]), unit="ns")
    prediction = manifest["outputs"].get("sg_preds_resids.csv", {})
    if prediction.get("sha256") != bt.sha256_file(bt.DEFAULT_PREDS):
        raise ValueError("Weekly manifest does not describe the frozen v4 prediction file")
    return started.normalize()


def build_exact_profile_panel(
    db_path: Path,
    event_keys: Iterable[str],
    random_effects: pd.DataFrame,
    low_coefficient: float,
    high_coefficient: float,
    synthetic_date: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reconstruct the later weekly-model profile term, including its quirks."""

    started = time.perf_counter()
    uri = f"file:{db_path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rounds = pd.read_sql_query(
            "SELECT player_name, event_id, dg_id, year, tour, round_date, "
            "course_num, driving_acc, round_num, sg_total_adj "
            "FROM player_rounds WHERE sg_total_adj IS NOT NULL "
            "AND round_date > '2015-04-01'",
            connection,
        )
        model = pd.read_sql_query(
            "SELECT rowid, dg_id, Date, ag_putt, sg_app, rel_dd, ag_arg, skill_est "
            "FROM model_results2 WHERE skill_est IS NOT NULL AND dg_id IS NOT NULL",
            connection,
        )

    rounds["round_date"] = pd.to_datetime(rounds["round_date"], errors="coerce")
    rounds = rounds.dropna(subset=["round_date"]).copy()
    rounds["is_synth"] = False
    latest = (
        rounds.sort_values("round_date")
        .groupby("player_name", as_index=False)
        .tail(1)
        .copy()
    )
    latest["round_date"] = synthetic_date
    latest["round_num"] = latest["round_num"] + 1
    latest["is_synth"] = True
    rounds = pd.concat([rounds, latest], ignore_index=True)

    model["Date"] = pd.to_datetime(model["Date"], errors="coerce")
    model = (
        model.dropna(subset=["Date"])
        .sort_values(["dg_id", "Date", "rowid"], kind="mergesort")
        .drop_duplicates(["dg_id", "Date"], keep="last")
    )
    aligned = pd.merge_asof(
        rounds.sort_values("round_date"),
        model.sort_values("Date"),
        left_on="round_date",
        right_on="Date",
        by="dg_id",
        direction="backward",
    )
    aligned["ag_putt"] = aligned["ag_putt"].fillna(0.0)
    aligned = aligned.dropna(subset=["skill_est", "sg_total_adj"]).copy()

    # This intentionally mirrors weekly_model.py: centering is by whole event,
    # not by round, and includes the appended synthetic latest rows.
    aligned["field_avg_acc"] = aligned.groupby(
        ["tour", "year", "event_id"]
    )["driving_acc"].transform("mean")
    aligned["rel_acc"] = aligned["driving_acc"] - aligned["field_avg_acc"]

    for output, source, span in PROFILE_SOURCES:
        aligned[output] = (
            aligned.groupby("player_name")[source]
            .transform(lambda values: values.ewm(span=span, adjust=False).mean().shift(1))
            .fillna(0.0)
        )
    aligned["adj_skill_est"] = aligned["skill_est"]
    aligned["ema_adj_skill_est"] = (
        aligned.groupby("player_name")["adj_skill_est"]
        .transform(lambda values: values.ewm(span=11, adjust=False).mean().shift(1))
        .fillna(-2.15)
    )
    aligned["event_key"] = [
        bt.canonical_event_key(year, event_id)
        for year, event_id in aligned[["year", "event_id"]].itertuples(index=False, name=None)
    ]
    requested = set(event_keys)
    panel = aligned.loc[
        (~aligned["is_synth"])
        & aligned["tour"].eq("pga")
        & aligned["round_num"].eq(1)
        & aligned["event_key"].isin(requested)
    ].copy()
    del aligned, rounds, latest, model
    gc.collect()

    panel = panel.merge(
        random_effects,
        on="course_num",
        how="left",
        suffixes=("", "_courseRE"),
        indicator="_course_re_merge",
        validate="many_to_one",
    )
    panel["profile_re_available"] = panel["_course_re_merge"].eq("both")
    panel["profile_raw"] = sum(
        panel[term] * panel[f"{term}_courseRE"] for term in PROFILE_TERMS
    ).fillna(0.0)
    high_branch = panel["ema_adj_skill_est"] > 0.5
    panel["profile_skill_bucket"] = np.where(high_branch, "high", "low")
    panel["profile_coefficient"] = np.where(
        high_branch, high_coefficient, low_coefficient
    )
    panel["profile_contribution"] = panel["profile_raw"] * panel["profile_coefficient"]
    panel["profile_half_delta"] = -0.5 * panel["profile_contribution"]
    panel["profile_none_delta"] = -panel["profile_contribution"]
    panel["round_date"] = pd.to_datetime(panel["round_date"], errors="coerce").dt.normalize()
    panel["dg_id"] = pd.to_numeric(panel["dg_id"], errors="raise").astype(int)
    if panel.duplicated(PROFILE_KEYS).any():
        duplicates = panel.loc[panel.duplicated(PROFILE_KEYS, keep=False), PROFILE_KEYS]
        raise ValueError(f"Profile panel contains duplicate event/player rows:\n{duplicates.head(20)}")
    missing_events = sorted(requested - set(panel["event_key"]))
    if missing_events:
        raise ValueError(f"No exact profile rows reconstructed for events: {missing_events}")

    keep = [
        "event_key",
        "year",
        "event_id",
        "round_date",
        "round_num",
        "course_num",
        "dg_id",
        "player_name",
        "skill_est",
        "ema_adj_skill_est",
        *PROFILE_TERMS,
        "profile_re_available",
        "profile_raw",
        "profile_skill_bucket",
        "profile_coefficient",
        "profile_contribution",
        "profile_half_delta",
        "profile_none_delta",
    ]
    panel = panel[keep].sort_values(
        ["round_date", "event_key", "player_name", "dg_id"], kind="mergesort"
    ).reset_index(drop=True)
    audit = {
        "database_round_rows": int(len(panel)),
        "profile_rows": int(len(panel)),
        "events": int(panel["event_key"].nunique()),
        "players": int(panel["dg_id"].nunique()),
        "missing_course_re_rows": int((~panel["profile_re_available"]).sum()),
        "high_branch_rows": int(panel["profile_skill_bucket"].eq("high").sum()),
        "low_branch_rows": int(panel["profile_skill_bucket"].eq("low").sum()),
        "synthetic_date": synthetic_date.date().isoformat(),
        "runtime_seconds": time.perf_counter() - started,
    }
    return panel, audit


def attach_profile(field: pd.DataFrame, event_profile: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dg_id",
        "course_num",
        "ema_adj_skill_est",
        "profile_raw",
        "profile_skill_bucket",
        "profile_coefficient",
        "profile_contribution",
        "profile_re_available",
    ]
    output = field.copy()
    output["_profile_field_order"] = np.arange(len(output), dtype=int)
    output = output.merge(
        event_profile[columns],
        on="dg_id",
        how="left",
        sort=False,
        validate="one_to_one",
    )
    output = (
        output.sort_values("_profile_field_order", kind="mergesort")
        .drop(columns="_profile_field_order")
        .reset_index(drop=True)
    )
    exact = output["skill_source"].eq("r1_prediction")
    missing = exact & output["profile_contribution"].isna()
    if missing.any():
        players = output.loc[missing, ["dg_id", "player_name"]].to_dict("records")
        raise ValueError(f"Exact R1 predictions are missing reconstructed profile rows: {players}")
    # Fallback skills did not come from pred_step7 and therefore do not contain
    # this legacy profile term.  Removing anything from them would be invented.
    output["profile_in_baseline"] = np.where(
        exact,
        output["profile_contribution"].fillna(0.0),
        0.0,
    )
    output["profile_applied_to_r1_prediction"] = exact
    for arm, scale in PROFILE_SCALE.items():
        output[f"delta_{arm}"] = scale * output["profile_in_baseline"]
        output[f"skill_{arm}"] = output["skill_est"] + output[f"delta_{arm}"]
    return output


def scalar_arm_deltas(field: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        arm: np.ascontiguousarray(field[f"delta_{arm}"].to_numpy(float))
        for arm in PROFILE_ARMS
    }


def simulate_scalar_skill_shift(
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
    """Run v4 with a scalar mean shift spread equally over four categories."""

    # Use the frozen v4 function verbatim for the zero-delta arm.  This makes
    # PROFILE_FULL an executable replication control, not merely an assumption.
    if np.count_nonzero(delta) == 0:
        return bt.simulate_scores(
            params,
            field,
            chol,
            coefficients,
            delta,
            simulations,
            seed,
            cut_line,
            week_latent_sd,
        )

    import sims_kernel

    n_players = len(field)
    skill = field["skill_est"].to_numpy(float)
    total_mean = np.ascontiguousarray(skill + delta)
    mu = np.asarray(params["mu"], float).copy()
    mu += delta[:, None] / float(len(bt.CAT_ORDER))
    category_error = np.max(np.abs(mu.sum(axis=1) - total_mean)) if len(field) else 0.0
    if not np.isfinite(category_error) or category_error > 1e-10:
        raise AssertionError(
            f"Category means do not sum to scalar arm skill; max error={category_error}"
        )
    common = (
        total_mean,
        total_mean,
        total_mean,
        total_mean,
        np.zeros(n_players, dtype=float),
        np.zeros(n_players, dtype=float),
        bt._r1_coefficients(coefficients, "coefficients_r1_high"),
        bt._r1_coefficients(coefficients, "coefficients_r1_midh"),
        bt._r1_coefficients(coefficients, "coefficients_r1_midl"),
        bt._r1_coefficients(coefficients, "coefficients_r1_low"),
        bt._r2_coefficients(coefficients, "coefficients_r2"),
        bt._r2_coefficients(coefficients, "coefficients_r2_6_30"),
        bt._r2_coefficients(coefficients, "coefficients_r2_30_up"),
        bt._r3_coefficients(coefficients, "coefficients_r3"),
        bt._r3_coefficients(coefficients, "coefficients_r3_mid"),
        bt._r3_coefficients(coefficients, "coefficients_r3_high"),
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


def v4_checkpoint(v4_dir: Path, event_key: str, v4_hash: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = v4_dir / "event_probabilities" / f"{event_key.replace(':', '_')}.parquet"
    meta_path = path.with_suffix(".meta.json")
    if not path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"Frozen v4 checkpoint is missing for {event_key}")
    meta = load_json(meta_path)
    if meta.get("config_hash") != v4_hash or meta.get("complete") is not True:
        raise ValueError(f"Frozen v4 checkpoint metadata is invalid for {event_key}")
    actual_hash = bt.sha256_file(path)
    if actual_hash != meta.get("checkpoint_sha256"):
        raise ValueError(f"Frozen v4 checkpoint hash mismatch for {event_key}")
    frame = pd.read_parquet(path)
    baseline = frame[frame["arm"].eq("baseline")].copy()
    if baseline.empty:
        raise ValueError(f"Frozen v4 checkpoint has no baseline rows for {event_key}")
    return baseline, {"path": str(path), "sha256": actual_hash, "meta": meta}


def validate_full_replication(
    actual: pd.DataFrame, expected: pd.DataFrame, event_key: str
) -> dict[str, Any]:
    keys = ["event_key", "market", "selection_id", "opponent_id"]
    left = actual.copy()
    right = expected.copy()
    for frame in (left, right):
        frame["selection_id"] = frame["selection_id"].map(bt.canonical_player_id)
        frame["opponent_id"] = frame["opponent_id"].map(bt.canonical_player_id)
    left = left.sort_values(keys, kind="mergesort").reset_index(drop=True)
    right = right.sort_values(keys, kind="mergesort").reset_index(drop=True)
    if len(left) != len(right) or not left[keys].equals(right[keys]):
        raise AssertionError(f"PROFILE_FULL opportunity keys do not equal v4 baseline for {event_key}")
    probability_diff = np.abs(
        left["probability"].to_numpy(float) - right["probability"].to_numpy(float)
    )
    push_diff = np.abs(
        left["push_probability"].to_numpy(float)
        - right["push_probability"].to_numpy(float)
    )
    max_probability = float(probability_diff.max(initial=0.0))
    max_push = float(push_diff.max(initial=0.0))
    bit_exact_probability = bool(
        np.array_equal(probability_diff, np.zeros_like(probability_diff))
    )
    bit_exact_push = bool(np.array_equal(push_diff, np.zeros_like(push_diff)))
    matchup = left["market"].eq("tournament_matchup").to_numpy()
    matchup_probability_bit_exact = bool(
        np.array_equal(
            probability_diff[matchup], np.zeros_like(probability_diff[matchup])
        )
    )
    matchup_push_bit_exact = bool(
        np.array_equal(push_diff[matchup], np.zeros_like(push_diff[matchup]))
    )
    differing_by_market: dict[str, dict[str, Any]] = {}
    for market in left["market"].drop_duplicates().astype(str):
        mask = left["market"].eq(market).to_numpy()
        market_probability = probability_diff[mask]
        market_push = push_diff[mask]
        differing_by_market[market] = {
            "rows": int(mask.sum()),
            "probability_differing_rows": int(np.count_nonzero(market_probability)),
            "push_probability_differing_rows": int(np.count_nonzero(market_push)),
            "max_probability_abs_diff": float(market_probability.max(initial=0.0)),
            "max_push_probability_abs_diff": float(market_push.max(initial=0.0)),
        }
    tolerance = 4.0 * np.finfo(np.float64).eps
    if (
        max_probability > tolerance
        or max_push > tolerance
        or not matchup_probability_bit_exact
        or not matchup_push_bit_exact
    ):
        raise AssertionError(
            f"PROFILE_FULL failed numerical v4 replication for {event_key}: "
            f"max probability diff={max_probability}, max push diff={max_push}"
        )
    return {
        "status": "tolerance_pass",
        "precanonical_status": (
            "bit_exact" if bit_exact_probability and bit_exact_push else "within_4eps"
        ),
        "rows": int(len(left)),
        "precanonical_max_probability_abs_diff": max_probability,
        "precanonical_max_push_probability_abs_diff": max_push,
        "precanonical_probability_differing_rows": int(np.count_nonzero(probability_diff)),
        "precanonical_push_probability_differing_rows": int(np.count_nonzero(push_diff)),
        "precanonical_differences_by_market": differing_by_market,
        "all_probability_bit_exact": bit_exact_probability,
        "all_push_probability_bit_exact": bit_exact_push,
        "tournament_matchup_probability_bit_exact": matchup_probability_bit_exact,
        "tournament_matchup_push_probability_bit_exact": matchup_push_bit_exact,
        "numerical_tolerance": tolerance,
    }


def validate_full_portfolio(
    actual_bets: pd.DataFrame,
    v4_bets_path: Path,
    events: Sequence[str],
    universe_events: Sequence[str],
) -> dict[str, Any]:
    """Fail closed unless scored PROFILE_FULL matches the frozen v4 portfolio."""

    expected_all = pd.read_parquet(v4_bets_path)
    expected = expected_all[
        expected_all["arm"].eq("baseline")
        & expected_all["challenger_name"].eq(bt.PRIMARY_ARM)
        & expected_all["line_policy_name"].eq("best_of_sharp")
        & expected_all["sizing_name"].eq("production_style")
        & expected_all["market_group"].eq("matchup")
        & expected_all["event_key"].isin(events)
    ].copy()
    if expected.empty:
        raise AssertionError("Frozen v4 has no baseline best-of-sharp matchup portfolio")

    identity = [
        "event_key",
        "market",
        "selection_id",
        "opponent_id",
        "book",
        "record_id",
    ]
    text_columns = [
        *identity,
        "selection_name",
        "opponent_name",
        "tie_rule",
        "outcome",
        "settlement_status",
    ]
    numeric_columns = [
        "close_odds",
        "probability",
        "push_probability",
        "push_probability_used",
        "pricing_probability",
        "implied_probability",
        "probability_edge",
        "ev_per_unit",
        "stake",
        "pnl",
        "return_amount",
        "paid_fraction",
    ]
    boolean_columns = ["is_graded"]

    def normalized(frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        for column in ("selection_id", "opponent_id"):
            output[column] = output[column].map(bt.canonical_player_id)
        for column in text_columns:
            output[column] = output[column].fillna("<NA>").astype(str)
        return output.sort_values(identity, kind="mergesort").reset_index(drop=True)

    expected = normalized(expected)
    challenger_audits: dict[str, dict[str, Any]] = {}
    for challenger in (PROFILE_HALF, PROFILE_NONE):
        actual = actual_bets[
            actual_bets["arm"].eq("baseline")
            & actual_bets["challenger_name"].eq(challenger)
            & actual_bets["line_policy_name"].eq("best_of_sharp")
            & actual_bets["sizing_name"].eq("production_style")
            & actual_bets["market_group"].eq("matchup")
            & actual_bets["event_key"].isin(events)
        ].copy()
        actual = normalized(actual)
        if len(actual) != len(expected):
            raise AssertionError(
                f"PROFILE_FULL scored row count differs from v4 for {challenger}: "
                f"actual={len(actual)}, expected={len(expected)}"
            )
        if not actual[identity].equals(expected[identity]):
            raise AssertionError(
                f"PROFILE_FULL scored keys/book/record differ from v4 for {challenger}"
            )
        for column in text_columns:
            if not actual[column].equals(expected[column]):
                raise AssertionError(
                    f"PROFILE_FULL scored text column differs from v4: {challenger}/{column}"
                )
        for column in numeric_columns:
            left = pd.to_numeric(actual[column], errors="coerce").to_numpy(float)
            right = pd.to_numeric(expected[column], errors="coerce").to_numpy(float)
            if not np.array_equal(left, right, equal_nan=True):
                difference = np.abs(left - right)
                raise AssertionError(
                    f"PROFILE_FULL scored numeric column differs from v4: "
                    f"{challenger}/{column}, max={float(np.nanmax(difference))}"
                )
        for column in boolean_columns:
            if not actual[column].fillna(False).astype(bool).equals(
                expected[column].fillna(False).astype(bool)
            ):
                raise AssertionError(
                    f"PROFILE_FULL scored boolean column differs from v4: {challenger}/{column}"
                )
        challenger_audits[challenger] = {
            "rows": int(len(actual)),
            "stake": float(actual["stake"].sum()),
            "pnl": float(actual["pnl"].sum()),
            "status": "bit_exact",
        }

    expected_stake = float(expected["stake"].sum())
    expected_pnl = float(expected["pnl"].sum())
    full_universe = set(events) == set(universe_events)
    if full_universe:
        if len(expected) != 2_808:
            raise AssertionError(f"Frozen full v4 portfolio row count changed: {len(expected)}")
        if not np.isclose(expected_stake, 3_056.19, atol=1e-9, rtol=0.0):
            raise AssertionError(f"Frozen full v4 stake changed: {expected_stake}")
        if not np.isclose(expected_pnl, -72.57, atol=1e-9, rtol=0.0):
            raise AssertionError(f"Frozen full v4 P&L changed: {expected_pnl}")
    return {
        "status": "bit_exact",
        "scope": "full_105_event_universe" if full_universe else "selected_event_slice",
        "events": int(len(set(events))),
        "rows": int(len(expected)),
        "stake": expected_stake,
        "pnl": expected_pnl,
        "validated_baseline_copies": challenger_audits,
        "full_universe_contract": {
            "rows": 2_808,
            "stake": 3_056.19,
            "pnl": -72.57,
            "asserted": full_universe,
        },
    }


def build_report(
    output_dir: Path,
    config: dict[str, Any],
    audit: pd.DataFrame,
    performance: pd.DataFrame,
    portfolio_replication: dict[str, Any],
) -> None:
    lines = [
        "# Course-profile sensitivity",
        "",
        "This replay changes only the legacy mean course-profile contribution embedded in v4 R1 skill. Course history and course experience remain fixed.",
        "",
        "## Arms",
        "",
        "- `profile_full`: frozen v4 skill.",
        "- `profile_half`: frozen v4 skill minus half of the exact legacy profile term.",
        "- `profile_none`: frozen v4 skill minus the full exact legacy profile term.",
        "",
        f"Events simulated: {config['event_count']}. Simulations per arm/event: {config['simulations']:,}.",
        "",
        "Raw PROFILE_FULL must match v4 opportunity keys, tournament-matchup probabilities and pushes bit-for-bit, and every all-market probability within four float64 epsilons. Saved and scored PROFILE_FULL rows are then canonicalized to the frozen v4 values.",
        "",
    ]
    if not audit.empty:
        lines.extend(
            [
                "## Replication",
                "",
                f"Tolerance-pass/canonicalized control events: {int(audit['replication_tolerance_pass'].sum())}/{len(audit)}.",
                "Maximum pre-canonicalization probability difference: "
                f"{audit['replication_max_probability_abs_diff'].max():.12g}.",
                "",
            ]
        )
    lines.extend(
        [
            "## PROFILE_FULL scored-portfolio gate",
            "",
            f"Status: `{portfolio_replication['status']}`. Scope: `{portfolio_replication['scope']}`.",
            f"Best-of-sharp production-style matchup portfolio: {portfolio_replication['rows']:,} bets, {portfolio_replication['stake']:.2f}u stake, {portfolio_replication['pnl']:.2f}u P&L.",
            "Keys, book, odds, probability, push, stake, settlement, and P&L are bit-exact to the corresponding frozen v4 event slice for both comparison copies.",
            "The full-run gate additionally requires exactly 2,808 bets, 3,056.19u stake, and -72.57u P&L.",
            "",
        ]
    )
    if not performance.empty:
        display = performance[
            [
                column
                for column in [
                    "challenger_name",
                    "line_policy_name",
                    "sizing_name",
                    "market_group",
                    "arm",
                    "bets",
                    "stake",
                    "pnl",
                    "roi",
                ]
                if column in performance
            ]
        ]
        # Keep the harness dependency-free; pandas.to_markdown requires the
        # optional `tabulate` package, which is not part of the production env.
        lines.extend(["## Performance (CSV)", "", "```csv", display.to_csv(index=False).strip(), "```", ""])
    atomic_text(output_dir / "REPORT.md", "\n".join(lines))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    v4_config_path = args.v4_dir / "run_config.json"
    v4_manifest_path = args.v4_dir / "manifest.json"
    for path in (
        v4_config_path,
        v4_manifest_path,
        LOW_COEFFICIENTS,
        HIGH_COEFFICIENTS,
        WEEKLY_MANIFEST,
        WEEKLY_MODEL_SOURCE,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    v4_config = load_json(v4_config_path)
    v4_manifest = load_json(v4_manifest_path)
    if v4_manifest.get("completed") is not True:
        raise ValueError("Frozen v4 backtest is not complete")
    if v4_manifest.get("config_hash") != v4_config.get("config_hash"):
        raise ValueError("Frozen v4 config and manifest disagree")
    if int(v4_config["simulations"]) != 50_000:
        raise ValueError("Expected the authoritative 50,000-simulation v4 replay")
    v4_bets_path = args.v4_dir / "bets.parquet"
    if not v4_bets_path.is_file():
        raise FileNotFoundError(v4_bets_path)
    v4_bets_hash = bt.sha256_file(v4_bets_path)
    expected_v4_bets_hash = v4_manifest.get("outputs", {}).get("bets.parquet", {}).get(
        "sha256"
    )
    if v4_bets_hash != expected_v4_bets_hash:
        raise ValueError(
            "Frozen v4 bets hash mismatch: "
            f"expected {expected_v4_bets_hash}, got {v4_bets_hash}"
        )

    events = selected_events(v4_config, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Validating frozen v4 inputs ...", flush=True)
    v4_hashes = validate_v4_inputs(v4_config)
    random_effects, re_audit, re_content = load_legacy_random_effects()
    weekly_manifest = load_json(WEEKLY_MANIFEST)
    low_model = load_json(LOW_COEFFICIENTS)
    high_model = load_json(HIGH_COEFFICIENTS)
    for name, path in (
        ("model_coeffs_low_skill.json", LOW_COEFFICIENTS),
        ("model_coeffs_high_skill.json", HIGH_COEFFICIENTS),
    ):
        expected_hash = weekly_manifest["outputs"][name]["sha256"]
        actual_hash = bt.sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Weekly coefficient manifest mismatch for {name}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
    low_coefficient = float(low_model["step4"]["course_adjustment"])
    high_coefficient = float(high_model["step4"]["course_adjustment"])
    synthetic_date = synthetic_date_from_manifest(weekly_manifest)

    v4_checkpoints: dict[str, pd.DataFrame] = {}
    checkpoint_audits: dict[str, dict[str, Any]] = {}
    for event_key in events:
        baseline, checkpoint_audit = v4_checkpoint(
            args.v4_dir, event_key, v4_config["config_hash"]
        )
        v4_checkpoints[event_key] = baseline
        checkpoint_audits[event_key] = checkpoint_audit

    auxiliary_hashes = {
        "legacy_course_re": re_audit["sha256"],
        "weekly_low_coefficients": bt.sha256_file(LOW_COEFFICIENTS),
        "weekly_high_coefficients": bt.sha256_file(HIGH_COEFFICIENTS),
        "weekly_manifest": bt.sha256_file(WEEKLY_MANIFEST),
        "weekly_model_source": bt.sha256_file(WEEKLY_MODEL_SOURCE),
        "v4_run_config": bt.sha256_file(v4_config_path),
        "v4_manifest": bt.sha256_file(v4_manifest_path),
        "v4_bets": v4_bets_hash,
        "harness": bt.sha256_file(Path(__file__)),
        **{
            f"v4_checkpoint::{event}": checkpoint_audits[event]["sha256"]
            for event in events
        },
    }
    config: dict[str, Any] = {
        "contract": "exact_legacy_profile_only_v1",
        "source_v4_config_hash": v4_config["config_hash"],
        "source_v4_dir": str(args.v4_dir.resolve()),
        "simulations": int(v4_config["simulations"]),
        "bootstrap": int(v4_config["bootstrap"]),
        "seed": int(v4_config["seed"]),
        "event_count": len(events),
        "event_keys": events,
        "arms": list(PROFILE_ARMS),
        "arm_definitions": {
            PROFILE_FULL: "v4",
            PROFILE_HALF: "v4 - 0.5 * exact_legacy_profile_contribution",
            PROFILE_NONE: "v4 - exact_legacy_profile_contribution",
        },
        "category_mean_shift": "scalar_delta_divided_equally_across_ott_app_arg_putt",
        "fixed_terms": [
            "course_history",
            "course_experience",
            "shot_variance",
            "course_shape",
            "odds",
            "seed",
            "all_non_profile_skill_components",
        ],
        "broader_course_bundle_arms": "omitted_not_exactly_decomposable",
        "legacy_profile_re": re_audit,
        "legacy_low_coefficient": low_coefficient,
        "legacy_high_coefficient": high_coefficient,
        "legacy_bucket_rule": "ema_adj_skill_est > 0.5",
        "legacy_arg_ema_quirk": "ema_11_sg_arg label uses span=21",
        "legacy_accuracy_centering_quirk": "whole event including appended synthetic latest rows",
        "synthetic_date": synthetic_date.date().isoformat(),
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "v4_input_hashes": v4_hashes,
        "auxiliary_hashes": auxiliary_hashes,
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, default=bt._json_default).encode("utf-8")
    ).hexdigest()
    config["config_hash"] = config_hash
    config_path = args.output_dir / "run_config.json"
    if config_path.exists():
        existing = load_json(config_path)
        if existing.get("config_hash") != config_hash:
            raise ValueError(
                f"Output directory has a different run_config: {config_path}. "
                "Choose a new output directory; existing artifacts were preserved."
            )
    else:
        bt.atomic_json(config_path, config)
        atomic_text(args.output_dir / "legacy_course_random_effects.csv", re_content.decode("utf-8"))

    profile_path = args.output_dir / "exact_legacy_profile_panel.parquet"
    profile_meta_path = args.output_dir / "exact_legacy_profile_panel.meta.json"
    if profile_path.exists() and profile_meta_path.exists():
        profile_meta = load_json(profile_meta_path)
        if profile_meta.get("config_hash") != config_hash:
            raise ValueError("Existing exact profile panel belongs to another run configuration")
        if profile_meta.get("sha256") != bt.sha256_file(profile_path):
            raise ValueError("Existing exact profile panel hash mismatch")
        profile_panel = pd.read_parquet(profile_path)
        profile_audit = profile_meta["audit"]
        print(f"Resuming exact profile panel ({len(profile_panel):,} rows) ...", flush=True)
    else:
        print("Reconstructing exact later-model course-profile panel ...", flush=True)
        profile_panel, profile_audit = build_exact_profile_panel(
            bt.DEFAULT_DB,
            events,
            random_effects,
            low_coefficient,
            high_coefficient,
            synthetic_date,
        )
        bt.atomic_parquet(profile_panel, profile_path)
        bt.atomic_json(
            profile_meta_path,
            {
                "config_hash": config_hash,
                "sha256": bt.sha256_file(profile_path),
                "audit": profile_audit,
            },
        )
        print(
            f"Profile panel: {len(profile_panel):,} rows in "
            f"{profile_audit['runtime_seconds']:.1f}s",
            flush=True,
        )
    if args.preflight_only:
        print(f"Preflight complete: {args.output_dir}", flush=True)
        return 0

    replacements = bt.load_name_replacements()
    factor_panel = pd.read_parquet(bt.DEFAULT_FACTOR_PANEL)
    factor_panel["start"] = pd.to_datetime(factor_panel["start"], errors="coerce").dt.normalize()
    reader = bt.HistoricalOddsCacheReader(bt.DEFAULT_ODDS_CACHE)
    matchup_lines = bt.normalize_matchup_lines(reader.matchups())
    outright_lines = bt.normalize_outright_lines(reader.outrights())
    raw_lines = pd.concat([matchup_lines, outright_lines], ignore_index=True, sort=False)
    rosters = bt.normalize_prediction_rosters(reader.predictions(), replacements)
    line_sets = bt.select_line_sets(raw_lines)
    universe_events = list(v4_config["event_keys"])
    universe_metadata = bt.read_event_metadata(bt.DEFAULT_DB, universe_events)
    metadata = {event: universe_metadata[event] for event in events if event in universe_metadata}
    missing_metadata = sorted(set(events) - set(metadata))
    if missing_metadata:
        raise ValueError(f"Local metadata missing for frozen v4 events: {missing_metadata}")

    union_ids = sorted(
        rosters[rosters["event_key"].isin(universe_events)]["dg_id"]
        .dropna()
        .astype(int)
        .unique()
    )
    missing_universe_metadata = sorted(set(universe_events) - set(universe_metadata))
    if missing_universe_metadata:
        raise ValueError(
            f"Local metadata missing for frozen v4 universe: {missing_universe_metadata}"
        )
    universe_cutoff = max(meta.start for meta in universe_metadata.values()) + pd.Timedelta(
        days=1
    )
    print(
        f"Loading frozen 105-event v4 history universe for {len(union_ids)} players ...",
        flush=True,
    )
    histories, model_histories = bt.load_player_histories(
        bt.DEFAULT_DB,
        union_ids,
        universe_cutoff,
    )
    r1_predictions = bt.load_r1_predictions(bt.DEFAULT_PREDS, replacements)
    source_crosswalk = bt.load_source_crosswalk(bt.DEFAULT_CROSSWALK)
    corr, chol = bt.load_correlation(bt.CORR_PATH)
    coefficients = bt.load_coefficients()
    week_latent_sd = float(bt.load_sim_input_literal("WEEK_LATENT_SD"))

    prepared_fields: dict[str, pd.DataFrame] = {}
    field_audits: dict[str, dict[str, Any]] = {}
    union_source_ids: set[str] = set(
        source_crosswalk.loc[
            source_crosswalk["dg_id"].isin(union_ids)
            & source_crosswalk["source_player_id"].ne(""),
            "source_player_id",
        ].astype(str)
    )
    field_export: list[pd.DataFrame] = []
    for event_key in events:
        field, audit = bt.build_event_field(
            rosters[rosters["event_key"].eq(event_key)],
            metadata[event_key],
            r1_predictions,
            model_histories,
            source_crosswalk,
            factor_panel[factor_panel["event_key"].eq(event_key)],
        )
        field = attach_profile(
            field, profile_panel[profile_panel["event_key"].eq(event_key)]
        )
        prepared_fields[event_key] = field
        audit.update(
            {
                "profile_applied_players": int(
                    field["profile_applied_to_r1_prediction"].sum()
                ),
                "profile_contribution_min": float(field["profile_in_baseline"].min()),
                "profile_contribution_max": float(field["profile_in_baseline"].max()),
                "profile_contribution_mean": float(field["profile_in_baseline"].mean()),
                "profile_contribution_abs_mean": float(
                    field["profile_in_baseline"].abs().mean()
                ),
            }
        )
        field_audits[event_key] = audit
        exported = field.copy()
        exported.insert(0, "event_key", event_key)
        field_export.append(exported)
    bt.atomic_parquet(
        pd.concat(field_export, ignore_index=True, sort=False),
        args.output_dir / "event_field_profile_attribution.parquet",
    )

    print(
        f"Loading v4 shot-variance histories for {len(union_source_ids)} source players ...",
        flush=True,
    )
    event_dates = bt.load_event_dates(bt.DEFAULT_DB)
    shot_stats, valid_shots = bt.load_shot_round_stats(
        bt.EXPORT,
        sorted(union_source_ids),
        event_dates,
        universe_cutoff,
    )
    print(f"Loaded {len(shot_stats):,} category-round rows from {valid_shots:,} shots.", flush=True)

    checkpoint_dir = args.output_dir / "event_probabilities"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    probability_pieces: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for event_index, event_key in enumerate(events, start=1):
        meta = metadata[event_key]
        checkpoint = checkpoint_dir / f"{event_key.replace(':', '_')}.parquet"
        checkpoint_meta = checkpoint.with_suffix(".meta.json")
        if checkpoint.exists() and checkpoint_meta.exists():
            saved = load_json(checkpoint_meta)
            if saved.get("config_hash") == config_hash and saved.get("complete") is True:
                if saved.get("checkpoint_sha256") != bt.sha256_file(checkpoint):
                    raise ValueError(f"Saved sensitivity checkpoint hash mismatch: {event_key}")
                probability_pieces.append(pd.read_parquet(checkpoint))
                audit_rows.append(saved["event_audit"])
                print(f"[{event_index}/{len(events)}] resume {event_key}", flush=True)
                continue

        started = time.perf_counter()
        field = prepared_fields[event_key]
        print(
            f"[{event_index}/{len(events)}] {meta.event_name} {meta.start.date()} "
            f"({len(field)} players)",
            flush=True,
        )
        shot_input = shot_stats[shot_stats["event_date"].lt(meta.start)]
        feature_ids = (
            field.loc[field["source_player_id"].ne(""), ["source_player_id"]]
            .drop_duplicates("source_player_id")
        )
        features, feature_prior = bt.build_dispersion_features(shot_input, feature_ids)
        mults, skews, course_audit = bt.course_shape(bt.DEFAULT_DB, meta, False)
        params, distribution_audit = bt.build_category_params(
            histories,
            field,
            meta.start,
            features,
            mults,
            skews,
        )
        base_category_error = float(
            np.max(
                np.abs(
                    np.asarray(params["mu"], float).sum(axis=1)
                    - field["skill_est"].to_numpy(float)
                )
            )
        )
        if base_category_error > 1e-10:
            raise AssertionError(
                f"v4 category means are not recentered to scalar skill: {base_category_error}"
            )
        cut_line, cut_rule = bt.cut_line_for_event(meta, len(field))
        event_seed = bt.stable_seed(event_key, int(v4_config["seed"]))
        event_lines = raw_lines[raw_lines["event_key"].eq(event_key)]
        deltas = scalar_arm_deltas(field)
        arm_probabilities: list[pd.DataFrame] = []
        probability_audit: dict[str, int] = {}
        for arm in PROFILE_ARMS:
            scores = simulate_scalar_skill_shift(
                params,
                field,
                chol,
                coefficients,
                deltas[arm],
                int(v4_config["simulations"]),
                event_seed,
                cut_line,
                week_latent_sd,
            )
            probabilities, probability_audit = bt.probabilities_for_lines(
                scores, field, event_lines, event_key, meta.start
            )
            probabilities["arm"] = arm
            arm_probabilities.append(probabilities)
            del scores
            gc.collect()
        event_probabilities = pd.concat(arm_probabilities, ignore_index=True, sort=False)
        replication = validate_full_replication(
            event_probabilities[event_probabilities["arm"].eq(PROFILE_FULL)],
            v4_checkpoints[event_key],
            event_key,
        )
        # Preserve the measured pre-canonicalization diagnostics above, then
        # make the saved control rows literally the frozen v4 values.  This
        # avoids presenting a one-ULP outright aggregation difference as a new
        # baseline while leaving the audit truth intact.
        canonical_full = v4_checkpoints[event_key].copy()
        canonical_full["arm"] = PROFILE_FULL
        event_probabilities = pd.concat(
            [
                canonical_full,
                event_probabilities[~event_probabilities["arm"].eq(PROFILE_FULL)],
            ],
            ignore_index=True,
            sort=False,
        )
        replication["saved_full_canonicalized_to_v4"] = True
        replication["saved_max_probability_abs_diff"] = 0.0
        replication["saved_max_push_probability_abs_diff"] = 0.0
        replication["status"] = "tolerance_pass_canonicalized"
        bt.atomic_parquet(event_probabilities, checkpoint)
        event_audit = {
            "event_key": event_key,
            "year": meta.year,
            "dg_event_id": meta.event_id,
            "event_name": meta.event_name,
            "start": meta.start,
            "seed": event_seed,
            "simulations": int(v4_config["simulations"]),
            "cut_line": cut_line,
            "cut_rule": cut_rule,
            "course_count": len(meta.course_counts),
            "course_shape": course_audit,
            "field": field_audits[event_key],
            "distribution": distribution_audit,
            "shot_variance_category_priors": feature_prior,
            "base_category_recenter_max_abs_error": base_category_error,
            "replication": replication,
            **probability_audit,
            "runtime_seconds": time.perf_counter() - started,
        }
        bt.atomic_json(
            checkpoint_meta,
            {
                "complete": True,
                "config_hash": config_hash,
                "checkpoint_sha256": bt.sha256_file(checkpoint),
                "event_audit": event_audit,
            },
        )
        probability_pieces.append(event_probabilities)
        audit_rows.append(event_audit)
        print(
            f"  {replication['status']} FULL replication; completed three arms in "
            f"{event_audit['runtime_seconds']:.1f}s",
            flush=True,
        )

    probabilities = pd.concat(probability_pieces, ignore_index=True, sort=False)
    audit = bt.normalize_event_audit_rows(audit_rows)
    if "replication.status" not in audit:
        raise AssertionError("Normalized event audit lost FULL replication status")
    audit["replication_tolerance_pass"] = audit["replication.status"].isin(
        ["tolerance_pass_canonicalized"]
    )
    audit["replication_max_probability_abs_diff"] = pd.to_numeric(
        audit["replication.precanonical_max_probability_abs_diff"], errors="coerce"
    )
    audit["replication_max_push_probability_abs_diff"] = pd.to_numeric(
        audit["replication.precanonical_max_push_probability_abs_diff"], errors="coerce"
    )
    bt.atomic_parquet(probabilities, args.output_dir / "probabilities.parquet")
    bt.atomic_parquet(audit, args.output_dir / "event_audit.parquet")

    coverage = pd.DataFrame(
        {
            "event_key": events,
            "start": [metadata[event].start for event in events],
            "eligible": True,
            "simulated": True,
            "exclusion_reason": "",
        }
    )
    scoring_outputs: dict[str, list[pd.DataFrame]] = {
        "bets": [],
        "attribution": [],
        "attribution_summary": [],
        "performance": [],
        "bootstrap": [],
    }
    baseline_probabilities = probabilities[probabilities["arm"].eq(PROFILE_FULL)]
    for challenger_index, challenger_name in enumerate((PROFILE_HALF, PROFILE_NONE)):
        challenger_probabilities = probabilities[
            probabilities["arm"].eq(challenger_name)
        ]
        for line_policy, selected_lines in line_sets.items():
            selected_lines = selected_lines[selected_lines["event_key"].isin(events)].copy()
            selected_lines = selected_lines.merge(
                coverage[["event_key", "start"]].rename(columns={"start": "event_start"}),
                on="event_key",
                how="left",
                validate="many_to_one",
            )
            for sizing_name in ("flat_1u", "production_style"):
                scored = bt.score_comparison(
                    selected_lines,
                    baseline_probabilities,
                    challenger_probabilities,
                    challenger_name,
                    line_policy,
                    sizing_name,
                    int(v4_config["bootstrap"]),
                    int(v4_config["seed"]) + 1000 * challenger_index,
                )
                for name, frame in scored.items():
                    scoring_outputs[name].append(frame)
    combined = {
        name: pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
        for name, frames in scoring_outputs.items()
    }
    portfolio_replication = validate_full_portfolio(
        combined["bets"], v4_bets_path, events, universe_events
    )
    bt.atomic_json(
        args.output_dir / "profile_full_portfolio_replication.json",
        portfolio_replication,
    )
    year_market = bt.build_year_market_summary(combined["bets"])
    bt.atomic_parquet(combined["bets"], args.output_dir / "bets.parquet")
    bt.atomic_parquet(combined["attribution"], args.output_dir / "attribution.parquet")
    combined["attribution_summary"].to_csv(
        args.output_dir / "attribution_summary.csv", index=False
    )
    combined["performance"].to_csv(args.output_dir / "performance.csv", index=False)
    combined["bootstrap"].to_csv(args.output_dir / "bootstrap.csv", index=False)
    year_market.to_csv(args.output_dir / "year_market_summary.csv", index=False)
    build_report(
        args.output_dir,
        config,
        audit,
        combined["performance"],
        portfolio_replication,
    )

    manifest = {
        **config,
        "completed": True,
        "valid_shots_loaded": int(valid_shots),
        "profile_panel_audit": profile_audit,
        "replication_all_tolerance_pass": bool(
            audit["replication_tolerance_pass"].all()
        ),
        "profile_full_portfolio_replication": portfolio_replication,
        "outputs": {
            path.name: {"sha256": bt.sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted(args.output_dir.iterdir())
            if path.is_file() and path.name != "manifest.json"
        },
    }
    bt.atomic_json(args.output_dir / "manifest.json", manifest)
    print(f"Course-profile sensitivity complete: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
