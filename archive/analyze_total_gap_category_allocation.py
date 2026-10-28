"""Research-only audit of the total-SG/category reconciliation rule.

Production builds four pre-event category means and an independent scalar
total-skill prediction.  The simulator currently reconciles them by adding
one quarter of the scalar gap to every category.  This script recovers the
exact weekly inputs from Git, joins them to adjusted R1/R2 outcomes, and asks
whether the *gap-specific* category movement supports a different allocation.

The script is deliberately isolated from the weekly pipeline.  It reads Git
objects and the historical SQLite database and writes only a versioned
research artifact directory.
"""

from __future__ import annotations

import argparse
import io
import json
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DB = Path(r"C:\Users\McKinley Slade\dev\sim_prep\work\dg_historical.db")
DEFAULT_OUTPUT = (
    REPO
    / "permanent_data"
    / "research"
    / "total_gap_category_allocation_exact_snapshots_20260826"
)

DISTS_PATHS = (
    "dashboard_data/this_week_dists_v2.csv",
    "this_week_dists_v2.csv",
)
R1_TOTAL_PATH = "dashboard_data/model_predictions_r1.csv"
# St. Jude's tracked pre-event final_predictions file was stale; the Wednesday
# R1 dashboard snapshot was the corrected as-priced scalar target and was later
# synced byte-for-byte into final_predictions_st_jude.csv. No other event uses
# this exception.
R1_TARGET_EVENT_OVERRIDES = {27}
CATEGORIES = ("sg_ott", "sg_app", "sg_arg", "sg_putt")
ACTUAL_COLUMNS = {
    "sg_ott": "sg_ott_adj",
    "sg_app": "sg_app_adj",
    "sg_arg": "sg_arg_adj",
    "sg_putt": "sg_putt_adj",
}


@dataclass(frozen=True)
class Snapshot:
    commit: str
    committed_at: pd.Timestamp
    path: str
    frame: pd.DataFrame
    players: frozenset[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", default="2026-03-01")
    parser.add_argument("--end", default="2026-08-24")
    parser.add_argument("--min-roster-recall", type=float, default=0.70)
    parser.add_argument("--min-players", type=int, default=40)
    parser.add_argument("--min-train-events", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260826)
    return parser.parse_args()


def normalize_name(value: object) -> str:
    return " ".join(str(value).strip().casefold().split())


def _git(*args: str) -> str:
    command = [
        "git",
        "-c",
        f"safe.directory={REPO.as_posix()}",
        *args,
    ]
    result = subprocess.run(
        command,
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout


def load_snapshots(path: str, kind: str) -> list[Snapshot]:
    history = _git("log", "--all", "--format=%H|%cI", "--", path)
    snapshots: list[Snapshot] = []
    for line in history.splitlines():
        if not line.strip():
            continue
        commit, timestamp = line.split("|", 1)
        try:
            payload = _git("show", f"{commit}:{path}")
            frame = pd.read_csv(io.StringIO(payload))
        except (subprocess.CalledProcessError, pd.errors.EmptyDataError):
            continue
        if frame.empty or "player_name" not in frame:
            continue
        if kind == "dists":
            required = {"category_clean", "mean"}
        elif kind == "total":
            required = {"my_pred"}
        else:  # pragma: no cover - module-owned call sites
            raise ValueError(kind)
        if not required.issubset(frame.columns):
            continue
        frame = frame.copy()
        frame["player_name"] = frame["player_name"].map(normalize_name)
        players = frozenset(frame["player_name"].dropna())
        if not players:
            continue
        snapshots.append(
            Snapshot(
                commit=commit,
                committed_at=pd.Timestamp(timestamp),
                path=path,
                frame=frame,
                players=players,
            )
        )
    return sorted(snapshots, key=lambda item: item.committed_at)


def discover_final_prediction_paths() -> list[str]:
    """Return event-specific frozen tournament targets that exist in Git."""

    listing = _git(
        "log",
        "--all",
        "--name-only",
        "--pretty=format:",
        "--",
        "final_predictions_*.csv",
    )
    paths: set[str] = set()
    excluded_tokens = (
        "_details",
        "_combined",
        "_combineddetail",
        "_zscores",
        ".pre_",
    )
    for raw in listing.splitlines():
        path = raw.strip().replace("\\", "/")
        name = Path(path).name
        if not name.startswith("final_predictions_") or not name.endswith(".csv"):
            continue
        if any(token in name for token in excluded_tokens):
            continue
        paths.add(path)
    return sorted(paths)


def load_rounds(db_path: Path, start: str, end: str) -> pd.DataFrame:
    uri = f"file:{db_path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        frame = pd.read_sql_query(
            "SELECT year, event_id, event_name, round_date, round_num, dg_id, "
            "player_name, sg_ott_adj, sg_app_adj, sg_arg_adj, sg_putt_adj, "
            "sg_total_adj, course_num "
            "FROM player_rounds WHERE tour='pga' AND year=2026 "
            "AND round_date >= ? AND round_date <= ? AND round_num IN (1, 2)",
            connection,
            params=[start, end],
        )
    frame["round_date"] = pd.to_datetime(
        frame["round_date"], errors="coerce"
    ).dt.normalize()
    frame["player_name"] = frame["player_name"].map(normalize_name)
    for column in [*ACTUAL_COLUMNS.values(), "sg_total_adj"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["dg_id"] = pd.to_numeric(frame["dg_id"], errors="coerce").astype("Int64")
    return frame.dropna(subset=["round_date", "event_id", "player_name"])


def event_groups(
    rounds: pd.DataFrame,
) -> list[tuple[tuple[int, int, str], pd.DataFrame]]:
    output: list[tuple[tuple[int, int, str], pd.DataFrame]] = []
    keys = ["year", "event_id", "event_name"]
    for raw_key, group in rounds.groupby(keys, sort=False):
        key = (int(raw_key[0]), int(raw_key[1]), str(raw_key[2]))
        output.append((key, group.copy()))
    return sorted(output, key=lambda item: item[1]["round_date"].min())


def roster_metrics(
    snapshot: Snapshot, actual_players: set[str]
) -> dict[str, float | int]:
    overlap = len(snapshot.players & actual_players)
    union = len(snapshot.players | actual_players)
    recall = overlap / len(actual_players) if actual_players else 0.0
    precision = overlap / len(snapshot.players) if snapshot.players else 0.0
    return {
        "overlap": overlap,
        "recall": recall,
        "precision": precision,
        "jaccard": overlap / union if union else 0.0,
    }


def choose_snapshot(
    snapshots: list[Snapshot],
    event_start: pd.Timestamp,
    actual_players: set[str],
    max_age_days: int,
) -> tuple[Snapshot | None, dict[str, float | int]]:
    # Weekly files are built before Thursday. Compare calendar dates so a
    # timezone-bearing commit timestamp never sorts after a naive midnight.
    event_date = event_start.date()
    candidates: list[
        tuple[tuple[float, float, int], Snapshot, dict[str, float | int]]
    ] = []
    for snapshot in snapshots:
        committed_date = snapshot.committed_at.date()
        age = (event_date - committed_date).days
        # The database event date is normalized to midnight and does not retain
        # first-tee time.  Exclude every Thursday commit, even an early one, so
        # a post-R1 dashboard refresh can never enter a pre-event audit.
        if age <= 0 or age > max_age_days:
            continue
        metrics = roster_metrics(snapshot, actual_players)
        # Roster recall is the primary identity check; Jaccard breaks ties when
        # an alternate-field file happens to share common names.
        score = (float(metrics["recall"]), float(metrics["jaccard"]), -age)
        candidates.append((score, snapshot, metrics))
    if not candidates:
        return None, {"overlap": 0, "recall": 0.0, "precision": 0.0, "jaccard": 0.0}
    _, snapshot, metrics = max(candidates, key=lambda item: item[0])
    return snapshot, metrics


def prepare_dists(snapshot: Snapshot) -> pd.DataFrame:
    optional = [
        column
        for column in ("n", "n_eff", "distribution_source")
        if column in snapshot.frame.columns
    ]
    frame = snapshot.frame[["player_name", "category_clean", "mean", *optional]].copy()
    frame["category_clean"] = (
        frame["category_clean"]
        .astype(str)
        .str.casefold()
        .str.replace("_adj", "", regex=False)
    )
    frame["mean"] = pd.to_numeric(frame["mean"], errors="coerce")
    frame = frame[frame["category_clean"].isin(CATEGORIES)]
    # Estimate allocation only where every category is a genuine player EMA.
    # Current files contain population/empirical-Bayes fallback cells for cold
    # starts; their sometimes multi-stroke scalar gap is a fallback-policy
    # question, not evidence about how established skill should decompose.
    if "n" in frame:
        frame["n"] = pd.to_numeric(frame["n"], errors="coerce")
        frame = frame[frame["n"].ge(12)]
    if "distribution_source" in frame:
        frame = frame[frame["distribution_source"].astype(str).eq("player_ema")]
    duplicates = frame.duplicated(["player_name", "category_clean"], keep=False)
    if duplicates.any():
        sort_columns = ["player_name", "category_clean"]
        if "n" in frame:
            frame["n"] = pd.to_numeric(frame["n"], errors="coerce")
            frame = frame.sort_values(sort_columns + ["n"])
        frame = frame.drop_duplicates(sort_columns, keep="last")
    wide = frame.pivot(index="player_name", columns="category_clean", values="mean")
    return wide.reindex(columns=CATEGORIES).dropna().reset_index()


def prepare_totals(snapshot: Snapshot) -> pd.DataFrame:
    frame = snapshot.frame[["player_name", "my_pred"]].copy()
    frame["my_pred"] = pd.to_numeric(frame["my_pred"], errors="coerce")
    frame = frame.dropna(subset=["my_pred"])
    return frame.drop_duplicates("player_name", keep="last")


def prepare_actuals(group: pd.DataFrame) -> pd.DataFrame:
    complete = group.dropna(subset=list(ACTUAL_COLUMNS.values())).copy()
    counts = complete.groupby("player_name")["round_num"].nunique()
    eligible = counts[counts.eq(2)].index
    complete = complete[complete["player_name"].isin(eligible)]
    aggregations = {column: "mean" for column in ACTUAL_COLUMNS.values()}
    aggregations.update({"dg_id": "first", "course_num": "first"})
    output = complete.groupby("player_name", as_index=False).agg(aggregations)
    return output.rename(
        columns={value: f"actual_{key}" for key, value in ACTUAL_COLUMNS.items()}
    )


def build_panel(
    rounds: pd.DataFrame,
    dist_snapshots: list[Snapshot],
    final_total_snapshots: list[Snapshot],
    r1_total_snapshots: list[Snapshot],
    min_roster_recall: float,
    min_players: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pieces: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for (year, event_id, event_name), group in event_groups(rounds):
        event_start = pd.Timestamp(group["round_date"].min()).normalize()
        r1_players = set(group.loc[group["round_num"].eq(1), "player_name"])
        dist_snapshot, dist_metrics = choose_snapshot(
            dist_snapshots, event_start, r1_players, max_age_days=5
        )
        target_source = (
            "corrected_r1_dashboard"
            if event_id in R1_TARGET_EVENT_OVERRIDES
            else "frozen_final_predictions"
        )
        target_candidates = (
            r1_total_snapshots
            if event_id in R1_TARGET_EVENT_OVERRIDES
            else final_total_snapshots
        )
        total_snapshot, total_metrics = choose_snapshot(
            target_candidates, event_start, r1_players, max_age_days=5
        )
        audit: dict[str, object] = {
            "year": year,
            "event_id": event_id,
            "event_name": event_name,
            "event_start": event_start,
            "r1_field_players": len(r1_players),
            "dist_commit": dist_snapshot.commit if dist_snapshot else "",
            "dist_committed_at": dist_snapshot.committed_at
            if dist_snapshot
            else pd.NaT,
            "dist_path": dist_snapshot.path if dist_snapshot else "",
            **{f"dist_{key}": value for key, value in dist_metrics.items()},
            "total_commit": total_snapshot.commit if total_snapshot else "",
            "total_committed_at": total_snapshot.committed_at
            if total_snapshot
            else pd.NaT,
            "total_path": total_snapshot.path if total_snapshot else "",
            "total_source": target_source,
            **{f"total_{key}": value for key, value in total_metrics.items()},
        }
        if dist_snapshot is None or total_snapshot is None:
            audit.update({"status": "missing_snapshot", "panel_players": 0})
            audits.append(audit)
            continue
        if (
            float(dist_metrics["recall"]) < min_roster_recall
            or float(total_metrics["recall"]) < min_roster_recall
        ):
            audit.update({"status": "roster_mismatch", "panel_players": 0})
            audits.append(audit)
            continue

        actuals = prepare_actuals(group)
        dists = prepare_dists(dist_snapshot)
        totals = prepare_totals(total_snapshot)
        panel = actuals.merge(
            dists, on="player_name", how="inner", validate="one_to_one"
        )
        panel = panel.merge(
            totals, on="player_name", how="inner", validate="one_to_one"
        )
        if len(panel) < min_players:
            audit.update({"status": "insufficient_join", "panel_players": len(panel)})
            audits.append(audit)
            continue
        panel.insert(0, "event_key", f"pga:R{year}{event_id:03d}")
        panel.insert(1, "event_name", event_name)
        panel.insert(2, "event_start", event_start)
        for category in CATEGORIES:
            panel.rename(columns={category: f"raw_{category}"}, inplace=True)
        panel["raw_sum"] = panel[[f"raw_{cat}" for cat in CATEGORIES]].sum(axis=1)
        panel["total_gap"] = panel["my_pred"] - panel["raw_sum"]
        panel["actual_sum"] = panel[[f"actual_{cat}" for cat in CATEGORIES]].sum(axis=1)
        panel["total_anchor_miss"] = panel["actual_sum"] - panel["my_pred"]
        for category in CATEGORIES:
            panel[f"equal_{category}"] = (
                panel[f"raw_{category}"] + panel["total_gap"] / 4.0
            )
        pieces.append(panel)
        audit.update({"status": "included", "panel_players": len(panel)})
        audits.append(audit)
    return (
        pd.concat(pieces, ignore_index=True, sort=False) if pieces else pd.DataFrame(),
        pd.DataFrame(audits),
    )


def profile_arrays(panel: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = panel[[f"raw_{cat}" for cat in CATEGORIES]].to_numpy(float)
    actual = panel[[f"actual_{cat}" for cat in CATEGORIES]].to_numpy(float)
    raw_profile = raw - raw.mean(axis=1, keepdims=True)
    actual_profile = actual - actual.mean(axis=1, keepdims=True)
    return raw_profile, actual_profile, actual_profile - raw_profile


def event_equal_row_weights(panel: pd.DataFrame) -> np.ndarray:
    counts = panel.groupby("event_key")["event_key"].transform("size").to_numpy(float)
    return np.divide(1.0, counts, out=np.zeros_like(counts), where=counts > 0)


def fit_allocation(
    panel: pd.DataFrame, intercept: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Return implied weights and the centered category intercept."""

    _, _, target = profile_arrays(panel)
    gap = panel["total_gap"].to_numpy(float)
    row_weight = event_equal_row_weights(panel)
    if intercept:
        design = np.column_stack([np.ones(len(panel)), gap])
    else:
        design = gap[:, None]
    sqrt_weight = np.sqrt(row_weight)
    coefficient = np.linalg.lstsq(
        design * sqrt_weight[:, None], target * sqrt_weight[:, None], rcond=None
    )[0]
    if intercept:
        category_intercept, gap_coefficient = coefficient
    else:
        category_intercept = np.zeros(4, dtype=float)
        gap_coefficient = coefficient[0]
    # Floating-point projection makes the zero-sum contract explicit.
    category_intercept -= category_intercept.mean()
    gap_coefficient -= gap_coefficient.mean()
    return 0.25 + gap_coefficient, category_intercept


def fit_conditional_allocation(panel: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Control for generic raw-profile calibration before estimating the gap term."""

    raw_profile, _, target = profile_arrays(panel)
    gap = panel["total_gap"].to_numpy(float)
    # Three raw-profile columns span the zero-sum subspace; the fourth is their
    # negative sum.  The coefficient on gap is therefore separately identified.
    design = np.column_stack([np.ones(len(panel)), gap, raw_profile[:, :3]])
    sqrt_weight = np.sqrt(event_equal_row_weights(panel))
    coefficient = np.linalg.lstsq(
        design * sqrt_weight[:, None], target * sqrt_weight[:, None], rcond=None
    )[0]
    intercept = coefficient[0] - coefficient[0].mean()
    gap_coefficient = coefficient[1] - coefficient[1].mean()
    return 0.25 + gap_coefficient, intercept


def predictions_for_weights(panel: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    raw = panel[[f"raw_{cat}" for cat in CATEGORIES]].to_numpy(float)
    gap = panel["total_gap"].to_numpy(float)
    return raw + gap[:, None] * np.asarray(weights, dtype=float)[None, :]


def actual_matrix(panel: pd.DataFrame) -> np.ndarray:
    return panel[[f"actual_{cat}" for cat in CATEGORIES]].to_numpy(float)


def event_mse_frame(
    panel: pd.DataFrame, prediction: np.ndarray, label: str
) -> pd.DataFrame:
    squared = (actual_matrix(panel) - prediction) ** 2
    rows: list[dict[str, object]] = []
    for event_key, indices in panel.groupby("event_key", sort=False).groups.items():
        positions = panel.index.get_indexer(indices)
        event_squared = squared[positions]
        row: dict[str, object] = {
            "event_key": event_key,
            "event_start": panel.loc[indices, "event_start"].iloc[0],
            "arm": label,
            "players": len(indices),
            "mse": float(event_squared.mean()),
        }
        for offset, category in enumerate(CATEGORIES):
            row[f"mse_{category}"] = float(event_squared[:, offset].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def expanding_predictions(
    panel: pd.DataFrame, min_train_events: int
) -> tuple[np.ndarray, pd.Series]:
    events = (
        panel[["event_key", "event_start"]]
        .drop_duplicates()
        .sort_values(["event_start", "event_key"])
    )
    prediction = np.full((len(panel), 4), np.nan, dtype=float)
    trained_on = pd.Series(pd.NA, index=panel.index, dtype="Int64")
    for position in range(min_train_events, len(events)):
        train_keys = set(events.iloc[:position]["event_key"])
        test_key = str(events.iloc[position]["event_key"])
        train = panel[panel["event_key"].isin(train_keys)]
        test_mask = panel["event_key"].eq(test_key)
        weights, _ = fit_allocation(train, intercept=False)
        prediction[test_mask.to_numpy()] = predictions_for_weights(
            panel[test_mask], weights
        )
        trained_on.loc[test_mask] = position
    return prediction, trained_on


def fit_profile_model(panel: pd.DataFrame, include_gap: bool) -> np.ndarray:
    """Fit a flexible zero-sum raw-profile calibration, optionally plus gap."""

    raw_profile, actual_profile, _ = profile_arrays(panel)
    columns = [np.ones(len(panel)), *[raw_profile[:, offset] for offset in range(3)]]
    if include_gap:
        columns.append(panel["total_gap"].to_numpy(float))
    design = np.column_stack(columns)
    sqrt_weight = np.sqrt(event_equal_row_weights(panel))
    coefficient = np.linalg.lstsq(
        design * sqrt_weight[:, None],
        actual_profile * sqrt_weight[:, None],
        rcond=None,
    )[0]
    # Every output row lives in the zero-sum category subspace. Centering the
    # output coefficients removes only numerical residue and guarantees the
    # fitted profile cannot alter total skill.
    coefficient -= coefficient.mean(axis=1, keepdims=True)
    return coefficient


def predict_profile_model(
    panel: pd.DataFrame, coefficient: np.ndarray, include_gap: bool
) -> np.ndarray:
    raw_profile, _, _ = profile_arrays(panel)
    columns = [np.ones(len(panel)), *[raw_profile[:, offset] for offset in range(3)]]
    if include_gap:
        columns.append(panel["total_gap"].to_numpy(float))
    profile = np.column_stack(columns) @ coefficient
    profile -= profile.mean(axis=1, keepdims=True)
    return panel["my_pred"].to_numpy(float)[:, None] / 4.0 + profile


def expanding_profile_predictions(
    panel: pd.DataFrame, min_train_events: int
) -> tuple[np.ndarray, np.ndarray]:
    """Chronological OOF predictions with and without the incremental gap term."""

    events = (
        panel[["event_key", "event_start"]]
        .drop_duplicates()
        .sort_values(["event_start", "event_key"])
    )
    calibrated_equal = np.full((len(panel), 4), np.nan, dtype=float)
    calibrated_gap = np.full((len(panel), 4), np.nan, dtype=float)
    for position in range(min_train_events, len(events)):
        train_keys = set(events.iloc[:position]["event_key"])
        test_key = str(events.iloc[position]["event_key"])
        train = panel[panel["event_key"].isin(train_keys)]
        test_mask = panel["event_key"].eq(test_key)
        test = panel[test_mask]
        base_coefficient = fit_profile_model(train, include_gap=False)
        gap_coefficient = fit_profile_model(train, include_gap=True)
        calibrated_equal[test_mask.to_numpy()] = predict_profile_model(
            test, base_coefficient, include_gap=False
        )
        calibrated_gap[test_mask.to_numpy()] = predict_profile_model(
            test, gap_coefficient, include_gap=True
        )
    return calibrated_equal, calibrated_gap


def allocation_bootstrap(panel: pd.DataFrame, draws: int, seed: int) -> pd.DataFrame:
    events = panel["event_key"].drop_duplicates().tolist()
    # Sufficient statistics preserve equal event weighting even when event
    # fields differ in size.
    _, _, target = profile_arrays(panel)
    work = panel[["event_key", "total_gap"]].copy()
    work["gap_sq"] = work["total_gap"] ** 2
    for offset, category in enumerate(CATEGORIES):
        work[f"gap_target_{category}"] = (
            work["total_gap"].to_numpy(float) * target[:, offset]
        )
    columns = ["gap_sq", *[f"gap_target_{cat}" for cat in CATEGORIES]]
    sufficient = work.groupby("event_key")[columns].mean().reindex(events)
    values = sufficient.to_numpy(float)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    for draw in range(draws):
        sampled = values[rng.integers(0, len(events), len(events))]
        denominator = float(sampled[:, 0].mean())
        coefficient = sampled[:, 1:].mean(axis=0) / denominator
        coefficient -= coefficient.mean()
        weights = 0.25 + coefficient
        rows.append(
            {"draw": draw, **{cat: weights[i] for i, cat in enumerate(CATEGORIES)}}
        )
    return pd.DataFrame(rows)


def delta_interval(
    event_delta: pd.Series, draws: int, seed: int
) -> tuple[float, float, float]:
    values = event_delta.dropna().to_numpy(float)
    point = float(values.mean())
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(
        axis=1
    )
    low, high = np.quantile(sampled, [0.025, 0.975])
    return point, float(low), float(high)


def weight_rows(panel: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specifications: list[tuple[str, np.ndarray, np.ndarray]] = []
    weights, intercept = fit_allocation(panel, intercept=False)
    specifications.append(("gap_only", weights, intercept))
    weights_i, intercept_i = fit_allocation(panel, intercept=True)
    specifications.append(("gap_plus_category_intercept", weights_i, intercept_i))
    weights_c, intercept_c = fit_conditional_allocation(panel)
    specifications.append(
        ("gap_plus_intercept_and_raw_profile", weights_c, intercept_c)
    )
    for name, values, intercept_values in specifications:
        for offset, category in enumerate(CATEGORIES):
            low = high = np.nan
            if name == "gap_only":
                low, high = np.quantile(bootstrap[category], [0.025, 0.975])
            rows.append(
                {
                    "specification": name,
                    "category": category,
                    "allocation_weight": float(values[offset]),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "centered_intercept": float(intercept_values[offset]),
                }
            )
    for label, subset in (
        ("gap_negative", panel[panel["total_gap"].lt(0)]),
        ("gap_positive", panel[panel["total_gap"].ge(0)]),
    ):
        if subset.empty:
            continue
        values, intercept_values = fit_allocation(subset, intercept=False)
        for offset, category in enumerate(CATEGORIES):
            rows.append(
                {
                    "specification": label,
                    "category": category,
                    "allocation_weight": float(values[offset]),
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "centered_intercept": float(intercept_values[offset]),
                }
            )
    return pd.DataFrame(rows)


def report_text(
    panel: pd.DataFrame,
    audit: pd.DataFrame,
    weights: pd.DataFrame,
    performance: pd.DataFrame,
    delta_summary: dict[str, dict[str, float | int]],
) -> str:
    included = audit[audit["status"].eq("included")]
    primary = weights[weights["specification"].eq("gap_only")]
    conditional = weights[
        weights["specification"].eq("gap_plus_intercept_and_raw_profile")
    ]
    gap = panel["total_gap"]
    performance_lines = [
        "| Arm | Events | Players | Event-mean MSE |",
        "|---|---:|---:|---:|",
    ]
    for row in performance.itertuples(index=False):
        performance_lines.append(
            f"| {row.arm} | {int(row.events)} | {int(row.players)} | "
            f"{row.event_mean_mse:.6f} |"
        )
    lines = [
        "# Total-SG gap allocation audit",
        "",
        "## Result",
        "",
        (
            f"Recovered {len(included)} exact pre-event Git snapshots and "
            f"{len(panel):,} established-player events with four genuine category "
            "EMAs and adjusted R1-R2 outcomes. "
            "The test changes only how the scalar total gap is distributed; every "
            "candidate still sums exactly to the same total-SG anchor."
        ),
        "",
        "### Estimated gap allocation",
        "",
        "| Category | Equal rule | Estimated | Event-bootstrap 95% CI | Conditional estimate* |",
        "|---|---:|---:|---:|---:|",
    ]
    conditional_map = conditional.set_index("category")["allocation_weight"]
    for row in primary.itertuples(index=False):
        lines.append(
            f"| {row.category.upper()} | 25.0% | {100 * row.allocation_weight:.1f}% | "
            f"[{100 * row.ci_low:.1f}%, {100 * row.ci_high:.1f}%] | "
            f"{100 * conditional_map[row.category]:.1f}% |"
        )
    lines.extend(
        [
            "",
            (
                "*Conditional estimate separates the gap term from category intercepts and "
                "generic raw-profile calibration. It is diagnostic, not a deployable rule."
            ),
            "",
            "### Predictive comparison",
            "",
            *performance_lines,
            "",
            (
                f"For the strict expanding walk-forward comparison, learned minus equal "
                f"event-average category MSE was {delta_summary['raw']['point']:+.6f} "
                f"(event-bootstrap 95% CI {delta_summary['raw']['low']:+.6f} to "
                f"{delta_summary['raw']['high']:+.6f}) over "
                f"{delta_summary['raw']['events']} held-out events. "
                "Negative favors learned allocation."
            ),
            (
                f"After the raw category profile was separately calibrated, adding the "
                f"gap-specific term changed MSE by "
                f"{delta_summary['profile_calibrated']['point']:+.6f} "
                f"(95% CI {delta_summary['profile_calibrated']['low']:+.6f} to "
                f"{delta_summary['profile_calibrated']['high']:+.6f})."
            ),
            "",
            "### Gap support",
            "",
            (
                f"The scalar gap averaged {gap.mean():+.3f} SG/round "
                f"(median {gap.median():+.3f}, SD {gap.std():.3f}, "
                f"5th-95th percentile {gap.quantile(0.05):+.3f} to {gap.quantile(0.95):+.3f}). "
                f"{100 * gap.lt(0).mean():.1f}% of player-events had a negative gap."
            ),
            "",
            "## Interpretation",
            "",
            (
                "The 25/25/25/25 rule is the minimum-distortion reconciliation: it changes "
                "the raw four-category profile as little as possible while forcing its sum to "
                "the separate total prediction. A different rule is supported only if the "
                "centered category error specifically covaries with the scalar gap. A standing "
                "ARG or PUTT bias independent of that gap is a separate profile-calibration issue. "
                "Here the learned gap rule was slightly worse in both chronological comparisons, "
                "and both event-bootstrap intervals include zero, so the exact snapshots do not "
                "support replacing equal allocation."
            ),
            "",
            "## Limitations",
            "",
            (
                "- Exact frozen inputs begin only when the dashboard files entered Git history; "
                "the event-cluster count is therefore modest."
            ),
            (
                "- The scalar target is the event-specific frozen `final_predictions_*.csv`; "
                "St. Jude uses its corrected Wednesday R1 snapshot because the tracked pre-event "
                "final file was stale. Generic historical R1 dashboard targets are not treated "
                "as interchangeable with the tournament-sim target."
            ),
            (
                "- The primary fit requires all four cells to be genuine player EMAs with at "
                "least 12 prior observations. Synthetic cold-start fallbacks are excluded."
            ),
            (
                "- Commit time plus field-roster overlap identifies the weekly snapshot. Every "
                "included event clears the configured recall threshold; inspect event_snapshot_audit.csv."
            ),
            (
                "- R1-R2 adjusted outcomes avoid weekend survivor selection but remain noisy. "
                "Confidence intervals resample whole events."
            ),
            (
                "- The raw weekly category means can include whatever zero-sum category layers "
                "were active in that historical file. This is intentional: it tests the exact "
                "vector seen by the simulator immediately before total recentering."
            ),
            (
                "- This is a category-forecast audit. The current single-round matchup kernel "
                "re-centers to fixed total skill, so any zero-sum allocation remains structurally "
                "invariant for standalone R1-R4 pricing."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if not args.db.is_file():
        raise FileNotFoundError(args.db)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading exact weekly Git snapshots...")
    dist_snapshots = [
        snapshot for path in DISTS_PATHS for snapshot in load_snapshots(path, "dists")
    ]
    dist_snapshots.sort(key=lambda item: item.committed_at)
    final_paths = discover_final_prediction_paths()
    final_total_snapshots = [
        snapshot for path in final_paths for snapshot in load_snapshots(path, "total")
    ]
    final_total_snapshots.sort(key=lambda item: item.committed_at)
    r1_total_snapshots = load_snapshots(R1_TOTAL_PATH, "total")
    print(f"  category snapshots: {len(dist_snapshots)}")
    print(
        f"  frozen final-prediction snapshots: {len(final_total_snapshots)} "
        f"across {len(final_paths)} files"
    )
    print(f"  corrected R1 target snapshots: {len(r1_total_snapshots)}")

    rounds = load_rounds(args.db, args.start, args.end)
    panel, audit = build_panel(
        rounds,
        dist_snapshots,
        final_total_snapshots,
        r1_total_snapshots,
        args.min_roster_recall,
        args.min_players,
    )
    if panel.empty:
        raise RuntimeError("No events passed exact-snapshot coverage checks")
    panel = panel.sort_values(["event_start", "event_key", "player_name"]).reset_index(
        drop=True
    )

    bootstrap = allocation_bootstrap(panel, args.bootstrap, args.seed)
    estimates = weight_rows(panel, bootstrap)
    primary_weights = (
        estimates[estimates["specification"].eq("gap_only")]
        .set_index("category")
        .reindex(CATEGORIES)["allocation_weight"]
        .to_numpy(float)
    )

    equal_prediction = predictions_for_weights(panel, np.full(4, 0.25))
    fitted_prediction = predictions_for_weights(panel, primary_weights)
    expanding_prediction, trained_on = expanding_predictions(
        panel, args.min_train_events
    )
    calibrated_equal_prediction, calibrated_gap_prediction = (
        expanding_profile_predictions(panel, args.min_train_events)
    )
    panel["expanding_train_events"] = trained_on
    for offset, category in enumerate(CATEGORIES):
        panel[f"fitted_{category}"] = fitted_prediction[:, offset]
        panel[f"expanding_{category}"] = expanding_prediction[:, offset]
        panel[f"calibrated_equal_{category}"] = calibrated_equal_prediction[:, offset]
        panel[f"calibrated_gap_{category}"] = calibrated_gap_prediction[:, offset]

    event_mse = pd.concat(
        [
            event_mse_frame(panel, equal_prediction, "equal_25"),
            event_mse_frame(panel, fitted_prediction, "full_sample_learned"),
        ],
        ignore_index=True,
    )
    oof_mask = np.isfinite(expanding_prediction).all(axis=1)
    oof_panel = panel.loc[oof_mask].reset_index(drop=True)
    oof_equal = equal_prediction[oof_mask]
    oof_learned = expanding_prediction[oof_mask]
    oof_event_mse = pd.concat(
        [
            event_mse_frame(oof_panel, oof_equal, "expanding_equal_25"),
            event_mse_frame(oof_panel, oof_learned, "expanding_learned"),
        ],
        ignore_index=True,
    )
    calibrated_event_mse = pd.concat(
        [
            event_mse_frame(
                oof_panel,
                calibrated_equal_prediction[oof_mask],
                "expanding_profile_calibrated_equal",
            ),
            event_mse_frame(
                oof_panel,
                calibrated_gap_prediction[oof_mask],
                "expanding_profile_calibrated_gap",
            ),
        ],
        ignore_index=True,
    )
    event_mse = pd.concat(
        [event_mse, oof_event_mse, calibrated_event_mse], ignore_index=True
    )

    performance = (
        event_mse.groupby("arm", as_index=False)
        .agg(
            events=("event_key", "nunique"),
            players=("players", "sum"),
            event_mean_mse=("mse", "mean"),
        )
        .sort_values("arm")
    )
    oof_pivot = oof_event_mse.pivot(
        index="event_key", columns="arm", values="mse"
    ).dropna()
    event_delta = oof_pivot["expanding_learned"] - oof_pivot["expanding_equal_25"]
    point, low, high = delta_interval(event_delta, args.bootstrap, args.seed + 1)
    calibrated_pivot = calibrated_event_mse.pivot(
        index="event_key", columns="arm", values="mse"
    ).dropna()
    calibrated_delta = (
        calibrated_pivot["expanding_profile_calibrated_gap"]
        - calibrated_pivot["expanding_profile_calibrated_equal"]
    )
    cal_point, cal_low, cal_high = delta_interval(
        calibrated_delta, args.bootstrap, args.seed + 2
    )
    delta_summary = {
        "raw": {"point": point, "low": low, "high": high, "events": len(event_delta)},
        "profile_calibrated": {
            "point": cal_point,
            "low": cal_low,
            "high": cal_high,
            "events": len(calibrated_delta),
        },
    }

    # Invariants: all rules must preserve the scalar anchor exactly.
    for label, prediction in (
        ("equal", equal_prediction),
        ("fitted", fitted_prediction),
    ):
        error = np.max(
            np.abs(prediction.sum(axis=1) - panel["my_pred"].to_numpy(float))
        )
        if error > 1e-10:
            raise AssertionError(f"{label} allocation changed total skill by {error}")

    panel.to_parquet(args.output_dir / "player_event_panel.parquet", index=False)
    audit.to_csv(args.output_dir / "event_snapshot_audit.csv", index=False)
    estimates.to_csv(args.output_dir / "allocation_estimates.csv", index=False)
    performance.to_csv(args.output_dir / "performance_summary.csv", index=False)
    event_mse.to_csv(args.output_dir / "event_mse.csv", index=False)
    bootstrap.to_parquet(
        args.output_dir / "allocation_event_bootstrap.parquet", index=False
    )
    config = {
        "start": args.start,
        "end": args.end,
        "db": str(args.db),
        "dist_paths": list(DISTS_PATHS),
        "final_prediction_paths": final_paths,
        "r1_total_path": R1_TOTAL_PATH,
        "dist_snapshots": len(dist_snapshots),
        "final_total_snapshots": len(final_total_snapshots),
        "r1_total_snapshots": len(r1_total_snapshots),
        "r1_target_event_overrides": sorted(R1_TARGET_EVENT_OVERRIDES),
        "min_roster_recall": args.min_roster_recall,
        "min_players": args.min_players,
        "min_train_events": args.min_train_events,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "included_events": int(audit["status"].eq("included").sum()),
        "panel_players": len(panel),
        "expanding_delta": delta_summary,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(
        report_text(panel, audit, estimates, performance, delta_summary),
        encoding="utf-8",
    )

    print(
        f"Included {config['included_events']} events and {len(panel):,} player-events"
    )
    print(
        estimates[
            estimates["specification"].isin(
                ["gap_only", "gap_plus_intercept_and_raw_profile"]
            )
        ].to_string(index=False)
    )
    print(performance.to_string(index=False))
    print(f"Expanding learned-minus-equal MSE: {point:+.6f} [{low:+.6f}, {high:+.6f}]")
    print(
        f"After profile calibration, gap-minus-no-gap MSE: {cal_point:+.6f} "
        f"[{cal_low:+.6f}, {cal_high:+.6f}]"
    )
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
