"""Extended leakage-safe approach distance x lie course-fit study.

This is an archived research harness, not a weekly-pipeline input.  It rebuilds
the original 2026-08 inline analysis from the local R2 export and extends it
with:

* full-history 8x2 and 4x2 profiles with k={100, 200, 400, 800};
* 40/80/160-round histories, a smooth cell ridge, and a fixed low-rank basis;
* strict prior physical-course mixes reconstructed from tee coordinates;
* event-average and player-specific realized-mix oracle ceilings;
* a leakage-safe player/course mix blend;
* expanding-year OOS tests against APP event signal and production R1 residual;
* event-cluster uncertainty, year stability, and a pre-specified distinctive-
  course subgroup (prior course mix L1 distance from tour mix >= 0.15).

No production files are read-write.  The only optional write is the JSON result
path supplied with ``--output``.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


DEFAULT_EXPORT = Path(
    r"C:\Users\McKinley Slade\dev\golf_scraping\output\export"
)
DEFAULT_RESIDS = Path(
    r"C:\Users\McKinley Slade\dev\sim_prep\work\sg_preds_resids.csv"
)
EDGES = np.array([50, 75, 100, 125, 150, 175, 200, 225], dtype=float)
TEST_YEARS = (2023, 2024, 2025, 2026)
KS = (100, 200, 400, 800)
DISTINCTIVE_L1 = 0.15

# Four fixed smooth factors: linear and quadratic distance, lie, and their
# interaction.  Fixed factors avoid fitting PCA loadings on the test period.
MIDS = np.array([62.5, 87.5, 112.5, 137.5, 162.5, 187.5, 212.5, 250.0])
LOWRANK = []
for distance in MIDS:
    z = (distance - 170.0) / 50.0
    for bad_lie in (0.0, 1.0):
        LOWRANK.append([z, z * z - 1.0, bad_lie - 0.5, z * (bad_lie - 0.5)])
LOWRANK = np.asarray(LOWRANK, dtype=float)

# Adjacent-distance smoothness, separately for clean and bad lies.
SMOOTH_D = np.zeros((14, 16), dtype=float)
_row = 0
for lie in (0, 1):
    for distance_bin in range(7):
        SMOOTH_D[_row, distance_bin * 2 + lie] = -1.0
        SMOOTH_D[_row, (distance_bin + 1) * 2 + lie] = 1.0
        _row += 1
SMOOTH_PENALTY = SMOOTH_D.T @ SMOOTH_D
MAP_4X2 = np.array([(cell // 2 // 2) * 2 + cell % 2 for cell in range(16)])


def norm_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def geometry_rmse(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return float("inf")
    okay = np.isfinite(left) & np.isfinite(right) & ((np.abs(left) + np.abs(right)) > 1)
    if okay.sum() < 30:
        return float("inf")
    return float(np.sqrt(np.mean((left[okay] - right[okay]) ** 2)))


def arrays_from_group(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.zeros(16, dtype=float)
    sums = np.zeros(16, dtype=float)
    sums_sq = np.zeros(16, dtype=float)
    index = group["cell"].to_numpy(int)
    counts[index] = group["size"].to_numpy(float)
    sums[index] = group["sum"].to_numpy(float)
    sums_sq[index] = group["sum_sq"].to_numpy(float)
    return counts, sums, sums_sq


def history_arrays(
    items: list[tuple[float, np.ndarray, np.ndarray, float]],
    max_rounds: int | None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    counts = np.zeros(16, dtype=float)
    sums = np.zeros(16, dtype=float)
    rounds = 0.0
    signal_sum = 0.0
    iterable = items if max_rounds is None else reversed(items)
    for n_rounds, cell_counts, cell_sums, app_signal_sum in iterable:
        counts += cell_counts
        sums += cell_sums
        rounds += n_rounds
        signal_sum += app_signal_sum
        if max_rounds is not None and rounds >= max_rounds:
            break
    return counts, sums, rounds, signal_sum


def exact_profile(
    counts: np.ndarray,
    sums: np.ndarray,
    base: np.ndarray,
    k: float,
    mapping: np.ndarray | None = None,
) -> np.ndarray:
    residual_sums = sums - counts * base
    if mapping is None:
        return base + residual_sums / (counts + k)
    groups = int(mapping.max()) + 1
    group_counts = np.bincount(mapping, weights=counts, minlength=groups)
    group_residuals = np.bincount(mapping, weights=residual_sums, minlength=groups)
    return base + (group_residuals / (group_counts + k))[mapping]


def smooth_profile(
    counts: np.ndarray, sums: np.ndarray, base: np.ndarray, k: float
) -> np.ndarray:
    residual_sums = sums - counts * base
    matrix = np.diag(counts + k) + 4.0 * k * SMOOTH_PENALTY
    try:
        adjustment = np.linalg.solve(matrix, residual_sums)
    except np.linalg.LinAlgError:
        adjustment = np.zeros(16)
    return base + adjustment


def lowrank_profile(
    counts: np.ndarray, sums: np.ndarray, base: np.ndarray, k: float
) -> np.ndarray:
    residual_sums = sums - counts * base
    matrix = LOWRANK.T @ (counts[:, None] * LOWRANK) + k * np.eye(LOWRANK.shape[1])
    try:
        coefficients = np.linalg.solve(matrix, LOWRANK.T @ residual_sums)
    except np.linalg.LinAlgError:
        coefficients = np.zeros(LOWRANK.shape[1])
    return base + LOWRANK @ coefficients


def profile_specs() -> list[tuple[str, str, int | None, int]]:
    specs: list[tuple[str, str, int | None, int]] = []
    for k in KS:
        specs.append((f"exact8_full_k{k}", "exact8", None, k))
        specs.append((f"exact4_full_k{k}", "exact4", None, k))
    for rounds in (40, 80, 160):
        specs.append((f"exact8_r{rounds}_k400", "exact8", rounds, 400))
        specs.append((f"exact4_r{rounds}_k400", "exact4", rounds, 400))
    specs.append(("smooth8_full_k400", "smooth", None, 400))
    specs.append(("lowrank4_full_k400", "lowrank", None, 400))
    return specs


def build_panel(export_root: Path, residual_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    date_rows = pd.read_csv(
        residual_path, usecols=["year", "event_id", "round_date"]
    )
    date_rows["event_id"] = pd.to_numeric(date_rows["event_id"], errors="coerce")
    date_rows["date"] = pd.to_datetime(date_rows["round_date"], errors="coerce")
    date_rows = (
        date_rows.dropna(subset=["event_id", "date"])
        .groupby(["year", "event_id"], as_index=False)["date"]
        .min()
    )
    date_rows["event_id"] = date_rows["event_id"].astype(int)

    metadata = pd.read_parquet(export_root / "events.parquet")
    metadata = metadata[metadata["tour"].eq("pga")].copy()
    metadata["event_id"] = metadata["event_key"].str[-3:].astype(int)
    metadata = metadata.merge(
        date_rows,
        left_on=["season", "event_id"],
        right_on=["year", "event_id"],
        how="left",
    ).drop(columns="year")
    metadata = (
        metadata.dropna(subset=["date"])
        .drop_duplicates("event_key")
        .set_index("event_key")
    )

    geometry_parts = []
    signal_parts = []
    shot_parts = []
    all_app_shots = 0
    valid_complete_app_shots = 0
    eligible_cell_app_shots = 0
    for year in range(2020, 2027):
        geometry_parts.append(
            pd.read_parquet(
                export_root / f"pga_{year}_hole_geometry.parquet",
                columns=["event_key", "hole_no", "tee_x", "tee_y"],
            )
        )
        signals = pd.read_parquet(
            export_root / f"pga_{year}_player_round_signals.parquet",
            columns=[
                "event_key", "round_no", "source_player_id", "player_name",
                "complete", "sg_app_signal",
            ],
        )
        signal_parts.append(signals[signals["complete"].eq(1)])

        shots = pd.read_parquet(
            export_root / f"pga_{year}_shot_sg.parquet",
            columns=[
                "event_key", "source_player_id", "sg_category", "sg_valid",
                "complete_hole", "start_lie", "start_distance_yd", "sg_raw",
                "end_lie",
            ],
        )
        app_mask = shots["sg_category"].eq("app")
        valid_mask = (
            app_mask
            & shots["sg_valid"].eq(1)
            & shots["complete_hole"].eq(1)
            & shots["sg_raw"].notna()
        )
        eligible_mask = (
            valid_mask
            & shots["start_lie"].isin(["tee", "fairway", "rough", "sand", "recovery"])
            & shots["start_distance_yd"].ge(50)
        )
        all_app_shots += int(app_mask.sum())
        valid_complete_app_shots += int(valid_mask.sum())
        eligible_cell_app_shots += int(eligible_mask.sum())
        shots = shots[eligible_mask].copy()
        distance_bin = np.clip(
            np.searchsorted(EDGES, shots["start_distance_yd"], side="right") - 1,
            0,
            7,
        )
        lie_bin = np.where(shots["start_lie"].isin(["tee", "fairway"]), 0, 1)
        shots["cell"] = (distance_bin * 2 + lie_bin).astype("int8")
        # Match the R2 predictive APP transform: discount rare long hole-outs.
        shots["y"] = shots["sg_raw"] - np.where(
            shots["end_lie"].eq("hole") & shots["start_distance_yd"].gt(50),
            1.0,
            0.0,
        )
        shots["y2"] = shots["y"] ** 2
        shot_parts.append(shots[["event_key", "source_player_id", "cell", "y", "y2"]])

    geometry = pd.concat(geometry_parts, ignore_index=True)
    geometry = geometry[
        geometry["event_key"].isin(metadata.index) & geometry["hole_no"].between(1, 18)
    ]
    signature_frame = (
        geometry.groupby(["event_key", "hole_no"])[["tee_x", "tee_y"]]
        .mean()
        .unstack()
        .reindex(columns=pd.MultiIndex.from_product([["tee_x", "tee_y"], range(1, 19)]))
    )
    signatures = {event: row.to_numpy(float) for event, row in signature_frame.iterrows()}

    signals = pd.concat(signal_parts, ignore_index=True)
    signals = signals[signals["event_key"].isin(metadata.index)].copy()
    signal_events = (
        signals.groupby(["event_key", "source_player_id"], as_index=False)
        .agg(
            player_name=("player_name", "last"),
            nrounds=("round_no", "nunique"),
            target=("sg_app_signal", "mean"),
            signal_sum=("sg_app_signal", "sum"),
        )
    )
    event_rounds = signal_events.groupby("event_key")["nrounds"].sum().to_dict()
    players_by_event = {event: group for event, group in signal_events.groupby("event_key")}

    app = pd.concat(shot_parts, ignore_index=True)
    app = app[app["event_key"].isin(metadata.index)]
    grouped = (
        app.groupby(["event_key", "source_player_id", "cell"])
        .agg(size=("y", "size"), sum=("y", "sum"), sum_sq=("y2", "sum"))
        .reset_index()
    )
    player_event: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    by_event_cells: dict[str, list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(list)
    for (event, player), group in grouped.groupby(["event_key", "source_player_id"]):
        arrays = arrays_from_group(group)
        key = (event, str(player))
        player_event[key] = arrays
        by_event_cells[event].append((str(player), *arrays))
    event_cells = (
        app.groupby(["event_key", "cell"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=range(16), fill_value=0)
    )
    shot_count = int(len(app))
    del app, grouped, shot_parts

    event_order = [
        event for event in metadata.sort_values("date").index
        if event in players_by_event and event in event_cells.index
    ]
    events_by_date: dict[pd.Timestamp, list[str]] = defaultdict(list)
    for event in event_order:
        events_by_date[metadata.loc[event, "date"]].append(event)

    specs = profile_specs()
    state: dict[str, list[tuple[float, np.ndarray, np.ndarray, float]]] = defaultdict(list)
    prior_courses: list[tuple[np.ndarray | None, np.ndarray, float, str]] = []
    global_counts = np.zeros(16, dtype=float)
    global_sums = np.zeros(16, dtype=float)
    global_rounds = 0.0
    rows: list[dict] = []
    mix_rows: list[dict] = []

    for date in sorted(events_by_date):
        # Snapshot every concurrent event before adding any event on this date.
        for event in events_by_date[date]:
            current_counts = event_cells.loc[event].to_numpy(float)
            current_rounds = float(event_rounds[event])
            if global_counts.sum() > 0:
                global_mix = global_counts / global_counts.sum()
                global_expected = global_counts.sum() / max(global_rounds, 1.0)
                global_overall = global_sums.sum() / global_counts.sum()
                global_cell_mean = np.divide(
                    global_sums,
                    global_counts,
                    out=np.full(16, global_overall),
                    where=global_counts > 0,
                )
            else:
                global_mix = np.ones(16) / 16
                global_expected = 12.0
                global_overall = 0.0
                global_cell_mean = np.zeros(16)
            global_cell_deviation = global_cell_mean - global_overall

            signature = signatures.get(event)
            matches = [
                item for item in prior_courses
                if geometry_rmse(signature, item[0]) <= 25.0
            ]
            if matches:
                prior_counts = sum((item[1] for item in matches), np.zeros(16))
                prior_rounds = sum(item[2] for item in matches)
                physical_mix = prior_counts / prior_counts.sum()
                physical_expected = prior_counts.sum() / max(prior_rounds, 1.0)
            else:
                physical_mix = global_mix
                physical_expected = global_expected
            current_mix = current_counts / current_counts.sum()
            current_expected = current_counts.sum() / max(current_rounds, 1.0)
            distinctive_l1 = float(np.abs(physical_mix - global_mix).sum())
            mix_rows.append(
                {
                    "event_key": event,
                    "year": int(metadata.loc[event, "season"]),
                    "n_physical_editions": len(matches),
                    "physical_l1_error": float(np.abs(physical_mix - current_mix).sum()),
                    "global_l1_error": float(np.abs(global_mix - current_mix).sum()),
                    "distinctive_l1": distinctive_l1,
                }
            )

            for target_row in players_by_event[event].itertuples(index=False):
                player = str(target_row.source_player_id)
                current_n, current_y, current_y2 = player_event.get(
                    (event, player),
                    (np.zeros(16), np.zeros(16), np.zeros(16)),
                )
                items = state[player]
                full_n, full_y, full_rounds, _ = history_arrays(items, None)
                base_n, _, base_rounds, base_signal_sum = history_arrays(items, 40)
                del base_n
                total_history = full_n.sum()
                player_overall = (
                    full_y.sum() / total_history if total_history > 0 else global_overall
                )
                base = player_overall + global_cell_deviation

                player_prior_mix = (
                    (full_n + 1000.0 * global_mix) / (total_history + 1000.0)
                    if total_history > 0
                    else global_mix
                )
                ratio = np.divide(
                    player_prior_mix,
                    np.maximum(global_mix, 1e-6),
                    out=np.ones(16),
                )
                personalized_mix = physical_mix * np.sqrt(np.clip(ratio, 0.25, 4.0))
                personalized_mix /= personalized_mix.sum()

                if current_n.sum() > 0:
                    player_oracle_mix = current_n / current_n.sum()
                    player_oracle_expected = current_n.sum() / max(float(target_row.nrounds), 1.0)
                else:
                    player_oracle_mix = current_mix
                    player_oracle_expected = current_expected

                out = {
                    "event_key": event,
                    "source_player_id": player,
                    "player_name": target_row.player_name,
                    "year": int(metadata.loc[event, "season"]),
                    "target": float(target_row.target),
                    "app_base": base_signal_sum / base_rounds if base_rounds > 0 else 0.0,
                    "hist_shots": float(total_history),
                    "n_physical_editions": len(matches),
                    "distinctive_l1": distinctive_l1,
                    "distinctive": bool(len(matches) and distinctive_l1 >= DISTINCTIVE_L1),
                    "shot_n": float(current_n.sum()),
                    "shot_sse_base": float(
                        np.sum(current_y2 - 2.0 * current_y * base + current_n * base * base)
                    ),
                }

                history_cache: dict[int | None, tuple[np.ndarray, np.ndarray]] = {
                    None: (full_n, full_y)
                }
                for rounds in (40, 80, 160):
                    hn, hy, _, _ = history_arrays(items, rounds)
                    history_cache[rounds] = (hn, hy)

                for name, method, rounds, k in specs:
                    hist_n, hist_y = history_cache[rounds]
                    hist_total = hist_n.sum()
                    hist_overall = (
                        hist_y.sum() / hist_total if hist_total > 0 else global_overall
                    )
                    hist_base = hist_overall + global_cell_deviation
                    if method == "exact8":
                        skill = exact_profile(hist_n, hist_y, hist_base, k)
                    elif method == "exact4":
                        skill = exact_profile(hist_n, hist_y, hist_base, k, MAP_4X2)
                    elif method == "smooth":
                        skill = smooth_profile(hist_n, hist_y, hist_base, k)
                    elif method == "lowrank":
                        skill = lowrank_profile(hist_n, hist_y, hist_base, k)
                    else:
                        raise ValueError(method)

                    out[f"phys__{name}"] = (
                        physical_expected * float(skill @ (physical_mix - global_mix))
                        if matches else 0.0
                    )
                    out[f"event_oracle__{name}"] = current_expected * float(
                        skill @ (current_mix - global_mix)
                    )
                    out[f"player_oracle__{name}"] = player_oracle_expected * float(
                        skill @ (player_oracle_mix - global_mix)
                    )
                    out[f"personal__{name}"] = (
                        physical_expected * float(skill @ (personalized_mix - global_mix))
                        if matches else 0.0
                    )
                    feature_sse = float(
                        np.sum(current_y2 - 2.0 * current_y * skill + current_n * skill * skill)
                    )
                    out[f"shot_gain__{name}"] = out["shot_sse_base"] - feature_sse
                rows.append(out)

        # Only now update player, population, and course histories.
        for event in events_by_date[date]:
            current_counts = event_cells.loc[event].to_numpy(float)
            current_rounds = float(event_rounds[event])
            prior_courses.append((signatures.get(event), current_counts, current_rounds, event))
            global_counts += current_counts
            global_rounds += current_rounds
            target_lookup = {
                str(row.source_player_id): row
                for row in players_by_event[event].itertuples(index=False)
            }
            for player, cell_n, cell_y, _ in by_event_cells[event]:
                global_sums += cell_y
                target_row = target_lookup.get(player)
                if target_row is None:
                    continue
                state[player].append(
                    (
                        float(target_row.nrounds),
                        cell_n.copy(),
                        cell_y.copy(),
                        float(target_row.signal_sum),
                    )
                )

    panel = pd.DataFrame(rows)
    mix_panel = pd.DataFrame(mix_rows)
    diagnostics = {
        "all_app_shots": all_app_shots,
        "valid_complete_app_shots": valid_complete_app_shots,
        "eligible_distance_lie_app_shots": eligible_cell_app_shots,
        "model_panel_app_shots": shot_count,
        "player_events": int(len(panel)),
        "events": int(panel["event_key"].nunique()),
        "hist100_player_events": int(panel["hist_shots"].ge(100).sum()),
        "bellerive_events_in_export": metadata[
            metadata["event_name"].str.contains("bellerive", case=False, na=False)
        ].reset_index()["event_key"].tolist(),
    }
    return panel, mix_panel, diagnostics


def bootstrap_event_delta(values: np.ndarray, seed: int) -> dict:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(4000, len(values)), replace=True).mean(axis=1)
    standard_error = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None
    return {
        "mean_delta_mse": float(values.mean()),
        "ci95_delta_mse": np.quantile(draws, [0.025, 0.975]).tolist(),
        "probability_better": float((draws > 0).mean()),
        "event_cluster_t": float(values.mean() / standard_error) if standard_error else None,
    }


def oos_eval(
    frame: pd.DataFrame,
    target: str,
    base_columns: list[str],
    feature: str,
    seed: int,
) -> dict | None:
    columns = [target, *base_columns, feature]
    data = frame.dropna(subset=columns).copy()
    if len(data) < 200:
        return None
    for column in columns:
        data[column] = data[column] - data.groupby("event_key")[column].transform("mean")

    folds = []
    predictions = []
    for year in TEST_YEARS:
        train = data[data["year"].lt(year)]
        test = data[data["year"].eq(year)].copy()
        if len(train) < 100 or len(test) < 40:
            continue
        train_base = train[base_columns].to_numpy() if base_columns else np.zeros((len(train), 0))
        test_base = test[base_columns].to_numpy() if base_columns else np.zeros((len(test), 0))
        baseline_model = LinearRegression().fit(train_base, train[target]) if base_columns else None
        feature_model = LinearRegression().fit(
            np.column_stack([train_base, train[feature]]), train[target]
        )
        pred0 = (
            baseline_model.predict(test_base)
            if baseline_model is not None
            else np.repeat(float(train[target].mean()), len(test))
        )
        pred1 = feature_model.predict(np.column_stack([test_base, test[feature]]))
        error0 = test[target].to_numpy() - pred0
        error1 = test[target].to_numpy() - pred1
        folds.append(
            {
                "year": year,
                "n": int(len(test)),
                "events": int(test["event_key"].nunique()),
                "rmse0": float(np.sqrt(np.mean(error0 * error0))),
                "rmse1": float(np.sqrt(np.mean(error1 * error1))),
                "delta_rmse": float(
                    np.sqrt(np.mean(error0 * error0)) - np.sqrt(np.mean(error1 * error1))
                ),
                "coefficient": float(feature_model.coef_[-1]),
            }
        )
        predictions.append(
            test[["event_key", "year"]].assign(error0=error0, error1=error1)
        )
    if not predictions:
        return None
    output = pd.concat(predictions, ignore_index=True)
    rmse0 = float(np.sqrt(np.mean(output["error0"] ** 2)))
    rmse1 = float(np.sqrt(np.mean(output["error1"] ** 2)))
    event_delta = (
        output.assign(delta=output["error0"] ** 2 - output["error1"] ** 2)
        .groupby("event_key")["delta"]
        .mean()
        .to_numpy()
    )
    result = {
        "n": int(len(output)),
        "events": int(output["event_key"].nunique()),
        "rmse0": rmse0,
        "rmse1": rmse1,
        "delta_rmse": rmse0 - rmse1,
        "relative_mse_improvement_pct": float(
            100.0 * (rmse0 * rmse0 - rmse1 * rmse1) / (rmse0 * rmse0)
        ),
        "folds": folds,
    }
    result.update(bootstrap_event_delta(event_delta, seed))
    return result


def add_production_residuals(
    panel: pd.DataFrame, export_root: Path, residual_path: Path
) -> pd.DataFrame:
    signal_parts = []
    for year in range(2020, 2027):
        signal_parts.append(
            pd.read_parquet(
                export_root / f"pga_{year}_player_round_signals.parquet",
                columns=["event_key", "source_player_id", "player_name"],
            )
        )
    crosswalk = pd.concat(signal_parts, ignore_index=True).drop_duplicates()
    crosswalk["year"] = crosswalk["event_key"].str[5:9].astype(int)
    crosswalk["event_id"] = crosswalk["event_key"].str[-3:].astype(int)
    crosswalk["name_norm"] = crosswalk["player_name"].map(norm_name)
    crosswalk = crosswalk.drop_duplicates(["year", "event_id", "name_norm"])

    residuals = pd.read_csv(
        residual_path,
        usecols=["player_name", "event_id", "year", "round_num", "actual", "prediction"],
    )
    residuals = residuals[
        residuals["year"].between(2020, 2026)
        & residuals["actual"].notna()
        & residuals["prediction"].notna()
        & residuals["round_num"].eq(1)
    ].copy()
    residuals["event_id"] = residuals["event_id"].astype(int)
    residuals["name_norm"] = residuals["player_name"].map(norm_name)
    residuals = residuals.merge(
        crosswalk,
        on=["year", "event_id", "name_norm"],
        how="inner",
        suffixes=("", "_r2"),
    )
    residuals["residual"] = residuals["actual"] - residuals["prediction"]
    feature_frame = panel.drop(columns=["player_name", "year"])
    return residuals.merge(
        feature_frame,
        on=["event_key", "source_player_id"],
        how="inner",
    )


def run_study(export_root: Path, residual_path: Path) -> dict:
    panel, mix_panel, diagnostics = build_panel(export_root, residual_path)
    production = add_production_residuals(panel, export_root, residual_path)
    diagnostics["production_r1_merged"] = int(len(production))

    mix_test = mix_panel[
        mix_panel["year"].ge(2022) & mix_panel["n_physical_editions"].ge(1)
    ]
    diagnostics["physical_mix"] = {
        "events": int(len(mix_test)),
        "prior_beats_global_fraction": float(
            (mix_test["physical_l1_error"] < mix_test["global_l1_error"]).mean()
        ),
        "median_physical_l1": float(mix_test["physical_l1_error"].median()),
        "median_global_l1": float(mix_test["global_l1_error"].median()),
        "distinctive_events": int(
            mix_test["distinctive_l1"].ge(DISTINCTIVE_L1).sum()
        ),
    }

    specs = [name for name, _, _, _ in profile_specs()]
    physical_features = [f"phys__{name}" for name in specs]
    event_oracles = [f"event_oracle__{name}" for name in specs]
    player_oracles = [f"player_oracle__{name}" for name in specs]
    personal_features = [f"personal__{name}" for name in specs]

    physical_app = panel[
        panel["hist_shots"].ge(100) & panel["n_physical_editions"].ge(1)
    ]
    physical_r1 = production[
        production["hist_shots"].ge(100) & production["n_physical_editions"].ge(1)
    ]
    all_app = panel[panel["hist_shots"].ge(100)]
    all_r1 = production[production["hist_shots"].ge(100)]

    results: dict[str, dict] = {
        "app_physical": {},
        "r1_physical": {},
        "app_event_oracle": {},
        "r1_event_oracle": {},
        "app_player_oracle": {},
        "r1_player_oracle": {},
        "app_personalized": {},
        "r1_personalized": {},
        "app_distinctive": {},
        "r1_distinctive": {},
    }
    seed = 20260819
    for index, feature in enumerate(physical_features):
        results["app_physical"][feature] = oos_eval(
            physical_app, "target", ["app_base"], feature, seed + index
        )
        results["r1_physical"][feature] = oos_eval(
            physical_r1, "residual", [], feature, seed + 100 + index
        )
    for index, feature in enumerate(event_oracles):
        results["app_event_oracle"][feature] = oos_eval(
            all_app, "target", ["app_base"], feature, seed + 200 + index
        )
        results["r1_event_oracle"][feature] = oos_eval(
            all_r1, "residual", [], feature, seed + 300 + index
        )
    for index, feature in enumerate(player_oracles):
        results["app_player_oracle"][feature] = oos_eval(
            all_app, "target", ["app_base"], feature, seed + 400 + index
        )
        results["r1_player_oracle"][feature] = oos_eval(
            all_r1, "residual", [], feature, seed + 500 + index
        )
    for index, feature in enumerate(personal_features):
        results["app_personalized"][feature] = oos_eval(
            physical_app, "target", ["app_base"], feature, seed + 600 + index
        )
        results["r1_personalized"][feature] = oos_eval(
            physical_r1, "residual", [], feature, seed + 700 + index
        )

    distinctive_app = physical_app[physical_app["distinctive"]]
    distinctive_r1 = physical_r1[physical_r1["distinctive"]]
    selected = [
        "phys__exact8_full_k400",
        "phys__exact4_full_k400",
        "phys__smooth8_full_k400",
        "phys__lowrank4_full_k400",
    ]
    for index, feature in enumerate(selected):
        results["app_distinctive"][feature] = oos_eval(
            distinctive_app, "target", ["app_base"], feature, seed + 800 + index
        )
        results["r1_distinctive"][feature] = oos_eval(
            distinctive_r1, "residual", [], feature, seed + 900 + index
        )

    shot_results = {}
    shot_data = panel[panel["hist_shots"].ge(100) & panel["year"].ge(2023)]
    for index, name in enumerate(specs):
        event_shot = (
            shot_data.groupby("event_key")
            .agg(gain=(f"shot_gain__{name}", "sum"), shots=("shot_n", "sum"))
        )
        event_gain_per_shot = (
            event_shot["gain"] / event_shot["shots"].replace(0, np.nan)
        ).dropna()
        total_shots = float(shot_data["shot_n"].sum())
        total_base_sse = float(shot_data["shot_sse_base"].sum())
        total_gain = float(shot_data[f"shot_gain__{name}"].sum())
        entry = {
            "player_events": int(len(shot_data)),
            "events": int(shot_data["event_key"].nunique()),
            "shots": int(total_shots),
            "mse_improvement_per_shot": total_gain / total_shots,
            "relative_mse_improvement_pct": 100.0 * total_gain / total_base_sse,
            "by_year": {},
        }
        # Event-cluster uncertainty uses one per-shot value per tournament.
        # The headline above remains the shot-weighted pooled estimate.
        entry.update(
            bootstrap_event_delta(event_gain_per_shot.to_numpy(), seed + 1000 + index)
        )
        for year, group in shot_data.groupby("year"):
            entry["by_year"][str(int(year))] = {
                "shots": int(group["shot_n"].sum()),
                "mse_improvement_per_shot": float(
                    group[f"shot_gain__{name}"].sum() / group["shot_n"].sum()
                ),
            }
        shot_results[name] = entry

    diagnostics["samples"] = {
        "physical_app_player_events": int(len(physical_app)),
        "physical_app_events": int(physical_app["event_key"].nunique()),
        "physical_r1_player_events": int(len(physical_r1)),
        "physical_r1_events": int(physical_r1["event_key"].nunique()),
        "distinctive_app_player_events": int(len(distinctive_app)),
        "distinctive_app_events": int(distinctive_app["event_key"].nunique()),
        "distinctive_r1_player_events": int(len(distinctive_r1)),
        "distinctive_r1_events": int(distinctive_r1["event_key"].nunique()),
    }
    return {
        "generated_by": "archive/approach_lie_course_fit_extended.py",
        "definitions": {
            "distance_edges_yd": EDGES.tolist(),
            "lie_groups": {
                "clean": ["tee", "fairway"],
                "bad": ["rough", "sand", "recovery"],
            },
            "test_years": list(TEST_YEARS),
            "distinctive_course_l1_threshold": DISTINCTIVE_L1,
            "physical_match_tee_coordinate_rmse_max": 25.0,
        },
        "diagnostics": diagnostics,
        "results": results,
        "shot_level": shot_results,
    }


def compact_print(results: dict) -> None:
    print(json.dumps(results["diagnostics"], indent=2))
    for section in (
        "r1_physical", "r1_event_oracle", "r1_player_oracle",
        "r1_personalized", "r1_distinctive",
    ):
        ranked = [
            (name, value["delta_rmse"], value["probability_better"])
            for name, value in results["results"][section].items()
            if value is not None
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        print(f"\n{section}")
        for row in ranked[:8]:
            print(row)
    print("\nshot_level")
    ranked_shots = sorted(
        (
            (name, value["mse_improvement_per_shot"], value["probability_better"])
            for name, value in results["shot_level"].items()
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    for row in ranked_shots:
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--residuals", type=Path, default=DEFAULT_RESIDS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = run_study(args.export_root, args.residuals)
    compact_print(results)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
