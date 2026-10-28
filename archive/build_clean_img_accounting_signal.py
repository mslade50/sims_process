"""Build chronology-clean IMG player-round SG signals for base-skill research.

The materialized IMG accounting archive is useful, but its current ``v4_lies2026``
surface-code map was selected with evidence from the full 2020-2026 archive and its
putting curve is fitted separately for each event from other rounds.  Both choices
are unsuitable for a sealed 2021+ walk-forward test (and the latter is especially
unsuitable for a live round cascade).

This research-only extractor uses the archived canonical shot *states* but ignores
their stored expected-strokes and SG values.  It freezes every learned measurement
choice using completed 2020 PGA data:

* the putting curve is one monotone 2020-only curve;
* standard surface codes use their literal lie meaning;
* ambiguous/non-standard codes are mapped to the Broadie lie curve with the lowest
  2020 expected-strokes RMSE, subject to a minimum sample size;
* a stricter sensitivity drops every ambiguous-code hole instead of fitting a map.

Non-green expected-strokes curves are the published 2003-2010 PGA TOUR benchmarks
from Broadie (2012), Table B.1.  All 2020-2026 shot SG is recomputed under the frozen
transform.  Player-hole category SG is then centered by physical course, round and
hole before being summed to player-round signals.  The output is isolated research
data; the IMG database, production bridge, and weekly pipeline are never modified.

The 2020 rows are warm-up measurements only.  The clean evaluation period begins
2021-01-01, after every fitted measurement parameter was frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import statistics
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
DEFAULT_IMG_DB = Path(
    r"C:\Users\McKinley Slade\dev\golf_scraping\output\imgarena_shots_lean.sqlite3"
)
DEFAULT_PANEL_DIR = (
    REPO / "permanent_data" / "research" / "base_skill_rebuild_20200101_20260822"
)
DEFAULT_OUTPUT = DEFAULT_PANEL_DIR / "clean_img_accounting"

FIT_START = pd.Timestamp("2020-01-01")
FIT_END_EXCLUSIVE = pd.Timestamp("2021-01-01")
EVALUATION_START = pd.Timestamp("2021-01-01")
MIN_SURFACE_FIT_OBSERVATIONS = 100
MIN_CLEAN_HOLES = 12

CATEGORY_ORDER = ("ott", "app", "arg", "putt")
NON_GREEN_CANDIDATES = ("fairway", "rough", "sand", "recovery")

# These codes have direct, non-empirical meanings.  Codes whose treatment was
# selected in the 2026 full-archive fidelity audit are deliberately absent and are
# either fit on 2020 only (warmup arm) or excluded (unambiguous arm).
UNAMBIGUOUS_SURFACE_MAP: dict[str, str] = {
    "TEE": "tee",
    "OTB": "tee",
    "OGR": "green",
    "ELG": "green",
    "ERG": "green",
    "EGR": "green",
    "OFW": "fairway",
    "ELF": "fairway",
    "ERF": "fairway",
    "ORO": "rough",
    "ELR": "rough",
    "ERR": "rough",
    "ELI": "rough",
    "ERI": "rough",
    "OST": "sand",
    "OBU": "sand",
    "ELB": "sand",
    "ERB": "sand",
    "OWA": "recovery",
    "OTH": "recovery",
    "OUK": "recovery",
    "OTO": "recovery",
    "OCA": "recovery",
    "ORK": "recovery",
    "HOLED": "hole",
}

# Explicitly named because the current v4 map documents outcome-based decisions for
# them.  Additional non-standard codes discovered in the warm-up data are handled by
# the same 2020-only fit and recorded in the audit.
KNOWN_EMPIRICALLY_SELECTED_CODES = frozenset({"OCO", "OIR", "ONA", "OWD", "ODO", "OGS"})

PUTTING_BIN_EDGES_FT = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    8,
    10,
    12,
    15,
    20,
    25,
    30,
    40,
    50,
    60,
    75,
    100,
    125,
    150,
    200,
    300,
)

DISTANCES = (
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
    120,
    140,
    160,
    180,
    200,
    220,
    240,
    260,
    280,
    300,
    320,
    340,
    360,
    380,
    400,
    420,
    440,
    460,
    480,
    500,
    520,
    540,
    560,
    580,
    600,
)
TEE_DISTANCES = DISTANCES[9:]
BROADIE_VALUES: dict[str, tuple[float, ...]] = {
    "tee": (
        2.92, 2.99, 2.97, 2.99, 3.05, 3.12, 3.17, 3.25, 3.45,
        3.65, 3.71, 3.79, 3.86, 3.92, 3.96, 3.99, 4.02, 4.08,
        4.17, 4.28, 4.41, 4.54, 4.65, 4.74, 4.79, 4.82,
    ),
    "fairway": (
        2.18, 2.40, 2.52, 2.60, 2.66, 2.70, 2.72, 2.75, 2.77,
        2.80, 2.85, 2.91, 2.98, 3.08, 3.19, 3.32, 3.45, 3.58,
        3.69, 3.78, 3.84, 3.88, 3.95, 4.03, 4.11, 4.19, 4.27,
        4.34, 4.42, 4.50, 4.58, 4.66, 4.74, 4.82, 4.89,
    ),
    "rough": (
        2.34, 2.59, 2.70, 2.78, 2.87, 2.91, 2.93, 2.96, 2.99,
        3.02, 3.08, 3.15, 3.23, 3.31, 3.42, 3.53, 3.64, 3.74,
        3.83, 3.90, 3.95, 4.02, 4.11, 4.21, 4.30, 4.40, 4.49,
        4.58, 4.68, 4.77, 4.87, 4.96, 5.06, 5.15, 5.25,
    ),
    "sand": (
        2.43, 2.53, 2.66, 2.82, 2.92, 3.15, 3.21, 3.24, 3.24,
        3.23, 3.21, 3.22, 3.28, 3.40, 3.55, 3.70, 3.84, 3.93,
        4.00, 4.04, 4.12, 4.26, 4.41, 4.55, 4.69, 4.83, 4.97,
        5.11, 5.25, 5.40, 5.54, 5.68, 5.82, 5.96, 6.10,
    ),
    "recovery": (
        3.45, 3.51, 3.57, 3.71, 3.79, 3.83, 3.84, 3.84, 3.82,
        3.80, 3.78, 3.80, 3.81, 3.82, 3.87, 3.92, 3.97, 4.03,
        4.10, 4.20, 4.31, 4.44, 4.56, 4.66, 4.75, 4.84, 4.94,
        5.03, 5.13, 5.22, 5.32, 5.41, 5.51, 5.60, 5.70,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--img-db", type=Path, default=DEFAULT_IMG_DB)
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-clean-holes", type=int, default=MIN_CLEAN_HOLES)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp.parquet", dir=path.parent)
    os.close(fd)
    try:
        frame.to_parquet(temp_name, index=False)
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def sqlite_ro(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve().as_posix()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA temp_store=MEMORY")
    return connection


def linear_interpolate(
    values: np.ndarray,
    x_values: Sequence[float],
    y_values: Sequence[float],
    *,
    extrapolate: bool,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    if not extrapolate:
        return np.interp(values, x, y)
    result = np.interp(values, x, y)
    below = values < x[0]
    above = values > x[-1]
    if below.any():
        result[below] = y[0] + (values[below] - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
    if above.any():
        result[above] = y[-2] + (values[above] - x[-2]) * (y[-1] - y[-2]) / (x[-1] - x[-2])
    return result


def expected_non_green(distance_yards: np.ndarray, lie: str) -> np.ndarray:
    grid = TEE_DISTANCES if lie == "tee" else DISTANCES
    return linear_interpolate(
        np.maximum(np.asarray(distance_yards, dtype=float), 0.0),
        grid,
        BROADIE_VALUES[lie],
        extrapolate=True,
    )


def weighted_isotonic(values: Sequence[float], weights: Sequence[int]) -> list[float]:
    blocks: list[list[float]] = []
    for index, (value, weight) in enumerate(zip(values, weights)):
        blocks.append([float(index), float(index), float(value), float(weight)])
        while len(blocks) >= 2 and blocks[-2][2] > blocks[-1][2]:
            right = blocks.pop()
            left = blocks.pop()
            total = left[3] + right[3]
            mean = (left[2] * left[3] + right[2] * right[3]) / total
            blocks.append([left[0], right[1], mean, total])
    result = [0.0] * len(values)
    for start, end, value, _ in blocks:
        for index in range(int(start), int(end) + 1):
            result[index] = value
    return result


def fit_putting_curve(observations: pd.DataFrame) -> dict[str, Any]:
    use = observations.loc[
        observations["distance_ft"].notna() & observations["remaining_strokes"].notna()
    ].copy()
    if len(use) < 100:
        raise ValueError(f"Only {len(use)} valid 2020 putting observations")
    binned: list[tuple[float, float, int]] = [(0.0, 1.0, 50)]
    for lower, upper in zip(PUTTING_BIN_EDGES_FT, PUTTING_BIN_EDGES_FT[1:]):
        values = use.loc[use["distance_ft"].between(lower, upper, inclusive="left")]
        if values.empty:
            continue
        binned.append(
            (
                float(values["distance_ft"].mean()),
                float(values["remaining_strokes"].mean()),
                int(len(values)),
            )
        )
    beyond = use.loc[use["distance_ft"] >= PUTTING_BIN_EDGES_FT[-1]]
    if not beyond.empty:
        binned.append(
            (
                float(beyond["distance_ft"].mean()),
                float(beyond["remaining_strokes"].mean()),
                int(len(beyond)),
            )
        )
    binned.sort()
    fitted = weighted_isotonic(
        [value[1] for value in binned], [value[2] for value in binned]
    )
    return {
        "distance_ft": [value[0] for value in binned],
        "expected_strokes": fitted,
        "bin_samples": [value[2] for value in binned],
        "observations": int(len(use)),
    }


def selected_model_inventory(connection: sqlite3.Connection) -> pd.DataFrame:
    query = """
        WITH distinct_models AS (
            SELECT DISTINCT p.event_key, p.round_no, p.model_id
            FROM archive_player_round_strokes_gained p
        ),
        ranked AS (
            SELECT
                p.event_key, p.round_no, p.model_id, m.model_version,
                m.created_at, e.season, e.start_date, e.end_date,
                ROW_NUMBER() OVER (
                    PARTITION BY p.event_key, p.round_no
                    ORDER BY m.created_at DESC, p.model_id DESC
                ) AS model_rank
            FROM distinct_models p
            JOIN sg_models m ON m.model_id = p.model_id
            JOIN archive_events e ON e.event_key = p.event_key
            WHERE e.tour = 'pga'
              AND m.model_version LIKE '%v4_lies2026%'
        )
        SELECT event_key, round_no, model_id, model_version, created_at,
               season, start_date, end_date
        FROM ranked
        WHERE model_rank = 1
        ORDER BY start_date, event_key, round_no
    """
    frame = pd.read_sql_query(query, connection)
    return frame


def attach_frozen_crosswalk_dates(
    inventory: pd.DataFrame, panel_dir: Path
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Attach chronology from the frozen IMG↔DG crosswalk.

    ``archive_events.start_date/end_date`` are NULL for this archive, so they
    cannot define the 2020 measurement window.  The research crosswalk already
    records one DG round date for every mapped IMG event/round.  Assert that map
    is unique and use the min/max completed round date as event boundaries.
    """
    crosswalk = pd.read_parquet(
        panel_dir / "img_crosswalk_research_extended.parquet",
        columns=["event_key", "round_no", "year", "round_date"],
    )
    crosswalk["round_no"] = pd.to_numeric(
        crosswalk["round_no"], errors="coerce"
    ).astype("Int64")
    crosswalk["year"] = pd.to_numeric(crosswalk["year"], errors="coerce").astype(
        "Int64"
    )
    crosswalk["round_date"] = pd.to_datetime(
        crosswalk["round_date"], errors="coerce"
    ).dt.normalize()
    round_dates = crosswalk.drop_duplicates()
    ambiguity = round_dates.groupby(
        ["event_key", "round_no"], observed=True, dropna=False
    ).agg(years=("year", "nunique"), dates=("round_date", "nunique"))
    if (ambiguity[["years", "dates"]] > 1).any().any():
        raise ValueError("Frozen IMG crosswalk has ambiguous event/round chronology")
    round_dates = round_dates.drop_duplicates(["event_key", "round_no"])
    boundaries = round_dates.groupby("event_key", observed=True).agg(
        crosswalk_year=("year", "first"),
        start_date=("round_date", "min"),
        end_date=("round_date", "max"),
    )
    output = inventory.drop(columns=["start_date", "end_date"], errors="ignore").merge(
        boundaries.reset_index(), on="event_key", how="left", validate="many_to_one"
    )
    season_year_mismatches = int(
        (
            output["crosswalk_year"].notna()
            & pd.to_numeric(output["season"], errors="coerce").ne(
                output["crosswalk_year"]
            )
        ).sum()
    )
    return output, {
        "inventory_rows": int(len(output)),
        "mapped_inventory_rows": int(output["end_date"].notna().sum()),
        "unmapped_inventory_rows": int(output["end_date"].isna().sum()),
        "mapped_events": int(output.loc[output["end_date"].notna(), "event_key"].nunique()),
        "season_crosswalk_year_mismatches": season_year_mismatches,
    }


SHOT_COLUMNS = [
    "event_key",
    "round_no",
    "hole_no",
    "source_player_id",
    "source_team_id",
    "player_name",
    "stroke_no",
    "start_surface_code",
    "start_distance_yd",
    "end_surface_code",
    "end_distance_yd",
    "sg_category",
    "sg_valid",
    "complete_hole",
    "state_inferred",
]


def load_model_shots(
    connection: sqlite3.Connection,
    model_id: str,
    event_key: str,
    round_no: int,
) -> pd.DataFrame:
    query = f"""
        SELECT {', '.join(SHOT_COLUMNS)}
        FROM archive_shot_strokes_gained
        WHERE event_key = ? AND round_no = ? AND model_id = ?
        ORDER BY source_player_id, source_team_id, hole_no, stroke_no
    """
    # The immutable DB has an index beginning (event_key, round_no, ...,
    # model_id), but no model_id-leading index.  Supplying all three keys avoids
    # a full 17 GB table scan for every selected model.
    frame = pd.read_sql_query(
        query, connection, params=(event_key, int(round_no), model_id)
    )
    for column in ("round_no", "hole_no", "stroke_no", "sg_valid", "complete_hole"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("start_distance_yd", "end_distance_yd"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("source_player_id", "source_team_id"):
        frame[column] = frame[column].fillna("").astype(str)
    for column in ("start_surface_code", "end_surface_code"):
        frame[column] = frame[column].fillna("").astype(str).str.upper().str.strip()
    return frame


def complete_hole_rows(shots: pd.DataFrame) -> pd.DataFrame:
    key = ["source_player_id", "source_team_id", "hole_no"]
    grouped = shots.groupby(key, observed=True, dropna=False)
    max_stroke = grouped["stroke_no"].transform("max")
    complete = grouped["complete_hole"].transform("min").eq(1)
    holed = grouped["end_surface_code"].transform(
        lambda values: bool(values.eq("HOLED").any())
    )
    output = shots.copy()
    output["remaining_strokes"] = max_stroke - output["stroke_no"] + 1.0
    output["canonical_hole_complete"] = complete & holed
    return output


def collect_warmup_observations(
    connection: sqlite3.Connection, inventory: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    fit_models = inventory.loc[
        (inventory["end_date"] >= FIT_START) & (inventory["end_date"] < FIT_END_EXCLUSIVE)
    ]
    pieces: list[pd.DataFrame] = []
    code_counts: Counter[str] = Counter()
    for index, model in enumerate(fit_models.itertuples(index=False), start=1):
        shots = complete_hole_rows(
            load_model_shots(
                connection, model.model_id, model.event_key, model.round_no
            )
        )
        shots = shots.loc[shots["canonical_hole_complete"]].copy()
        code_counts.update(shots["start_surface_code"].tolist())
        pieces.append(
            shots[
                [
                    "event_key",
                    "round_no",
                    "source_player_id",
                    "source_team_id",
                    "hole_no",
                    "stroke_no",
                    "start_surface_code",
                    "start_distance_yd",
                    "remaining_strokes",
                ]
            ]
        )
        if index % 50 == 0 or index == len(fit_models):
            print(f"Warm-up states: {index}/{len(fit_models)} models", flush=True)
    if not pieces:
        raise ValueError("No completed pre-2021 IMG models are available")
    return pd.concat(pieces, ignore_index=True), dict(code_counts)


def fit_surface_map(
    warmup: pd.DataFrame, putting_curve: Mapping[str, Any]
) -> tuple[dict[str, str], pd.DataFrame]:
    surface_map = dict(UNAMBIGUOUS_SURFACE_MAP)
    all_codes = sorted(code for code in warmup["start_surface_code"].unique() if code)
    fitted_rows: list[dict[str, Any]] = []
    for code in all_codes:
        if code in surface_map:
            continue
        sample = warmup.loc[
            warmup["start_surface_code"].eq(code)
            & warmup["start_distance_yd"].notna()
            & warmup["remaining_strokes"].notna()
        ].copy()
        candidate_metrics: dict[str, dict[str, float]] = {}
        for lie in NON_GREEN_CANDIDATES:
            expected = expected_non_green(sample["start_distance_yd"].to_numpy(float), lie)
            error = expected - sample["remaining_strokes"].to_numpy(float)
            candidate_metrics[lie] = {
                "rmse": float(np.sqrt(np.mean(np.square(error)))) if len(error) else np.nan,
                "mae": float(np.mean(np.abs(error))) if len(error) else np.nan,
                "bias": float(np.mean(error)) if len(error) else np.nan,
            }
        eligible = len(sample) >= MIN_SURFACE_FIT_OBSERVATIONS
        selected = (
            min(NON_GREEN_CANDIDATES, key=lambda lie: candidate_metrics[lie]["rmse"])
            if eligible
            else None
        )
        ordered = sorted(
            (values["rmse"], lie) for lie, values in candidate_metrics.items()
        )
        margin = ordered[1][0] - ordered[0][0] if len(ordered) > 1 else np.nan
        if selected is not None:
            surface_map[code] = selected
        fitted_rows.append(
            {
                "surface_code": code,
                "warmup_observations": int(len(sample)),
                "selected_lie": selected,
                "selection_eligible": bool(eligible),
                "best_vs_second_rmse_margin": float(margin),
                "known_2026_empirically_selected_code": code in KNOWN_EMPIRICALLY_SELECTED_CODES,
                "candidate_metrics_json": json.dumps(candidate_metrics, sort_keys=True),
            }
        )
    return surface_map, pd.DataFrame(fitted_rows)


def expected_values(
    distance_yards: pd.Series,
    surface_codes: pd.Series,
    surface_map: Mapping[str, str],
    putting_curve: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    distance = pd.to_numeric(distance_yards, errors="coerce").to_numpy(float)
    codes = surface_codes.fillna("").astype(str).str.upper().to_numpy()
    lie = np.asarray([surface_map.get(code) for code in codes], dtype=object)
    expected = np.full(len(distance), np.nan, dtype=float)
    for category in ("tee", *NON_GREEN_CANDIDATES):
        mask = (lie == category) & np.isfinite(distance)
        if mask.any():
            expected[mask] = expected_non_green(distance[mask], category)
    green = (lie == "green") & np.isfinite(distance)
    if green.any():
        expected[green] = linear_interpolate(
            np.maximum(distance[green], 0.0) * 3.0,
            putting_curve["distance_ft"],
            putting_curve["expected_strokes"],
            extrapolate=False,
        )
    expected[lie == "hole"] = 0.0
    return expected, lie


def attach_recomputed_sg(
    shots: pd.DataFrame,
    surface_map: Mapping[str, str],
    putting_curve: Mapping[str, Any],
    prefix: str,
) -> pd.DataFrame:
    output = shots.copy()
    start, start_lie = expected_values(
        output["start_distance_yd"], output["start_surface_code"], surface_map, putting_curve
    )
    end, end_lie = expected_values(
        output["end_distance_yd"], output["end_surface_code"], surface_map, putting_curve
    )
    categories = np.full(len(output), None, dtype=object)
    distances = pd.to_numeric(output["start_distance_yd"], errors="coerce").to_numpy(float)
    tee = start_lie == "tee"
    stored_categories = output["sg_category"].fillna("").astype(str).str.lower().to_numpy()
    # Hole par is not materialized in archive_shot_strokes_gained.  The stored
    # category of a TEE/OTB start is nevertheless chronology-safe: the v4 lie map
    # always treats those literal codes as tee, so it only encodes the source hole
    # par (par 3 -> APP, par 4/5 -> OTT).  No other stored category is reused.
    categories[tee & (stored_categories == "ott")] = "ott"
    categories[tee & (stored_categories == "app")] = "app"
    categories[start_lie == "green"] = "putt"
    non_green = np.isin(start_lie, NON_GREEN_CANDIDATES) & np.isfinite(distances)
    categories[non_green & (distances <= 50.0)] = "arg"
    categories[non_green & (distances > 50.0)] = "app"

    # The canonical state builder assigns a penalty to the preceding shot's
    # category.  Reapply that rule after the frozen lie map rather than inheriting
    # v4's potentially outcome-selected category.
    penalty = output["state_inferred"].fillna("").astype(str).str.contains(
        "penalty_category_from_previous_stroke", regex=False
    ).to_numpy()
    group_key = ["source_player_id", "source_team_id", "hole_no"]
    for indices in output.groupby(group_key, observed=True, dropna=False, sort=False).indices.values():
        previous: str | None = None
        for row_index in indices:
            if penalty[row_index] and previous in CATEGORY_ORDER:
                categories[row_index] = previous
            if categories[row_index] in CATEGORY_ORDER:
                previous = str(categories[row_index])

    valid_category = np.isin(categories, CATEGORY_ORDER)
    valid = (
        output["canonical_hole_complete"].to_numpy()
        & np.isfinite(start)
        & np.isfinite(end)
        & valid_category
    )
    output[f"{prefix}_start_lie"] = start_lie
    output[f"{prefix}_end_lie"] = end_lie
    output[f"{prefix}_category"] = categories
    output[f"{prefix}_category_changed_from_v4"] = (
        valid_category
        & np.isin(stored_categories, CATEGORY_ORDER)
        & (categories != stored_categories)
    )
    output[f"{prefix}_valid"] = valid
    output[f"{prefix}_sg_raw"] = np.where(valid, start - end - 1.0, np.nan)
    output[f"{prefix}_ambiguous_state"] = (
        pd.isna(start_lie) | pd.isna(end_lie)
    )
    return output


def build_round_lookup(panel_dir: Path) -> pd.DataFrame:
    crosswalk = pd.read_parquet(panel_dir / "img_crosswalk_research_extended.parquet")
    rounds = pd.read_parquet(
        panel_dir / "player_round_panel.parquet",
        columns=[
            "year",
            "event_id",
            "round_num",
            "round_date",
            "dg_id",
            "course_num",
            "event_start_date",
            "player_round_key",
        ],
    )
    for frame in (crosswalk, rounds):
        for column in ("year", "event_id", "round_num", "dg_id"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    crosswalk["round_date"] = pd.to_datetime(crosswalk["round_date"], errors="coerce").dt.normalize()
    rounds["round_date"] = pd.to_datetime(rounds["round_date"], errors="coerce").dt.normalize()
    join = ["year", "event_id", "round_num", "round_date", "dg_id"]
    if rounds.duplicated(join).any():
        raise ValueError("DG round panel is ambiguous for IMG course attachment")
    output = crosswalk.merge(
        rounds[join + ["course_num", "event_start_date", "player_round_key"]],
        on=join,
        how="left",
        validate="one_to_one",
    )
    for column in ("source_player_id", "source_team_id"):
        output[column] = output[column].fillna("").astype(str)
    if output.duplicated(["event_key", "round_no", "source_player_id", "source_team_id"]).any():
        raise ValueError("Research IMG crosswalk is ambiguous for source round/player")
    return output


def aggregate_one_arm(
    shots: pd.DataFrame,
    lookup: pd.DataFrame,
    prefix: str,
    min_clean_holes: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    player_key = ["source_player_id", "source_team_id"]
    hole_key = player_key + ["hole_no"]
    valid_column = f"{prefix}_valid"
    raw_column = f"{prefix}_sg_raw"
    group = shots.groupby(hole_key, observed=True, dropna=False)
    strict_hole = group[valid_column].transform("all") & group[valid_column].transform("size").gt(0)
    use = shots.loc[strict_hole & shots[valid_column]].copy()
    if use.empty:
        return pd.DataFrame(), {"strict_holes": 0, "rounds": 0}
    values = (
        use.pivot_table(
            index=hole_key,
            columns=f"{prefix}_category",
            values=raw_column,
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(columns=CATEGORY_ORDER, fill_value=0.0)
        .reset_index()
    )
    values = values.merge(
        lookup,
        on=player_key,
        how="inner",
        validate="many_to_one",
    )
    if values.empty:
        return pd.DataFrame(), {"strict_holes": 0, "rounds": 0}
    field_key = ["event_key", "round_no", "course_num", "hole_no"]
    for category in CATEGORY_ORDER:
        mean = values.groupby(field_key, dropna=False, observed=True)[category].transform("mean")
        values[f"{category}_adj"] = values[category] - mean
    values["field_n"] = values.groupby(field_key, dropna=False, observed=True)["dg_id"].transform("count")
    centering = values.groupby(field_key, dropna=False, observed=True)[
        [f"{category}_adj" for category in CATEGORY_ORDER]
    ].mean()
    max_centering_error = float(centering.abs().max().max()) if len(centering) else np.nan

    round_group = [
        "event_key",
        "round_no",
        "year",
        "event_id",
        "round_num",
        "round_date",
        "event_start_date",
        "dg_id",
        "dg_player_name",
        "course_num",
        "player_round_key",
    ]
    aggregation: dict[str, tuple[str, str]] = {
        f"{prefix}_clean_holes": ("hole_no", "nunique"),
        f"{prefix}_min_hole_field_n": ("field_n", "min"),
    }
    for category in CATEGORY_ORDER:
        aggregation[f"{prefix}_sg_{category}_raw_observed"] = (category, "sum")
        aggregation[f"{prefix}_sg_{category}_field_adjusted_observed"] = (
            f"{category}_adj",
            "sum",
        )
    result = values.groupby(round_group, dropna=False, observed=True).agg(**aggregation).reset_index()
    hole_count = result[f"{prefix}_clean_holes"].astype(float)
    scale = np.divide(18.0, hole_count, out=np.full(len(result), np.nan), where=hole_count > 0)
    for category in CATEGORY_ORDER:
        result[f"{prefix}_sg_{category}_raw_per18"] = (
            result[f"{prefix}_sg_{category}_raw_observed"] * scale
        )
        result[f"{prefix}_sg_{category}_field_adjusted_per18"] = (
            result[f"{prefix}_sg_{category}_field_adjusted_observed"] * scale
        )
    result[f"{prefix}_sg_total_raw_per18"] = result[
        [f"{prefix}_sg_{category}_raw_per18" for category in CATEGORY_ORDER]
    ].sum(axis=1)
    result[f"{prefix}_sg_total_field_adjusted_per18"] = result[
        [f"{prefix}_sg_{category}_field_adjusted_per18" for category in CATEGORY_ORDER]
    ].sum(axis=1)
    result[f"{prefix}_signal_available"] = hole_count >= int(min_clean_holes)
    result[f"{prefix}_strict_complete_18"] = hole_count.eq(18)
    return result, {
        "strict_holes": int(len(values)),
        "rounds": int(len(result)),
        "available_rounds": int(result[f"{prefix}_signal_available"].sum()),
        "complete_18_rounds": int(result[f"{prefix}_strict_complete_18"].sum()),
        "max_abs_hole_field_centering_error": max_centering_error,
    }


def process_models(
    connection: sqlite3.Connection,
    inventory: pd.DataFrame,
    lookup: pd.DataFrame,
    warmup_map: Mapping[str, str],
    putting_curve: Mapping[str, Any],
    min_clean_holes: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    output: list[pd.DataFrame] = []
    audit = {
        "shot_rows": 0,
        "warmup_ambiguous_state_rows": 0,
        "unambiguous_ambiguous_state_rows": 0,
        "warmup_categories_changed_from_v4": 0,
        "unambiguous_categories_changed_from_v4": 0,
        "unmapped_source_player_rows": 0,
        "arm_totals": {
            "img_clean_warmup": Counter(),
            "img_clean_unambiguous": Counter(),
        },
    }
    for index, model in enumerate(inventory.itertuples(index=False), start=1):
        shots = complete_hole_rows(
            load_model_shots(
                connection, model.model_id, model.event_key, model.round_no
            )
        )
        audit["shot_rows"] += int(len(shots))
        shots = attach_recomputed_sg(
            shots, warmup_map, putting_curve, "img_clean_warmup"
        )
        shots = attach_recomputed_sg(
            shots,
            UNAMBIGUOUS_SURFACE_MAP,
            putting_curve,
            "img_clean_unambiguous",
        )
        audit["warmup_ambiguous_state_rows"] += int(
            shots["img_clean_warmup_ambiguous_state"].sum()
        )
        audit["unambiguous_ambiguous_state_rows"] += int(
            shots["img_clean_unambiguous_ambiguous_state"].sum()
        )
        audit["warmup_categories_changed_from_v4"] += int(
            shots["img_clean_warmup_category_changed_from_v4"].sum()
        )
        audit["unambiguous_categories_changed_from_v4"] += int(
            shots["img_clean_unambiguous_category_changed_from_v4"].sum()
        )
        event_lookup = lookup.loc[
            lookup["event_key"].eq(model.event_key)
            & pd.to_numeric(lookup["round_no"], errors="coerce").eq(model.round_no)
        ].copy()
        source_players = shots[["source_player_id", "source_team_id"]].drop_duplicates()
        matched_players = event_lookup[["source_player_id", "source_team_id"]].drop_duplicates()
        audit["unmapped_source_player_rows"] += int(
            len(source_players.merge(matched_players, on=["source_player_id", "source_team_id"], how="left", indicator=True).query("_merge == 'left_only'"))
        )
        pieces: list[pd.DataFrame] = []
        for prefix in ("img_clean_warmup", "img_clean_unambiguous"):
            piece, arm_audit = aggregate_one_arm(
                shots, event_lookup, prefix, min_clean_holes
            )
            max_error = arm_audit.pop("max_abs_hole_field_centering_error")
            audit["arm_totals"][prefix].update(arm_audit)
            previous_max = audit["arm_totals"][prefix].get(
                "max_abs_hole_field_centering_error", float("nan")
            )
            finite_errors = [
                value for value in (previous_max, max_error) if np.isfinite(value)
            ]
            audit["arm_totals"][prefix][
                "max_abs_hole_field_centering_error"
            ] = max(finite_errors) if finite_errors else float("nan")
            if not piece.empty:
                pieces.append(piece)
        if pieces:
            merged = pieces[0]
            for piece in pieces[1:]:
                join = [
                    "event_key",
                    "round_no",
                    "year",
                    "event_id",
                    "round_num",
                    "round_date",
                    "event_start_date",
                    "dg_id",
                    "dg_player_name",
                    "course_num",
                    "player_round_key",
                ]
                merged = merged.merge(piece, on=join, how="outer", validate="one_to_one")
            merged["img_clean_source_model_id"] = model.model_id
            merged["img_clean_source_model_version"] = model.model_version
            merged["img_clean_measurement_frozen_before"] = FIT_END_EXCLUSIVE
            merged["img_clean_evaluation_eligible"] = pd.to_datetime(
                merged["round_date"]
            ).ge(EVALUATION_START)
            output.append(merged)
        if index % 50 == 0 or index == len(inventory):
            print(f"Clean IMG signals: {index}/{len(inventory)} models", flush=True)
    result = pd.concat(output, ignore_index=True) if output else pd.DataFrame()
    audit["arm_totals"] = {
        key: dict(value) for key, value in audit["arm_totals"].items()
    }
    return result, audit


def main() -> None:
    args = parse_args()
    img_db = args.img_db.resolve()
    panel_dir = args.panel_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not img_db.exists():
        raise FileNotFoundError(img_db)
    if args.min_clean_holes < 1 or args.min_clean_holes > 18:
        raise ValueError("--min-clean-holes must be between 1 and 18")
    output_dir.mkdir(parents=True, exist_ok=True)

    with sqlite_ro(img_db) as connection:
        raw_inventory = selected_model_inventory(connection)
        if raw_inventory.empty:
            raise ValueError("No canonical v4 PGA archive models found")
        inventory, inventory_date_audit = attach_frozen_crosswalk_dates(
            raw_inventory, panel_dir
        )
        inventory = inventory.loc[inventory["end_date"].notna()].copy()
        if inventory.empty:
            raise ValueError("No IMG models map to frozen crosswalk chronology")
        warmup, warmup_code_counts = collect_warmup_observations(connection, inventory)
        putt_observations = warmup.loc[
            warmup["start_surface_code"].isin(
                [code for code, lie in UNAMBIGUOUS_SURFACE_MAP.items() if lie == "green"]
            )
        ].copy()
        putt_observations["distance_ft"] = putt_observations["start_distance_yd"] * 3.0
        putting_curve = fit_putting_curve(putt_observations)
        warmup_map, surface_fit = fit_surface_map(warmup, putting_curve)
        lookup = build_round_lookup(panel_dir)
        signals, signal_audit = process_models(
            connection,
            inventory,
            lookup,
            warmup_map,
            putting_curve,
            args.min_clean_holes,
        )

    if signals.empty:
        raise ValueError("Clean IMG extractor produced no player-round signals")
    if signals.duplicated("player_round_key").any():
        raise AssertionError("Clean IMG signal is not unique by player_round_key")
    fit_models = inventory.loc[
        (inventory["end_date"] >= FIT_START) & (inventory["end_date"] < FIT_END_EXCLUSIVE)
    ]
    fit_max_end = fit_models["end_date"].max()
    if pd.isna(fit_max_end) or fit_max_end >= EVALUATION_START:
        raise AssertionError("Measurement fit crossed the 2021 evaluation boundary")

    signal_path = output_dir / "clean_img_player_round.parquet"
    inventory_path = output_dir / "selected_model_inventory.parquet"
    curve_path = output_dir / "clean_img_fixed_measurement.json"
    surface_fit_path = output_dir / "clean_img_surface_fit.csv"
    audit_path = output_dir / "audit.json"
    manifest_path = output_dir / "manifest.json"
    atomic_parquet(signals.sort_values(["round_date", "event_key", "round_no", "dg_id"]), signal_path)
    atomic_parquet(
        inventory.sort_values(["start_date", "event_key", "round_no", "model_id"]),
        inventory_path,
    )
    surface_fit.to_csv(surface_fit_path, index=False)
    measurement = {
        "measurement_version": "broadie_fixed_non_green_2020_putting_surface_fit_v1",
        "research_only": True,
        "fit_start": str(FIT_START.date()),
        "fit_end_exclusive": str(FIT_END_EXCLUSIVE.date()),
        "fit_max_event_end": str(pd.Timestamp(fit_max_end).date()),
        "evaluation_start": str(EVALUATION_START.date()),
        "putting_curve": putting_curve,
        "unambiguous_surface_map": UNAMBIGUOUS_SURFACE_MAP,
        "warmup_fitted_surface_map": warmup_map,
        "warmup_surface_code_counts": warmup_code_counts,
        "known_full_archive_empirically_selected_codes": sorted(
            KNOWN_EMPIRICALLY_SELECTED_CODES
        ),
        "surface_fit_rule": (
            "For non-standard codes with >=100 completed 2020 states, choose the "
            "Broadie fairway/rough/sand/recovery curve minimizing state-level RMSE "
            "against actual remaining strokes. Freeze before 2021."
        ),
        "strict_sensitivity_rule": (
            "Drop an entire player-hole if any shot start/end code is outside the "
            "literal unambiguous map."
        ),
        "non_green_source": (
            "Mark Broadie (2012), Assessing Golfer Performance on the PGA TOUR, "
            "Table B.1 (2003-2010 PGA TOUR benchmarks)"
        ),
    }
    atomic_json(measurement, curve_path)

    coverage_rows: list[dict[str, Any]] = []
    for year, group in signals.groupby("year", observed=True):
        row: dict[str, Any] = {
            "year": int(year),
            "rows": int(len(group)),
            "players": int(group["dg_id"].nunique()),
            "events": int(group["event_key"].nunique()),
        }
        for prefix in ("img_clean_warmup", "img_clean_unambiguous"):
            row[f"{prefix}_available_rate"] = float(
                group[f"{prefix}_signal_available"].fillna(False).mean()
            )
            row[f"{prefix}_median_clean_holes"] = float(
                group[f"{prefix}_clean_holes"].median()
            )
        coverage_rows.append(row)
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(output_dir / "coverage_by_year.csv", index=False)

    audit = {
        "created_at_utc": utc_now(),
        "research_only": True,
        "production_changes": False,
        "inventory_models": int(len(inventory)),
        "raw_inventory_models": int(len(raw_inventory)),
        "inventory_events": int(inventory["event_key"].nunique()),
        "inventory_date_range": [
            str(inventory["start_date"].min().date()),
            str(inventory["end_date"].max().date()),
        ],
        "fit_models": int(len(fit_models)),
        "fit_events": int(fit_models["event_key"].nunique()),
        "fit_max_event_end": fit_max_end,
        "fit_boundary_violations": int((fit_models["end_date"] >= EVALUATION_START).sum()),
        "putting_observations": int(putting_curve["observations"]),
        "surface_codes_fit": int(surface_fit["selection_eligible"].sum()) if len(surface_fit) else 0,
        "surface_codes_excluded_for_low_warmup_n": int((~surface_fit["selection_eligible"]).sum()) if len(surface_fit) else 0,
        "signal_rows": int(len(signals)),
        "signal_events": int(signals["event_key"].nunique()),
        "signal_players": int(signals["dg_id"].nunique()),
        "duplicate_player_round_keys": int(signals.duplicated("player_round_key").sum()),
        "source_current_v4_expected_sg_reused": False,
        "source_current_v4_surface_map_reused": False,
        "source_current_v4_per_event_putting_curve_reused": False,
        "inventory_chronology": inventory_date_audit,
        "signal_processing": signal_audit,
    }
    atomic_json(audit, audit_path)

    artifacts = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            artifacts.append(
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    manifest = {
        "complete": True,
        "research_only": True,
        "created_at_utc": utc_now(),
        "script": str(Path(__file__).resolve()),
        "inputs": {
            "img_db": {
                "path": str(img_db),
                "bytes": img_db.stat().st_size,
                "sha256": None,
                "sha256_note": "Large immutable database; selected model IDs and output hashes are recorded.",
            },
            "research_crosswalk": {
                "path": str(panel_dir / "img_crosswalk_research_extended.parquet"),
                "sha256": sha256_file(panel_dir / "img_crosswalk_research_extended.parquet"),
            },
            "round_panel": {
                "path": str(panel_dir / "player_round_panel.parquet"),
                "sha256": sha256_file(panel_dir / "player_round_panel.parquet"),
            },
            "selected_model_inventory": {
                "path": str(inventory_path),
                "sha256": sha256_file(inventory_path),
                "note": "Exact latest v4 model IDs selected from the live read-only IMG DB.",
            },
        },
        "parameters": {
            "fit_start": str(FIT_START.date()),
            "fit_end_exclusive": str(FIT_END_EXCLUSIVE.date()),
            "evaluation_start": str(EVALUATION_START.date()),
            "min_clean_holes": int(args.min_clean_holes),
            "min_surface_fit_observations": MIN_SURFACE_FIT_OBSERVATIONS,
        },
        "artifacts": artifacts,
    }
    atomic_json(manifest, manifest_path)
    print(f"Completed clean IMG accounting signal: {signal_path}", flush=True)


if __name__ == "__main__":
    main()
