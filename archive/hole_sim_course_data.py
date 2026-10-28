"""Read-only course data helpers for a hole-level shot simulator prototype.

The R2 export assigns a different ``course_id`` layout hash to most rounds, so
those hashes cannot identify a physical course across editions.  This module
matches recurring editions by event metadata and verifies the match from hole
geometry.  It also converts raw ShotLink coordinates into a setup-local frame:

* longitudinal yards from the tee toward the pin;
* lateral yards, positive to the player's left of the tee-to-pin axis.

Nothing in the weekly production pipeline imports this archived research file.
The CLI only reads the R2 export and prints an audit as JSON.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_EXPORT = Path(
    r"C:\Users\McKinley Slade\dev\golf_scraping\output\export"
)
EVENT_KEY_RE = re.compile(
    r"^(?P<tour>[a-z0-9_]+):(?P<prefix>R)?(?P<season>\d{4})(?P<event_id>\d{3})$",
    re.IGNORECASE,
)
FEET_PER_YARD = 3.0


@dataclass(frozen=True)
class EventIdentity:
    tour: str
    season: int
    event_id: int


def parse_event_key(value: str) -> EventIdentity:
    """Parse an R2 key such as ``pga:R2026034`` without string slicing."""
    match = EVENT_KEY_RE.fullmatch(str(value).strip())
    if match is None:
        raise ValueError(f"Unrecognized R2 event key: {value!r}")
    return EventIdentity(
        tour=match.group("tour").lower(),
        season=int(match.group("season")),
        event_id=int(match.group("event_id")),
    )


def load_event_catalog(export_root: Path = DEFAULT_EXPORT, tour: str = "pga") -> pd.DataFrame:
    catalog = pd.read_parquet(export_root / "events.parquet").copy()
    catalog = catalog[catalog["tour"].str.lower().eq(tour.lower())].copy()
    parsed = catalog["event_key"].map(parse_event_key)
    catalog["parsed_season"] = [item.season for item in parsed]
    catalog["event_id"] = [item.event_id for item in parsed]
    mismatch = catalog["season"].astype(int).ne(catalog["parsed_season"])
    if mismatch.any():
        raise ValueError(f"Season mismatch in {int(mismatch.sum())} catalog rows")
    return catalog.sort_values(["season", "event_id"]).reset_index(drop=True)


def _read_year_parts(
    export_root: Path,
    stem: str,
    event_keys: list[str],
    years: list[int],
    columns: list[str] | None = None,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for year in years:
        path = export_root / f"pga_{year}_{stem}.parquet"
        if not path.exists():
            continue
        part = pd.read_parquet(
            path,
            columns=columns,
            filters=[("event_key", "in", event_keys)],
        )
        if len(part):
            parts.append(part)
    if not parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(parts, ignore_index=True)


def load_recurring_event(
    event_name: str,
    export_root: Path = DEFAULT_EXPORT,
    tour: str = "pga",
) -> pd.DataFrame:
    """Return exact-name recurring editions; callers must verify geometry."""
    catalog = load_event_catalog(export_root, tour=tour)
    editions = catalog[catalog["event_name"].eq(event_name)].copy()
    if editions.empty:
        choices = catalog.loc[
            catalog["event_name"].str.contains(event_name, case=False, regex=False),
            "event_name",
        ].drop_duplicates()
        hint = f" Similar names: {choices.tolist()}" if len(choices) else ""
        raise ValueError(f"No exact recurring event named {event_name!r}.{hint}")
    return editions.sort_values("season").reset_index(drop=True)


def load_geometry(editions: pd.DataFrame, export_root: Path = DEFAULT_EXPORT) -> pd.DataFrame:
    columns = [
        "event_key", "season", "round_no", "hole_no", "course_id", "par",
        "yardage", "tee_x", "tee_y", "tee_z", "pin_x", "pin_y", "pin_z",
    ]
    geometry = _read_year_parts(
        export_root,
        "hole_geometry",
        editions["event_key"].tolist(),
        sorted(editions["season"].astype(int).unique().tolist()),
        columns,
    )
    return geometry[geometry["hole_no"].between(1, 18)].copy()


def load_shots(editions: pd.DataFrame, export_root: Path = DEFAULT_EXPORT) -> pd.DataFrame:
    columns = [
        "event_key", "season", "source_player_id", "player_name", "round_no",
        "hole_no", "stroke_no", "event_type", "start_lie", "end_lie",
        "shot_distance_yd", "distance_remaining_yd", "start_x", "start_y",
        "start_z", "end_x", "end_y", "end_z", "ball_holed", "course_id",
        "shot_key",
    ]
    return _read_year_parts(
        export_root,
        "shots",
        editions["event_key"].tolist(),
        sorted(editions["season"].astype(int).unique().tolist()),
        columns,
    )


def _geometry_points(frame: pd.DataFrame, round_specific: bool) -> pd.DataFrame:
    keys = ["hole_no"] + (["round_no"] if round_specific else [])
    columns = ["tee_x", "tee_y", "pin_x", "pin_y", "yardage"]
    return frame.groupby(keys, as_index=False)[columns].mean()


def _rigid_aligned_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Point distances after optimal translation and proper 2-D rotation."""
    left_centered = left - left.mean(axis=0)
    right_centered = right - right.mean(axis=0)
    u, _, vt = np.linalg.svd(left_centered.T @ right_centered)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    fitted = left_centered @ rotation
    return np.linalg.norm(fitted - right_centered, axis=1) / FEET_PER_YARD


def compare_geometry(
    left: pd.DataFrame,
    right: pd.DataFrame,
    round_specific: bool = True,
) -> dict[str, float | int]:
    """Compare editions while tolerating a global coordinate-origin change."""
    a = _geometry_points(left, round_specific)
    b = _geometry_points(right, round_specific)
    keys = ["hole_no"] + (["round_no"] if round_specific else [])
    merged = a.merge(b, on=keys, suffixes=("_a", "_b"))
    if len(merged) < (60 if round_specific else 15):
        raise ValueError(f"Too few common geometry rows: {len(merged)}")

    left_points = np.vstack(
        [merged[["tee_x_a", "tee_y_a"]].to_numpy(), merged[["pin_x_a", "pin_y_a"]].to_numpy()]
    )
    right_points = np.vstack(
        [merged[["tee_x_b", "tee_y_b"]].to_numpy(), merged[["pin_x_b", "pin_y_b"]].to_numpy()]
    )
    finite = np.isfinite(left_points).all(axis=1) & np.isfinite(right_points).all(axis=1)
    left_points = left_points[finite]
    right_points = right_points[finite]
    raw = np.linalg.norm(left_points - right_points, axis=1) / FEET_PER_YARD
    aligned = _rigid_aligned_distances(left_points, right_points)
    yardage_delta = np.abs(merged["yardage_a"] - merged["yardage_b"])
    return {
        "common_setups": int(len(merged)),
        "raw_rmse_yd": float(np.sqrt(np.mean(raw**2))),
        "aligned_rmse_yd": float(np.sqrt(np.mean(aligned**2))),
        "aligned_median_yd": float(np.median(aligned)),
        "aligned_p90_yd": float(np.quantile(aligned, 0.90)),
        "mean_abs_yardage_delta": float(yardage_delta.mean()),
    }


def add_local_coordinates(shots: pd.DataFrame, geometry: pd.DataFrame) -> pd.DataFrame:
    """Attach setup-local start/end coordinates in yards.

    Keeping continuous local coordinates avoids throwing away dogleg angle and
    lateral hazard information when yardage buckets are added later.
    """
    keys = ["event_key", "round_no", "hole_no", "course_id"]
    setup = geometry.groupby(keys, as_index=False).agg(
        tee_x=("tee_x", "mean"), tee_y=("tee_y", "mean"),
        pin_x=("pin_x", "mean"), pin_y=("pin_y", "mean"),
    )
    out = shots.merge(setup, on=keys, how="left", validate="many_to_one")
    dx = out["pin_x"].to_numpy(float) - out["tee_x"].to_numpy(float)
    dy = out["pin_y"].to_numpy(float) - out["tee_y"].to_numpy(float)
    length = np.hypot(dx, dy)
    valid_axis = np.isfinite(length) & (length > 30.0)
    ux = np.divide(dx, length, out=np.full(len(out), np.nan), where=valid_axis)
    uy = np.divide(dy, length, out=np.full(len(out), np.nan), where=valid_axis)

    for endpoint in ("start", "end"):
        px = out[f"{endpoint}_x"].to_numpy(float) - out["tee_x"].to_numpy(float)
        py = out[f"{endpoint}_y"].to_numpy(float) - out["tee_y"].to_numpy(float)
        out[f"{endpoint}_long_yd"] = (px * ux + py * uy) / FEET_PER_YARD
        out[f"{endpoint}_lateral_yd"] = (-px * uy + py * ux) / FEET_PER_YARD
    out["setup_length_yd"] = length / FEET_PER_YARD
    return out


def sequence_continuity(shots: pd.DataFrame, tolerance_yd: float = 0.01) -> dict[str, float | int]:
    ordered = shots.sort_values(
        ["event_key", "source_player_id", "round_no", "hole_no", "stroke_no"]
    ).copy()
    keys = ["event_key", "source_player_id", "round_no", "hole_no"]
    ordered["next_start_x"] = ordered.groupby(keys)["start_x"].shift(-1)
    ordered["next_start_y"] = ordered.groupby(keys)["start_y"].shift(-1)
    columns = ["end_x", "end_y", "next_start_x", "next_start_y"]
    valid = ordered[columns].notna().all(axis=1)
    distance = np.hypot(
        ordered.loc[valid, "end_x"] - ordered.loc[valid, "next_start_x"],
        ordered.loc[valid, "end_y"] - ordered.loc[valid, "next_start_y"],
    ) / FEET_PER_YARD
    return {
        "comparable_transitions": int(len(distance)),
        "within_tolerance_rate": float(distance.le(tolerance_yd).mean()),
        "median_gap_yd": float(distance.median()),
        "p99_gap_yd": float(distance.quantile(0.99)),
    }


def audit_event(
    event_name: str,
    holdout_year: int,
    export_root: Path = DEFAULT_EXPORT,
) -> dict:
    editions = load_recurring_event(event_name, export_root)
    if holdout_year not in editions["season"].astype(int).tolist():
        raise ValueError(f"{event_name!r} has no {holdout_year} edition")
    geometry = load_geometry(editions, export_root)
    shots = load_shots(editions, export_root)
    holdout_keys = editions.loc[editions["season"].eq(holdout_year), "event_key"]
    if len(holdout_keys) != 1:
        raise ValueError(f"Expected one holdout key, got {holdout_keys.tolist()}")
    holdout_key = holdout_keys.iloc[0]
    holdout_geometry = geometry[geometry["event_key"].eq(holdout_key)]

    comparisons = []
    for row in editions.itertuples(index=False):
        if int(row.season) == holdout_year:
            continue
        prior = geometry[geometry["event_key"].eq(row.event_key)]
        metrics = compare_geometry(prior, holdout_geometry, round_specific=True)
        metrics.update({"season": int(row.season), "event_key": row.event_key})
        comparisons.append(metrics)

    rows = []
    for row in editions.itertuples(index=False):
        sample = shots[shots["event_key"].eq(row.event_key)]
        tee = sample[sample["stroke_no"].eq(1) & sample["event_type"].eq("STROKE")]
        cells = tee.groupby(["round_no", "hole_no"]).size()
        coordinate_rate = sample[["start_x", "start_y", "end_x", "end_y"]].notna().all(axis=1).mean()
        rows.append(
            {
                "season": int(row.season),
                "event_key": row.event_key,
                "players": int(sample["source_player_id"].nunique()),
                "player_rounds": int(sample.groupby(["source_player_id", "round_no"]).ngroups),
                "shots": int(len(sample)),
                "tee_shots": int(len(tee)),
                "tee_per_round_hole_min": int(cells.min()),
                "tee_per_round_hole_median": float(cells.median()),
                "tee_per_round_hole_max": int(cells.max()),
                "coordinate_rate": float(coordinate_rate),
            }
        )

    training = shots[shots["season"].lt(holdout_year)]
    training_tee = training[
        training["stroke_no"].eq(1) & training["event_type"].eq("STROKE")
    ]
    training_cells = training_tee.groupby(["round_no", "hole_no"]).size()
    return {
        "event_name": event_name,
        "holdout_year": holdout_year,
        "holdout_event_key": holdout_key,
        "prior_editions": int(editions["season"].lt(holdout_year).sum()),
        "training_shots": int(len(training)),
        "training_tee_per_round_hole": {
            "min": int(training_cells.min()),
            "median": float(training_cells.median()),
            "max": int(training_cells.max()),
        },
        "geometry_vs_holdout": comparisons,
        "edition_samples": rows,
        "sequence_continuity": sequence_continuity(shots),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--holdout-year", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_event(args.event_name, args.holdout_year, args.export)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
