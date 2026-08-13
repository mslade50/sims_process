"""Build adj_hole_dist_{tourney}_{course_id}.csv from our own shot archive.

Replaces OneDrive/baselines.py, which pulled hole scores from ETR's hosted
SQL Server, weather from a stale local pga_weather.csv, and stripped skill
with the retired pre-refit coefficient model, weighting all years equally.

This version:
  - Hole scores: IMG shot archive (archive_inspect.sqlite3) - per-player
    per-hole stroke counts (STROKE + PENALTY events, holed-out holes only),
    with par/yardage from hole_geometry.
  - Weather strip: player_weather_data_test (complete panel) field wind per
    year/round x the course wind blend (wind_test.csv x 0.4 + baseline_wind
    x 0.6, or wind_override), plus a dew term centered on the course's
    multi-year average. Rounds in weather_exclusion_rounds.csv (stored
    round_date/teetime no longer matches actual play) get no weather strip.
  - Skill strip: field mean skill_est from model_results2 per year/round
    (merge_asof at the round date) - positive = stronger-than-average field,
    added back to neutralize.
  - Recency decay: each season's hole counts are weighted by
    YEAR_DECAY ** (target_season - season) when pooling, so recent editions
    dominate the scoring expectation.

Output format is byte-compatible with the old adj_hole_dist file
(Round, Hole, Par, mean, adj_mean, integer score-count columns); the file is
also copied to ~/OneDrive/etr-golf-sims. Verification tables print per year.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from sim_inputs import (
    event_ids, tourney, course_id, tour, wind_override, baseline_wind,
    dew_calculation,
)
try:
    from sim_inputs import historical_event_ids as _historical_event_ids
except ImportError:
    _historical_event_ids = []
try:
    from sim_inputs import tour_override as _historical_tour
except ImportError:
    _historical_tour = tour

from api_utils import lookup_course_wind_effect

ARCHIVE_DB = Path(os.getenv(
    "IMG_INSPECT_DB",
    r"C:\Users\McKinley Slade\dev\golf_scraping\output\export"
    r"\archive_inspect.sqlite3",
))
DG_DB = Path(os.getenv(
    "DG_DB", r"C:\Users\McKinley Slade\dev\sim_prep\work\dg_historical.db"
))
CROSSWALK = Path(r"C:\Users\McKinley Slade\dev\sim_prep\shot_bridge\crosswalk.parquet")
EXCLUSIONS = Path(r"C:\Users\McKinley Slade\dev\sim_prep\weather_exclusion_rounds.csv")
WIND_TEST = Path(__file__).resolve().parent / "permanent_data" / "wind_test.csv"
ETR_REPO = Path.home() / "OneDrive" / "etr-golf-sims"

YEAR_DECAY = float(os.getenv("HOLE_BASELINE_YEAR_DECAY", "0.85"))
MAX_HOLE_SCORE = 12
HISTORY_MIN_YEAR = int(os.getenv("HOLE_BASELINE_MIN_YEAR", "2018"))
# Never learn a pre-event baseline from the in-progress tournament.
HISTORY_MAX_YEAR = int(os.getenv(
    "HOLE_BASELINE_MAX_YEAR", str(pd.Timestamp.now().year - 1)
))
HISTORICAL_EVENT_IDS = [int(x) for x in (_historical_event_ids or [])]
HISTORICAL_TOUR = str(_historical_tour or tour)


def _ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)


def resolve_course_history(
    min_year: int = HISTORY_MIN_YEAR,
    max_year: int = HISTORY_MAX_YEAR,
) -> pd.DataFrame:
    """Resolve the exact historical tournament at this physical course/year."""
    clauses = ["course_num = ?", "tour = ?", "year BETWEEN ? AND ?"]
    params: list[object] = [course_id, HISTORICAL_TOUR, min_year, max_year]
    if HISTORICAL_EVENT_IDS:
        marks = ",".join("?" for _ in HISTORICAL_EVENT_IDS)
        clauses.append(f"event_id IN ({marks})")
        params.extend(HISTORICAL_EVENT_IDS)

    with _ro(DG_DB) as conn:
        history = pd.read_sql_query(
            f"""
            SELECT year, event_id, event_name, course_num,
                   MIN(round_date) AS event_start
            FROM player_rounds
            WHERE {' AND '.join(clauses)}
            GROUP BY year, event_id, event_name, course_num
            ORDER BY year, event_start
            """,
            conn, params=params,
        )
    if history.empty:
        override = (f", event_ids={HISTORICAL_EVENT_IDS}"
                    if HISTORICAL_EVENT_IDS else "")
        raise RuntimeError(
            f"no {HISTORICAL_TOUR} history for course_id={course_id}, "
            f"years={min_year}-{max_year}{override}"
        )

    duplicates = history.groupby("year").size()
    ambiguous_years = duplicates[duplicates > 1].index.tolist()
    if ambiguous_years:
        details = history.loc[
            history["year"].isin(ambiguous_years),
            ["year", "event_id", "event_name", "event_start"],
        ].to_dict("records")
        raise RuntimeError(
            "multiple tournaments at the configured course in one year: "
            f"{details}. Set historical_event_ids explicitly to disambiguate."
        )

    history["year"] = history["year"].astype(int)
    history["event_id"] = history["event_id"].astype(int)
    history["event_key"] = history.apply(
        lambda row: f"{HISTORICAL_TOUR}:R{row['year']}{row['event_id']:03d}",
        axis=1,
    )
    print(
        f"[scope] course_id={course_id}; "
        + ", ".join(
            f"{row.year}:{row.event_id}" for row in history.itertuples()
        )
    )
    return history


def _pair_filter(history: pd.DataFrame, alias: str = "") -> tuple[str, list[int]]:
    """SQL filter for the resolved (year, event_id) pairs."""
    if history.empty:
        raise RuntimeError("course history is empty")
    prefix = f"{alias}." if alias else ""
    clauses = []
    params: list[int] = []
    for row in history.itertuples():
        clauses.append(f"({prefix}year = ? AND {prefix}event_id = ?)")
        params.extend([int(row.year), int(row.event_id)])
    return "(" + " OR ".join(clauses) + ")", params


def load_hole_scores(
    history: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-player hole scores + geometry for resolved same-course events."""
    keys = history["event_key"].tolist()
    placeholders = ",".join("?" for _ in keys)
    with _ro(ARCHIVE_DB) as conn:
        events = pd.read_sql_query(
            f"SELECT event_key, season FROM events WHERE event_key IN ({placeholders})",
            conn, params=keys,
        )
        if events.empty:
            raise RuntimeError(
                f"shot archive has no events for course_id={course_id} "
                f"({tourney}); tried {keys}"
            )
        missing = sorted(set(keys) - set(events["event_key"]))
        if missing:
            print(f"[holes] WARNING: archive missing same-course events: {missing}")
        event_keys = tuple(events["event_key"])
        marks = ",".join("?" for _ in event_keys)
        scores = pd.read_sql_query(
            f"""
            SELECT s.event_key, e.season, s.round_no, s.hole_no,
                   s.source_player_id,
                   SUM(s.event_type IN ('STROKE','PENALTY')) AS score,
                   MAX(s.event_type = 'STROKE' AND s.ball_holed = 1) AS holed
            FROM shots s JOIN events e ON s.event_key = e.event_key
            WHERE s.event_key IN ({marks})
            GROUP BY s.event_key, e.season, s.round_no, s.hole_no, s.source_player_id
            """,
            conn, params=list(event_keys),
        )
        geometry = pd.read_sql_query(
            f"SELECT event_key, round_no, hole_no, par, yardage "
            f"FROM hole_geometry WHERE event_key IN ({marks})",
            conn, params=list(event_keys),
        )
    total = len(scores)
    scores = scores[
        (scores["holed"] == 1)
        & scores["score"].between(1, MAX_HOLE_SCORE)
    ].copy()
    print(f"[holes] {len(scores):,} completed player-holes "
          f"({total - len(scores):,} incomplete/invalid dropped) across "
          f"{sorted(events['season'].tolist())}")
    seasons = events.set_index("event_key")["season"]
    geometry["season"] = geometry["event_key"].map(seasons)
    used_history = history[history["event_key"].isin(event_keys)].copy()
    return scores, geometry, used_history


def load_round_dates(history: pd.DataFrame) -> pd.DataFrame:
    """Round dates for the resolved same-course tournament editions."""
    pair_where, params = _pair_filter(history)
    with _ro(DG_DB) as conn:
        rounds = pd.read_sql_query(
            f"""
            SELECT year, event_id, round_num, round_date, COUNT(*) n
            FROM player_rounds
            WHERE {pair_where} AND course_num = ? AND tour = ?
            GROUP BY 1,2,3,4
            """,
            conn, params=params + [course_id, HISTORICAL_TOUR],
        )
    rounds["round_date"] = pd.to_datetime(rounds["round_date"])
    # one date per (year, round): keep the row with the most players
    rounds = (rounds.sort_values("n", ascending=False)
              .drop_duplicates(["year", "round_num"], keep="first"))
    return rounds[["year", "event_id", "round_num", "round_date"]]


def wind_blend() -> float:
    if wind_override:
        print(f"[wind] using wind_override={wind_override}")
        return float(wind_override)
    try:
        course_effect, source = lookup_course_wind_effect(
            course_id=course_id,
            event_ids=event_ids,
            wind_test_path=str(WIND_TEST),
        )
    except FileNotFoundError:
        course_effect, source = 0.08, "default (wind_test.csv missing)"
    course_effect = max(course_effect, 0.06)
    blended = course_effect * 0.4 + baseline_wind * 0.6
    print(f"[wind] {source}; course_effect={course_effect:.3f} baseline={baseline_wind} "
          f"-> blend={blended:.4f}")
    return blended


def load_weather(history: pd.DataFrame, round_dates: pd.DataFrame) -> pd.DataFrame:
    """Field-average wind/dew per (year, round_num), exclusion-aware."""
    pair_where, params = _pair_filter(history, alias="p")
    with _ro(DG_DB) as conn:
        weather = pd.read_sql_query(
            f"""
            SELECT p.year, p.event_id, p.round_num, p.round_date,
                   AVG(w.wind) AS avg_wind, AVG(w.dew) AS avg_dew,
                   SUM(w.wind IS NOT NULL) AS n_wx, COUNT(*) AS n
            FROM player_rounds p
            LEFT JOIN player_weather_data_test w
              ON p.player_name = w.player_name
             AND p.event_id = w.event_id AND p.round_date = w.round_date
            WHERE {pair_where} AND p.course_num = ? AND p.tour = ?
            GROUP BY 1, 2, 3, 4
            """,
            conn, params=params + [course_id, HISTORICAL_TOUR],
        )
    weather["round_date"] = pd.to_datetime(weather["round_date"])
    weather = weather.merge(
        round_dates,
        on=["year", "event_id", "round_num", "round_date"],
    )
    factor = wind_blend()
    weather["wind_adj"] = weather["avg_wind"] * factor
    weather["dew_adj"] = (
        (weather["avg_dew"] - weather["avg_dew"].mean()) * dew_calculation
    )
    if EXCLUSIONS.is_file():
        excluded = pd.read_csv(EXCLUSIONS, usecols=["event_id", "round_date"])
        excluded["event_id"] = pd.to_numeric(
            excluded["event_id"], errors="coerce"
        )
        excluded["round_date"] = pd.to_datetime(excluded["round_date"])
        excluded_pairs = set(
            zip(excluded["event_id"], excluded["round_date"])
        )
        mask = np.array([
            (row.event_id, row.round_date) in excluded_pairs
            for row in weather.itertuples()
        ])
        if mask.any():
            print(f"[weather] zeroing weather strip for {int(mask.sum())} "
                  f"chaotic round(s) (schedule/teetime mismatch)")
            weather.loc[mask, ["wind_adj", "dew_adj"]] = 0.0
    weather["weather_adj"] = weather["wind_adj"] + weather["dew_adj"].fillna(0.0)
    coverage = (weather["n_wx"].sum() / weather["n"].sum()
                if weather["n"].sum() else 0.0)
    print(f"[weather] panel coverage {coverage:.1%} over "
          f"{len(weather)} event-rounds")
    return weather[["year", "round_num", "wind_adj", "dew_adj", "weather_adj"]]


def load_field_skill(history: pd.DataFrame, round_dates: pd.DataFrame) -> pd.DataFrame:
    """Mean field skill_est per (year, round_num) via crosswalked players."""
    crosswalk = pd.read_parquet(
        CROSSWALK,
        columns=["event_key", "source_player_id", "round_no", "dg_player_name",
                 "event_id", "year", "round_num", "round_date"],
    )
    event_keys = set(history["event_key"])
    crosswalk = crosswalk[crosswalk["event_key"].isin(event_keys)].copy()
    missing = sorted(event_keys - set(crosswalk["event_key"]))
    if missing:
        print(f"[skill] WARNING: crosswalk missing same-course events: {missing}")
    if crosswalk.empty:
        print("[skill] WARNING: no crosswalk rows; field neutralization unavailable")
        return pd.DataFrame(columns=["year", "round_num", "field_adj"])
    crosswalk = crosswalk.rename(columns={"dg_player_name": "player_name"})
    crosswalk["round_date"] = pd.to_datetime(crosswalk["round_date"])
    with _ro(DG_DB) as conn:
        skills = pd.read_sql_query(
            "SELECT player_name, Date, skill_est FROM model_results2 "
            "WHERE skill_est IS NOT NULL", conn,
        )
    skills["Date"] = pd.to_datetime(skills["Date"], errors="coerce")
    skills = skills.dropna(subset=["Date"]).sort_values("Date")
    crosswalk = crosswalk.dropna(subset=["round_date"]).sort_values("round_date")
    merged = pd.merge_asof(
        crosswalk, skills, left_on="round_date", right_on="Date",
        by="player_name", direction="backward",
    )
    merged["skill_est"] = merged["skill_est"].fillna(-1.5)
    field = (merged.groupby(["year", "round_num"])["skill_est"]
             .mean().reset_index(name="field_adj"))
    return field


def build(scores: pd.DataFrame, geometry: pd.DataFrame,
          weather: pd.DataFrame, field: pd.DataFrame) -> pd.DataFrame:
    target_season = int(scores["season"].max()) + 1
    adjustments = weather.merge(field, on=["year", "round_num"], how="outer")
    adjustments[["weather_adj", "field_adj"]] = (
        adjustments[["weather_adj", "field_adj"]].fillna(0.0)
    )

    par = geometry.rename(columns={"round_no": "round_num"})
    par_by_round = (par.groupby(["event_key", "season", "round_num"])["par"].sum()
                    .reset_index(name="course_par"))

    rows = scores.rename(columns={"round_no": "round_num"}).merge(
        par[["event_key", "season", "round_num", "hole_no", "par"]],
        on=["event_key", "season", "round_num", "hole_no"], how="left",
    ).merge(
        par_by_round, on=["event_key", "season", "round_num"], how="left"
    )
    rows = rows.merge(
        adjustments.rename(columns={"year": "season"}),
        on=["season", "round_num"], how="left",
    )
    rows[["weather_adj", "field_adj"]] = rows[["weather_adj", "field_adj"]].fillna(0.0)
    rows["weight"] = YEAR_DECAY ** (target_season - rows["season"])
    rows["wind_adj_per_hole"] = rows["weather_adj"] * rows["par"] / rows["course_par"]
    rows["field_adj_per_hole"] = rows["field_adj"] * rows["par"] / rows["course_par"]

    print("\nPer-season neutralization (18-hole totals) and pooling weights:")
    season_view = (rows.groupby("season")
                   .agg(weight=("weight", "first"),
                        raw_mean=("score", "mean"),
                        weather=("weather_adj", "mean"),
                        field=("field_adj", "mean"))
                   .reset_index())
    season_view["raw_18"] = season_view["raw_mean"] * 18
    print(season_view[["season", "weight", "raw_18", "weather", "field"]]
          .round(3).to_string(index=False))

    # Pool decayed counts per (Round, Hole, score)
    score_range = range(1, MAX_HOLE_SCORE + 1)
    grouped_rows = []
    for (round_num, hole), group in rows.groupby(["round_num", "hole_no"]):
        counts = {
            str(s): float(group.loc[group["score"] == s, "weight"].sum())
            for s in score_range
        }
        weight_sum = group["weight"].sum()
        mean = float((group["score"] * group["weight"]).sum() / weight_sum)
        adj_mean = mean - float(
            (group["wind_adj_per_hole"] * group["weight"]).sum() / weight_sum
        ) + float(
            (group["field_adj_per_hole"] * group["weight"]).sum() / weight_sum
        )
        par_mode = int(group["par"].mode().iat[0])
        grouped_rows.append({"Round": int(round_num), "Hole": int(hole),
                             "Par": par_mode, "mean": mean, "adj_mean": adj_mean,
                             **counts})
    grouped = pd.DataFrame(grouped_rows).sort_values(["Round", "Hole"])

    score_columns = [str(s) for s in score_range]

    def rescale(row) -> np.ndarray:
        counts = np.array([row[c] for c in score_columns], dtype=float)
        total = counts.sum()
        if total == 0 or np.isnan(row["adj_mean"]):
            return counts
        mask = counts > 0
        values = np.array([float(c) for c in score_columns])[mask]
        active = counts[mask]

        def objective(scale):
            dist = active * scale
            return (row["adj_mean"] - (values * dist).sum() / dist.sum()) ** 2

        result = minimize(objective, np.ones(mask.sum()),
                          bounds=[(0, None)] * int(mask.sum()), method="L-BFGS-B")
        scaled = active * result.x if result.success else active
        out = np.zeros_like(counts)
        out[mask] = scaled
        return out / out.sum() * total

    adjusted = np.vstack(grouped.apply(rescale, axis=1))
    final = pd.concat(
        [grouped[["Round", "Hole", "Par", "mean", "adj_mean"]].reset_index(drop=True),
         pd.DataFrame(adjusted, columns=score_columns)], axis=1,
    )
    # drop all-zero score columns to mirror the old file's compact shape
    final = final.drop(columns=[c for c in score_columns if final[c].sum() == 0])
    return final


def main() -> None:
    history = resolve_course_history()
    scores, geometry, history = load_hole_scores(history)
    round_dates = load_round_dates(history)
    weather = load_weather(history, round_dates)
    field = load_field_skill(history, round_dates)
    final = build(scores, geometry, weather, field)

    out_name = f"adj_hole_dist_{tourney}_{course_id}.csv"
    final.to_csv(out_name, index=False)
    print(f"\n[ok] wrote {out_name}: {len(final)} hole-rounds, "
          f"par={int(final[final['Round'] == 1]['Par'].sum())}, "
          f"decay={YEAR_DECAY}/yr")

    course_par = final[final["Round"] == 1][["Hole", "Par"]]
    course_par.to_csv("course_par.csv", index=False)

    verify = final.groupby("Round").agg(
        raw_18=("mean", "sum"), neutral_18=("adj_mean", "sum")).round(3)
    print(verify.to_string())

    if ETR_REPO.is_dir():
        shutil.copy2(out_name, ETR_REPO / out_name)
        print(f"[ok] copied {out_name} -> {ETR_REPO}")


if __name__ == "__main__":
    main()
