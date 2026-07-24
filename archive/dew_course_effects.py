"""
Per-course dewpoint coefficients -> permanent_data/dew_test.csv

Provenance: July 2026 dew deep-dive. Field scoring regressed on realized
dewpoint using dg_historical.db (player_rounds joined with
player_weather_data_test, 2017-2025, ~1,570 tournament round-days). The
global linear coefficient (-0.022 strokes/degF) validated cleanly, but
course-to-course slope heterogeneity is real (I2 ~ 60%, tau ~ 0.021):
tropical/coastal venues with a pinned dew range show no effect, while
fall bermuda/transition-zone venues run 1.5-2x baseline.

Method per course (>= MIN_DAYS round-days, dew SD >= MIN_DEW_SD):
  1. OLS of course-demeaned field scoring avg on course-centered dew,
     controlling for wind and field skill -> raw slope + SE.
  2. Empirical-Bayes shrinkage toward the precision-weighted pooled slope:
     w = tau^2 / (tau^2 + se^2), tau from DerSimonian-Laird.
  3. Clamp to [CLAMP_LO, 0]: positive dew slopes (humid = harder) are
     physically implausible noise, so tropical venues land at ~0
     (dew adjustment effectively off there).

Consumed by api_utils.compute_dew_factor() (analog of compute_wind_factor,
which blends wind_test.csv the same way). Courses not in the CSV fall back
to the sim_inputs baseline blend.

Re-run yearly (or after adding a season to dg_historical.db):
    python archive/dew_course_effects.py
"""
import os
import sqlite3

import numpy as np
import pandas as pd
import statsmodels.api as sm

DB = os.path.join(os.path.expanduser("~"), "OneDrive", "dg_historical.db")
OUT = os.path.join(os.path.dirname(__file__), "..", "permanent_data", "dew_test.csv")

MIN_DAYS = 20       # round-days required for a course-specific estimate
MIN_DEW_SD = 2.0    # degF of dew variance required (else slope is noise)
MIN_PLAYERS = 25    # players required for a round-day scoring average
CLAMP_LO = -0.06
CLAMP_HI = 0.0


def build_round_days():
    """Round-day field scoring averages joined with realized weather."""
    conn = sqlite3.connect(DB)
    w = pd.read_sql(
        "SELECT player_name, event_id, round_date, course_num, wind, dew "
        "FROM player_weather_data_test", conn)
    pr = pd.read_sql(
        "SELECT player_name, event_id, year, round_num, round_date, course_num, "
        "course_par, score, skill, course_name, event_name "
        "FROM player_rounds WHERE tour='pga'", conn)
    conn.close()

    for df in (w, pr):
        df["round_date"] = pd.to_datetime(df["round_date"]).dt.normalize()
        df["player_name"] = df["player_name"].str.strip().str.lower()

    m = pr.merge(w, on=["player_name", "event_id", "round_date", "course_num"],
                 how="inner")
    m = m.dropna(subset=["dew", "wind", "score", "course_par"])
    m["vs_par"] = m["score"] - m["course_par"]
    m = m[(m["vs_par"] > -15) & (m["vs_par"] < 20)]
    m["skill"] = m["skill"].fillna(m["skill"].median())

    day = m.groupby(
        ["event_id", "year", "course_num", "course_name", "round_num", "round_date"]
    ).agg(
        n=("score", "size"),
        vs_par=("vs_par", "mean"),
        dew=("dew", "mean"),
        wind=("wind", "mean"),
        skill=("skill", "mean"),
    ).reset_index()
    return day[day["n"] >= MIN_PLAYERS]


def course_slopes(day):
    """Raw OLS dew slope per eligible course (wind + field-skill controls)."""
    rows = []
    for cid, g in day.groupby("course_num"):
        if len(g) < MIN_DAYS or g["dew"].std() < MIN_DEW_SD:
            continue
        X = sm.add_constant(pd.DataFrame({
            "dew_c": g["dew"] - g["dew"].mean(),
            "wind_c": g["wind"] - g["wind"].mean(),
            "skill": g["skill"],
        }))
        fit = sm.OLS(g["vs_par"] - g["vs_par"].mean(), X).fit()
        rows.append({
            "course_num": int(cid),
            "course_name": g["course_name"].iloc[-1],
            "raw_slope": fit.params["dew_c"],
            "se": fit.bse["dew_c"],
            "n_round_days": len(g),
            "dew_sd": g["dew"].std(),
            "years": f"{g['year'].min()}-{g['year'].max()}",
            "event_ids": ",".join(str(e) for e in sorted(g["event_id"].unique())),
        })
    return pd.DataFrame(rows)


def shrink(cs):
    """Empirical-Bayes shrinkage toward the pooled slope (DerSimonian-Laird tau)."""
    w = 1.0 / cs["se"] ** 2
    pooled = float((cs["raw_slope"] * w).sum() / w.sum())
    q = float(((cs["raw_slope"] - pooled) ** 2 * w).sum())
    k = len(cs) - 1
    tau2 = max(0.0, (q - k) / (w.sum() - (w ** 2).sum() / w.sum()))
    print(f"pooled slope {pooled:+.4f} | Q={q:.1f} df={k} | tau={np.sqrt(tau2):.4f}")

    wt = tau2 / (tau2 + cs["se"] ** 2)
    cs["shrink_wt"] = wt.round(3)
    cs["dew_coef"] = (wt * cs["raw_slope"] + (1 - wt) * pooled).clip(
        CLAMP_LO, CLAMP_HI).round(4)
    cs["raw_slope"] = cs["raw_slope"].round(4)
    cs["se"] = cs["se"].round(4)
    cs["dew_sd"] = cs["dew_sd"].round(1)
    return cs.sort_values("dew_coef")


def main():
    day = build_round_days()
    print(f"round-days: {len(day)} | courses: {day['course_num'].nunique()} "
          f"| {day['year'].min()}-{day['year'].max()}")

    cs = course_slopes(day)
    print(f"courses with >= {MIN_DAYS} round-days and dew SD >= {MIN_DEW_SD}: {len(cs)}")
    cs = shrink(cs)

    cols = ["course_num", "course_name", "dew_coef", "raw_slope", "se",
            "shrink_wt", "n_round_days", "dew_sd", "years", "event_ids"]
    out = os.path.normpath(OUT)
    cs[cols].to_csv(out, index=False)
    print(f"wrote {out} ({len(cs)} courses)")
    print(cs[cols].drop(columns=["event_ids"]).to_string(index=False))


if __name__ == "__main__":
    main()
