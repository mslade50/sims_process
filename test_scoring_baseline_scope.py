import os
import sqlite3

import pandas as pd
import pytest


os.environ.setdefault("COEFFS_FROM_CACHE", "1")

import scoring_baseline as sb  # noqa: E402


def _history_db(path):
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE player_rounds (
            year INTEGER,
            event_id INTEGER,
            event_name TEXT,
            course_num INTEGER,
            round_num INTEGER,
            round_date TEXT,
            tour TEXT,
            score REAL,
            teetime TEXT
        )
        """
    )
    rows = [
        # Reused event ID at a different venue: must never enter course 513.
        (2021, 27, "THE NORTHERN TRUST", 886, 1, "2021-08-19", "pga", 71, "08:00am"),
        (2021, 27, "THE NORTHERN TRUST", 886, 1, "2021-08-19", "pga", 72, "08:10am"),
        # Same venue under an older event ID: must be included automatically.
        (2021, 476, "WGC-FedEx St. Jude Invitational", 513, 1, "2021-08-05", "pga", 68, "08:00am"),
        (2021, 476, "WGC-FedEx St. Jude Invitational", 513, 1, "2021-08-05", "pga", 69, "08:10am"),
        (2022, 27, "FedEx St. Jude Championship", 513, 1, "2022-08-11", "pga", 69, "08:00am"),
        (2022, 27, "FedEx St. Jude Championship", 513, 1, "2022-08-11", "pga", 70, "08:10am"),
    ]
    conn.executemany(
        "INSERT INTO player_rounds VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
    )
    conn.commit()
    conn.close()


def test_course_scope_follows_venue_across_event_id_changes(tmp_path, monkeypatch):
    db = tmp_path / "history.db"
    _history_db(db)
    monkeypatch.setattr(sb, "DG_HISTORICAL_DB", str(db))
    monkeypatch.setattr(sb, "tour_override", "pga")

    events = sb.validate_history_scope(
        [27], 2019, course_num=513, restrict_event_ids=False
    )
    assert set(events["event_id"]) == {27, 476}
    assert set(events["course_num"]) == {513}

    scoring = sb.get_scoring_averages(
        [27], 2019, course_num=513, restrict_event_ids=False
    )
    r1 = scoring.set_index("year")["avg_score"]
    assert r1.loc[2021] == pytest.approx(68.5)
    assert r1.loc[2022] == pytest.approx(69.5)

    dates = sb.get_tournament_dates(
        [27], 2019, course_num=513, restrict_event_ids=False
    )
    assert list(dates["round_date"].dt.strftime("%Y-%m-%d")) == [
        "2021-08-05",
        "2022-08-11",
    ]


def test_course_scope_rejects_two_events_in_same_year(tmp_path, monkeypatch):
    db = tmp_path / "history.db"
    _history_db(db)
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO player_rounds VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (2022, 999, "Second TPC Southwind Event", 513, 1,
         "2022-10-01", "pga", 70, "08:00am"),
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(sb, "DG_HISTORICAL_DB", str(db))
    monkeypatch.setattr(sb, "tour_override", "pga")

    with pytest.raises(RuntimeError, match="Multiple tournaments"):
        sb.validate_history_scope(
            [27], 2019, course_num=513, restrict_event_ids=False
        )


def test_wind_coefficient_prefers_exact_course(tmp_path, monkeypatch):
    wind_csv = tmp_path / "wind_test.csv"
    pd.DataFrame(
        [
            {
                "course_num": 513,
                "course_name": "TPC Southwind",
                "wind_effect_adj_score": 0.12,
                "event_ids": "476, 27",
            },
            {
                "course_num": 700,
                "course_name": "TPC Boston",
                "wind_effect_adj_score": 0.06,
                "event_ids": "27",
            },
        ]
    ).to_csv(wind_csv, index=False)
    monkeypatch.setattr(sb, "WIND_TEST_CSV", str(wind_csv))

    blended, course = sb.get_wind_coefficient([27], 0.155, 0.0, 513)
    assert course == pytest.approx(0.12)
    assert blended == pytest.approx(0.4 * 0.12 + 0.6 * 0.155)

    fallback, course = sb.get_wind_coefficient([27], 0.155, 0.0, 999)
    assert course == pytest.approx(0.08)
    assert fallback == pytest.approx(0.4 * 0.08 + 0.6 * 0.155)
