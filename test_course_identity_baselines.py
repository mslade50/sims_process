import sqlite3
from pathlib import Path

import pandas as pd
import pytest
import yaml

import api_utils
import hole_baselines as hb
import scoring_baseline as scoring
import write_base_rates as base_rates


def _write_wind_csv(path: Path) -> None:
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
            {
                "course_num": 886,
                "course_name": "Liberty National",
                "wind_effect_adj_score": 0.19,
                "event_ids": "27",
            },
        ]
    ).to_csv(path, index=False)


def test_wind_lookup_requires_course_when_event_id_spans_venues(tmp_path):
    wind_csv = tmp_path / "wind_test.csv"
    _write_wind_csv(wind_csv)

    effect, source = api_utils.lookup_course_wind_effect(
        course_id=513, event_ids=[27], wind_test_path=str(wind_csv)
    )
    assert effect == pytest.approx(0.12)
    assert "course_id=513" in source

    with pytest.raises(RuntimeError, match="multiple course wind coefficients"):
        api_utils.lookup_course_wind_effect(
            event_ids=[27], wind_test_path=str(wind_csv)
        )


def _create_history_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE player_rounds (
            year INTEGER, event_id INTEGER, event_name TEXT,
            course_num INTEGER, round_num INTEGER, round_date TEXT,
            tour TEXT, player_name TEXT, teetime TEXT, score REAL,
            course_par REAL, sg_ott REAL, sg_app REAL, sg_arg REAL,
            sg_putt REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE player_weather_data_test (
            player_name TEXT, event_id INTEGER, round_date TEXT,
            wind REAL, dew REAL, temp REAL
        )
        """
    )

    events = [
        (2020, 27, "Northern Trust", 700, "2020-08-20"),
        (2020, 476, "WGC St Jude", 513, "2020-07-30"),
        (2021, 27, "Northern Trust", 886, "2021-08-19"),
        (2021, 476, "WGC St Jude", 513, "2021-08-05"),
        (2022, 27, "FedEx St Jude", 513, "2022-08-11"),
    ]
    rows = []
    weather_rows = []
    for year, event_id, name, course_num, date in events:
        for i in range(12):
            southwind = course_num == 513
            am = i < 6
            teetime = f"0{7 + (i % 4)}:00am" if am else "12:30pm"
            score = (68 if am else 70) if southwind else (76 if am else 70)
            spread = float((i % 6) - 2.5)
            if southwind:
                cats = (spread, spread * 0.1, spread * 0.1, spread * 0.1)
            else:
                cats = (spread * 0.1, spread * 0.1, spread * 0.1, spread)
            rows.append(
                (
                    year, event_id, name, course_num, 1, date, "pga",
                    f"player-{i}", teetime, score, 70, *cats,
                )
            )
            weather_rows.append(
                (
                    f"player-{i}", event_id, date,
                    5.0 if southwind else 20.0, 60.0, 80.0,
                )
            )
    conn.executemany(
        "INSERT INTO player_rounds VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows
    )
    conn.executemany(
        "INSERT INTO player_weather_data_test VALUES (?,?,?,?,?,?)", weather_rows
    )
    conn.commit()
    conn.close()


def test_hole_history_follows_course_across_event_id_changes(tmp_path, monkeypatch):
    db = tmp_path / "history.db"
    _create_history_db(db)
    monkeypatch.setattr(hb, "DG_DB", db)
    monkeypatch.setattr(hb, "course_id", 513)
    monkeypatch.setattr(hb, "HISTORICAL_TOUR", "pga")
    monkeypatch.setattr(hb, "HISTORICAL_EVENT_IDS", [])

    history = hb.resolve_course_history(2020, 2022)

    assert list(zip(history["year"], history["event_id"])) == [
        (2020, 476),
        (2021, 476),
        (2022, 27),
    ]
    assert history["event_key"].tolist() == [
        "pga:R2020476",
        "pga:R2021476",
        "pga:R2022027",
    ]


def test_hole_history_rejects_two_same_course_events_in_one_year(
    tmp_path, monkeypatch
):
    db = tmp_path / "history.db"
    _create_history_db(db)
    conn = sqlite3.connect(db)
    conn.execute(
        """
        INSERT INTO player_rounds VALUES (
            2022, 999, 'Second Southwind Event', 513, 1, '2022-10-01',
            'pga', 'player-x', '08:00am', 70, 70, 0, 0, 0, 0
        )
        """
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(hb, "DG_DB", db)
    monkeypatch.setattr(hb, "course_id", 513)
    monkeypatch.setattr(hb, "HISTORICAL_TOUR", "pga")
    monkeypatch.setattr(hb, "HISTORICAL_EVENT_IDS", [])

    with pytest.raises(RuntimeError, match="multiple tournaments"):
        hb.resolve_course_history(2020, 2022)


def test_base_rates_course_diagnostics_exclude_reused_event_venues(
    tmp_path, monkeypatch
):
    db = tmp_path / "history.db"
    _create_history_db(db)
    monkeypatch.setattr(base_rates, "DB_PATH", str(db))

    exact_split, _ = base_rates._query_historical_am_pm([27], course_id=513)
    assert exact_split[1] == pytest.approx(-2.0)
    with pytest.raises(ValueError, match="course_id is required"):
        base_rates._query_historical_am_pm([27])

    _, exact_var, _, _ = base_rates._query_variance_attribution(
        [27], course_id=513
    )
    assert exact_var["sg_ott"] > 90
    with pytest.raises(ValueError, match="course_id is required"):
        base_rates._query_variance_attribution([27])


def test_scoring_weather_fallback_uses_course_not_current_event_id(
    tmp_path, monkeypatch
):
    db = tmp_path / "history.db"
    _create_history_db(db)
    monkeypatch.setattr(scoring, "DG_HISTORICAL_DB", str(db))
    monkeypatch.setattr(scoring, "course_id", 513)
    monkeypatch.setattr(scoring, "tour_override", "pga")

    cache = sqlite3.connect(":memory:")
    scoring._ensure_wind_db_schema(cache)
    scoring._fallback_weather(cache, "st_jude", 2020, pd.DataFrame())
    rows = pd.read_sql_query(
        "SELECT DISTINCT event_id, wind_speed FROM wind_data", cache
    )
    cache.close()

    assert rows.to_dict("records") == [
        {"event_id": 476, "wind_speed": 5.0}
    ]


def test_base_rates_refresh_preserves_skill_docks_without_explicit_reset(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "sim_config.yaml"
    config = {
        "event": {},
        "category_first": {
            "skill_docks": [
                {"player": "coody, pierceson", "sg_dock": 0.5, "dock_pct": 0.25}
            ]
        },
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(base_rates, "ETR_SIM_CONFIG", str(config_path))

    base_rates._update_etr_sim_config(
        {"sg_ott": 1.08}, {"sg_ott": -0.81}, reset_skill_docks=False
    )
    preserved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert preserved["category_first"]["skill_docks"] == config[
        "category_first"
    ]["skill_docks"]

    base_rates._update_etr_sim_config(
        {"sg_ott": 1.08}, {"sg_ott": -0.81}, reset_skill_docks=True
    )
    reset = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert reset["category_first"]["skill_docks"] == []
