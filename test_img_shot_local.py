import csv
import json
import sqlite3

from img_shot_local import (
    _readonly_uri,
    read_hole_geometry,
    read_player_rounds,
    read_player_shots,
    write_player_crosswalk,
)


def build_db(path):
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE archive_events(event_key TEXT PRIMARY KEY,tour TEXT,season INTEGER);
        CREATE TABLE archive_event_sources(event_key TEXT,source_event_id TEXT);
        CREATE TABLE archive_players(event_key TEXT,source_player_id TEXT,player_name TEXT);
        CREATE TABLE sg_models(model_id TEXT PRIMARY KEY,metadata_json TEXT);
        CREATE TABLE archive_player_round_strokes_gained(
          event_key TEXT,round_no INTEGER,source_player_id TEXT,source_team_id TEXT,
          player_name TEXT,model_id TEXT,round_strokes REAL,total_strokes INTEGER,
          valid_sg_strokes INTEGER,complete INTEGER,sg_ott_raw REAL,sg_app_raw REAL,
          sg_arg_raw REAL,sg_putt_raw REAL,sg_total_raw REAL,sg_ott REAL,sg_app REAL,
          sg_arg REAL,sg_putt REAL,sg_bs REAL,sg_t2g REAL,sg_total REAL,
          adjustment_method TEXT,calculated_at TEXT
        );
        CREATE TABLE archive_player_round_skill_signal(
          event_key TEXT,round_no INTEGER,source_player_id TEXT,source_team_id TEXT,
          raw_model_id TEXT,signal_model_id TEXT,course_id TEXT,
          sg_ott_signal REAL,sg_app_signal REAL,sg_arg_signal REAL,sg_putt_signal REAL,
          sg_bs_signal REAL,sg_t2g_signal REAL,sg_total_signal REAL,calculated_at TEXT
        );
        CREATE TABLE archive_shots(
          shot_key TEXT PRIMARY KEY,course_id TEXT,event_type TEXT,shot_distance_yd REAL,
          start_x REAL,start_y REAL,start_z REAL,end_x REAL,end_y REAL,end_z REAL
        );
        CREATE TABLE archive_shot_strokes_gained(
          shot_key TEXT,event_key TEXT,round_no INTEGER,hole_no INTEGER,
          source_player_id TEXT,source_team_id TEXT,player_name TEXT,stroke_no INTEGER,
          start_lie TEXT,start_distance_yd REAL,end_lie TEXT,end_distance_yd REAL,
          sg_category TEXT,sg_raw REAL,sg_valid INTEGER,sg_issue TEXT,
          complete_hole INTEGER,state_inferred TEXT,model_id TEXT
        );
        CREATE TABLE archive_shot_skill_signal(
          shot_key TEXT,raw_model_id TEXT,signal_model_id TEXT,is_penalty INTEGER,
          sg_signal_uncentered REAL,transform_reason TEXT
        );
        CREATE TABLE archive_hole_geometry(
          geometry_key TEXT PRIMARY KEY,event_key TEXT,source TEXT,
          source_event_id TEXT,course_id TEXT,round_no INTEGER,hole_no INTEGER,
          par INTEGER,yardage REAL,coordinate_system TEXT,tee_x REAL,tee_y REAL,
          tee_z REAL,pin_x REAL,pin_y REAL,pin_z REAL,fairway_center_x REAL,
          fairway_center_y REAL,fairway_center_z REAL,first_seen_at TEXT,
          last_seen_at TEXT,payload_json TEXT
        );
        """
    )
    connection.execute("INSERT INTO archive_events VALUES ('pga:R2026525','pga',2026)")
    connection.execute("INSERT INTO archive_event_sources VALUES ('pga:R2026525','R2026525')")
    connection.execute("INSERT INTO archive_players VALUES ('pga:R2026525','12716','Charley Hoffman')")
    connection.executemany(
        "INSERT INTO sg_models VALUES (?,?)",
        [
            ("old", json.dumps({"archive_adapter": "rich_numbered_strokes_course_aware_v2"})),
            ("raw", json.dumps({"archive_adapter": "rich_numbered_strokes_course_aware_v2"})),
            ("signal", json.dumps({
                "status": "validated_pga_2020_2024_plus_2026",
                "method": "predictive_skill_pga_validated_v2",
            })),
        ],
    )
    row = (
        'pga:R2026525',4,'12716','', 'Charley Hoffman','raw',69,69,69,1,
        .1,.2,.3,.4,1.0,.11,.22,.33,.44,.33,.66,1.1,'field','2026-07-28T20:00:00Z'
    )
    connection.execute(
        "INSERT INTO archive_player_round_strokes_gained VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        row,
    )
    connection.execute(
        "INSERT INTO archive_player_round_strokes_gained VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (*row[:5],'old',*row[6:-1],'2026-07-27T20:00:00Z'),
    )
    connection.execute(
        "INSERT INTO archive_player_round_skill_signal VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ('pga:R2026525',4,'12716','','raw','signal','course-a',.1,.2,.3,.4,.3,.6,1.0,'2026-07-28T20:00:01Z'),
    )
    connection.execute(
        "INSERT INTO archive_shots VALUES ('shot-1','course-a','STROKE',250,1,2,3,4,5,6)"
    )
    connection.execute(
        "INSERT INTO archive_shot_strokes_gained VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ('shot-1','pga:R2026525',4,1,'12716','','Charley Hoffman',1,'tee',450,'fairway',200,'ott',.2,1,None,1,'','raw'),
    )
    connection.execute(
        "INSERT INTO archive_shot_skill_signal VALUES ('shot-1','raw','signal',0,.2,'positive_linear')"
    )
    connection.execute(
        """
        INSERT INTO archive_hole_geometry VALUES (
          'geo-4-1','pga:R2026525','pga_tourcast','R2026525',
          'layout:daily-r4',4,1,4,451,'tourcast_course_xyz_feet',
          1,2,3,4,5,6,NULL,NULL,NULL,'2026-07-28T20:00:00Z',
          '2026-07-28T20:00:00Z','{}'
        )
        """
    )
    connection.commit()
    connection.close()


def test_readonly_uri_does_not_require_sqlite_sidecar_writes(tmp_path):
    uri = _readonly_uri((tmp_path / "shots.sqlite3").resolve())
    assert "mode=ro" in uri
    assert "immutable=1" in uri


def test_reads_newest_player_and_signal_models(tmp_path):
    db = tmp_path / "shots.sqlite3"
    build_db(db)
    event_key, rows = read_player_rounds(525, 4, db_path=db, tour="pga", season=2026)
    assert event_key == "pga:R2026525"
    assert len(rows) == 1
    assert rows[0]["img_player_id"] == "12716"
    assert rows[0]["raw_model_id"] == "raw"
    assert rows[0]["sg_total"] == 1.1
    assert rows[0]["sg_total_signal"] == 1.0


def test_reads_shot_level_accounting_and_signal(tmp_path):
    db = tmp_path / "shots.sqlite3"
    build_db(db)
    event_key, rows = read_player_shots(525, 4, db_path=db, tour="pga", season=2026)
    assert event_key == "pga:R2026525"
    assert len(rows) == 1
    assert rows[0]["shot_key"] == "shot-1"
    assert rows[0]["sg_category"] == "ott"
    assert rows[0]["signal_model_id"] == "signal"


def test_reads_round_specific_hole_geometry_without_course_join(tmp_path):
    db = tmp_path / "shots.sqlite3"
    build_db(db)
    event_key, rows = read_hole_geometry(
        525, 4, db_path=db, tour="pga", season=2026
    )
    assert event_key == "pga:R2026525"
    assert len(rows) == 1
    assert rows[0]["course_id"] == "layout:daily-r4"
    assert rows[0]["yardage"] == 451


def test_crosswalk_is_atomic_and_persistent(tmp_path):
    target = tmp_path / "crosswalk.csv"
    match = {
        "source_player_id": "12716", "source_player_name": "Charley Hoffman",
        "datagolf_player_name": "hoffman, charley",
        "canonical_name": "hoffman, charley", "event_key": "pga:R2026525",
    }
    write_player_crosswalk([match], target)
    write_player_crosswalk([match], target)
    with target.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["source_player_id"] == "12716"
    assert rows[0]["first_seen_at"]
    assert rows[0]["last_seen_at"]
def test_api_prefers_local_archive_and_normalizes_names(tmp_path, monkeypatch):
    db = tmp_path / "shots.sqlite3"
    build_db(db)
    monkeypatch.setenv("IMG_SHOT_DB_PATH", str(db))
    monkeypatch.delenv("IMG_SHOT_SOURCE", raising=False)
    from api_utils import fetch_img_player_rounds

    frame = fetch_img_player_rounds(525, 4, season=2026)
    assert frame.attrs == {"event_key": "pga:R2026525", "source": "local_sqlite"}
    assert frame.loc[0, "player_name"] == "hoffman, charley"
    assert frame.loc[0, "source_player_name"] == "Charley Hoffman"
