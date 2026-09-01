import os
import sqlite3

import pandas as pd


os.environ.setdefault("COEFFS_FROM_CACHE", "1")

import scoring_baseline as sb  # noqa: E402
import sheets_storage  # noqa: E402


def _history_db(path):
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE player_rounds (
                year INTEGER, event_id INTEGER, event_name TEXT, tour TEXT,
                course_num INTEGER, course_name TEXT, course_par INTEGER
            )
            """
        )
        conn.executemany(
            "INSERT INTO player_rounds VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (2023, 60, "TOUR Championship", "pga", 688, "East Lake Golf Club", 70),
                (2024, 60, "TOUR Championship", "pga", 933, "East Lake Golf Club", 71),
                (2025, 60, "TOUR Championship", "pga", 688, "East Lake Golf Club", 70),
                # Same event label at another venue must stay out.
                (2025, 60, "TOUR Championship", "pga", 999, "Other Course", 72),
            ],
        )


def _geometry_frame(*, par=70, source="local_sqlite"):
    rows = []
    for round_no, total in {1: 7270, 2: 7250, 3: 7260, 4: 7369}.items():
        base, extra = divmod(total, 18)
        for hole_no in range(1, 19):
            canonical = base + (1 if hole_no <= extra else 0)
            rows.append({
                "round_no": round_no,
                "hole_no": hole_no,
                "par": 4 if hole_no <= par - 54 else 3,
                "course_id": f"layout-r{round_no}",
                "yardage": canonical,
                "official_yardage": canonical - 5,
                "actual_yardage": canonical,
            })
    frame = pd.DataFrame(rows)
    frame.attrs["source"] = source
    return frame


def test_history_editions_use_exact_course_ids_and_add_current_year(tmp_path, monkeypatch):
    db = tmp_path / "history.sqlite3"
    _history_db(db)
    monkeypatch.setattr(sb, "DG_HISTORICAL_DB", str(db))
    monkeypatch.setattr(sb, "yardage_history_course_ids", [688, 933])
    monkeypatch.setattr(sb, "course_id", 688)
    monkeypatch.setattr(sb, "course_par", 70)
    monkeypatch.setattr(sb, "event_ids", [60])
    monkeypatch.setattr(sb, "start_yr", 2023)
    monkeypatch.setattr(sb, "tour_override", "pga")
    monkeypatch.setattr(sb, "HISTORICAL_EVENT_FILTER_EXPLICIT", False)

    editions = sb.load_yardage_history_editions(current_year=2026)

    assert [(row["year"], row["course_id"]) for row in editions] == [
        (2023, 688), (2024, 933), (2025, 688), (2026, 688)
    ]
    assert editions[-1]["event_key"] == "pga:R2026060"
    assert all(row["course_name"] == "East Lake Golf Club" for row in editions)


def test_geometry_summary_prefers_actual_yardage_and_fails_closed():
    edition = {
        "year": 2026,
        "event_id": 60,
        "event_name": "TOUR Championship",
        "course_id": 688,
        "course_name": "East Lake Golf Club",
        "course_par": 70,
        "event_key": "pga:R2026060",
    }
    frame = _geometry_frame()
    summary, issue = sb._summarize_yardage_geometry(frame, edition)
    assert issue is None
    assert [summary[f"r{rnd}_yards"] for rnd in range(1, 5)] == [
        7270, 7250, 7260, 7369
    ]

    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    summary, issue = sb._summarize_yardage_geometry(duplicate, edition)
    assert summary is None
    assert "complete 18-hole" in issue

    multi_layout = frame.copy()
    multi_layout.loc[multi_layout.index[0], "course_id"] = "other-layout"
    summary, issue = sb._summarize_yardage_geometry(multi_layout, edition)
    assert summary is None
    assert "layout identities" in issue


def test_yardage_block_uses_live_sheet_formulas():
    row = {
        "year": 2026,
        "event_name": "TOUR Championship",
        "course_id": 688,
        "course_name": "East Lake Golf Club",
        "par": 70,
        "r1_yards": 7270,
        "r2_yards": 7250,
        "r3_yards": 7260,
        "r4_yards": 7369,
        "event_key": "pga:R2026060",
    }
    block = sb.build_yardage_history_block([row], start_row=32)
    data = block[2]

    assert data[:8] == [
        2026, "TOUR Championship", 688, 70, 7270, 7250, 7260, 7369
    ]
    assert data[8:] == [
        "=ROUND(F34-E34,1)",
        "=ROUND(G34-AVERAGE(E34:F34),1)",
        "=ROUND(H34-AVERAGE(E34:G34),1)",
        "pga:R2026060",
    ]


class _FakeSpreadsheet:
    def __init__(self):
        self.dimension_updates = []

    def batch_update(self, payload):
        self.dimension_updates.append(payload)


class _FakeWorksheet:
    id = 123

    def __init__(self):
        self.values = []
        self.formats = []
        self.cleared_ranges = []
        self.spreadsheet = _FakeSpreadsheet()

    def clear(self):
        self.values = []

    def append_row(self, row, value_input_option=None):
        self.values.append(list(row))

    def append_rows(self, rows, value_input_option=None):
        self.values.extend([list(row) for row in rows])

    def get_all_values(self):
        return [list(row) for row in self.values]

    def batch_clear(self, ranges):
        self.cleared_ranges.extend(ranges)

    def update(self, *, values, range_name, value_input_option=None):
        start_row = int(range_name.split(":", 1)[0][1:])
        while len(self.values) < start_row - 1:
            self.values.append([])
        for offset, row in enumerate(values):
            index = start_row - 1 + offset
            if index < len(self.values):
                self.values[index] = list(row)
            else:
                self.values.append(list(row))

    def batch_format(self, formats):
        self.formats.extend(formats)


def test_sheet_snapshot_aligns_final_rows_and_recreates_yardage(monkeypatch):
    ws = _FakeWorksheet()
    monkeypatch.setattr(
        sheets_storage, "_get_or_create_tab", lambda spreadsheet, tab, headers: ws
    )
    detail = pd.DataFrame([{
        "year": 2025,
        "round_num": 1,
        "raw_avg": 69.1,
        "field_strength": 0.2,
        "field_adj": 0.2,
        "cut_applied": False,
        "wind_avg_mph": 8.0,
        "wind_adj": 0.1,
        "dew_delta": 0.0,
        "baseline": 69.0,
    }])
    yardage = [{
        "year": 2026,
        "event_name": "TOUR Championship",
        "course_id": 688,
        "course_name": "East Lake Golf Club",
        "par": 70,
        "r1_yards": 7270,
        "r2_yards": 7250,
        "r3_yards": 7260,
        "r4_yards": 7369,
        "event_key": "pga:R2026060",
    }]

    sb.save_to_sheets(
        detail,
        {1: 68.1, 2: 68.2, 3: 68.3, 4: 68.4},
        {2025: 1.0},
        yardage_rows=yardage,
        yardage_warnings=[],
        spreadsheet=object(),
    )

    final_rows = [row for row in ws.values if row and row[0] == "FINAL"]
    assert all(len(row) == len(sb.SCORING_BASELINE_HEADERS) for row in final_rows)
    assert final_rows[0][9:11] == [68.1, ""]
    title_rows = [
        index for index, row in enumerate(ws.values, start=1)
        if row and str(row[0]).startswith(sb.YARDAGE_HISTORY_TITLE)
    ]
    assert len(title_rows) == 1

    # A second full snapshot clears and recreates the owned section once.
    sb.save_to_sheets(
        detail,
        {1: 68.1, 2: 68.2, 3: 68.3, 4: 68.4},
        {2025: 1.0},
        yardage_rows=yardage,
        yardage_warnings=[],
        spreadsheet=object(),
    )
    title_rows = [
        index for index, row in enumerate(ws.values, start=1)
        if row and str(row[0]).startswith(sb.YARDAGE_HISTORY_TITLE)
    ]
    assert len(title_rows) == 1
