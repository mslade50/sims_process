from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from archive import hole_sim_production_anchor_inputs as target
from archive import hole_sim_transition_mean_chain_adapter as adapter


def _make_db(path: Path) -> None:
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE player_rounds (dg_id INTEGER, round_date TEXT, tour TEXT, "
            "sg_total_adj REAL, sg_ott_adj REAL, sg_app_adj REAL, "
            "sg_arg_adj REAL, sg_putt_adj REAL)"
        )
        for dg_id, base in ((1, 0.0), (2, 0.4)):
            for day in range(1, 14):
                db.execute(
                    "INSERT INTO player_rounds VALUES (?,?,?,?,?,?,?,?)",
                    (
                        dg_id,
                        f"2026-06-{day:02d}",
                        "pga",
                        1.0,
                        base + 0.1,
                        base + 0.2,
                        base + 0.3,
                        base + 0.4,
                    ),
                )
            # Must be excluded by strict round_date < event_start.
            db.execute(
                "INSERT INTO player_rounds VALUES (?,?,?,?,?,?,?,?)",
                (dg_id, "2026-06-25", "pga", 1.0, 99.0, 99.0, 99.0, 99.0),
            )
        db.execute(
            "CREATE TABLE model_results2 (Date TEXT, dg_id INTEGER, player_name TEXT, "
            "skill_est REAL, sg_ott REAL, sg_app REAL, ag_arg REAL, ag_putt REAL)"
        )
        for dg_id, name, total in ((1, "One, Player", 1.0), (2, "Two, Player", 2.0)):
            db.execute(
                "INSERT INTO model_results2 VALUES (?,?,?,?,?,?,?,?)",
                ("2026-06-24", dg_id, name, total, 0.1, 0.2, 0.3, 0.4),
            )
            # Equal-date snapshots are deliberately inadmissible for quality beta.
            db.execute(
                "INSERT INTO model_results2 VALUES (?,?,?,?,?,?,?,?)",
                ("2026-06-25", dg_id, name, 9.0, 9.0, 9.0, 9.0, 9.0),
            )
        db.commit()


def _make_baseline(path: Path) -> None:
    rows = []
    # Player IDs intentionally sort opposite to names. This catches attaching
    # hashes positionally after the adapter returns canonical key order.
    for dg_id, player_id, name, total in (
        (1, "22", "one, player", 1.0),
        (2, "11", "two, player", 2.0),
    ):
        for round_no in range(1, 5):
            for arm in ("category_current", "category_shot"):
                rows.append(
                    {
                        "event_key": "2026-06-25|34|Travelers Championship",
                        "event": "Travelers Championship",
                        "event_start": pd.Timestamp("2026-06-25"),
                        "dg_id": dg_id,
                        "source_player_id": player_id,
                        "player_name": name,
                        "round_num": round_no,
                        "arm": arm,
                        "target_sg": total,
                        "skill_date": pd.Timestamp("2026-06-25"),
                        # Present in the source, but the exporter never requests it.
                        "actual_strokes": 55.0,
                        "field_scoring_center": 66.0,
                    }
                )
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_ema_mean_matches_direct_weights() -> None:
    values = np.array([-1.0, 0.5, 2.0, 20.0])
    clipped = np.clip(values, -8.0, 8.0)
    alpha = 2.0 / 51.0
    weights = (1.0 - alpha) ** np.arange(3, -1, -1, dtype=float)
    weights /= weights.sum()
    assert target.ema_mean(values) == float(weights @ clipped)


def test_real_contract_builder_is_strict_and_adapter_compatible(tmp_path: Path) -> None:
    db_path = tmp_path / "history.db"
    baseline_path = tmp_path / "baseline.parquet"
    _make_db(db_path)
    _make_baseline(baseline_path)
    frame, diagnostics = target.build_production_anchor_inputs(
        baseline_path=baseline_path,
        db_path=db_path,
        event_name="Travelers Championship",
        event_start="2026-06-25",
        category_source_path=Path(target.__file__),
    )
    assert len(frame) == 8
    assert diagnostics["ability_cutoff_violations"] == 0
    assert diagnostics["heldout_score_columns_read"] == []
    assert not diagnostics["heldout_field_scoring_center_used"]
    assert set(frame["event_key"]) == {"pga:R2026034"}
    assert set(frame["ability_skill_date"]) == {pd.Timestamp("2026-06-24")}
    assert np.allclose(
        frame[[f"target_{category}_sg" for category in target.CATEGORIES]].sum(axis=1),
        frame["target_total_sg"],
        atol=1e-12,
        rtol=0.0,
    )
    # Equal-date 9.0 rows were excluded; the prior vector is retained separately.
    assert np.allclose(frame["ability_ott_sg"], 0.1)
    assert np.allclose(frame["ability_putt_sg"], 0.4)
    validated = adapter.validate_production_target_frame(
        frame[list(adapter.PRODUCTION_TARGET_REQUIRED_COLUMNS)]
    )
    assert len(validated) == len(frame)
    assert frame["production_anchor_input_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()


def test_category_target_mutation_changes_rich_digest(tmp_path: Path) -> None:
    db_path = tmp_path / "history.db"
    baseline_path = tmp_path / "baseline.parquet"
    _make_db(db_path)
    _make_baseline(baseline_path)
    frame, _ = target.build_production_anchor_inputs(
        baseline_path=baseline_path,
        db_path=db_path,
        event_name="Travelers Championship",
        category_source_path=Path(target.__file__),
    )
    row = frame.iloc[0].to_dict()
    original = target._row_digest(row)
    row["target_app_sg"] = float(row["target_app_sg"]) + 0.01
    assert target._row_digest(row) != original
