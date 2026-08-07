"""Generate the live R1 pin-high adjustment inside a clean sim checkout."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACT = ROOT / "sim_fairs.json"
DEFAULT_PREDS = ROOT / "dashboard_data" / "model_predictions_r1.csv"
DEFAULT_OUTPUT = ROOT / "pin_high_r1.csv"


def build_event_key(tour_name, event_id, season):
    """Return the rich-shot archive key for a PGA event."""
    normalized_tour = str(tour_name).strip().lower()
    if normalized_tour != "pga":
        raise ValueError("pin-high live adjustment currently supports PGA events only")
    return f"pga:R{int(season)}{int(event_id):03d}"


def validate_matching_predictions(
    artifact_path,
    predictions_path,
    *,
    expected_event_ids,
    expected_tourney,
):
    """Validate that the persisted full R1 artifact matches the active sim field."""
    artifact = Path(artifact_path)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    expected_ids = {str(event_id).strip() for event_id in expected_event_ids}
    artifact_event = str(payload.get("event_id", "")).strip()
    artifact_tourney = str(payload.get("tourney", "")).strip().lower()
    if artifact_event not in expected_ids:
        raise ValueError(
            f"sim artifact event {artifact_event!r} does not match {sorted(expected_ids)}"
        )
    if artifact_tourney != str(expected_tourney).strip().lower():
        raise ValueError(
            f"sim artifact tournament {artifact_tourney!r} does not match "
            f"{expected_tourney!r}"
        )

    import pandas as pd

    predictions = pd.read_csv(predictions_path)
    pred_column = next(
        (column for column in ("pred", "my_pred", "my_pred2")
         if column in predictions.columns),
        None,
    )
    required = ["player_name", "wind_adj1", "dew_adj1"]
    missing_columns = [column for column in required if column not in predictions]
    if pred_column is None:
        missing_columns.append("prediction")
    if missing_columns:
        raise ValueError(f"R1 prediction artifact is incomplete: {missing_columns}")
    check_columns = ["player_name", pred_column, "wind_adj1", "dew_adj1"]
    if predictions[check_columns].isna().any().any():
        raise ValueError("R1 prediction artifact contains missing prediction/weather values")

    players = {
        str(player).strip().lower()
        for player in predictions["player_name"]
        if str(player).strip()
    }
    field = {
        str(player).strip().lower()
        for player in payload.get("field", [])
        if str(player).strip()
    }
    if len(players) < 10 or len(field) < 10:
        raise ValueError(
            f"R1 prediction/field artifact is too small ({len(players)}/{len(field)})"
        )
    overlap = len(players & field) / len(field)
    if overlap < 0.95:
        raise ValueError(
            f"R1 prediction artifact overlaps only {overlap:.1%} of the active field"
        )
    return len(players)


def generate(args):
    import sim_inputs

    event_ids = sim_inputs.event_ids
    tour = sim_inputs.tour
    tourney = sim_inputs.tourney
    if len(event_ids) != 1:
        raise ValueError(f"pin-high generation requires one active event: {event_ids}")
    collector_root = Path(args.collector_root).resolve()
    database = Path(args.db).resolve()
    if not (collector_root / "shot_collector" / "pin_high_correction.py").is_file():
        raise FileNotFoundError(f"shot collector is unavailable: {collector_root}")
    if not database.is_file():
        raise FileNotFoundError(f"shot database is unavailable: {database}")

    prediction_count = validate_matching_predictions(
        args.artifact,
        args.preds_source,
        expected_event_ids=event_ids,
        expected_tourney=tourney,
    )
    event_key = build_event_key(tour, event_ids[0], args.season)
    command = [
        sys.executable,
        "-m",
        "shot_collector.pin_high_correction",
        "--db",
        str(database),
        "--event-key",
        event_key,
        "--preds-csv",
        str(Path(args.preds_source).resolve()),
        "--output",
        str(Path(args.output).resolve()),
    ]
    subprocess.run(command, cwd=collector_root, check=True)
    print(
        f"Generated {Path(args.output).name} for {event_key} "
        f"from {prediction_count} persisted R1 predictions"
    )


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    root.add_argument("--preds-source", type=Path, default=DEFAULT_PREDS)
    root.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    root.add_argument(
        "--collector-root",
        default=os.getenv(
            "IMG_SHOT_COLLECTOR_ROOT",
            r"C:\Users\McKinley Slade\dev\golf_scraping-collector-prod",
        ),
    )
    root.add_argument(
        "--db",
        default=os.getenv(
            "IMG_SHOT_DB_PATH",
            r"C:\Users\McKinley Slade\dev\golf_scraping\output\imgarena_shots_lean.sqlite3",
        ),
    )
    root.add_argument("--season", type=int, default=datetime.now().year)
    return root


def main():
    generate(parser().parse_args())


if __name__ == "__main__":
    main()
