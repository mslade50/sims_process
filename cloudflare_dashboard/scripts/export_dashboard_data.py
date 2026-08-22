"""Build browser-ready dashboard snapshots from the existing Python data layer.

The simulation pipeline remains the source of truth. This exporter converts its
CSV/Parquet/Google Sheets inputs into small JSON contracts that the Cloudflare
dashboard can serve from static assets or R2.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SITE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SITE_ROOT.parent
DEFAULT_OUTPUT = SITE_ROOT / "public" / "data"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.data_layer import (  # noqa: E402
    get_bet_ledger,
    get_h2h_matrix,
    get_historical_events,
    get_historical_rank_probs,
    get_mkt_regress_diagnostics,
    get_model_predictions,
    get_rank_probs_live,
    get_rank_probs_pre,
    get_round_score_probs,
    get_sg_diagnostics,
    get_sg_dist_player,
    get_tournament_config,
    get_v2_dists,
    get_weather_forecast,
    get_weather_impact_players,
    get_weather_matchup_edges,
)


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if pd.isna(value):
        return None
    return value


def _records(frame: pd.DataFrame, columns: list[str] | None = None) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    clean = frame.copy()
    if columns:
        clean = clean[[column for column in columns if column in clean.columns]]
    clean = clean.replace([np.inf, -np.inf], np.nan)
    return [
        {str(key): _json_value(value) for key, value in row.items()}
        for row in clean.to_dict(orient="records")
    ]


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(payload), ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    try:
        display_path = path.relative_to(SITE_ROOT)
    except ValueError:
        display_path = path
    print(f"  + {display_path} ({path.stat().st_size / 1024:.1f} KB)")


def _diagnostic_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    adjusted = frame.copy()
    if "sg_type" in adjusted.columns:
        adjusted = adjusted[adjusted["sg_type"].astype(str).str.lower() == "adjusted"]
    keys = [
        column
        for column in ["event_id", "event_name", "year", "player_name", "archetype", "category"]
        if column in adjusted.columns
    ]
    metrics = {
        column: "mean"
        for column in ["predicted_sg", "actual_sg", "miss", "miss_centered"]
        if column in adjusted.columns
    }
    if not keys or not metrics:
        return _records(adjusted)
    summary = adjusted.groupby(keys, dropna=False).agg(metrics).reset_index()
    counts = adjusted.groupby(keys, dropna=False).size().rename("rounds").reset_index()
    summary = summary.merge(counts, on=keys, how="left")
    return _records(summary)


def export(output: Path) -> None:
    generated_at = datetime.now(timezone.utc).isoformat()
    config = get_tournament_config()
    tourney = str(config.get("tourney", "")).strip().lower()

    prediction_frames = []
    for round_num in range(1, 5):
        frame = get_model_predictions(round_num)
        if not frame.empty:
            tagged = frame.copy()
            tagged["round"] = round_num
            prediction_frames.append(tagged)

    _write(
        output / "distributions.json",
        {
            "pre": _records(get_rank_probs_pre(tourney)),
            "live": _records(get_rank_probs_live(tourney)),
            "h2h": _records(get_h2h_matrix(tourney)),
        },
    )

    adjusted = get_v2_dists()
    raw = get_sg_dist_player()
    if not adjusted.empty and "player_name" in adjusted.columns and not raw.empty:
        field = set(adjusted["player_name"].astype(str).str.lower().str.strip())
        raw = raw[raw["player_name"].astype(str).str.lower().str.strip().isin(field)]
    _write(
        output / "sg-distributions.json",
        {
            "raw": _records(raw),
            "adjusted": _records(adjusted),
            "predictions": _records(pd.concat(prediction_frames, ignore_index=True))
            if prediction_frames
            else [],
        },
    )

    rounds: dict[str, list[dict[str, Any]]] = {}
    for round_num in range(1, 5):
        frame = get_round_score_probs(round_num)
        if not frame.empty:
            rounds[str(round_num)] = _records(frame)
    _write(output / "round-scores.json", {"rounds": rounds})

    history_manifest = []
    for event_id, event_name in get_historical_events():
        event_key = f"{event_id}-{event_name}"
        modes = []
        for mode in ("pre", "live"):
            frame = get_historical_rank_probs(event_id, event_name, mode)
            if frame.empty:
                continue
            _write(output / "history" / f"{event_key}-{mode}.json", {"rows": _records(frame)})
            modes.append(mode)
        if modes:
            history_manifest.append(
                {
                    "event_id": event_id,
                    "event_name": event_name,
                    "key": event_key,
                    "modes": modes,
                }
            )
    _write(output / "history.json", {"events": history_manifest})

    ledger = get_bet_ledger()
    _write(output / "performance.json", {"bets": _records(ledger)})

    diagnostics = get_sg_diagnostics()
    market_regress = get_mkt_regress_diagnostics()
    _write(
        output / "diagnostics.json",
        {
            "sg": _diagnostic_summary(diagnostics),
            "market_regression": _records(market_regress),
        },
    )

    _write(
        output / "weather.json",
        {
            "forecast": _json_value(get_weather_forecast()),
            "players": _records(get_weather_impact_players()),
            "matchups": _records(get_weather_matchup_edges()),
        },
    )

    _write(
        output / "manifest.json",
        {
            "schema_version": 1,
            "generated_at": generated_at,
            "event": tourney,
            "event_id": (config.get("event_ids") or [None])[0]
            if isinstance(config.get("event_ids"), (list, tuple))
            else config.get("event_ids"),
            "course": config.get("course_name") or tourney,
            "course_id": config.get("course_id"),
            "par": config.get("course_par") or config.get("PAR"),
            "rounds": sorted(int(key) for key in rounds),
            "views": [
                "distributions",
                "sg-distributions",
                "round-scores",
                "history",
                "performance",
                "diagnostics",
                "weather",
            ],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Cloudflare dashboard JSON snapshots")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    export(args.output.resolve())


if __name__ == "__main__":
    main()
