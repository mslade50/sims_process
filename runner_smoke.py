"""Read-only validation for the Windows self-hosted golf simulation runner."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent


def _require_environment(names):
    missing = [name for name in names if not os.getenv(name, "").strip()]
    if missing:
        raise RuntimeError(
            "Required GitHub Actions secrets are unavailable: "
            + ", ".join(missing)
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run read-only dependency, Sheet, API, and weather checks"
    )
    parser.add_argument(
        "--odds-file",
        default=str(
            ROOT
            / "permanent_data"
            / "scraped_odds"
            / "round_matchups_latest.json"
        ),
    )
    args = parser.parse_args()
    load_dotenv()

    _require_environment(["DATAGOLF_API_KEY", "GOOGLE_CREDS_JSON"])

    import sims_kernel

    kernel_check = sims_kernel.selftest()
    if kernel_check is False:
        raise RuntimeError("sims_kernel self-test returned False")
    print(f"  Rust kernel ready: {sims_kernel.version()}")

    from sheet_config import load_config

    config = load_config(verbose=False)
    required_config = ("tourney", "event_id", "course_id")
    missing_config = [key for key in required_config if not config.get(key)]
    if missing_config:
        raise RuntimeError(
            "Sheet round_config is missing: " + ", ".join(missing_config)
        )
    latitude = config.get("course_latitude")
    longitude = config.get("course_longitude")
    if latitude is None or longitude is None:
        raise RuntimeError("Sheet round_config is missing a valid course_lat_lon")
    print(
        f"  Sheet ready: {config['tourney']} event={config['event_id']} "
        f"round={config['round_num']} coords={latitude},{longitude}"
    )

    from api_utils import (
        fetch_event_weather_forecast,
        fetch_field_updates,
        get_round_dates,
    )

    field = fetch_field_updates(
        os.environ["DATAGOLF_API_KEY"],
        teetime_col="r1_teetime",
        include_course=True,
        fill_missing_teetimes=False,
    )
    if field is None or field.empty:
        raise RuntimeError("DataGolf field-updates returned no players")
    print(f"  DataGolf ready: {len(field)} field rows")

    weather = fetch_event_weather_forecast(
        latitude,
        longitude,
        get_round_dates(),
        timezone="auto",
    )
    if not weather:
        raise RuntimeError("Open-Meteo returned no event forecast")
    for rnd in range(1, 5):
        round_weather = weather["rounds"].get(rnd) or {}
        if len(round_weather.get("wind") or []) != 15:
            raise RuntimeError(f"Open-Meteo R{rnd} wind array is incomplete")
        if len(round_weather.get("dew") or []) != 15:
            raise RuntimeError(f"Open-Meteo R{rnd} dew array is incomplete")
    print(
        f"  Open-Meteo ready: timezone={weather['timezone']} "
        f"AI hours={weather['ai_hours']}"
    )

    odds_path = Path(args.odds_file).resolve()
    if not odds_path.exists():
        raise RuntimeError(f"Odds smoke file is missing: {odds_path}")
    with odds_path.open(encoding="utf-8") as handle:
        odds = json.load(handle)
    if not isinstance(odds, dict) or not isinstance(odds.get("match_list"), list):
        raise RuntimeError("Odds smoke file has an invalid matchup payload")
    print(
        f"  Odds access ready: R{odds.get('round')} "
        f"rows={len(odds['match_list'])}"
    )

    print("\n  LOCAL RUNNER SMOKE PASSED (read-only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
