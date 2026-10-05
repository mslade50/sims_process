"""Event-bound scoring receipts. Actuals never depend on weather availability.

The Scoring Feedback Sheet tab retains immutable pre-start forecasts and
content-addressed actual/ingestion receipts. This module does not send email.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import re
from pathlib import Path
from zoneinfo import ZoneInfo

from forecast_feedback import forecast_feedback, single_published_forecast

TAB = "Scoring Feedback"
LEDGER_HEADERS = ["Receipt ID", "Event ID", "Year", "Course ID", "Round", "Stage", "Status", "Receipt JSON"]
ACTUAL_HEADERS = [
    "Round", "Realized Wind", "Forecast Dew", "Realized Dew", "Wind Impact",
    "Dew Impact", "Published Forecast", "Actual", "Forecast Miss",
    "Structural Wx Baseline", "Structural Residual", "Event ID", "Year",
    "Course ID", "Actual Receipt", "Feedback Status", "Forecast Receipt",
]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def stamp(value):
    value = str(value).replace(" UTC", "+00:00").replace("Z", "+00:00")
    # GitHub logs use seven fractional digits; Python 3.10 accepts six.
    value = re.sub(r"(\.\d{6})\d+(?=[+-])", r"\1", value)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("Forecast timestamp requires a timezone")
    return parsed.astimezone(timezone.utc)


def context(config, round_num, year=None):
    return {"event_id": str(config["event_id"]), "year": int(year or datetime.now(timezone.utc).year),
            "course_id": int(config["course_id"]), "tourney": str(config["tourney"]),
            "round": int(round_num)}


def receipt(ctx, stage, status, **details):
    data = {**ctx, "stage": stage, "status": status, **details}
    return {**data, "receipt_id": digest(data)}


def same_event(left, right):
    return all(str(left.get(k)) == str(right.get(k))
               for k in ("event_id", "year", "course_id", "tourney"))


def read_receipts(spreadsheet):
    import gspread
    try:
        rows = spreadsheet.worksheet(TAB).get_all_values()
    except gspread.WorksheetNotFound:
        return []
    if not rows or rows[0] != LEDGER_HEADERS:
        raise ValueError("Scoring Feedback ledger headers are invalid")
    records = []
    for row in rows[1:]:
        data = json.loads(row[7])
        if data.get("receipt_id") != row[0] or digest({k: v for k, v in data.items() if k != "receipt_id"}) != row[0]:
            raise ValueError("Scoring Feedback receipt checksum mismatch")
        records.append(data)
    return records


def save_receipt(spreadsheet, record):
    import gspread
    records = read_receipts(spreadsheet)
    if any(r["receipt_id"] == record["receipt_id"] for r in records):
        return record
    try:
        ws = spreadsheet.worksheet(TAB)
    except gspread.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=TAB, rows=1000, cols=8)
        ws.append_row(LEDGER_HEADERS, value_input_option="RAW")
    ws.append_row([record["receipt_id"], record["event_id"], record["year"],
                   record["course_id"], record["round"], record["stage"], record["status"],
                   json.dumps(record, sort_keys=True, allow_nan=False)], value_input_option="RAW")
    if not any(r["receipt_id"] == record["receipt_id"] for r in read_receipts(spreadsheet)):
        raise RuntimeError("Scoring receipt readback failed")
    print(f"  [feedback] {record['stage']} R{record['round']}: {record['status']} ({record['receipt_id'][:12]})")
    return record


def forecast_record(ctx, *, value, players, published_at, round_start_at, source,
                    allowed_missing=(), allowed_extra=()):
    value = single_published_forecast(value)
    if value is None or not players or len(set(players)) != len(players):
        raise ValueError("Forecast requires a finite score and a unique player cohort")
    published, started = stamp(published_at), stamp(round_start_at)
    if published >= started:
        raise ValueError("Hindsight forecast: publication is not before first tee")
    if started.year != ctx["year"] or published.year != ctx["year"]:
        raise ValueError("Forecast season mismatch")
    return receipt(ctx, "forecast", "applied", value=value, players=sorted(players),
                   published_at=published.isoformat(), round_start_at=started.isoformat(),
                   source=source, allowed_missing=sorted(allowed_missing), allowed_extra=sorted(allowed_extra))


def capture_forecast(spreadsheet, config, round_num, value, predictions, *, published_at=None, source=None):
    """Retain a publication only when its first tee and event are known."""
    ctx = context(config, round_num)
    try:
        zone = config.get("course_timezone")
        if not zone:
            raise ValueError("Missing course timezone")
        times = predictions[f"r{round_num}_teetime"].dropna().tolist()
        if not times or len(times) != len(predictions):
            raise ValueError("Incomplete dated tee times")
        parsed = [datetime.fromisoformat(str(t)) for t in times]
        first = min(t.replace(tzinfo=ZoneInfo(zone)) if t.tzinfo is None else t for t in parsed)
        names = predictions["player_name"].str.lower().str.strip().tolist()
        record = forecast_record(ctx, value=value, players=names,
                                 published_at=published_at or datetime.now(timezone.utc).isoformat(),
                                 round_start_at=first.isoformat(), source=source or {"kind": "sheet_publication"})
    except (KeyError, TypeError, ValueError) as exc:
        status = "not_due" if "Hindsight" in str(exc) else "missing"
        record = receipt(ctx, "forecast", status, reason=str(exc))
    return save_receipt(spreadsheet, record)


def completed_scores(ctx, frame, par):
    """Require a finished round, including valid scores of later WD/CUT players."""
    if not same_event(frame.attrs, ctx) or int(frame.attrs.get("round_num", 0)) != ctx["round"]:
        # The API provides the event name, while the context's tournament slug
        # is attached by the caller after its separate Sheet identity check.
        raise ValueError("Stale-event or wrong-round scores")
    if frame.empty:
        return "missing", {}, []
    if frame["player_name"].duplicated().any() or frame["player_name"].isna().any():
        raise ValueError("Duplicate or missing official player names")
    scores, inactive = {}, []
    for row in frame.to_dict("records"):
        name = row["player_name"]
        terminal = str(row.get("position", "")).strip().upper() in {"CUT", "WD", "DQ", "W/D"}
        try:
            thru, score = float(row["thru"]), float(row["round"])
        except (KeyError, TypeError, ValueError):
            thru, score = math.nan, math.nan
        if thru == 18 and math.isfinite(score) and score == int(score):
            scores[name] = float(par) + score
        elif terminal and (not math.isfinite(thru) or thru < 18):
            inactive.append(name)
        elif thru == 18:
            raise ValueError("Completed player has no valid official score")
        else:
            return "not_due", {}, []
    return ("applied" if scores else "missing"), dict(sorted(scores.items())), sorted(inactive)


def select_forecast(records, ctx):
    candidates = [r for r in records if same_event(r, ctx) and r["round"] == ctx["round"]
                  and r["stage"] == "forecast" and r["status"] == "applied"]
    return max(candidates, key=lambda r: stamp(r["published_at"])) if candidates else None


def record_actuals(spreadsheet, config, round_num, *, year=None, frame=None,
                   forecast=None, weather_loader=None):
    """Record official scores first; weather failures affect diagnostics only."""
    ctx = context(config, round_num, year)
    try:
        if frame is None:
            from api_utils import fetch_scoring_round
            import os
            frame = fetch_scoring_round(ctx["event_id"], ctx["year"], round_num,
                                        ctx["course_id"], os.getenv("DATAGOLF_API_KEY"))
        frame = frame.copy()
        # fetch_scoring_round validates provider identity against the schedule.
        frame.attrs["tourney"] = config["tourney"]
        status, scores, inactive = completed_scores(ctx, frame, config["course_par"])
        if status != "applied":
            return save_receipt(spreadsheet, receipt(ctx, "actuals", status, reason="Official round is incomplete or unavailable"))
        feedback_status, coverage = "missing", {}
        try:
            supplied_forecast = forecast is not None
            if forecast is not None:
                if not same_event(forecast, ctx) or forecast["round"] != round_num:
                    raise ValueError("Backfill forecast identity mismatch")
                forecast = forecast_record(ctx, **{k: forecast[k] for k in (
                    "value", "players", "published_at", "round_start_at", "source", "allowed_missing", "allowed_extra")})
            else:
                forecast = select_forecast(read_receipts(spreadsheet), ctx)
            if forecast:
                round_date = datetime.fromisoformat(frame.attrs["start_date"]).date() + timedelta(days=round_num - 1)
                forecast_date = stamp(forecast["round_start_at"]).astimezone(ZoneInfo(config["course_timezone"])).date()
                if forecast_date != round_date:
                    raise ValueError("Forecast tee date does not match the official round")
                if supplied_forecast:
                    save_receipt(spreadsheet, forecast)
        except (KeyError, TypeError, ValueError) as exc:
            save_receipt(spreadsheet, receipt(ctx, "forecast", "failed", reason=str(exc)))
            forecast, feedback_status = None, "failed"
        if forecast:
            missing = sorted(set(forecast["players"]) - scores.keys())
            extra = sorted(scores.keys() - set(forecast["players"]))
            coverage = {"missing": missing, "extra": extra, "inactive": inactive,
                        "allowed_missing": forecast["allowed_missing"], "allowed_extra": forecast["allowed_extra"]}
            permitted = (set(missing) <= set(inactive) | set(forecast["allowed_missing"])
                         and set(extra) <= set(forecast["allowed_extra"]))
            feedback_status = "applied" if permitted else "failed"
        actual = sum(scores.values()) / len(scores)
        published = forecast["value"] if forecast else None
        # Match the established two-decimal Sheet feedback precision.
        miss = round(actual - published, 2) if feedback_status == "applied" else None
        record = receipt(ctx, "actuals", "applied", actual=actual, completed_players=len(scores),
                         scores=scores, forecast=published, forecast_receipt=forecast["receipt_id"] if forecast else "",
                         feedback_status=feedback_status, forecast_miss=miss, coverage=coverage)
        ws = spreadsheet.worksheet("round_config")
        values = [f"R{round_num}", "", "", "", "", "", published if published is not None else "",
                  round(actual, 2), miss if miss is not None else "", "", "", ctx["event_id"],
                  ctx["year"], ctx["course_id"], record["receipt_id"], feedback_status, record["forecast_receipt"]]
        ws.update(range_name="U10:AK10", values=[ACTUAL_HEADERS], value_input_option="RAW")
        ws.update(range_name=f"U{10 + round_num}:AK{10 + round_num}", values=[values], value_input_option="RAW")
        if ws.get(f"AI{10 + round_num}")[0][0] != record["receipt_id"]:
            raise RuntimeError("Actuals Sheet readback failed")
        save_receipt(spreadsheet, record)
    except Exception as exc:
        # No exception strings from HTTP/auth failures (they can contain keys).
        failed = receipt(ctx, "actuals", "failed", reason=type(exc).__name__)
        save_receipt(spreadsheet, failed)
        print(f"  [actuals] R{round_num} failed: {type(exc).__name__}")
        return failed
    # A weather outage cannot prevent recording scores or supplying feedback.
    try:
        diagnostics = weather_loader(frame.attrs) if weather_loader else {}
        fields = [diagnostics.get(k, "") for k in ("wind", "forecast_dew", "dew", "wind_impact", "dew_impact")]
        baseline = diagnostics.get("structural_baseline")
        if baseline is not None:
            ws.update(range_name=f"AD{10 + round_num}:AE{10 + round_num}",
                      values=[[round(baseline, 2), round(actual - baseline, 2)]], value_input_option="RAW")
        if diagnostics:
            ws.update(range_name=f"V{10 + round_num}:Z{10 + round_num}", values=[fields], value_input_option="RAW")
        weather_status = "applied" if baseline is not None else "missing"
        save_receipt(spreadsheet, receipt(ctx, "weather", weather_status, diagnostics=diagnostics))
    except Exception as exc:
        save_receipt(spreadsheet, receipt(ctx, "weather", "failed", reason=type(exc).__name__))
    return record


def ingest_feedback(spreadsheet, config, completed_round, *, year=None):
    """Use only verified current-event receipts that agree with the Sheet rows."""
    ctx = context(config, completed_round, year)
    if completed_round not in (1, 2, 3):
        return save_receipt(spreadsheet, receipt(ctx, "ingestion", "not_due", correction=0.0,
                                               reason="No later round to adjust"))
    try:
        records = {r["receipt_id"]: r for r in read_receipts(spreadsheet)}
        grid = spreadsheet.worksheet("round_config").get("U10:AK14")
        usable, sources, missing, failed = [], [], [], []
        for rnd in range(1, completed_round + 1):
            row = grid[rnd] if len(grid) > rnd else []
            if len(row) <= 7 or not row[7]:
                missing.append(rnd)
                continue
            try:
                data = records[row[14]]
                if (grid[0] != ACTUAL_HEADERS or not same_event(data, ctx)
                        or data["round"] != rnd or data["stage"] != "actuals"
                        or data["status"] != "applied"
                        or row[11:14] != [str(ctx["event_id"]), str(ctx["year"]), str(ctx["course_id"])]
                        or row[15] != data["feedback_status"]
                        or float(row[7]) != round(data["actual"], 2)):
                    raise ValueError("Unverified or stale actuals")
                if data["feedback_status"] == "missing":
                    missing.append(rnd)
                    continue
                fc = records[data["forecast_receipt"]]
                if (data["feedback_status"] != "applied" or row[16] != fc["receipt_id"]
                        or not same_event(fc, ctx) or fc["round"] != rnd
                        or fc["stage"] != "forecast" or fc["status"] != "applied"
                        or stamp(fc["published_at"]) >= stamp(fc["round_start_at"])):
                    raise ValueError("Unverified or stale actuals")
                if data["forecast_miss"] is None:
                    missing.append(rnd)
                    continue
                value = float(row[8])
                if (not math.isfinite(value) or abs(value) > 5 or value != data["forecast_miss"]
                        or float(row[6]) != fc["value"] or float(row[7]) != round(data["actual"], 2)):
                    raise ValueError("Actuals readback disagrees with its receipt")
                usable.append(value)
                sources.append(data["receipt_id"])
            except (IndexError, KeyError, TypeError, ValueError):
                failed.append(rnd)
        correction, weight = forecast_feedback(usable)
        status = "failed" if failed else ("applied" if usable else "missing")
        if failed:
            correction = 0.0
        record = receipt(ctx, "ingestion", status, correction=correction, weight=weight,
                         misses=usable, source_receipts=sources, missing_rounds=missing, failed_rounds=failed)
    except Exception as exc:
        record = receipt(ctx, "ingestion", "failed", correction=0.0, reason=type(exc).__name__)
    return save_receipt(spreadsheet, record)


def realized_diagnostics(config, round_num, metadata, spreadsheet):
    """Optional observed diagnostics; never substitute refreshed forecasts for observations."""
    from api_utils import fetch_realized_wind, compute_wind_factor
    from sim_inputs import event_ids, baseline_wind
    wind = config.get(f"realized_wind_r{round_num}")
    if wind is None and metadata.get("latitude") is not None and metadata.get("longitude") is not None:
        date = datetime.fromisoformat(metadata["start_date"]) + timedelta(days=round_num - 1)
        wind = fetch_realized_wind(metadata["latitude"], metadata["longitude"], date.date().isoformat())
    dew = config.get(f"realized_dew_r{round_num}")
    result = {}
    if wind is not None and math.isfinite(float(wind)):
        result["wind"] = float(wind)
        result["wind_impact"] = float(wind) * compute_wind_factor(
            event_ids, config.get("wind_override") or 0, baseline_wind, config["course_id"])
    if dew is not None and math.isfinite(float(dew)):
        result["dew"] = float(dew)
        result["dew_impact"] = (float(dew) - float(config.get("dewpoint_base") or 0)) * float(config.get("dew_calculation") or 0)
    if "wind_impact" in result and "dew_impact" in result:
        baseline = spreadsheet.worksheet("round_config").get(f"V{2 + round_num}:W{2 + round_num}")[0]
        result["structural_baseline"] = sum(map(float, baseline)) + result["wind_impact"] + result["dew_impact"]
    return result


def main():
    import argparse
    from dotenv import load_dotenv
    from sheets_storage import get_spreadsheet, get_round_config_params
    from sheet_config import load_config
    parser = argparse.ArgumentParser(description="No-email official scoring backfill and feedback verification")
    parser.add_argument("--plan", required=True, help="JSON containing year/event/course/tourney and proven forecasts")
    parser.add_argument("--evidence-dir", required=True)
    args = parser.parse_args()
    load_dotenv(Path(__file__).with_name(".env"))
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8-sig"))
    ss = get_spreadsheet()
    config = load_config(verbose=False)
    raw = get_round_config_params(spreadsheet=ss)
    for rnd in range(1, 5):
        if raw.get(f"realized_dew_r{rnd}"):
            config[f"realized_dew_r{rnd}"] = float(raw[f"realized_dew_r{rnd}"])
    if any(str(config.get(k)) != str(plan[k]) for k in ("event_id", "course_id", "tourney")):
        raise ValueError("Backfill plan does not match the live Sheet")
    config["course_par"] = plan["course_par"]
    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "round_config_before.json"
    if not snapshot.exists():
        snapshot.write_text(json.dumps(ss.worksheet("round_config").get_all_values()), encoding="utf-8")
    actuals = []
    for forecast in plan["forecasts"]:
        actuals.append(record_actuals(ss, config, forecast["round"], year=plan["year"], forecast=forecast,
                      weather_loader=lambda metadata, rnd=forecast["round"]: realized_diagnostics(config, rnd, metadata, ss)))
    ingestion = [ingest_feedback(ss, config, rnd, year=plan["year"]) for rnd in range(1, 5)]
    evidence = {"actuals": actuals, "ingestion": ingestion,
                "sheet_actuals": ss.worksheet("round_config").get("U10:AK14")}
    (output / "verification.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    if any(r["status"] != "applied" or r["feedback_status"] != "applied" for r in actuals):
        raise RuntimeError("Backfill actuals/forecast validation did not pass")
    if [r["status"] for r in ingestion] != ["applied", "applied", "applied", "not_due"]:
        raise RuntimeError("Backfill feedback ingestion did not pass")


if __name__ == "__main__":
    main()
