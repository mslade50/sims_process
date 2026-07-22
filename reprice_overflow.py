"""Import golf_scraping's credential-free reprice overflow before grading."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


DEFAULT_PATH = Path(__file__).parent / "permanent_data" / "scraped_odds" / "reprice_bet_overflow.json"
SUPPORTED_SCHEMA = 1


def _market(value) -> str:
    market = str(value or "").strip().lower()
    return "win" if market == "winner" else market


def _player(value) -> str:
    name = " ".join(str(value or "").strip().lower().split())
    if "," in name:
        last, first = (part.strip() for part in name.split(",", 1))
        name = f"{first} {last}".strip()
    return name


def _key(event_id, player, market) -> tuple[str, str, str]:
    return str(event_id).strip(), _player(player), _market(market)


def load_entries(path=None) -> list[dict]:
    source = Path(path or os.getenv("REPRICE_OVERFLOW_PATH") or DEFAULT_PATH)
    if not source.exists():
        print(f"  [overflow] No reprice overflow file at {source}; nothing to import.")
        return []
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not read reprice overflow {source}: {exc}") from exc
    if payload.get("schema_version") != SUPPORTED_SCHEMA:
        raise RuntimeError(
            f"unsupported reprice overflow schema {payload.get('schema_version')!r}; "
            f"expected {SUPPORTED_SCHEMA}"
        )
    entries = []
    for event in (payload.get("events") or {}).values():
        if isinstance(event, dict):
            entries.extend(item for item in (event.get("bets") or []) if isinstance(item, dict))
    return entries


def select_new_entries(entries, event_ids, existing_keys) -> list[dict]:
    wanted = {str(event_id).strip() for event_id in event_ids}
    seen = set(existing_keys)
    selected = []
    for entry in sorted(entries, key=lambda item: (str(item.get("discovered_at") or ""), str(item.get("key") or ""))):
        key = _key(entry.get("event_id"), entry.get("player_name"), entry.get("market_type"))
        if key[0] not in wanted or not key[1] or not key[2] or key in seen:
            continue
        seen.add(key)
        selected.append(entry)
    return selected


def _frame(entries: list[dict]) -> pd.DataFrame:
    rows = []
    for entry in entries:
        market = _market(entry.get("market_type"))
        fair = float(entry.get("fair_prob") or 0.0)
        row = {
            "player_name": _player(entry.get("player_name")),
            "market_type": market,
            "bookmaker": str(entry.get("sportsbook") or "").strip().lower(),
            "decimal_odds": float(entry.get("decimal_odds") or 0.0),
            "american_odds": int(entry.get("american_odds") or 0),
            "my_fair": int(entry.get("fair_american") or 0),
            "sim_prob": fair,
            "edge": float(entry.get("edge_pp") or 0.0),
            "stake": float(entry.get("kelly_stake") or 0.0),
            "my_pred": None,
            "sample": None,
            "type_on": "",
        }
        probability_column = {
            "win": "simulated_win_prob",
            "top_5": "top_5",
            "top_10": "top_10",
            "top_20": "top_20",
        }.get(market)
        if probability_column:
            row[probability_column] = fair
        rows.append(row)
    return pd.DataFrame(rows)


def import_overflow(spreadsheet, week_events, *, path=None, dry_run=False) -> int:
    """Import unseen overflow rows for completed events; safe to retry."""
    from sheets_storage import load_finish_position_keys, store_finish_positions

    event_map = {str(event_id): str(event_name) for event_id, event_name, _ in week_events}
    existing = set()
    for event_id in event_map:
        for item in load_finish_position_keys(spreadsheet, event_id):
            existing.add(_key(event_id, item.get("player_name"), item.get("market_type")))

    selected = select_new_entries(load_entries(path), event_map, existing)
    if not selected:
        print("  [overflow] No new reprice bets to import.")
        return 0
    if dry_run:
        print(f"  [overflow] DRY RUN: would import {len(selected)} reprice bet(s).")
        return len(selected)

    imported = 0
    for event_id, event_name in event_map.items():
        event_entries = [item for item in selected if str(item.get("event_id")) == event_id]
        if not event_entries:
            continue
        frame = _frame(event_entries)
        store_finish_positions(
            frame, event_name, event_id,
            dg_id_lookup={}, spreadsheet=spreadsheet,
        )
        imported += len(frame)
    print(f"  [overflow] Imported {imported} new reprice bet(s) before grading.")
    return imported
