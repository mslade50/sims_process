"""Rebuild an assumed skill chain only from validated completed results."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROVISIONAL_COLUMNS = {'assumed_r2_strokes', 'provisional_assumptions',
                       'provisional_input_sha256'}


def historical_config(config):
    """Keep per-round forecasts; remove generic rows for today's live target."""
    return {k: v for k, v in config.items() if k not in (
        'wind', 'dew', 'expected_score_1', 'expected_score_2', 'expected_score_3')}


def completed_strokes(row, par):
    """Provider round is score to par; SG is relative to the field."""
    try:
        score, thru = float(row.get('round')), float(row.get('thru'))
    except (TypeError, ValueError):
        return None
    if thru == 18 and math.isfinite(score) and score.is_integer():
        return int(par + score)
    return None


def validate_completed_stats(stats, event_name, par):
    required = {'player_name', 'position', 'thru', 'round', 'sg_total', 'event_name'}
    if stats is None or not required.issubset(stats.columns):
        raise ValueError('Official round results are unavailable or incomplete')
    if not stats.event_name.eq(event_name).all():
        raise ValueError('Official round results have the wrong event identity')
    frame = stats.loc[~stats.position.astype(str).str.contains('WD|DQ', case=False)].copy()
    frame['player_name'] = frame.player_name.astype(str).str.lower().str.strip()
    if frame.empty or frame.player_name.duplicated().any():
        raise ValueError('Official round results have an empty or duplicate field')
    if not pd.to_numeric(frame.thru, errors='coerce').eq(18).all():
        raise ValueError('Official round results require 18 holes for every player, including CUT')
    if any(completed_strokes(row, par) is None for _, row in frame.iterrows()):
        raise ValueError('Official round results have invalid round scores')
    if not pd.to_numeric(frame.sg_total, errors='coerce').map(
        lambda value: pd.notna(value) and math.isfinite(value)).all():
        raise ValueError('Official round results have invalid strokes gained')
    return frame


def reconcile(root, completed_round, event_id, course_id, tourney, par, api_key,
              run, *, dry_run=False):
    """Archive assumptions and rebuild earlier rounds before the live update."""
    root = Path(root)
    from sync_event_files import _manifest_allowed
    allowed = _manifest_allowed(str(tourney).lower())
    files = []
    provisional = False
    for rnd in range(1, completed_round + 1):
        for name in (f'r{rnd}_live_model.csv', f'model_predictions_r{rnd}.csv'):
            path = root / name
            if not path.exists() and name in allowed:
                path = root / 'dashboard_data' / name
            if path.exists():
                frame = pd.read_csv(path)
                files.append(path)
                provisional |= bool(PROVISIONAL_COLUMNS.intersection(frame.columns))
    if not provisional:
        return
    if os.getenv('PROVISIONAL_ROUND_INPUT'):
        raise ValueError('Official history rebuild cannot use a provisional override')
    from api_utils import fetch_field_updates, fetch_live_stats
    from live_sim_exclusions import filter_live_sim_players
    field = fetch_field_updates(api_key, teetime_col=f'r{completed_round + 1}_teetime',
                               include_course=True, fill_missing_teetimes=False)
    if (field is None or str(field.attrs.get('event_id')) != str(event_id)
            or field.attrs.get('course_ids') != [int(course_id)]):
        raise ValueError('Official history rebuild has the wrong event/course identity')
    observations = {}
    for rnd in range(1, completed_round):
        stats = fetch_live_stats(rnd, api_key, include_score=True)
        if stats is not None:
            stats = filter_live_sim_players(stats)
        observations[rnd] = validate_completed_stats(stats, field.attrs.get('event_name'), par)
    print(f'  Official history: rebuilding R1-R{completed_round - 1} for event {event_id}')
    if dry_run:
        return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive = root / 'sheet_backups' / 'official_history' / stamp
    archive.mkdir(parents=True)
    for path in files:
        destination = archive / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    receipt = dict(event_id=event_id, course_id=course_id, tourney=tourney,
                   completed_round=completed_round, status='rebuilding', rounds={})
    receipt_path = archive / 'receipt.json'
    def save():
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    for rnd, stats in observations.items():
        source = stats.to_csv(index=False).encode()
        (archive / f'official_r{rnd}.csv').write_bytes(source)
        receipt['rounds'][str(rnd)] = dict(source_sha256=hashlib.sha256(source).hexdigest(),
                                         players=len(stats))
    save()
    try:
        for rnd, stats in observations.items():
            run([sys.executable, 'live_stats_engine.py', '--round', str(rnd),
                 '--dry-run', '--no-sheet-writes', '--historical-rebuild'],
                f'Rebuild official R{rnd} history')
            rebuilt = pd.read_csv(root / f'r{rnd}_live_model.csv')
            if PROVISIONAL_COLUMNS.intersection(rebuilt.columns):
                raise ValueError(f'Rebuilt R{rnd} still contains provisional assumptions')
            rebuilt = validate_completed_stats(rebuilt, field.attrs.get('event_name'), par)
            actual = rebuilt.set_index('player_name')[['round', 'sg_total']].sort_index()
            expected = stats.set_index('player_name')[['round', 'sg_total']].sort_index()
            if not actual.equals(expected):
                raise ValueError(f'Rebuilt R{rnd} differs from official completed results')
            predictions = pd.read_csv(root / f'model_predictions_r{rnd + 1}.csv')
            if PROVISIONAL_COLUMNS.intersection(predictions.columns) or predictions.empty:
                raise ValueError(f'Rebuilt R{rnd + 1} predictions are still provisional or empty')
            receipt['rounds'][str(rnd)]['rebuilt_sha256'] = hashlib.sha256(
                (root / f'r{rnd}_live_model.csv').read_bytes()).hexdigest()
        receipt['status'] = 'complete'
        save()
    except Exception:
        receipt['status'] = 'failed'
        save()
        raise
