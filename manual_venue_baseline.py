"""Explicit owner-supplied scoring and variance priors for a new venue."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

CATEGORIES = ('sg_ott', 'sg_app', 'sg_arg', 'sg_putt')


def resolve_baselines(profile, event_id, course_id, par):
    if (profile['event_id'], profile['course_id']) != (event_id, course_id):
        raise ValueError('Manual venue profile belongs to another event/course')
    raw = {int(k): float(v) for k, v in profile['historical_relative_scores'].items()}
    if set(raw) != {1, 2, 3, 4}:
        raise ValueError('Manual scoring prior must specify all four rounds')
    uplift = float(profile['historical_weekend_field_adj'])
    current_uplift = float(profile['weekend_field_adj'])
    if not all(math.isfinite(v) for v in [*raw.values(), uplift, current_uplift]):
        raise ValueError('Scoring priors must be finite')
    if uplift < 0 or current_uplift < 0:
        raise ValueError('Surviving-field strength adjustments must be nonnegative')
    for key in ('category_mults', 'category_skew'):
        values = profile[key]
        if set(values) != set(CATEGORIES) or not all(math.isfinite(float(v)) for v in values.values()):
            raise ValueError('Manual category profile must have four finite values')
    if any(float(v) <= 0 for v in profile['category_mults'].values()):
        raise ValueError('Category SD multipliers must be positive')
    if not math.isfinite(profile['suggested_var_mult']) or profile['suggested_var_mult'] <= 0:
        raise ValueError('Total SD multiplier must be finite and positive')
    return {r: float(par) + raw[r] + (uplift if r >= 3 else 0) for r in raw}


def expected_score(base, field_strength, weekend_uplift, wind, dew):
    return base - field_strength - weekend_uplift + wind + dew


def run_manual_venue_baseline(sb, *, write_sheet=True):
    import sim_inputs as config
    profile = getattr(config, 'manual_venue_profile', None)
    if profile is None:
        return False
    baselines = resolve_baselines(profile, config.event_ids[0], config.course_id, config.course_par)
    import pandas as pd
    from sheets_storage import get_spreadsheet

    field = pd.read_csv('field_updates.csv')
    preds = pd.read_csv(f'pre_course_fit_{config.tourney}.csv')
    if (field.empty or set(pd.to_numeric(field['event_id'])) != {config.event_ids[0]}
            or field['dg_id'].duplicated().any() or preds['dg_id'].duplicated().any()
            or set(field['dg_id']) != set(preds['dg_id'])
            or not pd.to_numeric(preds['pred'], errors='coerce').map(math.isfinite).all()):
        raise ValueError('Current field and finite course-fit predictions must match exactly')
    strength = float(preds['pred'].mean())
    ws = get_spreadsheet().worksheet('round_config')
    grid = ws.get_all_values()
    params = {r[0].strip().lower(): r[1] if len(r) > 1 else '' for r in grid if r and r[0].strip()}
    if (params.get('tourney'), params.get('event_id'), params.get('course_id'), params.get('round')) != (
            config.tourney, str(config.event_ids[0]), str(config.course_id), '0'):
        raise ValueError('Live Sheet must identify this event/course at round zero')
    wind_factor, _ = sb.get_wind_coefficient(config.event_ids, config.baseline_wind, config.wind_override, config.course_id)
    dew_factor = float(params.get('dew_calculation') or 0)
    dew_base = float(params.get('dewpoint_base') or 0)
    breakdown = []
    for rnd in range(1, 5):
        winds = []
        for row in grid[17:21]:
            if len(row) > 5 and row[4].strip() == f'wind_r{rnd}':
                winds = [float(v) for v in row[5].split(',') if v.strip()]
        if not winds or not all(math.isfinite(x) for x in winds):
            raise ValueError(f'Missing finite R{rnd} weather; run humidity.py first')
        tee_column = f'r{rnd}_teetime'
        if tee_column not in field or field[tee_column].isna().all():
            # Monday's real field has no tee times; use one common full-window
            # forecast instead of the API helper's synthetic 10 AM defaults.
            average_wind = sum(winds) / len(winds)
            print(f'R{rnd}: no published tee times; common forecast-window wind')
        else:
            average_wind = sb._compute_field_avg_wind(winds, os.getenv('DATAGOLF_API_KEY'), rnd)
        wind = average_wind * wind_factor
        col = {1: 6, 2: 10, 3: 14, 4: 18}[rnd]
        dews = []
        for row in grid[2:17]:
            try:
                v = float(row[col])
                if math.isfinite(v):
                    dews.append(v)
            except (IndexError, ValueError):
                pass
        if not dews:
            raise ValueError(f'Missing R{rnd} dew forecast; run humidity.py first')
        dew = (sum(dews) / len(dews) - dew_base) * dew_factor
        weekend = profile['weekend_field_adj'] if rnd >= 3 else 0.0
        score = expected_score(baselines[rnd], strength, weekend, wind, dew)
        breakdown.append([f'R{rnd}', round(baselines[rnd], 4), round(-strength-weekend, 4),
                          round(wind, 4), round(dew, 4), round(score, 4)])
    output = pd.DataFrame([{'year': 'FINAL', 'round_num': r, 'baseline': b,
                           'source': profile['source'],
                           'historical_relative_score': profile['historical_relative_scores'][r],
                           'historical_field_adj': profile['historical_weekend_field_adj'] if r >= 3 else 0}
                          for r, b in baselines.items()])
    output.to_csv(f'scoring_baseline_{config.tourney}.csv', index=False)
    updates = {f'expected_score_r{r}': breakdown[r-1][-1] for r in range(1, 5)}
    updates.update(expected_score_1=breakdown[0][-1], wind_override=round(wind_factor, 6),
                   suggested_var_mult=profile['suggested_var_mult'])
    for name in CATEGORIES:
        updates[f'cat_mult_{name[3:]}'] = profile['category_mults'][name]
        updates[f'cat_skew_{name[3:]}'] = profile['category_skew'][name]
    receipt = {'event_id': config.event_ids[0], 'course_id': config.course_id,
               'source': profile['source'], 'baselines': baselines, 'field_strength': strength,
               'breakdown': breakdown, 'parameters': updates, 'sheet_written': False}
    if write_sheet:
        row_map = {r[0].strip().lower(): i+1 for i, r in enumerate(grid) if r and r[0].strip()}
        missing = set(updates) - row_map.keys()
        if missing:
            raise ValueError(f'Missing Sheet parameter rows: {sorted(missing)}')
        cells = [{'range': f'B{row_map[k]}', 'values': [[v]]} for k, v in updates.items()]
        cells.extend([
            {'range': 'T2', 'values': [['PRE-TOURNEY EXPECTED: manual venue prior']]},
            {'range': 'U2:Z6', 'values': [['Round','Base Score','Field Adj','Wind Impact','Dew Impact','= Expected'], *breakdown]},
        ])
        ws.batch_update(cells, value_input_option='RAW')
        actual = {r[0].strip().lower(): r[1] for r in ws.get('A:B') if len(r) > 1}
        for k, v in updates.items():
            if not math.isclose(float(actual[k]), v, abs_tol=1e-7):
                raise RuntimeError(f'Sheet readback failed for {k}')
        verified = ws.get('U3:Z6')
        if len(verified) != 4 or any(
                len(row) != 6 or row[0] != want[0]
                or any(not math.isclose(float(a), b, abs_tol=1e-7) for a,b in zip(row[1:],want[1:]))
                for row,want in zip(verified,breakdown)):
            raise RuntimeError('Sheet baseline breakdown readback failed')
        receipt['sheet_written'] = True
    Path(f'manual_venue_baseline_{config.tourney}.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2))
    return True
