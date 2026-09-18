from datetime import datetime, timezone

import pytest

from midweek_round_automation import NotReady, evaluate_odds_readiness
from odds_loader import guard_scraped_data


def payload(mixed=True):
    rows = []
    for i in range(5):
        rows.append({
            'event_id': '557', 'round': 2,
            'p1_player_name': f'local {i}', 'p2_player_name': f'other {i}',
            'odds': {'betonline': {'p1': '-110', 'p2': '-110'},
                     'pinnacle': {'p1': '-110', 'p2': '-110'}},
        })
        rows.append({
            'round': 2, 'p1_player_name': f'unscoped {i}',
            'p2_player_name': f'unknown {i}',
            'odds': {'betcris': {'p1': '-110', 'p2': '-110', 'line_verified': True,
                                'p1_line': None, 'p2_line': None}},
        })
    if mixed:
        rows.append({'event_id': '2026136', 'round': 2})
    return {'event_id': '557', 'round': 2, 'match_list': rows,
            'last_updated': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}


def test_mixed_events_cannot_lend_top_level_identity_to_untagged_book():
    data = payload()
    guarded = guard_scraped_data(data, 'round_matchups', round=2, event_ids={'557'})
    assert len(guarded['match_list']) == 5
    assert len(data['match_list']) == 11
    with pytest.raises(NotReady, match='betcris=0'):
        evaluate_odds_readiness(data, 557)
    ready = evaluate_odds_readiness(data, 557, required_books=('betonline', 'pinnacle'))
    assert ready.counts == {'betonline': 5, 'pinnacle': 5}


def test_single_event_legacy_untagged_rows_remain_usable():
    data = payload(mixed=False)
    guarded = guard_scraped_data(data, 'round_matchups', round=2, event_ids={'557'})
    assert len(guarded['match_list']) == 10
    assert evaluate_odds_readiness(data, 557).counts == {'betcris': 5, 'betonline': 5}
