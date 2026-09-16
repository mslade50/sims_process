import sys
from datetime import datetime
from types import SimpleNamespace

import pytest
import nightly_round_sim as nightly


def test_off_week_skips_stale_sheet_and_never_publishes(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'argv', ['nightly_round_sim.py'])
    monkeypatch.setattr(nightly, '_setup_env', lambda: str(tmp_path))
    monkeypatch.setitem(sys.modules, 'api_utils', SimpleNamespace(fetch_pga_events_this_week=lambda key: []))
    def forbidden():
        raise AssertionError('off week must not read stale Sheet config')
    monkeypatch.setitem(sys.modules, 'sheet_config', SimpleNamespace(load_config=forbidden))
    output = tmp_path / 'outputs.txt'
    monkeypatch.setenv('GITHUB_OUTPUT', str(output))
    with pytest.raises(SystemExit) as result:
        nightly.main()
    assert result.value.code == 0
    assert output.read_text().splitlines() == ['should_publish=false']


@pytest.mark.parametrize('events,code,reads', [([{'event_id':'557'}],0,1),(None,1,0)])
def test_event_week_reads_sheet_but_schedule_error_does_not(monkeypatch, tmp_path, events, code, reads):
    monkeypatch.setattr(sys, 'argv', ['nightly_round_sim.py'])
    monkeypatch.setattr(nightly, '_setup_env', lambda: str(tmp_path))
    def schedule(key):
        if events is None:
            raise RuntimeError('schedule unavailable')
        return events
    monkeypatch.setitem(sys.modules, 'api_utils', SimpleNamespace(fetch_pga_events_this_week=schedule))
    observed=[]
    def sheet():
        observed.append(True)
        return {'round_num':4,'tourney':'test'}
    monkeypatch.setitem(sys.modules, 'sheet_config', SimpleNamespace(load_config=sheet))
    monkeypatch.setitem(sys.modules, 'sim_inputs', SimpleNamespace(tourney='test'))
    monkeypatch.setattr(nightly,'_run_subprocess',lambda *a:pytest.fail('must not simulate'))
    with pytest.raises(SystemExit) as result:
        nightly.main()
    assert result.value.code==code
    assert len(observed)==reads


def schedule_payload(year=2026, rows=None):
    return {'tour':'pga','season':str(year),'schedule':rows if rows is not None else [
        {'tour':'pga','start_date':'2026-08-27','event_id':'60','status':'completed'},
        {'tour':'pga','start_date':'2026-09-17','event_id':'557','status':'upcoming'},
    ]}


@pytest.mark.parametrize('now,expected', [
    ('2026-09-04T07:00:00+00:00',[]),
    ('2026-09-13T07:00:00+00:00',[]),
    ('2026-09-16T11:00:00+00:00',['557']),
    ('2026-09-21T02:00:00+00:00',['557']),  # Still Sunday in New York.
    ('2026-09-21T05:00:00+00:00',[]),
])
def test_schedule_week_is_eastern_and_does_not_reuse_last_event(monkeypatch,now,expected):
    import api_utils
    calls=[]
    def get(url,**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(raise_for_status=lambda:None,json=schedule_payload)
    monkeypatch.setattr(api_utils.requests,'get',get)
    events=api_utils.fetch_pga_events_this_week('test',now=datetime.fromisoformat(now))
    assert [v['event_id'] for v in events]==expected
    assert calls[0]['params']['tour']=='pga' and calls[0]['timeout']==20


@pytest.mark.parametrize('payload',[{},schedule_payload(rows=[]),schedule_payload(rows=[{'tour':'pga','start_date':'bad'}])])
def test_bad_schedule_is_an_error_not_an_off_week(monkeypatch,payload):
    import api_utils
    monkeypatch.setattr(api_utils.requests,'get',lambda *a,**k:SimpleNamespace(
        raise_for_status=lambda:None,json=lambda:payload))
    with pytest.raises(RuntimeError,match='Could not verify'):
        api_utils.fetch_pga_events_this_week('test',now=datetime.fromisoformat('2026-09-16T12:00:00+00:00'))


def test_http_error_does_not_expose_key(monkeypatch):
    import api_utils
    def get(*a,**k):
        raise api_utils.requests.HTTPError('url?key=private-test-key')
    monkeypatch.setattr(api_utils.requests,'get',get)
    with pytest.raises(RuntimeError) as result:
        api_utils.fetch_pga_events_this_week('private-test-key')
    assert 'private-test-key' not in str(result.value)


def test_year_boundary_checks_both_seasons_and_skips_cancelled(monkeypatch):
    import api_utils
    years=[]
    def get(url,params,**kwargs):
        year=params['season'];years.append(year)
        return SimpleNamespace(raise_for_status=lambda:None,json=lambda:schedule_payload(year,[
            {'tour':'pga','start_date':f'{year}-01-01','event_id':str(year),'status':'cancelled' if year==2026 else 'upcoming'}]))
    monkeypatch.setattr(api_utils.requests,'get',get)
    events=api_utils.fetch_pga_events_this_week('test',now=datetime.fromisoformat('2027-01-01T12:00:00+00:00'))
    assert years==[2026,2027] and [v['event_id'] for v in events]==['2027']
