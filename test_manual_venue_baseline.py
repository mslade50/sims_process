from copy import deepcopy
import pytest
from manual_venue_baseline import resolve_baselines, expected_score


def profile():
    cats = ['sg_ott','sg_app','sg_arg','sg_putt']
    return dict(event_id=557, course_id=942,
                historical_relative_scores={1:-.5,2:-.5,3:-.75,4:-.75},
                historical_weekend_field_adj=.3, weekend_field_adj=.3,
                category_mults=dict.fromkeys(cats,1.), category_skew=dict.fromkeys(cats,0.),
                suggested_var_mult=1.05)


def test_normalizes_historical_weekend_without_double_counting():
    b=resolve_baselines(profile(),557,942,71)
    assert b == pytest.approx({1:70.5,2:70.5,3:70.55,4:70.55})
    # Normalize historical survivor selection once, then apply THIS weekend's
    # survivor strength once; zero overall skill/weather recovers the raw prior.
    assert expected_score(b[3],0,.3,0,0) == pytest.approx(70.25)
    assert expected_score(b[1],.4,0,.2,-.1) == pytest.approx(70.2)


def test_rejects_stale_event_identity():
    with pytest.raises(ValueError,match='another event/course'):
        resolve_baselines(profile(),60,688,70)


@pytest.mark.parametrize('mutation', ['missing_round','nonfinite','negative_sd','negative_uplift'])
def test_invalid_profiles_fail_closed(mutation):
    p=deepcopy(profile())
    if mutation=='missing_round':p['historical_relative_scores'].pop(4)
    if mutation=='nonfinite':p['historical_relative_scores'][1]=float('nan')
    if mutation=='negative_sd':p['category_mults']['sg_app']=-1
    if mutation=='negative_uplift':p['historical_weekend_field_adj']=-.3
    with pytest.raises(ValueError):resolve_baselines(p,557,942,71)
