import pytest

from publish_sim_fairs import _scope_live_tee_groups


def test_only_explicitly_unavailable_groups_are_omitted():
    contract = {"groups": [["a", "b", "c"], ["d", "e", "streb"], ["f", "g", "wd"]]}
    scoped = _scope_live_tee_groups(contract, set("abcdefg"), {"streb", "wd"})
    assert scoped["groups"] == [["a", "b", "c"]]
    assert scoped["excluded_players"] == ["streb", "wd"]
    assert len(scoped["excluded_groups"]) == 2
    assert len(contract["groups"]) == 3


def test_unexplained_missing_player_still_blocks_publish():
    with pytest.raises(RuntimeError, match="unexplained"):
        _scope_live_tee_groups({"groups": [["a", "b", "unknown"]]}, {"a", "b"}, {"streb"})


def test_no_eligible_groups_cannot_claim_no_groups_offered():
    with pytest.raises(RuntimeError, match="no eligible groups"):
        _scope_live_tee_groups({"groups": [["a", "b", "streb"]]}, {"a", "b"}, {"streb"})
