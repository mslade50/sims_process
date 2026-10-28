import pandas as pd
import publish_sim_fairs as publisher


def test_field_keeps_zero_win_players_and_ignores_generic_winners(tmp_path, monkeypatch):
    pd.DataFrame({"player_name": ["alpha, a", "zero, z"],
                  "simulated_win_prob": [1.0, 0.0]}).to_csv(
        tmp_path / "finish_equity_current.csv", index=False)
    pd.DataFrame({"player_name": ["alpha, a"]}).to_csv(
        tmp_path / "simulated_probs.csv", index=False)
    monkeypatch.setattr(publisher, "_find", lambda *names: next(
        (tmp_path / name for name in names if (tmp_path / name).is_file()), None))
    assert publisher._build_field("current", {}, use_live=False) == ["alpha, a", "zero, z"]
