import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pyarrow as pa

import nightly_round_sim as nightly
import publish_sim_fairs as psf


def _write_complete_package(root: Path, *, tourney="test_event", sim_round=3):
    started_at = time.time()
    event_dir = root / tourney
    event_dir.mkdir()
    players = ["alpha player", "beta player"]
    draws = 4

    pd.DataFrame(
        {"player_name": players, f"scores_r{sim_round}": [0.2, -0.2]}
    ).to_csv(root / f"model_predictions_r{sim_round}.csv", index=False)

    cache = pd.DataFrame(
        [[68, 69, 70, 71], [69, 70, 71, 72]],
        index=pd.Index(players, name="player_name"),
    )
    cache.to_parquet(event_dir / f"sim_cache_r{sim_round}.parquet")
    (event_dir / f"sim_cache_r{sim_round}_meta.json").write_text(
        json.dumps(
            {
                "sim_round": sim_round,
                "num_players": len(players),
                "num_sims": draws,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"player_name": player, "score": score, "prob": 0.5}
            for player in players
            for score in (69, 70)
        ]
    ).to_parquet(event_dir / f"round_score_probs_r{sim_round}.parquet", index=False)

    final_scores = np.array([[270, 271, 272, 273], [271, 270, 273, 272]])
    np.save(root / f"final_scores_live_{tourney}.npy", final_scores)
    np.save(root / f"made_cut_live_{tourney}.npy", np.ones_like(final_scores, dtype=bool))
    np.save(root / f"standings_r2_live_{tourney}.npy", final_scores - 70)
    np.save(root / f"standings_r3_live_{tourney}.npy", final_scores - 35)
    (root / f"player_names_live_{tourney}.json").write_text(
        json.dumps(players), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "player_name": players,
            "simulated_win_prob": [0.5, 0.5],
            "top_5": [1.0, 1.0],
            "top_10": [1.0, 1.0],
            "top_20": [1.0, 1.0],
        }
    ).to_csv(root / "simulated_probs_live.csv", index=False)
    pd.DataFrame(
        [
            {"player_name": "alpha player", "rank": 1, "prob_u": 0.5},
            {"player_name": "alpha player", "rank": 2, "prob_u": 0.5},
            {"player_name": "beta player", "rank": 1, "prob_u": 0.5},
            {"player_name": "beta player", "rank": 2, "prob_u": 0.5},
        ]
    ).to_parquet(root / f"rank_probs_live_{tourney}.parquet", index=False)
    return started_at


def test_complete_live_artifact_contract_accepts_one_fresh_paired_run(tmp_path):
    started_at = _write_complete_package(tmp_path)

    result = nightly._validate_complete_live_artifacts(
        tmp_path, "test_event", 3, started_at
    )

    assert result == {
        "tourney": "test_event",
        "sim_round": 3,
        "round_players": 2,
        "round_draws": 4,
        "tournament_players": 2,
        "tournament_draws": 4,
    }


def test_complete_live_artifact_contract_rejects_stale_rank_tape(tmp_path):
    started_at = _write_complete_package(tmp_path)
    rank_path = tmp_path / "rank_probs_live_test_event.parquet"
    old = started_at - 60
    os.utime(rank_path, (old, old))

    with pytest.raises(RuntimeError, match="live rank probabilities"):
        nightly._validate_complete_live_artifacts(
            tmp_path, "test_event", 3, started_at
        )


def test_complete_live_artifact_contract_rejects_mismatched_standings(tmp_path):
    started_at = _write_complete_package(tmp_path)
    np.save(tmp_path / "standings_r3_live_test_event.npy", np.zeros((2, 3)))

    with pytest.raises(RuntimeError, match="R3 standings tape is not paired"):
        nightly._validate_complete_live_artifacts(
            tmp_path, "test_event", 3, started_at
        )


def test_complete_live_artifact_contract_rejects_prediction_field_mismatch(tmp_path):
    started_at = _write_complete_package(tmp_path)
    pd.DataFrame(
        {"player_name": ["last week's player"], "scores_r3": [0.0]}
    ).to_csv(tmp_path / "model_predictions_r3.csv", index=False)

    with pytest.raises(RuntimeError, match="prediction player set"):
        nightly._validate_complete_live_artifacts(
            tmp_path, "test_event", 3, started_at
        )


def test_nightly_exports_publish_gate_only_after_validation(tmp_path, monkeypatch):
    output = tmp_path / "github-output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    nightly._write_github_outputs(
        {"should_publish": True, "sim_round": 3, "tourney": "test_event"}
    )

    assert output.read_text(encoding="utf-8").splitlines() == [
        "should_publish=true",
        "sim_round=3",
        "tourney=test_event",
    ]


def _complete_payload():
    return {
        "event_id": "99",
        "tourney": "test_event",
        "sim_run_at": "2026-08-23 12:00:00 UTC",
        "round": 3,
        "field": ["alpha player", "beta player"],
        "outrights": {
            "winner": {"alpha player": 0.5},
            "top_5": {"alpha player": 1.0},
            "top_10": {"alpha player": 1.0},
            "top_20": {"alpha player": 1.0},
            "make_cut": {"alpha player": 1.0},
        },
        "outrights_nodh": {
            "top_5": {"alpha player": 1.0},
            "top_10": {"alpha player": 1.0},
            "top_20": {"alpha player": 1.0},
        },
        "matchups": [["alpha player", "beta player", 0.5]],
        "round_scores": {"alpha player": {69: 0.5, 70: 0.5}},
        "outrights_source": "live",
        "outrights_sim_run_at": "2026-08-23 12:00:00 UTC",
        "matchups_source": "final_scores_live",
        "matchups_sim_run_at": "2026-08-23 12:00:00 UTC",
    }


def test_complete_live_publish_contract_rejects_wrong_round_and_pre_market():
    payload = _complete_payload()
    payload["outrights_source"] = "pre"

    with pytest.raises(RuntimeError) as exc:
        psf._validate_complete_live_payload(payload, expected_round=4)

    message = str(exc.value)
    assert "outrights_source='pre'" in message
    assert "round=3 (expected 4)" in message


def test_complete_live_publish_contract_accepts_complete_payload():
    psf._validate_complete_live_payload(_complete_payload(), expected_round=3)


@pytest.mark.parametrize(
    ("full_ok", "mask_ok", "matchup_ok", "failure_text"),
    [
        (False, True, True, "full tournament tape"),
        (True, False, True, "full made-cut tape"),
        (True, True, False, "live matchup tape"),
    ],
)
def test_strict_release_requires_every_full_tape(
    monkeypatch, full_ok, mask_ok, matchup_ok, failure_text
):
    table = pa.Table.from_pandas(pd.DataFrame({"0": [1]}, index=["alpha player"]))
    monkeypatch.setenv("GH_TOKEN", "test-token")
    monkeypatch.setattr(psf, "_upload_full_tape_release", lambda *_a, **_k: full_ok)
    monkeypatch.setattr(
        psf, "_build_made_cut_mask", lambda *_a, **_k: table if mask_ok else None
    )
    monkeypatch.setattr(psf, "_upload_release_asset", lambda *_a, **_k: None)
    monkeypatch.setattr(
        psf, "_upload_matchup_tape_release", lambda *_a, **_k: matchup_ok
    )

    with pytest.raises(RuntimeError, match=failure_text):
        psf._upload_release_tape_family(
            _complete_payload(), {}, strict=True
        )


def test_strict_release_failure_aborts_before_git_publish(tmp_path, monkeypatch):
    table = pa.Table.from_pandas(pd.DataFrame({"0": [1]}, index=["alpha player"]))
    monkeypatch.setattr(psf, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(psf, "LOCAL_OUT", tmp_path / "sim_fairs.json")
    monkeypatch.setattr(psf, "LOCAL_SAMPLES", tmp_path / "round_samples.parquet")
    monkeypatch.setattr(
        psf, "LOCAL_TOURN_SAMPLES", tmp_path / "tournament_samples.parquet"
    )
    monkeypatch.setattr(psf, "build_payload", _complete_payload)
    monkeypatch.setattr(psf, "_name_replacements", lambda: {})
    monkeypatch.setattr(psf, "sync_r1_prediction_artifact", lambda **_k: None)
    monkeypatch.setattr(
        psf,
        "_build_round_samples",
        lambda *_a, **_k: pd.DataFrame([[68, 69]], index=["alpha player"]),
    )
    monkeypatch.setattr(psf, "write_round_h2h", lambda *_a, **_k: ["round_h2h_r3.parquet"])
    monkeypatch.setattr(psf, "write_round_3ball", lambda *_a, **_k: [])
    monkeypatch.setattr(psf, "_build_tournament_samples", lambda *_a, **_k: table)
    monkeypatch.setattr(psf, "_build_made_cut_mask", lambda *_a, **_k: table)
    monkeypatch.setattr(
        psf,
        "_upload_release_tape_family",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("release incomplete")),
    )
    git_calls = []
    monkeypatch.setattr(psf, "_git_push", lambda *_a, **_k: git_calls.append(1))

    with pytest.raises(RuntimeError, match="release incomplete"):
        psf.publish(push=True, require_complete_live=True, expected_round=3)

    assert git_calls == []


def test_live_make_cut_uses_paired_mask_not_conflicting_pre_event_csv(
    tmp_path, monkeypatch
):
    tourney = "test_event"
    players = ["alpha player", "beta player"]
    finish = pd.DataFrame(
        {
            "player_name": players,
            "simulated_win_prob": [0.5, 0.5],
            "top_5": [1.0, 1.0],
            "top_10": [1.0, 1.0],
            "top_20": [1.0, 1.0],
            "top_5_nodh": [1.0, 1.0],
            "top_10_nodh": [1.0, 1.0],
            "top_20_nodh": [1.0, 1.0],
        }
    )
    finish.to_csv(tmp_path / "simulated_probs_live.csv", index=False)
    pd.DataFrame(
        {"player_name": players, "make_cut": [0.99, 0.99]}
    ).to_csv(tmp_path / f"make_cut_probs_{tourney}.csv", index=False)
    np.save(tmp_path / f"final_scores_live_{tourney}.npy", np.zeros((2, 4)))
    np.save(
        tmp_path / f"made_cut_live_{tourney}.npy",
        np.array([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=bool),
    )
    (tmp_path / f"player_names_live_{tourney}.json").write_text(
        json.dumps(players), encoding="utf-8"
    )
    monkeypatch.setattr(psf, "PROJECT_ROOT", tmp_path)

    outrights, _ = psf._build_outrights(
        tourney, cut_line=65, repl={}, use_live=True
    )

    assert outrights["make_cut"] == {
        "alpha player": 0.5,
        "beta player": 0.25,
    }


def test_nightly_workflow_is_strict_and_side_effect_free():
    root = Path(__file__).parent
    workflow = (root / ".github/workflows/nightly-round-sim.yml").read_text(
        encoding="utf-8"
    )
    nightly_source = (root / "nightly_round_sim.py").read_text(encoding="utf-8")

    assert "REQUIRE_SIM_FAIRS_PUBLISH: '1'" in workflow
    assert "BOARD_SUPPRESS_SIM_CASCADE: '1'" in workflow
    assert "--require-complete-live" in workflow
    assert "--expected-round" in workflow
    assert "steps.live_sim.outputs.should_publish == 'true'" in workflow
    assert "publish_sim_fairs.py --round-h2h-only" not in workflow
    assert '[python, "round_sim.py", "--dry-run"]' in nightly_source
    assert '"live_stats_engine.py", "--dry-run", "--no-sheet-writes"' in nightly_source
    assert "already exists, skipping live_stats_engine.py" not in nightly_source
    assert "copied to root. Skipping live_stats_engine.py" not in nightly_source
    assert 'from maker_alerts import send_telegram' not in nightly_source


def test_live_stats_engine_exposes_no_sheet_write_mode():
    source = (Path(__file__).parent / "live_stats_engine.py").read_text(
        encoding="utf-8"
    )
    assert '"--no-sheet-writes"' in source
    assert "if args.no_sheet_writes:" in source
    assert "round_num < 4 and not args.no_sheet_writes" in source
