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


def _complete_payload(**_kwargs):
    return {
        "event_id": "99",
        "tourney": "test_event",
        "sim_run_at": "2026-08-23 12:00:00 UTC",
        "round": 3,
        "field": ["alpha player", "beta player"],
        "outrights": {
            "winner": {"alpha player": 0.5, "beta player": 0.5},
            "top_5": {"alpha player": 1.0, "beta player": 1.0},
            "top_10": {"alpha player": 1.0, "beta player": 1.0},
            "top_20": {"alpha player": 1.0, "beta player": 1.0},
            "make_cut": {"alpha player": 1.0, "beta player": 0.0},
        },
        "outrights_nodh": {
            "top_5": {"alpha player": 1.0, "beta player": 1.0},
            "top_10": {"alpha player": 1.0, "beta player": 1.0},
            "top_20": {"alpha player": 1.0, "beta player": 1.0},
        },
        "matchups": [["alpha player", "beta player", 0.5]],
        "round_scores": {
            "alpha player": {69: 0.5, 70: 0.5},
            "beta player": {70: 0.5, 71: 0.5},
        },
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


def test_complete_live_publish_contract_rejects_partial_outright_coverage():
    payload = _complete_payload()
    del payload["outrights"]["winner"]["beta player"]

    with pytest.raises(RuntimeError, match=r"outrights\.winner field coverage=1/2"):
        psf._validate_complete_live_payload(payload, expected_round=3)


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
    monkeypatch.setattr(
        psf, "STRICT_RELEASE_MANIFEST", tmp_path / "sim_release_manifest.json"
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
    health = {
        "manifest_sha256": "live-health",
        "simulation_manifest": {"manifest_sha256": "simulation-health"},
    }
    prepared = {
        "simulation_manifest_sha256": "simulation-health",
        "live_tournament_manifest_sha256": "live-health",
        "generation": "test-generation",
        "generated_at": "2026-08-23 12:00:00 UTC",
        "git_tournament_samples": table,
        "git_made_cut": table,
    }
    monkeypatch.setattr(
        psf, "_load_and_validate_strict_live_health", lambda *_a, **_k: (health, {})
    )
    monkeypatch.setattr(
        psf, "_require_strict_live_outright_payload", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        psf, "_build_strict_release_package", lambda *_a, **_k: prepared
    )
    monkeypatch.setattr(
        psf,
        "_write_strict_release_manifest",
        lambda *_a, **_k: {"manifest_sha256": "package"},
    )
    monkeypatch.setattr(
        psf, "_require_strict_release_manifest_current", lambda *_a, **_k: None
    )
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


def test_strict_live_outrights_ignore_fallbacks_and_preserve_zeroes(
    tmp_path, monkeypatch
):
    tourney = "test_event"
    players = ["alpha player", "beta player"]
    finish = pd.DataFrame(
        {
            "player_name": players,
            "simulated_win_prob": [0.0, 1.0],
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
        {
            "player_name": players,
            "simulated_win_prob": [0.9, 0.1],
            "top_5": [0.9, 0.1],
            "top_10": [0.9, 0.1],
            "top_20": [0.9, 0.1],
        }
    ).to_csv(tmp_path / "simulated_probs.csv", index=False)
    pd.DataFrame(
        {
            "player_name": players,
            "simulated_win_prob": [0.8, 0.2],
            "top_5": [0.8, 0.2],
            "top_10": [0.8, 0.2],
            "top_20": [0.8, 0.2],
        }
    ).to_csv(tmp_path / f"finish_equity_{tourney}.csv", index=False)
    np.save(tmp_path / f"final_scores_live_{tourney}.npy", np.zeros((2, 4)))
    np.save(
        tmp_path / f"made_cut_live_{tourney}.npy",
        np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=bool),
    )
    (tmp_path / f"player_names_live_{tourney}.json").write_text(
        json.dumps(players), encoding="utf-8"
    )
    monkeypatch.setattr(psf, "PROJECT_ROOT", tmp_path)

    outrights, nodh, field = psf._build_strict_live_outright_family(
        tourney, {}
    )

    assert field == players
    assert outrights["winner"] == {
        "alpha player": 0.0,
        "beta player": 1.0,
    }
    assert outrights["make_cut"] == {
        "alpha player": 0.0,
        "beta player": 1.0,
    }
    assert set(outrights["top_5"]) == set(players)
    assert set(nodh["top_5"]) == set(players)


def test_strict_live_outright_provenance_rejects_payload_mutation(
    tmp_path, monkeypatch
):
    tourney = "test_event"
    players = ["alpha player", "beta player"]
    finish = pd.DataFrame(
        {
            "player_name": players,
            "simulated_win_prob": [0.0, 1.0],
            "top_5": [1.0, 1.0],
            "top_10": [1.0, 1.0],
            "top_20": [1.0, 1.0],
            "top_5_nodh": [1.0, 1.0],
            "top_10_nodh": [1.0, 1.0],
            "top_20_nodh": [1.0, 1.0],
        }
    )
    finish.to_csv(tmp_path / "simulated_probs_live.csv", index=False)
    np.save(tmp_path / f"final_scores_live_{tourney}.npy", np.zeros((2, 4)))
    np.save(
        tmp_path / f"made_cut_live_{tourney}.npy",
        np.ones((2, 4), dtype=bool),
    )
    (tmp_path / f"player_names_live_{tourney}.json").write_text(
        json.dumps(players), encoding="utf-8"
    )
    monkeypatch.setattr(psf, "PROJECT_ROOT", tmp_path)
    files = psf._strict_live_health_files(tourney)
    outrights, nodh, field = psf._build_strict_live_outright_family(
        tourney, {}, files=files
    )
    payload = {
        "tourney": tourney,
        "field": field,
        "outrights": outrights,
        "outrights_nodh": nodh,
        "outrights_source": "live",
        "outrights_sim_run_at": psf._utc_stamp(
            files["finish_probs"].stat().st_mtime
        ),
        "matchups_source": "final_scores_live",
        "matchups_sim_run_at": psf._utc_stamp(
            files["final_scores"].stat().st_mtime
        ),
    }
    psf._require_strict_live_outright_payload(payload, {}, files)

    payload["outrights"]["winner"]["alpha player"] = 0.25
    with pytest.raises(psf.SimulationHealthError, match="not exactly derived"):
        psf._require_strict_live_outright_payload(payload, {}, files)

    payload["outrights"]["winner"]["alpha player"] = 0.0
    payload["outrights_sim_run_at"] = "2026-01-01 00:00:00 UTC"
    with pytest.raises(psf.SimulationHealthError, match="market provenance"):
        psf._require_strict_live_outright_payload(payload, {}, files)


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


def test_strict_r4_threeball_writes_explicit_current_no_groups_contract(
    tmp_path, monkeypatch
):
    event_dir = tmp_path / "test_event"
    event_dir.mkdir()
    pd.DataFrame(
        [[68, 69], [69, 68], [70, 69], [69, 70]],
        index=["alpha player", "beta player", "gamma player", "delta player"],
    ).to_parquet(event_dir / "sim_cache_r4.parquet")
    monkeypatch.setattr(psf, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        psf,
        "_fetch_tee_group_contract",
        lambda *_a, **_k: {
            "status": "no_groups_offered",
            "groups": [],
            "field_names": [
                "alpha player", "beta player", "gamma player", "delta player"
            ],
            "tee_time_names": [
                "alpha player", "beta player", "gamma player", "delta player"
            ],
            "field_players": 4,
            "tee_time_players": 4,
            "group_sizes": [2, 2],
            "round": 4,
            "event_id": "99",
            "tour": "pga",
            "fetched_at": "2026-08-23 12:00:00 UTC",
            "attempts": 1,
        },
    )
    monkeypatch.setattr(
        psf,
        "_cache_meta",
        lambda *_a, **_k: {"health_manifest": {"manifest_sha256": "sim"}},
    )

    def fake_bound(destination, **kwargs):
        Path(destination).write_text(
            json.dumps({"kind": kwargs["kind"], "extra": kwargs["extra"]}),
            encoding="utf-8",
        )

    monkeypatch.setattr(psf, "write_bound_artifact_manifest", fake_bound)
    files = psf.write_round_3ball(
        "test_event", 4, {}, require_contract=True, event_id=99
    )

    assert files == [
        "round_3ball_r4.parquet",
        "round_3ball_r4_meta.json",
        "round_3ball_r4_contract.json",
    ]
    assert pd.read_parquet(tmp_path / files[0]).empty
    meta = json.loads((tmp_path / files[1]).read_text(encoding="utf-8"))
    assert meta["status"] == "no_groups_offered"
    assert meta["event_id"] == "99"
    contract = json.loads((tmp_path / files[2]).read_text(encoding="utf-8"))
    assert contract["extra"]["round"] == 4
    assert contract["extra"]["status"] == "no_groups_offered"


def test_strict_threeball_rejects_eighty_percent_tee_time_coverage(
    tmp_path, monkeypatch
):
    event_dir = tmp_path / "test_event"
    event_dir.mkdir()
    players = ["alpha", "beta", "gamma", "delta", "epsilon"]
    pd.DataFrame([[68, 69]] * len(players), index=players).to_parquet(
        event_dir / "sim_cache_r2.parquet"
    )
    covered = players[:4]
    monkeypatch.setattr(psf, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        psf,
        "_fetch_tee_group_contract",
        lambda *_a, **_k: {
            "status": "no_groups_offered",
            "groups": [],
            "field_names": covered,
            "tee_time_names": covered,
            "field_players": 4,
            "tee_time_players": 4,
            "group_sizes": [2, 2],
            "round": 2,
            "event_id": "99",
        },
    )
    with pytest.raises(RuntimeError, match="active sim player"):
        psf.write_round_3ball(
            "test_event", 2, {}, require_contract=True, event_id=99
        )


def test_tee_group_fetch_retries_and_never_calls_empty_field_no_groups(
    monkeypatch,
):
    import requests

    monkeypatch.setenv("DATAGOLF_API_KEY", "test")
    calls = []

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "event_id": 99,
                "field": [
                    {
                        "player_name": "Alpha Player",
                        "teetimes": [
                            {"round_num": 4, "teetime": "08:00", "start_hole": 1}
                        ],
                    },
                    {
                        "player_name": "Beta Player",
                        "teetimes": [
                            {"round_num": 4, "teetime": "08:00", "start_hole": 1}
                        ],
                    },
                    {
                        "player_name": "Gamma Player",
                        "teetimes": [
                            {"round_num": 4, "teetime": "09:00", "start_hole": 1}
                        ],
                    },
                    {
                        "player_name": "Delta Player",
                        "teetimes": [
                            {"round_num": 4, "teetime": "09:00", "start_hole": 1}
                        ],
                    },
                ],
            }

    def get(*_a, **_k):
        calls.append(1)
        if len(calls) < 3:
            raise requests.ConnectionError("temporary")
        return Response()

    monkeypatch.setattr(requests, "get", get)
    contract = psf._fetch_tee_group_contract(
        4, {}, event_id=99, max_attempts=3, retry_delay_seconds=0
    )
    assert len(calls) == 3
    assert contract["status"] == "no_groups_offered"
    assert contract["attempts"] == 3

    class MissingR2TeeTimes:
        status_code = 200

        @staticmethod
        def json():
            return {
                "event_id": 99,
                "field": [
                    {"player_name": "Alpha Player", "teetimes": []},
                    {"player_name": "Beta Player", "teetimes": []},
                ],
            }

    monkeypatch.setattr(requests, "get", lambda *_a, **_k: MissingR2TeeTimes())
    with pytest.raises(RuntimeError, match="cannot conclude"):
        psf._fetch_tee_group_contract(
            2, {}, event_id=99, max_attempts=2, retry_delay_seconds=0
        )

    monkeypatch.setattr(
        requests,
        "get",
        lambda *_a, **_k: type(
            "EmptyResponse", (), {"status_code": 200, "json": lambda self: {"field": []}}
        )(),
    )
    with pytest.raises(RuntimeError, match="failed after 2 attempts"):
        psf._fetch_tee_group_contract(
            4, {}, event_id=99, max_attempts=2, retry_delay_seconds=0
        )


def test_tee_group_fetch_accepts_missing_or_equivalent_typed_event_id(monkeypatch):
    import requests

    monkeypatch.setenv("DATAGOLF_API_KEY", "test")

    def response(source_event_marker):
        payload = {
            "field": [
                {
                    "player_name": name,
                    "teetimes": [
                        {
                            "round_num": "4",
                            "teetime": tee,
                            "start_hole": 1,
                            "course_num": 1,
                        }
                    ],
                }
                for name, tee in (
                    ("Alpha", "08:00"),
                    ("Beta", "08:00"),
                    ("Gamma", "09:00"),
                    ("Delta", "09:00"),
                )
            ]
        }
        if source_event_marker is not None:
            payload["event_id"] = source_event_marker
        return type(
            "Response",
            (),
            {"status_code": 200, "json": lambda self: payload},
        )()

    for marker, basis in ((None, "pending_field_overlap"), (99.0, "datagolf_event_id")):
        monkeypatch.setattr(requests, "get", lambda *_a, _m=marker, **_k: response(_m))
        contract = psf._fetch_tee_group_contract(
            4, {}, event_id="99", max_attempts=1, retry_delay_seconds=0
        )
        assert contract["requested_event_id"] == "99"
        assert contract["source_event_id"] == (None if marker is None else "99")
        assert contract["event_identity_basis"] == basis


def test_strict_release_manifest_detects_git_file_mutation(tmp_path, monkeypatch):
    monkeypatch.setattr(psf, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        psf, "STRICT_RELEASE_MANIFEST", tmp_path / "sim_release_manifest.json"
    )
    monkeypatch.setattr(
        psf,
        "_git_filtered_blob_bytes",
        lambda relative: (tmp_path / relative).read_bytes(),
    )
    (tmp_path / "sim_fairs.json").write_text("{}", encoding="utf-8")
    data = b"release-bytes"
    prepared = {
        "schema_version": psf.STRICT_RELEASE_SCHEMA,
        "generation": "event-99-r4-sim",
        "generated_at": "2026-08-23 12:00:00 UTC",
        "event_id": "99",
        "tourney": "test_event",
        "round": 4,
        "simulation_manifest_sha256": "sim",
        "live_tournament_manifest_sha256": "live",
        "assets": {
            "tournament_samples_full": {
                "name": "finish.version.parquet",
                "sha256": __import__("hashlib").sha256(data).hexdigest(),
                "size": len(data),
                "data": data,
            }
        },
    }
    manifest = psf._write_strict_release_manifest(
        prepared, files=["sim_fairs.json"]
    )
    psf._require_strict_release_manifest_current(manifest, prepared)
    tampered_binding = json.loads(json.dumps(manifest))
    tampered_binding["release_assets"]["tournament_samples_full"][
        "name"
    ] = "missing-version.parquet"
    tampered_binding = psf.seal_manifest(tampered_binding)
    with pytest.raises(RuntimeError, match="asset bindings changed"):
        psf._require_strict_release_manifest_current(tampered_binding, prepared)
    tampered_core = dict(manifest)
    tampered_core["live_tournament_manifest_sha256"] = "other-live"
    tampered_core = psf.seal_manifest(tampered_core)
    with pytest.raises(RuntimeError, match="core binding changed"):
        psf._require_strict_release_manifest_current(tampered_core, prepared)
    staged = {
        "sim_fairs.json": (tmp_path / "sim_fairs.json").read_bytes(),
        "sim_release_manifest.json": (
            tmp_path / "sim_release_manifest.json"
        ).read_bytes(),
    }
    psf._require_strict_git_blob_snapshot(
        manifest,
        ["sim_fairs.json", "sim_release_manifest.json"],
        staged,
    )
    mutated_staged = dict(staged)
    mutated_staged["sim_fairs.json"] = b'{"changed-during-hash":true}'
    with pytest.raises(RuntimeError, match="staged git blob (size|hash) mismatch"):
        psf._require_strict_git_blob_snapshot(
            manifest,
            ["sim_fairs.json", "sim_release_manifest.json"],
            mutated_staged,
        )
    (tmp_path / "sim_fairs.json").write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed after sealing"):
        psf._require_strict_release_manifest_current(manifest, prepared)


def test_strict_release_generation_is_stable_and_includes_live_health(
    tmp_path, monkeypatch
):
    import pyarrow as pa

    table = pa.Table.from_pandas(
        pd.DataFrame([[270, 271], [271, 270]], index=["alpha", "beta"]),
        preserve_index=True,
    )
    monkeypatch.setattr(psf, "_build_tournament_samples", lambda *_a, **_k: table)
    monkeypatch.setattr(psf, "_build_made_cut_mask", lambda *_a, **_k: table)
    monkeypatch.setattr(psf, "_build_live_matchup_tape", lambda *_a, **_k: table)
    payload = {"event_id": "99", "tourney": "test_event", "round": 4}

    def health(live_id):
        return {
            "manifest_sha256": live_id,
            "generated_at": "2026-08-23 12:00:00 UTC",
            "simulation_manifest": {
                "manifest_sha256": "simulation-id",
                "source": {"generated_at": "2026-08-23 11:59:00 UTC"},
            },
        }

    first = psf._build_strict_release_package(payload, {}, health("live-id-one"))
    retry = psf._build_strict_release_package(payload, {}, health("live-id-one"))
    changed = psf._build_strict_release_package(payload, {}, health("live-id-two"))
    assert first["generation"] == retry["generation"]
    assert first["generated_at"] == retry["generated_at"]
    assert {
        label: (asset["name"], asset["sha256"], asset["size"])
        for label, asset in first["assets"].items()
    } == {
        label: (asset["name"], asset["sha256"], asset["size"])
        for label, asset in retry["assets"].items()
    }
    assert changed["generation"] != first["generation"]

    monkeypatch.setattr(psf, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        psf, "STRICT_RELEASE_MANIFEST", tmp_path / "sim_release_manifest.json"
    )
    monkeypatch.setattr(
        psf,
        "_git_filtered_blob_bytes",
        lambda relative: (tmp_path / relative).read_bytes(),
    )
    (tmp_path / "sim_fairs.json").write_text("{}", encoding="utf-8")
    first_manifest = psf._write_strict_release_manifest(
        first, files=["sim_fairs.json"]
    )
    retry_manifest = psf._write_strict_release_manifest(
        retry, files=["sim_fairs.json"]
    )
    assert first_manifest == retry_manifest


def test_strict_release_manifest_binds_filtered_git_bytes_with_crlf(
    tmp_path, monkeypatch
):
    import hashlib
    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "core.autocrlf", "true"],
        check=True,
    )
    monkeypatch.setattr(psf, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        psf, "STRICT_RELEASE_MANIFEST", tmp_path / "sim_release_manifest.json"
    )
    working_tree_bytes = b'{\r\n  "live": true\r\n}\r\n'
    (tmp_path / "sim_fairs.json").write_bytes(working_tree_bytes)
    release_bytes = b"release-bytes"
    prepared = {
        "schema_version": psf.STRICT_RELEASE_SCHEMA,
        "generation": "event-99-r4-sim",
        "generated_at": "2026-08-23 12:00:00 UTC",
        "event_id": "99",
        "tourney": "test_event",
        "round": 4,
        "simulation_manifest_sha256": "sim",
        "live_tournament_manifest_sha256": "live",
        "assets": {
            "tournament_samples_full": {
                "name": "finish.version.parquet",
                "sha256": hashlib.sha256(release_bytes).hexdigest(),
                "size": len(release_bytes),
                "data": release_bytes,
            }
        },
    }

    manifest = psf._write_strict_release_manifest(
        prepared, files=["sim_fairs.json"]
    )
    binding = manifest["git_files"]["sim_fairs.json"]
    committed_bytes = psf._git_filtered_blob_bytes("sim_fairs.json")

    assert b"\r\n" not in committed_bytes
    assert committed_bytes == working_tree_bytes.replace(b"\r\n", b"\n")
    assert binding["sha256"] == hashlib.sha256(committed_bytes).hexdigest()
    assert binding["size"] == len(committed_bytes)
    psf._require_strict_release_manifest_current(manifest, prepared)

    staged = {
        "sim_fairs.json": committed_bytes,
        "sim_release_manifest.json": psf._git_filtered_blob_bytes(
            "sim_release_manifest.json"
        ),
    }
    psf._require_strict_git_blob_snapshot(
        manifest,
        ["sim_fairs.json", "sim_release_manifest.json"],
        staged,
    )

    (tmp_path / "sim_fairs.json").write_bytes(
        b'{\r\n  "live": false\r\n}\r\n'
    )
    with pytest.raises(RuntimeError, match="changed after sealing"):
        psf._require_strict_release_manifest_current(manifest, prepared)


def test_strict_release_stages_only_versioned_assets(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "test")
    uploaded = []
    monkeypatch.setattr(
        psf,
        "_upload_release_asset",
        lambda name, data, token, **_kwargs: uploaded.append(name),
    )
    prepared = {"assets": {}}
    for label in (
        "tournament_samples_full",
        "tournament_made_cut_full",
        "matchup_scores_live",
    ):
        data = label.encode()
        prepared["assets"][label] = {
            "name": f"{label}.event-99-r4-sim.abc.parquet",
            "sha256": __import__("hashlib").sha256(data).hexdigest(),
            "size": len(data),
            "data": data,
        }
    assert psf._upload_release_tape_family(
        _complete_payload(), {}, strict=True, prepared=prepared
    )
    assert set(uploaded) == {
        asset["name"] for asset in prepared["assets"].values()
    }
    assert psf.FULL_TAPE_ASSET not in uploaded
    assert psf.MADE_CUT_ASSET not in uploaded
    assert psf.MATCHUP_TAPE_ASSET not in uploaded


def test_retry_after_activation_never_deletes_immutable_release_asset(monkeypatch):
    import requests

    data = b"same immutable bytes"
    state = {"assets": [], "uploads": 0, "deletes": 0}

    class Response:
        def __init__(self, status=200, payload=None, content=b""):
            self.status_code = status
            self._payload = payload
            self.content = content

        def json(self):
            return self._payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(str(self.status_code))

    def get(url, **_kwargs):
        if "/releases/tags/" in url:
            return Response(payload={"id": 7, "assets": list(state["assets"])})
        asset = next(item for item in state["assets"] if item["url"] == url)
        return Response(content=asset["content"])

    def post(url, **kwargs):
        state["uploads"] += 1
        name = url.split("name=", 1)[1]
        state["assets"].append(
            {
                "id": 11,
                "name": name,
                "size": len(kwargs["data"]),
                "url": "https://api.github.test/assets/11",
                "content": kwargs["data"],
            }
        )
        return Response(status=201, payload=state["assets"][-1])

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(
        requests,
        "delete",
        lambda *_a, **_k: state.__setitem__("deletes", state["deletes"] + 1),
    )
    name = "matchup_scores_live.event-99-r4-sim.abc.parquet"
    psf._upload_release_asset(name, data, "token", immutable=True)
    psf._upload_release_asset(name, data, "token", immutable=True)

    assert state["uploads"] == 1
    assert state["deletes"] == 0


def test_strict_publisher_rehashes_exact_live_health_family(tmp_path, monkeypatch):
    import hashlib

    tourney = "test_event"
    final_path = tmp_path / f"final_scores_live_{tourney}.npy"
    names_path = tmp_path / f"player_names_live_{tourney}.json"
    mask_path = tmp_path / f"made_cut_live_{tourney}.npy"
    finish_path = tmp_path / "simulated_probs_live.csv"
    event_finish_path = tmp_path / f"top_finish_probs_live_{tourney}.csv"
    np.save(final_path, np.array([[270, 271], [271, 270]]))
    np.save(mask_path, np.ones((2, 2), dtype=bool))
    names_path.write_text(json.dumps(["alpha", "beta"]), encoding="utf-8")
    finish = pd.DataFrame(
        {
            "player_name": ["alpha", "beta"],
            "simulated_win_prob": [0.5, 0.5],
            "top_5": [1.0, 1.0],
            "top_10": [1.0, 1.0],
            "top_20": [1.0, 1.0],
        }
    )
    finish.to_csv(finish_path, index=False)
    finish.to_csv(event_finish_path, index=False)
    bound_paths = {
        "final_scores": final_path,
        "player_names": names_path,
        "made_cut": mask_path,
        "finish_probs": finish_path,
        "finish_probs_event": event_finish_path,
    }
    manifest = {
        "manifest_sha256": "live-id",
        "simulation_manifest": {
            "manifest_sha256": "sim-id",
            "scoring": {"expected_avg": 68.7},
            "model": {"selected": "category_first"},
        },
        "files": {
            label: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for label, path in bound_paths.items()
        },
    }
    (tmp_path / f"tournament_live_{tourney}_health.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    monkeypatch.setattr(psf, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(psf, "collect_overlay_provenance", lambda **_k: {})
    monkeypatch.setattr(
        psf,
        "_cache_meta",
        lambda *_a, **_k: {"health_manifest": {"manifest_sha256": "sim-id"}},
    )
    checked = []

    def require_exact(loaded, *, files, **_kwargs):
        checked.append(set(files))
        for label, path in files.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != loaded["files"][label]["sha256"]:
                raise psf.SimulationHealthError(f"hash mismatch: {label}")

    monkeypatch.setattr(psf, "require_bound_artifact", require_exact)
    monkeypatch.setattr(psf, "require_live_tournament_alignment", lambda **_k: None)
    payload = {"tourney": tourney, "round": 3, "event_id": 99}
    loaded, files = psf._load_and_validate_strict_live_health(payload)
    assert loaded["manifest_sha256"] == "live-id"
    assert checked[-1] == set(bound_paths)
    assert set(files) == set(bound_paths)

    np.save(final_path, np.array([[999, 999], [999, 999]]))
    with pytest.raises(psf.SimulationHealthError, match="final_scores"):
        psf._load_and_validate_strict_live_health(payload)
