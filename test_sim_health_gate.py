import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import sim_health_gate as shg
import publish_sim_fairs as psf


NOW = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)
OVERLAY = {
    "status": "not_applicable_to_event",
    "configured_for_active_event": False,
    "used_by_selected_tape": False,
    "config_sha256": "cfg",
    "feature_sha256": None,
    "distribution_sha256": "dists",
    "configured_feature_sha256": None,
    "configured_distribution_sha256": None,
}


def _players(n=20):
    return [f"player {i:02d}" for i in range(n)]


def _tape(mean=68.7, n=20):
    # Exact one-decimal empirical mean with 10,000 integer draws per player.
    lo = int(np.floor(mean))
    high = lo + 1
    high_count = int(round((mean - lo) * 10_000))
    values = np.concatenate([
        np.full(10_000 - high_count, lo, dtype=np.int16),
        np.full(high_count, high, dtype=np.int16),
    ])
    return {player: np.roll(values, i) for i, player in enumerate(_players(n))}


def _manifest(tape=None, **kwargs):
    tape = tape or _tape()
    return shg.build_simulation_manifest(
        tape,
        tourney="test_event",
        event_id=77,
        sim_round=3,
        expected_avg=kwargs.pop("expected_avg", 68.7),
        expected_field_mean=kwargs.pop("expected_field_mean", 68.7),
        model_players=list(tape),
        expected_avg_authority=kwargs.pop("authority", "sheet"),
        selected_model="category_first",
        skew_calibrated=True,
        overlay=OVERLAY,
        generated_at=kwargs.pop("generated_at", NOW),
        **kwargs,
    )


def _validate(manifest, tape=None, **kwargs):
    tape = tape or _tape()
    return shg.validate_simulation_manifest(
        manifest,
        tourney="test_event",
        event_id=77,
        sim_round=3,
        configured_expected_avg=68.7,
        sim_dict=tape,
        model_players=list(tape),
        current_overlay=OVERLAY,
        now=NOW + timedelta(minutes=5),
        **kwargs,
    )


def _h2h(players):
    rows = []
    for i, a in enumerate(players):
        for b in players[i + 1:]:
            rows.append({"player_a": a, "player_b": b, "p_a_lt_b": 0.0, "p_tie": 1.0})
    return pd.DataFrame(rows)


def test_healthy_centered_tape_is_content_approved():
    tape = _tape()
    manifest = _manifest(tape)
    report = _validate(manifest, tape)

    assert report.ok, report.errors
    assert manifest["approval"]["status"] == "approved"
    assert manifest["scoring"]["empirical_field_mean"] == 68.7
    assert manifest["simulation"]["num_players"] == 20
    assert manifest["simulation"]["num_sims"] == 10_000


def test_mislabeled_tape_three_tenths_off_is_rejected():
    tape = _tape(69.0)
    manifest = _manifest(tape, expected_avg=68.7, expected_field_mean=68.7)
    report = _validate(manifest, tape)

    assert manifest["approval"]["status"] == "rejected"
    assert not report.ok
    assert any("not centered" in error or "miscentered" in error for error in report.errors)


def test_truncated_live_field_is_rejected():
    tape = _tape(68.7, n=19)
    manifest = _manifest(tape)
    report = _validate(manifest, tape)

    assert manifest["approval"]["status"] == "rejected"
    assert not report.ok
    assert any("field contains only 19" in error for error in manifest["checks"]["errors"])


def test_cli_manifest_requires_named_manual_approver():
    tape = _tape()
    rejected = _manifest(tape, authority="cli")
    approved = _manifest(tape, authority="cli", manual_approved_by="github:operator")

    assert rejected["approval"]["status"] == "rejected"
    assert approved["approval"] == {
        "status": "approved",
        "mode": "manual_cli",
        "approved_by": "github:operator",
        "approved_at": shg.utc_stamp(NOW),
    }


def test_event_scoped_overlay_hashes_do_not_block_a_different_week():
    tape = _tape()
    prior_event_overlay = {
        **OVERLAY,
        "configured_feature_sha256": "last-week-feature",
        "configured_distribution_sha256": "last-week-dists",
        "feature_sha256": "current-feature-file",
        "distribution_sha256": "current-dists-file",
    }
    manifest = shg.build_simulation_manifest(
        tape,
        tourney="test_event",
        event_id=77,
        sim_round=3,
        expected_avg=68.7,
        expected_field_mean=68.7,
        model_players=list(tape),
        expected_avg_authority="sheet",
        selected_model="category_first",
        skew_calibrated=True,
        overlay=prior_event_overlay,
        generated_at=NOW,
    )
    assert manifest["approval"]["status"] == "approved"


def test_stale_or_wrong_event_manifest_fails_closed():
    tape = _tape()
    stale = _manifest(tape, generated_at=NOW - timedelta(hours=19))
    stale_report = _validate(stale, tape)
    wrong_event = shg.validate_simulation_manifest(
        _manifest(tape),
        tourney="another_event",
        event_id=99,
        sim_round=4,
        configured_expected_avg=68.7,
        now=NOW,
    )

    assert not stale_report.ok
    assert any("19.1h old" in error for error in stale_report.errors)
    assert not wrong_event.ok
    assert any("tourney" in error for error in wrong_event.errors)
    assert any("event" in error for error in wrong_event.errors)
    assert any("round" in error for error in wrong_event.errors)


def test_bound_h2h_detects_exact_file_tampering(tmp_path):
    tape = _tape()
    sim_manifest = _manifest(tape)
    h2h = _h2h(list(tape))
    parquet = tmp_path / "round_h2h_r3.parquet"
    meta = tmp_path / "round_h2h_r3_meta.json"
    health_path = tmp_path / "round_h2h_r3_health.json"
    h2h.to_parquet(parquet, index=False)
    meta.write_text(
        json.dumps({"source_manifest_sha256": sim_manifest["manifest_sha256"]}),
        encoding="utf-8",
    )
    health = shg.write_bound_artifact_manifest(
        health_path,
        kind="published_round_h2h",
        simulation_manifest=sim_manifest,
        files={"h2h_parquet": parquet, "h2h_meta": meta},
    )

    clean = shg.validate_bound_artifact(
        health,
        kind="published_round_h2h",
        files={"h2h_parquet": parquet, "h2h_meta": meta},
        tourney="test_event",
        event_id=77,
        sim_round=3,
        configured_expected_avg=68.7,
        current_overlay=OVERLAY,
        now=NOW,
    )
    assert clean.ok, clean.errors
    assert not shg.validate_h2h_probability_table(h2h, sim_manifest)

    with parquet.open("ab") as handle:
        handle.write(b"tampered")
    tampered = shg.validate_bound_artifact(
        health,
        kind="published_round_h2h",
        files={"h2h_parquet": parquet, "h2h_meta": meta},
        tourney="test_event",
        event_id=77,
        sim_round=3,
        configured_expected_avg=68.7,
        current_overlay=OVERLAY,
        now=NOW,
    )
    assert not tampered.ok
    assert any("exact h2h_parquet file hash" in error for error in tampered.errors)


def test_h2h_probability_mass_and_pair_coverage_fail_closed():
    tape = _tape()
    manifest = _manifest(tape)
    h2h = _h2h(list(tape)).iloc[:-1].copy()
    h2h.loc[h2h.index[0], ["p_a_lt_b", "p_tie"]] = [0.8, 0.4]
    errors = shg.validate_h2h_probability_table(h2h, manifest)

    assert any("pair coverage" in error for error in errors)
    assert any("probability mass" in error for error in errors)


def test_live_outright_tape_requires_aligned_names_draws_and_probability_mass(tmp_path):
    players = _players()
    final_path = tmp_path / "final.npy"
    names_path = tmp_path / "names.json"
    cut_path = tmp_path / "cut.npy"
    np.save(final_path, np.full((20, 100), 280, dtype=np.int16))
    np.save(cut_path, np.ones((20, 100), dtype=bool))
    names_path.write_text(json.dumps(players), encoding="utf-8")
    finish = pd.DataFrame({
        "player_name": players,
        "simulated_win_prob": np.full(20, 0.05),
        "top_5": np.full(20, 0.25),
        "top_10": np.full(20, 0.5),
        "top_20": np.ones(20),
    })
    artifact = {
        "extra": {
            "num_players": 20,
            "num_sims": 100,
            "field_player_set_sha256": shg.names_sha256(players),
        }
    }

    assert not shg.validate_live_tournament_alignment(
        final_scores_path=final_path,
        player_names_path=names_path,
        made_cut_path=cut_path,
        finish_probs=finish,
        artifact_manifest=artifact,
    )
    np.save(cut_path, np.ones((19, 100), dtype=bool))
    errors = shg.validate_live_tournament_alignment(
        final_scores_path=final_path,
        player_names_path=names_path,
        made_cut_path=cut_path,
        finish_probs=finish,
        artifact_manifest=artifact,
    )
    assert any("made-cut shape" in error for error in errors)


def test_health_gate_precedes_every_betting_side_effect_in_entrypoints():
    root = Path(__file__).resolve().parent
    round_source = (root / "round_sim.py").read_text(encoding="utf-8")
    reprice_source = (root / "reprice.py").read_text(encoding="utf-8")

    reprice_block = round_source[round_source.index("if args.reprice:"):]
    assert reprice_block.index("_require_betting_health()") < reprice_block.index(
        "_reprice_store_and_alert("
    )
    email_block = round_source[round_source.index("# ── Step 6: Email"):]
    assert email_block.index("_require_betting_health()") < email_block.index(
        "send_round_sim_email("
    )
    storage_block = round_source[round_source.index("# ── Storage"):]
    assert storage_block.index("_require_betting_health()") < storage_block.index(
        "store_round_matchups("
    )

    side_effect = min(
        reprice_source.index("get_spreadsheet()"),
        reprice_source.index("store_round_matchups("),
        reprice_source.index("rc.send_matchup_alert("),
    )
    second_gate = reprice_source.index(
        "require_pricing_health()",
        reprice_source.index("# ── 5. Dedup"),
    )
    assert second_gate < side_effect

    workflow = (root / ".github" / "workflows" / "run-sim.yml").read_text(
        encoding="utf-8"
    )
    assert '"--health-approved-by", "github:${{ github.actor }}"' in workflow


def test_round_h2h_publisher_emits_exact_three_file_health_bundle(tmp_path, monkeypatch):
    tape = _tape()
    manifest = _manifest(tape, generated_at=shg.utc_now())
    cache = pd.DataFrame.from_dict(tape, orient="index")
    cache.index.name = "player_name"
    cache_path = tmp_path / "sim_cache_r3.parquet"
    cache.to_parquet(cache_path)
    cache_meta = {
        "health_manifest": manifest,
        "pred_lookup": {player: 1.0 for player in tape},
        "wx_lookup": {player: 0.0 for player in tape},
    }

    monkeypatch.setattr(psf, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(psf, "_find_fresh", lambda *_args: cache_path)
    monkeypatch.setattr(psf, "_cache_meta", lambda *_args: cache_meta)
    monkeypatch.setattr(psf, "_sim_inputs", lambda: type("SI", (), {"event_ids": [77]})())
    monkeypatch.setattr(psf, "_sample_lookup", lambda *_args: {})
    monkeypatch.setattr(psf, "collect_overlay_provenance", lambda **_kwargs: OVERLAY)

    files = psf.write_round_h2h("test_event", 3, {})

    assert files == [
        "round_h2h_r3.parquet",
        "round_h2h_r3_meta.json",
        "round_h2h_r3_health.json",
    ]
    meta = json.loads((tmp_path / files[1]).read_text(encoding="utf-8"))
    health = json.loads((tmp_path / files[2]).read_text(encoding="utf-8"))
    assert meta["source_manifest_sha256"] == manifest["manifest_sha256"]
    assert meta["sim_run_at"] == manifest["source"]["generated_at"]
    report = shg.validate_bound_artifact(
        health,
        kind="published_round_h2h",
        files={
            "h2h_parquet": tmp_path / files[0],
            "h2h_meta": tmp_path / files[1],
        },
        tourney="test_event",
        event_id=77,
        sim_round=3,
        configured_expected_avg=68.7,
        current_overlay=OVERLAY,
    )
    assert report.ok, report.errors


def test_round_h2h_publisher_rejects_unmanifested_cache(tmp_path, monkeypatch):
    cache_path = tmp_path / "sim_cache_r3.parquet"
    pd.DataFrame.from_dict(_tape(), orient="index").to_parquet(cache_path)
    monkeypatch.setattr(psf, "_find_fresh", lambda *_args: cache_path)
    monkeypatch.setattr(psf, "_cache_meta", lambda *_args: {})

    try:
        psf._build_round_h2h("test_event", 3, {})
    except shg.SimulationHealthError as exc:
        assert "no simulation health manifest" in str(exc)
    else:
        raise AssertionError("unmanifested cache was published")


def test_rejected_simulation_cannot_be_wrapped_in_a_bound_manifest(tmp_path):
    rejected = _manifest(_tape(69.0), expected_avg=68.7, expected_field_mean=68.7)
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"data")

    try:
        shg.write_bound_artifact_manifest(
            tmp_path / "health.json",
            kind="published_round_h2h",
            simulation_manifest=rejected,
            files={"h2h_parquet": artifact},
        )
    except shg.SimulationHealthError as exc:
        assert "unapproved or invalid" in str(exc)
    else:
        raise AssertionError("rejected simulation was rebound as approved")
