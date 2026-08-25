import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import sim_health_gate as shg
import publish_sim_fairs as psf
from portable_hash import LF_NORMALIZED_HASH_MODE, lf_normalized_sha256


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


def _centered_prediction_frame(course_score=68.7):
    players = _players()
    skill = np.tile(np.array([-0.5, 0.5]), len(players) // 2)
    return pd.DataFrame({
        "player_name": players,
        "my_pred3": skill,
        "field_skill_mean": np.zeros(len(players)),
        "weather_sg_r3": np.zeros(len(players)),
        "scores_r3": skill,
        "centering_version": "field_relative_v1",
        "centering_group": "field",
        "course_score_adj": np.full(len(players), course_score),
    })


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


def test_self_consistent_wrong_single_course_target_cannot_approve_itself():
    tape = _tape(69.7)
    manifest = _manifest(
        tape,
        expected_avg=68.7,
        expected_field_mean=69.7,
        expected_player_means={player: 69.7 for player in tape},
        configured_course_averages={"field": 68.7},
    )

    assert manifest["approval"]["status"] == "rejected"
    assert any(
        "not anchored" in error
        for error in manifest["checks"]["errors"]
    )


def test_authoritative_target_rejects_stale_prediction_baseline_and_partial_markers():
    stale = _centered_prediction_frame(course_score=69.7)
    try:
        shg.derive_authoritative_scoring_targets(
            stale,
            sim_round=3,
            skill_col="my_pred3",
            configured_expected_avg=68.7,
            course_averages={"field": 68.7},
        )
    except shg.SimulationHealthError as exc:
        assert "course_score_adj" in str(exc)
    else:
        raise AssertionError("stale 69.7 prediction baseline approved under 68.7 Sheet")

    partial = _centered_prediction_frame()
    partial.loc[0, "centering_version"] = None
    try:
        shg.derive_authoritative_scoring_targets(
            partial,
            sim_round=3,
            skill_col="my_pred3",
            configured_expected_avg=68.7,
            course_averages={"field": 68.7},
        )
    except shg.SimulationHealthError as exc:
        assert "exclusively field_relative_v1" in str(exc)
    else:
        raise AssertionError("partially unmarked prediction artifact was approved")


def test_multicourse_targets_use_exact_sheet_mapping_and_secondary_change_invalidates():
    players = _players()
    courses = np.array(["north"] * 10 + ["south"] * 10)
    skill = np.tile(np.array([-0.5, 0.5]), 10)
    baselines = np.where(courses == "north", 68.0, 70.0)
    frame = pd.DataFrame({
        "player_name": players,
        "course": courses,
        "course_score_adj": baselines,
        "my_pred3": skill,
        "field_skill_mean": np.zeros(20),
        "weather_sg_r3": np.zeros(20),
        "scores_r3": skill,
        "centering_version": "field_relative_v1",
        "centering_group": "course_score_adj",
    })
    target, player_targets = shg.derive_authoritative_scoring_targets(
        frame,
        sim_round=3,
        skill_col="my_pred3",
        configured_expected_avg=68.0,
        course_averages={"north": 68.0, "south": 70.0},
    )
    assert target == 69.0
    assert player_targets[players[0]] == 68.5
    assert player_targets[players[-1]] == 69.5

    tape = {
        player: np.full(10_000, player_targets[player], dtype=float)
        for player in players
    }
    manifest = _manifest(
        tape,
        expected_avg=68.0,
        expected_field_mean=target,
        expected_player_means=player_targets,
        configured_course_averages={"north": 68.0, "south": 70.0},
    )
    report = shg.validate_simulation_manifest(
        manifest,
        tourney="test_event",
        event_id=77,
        sim_round=3,
        configured_expected_avg=68.0,
        configured_course_averages={"north": 68.0, "south": 70.3},
        now=NOW,
        current_overlay=OVERLAY,
    )
    assert manifest["approval"]["status"] == "approved", manifest["checks"]["errors"]
    assert not report.ok
    assert any("per-course" in error for error in report.errors)


def test_course_baseline_config_resolves_each_course_par_and_rejects_incomplete_map():
    resolved = shg.configured_round_scoring_baselines({
        "expected_score_1": -1.3,
        "expected_score_2": -2.0,
        "expected_score_3": None,
        "course_codes": ["N", "S"],
        "course_pars": [70, 72],
    })
    assert resolved == {"n": 68.7, "s": 70.0}

    try:
        shg.configured_round_scoring_baselines({
            "expected_score_1": 68.7,
            "expected_score_2": 70.0,
            "course_codes": ["N"],
            "course_pars": [70, 72],
        })
    except shg.SimulationHealthError as exc:
        assert "course_codes" in str(exc)
    else:
        raise AssertionError("incomplete multi-course Sheet mapping was accepted")


def test_score_est_shift_anchor_comes_from_sealed_manifest_not_mutable_cache_label():
    manifest = _manifest(_tape())
    assert shg.sealed_cache_expected_avg({
        "expected_avg": 68.7,
        "health_manifest": manifest,
    }) == 68.7

    try:
        shg.sealed_cache_expected_avg({
            "expected_avg": 69.0,
            "health_manifest": manifest,
        })
    except shg.SimulationHealthError as exc:
        assert "disagrees" in str(exc)
    else:
        raise AssertionError("mutable cache expected_avg overrode the sealed shift anchor")


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


def test_required_weekly_overlay_mismatch_is_health_unsafe(tmp_path):
    config_path = tmp_path / "shot_dispersion_config.json"
    config_path.write_text(
        json.dumps(
            {
                "enabled": True,
                "required_current_event": True,
                "tourney": "configured_event",
                "event_id": 88,
                "feature_file": "shot_dispersion_features.csv",
                "feature_sha256": "feature",
                "distribution_sha256": "dists",
            }
        ),
        encoding="utf-8",
    )
    provenance = shg.collect_overlay_provenance(
        tourney="test_event",
        event_id=77,
        dists_path=None,
        selected_model="category_first",
        config_path=config_path,
    )

    assert provenance["status"] == "required_event_mismatch"
    assert provenance["required_current_event"] is True
    manifest = shg.build_simulation_manifest(
        _tape(),
        tourney="test_event",
        event_id=77,
        sim_round=3,
        expected_avg=68.7,
        expected_field_mean=68.7,
        model_players=_players(),
        expected_avg_authority="sheet",
        selected_model="category_first",
        skew_calibrated=True,
        overlay=provenance,
        generated_at=NOW,
    )
    assert manifest["approval"]["status"] == "rejected"
    assert any(
        "required_event_mismatch" in error
        for error in manifest["checks"]["errors"]
    )


def test_active_overlay_provenance_uses_lf_normalized_hashes(tmp_path):
    feature_path = tmp_path / "shot_dispersion_features.csv"
    dists_path = tmp_path / "this_week_dists_v2.csv"
    feature_path.write_bytes(b"player_name,value\r\nalpha,1\r\n")
    dists_path.write_bytes(b"player_name,value\r\nalpha,2\r\n")
    feature_hash = lf_normalized_sha256(feature_path)
    dists_hash = lf_normalized_sha256(dists_path)
    config_path = tmp_path / "shot_dispersion_config.json"
    config_path.write_text(
        json.dumps(
            {
                "enabled": True,
                "required_current_event": True,
                "tourney": "test_event",
                "event_id": 77,
                "feature_file": str(feature_path),
                "feature_sha256": feature_hash,
                "distribution_sha256": dists_hash,
            }
        ),
        encoding="utf-8",
    )

    provenance = shg.collect_overlay_provenance(
        tourney="test_event",
        event_id=77,
        dists_path=dists_path,
        selected_model="category_first",
        config_path=config_path,
    )

    assert provenance["status"] == "active"
    assert provenance["feature_sha256"] == feature_hash
    assert provenance["distribution_sha256"] == dists_hash
    assert provenance["text_hash_mode"] == LF_NORMALIZED_HASH_MODE

    manifest = shg.build_simulation_manifest(
        _tape(),
        tourney="test_event",
        event_id=77,
        sim_round=3,
        expected_avg=68.7,
        expected_field_mean=68.7,
        model_players=_players(),
        expected_avg_authority="sheet",
        selected_model="category_first",
        skew_calibrated=True,
        overlay=provenance,
        generated_at=NOW,
    )
    assert manifest["approval"]["status"] == "approved"

    wrong_hash = {**provenance, "configured_feature_sha256": "wrong"}
    rejected = shg.build_simulation_manifest(
        _tape(),
        tourney="test_event",
        event_id=77,
        sim_round=3,
        expected_avg=68.7,
        expected_field_mean=68.7,
        model_players=_players(),
        expected_avg_authority="sheet",
        selected_model="category_first",
        skew_calibrated=True,
        overlay=wrong_hash,
        generated_at=NOW,
    )
    assert rejected["approval"]["status"] == "rejected"
    assert any(
        "feature_sha256 does not match" in error
        for error in rejected["checks"]["errors"]
    )


def test_absent_shot_config_cannot_approve_or_revalidate_a_live_tape():
    absent_overlay = {
        **OVERLAY,
        "status": "config_absent",
        "config_sha256": None,
        "distribution_sha256": None,
    }
    rejected = shg.build_simulation_manifest(
        _tape(),
        tourney="test_event",
        event_id=77,
        sim_round=3,
        expected_avg=68.7,
        expected_field_mean=68.7,
        model_players=_players(),
        expected_avg_authority="sheet",
        selected_model="category_first",
        skew_calibrated=True,
        overlay=absent_overlay,
        generated_at=NOW,
    )
    assert rejected["approval"]["status"] == "rejected"
    assert any(
        "config_absent" in error for error in rejected["checks"]["errors"]
    )

    missing_provenance = shg.build_simulation_manifest(
        _tape(),
        tourney="test_event",
        event_id=77,
        sim_round=3,
        expected_avg=68.7,
        expected_field_mean=68.7,
        model_players=_players(),
        expected_avg_authority="sheet",
        selected_model="category_first",
        skew_calibrated=True,
        generated_at=NOW,
    )
    assert missing_provenance["approval"]["status"] == "rejected"

    formerly_approved = _manifest()
    report = shg.validate_simulation_manifest(
        formerly_approved,
        tourney="test_event",
        event_id=77,
        sim_round=3,
        configured_expected_avg=68.7,
        sim_dict=_tape(),
        model_players=_players(),
        current_overlay=absent_overlay,
        now=NOW + timedelta(minutes=5),
    )
    assert not report.ok
    assert any(
        "current shot-dispersion overlay provenance is unsafe: config_absent"
        in error
        for error in report.errors
    )


def test_explicit_disabled_shot_config_remains_health_safe():
    disabled_overlay = {
        **OVERLAY,
        "status": "disabled_in_config",
        "config_sha256": "disabled-config",
    }
    manifest = shg.build_simulation_manifest(
        _tape(),
        tourney="test_event",
        event_id=77,
        sim_round=3,
        expected_avg=68.7,
        expected_field_mean=68.7,
        model_players=_players(),
        expected_avg_authority="sheet",
        selected_model="category_first",
        skew_calibrated=True,
        overlay=disabled_overlay,
        generated_at=NOW,
    )
    assert manifest["approval"]["status"] == "approved"


def test_structurally_invalid_shot_config_is_unsafe_for_cached_pricing(tmp_path):
    config_path = tmp_path / "shot_dispersion_config.json"
    config_path.write_text('{"enabled": "false"}', encoding="utf-8")
    provenance = shg.collect_overlay_provenance(
        tourney="test_event",
        event_id=77,
        dists_path=None,
        selected_model="category_first",
        config_path=config_path,
    )
    assert provenance["status"] == "invalid_config"

    manifest = shg.build_simulation_manifest(
        _tape(),
        tourney="test_event",
        event_id=77,
        sim_round=3,
        expected_avg=68.7,
        expected_field_mean=68.7,
        model_players=_players(),
        expected_avg_authority="sheet",
        selected_model="category_first",
        skew_calibrated=True,
        overlay=provenance,
        generated_at=NOW,
    )
    assert manifest["approval"]["status"] == "rejected"
    assert any(
        "invalid_config" in error for error in manifest["checks"]["errors"]
    )


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

    main_source = reprice_source[reprice_source.index("def main():"):]
    sheet_read = main_source.index("get_spreadsheet()")
    second_gate = main_source.index(
        "require_pricing_health()",
        main_source.index("# ── 5. Dedup"),
    )
    assert second_gate < sheet_read

    delivery = main_source.index("_deliver_then_store_matchups(")
    final_gate = main_source.rfind("require_pricing_health()", 0, delivery)
    assert sheet_read < final_gate < delivery

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


def test_live_outright_parent_manifest_is_rejected_after_decimal_derivation():
    parent_tape = _tape(68.7)
    parent = _manifest(parent_tape)
    shifted_tape = {
        player: scores.astype(float) + 0.3
        for player, scores in parent_tape.items()
    }
    derived = _manifest(
        shifted_tape,
        expected_avg=69.0,
        expected_field_mean=69.0,
        authority="sheet_score_est",
        parent_manifest=parent,
        generated_at=NOW + timedelta(minutes=1),
        derivation={
            "method": "uniform_rounding_bin_v1",
            "from_expected_avg": 68.7,
            "to_expected_avg": 69.0,
            "delta": 0.3,
        },
    )
    stale_outright = {"simulation_manifest": parent}

    try:
        shg.require_exact_simulation_source(
            stale_outright,
            derived,
            artifact_label="live outright tape",
        )
    except shg.SimulationHealthError as exc:
        assert "exact active simulation manifest" in str(exc)
        assert "rebuild" in str(exc)
    else:
        raise AssertionError("parent outright tape was reused under a derived score-est manifest")

    shg.require_exact_simulation_source(
        {"simulation_manifest": derived},
        derived,
        artifact_label="refreshed live outright tape",
    )

    report = shg.validate_simulation_manifest(
        derived,
        tourney="test_event",
        event_id=77,
        sim_round=3,
        configured_expected_avg=69.0,
        sim_dict=shifted_tape,
        model_players=list(shifted_tape),
        current_overlay=OVERLAY,
        now=NOW + timedelta(minutes=2),
    )
    assert report.ok, report.errors

    bad = json.loads(json.dumps(derived))
    bad["source"]["derivation"]["delta"] = 0.0
    bad = shg.seal_manifest(bad)
    rejected = shg.validate_simulation_manifest(
        bad,
        tourney="test_event",
        event_id=77,
        sim_round=3,
        configured_expected_avg=69.0,
        sim_dict=shifted_tape,
        model_players=list(shifted_tape),
        current_overlay=OVERLAY,
        now=NOW + timedelta(minutes=2),
    )
    assert any("derivation baseline/delta" in error for error in rejected.errors)


def test_derived_generation_refreshes_consumers_without_laundering_root_tape_age():
    tape = _tape(68.7)
    parent_time = NOW - timedelta(hours=17)
    parent = _manifest(tape, generated_at=parent_time)
    derived_time = NOW
    derived = _manifest(tape, generated_at=derived_time, parent_manifest=parent)

    assert derived["source"]["generated_at"] == shg.utc_stamp(derived_time)
    assert derived["source"]["root_generated_at"] == shg.utc_stamp(parent_time)
    report = shg.validate_simulation_manifest(
        derived,
        tourney="test_event",
        event_id=77,
        sim_round=3,
        configured_expected_avg=68.7,
        sim_dict=tape,
        model_players=list(tape),
        current_overlay=OVERLAY,
        now=NOW + timedelta(hours=2),
    )
    assert not report.ok
    assert any("root simulation tape" in error for error in report.errors)


def test_fractional_round_score_pmf_is_content_bound_and_source_aligned(tmp_path):
    tape = _tape(68.7)
    manifest = _manifest(tape)
    rows = []
    for player, draws in tape.items():
        values, counts = np.unique(draws, return_counts=True)
        for score, count in zip(values, counts):
            rows.append({
                "player_name": player,
                "score": int(score),
                "prob": float(count / len(draws)),
            })
    score_df = pd.DataFrame(rows)
    score_path = tmp_path / "round_score_probs_r3.parquet"
    score_df.to_parquet(score_path, index=False)
    health_path = tmp_path / "round_score_probs_r3_health.json"
    health = shg.write_bound_artifact_manifest(
        health_path,
        kind="round_score_pmf",
        simulation_manifest=manifest,
        files={"score_pmf": score_path},
        extra={"reprice_method": "uniform_rounding_bin_v1"},
    )

    assert not shg.validate_round_score_probability_table(score_df, manifest)
    report = shg.validate_bound_artifact(
        health,
        kind="round_score_pmf",
        files={"score_pmf": score_path},
        tourney="test_event",
        event_id=77,
        sim_round=3,
        configured_expected_avg=68.7,
        current_overlay=OVERLAY,
        now=NOW + timedelta(minutes=5),
    )
    assert report.ok, report.errors

    tampered = score_df.copy()
    tampered.loc[tampered.index[0], "prob"] += 0.01
    assert any(
        "sum to one" in error
        for error in shg.validate_round_score_probability_table(tampered, manifest)
    )


def test_round_score_publisher_rejects_pmf_from_parent_cache(tmp_path, monkeypatch):
    tape = _tape(68.7)
    parent = _manifest(tape)
    rows = []
    for player, draws in tape.items():
        values, counts = np.unique(draws, return_counts=True)
        rows.extend({
            "player_name": player,
            "score": int(score),
            "prob": float(count / len(draws)),
        } for score, count in zip(values, counts))
    score_df = pd.DataFrame(rows)
    score_path = tmp_path / "round_score_probs_r3.parquet"
    score_df.to_parquet(score_path, index=False)
    shg.write_bound_artifact_manifest(
        tmp_path / "round_score_probs_r3_health.json",
        kind="round_score_pmf",
        simulation_manifest=parent,
        files={"score_pmf": score_path},
        extra={"reprice_method": "uniform_rounding_bin_v1"},
    )
    monkeypatch.setattr(psf, "_find_fresh", lambda *_args: score_path)
    monkeypatch.setattr(psf, "collect_overlay_provenance", lambda **_kwargs: OVERLAY)
    monkeypatch.setattr(psf, "_cache_meta", lambda *_args: {"health_manifest": parent})
    monkeypatch.setattr(shg, "utc_now", lambda: NOW)

    published = psf._build_round_scores("test_event", 3, {})
    assert len(published) == len(tape)

    derived = _manifest(tape, parent_manifest=parent, generated_at=NOW + timedelta(minutes=1))
    monkeypatch.setattr(psf, "_cache_meta", lambda *_args: {"health_manifest": derived})
    try:
        psf._build_round_scores("test_event", 3, {})
    except shg.SimulationHealthError as exc:
        assert "exact active simulation manifest" in str(exc)
    else:
        raise AssertionError("parent PMF was published beside a derived round cache")
