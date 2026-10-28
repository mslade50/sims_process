from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from archive import hole_sim_streaming_transition_ledger as ledger
from archive.hole_sim_streaming_transition_ledger import (
    EXPLICIT_APP_HOLEOUT_VERSION,
    build_explicit_app_support,
)


def _fixture(monkeypatch, *, probability: np.ndarray | None = None):
    spatial_ids = np.asarray([f"spatial-{index:02d}" for index in range(12)])
    terminal_ids = np.asarray(["holed-a", "holed-b"])
    rows = pd.DataFrame(
        {
            "shot_key": np.concatenate([spatial_ids, terminal_ids]),
            "spatial_valid": np.concatenate(
                [np.ones(12, dtype=bool), np.zeros(2, dtype=bool)]
            ),
            "is_holeout": np.concatenate(
                [np.zeros(12, dtype=bool), np.ones(2, dtype=bool)]
            ),
            "event_key": "pga:R2025034",
            "source_player_id": [str(100 + index) for index in range(14)],
        }
    )
    # The current branch contract conditionally normalizes the 80% spatial
    # mass and retains the 20% terminal mass as a scalar.
    weights = np.concatenate(
        [np.full(12, 0.8 / 12.0), np.asarray([0.05, 0.15])]
    )
    distribution = ledger.app.AppDistribution(
        rows=rows,
        weights=weights,
        source="synthetic",
        n_setup=12,
        n_hole=12,
        n_lie=12,
    )
    monkeypatch.setattr(
        ledger.app, "build_distribution", lambda *args, **kwargs: distribution
    )
    if probability is None:
        probability = np.full(12, 1.0 / 12.0)
    support = SimpleNamespace(
        key="app|normalized|R1|H6|fairway|B12|33204",
        atom_ids=spatial_ids,
        probability=np.asarray(probability, float),
        holeout_probability=0.2,
    )
    provider = SimpleNamespace(atoms=SimpleNamespace(raw_course_app=pd.DataFrame()))
    return provider, support, spatial_ids


def test_explicit_app_support_preserves_scalar_mass_and_legacy_holeout_flags(
    monkeypatch,
) -> None:
    normalized_probability = np.arange(1.0, 13.0)
    normalized_probability /= normalized_probability.sum()
    provider, support, spatial_ids = _fixture(
        monkeypatch, probability=normalized_probability
    )
    explicit = build_explicit_app_support(
        provider=provider,
        spatial_support=support,
        round_no=1,
        hole_no=6,
        start_lie="fairway",
        distance_bucket=12,
        player_id="33204",
    )

    assert explicit.rows["atom_id"].is_unique
    assert len(explicit.rows) == 14
    assert explicit.rows["holeout_representation"].eq(
        EXPLICIT_APP_HOLEOUT_VERSION
    ).all()
    assert np.isclose(explicit.probability.sum(), 1.0, atol=0.0, rtol=0.0)
    assert np.isclose(
        explicit.rows.loc[explicit.terminal_mask, "quality_base_mass"].sum(),
        0.2,
        atol=2e-14,
        rtol=0.0,
    )
    assert np.allclose(
        explicit.rows["quality_base_mass"],
        explicit.rows["base_hierarchy_mass"]
        * explicit.rows["spatial_model_weight"],
        rtol=1e-12,
        atol=2e-14,
    )

    # p=.2. Values strictly below p stay exactly the legacy hole-out set. The
    # residual uniform chooses holed-a for .01 and holed-b for .19.
    uniforms = np.asarray([0.01, 0.19, 0.20, np.nextafter(1.0, 0.0)])
    selected_spatial = spatial_ids[[0, 1, 2, 11]]
    selected = explicit.select_preserving_legacy_streams(
        spatial_atom_ids=selected_spatial,
        holeout_uniform=uniforms,
    )
    assert selected["legacy_holeout_decision"].tolist() == [True, True, False, False]
    assert selected["terminal"].tolist() == [True, True, False, False]
    assert selected["atom_id"].tolist() == [
        "holed-a",
        "holed-b",
        "spatial-02",
        "spatial-11",
    ]
    assert selected["selected_spatial_proposal_atom_id"].tolist() == list(
        selected_spatial
    )
    assert selected.loc[0, "holeout_selection_residual_uniform"] == pytest.approx(0.05)
    assert selected.loc[1, "holeout_selection_residual_uniform"] == pytest.approx(0.95)
    assert selected.loc[2:, "holeout_selection_residual_uniform"].isna().all()


def test_explicit_app_support_rejects_local_ids_or_scalar_mass_drift(monkeypatch) -> None:
    provider, support, _ = _fixture(monkeypatch)
    support.atom_ids = np.arange(12, dtype=np.int32)
    with pytest.raises(ValueError, match="physical shot IDs"):
        build_explicit_app_support(
            provider=provider,
            spatial_support=support,
            round_no=1,
            hole_no=6,
            start_lie="fairway",
            distance_bucket=12,
            player_id="33204",
        )

    provider, support, _ = _fixture(monkeypatch)
    support.holeout_probability = 0.21
    with pytest.raises(AssertionError, match="holeout probability changed"):
        build_explicit_app_support(
            provider=provider,
            spatial_support=support,
            round_no=1,
            hole_no=6,
            start_lie="fairway",
            distance_bucket=12,
            player_id="33204",
        )


def test_proposal_support_contract_persists_every_atom_and_terminal_mass(
    monkeypatch,
) -> None:
    provider, support, _ = _fixture(monkeypatch)
    explicit = build_explicit_app_support(
        provider=provider,
        spatial_support=support,
        round_no=1,
        hole_no=6,
        start_lie="fairway",
        distance_bucket=12,
        player_id="33204",
    )
    tee_key = "tee|normalized|R1|H6|P4|33204"
    tee = pd.DataFrame(
        {
            "support_key": tee_key,
            "atom_id": ["tee-a", "tee-b"],
            "shot_key": ["tee-a", "tee-b"],
            "category": "ott",
            "atom_kind": "tee_spatial",
            "terminal": False,
            "holed": False,
            "event_key": "pga:R2025034",
            "source_player_id": ["1", "2"],
            "base_hierarchy_mass": [0.4, 0.6],
            "spatial_model_weight": [1.0, 1.0],
            "quality_base_mass": [0.4, 0.6],
            "holeout_representation": "not_applicable",
            "declared_app_base_holeout_probability": 0.0,
            "implicit_holeout_probability": 0.0,
        }
    )
    app_key = explicit.key
    repository = SimpleNamespace(
        rich_supports={tee_key: tee, app_key: explicit.rows},
        supports={
            tee_key: SimpleNamespace(
                cache_sha256="a" * 64, payload_sha256="b" * 64
            ),
            app_key: SimpleNamespace(
                cache_sha256="c" * 64, payload_sha256="d" * 64
            ),
        },
    )
    atoms, inventory = ledger.materialize_proposal_support_contract(
        {"normalized": repository},
        target_event_key="pga:R2026034",
        target_event_start="2026-06-25",
        target_player_id="33204",
    )
    assert len(atoms) == 16
    assert len(inventory) == 2
    app_atoms = atoms.loc[atoms["category"].eq("app")]
    assert app_atoms["terminal"].sum() == 2
    assert app_atoms.loc[app_atoms["terminal"], "quality_base_mass"].sum() == pytest.approx(
        0.2
    )
    assert inventory["quality_mass"].to_numpy().tolist() == pytest.approx([1.0, 1.0])
    assert atoms.groupby(["arm", "support_key"])[
        "proposal_support_measure_sha256"
    ].nunique().eq(1).all()
    atoms["donor_event_start"] = pd.Timestamp("2025-06-20", tz="UTC")
    atoms["donor_skill_date"] = pd.Timestamp("2025-06-19", tz="UTC")
    atoms["donor_ability_sg"] = 0.1
    atoms["donor_ability_available"] = True
    atoms["donor_ability_snapshot_kind"] = ledger.chain_adapter.DONOR_SNAPSHOT_CONTRACT
    atoms, inventory, seal = ledger.seal_proposal_support_ability_hashes(
        atoms, inventory
    )
    audit = ledger.validate_persisted_proposal_support_contract(atoms, inventory)
    assert audit["supports"] == 2
    assert audit["atoms"] == 16
    assert seal["missing_rows"] == 0

    tampered = atoms.copy()
    selected = tampered["category"].eq("app") & tampered["terminal"]
    tampered.loc[selected.idxmax(), "quality_base_mass"] += 0.01
    with pytest.raises(AssertionError, match="mechanics hash changed"):
        ledger.validate_persisted_proposal_support_contract(tampered, inventory)


def test_verified_parquet_hash_is_the_persisted_reader_hash(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "identity": ["a", "b"],
            "nullable_text_object": ["query", np.nan],
            "nullable_object": [True, None],
            "when": pd.to_datetime(["2025-01-01", None], utc=True),
            "value": [1.25, np.nan],
        }
    )
    path = tmp_path / "roundtrip.parquet"
    persisted, logical_sha = ledger._write_verified_parquet(frame, path)
    assert logical_sha == ledger.streaming.frame_logical_sha256(pd.read_parquet(path))
    assert logical_sha == ledger.streaming.frame_logical_sha256(persisted)


def test_rich_production_target_validation_rejects_unsealed_ability_mutation(
    tmp_path,
) -> None:
    source = ledger.DEFAULT_TARGET_DIR
    targets, validated, audit = ledger.load_validated_production_targets(
        source,
        expected_event_key="pga:R2026034",
        expected_event_start="2026-06-25",
    )
    assert len(validated) == len(targets) == 288
    assert audit["rich_row_hashes_verified"]

    output = tmp_path / "targets"
    output.mkdir()
    target_path = output / "production_anchor_inputs.parquet"
    manifest_path = output / "manifest.json"
    tampered = targets.copy()
    tampered.loc[tampered.index[0], "ability_ott_sg"] = 999.0
    tampered.to_parquet(target_path, index=False)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    manifest["artifact_sha256"][target_path.name] = ledger.streaming.sha256_file(
        target_path
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="rich production target row hash"):
        ledger.load_validated_production_targets(
            output,
            expected_event_key="pga:R2026034",
            expected_event_start="2026-06-25",
        )


def test_counter_trace_replays_initial_continuation_and_deferred_hazard_indices() -> None:
    event_key = "pga:R2026034"
    player_id = "33204"
    seed = ledger.streaming.DEFAULT_SEED
    continuation_seed = ledger.streaming.DEFAULT_CONTINUATION_SEED

    def uniform(*, particle: int, index: int, stream: str) -> float:
        initial = index == -1
        return float(
            ledger.streaming.hook.counter_uniforms(
                seed=seed if initial else continuation_seed,
                event_key=event_key,
                player_id=player_id,
                round_no=1,
                hole_no=6,
                transition_index=index,
                stream=f"initial_{stream}" if initial else stream,
                particle_ids=np.asarray([particle], dtype=np.int64),
            )[0]
        )

    rows = []
    for particle, category, index in (
        (1, "ott", -1),
        (2, "app", 2),
        (3, "arg", 3),
        (4, "putt", 4),
    ):
        row = {
            "target_event_key": event_key,
            "target_player_id": player_id,
            "round_no": 1,
            "hole_no": 6,
            "particle_id": particle,
            "category": category,
            "hazard_resolution_status": "none",
            "rng_shot_transition_index": index,
            "rng_hazard_transition_index": np.nan,
            "rng_index_uniform": np.nan,
            "rng_lie_uniform": np.nan,
            "rng_holeout_uniform": np.nan,
            "rng_holeout_residual_uniform": np.nan,
            "rng_population_uniform": np.nan,
            "rng_hazard_uniform": np.nan,
            "legacy_holeout_decision": False,
            "declared_app_base_holeout_probability": 0.0,
        }
        if category in {"ott", "app"}:
            prefix = "tee" if category == "ott" else "app"
            row["rng_index_uniform"] = uniform(
                particle=particle, index=index, stream=f"{prefix}_index"
            )
            row["rng_lie_uniform"] = uniform(
                particle=particle, index=index, stream=f"{prefix}_lie"
            )
        if category == "app":
            holeout = uniform(particle=particle, index=index, stream="app_holeout")
            declared = (holeout + 1.0) / 2.0
            row["rng_holeout_uniform"] = holeout
            row["rng_holeout_residual_uniform"] = holeout / declared
            row["legacy_holeout_decision"] = True
            row["declared_app_base_holeout_probability"] = declared
            # The causal APP shot was selected at index 2 and its deferred
            # hazard was resolved separately at continuation index 3.
            row["rng_hazard_transition_index"] = 3
            row["rng_hazard_uniform"] = uniform(
                particle=particle, index=3, stream="hazard"
            )
            row["hazard_resolution_status"] = "resolved_complete_atom"
        if category in {"arg", "putt"}:
            row["rng_population_uniform"] = uniform(
                particle=particle, index=index, stream=f"state_{category}"
            )
        rows.append(row)

    frame = pd.DataFrame(rows)
    audit = ledger.validate_counter_rng_trace(
        frame,
        seed=seed,
        continuation_seed=continuation_seed,
    )
    assert audit["all_exact"]
    assert audit["uniforms_replayed"] == 8
    assert audit["max_abs_uniform_residual"] == 0.0

    tampered = frame.copy()
    tampered.loc[tampered["category"].eq("app"), "rng_hazard_uniform"] = 0.0
    with pytest.raises(AssertionError, match="Counter RNG replay changed"):
        ledger.validate_counter_rng_trace(
            tampered,
            seed=seed,
            continuation_seed=continuation_seed,
        )

    nonintegral_hazard = frame.copy()
    nonintegral_hazard.loc[
        nonintegral_hazard["category"].eq("app"), "rng_hazard_transition_index"
    ] = 3.5
    with pytest.raises(AssertionError, match="nonintegral"):
        ledger.validate_counter_rng_trace(
            nonintegral_hazard,
            seed=seed,
            continuation_seed=continuation_seed,
        )

    wrong_causal_relation = frame.copy()
    app_mask = wrong_causal_relation["category"].eq("app")
    wrong_causal_relation.loc[app_mask, "rng_hazard_transition_index"] = 4
    wrong_causal_relation.loc[app_mask, "rng_hazard_uniform"] = uniform(
        particle=2, index=4, stream="hazard"
    )
    with pytest.raises(AssertionError, match="one step after its causal shot"):
        ledger.validate_counter_rng_trace(
            wrong_causal_relation,
            seed=seed,
            continuation_seed=continuation_seed,
        )

    missing_uniform = frame.copy()
    missing_uniform.loc[
        missing_uniform["category"].eq("app"), "rng_holeout_uniform"
    ] = np.nan
    with pytest.raises(AssertionError, match="holeout uniform null semantics"):
        ledger.validate_counter_rng_trace(
            missing_uniform,
            seed=seed,
            continuation_seed=continuation_seed,
        )

    unpaired_hazard = frame.copy()
    unpaired_hazard.loc[
        unpaired_hazard["category"].eq("app"), "rng_hazard_transition_index"
    ] = np.nan
    with pytest.raises(AssertionError, match="Hazard uniform/index null semantics"):
        ledger.validate_counter_rng_trace(
            unpaired_hazard,
            seed=seed,
            continuation_seed=continuation_seed,
        )
