from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from archive import evaluate_frozen_base_skill_sealed_matchups as sealed


def _frozen_selection(candidate: str = "complete__ridge__total__w0__quarterly__a10") -> dict:
    mapper = []
    for name in (candidate, sealed.STRICT_PT_BASELINE):
        for market, sigma, tie in (("72h", 9.0, 0.6), ("r1", 4.5, 0.2)):
            mapper.append(
                {
                    "candidate": name,
                    "market": market,
                    "fit_end": "2022-12-31",
                    "sigma": sigma,
                    "tie_threshold": tie,
                    "frozen_before_sealed_scoring": True,
                }
            )
    return {
        "contract": sealed.SELECTION_CONTRACT,
        "complete": True,
        "selection_window_end": "2022-12-31",
        "selection_uses_sealed": False,
        "sealed_data_used_for_selection": False,
        "uses_closing_odds_as_feature": False,
        "frozen_candidate_catalog_sha256": (
            sealed.discovery.EXPECTED_FROZEN_CATALOG_SHA256
        ),
        "frozen_selection_parameter_contract": (
            sealed.discovery.validate_frozen_selection_cli(
                sealed.discovery.build_parser().parse_args([])
            )
        ),
        "finalists": [candidate],
        "promotion_gate_passed": True,
        "locked_diagnostic_ablation_set": {
            "selected_candidate": candidate,
            "locked_before_sealed_scoring": True,
            "may_reselect_or_promote_from_sealed": False,
            "included_candidates": [candidate],
        },
        "final_frozen_probability_parameters": mapper,
        "opener_blend_sensitivity": {
            "neutral_base_redefined": False,
            "asynchronous_openers_not_execution_proof": True,
            "formula": "logit(p_blend)=(1-w)*logit(p_model)+w*logit(p_open_novig)",
            "choices": [
                {
                    "candidate": candidate,
                    "selected_opener_weight": 0.5,
                    "selection_window_end": "2022-12-31",
                    "sealed_data_used_for_selection": False,
                    "neutral_base_redefined": False,
                    "same_book_open_close_required": True,
                }
            ],
        },
        "artifacts": [],
    }


def test_frozen_handoff_delegates_to_shared_chain_and_preserves_cutoff_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = "complete__ridge__total__w0__quarterly__a10"
    selection_path = tmp_path / "selection_manifest.json"
    selection_path.write_text(
        json.dumps(_frozen_selection(candidate)), encoding="utf-8"
    )
    prediction_path = tmp_path / "player_event_predictions.parquet"
    prediction_path.write_bytes(b"synthetic compact handoff")
    compact = {"selected_model_name": candidate}
    compact_path = tmp_path / "manifest.json"
    compact_path.write_text(json.dumps(compact), encoding="utf-8")
    handoff = SimpleNamespace(
        compact=compact,
        selection=_frozen_selection(candidate),
        compact_prediction_path=prediction_path.resolve(),
        selection_manifest_path=selection_path.resolve(),
        selected_name=candidate,
        verified_paths=(compact_path.resolve(), prediction_path.resolve()),
        audit={"selection_artifacts_verified": 0, "completion_input_bindings": {}},
    )
    calls: list[Path] = []

    def validate(path: Path):
        calls.append(path.resolve())
        return handoff

    monkeypatch.setattr(sealed, "validate_frozen_handoff_chain", validate)
    _, _, found_prediction, found_selection, found_candidate = (
        sealed.validate_frozen_handoff(compact_path, selection_path)
    )
    assert found_prediction == prediction_path
    assert found_selection == selection_path
    assert found_candidate == candidate
    assert calls == [compact_path.resolve()]

    def reject_cutoff(_path: Path):
        raise ValueError("selection cutoff exceeds 2022-12-31")

    monkeypatch.setattr(sealed, "validate_frozen_handoff_chain", reject_cutoff)
    with pytest.raises(ValueError, match="cutoff"):
        sealed.validate_frozen_handoff(compact_path, selection_path)


def _sealed_panel_authority_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[SimpleNamespace, Path, Path, Path]:
    round_panel = tmp_path / "player_round_panel.parquet"
    round_panel.write_bytes(b"synthetic authoritative round panel")
    crosswalk = tmp_path / "odds_event_crosswalk.parquet"
    crosswalk.write_bytes(b"synthetic authoritative event crosswalk")
    panel_manifest = tmp_path / "manifest.json"
    payload = {
        "purpose": sealed.EXPECTED_ROOT_PANEL_PURPOSE,
        "production_changes": False,
        "parameters": {
            "end": "2026-08-22",
            "history_start": "2015-04-01",
            "img_signal_model_version": "predictive_skill_pga_validated_v2",
            "market_start": "2020-01-01",
            "tour": "pga",
        },
        "outputs": {
            "odds_event_crosswalk": {
                "path": str(crosswalk.resolve()),
                "sha256": sealed._sha256(crosswalk),
                "rows": 302,
            }
        },
    }
    panel_manifest.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(sealed, "DEFAULT_ROOT_PANEL_MANIFEST", panel_manifest)
    monkeypatch.setattr(sealed, "DEFAULT_EVENT_CROSSWALK", crosswalk)
    monkeypatch.setattr(
        sealed, "EXPECTED_ROOT_PANEL_MANIFEST_SHA256", sealed._sha256(panel_manifest)
    )
    monkeypatch.setattr(
        sealed, "EXPECTED_ROOT_PANEL_MANIFEST_BYTES", panel_manifest.stat().st_size
    )
    monkeypatch.setattr(
        sealed, "EXPECTED_ROOT_EVENT_CROSSWALK_SHA256", sealed._sha256(crosswalk)
    )
    handoff = SimpleNamespace(
        audit={
            "completion_input_bindings": {
                "round_panel": {
                    "path": str(round_panel.resolve()),
                    "sha256": sealed._sha256(round_panel),
                }
            }
        }
    )
    return handoff, round_panel, crosswalk, panel_manifest


def test_sealed_panel_authority_rejects_alternate_or_tampered_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handoff, round_panel, crosswalk, panel_manifest = (
        _sealed_panel_authority_fixture(tmp_path, monkeypatch)
    )
    audit = sealed.validate_sealed_panel_authorities(
        handoff,
        round_panel,
        crosswalk,
        panel_manifest_path=panel_manifest,
    )
    assert set(audit["verified_paths"]) == {
        round_panel.resolve(),
        crosswalk.resolve(),
        panel_manifest.resolve(),
    }

    alternate_round = tmp_path / "alternate_round.parquet"
    alternate_round.write_bytes(round_panel.read_bytes())
    with pytest.raises(ValueError, match="round-panel differs"):
        sealed.validate_sealed_panel_authorities(
            handoff,
            alternate_round,
            crosswalk,
            panel_manifest_path=panel_manifest,
        )

    alternate_crosswalk = tmp_path / "alternate_crosswalk.parquet"
    alternate_crosswalk.write_bytes(crosswalk.read_bytes())
    with pytest.raises(ValueError, match="event-crosswalk differs"):
        sealed.validate_sealed_panel_authorities(
            handoff,
            round_panel,
            alternate_crosswalk,
            panel_manifest_path=panel_manifest,
        )

    crosswalk.write_bytes(b"tampered after authority freeze")
    with pytest.raises(ValueError, match="event-crosswalk bytes differ"):
        sealed.validate_sealed_panel_authorities(
            handoff,
            round_panel,
            crosswalk,
            panel_manifest_path=panel_manifest,
        )


def test_sealed_postbegin_rejects_handoff_path_switch_and_untracked_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority_handoff, round_panel, crosswalk, panel_manifest = (
        _sealed_panel_authority_fixture(tmp_path, monkeypatch)
    )
    compact_manifest = tmp_path / "compact_manifest.json"
    selection_manifest = tmp_path / "selection_manifest.json"
    frozen_input = tmp_path / "frozen_input.parquet"
    switched_input = tmp_path / "switched_input.parquet"
    for path in (compact_manifest, selection_manifest, frozen_input, switched_input):
        path.write_bytes(path.name.encode("utf-8"))
    pre_paths = (
        compact_manifest.resolve(),
        selection_manifest.resolve(),
        frozen_input.resolve(),
    )
    authority_paths = (
        round_panel.resolve(),
        crosswalk.resolve(),
        panel_manifest.resolve(),
    )

    def make_handoff(paths: tuple[Path, ...]) -> SimpleNamespace:
        return SimpleNamespace(
            selection_manifest_path=selection_manifest.resolve(),
            verified_paths=paths,
            audit=authority_handoff.audit,
        )

    tracked = tuple(
        {"path": str(path)} for path in {*pre_paths, *authority_paths}
    )
    state = SimpleNamespace(input_fingerprints=tracked)
    monkeypatch.setattr(
        sealed,
        "validate_frozen_handoff_chain",
        lambda _path: make_handoff(
            (
                compact_manifest.resolve(),
                selection_manifest.resolve(),
                switched_input.resolve(),
            )
        ),
    )
    with pytest.raises(ValueError, match="resolved-path set changed"):
        sealed.revalidate_sealed_handoff_after_begin(
            compact_manifest,
            selection_manifest,
            expected_handoff_paths=pre_paths,
            expected_authority_paths=authority_paths,
            provenance_state=state,
            round_panel_path=round_panel,
            event_crosswalk_path=crosswalk,
            panel_manifest_path=panel_manifest,
        )

    monkeypatch.setattr(
        sealed,
        "validate_frozen_handoff_chain",
        lambda _path: make_handoff(pre_paths),
    )
    incomplete_state = SimpleNamespace(
        input_fingerprints=tuple(
            {"path": str(path)}
            for path in {*pre_paths, *authority_paths}
            if path != frozen_input.resolve()
        )
    )
    with pytest.raises(ValueError, match="input set differs"):
        sealed.revalidate_sealed_handoff_after_begin(
            compact_manifest,
            selection_manifest,
            expected_handoff_paths=pre_paths,
            expected_authority_paths=authority_paths,
            provenance_state=incomplete_state,
            round_panel_path=round_panel,
            event_crosswalk_path=crosswalk,
            panel_manifest_path=panel_manifest,
        )


def test_frozen_mapper_uses_only_declared_discovery_parameters() -> None:
    selection = _frozen_selection()
    names = [
        "complete__ridge__total__w0__quarterly__a10",
        sealed.STRICT_PT_BASELINE,
    ]
    mapper = sealed.frozen_probability_mapper(selection, names)
    assert len(mapper) == 4
    assert mapper["fit_end"].max() == pd.Timestamp("2022-12-31")
    assert set(mapper.loc[mapper["market"].eq("r1"), "sigma"]) == {4.5}
    assert not mapper["sealed_data_used_for_mapper"].any()

    selection["final_frozen_probability_parameters"][0]["fit_end"] = "2023-01-01"
    with pytest.raises(ValueError, match="discovery-only"):
        sealed.frozen_probability_mapper(selection, names)


def test_frozen_opener_blend_is_discovery_locked_and_close_independent() -> None:
    candidate = "complete__ridge__total__w0__quarterly__a10"
    selection = _frozen_selection(candidate)
    scored = pd.DataFrame(
        {
            "candidate": [candidate],
            "quote_id": ["quote-1"],
            "book": ["pinnacle"],
            "p1_id": ["1"],
            "p2_id": ["2"],
            "p1_outcome": [1.0],
            "p2_outcome": [0.0],
            "skill_difference": [0.40],
            "p1_open": [120.0],
            "p2_open": [-130.0],
            "p1_close": [-200.0],
            "p2_close": [180.0],
            "p1_probability": [0.60],
            "push_probability": [0.10],
            "probability_calibration_source": ["discovery_mapper"],
            "mapped_event_key": ["pga:R2023001"],
            "market": ["r1"],
            "pair_key": ["1|2"],
        }
    )
    predictions = pd.DataFrame(
        {
            "candidate": [candidate],
            "event_key": ["pga:R2023001"],
            "dg_id_key": ["1"],
            "prediction": [1.25],
        }
    )
    roles = {candidate: "diagnostic_challenger"}
    blended, cloned, names, output_roles, audit = (
        sealed.append_frozen_opener_blend_sensitivity(
            scored, predictions, selection, candidate, roles
        )
    )
    sensitivity_name = f"{candidate}__frozen_opener_blend"
    result = blended.loc[blended["candidate"].eq(sensitivity_name)].iloc[0]
    model_conditional = 0.60 / 0.90
    left_open = 100.0 / 220.0
    right_open = 130.0 / 230.0
    open_novig = left_open / (left_open + right_open)
    expected_conditional = 1.0 / (
        1.0
        + np.exp(
            -0.5 * np.log(model_conditional / (1.0 - model_conditional))
            -0.5 * np.log(open_novig / (1.0 - open_novig))
        )
    )
    assert np.isclose(result["p1_probability"] / 0.90, expected_conditional)
    assert audit["uses_closing_odds_as_feature"] is False
    assert audit["selected_opener_weight"] == 0.5
    assert sensitivity_name in names
    assert output_roles[sensitivity_name].endswith("opener_blend_sensitivity")
    assert set(cloned["candidate"]) == {candidate, sensitivity_name}

    changed_close = scored.assign(p1_close=500.0, p2_close=-800.0)
    changed, *_ = sealed.append_frozen_opener_blend_sensitivity(
        changed_close, predictions, selection, candidate, roles
    )
    changed_probability = changed.loc[
        changed["candidate"].eq(sensitivity_name), "p1_probability"
    ].iloc[0]
    assert np.isclose(changed_probability, result["p1_probability"])

    metadata = pd.DataFrame(
        {
            "candidate": [candidate],
            "control": [False],
            "sensitivity_only": [False],
            "selection_eligible": [True],
            "candidate_role": ["diagnostic_challenger"],
            "sealed_selection_eligible": [False],
            "sealed_result_can_reselect": [False],
        }
    )
    metadata = sealed.append_sensitivity_metadata(
        metadata,
        source_candidate=candidate,
        sensitivity_candidate=sensitivity_name,
        role=output_roles[sensitivity_name],
    )
    sensitivity_metadata = metadata.loc[
        metadata["candidate"].eq(sensitivity_name)
    ].iloc[0]
    assert bool(sensitivity_metadata["sensitivity_only"])
    assert not bool(sensitivity_metadata["selection_eligible"])
    assert not bool(sensitivity_metadata["sealed_result_can_reselect"])


def test_sealed_opener_blend_uses_one_best_three_book_canonical_probability() -> None:
    candidate = "complete__ridge__total__w0__quarterly__a10"
    selection = _frozen_selection(candidate)
    books = ["betonline", "betcris", "pinnacle"]
    scored = pd.DataFrame(
        {
            "candidate": [candidate] * 3,
            "quote_id": [f"quote-{book}" for book in books],
            "book": books,
            "p1_id": ["1"] * 3,
            "p2_id": ["2"] * 3,
            "p1_outcome": [1.0] * 3,
            "p2_outcome": [0.0] * 3,
            "skill_difference": [0.35] * 3,
            # BetOnline is the frozen-rule best canonical-left opener.
            "p1_open": [140.0, 110.0, 125.0],
            "p2_open": [-150.0, -120.0, -135.0],
            "p1_close": [115.0, 105.0, 120.0],
            "p2_close": [-125.0, -115.0, -130.0],
            "p1_probability": [0.54] * 3,
            "push_probability": [0.04] * 3,
            "probability_calibration_source": ["discovery_mapper"] * 3,
            "mapped_event_key": ["pga:R2023001"] * 3,
            "market": ["r1"] * 3,
            "pair_key": ["1|2"] * 3,
        }
    )
    predictions = pd.DataFrame(
        {
            "candidate": [candidate],
            "event_key": ["pga:R2023001"],
            "dg_id_key": ["1"],
            "prediction": [1.25],
        }
    )
    blended, _, _, _, audit = sealed.append_frozen_opener_blend_sensitivity(
        scored,
        predictions,
        selection,
        candidate,
        {candidate: "diagnostic_challenger"},
    )
    sensitivity_name = f"{candidate}__frozen_opener_blend"
    sensitivity = blended.loc[blended["candidate"].eq(sensitivity_name)].copy()

    # All three execution rows survive with their own same-book open/close data.
    assert len(sensitivity) == 3
    execution_columns = ["book", "p1_open", "p2_open", "p1_close", "p2_close"]
    pd.testing.assert_frame_equal(
        sensitivity[execution_columns].sort_values("book").reset_index(drop=True),
        scored[execution_columns].sort_values("book").reset_index(drop=True),
        check_dtype=False,
    )
    assert set(sensitivity["opener_blend_reference_book"]) == {"betonline"}

    model_conditional = 0.54 / 0.96
    left_open = 100.0 / 240.0
    right_open = 150.0 / 250.0
    open_novig = left_open / (left_open + right_open)
    expected_conditional = 1.0 / (
        1.0
        + np.exp(
            -0.5 * np.log(model_conditional / (1.0 - model_conditional))
            - 0.5 * np.log(open_novig / (1.0 - open_novig))
        )
    )
    assert np.allclose(sensitivity["p1_pricing_probability"], expected_conditional)

    # This is the actual downstream canonicalization gate that previously
    # failed when each book received its own independently blended probability.
    canonical = sealed.market_eval._canonical_logical_pairs(sensitivity)
    assert len(canonical) == 1
    assert np.isclose(
        canonical.iloc[0]["canonical_probability"],
        expected_conditional * 0.96,
    )
    assert audit["propagated_execution_quote_rows"] == 3
    assert audit["canonical_probability_is_book_invariant"] is True
    assert (
        audit["opener_reference_rule"]["selection_rule"]
        == "maximum canonical-left American opening price; frozen book order "
        "breaks exact price ties only"
    )


def test_sealed_opener_filters_invalid_same_book_prices_before_conversion() -> None:
    candidate = "complete__ridge__total__w0__quarterly__a10"
    selection = _frozen_selection(candidate)
    scored = pd.DataFrame(
        {
            "candidate": [candidate, candidate],
            "quote_id": ["valid", "invalid"],
            "book": ["pinnacle", "betonline"],
            "p1_id": ["1", "3"],
            "p2_id": ["2", "4"],
            "p1_outcome": [1.0, 0.0],
            "p2_outcome": [0.0, 1.0],
            "skill_difference": [0.30, 0.20],
            "mapped_event_key": ["pga:R2023001", "pga:R2023001"],
            "market": ["r1", "r1"],
            "pair_key": ["1|2", "3|4"],
            "p1_open": [110.0, 0.0],
            "p2_open": [-120.0, -120.0],
            "p1_close": [-110.0, -110.0],
            "p2_close": [-110.0, -110.0],
            "p1_probability": [0.55, 0.55],
            "push_probability": [0.05, 0.05],
            "probability_calibration_source": ["discovery", "discovery"],
        }
    )
    predictions = pd.DataFrame(
        {
            "candidate": [candidate],
            "event_key": ["pga:R2023001"],
            "dg_id_key": ["1"],
            "prediction": [1.0],
        }
    )
    blended, _, _, _, audit = sealed.append_frozen_opener_blend_sensitivity(
        scored, predictions, selection, candidate, {candidate: "diagnostic_challenger"}
    )
    sensitivity = blended["candidate"].str.endswith("__frozen_opener_blend")
    assert sensitivity.sum() == 1
    assert audit["invalid_or_missing_same_book_open_rows"] == 1
    assert audit["strict_odds_conversion_after_validity_filter"]
    price_view = sealed._price_view(scored, "open")
    assert price_view["quote_id"].tolist() == ["valid"]


def test_sealed_source_path_hash_full_menu_and_values_are_exact(
    tmp_path: Path,
) -> None:
    candidate = "complete__ridge__total__w0__quarterly__a10"
    selection = _frozen_selection(candidate)
    names = [
        candidate,
        sealed.STRICT_PT_BASELINE,
        sealed.STRICT_LEGACY_CONTROL,
        sealed.RETROSPECTIVE_CONTROL,
    ]
    units = {
        candidate: "sg_per_round",
        **sealed.discovery.KNOWN_CONTROL_UNITS,
    }
    values = {
        candidate: (1.25, 1.10),
        sealed.STRICT_PT_BASELINE: (0.75, 0.60),
        sealed.STRICT_LEGACY_CONTROL: (0.70, 0.55),
        sealed.RETROSPECTIVE_CONTROL: (0.80, 0.65),
    }
    rows = []
    for name in names:
        for dg_id, prediction in zip((1, 2), values[name]):
            rows.append(
                {
                    "candidate": name,
                    "event_key": "pga|2023|1|2023-01-05",
                    "canonical_event_key": "pga:R2023001",
                    "dg_id_key": str(dg_id),
                    "event_start_date": pd.Timestamp("2023-01-05"),
                    "prediction": prediction,
                    "prediction_units": units[name],
                    "prediction_asof": pd.Timestamp("2023-01-04"),
                    "training_max_event_end": pd.Timestamp("2022-12-31"),
                    "chronology_ok": True,
                }
            )
    source_path = tmp_path / "source_predictions.parquet"
    pd.DataFrame(rows).to_parquet(source_path, index=False)
    source_manifest_path = tmp_path / "source_manifest.json"
    source_manifest_path.write_text(
        json.dumps(
            {
                "outputs": {
                    "player_event_predictions.parquet": {
                        "path": str(source_path.resolve()),
                        "sha256": sealed._sha256(source_path),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    selection["inputs"] = {
        "predictions": {
            "path": str(source_path.resolve()),
            "sha256": sealed._sha256(source_path),
        },
        "candidate_manifest": {
            "path": str(source_manifest_path.resolve()),
            "sha256": sealed._sha256(source_manifest_path),
        },
    }
    compact_rows = []
    for dg_id, finalist, control in zip(
        (1, 2), values[candidate], values[sealed.STRICT_PT_BASELINE]
    ):
        compact_rows.append(
            {
                "event_key": "pga:R2023001",
                "dg_id": dg_id,
                "event_start": pd.Timestamp("2023-01-05"),
                "candidate": candidate,
                "finalist_core": finalist,
                "control_core": control,
                "split": "sealed",
                "prediction_asof": pd.Timestamp("2023-01-04"),
                "chronology_ok": True,
            }
        )
    compact_path = tmp_path / "compact.parquet"
    pd.DataFrame(compact_rows).to_parquet(compact_path, index=False)
    compact = {
        "source_prediction_file": str(source_path.resolve()),
        "source_prediction_sha256": sealed._sha256(source_path),
    }
    loaded, _, loaded_names, _, audit = sealed.load_frozen_predictions(
        compact, compact_path, selection, candidate
    )
    assert len(loaded) == 8
    assert set(loaded_names) == set(names)
    assert audit["selection_source_path_and_hash_match"]
    assert audit["compact_sealed_full_menu_match"]

    alternate = tmp_path / "alternate.parquet"
    alternate.write_bytes(source_path.read_bytes())
    bad_selection = json.loads(json.dumps(selection, default=str))
    bad_selection["inputs"]["predictions"]["path"] = str(alternate.resolve())
    with pytest.raises(ValueError, match="path differs"):
        sealed.load_frozen_predictions(
            compact, compact_path, bad_selection, candidate
        )

    incomplete_path = tmp_path / "compact_incomplete.parquet"
    pd.DataFrame(compact_rows[:1]).to_parquet(incomplete_path, index=False)
    with pytest.raises(ValueError, match="full selected/control source menus"):
        sealed.load_frozen_predictions(
            compact, incomplete_path, selection, candidate
        )


def test_primary_quote_coverage_fails_closed() -> None:
    selected = "complete__ridge__total__w0__quarterly__a10"
    coverage = pd.DataFrame(
        [
            {
                "candidate": candidate,
                "year": 2023,
                "market": "r1",
                "quote_rows": 10,
                "covered_quote_rows": covered,
                "logical_pairs": 5,
                "covered_logical_pairs": covered // 2,
                "missing_players": missing,
            }
            for candidate, covered, missing in (
                (selected, 10, 0),
                (sealed.STRICT_PT_BASELINE, 10, 0),
            )
        ]
    )
    sealed.enforce_primary_quote_coverage(coverage, selected)
    coverage.loc[
        coverage["candidate"].eq(selected), "covered_quote_rows"
    ] = 9
    with pytest.raises(ValueError, match="coverage is incomplete"):
        sealed.enforce_primary_quote_coverage(coverage, selected)


def test_sealed_reliability_gate_keeps_unreliable_mid_edge_and_open_clv() -> None:
    rows = []
    for index, (reliable, probability) in enumerate(
        ((False, 0.80), (True, 0.80), (False, 0.505)), start=1
    ):
        rows.append(
            {
                "candidate": "candidate",
                "candidate_role": "diagnostic_challenger",
                "quote_id": f"quote-{index}",
                "mapped_event_key": f"event-{index}",
                "event_start_date": pd.Timestamp("2023-06-01"),
                "market": "r1",
                "book": "pinnacle",
                "pair_key": f"{index}|{index + 10}",
                "p1_id": str(index),
                "p2_id": str(index + 10),
                "p1_player_name": "left",
                "p2_player_name": "right",
                "p1_open": 110.0,
                "p2_open": -110.0,
                "p1_close": 100.0,
                "p2_close": -110.0,
                "p1_outcome": 1.0,
                "p2_outcome": 0.0,
                "p1_outcome_text": "win",
                "p2_outcome_text": "loss",
                "tie_rule": "ties push",
                "pair_reliability_ok": reliable,
                "p1_probability": probability,
                "push_probability": 0.02,
            }
        )
    kept, suppressed = sealed.select_bet_slice(
        pd.DataFrame(rows),
        price_basis="open",
        line_policy="best_of_three",
        sizing="flat_1u",
        reliability_policy="suppress_unreliable_ge10pct",
    )
    assert len(kept) == 2
    assert len(suppressed) == 1
    assert (~kept["pair_reliability_ok"]).sum() == 1
    assert kept.loc[~kept["pair_reliability_ok"], "ev_per_unit"].between(
        0.03, 0.10, inclusive="left"
    ).all()
    assert suppressed.iloc[0]["ev_per_unit"] >= 0.10
    assert kept["open_to_close_clv_pp"].notna().all()
    assert (kept["open_to_close_clv_pp"] > 0.0).all()
    expected_open_novig = (100.0 / 210.0) / (
        (100.0 / 210.0) + (110.0 / 210.0)
    )
    expected_close_novig = 0.5 / (0.5 + 110.0 / 210.0)
    assert np.allclose(
        kept["open_to_close_novig_clv_pp"],
        expected_close_novig - expected_open_novig,
    )
    assert "raw_implied_open_to_close_move_pp" in kept


def test_weather_join_accepts_local_or_canonical_identity_and_fails_on_miss() -> None:
    weather = pd.DataFrame(
        {
            "local_event_key": ["pga|2024|54|2023-09-14"],
            "canonical_event_key": ["pga:R2023544"],
            "event_start_date": [pd.Timestamp("2023-09-14")],
            "weather_round_rows": [100],
            "field_wind_mean": [10.0],
            "weather_wind_bucket": ["8-12"],
        }
    )
    bets = pd.DataFrame(
        {"event_key": ["pga|2024|54|2023-09-14", "pga:R2023544"]}
    )
    joined = sealed.attach_weather_to_bets(bets, weather)
    assert joined["weather_round_rows"].tolist() == [100, 100]
    with pytest.raises(ValueError, match="weather attribution failed"):
        sealed.attach_weather_to_bets(
            pd.DataFrame({"event_key": ["pga:R2099999"]}), weather
        )
    ambiguous = pd.concat(
        [weather, weather.assign(canonical_event_key="pga:R2023545")],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="ambiguous before"):
        sealed.attach_weather_to_bets(
            pd.DataFrame({"event_key": ["pga:R2023544"]}), ambiguous
        )


def test_weather_profiles_exclude_only_panel_events_without_matchup_boards(
    tmp_path: Path,
) -> None:
    quoted_local = "pga|2024|54|2023-09-14"
    unquoted_local = "pga|2024|999|2023-10-05"
    round_panel = tmp_path / "round_panel.parquet"
    rows = []
    for event_key, event_id, start in (
        (quoted_local, 54, "2023-09-14"),
        (unquoted_local, 999, "2023-10-05"),
    ):
        for player in range(2):
            rows.append(
                {
                    "event_key": event_key,
                    "year": 2024,
                    "event_id": event_id,
                    "event_start_date": pd.Timestamp(start),
                    "round_num": 1,
                    "weather_field_mean_wind": 10.0,
                    "weather_field_mean_dew": 55.0,
                    "weather_field_mean_temp": 70.0,
                    "weather_field_mean_rain": 0.0,
                    "weather_delta_wind": float(player),
                    "weather_delta_dew": float(player),
                    "weather_delta_temp": float(player),
                    "weather_delta_rain": 0.0,
                }
            )
    pd.DataFrame(rows).to_parquet(round_panel, index=False)
    source_manifest = tmp_path / "candidate_manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "inputs": {
                    "round_panel": {
                        "path": str(round_panel.resolve()),
                        "sha256": sealed._sha256(round_panel),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    selection = {
        "inputs": {
            "candidate_manifest": {
                "path": str(source_manifest.resolve()),
                "sha256": sealed._sha256(source_manifest),
            }
        }
    }
    crosswalk = tmp_path / "crosswalk.parquet"
    pd.DataFrame(
        {
            "odds_calendar_year": [2023],
            "odds_event_id": [544],
            "local_event_key": [quoted_local],
            "event_identity_status": ["matched"],
            "event_identity_ambiguous": [False],
            "local_event_has_multiple_odds_ids": [False],
        }
    ).to_parquet(crosswalk, index=False)
    profiles, audit = sealed.load_weather_profiles(
        selection, round_panel, crosswalk
    )
    assert profiles["local_event_key"].tolist() == [quoted_local]
    assert profiles["canonical_event_key"].tolist() == ["pga:R2023544"]
    assert audit["unquoted_panel_event_profiles_excluded"] == 1
    sealed.attach_weather_to_bets(
        pd.DataFrame({"event_key": ["pga:R2023544"]}), profiles
    )


def test_enrichment_uses_common_baseline_skill_history_img_and_fallback_buckets() -> None:
    event = "pga|2024|10|2024-02-01"
    quote = "quote-1"
    predictions = pd.DataFrame(
        [
            {
                "candidate": candidate,
                "event_key": event,
                "dg_id_key": player,
                "prediction": prediction,
                "reliability_ok": True,
                "history_rounds": history,
                "img_history_coverage": img_coverage,
                "cross_tour_fallback_reason": fallback,
                "prior_event_withdrawal": False,
            }
            for candidate, player, prediction, history, img_coverage, fallback in (
                ("challenger", "1", 2.5, 15, 0.4, "candidate_specific"),
                ("challenger", "2", 0.0, 60, 0.8, "candidate_specific"),
                (sealed.STRICT_PT_BASELINE, "1", 1.2, 15, 0.4, "weekly_pga_core"),
                (sealed.STRICT_PT_BASELINE, "2", 0.4, 60, 0.8, "tour_prior"),
            )
        ]
    )
    scored = pd.DataFrame(
        {
            "candidate": ["challenger"],
            "quote_id": [quote],
            "p1_outcome_text": ["win"],
            "p2_outcome_text": ["loss"],
        }
    )
    bets = pd.DataFrame(
        {
            "selection_key": [f"challenger|{quote}|p1"],
            "candidate": ["challenger"],
            "event_key": [event],
            "selection_id": ["1"],
            "opponent_id": ["2"],
            "pair_reliability_ok": [True],
        }
    )
    weather = pd.DataFrame(
        {
            "local_event_key": [event],
            "canonical_event_key": ["pga:R2024010"],
            "event_start_date": [pd.Timestamp("2024-02-01")],
            "weather_round_rows": [100],
            "weather_wind_bucket": ["8-12"],
        }
    )
    enriched = sealed.enrich_bets(bets, scored, predictions, weather)
    row = enriched.iloc[0]
    assert row["baseline_selection_prediction"] == 1.2
    assert row["baseline_opponent_prediction"] == 0.4
    assert np.isclose(row["baseline_skill_gap"], 0.8)
    assert row["baseline_skill_gap_bucket"] == "0.5-1"
    assert row["baseline_selection_history_bucket"] == "10-19"
    assert row["baseline_selection_img_coverage_bucket"] == "25-50%"
    assert row["baseline_selection_cross_tour_fallback_reason"] == "weekly_pga_core"


def _money_event_fixture() -> pd.DataFrame:
    rows = []
    for candidate, scale in (("challenger_a", 1.0), ("challenger_b", -0.5)):
        for index in range(8):
            rows.append(
                {
                    "event_key": f"event-{index}",
                    "event_start_date": pd.Timestamp("2023-01-01")
                    + pd.Timedelta(days=7 * index),
                    "event_delta": scale * (index - 2.5),
                    "candidate": candidate,
                    "baseline": sealed.STRICT_PT_BASELINE,
                    "opportunity_mode": "all_available",
                    "market": "r1",
                    "price_basis": "close",
                    "line_policy": "best_of_three",
                    "sizing": "flat_1u",
                    "reliability_policy": "ungated",
                    "metric": "pnl_delta",
                    "zero_bet_event": index == 0,
                }
            )
    return pd.DataFrame(rows)


def test_event_bootstrap_includes_zero_events_and_block_max_stat_is_deterministic() -> None:
    fixture = _money_event_fixture()
    first = sealed.event_cluster_bootstrap(
        fixture, draws=10_000, seed=7, better_when="positive"
    )
    second = sealed.event_cluster_bootstrap(
        fixture, draws=10_000, seed=7, better_when="positive"
    )
    pd.testing.assert_frame_equal(first, second)
    assert set(first["events"]) == {8}
    assert set(first["zero_bet_events"]) == {1}
    reality_first = sealed.block_max_stat_reality_check(
        fixture, draws=10_000, seed=11, block_sizes=(2, 3, 4)
    )
    reality_second = sealed.block_max_stat_reality_check(
        fixture, draws=10_000, seed=11, block_sizes=(2, 3, 4)
    )
    pd.testing.assert_frame_equal(reality_first, reality_second)
    assert len(reality_first) == 6
    assert set(reality_first["block_size_events"]) == {2, 3, 4}
    assert reality_first["max_stat_adjusted_p"].between(0.0, 1.0).all()


def test_sealed_market_scoring_module_is_an_explicit_provenance_dependency() -> None:
    expected = tuple(
        dict.fromkeys(
            (
                sealed.REPO / "archive" / "base_skill_frozen_handoff.py",
                Path(sealed.discovery.__file__).resolve(),
                *(
                    Path(path).resolve()
                    for path in sealed.discovery.DISCOVERY_PROVENANCE_SOURCE_CLOSURE
                ),
            )
        )
    )
    assert sealed.PROVENANCE_CODE_PATHS == expected
    assert {path.name for path in expected} >= {
        "base_skill_frozen_handoff.py",
        "evaluate_complete_base_skill_market_candidates.py",
        "evaluate_complete_base_skill_market.py",
        "backtest_base_skill_market_models.py",
        "historical_odds_cache.py",
        "market_backtest_scoring.py",
        "research_execution_provenance.py",
    }
    assert all(path.is_file() for path in expected)


def test_sealed_complete_manifest_rejects_post_finalize_input_mutation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sealed.py"
    direct_input = tmp_path / "sealed_input.parquet"
    source.write_text("# sealed synthetic\n", encoding="utf-8")
    direct_input.write_bytes(b"before")
    output = tmp_path / "sealed_output"
    state = sealed.execution_provenance.begin(
        output_dir=output,
        primary_source=source,
        input_paths=[direct_input],
    )
    sealed.execution_provenance.finalize(state)
    direct_input.write_bytes(b"mutated after finalize")
    manifest_path = output / "manifest.json"
    with pytest.raises(RuntimeError, match="input changed"):
        sealed.write_sealed_complete_manifest(
            manifest_path, {"complete": True}, state
        )
    assert not manifest_path.exists()
