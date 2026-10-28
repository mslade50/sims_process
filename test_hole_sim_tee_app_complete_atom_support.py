from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import inspect
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import pytest


ARCHIVE = Path(__file__).resolve().parent / "archive"
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

import hole_sim_donor_ability_policy as ability_policy
import hole_sim_fixed_reference_value as fixed_reference
import hole_sim_hazard_resolution as hazard
import hole_sim_normalized_chain_streaming as streaming
import hole_sim_streaming_transition_ledger as proposal_ledger
import hole_sim_tee_app_complete_atom_support as complete
import hole_sim_transition_mean_anchor as anchor


REFERENCE_DIR = Path("permanent_data/hole_sim_fixed_reference_travelers_v2")
_PRE_ADMISSION_SEAM_SHA256 = (
    "70c41fdb11aa2ae369caa474cfeb9c8fab423d740acd3d19f81c35a3d7a1da83"
)
_ADMITTED_SEAM_SHA256 = (
    "597f068d68cee7580a964afc687b29a66ad3cfe29b8c2fa47b11172abedf6498"
)
_PRE_ADMISSION_AUTHORITY_LOCK_SHA256 = (
    "d2231cb2d91f50b42a0e460a94393e598c179fca137b8695ca6b1e7de8e7c1f0"
)
_ADMITTED_AUTHORITY_LOCK_SHA256 = (
    "edbadf6498b1fd5166c6711f0ac20e87bb7d691ec58f7f511f660cd38b978a4c"
)
_H8_AUTHORITY_RECORD_SHA256 = (
    "efaf969daea1ac9a7680b0fd8d534ffa700430523aa7cb13cec6fabaed9c347c"
)
_EXPECTED_H8_REVIEW_EVIDENCE = {
    "candidate_id": (
        "341ce3818fcc5301179c8858fcba02a94778384e2e554e81fd762d2f9626b7ff"
    ),
    "candidate_output_index_sha256": (
        "28bc60ca52ec42ee1abba4eaa26d5d10a6cc50f6df1fdb4b0e1631594c977aa5"
    ),
    "candidate_manifest_sha256": (
        "d0be4068daf67c05243eeed4eb2963168ec8555ba2dc5842d72ade434e09bfc2"
    ),
    "authority_candidate_sha256": (
        "f046624aa48601a37b7ad6e8b2ab94f8558e857fca6220e286b9611dc010d5d4"
    ),
    "candidate_builder_sha256": (
        "809c163db737bc4d628101d4f4687543863ffd7a995f5ee9ece04d5d88c47182"
    ),
    "reviewed_pre_admission_seam_sha256": _PRE_ADMISSION_SEAM_SHA256,
    "audit_id": (
        "16c79fb1fd622db1ee7f4b8cb638ec0b72b746d0c2370541c23957d12d23b6e1"
    ),
    "audit_output_index_sha256": (
        "cdfb7b50a958ddd397aaf5297767d0f1d5e11ca80b946eb0a13475a04f8f3431"
    ),
    "audit_json_sha256": (
        "57231a27cbb7df00fe4af95d957efccdcbdad67d5e202340a3ff2f4cd0c43402"
    ),
    "audit_report_sha256": (
        "405428f00644bb53984dbc89ec2394d6f54441865d55f971324329b5de2f5368"
    ),
    "auditor_source_sha256": (
        "8d9bb4ae1f3e3c8c0cca875af0aa827afe237545dc657f86f1ed8963787aac06"
    ),
    "auditor_test_sha256": (
        "aa44f738e85db98a3faf287f553fec7cb9651266961cc1e6fee4c77eef1a350c"
    ),
}
_EXPECTED_H8_QUERY_PROJECTION = {
    "query_key": (
        "943f67faa5e345d395bf706e29c1a526e69879f80e14e81f55e255fd81dbf43e"
    ),
    "proposal_source_inventory_records": 904,
    "proposal_source_inventory_sha256": (
        "1f3d370647885ac3e9917fc1ebac81a7594b6a11c592b92e86ab65cda362d7a8"
    ),
    "surface_source_inventory_records": 903,
    "surface_source_inventory_sha256": (
        "2fda69c1b3a666f23d9041023a75e68efe36ed0f68816fde4fa90fd7cbc9ae38"
    ),
    "hazard_source_inventory_records": 30702,
    "hazard_source_inventory_sha256": (
        "55b324ee433f0e2841e896f0862a5e829c4d40a37358e7fb14fd4d4f93bbb465"
    ),
}
_EXPECTED_H8_AUTHORITY_RECORD = {
    "authority_id": "lowry_r1h8_raw_app_exact_query_v1",
    "scope": "lowry_r1h8_raw_app_exact_query_only",
    "review_evidence": _EXPECTED_H8_REVIEW_EVIDENCE,
    "source_inventory_sha256": (
        "37a91ecae53d7945132a9eca7871490250b57f978ec95df4f06e477550c9f92c"
    ),
    "queries": (_EXPECTED_H8_QUERY_PROJECTION,),
}
_H8_AUTHORITY_MUTATION_PATHS = (
    ("authority_id",),
    ("scope",),
    ("source_inventory_sha256",),
    *(("review_evidence", field) for field in _EXPECTED_H8_REVIEW_EVIDENCE),
    *(("queries", 0, field) for field in _EXPECTED_H8_QUERY_PROJECTION),
)


class FixedSurface:
    def __init__(self, probability: np.ndarray):
        self._probability = np.asarray(probability, float)
        self.points = np.column_stack(
            [np.arange(self._probability.shape[1], dtype=float), np.zeros(self._probability.shape[1])]
        )
        self.codes = np.arange(self._probability.shape[1], dtype=int)
        self.prior = np.full(
            self._probability.shape[1], 1.0 / self._probability.shape[1]
        )
        self.prior_strength = 1.0

    def probabilities(self, queries: np.ndarray) -> np.ndarray:
        if len(queries) != len(self._probability):
            if len(self._probability) == 1:
                return np.repeat(self._probability, len(queries), axis=0)
            raise AssertionError("fixed surface query count changed")
        return self._probability.copy()


def _hazard_donors(category: str, endpoint_feet: tuple[float, ...]) -> pd.DataFrame:
    rows = []
    for index in range(12):
        causal_x = float(endpoint_feet[index % len(endpoint_feet)]) + (index % 3 - 1) * 3.0
        rows.append(
            {
                "donor_key": f"{category}-hazard-{index:02d}",
                "event_key": f"pga:R202{index % 5}034",
                "season": 2020 + index % 5,
                "round_no": 1 + index % 4,
                "hole_no": 8,
                "category": category,
                "causal_endpoint_x": causal_x,
                "causal_endpoint_y": float((index % 3 - 1) * 2.0),
                "penalty_count": 1,
                "penalty_stroke_cost": 1 + int(index == 11),
                "resolution_shot_key": f"resolution-{category}-{index:02d}",
                "resolution_event_type": "STROKE",
                "resolution_x": causal_x + 9.0,
                "resolution_y": 0.0,
                "resolution_lie": "fairway",
                "resolution_lie_raw": "Fairway",
                "donor_geometry_resolution_distance_yd": abs(causal_x + 9.0 - 600.0) / 3.0,
                "donor_sg_resolution_distance_yd": abs(causal_x + 9.0 - 600.0) / 3.0,
                "donor_sg_resolution_lie": "fairway",
                "donor_pin_x": 600.0,
                "donor_pin_y": 0.0,
                "physical_course_id": "course:test",
                "foundation_version": hazard.FOUNDATION_VERSION,
                "coordinate_reference_year": 2025,
                "coordinate_reference_event_key": "pga:R2025034",
                "coordinate_reference_mode": "latest_prior",
                "data_role": "predictor_donor",
            }
        )
    return pd.DataFrame(rows)


def _resolver(category: str, endpoints: tuple[float, ...]) -> hazard.EndpointLinkedHazardResolver:
    return hazard.EndpointLinkedHazardResolver(
        _hazard_donors(category, endpoints),
        max_nearest_distance_yd=120.0,
        minimum_category_support=12,
    )


def _query(*, category: str, distance: float = 200.0) -> complete.CompleteAtomQuery:
    par = 4 if category == "ott" else 3
    return complete.CompleteAtomQuery(
        arm="raw",
        round_no=1,
        hole_no=8,
        par=par,
        target_event_key="pga:R2026034",
        target_event_start="2026-06-25T00:00:00Z",
        target_player_id="33204",
        start_lie="tee",
        start_distance_yd=distance,
        start_long_yd=0.0,
        start_lat_yd=0.0,
        forecast_tee_x=0.0,
        forecast_tee_y=0.0,
        forecast_pin_x=distance * 3.0,
        forecast_pin_y=0.0,
        forecast_pin_long_yd=distance,
        physical_course_id="course:test",
    )


def _proposal_and_numeric(
    *,
    category: str,
    masses: list[float],
    terminal: list[bool],
    points: np.ndarray,
    availability: list[bool] | None = None,
) -> tuple[pd.DataFrame, streaming.NumericSupport]:
    if category == "ott":
        support_key = "tee|raw|R1|H8|P4|33204"
        atom_kind = ["tee_spatial"] * len(masses)
        start_lie = "tee"
        bucket = -1
        target_par = 4
    else:
        support_key = "app|raw|R1|H8|tee|B14|33204"
        atom_kind = ["app_holeout" if value else "app_spatial" for value in terminal]
        start_lie = "tee"
        bucket = 14
        target_par = -1
    atom_ids = [f"atom-{index}" for index in range(len(masses))]
    spatial_ids = [atom_id for atom_id, is_terminal in zip(atom_ids, terminal) if not is_terminal]
    spatial_mass = np.asarray(
        [mass for mass, is_terminal in zip(masses, terminal) if not is_terminal], float
    )
    numeric_probability = spatial_mass / spatial_mass.sum()
    numeric_cdf = np.cumsum(numeric_probability)
    numeric_cdf[-1] = 1.0
    declared = float(sum(mass for mass, value in zip(masses, terminal) if value))
    cache_sha = "a" * 64
    payload_sha = streaming.support_payload_sha256(
        support_key,
        np.asarray(spatial_ids, dtype=object),
        np.asarray(points, float),
        numeric_probability,
        numeric_cdf,
        declared,
    )
    numeric = streaming.NumericSupport(
        key=support_key,
        atom_ids=np.asarray(spatial_ids, dtype=object),
        points=np.asarray(points, float),
        probability=numeric_probability,
        cdf=numeric_cdf,
        holeout_probability=declared,
        cache_sha256=cache_sha,
        payload_sha256=payload_sha,
    )
    proposal_probability = np.asarray(masses, float)
    proposal_upper = np.cumsum(proposal_probability)
    proposal_upper[-1] = 1.0
    proposal_lower = np.concatenate(([0.0], proposal_upper[:-1]))
    terminal_array = np.asarray(terminal, bool)
    branch = np.where(terminal_array, "terminal", "spatial")
    branch_total = np.asarray(
        [declared if value else 1.0 - declared for value in terminal_array], float
    )
    if category == "ott":
        branch_total[:] = 1.0
    branch_conditional = proposal_probability / branch_total
    branch_lower = np.zeros(len(masses), float)
    branch_upper = np.zeros(len(masses), float)
    for branch_name in dict.fromkeys(branch):
        index = np.flatnonzero(branch == branch_name)
        upper = np.cumsum(branch_conditional[index])
        upper[-1] = 1.0
        branch_lower[index] = np.concatenate(([0.0], upper[:-1]))
        branch_upper[index] = upper
    available = availability or [True] * len(masses)
    raw_ability = [0.2 + index * 0.1 if value else np.nan for index, value in enumerate(available)]
    rows = pd.DataFrame(
        {
            "arm": "raw",
            "support_key": support_key,
            "proposal_row_order": np.arange(len(masses), dtype=np.int32),
            "target_event_key": "pga:R2026034",
            "target_event_start": pd.Timestamp("2026-06-25", tz="UTC"),
            "target_player_id": "33204",
            "target_round_no": 1,
            "target_hole_no": 8,
            "target_par": target_par,
            "target_start_lie": start_lie,
            "target_distance_bucket": bucket,
            "atom_id": atom_ids,
            "category": category,
            "atom_kind": atom_kind,
            "terminal": terminal_array,
            "holed": terminal_array,
            "donor_event_key": [f"pga:R202{index}034" for index in range(len(masses))],
            "donor_player_id": [str(40000 + index) for index in range(len(masses))],
            "donor_player_name": [f"player {index}" for index in range(len(masses))],
            "donor_round_no": np.ones(len(masses), dtype=np.int16),
            "donor_hole_no": np.full(len(masses), 8, dtype=np.int16),
            "donor_stroke_no": np.ones(len(masses), dtype=np.int16),
            "source_shot_key": atom_ids,
            "source_start_lie": start_lie,
            "source_start_distance_yd": 200.0,
            "source_end_lie": ["hole" if value else "fairway" for value in terminal],
            "source_end_distance_yd": [0.0 if value else 100.0 for value in terminal],
            "source_sg_raw": np.nan,
            "source_model_id": "",
            "base_hierarchy_mass": proposal_probability,
            "spatial_model_weight": 1.0,
            "quality_base_mass": proposal_probability,
            "holeout_representation": (
                proposal_ledger.EXPLICIT_APP_HOLEOUT_VERSION
                if category == "app"
                else "not_applicable"
            ),
            "declared_app_base_holeout_probability": declared,
            "implicit_holeout_probability": 0.0,
            "app_holeout_equivalence_sha256": "b" * 64 if category == "app" else "not_applicable",
            "source_cache_sha256": cache_sha,
            "numeric_payload_sha256": payload_sha,
            "proposal_probability": proposal_probability,
            "proposal_cdf_lower": proposal_lower,
            "proposal_cdf_upper": proposal_upper,
            "proposal_branch": branch,
            "branch_total_mass": branch_total,
            "branch_conditional_probability": branch_conditional,
            "branch_cdf_lower": branch_lower,
            "branch_cdf_upper": branch_upper,
            "donor_event_start": [pd.Timestamp(f"202{index}-06-25", tz="UTC") for index in range(len(masses))],
            "donor_skill_date": [
                pd.Timestamp(f"202{index}-06-24", tz="UTC") if value else pd.NaT
                for index, value in enumerate(available)
            ],
            "donor_ability_sg": raw_ability,
            "donor_ability_available": available,
            "donor_ability_snapshot_kind": [
                "strict_prior_asof_content_verified_v2" if value else "strict_prior_snapshot_missing_fail_closed"
                for value in available
            ],
        }
    )
    measure_sha = proposal_ledger._frame_digest(
        rows, proposal_ledger.PROPOSAL_SUPPORT_DIGEST_COLUMNS
    )
    rows["proposal_support_measure_sha256"] = measure_sha
    rows = complete.apply_frozen_policy_to_proposal(rows)
    return rows, numeric


def _collect(request: complete.CompleteAtomRequest, batch: int = 9) -> tuple[pd.DataFrame, complete.CompleteAtomExpansionAudit]:
    expansion = complete.CompleteAtomExpansion(request, max_batch_rows=batch)
    frame = pd.concat(list(expansion.iter_batches()), ignore_index=True)
    return frame, expansion.audit


def _persistence_request() -> complete.CompleteAtomRequest:
    proposal, numeric = _proposal_and_numeric(
        category="app",
        masses=[0.9, 0.1],
        terminal=[False, True],
        points=np.asarray([[4.0, 0.0]]),
    )
    return complete.CompleteAtomRequest(
        proposal,
        numeric,
        _query(category="app", distance=210.0),
        FixedSurface(
            np.asarray([[0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        ),
        _resolver("app", (600.0,)),
        fixed_reference.ReferenceValueModel.load(REFERENCE_DIR),
    )


def _hazard_persistence_request() -> complete.CompleteAtomRequest:
    proposal, numeric = _proposal_and_numeric(
        category="ott",
        masses=[1.0],
        terminal=[False],
        points=np.asarray([[100.0, 0.0]]),
    )
    return complete.CompleteAtomRequest(
        proposal,
        numeric,
        _query(category="ott"),
        FixedSurface(np.asarray([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])),
        _resolver("ott", (300.0,)),
        fixed_reference.ReferenceValueModel.load(REFERENCE_DIR),
    )


def _recompute_expanded_cdfs(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy().sort_values(
        ["query_key", "proposal_row_order", "surface_class_index", "hazard_donor_order"],
        kind="mergesort",
    ).reset_index(drop=True)
    for _, query_rows in out.groupby("query_key", sort=False, observed=True):
        for _, proposal_rows in query_rows.groupby(
            "proposal_row_order", sort=False, observed=True
        ):
            for _, surface_rows in proposal_rows.groupby(
                ["surface_class_index", "surface_class_label"],
                sort=False,
                observed=True,
            ):
                hazard_probability = surface_rows["hazard_probability"].to_numpy(float)
                hazard_upper = np.cumsum(hazard_probability)
                hazard_lower = np.concatenate(([0.0], hazard_upper[:-1]))
                out.loc[surface_rows.index, "hazard_cdf_lower"] = hazard_lower
                out.loc[surface_rows.index, "hazard_cdf_upper"] = hazard_upper
            mechanics = proposal_rows["mechanics_conditional_probability"].to_numpy(float)
            mechanics_upper = np.cumsum(mechanics)
            mechanics_lower = np.concatenate(([0.0], mechanics_upper[:-1]))
            mechanics_upper[-1] = 1.0
            out.loc[proposal_rows.index, "mechanics_cdf_lower"] = mechanics_lower
            out.loc[proposal_rows.index, "mechanics_cdf_upper"] = mechanics_upper
        complete_mass = query_rows["quality_base_mass"].to_numpy(float)
        complete_upper = np.cumsum(complete_mass)
        complete_lower = np.concatenate(([0.0], complete_upper[:-1]))
        complete_upper[-1] = 1.0
        out.loc[query_rows.index, "complete_cdf_lower"] = complete_lower
        out.loc[query_rows.index, "complete_cdf_upper"] = complete_upper
        out.loc[query_rows.index, "complete_row_order"] = np.arange(
            len(query_rows), dtype=np.int64
        )
    return out.sort_values(
        ["query_key", "complete_row_order"], kind="mergesort"
    ).reset_index(drop=True)


def _reseal_artifact(
    output: Path, *, recompute_semantic_inventory: bool = False
) -> None:
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if recompute_semantic_inventory:
        all_rows = pd.concat(
            [pd.read_parquet(output / part["path"]) for part in manifest["parts"]],
            ignore_index=True,
        ).sort_values(["query_key", "complete_row_order"], kind="mergesort")
        declared_by_query = {
            str(item["query_key"]): item for item in manifest["inventory"]
        }
        rebuilt_inventory = []
        for query_key, query_rows in all_rows.groupby(
            "query_key", sort=True, observed=True
        ):
            query_rows = query_rows.sort_values(
                "complete_row_order", kind="mergesort"
            ).reset_index(drop=True)
            proposal_hasher = complete._OrderedInventoryHasher(
                complete.PROPOSAL_SOURCE_INVENTORY_CONTRACT,
                complete._PROPOSAL_SOURCE_INVENTORY_COLUMNS,
            )
            surface_hasher = complete._OrderedInventoryHasher(
                complete.SURFACE_SOURCE_INVENTORY_CONTRACT,
                complete._SURFACE_SOURCE_INVENTORY_COLUMNS,
            )
            hazard_hasher = complete._OrderedInventoryHasher(
                complete.HAZARD_SOURCE_INVENTORY_CONTRACT,
                complete._HAZARD_SOURCE_INVENTORY_COLUMNS,
            )
            seen_proposals = set()
            for row in query_rows.itertuples(index=False):
                values = row._asdict()
                proposal_order = int(values["proposal_row_order"])
                if proposal_order not in seen_proposals:
                    proposal_hasher.add(values)
                    if not bool(values["terminal"]):
                        surface_hasher.add(values)
                    seen_proposals.add(proposal_order)
                if values["hazard_resolution_status"] == "resolved_complete_atom":
                    hazard_hasher.add(values)
            declared = dict(declared_by_query[str(query_key)])
            proposal_residuals = []
            for _, proposal_rows in query_rows.groupby(
                "proposal_row_order", sort=False, observed=True
            ):
                proposal_residuals.append(
                    abs(
                        float(proposal_rows["quality_base_mass"].sum())
                        - float(proposal_rows["proposal_probability"].iloc[0])
                    )
                )
            imputed_atoms = int(
                query_rows["donor_ability_value_kind"]
                .eq(ability_policy.IMPUTED_VALUE_KIND)
                .sum()
            )
            declared.update(
                {
                    "proposal_atoms": int(len(seen_proposals)),
                    "complete_atoms": int(len(query_rows)),
                    "terminal_atoms": int(query_rows["terminal"].astype(bool).sum()),
                    "hazard_atoms": int(
                        query_rows["hazard_resolution_status"]
                        .eq("resolved_complete_atom")
                        .sum()
                    ),
                    "imputed_ability_atoms": imputed_atoms,
                    "quality_mass": float(query_rows["quality_base_mass"].sum()),
                    "terminal_mass": float(
                        query_rows.loc[
                            query_rows["terminal"].astype(bool), "quality_base_mass"
                        ].sum()
                    ),
                    "max_proposal_mass_residual": float(max(proposal_residuals)),
                    "current_chain_adapter_v3_ability_ready": bool(
                        query_rows["donor_ability_available"].astype(bool).all()
                    ),
                    "required_adapter_contract_bump": bool(imputed_atoms > 0),
                    "anchor_ready": False,
                    "anchor_hold_reason": complete._anchor_hold_reason(imputed_atoms),
                    "proposal_source_inventory_records": int(
                        proposal_hasher.records
                    ),
                    "proposal_source_inventory_sha256": proposal_hasher.hexdigest(),
                    "surface_source_inventory_records": int(surface_hasher.records),
                    "surface_source_inventory_sha256": surface_hasher.hexdigest(),
                    "hazard_source_inventory_records": int(hazard_hasher.records),
                    "hazard_source_inventory_sha256": hazard_hasher.hexdigest(),
                }
            )
            rebuilt_inventory.append(declared)
        manifest["inventory"] = rebuilt_inventory
        manifest["requests"] = len(rebuilt_inventory)
        manifest["complete_atoms"] = int(len(all_rows))
        manifest["source_inventory_sha256"] = complete._source_inventory_sha256(
            rebuilt_inventory
        )
    for part in manifest["parts"]:
        path = output / part["path"]
        frame = pd.read_parquet(path)
        part["rows"] = len(frame)
        part["bytes"] = path.stat().st_size
        part["byte_sha256"] = complete._sha256_file(path)
        part["logical_sha256"] = complete._persisted_frame_sha(frame)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    files = {}
    for path in sorted(output.rglob("*")):
        relative = path.relative_to(output).as_posix()
        if path.is_file() and relative != "OUTPUT_SHA256.json":
            files[relative] = {
                "sha256": complete._sha256_file(path),
                "bytes": path.stat().st_size,
            }
    (output / "OUTPUT_SHA256.json").write_text(
        json.dumps(
            {"contract": "non_circular_output_sha256_v1", "files": files},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _claim_fresh_self_derived_authority(output: Path) -> None:
    """Model a candidate trying to make its freshly resealed digest authority."""

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_inventory_authority_id"] = (
        "candidate_self_derived_" + str(manifest["source_inventory_sha256"])
    )
    manifest["source_inventory_authority_scope"] = "candidate_supplied"
    manifest["source_inventory_authority_ready"] = True
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    _reseal_artifact(output)


def test_synthetic_tee_expands_surface_and_exact_hazard_branches_with_crn_parity() -> None:
    proposal, numeric = _proposal_and_numeric(
        category="ott",
        masses=[0.4, 0.6],
        terminal=[False, False],
        points=np.asarray([[100.0, 0.0], [200.0, 0.0]]),
    )
    surface = FixedSurface(
        np.asarray(
            [
                [0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.25],
            ]
        )
    )
    resolver = _resolver("ott", (300.0, 600.0))
    model = fixed_reference.ReferenceValueModel.load(REFERENCE_DIR)
    query = _query(category="ott")
    request = complete.CompleteAtomRequest(proposal, numeric, query, surface, resolver, model)
    atoms, audit = _collect(request)

    assert len(atoms) == 27
    assert audit.complete_atoms == 27
    assert audit.hazard_atoms == 24
    assert audit.terminal_atoms == 0
    assert audit.quality_mass == pytest.approx(1.0, abs=complete.MASS_TOLERANCE)
    assert audit.current_chain_adapter_v3_ability_ready is True
    assert audit.anchor_ready is False
    assert atoms["complete_atom"].all()
    assert atoms["end_lie"].ne("penalty").all()
    assert atoms["quality_base_mass"].sum() == pytest.approx(1.0)
    assert np.allclose(
        atoms["quality_base_mass"],
        atoms["base_hierarchy_mass"] * atoms["spatial_model_weight"],
    )
    expected_reward = (
        atoms["start_reference_value"]
        - atoms["stroke_cost"]
        - atoms["end_reference_value"]
    )
    assert np.allclose(atoms["fixed_reference_reward"], expected_reward)
    by_proposal = atoms.groupby("proposal_row_order")["quality_base_mass"].sum()
    assert by_proposal.to_dict() == pytest.approx({0: 0.4, 1: 0.6})
    zero_boundary = atoms.loc[
        atoms["proposal_row_order"].eq(1)
        & atoms["surface_class_index"].eq(0)
    ]
    assert len(zero_boundary) == 1
    assert zero_boundary.iloc[0]["quality_base_mass"] == 0.0
    assert bool(zero_boundary.iloc[0]["boundary_only"]) is True
    assert json.loads(zero_boundary.iloc[0]["surface_full_probability_json"]) == [
        0.0,
        0.75,
        0.0,
        0.0,
        0.0,
        0.0,
        0.25,
    ]

    # Proposal and hazard stages retain side='right'.  Surface selection keeps
    # the parent's strict `u > cumulative` boundary behavior.
    selected = complete.select_complete_atom_crn(
        atoms,
        proposal_uniform=0.2,
        surface_uniform=0.6,
        hazard_uniform=0.42,
        holeout_uniform=0.9,
    )
    expected_hazard = resolver.resolve(
        endpoint_x=300.0,
        endpoint_y=0.0,
        hole_no=8,
        round_no=1,
        category="ott",
        common_uniform=0.42,
        forecast_pin_x=600.0,
        forecast_pin_y=0.0,
    )
    assert selected["proposal_row_order"] == 0
    assert selected["surface_class_label"] == "water"
    assert selected["hazard_donor_key"] == expected_hazard.donor_key

    proposal_boundary = complete.select_complete_atom_crn(
        atoms,
        proposal_uniform=0.4,
        surface_uniform=0.0,
        hazard_uniform=0.0,
        holeout_uniform=0.9,
    )
    assert proposal_boundary["proposal_row_order"] == 1
    surface_boundary = complete.select_complete_atom_crn(
        atoms,
        proposal_uniform=0.2,
        surface_uniform=0.5,
        hazard_uniform=0.0,
        holeout_uniform=0.9,
    )
    assert surface_boundary["surface_class_label"] == "fairway"
    leading_zero = complete.select_complete_atom_crn(
        atoms,
        proposal_uniform=0.8,
        surface_uniform=0.0,
        hazard_uniform=0.0,
        holeout_uniform=0.9,
    )
    assert bool(leading_zero["boundary_only"]) is True
    assert leading_zero["surface_class_label"] == "fairway"
    immediately_above_zero = complete.select_complete_atom_crn(
        atoms,
        proposal_uniform=0.8,
        surface_uniform=np.nextafter(0.0, 1.0),
        hazard_uniform=0.0,
        holeout_uniform=0.9,
    )
    assert bool(immediately_above_zero["boundary_only"]) is False
    assert immediately_above_zero["surface_class_label"] == "rough"


def test_explicit_app_holeouts_retain_mass_and_legacy_residual_selection() -> None:
    proposal, numeric = _proposal_and_numeric(
        category="app",
        masses=[0.3, 0.6, 0.04, 0.06],
        terminal=[False, False, True, True],
        points=np.asarray([[5.0, -2.0], [10.0, 1.0]]),
    )
    surface = FixedSurface(
        np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
    )
    resolver = _resolver("app", (600.0,))
    model = fixed_reference.ReferenceValueModel.load(REFERENCE_DIR)
    query = _query(category="app", distance=210.0)
    atoms, audit = _collect(
        complete.CompleteAtomRequest(proposal, numeric, query, surface, resolver, model)
    )
    terminal = atoms.loc[atoms["terminal"]]
    assert len(atoms) == 4
    assert len(terminal) == 2
    assert terminal["atom_kind"].eq("app_holeout").all()
    assert terminal["surface_class_label"].eq("explicit_app_holeout").all()
    assert terminal["end_state_key"].eq("terminal").all()
    assert terminal["stroke_cost"].eq(1).all()
    assert terminal["quality_base_mass"].sum() == pytest.approx(0.1)
    assert audit.terminal_mass == pytest.approx(0.1)
    assert audit.declared_app_holeout_probability == pytest.approx(0.1)

    selected_holeout = complete.select_complete_atom_crn(
        atoms,
        proposal_uniform=0.99,
        surface_uniform=0.99,
        hazard_uniform=0.99,
        holeout_uniform=0.02,
    )
    assert bool(selected_holeout["terminal"]) is True
    assert selected_holeout["proposal_row_order"] == 2
    selected_boundary = complete.select_complete_atom_crn(
        atoms,
        proposal_uniform=1.0 / 3.0,
        surface_uniform=0.0,
        hazard_uniform=0.0,
        holeout_uniform=0.1,
    )
    assert bool(selected_boundary["terminal"]) is False
    # 1/3 is just below the frozen numeric CDF because the spatial branch
    # normalization happened in the parent support, so no reconstruction is
    # allowed here.
    assert selected_boundary["proposal_row_order"] == 0
    frozen_boundary = float(numeric.cdf[0])
    selected_exact_frozen = complete.select_complete_atom_crn(
        atoms,
        proposal_uniform=frozen_boundary,
        surface_uniform=0.0,
        hazard_uniform=0.0,
        holeout_uniform=0.1,
    )
    assert selected_exact_frozen["proposal_row_order"] == 1
    selected_below_frozen = complete.select_complete_atom_crn(
        atoms,
        proposal_uniform=np.nextafter(frozen_boundary, 0.0),
        surface_uniform=0.0,
        hazard_uniform=0.0,
        holeout_uniform=0.1,
    )
    assert selected_below_frozen["proposal_row_order"] == 0


def test_beta_zero_replays_exact_53_bit_numeric_cdf() -> None:
    first_mass = np.nextafter(0.5, 1.0)
    proposal, numeric = _proposal_and_numeric(
        category="ott",
        masses=[float(first_mass), float(1.0 - first_mass)],
        terminal=[False, False],
        points=np.asarray([[100.0, 0.0], [200.0, 0.0]]),
    )
    grid_uniform = (2**52) / (2**53)
    assert grid_uniform == 0.5
    assert float(numeric.probability[0]) > grid_uniform
    frozen_cdf = np.asarray([grid_uniform, 1.0], dtype=float)
    payload_sha = streaming.support_payload_sha256(
        numeric.key,
        numeric.atom_ids,
        numeric.points,
        numeric.probability,
        frozen_cdf,
        numeric.holeout_probability,
    )
    numeric = replace(numeric, cdf=frozen_cdf, payload_sha256=payload_sha)
    proposal = proposal.copy()
    proposal["numeric_payload_sha256"] = payload_sha
    proposal["proposal_support_measure_sha256"] = proposal_ledger._frame_digest(
        proposal, proposal_ledger.PROPOSAL_SUPPORT_DIGEST_COLUMNS
    )
    atoms, _ = _collect(
        complete.CompleteAtomRequest(
            proposal,
            numeric,
            _query(category="ott"),
            FixedSurface(
                np.asarray(
                    [
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                )
            ),
            _resolver("ott", (300.0, 600.0)),
            fixed_reference.ReferenceValueModel.load(REFERENCE_DIR),
        )
    )
    assert atoms.loc[
        atoms["proposal_row_order"].eq(0), "proposal_branch_cdf_upper"
    ].iloc[0] > grid_uniform
    exact = complete.select_complete_atom_crn(
        atoms,
        proposal_uniform=grid_uniform,
        surface_uniform=0.0,
        hazard_uniform=0.0,
        holeout_uniform=0.9,
    )
    below = complete.select_complete_atom_crn(
        atoms,
        proposal_uniform=np.nextafter(grid_uniform, 0.0),
        surface_uniform=0.0,
        hazard_uniform=0.0,
        holeout_uniform=0.9,
    )
    assert exact["proposal_row_order"] == 1
    assert below["proposal_row_order"] == 0


def test_raw_hazard_right_cdf_matches_resolver_at_boundary_and_nextafter() -> None:
    proposal, numeric = _proposal_and_numeric(
        category="ott",
        masses=[1.0],
        terminal=[False],
        points=np.asarray([[100.0, 0.0]]),
    )
    resolver = _resolver("ott", (300.0,))
    atoms, _ = _collect(
        complete.CompleteAtomRequest(
            proposal,
            numeric,
            _query(category="ott"),
            FixedSurface(
                np.asarray([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
            ),
            resolver,
            fixed_reference.ReferenceValueModel.load(REFERENCE_DIR),
        )
    )
    hazard_rows = atoms.loc[
        atoms["hazard_resolution_status"].eq("resolved_complete_atom")
    ].sort_values("complete_row_order", kind="mergesort").reset_index(drop=True)
    boundary_index = 4
    boundary = float(hazard_rows.loc[boundary_index, "hazard_cdf_upper"])
    assert 0.0 < boundary < 1.0

    def selected(uniform: float) -> pd.Series:
        return complete.select_complete_atom_crn(
            atoms,
            proposal_uniform=0.0,
            surface_uniform=0.5,
            hazard_uniform=uniform,
            holeout_uniform=0.9,
        )

    exact = selected(boundary)
    below = selected(np.nextafter(boundary, 0.0))
    above = selected(np.nextafter(boundary, 1.0))
    parent = resolver.resolve(
        endpoint_x=300.0,
        endpoint_y=0.0,
        hole_no=8,
        round_no=1,
        category="ott",
        common_uniform=boundary,
        forecast_pin_x=600.0,
        forecast_pin_y=0.0,
    )
    assert below["hazard_donor_key"] == hazard_rows.loc[
        boundary_index, "hazard_donor_key"
    ]
    assert exact["hazard_donor_key"] == hazard_rows.loc[
        boundary_index + 1, "hazard_donor_key"
    ]
    assert above["hazard_donor_key"] == exact["hazard_donor_key"]
    assert exact["hazard_donor_key"] == parent.donor_key


def test_child_query_locator_binds_orientation_and_exact_end_state() -> None:
    query = _query(category="ott")
    rotated = replace(
        query,
        forecast_pin_x=0.0,
        forecast_pin_y=600.0,
        forecast_pin_long_yd=200.0,
    )
    model = fixed_reference.ReferenceValueModel.load(REFERENCE_DIR)
    state_key = model.state_key("fairway", 100.0)
    base = complete._next_support_locator(
        query=query,
        end_state_key=state_key,
        category="app",
        end_lie="fairway",
        end_distance_yd=100.0,
        end_endpoint=np.asarray([100.0, 0.0]),
    )
    orientation = complete._next_support_locator(
        query=rotated,
        end_state_key=state_key,
        category="app",
        end_lie="fairway",
        end_distance_yd=100.0,
        end_endpoint=np.asarray([100.0, 0.0]),
    )
    lateral = complete._next_support_locator(
        query=query,
        end_state_key=state_key,
        category="app",
        end_lie="fairway",
        end_distance_yd=100.0,
        end_endpoint=np.asarray([100.0, 1.0]),
    )
    assert len({base, orientation, lateral}) == 3


@pytest.mark.parametrize(
    ("limits", "message"),
    [
        ({"max_hazard_donors_per_surface": 5}, "Hazard donor fanout"),
        (
            {
                "max_hazard_donors_per_surface": 12,
                "max_complete_atoms_per_proposal": 12,
            },
            "proposal fanout",
        ),
    ],
)
def test_expansion_enforces_hazard_and_complete_proposal_fanout_bounds(
    limits: dict[str, int], message: str
) -> None:
    proposal, numeric = _proposal_and_numeric(
        category="ott",
        masses=[1.0],
        terminal=[False],
        points=np.asarray([[100.0, 0.0]]),
    )
    with pytest.raises(complete.CompleteAtomSupportError, match=message):
        expansion = complete.CompleteAtomExpansion(
            complete.CompleteAtomRequest(
                proposal,
                numeric,
                _query(category="ott"),
                FixedSurface(
                    np.asarray([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
                ),
                _resolver("ott", (300.0,)),
                fixed_reference.ReferenceValueModel.load(REFERENCE_DIR),
            ),
            max_batch_rows=1,
            **limits,
        )
        list(expansion.iter_batches())


def test_oversize_hazard_provider_rejects_before_hash_or_distribution_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposal, numeric = _proposal_and_numeric(
        category="ott",
        masses=[1.0],
        terminal=[False],
        points=np.asarray([[100.0, 0.0]]),
    )
    resolver = _resolver("ott", (300.0,))
    calls = {"hash": 0, "distribution": 0}

    def forbidden_hash(*args: object, **kwargs: object) -> str:
        calls["hash"] += 1
        raise AssertionError("resolver hashing ran before provider preflight")

    def forbidden_distribution(*args: object, **kwargs: object) -> object:
        calls["distribution"] += 1
        raise AssertionError("distribution materialized before provider preflight")

    monkeypatch.setattr(complete, "hazard_resolver_sha256", forbidden_hash)
    monkeypatch.setattr(resolver, "_distribution", forbidden_distribution)
    with pytest.raises(
        complete.CompleteAtomSupportError,
        match="provider preflight exceeded bound",
    ):
        complete.CompleteAtomExpansion(
            complete.CompleteAtomRequest(
                proposal,
                numeric,
                _query(category="ott"),
                FixedSurface(
                    np.asarray([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
                ),
                resolver,
                fixed_reference.ReferenceValueModel.load(REFERENCE_DIR),
            ),
            max_hazard_donors_per_surface=11,
        )
    assert calls == {"hash": 0, "distribution": 0}


def test_8193_hazard_bound_rejects_every_entry_before_copy_or_distribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _hazard_persistence_request()
    resolver = request.hazard_resolver
    original_hash = complete.hazard_resolver_sha256
    calls = {"copy": 0, "hash": 0, "distribution": 0}

    def forbidden_copy(*args: object, **kwargs: object) -> object:
        calls["copy"] += 1
        raise AssertionError("a DataFrame copy ran before the absolute bound check")

    def forbidden_hash(*args: object, **kwargs: object) -> str:
        calls["hash"] += 1
        raise AssertionError("resolver hashing ran before the absolute bound check")

    def forbidden_distribution(*args: object, **kwargs: object) -> object:
        calls["distribution"] += 1
        raise AssertionError("distribution ran before the absolute bound check")

    monkeypatch.setattr(pd.DataFrame, "copy", forbidden_copy)
    monkeypatch.setattr(complete, "hazard_resolver_sha256", forbidden_hash)
    monkeypatch.setattr(resolver, "_distribution", forbidden_distribution)

    with pytest.raises(ValueError, match="exceeds internal absolute ceiling"):
        complete.CompleteAtomExpansion(
            request, max_hazard_donors_per_surface=8_193
        )
    assert calls["hash"] == 0
    monkeypatch.setattr(complete, "hazard_resolver_sha256", original_hash)
    with pytest.raises(
        complete.CompleteAtomSupportError,
        match="exceeds internal absolute ceiling",
    ):
        complete._preflight_hazard_provider_count(
            resolver, max_donor_count=8_193
        )
    with pytest.raises(
        complete.CompleteAtomSupportError,
        match="exceeds internal absolute ceiling",
    ):
        complete.hazard_resolver_sha256(resolver, max_donor_count=8_193)
    output = tmp_path / "invalid-8193-persistence"
    with pytest.raises(ValueError, match="exceeds internal absolute ceiling"):
        complete.persist_complete_atom_requests(
            [request], output, max_hazard_donors_per_surface=8_193
        )
    assert not output.exists()
    assert calls == {"copy": 0, "hash": 0, "distribution": 0}


def test_policy_v2_imputation_is_preserved_but_current_anchor_adapter_is_hold() -> None:
    proposal, numeric = _proposal_and_numeric(
        category="app",
        masses=[0.8, 0.2],
        terminal=[False, True],
        points=np.asarray([[4.0, 0.0]]),
        availability=[False, True],
    )
    assert proposal.loc[0, "donor_ability_value_kind"] == ability_policy.IMPUTED_VALUE_KIND
    assert proposal.loc[0, "donor_ability_sg"] == 0.0
    assert bool(proposal.loc[0, "donor_ability_available"]) is False
    surface = FixedSurface(
        np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    )
    model = fixed_reference.ReferenceValueModel.load(REFERENCE_DIR)
    atoms, audit = _collect(
        complete.CompleteAtomRequest(
            proposal,
            numeric,
            _query(category="app", distance=210.0),
            surface,
            _resolver("app", (600.0,)),
            model,
        )
    )
    imputed = atoms.loc[atoms["proposal_row_order"].eq(0)]
    assert imputed["donor_ability_value_kind"].eq(
        ability_policy.IMPUTED_VALUE_KIND
    ).all()
    assert imputed["donor_ability_sg"].eq(0.0).all()
    assert not imputed["donor_ability_available"].any()
    assert audit.imputed_ability_atoms == len(imputed)
    assert audit.current_chain_adapter_v3_ability_ready is False
    assert audit.required_adapter_contract_bump is True
    assert audit.anchor_ready is False


def test_complete_rows_feed_whole_atom_weights_without_splitting_mechanics() -> None:
    proposal, numeric = _proposal_and_numeric(
        category="ott",
        masses=[0.4, 0.6],
        terminal=[False, False],
        points=np.asarray([[100.0, 0.0], [200.0, 0.0]]),
    )
    surface = FixedSurface(
        np.asarray(
            [
                [0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
    )
    atoms, _ = _collect(
        complete.CompleteAtomRequest(
            proposal,
            numeric,
            _query(category="ott"),
            surface,
            _resolver("ott", (300.0, 600.0)),
            fixed_reference.ReferenceValueModel.load(REFERENCE_DIR),
        )
    )
    parameters = anchor.TransitionQualityParameters(
        transfer_beta=(0.0, 0.0, 0.0, 0.0),
        target_ability_sg=(0.0, 0.0, 0.0, 0.0),
        lambda_correction=(0.0, 0.0, 0.0, 0.0),
        reward_reference_sha256="c" * 64,
    )
    weighted = anchor.whole_atom_probabilities(
        base_mass=atoms["quality_base_mass"],
        fixed_reference_reward=atoms["fixed_reference_reward"],
        donor_ability_sg=atoms["donor_ability_sg"],
        donor_group_id=atoms["donor_quality_group_id"],
        category="ott",
        parameters=parameters,
    )
    assert np.allclose(
        weighted.probability,
        atoms["quality_base_mass"].to_numpy(float),
        atol=2e-15,
        rtol=2e-15,
    )
    selected = complete.select_complete_atom_crn(
        atoms.assign(tilted_probability=weighted.probability),
        proposal_uniform=0.2,
        surface_uniform=0.7,
        hazard_uniform=0.3,
        holeout_uniform=0.9,
        probability_column="tilted_probability",
    )
    assert selected["hazard_resolution_status"] == "resolved_complete_atom"


def test_unsealed_or_tampered_ability_and_missing_provenance_fail_closed() -> None:
    proposal, numeric = _proposal_and_numeric(
        category="ott",
        masses=[1.0],
        terminal=[False],
        points=np.asarray([[100.0, 0.0]]),
    )
    surface = FixedSurface(np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))
    query = _query(category="ott")
    model = fixed_reference.ReferenceValueModel.load(REFERENCE_DIR)
    resolver = _resolver("ott", (300.0,))

    unsealed = proposal.drop(columns=list(ability_policy.OUTPUT_COLUMNS))
    with pytest.raises(complete.CompleteAtomSupportError, match="unsealed"):
        complete.CompleteAtomExpansion(
            complete.CompleteAtomRequest(unsealed, numeric, query, surface, resolver, model)
        )
    tampered = proposal.copy()
    tampered.loc[0, "donor_ability_row_sha256"] = "0" * 64
    with pytest.raises(complete.CompleteAtomSupportError, match="policy is invalid"):
        complete.CompleteAtomExpansion(
            complete.CompleteAtomRequest(tampered, numeric, query, surface, resolver, model)
        )
    missing_identity = proposal.copy()
    missing_identity.loc[0, "donor_event_key"] = ""
    with pytest.raises(complete.CompleteAtomSupportError):
        complete.CompleteAtomExpansion(
            complete.CompleteAtomRequest(missing_identity, numeric, query, surface, resolver, model)
        )
    with pytest.raises(complete.CompleteAtomSupportError, match="audited"):
        complete.CompleteAtomExpansion(
            complete.CompleteAtomRequest(proposal, numeric, query, surface, object(), model)  # type: ignore[arg-type]
        )
    with pytest.raises(complete.CompleteAtomSupportError, match="frozen Reference"):
        complete.CompleteAtomExpansion(
            complete.CompleteAtomRequest(proposal, numeric, query, surface, resolver, object())  # type: ignore[arg-type]
        )


def test_sharded_persistence_obeys_row_bound_and_verifies_parts(tmp_path: Path) -> None:
    output = tmp_path / "complete-support"
    manifest = complete.persist_complete_atom_requests(
        [_persistence_request()], output, expansion_batch_rows=1, max_rows_per_part=2
    )
    assert manifest["requests"] == 1
    assert manifest["peak_buffered_rows"] <= 2
    assert all(part["rows"] <= 2 for part in manifest["parts"])
    assert (output / "manifest.json").exists()
    assert (output / "OUTPUT_SHA256.json").exists()
    with pytest.raises(TypeError):
        complete.validate_persisted_complete_atom_artifact(
            output,
            expected_source_inventory_sha256=manifest["source_inventory_sha256"],
        )  # type: ignore[call-arg]
    validation = complete.validate_persisted_complete_atom_artifact(output)
    assert validation["complete_atoms"] == manifest["complete_atoms"]
    assert validation["all_policy_rows_exact"] is True
    assert validation["closed_source_inventory_authority_exact"] is True
    persisted = pd.concat(
        [pd.read_parquet(output / part["path"]) for part in manifest["parts"]],
        ignore_index=True,
    )
    assert len(persisted) == manifest["complete_atoms"]
    assert persisted["quality_base_mass"].sum() == pytest.approx(1.0)
    assert manifest["source_sha256_start"] == manifest["source_sha256_end"]
    assert manifest["dependency_sha256_start"] == manifest["dependency_sha256_end"]
    assert manifest["source_inventory_contract"] == complete.SOURCE_INVENTORY_CONTRACT
    assert len(manifest["source_inventory_sha256"]) == 64
    assert manifest["inventory"][0]["proposal_source_inventory_records"] == 2
    assert manifest["inventory"][0]["surface_source_inventory_records"] == 1
    assert manifest["source_inventory_authority_ready"] is True
    assert manifest["source_inventory_authority_scope"] == "synthetic_test_fixture_only"
    assert manifest["absolute_max_hazard_provider_donors"] == (
        complete.ABSOLUTE_MAX_HAZARD_PROVIDER_DONORS
    )
    assert {
        "tee",
        "app",
        "full_chain",
        "batched_population",
    }.issubset(manifest["dependency_sha256_end"])


def test_unminted_candidate_inventory_fails_closed_without_caller_unlock(
    tmp_path: Path,
) -> None:
    proposal, numeric = _proposal_and_numeric(
        category="app",
        masses=[0.9, 0.1],
        terminal=[False, True],
        points=np.asarray([[4.0, 0.0]]),
        availability=[False, True],
    )
    request = complete.CompleteAtomRequest(
        proposal,
        numeric,
        _query(category="app", distance=210.0),
        FixedSurface(
            np.asarray([[0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        ),
        _resolver("app", (600.0,)),
        fixed_reference.ReferenceValueModel.load(REFERENCE_DIR),
    )
    output = tmp_path / "unminted-candidate"
    manifest = complete.persist_complete_atom_requests(
        [request], output, expansion_batch_rows=8, max_rows_per_part=64
    )
    assert manifest["source_inventory_authority_ready"] is False
    assert manifest["source_inventory_authority_id"] == (
        complete.UNMINTED_SOURCE_INVENTORY_AUTHORITY_ID
    )
    with pytest.raises(
        complete.CompleteAtomSupportError,
        match="not admitted by the closed authority",
    ):
        complete.validate_persisted_complete_atom_artifact(output)


def test_h8_authority_record_review_evidence_and_lock_are_exact() -> None:
    records = complete.source_inventory_authority_records()
    plain = tuple(complete._plain_authority_tree(record) for record in records)
    assert len(plain) == 3
    assert [complete._stable_sha(record) for record in plain[:2]] == [
        "423abbc94be245f242036dfb3f606026f5fef8bbfec8197a5d6765b0dd1108bb",
        "c42d032d3e50bf6780e677eb841d376cbc6c1612e2ae1d31e90f020d8c0543a7",
    ]
    assert plain[2] == _EXPECTED_H8_AUTHORITY_RECORD
    assert complete._stable_sha(plain[2]) == _H8_AUTHORITY_RECORD_SHA256
    assert complete._SOURCE_PINNED_AUTHORITY_LOCK_SHA256 == (
        _ADMITTED_AUTHORITY_LOCK_SHA256
    )
    assert complete._stable_sha(
        {
            "contract": complete.SOURCE_INVENTORY_AUTHORITY_CONTRACT,
            "records": plain,
        }
    ) == _ADMITTED_AUTHORITY_LOCK_SHA256


def test_h8_query_projection_admits_exactly_and_one_field_changes_reject() -> None:
    records = complete.source_inventory_authority_records()
    h8_record = records[2]
    inventory = [dict(_EXPECTED_H8_QUERY_PROJECTION)]
    assert complete._source_inventory_authority(inventory) is h8_record
    for field, original in _EXPECTED_H8_QUERY_PROJECTION.items():
        changed = dict(_EXPECTED_H8_QUERY_PROJECTION)
        changed[field] = int(original) + 1 if isinstance(original, int) else "0" * 64
        assert complete._source_inventory_authority([changed]) is None, field


@pytest.mark.parametrize("path", _H8_AUTHORITY_MUTATION_PATHS)
def test_h8_authority_one_field_mutation_breaks_closed_lock(
    monkeypatch: pytest.MonkeyPatch, path: tuple[object, ...]
) -> None:
    mutable = [
        json.loads(json.dumps(complete._plain_authority_tree(record)))
        for record in complete.source_inventory_authority_records()
    ]
    target: object = mutable[2]
    for component in path[:-1]:
        target = target[component]  # type: ignore[index]
    field = path[-1]
    original = target[field]  # type: ignore[index]
    target[field] = int(original) + 1 if isinstance(original, int) else "0" * 64  # type: ignore[index]
    mutated = tuple(
        complete._freeze_authority_tree(record) for record in mutable
    )
    monkeypatch.setattr(complete, "_SOURCE_PINNED_AUTHORITY_RECORDS", mutated)
    with pytest.raises(
        complete.CompleteAtomSupportError,
        match="Closed source-inventory authority lock changed",
    ):
        complete._closed_source_inventory_authority_records()


def test_h8_data_only_authority_delta_reverses_to_reviewed_seam() -> None:
    source = Path(complete.__file__).read_text(encoding="utf-8")
    assert sha256(source.encode("utf-8")).hexdigest() == _ADMITTED_SEAM_SHA256
    record_pattern = re.compile(
        r'        \{\n'
        r'            "authority_id": "lowry_r1h8_raw_app_exact_query_v1",\n'
        r'.*?'
        r'\n        \},\n'
        r'(?=    \)\n\)\n_SOURCE_PINNED_AUTHORITY_LOCK_SHA256)',
        flags=re.DOTALL,
    )
    assert len(record_pattern.findall(source)) == 1
    reconstructed = record_pattern.sub("", source, count=1)
    assert reconstructed.count(_ADMITTED_AUTHORITY_LOCK_SHA256) == 1
    reconstructed = reconstructed.replace(
        _ADMITTED_AUTHORITY_LOCK_SHA256,
        _PRE_ADMISSION_AUTHORITY_LOCK_SHA256,
        1,
    )
    assert sha256(reconstructed.encode("utf-8")).hexdigest() == (
        _PRE_ADMISSION_SEAM_SHA256
    )


def test_complete_seam_exposes_no_candidate_or_self_digest_unlock_api() -> None:
    assert tuple(
        inspect.signature(complete._source_inventory_authority).parameters
    ) == ("inventory",)
    assert tuple(
        inspect.signature(complete.persist_complete_atom_requests).parameters
    ) == (
        "requests",
        "output_dir",
        "expansion_batch_rows",
        "max_rows_per_part",
        "max_hazard_donors_per_surface",
        "max_complete_atoms_per_proposal",
    )
    assert tuple(
        inspect.signature(
            complete.validate_persisted_complete_atom_artifact
        ).parameters
    ) == ("output_dir", "require_current_source")
    forbidden_public_tokens = ("admit", "candidate", "mint", "unlock")
    assert not any(
        token in exported.casefold()
        for exported in complete.__all__
        if callable(getattr(complete, exported))
        for token in forbidden_public_tokens
    )


def test_closed_authority_records_are_recursively_immutable() -> None:
    public_records = complete.source_inventory_authority_records()
    private_records = complete._SOURCE_PINNED_AUTHORITY_RECORDS
    assert public_records is private_records
    public_record = public_records[0]
    public_query = public_record["queries"][0]
    with pytest.raises(TypeError):
        public_record["authority_id"] = "mutated"  # type: ignore[index]
    with pytest.raises(TypeError):
        public_query["query_key"] = "mutated"  # type: ignore[index]
    with pytest.raises(TypeError):
        private_records[0]["scope"] = "mutated"  # type: ignore[index]
    with pytest.raises(TypeError):
        private_records[0]["queries"][0]["query_key"] = "mutated"  # type: ignore[index]
    with pytest.raises(TypeError):
        private_records[2]["review_evidence"]["audit_id"] = "mutated"  # type: ignore[index]
    with pytest.raises(TypeError):
        private_records[2]["queries"][0]["query_key"] = "mutated"  # type: ignore[index]
    with pytest.raises(TypeError):
        public_records[0] = public_records[1]  # type: ignore[index]


def test_persisted_validation_rederives_factorization_after_hash_reseal(
    tmp_path: Path,
) -> None:
    output = tmp_path / "factor-tamper"
    manifest = complete.persist_complete_atom_requests(
        [_persistence_request()], output, expansion_batch_rows=4, max_rows_per_part=64
    )
    path = output / manifest["parts"][0]["path"]
    frame = pd.read_parquet(path)
    positive_index = frame.index[frame["quality_base_mass"].gt(0.0)][0]
    frame.loc[positive_index, "mechanics_conditional_probability"] += 0.01
    frame.to_parquet(path, index=False)
    _reseal_artifact(output)
    with pytest.raises(
        complete.CompleteAtomSupportError,
        match="proposal×surface×hazard factorization",
    ):
        complete.validate_persisted_complete_atom_artifact(output)


def test_persisted_validation_rederives_raw_hazard_cdf_after_hash_reseal(
    tmp_path: Path,
) -> None:
    proposal, numeric = _proposal_and_numeric(
        category="ott",
        masses=[1.0],
        terminal=[False],
        points=np.asarray([[100.0, 0.0]]),
    )
    request = complete.CompleteAtomRequest(
        proposal,
        numeric,
        _query(category="ott"),
        FixedSurface(np.asarray([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])),
        _resolver("ott", (300.0,)),
        fixed_reference.ReferenceValueModel.load(REFERENCE_DIR),
    )
    output = tmp_path / "hazard-cdf-tamper"
    manifest = complete.persist_complete_atom_requests(
        [request], output, expansion_batch_rows=32, max_rows_per_part=64
    )
    path = output / manifest["parts"][0]["path"]
    frame = pd.read_parquet(path)
    hazard_index = frame.index[
        frame["hazard_resolution_status"].eq("resolved_complete_atom")
    ].tolist()
    assert len(hazard_index) >= 2
    changed = np.nextafter(
        float(frame.loc[hazard_index[0], "hazard_cdf_upper"]), 1.0
    )
    frame.loc[hazard_index[0], "hazard_cdf_upper"] = changed
    frame.loc[hazard_index[1], "hazard_cdf_lower"] = changed
    frame.to_parquet(path, index=False)
    _reseal_artifact(output)
    with pytest.raises(
        complete.CompleteAtomSupportError,
        match="raw hazard CDF width",
    ):
        complete.validate_persisted_complete_atom_artifact(output)


def test_persisted_validation_checks_policy_on_every_expanded_row_after_reseal(
    tmp_path: Path,
) -> None:
    output = tmp_path / "policy-tamper"
    manifest = complete.persist_complete_atom_requests(
        [_persistence_request()], output, expansion_batch_rows=4, max_rows_per_part=64
    )
    path = output / manifest["parts"][0]["path"]
    frame = pd.read_parquet(path)
    proposal_rows = frame.index[frame["proposal_row_order"].eq(0)].tolist()
    assert len(proposal_rows) >= 2
    frame.loc[proposal_rows[1], "donor_ability_row_sha256"] = "0" * 64
    frame.to_parquet(path, index=False)
    _reseal_artifact(output)
    with pytest.raises(
        complete.CompleteAtomSupportError,
        match="donor ability attestation",
    ):
        complete.validate_persisted_complete_atom_artifact(output)


def test_persisted_validation_rejects_dependency_start_end_tamper_after_reseal(
    tmp_path: Path,
) -> None:
    output = tmp_path / "dependency-tamper"
    original_manifest = complete.persist_complete_atom_requests(
        [_persistence_request()], output, expansion_batch_rows=4, max_rows_per_part=64
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dependency_sha256_end"]["tee"] = "0" * 64
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _reseal_artifact(output)
    with pytest.raises(
        complete.CompleteAtomSupportError,
        match="dependency attestation",
    ):
        complete.validate_persisted_complete_atom_artifact(output)


@pytest.mark.parametrize("reserved_kind", ["file", "directory"])
def test_persisted_closure_rejects_nested_output_index_reserved_name(
    tmp_path: Path, reserved_kind: str
) -> None:
    output = tmp_path / f"nested-index-{reserved_kind}"
    manifest = complete.persist_complete_atom_requests(
        [_persistence_request()], output, expansion_batch_rows=4, max_rows_per_part=64
    )
    nested = output / "untrusted" / "deeper" / "OUTPUT_SHA256.json"
    nested.parent.mkdir(parents=True)
    if reserved_kind == "file":
        nested.write_text("{}\n", encoding="utf-8")
    else:
        nested.mkdir()
    with pytest.raises(
        complete.CompleteAtomSupportError,
        match="nested reserved name",
    ):
        complete.validate_persisted_complete_atom_artifact(output)


def test_persisted_validation_rejects_valid_policy_row_swapped_across_proposals(
    tmp_path: Path,
) -> None:
    output = tmp_path / "swapped-policy"
    original = complete.persist_complete_atom_requests(
        [_persistence_request()], output, expansion_batch_rows=8, max_rows_per_part=64
    )
    path = output / original["parts"][0]["path"]
    frame = pd.read_parquet(path)
    target = frame.index[frame["proposal_row_order"].eq(0)].tolist()[1]
    source = frame.index[frame["proposal_row_order"].eq(1)].tolist()[0]
    raw = {
        column: frame.loc[source, column] for column in ability_policy.INPUT_COLUMNS
    }
    raw["bridge_row_id"] = complete._proposal_bridge_row_id(
        frame.loc[target, "proposal_support_key"],
        frame.loc[target, "proposal_row_order"],
        frame.loc[target, "proposal_atom_id"],
    )
    evaluated = ability_policy.apply_frozen_policy(pd.DataFrame([raw]))
    for column in (*ability_policy.INPUT_COLUMNS, *ability_policy.OUTPUT_COLUMNS):
        frame.loc[target, column] = evaluated.loc[0, column]
    frame.loc[target, "donor_ability_snapshot_kind"] = frame.loc[
        source, "donor_ability_snapshot_kind"
    ]
    values = frame.loc[target].to_dict()
    frame.loc[target, "proposal_policy_binding_sha256"] = (
        complete._proposal_policy_binding_sha256(values)
    )
    values = frame.loc[target].to_dict()
    frame.loc[target, "donor_quality_group_id"] = (
        complete.chain_adapter.canonical_donor_quality_group_id(values)
    )
    frame.to_parquet(path, index=False)
    _reseal_artifact(output, recompute_semantic_inventory=True)
    attacker = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert attacker["source_inventory_sha256"] == original["source_inventory_sha256"]
    with pytest.raises(
        complete.CompleteAtomSupportError,
        match="proposal attestation varies",
    ):
        complete.validate_persisted_complete_atom_artifact(output)


def test_persisted_validation_rejects_self_authorized_removed_hazard_donor(
    tmp_path: Path,
) -> None:
    output = tmp_path / "removed-hazard-donor"
    original = complete.persist_complete_atom_requests(
        [_hazard_persistence_request()],
        output,
        expansion_batch_rows=32,
        max_rows_per_part=64,
    )
    path = output / original["parts"][0]["path"]
    frame = pd.read_parquet(path)
    hazard_indices = frame.index[
        frame["hazard_resolution_status"].eq("resolved_complete_atom")
    ].tolist()
    assert len(hazard_indices) == 12
    frame = frame.drop(index=hazard_indices[4]).reset_index(drop=True)
    hazard_indices = frame.index[
        frame["hazard_resolution_status"].eq("resolved_complete_atom")
    ].tolist()
    probability = frame.loc[hazard_indices, "hazard_probability"].to_numpy(float)
    probability /= probability.sum()
    frame.loc[hazard_indices, "hazard_probability"] = probability
    frame.loc[hazard_indices, "hazard_donor_order"] = np.arange(
        len(hazard_indices), dtype=np.int64
    )
    surface_probability = frame.loc[
        hazard_indices, "surface_probability"
    ].to_numpy(float)
    mechanics = surface_probability * probability
    frame.loc[hazard_indices, "mechanics_conditional_probability"] = mechanics
    frame.loc[hazard_indices, "base_hierarchy_mass"] = (
        frame.loc[hazard_indices, "proposal_base_hierarchy_mass"].to_numpy(float)
        * mechanics
    )
    frame.loc[hazard_indices, "quality_base_mass"] = (
        frame.loc[hazard_indices, "proposal_probability"].to_numpy(float)
        * mechanics
    )
    frame = _recompute_expanded_cdfs(frame)
    frame.to_parquet(path, index=False)
    _reseal_artifact(output, recompute_semantic_inventory=True)
    _claim_fresh_self_derived_authority(output)
    attacker = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert attacker["source_inventory_sha256"] != original["source_inventory_sha256"]
    with pytest.raises(
        complete.CompleteAtomSupportError,
        match="not admitted by the closed authority",
    ):
        complete.validate_persisted_complete_atom_artifact(output)


def test_persisted_validation_rejects_self_authorized_renamed_hazard_donor(
    tmp_path: Path,
) -> None:
    output = tmp_path / "forged-hazard-key"
    original = complete.persist_complete_atom_requests(
        [_hazard_persistence_request()],
        output,
        expansion_batch_rows=32,
        max_rows_per_part=64,
    )
    path = output / original["parts"][0]["path"]
    frame = pd.read_parquet(path)
    target = frame.index[
        frame["hazard_resolution_status"].eq("resolved_complete_atom")
    ].tolist()[-1]
    frame.loc[target, "hazard_donor_key"] = (
        str(frame.loc[target, "hazard_donor_key"]) + "z"
    )
    values = frame.loc[target].to_dict()
    frame.loc[target, "atom_id"] = complete._complete_atom_id(values)
    values = frame.loc[target].to_dict()
    frame.loc[target, "donor_quality_group_id"] = (
        complete.chain_adapter.canonical_donor_quality_group_id(values)
    )
    frame.to_parquet(path, index=False)
    _reseal_artifact(output, recompute_semantic_inventory=True)
    _claim_fresh_self_derived_authority(output)
    with pytest.raises(
        complete.CompleteAtomSupportError,
        match="not admitted by the closed authority",
    ):
        complete.validate_persisted_complete_atom_artifact(output)


def test_persisted_validation_rejects_self_authorized_rewritten_surface_vector(
    tmp_path: Path,
) -> None:
    output = tmp_path / "rewritten-surface"
    original = complete.persist_complete_atom_requests(
        [_persistence_request()], output, expansion_batch_rows=8, max_rows_per_part=64
    )
    path = output / original["parts"][0]["path"]
    frame = pd.read_parquet(path)
    remove = frame.index[
        frame["proposal_row_order"].eq(0)
        & frame["surface_class_index"].eq(1)
    ].tolist()
    assert len(remove) == 1
    frame = frame.drop(index=remove).reset_index(drop=True)
    target = frame.index[frame["proposal_row_order"].eq(0)].tolist()[0]
    class_count = int(frame.loc[target, "surface_class_count"])
    probability = np.zeros(class_count, dtype=float)
    probability[0] = 1.0
    frame.loc[target, "surface_full_probability_json"] = json.dumps(
        probability.tolist(), separators=(",", ":")
    )
    frame.loc[target, "surface_full_cdf_upper_json"] = json.dumps(
        np.cumsum(probability).tolist(), separators=(",", ":")
    )
    frame.loc[target, "surface_probability"] = 1.0
    frame.loc[target, "surface_cdf_lower"] = 0.0
    frame.loc[target, "surface_cdf_upper"] = 1.0
    frame.loc[target, "mechanics_conditional_probability"] = 1.0
    frame.loc[target, "base_hierarchy_mass"] = frame.loc[
        target, "proposal_base_hierarchy_mass"
    ]
    frame.loc[target, "quality_base_mass"] = frame.loc[
        target, "proposal_probability"
    ]
    frame = _recompute_expanded_cdfs(frame)
    frame.to_parquet(path, index=False)
    _reseal_artifact(output, recompute_semantic_inventory=True)
    _claim_fresh_self_derived_authority(output)
    with pytest.raises(
        complete.CompleteAtomSupportError,
        match="not admitted by the closed authority",
    ):
        complete.validate_persisted_complete_atom_artifact(output)


def test_persisted_validation_rederives_readiness_and_rejects_go_flip(
    tmp_path: Path,
) -> None:
    output = tmp_path / "readiness-go-flip"
    original = complete.persist_complete_atom_requests(
        [_persistence_request()], output, expansion_batch_rows=8, max_rows_per_part=64
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inventory"][0]["anchor_ready"] = True
    manifest["inventory"][0]["anchor_hold_reason"] = ""
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _reseal_artifact(output)
    with pytest.raises(complete.CompleteAtomSupportError, match="readiness changed"):
        complete.validate_persisted_complete_atom_artifact(output)


@pytest.mark.parametrize("dependency", ["full_chain", "batched_population"])
def test_persisted_validation_rejects_missing_direct_runtime_dependency(
    tmp_path: Path, dependency: str
) -> None:
    output = tmp_path / f"missing-{dependency}"
    original = complete.persist_complete_atom_requests(
        [_persistence_request()], output, expansion_batch_rows=8, max_rows_per_part=64
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dependency_sha256_start"].pop(dependency)
    manifest["dependency_sha256_end"].pop(dependency)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _reseal_artifact(output)
    with pytest.raises(
        complete.CompleteAtomSupportError, match="dependency attestation"
    ):
        complete.validate_persisted_complete_atom_artifact(output)


def test_persisted_validation_rejects_8193_hazard_ceiling_after_reseal(
    tmp_path: Path,
) -> None:
    output = tmp_path / "persisted-8193-ceiling"
    complete.persist_complete_atom_requests(
        [_persistence_request()], output, expansion_batch_rows=8, max_rows_per_part=64
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["max_hazard_donors_per_surface"] = 8_193
    manifest["absolute_max_hazard_provider_donors"] = 8_193
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _reseal_artifact(output)
    with pytest.raises(
        complete.CompleteAtomSupportError,
        match="exceeds internal absolute ceiling",
    ):
        complete.validate_persisted_complete_atom_artifact(output)
