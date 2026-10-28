"""Archive-only real-kernel diagnostic for the explicit R1 H8 APP holeout.

This runner exercises five predeclared counter-particle identities through the
audited traced seam and its unmodified streaming parent.  Each identity is
known to fall below the real raw H8 APP support's 0.002 holeout threshold.  No
particle search, filtering, reroll, or broad simulation occurs here.

The output is a bounded mechanics/provenance diagnostic.  It does not alter
the frozen v6 seam, the all-18 artifact, or any production source, and it does
not authorize an anchor or production use.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping

import numpy as np
import pandas as pd


ARCHIVE = Path(__file__).resolve().parent
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

import hole_sim_normalized_chain_streaming as streaming  # noqa: E402
import hole_sim_streaming_transition_ledger as ledger_v6  # noqa: E402


CONTRACT = "streaming_real_app_holeout_r1h8_raw_diagnostic_v1"
FROZEN_V6_SOURCE_SHA256 = (
    "11b14109b938cda5da4b4eaae4d00eb43f9d44074fdbd9e48328a4f889d7ab97"
)
TARGET_EVENT_KEY = "pga:R2026034"
TARGET_PLAYER_ID = "33204"
ROUND_NO = 1
HOLE_NO = 8
ARM = "raw"
PARTICLE_IDS = np.asarray([345, 1915, 2461, 2573, 3382], dtype=np.int64)
COUNTER_NAMESPACE_SIZE = 50_000
EXPECTED_SUPPORT_KEY = "app|raw|R1|H8|tee|B14|33204"
EXPECTED_TERMINAL_ATOM_ID = (
    "9e5a473dd8b13b2a2da9827522ef029dd5127e339a02dde63c3c6accbc0ecb46"
)
EXPECTED_SUPPORT_ATOMS = 904
EXPECTED_SPATIAL_ATOMS = 903
EXPECTED_TERMINAL_PROPOSAL_ROW = 903
EXPECTED_HOLEOUT_PROBABILITY = 0.002
EXPECTED_PROPOSAL_CDF_LOWER = 0.9979999999999989
EXPECTED_DONOR_EVENT_KEY = "pga:R2023034"
EXPECTED_DONOR_PLAYER_ID = "28237"
EXPECTED_DONOR_PLAYER_NAME = "mcilroy, rory"
EXPECTED_DONOR_ROUND_NO = 1
EXPECTED_DONOR_HOLE_NO = 8
EXPECTED_DONOR_STROKE_NO = 1
EXPECTED_DONOR_ABILITY_SG = 0.65
EXPECTED_HOLEOUT_UNIFORMS = np.asarray(
    [
        0.00014852358789446685,
        0.0017673237459028823,
        0.0011182335379316677,
        0.0008559955261780439,
        0.0008180077005007225,
    ],
    dtype=float,
)
DEFAULT_OUTPUT_DIR = Path(
    "permanent_data/"
    "hole_sim_streaming_transition_ledger_lowry_r1h8_raw_app_holeout_5_v1"
)
OUTPUT_PAYLOAD_FILES = (
    "REPORT.md",
    "counter_uniform_replay.parquet",
    "explicit_holeout_selection.parquet",
    "explicit_holeout_transition_ledger.parquet",
    "manifest.json",
    "particle_telescope_audit.parquet",
    "proposal_support_atoms.parquet",
    "proposal_support_inventory.parquet",
    "traced_parent_differential.parquet",
    "traced_plain_results.parquet",
)


def _sha256(path: Path) -> str:
    return streaming.sha256_file(Path(path))


def _dependency_paths() -> dict[str, Path]:
    """Return the complete local executable dependency closure."""

    paths = dict(streaming.dependency_paths())
    paths.update(
        {
            "frozen_v6_ledger_source": Path(ledger_v6.__file__),
            "streaming_source": Path(streaming.__file__),
            "traced_population_source": Path(
                ledger_v6.traced_population.__file__
            ),
            "precompute_source": Path(ledger_v6.precompute.__file__),
            "asof_source": Path(ledger_v6.asof.__file__),
            "app_source": Path(ledger_v6.app.__file__),
            "fixed_reference_source": Path(
                ledger_v6.fixed_reference.__file__
            ),
            "chain_adapter_source": Path(ledger_v6.chain_adapter.__file__),
            "production_target_source": Path(
                ledger_v6.production_targets.__file__
            ),
        }
    )
    return paths


def _hash_paths(paths: Mapping[str, Path]) -> dict[str, str]:
    return {name: _sha256(path) for name, path in paths.items()}


def _assert_frozen_v6_source() -> None:
    observed = _sha256(Path(ledger_v6.__file__))
    if observed != FROZEN_V6_SOURCE_SHA256:
        raise RuntimeError(
            "The audited v6 trace seam changed; "
            f"expected={FROZEN_V6_SOURCE_SHA256}, observed={observed}"
        )


def _validated_particle_ids(particle_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(particle_ids, dtype=np.int64)
    if ids.ndim != 1 or not np.array_equal(ids, PARTICLE_IDS):
        raise ValueError(
            "The explicit-holeout diagnostic requires exactly particle IDs "
            f"{PARTICLE_IDS.tolist()} in frozen order"
        )
    if len(np.unique(ids)) != len(ids):
        raise AssertionError("The frozen diagnostic particle IDs are duplicated")
    return ids


def _select_exact_record(grid: pd.DataFrame) -> object:
    """Select the unique real Lowry R1 H8 par-3 initializer record."""

    required = {
        "event_key",
        "source_player_id",
        "round_no",
        "hole_no",
        "par",
        "forecast_pin_long_yd",
    }
    missing = sorted(required.difference(grid.columns))
    if missing:
        raise ValueError(f"Prepared grid misses H8 diagnostic columns={missing}")
    selected = grid.loc[
        grid["event_key"].astype(str).eq(TARGET_EVENT_KEY)
        & grid["source_player_id"].astype(str).eq(TARGET_PLAYER_ID)
        & grid["round_no"].eq(ROUND_NO)
        & grid["hole_no"].eq(HOLE_NO)
    ]
    if len(selected) != 1:
        raise ValueError(
            "The real Lowry R1 H8 grid record is absent or duplicated: "
            f"rows={len(selected)}"
        )
    record = next(selected.itertuples(index=False))
    if int(record.par) != 3:
        raise AssertionError("The frozen H8 initializer is no longer a par 3")
    distance_bucket = max(
        1, int(np.rint(float(record.forecast_pin_long_yd) / 15.0))
    )
    if distance_bucket != 14:
        raise AssertionError(
            f"The frozen H8 initializer bucket changed: B{distance_bucket}"
        )
    return record


def regenerate_initial_app_uniforms(
    particle_ids: np.ndarray = PARTICLE_IDS,
    *,
    seed: int = streaming.DEFAULT_SEED,
) -> pd.DataFrame:
    """Regenerate all three initial APP streams for the exact five IDs."""

    ids = _validated_particle_ids(particle_ids)
    output = pd.DataFrame({"particle_id": ids})
    for stream in ("app_index", "app_lie", "app_holeout"):
        output[f"expected_rng_{stream}_uniform"] = (
            streaming.hook.counter_uniforms(
                seed=int(seed),
                event_key=TARGET_EVENT_KEY,
                player_id=TARGET_PLAYER_ID,
                round_no=ROUND_NO,
                hole_no=HOLE_NO,
                transition_index=-1,
                stream=f"initial_{stream}",
                particle_ids=ids,
            )
        )
    observed_holeout = output[
        "expected_rng_app_holeout_uniform"
    ].to_numpy(float)
    if not np.array_equal(observed_holeout, EXPECTED_HOLEOUT_UNIFORMS):
        raise AssertionError("The predeclared H8 holeout counter uniforms changed")
    if not (observed_holeout < EXPECTED_HOLEOUT_PROBABILITY).all():
        raise AssertionError("A predeclared particle no longer selects holeout")
    output["declared_app_base_holeout_probability"] = (
        EXPECTED_HOLEOUT_PROBABILITY
    )
    output["expected_rng_holeout_residual_uniform"] = (
        observed_holeout / EXPECTED_HOLEOUT_PROBABILITY
    )
    return output


def _support_snapshot(
    repository: ledger_v6.IdentityPreservingSupportRepository,
) -> dict[str, tuple[str, ...]]:
    return {
        "numeric": tuple(sorted(map(str, repository.supports))),
        "explicit_app": tuple(
            sorted(map(str, repository.explicit_app_supports))
        ),
    }


def run_traced_plain_pair(
    *,
    traced_chain: ledger_v6.TracedStreamingShotChain,
    plain_chain: streaming.StreamingShotChain,
    repository: ledger_v6.IdentityPreservingSupportRepository,
    record: object,
    particle_ids: np.ndarray = PARTICLE_IDS,
    seed: int = streaming.DEFAULT_SEED,
    continuation_seed: int = streaming.DEFAULT_CONTINUATION_SEED,
) -> tuple[
    ledger_v6.TracedHoleExecution,
    streaming.StreamingHoleResult,
    dict[str, object],
    dict[str, object],
]:
    """Run exactly five traced IDs, then replay the unmodified parent."""

    ids = _validated_particle_ids(particle_ids)
    execution = traced_chain.simulate_hole_with_ledger(
        record=record,
        arm=ARM,
        particle_ids=ids,
        total_particles=COUNTER_NAMESPACE_SIZE,
        rng_mode="counter",
        seed=int(seed),
        continuation_seed=int(continuation_seed),
    )
    before = _support_snapshot(repository)
    plain_result, _ = plain_chain.simulate_hole(
        record=record,
        arm=ARM,
        particle_ids=ids,
        total_particles=COUNTER_NAMESPACE_SIZE,
        rng_mode="counter",
        seed=int(seed),
        continuation_seed=int(continuation_seed),
    )
    parity = ledger_v6._assert_hole_results_exact(
        execution.result,
        plain_result,
        arm=ARM,
    )
    after = _support_snapshot(repository)
    if after != before:
        raise AssertionError(
            "Plain-parent H8 replay discovered support absent from trace"
        )
    if before != {
        "numeric": (EXPECTED_SUPPORT_KEY,),
        "explicit_app": (EXPECTED_SUPPORT_KEY,),
    }:
        raise AssertionError(f"H8 materialized support set changed: {before}")
    return execution, plain_result, parity, {
        "exact_key_sets": True,
        "before": {name: list(keys) for name, keys in before.items()},
        "after": {name: list(keys) for name, keys in after.items()},
    }


def _with_target_context(frame: pd.DataFrame, *, event_start: object) -> pd.DataFrame:
    output = frame.copy()
    output.insert(0, "target_event_key", TARGET_EVENT_KEY)
    output.insert(1, "target_event_start", pd.Timestamp(event_start))
    output.insert(2, "target_player_id", TARGET_PLAYER_ID)
    output.insert(3, "round_no", ROUND_NO)
    output.insert(4, "hole_no", HOLE_NO)
    return output


def validate_exact_terminal_ledger(ledger: pd.DataFrame) -> dict[str, object]:
    """Fail closed on every causal field of the five explicit terminals."""

    required = {
        "particle_id",
        "atom_sequence",
        "arm",
        "category",
        "support_key",
        "atom_id",
        "selected_spatial_proposal_atom_id",
        "donor_event_key",
        "donor_player_id",
        "donor_player_name",
        "donor_round_no",
        "donor_hole_no",
        "donor_stroke_no",
        "donor_ability_sg",
        "donor_ability_available",
        "start_lie",
        "end_lie",
        "causal_strokes",
        "penalty_strokes",
        "stroke_cost",
        "terminal",
        "holed",
        "legacy_holeout_decision",
        "declared_app_base_holeout_probability",
        "rng_shot_transition_index",
        "rng_holeout_uniform",
        "rng_holeout_residual_uniform",
        "start_reference_value",
        "end_reference_value",
        "fixed_reference_reward",
    }
    missing = sorted(required.difference(ledger.columns))
    if missing:
        raise ValueError(f"Explicit-holeout ledger misses columns={missing}")
    ordered = ledger.sort_values(
        ["particle_id", "atom_sequence"], kind="mergesort"
    ).reset_index(drop=True)
    if len(ordered) != len(PARTICLE_IDS):
        raise AssertionError("Each exact H8 path must contain one causal atom")
    if not np.array_equal(ordered["particle_id"].to_numpy(np.int64), PARTICLE_IDS):
        raise AssertionError("The exact H8 ledger particle axis changed")
    exact_values: tuple[tuple[str, object], ...] = (
        ("atom_sequence", 0),
        ("arm", ARM),
        ("category", "app"),
        ("support_key", EXPECTED_SUPPORT_KEY),
        ("atom_id", EXPECTED_TERMINAL_ATOM_ID),
        ("donor_event_key", EXPECTED_DONOR_EVENT_KEY),
        ("donor_player_id", EXPECTED_DONOR_PLAYER_ID),
        ("donor_player_name", EXPECTED_DONOR_PLAYER_NAME),
        ("donor_round_no", EXPECTED_DONOR_ROUND_NO),
        ("donor_hole_no", EXPECTED_DONOR_HOLE_NO),
        ("donor_stroke_no", EXPECTED_DONOR_STROKE_NO),
        ("start_lie", "tee"),
        ("end_lie", "hole"),
        ("causal_strokes", 1),
        ("penalty_strokes", 0),
        ("stroke_cost", 1),
        ("terminal", True),
        ("holed", True),
        ("legacy_holeout_decision", True),
        ("rng_shot_transition_index", -1),
        ("donor_ability_available", True),
    )
    for column, expected in exact_values:
        if not ordered[column].eq(expected).all():
            raise AssertionError(
                f"Explicit H8 terminal field changed: {column} != {expected}"
            )
    if ordered["selected_spatial_proposal_atom_id"].astype(str).eq("").any():
        raise AssertionError("An explicit terminal lost its causal spatial selection")
    for column, expected in (
        ("declared_app_base_holeout_probability", EXPECTED_HOLEOUT_PROBABILITY),
        ("donor_ability_sg", EXPECTED_DONOR_ABILITY_SG),
    ):
        if not np.array_equal(
            ordered[column].to_numpy(float),
            np.full(len(ordered), expected, dtype=float),
        ):
            raise AssertionError(f"Explicit H8 terminal scalar changed: {column}")
    if not np.array_equal(
        ordered["rng_holeout_uniform"].to_numpy(float),
        EXPECTED_HOLEOUT_UNIFORMS,
    ):
        raise AssertionError("Explicit H8 ledger holeout uniforms changed")
    expected_residual = EXPECTED_HOLEOUT_UNIFORMS / EXPECTED_HOLEOUT_PROBABILITY
    if not np.array_equal(
        ordered["rng_holeout_residual_uniform"].to_numpy(float),
        expected_residual,
    ):
        raise AssertionError("Explicit H8 residual uniforms changed")
    expected_reward = (
        ordered["start_reference_value"].to_numpy(float)
        - ordered["stroke_cost"].to_numpy(float)
        - ordered["end_reference_value"].to_numpy(float)
    )
    if not np.array_equal(
        ordered["fixed_reference_reward"].to_numpy(float), expected_reward
    ):
        raise AssertionError("Explicit H8 reference reward algebra changed")
    return {
        "rows": int(len(ordered)),
        "particles": PARTICLE_IDS.tolist(),
        "one_terminal_atom_per_path": True,
        "all_expected_terminal_atom": True,
        "reference_reward_exact": True,
        "ability_exact": True,
    }


def validate_exact_terminal_proposal(
    atoms: pd.DataFrame,
    inventory: pd.DataFrame,
) -> pd.Series:
    """Validate the full 904-atom proposal and return its one terminal row."""

    if len(inventory) != 1:
        raise AssertionError("The H8 diagnostic must materialize one support")
    inv = inventory.iloc[0]
    exact_inventory = {
        "arm": ARM,
        "support_key": EXPECTED_SUPPORT_KEY,
        "category": "app",
        "atoms": EXPECTED_SUPPORT_ATOMS,
        "spatial_atoms": EXPECTED_SPATIAL_ATOMS,
        "terminal_atoms": 1,
    }
    for column, expected in exact_inventory.items():
        if inv[column] != expected:
            raise AssertionError(
                f"Explicit H8 proposal inventory changed: {column}={inv[column]}"
            )
    if len(atoms) != EXPECTED_SUPPORT_ATOMS:
        raise AssertionError("The explicit H8 proposal atom count changed")
    if not atoms["support_key"].astype(str).eq(EXPECTED_SUPPORT_KEY).all():
        raise AssertionError("The explicit H8 proposal support key changed")
    terminal = atoms.loc[atoms["terminal"]]
    if len(terminal) != 1:
        raise AssertionError("The explicit H8 proposal needs one terminal donor")
    row = terminal.iloc[0]
    exact_terminal = {
        "proposal_row_order": EXPECTED_TERMINAL_PROPOSAL_ROW,
        "atom_id": EXPECTED_TERMINAL_ATOM_ID,
        "source_shot_key": EXPECTED_TERMINAL_ATOM_ID,
        "atom_kind": "app_holeout",
        "holed": True,
        "donor_event_key": EXPECTED_DONOR_EVENT_KEY,
        "donor_player_id": EXPECTED_DONOR_PLAYER_ID,
        "donor_player_name": EXPECTED_DONOR_PLAYER_NAME,
        "donor_round_no": EXPECTED_DONOR_ROUND_NO,
        "donor_hole_no": EXPECTED_DONOR_HOLE_NO,
        "donor_stroke_no": EXPECTED_DONOR_STROKE_NO,
        "proposal_branch": "terminal",
        "donor_ability_available": True,
    }
    for column, expected in exact_terminal.items():
        if row[column] != expected:
            raise AssertionError(
                f"Explicit H8 proposal terminal changed: {column}={row[column]}"
            )
    exact_floats = {
        "base_hierarchy_mass": EXPECTED_HOLEOUT_PROBABILITY,
        "spatial_model_weight": 1.0,
        "quality_base_mass": EXPECTED_HOLEOUT_PROBABILITY,
        "proposal_probability": EXPECTED_HOLEOUT_PROBABILITY,
        "proposal_cdf_lower": EXPECTED_PROPOSAL_CDF_LOWER,
        "proposal_cdf_upper": 1.0,
        "branch_total_mass": EXPECTED_HOLEOUT_PROBABILITY,
        "branch_conditional_probability": 1.0,
        "branch_cdf_lower": 0.0,
        "branch_cdf_upper": 1.0,
        "declared_app_base_holeout_probability": EXPECTED_HOLEOUT_PROBABILITY,
        "donor_ability_sg": EXPECTED_DONOR_ABILITY_SG,
    }
    for column, expected in exact_floats.items():
        if float(row[column]) != float(expected):
            raise AssertionError(
                f"Explicit H8 proposal terminal mass/CDF changed: {column}"
            )
    return row


def build_counter_replay(
    ledger: pd.DataFrame,
    *,
    seed: int = streaming.DEFAULT_SEED,
) -> pd.DataFrame:
    """Persist exact observed-versus-regenerated initial APP uniforms."""

    expected = regenerate_initial_app_uniforms(seed=int(seed))
    columns = [
        "particle_id",
        "rng_index_uniform",
        "rng_lie_uniform",
        "rng_holeout_uniform",
        "rng_holeout_residual_uniform",
        "rng_shot_transition_index",
    ]
    observed = ledger.loc[:, columns].copy().sort_values(
        "particle_id", kind="mergesort"
    ).reset_index(drop=True)
    if len(observed) != len(PARTICLE_IDS):
        raise AssertionError("Counter replay requires one initial APP atom per ID")
    replay = observed.merge(expected, on="particle_id", validate="one_to_one")
    pairs = (
        ("rng_index_uniform", "expected_rng_app_index_uniform"),
        ("rng_lie_uniform", "expected_rng_app_lie_uniform"),
        ("rng_holeout_uniform", "expected_rng_app_holeout_uniform"),
        (
            "rng_holeout_residual_uniform",
            "expected_rng_holeout_residual_uniform",
        ),
    )
    for observed_column, expected_column in pairs:
        values = replay[observed_column].to_numpy(float)
        expected_values = replay[expected_column].to_numpy(float)
        replay[f"{observed_column}_exact"] = values == expected_values
        replay[f"{observed_column}_residual"] = values - expected_values
        if not np.array_equal(values, expected_values):
            raise AssertionError(f"Counter replay changed: {observed_column}")
    if not replay["rng_shot_transition_index"].eq(-1).all():
        raise AssertionError("H8 initial APP counter index changed")
    replay.insert(1, "rng_contract", str(streaming.hook.RNG_VERSION))
    replay.insert(2, "rng_seed", int(seed))
    replay.insert(3, "event_key", TARGET_EVENT_KEY)
    replay.insert(4, "player_id", TARGET_PLAYER_ID)
    replay.insert(5, "round_no", ROUND_NO)
    replay.insert(6, "hole_no", HOLE_NO)
    return replay


def build_result_differential(
    traced: streaming.StreamingHoleResult,
    plain: streaming.StreamingHoleResult,
) -> pd.DataFrame:
    """Summarize every parent result field with exact mismatch counts."""

    rows: list[dict[str, object]] = []
    for name in traced.__dataclass_fields__:
        left = np.asarray(getattr(traced, name))
        right = np.asarray(getattr(plain, name))
        if left.shape != right.shape:
            raise AssertionError(f"Result differential shape changed: {name}")
        if left.dtype.kind in "fc" or right.dtype.kind in "fc":
            equal = np.isclose(left, right, atol=0.0, rtol=0.0, equal_nan=True)
            finite = np.isfinite(left.astype(float)) & np.isfinite(
                right.astype(float)
            )
            maximum = float(
                np.max(
                    np.abs(left.astype(float)[finite] - right.astype(float)[finite]),
                    initial=0.0,
                )
            )
        else:
            equal = left == right
            maximum = 0.0
        rows.append(
            {
                "result_field": str(name),
                "shape": json.dumps(list(left.shape)),
                "traced_dtype": str(left.dtype),
                "plain_dtype": str(right.dtype),
                "mismatch_count": int((~equal).sum()),
                "max_abs_numeric_difference": maximum,
                "exact": bool(equal.all()),
            }
        )
    output = pd.DataFrame(rows)
    if not output["exact"].all():
        raise AssertionError("A persisted traced-parent result field differs")
    return output


def build_selection_evidence(
    ledger: pd.DataFrame,
    proposal_terminal: pd.Series,
    result: streaming.StreamingHoleResult,
) -> pd.DataFrame:
    """Build one self-contained terminal-selection record per exact ID."""

    score = pd.DataFrame(
        {
            "particle_id": result.particle_id.astype(np.int64),
            "hole_strokes": result.hole_strokes.astype(np.int16),
            "result_terminal": result.terminal.astype(bool),
            "result_app_atom_id": result.app_atom_id.astype(str),
            "result_final_lie": result.final_lie.astype(np.int8),
            "result_final_remaining_yd": result.final_remaining_yd.astype(float),
        }
    )
    selected_columns = [
        "particle_id",
        "support_key",
        "selected_spatial_proposal_atom_id",
        "atom_id",
        "donor_event_key",
        "donor_player_id",
        "donor_player_name",
        "donor_round_no",
        "donor_hole_no",
        "donor_stroke_no",
        "donor_ability_sg",
        "donor_ability_available",
        "rng_index_uniform",
        "rng_lie_uniform",
        "rng_holeout_uniform",
        "rng_holeout_residual_uniform",
        "declared_app_base_holeout_probability",
        "start_reference_value",
        "end_reference_value",
        "fixed_reference_reward",
        "stroke_cost",
        "terminal",
        "holed",
    ]
    evidence = ledger.loc[:, selected_columns].merge(
        score, on="particle_id", validate="one_to_one"
    )
    for column in (
        "proposal_row_order",
        "quality_base_mass",
        "proposal_probability",
        "proposal_cdf_lower",
        "proposal_cdf_upper",
        "branch_total_mass",
        "branch_conditional_probability",
        "branch_cdf_lower",
        "branch_cdf_upper",
        "proposal_support_measure_sha256",
        "proposal_support_ability_sha256",
    ):
        evidence[f"terminal_{column}"] = proposal_terminal[column]
    residual = evidence["rng_holeout_residual_uniform"].to_numpy(float)
    lower = evidence["terminal_branch_cdf_lower"].to_numpy(float)
    upper = evidence["terminal_branch_cdf_upper"].to_numpy(float)
    evidence["residual_inside_terminal_branch_cdf"] = (residual >= lower) & (
        residual < upper
    )
    if not evidence["residual_inside_terminal_branch_cdf"].all():
        raise AssertionError("A residual uniform escaped the terminal donor CDF")
    if not evidence["hole_strokes"].eq(1).all():
        raise AssertionError("An explicit initial APP holeout score changed from one")
    if not evidence["result_terminal"].all():
        raise AssertionError("An explicit initial APP holeout is not terminal")
    if not evidence["result_app_atom_id"].astype(str).eq(
        evidence["selected_spatial_proposal_atom_id"].astype(str)
    ).all():
        raise AssertionError(
            "The legacy result lost its selected spatial APP atom identity"
        )
    if not evidence["result_final_lie"].eq(streaming.LIE_HOLE).all():
        raise AssertionError("An explicit initial APP holeout did not end in hole")
    if not evidence["result_final_remaining_yd"].eq(0.0).all():
        raise AssertionError("An explicit initial APP holeout retained distance")
    return evidence.sort_values("particle_id", kind="mergesort").reset_index(
        drop=True
    )


def _result_frame(
    traced: streaming.StreamingHoleResult,
    plain: streaming.StreamingHoleResult,
    *,
    event_start: object,
) -> pd.DataFrame:
    frames = []
    for implementation, result in (
        ("traced", traced),
        ("plain_parent", plain),
    ):
        frame = ledger_v6._hole_result_frame(
            result,
            arm=ARM,
            implementation=implementation,
        )
        frames.append(_with_target_context(frame, event_start=event_start))
    output = pd.concat(frames, ignore_index=True, sort=False)
    if len(output) != 2 * len(PARTICLE_IDS):
        raise AssertionError("The saved traced/plain result axis changed")
    expected_axis = {
        (implementation, int(particle_id))
        for implementation in ("traced", "plain_parent")
        for particle_id in PARTICLE_IDS
    }
    observed_axis = set(
        output[["implementation", "particle_id"]].itertuples(
            index=False, name=None
        )
    )
    if observed_axis != expected_axis:
        raise AssertionError("The saved traced/plain result IDs changed")
    return output


def _validate_output_index(directory: Path) -> dict[str, object]:
    """Recompute the non-circular output index and require exact closure."""

    directory = Path(directory)
    index_path = directory / "OUTPUT_SHA256.json"
    declared = json.loads(index_path.read_text(encoding="utf-8"))
    observed: dict[str, dict[str, object]] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise AssertionError("The explicit-holeout output contains a symlink")
        if path.is_dir() and path != directory:
            raise AssertionError(
                "The explicit-holeout output contains a nested directory"
            )
        if not path.is_file() or path.name == "OUTPUT_SHA256.json":
            continue
        relative = path.relative_to(directory).as_posix()
        observed[relative] = {
            "sha256": _sha256(path),
            "bytes": int(path.stat().st_size),
        }
    if declared.get("contract") != "non_circular_output_sha256_v1":
        raise AssertionError("The output index contract changed")
    if set(observed) != set(OUTPUT_PAYLOAD_FILES):
        raise AssertionError(
            "The explicit-holeout output file set changed: "
            f"{sorted(observed)}"
        )
    if declared.get("files") != observed:
        raise AssertionError("The explicit-holeout output closure changed")
    return {
        "contract": str(declared["contract"]),
        "files": int(len(observed)),
        "all_exact": True,
    }


def build_real_h8_holeout_diagnostic(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    export_root: Path = ledger_v6.precompute.DEFAULT_EXPORT,
    history_db: Path = ledger_v6.asof.DEFAULT_HISTORY_DB,
    profile_dir: Path = ledger_v6.precompute.DEFAULT_PROFILES,
    baseline_cases: Path = ledger_v6.precompute.DEFAULT_BASELINES,
    streaming_reference_dir: Path = streaming.DEFAULT_REFERENCE,
    reference_value_dir: Path = ledger_v6.DEFAULT_REFERENCE_DIR,
    bridge_path: Path = ledger_v6.asof.DEFAULT_BRIDGE,
) -> Path:
    """Build the exact five-ID real-kernel APP-holeout diagnostic."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output path already exists: {output_dir}")
    _validated_particle_ids(PARTICLE_IDS)
    _assert_frozen_v6_source()
    frozen_dependency_attestation = streaming.validate_frozen_dependencies()
    source_path = Path(__file__)
    source_start = _sha256(source_path)
    dependency_paths = _dependency_paths()
    dependency_start = _hash_paths(dependency_paths)

    prepared = streaming.prepare_streaming_event(
        export_root=Path(export_root),
        history_db=Path(history_db),
        profile_dir=Path(profile_dir),
        baseline_cases=Path(baseline_cases),
        reference_dir=Path(streaming_reference_dir),
        event_name=ledger_v6.precompute.DEFAULT_EVENT,
        holdout_year=ledger_v6.precompute.DEFAULT_HOLDOUT,
        course_start_year=2020,
        roster_seed=streaming.DEFAULT_SEED,
    )
    if str(prepared.atoms.target_event_key) != TARGET_EVENT_KEY:
        raise ValueError("Prepared event changed from the 2026 Travelers target")
    record = _select_exact_record(prepared.grid)
    prior = pd.DataFrame(
        [
            {"year": int(year), "event_key": str(event_key)}
            for year, event_key in sorted(
                prepared.atoms.prepared["event_keys"].items()
            )
            if int(year) < int(prepared.atoms.holdout_year)
        ]
    )
    population_bundle = (
        ledger_v6.traced_population.build_strict_prior_course_trace_bundle(
            Path(export_root),
            prior,
            prepared.atoms.target_event_key,
            categories=("arg", "putt"),
            min_exact=12,
            knn=64,
        )
    )
    traced_sampler = population_bundle.traced_sampler(
        chunk_size=len(PARTICLE_IDS)
    )
    reference_model = ledger_v6.fixed_reference.ReferenceValueModel.load(
        Path(reference_value_dir)
    )
    bridge, bridge_meta = ledger_v6.asof.load_strict_bridge(Path(bridge_path))
    repository = ledger_v6.IdentityPreservingSupportRepository(
        arm=ARM,
        provider=ledger_v6.precompute.AtomSupportProvider(
            prepared.atoms,
            arm=ARM,
            cache_enabled=True,
        ),
    )
    repositories = {ARM: repository}
    traced_chain = ledger_v6.TracedStreamingShotChain(
        repositories=repositories,
        course_drives=prepared.atoms.raw_course_drives,
        course_app=prepared.atoms.raw_course_app,
        traced_state_sampler=traced_sampler,
        hazard_resolver=prepared.batched_hazard,
        reference_model=reference_model,
    )
    # The plain parent is constructed against the same repository, then runs
    # only after the traced execution in run_traced_plain_pair.
    plain_chain = streaming.StreamingShotChain(
        repositories=repositories,
        course_drives=prepared.atoms.raw_course_drives,
        course_app=prepared.atoms.raw_course_app,
        state_sampler=traced_sampler.base_sampler,
        hazard_resolver=prepared.batched_hazard,
    )
    execution, plain_result, parity, support_discovery = run_traced_plain_pair(
        traced_chain=traced_chain,
        plain_chain=plain_chain,
        repository=repository,
        record=record,
    )

    ledger = _with_target_context(
        execution.ledger,
        event_start=prepared.atoms.target_event_start,
    )
    ledger, ability_audit = ledger_v6.attach_strict_prior_donor_abilities(
        ledger,
        bridge,
        target_event_start=prepared.atoms.target_event_start,
    )
    ledger_validation = validate_exact_terminal_ledger(ledger)
    counter_rng_audit = ledger_v6.validate_counter_rng_trace(
        ledger,
        seed=streaming.DEFAULT_SEED,
        continuation_seed=streaming.DEFAULT_CONTINUATION_SEED,
    )
    if counter_rng_audit["uniforms_replayed"] != 3 * len(PARTICLE_IDS):
        raise AssertionError("The bounded H8 counter replay count changed")
    counter_replay = build_counter_replay(ledger)

    proposal_atoms, proposal_inventory = (
        ledger_v6.materialize_proposal_support_contract(
            repositories,
            target_event_key=TARGET_EVENT_KEY,
            target_event_start=prepared.atoms.target_event_start,
            target_player_id=TARGET_PLAYER_ID,
        )
    )
    proposal_atoms, proposal_ability_audit = (
        ledger_v6.attach_strict_prior_donor_abilities(
            proposal_atoms,
            bridge,
            target_event_start=prepared.atoms.target_event_start,
        )
    )
    proposal_atoms, proposal_inventory, proposal_ability_seal = (
        ledger_v6.seal_proposal_support_ability_hashes(
            proposal_atoms,
            proposal_inventory,
        )
    )
    persisted_proposal_audit = (
        ledger_v6.validate_persisted_proposal_support_contract(
            proposal_atoms,
            proposal_inventory,
        )
    )
    proposal_terminal = validate_exact_terminal_proposal(
        proposal_atoms,
        proposal_inventory,
    )

    result_frame = _result_frame(
        execution.result,
        plain_result,
        event_start=prepared.atoms.target_event_start,
    )
    result_differential = build_result_differential(
        execution.result,
        plain_result,
    )
    telescope_audit = _with_target_context(
        execution.telescope_audit,
        event_start=prepared.atoms.target_event_start,
    )
    selection_evidence = build_selection_evidence(
        ledger,
        proposal_terminal,
        execution.result,
    )
    if not telescope_audit["stroke_identity_residual"].eq(0).all():
        raise AssertionError("Explicit H8 stroke identity changed")
    if telescope_audit["reward_telescope_residual"].abs().max() > 5e-12:
        raise AssertionError("Explicit H8 reference rewards do not telescope")

    _assert_frozen_v6_source()
    dependency_end = _hash_paths(dependency_paths)
    if source_start != _sha256(source_path):
        raise RuntimeError("The H8 diagnostic source changed during execution")
    if dependency_start != dependency_end:
        raise RuntimeError("An H8 diagnostic dependency changed during execution")

    output_dir.mkdir(parents=True)
    frames = {
        "explicit_holeout_transition_ledger.parquet": ledger,
        "particle_telescope_audit.parquet": telescope_audit,
        "proposal_support_atoms.parquet": proposal_atoms,
        "proposal_support_inventory.parquet": proposal_inventory,
        "traced_plain_results.parquet": result_frame,
        "traced_parent_differential.parquet": result_differential,
        "counter_uniform_replay.parquet": counter_replay,
        "explicit_holeout_selection.parquet": selection_evidence,
    }
    persisted_frames: dict[str, pd.DataFrame] = {}
    logical_hashes: dict[str, str] = {}
    for filename, frame in frames.items():
        persisted, logical_sha = ledger_v6._write_verified_parquet(
            frame,
            output_dir / filename,
        )
        persisted_frames[filename] = persisted
        logical_hashes[filename] = logical_sha

    # Re-run every consumer-facing validator against Parquet-reloaded frames.
    persisted_ledger = persisted_frames[
        "explicit_holeout_transition_ledger.parquet"
    ]
    persisted_inventory = persisted_frames[
        "proposal_support_inventory.parquet"
    ]
    persisted_atoms = persisted_frames["proposal_support_atoms.parquet"]
    validate_exact_terminal_ledger(persisted_ledger)
    validate_exact_terminal_proposal(persisted_atoms, persisted_inventory)
    persisted_counter_audit = ledger_v6.validate_counter_rng_trace(
        persisted_ledger,
        seed=streaming.DEFAULT_SEED,
        continuation_seed=streaming.DEFAULT_CONTINUATION_SEED,
    )
    ledger_v6.validate_persisted_proposal_support_contract(
        persisted_atoms,
        persisted_inventory,
    )
    if not persisted_frames["traced_parent_differential.parquet"][
        "exact"
    ].all():
        raise AssertionError("Persisted traced-parent differential changed")
    if not persisted_frames["explicit_holeout_selection.parquet"][
        "residual_inside_terminal_branch_cdf"
    ].all():
        raise AssertionError("Persisted terminal CDF selection changed")

    # These are the final source observations after all materialization,
    # persistence, reload, and replay gates have completed.
    _assert_frozen_v6_source()
    source_end = _sha256(source_path)
    dependency_end = _hash_paths(dependency_paths)
    if source_start != source_end:
        raise RuntimeError("The H8 diagnostic source changed before sealing")
    if dependency_start != dependency_end:
        raise RuntimeError("An H8 diagnostic dependency changed before sealing")

    terminal_dict = {
        name: (
            value.item() if isinstance(value, np.generic) else value
        )
        for name, value in proposal_terminal.to_dict().items()
        if name
        in {
            "proposal_row_order",
            "atom_id",
            "source_shot_key",
            "donor_event_key",
            "donor_player_id",
            "donor_player_name",
            "donor_round_no",
            "donor_hole_no",
            "donor_stroke_no",
            "base_hierarchy_mass",
            "spatial_model_weight",
            "quality_base_mass",
            "proposal_probability",
            "proposal_cdf_lower",
            "proposal_cdf_upper",
            "branch_total_mass",
            "branch_conditional_probability",
            "branch_cdf_lower",
            "branch_cdf_upper",
            "declared_app_base_holeout_probability",
            "donor_ability_sg",
            "donor_ability_available",
            "proposal_support_measure_sha256",
            "proposal_support_ability_sha256",
        }
    }
    manifest = {
        "contract": CONTRACT,
        "parent_trace_contract": str(ledger_v6.LEDGER_SLICE_VERSION),
        "research_only": True,
        "production_untouched": True,
        "mechanics_only": True,
        "anchor_enabled": False,
        "scope": {
            "event_key": TARGET_EVENT_KEY,
            "event_start": prepared.atoms.target_event_start.isoformat(),
            "player_id": TARGET_PLAYER_ID,
            "round_no": ROUND_NO,
            "hole_no": HOLE_NO,
            "par": int(record.par),
            "arm": ARM,
            "initializer_route": "par3_initial_app",
            "particle_ids": PARTICLE_IDS.tolist(),
            "executed_particles": int(len(PARTICLE_IDS)),
            "counter_namespace_size": COUNTER_NAMESPACE_SIZE,
            "particle_search_performed": False,
            "filtering": False,
            "reroll": False,
            "execution_order": ["traced", "plain_parent"],
        },
        "frozen_expectations": {
            "support_key": EXPECTED_SUPPORT_KEY,
            "support_atoms": EXPECTED_SUPPORT_ATOMS,
            "spatial_atoms": EXPECTED_SPATIAL_ATOMS,
            "terminal_atoms": 1,
            "terminal_proposal_row": EXPECTED_TERMINAL_PROPOSAL_ROW,
            "terminal_atom_id": EXPECTED_TERMINAL_ATOM_ID,
            "holeout_probability": EXPECTED_HOLEOUT_PROBABILITY,
            "holeout_uniforms": EXPECTED_HOLEOUT_UNIFORMS.tolist(),
            "holeout_residual_uniforms": (
                EXPECTED_HOLEOUT_UNIFORMS / EXPECTED_HOLEOUT_PROBABILITY
            ).tolist(),
            "donor_event_key": EXPECTED_DONOR_EVENT_KEY,
            "donor_player_id": EXPECTED_DONOR_PLAYER_ID,
            "donor_player_name": EXPECTED_DONOR_PLAYER_NAME,
            "donor_round_no": EXPECTED_DONOR_ROUND_NO,
            "donor_hole_no": EXPECTED_DONOR_HOLE_NO,
            "donor_stroke_no": EXPECTED_DONOR_STROKE_NO,
            "donor_ability_sg": EXPECTED_DONOR_ABILITY_SG,
        },
        "terminal_proposal_atom": terminal_dict,
        "plain_parent_parity": parity,
        "support_discovery": support_discovery,
        "ledger_validation": ledger_validation,
        "counter_rng_audit": counter_rng_audit,
        "persisted_counter_rng_audit": persisted_counter_audit,
        "ability_audit": ability_audit,
        "proposal_support_ability_audit": proposal_ability_audit,
        "proposal_support_ability_seal": proposal_ability_seal,
        "proposal_support_contract_audit": persisted_proposal_audit,
        "bridge": bridge_meta,
        "reference": {
            "descriptor": reference_model.descriptor,
            "function_sha256": reference_model.model_sha256,
            "putting_curve_sha256": reference_model.putting_curve_sha256,
        },
        "gates": {
            "all_five_predeclared_ids_executed": True,
            "all_five_select_expected_terminal_atom": True,
            "all_five_holeout_uniforms_exact": True,
            "all_five_residual_uniforms_exact": True,
            "all_five_residuals_inside_terminal_branch_cdf": True,
            "terminal_proposal_mass_cdf_exact": True,
            "terminal_donor_identity_exact": True,
            "terminal_donor_ability_exact": True,
            "terminal_state_and_score_exact": True,
            "reference_reward_exact": True,
            "reward_telescope_exact": True,
            "traced_plain_parent_result_exact": bool(parity["exact"]),
            "plain_parent_new_supports": 0,
            "counter_rng_uniforms_exact": bool(counter_rng_audit["all_exact"]),
            "source_dependency_seals_exact": True,
            "diagnostic_status": "GO",
            "anchor_enabled": False,
            "production_authorized": False,
        },
        "inputs": {
            "runner_source_sha256_start": source_start,
            "runner_source_sha256_end": source_end,
            "frozen_v6_expected_sha256": FROZEN_V6_SOURCE_SHA256,
            "streaming_frozen_dependency_attestation": {
                "all_exact": True,
                "expected": dict(streaming.FROZEN_DEPENDENCY_SHA256),
                "observed": frozen_dependency_attestation,
            },
            "dependency_sha256_start": dependency_start,
            "dependency_sha256_end": dependency_end,
            "reference_manifest_sha256": _sha256(
                Path(reference_value_dir) / "manifest.json"
            ),
            "bridge_sha256": _sha256(Path(bridge_path)),
            "bridge_manifest_sha256": _sha256(
                Path(bridge_path).with_name("manifest.json")
            ),
            "prepared_input_sha256": prepared.input_hashes,
            "prepared_logical_sha256": prepared.logical_hashes,
            "population_transition_frame_logical_sha256": (
                streaming.frame_logical_sha256(population_bundle.transitions)
            ),
        },
        "output_logical_sha256": logical_hashes,
        "expected_output_files_before_index": list(OUTPUT_PAYLOAD_FILES),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    report = (
        "# Real R1 H8 explicit APP-holeout diagnostic\n\n"
        "Five predeclared counter IDs were run through the raw par-3 initial "
        "APP kernel and replayed through the plain parent. No particle search, "
        "filtering, or reroll occurred in this runner.\n\n"
        f"All {len(PARTICLE_IDS)} selected terminal atom "
        f"`{EXPECTED_TERMINAL_ATOM_ID}` from support "
        f"`{EXPECTED_SUPPORT_KEY}` at declared mass "
        f"{EXPECTED_HOLEOUT_PROBABILITY:.6f}. Traced/plain result parity, "
        "counter replay, residual branch CDF, terminal score, reference "
        "reward, donor identity, and strict-prior ability are exact.\n\n"
        "Diagnostic status: GO. This is mechanics/provenance evidence only; "
        "anchor solving and production use remain disabled.\n"
    )
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")

    # Guard the small manifest/report sealing window as well.
    _assert_frozen_v6_source()
    if source_start != _sha256(source_path):
        raise RuntimeError("The H8 diagnostic source changed during sealing")
    if dependency_start != _hash_paths(dependency_paths):
        raise RuntimeError("An H8 diagnostic dependency changed during sealing")
    ledger_v6._output_index(output_dir)
    _validate_output_index(output_dir)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--export-root", type=Path, default=ledger_v6.precompute.DEFAULT_EXPORT
    )
    parser.add_argument(
        "--history-db", type=Path, default=ledger_v6.asof.DEFAULT_HISTORY_DB
    )
    parser.add_argument(
        "--profile-dir", type=Path, default=ledger_v6.precompute.DEFAULT_PROFILES
    )
    parser.add_argument(
        "--baseline-cases", type=Path, default=ledger_v6.precompute.DEFAULT_BASELINES
    )
    parser.add_argument(
        "--streaming-reference-dir", type=Path, default=streaming.DEFAULT_REFERENCE
    )
    parser.add_argument(
        "--reference-value-dir", type=Path, default=ledger_v6.DEFAULT_REFERENCE_DIR
    )
    parser.add_argument("--bridge", type=Path, default=ledger_v6.asof.DEFAULT_BRIDGE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = build_real_h8_holeout_diagnostic(
        output_dir=args.output_dir,
        export_root=args.export_root,
        history_db=args.history_db,
        profile_dir=args.profile_dir,
        baseline_cases=args.baseline_cases,
        streaming_reference_dir=args.streaming_reference_dir,
        reference_value_dir=args.reference_value_dir,
        bridge_path=args.bridge,
    )
    print(output)


if __name__ == "__main__":
    main()


__all__ = [
    "ARM",
    "CONTRACT",
    "COUNTER_NAMESPACE_SIZE",
    "DEFAULT_OUTPUT_DIR",
    "EXPECTED_DONOR_ABILITY_SG",
    "EXPECTED_DONOR_EVENT_KEY",
    "EXPECTED_DONOR_PLAYER_ID",
    "EXPECTED_HOLEOUT_PROBABILITY",
    "EXPECTED_HOLEOUT_UNIFORMS",
    "EXPECTED_SUPPORT_KEY",
    "EXPECTED_TERMINAL_ATOM_ID",
    "FROZEN_V6_SOURCE_SHA256",
    "HOLE_NO",
    "PARTICLE_IDS",
    "ROUND_NO",
    "TARGET_EVENT_KEY",
    "TARGET_PLAYER_ID",
    "build_counter_replay",
    "build_real_h8_holeout_diagnostic",
    "build_result_differential",
    "build_selection_evidence",
    "regenerate_initial_app_uniforms",
    "run_traced_plain_pair",
    "validate_exact_terminal_ledger",
    "validate_exact_terminal_proposal",
]
