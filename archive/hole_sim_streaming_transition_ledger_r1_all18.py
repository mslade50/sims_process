"""Archive-only one-pass R1 all-18 transition-ledger runner.

This module orchestrates the already-audited bounded trace seam in
``hole_sim_streaming_transition_ledger``.  It deliberately does not alter or
replace the streaming kernel.  Event inputs, the strict-prior population
model, reference values, donor-ability bridge, and frozen target bundle are
each prepared once; all traced hole/arm cases run before any plain-parent
replay begins.

The output is mechanics-only.  It is not an anchor solve and cannot authorize
one, even if a future donor bridge happens to have complete coverage.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


ARCHIVE = Path(__file__).resolve().parent
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

import hole_sim_normalized_chain_streaming as streaming  # noqa: E402
import hole_sim_streaming_transition_ledger as ledger_v6  # noqa: E402


CONTRACT = "streaming_complete_transition_ledger_r1_all18_v1"
FROZEN_V6_SOURCE_SHA256 = (
    "11b14109b938cda5da4b4eaae4d00eb43f9d44074fdbd9e48328a4f889d7ab97"
)
TARGET_PLAYER_ID = "33204"
TARGET_EVENT_KEY = "pga:R2026034"
ROUND_NO = 1
HOLES = tuple(range(1, 19))
ARMS = ("raw", "normalized")
PARTICLES = 128
PARTICLE_IDS = np.arange(PARTICLES, dtype=np.int64)
DEFAULT_OUTPUT_DIR = Path(
    "permanent_data/"
    "hole_sim_streaming_transition_ledger_lowry_r1_all18_128_v1"
)


@dataclass(frozen=True, order=True)
class CaseKey:
    """Stable identity for one round/hole/arm execution."""

    round_no: int
    hole_no: int
    arm: str

    @property
    def label(self) -> str:
        return f"R{self.round_no}H{self.hole_no:02d}|{self.arm}"


@dataclass(frozen=True)
class TracedCase:
    """One traced case bound to the exact grid record used to run it."""

    key: CaseKey
    record: object
    execution: ledger_v6.TracedHoleExecution


@dataclass(frozen=True)
class PlainCase:
    """One plain-parent replay aligned to a traced case."""

    key: CaseKey
    result: streaming.StreamingHoleResult


def _sha256(path: Path) -> str:
    return streaming.sha256_file(Path(path))


def _dependency_paths() -> dict[str, Path]:
    """Return every local source whose behavior is used by this runner."""

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


def _records_for_exact_axis(grid: pd.DataFrame) -> tuple[object, ...]:
    """Select Lowry R1 holes 1..18 and reject every axis ambiguity."""

    required = {"event_key", "source_player_id", "round_no", "hole_no"}
    missing = sorted(required.difference(grid.columns))
    if missing:
        raise ValueError(f"Prepared grid misses all-18 axis columns={missing}")
    player = grid["source_player_id"].astype(str).eq(TARGET_PLAYER_ID)
    selected = grid.loc[
        player
        & grid["round_no"].eq(ROUND_NO)
        & grid["hole_no"].isin(HOLES)
    ].copy()
    if len(selected) != len(HOLES):
        raise ValueError(
            f"Expected exactly {len(HOLES)} Lowry R1 hole rows; got {len(selected)}"
        )
    if selected.duplicated(["round_no", "hole_no"]).any():
        raise ValueError("Prepared all-18 grid contains duplicate round/hole rows")
    selected = selected.sort_values("hole_no", kind="mergesort")
    observed_holes = tuple(selected["hole_no"].astype(int))
    if observed_holes != HOLES:
        raise ValueError(f"Prepared all-18 holes changed: {observed_holes}")
    if not selected["round_no"].eq(ROUND_NO).all():
        raise AssertionError("Prepared all-18 records escaped R1")
    if not selected["source_player_id"].astype(str).eq(TARGET_PLAYER_ID).all():
        raise AssertionError("Prepared all-18 records changed player")
    if not selected["event_key"].astype(str).eq(TARGET_EVENT_KEY).all():
        raise AssertionError("Prepared all-18 records changed target event")
    return tuple(selected.itertuples(index=False))


def _expected_case_keys() -> tuple[CaseKey, ...]:
    return tuple(
        CaseKey(ROUND_NO, hole_no, arm)
        for hole_no in HOLES
        for arm in ARMS
    )


def run_traced_cases(
    chain: ledger_v6.TracedStreamingShotChain,
    records: Sequence[object],
    *,
    particle_ids: np.ndarray = PARTICLE_IDS,
    seed: int = streaming.DEFAULT_SEED,
    continuation_seed: int = streaming.DEFAULT_CONTINUATION_SEED,
) -> tuple[TracedCase, ...]:
    """Run the complete traced matrix in stable hole-major/arm-minor order."""

    ids = np.asarray(particle_ids, dtype=np.int64)
    if not np.array_equal(ids, PARTICLE_IDS):
        raise ValueError("All-18 trace requires shared particle IDs 0..127")
    cases: list[TracedCase] = []
    for record in records:
        if int(record.round_no) != ROUND_NO or int(record.hole_no) not in HOLES:
            raise ValueError("A traced record is outside the frozen R1 all-18 axis")
        if str(record.source_player_id) != TARGET_PLAYER_ID:
            raise ValueError("A traced record is not the frozen Lowry player")
        if str(record.event_key) != TARGET_EVENT_KEY:
            raise ValueError("A traced record is outside the frozen target event")
        for arm in ARMS:
            key = CaseKey(ROUND_NO, int(record.hole_no), arm)
            execution = chain.simulate_hole_with_ledger(
                record=record,
                arm=arm,
                particle_ids=ids,
                total_particles=PARTICLES,
                rng_mode="counter",
                seed=int(seed),
                continuation_seed=int(continuation_seed),
            )
            if not np.array_equal(execution.result.particle_id, PARTICLE_IDS):
                raise AssertionError(f"Traced particle axis changed for {key.label}")
            if not execution.result.terminal.all():
                raise AssertionError(f"Traced case did not terminate: {key.label}")
            cases.append(TracedCase(key=key, record=record, execution=execution))
    observed = tuple(case.key for case in cases)
    expected = _expected_case_keys()
    if observed != expected:
        raise AssertionError(f"Traced case axis changed: {observed} != {expected}")
    return tuple(cases)


def _support_snapshot(
    repositories: Mapping[str, ledger_v6.IdentityPreservingSupportRepository],
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Capture exact support keys, not only counts, across parent replay."""

    if set(repositories) != set(ARMS):
        raise ValueError("All-18 repositories must contain exactly both arms")
    return {
        arm: {
            "numeric": tuple(sorted(map(str, repository.supports))),
            "explicit_app": tuple(
                sorted(map(str, repository.explicit_app_supports))
            ),
        }
        for arm, repository in repositories.items()
    }


def run_plain_parent_cases(
    plain_chain: streaming.StreamingShotChain,
    traced_cases: Sequence[TracedCase],
    repositories: Mapping[str, ledger_v6.IdentityPreservingSupportRepository],
    *,
    particle_ids: np.ndarray = PARTICLE_IDS,
    seed: int = streaming.DEFAULT_SEED,
    continuation_seed: int = streaming.DEFAULT_CONTINUATION_SEED,
) -> tuple[tuple[PlainCase, ...], tuple[dict[str, object], ...], dict[str, object]]:
    """Replay every parent case after tracing and prohibit support discovery."""

    ids = np.asarray(particle_ids, dtype=np.int64)
    if not np.array_equal(ids, PARTICLE_IDS):
        raise ValueError("Plain-parent replay requires particle IDs 0..127")
    expected_keys = _expected_case_keys()
    if tuple(case.key for case in traced_cases) != expected_keys:
        raise ValueError("Plain-parent replay received an incomplete traced matrix")
    before = _support_snapshot(repositories)
    plain_cases: list[PlainCase] = []
    parity: list[dict[str, object]] = []
    for traced_case in traced_cases:
        result, _ = plain_chain.simulate_hole(
            record=traced_case.record,
            arm=traced_case.key.arm,
            particle_ids=ids,
            total_particles=PARTICLES,
            rng_mode="counter",
            seed=int(seed),
            continuation_seed=int(continuation_seed),
        )
        try:
            comparison = ledger_v6._assert_hole_results_exact(
                traced_case.execution.result,
                result,
                arm=traced_case.key.arm,
            )
        except AssertionError as exc:
            raise AssertionError(
                f"Plain-parent parity failed at {traced_case.key.label}"
            ) from exc
        comparison.update(
            {
                "round_no": traced_case.key.round_no,
                "hole_no": traced_case.key.hole_no,
                "case_key": traced_case.key.label,
            }
        )
        parity.append(comparison)
        plain_cases.append(PlainCase(key=traced_case.key, result=result))
    after = _support_snapshot(repositories)
    if after != before:
        raise AssertionError(
            "Plain-parent all-18 replay discovered support absent from trace"
        )
    return (
        tuple(plain_cases),
        tuple(parity),
        {
            "exact_key_sets": True,
            "before": {
                arm: {name: len(keys) for name, keys in values.items()}
                for arm, values in before.items()
            },
            "after": {
                arm: {name: len(keys) for name, keys in values.items()}
                for arm, values in after.items()
            },
        },
    )


def _target_columns(
    frame: pd.DataFrame,
    *,
    event_key: str,
    event_start: object,
    round_no: int,
    hole_no: int,
) -> pd.DataFrame:
    output = frame.copy()
    output.insert(0, "target_event_key", str(event_key))
    output.insert(1, "target_event_start", event_start)
    output.insert(2, "target_player_id", TARGET_PLAYER_ID)
    output.insert(3, "round_no", int(round_no))
    output.insert(4, "hole_no", int(hole_no))
    return output


def assemble_case_frames(
    traced_cases: Sequence[TracedCase],
    plain_cases: Sequence[PlainCase],
    *,
    event_key: str,
    event_start: object,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Concatenate ledgers, audits, and the saved parent differential."""

    expected = _expected_case_keys()
    if tuple(case.key for case in traced_cases) != expected:
        raise ValueError("Cannot assemble an incomplete traced case matrix")
    if tuple(case.key for case in plain_cases) != expected:
        raise ValueError("Cannot assemble an incomplete parent case matrix")
    ledger_parts: list[pd.DataFrame] = []
    audit_parts: list[pd.DataFrame] = []
    traced_result_parts: list[pd.DataFrame] = []
    plain_result_parts: list[pd.DataFrame] = []
    for case in traced_cases:
        ledger_parts.append(
            _target_columns(
                case.execution.ledger,
                event_key=event_key,
                event_start=event_start,
                round_no=case.key.round_no,
                hole_no=case.key.hole_no,
            )
        )
        audit = _target_columns(
            case.execution.telescope_audit,
            event_key=event_key,
            event_start=event_start,
            round_no=case.key.round_no,
            hole_no=case.key.hole_no,
        )
        audit.insert(5, "arm", case.key.arm)
        audit_parts.append(audit)
        result = ledger_v6._hole_result_frame(
            case.execution.result,
            arm=case.key.arm,
            implementation="traced",
        )
        traced_result_parts.append(
            _target_columns(
                result,
                event_key=event_key,
                event_start=event_start,
                round_no=case.key.round_no,
                hole_no=case.key.hole_no,
            )
        )
    for case in plain_cases:
        result = ledger_v6._hole_result_frame(
            case.result,
            arm=case.key.arm,
            implementation="plain_parent",
        )
        plain_result_parts.append(
            _target_columns(
                result,
                event_key=event_key,
                event_start=event_start,
                round_no=case.key.round_no,
                hole_no=case.key.hole_no,
            )
        )
    return (
        pd.concat(ledger_parts, ignore_index=True, sort=False),
        pd.concat(audit_parts, ignore_index=True, sort=False),
        pd.concat(
            traced_result_parts + plain_result_parts,
            ignore_index=True,
            sort=False,
        ),
    )


def validate_complete_axes(
    ledger: pd.DataFrame,
    audit: pd.DataFrame,
    results: pd.DataFrame,
) -> dict[str, object]:
    """Fail closed on filtering, duplicate cases, or missing particles."""

    expected_case_particles = {
        (ROUND_NO, hole, arm, particle)
        for hole in HOLES
        for arm in ARMS
        for particle in PARTICLE_IDS.tolist()
    }
    ledger_paths = set(
        ledger[["round_no", "hole_no", "arm", "particle_id"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    audit_paths = set(
        audit[["round_no", "hole_no", "arm", "particle_id"]]
        .itertuples(index=False, name=None)
    )
    if ledger_paths != expected_case_particles:
        raise AssertionError("Complete ledger filtered or duplicated a path axis")
    if audit_paths != expected_case_particles or len(audit) != len(
        expected_case_particles
    ):
        raise AssertionError("Telescope audit filtered or duplicated a path axis")
    expected_results = {
        (ROUND_NO, hole, arm, implementation, particle)
        for hole in HOLES
        for arm in ARMS
        for implementation in ("traced", "plain_parent")
        for particle in PARTICLE_IDS.tolist()
    }
    result_axis = list(
        results[
            ["round_no", "hole_no", "arm", "implementation", "particle_id"]
        ].itertuples(index=False, name=None)
    )
    if len(result_axis) != len(expected_results) or set(result_axis) != expected_results:
        raise AssertionError("Saved result differential filtered or duplicated an axis")
    if not results["terminal"].to_numpy(bool).all():
        raise AssertionError("Saved all-18 results contain a nonterminal particle")
    for _, group in ledger.groupby(
        ["round_no", "hole_no", "arm", "particle_id"],
        observed=True,
        sort=False,
    ):
        ordered = group.sort_values("atom_sequence", kind="mergesort")
        if not np.array_equal(
            ordered["atom_sequence"].to_numpy(np.int64),
            np.arange(len(ordered), dtype=np.int64),
        ):
            raise AssertionError("All-18 atom sequence is not contiguous")
        if int(ordered["terminal"].sum()) != 1 or not bool(
            ordered["terminal"].iloc[-1]
        ):
            raise AssertionError("All-18 path does not have one final terminal atom")
    if not ledger["target_player_id"].astype(str).eq(TARGET_PLAYER_ID).all():
        raise AssertionError("Complete ledger changed the target player")
    score_key = ["round_no", "hole_no", "arm", "particle_id"]
    ledger_score = ledger.groupby(score_key, observed=True, sort=True)[
        "stroke_cost"
    ].sum()
    audit_score = audit.set_index(score_key)["hole_strokes"].sort_index()
    traced_score = (
        results.loc[results["implementation"].eq("traced")]
        .set_index(score_key)["hole_strokes"]
        .sort_index()
    )
    if not (
        np.array_equal(ledger_score.sort_index().to_numpy(int), audit_score.to_numpy(int))
        and np.array_equal(audit_score.to_numpy(int), traced_score.to_numpy(int))
    ):
        raise AssertionError("All-18 ledger/audit/result score identity changed")
    return {
        "expected_cases": len(HOLES) * len(ARMS),
        "traced_paths": len(expected_case_particles),
        "audit_rows": int(len(audit)),
        "saved_result_rows": int(len(results)),
        "particle_ids_exact": True,
        "path_filtering": False,
        "all_terminal": True,
        "score_identity_exact": True,
    }


def summarize_hole_coverage(
    traced_cases: Sequence[TracedCase],
    ledger: pd.DataFrame,
) -> list[dict[str, object]]:
    """Persist routing and selected rare-path coverage for every hole."""

    by_key = {case.key: case for case in traced_cases}
    rows: list[dict[str, object]] = []
    for hole_no in HOLES:
        record = by_key[CaseKey(ROUND_NO, hole_no, ARMS[0])].record
        par = int(record.par)
        hole_ledger = ledger.loc[ledger["hole_no"].eq(hole_no)]
        arm_rows: dict[str, dict[str, object]] = {}
        for arm in ARMS:
            case = by_key[CaseKey(ROUND_NO, hole_no, arm)]
            selected = hole_ledger.loc[hole_ledger["arm"].eq(arm)]
            categories = selected["category"].astype(str).value_counts()
            initializer = selected.loc[selected["atom_sequence"].eq(0)]
            initializer_app_atoms = int(
                initializer["category"].astype(str).eq("app").sum()
            )
            initializer_tee_atoms = int(
                initializer["category"].astype(str).eq("ott").sum()
            )
            initializer_minus_one_atoms = int(
                initializer["rng_shot_transition_index"].eq(-1).sum()
            )
            initializer_route_exact = bool(
                len(initializer) == PARTICLES
                and initializer_minus_one_atoms == PARTICLES
                and (
                    (par == 3 and initializer_app_atoms == PARTICLES)
                    or (par != 3 and initializer_tee_atoms == PARTICLES)
                )
            )
            result_hazard_transitions = int(
                case.execution.result.hazard_transitions.sum()
            )
            result_penalty_strokes = int(
                case.execution.result.penalty_strokes.sum()
            )
            resolved_spatial = int(
                selected["hazard_resolution_status"]
                .astype(str)
                .eq("resolved_complete_atom")
                .sum()
            )
            collapsed_population = int(
                selected["hazard_resolution_status"]
                .astype(str)
                .eq("collapsed_exact_transition")
                .sum()
            )
            if result_hazard_transitions != resolved_spatial + collapsed_population:
                raise AssertionError(
                    f"Hazard transition accounting changed at R1H{hole_no}/{arm}"
                )
            if result_penalty_strokes != int(selected["penalty_strokes"].sum()):
                raise AssertionError(
                    f"Penalty-stroke accounting changed at R1H{hole_no}/{arm}"
                )
            arm_rows[arm] = {
                "particles": int(len(case.execution.result.particle_id)),
                "terminal": int(case.execution.result.terminal.sum()),
                "ledger_atoms": int(len(selected)),
                "ott_atoms": int(categories.get("ott", 0)),
                "app_atoms": int(categories.get("app", 0)),
                "arg_atoms": int(categories.get("arg", 0)),
                "putt_atoms": int(categories.get("putt", 0)),
                "initializer_app_atoms": initializer_app_atoms,
                "initializer_tee_atoms": initializer_tee_atoms,
                "initializer_index_minus_one_atoms": initializer_minus_one_atoms,
                "initializer_route_exact": initializer_route_exact,
                "hazard_transitions": result_hazard_transitions,
                "penalty_strokes": result_penalty_strokes,
                "resolved_spatial_hazard_atoms": resolved_spatial,
                "collapsed_population_penalty_atoms": collapsed_population,
                "hazard_accounting_exact": True,
                "penalty_accounting_exact": True,
                "explicit_app_holeouts_selected": int(
                    selected["legacy_holeout_decision"].to_numpy(bool).sum()
                ),
                "min_hole_strokes": int(
                    case.execution.result.hole_strokes.min()
                ),
                "max_hole_strokes": int(
                    case.execution.result.hole_strokes.max()
                ),
                "max_transition_count": int(
                    case.execution.result.transition_count.max(initial=0)
                ),
            }
        rows.append(
            {
                "round_no": ROUND_NO,
                "hole_no": hole_no,
                "par": par,
                "initializer_route": (
                    "par3_initial_app" if par == 3 else "par4_5_initial_tee_then_app"
                ),
                "arms": arm_rows,
            }
        )
    return rows


def _coverage_totals(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    fields = (
        "hazard_transitions",
        "penalty_strokes",
        "resolved_spatial_hazard_atoms",
        "collapsed_population_penalty_atoms",
        "explicit_app_holeouts_selected",
    )
    return {
        field: int(
            sum(
                int(row["arms"][arm][field])  # type: ignore[index]
                for row in rows
                for arm in ARMS
            )
        )
        for field in fields
    }


def summarize_empirical_coverage(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Make rare-path empirical coverage an explicit GO/HOLD contract."""

    totals = _coverage_totals(rows)
    par3_initializer_holes = [
        int(row["hole_no"])
        for row in rows
        if str(row["initializer_route"]) == "par3_initial_app"
    ]
    par3_routes_exact = bool(
        par3_initializer_holes
        and all(
            bool(row["arms"][arm]["initializer_route_exact"])  # type: ignore[index]
            for row in rows
            if int(row["hole_no"]) in par3_initializer_holes
            for arm in ARMS
        )
    )
    gates = {
        "par3_initializer_cases_present": par3_routes_exact,
        "resolved_spatial_hazard_atoms_present": bool(
            totals["resolved_spatial_hazard_atoms"] > 0
        ),
        "penalty_strokes_present": bool(totals["penalty_strokes"] > 0),
        "explicit_app_holeouts_selected": bool(
            totals["explicit_app_holeouts_selected"] > 0
        ),
    }
    complete = bool(all(gates.values()))
    return {
        "status": "GO" if complete else "HOLD",
        "complete": complete,
        "gates": gates,
        "par3_initializer_holes": par3_initializer_holes,
        "collapsed_population_penalty_nonblocking": True,
    }


def _case_order_manifest() -> list[dict[str, object]]:
    return [
        {
            "case_key": key.label,
            "round_no": key.round_no,
            "hole_no": key.hole_no,
            "arm": key.arm,
        }
        for key in _expected_case_keys()
    ]


def build_r1_all18(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    export_root: Path = ledger_v6.precompute.DEFAULT_EXPORT,
    history_db: Path = ledger_v6.asof.DEFAULT_HISTORY_DB,
    profile_dir: Path = ledger_v6.precompute.DEFAULT_PROFILES,
    baseline_cases: Path = ledger_v6.precompute.DEFAULT_BASELINES,
    streaming_reference_dir: Path = streaming.DEFAULT_REFERENCE,
    reference_value_dir: Path = ledger_v6.DEFAULT_REFERENCE_DIR,
    target_input_dir: Path = ledger_v6.DEFAULT_TARGET_DIR,
    bridge_path: Path = ledger_v6.asof.DEFAULT_BRIDGE,
) -> Path:
    """Build the frozen one-player R1x18 mechanics trace in one pass."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output path already exists: {output_dir}")
    _assert_frozen_v6_source()
    frozen_dependency_attestation = streaming.validate_frozen_dependencies()
    source_path = Path(__file__)
    source_start = _sha256(source_path)
    dependency_paths = _dependency_paths()
    dependency_start = _hash_paths(dependency_paths)

    # Each expensive or authority-bearing input is prepared exactly once.
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
        raise ValueError(
            "Prepared event changed from the frozen 2026 Travelers target"
        )
    records = _records_for_exact_axis(prepared.grid)
    prior = pd.DataFrame(
        [
            {"year": int(year), "event_key": str(event_key)}
            for year, event_key in sorted(
                prepared.atoms.prepared["event_keys"].items()
            )
            if int(year) < int(prepared.atoms.holdout_year)
        ]
    )
    population_bundle = ledger_v6.traced_population.build_strict_prior_course_trace_bundle(
        Path(export_root),
        prior,
        prepared.atoms.target_event_key,
        categories=("arg", "putt"),
        min_exact=12,
        knn=64,
    )
    traced_sampler = population_bundle.traced_sampler(chunk_size=PARTICLES)
    reference_model = ledger_v6.fixed_reference.ReferenceValueModel.load(
        Path(reference_value_dir)
    )
    bridge, bridge_meta = ledger_v6.asof.load_strict_bridge(Path(bridge_path))
    targets_path = Path(target_input_dir) / "production_anchor_inputs.parquet"
    targets_manifest_path = Path(target_input_dir) / "manifest.json"
    targets, validated_targets, target_validation = (
        ledger_v6.load_validated_production_targets(
            Path(target_input_dir),
            expected_event_key=str(prepared.atoms.target_event_key),
            expected_event_start=prepared.atoms.target_event_start,
        )
    )
    target = validated_targets.loc[
        validated_targets["player_id"].astype(str).eq(TARGET_PLAYER_ID)
        & validated_targets["round_no"].eq(ROUND_NO)
    ]
    rich_target = targets.loc[
        targets["player_id"].astype(str).eq(TARGET_PLAYER_ID)
        & targets["round_no"].eq(ROUND_NO)
    ]
    if len(target) != 1 or len(rich_target) != 1:
        raise ValueError("Frozen Lowry R1 target row is absent or duplicated")
    target_row = target.iloc[0]
    rich_target_row = rich_target.iloc[0]

    repositories: dict[str, ledger_v6.IdentityPreservingSupportRepository] = {}
    for arm in ARMS:
        repositories[arm] = ledger_v6.IdentityPreservingSupportRepository(
            arm=arm,
            provider=ledger_v6.precompute.AtomSupportProvider(
                prepared.atoms,
                arm=arm,
                cache_enabled=True,
            ),
        )
    traced_chain = ledger_v6.TracedStreamingShotChain(
        repositories=repositories,
        course_drives=prepared.atoms.raw_course_drives,
        course_app=prepared.atoms.raw_course_app,
        traced_state_sampler=traced_sampler,
        hazard_resolver=prepared.batched_hazard,
        reference_model=reference_model,
    )
    traced_cases = run_traced_cases(traced_chain, records)

    # This constructor and every replay occur only after all traced cases.
    plain_chain = streaming.StreamingShotChain(
        repositories=repositories,
        course_drives=prepared.atoms.raw_course_drives,
        course_app=prepared.atoms.raw_course_app,
        state_sampler=traced_sampler.base_sampler,
        hazard_resolver=prepared.batched_hazard,
    )
    plain_cases, plain_parent_parity, support_discovery = run_plain_parent_cases(
        plain_chain,
        traced_cases,
        repositories,
    )

    complete_ledger, audit, result_frame = assemble_case_frames(
        traced_cases,
        plain_cases,
        event_key=str(prepared.atoms.target_event_key),
        event_start=prepared.atoms.target_event_start,
    )
    axis_audit = validate_complete_axes(complete_ledger, audit, result_frame)
    hole_coverage = summarize_hole_coverage(traced_cases, complete_ledger)
    coverage_totals = _coverage_totals(hole_coverage)
    empirical_coverage = summarize_empirical_coverage(hole_coverage)
    empirical_coverage_gates = empirical_coverage["gates"]
    empirical_coverage_complete = bool(empirical_coverage["complete"])
    par3_initializer_holes = empirical_coverage["par3_initializer_holes"]

    proposal_atoms, proposal_inventory = (
        ledger_v6.materialize_proposal_support_contract(
            repositories,
            target_event_key=str(prepared.atoms.target_event_key),
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
    complete_ledger, ability_audit = ledger_v6.attach_strict_prior_donor_abilities(
        complete_ledger,
        bridge,
        target_event_start=prepared.atoms.target_event_start,
    )

    _assert_frozen_v6_source()
    dependency_end = _hash_paths(dependency_paths)
    if source_start != _sha256(source_path):
        raise RuntimeError("All-18 runner source changed during execution")
    if dependency_start != dependency_end:
        raise RuntimeError("An all-18 runner dependency changed during execution")

    output_dir.mkdir(parents=True)
    ledger_path = output_dir / "complete_transition_ledger.parquet"
    audit_path = output_dir / "particle_telescope_audit.parquet"
    proposal_path = output_dir / "proposal_support_atoms.parquet"
    inventory_path = output_dir / "proposal_support_inventory.parquet"
    results_path = output_dir / "plain_parent_result_parity.parquet"
    complete_ledger, ledger_logical_sha = ledger_v6._write_verified_parquet(
        complete_ledger, ledger_path
    )
    audit, audit_logical_sha = ledger_v6._write_verified_parquet(audit, audit_path)
    proposal_atoms, proposal_logical_sha = ledger_v6._write_verified_parquet(
        proposal_atoms, proposal_path
    )
    proposal_inventory, inventory_logical_sha = ledger_v6._write_verified_parquet(
        proposal_inventory, inventory_path
    )
    result_frame, result_logical_sha = ledger_v6._write_verified_parquet(
        result_frame, results_path
    )

    counter_rng_audit = ledger_v6.validate_counter_rng_trace(
        complete_ledger,
        seed=streaming.DEFAULT_SEED,
        continuation_seed=streaming.DEFAULT_CONTINUATION_SEED,
    )
    persisted_proposal_audit = (
        ledger_v6.validate_persisted_proposal_support_contract(
            proposal_atoms,
            proposal_inventory,
        )
    )
    if not all(bool(row["exact"]) for row in plain_parent_parity):
        raise AssertionError("An all-18 parent differential is not exact")
    if not audit["stroke_identity_residual"].eq(0).all():
        raise AssertionError("All-18 atom stroke costs do not equal hole scores")
    if audit["reward_telescope_residual"].abs().max() > 5e-12:
        raise AssertionError("All-18 fixed-reference rewards do not telescope")

    # Recheck after persistence and validators so manifest start/end seals are
    # the final observed source state, not a pre-write snapshot.
    _assert_frozen_v6_source()
    source_end = _sha256(source_path)
    dependency_end = _hash_paths(dependency_paths)
    if source_start != source_end:
        raise RuntimeError("All-18 runner source changed before manifest sealing")
    if dependency_start != dependency_end:
        raise RuntimeError("An all-18 dependency changed before manifest sealing")

    manifest = {
        "contract": CONTRACT,
        "parent_trace_contract": str(ledger_v6.LEDGER_SLICE_VERSION),
        "research_only": True,
        "production_untouched": True,
        "mechanics_only": True,
        "anchor": {
            "enabled": False,
            "authorized": False,
            "population_neighborhoods_persisted": False,
            "missing_ability_policy": "fail_closed_not_neutral_zero",
        },
        "axes": {
            "event_key": str(prepared.atoms.target_event_key),
            "event_start": prepared.atoms.target_event_start.isoformat(),
            "player_id": TARGET_PLAYER_ID,
            "rounds": [ROUND_NO],
            "holes": list(HOLES),
            "arms": list(ARMS),
            "particles": PARTICLES,
            "particle_ids": PARTICLE_IDS.tolist(),
            "particle_id_bounds": [0, PARTICLES - 1],
            "particle_id_contract": "contiguous_zero_based_0_through_127",
            "case_order": _case_order_manifest(),
            "execution_phase_order": ["all_traced_cases", "all_plain_parent_cases"],
            "rng_mode": "counter",
            "rng_seed": int(streaming.DEFAULT_SEED),
            "rng_continuation_seed": int(streaming.DEFAULT_CONTINUATION_SEED),
            "rng_stream_contract": str(streaming.hook.RNG_VERSION),
        },
        "axis_audit": axis_audit,
        "hole_coverage": hole_coverage,
        "coverage_totals": coverage_totals,
        "empirical_coverage": empirical_coverage,
        "plain_parent_parity": list(plain_parent_parity),
        "support_discovery": support_discovery,
        "counter_rng_audit": counter_rng_audit,
        "ability_audit": ability_audit,
        "proposal_support_ability_audit": proposal_ability_audit,
        "proposal_support_ability_seal": proposal_ability_seal,
        "bridge": bridge_meta,
        "reference": {
            "descriptor": reference_model.descriptor,
            "function_sha256": reference_model.model_sha256,
            "putting_curve_sha256": reference_model.putting_curve_sha256,
        },
        "production_target": {
            "event_key": str(target_row.event_key),
            "player_id": str(target_row.player_id),
            "round_no": int(target_row.round_no),
            "ott": float(target_row.target_ott_sg),
            "app": float(target_row.target_app_sg),
            "arg": float(target_row.target_arg_sg),
            "putt": float(target_row.target_putt_sg),
            "total": float(target_row.target_total_sg),
            "production_input_sha256": str(target_row.production_input_sha256),
            "target_ability": {
                category: float(rich_target_row[f"ability_{category}_sg"])
                for category in ledger_v6.CATEGORY_ABILITY_COLUMN
            },
            "authority": str(rich_target_row["round_target_authority"]),
            "production_equivalence_status": str(
                rich_target_row["production_equivalence_status"]
            ),
            "validation": target_validation,
        },
        "proposal_support": {
            "scope": (
                "complete_materialized_r1_all18_tee_app_proposal_atoms_"
                "before_surface_hazard_resolution"
            ),
            "supports": int(len(proposal_inventory)),
            "atoms": int(len(proposal_atoms)),
            "app_supports": int(proposal_inventory["category"].eq("app").sum()),
            "app_terminal_atoms": int(
                proposal_atoms.loc[
                    proposal_atoms["category"].eq("app"), "terminal"
                ].sum()
            ),
            "implicit_holeout_probability": float(
                proposal_atoms.loc[
                    proposal_atoms["category"].eq("app"),
                    "implicit_holeout_probability",
                ].max()
            ),
            "population_neighborhoods_persisted": False,
            "persisted_contract_audit": persisted_proposal_audit,
        },
        "gates": {
            "complete_axes_exact": True,
            "path_filtering": False,
            "all_particles_terminal": bool(result_frame["terminal"].all()),
            "zero_stroke_identity_residual": bool(
                audit["stroke_identity_residual"].eq(0).all()
            ),
            "max_abs_reward_telescope_residual": float(
                audit["reward_telescope_residual"].abs().max()
            ),
            "plain_parent_result_exact": True,
            "plain_parent_new_supports": 0,
            "physical_state_seams_exact": True,
            "counter_rng_uniforms_exact": bool(counter_rng_audit["all_exact"]),
            "par3_initializer_cases_present": empirical_coverage_gates[
                "par3_initializer_cases_present"
            ],
            "resolved_spatial_hazard_atoms_present": empirical_coverage_gates[
                "resolved_spatial_hazard_atoms_present"
            ],
            "penalty_strokes_present": empirical_coverage_gates[
                "penalty_strokes_present"
            ],
            "explicit_app_holeouts_selected": empirical_coverage_gates[
                "explicit_app_holeouts_selected"
            ],
            "empirical_mechanics_coverage_complete": empirical_coverage_complete,
            "proposal_support_mass_exact": bool(
                np.isclose(
                    proposal_inventory["quality_mass"].to_numpy(float),
                    1.0,
                    atol=ledger_v6.APP_SUPPORT_MASS_TOLERANCE,
                    rtol=0.0,
                ).all()
            ),
            "proposal_app_terminal_mass_exact": bool(
                np.isclose(
                    proposal_inventory.loc[
                        proposal_inventory["category"].eq("app"), "terminal_mass"
                    ].to_numpy(float),
                    proposal_inventory.loc[
                        proposal_inventory["category"].eq("app"),
                        "declared_app_base_holeout_probability",
                    ].to_numpy(float),
                    atol=ledger_v6.APP_SUPPORT_MASS_TOLERANCE,
                    rtol=0.0,
                ).all()
            ),
            "anchor_enabled": False,
            "anchor_authorized": False,
            "anchor_ready_all_donor_abilities": bool(
                complete_ledger["donor_ability_available"].all()
            ),
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
            "target_manifest_sha256": _sha256(targets_manifest_path),
            "target_parquet_sha256": _sha256(targets_path),
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
        "output_logical_sha256": {
            "ledger": ledger_logical_sha,
            "telescope_audit": audit_logical_sha,
            "proposal_support_atoms": proposal_logical_sha,
            "proposal_support_inventory": inventory_logical_sha,
            "plain_parent_result_parity": result_logical_sha,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    report = (
        "# R1 all-18 complete transition ledger\n\n"
        "Research-only one-player, 128-particle mechanics trace. All 36 traced "
        "hole/arm cases ran before the 36 plain-parent replays. No particle "
        "filtering or support discovery is permitted.\n\n"
        f"Ledger rows: {len(complete_ledger):,}; traced paths: "
        f"{axis_audit['traced_paths']:,}; proposal measure: "
        f"{len(proposal_inventory):,} supports / {len(proposal_atoms):,} atoms.\n\n"
        f"Resolved spatial hazard atoms: "
        f"{coverage_totals['resolved_spatial_hazard_atoms']:,}; explicit APP "
        f"holeouts selected: {coverage_totals['explicit_app_holeouts_selected']:,}.\n\n"
        f"Empirical rare-path coverage: "
        f"{'GO' if empirical_coverage_complete else 'HOLD'}. Par-3 initializer "
        f"holes: {par3_initializer_holes}; penalty strokes: "
        f"{coverage_totals['penalty_strokes']:,}. A zero selected count is "
        "reported as missing coverage, never as empirical validation.\n\n"
        f"Strict-prior donor ability coverage: {ability_audit['available_rows']:,}/"
        f"{ability_audit['rows']:,} ({ability_audit['coverage']:.2%}). Missing "
        "ability remains fail-closed.\n\n"
        "Anchor solving and production use are disabled regardless of ability "
        "coverage; this artifact validates mechanics and provenance only.\n"
    )
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    ledger_v6._output_index(output_dir)
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
    parser.add_argument(
        "--target-input-dir", type=Path, default=ledger_v6.DEFAULT_TARGET_DIR
    )
    parser.add_argument("--bridge", type=Path, default=ledger_v6.asof.DEFAULT_BRIDGE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = build_r1_all18(
        output_dir=args.output_dir,
        export_root=args.export_root,
        history_db=args.history_db,
        profile_dir=args.profile_dir,
        baseline_cases=args.baseline_cases,
        streaming_reference_dir=args.streaming_reference_dir,
        reference_value_dir=args.reference_value_dir,
        target_input_dir=args.target_input_dir,
        bridge_path=args.bridge,
    )
    print(output)


if __name__ == "__main__":
    main()


__all__ = [
    "ARMS",
    "CONTRACT",
    "CaseKey",
    "DEFAULT_OUTPUT_DIR",
    "FROZEN_V6_SOURCE_SHA256",
    "HOLES",
    "PARTICLES",
    "PlainCase",
    "ROUND_NO",
    "TARGET_EVENT_KEY",
    "TARGET_PLAYER_ID",
    "TracedCase",
    "assemble_case_frames",
    "build_r1_all18",
    "run_plain_parent_cases",
    "run_traced_cases",
    "summarize_hole_coverage",
    "summarize_empirical_coverage",
    "validate_complete_axes",
]
