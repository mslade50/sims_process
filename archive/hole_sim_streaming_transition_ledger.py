"""Archive-only complete-transition trace seam for the streaming shot chain.

The production-shaped streaming cache intentionally keeps only the numeric
arrays needed by the fast kernel.  That is sufficient for exact mechanics,
but it discards the causal donor identity and represents APP hole-outs as a
single scalar Bernoulli probability.  Those two choices make the persisted
cache unsuitable for the category-mean anchor.

This module is the bounded research seam immediately *before* that lossy
conversion.  Its first contract expands one APP support into an explicit
probability measure:

* non-hole-out spatial atoms retain the existing conditional CDF and receive
  total mass ``1 - p_holeout``;
* empirical hole-out rows receive total mass ``p_holeout``; and
* the existing hole-out uniform both preserves the beta-zero Bernoulli flag
  and, through ``u / p_holeout``, deterministically chooses the physical
  terminal donor without consuming another random stream.

The resulting support is deliberately independent from the 50k numeric NPZ.
It must be built from :class:`hole_sim_normalized_atom_precompute.AtomSupportProvider`
while physical shot identities are still present.  Later sections of this
module add bounded path-ledger instrumentation; nothing here is imported by
production or by the frozen streaming kernel.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
from typing import Mapping

import numpy as np
import pandas as pd


ARCHIVE = Path(__file__).resolve().parent
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

import hole_sim_app_transition as app  # noqa: E402
import hole_sim_asof_spatial_profiles as asof  # noqa: E402
import hole_sim_fixed_reference_value as fixed_reference  # noqa: E402
import hole_sim_normalized_atom_precompute as precompute  # noqa: E402
import hole_sim_normalized_chain_streaming as streaming  # noqa: E402
import hole_sim_production_anchor_inputs as production_targets  # noqa: E402
import hole_sim_traced_population_transition as traced_population  # noqa: E402
import hole_sim_transition_mean_chain_adapter as chain_adapter  # noqa: E402


EXPLICIT_APP_HOLEOUT_VERSION = "explicit_app_holeout_support_v1"
APP_SUPPORT_MASS_TOLERANCE = 2e-14
LEDGER_SLICE_VERSION = "streaming_complete_transition_ledger_slice_v3"
DEFAULT_REFERENCE_DIR = Path("permanent_data/hole_sim_fixed_reference_travelers_v2")
DEFAULT_TARGET_DIR = Path("permanent_data/hole_sim_backtest_anchor_inputs_travelers_v2")
DEFAULT_OUTPUT_DIR = Path(
    "permanent_data/hole_sim_streaming_transition_ledger_lowry_r1h6_128_v6"
)
CATEGORY_ABILITY_COLUMN = {
    "ott": "sg_ott",
    "app": "sg_app",
    "arg": "ag_arg",
    "putt": "ag_putt",
}
STRICT_BRIDGE_COLUMNS = (
    "event_key",
    "source_player_id",
    "event_start",
    "skill_date",
    "sg_ott",
    "sg_app",
    "ag_arg",
    "ag_putt",
)
PROPOSAL_SUPPORT_DIGEST_COLUMNS = (
    "arm",
    "support_key",
    "proposal_row_order",
    "target_event_key",
    "target_event_start",
    "target_player_id",
    "target_round_no",
    "target_hole_no",
    "target_par",
    "target_start_lie",
    "target_distance_bucket",
    "atom_id",
    "category",
    "atom_kind",
    "terminal",
    "holed",
    "donor_event_key",
    "donor_player_id",
    "source_shot_key",
    "base_hierarchy_mass",
    "spatial_model_weight",
    "quality_base_mass",
    "proposal_probability",
    "proposal_cdf_lower",
    "proposal_cdf_upper",
    "holeout_representation",
    "declared_app_base_holeout_probability",
    "implicit_holeout_probability",
    "app_holeout_equivalence_sha256",
    "proposal_branch",
    "branch_total_mass",
    "branch_conditional_probability",
    "branch_cdf_lower",
    "branch_cdf_upper",
    "source_cache_sha256",
    "numeric_payload_sha256",
)
PROPOSAL_ABILITY_DIGEST_COLUMNS = (
    "arm",
    "support_key",
    "proposal_row_order",
    "atom_id",
    "proposal_support_measure_sha256",
    "donor_event_key",
    "donor_player_id",
    "donor_event_start",
    "donor_skill_date",
    "donor_ability_sg",
    "donor_ability_available",
    "donor_ability_snapshot_kind",
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def _frame_digest(frame: pd.DataFrame, columns: tuple[str, ...]) -> str:
    work = frame.loc[:, list(columns)].copy()
    digest = sha256(_canonical_json(list(columns)).encode("utf-8"))
    for row in work.itertuples(index=False, name=None):
        digest.update(_canonical_json(list(row)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_verified_parquet(
    frame: pd.DataFrame,
    path: Path,
) -> tuple[pd.DataFrame, str]:
    """Write one frame, verify exact values after reload, and hash that reload.

    ``frame_logical_sha256`` deliberately includes pandas dtypes.  Hashing an
    in-memory frame before Parquet normalizes nullable/object columns can
    therefore produce an honest byte artifact with a logical hash that cannot
    be reproduced by a consumer.  Every ledger artifact uses this helper so
    the declared logical identity is the identity a reader actually observes.
    """

    path = Path(path)
    frame.to_parquet(path, index=False)
    persisted = pd.read_parquet(path)
    if list(persisted.columns) != list(frame.columns):
        raise AssertionError(f"Parquet column order changed for {path.name}")
    if persisted.shape != frame.shape:
        raise AssertionError(f"Parquet shape changed for {path.name}")
    pd.testing.assert_frame_equal(
        frame.reset_index(drop=True),
        persisted.reset_index(drop=True),
        check_dtype=False,
        check_exact=True,
        check_categorical=False,
        obj=path.name,
    )
    logical_sha = streaming.frame_logical_sha256(persisted)
    if logical_sha != streaming.frame_logical_sha256(pd.read_parquet(path)):
        raise AssertionError(f"Parquet logical hash is not reproducible: {path.name}")
    return persisted, logical_sha


def _hole_result_frame(
    result: streaming.StreamingHoleResult,
    *,
    arm: str,
    implementation: str,
) -> pd.DataFrame:
    """Flatten every ``StreamingHoleResult`` field into a durable frame."""

    output: dict[str, object] = {
        "arm": np.full(len(result.particle_id), str(arm), dtype=object),
        "implementation": np.full(
            len(result.particle_id), str(implementation), dtype=object
        ),
    }
    for name in result.__dataclass_fields__:
        value = np.asarray(getattr(result, name))
        if value.ndim == 1:
            output[name] = value
        elif value.ndim == 2 and value.shape[1] == 2:
            output[f"{name}_long_yd"] = value[:, 0]
            output[f"{name}_lat_yd"] = value[:, 1]
        else:
            raise ValueError(f"Unsupported result field shape {name}={value.shape}")
    return pd.DataFrame(output)


def _assert_hole_results_exact(
    traced: streaming.StreamingHoleResult,
    plain: streaming.StreamingHoleResult,
    *,
    arm: str,
) -> dict[str, object]:
    """Require the traced subclass to leave every parent result bit-exact."""

    mismatches: dict[str, int] = {}
    for name in traced.__dataclass_fields__:
        left = np.asarray(getattr(traced, name))
        right = np.asarray(getattr(plain, name))
        if left.shape != right.shape:
            raise AssertionError(
                f"Plain-parent parity shape changed for {arm}.{name}: "
                f"{left.shape} != {right.shape}"
            )
        if left.dtype.kind in "fc" or right.dtype.kind in "fc":
            equal = np.isclose(left, right, atol=0.0, rtol=0.0, equal_nan=True)
        else:
            equal = left == right
        mismatches[name] = int((~equal).sum())
    if any(mismatches.values()):
        raise AssertionError(f"Plain-parent result parity failed for {arm}: {mismatches}")
    return {
        "arm": str(arm),
        "exact": True,
        "particles": int(len(traced.particle_id)),
        "mismatches": mismatches,
    }


def _right_cdf(probability: np.ndarray) -> np.ndarray:
    mass = np.asarray(probability, dtype=float)
    if mass.ndim != 1 or not len(mass):
        raise ValueError("A support probability vector must be non-empty and 1D")
    if not np.isfinite(mass).all() or (mass <= 0.0).any():
        raise ValueError("Every support atom needs finite positive probability")
    total = float(mass.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Support probability has invalid total mass")
    mass = mass / total
    cdf = np.cumsum(mass)
    cdf[-1] = 1.0
    if (np.diff(cdf) <= 0.0).any():
        raise ValueError("Support CDF is not strictly increasing")
    return cdf


def _select_right(cdf: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    values = np.asarray(uniforms, dtype=float)
    if not np.isfinite(values).all() or (values < 0.0).any() or (values >= 1.0).any():
        raise ValueError("Selection uniforms must be finite in [0, 1)")
    selected = np.searchsorted(np.asarray(cdf, float), values, side="right")
    np.minimum(selected, len(cdf) - 1, out=selected)
    return selected


@dataclass(frozen=True)
class ExplicitAppSupport:
    """One APP state support with physical terminal donors made explicit."""

    key: str
    rows: pd.DataFrame
    probability: np.ndarray
    cdf: np.ndarray
    declared_holeout_probability: float
    support_sha256: str

    @property
    def terminal_mask(self) -> np.ndarray:
        return self.rows["terminal"].to_numpy(bool)

    @property
    def spatial_mask(self) -> np.ndarray:
        return ~self.terminal_mask

    def select_complete(self, uniforms: np.ndarray) -> pd.DataFrame:
        """Select directly from the complete explicit probability measure."""

        index = _select_right(self.cdf, uniforms)
        result = self.rows.iloc[index].copy().reset_index(drop=True)
        result.insert(0, "selected_support_row", index)
        return result

    def select_preserving_legacy_streams(
        self,
        *,
        spatial_atom_ids: np.ndarray,
        holeout_uniform: np.ndarray,
    ) -> pd.DataFrame:
        """Attach physical donors without changing legacy APP draw semantics.

        ``spatial_atom_ids`` are the identities already selected by the
        existing APP index stream.  For non-hole-outs those remain the causal
        donors.  For hole-outs, the historical scalar decision is unchanged
        and ``holeout_uniform / p`` selects among the explicit empirical
        terminal atoms.
        """

        ids = np.asarray(spatial_atom_ids).astype(str)
        u = np.asarray(holeout_uniform, dtype=float)
        if ids.ndim != 1 or u.ndim != 1 or len(ids) != len(u):
            raise ValueError("APP selected IDs and holeout uniforms must align")
        if not np.isfinite(u).all() or (u < 0.0).any() or (u >= 1.0).any():
            raise ValueError("Holeout uniforms must be finite in [0, 1)")

        spatial = self.rows.loc[self.spatial_mask].copy()
        if spatial["atom_id"].duplicated().any():
            raise ValueError("Spatial APP atom IDs are not unique")
        spatial_lookup = spatial.set_index("atom_id", drop=False)
        missing = sorted(set(ids) - set(spatial_lookup.index))
        if missing:
            raise KeyError(f"Selected spatial APP atom IDs are absent: {missing[:5]}")

        result = spatial_lookup.loc[ids].copy().reset_index(drop=True)
        result.insert(0, "selected_spatial_proposal_atom_id", ids)
        p = float(self.declared_holeout_probability)
        holed = u < p
        if holed.any():
            terminal = self.rows.loc[self.terminal_mask].copy().reset_index(drop=True)
            if p <= 0.0 or terminal.empty:
                raise AssertionError("Positive APP holeout decisions lack terminal donors")
            terminal_probability = terminal["quality_base_mass"].to_numpy(float) / p
            terminal_cdf = _right_cdf(terminal_probability)
            residual = u[holed] / p
            terminal_index = _select_right(terminal_cdf, residual)
            replacement = terminal.iloc[terminal_index].copy()
            replacement.index = result.index[holed]
            for column in result.columns:
                if column == "selected_spatial_proposal_atom_id":
                    continue
                result.loc[holed, column] = replacement[column]
        result["legacy_holeout_decision"] = holed
        result["holeout_selection_residual_uniform"] = np.where(
            holed, u / max(p, np.finfo(float).tiny), np.nan
        )
        if not np.array_equal(result["terminal"].to_numpy(bool), holed):
            raise AssertionError("Explicit APP donor selection changed holeout flags")
        return result


def build_explicit_app_support(
    *,
    provider: precompute.AtomSupportProvider,
    spatial_support: precompute.SupportArrays | object,
    round_no: int,
    hole_no: int,
    start_lie: str,
    distance_bucket: int,
    player_id: str,
) -> ExplicitAppSupport:
    """Expand the current scalar APP-holeout contract into physical atoms.

    The function accepts either ``SupportArrays`` or the streaming
    ``NumericSupport`` duck type.  The support must retain physical string
    ``shot_key`` identities; local integer IDs from a persisted 50k NPZ are
    rejected because they cannot be enriched after the fact.
    """

    normalized_lie = app._normalize_start_lie(start_lie)
    distance_center = int(distance_bucket) * 15.0
    distribution = app.build_distribution(
        provider.atoms.raw_course_app,
        int(round_no),
        int(hole_no),
        normalized_lie,
        distance_center,
    )
    rows = distribution.rows.copy().reset_index(drop=True)
    weights = np.asarray(distribution.weights, dtype=float)
    if len(rows) != len(weights):
        raise AssertionError("APP distribution rows and weights diverged")
    holeout_mask = rows["is_holeout"].to_numpy(bool)
    spatial_mask = rows["spatial_valid"].to_numpy(bool) & ~holeout_mask
    if int(spatial_mask.sum()) < 12:
        raise ValueError("APP support has fewer than 12 spatial atoms")

    source_spatial = rows.loc[spatial_mask].copy().reset_index(drop=True)
    source_ids = source_spatial["shot_key"].astype(str).to_numpy()
    support_ids = np.asarray(getattr(spatial_support, "atom_ids"))
    if support_ids.dtype.kind not in {"O", "U", "S"}:
        raise ValueError(
            "Explicit APP support requires physical shot IDs, not local numeric IDs"
        )
    support_ids = support_ids.astype(str)
    if not np.array_equal(support_ids, source_ids):
        raise AssertionError("APP numeric support no longer aligns to source shot IDs")

    conditional_source_mass = weights[spatial_mask].astype(float)
    conditional_source_mass /= conditional_source_mass.sum()
    support_probability = np.asarray(
        getattr(spatial_support, "weights", getattr(spatial_support, "probability", None)),
        dtype=float,
    )
    if len(support_probability) != len(source_spatial):
        raise AssertionError("APP numeric probability does not align to source rows")
    support_probability = support_probability / support_probability.sum()
    declared_holeout = float(weights[holeout_mask].sum())
    if not np.isclose(
        declared_holeout,
        float(getattr(spatial_support, "holeout_probability")),
        atol=0.0,
        rtol=0.0,
    ):
        raise AssertionError("APP declared holeout probability changed")

    spatial = source_spatial.copy()
    spatial.insert(0, "atom_id", source_ids)
    spatial["atom_kind"] = "app_spatial"
    spatial["terminal"] = False
    spatial["holed"] = False
    spatial["base_hierarchy_mass"] = (
        (1.0 - declared_holeout) * conditional_source_mass
    )
    denominator = np.maximum(conditional_source_mass, np.finfo(float).tiny)
    spatial["spatial_model_weight"] = support_probability / denominator
    spatial["quality_base_mass"] = (
        (1.0 - declared_holeout) * support_probability
    )

    terminal_source = rows.loc[holeout_mask].copy().reset_index(drop=True)
    if declared_holeout > 0.0 and terminal_source.empty:
        raise AssertionError("Positive APP holeout mass has no physical donor rows")
    if declared_holeout == 0.0 and not terminal_source.empty:
        raise AssertionError("Zero APP holeout mass unexpectedly has donor rows")
    terminal = terminal_source.copy()
    if not terminal.empty:
        terminal_ids = terminal["shot_key"].astype(str).to_numpy()
        if len(set(terminal_ids)) != len(terminal_ids):
            raise ValueError("APP holeout donor shot IDs are duplicated")
        terminal.insert(0, "atom_id", terminal_ids)
        terminal["atom_kind"] = "app_holeout"
        terminal["terminal"] = True
        terminal["holed"] = True
        # These are unconditional empirical masses.  Their sum is the exact
        # scalar branch probability used by the legacy kernel.
        terminal["base_hierarchy_mass"] = weights[holeout_mask]
        terminal["spatial_model_weight"] = 1.0
        terminal["quality_base_mass"] = weights[holeout_mask]

    complete = pd.concat([spatial, terminal], ignore_index=True, sort=False)
    complete["category"] = "app"
    complete["support_key"] = str(getattr(spatial_support, "key"))
    complete["holeout_representation"] = EXPLICIT_APP_HOLEOUT_VERSION
    complete["declared_app_base_holeout_probability"] = declared_holeout
    complete["implicit_holeout_probability"] = 0.0
    complete_mass = complete["quality_base_mass"].to_numpy(float)
    if not np.isclose(
        complete_mass.sum(), 1.0, atol=APP_SUPPORT_MASS_TOLERANCE, rtol=0.0
    ):
        raise AssertionError("Explicit APP complete support does not sum to one")
    observed_holeout = float(complete.loc[complete["terminal"], "quality_base_mass"].sum())
    if not np.isclose(
        observed_holeout,
        declared_holeout,
        atol=APP_SUPPORT_MASS_TOLERANCE,
        rtol=0.0,
    ):
        raise AssertionError("Explicit APP terminal mass differs from scalar branch")
    if not np.allclose(
        complete["quality_base_mass"].to_numpy(float),
        complete["base_hierarchy_mass"].to_numpy(float)
        * complete["spatial_model_weight"].to_numpy(float),
        atol=APP_SUPPORT_MASS_TOLERANCE,
        rtol=1e-12,
    ):
        raise AssertionError("APP mass does not equal hierarchy mass times model weight")

    probability = complete_mass / complete_mass.sum()
    cdf = _right_cdf(probability)
    digest_columns = (
        "support_key",
        "atom_id",
        "atom_kind",
        "terminal",
        "holed",
        "base_hierarchy_mass",
        "spatial_model_weight",
        "quality_base_mass",
        "declared_app_base_holeout_probability",
        "implicit_holeout_probability",
    )
    support_sha = _frame_digest(complete, digest_columns)
    complete["app_holeout_equivalence_sha256"] = support_sha
    return ExplicitAppSupport(
        key=str(getattr(spatial_support, "key")),
        rows=complete,
        probability=probability,
        cdf=cdf,
        declared_holeout_probability=declared_holeout,
        support_sha256=support_sha,
    )


class IdentityPreservingSupportRepository(streaming.StreamingSupportRepository):
    """Provider-backed numeric repository that retains physical donor rows.

    This class is intentionally bounded to research traces.  Unlike the fast
    streaming repository it never accepts a persisted NPZ and never replaces
    physical ``shot_key`` values with local integers.  The numeric arrays are
    still built by the authoritative provider, so selection remains identical
    to the streaming kernel.
    """

    def __init__(
        self,
        *,
        arm: str,
        provider: precompute.AtomSupportProvider,
    ) -> None:
        super().__init__(
            arm=str(arm),
            provider=provider,
            fail_on_miss=False,
            keep_atom_ids=True,
        )
        self.rich_supports: dict[str, pd.DataFrame] = {}
        self.explicit_app_supports: dict[str, ExplicitAppSupport] = {}

    def _store_numeric(
        self,
        support: precompute.SupportArrays,
    ) -> streaming.NumericSupport:
        numeric = self._convert(support)
        if numeric.key in self.supports:
            existing = self.supports[numeric.key]
            if existing.payload_sha256 != numeric.payload_sha256:
                raise AssertionError("A traced support key changed payload")
            return existing
        self.supports[numeric.key] = numeric
        self.materialization_calls += 1
        self.materialized_atoms += len(numeric.atom_ids)
        return numeric

    def tee(
        self, round_no: int, hole_no: int, par: int, player_id: str
    ) -> streaming.NumericSupport:
        key = self.tee_key(self.arm, round_no, hole_no, par, player_id)
        existing = self.supports.get(key)
        if existing is not None:
            self.hit_count += 1
            return existing
        self.miss_count += 1
        started = time.perf_counter()
        source_rows, raw_probability, _ = precompute._tee_distribution_rows(
            self.provider.atoms.raw_course_drives,
            int(round_no),
            int(hole_no),
            int(par),
        )
        support = self.provider.tee_support(
            round_no=int(round_no),
            hole_no=int(hole_no),
            par=int(par),
            player_id=str(player_id),
        )
        numeric = self._store_numeric(support)
        source_rows = source_rows.copy().reset_index(drop=True)
        source_ids = source_rows["shot_key"].astype(str).to_numpy()
        if not np.array_equal(source_ids, numeric.atom_ids.astype(str)):
            raise AssertionError("TEE support no longer aligns to source donor rows")
        raw_probability = np.asarray(raw_probability, float)
        raw_probability /= raw_probability.sum()
        probability = numeric.probability / numeric.probability.sum()
        rich = source_rows.copy()
        rich.insert(0, "atom_id", source_ids)
        rich["support_key"] = key
        rich["category"] = "ott"
        rich["atom_kind"] = "tee_spatial"
        rich["terminal"] = False
        rich["holed"] = False
        rich["base_hierarchy_mass"] = raw_probability
        rich["spatial_model_weight"] = probability / np.maximum(
            raw_probability, np.finfo(float).tiny
        )
        rich["quality_base_mass"] = probability
        rich["holeout_representation"] = "not_applicable"
        rich["declared_app_base_holeout_probability"] = 0.0
        rich["implicit_holeout_probability"] = 0.0
        if not np.allclose(
            rich["quality_base_mass"],
            rich["base_hierarchy_mass"] * rich["spatial_model_weight"],
            rtol=1e-12,
            atol=APP_SUPPORT_MASS_TOLERANCE,
        ):
            raise AssertionError("TEE hierarchy and spatial masses diverged")
        self.rich_supports[key] = rich
        self.materialization_seconds += time.perf_counter() - started
        return numeric

    def app(
        self,
        round_no: int,
        hole_no: int,
        start_lie: str,
        bucket: int,
        player_id: str,
    ) -> streaming.NumericSupport:
        key = self.app_key(
            self.arm, round_no, hole_no, start_lie, bucket, player_id
        )
        existing = self.supports.get(key)
        if existing is not None:
            self.hit_count += 1
            return existing
        self.miss_count += 1
        started = time.perf_counter()
        support = self.provider.app_support(
            round_no=int(round_no),
            hole_no=int(hole_no),
            start_lie=str(start_lie),
            distance_bucket=int(bucket),
            player_id=str(player_id),
        )
        numeric = self._store_numeric(support)
        explicit = build_explicit_app_support(
            provider=self.provider,
            spatial_support=numeric,
            round_no=int(round_no),
            hole_no=int(hole_no),
            start_lie=str(start_lie),
            distance_bucket=int(bucket),
            player_id=str(player_id),
        )
        if explicit.key != key:
            raise AssertionError("Explicit APP support key changed")
        self.explicit_app_supports[key] = explicit
        self.rich_supports[key] = explicit.rows
        self.materialization_seconds += time.perf_counter() - started
        return numeric

    def donor_rows(self, key: str, atom_ids: np.ndarray) -> pd.DataFrame:
        try:
            rows = self.rich_supports[str(key)]
        except KeyError as exc:
            raise KeyError(f"Traced support rows were not materialized: {key}") from exc
        if rows["atom_id"].duplicated().any():
            raise AssertionError(f"Traced support {key} has duplicate atom IDs")
        lookup = rows.set_index("atom_id", drop=False)
        ids = np.asarray(atom_ids).astype(str)
        missing = sorted(set(ids) - set(lookup.index))
        if missing:
            raise KeyError(f"Traced donors are absent from {key}: {missing[:5]}")
        return lookup.loc[ids].copy().reset_index(drop=True)

    def explicit_app(self, key: str) -> ExplicitAppSupport:
        try:
            return self.explicit_app_supports[str(key)]
        except KeyError as exc:
            raise KeyError(f"Explicit APP support was not materialized: {key}") from exc


def materialize_proposal_support_contract(
    repositories: Mapping[str, IdentityPreservingSupportRepository],
    *,
    target_event_key: str,
    target_event_start: object,
    target_player_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Persist the complete materialized TEE/APP proposal measures.

    These rows are *proposal* atoms, not yet complete generated transitions:
    spatial surface and hazard outcomes are resolved only in the path ledger.
    Their purpose is to prove that every physical donor identity and every
    probability mass used by the bounded trace survived the numeric seam.
    """

    support_frames: list[pd.DataFrame] = []
    inventory_rows: list[dict[str, object]] = []
    for arm, repository in sorted(repositories.items()):
        for support_key, source in sorted(repository.rich_supports.items()):
            if support_key not in repository.supports:
                raise AssertionError(f"Rich support lacks numeric payload: {support_key}")
            numeric = repository.supports[support_key]
            frame = source.copy().reset_index(drop=True)
            if frame.empty:
                raise AssertionError(f"Proposal support is empty: {support_key}")
            if not frame["support_key"].astype(str).eq(str(support_key)).all():
                raise AssertionError(f"Proposal support key drift: {support_key}")
            if frame["atom_id"].astype(str).duplicated().any():
                raise AssertionError(f"Proposal atom IDs are duplicated: {support_key}")
            parts = str(support_key).split("|")
            if len(parts) not in {6, 7}:
                raise AssertionError(f"Proposal support key is malformed: {support_key}")
            support_category = str(parts[0])
            if support_category == "tee":
                _, key_arm, round_text, hole_text, par_text, key_player = parts
                start_lie = "tee"
                distance_bucket = -1
                par = int(str(par_text).removeprefix("P"))
            elif support_category == "app":
                (
                    _, key_arm, round_text, hole_text, start_lie,
                    bucket_text, key_player,
                ) = parts
                distance_bucket = int(str(bucket_text).removeprefix("B"))
                par = -1
            else:
                raise AssertionError(f"Unknown proposal support category: {support_key}")
            target_round_no = int(str(round_text).removeprefix("R"))
            target_hole_no = int(str(hole_text).removeprefix("H"))
            if key_arm != str(arm) or str(key_player) != str(target_player_id):
                raise AssertionError(f"Proposal support context drift: {support_key}")

            def optional(name: str, default: object) -> pd.Series:
                if name in frame.columns:
                    return frame[name].copy()
                return pd.Series([default] * len(frame), index=frame.index)

            compact = pd.DataFrame(
                {
                    "arm": str(arm),
                    "support_key": frame["support_key"].astype(str),
                    "proposal_row_order": np.arange(len(frame), dtype=np.int32),
                    "target_event_key": str(target_event_key),
                    "target_event_start": pd.Timestamp(target_event_start),
                    "target_player_id": str(target_player_id),
                    "target_round_no": int(target_round_no),
                    "target_hole_no": int(target_hole_no),
                    "target_par": int(par),
                    "target_start_lie": str(start_lie),
                    "target_distance_bucket": int(distance_bucket),
                    "atom_id": frame["atom_id"].astype(str),
                    "category": frame["category"].astype(str),
                    "atom_kind": frame["atom_kind"].astype(str),
                    "terminal": frame["terminal"].astype(bool),
                    "holed": frame["holed"].astype(bool),
                    "donor_event_key": optional("event_key", "").astype(str),
                    "donor_player_id": optional("source_player_id", "").astype(str),
                    "donor_player_name": optional("player_name", "").astype(str),
                    "donor_round_no": pd.to_numeric(
                        optional("round_no", -1), errors="coerce"
                    ).fillna(-1).astype(np.int16),
                    "donor_hole_no": pd.to_numeric(
                        optional("hole_no", -1), errors="coerce"
                    ).fillna(-1).astype(np.int16),
                    "donor_stroke_no": pd.to_numeric(
                        optional("stroke_no", -1), errors="coerce"
                    ).fillna(-1).astype(np.int16),
                    "source_shot_key": optional("shot_key", "").astype(str),
                    "source_start_lie": optional("start_lie", "").astype(str),
                    "source_start_distance_yd": pd.to_numeric(
                        optional("start_distance_yd", np.nan), errors="coerce"
                    ).astype(float),
                    "source_end_lie": optional("end_lie", "").astype(str),
                    "source_end_distance_yd": pd.to_numeric(
                        optional("end_distance_yd", np.nan), errors="coerce"
                    ).astype(float),
                    "source_sg_raw": pd.to_numeric(
                        optional("sg_raw", np.nan), errors="coerce"
                    ).astype(float),
                    "source_model_id": optional("model_id", "").astype(str),
                    "base_hierarchy_mass": pd.to_numeric(
                        frame["base_hierarchy_mass"], errors="raise"
                    ).astype(float),
                    "spatial_model_weight": pd.to_numeric(
                        frame["spatial_model_weight"], errors="raise"
                    ).astype(float),
                    "quality_base_mass": pd.to_numeric(
                        frame["quality_base_mass"], errors="raise"
                    ).astype(float),
                    "holeout_representation": frame[
                        "holeout_representation"
                    ].astype(str),
                    "declared_app_base_holeout_probability": pd.to_numeric(
                        frame["declared_app_base_holeout_probability"],
                        errors="raise",
                    ).astype(float),
                    "implicit_holeout_probability": pd.to_numeric(
                        frame["implicit_holeout_probability"], errors="raise"
                    ).astype(float),
                    "app_holeout_equivalence_sha256": optional(
                        "app_holeout_equivalence_sha256", "not_applicable"
                    ).astype(str),
                    "source_cache_sha256": str(numeric.cache_sha256),
                    "numeric_payload_sha256": str(numeric.payload_sha256),
                }
            )
            if not compact["source_shot_key"].eq(compact["atom_id"]).all():
                raise AssertionError(
                    f"Proposal physical shot identity changed: {support_key}"
                )
            mass = compact["quality_base_mass"].to_numpy(float)
            hierarchy = compact["base_hierarchy_mass"].to_numpy(float)
            model_weight = compact["spatial_model_weight"].to_numpy(float)
            if not np.isfinite(mass).all() or (mass <= 0.0).any():
                raise AssertionError(f"Proposal mass is invalid: {support_key}")
            if not np.isclose(
                mass.sum(), 1.0, atol=APP_SUPPORT_MASS_TOLERANCE, rtol=0.0
            ):
                raise AssertionError(f"Proposal mass does not sum to one: {support_key}")
            if not np.allclose(
                mass,
                hierarchy * model_weight,
                atol=APP_SUPPORT_MASS_TOLERANCE,
                rtol=1e-12,
            ):
                raise AssertionError(f"Proposal mass factorization failed: {support_key}")
            probability = mass / mass.sum()
            proposal_upper = np.cumsum(probability)
            proposal_upper[-1] = 1.0
            compact["proposal_probability"] = probability
            compact["proposal_cdf_lower"] = np.concatenate(
                ([0.0], proposal_upper[:-1])
            )
            compact["proposal_cdf_upper"] = proposal_upper
            categories = compact["category"].unique().tolist()
            if len(categories) != 1 or categories[0] not in {"ott", "app"}:
                raise AssertionError(f"Proposal category is invalid: {support_key}")
            category = str(categories[0])
            terminal_mass = float(
                compact.loc[compact["terminal"], "quality_base_mass"].sum()
            )
            spatial_mass = float(
                compact.loc[~compact["terminal"], "quality_base_mass"].sum()
            )
            declared_values = compact[
                "declared_app_base_holeout_probability"
            ].unique()
            if len(declared_values) != 1:
                raise AssertionError(f"Proposal holeout declaration varies: {support_key}")
            declared = float(declared_values[0])
            if category == "app":
                if not np.isclose(
                    terminal_mass,
                    declared,
                    atol=APP_SUPPORT_MASS_TOLERANCE,
                    rtol=0.0,
                ):
                    raise AssertionError(
                        f"Explicit APP terminal mass does not match declaration: {support_key}"
                    )
                if not np.isclose(
                    spatial_mass,
                    1.0 - declared,
                    atol=APP_SUPPORT_MASS_TOLERANCE,
                    rtol=0.0,
                ):
                    raise AssertionError(
                        f"Explicit APP spatial mass does not match declaration: {support_key}"
                    )
                if compact["implicit_holeout_probability"].ne(0.0).any():
                    raise AssertionError(f"APP support retains implicit mass: {support_key}")
                if not compact.loc[compact["terminal"], "holed"].all():
                    raise AssertionError(f"APP terminal atom is not holed: {support_key}")
                if compact.loc[~compact["terminal"], "holed"].any():
                    raise AssertionError(f"APP nonterminal atom is holed: {support_key}")
                equivalence = compact[
                    "app_holeout_equivalence_sha256"
                ].unique().tolist()
                if len(equivalence) != 1:
                    raise AssertionError(f"APP equivalence hash varies: {support_key}")
                original_digest = _frame_digest(
                    frame,
                    (
                        "support_key",
                        "atom_id",
                        "atom_kind",
                        "terminal",
                        "holed",
                        "base_hierarchy_mass",
                        "spatial_model_weight",
                        "quality_base_mass",
                        "declared_app_base_holeout_probability",
                        "implicit_holeout_probability",
                    ),
                )
                if equivalence[0] != original_digest:
                    raise AssertionError(f"APP equivalence digest changed: {support_key}")
            elif terminal_mass != 0.0 or compact["terminal"].any():
                raise AssertionError(f"TEE proposal unexpectedly terminates: {support_key}")

            compact["proposal_branch"] = np.where(
                compact["terminal"], "terminal", "spatial"
            )
            compact["branch_total_mass"] = 0.0
            compact["branch_conditional_probability"] = 0.0
            compact["branch_cdf_lower"] = 0.0
            compact["branch_cdf_upper"] = 0.0
            for branch, branch_frame in compact.groupby(
                "proposal_branch", observed=True, sort=False
            ):
                branch_index = branch_frame.index.to_numpy()
                branch_mass = compact.loc[
                    branch_index, "quality_base_mass"
                ].to_numpy(float)
                total = float(branch_mass.sum())
                if total <= 0.0:
                    raise AssertionError(
                        f"Proposal branch has nonpositive mass: {support_key}/{branch}"
                    )
                conditional = branch_mass / total
                upper = np.cumsum(conditional)
                upper[-1] = 1.0
                lower = np.concatenate(([0.0], upper[:-1]))
                compact.loc[branch_index, "branch_total_mass"] = total
                compact.loc[
                    branch_index, "branch_conditional_probability"
                ] = conditional
                compact.loc[branch_index, "branch_cdf_lower"] = lower
                compact.loc[branch_index, "branch_cdf_upper"] = upper
                if not np.isclose(upper[-1], 1.0, atol=0.0, rtol=0.0):
                    raise AssertionError(
                        f"Proposal branch CDF does not close: {support_key}/{branch}"
                    )

            measure_sha = _frame_digest(compact, PROPOSAL_SUPPORT_DIGEST_COLUMNS)
            compact["proposal_support_measure_sha256"] = measure_sha
            support_frames.append(compact)
            inventory_rows.append(
                {
                    "arm": str(arm),
                    "support_key": str(support_key),
                    "category": category,
                    "atoms": int(len(compact)),
                    "spatial_atoms": int((~compact["terminal"]).sum()),
                    "terminal_atoms": int(compact["terminal"].sum()),
                    "quality_mass": float(mass.sum()),
                    "spatial_mass": spatial_mass,
                    "terminal_mass": terminal_mass,
                    "declared_app_base_holeout_probability": declared,
                    "implicit_holeout_probability": float(
                        compact["implicit_holeout_probability"].max()
                    ),
                    "source_cache_sha256": str(numeric.cache_sha256),
                    "numeric_payload_sha256": str(numeric.payload_sha256),
                    "proposal_support_measure_sha256": measure_sha,
                    "app_holeout_equivalence_sha256": str(
                        compact["app_holeout_equivalence_sha256"].iloc[0]
                    ),
                }
            )
    if not support_frames:
        raise ValueError("No proposal supports were materialized")
    atoms = pd.concat(support_frames, ignore_index=True, sort=False)
    inventory = pd.DataFrame(inventory_rows).sort_values(
        ["arm", "support_key"], kind="mergesort"
    ).reset_index(drop=True)
    if atoms.groupby(["arm", "support_key"], observed=True).ngroups != len(inventory):
        raise AssertionError("Proposal support inventory cardinality changed")
    return atoms, inventory


@dataclass(frozen=True)
class TracedHoleExecution:
    """One exact streaming result plus its complete causal transition ledger."""

    result: streaming.StreamingHoleResult
    diagnostics: streaming.KernelDiagnostics
    ledger: pd.DataFrame
    telescope_audit: pd.DataFrame


class TracedStreamingShotChain(streaming.StreamingShotChain):
    """Bounded identity-preserving wrapper around ``StreamingShotChain``.

    The inherited state machine and RNG contract remain authoritative.  TEE
    and APP selection are inspected immediately before the parent methods run;
    ARG/PUTT use the separately parity-checked traced batch sampler and then
    execute the same state mutation as the parent implementation.
    """

    def __init__(
        self,
        *,
        repositories: Mapping[str, IdentityPreservingSupportRepository],
        course_drives: pd.DataFrame,
        course_app: pd.DataFrame,
        traced_state_sampler: traced_population.TracedBatchedPopulationTransitionSampler,
        hazard_resolver: streaming.BatchedEndpointHazardResolver,
        reference_model: fixed_reference.ReferenceValueModel,
    ) -> None:
        for arm, repo in repositories.items():
            if not isinstance(repo, IdentityPreservingSupportRepository):
                raise TypeError(f"Trace repository for {arm} is not identity-preserving")
        self.traced_state_sampler = traced_state_sampler
        self.reference_model = reference_model
        super().__init__(
            repositories=repositories,
            course_drives=course_drives,
            course_app=course_app,
            state_sampler=traced_state_sampler.base_sampler,
            hazard_resolver=hazard_resolver,
        )
        self._trace_rows: list[dict[str, object]] = []
        self._pending: dict[int, dict[str, object]] = {}
        self._particle_ids = np.asarray([], dtype=np.int64)
        self._sequence = np.asarray([], dtype=np.int16)
        self._trace_arm = ""
        self._trace_transition_index = -1

    @staticmethod
    def _physical_lie(code: int) -> str:
        lie = str(streaming.LIE_NAMES[int(code)])
        if lie in {"unknown", "penalty"}:
            raise ValueError(f"Cannot materialize reference value for lie={lie}")
        return lie

    @staticmethod
    def _row_value(row: pd.Series, name: str, default: object = "") -> object:
        value = row[name] if name in row.index else default
        if isinstance(value, float) and np.isnan(value):
            return default
        return value

    def _base_row(
        self,
        *,
        local_index: int,
        category: str,
        support_key: str,
        donor: pd.Series,
        start_lie: str,
        start_distance_yd: float,
        start_endpoint: np.ndarray,
        selected_spatial_proposal_atom_id: str = "not_applicable",
        population_fields: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        sequence = int(self._sequence[int(local_index)])
        self._sequence[int(local_index)] += 1
        row: dict[str, object] = {
            "particle_id": int(self._particle_ids[int(local_index)]),
            "atom_sequence": sequence,
            "arm": self._trace_arm,
            "category": str(category),
            "support_key": str(support_key),
            "atom_id": str(self._row_value(donor, "atom_id", self._row_value(donor, "source_shot_key", ""))),
            "selected_spatial_proposal_atom_id": str(selected_spatial_proposal_atom_id),
            "donor_event_key": str(self._row_value(donor, "event_key", self._row_value(donor, "source_event_key", ""))),
            "donor_player_id": str(self._row_value(donor, "source_player_id", "")),
            "donor_player_name": str(self._row_value(donor, "player_name", self._row_value(donor, "source_player_name", ""))),
            "donor_round_no": int(self._row_value(donor, "round_no", self._row_value(donor, "source_round_no", -1))),
            "donor_hole_no": int(self._row_value(donor, "hole_no", self._row_value(donor, "source_hole_no", -1))),
            "donor_stroke_no": int(self._row_value(donor, "stroke_no", self._row_value(donor, "source_stroke_no", -1))),
            "start_lie": str(start_lie),
            "start_distance_yd": float(start_distance_yd),
            "start_long_yd": float(start_endpoint[0]),
            "start_lat_yd": float(start_endpoint[1]),
            "causal_strokes": 1,
            "penalty_strokes": 0,
            "stroke_cost": 1,
            "hazard_resolution_status": "none",
            "hazard_donor_key": "not_applicable",
            "base_hierarchy_mass": float(self._row_value(donor, "base_hierarchy_mass", np.nan)),
            "spatial_model_weight": float(self._row_value(donor, "spatial_model_weight", 1.0)),
            "quality_base_mass": float(self._row_value(donor, "quality_base_mass", np.nan)),
            "atom_kind": str(self._row_value(donor, "atom_kind", f"{category}_transition")),
            "legacy_holeout_decision": bool(self._row_value(donor, "legacy_holeout_decision", False)),
            "holeout_representation": str(self._row_value(donor, "holeout_representation", "not_applicable")),
            "declared_app_base_holeout_probability": float(self._row_value(donor, "declared_app_base_holeout_probability", 0.0)),
            "implicit_holeout_probability": float(self._row_value(donor, "implicit_holeout_probability", 0.0)),
            "app_holeout_equivalence_sha256": str(self._row_value(donor, "app_holeout_equivalence_sha256", "not_applicable")),
            "rng_index_uniform": np.nan,
            "rng_lie_uniform": np.nan,
            "rng_holeout_uniform": np.nan,
            "rng_holeout_residual_uniform": np.nan,
            "rng_population_uniform": np.nan,
            "rng_hazard_uniform": np.nan,
            "rng_shot_transition_index": int(self._trace_transition_index),
            "rng_hazard_transition_index": np.nan,
        }
        if population_fields:
            row.update(dict(population_fields))
        return row

    def _finish_row(
        self,
        row: dict[str, object],
        *,
        local_index: int,
        state: streaming.HoleState,
    ) -> None:
        index = int(local_index)
        terminal = bool(state.terminal[index] and not state.invalid[index])
        end_lie = "hole" if terminal else self._physical_lie(int(state.lie[index]))
        end_distance = 0.0 if terminal else float(state.remaining_yd[index])
        if not np.isfinite(end_distance) or end_distance < 0.0:
            raise ValueError("A complete traced atom has invalid end distance")
        start_lie = str(row["start_lie"])
        start_distance = float(row["start_distance_yd"])
        start_value = self.reference_model.value(start_lie, start_distance)
        end_value = self.reference_model.value(end_lie, end_distance)
        cost = int(row["stroke_cost"])
        reward = start_value - float(cost) - end_value
        row.update(
            {
                "end_lie": end_lie,
                "end_distance_yd": end_distance,
                "end_long_yd": float(state.endpoint[index, 0]),
                "end_lat_yd": float(state.endpoint[index, 1]),
                "terminal": terminal,
                "holed": terminal,
                "start_state_key": self.reference_model.state_key(
                    start_lie, start_distance
                ),
                "end_state_key": self.reference_model.state_key(
                    end_lie, end_distance
                ),
                "start_reference_value": start_value,
                "end_reference_value": end_value,
                "fixed_reference_reward": reward,
                "reward_reference_descriptor": self.reference_model.descriptor,
                "reward_reference_function_sha256": self.reference_model.model_sha256,
                "reward_reference_putting_sha256": self.reference_model.putting_curve_sha256,
            }
        )
        self._trace_rows.append(row)

    def _sample_tee(
        self,
        *,
        state: streaming.HoleState,
        active: np.ndarray,
        record: object,
        repo: streaming.StreamingSupportRepository,
        index_uniform: np.ndarray,
        lie_uniform: np.ndarray,
        atom_id: np.ndarray,
        diagnostics: streaming.KernelDiagnostics,
    ) -> None:
        if not isinstance(repo, IdentityPreservingSupportRepository):
            raise TypeError("TEE trace requires an identity-preserving repository")
        positions = np.flatnonzero(np.asarray(active, bool))
        if not len(positions):
            return super()._sample_tee(
                state=state, active=active, record=record, repo=repo,
                index_uniform=index_uniform, lie_uniform=lie_uniform,
                atom_id=atom_id, diagnostics=diagnostics,
            )
        support = repo.tee(
            int(record.round_no), int(record.hole_no), int(record.par),
            str(record.source_player_id),
        )
        selected, ids = support.select(np.asarray(index_uniform, float)[positions])
        del selected
        donors = repo.donor_rows(support.key, ids)
        pending_rows: dict[int, dict[str, object]] = {}
        for offset, local_index in enumerate(positions):
            current_lie = int(state.lie[local_index])
            if current_lie == int(streaming.LIE_TEE) and np.isfinite(
                state.remaining_yd[local_index]
            ):
                start_distance = float(state.remaining_yd[local_index])
                start_endpoint = state.endpoint[local_index].copy()
            else:
                start_distance = float(record.forecast_pin_long_yd)
                start_endpoint = np.asarray([0.0, 0.0])
            pending_rows[int(local_index)] = self._base_row(
                local_index=int(local_index),
                category="ott",
                support_key=support.key,
                donor=donors.iloc[offset],
                start_lie="tee",
                start_distance_yd=start_distance,
                start_endpoint=start_endpoint,
            )
            pending_rows[int(local_index)]["rng_index_uniform"] = float(
                np.asarray(index_uniform, float)[int(local_index)]
            )
            pending_rows[int(local_index)]["rng_lie_uniform"] = float(
                np.asarray(lie_uniform, float)[int(local_index)]
            )
        super()._sample_tee(
            state=state, active=active, record=record, repo=repo,
            index_uniform=index_uniform, lie_uniform=lie_uniform,
            atom_id=atom_id, diagnostics=diagnostics,
        )
        for local_index, row in pending_rows.items():
            if state.invalid[local_index]:
                raise ValueError("TEE trace became invalid during sampling")
            if state.status[local_index] == streaming.STATUS_TEE_HAZARD:
                if local_index in self._pending:
                    raise AssertionError("Particle already has a pending causal atom")
                self._pending[local_index] = row
            else:
                self._finish_row(row, local_index=local_index, state=state)

    def _sample_app(
        self,
        *,
        state: streaming.HoleState,
        active: np.ndarray,
        record: object,
        repo: streaming.StreamingSupportRepository,
        uniforms: Mapping[str, np.ndarray],
        atom_id: np.ndarray,
        diagnostics: streaming.KernelDiagnostics,
    ) -> None:
        if not isinstance(repo, IdentityPreservingSupportRepository):
            raise TypeError("APP trace requires an identity-preserving repository")
        eligible = np.asarray(active, bool).copy()
        eligible &= np.isfinite(state.endpoint).all(axis=1)
        eligible &= np.isfinite(state.remaining_yd)
        eligible &= np.isin(
            state.lie,
            np.asarray(
                [
                    streaming.LIE_TEE,
                    streaming.LIE_FAIRWAY,
                    streaming.LIE_ROUGH,
                    streaming.LIE_SAND,
                    streaming.LIE_RECOVERY,
                ]
            ),
        )
        positions = np.flatnonzero(eligible)
        pending_rows: dict[int, dict[str, object]] = {}
        if len(positions):
            buckets = np.maximum(
                1, np.rint(state.remaining_yd[positions] / 15.0).astype(np.int32)
            )
            encoded = state.lie[positions].astype(np.int64) * 100_000 + buckets
            unique, inverse = np.unique(encoded, return_inverse=True)
            for group_index, code in enumerate(unique):
                group_positions = positions[inverse == group_index]
                lie_code = int(code // 100_000)
                bucket = int(code % 100_000)
                start_lie = str(streaming.APP_START_NAMES[lie_code])
                support = repo.app(
                    int(record.round_no), int(record.hole_no), start_lie,
                    bucket, str(record.source_player_id),
                )
                _, selected_ids = support.select(
                    np.asarray(uniforms["index"], float)[group_positions]
                )
                explicit = repo.explicit_app(support.key)
                selected = explicit.select_preserving_legacy_streams(
                    spatial_atom_ids=selected_ids,
                    holeout_uniform=np.asarray(uniforms["holeout"], float)[
                        group_positions
                    ],
                )
                for offset, local_index in enumerate(group_positions):
                    donor = selected.iloc[offset]
                    pending_rows[int(local_index)] = self._base_row(
                        local_index=int(local_index),
                        category="app",
                        support_key=support.key,
                        donor=donor,
                        start_lie=start_lie,
                        start_distance_yd=float(state.remaining_yd[local_index]),
                        start_endpoint=state.endpoint[local_index].copy(),
                        selected_spatial_proposal_atom_id=str(selected_ids[offset]),
                    )
                    pending_rows[int(local_index)]["rng_index_uniform"] = float(
                        np.asarray(uniforms["index"], float)[int(local_index)]
                    )
                    pending_rows[int(local_index)]["rng_lie_uniform"] = float(
                        np.asarray(uniforms["lie"], float)[int(local_index)]
                    )
                    pending_rows[int(local_index)]["rng_holeout_uniform"] = float(
                        np.asarray(uniforms["holeout"], float)[int(local_index)]
                    )
                    pending_rows[int(local_index)][
                        "rng_holeout_residual_uniform"
                    ] = float(
                        donor["holeout_selection_residual_uniform"]
                    ) if bool(donor["legacy_holeout_decision"]) else np.nan
        super()._sample_app(
            state=state, active=active, record=record, repo=repo,
            uniforms=uniforms, atom_id=atom_id, diagnostics=diagnostics,
        )
        for local_index, row in pending_rows.items():
            if state.invalid[local_index]:
                raise ValueError("APP trace became invalid during sampling")
            legacy_holed = bool(state.terminal[local_index])
            if legacy_holed != bool(row["legacy_holeout_decision"]):
                raise AssertionError("Explicit APP donor changed legacy holeout decision")
            if state.status[local_index] == streaming.STATUS_APP_HAZARD:
                if local_index in self._pending:
                    raise AssertionError("Particle already has a pending causal atom")
                self._pending[local_index] = row
            else:
                self._finish_row(row, local_index=local_index, state=state)

    def _resolve_hazards(
        self,
        *,
        state: streaming.HoleState,
        active: np.ndarray,
        record: object,
        uniform: np.ndarray,
        diagnostics: streaming.KernelDiagnostics,
    ) -> None:
        loop_positions = np.flatnonzero(np.asarray(active, bool))
        if not len(loop_positions):
            raise AssertionError("Hazard resolver was called without an active row")
        transition_indices = (
            state.transition_count[loop_positions].astype(np.int64) - 1
        )
        unique_transition_indices = np.unique(transition_indices)
        if len(unique_transition_indices) != 1:
            raise AssertionError(
                "Active particles disagree on the counter transition index"
            )
        transition_index = int(unique_transition_indices[0])
        if not 0 <= transition_index < int(self.max_transitions):
            raise AssertionError(
                f"Counter transition index is outside the tail policy: {transition_index}"
            )
        self._trace_transition_index = transition_index
        frame = self._frame(record)
        traced_resolution: dict[int, dict[str, object]] = {}
        for status_code, category in (
            (streaming.STATUS_TEE_HAZARD, "ott"),
            (streaming.STATUS_APP_HAZARD, "app"),
        ):
            positions = np.flatnonzero(np.asarray(active, bool) & (state.status == status_code))
            if not len(positions):
                continue
            absolute = frame.local_yards_to_absolute_feet_many(state.endpoint[positions])
            resolved = self.hazard_resolver.resolve_many(
                endpoint_x=absolute[:, 0],
                endpoint_y=absolute[:, 1],
                round_no=int(record.round_no),
                hole_no=int(record.hole_no),
                category=category,
                common_uniform=np.asarray(uniform, float)[positions],
                forecast_pin_x=float(record.forecast_pin_x),
                forecast_pin_y=float(record.forecast_pin_y),
            )
            for offset, local_index in enumerate(positions):
                if not bool(resolved.available[offset]):
                    raise ValueError("Traced hazard resolution is unavailable")
                traced_resolution[int(local_index)] = {
                    "hazard_donor_key": str(resolved.donor_key[offset]),
                    "penalty_strokes": int(resolved.penalty_stroke_cost[offset]),
                    "rng_hazard_uniform": float(
                        np.asarray(uniform, float)[int(local_index)]
                    ),
                    "rng_hazard_transition_index": transition_index,
                }
        super()._resolve_hazards(
            state=state, active=active, record=record, uniform=uniform,
            diagnostics=diagnostics,
        )
        for local_index, resolved in traced_resolution.items():
            try:
                row = self._pending.pop(local_index)
            except KeyError as exc:
                raise AssertionError("Resolved hazard lacks a pending causal atom") from exc
            penalty = int(resolved["penalty_strokes"])
            row["penalty_strokes"] = penalty
            row["stroke_cost"] = 1 + penalty
            row["hazard_resolution_status"] = "resolved_complete_atom"
            row["hazard_donor_key"] = resolved["hazard_donor_key"]
            row["rng_hazard_uniform"] = float(resolved["rng_hazard_uniform"])
            row["rng_hazard_transition_index"] = int(
                resolved["rng_hazard_transition_index"]
            )
            if state.invalid[local_index]:
                raise ValueError("Hazard trace became invalid during resolution")
            self._finish_row(row, local_index=local_index, state=state)

    def _population_transition(
        self,
        *,
        state: streaming.HoleState,
        active: np.ndarray,
        category: str,
        record: object,
        common_uniform: np.ndarray,
        diagnostics: streaming.KernelDiagnostics,
    ) -> None:
        positions = np.flatnonzero(active)
        if not len(positions):
            return
        started = time.perf_counter()
        traced = self.traced_state_sampler.sample_many(
            round_no=int(record.round_no),
            hole_no=int(record.hole_no),
            category=category,
            lie=streaming.LIE_NAMES[state.lie[positions]],
            distance_yd=state.remaining_yd[positions],
            common_uniform=np.asarray(common_uniform, float)[positions],
        )
        result = traced.transition
        diagnostics.population_seconds += time.perf_counter() - started
        if not result.available.all():
            bad = positions[~result.available]
            state.invalid[bad] = True
            state.status[bad] = streaming.STATUS_POPULATION_UNAVAILABLE
            diagnostics.unsupported_state_rows += len(bad)
            raise ValueError("Population trace encountered an unavailable state")

        trace_rows: dict[int, dict[str, object]] = {}
        for offset, local_index in enumerate(positions):
            donor = pd.Series(
                {
                    "source_event_key": traced.source_event_key[offset],
                    "source_player_id": traced.source_player_id[offset],
                    "source_player_name": traced.source_player_name[offset],
                    "source_round_no": traced.source_round_no[offset],
                    "source_hole_no": traced.source_hole_no[offset],
                    "source_stroke_no": traced.source_stroke_no[offset],
                    "source_shot_key": traced.source_shot_key[offset],
                    "atom_id": traced.source_shot_key[offset],
                    "base_hierarchy_mass": traced.base_hierarchy_mass[offset],
                    "spatial_model_weight": 1.0,
                    "quality_base_mass": traced.base_hierarchy_mass[offset],
                    "atom_kind": f"{category}_population_transition",
                }
            )
            support_key = (
                f"population|{self._trace_arm}|R{int(record.round_no)}|"
                f"H{int(record.hole_no)}|{category}|{traced.query_state_key[offset]}|"
                f"{record.source_player_id}"
            )
            trace_rows[int(local_index)] = self._base_row(
                local_index=int(local_index),
                category=category,
                support_key=support_key,
                donor=donor,
                start_lie=self._physical_lie(int(state.lie[local_index])),
                start_distance_yd=float(state.remaining_yd[local_index]),
                start_endpoint=state.endpoint[local_index].copy(),
                population_fields={
                    "population_query_state_key": str(traced.query_state_key[offset]),
                    "population_pool_key": str(traced.pool_key[offset]),
                    "population_pool_level": str(result.level[offset]),
                    "population_pool_index_level": str(traced.pool_index_level[offset]),
                    "population_source_scope": str(traced.source_scope[offset]),
                    "population_donor_pool_row": int(
                        traced.donor_pool_row[offset]
                    ),
                    "population_donor_transition_row_id": int(
                        traced.donor_transition_row_id[offset]
                    ),
                    "population_normalized_knn_probability": float(
                        traced.normalized_knn_probability[offset]
                    ),
                    "population_cdf_lower": float(traced.cdf_lower[offset]),
                    "population_cdf_upper": float(traced.cdf_upper[offset]),
                    "population_donor_sg": float(traced.donor_sg[offset]),
                    "population_donor_sg_weight_valid": bool(
                        traced.donor_sg_weight_valid[offset]
                    ),
                },
            )
            trace_rows[int(local_index)]["rng_population_uniform"] = float(
                np.asarray(common_uniform, float)[int(local_index)]
            )

        # This block is intentionally kept algebraically identical to the
        # parent state update; ``traced.transition`` has already asserted its
        # selected rows exactly match the existing batched sampler.
        cost = result.cost
        if (cost < 1).any():
            raise AssertionError("Population transition cost must include the shot")
        state.strokes[positions] += cost
        penalty = cost > 1
        if penalty.any():
            penalty_positions = positions[penalty]
            state.hazard_count[penalty_positions] += 1
            state.penalty_strokes[penalty_positions] += cost[penalty] - 1
            diagnostics.population_penalty_draws += int(penalty.sum())
        normalized = streaming._encode_population_lies(result.next_lie)
        holed = result.holed | (normalized == streaming.LIE_HOLE)
        if holed.any():
            target = positions[holed]
            state.terminal[target] = True
            state.lie[target] = streaming.LIE_HOLE
            state.remaining_yd[target] = 0.0
            state.endpoint[target] = np.asarray(
                [float(record.forecast_pin_long_yd), 0.0]
            )
            state.status[target] = streaming.STATUS_TERMINAL
        continuing = ~holed
        target = positions[continuing]
        if len(target):
            next_distance = result.next_distance[continuing]
            next_lie = normalized[continuing]
            invalid = (next_lie == streaming.LIE_UNKNOWN) | ~np.isfinite(next_distance)
            if invalid.any():
                bad = target[invalid]
                state.invalid[bad] = True
                state.status[bad] = streaming.STATUS_POPULATION_INVALID
                diagnostics.unsupported_state_rows += len(bad)
            accepted = ~invalid
            accepted_target = target[accepted]
            if len(accepted_target):
                accepted_distance = np.maximum(next_distance[accepted], 0.0)
                pin = np.asarray([float(record.forecast_pin_long_yd), 0.0])
                vector = state.endpoint[accepted_target] - pin[None, :]
                length = np.linalg.norm(vector, axis=1)
                unit = np.zeros_like(vector)
                nonzero = np.isfinite(length) & (length > 1e-9)
                unit[nonzero] = vector[nonzero] / length[nonzero, None]
                unit[~nonzero, 0] = 1.0
                state.endpoint[accepted_target] = (
                    pin[None, :] + unit * accepted_distance[:, None]
                )
                state.remaining_yd[accepted_target] = accepted_distance
                state.lie[accepted_target] = next_lie[accepted]
                returned_app = streaming._state_categories(
                    state.lie,
                    state.remaining_yd,
                    np.isin(np.arange(len(state.lie)), accepted_target),
                    int(record.par),
                ) == streaming.CATEGORY_APP
                state.return_to_app_count[returned_app] += 1
                state.status[accepted_target] = streaming.STATUS_ACTIVE
        if category == "arg":
            diagnostics.arg_draws += len(positions)
        else:
            diagnostics.putt_draws += len(positions)

        for offset, local_index in enumerate(positions):
            row = trace_rows[int(local_index)]
            penalty_strokes = int(cost[offset]) - 1
            row["penalty_strokes"] = penalty_strokes
            row["stroke_cost"] = int(cost[offset])
            row["hazard_resolution_status"] = (
                "collapsed_exact_transition" if penalty_strokes else "none"
            )
            if state.invalid[local_index]:
                raise ValueError("Population trace became invalid during state update")
            self._finish_row(row, local_index=local_index, state=state)

    def simulate_hole_with_ledger(
        self,
        *,
        record: object,
        arm: str,
        particle_ids: np.ndarray,
        total_particles: int,
        rng_mode: str,
        seed: int = streaming.DEFAULT_SEED,
        continuation_seed: int = streaming.DEFAULT_CONTINUATION_SEED,
    ) -> TracedHoleExecution:
        ids = np.asarray(particle_ids, dtype=np.int64)
        if len(np.unique(ids)) != len(ids):
            raise ValueError("Traced particle IDs must be unique")
        self._trace_rows = []
        self._pending = {}
        self._particle_ids = ids
        self._sequence = np.zeros(len(ids), dtype=np.int16)
        self._trace_arm = str(arm)
        self._trace_transition_index = -1
        result, diagnostics = super().simulate_hole(
            record=record,
            arm=arm,
            particle_ids=ids,
            total_particles=int(total_particles),
            rng_mode=str(rng_mode),
            seed=int(seed),
            continuation_seed=int(continuation_seed),
        )
        if self._pending:
            raise AssertionError(
                f"Traced hole ended with {len(self._pending)} pending causal atoms"
            )
        if not result.terminal.all():
            raise ValueError("Traced hole must complete every particle fail-closed")
        ledger = pd.DataFrame(self._trace_rows).sort_values(
            ["particle_id", "atom_sequence"], kind="mergesort"
        ).reset_index(drop=True)
        if ledger.empty:
            raise ValueError("Traced hole emitted no causal atoms")
        ledger["next_support_key"] = ""
        for _, group in ledger.groupby("particle_id", observed=True, sort=False):
            index = group.index.to_numpy()
            if len(index) > 1:
                ledger.loc[index[:-1], "next_support_key"] = ledger.loc[
                    index[1:], "support_key"
                ].to_numpy()
            if not bool(ledger.loc[index[-1], "terminal"]):
                raise AssertionError("Last traced atom is not terminal")
            if ledger.loc[index[:-1], "terminal"].any():
                raise AssertionError("A traced particle continues after terminal")
            start_keys = ledger.loc[index[1:], "start_state_key"].to_numpy()
            prior_end = ledger.loc[index[:-1], "end_state_key"].to_numpy()
            if not np.array_equal(start_keys, prior_end):
                raise AssertionError("Traced state keys are discontinuous")
            if len(index) > 1:
                if not np.array_equal(
                    ledger.loc[index[:-1], "end_lie"].astype(str).to_numpy(),
                    ledger.loc[index[1:], "start_lie"].astype(str).to_numpy(),
                ):
                    raise AssertionError("Traced physical lies are discontinuous")
                for end_name, start_name in (
                    ("end_distance_yd", "start_distance_yd"),
                    ("end_long_yd", "start_long_yd"),
                    ("end_lat_yd", "start_lat_yd"),
                ):
                    if not np.array_equal(
                        ledger.loc[index[:-1], end_name].to_numpy(float),
                        ledger.loc[index[1:], start_name].to_numpy(float),
                    ):
                        raise AssertionError(
                            f"Traced physical seam is discontinuous: "
                            f"{end_name}->{start_name}"
                        )

        audit_rows: list[dict[str, object]] = []
        score_by_id = pd.Series(
            result.hole_strokes.astype(int), index=result.particle_id.astype(int)
        )
        for particle_id, group in ledger.groupby(
            "particle_id", observed=True, sort=False
        ):
            cost = int(group["stroke_cost"].sum())
            score = int(score_by_id.loc[int(particle_id)])
            reward_sum = float(group["fixed_reference_reward"].sum())
            expected_reward = float(group["start_reference_value"].iloc[0]) - score
            stroke_residual = cost - score
            reward_residual = reward_sum - expected_reward
            audit_rows.append(
                {
                    "particle_id": int(particle_id),
                    "atoms": int(len(group)),
                    "hole_strokes": score,
                    "atom_stroke_cost": cost,
                    "stroke_identity_residual": stroke_residual,
                    "reward_sum": reward_sum,
                    "expected_reward": expected_reward,
                    "reward_telescope_residual": reward_residual,
                    "terminal_atoms": int(group["terminal"].sum()),
                }
            )
        audit = pd.DataFrame(audit_rows)
        if not audit["stroke_identity_residual"].eq(0).all():
            raise AssertionError("Traced causal atom costs do not equal hole scores")
        if audit["reward_telescope_residual"].abs().max() > 5e-12:
            raise AssertionError("Traced fixed-reference rewards do not telescope")
        if not audit["terminal_atoms"].eq(1).all():
            raise AssertionError("Each traced particle needs exactly one terminal atom")
        return TracedHoleExecution(
            result=result,
            diagnostics=diagnostics,
            ledger=ledger,
            telescope_audit=audit,
        )


def attach_strict_prior_donor_abilities(
    ledger: pd.DataFrame,
    bridge: pd.DataFrame,
    *,
    target_event_start: object,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Attach four-category donor ability without converting missing to zero."""

    required = STRICT_BRIDGE_COLUMNS
    missing = sorted(set(required).difference(bridge.columns))
    if missing:
        raise ValueError(f"Strict-prior ability bridge misses columns={missing}")
    out = ledger.copy()
    out["donor_event_key"] = out["donor_event_key"].astype(str)
    out["donor_player_id"] = out["donor_player_id"].astype(str)
    skills = bridge.loc[:, list(required)].copy()
    skills["event_key"] = skills["event_key"].astype(str)
    skills["source_player_id"] = skills["source_player_id"].astype(str)
    skills = skills.rename(
        columns={
            "event_key": "donor_event_key",
            "source_player_id": "donor_player_id",
            "event_start": "donor_event_start",
            "skill_date": "donor_skill_date",
        }
    )
    if skills.duplicated(["donor_event_key", "donor_player_id"]).any():
        raise ValueError("Strict-prior donor ability bridge has duplicate keys")
    out = out.merge(
        skills,
        on=["donor_event_key", "donor_player_id"],
        how="left",
        validate="many_to_one",
    )
    category = out["category"].astype(str).to_numpy()
    ability = np.full(len(out), np.nan, dtype=float)
    for category_name, column in CATEGORY_ABILITY_COLUMN.items():
        selected = category == category_name
        ability[selected] = pd.to_numeric(
            out.loc[selected, column], errors="coerce"
        ).to_numpy(float)
    out["donor_ability_sg"] = ability
    out["donor_skill_date"] = pd.to_datetime(
        out["donor_skill_date"], errors="coerce", utc=True
    )
    out["donor_event_start"] = pd.to_datetime(
        out["donor_event_start"], errors="coerce", utc=True
    )
    target_start = pd.Timestamp(target_event_start)
    if target_start.tzinfo is None:
        target_start = target_start.tz_localize("UTC")
    else:
        target_start = target_start.tz_convert("UTC")
    ability_available = (
        np.isfinite(out["donor_ability_sg"].to_numpy(float))
        & out["donor_skill_date"].notna().to_numpy()
        & out["donor_event_start"].notna().to_numpy()
        & (out["donor_skill_date"] < out["donor_event_start"]).to_numpy()
        & (out["donor_event_start"] < target_start).to_numpy()
    )
    out["donor_ability_available"] = ability_available
    out["donor_ability_snapshot_kind"] = np.where(
        ability_available,
        chain_adapter.DONOR_SNAPSHOT_CONTRACT,
        "strict_prior_snapshot_missing_fail_closed",
    )
    cutoff_violation = (
        out["donor_skill_date"].notna()
        & out["donor_event_start"].notna()
        & (out["donor_skill_date"] >= out["donor_event_start"])
    )
    target_violation = out["donor_event_start"].notna() & (
        out["donor_event_start"] >= target_start
    )
    if cutoff_violation.any() or target_violation.any():
        raise ValueError("Donor ability attachment violates a strict cutoff")
    category_coverage = {
        name: {
            "rows": int((out["category"] == name).sum()),
            "available": int(
                ((out["category"] == name) & out["donor_ability_available"]).sum()
            ),
        }
        for name in CATEGORY_ABILITY_COLUMN
    }
    return out.drop(columns=list(CATEGORY_ABILITY_COLUMN.values())), {
        "rows": int(len(out)),
        "available_rows": int(ability_available.sum()),
        "coverage": float(ability_available.mean()) if len(out) else 0.0,
        "strict_cutoff_violations": int(cutoff_violation.sum()),
        "target_cutoff_violations": int(target_violation.sum()),
        "category": category_coverage,
        "missing_policy": "fail_closed_not_neutral_zero",
    }


def seal_proposal_support_ability_hashes(
    atoms: pd.DataFrame,
    inventory: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Bind every proposal donor's observed/missing ability semantics."""

    required = set(PROPOSAL_ABILITY_DIGEST_COLUMNS)
    missing = sorted(required.difference(atoms.columns))
    if missing:
        raise ValueError(f"Proposal ability seal misses columns={missing}")
    output = atoms.copy()
    output["proposal_support_ability_sha256"] = ""
    rows: list[dict[str, object]] = []
    max_beta_zero_residual = 0.0
    for (arm, support_key), group in output.groupby(
        ["arm", "support_key"], observed=True, sort=False
    ):
        ordered = group.sort_values("proposal_row_order", kind="mergesort")
        expected_order = np.arange(len(ordered), dtype=np.int32)
        if not np.array_equal(
            ordered["proposal_row_order"].to_numpy(np.int32), expected_order
        ):
            raise AssertionError(f"Proposal row order is not contiguous: {support_key}")
        ability_sha = _frame_digest(ordered, PROPOSAL_ABILITY_DIGEST_COLUMNS)
        output.loc[ordered.index, "proposal_support_ability_sha256"] = ability_sha
        base = ordered["quality_base_mass"].to_numpy(float)
        beta_zero = base / base.sum()
        proposal = ordered["proposal_probability"].to_numpy(float)
        residual = float(np.max(np.abs(beta_zero - proposal), initial=0.0))
        max_beta_zero_residual = max(max_beta_zero_residual, residual)
        if residual > 2e-15:
            raise AssertionError(f"Beta-zero proposal identity failed: {support_key}")
        rows.append(
            {
                "arm": str(arm),
                "support_key": str(support_key),
                "proposal_support_ability_sha256": ability_sha,
                "ability_rows": int(len(ordered)),
                "ability_observed_rows": int(
                    ordered["donor_ability_available"].sum()
                ),
                "ability_missing_rows": int(
                    (~ordered["donor_ability_available"]).sum()
                ),
                "beta_zero_probability_max_abs_residual": residual,
            }
        )
    ability_inventory = pd.DataFrame(rows)
    sealed_inventory = inventory.merge(
        ability_inventory,
        on=["arm", "support_key"],
        how="left",
        validate="one_to_one",
    )
    if sealed_inventory["proposal_support_ability_sha256"].isna().any():
        raise AssertionError("Proposal inventory misses an ability-bound hash")
    return output, sealed_inventory, {
        "supports": int(len(ability_inventory)),
        "rows": int(len(output)),
        "observed_rows": int(output["donor_ability_available"].sum()),
        "missing_rows": int((~output["donor_ability_available"]).sum()),
        "all_rows_have_declared_observation_status": bool(
            output["donor_ability_snapshot_kind"].astype(str).ne("").all()
        ),
        "beta_zero_support_identity_verified": True,
        "beta_zero_probability_max_abs_residual": max_beta_zero_residual,
        "policy": "observed_strict_prior_only_missing_remains_fail_closed",
    }


def load_validated_production_targets(
    target_input_dir: Path,
    *,
    expected_event_key: str,
    expected_event_start: object,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Validate both adapter means and the richer target/ability provenance."""

    directory = Path(target_input_dir)
    target_path = directory / "production_anchor_inputs.parquet"
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    targets = pd.read_parquet(target_path)
    if manifest.get("contract") != production_targets.CONTRACT:
        raise ValueError("Frozen production target contract is incompatible")
    declared_target_sha = manifest.get("artifact_sha256", {}).get(target_path.name)
    if declared_target_sha != streaming.sha256_file(target_path):
        raise ValueError("Frozen production target Parquet hash changed")
    if int(manifest.get("round_rows", -1)) != len(targets):
        raise ValueError("Frozen production target row count changed")
    if targets["production_anchor_input_sha256"].astype(str).ne(
        [production_targets._row_digest(row) for row in targets.to_dict("records")]
    ).any():
        raise ValueError("Frozen rich production target row hash changed")

    adapter_columns = list(chain_adapter.PRODUCTION_TARGET_REQUIRED_COLUMNS)
    validated = chain_adapter.validate_production_target_frame(
        targets[adapter_columns]
    )
    bundle_columns = [
        "event_key",
        "player_id",
        "round_no",
        "production_anchor_input_sha256",
        "production_input_sha256",
    ]
    sources = dict(manifest.get("sources", {}))
    bundle_records = targets[bundle_columns].astype(str).to_dict("records")
    observed_bundle_sha = production_targets._stable_sha(
        {
            "contract": production_targets.CONTRACT,
            "records": bundle_records,
            "sources": sources,
        }
    )
    if observed_bundle_sha != manifest.get("bundle_sha256"):
        raise ValueError("Frozen production target bundle hash changed")
    for source_name, source_sha in sources.items():
        if source_name not in targets.columns:
            raise ValueError(f"Frozen production rows omit source={source_name}")
        if not targets[source_name].astype(str).eq(str(source_sha)).all():
            raise ValueError(f"Frozen production source hash drift={source_name}")

    expected_start = pd.Timestamp(expected_event_start)
    event_start = pd.to_datetime(targets["event_start"], errors="coerce")
    if event_start.isna().any() or not event_start.eq(expected_start.tz_localize(None)).all():
        raise ValueError("Frozen production target event start changed")
    if not targets["event_key"].astype(str).eq(str(expected_event_key)).all():
        raise ValueError("Frozen production target event identity changed")
    skill_date = pd.to_datetime(targets["ability_skill_date"], errors="coerce")
    if skill_date.isna().any() or not skill_date.lt(event_start).all():
        raise ValueError("Frozen production ability is not strictly prior")
    expected_authority = np.where(
        targets["round_no"].eq(1),
        "deterministic_frozen_backtest_r1_category_mean",
        "static_pre_event_backtest_mean_not_live_cascade",
    )
    if not np.array_equal(
        targets["round_target_authority"].astype(str).to_numpy(),
        expected_authority,
    ):
        raise ValueError("Frozen production target authority changed")
    if not targets["production_equivalence_status"].astype(str).eq(
        "not_exact_historical_production_reconstruction"
    ).all():
        raise ValueError("Frozen production equivalence status changed")
    return targets, validated, {
        "contract": str(manifest["contract"]),
        "manifest_sha256": streaming.sha256_file(manifest_path),
        "parquet_sha256": streaming.sha256_file(target_path),
        "bundle_sha256": observed_bundle_sha,
        "rows": int(len(targets)),
        "ability_cutoff_violations": int((skill_date >= event_start).sum()),
        "rich_row_hashes_verified": True,
        "adapter_row_hashes_verified": True,
    }


def validate_persisted_proposal_support_contract(
    atoms: pd.DataFrame,
    inventory: pd.DataFrame,
) -> dict[str, object]:
    """Recompute every proposal measure/ability hash from persisted rows."""

    inventory_lookup = inventory.set_index(["arm", "support_key"], drop=False)
    support_count = 0
    atom_count = 0
    max_mass_residual = 0.0
    max_cdf_width_residual = 0.0
    max_branch_cdf_width_residual = 0.0
    for key, raw_group in atoms.groupby(
        ["arm", "support_key"], observed=True, sort=False
    ):
        arm, support_key = map(str, key)
        if (arm, support_key) not in inventory_lookup.index:
            raise AssertionError(f"Persisted proposal lacks inventory row: {key}")
        group = raw_group.sort_values("proposal_row_order", kind="mergesort")
        order = group["proposal_row_order"].to_numpy(np.int32)
        if not np.array_equal(order, np.arange(len(group), dtype=np.int32)):
            raise AssertionError(f"Persisted proposal row order changed: {key}")
        if group["atom_id"].astype(str).duplicated().any():
            raise AssertionError(f"Persisted proposal atom IDs duplicate: {key}")
        declared_measure = group[
            "proposal_support_measure_sha256"
        ].astype(str).unique().tolist()
        declared_ability = group[
            "proposal_support_ability_sha256"
        ].astype(str).unique().tolist()
        if len(declared_measure) != 1 or len(declared_ability) != 1:
            raise AssertionError(f"Persisted proposal hash varies within support: {key}")
        observed_measure = _frame_digest(group, PROPOSAL_SUPPORT_DIGEST_COLUMNS)
        observed_ability = _frame_digest(group, PROPOSAL_ABILITY_DIGEST_COLUMNS)
        if observed_measure != declared_measure[0]:
            raise AssertionError(f"Persisted proposal mechanics hash changed: {key}")
        if observed_ability != declared_ability[0]:
            raise AssertionError(f"Persisted proposal ability hash changed: {key}")
        inventory_row = inventory_lookup.loc[(arm, support_key)]
        if str(inventory_row["proposal_support_measure_sha256"]) != observed_measure:
            raise AssertionError(f"Proposal inventory mechanics hash changed: {key}")
        if str(inventory_row["proposal_support_ability_sha256"]) != observed_ability:
            raise AssertionError(f"Proposal inventory ability hash changed: {key}")
        if int(inventory_row["atoms"]) != len(group):
            raise AssertionError(f"Proposal inventory atom count changed: {key}")

        probability = group["proposal_probability"].to_numpy(float)
        lower = group["proposal_cdf_lower"].to_numpy(float)
        upper = group["proposal_cdf_upper"].to_numpy(float)
        mass_residual = abs(float(group["quality_base_mass"].sum()) - 1.0)
        cdf_residual = float(
            np.max(np.abs((upper - lower) - probability), initial=0.0)
        )
        max_mass_residual = max(max_mass_residual, mass_residual)
        max_cdf_width_residual = max(max_cdf_width_residual, cdf_residual)
        expected_upper = np.cumsum(probability)
        expected_upper[-1] = 1.0
        expected_lower = np.concatenate(([0.0], expected_upper[:-1]))
        if lower[0] != 0.0 or upper[-1] != 1.0:
            raise AssertionError(f"Persisted proposal CDF does not close: {key}")
        if not (
            np.array_equal(lower, expected_lower)
            and np.array_equal(upper, expected_upper)
        ):
            raise AssertionError(f"Persisted proposal CDF changed: {key}")
        if mass_residual > APP_SUPPORT_MASS_TOLERANCE:
            raise AssertionError(f"Persisted proposal probability changed: {key}")
        for branch, branch_frame in group.groupby(
            "proposal_branch", observed=True, sort=False
        ):
            branch_lower = branch_frame["branch_cdf_lower"].to_numpy(float)
            branch_upper = branch_frame["branch_cdf_upper"].to_numpy(float)
            branch_probability = branch_frame[
                "branch_conditional_probability"
            ].to_numpy(float)
            residual = float(
                np.max(
                    np.abs((branch_upper - branch_lower) - branch_probability),
                    initial=0.0,
                )
            )
            max_branch_cdf_width_residual = max(
                max_branch_cdf_width_residual, residual
            )
            expected_branch_upper = np.cumsum(branch_probability)
            expected_branch_upper[-1] = 1.0
            expected_branch_lower = np.concatenate(
                ([0.0], expected_branch_upper[:-1])
            )
            if branch_lower[0] != 0.0 or branch_upper[-1] != 1.0:
                raise AssertionError(
                    f"Persisted proposal branch CDF does not close: {key}/{branch}"
                )
            if not (
                np.array_equal(branch_lower, expected_branch_lower)
                and np.array_equal(branch_upper, expected_branch_upper)
            ):
                raise AssertionError(
                    f"Persisted proposal branch CDF changed: {key}/{branch}"
                )
        if group["category"].astype(str).eq("app").all():
            terminal_mass = float(
                group.loc[group["terminal"], "quality_base_mass"].sum()
            )
            declared = float(
                group["declared_app_base_holeout_probability"].iloc[0]
            )
            if not np.isclose(
                terminal_mass,
                declared,
                atol=APP_SUPPORT_MASS_TOLERANCE,
                rtol=0.0,
            ):
                raise AssertionError(f"Persisted APP terminal mass changed: {key}")
            if (declared > 0.0) != bool(group["terminal"].any()):
                raise AssertionError(f"Persisted APP terminal inventory changed: {key}")
        support_count += 1
        atom_count += len(group)
    if support_count != len(inventory):
        raise AssertionError("Persisted proposal inventory has extra support rows")
    return {
        "supports": int(support_count),
        "atoms": int(atom_count),
        "mechanics_hashes_recomputed": True,
        "ability_hashes_recomputed": True,
        "max_abs_mass_residual": max_mass_residual,
        "max_abs_cdf_width_residual": max_cdf_width_residual,
        "max_abs_branch_cdf_width_residual": max_branch_cdf_width_residual,
    }


def validate_counter_rng_trace(
    ledger: pd.DataFrame,
    *,
    seed: int,
    continuation_seed: int,
) -> dict[str, object]:
    """Regenerate every persisted semantic counter uniform exactly."""

    required = {
        "target_event_key",
        "target_player_id",
        "round_no",
        "hole_no",
        "particle_id",
        "category",
        "hazard_resolution_status",
        "rng_shot_transition_index",
        "rng_hazard_transition_index",
        "rng_index_uniform",
        "rng_lie_uniform",
        "rng_holeout_uniform",
        "rng_holeout_residual_uniform",
        "rng_population_uniform",
        "rng_hazard_uniform",
        "legacy_holeout_decision",
        "declared_app_base_holeout_probability",
    }
    missing = sorted(required.difference(ledger.columns))
    if missing:
        raise ValueError(f"Counter trace misses columns={missing}")
    category = ledger["category"].astype(str)
    ott = category.eq("ott").to_numpy()
    app_rows = category.eq("app").to_numpy()
    arg = category.eq("arg").to_numpy()
    putt = category.eq("putt").to_numpy()
    if not (ott | app_rows | arg | putt).all():
        raise ValueError("Counter trace contains an unsupported category")

    def finite(name: str) -> np.ndarray:
        return np.isfinite(pd.to_numeric(ledger[name], errors="coerce").to_numpy(float))

    index_present = finite("rng_index_uniform")
    lie_present = finite("rng_lie_uniform")
    holeout_present = finite("rng_holeout_uniform")
    population_present = finite("rng_population_uniform")
    residual_present = finite("rng_holeout_residual_uniform")
    expected_spatial = ott | app_rows
    for name, observed, expected in (
        ("index", index_present, expected_spatial),
        ("lie", lie_present, expected_spatial),
        ("holeout", holeout_present, app_rows),
        ("population", population_present, arg | putt),
    ):
        if not np.array_equal(observed, expected):
            raise AssertionError(f"Counter trace {name} uniform null semantics changed")
    selected_holeout = ledger["legacy_holeout_decision"].to_numpy(bool)
    if (selected_holeout & ~app_rows).any():
        raise AssertionError("A non-APP atom declares an APP holeout decision")
    if not np.array_equal(residual_present, selected_holeout):
        raise AssertionError("APP residual uniform null semantics changed")
    resolved_hazard = ledger["hazard_resolution_status"].astype(str).eq(
        "resolved_complete_atom"
    ).to_numpy()
    hazard_uniform_present = finite("rng_hazard_uniform")
    hazard_index_present = finite("rng_hazard_transition_index")
    if not (
        np.array_equal(hazard_uniform_present, resolved_hazard)
        and np.array_equal(hazard_index_present, resolved_hazard)
    ):
        raise AssertionError("Hazard uniform/index null semantics changed")
    shot_index = pd.to_numeric(
        ledger["rng_shot_transition_index"], errors="coerce"
    ).to_numpy(float)
    if not np.isfinite(shot_index).all() or not np.equal(
        shot_index, np.floor(shot_index)
    ).all():
        raise AssertionError("Shot transition index is missing or nonintegral")
    if ((shot_index == -1) & ~(ott | app_rows)).any():
        raise AssertionError("Population atoms cannot use the initializer index")
    hazard_index = pd.to_numeric(
        ledger["rng_hazard_transition_index"], errors="coerce"
    ).to_numpy(float)
    if resolved_hazard.any():
        resolved_hazard_index = hazard_index[resolved_hazard]
        if not np.isfinite(resolved_hazard_index).all() or not np.equal(
            resolved_hazard_index, np.floor(resolved_hazard_index)
        ).all():
            raise AssertionError("Hazard transition index is missing or nonintegral")
        if (
            (resolved_hazard_index < 0).any()
            or (
                resolved_hazard_index
                >= int(streaming.tail_policy.MAX_CONTINUATION_TRANSITIONS)
            ).any()
        ):
            raise AssertionError("Hazard transition index is outside the tail policy")
        if not np.array_equal(
            resolved_hazard_index,
            shot_index[resolved_hazard] + 1.0,
        ):
            raise AssertionError(
                "Hazard transition index is not one step after its causal shot"
            )
    checks = 0
    stream_counts: dict[str, int] = {}
    max_abs_residual = 0.0

    def check_rows(
        rows: pd.DataFrame,
        *,
        stream: str,
        observed_column: str,
        transition_column: str,
    ) -> None:
        nonlocal checks, max_abs_residual
        selected = rows.loc[np.isfinite(rows[observed_column].to_numpy(float))]
        for context, group in selected.groupby(
            [
                "target_event_key",
                "target_player_id",
                "round_no",
                "hole_no",
                transition_column,
            ],
            observed=True,
            sort=False,
        ):
            event_key, player_id, round_no, hole_no, transition_value = context
            transition_index = int(transition_value)
            particle_ids = group["particle_id"].to_numpy(np.int64)
            if transition_index == -1:
                expected = streaming.hook.counter_uniforms(
                    seed=int(seed),
                    event_key=str(event_key),
                    player_id=str(player_id),
                    round_no=int(round_no),
                    hole_no=int(hole_no),
                    transition_index=-1,
                    stream=f"initial_{stream}",
                    particle_ids=particle_ids,
                )
            elif 0 <= transition_index < int(streaming.tail_policy.MAX_CONTINUATION_TRANSITIONS):
                expected = streaming.hook.counter_uniforms(
                    seed=int(continuation_seed),
                    event_key=str(event_key),
                    player_id=str(player_id),
                    round_no=int(round_no),
                    hole_no=int(hole_no),
                    transition_index=transition_index,
                    stream=str(stream),
                    particle_ids=particle_ids,
                )
            else:
                raise AssertionError(
                    f"Persisted counter index is invalid: {transition_index}"
                )
            observed = group[observed_column].to_numpy(float)
            residual = float(np.max(np.abs(observed - expected), initial=0.0))
            max_abs_residual = max(max_abs_residual, residual)
            if not np.array_equal(observed, expected):
                raise AssertionError(
                    f"Counter RNG replay changed for {stream}/{transition_index}"
                )
            checks += len(group)
            stream_counts[stream] = stream_counts.get(stream, 0) + len(group)

    ott_series = pd.Series(ott, index=ledger.index)
    app_series = pd.Series(app_rows, index=ledger.index)
    for mask, prefix in ((ott_series, "tee"), (app_series, "app")):
        check_rows(
            ledger.loc[mask],
            stream=f"{prefix}_index",
            observed_column="rng_index_uniform",
            transition_column="rng_shot_transition_index",
        )
        check_rows(
            ledger.loc[mask],
            stream=f"{prefix}_lie",
            observed_column="rng_lie_uniform",
            transition_column="rng_shot_transition_index",
        )
    check_rows(
        ledger.loc[app_series],
        stream="app_holeout",
        observed_column="rng_holeout_uniform",
        transition_column="rng_shot_transition_index",
    )
    for category in ("arg", "putt"):
        check_rows(
            ledger.loc[ledger["category"].astype(str).eq(category)],
            stream=f"state_{category}",
            observed_column="rng_population_uniform",
            transition_column="rng_shot_transition_index",
        )
    check_rows(
        ledger,
        stream="hazard",
        observed_column="rng_hazard_uniform",
        transition_column="rng_hazard_transition_index",
    )

    selected_app = ledger.loc[app_series]
    expected_residual = np.full(len(selected_app), np.nan, dtype=float)
    selected_holeout = selected_app["legacy_holeout_decision"].to_numpy(bool)
    declared = selected_app[
        "declared_app_base_holeout_probability"
    ].to_numpy(float)
    holeout_uniform = selected_app["rng_holeout_uniform"].to_numpy(float)
    if selected_holeout.any():
        if (declared[selected_holeout] <= 0.0).any():
            raise AssertionError("Selected APP holeout has nonpositive declared mass")
        expected_residual[selected_holeout] = (
            holeout_uniform[selected_holeout] / declared[selected_holeout]
        )
    observed_residual = selected_app[
        "rng_holeout_residual_uniform"
    ].to_numpy(float)
    if not np.array_equal(
        observed_residual,
        expected_residual,
        equal_nan=True,
    ):
        raise AssertionError("APP holeout residual uniform changed")
    expected_checks = int(
        2 * ott.sum()
        + 3 * app_rows.sum()
        + arg.sum()
        + putt.sum()
        + resolved_hazard.sum()
    )
    if checks != expected_checks:
        raise AssertionError(
            f"Counter RNG replay count changed: {checks} != {expected_checks}"
        )
    return {
        "contract": str(streaming.hook.RNG_VERSION),
        "seed": int(seed),
        "continuation_seed": int(continuation_seed),
        "uniforms_replayed": int(checks),
        "expected_uniforms": expected_checks,
        "stream_counts": stream_counts,
        "max_abs_uniform_residual": max_abs_residual,
        "initial_transition_index": -1,
        "continuation_transition_indices": list(
            range(int(streaming.tail_policy.MAX_CONTINUATION_TRANSITIONS))
        ),
        "all_exact": True,
    }


def _output_index(directory: Path) -> dict[str, object]:
    files: dict[str, dict[str, object]] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name == "OUTPUT_SHA256.json":
            continue
        relative = path.relative_to(directory).as_posix()
        files[relative] = {
            "sha256": streaming.sha256_file(path),
            "bytes": int(path.stat().st_size),
        }
    payload = {"contract": "non_circular_output_sha256_v1", "files": files}
    (directory / "OUTPUT_SHA256.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def build_bounded_real_slice(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    export_root: Path = precompute.DEFAULT_EXPORT,
    history_db: Path = asof.DEFAULT_HISTORY_DB,
    profile_dir: Path = precompute.DEFAULT_PROFILES,
    baseline_cases: Path = precompute.DEFAULT_BASELINES,
    streaming_reference_dir: Path = streaming.DEFAULT_REFERENCE,
    reference_value_dir: Path = DEFAULT_REFERENCE_DIR,
    target_input_dir: Path = DEFAULT_TARGET_DIR,
    bridge_path: Path = asof.DEFAULT_BRIDGE,
    round_no: int = 1,
    hole_no: int = 6,
    particles: int = 128,
) -> Path:
    """Build the one-player/one-hole physical ledger vertical slice."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output path already exists: {output_dir}")
    if int(particles) != 128:
        raise ValueError("The first bounded ledger slice is frozen at 128 particles")
    source_path = Path(__file__)
    source_start = streaming.sha256_file(source_path)
    dependency_paths = {
        "streaming_source": Path(streaming.__file__),
        "traced_population_source": Path(traced_population.__file__),
        "precompute_source": Path(precompute.__file__),
        "asof_source": Path(asof.__file__),
        "app_source": Path(app.__file__),
        "fixed_reference_source": Path(fixed_reference.__file__),
        "chain_adapter_source": Path(chain_adapter.__file__),
        "production_target_source": Path(production_targets.__file__),
    }
    dependency_start = {
        name: streaming.sha256_file(path)
        for name, path in dependency_paths.items()
    }
    prepared = streaming.prepare_streaming_event(
        export_root=Path(export_root),
        history_db=Path(history_db),
        profile_dir=Path(profile_dir),
        baseline_cases=Path(baseline_cases),
        reference_dir=Path(streaming_reference_dir),
        event_name=precompute.DEFAULT_EVENT,
        holdout_year=precompute.DEFAULT_HOLDOUT,
        course_start_year=2020,
        roster_seed=streaming.DEFAULT_SEED,
    )
    selected = prepared.grid.loc[
        prepared.grid["round_no"].eq(int(round_no))
        & prepared.grid["hole_no"].eq(int(hole_no))
    ]
    if len(selected) != 1:
        raise ValueError("Bounded slice grid row is absent or duplicated")
    record = next(selected.itertuples(index=False))

    prior = pd.DataFrame(
        [
            {"year": int(year), "event_key": str(event_key)}
            for year, event_key in sorted(prepared.atoms.prepared["event_keys"].items())
            if int(year) < int(prepared.atoms.holdout_year)
        ]
    )
    population_bundle = traced_population.build_strict_prior_course_trace_bundle(
        Path(export_root),
        prior,
        prepared.atoms.target_event_key,
        categories=("arg", "putt"),
        min_exact=12,
        knn=64,
    )
    traced_sampler = population_bundle.traced_sampler(chunk_size=128)
    reference_model = fixed_reference.ReferenceValueModel.load(
        Path(reference_value_dir)
    )
    repositories: dict[str, IdentityPreservingSupportRepository] = {}
    for arm in ("raw", "normalized"):
        repositories[arm] = IdentityPreservingSupportRepository(
            arm=arm,
            provider=precompute.AtomSupportProvider(
                prepared.atoms, arm=arm, cache_enabled=True
            ),
        )
    chain = TracedStreamingShotChain(
        repositories=repositories,
        course_drives=prepared.atoms.raw_course_drives,
        course_app=prepared.atoms.raw_course_app,
        traced_state_sampler=traced_sampler,
        hazard_resolver=prepared.batched_hazard,
        reference_model=reference_model,
    )
    executions: list[TracedHoleExecution] = []
    for arm in ("raw", "normalized"):
        executions.append(
            chain.simulate_hole_with_ledger(
                record=record,
                arm=arm,
                particle_ids=np.arange(int(particles), dtype=np.int64),
                total_particles=int(particles),
                rng_mode="counter",
            )
        )
    support_count_before_plain = {
        arm: len(repository.supports)
        for arm, repository in repositories.items()
    }
    plain_chain = streaming.StreamingShotChain(
        repositories=repositories,
        course_drives=prepared.atoms.raw_course_drives,
        course_app=prepared.atoms.raw_course_app,
        state_sampler=traced_sampler.base_sampler,
        hazard_resolver=prepared.batched_hazard,
    )
    plain_results: list[streaming.StreamingHoleResult] = []
    plain_parent_parity: list[dict[str, object]] = []
    for arm, execution in zip(("raw", "normalized"), executions):
        plain_result, _ = plain_chain.simulate_hole(
            record=record,
            arm=arm,
            particle_ids=np.arange(int(particles), dtype=np.int64),
            total_particles=int(particles),
            rng_mode="counter",
        )
        plain_results.append(plain_result)
        plain_parent_parity.append(
            _assert_hole_results_exact(execution.result, plain_result, arm=arm)
        )
    support_count_after_plain = {
        arm: len(repository.supports)
        for arm, repository in repositories.items()
    }
    if support_count_after_plain != support_count_before_plain:
        raise AssertionError(
            "Plain-parent replay discovered supports absent from the traced execution"
        )

    ledger = pd.concat([execution.ledger for execution in executions], ignore_index=True)
    audit = pd.concat(
        [
            execution.telescope_audit.assign(arm=arm)
            for arm, execution in zip(("raw", "normalized"), executions)
        ],
        ignore_index=True,
    )
    ledger.insert(0, "target_event_key", str(prepared.atoms.target_event_key))
    ledger.insert(1, "target_event_start", prepared.atoms.target_event_start)
    ledger.insert(2, "target_player_id", str(record.source_player_id))
    ledger.insert(3, "round_no", int(record.round_no))
    ledger.insert(4, "hole_no", int(record.hole_no))

    bridge, bridge_meta = asof.load_strict_bridge(Path(bridge_path))
    proposal_atoms, proposal_inventory = materialize_proposal_support_contract(
        repositories,
        target_event_key=str(prepared.atoms.target_event_key),
        target_event_start=prepared.atoms.target_event_start,
        target_player_id=str(record.source_player_id),
    )
    proposal_atoms, proposal_ability_audit = attach_strict_prior_donor_abilities(
        proposal_atoms,
        bridge,
        target_event_start=prepared.atoms.target_event_start,
    )
    (
        proposal_atoms,
        proposal_inventory,
        proposal_ability_seal,
    ) = seal_proposal_support_ability_hashes(
        proposal_atoms,
        proposal_inventory,
    )
    ledger, ability_audit = attach_strict_prior_donor_abilities(
        ledger,
        bridge,
        target_event_start=prepared.atoms.target_event_start,
    )

    targets_path = Path(target_input_dir) / "production_anchor_inputs.parquet"
    targets_manifest_path = Path(target_input_dir) / "manifest.json"
    targets, validated_targets, target_validation = load_validated_production_targets(
        Path(target_input_dir),
        expected_event_key=str(prepared.atoms.target_event_key),
        expected_event_start=prepared.atoms.target_event_start,
    )
    target = validated_targets.loc[
        validated_targets["player_id"].astype(str).eq(str(record.source_player_id))
        & validated_targets["round_no"].eq(int(record.round_no))
    ]
    if len(target) != 1:
        raise ValueError("Frozen production target row is absent or duplicated")
    target_row = target.iloc[0]
    rich_target = targets.loc[
        targets["player_id"].astype(str).eq(str(record.source_player_id))
        & targets["round_no"].eq(int(record.round_no))
    ]
    if len(rich_target) != 1:
        raise ValueError("Rich frozen target row is absent or duplicated")
    rich_target_row = rich_target.iloc[0]

    dependency_end = {
        name: streaming.sha256_file(path)
        for name, path in dependency_paths.items()
    }
    if source_start != streaming.sha256_file(source_path):
        raise RuntimeError("Ledger source changed during bounded execution")
    if dependency_start != dependency_end:
        raise RuntimeError("A ledger dependency changed during bounded execution")
    output_dir.mkdir(parents=True)
    ledger_path = output_dir / "complete_transition_ledger.parquet"
    audit_path = output_dir / "particle_telescope_audit.parquet"
    proposal_path = output_dir / "proposal_support_atoms.parquet"
    inventory_path = output_dir / "proposal_support_inventory.parquet"
    results_path = output_dir / "plain_parent_result_parity.parquet"
    result_frame = pd.concat(
        [
            _hole_result_frame(execution.result, arm=arm, implementation="traced")
            for arm, execution in zip(("raw", "normalized"), executions)
        ]
        + [
            _hole_result_frame(result, arm=arm, implementation="plain_parent")
            for arm, result in zip(("raw", "normalized"), plain_results)
        ],
        ignore_index=True,
    )
    ledger, ledger_logical_sha = _write_verified_parquet(ledger, ledger_path)
    audit, audit_logical_sha = _write_verified_parquet(audit, audit_path)
    proposal_atoms, proposal_logical_sha = _write_verified_parquet(
        proposal_atoms, proposal_path
    )
    proposal_inventory, inventory_logical_sha = _write_verified_parquet(
        proposal_inventory, inventory_path
    )
    result_frame, result_logical_sha = _write_verified_parquet(
        result_frame, results_path
    )
    counter_rng_audit = validate_counter_rng_trace(
        ledger,
        seed=streaming.DEFAULT_SEED,
        continuation_seed=streaming.DEFAULT_CONTINUATION_SEED,
    )
    persisted_proposal_audit = validate_persisted_proposal_support_contract(
        proposal_atoms,
        proposal_inventory,
    )
    result_summary = {
        arm: {
            "particles": int(len(execution.result.particle_id)),
            "terminal": int(execution.result.terminal.sum()),
            "min_hole_strokes": int(execution.result.hole_strokes.min()),
            "max_hole_strokes": int(execution.result.hole_strokes.max()),
            "hazard_transitions": int(execution.result.hazard_transitions.sum()),
            "penalty_strokes": int(execution.result.penalty_strokes.sum()),
            "ledger_atoms": int(len(execution.ledger)),
            "explicit_app_holeouts_selected": int(
                execution.ledger["legacy_holeout_decision"].sum()
            ),
        }
        for arm, execution in zip(("raw", "normalized"), executions)
    }
    manifest = {
        "contract": LEDGER_SLICE_VERSION,
        "research_only": True,
        "production_untouched": True,
        "scope": {
            "event_key": str(prepared.atoms.target_event_key),
            "event_start": prepared.atoms.target_event_start.isoformat(),
            "player_id": str(record.source_player_id),
            "round_no": int(record.round_no),
            "hole_no": int(record.hole_no),
            "particles": int(particles),
            "particle_id_contract": "contiguous_zero_based_0_through_127",
            "arms": ["raw", "normalized"],
            "rng_mode": "counter",
            "rng_seed": int(streaming.DEFAULT_SEED),
            "rng_continuation_seed": int(streaming.DEFAULT_CONTINUATION_SEED),
            "rng_stream_contract": str(streaming.hook.RNG_VERSION),
        },
        "reference": {
            "descriptor": reference_model.descriptor,
            "function_sha256": reference_model.model_sha256,
            "putting_curve_sha256": reference_model.putting_curve_sha256,
        },
        "ability_audit": ability_audit,
        "proposal_support_ability_audit": proposal_ability_audit,
        "proposal_support_ability_seal": proposal_ability_seal,
        "bridge": bridge_meta,
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
                for category in CATEGORY_ABILITY_COLUMN
            },
            "authority": str(rich_target_row["round_target_authority"]),
            "production_equivalence_status": str(
                rich_target_row["production_equivalence_status"]
            ),
            "validation": target_validation,
        },
        "results": result_summary,
        "plain_parent_parity": plain_parent_parity,
        "counter_rng_audit": counter_rng_audit,
        "proposal_support": {
            "scope": "materialized_tee_app_proposal_atoms_before_surface_hazard_resolution",
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
            "all_particles_terminal": bool(
                all(execution.result.terminal.all() for execution in executions)
            ),
            "zero_stroke_identity_residual": bool(
                audit["stroke_identity_residual"].eq(0).all()
            ),
            "max_abs_reward_telescope_residual": float(
                audit["reward_telescope_residual"].abs().max()
            ),
            "implicit_app_holeout_probability": float(
                ledger.loc[ledger["category"].eq("app"), "implicit_holeout_probability"].max()
            ),
            "explicit_app_supports": int(
                sum(len(repo.explicit_app_supports) for repo in repositories.values())
            ),
            "proposal_support_mass_exact": bool(
                np.isclose(
                    proposal_inventory["quality_mass"].to_numpy(float),
                    1.0,
                    atol=APP_SUPPORT_MASS_TOLERANCE,
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
                    atol=APP_SUPPORT_MASS_TOLERANCE,
                    rtol=0.0,
                ).all()
            ),
            "plain_parent_result_exact": bool(
                all(bool(row["exact"]) for row in plain_parent_parity)
            ),
            "plain_parent_support_discovery_zero": bool(
                support_count_before_plain == support_count_after_plain
            ),
            "physical_state_seams_exact": True,
            "counter_rng_uniforms_exact": bool(counter_rng_audit["all_exact"]),
            "anchor_ready_all_donor_abilities": bool(
                ledger["donor_ability_available"].all()
            ),
        },
        "inputs": {
            "ledger_source_sha256_start": source_start,
            "ledger_source_sha256_end": streaming.sha256_file(source_path),
            "dependency_sha256_start": dependency_start,
            "dependency_sha256_end": dependency_end,
            "reference_manifest_sha256": streaming.sha256_file(
                Path(reference_value_dir) / "manifest.json"
            ),
            "target_manifest_sha256": streaming.sha256_file(targets_manifest_path),
            "target_parquet_sha256": streaming.sha256_file(targets_path),
            "bridge_sha256": streaming.sha256_file(Path(bridge_path)),
            "bridge_manifest_sha256": streaming.sha256_file(
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
        "# Bounded complete transition ledger\n\n"
        "Research-only 128-particle Travelers slice. Physical donor identity is "
        "retained before numeric cache conversion; APP hole-outs are explicit "
        "terminal donor atoms while preserving the legacy Bernoulli flags.\n\n"
        f"Rows: {len(ledger):,}; supports with explicit APP holeouts: "
        f"{manifest['gates']['explicit_app_supports']}.\n\n"
        f"Persisted proposal measure: {len(proposal_inventory):,} supports / "
        f"{len(proposal_atoms):,} physical atoms; plain-parent replay exact: "
        f"{manifest['gates']['plain_parent_result_exact']}.\n\n"
        f"Maximum reward telescope residual: "
        f"{manifest['gates']['max_abs_reward_telescope_residual']:.3g}.\n\n"
        f"Strict-prior donor ability coverage: {ability_audit['available_rows']:,}/"
        f"{ability_audit['rows']:,} ({ability_audit['coverage']:.2%}). "
        "Missing donor ability remains fail-closed; it is not converted to zero.\n\n"
        "This artifact validates transition accounting and provenance only. It "
        "does not yet authorize an anchor solve or predictive scoring.\n"
    )
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    _output_index(output_dir)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--hole", type=int, default=6)
    parser.add_argument("--particles", type=int, default=128)
    parser.add_argument("--export-root", type=Path, default=precompute.DEFAULT_EXPORT)
    parser.add_argument("--history-db", type=Path, default=asof.DEFAULT_HISTORY_DB)
    parser.add_argument("--profile-dir", type=Path, default=precompute.DEFAULT_PROFILES)
    parser.add_argument("--baseline-cases", type=Path, default=precompute.DEFAULT_BASELINES)
    parser.add_argument(
        "--streaming-reference-dir", type=Path, default=streaming.DEFAULT_REFERENCE
    )
    parser.add_argument("--reference-value-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--target-input-dir", type=Path, default=DEFAULT_TARGET_DIR)
    parser.add_argument("--bridge", type=Path, default=asof.DEFAULT_BRIDGE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = build_bounded_real_slice(
        output_dir=args.output_dir,
        export_root=args.export_root,
        history_db=args.history_db,
        profile_dir=args.profile_dir,
        baseline_cases=args.baseline_cases,
        streaming_reference_dir=args.streaming_reference_dir,
        reference_value_dir=args.reference_value_dir,
        target_input_dir=args.target_input_dir,
        bridge_path=args.bridge,
        round_no=args.round,
        hole_no=args.hole,
        particles=args.particles,
    )
    print(output)


__all__ = [
    "APP_SUPPORT_MASS_TOLERANCE",
    "EXPLICIT_APP_HOLEOUT_VERSION",
    "ExplicitAppSupport",
    "IdentityPreservingSupportRepository",
    "TracedHoleExecution",
    "TracedStreamingShotChain",
    "attach_strict_prior_donor_abilities",
    "build_bounded_real_slice",
    "build_explicit_app_support",
]


if __name__ == "__main__":
    main()
