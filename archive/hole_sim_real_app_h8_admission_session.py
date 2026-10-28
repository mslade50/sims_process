"""Dormant fixed admission/session wrapper for the reviewed Lowry R1 H8 APP support.

The authority registry is deliberately empty in this source version.  Public
admission therefore fails before any source or artifact read.  A later,
separately reviewed change may populate only the marked literal-data block.
The physical weighting and retained-uniform selector remain frozen in
``hole_sim_real_app_whole_atom_selector_lowry_r1h8.py``.

This module exposes no callback, chain, anchor solve, simulation, or production
entry point.
"""

from __future__ import annotations

import ast as _ast
import builtins as _builtins
from dataclasses import Field as _DataclassField, dataclass as _dataclass
from hashlib import sha256 as _sha256
import _imp as _imp
import json as _json
import math as _math
import os as _os
import pathlib as _pathlib
import stat as _stat
import sys as _sys
from types import (
    FunctionType as _FunctionType,
    MappingProxyType as _MappingProxyType,
    ModuleType as _ModuleType,
)
from typing import Mapping as _Mapping, NoReturn as _NoReturn

import numpy as _np
import pandas as _pd
import pyarrow as _pa
import pyarrow.parquet as _pq


_INITIAL_PATH_ABSPATH = _os.path.abspath
_INITIAL_PATH_DIRNAME = _os.path.dirname
_PATH_SEPARATOR = _os.sep
_ARCHIVE = _INITIAL_PATH_DIRNAME(_INITIAL_PATH_ABSPATH(__file__))
_REPO = _INITIAL_PATH_DIRNAME(_ARCHIVE)
if _ARCHIVE not in _sys.path:
    _sys.path.insert(0, _ARCHIVE)

globals().pop("annotations", None)


REAL_APP_H8_ADMISSION_CONTRACT = (
    "fixed_lowry_r1h8_real_app_selector_admission_session_v1"
)
REAL_APP_H8_AUTHORITY_CONTRACT = (
    "fixed_lowry_r1h8_real_app_selector_authority_registry_v1"
)
REAL_APP_H8_RECORD_CONTRACT = "reviewed_data_only_lowry_r1h8_app_authority_v1"
REAL_APP_H8_SESSION_CONTRACT = "opaque_lowry_r1h8_app_selector_session_v1"
INTEGRATION_STATUS = "dormant_empty_registry_no_source_or_artifact_reads"

_SELF_RELATIVE = "archive/hole_sim_real_app_h8_admission_session.py"
_TEST_RELATIVE = "test_hole_sim_real_app_h8_admission_session.py"
_RECORD_TEST_RELATIVE = "test_hole_sim_real_app_h8_admission_record.py"
_SELF = _REPO + _PATH_SEPARATOR + _SELF_RELATIVE.replace("/", _PATH_SEPARATOR)
_TEST = _REPO + _PATH_SEPARATOR + _TEST_RELATIVE.replace("/", _PATH_SEPARATOR)
_RECORD_TEST = (
    _REPO + _PATH_SEPARATOR + _RECORD_TEST_RELATIVE.replace("/", _PATH_SEPARATOR)
)
_MAX_WRAPPER_SOURCE_BYTES = 512_000
_MAX_REVIEW_SOURCE_BYTES = 2_000_000
_EMPTY_CELL = object()
_SESSION_RECEIPT_BOUNDARY = (
    "identity_bound_exact_code_and_closure_no_mutable_registry_"
    "arbitrary_function_cell_forgery_outside_pure_python_boundary"
)

_SUPPORT_RELATIVE = (
    "permanent_data/"
    "hole_sim_tee_app_complete_atom_support_lowry_r1h8_raw_app_v1"
)
_CANDIDATE_RELATIVE = (
    "permanent_data/hole_sim_real_app_h8_selector_authority_candidate_v1"
)
_AUDIT_RELATIVE = (
    "permanent_data/hole_sim_real_app_h8_selector_authority_candidate_audit_v2"
)
_SUPPORT_ROOT = (
    _REPO + _PATH_SEPARATOR + _SUPPORT_RELATIVE.replace("/", _PATH_SEPARATOR)
)
_CANDIDATE_ROOT = (
    _REPO + _PATH_SEPARATOR + _CANDIDATE_RELATIVE.replace("/", _PATH_SEPARATOR)
)
_AUDIT_ROOT = _REPO + _PATH_SEPARATOR + _AUDIT_RELATIVE.replace("/", _PATH_SEPARATOR)
del _INITIAL_PATH_ABSPATH, _INITIAL_PATH_DIRNAME

_QUERY_KEY = "943f67faa5e345d395bf706e29c1a526e69879f80e14e81f55e255fd81dbf43e"
_SUPPORT_KEY = "app|raw|R1|H8|tee|B14|33204"
_SELECTOR_SHA256 = "f47c35d0345410bf165885e81afd8c0a0320f4950118889d2c4ac3fce6148bd0"
_SELECTOR_TEST_SHA256 = (
    "bd3875f4de9e9082597a065f719e5b09bc053a5722e30f461f219e6dd1a1f4c9"
)
_SELECTOR_RUNTIME_SHA256 = (
    "ab4607085ca3dfd75ee38deab7fddd5932be644aaac1dad5688622b4a452660c"
)
_SELECTOR_SUPPORT_SHA256 = (
    "f5fd780ea6c2a3e2795c8197f45f3486cd4a03606bac8e8c17b0548d7a07dc4c"
)
_FROZEN_SELECTOR_MODULE_SPECS = (
    (
        "policy",
        "hole_sim_donor_ability_policy",
        "archive/hole_sim_donor_ability_policy.py",
    ),
    (
        "anchor",
        "hole_sim_transition_mean_anchor",
        "archive/hole_sim_transition_mean_anchor.py",
    ),
    (
        "synthetic",
        "hole_sim_compact_app_whole_atom_selector_v1",
        "archive/hole_sim_compact_app_whole_atom_selector_v1.py",
    ),
    (
        "chain",
        "hole_sim_transition_mean_chain_adapter",
        "archive/hole_sim_transition_mean_chain_adapter.py",
    ),
    (
        "selector",
        "hole_sim_real_app_whole_atom_selector_lowry_r1h8",
        "archive/hole_sim_real_app_whole_atom_selector_lowry_r1h8.py",
    ),
)
_SELECTOR_CAPSULE_CONTRACT = "captured_byte_real_app_h8_selector_capsule_v9"
_SELECTOR_CAPSULE_INTEGRITY_CONTRACT = (
    "self_bound_state_identity_receipt_immutable_exports_integrity_v7"
)
_SELECTOR_CAPSULE_SHA256 = (
    "ada21c9666ab46604381f5e31c330b97125f8752648031cc8b0b9879455084f1"
)
_FROZEN_SELECTOR_SOURCE_BYTES = _MappingProxyType(
    {
        "archive/hole_sim_donor_ability_policy.py": 42_034,
        "archive/hole_sim_transition_mean_anchor.py": 46_786,
        "archive/hole_sim_transition_mean_chain_adapter.py": 84_720,
        "archive/hole_sim_compact_app_whole_atom_selector_v1.py": 90_091,
        "archive/hole_sim_real_app_whole_atom_selector_lowry_r1h8.py": 67_586,
    }
)

_COUNTS = _MappingProxyType(
    {
        "parts": 5,
        "complete_atoms": 35_218,
        "proposals": 904,
        "spatial_proposals": 903,
        "terminal_atoms": 1,
        "resolved_hazard_atoms": 30_702,
        "donor_quality_groups": 6,
        "mixed_policy_groups": 4,
        "observed_ability_atoms": 33_229,
        "imputed_ability_atoms": 1_989,
    }
)
_LIMITS = _MappingProxyType(
    {
        "parts": 5,
        "max_rows_per_part": 8_192,
        "max_atoms": 35_218,
        "max_proposals": 904,
        "surface_slots": 9,
        "max_hazards_per_surface": 34,
        "max_atoms_per_proposal": 16_384,
    }
)
_CAPABILITIES = _MappingProxyType(
    {
        "support_admission": True,
        "weighting": True,
        "selection": True,
        "callback": False,
        "chain": False,
        "anchor": False,
        "mean_solve": False,
        "simulation": False,
        "production": False,
        "research_only": True,
    }
)
_SUPPORT_IDENTITY = _MappingProxyType(
    {
        "outer_output_index_sha256": (
            "5a502f524ada60bb91e85f1e18474056d4b2a883be14c0e1f7df3450393093b2"
        ),
        "outer_manifest_sha256": (
            "c96a8e8333c8e535a82e9e265f74cde41d6a5a717e36c68c9dea9894e39f5464"
        ),
        "inner_output_index_sha256": (
            "e300bf8b00066ec02e03128ca8e0e6b7a89c4e5a772a2ac5f2dd5794dbe2693f"
        ),
        "inner_manifest_sha256": (
            "4dec00946a085fedb489aa51fef672188d4205c6a26c85faeb7156e5234da7f2"
        ),
        "source_authority_id": "lowry_r1h8_raw_app_exact_query_v1",
        "source_authority_record_sha256": (
            "efaf969daea1ac9a7680b0fd8d534ffa700430523aa7cb13cec6fabaed9c347c"
        ),
        "source_authority_lock_sha256": (
            "edbadf6498b1fd5166c6711f0ac20e87bb7d691ec58f7f511f660cd38b978a4c"
        ),
        "query_key": _QUERY_KEY,
        "support_key": _SUPPORT_KEY,
        "selector_support_sha256": _SELECTOR_SUPPORT_SHA256,
    }
)
_CANDIDATE_IDENTITY = _MappingProxyType(
    {
        "candidate_id": (
            "b5552aaea4e7c216b8bae8dc9ccdb68d122df54523d7b8faf1ddb70664dde942"
        ),
        "output_index_sha256": (
            "ee2c086ee7b42a816f33d1894251e01eb4a58bc1d673b7a866491931fe553b43"
        ),
        "candidate_json_sha256": (
            "689b5c0ae5827ea99fd603a7b3af39163685d498dd7c1ac85c0777758e2d6e00"
        ),
        "manifest_sha256": (
            "b5b5f07740dd43e8933aca21674351b435918b803a60fdfc5ad7dad78364c040"
        ),
        "report_sha256": (
            "5489db51378a60035b7d090036abcfe43f34878cf7ba2e6ce2fa4d1bbeacf879"
        ),
        "status": "candidate_external_source_and_artifact_review_required",
    }
)
_AUDIT_IDENTITY = _MappingProxyType(
    {
        "audit_id": (
            "f6b2392ceda66ac7c32e360dc63910fe2b1d8844dc9ed8da2d12d68276996cf0"
        ),
        "output_index_sha256": (
            "61ffccea5076070dfe1d0af65c40726fd509d9963b5583fa58af892d76104e57"
        ),
        "audit_json_sha256": (
            "074081e977225c97e5783f6addb4c928b3389b240f1a8d9f2867641423eff325"
        ),
        "manifest_sha256": (
            "5faae57aad9021d83e0fe29bebc9a8ce1b8c6da3952e9dd8ad77be5fe8b2e134"
        ),
        "report_sha256": (
            "40eddcc7ebd17e87e9216efdaa577c2f46b3a85394d1b3a2c36717f6b5649f5a"
        ),
        "status": "candidate_byte_and_vectorized_audit_passed_record_review_required",
        "decision": "GO_for_separate_data_only_authority_record_review_only",
    }
)

_SUPPORT_FILES = (
    "OUTPUT_SHA256.json",
    "REPORT.md",
    "complete_atom_support/OUTPUT_SHA256.json",
    "complete_atom_support/manifest.json",
    "complete_atom_support/parts/part-000000.parquet",
    "complete_atom_support/parts/part-000001.parquet",
    "complete_atom_support/parts/part-000002.parquet",
    "complete_atom_support/parts/part-000003.parquet",
    "complete_atom_support/parts/part-000004.parquet",
    "manifest.json",
)
_SUPPORT_DIRECTORIES = (
    "complete_atom_support",
    "complete_atom_support/parts",
)
_PART_FILES = _SUPPORT_FILES[4:9]
_CANDIDATE_FILES = (
    "OUTPUT_SHA256.json",
    "REPORT.md",
    "authority_candidate.json",
    "manifest.json",
)
_AUDIT_FILES = (
    "OUTPUT_SHA256.json",
    "REPORT.md",
    "audit.json",
    "manifest.json",
)

_KNOWN_SOURCE_SHA256 = _MappingProxyType(
    {
        "archive/audit_hole_sim_real_app_h8_selector_authority_candidate_v2.py": (
            "93b790ba2adc34df5c6ea2e65ae88c989c4f55a2747c922e11cc4f786b0e35f0"
        ),
        "archive/hole_sim_compact_app_whole_atom_selector_v1.py": (
            "e0606e0818fdeb625f582b3379b109d50d9485e2866268930b68b08c4dd238e7"
        ),
        "archive/hole_sim_donor_ability_policy.py": (
            "3f9593333a79ffde258ec1efb149294bd1b036721c30cfcf5d47cf5ea2aa7339"
        ),
        "archive/hole_sim_real_app_h8_selector_authority_candidate.py": (
            "b3fdbd12f2e466bdac92f71f2d60a3e9fd0aec18a58f309248dc1c783046fdd7"
        ),
        "archive/hole_sim_real_app_whole_atom_selector_lowry_r1h8.py": (
            _SELECTOR_SHA256
        ),
        "archive/hole_sim_tee_app_complete_atom_support.py": (
            "597f068d68cee7580a964afc687b29a66ad3cfe29b8c2fa47b11172abedf6498"
        ),
        "archive/hole_sim_tee_app_complete_atom_support_lowry_r1h8_runner.py": (
            "ed12a40ad7a65117cf00d846997686a511d85b3cdf5537e3d5671b1464e96460"
        ),
        "archive/hole_sim_tee_app_h8_source_authority.py": (
            "809c163db737bc4d628101d4f4687543863ffd7a995f5ee9ece04d5d88c47182"
        ),
        "archive/hole_sim_transition_mean_anchor.py": (
            "55071ec977770897de18d0761d02fef32ce640343f760ad1f40e63eeb553be28"
        ),
        "archive/hole_sim_transition_mean_chain_adapter.py": (
            "84137c1bc80b198225796cee40ad6449810f72491d0194f2314e033e52734cff"
        ),
        "archive/run_audit_hole_sim_real_app_h8_selector_authority_candidate_v2.py": (
            "f36852ccf397dfb46c8853577a5c66587b8f78cc3c98fa2a74d493b84c5e21e0"
        ),
        "test_audit_hole_sim_real_app_h8_selector_authority_candidate_v2.py": (
            "11f9a55e89a22e1bb77addf09dc9fb395fd645cae6c6337c0d4a242147017eb9"
        ),
        "test_hole_sim_compact_app_whole_atom_selector_v1.py": (
            "21bd3aa8f0f1d2340bdefb32b3fce4bae59fdbdc87c23de3129710c7287b77dd"
        ),
        "test_hole_sim_real_app_h8_selector_authority_candidate.py": (
            "ebea0b3f4f4e196931908f89eeb3c7189b79784cfb5eb53a2f8027a342773db4"
        ),
        "test_hole_sim_real_app_whole_atom_selector_lowry_r1h8.py": (
            _SELECTOR_TEST_SHA256
        ),
        "test_hole_sim_tee_app_complete_atom_support.py": (
            "eb648d94447d8a60497dcf95b34dcbbe1a64c21cd5458bb27ad87c1673898397"
        ),
        "test_hole_sim_tee_app_complete_atom_support_lowry_r1h8_runner.py": (
            "61103a4f410fb2f2fcbdcec2ad20dc40c13c2c37a8975bae4587aa839b78153f"
        ),
        "test_hole_sim_tee_app_h8_source_authority.py": (
            "93966e84c85d7d3440516f0c80492e8c7d9145448aeaf88859a8cd833f115813"
        ),
        "archive/audit_hole_sim_tee_app_h8_source_authority.py": (
            "8d9bb4ae1f3e3c8c0cca875af0aa827afe237545dc657f86f1ed8963787aac06"
        ),
        "test_audit_hole_sim_tee_app_h8_source_authority.py": (
            "aa44f738e85db98a3faf287f553fec7cb9651266961cc1e6fee4c77eef1a350c"
        ),
        "test_run_audit_hole_sim_real_app_h8_selector_authority_candidate_v2.py": (
            "5e6db4fc2f767c1f3ab437f661f79b327f8dbf45739d24984f562fc6870730b6"
        ),
    }
)
_KNOWN_SOURCE_PATHS = _MappingProxyType(
    {
        relative: _REPO
        + _PATH_SEPARATOR
        + relative.replace("/", _PATH_SEPARATOR)
        for relative in _KNOWN_SOURCE_SHA256
    }
)

_NDJSON = _MappingProxyType(
    {
        "schema": (
            "lowry_r1h8_v2_schema_ndjson_sha256",
            151,
            14_888,
            "f711d7c44747222b7a5abde522996b8e4d341c694f0667acf936024d54cee944",
        ),
        "topology": (
            "lowry_r1h8_v2_topology_ndjson_sha256",
            905,
            2_968_153,
            "f20b53bf6cb5310685281f4519fbeeaa6ae6937122c13079c7a6175020924d97",
        ),
        "mass_cdf": (
            "lowry_r1h8_v2_mass_cdf_ndjson_sha256",
            905,
            13_115_551,
            "bcd7acafb2fed51320e96e0e9dbf2b30ca4e50a964d5ddaefa1c9428d5df19be",
        ),
        "group": (
            "lowry_r1h8_v2_group_ndjson_sha256",
            7,
            5_653,
            "cebf146bf965e293e2b5be8cb29895493c4cddba08530f8764f78b021aa33484",
        ),
        "ability_policy": (
            "lowry_r1h8_v2_ability_policy_ndjson_sha256",
            905,
            887_469,
            "acda28fae2bdf86a038b369f5bfcb6f711ca26204acfc474ed22b8421fafba55",
        ),
        "reference_reward": (
            "lowry_r1h8_v2_reference_reward_ndjson_sha256",
            35_219,
            16_878_023,
            "45cf3527ee448cb2d008ea155ec6220d2a0a80391e970d5a48842d75f953e88c",
        ),
        "semantic_root": (
            "lowry_r1h8_v2_semantic_root_ndjson_sha256",
            1,
            3_379,
            "b8312616d491bdce5e97fa438e95a8b9ead36ce1cb0fb9d7274605cfe3675dae",
        ),
    }
)
_EXTERNAL_EVIDENCE = _MappingProxyType(
    {
        "full_schema_sha256": (
            "113ef43b17f03ad9d9d7c9112d0636a87445ca9b97738935bec969f98b7b6a4d"
        ),
        "full_support_sha256": (
            "b66b8703d1b156aedbadded022a9ba1f1bdd821c7f8b585103fa4f8de747dce7"
        ),
        "projection_schema_sha256": (
            "855c4b57abfe5ea0f29e23f1e3ffab82e58b186f09358bb1adcff17eb59c87f2"
        ),
        "projection_54_sha256": (
            "a9957e4e20ed0259ce90a5abf27d29b7d4dcc8ee7259362e41c7ba7c9c178889"
        ),
        "selector_native_sha256": _SELECTOR_SUPPORT_SHA256,
        "topology_30_sha256": (
            "9e8e3a3128aa12ddae281ed0b37e036bee833232aa01ca9b93bd0370552af484"
        ),
        "reference_model_sha256": (
            "35ffdc1aa21ccc209ceb56a8b8e78b3118aa1cc8bd4c7372a92a0dc01b0bc68c"
        ),
        "putting_model_sha256": (
            "aace8326debb5c2af07b16e319730601a56d9ddd978d31ce4a0d3ee1d82f7c93"
        ),
        "policy_identity_sha256": (
            "f30e0161a1767d42f01dc0a33f617d93327286e3f49c59287c9b751686ba5bc7"
        ),
    }
)


class RealAppH8AdmissionUnavailable(ValueError):
    """Raised whenever the fixed real-H8 admission is not exactly authorized."""


def _make_low_level_filesystem_capsule() -> _MappingProxyType:
    """Capture the exact filesystem primitives used after module initialization.

    Runtime admission never dispatches through ``pathlib``.  The small pathlib
    snapshot below is only a fail-closed sentinel for the private frozen modules,
    which constructed their inert path constants while the capsule was loaded.
    Sentinel checks compare identities and code objects; they never invoke a
    replacement collaborator.
    """

    os_module = _os
    stat_module = _stat
    pathlib_module = _pathlib
    error_type = RealAppH8AdmissionUnavailable
    exact_type = type
    exact_int = int
    exact_float = float
    exact_len = len
    exact_str = str
    exact_tuple = tuple
    exact_dict = dict
    exact_set = set
    exact_frozenset = frozenset
    exact_sorted = sorted
    exact_bytes = bytes
    exact_bytearray = bytearray
    exact_list = list
    exact_bool = bool
    exact_getattr = getattr
    exact_vars = vars
    exact_any = any
    exact_id = id
    exact_float_hex = float.hex
    exact_object_getattribute = object.__getattribute__
    exact_mappingproxy_items = _MappingProxyType.items
    exact_dict_items = dict.items
    exact_staticmethod = staticmethod
    exact_classmethod = classmethod
    exact_property = property
    attribute_error = AttributeError
    type_error = TypeError
    os_error = OSError

    os_lstat = _os.lstat
    os_open = _os.open
    os_read = _os.read
    os_close = _os.close
    os_fstat = _os.fstat
    os_listdir = _os.listdir
    os_flag_rows = tuple(
        (name, getattr(_os, name, None))
        for name in (
            "O_RDONLY",
            "O_BINARY",
            "O_CLOEXEC",
            "O_NOINHERIT",
            "O_NOFOLLOW",
        )
    )
    stat_s_isreg = _stat.S_ISREG
    stat_s_isdir = _stat.S_ISDIR
    stat_s_islnk = _stat.S_ISLNK
    reparse_flag = exact_int(
        exact_getattr(_stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )
    open_flags = exact_int(_os.O_RDONLY)
    open_flags |= exact_int(exact_getattr(_os, "O_BINARY", 0))
    open_flags |= exact_int(exact_getattr(_os, "O_CLOEXEC", 0))
    open_flags |= exact_int(exact_getattr(_os, "O_NOINHERIT", 0))
    open_flags |= exact_int(exact_getattr(_os, "O_NOFOLLOW", 0))
    separator = _PATH_SEPARATOR

    concrete_path_type = exact_type(_pathlib.Path(_ARCHIVE))
    path_factory = _pathlib.Path
    path_mro = exact_tuple(
        value for value in concrete_path_type.__mro__ if value is not object
    )
    path_accessor = concrete_path_type._accessor
    path_flavour = concrete_path_type._flavour
    collaborator_types = (exact_type(path_accessor), exact_type(path_flavour))
    sentinel_classes = path_mro + collaborator_types

    def sentinel_snapshot(value: object) -> object:
        """Describe inert pathlib mutable state without polymorphic callbacks."""

        remaining = [100_000]
        active: set[int] = exact_set()

        def visit(item: object) -> object:
            remaining[0] -= 1
            if remaining[0] < 0:
                raise error_type("Inert pathlib sentinel exceeded its fixed budget")
            item_type = exact_type(item)
            if item is None:
                return ("none",)
            if item_type is exact_bool:
                return ("bool", item)
            if item_type is exact_int:
                return ("int", item)
            if item_type is exact_float:
                return ("float", exact_float_hex(item))
            if item_type is exact_str:
                return ("str", item)
            if item_type is exact_bytes:
                return ("bytes", item)
            if item_type not in (
                exact_tuple,
                exact_list,
                exact_set,
                exact_frozenset,
                exact_dict,
                _MappingProxyType,
            ):
                return ("identity", exact_id(item_type), exact_id(item))
            item_id = exact_id(item)
            if item_id in active:
                return ("cycle", item_id)
            active.add(item_id)
            try:
                if item_type is exact_tuple:
                    return ("tuple", exact_tuple(visit(child) for child in item))
                if item_type is exact_list:
                    return ("list", exact_tuple(visit(child) for child in item))
                if item_type is exact_set:
                    return (
                        "set",
                        exact_frozenset(visit(child) for child in item),
                    )
                if item_type is exact_frozenset:
                    return (
                        "frozenset",
                        exact_frozenset(visit(child) for child in item),
                    )
                items = (
                    exact_dict_items(item)
                    if item_type is exact_dict
                    else exact_mappingproxy_items(item)
                )
                return (
                    "mapping",
                    exact_frozenset(
                        (visit(key), visit(child)) for key, child in items
                    ),
                )
            finally:
                active.remove(item_id)

        return visit(value)

    def sentinel_instance_state(value: object) -> object:
        try:
            state = exact_object_getattribute(value, "__dict__")
        except (attribute_error, type_error):
            return ("missing",)
        if exact_type(state) is not exact_dict:
            return ("non_dict", exact_id(exact_type(state)), exact_id(state))
        return ("dict", exact_id(state), sentinel_snapshot(state))

    def sentinel_class_row(owner: type) -> tuple[object, ...]:
        items = exact_dict(exact_vars(owner))
        function_rows: list[tuple[str, object, object]] = exact_list()
        mutable_rows: list[tuple[str, object, object]] = exact_list()
        for name, value in exact_dict_items(items):
            if exact_type(value) is _FunctionType:
                function_rows.append((name, value, value.__code__))
            elif exact_type(value) in (exact_staticmethod, exact_classmethod):
                function = value.__func__
                if exact_type(function) is _FunctionType:
                    function_rows.append((name, function, function.__code__))
            elif exact_type(value) is exact_property:
                for suffix, function in (
                    ("fget", value.fget),
                    ("fset", value.fset),
                    ("fdel", value.fdel),
                ):
                    if exact_type(function) is _FunctionType:
                        function_rows.append(
                            (f"{name}.{suffix}", function, function.__code__)
                        )
            if exact_type(value) in (
                exact_dict,
                exact_list,
                exact_set,
                exact_tuple,
                exact_frozenset,
                _MappingProxyType,
            ):
                mutable_rows.append((name, value, sentinel_snapshot(value)))
        return (
            owner,
            _MappingProxyType(items),
            exact_tuple(function_rows),
            exact_tuple(mutable_rows),
        )

    sentinel_class_rows = exact_tuple(
        sentinel_class_row(owner)
        for owner in sentinel_classes
    )
    sentinel_instance_rows = (
        ("path accessor", path_accessor, sentinel_instance_state(path_accessor)),
        ("path flavour", path_flavour, sentinel_instance_state(path_flavour)),
    )
    pathlib_s_islnk = _pathlib.S_ISLNK
    pathlib_os = _pathlib.os
    pathlib_normal_accessor = _pathlib._normal_accessor

    def is_reparse(stat_result: object) -> bool:
        attributes = exact_int(
            exact_getattr(stat_result, "st_file_attributes", 0)
        )
        return exact_bool(attributes & reparse_flag)

    def regular(stat_result: object) -> bool:
        return (
            stat_s_isreg(stat_result.st_mode)
            and not stat_s_islnk(stat_result.st_mode)
            and not is_reparse(stat_result)
        )

    def directory(stat_result: object) -> bool:
        return (
            stat_s_isdir(stat_result.st_mode)
            and not stat_s_islnk(stat_result.st_mode)
            and not is_reparse(stat_result)
        )

    def stat_signature(stat_result: object) -> tuple[object, ...]:
        return (
            stat_result.st_mode,
            stat_result.st_dev,
            stat_result.st_ino,
            stat_result.st_size,
            stat_result.st_mtime_ns,
            exact_int(exact_getattr(stat_result, "st_file_attributes", 0)),
        )

    def filesystem_state() -> dict[str, object]:
        if (
            os_module.lstat is not os_lstat
            or os_module.open is not os_open
            or os_module.read is not os_read
            or os_module.close is not os_close
            or os_module.fstat is not os_fstat
            or os_module.listdir is not os_listdir
            or exact_any(
                exact_getattr(os_module, name, None) != expected
                or exact_type(exact_getattr(os_module, name, None))
                is not exact_type(expected)
                for name, expected in os_flag_rows
            )
            or stat_module.S_ISREG is not stat_s_isreg
            or stat_module.S_ISDIR is not stat_s_isdir
            or stat_module.S_ISLNK is not stat_s_islnk
            or exact_int(
                exact_getattr(stat_module, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            )
            != reparse_flag
        ):
            raise error_type("Captured low-level filesystem dependency changed")
        if (
            pathlib_module.Path is not path_factory
            or pathlib_module.S_ISLNK is not pathlib_s_islnk
            or pathlib_module.os is not pathlib_os
            or pathlib_module._normal_accessor is not pathlib_normal_accessor
            or concrete_path_type._accessor is not path_accessor
            or concrete_path_type._flavour is not path_flavour
            or exact_type(path_accessor) is not collaborator_types[0]
            or exact_type(path_flavour) is not collaborator_types[1]
        ):
            raise error_type("Inert pathlib sentinel changed")
        for owner, expected_items, expected_functions, expected_mutable in (
            sentinel_class_rows
        ):
            observed = exact_vars(owner)
            if (
                exact_set(observed) != exact_set(expected_items)
                or exact_any(
                    observed[name] is not value
                    for name, value in expected_items.items()
                )
            ):
                raise error_type("Inert pathlib collaborator changed")
            for _name, function, code in expected_functions:
                if (
                    exact_type(function) is not _FunctionType
                    or function.__code__ is not code
                ):
                    raise error_type("Inert pathlib descriptor changed")
            for name, value, snapshot in expected_mutable:
                if (
                    observed[name] is not value
                    or sentinel_snapshot(value) != snapshot
                ):
                    raise error_type("Inert pathlib class constant changed")
        for _label, value, expected_state in sentinel_instance_rows:
            if sentinel_instance_state(value) != expected_state:
                raise error_type("Inert pathlib instance state changed")
        return {
            "contract": "captured_low_level_filesystem_primitives_v2",
            "open_flags": open_flags,
            "separator": separator,
        }

    def read_file(
        path: str,
        maximum_bytes: int,
        expected_bytes: int | None,
        label: str,
    ) -> bytes:
        if (
            exact_type(path) is not exact_str
            or exact_type(maximum_bytes) is not exact_int
            or maximum_bytes < 0
            or exact_type(label) is not exact_str
            or (
                expected_bytes is not None
                and (
                    exact_type(expected_bytes) is not exact_int
                    or expected_bytes < 0
                    or expected_bytes > maximum_bytes
                )
            )
        ):
            raise error_type(f"{label} read contract changed")
        try:
            before = os_lstat(path)
        except os_error as exc:
            raise error_type(f"{label} cannot be inspected") from exc
        if not regular(before) or before.st_size > maximum_bytes:
            raise error_type(f"{label} is not a bounded regular file")
        if expected_bytes is not None and before.st_size != expected_bytes:
            raise error_type(f"{label} byte size changed")
        descriptor: int | None = None
        chunks = exact_bytearray()
        try:
            descriptor = os_open(path, open_flags)
            opened = os_fstat(descriptor)
            if (
                not regular(opened)
                or opened.st_size != before.st_size
                or (opened.st_dev, opened.st_ino)
                != (before.st_dev, before.st_ino)
            ):
                raise error_type(f"{label} opened file changed")
            while exact_len(chunks) <= maximum_bytes:
                block = os_read(descriptor, maximum_bytes + 1 - exact_len(chunks))
                if exact_type(block) is not exact_bytes:
                    raise error_type(f"{label} low-level read type changed")
                if not block:
                    break
                chunks.extend(block)
            opened_after = os_fstat(descriptor)
            if (
                not regular(opened_after)
                or stat_signature(opened) != stat_signature(opened_after)
            ):
                raise error_type(f"{label} opened file changed during read")
        except error_type:
            raise
        except os_error as exc:
            raise error_type(f"{label} low-level read failed") from exc
        finally:
            if descriptor is not None:
                try:
                    os_close(descriptor)
                except os_error as exc:
                    raise error_type(f"{label} descriptor close failed") from exc
        raw = exact_bytes(chunks)
        if exact_len(raw) != before.st_size or exact_len(raw) > maximum_bytes:
            raise error_type(f"{label} bounded read changed")
        if expected_bytes is not None and exact_len(raw) != expected_bytes:
            raise error_type(f"{label} bounded read changed")
        try:
            after = os_lstat(path)
        except os_error as exc:
            raise error_type(f"{label} changed during read") from exc
        if not regular(after) or stat_signature(before) != stat_signature(after):
            raise error_type(f"{label} changed during read")
        return raw

    def read_capped(path: str, maximum_bytes: int, label: str) -> bytes:
        if exact_type(maximum_bytes) is not exact_int or maximum_bytes <= 0:
            raise error_type(f"{label} maximum byte count changed")
        return read_file(path, maximum_bytes, None, label)

    def read_expected(path: str, expected_bytes: int, label: str) -> bytes:
        if exact_type(expected_bytes) is not exact_int or expected_bytes < 0:
            raise error_type(f"{label} expected byte count changed")
        return read_file(path, expected_bytes, expected_bytes, label)

    def capture_layout(
        root: str,
        expected_paths: tuple[str, ...],
        expected_directories: tuple[str, ...],
        label: str,
    ) -> dict[str, object]:
        if (
            exact_type(root) is not exact_str
            or exact_type(expected_paths) is not exact_tuple
            or exact_type(expected_directories) is not exact_tuple
            or exact_type(label) is not exact_str
            or exact_any(exact_type(value) is not exact_str for value in expected_paths)
            or exact_any(
                exact_type(value) is not exact_str for value in expected_directories
            )
        ):
            raise error_type(f"{label} layout contract changed")
        expected_path_set = exact_set(expected_paths)
        expected_directory_set = exact_set(expected_directories)
        if (
            exact_len(expected_path_set) != exact_len(expected_paths)
            or exact_len(expected_directory_set) != exact_len(expected_directories)
        ):
            raise error_type(f"{label} layout closure changed")
        try:
            root_before = os_lstat(root)
        except os_error as exc:
            raise error_type(f"{label} root is invalid") from exc
        if not directory(root_before):
            raise error_type(f"{label} root is invalid")
        paths: dict[str, str] = exact_dict()
        directories: list[str] = exact_list()
        rows: list[tuple[object, ...]] = exact_list()

        def walk(folder: str, relative: str = "") -> None:
            try:
                names = os_listdir(folder)
            except os_error as exc:
                raise error_type(f"{label} directory cannot be listed") from exc
            if exact_type(names) is not exact_list or exact_any(
                exact_type(name) is not exact_str
                or not name
                or name in (".", "..")
                or "/" in name
                or "\\" in name
                for name in names
            ):
                raise error_type(f"{label} directory listing changed")
            for name in exact_sorted(names):
                child_relative = f"{relative}/{name}" if relative else name
                child = folder + separator + name
                try:
                    stat_result = os_lstat(child)
                except os_error as exc:
                    raise error_type(f"{label} entry cannot be inspected") from exc
                signature = stat_signature(stat_result)
                if directory(stat_result):
                    if child_relative not in expected_directory_set:
                        raise error_type(f"{label} directory closure changed")
                    directories.append(child_relative)
                    rows.append(("directory", child_relative, *signature))
                    walk(child, child_relative)
                elif regular(stat_result):
                    if child_relative not in expected_path_set:
                        raise error_type(f"{label} file closure changed")
                    paths[child_relative] = child
                    rows.append(("file", child_relative, *signature))
                else:
                    raise error_type(f"{label} contains a link or special entry")

        walk(root)
        try:
            root_after = os_lstat(root)
        except os_error as exc:
            raise error_type(f"{label} root changed during scan") from exc
        if (
            not directory(root_after)
            or stat_signature(root_before) != stat_signature(root_after)
        ):
            raise error_type(f"{label} root changed during scan")
        if exact_tuple(directories) != expected_directories:
            raise error_type(f"{label} directory closure changed")
        if exact_set(paths) != expected_path_set:
            raise error_type(f"{label} file closure changed")
        return {
            "paths": paths,
            "directories": exact_tuple(directories),
            "rows": exact_tuple(rows),
            "root": stat_signature(root_after),
        }

    filesystem_state()
    helpers = _MappingProxyType(
        {
            "state": filesystem_state,
            "sentinel_snapshot": sentinel_snapshot,
            "sentinel_instance_state": sentinel_instance_state,
            "is_reparse": is_reparse,
            "regular": regular,
            "directory": directory,
            "stat_signature": stat_signature,
            "read_file": read_file,
            "read_capped": read_capped,
            "read_expected": read_expected,
            "capture_layout": capture_layout,
        }
    )
    return _MappingProxyType({"helpers": helpers})


_LOW_LEVEL_FILESYSTEM = _make_low_level_filesystem_capsule()
_FILESYSTEM_HELPERS = _LOW_LEVEL_FILESYSTEM["helpers"]
_FILESYSTEM_STATE = _FILESYSTEM_HELPERS["state"]
_FILESYSTEM_IS_REPARSE = _FILESYSTEM_HELPERS["is_reparse"]
_FILESYSTEM_READ_CAPPED = _FILESYSTEM_HELPERS["read_capped"]
_FILESYSTEM_READ_EXPECTED = _FILESYSTEM_HELPERS["read_expected"]
_FILESYSTEM_CAPTURE_LAYOUT = _FILESYSTEM_HELPERS["capture_layout"]
del _LOW_LEVEL_FILESYSTEM, _make_low_level_filesystem_capsule


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _json_bytes(value: object) -> bytes:
    return _json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _stable_sha(value: object) -> str:
    return _sha256(_json_bytes(value)).hexdigest()


def _domain_sha(value: object, domain: str) -> str:
    if type(domain) is not str or not domain.isascii():
        raise RealAppH8AdmissionUnavailable("Digest domain must be plain ASCII")
    return _sha256(domain.encode("ascii") + b"\0" + _json_bytes(value)).hexdigest()


def _freeze_tree(value: object) -> object:
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise RealAppH8AdmissionUnavailable(
                "Authority mapping keys must be exact built-in strings"
            )
        return _MappingProxyType(
            {key: _freeze_tree(item) for key, item in value.items()}
        )
    if type(value) in (list, tuple):
        return tuple(_freeze_tree(item) for item in value)
    if value is None or type(value) in (str, int, bool):
        return value
    raise RealAppH8AdmissionUnavailable(
        "Authority literals must contain exact JSON scalar/container types"
    )


def _plain_tree(value: object) -> object:
    if type(value) is _MappingProxyType:
        return {key: _plain_tree(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_plain_tree(item) for item in value]
    return value


_BLOCK_BEGIN = bytes.fromhex(
    "23203d3d3d20424547494e204c4f5752595f523148385f415554484f524954595f"
    "444154415f5631203d3d3d"
)
_BLOCK_END = bytes.fromhex(
    "23203d3d3d20454e44204c4f5752595f523148385f415554484f524954595f4441"
    "54415f5631203d3d3d"
)
_EMPTY_AUTHORITY_LOCK = (
    "ff3f6d679ea4ab271f88b7869b9317608492c9e464d394a956ab4a20191bd814"
)
_DORMANT_BLOCK = (
    _BLOCK_BEGIN
    + b"\n_AUTHORITY_RECORD_LITERALS = ()\n_AUTHORITY_LOCK_SHA256 = \""
    + _EMPTY_AUTHORITY_LOCK.encode("ascii")
    + b"\"\n"
    + _BLOCK_END
    + b"\n"
)

# === BEGIN LOWRY_R1H8_AUTHORITY_DATA_V1 ===
_AUTHORITY_RECORD_LITERALS = ({'audit_artifact': {'bytes': 14450,
                     'directories': [],
                     'files': {'OUTPUT_SHA256.json': {'bytes': 462,
                                                      'sha256': '61ffccea5076070dfe1d0af65c40726fd509d9963b5583fa58af892d76104e57'},
                               'REPORT.md': {'bytes': 634,
                                             'sha256': '40eddcc7ebd17e87e9216efdaa577c2f46b3a85394d1b3a2c36717f6b5649f5a'},
                               'audit.json': {'bytes': 12110,
                                              'sha256': '074081e977225c97e5783f6addb4c928b3389b240f1a8d9f2867641423eff325'},
                               'manifest.json': {'bytes': 1244,
                                                 'sha256': '5faae57aad9021d83e0fe29bebc9a8ce1b8c6da3952e9dd8ad77be5fe8b2e134'}},
                     'identity': {'audit_id': 'f6b2392ceda66ac7c32e360dc63910fe2b1d8844dc9ed8da2d12d68276996cf0',
                                  'audit_json_sha256': '074081e977225c97e5783f6addb4c928b3389b240f1a8d9f2867641423eff325',
                                  'decision': 'GO_for_separate_data_only_authority_record_review_only',
                                  'manifest_sha256': '5faae57aad9021d83e0fe29bebc9a8ce1b8c6da3952e9dd8ad77be5fe8b2e134',
                                  'output_index_sha256': '61ffccea5076070dfe1d0af65c40726fd509d9963b5583fa58af892d76104e57',
                                  'report_sha256': '40eddcc7ebd17e87e9216efdaa577c2f46b3a85394d1b3a2c36717f6b5649f5a',
                                  'status': 'candidate_byte_and_vectorized_audit_passed_record_review_required'},
                     'inventory_contract': 'audit_v2_output_tree',
                     'inventory_sha256': 'af4c1c2fe1087c730385a316f610850dacd5bb2a1dba76c6c0714487f6a01687',
                     'root': 'permanent_data/hole_sim_real_app_h8_selector_authority_candidate_audit_v2'},
  'authority_id': '1a9d6fd9133811799a74cc61fcdc0ccbeec47617da11ef4a993ce3b2dcd529ce',
  'candidate_artifact': {'bytes': 7203,
                         'directories': [],
                         'files': {'OUTPUT_SHA256.json': {'bytes': 474,
                                                          'sha256': 'ee2c086ee7b42a816f33d1894251e01eb4a58bc1d673b7a866491931fe553b43'},
                                   'REPORT.md': {'bytes': 519,
                                                 'sha256': '5489db51378a60035b7d090036abcfe43f34878cf7ba2e6ce2fa4d1bbeacf879'},
                                   'authority_candidate.json': {'bytes': 5283,
                                                                'sha256': '689b5c0ae5827ea99fd603a7b3af39163685d498dd7c1ac85c0777758e2d6e00'},
                                   'manifest.json': {'bytes': 927,
                                                     'sha256': 'b5b5f07740dd43e8933aca21674351b435918b803a60fdfc5ad7dad78364c040'}},
                         'identity': {'candidate_id': 'b5552aaea4e7c216b8bae8dc9ccdb68d122df54523d7b8faf1ddb70664dde942',
                                      'candidate_json_sha256': '689b5c0ae5827ea99fd603a7b3af39163685d498dd7c1ac85c0777758e2d6e00',
                                      'manifest_sha256': 'b5b5f07740dd43e8933aca21674351b435918b803a60fdfc5ad7dad78364c040',
                                      'output_index_sha256': 'ee2c086ee7b42a816f33d1894251e01eb4a58bc1d673b7a866491931fe553b43',
                                      'report_sha256': '5489db51378a60035b7d090036abcfe43f34878cf7ba2e6ce2fa4d1bbeacf879',
                                      'status': 'candidate_external_source_and_artifact_review_required'},
                         'inventory_contract': 'audit_v2_input_tree',
                         'inventory_sha256': '88268ebdef2bb6ff872ad232f08cf7c398dc18dc82fc3f80f4424a549a037d59',
                         'root': 'permanent_data/hole_sim_real_app_h8_selector_authority_candidate_v1'},
  'capabilities': {'anchor': False,
                   'callback': False,
                   'chain': False,
                   'mean_solve': False,
                   'production': False,
                   'research_only': True,
                   'selection': True,
                   'simulation': False,
                   'support_admission': True,
                   'weighting': True},
  'limits': {'max_atoms': 35218,
             'max_atoms_per_proposal': 16384,
             'max_hazards_per_surface': 34,
             'max_proposals': 904,
             'max_rows_per_part': 8192,
             'parts': 5,
             'surface_slots': 9},
  'logic': {'reviewed_dormant_test_sha256': 'fe1d9b11942838e3983ab29a45399a8f551ff0578256b176219f3601948bb0be',
            'reviewed_dormant_wrapper_sha256': '7863f020a0f4b19f00f8fe53a26eb623d4ef2d84a8563e491bb56f86b049e362',
            'reviewed_record_test_sha256': '30f5d300356cce91b3f19fbe2e90924494b9918ef2aab22cd40c17c0f1864b54',
            'selector_capsule_sha256': 'ada21c9666ab46604381f5e31c330b97125f8752648031cc8b0b9879455084f1',
            'selector_runtime_sha256': 'ab4607085ca3dfd75ee38deab7fddd5932be644aaac1dad5688622b4a452660c',
            'selector_source_sha256': 'f47c35d0345410bf165885e81afd8c0a0320f4950118889d2c4ac3fce6148bd0',
            'selector_support_sha256': 'f5fd780ea6c2a3e2795c8197f45f3486cd4a03606bac8e8c17b0548d7a07dc4c',
            'selector_test_sha256': 'bd3875f4de9e9082597a065f719e5b09bc053a5722e30f461f219e6dd1a1f4c9'},
  'record_contract': 'reviewed_data_only_lowry_r1h8_app_authority_v1',
  'review_sources': {'archive/audit_hole_sim_real_app_h8_selector_authority_candidate_v2.py': '93b790ba2adc34df5c6ea2e65ae88c989c4f55a2747c922e11cc4f786b0e35f0',
                     'archive/audit_hole_sim_tee_app_h8_source_authority.py': '8d9bb4ae1f3e3c8c0cca875af0aa827afe237545dc657f86f1ed8963787aac06',
                     'archive/hole_sim_compact_app_whole_atom_selector_v1.py': 'e0606e0818fdeb625f582b3379b109d50d9485e2866268930b68b08c4dd238e7',
                     'archive/hole_sim_donor_ability_policy.py': '3f9593333a79ffde258ec1efb149294bd1b036721c30cfcf5d47cf5ea2aa7339',
                     'archive/hole_sim_real_app_h8_selector_authority_candidate.py': 'b3fdbd12f2e466bdac92f71f2d60a3e9fd0aec18a58f309248dc1c783046fdd7',
                     'archive/hole_sim_real_app_whole_atom_selector_lowry_r1h8.py': 'f47c35d0345410bf165885e81afd8c0a0320f4950118889d2c4ac3fce6148bd0',
                     'archive/hole_sim_tee_app_complete_atom_support.py': '597f068d68cee7580a964afc687b29a66ad3cfe29b8c2fa47b11172abedf6498',
                     'archive/hole_sim_tee_app_complete_atom_support_lowry_r1h8_runner.py': 'ed12a40ad7a65117cf00d846997686a511d85b3cdf5537e3d5671b1464e96460',
                     'archive/hole_sim_tee_app_h8_source_authority.py': '809c163db737bc4d628101d4f4687543863ffd7a995f5ee9ece04d5d88c47182',
                     'archive/hole_sim_transition_mean_anchor.py': '55071ec977770897de18d0761d02fef32ce640343f760ad1f40e63eeb553be28',
                     'archive/hole_sim_transition_mean_chain_adapter.py': '84137c1bc80b198225796cee40ad6449810f72491d0194f2314e033e52734cff',
                     'archive/run_audit_hole_sim_real_app_h8_selector_authority_candidate_v2.py': 'f36852ccf397dfb46c8853577a5c66587b8f78cc3c98fa2a74d493b84c5e21e0',
                     'test_audit_hole_sim_real_app_h8_selector_authority_candidate_v2.py': '11f9a55e89a22e1bb77addf09dc9fb395fd645cae6c6337c0d4a242147017eb9',
                     'test_audit_hole_sim_tee_app_h8_source_authority.py': 'aa44f738e85db98a3faf287f553fec7cb9651266961cc1e6fee4c77eef1a350c',
                     'test_hole_sim_compact_app_whole_atom_selector_v1.py': '21bd3aa8f0f1d2340bdefb32b3fce4bae59fdbdc87c23de3129710c7287b77dd',
                     'test_hole_sim_real_app_h8_admission_record.py': '30f5d300356cce91b3f19fbe2e90924494b9918ef2aab22cd40c17c0f1864b54',
                     'test_hole_sim_real_app_h8_admission_session.py': 'fe1d9b11942838e3983ab29a45399a8f551ff0578256b176219f3601948bb0be',
                     'test_hole_sim_real_app_h8_selector_authority_candidate.py': 'ebea0b3f4f4e196931908f89eeb3c7189b79784cfb5eb53a2f8027a342773db4',
                     'test_hole_sim_real_app_whole_atom_selector_lowry_r1h8.py': 'bd3875f4de9e9082597a065f719e5b09bc053a5722e30f461f219e6dd1a1f4c9',
                     'test_hole_sim_tee_app_complete_atom_support.py': 'eb648d94447d8a60497dcf95b34dcbbe1a64c21cd5458bb27ad87c1673898397',
                     'test_hole_sim_tee_app_complete_atom_support_lowry_r1h8_runner.py': '61103a4f410fb2f2fcbdcec2ad20dc40c13c2c37a8975bae4587aa839b78153f',
                     'test_hole_sim_tee_app_h8_source_authority.py': '93966e84c85d7d3440516f0c80492e8c7d9145448aeaf88859a8cd833f115813',
                     'test_run_audit_hole_sim_real_app_h8_selector_authority_candidate_v2.py': '5e6db4fc2f767c1f3ab437f661f79b327f8dbf45739d24984f562fc6870730b6'},
  'scope': {'arm': 'raw',
            'category': 'app',
            'counts': {'complete_atoms': 35218,
                       'donor_quality_groups': 6,
                       'imputed_ability_atoms': 1989,
                       'mixed_policy_groups': 4,
                       'observed_ability_atoms': 33229,
                       'parts': 5,
                       'proposals': 904,
                       'resolved_hazard_atoms': 30702,
                       'spatial_proposals': 903,
                       'terminal_atoms': 1},
            'hole_no': 8,
            'query_key': '943f67faa5e345d395bf706e29c1a526e69879f80e14e81f55e255fd81dbf43e',
            'round_no': 1,
            'start_lie': 'tee',
            'support_key': 'app|raw|R1|H8|tee|B14|33204',
            'target_event_key': 'pga:R2026034',
            'target_player_id': '33204'},
  'semantic_evidence': {'external': {'full_schema_sha256': '113ef43b17f03ad9d9d7c9112d0636a87445ca9b97738935bec969f98b7b6a4d',
                                     'full_support_sha256': 'b66b8703d1b156aedbadded022a9ba1f1bdd821c7f8b585103fa4f8de747dce7',
                                     'policy_identity_sha256': 'f30e0161a1767d42f01dc0a33f617d93327286e3f49c59287c9b751686ba5bc7',
                                     'projection_54_sha256': 'a9957e4e20ed0259ce90a5abf27d29b7d4dcc8ee7259362e41c7ba7c9c178889',
                                     'projection_schema_sha256': '855c4b57abfe5ea0f29e23f1e3ffab82e58b186f09358bb1adcff17eb59c87f2',
                                     'putting_model_sha256': 'aace8326debb5c2af07b16e319730601a56d9ddd978d31ce4a0d3ee1d82f7c93',
                                     'reference_model_sha256': '35ffdc1aa21ccc209ceb56a8b8e78b3118aa1cc8bd4c7372a92a0dc01b0bc68c',
                                     'selector_native_sha256': 'f5fd780ea6c2a3e2795c8197f45f3486cd4a03606bac8e8c17b0548d7a07dc4c',
                                     'topology_30_sha256': '9e8e3a3128aa12ddae281ed0b37e036bee833232aa01ca9b93bd0370552af484'},
                        'ndjson': {'ability_policy': {'contract': 'lowry_r1h8_v2_ability_policy_ndjson_sha256',
                                                      'framed_bytes': 887469,
                                                      'records': 905,
                                                      'sha256': 'acda28fae2bdf86a038b369f5bfcb6f711ca26204acfc474ed22b8421fafba55'},
                                   'group': {'contract': 'lowry_r1h8_v2_group_ndjson_sha256',
                                             'framed_bytes': 5653,
                                             'records': 7,
                                             'sha256': 'cebf146bf965e293e2b5be8cb29895493c4cddba08530f8764f78b021aa33484'},
                                   'mass_cdf': {'contract': 'lowry_r1h8_v2_mass_cdf_ndjson_sha256',
                                                'framed_bytes': 13115551,
                                                'records': 905,
                                                'sha256': 'bcd7acafb2fed51320e96e0e9dbf2b30ca4e50a964d5ddaefa1c9428d5df19be'},
                                   'reference_reward': {'contract': 'lowry_r1h8_v2_reference_reward_ndjson_sha256',
                                                        'framed_bytes': 16878023,
                                                        'records': 35219,
                                                        'sha256': '45cf3527ee448cb2d008ea155ec6220d2a0a80391e970d5a48842d75f953e88c'},
                                   'schema': {'contract': 'lowry_r1h8_v2_schema_ndjson_sha256',
                                              'framed_bytes': 14888,
                                              'records': 151,
                                              'sha256': 'f711d7c44747222b7a5abde522996b8e4d341c694f0667acf936024d54cee944'},
                                   'semantic_root': {'contract': 'lowry_r1h8_v2_semantic_root_ndjson_sha256',
                                                     'framed_bytes': 3379,
                                                     'records': 1,
                                                     'sha256': 'b8312616d491bdce5e97fa438e95a8b9ead36ce1cb0fb9d7274605cfe3675dae'},
                                   'topology': {'contract': 'lowry_r1h8_v2_topology_ndjson_sha256',
                                                'framed_bytes': 2968153,
                                                'records': 905,
                                                'sha256': 'f20b53bf6cb5310685281f4519fbeeaa6ae6937122c13079c7a6175020924d97'}}},
  'support_artifact': {'bytes': 7647733,
                       'directories': ['complete_atom_support',
                                       'complete_atom_support/parts'],
                       'files': {'OUTPUT_SHA256.json': {'bytes': 1642,
                                                        'sha256': '5a502f524ada60bb91e85f1e18474056d4b2a883be14c0e1f7df3450393093b2'},
                                 'REPORT.md': {'bytes': 395,
                                               'sha256': 'c56b043fd88dae28f4c4af23fe4b0c6d5d8f4139115acb3a9ae495b700f62dc7'},
                                 'complete_atom_support/OUTPUT_SHA256.json': {'bytes': 974,
                                                                              'sha256': 'e300bf8b00066ec02e03128ca8e0e6b7a89c4e5a772a2ac5f2dd5794dbe2693f'},
                                 'complete_atom_support/manifest.json': {'bytes': 6495,
                                                                         'sha256': '4dec00946a085fedb489aa51fef672188d4205c6a26c85faeb7156e5234da7f2'},
                                 'complete_atom_support/parts/part-000000.parquet': {'bytes': 1751670,
                                                                                     'sha256': '3f3905bc2874e81baade5618a26b316e7a97da17dc3966766e22bfef602e38c7'},
                                 'complete_atom_support/parts/part-000001.parquet': {'bytes': 1767025,
                                                                                     'sha256': '4c1c349977834aeab644b1a0b37fe99512c66fbe3332495eaf871788f1131ed6'},
                                 'complete_atom_support/parts/part-000002.parquet': {'bytes': 1748952,
                                                                                     'sha256': 'bcac8a464d13dd52eaaa3a9bc3e135d9022843c3c52613f0feca491aa72bb306'},
                                 'complete_atom_support/parts/part-000003.parquet': {'bytes': 1762480,
                                                                                     'sha256': '9fd805ad604790e366f99f2435742866e2e72bf771a5a97bfbf6327d14ef33ca'},
                                 'complete_atom_support/parts/part-000004.parquet': {'bytes': 600833,
                                                                                     'sha256': 'ef93b0721901b106f20e6bb6518c2875c5224bb545f988b934fd254e5b23e0a2'},
                                 'manifest.json': {'bytes': 7267,
                                                   'sha256': 'c96a8e8333c8e535a82e9e265f74cde41d6a5a717e36c68c9dea9894e39f5464'}},
                       'identity': {'inner_manifest_sha256': '4dec00946a085fedb489aa51fef672188d4205c6a26c85faeb7156e5234da7f2',
                                    'inner_output_index_sha256': 'e300bf8b00066ec02e03128ca8e0e6b7a89c4e5a772a2ac5f2dd5794dbe2693f',
                                    'outer_manifest_sha256': 'c96a8e8333c8e535a82e9e265f74cde41d6a5a717e36c68c9dea9894e39f5464',
                                    'outer_output_index_sha256': '5a502f524ada60bb91e85f1e18474056d4b2a883be14c0e1f7df3450393093b2',
                                    'query_key': '943f67faa5e345d395bf706e29c1a526e69879f80e14e81f55e255fd81dbf43e',
                                    'selector_support_sha256': 'f5fd780ea6c2a3e2795c8197f45f3486cd4a03606bac8e8c17b0548d7a07dc4c',
                                    'source_authority_id': 'lowry_r1h8_raw_app_exact_query_v1',
                                    'source_authority_lock_sha256': 'edbadf6498b1fd5166c6711f0ac20e87bb7d691ec58f7f511f660cd38b978a4c',
                                    'source_authority_record_sha256': 'efaf969daea1ac9a7680b0fd8d534ffa700430523aa7cb13cec6fabaed9c347c',
                                    'support_key': 'app|raw|R1|H8|tee|B14|33204'},
                       'inventory_contract': 'audit_v2_input_tree',
                       'inventory_sha256': '0814ec477eae91319b0af70c51ab92935f539db51ed0481a11e4bf773e91b343',
                       'root': 'permanent_data/hole_sim_tee_app_complete_atom_support_lowry_r1h8_raw_app_v1'}},)
_AUTHORITY_LOCK_SHA256 = "6b3e75849a44c4e6b880af09f7bac86e609bffb5bf64401c9ddf89fe25c30da5"
# === END LOWRY_R1H8_AUTHORITY_DATA_V1 ===

_AUTHORITY_RECORDS = tuple(
    _freeze_tree(record) for record in _AUTHORITY_RECORD_LITERALS
)
del _AUTHORITY_RECORD_LITERALS


def _authority_payload(records: object) -> dict[str, object]:
    return {
        "contract": REAL_APP_H8_AUTHORITY_CONTRACT,
        "records": _plain_tree(records),
    }


def _require_keys(value: object, expected: set[str], label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != expected:
        raise RealAppH8AdmissionUnavailable(f"{label} has the wrong exact fields")
    return value


def _require_sha(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise RealAppH8AdmissionUnavailable(f"{label} must be a lowercase SHA-256")
    return value


def _exact_tree_equal(observed: object, expected: object) -> bool:
    """Compare JSON-shaped evidence without Python's bool/int aliasing."""

    if type(observed) is not type(expected):
        return False
    if type(expected) is dict:
        if set(observed) != set(expected):
            return False
        return all(
            type(key) is str
            and _exact_tree_equal(observed[key], expected[key])
            for key in expected
        )
    if type(expected) in (list, tuple):
        return len(observed) == len(expected) and all(
            _exact_tree_equal(left, right)
            for left, right in zip(observed, expected, strict=True)
        )
    if expected is None:
        return True
    if type(expected) is str:
        return str.__eq__(observed, expected)
    if type(expected) is int:
        return int.__eq__(observed, expected)
    if type(expected) is bool:
        return observed is expected
    return False


def _require_exact_tree(observed: object, expected: object, label: str) -> None:
    if not _exact_tree_equal(observed, expected):
        raise RealAppH8AdmissionUnavailable(f"{label} changed")


def _record_id(body: dict[str, object]) -> str:
    return _stable_sha(
        {
            "role": "manually_reviewed_data_only_real_h8_authority_v1",
            "payload": body,
        }
    )


def _validate_file_map(
    value: object,
    expected_paths: tuple[str, ...],
    label: str,
) -> dict[str, object]:
    mapping = _require_keys(value, set(expected_paths), label)
    for path, metadata in mapping.items():
        row = _require_keys(metadata, {"bytes", "sha256"}, f"{label} {path}")
        if type(row["bytes"]) is not int or row["bytes"] <= 0:
            raise RealAppH8AdmissionUnavailable(f"{label} {path} bytes changed")
        _require_sha(row["sha256"], f"{label} {path} digest")
    return mapping


def _validate_artifact_section(
    value: object,
    *,
    root: str,
    files: tuple[str, ...],
    directories: tuple[str, ...],
    identity: _Mapping[str, object],
    label: str,
) -> dict[str, object]:
    section = _require_keys(
        value,
        {
            "root",
            "files",
            "directories",
            "bytes",
            "inventory_contract",
            "inventory_sha256",
            "identity",
        },
        label,
    )
    if type(section["root"]) is not str or section["root"] != root:
        raise RealAppH8AdmissionUnavailable(f"{label} root changed")
    _require_exact_tree(section["directories"], list(directories), f"{label} directories")
    if type(section["bytes"]) is not int or section["bytes"] <= 0:
        raise RealAppH8AdmissionUnavailable(f"{label} byte total changed")
    expected_contract = (
        "audit_v2_output_tree" if label == "audit artifact" else "audit_v2_input_tree"
    )
    if section["inventory_contract"] != expected_contract:
        raise RealAppH8AdmissionUnavailable(f"{label} inventory contract changed")
    _require_sha(section["inventory_sha256"], f"{label} inventory")
    _validate_file_map(section["files"], files, f"{label} files")
    _require_exact_tree(section["identity"], dict(identity), f"{label} identity")
    return section


def _validate_record(value: object) -> dict[str, object]:
    record = _require_keys(
        value,
        {
            "record_contract",
            "authority_id",
            "scope",
            "logic",
            "support_artifact",
            "candidate_artifact",
            "audit_artifact",
            "review_sources",
            "semantic_evidence",
            "limits",
            "capabilities",
        },
        "authority record",
    )
    if record["record_contract"] != REAL_APP_H8_RECORD_CONTRACT:
        raise RealAppH8AdmissionUnavailable("Authority record contract changed")
    authority_id = _require_sha(record["authority_id"], "authority id")
    if authority_id in {
        "b5552aaea4e7c216b8bae8dc9ccdb68d122df54523d7b8faf1ddb70664dde942",
        "f6b2392ceda66ac7c32e360dc63910fe2b1d8844dc9ed8da2d12d68276996cf0",
    }:
        raise RealAppH8AdmissionUnavailable("Candidate/audit identity is not authority")
    body = dict(record)
    body.pop("authority_id")
    if authority_id != _record_id(body):
        raise RealAppH8AdmissionUnavailable("Authority id is not domain-separated")

    expected_scope = {
        "query_key": _QUERY_KEY,
        "support_key": _SUPPORT_KEY,
        "category": "app",
        "arm": "raw",
        "round_no": 1,
        "hole_no": 8,
        "start_lie": "tee",
        "target_player_id": "33204",
        "target_event_key": "pga:R2026034",
        "counts": dict(_COUNTS),
    }
    _require_exact_tree(record["scope"], expected_scope, "Authority scope")
    _require_exact_tree(record["limits"], dict(_LIMITS), "Authority limits")
    _require_exact_tree(
        record["capabilities"], dict(_CAPABILITIES), "Authority capabilities"
    )

    logic = _require_keys(
        record["logic"],
        {
            "reviewed_dormant_wrapper_sha256",
            "reviewed_dormant_test_sha256",
            "reviewed_record_test_sha256",
            "selector_capsule_sha256",
            "selector_source_sha256",
            "selector_test_sha256",
            "selector_runtime_sha256",
            "selector_support_sha256",
        },
        "authority logic",
    )
    for field in logic:
        _require_sha(logic[field], f"authority logic {field}")
    if logic["selector_source_sha256"] != _SELECTOR_SHA256:
        raise RealAppH8AdmissionUnavailable("Reviewed selector source changed")
    if logic["selector_test_sha256"] != _SELECTOR_TEST_SHA256:
        raise RealAppH8AdmissionUnavailable("Reviewed selector test changed")
    if logic["selector_runtime_sha256"] != _SELECTOR_RUNTIME_SHA256:
        raise RealAppH8AdmissionUnavailable("Reviewed selector runtime changed")
    if logic["selector_support_sha256"] != _SELECTOR_SUPPORT_SHA256:
        raise RealAppH8AdmissionUnavailable("Reviewed selector support changed")
    if logic["selector_capsule_sha256"] != _SELECTOR_CAPSULE_SHA256:
        raise RealAppH8AdmissionUnavailable("Reviewed selector capsule changed")

    _validate_artifact_section(
        record["support_artifact"],
        root=_SUPPORT_RELATIVE,
        files=_SUPPORT_FILES,
        directories=_SUPPORT_DIRECTORIES,
        identity=_SUPPORT_IDENTITY,
        label="support artifact",
    )
    _validate_artifact_section(
        record["candidate_artifact"],
        root=_CANDIDATE_RELATIVE,
        files=_CANDIDATE_FILES,
        directories=(),
        identity=_CANDIDATE_IDENTITY,
        label="candidate artifact",
    )
    _validate_artifact_section(
        record["audit_artifact"],
        root=_AUDIT_RELATIVE,
        files=_AUDIT_FILES,
        directories=(),
        identity=_AUDIT_IDENTITY,
        label="audit artifact",
    )

    review_sources = _require_keys(
        record["review_sources"],
        {*_KNOWN_SOURCE_SHA256, _TEST_RELATIVE, _RECORD_TEST_RELATIVE},
        "review sources",
    )
    for path, digest in _KNOWN_SOURCE_SHA256.items():
        if review_sources[path] != digest:
            raise RealAppH8AdmissionUnavailable(f"Reviewed source changed: {path}")
    dormant_test_sha = _require_sha(
        review_sources[_TEST_RELATIVE], "dormant wrapper test digest"
    )
    record_test_sha = _require_sha(
        review_sources[_RECORD_TEST_RELATIVE], "authority-record test digest"
    )
    if not str.__eq__(dormant_test_sha, logic["reviewed_dormant_test_sha256"]):
        raise RealAppH8AdmissionUnavailable("Dormant wrapper test binding changed")
    if not str.__eq__(record_test_sha, logic["reviewed_record_test_sha256"]):
        raise RealAppH8AdmissionUnavailable("Authority-record test binding changed")

    semantic = _require_keys(
        record["semantic_evidence"], {"ndjson", "external"}, "semantic evidence"
    )
    ndjson = _require_keys(semantic["ndjson"], set(_NDJSON), "NDJSON evidence")
    for name, expected in _NDJSON.items():
        row = _require_keys(
            ndjson[name], {"contract", "records", "framed_bytes", "sha256"}, name
        )
        observed = [
            row["contract"], row["records"], row["framed_bytes"], row["sha256"]
        ]
        _require_exact_tree(observed, list(expected), f"NDJSON evidence {name}")
    external = _require_keys(
        semantic["external"],
        set(_EXTERNAL_EVIDENCE),
        "external semantic evidence",
    )
    for field, digest in _EXTERNAL_EVIDENCE.items():
        if external[field] != digest:
            raise RealAppH8AdmissionUnavailable(
                f"External semantic evidence changed: {field}"
            )
        _require_sha(digest, f"external evidence {field}")
    return record


def _make_authority_gate(
    records: tuple[object, ...], lock_sha256: str
):
    if type(records) is not tuple or len(records) > 1:
        raise RealAppH8AdmissionUnavailable("Authority registry cardinality changed")
    if not _is_sha256(lock_sha256):
        raise RealAppH8AdmissionUnavailable("Authority lock is malformed")
    captured_records = records
    captured_lock = lock_sha256

    def closed_authority() -> tuple[dict[str, object] | None, str]:
        if type(captured_records) is not tuple or len(captured_records) > 1:
            raise RealAppH8AdmissionUnavailable("Authority registry cardinality changed")
        observed_lock = _stable_sha(_authority_payload(captured_records))
        if not str.__eq__(observed_lock, captured_lock):
            raise RealAppH8AdmissionUnavailable("Authority registry lock changed")
        if not captured_records:
            if not str.__eq__(captured_lock, _EMPTY_AUTHORITY_LOCK):
                raise RealAppH8AdmissionUnavailable("Empty authority lock changed")
            return None, captured_lock
        record = _plain_tree(captured_records[0])
        return _validate_record(record), captured_lock

    return closed_authority


_closed_authority = _make_authority_gate(
    _AUTHORITY_RECORDS, _AUTHORITY_LOCK_SHA256
)
del _make_authority_gate, _AUTHORITY_RECORDS, _AUTHORITY_LOCK_SHA256


def _strict_json(raw: bytes, label: str) -> object:
    def pairs(rows: list[tuple[str, object]]) -> dict[str, object]:
        out: dict[str, object] = {}
        for key, value in rows:
            if type(key) is not str or key in out:
                raise RealAppH8AdmissionUnavailable(f"{label} has duplicate keys")
            out[key] = value
        return out

    def constant(_: str) -> _NoReturn:
        raise RealAppH8AdmissionUnavailable(f"{label} contains a nonfinite number")

    try:
        return _json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeDecodeError, _json.JSONDecodeError) as exc:
        raise RealAppH8AdmissionUnavailable(f"{label} is not strict JSON") from exc


def _sha256_file(path: str) -> str:
    return _sha256(
        _FILESYSTEM_READ_CAPPED(path, _MAX_REVIEW_SOURCE_BYTES, "review source")
    ).hexdigest()


def _project_dormant_source(raw: bytes) -> bytes:
    if raw.count(_BLOCK_BEGIN) != 1 or raw.count(_BLOCK_END) != 1:
        raise RealAppH8AdmissionUnavailable("Authority block markers changed")
    start = raw.index(_BLOCK_BEGIN)
    end_marker = raw.index(_BLOCK_END)
    if end_marker <= start:
        raise RealAppH8AdmissionUnavailable("Authority block marker order changed")
    end = raw.find(b"\n", end_marker + len(_BLOCK_END))
    if end < 0:
        raise RealAppH8AdmissionUnavailable("Authority block lacks its final newline")
    end += 1
    block = raw[start:end]
    lines = block.splitlines()
    if lines[0] != _BLOCK_BEGIN or lines[-1] != _BLOCK_END:
        raise RealAppH8AdmissionUnavailable("Authority marker lines changed")
    try:
        tree = _ast.parse(b"\n".join(lines[1:-1]).decode("ascii"), mode="exec")
    except (SyntaxError, UnicodeDecodeError) as exc:
        raise RealAppH8AdmissionUnavailable("Authority block is not literal Python") from exc
    if len(tree.body) != 2 or not all(type(node) is _ast.Assign for node in tree.body):
        raise RealAppH8AdmissionUnavailable("Authority block gained executable code")
    expected_names = ("_AUTHORITY_RECORD_LITERALS", "_AUTHORITY_LOCK_SHA256")
    for node, name in zip(tree.body, expected_names, strict=True):
        if (
            len(node.targets) != 1
            or type(node.targets[0]) is not _ast.Name
            or node.targets[0].id != name
        ):
            raise RealAppH8AdmissionUnavailable("Authority block assignments changed")
        try:
            _ast.literal_eval(node.value)
        except (TypeError, ValueError) as exc:
            raise RealAppH8AdmissionUnavailable("Authority block is not literal-only") from exc
    return raw[:start] + _DORMANT_BLOCK + raw[end:]


def _source_state(record: dict[str, object]) -> dict[str, object]:
    capsule_start = _selector_capsule_state()
    filesystem_start = _FILESYSTEM_STATE()
    if type(__file__) is not str or not str.__eq__(__file__, _SELF):
        raise RealAppH8AdmissionUnavailable("Admission wrapper path changed")
    raw = _read_capped_file(_SELF, _MAX_WRAPPER_SOURCE_BYTES, "admission wrapper")
    projected = _project_dormant_source(raw)
    logic = record["logic"]
    if type(logic) is not dict:
        raise RealAppH8AdmissionUnavailable("Authority logic type changed")
    if _sha256(projected).hexdigest() != logic["reviewed_dormant_wrapper_sha256"]:
        raise RealAppH8AdmissionUnavailable("Dormant wrapper projection changed")
    dormant_test_sha = _closed_file_sha(_TEST, "dormant wrapper test")
    record_test_sha = _closed_file_sha(_RECORD_TEST, "authority-record test")
    if not str.__eq__(dormant_test_sha, logic["reviewed_dormant_test_sha256"]):
        raise RealAppH8AdmissionUnavailable("Dormant wrapper test changed")
    if not str.__eq__(record_test_sha, logic["reviewed_record_test_sha256"]):
        raise RealAppH8AdmissionUnavailable("Authority-record test changed")
    review_sources = record["review_sources"]
    if type(review_sources) is not dict:
        raise RealAppH8AdmissionUnavailable("Review-source map type changed")
    observed: dict[str, str] = {}
    for relative, expected in _KNOWN_SOURCE_SHA256.items():
        digest = _closed_file_sha(
            _KNOWN_SOURCE_PATHS[relative], f"frozen source {relative}"
        )
        observed[relative] = digest
        if digest != expected or digest != review_sources[relative]:
            raise RealAppH8AdmissionUnavailable(f"Frozen source changed: {relative}")
    if not str.__eq__(dormant_test_sha, review_sources[_TEST_RELATIVE]):
        raise RealAppH8AdmissionUnavailable("Dormant test review binding changed")
    if not str.__eq__(record_test_sha, review_sources[_RECORD_TEST_RELATIVE]):
        raise RealAppH8AdmissionUnavailable("Authority-record test review binding changed")
    observed[_TEST_RELATIVE] = dormant_test_sha
    observed[_RECORD_TEST_RELATIVE] = record_test_sha
    if (
        capsule_start["capsule_sha256"] != _SELECTOR_CAPSULE_SHA256
        or capsule_start["capsule_sha256"] != logic["selector_capsule_sha256"]
    ):
        raise RealAppH8AdmissionUnavailable("Frozen selector capsule binding changed")
    runtime = _SELECTOR_VALIDATE_RUNTIME()
    if runtime != _SELECTOR_RUNTIME_SHA256:
        raise RealAppH8AdmissionUnavailable("Frozen selector runtime changed")
    capsule_end = _selector_capsule_state()
    if capsule_start != capsule_end:
        raise RealAppH8AdmissionUnavailable(
            "Frozen selector capsule changed during source validation"
        )
    if filesystem_start != _FILESYSTEM_STATE():
        raise RealAppH8AdmissionUnavailable(
            "Captured filesystem changed during source validation"
        )
    return {
        "wrapper_sha256": _sha256(raw).hexdigest(),
        "dormant_projection_sha256": _sha256(projected).hexdigest(),
        "test_sha256": dormant_test_sha,
        "record_test_sha256": record_test_sha,
        "selector_runtime_sha256": runtime,
        "selector_capsule_sha256": capsule_end["capsule_sha256"],
        "sources_sha256": _stable_sha(observed),
    }


def _is_reparse(stat_result: _os.stat_result) -> bool:
    return _FILESYSTEM_IS_REPARSE(stat_result)


def _closed_file_sha(path: str, label: str) -> str:
    raw = _read_capped_file(path, _MAX_REVIEW_SOURCE_BYTES, label)
    return _sha256(raw).hexdigest()


def _read_capped_file(path: str, maximum_bytes: int, label: str) -> bytes:
    return _FILESYSTEM_READ_CAPPED(path, maximum_bytes, label)


def _read_expected_file(path: str, expected_bytes: int, label: str) -> bytes:
    return _FILESYSTEM_READ_EXPECTED(path, expected_bytes, label)


def _closed_source_bytes(path: str, expected_bytes: int, label: str) -> bytes:
    return _read_expected_file(path, expected_bytes, label)


def _integrity_payload(value: object) -> object:
    """Return a bounded, non-polymorphic description of mutable trusted state."""

    remaining = [100_000]
    active: set[int] = set()

    def visit(item: object) -> object:
        remaining[0] -= 1
        if remaining[0] < 0:
            raise RealAppH8AdmissionUnavailable(
                "Frozen selector integrity state exceeded its fixed budget"
            )
        item_type = type(item)
        if item is None:
            return ["none"]
        if item_type is bool:
            return ["bool", item]
        if item_type is int:
            return ["int", str(item)]
        if item_type is float:
            return ["float", item.hex()]
        if item_type is str:
            return ["str", item]
        if item_type is bytes:
            return ["bytes", item.hex()]
        if item_type is _DataclassField:
            return [
                "dataclass_field",
                id(item),
                [
                    [slot, visit(getattr(item, slot))]
                    for slot in _DataclassField.__slots__
                ],
            ]
        if item_type in (tuple, list, dict, set, frozenset, _MappingProxyType):
            identity = id(item)
            if identity in active:
                return ["cycle", identity]
            active.add(identity)
            try:
                if item_type in (tuple, list):
                    return [
                        "tuple" if item_type is tuple else "list",
                        [visit(child) for child in item],
                    ]
                if item_type in (set, frozenset):
                    children = [visit(child) for child in item]
                    children.sort(key=lambda child: _json_bytes(child))
                    return [
                        "frozenset" if item_type is frozenset else "set",
                        children,
                    ]
                pairs = [[visit(key), visit(child)] for key, child in item.items()]
                return [
                    "mappingproxy" if item_type is _MappingProxyType else "dict",
                    pairs,
                ]
            finally:
                active.remove(identity)
        return ["identity", id(item_type), id(item)]

    return visit(value)


def _integrity_digest(value: object) -> str:
    return _stable_sha(_integrity_payload(value))


def _semantic_object_digest(value: object) -> str | None:
    """Bind mutable state carried by runtime-semantic non-container aliases."""

    if type(value) in (
        type(None),
        bool,
        int,
        float,
        str,
        bytes,
        tuple,
        list,
        dict,
        set,
        frozenset,
        _MappingProxyType,
        _ModuleType,
        _FunctionType,
        type,
    ):
        return None
    try:
        state = object.__getattribute__(value, "__dict__")
    except (AttributeError, TypeError):
        return None
    if type(state) is not dict:
        return None
    if len(state) > 10_000:
        raise RealAppH8AdmissionUnavailable(
            "Frozen selector semantic object exceeded its fixed state budget"
        )
    return _integrity_digest(state)


def _capture_function_integrity(value: object, label: str) -> _MappingProxyType:
    if type(value) is not _FunctionType:
        raise RealAppH8AdmissionUnavailable(
            f"Frozen selector function type changed: {label}"
        )
    closure = value.__closure__
    closure_rows: list[tuple[object, object, str]] = []
    if closure is not None:
        if type(closure) is not tuple:
            raise RealAppH8AdmissionUnavailable(
                f"Frozen selector function closure changed: {label}"
            )
        for cell in closure:
            try:
                contents = cell.cell_contents
            except ValueError:
                contents = _EMPTY_CELL
            closure_rows.append((cell, contents, _integrity_digest(contents)))
    return _MappingProxyType(
        {
            "label": label,
            "function": value,
            "code": value.__code__,
            "name": value.__name__,
            "qualname": value.__qualname__,
            "module": value.__module__,
            "doc": value.__doc__,
            "globals": value.__globals__,
            "builtins": value.__builtins__,
            "defaults": value.__defaults__,
            "defaults_sha256": _integrity_digest(value.__defaults__),
            "kwdefaults": value.__kwdefaults__,
            "kwdefaults_sha256": _integrity_digest(value.__kwdefaults__),
            "annotations": value.__annotations__,
            "annotations_sha256": _integrity_digest(value.__annotations__),
            "attributes": value.__dict__,
            "attributes_sha256": _integrity_digest(value.__dict__),
            "closure": closure,
            "closure_rows": tuple(closure_rows),
        }
    )


def _check_function_integrity(guard: _Mapping[str, object]) -> None:
    value = guard["function"]
    label = guard["label"]
    if (
        type(value) is not _FunctionType
        or value.__code__ is not guard["code"]
        or type(value.__name__) is not str
        or not str.__eq__(value.__name__, guard["name"])
        or type(value.__qualname__) is not str
        or not str.__eq__(value.__qualname__, guard["qualname"])
        or type(value.__module__) is not str
        or not str.__eq__(value.__module__, guard["module"])
        or value.__doc__ is not guard["doc"]
        or value.__globals__ is not guard["globals"]
        or value.__builtins__ is not guard["builtins"]
        or value.__defaults__ is not guard["defaults"]
        or value.__kwdefaults__ is not guard["kwdefaults"]
        or value.__annotations__ is not guard["annotations"]
        or value.__dict__ is not guard["attributes"]
        or value.__closure__ is not guard["closure"]
        or not str.__eq__(
            _integrity_digest(value.__defaults__), guard["defaults_sha256"]
        )
        or not str.__eq__(
            _integrity_digest(value.__kwdefaults__), guard["kwdefaults_sha256"]
        )
        or not str.__eq__(
            _integrity_digest(value.__annotations__), guard["annotations_sha256"]
        )
        or not str.__eq__(
            _integrity_digest(value.__dict__), guard["attributes_sha256"]
        )
    ):
        raise RealAppH8AdmissionUnavailable(
            f"Frozen selector function implementation changed: {label}"
        )
    observed_closure = value.__closure__ or ()
    expected_closure = guard["closure_rows"]
    if len(observed_closure) != len(expected_closure):
        raise RealAppH8AdmissionUnavailable(
            f"Frozen selector function closure changed: {label}"
        )
    for cell, expected in zip(observed_closure, expected_closure, strict=True):
        expected_cell, expected_contents, expected_sha = expected
        try:
            contents = cell.cell_contents
        except ValueError:
            contents = _EMPTY_CELL
        if (
            cell is not expected_cell
            or contents is not expected_contents
            or not str.__eq__(_integrity_digest(contents), expected_sha)
        ):
            raise RealAppH8AdmissionUnavailable(
                f"Frozen selector function closure changed: {label}"
            )


def _class_function(value: object) -> _FunctionType | None:
    if type(value) is _FunctionType:
        return value
    if type(value) in (staticmethod, classmethod):
        return value.__func__
    return None


def _capture_class_integrity(value: type, label: str) -> _MappingProxyType:
    namespace = vars(value)
    if any(type(key) is not str for key in namespace):
        raise RealAppH8AdmissionUnavailable(
            f"Frozen selector class namespace changed: {label}"
        )
    items = {key: child for key, child in namespace.items()}
    mutable = {
        key: _integrity_digest(child)
        for key, child in items.items()
        if type(child) in (dict, list, set, tuple, frozenset, _MappingProxyType)
    }
    semantic = {
        key: digest
        for key, child in items.items()
        if (digest := _semantic_object_digest(child)) is not None
    }
    functions: list[_MappingProxyType] = []
    for key, child in items.items():
        function = _class_function(child)
        if function is not None:
            functions.append(_capture_function_integrity(function, f"{label}.{key}"))
        elif type(child) is property:
            for suffix, function in (
                ("fget", child.fget),
                ("fset", child.fset),
                ("fdel", child.fdel),
            ):
                if type(function) is _FunctionType:
                    functions.append(
                        _capture_function_integrity(
                            function, f"{label}.{key}.{suffix}"
                        )
                    )
    return _MappingProxyType(
        {
            "label": label,
            "class": value,
            "metaclass": type(value),
            "bases": tuple(value.__bases__),
            "mro": tuple(value.__mro__),
            "items": _MappingProxyType(items),
            "mutable": _MappingProxyType(mutable),
            "semantic": _MappingProxyType(semantic),
            "functions": tuple(functions),
        }
    )


def _check_class_integrity(guard: _Mapping[str, object]) -> None:
    value = guard["class"]
    label = guard["label"]
    namespace = vars(value)
    expected_items = guard["items"]
    if (
        type(value) is not guard["metaclass"]
        or type(value.__bases__) is not tuple
        or len(value.__bases__) != len(guard["bases"])
        or any(
            observed is not expected
            for observed, expected in zip(
                value.__bases__, guard["bases"], strict=True
            )
        )
        or type(value.__mro__) is not tuple
        or len(value.__mro__) != len(guard["mro"])
        or any(
            observed is not expected
            for observed, expected in zip(value.__mro__, guard["mro"], strict=True)
        )
        or any(type(key) is not str for key in namespace)
        or set(namespace) != set(expected_items)
        or any(namespace[key] is not child for key, child in expected_items.items())
    ):
        raise RealAppH8AdmissionUnavailable(
            f"Frozen selector class namespace changed: {label}"
        )
    for key, expected_sha in guard["mutable"].items():
        if not str.__eq__(_integrity_digest(namespace[key]), expected_sha):
            raise RealAppH8AdmissionUnavailable(
                f"Frozen selector class constant changed: {label}.{key}"
            )
    for key, expected_sha in guard["semantic"].items():
        observed_sha = _semantic_object_digest(namespace[key])
        if observed_sha is None or not str.__eq__(observed_sha, expected_sha):
            raise RealAppH8AdmissionUnavailable(
                f"Frozen selector class semantic object changed: {label}.{key}"
            )
    for function in guard["functions"]:
        _check_function_integrity(function)


def _capture_module_integrity(
    role: str,
    module: _ModuleType,
    builtins_map: dict[str, object],
) -> _MappingProxyType:
    namespace = module.__dict__
    namespace_keys = tuple(namespace)
    if any(type(key) is not str for key in namespace_keys):
        raise RealAppH8AdmissionUnavailable(
            f"Frozen selector module namespace changed: {role}"
        )
    missing_warning_registry = object()
    warning_registry = namespace.get(
        "__warningregistry__", missing_warning_registry
    )
    if (
        warning_registry is not missing_warning_registry
        and type(warning_registry) is not dict
    ):
        raise RealAppH8AdmissionUnavailable(
            f"Frozen selector module warning registry changed: {role}"
        )
    keys = tuple(
        key
        for key in namespace_keys
        if not str.__eq__(key, "__warningregistry__")
    )
    items = {key: namespace[key] for key in keys}
    mutable = {
        key: _integrity_digest(child)
        for key, child in items.items()
        if key != "__builtins__"
        and type(child) in (dict, list, set, tuple, frozenset, _MappingProxyType)
    }
    semantic = {
        key: digest
        for key, child in items.items()
        if (digest := _semantic_object_digest(child)) is not None
    }
    semantic_types: list[_MappingProxyType] = []
    seen_semantic_types: set[type] = set()
    for key in semantic:
        for semantic_type in type(items[key]).__mro__:
            if semantic_type is object or semantic_type in seen_semantic_types:
                continue
            seen_semantic_types.add(semantic_type)
            semantic_types.append(
                _capture_class_integrity(
                    semantic_type, f"{role}.{key}.runtime_type.{semantic_type.__name__}"
                )
            )
    functions = tuple(
        _capture_function_integrity(child, f"{role}.{key}")
        for key, child in items.items()
        if type(child) is _FunctionType and child.__globals__ is namespace
    )
    classes = tuple(
        _capture_class_integrity(child, f"{role}.{key}")
        for key, child in items.items()
        if isinstance(child, type) and child.__module__ == module.__name__
    )
    return _MappingProxyType(
        {
            "role": role,
            "module": module,
            "items": _MappingProxyType(items),
            "mutable": _MappingProxyType(mutable),
            "semantic": _MappingProxyType(semantic),
            "semantic_types": tuple(semantic_types),
            "functions": functions,
            "classes": classes,
            "builtins": builtins_map,
            "builtins_items": _MappingProxyType(dict(builtins_map)),
        }
    )


def _check_module_integrity(guard: _Mapping[str, object]) -> None:
    role = guard["role"]
    module = guard["module"]
    namespace = module.__dict__
    namespace_keys = tuple(namespace)
    expected_items = guard["items"]
    if any(type(key) is not str for key in namespace_keys):
        raise RealAppH8AdmissionUnavailable(
            f"Frozen selector module namespace changed: {role}"
        )
    missing_warning_registry = object()
    warning_registry = namespace.get(
        "__warningregistry__", missing_warning_registry
    )
    if (
        warning_registry is not missing_warning_registry
        and type(warning_registry) is not dict
    ):
        raise RealAppH8AdmissionUnavailable(
            f"Frozen selector module warning registry changed: {role}"
        )
    observed_keys = tuple(
        key
        for key in namespace_keys
        if not str.__eq__(key, "__warningregistry__")
    )
    if set(observed_keys) != set(expected_items) or any(
        namespace[key] is not child for key, child in expected_items.items()
    ):
        raise RealAppH8AdmissionUnavailable(
            f"Frozen selector module namespace changed: {role}"
        )
    for key, expected_sha in guard["mutable"].items():
        if not str.__eq__(_integrity_digest(namespace[key]), expected_sha):
            raise RealAppH8AdmissionUnavailable(
                f"Frozen selector module constant changed: {role}.{key}"
            )
    for key, expected_sha in guard["semantic"].items():
        observed_sha = _semantic_object_digest(namespace[key])
        if observed_sha is None or not str.__eq__(observed_sha, expected_sha):
            raise RealAppH8AdmissionUnavailable(
                f"Frozen selector module semantic object changed: {role}.{key}"
            )
    for semantic_type in guard["semantic_types"]:
        _check_class_integrity(semantic_type)
    builtins_map = guard["builtins"]
    expected_builtins = guard["builtins_items"]
    if (
        namespace.get("__builtins__") is not builtins_map
        or type(builtins_map) is not dict
        or any(type(key) is not str for key in builtins_map)
        or set(builtins_map) != set(expected_builtins)
        or any(
            builtins_map[key] is not child
            for key, child in expected_builtins.items()
        )
    ):
        raise RealAppH8AdmissionUnavailable(
            f"Frozen selector builtins changed: {role}"
        )
    for function in guard["functions"]:
        _check_function_integrity(function)
    for value in guard["classes"]:
        _check_class_integrity(value)


def _load_frozen_selector_capsule() -> _MappingProxyType:
    real_import = _builtins.__import__
    sources: dict[str, bytes] = {}
    source_sha256: dict[str, str] = {}
    for role, _canonical, relative in _FROZEN_SELECTOR_MODULE_SPECS:
        expected = _KNOWN_SOURCE_SHA256[relative]
        raw = _closed_source_bytes(
            _KNOWN_SOURCE_PATHS[relative],
            _FROZEN_SELECTOR_SOURCE_BYTES[relative],
            f"frozen {role} source",
        )
        digest = _sha256(raw).hexdigest()
        if digest != expected:
            raise RealAppH8AdmissionUnavailable(
                f"Frozen selector-capsule source changed: {relative}"
            )
        sources[role] = raw
        source_sha256[relative] = digest

    modules: dict[str, _ModuleType] = {}
    private_names: dict[str, str] = {}
    nonces: list[object] = []
    for role, canonical, relative in _FROZEN_SELECTOR_MODULE_SPECS:
        nonce = object()
        nonces.append(nonce)
        expected = source_sha256[relative]
        private_name = f"_lowry_r1h8_frozen_{role}_{expected[:12]}_{id(nonce):x}"
        if private_name in _sys.modules:
            raise RealAppH8AdmissionUnavailable(
                f"Private selector-capsule module slot is occupied: {role}"
            )
        module = _ModuleType(private_name)
        module.__file__ = _KNOWN_SOURCE_PATHS[relative]
        module.__package__ = ""
        module.__loader__ = None
        module.__spec__ = None
        modules[canonical] = module
        private_names[canonical] = private_name

    frozen_modules = _MappingProxyType(dict(modules))

    def guarded_import(
        name: str,
        globals_: object = None,
        locals_: object = None,
        fromlist: object = (),
        level: int = 0,
    ) -> object:
        if type(name) is not str or level != 0:
            raise ImportError("Frozen selector capsule permits absolute imports only")
        if name in frozen_modules:
            return frozen_modules[name]
        if name.startswith("hole_sim_"):
            raise ImportError(f"Unreviewed local selector dependency: {name}")
        if name == "numpy":
            return _np
        if name == "pandas":
            return _pd
        return real_import(name, globals_, locals_, fromlist, level)

    builtins_map = dict(vars(_builtins))
    builtins_map["__import__"] = guarded_import
    _imp.acquire_lock()
    try:
        for role, canonical, relative in _FROZEN_SELECTOR_MODULE_SPECS:
            module = modules[canonical]
            private_name = private_names[canonical]
            if private_name in _sys.modules:
                raise RealAppH8AdmissionUnavailable(
                    f"Private selector-capsule module slot became occupied: {role}"
                )
            module.__dict__["__builtins__"] = builtins_map
            _sys.modules[private_name] = module
            slot_changed = False
            try:
                code = compile(
                    sources[role],
                    _KNOWN_SOURCE_PATHS[relative],
                    "exec",
                    dont_inherit=True,
                    optimize=0,
                )
                exec(code, module.__dict__, module.__dict__)
            finally:
                if _sys.modules.get(private_name) is module:
                    del _sys.modules[private_name]
                else:
                    slot_changed = True
            if slot_changed:
                raise RealAppH8AdmissionUnavailable(
                    f"Private selector-capsule module slot changed: {role}"
                )
            if (
                type(module) is not _ModuleType
                or module.__name__ != private_name
                or module.__file__ != _KNOWN_SOURCE_PATHS[relative]
            ):
                raise RealAppH8AdmissionUnavailable(
                    f"Frozen selector-capsule module identity changed: {role}"
                )
    except RealAppH8AdmissionUnavailable:
        raise
    except Exception as exc:
        raise RealAppH8AdmissionUnavailable(
            f"Frozen selector capsule could not be executed: {exc}"
        ) from exc
    finally:
        _imp.release_lock()

    by_role = {
        role: modules[canonical]
        for role, canonical, _relative in _FROZEN_SELECTOR_MODULE_SPECS
    }
    selector = by_role["selector"]
    runtime_source_paths = _MappingProxyType(
        {
            descriptor: _KNOWN_SOURCE_PATHS[descriptor]
            for _role, descriptor, _digest in selector._SOURCE_BINDINGS
        }
    )
    bounded_cap = _MAX_REVIEW_SOURCE_BYTES
    bounded_hash = _sha256
    bounded_read = _FILESYSTEM_READ_CAPPED
    runtime_filesystem_state = _FILESYSTEM_STATE
    bounded_error = RealAppH8AdmissionUnavailable
    runtime_stable_sha = _stable_sha
    exact_int = int
    exact_type = type
    exact_str = str
    exact_sorted = sorted
    expected_source_bindings = selector._SOURCE_BINDINGS
    expected_anchor_contract = "whole_atom_donor_detilt_target_reapply_lambda_v1"
    expected_policy_contract = "frozen_strict_prior_donor_ability_policy_v2"
    expected_boundary_contract = (
        "holeout_strict_terminal_unconditional_prefix_proposal_right_surface_"
        "strict_hazard_right_binary64_pathwise_v2"
    )

    def bounded_selector_source_hash(descriptor: object) -> str:
        if exact_type(descriptor) is not exact_str:
            raise bounded_error("Frozen selector runtime source descriptor changed")
        absolute_path = runtime_source_paths.get(descriptor)
        if absolute_path is None:
            raise bounded_error("Frozen selector runtime source descriptor changed")
        raw = bounded_read(
            absolute_path,
            bounded_cap,
            f"frozen selector runtime source {descriptor}",
        )
        return bounded_hash(raw).hexdigest()

    anchor = by_role["anchor"]
    policy = by_role["policy"]
    synthetic = by_role["synthetic"]
    chain = by_role["chain"]
    expected_group_function = chain.canonical_donor_quality_group_id
    runtime_payload_function = selector._runtime_payload

    def capsule_validate_runtime() -> str:
        filesystem_start = runtime_filesystem_state()
        if anchor.ATOM_WEIGHT_CONTRACT != expected_anchor_contract:
            raise bounded_error("Whole-atom weighting contract changed")
        if policy.POLICY_CONTRACT != expected_policy_contract:
            raise bounded_error("Donor-ability policy contract changed")
        if synthetic.APP_SELECTION_BOUNDARY_CONTRACT != expected_boundary_contract:
            raise bounded_error("Reviewed APP boundary reference changed")
        if (
            chain.canonical_donor_quality_group_id is not expected_group_function
            or expected_group_function.__name__
            != "canonical_donor_quality_group_id"
        ):
            raise bounded_error("Canonical group derivation changed")
        if selector._SOURCE_BINDINGS is not expected_source_bindings:
            raise bounded_error("Frozen selector source bindings changed")
        observed = {
            descriptor: bounded_selector_source_hash(descriptor)
            for _role, descriptor, _digest in expected_source_bindings
        }
        expected = {
            descriptor: digest
            for _role, descriptor, digest in expected_source_bindings
        }
        if observed != expected:
            changed = exact_sorted(
                descriptor
                for descriptor, digest in expected.items()
                if observed.get(descriptor) != digest
            )
            raise bounded_error(f"Frozen real APP sources changed: {changed}")
        if selector._runtime_payload is not runtime_payload_function:
            raise bounded_error("Frozen selector runtime payload changed")
        result = runtime_stable_sha(runtime_payload_function())
        if filesystem_start != runtime_filesystem_state():
            raise bounded_error(
                "Captured filesystem changed during selector runtime validation"
            )
        return result

    selector.__dict__["_sha256_file"] = bounded_selector_source_hash
    selector.__dict__["_validate_runtime"] = capsule_validate_runtime
    functions: dict[str, _FunctionType] = {}
    for name in (
        "_canonicalize_fixture",
        "_support_sha",
        "_weight_fixture",
        "_select_fixture",
        "_validate_runtime",
    ):
        value = selector.__dict__.get(name)
        if type(value) is not _FunctionType or (
            name != "_validate_runtime" and value.__globals__ is not selector.__dict__
        ):
            raise RealAppH8AdmissionUnavailable(
                f"Frozen selector function provenance changed: {name}"
            )
        functions[name] = value

    selector_types: dict[str, type] = {}
    for name in (
        "RealAppWeightingParameters",
        "RealAppSemanticUniforms",
        "RealAppWeightedSupport",
        "RealAppSelectionResult",
    ):
        value = selector.__dict__.get(name)
        if type(value) is not type or value.__module__ != selector.__name__:
            raise RealAppH8AdmissionUnavailable(
                f"Frozen selector type provenance changed: {name}"
            )
        selector_types[name] = value

    required_dependency_names = (
        (policy, "policy_payload", _FunctionType),
        (policy, "validate_applied_frame", _FunctionType),
        (anchor, "TransitionQualityParameters", type),
        (anchor, "whole_atom_probabilities", _FunctionType),
        (chain, "canonical_donor_quality_group_id", _FunctionType),
    )
    for module, name, expected_type in required_dependency_names:
        value = module.__dict__.get(name)
        if type(value) is not expected_type:
            raise RealAppH8AdmissionUnavailable(
                f"Frozen selector dependency is missing: {name}"
            )
        if expected_type is _FunctionType and value.__globals__ is not module.__dict__:
            raise RealAppH8AdmissionUnavailable(
                f"Frozen selector dependency provenance changed: {name}"
            )

    bindings = (
        (selector, "_synthetic", synthetic),
        (selector, "_policy", policy),
        (selector, "_anchor", anchor),
        (selector, "_chain_v3", chain),
        (selector, "_np", _np),
        (selector, "_pd", _pd),
        (selector, "_sha256_file", bounded_selector_source_hash),
        (synthetic, "_np", _np),
        (synthetic, "_pd", _pd),
        (synthetic, "_policy_payload", policy.policy_payload),
        (synthetic, "_validate_applied_policy_frame", policy.validate_applied_frame),
        (synthetic, "_TransitionQualityParameters", anchor.TransitionQualityParameters),
        (synthetic, "_whole_atom_probabilities", anchor.whole_atom_probabilities),
        (chain, "anchor", anchor),
        (chain, "np", _np),
        (chain, "pd", _pd),
        (policy, "np", _np),
        (policy, "pd", _pd),
        (anchor, "np", _np),
    )
    relevant_functions = (
        (policy, "policy_payload", policy.policy_payload),
        (policy, "validate_applied_frame", policy.validate_applied_frame),
        (anchor, "whole_atom_probabilities", anchor.whole_atom_probabilities),
        (
            chain,
            "canonical_donor_quality_group_id",
            chain.canonical_donor_quality_group_id,
        ),
    )
    for module, name, expected in (*bindings, *relevant_functions):
        if module.__dict__.get(name) is not expected:
            raise RealAppH8AdmissionUnavailable(
                f"Frozen selector dependency binding changed: {name}"
            )
    if anchor.TransitionQualityParameters.__module__ != anchor.__name__:
        raise RealAppH8AdmissionUnavailable(
            "Frozen selector anchor parameter type provenance changed"
        )

    binding_labels = (
        "selector._synthetic->synthetic",
        "selector._policy->policy",
        "selector._anchor->anchor",
        "selector._chain_v3->chain",
        "synthetic.policy_aliases->policy",
        "synthetic.anchor_aliases->anchor",
        "chain.anchor->anchor",
        "all.numpy_pandas->wrapper_captured_modules",
        "selector._sha256_file->fixed_descriptor_low_level_captured_bytes",
        "selector._validate_runtime->wrapper_owned_no_path_runtime_validator",
        "all.runtime_filesystem->captured_low_level_os_primitives",
        "all.pathlib_objects->inert_initialization_only_sentinel",
        "all.pathlib_sentinel->descriptor_code_mutable_constant_instance_state",
        "all.low_level_helpers->recursive_reachable_function_integrity",
        "all.guard_metadata->deep_immutable_mappingproxy_tuple",
        "private_modules.namespace->exact_str_warningregistry_seal",
        "wrapper.namespace->final_frozenset_key_seal",
        "wrapper.bindings->final_mappingproxy_identity_seal",
        "wrapper.call_graph->all_defined_function_and_class_integrity",
        "wrapper.state->self_included_mappingproxy_binding_seal",
        "session.capability->identity_bound_exact_code_closure_receipt",
        "wrapper.public_exports->immutable_tuple",
    )
    module_guards = tuple(
        _capture_module_integrity(role, module, builtins_map)
        for role, module in by_role.items()
    )
    import_guard = _capture_function_integrity(
        guarded_import, "capsule.guarded_import"
    )
    bounded_hash_guard = _capture_function_integrity(
        bounded_selector_source_hash, "capsule.bounded_selector_source_hash"
    )
    runtime_validator_guard = _capture_function_integrity(
        capsule_validate_runtime, "capsule.capsule_validate_runtime"
    )
    filesystem_function_guards = tuple(
        _capture_function_integrity(value, f"capsule.filesystem.{name}")
        for name, value in _FILESYSTEM_HELPERS.items()
    )
    capsule_sha256 = _stable_sha(
        {
            "contract": _SELECTOR_CAPSULE_CONTRACT,
            "sources": source_sha256,
            "bindings": list(binding_labels),
            "integrity": _SELECTOR_CAPSULE_INTEGRITY_CONTRACT,
        }
    )
    module_files = _MappingProxyType(
        {role: by_role[role].__file__ for role in by_role}
    )
    return _MappingProxyType(
        {
            "modules": _MappingProxyType(by_role),
            "private_names": _MappingProxyType(dict(private_names)),
            "module_files": module_files,
            "source_sha256": _MappingProxyType(source_sha256),
            "functions": _MappingProxyType(functions),
            "types": _MappingProxyType(selector_types),
            "bindings": bindings,
            "relevant_functions": relevant_functions,
            "module_guards": module_guards,
            "import_guard": import_guard,
            "bounded_hash_guard": bounded_hash_guard,
            "runtime_validator_guard": runtime_validator_guard,
            "filesystem_function_guards": filesystem_function_guards,
            "capsule_sha256": capsule_sha256,
        }
    )


def _make_selector_capsule_state(
    capsule: _MappingProxyType,
    expected_wrapper_keys: frozenset[str],
    expected_wrapper_items: _MappingProxyType,
    wrapper_function_guards: tuple[_MappingProxyType, ...],
    wrapper_class_guards: tuple[_MappingProxyType, ...],
):
    state_binding_name = "_selector_capsule_state"
    if type(expected_wrapper_keys) is not frozenset or any(
        type(key) is not str for key in expected_wrapper_keys
    ):
        raise RealAppH8AdmissionUnavailable(
            "Admission wrapper namespace seal is invalid"
        )
    if type(expected_wrapper_items) is not _MappingProxyType or any(
        type(key) is not str for key in expected_wrapper_items
    ):
        raise RealAppH8AdmissionUnavailable(
            "Admission wrapper binding seal is invalid"
        )
    if (
        state_binding_name not in expected_wrapper_keys
        or state_binding_name in expected_wrapper_items
        or frozenset(expected_wrapper_items)
        != expected_wrapper_keys - frozenset({state_binding_name})
    ):
        raise RealAppH8AdmissionUnavailable(
            "Admission wrapper self-binding seal is invalid"
        )
    if (
        type(wrapper_function_guards) is not tuple
        or any(type(guard) is not _MappingProxyType for guard in wrapper_function_guards)
        or type(wrapper_class_guards) is not tuple
        or any(type(guard) is not _MappingProxyType for guard in wrapper_class_guards)
    ):
        raise RealAppH8AdmissionUnavailable(
            "Admission wrapper call-graph seal is invalid"
        )
    modules = capsule["modules"]
    private_names = capsule["private_names"]
    module_files = capsule["module_files"]
    source_sha256 = capsule["source_sha256"]
    functions = capsule["functions"]
    selector_types = capsule["types"]
    bindings = capsule["bindings"]
    relevant_functions = capsule["relevant_functions"]
    module_guards = capsule["module_guards"]
    import_guard = capsule["import_guard"]
    bounded_hash_guard = capsule["bounded_hash_guard"]
    runtime_validator_guard = capsule["runtime_validator_guard"]
    filesystem_function_guards = capsule["filesystem_function_guards"]
    capsule_sha256 = capsule["capsule_sha256"]
    wrapper_namespace = globals()
    wrapper_builtins = wrapper_namespace["__builtins__"]
    if type(wrapper_builtins) is not dict:
        raise RealAppH8AdmissionUnavailable(
            "Admission wrapper builtins are not an exact dictionary"
        )
    wrapper_builtins_items = _MappingProxyType(dict(wrapper_builtins))
    error_type = RealAppH8AdmissionUnavailable
    exact_type = type
    exact_str = str
    exact_int = int
    exact_any = any
    exact_set = set
    exact_frozenset = frozenset
    exact_tuple = tuple
    str_equals = str.__eq__
    warning_name = "__warningregistry__"
    missing_namespace_value = object()
    dict_type = dict
    dict_get = dict.get
    str_type = str
    module_type = _ModuleType
    function_type = _FunctionType
    sys_modules = _sys.modules
    module_specs = tuple(_FROZEN_SELECTOR_MODULE_SPECS)
    capsule_contract = _SELECTOR_CAPSULE_CONTRACT
    filesystem_state = _FILESYSTEM_STATE
    check_function = _check_function_integrity
    check_class = _check_class_integrity
    check_module = _check_module_integrity
    checker_functions = tuple(
        (name, value, value.__code__)
        for name, value in (
            ("_json_bytes", _json_bytes),
            ("_stable_sha", _stable_sha),
            ("_integrity_payload", _integrity_payload),
            ("_integrity_digest", _integrity_digest),
            ("_semantic_object_digest", _semantic_object_digest),
            ("_check_function_integrity", _check_function_integrity),
            ("_check_class_integrity", _check_class_integrity),
            ("_check_module_integrity", _check_module_integrity),
            ("_is_reparse", _is_reparse),
            ("_read_capped_file", _read_capped_file),
        )
    )
    checker_values = tuple(
        (name, value)
        for name, value in (
            ("RealAppH8AdmissionUnavailable", RealAppH8AdmissionUnavailable),
            ("_DataclassField", _DataclassField),
            ("_FunctionType", _FunctionType),
            ("_MappingProxyType", _MappingProxyType),
            ("_ModuleType", _ModuleType),
            ("_EMPTY_CELL", _EMPTY_CELL),
            ("_FROZEN_SELECTOR_MODULE_SPECS", _FROZEN_SELECTOR_MODULE_SPECS),
            ("_KNOWN_SOURCE_PATHS", _KNOWN_SOURCE_PATHS),
            ("_MAX_REVIEW_SOURCE_BYTES", _MAX_REVIEW_SOURCE_BYTES),
            ("_FILESYSTEM_HELPERS", _FILESYSTEM_HELPERS),
            ("_FILESYSTEM_STATE", _FILESYSTEM_STATE),
            ("_FILESYSTEM_IS_REPARSE", _FILESYSTEM_IS_REPARSE),
            ("_FILESYSTEM_READ_CAPPED", _FILESYSTEM_READ_CAPPED),
            ("_FILESYSTEM_READ_EXPECTED", _FILESYSTEM_READ_EXPECTED),
            ("_FILESYSTEM_CAPTURE_LAYOUT", _FILESYSTEM_CAPTURE_LAYOUT),
            ("_json", _json),
            ("_os", _os),
            ("_pathlib", _pathlib),
            ("_sha256", _sha256),
            ("_stat", _stat),
        )
    )
    expected_aliases = _MappingProxyType(
        {
            "_SELECTOR_CANONICALIZE": functions["_canonicalize_fixture"],
            "_SELECTOR_SUPPORT_SHA": functions["_support_sha"],
            "_SELECTOR_WEIGHT": functions["_weight_fixture"],
            "_SELECTOR_SELECT": functions["_select_fixture"],
            "_SELECTOR_VALIDATE_RUNTIME": functions["_validate_runtime"],
            "_SELECTOR_WEIGHTING_PARAMETERS": selector_types[
                "RealAppWeightingParameters"
            ],
            "_SELECTOR_SEMANTIC_UNIFORMS": selector_types[
                "RealAppSemanticUniforms"
            ],
            "_SELECTOR_WEIGHTED_SUPPORT": selector_types[
                "RealAppWeightedSupport"
            ],
            "_SELECTOR_SELECTION_RESULT": selector_types[
                "RealAppSelectionResult"
            ],
        }
    )

    def state() -> dict[str, object]:
        namespace_keys = exact_tuple(wrapper_namespace)
        if exact_any(exact_type(key) is not str_type for key in namespace_keys):
            raise error_type("Admission wrapper namespace key type changed")
        warning_registry = dict_get(
            wrapper_namespace, warning_name, missing_namespace_value
        )
        if (
            warning_registry is not missing_namespace_value
            and exact_type(warning_registry) is not dict_type
        ):
            raise error_type("Admission wrapper warning registry type changed")
        observed_wrapper_keys = exact_frozenset(
            key
            for key in namespace_keys
            if not str_equals(key, warning_name)
        )
        if observed_wrapper_keys != expected_wrapper_keys:
            raise error_type("Admission wrapper namespace changed in memory")
        if (
            dict_get(wrapper_namespace, "__builtins__") is not wrapper_builtins
            or exact_type(wrapper_builtins) is not dict_type
            or exact_any(exact_type(key) is not str_type for key in wrapper_builtins)
            or exact_set(wrapper_builtins) != exact_set(wrapper_builtins_items)
            or exact_any(
                wrapper_builtins[key] is not value
                for key, value in wrapper_builtins_items.items()
            )
        ):
            raise error_type("Admission wrapper builtins changed in memory")
        for name, expected, expected_code in checker_functions:
            if (
                dict_get(wrapper_namespace, name) is not expected
                or expected.__code__ is not expected_code
            ):
                raise error_type(
                    f"Frozen selector capsule checker changed in memory: {name}"
                )
        for name, expected in checker_values:
            if dict_get(wrapper_namespace, name) is not expected:
                raise error_type(
                    f"Frozen selector capsule dependency changed in memory: {name}"
                )
        if exact_any(
            dict_get(wrapper_namespace, name, missing_namespace_value) is not value
            for name, value in expected_wrapper_items.items()
        ):
            raise error_type("Admission wrapper binding changed in memory")
        check_function(import_guard)
        check_function(bounded_hash_guard)
        check_function(runtime_validator_guard)
        for guard in filesystem_function_guards:
            check_function(guard)
        for guard in wrapper_function_guards:
            check_function(guard)
        for guard in wrapper_class_guards:
            check_class(guard)
        filesystem_start = filesystem_state()
        for guard in module_guards:
            check_module(guard)
        for role, canonical, relative in module_specs:
            module = modules[role]
            if (
                exact_type(module) is not module_type
                or module.__name__ != private_names[canonical]
                or module.__file__ != module_files[role]
                or sys_modules.get(canonical) is module
            ):
                raise error_type(
                    f"Frozen selector-capsule module changed in memory: {role}"
                )
        selector = modules["selector"]
        for name, expected in functions.items():
            if (
                selector.__dict__.get(name) is not expected
                or exact_type(expected) is not function_type
                or (
                    name != "_validate_runtime"
                    and expected.__globals__ is not selector.__dict__
                )
            ):
                raise error_type(
                    f"Frozen selector function changed in memory: {name}"
                )
        for name, expected in selector_types.items():
            if selector.__dict__.get(name) is not expected:
                raise error_type(
                    f"Frozen selector type changed in memory: {name}"
                )
        for module, name, expected in bindings:
            if module.__dict__.get(name) is not expected:
                raise error_type(
                    f"Frozen selector dependency binding changed in memory: {name}"
                )
        for module, name, expected in relevant_functions:
            if (
                module.__dict__.get(name) is not expected
                or exact_type(expected) is not function_type
                or expected.__globals__ is not module.__dict__
            ):
                raise error_type(
                    f"Frozen selector dependency function changed in memory: {name}"
                )
        for name, expected in expected_aliases.items():
            if dict_get(wrapper_namespace, name) is not expected:
                raise error_type(
                    f"Frozen selector wrapper alias changed in memory: {name}"
                )
        if filesystem_start != filesystem_state():
            raise error_type("Captured filesystem changed during capsule validation")
        return {
            "contract": capsule_contract,
            "capsule_sha256": capsule_sha256,
            "source_sha256": dict_type(source_sha256),
        }

    completed_wrapper_items = dict_type(expected_wrapper_items)
    completed_wrapper_items[state_binding_name] = state
    expected_wrapper_items = _MappingProxyType(completed_wrapper_items)
    return state


_SELECTOR_CAPSULE = _load_frozen_selector_capsule()
del _load_frozen_selector_capsule
if _SELECTOR_CAPSULE["capsule_sha256"] != _SELECTOR_CAPSULE_SHA256:
    raise RealAppH8AdmissionUnavailable("Frozen selector capsule digest changed")
_SELECTOR_CANONICALIZE = _SELECTOR_CAPSULE["functions"]["_canonicalize_fixture"]
_SELECTOR_SUPPORT_SHA = _SELECTOR_CAPSULE["functions"]["_support_sha"]
_SELECTOR_WEIGHT = _SELECTOR_CAPSULE["functions"]["_weight_fixture"]
_SELECTOR_SELECT = _SELECTOR_CAPSULE["functions"]["_select_fixture"]
_SELECTOR_VALIDATE_RUNTIME = _SELECTOR_CAPSULE["functions"]["_validate_runtime"]
_SELECTOR_WEIGHTING_PARAMETERS = _SELECTOR_CAPSULE["types"][
    "RealAppWeightingParameters"
]
_SELECTOR_SEMANTIC_UNIFORMS = _SELECTOR_CAPSULE["types"][
    "RealAppSemanticUniforms"
]
_SELECTOR_WEIGHTED_SUPPORT = _SELECTOR_CAPSULE["types"]["RealAppWeightedSupport"]
_SELECTOR_SELECTION_RESULT = _SELECTOR_CAPSULE["types"]["RealAppSelectionResult"]


def _capture_tree(
    root: str,
    expected_files: dict[str, object],
    expected_directories: tuple[str, ...],
    label: str,
    inventory_contract: str,
) -> tuple[dict[str, bytes], dict[str, object]]:
    if type(expected_files) is not dict or any(
        type(path) is not str for path in expected_files
    ):
        raise RealAppH8AdmissionUnavailable(f"{label} expected file map changed")
    if type(expected_directories) is not tuple or any(
        type(path) is not str for path in expected_directories
    ):
        raise RealAppH8AdmissionUnavailable(f"{label} expected directories changed")
    for relative, metadata in expected_files.items():
        row = _require_keys(metadata, {"bytes", "sha256"}, f"{label} {relative}")
        if type(row["bytes"]) is not int or row["bytes"] < 0:
            raise RealAppH8AdmissionUnavailable(f"{label} expected bytes changed")
        _require_sha(row["sha256"], f"{label} expected digest")
    expected_paths = tuple(sorted(expected_files))
    first_layout = _FILESYSTEM_CAPTURE_LAYOUT(
        root, expected_paths, expected_directories, label
    )
    first_rows = {
        row[1]: row
        for row in first_layout["rows"]
        if row[0] == "file"
    }
    for relative in expected_paths:
        metadata = expected_files[relative]
        if type(metadata) is not dict:
            raise RealAppH8AdmissionUnavailable(
                f"{label} metadata type changed: {relative}"
            )
        if first_rows[relative][5] != metadata["bytes"]:
            raise RealAppH8AdmissionUnavailable(
                f"{label} byte size changed: {relative}"
            )
    files: dict[str, bytes] = {}
    for relative in expected_paths:
        metadata = expected_files[relative]
        files[relative] = _FILESYSTEM_READ_EXPECTED(
            first_layout["paths"][relative],
            metadata["bytes"],
            f"{label} {relative}",
        )
    final_layout = _FILESYSTEM_CAPTURE_LAYOUT(
        root, expected_paths, expected_directories, label
    )
    if (
        first_layout["root"] != final_layout["root"]
        or first_layout["rows"] != final_layout["rows"]
        or first_layout["directories"] != final_layout["directories"]
        or first_layout["paths"] != final_layout["paths"]
    ):
        raise RealAppH8AdmissionUnavailable(f"{label} closure changed during capture")
    inventory: list[dict[str, object]] = []
    for path in sorted(files):
        metadata = expected_files[path]
        if type(metadata) is not dict:
            raise RealAppH8AdmissionUnavailable(
                f"{label} metadata type changed: {path}"
            )
        raw = files[path]
        digest = _sha256(raw).hexdigest()
        if len(raw) != metadata["bytes"] or digest != metadata["sha256"]:
            raise RealAppH8AdmissionUnavailable(f"{label} bytes changed: {path}")
        inventory.append({"path": path, "bytes": len(raw), "sha256": digest})
    inventory_payload: object
    if inventory_contract == "audit_v2_input_tree":
        inventory_payload = {
            "directories": list(expected_directories),
            "files": inventory,
        }
    elif inventory_contract == "audit_v2_output_tree":
        if expected_directories:
            raise RealAppH8AdmissionUnavailable(
                "Output-tree inventory cannot contain directories"
            )
        inventory_payload = inventory
    else:
        raise RealAppH8AdmissionUnavailable("Unknown tree inventory contract")
    return files, {
        "files": len(files),
        "directories": len(expected_directories),
        "bytes": sum(len(raw) for raw in files.values()),
        "inventory_sha256": _domain_sha(inventory_payload, inventory_contract),
    }


def _validate_output_index(
    raw: bytes,
    captured: dict[str, bytes],
    *,
    index_path: str,
    prefix: str,
    contract: str,
    directories: tuple[str, ...] | None,
    label: str,
) -> None:
    payload = _strict_json(raw, label)
    fields = {"contract", "files"}
    if directories is not None:
        fields.add("directories")
    index = _require_keys(payload, fields, label)
    if type(index["contract"]) is not str or not str.__eq__(index["contract"], contract):
        raise RealAppH8AdmissionUnavailable(f"{label} contract changed")
    if directories is not None:
        _require_exact_tree(index["directories"], list(directories), f"{label} dirs")
    files = index["files"]
    if type(files) is not dict or any(type(path) is not str for path in files):
        raise RealAppH8AdmissionUnavailable(f"{label} file map changed")
    expected_paths = {
        path[len(prefix) :] if prefix and path.startswith(prefix) else path
        for path in captured
        if path != index_path and (not prefix or path.startswith(prefix))
    }
    if set(files) != expected_paths:
        raise RealAppH8AdmissionUnavailable(f"{label} file closure changed")
    for relative in sorted(files):
        row = _require_keys(files[relative], {"bytes", "sha256"}, f"{label} {relative}")
        captured_path = f"{prefix}{relative}" if prefix else relative
        value = captured[captured_path]
        if type(row["bytes"]) is not int or row["bytes"] != len(value):
            raise RealAppH8AdmissionUnavailable(f"{label} bytes changed: {relative}")
        digest = _require_sha(row["sha256"], f"{label} digest {relative}")
        if not str.__eq__(digest, _sha256(value).hexdigest()):
            raise RealAppH8AdmissionUnavailable(f"{label} digest changed: {relative}")


def _require_disabled(payload: dict[str, object], fields: tuple[str, ...], label: str) -> None:
    for field in fields:
        if payload.get(field) is not False:
            raise RealAppH8AdmissionUnavailable(f"{label} readiness changed: {field}")


def _cross_bind_evidence(
    support: dict[str, bytes],
    candidate: dict[str, bytes],
    audit: dict[str, bytes],
) -> dict[str, str]:
    fixed_hashes = {
        "support_outer_index": _SUPPORT_IDENTITY["outer_output_index_sha256"],
        "support_outer_manifest": _SUPPORT_IDENTITY["outer_manifest_sha256"],
        "support_inner_index": _SUPPORT_IDENTITY["inner_output_index_sha256"],
        "support_inner_manifest": _SUPPORT_IDENTITY["inner_manifest_sha256"],
        "candidate_index": _CANDIDATE_IDENTITY["output_index_sha256"],
        "candidate_json": _CANDIDATE_IDENTITY["candidate_json_sha256"],
        "candidate_manifest": _CANDIDATE_IDENTITY["manifest_sha256"],
        "candidate_report": _CANDIDATE_IDENTITY["report_sha256"],
        "audit_index": _AUDIT_IDENTITY["output_index_sha256"],
        "audit_json": _AUDIT_IDENTITY["audit_json_sha256"],
        "audit_manifest": _AUDIT_IDENTITY["manifest_sha256"],
        "audit_report": _AUDIT_IDENTITY["report_sha256"],
    }
    observed_hashes = {
        "support_outer_index": _sha256(support["OUTPUT_SHA256.json"]).hexdigest(),
        "support_outer_manifest": _sha256(support["manifest.json"]).hexdigest(),
        "support_inner_index": _sha256(
            support["complete_atom_support/OUTPUT_SHA256.json"]
        ).hexdigest(),
        "support_inner_manifest": _sha256(
            support["complete_atom_support/manifest.json"]
        ).hexdigest(),
        "candidate_index": _sha256(candidate["OUTPUT_SHA256.json"]).hexdigest(),
        "candidate_json": _sha256(candidate["authority_candidate.json"]).hexdigest(),
        "candidate_manifest": _sha256(candidate["manifest.json"]).hexdigest(),
        "candidate_report": _sha256(candidate["REPORT.md"]).hexdigest(),
        "audit_index": _sha256(audit["OUTPUT_SHA256.json"]).hexdigest(),
        "audit_json": _sha256(audit["audit.json"]).hexdigest(),
        "audit_manifest": _sha256(audit["manifest.json"]).hexdigest(),
        "audit_report": _sha256(audit["REPORT.md"]).hexdigest(),
    }
    if not _exact_tree_equal(observed_hashes, fixed_hashes):
        raise RealAppH8AdmissionUnavailable("Fixed artifact identity hashes changed")

    _validate_output_index(
        support["OUTPUT_SHA256.json"],
        support,
        index_path="OUTPUT_SHA256.json",
        prefix="",
        contract="fixed_lowry_r1h8_outer_output_sha256_v1",
        directories=_SUPPORT_DIRECTORIES,
        label="support outer index",
    )
    _validate_output_index(
        support["complete_atom_support/OUTPUT_SHA256.json"],
        support,
        index_path="complete_atom_support/OUTPUT_SHA256.json",
        prefix="complete_atom_support/",
        contract="non_circular_output_sha256_v1",
        directories=None,
        label="support inner index",
    )
    _validate_output_index(
        candidate["OUTPUT_SHA256.json"],
        candidate,
        index_path="OUTPUT_SHA256.json",
        prefix="",
        contract="non_circular_output_sha256_v1",
        directories=None,
        label="candidate index",
    )
    _validate_output_index(
        audit["OUTPUT_SHA256.json"],
        audit,
        index_path="OUTPUT_SHA256.json",
        prefix="",
        contract="non_circular_output_sha256_v1",
        directories=None,
        label="audit index",
    )

    support_manifest = _strict_json(support["manifest.json"], "support manifest")
    if type(support_manifest) is not dict:
        raise RealAppH8AdmissionUnavailable("Support manifest changed")
    for field, expected in {
        "contract": "fixed_lowry_r1h8_complete_atom_materialization_v1",
        "complete_atoms": _COUNTS["complete_atoms"],
        "terminal_atoms": _COUNTS["terminal_atoms"],
        "imputed_ability_atoms": _COUNTS["imputed_ability_atoms"],
        "query_key": _QUERY_KEY,
        "support_key": _SUPPORT_KEY,
        "authority_id": _SUPPORT_IDENTITY["source_authority_id"],
        "authority_record_sha256": _SUPPORT_IDENTITY[
            "source_authority_record_sha256"
        ],
        "authority_lock_sha256": _SUPPORT_IDENTITY["source_authority_lock_sha256"],
        "inner_output_index_sha256": _SUPPORT_IDENTITY[
            "inner_output_index_sha256"
        ],
    }.items():
        _require_exact_tree(support_manifest.get(field), expected, f"support {field}")
    _require_disabled(
        support_manifest,
        ("anchor_ready", "mean_solve_launched", "simulation_launched", "production_enabled"),
        "support manifest",
    )
    if support_manifest.get("research_only") is not True:
        raise RealAppH8AdmissionUnavailable("Support research-only status changed")

    inner_manifest = _strict_json(
        support["complete_atom_support/manifest.json"], "inner support manifest"
    )
    if type(inner_manifest) is not dict:
        raise RealAppH8AdmissionUnavailable("Inner support manifest changed")
    for field, expected in {
        "contract": "bounded_sharded_complete_atom_persistence_v4",
        "complete_atoms": _COUNTS["complete_atoms"],
        "requests": 1,
        "max_rows_per_part": _LIMITS["max_rows_per_part"],
        "max_hazard_donors_per_surface": 8192,
        "max_complete_atoms_per_proposal": _LIMITS["max_atoms_per_proposal"],
        "absolute_max_hazard_provider_donors": 8192,
        "source_inventory_authority_id": _SUPPORT_IDENTITY["source_authority_id"],
        "source_inventory_authority_lock_sha256": _SUPPORT_IDENTITY[
            "source_authority_lock_sha256"
        ],
    }.items():
        _require_exact_tree(inner_manifest.get(field), expected, f"inner support {field}")
    _require_disabled(inner_manifest, ("production_enabled",), "inner support manifest")
    if inner_manifest.get("research_only") is not True:
        raise RealAppH8AdmissionUnavailable("Inner support research-only status changed")

    candidate_payload = _strict_json(candidate["authority_candidate.json"], "candidate")
    candidate_manifest = _strict_json(candidate["manifest.json"], "candidate manifest")
    audit_payload = _strict_json(audit["audit.json"], "v2 audit")
    audit_manifest = _strict_json(audit["manifest.json"], "v2 audit manifest")
    for payload, label in (
        (candidate_payload, "candidate"),
        (candidate_manifest, "candidate manifest"),
        (audit_payload, "v2 audit"),
        (audit_manifest, "v2 audit manifest"),
    ):
        if type(payload) is not dict:
            raise RealAppH8AdmissionUnavailable(f"{label} payload changed")
    _require_exact_tree(
        candidate_payload.get("candidate_id"),
        _CANDIDATE_IDENTITY["candidate_id"],
        "candidate id",
    )
    _require_exact_tree(
        candidate_manifest.get("candidate_id"),
        _CANDIDATE_IDENTITY["candidate_id"],
        "candidate manifest id",
    )
    _require_exact_tree(
        audit_payload.get("audit_id"), _AUDIT_IDENTITY["audit_id"], "audit id"
    )
    _require_exact_tree(
        audit_manifest.get("audit_id"),
        _AUDIT_IDENTITY["audit_id"],
        "audit manifest id",
    )
    _require_exact_tree(
        audit_payload.get("reviewed_candidate_id"),
        _CANDIDATE_IDENTITY["candidate_id"],
        "audit reviewed candidate",
    )
    _require_exact_tree(
        audit_payload.get("decision"), _AUDIT_IDENTITY["decision"], "audit decision"
    )
    disabled = (
        "authority_ready",
        "real_support_admitted",
        "weighting_enabled",
        "selection_enabled",
        "chain_ready",
        "anchor_ready",
        "callback_enabled",
        "simulation_launched",
        "mean_solve_launched",
        "production_enabled",
    )
    _require_disabled(candidate_payload, disabled, "candidate")
    _require_disabled(audit_payload, disabled, "v2 audit")
    _require_disabled(
        candidate_manifest,
        (
            "anchor_ready",
            "authority_ready",
            "callback_enabled",
            "callback_launched",
            "chain_ready",
            "mean_solve_launched",
            "production_enabled",
            "real_support_admitted",
            "selection_enabled",
            "selector_support_ready",
            "simulation_launched",
            "weighting_enabled",
        ),
        "candidate manifest",
    )
    _require_disabled(
        audit_manifest,
        (
            "anchor_ready",
            "authority_ready",
            "callback_enabled",
            "callback_launched",
            "chain_ready",
            "data_only_authority_record_ready",
            "mean_solve_launched",
            "production_enabled",
            "selection_enabled",
            "selector_support_ready",
            "simulation_launched",
            "weighting_enabled",
        ),
        "v2 audit manifest",
    )
    for payload, label in (
        (candidate_payload, "candidate"),
        (candidate_manifest, "candidate manifest"),
        (audit_payload, "v2 audit"),
        (audit_manifest, "v2 audit manifest"),
    ):
        if payload.get("research_only") is not True:
            raise RealAppH8AdmissionUnavailable(f"{label} research-only status changed")
    vectorized = audit_payload.get("vectorized_invariants")
    if type(vectorized) is not dict:
        raise RealAppH8AdmissionUnavailable("V2 audit vector evidence changed")
    _require_exact_tree(
        vectorized.get("audit_only_semantic_root_sha256"),
        _NDJSON["semantic_root"][3],
        "V2 semantic root",
    )
    components = vectorized.get("ndjson_components")
    if type(components) is not dict or set(components) != set(_NDJSON):
        raise RealAppH8AdmissionUnavailable("V2 NDJSON component map changed")
    for name, expected in _NDJSON.items():
        _require_exact_tree(
            components[name],
            {
                "contract": expected[0],
                "records": expected[1],
                "framed_bytes": expected[2],
                "sha256": expected[3],
            },
            f"V2 NDJSON component {name}",
        )
    return observed_hashes


def _evidence_state(
    record: dict[str, object],
) -> tuple[dict[str, bytes], dict[str, object]]:
    capsule_start = _selector_capsule_state()
    filesystem_start = _FILESYSTEM_STATE()
    result: dict[str, object] = {}
    support_section = record["support_artifact"]
    candidate_section = record["candidate_artifact"]
    audit_section = record["audit_artifact"]
    if type(support_section) is not dict:
        raise RealAppH8AdmissionUnavailable("Support section type changed")
    if type(candidate_section) is not dict:
        raise RealAppH8AdmissionUnavailable("Candidate section type changed")
    if type(audit_section) is not dict:
        raise RealAppH8AdmissionUnavailable("Audit section type changed")
    support, result["support"] = _capture_tree(
        _SUPPORT_ROOT,
        support_section["files"],
        _SUPPORT_DIRECTORIES,
        "fixed support",
        str(support_section["inventory_contract"]),
    )
    candidate, result["candidate"] = _capture_tree(
        _CANDIDATE_ROOT,
        candidate_section["files"],
        (),
        "fixed candidate",
        str(candidate_section["inventory_contract"]),
    )
    audit, result["audit"] = _capture_tree(
        _AUDIT_ROOT,
        audit_section["files"],
        (),
        "fixed audit",
        str(audit_section["inventory_contract"]),
    )
    if result["support"]["bytes"] != support_section["bytes"]:
        raise RealAppH8AdmissionUnavailable("Support byte total changed")
    if result["candidate"]["bytes"] != candidate_section["bytes"]:
        raise RealAppH8AdmissionUnavailable("Candidate byte total changed")
    if result["audit"]["bytes"] != audit_section["bytes"]:
        raise RealAppH8AdmissionUnavailable("Audit byte total changed")
    for name, section in (
        ("support", support_section),
        ("candidate", candidate_section),
        ("audit", audit_section),
    ):
        if result[name]["inventory_sha256"] != section["inventory_sha256"]:
            raise RealAppH8AdmissionUnavailable(f"{name} inventory changed")

    result["fixed_identity_sha256"] = _stable_sha(
        _cross_bind_evidence(support, candidate, audit)
    )
    result["combined_sha256"] = _stable_sha(result)
    capsule_end = _selector_capsule_state()
    if capsule_start != capsule_end:
        raise RealAppH8AdmissionUnavailable(
            "Frozen selector capsule changed during evidence validation"
        )
    if filesystem_start != _FILESYSTEM_STATE():
        raise RealAppH8AdmissionUnavailable(
            "Captured filesystem changed during evidence validation"
        )
    return support, result


def _load_frame(support: dict[str, bytes]) -> _pd.DataFrame:
    tables: list[_pa.Table] = []
    schema: _pa.Schema | None = None
    for path in _PART_FILES:
        table = _pq.read_table(_pa.BufferReader(support[path]))
        if schema is None:
            schema = table.schema
        elif not table.schema.equals(schema, check_metadata=True):
            raise RealAppH8AdmissionUnavailable("Parquet part schema changed")
        tables.append(table)
    frame = _pa.concat_tables(tables, promote_options="none").to_pandas()
    canonical = _SELECTOR_CANONICALIZE(frame)
    if len(canonical) != _COUNTS["complete_atoms"]:
        raise RealAppH8AdmissionUnavailable("Canonical support row count changed")
    if canonical["proposal_row_order"].nunique() != _COUNTS["proposals"]:
        raise RealAppH8AdmissionUnavailable("Canonical proposal count changed")
    if _SELECTOR_SUPPORT_SHA(canonical) != _SELECTOR_SUPPORT_SHA256:
        raise RealAppH8AdmissionUnavailable("Canonical selector support changed")
    return canonical


def _record_sha(record: dict[str, object]) -> str:
    return _stable_sha(
        {"contract": REAL_APP_H8_RECORD_CONTRACT, "record": record}
    )


@_dataclass(frozen=True)
class RealAppH8Readiness:
    contract: str
    authority_id: str
    authority_record_sha256: str
    authority_lock_sha256: str
    selector_runtime_sha256: str
    selector_support_sha256: str
    complete_atoms: int
    proposals: int
    spatial_proposals: int
    terminal_atoms: int
    donor_quality_groups: int
    mixed_policy_groups: int
    imputed_ability_atoms: int
    real_support_admitted: bool
    weighting_ready: bool
    selection_ready: bool
    callback_ready: bool
    chain_ready: bool
    anchor_ready: bool
    mean_solve_ready: bool
    simulation_ready: bool
    production_ready: bool
    research_only: bool
    blockers: tuple[str, ...]


@_dataclass(frozen=True)
class _SessionSeal:
    authority_id: str
    record_sha256: str
    authority_lock_sha256: str
    wrapper_sha256: str


RealAppWeightingParameters = _SELECTOR_WEIGHTING_PARAMETERS
RealAppSemanticUniforms = _SELECTOR_SEMANTIC_UNIFORMS
RealAppWeightedSupport = _SELECTOR_WEIGHTED_SUPPORT
RealAppSelectionResult = _SELECTOR_SELECTION_RESULT


class RealAppH8SelectorSession:
    """Opaque capability minted only after fixed source and artifact admission."""

    __slots__ = ("_receipt", "__weakref__")

    def __init__(self, *_: object, **__: object) -> None:
        raise RealAppH8AdmissionUnavailable(
            "Real H8 sessions are minted only by fixed zero-argument admission"
        )

    @property
    def readiness(self) -> RealAppH8Readiness:
        record, _frame, before = _live_frame(self)
        seal = _session_seal(self)
        readiness = _readiness(record, seal)
        if before != _integrity_state(self):
            raise RealAppH8AdmissionUnavailable("Integrity changed during readiness")
        return readiness

    def weight(self, parameters: RealAppWeightingParameters) -> RealAppWeightedSupport:
        record, frame, before = _live_frame(self)
        weighted = _SELECTOR_WEIGHT(frame, parameters).weighted
        after = _integrity_state(self)
        if before != after:
            raise RealAppH8AdmissionUnavailable("Integrity changed during weighting")
        return weighted

    def select(
        self,
        parameters: RealAppWeightingParameters,
        uniforms: RealAppSemanticUniforms,
    ) -> RealAppSelectionResult:
        record, frame, before = _live_frame(self)
        selected = _SELECTOR_SELECT(frame, parameters, uniforms)
        after = _integrity_state(self)
        if before != after:
            raise RealAppH8AdmissionUnavailable("Integrity changed during selection")
        return selected


def _state_payload(
    seal: _SessionSeal,
    source: dict[str, object],
    evidence: dict[str, object],
) -> dict[str, object]:
    return {
        "source": source,
        "evidence": evidence,
        "record": seal.record_sha256,
        "authority_id": seal.authority_id,
        "authority_lock_sha256": seal.authority_lock_sha256,
        "wrapper_sha256": seal.wrapper_sha256,
    }


def _live_state(
    session: object,
) -> tuple[dict[str, object], dict[str, bytes], dict[str, object], _SessionSeal]:
    seal = _session_seal(session)
    record, authority_lock = _closed_authority()
    if record is None:
        raise RealAppH8AdmissionUnavailable("Real H8 authority was removed")
    record_sha = _record_sha(record)
    authority_id = record["authority_id"]
    if type(authority_id) is not str:
        raise RealAppH8AdmissionUnavailable("Real H8 authority id changed")
    if not str.__eq__(record_sha, seal.record_sha256):
        raise RealAppH8AdmissionUnavailable("Real H8 authority record changed")
    if not str.__eq__(authority_id, seal.authority_id):
        raise RealAppH8AdmissionUnavailable("Real H8 authority identity changed")
    if not str.__eq__(authority_lock, seal.authority_lock_sha256):
        raise RealAppH8AdmissionUnavailable("Real H8 authority lock changed")
    source = _source_state(record)
    if not str.__eq__(source["wrapper_sha256"], seal.wrapper_sha256):
        raise RealAppH8AdmissionUnavailable("Admission wrapper changed after mint")
    support, evidence = _evidence_state(record)
    return record, support, _state_payload(seal, source, evidence), seal


def _integrity_state(session: object) -> dict[str, object]:
    _record, _support, state, _seal = _live_state(session)
    return state


def _live_frame(
    session: object,
) -> tuple[dict[str, object], _pd.DataFrame, dict[str, object]]:
    record, support, before, _seal = _live_state(session)
    frame = _load_frame(support)
    after = _integrity_state(session)
    if before != after:
        raise RealAppH8AdmissionUnavailable("Integrity changed while loading support")
    return record, frame, after


def _readiness(record: dict[str, object], seal: _SessionSeal) -> RealAppH8Readiness:
    return RealAppH8Readiness(
        contract=REAL_APP_H8_SESSION_CONTRACT,
        authority_id=seal.authority_id,
        authority_record_sha256=seal.record_sha256,
        authority_lock_sha256=seal.authority_lock_sha256,
        selector_runtime_sha256=_SELECTOR_RUNTIME_SHA256,
        selector_support_sha256=_SELECTOR_SUPPORT_SHA256,
        complete_atoms=_COUNTS["complete_atoms"],
        proposals=_COUNTS["proposals"],
        spatial_proposals=_COUNTS["spatial_proposals"],
        terminal_atoms=_COUNTS["terminal_atoms"],
        donor_quality_groups=_COUNTS["donor_quality_groups"],
        mixed_policy_groups=_COUNTS["mixed_policy_groups"],
        imputed_ability_atoms=_COUNTS["imputed_ability_atoms"],
        real_support_admitted=True,
        weighting_ready=True,
        selection_ready=True,
        callback_ready=False,
        chain_ready=False,
        anchor_ready=False,
        mean_solve_ready=False,
        simulation_ready=False,
        production_ready=False,
        research_only=True,
        blockers=(
            "no_physical_chain_callback",
            "no_anchor_or_mean_solve_integration",
            "no_simulation_or_production_integration",
        ),
    )


def real_app_h8_authority_manifest() -> dict[str, object]:
    """Report the closed registry without reading any source or artifact."""

    record, authority_lock = _closed_authority()
    return {
        "contract": REAL_APP_H8_AUTHORITY_CONTRACT,
        "authority_lock_sha256": authority_lock,
        "records": [] if record is None else [record],
        "authority_record_present": record is not None,
        "real_support_admitted": False,
        "weighting_enabled": False,
        "selection_enabled": False,
        "callback_enabled": False,
        "chain_ready": False,
        "anchor_ready": False,
        "mean_solve_launched": False,
        "simulation_launched": False,
        "production_enabled": False,
        "research_only": True,
        "integration_status": INTEGRATION_STATUS if record is None else (
            "record_present_session_not_yet_minted"
        ),
    }


def _make_session_api():
    """Build a registry-free, identity-bound pure-Python session capability.

    An exact receipt function and its closure bind each seal to one session.
    This removes the writable closure registry.  Arbitrary construction or
    mutation of CPython function cells is the explicit pure-Python boundary;
    defending that stronger interpreter-introspection model requires a
    C-backed or cryptographic capability.
    """

    error_type = RealAppH8AdmissionUnavailable
    session_type = RealAppH8SelectorSession
    seal_type = _SessionSeal
    function_type = _FunctionType
    exact_type = type
    exact_any = any
    exact_zip = zip
    tuple_type = tuple
    dict_type = dict
    exact_len = len
    str_type = str
    str_equals = str.__eq__
    object_new = object.__new__
    object_getattribute = object.__getattribute__
    object_setattr = object.__setattr__
    attribute_error = AttributeError
    value_error = ValueError
    receipt_attribute = "_receipt"
    receipt_code: object | None = None
    receipt_error_index = -1
    receipt_seal_index = -1
    receipt_session_index = -1

    def session_seal(session: object) -> _SessionSeal:
        if exact_type(session) is not session_type:
            raise error_type("Real H8 session type changed")
        try:
            receipt = object_getattribute(session, receipt_attribute)
        except attribute_error as exc:
            raise error_type("Real H8 session was not minted") from exc
        if (
            receipt_code is None
            or exact_type(receipt) is not function_type
            or receipt.__code__ is not receipt_code
            or receipt.__defaults__ is not None
            or receipt.__kwdefaults__ is not None
            or exact_type(receipt.__annotations__) is not dict_type
            or receipt.__annotations__
            or exact_type(receipt.__dict__) is not dict_type
            or receipt.__dict__
            or exact_type(receipt.__name__) is not str_type
            or not str_equals(receipt.__name__, "receipt")
        ):
            raise error_type("Real H8 session receipt implementation changed")
        closure = receipt.__closure__
        if exact_type(closure) is not tuple_type or exact_len(closure) != 3:
            raise error_type("Real H8 session receipt closure changed")
        try:
            before = tuple_type(cell.cell_contents for cell in closure)
        except value_error as exc:
            raise error_type("Real H8 session receipt closure changed") from exc
        if (
            before[receipt_error_index] is not error_type
            or exact_type(before[receipt_seal_index]) is not seal_type
            or before[receipt_session_index] is not session
        ):
            raise error_type("Real H8 session receipt identity changed")
        seal = receipt(session)
        after_closure = receipt.__closure__
        try:
            after = tuple_type(cell.cell_contents for cell in after_closure or ())
        except value_error as exc:
            raise error_type("Real H8 session receipt closure changed") from exc
        if (
            receipt.__code__ is not receipt_code
            or after_closure is not closure
            or exact_len(after) != exact_len(before)
            or exact_any(
                left is not right
                for left, right in exact_zip(before, after, strict=True)
            )
            or seal is not before[receipt_seal_index]
            or exact_type(seal) is not seal_type
        ):
            raise error_type("Real H8 session receipt changed during validation")
        return seal

    def admit() -> RealAppH8SelectorSession:
        record, authority_lock = _closed_authority()
        if record is None:
            raise RealAppH8AdmissionUnavailable(
                "No reviewed real-H8 selector authority record is present"
            )
        source_start = _source_state(record)
        support, evidence_start = _evidence_state(record)
        _load_frame(support)
        source_end = _source_state(record)
        _support_end, evidence_end = _evidence_state(record)
        record_end, authority_lock_end = _closed_authority()
        if record_end is None:
            raise RealAppH8AdmissionUnavailable("Real H8 authority was removed")
        if (
            source_start != source_end
            or evidence_start != evidence_end
            or _record_sha(record) != _record_sha(record_end)
            or not str.__eq__(authority_lock, authority_lock_end)
        ):
            raise RealAppH8AdmissionUnavailable("Admission inputs changed during mint")
        session = object_new(session_type)
        seal = seal_type(
            authority_id=str(record_end["authority_id"]),
            record_sha256=_record_sha(record_end),
            authority_lock_sha256=authority_lock_end,
            wrapper_sha256=str(source_end["wrapper_sha256"]),
        )

        def receipt(candidate):
            if candidate is not session:
                raise error_type("Real H8 session receipt identity changed")
            return seal

        if receipt.__code__ is not receipt_code:
            raise error_type("Real H8 session receipt implementation changed")
        object_setattr(session, receipt_attribute, receipt)
        if session_seal(session) is not seal:
            raise error_type("Real H8 session receipt installation changed")
        expected = _state_payload(seal, source_end, evidence_end)
        if expected != _integrity_state(session):
            raise error_type("Admission integrity changed after mint")
        return session

    code_type = exact_type(admit.__code__)
    receipt_codes = tuple_type(
        value
        for value in admit.__code__.co_consts
        if exact_type(value) is code_type and value.co_name == "receipt"
    )
    if exact_len(receipt_codes) != 1:
        raise error_type("Real H8 session receipt code is not unique")
    receipt_code = receipt_codes[0]
    if receipt_code.co_freevars != ("error_type", "seal", "session"):
        raise error_type("Real H8 session receipt closure contract changed")
    receipt_error_index = 0
    receipt_seal_index = 1
    receipt_session_index = 2
    return session_seal, admit


_session_seal, admit_lowry_r1h8_app_support = _make_session_api()
_session_seal.__name__ = "_session_seal"
admit_lowry_r1h8_app_support.__name__ = "admit_lowry_r1h8_app_support"
admit_lowry_r1h8_app_support.__doc__ = (
    "Admit only the one source-locked fixed support; no caller inputs exist."
)
del _make_session_api


__all__ = (
    "REAL_APP_H8_ADMISSION_CONTRACT",
    "REAL_APP_H8_AUTHORITY_CONTRACT",
    "REAL_APP_H8_RECORD_CONTRACT",
    "REAL_APP_H8_SESSION_CONTRACT",
    "INTEGRATION_STATUS",
    "RealAppH8AdmissionUnavailable",
    "RealAppH8Readiness",
    "RealAppWeightingParameters",
    "RealAppSemanticUniforms",
    "RealAppWeightedSupport",
    "RealAppSelectionResult",
    "RealAppH8SelectorSession",
    "real_app_h8_authority_manifest",
    "admit_lowry_r1h8_app_support",
)


def _finalize_wrapper_integrity() -> None:
    namespace = globals()
    if "_selector_capsule_state" in namespace:
        raise RealAppH8AdmissionUnavailable(
            "Admission wrapper namespace was already finalized"
        )
    temporary = frozenset(
        {
            "_SELECTOR_CAPSULE",
            "_make_selector_capsule_state",
            "_finalize_wrapper_integrity",
        }
    )
    namespace_keys = tuple(namespace)
    if any(type(key) is not str for key in namespace_keys):
        raise RealAppH8AdmissionUnavailable(
            "Admission wrapper namespace key type changed before finalization"
        )
    missing_warning_registry = object()
    warning_registry = namespace.get(
        "__warningregistry__", missing_warning_registry
    )
    if (
        warning_registry is not missing_warning_registry
        and type(warning_registry) is not dict
    ):
        raise RealAppH8AdmissionUnavailable(
            "Admission wrapper warning registry type changed before finalization"
        )
    expected_keys = frozenset(
        key
        for key in namespace_keys
        if not str.__eq__(key, "__warningregistry__") and key not in temporary
    ) | frozenset({"_selector_capsule_state"})
    expected_items = _MappingProxyType(
        {
            key: namespace[key]
            for key in namespace_keys
            if not str.__eq__(key, "__warningregistry__") and key not in temporary
        }
    )
    module_name = __name__
    wrapper_function_guards = tuple(
        _capture_function_integrity(value, f"wrapper.{name}")
        for name, value in tuple(namespace.items())
        if type(name) is str
        and type(value) is _FunctionType
        and type(value.__module__) is str
        and str.__eq__(value.__module__, module_name)
        and name not in {"_make_selector_capsule_state", "_finalize_wrapper_integrity"}
    )
    wrapper_class_guards = tuple(
        _capture_class_integrity(value, f"wrapper.{name}")
        for name, value in tuple(namespace.items())
        if type(name) is str
        and type(value) is type
        and type(value.__module__) is str
        and str.__eq__(value.__module__, module_name)
    )
    # The returned state function is the unavoidable self-check boundary.  All
    # wrapper-defined admission functions and classes it authenticates are
    # captured above before that one state function exists.
    namespace["_selector_capsule_state"] = _make_selector_capsule_state(
        _SELECTOR_CAPSULE,
        expected_keys,
        expected_items,
        wrapper_function_guards,
        wrapper_class_guards,
    )


_finalize_wrapper_integrity()
del _SELECTOR_CAPSULE, _make_selector_capsule_state, _finalize_wrapper_integrity
