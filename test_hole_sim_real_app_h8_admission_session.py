from __future__ import annotations

import copy
from hashlib import sha256
import inspect
import json
from pathlib import Path
import sys
from types import MappingProxyType, SimpleNamespace
from weakref import WeakKeyDictionary

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


ARCHIVE = Path(__file__).resolve().parent / "archive"
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

import hole_sim_real_app_h8_admission_session as admission


SELECTOR = ARCHIVE / "hole_sim_real_app_whole_atom_selector_lowry_r1h8.py"


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _closure_value(function, name: str):
    cells = dict(zip(function.__code__.co_freevars, function.__closure__, strict=True))
    return cells[name].cell_contents


def _synthetic_authority_record() -> dict[str, object]:
    def section(root, paths, directories, identity, contract):
        files = {
            path: {"bytes": 1, "sha256": "0" * 64}
            for path in paths
        }
        return {
            "root": root,
            "files": files,
            "directories": list(directories),
            "bytes": len(files),
            "inventory_contract": contract,
            "inventory_sha256": "1" * 64,
            "identity": dict(identity),
        }

    logic = {
        "reviewed_dormant_wrapper_sha256": sha256(
            admission._project_dormant_source(Path(admission.__file__).read_bytes())
        ).hexdigest(),
        "reviewed_dormant_test_sha256": _sha(Path(__file__)),
        "reviewed_record_test_sha256": _sha(
            Path(__file__).parent / admission._RECORD_TEST_RELATIVE
        ),
        "selector_capsule_sha256": admission._SELECTOR_CAPSULE_SHA256,
        "selector_source_sha256": admission._SELECTOR_SHA256,
        "selector_test_sha256": admission._SELECTOR_TEST_SHA256,
        "selector_runtime_sha256": admission._SELECTOR_RUNTIME_SHA256,
        "selector_support_sha256": admission._SELECTOR_SUPPORT_SHA256,
    }
    review_sources = dict(admission._KNOWN_SOURCE_SHA256)
    review_sources[admission._TEST_RELATIVE] = logic[
        "reviewed_dormant_test_sha256"
    ]
    review_sources[admission._RECORD_TEST_RELATIVE] = logic[
        "reviewed_record_test_sha256"
    ]
    record: dict[str, object] = {
        "record_contract": admission.REAL_APP_H8_RECORD_CONTRACT,
        "authority_id": "0" * 64,
        "scope": {
            "query_key": admission._QUERY_KEY,
            "support_key": admission._SUPPORT_KEY,
            "category": "app",
            "arm": "raw",
            "round_no": 1,
            "hole_no": 8,
            "start_lie": "tee",
            "target_player_id": "33204",
            "target_event_key": "pga:R2026034",
            "counts": dict(admission._COUNTS),
        },
        "logic": logic,
        "support_artifact": section(
            admission._SUPPORT_RELATIVE,
            admission._SUPPORT_FILES,
            admission._SUPPORT_DIRECTORIES,
            admission._SUPPORT_IDENTITY,
            "audit_v2_input_tree",
        ),
        "candidate_artifact": section(
            admission._CANDIDATE_RELATIVE,
            admission._CANDIDATE_FILES,
            (),
            admission._CANDIDATE_IDENTITY,
            "audit_v2_input_tree",
        ),
        "audit_artifact": section(
            admission._AUDIT_RELATIVE,
            admission._AUDIT_FILES,
            (),
            admission._AUDIT_IDENTITY,
            "audit_v2_output_tree",
        ),
        "review_sources": review_sources,
        "semantic_evidence": {
            "ndjson": {
                name: {
                    "contract": row[0],
                    "records": row[1],
                    "framed_bytes": row[2],
                    "sha256": row[3],
                }
                for name, row in admission._NDJSON.items()
            },
            "external": dict(admission._EXTERNAL_EVIDENCE),
        },
        "limits": dict(admission._LIMITS),
        "capabilities": dict(admission._CAPABILITIES),
    }
    body = dict(record)
    body.pop("authority_id")
    record["authority_id"] = admission._record_id(body)
    return record


def _reseal_record(record: dict[str, object]) -> None:
    body = dict(record)
    body.pop("authority_id")
    record["authority_id"] = admission._record_id(body)


def test_frozen_selector_is_unchanged_and_wrapper_compiles() -> None:
    assert _sha(SELECTOR) == (
        "f47c35d0345410bf165885e81afd8c0a0320f4950118889d2c4ac3fce6148bd0"
    )
    compile(Path(admission.__file__).read_bytes(), admission.__file__, "exec")


def test_selector_capsule_is_private_and_loader_is_one_shot() -> None:
    assert "_load_frozen_selector_capsule" not in vars(admission)
    assert "_finalize_wrapper_integrity" not in vars(admission)
    assert "_make_selector_capsule_state" not in vars(admission)
    selector = admission._SELECTOR_WEIGHT.__globals__
    assert selector["_weight_fixture"] is admission._SELECTOR_WEIGHT
    assert selector["__name__"].startswith("_lowry_r1h8_frozen_selector_")
    for _role, canonical, _relative in admission._FROZEN_SELECTOR_MODULE_SPECS:
        assert sys.modules.get(canonical) is not selector


def test_capsule_integrity_rejects_bound_function_replacement(monkeypatch) -> None:
    selector_globals = admission._SELECTOR_WEIGHT.__globals__
    monkeypatch.setitem(selector_globals, "_weight_fixture", lambda *_: None)
    with pytest.raises(
        admission.RealAppH8AdmissionUnavailable,
        match="selector .*changed",
    ):
        admission._selector_capsule_state()


def test_capsule_integrity_rejects_helper_code_mutation(monkeypatch) -> None:
    selector_globals = admission._SELECTOR_WEIGHT.__globals__
    helper = selector_globals["_validate_policy_and_groups"]
    monkeypatch.setattr(helper, "__code__", (lambda *_args: None).__code__)
    with pytest.raises(
        admission.RealAppH8AdmissionUnavailable,
        match="function implementation changed",
    ):
        admission._selector_capsule_state()


def test_capsule_integrity_rejects_mutable_dependency_constant() -> None:
    selector_globals = admission._SELECTOR_WEIGHT.__globals__
    numeric_columns = selector_globals["_chain_v3"]._NUMERIC_COLUMNS
    numeric_columns.add("attacker_column")
    try:
        with pytest.raises(
            admission.RealAppH8AdmissionUnavailable,
            match="module constant changed",
        ):
            admission._selector_capsule_state()
    finally:
        numeric_columns.remove("attacker_column")


def test_capsule_integrity_rejects_class_method_code_mutation(monkeypatch) -> None:
    equality = admission.RealAppWeightingParameters.__eq__
    monkeypatch.setattr(equality, "__code__", (lambda self, other: True).__code__)
    with pytest.raises(
        admission.RealAppH8AdmissionUnavailable,
        match="function implementation changed",
    ):
        admission._selector_capsule_state()


def test_capsule_integrity_rejects_dataclass_field_mutation() -> None:
    synthetic = admission._SELECTOR_WEIGHT.__globals__["_synthetic"]
    field = synthetic.CompactAppSupport.__dataclass_fields__["support_key"]
    original = field.name
    field.name = "attacker_support_key"
    try:
        with pytest.raises(
            admission.RealAppH8AdmissionUnavailable,
            match="class constant changed",
        ):
            admission._selector_capsule_state()
    finally:
        field.name = original


def test_capsule_integrity_rejects_guarded_builtins_mutation(monkeypatch) -> None:
    private_builtins = admission._SELECTOR_WEIGHT.__builtins__
    monkeypatch.setitem(private_builtins, "len", lambda _value: 0)
    with pytest.raises(
        admission.RealAppH8AdmissionUnavailable,
        match="builtins changed",
    ):
        admission._selector_capsule_state()


def test_capsule_integrity_rejects_runtime_typing_alias_mutation() -> None:
    policy = admission._SELECTOR_WEIGHT.__globals__["_policy"]
    mapping = policy.Mapping
    original = mapping.__origin__
    mapping.__origin__ = int
    try:
        with pytest.raises(
            admission.RealAppH8AdmissionUnavailable,
            match="semantic object changed",
        ):
            admission._selector_capsule_state()
    finally:
        mapping.__origin__ = original


def test_capsule_integrity_rejects_runtime_typing_type_mutation() -> None:
    policy = admission._SELECTOR_WEIGHT.__globals__["_policy"]
    alias_type = type(policy.Mapping)
    missing = object()
    original = vars(alias_type).get("__instancecheck__", missing)
    alias_type.__instancecheck__ = lambda _self, _value: True
    try:
        with pytest.raises(
            admission.RealAppH8AdmissionUnavailable,
            match="class namespace changed",
        ):
            admission._selector_capsule_state()
    finally:
        if original is missing:
            delattr(alias_type, "__instancecheck__")
        else:
            alias_type.__instancecheck__ = original


def test_capsule_integrity_rejects_checker_rebinding(monkeypatch) -> None:
    monkeypatch.setattr(admission, "_check_module_integrity", lambda _guard: None)
    with pytest.raises(
        admission.RealAppH8AdmissionUnavailable,
        match="capsule checker changed",
    ):
        admission._selector_capsule_state()


@pytest.mark.parametrize(
    "helper_name",
    ("read_file", "regular", "directory", "stat_signature", "capture_layout"),
)
def test_capsule_integrity_recursively_guards_low_level_helper_code(
    helper_name: str,
) -> None:
    helper = admission._FILESYSTEM_HELPERS[helper_name]
    original = helper.__code__
    helper.__code__ = original.replace()
    try:
        with pytest.raises(
            admission.RealAppH8AdmissionUnavailable,
            match=rf"function implementation changed: capsule\.filesystem\.{helper_name}",
        ):
            admission._selector_capsule_state()
    finally:
        helper.__code__ = original


def test_filesystem_guard_metadata_is_deeply_immutable_and_cannot_be_resealed() -> None:
    guards = _closure_value(
        admission._selector_capsule_state, "filesystem_function_guards"
    )
    guard = next(
        value
        for value in guards
        if value["label"] == "capsule.filesystem.is_reparse"
    )
    assert type(guard) is MappingProxyType
    helper = admission._FILESYSTEM_HELPERS["is_reparse"]
    original = helper.__code__
    changed = original.replace()
    helper.__code__ = changed
    try:
        with pytest.raises(TypeError):
            guard["code"] = changed
        with pytest.raises(
            admission.RealAppH8AdmissionUnavailable,
            match="function implementation changed: capsule.filesystem.is_reparse",
        ):
            admission._selector_capsule_state()
    finally:
        helper.__code__ = original


def test_capsule_integrity_rejects_added_wrapper_global(monkeypatch) -> None:
    assert "_forwarding_global" not in vars(admission)
    monkeypatch.setattr(admission, "_forwarding_global", object(), raising=False)
    with pytest.raises(
        admission.RealAppH8AdmissionUnavailable,
        match="wrapper namespace changed",
    ):
        admission._selector_capsule_state()


def test_capsule_integrity_rejects_existing_wrapper_binding_replacement(
    monkeypatch,
) -> None:
    monkeypatch.setattr(admission, "_source_state", lambda _record: {})
    with pytest.raises(
        admission.RealAppH8AdmissionUnavailable,
        match="wrapper binding changed",
    ):
        admission._selector_capsule_state()


def test_capsule_state_is_included_in_its_own_immutable_binding_seal(
    monkeypatch,
) -> None:
    saved_state = admission._selector_capsule_state
    expected_items = _closure_value(saved_state, "expected_wrapper_items")
    assert type(expected_items) is MappingProxyType
    assert expected_items["_selector_capsule_state"] is saved_state
    with pytest.raises(TypeError):
        expected_items["_selector_capsule_state"] = lambda: {}
    monkeypatch.setattr(admission, "_selector_capsule_state", lambda: {})
    with pytest.raises(
        admission.RealAppH8AdmissionUnavailable,
        match="wrapper binding changed",
    ):
        saved_state()


def test_wrapper_namespace_rejects_hostile_non_string_key_without_equality() -> None:
    calls: list[object] = []

    class HostileKey:
        def __hash__(self) -> int:
            return 928_374_657

        def __eq__(self, other: object) -> bool:
            calls.append(other)
            return other == "__warningregistry__"

    key = HostileKey()
    admission.__dict__[key] = object()
    try:
        with pytest.raises(
            admission.RealAppH8AdmissionUnavailable,
            match="namespace key type changed",
        ):
            admission._selector_capsule_state()
    finally:
        admission.__dict__.pop(key)
    assert calls == []


def test_private_module_namespace_rejects_hostile_key_without_equality() -> None:
    calls: list[object] = []

    class HostileKey:
        def __hash__(self) -> int:
            return 472_903_861

        def __eq__(self, other: object) -> bool:
            calls.append(other)
            return other == "__warningregistry__"

    namespace = admission._SELECTOR_WEIGHT.__globals__
    key = HostileKey()
    namespace[key] = object()
    try:
        with pytest.raises(
            admission.RealAppH8AdmissionUnavailable,
            match="module namespace changed",
        ):
            admission._selector_capsule_state()
    finally:
        namespace.pop(key)
    assert calls == []


def test_wrapper_warning_registry_must_be_an_exact_dict(monkeypatch) -> None:
    monkeypatch.setitem(admission.__dict__, "__warningregistry__", object())
    with pytest.raises(
        admission.RealAppH8AdmissionUnavailable,
        match="warning registry type changed",
    ):
        admission._selector_capsule_state()


def test_capsule_integrity_rejects_wrapper_builtin_mutation(monkeypatch) -> None:
    monkeypatch.setitem(admission.__dict__["__builtins__"], "object", lambda: None)
    with pytest.raises(
        admission.RealAppH8AdmissionUnavailable,
        match="wrapper builtins changed",
    ):
        admission._selector_capsule_state()


def test_capsule_integrity_rejects_global_reparse_rebinding(monkeypatch) -> None:
    monkeypatch.setattr(admission, "_is_reparse", lambda _value: False)
    assert admission._SELECTOR_VALIDATE_RUNTIME() == admission._SELECTOR_RUNTIME_SHA256
    with pytest.raises(
        admission.RealAppH8AdmissionUnavailable,
        match="capsule checker changed",
    ):
        admission._selector_capsule_state()


@pytest.mark.parametrize(
    "method_name",
    ("is_symlink", "lstat", "open", "resolve", "__truediv__"),
)
def test_capsule_integrity_rejects_path_mro_method_mutation_before_use(
    monkeypatch,
    method_name: str,
) -> None:
    concrete = type(Path(".").resolve())
    owner = next(
        path_class
        for path_class in concrete.__mro__
        if method_name in vars(path_class)
    )
    original = vars(owner)[method_name]
    calls: list[str] = []

    def delegating(self, *args, **kwargs):
        calls.append(method_name)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(owner, method_name, delegating)
    try:
        with pytest.raises(
            admission.RealAppH8AdmissionUnavailable,
            match="pathlib collaborator",
        ):
            admission._selector_capsule_state()
    finally:
        monkeypatch.undo()
    assert calls == []


@pytest.mark.parametrize(
    ("collaborator", "method_name"),
    (
        ("accessor", "open"),
        ("accessor", "stat"),
        ("accessor", "realpath"),
        ("flavour", "splitroot"),
        ("flavour", "casefold"),
        ("flavour", "casefold_parts"),
    ),
)
def test_pathlib_collaborator_replacements_are_never_invoked_and_fail_closed(
    monkeypatch,
    collaborator: str,
    method_name: str,
) -> None:
    record = _synthetic_authority_record()
    concrete = type(Path(".").resolve())
    instance = concrete._accessor if collaborator == "accessor" else concrete._flavour
    owner = type(instance)
    calls: list[str] = []

    def replacement(*_args, **_kwargs):
        calls.append(method_name)
        raise AssertionError("pathlib replacement executed")

    monkeypatch.setattr(owner, method_name, replacement)
    try:
        for operation in (
                admission._selector_capsule_state,
                admission._SELECTOR_VALIDATE_RUNTIME,
                lambda: admission._source_state(record),
        ):
            with pytest.raises(
                admission.RealAppH8AdmissionUnavailable,
                match="pathlib",
            ):
                operation()
    finally:
        monkeypatch.undo()
    assert calls == []


def test_pathlib_global_replacement_is_never_invoked_and_fails_closed(
    monkeypatch,
) -> None:
    record = _synthetic_authority_record()
    calls: list[str] = []

    def replacement(*_args, **_kwargs):
        calls.append("S_ISLNK")
        raise AssertionError("pathlib global replacement executed")

    monkeypatch.setattr(admission._pathlib, "S_ISLNK", replacement)
    try:
        for operation in (
            admission._selector_capsule_state,
            admission._SELECTOR_VALIDATE_RUNTIME,
            lambda: admission._source_state(record),
        ):
            with pytest.raises(
                admission.RealAppH8AdmissionUnavailable,
                match="pathlib",
            ):
                operation()
    finally:
        monkeypatch.undo()
    assert calls == []


@pytest.mark.parametrize("descriptor_kind", ("staticmethod", "classmethod", "property"))
def test_pathlib_descriptor_function_code_is_sealed(
    descriptor_kind: str,
) -> None:
    concrete = type(Path(".").resolve())
    owners = tuple(
        value for value in concrete.__mro__ if value is not object
    ) + (type(concrete._accessor), type(concrete._flavour))
    descriptor_type = {
        "staticmethod": staticmethod,
        "classmethod": classmethod,
        "property": property,
    }[descriptor_kind]
    function = None
    for owner in owners:
        for descriptor in vars(owner).values():
            if type(descriptor) is not descriptor_type:
                continue
            candidates = (
                (descriptor.fget, descriptor.fset, descriptor.fdel)
                if descriptor_kind == "property"
                else (descriptor.__func__,)
            )
            function = next(
                (candidate for candidate in candidates if inspect.isfunction(candidate)),
                None,
            )
            if function is not None:
                break
        if function is not None:
            break
    assert function is not None
    original = function.__code__
    function.__code__ = original.replace()
    try:
        with pytest.raises(
            admission.RealAppH8AdmissionUnavailable,
            match="pathlib descriptor changed",
        ):
            admission._selector_capsule_state()
    finally:
        function.__code__ = original


def test_pathlib_mutable_reserved_names_content_is_sealed() -> None:
    flavour_type = type(type(Path(".").resolve())._flavour)
    reserved_names = vars(flavour_type)["reserved_names"]
    marker = "CODEX_ATTACKER_RESERVED_NAME"
    assert type(reserved_names) is set and marker not in reserved_names
    reserved_names.add(marker)
    try:
        with pytest.raises(
            admission.RealAppH8AdmissionUnavailable,
            match=(
                "function closure changed: capsule.filesystem.state|"
                "pathlib class constant changed"
            ),
        ):
            admission._selector_capsule_state()
    finally:
        reserved_names.remove(marker)


@pytest.mark.parametrize("collaborator", ("accessor", "flavour"))
def test_pathlib_collaborator_instance_dictionary_is_sealed(
    collaborator: str,
) -> None:
    concrete = type(Path(".").resolve())
    instance = concrete._accessor if collaborator == "accessor" else concrete._flavour
    state = object.__getattribute__(instance, "__dict__")
    assert type(state) is dict
    if collaborator == "flavour":
        name = "join"
        original = state[name]
        state[name] = object()
    else:
        name = "_codex_attacker"
        assert name not in state
        original = None
        state[name] = object()
    try:
        with pytest.raises(
            admission.RealAppH8AdmissionUnavailable,
            match="pathlib instance state changed",
        ):
            admission._selector_capsule_state()
    finally:
        if collaborator == "flavour":
            state[name] = original
        else:
            state.pop(name)


@pytest.mark.parametrize(
    "function_name",
    (
        "_source_state",
        "_capture_tree",
        "_evidence_state",
        "_live_state",
        "admit_lowry_r1h8_app_support",
    ),
)
def test_wrapper_admission_call_graph_function_code_is_sealed(
    function_name: str,
) -> None:
    function = vars(admission)[function_name]
    original = function.__code__
    function.__code__ = original.replace()
    try:
        with pytest.raises(
            admission.RealAppH8AdmissionUnavailable,
            match=rf"function implementation changed: wrapper\.{function_name}",
        ):
            admission._selector_capsule_state()
    finally:
        function.__code__ = original


@pytest.mark.parametrize("method_name", ("readiness", "weight"))
def test_wrapper_session_method_code_is_sealed(method_name: str) -> None:
    descriptor = vars(admission.RealAppH8SelectorSession)[method_name]
    function = descriptor.fget if type(descriptor) is property else descriptor
    assert inspect.isfunction(function)
    original = function.__code__
    function.__code__ = original.replace()
    try:
        with pytest.raises(
            admission.RealAppH8AdmissionUnavailable,
            match=(
                "function implementation changed: "
                rf"wrapper\.RealAppH8SelectorSession\.{method_name}"
            ),
        ):
            admission._selector_capsule_state()
    finally:
        function.__code__ = original


def test_wrapper_call_graph_guard_metadata_is_deeply_immutable() -> None:
    function_guards = _closure_value(
        admission._selector_capsule_state, "wrapper_function_guards"
    )
    class_guards = _closure_value(
        admission._selector_capsule_state, "wrapper_class_guards"
    )
    expected_items = _closure_value(
        admission._selector_capsule_state, "expected_wrapper_items"
    )
    assert type(function_guards) is tuple
    assert type(class_guards) is tuple
    assert all(type(guard) is MappingProxyType for guard in function_guards)
    assert all(type(guard) is MappingProxyType for guard in class_guards)
    assert type(expected_items) is MappingProxyType
    source_guard = next(
        guard for guard in function_guards if guard["label"] == "wrapper._source_state"
    )
    session_guard = next(
        guard
        for guard in class_guards
        if guard["label"] == "wrapper.RealAppH8SelectorSession"
    )
    with pytest.raises(TypeError):
        source_guard["code"] = admission._source_state.__code__
    with pytest.raises(TypeError):
        session_guard["items"] = MappingProxyType({})
    with pytest.raises(TypeError):
        expected_items["_source_state"] = admission._source_state
    assert type(session_guard["items"]) is MappingProxyType
    assert type(session_guard["mutable"]) is MappingProxyType
    assert type(session_guard["semantic"]) is MappingProxyType
    assert type(session_guard["functions"]) is tuple


def test_manifest_is_exactly_scoped_in_both_registry_states() -> None:
    manifest = admission.real_app_h8_authority_manifest()
    if manifest["authority_record_present"] is False:
        assert manifest["records"] == []
        assert manifest["real_support_admitted"] is False
        assert manifest["weighting_enabled"] is False
        assert manifest["selection_enabled"] is False
        assert manifest["authority_lock_sha256"] == admission._EMPTY_AUTHORITY_LOCK
    else:
        assert len(manifest["records"]) == 1
    assert manifest["callback_enabled"] is False
    assert manifest["chain_ready"] is False
    assert manifest["anchor_ready"] is False
    assert manifest["mean_solve_launched"] is False
    assert manifest["simulation_launched"] is False
    assert manifest["production_enabled"] is False
    assert manifest["research_only"] is True


def test_zero_argument_admission_fails_before_every_read(monkeypatch) -> None:
    assert len(inspect.signature(admission.admit_lowry_r1h8_app_support).parameters) == 0
    if admission.real_app_h8_authority_manifest()["authority_record_present"] is True:
        return
    calls: list[str] = []

    def bomb(name: str):
        def inner(*_args, **_kwargs):
            calls.append(name)
            raise AssertionError(name)

        return inner

    monkeypatch.setattr(admission, "_source_state", bomb("source"))
    monkeypatch.setattr(admission, "_evidence_state", bomb("artifact"))
    monkeypatch.setattr(admission, "_load_frame", bomb("parquet"))
    with pytest.raises(
        admission.RealAppH8AdmissionUnavailable,
        match="No reviewed real-H8 selector authority",
    ):
        admission.admit_lowry_r1h8_app_support()
    assert calls == []
    with pytest.raises(TypeError):
        admission.admit_lowry_r1h8_app_support("candidate")
    with pytest.raises(TypeError):
        admission.admit_lowry_r1h8_app_support(path="candidate")


def test_manifest_also_performs_no_source_or_artifact_read(monkeypatch) -> None:
    monkeypatch.setattr(
        admission,
        "_source_state",
        lambda *_: (_ for _ in ()).throw(AssertionError("source read")),
    )
    monkeypatch.setattr(
        admission,
        "_evidence_state",
        lambda *_: (_ for _ in ()).throw(AssertionError("artifact read")),
    )
    manifest = admission.real_app_h8_authority_manifest()
    assert manifest["authority_record_present"] is bool(manifest["records"])


def test_authority_gate_has_no_mutable_globals_or_default_authority(monkeypatch) -> None:
    assert "_AUTHORITY_RECORDS" not in vars(admission)
    assert "_AUTHORITY_LOCK_SHA256" not in vars(admission)
    assert admission._closed_authority.__defaults__ is None
    assert admission._closed_authority.__kwdefaults__ is None
    expected_record, expected_lock = admission._closed_authority()
    fake = MappingProxyType({"record_contract": admission.REAL_APP_H8_RECORD_CONTRACT})
    records = (fake,)
    lock = admission._stable_sha(admission._authority_payload(records))
    monkeypatch.setattr(admission, "_AUTHORITY_RECORDS", records, raising=False)
    monkeypatch.setattr(admission, "_AUTHORITY_LOCK_SHA256", lock, raising=False)
    record, observed_lock = admission._closed_authority()
    assert admission._exact_tree_equal(record, expected_record)
    assert observed_lock == expected_lock


def test_session_gate_has_no_exposed_registry_token_or_mint() -> None:
    assert admission._session_seal.__defaults__ is None
    assert admission._session_seal.__kwdefaults__ is None
    assert "_SESSIONS" not in vars(admission)
    assert "_SESSION_TOKEN" not in vars(admission)
    assert "_mint" not in vars(admission)


def test_session_gate_has_no_mutable_closure_registry() -> None:
    assert "_WeakKeyDictionary" not in vars(admission)
    for function in (
        admission._session_seal,
        admission.admit_lowry_r1h8_app_support,
    ):
        contents = tuple(cell.cell_contents for cell in function.__closure__ or ())
        assert not any(isinstance(value, WeakKeyDictionary) for value in contents)


def test_forged_session_cannot_install_an_arbitrary_receipt() -> None:
    forged = object.__new__(admission.RealAppH8SelectorSession)
    with pytest.raises(
        admission.RealAppH8AdmissionUnavailable,
        match="was not minted",
    ):
        admission._session_seal(forged)
    forged_seal = admission._SessionSeal(
        authority_id="a" * 64,
        record_sha256="b" * 64,
        authority_lock_sha256="c" * 64,
        wrapper_sha256="d" * 64,
    )

    def fake_receipt(_candidate):
        return forged_seal

    object.__setattr__(forged, "_receipt", fake_receipt)
    with pytest.raises(
        admission.RealAppH8AdmissionUnavailable,
        match="receipt implementation changed",
    ):
        admission._session_seal(forged)


def test_session_receipt_documents_the_practical_pure_python_boundary() -> None:
    assert admission._SESSION_RECEIPT_BOUNDARY == (
        "identity_bound_exact_code_and_closure_no_mutable_registry_"
        "arbitrary_function_cell_forgery_outside_pure_python_boundary"
    )
    receipt_code = _closure_value(admission._session_seal, "receipt_code")
    assert type(receipt_code) is type(admission._session_seal.__code__)
    assert receipt_code.co_freevars == ("error_type", "seal", "session")


def test_authority_tree_is_deeply_immutable() -> None:
    frozen = admission._freeze_tree(
        {"outer": [{"inner": "value"}], "flag": False, "count": 1}
    )
    assert type(frozen) is MappingProxyType
    with pytest.raises(TypeError):
        frozen["flag"] = True
    with pytest.raises(TypeError):
        frozen["outer"][0]["inner"] = "changed"
    assert admission._plain_tree(frozen) == {
        "outer": [{"inner": "value"}],
        "flag": False,
        "count": 1,
    }
    with pytest.raises(admission.RealAppH8AdmissionUnavailable):
        admission._freeze_tree({"float": 1.0})


def test_marked_block_is_unique_literal_only_and_projects_to_current_bytes() -> None:
    raw = Path(admission.__file__).read_bytes()
    assert raw.count(admission._BLOCK_BEGIN) == 1
    assert raw.count(admission._BLOCK_END) == 1
    projected = admission._project_dormant_source(raw)
    if admission.real_app_h8_authority_manifest()["authority_record_present"] is False:
        assert projected == raw
    else:
        assert projected != raw
        assert admission._project_dormant_source(projected) == projected


@pytest.mark.parametrize("attack", ["duplicate", "missing", "call", "extra"])
def test_marked_block_attacks_fail_closed(attack: str) -> None:
    raw = Path(admission.__file__).read_bytes()
    start = raw.index(admission._BLOCK_BEGIN)
    end_marker = raw.index(admission._BLOCK_END)
    end = raw.index(b"\n", end_marker + len(admission._BLOCK_END)) + 1
    block = raw[start:end]
    if attack == "duplicate":
        changed = raw + admission._BLOCK_BEGIN
    elif attack == "missing":
        changed = raw.replace(admission._BLOCK_END, b"", 1)
    elif attack == "call":
        records_start = block.index(b"_AUTHORITY_RECORD_LITERALS = ")
        lock_start = block.index(b"\n_AUTHORITY_LOCK_SHA256 = ", records_start) + 1
        attacked = (
            block[:records_start]
            + b"_AUTHORITY_RECORD_LITERALS = tuple()\n"
            + block[lock_start:]
        )
        changed = raw[:start] + attacked + raw[end:]
    else:
        marker_start = block.rindex(b"\n" + admission._BLOCK_END)
        attacked = block[:marker_start] + b"\npass" + block[marker_start:]
        changed = raw[:start] + attacked + raw[end:]
    with pytest.raises(admission.RealAppH8AdmissionUnavailable):
        admission._project_dormant_source(changed)


def test_session_construction_subclass_and_object_new_forgery_reject() -> None:
    with pytest.raises(admission.RealAppH8AdmissionUnavailable):
        admission.RealAppH8SelectorSession()

    class Hostile(admission.RealAppH8SelectorSession):
        pass

    with pytest.raises(admission.RealAppH8AdmissionUnavailable):
        Hostile()
    forged = object.__new__(admission.RealAppH8SelectorSession)
    with pytest.raises(
        admission.RealAppH8AdmissionUnavailable,
        match="was not minted",
    ):
        _ = forged.readiness


def test_public_namespace_is_closed_and_has_no_runtime_widening() -> None:
    assert type(admission.__all__) is tuple
    public = {name for name in vars(admission) if not name.startswith("_")}
    assert public == set(admission.__all__)
    forbidden = {
        "callback",
        "run_chain",
        "solve",
        "simulate",
        "production",
        "load_path",
        "authority_digest",
    }
    assert not (forbidden & public)
    methods = {
        name
        for name in vars(admission.RealAppH8SelectorSession)
        if not name.startswith("_")
    }
    assert methods == {"readiness", "weight", "select"}


def test_public_export_tuple_cannot_be_widened(monkeypatch) -> None:
    assert type(admission.__all__) is tuple
    with pytest.raises(AttributeError):
        admission.__all__.append("simulate")
    saved_state = admission._selector_capsule_state
    monkeypatch.setattr(admission, "__all__", admission.__all__ + ("simulate",))
    with pytest.raises(
        admission.RealAppH8AdmissionUnavailable,
        match="wrapper binding changed",
    ):
        saved_state()


def test_strict_json_rejects_duplicate_and_nonfinite_values() -> None:
    with pytest.raises(admission.RealAppH8AdmissionUnavailable):
        admission._strict_json(b'{"a":1,"a":2}', "duplicate")
    with pytest.raises(admission.RealAppH8AdmissionUnavailable):
        admission._strict_json(b'{"a":NaN}', "nonfinite")
    assert admission._strict_json(b'{"a":1}', "good") == {"a": 1}


def test_exact_tree_comparison_never_aliases_bool_and_int() -> None:
    assert admission._exact_tree_equal({"enabled": False}, {"enabled": False})
    assert not admission._exact_tree_equal({"enabled": 0}, {"enabled": False})
    assert not admission._exact_tree_equal({"count": True}, {"count": 1})


def test_tree_capture_is_exact_and_rejects_widening(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    (root / "sub").mkdir(parents=True)
    (root / "a.bin").write_bytes(b"a")
    (root / "sub" / "b.bin").write_bytes(b"bb")
    expected = {
        "a.bin": {"bytes": 1, "sha256": sha256(b"a").hexdigest()},
        "sub/b.bin": {"bytes": 2, "sha256": sha256(b"bb").hexdigest()},
    }
    captured, state = admission._capture_tree(
        str(root), expected, ("sub",), "fixture", "audit_v2_input_tree"
    )
    assert captured == {"a.bin": b"a", "sub/b.bin": b"bb"}
    assert state["files"] == 2
    assert state["directories"] == 1
    assert state["bytes"] == 3

    (root / "extra.bin").write_bytes(b"extra")
    with pytest.raises(
        admission.RealAppH8AdmissionUnavailable,
        match="file closure",
    ):
        admission._capture_tree(
            str(root), expected, ("sub",), "fixture", "audit_v2_input_tree"
        )


def test_tree_capture_rejects_extra_or_oversized_before_any_read(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    expected = {"a.bin": {"bytes": 1, "sha256": sha256(b"a").hexdigest()}}
    (root / "a.bin").write_bytes(b"a")
    (root / "extra.bin").write_bytes(b"extra")
    monkeypatch.setattr(
        admission,
        "_read_expected_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("read")),
    )
    with pytest.raises(admission.RealAppH8AdmissionUnavailable, match="file closure"):
        admission._capture_tree(
            str(root), expected, (), "fixture", "audit_v2_input_tree"
        )
    oversized_root = tmp_path / "oversized"
    oversized_root.mkdir()
    (oversized_root / "a.bin").write_bytes(b"too large")
    with pytest.raises(admission.RealAppH8AdmissionUnavailable, match="byte size"):
        admission._capture_tree(
            str(oversized_root), expected, (), "fixture", "audit_v2_input_tree"
        )


def test_review_source_hash_uses_only_the_capped_captured_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "review.py"
    source.write_bytes(b"review")
    observed: list[tuple[str, int, str]] = []

    def capture(path: str, maximum: int, label: str) -> bytes:
        observed.append((path, maximum, label))
        return b"review"

    monkeypatch.setattr(admission, "_read_capped_file", capture)
    monkeypatch.setattr(
        admission,
        "_sha256_file",
        lambda *_args: (_ for _ in ()).throw(AssertionError("uncapped hash")),
    )
    assert admission._closed_file_sha(str(source), "review source") == sha256(
        b"review"
    ).hexdigest()
    assert observed == [
        (str(source), admission._MAX_REVIEW_SOURCE_BYTES, "review source")
    ]


@pytest.mark.parametrize(
    "primitive",
    ("lstat", "open", "read", "close", "fstat", "listdir"),
)
def test_low_level_filesystem_primitives_are_captured_and_fail_closed(
    tmp_path: Path, monkeypatch, primitive: str
) -> None:
    record = _synthetic_authority_record()
    source = tmp_path / "review.py"
    source.write_bytes(b"review")
    root = tmp_path / "tree"
    root.mkdir()
    (root / "a.bin").write_bytes(b"a")
    expected = {"a.bin": {"bytes": 1, "sha256": sha256(b"a").hexdigest()}}
    original = getattr(admission._os, primitive)
    calls: list[int] = []

    def replacement(*_args, **_kwargs):
        calls.append(1)
        raise AssertionError(f"rebound os.{primitive} executed")

    monkeypatch.setattr(admission._os, primitive, replacement)
    try:
        assert admission._read_capped_file(str(source), 64, "review source") == b"review"
        captured, _state = admission._capture_tree(
            str(root), expected, (), "fixture", "audit_v2_input_tree"
        )
        assert captured == {"a.bin": b"a"}
        for operation in (
            admission._selector_capsule_state,
            admission._SELECTOR_VALIDATE_RUNTIME,
            lambda: admission._source_state(record),
        ):
            with pytest.raises(
                admission.RealAppH8AdmissionUnavailable,
                match="filesystem",
            ):
                operation()
    finally:
        monkeypatch.setattr(admission._os, primitive, original)
    assert calls == []


def test_review_source_size_cap_is_enforced(tmp_path: Path) -> None:
    source = tmp_path / "review.py"
    source.write_bytes(b"abcde")
    with pytest.raises(
        admission.RealAppH8AdmissionUnavailable,
        match="bounded regular file",
    ):
        admission._read_capped_file(str(source), 4, "review source")


def test_selector_runtime_hash_uses_lexically_captured_bounded_reader(
    monkeypatch,
) -> None:
    bounded_hash = admission._SELECTOR_WEIGHT.__globals__["_sha256_file"]
    closure = tuple(cell.cell_contents for cell in bounded_hash.__closure__)
    assert any(
        value is admission._FILESYSTEM_READ_CAPPED
        for value in closure
    )
    assert admission._MAX_REVIEW_SOURCE_BYTES in closure
    monkeypatch.setattr(
        admission,
        "_read_capped_file",
        lambda *_args: (_ for _ in ()).throw(AssertionError("global reader")),
    )
    assert admission._SELECTOR_VALIDATE_RUNTIME() == admission._SELECTOR_RUNTIME_SHA256
    with pytest.raises(
        admission.RealAppH8AdmissionUnavailable,
        match="capsule checker changed",
    ):
        admission._selector_capsule_state()


def test_weight_and_select_require_the_end_integrity_state(monkeypatch) -> None:
    session = object.__new__(admission.RealAppH8SelectorSession)
    frame = object()
    before = {"sealed": True}
    monkeypatch.setattr(admission, "_live_frame", lambda _session: ({}, frame, before))
    monkeypatch.setattr(
        admission,
        "_SELECTOR_WEIGHT",
        lambda *_args: SimpleNamespace(weighted="weighted"),
    )
    monkeypatch.setattr(admission, "_SELECTOR_SELECT", lambda *_args: "selected")
    monkeypatch.setattr(admission, "_integrity_state", lambda _session: {"sealed": False})
    with pytest.raises(admission.RealAppH8AdmissionUnavailable, match="during weighting"):
        session.weight(object())
    with pytest.raises(admission.RealAppH8AdmissionUnavailable, match="during selection"):
        session.select(object(), object())


def test_live_state_binds_authority_id_and_lock(monkeypatch) -> None:
    session = object.__new__(admission.RealAppH8SelectorSession)
    record = {"authority_id": "a" * 64}
    record_sha = admission._record_sha(record)
    monkeypatch.setattr(
        admission,
        "_session_seal",
        lambda _session: admission._SessionSeal(
            authority_id="b" * 64,
            record_sha256=record_sha,
            authority_lock_sha256=admission._EMPTY_AUTHORITY_LOCK,
            wrapper_sha256="c" * 64,
        ),
    )
    monkeypatch.setattr(
        admission,
        "_closed_authority",
        lambda: (record, admission._EMPTY_AUTHORITY_LOCK),
    )
    with pytest.raises(
        admission.RealAppH8AdmissionUnavailable, match="authority identity"
    ):
        admission._live_state(session)


def test_admission_recloses_authority_after_all_input_checks(monkeypatch) -> None:
    record = {"authority_id": "a" * 64}
    calls = 0

    def authority():
        nonlocal calls
        calls += 1
        if calls == 1:
            return record, admission._EMPTY_AUTHORITY_LOCK
        return None, admission._EMPTY_AUTHORITY_LOCK

    monkeypatch.setattr(admission, "_closed_authority", authority)
    monkeypatch.setattr(
        admission, "_source_state", lambda _record: {"wrapper_sha256": "c" * 64}
    )
    monkeypatch.setattr(
        admission, "_evidence_state", lambda _record: ({}, {"sealed": True})
    )
    monkeypatch.setattr(admission, "_load_frame", lambda _support: object())
    with pytest.raises(admission.RealAppH8AdmissionUnavailable, match="was removed"):
        admission.admit_lowry_r1h8_app_support()


def test_tree_capture_uses_captured_reparse_check(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "a.bin").write_bytes(b"a")
    monkeypatch.setattr(admission, "_is_reparse", lambda _stat: True)
    captured, _state = admission._capture_tree(
        str(root),
        {"a.bin": {"bytes": 1, "sha256": sha256(b"a").hexdigest()}},
        (),
        "fixture",
        "audit_v2_input_tree",
    )
    assert captured == {"a.bin": b"a"}
    with pytest.raises(
        admission.RealAppH8AdmissionUnavailable,
        match="capsule checker changed",
    ):
        admission._selector_capsule_state()


def _parquet_bytes(values: np.ndarray) -> bytes:
    table = pa.table({"proposal_row_order": pa.array(values)})
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return sink.getvalue().to_pybytes()


def test_part_loader_uses_exactly_five_buffer_readers(monkeypatch) -> None:
    rows = (8192, 8192, 8192, 8192, 2450)
    support: dict[str, bytes] = {}
    start = 0
    for path, count in zip(admission._PART_FILES, rows, strict=True):
        values = np.arange(start, start + count, dtype=np.int64) % 904
        support[path] = _parquet_bytes(values)
        start += count

    observed: list[object] = []
    real_read = admission._pq.read_table

    def read(where, *args, **kwargs):
        observed.append(where)
        assert type(where) is pa.BufferReader
        return real_read(where, *args, **kwargs)

    monkeypatch.setattr(admission._pq, "read_table", read)
    monkeypatch.setattr(admission, "_SELECTOR_CANONICALIZE", lambda frame: frame)
    monkeypatch.setattr(
        admission,
        "_SELECTOR_SUPPORT_SHA",
        lambda _frame: admission._SELECTOR_SUPPORT_SHA256,
    )
    frame = admission._load_frame(support)
    assert len(frame) == 35_218
    assert frame["proposal_row_order"].nunique() == 904
    assert len(observed) == 5


def test_record_schema_rejects_candidate_or_audit_as_authority() -> None:
    base = {
        "record_contract": admission.REAL_APP_H8_RECORD_CONTRACT,
        "authority_id": (
            "b5552aaea4e7c216b8bae8dc9ccdb68d122df54523d7b8faf1ddb70664dde942"
        ),
    }
    with pytest.raises(admission.RealAppH8AdmissionUnavailable):
        admission._validate_record(base)


def test_complete_synthetic_record_schema_and_review_test_binding() -> None:
    record = _synthetic_authority_record()
    assert admission._validate_record(record)["authority_id"] == record["authority_id"]

    changed = copy.deepcopy(record)
    changed["review_sources"][admission._TEST_RELATIVE] = "f" * 64
    _reseal_record(changed)
    with pytest.raises(
        admission.RealAppH8AdmissionUnavailable,
        match="Dormant wrapper test binding",
    ):
        admission._validate_record(changed)

    changed = copy.deepcopy(record)
    changed["review_sources"][admission._RECORD_TEST_RELATIVE] = "e" * 64
    _reseal_record(changed)
    with pytest.raises(
        admission.RealAppH8AdmissionUnavailable,
        match="Authority-record test binding",
    ):
        admission._validate_record(changed)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("scope", "round_no"), True),
        (("scope", "counts", "parts"), True),
        (("capabilities", "weighting"), 1),
        (("limits", "parts"), True),
        (("semantic_evidence", "ndjson", "schema", "records"), True),
    ],
)
def test_record_nested_exact_types_reject_bool_int_aliases(path, value) -> None:
    record = _synthetic_authority_record()
    node = record
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    _reseal_record(record)
    with pytest.raises(admission.RealAppH8AdmissionUnavailable):
        admission._validate_record(record)
