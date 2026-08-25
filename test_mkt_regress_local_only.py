"""Regression checks for mkt_regress's isolated shadow-run mode."""

import ast
from pathlib import Path


SOURCE_PATH = Path(__file__).with_name("mkt_regress.py")
SOURCE = SOURCE_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE, filename=str(SOURCE_PATH))


def test_local_only_flag_is_declared_and_local_files_precede_delivery_guard():
    assert '"--local-only"' in SOURCE
    guard = next(
        node
        for node in TREE.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "_args.local_only"
    )

    local_writes = [
        node
        for node in TREE.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "to_csv"
    ]
    assert len(local_writes) >= 2
    assert all(node.lineno < guard.lineno for node in local_writes[-2:])


def test_onedrive_copy_and_skill_merge_are_only_in_non_local_branch():
    guard = next(
        node
        for node in TREE.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "_args.local_only"
    )
    local_source = "\n".join(ast.unparse(node) for node in guard.body)
    delivery_source = "\n".join(ast.unparse(node) for node in guard.orelse)

    assert "os.path.expanduser" not in local_source
    assert "subprocess.run" not in local_source
    assert "OneDrive" in delivery_source
    assert "skill_merge.py" in delivery_source
    assert "subprocess.run" in delivery_source
