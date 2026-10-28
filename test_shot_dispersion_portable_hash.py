import hashlib
from pathlib import Path

import pytest

from shot_dispersion_overlay import _sha256


def test_portable_hash_accepts_windows_and_unix_lines(tmp_path):
    path = tmp_path / "input.csv"
    path.write_bytes(b"player,std\r\nalpha,1.2\r\n")
    expected = hashlib.sha256(b"player,std\nalpha,1.2\n").hexdigest()
    assert _sha256(path, "sha256_lf_normalized_v1") == expected
    assert _sha256(path) != expected
    path.write_bytes(b"player,std\nalpha,1.2\n")
    assert _sha256(path, "sha256_lf_normalized_v1") == expected


def test_portable_hash_still_rejects_changed_values(tmp_path):
    path = tmp_path / "input.csv"
    path.write_bytes(b"alpha,1.2\n")
    original = _sha256(path, "sha256_lf_normalized_v1")
    path.write_bytes(b"alpha,1.3\r\n")
    assert _sha256(path, "sha256_lf_normalized_v1") != original


def test_unknown_hash_mode_fails(tmp_path):
    path = tmp_path / "input.csv"
    path.write_bytes(b"data")
    with pytest.raises(ValueError, match="Unsupported"):
        _sha256(path, "unknown")
