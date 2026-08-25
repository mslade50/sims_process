"""Portable fingerprints for generated text inputs.

Git and Windows tooling may materialize the same CSV with either LF or CRLF
line endings.  Weekly model-input hashes should identify the data, not the
checkout's newline convention, so text inputs are fingerprinted after newline
normalization.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


LF_NORMALIZED_HASH_MODE = "sha256_lf_normalized_v1"


def lf_normalized_bytes(path: str | Path) -> bytes:
    """Return file bytes with CRLF and lone CR converted to LF."""
    raw = Path(path).read_bytes()
    return raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def lf_normalized_sha256(path: str | Path) -> str:
    """Hash a text file independently of its platform line endings."""
    return hashlib.sha256(lf_normalized_bytes(path)).hexdigest()


def lf_normalized_size(path: str | Path) -> int:
    """Return the normalized byte length used by portable text contracts."""
    return len(lf_normalized_bytes(path))
