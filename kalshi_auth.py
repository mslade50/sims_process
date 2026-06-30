"""Kalshi Trade API v2 — RSA-PSS request signing.

Each authenticated request needs three headers:
    KALSHI-ACCESS-KEY        the access key ID
    KALSHI-ACCESS-TIMESTAMP  current time in milliseconds (string)
    KALSHI-ACCESS-SIGNATURE  base64( RSA-PSS-SHA256( f"{ts}{METHOD}{path}" ) )

`path` is the URL path including query string, but NOT the host.
"""
from __future__ import annotations

import base64
import os
import time

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

_PRIVATE_KEY = None  # cached parsed key


def _load_key():
    global _PRIVATE_KEY
    if _PRIVATE_KEY is not None:
        return _PRIVATE_KEY
    path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
    if not path:
        raise RuntimeError(
            "KALSHI_PRIVATE_KEY_PATH not set in .env — generate a Kalshi API "
            "key, save the .pem outside the repo, and point this env var at it."
        )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Kalshi private key not found at {path}")
    # If the path points at a directory (a common misconfig — e.g. ~/.kalshi
    # instead of the .pem inside it), pick private.pem (the working key), else
    # the single .pem present.
    if os.path.isdir(path):
        cand = os.path.join(path, "private.pem")
        if not os.path.exists(cand):
            pems = [f for f in os.listdir(path) if f.endswith(".pem")]
            if len(pems) != 1:
                raise FileNotFoundError(
                    f"KALSHI_PRIVATE_KEY_PATH is a directory ({path}) with "
                    f"{'no' if not pems else 'multiple'} .pem files — point the env var "
                    "at the .pem key file itself.")
            cand = os.path.join(path, pems[0])
        path = cand
    with open(path, "rb") as f:
        _PRIVATE_KEY = serialization.load_pem_private_key(f.read(), password=None)
    return _PRIVATE_KEY


def access_key():
    k = os.getenv("KALSHI_ACCESS_KEY")
    if not k:
        raise RuntimeError("KALSHI_ACCESS_KEY not set in .env")
    return k


def sign_headers(method, path):
    """Return the three KALSHI-ACCESS-* headers for a request.

    Kalshi signs `timestamp + METHOD + path_without_query`. The query string,
    if any, is stripped before hashing — including it produces a 401.

    Args:
        method: "GET" | "POST" | "DELETE" (upper-case)
        path:   request path, with or without query string. The query string
                will be stripped for signing but the caller should still
                send the full path on the wire.
    """
    ts = str(int(time.time() * 1000))
    sign_path = path.split("?", 1)[0]
    msg = (ts + method.upper() + sign_path).encode("utf-8")
    key = _load_key()
    sig = key.sign(
        msg,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY": access_key(),
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode("ascii"),
    }
