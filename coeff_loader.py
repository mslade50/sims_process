"""Load model coefficients from the golf_sims 'coefficients' sheet tab.

THE SHEET IS THE SINGLE SOURCE OF TRUTH for every fitted coefficient dict.
sim_inputs.py calls load_sheet_coefficients() at import and materializes the
dicts into its own namespace, so consumers keep doing
`from sim_inputs import coefficients_2` unchanged. This file is synced to
sims_process alongside sim_inputs.py (push_sim_inputs.py).

Loader contract with the tab: any row where BOTH the dict_key and term columns
are non-empty is a coefficient; the value comes from the 'live' column.
Everything else (section bands, labels, notes, reference rows) is decoration.

Failure semantics: fetch failure raises (loud, like the fetch guards elsewhere
in the pipeline). Set COEFFS_FROM_CACHE=1 to deliberately run from the last
good coeffs_cache.json (written to the CWD after every successful fetch).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

WORKSHEET = "coefficients"
CACHE_FILE = "coeffs_cache.json"
SCALARS_KEY = "scalars"


def _worksheet_rows() -> list[list[str]]:
    import gspread
    from google.oauth2.service_account import Credentials

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    serialized = os.getenv("GOOGLE_CREDS_JSON")
    if serialized:
        credentials = Credentials.from_service_account_info(json.loads(serialized), scopes=scopes)
    else:
        credentials = Credentials.from_service_account_file(
            str(Path(__file__).with_name("credentials.json")), scopes=scopes
        )
    client = gspread.authorize(credentials)
    # UNFORMATTED_VALUE: display formatting (e.g. 4dp rounding) must never leak
    # into the numbers the model runs on
    return client.open("golf_sims").worksheet(WORKSHEET).get(
        value_render_option="UNFORMATTED_VALUE"
    )


def parse_rows(rows: list[list]) -> dict[str, dict[str, float]]:
    def text(row: list, i: int) -> str:
        return str(row[i]).strip() if i < len(row) and row[i] is not None else ""

    header = None
    for index, row in enumerate(rows):
        cells = [text(row, i).lower() for i in range(len(row))]
        if "dict_key" in cells and "term" in cells:
            header = {name: cells.index(name) for name in ("dict_key", "term")}
            header["live"] = next(i for i, cell in enumerate(cells) if cell.startswith("live"))
            start = index + 1
            break
    if header is None:
        raise RuntimeError("coefficients tab: no header row with dict_key/term columns found")

    dicts: dict[str, dict[str, float]] = {}
    for row in rows[start:]:
        dict_key, term = text(row, header["dict_key"]), text(row, header["term"])
        if not dict_key or not term:
            continue
        live = row[header["live"]] if header["live"] < len(row) else ""
        if live == "" or live is None:
            raise RuntimeError(f"coefficients tab: empty live value for {dict_key}.{term}")
        dicts.setdefault(dict_key, {})[term] = float(live)
    if not dicts:
        raise RuntimeError("coefficients tab: parsed zero coefficient rows")
    return dicts


def _hash(dicts: dict) -> str:
    return hashlib.sha256(json.dumps(dicts, sort_keys=True).encode()).hexdigest()[:12]


def load_sheet_coefficients(verbose: bool = True) -> dict:
    """Return {dict_name: {term: value}, ...} plus scalars unpacked at top level."""
    cache_path = Path(CACHE_FILE)
    if os.getenv("COEFFS_FROM_CACHE") == "1":
        if not cache_path.is_file():
            raise RuntimeError("COEFFS_FROM_CACHE=1 but no coeffs_cache.json in CWD")
        dicts = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"[coeff_loader] {len(dicts)} dicts loaded from CACHE (hash={_hash(dicts)}) - deliberate override")
    else:
        # 3-attempt backoff: one transient Sheets 429/503 at a scheduled run used
        # to kill the whole nightly sim (no round parquet -> board on consensus,
        # silently). After retries, fall back LOUDLY to the last good cache when
        # one exists; hard-raise only with no cache at all.
        import time as _time
        dicts = None
        last_exc = None
        for attempt in range(3):
            try:
                dicts = parse_rows(_worksheet_rows())
                break
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    wait = 5 * (attempt + 1)
                    print(f"[coeff_loader] sheet fetch failed ({exc}); retry in {wait}s "
                          f"({attempt + 2}/3)")
                    _time.sleep(wait)
        if dicts is None:
            if cache_path.is_file():
                dicts = json.loads(cache_path.read_text(encoding="utf-8"))
                msg = (f"[coeff_loader] sheet unreachable after 3 attempts ({last_exc}) - "
                       f"running from CACHED coefficients (hash={_hash(dicts)})")
                print(msg)
                try:
                    from maker_alerts import send_telegram
                    send_telegram(msg)
                except Exception:
                    pass
            else:
                raise RuntimeError(
                    f"[coeff_loader] failed to load coefficients from sheet: {last_exc}. "
                    "Fix connectivity or set COEFFS_FROM_CACHE=1 to run from the last good cache."
                ) from last_exc
        else:
            cache_path.write_text(json.dumps(dicts, indent=2, sort_keys=True), encoding="utf-8")
            if verbose:
                print(f"[coeff_loader] {len(dicts)} coefficient dicts loaded from sheet (hash={_hash(dicts)})")

    result = dict(dicts)
    for term, value in dicts.get(SCALARS_KEY, {}).items():
        result[term] = value
    return result
