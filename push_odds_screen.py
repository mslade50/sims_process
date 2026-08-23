"""Upload joined odds + fair prices to Cloudflare R2 for the odds screen.

Fetches scraped odds from GitHub (same pattern as odds_loader.py), reads
fair price CSVs from disk, joins into unified JSON per market type, and
either writes publish-ready files or uploads them to R2.

Usage:
    python push_odds_screen.py             # Upload all markets
    python push_odds_screen.py --dry-run   # Print JSON to stdout, skip upload
    python push_odds_screen.py --output-dir /tmp/odds-screen

Env vars:
    CF_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY  (for R2 upload)
    GH_TOKEN or GITHUB_TOKEN  (for golf_scraping repo fetch)
    GOOGLE_CREDS_JSON or credentials.json  (for sheet_config)
"""

import json
import hashlib
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from glob import glob
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="  %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
R2_BUCKET = "golf-odds-data"
R2_PREFIX = "odds_data"

GITHUB_REPO = "mslade50/golf_scraping"
GITHUB_BRANCH = "master"
GITHUB_DATA_PATH = "data"

SCRAPED_LOCAL = PROJECT_ROOT / "permanent_data" / "scraped_odds"


# ─── helpers ────────────────────────────────────────────────────────────────

def _get_r2_client():
    """Return a boto3 S3 client configured for Cloudflare R2."""
    import boto3
    account_id = os.environ["CF_ACCOUNT_ID"]
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def _upload_json(client, key: str, data: dict):
    """Upload a JSON object to R2."""
    body = json.dumps(data, default=str)
    client.put_object(
        Bucket=R2_BUCKET,
        Key=f"{R2_PREFIX}/{key}",
        Body=body.encode(),
        ContentType="application/json",
    )
    logger.info(f"Uploaded {key} ({len(body)} bytes)")


def _write_payload_files(payloads: dict, output_dir: Path) -> list[Path]:
    """Atomically write publish-ready JSON files for an external uploader.

    GitHub Actions uses this path with Wrangler's scoped Cloudflare API token.
    Keeping generation separate from transport means a failed upload cannot be
    mistaken for a successful odds-screen refresh.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for key, data in payloads.items():
        target = output_dir / key
        fd, temp_name = tempfile.mkstemp(
            dir=str(output_dir), prefix=f".{key}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(
                    data,
                    handle,
                    default=str,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, target)
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise
        written.append(target)
        logger.info(f"Wrote {target} ({target.stat().st_size} bytes)")
    return written


def _json_bytes(data: dict) -> bytes:
    return (
        json.dumps(data, default=str, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _build_atomic_payload_bundle(payloads: dict, generation: str | None = None):
    """Build immutable odds-screen objects plus the pointer readers fetch first.

    Market objects are never overwritten in place.  A failed upload leaves the
    old root ``meta.json`` pointer untouched, so readers keep one complete prior
    generation instead of combining new markets with stale metadata.
    """
    encoded = {name: _json_bytes(data) for name, data in sorted(payloads.items())}
    content_digest = hashlib.sha256(
        b"".join(name.encode("utf-8") + b"\0" + data for name, data in encoded.items())
    ).hexdigest()
    if generation is None:
        raw_time = str((payloads.get("meta.json") or {}).get("last_updated") or "")
        time_part = "".join(ch for ch in raw_time if ch.isdigit())[:14] or "unstamped"
        generation = f"{time_part}-{content_digest[:16]}"
    if not generation or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in generation):
        raise ValueError("odds-screen generation must contain only letters, digits, '-' or '_'")

    base = f"generations/{generation}"
    file_bindings = {
        name: {
            "key": f"{base}/{name}",
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
        }
        for name, data in encoded.items()
    }
    meta = dict(payloads.get("meta.json") or {})
    pointer = {
        **meta,
        "schema_version": "odds-screen-generation/v1",
        "generation": generation,
        "content_sha256": content_digest,
        "files": file_bindings,
    }
    return {
        "generation": generation,
        "encoded": encoded,
        "pointer": pointer,
        "pointer_bytes": _json_bytes(pointer),
    }


def _write_atomic_payload_bundle(
    payloads: dict, output_dir: Path, generation: str | None = None
) -> list[Path]:
    """Write versioned objects first and root meta.json publication pointer last."""
    bundle = _build_atomic_payload_bundle(payloads, generation=generation)
    generation_dir = Path(output_dir) / "generations" / bundle["generation"]
    # Reuse the fsync + os.replace writer for each immutable generation object.
    decoded = {
        name: json.loads(data.decode("utf-8"))
        for name, data in bundle["encoded"].items()
    }
    written = _write_payload_files(decoded, generation_dir)
    written.extend(_write_payload_files({"meta.json": bundle["pointer"]}, output_dir))
    return written


def _atomic_upload_plan(output_dir: Path) -> list[tuple[str, Path]]:
    """Verify and enumerate every pointer binding, with meta.json strictly last."""
    output_dir = Path(output_dir)
    pointer_path = output_dir / "meta.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    if pointer.get("schema_version") != "odds-screen-generation/v1":
        raise RuntimeError("odds-screen pointer schema is missing or invalid")
    generation = str(pointer.get("generation") or "")
    prefix = f"generations/{generation}/"
    files = pointer.get("files") or {}
    if not files:
        raise RuntimeError("odds-screen pointer declares no generation files")
    plan = []
    declared_paths = set()
    for name, binding in sorted(files.items()):
        key = str(binding.get("key") or "")
        if not key.startswith(prefix) or Path(key).name != name:
            raise RuntimeError(f"unsafe odds-screen pointer binding: {name}")
        path = output_dir / Path(key)
        if not path.is_file():
            raise RuntimeError(f"declared odds-screen generation file is missing: {key}")
        data = path.read_bytes()
        if len(data) != int(binding.get("size") or -1):
            raise RuntimeError(f"odds-screen generation size mismatch: {key}")
        if hashlib.sha256(data).hexdigest() != binding.get("sha256"):
            raise RuntimeError(f"odds-screen generation hash mismatch: {key}")
        declared_paths.add(path.resolve())
        plan.append((key, path))
    actual_paths = {
        path.resolve()
        for path in (output_dir / "generations" / generation).glob("*.json")
    }
    if actual_paths != declared_paths:
        raise RuntimeError("odds-screen generation contains undeclared or omitted JSON files")
    plan.append(("meta.json", pointer_path))
    return plan


def _upload_atomic_payload_bundle(client, payloads: dict, generation: str | None = None):
    """Stage one complete R2 generation, then atomically advance root meta.json."""
    bundle = _build_atomic_payload_bundle(payloads, generation=generation)
    for name, data in bundle["encoded"].items():
        key = f"generations/{bundle['generation']}/{name}"
        client.put_object(
            Bucket=R2_BUCKET,
            Key=f"{R2_PREFIX}/{key}",
            Body=data,
            ContentType="application/json",
        )
        logger.info(f"Uploaded {key} ({len(data)} bytes)")
    # Consumer-visible commit point. Never execute this if any staged write fails.
    client.put_object(
        Bucket=R2_BUCKET,
        Key=f"{R2_PREFIX}/meta.json",
        Body=bundle["pointer_bytes"],
        ContentType="application/json",
    )
    logger.info(
        f"Published odds-screen generation {bundle['generation']} via meta.json"
    )
    return bundle["pointer"]


def _fetch_scraped_guarded(market: str, *, round=None) -> "dict | None":
    """Fetch a scraped market JSON via odds_loader (GitHub->local) and apply the
    SAME guards the sim pricer uses: freshness (MAX_AGE_HOURS), event_id scope,
    file-level round, and per-row round filtering. Returns a safe dict or None.

    `market` is the bare market name ('round_matchups', 'tournament_matchups',
    'round_scores') — odds_loader._fetch_scraped_json appends '_latest.json'. This
    replaces push_odds_screen's old un-guarded fetch so the odds screen can never
    price a stale / wrong-event / wrong-round file the sim itself would reject."""
    from odds_loader import _fetch_scraped_json as _ol_fetch, guard_scraped_data
    data = _ol_fetch(market)
    return guard_scraped_data(data, market, round=round)


def _first_existing(paths):
    """Return the first existing Path from `paths`, or None."""
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return None


def _load_name_replacements() -> dict:
    try:
        from sim_inputs import name_replacements
        return name_replacements
    except ImportError:
        return {}


def _norm(name: str, replacements: dict) -> str:
    """Lowercase + apply name_replacements."""
    name = name.strip().lower()
    return replacements.get(name, name)


def _first_last(name: str) -> str:
    """Convert "last, first" -> "first last" (unambiguous; split on first comma).
    load_betcris_outrights returns names in lowercase "first last", while the
    outrights loop keys on "last, first" (top_finish_probs / DataGolf) — this
    bridges the two so the Betcris overlay actually joins."""
    if "," in name:
        last, first = [p.strip() for p in name.split(",", 1)]
        return f"{first} {last}"
    return name


def _american_to_prob(odds: float) -> float:
    """Convert American odds to implied probability."""
    if odds >= 100:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def _prob_to_american(prob: float) -> int:
    """Convert probability to American odds."""
    if prob <= 0 or prob >= 1:
        return 0
    if prob >= 0.5:
        return round(-prob / (1 - prob) * 100)
    else:
        return round((1 - prob) / prob * 100)


def _edge_pct(fair_prob: float, market_odds: float) -> float:
    """Edge % = fair_prob - implied_prob (positive = value)."""
    implied = _american_to_prob(market_odds)
    return round((fair_prob - implied) * 100, 1)


# ─── market builders ────────────────────────────────────────────────────────

def _build_round_matchups(tourney: str, sim_round: int, repl: dict) -> list:
    """Build round matchup records from scraped odds + fair prices.

    `sim_round` is the DOW-derived round (1-4) passed in by main(), matching the
    sim pricer's target round exactly."""
    # Load fair prices
    fair_path = PROJECT_ROOT / tourney / f"all_books_fair_matchups_r{sim_round}.csv"
    if not fair_path.exists():
        fair_path = PROJECT_ROOT / f"all_books_fair_matchups_r{sim_round}.csv"
    fair_df = pd.DataFrame()
    if fair_path.exists():
        fair_df = pd.read_csv(fair_path)
        fair_df["Player 1"] = fair_df["Player 1"].str.lower().replace(repl)
        fair_df["Player 2"] = fair_df["Player 2"].str.lower().replace(repl)
        logger.info(f"Loaded fair matchups: {fair_path.name} ({len(fair_df)} rows)")

    # Load scraped odds (guarded: freshness + event_id + round)
    scraped = _fetch_scraped_guarded("round_matchups", round=sim_round)
    if not scraped:
        # Fall back to fair_df only (it has book odds too)
        if fair_df.empty:
            return []
        return _matchups_from_fair_df(fair_df)

    matchups = {}
    for match in scraped.get("match_list", []):
        p1 = _norm(match.get("p1_player_name", ""), repl)
        p2 = _norm(match.get("p2_player_name", ""), repl)
        key = (p1, p2)
        if key not in matchups:
            matchups[key] = {"p1": p1, "p2": p2, "books": {}, "fair": {}, "edge": {}}
        for book, odds in match.get("odds", {}).items():
            if book == "datagolf":
                continue
            p1_odds = odds.get("p1")
            p2_odds = odds.get("p2")
            if book == "kalshi" and odds.get("p1_mid"):
                p1_odds = odds["p1_mid"]
                p2_odds = odds.get("p2_mid", p2_odds)
            if p1_odds is not None and p2_odds is not None:
                matchups[key]["books"][book] = {"p1": int(p1_odds), "p2": int(p2_odds)}

    # Merge fair prices
    if not fair_df.empty:
        for key, rec in matchups.items():
            mask = (fair_df["Player 1"] == key[0]) & (fair_df["Player 2"] == key[1])
            rows = fair_df[mask]
            if rows.empty:
                # Try reversed
                mask = (fair_df["Player 1"] == key[1]) & (fair_df["Player 2"] == key[0])
                rows = fair_df[mask]
            if not rows.empty:
                row = rows.iloc[0]
                if "Fair_p1" in row and pd.notna(row["Fair_p1"]):
                    fair_p1 = int(row["Fair_p1"])
                    fair_p2 = int(row["Fair_p2"])
                    p1_prob = _american_to_prob(fair_p1)
                    p2_prob = _american_to_prob(fair_p2)
                    rec["fair"] = {
                        "p1": fair_p1, "p2": fair_p2,
                        "p1_prob": round(p1_prob, 3), "p2_prob": round(p2_prob, 3),
                    }
                    # Compute edges vs each book
                    best_edge = 0
                    for book, odds in rec["books"].items():
                        e1 = _edge_pct(p1_prob, odds["p1"])
                        e2 = _edge_pct(p2_prob, odds["p2"])
                        rec["edge"][f"{book}_p1"] = e1
                        rec["edge"][f"{book}_p2"] = e2
                        best_edge = max(best_edge, e1, e2)
                    rec["best_edge"] = best_edge

    return sorted(matchups.values(), key=lambda x: x.get("best_edge", 0), reverse=True)


def _matchups_from_fair_df(fair_df: pd.DataFrame) -> list:
    """Build matchup records from the fair CSV alone (has book odds in it)."""
    grouped = fair_df.groupby(["Player 1", "Player 2"])
    matchups = []
    for (p1, p2), grp in grouped:
        rec = {"p1": p1, "p2": p2, "books": {}, "fair": {}, "edge": {}}
        fair_row = grp.iloc[0]
        if "Fair_p1" in fair_row and pd.notna(fair_row["Fair_p1"]):
            fair_p1 = int(fair_row["Fair_p1"])
            fair_p2 = int(fair_row["Fair_p2"])
            p1_prob = _american_to_prob(fair_p1)
            p2_prob = _american_to_prob(fair_p2)
            rec["fair"] = {
                "p1": fair_p1, "p2": fair_p2,
                "p1_prob": round(p1_prob, 3), "p2_prob": round(p2_prob, 3),
            }
        for _, row in grp.iterrows():
            book = row.get("Bookmaker", "").lower()
            if book and pd.notna(row.get("P1 Odds")) and pd.notna(row.get("P2 Odds")):
                rec["books"][book] = {"p1": int(row["P1 Odds"]), "p2": int(row["P2 Odds"])}
                if rec["fair"]:
                    rec["edge"][f"{book}_p1"] = _edge_pct(p1_prob, row["P1 Odds"])
                    rec["edge"][f"{book}_p2"] = _edge_pct(p2_prob, row["P2 Odds"])
        if rec["edge"]:
            rec["best_edge"] = max(rec["edge"].values())
        matchups.append(rec)
    return sorted(matchups, key=lambda x: x.get("best_edge", 0), reverse=True)


def _build_tournament_matchups(tourney: str, repl: dict) -> list:
    """Build tournament matchup records from scraped odds + fair prices."""
    # Find latest matchups_ftsimp file
    pattern = str(PROJECT_ROOT / tourney / f"matchups_ftsimp_{tourney}_*.csv")
    files = sorted(glob(pattern))
    if not files:
        pattern = str(PROJECT_ROOT / f"matchups_ftsimp_{tourney}_*.csv")
        files = sorted(glob(pattern))
    fair_df = pd.DataFrame()
    if files:
        fair_df = pd.read_csv(files[-1])
        fair_df["Player 1"] = fair_df["Player 1"].str.lower().replace(repl)
        fair_df["Player 2"] = fair_df["Player 2"].str.lower().replace(repl)
        logger.info(f"Loaded tournament fair matchups: {Path(files[-1]).name} ({len(fair_df)} rows)")

    # Load scraped odds (guarded: freshness + event_id; tournament = no round)
    scraped = _fetch_scraped_guarded("tournament_matchups")
    if not scraped:
        if fair_df.empty:
            return []
        return _matchups_from_fair_df(fair_df)

    matchups = {}
    for match in scraped.get("match_list", []):
        p1 = _norm(match.get("p1_player_name", ""), repl)
        p2 = _norm(match.get("p2_player_name", ""), repl)
        key = (p1, p2)
        if key not in matchups:
            matchups[key] = {"p1": p1, "p2": p2, "books": {}, "fair": {}, "edge": {}}
        for book, odds in match.get("odds", {}).items():
            if book == "datagolf":
                continue
            p1_odds = odds.get("p1")
            p2_odds = odds.get("p2")
            if book == "kalshi" and odds.get("p1_mid"):
                p1_odds = odds["p1_mid"]
                p2_odds = odds.get("p2_mid", p2_odds)
            if p1_odds is not None and p2_odds is not None:
                matchups[key]["books"][book] = {"p1": int(p1_odds), "p2": int(p2_odds)}

    # Merge fair prices from CSV
    if not fair_df.empty:
        for key, rec in matchups.items():
            mask = (fair_df["Player 1"] == key[0]) & (fair_df["Player 2"] == key[1])
            rows = fair_df[mask]
            if rows.empty:
                mask = (fair_df["Player 1"] == key[1]) & (fair_df["Player 2"] == key[0])
                rows = fair_df[mask]
            if not rows.empty:
                row = rows.iloc[0]
                if "Fair_p1" in row and pd.notna(row["Fair_p1"]):
                    fair_p1 = int(row["Fair_p1"])
                    fair_p2 = int(row["Fair_p2"])
                    p1_prob = _american_to_prob(fair_p1)
                    p2_prob = _american_to_prob(fair_p2)
                    rec["fair"] = {
                        "p1": fair_p1, "p2": fair_p2,
                        "p1_prob": round(p1_prob, 3), "p2_prob": round(p2_prob, 3),
                    }
                    best_edge = 0
                    for book, odds in rec["books"].items():
                        e1 = _edge_pct(p1_prob, odds["p1"])
                        e2 = _edge_pct(p2_prob, odds["p2"])
                        rec["edge"][f"{book}_p1"] = e1
                        rec["edge"][f"{book}_p2"] = e2
                        best_edge = max(best_edge, e1, e2)
                    rec["best_edge"] = best_edge

    return sorted(matchups.values(), key=lambda x: x.get("best_edge", 0), reverse=True)


def _build_score_lines(tourney: str, sim_round: int, repl: dict) -> list:
    """Build score line records from scraped odds + fair card.

    `sim_round` is the DOW-derived round (1-4) passed in by main()."""
    # Load fair card
    fair_path = PROJECT_ROOT / tourney / f"fair_card_r{sim_round}.csv"
    if not fair_path.exists():
        fair_path = PROJECT_ROOT / f"fair_card_r{sim_round}.csv"
    fair_df = pd.DataFrame()
    if fair_path.exists():
        fair_df = pd.read_csv(fair_path)
        fair_df["Player"] = fair_df["Player"].str.lower().replace(repl)
        logger.info(f"Loaded fair card: {fair_path.name} ({len(fair_df)} rows)")

    # Load scraped score lines (guarded: freshness + event_id + round)
    scraped = _fetch_scraped_guarded("round_scores", round=sim_round)
    scraped_lines = {}
    if scraped:
        for item in scraped.get("lines", []):
            player = _norm(item.get("player_name", ""), repl)
            line = item.get("line")
            if line is None:
                continue
            odds = item.get("odds", {})
            books = {}
            for book, book_odds in odds.items():
                if isinstance(book_odds, dict):
                    over = book_odds.get("over")
                    under = book_odds.get("under")
                    if over is not None and under is not None:
                        books[book] = {"over": int(over), "under": int(under)}
            scraped_lines[player] = {"line": float(line), "books": books}
        logger.info(f"Loaded {len(scraped_lines)} scraped score lines")

    # Build output
    results = []
    players = set(scraped_lines.keys())
    if not fair_df.empty:
        players |= set(fair_df["Player"].tolist())

    for player in players:
        rec = {"player": player, "books": {}, "fair": {}, "edge": {}}
        sl = scraped_lines.get(player, {})
        rec["books"] = sl.get("books", {})

        line = sl.get("line")
        pred = None

        if not fair_df.empty:
            mask = fair_df["Player"] == player
            if mask.any():
                row = fair_df[mask].iloc[0]
                pred = row.get("Pred")
                if pred is not None:
                    rec["pred"] = round(float(pred), 2)

                # Find the matching line column in fair_df
                if line is not None:
                    line_col = str(line)
                    if line_col in row.index and pd.notna(row[line_col]):
                        fair_under = int(row[line_col])
                        fair_under_prob = _american_to_prob(fair_under)
                        fair_over_prob = 1 - fair_under_prob
                        fair_over = _prob_to_american(fair_over_prob)
                        rec["fair"] = {"over": fair_over, "under": fair_under}

                        for book, odds in rec["books"].items():
                            rec["edge"][f"{book}_over"] = _edge_pct(fair_over_prob, odds["over"])
                            rec["edge"][f"{book}_under"] = _edge_pct(fair_under_prob, odds["under"])

        if line is not None:
            rec["line"] = line
        if rec["books"] or rec.get("pred"):
            if rec["edge"]:
                rec["best_edge"] = max(rec["edge"].values())
            results.append(rec)

    return sorted(results, key=lambda x: x.get("best_edge", 0), reverse=True)


def _decimal_to_american(dec: float) -> int:
    """Convert decimal odds to American odds."""
    if dec >= 2.0:
        return int(round((dec - 1) * 100))
    else:
        return int(round(-100 / (dec - 1)))


def _fetch_dg_outrights(market_name: str, repl: dict) -> dict:
    """Fetch outright odds from DataGolf API. Returns {player: {book: american_odds}}."""
    api_key = os.getenv("DATAGOLF_API_KEY")
    if not api_key:
        return {}
    try:
        resp = requests.get(
            "https://feeds.datagolf.com/betting-tools/outrights",
            params={
                "tour": "pga", "market": market_name,
                "odds_format": "decimal", "file_format": "json", "key": api_key,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"DG outright API failed for {market_name}: {e}")
        return {}

    result = {}
    for entry in data.get("odds", []):
        if not isinstance(entry, dict):
            continue
        player = _norm(entry.get("player_name", ""), repl)
        if not player:
            continue
        books = {}
        for book in ["pinnacle", "betcris", "betonline", "draftkings", "fanduel",
                      "betmgm", "caesars", "bovada", "unibet"]:
            dec = entry.get(book)
            if dec is not None:
                try:
                    books[book] = _decimal_to_american(float(dec))
                except (ValueError, ZeroDivisionError):
                    pass
        if books:
            result[player] = books
    logger.info(f"DG outrights ({market_name}): {len(result)} players")
    return result


def _build_outrights(tourney: str, repl: dict) -> dict:
    """Build outright records from OUR sim fair prices + DataGolf/Kalshi book odds.

    Fairs come from the full-field, dense top_finish_probs_live_{tourney}.csv (not
    the edge-filtered finish_equity file, which is sparse post-cut). Book-aware
    dead-heat handling: the WIN market uses dead-heat-resolved probs for every book;
    top-5/10/20 use dead-heat fairs (top_N) for standard books but the no-dead-heat
    columns (top_N_nodh) for Kalshi and NoVig, which do NOT dead-heat their payouts."""
    sim_path = _first_existing([
        PROJECT_ROOT / tourney / f"top_finish_probs_live_{tourney}.csv",
        PROJECT_ROOT / f"top_finish_probs_live_{tourney}.csv",
        # legacy fallbacks (sparse, but better than nothing)
        PROJECT_ROOT / tourney / f"finish_equity_live_{tourney}.csv",
        PROJECT_ROOT / f"finish_equity_live_{tourney}.csv",
    ])
    sim_df = pd.DataFrame()
    if sim_path is not None:
        sim_df = pd.read_csv(sim_path)
        sim_df["player_name"] = sim_df["player_name"].str.lower().replace(repl)
        logger.info(f"Loaded sim fair probs: {sim_path.name} ({len(sim_df)} rows)")

    def _prob(player, col):
        """Our fair probability for `player` in column `col`, or None."""
        if sim_df.empty or col not in sim_df.columns:
            return None
        mask = sim_df["player_name"] == player
        if not mask.any():
            return None
        v = sim_df[mask].iloc[0][col]
        return float(v) if pd.notna(v) and v > 0 else None

    # Kalshi/NoVig outrights are no longer published as scraped JSON (the sim
    # fetches them live); kept here so the dead-heat split still applies if they return.
    kalshi_by_market = {}

    # Fetch DataGolf API outrights (pinnacle, betcris, betonline, etc.)
    dg_by_market = {}
    for dg_market in ["win", "top_5", "top_10", "top_20"]:
        dg_by_market[dg_market] = _fetch_dg_outrights(dg_market, repl)

    # The DataGolf outrights feed FREEZES during live play ("Tournament is live,
    # baseline model not available") and drops most Betcris quotes, so the screen's
    # Betcris column goes stale. Overlay the FRESH scraped Betcris outrights
    # (betcris_outrights_latest.json, refreshed ~every 30 min) on top. Keyed by
    # market_type (winner/top_5/top_10/top_20), which matches our market_key below.
    betcris_scraped = {}
    try:
        from odds_loader import load_betcris_outrights
        _bc = load_betcris_outrights()
        if not _bc.empty:
            _bc = _bc.copy()
            _bc["player_name"] = _bc["player_name"].str.lower().replace(repl)
            for _mt, _grp in _bc.groupby("market_type"):
                betcris_scraped[_mt] = dict(zip(_grp["player_name"], _grp["american_odds"]))
            logger.info(f"Loaded fresh scraped Betcris outrights: {len(_bc)} lines "
                        f"({', '.join(sorted(betcris_scraped))})")
    except Exception as e:
        logger.warning(f"scraped Betcris overlay unavailable ({e}); using DataGolf Betcris only")

    DG_MARKET_MAP = {"winner": "win", "top_5": "top_5", "top_10": "top_10", "top_20": "top_20"}
    NODH_BOOKS = {"kalshi", "novig"}   # these don't dead-heat → use the _nodh fair

    markets = {}
    # (market_key, dead-heat col, no-dead-heat col). WIN: everyone dead-heats → same col.
    for market_key, dh_col, nodh_col in [
        ("winner", "simulated_win_prob", "simulated_win_prob"),
        ("top_5", "top_5", "top_5_nodh"),
        ("top_10", "top_10", "top_10_nodh"),
        ("top_20", "top_20", "top_20_nodh"),
    ]:
        records = []
        dg_market_name = DG_MARKET_MAP.get(market_key, market_key)
        dg_odds = dg_by_market.get(dg_market_name, {})

        players = set()
        if not sim_df.empty and dh_col in sim_df.columns:
            players = set(sim_df["player_name"].tolist())
        players |= set(kalshi_by_market.get(market_key, {}).keys())
        players |= set(dg_odds.keys())

        for player in players:
            rec = {"player": player, "books": {}, "edge": {}}
            dh_prob = _prob(player, dh_col)
            nodh_prob = _prob(player, nodh_col)

            # Display fair = dead-heat-resolved (what most books pay).
            if dh_prob is not None:
                rec["sim_prob"] = round(dh_prob, 4)
                rec["fair_odds"] = _prob_to_american(dh_prob)
                if nodh_col != dh_col and nodh_prob is not None:
                    rec["sim_prob_nodh"] = round(nodh_prob, 4)

            # DataGolf book odds: standard books dead-heat → dh fair; (none of the
            # DG books are in NODH_BOOKS today, but the split is applied per-book).
            player_dg = dict(dg_odds.get(player, {}))
            # Prefer the fresh scraped Betcris over DataGolf's frozen live-play quote
            # (and add Betcris for players DG dropped entirely during live play).
            # Betcris scrape is keyed "first last"; our loop key is "last, first".
            bc_live = betcris_scraped.get(market_key, {}).get(_first_last(player))
            if bc_live is not None:
                player_dg["betcris"] = int(bc_live)
            for book, american in player_dg.items():
                rec["books"][book] = {"yes": american}
                fair = nodh_prob if book in NODH_BOOKS else dh_prob
                if fair is not None:
                    rec["edge"][book] = _edge_pct(fair, american)

            # Kalshi overlay (no dead-heat → no-DH fair).
            kalshi_odds = kalshi_by_market.get(market_key, {}).get(player)
            if kalshi_odds:
                rec["books"]["kalshi"] = kalshi_odds
                fair = nodh_prob if nodh_prob is not None else dh_prob
                if fair is not None and kalshi_odds.get("yes"):
                    rec["edge"]["kalshi"] = _edge_pct(fair, kalshi_odds["yes"])

            if rec.get("sim_prob") or rec.get("books"):
                if rec["edge"]:
                    rec["best_edge"] = max(rec["edge"].values())
                records.append(rec)

        markets[market_key] = sorted(
            records, key=lambda x: x.get("best_edge", x.get("sim_prob", 0)), reverse=True
        )

    return markets


# ─── main ───────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Upload odds screen data to R2")
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--dry-run", action="store_true", help="Print JSON, skip upload"
    )
    output_mode.add_argument(
        "--output-dir",
        type=Path,
        help="Write publish-ready JSON files here and skip the built-in R2 upload",
    )
    args = parser.parse_args()

    # Load config from Sheet
    try:
        from sheet_config import load_config
        config = load_config()
        tourney = config["tourney"]
        round_num = config["round_num"]
        event_name = tourney.replace("_", " ").title()
    except Exception as e:
        logger.warning(f"Could not load sheet config: {e}")
        try:
            from sim_inputs import tourney
            round_num = 0
            event_name = tourney.replace("_", " ").title()
        except ImportError:
            logger.error("Cannot determine tourney from sheet_config or sim_inputs")
            sys.exit(1)

    repl = _load_name_replacements()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Round is DOW-derived (always set), consistent with the board + sim pricer.
    # Sheet round_num is kept only for the log line / fallback.
    try:
        from api_utils import sim_round_from_dow
        sim_round = sim_round_from_dow()
    except Exception as e:
        sim_round = round_num + 1 if round_num < 4 else 4
        logger.warning(f"DOW round helper unavailable ({e}); falling back to sheet round R{sim_round}")

    logger.info(f"Building odds screen data: {tourney} R{sim_round} (sheet round_num={round_num})")

    # Build all markets
    round_mu = _build_round_matchups(tourney, sim_round, repl)
    tourn_mu = _build_tournament_matchups(tourney, repl)
    score_lines = _build_score_lines(tourney, sim_round, repl)
    outrights = _build_outrights(tourney, repl)

    payloads = {
        "round_matchups.json": {
            "event_name": event_name, "last_updated": now, "round": sim_round,
            "matchups": round_mu,
        },
        "tournament_matchups.json": {
            "event_name": event_name, "last_updated": now,
            "matchups": tourn_mu,
        },
        "score_lines.json": {
            "event_name": event_name, "last_updated": now, "round": sim_round,
            "lines": score_lines,
        },
        "outrights.json": {
            "event_name": event_name, "last_updated": now,
            "markets": outrights,
        },
        "meta.json": {
            "event_name": event_name, "tourney": tourney,
            "round": sim_round, "last_updated": now,
        },
    }

    for key, data in payloads.items():
        count = len(data.get("matchups", data.get("lines", data.get("markets", {}))))
        logger.info(f"  {key}: {count} records")

    if args.dry_run:
        for key, data in payloads.items():
            print(f"\n{'='*60}")
            print(f"  {key}")
            print(f"{'='*60}")
            print(json.dumps(data, indent=2, default=str)[:2000])
        return

    if args.output_dir:
        written = _write_atomic_payload_bundle(payloads, args.output_dir)
        logger.info(
            f"Done — staged {len(written) - 1} immutable files and wrote meta.json last"
        )
        return

    # Upload to R2
    client = _get_r2_client()
    pointer = _upload_atomic_payload_bundle(client, payloads)
    logger.info(f"Done — published atomic generation {pointer['generation']}")


if __name__ == "__main__":
    main()
