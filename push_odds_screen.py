"""Upload joined odds + fair prices to Cloudflare R2 for the odds screen.

Fetches scraped odds from GitHub (same pattern as odds_loader.py), reads
fair price CSVs from disk, joins into unified JSON per market type, and
uploads to an R2 bucket via the S3-compatible API.

Usage:
    python push_odds_screen.py             # Upload all markets
    python push_odds_screen.py --dry-run   # Print JSON to stdout, skip upload

Env vars:
    CF_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY  (for R2 upload)
    GH_TOKEN or GITHUB_TOKEN  (for golf_scraping repo fetch)
    GOOGLE_CREDS_JSON or credentials.json  (for sheet_config)
"""

import json
import logging
import os
import sys
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


def _fetch_scraped_json(filename: str) -> dict | None:
    """Fetch scraped odds JSON from GitHub, falling back to local paths."""
    gh_token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
    api_url = (
        f"https://api.github.com/repos/{GITHUB_REPO}/contents/"
        f"{GITHUB_DATA_PATH}/{filename}?ref={GITHUB_BRANCH}"
    )
    try:
        headers = {"Accept": "application/vnd.github.raw+json"}
        if gh_token:
            headers["Authorization"] = f"Bearer {gh_token}"
        resp = requests.get(api_url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        logger.info(f"Fetched {filename} from GitHub")
        return data
    except Exception as e:
        logger.warning(f"GitHub fetch failed for {filename}: {e}")

    local = SCRAPED_LOCAL / filename
    if local.exists():
        logger.info(f"Using local {local}")
        with open(local) as f:
            return json.load(f)
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

def _build_round_matchups(tourney: str, round_num: int, repl: dict) -> list:
    """Build round matchup records from scraped odds + fair prices."""
    sim_round = round_num + 1 if round_num < 4 else 4

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

    # Load scraped odds
    scraped = _fetch_scraped_json("round_matchups_latest.json")
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

    # Load scraped odds
    scraped = _fetch_scraped_json("tournament_matchups_latest.json")
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


def _build_score_lines(tourney: str, round_num: int, repl: dict) -> list:
    """Build score line records from scraped odds + fair card."""
    sim_round = round_num + 1 if round_num < 4 else 4

    # Load fair card
    fair_path = PROJECT_ROOT / tourney / f"fair_card_r{sim_round}.csv"
    if not fair_path.exists():
        fair_path = PROJECT_ROOT / f"fair_card_r{sim_round}.csv"
    fair_df = pd.DataFrame()
    if fair_path.exists():
        fair_df = pd.read_csv(fair_path)
        fair_df["Player"] = fair_df["Player"].str.lower().replace(repl)
        logger.info(f"Loaded fair card: {fair_path.name} ({len(fair_df)} rows)")

    # Load scraped score lines
    scraped = _fetch_scraped_json("round_scores_latest.json")
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
    """Build outright records from DataGolf API + Kalshi scraped + finish equity CSV."""
    # Load sim probabilities
    eq_path = PROJECT_ROOT / tourney / f"finish_equity_live_{tourney}.csv"
    if not eq_path.exists():
        eq_path = PROJECT_ROOT / f"finish_equity_live_{tourney}.csv"
    if not eq_path.exists():
        eq_path = PROJECT_ROOT / tourney / f"finish_equity_{tourney}.csv"
        if not eq_path.exists():
            eq_path = PROJECT_ROOT / f"finish_equity_{tourney}.csv"

    sim_df = pd.DataFrame()
    if eq_path.exists():
        sim_df = pd.read_csv(eq_path)
        sim_df["player_name"] = sim_df["player_name"].str.lower().replace(repl)
        logger.info(f"Loaded finish equity: {eq_path.name} ({len(sim_df)} rows)")

    # Load Kalshi outrights (scraped)
    kalshi = _fetch_scraped_json("kalshi_outrights_latest.json")
    kalshi_by_market = {}
    if kalshi:
        for item in kalshi.get("lines", []):
            player = _norm(item.get("player_name", ""), repl)
            mtype = item.get("market_type", "").lower()
            if item.get("bid", 1) == 0:
                continue
            if mtype not in kalshi_by_market:
                kalshi_by_market[mtype] = {}
            yes_price = item.get("yes_price") or item.get("ask")
            no_price = item.get("no_price") or item.get("bid")
            if yes_price:
                kalshi_by_market[mtype][player] = {
                    "yes": int(yes_price) if yes_price > 1 else int(round((1/yes_price - 1) * 100)),
                    "no": int(no_price) if no_price and no_price > 1 else None,
                }
        logger.info(f"Loaded Kalshi outrights: {sum(len(v) for v in kalshi_by_market.values())} lines")

    # Fetch DataGolf API outrights (pinnacle, betcris, betonline, etc.)
    dg_by_market = {}
    for dg_market in ["win", "top_5", "top_10", "top_20"]:
        dg_by_market[dg_market] = _fetch_dg_outrights(dg_market, repl)

    # Map DG market names to our market keys
    DG_MARKET_MAP = {"winner": "win", "top_5": "top_5", "top_10": "top_10", "top_20": "top_20"}

    markets = {}
    for market_key, sim_col, sim_odds_col in [
        ("winner", "simulated_win_prob", "simulated_win_prob_a"),
        ("top_5", "top_5", "top_5_a"),
        ("top_10", "top_10", "top_10_a"),
        ("top_20", "top_20", "top_20_a"),
    ]:
        records = []
        dg_market_name = DG_MARKET_MAP.get(market_key, market_key)
        dg_odds = dg_by_market.get(dg_market_name, {})

        players = set()
        if not sim_df.empty and sim_col in sim_df.columns:
            players = set(sim_df["player_name"].tolist())
        players |= set(kalshi_by_market.get(market_key, {}).keys())
        players |= set(dg_odds.keys())

        for player in players:
            rec = {"player": player, "books": {}, "edge": {}}

            # Sim probability
            if not sim_df.empty and sim_col in sim_df.columns:
                mask = sim_df["player_name"] == player
                if mask.any():
                    row = sim_df[mask].iloc[0]
                    prob = row[sim_col]
                    if pd.notna(prob) and prob > 0:
                        rec["sim_prob"] = round(float(prob), 4)
                        rec["fair_odds"] = _prob_to_american(float(prob))

            # DataGolf API odds (sharp + retail books)
            player_dg = dg_odds.get(player, {})
            for book, american in player_dg.items():
                rec["books"][book] = {"yes": american}
                if "sim_prob" in rec:
                    rec["edge"][book] = _edge_pct(rec["sim_prob"], american)

            # Kalshi odds (overlay, don't overwrite DG books)
            kalshi_odds = kalshi_by_market.get(market_key, {}).get(player)
            if kalshi_odds:
                rec["books"]["kalshi"] = kalshi_odds
                if "sim_prob" in rec and kalshi_odds.get("yes"):
                    rec["edge"]["kalshi"] = _edge_pct(rec["sim_prob"], kalshi_odds["yes"])

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
    parser.add_argument("--dry-run", action="store_true", help="Print JSON, skip upload")
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
    sim_round = round_num + 1 if round_num < 4 else 4

    logger.info(f"Building odds screen data: {tourney} R{sim_round}")

    # Build all markets
    round_mu = _build_round_matchups(tourney, round_num, repl)
    tourn_mu = _build_tournament_matchups(tourney, repl)
    score_lines = _build_score_lines(tourney, round_num, repl)
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

    # Upload to R2
    client = _get_r2_client()
    for key, data in payloads.items():
        _upload_json(client, key, data)

    logger.info(f"Done — uploaded {len(payloads)} files to R2")


if __name__ == "__main__":
    main()
