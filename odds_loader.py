"""Load matchup odds from scraped JSON + DataGolf API (merged).

For books we scrape (betonline, pinnacle, betcris): use our scraped odds
when fresh, giving us earlier/faster lines than DataGolf aggregates.

For all other books (draftkings, fanduel, bet365, etc.): always pull from
DataGolf API since we don't scrape them.

Resolution:
  1. Load scraped JSON (local sibling repo or CI artifact)
  2. Load DataGolf API (always, for non-scraped books + DG model odds)
  3. Merge: scraped books take priority, API fills in the rest

Usage:
    from odds_loader import load_matchup_odds

    # Returns DataFrame with: Player 1, Player 2, Bookmaker, P1 Odds, P2 Odds, Ties, source
    df = load_matchup_odds("round_matchups")
    df = load_matchup_odds("tournament_matchups")
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Paths to check for scraped odds (in priority order)
SIMS_ROOT = Path(__file__).parent
SCRAPED_PATHS = [
    # CI: fetched into permanent_data before sim runs
    SIMS_ROOT / "permanent_data" / "scraped_odds",
]

# GitHub API URL for scraped odds (primary source — always fresh from CI)
# Uses GitHub API (works for private repos when GH_TOKEN is set)
GITHUB_REPO = "mslade50/golf_scraping"
GITHUB_BRANCH = "master"
GITHUB_DATA_PATH = "data"

DATAGOLF_BASE = "https://feeds.datagolf.com/betting-tools/matchups"
MAX_AGE_HOURS = 6  # ignore scraped files older than this

# Books we scrape ourselves — prefer our odds over DataGolf for these
SCRAPED_BOOKS = {"betonline", "pinnacle", "betcris"}


def _fetch_scraped_json(market: str) -> dict | None:
    """Fetch scraped odds JSON from GitHub, falling back to local paths.

    Returns parsed JSON dict or None if unavailable/stale.
    """
    filename = f"{market}_latest.json"

    # 1. Try GitHub API (always has the latest CI-committed data)
    gh_token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
    api_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_DATA_PATH}/{filename}?ref={GITHUB_BRANCH}"
    try:
        headers = {"Accept": "application/vnd.github.raw+json"}
        if gh_token:
            headers["Authorization"] = f"Bearer {gh_token}"
        resp = requests.get(api_url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        # Check freshness via last_updated field
        last_updated = data.get("last_updated", "")
        if last_updated:
            try:
                ts = datetime.strptime(last_updated, "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=timezone.utc)
                age_hours = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
                if age_hours > MAX_AGE_HOURS:
                    logger.info(f"GitHub scraped odds stale ({age_hours:.1f}h old)")
                else:
                    logger.info(f"Fetched scraped odds from GitHub ({age_hours:.1f}h old, {len(data.get('match_list', []))} matchups)")
                    return data
            except ValueError:
                # Can't parse timestamp, use it anyway
                logger.info(f"Fetched scraped odds from GitHub (unknown age)")
                return data
        else:
            logger.info(f"Fetched scraped odds from GitHub (no timestamp)")
            return data
    except Exception as e:
        logger.warning(f"GitHub fetch failed for {filename}: {e}")

    # 2. Fall back to local paths
    for base in SCRAPED_PATHS:
        path = base / filename
        if path.exists():
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            age_hours = (datetime.now(timezone.utc) - mtime).total_seconds() / 3600
            if age_hours > MAX_AGE_HOURS:
                logger.info(f"Local scraped odds stale ({age_hours:.1f}h old): {path}")
                continue
            logger.info(f"Found local scraped odds ({age_hours:.1f}h old): {path}")
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to parse {path}: {e}")

    return None


def load_betcris_outrights() -> pd.DataFrame:
    """Load scraped Betcris outright odds (winner / top_5 / top_10 / top_20).

    Source: betcris_outrights_latest.json published by sentinel.py in the
    golf_scraping repo. Falls back to local permanent_data/scraped_odds/.

    Returns DataFrame with columns:
        player_name (lower-case "first last")
        market_type (winner / top_5 / top_10 / top_20)
        american_odds (int)
        decimal_odds (float)

    Returns an empty DataFrame if the file is unavailable or empty. The
    caller is responsible for `if df.empty: skip-fallback`.
    """
    data = _fetch_scraped_json("betcris_outrights")
    if not data:
        return pd.DataFrame()

    rows = []
    for line in data.get("lines", []):
        # JSON stores names as "Last, First". new_sim's player_name pipeline
        # uses lowercase "first last" — convert here so the merge keys align.
        raw = str(line.get("player", "")).strip()
        if "," in raw:
            last, first = [p.strip() for p in raw.split(",", 1)]
            player = f"{first} {last}".lower()
        else:
            player = raw.lower()
        if not player:
            continue

        try:
            am = int(line["odds"])
        except (KeyError, TypeError, ValueError):
            continue
        # American -> decimal
        if am > 0:
            dec = am / 100.0 + 1.0
        elif am < 0:
            dec = 100.0 / abs(am) + 1.0
        else:
            continue

        rows.append({
            "player_name": player,
            "market_type": str(line.get("market_type", "")),
            "bookmaker": "betcris",
            "american_odds": am,
            "decimal_odds": dec,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["player_name", "market_type"], keep="first")
    return df


def _parse_datagolf_json(data: dict) -> pd.DataFrame:
    """Parse DataGolf-schema JSON into the standard matchup DataFrame.

    Works for both our scraped JSON and the DataGolf API response
    since they share the same schema.
    """
    rows = []
    for match in data.get("match_list", []):
        p1 = match.get("p1_player_name", "").lower()
        p2 = match.get("p2_player_name", "").lower()
        ties = match.get("ties", "unknown")

        for book, odds in match.get("odds", {}).items():
            if book in ("datagolf", "kalshi"):
                continue  # Kalshi H2H handled separately by price_kalshi_matchups_tourney
            p1_odds = odds.get("p1")
            p2_odds = odds.get("p2")
            rows.append({
                "Player 1": p1,
                "Player 2": p2,
                "Bookmaker": book,
                "P1 Odds": p1_odds,
                "P2 Odds": p2_odds,
                "DG_p1": match.get("odds", {}).get("datagolf", {}).get("p1"),
                "DG_p2": match.get("odds", {}).get("datagolf", {}).get("p2"),
                "Ties": ties,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        # Apply name_replacements for consistency with sim player names
        try:
            from sim_inputs import name_replacements
            df["Player 1"] = df["Player 1"].replace(name_replacements)
            df["Player 2"] = df["Player 2"].replace(name_replacements)
        except ImportError:
            pass
        df = df.drop_duplicates(subset=["Player 1", "Player 2", "Bookmaker"], keep="first")
        df["P1 Odds"] = pd.to_numeric(df["P1 Odds"], errors="coerce")
        df["P2 Odds"] = pd.to_numeric(df["P2 Odds"], errors="coerce")
    return df


def _fetch_datagolf_api(market: str, api_key: str) -> pd.DataFrame:
    """Fetch odds from DataGolf API."""
    params = {
        "tour": "pga",
        "market": market,
        "odds_format": "american",
        "file_format": "json",
        "key": api_key,
    }
    try:
        resp = requests.get(DATAGOLF_BASE, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        df = _parse_datagolf_json(data)
        logger.info(f"DataGolf API: {len(df)} lines across {df['Bookmaker'].nunique() if not df.empty else 0} books")
        return df
    except Exception as e:
        logger.error(f"DataGolf API failed: {e}")
        return pd.DataFrame()


def load_matchup_odds(
    market: str = "round_matchups",
    api_key: str | None = None,
    force_api: bool = False,
) -> pd.DataFrame:
    """Load matchup odds — merges scraped odds with DataGolf API.

    For scraped books (betonline, pinnacle, betcris): uses our fresh scraped
    odds when available, which arrive ahead of DataGolf's aggregation.

    For all other books: always pulls from DataGolf API.

    Args:
        market: "round_matchups" or "tournament_matchups"
        api_key: DataGolf API key (falls back to DATAGOLF_API_KEY env var)
        force_api: skip scraped files, use DataGolf API for everything

    Returns:
        DataFrame with columns: Player 1, Player 2, Bookmaker, P1 Odds, P2 Odds,
                                DG_p1, DG_p2, Ties, source
    """
    api_key = api_key or os.getenv("DATAGOLF_API_KEY")
    scraped_df = pd.DataFrame()
    api_df = pd.DataFrame()

    # 1. Load scraped odds for our books (GitHub -> local fallback)
    if not force_api:
        scraped_data = _fetch_scraped_json(market)
        if scraped_data:
            try:
                scraped_df = _parse_datagolf_json(scraped_data)
                scraped_df["source"] = "scraped"
                logger.info(f"Scraped: {len(scraped_df)} lines")
            except Exception as e:
                logger.warning(f"Failed to parse scraped data: {e}")

    # 2. Always fetch DataGolf API (for non-scraped books + DG model odds)
    if api_key:
        api_df = _fetch_datagolf_api(market, api_key)
        if not api_df.empty:
            api_df["source"] = "datagolf_api"
    else:
        logger.info("No DATAGOLF_API_KEY — skipping API fetch")

    # 3. Merge: scraped books win, API fills the rest
    if not scraped_df.empty and not api_df.empty:
        # Keep scraped lines for books we scrape
        scraped_lines = scraped_df[scraped_df["Bookmaker"].isin(SCRAPED_BOOKS)]
        # For scraped books missing from scraped data, fall back to API
        scraped_books_present = set(scraped_lines["Bookmaker"].unique())
        missing_scraped = SCRAPED_BOOKS - scraped_books_present
        # Keep API lines for books we DON'T scrape + any scraped books that are missing
        api_other = api_df[
            (~api_df["Bookmaker"].isin(SCRAPED_BOOKS)) |
            (api_df["Bookmaker"].isin(missing_scraped))
        ]
        if missing_scraped:
            logger.info(f"Scraped data missing {missing_scraped} — falling back to API for those")

        # Also grab DG model odds from API for matchups the scraper found
        # (our scraped JSON won't have datagolf model odds)
        if "DG_p1" in api_df.columns and not scraped_lines.empty:
            # Build a lookup of DG model odds by player pair
            dg_odds_lookup = {}
            for _, row in api_df.iterrows():
                key = (row["Player 1"], row["Player 2"])
                if pd.notna(row.get("DG_p1")):
                    dg_odds_lookup[key] = (row["DG_p1"], row["DG_p2"])
            # Backfill DG model odds into scraped lines
            for idx, row in scraped_lines.iterrows():
                key = (row["Player 1"], row["Player 2"])
                if key in dg_odds_lookup:
                    scraped_lines.loc[idx, "DG_p1"] = dg_odds_lookup[key][0]
                    scraped_lines.loc[idx, "DG_p2"] = dg_odds_lookup[key][1]

        df = pd.concat([scraped_lines, api_other], ignore_index=True)
        scraped_books = scraped_lines["Bookmaker"].nunique() if not scraped_lines.empty else 0
        api_books = api_other["Bookmaker"].nunique() if not api_other.empty else 0
        logger.info(f"Merged: {scraped_books} scraped books + {api_books} API books = {len(df)} total lines")

    elif not scraped_df.empty:
        df = scraped_df
        logger.info(f"Using scraped odds only ({len(df)} lines)")
    elif not api_df.empty:
        df = api_df
        logger.info(f"Using DataGolf API only ({len(df)} lines)")
    else:
        logger.warning("No odds available from any source")
        df = pd.DataFrame()

    if not df.empty:
        df = df.drop_duplicates(subset=["Player 1", "Player 2", "Bookmaker"], keep="first")

    return df


def scrape_betonline_live(market_type: str = "tournament_matchup") -> pd.DataFrame:
    """Run BetOnline scraper directly via Playwright for fresh odds.

    Used on Mondays before 5 PM EST when scraped JSONs and DataGolf API
    don't have lines yet but BetOnline has already posted.

    Returns DataFrame in standard format: Player 1, Player 2, Bookmaker,
    P1 Odds, P2 Odds, source.
    """
    # Import scraper from golf_scraping repo
    scraping_repo = Path(r"C:\Users\mckin\Documents\golf_scraping")
    if str(scraping_repo) not in sys.path:
        sys.path.insert(0, str(scraping_repo))

    try:
        from scrapers.betonline import BetOnlineScraper
    except ImportError as e:
        logger.error(f"Cannot import BetOnlineScraper: {e}")
        return pd.DataFrame()

    scraper = BetOnlineScraper(headless=True)
    try:
        matchups = asyncio.run(scraper.scrape(market_type=market_type))
    except Exception as e:
        logger.error(f"BetOnline live scrape failed: {e}")
        return pd.DataFrame()

    if not matchups:
        logger.info("BetOnline live scrape returned no matchups")
        return pd.DataFrame()

    from utils.names import to_last_first

    rows = []
    for m in matchups:
        rows.append({
            "Player 1": to_last_first(m.player_a).lower(),
            "Player 2": to_last_first(m.player_b).lower(),
            "Bookmaker": "betonline",
            "P1 Odds": m.odds_a,
            "P2 Odds": m.odds_b,
            "DG_p1": None,
            "DG_p2": None,
            "Ties": "no_tie",
            "source": "betonline_live",
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        try:
            from sim_inputs import name_replacements
            df["Player 1"] = df["Player 1"].replace(name_replacements)
            df["Player 2"] = df["Player 2"].replace(name_replacements)
        except ImportError:
            pass
        df["P1 Odds"] = pd.to_numeric(df["P1 Odds"], errors="coerce")
        df["P2 Odds"] = pd.to_numeric(df["P2 Odds"], errors="coerce")
        df = df.drop_duplicates(subset=["Player 1", "Player 2", "Bookmaker"], keep="first")

    logger.info(f"BetOnline live scrape: {len(df)} tournament matchup lines")
    return df
