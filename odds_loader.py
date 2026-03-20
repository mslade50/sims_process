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

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Paths to check for scraped odds (in priority order)
SIMS_ROOT = Path(__file__).parent
SCRAPED_PATHS = [
    # Local: sibling golf_scraping repo (Documents)
    Path.home() / "Documents" / "golf_scraping" / "data",
    # Local: sibling golf_scraping repo (relative)
    SIMS_ROOT.parent / "golf_scraping" / "data",
    # CI: fetched into permanent_data before sim runs
    SIMS_ROOT / "permanent_data" / "scraped_odds",
]

DATAGOLF_BASE = "https://feeds.datagolf.com/betting-tools/matchups"
MAX_AGE_HOURS = 6  # ignore scraped files older than this

# Books we scrape ourselves — prefer our odds over DataGolf for these
SCRAPED_BOOKS = {"betonline", "pinnacle", "betcris"}


def _find_scraped_json(market: str) -> Path | None:
    """Find the most recent scraped JSON file for the given market."""
    filename = f"{market}_latest.json"
    for base in SCRAPED_PATHS:
        path = base / filename
        if path.exists():
            # Check freshness
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            age_hours = (datetime.now(timezone.utc) - mtime).total_seconds() / 3600
            if age_hours > MAX_AGE_HOURS:
                logger.info(f"Scraped odds stale ({age_hours:.1f}h old): {path}")
                continue
            logger.info(f"Found scraped odds ({age_hours:.1f}h old): {path}")
            return path
    return None


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
            if book == "datagolf":
                continue
            # Kalshi H2H: use mid odds (maker) when available — liquidity too thin for taker
            p1_odds = odds.get("p1")
            p2_odds = odds.get("p2")
            if book == "kalshi" and odds.get("p1_mid"):
                p1_odds = odds["p1_mid"]
                p2_odds = odds.get("p2_mid", p2_odds)
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

    # 1. Load scraped odds for our books
    if not force_api:
        path = _find_scraped_json(market)
        if path:
            try:
                with open(path) as f:
                    data = json.load(f)
                scraped_df = _parse_datagolf_json(data)
                scraped_df["source"] = "scraped"
                logger.info(f"Scraped: {len(scraped_df)} lines from {path.name}")
            except Exception as e:
                logger.warning(f"Failed to parse {path}: {e}")

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
        # Keep API lines for books we DON'T scrape
        api_other = api_df[~api_df["Bookmaker"].isin(SCRAPED_BOOKS)]

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
