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


def _target_event_ids() -> set[str]:
    """This week's canonical DataGolf event id(s) from sim_inputs (as strings).

    The scraped JSON now tags each line/file with the DataGolf event_id; we use
    this to keep only the current event when a file spans multiple events (e.g.
    a live event plus the next major's winner board). Empty set => don't filter.
    """
    try:
        from sim_inputs import event_ids
        return {str(e).strip() for e in event_ids if str(e).strip()}
    except Exception:
        return set()


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


def _scraped_age_hours(data: dict) -> "float | None":
    """Age in hours from a scraped file's `last_updated` (UTC), or None if the
    field is missing/unparseable."""
    last_updated = (data or {}).get("last_updated", "")
    if not last_updated:
        return None
    try:
        ts = datetime.strptime(last_updated, "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - ts).total_seconds() / 3600
    except ValueError:
        return None


def guard_scraped_data(data, market, *, round=None, event_ids=None,
                       max_age_hours=MAX_AGE_HOURS):
    """Apply the standard scraped-odds guards; return a safe-to-use dict (rows
    possibly filtered) or None if the file must be rejected. SHARED by
    load_matchup_odds and push_odds_screen so the sim pricer and the odds screen
    can never diverge on what they trust.

    Guards, in order:
      1. Freshness: reject if last_updated is older than max_age_hours. A
         missing/unparseable timestamp = unknown age = passes (matches the
         existing GitHub-fetch behaviour; the DataGolf fallback covers it).
      2. event_id: reject if the top-level event_id is set and not in event_ids.
      3. file-round (round_matchups / round_scores only): reject if the top-level
         `round` is set and != target. A MISSING top-level round is SUSPECT and
         rejected when a target round is supplied — under the day-of-week rule the
         board always stamps the round, so a missing one signals a stale/foreign
         writer (closes the old 'round=None silently disables the guard' gap).
      4. per-row round: keep only rows whose `round` is in (None, 0, round).
    """
    if not data:
        return None

    age = _scraped_age_hours(data)
    if age is not None and age > max_age_hours:
        logger.info(f"Scraped {market} stale ({age:.1f}h old) — rejecting")
        return None

    targets = event_ids if event_ids is not None else _target_event_ids()
    sid = str((data or {}).get("event_id") or "").strip()
    if targets and sid and sid not in targets:
        logger.warning(f"Scraped {market} is for event {sid}, not target {targets} — rejecting")
        return None

    if market in ("round_matchups", "round_scores") and round is not None:
        file_round = data.get("round")
        if file_round is None:
            logger.warning(f"Scraped {market} has no top-level round but target is R{round} "
                           f"— rejecting as suspect (writer must stamp round)")
            return None
        if file_round != round:
            logger.warning(f"Scraped {market} is for R{file_round}, not target R{round} — rejecting")
            return None
        # Defensive per-row scope (untagged round-0 rows pass through).
        if "match_list" in data:
            data = {**data, "match_list": [m for m in data.get("match_list", [])
                                           if m.get("round") in (None, 0, round)]}
        if "lines" in data:
            data = {**data, "lines": [l for l in data.get("lines", [])
                                      if l.get("round") in (None, 0, round)]}
    return data


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

    # The betcris outrights file can hold multiple events at once (the current
    # event plus the next major's winner board), each line tagged with its
    # DataGolf event_id. Keep only this week's event so a player who appears in
    # both (e.g. Scheffler to win here AND to win the next major) doesn't collide
    # on the (player, market_type) dedup below.
    lines = data.get("lines", [])
    targets = _target_event_ids()
    if targets:
        scoped = [l for l in lines if str(l.get("event_id") or "").strip() in targets]
        if scoped:
            lines = scoped
        else:
            logger.warning(
                "No betcris outright lines matched target event_ids %s "
                "(event_id missing/unresolved?) — using all lines unfiltered", targets
            )

    rows = []
    for line in lines:
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
            "event_id": str(line.get("event_id") or "").strip(),
            "bookmaker": "betcris",
            "american_odds": am,
            "decimal_odds": dec,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        # Include event_id in the key so distinct events never overwrite each
        # other even in the unfiltered fallback path.
        df = df.drop_duplicates(subset=["player_name", "market_type", "event_id"], keep="first")
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


def _fetch_datagolf_api(market: str, api_key: str, target_round: int | None = None) -> pd.DataFrame:
    """Fetch odds from DataGolf API.

    DataGolf stamps round_matchups responses with a top-level `round_num`. When
    `target_round` is supplied and DataGolf is serving a DIFFERENT round, the
    whole response is stale prior/next-round data — we drop it so this round's
    bets are never priced or stored against another round's DataGolf prices.
    This is the API-side equivalent of guard_scraped_data()'s file-round check
    (the scraped guard only protects the scraped feed; DataGolf round_num gates
    the API feed).
    """
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
        if market == "round_matchups" and target_round is not None:
            dg_round = data.get("round_num")
            try:
                dg_round_int = int(dg_round) if dg_round is not None else None
            except (TypeError, ValueError):
                dg_round_int = None
            if dg_round_int is not None and dg_round_int != int(target_round):
                logger.warning(
                    f"DataGolf round_matchups is for R{dg_round_int}, not target "
                    f"R{target_round} — dropping API lines as stale prior-round odds"
                )
                return pd.DataFrame()
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
    round: int | None = None,
) -> pd.DataFrame:
    """Load matchup odds — merges scraped odds with DataGolf API.

    For scraped books (betonline, pinnacle, betcris): uses our fresh scraped
    odds when available, which arrive ahead of DataGolf's aggregation.

    For all other books: always pulls from DataGolf API.

    Args:
        market: "round_matchups" or "tournament_matchups"
        api_key: DataGolf API key (falls back to DATAGOLF_API_KEY env var)
        force_api: skip scraped files, use DataGolf API for everything
        round: when set (for round_matchups), only use scraped odds for THIS round.
            The scraped file is stamped with the round it holds and each row carries
            a `round`; if the file is for a different round (e.g. R2 lines still up
            while we're pricing R3, before the scraper has posted/scoped R3), the
            scraped sharp-book lines are dropped so we don't price this round's fairs
            against last round's prices. None = no round check (tournament matchups).

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
        # All event/round/freshness guards live in guard_scraped_data() now, shared
        # with push_odds_screen so the sim and the odds screen trust identical data.
        scraped_data = guard_scraped_data(scraped_data, market, round=round)
        if scraped_data:
            try:
                scraped_df = _parse_datagolf_json(scraped_data)
                scraped_df["source"] = "scraped"
                logger.info(f"Scraped: {len(scraped_df)} lines")
            except Exception as e:
                logger.warning(f"Failed to parse scraped data: {e}")

    # 2. Always fetch DataGolf API (for non-scraped books + DG model odds).
    #    Pass the target round so DataGolf's own round_num gates out stale
    #    prior-round odds before they can be priced/stored.
    if api_key:
        api_df = _fetch_datagolf_api(market, api_key, target_round=round)
        if not api_df.empty:
            api_df["source"] = "datagolf_api"
    else:
        logger.info("No DATAGOLF_API_KEY — skipping API fetch")

    # 3. Merge: scraped books win, API fills in ONLY the books we don't scrape.
    if not scraped_df.empty and not api_df.empty:
        # Keep scraped lines for books we scrape
        scraped_lines = scraped_df[scraped_df["Bookmaker"].isin(SCRAPED_BOOKS)]
        # NEVER backfill a sharp/scraped book (betonline, pinnacle, betcris) from
        # DataGolf: DG's aggregate carries stale/phantom lines for these books
        # (e.g. a Pinnacle price for a matchup Pinnacle isn't actually offering,
        # which scanned and stored as real). If our direct scrape doesn't have a
        # sharp-book line, the correct answer is "that book isn't offering it",
        # not "ask DataGolf". So API contributes only books we don't scrape.
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
        # Even with no scraped data at all, don't let DataGolf stand in for our
        # sharp books — their prices must come from our own scrape or not at all.
        df = api_df[~api_df["Bookmaker"].isin(SCRAPED_BOOKS)]
        logger.info(f"Using DataGolf API only, sharp books excluded ({len(df)} lines)")
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
