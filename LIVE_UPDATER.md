# Live Updater — Odds Movement Tracker

## Concept

After running the round sim, lock in our fair prices and then scrape live odds hourly to track how lines move relative to our edge. Stores snapshots to a "Live Updater" Google Sheet tab.

## Schedule

- **Sim run → midnight**: scrape every hour after the sim completes
- **6am → noon next day**: resume hourly scrapes (pre-round lines settling)
- Stops at noon (round starts, lines lock)

## What It Does

1. Round sim runs, produces fair prices for matchups + outrights
2. Live updater kicks off, scrapes Betcris, Pinnacle, BetOnline every hour
3. Each scrape: compare current market odds against our locked-in fair prices
4. Compute edge at each snapshot
5. Store to "Live Updater" tab in Google Sheets

## Data Per Snapshot

For each matchup/outright at each hourly scrape:

- `snapshot_time` — when this scrape ran
- `player` / `player_1` / `player_2` — who
- `market_type` — round_matchup, top_5, etc.
- `book` — betcris, pinnacle, betonline
- `market_odds` — current line from the book
- `fair_odds` — our locked-in fair price (from sim)
- `edge` — current edge (may have changed from sim-time)
- `edge_at_sim` — edge when we first ran the sim (for comparison)
- `odds_movement` — change from previous snapshot

## Questions to Resolve

- Run as a local cron/scheduled task, or GitHub Actions?
  - Local makes sense since it needs the sim's fair prices in memory
  - Could also save fair prices to a JSON file and have a standalone script read them
- How to handle the overnight gap (midnight → 6am)?
- Do we want Kalshi in this too, or just traditional books?
- Should it trigger alerts when edge crosses a threshold (e.g., edge was 5% at sim time, now 8%)?

## Implementation Sketch

- New script: `live_updater.py`
- Reads fair prices from sim output (CSV or JSON)
- Calls scrapers directly (same as sentinel but targeted)
- Appends rows to "Live Updater" sheet tab
- Runs via `schtasks` (Windows) or a simple loop with sleep

## Files to Create/Modify

- `live_updater.py` (new) — main script
- `sheets_storage.py` — add `store_live_updates()` + `LIVE_UPDATER_HEADERS`
- `round_sim.py` — export fair prices to a JSON file after sim for the updater to read
