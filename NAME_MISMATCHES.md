# Name Mismatch Resolution Guide

When the round sim sends a Telegram alert like this:

```
R2 Name Mismatches — Valspar Championship

3 scraped players not found in sim:
  • hovland, viktor  (betcris, pinnacle)
  • kim, si woo  (kalshi_outrights)
  • mcilroy, rory  (betonline)

Fix: add to name_replacements in sim_inputs.py
```

...it means scraped odds exist for players whose names don't match the sim's player list. Edges are being silently dropped.

## Quick Fix (< 1 min)

**1. Open `sim_inputs.py` and add to `name_replacements`:**

```python
name_replacements = {
    # existing entries...
    'hovland, viktor': 'hovland, victor',      # scraper has Viktor, DG has Victor
    'kim, si woo': 'kim, siwoo',               # spacing difference
    'mcilroy, rory': 'mc ilroy, rory',          # however DG spells it
}
```

**Key = the bad name** (from the Telegram alert, already in `last, first` lowercase format).
**Value = the canonical name** the sim uses (from DataGolf / `model_predictions_rN.csv`).

**2. Find the canonical name:**

```bash
# Check what the sim actually has
grep -i "hovland" model_predictions_r*.csv | head -3

# Or check DataGolf directly
grep -i "hovland" simulated_probs_live.csv | head -3
```

**3. Re-run the sim.** No other files need changes.

## How Names Flow Through the System

```
Sportsbook raw name  (e.g. "Viktor Hovland")
       |
       v
golf_scraping normalize_name()     →  "viktor hovland"
golf_scraping to_last_first()      →  "Hovland, Viktor"
       |
       v
Scraped JSON  (e.g. round_matchups_latest.json)
  p1_player_name: "Hovland, Viktor"
       |
       v
odds_loader.py lowercases          →  "hovland, viktor"
odds_loader.py applies name_replacements  →  "hovland, victor"  (if entry exists)
       |
       v
round_sim.py matches against sim_dict / finish_probs
```

For Kalshi outrights, the flow is slightly different — `price_kalshi_outrights()` has its own `norm()` function that converts `"First Last"` → `"last, first"` and applies `name_replacements`.

## Where `name_replacements` Is Applied

| File | Function | What It Joins Against |
|------|----------|----------------------|
| `odds_loader.py` | `_parse_datagolf_json()` | H2H matchup `sim_dict` keys |
| `round_sim.py` | `price_kalshi_outrights()` | `finish_probs["player_name"]` |
| `round_sim.py` | `price_round_score_lines()` | `card_lookup` keys |
| `new_sim.py` | main | `model_preds["player_name"]` |
| `grade_bets.py` | grading | API scores player names |

One entry in `name_replacements` fixes the name everywhere.

## Common Mismatch Patterns

| Pattern | Example | Fix |
|---------|---------|-----|
| Nickname vs full name | `cam` vs `cameron` | `'davis, cam': 'davis, cameron'` |
| Initials | `k.h.` vs `kyounghoon` | `'lee, k.h.': 'lee, kyounghoon'` |
| Accents stripped differently | `viktor` vs `victor` | `'hovland, viktor': 'hovland, victor'` |
| Spacing/hyphens | `si woo` vs `siwoo` | `'kim, si woo': 'kim, siwoo'` |
| Suffix handling | `stallings, stephen jr` vs `stallings jr., stephen` | map to canonical form |
| Multi-part last names | `norgaard, niklas` vs `norgaard moller, niklas` | `'norgaard, niklas': 'norgaard moller, niklas'` |

## When It's NOT a Name Mismatch

Sometimes the alert fires for players legitimately absent from the sim:

- **Player withdrew** after odds were posted but before tee times
- **Alternate** added to the field after predictions were generated
- **Different tour** — scraper picked up a DP World matchup, sim only has PGA

In these cases, ignore the alert. The sim correctly skips players it can't price.

## Debugging Deeper Issues

If the same name keeps mismatching week after week, the fix belongs in the scraper's normalization layer instead:

```
golf_scraping/utils/names.py → normalize_name()
```

This handles accent stripping, suffix removal, initial expansion, etc. Adding a case there prevents the mismatch from ever reaching the sim. But `name_replacements` in `sim_inputs.py` is the faster, safer fix for one-off issues.
