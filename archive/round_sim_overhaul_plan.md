# Round Sim Overhaul: Live Outright & Finish Position Pricing

## Table of Contents
1. [Google Sheet Changes](#1-google-sheet-changes)
2. [Auto-Populate Script: `update_sheet_courses.py`](#2-auto-populate-script)
3. [Implementation Plan: `round_sim.py`](#3-implementation-plan)
4. [Data Flow Diagrams](#4-data-flow-diagrams)
5. [File Dependencies](#5-file-dependencies)

---

## 1. Google Sheet Changes

### Tab: `round_config`

Add these new rows to your existing `round_config` tab. **Column A** = parameter name, **Column B** = value, **Column C** = optional notes.

#### New Rows to Add

| Row | Column A (Parameter) | Column B (Example Value) | Column C (Notes) |
|-----|---------------------|--------------------------|-------------------|
| NEW | `course_codes` | `TS` | Comma-separated `course_x` values from API. Auto-populated by script. Single course = one value. Multi-course example: `PB,SG` |
| NEW | `course_pars` | `72` | Comma-separated par values matching `course_codes` order. Multi-course example: `72,72` |
| NEW | `expected_score_r2` | `71.8` | Expected scoring avg for R2. Multi-course: comma-separated matching `course_codes` order, e.g. `71.5,72.3`. Leave blank to use `expected_score_1` |
| NEW | `expected_score_r3` | `71.5` | Same format as R2. Leave blank to use `expected_score_1` |
| NEW | `expected_score_r4` | `71.2` | Same format as R2. Leave blank to use `expected_score_1` |
| NEW | `wind_r2` | `5,5,6,6,7,7,8,8,7,7,6,6,5,5,5` | Hourly wind for R2 (6AM onward). Leave blank if `wind` should be used for all rounds |
| NEW | `wind_r3` | | Same format. Leave blank = use `wind` |
| NEW | `wind_r4` | | Same format. Leave blank = use `wind` |
| NEW | `dew_r2` | `36,36,38,38,40,40,42,42,44,44,44,42,40,38,36` | Hourly dew for R2. Leave blank = use `dew` |
| NEW | `dew_r3` | | Same format. Leave blank = use `dew` |
| NEW | `dew_r4` | | Same format. Leave blank = use `dew` |

#### Complete Sheet Layout After Changes

Your `round_config` tab should look like this (rows 1+ in the sheet):

```
Row  | A (Parameter)       | B (Value)                              | C (Notes)
-----|---------------------|----------------------------------------|-------------------
1    | Parameter           | Value                                  | Notes (header row)
2    | round               | 1                                      | Round just completed
3    | expected_score_1    | 0.15                                   | Next-round score adj (existing behavior)
4    | expected_score_2    |                                        | 2nd course next-round adj (existing)
5    | expected_score_3    |                                        | 3rd course next-round adj (existing)
6    | wind                | 5,5,5,5,5,5,5,5,5,5,5,5,5,5,5         | Next-round wind (existing)
7    | dew                 | 36,36,38,38,38,38,38,40,40,42,44,44   | Next-round dew (existing)
8    | wind_paste          |                                        | (existing)
9    | dew_paste           |                                        | (existing)
10   | dew_calculation     |                                        | (existing)
11   | wind_override       |                                        | (existing)
12   | course_codes        | TS                                     | ← NEW: auto-populated
13   | course_pars         | 72                                     | ← NEW: you fill this
14   | expected_score_r2   | 71.8                                   | ← NEW: for tournament sim
15   | expected_score_r3   | 71.5                                   | ← NEW
16   | expected_score_r4   | 71.2                                   | ← NEW
17   | wind_r2             |                                        | ← NEW: blank = use row 6
18   | wind_r3             |                                        | ← NEW
19   | wind_r4             |                                        | ← NEW
20   | dew_r2              |                                        | ← NEW
21   | dew_r3              |                                        | ← NEW
22   | dew_r4              |                                        | ← NEW
```

#### How the Code Reads These

**`course_codes` + `course_pars`:**
- Parsed as comma-separated lists, zipped into a mapping: `{"TS": 72}` or `{"PB": 72, "SG": 72}`
- Each player's `course_x` from the live model file is matched against this mapping
- If `course_codes` is blank → falls back to single `PAR` from sim_inputs (current behavior)

**`expected_score_rN`:**
- For single-course: a single float like `71.8`
- For multi-course: comma-separated matching `course_codes` order, e.g. `71.5,72.3`
- If blank → falls back to `expected_score_1` (the "next round" value from existing config)
- Used by the tournament sim to convert SG draws to integer strokes per future round

**`wind_rN` / `dew_rN`:**
- Comma-separated hourly arrays, same format as existing `wind` / `dew`
- If blank → the sim uses `wind` / `dew` (the generic next-round arrays) for all future rounds
- This lets you enter different forecasts per round when you have them

---

## 2. Auto-Populate Script

### `update_sheet_courses.py`

Run this **before R1 finishes** (e.g., Wednesday/Thursday) to auto-populate `course_codes` in the sheet:

```
python update_sheet_courses.py
```

**What it does:**
1. Calls DataGolf `field-updates` API
2. Extracts unique `course` values from the field data
3. Writes them as comma-separated string to the `course_codes` cell in the sheet
4. Prints the mapping so you know which codes to use for `course_pars` and expected scores

**You still manually fill in:** `course_pars`, `expected_score_r2/r3/r4` — because these require your judgment (course-specific par, weather forecast, etc.)

### Script Logic

```python
"""
update_sheet_courses.py — Auto-populate course_codes in Google Sheet

Run before R1 to detect course codes from the DataGolf field API.
Writes course_codes to the round_config sheet tab.

Usage:
    python update_sheet_courses.py
"""

import os
import json
import requests
import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DATAGOLF_API_KEY")
SHEET_NAME = "golf_sims"
TAB_NAME = "round_config"

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",  # NOTE: read-write, not read-only
    "https://www.googleapis.com/auth/drive",
]


def get_course_codes():
    """Fetch unique course codes from DataGolf field-updates API."""
    params = {"tour": "pga", "file_format": "json", "key": API_KEY}
    resp = requests.get("https://feeds.datagolf.com/field-updates", params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    tourney_name = data.get("tournament_name", "Unknown")
    field = data.get("field", [])

    # Extract unique course values
    courses = set()
    for player in field:
        course = player.get("course")
        if course and str(course).strip():
            courses.add(str(course).strip())

    courses_sorted = sorted(courses)
    print(f"Tournament: {tourney_name}")
    print(f"Field size: {len(field)} players")
    print(f"Course codes found: {courses_sorted}")

    return courses_sorted, tourney_name


def connect_sheet_writable():
    """Connect to Google Sheet with WRITE permissions."""
    creds_json = os.getenv("GOOGLE_CREDS_JSON")
    if creds_json:
        creds_dict = json.loads(creds_json)
        creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    else:
        creds = Credentials.from_service_account_file("credentials.json", scopes=SCOPES)

    client = gspread.authorize(creds)
    spreadsheet = client.open(SHEET_NAME)
    return spreadsheet.worksheet(TAB_NAME)


def update_course_codes(ws, course_codes):
    """Find the course_codes row and update its value."""
    all_values = ws.get("A:B")

    target_row = None
    for i, row in enumerate(all_values):
        if len(row) >= 1 and row[0].strip().lower() == "course_codes":
            target_row = i + 1  # 1-indexed for gspread
            break

    if target_row is None:
        # Append it
        next_row = len(all_values) + 1
        ws.update_cell(next_row, 1, "course_codes")
        ws.update_cell(next_row, 2, ",".join(course_codes))
        print(f"Added course_codes at row {next_row}: {','.join(course_codes)}")
    else:
        ws.update_cell(target_row, 2, ",".join(course_codes))
        print(f"Updated course_codes at row {target_row}: {','.join(course_codes)}")


if __name__ == "__main__":
    courses, tourney = get_course_codes()

    if not courses:
        print("No course codes found in API response. Single-course week or API issue.")
    else:
        ws = connect_sheet_writable()
        update_course_codes(ws, courses)

        print(f"\n{'='*50}")
        print(f"NEXT STEPS — fill in manually:")
        print(f"{'='*50}")
        for i, code in enumerate(courses):
            print(f"  Course {i+1}: '{code}'")
        print(f"\n  → Set 'course_pars' to the par for each course (comma-separated, same order)")
        print(f"  → Set 'expected_score_rN' values (comma-separated if multi-course)")
        print(f"\nExample for Pebble Beach week:")
        print(f"  course_codes:      PB,SG,ML")
        print(f"  course_pars:       72,72,71")
        print(f"  expected_score_r2: 72.1,71.5,70.8")
```

### Google Sheet Permissions Note

Your current `sheet_config.py` uses **read-only** scopes. This new script needs **write** access. Two options:

1. **Recommended**: Keep `sheet_config.py` as read-only (it only reads). Give `update_sheet_courses.py` its own write scopes (as shown above). Same credentials file works for both — the scopes are per-connection, not per-credential.
2. **Alternative**: Upgrade both to read-write scopes. No downside other than principle-of-least-privilege.

---

## 3. Implementation Plan

### Overview: What Changes in `round_sim.py`

The current `round_sim.py` does:
- Simulate ONE round → matchups + score card

The new `round_sim.py` will do:
- Simulate ONE round → matchups + score card (**unchanged**)
- **NEW**: Simulate REMAINING rounds → outrights + finish positions (win, top 5/10/20)
- **NEW**: Fetch outright/top-N market odds from DataGolf
- **NEW**: Calculate edges + Kelly sizing
- **NEW**: Email with matchup picks + outright/finish picks

### Architecture: New Functions

```
round_sim.py (expanded)
│
├── EXISTING (unchanged)
│   ├── simulate_round_scores()      — single round Normal(skill, STD_DEV)
│   ├── fetch_matchup_odds()         — DataGolf round_matchups API
│   ├── price_matchups()             — fair win probs from sim
│   ├── calculate_edges()            — edge calculation
│   ├── build_matchup_outputs()      — filter + format
│   ├── build_score_card()           — fair under prices
│   └── export_results()             — Excel + CSV
│
├── NEW: Config Loading
│   ├── load_tournament_config()     — read new sheet fields
│   └── build_course_mapping()       — course_codes → par + expected_avg
│
├── NEW: Tournament Simulation Engine
│   ├── load_known_rounds()          — read actual scores/SG from rN_live_model.csv
│   ├── load_player_params()         — dists + correlation → Cholesky per player
│   ├── estimate_missing_categories()— for no-ShotLink courses
│   ├── simulate_remaining_rounds()  — MAIN: sim R(N+1) through R4
│   │   ├── sim one round (Normal draw + category decomposition)
│   │   ├── apply cut after R2 (if applicable)
│   │   ├── apply R→R+1 skill updates (coefficient buckets)
│   │   └── accumulate integer strokes
│   └── compute_finish_probabilities()— win + dead-heat top 5/10/20
│
├── NEW: Outright/Finish Market Pricing
│   ├── fetch_outright_odds()        — DataGolf outrights API (win, top_5, top_10, top_20)
│   ├── price_outrights()            — merge sim probs with market odds → edges
│   ├── apply_kelly_sizing()         — fractional Kelly per book group
│   └── build_finish_outputs()       — filter, format, sharp highlights
│
├── NEW: Export + Email (extended)
│   ├── export_results()             — add outright/finish tabs to Excel
│   └── build_email_html()           — add finish position section
│
└── main()                           — orchestrates everything
```

### Detailed Function Specifications

#### 3.1 `load_tournament_config()` → dict

Extends the existing sheet config reading to include the new fields:

```python
def load_tournament_config():
    """
    Read tournament-sim-specific config from Google Sheet.
    
    Returns dict with:
        course_map:     dict {course_code: {"par": int, "expected_r2": float, ...}}
        default_par:    int (from sim_inputs.PAR, used when course_map is empty)
        wind_arrays:    dict {2: [...], 3: [...], 4: [...]}
        dew_arrays:     dict {2: [...], 3: [...], 4: [...]}
        cut_line:       int (from sim_inputs.CUT_LINE)
        use_10_shot:    bool (from sim_inputs.USE_10_SHOT_RULE)
    """
```

**Fallback chain for expected scoring avg per round:**
1. `expected_score_rN` (round-specific from sheet)
2. `expected_score_1` (generic "next round" from sheet)
3. `PAR` from sim_inputs

**Fallback chain for wind/dew per round:**
1. `wind_rN` / `dew_rN` (round-specific from sheet)
2. `wind` / `dew` (generic "next round" from sheet)
3. Empty array → no weather adjustment for that round

#### 3.2 `load_known_rounds(completed_round, course_map)` → dict

```python
def load_known_rounds(completed_round, course_map):
    """
    Load actual scores and category SG from completed rounds.
    
    Reads: r1_live_model.csv, r2_live_model.csv, etc.
    
    For each completed round N (1..completed_round):
      - actual_strokes[player] = PAR(player_course) - total_sg  (or from 'total' col + course_par)
      - actual_cats[player] = [sg_ott, sg_app, sg_arg, sg_putt]
      - If sg_ott/app/arg/putt are NaN (no ShotLink):
          → estimate using categories_given_total_for_player()
    
    Returns dict:
        player_names:   list[str]
        strokes:        dict {round_num: np.array shape (n_players,)}
        categories:     dict {round_num: np.array shape (n_players, 4)}
        positions:      dict {round_num: np.array shape (n_players,)} — leaderboard position
        cumulative:     np.array shape (n_players,) — total strokes through completed rounds
        made_cut:       np.array[bool] shape (n_players,) — True if player made cut (or cut not yet applied)
        course_x:       dict {player: course_code} — per player course assignment
    """
```

**Key detail: Cut status detection**
- If `completed_round >= 2` AND R3 tee times exist in API → cut is official, use it
- If `completed_round >= 2` AND no R3 tee times → cut not official yet
  - Players currently outside CUT_LINE are excluded from sim
  - But `model_predictions_r3.csv` still generated for full field (by live_stats_engine, not us)

#### 3.3 `load_player_params(player_names)` → list[tuple]

```python
def load_player_params(player_names):
    """
    Load per-player category distributions and build Cholesky params.
    
    Reads: this_week_dists_adjusted.csv + correlation matrix files
    
    Same logic as new_sim.py's player_params construction:
      - Pivot mean_adj / std_adj by category
      - Fall back to global means/medians for missing players
      - Build (mu, std, Sigma, L, v, denom) per player
    
    Returns:
        player_params: list of tuples, indexed same as player_names
    """
```

#### 3.4 `simulate_remaining_rounds(...)` → np.array

This is the core new function. Here's the detailed logic per scenario:

```python
def simulate_remaining_rounds(
    completed_round,      # int: 1, 2, or 3
    player_names,         # list[str]
    known_strokes,        # dict {1: array, 2: array, ...} — actual integer strokes
    known_categories,     # dict {1: array(n,4), 2: array(n,4), ...}
    model_preds,          # DataFrame with scores_rN columns
    player_params,        # list of (mu, std, Sigma, L, v, denom)
    course_config,        # dict with per-round par + expected_avg per player
    num_sims,             # int
):
    """
    Simulate from round (completed_round + 1) through R4.
    
    Returns:
        final_scores: np.array shape (n_players, num_sims) — 72-hole integer totals
        made_cut_mask: np.array[bool] shape (n_players, num_sims)
        
    Internal flow (example: completed_round=1, simming R2+R3+R4):
    
    1. KNOWN R1:
       - strokes_r1 = known_strokes[1]                    # shape (n_players,)
       - cats_r1 = known_categories[1]                     # shape (n_players, 4)
       - Broadcast to (n_players, num_sims) by repeating
    
    2. SIMULATE R2:
       - sg_r2_pred = model_preds['scores_r2']             # from live_stats_engine
       - sg_r2 = Normal(sg_r2_pred, STD_DEV) per sim       # shape (n_players, num_sims)
       - cats_r2 = categories_given_total(sg_r2)            # decompose
       - strokes_r2 = round(PAR_r2[player] - sg_r2)        # per-player PAR from course_config
       - r1_r2_total = strokes_r1[:, None] + strokes_r2    # broadcast known R1
    
    3. CUT (if completed_round < 2):
       - Apply CUT_LINE + 10-shot rule to r1_r2_total per sim
       - made_cut_mask = True/False per player per sim
    
    4. R2→R3 SKILL UPDATE (coefficient buckets):
       - resid_r2 = sg_r2 - sg_r2_pred[:, None]
       - avg_ott = 0.5 * (cats_r1 + cats_r2)[:, :, 0]     # known R1 + simulated R2
       - Position buckets from r1_r2_total rankings
       - Apply coefficients_r2 / r2_6_30 / r2_30_up
       - updated_skill_r3 = skill_r2 + total_adjustment_r2
    
    5. SIMULATE R3:
       - If model_preds has 'scores_r3': use as base prediction
       - Else: use updated_skill_r3 as mean
       - sg_r3 = Normal(mean_r3, STD_DEV)
       - cats_r3 = categories_given_total(sg_r3)
       - strokes_r3 = round(PAR_r3[player] - sg_r3)
    
    6. R3→R4 SKILL UPDATE:
       - avg_ott_r3 = 0.66 * avg_ott_r2 + 0.34 * cats_r3[:,:,0]
       - Position buckets from r1_r3_total rankings
       - Apply coefficients_r3 / r3_mid / r3_high (SG-only, no residual)
       - Undo R2 adjustments, apply R3 adjustments
       - updated_skill_r4 = ...
    
    7. SIMULATE R4:
       - sg_r4 = Normal(mean_r4, STD_DEV)
       - strokes_r4 = round(PAR_r4[player] - sg_r4)
       - r3_r4[~made_cut] = 200  (penalty for missed cut)
    
    8. FINAL:
       - final_scores = strokes_r1 + strokes_r2 + strokes_r3 + strokes_r4
    """
```

**Scenario table — what gets simulated vs. known:**

| After Round | Known (actual) | Simulated | Cut Handling |
|-------------|----------------|-----------|--------------|
| R1 | R1 strokes + cats | R2, R3, R4 | Simulated per iteration after R2 |
| R2 (cut made) | R1+R2 strokes + cats | R3, R4 | Known: only made-cut players in sim |
| R2 (cut NOT made) | R1+R2 strokes + cats | R3, R4 | Exclude players outside CUT_LINE from sim; full field preds still available |
| R3 | R1+R2+R3 strokes + cats | R4 | Known: only made-cut players | 

#### 3.5 `compute_finish_probabilities(final_scores, player_names, made_cut_mask, num_sims)` → DataFrame

```python
def compute_finish_probabilities(final_scores, player_names, made_cut_mask, num_sims):
    """
    From simulated 72-hole totals, compute:
      - Win probability (playoff tiebreaker: random winner among tied)
      - Top 5/10/20 with dead-heat adjustment
    
    Same logic as new_sim.py's market section.
    
    Returns DataFrame with columns:
        player_name, simulated_win_prob, top_5, top_10, top_20
    """
```

#### 3.6 Outright/Finish Market Pricing

Reuses the `fetch_market_data()` / `extract_market_rows()` / Kelly sizing pattern from new_sim.py:

```python
def fetch_outright_odds(books_to_use):
    """Fetch win, top_5, top_10, top_20 from DataGolf outrights API."""
    
def price_outrights(finish_probs, market_odds):
    """
    Merge simulated probabilities with market odds.
    Calculate edges, Kelly stakes, book-group sizing.
    Filter by EDGE_THRESHOLD.
    
    Returns DataFrame with: player_name, market, bookmaker, decimal_odds,
        american_odds, my_fair, edge, stake, size, book_group, sample, my_pred
    """

def build_finish_outputs(priced_df, pred_lookup, sample_lookup):
    """
    Format for output:
      - Combined: all edges above threshold
      - Sharp: best price per player/market from sharp books
    """
```

#### 3.7 Extended Export + Email

The Excel workbook gains new tabs:
- `matchups_all` (existing)
- `matchups_sharp` (existing)
- `score_card` (existing)
- **`outrights`** (new) — win/top-N edges, all books
- **`outrights_sharp`** (new) — sharp book best prices
- **`finish_probs`** (new) — raw sim probabilities for all players

Email gains a new section between matchups and footer:
- Finish position edges table (same HTML pattern as existing matchup table)
- Filtered by pred > 0 and edge > threshold

### 3.8 Updated `main()` Flow

```python
def main():
    # 1. Read config (existing + new sheet fields)
    # 2. Load predictions for next round (existing)
    # 3. Simulate next round scores (existing → matchups + score card)
    
    # 4. NEW: Tournament simulation
    #    - Load known rounds from live model files
    #    - Load player params (dists + correlation)
    #    - Simulate remaining rounds through R4
    #    - Compute finish probabilities
    
    # 5. NEW: Outright market pricing
    #    - Fetch market odds
    #    - Calculate edges + sizing
    
    # 6. Export everything (extended Excel + email)
    # 7. Storage (existing Google Sheets storage)
```

---

## 4. Data Flow Diagrams

### After R1 (most complex case)

```
INPUTS:
  r1_live_model.csv ──────────────────┐ actual R1 strokes + SG categories
  model_predictions_r2.csv ───────────┤ weather-adjusted R2 skill predictions
  this_week_dists_adjusted.csv ───────┤ per-player category distributions
  sg_cat_corr_*.csv ──────────────────┤ correlation matrix
  pre_sim_summary_{tourney}.csv ──────┤ sample sizes
  Google Sheet (round_config) ────────┤ course_codes, pars, expected_scores, wind/dew
  DataGolf API ───────────────────────┘ matchup odds + outright odds

PROCESSING:
  ┌─────────────────────────────────────────────────────────────┐
  │ 1. Single-round sim (EXISTING)                              │
  │    scores_r2 → Normal draw → integer scores                 │
  │    → matchup pricing + score card                           │
  ├─────────────────────────────────────────────────────────────┤
  │ 2. Tournament sim (NEW)                                     │
  │    Known R1 (actual) + Sim R2 + Cut + Sim R3 + Sim R4      │
  │    With cascading skill updates between rounds              │
  │    → 72-hole totals → finish probabilities                  │
  ├─────────────────────────────────────────────────────────────┤
  │ 3. Market pricing (NEW)                                     │
  │    Sim probs vs DG API odds → edges + Kelly sizing          │
  └─────────────────────────────────────────────────────────────┘

OUTPUTS:
  {tourney}/round_{N}_sim_{HHMM}.xlsx
    ├── matchups_all          (existing)
    ├── matchups_sharp        (existing)
    ├── score_card            (existing)
    ├── outrights             (NEW)
    ├── outrights_sharp       (NEW)
    └── finish_probs          (NEW)
  
  {tourney}/fair_card_r{N}.csv              (existing)
  simulated_probs_live.csv                  (NEW — win probs)
  top_finish_probs_live_{tourney}.csv       (NEW — top 5/10/20)
  finish_equity_live_{tourney}.csv          (NEW — combined edges)
```

### After R2 (cut known)

```
Same as above but:
  - Known: R1 + R2 actual strokes/categories
  - Simulated: R3 + R4 only
  - Cut: applied from actual (only made-cut players enter sim)
  - Reads: r1_live_model.csv + r2_live_model.csv + model_predictions_r3.csv
```

### After R3

```
Same but:
  - Known: R1 + R2 + R3
  - Simulated: R4 only
  - No skill update needed (just simulate R4 directly)
  - Reads: r1/r2/r3_live_model.csv + model_predictions_r4.csv
```

---

## 5. File Dependencies

### Files round_sim.py Will Read

| File | When | Purpose |
|------|------|---------|
| `model_predictions_r{N}.csv` | Always | Next-round SG predictions (from live_stats_engine) |
| `r1_live_model.csv` | After R1+ | Actual R1 scores + SG categories |
| `r2_live_model.csv` | After R2+ | Actual R2 scores + SG categories |
| `r3_live_model.csv` | After R3 | Actual R3 scores + SG categories |
| `this_week_dists_adjusted.csv` | Always (for tournament sim) | Per-player category distributions |
| `sg_cat_corr_*.csv` | Always (for tournament sim) | Category correlation matrix |
| `pre_sim_summary_{tourney}.csv` | Always | Sample sizes for filtering |
| `pre_course_fit_{tourney}.csv` | Always | Base predictions (my_pred, std) |

### Files round_sim.py Will Write

| File | Contents |
|------|----------|
| `{tourney}/round_{N}_sim_{HHMM}.xlsx` | Full workbook (matchups + score card + outrights) |
| `{tourney}/fair_card_r{N}.csv` | Score card CSV |
| `simulated_probs_live.csv` | Live win probabilities |
| `top_finish_probs_live_{tourney}.csv` | Live top 5/10/20 probabilities |
| `finish_equity_live_{tourney}.csv` | Combined finish position edges |

### New Imports Needed

```python
# Already available in project (used by new_sim.py):
from numpy.linalg import cholesky

# New sim_inputs imports:
from sim_inputs import (
    tourney, STD_DEV, PAR, CUT_LINE, USE_10_SHOT_RULE, name_replacements,
    SIMULATIONS,  # for tournament sim count (can differ from round sim)
    # Coefficients (all already exported from sim_inputs):
    coefficients_r1_high, coefficients_r1_midh, coefficients_r1_midl, coefficients_r1_low,
    coefficients_r2, coefficients_r2_6_30, coefficients_r2_30_up,
    coefficients_r3, coefficients_r3_mid, coefficients_r3_high,
)
```

### No New External Dependencies

Everything uses libraries already in the project: numpy, pandas, scipy, requests, xlsxwriter, gspread.

---

## Implementation Order

I recommend building this in stages so you can validate each piece:

1. **Sheet config + auto-populate script** — can test immediately
2. **`load_known_rounds()` + `load_player_params()`** — data loading, can validate against existing files
3. **`simulate_remaining_rounds()`** — the core engine, test by comparing against new_sim.py outputs
4. **`compute_finish_probabilities()`** — validate win/top-N probs match new_sim.py
5. **Outright market pricing** — fetch odds + edge calculation
6. **Export + email integration** — wire everything together

Ready to start coding when you give the go-ahead.
