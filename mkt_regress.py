"""
mkt_regress.py — Regress sim predictions toward market prices.

Three adjustments layered onto base pred:
  1. mkt_adj:      outright win odds disagreement (raw + magnitude, avg'd)
  2. mu_adj:       matchup edge disagreement (z-scored, inverted)
  3. c_adj_regress: extra regression when course-fit is driving the edge

Output: final_predictions_{tourney}.csv
"""

import os
import pandas as pd
import numpy as np
import requests
from scipy.stats import zscore
from dotenv import load_dotenv
from sim_inputs import tourney, name_replacements, dg_override_players

load_dotenv()

# ---------------------------------------------------------------------------
# Tuning knobs
# ---------------------------------------------------------------------------
OUTRIGHT_EFFECT = 0.12      # max mkt_adj magnitude (SG)
MU_EFFECT = 0.13            # max mu_adj magnitude (SG)
CADJ_MAX_PCT = 0.30         # max c_adj regression at tails (30% of c_adj)
UP_DAMPEN = 0.35            # dampen upward regression (UP is ~40% accurate vs ~67% for DOWN)

# ---------------------------------------------------------------------------
# Load sim outputs
# ---------------------------------------------------------------------------
# init_sim_skill carries DG-overridden predictions from the first-pass new_sim run.
# pred = DG-overridden skill estimate; c_adj and sample from pre_sim_summary.
merged_df = pd.read_csv(f'init_sim_skill_{tourney}.csv')
merged_df['player_name'] = merged_df['player_name'].str.lower().replace(name_replacements)

API_KEY = os.getenv("DATAGOLF_API_KEY")

player_statistics = pd.read_csv(f'finish_equity_{tourney}.csv')
player_statistics['top_1_pct'] = player_statistics['simulated_win_prob'] * 100
player_statistics['player_name'] = player_statistics['player_name'].str.lower().replace(name_replacements)

try:
    mu_df = pd.read_csv(f'betcris_{tourney}.csv')
except FileNotFoundError:
    mu_df = pd.read_csv('betonline_odds_with_my_odds_tu.csv')

# ---------------------------------------------------------------------------
# 1. Outright win odds (mkt_adj)
# ---------------------------------------------------------------------------
url = "https://feeds.datagolf.com/betting-tools/outrights"
params = {
    "tour": "pga",
    "market": "win",
    "odds_format": "percent",
    "key": os.getenv("DATAGOLF_API_KEY"),
}

response = requests.get(url, params=params)
if response.status_code != 200:
    raise RuntimeError(f"DG outrights API failed: {response.status_code} - {response.text}")

data = response.json()
win_odds_df = pd.DataFrame(data['odds'])

bookmakers = ['betcris', 'pinnacle', 'fanduel', 'betonline']
for bk in bookmakers:
    if bk in win_odds_df.columns:
        win_odds_df[bk] = pd.to_numeric(win_odds_df[bk], errors='coerce')
    else:
        win_odds_df[bk] = np.nan

sharp_books = ['betcris', 'pinnacle', 'betonline']
win_odds_df['pin_cris'] = win_odds_df.apply(
    lambda row: row[sharp_books].mean(skipna=True)
    if row[sharp_books].notna().any() else row['fanduel'],
    axis=1,
)
win_odds_df['win_odds'] = win_odds_df['pin_cris'].fillna(win_odds_df['fanduel'])
win_odds_df['player_name'] = win_odds_df['player_name'].str.lower().replace(name_replacements)
win_odds_df.to_csv('win_odds.csv', index=False)
print(f"[mkt] Fetched outright win odds for {len(win_odds_df)} players")

# Merge win odds + sim win probs
merged_df = merged_df.merge(
    win_odds_df[['player_name', 'win_odds']], on='player_name', how='left'
)
merged_df = merged_df.merge(
    player_statistics[['player_name', 'top_1_pct']], on='player_name', how='left'
)
merged_df = merged_df.drop_duplicates(subset='player_name')

# Normalize win odds to sum to 100
merged_df['win_odds'] *= 100
merged_df['win_odds'] = (merged_df['win_odds'] / merged_df['win_odds'].sum()) * 100

# Raw diff and magnitude-scaled diff
merged_df['win_diff_raw'] = merged_df['win_odds'] - merged_df['top_1_pct']
merged_df['win_diff_mag'] = merged_df['win_diff_raw'] / merged_df['win_odds']
merged_df.loc[
    (merged_df['win_odds'] < 0.2) & (merged_df['top_1_pct'] < 0.2), 'win_diff_mag'
] = 0

merged_df['raw_adj'] = (merged_df['win_diff_raw'] / 2) * OUTRIGHT_EFFECT
merged_df['raw_adj'] = merged_df['raw_adj'].clip(-OUTRIGHT_EFFECT, OUTRIGHT_EFFECT)

merged_df['mag_adj'] = (merged_df['win_diff_mag'] / 1.5) * OUTRIGHT_EFFECT
merged_df['mag_adj'] = merged_df['mag_adj'].clip(-OUTRIGHT_EFFECT, OUTRIGHT_EFFECT)

# Zero-center both components
merged_df['raw_adj'] -= merged_df['raw_adj'].mean()
merged_df['mag_adj'] -= merged_df['mag_adj'].mean()

merged_df['mkt_adj'] = (merged_df['raw_adj'] + merged_df['mag_adj']) / 2

# ---------------------------------------------------------------------------
# 2. Matchup edge disagreement (mu_adj)
# ---------------------------------------------------------------------------
edges_p1 = mu_df[['Player 1', 'edge_p1']].rename(columns={'Player 1': 'player_name', 'edge_p1': 'edge'})
edges_p2 = mu_df[['Player 2', 'edge_p2']].rename(columns={'Player 2': 'player_name', 'edge_p2': 'edge'})

all_edges = pd.concat([edges_p1, edges_p2], ignore_index=True)
all_edges['player_name'] = all_edges['player_name'].str.lower().str.strip()
avg_edge = all_edges.groupby('player_name')['edge'].mean().reset_index()

avg_edge['mu_adj'] = -zscore(avg_edge['edge'])  # negative: big edge = market disagrees
avg_edge['mu_adj'] = avg_edge['mu_adj'].clip(-3, 3)
avg_edge['mu_adj'] = (avg_edge['mu_adj'] / 3) * MU_EFFECT

merged_df = merged_df.merge(avg_edge[['player_name', 'mu_adj']], on='player_name', how='left')
merged_df['mu_adj'] = merged_df['mu_adj'].fillna(0)

# Boost mu_adj for weaker players (pred < 1) — market info is more valuable
merged_df['mu_adj'] = np.where(
    merged_df['pred'] < 1,
    merged_df['mu_adj'] * 1.5,
    merged_df['mu_adj'],
)

# ---------------------------------------------------------------------------
# 3. Course-fit regression (c_adj_regress)
#
# Idea: when our edge vs market is largely driven by course adjustment,
# regress harder. Scale = up to CADJ_MAX_PCT of c_adj at the tails,
# declining smoothly as edge shrinks.
#
#   edge_rel  = (our_win% - mkt_win%) / mkt_win%   (positive = we're higher)
#   edge_norm = clip(edge_rel, -1, 1)               (saturates at ±100% relative edge)
#
#   c_adj_regress = -CADJ_MAX_PCT * edge_norm * c_adj
#
# When c_adj and edge point the same way:  regress toward market (penalize)
# When they oppose:                        slight anti-regression (reward)
# ---------------------------------------------------------------------------
merged_df['c_adj'] = merged_df['c_adj'].fillna(0)

# Our edge in relative terms (positive = sim is higher than market)
our_edge = merged_df['top_1_pct'] - merged_df['win_odds']
mkt_base = merged_df['win_odds'].clip(lower=0.1)  # avoid div-by-zero on longshots
edge_rel = our_edge / mkt_base

# Saturate at ±100% relative edge
edge_norm = edge_rel.clip(-1, 1)

# Regression: up to 30% of c_adj at the tails, linear decline toward center
merged_df['c_adj_regress'] = -CADJ_MAX_PCT * edge_norm * merged_df['c_adj']
merged_df['c_adj_regress'] = merged_df['c_adj_regress'].fillna(0)

# ---------------------------------------------------------------------------
# Final prediction — asymmetric regression
#
# DOWN adjustments (~67% accurate) get full weight.
# UP adjustments (~40% accurate) are dampened to UP_DAMPEN strength.
# ---------------------------------------------------------------------------
merged_df['mkt_adj'] = merged_df['mkt_adj'].fillna(0)

# Dampen each component's positive (upward) portion
for col in ['mkt_adj', 'mu_adj', 'c_adj_regress']:
    merged_df[col] = np.where(merged_df[col] > 0, merged_df[col] * UP_DAMPEN, merged_df[col])

merged_df['tot_rgrs'] = merged_df['mkt_adj'] + merged_df['mu_adj'] + merged_df['c_adj_regress']
merged_df['my_pred_final'] = merged_df['pred'] + merged_df['tot_rgrs']

# Filter to active player pool only
merged_df['player_name'] = merged_df['player_name'].replace(name_replacements)
pool = pd.read_csv(f'init_sim_skill_{tourney}.csv')
pool['player_name'] = pool['player_name'].str.lower().replace(name_replacements)
merged_df = merged_df[merged_df['player_name'].isin(pool['player_name'])]

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print(f"  MARKET REGRESSION SUMMARY — {tourney.upper()}")
print(f"{'='*70}")
print(f"  Players: {len(merged_df)}")
print(f"  mkt_adj  range: [{merged_df['mkt_adj'].min():.3f}, {merged_df['mkt_adj'].max():.3f}]")
print(f"  mu_adj   range: [{merged_df['mu_adj'].min():.3f}, {merged_df['mu_adj'].max():.3f}]")
print(f"  c_adj_rg range: [{merged_df['c_adj_regress'].min():.3f}, {merged_df['c_adj_regress'].max():.3f}]")
print(f"  tot_rgrs range: [{merged_df['tot_rgrs'].min():.3f}, {merged_df['tot_rgrs'].max():.3f}]")
print(f"  tot_rgrs mean:  {merged_df['tot_rgrs'].mean():.4f}")

# Show biggest movers
print(f"\n  Top 10 regressed DOWN (market thinks we're too high):")
down = merged_df.nsmallest(10, 'tot_rgrs')[['player_name', 'pred', 'c_adj', 'mkt_adj', 'mu_adj', 'c_adj_regress', 'tot_rgrs', 'my_pred_final']]
for _, r in down.iterrows():
    print(f"    {r['player_name']:25s}  pred={r['pred']:+.2f}  c_adj={r['c_adj']:+.2f}  "
          f"mkt={r['mkt_adj']:+.3f}  mu={r['mu_adj']:+.3f}  cfit={r['c_adj_regress']:+.3f}  "
          f"tot={r['tot_rgrs']:+.3f}  final={r['my_pred_final']:+.2f}")

print(f"\n  Top 10 regressed UP (market thinks we're too low):")
up = merged_df.nlargest(10, 'tot_rgrs')[['player_name', 'pred', 'c_adj', 'mkt_adj', 'mu_adj', 'c_adj_regress', 'tot_rgrs', 'my_pred_final']]
for _, r in up.iterrows():
    print(f"    {r['player_name']:25s}  pred={r['pred']:+.2f}  c_adj={r['c_adj']:+.2f}  "
          f"mkt={r['mkt_adj']:+.3f}  mu={r['mu_adj']:+.3f}  cfit={r['c_adj_regress']:+.3f}  "
          f"tot={r['tot_rgrs']:+.3f}  final={r['my_pred_final']:+.2f}")

# ---------------------------------------------------------------------------
# Save detailed output
# ---------------------------------------------------------------------------
detail_path = f'final_predictions_{tourney}_details.csv'
merged_df.to_csv(detail_path, index=False)
print(f"\n[ok] Saved {detail_path}")

# ---------------------------------------------------------------------------
# Save slim output for tournament sim (pred, std_dev, tee times)
# ---------------------------------------------------------------------------
pcf = pd.read_csv(f'pre_course_fit_{tourney}.csv', usecols=['player_name', 'r1_teetime', 'r2_teetime', 'std_dev', 'sample'])
pcf['player_name'] = pcf['player_name'].astype(str).str.lower().str.strip().replace(name_replacements)

slim = merged_df[['player_name', 'my_pred_final']].rename(columns={'my_pred_final': 'my_pred'})
slim = slim.merge(pcf, on='player_name', how='left')

slim_path = f'final_predictions_{tourney}.csv'
slim.to_csv(slim_path, index=False)
print(f"[ok] Saved {slim_path} ({len(slim)} players, cols: {list(slim.columns)})")

# Copy to OneDrive for etr-golf-sims consumption
_onedrive_path = os.path.join(os.path.expanduser("~"), "OneDrive", f"final_predictions_{tourney}.csv")
slim.to_csv(_onedrive_path, index=False)
print(f"[ok] Copied to {_onedrive_path}")
