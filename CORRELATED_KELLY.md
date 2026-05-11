# Correlated Kelly Staking

## What It Does

Sizes all bets on the same golfer jointly, accounting for the fact that finish positions (win, top 5, top 10, top 20) and tournament matchups on the same player share the same underlying finishing outcome. Independent Kelly treats each bet in isolation and overestimates safe sizing when bets are structurally correlated.

The optimizer runs after Email 1 is built. It consumes only the bets that passed email filters (sharp books, edge/pred/sample thresholds) plus qualifying exchange bets, and produces a separate Email 3 with per-bet sizing.

## Why It Matters

If you bet Noren top 5 (+900) and Noren matchup vs Taylor (-115), those bets are not independent. When Noren finishes top 5, he almost certainly also beats Taylor. Both bets win in the same sims and lose in the same sims. Independent Kelly doesn't see this — it sizes each bet as if the other doesn't exist, leading to overexposure on a single golfer's outcome.

Nested finish positions have the same problem: top 5 is a strict subset of top 10, which is a strict subset of top 20. Betting all three at independent Kelly sizes means triple-counting the overlapping probability mass.

## How It Works

### Step 1: Collect qualifying bets

After `build_tournament_email_html()` runs, we re-apply the same filters to build two bet lists:

**Finish positions** (from `combined_finish_df`):
- Sharp books only (pinnacle, betonline, betcris)
- Best price per player/market (deduped by highest edge)
- Edge >= 0.4%
- Plus exchange finish bets from `_exchange_bets` (Kalshi/NoVig outrights that beat sportsbook edge)

**Tournament matchups** (from `sharp_df`):
- Sharp books, deduped by matchup key
- `pred_on > 0.75`, `sample_on >= 8`, `dist_rounds >= 20`
- `edge_on > 3`
- Plus exchange-replaced matchups from `_exchange_mu_replacements`

Both sportsbook and exchange versions of the same player/market enter the optimizer as separate bets — the optimizer sizes them jointly since they pay out in the same states.

### Step 2: Group by golfer

Each bet is tagged with its primary golfer (`player_name` for finish bets, `bet_on` for matchups). Bets are grouped by golfer. Single-bet groups skip optimization and use independent Kelly directly.

### Step 3: Compute per-sim bet outcomes from `final_scores`

For each bet in a multi-bet group, we determine win/loss in every sim using the actual `final_scores` array (132 players x 100,000 sims of integer 72-hole stroke totals):

**Finish bets**: For each sim, compute the golfer's rank (number of players with strictly lower score + 1) and tie count. Apply dead-heat adjustment matching the sim's own `dead_heat_factor()` logic:

```
rank = count(players scoring < golfer) + 1
tie_count = count(players scoring == golfer)
end_rank = rank + tie_count - 1
overlap = max(min(end_rank, threshold) - max(rank, 1) + 1, 0)
outcome = overlap / tie_count
```

For a top 10 bet (threshold=10): if golfer is rank 8 with 4-way tie (ranks 8-11), overlap with positions 1-10 is 3, so outcome = 3/4 = 0.75.

**Matchup bets**: Direct head-to-head comparison of golfer's score vs opponent's score per sim. Ties get 0.5 credit (split).

This gives exact co-occurrence — we know in precisely which sims both the top 10 and the matchup win, which sims one wins and the other loses, etc.

### Step 4: Build joint state probability vector

With N bets in a golfer group, there are 2^N possible joint outcomes (each bet wins or loses). We encode each sim as a state index (binary: bit i = 1 if bet i wins) and count occurrences across all 100K sims.

For clean outcomes (all 0 or 1), this is vectorized via `np.bincount`. For fractional outcomes (dead heats, matchup ties), we split the sim's weight across the relevant states proportionally.

The result is a probability vector over all 2^N states that captures the exact joint distribution — including the correlation structure. For example, for Fowler's 3-bet group (winner, top_10, matchup), the 8 states might look like:

| State | Winner | Top 10 | Matchup | Probability |
|-------|--------|--------|---------|-------------|
| 0 | lose | lose | lose | 39.2% |
| 1 | lose | lose | win | 36.4% |
| 2 | lose | win | lose | 7.6% |
| 3 | lose | win | win | 12.7% |
| 4 | win | lose | lose | ~0% |
| 5 | win | lose | win | ~0% |
| 6 | win | win | lose | 0.5% |
| 7 | win | win | win | 3.6% |

Note: states 4 and 5 (win tournament but not top 10) are near-impossible. State 2 (top 10 but lose matchup) is the key hedging state — ~7.6% of sims where the top 10 bet provides value that the matchup doesn't.

### Step 5: Build payoff matrix

Each state's payoff per bet is:
- **Win**: `decimal_odds - 1` (net profit on a $1 bet)
- **Lose**: `-1` (lose the stake)

This gives an (n_states x n_bets) matrix.

### Step 6: Solve Kelly

Maximize expected log-wealth growth:

```
max  Σ P(s) * log(1 + Σ f_i * r_i(s))
s.t. f_i >= 0 for all i
     Σ f_i <= golfer_cap  (15% pre-fractional)
```

Where `f_i` is the fraction of bankroll on bet i, `r_i(s)` is the payoff of bet i in state s, and `P(s)` is the probability of state s.

This is a concave optimization (log of linear is concave, weighted sum of concave is concave) with linear constraints — unique global optimum, solved via `scipy.optimize.minimize` (SLSQP) in ~1ms per golfer group.

The solver naturally:
- Reduces bets that overlap heavily (top 10 + matchup on same golfer)
- Preserves bets that cover unique states (top 10 when matchup loses)
- Zeros bets whose marginal contribution is negligible given other bets in the group

### Step 7: Apply fractional Kelly and report

Multiply all fractions by `kelly_fraction` (0.60) for safety. Convert to dollar amounts at the configured bankroll ($30,000). No portfolio cap or minimum bet filter — the output shows the raw optimal allocation per golfer group.

## What the output tells you

**Indep f***: Standard Kelly fraction for the bet in isolation: `f* = (bp - q) / b` where `b = decimal_odds - 1`, `p = sim_prob`, `q = 1 - p`. Already multiplied by kelly_fraction.

**Corr f***: Optimal Kelly fraction accounting for all other bets on the same golfer. Always <= Indep f* for multi-bet groups (the optimizer can only reduce or maintain, never increase, since it could always choose the independent allocation).

**Reduction**: `1 - corr_f / indep_f`. Measures how much the bet's sizing drops due to correlation with other bets in the group. 0% for single-bet golfers. Typical ranges:
- Matchup + nested finish on same golfer: 10-30% on the dominant bet, 40-100% on the subordinate
- Two nested finish bets (top_5 + top_10): 35-55% each
- Win bet alongside a matchup: usually 90-100% (almost entirely redundant)

**$ Size**: `bankroll * corr_f`. The recommended wager.

## Key behaviors

**Matchups dominate finish bets in mixed groups.** A -115 matchup covers ~57% of sims at decent edge. A +900 top 5 covers ~10% of sims. The matchup's broader coverage means the finish bet adds less marginal information. The optimizer reduces finish bets more aggressively than matchups.

**Winner bets get heavily reduced or zeroed in groups.** If you have a matchup on the same golfer, the winner bet's tiny probability mass (~3-5%) is almost entirely a subset of the matchup's win states. The marginal value is near zero.

**Top 5 vs top 10 on the same golfer:** the optimizer often keeps one and zeros the other. It picks the one with better risk-adjusted marginal contribution. Top 5 (rarer, higher payout) sometimes survives over top 10 because it provides more unique right-tail exposure.

## Files

| File | Role |
|------|------|
| `correlated_kelly.py` | Pure math module: optimizer, console report, HTML email builder |
| `new_sim.py` (~line 3228) | Integration: collects bets, calls optimizer, sends Email 3 |
| `final_scores_{tourney}.npy` | Saved per-sim 72-hole totals (132 x 100K), used for co-occurrence |

## Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `bankroll` | $30,000 | Total capital for correlated sizing |
| `kelly_fraction` | 0.60 | Safety multiplier (60% Kelly) |
| `golfer_cap` | 0.15 | Max 15% of bankroll on any single golfer (pre-fractional) |
| `SKIP_STORAGE` | env var | Set to skip Google Sheets writes on rerun |
