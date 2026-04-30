# Correlated Kelly Staking Module

## What This Document Is

A rigorous specification of the math and logic needed to build a correlated Kelly staking system for golf betting. This document contains zero code. It describes what needs to be computed, why, and what inputs are required from the existing simulation infrastructure. The implementer should read this end-to-end, inspect the existing sim output structure, and build the integration accordingly.

---

## The Problem

We have a tournament simulation that prices every outcome — win, top 5, top 10, top 20, matchups — and identifies bettable edges across exchanges and sportsbooks. We currently size each bet independently using the standard Kelly criterion. This is wrong when multiple bets share the same underlying golfer.

**Example:** We identify four edges on Alex Noren in the same event — top 5 at +900, top 10 at +405, top 20 at +181, and a head-to-head matchup against Thorbjornsen at -111. Independent Kelly says wager roughly 20.6% of bankroll across these four bets. At 60% Kelly on a $30,000 bankroll, that is $3,711 on a single golfer.

The problem is that these bets are not independent. Top 5 is a strict subset of top 10, which is a strict subset of top 20. If Noren finishes 3rd, all three placement bets win. If he finishes 25th, all three lose. The matchup is also correlated — Noren is far more likely to beat Thorbjornsen when he finishes 5th than when he finishes 50th. Independent Kelly ignores this structural correlation entirely, treating each bet as if the others do not exist, and dramatically overestimates safe sizing because it counts the same golfer-performance exposure multiple times.

---

## The Solution: Joint-State Optimization

Instead of optimizing each bet in isolation, we optimize all bets on a golfer simultaneously by reasoning about the **joint distribution of outcomes** — every possible combination of placement finish and matchup result, with its probability and its payoff across all bets at once.

### Step 1: Define the Finishing Buckets

Every golfer's tournament result falls into exactly one of four mutually exclusive finishing buckets:

- **Bucket A:** Finish 1st through 5th
- **Bucket B:** Finish 6th through 10th
- **Bucket C:** Finish 11th through 20th
- **Bucket D:** Finish 21st or worse (includes MC, WD, DQ)

Each bucket deterministically resolves every placement bet:

| Bucket | Top 5 | Top 10 | Top 20 |
|--------|-------|--------|--------|
| A (1-5) | Win | Win | Win |
| B (6-10) | Lose | Win | Win |
| C (11-20) | Lose | Lose | Win |
| D (21+) | Lose | Lose | Lose |

This is not an approximation. It is the structural truth of how these bets resolve. A golfer who finishes 7th always wins top 10 and top 20 and always loses top 5. There is no randomness conditional on the finishing bucket.

### Step 2: Cross with Matchup Outcomes

A head-to-head matchup (Noren > Thorbjornsen) is a separate binary outcome — win or loss — but it is **conditionally correlated** with the finishing bucket. Noren is much more likely to beat Thorbjornsen when he finishes in the top 5 than when he finishes 30th, because a top-5 finish implies Noren played well, which usually means he outperformed most of the field including Thorbjornsen.

The full joint state space is the cross-product of finishing buckets and matchup outcomes:

| State | Finish Bucket | Matchup | All placement outcomes | Matchup outcome |
|-------|--------------|---------|----------------------|-----------------|
| S1 | A (1-5) | Win | T5 ✓, T10 ✓, T20 ✓ | ✓ |
| S2 | A (1-5) | Loss | T5 ✓, T10 ✓, T20 ✓ | ✗ |
| S3 | B (6-10) | Win | T5 ✗, T10 ✓, T20 ✓ | ✓ |
| S4 | B (6-10) | Loss | T5 ✗, T10 ✓, T20 ✓ | ✗ |
| S5 | C (11-20) | Win | T5 ✗, T10 ✗, T20 ✓ | ✓ |
| S6 | C (11-20) | Loss | T5 ✗, T10 ✗, T20 ✓ | ✗ |
| S7 | D (21+) | Win | T5 ✗, T10 ✗, T20 ✗ | ✓ |
| S8 | D (21+) | Loss | T5 ✗, T10 ✗, T20 ✗ | ✗ |

These eight states are **mutually exclusive and exhaustive**. Every possible tournament outcome for this golfer maps to exactly one state. Each state fully determines the payoff of every bet in the group.

If the golfer has no matchup bets, the state space collapses to just the four finishing buckets. If the golfer has multiple matchups (say, Noren > Thorbjornsen AND Noren > Player X), the state space becomes 4 buckets × 2^n matchup combinations. Two matchups = 16 states. Three matchups = 32 states. The framework scales without modification.

### Step 3: Extract State Probabilities from the Sim

This is the critical connection to the existing infrastructure. The tournament sim already produces finishing positions for every golfer in every iteration. For each sim iteration, we know Noren's exact finish AND Thorbjornsen's exact finish. We do not need to estimate or model the correlation. We extract it directly.

**What must be derived from the sim:**

For each golfer with correlated bets, tabulate how many sim iterations fall into each joint state. Divide by total iterations. That is the probability vector.

Concretely: iterate through every sim run. For each run, determine which finishing bucket Noren falls in (A, B, C, or D) and whether Noren's finish position is strictly better than Thorbjornsen's (matchup win) or not (matchup loss). Increment the counter for the corresponding joint state. After all iterations, divide each counter by total iterations to get the eight-state probability vector.

**Handling ties in matchups:** If Noren and Thorbjornsen finish in the same position in a sim iteration, this is typically a push (money returned). Two options: (a) treat ties as half-win, half-loss by splitting them evenly between the win and loss states for that bucket, or (b) add explicit push states where the matchup payoff is zero. Option (a) is simpler and sufficient unless pushes are very frequent (>5% of iterations).

**Sim sample size requirement:** Rare joint states (e.g., Noren finishes top 5 AND loses the matchup) may have very few sim observations. If any non-zero state has fewer than ~50 observations, the probability estimate is unstable. For golfers with 3+ correlated bets, run a minimum of 200,000 sim iterations.

### Step 4: Build the Payoff Matrix

For each state, define the net return per unit wagered on each bet.

If a bet **wins** in that state: the payoff is the decimal profit from the market odds. For example, +900 means the payoff is +9.0 per unit wagered (you risked 1, you profit 9). For -111, the payoff is +0.9009 per unit wagered.

If a bet **loses** in that state: the payoff is -1.0 per unit wagered (you lose your stake).

If a bet **pushes** (ties, if using explicit push states): the payoff is 0.0.

This produces a matrix with one row per state and one column per bet. Every entry is either the bet's decimal profit, -1, or 0. Combined with the probability vector from Step 3, this fully specifies the optimization problem.

### Step 5: Solve the Multivariate Kelly Optimization

Let **f** = (f₁, f₂, ..., fₙ) be the vector of fractions of bankroll to wager on each of the n bets. Let **P(s)** be the probability of state s. Let **rᵢ(s)** be the net return per unit on bet i in state s (from the payoff matrix).

The multivariate Kelly criterion maximizes the expected logarithmic growth rate of the bankroll:

**maximize  Σₛ P(s) · log(1 + Σᵢ fᵢ · rᵢ(s))**

subject to:
- **fᵢ ≥ 0** for all i (no negative bets)
- **Σᵢ fᵢ ≤ golfer_cap** (maximum total exposure to one golfer)

**Why log-growth, not expected value:** This is a repeated game with multiplicative dynamics. The bankroll compounds over hundreds of bets per season. In this regime, maximizing the time-average growth rate (which is what log-growth does) is the mathematically correct objective. Maximizing raw expected value leads to overbetting and eventual ruin. This is the foundational insight of the Kelly criterion.

**Why there is no closed-form solution:** The single-bet Kelly formula f* = (bp - q) / b does not generalize to multiple correlated bets. With more than two bets, the system of partial derivatives of the log-growth function produces a nonlinear system that must be solved numerically.

**Why this is computationally easy:** The objective function is concave (log of a linear function is concave; a positively-weighted sum of concave functions is concave). The constraints are linear. Therefore there is exactly one global maximum, and any standard convex optimizer finds it in milliseconds. This is not an approximation — it is the exact solution.

**Why the correlation adjustment reduces sizing:** Consider the worst-case state — Noren finishes 21+ and loses the matchup (state S8). ALL bets lose. The argument of the log becomes 1 - f₁ - f₂ - f₃ - f₄. If total stakes are large, this term approaches zero, and log(0) goes to negative infinity, which dominates the entire objective. The optimizer cannot push total exposure high without making the growth rate in the worst-case state catastrophically negative. This is the mathematical mechanism by which correlation automatically constrains sizing. Independent Kelly, by contrast, never sees the worst-case joint state because it evaluates each bet in isolation.

### Step 6: Apply Fractional Kelly

After the optimizer produces the full Kelly fractions, multiply every fraction by the fractional Kelly multiplier (currently 0.60). This accounts for parameter uncertainty — the edge estimates from the sim are not exact. Even small errors in probability estimation can make full Kelly too aggressive. Fractional Kelly sacrifices a small amount of theoretical growth rate in exchange for substantial reduction in variance and drawdown.

This multiplier is applied AFTER the joint optimization, not before. The optimizer solves for full Kelly; then we scale down uniformly.

### Step 7: Convert to Dollars and Apply Portfolio Constraints

Multiply each fractional Kelly fraction by the bankroll to get dollar wager amounts. Then apply the following constraints in order:

**Portfolio-level cap:** Sum all dollar amounts across all golfer groups for the entire slate. If total deployed exceeds the portfolio cap (currently 35% of bankroll pre-fractional, which is ~21% at 60% Kelly, which is ~$6,300 on a $30k bankroll), scale all positions proportionally so total deployed equals the cap.

**Minimum bet filter:** Zero out any bet below $100. These are not worth the execution cost.

---

## What the Correlated Solution Actually Does (Noren Example)

Using the four Noren bets from the problem statement with reasonable conditional matchup probabilities derived from typical sim output:

| Bet | Independent Kelly | Correlated Kelly | Reduction |
|-----|-------------------|------------------|-----------|
| Top 5 | 2.92% | ~1.8% | ~38% |
| Top 10 | 2.89% | ~1.6% | ~45% |
| Top 20 | 2.00% | ~1.1% | ~45% |
| Matchup | 12.81% | ~10.5% | ~18% |
| **Total** | **20.62%** | **~15.0%** | **~27%** |

The placement bets absorb the largest proportional reduction because they are the most correlated with each other — nested subsets of the same outcome. The matchup retains more of its independent sizing because it has meaningful independent variance (Noren can beat Thorbjornsen from many finishing positions, and can lose to him even from good finishes).

At 60% Kelly on $30,000: total Noren exposure drops from ~$3,711 (naive) to ~$2,700 (correlated). The 27% reduction is not a tuning parameter — it falls directly out of the joint probability structure.

---

## How This Applies Across the Full Slate

### Grouping logic

Every bet in the slate gets tagged with a **primary golfer**. Placement bets (top 5/10/20, win outright) are obviously tagged to their golfer. Matchups get tagged to the golfer whose correlation with the placement bets matters. If we have edges on both sides of a matchup (unlikely but possible), the matchup appears in both golfer groups.

All bets in a golfer group are jointly optimized together. Each golfer group is solved independently.

### Cross-golfer correlation

In a 156-player field, one golfer finishing top 10 has negligible impact on another golfer's placement probability. Cross-golfer placement correlation is second-order and should be ignored. The field is too large for it to matter.

**Exception:** If we have multiple matchups involving the same golfer (Noren > Thorbjornsen AND Noren > Player X), those matchups must be in the same optimization group because they share structural correlation through Noren's performance.

### Portfolio-level risk control

After all golfer groups are sized independently, the portfolio cap ensures total deployment stays within bounds. This is a simple proportional scaling — if total exceeds the cap, every bet gets scaled down by the same factor. This is not mathematically optimal (ideally you would prioritize the highest edge-per-dollar bets), but it is simple, robust, and the difference is negligible in practice.

---

## Risk Management Parameters

| Parameter | Current Value | What It Controls |
|-----------|--------------|-----------------|
| Kelly fraction | 0.60 | Buffer against edge estimation error. Applied after optimization. |
| Per-golfer cap | 15% (pre-fractional) | Maximum total Kelly fraction on all bets for one golfer. Becomes ~9% at 60% Kelly. Passed as a constraint to the optimizer. |
| Portfolio cap | 35% (pre-fractional) | Maximum total deployment across all golfers per slate. Becomes ~21% at 60% Kelly (~$6,300 on $30k). Applied as proportional scaling after all groups are solved. |
| Minimum bet | $100 | Bets below this are zeroed out. Not worth execution cost. |

---

## Summary of What the Implementation Must Do

1. **Group** all identified edges by primary golfer.

2. **For each golfer group**, query the sim output to tabulate the joint probability of every combination of finishing bucket × matchup outcomes. This requires iterating through sim results and counting co-occurrences. The output is a probability vector over the joint state space.

3. **Construct the payoff matrix** from market odds. Each cell is either the bet's decimal profit (win), -1 (loss), or 0 (push). One row per state, one column per bet.

4. **Solve the concave optimization problem**: maximize expected log-growth over all states, subject to non-negative fractions and the per-golfer cap. Use any standard constrained optimizer. The problem is well-conditioned and solves in milliseconds.

5. **Multiply** all resulting fractions by the Kelly fraction (0.60).

6. **Convert** to dollar amounts. Apply portfolio cap scaling if total deployment exceeds the portfolio limit. Zero out bets below the minimum.

7. **Log diagnostics**: for every bet, output the independent Kelly fraction, the correlated Kelly fraction, the percentage reduction, and the total golfer exposure. This is essential for building intuition about when the correlation adjustment matters most.

---

## Diagnostic Output

Every time the staking module runs, it should produce a report showing, for each golfer group:

- Each bet's edge, market odds, and fair probability (already known from the scanner)
- The independent Kelly fraction (what naive Kelly would say)
- The correlated Kelly fraction (what the joint optimizer says)
- The reduction percentage (how much correlation cost)
- The final dollar amount after fractional Kelly and portfolio scaling
- The total exposure to that golfer
- Whether any bet was zeroed and why (below minimum, optimizer chose zero, portfolio scaling)

And at the portfolio level: total dollars deployed, percentage of bankroll, and whether the portfolio cap was binding.
