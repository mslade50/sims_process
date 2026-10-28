# Published-Principles Review: Broadie and Peabody

Date: 2026-08-22  
Scope: adversarial review of the frozen base-skill/market-backtest design  
Important: this uses published work and public interviews. It does not claim that
Mark Broadie or Rufus Peabody personally reviewed or endorsed this project.

## Mark Broadie lens

Broadie's strokes-gained framework values a shot from its starting and ending state,
and the shot values add back to the golfer's scoring result. That supports using the
raw IMG accounting stream as the clean independent shot source. It argues against
making the materialized, hand-shaped IMG predictive transform the headline signal,
especially because that transform was tuned using seasons inside the proposed test.

Broadie's PGA TOUR analysis also finds that long-game performance explains most of
the separation between golfers: roughly two-thirds for the elite examples in the
paper, while putting and short-game rankings are substantially less aligned with total
ability. The model should therefore be allowed to learn different persistence and
shrinkage by category. Equal category weights, or a category model whose putting
history can overwhelm approach/OTT because it is noisy, would be difficult to defend.

Source: [Mark Broadie, *Assessing Golfer Performance on the PGA TOUR*](https://business.columbia.edu/sites/default/files-efs/pubfiles/4996/assessing_golfer_performance.full.pdf).

### Changes required by this lens

- Keep raw IMG accounting SG primary; keep validated-v2 as a retrospective ablation.
- Express adjusted categories as deviations orthogonal to adjusted total so the fixed
  field-strength allocation is not counted twice.
- Add a prespecified long-game-only arm: adjusted total + OTT/APP category deviations
  + IMG OTT/APP accounting histories.
- Permit separate ridge penalties/persistence for OTT, APP, ARG, and PUTT; force the
  short-game/putting coefficients toward zero more strongly unless discovery markets
  clearly reject that shrinkage.
- Test additivity and reconciliation at every player-round. Flag disagreement between
  DataGolf accounting SG and IMG accounting SG instead of averaging it away.
- Keep detailed shot transforms primitive and auditable: penalties, hole-outs, and
  tail winsorization learned only inside earlier folds.

## Rufus Peabody lens

Peabody has described successful modeling as contextualizing performance, accounting
for opponent/field strength and weather, and deciding carefully how much to weight
recent versus older results; he specifically cautions that people commonly overweight
recent form. He also emphasizes that the market often incorporates injuries, player-
specific news, and qualitative information better than a standalone model. A model
does not need to dominate the market everywhere; it needs incremental information the
market has not already priced.

Source: [Rufus Peabody interview, RotoWire](https://www.rotowire.com/football/article/beating-the-book-qa-with-rufus-peabody-40172).

Unabated's published process treats vig-free sharp prices, line comparison, and closing
line value as core evaluation tools. Its description of CLV explicitly distinguishes a
raw sportsbook close from a vig-free fair closing price. That is directly relevant here:
raw execution odds settle P&L, while the no-vig sharp close is the calibration and CLV
benchmark.

Sources: [Unabated on CLV and no-vig pricing](https://unabated.com/post/ten-ways-to-win-at-sports-betting-with-unabated),
[Unabated CLV methodology](https://unabated.com/tools/calculators/closing-line-value), and
[Unabated no-vig methodology](https://www.unabated.com/tools/calculators/no-vig-odds).

Public descriptions of Peabody's golf process also emphasize high-volume simulation,
matchups/top-10/top-20 markets, and line shopping rather than relying on winner bets
alone. That supports using H2H markets as the main identification test and tournament
finish markets as confirmation.

Sources: [Golf Digest profile of Peabody's golf modeling](https://www.golfdigest.com/story/golf-betting-high-stakes) and
[Peabody golf interview topics](https://podcasts.apple.com/us/podcast/professional-golf-bettor-rufus-peabody-golf-betting/id1047779421?i=1000569276341).

### Changes required by this lens

- Do not let the search settle on a short half-life merely because it improves next-
  event RMSE. Include long calendar and round horizons and select with out-of-sample
  market proper scores.
- Rebuild every historical state from weather-neutralized observations; adding weather
  only as a final covariate leaves contaminated recent-form features in place.
- Add reliability variables and explicit stale/missing/new-player flags. Report a
  conservative arm that suppresses nominal over-10% edges when either golfer has thin
  history, missing shot coverage, a withdrawal in the immediately prior event, or an
  unresolved identity join.
- Compare the pure model to a vig-free sharp-close consensus on every quoted pair.
  Report whether the model adds information beyond the close; do not declare victory
  merely because its raw outcome RMSE improved.
- Keep closing odds out of the model used to wager at the close. A market blend may use
  opening prices and must then be evaluated against later closes/outcomes.
- Line-shop BetOnline, Betcris, and Pinnacle, but retain fixed-book results and flag the
  simultaneity/limit assumption behind best-of-three closing prices.
- Lead with matchup performance and include top-10/top-20 confirmation; do not let a
  few winner longshots dominate the model verdict.
- Use event-level paired uncertainty, concentration, and drawdown. One lucky winner or
  one mass-injury week cannot constitute model validation or rejection.

## Pre-unseal revised candidate set

The review leaves the frozen question unchanged but revises the prespecified candidate
set before the sealed 2023-2026 P&L is opened:

1. total multi-horizon ridge;
2. total + orthogonal categories with category-specific shrinkage;
3. total + raw IMG accounting;
4. total + orthogonal categories + raw IMG accounting;
5. long-game-only Broadie arm;
6. each of the above using weather-neutralized history;
7. robust/Huber versions for tail resistance;
8. a regularized pairwise-logistic score using differences of the same strict-prior
   features, trained only on discovery matchup outcomes;
9. conservative reliability-gated versions;
10. materialized IMG validated-v2 only as retrospective sensitivity.

For each family, short half-lives compete against 40/80/160-round and one-/two-year
horizons. Complexity cannot be justified by a small RMSE gain: the candidate must show
market calibration, edge monotonicity, and paired money/CLV improvement on discovery
data before it is allowed into the sealed test.

## Remaining pitfalls neither framework makes disappear

- Exact historical production is unavailable, so the legacy arm remains a current-code
  retrospective reconstruction.
- The sharp close contains information the model cannot observe, particularly injury
  and status news. This is a feature of the benchmark, not evidence of data leakage,
  provided the close never becomes a same-snapshot model input.
- Market outcomes are highly correlated inside an event, and available matchup menus
  are selected by bookmakers rather than random samples of golfer pairs.
- Raw IMG and DataGolf SG may share underlying ShotLink observations even though their
  accounting pipelines differ. Their errors are not guaranteed independent.
- No archived three-year forecast history exists. Realized upcoming weather can explain
  weeks but cannot be credited as tradable foresight.

