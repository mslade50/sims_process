# Balanced APP: finish-position follow-up

2026-09-09. Read-only season audit; logged signals, not reconciled account fills.

**Random variation remains the leading working explanation for the round-matchup swing. The finish analysis identifies a narrower concern: APP underperformance among selected Balanced APP players, rather than a demonstrated bias affecting the entire archetype.** This is exploratory evidence from the same season, not independent confirmation or a reason to apply an automatic penalty.

## Finish results

Primary sample: pre-event YES top-5/10/20 signals at Pinnacle, BetOnline and Betcris. Keep the first quote per event/player/market/book, then the highest theoretical EV available in the earliest run per event/player/market. Outrights, NO contracts, exchanges and live bets are separate. The 783 selected signals cover 28 events.

| Balanced APP, sharp books | Signals | Recorded profit | ROI | Model-expected profit |
|---|---:|---:|---:|---:|
| All edges | 75 | −3.639u | −22.3% | +3.441u |
| ≥5% Kelly theoretical EV | 52 | −4.269u | −29.0% | +3.395u |
| <5% | 23 | +0.630u | +39.0% | +0.047u |

Risk is recorded Kelly dollars divided by $200 per unit. The group risked 16.316u total, including only 1.617u at <5%. These are small exposures; the large ROI percentages should be read accordingly. Expected profit is the model's claim, not an independently established expectation.

For ≥5%, mean expected payout share was 14.17%, versus 7.69% realized. Payout share accounts for dead heats and is not an ordinary binary hit rate. All edges together: 16.03% expected versus 12.44% realized. The other archetypes returned −5.5% ROI overall at these sharp books.

The negative result is not uniform across venues or thresholds. Across all sportsbooks, the comparable best-first-run Balanced APP sample had 151 signals, +0.134u profit and +0.4% ROI; ≥5% returned −3.2%, while <5% returned +33.0%. These views have different opportunity sets. They do not isolate a causal bookmaker effect.

Nor does this reproduce a midseason finish-bet collapse: the sharp sample lost 7.262u before June 9 and gained 3.623u afterward, although only ten signals were later. Top-10 signals lost 3.259u; top-20 lost 0.943u; top-5 gained 0.563u. Small, correlated slices should not become separate rules.

## The approach-performance check

Count each player-event once and evaluate adjusted APP SG averaged over rounds 1–2. This avoids giving a player extra statistical weight because we logged several markets/books, and avoids weekend survivor bias. The benchmark is the strictly pre-event historical category profile used in the archetype study: an average of the last-50 and last-100 means where history permits. **It is a historical skill benchmark, not an exact recovered tournament-simulation APP forecast.** Finish probability comparisons above use the actual recorded simulation probabilities.

| Player-event population at the 28 sharp-bet events | Observations | APP actual minus historical baseline, SG/round |
|---|---:|---:|
| Balanced APP, selected sharp finish signals | 45 | **−0.268** |
| Balanced APP, outside that selected set | 91 | **−0.005** |
| Other profiles, selected sharp finish signals | 493 | +0.038 |
| Other profiles, outside that selected set | 2,479 | +0.006 |

“Outside” means absent from the selected sharp finish sample; these players may have other-book or round-matchup bets. Balanced APP historical APP expectations were nearly identical in the two groups: +0.738 versus +0.740 SG. Realized APP was +0.470 versus +0.734. The archetype as a whole therefore does not show the same blanket shortfall as the selected subset.

Selected Balanced APP putting was **+0.132 SG above** its historical baseline. The approach shortfall in this finish sample cannot be attributed to bad putting. APP errors were −0.308 SG in ≥5% and −0.138 in <5%, counting unique player-events within each band; a player-event can appear in both bands.

Across all sportsbooks, the selected Balanced APP approach error remained −0.258 SG over 75 player-events. Changing the benchmark to the last 20, 50 or 100 rounds gives sharp-sample errors of −0.246, −0.269 and −0.266 SG. A simple shorter or longer averaging window does not make the descriptive deficit disappear. Recent-20 minus last-100 APP was only −0.020 SG for the sharp selections, versus −0.002 outside them; this does not establish a large overlooked recent-form decline.

Cantlay, Rose and Nicolai Hojgaard are useful replay cases: their approach errors were −0.657, −0.671 and −0.568 SG, across seven, five and three selected player-events. Lowry lost 3.627u across four events, nearly the entire net monetary loss, although removing him still leaves the APP deficit. Hovland and Straka made money. These examples guide case review, not player bans.

## Uncertainty and checks

- The 75 sharp signals represent 45 player-events, 13 players and 18 events. Resampling 20,000 whole-event clusters gives a 95% ROI interval of approximately **−83% to +86%** and a payout-share error interval of **−10.3 to +4.0 percentage points**. Finish profitability/calibration bias is not established.
- The unique player-event APP error interval is approximately **−0.516 to −0.013 SG**. It is an exploratory, unadjusted interval after looking across many categories; recurring players create additional dependence. The within-event difference versus other selected profiles has an interval crossing zero (−0.489 to +0.079 SG). Do not treat the barely negative marginal interval as proof.
- Among the 17 events containing both selected and unselected Balanced APP players, the equal-event approach-error difference is −0.221 SG, with a 95% event-bootstrap interval of −0.555 to +0.171. The apparent selection pattern is a testable lead, not a statistically established selection effect.
- Removing any single player leaves APP error between −0.354 and −0.196 SG; removing any event leaves it between −0.338 and −0.199. No one player or week explains the approach shortfall.
- Twenty of the 45 Balanced APP finish player-events also had a backed player in the sharp round audit. The two market studies share golfers and outcomes and are not independent replications.
- Classifications use only rounds before each event and the same reference population as the earlier round study. All 778 overlapping original profiles reproduce exactly. Using the labels stamped on the finish rows instead yields 54 signals and −0.660u: label/version sensitivity matters.
- All 5,545 eligible first pre-event rows and 2,186 first live rows independently reconcile final finish and monetary settlement to the historical database and grader conventions. There were 138 exchange rows with missing stakes; the grader's documented 1u fallback reconciles them. None affects the primary sharp sample. No shared grades were changed.
- First-quote selection precedes filtering grading status: 50 pre-event first rows marked duplicate and 50 marked no_data are excluded; 22 duplicate and two no_data live first rows are excluded. These are excluded signals, not assumed losses. Snapshot hashes and row counts are saved.
- Only two selected live sharp placement signals were Balanced APP. That cannot support a separate live conclusion. Historical data may include later statistical revisions; stopping before the event prevents future-round leakage, not revisions to past data.

## Recommended next work

1. **Audit selection conditional on model-market disagreement.** Compare the backed player with comparable Balanced APP players not selected at the same events. Control probability/price, player, course, history depth and model version. A model can look adequate across the population while its largest apparent edges select optimistic errors. That mechanism is plausible here, but remains unproven.
2. **Replay the largest APP misses against issued targets.** Start with Cantlay, Rose and Hojgaard. Separate raw category skill, the total-SG recentering, course adjustment and any pre-event skill cascade. The historical benchmark has isolated a pattern; it has not identified which production component caused it.
3. **Evaluate any correction chronologically.** Use earlier events/seasons to fit shrinkage or market-calibration candidates and evaluate unseen events. Require improvement in APP and total-SG errors and finish/matchup probability calibration. Preserve the ≥5%/<5% split and correlated player exposure. Do not tune an archetype penalty or a stake rule on this season's losses.

Evidence and reproducibility: [analysis folder](permanent_data/research/balanced_app_finish_20260909/), including `analyze_finish.py`, `summarize_finish.py`, immutable input snapshots, `primary_evidence.csv`, `field_evidence.csv`, `summary.csv`, `uncertainty.json`, player/event summaries and leave-one-out checks. No production model, betting rule, Sheet, account or deployment was modified.
