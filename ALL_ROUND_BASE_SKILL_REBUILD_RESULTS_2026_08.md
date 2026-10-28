# All-Round Base-Skill Rebuild Results

Status: completed research evaluation; **do not promote**. No production file
or consumer was changed.

## Design actually run

- Coefficient fitting, architecture and hyperparameter selection, probability
  calibration, course-fit/downstream lambda selection, live-cascade lambda
  selection, and betting-rule selection used 2023 only.
- Pre-2023 data was used only to construct point-in-time histories and the
  frozen weather transform available entering each 2023 event.
- The target used every observed adjusted-SG round from R1 through R4. The 2023
  target contains 17,639 observed rounds across 44 events: 3,024 player-events
  with four rounds, 2,519 with two rounds, 156 with three rounds, and 37
  one-round WD/DQ diagnostics. Headline fitting required at least two rounds.
- Event weights were normalized so large fields did not dominate. The selected
  target sensitivity subtracts the strict-prior skill-composition shift of each
  surviving round rather than using the invalid retired post-cut correction.
- Thirty-four selectable challengers covered total SG, APP, OTT+APP, all four
  categories, clean IMG histories, ridge strength, robust fitting, survivor
  correction, and a weather-neutral second pass. Two raw-label arms were
  nonselectable diagnostics.
- 2023 OOF prediction was chronological by completed event, with same-date
  events batched. April-June fit probability maps and July-December selected the
  architecture. The five markets received equal selection weight: 72h and
  R1-R4.
- The locked evaluation used 2024 through 2026-08-13: 128 distinct events at
  72h/R1, 107 at R2/R3, and 104 at R4. It used BetOnline, Betcris, and Pinnacle,
  taking the best available price when books overlapped. Close was primary;
  open was a frozen sensitivity. The primary betting gate was at least 3%
  theoretical EV.
- Physical results and bookmaker settlement were kept separate. R1-R4
  0.5/0.5 void encodings were verified against equal round scores. For 72h,
  0.5/0.5 was accepted as a physical tie only when both golfers had the same
  completion round and final cumulative score; other book voids remained
  refunds rather than being mislabeled as score ties.

## 2023 selection result

No challenger passed the frozen baseline-dominance gate. All 34 challengers
had both worse aggregate proper score and worse flat-stake P&L than the control.

| 2023 selection result | Control | Best-ranked challenger |
|---|---:|---:|
| Mean market event-equal three-way log loss | 0.929794 | 0.932788 |
| Flat bets | 1,355 | 1,166 |
| Flat P&L | +11.01u | -54.74u |
| Mean market ROI | +2.75% | -4.20% |

The primary selection therefore ended with
`no_challenger_survived`. To complete the requested locked evaluation without
pretending there was a qualified winner, the deterministic best-ranked
non-control arm was frozen as a diagnostic fallback:

`allr__t1__w0__t_full__ridge__a1000`

This is the survivor-corrected all-round target, raw-weather history, total plus
OTT, APP, ARG, and putting history/reliability (103 features), ridge alpha
1,000, 0.5 preserved course-fit/downstream multiplier, and 0.5 common R2-R4
live-cascade multiplier. It is explicitly marked `promotion_eligible=false` in
every handoff and result.

## Locked 2024-2026 result

Primary close-price flat-risk performance at the 3% EV gate:

| Market | Diagnostic bets | Diagnostic P&L | Diagnostic ROI | Control bets | Control P&L | Control ROI |
|---|---:|---:|---:|---:|---:|---:|
| 72h | 2,807 | -211.28u | -7.53% | 2,944 | -88.98u | -3.02% |
| R1 | 1,275 | -38.63u | -3.03% | 1,330 | -8.19u | -0.62% |
| R2 | 1,906 | -30.45u | -1.60% | 954 | +13.48u | +1.41% |
| R3 | 1,227 | +12.36u | +1.01% | 1,040 | +4.78u | +0.46% |
| R4 | 1,290 | -79.17u | -6.14% | 1,327 | +32.66u | +2.46% |
| **Pooled** | **8,505** | **-347.17u** | **-4.08%** | **7,595** | **-46.25u** | **-0.61%** |

The paired event bootstrap estimates a pooled diagnostic-minus-control flat
P&L difference of **-300.93u**, with a 95% interval of **[-539.73u,
-63.79u]** and only a **0.71%** probability that the diagnostic was better.
R3 was the only positive relative betting result; R4 was decisively worse.

Proper scoring agreed with the betting result. Event-equal three-way log loss
was worse for the diagnostic in every market:

| Market | Diagnostic | Control | Delta (lower is better) |
|---|---:|---:|---:|
| 72h | 0.840144 | 0.839112 | +0.001033 |
| R1 | 0.961340 | 0.960335 | +0.001004 |
| R2 | 0.965773 | 0.964678 | +0.001095 |
| R3 | 0.972710 | 0.972211 | +0.000499 |
| R4 | 0.959937 | 0.956545 | +0.003391 |

The equal-market average regression was +0.001405. The R4 log-loss regression
was individually clear in the event bootstrap: +0.003391, 95% interval
[+0.001135, +0.005758].

Opening-price sensitivity reached the same conclusion: the diagnostic lost
-260.11u on 8,433 bets (-3.08%) versus -10.29u on 7,993 control bets (-0.13%).

## Stability checks

Close-price flat P&L was worse in every evaluation year:

| Year | Diagnostic P&L / ROI | Control P&L / ROI |
|---|---:|---:|
| 2024 | -191.20u / -6.38% | -37.07u / -1.31% |
| 2025 | -61.62u / -2.11% | +23.06u / +0.91% |
| 2026 partial | -94.35u / -3.64% | -32.24u / -1.44% |

The diagnostic lost at all three selected books: Betcris -175.79u (-6.12%),
BetOnline -118.68u (-3.57%), and Pinnacle -52.71u (-2.29%). It also lost in
all four projected-skill buckets.

At the primary gate it lost in every theoretical-edge bucket: 3-5% was
-148.15u (-4.80%), 5-10% was -123.54u (-3.48%), and greater than 10% was
-75.48u (-4.04%). The reported but non-betting 0-3% sensitivity was also
negative at -188.77u (-2.36%).

The weather-neutral arms did not rescue selection; none passed the 2023 gate,
and the frozen diagnostic came from W0. Weather coverage is incomplete: 443 of
the 17,639 2023 target rounds lacked both weather covariates and therefore
could not be neutralized in W1.

## Verdict and limits

This rerun answers the requested question negatively: replacing the legacy
control with this all-round regression family would have made the 2024-2026
matchup portfolio materially worse. It should not be promoted or averaged into
production.

This does not prove that all possible all-round skill models are inferior. The
largest remaining design limitation is the short, single-year coefficient
training window. The control, course-fit term, and live cascade are also
retrospective current-code reconstructions rather than versioned historical
production snapshots. Finally, the 2024-2026 archive had been viewed during an
earlier incorrectly targeted study, so a future candidate still requires a
prospective shadow period.

## Reproducibility

- Frozen contract SHA-256:
  `4091A0192A959640A9DC88FF6B348759CF3052ED7AED3A116B5989019111374C`
- Selection freeze SHA-256:
  `BDB4D4C3A12A89B93E8F2E014CD6326DFA4760A42594F4C84BF21E8F4EBCD20A`
- Score-audited tournament-outcome authority SHA-256:
  `999E14F09D04063EAF3ED62C9EC8970F46FE4C9978B20111E826A96DE6F09B39`
- Prediction-authority SHA-256:
  `7E37D60B9507F9E8BAF3FCBE1ABD2E9848E15F301A14E5EFC1ED439D2FDA212F`
- Evaluation summary SHA-256:
  `A3E2BCFA1166E9B31E509B38BAF3122C36A4671D39DDBA7D31D5FD9DF7AF92ED`
- Betting summary SHA-256:
  `364A04D335D829F4CBF85B845748D4176ED4E0E4AAF73261560D0A2A179267DB`
- Verification: 60 focused tests passed.

Authoritative result directory:

`permanent_data/research/base_skill_rebuild_20200101_20260822/all_round_joint_market_eval_2024_2026_score_authority_v5_authorized_20260823`
