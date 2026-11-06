# Codex product-design review — round 4

Overall **9.1/10**, the minimum category score; target not met. The scoring view, theme treatment and core wayfinding earn 9.5. The remaining work is specific: reading order on phone, probability units, filter feedback and load stability. Benchmark products set the supplied quality bar; this is not a measured comparison against their current releases.

| Category | r3 | r4 | Reason |
|---|---:|---:|---|
| Information architecture & wayfinding | 9.2 | 9.5 | All-events Bet card shell fixes scope; event/run context, archive destinations and grouped navigation are coherent. |
| Clarity of meaning | 8.9 | 9.3 | Why priced now uses points consistently; Bet card still calls a probability difference +16.1%, and Weather's forecast age has an ambiguous reference (#2, #4). |
| Visual hierarchy & layout | 9.2 | 9.2 | Scoring now leads with a complete plot; Weather and Inputs still consume the phone opening with framing, and populated card starts below filters (#3–5). |
| Typography | 9.3 | 9.5 | Numeric drawer evidence, first-last identities and compact tour labels resolve the prior defects; strong tabular comparison and readable themes. |
| Colour, theming & contrast | 9.5 | 9.5 | Restrained surfaces, explicit signs, distinct curve patterns and legible inspected light/dark selections retain the earned score. Not blanket AA certification. |
| Data visualisation | 9.3 | 9.5 | Complete phone histogram, named populations, direct curve labels and bounded near-certain odds now meet the bar. |
| Interaction & feedback | 9.1 | 9.4 | Native drawer and More work; Why priced search lacks count feedback/reset parity, while populated-card proof is supplied rather than independently repeated (#6). |
| Responsiveness & mobile | 9.2 | 9.2 | Comparison rows and scoring are effective; Weather/Inputs and populated-card controls still postpone the working surface (#3–5). |
| Accessibility | 9.2 | 9.3 | Modal entry/inert/Escape/return work; live search still says “1 players, win” without a result announcement (#6). |
| Consistency & design system | 9.4 | 9.4 | Names, KPI units and most gaps align; card edge notation and filter feedback still break the shared contract (#2, #6). |
| Performance & perceived speed | 9.1 | 9.1 | Slim payload helps; independent cold phone LCP remains 2.68–3.40 s, golfer has a 337 ms task and repeatable desktop CLS .113 (#1). |
| Craft & delight | 9.2 | 9.3 | Distinctive, useful golf desk; duplicated quiet-state headlines, verbose openings and imprecise edge notation prevent effortless daily use (#2–5, #7). |

## Evidence and limitations

Read rubric, reviews r0–r3, DESIGN_SYSTEM, PERF including Round 4b, perf-r4b.json, drawer and populated-card interaction evidence. Inspected actual images: all 20 r4-full phone-dark crops in a contact sheet; full-size Weather phone/desktop, Inputs phone and Adjust desktop light, Why priced phone light, Distributions desktop light, quiet Bet card phone dark, R4b scoring phone, R4b populated full-length phone light, R4-codexA-pop populated crop and R4-claude golfer/Adjust. This samples themes/viewports, not every capture individually. Earlier populated captures retain the old event shell; current r4-full and R4b show the corrected All events shell.

Live local owner data: pointer opens Rory's drawer, initial focus is Close, main is inert, Tab works, Escape removes modal/inert and restores Rory focus. At 375×812, theme changes computed background; More opens named Site menu with focus on Close and inert main; Tab/Escape work. Event switcher ArrowDown/Enter navigates to the PGA event URL and PGA data. Search James produces “1 players, win”, no matching live result announcement, and no horizontal overflow. A bottom-link click timed out without navigation; no product failure inferred from this one automation timeout. Perf harness independently exercised route changes successfully. No submission, job, fixture replacement or production mutation performed. Populated-card Enter/Space/filter/Clear evidence is the supplied 30-pass harness report and screenshots; no live populated dataset was substituted. No screen-reader session or deployed Worker measurement.

Fresh independent `node scripts/perf-check.mjs --routes this-week,why-priced,players?player=10091,weather-effects --viewports phone,desktop --runs 1`, no output JSON. Initial sandbox DevTools connection failed; authorized local-browser retry succeeded. One sample per route/profile, plus a second golfer desktop cold load; these are not p75 measurements. Phone: 375×812, 4× CPU, 1.6 Mbps and 150 ms RTT; desktop 1280×900 unthrottled.

| Route | Phone LCP ms / CLS | Phone max task / TBT ms | Desktop LCP ms / CLS |
|---|---|---|---|
| This week | 2872 / .007 | 180 / 166 | 204 / .046 |
| Why priced | 2676 / .000 | 67 / 0 | 172 / .044 |
| One golfer | 2988 / .007 | 337 / 446 | 364 / .113 |
| Weather | 3396 / .000 | 60 / 1 | 172 / .045 |

Second golfer desktop: LCP 216 ms, CLS .113; page container grows from 2794 to 4495 px during initial resolution. Identify actual shifting nodes before selecting a fix. Measured interaction event durations were 16–64 ms, with no new long interaction task. Supplied medians of three put the slim core routes around 2.65–2.71 s; my slower sample does not negate that improvement.

Reference thresholds: CWV good LCP ≤2.5 s and CLS ≤.1 at p75 on real devices; INP ≤200 ms. For this data-heavy private Worker dashboard, ≤3.5 s under this constrained synthetic profile is tolerable, but 9.5 speed calls for core routes consistently near/below 2.5 s, CLS preferably <.05, and no conspicuous >200 ms main-thread stall. Lab event timings are not field INP. Local compression/throttling does not reproduce Access, edge placement, cold isolates or R2 latency; HTTP/2 improvement is a hypothesis, not verified delivery evidence.

## Each r3 remaining item verified

1. **Fixed:** All-events shell and normalized stage notes remove the conflicting selected-event presentation; current quiet card no longer repeats “R1 in play” beneath “R1 complete”. Populated R4b uses All events too.
2. **Partly:** Each quiet event is rendered once; global “No bets to suggest right now” then “Nothing to suggest” still repeats the conclusion (#7).
3. **Partly:** Why priced summary/rows now use pts with explicit model/book comparisons; Bet card retains % for percentage-point edge (#2).
4. **Partly:** Weather sensitivity/extremes moved into disclosure; remaining first-screen brief still delays all four-round evidence (#3).
5. **Partly:** Inputs metadata chips consolidated; selector, run details, introduction and search instructions still consume the phone opening (#4).
6. **Fixed:** R4b phone scoring shows histogram, axes, par/median and individual-round legend above the bottom bar.
7. **Fixed:** Drawer uses numeric value plus sentence-case unit; golfer reference label is shorter and comparison now precedes biography.
8. **Fixed:** Make-cut legend bounds near-certain display to >99% and −9900 or shorter, avoiding giant American odds while retaining the underlying values.
9. **Fixed:** Independently reproduced native drawer open/focus/Tab/Escape/return; supplied phone/desktop trap and close-button report has zero failures.
10. **Fixed for requested explain/directory surfaces:** First-last explain identities and compact tour summaries replace competing name order/long tour strings. Inputs still contains “McIlroy, Rory” in its field-best tile; unify during #4.
11. **Not fixed on Why priced:** James search still renders “1 players, win”; no result live region. Bet card has a polite count, demonstrating the available pattern (#6).
12. **Partly:** Slim explain and early diff reduce transfer; populated harness proves expansion/filters. Phone target, golfer parse cost and now reproduced desktop shift remain (#1).

## Remaining changes, ranked (covers every score below 9.5)

1. **Stabilize and accelerate One golfer; finish core cold-load work** — One golfer desktop both themes, core routes phone both. Trace the .113 shift through staged profile/current-event data; reserve the actual comparison geometry or resolve its inputs together. Profile the 337 ms phone task before splitting/defering parsing. Resolve event index plus slim first-screen data together, or render the initial answer server-side, to remove sequential waits. Likely `PlayerProfilesView.tsx`, `PlayerDeepDive.tsx`, profile CSS, `data.ts`, `worker/player-one.ts`, `worker/route-preload.ts`, `worker/explain-slim.ts`. Recheck cold LCP/CLS and task traces; do not sacrifice visible evidence. Addresses performance.
2. **Use explicit percentage-point edge everywhere** — Bet card populated, phone/desktop, both themes. Replace +16.1% edge/headline/accessible summary with +16.1 pp edge; explain chance minus break-even chance in place. Distinguish any ROI separately. Likely `BetCardView.tsx`, `bet-card-rules.ts`, common formatting. Addresses clarity/consistency/craft.
3. **Bring Weather's round comparison into the opening** — Weather 375×812 both themes. One numeric easier/harder line plus tee-wave finding; put repeated “moves average and cut line, not who wins” explanation and forecast model metadata behind disclosure. Follow immediately with compact four-round bars, shared scale and next drill-down. Current full-size phone crop reaches Four rounds only near y=750. Likely `app/WeatherEffectsView.tsx`, weather brief component and weather CSS. Addresses hierarchy/mobile/craft.
4. **Put Inputs' player answer before the setup prose** — Inputs Players 375×812 both themes. Retain one compact run selector, collapse run details, merge the repeated pick/search instructions, then show search and selected-player evidence. Field tiles currently start beneath the bottom bar; avoid consuming a screen to explain a search box. Normalize the remaining field-best name. Weather also says “forecast under an hour old” beside yesterday's update: explicitly label that as forecast age at pricing, and show current forecast timestamp separately. Likely `InputsView.tsx`, inputs CSS, Weather brief component/formatter. Addresses hierarchy/mobile/clarity/craft.
5. **Compact populated Bet card filters** — Bet card 375×812 both themes. Lead with first suggestion after a compact applied-filter row; disclose Market/Book controls behind Filters while preserving count and Clear. R4b populated image places first bet near y=650. Likely `BetCardView.tsx`, `bet-card.css`. Addresses hierarchy/mobile/craft.
6. **Finish Why priced result feedback** — Why priced phone/desktop both themes. Correct singular/plural; add polite atomic result-count status, and one clear reset covering query plus active filters. Independently verify keyboard and screen-reader announcement once. Likely `ExplainWhy.tsx`, shared filter/status primitive in `ui.tsx`. Addresses interaction/accessibility/consistency.
7. **Remove repeated quiet-state headline** — Bet card phone/desktop both themes. Keep one global no-bets finding followed directly by the event rows and next checks; omit “Nothing to suggest” when it repeats that finding. Likely `BetCardView.tsx`, `bet-card.css`. Addresses craft.

Top remaining: (1) repeatable golfer shift/main-thread cost and core phone LCP; (2) percentage-point edge labels; (3) Weather/Inputs/populated-card opening order; (4) Why priced result announcements/reset. Only this review file was written in the project. No commit, push, deploy, form submission or job start. No additional ideas artifact: the explicit one-output constraint takes precedence.
