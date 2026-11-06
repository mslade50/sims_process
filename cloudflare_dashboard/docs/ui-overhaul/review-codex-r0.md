# Codex product-design review — baseline r0

Reviewed baseline `2026-11-06T01-34-32`: all 72 actual screenshots in viewport/theme contact sheets, with full-length inspection of This week, Weather, Round scores and Why priced. Read RUBRIC.md, INVENTORY.md, report.json and shared UI/component/token/shell source. This is a design assessment, not a runtime certification. Keyboard, screen reader, loading transitions and measured speed were not exercised; those category scores are provisional. No production data or application code was changed.

The interface has useful analytical depth, calm themes and unusually good plain-language explanations. It currently feels like a collection of internal reports. It lacks the decisive first screen, compact working surfaces and trustworthy context of an exceptional daily instrument. Preserve the depth; redesign its order and access.

## Scores

| Category | Score | Reason |
|---|---:|---|
| Information architecture & wayfinding | 6.4 | Sensible groups, but competing event contexts and buried Bet card weaken orientation. |
| Clarity of meaning | 7.3 | Strong explanations; global freshness and forecast/model/baseline comparisons remain ambiguous. |
| Visual hierarchy & layout | 6.1 | Similar-weight panels, filter walls and long tables repeatedly bury the answer. |
| Typography | 6.7 | Headings work; 11–12px supporting labels and narrow verbose tiles weaken scanability. |
| Colour, theming & contrast | 7.4 | Both themes are coherent; blue scaffolding and muted detail dilute semantic emphasis. |
| Data visualisation | 6.8 | Useful distributions and signed effects, but inconsistent baselines and buried charts limit comparison. |
| Interaction & feedback | 6.3 | Shared primitives exist; empty states, tiny controls and separated run/status flows need refinement. |
| Responsiveness & mobile | 4.8 | Run overflows to 754px at 375px; stacked report layouts become exceptionally long. |
| Accessibility | 6.0 | Helpful reduced-motion and region primitives; small targets, tiny labels and state semantics need proof. |
| Consistency & design system | 6.8 | Shared tokens are real, but headers, controls, cards and responsive patterns diverge. |
| Performance & perceived speed | 6.2 | Lazy views and skeletons help; recorded prefetch errors and long dense DOMs undermine confidence. |
| Craft & delight | 6.0 | Competent visual foundation; little distinctive golf intelligence in the reading experience. |

**Overall: 4.8/10**, the minimum category score required by the rubric. No category meets 9.5. Scores measure this instrument against the requested exceptional bar; they are not claims of measured superiority or inferiority to a competitor's current release.

## Changes by category

Within each category, changes are ranked by impact. “Both” means dark and light; desktop and phone are specified where the treatment differs.

### 1. Information architecture & wayfinding — 6.4

1. **All event routes, both viewports/themes:** establish one selected event and run context. The shell combines WWT and Abu Dhabi while This week shows Abu Dhabi and Weather shows WWT. Replace the combined title with the active event, course, stage and run time; explicitly mark any intentional cross-event comparison. The owner must never infer which tournament a number belongs to.
2. **All routes, desktop both:** make header columns shrink safely: page label, search, event context with `min-width:0`, nonshrinking freshness. The report records 31% event/freshness overlap across 30 desktop cases. Use a deliberate second line before text collides.
3. **Phone shell, both:** provide persistent direct access to This week, Bet card, Players and More, with current destination named. Move Bet card out of a deep owner grouping; this is a primary daily destination for this job.
4. **This week / Why priced / Distributions / Scoring, both:** add a compact local analysis navigation that preserves event/run/player. Give the destinations distinct questions: overview, price drivers, finish chances, scoring. The current sidebar makes related work feel scattered.

### 2. Clarity of meaning — 7.3

1. **All event routes, both:** replace “Oldest 15 h ago” as the dominant trust signal with separate “Model priced …” and “Forecast updated …” timestamps. Put the oldest dependency and its consequence in a disclosure; a green general pill cannot explain whether these prices are usable.
2. **Weather, both:** distinguish absolute course difficulty, weather versus typical conditions, and player-relative advantage in each chart title and unit. Show “+0.74 strokes vs typical weather; harder” beside the field effect and “−0.27 vs field; helped” beside a player. Preserve the current explanation that the effect moves everyone, not just the cut line.
3. **Performance / Bet breakdowns, both:** put sample size and uncertainty next to each result, with live/paper and open/close basis in the same reading unit. The scorecard's “too few events” caveat is valuable; make it govern the displayed precision and emphasis rather than compete with a large decimal.
4. **Ops / Model walkthrough, both:** separate “no evidence received” from “failed” and “pending.” Show last observed success, missing proof and the next useful action. Repeated NOT YET VERIFIED chips currently require the owner to reconstruct the diagnosis.

### 3. Visual hierarchy & layout — 6.1

1. **Bet breakdowns, phone both:** put a result summary and edge-bucket table before the enormous filter panel. Replace the chip wall with a Filters button and applied-filter summary. The baseline spends almost the entire captured opening on configuring the answer.
2. **Weekly players / Inputs / Why priced, phone both:** show search and useful player rows immediately after the title/context. Collapse methodology and advanced filters; render compact player rows with three decision columns and expandable drivers. Keep full data available without rendering every player as a tall report.
3. **This week, both:** make one compact event brief: stage, favorite, largest meaningful move, expected winning score and data confidence. Put leaderboard and market comparison next; course/weather methodology becomes selectable detail. The current full page repeats the same card rhythm for many screens.
4. **Weather, desktop both:** use a deliberate four-round strip or 2×2 comparison. The fourth round currently wraps alone below three tall cards, leaving a large dead quadrant. Shorten the round summaries and place shared explanation once below them.
5. **Bet card, both:** replace two large nested empty boxes with compact event status rows, why no suggestions are available, and when prices will next be evaluated. Empty whitespace should not be the primary content.

### 4. Typography — 6.7

1. **All phone routes, both:** set primary reading text to 15–16px, supporting text to at least 12–13px, inputs to 16px. The source's 11px labels and 14px body do not support long explanations comfortably; compact content by editing/restructuring rather than shrinking it.
2. **Inputs / Performance / Weather, both:** replace uppercase multiline KPI explanations with sentence-case labels and one concise qualifier. Reserve tracking and capitals for short structural labels. Narrow two-column phone tiles currently turn definitions into thin columns of tiny text.
3. **All comparison tables, both:** consistently use lining tabular figures, right-aligned numeric columns, stable decimal precision and units in headers. Give player names a stronger weight than metadata; avoid making every row component equally emphatic.
4. **All routes, desktop both:** specify an actual locally bundled typeface or a deliberate system stack. Inter is currently only a preferred installed font. Fix the intended metrics so labels and headers do not vary silently by machine.

### 5. Colour, theming & contrast — 7.4

1. **Shell and every analytical page, both:** reserve accent blue for active selection, links and the primary comparison. Neutralize most eyebrow labels, borders and pills; their repeated blue emphasis currently competes with the evidence.
2. **Weather / Why priced / Performance, both:** define color by comparison meaning and pair it with a signed value and explicit direction. A positive stroke change can be bad for scoring; a player benefit is a different baseline. Never make the owner decode green without its basis.
3. **Tables/forms, light:** reduce the repeated elevated white-card effect and use stronger neutral rules and row highlighting. Pale surrounding blue plus many white panels reads as a generic admin template; a flatter working canvas will make selected evidence stand out.
4. **Small chart labels / disabled controls / metadata, both:** audit rendered contrast for the actual combinations, including opacity and custom accents. Token-test comments are useful but do not prove composed chart/hover states pass AA; retain a readable neutral fallback.

### 6. Data visualisation — 6.8

1. **This week / Why priced, both:** create a shared model-versus-book comparison row: player, model probability, market probability, signed gap on a zero-centered scale. On phone preserve those columns; put the driver explanation in an expander. This makes disagreement visible before prose.
2. **Weather, both:** use a zero-centered diverging player-effect chart with strokes-per-round ticks, shared scale and directly labeled extremes. Current rightward green bars convey ranking better than magnitude relative to zero; show helped and hurt together.
3. **Scoring, phone both:** move the outcome histogram directly under the expected-score headline. Mark median, the existing 80% interval and par on the plot, and label that it represents an individual player's round across the field. Do not let the interval for the field average be confused with individual score dispersion.
4. **Distributions, phone both:** lead with the selected two-player comparison and label curve endpoints directly. Keep a fixed player identity/style across every chart, and offer a compact probability table alongside the plot. Introductory text should follow the comparison.
5. **Weather forecast history / historical scoring, both:** put dates, sample size and units on the chart itself. Tiny unscaled sparklines cannot explain the size of a forecast revision; use consistent axes when comparing rounds and show the change numerically.

### 7. Interaction & feedback — 6.3, provisional

1. **Run, phone both:** keep the selected job, event, target machine and latest queue status together. After submission, replace the action area with queued/running/succeeded/failed progress and a clear next action; do not require a trip to the bottom of the page to find the result. This is a recommended behavior, not an observed submission test.
2. **Weather edits / Inputs, both:** use an explicit change tray showing original → edited value, affected event/round, saved versus applied status, and the reprice action. A save must not visually imply that existing prices have already changed. Keep feedback inline and persistent.
3. **Bet card / Players / archive routes, both:** distinguish true empty results, no selection, intentionally unpublished archive data and service failure. The archive's “temporarily unavailable” wording misrepresents an intentionally absent dataset; offer the current destination without making it look like a system outage.
4. **Filters and chart selections, both:** show an applied-filter count, reset action and result count adjacent to the control. Standardize selected/hover/focus/disabled treatments; preserve selections when moving between related event analyses. Verify these states interactively before raising this score.

### 8. Responsiveness & mobile — 4.8

1. **Run, phone both:** fix the actual 754px document width at 375px. Bound long freshness/source names, use `min-width:0` on grid children, and constrain any wide table to its own labeled scroll region. Resolve the reported course-history/Current collision; recheck at 320, 375 and 430px.
2. **This week / Why priced / Weather, phone both:** default to 8–12 compact rows with Show more and player search. Keep all data available on demand. The full-length Why priced capture is a long column of expanded player blocks; this is a navigation failure, not useful density.
3. **Bet breakdowns / Weekly players / Inputs, phone both:** move advanced filters and explanations into disclosures after the first answer. Do not stack the entire desktop control surface before the result.
4. **Every phone route, both:** make action hit areas at least 44px, including menu, golfer names, term triggers and filters. The report finds 35 of 64 targets below 40px on This week; expand hit areas without adding a border around every term.
5. **Phone shell, both:** reduce repeated page title/search/event/freshness scaffolding to a compact two-row context block. Let search expand on demand; keep event and stage readable instead of truncating the most important orientation text.

### 9. Accessibility — 6.0, provisional

1. **All phone routes, both:** address measured target sizes and collisions first. Labels that visually overlap and 17px-high term buttons are concrete barriers even before assistive-technology testing.
2. **Distributions / Weather / Scoring, both:** give charts a concise finding plus an accessible structured data alternative, named axes and non-color series differentiation. The generic Sparkline label default “Trend” cannot communicate a result; require meaningful labels from callers.
3. **Navigation / player search / disclosures, both:** verify keyboard order, escape-to-close, focus return, current-page indication and expanded state. SegmentedControl source uses a group and visual active class; add programmatic pressed state for toggle buttons or correct tab semantics when they switch panels.
4. **Run / editable forms, both:** announce asynchronous completion/errors without announcing every animated numerical frame. Keep durable error text associated with the relevant field. Check reduced motion, 200% zoom and a screen reader; none was proven by these screenshots.

### 10. Consistency & design system — 6.8

1. **All routes, both:** standardize a compact page header contract: question/title, event/run context, short answer, actions. Shot data's understated heading, Ops' report layout and This week's hero currently feel like separate products.
2. **Shared components, both:** define three surface roles: page section, data surface, exceptional callout. Remove routine card-inside-card nesting and choose one radius per role. Preserve true grouping with whitespace and rules.
3. **All controls, both:** use segmented controls only for mutually exclusive views; checkboxes/multiselect for combined filters; buttons for actions; links for navigation. Standardize height, padding, focus and selection rather than mixing tiny pills, boxes and text controls.
4. **Shared tokens / view CSS, both:** introduce an explicit 4/8/12/16/24/32 spacing scale and shared responsive table/player-row patterns. Keep view-specific CSS for genuinely different visualizations; consolidate duplicated layout patterns before tuning every route independently.

### 11. Performance & perceived speed — 6.2, provisional

1. **Routes recorded in report, both viewports/themes:** resolve the vinext RSC prefetch error (`TypeError: f is not a function`) and test navigation on the local build and deployed application. The report contains 12 console-error cases, including desktop cases; do not dismiss these as phone-only or assume production is affected without checking.
2. **This week / Why priced / Weather / large directories, both:** render only the initial useful result slice and defer collapsed details. Measure large list costs; paginate before reaching for virtualization. Keep shell/context visible during event changes.
3. **All data routes, both:** use section-shaped skeletons that reserve final geometry and keep previous valid content visible with an explicit updating state. Shared generic KPI skeletons are a foundation, not evidence that each page avoids layout shift.
4. **All routes, phone priority:** measure cold/warm navigation, LCP, INP, CLS and API timings before making speed claims. Prefer animated changes from the previous value over counting trusted probabilities up from zero on each mount; provisional intermediate numbers are distracting in a pricing instrument.

### 12. Craft & delight — 6.0

1. **This week and Bet card, both:** establish a signature “golf intelligence brief”: a confident event/stage line, one consequential finding, precise evidence and one useful drill-down. Make no-bet states equally decisive. This is where the product can exceed a generic dashboard.
2. **Every dense analytical route, both:** use editorial rhythm: claim → compact comparison → detail on demand. Remove repeated eyebrows and inset explanatory cards; give the one key number the strongest visual emphasis. Rich information can remain calm when it has order.
3. **Shot data, both:** replace repeated decorative golf-hole drawings with a compact, truthful collect → validate → publish progression, timestamp and blocking action. The diagrams add identity but consume space without distinguishing tour health.
4. **Live price/mover surfaces, both:** use a brief changed-value highlight tied to real updates, with previous value and timestamp available. Avoid perpetual pulse as decoration or zero-to-value flourishes. Polish alignment, signs, units and empty states before adding animation.

## North star

Build an expert golf desk with the clarity of a sports scorecard and the precision of a trading blotter.
Desktop: 208px quiet navigation, 56px context bar, fluid content grid; two columns only when comparing evidence.
Phone: two-row event/stage context, a short finding, usable result rows, and four primary destinations.
Every event page shares one event/run identity; comparisons are explicit, never implied in a concatenated title.
Use a bundled neutral sans with excellent numerals; 28px page titles, 18px section titles, 14px desktop data, 16px phone body.
Use tabular figures, consistent signs and restrained decimal precision; units live beside values or in headers.
Dark: graphite/navy ground, subtly lighter data surface, near-white text; light: pale neutral ground and crisp white data surface.
One cool blue accent for selection; green/red only for stated favorable/adverse comparisons, amber for actionable uncertainty.
Prefer rules and compact rows to nested rounded cards; 8px control radius, 12px exceptional surfaces, spacing in 4px increments.
The signature is a model-versus-market gap rail with a shared zero, not a decorative graphic.
A price row reads: player → model → book → gap → concise driver; tap opens its full evidence.
This week opens with what changed, what matters and whether the data is current enough to act on.
Weather opens with field difficulty and wave advantage; scoring opens with the outcome distribution and par.
Bet card states the current opportunity or explains the absence, with an honest next-check time.
Keep methodology one action away; never sacrifice analytical depth to achieve a sparse screenshot.
Motion marks real changes once; empty, stale and unverified states receive the same care as success.

## Ten highest-impact changes

1. Unify selected event/run identity and fix shell event/freshness collisions.
2. Fix Run's 754px phone overflow and freshness-row text collision.
3. Put answers before filter walls on Bet breakdowns, Weekly players and Inputs.
4. Replace expanded phone player reports with compact comparison rows and on-demand evidence.
5. Rebuild This week around a concise event brief and prioritized live leaderboard.
6. Make model-versus-market gaps a directly labeled, zero-centered comparison pattern.
7. Put Bet card within one tap and replace oversized empty boxes with useful event status rows.
8. Raise phone type/target sizes and make selected states programmatically accessible.
9. Reorganize Weather into a four-round comparison with explicit baselines and a signed player-effect chart.
10. Repair recorded prefetch errors and verify real navigation, loading and status transitions.

## Evidence limits and delivery

The baseline report records 72 cases, 46 errors and 40 warnings. These are checks, not 46 distinct product defects. Intentional missing legacy/market files were not scored as reliability failures. No live submission, override, reprice or betting action was performed. No accessibility audit, contrast measurement or speed benchmark was run. Player detail and populated Bet card states are absent from the baseline; they need their own review before a 9.5 claim. Only this requested project output file was written; no commit, push or deploy.
