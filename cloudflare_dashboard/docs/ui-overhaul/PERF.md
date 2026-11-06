# Performance pass r3-perf (2026-11-05)

Measure bundles with `node scripts/measure-bundle.mjs`; perceived speed with `node scripts/perf-check.mjs` (headless Chrome over CDP; needs
`node scripts/ui-preview.mjs --detach --no-build`). Raw runs: `perf-before.json`, `perf-after.json` (this folder).

## Bundle (gzip)

| Measure | Before (round 2) | After | Original budget | Ratchet now |
|---|---|---|---|---|
| Shell JS (first HTML, before any view) | 157.4 KB | 148.6 KB | 150 | 150 |
| This week, shell + view chunks | 216.4 KB | 196.8 KB | 205 | 198 |
| Why priced, shell + view | 216.4 KB (same chunk as This week) | 199.1 KB | | |
| Total CSS (sum of all CSS files) | 67.1 KB | 63.5 KB | 60 | 64 |
| This week 150 KB target | not met | not met (196.8) | | still `todo` |

What moved:
- Why priced and the player drawer split out of the This week chunk (`ExplainWhy.tsx`, `ExplainDrawer.tsx`); the drawer loads when a player is opened. -6.5 KB on This week.
- The weather parsers (`scoring-rules`, `weather-rules`) are read inside the lazy weather block (`WeatherThisRoundFor` in `ScoringView.tsx`) instead of in This week. -9.4 KB.
- Golfer search (`HeaderSearch`, player directory, profile rules) loads after hydration; the server-rendered input stays in place. -4.6 KB shell.
- Customize dialog is fetched on first click (`ShellExtras.tsx`); the shell no longer imports `components.tsx` (view boundary uses the same skeleton from `ui.tsx`; not-found/moved notices use `ui.tsx` primitives, identical markup). -5.5 KB shell.
- CSS: removed 190 dead rules (selectors whose classes no longer exist in any app source: old topbar, run machines, wx steps, shape pair, scoring confidence, etc.) and 6 KB of declarations shadowed by a later rule of the same selector. Pages already load only their own CSS (one CSS file per route chunk); the 64 KB budget sums all of them.
- Not reached: CSS 60 KB and This week 150 KB. A Chrome coverage run shows about 80 KB raw of rules never matched on any route, but they are conditional UI (menus, error and empty states, Tailwind utilities generated from words in source), so they were left. The 150 KB shell floor is React + vinext client (105 KB) plus the shell itself.

## Perceived speed (local preview, compressed like the edge, cold cache, median of 3 for the final run)

Phone = 375x812 touch, 4x CPU, fast 4G (1.6 Mbps, 150 ms RTT). Desktop = 1280x900 unthrottled.
"Before" is the build after the bundle cuts but before the perf fixes (single run). The preview server used to send everything uncompressed
(first run: phone LCP 5-7 s from 685 KB of JS alone); `ui-preview.mjs` now serves brotli/gzip like Cloudflare so the numbers are realistic.

| Route (phone) | LCP before | LCP after | CLS before | CLS after | TBT after | Longest task after |
|---|---|---|---|---|---|---|
| this-week | 5100 ms | 3340 ms | 0.200 | 0.020 | 25 ms | 103 ms |
| why-priced | 2880 ms | 2684 ms | 0.013 | 0.013 | 0 | 106 ms |
| players?player=10091 | 6124 ms | 5948 ms | 0.013 | 0.013 | 220 ms | 201 ms |
| weekly-players | 3204 ms | 3040 ms | 0.082 | 0.013 | 40 ms | 124 ms |
| bet-breakdowns | 3192 ms | 2796 ms | 0.013 | 0.013 | 5 ms | 99 ms |
| weather-effects | 4528 ms | 3692 ms | 0.013 | 0.013 | 0 | 103 ms |

| Route (desktop) | LCP after | CLS before | CLS after | Longest task |
|---|---|---|---|---|
| this-week | 180 ms | 0.131 | 0.036 | 0 |
| why-priced | 452 ms | 0.035 | 0.044 | 0 |
| players?player=10091 | 528 ms | 0.035 | 0.035 | 0 |
| weekly-players | 428 ms | 0.036 | 0.036 | 50 ms |
| bet-breakdowns | 428 ms | 0.042 | 0.045 | 0 |
| weather-effects | 460 ms | 0.036 | 0.036 | 0 |

Interaction (click a tab / toggle / row): event duration 16-72 ms, next two frames 7-39 ms, longest task on interaction 0-62 ms on the phone (budget 200 ms). Route change (nav link, a full page load in this app, warm cache) is 2.7-3.6 s to LCP on the throttled phone, 0.15-0.48 s on desktop.

Targets: CLS < 0.05 met on every route and viewport; no task > 200 ms on interaction met (the one 201 ms task is the Players page load, not an interaction); LCP < 2.0 s on the throttled phone NOT met (best 2.7 s).

## Why phone LCP is still 2.7-3.7 s

Waterfall (`perf-check.mjs --waterfall`): the shell (framework + vinext client + app shell, ~330 KB brotli with CSS and the 47 KB font) saturates the 1.6 Mbps link until about 1.8 s; hydration finishes about 2.1 s; the view's code and data (index, then the 87 KB explain document) follow. The LCP element is a text block that only exists once that data is in, so it cannot land before about 2.7 s without server-rendering the data. Remaining ideas that need Worker or architecture changes (not done): render the first screen of data on the server; a single-golfer lookup endpoint for the Players page (it downloads the 428 KB search index, then a batch, before it can draw: LCP 5.9 s, one 200 ms parse task); a preload hint for `index.json`.

## Perf fixes made (all in `app/`, no copy or style changes)

- Idle prefetch of the three busiest views and the golfer-search index now wait 8 s after load (they used to start at 2 s and share the link with the page's own data; the search index is 428 KB).
- Below-the-fold blocks on This week (hole strip, course geometry, weather cards, shape spotlight) request their chunk (Recharts is 88 KB) only when within 600 px of the viewport; placeholders keep their final height.
- This week asks for the event's explain document as soon as the index names it (`warmDashboardData`, from the shell), and asks for the stored "What changed" diff in parallel with the document and waits for it, so the strip arrives with the page instead of pushing it down (CLS 0.200 to 0.020 on the phone).
- Whole field and This week/Why priced no longer render the default event for one frame before the bar's event is known: that fetched the PGA run (224 KB document plus inputs) for nothing on a week with two events. The saved-skill breakdown under the field board appears only once the board's document is in, and a "Recent form" row reserves its height while ratings load (CLS 0.082 to 0.013).
- `scripts/ui-preview.mjs` compresses text responses (brotli/gzip, cached for static files) so local measurements match the edge. Set `UI_PREVIEW_NO_COMPRESS=1` for the old behaviour.
- `.archived-state li > a` gets a 40 px tap target (two phone warnings on History and Diagnostics from the archive notice).

## Verification

`npx tsc --noEmit` clean; `eslint . --ignore-pattern dist --ignore-pattern .next` clean; `npm test` 798 tests, 0 failures, 1 todo (the 150 KB This week target); `node scripts/ui-check.mjs --extra --crops --out dist-visual/r3-perf`: 80 cases, 0 errors, 0 warnings.
Tests changed: `tests/budgets.test.mjs` (ratchets and notes), `tests/views-render.test.mjs` (Why priced now renders from `ExplainWhy.tsx`).
