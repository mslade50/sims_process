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

# Integration pass r3-int (2026-11-06)

Raw run: `perf-r3-final.json` (3 runs, medians). Phone = 375x812, 4x CPU, fast 4G. Before = start of this pass (single run).

| Route (phone) | LCP before | LCP after | CLS before | CLS after |
|---|---|---|---|---|
| this-week | 3088 ms | 2944 ms | 0.412 | 0.007 |
| why-priced | 3528 ms | 2952 ms | 0.004 | 0.000 |
| players?player=10091 | 5948 ms (r3-perf) / 3000 ms | 3060 ms | 0.134 (intermittent) | 0.006 |
| weekly-players | 2576 ms | 2584 ms | 0.000 | 0.000 |
| weather-effects | 3400 ms | 3336 ms | 0.000 | 0.000 |

Desktop: LCP 128-204 ms, CLS 0.030-0.046 on every route (target < 0.05 met).

What changed
- CLS 0.41 on This week: `usePhone()` started false and flipped after mount, so the board section moved from below the lede to above "What changed" (shift of about 490 px). It now reads the viewport on first render (every caller mounts only after data is in the browser, never in server markup).
- Route code hints: `scripts/route-chunks.mjs` (postbuild) writes `dist/client/route-chunks.json` (route to view chunk, its non-shell imports and CSS from Vite's manifest). `worker/route-preload.ts` adds `Link: rel=modulepreload` headers and the route's stylesheets as `<link>` tags at the end of `<head>` on the HTML response (additive, streamed, any miss returns the response untouched; tests/route-preload.test.mjs). The stylesheets matter: once code and data arrive early, the lazy view's CSS (injected only when its code runs) arrived after the first render and shifted the Players page (CLS 0.13). Deploy note: `postbuild` runs with `npm run build`; if the deploy builds some other way the file is absent and the Worker simply sends no hints.
- Early fetch (head script): `/players?player=<id>` now starts `player_profiles/golfer/search-index.json?id=` (the one-golfer slice) before hydration; `/why-priced` starts the document it will open (pre-tournament run when a live run exists, else the latest) after the index names it. Players LCP 5.9 s to 3.0 s; Why priced 3.5 s to 2.95 s. Covered in tests/shell.test.mjs.
- No slim explain variant exists (the document is 111 KB brotli for Abu Dhabi); a slim one needs a publisher change, not done.

LCP < 2.5 s on the throttled phone: not reached (2.9-3.1 s on the three target routes). Arithmetic at 1.6 Mbps (200 KB/s) and 150 ms RTT: the first screen needs the shell JS 147 KB + CSS about 30 KB + HTML, then (sequentially, because the document key is in index.json) index 8 KB and the explain document 111 KB, plus the view 60 KB: about 370 KB = 1.85 s of pure transfer, about 0.45 s of round trips (HTML, then index, then document) and about 0.4 s of hydration at 4x CPU, which is 2.7-2.9 s. The waterfall shows the document arriving at 2.77 s. Getting under 2.5 s needs a smaller shell (floor is React + vinext client 105 KB) or server-rendering the first screen, both outside this pass.

Other fixes: Why priced rows use one stretched name button (one tab stop per row, whole row clickable, no nested handler); Ops and Inputs use `OnThisPage` tab mode (Players one-golfer already used `OnThisPage`); `/players` eyebrow is "Players" while `/weekly-players` stays "Week"; weather wave bars carry an axis ("-X easier / 0 = field average / +X harder", unit line); the contenders table is now compact rows; shapes scatter text is 13 user units (about 12.4 px on a 375 px phone) with crowded x ticks dropped; trend end labels sit in a right margin instead of on the line.

Budgets unchanged: shell 146.5 KB gz (148), This week + view 200.1 KB (201), CSS 65.3 KB (66). CSS did not get back to 64 KB: the contenders rows and weather axis added about 0.4 KB; removed `.in-tabs` and dead differ rules.
Verification: tsc clean, eslint 0/0, `npm test` 807 pass / 0 fail / 1 todo; ui-check r3-full 80 cases 0/0; r3-events 56 cases 0/0; contrast 0 failing groups.

# Round 4b: slim explain document, early diff, drawer on demand (2026-11-06)

Raw runs: `perf-r4b.json` (after), before = `dist-visual/R4b/perf-before.json`; phone = 375x812, 4x CPU, fast 4G, medians of 3, local preview (HTTP/1.1, so the index request queues behind ~40 script requests; a deployed HTTP/2 edge will do better than these numbers).

| Route (phone) | LCP before | LCP after | CLS before | CLS after | TBT before | TBT after |
|---|---|---|---|---|---|---|
| this-week | 2968 ms | 2688-2708 ms | 0.007 | 0.007 | 158 ms | about 130 ms |
| why-priced | 2876 ms | 2652-2660 ms | 0.000 | 0.000 | 1 ms | 0 ms |

LCP < 2.5 s was NOT reached (about 0.2-0.3 s short); the gain is -270 ms (This week) and -220 ms (Why priced).

What changed
- Slim explain variant, computed in memory by the Worker (`worker/explain-slim.ts`, `GET /api/golfprice/explain/<event>/<run>.json?slim=1`): same document without `scoring` (the round-score grid, 47% of the raw bytes) and `head_to_heads` (Distributions only), and per player without `scores`, `neighbors`, `form` and the drawer-only parts of `diverge` (peers, compare, sentences; `dg_check` and an empty `edge_split` stay because the Data Golf skill board and the `why_edge` wording read them). No R2 write, no write path; the body is cached in the Cache API per object etag; any miss or error falls through to the normal full document (older Workers ignore the query and serve the full one). Abu Dhabi live document: 88.7 KB to 41.3 KB brotli; the pre-tournament document 110.9 KB to about 67 KB. Parity: `node scripts/slim-parity.mjs` loads This week and Why priced for both live events with the slim document and with the `?slim=1` request rewritten to the full document; the visible text is identical (dist-visual/R4b/slim-parity.json).
- Client: This week and Why priced fetch the slim document first (head script, shell warm-up and `ExplainBody` all use `slimExplainKey`); the full document is prefetched when the page is idle (5 s after the data arrives, requestIdleCallback) and fetched at once when a player drawer opens (`DrawerHost`, shows "Loading player detail" until it is in).
- The head script now also asks for the stored "What changed" diff of the latest run in parallel with the document (it used to start after the document arrived and held the page back about 0.25 s). It computes exactly the name `latestChangesDataName` computes (tests/shell.test.mjs).
- Remaining floor: the document request still starts only after `index.json` names it (index arrival about 1.25 s locally because it shares the link with the 330 KB shell), then about 1 s for the slim document, then render about 0.3 s at 4x CPU. Going below 2.5 s needs the first screen's data resolved server side in one request (the Worker reading index.json and the slim document and returning them together, keyed by the event the head script already knows from `?e=` or local storage), or server-rendered markup; both change the loading contract and were not done.
- Budget: This week + view ratchet 201 to 201.5 KB gz (measured 201.05, +0.9 KB for the early diff request, drawer host and idle prefetch).

Other R4b changes (no performance impact): Round scores phone headline is one compact row so the complete histogram with its axis sits above the tab bar (dist-visual/R4b/s1); explain players display as "First Last" everywhere (parsed once in `parseExplain`, raw name kept for sorting); `tourLabel` shows two tours and "+N" in pickers; `bet-card-rules` STAGE_CLAIM regex had literal backspace characters instead of `\b` (so "R1 complete" notes were never treated as stage claims) and is fixed; the shell bar names "All events" on /bet-card and drops the single-event status chips there.

Evidence
- Drawer keyboard proof (`node scripts/a11y-drawer.mjs`): row Enter opens, focus inside, Tab x45 and Shift+Tab x12 stay inside, main and tab bar inert, Escape and the close button both close and return focus to the opener, on phone and desktop: 36 checks, 0 failures (`dist-visual/R4b/a11y-drawer.json`). The earlier "native actions timed out" was the harness sending `rawKeyDown` for Enter, which never fires a button click; a `keyDown` with text does.
- Populated Bet card (`node scripts/bet-card-interact.mjs`, fixture `dist-visual/D-r2/bet_card.populated.json` swapped into `.wrangler/live/owner/bet_card.json`, preview restarted, then restored: sha256 96b4586d...679ebe1 verified identical before and after): Enter and Space expand and collapse rows, group and book filters change the count and the polite status, Clear restores all 5 rows, combined filters keep status and Clear: 30 checks, 0 failures (`dist-visual/R4b/bet-card-interact.json`).
- Verification: `npx tsc --noEmit` clean; eslint 0/0; `npm test` 818 tests, 817 pass, 0 fail, 1 todo (150 KB target); `ui-check --extra --crops --out dist-visual/r4-full` 80 cases 0 errors 0 warnings; `contrast-check` 72 cases, 0 failing groups.
