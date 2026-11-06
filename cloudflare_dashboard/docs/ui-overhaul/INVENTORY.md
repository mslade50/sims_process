# Dashboard UI inventory (baseline, 2026-11-05)

Baseline screenshots: `dist-visual/2026-11-06T01-34-32/` (`report.html`, `report.json`, `shots/<route>__<phone|desktop>__<dark|light>.png`, 72 cases).
Preview: `node scripts/ui-preview.mjs --detach` (see script header). Live data lives in `.wrangler/live/` (key == path), pulled read-only from R2.

## Architecture in brief
- vinext app + Worker (`worker/index.ts`). One catch-all page `app/[view]/page.tsx` renders `app/DashboardApp.tsx` (shell, nav, header, route -> view).
  Views are `lazy()` imports registered in `DashboardApp.tsx` (about lines 49-70). Routes, nav groups, owner-only list, redirects: `app/shell-rules.ts`.
- Data: `useDashboardData(key)` in `app/data.ts` -> `/api/golfprice/<key>` (R2 `golfprice/...`), `/api/data/<key>` (R2 `data/...`),
  `/api/owner-data/<key>` (R2 `owner/...`, owner only). Event picker is shared: `golfprice/index.json` (event list, run checkpoints, keys per event).
- Pure logic sits in `app/*-rules.ts` (tested without DOM); keep rendering and rules separate when restyling.
- Header (in DashboardApp): page title, golfer search (`HeaderSearch.tsx`), current event + course, freshness pill ("Oldest 15 h ago").
- Owner pages: `model-walkthrough, shot-data, ops, run, weather-edits, bet-breakdowns, bet-card` (OWNER_ONLY_VIEWS). Redirects: `weather`->`weather-effects`, `sg-distributions`->`players`. `research` = retired notice.

## Routes
All rendered with real live data unless noted. CSS shown is route-specific; every page also uses `tokens.css`, `globals.css`, `shell.css`, `motion.css`.

| Route | Shows | Main files (+ CSS) | Data (R2 key under golfprice/ unless noted) | Baseline |
|---|---|---|---|---|
| `/this-week` (also `/`) | Event hero, favorite, biggest mover, winning score, what changed, in-play leaderboard, shape matchups, course holes, weather summary, top-10 table | `ExplainViews.tsx` (ThisWeekView), `ExplainBits.tsx`, `ExplainCharts.tsx`, `ShapeViews.tsx`, `DivergeView.tsx`, `CourseGeometry.tsx`, `OddsSignalsPanel.tsx`; explain.css, shapes.css, course-geometry.css | index, explain/<ev>/latest + run files, weather/<ev>/*, course_geometry/*, changes.json (404 ok), odds_signals/latest | OK. Defaults to Abu Dhabi (in play); WWT selectable. Longest page (4816px desktop) |
| `/weather-effects` | How weather shifts scoring, per-player winners/losers, AM/PM waves | `WeatherEffectsView.tsx`, `WeatherBriefView.tsx`, `WeatherHourTable.tsx`; weather-brief.css | index, weather/<ev>/{latest,brief_latest,final}, explain | OK (6230px) |
| `/weekly-players` | Whole field table with prices and drivers | `WeeklyPlayersView.tsx`, `PlayerEventTable.tsx`; player-events.css | index, player_profiles/fields.json, run model_inputs, profile batches | OK |
| `/players` | Player directory (16,626) and one-golfer deep dive (hero, SG bars, dispersion, style, history, shots) | `PlayerProfilesView.tsx`, `PlayerHero.tsx`, `PlayerDeepDive.tsx`, `PlayerSgBars.tsx`, `PlayerDispersion.tsx`, `PlayingStyle.tsx`, `PlayerPerformanceTrend.tsx`, `PlayerHistoryExplorer.tsx`, `MetricTrendPanel.tsx`; player-profiles/-hero/-deep/-dispersion/-style.css | player_profiles/{index,search-index (Worker-derived),batches,dispersion,fields}.json | OK list; detail only after picking a golfer (not in baseline shots) |
| `/why-priced` | Every player's price, shape, drivers, run-to-run changes | `ExplainViews.tsx` (WhyPricedView), `ExplainBits/Charts`; explain.css | explain/<ev>/*, changes.json | OK |
| `/distributions` | Finish-position probability curves | `DistributionView.tsx`, `distributions-rules.ts`; distributions.css | index, explain run files | OK |
| `/round-scores` | Expected score by round, spread, scoring history by edition | `ScoringView.tsx`, `CourseHistoryPanel.tsx`, `MarketScoringPanel.tsx`; scoring.css, market-scoring.css | index, explain, history/<ev>.json, market_scoring/<ev>/latest (404 for Abu Dhabi, normal) | OK |
| `/model-walkthrough` (owner) | Graphical guide to skill and simulation with this week's verified process | `WalkthroughView.tsx`; walkthrough.css | index, runs/<ev>/*/model_inputs, explain | OK |
| `/inputs` (owner) | Model inputs: skill, course fit, variance, adjustments; override forms | `InputsView.tsx` (1600+ lines), `PlayerMethod.tsx`; globals.css | index, runs/<ev>/*/model_inputs.json, feature_glossary.json, `/api/overrides` (in-memory locally) | OK |
| `/weather-edits` (owner) | Replace forecast hours, then reprice | `WeatherEditsView.tsx`; globals.css | index, weather/<ev>/latest, `/api/weather-overrides` | OK |
| `/shot-data` (owner) | Collector health and publication readiness per tour | `ShotDataView.tsx`, `shot-monitor-*.ts`; shot-monitor.css | owner/shot-monitor/{collectors-desktop,publications}.json | OK (data says 2 tours need attention) |
| `/ops` (owner) | Weekly workflow: delivery chain, planned vs observed runs, agent handoffs | `OpsView.tsx`, `DataFreshnessPanel.tsx`; ops.css, freshness.css | jobs/queue, jobs/heartbeat/*, owner/ops/observations.json (not published: shows "not yet verified"), ops tasks API | Renders; mostly "NOT YET VERIFIED" because observations are absent |
| `/run` (owner) | Start a pricing job, job queue/log, odds signals, data freshness | `RunView.tsx`, `DataFreshnessPanel.tsx`, `OddsSignalsPanel.tsx`; globals.css, freshness.css | jobs/*, owner/data_freshness.json, odds_signals/latest | OK; phone has horizontal scroll (see issues) |
| `/performance` | Model scorecard vs market close, live/paper bets, legacy P&L | `views.tsx` (PerformanceView), `ScorecardSection.tsx`; globals.css | scorecard/latest.json, index, data/* (legacy) | OK |
| `/bet-breakdowns` (owner) | ROI by edge bucket, book, market, with filters | `BetBreakdownsView.tsx`, `bet-breakdowns-rules.ts`; bet-breakdowns.css | owner/bet_breakdowns.json | OK |
| `/bet-card` (owner) | Current suggested bets and lowest price worth taking | `BetCardView.tsx`, `bet-card-rules.ts`; bet-card.css | owner/bet_card.json | OK (short, 1591px) |
| `/history`, `/diagnostics` (archive) | Legacy pipeline tables | `views.tsx`; globals.css | `/api/data/history.json`, `diagnostics.json` | Empty/not-published notice: legacy data is not published in R2 (404 by design; archived, out of scope) |
| `/research` | Retired notice | `ResearchView.tsx` | none | Not in nav, not screenshotted |

## Known baseline issues (visual-check: 72 cases, 46 errors, 40 warnings)
- 30 errors, desktop, every page: header event title ("World Wide Technology Championship - Abu Dhabi A...") overlaps the freshness pill ("Oldest 15 h ago"), 31% overlap. One shell fix clears them all.
- 12 errors, phone: console `[vinext] RSC prefetch setup error: TypeError: f is not a function` (chunk `link-*.js`) on this-week, round-scores, model-walkthrough, run etc. Check whether it also happens on the deployed site.
- `/run` phone: horizontal scroll (scrollWidth 754 > 375) and "course history ..." overlapping "Current".
- Warnings: many tap targets under 40px (term buttons, golfer-name buttons, filter inputs, menu button), pages over 12 screens tall on phone (this-week, weather-effects).
- Infos: 404 `market_scoring/euro_2026142.../latest.json`, `/api/data/history.json`, `/api/data/diagnostics.json` (expected).

## Design system
- **Tokens** (`app/tokens.css`, 173 lines): dark default, light via `prefers-color-scheme` or `data-theme="dark|light"` on `<html>`; both blocks must keep identical token names (`tests/theme.test.mjs` also checks contrast: text >= 4.5:1, chart >= 3:1).
  Surfaces `--background #09101d`, `--surface`, `--surface-raised`, `--surface-soft`, `--popover`; text `--foreground`, `--muted`, `--muted-strong`; borders `--line`, `--line-strong`; tints `--tint-1/2`.
  Accent is user-selectable (Customize sets `--accent-d/-l`); semantic `--positive --negative --warning --model --market`, wave AM/PM, weather series `--wx-*`, 8 colorblind-safe `--chart-1..8`.
  Type: `--font-sans` Inter via system fallback stack (no webfont files or font-face; Inter only if installed), `--font-mono` system mono; scale `--fs-label 11 / small 12 / body 14 / h2 17 / kpi clamp(26,6.4vw,34) / h1 clamp(26,4.6vw,38)`.
  Spacing: only `--gutter 16px` (no spacing scale; pages use ad hoc px). Radius 8/12/16. Shadows `--shadow, --shadow-card, --shadow-lift`. Motion `--ease`, `--dur-fast 120 / --dur 200 / --dur-slow 220`, `--focus-ring`.
- **CSS layout**: `globals.css` (659 lines, base + many view-specific rules for inputs/run/etc.), `explain.css` (401, This week / Why priced), `shapes.css`, `shell.css` (nav/header), `motion.css`; 20+ per-feature files imported by their components (2,312 lines total).
- **Primitives** `app/ui.tsx`: `Badge`, `LiveDot/LivePulse`, `Delta`, `Sparkline`, `DistributionBar`, `AnimatedNumber`, `ScrollTable`, `Skeleton/SkeletonPage`, `FreshnessBadge`, `useNow`, `useReducedMotion`. Tones: positive/negative/warning/model/market/am/pm/accent/neutral.
- **Composites** `app/components.tsx`: `Panel`, `Kpi/KpiTile`, `PageIntro`, `LegacyNotice`, `EmptyState/LoadingState/ErrorState`, `SegmentedControl`, `PlayerPicker`, `DataTable`, `humanTimestamp/humanValue`. Also `ViewBoundary.tsx` (error boundary), `ExplainBits.tsx` (term buttons, glossary popovers), `ExplainCharts.tsx`/`recharts` charts.
- **Observed look**: navy cards on a near-black ground, uppercase blue eyebrow labels, large bold h1, rounded 12-16px panels, pill filters; dense data tables; sidebar nav (Week / Model / Operate / Review / Archive) collapses to a menu button on phone.
- **Inconsistencies worth knowing**: several pages use bare section headings with no `PageIntro` (shot-data, ops differ from others); `InputsView.tsx` is one huge file styled from globals.css; mixed button styles (pill, boxed, link); header text truncates on desktop and phone.
