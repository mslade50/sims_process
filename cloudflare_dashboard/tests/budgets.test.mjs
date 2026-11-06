// Bundle budgets. Needs a finished build in dist/ (npm test builds first). Numbers come from scripts/measure-bundle.mjs.
import assert from "node:assert/strict";
import test from "node:test";
import { measureAllCss, measureChunks, measureRoute } from "../scripts/measure-bundle.mjs";

const KB = 1024;
const SHELL_JS_GZIP_BUDGET = 148 * KB; // 148 (r3-shell 2026-11-06): measured 146.4 KB gz; next/link left the shell (full-page navigation everywhere) which paid for the early-fetch hook, stale-chunk recovery and tab-mode OnThisPage. Previously 150: 150 (ratchet, r3-perf 2026-11-05): measured 148.6 KB gz after the golfer search, Customize dialog and not-found notices went lazy and the shell stopped importing components.tsx. Was 160 (round-2 shell, 157.4); original budget 150
const LAZY_CHUNK_GZIP_BUDGET = 250 * KB;
const TOTAL_CSS_GZIP_BUDGET = 66 * KB; // 66 (r3 2026-11-06): measured 65.1 KB gz with six agents adding round-3 styles (shell part: tab strip states, disclosure motion, font fallback, archived status ~0.8 KB). Was 64: 64 (ratchet, r3-perf 2026-11-05): measured 63.4 KB gz after removing ~190 dead rules and ~6 KB of shadowed declarations (was 68 / 67.1). The original 60 is not reachable without dropping design-system v2 styling: what is left is live or conditional UI (menus, empty/error states) plus the Tailwind layer
// Ratchet for shell + the This week view chunks together. Lower it whenever the view's heavy imports (Recharts) go lazy.
const THIS_WEEK_WITH_VIEW_RATCHET = 201.5 * KB; // 201.5 (R4b 2026-11-06): measured 201.05 KB gz; the slim explain document (early diff request in the head script, drawer host, idle full-document prefetch) cost about 0.9 KB and bought about 47 KB less on the wire for the first screen. Was 201: 201 (r3 2026-11-06): measured 200.3 KB gz; the shell shrank 2.2 KB but next/link (4.7 KB) now loads with the view instead of the shell. Was 198: 198 (r3-perf 2026-11-05): measured 196.8 KB gz after Why priced and the player drawer left the This week chunk and the weather parsers went lazy (was 220 / 216.4). 205 was the original ratchet. Earlier: 195.6 KB gz on 2026-10-14 after lazy Recharts; was 432 KB
const THIS_WEEK_WITH_VIEW_TARGET = 150 * KB;

const root = await measureRoute("/");
const { chunks, duplicates } = await measureChunks();

test("the first route's initial JS (framework + shell) is within 148 KB gzip", () => {
  assert.ok(root.jsGzip <= SHELL_JS_GZIP_BUDGET, `/ ships ${(root.jsGzip / KB).toFixed(1)} KB gzip of initial JS (budget 148 KB)`);
  assert.deepEqual(root.missing, [], "every asset the HTML references exists in dist/client");
});

test("the initial JS of the first route does not pull in a view or Recharts", () => {
  const heavy = root.jsFiles.filter((f) => /(?:^|\/)(?:CartesianChart|PlayerProfilesView|InputsView|ScoringView|ExplainViews|views|WeatherEffectsView|WeatherEditsView|RunView|BetBreakdownsView)-/.test(f.path));
  assert.deepEqual(heavy.map((f) => f.path), [], "views and chart libraries load lazily");
});

test("no single built JS chunk exceeds 250 KB gzip", () => {
  const big = chunks.filter((c) => c.gzip > LAZY_CHUNK_GZIP_BUDGET);
  assert.deepEqual(big.map((c) => `${c.file} ${(c.gzip / KB).toFixed(1)} KB`), []);
});

test("total CSS stays within 66 KB gzip", async () => {
  const css = await measureAllCss();
  const total = css.reduce((a, f) => a + f.gzip, 0);
  assert.ok(total <= TOTAL_CSS_GZIP_BUDGET, `CSS is ${(total / KB).toFixed(1)} KB gzip (budget 66 KB)`);
});

test("dist has no stale duplicate chunks (same chunk name under two hashes)", () => {
  assert.deepEqual(duplicates, []);
});

test("each other route's lazy view adds a bounded amount on top of the shell", async () => {
  for (const route of ["/weekly-players", "/players", "/inputs", "/run", "/performance"]) {
    const r = await measureRoute(route);
    assert.ok(r.withViewGzip < 450 * KB, `${route} loads ${(r.withViewGzip / KB).toFixed(1)} KB gzip including its view`);
    assert.ok(r.withViewGzip > r.jsGzip || route === "/run", `${route} has its own view chunk`);
  }
});

test("This week with its view chunks stays under the ratchet", () => {
  assert.ok(root.withViewGzip <= THIS_WEEK_WITH_VIEW_RATCHET, `/ with view is ${(root.withViewGzip / KB).toFixed(1)} KB gzip (ratchet ${THIS_WEEK_WITH_VIEW_RATCHET / KB} KB)`);
});

test("This week including its view reaches the 150 KB target", { todo: "196.8 KB after r3-perf; the rest is ExplainViews 19.6 + explain-rules 14.9 + glossary 4.6 + shell 148.6. Needs a shorter shell (framework + vinext client are 105 KB of it) or splitting explain-rules" }, () => {
  assert.ok(root.withViewGzip <= THIS_WEEK_WITH_VIEW_TARGET, `/ with view is ${(root.withViewGzip / KB).toFixed(1)} KB gzip`);
});
