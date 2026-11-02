// Bundle budgets. Needs a finished build in dist/ (npm test builds first). Numbers come from scripts/measure-bundle.mjs.
import assert from "node:assert/strict";
import test from "node:test";
import { measureAllCss, measureChunks, measureRoute } from "../scripts/measure-bundle.mjs";

const KB = 1024;
const SHELL_JS_GZIP_BUDGET = 150 * KB; // framework + shell, the JS in the first HTML, before any view chunk
const LAZY_CHUNK_GZIP_BUDGET = 250 * KB;
const TOTAL_CSS_GZIP_BUDGET = 60 * KB;
// Ratchet for shell + the This week view chunks together. Lower it whenever the view's heavy imports (Recharts) go lazy.
const THIS_WEEK_WITH_VIEW_RATCHET = 206 * KB; // measured 195.6 KB gz on 2026-10-14 after lazy Recharts; was 432 KB. 206: the Bet card nav entry (2026-10-22) added ~0.1 KB at 205.0
const THIS_WEEK_WITH_VIEW_TARGET = 150 * KB;

const root = await measureRoute("/");
const { chunks, duplicates } = await measureChunks();

test("the first route's initial JS (framework + shell) is within 150 KB gzip", () => {
  assert.ok(root.jsGzip <= SHELL_JS_GZIP_BUDGET, `/ ships ${(root.jsGzip / KB).toFixed(1)} KB gzip of initial JS (budget 150 KB)`);
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

test("total CSS stays within 60 KB gzip", async () => {
  const css = await measureAllCss();
  const total = css.reduce((a, f) => a + f.gzip, 0);
  assert.ok(total <= TOTAL_CSS_GZIP_BUDGET, `CSS is ${(total / KB).toFixed(1)} KB gzip (budget 60 KB)`);
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

test("This week including its view reaches the 150 KB target", { todo: "needs ExplainViews to lazy-load Recharts, ScoringView, WeatherBriefView and CourseGeometry" }, () => {
  assert.ok(root.withViewGzip <= THIS_WEEK_WITH_VIEW_TARGET, `/ with view is ${(root.withViewGzip / KB).toFixed(1)} KB gzip`);
});
