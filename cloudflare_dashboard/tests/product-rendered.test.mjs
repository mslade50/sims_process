import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function loadWorker() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}-${Math.random()}`);
  return (await import(workerUrl.href)).default;
}

async function render(path = "/") {
  const worker = await loadWorker();

  return worker.fetch(
    new Request(`https://golf.example${path}`, {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async (request) => {
          const pathname = new URL(request.url).pathname;
          if (pathname === "/data/manifest.json") {
            return Response.json({
              tournament: "BMW Championship",
              exported_at: "2026-08-18T10:00:00Z",
            });
          }
          return new Response("Not found", { status: 404 });
        },
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the Golf Model application shell", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>Golf Model<\/title>/i);
  assert.match(html, /Golf Model/);
  assert.match(html, /Performance/i);
  assert.match(html, /Finish distributions/i);
  assert.match(html, /Performance/);
  assert.doesNotMatch(html, /Your site is taking shape/);
  assert.doesNotMatch(html, />Home</);
  assert.doesNotMatch(html, />Matchups</);
  assert.doesNotMatch(html, />Bets</);
  assert.doesNotMatch(html, />Pricer</);
  assert.doesNotMatch(html, /Outrights/);
});

test("publishes absolute social metadata and supports retained routes", async () => {
  const [response, layout, packageJson] = await Promise.all([
    render("/performance"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
  ]);
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /https?:\/\/[^"<]+\/og\.png/);
  assert.match(html, /Performance/);
  assert.match(layout, /x-forwarded-host/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
});

test("forces full-page navigation for Cloudflare route compatibility", async () => {
  const dashboardApp = await readFile(new URL("../app/DashboardApp.tsx", import.meta.url), "utf8");
  assert.match(dashboardApp, /window\.location\.assign\(href\)/);
  assert.match(dashboardApp, /navigateWithReload\(event, `\/\$\{item\.key\}`\)/);
  assert.match(dashboardApp, /href="\/performance"/);
});

test("restores the legacy Performance analysis controls and default exclusions", async () => {
  const performanceView = await readFile(new URL("../app/views.tsx", import.meta.url), "utf8");
  for (const expected of [
    "Excluded by default",
    "Hidden until selected",
    "finish_position_live",
    "score_bet",
    "Kalshi / NoVig",
    "Analysis mode",
    "Kelly % edge",
    "Raw % edge",
    "Archetype against",
    "ROI by bucket",
    "Event summary",
    "Filtered bets",
  ]) assert.match(performanceView, new RegExp(expected.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "i"));
});

test("serves dashboard data from R2 and falls back to packaged assets", async () => {
  const worker = await loadWorker();
  const context = { waitUntil() {}, passThroughOnException() {} };
  const assetPaths = [];
  const assetResponse = await worker.fetch(
    new Request("https://golf.example/api/data/manifest.json"),
    {
      ASSETS: {
        fetch: async (request) => {
          assetPaths.push(new URL(request.url).pathname);
          return Response.json({ source: "asset" });
        },
      },
    },
    context,
  );
  assert.equal(assetResponse.status, 200);
  assert.deepEqual(await assetResponse.json(), { source: "asset" });
  assert.deepEqual(assetPaths, ["/data/manifest.json"]);

  const r2Response = await worker.fetch(
    new Request("https://golf.example/api/data/manifest.json"),
    {
      ASSETS: { fetch: async () => new Response("unexpected", { status: 500 }) },
      DASHBOARD_DATA: {
        get: async (key) => ({
          body: JSON.stringify({ source: "r2", key }),
          httpEtag: '"etag"',
          writeHttpMetadata() {},
        }),
      },
    },
    context,
  );
  assert.equal(r2Response.status, 200);
  assert.deepEqual(await r2Response.json(), { source: "r2", key: "data/manifest.json" });
});
