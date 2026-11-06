import assert from "node:assert/strict";
import test from "node:test";
import { injectBeforeHeadEnd, linkHeaderFor, routeKeyFor, stylesheetTagsFor, withViewPreload } from "../worker/route-preload.ts";
import { buildRouteChunks } from "../scripts/route-chunks.mjs";

test("route keys: root is This week; nested, dotted and API paths get no hint", () => {
  assert.equal(routeKeyFor("/"), "this-week");
  assert.equal(routeKeyFor("/Players/"), "players");
  assert.equal(routeKeyFor("/api/golfprice/index.json"), null);
  assert.equal(routeKeyFor("/_next/static/x.js"), null);
});

test("link header: modulepreload for chunks only; css becomes head tags; nothing unsafe", () => {
  const files = ["/_next/static/chunks/A-1.js", "/_next/static/css/A.2.css", "/evil/x.js", "/_next/static/chunks/../x.js"];
  assert.equal(linkHeaderFor(files), "</_next/static/chunks/A-1.js>; rel=modulepreload; crossorigin");
  assert.equal(stylesheetTagsFor(files), '<link rel="stylesheet" href="/_next/static/css/A.2.css">');
  assert.equal(linkHeaderFor(undefined), null);
});

test("buildRouteChunks keeps view chunks, drops shell chunks, adds css", () => {
  const manifest = {
    "virtual:vinext-app-browser-entry": { file: "_next/static/chunks/index.js", imports: ["_rt.js"] },
    "_rt.js": { file: "_next/static/chunks/rt.js" },
    "app/DashboardApp.tsx": { file: "_next/static/chunks/DashboardApp.js", imports: ["_rt.js"] },
    "app/V.tsx": { file: "_next/static/chunks/V-1.js", imports: ["_rt.js", "_dep.js"], css: ["_next/static/css/V.css"] },
    "_dep.js": { file: "_next/static/chunks/dep.js" },
  };
  const r = buildRouteChunks(manifest, 'const loaders: Record<string, () => Promise<unknown>> = {\n  "v": () => import("./V"),\n};\n');
  assert.deepEqual(r.v, ["/_next/static/chunks/V-1.js", "/_next/static/chunks/dep.js", "/_next/static/css/V.css"]);
});

test("withViewPreload is additive and leaves non-HTML and missing manifests alone", async () => {
  const assets = { fetch: async () => Response.json({ "this-week": ["/_next/static/chunks/A-1.js"] }) };
  const html = new Response("<html>", { headers: { "content-type": "text/html; charset=utf-8" } });
  const out = await withViewPreload(new Request("https://x.test/this-week"), html, assets);
  assert.match(out.headers.get("link") ?? "", /A-1\.js/);
  assert.equal(await out.text(), "<html>");
  const json = new Response("{}", { headers: { "content-type": "application/json" } });
  assert.equal((await withViewPreload(new Request("https://x.test/this-week"), json, assets)).headers.get("link"), null);
});

test("injectBeforeHeadEnd inserts once, even when </head> is split across chunks", async () => {
  const enc = new TextEncoder();
  const body = new ReadableStream({ start(c) { c.enqueue(enc.encode("<html><head><title>x</title></he")); c.enqueue(enc.encode("ad><body>b</body></html>")); c.close(); } });
  const out = await new Response(injectBeforeHeadEnd(body, "<link rel=\"stylesheet\" href=\"/a.css\">")).text();
  assert.equal(out, '<html><head><title>x</title><link rel="stylesheet" href="/a.css"></head><body>b</body></html>');
});
