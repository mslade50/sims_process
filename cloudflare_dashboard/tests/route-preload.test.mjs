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

/* ------------------------------------------------------------------ Round 5: first-screen data inlined into the HTML */
import { collectFirstScreen, defaultEvent, explainPaths, firstScreenPage, profileBatchPath, weatherPaths, withFirstScreen, WRAPPER_SCRIPT } from "../worker/first-screen.ts";

const INDEX = { events: [
  { event_uid: "pga:1:2026-11-05", date_start: "2026-11-05", tour: "pga", explain_key: "golfprice/explain/pga_1/pre.json", explain_run: "week_1", distribution_runs: [{ kind: "week", key: "golfprice/explain/pga_1/pre.json", as_of: "2026-11-02", validated: true }] },
  { event_uid: "euro:2:2026-11-05", date_start: "2026-11-05", tour: "euro", explain_key: "golfprice/explain/euro_2/live_R1.json", explain_run: "live_R1", distribution_runs: [{ kind: "live" }, { kind: "week", key: "golfprice/explain/euro_2/wk.json", as_of: "2026-11-02" }] },
] };

test("first screen: pages and the default event mirror the head script", () => {
  assert.equal(firstScreenPage("/"), "this-week");
  assert.equal(firstScreenPage("/Why-Priced/"), "why-priced");
  assert.equal(firstScreenPage("/players"), "players");
  assert.equal(firstScreenPage("/this-week.rsc"), null);
  assert.equal(firstScreenPage("/api/golfprice/index.json"), null);
  assert.equal(defaultEvent(INDEX, null).event_uid, "euro:2:2026-11-05");   // first tour with an explanation, tours sorted: euro before pga
  assert.equal(defaultEvent(INDEX, "pga:1:2026-11-05").event_uid, "pga:1:2026-11-05");
  assert.equal(defaultEvent(INDEX, "nope").event_uid, "euro:2:2026-11-05");
  assert.equal(defaultEvent({}, null), null);
});

test("first screen: explain paths per page (live events read the week run on Why priced)", () => {
  const [pga, euro] = INDEX.events;
  assert.deepEqual(explainPaths("this-week", pga, null), ["/api/golfprice/explain/pga_1/pre.json?slim=1"]);
  assert.deepEqual(explainPaths("why-priced", pga, null), ["/api/golfprice/explain/pga_1/pre.json?slim=1"]);
  assert.deepEqual(explainPaths("this-week", euro, null), ["/api/golfprice/explain/euro_2/live_R1.json?slim=1"]);
  assert.deepEqual(explainPaths("why-priced", euro, null), ["/api/golfprice/explain/euro_2/wk.json?slim=1"]);
  assert.deepEqual(explainPaths("why-priced", euro, "live"), []);
  assert.deepEqual(explainPaths("this-week", { event_uid: "x" }, null), []);
});

test("first screen: profile batch path follows the slice's key table", () => {
  assert.equal(profileBatchPath({ keys: ["abc/batch_0004.json"], key: [0] }), "/api/golfprice/player_profiles/abc/batch_0004.json");
  assert.equal(profileBatchPath({ keys: ["golfprice/player_profiles/abc/b.json"], key: [0] }), "/api/golfprice/player_profiles/abc/b.json");
  assert.equal(profileBatchPath({ keys: ["../x.json"], key: [0] }), null);
  assert.equal(profileBatchPath({}), null);
});

test("first screen: collects index + slim doc; a failed read inlines less, never throws", async () => {
  const calls = [];
  const ok = (body) => new Response(body, { status: 200 });
  const api = async (path) => { calls.push(path); if (path.endsWith("index.json")) return ok(JSON.stringify(INDEX)); if (path.includes("euro_2")) return ok('{"doc":1}'); return null; };
  const got = await collectFirstScreen(new URL("https://x.test/this-week"), api);
  assert.deepEqual(Object.keys(got).sort(), ["/api/golfprice/explain/euro_2/live_R1.json?slim=1", "/api/golfprice/index.json"]);
  const none = await collectFirstScreen(new URL("https://x.test/this-week"), async () => { throw new Error("r2 down"); });
  assert.deepEqual(none, {});
  const slice = { keys: ["g/batch_1.json"], key: [0] };
  const pl = await collectFirstScreen(new URL("https://x.test/players?player=10091"), async (p) => p.endsWith("index.json") ? ok("{}") : p.includes("search-index") ? ok(JSON.stringify(slice)) : ok("{\"p\":1}"));
  assert.deepEqual(Object.keys(pl).sort(), ["/api/golfprice/index.json", "/api/golfprice/player_profiles/g/batch_1.json", "/api/golfprice/player_profiles/golfer/search-index.json?id=10091"]);
  assert.deepEqual(await collectFirstScreen(new URL("https://x.test/players?player=bad<"), api), {});
});

async function run(html, inlined, chunk = 7) {
  const bytes = new TextEncoder().encode(html);
  const stream = new ReadableStream({ start(c) { for (let i = 0; i < bytes.length; i += chunk) c.enqueue(bytes.slice(i, i + chunk)); c.close(); } });
  return await new Response(withFirstScreen(stream, Promise.resolve(inlined))).text();
}

test("first screen: wrapper after <head>, data before </body>, even when markers split across chunks; script-safe JSON", async () => {
  const html = '<!DOCTYPE html><html><head><meta charset="utf-8"/></head><body><p>hi</p></body></html>';
  for (const size of [1, 3, 7, 64, 4096]) {
    const out = await run(html, { "/api/golfprice/index.json": '{"a":"</script><!--"}' }, size);
    assert.ok(out.startsWith("<!DOCTYPE html><html><head>" + WRAPPER_SCRIPT + '<meta charset="utf-8"/>'), `size ${size}`);
    assert.ok(out.endsWith("</body></html>"));
    const data = out.slice(out.indexOf("<script>window.__inlineFirst&&"), out.lastIndexOf("</body>"));
    assert.equal((data.match(/<\/script>/g) ?? []).length, 1, "payload cannot close the script early");
    assert.ok(data.includes("\u003c/script>"));
    assert.equal(out.replace(WRAPPER_SCRIPT, "").replace(data, ""), html);
  }
});

test("first screen: the wrapper serves inlined URLs once and passes everything else through", async () => {
  const calls = [];
  const win = { fetch: (u) => { calls.push(u); return Promise.resolve(new Response("net")); } };
  new Function("window", "setTimeout", WRAPPER_SCRIPT.replace(/^<script>/, "").replace(/<\/script>$/, ""))(win, () => 0);
  const first = win.fetch("/api/golfprice/index.json", {});
  win.__inlineFirst({ "/api/golfprice/index.json": "{\"inline\":1}" });
  assert.equal(await (await first).text(), "{\"inline\":1}");
  assert.equal(await (await win.fetch("/api/golfprice/index.json", {})).text(), "net");     // consumed: later requests use the network
  assert.equal(await (await win.fetch("/api/golfprice/other.json", {})).text(), "net");
  assert.equal(await (await win.fetch("/api/whoami", {})).text(), "net");
  assert.equal(calls.length, 3);
});

test("round 5b: Weather inlines the brief and the weather document under the names the view uses", async () => {
  assert.equal(firstScreenPage("/weather-effects"), "weather");
  const ev = { event_uid: "euro:1:2026-11-05", date_start: "2026-11-05", tour: "euro", explain_key: "golfprice/explain/x/latest.json", weather_brief_key: "golfprice/weather/euro_1_2026-11-05/brief_latest.json" };
  assert.deepEqual(weatherPaths(ev), ["/api/golfprice/weather/euro_1_2026-11-05/brief_latest.json", "/api/golfprice/weather/euro_1_2026-11-05/latest.json"]);
  assert.deepEqual(weatherPaths({ ...ev, weather_brief_key: undefined, weather_key: "golfprice/weather/custom.json" }), ["/api/golfprice/weather/custom.json"]);
  assert.deepEqual(weatherPaths(null), []);
  const served = [];
  const out = await collectFirstScreen(new URL("https://x.test/weather-effects"), async (path) => { served.push(path); return new Response(path.endsWith("index.json") ? JSON.stringify({ events: [ev] }) : "{}", { status: 200 }); });
  assert.deepEqual(Object.keys(out), ["/api/golfprice/index.json", ...weatherPaths(ev)]);
  assert.equal(served.length, 3);
});
