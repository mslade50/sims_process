// Data loading resilience: timeouts, retries, dedupe, last-good fallback, outage wording, cache policy.
import assert from "node:assert/strict";
import test from "node:test";
import {
  DEFAULT_TIMEOUT_MS, LARGE_TIMEOUT_MS, backoffMs, classifyFailure, createDeduper, fetchCacheMode, fetchWithRetry, formatAge, isImmutableKey,
  isRetryable, loadResource, needsRevalidation, timeoutFor,
} from "../app/data-rules.ts";
import { cacheControlFor, etagMatches, parseIfNoneMatch, respondWithObject } from "../worker/data-http.ts";

const noSleep = async () => {};
const reply = (status, body, headers = {}) => ({ ok: status >= 200 && status < 300, status, headers: new Headers(headers), text: async () => (typeof body === "string" ? body : JSON.stringify(body)) });
/** A fetch that answers from a script, one entry per call; a function entry gets (url, init). */
function scripted(steps) {
  const calls = [];
  const fetchImpl = async (url, init) => {
    calls.push(url);
    const step = steps[Math.min(calls.length - 1, steps.length - 1)];
    return typeof step === "function" ? step(url, init) : step;
  };
  return { fetchImpl, calls };
}
const memoryStore = (initial = {}) => {
  const map = new Map(Object.entries(initial));
  return { map, get: async (u) => map.get(u) ?? null, put: async (u, c) => { map.set(u, c); } };
};
const base = { url: "/api/golfprice/index.json", packagedUrl: "/golfprice/index.json", sleep: noSleep, rand: () => 0.5 };

test("policy: timeouts, retry classes, backoff stays jittered and growing", () => {
  assert.equal(DEFAULT_TIMEOUT_MS, 12_000);
  assert.equal(timeoutFor("golfprice/index.json"), DEFAULT_TIMEOUT_MS);
  assert.equal(timeoutFor("golfprice/player_profiles/index.json"), LARGE_TIMEOUT_MS);
  assert.equal(timeoutFor("golfprice/player_profiles/44ffb7d799d33179b86d/catalog.json"), LARGE_TIMEOUT_MS);
  assert.equal(timeoutFor("golfprice/player_profiles/deep-reference-f8f95f23d09d1ab85459/rounds.json"), LARGE_TIMEOUT_MS);
  assert.equal(isRetryable({ kind: "network" }), true);
  assert.equal(isRetryable({ kind: "response", status: 503 }), true);
  assert.equal(isRetryable({ kind: "response", status: 404 }), false);
  assert.equal(isRetryable({ kind: "response", status: 429 }), false, "network and 5xx only");
  for (const attempt of [0, 1, 2]) {
    const lo = backoffMs(attempt, () => 0), hi = backoffMs(attempt, () => 1);
    assert.ok(lo >= 200 * 2 ** attempt && hi <= 400 * 2 ** attempt && lo < hi);
  }
});

test("cache policy: immutable generation keys, revalidated pointers, background refresh of old mutable entries", () => {
  assert.equal(isImmutableKey("golfprice/player_profiles/44ffb7d799d33179b86d/batch_0015.json"), true);
  assert.equal(isImmutableKey("golfprice/player_profiles/deep-2d19f94ea2e5bf3c14b1/x.json"), true);
  assert.equal(isImmutableKey("golfprice/player_profiles/search-index.json?gen=44ffb7d799d33179b86d"), true);
  assert.equal(isImmutableKey("golfprice/index.json"), false);
  assert.equal(fetchCacheMode("golfprice/index.json"), "no-cache");
  assert.equal(fetchCacheMode("manifest.json"), "no-cache");
  assert.equal(fetchCacheMode("golfprice/player_profiles/search-index.json"), "no-cache");
  assert.equal(fetchCacheMode("golfprice/player_profiles/search-index.json?gen=44ffb7d799d33179b86d"), "default");
  assert.equal(fetchCacheMode("golfprice/runs/e/pre_R0/explain.json"), "default");
  const now = 1_000_000_000;
  assert.equal(needsRevalidation("golfprice/index.json", now - 10_000, false, now), false);
  assert.equal(needsRevalidation("golfprice/index.json", now - 5 * 60_000, false, now), true);
  assert.equal(needsRevalidation("golfprice/player_profiles/44ffb7d799d33179b86d/batch_0001.json", now - 3600_000, false, now), false, "immutable never refreshes");
  assert.equal(needsRevalidation("golfprice/index.json", now - 1000, true, now), false, "a stale entry is not retried on every mount");
  assert.equal(needsRevalidation("golfprice/index.json", now - 20_000, true, now), true);
});

test("a good response is returned and saved as the last good copy", async () => {
  const { fetchImpl, calls } = scripted([reply(200, { events: [1] }, { etag: '"abc"' })]);
  const store = memoryStore();
  const r = await loadResource({ ...base, fetchImpl, store, now: () => 42 });
  assert.equal(r.status, "ok");
  assert.deepEqual(r.json, { events: [1] });
  assert.equal(calls.length, 1);
  assert.deepEqual(store.map.get(base.url), { text: '{"events":[1]}', savedAt: 42, etag: '"abc"' });
});

test("5xx and network failures are retried twice with backoff, then succeed", async () => {
  const sleeps = [];
  const { fetchImpl, calls } = scripted([reply(503, { error: "x" }), () => { throw new TypeError("offline"); }, reply(200, { ok: 1 })]);
  const r = await loadResource({ ...base, fetchImpl, sleep: async (ms) => { sleeps.push(ms); } });
  assert.equal(r.status, "ok");
  assert.equal(calls.length, 3);
  assert.equal(sleeps.length, 2);
  assert.ok(sleeps[1] > sleeps[0], "backoff grows");
});

test("4xx is never retried; a 404 reads as not published, never as an outage", async () => {
  const { fetchImpl, calls } = scripted([reply(404, { ok: false, error: "not published yet" })]);
  const r = await loadResource({ ...base, fetchImpl, store: memoryStore({ [base.url]: { text: "{}", savedAt: 1, etag: null } }) });
  assert.equal(calls.length, 1);
  assert.equal(r.status, "error");
  assert.equal(r.kind, "missing");
  assert.match(r.message, /not been published yet/);
  assert.doesNotMatch(r.message, /outage/);
});

test("a 500 with no last good copy is an outage with its own wording", async () => {
  const { fetchImpl, calls } = scripted([reply(500, { ok: false, error: "internal error" })]);
  const r = await loadResource({ ...base, fetchImpl, store: memoryStore() });
  assert.equal(calls.length >= 3, true, "tried 3 times before giving up");
  assert.equal(r.status, "error");
  assert.equal(r.kind, "outage");
  assert.match(r.message, /unavailable right now/);
  assert.doesNotMatch(r.message, /not been published/);
});

test("when the service fails, the last good copy is served flagged stale with its age", async () => {
  const { fetchImpl } = scripted([reply(500, { ok: false, error: "internal error" })]);
  const store = memoryStore({ [base.url]: { text: '{"events":["last"]}', savedAt: 1234, etag: '"e"' } });
  const r = await loadResource({ ...base, fetchImpl, store });
  assert.equal(r.status, "stale");
  assert.equal(r.stale, true);
  assert.equal(r.savedAt, 1234);
  assert.deepEqual(r.json, { events: ["last"] });
  assert.equal(r.failure.kind, "outage");
});

test("a timeout aborts the request, is retried, then falls back to the last good copy", async () => {
  let aborted = 0;
  const hang = (_url, init) => new Promise((_resolve, reject) => { init.signal.addEventListener("abort", () => { aborted++; reject(new Error("aborted")); }); });
  const { fetchImpl, calls } = scripted([hang]);
  const store = memoryStore({ [base.url]: { text: '{"a":1}', savedAt: 5, etag: null } });
  const r = await loadResource({ ...base, fetchImpl, store, timeoutMs: 15 });
  assert.equal(r.status, "stale");
  assert.equal(aborted >= 3, true);
  assert.equal(calls.length >= 3, true);
  const bare = await fetchWithRetry(base.url, { fetchImpl: scripted([hang]).fetchImpl, timeoutMs: 10, retries: 0, sleep: noSleep });
  assert.equal(bare.kind, "network");
  assert.match(bare.reason, /timed out/);
});

test("no Worker at all (network failure or an HTML body) falls through to the packaged copy, then last good", async () => {
  const urlSwitch = (url) => (url.startsWith("/api/") ? reply(200, "<html>nope</html>") : reply(200, { packaged: true }));
  const a = await loadResource({ ...base, fetchImpl: scripted([urlSwitch]).fetchImpl });
  assert.equal(a.status, "ok");
  assert.equal(a.source, "packaged");
  assert.deepEqual(a.json, { packaged: true });
  const down = () => { throw new TypeError("offline"); };
  const store = memoryStore({ [base.url]: { text: '{"kept":1}', savedAt: 9, etag: null } });
  const b = await loadResource({ ...base, fetchImpl: scripted([down]).fetchImpl, store });
  assert.equal(b.status, "stale");
  const c = await loadResource({ ...base, fetchImpl: scripted([down]).fetchImpl });
  assert.equal(c.status, "error");
  assert.equal(c.kind, "outage");
});

test("a broken or throwing store never breaks a load", async () => {
  const bad = { get: async () => { throw new Error("quota"); }, put: async () => { throw new Error("quota"); } };
  const ok = await loadResource({ ...base, fetchImpl: scripted([reply(200, { v: 1 })]).fetchImpl, store: bad });
  assert.equal(ok.status, "ok");
  const failed = await loadResource({ ...base, fetchImpl: scripted([reply(500, { error: "x" })]).fetchImpl, store: bad });
  assert.equal(failed.status, "error");
  const corrupt = memoryStore({ [base.url]: { text: "{not json", savedAt: 1, etag: null } });
  assert.equal((await loadResource({ ...base, fetchImpl: scripted([reply(500, { error: "x" })]).fetchImpl, store: corrupt })).status, "error");
});

test("concurrent requests for one key share one in-flight load", async () => {
  const deduper = createDeduper();
  let starts = 0;
  const slow = () => deduper.run("k", async () => { starts++; await new Promise((r) => setTimeout(r, 10)); return starts; });
  const results = await Promise.all([slow(), slow(), slow()]);
  assert.deepEqual(results, [1, 1, 1]);
  assert.equal(starts, 1);
  assert.equal(deduper.size(), 0, "settled entries are released");
  await deduper.run("k", async () => { starts++; });
  assert.equal(starts, 2);
  await assert.rejects(deduper.run("e", async () => { throw new Error("boom"); }));
  assert.equal(deduper.size(), 0, "a failed load is not cached");
});

test("wording: outage versus missing versus invalid, and ages", () => {
  assert.equal(classifyFailure(500).kind, "outage");
  assert.equal(classifyFailure(503, "dashboard bucket is not bound").kind, "outage");
  assert.equal(classifyFailure(null).kind, "outage");
  assert.equal(classifyFailure(404, "not published yet").kind, "missing");
  assert.equal(classifyFailure(400, "Invalid data path").kind, "invalid");
  assert.equal(formatAge(10_000), "moments ago");
  assert.equal(formatAge(10 * 60_000), "10 min ago");
  assert.equal(formatAge(5 * 3600_000), "5 h ago");
  assert.equal(formatAge(4 * 86_400_000), "4 days ago");
});

/* ------------------------------------------------------------------ Worker-side HTTP caching */
test("cache-control: immutable for generation keys, short for pointers, 5 minutes otherwise", () => {
  assert.match(cacheControlFor("player_profiles/44ffb7d799d33179b86d/batch_0015.json"), /immutable/);
  assert.match(cacheControlFor("player_profiles/derived/44ffb7d799d33179b86d/search-index.json"), /immutable/);
  assert.match(cacheControlFor("player_profiles/deep-reference-f8f95f23d09d1ab85459/rounds.json"), /immutable/);
  for (const pointer of ["index.json", "manifest.json", "player_profiles/index.json", "player_profiles/dossier-review.json", "weather/e/latest.json", "weather/e/brief_latest.json"]) {
    assert.equal(cacheControlFor(pointer), "no-cache", pointer);
  }
  for (const derived of ["player_profiles/search-index.json", "player_profiles/benchmarks.json"]) assert.equal(cacheControlFor(derived), "public, max-age=15, stale-while-revalidate=120", derived);
  assert.equal(cacheControlFor("runs/e/pre_R0/explain.json"), "public, max-age=300, stale-while-revalidate=3600");
});

test("If-None-Match: weak and quoted tags match, lists and * work", () => {
  assert.deepEqual(parseIfNoneMatch('W/"a", "b"'), ["a", "b"]);
  assert.equal(etagMatches('"abc"', '"abc"'), true);
  assert.equal(etagMatches('W/"abc"', '"abc"'), true);
  assert.equal(etagMatches('"x", "abc"', '"abc"'), true);
  assert.equal(etagMatches("*", '"abc"'), true);
  assert.equal(etagMatches('"zzz"', '"abc"'), false);
  assert.equal(etagMatches(null, '"abc"'), false);
});

test("respondWithObject answers 304 without a body when the tag matches, 200 otherwise", async () => {
  let cancelled = false;
  const object = () => ({ httpEtag: '"v1"', body: new ReadableStream({ start(c) { c.enqueue(new TextEncoder().encode("{}")); c.close(); }, cancel() { cancelled = true; } }), writeHttpMetadata() {} });
  const opts = { contentType: "application/json", cacheControl: "public, max-age=15, stale-while-revalidate=120" };
  const hit = respondWithObject(new Request("https://x/api", { headers: { "if-none-match": '"v1"' } }), object(), opts);
  assert.equal(hit.status, 304);
  assert.equal(hit.headers.get("etag"), '"v1"');
  assert.equal(hit.headers.get("cache-control"), opts.cacheControl);
  assert.equal(await hit.text(), "");
  const miss = respondWithObject(new Request("https://x/api", { headers: { "if-none-match": '"old"' } }), object(), opts);
  assert.equal(miss.status, 200);
  assert.equal(await miss.text(), "{}");
  const head = respondWithObject(new Request("https://x/api", { method: "HEAD" }), object(), opts);
  assert.equal(head.status, 200);
  assert.equal(await head.text(), "");
  void cancelled;
});

test("without a packagedUrl (legacy /api/data) a missing Worker never yields the bundled snapshot", async () => {
  const urlSwitch = (url) => (url.startsWith("/api/") ? reply(200, "<html>nope</html>") : reply(200, { packaged: true }));
  const result = await loadResource({ ...base, packagedUrl: undefined, url: "/api/data/manifest.json", fetchImpl: scripted([urlSwitch]).fetchImpl });
  assert.notEqual(result.status, "ok");
});
