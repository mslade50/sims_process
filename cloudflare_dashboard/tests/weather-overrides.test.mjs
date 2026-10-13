import assert from "node:assert/strict";
import { createSign, generateKeyPairSync } from "node:crypto";
import test from "node:test";
import { BOUNDS, EVENT_SAFE_RE, checkHourFields, chooseRepriceJob, etDate, eventSafe, historyKey, validateWeatherOverride } from "../app/weather-overrides-rules.ts";
import { parseWeather } from "../app/weather-rules.ts";

const T = "2026-10-15T08:00";

test("bounds, rounding and normalization", () => {
  const ok = validateWeatherOverride({ hours: [{ time_local: T, wind: 12.04, gust: 18, temp: null, rain: null, dir: 360 }], note: "  front early " });
  assert.deepEqual(ok.problems, []);
  assert.deepEqual(ok.hours, [{ time_local: T, wind: 12, gust: 18, temp: null, rain: null, dir: 0 }]);
  assert.equal(ok.note, "front early");
  for (const [field, lo, hi] of [["wind", 0, 60], ["gust", 0, 90], ["temp", -10, 120], ["rain", 0, 50], ["dir", 0, 360]]) {
    assert.deepEqual([BOUNDS[field].lo, BOUNDS[field].hi], [lo, hi]);
    assert.deepEqual(validateWeatherOverride({ hours: [{ time_local: T, [field]: lo }] }).problems, [], `${field} low edge`);
    assert.deepEqual(validateWeatherOverride({ hours: [{ time_local: T, [field]: hi }] }).problems, [], `${field} high edge`);
    assert.match(validateWeatherOverride({ hours: [{ time_local: T, [field]: hi + 0.5 }] }).problems.join(), /between/, `${field} above`);
    assert.match(validateWeatherOverride({ hours: [{ time_local: T, [field]: lo - 0.5 }] }).problems.join(), /between/, `${field} below`);
  }
});

test("gusts cannot be below wind in the same hour, either alone is fine", () => {
  assert.match(validateWeatherOverride({ hours: [{ time_local: T, wind: 20, gust: 15 }] }).problems.join(), /Gusts cannot be lower/);
  assert.deepEqual(validateWeatherOverride({ hours: [{ time_local: T, wind: 20, gust: 20 }] }).problems, []);
  assert.deepEqual(validateWeatherOverride({ hours: [{ time_local: T, gust: 3 }] }).problems, []);
  assert.deepEqual(checkHourFields({ wind: "20", gust: "15" }), { gust: "Gusts cannot be lower than the wind." });
  assert.match(checkHourFields({ temp: "warm" }).temp, /must be a number/);
});

test("drops empty hours, sorts, rejects bad times, duplicates, strings and oversize input", () => {
  const out = validateWeatherOverride({ hours: [{ time_local: "2026-10-15T09:00", wind: 5 }, { time_local: T, wind: null, gust: null }, { time_local: "2026-10-15T07:00", rain: 0 }] });
  assert.deepEqual(out.hours.map((h) => h.time_local), ["2026-10-15T07:00", "2026-10-15T09:00"]);
  assert.deepEqual(validateWeatherOverride({ hours: [] }).hours, []);
  assert.deepEqual(validateWeatherOverride({}).problems, []);
  for (const bad of ["2026-10-15T08:30", "2026-10-15T8:00", "2026-02-30T08:00", "2026-10-15T24:00", "", 5]) assert.match(validateWeatherOverride({ hours: [{ time_local: bad, wind: 1 }] }).problems.join(), /time_local/, String(bad));
  assert.match(validateWeatherOverride({ hours: [{ time_local: T, wind: 1 }, { time_local: T, wind: 2 }] }).problems.join(), /more than once/);
  assert.match(validateWeatherOverride({ hours: [{ time_local: T, wind: "12" }] }).problems.join(), /must be a number/);
  assert.ok(validateWeatherOverride({ hours: [{ time_local: T, wind: Infinity }] }).problems.length > 0);
  assert.match(validateWeatherOverride({ hours: "no" }).problems.join(), /list/);
  assert.match(validateWeatherOverride({ hours: [], note: "x".repeat(501) }).problems.join(), /too long/);
  assert.deepEqual(validateWeatherOverride({ hours: [], note: "x".repeat(500) }).problems, []);
  const many = Array.from({ length: 401 }, (_, i) => ({ time_local: `2026-10-${String(1 + (i % 28)).padStart(2, "0")}T${String(i % 24).padStart(2, "0")}:00`, wind: 1 }));
  assert.match(validateWeatherOverride({ hours: many }).problems.join(), /Too many hours/);
});

test("event_safe pattern and history key", () => {
  assert.equal(eventSafe("euro:2026140:2026-10-15"), "euro_2026140_2026-10-15");
  assert.ok(EVENT_SAFE_RE.test("euro_2026140_2026-10-15") && EVENT_SAFE_RE.test("pga_554_2026-10-01"));
  for (const bad of ["../x", "euro_2026140", "Euro_1_2026-10-15", "euro_1_2026-10-15/x", "euro:1:2026-10-15"]) assert.ok(!EVENT_SAFE_RE.test(bad), bad);
  assert.equal(historyKey(Date.parse("2026-10-13T20:00:05.123Z"), "euro_1_2026-10-15"), "overrides/weather/history/20261013T200005Z-euro_1_2026-10-15.json");
});

test("reprice job choice", () => {
  const full = { 1: 1, 2: 0.98 };
  const noon = (d) => Date.parse(`${d}T16:00:00Z`);
  assert.equal(etDate(Date.parse("2026-10-15T02:00:00Z")), "2026-10-14");   // still Wednesday evening in New York
  assert.equal(chooseRepriceJob(full, "2026-10-15", noon("2026-10-13")).type, "wednesday");
  assert.equal(chooseRepriceJob({ 1: 1, 2: 0.97 }, "2026-10-15", noon("2026-10-13")).type, "tuesday");
  assert.equal(chooseRepriceJob({ 1: 1 }, "2026-10-15", noon("2026-10-13")).type, "tuesday");
  assert.equal(chooseRepriceJob({}, undefined, noon("2026-10-13")).type, "tuesday");
  assert.equal(chooseRepriceJob(full, "2026-10-15", Date.parse("2026-10-15T02:00:00Z")).type, "wednesday");   // 22:00 ET the night before
  assert.equal(chooseRepriceJob(full, "2026-10-15", noon("2026-10-15")).type, "after_round");
  assert.equal(chooseRepriceJob({}, "2026-10-15", noon("2026-10-17")).type, "after_round");
  assert.deepEqual(chooseRepriceJob(full, "2026-10-15", noon("2026-10-15")).params, { supersede: true });
});

test("parseWeather reads owner edits, p_wet and tee coverage, and tolerates their absence", () => {
  const doc = parseWeather({
    schema: "golfprice.weather.v1", event_uid: "euro:2026140:2026-10-15", rounds: { 1: { tee_coverage: 1 }, 2: { tee_coverage: null } },
    forecast: { owner_overrides: { count: 2, updated_at: "2026-10-13T20:00:00Z", updated_by: "o@x.com", note: "n" }, hourly: [
      { round: 1, time_local: T, wind: 12, p_wet: 0.03, owner_override: { wind: { forecast: 3.1, override: 12 }, dir: { forecast: 144, override: null } } },
      { round: 1, time_local: "2026-10-15T09:00", wind: 4 },
    ] },
  });
  assert.deepEqual(doc.forecast.owner_overrides, { count: 2, updated_at: "2026-10-13T20:00:00Z", updated_by: "o@x.com", note: "n" });
  assert.deepEqual(doc.forecast.hourly[0].owner_override, { wind: { forecast: 3.1, override: 12 } });
  assert.equal(doc.forecast.hourly[0].p_wet, 0.03);
  assert.equal(doc.forecast.hourly[1].owner_override, null);
  assert.equal(doc.forecast.hourly[1].p_wet, null);
  assert.deepEqual(doc.teeCoverage, { 1: 1, 2: null });
  const old = parseWeather({ schema: "golfprice.weather.v1", forecast: { hourly: [] } });
  assert.equal(old.forecast.owner_overrides, null);
  assert.deepEqual(old.teeCoverage, {});
});

/* ------------------------------------------------------------------ worker */
async function loadWorker() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}-${Math.random()}`);
  return (await import(workerUrl.href)).default;
}

function fakeBucket(initial = {}) {
  const store = new Map(Object.entries(initial).map(([k, v]) => [k, { body: typeof v === "string" ? v : JSON.stringify(v), v: 1 }]));
  const wrap = (key) => {
    const o = store.get(key);
    if (!o) return null;
    return { key, etag: `e${o.v}`, httpEtag: `"e${o.v}"`, body: o.body, text: async () => o.body, json: async () => JSON.parse(o.body), writeHttpMetadata() {} };
  };
  return {
    store,
    get: async (key) => wrap(key),
    put: async function (key, body, options = {}) {
      const existing = store.get(key);
      const cond = options.onlyIf;
      if (cond?.etagMatches && (!existing || `e${existing.v}` !== cond.etagMatches)) return null;
      if (cond?.etagDoesNotMatch === "*" && existing) return null;
      store.set(key, { body: String(body), v: (existing?.v ?? 0) + 1 });
      return { key, etag: `e${store.get(key).v}` };
    },
    list: async ({ prefix = "" } = {}) => ({ objects: [...store.keys()].filter((k) => k.startsWith(prefix)).map((key) => ({ key })), truncated: false }),
  };
}

const b64url = (value) => Buffer.from(JSON.stringify(value)).toString("base64url");
const TEAM = "team.example.cloudflareaccess.com";
const AUD = "aud-123";
const { publicKey, privateKey } = generateKeyPairSync("rsa", { modulusLength: 2048 });
function accessHeaders(email = "owner@example.com") {
  const head = b64url({ alg: "RS256", kid: "k1" });
  const payload = b64url({ email, aud: [AUD], iss: `https://${TEAM}`, exp: Math.floor(Date.now() / 1000) + 3600, sub: "u1" });
  const sig = createSign("RSA-SHA256").update(`${head}.${payload}`).sign(privateKey).toString("base64url");
  return { "cf-access-authenticated-user-email": email, "cf-access-jwt-assertion": `${head}.${payload}.${sig}` };
}
const unsigned = () => ({ "cf-access-authenticated-user-email": "o@x.com", "cf-access-jwt-assertion": `${b64url({ alg: "RS256", kid: "k" })}.${b64url({ email: "o@x.com", exp: Math.floor(Date.now() / 1000) + 3600, sub: "u" })}.c2ln` });

const realFetch = globalThis.fetch;
test.before(() => {
  globalThis.fetch = async (input, init) => {
    const href = typeof input === "string" ? input : input.url;
    if (href === `https://${TEAM}/cdn-cgi/access/certs`) return Response.json({ keys: [{ ...publicKey.export({ format: "jwk" }), kid: "k1" }] });
    return realFetch(input, init);
  };
});
test.after(() => { globalThis.fetch = realFetch; });

const context = { waitUntil() {}, passThroughOnException() {} };
const assets = { fetch: async () => new Response("nf", { status: 404 }) };
async function call(worker, bucket, path, { method = "GET", headers = {}, body } = {}) {
  const init = { method, headers: { ...headers, ...(body !== undefined ? { "content-type": "application/json" } : {}) } };
  if (body !== undefined) init.body = typeof body === "string" ? body : JSON.stringify(body);
  return worker.fetch(new Request(`https://golf.example${path}`, init), { ASSETS: assets, DASHBOARD_DATA: bucket, ACCESS_TEAM_DOMAIN: TEAM, ACCESS_AUD: AUD }, context);
}

const SAFE = "euro_2026140_2026-10-15";
const PATH = `/api/weather-overrides/${SAFE}`;
const KEY = `overrides/weather/${SAFE}.json`;
const put = (worker, bucket, body, headers = accessHeaders()) => call(worker, bucket, PATH, { method: "PUT", headers, body });

test("GET returns null before the first save, then the saved object and its etag", async () => {
  const worker = await loadWorker();
  const bucket = fakeBucket();
  const empty = await (await call(worker, bucket, PATH)).json();
  assert.deepEqual(empty, { ok: true, doc: null, etag: null });
  const saved = await put(worker, bucket, { hours: [{ time_local: T, wind: 12, dir: 360 }], note: "n", etag: null });
  assert.equal(saved.status, 200);
  const body = await saved.json();
  assert.equal(body.doc.schema, "golfprice.weather_override.v1");
  assert.equal(body.doc.event_uid, "euro:2026140:2026-10-15");
  assert.equal(body.doc.updated_by, "owner@example.com");
  assert.match(body.doc.updated_at, /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/);
  assert.deepEqual(body.doc.hours, [{ time_local: T, wind: 12, gust: null, temp: null, rain: null, dir: 0 }]);
  const again = await (await call(worker, bucket, PATH)).json();
  assert.deepEqual(again.doc, body.doc);
  assert.equal(again.etag, body.etag);
});

test("each save writes a history copy first and the main object is etag guarded (409)", async () => {
  const worker = await loadWorker();
  const bucket = fakeBucket();
  const first = await (await put(worker, bucket, { hours: [{ time_local: T, wind: 5 }], etag: null })).json();
  const histKeys = () => [...bucket.store.keys()].filter((k) => k.startsWith("overrides/weather/history/"));
  assert.equal(histKeys().length, 1);
  assert.match(histKeys()[0], new RegExp(`^overrides/weather/history/\\d{8}T\\d{6}Z-${SAFE}\\.json$`));
  assert.deepEqual(JSON.parse(bucket.store.get(histKeys()[0]).body), JSON.parse(bucket.store.get(KEY).body));
  const stale = await put(worker, bucket, { hours: [], etag: null });
  assert.equal(stale.status, 409);
  assert.match((await stale.json()).error, /Reload/);
  assert.equal(JSON.parse(bucket.store.get(KEY).body).hours.length, 1, "a rejected save changes nothing");
  assert.equal(histKeys().length, 1, "a rejected save leaves no history copy");
  await new Promise((r) => setTimeout(r, 1100));   // history keys are per second
  const cleared = await put(worker, bucket, { hours: [], note: "", etag: first.etag });
  assert.equal(cleared.status, 200);
  assert.deepEqual(JSON.parse(bucket.store.get(KEY).body).hours, []);
  assert.equal(histKeys().length, 2);
});

test("writes need a verified Access identity; reads do not; bad input gets a plain message", async () => {
  const worker = await loadWorker();
  const bucket = fakeBucket();
  assert.equal((await put(worker, bucket, { hours: [], etag: null }, {})).status, 401);
  assert.equal((await put(worker, bucket, { hours: [], etag: null }, unsigned())).status, 401);
  assert.equal((await call(worker, bucket, PATH)).status, 200);
  assert.equal(bucket.store.size, 0);
  const bad = await put(worker, bucket, { hours: [{ time_local: T, wind: 99 }], etag: null });
  assert.equal(bad.status, 400);
  const body = await bad.json();
  assert.match(body.error, /Wind must be between 0 and 60/);
  assert.equal((await put(worker, bucket, { hours: [], note: "x".repeat(501), etag: null })).status, 400);
  assert.equal((await put(worker, bucket, { hours: [] })).status, 400, "etag is required");
  assert.equal((await call(worker, bucket, PATH, { method: "PUT", headers: { ...accessHeaders(), origin: "https://evil.example" }, body: { hours: [], etag: null } })).status, 403);
  assert.equal((await call(worker, bucket, PATH, { method: "DELETE", headers: accessHeaders() })).status, 405);
  assert.equal(bucket.store.size, 0);
});

test("event_safe must match the pattern", async () => {
  const worker = await loadWorker();
  const bucket = fakeBucket();
  for (const bad of ["euro_2026140", "..%2Fsecret", "Euro_1_2026-10-15", "euro_1_2026-10-15%2Fx"]) {
    assert.equal((await call(worker, bucket, `/api/weather-overrides/${bad}`)).status, 400, bad);
  }
  assert.equal((await put(worker, bucket, { hours: [], etag: null }, accessHeaders())).status, 200);
});
