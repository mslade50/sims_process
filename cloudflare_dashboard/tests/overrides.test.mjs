import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { ALLOWED, CUT_BOUNDS, buildRecord, checkOverride } from "../app/overrides-rules.ts";

const NOW = Date.parse("2026-10-05T14:00:00Z");
const DAY = 86_400_000;

async function loadWorker() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}-${Math.random()}`);
  return (await import(workerUrl.href)).default;
}

/** In-memory R2 with the subset the Worker uses: get, put (onlyIf), list. */
function fakeBucket(initial = {}) {
  const store = new Map(Object.entries(initial).map(([k, v]) => [k, { body: typeof v === "string" ? v : JSON.stringify(v), v: 1 }]));
  const wrap = (key) => {
    const o = store.get(key);
    if (!o) return null;
    return {
      key,
      etag: `e${o.v}`,
      httpEtag: `"e${o.v}"`,
      body: o.body,
      text: async () => o.body,
      json: async () => JSON.parse(o.body),
      writeHttpMetadata() {},
    };
  };
  return {
    store,
    get: async (key) => wrap(key),
    put: async (key, body, options = {}) => {
      const existing = store.get(key);
      const cond = options.onlyIf;
      if (cond?.etagMatches && (!existing || `e${existing.v}` !== cond.etagMatches)) return null;
      if (cond?.etagDoesNotMatch === "*" && existing) return null;
      store.set(key, { body: String(body), v: (existing?.v ?? 0) + 1 });
      return { key };
    },
    list: async ({ prefix = "" } = {}) => ({ objects: [...store.keys()].filter((k) => k.startsWith(prefix)).map((key) => ({ key })), truncated: false }),
  };
}

function b64url(value) {
  return Buffer.from(JSON.stringify(value)).toString("base64url");
}

function accessHeaders(email = "owner@example.com", exp = Math.floor(Date.now() / 1000) + 3600) {
  const jwt = `${b64url({ alg: "RS256", kid: "k" })}.${b64url({ email, exp, sub: "u1" })}.c2ln`;
  return { "cf-access-authenticated-user-email": email, "cf-access-jwt-assertion": jwt };
}

const context = { waitUntil() {}, passThroughOnException() {} };
const assets = { fetch: async () => new Response("nf", { status: 404 }) };

async function call(worker, bucket, path, { method = "GET", headers = {}, body } = {}) {
  const init = { method, headers: { ...headers, ...(body !== undefined ? { "content-type": "application/json" } : {}) } };
  if (body !== undefined) init.body = typeof body === "string" ? body : JSON.stringify(body);
  return worker.fetch(new Request(`https://golf.example${path}`, init), { ASSETS: assets, DASHBOARD_DATA: bucket }, context);
}

const goodBody = (extra = {}) => ({
  event: "pga:554:2026-10-01",
  scope: "player",
  target: 32652,
  field: "skill_delta",
  value: -0.25,
  reason: "reported back injury",
  expires_at: new Date(Date.now() + 3 * DAY).toISOString(),
  ...extra,
});

test("compiled-in bounds equal the published golfprice overrides schema", async () => {
  const schema = JSON.parse(await readFile(new URL("./fixtures/overrides_schema.json", import.meta.url), "utf8"));
  assert.equal(schema.fields.length, ALLOWED.length);
  for (const spec of ALLOWED) {
    const published = schema.fields.find((f) => f.scope === spec.scope && f.field === spec.field);
    assert.ok(published, `${spec.scope}/${spec.field} is published`);
    assert.equal(published.kind, spec.kind);
    if (spec.kind === "number") assert.deepEqual([published.lo, published.hi], [spec.lo, spec.hi]);
    if (spec.kind === "cut_rule") assert.deepEqual(published.keys, Object.fromEntries(Object.entries(CUT_BOUNDS)));
  }
});

test("hard bounds reject and never clip", () => {
  const base = (over) => ({ id: "x", created_at: "2026-10-05T14:00:00Z", author: "a", event: "all", reason: "because", expires_at: "2026-10-08T14:00:00Z", ...over });
  const player = (field, value) => base({ scope: "player", target: 1, field, value });
  assert.deepEqual(checkOverride(player("skill_delta", 1.0)), []);
  assert.deepEqual(checkOverride(player("skill_delta", -1.0)), []);
  assert.match(checkOverride(player("skill_delta", 1.01)).join(), /outside the hard bounds/);
  assert.match(checkOverride(player("sd_mult", 0.69)).join(), /outside/);
  assert.match(checkOverride(player("sd_mult", 1.51)).join(), /outside/);
  assert.match(checkOverride(player("withdraw", false)).join(), /must be true/);
  assert.match(checkOverride(base({ scope: "course", field: "scoring_avg_delta", value: 3.5 })).join(), /outside/);
  assert.match(checkOverride(base({ scope: "course", field: "sd_mult", value: 1.3 })).join(), /outside/);
  assert.deepEqual(checkOverride(base({ scope: "course", field: "sd_mult", value: 1.25 })), []);
  assert.match(checkOverride(base({ scope: "engine", field: "cut_rule", value: { cut_round: 1 } })).join(), /cut_round must be 0/);
  assert.match(checkOverride(base({ scope: "engine", field: "cut_rule", value: { top_n: 157 } })).join(), /integer in/);
  assert.match(checkOverride(base({ scope: "engine", field: "cut_rule", value: { bogus: 1 } })).join(), /not allowed/);
  assert.deepEqual(checkOverride(base({ scope: "engine", field: "cut_rule", value: { cut_round: 2, top_n: 65, within: 10 } })), []);
  assert.match(checkOverride(base({ scope: "player", field: "skill_delta", value: 0.1 })).join(), /integer target/);
  assert.match(checkOverride(base({ scope: "course", field: "sd_mult", value: 1, target: 5 })).join(), /takes no target/);
  assert.match(checkOverride(player("skill_delta", 0.1) && { ...player("skill_delta", 0.1), reason: "no" }).join(), /at least 5/);
  assert.match(checkOverride({ ...player("skill_delta", 0.1), expires_at: "2026-11-30T00:00:00Z" }).join(), /more than 28 days/);
  assert.match(checkOverride({ ...player("skill_delta", 0.1), expires_at: "2026-10-05T14:00:00Z" }).join(), /after created_at/);
  assert.match(checkOverride({ ...player("skill_delta", 0.1), scope: "player", field: "mystery" }).join(), /not allowed/);
});

test("the server owns id, created_at and author", () => {
  const built = buildRecord(goodBody(), "owner@example.com", NOW);
  assert.ok(built.record, built.problems.join());
  assert.equal(built.record.author, "owner@example.com");
  assert.equal(built.record.created_at, "2026-10-05T14:00:00Z");
  assert.match(built.record.id, /^ov-20261005T140000-[0-9a-f]{4}$/);
  assert.match(buildRecord({ ...goodBody(), author: "someone else" }, "owner@example.com", NOW).problems.join(), /set by the server/);
});

test("override writes require a Cloudflare Access identity", async () => {
  const worker = await loadWorker();
  const bucket = fakeBucket();
  for (const headers of [{}, { "cf-access-authenticated-user-email": "owner@example.com" }, { "cf-access-jwt-assertion": "a.b.c" }, { ...accessHeaders(), "cf-access-authenticated-user-email": "other@example.com" }, accessHeaders("owner@example.com", 1)]) {
    const response = await call(worker, bucket, "/api/overrides", { method: "POST", headers, body: goodBody() });
    assert.equal(response.status, 401, JSON.stringify(headers).slice(0, 60));
  }
  const del = await call(worker, bucket, "/api/overrides/ov-x", { method: "DELETE" });
  assert.equal(del.status, 401);
  assert.equal(bucket.store.size, 0, "nothing was written");
});

test("create validates bounds, writes active.json and an audit entry with the Access user", async () => {
  const worker = await loadWorker();
  const bucket = fakeBucket();
  const bad = await call(worker, bucket, "/api/overrides", { method: "POST", headers: accessHeaders(), body: goodBody({ value: 5 }) });
  assert.equal(bad.status, 422);
  assert.match((await bad.json()).problems.join(), /outside the hard bounds/);
  assert.equal(bucket.store.size, 0);

  const crossOrigin = await call(worker, bucket, "/api/overrides", { method: "POST", headers: { ...accessHeaders(), origin: "https://evil.example" }, body: goodBody() });
  assert.equal(crossOrigin.status, 403);

  const ok = await call(worker, bucket, "/api/overrides", { method: "POST", headers: accessHeaders("Owner@Example.com"), body: goodBody() });
  assert.equal(ok.status, 201);
  const created = (await ok.json()).record;
  assert.equal(created.author, "owner@example.com");
  const active = JSON.parse(bucket.store.get("overrides/active.json").body);
  assert.equal(active.length, 1);
  assert.equal(active[0].id, created.id);
  const historyKeys = [...bucket.store.keys()].filter((k) => k.startsWith("overrides/history/"));
  assert.equal(historyKeys.length, 1);
  const entry = JSON.parse(bucket.store.get(historyKeys[0]).body);
  assert.equal(entry.action, "create");
  assert.equal(entry.by.email, "owner@example.com");
  assert.equal(entry.by.verified, false);

  const list = await (await call(worker, bucket, "/api/overrides")).json();
  assert.equal(list.records.length, 1);
  const hist = await (await call(worker, bucket, "/api/overrides/history")).json();
  assert.equal(hist.entries.length, 1);
});

test("expire keeps the record, remove deletes it, both are logged", async () => {
  const worker = await loadWorker();
  const bucket = fakeBucket();
  const headers = accessHeaders();
  const a = (await (await call(worker, bucket, "/api/overrides", { method: "POST", headers, body: goodBody() })).json()).record;
  const b = (await (await call(worker, bucket, "/api/overrides", { method: "POST", headers, body: goodBody({ target: 1, value: 0.1 }) })).json()).record;
  const expired = await call(worker, bucket, `/api/overrides/${a.id}?mode=expire`, { method: "DELETE", headers });
  assert.equal(expired.status, 200);
  let active = JSON.parse(bucket.store.get("overrides/active.json").body);
  assert.equal(active.length, 2);
  assert.ok(Date.parse(active.find((r) => r.id === a.id).expires_at) <= Date.now() + 1500);
  await new Promise((resolve) => setTimeout(resolve, 1200));
  assert.equal((await call(worker, bucket, `/api/overrides/${a.id}?mode=expire`, { method: "DELETE", headers })).status, 409);
  assert.equal((await call(worker, bucket, `/api/overrides/${b.id}`, { method: "DELETE", headers })).status, 200);
  active = JSON.parse(bucket.store.get("overrides/active.json").body);
  assert.deepEqual(active.map((r) => r.id), [a.id]);
  assert.equal((await call(worker, bucket, "/api/overrides/nope", { method: "DELETE", headers })).status, 404);
  const actions = [...bucket.store.keys()].filter((k) => k.startsWith("overrides/history/")).map((k) => JSON.parse(bucket.store.get(k).body).action).sort();
  assert.deepEqual(actions, ["create", "create", "expire", "remove"]);
});

test("a corrupt active.json is never overwritten", async () => {
  const worker = await loadWorker();
  const bucket = fakeBucket({ "overrides/active.json": "{not json" });
  const response = await call(worker, bucket, "/api/overrides", { method: "POST", headers: accessHeaders(), body: goodBody() });
  assert.equal(response.status, 500);
  assert.equal(bucket.store.get("overrides/active.json").body, "{not json");
});

test("golfprice objects are readable but the path cannot escape the prefix", async () => {
  const worker = await loadWorker();
  const bucket = fakeBucket({ "golfprice/index.json": { events: [] }, "data/manifest.json": { secret: true } });
  const ok = await call(worker, bucket, "/api/golfprice/index.json");
  assert.equal(ok.status, 200);
  assert.deepEqual(await ok.json(), { events: [] });
  assert.equal((await call(worker, bucket, "/api/golfprice/a%2F..%2Fdata%2Fmanifest.json")).status, 400);
  assert.equal((await call(worker, bucket, "/api/golfprice/runs/missing/model_inputs.json")).status, 404);
  assert.equal((await call(worker, bucket, "/api/golfprice/index.json", { method: "POST", headers: accessHeaders(), body: {} })).status, 405);
});

test("serves the Model inputs route", async () => {
  const worker = await loadWorker();
  const response = await worker.fetch(new Request("https://golf.example/inputs", { headers: { accept: "text/html" } }), { ASSETS: assets }, context);
  assert.equal(response.status, 200);
  assert.match(await response.text(), /Model inputs/);
});
