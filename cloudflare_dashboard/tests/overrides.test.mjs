import assert from "node:assert/strict";
import { createSign, generateKeyPairSync } from "node:crypto";
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
    /** Fault injection: (key, body) => "throw" | "null" | undefined, consulted before every put. */
    fault: null,
    get: async (key) => wrap(key),
    put: async function (key, body, options = {}) {
      const injected = this.fault?.(key, String(body));
      if (injected === "throw") throw new Error(`injected put failure for ${key}`);
      if (injected === "null") return null;
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

function b64url(value) {
  return Buffer.from(JSON.stringify(value)).toString("base64url");
}

const TEAM = "team.example.cloudflareaccess.com";
const AUD = "aud-123";
const { publicKey, privateKey } = generateKeyPairSync("rsa", { modulusLength: 2048 });
const { privateKey: rotatedPrivateKey, publicKey: rotatedPublicKey } = generateKeyPairSync("rsa", { modulusLength: 2048 });
const certState = { keys: [{ ...publicKey.export({ format: "jwk" }), kid: "k1" }], fetches: 0 };

/** A correctly signed Access assertion (verified identity once the Worker has ACCESS_TEAM_DOMAIN and ACCESS_AUD). */
function accessHeaders(email = "owner@example.com", exp = Math.floor(Date.now() / 1000) + 3600, { kid = "k1", key = privateKey } = {}) {
  const head = b64url({ alg: "RS256", kid });
  const payload = b64url({ email, aud: [AUD], iss: `https://${TEAM}`, exp, sub: "u1" });
  const sig = createSign("RSA-SHA256").update(`${head}.${payload}`).sign(key).toString("base64url");
  return { "cf-access-authenticated-user-email": email, "cf-access-jwt-assertion": `${head}.${payload}.${sig}` };
}

/** Well-formed but unsigned assertion: the weaker mode that must no longer be able to write overrides. */
function unsignedHeaders(email = "owner@example.com") {
  const jwt = `${b64url({ alg: "RS256", kid: "k" })}.${b64url({ email, exp: Math.floor(Date.now() / 1000) + 3600, sub: "u1" })}.c2ln`;
  return { "cf-access-authenticated-user-email": email, "cf-access-jwt-assertion": jwt };
}

const realFetch = globalThis.fetch;
test.before(() => {
  globalThis.fetch = async (input, init) => {
    const href = typeof input === "string" ? input : input.url;
    if (href === `https://${TEAM}/cdn-cgi/access/certs`) {
      certState.fetches += 1;
      return Response.json({ keys: certState.keys });
    }
    return realFetch(input, init);
  };
});
test.after(() => {
  globalThis.fetch = realFetch;
});

const context = { waitUntil() {}, passThroughOnException() {} };
const assets = { fetch: async () => new Response("nf", { status: 404 }) };

async function call(worker, bucket, path, { method = "GET", headers = {}, body, access = true } = {}) {
  const init = { method, headers: { ...headers, ...(body !== undefined ? { "content-type": "application/json" } : {}) } };
  if (body !== undefined) init.body = typeof body === "string" ? body : JSON.stringify(body);
  const env = { ASSETS: assets, DASHBOARD_DATA: bucket, ...(access ? { ACCESS_TEAM_DOMAIN: TEAM, ACCESS_AUD: AUD } : {}) };
  return worker.fetch(new Request(`https://golf.example${path}`, init), env, context);
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

test("E4: an unverified identity (no Worker Access config, or an unsigned assertion) cannot write overrides", async () => {
  const worker = await loadWorker();
  const bucket = fakeBucket();
  const noConfig = await call(worker, bucket, "/api/overrides", { method: "POST", headers: unsignedHeaders(), body: goodBody(), access: false });
  assert.equal(noConfig.status, 403);
  assert.match((await noConfig.json()).error, /could not be verified/);
  const forged = await call(worker, bucket, "/api/overrides", { method: "POST", headers: unsignedHeaders(), body: goodBody() });
  assert.equal(forged.status, 401, "with Access configured an unsigned assertion fails signature verification");
  const del = await call(worker, bucket, "/api/overrides/ov-x", { method: "DELETE", headers: unsignedHeaders(), access: false });
  assert.equal(del.status, 403);
  assert.equal(bucket.store.size, 0, "nothing was written");
  assert.equal((await call(worker, bucket, "/api/overrides")).status, 200, "reads stay open");
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
  assert.equal(entry.by.verified, true);

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

const historyKeys = (bucket) => [...bucket.store.keys()].filter((k) => k.startsWith("overrides/history/"));
const activeIds = (bucket) => JSON.parse(bucket.store.get("overrides/active.json")?.body ?? "[]").map((r) => r.id);

test("E4: history key carries a random suffix, so same-millisecond writes cannot collide", async () => {
  const worker = await loadWorker();
  const bucket = fakeBucket();
  const headers = accessHeaders();
  await call(worker, bucket, "/api/overrides", { method: "POST", headers, body: goodBody() });
  await call(worker, bucket, "/api/overrides", { method: "POST", headers, body: goodBody({ target: 2, value: 0.1 }) });
  const keys = historyKeys(bucket);
  assert.equal(keys.length, 2);
  for (const key of keys) assert.match(key, /^overrides\/history\/\d{8}T\d{9}Z-create-[A-Za-z0-9._:-]+-[0-9a-f]{6}\.json$/);
});

test("E4 fault injection: a transient history failure is retried and the change stays applied", async () => {
  const worker = await loadWorker();
  const bucket = fakeBucket();
  let failures = 1;
  bucket.fault = (key) => (key.startsWith("overrides/history/") && failures-- > 0 ? "throw" : undefined);
  const response = await call(worker, bucket, "/api/overrides", { method: "POST", headers: accessHeaders(), body: goodBody() });
  assert.equal(response.status, 201);
  assert.equal(activeIds(bucket).length, 1);
  assert.equal(historyKeys(bucket).length, 1);
});

test("E4 fault injection: history put fails after retries -> previous records restored, 500 change not applied", async () => {
  const worker = await loadWorker();
  const bucket = fakeBucket();
  const headers = accessHeaders();
  const first = (await (await call(worker, bucket, "/api/overrides", { method: "POST", headers, body: goodBody() })).json()).record;
  const before = bucket.store.get("overrides/active.json").body;
  bucket.fault = (key) => (key.startsWith("overrides/history/") ? "throw" : undefined);
  const response = await call(worker, bucket, "/api/overrides", { method: "POST", headers, body: goodBody({ target: 2, value: 0.1 }) });
  assert.equal(response.status, 500);
  assert.match((await response.json()).error, /change not applied/);
  assert.deepEqual(activeIds(bucket), [first.id], "the failed create was rolled back");
  assert.equal(JSON.parse(bucket.store.get("overrides/active.json").body).length, JSON.parse(before).length);
  assert.equal(historyKeys(bucket).length, 1, "only the first create is audited");
  // The same guarantee for remove: the removed record comes back.
  const removed = await call(worker, bucket, `/api/overrides/${first.id}`, { method: "DELETE", headers });
  assert.equal(removed.status, 500);
  assert.match((await removed.json()).error, /change not applied/);
  assert.deepEqual(activeIds(bucket), [first.id], "a failed remove audit restores the record");
});

test("E4 fault injection: history put fails AND the rollback put fails -> 500 audit missing; live override present, with the record id", async () => {
  const worker = await loadWorker();
  const bucket = fakeBucket();
  let activeWrites = 0;
  bucket.fault = (key) => {
    if (key.startsWith("overrides/history/")) return "throw";
    if (key === "overrides/active.json" && ++activeWrites === 2) return "throw"; // 1st = the change, 2nd = the compensating restore
    return undefined;
  };
  const response = await call(worker, bucket, "/api/overrides", { method: "POST", headers: accessHeaders(), body: goodBody() });
  assert.equal(response.status, 500);
  const [liveId] = activeIds(bucket);
  assert.ok(liveId, "the override is live");
  const error = (await response.json()).error;
  assert.match(error, /audit missing; live override present/);
  assert.ok(error.includes(liveId), "the message names the record id");
  assert.equal(historyKeys(bucket).length, 0);
});

test("E4 fault injection: a rollback whose etag guard loses a race never clobbers the newer list", async () => {
  const worker = await loadWorker();
  const bucket = fakeBucket();
  const headers = accessHeaders();
  bucket.fault = (key) => {
    if (!key.startsWith("overrides/history/")) return undefined;
    // Someone else edits active.json between our change and our rollback.
    const current = JSON.parse(bucket.store.get("overrides/active.json").body);
    bucket.store.set("overrides/active.json", { body: JSON.stringify([...current, { id: "ov-other", note: "concurrent" }]), v: 99 });
    return "throw";
  };
  const response = await call(worker, bucket, "/api/overrides", { method: "POST", headers, body: goodBody() });
  assert.equal(response.status, 500);
  assert.match((await response.json()).error, /audit missing; live override present/);
  assert.ok(activeIds(bucket).includes("ov-other"), "the concurrent edit survives");
});

test("E7: malformed percent-encoding and a non-URL Origin return JSON 4xx, never an unhandled 500", async () => {
  const worker = await loadWorker();
  const bucket = fakeBucket();
  const headers = accessHeaders();
  const badPath = await call(worker, bucket, "/api/golfprice/%E0%A4%A", { headers });
  assert.equal(badPath.status, 400);
  assert.equal((await badPath.json()).ok, false);
  const badId = await call(worker, bucket, "/api/overrides/%E0%A4%A", { method: "DELETE", headers });
  assert.equal(badId.status, 400);
  assert.equal((await badId.json()).error, "invalid override id");
  const badOrigin = await call(worker, bucket, "/api/overrides", { method: "POST", headers: { ...headers, origin: "null" }, body: goodBody() });
  assert.equal(badOrigin.status, 403);
  assert.equal((await badOrigin.json()).error, "cross-origin write refused");
  const badData = await worker.fetch(new Request("https://golf.example/api/data/%E0%A4%A"), { ASSETS: assets, DASHBOARD_DATA: bucket }, context);
  assert.equal(badData.status, 400);
  assert.equal(bucket.store.size, 0);
});

test("E7: the top-level catch turns an unexpected throw into a JSON 500", async () => {
  const worker = await loadWorker();
  const exploding = { get: async () => { throw new Error("r2 exploded"); }, put: async () => null, list: async () => { throw new Error("r2 exploded"); } };
  const response = await call(worker, exploding, "/api/overrides/history");
  assert.equal(response.status, 500);
  assert.match(response.headers.get("content-type") ?? "", /application\/json/);
  assert.deepEqual(await response.json(), { ok: false, error: "internal error" });
});

test("E7: access.ts refetches the certs once when the token's kid is not cached (key rotation)", async () => {
  const worker = await loadWorker();
  const bucket = fakeBucket();
  const warm = await call(worker, bucket, "/api/overrides", { method: "POST", headers: accessHeaders(), body: goodBody() });
  assert.equal(warm.status, 201);
  const fetchesBefore = certState.fetches;
  certState.keys = [...certState.keys, { ...rotatedPublicKey.export({ format: "jwk" }), kid: "k2" }];
  const rotated = await call(worker, bucket, "/api/overrides", { method: "POST", headers: accessHeaders("owner@example.com", undefined, { kid: "k2", key: rotatedPrivateKey }), body: goodBody({ target: 2, value: 0.1 }) });
  assert.equal(rotated.status, 201, "the new kid verified after one refetch");
  assert.equal(certState.fetches, fetchesBefore + 1);
  const unknown = await call(worker, bucket, "/api/overrides", { method: "POST", headers: accessHeaders("owner@example.com", undefined, { kid: "k3", key: rotatedPrivateKey }), body: goodBody({ target: 3, value: 0.1 }) });
  assert.equal(unknown.status, 401);
  assert.equal(certState.fetches, fetchesBefore + 2, "an unknown kid costs exactly one refetch per request");
  certState.keys = certState.keys.filter((k) => k.kid === "k1");
});

test("E4 fault injection: a failed audit on expire restores the original expiry", async () => {
  const worker = await loadWorker();
  const bucket = fakeBucket();
  const headers = accessHeaders();
  const first = (await (await call(worker, bucket, "/api/overrides", { method: "POST", headers, body: goodBody() })).json()).record;
  const before = JSON.parse(bucket.store.get("overrides/active.json").body);
  bucket.fault = (key) => (key.startsWith("overrides/history/") ? "throw" : undefined);
  const response = await call(worker, bucket, `/api/overrides/${first.id}?mode=expire`, { method: "DELETE", headers });
  assert.equal(response.status, 500);
  assert.match((await response.json()).error, /change not applied/);
  const after = JSON.parse(bucket.store.get("overrides/active.json").body);
  assert.deepEqual(after, before, "the record keeps its original expires_at");
  assert.equal(historyKeys(bucket).length, 1, "only the create is audited");
});
