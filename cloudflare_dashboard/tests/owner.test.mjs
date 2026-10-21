import assert from "node:assert/strict";
import { createSign, generateKeyPairSync } from "node:crypto";
import test from "node:test";
import { NAVIGATION, OWNER_ONLY_VIEWS, navigationFor, ownerOnlyPath, viewerFrom } from "../app/shell-rules.ts";

// Owner-only areas (owner, October 15, 2026): colleagues let in through Access never see or reach Run, weather edits or overrides.

test("owner-only paths: Run, weather edits and the jobs/override APIs, however they are spelled", () => {
  for (const path of ["/run", "/RUN", "/run/", "/run.rsc", "/weather-edits", "/weather-edits/", "/bet-breakdowns", "/Bet-Breakdowns/", "/bet-breakdowns.rsc"]) assert.equal(ownerOnlyPath(path), "page", path);
  for (const path of ["/api/jobs", "/api/jobs/j-1/log", "/api/jobs/j-1/cancel", "/api/overrides", "/api/overrides/history", "/api/overrides/abc", "/api/weather-overrides/pga_1_2026", "/API/Jobs/", "/api/owner-data", "/api/owner-data/bet_breakdowns.json", "/API/Owner-Data/x.json"]) assert.equal(ownerOnlyPath(path), "api", path);
  for (const path of ["/", "/this-week", "/inputs", "/weather-effects", "/running", "/api/data/manifest.json", "/api/golfprice/index.json", "/api/overridesx", "/api/whoami"]) assert.equal(ownerOnlyPath(path), null, path);
});

test("colleague navigation drops Run (and its now empty Operate group) and Weather edits; the owner keeps everything", () => {
  assert.deepEqual(navigationFor(true), NAVIGATION);
  const colleague = navigationFor(false);
  const keys = colleague.flatMap((g) => g.items.map((i) => i.key));
  for (const key of OWNER_ONLY_VIEWS) assert.ok(!keys.includes(key), key);
  assert.ok(!colleague.some((g) => g.label === "Operate"));
  assert.ok(keys.includes("inputs") && keys.includes("this-week") && keys.includes("performance"));
  assert.ok(NAVIGATION.some((g) => g.label === "Operate"), "the shared list is not mutated");
});

test("only an explicit owner: true from whoami counts as owner", () => {
  assert.deepEqual(viewerFrom({ ok: true, email: "a@b.c", owner: true }), { email: "a@b.c", owner: true });
  for (const body of [null, undefined, "x", { owner: "true" }, { owner: 1 }, { error: "nope" }]) assert.equal(viewerFrom(body).owner, false);
});

/* ------------------------------------------------------------------ the built Worker */
async function loadWorker() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}-${Math.random()}`);
  return (await import(workerUrl.href)).default;
}

const TEAM = "team.example.cloudflareaccess.com";
const AUD = "aud-123";
const OWNER = "owner@example.com";
const { publicKey, privateKey } = generateKeyPairSync("rsa", { modulusLength: 2048 });
const b64url = (value) => Buffer.from(typeof value === "string" ? value : JSON.stringify(value)).toString("base64url");

function signedHeaders(email) {
  const head = b64url({ alg: "RS256", kid: "k1" });
  const payload = b64url({ email, aud: [AUD], iss: `https://${TEAM}`, exp: Math.floor(Date.now() / 1000) + 3600, sub: "u1" });
  const sig = createSign("RSA-SHA256").update(`${head}.${payload}`).sign(privateKey).toString("base64url");
  return { "cf-access-authenticated-user-email": email, "cf-access-jwt-assertion": `${head}.${payload}.${sig}` };
}
function unsignedHeaders(email) {
  const jwt = `${b64url({ alg: "RS256", kid: "k1" })}.${b64url({ email, exp: Math.floor(Date.now() / 1000) + 3600 })}.c2ln`;
  return { "cf-access-authenticated-user-email": email, "cf-access-jwt-assertion": jwt };
}

const realFetch = globalThis.fetch;
test.before(() => {
  globalThis.fetch = async (input, init) => {
    const href = typeof input === "string" ? input : input.url;
    if (href === `https://${TEAM}/cdn-cgi/access/certs`) return Response.json({ keys: [{ ...publicKey.export({ format: "jwk" }), kid: "k1" }] });
    return realFetch(input, init);
  };
});
test.after(() => {
  globalThis.fetch = realFetch;
});

const context = { waitUntil() {}, passThroughOnException() {} };
const assets = { fetch: async () => new Response("nf", { status: 404 }) };
const writes = [];
const bucket = {
  get: async () => null,
  head: async () => null,
  put: async (key) => (writes.push(key), { etag: "e1" }),
  delete: async () => {},
  list: async () => ({ objects: [], truncated: false }),
};

async function call(worker, path, { method = "GET", headers = {}, body, owners = OWNER } = {}) {
  const init = { method, headers: { ...headers, ...(body !== undefined ? { "content-type": "application/json" } : {}) } };
  if (body !== undefined) init.body = JSON.stringify(body);
  const env = { ASSETS: assets, DASHBOARD_DATA: bucket, ACCESS_TEAM_DOMAIN: TEAM, ACCESS_AUD: AUD, ...(owners ? { OWNER_EMAILS: owners } : {}) };
  return worker.fetch(new Request(`https://golf.example${path}`, init), env, context);
}

test("whoami: owner, colleague, unsigned and anonymous", async () => {
  const worker = await loadWorker();
  const who = async (headers, owners) => (await call(worker, "/api/whoami", { headers, owners })).json();
  assert.deepEqual(await who(signedHeaders("Owner@Example.com")), { ok: true, email: OWNER, owner: true });
  assert.deepEqual(await who(signedHeaders("colleague@example.com")), { ok: true, email: "colleague@example.com", owner: false });
  assert.equal((await who(unsignedHeaders(OWNER))).owner, false, "a forged assertion claiming the owner's email is not the owner");
  assert.deepEqual(await who({}), { ok: true, email: null, owner: false });
  assert.equal((await who(signedHeaders("a@x.com"), " a@x.com , b@y.com ")).owner, true, "list is trimmed and comma separated");
  const res = await call(worker, "/api/whoami", { headers: signedHeaders(OWNER) });
  assert.equal(res.headers.get("cache-control"), "no-store");
  assert.equal((await call(worker, "/api/whoami", { method: "POST" })).status, 405);
});

test("a colleague is refused every owner-only API before it reads or writes anything", async () => {
  const worker = await loadWorker();
  const colleague = signedHeaders("colleague@example.com");
  const cases = [
    ["/api/jobs", "GET"],
    ["/api/jobs", "POST", { type: "tuesday" }],
    ["/api/jobs/j-20261015T120000-abcdef/cancel", "POST", {}],
    ["/api/jobs/j-20261015T120000-abcdef/log", "GET"],
    ["/api/overrides", "GET"],
    ["/api/overrides/history", "GET"],
    ["/api/overrides", "POST", { kind: "x" }],
    ["/api/overrides/abc?mode=remove", "DELETE"],
    ["/api/weather-overrides/pga_1_2026-10-15", "GET"],
    ["/api/weather-overrides/pga_1_2026-10-15", "PUT", { phase: "pre", hours: [] }],
    ["/api/owner-data/bet_breakdowns.json", "GET"],
  ];
  for (const [path, method, body] of cases) {
    for (const headers of [colleague, unsignedHeaders(OWNER), {}]) {
      const res = await call(worker, path, { method, headers, body });
      assert.equal(res.status, 403, `${method} ${path}`);
      assert.match((await res.json()).error, /owner only/);
    }
  }
  assert.deepEqual(writes, [], "nothing written");
});

test("owner-only pages redirect a colleague home; the owner and shared pages pass through", async () => {
  const worker = await loadWorker();
  const colleague = signedHeaders("colleague@example.com");
  for (const path of ["/run", "/weather-edits?e=pga_1", "/Run/", "/bet-breakdowns"]) {
    const res = await call(worker, path, { headers: colleague });
    assert.equal(res.status, 302, path);
    assert.equal(res.headers.get("location"), "https://golf.example/this-week");
  }
  const owner = await call(worker, "/run", { headers: signedHeaders(OWNER) });
  assert.notEqual(owner.status, 302, "owner reaches /run");
  const shared = await call(worker, "/this-week", { headers: colleague });
  assert.notEqual(shared.status, 302, "colleague reaches shared pages");
  const data = await call(worker, "/api/golfprice/index.json", { headers: colleague });
  assert.notEqual(data.status, 403, "published model data stays readable");
});

test("the owner still reaches the owner-only APIs; with OWNER_EMAILS unset the gate is off", async () => {
  const worker = await loadWorker();
  const jobs = await call(worker, "/api/jobs", { headers: signedHeaders(OWNER) });
  assert.equal(jobs.status, 200);
  assert.equal((await call(worker, "/api/jobs", { headers: signedHeaders("colleague@example.com"), owners: null })).status, 200);
  assert.equal((await call(worker, "/run", { headers: {}, owners: null })).status === 302, false);
});
