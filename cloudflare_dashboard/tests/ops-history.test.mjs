import assert from "node:assert/strict";
import { createSign, generateKeyPairSync } from "node:crypto";
import test from "node:test";

const NOW = Date.parse("2026-10-27T12:00:00Z");
const TEAM = "ops-history-test.cloudflareaccess.com", AUD = "ops-history-aud";
const OWNER = "owner@example.com", CLIENT = "history-test.access";
const { publicKey, privateKey } = generateKeyPairSync("rsa", { modulusLength: 2048 });
const b64 = (value) => Buffer.from(JSON.stringify(value)).toString("base64url");
function auth(client) {
  const header = b64({ alg: "RS256", kid: "history-key" });
  const payload = b64({ aud: [AUD], iss: `https://${TEAM}`, exp: NOW / 1000 + 3600,
    ...(client ? { common_name: client, sub: "" } : { email: OWNER, sub: "history-owner" }) });
  const signature = createSign("RSA-SHA256").update(`${header}.${payload}`).sign(privateKey).toString("base64url");
  return { "cf-access-jwt-assertion": `${header}.${payload}.${signature}`,
    ...(!client ? { "cf-access-authenticated-user-email": OWNER } : {}) };
}
class Bucket {
  objects = new Map(); counter = 0; writes = 0; conflict = false;
  async get(key) {
    const value = this.objects.get(key);
    return value ? { etag: value.etag, size: Buffer.byteLength(value.text), text: async () => value.text } : null;
  }
  async put(key, text, options) {
    this.writes++;
    assert.ok(options?.onlyIf, "audit writes must remain conditional");
    if (this.conflict) { this.conflict = false; return null; }
    const old = this.objects.get(key);
    if (options.onlyIf.etagDoesNotMatch === "*" && old) return null;
    if (options.onlyIf.etagMatches && old?.etag !== options.onlyIf.etagMatches) return null;
    const etag = `history-${++this.counter}`;
    this.objects.set(key, { text, etag });
    return { etag };
  }
  seed(key, value) { this.objects.set(key, { text: JSON.stringify(value), etag: `history-${++this.counter}` }); }
}
const originalFetch = globalThis.fetch, originalNow = Date.now;
let worker;
test.before(async () => {
  Date.now = () => NOW;
  globalThis.fetch = async (input) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    assert.equal(url, `https://${TEAM}/cdn-cgi/access/certs`, "no external calls are permitted");
    return Response.json({ keys: [{ ...publicKey.export({ format: "jwk" }), kid: "history-key" }] });
  };
  worker = (await import("../dist/server/index.js?ops-history")).default;
});
test.after(() => { globalThis.fetch = originalFetch; Date.now = originalNow; });
async function call(bucket, path, body, client) {
  const request = new Request(`https://golf.example${path}`, {
    method: body ? "POST" : "GET", headers: { ...auth(client), ...(body ? { "content-type": "application/json" } : {}) },
    ...(body ? { body: JSON.stringify(body) } : {}),
  });
  return worker.fetch(request, { DASHBOARD_DATA: bucket, ACCESS_TEAM_DOMAIN: TEAM, ACCESS_AUD: AUD,
    OWNER_EMAILS: OWNER, OPS_AGENT_CLIENT_IDS: CLIENT,
    ASSETS: { fetch: async () => new Response("not found", { status: 404 }) } },
  { waitUntil() {}, passThroughOnException() {} });
}
const path = (id) => `/api/ops/tasks/${id}`;
const key = (id) => `owner/ops/tasks/${id}.json`;
const snapshot = ({ state, next_step }) => ({ state, next_step });
const note = "Reviewed the next safe action.";
const proof = "https://github.com/mslade50/golf_ops/commit/c65cf6b";
async function read(bucket, id) {
  const response = await call(bucket, path(id));
  assert.equal(response.status, 200);
  return (await response.json()).task;
}
async function save(bucket, id, body, client) {
  const response = await call(bucket, path(id), body, client);
  assert.equal(response.status, 200);
  return response.json();
}

test("every update retains exact before/after values and prior transition proof across later writes", async () => {
  const bucket = new Bucket(), initial = await read(bucket, "B02");
  const claim = await save(bucket, "B02", { op: "claim", revision: 0, note }, CLIENT);
  const update = { op: "update", revision: 1, note: "Implementation is ready for review.",
    state: "review", next_step: "  Obtain independent review.  ", evidence: [proof], lease_token: claim.lease_token };
  const first = await save(bucket, "B02", update, CLIENT);
  assert.deepEqual(first.task.history[1].changes, { before: snapshot(initial), after: { state: "review", next_step: "Obtain independent review." } });
  assert.deepEqual(first.task.history[1].actor, { kind: "agent", id: CLIENT });
  assert.equal(Object.hasOwn(first.task.history[0], "changes"), false);
  const retained = structuredClone(first.task.history);
  const second = await save(bucket, "B02", { ...update, revision: 2, state: "verified",
    next_step: "Observe the next real occurrence.", note: "Review and proof are complete.", evidence: ["/api/jobs/j-20261027T110000-aaaaaa/log"] }, CLIENT);
  assert.deepEqual(second.task.history[2].changes, { before: snapshot(first.task), after: snapshot(second.task) });
  const annotated = await save(bucket, "B02", { op: "note", revision: 3, note, lease_token: claim.lease_token }, CLIENT);
  const released = await save(bucket, "B02", { op: "release", revision: 4, note, lease_token: claim.lease_token }, CLIENT);
  const stored = await read(bucket, "B02");
  assert.deepEqual(stored.history, released.task.history);
  assert.deepEqual(stored.history.slice(0, 2), retained);
  assert.deepEqual(stored.history[2], second.task.history[2]);
  assert.deepEqual(stored.history[1].evidence, [proof]);
  assert.equal(Object.hasOwn(annotated.task.history[3], "changes"), false);
  assert.equal(Object.hasOwn(stored.history[4], "changes"), false);
});

test("owner decisions retain exact approved and deferred terms without turning agent notes into decisions", async () => {
  const bucket = new Bucket(), initial = await read(bucket, "D01");
  const approved = await save(bucket, "D01", { op: "decision", revision: 0, state: "verified",
    note: "  Retain the current settings for the explicitly reviewed components.  ", evidence: [proof] });
  assert.deepEqual(approved.task.history[0].changes, { before: snapshot(initial), after: snapshot(approved.task) });
  assert.equal(approved.task.next_step, "Retain the current settings for the explicitly reviewed components.");
  assert.equal(approved.task.history[0].action, "owner_decision_approved");
  assert.deepEqual(approved.task.history[0].actor, { kind: "owner", id: OWNER });
  const deferred = await save(bucket, "D01", { op: "decision", revision: 1, state: "blocked", note: "Defer pending a separate owner review." });
  assert.deepEqual(deferred.task.history[1].changes, { before: snapshot(approved.task), after: snapshot(deferred.task) });
  assert.equal(deferred.task.history[1].action, "owner_decision_deferred");
  const annotated = await save(bucket, "D01", { op: "note", revision: 2, note }, CLIENT);
  assert.deepEqual(annotated.task.history.slice(0, 2), deferred.task.history);
  assert.equal(Object.hasOwn(annotated.task.history[2], "changes"), false);
  const before = structuredClone(bucket.objects.get(key("D01")));
  assert.equal((await call(bucket, path("D01"), { op: "decision", revision: 3, state: "verified", note }, CLIENT)).status, 403);
  assert.deepEqual(bucket.objects.get(key("D01")), before);
});

test("legacy v1 histories remain readable and appendable without invented historical snapshots", async () => {
  const bucket = new Bucket();
  const claim = await save(bucket, "B02", { op: "claim", revision: 0, note });
  const first = await save(bucket, "B02", { op: "update", revision: 1, state: "review", next_step: "Review the retained proof.", note, evidence: [proof], lease_token: claim.lease_token });
  const legacy = JSON.parse(bucket.objects.get(key("B02")).text);
  for (const entry of legacy.history) delete entry.changes;
  bucket.seed(key("B02"), legacy);
  assert.deepEqual((await read(bucket, "B02")).history, legacy.history);
  const second = await save(bucket, "B02", { op: "update", revision: 2, state: "observation_pending", next_step: "Wait for current-event observations.", note, lease_token: claim.lease_token });
  assert.deepEqual(second.task.history.slice(0, 2), legacy.history);
  assert.deepEqual(second.task.history[2].changes, { before: snapshot(first.task), after: snapshot(second.task) });
  assert.equal(second.task.schema, "golfops.task.v1");
});

test("stored transition snapshots reject malformed nested fields before any readback or mutation", async (t) => {
  const base = await read(new Bucket(), "D01");
  const good = { before: snapshot(base), after: { state: "verified", next_step: "Retain the reviewed terms." } };
  const fixture = { ...base, ...good.after, revision: 1, updated_at: new Date(NOW).toISOString(), history: [
    { revision: 1, at: new Date(NOW).toISOString(), actor: { kind: "owner", id: OWNER }, action: "owner_decision_approved", note, evidence: [proof], changes: good },
  ] };
  const valid = new Bucket(); valid.seed(key("D01"), fixture);
  assert.deepEqual((await read(valid, "D01")).history[0].changes, good);
  const cases = [["null", null], ["array", []], ["empty", {}],
    ["missing before", { after: good.after }], ["missing after", { before: good.before }],
    ["unknown outer field", { ...good, extra: true }]];
  for (const side of ["before", "after"]) {
    const value = good[side];
    for (const [name, invalid] of [["null", null], ["array", []], ["empty", {}],
      ["missing state", { next_step: value.next_step }], ["missing next step", { state: value.state }],
      ["unknown field", { ...value, extra: true }], ["invalid state", { ...value, state: "done" }],
      ["non-string step", { ...value, next_step: 123 }], ["blank step", { ...value, next_step: "  " }],
      ["oversized step", { ...value, next_step: "x".repeat(2001) }],
      ["control character", { ...value, next_step: "unsafe\u0000step" }],
      ["secret-shaped step", { ...value, next_step: `Bearer ${"a".repeat(32)}` }]]) {
      cases.push([`${side}: ${name}`, { ...good, [side]: invalid }]);
    }
  }
  for (const [name, changes] of cases) await t.test(name, async () => {
    const bucket = new Bucket(), invalid = structuredClone(fixture);
    invalid.history[0].changes = changes; bucket.seed(key("D01"), invalid);
    const retained = structuredClone(bucket.objects.get(key("D01")));
    assert.equal((await call(bucket, path("D01"))).status, 503);
    assert.equal((await call(bucket, path("D01"), { op: "decision", revision: 1, state: "blocked", note })).status, 503);
    assert.equal(bucket.writes, 0);
    assert.deepEqual(bucket.objects.get(key("D01")), retained);
  });
});

test("failed update and decision CAS preserve stored audit bytes and etags", async () => {
  const bucket = new Bucket();
  const claim = await save(bucket, "B02", { op: "claim", revision: 0, note });
  await save(bucket, "B02", { op: "update", revision: 1, state: "review", next_step: "Review this change.", note, evidence: [proof], lease_token: claim.lease_token });
  await save(bucket, "D01", { op: "decision", revision: 0, state: "blocked", note });
  for (const [id, body] of [
    ["B02", { op: "update", revision: 2, state: "verified", next_step: "Observe the next run.", note, evidence: [proof], lease_token: claim.lease_token }],
    ["D01", { op: "decision", revision: 1, state: "verified", note, evidence: [proof] }],
    ["D02", { op: "decision", revision: 0, state: "verified", note, evidence: [proof] }],
  ]) {
    const retained = structuredClone(bucket.objects.get(key(id)));
    bucket.conflict = true;
    assert.equal((await call(bucket, path(id), body)).status, 409);
    assert.deepEqual(bucket.objects.get(key(id)), retained);
  }
});

test("asynchronous workflow read failures return the operations 503 response", async (t) => {
  for (const failure of ["unreadable JSON", "oversized object", "rejected queue read", "rejected status read"]) await t.test(failure, async () => {
    const bucket = new Bucket(), get = bucket.get.bind(bucket);
    if (failure === "unreadable JSON") bucket.objects.set("jobs/queue.json", { etag: "bad", text: "{" });
    if (failure === "oversized object") bucket.get = async (key) => key === "jobs/queue.json" ? { size: 1_000_001 } : get(key);
    if (failure === "rejected queue read") bucket.get = async (key) => { if (key === "jobs/queue.json") throw new Error("fake storage failure"); return get(key); };
    if (failure === "rejected status read") {
      bucket.seed("jobs/queue.json", { jobs: [{ id: "j-20261027T110000-aaaaaa", type: "monday_week", requested_at: "2026-10-27T11:00:00Z" }] });
      bucket.get = async (key) => { if (key.startsWith("jobs/status/")) throw new Error("fake status failure"); return get(key); };
    }
    const retained = structuredClone([...bucket.objects]);
    const response = await call(bucket, "/api/ops/workflow?week=2026-10-25");
    assert.equal(response.status, 503);
    assert.equal(response.headers.get("cache-control"), "private, no-store");
    assert.deepEqual(await response.json(), { ok: false, error: "Operations records could not be read or saved safely. Existing records were preserved; check the service before retrying." });
    assert.equal(bucket.writes, 0);
    assert.deepEqual([...bucket.objects], retained);
  });
});
