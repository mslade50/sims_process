import assert from "node:assert/strict";
import { createSign, generateKeyPairSync } from "node:crypto";
import test from "node:test";
import { JOB_ID_RE, JOB_TYPES, MACHINES, MAX_QUEUE, machineStatus, waitingFor, buildJob, cancelProblem, checkRequest, effectiveState, makeJobId, rateLimitProblem, trimQueue, listWindow } from "../app/jobs-rules.ts";

const NOW = Date.parse("2026-10-05T14:00:00Z");

const job = (over = {}) => ({ id: "j-20261005T130000-aaaaaa", type: "tuesday", params: {}, requested_by: "o@example.com", requested_at: "2026-10-05T13:00:00Z", verified: true, ...over });

test("the allow-list is exactly the owner-approved moments plus publish, the cross-tour base refresh and publish, the odds check and the input watch", () => {
  assert.deepEqual(
    JOB_TYPES.map((s) => s.type),
    ["monday_settle", "monday_week", "tuesday", "wednesday", "thursday", "thursday_close", "after_round", "daily_health", "xtour", "xtour_publish", "publish", "odds_reprice", "watch"],
  );
});

test("unknown types, unknown keys and free text are rejected", () => {
  assert.match(checkRequest({ type: "rm -rf" }).problems.join(), /type must be one of/);
  assert.match(checkRequest({ type: "week" }).problems.join(), /type must be one of/);
  assert.match(checkRequest({ type: "tuesday", command: "x" }).problems.join(), /command is set by the server/);
  assert.match(checkRequest({ type: "tuesday", id: "j-1" }).problems.join(), /id is set by the server/);
  assert.match(checkRequest({ type: "tuesday", params: { event: "pga:1" } }).problems.join(), /param event is not allowed/);
  assert.match(checkRequest({ type: "tuesday", params: { supersede: "yes" } }).problems.join(), /true or false/);
  assert.match(checkRequest(null).problems.join(), /JSON object/);
  assert.match(checkRequest([]).problems.join(), /JSON object/);
  assert.match(checkRequest({ type: "tuesday", params: [] }).problems.join(), /params must be an object/);
});

test("flags are accepted only where the moment accepts them", () => {
  assert.deepEqual(checkRequest({ type: "tuesday", params: { no_pull: true, supersede: true } }), { problems: [], type: "tuesday", params: { no_pull: true, supersede: true } });
  assert.deepEqual(checkRequest({ type: "tuesday", params: { no_pull: false } }).params, {});
  assert.match(checkRequest({ type: "daily_health", params: { supersede: true } }).problems.join(), /does not accept supersede/);
  assert.match(checkRequest({ type: "publish", params: { no_pull: true } }).problems.join(), /does not accept no_pull/);
  assert.match(checkRequest({ type: "monday_settle", params: { supersede: true } }).problems.join(), /does not accept supersede/);
  assert.match(checkRequest({ type: "thursday_close", params: { no_pull: true } }).problems.join(), /does not accept no_pull/);
  assert.deepEqual(checkRequest({ type: "monday_settle", params: { no_pull: true } }).problems, []);
  for (const type of ["xtour", "xtour_publish"]) {
    assert.deepEqual(checkRequest({ type }).problems, []);
    for (const flag of ["no_pull", "supersede"]) assert.match(checkRequest({ type, params: { [flag]: true } }).problems.join(), new RegExp(`does not accept ${flag}`));
    assert.match(checkRequest({ type, params: { after_round: 1 } }).problems.join(), /does not take after_round/);
  }
});

test("after_round is 1, 2 or 3 and only on after_round", () => {
  for (const n of [1, 2, 3]) assert.deepEqual(checkRequest({ type: "after_round", params: { after_round: n } }).params, { after_round: n });
  for (const n of [0, 4, "2", 1.5, null]) assert.match(checkRequest({ type: "after_round", params: { after_round: n } }).problems.join(), /after_round must be 1, 2 or 3/);
  assert.deepEqual(checkRequest({ type: "after_round" }).problems, []);
  assert.match(checkRequest({ type: "wednesday", params: { after_round: 1 } }).problems.join(), /does not take after_round/);
});

test("job creation needs a verified identity; the server sets id, requester and time", () => {
  assert.match(buildJob({ type: "tuesday" }, "o@example.com", false, NOW, "abc123").problems.join(), /not verified/);
  const built = buildJob({ type: "tuesday", params: { supersede: true } }, "o@example.com", true, NOW, "abc123");
  assert.deepEqual(built.job, { id: "j-20261005T140000-abc123", type: "tuesday", params: { supersede: true }, requested_by: "o@example.com", requested_at: "2026-10-05T14:00:00Z", verified: true });
  assert.match(built.job.id, JOB_ID_RE);
  assert.ok(makeJobId(NOW, "000000") < makeJobId(NOW + 1000, "000000"), "ids sort by time");
});

test("rate limit: one non-terminal job per type, unless older than 6 h", () => {
  const none = {};
  assert.match(rateLimitProblem("tuesday", [job()], none, NOW), /already queued/);
  assert.equal(rateLimitProblem("wednesday", [job()], none, NOW), null, "other types are unaffected");
  assert.match(rateLimitProblem("tuesday", [job()], { [job().id]: { id: "x", state: "running" } }, NOW), /already running/);
  assert.match(rateLimitProblem("tuesday", [job()], { [job().id]: { id: "x", state: "claimed" } }, NOW), /already claimed/);
  for (const state of ["done", "failed", "expired", "cancelled"]) assert.equal(rateLimitProblem("tuesday", [job()], { [job().id]: { id: "x", state } }, NOW), null, state);
  assert.equal(rateLimitProblem("tuesday", [job({ cancelled_at: "2026-10-05T13:30:00Z" })], none, NOW), null, "cancelled in the queue is terminal");
  assert.equal(rateLimitProblem("tuesday", [job({ requested_at: "2026-10-05T07:59:00Z" })], none, NOW), null, "older than 6 h no longer blocks");
  assert.match(rateLimitProblem("tuesday", [job({ requested_at: "2026-10-05T08:01:00Z" })], none, NOW), /already queued/);
  assert.equal(rateLimitProblem("tuesday", [], none, NOW), null);
});

test("cancel applies only to a job the runner has not claimed", () => {
  assert.equal(cancelProblem(job(), null), null);
  assert.match(cancelProblem(undefined, null), /no such job/);
  assert.match(cancelProblem(job({ cancelled_at: "2026-10-05T13:30:00Z" }), null), /already cancelled/);
  assert.match(cancelProblem(job(), { id: "x", state: "claimed" }), /already claimed/);
  assert.match(cancelProblem(job(), { id: "x", state: "done" }), /already done/);
  assert.equal(effectiveState(job({ cancelled_at: "2026-10-05T13:30:00Z" }), null), "cancelled");
  assert.equal(effectiveState(job(), { id: "x", state: "running" }), "running");
});

test("the queue keeps the newest 200", () => {
  const many = Array.from({ length: 230 }, (_, i) => job({ id: `j-${i}` }));
  const kept = trimQueue(many);
  assert.equal(kept.length, MAX_QUEUE);
  assert.equal(kept[0].id, "j-30");
  assert.equal(kept.at(-1).id, "j-229");
});

test("a full queue drops the oldest automatic watch / odds jobs before any pricing job", () => {
  const jobs = [];
  for (let i = 0; i < 210; i += 1) jobs.push(job({ id: `bg-${i}`, type: i % 2 ? "watch" : "odds_reprice", requested_by: "cron" }));
  jobs.splice(5, 0, job({ id: "tue-priced", type: "tuesday", requested_by: "cron" }));
  jobs.splice(9, 0, job({ id: "phone-watch", type: "watch", requested_by: "o@example.com" }));
  const kept = trimQueue(jobs);
  assert.equal(kept.length, MAX_QUEUE);
  assert.ok(kept.some((j) => j.id === "tue-priced") && kept.some((j) => j.id === "phone-watch"));
  assert.equal(kept.at(-1).id, "bg-209");
});

/* ---------------------------------------------------------------- Worker integration (built bundle, signed Access JWT, in-memory R2) */
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
    return { key, etag: `e${o.v}`, text: async () => o.body, json: async () => JSON.parse(o.body), writeHttpMetadata() {} };
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

const TEAM = "team.example.cloudflareaccess.com";
const AUD = "aud-123";
const { publicKey, privateKey } = generateKeyPairSync("rsa", { modulusLength: 2048 });
const b64url = (value) => Buffer.from(typeof value === "string" ? value : JSON.stringify(value)).toString("base64url");

function signedHeaders(email = "owner@example.com") {
  const head = b64url({ alg: "RS256", kid: "k1" });
  const payload = b64url({ email, aud: [AUD], iss: `https://${TEAM}`, exp: Math.floor(Date.now() / 1000) + 3600, sub: "u1" });
  const sig = createSign("RSA-SHA256").update(`${head}.${payload}`).sign(privateKey).toString("base64url");
  return { "cf-access-authenticated-user-email": email, "cf-access-jwt-assertion": `${head}.${payload}.${sig}` };
}

function unsignedHeaders(email = "owner@example.com") {
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

async function call(worker, bucket, path, { method = "GET", headers = {}, body, access = true } = {}) {
  const init = { method, headers: { ...headers, ...(body !== undefined ? { "content-type": "application/json" } : {}) } };
  if (body !== undefined) init.body = typeof body === "string" ? body : JSON.stringify(body);
  const env = { ASSETS: assets, DASHBOARD_DATA: bucket, ...(access ? { ACCESS_TEAM_DOMAIN: TEAM, ACCESS_AUD: AUD } : {}) };
  return worker.fetch(new Request(`https://golf.example${path}`, init), env, context);
}

test("POST /api/jobs: identity, verification, origin, allow-list, rate limit", async () => {
  const worker = await loadWorker();
  const bucket = fakeBucket();
  const body = { type: "tuesday", params: { supersede: true } };
  assert.equal((await call(worker, bucket, "/api/jobs", { method: "POST", body })).status, 401, "no identity");
  assert.equal((await call(worker, bucket, "/api/jobs", { method: "POST", headers: unsignedHeaders(), body, access: false })).status, 403, "identity not verifiable");
  assert.equal((await call(worker, bucket, "/api/jobs", { method: "POST", headers: unsignedHeaders(), body })).status, 401, "bad signature");
  assert.equal((await call(worker, bucket, "/api/jobs", { method: "POST", headers: { ...signedHeaders(), origin: "https://evil.example" }, body })).status, 403);
  const bad = await call(worker, bucket, "/api/jobs", { method: "POST", headers: signedHeaders(), body: { type: "shell" } });
  assert.equal(bad.status, 422);
  assert.equal(bucket.store.size, 0, "nothing written by refused requests");

  const ok = await call(worker, bucket, "/api/jobs", { method: "POST", headers: signedHeaders("Owner@Example.com"), body });
  assert.equal(ok.status, 201);
  const created = (await ok.json()).job;
  assert.match(created.id, JOB_ID_RE);
  assert.equal(created.requested_by, "owner@example.com");
  assert.equal(created.verified, true);
  const queue = JSON.parse(bucket.store.get("jobs/queue.json").body);
  assert.deepEqual(queue.jobs.map((j) => j.id), [created.id]);

  const again = await call(worker, bucket, "/api/jobs", { method: "POST", headers: signedHeaders(), body: { type: "tuesday" } });
  assert.equal(again.status, 409, "same type already queued");
  assert.equal((await call(worker, bucket, "/api/jobs", { method: "POST", headers: signedHeaders(), body: { type: "wednesday" } })).status, 201);

  // once the runner reports the first job done, the type is free again
  bucket.store.set(`jobs/status/${created.id}.json`, { body: JSON.stringify({ id: created.id, state: "done" }), v: 1 });
  assert.equal((await call(worker, bucket, "/api/jobs", { method: "POST", headers: signedHeaders(), body: { type: "tuesday" } })).status, 201);
});

test("GET /api/jobs lists newest first with statuses, derived states and heartbeats; log endpoint", async () => {
  const worker = await loadWorker();
  const recent = (minutesAgo) => new Date(Date.now() - minutesAgo * 60_000).toISOString().replace(/\.\d+Z$/, "Z");
  const a = job({ id: "j-20261005T120000-aaaaaa", type: "tuesday", requested_at: recent(120) });
  const b = job({ id: "j-20261005T130000-bbbbbb", type: "wednesday", requested_at: recent(60) });
  const bucket = fakeBucket({
    "jobs/queue.json": { schema: "golfprice.job_queue.v1", jobs: [a, b] },
    [`jobs/status/${a.id}.json`]: { id: a.id, state: "done", machine: "desktop-2ki41v6", exit_code: 0, summary: "ok" },
    "jobs/heartbeat/desktop-2ki41v6.json": { machine: "desktop-2ki41v6", at: "2026-10-05T13:58:00Z", state: "idle" },
    [`jobs/logs/${a.id}.txt`]: "line 1\nline 2\n",
  });
  const response = await call(worker, bucket, "/api/jobs");
  assert.equal(response.status, 200);
  const data = await response.json();
  assert.deepEqual(data.jobs.map((j) => [j.id, j.state]), [[b.id, "queued"], [a.id, "done"]]);
  assert.equal(data.jobs[1].status.machine, "desktop-2ki41v6");
  assert.equal(data.heartbeats.length, 1);
  const log = await call(worker, bucket, `/api/jobs/${a.id}/log`);
  assert.equal(await log.text(), "line 1\nline 2\n");
  assert.equal((await call(worker, bucket, "/api/jobs/not-an-id/log")).status, 400);
  assert.equal((await call(worker, fakeBucket(), "/api/jobs")).status, 200, "an empty bucket is an empty list");
});

test("POST /api/jobs/<id>/cancel marks a queued job; claimed or unknown jobs are refused", async () => {
  const worker = await loadWorker();
  const a = job({ id: "j-20261005T120000-aaaaaa", type: "tuesday" });
  const b = job({ id: "j-20261005T130000-bbbbbb", type: "wednesday" });
  const bucket = fakeBucket({
    "jobs/queue.json": { schema: "golfprice.job_queue.v1", jobs: [a, b] },
    [`jobs/status/${b.id}.json`]: { id: b.id, state: "claimed", machine: "m" },
  });
  assert.equal((await call(worker, bucket, `/api/jobs/${a.id}/cancel`, { method: "POST", body: {} })).status, 401);
  assert.equal((await call(worker, bucket, `/api/jobs/${a.id}/cancel`, { method: "POST", headers: unsignedHeaders(), body: {}, access: false })).status, 403);
  const ok = await call(worker, bucket, `/api/jobs/${a.id}/cancel`, { method: "POST", headers: signedHeaders(), body: {} });
  assert.equal(ok.status, 200);
  const queue = JSON.parse(bucket.store.get("jobs/queue.json").body);
  assert.equal(queue.jobs[0].cancelled_by, "owner@example.com");
  assert.ok(queue.jobs[0].cancelled_at);
  assert.equal(queue.jobs[1].cancelled_at, undefined);
  assert.equal((await call(worker, bucket, `/api/jobs/${a.id}/cancel`, { method: "POST", headers: signedHeaders(), body: {} })).status, 409, "already cancelled");
  assert.equal((await call(worker, bucket, `/api/jobs/${b.id}/cancel`, { method: "POST", headers: signedHeaders(), body: {} })).status, 409, "claimed");
  assert.equal((await call(worker, bucket, "/api/jobs/j-20261005T110000-cccccc/cancel", { method: "POST", headers: signedHeaders(), body: {} })).status, 404);
});

test("a corrupt queue is never overwritten", async () => {
  const worker = await loadWorker();
  const bucket = fakeBucket({ "jobs/queue.json": "{nope" });
  const response = await call(worker, bucket, "/api/jobs", { method: "POST", headers: signedHeaders(), body: { type: "tuesday" } });
  assert.equal(response.status, 500);
  assert.equal(bucket.store.get("jobs/queue.json").body, "{nope");
});

test("target_machine: allow-listed names only; auto/absent means untargeted", () => {
  assert.deepEqual(MACHINES.map((m) => m.name), ["desktop-2ki41v6", "mckinley_home"]);
  assert.equal(checkRequest({ type: "tuesday" }).target_machine, undefined);
  assert.equal(checkRequest({ type: "tuesday", target_machine: "auto" }).target_machine, undefined);
  assert.equal(checkRequest({ type: "tuesday", target_machine: null }).target_machine, undefined);
  assert.equal(checkRequest({ type: "tuesday", target_machine: "mckinley_home" }).target_machine, "mckinley_home");
  for (const bad of ["laptop", "evil", "", 3, ["mckinley_home"], "mckinley_home ", "MCKINLEY_HOME"]) {
    assert.match(checkRequest({ type: "tuesday", target_machine: bad }).problems.join(), /target_machine must be one of/);
  }
  const built = buildJob({ type: "tuesday", target_machine: "desktop-2ki41v6" }, "o@example.com", true, NOW, "abc123");
  assert.equal(built.job.target_machine, "desktop-2ki41v6");
  assert.equal("target_machine" in buildJob({ type: "tuesday" }, "o@example.com", true, NOW, "abc123").job, false);
});

test("machineStatus: offline without a heartbeat or when stale; running and idle otherwise", () => {
  const beat = (machine, ago, extra = {}) => ({ machine, at: new Date(NOW - ago).toISOString().replace(/\.\d{3}Z$/, "Z"), state: "idle", ...extra });
  const [desk, lap] = MACHINES;
  assert.equal(machineStatus(desk, [], NOW).status, "offline");
  assert.equal(machineStatus(desk, undefined, NOW).status, "offline");
  assert.equal(machineStatus(desk, [beat("desktop-2ki41v6", 60_000, { commit: "abc" })], NOW).status, "idle");
  assert.equal(machineStatus(desk, [beat("desktop-2ki41v6", 60_000, { commit: "abc" })], NOW).commit, "abc");
  assert.equal(machineStatus(desk, [beat("desktop-2ki41v6", 20 * 60_000)], NOW).status, "offline");
  const running = machineStatus(desk, [beat("desktop-2ki41v6", 30_000, { state: "running", job_id: "j-x" })], NOW);
  assert.deepEqual([running.status, running.jobId], ["running", "j-x"]);
  assert.equal(machineStatus(lap, [beat("desktop-2ki41v6", 1000)], NOW).status, "offline", "another machine's heartbeat does not count");
});

test("waitingFor: only queued jobs addressed to a machine; says why", () => {
  const beats = [{ machine: "desktop-2ki41v6", at: "2026-10-05T13:59:30Z", state: "idle" }];
  assert.equal(waitingFor(job(), "queued", beats, NOW), null);
  assert.match(waitingFor(job({ target_machine: "mckinley_home" }), "queued", beats, NOW), /waiting for Desktop \(big\) .*offline/);
  assert.match(waitingFor(job({ target_machine: "desktop-2ki41v6" }), "queued", beats, NOW), /waiting for Desktop \(always-on\) to pick it up/);
  assert.match(waitingFor(job({ target_machine: "desktop-2ki41v6" }), "queued", [{ ...beats[0], state: "running" }], NOW), /finish its current job/);
  assert.equal(waitingFor(job({ target_machine: "mckinley_home" }), "running", beats, NOW), null);
});

test("a queued job addressed to an offline machine can be cancelled and is not blocked by the target", () => {
  const j = job({ target_machine: "mckinley_home" });
  assert.equal(cancelProblem(j, null), null);
  assert.equal(effectiveState({ ...j, cancelled_at: "2026-10-05T13:30:00Z" }, null), "cancelled");
});

test("POST /api/jobs with target_machine: stored on the job, unknown machine refused, cancel works while it waits", async () => {
  const worker = await loadWorker();
  const bucket = fakeBucket();
  const bad = await call(worker, bucket, "/api/jobs", { method: "POST", headers: signedHeaders(), body: { type: "tuesday", target_machine: "evil-box" } });
  assert.equal(bad.status, 422);
  assert.match((await bad.json()).problems.join(), /target_machine must be one of/);
  assert.equal(bucket.store.size, 0);
  const ok = await call(worker, bucket, "/api/jobs", { method: "POST", headers: signedHeaders(), body: { type: "tuesday", target_machine: "mckinley_home" } });
  assert.equal(ok.status, 201);
  const created = (await ok.json()).job;
  assert.equal(created.target_machine, "mckinley_home");
  assert.equal(JSON.parse(bucket.store.get("jobs/queue.json").body).jobs[0].target_machine, "mckinley_home");
  const auto = await call(worker, bucket, "/api/jobs", { method: "POST", headers: signedHeaders(), body: { type: "wednesday", target_machine: "auto" } });
  assert.equal("target_machine" in (await auto.json()).job, false);
  const cancel = await call(worker, bucket, `/api/jobs/${created.id}/cancel`, { method: "POST", headers: signedHeaders(), body: {} });
  assert.equal(cancel.status, 200);
});


test("listWindow: last 50 non-cron jobs from 48 h plus the last 20 cron jobs, newest first", () => {
  const at = (minutesAgo) => new Date(NOW - minutesAgo * 60_000).toISOString().replace(/\.\d+Z$/, "Z");
  const jobs = [];
  jobs.push(job({ id: "j-20261001T000000-old000", requested_at: at(60 * 24 * 4) })); // 4 days old, manual: outside the 48 h window
  for (let i = 0; i < 60; i += 1) jobs.push(job({ id: `j-m${String(i).padStart(3, "0")}`, requested_at: at(1200 - i) }));
  for (let i = 0; i < 150; i += 1) jobs.push(job({ id: `j-c${String(i).padStart(3, "0")}`, requested_by: "cron", type: "watch", requested_at: at(900 - i) }));
  const listed = listWindow(jobs, NOW);
  assert.equal(listed.filter((j) => j.requested_by === "cron").length, 20);
  assert.equal(listed.filter((j) => j.requested_by !== "cron").length, 50);
  assert.equal(listed.some((j) => j.id.startsWith("j-20261001")), false);
  assert.equal(listed[0].id, "j-c149", "newest first");
  assert.equal(listed.filter((j) => j.requested_by !== "cron")[0].id, "j-m059");
  assert.equal(listed.find((j) => j.id === "j-m009"), undefined, "the 10 oldest manual jobs fall off the 50-job window");
  assert.deepEqual(listWindow([], NOW), []);
});

test("GET /api/jobs reads exactly the listed jobs' status keys and never lists the status prefix (the 1000-key list cap)", async () => {
  const worker = await loadWorker();
  const recent = (minutesAgo) => new Date(Date.now() - minutesAgo * 60_000).toISOString().replace(/\.\d+Z$/, "Z");
  const jobs = Array.from({ length: 3 }, (_, i) => job({ id: `j-20261005T12000${i}-aaaaa${i}`, type: "tuesday", requested_at: recent(30 - i) }));
  const initial = { "jobs/queue.json": { schema: "golfprice.job_queue.v1", jobs } };
  // 1,200 older unrelated status objects fill a 1000-key listing before the listed job's status key.
  for (let i = 0; i < 1200; i += 1) initial[`jobs/status/j-19990101T${String(i).padStart(6, "0")}-zzzzzz.json`] = { id: "x", state: "done" };
  initial[`jobs/status/${jobs[2].id}.json`] = { id: jobs[2].id, state: "done", machine: "desktop-2ki41v6", summary: "found" };
  const bucket = fakeBucket(initial);
  const realList = bucket.list;
  const listed = [];
  bucket.list = async (options = {}) => {
    listed.push(options.prefix);
    return realList({ ...options, limit: 1000 }).then((r) => ({ ...r, objects: r.objects.slice(0, 1000) }));
  };
  const data = await (await call(worker, bucket, "/api/jobs")).json();
  assert.equal(data.jobs[0].id, jobs[2].id);
  assert.equal(data.jobs[0].status.summary, "found");
  assert.equal(data.jobs[0].state, "done");
  assert.equal(listed.includes("jobs/status/"), false, "no list of the status prefix");
});

test("GET /api/jobs: a job with no status object is queued; an empty queue and a cron-only queue list cleanly", async () => {
  const worker = await loadWorker();
  const recent = (minutesAgo) => new Date(Date.now() - minutesAgo * 60_000).toISOString().replace(/\.\d+Z$/, "Z");
  const lone = job({ id: "j-20261005T120000-aaaaaa", type: "tuesday", requested_at: recent(10) });
  const one = await (await call(worker, fakeBucket({ "jobs/queue.json": { schema: "golfprice.job_queue.v1", jobs: [lone] } }), "/api/jobs")).json();
  assert.equal(one.jobs.length, 1);
  assert.equal(one.jobs[0].state, "queued");
  assert.equal(one.jobs[0].status ?? null, null, "no status object -> no status");

  const empty = await call(worker, fakeBucket({ "jobs/queue.json": { schema: "golfprice.job_queue.v1", jobs: [] } }), "/api/jobs");
  assert.equal(empty.status, 200);
  assert.deepEqual((await empty.json()).jobs, []);

  // Cron-only queue: 30 cron jobs, some older than 48 h; the last 20 are listed regardless of age.
  const cron = Array.from({ length: 30 }, (_, i) => job({ id: `j-c${String(i).padStart(3, "0")}`, type: "watch", requested_by: "cron", requested_at: recent(60 * 60 - i * 60) }));
  const listed = await (await call(worker, fakeBucket({ "jobs/queue.json": { schema: "golfprice.job_queue.v1", jobs: cron } }), "/api/jobs")).json();
  assert.equal(listed.jobs.length, 20);
  assert.equal(listed.jobs[0].id, "j-c029", "newest first");
  assert.ok(listed.jobs.every((j) => j.state === "queued"));
});
