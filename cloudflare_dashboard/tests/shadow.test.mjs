import assert from "node:assert/strict";
import test from "node:test";
import { evaluateShadow, buildSummary, MISSED_GRACE_MS } from "../worker/shadow-rules.ts";
import { dueJobs } from "../worker/cron-rules.ts";
import { MACHINES, machinesWithRoles, parseRoles } from "../app/jobs-rules.ts";

const MIN = 60_000;
const NOW = Date.parse("2026-10-07T16:07:00Z"); // Wed 12:07 EDT
const iso = (ms) => new Date(ms).toISOString().replace(/\.\d{3}Z$/, "Z");
const job = (id, type, agoMin, state, extra = {}) => ({ id, type, requested_at: iso(NOW - agoMin * MIN), state, ...extra });
const fresh = (machine, role) => ({ machine, role, beat: { at: iso(NOW - MIN) } });

/* ------------------------------------------------------------------ roles */
test("parseRoles accepts only two distinct allow-listed machines; machinesWithRoles falls back to the constants", () => {
  assert.deepEqual(parseRoles({ primary: "mckinley_home", backup: "desktop-2ki41v6" }), { primary: "mckinley_home", backup: "desktop-2ki41v6" });
  for (const bad of [null, "x", {}, { primary: "mckinley_home" }, { primary: "a", backup: "b" }, { primary: "mckinley_home", backup: "mckinley_home" }, { primary: 1, backup: 2 }]) {
    assert.equal(parseRoles(bad), null);
  }
  assert.equal(machinesWithRoles(null), MACHINES);
  const swapped = machinesWithRoles({ primary: "mckinley_home", backup: "desktop-2ki41v6" });
  assert.equal(swapped.find((m) => m.role === "primary").name, "mckinley_home");
  assert.equal(swapped.find((m) => m.role === "backup").name, "desktop-2ki41v6");
  assert.equal(MACHINES.find((m) => m.role === "primary").name, "desktop-2ki41v6", "constants are not mutated");
});

/* ------------------------------------------------------------------ rules */
test("shadow rules: failed, stuck and stale-heartbeat findings have stable keys", () => {
  const input = {
    now: NOW,
    jobs: [job("j-1", "tuesday", 60, "failed"), job("j-2", "watch", 45, "queued"), job("j-3", "watch", 5, "queued"), job("j-4", "tuesday", 240, "running"), job("j-5", "tuesday", 60, "done")],
    created: [],
    beats: [{ machine: "desktop-2ki41v6", role: "primary", beat: { at: iso(NOW - 90 * MIN) } }, fresh("mckinley_home", "backup")],
    dueAt: () => [],
  };
  const a = evaluateShadow(input);
  const keys = a.map((f) => f.key).sort();
  assert.deepEqual(keys, ["heartbeat-stale-desktop-2ki41v6-20261007T143700Z", "job-failed-j-1", "job-stuck-j-2", "job-stuck-j-4"]);
  assert.deepEqual(evaluateShadow(input).map((f) => f.key).sort(), keys, "same input, same keys");
  assert.equal(a.find((f) => f.kind === "heartbeat_stale").severity, "high");
  assert.equal(evaluateShadow({ ...input, beats: [] }).some((f) => f.kind === "heartbeat_stale"), false);
});

test("shadow rules: a never-seen machine yields one stale finding keyed 'never'", () => {
  const f = evaluateShadow({ now: NOW, jobs: [], created: [], beats: [{ machine: "mckinley_home", role: "backup", beat: null }], dueAt: () => [] });
  assert.deepEqual(f.map((x) => [x.key, x.severity]), [["heartbeat-stale-mckinley_home-never", "warn"]]);
});

test("shadow rules: expected cron occurrences without a created job are missed after the grace; created or rate-limited ones are not", () => {
  const slot = Date.parse("2026-10-07T15:00:00Z"); // 11:00 ET: odds_reprice (Wed window) and watch
  const dueAt = (ms) => (ms === slot ? [{ type: "watch" }, { type: "odds_reprice" }] : []);
  const base = { now: slot + MISSED_GRACE_MS, jobs: [], created: [], beats: [], dueAt };
  assert.deepEqual(evaluateShadow(base).map((f) => f.key).sort(), ["missed-odds_reprice-20261007T150000Z", "missed-watch-20261007T150000Z"]);
  assert.deepEqual(evaluateShadow({ ...base, now: slot + MISSED_GRACE_MS - 1 }), [], "inside the grace");
  const made = evaluateShadow({ ...base, created: [{ type: "watch", requested_at: iso(slot + 20_000) }] });
  assert.deepEqual(made.map((f) => f.subject), ["odds_reprice"]);
  const limited = evaluateShadow({ ...base, jobs: [job("j-9", "odds_reprice", 0, "running", { requested_at: iso(slot - 2 * 3_600_000) })], created: [{ type: "watch", requested_at: iso(slot) }] });
  assert.deepEqual(limited, [], "a live same-type job inside the 6 h rate limit explains the skipped enqueue");
});

test("shadow rules use the real cron SLOTS: the 11:30 ET Monday settle slot is expected", () => {
  const mon = Date.parse("2026-10-12T15:30:00Z"); // Mon 11:30 EDT
  const f = evaluateShadow({ now: mon + MISSED_GRACE_MS, jobs: [], created: [], beats: [], dueAt: dueJobs });
  assert.ok(f.some((x) => x.key === "missed-monday_settle-20261012T153000Z"));
});

test("summary counts findings and reads the registry shape defensively", () => {
  const s = buildSummary(NOW, [{ key: "k", kind: "job_failed", severity: "high", date: "2026-10-07", subject: "j", detail: "d" }], { jobs: [{}, {}] }, { jobs: 1, machines: 2 });
  assert.deepEqual([s.open, s.by_kind.job_failed, s.registry], [1, 1, { present: true, jobs: 2 }]);
  assert.deepEqual(buildSummary(NOW, [], null, { jobs: 0, machines: 0 }).registry, { present: false, jobs: null });
});

/* ------------------------------------------------------------------ through the built Worker */
async function loadWorker() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}-${Math.random()}`);
  return (await import(workerUrl.href)).default;
}

function fakeBucket(initial = {}) {
  const store = new Map(Object.entries(initial).map(([k, v]) => [k, { body: typeof v === "string" ? v : JSON.stringify(v), v: 1 }]));
  return {
    store,
    puts: [],
    get: async (key) => {
      const o = store.get(key);
      return o ? { key, etag: `e${o.v}`, text: async () => o.body } : null;
    },
    async put(key, body, options = {}) {
      this.puts.push(key);
      const existing = store.get(key);
      const cond = options.onlyIf;
      if (cond?.etagMatches && (!existing || `e${existing.v}` !== cond.etagMatches)) return null;
      if (cond?.etagDoesNotMatch === "*" && existing) return null;
      store.set(key, { body: String(body), v: (existing?.v ?? 0) + 1 });
      return { key, etag: `e${store.get(key).v}` };
    },
    list: async () => ({ objects: [], truncated: false }),
  };
}

const realFetch = globalThis.fetch;
let fetched = [];
test.before(() => {
  globalThis.fetch = async (input, init) => {
    fetched.push(typeof input === "string" ? input : input.url);
    return realFetch(input, init);
  };
});
test.after(() => {
  globalThis.fetch = realFetch;
});
test.beforeEach(() => {
  fetched = [];
});

async function tick(worker, env, scheduledTime) {
  const pending = [];
  await worker.scheduled({ scheduledTime, cron: "*/15 * * * *" }, env, { waitUntil: (p) => pending.push(p), passThroughOnException() {} });
  await Promise.all(pending);
}

const failedJob = { id: "j-20261007T150000-aaaaaa", type: "tuesday", params: {}, requested_by: "o@example.com", requested_at: iso(NOW - 60 * MIN), verified: true };
const seeded = () =>
  fakeBucket({
    "jobs/queue.json": { schema: "golfprice.job_queue.v1", jobs: [failedJob] },
    [`jobs/status/${failedJob.id}.json`]: { id: failedJob.id, state: "failed" },
    "jobs/heartbeat/desktop-2ki41v6.json": { machine: "desktop-2ki41v6", at: iso(NOW - MIN), state: "idle" },
    "jobs/heartbeat/mckinley_home.json": { machine: "mckinley_home", at: iso(NOW - MIN), state: "idle" },
    "ops/registry.json": { jobs: [{ id: "a" }] },
  });

test("shadow observer is off unless OPS_SHADOW_ENABLED is exactly 1: no ops/ writes", async () => {
  const worker = await loadWorker();
  for (const flag of [undefined, "", "0", "true"]) {
    const bucket = seeded();
    await tick(worker, { DASHBOARD_DATA: bucket, ...(flag === undefined ? {} : { OPS_SHADOW_ENABLED: flag }) }, NOW);
    assert.equal(bucket.puts.some((k) => k.startsWith("ops/")), false, `flag ${String(flag)}`);
  }
});

test("shadow observer on: writes idempotent incident objects and a summary, sends nothing, makes no network call", async () => {
  const worker = await loadWorker();
  const bucket = seeded();
  const env = { DASHBOARD_DATA: bucket, OPS_SHADOW_ENABLED: "1", TELEGRAM_BOT_TOKEN: "t", TELEGRAM_CHAT_ID: "1" }; // secrets without the kill-switch var
  await tick(worker, env, NOW);
  const incident = bucket.store.get("ops/shadow/incidents/2026-10-07/job-failed-j-20261007T150000-aaaaaa.json");
  assert.ok(incident, "failed job incident written under its date");
  const first = JSON.parse(incident.body);
  assert.deepEqual([first.kind, first.subject], ["job_failed", failedJob.id]);
  const summary = JSON.parse(bucket.store.get("ops/shadow/summary.json").body);
  assert.deepEqual([summary.registry.present, summary.registry.jobs, summary.checked.machines], [true, 1, 2]);
  assert.ok(summary.open >= 1);
  const objectsAfterFirst = [...bucket.store.keys()].filter((k) => k.startsWith("ops/shadow/incidents/")).length;
  await tick(worker, env, NOW + 2 * MIN);
  assert.equal([...bucket.store.keys()].filter((k) => k.startsWith("ops/shadow/incidents/")).length, objectsAfterFirst, "no duplicate incident objects");
  assert.equal(JSON.parse(bucket.store.get("ops/shadow/incidents/2026-10-07/job-failed-j-20261007T150000-aaaaaa.json").body).first_seen, first.first_seen, "first_seen is preserved");
  assert.deepEqual(fetched, [], "no outbound network call of any kind");
  assert.ok([...bucket.puts].every((k) => k.startsWith("ops/shadow/") || k.startsWith("jobs/")), "writes only ops/shadow/ plus the existing cron/queue keys");
  assert.equal(bucket.puts.some((k) => k.startsWith("jobs/alerts/")), false);
});

test("shadow observer tolerates a missing registry", async () => {
  const worker = await loadWorker();
  const bucket = seeded();
  bucket.store.delete("ops/registry.json");
  await tick(worker, { DASHBOARD_DATA: bucket, OPS_SHADOW_ENABLED: "1" }, NOW);
  assert.equal(JSON.parse(bucket.store.get("ops/shadow/summary.json").body).registry.present, false);
});

test("P1 uses jobs/roles.json: swapping primary/backup changes which heartbeat matters; bad roles fall back to the constants", async () => {
  const worker = await loadWorker();
  const stuck = { id: "j-20261007T150000-bbbbbb", type: "watch", params: {}, requested_by: "cron", requested_at: iso(NOW - 45 * MIN), verified: true };
  const make = (extra = {}) =>
    fakeBucket({
      "jobs/queue.json": { schema: "golfprice.job_queue.v1", jobs: [stuck] },
      "jobs/heartbeat/desktop-2ki41v6.json": { machine: "desktop-2ki41v6", at: iso(NOW - MIN), state: "idle" },
      "jobs/heartbeat/mckinley_home.json": { machine: "mckinley_home", at: iso(NOW - 90 * MIN), state: "idle" },
      ...extra,
    });
  const env = (bucket) => ({ DASHBOARD_DATA: bucket, TELEGRAM_BOT_TOKEN: "t", TELEGRAM_CHAT_ID: "1", LEGACY_ALERTS_ENABLED: "1" });
  const sent = [];
  const saved = globalThis.fetch;
  globalThis.fetch = async (input, init) => {
    const href = typeof input === "string" ? input : input.url;
    if (href.startsWith("https://api.telegram.org/")) {
      sent.push(JSON.parse(init.body).text);
      return new Response("{}", { status: 200 });
    }
    return saved(input, init);
  };
  try {
    await tick(worker, env(make()), NOW);
    assert.equal(sent.some((t) => /runner ALERT/.test(t)), false, "default roles: primary desktop-2ki41v6 is fresh");
    await tick(worker, env(make({ "jobs/roles.json": { primary: "mckinley_home", backup: "desktop-2ki41v6" } })), NOW);
    assert.equal(sent.filter((t) => /runner ALERT/.test(t)).length, 1, "swapped roles: stale primary plus a job queued over 30 min pages");
    sent.length = 0;
    await tick(worker, env(make({ "jobs/roles.json": { primary: "evil", backup: "mckinley_home" } })), NOW);
    assert.equal(sent.some((t) => /runner ALERT/.test(t)), false, "invalid roles object is ignored");
  } finally {
    globalThis.fetch = saved;
  }
});
