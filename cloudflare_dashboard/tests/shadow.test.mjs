import assert from "node:assert/strict";
import test from "node:test";
import { evaluateShadow, buildSummary, MISSED_GRACE_MS } from "../worker/shadow-rules.ts";
import { dueJobs } from "../worker/cron-rules.ts";
import { MACHINES, loadRoles, machinesWithRoles, parseRoles, rolesInvalid } from "../app/jobs-rules.ts";

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
    lists: [],
    failGet: new Set(),
    failPut: new Set(),
    get: async function (key) {
      if (this.failGet.has(key)) throw new Error(`simulated R2 failure for ${key}`);
      const o = store.get(key);
      const result = o ? { key, etag: `e${o.v}`, text: async () => o.body } : null;
      if (this.onGet) await this.onGet(key);
      return result;
    },
    async put(key, body, options = {}) {
      this.puts.push(key);
      if (this.failPut.has(key)) return null;
      const existing = store.get(key);
      const cond = options.onlyIf;
      if (cond?.etagMatches && (!existing || `e${existing.v}` !== cond.etagMatches)) return null;
      if (cond?.etagDoesNotMatch === "*" && existing) return null;
      store.set(key, { body: String(body), v: (existing?.v ?? 0) + 1 });
      return { key, etag: `e${store.get(key).v}` };
    },
    async delete(key) {
      store.delete(key);
    },
    async list(options = {}) {
      this.lists.push(options.prefix);
      return { objects: [...store.keys()].filter((k) => k.startsWith(options.prefix ?? "")).map((key) => ({ key })), truncated: false };
    },
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

/* ------------------------------------------------------------------ review round 1 regressions */
test("rules: a blocker is judged at the tick, not now (finished between the tick and the evaluation still blocked it)", () => {
  const slot = Date.parse("2026-10-07T15:00:00Z");
  const dueAt = (ms) => (ms === slot ? [{ type: "watch" }] : []);
  const base = { now: slot + 15 * MIN, created: [], beats: [], dueAt };
  const blocker = (over) => ({ id: "j-b", type: "watch", requested_at: iso(slot - 5 * MIN), state: "done", requested_by: "cron", ...over });
  assert.deepEqual(evaluateShadow({ ...base, jobs: [blocker({ finished_at: slot + 5 * MIN })] }), [], "finished after the tick: it was running at the tick");
  assert.equal(evaluateShadow({ ...base, jobs: [blocker({ finished_at: slot - MIN })] }).length, 1, "finished before the tick: no excuse");
  assert.deepEqual(evaluateShadow({ ...base, jobs: [blocker({ finished_at: null })] }), [], "unknown finish time is not turned into an incident");
  for (const state of ["queued", "claimed", "running"]) assert.deepEqual(evaluateShadow({ ...base, jobs: [blocker({ state, finished_at: null })] }), [], state);
  assert.equal(evaluateShadow({ ...base, jobs: [blocker({ state: "failed", finished_at: slot - MIN })] }).filter((f) => f.kind === "occurrence_missed").length, 1, "a job finished before the tick blocks nothing");
  assert.equal(evaluateShadow({ ...base, jobs: [blocker({ state: "running", requested_at: iso(slot - 7 * 3_600_000) })] }).filter((f) => f.kind === "occurrence_missed").length, 1, "older than 6 h no longer blocks");
});

test("rules: DST fall-back repeats the 01:00 ET slot (Nov 1 2026) and both occurrences are expected separately", () => {
  const now = Date.parse("2026-11-01T06:10:00Z"); // 01:10 EST, second pass through 01:00 ET
  const f = evaluateShadow({ now, jobs: [], created: [], beats: [], dueAt: dueJobs }).filter((x) => x.subject === "after_round");
  assert.deepEqual(f.map((x) => x.key).sort(), ["missed-after_round-20261101T050000Z", "missed-after_round-20261101T060000Z"]);
  const covered = evaluateShadow({ now, jobs: [], created: [{ type: "after_round", requested_at: "2026-11-01T05:00:20Z" }], beats: [], dueAt: dueJobs }).filter((x) => x.subject === "after_round");
  assert.deepEqual(covered.map((x) => x.key), ["missed-after_round-20261101T060000Z"], "the first pass does not cover the second");
});

test("observer selects records independent of the Run page limits: an old failure survives 30 later cron jobs", async () => {
  const worker = await loadWorker();
  const old = { id: "j-20261007T020000-oldfail", type: "tuesday", params: {}, requested_by: "o@example.com", requested_at: iso(NOW - 14 * 60 * MIN), verified: true };
  const cron = Array.from({ length: 30 }, (_, i) => ({ id: `j-20261007T1${String(i).padStart(5, "0")}-c${i}`, type: "odds_reprice", params: {}, requested_by: "cron", requested_at: iso(NOW - (300 - i * 5) * MIN), verified: true }));
  const bucket = fakeBucket({
    "jobs/queue.json": { schema: "golfprice.job_queue.v1", jobs: [old, ...cron] },
    [`jobs/status/${old.id}.json`]: { id: old.id, state: "failed" },
    ...Object.fromEntries(cron.map((j) => [`jobs/status/${j.id}.json`, { id: j.id, state: "done", finished_at: iso(Date.parse(j.requested_at) + MIN) }])),
    "jobs/heartbeat/desktop-2ki41v6.json": { machine: "desktop-2ki41v6", at: iso(NOW - MIN) },
    "jobs/heartbeat/mckinley_home.json": { machine: "mckinley_home", at: iso(NOW - MIN) },
  });
  await tick(worker, { DASHBOARD_DATA: bucket, OPS_SHADOW_ENABLED: "1" }, NOW);
  assert.ok(bucket.store.has(`ops/shadow/incidents/2026-10-07/job-failed-${old.id}.json`));
});

test("summary.json is etag-guarded: a newer summary is not replaced, and a lost race writes nothing", async () => {
  const worker = await loadWorker();
  const newer = { schema: "golf_ops.shadow.v1", updated_at: iso(NOW + 5 * MIN), open: 99 };
  const a = seeded();
  a.store.set("ops/shadow/summary.json", { body: JSON.stringify(newer), v: 1 });
  await tick(worker, { DASHBOARD_DATA: a, OPS_SHADOW_ENABLED: "1" }, NOW);
  assert.equal(JSON.parse(a.store.get("ops/shadow/summary.json").body).open, 99);
  assert.equal(a.puts.includes("ops/shadow/summary.json"), false);
  const b = seeded();
  b.failPut.add("ops/shadow/summary.json"); // the conditional PUT reports a lost race
  await tick(worker, { DASHBOARD_DATA: b, OPS_SHADOW_ENABLED: "1" }, NOW);
  assert.equal(b.store.has("ops/shadow/summary.json"), false);
  const c = seeded();
  await tick(worker, { DASHBOARD_DATA: c, OPS_SHADOW_ENABLED: "1" }, NOW);
  const first = JSON.parse(c.store.get("ops/shadow/summary.json").body).updated_at;
  await tick(worker, { DASHBOARD_DATA: c, OPS_SHADOW_ENABLED: "1" }, NOW - 5 * MIN); // an older overlapping run finishing late
  assert.equal(JSON.parse(c.store.get("ops/shadow/summary.json").body).updated_at, first);
});

test("R2 read errors are not absence: a failed queue GET skips the tick; a failed status GET omits only that job", async () => {
  const worker = await loadWorker();
  const q = seeded();
  q.failGet.add("jobs/queue.json");
  await tick(worker, { DASHBOARD_DATA: q, OPS_SHADOW_ENABLED: "1" }, NOW);
  assert.equal(q.puts.some((k) => k.startsWith("ops/")), false, "nothing written, no false occurrence_missed");
  const s = seeded();
  s.failGet.add(`jobs/status/${failedJob.id}.json`);
  await tick(worker, { DASHBOARD_DATA: s, OPS_SHADOW_ENABLED: "1" }, NOW);
  assert.equal([...s.store.keys()].some((k) => k.includes("job-failed-") || k.includes("job-stuck-")), false, "the unreadable job yields no finding");
});

test("known incidents are listed, not re-PUT; new findings per tick are capped with a note", async () => {
  const worker = await loadWorker();
  const bucket = seeded();
  const env = { DASHBOARD_DATA: bucket, OPS_SHADOW_ENABLED: "1" };
  await tick(worker, env, NOW);
  assert.ok(bucket.puts.filter((k) => k.startsWith("ops/shadow/incidents/")).length >= 1 && bucket.lists.length >= 1);
  bucket.puts.length = 0;
  await tick(worker, env, NOW + 2 * MIN);
  assert.deepEqual(bucket.puts.filter((k) => k.startsWith("ops/shadow/incidents/")), [], "second tick PUTs no known incident");
  const lines = [];
  const log = console.log;
  console.log = (...a) => lines.push(a.join(" "));
  try {
    const empty = fakeBucket({ "jobs/queue.json": { schema: "golfprice.job_queue.v1", jobs: [] } });
    await tick(worker, { DASHBOARD_DATA: empty, OPS_SHADOW_ENABLED: "1" }, Date.parse("2026-10-14T16:07:00Z"));
    assert.ok(empty.puts.filter((k) => k.startsWith("ops/shadow/incidents/")).length <= 25);
  } finally {
    console.log = log;
  }
});

test("disabled shadow logs once per tick; invalid roles are reported by the shadow and alerts logs", async () => {
  const worker = await loadWorker();
  const lines = [];
  const log = console.log;
  console.log = (...a) => lines.push(a.join(" "));
  try {
    await tick(worker, { DASHBOARD_DATA: seeded() }, NOW);
    assert.equal(lines.filter((l) => l.startsWith("shadow:") && l.includes("disabled")).length, 1);
    lines.length = 0;
    const bucket = seeded();
    bucket.store.set("jobs/roles.json", { body: JSON.stringify({ primary: "desktop-2ki41v6", backup: "mckinley-home" }), v: 1 });
    await tick(worker, { DASHBOARD_DATA: bucket, OPS_SHADOW_ENABLED: "1" }, NOW);
    assert.ok(lines.some((l) => l.startsWith("shadow:") && l.includes("roles_invalid")));
    assert.ok(lines.some((l) => l.startsWith("alerts:") && l.includes("roles_invalid")));
  } finally {
    console.log = log;
  }
  assert.deepEqual([rolesInvalid(null), rolesInvalid(undefined), rolesInvalid({ primary: "desktop-2ki41v6", backup: "mckinley_home" }), rolesInvalid({ primary: "x", backup: "y" }), rolesInvalid("junk")], [false, false, false, true, true]);
});

/* ------------------------------------------------------------------ review round 2 regressions */
test("rules: an unreadable status suppresses only the 6 h after its own request, and is itself a finding", () => {
  const slot = Date.parse("2026-10-07T15:00:00Z");
  const dueAt = (ms) => (ms === slot || ms === slot + 15 * MIN ? [{ type: "watch" }] : []);
  const base = { now: slot + 30 * MIN, jobs: [], created: [], beats: [], dueAt };
  const missed = (u) => evaluateShadow({ ...base, unreadable: u }).filter((f) => f.kind === "occurrence_missed").map((f) => f.key);
  assert.deepEqual(missed([]), ["missed-watch-20261007T150000Z", "missed-watch-20261007T151500Z"]);
  const old = { id: "j-old", type: "watch", requested_at: iso(slot - 20 * 3_600_000) };
  assert.deepEqual(missed([old]), ["missed-watch-20261007T150000Z", "missed-watch-20261007T151500Z"], "a corrupt status requested 20 h ago suppresses nothing today");
  const near = { id: "j-near", type: "watch", requested_at: iso(slot + 10 * MIN) };
  assert.deepEqual(missed([near]), ["missed-watch-20261007T150000Z"], "requested 10 min after the first slot: suppresses the 15:15 tick only");
  const before = { id: "j-before", type: "watch", requested_at: iso(slot - 10 * MIN) };
  assert.deepEqual(missed([before]), [], "requested 10 min before: both ticks fall in its 6 h window");
  assert.deepEqual(missed([{ ...before, type: "tuesday" }]), ["missed-watch-20261007T150000Z", "missed-watch-20261007T151500Z"], "other types are unaffected");
  assert.ok(evaluateShadow({ ...base, unreadable: [{ ...near, corrupt: true }] }).some((f) => f.key === "status-unreadable-j-near" && f.kind === "status_unreadable"));
});

test("Tuesday due slot through the Worker: an unreadable status for that type suppresses its slot and is reported; a readable finished one does not", async () => {
  const worker = await loadWorker();
  const T = Date.parse("2026-10-13T16:12:00Z"); // Tue 12:12 EDT, the 12:00 ET tuesday slot (16:00Z) is past its grace
  const tuesdayJob = { id: "j-20261013T155000-tttttt", type: "tuesday", params: {}, requested_by: "o@example.com", requested_at: "2026-10-13T15:50:00Z", verified: true };
  const make = (status) =>
    fakeBucket({
      "jobs/queue.json": { schema: "golfprice.job_queue.v1", jobs: [tuesdayJob] },
      "jobs/heartbeat/desktop-2ki41v6.json": { machine: "desktop-2ki41v6", at: iso(T - MIN) },
      "jobs/heartbeat/mckinley_home.json": { machine: "mckinley_home", at: iso(T - MIN) },
      ...(status === undefined ? {} : { [`jobs/status/${tuesdayJob.id}.json`]: status }),
    });
  const env = (bucket) => ({ DASHBOARD_DATA: bucket, OPS_SHADOW_ENABLED: "1" });
  const key = "ops/shadow/incidents/2026-10-13/missed-tuesday-20261013T160000Z.json";
  const readable = make({ id: tuesdayJob.id, state: "done", finished_at: "2026-10-13T15:55:00Z" });
  await tick(worker, env(readable), T);
  assert.ok(readable.store.has(key), "a job that finished before the slot does not excuse it");
  const corrupt = make("{not json");
  await tick(worker, env(corrupt), T);
  assert.equal(corrupt.store.has(key), false, "unreadable status suppresses the slot it may have blocked");
  assert.ok(corrupt.store.has(`ops/shadow/incidents/2026-10-13/status-unreadable-${tuesdayJob.id}.json`));
  assert.ok(JSON.parse(corrupt.store.get("ops/shadow/summary.json").body).degraded.includes("status_unreadable:1"));
  const failing = make({ id: tuesdayJob.id, state: "done" });
  failing.failGet.add(`jobs/status/${tuesdayJob.id}.json`);
  await tick(worker, env(failing), T);
  assert.equal(failing.store.has(key), false);
  assert.equal(failing.store.has(`ops/shadow/incidents/2026-10-13/status-unreadable-${tuesdayJob.id}.json`), false, "an R2 read failure is not a permanent incident");
  assert.ok(JSON.parse(failing.store.get("ops/shadow/summary.json").body).degraded.includes("status_read_error:1"));
});

test("the per-tick cap never loses a perishable finding: missed slots are written first and every finding is eventually written", async () => {
  const worker = await loadWorker();
  const T0 = Date.parse("2026-10-12T16:07:00Z"); // Monday: many due slots, empty cron history
  const failed = Array.from({ length: 60 }, (_, i) => ({ id: `j-20261012T14${String(i).padStart(4, "0")}-f${i}`, type: "wednesday", params: {}, requested_by: "o@example.com", requested_at: iso(T0 - 120 * MIN), verified: true }));
  const bucket = fakeBucket({
    "jobs/queue.json": { schema: "golfprice.job_queue.v1", jobs: failed },
    ...Object.fromEntries(failed.map((j) => [`jobs/status/${j.id}.json`, { id: j.id, state: "failed" }])),
  });
  const env = { DASHBOARD_DATA: bucket, OPS_SHADOW_ENABLED: "1" };
  const expectedMissed = evaluateShadow({ now: T0, jobs: [], created: [], beats: [], dueAt: dueJobs }).filter((f) => f.kind === "occurrence_missed");
  assert.ok(expectedMissed.length > 5);
  await tick(worker, env, T0);
  const incidentKeys = () => [...bucket.store.keys()].filter((k) => k.startsWith("ops/shadow/incidents/"));
  assert.ok(incidentKeys().length <= 25, "capped");
  for (const f of expectedMissed) assert.ok(bucket.store.has(`ops/shadow/incidents/${f.date}/${f.key}.json`), `perishable ${f.key} written first`);
  for (let i = 1; i <= 6; i += 1) await tick(worker, env, T0 + i * 15 * MIN);
  const have = incidentKeys().join("\n");
  for (const j of failed) assert.ok(have.includes(`job-failed-${j.id}.json`), j.id);
  for (const f of expectedMissed) assert.ok(have.includes(`${f.key}.json`));
});

test("queue / heartbeat / summary: parse failures become findings, R2 failures do not", async () => {
  const worker = await loadWorker();
  const env = (bucket) => ({ DASHBOARD_DATA: bucket, OPS_SHADOW_ENABLED: "1" });
  const q = seeded();
  q.store.set("jobs/queue.json", { body: "{broken", v: 1 });
  await tick(worker, env(q), NOW);
  assert.ok(q.store.has("ops/shadow/incidents/2026-10-07/queue-unreadable.json"));
  assert.equal([...q.store.keys()].some((k) => k.includes("missed-")), false, "no false missed slots from an unparsable queue");
  const h = seeded();
  h.store.set("jobs/heartbeat/mckinley_home.json", { body: "nope", v: 1 });
  await tick(worker, env(h), NOW);
  assert.ok(h.store.has("ops/shadow/incidents/2026-10-07/heartbeat-unreadable-mckinley_home.json"));
  const e = seeded();
  e.failGet.add("jobs/heartbeat/mckinley_home.json");
  await tick(worker, env(e), NOW);
  assert.equal([...e.store.keys()].some((k) => k.includes("heartbeat-")), false, "an R2 error on a heartbeat is not a finding and not a stale heartbeat");
  assert.ok(JSON.parse(e.store.get("ops/shadow/summary.json").body).degraded.includes("heartbeat_read_error:mckinley_home"));
  const s = seeded();
  s.store.set("ops/shadow/summary.json", { body: "corrupt{", v: 1 });
  await tick(worker, env(s), NOW);
  assert.equal(JSON.parse(s.store.get("ops/shadow/summary.json").body).schema, "golf_ops.shadow.v1", "a corrupt summary is replaced");
  const f = seeded();
  f.store.set("ops/shadow/summary.json", { body: "corrupt{", v: 1 });
  f.failPut.add("ops/shadow/summary.json");
  await tick(worker, env(f), NOW);
  assert.equal(f.store.get("ops/shadow/summary.json").body, "corrupt{");
});

test("loadRoles separates absent, invalid content (including a parse error) and a read failure; the fallback is unchanged", async () => {
  const obj = (text) => ({ get: async () => (text === null ? null : { text: async () => text }) });
  const roles = JSON.stringify({ primary: "mckinley_home", backup: "desktop-2ki41v6" });
  assert.deepEqual(await loadRoles(obj(null)), { roles: null, problem: null });
  assert.deepEqual(await loadRoles(obj(roles)), { roles: { primary: "mckinley_home", backup: "desktop-2ki41v6" }, problem: null });
  assert.deepEqual(await loadRoles(obj("{oops")), { roles: null, problem: "roles_invalid" });
  assert.deepEqual(await loadRoles(obj(JSON.stringify({ primary: "a", backup: "b" }))), { roles: null, problem: "roles_invalid" });
  assert.deepEqual(await loadRoles({ get: async () => { throw new Error("r2 down"); } }), { roles: null, problem: "roles_read_error" });
  assert.equal(machinesWithRoles(null), MACHINES);
});

test("roles problems are logged by shadow and alerts for malformed JSON and for read failures", async () => {
  const worker = await loadWorker();
  const lines = [];
  const log = console.log;
  console.log = (...a) => lines.push(a.join(" "));
  try {
    const bad = seeded();
    bad.store.set("jobs/roles.json", { body: "{typo", v: 1 });
    await tick(worker, { DASHBOARD_DATA: bad, OPS_SHADOW_ENABLED: "1" }, NOW);
    assert.ok(lines.some((l) => l.startsWith("shadow:") && l.includes("roles_invalid")));
    assert.ok(lines.some((l) => l.startsWith("alerts:") && l.includes("roles_invalid")));
    lines.length = 0;
    const err = seeded();
    err.failGet.add("jobs/roles.json");
    await tick(worker, { DASHBOARD_DATA: err, OPS_SHADOW_ENABLED: "1" }, NOW);
    assert.ok(lines.some((l) => l.startsWith("shadow:") && l.includes("roles_read_error")));
    assert.ok(lines.some((l) => l.startsWith("alerts:") && l.includes("roles_read_error")));
  } finally {
    console.log = log;
  }
});

/* ------------------------------------------------------------------ review round 3 regressions */
test("rules: only a corrupt status is an incident; an R2 read failure keeps the suppression without a permanent finding", () => {
  const slot = Date.parse("2026-10-07T15:00:00Z");
  const base = { now: slot + 12 * MIN, jobs: [], created: [], beats: [], dueAt: (ms) => (ms === slot ? [{ type: "watch" }] : []) };
  const readError = { id: "j-e", type: "watch", requested_at: iso(slot - 10 * MIN), corrupt: false };
  assert.deepEqual(evaluateShadow({ ...base, unreadable: [readError] }), [], "no incident, and the slot it may have blocked is still suppressed");
  const found = evaluateShadow({ ...base, unreadable: [{ ...readError, corrupt: true }] });
  assert.deepEqual(found.map((f) => f.key), ["status-unreadable-j-e"]);
  assert.ok(evaluateShadow({ now: slot, jobs: [{ id: "j-1", type: "t", requested_at: iso(slot - MIN), state: "failed" }], created: [], beats: [], dueAt: () => [] })[0].expires_at > slot, "every finding carries an expiry");
});

test("deferred findings survive the 48 h scan cutoff through ops/shadow/pending.json", async () => {
  const worker = await loadWorker();
  const T0 = Date.parse("2026-10-14T12:00:00Z");
  const jobs = Array.from({ length: 60 }, (_, i) => ({ id: `j-20261012T1215${String(i).padStart(2, "0")}-d${i}`, type: "wednesday", params: {}, requested_by: "o@example.com", requested_at: iso(T0 - (47 * 60 + 50) * MIN), verified: true }));
  const bucket = fakeBucket({
    "jobs/queue.json": { schema: "golfprice.job_queue.v1", jobs },
    ...Object.fromEntries(jobs.map((j) => [`jobs/status/${j.id}.json`, { id: j.id, state: "failed" }])),
    "jobs/heartbeat/desktop-2ki41v6.json": { machine: "desktop-2ki41v6", at: iso(T0 - MIN) },
    "jobs/heartbeat/mckinley_home.json": { machine: "mckinley_home", at: iso(T0 - MIN) },
  });
  const env = { DASHBOARD_DATA: bucket, OPS_SHADOW_ENABLED: "1" };
  const failedKeys = () => [...bucket.store.keys()].filter((k) => k.includes("/job-failed-"));
  await tick(worker, env, T0);
  assert.ok(failedKeys().length <= 25);
  const pending = JSON.parse(bucket.store.get("ops/shadow/pending.json").body);
  assert.ok(pending.findings.length >= 35 - 0, "the overflow is persisted");
  // Past the 48 h cutoff the jobs are no longer scanned, yet the pending findings still drain (and not before the cap allows).
  for (let i = 1; i <= 6; i += 1) await tick(worker, env, T0 + i * 15 * MIN);
  assert.equal(failedKeys().length, 60, "every deferred finding was eventually written");
  assert.equal(JSON.parse(bucket.store.get("ops/shadow/pending.json").body).findings.length, 0, "pending is drained");
});

test("a corrupt or wrong-shaped queue is queue_unreadable (never an empty queue) and the heartbeat check still runs", async () => {
  const worker = await loadWorker();
  for (const body of ["{broken", "{}", { schema: "x", jobs: {} }, [], "null", { jobs: "nope" }]) {
    const bucket = seeded();
    bucket.store.set("jobs/queue.json", { body: typeof body === "string" ? body : JSON.stringify(body), v: 1 });
    bucket.store.set("jobs/heartbeat/desktop-2ki41v6.json", { body: JSON.stringify({ machine: "desktop-2ki41v6", at: iso(NOW - 90 * MIN) }), v: 1 });
    await tick(worker, { DASHBOARD_DATA: bucket, OPS_SHADOW_ENABLED: "1" }, NOW);
    const keys = [...bucket.store.keys()];
    assert.ok(keys.includes("ops/shadow/incidents/2026-10-07/queue-unreadable.json"), JSON.stringify(body));
    assert.equal(keys.some((k) => k.includes("missed-") || k.includes("job-failed-")), false, "no job or slot findings from an unusable queue");
    assert.ok(keys.some((k) => k.includes("heartbeat-stale-desktop-2ki41v6")), "a dead primary is still reported");
    const summary = JSON.parse(bucket.store.get("ops/shadow/summary.json").body);
    assert.deepEqual([summary.checked.jobs, summary.checked.machines], [0, 2]);
  }
  const ok = seeded();
  ok.store.set("jobs/queue.json", { body: JSON.stringify({ schema: "golfprice.job_queue.v1", jobs: [] }), v: 1 });
  await tick(worker, { DASHBOARD_DATA: ok, OPS_SHADOW_ENABLED: "1" }, NOW);
  assert.equal(ok.store.has("ops/shadow/incidents/2026-10-07/queue-unreadable.json"), false, "an empty jobs array is a valid queue");
});

/* ------------------------------------------------------------------ review round 4 regressions */
function nearCutoffBucket(T0) {
  const near = Array.from({ length: 60 }, (_, i) => ({ id: `j-20261012T1210${String(i).padStart(2, "0")}-n${i}`, type: "wednesday", params: {}, requested_by: "o@example.com", requested_at: iso(T0 - (47 * 60 + 50) * MIN), verified: true }));
  const recent = Array.from({ length: 40 }, (_, i) => ({ id: `j-20261014T1000${String(i).padStart(2, "0")}-r${i}`, type: "wednesday", params: {}, requested_by: "o@example.com", requested_at: iso(T0 - 120 * MIN), verified: true }));
  const jobs = [...near, ...recent];
  const bucket = fakeBucket({
    "jobs/queue.json": { schema: "golfprice.job_queue.v1", jobs },
    ...Object.fromEntries(jobs.map((j) => [`jobs/status/${j.id}.json`, { id: j.id, state: "failed" }])),
    "jobs/heartbeat/desktop-2ki41v6.json": { machine: "desktop-2ki41v6", at: iso(T0 - MIN) },
    "jobs/heartbeat/mckinley_home.json": { machine: "mckinley_home", at: iso(T0 - MIN) },
  });
  return { bucket, jobs };
}
const failedIncidents = (bucket) => [...bucket.store.keys()].filter((k) => k.includes("/job-failed-"));

test("pending.json CAS conflict: an interleaved newer tick wins the race, the older tick merges and retries, nothing near the cutoff is lost", async () => {
  const worker = await loadWorker();
  const T0 = Date.parse("2026-10-14T12:00:00Z");
  const { bucket, jobs } = nearCutoffBucket(T0);
  const env = { DASHBOARD_DATA: bucket, OPS_SHADOW_ENABLED: "1" };
  let fired = false;
  bucket.onGet = async (key) => {
    if (key !== "ops/shadow/pending.json" || fired) return;
    fired = true; // tick A has just read pending.json; tick B (later, with its own findings) now runs to completion and writes it first
    await tick(worker, env, T0 + 15 * MIN);
  };
  await tick(worker, env, T0);
  assert.equal(fired, true);
  assert.ok(bucket.puts.filter((k) => k === "ops/shadow/pending.json").length >= 3, "tick B wrote first, tick A lost its conditional PUT and retried");
  for (let i = 2; i <= 9; i += 1) await tick(worker, env, T0 + i * 15 * MIN);
  assert.equal(failedIncidents(bucket).length, jobs.length, "every finding, including A's near-cutoff overflow, was written");
  assert.equal([...bucket.store.keys()].some((k) => k.startsWith("ops/shadow/pending-overflow/")), false);
});

test("pending.json that cannot be updated falls back to a unique overflow object that the next tick drains and deletes", async () => {
  const worker = await loadWorker();
  const T0 = Date.parse("2026-10-14T12:00:00Z");
  const { bucket, jobs } = nearCutoffBucket(T0);
  const env = { DASHBOARD_DATA: bucket, OPS_SHADOW_ENABLED: "1" };
  bucket.failPut.add("ops/shadow/pending.json"); // every conditional update loses
  await tick(worker, env, T0);
  const overflow = () => [...bucket.store.keys()].filter((k) => k.startsWith("ops/shadow/pending-overflow/"));
  assert.equal(overflow().length, 1);
  assert.ok(JSON.parse(bucket.store.get(overflow()[0]).body).findings.length > 25);
  bucket.failPut.delete("ops/shadow/pending.json");
  for (let i = 1; i <= 9; i += 1) await tick(worker, env, T0 + i * 15 * MIN);
  assert.equal(failedIncidents(bucket).length, jobs.length);
  assert.deepEqual(overflow(), [], "drained overflow is deleted");
});

test("invalid queue entries are dropped and reported; valid jobs, heartbeats and the summary still work", async () => {
  const worker = await loadWorker();
  const bucket = seeded();
  const queue = { schema: "golfprice.job_queue.v1", jobs: [null, 5, "x", {}, { id: "j-1" }, failedJob] };
  bucket.store.set("jobs/queue.json", { body: JSON.stringify(queue), v: 1 });
  bucket.store.set("jobs/heartbeat/desktop-2ki41v6.json", { body: JSON.stringify({ machine: "desktop-2ki41v6", at: iso(NOW - 90 * MIN) }), v: 1 });
  await tick(worker, { DASHBOARD_DATA: bucket, OPS_SHADOW_ENABLED: "1" }, NOW);
  const keys = [...bucket.store.keys()];
  assert.ok(keys.includes("ops/shadow/incidents/2026-10-07/queue-entry-invalid.json"));
  assert.ok(keys.some((k) => k.includes("heartbeat-stale-desktop-2ki41v6")), "stale primary still reported");
  assert.ok(keys.includes(`ops/shadow/incidents/2026-10-07/job-failed-${failedJob.id}.json`), "the valid job is still checked");
  const summary = JSON.parse(bucket.store.get("ops/shadow/summary.json").body);
  assert.ok(summary.degraded.includes("queue_entry_invalid:5"));
});
