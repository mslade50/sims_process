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
    gets: [],
    delete_calls: 0,
    failGet: new Set(),
    failPut: new Set(),
    get: async function (key) {
      this.gets.push(key);
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
      this.delete_calls += 1;
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

function recordedFindings(bucket) {
  return [...bucket.store.entries()].filter(([k]) => k.startsWith("ops/shadow/batches/")).flatMap(([, v]) => JSON.parse(v.body).findings);
}
const recordedKeys = (bucket) => recordedFindings(bucket).map((f) => f.key);
const hasIncident = (bucket, key) => recordedKeys(bucket).includes(key);

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

test("shadow observer on: records findings in a batch, dedupes by index, sends nothing, makes no network call", async () => {
  const worker = await loadWorker();
  const bucket = seeded();
  const env = { DASHBOARD_DATA: bucket, OPS_SHADOW_ENABLED: "1", TELEGRAM_BOT_TOKEN: "t", TELEGRAM_CHAT_ID: "1" }; // secrets without the kill-switch var
  await tick(worker, env, NOW);
  const first = recordedFindings(bucket).find((f) => f.key === `job-failed-${failedJob.id}`);
  assert.ok(first, "failed job recorded");
  assert.deepEqual([first.kind, first.subject], ["job_failed", failedJob.id]);
  const summary = JSON.parse(bucket.store.get("ops/shadow/summary.json").body);
  assert.deepEqual([summary.registry.present, summary.registry.jobs, summary.checked.machines], [true, 1, 2]);
  assert.ok(summary.open >= 1 && summary.new_findings >= 1 && summary.index_conflict === false);
  const batchesAfterFirst = batchKeys(bucket).length;
  await tick(worker, env, NOW + 2 * MIN);
  assert.equal(recordedKeys(bucket).filter((k) => k === `job-failed-${failedJob.id}`).length, 1, "recorded once");
  assert.equal(recordedFindings(bucket).find((f) => f.key === `job-failed-${failedJob.id}`).first_seen, first.first_seen, "first_seen is preserved");
  assert.ok(batchKeys(bucket).length - batchesAfterFirst <= 1);
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
  assert.ok(hasIncident(bucket, `job-failed-${old.id}`));
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
  assert.equal(recordedKeys(s).some((k) => k.includes("job-failed-") || k.includes("job-stuck-")), false, "the unreadable job yields no finding");
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
  const key = "missed-tuesday-20261013T160000Z";
  const readable = make({ id: tuesdayJob.id, state: "done", finished_at: "2026-10-13T15:55:00Z" });
  await tick(worker, env(readable), T);
  assert.ok(hasIncident(readable, key), "a job that finished before the slot does not excuse it");
  const corrupt = make("{not json");
  await tick(worker, env(corrupt), T);
  assert.equal(hasIncident(corrupt, key), false, "unreadable status suppresses the slot it may have blocked");
  assert.ok(hasIncident(corrupt, `status-unreadable-${tuesdayJob.id}`));
  assert.ok(JSON.parse(corrupt.store.get("ops/shadow/summary.json").body).degraded.includes("status_unreadable:1"));
  const failing = make({ id: tuesdayJob.id, state: "done" });
  failing.failGet.add(`jobs/status/${tuesdayJob.id}.json`);
  await tick(worker, env(failing), T);
  assert.equal(hasIncident(failing, key), false);
  assert.equal(hasIncident(failing, `status-unreadable-${tuesdayJob.id}`), false, "an R2 read failure is not a permanent incident");
  assert.ok(JSON.parse(failing.store.get("ops/shadow/summary.json").body).degraded.includes("status_read_error:1"));
});

test("queue / heartbeat / summary: parse failures become findings, R2 failures do not", async () => {
  const worker = await loadWorker();
  const env = (bucket) => ({ DASHBOARD_DATA: bucket, OPS_SHADOW_ENABLED: "1" });
  const q = seeded();
  q.store.set("jobs/queue.json", { body: "{broken", v: 1 });
  await tick(worker, env(q), NOW);
  assert.ok(hasIncident(q, "queue-unreadable-2026-10-07"));
  assert.equal(recordedKeys(q).some((k) => k.includes("missed-")), false, "no false missed slots from an unparsable queue");
  const h = seeded();
  h.store.set("jobs/heartbeat/mckinley_home.json", { body: "nope", v: 1 });
  await tick(worker, env(h), NOW);
  assert.ok(hasIncident(h, "heartbeat-unreadable-mckinley_home-2026-10-07"));
  const e = seeded();
  e.failGet.add("jobs/heartbeat/mckinley_home.json");
  await tick(worker, env(e), NOW);
  assert.equal(recordedKeys(e).some((k) => k.includes("heartbeat-")), false, "an R2 error on a heartbeat is not a finding and not a stale heartbeat");
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

test("a corrupt or wrong-shaped queue is queue_unreadable (never an empty queue) and the heartbeat check still runs", async () => {
  const worker = await loadWorker();
  for (const body of ["{broken", "{}", { schema: "x", jobs: {} }, [], "null", { jobs: "nope" }]) {
    const bucket = seeded();
    bucket.store.set("jobs/queue.json", { body: typeof body === "string" ? body : JSON.stringify(body), v: 1 });
    bucket.store.set("jobs/heartbeat/desktop-2ki41v6.json", { body: JSON.stringify({ machine: "desktop-2ki41v6", at: iso(NOW - 90 * MIN) }), v: 1 });
    await tick(worker, { DASHBOARD_DATA: bucket, OPS_SHADOW_ENABLED: "1" }, NOW);
    const keys = recordedKeys(bucket);
    assert.ok(keys.includes("queue-unreadable-2026-10-07"), JSON.stringify(body));
    assert.equal(keys.some((k) => k.includes("missed-") || k.includes("job-failed-")), false, "no job or slot findings from an unusable queue");
    assert.ok(keys.some((k) => k.includes("heartbeat-stale-desktop-2ki41v6")), "a dead primary is still reported");
    const summary = JSON.parse(bucket.store.get("ops/shadow/summary.json").body);
    assert.deepEqual([summary.checked.jobs, summary.checked.machines], [0, 2]);
  }
  const ok = seeded();
  ok.store.set("jobs/queue.json", { body: JSON.stringify({ schema: "golfprice.job_queue.v1", jobs: [] }), v: 1 });
  await tick(worker, { DASHBOARD_DATA: ok, OPS_SHADOW_ENABLED: "1" }, NOW);
  assert.equal(hasIncident(ok, "queue-unreadable-2026-10-07"), false, "an empty jobs array is a valid queue");
});

/* ------------------------------------------------------------------ review round 4 regressions */

test("invalid queue entries are dropped and reported; valid jobs, heartbeats and the summary still work", async () => {
  const worker = await loadWorker();
  const bucket = seeded();
  const queue = { schema: "golfprice.job_queue.v1", jobs: [null, 5, "x", {}, { id: "j-1" }, failedJob] };
  bucket.store.set("jobs/queue.json", { body: JSON.stringify(queue), v: 1 });
  bucket.store.set("jobs/heartbeat/desktop-2ki41v6.json", { body: JSON.stringify({ machine: "desktop-2ki41v6", at: iso(NOW - 90 * MIN) }), v: 1 });
  await tick(worker, { DASHBOARD_DATA: bucket, OPS_SHADOW_ENABLED: "1" }, NOW);
  const keys = recordedKeys(bucket);
  assert.ok(keys.includes("queue-entry-invalid-2026-10-07"));
  assert.ok(keys.some((k) => k.includes("heartbeat-stale-desktop-2ki41v6")), "stale primary still reported");
  assert.ok(keys.includes(`job-failed-${failedJob.id}`), "the valid job is still checked");
  const summary = JSON.parse(bucket.store.get("ops/shadow/summary.json").body);
  assert.ok(summary.degraded.includes("queue_entry_invalid:5"));
});

/* ------------------------------------------------------------------ round 5: batch + index storage model */
const batchKeys = (bucket) => [...bucket.store.keys()].filter((k) => k.startsWith("ops/shadow/batches/"));
const manyFailed = (count, T0, agoMin = 120) => Array.from({ length: count }, (_, i) => ({ id: `j-20261014T10${String(i).padStart(5, "0")}-m${i}`, type: "wednesday", params: {}, requested_by: "o@example.com", requested_at: iso(T0 - agoMin * MIN), verified: true }));
const queueWith = (jobs, T0) =>
  fakeBucket({
    "jobs/queue.json": { schema: "golfprice.job_queue.v1", jobs },
    ...Object.fromEntries(jobs.map((j) => [`jobs/status/${j.id}.json`, { id: j.id, state: "failed" }])),
    "jobs/heartbeat/desktop-2ki41v6.json": { machine: "desktop-2ki41v6", at: iso(T0 - MIN) },
    "jobs/heartbeat/mckinley_home.json": { machine: "mckinley_home", at: iso(T0 - MIN) },
  });

test("600 findings in one tick all land in ONE batch object with a bounded number of R2 operations; a repeat tick records nothing", async () => {
  const worker = await loadWorker();
  const T0 = Date.parse("2026-10-14T12:00:00Z");
  const jobs = manyFailed(600, T0);
  const bucket = queueWith(jobs, T0);
  const env = { DASHBOARD_DATA: bucket, OPS_SHADOW_ENABLED: "1" };
  await tick(worker, env, T0);
  assert.equal(batchKeys(bucket).length, 1, "one batch, no cap");
  const keys = recordedKeys(bucket);
  for (const j of jobs) assert.ok(keys.includes(`job-failed-${j.id}`), j.id);
  const opsPuts = bucket.puts.filter((k) => k.startsWith("ops/"));
  assert.equal(opsPuts.filter((k) => k.startsWith("ops/shadow/batches/")).length, 1);
  assert.equal(opsPuts.filter((k) => k.startsWith("ops/shadow/index/")).length, 1);
  assert.ok(opsPuts.length <= 5, `ops puts ${opsPuts.length}`);
  assert.equal(bucket.gets.filter((k) => k.startsWith("ops/shadow/index/")).length, 3, "three index GETs");
  assert.equal(bucket.lists.length, 0, "no listing");
  assert.equal(JSON.parse(bucket.store.get("ops/shadow/summary.json").body).new_findings, keys.length);
  bucket.puts.length = 0;
  bucket.gets.length = 0;
  await tick(worker, env, T0 + 2 * MIN);
  assert.equal(batchKeys(bucket).length, 1);
  assert.deepEqual(bucket.puts.filter((k) => k.startsWith("ops/shadow/batches/") || k.startsWith("ops/shadow/index/")), [], "nothing new: no batch, no index write");
  assert.equal(bucket.gets.filter((k) => k.startsWith("ops/shadow/index/")).length, 3);
  assert.equal(bucket.delete_calls, 0, "the observer never deletes");
});

test("overlapping ticks that read the same index etag: both batches are durable, the index is the union, nothing is lost", async () => {
  const worker = await loadWorker();
  const T0 = Date.parse("2026-10-14T12:00:00Z");
  const near = Array.from({ length: 10 }, (_, i) => ({ id: `j-20261012T1210${String(i).padStart(2, "0")}-n${i}`, type: "wednesday", params: {}, requested_by: "o@example.com", requested_at: iso(T0 - (47 * 60 + 50) * MIN), verified: true }));
  const recent = manyFailed(5, T0);
  const bucket = queueWith([...near, ...recent], T0);
  const env = { DASHBOARD_DATA: bucket, OPS_SHADOW_ENABLED: "1" };
  let fired = false;
  bucket.onGet = async (key) => {
    if (key !== "ops/shadow/index/2026-10-14.json" || fired) return;
    fired = true; // tick A has read today's (absent) index; the later tick B now runs to completion and writes first
    await tick(worker, env, T0 + 15 * MIN);
  };
  await tick(worker, env, T0);
  assert.equal(fired, true);
  assert.equal(batchKeys(bucket).length, 2, "both ticks' batches are durable");
  const keys = recordedKeys(bucket);
  for (const j of near) assert.ok(keys.includes(`job-failed-${j.id}`), "A-only finding (job left B's scan window) survives");
  const index = JSON.parse(bucket.store.get("ops/shadow/index/2026-10-14.json").body).keys;
  for (const k of new Set(keys)) assert.ok(index[k], `index has ${k}`);
  assert.ok(batchKeys(bucket).every((b) => Object.values(index).includes(b)), "the index names both batches");
});

test("index CAS exhausted: no finding is lost, the next tick only duplicates it, and summary reports index_conflict", async () => {
  const worker = await loadWorker();
  const T0 = Date.parse("2026-10-14T12:00:00Z");
  const jobs = manyFailed(8, T0);
  const bucket = queueWith(jobs, T0);
  const env = { DASHBOARD_DATA: bucket, OPS_SHADOW_ENABLED: "1" };
  bucket.failPut.add("ops/shadow/index/2026-10-14.json"); // every conditional index update loses
  await tick(worker, env, T0);
  assert.equal(bucket.puts.filter((k) => k === "ops/shadow/index/2026-10-14.json").length, 3, "bounded to three attempts");
  assert.equal(bucket.gets.filter((k) => k === "ops/shadow/index/2026-10-14.json").length, 3, "one read plus two re-reads, none after the last failed PUT");
  assert.equal(batchKeys(bucket).length, 1);
  for (const j of jobs) assert.ok(recordedKeys(bucket).includes(`job-failed-${j.id}`));
  assert.equal(JSON.parse(bucket.store.get("ops/shadow/summary.json").body).index_conflict, true);
  await tick(worker, env, T0 + 15 * MIN);
  assert.equal(batchKeys(bucket).length, 2, "the same keys are recorded again in a new batch");
  const keys = recordedKeys(bucket);
  assert.ok(keys.length > new Set(keys).size, "duplicates only");
  for (const j of jobs) assert.ok(keys.includes(`job-failed-${j.id}`));
  bucket.failPut.delete("ops/shadow/index/2026-10-14.json");
  await tick(worker, env, T0 + 30 * MIN);
  const before = recordedKeys(bucket);
  await tick(worker, env, T0 + 45 * MIN);
  const added = recordedKeys(bucket).slice(before.length);
  assert.ok(added.length > 0 && added.every((k) => !before.includes(k)), "once the index is writable, dedup resumes: only genuinely new keys are recorded");
});

/* ------------------------------------------------------------------ round 6 */
test("a recurring condition on a later day is a new incident: corrupt queue Monday, fixed, corrupt again Tuesday", async () => {
  const worker = await loadWorker();
  const MON = Date.parse("2026-10-26T12:00:00Z");
  const good = { schema: "golfprice.job_queue.v1", jobs: [] };
  const bucket = queueWith([], MON);
  const env = { DASHBOARD_DATA: bucket, OPS_SHADOW_ENABLED: "1" };
  bucket.store.set("jobs/queue.json", { body: "{broken", v: 1 });
  await tick(worker, env, MON);
  bucket.store.set("jobs/queue.json", { body: JSON.stringify(good), v: 2 });
  await tick(worker, env, MON + 3 * 3_600_000);
  bucket.store.set("jobs/queue.json", { body: "{broken again", v: 3 });
  const TUE = MON + 24 * 3_600_000;
  bucket.store.set("jobs/heartbeat/mckinley_home.json", { body: "nope", v: 9 });
  await tick(worker, env, TUE);
  const keys = recordedKeys(bucket);
  assert.ok(keys.includes("queue-unreadable-2026-10-26"));
  assert.ok(keys.includes("queue-unreadable-2026-10-27"), "the Tuesday recurrence is recorded");
  assert.ok(keys.includes("heartbeat-unreadable-mckinley_home-2026-10-27"));
  await tick(worker, env, TUE + 15 * MIN);
  assert.equal(recordedKeys(bucket).filter((k) => k === "queue-unreadable-2026-10-27").length, 1, "but still once per day");
});

test("a corrupt today index is not replaced with only this tick's keys: the batch is written, the index is left alone, duplicates only", async () => {
  const worker = await loadWorker();
  const T0 = Date.parse("2026-10-14T12:00:00Z");
  const jobs = manyFailed(4, T0);
  const bucket = queueWith(jobs, T0);
  const env = { DASHBOARD_DATA: bucket, OPS_SHADOW_ENABLED: "1" };
  bucket.store.set("ops/shadow/index/2026-10-14.json", { body: "corrupt{", v: 1 });
  await tick(worker, env, T0);
  assert.equal(bucket.store.get("ops/shadow/index/2026-10-14.json").body, "corrupt{");
  assert.equal(bucket.puts.includes("ops/shadow/index/2026-10-14.json"), false);
  assert.equal(batchKeys(bucket).length, 1);
  for (const j of jobs) assert.ok(recordedKeys(bucket).includes(`job-failed-${j.id}`));
  assert.ok(JSON.parse(bucket.store.get("ops/shadow/summary.json").body).degraded.includes("index_unreadable:2026-10-14"));
  await tick(worker, env, T0 + 15 * MIN);
  assert.equal(batchKeys(bucket).length, 2, "duplicates, never loss");
});

test("day boundary: a finding recorded on day D is not re-recorded when re-detected on D+2 (three-day index window)", async () => {
  const worker = await loadWorker();
  const D = Date.parse("2026-10-12T23:50:00Z");
  const job = { id: "j-20261012T230000-dddddd", type: "wednesday", params: {}, requested_by: "o@example.com", requested_at: "2026-10-12T23:00:00Z", verified: true };
  const bucket = queueWith([job], D);
  const env = { DASHBOARD_DATA: bucket, OPS_SHADOW_ENABLED: "1" };
  await tick(worker, env, D);
  const D2 = Date.parse("2026-10-14T12:00:00Z"); // job is 37 h old: still scanned; D's index is two days back
  bucket.store.set("jobs/heartbeat/desktop-2ki41v6.json", { body: JSON.stringify({ machine: "desktop-2ki41v6", at: iso(D2 - MIN) }), v: 5 });
  bucket.store.set("jobs/heartbeat/mckinley_home.json", { body: JSON.stringify({ machine: "mckinley_home", at: iso(D2 - MIN) }), v: 5 });
  await tick(worker, env, D2);
  assert.equal(recordedKeys(bucket).filter((k) => k === `job-failed-${job.id}`).length, 1);
});

test("index retry path: valid at first read, the CAS loses, the re-read is corrupt -> no further PUT, object preserved, index_unreadable reported", async () => {
  const worker = await loadWorker();
  const T0 = Date.parse("2026-10-14T12:00:00Z");
  const jobs = manyFailed(3, T0);
  const bucket = queueWith(jobs, T0);
  bucket.store.set("ops/shadow/index/2026-10-14.json", { body: JSON.stringify({ keys: {} }), v: 1 });
  let fired = false;
  bucket.onGet = async (key) => {
    if (key !== "ops/shadow/index/2026-10-14.json" || fired) return;
    fired = true; // after tick A read a valid index: another writer replaces it with garbage (new etag)
    bucket.store.set(key, { body: "corrupt{", v: 7 });
  };
  await tick(worker, { DASHBOARD_DATA: bucket, OPS_SHADOW_ENABLED: "1" }, T0);
  assert.equal(fired, true);
  assert.equal(bucket.puts.filter((k) => k === "ops/shadow/index/2026-10-14.json").length, 1, "one lost PUT, then no overwrite of the corrupt object");
  assert.equal(bucket.store.get("ops/shadow/index/2026-10-14.json").body, "corrupt{");
  assert.equal(batchKeys(bucket).length, 1);
  for (const j of jobs) assert.ok(recordedKeys(bucket).includes(`job-failed-${j.id}`));
  assert.ok(JSON.parse(bucket.store.get("ops/shadow/summary.json").body).degraded.includes("index_unreadable:2026-10-14"));
});
