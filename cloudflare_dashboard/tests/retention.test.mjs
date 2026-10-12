import assert from "node:assert/strict";
import test from "node:test";
import { classify, parseConfig, planRetention, handleRetention, retentionDue, bundleContext } from "../worker/retention-rules.ts";

const DAY = 86_400_000;
const NOW = Date.parse("2026-10-10T12:00:00Z");
const jid = (n) => `j-2026090${n}T120000-00000${n}`;
const obj = (key, ageDays, size = 100) => ({ key, size, uploaded: NOW - ageDays * DAY });

test("classify touches only status, logs and claims of well-formed job ids", () => {
  assert.equal(classify(`jobs/status/${jid(1)}.json`).kind, "job_status");
  assert.equal(classify(`jobs/logs/${jid(1)}.txt`).kind, "job_log");
  assert.equal(classify(`jobs/claims/${jid(1)}/desktop-2ki41v6.json`).kind, "job_claim");
  for (const k of ["jobs/queue.json", "jobs/heartbeat/a.json", "jobs/bundles/x/y.zip", "jobs/bundles/index/a.json", "jobs/retention/config.json", "golfprice/latest.json",
    "overrides/active.json", "jobs/status/../queue.json", "jobs/status/notanid.json", "jobs/status/x/j-20260901T120000-000001.json"]) assert.equal(classify(k), null, k);
});

test("plan: 7-day window, newest of each kind and queue-referenced ids survive", () => {
  const objects = [
    obj(`jobs/status/${jid(1)}.json`, 30), obj(`jobs/status/${jid(2)}.json`, 20), obj(`jobs/status/${jid(3)}.json`, 9), obj(`jobs/status/${jid(4)}.json`, 8.9),
    obj(`jobs/status/${jid(5)}.json`, 6), obj(`jobs/logs/${jid(1)}.txt`, 30), obj(`jobs/logs/${jid(2)}.txt`, 10), obj(`jobs/claims/${jid(1)}/a.json`, 30),
    obj("jobs/queue.json", 99), obj("jobs/heartbeat/a.json", 99), obj("golfprice/latest.json", 99),
  ];
  const plan = planRetention(objects, new Set([jid(2)]), NOW, 7);
  const keys = plan.delete.map((o) => o.key);
  assert.deepEqual(keys, [`jobs/logs/${jid(1)}.txt`, `jobs/status/${jid(1)}.json`, `jobs/status/${jid(3)}.json`, `jobs/status/${jid(4)}.json`]);
  assert.ok(!keys.includes(`jobs/status/${jid(2)}.json`), "queue-referenced id kept");
  assert.ok(!keys.includes(`jobs/status/${jid(5)}.json`), "inside the window kept");
  assert.ok(!keys.includes(`jobs/logs/${jid(2)}.txt`), "newest log kept even though old");
  assert.ok(!keys.includes(`jobs/claims/${jid(1)}/a.json`), "newest (only) claim kept");
  assert.ok(!keys.some((k) => !k.startsWith("jobs/status/") && !k.startsWith("jobs/logs/") && !k.startsWith("jobs/claims/")));
  assert.equal(plan.unclassified_ignored, 3);
  assert.equal(plan.summary.count, keys.length);
  assert.equal(plan.summary.bytes, keys.length * 100);
});

test("plan never goes under 7 days and the config parser fails closed", () => {
  const objects = [obj(`jobs/status/${jid(1)}.json`, 3), obj(`jobs/status/${jid(2)}.json`, 2)];
  assert.equal(planRetention(objects, new Set(), NOW, 1).summary.count, 0);
  assert.equal(parseConfig(null).mode, "off");
  assert.equal(parseConfig("{oops").mode, "off");
  assert.equal(parseConfig('{"mode":"delete-everything"}').mode, "off");
  assert.equal(parseConfig('{"mode":"apply","days":1}').days, 7);
  assert.equal(parseConfig('{"mode":"dry-run"}').mode, "dry-run");
});

function fakeBucket(files) {
  const store = new Map(Object.entries(files).map(([k, v]) => [k, { text: v.text ?? "{}", size: v.size ?? 100, uploaded: new Date(v.uploaded ?? NOW - 30 * DAY) }]));
  const deleted = [];
  return {
    store, deleted,
    async get(key) { const o = store.get(key); return o ? { text: async () => o.text } : null; },
    async put(key, body) { store.set(key, { text: String(body), size: String(body).length, uploaded: new Date(NOW) }); return {}; },
    async delete(keys) { for (const k of [].concat(keys)) { deleted.push(k); store.delete(k); } },
    async list({ prefix }) { return { objects: [...store].filter(([k]) => k.startsWith(prefix)).map(([key, o]) => ({ key, size: o.size, uploaded: o.uploaded })), truncated: false }; },
  };
}
const files = () => ({
  "jobs/queue.json": { text: JSON.stringify({ jobs: [{ id: jid(2) }] }) },
  "jobs/heartbeat/a.json": { text: JSON.stringify({ job_id: jid(3) }) },
  [`jobs/status/${jid(1)}.json`]: {}, [`jobs/status/${jid(2)}.json`]: {}, [`jobs/status/${jid(3)}.json`]: {}, [`jobs/status/${jid(4)}.json`]: { uploaded: NOW - 29 * DAY },
  [`jobs/status/${jid(5)}.json`]: { uploaded: NOW - 40 * DAY },
});

test("handler is OFF without a config object and deletes nothing in dry-run", async () => {
  const b = fakeBucket(files());
  assert.equal((await handleRetention({ scheduledTime: NOW }, { DASHBOARD_DATA: b }, true)).ran, false);
  b.store.set("jobs/retention/config.json", { text: '{"mode":"dry-run"}', size: 1, uploaded: new Date(NOW) });
  const out = await handleRetention({ scheduledTime: NOW }, { DASHBOARD_DATA: b }, true);
  assert.equal(out.ran, true);
  assert.equal(b.deleted.length, 0);
  const report = JSON.parse(b.store.get("jobs/retention/last_report.json").text);
  assert.equal(report.mode, "dry-run");
  assert.deepEqual(report.delete_keys.sort(), [`jobs/status/${jid(1)}.json`, `jobs/status/${jid(5)}.json`]);
});

test("apply deletes exactly the planned keys, records them first, keeps protected ones", async () => {
  const b = fakeBucket(files());
  b.store.set("jobs/retention/config.json", { text: '{"mode":"apply"}', size: 1, uploaded: new Date(NOW) });
  const out = await handleRetention({ scheduledTime: NOW }, { DASHBOARD_DATA: b }, true);
  assert.equal(out.ran, true);
  assert.ok(!b.deleted.includes(`jobs/status/${jid(2)}.json`), "queue id");
  assert.ok(!b.deleted.includes(`jobs/status/${jid(3)}.json`), "heartbeat job id");
  assert.ok(!b.deleted.includes(`jobs/status/${jid(4)}.json`), "newest status");
  assert.deepEqual(b.deleted.sort(), [`jobs/status/${jid(1)}.json`, `jobs/status/${jid(5)}.json`]);
  assert.ok([...b.store.keys()].some((k) => k.startsWith("jobs/retention/applied/")), "applied record written");
  assert.ok(b.store.has("jobs/queue.json") && b.store.has("jobs/heartbeat/a.json"));
});

test("refuses to plan without a valid queue index", async () => {
  const b = fakeBucket({ ...files(), "jobs/queue.json": { text: "not json" } });
  b.store.set("jobs/retention/config.json", { text: '{"mode":"apply"}', size: 1, uploaded: new Date(NOW) });
  await assert.rejects(handleRetention({ scheduledTime: NOW }, { DASHBOARD_DATA: b }, true));
  assert.equal(b.deleted.length, 0);
});

test("daily slot is 03:15 New York time", () => {
  assert.equal(retentionDue(Date.parse("2026-10-10T07:15:00Z")), true); // EDT
  assert.equal(retentionDue(Date.parse("2026-11-10T08:15:00Z")), true); // EST
  assert.equal(retentionDue(Date.parse("2026-10-10T07:30:00Z")), false);
});

// ------------------------------------------------------------------------------------------------ bundles (jobs/bundles/<24 hex>/...)
const bid = (n) => String(n).padStart(2, "0").repeat(12).replace(/[^0-9a-f]/g, "a"); // 24 hex chars
const bundleObjs = (n, ageDays, parts = 2) => [
  obj(`jobs/bundles/${bid(n)}/manifest.json`, ageDays, 50),
  ...Array.from({ length: parts }, (_, i) => obj(`jobs/bundles/${bid(n)}/part-${String(i).padStart(4, "0")}.b64`, ageDays + 0.001, 1000)),
];
const ctxOf = (jobs, types, announced = []) => ({ jobOf: new Map(Object.entries(jobs)), typeOf: new Map(Object.entries(types)), announced: new Set(announced) });

test("classify: bundle manifest and parts only; the index and malformed ids stay untouched", () => {
  assert.equal(classify(`jobs/bundles/${bid(1)}/manifest.json`).kind, "bundle");
  assert.equal(classify(`jobs/bundles/${bid(1)}/part-0003.b64`).id, bid(1));
  for (const k of ["jobs/bundles/index/desktop.json", "jobs/bundles/index/x/manifest.json", `jobs/bundles/${bid(1)}.json`, `jobs/bundles/${bid(1)}/other.txt`,
    "jobs/bundles/abc/manifest.json", `jobs/bundles/${bid(1)}/../manifest.json`, `jobs/bundles/${bid(1)}/sub/manifest.json`]) assert.equal(classify(k), null, k);
});

test("bundles: 7-day window as a whole, whole-bundle deletion with the manifest first", () => {
  const objects = [...bundleObjs(1, 20), ...bundleObjs(2, 9), ...bundleObjs(3, 6), ...bundleObjs(4, 1)];
  const jobs = { [bid(1)]: jid(1), [bid(2)]: jid(2), [bid(3)]: jid(3), [bid(4)]: jid(4) };
  const types = { [jid(1)]: "a", [jid(2)]: "a", [jid(3)]: "a", [jid(4)]: "a" };
  const plan = planRetention(objects, new Set(), NOW, 7, ctxOf(jobs, types));
  const keys = plan.delete.map((o) => o.key);
  assert.deepEqual(keys, [1, 2].flatMap((n) => [`jobs/bundles/${bid(n)}/manifest.json`, `jobs/bundles/${bid(n)}/part-0000.b64`, `jobs/bundles/${bid(n)}/part-0001.b64`]));
  assert.equal(plan.bundles.delete_bundles, 2);
  assert.equal(plan.bundles.delete_objects, 6);
  assert.equal(plan.bundles.delete_bytes, 2 * 2050);
  assert.equal(plan.summary.count, 6);
  assert.equal(plan.bundles.kept_newest, bid(4));
});

test("bundles: a bundle with one recent object is inside the window", () => {
  const objects = [...bundleObjs(1, 20), obj(`jobs/bundles/${bid(1)}/part-0002.b64`, 2), ...bundleObjs(2, 1)];
  const plan = planRetention(objects, new Set(), NOW, 7, ctxOf({ [bid(1)]: jid(1), [bid(2)]: jid(2) }, { [jid(1)]: "a", [jid(2)]: "a" }));
  assert.equal(plan.delete.length, 0);
});

test("bundles: keep the newest, announced, queue-referenced and latest-of-each-job-type", () => {
  const all = [1, 2, 3, 4, 5, 6, 7].flatMap((n) => bundleObjs(n, 30 - n)); // bundle 7 is the newest (23 days old)
  const jobs = Object.fromEntries([1, 2, 3, 4, 5, 6, 7].map((n) => [bid(n), jid(n)]));
  const types = { [jid(1)]: "tuesday", [jid(2)]: "tuesday", [jid(3)]: "publish", [jid(4)]: "publish", [jid(5)]: "odds", [jid(6)]: "odds", [jid(7)]: "odds" };
  // 7 = newest overall (and latest odds); 2 = announced (and latest tuesday); 3 = queue-referenced; 4 = latest publish; so 1, 5 and 6 go
  const plan = planRetention(all, new Set([jid(3)]), NOW, 7, ctxOf(jobs, types, [bid(2)]));
  assert.deepEqual([...new Set(plan.delete.map((o) => classify(o.key).id))].sort(), [bid(1), bid(5), bid(6)].sort());
  assert.equal(plan.bundles.protected_referenced, 1);
  assert.equal(plan.bundles.protected_latest_of_type, 1); // bundle 4
  assert.equal(plan.bundles.protected_announced, 2); // bundle 7 (newest) and bundle 2
});

test("bundles: fail safe when the job or its type is unknown, and when no context is supplied", () => {
  const objects = [...bundleObjs(1, 30), ...bundleObjs(2, 29), ...bundleObjs(3, 28)];
  const unknownJob = planRetention(objects, new Set(), NOW, 7, ctxOf({ [bid(1)]: null, [bid(2)]: jid(2), [bid(3)]: jid(3) }, { [jid(2)]: null, [jid(3)]: "a" }));
  assert.equal(unknownJob.delete.length, 0);
  assert.equal(unknownJob.bundles.protected_unresolved, 2);
  assert.equal(planRetention(objects, new Set(), NOW, 7).delete.length, 0);
});

test("bundles: an orphan (no manifest) older than the window goes; a young one stays", () => {
  const orphanOld = [obj(`jobs/bundles/${bid(1)}/part-0000.b64`, 20)];
  const orphanNew = [obj(`jobs/bundles/${bid(2)}/part-0000.b64`, 1)];
  const good = bundleObjs(3, 0.5);
  const plan = planRetention([...orphanOld, ...orphanNew, ...good], new Set(), NOW, 7, ctxOf({ [bid(3)]: jid(3) }, { [jid(3)]: "a" }));
  assert.deepEqual(plan.delete.map((o) => o.key), [`jobs/bundles/${bid(1)}/part-0000.b64`]);
});

test("bundles: window never shorter than 7 days", () => {
  const plan = planRetention([...bundleObjs(1, 3), ...bundleObjs(2, 2)], new Set(), NOW, 1, ctxOf({ [bid(1)]: jid(1), [bid(2)]: jid(2) }, { [jid(1)]: "a", [jid(2)]: "a" }));
  assert.equal(plan.delete.length, 0);
});

const bundleFiles = () => {
  const f = {
    "jobs/queue.json": { text: JSON.stringify({ jobs: [{ id: jid(5), type: "publish" }] }) },
    "jobs/heartbeat/desktop.json": { text: JSON.stringify({ job_id: null, last_bundle: bid(4) }) },
    "jobs/bundles/index/desktop.json": { text: JSON.stringify({ machine: "desktop", bundles: [1, 2, 3, 4].map((n) => ({ bundle_id: bid(n), job_id: jid(n) })) }), uploaded: NOW },
  };
  for (const n of [1, 2, 3, 4, 6]) {
    f[`jobs/bundles/${bid(n)}/manifest.json`] = { text: JSON.stringify({ job_id: jid(n) }), size: 50, uploaded: NOW - (40 - n) * DAY };
    f[`jobs/bundles/${bid(n)}/part-0000.b64`] = { text: "x", size: 1000, uploaded: NOW - (40 - n) * DAY };
    f[`jobs/status/${jid(n)}.json`] = { text: JSON.stringify({ job_type: n <= 2 ? "tuesday" : n === 3 ? "publish" : "odds" }), uploaded: NOW }; // recent: not deleted
  }
  return f;
};

test("handler context: index, manifests, statuses and heartbeats decide; apply deletes whole old bundles only", async () => {
  const b = fakeBucket(bundleFiles());
  const ctx = await bundleContext(b, [...b.store.keys()].filter((k) => k.startsWith("jobs/bundles/") && !k.includes("/index/")).map((key) => ({ key, size: 1, uploaded: NOW })));
  assert.equal(ctx.jobOf.get(bid(6)), jid(6)); // not in the index: found through its manifest
  assert.equal(ctx.typeOf.get(jid(3)), "publish");
  assert.ok(ctx.announced.has(bid(4)), "index tail and heartbeat last_bundle");
  b.store.set("jobs/retention/config.json", { text: '{"mode":"apply"}', size: 1, uploaded: new Date(NOW) });
  const out = await handleRetention({ scheduledTime: NOW }, { DASHBOARD_DATA: b }, true);
  assert.equal(out.ran, true);
  // bundle 6 is newest overall; 4 announced; 3 = latest publish job; 2 = latest tuesday job -> only bundle 1 (an older tuesday job) goes
  const bundleDeleted = b.deleted.filter((k) => k.startsWith("jobs/bundles/"));
  assert.deepEqual(bundleDeleted, [`jobs/bundles/${bid(1)}/manifest.json`, `jobs/bundles/${bid(1)}/part-0000.b64`]);
  assert.ok(b.store.has("jobs/bundles/index/desktop.json"), "index untouched");
  assert.ok(b.store.has("jobs/queue.json"));
});

test("dry-run mode deletes no bundle and reports the bundle summary", async () => {
  const b = fakeBucket(bundleFiles());
  b.store.set("jobs/retention/config.json", { text: '{"mode":"dry-run"}', size: 1, uploaded: new Date(NOW) });
  await handleRetention({ scheduledTime: NOW }, { DASHBOARD_DATA: b }, true);
  assert.equal(b.deleted.length, 0);
  const report = JSON.parse(b.store.get("jobs/retention/last_report.json").text);
  assert.equal(report.bundles.delete_bundles, 1);
  assert.ok(report.delete_keys.every((k) => !k.startsWith("jobs/bundles/index/")));
});
