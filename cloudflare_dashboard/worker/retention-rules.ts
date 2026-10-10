/**
 * Retention rules for the job-queue objects in the dashboard bucket (owner approval of a 7-day window, Q11, October 10, 2026).
 * Pure TypeScript, no imports: shared by worker/retention.ts, scripts/retention-dryrun.mjs and tests/retention.test.mjs.
 *
 * Scope (the only keys this can ever touch):
 *   jobs/status/<job id>.json        job_status   (written by the runner)
 *   jobs/logs/<job id>.txt           job_log      (written by the runner)
 *   jobs/claims/<job id>/<m>.json    job_claim    (failover lease claims, written by the runners)
 * Never touched, by construction (they are not classified): jobs/queue.json, jobs/heartbeat/**, jobs/bundles/** (ledger bundles and their index),
 * jobs/retention/** (this mechanism's own config and reports), golfprice/** (published run data, pointers, indexes, manifests), overrides/**,
 * and anything else. Run folders, frozen/ and CHAIN files are not in this bucket's jobs/ area and are not classified either.
 *
 * Protected even when older than the window:
 *   - the newest object of each kind (by upload time, then key);
 *   - every object whose job id is still listed in jobs/queue.json (the current index, newest 200 jobs);
 *   - every object whose job id is in a heartbeat's job_id (a job a runner is working on right now).
 *
 * Mode lives in the R2 object jobs/retention/config.json: {"mode":"dry-run"|"apply","days":7}. A missing or unreadable file means OFF.
 */
export const RETENTION_DAYS = 7;
export const RETENTION_CONFIG_KEY = "jobs/retention/config.json";
export const RETENTION_REPORT_KEY = "jobs/retention/last_report.json";
export const RETENTION_APPLIED_PREFIX = "jobs/retention/applied/";
/** A single apply run deletes at most this many objects (the rest waits for the next daily run); a safety valve, not a policy. */
export const MAX_DELETES_PER_RUN = 500;
/** Eastern wall-clock minute of the daily run (the 15-minute cron trigger lands on it exactly). */
export const RETENTION_SLOT = "03:15";

export type RetentionMode = "off" | "dry-run" | "apply";
export type Kind = "job_status" | "job_log" | "job_claim";
export type StoredObject = { key: string; size: number; uploaded: number };
export type Classified = { kind: Kind; id: string };

export const KIND_PREFIXES: Record<Kind, string> = { job_status: "jobs/status/", job_log: "jobs/logs/", job_claim: "jobs/claims/" };

const ID = "j-\\d{8}T\\d{6}-[0-9a-f]{6}";
const PATTERNS: [Kind, RegExp][] = [
  ["job_status", new RegExp(`^jobs/status/(${ID})\\.json$`)],
  ["job_log", new RegExp(`^jobs/logs/(${ID})\\.txt$`)],
  ["job_claim", new RegExp(`^jobs/claims/(${ID})/[A-Za-z0-9._-]{1,64}\\.json$`)],
];

export function classify(key: string): Classified | null {
  for (const [kind, re] of PATTERNS) {
    const m = re.exec(key);
    if (m) return { kind, id: m[1] };
  }
  return null;
}

export function parseConfig(text: string | null | undefined): { mode: RetentionMode; days: number; note?: string } {
  if (!text) return { mode: "off", days: RETENTION_DAYS, note: "no config object: off" };
  try {
    const v = JSON.parse(text) as { mode?: unknown; days?: unknown };
    const mode: RetentionMode = v.mode === "apply" ? "apply" : v.mode === "dry-run" ? "dry-run" : "off";
    const days = typeof v.days === "number" && Number.isFinite(v.days) ? Math.max(7, Math.floor(v.days)) : RETENTION_DAYS; // never shorter than 7 days
    return { mode, days };
  } catch {
    return { mode: "off", days: RETENTION_DAYS, note: "config is not valid JSON: off" };
  }
}

export type KindSummary = {
  kind: Kind;
  scanned: number;
  scanned_bytes: number;
  older_than_window: number;
  kept_newest: string | null;
  protected_referenced: number;
  delete_count: number;
  delete_bytes: number;
};
export type Plan = {
  schema: "golfprice.retention_plan.v1";
  now: string;
  days: number;
  cutoff: string;
  delete: StoredObject[];
  summary: { count: number; bytes: number; oldest: string | null; newest: string | null };
  kinds: KindSummary[];
  unclassified_ignored: number;
};

const iso = (ms: number) => new Date(ms).toISOString().replace(/\.\d{3}Z$/, "Z");

/** Decide what a retention run would delete. `protectedIds` = job ids still in jobs/queue.json plus job ids named by heartbeats. */
export function planRetention(objects: StoredObject[], protectedIds: Set<string>, now: number, days = RETENTION_DAYS): Plan {
  const effectiveDays = Math.max(7, days);
  const cutoff = now - effectiveDays * 86_400_000;
  const byKind = new Map<Kind, { obj: StoredObject; id: string }[]>();
  let unclassified = 0;
  for (const obj of objects) {
    const c = classify(obj.key);
    if (!c) {
      unclassified += 1;
      continue;
    }
    byKind.set(c.kind, [...(byKind.get(c.kind) ?? []), { obj, id: c.id }]);
  }
  const del: StoredObject[] = [];
  const kinds: KindSummary[] = [];
  for (const kind of Object.keys(KIND_PREFIXES) as Kind[]) {
    const rows = byKind.get(kind) ?? [];
    const newest = rows.reduce<{ obj: StoredObject; id: string } | null>(
      (best, r) => (!best || r.obj.uploaded > best.obj.uploaded || (r.obj.uploaded === best.obj.uploaded && r.obj.key > best.obj.key) ? r : best),
      null,
    );
    let older = 0;
    let referenced = 0;
    const picked: StoredObject[] = [];
    for (const r of rows) {
      if (r.obj.uploaded >= cutoff) continue;
      older += 1;
      if (newest && r.obj.key === newest.obj.key) continue;
      if (protectedIds.has(r.id)) {
        referenced += 1;
        continue;
      }
      picked.push(r.obj);
    }
    del.push(...picked);
    kinds.push({
      kind,
      scanned: rows.length,
      scanned_bytes: rows.reduce((a, r) => a + r.obj.size, 0),
      older_than_window: older,
      kept_newest: newest?.obj.key ?? null,
      protected_referenced: referenced,
      delete_count: picked.length,
      delete_bytes: picked.reduce((a, o) => a + o.size, 0),
    });
  }
  del.sort((a, b) => a.uploaded - b.uploaded || (a.key < b.key ? -1 : 1));
  return {
    schema: "golfprice.retention_plan.v1",
    now: iso(now),
    days: effectiveDays,
    cutoff: iso(cutoff),
    delete: del,
    summary: { count: del.length, bytes: del.reduce((a, o) => a + o.size, 0), oldest: del.length ? iso(del[0].uploaded) : null, newest: del.length ? iso(del[del.length - 1].uploaded) : null },
    kinds,
    unclassified_ignored: unclassified,
  };
}

// ---------------------------------------------------------------------------------------------- R2 run (uses the Worker binding, no credential)
export interface RetentionEnv {
  DASHBOARD_DATA?: R2Bucket;
}

/** True at 03:15 America/New_York (Intl, so daylight saving is the time zone database's business). */
export function retentionDue(ms: number): boolean {
  const parts = new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York", hour: "2-digit", minute: "2-digit", hourCycle: "h23" }).formatToParts(new Date(ms));
  const pick = (type: string) => parts.find((p) => p.type === type)?.value ?? "";
  return `${pick("hour")}:${pick("minute")}` === RETENTION_SLOT;
}

async function listAll(bucket: R2Bucket, prefix: string): Promise<StoredObject[]> {
  const out: StoredObject[] = [];
  let cursor: string | undefined;
  for (let page = 0; page < 200; page += 1) {
    const res = await bucket.list({ prefix, limit: 1000, cursor });
    for (const o of res.objects) out.push({ key: o.key, size: o.size, uploaded: o.uploaded.getTime() });
    if (!res.truncated) return out;
    cursor = res.cursor;
  }
  throw new Error(`listing ${prefix} did not finish in 200 pages`);
}

/** Job ids that must never be deleted: the current queue index and the job each runner heartbeat names. */
export async function protectedJobIds(bucket: R2Bucket): Promise<Set<string>> {
  const ids = new Set<string>();
  const queue = await bucket.get("jobs/queue.json");
  if (!queue) throw new Error("jobs/queue.json is missing: refusing to plan a deletion without the current index");
  const parsed = JSON.parse(await queue.text()) as { jobs?: { id?: string }[] };
  if (!Array.isArray(parsed.jobs)) throw new Error("jobs/queue.json is not a valid queue: refusing to plan a deletion");
  for (const j of parsed.jobs) if (typeof j.id === "string") ids.add(j.id);
  for (const o of (await bucket.list({ prefix: "jobs/heartbeat/", limit: 100 })).objects) {
    const hb = await bucket.get(o.key);
    try {
      const id = (JSON.parse(await hb!.text()) as { job_id?: unknown }).job_id;
      if (typeof id === "string") ids.add(id);
    } catch {
      /* an unreadable heartbeat protects nothing extra */
    }
  }
  return ids;
}

export type RetentionOutcome = { ran: boolean; mode: string; note: string; deleted?: number };

export async function handleRetention(controller: { scheduledTime: number }, env: RetentionEnv, force = false): Promise<RetentionOutcome> {
  const bucket = env.DASHBOARD_DATA;
  if (!bucket) return { ran: false, mode: "off", note: "dashboard bucket is not bound" };
  if (!force && !retentionDue(controller.scheduledTime)) return { ran: false, mode: "off", note: "not the retention slot" };
  const cfgObj = await bucket.get(RETENTION_CONFIG_KEY);
  const cfg = parseConfig(cfgObj ? await cfgObj.text() : null);
  if (cfg.mode === "off") return { ran: false, mode: "off", note: cfg.note ?? "mode off" };
  const now = controller.scheduledTime;
  const objects = (await Promise.all(Object.values(KIND_PREFIXES).map((p) => listAll(bucket, p)))).flat().filter((o) => classify(o.key));
  const plan = planRetention(objects, await protectedJobIds(bucket), now, cfg.days);
  const toDelete = cfg.mode === "apply" ? plan.delete.slice(0, MAX_DELETES_PER_RUN) : [];
  const json = { httpMetadata: { contentType: "application/json; charset=utf-8", cacheControl: "no-store" } };
  const report = { ...plan, mode: cfg.mode, delete_keys: plan.delete.map((o) => o.key), deleted_this_run: toDelete.length };
  await bucket.put(RETENTION_REPORT_KEY, JSON.stringify(report, null, 1), json);
  if (toDelete.length) {
    const stamp = new Date(now).toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z");
    await bucket.put(`${RETENTION_APPLIED_PREFIX}${stamp}.json`, JSON.stringify({ at: plan.now, days: plan.days, keys: toDelete.map((o) => o.key) }, null, 1), json);
    for (let i = 0; i < toDelete.length; i += 1000) await bucket.delete(toDelete.slice(i, i + 1000).map((o) => o.key));
  }
  return { ran: true, mode: cfg.mode, note: `${plan.summary.count} planned, ${toDelete.length} deleted`, deleted: toDelete.length };
}
