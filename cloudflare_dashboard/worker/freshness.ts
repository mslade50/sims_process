/**
 * Data-staleness check on the 15-minute cron (owner intent 2026-10-28; rules in worker/freshness-rules.ts).
 *
 * Reads the published objects the dashboard depends on, judges each against the schedule, and writes ONE status object
 * (owner/data_freshness.json, shown as the "Data freshness" panel on the Run page, owner-only through /api/owner-data/). Stale findings go to the
 * ops agent inbox ONLY when DATA_FRESHNESS_INBOX is "1" and AGENT_INBOX is bound (bucket golf-odds-board, prefix ops/agent_inbox/dashboard-freshness/).
 * Both are off by default until the watcher treats source "dashboard-freshness" as internal (repair-only): an external source is escalated to Telegram
 * after 6 h unrecovered. The Worker itself never calls Telegram or any phone path. Nothing here sends a message or calls a network service other than R2.
 *
 * Safety: read-only on every data object. The only writes are the status object and, on a transition (newly stale, re-filed once a day while still stale,
 * recovered), one inbox object. It never touches the jobs queue, the board or the bet card, and it is independent of the other cron work (separate
 * waitUntil in worker/index.ts), time-bounded per read and overall. OFF unless the var DATA_FRESHNESS_ENABLED is exactly "1".
 */
import { inWindow } from "./cron-rules.ts";
import { scorecardLag, historyThrough, type EventLike } from "../app/freshness-rules.ts";
import {
  FRESHNESS_SCHEMA, FRESHNESS_STATUS_KEY, HISTORY_SPEC, HOUR_MS, ODDS_CHECK_MAX_HOURS, SOURCES, carryForward, eventsActive, inboxRecord, judge, oddsCheckEvidence, parseFreshness, runnerSilent, timestampMs,
  type FreshnessDoc, type OddsCheckEvidence, type ReadResult,
} from "./freshness-rules.ts";

export interface FreshnessEnv {
  DASHBOARD_DATA?: R2Bucket;
  /** Bucket the repair watcher lists (golf-odds-board). Not bound in wrangler.jsonc by default: see DATA_FRESHNESS_INBOX. */
  AGENT_INBOX?: R2Bucket;
  /** "1" = also file findings in the agent inbox (needs the AGENT_INBOX binding). Off by default: the watcher treats an unknown source as external and escalates it to a phone after 6 h. */
  DATA_FRESHNESS_INBOX?: string;
  DATA_FRESHNESS_ENABLED?: string;
}

type IndexEvent = EventLike & { history_key?: unknown };
export type FreshnessOutcome = { enabled: boolean; doc: FreshnessDoc | null; notes: string[] };

const READ_TIMEOUT_MS = 8_000;
const TOTAL_TIMEOUT_MS = 25_000;
const HEAD_BYTES = 8_192;
/** Status objects read for the newest odds_reprice jobs (every 30 minutes, so 8 covers four hours). */
const ODDS_STATUS_READS = 8;
const JSON_PUT = { contentType: "application/json; charset=utf-8", cacheControl: "no-store" };

function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_, reject) => { timer = setTimeout(() => reject(new Error(`${label} timed out`)), ms); });
  return Promise.race([promise, timeout]).finally(() => { if (timer) clearTimeout(timer); });
}

/** Leading bytes plus upload time. "absent" only for a real miss; any failure is "error". */
async function readHead(bucket: R2Bucket, key: string): Promise<ReadResult> {
  try {
    const object = await withTimeout(bucket.get(key, { range: { offset: 0, length: HEAD_BYTES } }), READ_TIMEOUT_MS, key);
    if (!object) return { kind: "absent" };
    const head = await withTimeout(object.text(), READ_TIMEOUT_MS, key);
    const uploaded = object.uploaded instanceof Date ? object.uploaded.getTime() : null;
    return { kind: "ok", uploadedMs: uploaded !== null && Number.isFinite(uploaded) ? uploaded : null, head };
  } catch {
    return { kind: "error" };
  }
}

async function readFull(bucket: R2Bucket, key: string): Promise<{ read: ReadResult; json: unknown }> {
  try {
    const object = await withTimeout(bucket.get(key), READ_TIMEOUT_MS, key);
    if (!object) return { read: { kind: "absent" }, json: null };
    const text = await withTimeout(object.text(), READ_TIMEOUT_MS, key);
    let json: unknown = null;
    try { json = JSON.parse(text); } catch { json = null; }
    const uploaded = object.uploaded instanceof Date ? object.uploaded.getTime() : null;
    return { read: { kind: "ok", uploadedMs: uploaded !== null && Number.isFinite(uploaded) ? uploaded : null, head: text.slice(0, HEAD_BYTES) }, json };
  } catch {
    return { read: { kind: "error" }, json: null };
  }
}

/** Newest clean odds_reprice run, from jobs/queue.json and the status object of the newest odds_reprice jobs. Null when the queue cannot be read. */
async function readOddsCheck(bucket: R2Bucket, now: number): Promise<OddsCheckEvidence | null> {
  const queue = await readFull(bucket, "jobs/queue.json");
  if (queue.read.kind === "absent") return { lastSuccessMs: null, jobsSeen: 0 };
  if (queue.read.kind !== "ok") return null;
  const jobs = (queue.json as { jobs?: unknown } | null)?.jobs;
  if (!Array.isArray(jobs)) return null;
  const newest = (jobs as Array<{ id?: unknown; type?: unknown; requested_at?: unknown }>)
    .filter((j) => j && j.type === "odds_reprice" && typeof j.id === "string" && /^[A-Za-z0-9_.-]{1,100}$/.test(j.id))
    .sort((a, b) => (timestampMs(b.requested_at) ?? 0) - (timestampMs(a.requested_at) ?? 0))
    .slice(0, ODDS_STATUS_READS);
  const statuses = new Map<string, unknown>();
  const got = await Promise.all(newest.map(async (job) => ({ id: job.id as string, read: await readFull(bucket, `jobs/status/${job.id}.json`) })));
  for (const g of got) if (g.read.read.kind === "ok") statuses.set(g.id, g.read.json);
  const evidence = oddsCheckEvidence({ jobs: newest }, statuses);
  // A failed status read may hide the newest success: unless a success already inside the limit proves freshness, the evidence is unknown (unreadable, never stale).
  const failedRead = got.some((g) => g.read.read.kind === "error");
  const proven = evidence !== null && evidence.lastSuccessMs !== null && now - evidence.lastSuccessMs <= ODDS_CHECK_MAX_HOURS * HOUR_MS;
  return failedRead && !proven ? null : evidence;
}

/** Runner heartbeat objects (jobs/heartbeat/<machine>.json). Empty (no evidence) when the list fails or any listed heartbeat read fails: a partial list must not show the runner as silent. A heartbeat that reads but is not valid JSON counts as non-live evidence (an empty beat). */
async function readHeartbeats(bucket: R2Bucket): Promise<unknown[]> {
  try {
    const listed = await withTimeout(bucket.list({ prefix: "jobs/heartbeat/", limit: 20 }), READ_TIMEOUT_MS, "heartbeats");
    const got = await Promise.all(listed.objects.map((o) => readFull(bucket, o.key)));
    return got.some((g) => g.read.kind === "error") ? [] : got.map((g) => g.json ?? {});
  } catch {
    return [];
  }
}

async function readStatus(bucket: R2Bucket): Promise<FreshnessDoc | null> {
  try {
    const object = await withTimeout(bucket.get(FRESHNESS_STATUS_KEY), READ_TIMEOUT_MS, FRESHNESS_STATUS_KEY);
    return object ? parseFreshness(JSON.parse(await object.text())) : null;
  } catch {
    return null;
  }
}

export async function runFreshness(env: FreshnessEnv, now: number): Promise<FreshnessOutcome> {
  if (env.DATA_FRESHNESS_ENABLED !== "1") return { enabled: false, doc: null, notes: ["disabled (DATA_FRESHNESS_ENABLED is not 1)"] };
  const bucket = env.DASHBOARD_DATA;
  if (!bucket) return { enabled: true, doc: null, notes: ["dashboard bucket is not bound"] };
  return withTimeout(check(bucket, env.DATA_FRESHNESS_INBOX === "1" ? env.AGENT_INBOX : undefined, now), TOTAL_TIMEOUT_MS, "freshness check").catch((error: unknown) => ({
    enabled: true, doc: null, notes: [`check failed: ${error instanceof Error ? error.message : String(error)}`],
  }));
}

async function check(bucket: R2Bucket, inbox: R2Bucket | undefined, now: number): Promise<FreshnessOutcome> {
  const notes: string[] = [];
  const previous = await readStatus(bucket);

  // Static sources, plus the golfprice index (full read: it names the history documents and feeds the scorecard comparison).
  const heads = await Promise.all(SOURCES.filter((s) => s.id !== "golfprice_index").map(async (spec) => ({ spec, read: await readHead(bucket, spec.key) })));
  const index = await readFull(bucket, "golfprice/index.json");
  const events: EventLike[] | null = Array.isArray((index.json as { events?: unknown } | null)?.events) ? ((index.json as { events: EventLike[] }).events) : null;
  // Off weeks (no event in play or just finished, from the published index) are not judged: nothing is expected to publish.
  const active = eventsActive(events, now);
  const oddsCheck = await readOddsCheck(bucket, now);
  const silent = active === false && runnerSilent(await readHeartbeats(bucket), now);
  const scheduledBreak = active === false && !silent;
  const ctx = { now, inWindow, active, runnerSilent: silent, ...(oddsCheck ? { oddsCheck } : {}) };
  if (silent) notes.push("No event is in play or published ahead, but no job runner is reporting idle or running within the last 6 h: this is not treated as a scheduled break, so sources are judged.");
  else if (scheduledBreak) notes.push("Scheduled inactivity: no event in play, just finished or published ahead in the index (off week or break); sources are not judged.");
  const judged = [
    ...SOURCES.map((spec) => {
      if (spec.id === "golfprice_index") return judge(spec, index.read, ctx);
      return judge(spec, heads.find((h) => h.spec.id === spec.id)!.read, ctx);
    }),
  ];

  // Course history of the newest events: the published `history_key` of each, judged on the document's own source_through_date.
  const historyKeys = ((events ?? []) as IndexEvent[])
    .filter((e): e is IndexEvent & { history_key: string } => typeof e.history_key === "string" && /^golfprice\/history\/[A-Za-z0-9_.-]+\.json$/.test(e.history_key))
    .sort((a, b) => (b.date_start ?? "").localeCompare(a.date_start ?? ""))
    .slice(0, HISTORY_SPEC.max);
  for (const event of historyKeys) {
    const got = await readFull(bucket, event.history_key);
    let read = got.read;
    if (read.kind === "ok") {
      const doc = got.json as { source_through_date?: unknown } | null;
      const through = historyThrough(doc?.source_through_date, []);
      // Only the published source_through_date counts as "data through"; the round dates of past editions are old by nature.
      read = { ...read, head: through ? JSON.stringify({ source_through_date: through }) : "" };
    }
    judged.push(judge({ id: `${HISTORY_SPEC.idPrefix}${event.event_uid}`, label: `course history, ${event.name ?? event.event_uid}`, key: event.history_key, schedule: HISTORY_SPEC.schedule, fields: HISTORY_SPEC.fields, limitHours: HISTORY_SPEC.limitHours }, read, ctx));
  }

  // Scorecard against the newest completed PGA event (not generated_at).
  const card = await readFull(bucket, "golfprice/scorecard/latest.json");
  const cardEvents = (card.json as { events?: Array<{ event_uid?: string; event_start?: string }> } | null)?.events;
  const cardBase = { id: "scorecard_lag", label: "scorecard vs newest completed event", key: "golfprice/scorecard/latest.json", schedule: "weekly" as const, observed_at: null, age_hours: null, limit_hours: null, via: "events" };
  if (card.read.kind === "error") {
    judged.push({ ...cardBase, state: "unreadable", detail: "The storage read failed this tick; no judgment made." });
  } else if (card.read.kind === "ok" && Array.isArray(cardEvents) && events) {
    const lag = scorecardLag(cardEvents.filter((e): e is { event_uid: string; event_start?: string } => typeof e.event_uid === "string"), events, now);
    judged.push({
      ...cardBase,
      state: lag.state === "behind" ? "stale" : lag.state === "current" ? "ok" : "skipped",
      observed_at: lag.newestSettled ? `${lag.newestSettled}T00:00:00.000Z` : null,
      detail: lag.state === "behind" ? lag.message : lag.state === "current" ? "The newest completed event is graded (or inside the settle grace)." : "No completed event to compare.",
    });
  }

  const { statuses, transitions } = carryForward(now, judged, previous?.sources ?? null, [...SOURCES.map((x) => x.id), "scorecard_lag"], { scheduledBreak });
  const nowIso = new Date(now).toISOString();
  const written: string[] = [];
  const failed: string[] = [];
  for (const t of transitions) {
    const record = inboxRecord(t, nowIso);
    if (!inbox) continue; // inbox routing is off: the finding lives in the status object only
    try {
      await withTimeout(inbox.put(record.key, JSON.stringify(record.body), { httpMetadata: JSON_PUT }), READ_TIMEOUT_MS, "inbox put");
      written.push(`${t.kind}:${t.id}`);
      const status = statuses.find((s) => s.id === t.id);
      if (status && t.kind !== "recovered") status.inboxed_at = nowIso;
    } catch {
      failed.push(`${t.id} (write failed)`);
    }
  }
  if (failed.length) notes.push(`inbox not written for: ${failed.join(", ")}`);
  const bad = statuses.filter((s) => s.state === "stale" || s.state === "missing");
  const judgedCount = statuses.filter((s) => s.state === "ok" || s.state === "stale" || s.state === "missing").length;
  if (judgedCount === 0) notes.push("Nothing could be judged this tick (off week, or reads failed).");
  const doc: FreshnessDoc = {
    schema: FRESHNESS_SCHEMA, checked_at: nowIso, ok: bad.length === 0 && judgedCount > 0, not_judged: judgedCount === 0, stale_count: bad.length, sources: statuses, notes,
    inbox: { bound: !!inbox, written, failed },
  };
  try {
    await withTimeout(bucket.put(FRESHNESS_STATUS_KEY, JSON.stringify(doc, null, 1), { httpMetadata: JSON_PUT }), READ_TIMEOUT_MS, "status put");
  } catch (error) {
    notes.push(`status not written: ${error instanceof Error ? error.message : String(error)}`);
  }
  return { enabled: true, doc, notes };
}
