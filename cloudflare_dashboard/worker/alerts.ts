/**
 * Scheduled-tick alerting (E5, shipped DARK on October 9, 2026): stale-primary, failed-job and daily-digest checks.
 *
 * With TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID absent this module only reads (queue, statuses, heartbeats), logs what it would send, and writes
 * nothing: no state object, no fetch. Setting both Worker secrets (owner action) turns sending on; no code change is needed.
 *
 * Sending is reserve-then-send against jobs/alerts/state.json (etag-guarded, per-key 24 h cooldown), the telegram_policy.py semantics: a lost
 * race or failed reservation means no page, never a duplicate; a failed send leaves only the reservation, so the next attempt waits an hour.
 */
import { easternClock, hhmm } from "./cron-rules";
import {
  ALERT_STATE_KEY,
  dueAlerts,
  emptyAlertState,
  parseAlertState,
  seedState,
  pruneState,
  shouldAlert,
  type Alert,
  type AlertState,
  type JobLike,
} from "./alerts-rules";
import { HEARTBEAT_PREFIX, MACHINES, QUEUE_KEY, STATUS_PREFIX, effectiveState, listWindow, parseUtc, type JobRecord, type JobStatus } from "../app/jobs-rules";

export interface AlertEnv {
  DASHBOARD_DATA?: R2Bucket;
  TELEGRAM_BOT_TOKEN?: string;
  TELEGRAM_CHAT_ID?: string;
}

export type AlertOutcome = { enabled: boolean; evaluated: Alert[]; sent: string[]; suppressed: string[]; notes: string[] };

const JSON_PUT = { contentType: "application/json; charset=utf-8", cacheControl: "no-store" };

async function readJson(bucket: R2Bucket, key: string): Promise<{ value: unknown; etag: string | null }> {
  const object = await bucket.get(key);
  if (!object) return { value: null, etag: null };
  try {
    return { value: JSON.parse(await object.text()), etag: object.etag };
  } catch {
    return { value: null, etag: object.etag };
  }
}

async function readState(bucket: R2Bucket): Promise<{ state: AlertState; etag: string | null }> {
  const got = await readJson(bucket, ALERT_STATE_KEY);
  return { state: parseAlertState(got.value), etag: got.etag };
}

/** Etag-guarded replace of the state object; false when another tick changed it first. */
async function writeState(bucket: R2Bucket, state: AlertState, etag: string | null): Promise<boolean> {
  const result = await bucket.put(ALERT_STATE_KEY, JSON.stringify(state, null, 1), {
    httpMetadata: JSON_PUT,
    onlyIf: etag ? { etagMatches: etag } : { etagDoesNotMatch: "*" },
  });
  return result !== null;
}

async function sendTelegram(token: string, chat: string, text: string, fetcher: typeof fetch): Promise<boolean> {
  const response = await fetcher(`https://api.telegram.org/bot${token}/sendMessage`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ chat_id: chat, text: text.slice(0, 3900), disable_web_page_preview: true }),
  });
  return response.status === 200;
}

async function collect(bucket: R2Bucket, now: number) {
  const queue = (await readJson(bucket, QUEUE_KEY)).value as { jobs?: unknown } | null;
  const jobs = Array.isArray(queue?.jobs) ? (queue.jobs as JobRecord[]) : [];
  const listed = listWindow(jobs, now);
  const likes: JobLike[] = await Promise.all(
    listed.map(async (job) => {
      const status = (await readJson(bucket, `${STATUS_PREFIX}${job.id}.json`).catch(() => ({ value: null }))).value as JobStatus | null;
      return { id: job.id, type: job.type, requested_at: job.requested_at, state: effectiveState(job, status && typeof status === "object" ? status : null) };
    }),
  );
  const beat = async (name: string) => (await readJson(bucket, `${HEARTBEAT_PREFIX}${name}.json`).catch(() => ({ value: null }))).value as { at?: string } | null;
  const primary = MACHINES.find((m) => m.role === "primary");
  const backup = MACHINES.find((m) => m.role === "backup");
  return {
    recentJobs: likes.filter((j) => now - (parseUtc(j.requested_at) ?? 0) <= 48 * 3_600_000),
    jobs24h: likes.filter((j) => now - (parseUtc(j.requested_at) ?? 0) <= 24 * 3_600_000),
    primary: primary ? await beat(primary.name) : null,
    backup: backup ? await beat(backup.name) : null,
  };
}

export async function runAlerts(env: AlertEnv, now: number, fetcher: typeof fetch = fetch): Promise<AlertOutcome> {
  const outcome: AlertOutcome = { enabled: false, evaluated: [], sent: [], suppressed: [], notes: [] };
  const bucket = env.DASHBOARD_DATA;
  if (!bucket) {
    outcome.notes.push("dashboard bucket is not bound");
    return outcome;
  }
  const token = env.TELEGRAM_BOT_TOKEN?.trim();
  const chat = env.TELEGRAM_CHAT_ID?.trim();
  outcome.enabled = !!token && !!chat;
  const clock = easternClock(now);
  const etDate = new Date(now).toLocaleDateString("en-CA", { timeZone: "America/New_York" });

  const data = await collect(bucket, now);
  // Dark mode never touches the state object; the stored state only matters once sending is on.
  const read = outcome.enabled ? await readState(bucket) : { state: emptyAlertState(), etag: null as string | null };
  let initial = read.state;
  const initialEtag = read.etag;
  if (outcome.enabled && !initial.seeded) {
    // First-run guard: record the failed jobs that already exist so switching the secrets on does not page a backlog of old failures.
    const seeded: AlertState = { ...initial, seeded: seedState(data.recentJobs, now).seeded };
    if (await writeState(bucket, seeded, initialEtag)) {
      initial = seeded;
      outcome.notes.push(`alert state seeded; ${seeded.seeded?.failed.length ?? 0} existing failed job(s) will not page`);
    } else {
      outcome.notes.push("alert state seeding lost a race; evaluating nothing this tick");
      return outcome;
    }
  }
  outcome.evaluated = dueAlerts({ ...data, state: initial, now, etHHMM: hhmm(clock), etDate });

  if (!outcome.enabled || !token || !chat) {
    if (outcome.evaluated.length) outcome.notes.push(`telegram not configured; would send: ${outcome.evaluated.map((a) => a.key).join(", ")}`);
    return outcome;
  }

  for (const alert of outcome.evaluated) {
    // Reserve (etag-guarded; the recovery note is already gated by the outage flag), send, then record the send.
    const { state, etag } = await readState(bucket);
    const recovered = alert.kind === "p1_recovered";
    if (!recovered && !shouldAlert(state.keys[alert.key], now)) {
      outcome.suppressed.push(alert.key);
      continue;
    }
    if (recovered && !state.p1_outage) continue;
    const reserved: AlertState = pruneState({ ...state, keys: { ...state.keys, [alert.key]: { ...state.keys[alert.key], reserved_at: now } } }, now);
    if (!(await writeState(bucket, reserved, etag))) {
      outcome.suppressed.push(`${alert.key} (reservation lost)`);
      continue;
    }
    let delivered = false;
    try {
      delivered = await sendTelegram(token, chat, alert.text, fetcher);
    } catch {
      delivered = false;
    }
    if (!delivered) {
      outcome.notes.push(`telegram send failed for ${alert.key}`);
      continue;
    }
    outcome.sent.push(alert.key);
    const latest = await readState(bucket);
    const next: AlertState = { ...latest.state, keys: { ...latest.state.keys, [alert.key]: { sent_at: now } } };
    if (alert.kind === "p1") next.p1_outage = { since: now };
    if (recovered) {
      next.p1_outage = null;
      delete next.keys["p1-primary-down"];
      delete next.keys[alert.key];
    }
    if (!(await writeState(bucket, next, latest.etag))) outcome.notes.push(`alert state write failed after sending ${alert.key}`);
  }
  return outcome;
}
