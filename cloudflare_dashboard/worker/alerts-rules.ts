/**
 * Alert rules for the Worker's scheduled tick (E5, site audit October 8, 2026). Pure TypeScript with no imports, so tests can load it directly.
 *
 * Only the Worker is independent of both desktops, so it is the one place that can report the always-on desktop's death. Semantics are ported
 * from sims_process/telegram_policy.py: per-key state {sent_at, reserved_at}; an alert is allowed when the last send is at least 24 h old AND no
 * reservation was made in the last hour (reserve first, then send, then record the send).
 *
 * Conditions:
 *   P1  primary heartbeat older than 30 min AND (backup heartbeat older than 30 min OR a job queued for more than 30 min).
 *       One page per outage (the key is cleared on recovery) and one "recovered" note.
 *   P2  one page per `failed` job id (a failed state is a real failure since E2).
 *   P3  one daily digest in the 07:30 ET tick.
 */

export const ALERT_STATE_KEY = "jobs/alerts/state.json";
export const ALERT_STATE_SCHEMA = "golfprice.alert_state.v1";
export const ALERT_COOLDOWN_MS = 24 * 3_600_000;
export const ALERT_RESERVE_MS = 3_600_000;
export const P1_STALE_MS = 30 * 60_000;
export const STATE_KEY_RETENTION_MS = 7 * 86_400_000;

export type AlertKeyState = { sent_at?: number; reserved_at?: number };
export type AlertState = {
  schema: string;
  keys: Record<string, AlertKeyState>;
  /** Set when a P1 page went out and no recovery note has been sent yet. */
  p1_outage?: { since: number } | null;
  /** First-run guard: written once when the state object is first created. `failed` lists the job ids already failed then; they never page. */
  seeded?: { at: number; failed: string[] } | null;
};

export const emptyAlertState = (): AlertState => ({ schema: ALERT_STATE_SCHEMA, keys: {}, p1_outage: null });

export function parseAlertState(value: unknown): AlertState {
  if (!value || typeof value !== "object") return emptyAlertState();
  const raw = value as Partial<AlertState>;
  const keys: Record<string, AlertKeyState> = {};
  for (const [key, entry] of Object.entries(raw.keys && typeof raw.keys === "object" ? raw.keys : {})) {
    if (entry && typeof entry === "object") keys[key] = { sent_at: Number((entry as AlertKeyState).sent_at) || 0, reserved_at: Number((entry as AlertKeyState).reserved_at) || 0 };
  }
  const outage = raw.p1_outage && typeof raw.p1_outage === "object" && Number.isFinite(Number(raw.p1_outage.since)) ? { since: Number(raw.p1_outage.since) } : null;
  const seed = raw.seeded && typeof raw.seeded === "object" && Number.isFinite(Number(raw.seeded.at)) ? raw.seeded : null;
  const seeded = seed ? { at: Number(seed.at), failed: Array.isArray(seed.failed) ? seed.failed.filter((id): id is string => typeof id === "string").slice(0, 500) : [] } : null;
  return { schema: ALERT_STATE_SCHEMA, keys, p1_outage: outage, seeded };
}

/** The seed written on the first enabled tick: every job already failed then is treated as known, so switching the secrets on does not page a backlog. */
export const seedState = (recentJobs: JobLike[], now: number): AlertState => ({ ...emptyAlertState(), seeded: { at: now, failed: failedJobs(recentJobs).map((j) => j.id).slice(0, 500) } });

/** telegram_policy.should_alert: 24 h since the last send and no reservation in the last hour. */
export function shouldAlert(previous: AlertKeyState | undefined, now: number): boolean {
  const p = previous ?? {};
  return now - (p.sent_at ?? 0) >= ALERT_COOLDOWN_MS && now - (p.reserved_at ?? 0) >= ALERT_RESERVE_MS;
}

export function pruneState(state: AlertState, now: number): AlertState {
  const keys: Record<string, AlertKeyState> = {};
  for (const [key, entry] of Object.entries(state.keys)) {
    const last = Math.max(entry.sent_at ?? 0, entry.reserved_at ?? 0);
    if (now - last < STATE_KEY_RETENTION_MS) keys[key] = entry;
  }
  return { ...state, keys };
}

export type BeatLike = { at?: string | null } | null | undefined;
export type JobLike = { id: string; type: string; requested_at: string; state: string };

const parseIso = (value: unknown): number | null => {
  if (typeof value !== "string") return null;
  const ms = Date.parse(value);
  return Number.isFinite(ms) ? ms : null;
};

export const beatAgeMs = (beat: BeatLike, now: number): number => {
  const at = parseIso(beat?.at);
  return at === null ? Number.POSITIVE_INFINITY : now - at;
};

export type P1Result = { down: boolean; reason: string };

export function evaluateP1(primary: BeatLike, backup: BeatLike, jobs: JobLike[], now: number): P1Result {
  const primaryAge = beatAgeMs(primary, now);
  if (primaryAge <= P1_STALE_MS) return { down: false, reason: "primary heartbeat is fresh" };
  const backupAge = beatAgeMs(backup, now);
  const stuck = jobs.filter((j) => j.state === "queued" && now - (parseIso(j.requested_at) ?? now) > P1_STALE_MS);
  const primaryText = Number.isFinite(primaryAge) ? `${Math.round(primaryAge / 60_000)} min` : "never";
  if (backupAge > P1_STALE_MS) return { down: true, reason: `primary heartbeat ${primaryText} old and the backup is silent too` };
  if (stuck.length) return { down: true, reason: `primary heartbeat ${primaryText} old and ${stuck.length} job(s) queued over 30 min (oldest ${stuck[0].type})` };
  return { down: false, reason: "primary is silent but the backup is alive and nothing is stuck" };
}

export const failedJobs = (jobs: JobLike[]): JobLike[] => jobs.filter((j) => j.state === "failed");

export type Alert = { key: string; text: string; kind: "p1" | "p1_recovered" | "p2" | "p3" };

export function digestText(jobs: JobLike[], primary: BeatLike, backup: BeatLike, now: number): string {
  const counts: Record<string, number> = {};
  for (const job of jobs) counts[job.state] = (counts[job.state] ?? 0) + 1;
  const summary = Object.entries(counts).sort().map(([state, n]) => `${n} ${state}`).join(", ") || "no jobs";
  const age = (beat: BeatLike) => (Number.isFinite(beatAgeMs(beat, now)) ? `${Math.round(beatAgeMs(beat, now) / 60_000)} min ago` : "no heartbeat");
  return `Golf dashboard daily digest: last 24 h ${summary}. Primary heartbeat ${age(primary)}; backup ${age(backup)}.`;
}

/** The alerts due at this tick, before the per-key cooldown is applied. */
export function dueAlerts(input: { jobs24h: JobLike[]; recentJobs: JobLike[]; primary: BeatLike; backup: BeatLike; state: AlertState; now: number; etHHMM: string; etDate: string }): Alert[] {
  const { jobs24h, recentJobs, primary, backup, state, now } = input;
  const alerts: Alert[] = [];
  const p1 = evaluateP1(primary, backup, recentJobs, now);
  if (p1.down) alerts.push({ key: "p1-primary-down", kind: "p1", text: `Golf runner ALERT: ${p1.reason}. Nothing is picking up jobs.` });
  else if (state.p1_outage) alerts.push({ key: "p1-recovered", kind: "p1_recovered", text: "Golf runner recovered: the primary heartbeat is fresh again." });
  const known = new Set(state.seeded?.failed ?? []);
  for (const job of failedJobs(recentJobs).filter((j) => !known.has(j.id))) alerts.push({ key: `p2-failed-${job.id}`, kind: "p2", text: `Golf job failed: ${job.type} (${job.id}). Open the Run page for the log.` });
  if (input.etHHMM === "07:30") alerts.push({ key: `p3-digest-${input.etDate}`, kind: "p3", text: digestText(jobs24h, primary, backup, now) });
  return alerts;
}
