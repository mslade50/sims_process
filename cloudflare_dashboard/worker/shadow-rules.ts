/**
 * Passive shadow observer rules (golf_ops docs/OBSERVABILITY.md sections 6 and 8, golfprice subset only). Pure TypeScript: no I/O, no sending.
 *
 * Three checks over the data the Worker already reads for alerts:
 *   (a) golfprice jobs that failed, or are stuck (queued over 30 min, claimed/running over 3 h);
 *   (b) heartbeat staleness per machine (older than 30 min, the same threshold as the P1 alert);
 *   (c) expected cron occurrences (SLOTS/WINDOWS of cron-rules.ts) with no job created within the grace period.
 *
 * Every finding has a stable key, so writing it twice produces the same R2 object name (worker/shadow.ts writes it only if absent).
 */
import { CRON_REQUESTER } from "./cron-rules.ts";
import { isTerminal } from "../app/jobs-rules.ts";
import { P1_STALE_MS, beatAgeMs, failedJobs, type BeatLike, type JobLike } from "./alerts-rules.ts";

export const SHADOW_PREFIX = "ops/shadow/";
export const SHADOW_SUMMARY_KEY = "ops/shadow/summary.json";
export const SHADOW_REGISTRY_KEY = "ops/registry.json";
export const SHADOW_SCHEMA = "golf_ops.shadow.v1";
export const TICK_MS = 15 * 60_000;
export const MISSED_GRACE_MS = 10 * 60_000;
export const MISSED_LOOKBACK_MS = 3 * 3_600_000;
/** How long a job stays in the observer's scan (worker/shadow.ts MONITOR_HORIZON_MS); a job finding expires with it. */
export const JOB_HORIZON_MS = 48 * 3_600_000;
export const STUCK_RUNNING_MS = 3 * 3_600_000;
const RATE_LIMIT_MS = 6 * 3_600_000; // same-type jobs still queued/claimed/running inside this window legitimately suppress a new enqueue

/** `finished_at` = when the job reached a terminal state (queue cancelled_at when cancellation supplies the effective state; otherwise status finished_at or updated_at), epoch ms; null if unknown. */
export type ShadowJob = JobLike & { requested_by?: string; visible_at?: number | null; state_published_at?: number | null; queue_cancelled_at?: number | null; finished_at?: number | null; terminal_source?: "queue" | "status"; terminal_published_at?: number | null };
export type ShadowBeat = { machine: string; role: "primary" | "backup"; beat: BeatLike };
export type Finding = {
  key: string;
  kind: "job_failed" | "job_stuck" | "heartbeat_stale" | "occurrence_missed" | "status_unreadable" | "queue_unreadable" | "queue_entry_invalid" | "heartbeat_unreadable";
  severity: "high" | "warn";
  /** UTC yyyy-mm-dd of the subject event; part of the R2 object path. */
  date: string;
  subject: string;
  detail: string;
  /** Epoch ms after which the condition can no longer be re-derived (a missed slot leaves the lookback). Informational; not stored. */
  expires_at?: number;
};

const parseMs = (value: unknown): number | null => {
  if (typeof value !== "string") return null;
  const ms = Date.parse(value);
  return Number.isFinite(ms) ? ms : null;
};
const day = (ms: number) => new Date(ms).toISOString().slice(0, 10);
const slug = (text: string) => text.replace(/[^A-Za-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 120);
const stamp = (ms: number) => new Date(ms).toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z");

export type ShadowInput = {
  now: number;
  /** Jobs with a resolved state (the alerts window). */
  jobs: ShadowJob[];
  /** Every queue record, state unknown: used only to prove that an occurrence created a job. */
  created: { type: string; requested_at: string; requested_by?: string }[];
  beats: ShadowBeat[];
  /** Receives job IDs explaining suppressed occurrences with uncertain chronology. */
  uncertain?: string[];
  /** Queue records whose status object could not be read (state unknown). */
  unreadable?: { id: string; type: string; requested_at: string; corrupt?: boolean }[];
  /** cron-rules.dueJobs, injected so this module stays import-light. */
  dueAt: (ms: number) => { type: string }[];
};

export function evaluateShadow(input: ShadowInput): Finding[] {
  const { now } = input;
  const out: Finding[] = [];

  for (const job of failedJobs(input.jobs)) {
    const at = parseMs(job.requested_at) ?? now;
    out.push({ key: `job-failed-${slug(job.id)}`, kind: "job_failed", severity: "high", date: day(at), subject: job.id, detail: `${job.type} failed`, expires_at: at + JOB_HORIZON_MS });
  }
  for (const job of input.jobs) {
    const at = parseMs(job.requested_at);
    if (at === null) continue;
    const age = now - at;
    const stuck = (job.state === "queued" && age > P1_STALE_MS) || ((job.state === "claimed" || job.state === "running") && age > STUCK_RUNNING_MS);
    if (stuck) out.push({ key: `job-stuck-${slug(job.id)}`, kind: "job_stuck", severity: "warn", date: day(at), subject: job.id, detail: `${job.type} ${job.state} for ${Math.round(age / 60_000)} min`, expires_at: at + JOB_HORIZON_MS });
  }

  for (const u of input.unreadable ?? []) {
    // Only a status that is present but invalid is an incident; an R2 read failure is transient (degraded coverage in the summary only).
    if (!u.corrupt) continue;
    out.push({ key: `status-unreadable-${slug(u.id)}`, kind: "status_unreadable", severity: "warn", date: day(parseMs(u.requested_at) ?? now), subject: u.id, detail: `status of ${u.type} job could not be read; missed-slot checks for it are suppressed for 6 h after its request`, expires_at: (parseMs(u.requested_at) ?? now) + JOB_HORIZON_MS });
  }

  for (const { machine, role, beat } of input.beats) {
    const age = beatAgeMs(beat, now);
    if (age <= P1_STALE_MS) continue;
    const last = parseMs(beat?.at ?? undefined);
    out.push({
      key: `heartbeat-stale-${slug(machine)}-${last === null ? "never" : stamp(last)}`,
      kind: "heartbeat_stale",
      severity: role === "primary" ? "high" : "warn",
      date: day(last ?? now),
      subject: machine,
      detail: `${role} heartbeat ${last === null ? "missing" : `${Math.round(age / 60_000)} min old`}`,
      expires_at: now + MISSED_LOOKBACK_MS,
    });
  }

  const firstTick = Math.floor((now - MISSED_LOOKBACK_MS) / TICK_MS) * TICK_MS;
  for (let tick = firstTick; tick + MISSED_GRACE_MS <= now; tick += TICK_MS) {
    for (const due of input.dueAt(tick)) {
      const made = input.created.some((j) => {
        const at = parseMs(j.requested_at);
        // The scheduler preserves scheduledTime to the second; dueJobs matches its minute.
        return j.type === due.type && j.requested_by === CRON_REQUESTER && at !== null && at >= tick && at < tick + 60_000 && at <= now;
      });
      if (made) continue;
      let blocked = false;
      const uncertain: string[] = [];
      for (const j of input.jobs) {
        const at = parseMs(j.requested_at);
        if (j.type !== due.type || at === null || at >= tick + 60_000 || tick - at >= RATE_LIMIT_MS) continue;
        // The cron may have run at any second in its scheduled minute. Ordering against a request in that minute is unknown.
        if (at >= tick) { uncertain.push(j.id); continue; }
        if (!isTerminal(j.state)) {
          // Existence alone cannot establish that a later claim was nonterminal at the slot.
          // In particular, a stale runner may publish claimed after a queue cancellation.
          const needsStateProof = j.state !== "queued" || j.queue_cancelled_at != null;
          const unknownState = needsStateProof && (j.state_published_at == null || j.state_published_at >= tick || j.state_published_at < at);
          if (unknownState || at + RATE_LIMIT_MS < tick + 60_000 || j.visible_at == null || j.visible_at >= tick || j.visible_at < at) uncertain.push(j.id);
          else blocked = true;
          continue;
        }
        if (j.finished_at == null) { uncertain.push(j.id); continue; }
        // Queue cancellation and runner terminal status both require known publication before the slot.
        // Equal timestamps cannot establish which read/write occurred first.
        if (j.terminal_published_at == null || j.terminal_published_at >= tick || j.terminal_published_at < j.finished_at) {
          uncertain.push(j.id);
        }

      }
      for (const u of input.unreadable ?? []) {
        const at = parseMs(u.requested_at);
        if (u.type === due.type && at !== null && at < tick + 60_000 && tick - at < RATE_LIMIT_MS) uncertain.push(u.id);
      }
      if (blocked) continue;
      if (uncertain.length) {
        for (const id of uncertain) {
          const label = `occurrence_chronology_unknown:${id}`;
          if (input.uncertain && !input.uncertain.includes(label)) input.uncertain.push(label);
        }
        continue;
      }
      out.push({ key: `missed-${slug(due.type)}-${stamp(tick)}`, kind: "occurrence_missed", severity: "warn", date: day(tick), subject: due.type, detail: `no ${due.type} job created for the ${new Date(tick).toISOString()} slot`, expires_at: tick + MISSED_LOOKBACK_MS });
    }
  }
  return out;
}

export type ShadowSummary = {
  schema: string;
  updated_at: string;
  registry: { present: boolean; jobs: number | null };
  checked: { jobs: number; machines: number };
  /** Coverage gaps this tick (unreadable objects); empty = full coverage. */
  degraded: string[];
  open: number;
  by_kind: Record<string, number>;
  incidents: { key: string; kind: string; severity: string; date: string; detail: string }[];
};

export function buildSummary(now: number, findings: Finding[], registry: unknown, checked: { jobs: number; machines: number }, degraded: string[] = [], extra: Record<string, unknown> = {}): ShadowSummary {
  const by_kind: Record<string, number> = {};
  for (const f of findings) by_kind[f.kind] = (by_kind[f.kind] ?? 0) + 1;
  const reg = registry && typeof registry === "object" ? (registry as { jobs?: unknown }) : null;
  const jobs = reg ? (Array.isArray(reg.jobs) ? reg.jobs.length : reg.jobs && typeof reg.jobs === "object" ? Object.keys(reg.jobs).length : null) : null;
  return {
    schema: SHADOW_SCHEMA,
    updated_at: new Date(now).toISOString(),
    registry: { present: !!reg, jobs },
    checked,
    ...extra,
    degraded,
    open: findings.length,
    by_kind,
    incidents: findings.slice(0, 100).map(({ key, kind, severity, date, detail }) => ({ key, kind, severity, date, detail })),
  };
}
