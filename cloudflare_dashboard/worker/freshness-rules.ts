/**
 * Data-staleness rules for the 15-minute cron (owner intent 2026-10-28). Pure TypeScript: shared by worker/freshness.ts and tests/freshness.test.mjs.
 *
 * Every dashboard data source is listed once, with how often it should change and the longest age that is still normal. Thresholds follow the
 * schedule (worker/cron-rules.ts), never a fixed date, season or generation:
 *   - weekly sources (published once per weekly run or settle): 8 days;
 *   - intra-week sources that the odds_reprice window refreshes (odds signals, the bet card): a few hours, judged only after the window has been
 *     open that long, so the first tick of a new window never reads last week's object as stale.
 * An object whose own timestamp cannot be read falls back to the R2 upload time; a missing object is a finding of its own. An R2 read failure is
 * never "missing": it is recorded as unreadable and raises no finding (the same three-way read as worker/shadow.ts).
 */

export const FRESHNESS_STATUS_KEY = "owner/data_freshness.json";
export const FRESHNESS_SCHEMA = "golfprice.dashboard_freshness.v1";
export const INBOX_SOURCE = "dashboard-freshness";
export const INBOX_PREFIX = "ops/agent_inbox/dashboard-freshness/";
/** A still-stale source is re-filed (same inbox object, rewritten in place) at most this often. */
export const REFILE_MS = 24 * 3_600_000;
export const HOUR_MS = 3_600_000;
export const WEEKLY_LIMIT_HOURS = 8 * 24;
/** Odds checks run every 30 minutes in the window; no successful run for this long (window open that long) means the checker stopped. */
export const ODDS_CHECK_MAX_HOURS = 3;

export type Schedule = "weekly" | "odds_window" | "odds_check" | "history";
export type SourceSpec = {
  id: string;
  label: string;
  key: string;
  schedule: Schedule;
  /** Timestamp fields tried in order inside the first bytes of the object; the R2 upload time is the fallback. */
  fields: string[];
  /** Normal maximum age. For odds_window sources this applies inside the window; outside it the weekly limit applies. */
  limitHours: number;
  note: string;
};

export const SOURCES: SourceSpec[] = [
  { id: "golfprice_manifest", label: "golfprice publish manifest", key: "golfprice/manifest.json", schedule: "weekly", fields: [], limitHours: WEEKLY_LIMIT_HOURS, note: "Written last by every golfprice publish; the pointer every golfprice view follows." },
  { id: "golfprice_index", label: "golfprice event index", key: "golfprice/index.json", schedule: "weekly", fields: ["published_at"], limitHours: WEEKLY_LIMIT_HOURS, note: "The event list, run checkpoints and weather pointers." },
  { id: "legacy_manifest", label: "legacy data manifest", key: "data/manifest.json", schedule: "weekly", fields: ["generated_at"], limitHours: WEEKLY_LIMIT_HOURS, note: "Weekly legacy-pipeline export (History and Diagnostics)." },
  { id: "odds_signals", label: "odds signals", key: "golfprice/odds_signals/latest.json", schedule: "odds_check", fields: ["generated_at"], limitHours: ODDS_CHECK_MAX_HOURS, note: "The producer publishes nothing when quotes are unchanged, so the age of the content is never judged. Judged on the odds_reprice job outcomes instead: stale only when no odds check has succeeded for the limit while the window is open." },
  { id: "bet_card", label: "owner bet card", key: "owner/bet_card.json", schedule: "odds_window", fields: ["generated_at"], limitHours: 12, note: "Refreshed after each odds reprice; judged during the odds_reprice window, weekly otherwise." },
  { id: "scorecard", label: "model scorecard", key: "golfprice/scorecard/latest.json", schedule: "weekly", fields: ["generated_at"], limitHours: WEEKLY_LIMIT_HOURS, note: "Updated by each Monday settle." },
];

// owner/bet_breakdowns.json is deliberately not listed: its only producer is the one-off experiments/bet_breakdown build, so a scheduled age check
// could never be satisfied. The built_at / max_event_date label stays in the view. Add it back once a scheduled producer exists.

/** Course history documents of the newest events (taken from the published index, never listed here). */
export const HISTORY_SPEC = { idPrefix: "history:", schedule: "history" as Schedule, fields: ["source_through_date"], limitHours: 14 * 24, max: 2 };

export type ReadResult =
  | { kind: "absent" }
  | { kind: "error" }
  | { kind: "ok"; uploadedMs: number | null; head: string; json?: unknown };

export type SourceState = "ok" | "stale" | "missing" | "unreadable" | "skipped";
export type SourceStatus = {
  id: string;
  label: string;
  key: string;
  schedule: Schedule;
  state: SourceState;
  observed_at: string | null;
  age_hours: number | null;
  limit_hours: number | null;
  /** How the observed time was found: a named field, or the R2 upload time. */
  via: string | null;
  detail: string;
  /** First tick this source was seen stale (kept while it stays stale). */
  stale_since: string | null;
  /** Last time an inbox record was filed for the current stale episode. */
  inboxed_at: string | null;
};

/** The value of the first listed field found as a top-level-looking `"field": "value"` in the leading bytes; null when none. */
export function fieldFromHead(head: string, fields: string[]): { field: string; value: string } | null {
  for (const field of fields) {
    const m = new RegExp(`"${field}"\\s*:\\s*"([^"\\\\]{4,40})"`).exec(head);
    if (m) return { field, value: m[1] };
  }
  return null;
}

/** Parse a publisher timestamp ("2026-10-22T16:02:00Z", "2026-10-22 16:02:00 UTC", or a bare date at 00:00 UTC) to epoch ms; null if unreadable. */
export function timestampMs(value: unknown): number | null {
  if (typeof value !== "string") return null;
  const text = value.trim();
  if (/^\d{4}-\d{2}-\d{2}$/.test(text)) return Date.parse(`${text}T00:00:00Z`);
  const normal = /(Z|[+-]\d{2}:?\d{2})$/.test(text) ? text : `${text.replace(" UTC", "").replace(" ", "T")}Z`;
  const ms = Date.parse(normal);
  return Number.isFinite(ms) ? ms : null;
}

export type WindowProbe = (type: string, ms: number) => boolean;

/** Milliseconds the window has been continuously open at `now` (15-minute steps, capped at `capMs`); 0 when it is closed now. */
export function windowOpenForMs(inWindow: WindowProbe, type: string, now: number, capMs = 24 * HOUR_MS): number {
  if (!inWindow(type, now)) return 0;
  const step = 15 * 60_000;
  let open = 0;
  while (open < capMs && inWindow(type, now - open - step)) open += step;
  return open >= capMs ? capMs : open + step;
}

/** `active`: an event is in play or just finished (from the published index); false = off week, null = unknown (index unreadable). */
/** Outcome of the odds_reprice jobs in jobs/queue.json: `lastSuccessMs` is the newest finished clean run (null = none seen). Undefined on the context = the queue could not be read. */
export type OddsCheckEvidence = { lastSuccessMs: number | null; jobsSeen: number };
export type EvalContext = { now: number; inWindow: WindowProbe; active?: boolean | null; oddsCheck?: OddsCheckEvidence; runnerSilent?: boolean };

export const RUNNER_SILENT_HOURS = 6;
/** Heartbeat states that show a runner doing or ready to do work; fenced, held, refusing, error and missing states are not proof of a live pipeline. */
const LIVE_BEAT_STATES = new Set(["idle", "running"]);
/** True on positive evidence the pipeline stopped: heartbeats exist but none in a working state (idle/running) is newer than RUNNER_SILENT_HOURS. A fenced standby keeps beating, so it does not count. No readable heartbeat = no evidence (false). */
export function runnerSilent(beats: unknown[], now: number): boolean {
  const parsed = beats.map((b) => b as { at?: unknown; state?: unknown } | null).filter((b) => b !== null && typeof b === "object");
  if (parsed.length === 0) return false;
  const alive = parsed.filter((b) => typeof b!.state === "string" && LIVE_BEAT_STATES.has(b!.state)).map((b) => timestampMs(b!.at)).filter((ms): ms is number => ms !== null);
  return alive.length === 0 || now - Math.max(...alive) > RUNNER_SILENT_HOURS * HOUR_MS;
}

/** Newest clean odds_reprice run from the queue records and the status object of each (statuses keyed by job id; missing = unknown). Null when the queue is not a readable queue. */
export function oddsCheckEvidence(queue: unknown, statuses: Map<string, unknown>): OddsCheckEvidence | null {
  const jobs = (queue as { jobs?: unknown } | null)?.jobs;
  if (!Array.isArray(jobs)) return null;
  let lastSuccessMs: number | null = null;
  let jobsSeen = 0;
  for (const job of jobs) {
    const j = job as { id?: unknown; type?: unknown };
    if (!j || j.type !== "odds_reprice" || typeof j.id !== "string") continue;
    jobsSeen += 1;
    const st = statuses.get(j.id) as { state?: unknown; finished_at?: unknown; exit_code?: unknown } | null | undefined;
    if (!st || st.state !== "done" || (st.exit_code !== undefined && st.exit_code !== null && st.exit_code !== 0)) continue;
    const at = timestampMs(st.finished_at);
    if (at !== null && (lastSuccessMs === null || at > lastSuccessMs)) lastSuccessMs = at;
  }
  return { lastSuccessMs, jobsSeen };
}

const ACTIVE_BEFORE_DAYS = 11;
const ACTIVE_AHEAD_DAYS = 7;
/**
 * True when some indexed event starts within 11 days back to 7 days ahead of `now`. False = scheduled inactivity: no event in that activity window and
 * none published ahead (the winter break, an off week), so nothing is expected to publish and nothing is judged. Null when there are no events to read.
 * The index lists only published events and carries no tour calendar, so a missed publication inside the active window is caught by the age limits and
 * the jobs alerts. A dead pipeline during a break is told apart from scheduled inactivity by the runner heartbeats (see runnerSilent): when no
 * idle/running heartbeat is newer than RUNNER_SILENT_HOURS (a fenced standby or an error/refusing runner does not count) the caller judges the sources despite `false`, so a publisher that died mid-season goes stale at about D+11.
 * Limitation: an idle, healthy primary whose weekly publish silently exits 0 without publishing is not detectable during a break; an unreadable heartbeat prefix counts as no evidence.
 */
export function eventsActive(events: Array<{ date_start?: string | null }> | null, now: number): boolean | null {
  if (!events || events.length === 0) return null;
  const day = 86_400_000;
  const starts = events.map((e) => timestampMs(typeof e.date_start === "string" ? e.date_start.slice(0, 10) : null)).filter((ms): ms is number => ms !== null);
  if (starts.length === 0) return null;
  return starts.some((ms) => ms >= now - ACTIVE_BEFORE_DAYS * day && ms <= now + ACTIVE_AHEAD_DAYS * day);
}

/** Judge one read of one source. `ageOverrideMs`: for sources whose observed time is a calendar date, the caller passes the reading itself. */
export function judge(spec: { id: string; label: string; key: string; schedule: Schedule; fields: string[]; limitHours: number }, read: ReadResult, ctx: EvalContext): Omit<SourceStatus, "stale_since" | "inboxed_at"> {
  const base = { id: spec.id, label: spec.label, key: spec.key, schedule: spec.schedule, observed_at: null as string | null, age_hours: null as number | null, limit_hours: spec.limitHours as number | null, via: null as string | null };
  if (read.kind === "error") return { ...base, state: "unreadable", detail: "The storage read failed this tick; no judgment made." };
  if (read.kind === "absent") {
    if ((spec.schedule === "odds_window" || spec.schedule === "odds_check") && windowOpenForMs(ctx.inWindow, "odds_reprice", ctx.now) === 0) return { ...base, state: "skipped", limit_hours: null, detail: "Not published and the odds_reprice window is closed; not judged." };
    return { ...base, state: "missing", detail: `${spec.key} is not published.` };
  }
  if (ctx.active === false && !ctx.runnerSilent) return { ...base, limit_hours: null, state: "skipped", detail: "No event is in play or just finished in the published index and none is published ahead (off week or break); not judged." };
  if (spec.schedule === "odds_check") return judgeOddsCheck(base, spec, read, ctx);
  let observed: number | null = null;
  let via: string | null = null;
  const hit = fieldFromHead(read.head, spec.fields);
  if (hit) {
    observed = timestampMs(hit.value);
    via = observed !== null ? hit.field : null;
  }
  if (observed === null && read.uploadedMs !== null) {
    observed = read.uploadedMs;
    via = "upload time";
  }
  if (observed === null) return { ...base, state: "unreadable", detail: "No timestamp field and no upload time were readable." };
  const ageHours = (ctx.now - observed) / HOUR_MS;
  const common = { ...base, observed_at: new Date(observed).toISOString(), age_hours: Math.round(ageHours * 10) / 10, via };
  let limit = spec.limitHours;
  if (spec.schedule === "odds_window") {
    const open = windowOpenForMs(ctx.inWindow, "odds_reprice", ctx.now);
    if (open === 0) limit = WEEKLY_LIMIT_HOURS;
    else if (open < spec.limitHours * HOUR_MS) return { ...common, state: "ok", limit_hours: spec.limitHours, detail: `The odds window opened ${Math.round(open / HOUR_MS)} h ago; not judged until it has been open ${spec.limitHours} h.` };
  }
  if (ageHours < -1) return { ...common, state: "unreadable", limit_hours: limit, detail: "The timestamp is in the future; clock or data problem." };
  if (ageHours > limit) return { ...common, state: "stale", limit_hours: limit, detail: `Last updated ${fmtAge(ageHours)} ago (${via}); the normal maximum is ${fmtAge(limit)}.` };
  return { ...common, state: "ok", limit_hours: limit, detail: `Updated ${fmtAge(ageHours)} ago (${via}); normal maximum ${fmtAge(limit)}.` };
}

type JudgeBase = Omit<SourceStatus, "state" | "detail" | "stale_since" | "inboxed_at">;
/** Odds signals: judged on successful checks (the odds_reprice jobs), never on how old the published content is. */
function judgeOddsCheck(base: JudgeBase, spec: { limitHours: number; fields: string[] }, read: Extract<ReadResult, { kind: "ok" }>, ctx: EvalContext): Omit<SourceStatus, "stale_since" | "inboxed_at"> {
  const open = windowOpenForMs(ctx.inWindow, "odds_reprice", ctx.now);
  if (open === 0) return { ...base, limit_hours: null, state: "skipped", detail: "The odds_reprice window is closed; the checker is not expected to run. Content age is never judged (no new prices means no new file)." };
  if (!ctx.oddsCheck) return { ...base, state: "unreadable", detail: "The job queue or the job status records could not be read, so whether odds checks are succeeding is unknown." };
  const hit = fieldFromHead(read.head, spec.fields);
  const content = hit ? timestampMs(hit.value) : null;
  const contentNote = content !== null ? ` Published signals date from ${new Date(content).toISOString()}; a quiet market publishes nothing new.` : "";
  const last = ctx.oddsCheck.lastSuccessMs;
  const sinceHours = last === null ? null : Math.round(((ctx.now - last) / HOUR_MS) * 10) / 10;
  const common = { ...base, observed_at: last === null ? null : new Date(last).toISOString(), age_hours: sinceHours, via: "odds_reprice job", limit_hours: spec.limitHours };
  if (open < spec.limitHours * HOUR_MS) return { ...common, state: "ok", detail: `The odds window opened ${Math.round(open / HOUR_MS)} h ago; not judged until it has been open ${spec.limitHours} h.${contentNote}` };
  if (last === null || (sinceHours as number) > spec.limitHours) {
    return { ...common, state: "stale", detail: `No successful odds check ${last === null ? `among the ${ctx.oddsCheck.jobsSeen} odds_reprice job record(s) in the queue` : `for ${fmtAge(sinceHours as number)}`} while the window has been open ${fmtAge(open / HOUR_MS)}; the normal maximum is ${fmtAge(spec.limitHours)}.${contentNote}` };
  }
  return { ...common, state: "ok", detail: `Last successful odds check ${fmtAge(sinceHours as number)} ago; normal maximum ${fmtAge(spec.limitHours)}.${contentNote}` };
}

export function fmtAge(hours: number): string {
  const h = Math.max(0, hours);
  return h >= 48 ? `${(h / 24).toFixed(1)} days` : `${h.toFixed(1)} h`;
}

export type Transition = { id: string; kind: "stale" | "refile" | "recovered"; status: SourceStatus };

/** Carry stale_since / inboxed_at forward and decide which sources need an inbox record this tick. */
export function carryForward(now: number, current: Array<Omit<SourceStatus, "stale_since" | "inboxed_at">>, previous: SourceStatus[] | null, knownIds: string[] = []): { statuses: SourceStatus[]; transitions: Transition[] } {
  const prev = new Map((previous ?? []).map((s) => [s.id, s]));
  const statuses: SourceStatus[] = [];
  const transitions: Transition[] = [];
  const isBad = (s: { state: SourceState }) => s.state === "stale" || s.state === "missing";
  // An open episode is remembered by stale_since, so a tick that could not judge (unreadable, or a closed window or scheduled break) neither clears nor re-files it.
  // A skipped or unreadable tick never creates or extends an episode: it only keeps the existing stale_since / inboxed_at on a non-stale state.
  const isOpen = (s: SourceStatus | undefined): s is SourceStatus => !!s && s.stale_since !== null;
  for (const cur of current) {
    const before = prev.get(cur.id);
    let status: SourceStatus = { ...cur, stale_since: null, inboxed_at: null };
    if (isBad(cur)) {
      const since = isOpen(before) ? before.stale_since! : new Date(now).toISOString();
      status = { ...status, stale_since: since, inboxed_at: isOpen(before) ? before.inboxed_at : null };
      if (!status.inboxed_at) transitions.push({ id: cur.id, kind: "stale", status });
      else if (now - (timestampMs(status.inboxed_at) ?? 0) >= REFILE_MS) transitions.push({ id: cur.id, kind: "refile", status });
    } else if ((cur.state === "unreadable" || cur.state === "skipped") && isOpen(before)) {
      status = { ...status, stale_since: before.stale_since, inboxed_at: before.inboxed_at };
    } else if (isOpen(before) && before.inboxed_at) {
      transitions.push({ id: cur.id, kind: "recovered", status });
    }
    statuses.push(status);
  }
  // An open episode whose id was not judged this tick must not be dropped silently: keep it while its source is only unjudged, and close it
  // (a recovered record) when the id left the set by design (a rolled-over history document, a source removed from the list).
  const seen = new Set(current.map((c) => c.id));
  for (const before of previous ?? []) {
    if (seen.has(before.id) || !isOpen(before)) continue;
    const retired = before.id.startsWith(HISTORY_SPEC.idPrefix) || !knownIds.includes(before.id);
    if (!retired) { statuses.push(before); continue; }
    if (before.inboxed_at) {
      const gone: SourceStatus = { ...before, state: "skipped", stale_since: null, inboxed_at: null, detail: "No longer checked (left the set of newest events or the source list)." };
      transitions.push({ id: before.id, kind: "recovered", status: gone });
    }
  }
  return { statuses, transitions };
}

/** The agent-inbox record (the shape tools/watcher/check.py reads: source, ts, severity, summary, detail, dedupe_key). Never sent to a phone. */
export function inboxRecord(t: Transition, nowIso: string): { key: string; body: Record<string, unknown> } {
  const s = t.status;
  const dedupe = `${INBOX_SOURCE}:${s.id}`;
  if (t.kind === "recovered") {
    return {
      key: `${INBOX_PREFIX}${s.id}.recovered.json`,
      body: { source: INBOX_SOURCE, ts: nowIso, severity: "recovered", kind: "recovered", summary: `${dedupe}: recovered`, detail: `${s.label} is no longer stale (${s.detail})`, dedupe_key: dedupe },
    };
  }
  return {
    key: `${INBOX_PREFIX}${s.id}.json`,
    body: {
      source: INBOX_SOURCE, ts: nowIso, severity: "warning",
      summary: `${dedupe}: ${s.label} is ${s.state === "missing" ? "not published" : "stale"}`,
      detail: `${s.detail} Object ${s.key}. First seen ${s.stale_since ?? nowIso}. Dashboard Data freshness panel (Run page) shows the current state.`,
      dedupe_key: dedupe,
    },
  };
}

export type FreshnessDoc = {
  schema: string;
  checked_at: string;
  ok: boolean;
  /** True when no source could be judged this tick (everything skipped or unreadable): not the same as healthy. */
  not_judged: boolean;
  stale_count: number;
  sources: SourceStatus[];
  notes: string[];
  inbox: { bound: boolean; written: string[]; failed: string[] };
};

export function parseFreshness(raw: unknown): FreshnessDoc | null {
  if (!raw || typeof raw !== "object") return null;
  const d = raw as Record<string, unknown>;
  if (d.schema !== FRESHNESS_SCHEMA || typeof d.checked_at !== "string" || !Array.isArray(d.sources)) return null;
  const sources = d.sources.flatMap((s): SourceStatus[] => {
    if (!s || typeof s !== "object") return [];
    const o = s as Record<string, unknown>;
    if (typeof o.id !== "string" || typeof o.state !== "string") return [];
    const str = (v: unknown) => (typeof v === "string" ? v : null);
    const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : null);
    return [{ id: o.id, label: str(o.label) ?? o.id, key: str(o.key) ?? "", schedule: (str(o.schedule) as Schedule) ?? "weekly", state: o.state as SourceState, observed_at: str(o.observed_at), age_hours: num(o.age_hours), limit_hours: num(o.limit_hours), via: str(o.via), detail: str(o.detail) ?? "", stale_since: str(o.stale_since), inboxed_at: str(o.inboxed_at) }];
  });
  const inbox = (d.inbox && typeof d.inbox === "object" ? d.inbox : {}) as Record<string, unknown>;
  return {
    schema: FRESHNESS_SCHEMA, checked_at: d.checked_at, ok: d.ok === true, not_judged: d.not_judged === true, stale_count: typeof d.stale_count === "number" ? d.stale_count : sources.filter((s) => s.state === "stale" || s.state === "missing").length,
    sources, notes: Array.isArray(d.notes) ? d.notes.filter((n): n is string => typeof n === "string") : [],
    inbox: { bound: inbox.bound === true, written: Array.isArray(inbox.written) ? inbox.written.filter((n): n is string => typeof n === "string") : [], failed: Array.isArray(inbox.failed) ? inbox.failed.filter((n): n is string => typeof n === "string") : [] },
  };
}

/** The checker itself must be fresh: a status object older than this is shown as "checker not running". */
export const CHECKER_MAX_AGE_MS = 45 * 60_000;
