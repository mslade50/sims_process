/**
 * Owner-override rules, mirrored exactly from golfprice/overrides.py (ALLOWED, CUT_BOUNDS, check_one).
 * Pure TypeScript with no runtime APIs, so the Worker (server-side validation) and the Adjust form share it.
 * A value outside a bound is rejected, never clipped. tests/overrides.test.mjs compares this table with the
 * published golfprice/overrides_schema.json so the two cannot drift silently.
 */

export const MAX_LIFETIME_DAYS = 28;
export const MIN_REASON_CHARS = 5;
export const OVERRIDES_KEY = "overrides/active.json";
export const HISTORY_PREFIX = "overrides/history/";

export type Scope = "player" | "course" | "engine";
export type FieldSpec =
  | { scope: Scope; field: string; kind: "number"; lo: number; hi: number; unit: string; what: string }
  | { scope: Scope; field: string; kind: "bool"; unit: string; what: string }
  | { scope: Scope; field: string; kind: "cut_rule"; unit: string; what: string };

export const ALLOWED: FieldSpec[] = [
  { scope: "player", field: "skill_delta", kind: "number", lo: -1.0, hi: 1.0, unit: "strokes gained per round", what: "added to the player's challenger mean" },
  { scope: "player", field: "sd_mult", kind: "number", lo: 0.7, hi: 1.5, unit: "x", what: "multiplies the player's per-round SD (the V1 SD the hole engine pins)" },
  { scope: "player", field: "withdraw", kind: "bool", unit: "flag", what: "player removed from the challenger field (zero prices; the rest is re-simulated without him)" },
  { scope: "course", field: "scoring_avg_delta", kind: "number", lo: -3.0, hi: 3.0, unit: "strokes per round", what: "shifts every hole's expected score so the course average moves by this much (level only)" },
  { scope: "course", field: "sd_mult", kind: "number", lo: 0.8, hi: 1.25, unit: "x", what: "multiplies every player's per-round SD" },
  { scope: "engine", field: "cut_rule", kind: "cut_rule", unit: "rule", what: "replaces fields of the event's cut rule (cut_round, top_n, within, mdf_trigger, mdf_top_n, mdf_round)" },
];

export const CUT_BOUNDS: Record<string, [number, number]> = {
  cut_round: [0, 3],
  top_n: [1, 156],
  within: [0, 20],
  mdf_trigger: [-1, 156],
  mdf_top_n: [0, 156],
  mdf_round: [2, 3],
};

export type OverrideRecord = {
  id: string;
  created_at: string;
  author: string;
  event: string;
  scope: Scope;
  target?: number;
  field: string;
  value: number | true | Record<string, number>;
  reason: string;
  expires_at: string;
};

const RECORD_KEYS = new Set(["id", "created_at", "author", "event", "scope", "target", "field", "value", "reason", "expires_at", "note"]);
const ISO_WITH_ZONE = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})$/;

export function parseUtc(value: unknown): number | null {
  if (typeof value !== "string" || !ISO_WITH_ZONE.test(value.trim())) return null;
  const ms = Date.parse(value.trim());
  return Number.isNaN(ms) ? null : ms;
}

/** Normalised UTC form used for stored records: 2026-10-05T14:00:00Z */
export function isoSeconds(ms: number): string {
  return new Date(ms).toISOString().replace(/\.\d{3}Z$/, "Z");
}

export function specFor(scope: unknown, field: unknown): FieldSpec | undefined {
  return ALLOWED.find((spec) => spec.scope === scope && spec.field === field);
}

/** Problems with one record, independent of event and time (empty list = well formed and inside the hard bounds). Port of check_one(). */
export function checkOverride(o: unknown, rounds = 4): string[] {
  if (typeof o !== "object" || o === null || Array.isArray(o)) return ["the override is not in a readable form"];
  const r = o as Record<string, unknown>;
  const bad: string[] = [];
  for (const key of ["id", "created_at", "author", "event", "scope", "field", "value", "reason", "expires_at"]) {
    const v = r[key];
    if (v === undefined || v === null || (typeof v === "string" && !v.trim())) bad.push(`missing ${key}`);
  }
  if (bad.length) return bad;
  const extra = Object.keys(r).filter((key) => !RECORD_KEYS.has(key)).sort();
  if (extra.length) bad.push(`unknown keys ${JSON.stringify(extra)}`);
  if (typeof r.id !== "string" || r.id.length > 80) bad.push("id must be a string of at most 80 characters");
  if (typeof r.reason !== "string" || r.reason.trim().length < MIN_REASON_CHARS) bad.push(`the reason must be at least ${MIN_REASON_CHARS} characters long`);
  if (typeof r.event !== "string" || !/^[A-Za-z0-9:_.-]{1,60}$/.test(r.event)) bad.push("choose an event (or all events)");
  const spec = specFor(r.scope, r.field);
  if (!spec) {
    return [...bad, `${String(r.scope)} / ${String(r.field)} is not allowed as an override`];
  }
  const created = parseUtc(r.created_at);
  const expires = parseUtc(r.expires_at);
  if (created === null) bad.push("the start time is not a valid date and time (created_at)");
  if (expires === null) bad.push("the expiry time is not a valid date and time (expires_at)");
  if (created !== null && expires !== null) {
    if (expires <= created) bad.push("expires_at must be after created_at");
    else if (expires - created > MAX_LIFETIME_DAYS * 86_400_000) bad.push(`expires_at is more than ${MAX_LIFETIME_DAYS} days after created_at`);
  }
  const v = r.value;
  if (spec.kind === "number") {
    if (typeof v !== "number" || !Number.isFinite(v)) bad.push("the value must be a number");
    else if (v < spec.lo || v > spec.hi) bad.push(`the value ${v} is outside the hard bounds of ${spec.lo} to ${spec.hi} (${spec.unit})`);
  } else if (spec.kind === "bool") {
    if (v !== true) bad.push("value must be true (an override that does nothing is deleted, not stored)");
  } else {
    if (typeof v !== "object" || v === null || Array.isArray(v) || !Object.keys(v).length) {
      bad.push("value must be a non-empty object of cut-rule fields");
    } else {
      const n0 = bad.length;
      const rule = v as Record<string, unknown>;
      for (const [k, x] of Object.entries(rule)) {
        const bounds = CUT_BOUNDS[k];
        if (!bounds) bad.push(`cut_rule key '${k}' not allowed`);
        else if (typeof x !== "number" || !Number.isInteger(x) || x < bounds[0] || x > bounds[1]) bad.push(`cut_rule ${k}=${JSON.stringify(x)} must be an integer in [${bounds[0]}, ${bounds[1]}]`);
        else if (k === "cut_round" && x === 1) bad.push("cut_round must be 0 (no cut), 2 (36-hole) or 3 (54-hole)");
      }
      const cutRound = typeof rule.cut_round === "number" ? rule.cut_round : 2;
      if (bad.length === n0 && cutRound >= rounds) bad.push(`cut_round must be below the number of rounds (${rounds})`);
    }
  }
  if (r.scope === "player") {
    if (typeof r.target !== "number" || !Number.isInteger(r.target)) bad.push("player scope needs an integer target (dg_id)");
  } else if (r.target !== undefined && r.target !== null && r.target !== "" && r.target !== "all") {
    bad.push(`${String(r.scope)} scope takes no target`);
  }
  return bad;
}

export function newOverrideId(now: number, random: () => number = Math.random): string {
  const stamp = isoSeconds(now).replace(/[-:]/g, "").slice(0, 15);
  return `ov-${stamp}-${Math.floor(random() * 0xffff).toString(16).padStart(4, "0")}`;
}

export type OverrideInput = { event?: unknown; scope?: unknown; target?: unknown; field?: unknown; value?: unknown; reason?: unknown; expires_at?: unknown };

/** Build the stored record from a client request. The server owns id, created_at and author; the client cannot set them. */
export function buildRecord(input: OverrideInput, author: string, now: number, random?: () => number): { record: OverrideRecord | null; problems: string[] } {
  if (typeof input !== "object" || input === null || Array.isArray(input)) return { record: null, problems: ["request body must be a JSON object"] };
  const allowedInput = new Set(["event", "scope", "target", "field", "value", "reason", "expires_at"]);
  const unknown = Object.keys(input).filter((k) => !allowedInput.has(k));
  if (unknown.length) return { record: null, problems: [`unknown keys ${JSON.stringify(unknown.sort())} (id, created_at and author are set by the server)`] };
  const expiresMs = parseUtc(input.expires_at);
  const record = {
    id: newOverrideId(now, random),
    created_at: isoSeconds(now),
    author,
    event: typeof input.event === "string" ? input.event.trim() : (input.event as string),
    scope: input.scope as Scope,
    ...(input.target !== undefined && input.target !== null && input.target !== "" ? { target: input.target as number } : {}),
    field: input.field as string,
    value: input.value as OverrideRecord["value"],
    reason: typeof input.reason === "string" ? input.reason.trim() : (input.reason as string),
    expires_at: expiresMs === null ? (input.expires_at as string) : isoSeconds(expiresMs),
  } as OverrideRecord;
  const problems = checkOverride(record);
  return { record: problems.length ? null : record, problems };
}

export function isExpired(record: { expires_at?: string }, now: number): boolean {
  const e = parseUtc(record.expires_at);
  return e === null || e <= now;
}
