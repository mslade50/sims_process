/** Private scheduled snapshots, not a streaming feed. No model or collector inference here. */
export type Tour = "pga" | "dpwt" | "kft";
export type Metric = { label: string; value: string | number };
export type Collector = {
  id: string; tour: Tour; label: string; state: "running" | "paused" | "idle" | "error" | "unknown";
  reason: string; observed_at: string; last_success_at?: string | null; heartbeat_at?: string | null;
  next_run_at?: string | null; schedule?: string; metrics?: Metric[]; issues?: string[];
};
export type Publication = {
  id: string; tour: Tour; label: string; generation?: string | null; published_at?: string | null;
  manifest_key?: string; manifest_sha256?: string; metrics?: Metric[];
  validation: { state: "passed" | "held" | "partial" | "unverified"; checked_at?: string; manifest_sha256?: string; summary: string; limits?: string[] };
  notes?: string[];
};
export type Snapshot = { schema_version: 1; observed_at: string; host?: string; collectors?: Collector[]; publications?: Publication[]; notes?: string[] };
export const SOURCES = [
  { file: "collectors-desktop.json", label: "Desktop collectors", field: "collectors", maxAgeMs: 15 * 60_000 },
  { file: "publications.json", label: "Publications", field: "publications", maxAgeMs: 30 * 60_000 },
] as const;
export type SourceFile = typeof SOURCES[number]["file"];
export type Source = { file: SourceFile; status: "loading" | "ready" | "missing" | "error"; data?: Snapshot; error?: string; checkedAt?: number };
export const initialSources = (): Source[] => SOURCES.map(({ file }) => ({ file, status: "loading" }));
export const TOURS: Tour[] = ["pga", "dpwt", "kft"];
export const TOUR_LABELS: Record<Tour, string> = { pga: "PGA Tour", dpwt: "DP World Tour", kft: "Korn Ferry Tour" };
export type Freshness = "fresh" | "stale" | "invalid";
// Require a timezone: local/ambiguous date strings must never produce an all-clear.
export function timestamp(value: unknown): number {
  return typeof value === "string" && /^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|[+-]\d\d:\d\d)$/.test(value) ? Date.parse(value) : NaN;
}
export function freshness(at: unknown, now: number, maxAgeMs: number): Freshness {
  const time = timestamp(at);
  if (!Number.isFinite(time) || time > now + 5 * 60_000) return "invalid";
  return now - time > maxAgeMs ? "stale" : "fresh";
}
export function sourceFreshness(source: Source, now: number): Freshness {
  if (!source.data) return "invalid";
  const age = freshness(source.data.observed_at, now, SOURCES.find((s) => s.file === source.file)!.maxAgeMs);
  return age === "invalid" ? "invalid" : source.status === "ready" ? age : "stale";
}
export function collectorStatus(row: Collector, source: Source, now: number): string {
  const snap = sourceFreshness(source, now);
  const own = freshness(row.observed_at, now, 15 * 60_000);
  if (snap === "invalid" || own === "invalid") return "Invalid check time";
  if (snap === "stale" || own === "stale") return `STALE · last reported ${row.state}`;
  return row.state;
}
export function publicationStatus(row: Publication, source: Source, now: number): string {
  const snap = sourceFreshness(source, now);
  if (snap === "invalid") return "Invalid check time";
  if (snap === "stale") return "STALE · readiness unknown";
  if (row.validation.state !== "passed") return row.validation.state;
  const hash = row.manifest_sha256;
  if (!hash || !/^[a-f0-9]{64}$/i.test(hash) || hash.toLowerCase() !== row.validation.manifest_sha256?.toLowerCase()) return "Unverified · manifest mismatch";
  // Publication age is dataset age, not machine freshness. Only absent/invalid/future dates block readiness.
  if (!Number.isFinite(timestamp(row.published_at)) || timestamp(row.published_at) > now + 5 * 60_000) return "Unverified · publication time";
  if (row.validation.checked_at && freshness(row.validation.checked_at, now, Infinity) === "invalid") return "Unverified · validation time";
  return "Ready";
}
const object = (x: unknown): x is Record<string, unknown> => !!x && typeof x === "object" && !Array.isArray(x);
const string = (x: unknown): x is string => typeof x === "string";
const strings = (x: unknown) => x === undefined || (Array.isArray(x) && x.every(string));
const optionalString = (x: unknown) => x === undefined || string(x);
const nullableString = (x: unknown) => x === null || optionalString(x);
const metrics = (x: unknown) => x === undefined || (Array.isArray(x) && x.every((m) => object(m) && string(m.label) && (string(m.value) || (typeof m.value === "number" && Number.isFinite(m.value)))));
function common(x: unknown): x is Record<string, unknown> {
  return object(x) && string(x.id) && !!x.id.trim() && TOURS.includes(x.tour as Tour) && string(x.label) && metrics(x.metrics);
}
export function decodeSnapshot(raw: unknown, file: SourceFile): Snapshot {
  if (!object(raw) || raw.schema_version !== 1 || !string(raw.observed_at) || !optionalString(raw.host) || !strings(raw.notes)) throw new Error("Invalid snapshot envelope (expected schema_version 1).");
  const field = SOURCES.find((s) => s.file === file)!.field;
  const rows = raw[field];
  if (!Array.isArray(rows)) throw new Error(`Snapshot is missing its ${field} array.`);
  const ids = new Set<string>();
  for (const row of rows) {
    if (!common(row)) throw new Error(`Invalid ${field} row.`);
    if (ids.has(row.id as string)) throw new Error(`Duplicate ${field} id.`);
    ids.add(row.id as string);
    if (field === "collectors") {
      if (!["running", "paused", "idle", "error", "unknown"].includes(row.state as string) || !string(row.reason) || !string(row.observed_at) || !nullableString(row.last_success_at) || !nullableString(row.heartbeat_at) || !nullableString(row.next_run_at) || !optionalString(row.schedule) || !strings(row.issues)) throw new Error("Invalid collector fields.");
    } else {
      const v = row.validation;
      if (!object(v) || !["passed", "held", "partial", "unverified"].includes(v.state as string) || !string(v.summary) || !optionalString(v.checked_at) || !optionalString(v.manifest_sha256) || !strings(v.limits) || !nullableString(row.generation) || !nullableString(row.published_at) || !optionalString(row.manifest_key) || !optionalString(row.manifest_sha256) || !strings(row.notes)) throw new Error("Invalid publication fields.");
    }
  }
  // Only the expected row family is consumed from each independent source.
  return { schema_version: 1, observed_at: raw.observed_at, host: raw.host, notes: raw.notes, [field]: rows } as Snapshot;
}
export type Attention = { key: string; message: string; priority: number };
export function attentionItems(sources: Source[], now: number): Attention[] {
  const items: Attention[] = [];
  for (const source of sources) {
    const spec = SOURCES.find((s) => s.file === source.file)!;
    if (source.status !== "ready") items.push({ key: source.file, priority: 0, message: `${spec.label}: ${source.error ?? "loading snapshot…"}${source.data ? " Previous copy is STALE." : ""}` });
    if (source.data && sourceFreshness(source, now) !== "fresh") items.push({ key: `${source.file}-age`, priority: 0, message: `${spec.label}: ${sourceFreshness(source, now) === "invalid" ? "invalid or future check time" : "STALE snapshot"}. Current state cannot be confirmed.` });
    for (const row of source.data?.collectors ?? []) {
      const state = collectorStatus(row, source, now);
      if (!["running", "idle"].includes(state) || row.issues?.length) items.push({ key: `${source.file}-${row.id}`, priority: row.state === "error" ? 0 : row.state === "paused" ? 1 : 2, message: `${TOUR_LABELS[row.tour]} / ${row.label}: ${state === "error" ? "Needs attention" : state}. ${row.reason}${row.issues?.length ? ` ${row.issues.join("; ")}` : ""}` });
    }
    for (const row of source.data?.publications ?? []) {
      const state = publicationStatus(row, source, now);
      if (state !== "Ready") items.push({ key: `${source.file}-${row.id}`, priority: 2, message: `${TOUR_LABELS[row.tour]} / ${row.label}: ${state}. ${row.validation.summary}` });
    }
    if (source.status === "ready" && source.data) for (const tour of TOURS) {
      if (!source.data[spec.field]?.some((row) => row.tour === tour)) items.push({ key: `${source.file}-${tour}-empty`, priority: 2, message: `${TOUR_LABELS[tour]}: no ${spec.field} reported. Coverage is unknown.` });
    }
  }
  return items.sort((a, b) => a.priority - b.priority);
}
