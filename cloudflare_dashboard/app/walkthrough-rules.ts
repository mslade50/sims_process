/** Display-only interpretation of the existing operations contract. Never infers delivery from a model artifact. */
import type { StageSummary } from "./ops-rules";

export type Signal = { tone: "green" | "yellow" | "red"; label: string; detail: string };
export type WalkthroughFlow = { ok: true; now: string; week: string; selected_event: string; stages: StageSummary[] };
const object = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object" && !Array.isArray(v);
const instant = (v: unknown) => typeof v === "string" ? Date.parse(v) : NaN;

/** Scope is checked again on the client, including while a different event request is in flight. */
export function workflowFor(value: unknown, event: string, week: string): WalkthroughFlow | null {
  if (!object(value) || value.ok !== true || value.selected_event !== event || value.week !== week || !Number.isFinite(instant(value.now)) || !Array.isArray(value.stages)) return null;
  const ids = new Set<string>();
  for (const s of value.stages) {
    if (!object(s) || typeof s.id !== "string" || ids.has(s.id) || typeof s.state !== "string" || typeof s.explanation !== "string" || !Array.isArray(s.proof) || !s.proof.every(p => typeof p === "string")) return null;
    ids.add(s.id);
  }
  return value as WalkthroughFlow;
}

export function stageSignal(stage: StageSummary | undefined, flow: WalkthroughFlow | null, now: number, error?: string | null): Signal {
  const age = now - instant(flow?.now);
  if (error || !flow || !Number.isFinite(age) || age > 60_000 || age < -60_000) return { tone: "yellow", label: "Unverified", detail: error ?? "Current workflow evidence is unavailable or stale. Refresh to check again." };
  if (!stage) return { tone: "yellow", label: "Unverified", detail: "No evidence is connected for this step." };
  if (stage.state === "failed" || stage.state === "needs_attention") return { tone: "red", label: stage.state === "failed" ? "Failed" : "Needs attention", detail: stage.explanation };
  if (stage.state === "verified") {
    const observed = instant(stage.evidence_at), expires = instant(stage.valid_until);
    if (Number.isFinite(observed) && observed <= now + 60_000 && Number.isFinite(expires) && expires > now && expires > observed && stage.proof.length > 0) return { tone: "green", label: "Verified", detail: stage.explanation };
    return { tone: "yellow", label: "Evidence expired or incomplete", detail: "The verification needs a current timestamp, expiry and output proof before it can be green." };
  }
  const labels: Record<string, string> = { running: "Running", scheduled: "Scheduled", completed_unverified: "Finished; outputs unverified", waiting_upstream: "Waiting upstream", retrying: "Retrying", skipped: "Skipped", stale: "Stale evidence", unknown: "Unverified" };
  return { tone: "yellow", label: labels[stage.state] ?? "Unverified", detail: stage.explanation };
}

export function matchingExplain<T extends { event_uid: string; kind: string; run: string }>(doc: T | null, event: { event_uid: string; explain_run?: string | null }): T | null {
  return doc && doc.event_uid === event.event_uid && !!event.explain_run && `${doc.kind}_${doc.run}` === event.explain_run ? doc : null;
}

export function finishBuckets(pos: readonly number[] | null | undefined): Array<{ label: string; p: number }> {
  if (!pos?.length || pos.some(p => !Number.isFinite(p) || p < 0 || p > 1)) return [];
  const out: Array<{ label: string; p: number }> = [];
  for (let start = 0; start < pos.length; start += 10) out.push({ label: `${start + 1}–${Math.min(start + 10, pos.length)}`, p: pos.slice(start, start + 10).reduce((a, b) => a + b, 0) });
  return out;
}

export function safeEvidenceLink(value: string): string | null {
  if (/^\/api\/[a-zA-Z0-9_/?=&.:%+-]+$/.test(value)) return value;
  try { const u = new URL(value); return u.protocol === "https:" && !u.username && !u.password ? u.href : null; } catch { return null; }
}
