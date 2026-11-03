import assert from "node:assert/strict";
import test from "node:test";
import { finishBuckets, matchingExplain, safeEvidenceLink, stageSignal, workflowFor } from "../app/walkthrough-rules.ts";
import { navigationFor, ownerOnlyPath } from "../app/shell-rules.ts";

const now = Date.parse("2026-11-03T12:00:00Z");
const stage = { id: "weekly", state: "verified", explanation: "Matching delivered outputs checked.", evidence_at: new Date(now - 1000).toISOString(), valid_until: new Date(now + 60_000).toISOString(), proof: ["/api/ops/receipt"] };
const flow = { ok: true, selected_event: "pga:457:2026-11-05", week: "2026-11-01", now: new Date(now).toISOString(), stages: [stage] };

test("weekly lights fail closed on wrong scope, stale snapshots and missing output proof", () => {
  assert.equal(workflowFor(flow, flow.selected_event, flow.week)?.ok, true);
  assert.equal(workflowFor(flow, "other-event", flow.week), null);
  assert.equal(workflowFor(flow, flow.selected_event, "2026-10-25"), null);
  assert.equal(workflowFor({ ...flow, stages: [stage, stage] }, flow.selected_event, flow.week), null);
  assert.equal(stageSignal(stage, flow, now).tone, "green");
  for (const s of [{ ...stage, proof: [] }, { ...stage, valid_until: null }, { ...stage, valid_until: new Date(now).toISOString() }, { ...stage, evidence_at: "bad" }, { ...stage, state: "completed_unverified" }]) assert.equal(stageSignal(s, flow, now).tone, "yellow");
  assert.equal(stageSignal(stage, flow, now + 60_001).tone, "yellow");
  assert.equal(stageSignal(stage, { ...flow, now: new Date(now + 60_001).toISOString() }, now).tone, "yellow");
  assert.equal(stageSignal(stage, flow, now, "Network failed").tone, "yellow");
  assert.equal(stageSignal(undefined, flow, now).tone, "yellow");
});

test("new failures remain red and unrecognized states never turn green", () => {
  for (const state of ["failed", "needs_attention"]) assert.equal(stageSignal({ ...stage, state }, flow, now).tone, "red");
  for (const state of ["running", "scheduled", "retrying", "unknown", "skipped", "invented"]) assert.equal(stageSignal({ ...stage, state }, flow, now).tone, "yellow");
});

test("a document must match both the selected event and indexed run", () => {
  const doc = { event_uid: flow.selected_event, kind: "week", run: "20261103T051600Z" };
  const event = { event_uid: flow.selected_event, explain_run: "week_20261103T051600Z" };
  assert.equal(matchingExplain(doc, event), doc);
  assert.equal(matchingExplain({ ...doc, event_uid: "euro:other" }, event), null);
  assert.equal(matchingExplain({ ...doc, run: "older" }, event), null);
  assert.equal(matchingExplain(doc, { event_uid: event.event_uid }), null);
});

test("finish bins conserve probability, retain a short last bin and reject invalid values", () => {
  const bins = finishBuckets(Array(23).fill(1 / 23));
  assert.deepEqual(bins.map(b => b.label), ["1–10", "11–20", "21–23"]);
  assert.ok(Math.abs(bins.reduce((s, b) => s + b.p, 0) - 1) < 1e-12);
  for (const p of [[NaN], [-.1], [1.1], null]) assert.deepEqual(finishBuckets(p), []);
});

test("walkthrough stays owner-only, including RSC and case variants", () => {
  for (const path of ["/model-walkthrough", "/MODEL-WALKTHROUGH/", "/model-walkthrough.rsc"]) assert.equal(ownerOnlyPath(path), "page");
  assert.equal(ownerOnlyPath("/model-walkthrough-other"), null);
  assert.ok(navigationFor(true).flatMap(g => g.items).some(i => i.key === "model-walkthrough"));
  assert.ok(!navigationFor(false).flatMap(g => g.items).some(i => i.key === "model-walkthrough"));
});

test("evidence links reject script schemes, protocol-relative URLs and credentials", () => {
  for (const value of ["javascript:alert(1)", "//attacker.invalid", "https://user:pass@example.com", "/outside"]) assert.equal(safeEvidenceLink(value), null);
  assert.equal(safeEvidenceLink("/api/ops/workflow?event=pga%3A457"), "/api/ops/workflow?event=pga%3A457");
  assert.equal(safeEvidenceLink("https://example.com/receipt"), "https://example.com/receipt");
});
