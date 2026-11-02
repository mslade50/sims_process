import assert from "node:assert/strict";
import test from "node:test";
import { STAGE_EVIDENCE_CONTRACT, stageSummaries } from "../app/ops-rules.ts";

const NOW = Date.parse("2026-10-27T12:00:00Z"), START = "2026-10-25";
const iso = (ms) => new Date(ms).toISOString();
const event = { uid: "pga_540_2026", name: "Test event", tour: "pga" };
const euro = { uid: "euro_1_2026", name: "Euro event", tour: "euro" };
// Literal consumer expectations, independent of the exported producer contract.
const weeklyChecks = ["pricing_outputs", "edge_outputs", "report_outputs", "dashboard_publication"];
const job = (over = {}) => ({ id: "j-20261027T110000-aaaaaa", type: "monday_week", requested_at: iso(NOW - 3_600_000), status: { state: "done", exit_code: 0, finished_at: iso(NOW - 600_000) }, ...over });
function receipt(over = {}, scope = event) {
  const occurrence = job().id;
  return { stage: "weekly", event_uid: scope.uid, tour: scope.tour, occurrence_id: occurrence, observed_at: iso(NOW - 60_000), expires_at: iso(NOW + 600_000), state: "verified", explanation: "Checked output contents", actor: "fixture-only", outputs: weeklyChecks.map((name) => ({ name, ok: true, event_uid: scope.uid, occurrence_id: occurrence, checked_at: iso(NOW - 120_000), proof: `r2://golf-model-dashboard-data/fixture/${scope.uid}/${name}.json` })), ...over };
}
const summary = (receipts = [], jobs = [job()], { events = [event], selected = "all", stage = "weekly", now = NOW } = {}) => stageSummaries(jobs, receipts, events, selected, START, now).find((s) => s.id === stage);
const changedOutput = (over) => receipt({ outputs: receipt().outputs.map((o, i) => i === 0 ? { ...o, ...over } : o) });

test("P1: arbitrary occurrences and one unrelated successful output cannot verify a stage", () => {
  const arbitrary = receipt({ occurrence_id: "arbitrary", outputs: [{ ...receipt().outputs[0], name: "unrelated", occurrence_id: "arbitrary" }] });
  assert.notEqual(summary([arbitrary]).state, "verified");
  assert.notEqual(summary([arbitrary], []).state, "verified");
  assert.notEqual(summary([receipt({ outputs: [{ ...receipt().outputs[0], name: "unrelated" }] })]).state, "verified");
  assert.equal(summary([receipt()]).state, "verified");
});

test("required names are complete, unique and successful; a generic or duplicate check cannot substitute", () => {
  for (const name of weeklyChecks) {
    assert.notEqual(summary([receipt({ outputs: receipt().outputs.filter((o) => o.name !== name) })]).state, "verified", name);
  }
  assert.notEqual(summary([changedOutput({ ok: false })]).state, "verified");
  assert.notEqual(summary([receipt({ outputs: [...receipt().outputs, receipt().outputs[0]] })]).state, "verified");
  assert.notEqual(summary([receipt({ outputs: [] })]).state, "verified");
});

test("event, tour, occurrence and safe linked proof must agree for each required output", () => {
  for (const over of [{ event_uid: euro.uid }, { tour: euro.tour }, { occurrence_id: "other" }]) {
    assert.notEqual(summary([receipt(over)]).state, "verified");
  }
  for (const over of [{ event_uid: euro.uid }, { occurrence_id: "other" }, { proof: "javascript:alert(1)" }, { proof: "https://github.com.evil.test/mslade50/x" }]) {
    assert.notEqual(summary([changedOutput(over)]).state, "verified");
  }
});

test("P1: a successful older receipt cannot hide a newer failed, running or queued execution", () => {
  for (const status of [{ state: "failed", exit_code: 1 }, { state: "running" }, { state: "claimed" }, null, { state: "queued" }, { state: "cancelled" }, { state: "expired" }]) {
    const latest = job({ id: "j-20261027T115000-bbbbbb", requested_at: iso(NOW - 600_000), status });
    const result = summary([receipt()], [job(), latest]);
    assert.equal(result.execution.id, latest.id);
    assert.equal(result.state, status?.state === "failed" ? "failed" : ["running", "claimed"].includes(status?.state) ? "running" : "unknown");
  }
});

test("latest successful job still requires its own exact receipt, never a schedule label or previous job", () => {
  const latest = job({ id: "j-20261027T115000-bbbbbb", requested_at: iso(NOW - 600_000), status: { state: "done", exit_code: 0, finished_at: iso(NOW - 300_000) } });
  assert.equal(summary([receipt()], [job(), latest]).state, "completed_unverified");
  assert.notEqual(summary([receipt({ occurrence_id: "monday_week@2026-10-27T11:00:00Z" })]).state, "verified");
  const current = receipt({ occurrence_id: latest.id, outputs: receipt().outputs.map((o) => ({ ...o, occurrence_id: latest.id })) });
  assert.equal(summary([current], [latest, job()]).state, "verified");
});

test("only finished successful execution can verify; missing, invalid or future completion fails closed", () => {
  for (const status of [{ state: "done", exit_code: 0 }, { state: "done", exit_code: 0, finished_at: "invalid" }, { state: "done", exit_code: 0, finished_at: iso(NOW + 120_000) }, { state: "done", exit_code: 0, finished_at: iso(NOW - 3_600_001) }, { state: "done", finished_at: iso(NOW - 600_000) }, { state: "done", exit_code: 2, finished_at: iso(NOW - 600_000) }]) {
    assert.notEqual(summary([receipt()], [job({ status })]).state, "verified", JSON.stringify(status));
  }
  assert.equal(summary([receipt()], []).state, "unknown");
  assert.equal(summary([receipt()], [job({ requested_at: "invalid" })]).state, "unknown");
  assert.equal(summary([receipt()], [job({ requested_at: iso(NOW + 120_000) })]).state, "unknown");
  assert.equal(summary([receipt()], [job({ requested_at: "2026-10-19T12:00:00Z" })]).state, "unknown");
});

test("latest execution and receipt use parsed time rather than lexicographic timestamp order", () => {
  const newer = job({ id: "j-20261027T113000-bbbbbb", requested_at: "2026-10-27T07:30:00-04:00", status: { state: "failed" } });
  assert.equal(summary([receipt()], [job(), newer]).state, "failed");
  const adverse = receipt({ observed_at: "2026-10-27T07:59:30-04:00", state: "needs_attention" });
  assert.equal(summary([receipt(), adverse]).state, "needs_attention");
});

test("P4: expired adverse evidence is stale for every receipt state, including partial multi-event coverage", () => {
  for (const state of ["failed", "needs_attention", "waiting_upstream", "retrying", "skipped", "verified", "running", "unknown", "completed_unverified", "scheduled", "stale"]) {
    const expired = receipt({ state, expires_at: iso(NOW - 1) });
    assert.equal(summary([expired]).state, "stale", state);
    assert.equal(summary([expired], [job()], { events: [event, euro] }).state, "stale", state);
  }
});

test("fresh adverse evidence is scoped to the expected job and cannot mask newer execution failure", () => {
  for (const state of ["failed", "needs_attention", "waiting_upstream", "retrying", "skipped"]) {
    assert.equal(summary([receipt({ state })]).state, state);
    assert.equal(summary([receipt({ state, occurrence_id: "older-job" })]).state, "completed_unverified");
  }
  assert.equal(summary([receipt({ state: "waiting_upstream" })], [job({ status: { state: "failed" } })]).state, "failed");
});

test("expired latest receipt cannot fall back to older success; invalid and overlong expiry are stale", () => {
  for (const expires_at of ["invalid", iso(NOW), iso(NOW + 8 * 86_400_000), iso(NOW - 120_000)]) {
    const latest = receipt({ observed_at: iso(NOW - 30_000), state: "needs_attention", expires_at });
    assert.equal(summary([receipt(), latest]).state, "stale");
  }
});

test("future, stale and pre-execution content checks cannot verify even with a future expiry", () => {
  for (const checked_at of ["invalid", iso(NOW + 120_000), iso(NOW - 360_001), iso(NOW - 600_001)]) {
    assert.notEqual(summary([changedOutput({ checked_at })]).state, "verified", checked_at);
  }
  assert.notEqual(summary([receipt({ outputs: receipt().outputs.map((o) => ({ ...o, checked_at: iso(NOW + 1) })) })]).state, "verified", "checks cannot postdate their observation beyond clock tolerance");
  for (const observed_at of ["invalid", iso(NOW + 120_000), "2026-10-19T12:00:00Z", iso(NOW - 600_001)]) {
    assert.notEqual(summary([receipt({ observed_at })]).state, "verified", observed_at);
  }
});

test("all selected events need complete evidence for the same latest job; selected event stays isolated", () => {
  const euroReceipt = receipt({}, euro);
  assert.equal(summary([receipt(), euroReceipt], [job()], { events: [event, euro] }).state, "verified");
  for (const receipts of [[receipt()], [receipt(), receipt({ outputs: euroReceipt.outputs.slice(1) }, euro)], [receipt(), receipt({ occurrence_id: "other" }, euro)]]) {
    assert.notEqual(summary(receipts, [job()], { events: [event, euro] }).state, "verified");
  }
  assert.equal(summary([receipt(), receipt({ state: "failed" }, euro)], [job()], { events: [event, euro], selected: event.uid }).state, "verified");
  assert.notEqual(summary([receipt()], [job()], { events: [], selected: "all" }).state, "verified");
});

test("non-job inputs and delivery cannot trust a receipt's self-declared expected occurrence", () => {
  for (const stage of ["inputs", "delivery"]) {
    for (const state of ["verified", "failed", "waiting_upstream"]) {
      const result = summary([receipt({ stage, state })], [], { stage });
      assert.equal(result.state, "unknown");
      assert.match(result.explanation, /expected occurrence.*not connected/i);
    }
  }
});

test("ambiguous latest jobs or receipts never verify based on array order", () => {
  const tied = job({ id: "j-20261027T110000-bbbbbb", status: { state: "failed" } });
  for (const jobs of [[job(), tied], [tied, job()]]) assert.notEqual(summary([receipt()], jobs).state, "verified");
  const tiedReceipt = receipt({ state: "failed" });
  for (const receipts of [[receipt(), tiedReceipt], [tiedReceipt, receipt()]]) assert.notEqual(summary(receipts).state, "verified");
});

const stageExpectations = {
  weekly: { type: "monday_week", names: weeklyChecks },
  refresh: { type: "tuesday", names: ["pricing_outputs", "edge_outputs", "report_outputs", "dashboard_publication"] },
  boards: { type: "publish", names: ["board_publication", "report_publication"] },
  checks: { type: "watch", names: ["input_watch", "quote_refresh", "daily_health"] },
  in_event: { type: "after_round", names: ["round_pricing", "cross_tour_refresh"] },
  settlement: { type: "monday_settle", names: ["event_grading", "bet_grading", "settlement_health"] },
};

test("each job-backed stage enforces its own complete named content-check set", () => {
  for (const [stage, { type, names }] of Object.entries(stageExpectations)) {
    const scoped = receipt({ stage, outputs: names.map((name) => ({ ...receipt().outputs[0], name })) });
    const jobs = [job({ type })];
    assert.equal(summary([scoped], jobs, { stage }).state, "verified", stage);
    for (const name of names) {
      assert.notEqual(summary([{ ...scoped, outputs: scoped.outputs.filter((o) => o.name !== name) }], jobs, { stage }).state, "verified", `${stage}/${name}`);
    }
    assert.notEqual(summary([{ ...scoped, outputs: [...scoped.outputs, { ...scoped.outputs[0], name: "unrelated" }] }], jobs, { stage }).state, "verified", stage);
  }
});

test("exported future producer contract explains each required name and marks disconnected occurrence sources", () => {
  const expected = { ...stageExpectations, inputs: { names: ["published_shot_data", "field_inputs", "skill_inputs", "course_fit_inputs"] }, delivery: { names: ["email_delivery", "onedrive_handoff"] } };
  assert.deepEqual(Object.keys(STAGE_EVIDENCE_CONTRACT).sort(), Object.keys(expected).sort());
  for (const [stage, { names }] of Object.entries(expected)) {
    const contract = STAGE_EVIDENCE_CONTRACT[stage];
    assert.deepEqual(Object.keys(contract.required_checks), names);
    assert.equal(contract.occurrence, ["inputs", "delivery"].includes(stage) ? "unconnected" : "latest_job_id");
    assert.ok(Object.values(contract.required_checks).every((description) => typeof description === "string" && description.length > 30));
  }
});

test("newer future-dated job evidence cannot uncover an older verified run",()=>{
 const newer=job({id:"j-20261027T120200-bbbbbb",requested_at:iso(NOW+120_000),status:{state:"failed",exit_code:1}});
 const result=summary([receipt()],[job(),newer]);
 assert.notEqual(result.state,"verified");
 assert.equal(result.execution.id,newer.id);
});

test("newer future-dated receipt evidence cannot uncover an older verified receipt",()=>{
 for(const offset of [120_000,8*86_400_000]){
  const newer=receipt({observed_at:iso(NOW+offset),expires_at:iso(NOW+offset+600_000),state:"failed"});
  const result=summary([receipt(),newer]);
  assert.notEqual(result.state,"verified");
 }
});

test("newer wrong-tour or wrong-occurrence receipts cannot uncover older event success",()=>{
 for(const mismatch of [{tour:"euro"},{occurrence_id:"j-20261027T115000-bbbbbb"}]){
  const newer=receipt({...mismatch,observed_at:iso(NOW-30_000),state:"failed"});
  assert.notEqual(summary([receipt(),newer]).state,"verified");
 }
});
