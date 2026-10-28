/** Reviewed rollout objectives. Private server-side seed, never a packaged public data file.
 * Progress is persisted under owner/ops/tasks/<id>.json; reads do not reset stored records.
 * Scope: golf_ops/docs/ROLLOUT_BRIEF.md, baseline delivery 1b2fad85. */
import { OPS_SCHEMA, type OpsTask, type TaskState } from "../app/ops-rules";
const SEEDS: Array<[string, string, TaskState, string[], string]> = [
  ["B00", "Publish the reviewed execution brief", "verified", [], "Delivered at c65cf6b; scope and review findings are retained in the brief."],
  ["B01", "Reconcile code and preserve existing work", "observation_pending", ["B00"], "Baseline delivered at 1b2fad8. Owner chose remote shared 0.4 Kelly; reconcile component scope and verify release observations."],
  ["B02", "Weekly workflow and shared handoff", "in_progress", ["B01"], "Implement, test, critically review and verify the owner page and bounded agent records."],
  ["B03", "Review and integrate the test cleanup", "planned", ["B01"], "Reconcile 64 merge-base changes and all 100 live-to-candidate paths; verify runtime equivalence."],
  ["B04", "Finish watcher integration", "planned", ["B01", "D01"], "Resolve CI policy, divergent history and final review; verify caps and safe runtime update."],
  ["B05", "Verify the Euro board fix", "planned", ["B03"], "Review current/superseded-event behavior and verify both tours' delivered board artifacts."],
  ["B06", "Connect monitoring and output evidence", "planned", ["B02"], "Connect active schedules, receipts and content checks; shadow delivery until graduation criteria pass."],
  ["B07", "Prepare matching code on big", "planned", ["B03", "B04", "B05"], "Review byte-exact release and verify manifest, model guard, dependencies and rollback."],
  ["B08", "Stage state, inputs and portable jobs", "planned", ["B07"], "Preserve both histories; stage jobs disabled and run isolated parity checks."],
  ["B09", "Prove unattended startup on both machines", "blocked", ["B08"], "Owner enters credentials locally; verify actual scheduled context, logoff/reboot and recovery."],
  ["B09F", "Fence publishing before the switch", "planned", ["B07"], "Verify ownership at every side-effect boundary, including old-primary revival and unavailable authority."],
  ["B10", "Switch the primary to big", "planned", ["B08", "B09", "B09F"], "Use a reviewed quiet-window cutover; small portable tasks stay disabled until backup gates pass."],
  ["B11", "Prove automatic backup and recovery", "planned", ["B10"], "Verify numeric recovery limits, state replication and all required off-peak failure/recovery drills."],
  ["B12", "Verify the complete weekly delivery", "planned", ["B05", "B06", "B11"], "Check current-event boards, emails, course-fit report, OneDrive handoff and real bet-message delivery."],
  ["B13", "Resolve the Pinnacle source blocker", "planned", ["D02"], "Investigate reliable US Pinnacle website access; use the owner-authorized DataGolf fallback if that route cannot be made reliable."],
  ["B14", "Close the rollout with evidence", "planned", ["B12", "B13"], "Close only accepted work; retain named owner, event and observation gates honestly."],
  ["D01", "Retain the remote shared 0.4-Kelly setting", "observation_pending", [], "Owner explicitly chose remote shared 0.4 Kelly in this T3 thread on October 27. Exact authenticated message provenance: golf_ops/docs/ROLLOUT_DECISIONS_20261027.json. Reconcile affected components and verify delivery; no Cloudflare decision action is implied."],
  ["D02", "Verify Pinnacle website access or DataGolf fallback", "observation_pending", [], "Owner directs investigation of reliable US website access, with DataGolf as worst-case fallback. Exact authenticated answer: golf_ops/docs/ROLLOUT_DECISIONS_20261027.json. Verify the chosen source; no Cloudflare decision action is implied."],
];
export const TASK_IDS = SEEDS.map((s) => s[0]);
export function seedTask(id: string): OpsTask | null {
  const item = SEEDS.find((s) => s[0] === id); if (!item) return null;
  return { schema: OPS_SCHEMA, id, title: item[1], state: item[2], revision: 0, updated_at: null, dependencies: [...item[3]], next_step: item[4], lease: null, history: [] };
}
