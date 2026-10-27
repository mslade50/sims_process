/** Owner operations and narrowly authorized Access-service task API. No commands, model updates, notifications or order actions. */
import { accessIdentity, accessServiceIdentity, type AccessEnv } from "./access";
import { ownerEmails } from "./owner";
import { attachExecutions, cleanText, STATE_LABEL, STAGES, isTaskState, OPS_SCHEMA, publicTask, safeProof, sameActor, stageSummaries, summarizeMachines, validWeek, weekSchedule, weekStart, type Actor, type EventScope, type JobEvidence, type OpsTask, type StageReceipt } from "../app/ops-rules";
import { TASK_IDS, seedTask } from "./ops-seed";

export interface OpsEnv extends AccessEnv { OWNER_EMAILS?: string; OPS_AGENT_CLIENT_IDS?: string; DASHBOARD_DATA?: R2Bucket }
const ROOT = "owner/ops/tasks/", MAX_HISTORY = 256, MAX_DOCUMENT = 200_000;
const json = (value: unknown, status = 200) => Response.json(value, { status, headers: { "cache-control": "private, no-store", "vary": "Cf-Access-Jwt-Assertion, Cookie" } });
const fail = (status: number, error: string) => json({ ok: false, error }, status);
const object = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object" && !Array.isArray(v);
const actorValid = (v: unknown): v is Actor => object(v) && Object.keys(v).every((k) => ["kind","id"].includes(k)) && (v.kind === "owner" || v.kind === "agent") && typeof v.id === "string" && v.id.length > 0 && v.id.length <= 254;
const snapshotValid = (v: unknown): boolean => object(v) && Object.keys(v).every((k) => ["state","next_step"].includes(k)) && isTaskState(v.state) && cleanText(v.next_step);
const changesValid = (v: unknown): boolean => object(v) && Object.keys(v).every((k) => ["before","after"].includes(k)) && snapshotValid(v.before) && snapshotValid(v.after);
const iso = (ms: number) => new Date(ms).toISOString();
async function hash(value: string): Promise<string> { const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value)); return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2,"0")).join(""); }
function validRecord(v: unknown, id: string): v is OpsTask {
  if (!object(v) || Object.keys(v).some((k) => !["schema","id","title","state","revision","updated_at","dependencies","next_step","lease","history"].includes(k)) || v.schema !== OPS_SCHEMA || v.id !== id || !isTaskState(v.state) || !Number.isSafeInteger(v.revision) || (v.revision as number) < 0 || !cleanText(v.title,200) || !cleanText(v.next_step) || !Array.isArray(v.dependencies) || !v.dependencies.every((d) => typeof d === "string" && TASK_IDS.includes(d)) || !Array.isArray(v.history) || v.history.length > MAX_HISTORY) return false;
  if (v.updated_at !== null && (typeof v.updated_at !== "string" || !Number.isFinite(Date.parse(v.updated_at)))) return false;
  if (v.lease !== null && (!object(v.lease) || Object.keys(v.lease).some((k) => !["actor","expires_at","token_hash"].includes(k)) || !actorValid(v.lease.actor) || typeof v.lease.expires_at !== "string" || !Number.isFinite(Date.parse(v.lease.expires_at)) || typeof v.lease.token_hash !== "string" || !/^[a-f0-9]{64}$/.test(v.lease.token_hash))) return false;
  return v.history.every((h, i) => object(h) && Object.keys(h).every((k) => ["revision","at","actor","action","note","evidence","changes"].includes(k)) && (!("changes" in h) || changesValid(h.changes)) && h.revision === i + 1 && actorValid(h.actor) && typeof h.at === "string" && Number.isFinite(Date.parse(h.at)) && cleanText(h.action,40) && cleanText(h.note) && Array.isArray(h.evidence) && h.evidence.every(safeProof)) && v.history.length === v.revision;
}
async function readJson(bucket: R2Bucket, key: string, max = MAX_DOCUMENT): Promise<{ value: unknown; etag: string } | null> {
  const got = await bucket.get(key); if (!got) return null;
  if (got.size > max) throw new Error("Stored operations data exceeds its size limit; refusing to replace it.");
  try { return { value: JSON.parse(await got.text()), etag: got.etag }; } catch { throw new Error("Stored operations data is unreadable; refusing to replace it."); }
}
async function readTask(bucket: R2Bucket, id: string): Promise<{ task: OpsTask; etag: string | null }> {
  const got = await readJson(bucket, `${ROOT}${id}.json`);
  if (!got) return { task: seedTask(id)!, etag: null };
  if (!validRecord(got.value, id)) throw new Error("Stored task history is invalid; refusing to replace it.");
  return { task: got.value, etag: got.etag };
}
async function identify(request: Request, env: OpsEnv, now: number): Promise<Actor | null> {
  const owners = ownerEmails(env); if (!owners) return null;
  const user = await accessIdentity(request, env, now);
  if (user?.verified && owners.includes(user.email)) return { kind: "owner", id: user.email };
  const allowed = (env.OPS_AGENT_CLIENT_IDS ?? "").split(",").map((v) => v.trim()).filter(Boolean);
  if (!allowed.length) return null;
  const service = await accessServiceIdentity(request, env, now);
  return service && allowed.includes(service.client_id) ? { kind: "agent", id: service.client_id } : null;
}
async function limitedBody(request: Request): Promise<Record<string, unknown> | null> {
  if (!request.headers.get("content-type")?.toLowerCase().startsWith("application/json")) return null;
  const reader = request.body?.getReader(); if (!reader) return null;
  const chunks: Uint8Array[] = []; let total = 0;
  while (true) { const part = await reader.read(); if (part.done) break; total += part.value.byteLength; if (total > 8192) { await reader.cancel(); return null; } chunks.push(part.value); }
  const bytes = new Uint8Array(total); let at = 0; for (const chunk of chunks) { bytes.set(chunk,at); at += chunk.length; }
  try { const value: unknown = JSON.parse(new TextDecoder().decode(bytes)); return object(value) ? value : null; } catch { return null; }
}
type QueueRow = Record<string, unknown> & { id: string; type: string; requested_at: string };
const queueRowValid = (v: unknown): v is QueueRow => object(v) && typeof v.id === "string" && /^j-\d{8}T\d{6}-[a-f0-9]{6}$/.test(v.id) && cleanText(v.type,100) && typeof v.requested_at === "string" && Number.isFinite(Date.parse(v.requested_at));
function normalizeJobs(rows: QueueRow[], statuses: Map<string, unknown>): JobEvidence[] {
  return rows.map((j) => {
    const s = statuses.get(j.id);
    return { id:j.id, type:j.type, requested_at:j.requested_at, requested_by:typeof j.requested_by === "string" ? j.requested_by : undefined,
      status: object(s) ? { state: typeof s.state === "string" ? s.state : undefined, started_at: typeof s.started_at === "string" ? s.started_at : undefined, finished_at: typeof s.finished_at === "string" ? s.finished_at : undefined, exit_code: typeof s.exit_code === "number" ? s.exit_code : undefined, machine: typeof s.machine === "string" ? s.machine : undefined } : null };
  });
}
function receiptValid(v: unknown): v is StageReceipt {
  return object(v) && STAGES.some((s)=>s.id===v.stage) && typeof v.event_uid === "string" && /^[A-Za-z0-9_:.-]{1,100}$/.test(v.event_uid) && cleanText(v.tour,40) && cleanText(v.occurrence_id,200) && typeof v.observed_at === "string" && Number.isFinite(Date.parse(v.observed_at)) && typeof v.expires_at === "string" && Number.isFinite(Date.parse(v.expires_at)) && typeof v.state === "string" && Object.hasOwn(STATE_LABEL,v.state) && cleanText(v.explanation) && cleanText(v.actor,254) && Array.isArray(v.outputs) && v.outputs.length<=20 && v.outputs.every((o)=>object(o) && cleanText(o.name,200) && typeof o.ok === "boolean" && typeof o.event_uid === "string" && cleanText(o.occurrence_id,200) && typeof o.checked_at === "string" && Number.isFinite(Date.parse(o.checked_at)) && safeProof(o.proof));
}
async function workflow(bucket: R2Bucket, url: URL, now: number): Promise<Response> {
  const start = url.searchParams.get("week") ?? weekStart(now);
  if (!validWeek(start, now)) return fail(400, "Choose a Sunday within the last eight or next four weeks.");
  const selected = url.searchParams.get("event") ?? "all";
  if (selected.length > 100 || !/^[A-Za-z0-9_:.-]+$/.test(selected)) return fail(400, "Invalid event selection.");
  const keys = ["jobs/queue.json", "golfprice/index.json", "owner/ops/observations.json", "jobs/heartbeat/mckinley_home.json", "jobs/heartbeat/desktop-2ki41v6.json"];
  const data = await Promise.all(keys.map((k) => readJson(bucket,k,1_000_000)));
  const queue = data[0]?.value;
  if (queue !== undefined && (!object(queue) || !Array.isArray(queue.jobs) || queue.jobs.length>1000 || !queue.jobs.every(queueRowValid) || new Set(queue.jobs.map((j)=>j.id)).size!==queue.jobs.length)) return fail(503, "The job queue is unreadable. Workflow health is unknown.");
  const raw:QueueRow[] = object(queue) && Array.isArray(queue.jobs) ? (queue.jobs as QueueRow[]).slice().sort((a,b) => Date.parse(b.requested_at)-Date.parse(a.requested_at)).slice(0,32) : [];
  const statuses = await Promise.all(raw.map((j) => readJson(bucket,`jobs/status/${j.id}.json`,100_000)));
  const jobs = normalizeJobs(raw,new Map(raw.map((j,i)=>[j.id,statuses[i]?.value])));
  const index = data[1]?.value;
  if(index!==undefined && (!object(index)||!Array.isArray(index.events)||index.events.length>200||!index.events.every((e)=>object(e)&&typeof e.event_uid==="string"&&/^[A-Za-z0-9_:.-]{1,100}$/.test(e.event_uid)&&cleanText(e.tour,40))||new Set(index.events.map((e)=>e.event_uid)).size!==index.events.length)) return fail(503,"Published event scope contains invalid evidence. Workflow health is unknown.");
  const events: EventScope[] = object(index) && Array.isArray(index.events) ? index.events.filter((e) => object(e) && typeof e.event_uid === "string" && /^[A-Za-z0-9_:.-]{1,100}$/.test(e.event_uid)).map((e) => ({uid:String(e.event_uid), name:cleanText(e.name,150) ? e.name : String(e.event_uid), tour:typeof e.tour === "string" ? e.tour.toLowerCase() : String(e.event_uid).split(/[:_]/)[0]})) : [];
  if (selected !== "all" && !events.some((e) => e.uid === selected)) return fail(400,"That event is not in the current published index.");
  const observation = data[2]?.value;
  let receipts: StageReceipt[] = [];
  if (observation !== undefined) {
    if (!object(observation) || observation.schema !== "golfops.observations.v1" || !Array.isArray(observation.receipts)) return fail(503,"Content-check receipts are unreadable. Workflow health is unknown.");
    if (observation.receipts.length>1000 || !observation.receipts.every(receiptValid)) return fail(503,"Content-check receipts contain invalid evidence. Workflow health is unknown.");
    receipts = observation.receipts;
  }
  const beats = data.slice(3).map((d) => d?.value).filter(object);
  const machines = summarizeMachines(beats,now);
  const aligned = machines.every((m) => m.online && !!m.code) && machines[0].code === machines[1].code;
  return json({ok:true, now:iso(now), week:start, timezone:"America/New_York", events, selected_event:selected, machines, code_aligned:aligned, backup_readiness:"not_verified", stages:stageSummaries(jobs,receipts,events,selected,start,now), schedule:attachExecutions(weekSchedule(start),jobs), jobs_observed:jobs.length, coverage:{execution:"Newest 32 queued job records; older schedule cells remain unverified without receipts.", outputs:observation ? "Stored content checks; exact event/occurrence and freshness required." : "Content-check receipts are not connected yet; exit zero does not mark outputs verified.", non_golfprice:"Producer, export and delivery observations are added as their reviewed instrumentation ships."}});
}
export async function handleOpsApi(request: Request, env: OpsEnv, now = Date.now()): Promise<Response | null> {
  const url = new URL(request.url), path = url.pathname;
  if (!/^\/api\/ops(?:\/|$)/i.test(path)) return null;
  const actor = await identify(request,env,now);
  if (!actor) return fail(403,"Operations is for the verified owner or an explicitly registered agent.");
  if (!env.DASHBOARD_DATA) return fail(503,"Operations storage is unavailable. No progress was changed.");
  const bucket = env.DASHBOARD_DATA;
  if (request.method !== "GET" && request.method !== "POST") return fail(405,"Method not allowed.");
  if (request.method === "POST") {
    const origin = request.headers.get("origin");
    if (origin && origin !== url.origin) return fail(403,"Cross-origin operations writes are refused.");
  }
  try {
    if (path === "/api/ops/workflow" && request.method === "GET") return await workflow(bucket,url,now);
    if (path === "/api/ops/tasks" && request.method === "GET") {
      const records = await Promise.all(TASK_IDS.map((id) => readTask(bucket,id)));
      return json({ok:true,now:iso(now),actor,tasks:records.map((r) => publicTask(r.task)),bootstrap:"/api/ops/bootstrap"});
    }
    if (path === "/api/ops/bootstrap" && request.method === "GET") return json({ok:true,actor,brief:"https://github.com/mslade50/golf_ops/blob/main/docs/ROLLOUT_BRIEF.md",tasks:"/api/ops/tasks",workflow:"/api/ops/workflow",rules:["Read the brief, current tasks and exact event/occurrence evidence before acting.","Claim a task before changes; leases expire after 15 minutes. Keep the returned lease token private and renew before expiry.","Send the exact current revision with every mutation. Conflict means reload and reconcile, never overwrite.","Agents may record work and proof but cannot approve owner decisions. Notes and acknowledgements never establish output health.","Implement, verify in code and browser, obtain critical independent review, resolve findings, commit/push and verify delivery before acceptance.","Do not change model/sizing/provider choices or run real bet/order actions without the owner's specific authority."]});
    const match = /^\/api\/ops\/tasks\/(B\d{2}F?|D0[12])$/.exec(path);
    if (!match || !TASK_IDS.includes(match[1])) return fail(404,"Operations endpoint not found.");
    const id=match[1], {task,etag}=await readTask(bucket,id);
    if (request.method === "GET") return json({ok:true,task:publicTask(task)});
    const body=await limitedBody(request);
    if (!body || Object.keys(body).some((k) => !["op","revision","note","evidence","state","next_step","lease_token"].includes(k)) || !Number.isSafeInteger(body.revision) || body.revision !== task.revision || !cleanText(body.note)) return fail(409,"Reload the task and send its current revision, a clear note, and only supported fields.");
    if (task.history.length >= MAX_HISTORY) return fail(409,"This task's audit history is full. Retain it and arrange a reviewed archive before further updates.");
    const op=body.op, decision=id.startsWith("D"), oldLease=task.lease;
    const active=!!oldLease && Date.parse(oldLease.expires_at)>now;
    const owns=active && oldLease && sameActor(oldLease.actor,actor) && typeof body.lease_token === "string" && await hash(body.lease_token)===oldLease.token_hash;
    let token:string|null=null, action:string;
    const next:OpsTask=structuredClone(task);
    const evidence=body.evidence === undefined ? [] : body.evidence;
    if (!Array.isArray(evidence) || evidence.length>8 || !evidence.every(safeProof)) return fail(400,"Evidence must be bounded approved log, GitHub or R2 references.");
    if (op === "claim") {
      if (decision) return fail(400,"Owner decisions use the decision action, not a work claim.");
      if (active && !owns) return fail(409,"Another live claim owns this task. Wait for expiry or ask its holder to release it.");
      token=owns && typeof body.lease_token === "string" ? body.lease_token : crypto.randomUUID();
      next.lease={actor,expires_at:iso(now+900_000),token_hash:await hash(token)};
      action=owns ? "claim_renewed" : oldLease ? "expired_claim_recovered" : "claimed";
    } else if (op === "decision") {
      if (!decision || actor.kind !== "owner") return fail(403,"Only the verified owner can approve or defer an owner decision.");
      if (body.state !== "verified" && body.state !== "blocked") return fail(400,"An owner decision must explicitly approve exact terms or remain deferred.");
      next.state=body.state; action=body.state==="verified" ? "owner_decision_approved" : "owner_decision_deferred";
      next.next_step=body.note.trim();
    } else if (op === "note" && decision) {
      action="decision_note_only";
    } else if (["note","update","release"].includes(String(op))) {
      if (!owns) return fail(409,"A current task claim and its private lease token are required.");
      if (op === "update") {
        if (!isTaskState(body.state) || (body.state === "verified" && evidence.length === 0)) return fail(400,"Choose a supported task state; verification needs linked proof.");
        if (!cleanText(body.next_step)) return fail(400,"State the next step, including any remaining observation gate.");
        next.state=body.state;next.next_step=body.next_step.trim();
      }
      if (op === "release") next.lease=null;
      action=String(op);
    } else return fail(400,"Unknown operation.");
    next.revision+=1;next.updated_at=iso(now);
    next.history.push({revision:next.revision,at:iso(now),actor,action,note:body.note.trim(),evidence,
      ...(op === "update" || op === "decision" ? {changes:{before:{state:task.state,next_step:task.next_step},after:{state:next.state,next_step:next.next_step}}} : {})});
    const text=JSON.stringify(next);
    if (new TextEncoder().encode(text).length>MAX_DOCUMENT) return fail(409,"Task history is at its size limit. No record was removed.");
    const saved=await bucket.put(`${ROOT}${id}.json`,text,{httpMetadata:{contentType:"application/json",cacheControl:"private, no-store"},onlyIf:etag?{etagMatches:etag}:{etagDoesNotMatch:"*"}});
    if (!saved) return fail(409,"The task changed while saving. Reload and reconcile; nothing was overwritten.");
    return json({ok:true,task:publicTask(next),...(token?{lease_token:token}:{})});
  } catch { return fail(503,"Operations records could not be read or saved safely. Existing records were preserved; check the service before retrying."); }
}
