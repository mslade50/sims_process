"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { EmptyState, LoadingState, PageIntro, Panel } from "./components";
import { useNow } from "./ui";
import { STATE_LABEL, TASK_LABEL, etParts, shiftDate, weekStart, type Actor, type EventScope, type MachineSummary, type Occurrence, type PublicTask, type StageSummary, type TaskState } from "./ops-rules";
import "./ops.css";

type Workflow = { ok: boolean; now: string; week: string; timezone: string; events: EventScope[]; selected_event: string; machines: MachineSummary[]; code_aligned: boolean; backup_readiness: string; stages: StageSummary[]; schedule: Occurrence[]; coverage: Record<string,string>; error?: string };
type Tasks = { ok: boolean; now: string; actor: Actor; tasks: PublicTask[]; error?: string };
const fmt = (value: string | null | undefined) => value ? new Date(value).toLocaleString("en-US", { timeZone: "America/New_York", month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZoneName: "short" }) : "No verified observation";
const dateLabel = (date: string, part: "weekday" | "date") => new Date(`${date}T12:00:00Z`).toLocaleDateString("en-US", part === "weekday" ? {timeZone:"UTC",weekday:"short"} : {timeZone:"UTC",month:"short",day:"numeric"});
const tokenKey = (id: string) => `golfops:lease:${id}`;
async function get<T>(path: string): Promise<T> {
  const r = await fetch(path,{cache:"no-store",credentials:"same-origin",signal:AbortSignal.timeout(10_000)});
  const d = await r.json().catch(() => ({error:"The data service did not return a usable response. Reload to check your sign-in."})) as T & {error?: string};
  if (!r.ok) throw new Error(d.error ?? "Operations data is unavailable."); return d as T;
}
function Proof({value}:{value:string}) { return value.startsWith("https://") || value.startsWith("/api/") ? <a href={value} target="_blank" rel="noreferrer">Open evidence</a> : <code>{value}</code>; }
export function OpsView() {
  const clock=useNow(15_000);
  const [week,setWeek]=useState(()=>weekStart(Date.now())),[event,setEvent]=useState("all"),[tab,setTab]=useState<"week"|"work"|"decisions">("week");
  const [flow,setFlow]=useState<Workflow|null>(null),[tasks,setTasks]=useState<Tasks|null>(null),[error,setError]=useState<string|null>(null),[taskError,setTaskError]=useState<string|null>(null);
  const [selectedStage,setSelectedStage]=useState("weekly"),[selectedTask,setSelectedTask]=useState("B02"),[showChecks,setShowChecks]=useState(false),[loading,setLoading]=useState(true),[busy,setBusy]=useState(false),[notice,setNotice]=useState<string|null>(null);
  const [note,setNote]=useState(""),[nextStep,setNextStep]=useState(""),[taskState,setTaskState]=useState<TaskState>("in_progress"),[evidence,setEvidence]=useState("");
  const [draftBase,setDraftBase]=useState<PublicTask|null>(null);
  const requestRevision=useRef(0);
  const refresh=useCallback(async(isCurrent:()=>boolean=()=>true)=>{
    const revision=++requestRevision.current;
    const current=()=>revision===requestRevision.current&&isCurrent();
    const flowRequest=get<Workflow>(`/api/ops/workflow?week=${week}&event=${encodeURIComponent(event)}`).then((value)=>{if(current()){setFlow(value);setError(null);}},(reason)=>{if(current())setError(reason instanceof Error?reason.message:"Workflow data is unavailable.");}).finally(()=>{if(current())setLoading(false);});
    const taskRequest=get<Tasks>("/api/ops/tasks").then((value)=>{if(current()){setTasks(value);setTaskError(null);}},(reason)=>{if(current())setTaskError(reason instanceof Error?reason.message:"Task records are unavailable.");});
    await Promise.all([flowRequest,taskRequest]);
  },[week,event]);
  useEffect(()=>{let alive=true; const run=()=>{if(alive&&document.visibilityState==="visible")void refresh(()=>alive);}; queueMicrotask(run);const timer=setInterval(run,30_000);return()=>{alive=false;clearInterval(timer);};},[refresh]);
  const chooseWeek=(days:number)=>{++requestRevision.current;setWeek((w)=>shiftDate(w,days));setFlow(null);setLoading(true);};
  const stage=flow?.stages.find((s)=>s.id===selectedStage);
  const task=tasks?.tasks.find((t)=>t.id===selectedTask);
  const selectTask=(id:string)=>{setSelectedTask(id);setNotice(null);setNote("");setEvidence(""); const t=tasks?.tasks.find((x)=>x.id===id); if(t){setDraftBase(structuredClone(t));setTaskState(t.state);setNextStep(t.next_step);}};
  const mutate=async(op:string,state?:TaskState)=>{
    if(!task||!draftBase||draftBase.id!==task.id)return;
    if(task.revision!==draftBase.revision){setNotice("The record changed while this draft was open. Reconcile it before saving.");return;}
    setBusy(true);setNotice(null);
    try{
      const lease=sessionStorage.getItem(tokenKey(task.id));
      const body={op,revision:draftBase.revision,note:op==="claim"?"Claiming this item to work on its recorded next step.":note.trim(),...(lease?{lease_token:lease}:{}),...(state?{state}:{}),...(op==="update"?{next_step:nextStep.trim(),evidence:evidence.split(/\r?\n/).map((v)=>v.trim()).filter(Boolean)}:{})};
      const r=await fetch(`/api/ops/tasks/${task.id}`,{method:"POST",credentials:"same-origin",headers:{"content-type":"application/json"},body:JSON.stringify(body),signal:AbortSignal.timeout(10_000)});
      const d=await r.json() as {error?:string;lease_token?:string;task?:PublicTask};if(!r.ok)throw new Error(d.error??"The record was not changed.");
      if(!d.task||d.task.id!==task.id)throw new Error("The save response was invalid. Reload before retrying.");
      ++requestRevision.current;
      const savedTask=d.task;
      setDraftBase(structuredClone(savedTask));
      if(op==="update"||op==="decision"){setTaskState(savedTask.state);setNextStep(savedTask.next_step);}
      if(d.lease_token)sessionStorage.setItem(tokenKey(task.id),d.lease_token);
      if(op==="release")sessionStorage.removeItem(tokenKey(task.id));
      setTasks((previous)=>previous?{...previous,tasks:previous.tasks.map((t)=>t.id===task.id?savedTask:t)}:previous);
      setNotice(op==="claim"?"Claim saved for 15 minutes. Renew it before expiry.":"Saved with your identity and a new revision.");setNote("");
    }catch(e){setNotice(e instanceof Error?e.message:"The change was not confirmed. Reload before retrying.");await refresh();}finally{setBusy(false);}
  };
  const now=clock,days=Array.from({length:7},(_,i)=>shiftDate(week,i));
  const decision=task?.id.startsWith("D");
  const activeLease=!!task?.lease&&Date.parse(task.lease.expires_at)>clock;
  const ownsLease=activeLease&&task?.lease?.actor.kind===tasks?.actor.kind&&task?.lease?.actor.id===tasks?.actor.id&&typeof window!=="undefined"&&!!sessionStorage.getItem(tokenKey(task.id));
  const draftStale=!!task&&(!draftBase||draftBase.id!==task.id||draftBase.revision!==task.revision);
  const flowAge=flow?clock-Date.parse(flow.now):Infinity;
  const healthStale=!!error||!Number.isFinite(flowAge)||flowAge>60_000||flowAge< -60_000;
  const recordBlocked=busy||draftStale||!!taskError;
  const stageState=(value:StageSummary)=>healthStale?"unknown":value.valid_until&&Date.parse(value.valid_until)<=clock?"stale":value.state;
  return <div className="ops-page">
    <PageIntro eyebrow="Owner operations" title="Weekly workflow" description="What has happened, what is due next, and what needs attention. Work records and decisions stay here for the next agent." controls={<button type="button" className="inputs-button" onClick={()=>void refresh()}>Refresh</button>}/>
    <div className="ops-toolbar">
      <div className="segmented" role="group" aria-label="Operations view">{([["week","The week"],["work","Rollout work"],["decisions","Owner decisions"]] as const).map(([key,label])=><button key={key} type="button" className={tab===key?"active":""} disabled={busy} onClick={()=>{setTab(key);if(key==="decisions")selectTask("D01");else if(key==="work")selectTask(selectedTask.startsWith("D")?"B02":selectedTask);}}>{label}</button>)}</div>
      <span className="ops-updated">Checked {fmt(flow?.now)}{error?" - refresh failed; previous view is stale":""}</span>
    </div>
    {healthStale&&flow&&<div className="inputs-banner warn" role="alert"><strong>Current workflow health is unknown.</strong> {error??"The displayed response is no longer fresh."} A previously loaded view is not fresh evidence.</div>}
    {error&&!flow&&<div className="inputs-banner warn" role="alert">{error}</div>}
    {taskError&&<div className="inputs-banner warn" role="alert">Task records: {taskError} Changes are unavailable until records load safely.</div>}
    {tab==="week"&&<>
      <div className="ops-week-controls"><div className="ops-week-picker"><button type="button" className="inputs-button" onClick={()=>chooseWeek(-7)} aria-label="Previous week">Previous</button><strong>{dateLabel(week,"date")} - {dateLabel(shiftDate(week,6),"date")}</strong><button type="button" className="inputs-button" onClick={()=>chooseWeek(7)} aria-label="Next week">Next</button><button type="button" className="ex-link" onClick={()=>{const current=weekStart(Date.now());if(current===week){void refresh();return;}++requestRevision.current;setWeek(current);setFlow(null);setLoading(true);}}>This week</button></div>
        <label>Event<select aria-label="Workflow event" value={event} onChange={(e)=>{++requestRevision.current;setEvent(e.target.value);setFlow(null);setLoading(true);}}><option value="all">All published events</option>{flow?.events.map((e)=><option value={e.uid} key={e.uid}>{e.name} ({e.tour.toUpperCase()})</option>)}</select></label>
      </div>
      {loading&&!flow?<LoadingState label="Loading live workflow records"/>:!flow?<EmptyState title="Workflow records are unavailable" detail="Sign in as the owner and refresh. No stage is assumed healthy." onRetry={()=>void refresh()}/>:<>
        <div className="ops-machines">{flow.machines.map((m)=><div key={m.id} className="ops-machine"><span className={`ops-dot ${m.online&&!healthStale&&clock-Date.parse(m.at??"")<=600_000?"verified":"failed"}`} aria-hidden="true"/><div><strong>{m.label}</strong><p>{healthStale?"Health unverified":m.online&&clock-Date.parse(m.at??"")<=600_000?"Online":"No fresh heartbeat"} - {m.role}</p><small>Last seen {fmt(m.at)}</small></div></div>)}<div className="ops-backup"><strong>Backup readiness: not verified</strong><p>{flow.code_aligned?"Reported code matches; unattended, state and takeover proof are still required.":"A matching, proven standby has not been established."}</p></div></div>
        <Panel eyebrow="Delivery chain" title="Current standing"><p className="ops-explain">A finished job stays unverified until its fresh output checks match the selected event. Each step lists its dependencies.</p>
          <div className="ops-flow" aria-label="Weekly workflow dependencies">{flow.stages.map((s)=><button type="button" key={s.id} className={`ops-node ${selectedStage===s.id?"selected":""}`} aria-pressed={selectedStage===s.id} onClick={()=>setSelectedStage(s.id)}><strong>{s.label}</strong><span className={`ops-status ${stageState(s)}`}>{STATE_LABEL[stageState(s)]}</span><small>{s.dependencies.length?`Needs ${s.dependencies.map((id)=>flow.stages.find((v)=>v.id===id)?.label??id).join(" + ")}`:"Starts the chain"}</small></button>)}</div>
          {stage&&<div className="ops-stage-detail" aria-live="polite"><h3>{stage.label}</h3><p>{stage.detail}</p><p><strong>{STATE_LABEL[stageState(stage)]}.</strong> {healthStale?"Refresh is unavailable or expired; displayed evidence is stale.":stageState(stage)==="stale"&&stage.state!=="stale"?"The content-check receipt expired while this page was open.":stage.explanation}</p><small>Evidence observed {fmt(stage.evidence_at)}</small>{stage.execution&&<p>Latest observed job: {stage.execution.type} - {stage.execution.status?.state??"queued"} <a className="ex-link" href={`/api/jobs/${stage.execution.id}/log`} target="_blank" rel="noreferrer">Open job log</a></p>}{stage.proof.map((p)=><p key={p}><Proof value={p}/></p>)}</div>}
        </Panel>
        <Panel eyebrow="America/New_York" title="Planned and observed runs"><div className="ops-calendar-options"><label className="run-check"><input type="checkbox" checked={showChecks} onChange={(e)=>setShowChecks(e.target.checked)}/>Show the frequent watch and odds checks</label><small>Grey means planned. An overdue cell without evidence is unverified, not a successful run.</small></div>
          <div className="ops-calendar">{days.map((day)=>{const all=flow.schedule.filter((o)=>o.local_day===day),visible=all.filter((o)=>showChecks||!o.frequent),hidden=all.length-visible.length;return <section className={`ops-day ${etParts(now).date===day?"today":""}`} key={day} aria-label={`${dateLabel(day,"weekday")} ${dateLabel(day,"date")}`}><header><strong>{dateLabel(day,"weekday")}</strong><span>{dateLabel(day,"date")}</span></header><div className="ops-day-events">{visible.length?visible.map((o)=>{const future=Date.parse(o.scheduled_at)>now;const state=healthStale?(future?"scheduled":"unknown"):o.execution_state==="done"?"completed_unverified":o.execution_state==="failed"?"failed":["running","claimed"].includes(o.execution_state??"")?"running":future?"scheduled":"unknown";return <details className="ops-occurrence" key={o.id}><summary><time dateTime={o.scheduled_at}>{o.local_time}</time><strong>{o.label}</strong><span className={`ops-status ${state}`}>{STATE_LABEL[state]}</span></summary><p>{o.why}</p>{o.job_id&&<a href={`/api/jobs/${o.job_id}/log`} target="_blank" rel="noreferrer">Open observed job log</a>}<small>{fmt(o.scheduled_at)}</small></details>}):<p className="ops-empty-day">No fixed pricing slot</p>}{hidden>0&&<p className="ops-frequent">{hidden} recurring checks scheduled</p>}</div></section>})}</div>
        </Panel>
        <details className="tech-details"><summary>Evidence coverage and machine versions</summary>{Object.entries(flow.coverage).map(([k,v])=><p key={k}>{v}</p>)}<p>Published events: {flow.events.map((e)=>`${e.name} (${e.uid})`).join(", ")||"No event scope loaded; verification stays unknown."}</p>{flow.machines.map((m)=><p key={m.id}>{m.label}: commit <code>{m.commit??"unreported"}</code>, code <code>{m.code??"unreported"}</code></p>)}<p>Schedule comes from the same slot/window definitions as the job scheduler. Jobs may legitimately wait for upstream data or have nothing to do; content receipts must make that distinction.</p></details>
      </>}
    </>}
    {tab!=="week"&&<Panel eyebrow={tab==="decisions"?"Owner decisions":"Shared work record"} title={tab==="decisions"?"Decisions that unblock work":"Rollout checklist"}>
      <p className="ops-explain">Task status records work and evidence; it does not override measured workflow health. Claims expire after 15 minutes. Decisions require the verified owner.</p>
      {!tasks?<LoadingState label="Loading retained task history"/>:<div className="ops-work-layout"><div className="ops-task-list" aria-label="Work items">{tasks.tasks.filter((t)=>tab==="decisions"?t.id.startsWith("D"):!t.id.startsWith("D")).map((t)=><button type="button" key={t.id} className={`ops-task ${selectedTask===t.id?"selected":""}`} disabled={busy} onClick={()=>selectTask(t.id)}><span>{t.id}</span><strong>{t.title}</strong><small>{TASK_LABEL[t.state]}</small></button>)}</div>
        {task&&<div className="ops-task-detail"><h3>{task.id} - {task.title}</h3><span className={`ops-status ${task.state==="verified"?"verified":task.state==="blocked"?"needs_attention":"unknown"}`}>{TASK_LABEL[task.state]}</span><p>{task.next_step}</p><p className="ops-claim">{activeLease?`Claim: ${task.lease?.actor.kind} ${task.lease?.actor.id}; expires ${fmt(task.lease?.expires_at)}`:"No active claim"}</p>{!decision&&<button type="button" className="inputs-button" disabled={recordBlocked||(activeLease&&!ownsLease)} onClick={()=>void mutate("claim")}>{activeLease?ownsLease?"Renew my claim":"Claim held by another session":"Claim this task"}</button>}
          {draftStale&&<div className="inputs-banner warn" role="alert">This record changed after the draft was opened. Review the current state and history below before replacing your draft.<button type="button" className="inputs-button" onClick={()=>selectTask(task.id)}>Reload current record and discard draft</button></div>}
          <form className="ops-editor" onSubmit={(e)=>{e.preventDefault();void mutate("note");}}><label>{decision?"Exact decision terms or context":"Handoff note"}<textarea aria-label={decision?"Exact decision terms or context":"Handoff note"} disabled={busy} value={note} onChange={(e)=>setNote(e.target.value)} maxLength={2000} rows={4} required placeholder={decision?"Name the affected components and exact approved setting, or explain the remaining blocker.":"What changed, what was checked, what remains, and who acts next."}/></label>
            {!decision&&<><label>Task state<select aria-label="Task state" disabled={busy} value={taskState} onChange={(e)=>setTaskState(e.target.value as TaskState)}>{Object.entries(TASK_LABEL).map(([v,l])=><option key={v} value={v}>{l}</option>)}</select></label><label>Next step<textarea aria-label="Next step" disabled={busy} value={nextStep} onChange={(e)=>setNextStep(e.target.value)} rows={2} maxLength={2000}/></label><label>Proof links, one per line<textarea aria-label="Proof links, one per line" disabled={busy} value={evidence} onChange={(e)=>setEvidence(e.target.value)} rows={2} placeholder="GitHub commit/review, approved job-log path, or R2 object reference"/></label></>}
            <div className="ops-actions"><button className="inputs-button" type="submit" disabled={recordBlocked||!note.trim()}>Save note</button>{decision?<><button className="inputs-button primary" type="button" disabled={recordBlocked||!note.trim()||tasks.actor.kind!=="owner"} onClick={()=>void mutate("decision","verified")}>Approve these exact terms</button><button className="inputs-button" type="button" disabled={recordBlocked||!note.trim()||tasks.actor.kind!=="owner"} onClick={()=>void mutate("decision","blocked")}>Keep blocked</button></>:<><button className="inputs-button primary" type="button" disabled={recordBlocked||!note.trim()} onClick={()=>void mutate("update",taskState)}>Save state and next step</button><button className="inputs-button" type="button" disabled={recordBlocked||!note.trim()} onClick={()=>void mutate("release")}>Release claim</button></>}</div>
          </form>{notice&&<p className="inputs-banner" role="status">{notice}</p>}
          <details className="ops-history" open><summary>Decision and handoff history ({task.history.length})</summary>{task.history.length?<ol>{[...task.history].reverse().map((h)=><li key={h.revision}><strong>{h.action.replaceAll("_"," ")} - {h.actor.kind}</strong><small>{fmt(h.at)} - {h.actor.id} - revision {h.revision}</small><p>{h.note}</p>{h.changes&&<div><p>State: {TASK_LABEL[h.changes.before.state]} → {TASK_LABEL[h.changes.after.state]}</p><p>Previous next step: {h.changes.before.next_step}</p><p>Recorded next step: {h.changes.after.next_step}</p></div>}{h.evidence.map((p)=><p key={p}><Proof value={p}/></p>)}</li>)}</ol>:<p>This item is seeded from the reviewed brief. No recorded API update yet.</p>}</details><details className="tech-details"><summary>Agent bootstrap and dependencies</summary><p>Read <a href="/api/ops/bootstrap" target="_blank" rel="noreferrer">the authenticated agent handoff</a> before acting. Dependencies: {task.dependencies.join(", ")||"none"}. Revision {task.revision}.</p><p>Never paste passwords, tokens, private credentials or raw secret-bearing logs into notes.</p></details>
        </div>}
      </div>}
    </Panel>}
  </div>;
}