import assert from "node:assert/strict";
import { createSign, generateKeyPairSync } from "node:crypto";
import test from "node:test";
import { attachExecutions, safeProof, stageSummaries, summarizeMachines, validWeek, weekSchedule, weekStart } from "../app/ops-rules.ts";

const NOW=Date.parse("2026-10-27T12:00:00Z"), START="2026-10-25", TEAM="ops-test.cloudflareaccess.com", AUD="ops-aud", OWNER="owner@example.com", CLIENT="registered-test.access";
const {publicKey,privateKey}=generateKeyPairSync("rsa",{modulusLength:2048});
const b64=(v)=>Buffer.from(JSON.stringify(v)).toString("base64url");
function auth({email=OWNER,client,expiry=NOW+3_600_000,aud=AUD}={}) {
  const h=b64({alg:"RS256",kid:"ops-key"});
  const p=b64({aud:[aud],iss:`https://${TEAM}`,exp:Math.floor(expiry/1000),...(client?{common_name:client,sub:""}:{email,sub:"owner-1"})});
  const sig=createSign("RSA-SHA256").update(`${h}.${p}`).sign(privateKey).toString("base64url");
  return {"cf-access-jwt-assertion":`${h}.${p}.${sig}`,...(!client?{"cf-access-authenticated-user-email":email}:{})};
}
class Bucket {
  objects=new Map();counter=0;reads=0;writes=0;conflict=false;
  async get(key){this.reads++;const v=this.objects.get(key);return v?{etag:v.etag,size:Buffer.byteLength(v.text),text:async()=>v.text}:null;}
  async head(key){const v=this.objects.get(key);return v?{etag:v.etag}:null;}
  async list(){return {objects:[],truncated:false};}
  async put(key,text,options){
    this.writes++;assert.ok(options?.onlyIf,"every progress mutation must use conditional writes");
    if(this.conflict){this.conflict=false;return null;}
    const old=this.objects.get(key);
    if(options.onlyIf.etagDoesNotMatch==="*"&&old)return null;
    if(options.onlyIf.etagMatches&&old?.etag!==options.onlyIf.etagMatches)return null;
    const etag=`v${++this.counter}`;this.objects.set(key,{text,etag});return {etag};
  }
  seed(key,value){this.objects.set(key,{text:JSON.stringify(value),etag:`v${++this.counter}`});}
}
let worker;
const originalFetch=globalThis.fetch, originalNow=Date.now;
test.before(async()=>{
  Date.now=()=>NOW;
  globalThis.fetch=async(input,init)=>{const url=typeof input==="string"?input:input.url;if(url===`https://${TEAM}/cdn-cgi/access/certs`)return Response.json({keys:[{...publicKey.export({format:"jwk"}),kid:"ops-key"}]});return originalFetch(input,init);};
  worker=(await import(`../dist/server/index.js?ops=${Date.now()}`)).default;
});
test.after(()=>{globalThis.fetch=originalFetch;Date.now=originalNow;});
async function call(bucket,path,{body,headers=auth(),now=NOW,owners=OWNER,clients=CLIENT,method=body?"POST":"GET"}={}) {
  const request=new Request(`https://golf.example${path}`,{method,headers:{...headers,...(body?{"content-type":"application/json"}:{})},...(body?{body:JSON.stringify(body)}:{})});
  // The built Worker uses real time. JWTs are valid around NOW for handler tests; freeze Date.now only within this process.
  const old=Date.now;Date.now=()=>now;
  try{return await worker.fetch(request,{DASHBOARD_DATA:bucket,ASSETS:{fetch:async()=>new Response("nf",{status:404})},ACCESS_TEAM_DOMAIN:TEAM,ACCESS_AUD:AUD,...(owners?{OWNER_EMAILS:owners}:{}),...(clients?{OPS_AGENT_CLIENT_IDS:clients}:{})},{waitUntil(){},passThroughOnException(){}});}finally{Date.now=old;}
}
const note="Reviewed this task and recorded the next safe action.";
const path=(id)=>`/api/ops/tasks/${id}`;

test("schedule is the actual slot/window definition and handles both DST transitions",()=>{
  const fall=weekSchedule("2026-11-01").filter((o)=>o.type==="watch"&&o.local_day==="2026-11-01");
  const spring=weekSchedule("2026-03-08").filter((o)=>o.type==="watch"&&o.local_day==="2026-03-08");
  assert.equal(fall.length,100);assert.equal(spring.length,92);assert.equal(new Set(fall.map((o)=>o.id)).size,100);
  assert.ok(weekSchedule(START).some((o)=>o.type==="tuesday"&&o.local_day==="2026-10-27"&&o.local_time==="12:00"));
  assert.equal(weekStart(NOW),START);assert.equal(validWeek("2026-10-26",NOW),false);assert.equal(validWeek("2020-01-05",NOW),false);
});
const event={uid:"pga_540_2026",name:"Test event",tour:"pga"};
const receipt=(over={})=>({stage:"weekly",event_uid:event.uid,tour:"pga",occurrence_id:"j-20261027T110000-aaaaaa",observed_at:new Date(NOW).toISOString(),expires_at:new Date(NOW+600_000).toISOString(),state:"verified",explanation:"checked",actor:"test-agent",outputs:["pricing_outputs","edge_outputs","report_outputs","dashboard_publication"].map((name)=>({name,ok:true,event_uid:event.uid,occurrence_id:"j-20261027T110000-aaaaaa",checked_at:new Date(NOW).toISOString(),proof:"r2://golf-model-dashboard-data/golfprice/test/manifest.json"})),...over});
const weekly=(receipts=[],jobs=[])=>stageSummaries(jobs,receipts,[event],"all",START,NOW).find((s)=>s.id==="weekly");
test("exit zero cannot establish output verification; event, occurrence, output proof and freshness must match",()=>{
  const jobs=[{id:"j-20261027T110000-aaaaaa",type:"monday_week",requested_at:"2026-10-27T11:00:00Z",status:{state:"done",exit_code:0,finished_at:"2026-10-27T11:30:00Z"}}];
  assert.equal(weekly([],jobs).state,"completed_unverified");assert.equal(weekly([receipt()],jobs).state,"verified");
  assert.equal(weekly([receipt({event_uid:"pga_wrong"})],jobs).state,"completed_unverified");
  assert.equal(weekly([receipt({tour:"euro"})],jobs).state,"completed_unverified");
  assert.equal(weekly([receipt({outputs:[]})],jobs).state,"completed_unverified");
  assert.equal(weekly([receipt({outputs:[{...receipt().outputs[0],occurrence_id:"other"}]})],jobs).state,"completed_unverified");
  assert.equal(weekly([receipt({observed_at:new Date(NOW-60_000).toISOString(),expires_at:new Date(NOW-1).toISOString()})],jobs).state,"stale");
  assert.equal(weekly([receipt({observed_at:"2026-10-19T12:00:00Z"})],jobs).state,"completed_unverified");
  assert.equal(weekly([receipt()]).state,"unknown");
  assert.equal(weekly([], [{...jobs[0],requested_at:"invalid"}]).state,"unknown");
});
test("one event cannot conceal another event's missing evidence, and heartbeat alignment is not failover proof",()=>{
  const second={uid:"euro_1_2026",name:"Euro",tour:"euro"};
  assert.notEqual(stageSummaries([{id:"j-20261027T110000-aaaaaa",type:"monday_week",requested_at:"2026-10-27T11:00:00Z",status:{state:"done",exit_code:0,finished_at:"2026-10-27T11:30:00Z"}}], [receipt()], [event,second],"all",START,NOW).find((s)=>s.id==="weekly").state,"verified");
  assert.equal(summarizeMachines([{machine:"mckinley_home",at:"bad"}],NOW)[0].online,false);
  assert.equal(safeProof("javascript:alert(1)"),false);assert.equal(safeProof("https://github.com.evil.test/mslade50/x"),false);
});
test("manual requests do not become scheduled occurrence evidence",()=>{
  const schedule=weekSchedule(START);
  const job={id:"j-20261027T160012-aaaaaa",type:"tuesday",requested_at:"2026-10-27T16:00:12Z",requested_by:"owner",status:{state:"done"}};
  assert.equal(attachExecutions(schedule,[job]).find((o)=>o.type==="tuesday").job_id,undefined);
  assert.equal(attachExecutions(schedule,[{...job,requested_by:"cron"}]).find((o)=>o.type==="tuesday").job_id,job.id);
});
test("anonymous, colleague, forged service, unregistered service and unset owner configuration fail before storage",async()=>{
  for(const options of [{headers:{}},{headers:auth({email:"colleague@example.com"})},{headers:{"cf-access-jwt-assertion":"fake"}},{headers:auth({client:"other.access"})},{headers:auth({client:CLIENT}),clients:null},{headers:auth(),owners:null},{headers:auth({client:CLIENT,aud:"wrong"})},{headers:auth({client:CLIENT,expiry:NOW-1})}]){
    const b=new Bucket();assert.equal((await call(b,"/api/ops/tasks",options)).status,403);assert.equal(b.reads,0);assert.equal(b.writes,0);
  }
  const b=new Bucket();assert.equal((await call(b,"/ops",{headers:{},owners:null})).status,302);
  assert.equal((await call(b,"/OPS.rsc/",{headers:auth({email:"other@example.com"})})).status,302);
});
test("owner and registered service read the same private records; no agent token enables owner decisions",async()=>{
  const b=new Bucket();const owner=await (await call(b,"/api/ops/tasks")).json();
  const agent=await (await call(b,"/api/ops/tasks",{headers:auth({client:CLIENT})})).json();
  assert.equal(owner.actor.kind,"owner");assert.equal(agent.actor.kind,"agent");assert.equal(agent.tasks.length,18);
  assert.equal((await call(b,path("D01"),{headers:auth({client:CLIENT}),body:{op:"decision",revision:0,note:"Set the shared fraction as requested",state:"verified"}})).status,403);
  assert.equal(b.writes,0);
});
test("claim excludes competitors, renews only with its token, and expiry recovers an abandoned claim",async()=>{
  const b=new Bucket();const first=await (await call(b,path("B02"),{body:{op:"claim",revision:0,note}})).json();
  assert.ok(first.lease_token);assert.equal(first.task.revision,1);assert.equal("token_hash"in first.task.lease,false);
  assert.equal((await call(b,path("B02"),{body:{op:"claim",revision:1,note}})).status,409);
  const renew=await (await call(b,path("B02"),{now:NOW+60_000,body:{op:"claim",revision:1,note,lease_token:first.lease_token}})).json();
  assert.equal(renew.task.history.at(-1).action,"claim_renewed");
  const recovered=await (await call(b,path("B02"),{headers:auth({client:CLIENT}),now:NOW+1_100_000,body:{op:"claim",revision:2,note}})).json();
  assert.equal(recovered.task.history.at(-1).action,"expired_claim_recovered");assert.equal(recovered.task.lease.actor.kind,"agent");
  assert.equal((await call(b,path("B02"),{body:{op:"note",revision:3,note,lease_token:first.lease_token},now:NOW+1_100_001})).status,409);
});
test("racing claims have one winner and a conditional-write conflict never overwrites retained history",async()=>{
  const b=new Bucket();const results=await Promise.all([call(b,path("B03"),{body:{op:"claim",revision:0,note}}),call(b,path("B03"),{headers:auth({client:CLIENT}),body:{op:"claim",revision:0,note}})]);
  assert.deepEqual(results.map((r)=>r.status).sort(),[200,409]);
  const stored=b.objects.get("owner/ops/tasks/B03.json").text;b.conflict=true;
  const read=JSON.parse(stored);assert.equal(read.history.length,1);
  assert.equal((await call(b,path("B03"),{body:{op:"claim",revision:read.revision,note},now:NOW+1_000_000})).status,409);
  assert.equal(b.objects.get("owner/ops/tasks/B03.json").text,stored);
});
test("task verification needs proof, stale revisions fail, and task state does not change workflow health",async()=>{
  const b=new Bucket();const first=await (await call(b,path("B02"),{body:{op:"claim",revision:0,note}})).json();
  const update={op:"update",revision:1,note,state:"verified",next_step:"Observe the next real production occurrence",lease_token:first.lease_token};
  assert.equal((await call(b,path("B02"),{body:update})).status,400);
  assert.equal((await call(b,path("B02"),{body:{...update,revision:0,evidence:["https://github.com/mslade50/golf_ops/commit/c65cf6b"]}})).status,409);
  const saved=await (await call(b,path("B02"),{body:{...update,evidence:["https://github.com/mslade50/golf_ops/commit/c65cf6b"]}})).json();
  assert.equal(saved.task.state,"verified");assert.equal(saved.task.history.length,2);
  const f=await (await call(b,"/api/ops/workflow")).json();assert.ok(f.stages.every((s)=>s.state!=="verified"));assert.equal(f.backup_readiness,"not_verified");
});
test("only the authenticated owner records exact decision terms with append-only provenance",async()=>{
  const b=new Bucket();const r=await (await call(b,path("D01"),{body:{op:"decision",revision:0,note:"Keep existing component sizing until a separate reviewed decision.",state:"blocked"}})).json();
  assert.equal(r.task.state,"blocked");assert.equal(r.task.history[0].actor.id,OWNER);assert.equal(r.task.history[0].action,"owner_decision_deferred");
  assert.equal((await call(b,path("D01"),{body:{op:"update",revision:1,note,state:"verified",next_step:note,evidence:["https://github.com/mslade50/golf_ops"]}})).status,409);
});
test("unknown command fields, cross-origin writes, corrupt history and secret-shaped notes are refused",async()=>{
  const b=new Bucket();assert.equal((await call(b,path("B02"),{body:{op:"claim",revision:0,note,command:"not allowed"}})).status,409);
  assert.equal((await call(b,path("B02"),{headers:{...auth(),origin:"https://other.example"},body:{op:"claim",revision:0,note}})).status,403);
  assert.equal((await call(b,path("B02"),{body:{op:"claim",revision:0,note:"Bearer "+"a".repeat(32)}})).status,409);
  b.seed("owner/ops/tasks/B02.json",{schema:"golfops.task.v1",id:"B02",history:[],revision:10});
  assert.equal((await call(b,path("B02"),{body:{op:"claim",revision:0,note}})).status,503);assert.equal(b.writes,0);
});
test("malformed queue cannot produce a healthy or silently empty workflow",async()=>{
  const b=new Bucket();b.seed("jobs/queue.json",{broken:true});assert.equal((await call(b,"/api/ops/workflow")).status,503);
  assert.equal((await call(new Bucket(),"/api/ops/workflow?week=2026-10-26")).status,400);
});

test("a signature-verified owner reaches the actual operations page shell",async()=>{
  const r=await call(new Bucket(),"/ops");
  assert.equal(r.status,200);assert.match(r.headers.get("content-type")??"",/text\/html/);assert.match(await r.text(),/Weekly workflow/);
});


test("malformed queue rows cannot shift status pairing or make a failed run verified",async()=>{
 const b=new Bucket();
 const failed={id:"j-20261027T110000-aaaaaa",type:"monday_week",requested_at:"2026-10-27T11:00:00Z"};
 b.seed("jobs/queue.json",{jobs:[{id:"j-20261027T115000-bbbbbb",type:null,requested_at:"2026-10-27T11:50:00Z"},failed]});
 b.seed("jobs/status/j-20261027T115000-bbbbbb.json",{state:"done",exit_code:0,finished_at:"2026-10-27T11:30:00Z"});
 b.seed(`jobs/status/${failed.id}.json`,{state:"failed",exit_code:1});
 b.seed("golfprice/index.json",{events:[{event_uid:event.uid,name:event.name,tour:event.tour}]});
 b.seed("owner/ops/observations.json",{schema:"golfops.observations.v1",receipts:[receipt()]});
 assert.equal((await call(b,"/api/ops/workflow")).status,503);
 b.seed("jobs/queue.json",{jobs:[{id:"j-20261027T115000-bbbbbb",type:"watch",requested_at:"2026-10-27T11:50:00Z"},failed]});
 const result=await (await call(b,"/api/ops/workflow")).json();
 assert.equal(result.stages.find((s)=>s.id==="weekly").state,"failed");
 b.seed("jobs/queue.json",{jobs:[failed,failed]});
 assert.equal((await call(b,"/api/ops/workflow")).status,503);
});

test("a malformed newer receipt refuses the observation document instead of exposing older success",async()=>{
 const b=new Bucket();
 b.seed("jobs/queue.json",{jobs:[{id:"j-20261027T110000-aaaaaa",type:"monday_week",requested_at:"2026-10-27T11:00:00Z"}]});
 b.seed("jobs/status/j-20261027T110000-aaaaaa.json",{state:"done",exit_code:0,finished_at:"2026-10-27T11:30:00Z"});
 b.seed("golfprice/index.json",{events:[{event_uid:event.uid,name:event.name,tour:event.tour}]});
 const older=receipt({observed_at:new Date(NOW-60_000).toISOString()});
 for(const bad of [receipt({state:"failed",outputs:null}),receipt({state:"verified",outputs:[null]}),receipt({observed_at:"bad"}),receipt({state:"unknown_future_state"})]){
  b.seed("owner/ops/observations.json",{schema:"golfops.observations.v1",receipts:[older,bad]});
  const response=await call(b,"/api/ops/workflow");assert.equal(response.status,503);assert.equal((await response.json()).ok,false);
 }
});


test("malformed or duplicate published event rows cannot silently reduce required coverage",async()=>{
 const b=new Bucket(); const valid={event_uid:event.uid,name:event.name,tour:event.tour};
 for(const events of [[valid,{event_uid:"euro_bad",tour:null}],[valid,valid],"invalid"]){
  b.seed("golfprice/index.json",{events}); assert.equal((await call(b,"/api/ops/workflow")).status,503);
 }
});
