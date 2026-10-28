/** Loopback-only browser verification harness. Generated test identity + in-memory R2 fixtures; external fetches are refused.
 * Build first, then: node scripts/ops-preview.mjs [port]. Never deploy this script or use production credentials/data. */
import http from "node:http";
import { readFile } from "node:fs/promises";
import { resolve, sep, extname } from "node:path";
import { createSign, generateKeyPairSync } from "node:crypto";
const port=Number(process.argv[2]??5317);
if(!Number.isInteger(port)||port<1024||port>65535)throw new Error("Choose an unprivileged loopback port");
const TEAM="ops-preview.cloudflareaccess.com",AUD="fixture-aud",OWNER="fixture-owner@example.com";
const {publicKey,privateKey}=generateKeyPairSync("rsa",{modulusLength:2048});
const b64=(v)=>Buffer.from(JSON.stringify(v)).toString("base64url");
function headers(email){const h=b64({alg:"RS256",kid:"preview-key"}),p=b64({email,aud:[AUD],iss:`https://${TEAM}`,exp:Math.floor(Date.now()/1000)+3600,sub:"fixture-user"});const s=createSign("RSA-SHA256").update(`${h}.${p}`).sign(privateKey).toString("base64url");return {"cf-access-authenticated-user-email":email,"cf-access-jwt-assertion":`${h}.${p}.${s}`};}
globalThis.fetch=async(input)=>{const url=typeof input==="string"?input:input.url;if(url===`https://${TEAM}/cdn-cgi/access/certs`)return Response.json({keys:[{...publicKey.export({format:"jwk"}),kid:"preview-key"}]});throw new Error("Fixture harness refuses external network fetches");};
const worker=(await import("../dist/server/index.js")).default;
let scenario="mixed",role="owner",counter=0;
const stored=new Map();
const events=[{event_uid:"pga_540_2026",name:"Example PGA week",tour:"pga",date_start:"2026-10-29",course:"Example course",runs:[{as_of:new Date().toISOString()}]},{event_uid:"euro_2026141_2026",name:"Example Euro week",tour:"euro",date_start:"2026-10-29",course:"Example venue",runs:[{as_of:new Date().toISOString()}]}];
function fixture(key){
 const now=Date.now(),at=new Date(now-30_000).toISOString(),requested=new Date(now-600_000).toISOString(),finished=new Date(now-300_000).toISOString();
 if(key==="golfprice/index.json")return {events};
 if(key==="golfprice/player_profiles/search-index.json")return {players:[]};
 if(key==="data/manifest.json")return {event:"Local verification fixtures",event_id:540,course:"No live data",generated_at:at};
 if(key==="jobs/roles.json")return scenario==="roles_mismatch"?{primary:"mckinley_home",backup:"desktop-2ki41v6"}:scenario==="roles_invalid"?{primary:"unknown",backup:"desktop-2ki41v6"}:null;
 if(key.startsWith("jobs/heartbeat/")){const machine=key.includes("mckinley_home")?"mckinley_home":"desktop-2ki41v6";return {machine,at:scenario==="stale"?new Date(now-3_600_000).toISOString():at,state:"idle",role:machine==="desktop-2ki41v6"?"primary":"backup",code:machine==="desktop-2ki41v6"?"new-fixture":"old-fixture",commit:"fixture-only"};}
 if(key==="jobs/queue.json")return {jobs:[{id:"j-20261027T110000-aaaaaa",type:"monday_week",requested_at:requested,requested_by:"fixture-owner"},{id:"j-20261027T110100-bbbbbb",type:"watch",requested_at:requested,requested_by:"cron"},{id:"j-20261027T110200-cccccc",type:"tuesday",requested_at:requested,requested_by:"fixture-owner"},{id:"j-20261027T110300-dddddd",type:"publish",requested_at:requested,requested_by:"fixture-owner"}]};
 if(key.startsWith("jobs/status/"))return key.includes("bbbbbb")?{state:"running",started_at:requested,machine:"desktop-2ki41v6"}:{state:"done",exit_code:0,finished_at:finished,machine:"desktop-2ki41v6"};
 if(key==="owner/ops/observations.json")return {schema:"golfops.observations.v1",receipts:events.flatMap((e)=>["weekly","refresh","boards"].map((stage)=>{const occurrence_id=stage==="weekly"?"j-20261027T110000-aaaaaa":stage==="refresh"?"j-20261027T110200-cccccc":"j-20261027T110300-dddddd";const names=stage==="boards"?["board_publication","report_publication"]:["pricing_outputs","edge_outputs","report_outputs","dashboard_publication"];return {stage,event_uid:e.event_uid,tour:e.tour,occurrence_id,observed_at:at,expires_at:scenario==="stale"||stage==="boards"?new Date(now-1000).toISOString():new Date(now+600_000).toISOString(),state:stage==="refresh"&&e.tour==="pga"?"failed":"verified",explanation:stage==="refresh"?"Fixture: latest refresh failed; a retry needs review.":"Fixture output check",actor:"local-fixture",outputs:names.map((name)=>({name,ok:true,event_uid:e.event_uid,occurrence_id,checked_at:at,proof:"r2://golf-model-dashboard-data/golfprice/fixture/manifest.json"}))};}))};
 return null;
}
const bucket={
 async get(key){if(scenario==="offline")throw new Error("Fixture storage outage");const saved=stored.get(key);const value=saved?JSON.parse(saved.text):fixture(key);if(value===null)return null;const text=JSON.stringify(value);return {etag:saved?.etag??"fixture",httpEtag:`"${saved?.etag??"fixture"}"`,size:Buffer.byteLength(text),text:async()=>text,body:new Response(text).body,writeHttpMetadata(h){h.set("content-type","application/json");}};},
 async head(){return null;},async list({prefix}={}){return {objects:prefix==="jobs/heartbeat/"?[{key:prefix+"mckinley_home.json"},{key:prefix+"desktop-2ki41v6.json"}]:[],truncated:false};},
 async put(key,text,options){const old=stored.get(key);if(options?.onlyIf?.etagMatches&&old?.etag!==options.onlyIf.etagMatches)return null;if(options?.onlyIf?.etagDoesNotMatch==="*"&&old)return null;const etag=`p${++counter}`;stored.set(key,{text,etag});return {etag};}
};
const assetRoot=resolve("dist/client");
const types={".js":"text/javascript",".css":"text/css",".json":"application/json",".svg":"image/svg+xml",".png":"image/png",".woff2":"font/woff2",".ico":"image/x-icon"};
const assets={async fetch(request){const pathname=decodeURIComponent(new URL(request.url).pathname),path=resolve(assetRoot,`.${pathname}`);if(!path.startsWith(assetRoot+sep))return new Response("not found",{status:404});try{return new Response(await readFile(path),{headers:{"content-type":types[extname(path)]??"application/octet-stream"}});}catch{return new Response("not found",{status:404});}}};
const server=http.createServer(async(req,res)=>{
 try{
  const host=req.headers.host; if(![`127.0.0.1:${port}`,`localhost:${port}`].includes(host)){res.writeHead(400);res.end("Loopback host required");return;}
  const url=new URL(req.url,`http://${host}`);
  if(url.pathname==="/__fixture"){
   if(url.searchParams.get("reset")==="1"){stored.clear();counter=0;}
   const next=url.searchParams.get("scenario");if(next&&["mixed","stale","offline","roles_mismatch","roles_invalid"].includes(next))scenario=next;
   const who=url.searchParams.get("role");if(who&&["owner","colleague"].includes(who))role=who;
   res.writeHead(200,{"content-type":"application/json"});res.end(JSON.stringify({fixture:true,scenario,role,notice:"IN-MEMORY LOCAL VERIFICATION ONLY"}));return;
  }
  const chunks=[];for await(const chunk of req)chunks.push(chunk);const body=Buffer.concat(chunks);
  const request=new Request(url,{method:req.method,headers:{...req.headers,...headers(role==="owner"?OWNER:"fixture-colleague@example.com")},...(["GET","HEAD"].includes(req.method)?{}:{body})});
  const response=url.pathname.startsWith("/_next/")||url.pathname==="/og.png" ? await assets.fetch(request) : await worker.fetch(request,{DASHBOARD_DATA:bucket,ASSETS:assets,ACCESS_TEAM_DOMAIN:TEAM,ACCESS_AUD:AUD,OWNER_EMAILS:OWNER},{waitUntil(){},passThroughOnException(){}});
  res.writeHead(response.status,Object.fromEntries(response.headers));res.end(Buffer.from(await response.arrayBuffer()));
 }catch(error){res.writeHead(500,{"content-type":"text/plain"});res.end("Local fixture error: "+String(error));}
});
server.listen(port,"127.0.0.1",()=>console.log(`LOCAL FIXTURES ONLY: http://127.0.0.1:${port}/ops (no real credentials, R2, model runs or external fetches)`));
process.on("SIGTERM",()=>server.close());process.on("SIGINT",()=>server.close());
