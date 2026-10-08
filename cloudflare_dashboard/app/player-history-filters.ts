import type {DeepProfile, EventFilters, HistoryEvent, RoundRow, ShotMetric} from "./player-deep-rules";
import type {PlayerProfile} from "./player-profile-rules";
const finite=(v:unknown):v is number=>typeof v==="number"&&Number.isFinite(v);
export type HistoryFilters=EventFilters & {rounds:number[];contention:"all"|"near"|"outside";event:string;course:string};
export const HISTORY_DEFAULTS:HistoryFilters={season:"",tour:"",major:false,difficulty:"",strength:"",search:"",window:"recent2y",rounds:[1,2,3,4],contention:"all",event:"",course:""};
export type ContextRound=RoundRow & {course?:string;strokes_behind_leader_before?:number|null;in_contention?:boolean|null};
export type ContextEvent=HistoryEvent & {courses?:string[];opening_difficulty?:number|null;opening_strength?:number|null};
export type ShotReference={id?:string;as_of?:string;metrics?:Record<string,{mean?:number;sd?:number;n_players?:number;higher_is_better?:boolean;cohort_values?:number[]}>};
export type ObservedCategoryReference={schema_version:string;id?:string;as_of?:string;axes:Record<string,{mean?:number|null;sd?:number|null;n_players?:number;min_rounds?:number;window_days?:number}>};
export type ShotRound={event_id:string;round:number;date:string;metrics:Record<string,{total:number;n_shots:number}>};
export type RoundReference={schema_version:string;as_of:string;columns:string[];events:Array<Omit<ContextEvent,"round_data"> & {round_data:unknown[][]}>;n?:number;population?:string};
const fold=(v:string)=>v.normalize("NFD").replace(/[\u0300-\u036f]/g,"").toLowerCase().trim();
export function matchesRound(r:ContextRound,f:HistoryFilters){const n=r.round??r.round_num;if(!n||!f.rounds.includes(n)||![1,2,3,4].includes(n))return false;if(f.contention!=="all"){if(![3,4].includes(n)||!finite(r.strokes_behind_leader_before)||r.strokes_behind_leader_before<0)return false;return f.contention==="near"?r.strokes_behind_leader_before<=2:r.strokes_behind_leader_before>2;}return true;}
function opening(e:ContextEvent,key:"difficulty"|"field_strength"){const preset=key==="difficulty"?e.opening_difficulty:e.opening_strength;if(finite(preset))return preset;const a=e.round_data.filter(r=>[1,2].includes(r.round??r.round_num??0)).map(r=>r[key]).filter(finite);return a.length?a.reduce((s,v)=>s+v,0)/a.length:null;}
export function eventMatches(e:ContextEvent,f:HistoryFilters){
  const timestamp=Date.parse(e.date),asof=f.asOf?Date.parse(f.asOf):NaN;
  if(f.window==="recent2y"&&finite(asof)){const start=new Date(asof);start.setUTCHours(0,0,0,0);start.setUTCFullYear(start.getUTCFullYear()-2);if(!finite(timestamp)||timestamp<start.getTime()||timestamp>asof)return false;}
  const difficulty=opening(e,"difficulty"),strength=opening(e,"field_strength");
  return (!f.season||String(e.year)===f.season)&&(!f.tour||fold(e.tour)===fold(f.tour))&&(!f.major||e.major===true)&&(!f.event||e.id===f.event)&&(!f.course||e.course===f.course||e.courses?.includes(f.course))&&(!f.search||fold(`${e.name} ${e.course??""}`).includes(fold(f.search)))&&(!f.difficulty||(finite(difficulty)&&(f.difficulty==="hard"?difficulty>1:f.difficulty==="easy"?difficulty< -1:Math.abs(difficulty)<=1)))&&(!f.strength||(finite(strength)&&(f.strength==="strong"?strength>=0:strength<0)));
}
function mean(v:unknown[]){const a=v.filter(finite);return a.length?a.reduce((s,x)=>s+x,0)/a.length:null;}
export function sliceHistory(events:ContextEvent[],f:HistoryFilters):ContextEvent[]{return events.filter(e=>eventMatches(e,f)).flatMap(e=>{
  const rounds=e.round_data.filter(r=>matchesRound(r as ContextRound,f)&&(!f.course||(r as ContextRound).course===f.course));if(!rounds.length)return [];
  const adjusted=rounds.filter(r=>r.sg_basis==="source_adjusted");
  return [{...e,round_data:rounds,rounds:rounds.length,sg_basis:adjusted.length?"source_adjusted":"unknown",sg_total:mean(adjusted.map(r=>r.sg_total)),sg_total_field:mean(rounds.map(r=>r.sg_total_field)),sg_ott:mean(rounds.map(r=>r.sg_ott)),sg_app:mean(rounds.map(r=>r.sg_app)),sg_arg:mean(rounds.map(r=>r.sg_arg)),sg_putt:mean(rounds.map(r=>r.sg_putt)),score_to_par:mean(rounds.map(r=>finite(r.score)&&finite(r.par)?r.score-r.par:null))}];
});}
export function adjustedValues(events:HistoryEvent[],anchor:number|null|undefined){if(!finite(anchor))return [];return events.flatMap(e=>e.round_data.filter(r=>r.sg_basis==="source_adjusted"&&finite(r.sg_total)).map(r=>r.sg_total!-anchor));}
export function decodeReference(raw:unknown,asOf:string,ids?:Set<string>):ContextEvent[]{const d=raw as RoundReference|null;if(d?.schema_version!=="player_round_reference.v1"||(!finite(Date.parse(d.as_of))||!finite(Date.parse(asOf))||Date.parse(d.as_of)>Date.parse(asOf)||new Date(d.as_of).toISOString().slice(0,10)!==new Date(asOf).toISOString().slice(0,10))||!Array.isArray(d.events)||!Array.isArray(d.columns))return [];return d.events.filter(e=>!ids||ids.has(e.id)).map(e=>({...e,round_data:e.round_data.map(row=>Object.fromEntries(d.columns.map((c,i)=>[c,row[i]])) as ContextRound)}));}
export function matchedPgaEvents(reference:ContextEvent[],selected:ContextEvent[],f:HistoryFilters){const ids=new Set(selected.filter(e=>e.tour.toLowerCase()==="pga").map(e=>e.id));return sliceHistory(reference.filter(e=>ids.has(e.id)),{...f,difficulty:"",strength:"",search:"",event:"",course:f.course});}
export function summaryWithMean(values:number[]){const a=values.filter(finite).sort((a,b)=>a-b);const q=(p:number)=>{if(!a.length)return null;const x=(a.length-1)*p,i=Math.floor(x);return a[i]+(a[Math.min(i+1,a.length-1)]-a[i])*(x-i);};return {n:a.length,mean:mean(a),median:q(.5),p10:q(.1),p90:q(.9)};}
export function filteredShotMetrics(deep:DeepProfile|null,events:ContextEvent[],reference:ShotReference):{metrics:ShotMetric[];n_rounds:number;available:boolean}{
  const shots=deep?.shots as (NonNullable<DeepProfile["shots"]>&{round_data?:ShotRound[]})|undefined;
  if(!Array.isArray(shots?.round_data))return {metrics:[],n_rounds:0,available:false};
  const selected=new Set(events.flatMap(e=>e.round_data.map(r=>`${e.id}|${r.round??r.round_num}`))),rows=shots.round_data.filter(r=>selected.has(`${r.event_id}|${r.round}`));
  const metrics=shots.metrics.map(m=>{const ms=rows.map(r=>({...r,v:r.metrics[m.key]})).filter(r=>r.v&&finite(r.v.total)&&finite(r.v.n_shots)&&r.v.n_shots>0),n=ms.reduce((s,r)=>s+r.v.n_shots,0),value=n?ms.reduce((s,r)=>s+r.v.total,0)/n:null,ref=reference.metrics?.[m.key],nr=ms.length,status=n>=20&&nr>=3?"available":n?"insufficient_data":"unavailable";
    const z=finite(value)&&finite(ref?.mean)&&finite(ref?.sd)&&ref.sd>0?(value-ref.mean)/ref.sd*(ref.higher_is_better===false?-1:1):null;
    const cohort=ref?.cohort_values?.filter(finite)??[];let pct=finite(value)&&cohort.length?cohort.reduce((s,v)=>s+(v<value?1:v===value ? .5:0),0)/cohort.length:null;if(pct!==null&&ref?.higher_is_better===false)pct=1-pct;
    return {...m,value,n_shots:n,n_rounds:nr,last_date:ms.map(r=>r.date).sort().at(-1),status,reference_z:z,reference_percentile:pct,reference_n_players:ref?.n_players,higher_is_better:ref?.higher_is_better,reference_id:reference.id,reference_as_of:reference.as_of};
  });return {metrics,n_rounds:rows.length,available:true};
}
export function filteredSkillProfile(profile:PlayerProfile,events:ContextEvent[],observed:ObservedCategoryReference|undefined):PlayerProfile{return {...profile,radar:profile.radar.map(m=>{const rounds=events.flatMap(e=>e.round_data),v=rounds.map(r=>r[m.key]).filter(finite),value=mean(v),ref=observed?.schema_version==="observed_category_reference.v1"?observed.axes[m.key]:undefined,z=finite(value)&&finite(ref?.mean)&&finite(ref?.sd)&&ref.sd>0?(value-ref.mean)/ref.sd*1:null;return {...m,value,estimate:value,z:v.length>=3?z:null,n:v.length,n_eff:v.length,shrinkage:null,status:v.length>=3?"available":"insufficient_data",last_observation:rounds.filter(r=>finite(r[m.key])).map(r=>r.date??"").sort().at(-1)??null};})};}
