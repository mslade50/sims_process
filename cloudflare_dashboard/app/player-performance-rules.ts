import type {HistoryEvent} from "./player-deep-rules";
const finite=(v:unknown):v is number=>typeof v==="number"&&Number.isFinite(v);
export type PerformancePoint={date:string;time:number;round:number;event:string;value:number;ema20:number|null;ema50:number|null;n:number};
/** Round-span EMA: seed with the first N observations, then alpha=2/(N+1). */
export function roundEma(values:number[],span:number):(number|null)[]{
  if(!Number.isInteger(span)||span<1)throw new Error("EMA span must be a positive integer");
  let sum=0,current:number|null=null;
  return values.map((value,i)=>{if(!finite(value))throw new Error("EMA needs finite observations");
    if(i<span){sum+=value;if(i===span-1)current=sum/span;}else current=2/(span+1)*value+(1-2/(span+1))*current!;
    return current;});
}
export function performanceSeries(events:HistoryEvent[],anchor:number|null|undefined,all=false):PerformancePoint[]{
  if(!finite(anchor))return [];
  const seen=new Set<string>();
  const rows=events.flatMap(e=>(e.round_data??[]).flatMap(r=>{const round=r.round??r.round_num;
    const time=Date.parse(r.date??"");const id=`${e.id}:${round}`;
    if(!Number.isInteger(round)||round!<1||(!all&&round!>2)||r.sg_basis!=="source_adjusted"||!finite(r.sg_total)||!Number.isFinite(time)||seen.has(id))return [];
    seen.add(id);return [{date:r.date!,time,round:round!,event:e.id,value:r.sg_total-anchor}];
  })).sort((a,b)=>a.time-b.time||a.event.localeCompare(b.event)||a.round-b.round);
  const fast=roundEma(rows.map(r=>r.value),20),slow=roundEma(rows.map(r=>r.value),50);
  return rows.map((r,i)=>({...r,ema20:fast[i],ema50:slow[i],n:i+1}));
}
export function visiblePerformance(points:PerformancePoint[],asOf:string,window:"3y"|"all"){
  if(window==="all")return points;const d=new Date(asOf);if(!Number.isFinite(d.getTime()))return [];
  d.setUTCFullYear(d.getUTCFullYear()-3);return points.filter(p=>p.time>=d.getTime());
}
