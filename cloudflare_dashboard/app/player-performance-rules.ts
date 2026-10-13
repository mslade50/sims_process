import type {HistoryEvent} from "./player-deep-rules";
import {SG_COLUMNS,divergingBackground} from "./player-event-table-rules.ts";
const finite=(v:unknown):v is number=>typeof v==="number"&&Number.isFinite(v);
export const SMA_WINDOWS=[20,50,100,200] as const;
export type SmaWindow=typeof SMA_WINDOWS[number];
export const DEFAULT_SMA:SmaWindow[]=[50];
export type PerformancePoint={date:string;time:number;round:number;event:string;eventName:string;course?:string;tour?:string;major:boolean;value:number;sma20:number|null;sma50:number|null;sma100:number|null;sma200:number|null;n:number};
export const smaOf=(p:PerformancePoint,w:SmaWindow):number|null=>p[`sma${w}` as const];
/** Shown windows in ascending order; never empty (falls back to the default 50). */
export function normalizeSmas(w:readonly number[]|undefined):SmaWindow[]{const out=SMA_WINDOWS.filter(k=>w?.includes(k));return out.length?out:[...DEFAULT_SMA];}
/** Shortest shown window minus the 50-round line; null when no window shorter than 50 is shown. */
export function recentVsLonger(p:PerformancePoint|undefined,shown:readonly SmaWindow[]):{window:SmaWindow;diff:number}|null{
  if(!p)return null;const w=normalizeSmas(shown)[0];if(w>=50)return null;
  const a=smaOf(p,w),b=p.sma50;return finite(a)&&finite(b)?{window:w,diff:a-b}:null;
}
/** Same range as the Total column of the event table; alpha-only mix, so it works in both themes. Majors are gold regardless of value. */
export const TOTAL_RANGE=SG_COLUMNS.find(c=>c.key==="sg_total")!.range;
export const MAJOR_FILL="var(--gold, #d4a017)";
export function roundBarFill(p:Pick<PerformancePoint,"value"|"major">):string{return p.major?MAJOR_FILL:divergingBackground(p.value,TOTAL_RANGE,85)??"color-mix(in srgb, var(--muted) 35%, transparent)";}
/** Equal-weight trailing N eligible rounds; no value before a full window. */
export function roundSma(values:number[],span:number):(number|null)[]{
  if(!Number.isInteger(span)||span<1)throw new Error("SMA span must be a positive integer");
  let sum=0;
  return values.map((value,i)=>{if(!finite(value))throw new Error("SMA needs finite observations");
    sum+=value;if(i>=span)sum-=values[i-span];
    return i>=span-1?sum/span:null;});
}
/** all=true (default): rounds 1-4; false: R1/R2 only. */
export function performanceSeries(events:HistoryEvent[],anchor:number|null|undefined,all=true):PerformancePoint[]{
  if(!finite(anchor))return [];
  const seen=new Set<string>();
  const rows=events.flatMap(e=>(e.round_data??[]).flatMap(r=>{const round=r.round??r.round_num;
    const time=Date.parse(r.date??"");const id=`${e.id}:${round}`;
    if(!Number.isInteger(round)||round!<1||(!all&&round!>2)||r.sg_basis!=="source_adjusted"||!finite(r.sg_total)||!Number.isFinite(time)||seen.has(id))return [];
    seen.add(id);return [{date:r.date!,time,round:round!,event:e.id,eventName:e.name||"Event name unavailable",course:e.course,tour:e.tour,major:e.major===true,value:r.sg_total-anchor}];
  })).sort((a,b)=>a.time-b.time||a.event.localeCompare(b.event)||a.round-b.round);
  const v=rows.map(r=>r.value),s20=roundSma(v,20),s50=roundSma(v,50),s100=roundSma(v,100),s200=roundSma(v,200);
  return rows.map((r,i)=>({...r,sma20:s20[i],sma50:s50[i],sma100:s100[i],sma200:s200[i],n:i+1}));
}
export function visiblePerformance(points:PerformancePoint[],asOf:string,window:"3y"|"all"){
  if(window==="all")return points;const d=new Date(asOf);if(!Number.isFinite(d.getTime()))return [];
  d.setUTCFullYear(d.getUTCFullYear()-3);return points.filter(p=>p.time>=d.getTime());
}

/** Snap to a real observed round, not an interpolated value between events. Ties use the earlier observation. */
export function nearestPerformanceIndex(points:Pick<PerformancePoint,"time">[],time:number):number{
 if(!points.length||!finite(time))return -1;
 let lo=0,hi=points.length;
 while(lo<hi){const mid=Math.floor((lo+hi)/2);if(points[mid].time<time)lo=mid+1;else hi=mid;}
 if(lo===0)return 0;if(lo===points.length)return points.length-1;
 return time-points[lo-1].time<=points[lo].time-time?lo-1:lo;
}
