/** Pure logic for the shot-fingerprint metric trend: a shot-weighted simple moving average over rounds, time agnostic. */
export type TrendRoundRow={event_id:string;round:number;date:string;metrics:Record<string,{total:number;n_shots:number}|undefined>};
export type TrendPoint={index:number;date:string;event_id:string;round:number;total:number;n:number;value:number;sma:number;windowRounds:number};
export const TREND_WINDOWS=[20,50,100] as const;
export const DEFAULT_TREND_WINDOW=50;
const fin=(v:unknown):v is number=>typeof v==="number"&&Number.isFinite(v);

/** Chronological order: date, then event, then round number. Input is not assumed sorted. */
export function orderRounds<T extends {date:string;event_id:string;round:number}>(rows:T[]):T[]{
  return [...rows].sort((a,b)=>a.date.localeCompare(b.date)||a.event_id.localeCompare(b.event_id)||a.round-b.round);
}
/** Keep the rounds whose event and round number are in the selected history slice (every round when `selected` is null). */
export function selectedRounds<T extends {event_id:string;round:number}>(rows:T[],selected:Set<string>|null):T[]{
  return selected?rows.filter(r=>selected.has(`${r.event_id}|${r.round}`)):rows;
}
/** Rolling shot-weighted mean over the trailing `window` rounds that recorded the metric: sum(total)/sum(n). Rounds without data are skipped, not zero-filled. */
export function movingAverage(rows:TrendRoundRow[],key:string,window:number):TrendPoint[]{
  const w=Math.max(1,Math.floor(window)),kept:Omit<TrendPoint,"sma"|"windowRounds"|"index">[]=[];
  for(const r of orderRounds(rows)){
    const m=r.metrics?.[key];
    if(m&&fin(m.total)&&fin(m.n_shots)&&m.n_shots>0)kept.push({date:r.date,event_id:r.event_id,round:r.round,total:m.total,n:m.n_shots,value:m.total/m.n_shots});
  }
  const out:TrendPoint[]=[];let sumT=0,sumN=0;
  kept.forEach((p,i)=>{
    sumT+=p.total;sumN+=p.n;
    if(i>=w){sumT-=kept[i-w].total;sumN-=kept[i-w].n;}
    out.push({...p,index:i+1,sma:sumT/sumN,windowRounds:Math.min(i+1,w)});
  });
  return out;
}
export type TrendSummary={rounds:number;current:number|null;previous:number|null;change:number|null;fullWindow:boolean;medianShotsPerRound:number|null;sparse:boolean};
/** Current average against the average `window` rounds earlier (same-length window). Needs window+1 rounds for a comparison. */
export function trendSummary(points:TrendPoint[],window:number):TrendSummary{
  const last=points.at(-1),prev=points.length>window?points[points.length-1-window]:undefined;
  const ns=points.map(p=>p.n).sort((a,b)=>a-b),median=ns.length?(ns[(ns.length-1)>>1]+ns[ns.length>>1])/2:null;
  return {rounds:points.length,current:last?.sma??null,previous:prev?.sma??null,change:last&&prev?last.sma-prev.sma:null,fullWindow:points.length>=window,medianShotsPerRound:median,sparse:median!==null&&median<3};
}
/** Whether a move is an improvement; null for ball-flight (descriptive) measures where bigger is not better. */
export function trendImproved(change:number|null,lowerIsBetter:boolean,descriptive:boolean):boolean|null{
  if(descriptive||!fin(change)||Math.abs(change)<1e-9)return null;
  return lowerIsBetter?change<0:change>0;
}
/** Human description of which rounds feed the panel. */
export function scopeLabel(count:number,filtered:boolean,descriptive:boolean):string{
  const noun=descriptive?"radar rounds":"rounds";
  return filtered?`${count} ${noun} from the rounds you selected in the filters, in date order`:`${count} ${noun} with published shot detail (last three years), in date order`;
}
