/** Pure logic for the shot-fingerprint metric trend: a shot-weighted simple moving average over rounds, time agnostic. */
export type TrendRoundRow={event_id:string;round:number;date:string;metrics:Record<string,{total:number;n_shots:number}|undefined>};
export type TrendPoint={index:number;date:string;event_id:string;round:number;total:number;n:number;value:number;sma:number|null;windowRounds:number};
export const TREND_WINDOWS=[20,50,100] as const;
export const DEFAULT_TREND_WINDOW=50;
import {windowWords} from "./player-deep-rules.ts";
const fin=(v:unknown):v is number=>typeof v==="number"&&Number.isFinite(v);

/** Chronological order: date, then event, then round number. Input is not assumed sorted. */
export function orderRounds<T extends {date:string;event_id:string;round:number}>(rows:T[]):T[]{
  return [...rows].sort((a,b)=>a.date.localeCompare(b.date)||a.event_id.localeCompare(b.event_id)||a.round-b.round);
}
/** Keep the rounds whose event and round number are in the selected history slice (every round when `selected` is null). */
export function selectedRounds<T extends {event_id:string;round:number}>(rows:T[],selected:Set<string>|null):T[]{
  return selected?rows.filter(r=>selected.has(`${r.event_id}|${r.round}`)):rows;
}
/** Rounds that recorded the metric, oldest first, with their shot counts (rounds without data are skipped, not zero-filled). */
function keptRounds(rows:TrendRoundRow[],key:string){
  const kept:Omit<TrendPoint,"sma"|"windowRounds"|"index">[]=[];
  for(const r of orderRounds(rows)){
    const m=r.metrics?.[key];
    if(m&&fin(m.total)&&fin(m.n_shots)&&m.n_shots>0)kept.push({date:r.date,event_id:r.event_id,round:r.round,total:m.total,n:m.n_shots,value:m.total/m.n_shots});
  }
  return kept;
}
/** How many rounds recorded the metric (the most a moving average could ever cover). */
export function trendRoundCount(rows:TrendRoundRow[],key:string):number{return keptRounds(rows,key).length;}
/** Start on the longest window that has enough rounds, so a player with 30 rounds does not open on an empty 50-round chart. */
export function defaultTrendWindow(count:number):number{
  return [...TREND_WINDOWS].reverse().find(w=>w<=DEFAULT_TREND_WINDOW&&count>=w)??DEFAULT_TREND_WINDOW;
}
/** Rolling shot-weighted mean over the trailing `window` rounds that recorded the metric: sum(total)/sum(n).
 *  A round's average exists only once a full window of rounds is behind it (sma is null before that, never a partial-window value). */
export function movingAverage(rows:TrendRoundRow[],key:string,window:number):TrendPoint[]{
  const w=Math.max(1,Math.floor(window)),kept=keptRounds(rows,key);
  const out:TrendPoint[]=[];let sumT=0,sumN=0;
  kept.forEach((p,i)=>{
    sumT+=p.total;sumN+=p.n;
    if(i>=w){sumT-=kept[i-w].total;sumN-=kept[i-w].n;}
    out.push({...p,index:i+1,sma:i+1>=w?sumT/sumN:null,windowRounds:Math.min(i+1,w)});
  });
  return out;
}
/** Only the points that have a full-window average. */
export function fullAveragePoints(points:TrendPoint[]):Array<TrendPoint&{sma:number}>{return points.filter((p):p is TrendPoint&{sma:number}=>fin(p.sma));}
export type TrendSummary={rounds:number;current:number|null;previous:number|null;change:number|null;fullWindow:boolean;medianShotsPerRound:number|null;sparse:boolean};
/** Current average against the average `window` rounds earlier (same-length window). Needs 2×window rounds (two full windows) for a comparison. */
export function trendSummary(all:TrendPoint[],window:number):TrendSummary{
  const points=fullAveragePoints(all),last=points.at(-1),prev=points.length>window?points[points.length-1-window]:undefined;
  const ns=all.map(p=>p.n).sort((a,b)=>a-b),median=ns.length?(ns[(ns.length-1)>>1]+ns[ns.length>>1])/2:null;
  return {rounds:all.length,current:last?.sma??null,previous:prev?.sma??null,change:last&&prev?last.sma-prev.sma:null,fullWindow:points.length>0,medianShotsPerRound:median,sparse:median!==null&&median<3};
}
/** Whether a move is an improvement; null for ball-flight (descriptive) measures where bigger is not better. */
export function trendImproved(change:number|null,lowerIsBetter:boolean,descriptive:boolean):boolean|null{
  if(descriptive||!fin(change)||Math.abs(change)<1e-9)return null;
  return lowerIsBetter?change<0:change>0;
}
/** Human description of which rounds feed the panel. */
export function scopeLabel(count:number,filtered:boolean,descriptive:boolean,windowDays?:number):string{
  const noun=descriptive?"radar rounds":"rounds";
  return filtered?`${count} ${noun} from the rounds you selected in the filters, in date order`:`${count} ${noun} with published shot detail (last ${windowWords(windowDays)}), in date order`;
}
