/** Pure logic for the per-shot fingerprint trend: rolling averages over the last N SHOTS, dated by the round each shot belongs to.
 *  Reads the lazily loaded `shots.series_key` object (schema player_shot_series.v1); the round-level view stays the fallback. */
export type SeriesMetric={n:number;rc:number[];v?:number[];b?:string};
export type ShotSeries={schema_version:string;rounds:Array<[string,string,number]>;metrics:Record<string,SeriesMetric|undefined>};
export type ShotPoint={index:number;date:string;time:number;event_id:string;round:number;value:number;sma:number|null};
const fin=(v:unknown):v is number=>typeof v==="number"&&Number.isFinite(v);

/** Rare events (penalties, bunker shots) need long windows to mean anything. */
const RARE=new Set(["tee_penalty","app_penalty","arg_sand","tee_relief_drop"]);

/** Validated series, or null when the object is missing or malformed (callers then fall back to the round-level view). */
export function parseSeries(raw:unknown):ShotSeries|null{
  const s=raw as Partial<ShotSeries>|null;
  if(!s||typeof s!=="object"||!Array.isArray(s.rounds)||!s.metrics||typeof s.metrics!=="object")return null;
  return s as ShotSeries;
}
export function seriesShotCount(series:ShotSeries|null|undefined,key:string,selected:Set<string>|null=null):number{
  const m=series?.metrics?.[key];if(!series||!m)return 0;
  if(!selected)return m.n;
  let n=0;for(let i=0;i+1<m.rc.length;i+=2){const r=series.rounds[m.rc[i]];if(r&&selected.has(`${r[1]}|${r[2]}`))n+=m.rc[i+1];}
  return n;
}
/** Every shot of the metric in play order, restricted to the selected `event_id|round` keys (all shots when `selected` is null). */
export function seriesValues(series:ShotSeries,key:string,selected:Set<string>|null=null):Array<Omit<ShotPoint,"sma"|"index">>{
  const m=series.metrics[key],out:Array<Omit<ShotPoint,"sma"|"index">>=[];
  if(!m||!Array.isArray(m.rc))return out;
  let at=0;
  for(let i=0;i+1<m.rc.length;i+=2){
    const r=series.rounds[m.rc[i]],count=m.rc[i+1],keep=!!r&&(!selected||selected.has(`${r[1]}|${r[2]}`));
    if(keep){
      const time=Date.parse(r[0]);
      for(let j=0;j<count;j++){
        const value=m.b!==undefined?(m.b.charCodeAt(at+j)===49?1:0):m.v?.[at+j];
        if(fin(value)&&Number.isFinite(time))out.push({date:r[0],time,event_id:r[1],round:r[2],value});
      }
    }
    at+=count;
  }
  return out;
}
/** Rolling mean of the trailing `window` shots; null until a full window is behind it (never a partial-window value). */
export function shotMovingAverage(shots:Array<Omit<ShotPoint,"sma"|"index">>,window:number):ShotPoint[]{
  const w=Math.max(1,Math.floor(window)),out:ShotPoint[]=[];let sum=0;
  shots.forEach((p,i)=>{sum+=p.value;if(i>=w)sum-=shots[i-w].value;out.push({...p,index:i+1,sma:i+1>=w?sum/w:null});});
  return out;
}
/** Selectable shot windows for the metric. Chosen October 15, 2026 (experiments/ma_window_20261015 in sim_prep): at 250-500 shots the line barely moves,
 *  and 50-200 still shows real movement (the luck band says how much of it could be chance). Rare events keep long windows. */
export function shotWindows(key:string):number[]{return RARE.has(key)?[250,500,1000]:[50,100,200];}
/** Longest window that still leaves at least two non-overlapping windows of shots, else the longest that fits, else the shortest. */
export function defaultShotWindow(count:number,key:string):number{
  const ws=shotWindows(key);
  return [...ws].reverse().find(w=>count>=2*w&&w<=ws[1])??[...ws].reverse().find(w=>count>=w)??ws[0];
}
export type ShotSummary={shots:number;current:number|null;previous:number|null;change:number|null};
/** Latest window against the window `window` shots earlier (needs 2×window shots). */
export function shotTrendSummary(points:ShotPoint[],window:number):ShotSummary{
  const last=points.at(-1),prev=points.length>window?points[points.length-1-window]:undefined;
  return {shots:points.length,current:last?.sma??null,previous:prev?.sma??null,change:fin(last?.sma)&&fin(prev?.sma)?last.sma-prev.sma:null};
}
/** Index of the point closest in time to `t` (points are in date order). */
export function nearestByTime(points:Array<{time:number}>,t:number):number{
  if(!points.length)return -1;
  let lo=0,hi=points.length-1;
  while(lo<hi){const mid=(lo+hi)>>1;if(points[mid].time<t)lo=mid+1;else hi=mid;}
  return lo>0&&Math.abs(points[lo-1].time-t)<=Math.abs(points[lo].time-t)?lo-1:lo;
}
/** Evenly spaced positions (0-based) along an index axis of `count` points, always including the first and last. */
export function indexTicks(count:number,max=5):number[]{
  if(count<=0)return [];if(count===1)return [0];
  const k=Math.max(2,Math.min(max,count));
  return [...new Set(Array.from({length:k},(_,i)=>Math.round((count-1)*i/(k-1))))];
}
/** Short date label for an index-axis marker; month + year when the chart spans more than ~18 months, else month, day and short year. */
export function tickDate(t:number,first:number,last:number):string{
  const long=Math.abs(last-first)>548*864e5;
  return new Date(t).toLocaleDateString("en-US",long?{month:"short",year:"numeric",timeZone:"UTC"}:{month:"short",day:"numeric",year:"2-digit",timeZone:"UTC"});
}
/** Mean and the band a w-shot average would wander in from luck alone if the player's true level never moved (±2 standard errors around the overall mean).
 *  Assumes shots are independent; null with fewer than two shots. */
export function luckBand(shots:Array<{value:number}>,window:number):{mean:number;lo:number;hi:number}|null{
  const n=shots.length;if(n<2||window<1)return null;
  let s=0;for(const p of shots)s+=p.value;const mean=s/n;
  let v=0;for(const p of shots)v+=(p.value-mean)**2;
  const half=2*Math.sqrt(v/(n-1)/window);
  return Number.isFinite(half)?{mean,lo:mean-half,hi:mean+half}:null;
}
