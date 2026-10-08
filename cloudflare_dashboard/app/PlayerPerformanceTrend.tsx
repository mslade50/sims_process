"use client";
import {useMemo,useState} from "react";
import {finite,type DeepProfile} from "./player-deep-rules";
import {performanceSeries,visiblePerformance} from "./player-performance-rules";
const fmt=(v:number|null|undefined)=>finite(v)?`${v>0?"+":""}${v.toFixed(2)}`:"—";
const day=(v:string)=>new Date(v).toLocaleDateString("en-US",{month:"short",year:"numeric",timeZone:"UTC"});
export function PerformanceTrend({deep,anchor}:{deep:DeepProfile;anchor:number|null|undefined}){
  const [all,setAll]=useState(false),[raw,setRaw]=useState(false),[window,setWindow]=useState<"3y"|"all">("3y");
  const history=deep.history;
  const series=useMemo(()=>performanceSeries(history?.events??[],anchor,all),[history,anchor,all]);
  const points=visiblePerformance(series,deep.as_of,window),latest=series.at(-1);
  const vals=points.flatMap(p=>[p.ema20,p.ema50,...(raw?[p.value]:[])]).filter(finite);
  const lo=Math.floor((Math.min(0,...vals)-.2)*2)/2,hi=Math.ceil((Math.max(0,...vals)+.2)*2)/2;
  const start=points[0]?.time??0,end=points.at(-1)?.time??start;
  const x=(t:number)=>60+(t-start)/Math.max(1,end-start)*672,y=(v:number)=>225-(v-lo)/Math.max(.1,hi-lo)*180;
  const line=(key:"ema20"|"ema50")=>{let last:number|null=null;return points.flatMap(p=>{if(!finite(p[key]))return [];const move=last===null||p.time-last>180*86400000;last=p.time;return [`${move?"M":"L"}${x(p.time).toFixed(2)},${y(p[key]!).toFixed(2)}`];}).join(" ");};
  const ticks=Array.from({length:5},(_,i)=>lo+(hi-lo)*i/4);
  const years=Array.from({length:Math.max(0,new Date(end).getUTCFullYear()-new Date(start).getUTCFullYear()+1)},(_,i)=>Date.UTC(new Date(start).getUTCFullYear()+i,0,1)).filter(t=>t>start&&t<end);
  return <section className="pd-performance"><div className="pd-heading"><div><span className="eyebrow">Performance over time</span><h3>Is their golf getting better?</h3></div><label className="pd-select">Chart range<select aria-label="Performance chart range" value={window} onChange={e=>setWindow(e.target.value as "3y"|"all")}><option value="3y">Past 3 years</option><option value="all">All available</option></select></label></div>
    <div className="pd-trend-cards"><article><span>20-round EMA · recent form</span><strong>{fmt(latest?.ema20)}</strong></article><article><span>50-round EMA · longer trend</span><strong>{fmt(latest?.ema50)}</strong></article><article><span>Recent − longer</span><strong>{finite(latest?.ema20)&&finite(latest?.ema50)?fmt(latest.ema20-latest.ema50):"—"}</strong><small>Descriptive change, not significance</small></article></div>
    <div className="pd-trend-options"><label className="pd-check"><input type="checkbox" checked={all} onChange={e=>setAll(e.target.checked)}/>Include played rounds 3 and 4</label><label className="pd-check"><input type="checkbox" checked={raw} onChange={e=>setRaw(e.target.checked)}/>Show individual rounds</label></div>
    {!vals.length?<p className="pd-empty">At least 20 eligible adjusted rounds are needed for the first trend line; 50 for the longer line. Missing or field-relative results do not become zero.</p>:<svg className="pd-trend-chart" viewBox="0 0 780 280" role="img" aria-label="20-round and 50-round exponential moving averages of observed adjusted SG, centered on the fixed PGA reference. Higher is better; this is performance history, not a forecast.">
      {ticks.map(v=><g key={v}><line x1="60" x2="732" y1={y(v)} y2={y(v)} stroke="var(--line)"/><text x="49" y={y(v)+4} textAnchor="end" fontSize="11" fill="var(--muted)">{fmt(v)}</text></g>)}
      {years.map(t=><g key={t}><line x1={x(t)} x2={x(t)} y1="45" y2="225" stroke="var(--line)" strokeDasharray="2 5"/><text x={x(t)} y="248" textAnchor="middle" fontSize="11" fill="var(--muted)">{new Date(t).getUTCFullYear()}</text></g>)}
      <line x1="60" x2="732" y1={y(0)} y2={y(0)} stroke="var(--muted)" strokeDasharray="5 5"/>
      {raw&&points.map((p,i)=><circle key={i} cx={x(p.time)} cy={y(p.value)} r="2.3" fill="var(--muted)" fillOpacity=".28"><title>{p.date} R{p.round}: {fmt(p.value)} adjusted SG vs fixed PGA reference</title></circle>)}
      <path d={line("ema50")} fill="none" stroke="var(--chart-2)" strokeWidth="3" strokeLinecap="round"/>
      <path d={line("ema20")} fill="none" stroke="var(--chart-1)" strokeWidth="2.5" strokeLinecap="round"/>
      {(["ema20","ema50"] as const).map((key,i)=>{const p=points.filter(p=>finite(p[key])).at(-1);return p?<circle key={key} cx={x(p.time)} cy={y(p[key]!)} r="4" fill={i===0?"var(--chart-1)":"var(--chart-2)"}><title>{p.date}: {key==="ema20"?"20":"50"}-round EMA {fmt(p[key])}</title></circle>:null;})}
      <text x="60" y="267" fontSize="11" fill="var(--muted)">{points.length?day(points[0].date):""}</text><text x="732" y="267" textAnchor="end" fontSize="11" fill="var(--muted)">{points.length?day(points.at(-1)!.date):""}</text>
    </svg>}
    <div className="pd-legend"><span><i style={{background:"var(--chart-1)"}}/>20-round EMA</span><span><i style={{background:"var(--chart-2)"}}/>50-round EMA</span><span>Dashed zero = fixed 2025 PGA-start reference</span></div>
    <p className="pd-muted">{points.length} adjusted {all?"played":"R1/R2"} rounds in view · {series.length} used for initialization and history · latest {latest?.date??"unavailable"}. Each observation advances the average once; days without play do not change it. Long gaps break the drawn line. All available earlier rounds initialize the averages before the selected chart range.</p>
    <details className="pd-method"><summary>What these trends mean</summary><p>Each round is provider-adjusted total SG minus the same fixed PGA-start anchor used above ({fmt(anchor)} provider SG). A positive value means performance above that reference. The 20-round line responds faster; the 50-round line shows the broader trend. Their difference is descriptive, not a test of statistical significance or predicted future improvement.</p><p>EMA spans are 20 and 50 eligible rounds, not days or half-lives: alpha = 2 / (span + 1), seeded with the mean of the first span observations. Warm-up values are withheld. Revised history, schedule, individual course fit and weather exposure can still affect the lines. They are observed-performance summaries, not a course-neutral latent-skill estimate or simulation input. R1/R2 is the default to reduce weekend survivor selection; including later rounds changes the population.</p><p>The trend uses complete available adjusted history independently of the event-table filters below, so choosing a course or major does not silently redefine long-term form.</p></details>
  </section>;
}
