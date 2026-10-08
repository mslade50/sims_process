"use client";
import {useMemo,useState,type KeyboardEvent,type PointerEvent} from "react";
import {finite,type DeepProfile} from "./player-deep-rules";
import {nearestPerformanceIndex,performanceSeries,visiblePerformance} from "./player-performance-rules";
const fmt=(v:number|null|undefined)=>finite(v)?`${v>0?"+":""}${v.toFixed(2)}`:"—";
const day=(v:string)=>new Date(v).toLocaleDateString("en-US",{month:"short",year:"numeric",timeZone:"UTC"});
export function PerformanceTrend({deep,anchor}:{deep:DeepProfile;anchor:number|null|undefined}){
  const [all,setAll]=useState(false),[raw,setRaw]=useState(false),[window,setWindow]=useState<"3y"|"all">("3y");
  const [inspected,setInspected]=useState<string|null>(null);
  const history=deep.history;
  const series=useMemo(()=>performanceSeries(history?.events??[],anchor,all),[history,anchor,all]);
  const points=visiblePerformance(series,deep.as_of,window),latest=series.at(-1);
  const plotted=points.filter(p=>finite(p.sma20)||finite(p.sma50)),activeIndex=plotted.findIndex(p=>`${p.event}:${p.round}`===inspected),active=plotted[activeIndex],selected=active??plotted.at(-1);
  const select=(index:number)=>{const p=plotted[index];setInspected(p?`${p.event}:${p.round}`:null);};
  const vals=points.flatMap(p=>[p.sma20,p.sma50,...(raw?[p.value]:[])]).filter(finite);
  const lo=Math.floor((Math.min(0,...vals)-.2)*2)/2,hi=Math.ceil((Math.max(0,...vals)+.2)*2)/2;
  const start=points[0]?.time??0,end=points.at(-1)?.time??start;
  const x=(t:number)=>60+(t-start)/Math.max(1,end-start)*672,y=(v:number)=>225-(v-lo)/Math.max(.1,hi-lo)*180;
  const line=(key:"sma20"|"sma50")=>{let last:number|null=null;return points.flatMap(p=>{if(!finite(p[key]))return [];const move=last===null||p.time-last>180*86400000;last=p.time;return [`${move?"M":"L"}${x(p.time).toFixed(2)},${y(p[key]!).toFixed(2)}`];}).join(" ");};
  const inspectPointer=(event:PointerEvent<SVGSVGElement>)=>{
    const box=event.currentTarget.getBoundingClientRect(),px=(event.clientX-box.left)/box.width*780,py=(event.clientY-box.top)/box.height*280;
    if(px<60||px>732||py<45||py>225){setInspected(null);return;}
    select(nearestPerformanceIndex(plotted,start+(px-60)/672*(end-start)));
  };
  const inspectKey=(event:KeyboardEvent<HTMLDivElement>)=>{
    const current=activeIndex<0?plotted.length-1:activeIndex;
    if(["ArrowLeft","ArrowDown","ArrowRight","ArrowUp","Home","End","Escape"].includes(event.key))event.preventDefault();
    if(event.key==="Escape")setInspected(null);
    else if(event.key==="Home")select(0);else if(event.key==="End")select(plotted.length-1);
    else if(event.key==="ArrowLeft"||event.key==="ArrowDown")select(Math.max(0,current-1));
    else if(event.key==="ArrowRight"||event.key==="ArrowUp")select(Math.min(plotted.length-1,current+1));
  };
  const fullDate=(value:string)=>new Date(value).toLocaleDateString("en-US",{month:"short",day:"numeric",year:"numeric",timeZone:"UTC"});
  const ticks=Array.from({length:5},(_,i)=>lo+(hi-lo)*i/4);
  const years=Array.from({length:Math.max(0,new Date(end).getUTCFullYear()-new Date(start).getUTCFullYear()+1)},(_,i)=>Date.UTC(new Date(start).getUTCFullYear()+i,0,1)).filter(t=>t>start&&t<end);
  return <section className="pd-performance"><div className="pd-heading"><div><span className="eyebrow">Performance over time</span><h3>Is their golf getting better?</h3></div><label className="pd-select">Chart range<select aria-label="Performance chart range" value={window} onChange={e=>{setWindow(e.target.value as "3y"|"all");setInspected(null);}}><option value="3y">Past 3 years</option><option value="all">All available</option></select></label></div>
    <div className="pd-trend-cards"><article><span>20-round SMA · recent form</span><strong>{fmt(latest?.sma20)}</strong></article><article><span>50-round SMA · longer trend</span><strong>{fmt(latest?.sma50)}</strong></article><article><span>Recent − longer</span><strong>{finite(latest?.sma20)&&finite(latest?.sma50)?fmt(latest.sma20-latest.sma50):"—"}</strong><small>Descriptive change, not significance</small></article></div>
    <div className="pd-trend-options"><label className="pd-check"><input type="checkbox" checked={all} onChange={e=>{setAll(e.target.checked);setInspected(null);}}/>Include played rounds 3 and 4</label><label className="pd-check"><input type="checkbox" checked={raw} onChange={e=>setRaw(e.target.checked)}/>Show individual rounds</label></div>
    {!vals.length?<p className="pd-empty">At least 20 eligible adjusted rounds are needed for the first trend line; 50 for the longer line. Missing or field-relative results do not become zero.</p>:<div className="pd-trend-interactive" role="slider" tabIndex={0} aria-label="Inspect player form history. Use left and right arrows to move between rounds." aria-valuemin={1} aria-valuemax={Math.max(1,plotted.length)} aria-valuenow={activeIndex<0?Math.max(1,plotted.length):activeIndex+1} aria-valuetext={selected?`${selected.eventName}, ${fullDate(selected.date)}, after round ${selected.round}. 20-round SMA ${fmt(selected.sma20)}, 50-round SMA ${fmt(selected.sma50)}.`:"No complete SMA windows available"} onKeyDown={inspectKey} onBlur={()=>setInspected(null)} onPointerLeave={()=>setInspected(null)}><svg className="pd-trend-chart" onPointerMove={inspectPointer} onPointerDown={inspectPointer} viewBox="0 0 780 280" role="img" aria-label="20-round and 50-round simple moving averages of observed adjusted SG, centered on the fixed PGA reference. Higher is better; this is performance history, not a forecast.">
      {ticks.map(v=><g key={v}><line x1="60" x2="732" y1={y(v)} y2={y(v)} stroke="var(--line)"/><text x="49" y={y(v)+4} textAnchor="end" fontSize="11" fill="var(--muted)">{fmt(v)}</text></g>)}
      {years.map(t=><g key={t}><line x1={x(t)} x2={x(t)} y1="45" y2="225" stroke="var(--line)" strokeDasharray="2 5"/><text x={x(t)} y="248" textAnchor="middle" fontSize="11" fill="var(--muted)">{new Date(t).getUTCFullYear()}</text></g>)}
      <line x1="60" x2="732" y1={y(0)} y2={y(0)} stroke="var(--muted)" strokeDasharray="5 5"/>
      {raw&&points.map((p,i)=><circle key={i} cx={x(p.time)} cy={y(p.value)} r="2.3" fill="var(--muted)" fillOpacity=".28"><title>{p.date} R{p.round}: {fmt(p.value)} adjusted SG vs fixed PGA reference</title></circle>)}
      <path d={line("sma50")} fill="none" stroke="var(--chart-2)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" vectorEffect="non-scaling-stroke"/>
      <path d={line("sma20")} fill="none" stroke="var(--chart-1)" strokeWidth="1.25" strokeLinecap="round" strokeLinejoin="round" vectorEffect="non-scaling-stroke"/>
      {(["sma20","sma50"] as const).map((key,i)=>{const p=points.filter(p=>finite(p[key])).at(-1);return p?<circle key={key} cx={x(p.time)} cy={y(p[key]!)} r="2.5" fill={i===0?"var(--chart-1)":"var(--chart-2)"}><title>{p.eventName} · {p.date} R{p.round}: {key==="sma20"?"20":"50"}-round SMA {fmt(p[key])}</title></circle>:null;})}
      {active&&<g className="pd-trend-crosshair" pointerEvents="none"><line x1={x(active.time)} x2={x(active.time)} y1="45" y2="225" stroke="var(--muted-strong)" strokeDasharray="3 4" strokeWidth="1" vectorEffect="non-scaling-stroke"/>{(["sma20","sma50"] as const).map((key,i)=>finite(active[key])?<circle key={key} cx={x(active.time)} cy={y(active[key]!)} r="3.5" fill="var(--surface)" stroke={i===0?"var(--chart-1)":"var(--chart-2)"} strokeWidth="1.5" vectorEffect="non-scaling-stroke"/>:null)}</g>}
      <text x="60" y="267" fontSize="11" fill="var(--muted)">{points.length?day(points[0].date):""}</text><text x="732" y="267" textAnchor="end" fontSize="11" fill="var(--muted)">{points.length?day(points.at(-1)!.date):""}</text>
    </svg>{active&&<div className="pd-trend-tooltip" style={x(active.time)>390?{left:8}:{right:8}} role="status"><strong>{active.eventName}</strong><span>{fullDate(active.date)} · after R{active.round}{active.tour?` · ${active.tour.toUpperCase()}`:""}</span>{active.course&&<small>{active.course}</small>}<div><span>20-round SMA <b>{fmt(active.sma20)}</b></span><span>50-round SMA <b>{fmt(active.sma50)}</b></span></div><small>Trailing averages through this round</small></div>}</div>}
    <p className="pd-trend-hint">Hover or tap to inspect the nearest round and event. Keyboard: focus the chart, then use ← / →.</p>
    <div className="pd-legend"><span><i style={{background:"var(--chart-1)"}}/>20-round SMA</span><span><i style={{background:"var(--chart-2)"}}/>50-round SMA</span><span>Dashed zero = fixed 2025 PGA-start reference</span></div>
    <p className="pd-muted">{points.length} adjusted {all?"played":"R1/R2"} rounds in view · {series.length} available in history · latest {latest?.date??"unavailable"}. Each observation advances the average once; days without play do not change it. Long gaps break the drawn line. Earlier history supplies complete trailing windows at the start of the selected chart range.</p>
    <details className="pd-method"><summary>What these trends mean</summary><p>Each round is provider-adjusted total SG minus the same fixed PGA-start anchor used above ({fmt(anchor)} provider SG). A positive value means performance above that reference. The 20-round line responds faster; the 50-round line shows the broader trend. Their difference is descriptive, not a test of statistical significance or predicted future improvement.</p><p>Each SMA is the equally weighted mean of the latest 20 or 50 eligible rounds, including the current round. Each new eligible round enters the window and the oldest leaves. These are round counts, not days; inactivity does not change the average. Values are withheld until the full 20- or 50-round window is available. Revised history, schedule, individual course fit and weather exposure can still affect the lines. They are observed-performance summaries, not a course-neutral latent-skill estimate or simulation input. R1/R2 is the default to reduce weekend survivor selection; including later rounds changes the population.</p><p>The trend uses complete available adjusted history independently of the event-table filters below, so choosing a course or major does not silently redefine long-term form.</p></details>
  </section>;
}
