"use client";
import {useMemo,useState,type KeyboardEvent,type PointerEvent} from "react";
import {RULER_NAME} from "./labels";
import {indexTicks,tickDate} from "./shot-series-rules";
import {finite,type DeepProfile} from "./player-deep-rules";
import {SMA_WINDOWS,DEFAULT_SMA,MAJOR_FILL,TOTAL_RANGE,nearestPerformanceIndex,normalizeSmas,performanceSeries,recentVsLonger,roundBarFill,smaOf,visiblePerformance,type SmaWindow} from "./player-performance-rules";
const fmt=(v:number|null|undefined)=>finite(v)?`${v>0?"+":""}${v.toFixed(2)}`:"—";
const SMA_COLOR:Record<SmaWindow,string>={20:"var(--chart-1)",50:"var(--chart-2)",100:"var(--chart-6)",200:"var(--chart-3)"};
const SMA_DASH:Record<SmaWindow,string|undefined>={20:undefined,50:undefined,100:"7 3",200:"2 3"};
export function PerformanceTrend({deep,anchor}:{deep:DeepProfile;anchor:number|null|undefined}){
  const [all,setAll]=useState(true),[raw,setRaw]=useState(false),[sel,setSel]=useState<SmaWindow[]>(DEFAULT_SMA),[window,setWindow]=useState<"3y"|"all">("3y");
  const [inspected,setInspected]=useState<string|null>(null);
  const history=deep.history;
  const series=useMemo(()=>performanceSeries(history?.events??[],anchor,all),[history,anchor,all]);
  const points=visiblePerformance(series,deep.as_of,window),latest=series.at(-1);
  const shown=normalizeSmas(sel),shownLabel=shown.join("/"),rel=recentVsLonger(latest,shown);
  const toggleSma=(w:SmaWindow)=>{setSel(normalizeSmas(shown.includes(w)?shown.filter(k=>k!==w):[...shown,w]));setInspected(null);};
  const plotted=points.filter(p=>shown.some(w=>finite(smaOf(p,w)))),activeIndex=plotted.findIndex(p=>`${p.event}:${p.round}`===inspected),active=plotted[activeIndex],selected=active??plotted.at(-1);
  const select=(index:number)=>{const p=plotted[index];setInspected(p?`${p.event}:${p.round}`:null);};
  const vals=points.flatMap(p=>[...shown.map(w=>smaOf(p,w)),...(raw?[p.value]:[])]).filter(finite);
  const lo=Math.floor((Math.min(0,...vals)-.2)*2)/2,hi=Math.ceil((Math.max(0,...vals)+.2)*2)/2;
  // Every round is one equal step along the axis; dates are only labels under it and never move the lines.
  const pos=new Map(points.map((p,i)=>[`${p.event}:${p.round}`,i])),at=(p:{event:string;round:number})=>pos.get(`${p.event}:${p.round}`)??0;
  const x=(i:number)=>60+(points.length>1?i/(points.length-1):.5)*672,y=(v:number)=>225-(v-lo)/Math.max(.1,hi-lo)*180;
  const line=(w:SmaWindow)=>{let first=true;return points.flatMap((p,i)=>{const v=smaOf(p,w);if(!finite(v))return [];const move=first;first=false;return [`${move?"M":"L"}${x(i).toFixed(2)},${y(v).toFixed(2)}`];}).join(" ");};
  const inspectPointer=(event:PointerEvent<SVGSVGElement>)=>{
    const box=event.currentTarget.getBoundingClientRect(),px=(event.clientX-box.left)/box.width*780,py=(event.clientY-box.top)/box.height*280;
    if(px<60||px>732||py<45||py>225){setInspected(null);return;}
    select(nearestPerformanceIndex(plotted.map(p=>({time:at(p)})),(px-60)/672*Math.max(0,points.length-1)));
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
  const dateTicks=indexTicks(points.length,4);
  return <section className="pd-performance"><div className="pd-heading"><div><span className="eyebrow">Performance over time</span><h3>Is their golf getting better?</h3></div><label className="pd-select">Chart range<select aria-label="Performance chart range" value={window} onChange={e=>{setWindow(e.target.value as "3y"|"all");setInspected(null);}}><option value="3y">Past 3 years</option><option value="all">All available</option></select></label></div>
    <div className="pd-trend-cards">{shown.map(w=><article key={w}><span><i style={{display:"inline-block",width:10,height:3,marginRight:6,verticalAlign:"middle",background:SMA_COLOR[w]}}/>{w}-round SMA{w===20?" · recent form":w===50?" · longer trend":""}</span><strong>{fmt(latest&&smaOf(latest,w))}</strong></article>)}{rel&&<article><span>{rel.window}-round − 50-round</span><strong>{fmt(rel.diff)}</strong><small>Descriptive change, not significance</small></article>}</div>
    <div className="pd-trend-options"><label className="pd-check"><input type="checkbox" checked={all} onChange={e=>{setAll(e.target.checked);setInspected(null);}}/>Include played rounds 3 and 4 (all rounds, R1–R4)</label><label className="pd-check"><input type="checkbox" checked={raw} onChange={e=>setRaw(e.target.checked)}/>Show individual rounds</label><span role="group" aria-label="Moving averages to show" style={{display:"inline-flex",gap:10,flexWrap:"wrap"}}>{SMA_WINDOWS.map(w=><label key={w} className="pd-check"><input type="checkbox" checked={shown.includes(w)} disabled={shown.length===1&&shown[0]===w} onChange={()=>toggleSma(w)}/><i style={{display:"inline-block",width:12,height:3,background:SMA_COLOR[w]}}/>{w}-round SMA</label>)}</span></div>
    {!vals.length?<p className="pd-empty">At least {shown[0]} eligible adjusted rounds are needed for the first trend line shown. Missing or field-relative results do not become zero.</p>:<div className="pd-trend-interactive" role="slider" tabIndex={0} aria-label="Inspect player form history. Use left and right arrows to move between rounds." aria-valuemin={1} aria-valuemax={Math.max(1,plotted.length)} aria-valuenow={activeIndex<0?Math.max(1,plotted.length):activeIndex+1} aria-valuetext={selected?`${selected.eventName}, ${fullDate(selected.date)}, after round ${selected.round}${selected.major?" (major)":""}. ${shown.map(w=>`${w}-round SMA ${fmt(smaOf(selected,w))}`).join(", ")}.`:"No complete SMA windows available"} onKeyDown={inspectKey} onBlur={()=>setInspected(null)} onPointerLeave={()=>setInspected(null)}><svg className="pd-trend-chart" onPointerMove={inspectPointer} onPointerDown={inspectPointer} viewBox="0 0 780 280" role="img" aria-label={`${shownLabel}-round simple moving averages of observed adjusted SG, centered on the fixed PGA reference. Higher is better; this is performance history, not a forecast.`}>
      {ticks.map(v=><g key={v}><line x1="60" x2="732" y1={y(v)} y2={y(v)} stroke="var(--line)"/><text x="49" y={y(v)+4} textAnchor="end" fontSize="11" fill="var(--muted)">{fmt(v)}</text></g>)}
      <line x1="60" x2="732" y1={y(0)} y2={y(0)} stroke="var(--muted)" strokeDasharray="5 5"/>
      {raw&&points.map((p,i)=><line key={i} x1={x(i)} x2={x(i)} y1={y(0)} y2={y(p.value)} strokeWidth="2" strokeLinecap="butt" style={{stroke:roundBarFill(p)}}><title>{p.date} R{p.round}{p.major?" (major)":""}: {fmt(p.value)} adjusted SG vs fixed PGA reference</title></line>)}
      {[...shown].reverse().map(w=><path key={w} d={line(w)} fill="none" stroke={SMA_COLOR[w]} strokeDasharray={SMA_DASH[w]} strokeWidth={w===50?"1.75":"1.4"} strokeLinecap="round" strokeLinejoin="round" vectorEffect="non-scaling-stroke"/>)}
      {shown.map(w=>{const p=points.filter(p=>finite(smaOf(p,w))).at(-1);return p?<circle key={w} cx={x(at(p))} cy={y(smaOf(p,w)!)} r="2.5" fill={SMA_COLOR[w]}><title>{p.eventName} · {p.date} R{p.round}: {w}-round SMA {fmt(smaOf(p,w))}</title></circle>:null;})}
      {active&&<g className="pd-trend-crosshair" pointerEvents="none"><line x1={x(at(active))} x2={x(at(active))} y1="45" y2="225" stroke="var(--muted-strong)" strokeDasharray="3 4" strokeWidth="1" vectorEffect="non-scaling-stroke"/>{shown.map(w=>finite(smaOf(active,w))?<circle key={w} cx={x(at(active))} cy={y(smaOf(active,w)!)} r="3.5" fill="var(--surface)" stroke={SMA_COLOR[w]} strokeWidth="1.5" vectorEffect="non-scaling-stroke"/>:null)}</g>}
      {dateTicks.map((i,k)=><text key={i} x={x(i)} y="248" textAnchor={dateTicks.length>1&&k===0?"start":dateTicks.length>1&&k===dateTicks.length-1?"end":"middle"} fontSize="11" fill="var(--muted)">{tickDate(points[i].time,points[0].time,points.at(-1)!.time)}</text>)}
      <text x="396" y="267" textAnchor="middle" fontSize="10" fill="var(--muted)">Rounds in order, oldest → newest (dates are rough markers)</text>
    </svg>{active&&<div className="pd-trend-tooltip" style={x(at(active))>390?{left:8}:{right:8}} role="status"><strong>{active.eventName}</strong><span>{fullDate(active.date)} · after R{active.round}{active.major?" · Major":""}{active.tour?` · ${active.tour.toUpperCase()}`:""}</span>{active.course&&<small>{active.course}</small>}<div><span>Round <b>{fmt(active.value)}</b></span>{shown.map(w=><span key={w}>{w}-round SMA <b>{fmt(smaOf(active,w))}</b></span>)}</div><small>Trailing averages through this round</small></div>}</div>}
    <p className="pd-trend-hint">Hover or tap to inspect the nearest round and event. Keyboard: focus the chart, then use ← / →.</p>
    <div className="pd-legend">{shown.map(w=><span key={w}><i style={{background:SMA_COLOR[w]}}/>{w}-round SMA</span>)}{raw&&<><span><i style={{background:MAJOR_FILL}}/>Major (gold)</span><span><i style={{background:"color-mix(in srgb, var(--positive) 70%, transparent)"}}/>Good round<i style={{background:"color-mix(in srgb, var(--negative) 70%, transparent)",marginLeft:6}}/>Poor round (full colour at ±{TOTAL_RANGE}, as in the event table)</span></>}<span>Dashed zero = {RULER_NAME}</span></div>
    <p className="pd-muted">{points.length} adjusted {all?"R1–R4":"R1/R2"} rounds in view · {series.length} available in history · latest {latest?.date??"unavailable"}. Each round is one equal step along the chart, so breaks between events take no space; the dates underneath only mark roughly when those rounds were played. Earlier history supplies complete trailing windows at the start of the selected chart range.</p>
    <details className="pd-method"><summary>What these trends mean</summary><p>Each round is provider-adjusted total SG minus the same fixed PGA-start anchor used above ({fmt(anchor)} provider SG). A positive value means performance above that reference. Shorter lines respond faster; longer lines show the broader trend (50 rounds by default; 20, 100 and 200 can be added). Differences between lines is descriptive, not a test of statistical significance or predicted future improvement.</p><p>Each SMA is the equally weighted mean of the latest N eligible rounds (N = 20, 50, 100 or 200), including the current round. Each new eligible round enters the window and the oldest leaves. These are round counts, not days; inactivity does not change the average. Values are withheld until the full window is available. Revised history, schedule, individual course fit and weather exposure can still affect the lines. They are observed-performance summaries, not a course-neutral latent-skill estimate or simulation input. Made-cut caveat: rounds 3 and 4 are only played by players who made the cut, so they are survivor-selected against an R1/R2-based reference; untick the rounds 3 and 4 box for R1/R2 only. Bars use the event table’s Total colour scale; major rounds are gold.</p><p>The trend uses complete available adjusted history independently of the event-table filters below, so choosing a course or major does not silently redefine long-term form.</p></details>
  </section>;
}
