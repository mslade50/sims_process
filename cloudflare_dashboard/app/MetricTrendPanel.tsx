"use client";
import {useEffect,useMemo,useRef,useState} from "react";
import {useDashboardData} from "./data";
import {HOLE_BASED} from "./player-fingerprint-rules";
import {profileKey} from "./player-profile-rules";
import {finite,formatMetric,dayText,windowWords,type ShotMetric} from "./player-deep-rules";
import {parseSeries,seriesValues,seriesShotCount,shotMovingAverage,shotWindows,defaultShotWindow,shotTrendSummary,indexTicks,tickDate,type ShotSeries} from "./shot-series-rules";
import {movingAverage,fullAveragePoints,trendRoundCount,defaultTrendWindow,trendSummary,trendImproved,scopeLabel,TREND_WINDOWS,type TrendRoundRow} from "./metric-trend-rules";

const W=640,H=250,L=58,R=16,T=14,B=34;
type PanelProps={metric:ShotMetric;title:string;rows:TrendRoundRow[];filtered:boolean;descriptive:boolean;onClose:()=>void;windowDays?:number;seriesKey?:string};
/** Shot-by-shot trend when the per-shot series is published and loaded; otherwise the round-level moving average. */
export function MetricTrendPanel(props:PanelProps){
  const load=useDashboardData<unknown>(props.seriesKey?profileKey(props.seriesKey):null),series=parseSeries(load.data);
  const selected=useMemo(()=>new Set(props.rows.map(r=>`${r.event_id}|${r.round}`)),[props.rows]);
  if(series&&seriesShotCount(series,props.metric.key,selected)>0)return <ShotTrendPanel {...props} series={series} selected={selected}/>;
  return <RoundTrendPanel {...props} loadingShots={!!props.seriesKey&&load.loading}/>;
}
function ShotTrendPanel({metric,title,series,selected,filtered,descriptive,onClose,windowDays}:PanelProps&{series:ShotSeries;selected:Set<string>}){
  const shots=useMemo(()=>seriesValues(series,metric.key,selected),[series,metric.key,selected]),total=shots.length,windows=shotWindows(metric.key);
  const [win,setWin]=useState<number>(()=>defaultShotWindow(total,metric.key)),[hover,setHover]=useState<number|null>(null),closeRef=useRef<HTMLButtonElement>(null);
  useEffect(()=>{closeRef.current?.focus();},[]);
  useEffect(()=>{const k=(e:KeyboardEvent)=>{if(e.key==="Escape"){e.stopPropagation();onClose();}};document.addEventListener("keydown",k);return()=>document.removeEventListener("keydown",k);},[onClose]);
  const all=useMemo(()=>shotMovingAverage(shots,win),[shots,win]),full=all.filter(p=>p.sma!==null) as Array<typeof all[number]&{sma:number}>,sum=shotTrendSummary(all,win);
  const unit=metric.unit,lower=!descriptive&&metric.higher_is_better===false,ref=finite(metric.reference_mean)?metric.reference_mean:null;
  const fmt=(v:number|null|undefined)=>formatMetric(v,unit)+(["fraction","ratio","SG/round","field SG/round","strokes/shot","strokes/round",""].includes(unit)?"":` ${unit}`);
  const vals:number[]=[...full.map(p=>p.sma),...(ref===null?[]:[ref])];
  let lo=Math.min(...vals),hi=Math.max(...vals);if(!vals.length){lo=0;hi=1;}const pad=(hi-lo)*.12||Math.abs(hi)*.05||.5;lo-=pad;hi+=pad;
  // Every shot is one equal step along the axis; dates are only labels under it and never move the line.
  const last=Math.max(1,full.length-1),x=(i:number)=>L+(full.length>1?i/last:.5)*(W-L-R),y=(v:number)=>T+(1-(v-lo)/(hi-lo))*(H-T-B);
  const step=Math.max(1,Math.ceil(full.length/400)),line=full.map((p,i)=>[p,i] as const).filter(([,i])=>i%step===0||i===full.length-1).map(([p,i])=>`${x(i).toFixed(1)},${y(p.sma).toFixed(1)}`).join(" ");
  const ticks=indexTicks(full.length,4),tickText=(i:number)=>tickDate(full[i].time,full[0].time,full.at(-1)!.time);
  const sel=hover!==null?full[hover]:null,improved=trendImproved(sum.change,lower,descriptive),arrow=(c:number)=>c>0?"+":c<0?"−":"";
  const word=HOLE_BASED.has(metric.key)?"hole":"shot",words=`${word}s`;
  const changeText=full.length===0?`Only ${total} ${words} recorded, and a ${win}-${word} average needs ${win}. Try a shorter average.`:sum.change===null?`Needs at least ${2*win} ${words} to compare two full ${win}-${word} averages (${total} available).`:`Trend: ${arrow(sum.change)}${formatMetric(Math.abs(sum.change),unit)} over the last ${win} ${words} (${fmt(sum.previous)} → ${fmt(sum.current)})${improved===null?"":improved?" · improving":" · slipping"}`;
  const key=(e:React.KeyboardEvent)=>{if(!full.length)return;if(e.key==="ArrowRight"||e.key==="ArrowLeft"){e.preventDefault();const d=(e.shiftKey?25:1)*(e.key==="ArrowRight"?1:-1);setHover(h=>Math.max(0,Math.min(full.length-1,(h??full.length-1)+d)));}};
  const pointer=(e:React.PointerEvent<SVGSVGElement>)=>{const r=e.currentTarget.getBoundingClientRect();if(!full.length||!r.width)return;const px=(e.clientX-r.left)/r.width*W;setHover(Math.round(Math.max(0,Math.min(1,(px-L)/(W-L-R)))*last));};
  return <div className="pd-trend-backdrop" role="presentation" onClick={e=>{if(e.target===e.currentTarget)onClose();}}><section className="pd-metric-trend" role="dialog" aria-modal="false" aria-label={`${title} trend, ${win}-${word} moving average`}>
    <div className="pd-mt-head"><div><span className="eyebrow">{win}-{word} moving average · every {word} one step, in play order</span><h4>{title}</h4></div><button ref={closeRef} type="button" className="pd-mt-close" onClick={onClose} aria-label="Close trend panel">Close ✕</button></div>
    <div className="pd-mt-tools"><div className="pd-mt-windows" role="group" aria-label={`Moving average window in ${words}`}>{windows.map(w=><button key={w} type="button" aria-pressed={win===w} onClick={()=>{setWin(w);setHover(null);}}>{w}</button>)}</div><span className="pd-muted">Last {win} {words}</span></div>
    <p className="pd-mt-trend" role="status"><b>{changeText}</b></p>
    <p className="pd-muted">{full.length?<>Latest {win}-{word} average <b>{fmt(sum.current)}</b> over {total.toLocaleString("en-US")} {words}</>:<>{total} {words} recorded</>}{ref!==null?<> · PGA average {fmt(ref)}</>:" · PGA average not published for this view"}.{lower&&" Lower is better for this measure (the radar flips it so further out is still better)."}{descriptive&&" Ball flight, not skill: higher is not better or worse."}</p>
    {full.length>1?<svg className="pd-mt-chart" viewBox={`0 0 ${W} ${H}`} role="img" tabIndex={0} aria-label={`${title}: ${win}-${word} moving average over ${total} ${words}, from ${fmt(full[0].sma)} to ${fmt(sum.current)}. Use left and right arrows to read along the line.`} onKeyDown={key} onPointerMove={pointer} onPointerLeave={()=>setHover(null)} onBlur={()=>setHover(null)}>
      {[0,.5,1].map(f=>{const v=lo+(hi-lo)*f;return <g key={f}><line x1={L} x2={W-R} y1={y(v)} y2={y(v)} stroke="var(--line)"/><text x={L-6} y={y(v)+4} textAnchor="end" fontSize="10" fill="var(--muted)">{formatMetric(v,unit)}</text></g>;})}
      {ref!==null&&ref>lo&&ref<hi&&<g><line x1={L} x2={W-R} y1={y(ref)} y2={y(ref)} stroke="var(--muted-strong)" strokeDasharray="5 4"/><text x={W-R} y={y(ref)-4} textAnchor="end" fontSize="10" fill="var(--muted-strong)">PGA average {formatMetric(ref,unit)}</text></g>}
      <polyline points={line} fill="none" stroke="var(--chart-2)" strokeWidth="2.5" strokeLinejoin="round"/>
      {ticks.map((i,k)=><text key={i} x={x(i)} y={H-14} textAnchor={k===0?"start":k===ticks.length-1?"end":"middle"} fontSize="10" fill="var(--muted)">{tickText(i)}</text>)}
      <text x={(L+W-R)/2} y={H-1} textAnchor="middle" fontSize="10" fill="var(--muted)">{word[0].toUpperCase()+word.slice(1)}s in play order (dates are rough markers) · the line starts once {win} {words} are behind it</text>
      {sel&&hover!==null&&<g><line x1={x(hover)} x2={x(hover)} y1={T} y2={H-B} stroke="var(--muted)"/><circle cx={x(hover)} cy={y(sel.sma)} r="4.5" fill="var(--chart-2)"/></g>}
    </svg>:<p className="pd-empty">Not enough {words} for a {win}-{word} average yet. Try a shorter average.</p>}
    <p className="pd-mt-readout" aria-live="polite">{sel?<>{dayText(sel.date)} · {word} {sel.index.toLocaleString("en-US")} of {total.toLocaleString("en-US")} · average {fmt(sel.sma)} (last {win} {words})</>:"Hover or use the arrow keys (Shift for bigger steps) to read along the line."}</p>
    <ul className="pd-mt-notes pd-muted">
      <li>Includes {total.toLocaleString("en-US")} {words} {filtered?"from the rounds you selected in the filters":"with published shot detail"}{descriptive?" (rounds with launch-monitor readings)":""}, in play order{windowDays&&!filtered?` (last ${windowWords(windowDays)} at most)`:""}.</li>
      <li>Each point is the plain average of the last {win} {words}, so the line shows form over time rather than round-to-round noise. Every {word} takes the same step along the axis, so the line never stretches over breaks between events; the dates underneath only mark roughly when those {words} were played.</li>
    </ul>
  </section></div>;
}
function RoundTrendPanel({metric,title,rows,filtered,descriptive,onClose,windowDays,loadingShots=false}:PanelProps&{loadingShots?:boolean}){
  const [win,setWin]=useState<number>(()=>defaultTrendWindow(trendRoundCount(rows,metric.key))),[dots,setDots]=useState(false),[hover,setHover]=useState<number|null>(null);
  const closeRef=useRef<HTMLButtonElement>(null);
  useEffect(()=>{closeRef.current?.focus();},[]);
  useEffect(()=>{const k=(e:KeyboardEvent)=>{if(e.key==="Escape"){e.stopPropagation();onClose();}};document.addEventListener("keydown",k);return()=>document.removeEventListener("keydown",k);},[onClose]);
  const all=useMemo(()=>movingAverage(rows,metric.key,win),[rows,metric.key,win]),full=fullAveragePoints(all),sum=trendSummary(all,win);
  const unit=metric.unit,lower=!descriptive&&metric.higher_is_better===false,ref=finite(metric.reference_mean)?metric.reference_mean:null;
  const fmt=(v:number|null|undefined)=>formatMetric(v,unit)+(["fraction","ratio","SG/round","field SG/round","strokes/shot","strokes/round",""].includes(unit)?"":` ${unit}`);
  // The average line starts at the first round that has a full window behind it; single rounds can still be shown before that.
  const total=all.length,showDots=dots||full.length===0,startIdx=showDots||!full.length?1:full[0].index,shown=all.filter(p=>p.index>=startIdx),n=shown.length;
  const vals:number[]=[...full.map(p=>p.sma),...(showDots?all.map(p=>p.value):[]),...(ref===null?[]:[ref])];
  let lo=Math.min(...vals),hi=Math.max(...vals);if(!vals.length){lo=0;hi=1;}const pad=(hi-lo)*.12||Math.abs(hi)*.05||.5;lo-=pad;hi+=pad;
  const x=(i:number)=>L+(total<=startIdx?.5:(i-startIdx)/(total-startIdx))*(W-L-R),y=(v:number)=>T+(1-(v-lo)/(hi-lo))*(H-T-B);
  const line=full.map(p=>`${x(p.index).toFixed(1)},${y(p.sma).toFixed(1)}`).join(" ");
  const ticks=indexTicks(total-startIdx+1,4).map(i=>startIdx+i);
  const dt=(i:number)=>Date.parse(all[i-1]?.date??""),roundTick=(t:number)=>Number.isFinite(dt(t))&&Number.isFinite(dt(startIdx))&&Number.isFinite(dt(total))?tickDate(dt(t),dt(startIdx),dt(total)):String(t);
  const sel=hover!==null?shown[hover]:null;
  const improved=trendImproved(sum.change,lower,descriptive);
  const arrow=(c:number)=>c>0?"+":c<0?"−":"";
  const changeText=full.length===0?`Only ${total} ${descriptive?"radar ":""}round${total===1?"":"s"} recorded this, and a ${win}-round average needs ${win}. Try a shorter average.`:sum.change===null?`Needs at least ${2*win} rounds to compare two full ${win}-round averages (${total} available).`:`Trend: ${arrow(sum.change)}${formatMetric(Math.abs(sum.change),unit)} over the last ${win} rounds (${fmt(sum.previous)} → ${fmt(sum.current)})${improved===null?"":improved?" · improving":" · slipping"}`;
  const key=(e:React.KeyboardEvent)=>{if(!n)return;if(e.key==="ArrowRight"||e.key==="ArrowLeft"){e.preventDefault();setHover(h=>Math.max(0,Math.min(n-1,(h??n-1)+(e.key==="ArrowRight"?1:-1))));}};
  const pointer=(e:React.PointerEvent<SVGSVGElement>)=>{const r=e.currentTarget.getBoundingClientRect();if(!n||!r.width)return;const px=(e.clientX-r.left)/r.width*W;setHover(Math.max(0,Math.min(n-1,Math.round((px-L)/(W-L-R)*(n-1)))));};
  return <div className="pd-trend-backdrop" role="presentation" onClick={e=>{if(e.target===e.currentTarget)onClose();}}><section className="pd-metric-trend" role="dialog" aria-modal="false" aria-label={`${title} trend, ${win}-round moving average`}>
    <div className="pd-mt-head"><div><span className="eyebrow">{win}-round moving average · round by round, not by date</span><h4>{title}</h4></div><button ref={closeRef} type="button" className="pd-mt-close" onClick={onClose} aria-label="Close trend panel">Close ✕</button></div>
    <div className="pd-mt-tools"><div className="pd-mt-windows" role="group" aria-label="Moving average window">{TREND_WINDOWS.map(w=><button key={w} type="button" aria-pressed={win===w} onClick={()=>{setWin(w);setHover(null);}}>{w}</button>)}</div><label><input type="checkbox" checked={showDots} disabled={full.length===0} onChange={e=>{setDots(e.target.checked);setHover(null);}}/> Show each {descriptive?"radar ":""}round</label></div>
    {total===0?<p className="pd-empty">No {descriptive?"radar ":""}rounds with this measure are available for the current selection.</p>:<>
      <p className="pd-mt-trend" role="status"><b>{changeText}</b></p>
      <p className="pd-muted">{full.length?<>Latest {win}-round average <b>{fmt(sum.current)}</b> over {total} {descriptive?"radar rounds":"rounds"}</>:<>{total} {descriptive?"radar rounds":"rounds"} shown as single rounds</>}{ref!==null?<> · PGA average {fmt(ref)}</>:" · PGA average not published for this view"}.{lower&&" Lower is better for this measure (the radar flips it so further out is still better)."}{descriptive&&" Ball flight, not skill: higher is not better or worse."}</p>
      <svg className="pd-mt-chart" viewBox={`0 0 ${W} ${H}`} role="img" tabIndex={0} aria-label={`${title}: ${win}-round moving average over ${total} rounds${full.length?`, from ${fmt(full[0].sma)} to ${fmt(sum.current)}`:" (not enough rounds yet; single rounds shown)"}. Use left and right arrows to read each round.`} onKeyDown={key} onPointerMove={pointer} onPointerLeave={()=>setHover(null)} onBlur={()=>setHover(null)}>
        {[0,.5,1].map(f=>{const v=lo+(hi-lo)*f;return <g key={f}><line x1={L} x2={W-R} y1={y(v)} y2={y(v)} stroke="var(--line)"/><text x={L-6} y={y(v)+4} textAnchor="end" fontSize="10" fill="var(--muted)">{formatMetric(v,unit)}</text></g>;})}
        {ref!==null&&ref>lo&&ref<hi&&<g><line x1={L} x2={W-R} y1={y(ref)} y2={y(ref)} stroke="var(--muted-strong)" strokeDasharray="5 4"/><text x={W-R} y={y(ref)-4} textAnchor="end" fontSize="10" fill="var(--muted-strong)">PGA average {formatMetric(ref,unit)}</text></g>}
        {showDots&&shown.map(p=><circle key={p.index} cx={x(p.index)} cy={y(p.value)} r="2.2" fill="var(--chart-1)" opacity=".28"/>)}
        {full.length>1&&<polyline points={line} fill="none" stroke="var(--chart-2)" strokeWidth="2.5" strokeLinejoin="round"/>}
        {full.length===1&&<circle cx={x(full[0].index)} cy={y(full[0].sma)} r="4" fill="var(--chart-2)"/>}
        {ticks.map((t,k)=><text key={t} x={x(t)} y={H-14} textAnchor={ticks.length>1&&k===0?"start":ticks.length>1&&k===ticks.length-1?"end":"middle"} fontSize="10" fill="var(--muted)">{roundTick(t)}</text>)}
        <text x={(L+W-R)/2} y={H-1} textAnchor="middle" fontSize="10" fill="var(--muted)">Rounds in order, oldest → newest (dates are rough markers){full.length>0&&full[0].index>1&&showDots?` · the average starts at round ${full[0].index}`:""}</text>
        {sel&&<g><line x1={x(sel.index)} x2={x(sel.index)} y1={T} y2={H-B} stroke="var(--muted)"/>{finite(sel.sma)&&<circle cx={x(sel.index)} cy={y(sel.sma)} r="4.5" fill="var(--chart-2)"/>}</g>}
      </svg>
      <p className="pd-mt-readout" aria-live="polite">{sel?<>Round {sel.index} · {dayText(sel.date)} · {finite(sel.sma)?<>average {fmt(sel.sma)} (last {win} rounds)</>:<>no {win}-round average yet (it starts at round {win})</>} · this round {fmt(sel.value)} from {sel.n} {sel.n===1?"shot":"shots"}</>:"Hover or use the arrow keys to read a round."}</p>
      <ul className="pd-mt-notes pd-muted">
        {loadingShots&&<li>Loading the shot-by-shot version of this chart…</li>}<li>Rounds included: {scopeLabel(total,filtered,descriptive,windowDays)}.{descriptive&&" Rounds without launch-monitor readings are skipped."}</li>
        <li>Each point is the shot-weighted average (total ÷ shots) over the {win} most recent rounds that recorded this measure. The line starts at round {win}, the first round with a full {win}-round window, so there are no short-window values.</li>
        {sum.sparse&&<li>Small samples: a typical round has about {formatMetric(sum.medianShotsPerRound,"")} shots for this measure, so single rounds are noisy.</li>}
      </ul></>}
  </section></div>;
}
