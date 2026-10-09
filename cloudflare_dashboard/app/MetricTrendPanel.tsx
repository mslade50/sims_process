"use client";
import {useEffect,useMemo,useRef,useState} from "react";
import {finite,formatMetric,dayText,type ShotMetric} from "./player-deep-rules";
import {movingAverage,trendSummary,trendImproved,scopeLabel,TREND_WINDOWS,DEFAULT_TREND_WINDOW,type TrendRoundRow} from "./metric-trend-rules";

const W=640,H=250,L=58,R=16,T=14,B=34;
export function MetricTrendPanel({metric,title,rows,filtered,descriptive,onClose}:{metric:ShotMetric;title:string;rows:TrendRoundRow[];filtered:boolean;descriptive:boolean;onClose:()=>void}){
  const [win,setWin]=useState<number>(DEFAULT_TREND_WINDOW),[dots,setDots]=useState(false),[hover,setHover]=useState<number|null>(null);
  const closeRef=useRef<HTMLButtonElement>(null);
  useEffect(()=>{closeRef.current?.focus();},[]);
  useEffect(()=>{const k=(e:KeyboardEvent)=>{if(e.key==="Escape"){e.stopPropagation();onClose();}};document.addEventListener("keydown",k);return()=>document.removeEventListener("keydown",k);},[onClose]);
  const pts=useMemo(()=>movingAverage(rows,metric.key,win),[rows,metric.key,win]),sum=trendSummary(pts,win);
  const unit=metric.unit,lower=!descriptive&&metric.higher_is_better===false,ref=finite(metric.reference_mean)?metric.reference_mean:null;
  const fmt=(v:number|null|undefined)=>formatMetric(v,unit)+(["fraction","ratio","SG/round","field SG/round","strokes/shot","strokes/round",""].includes(unit)?"":` ${unit}`);
  const vals=pts.flatMap(p=>[p.sma,...(dots?[p.value]:[])]).concat(ref===null?[]:[ref]);
  let lo=Math.min(...vals),hi=Math.max(...vals);if(!vals.length){lo=0;hi=1;}const pad=(hi-lo)*.12||Math.abs(hi)*.05||.5;lo-=pad;hi+=pad;
  const n=pts.length,x=(i:number)=>L+(n<=1?.5:(i-1)/(n-1))*(W-L-R),y=(v:number)=>T+(1-(v-lo)/(hi-lo))*(H-T-B);
  const line=pts.map(p=>`${x(p.index).toFixed(1)},${y(p.sma).toFixed(1)}`).join(" ");
  const step=Math.max(1,Math.ceil(n/6)),ticks=n?[...new Set([...Array.from({length:Math.floor((n-1)/step)+1},(_,i)=>1+i*step).filter(t=>n-t>=step/2||t===n),n])]:[];
  const sel=hover!==null?pts[hover]:null;
  const improved=trendImproved(sum.change,lower,descriptive);
  const arrow=(c:number)=>c>0?"+":c<0?"−":"";
  const changeText=sum.change===null?`Needs more than ${win} rounds for a ${win}-round comparison.`:`Trend: ${arrow(sum.change)}${formatMetric(Math.abs(sum.change),unit)} over the last ${win} rounds (${fmt(sum.previous)} → ${fmt(sum.current)})${improved===null?"":improved?" · improving":" · slipping"}`;
  const key=(e:React.KeyboardEvent)=>{if(!n)return;if(e.key==="ArrowRight"||e.key==="ArrowLeft"){e.preventDefault();setHover(h=>Math.max(0,Math.min(n-1,(h??n-1)+(e.key==="ArrowRight"?1:-1))));}};
  const pointer=(e:React.PointerEvent<SVGSVGElement>)=>{const r=e.currentTarget.getBoundingClientRect();if(!n||!r.width)return;const px=(e.clientX-r.left)/r.width*W;setHover(Math.max(0,Math.min(n-1,Math.round((px-L)/(W-L-R)*(n-1)))));};
  return <div className="pd-trend-backdrop" role="presentation" onClick={e=>{if(e.target===e.currentTarget)onClose();}}><section className="pd-metric-trend" role="dialog" aria-modal="false" aria-label={`${title} trend, ${win}-round moving average`}>
    <div className="pd-mt-head"><div><span className="eyebrow">{win}-round moving average · round by round, not by date</span><h4>{title}</h4></div><button ref={closeRef} type="button" className="pd-mt-close" onClick={onClose} aria-label="Close trend panel">Close ✕</button></div>
    <div className="pd-mt-tools"><div className="pd-mt-windows" role="group" aria-label="Moving average window">{TREND_WINDOWS.map(w=><button key={w} type="button" aria-pressed={win===w} onClick={()=>setWin(w)}>{w}</button>)}</div><label><input type="checkbox" checked={dots} onChange={e=>setDots(e.target.checked)}/> Show each {descriptive?"radar ":""}round</label></div>
    {n===0?<p className="pd-empty">No {descriptive?"radar ":""}rounds with this measure are available for the current selection.</p>:<>
      <p className="pd-mt-trend" role="status"><b>{changeText}</b></p>
      <p className="pd-muted">Latest {win}-round average <b>{fmt(sum.current)}</b> over {n} {descriptive?"radar rounds":"rounds"}{ref!==null?<> · PGA average {fmt(ref)}</>:" · PGA average not published for this view"}.{lower&&" Lower is better for this measure (the radar flips it so further out is still better)."}{descriptive&&" Ball flight, not skill: higher is not better or worse."}</p>
      <svg className="pd-mt-chart" viewBox={`0 0 ${W} ${H}`} role="img" tabIndex={0} aria-label={`${title}: ${win}-round moving average over ${n} rounds, from ${fmt(pts[0].sma)} to ${fmt(sum.current)}. Use left and right arrows to read each round.`} onKeyDown={key} onPointerMove={pointer} onPointerLeave={()=>setHover(null)} onBlur={()=>setHover(null)}>
        {[0,.5,1].map(f=>{const v=lo+(hi-lo)*f;return <g key={f}><line x1={L} x2={W-R} y1={y(v)} y2={y(v)} stroke="var(--line)"/><text x={L-6} y={y(v)+4} textAnchor="end" fontSize="10" fill="var(--muted)">{formatMetric(v,unit)}</text></g>;})}
        {ref!==null&&ref>lo&&ref<hi&&<g><line x1={L} x2={W-R} y1={y(ref)} y2={y(ref)} stroke="var(--muted-strong)" strokeDasharray="5 4"/><text x={W-R} y={y(ref)-4} textAnchor="end" fontSize="10" fill="var(--muted-strong)">PGA average {formatMetric(ref,unit)}</text></g>}
        {dots&&pts.map(p=><circle key={p.index} cx={x(p.index)} cy={y(p.value)} r="2.2" fill="var(--chart-1)" opacity=".28"/>)}
        <polyline points={line} fill="none" stroke="var(--chart-2)" strokeWidth="2.5" strokeLinejoin="round"/>
        {n===1&&<circle cx={x(1)} cy={y(pts[0].sma)} r="4" fill="var(--chart-2)"/>}
        {ticks.map(t=><text key={t} x={x(t)} y={H-14} textAnchor="middle" fontSize="10" fill="var(--muted)">{t}</text>)}
        <text x={(L+W-R)/2} y={H-1} textAnchor="middle" fontSize="10" fill="var(--muted)">Round number (oldest → newest)</text>
        {sel&&<g><line x1={x(sel.index)} x2={x(sel.index)} y1={T} y2={H-B} stroke="var(--muted)"/><circle cx={x(sel.index)} cy={y(sel.sma)} r="4.5" fill="var(--chart-2)"/></g>}
      </svg>
      <p className="pd-mt-readout" aria-live="polite">{sel?<>Round {sel.index} · {dayText(sel.date)} · average {fmt(sel.sma)} (last {sel.windowRounds} rounds) · this round {fmt(sel.value)} from {sel.n} {sel.n===1?"shot":"shots"}</>:"Hover or use the arrow keys to read a round."}</p>
      <ul className="pd-mt-notes pd-muted">
        <li>Rounds included: {scopeLabel(n,filtered,descriptive)}.{descriptive&&" Rounds without launch-monitor readings are skipped."}</li>
        <li>Each point is the shot-weighted average (total ÷ shots) over the {win} most recent rounds that recorded this measure{n<win?`; only ${n} rounds exist, so the line is an average of fewer than ${win}`:`; the first ${win-1} points average fewer rounds`}.</li>
        {sum.sparse&&<li>Small samples: a typical round has about {formatMetric(sum.medianShotsPerRound,"")} shots for this measure, so single rounds are noisy.</li>}
      </ul></>}
  </section></div>;
}
