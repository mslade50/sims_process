"use client";

import {useState} from "react";
import {Panel} from "./components";
import type {PlayerProfile} from "./player-profile-rules";
import {exploratoryStandouts, finite, formatMetric, shapePoints, dayText, unitText, GLOSS, type DeepProfile, type ShotMetric} from "./player-deep-rules";
import {HISTORY_DEFAULTS, type HistoryFilters} from "./player-history-filters";
import {PlayerHistoryExplorer} from "./PlayerHistoryExplorer";
import "./player-deep.css";
import {PerformanceTrend} from "./PlayerPerformanceTrend";

const date=(v:string|undefined)=>dayText(v,"Not available");
const signed=(v:unknown)=>finite(v)?`${v>0?"+":""}${v.toFixed(2)}`:"—";
const groups={
  "All-round detail":["tee_distance","tee_accuracy","tee_dispersion","app_great","app_150_200","arg_rough"],
  "Driving":["tee_distance","tee_accuracy","tee_dispersion","tee_sg","tee_great","tee_poor"],
  "Putting":["putt_0_5","putt_5_10","putt_10_20","putt_20_plus","putt_short_make","putt_lag_leave"],
  "Approach":["app_50_100","app_100_150","app_150_200","app_200_plus","app_great","prox_rough"],
  "Short game":["arg_sand","arg_rough","arg_fairway","prox_fairway","prox_rough","app_poor"],
};
const axisLabel:Record<string,string>={tee_sg:"Tee shots (SG)",tee_great:"Great tee shots",tee_poor:"Avoids poor tee shots",putt_0_5:"Under 5 ft (SG)",putt_5_10:"5–10 ft (SG)",putt_10_20:"10–20 ft (SG)",putt_20_plus:"20+ ft (SG)",putt_short_make:"Short-putt makes",putt_lag_leave:"Lag putt leave",tee_distance:"Distance",tee_accuracy:"Fairways hit",tee_dispersion:"Tee dispersion",app_great:"Great approaches",app_poor:"Avoids poor approaches",app_50_100:"50–100 yds",app_100_150:"100–150 yds",app_150_200:"150–200 yds",app_200_plus:"200+ yds",arg_sand:"Bunker",arg_rough:"From rough",arg_fairway:"From fairway",prox_fairway:"Proximity, fairway",prox_rough:"Proximity, rough"};

export function ShotDetailsShape({deep,filtered=false,baselineMetrics}:{deep:DeepProfile|null;filtered?:boolean;baselineMetrics?:ShotMetric[]}){
  const [group,setGroup]=useState<keyof typeof groups>("All-round detail");
  const metrics=deep?.shots?.metrics??[];
  const keys=groups[group];const shape=shapePoints(metrics,keys),baselineShape=shapePoints(baselineMetrics??[],keys);
  const point=(i:number,r:number)=>{const a=(i*60-90)*Math.PI/180;return [190+Math.cos(a)*106*r,158+Math.sin(a)*106*r];};
  const polygon=(r:number)=>keys.map((_,i)=>point(i,r).join(",")).join(" ");
  const strong=exploratoryStandouts(metrics,keys);const hasPct=metrics.some(m=>m.status==="available"&&finite(m.reference_percentile));
  return <div className="pd-shot"><div className="pd-heading"><div><span className="eyebrow">The details behind the score</span><h3>Shot fingerprint</h3></div><label className="pd-select"><span className="sr-only">Shot fingerprint axes</span><select value={group} onChange={e=>setGroup(e.target.value as keyof typeof groups)}>{Object.keys(groups).map(k=><option key={k}>{k}</option>)}</select></label></div>
    {!metrics.length?<p className="pd-empty">Published shot detail is unavailable for this player. Feeder and amateur results can still appear in event history.</p>:<>
      <svg className="pd-fingerprint" viewBox="0 0 380 326" role="img" aria-label={`Shot detail compared with PGA players. Further out is better on every axis.${baselineMetrics?" Dashed slate overlay shows this player’s baseline in the same period.":""} Exact values are listed below.`}>
        {[.25,.5,.75,1].map(r=><polygon key={r} points={polygon(r)} fill="none" stroke={r===.5?"var(--muted)":"var(--line)"} strokeDasharray={r===.5?"4 4":undefined}/>)}
        {keys.map((k,i)=>{const end=point(i,1),label=point(i,1.34);return <g key={k}><line x1="190" y1="158" x2={end[0]} y2={end[1]} stroke="var(--line)"/><text x={label[0]} y={label[1]} textAnchor="middle" fill="var(--muted-strong)" fontSize="11">{axisLabel[k]}</text></g>;})}
        {baselineMetrics&&<g data-player-baseline="shots">{baselineShape.complete&&<polygon points={baselineShape.radii.map((r,i)=>point(i,r!).join(",")).join(" ")} fill="var(--muted-strong)" fillOpacity=".04" stroke="var(--muted-strong)" strokeWidth="2" strokeDasharray="9 5"/>}{baselineShape.radii.map((r,i)=>r===null?null:<circle key={keys[i]} cx={point(i,r)[0]} cy={point(i,r)[1]} r="4" fill="var(--surface)" stroke="var(--muted-strong)" strokeWidth="1.5"><title>Player baseline: {baselineMetrics.find(m=>m.key===keys[i])?.label}: {signed(baselineMetrics.find(m=>m.key===keys[i])?.reference_z)} SDs vs PGA · {baselineMetrics.find(m=>m.key===keys[i])?.n_shots??0} shots</title></circle>)}</g>}
        {shape.complete&&<polygon points={shape.radii.map((r,i)=>point(i,r!).join(",")).join(" ")} fill="var(--chart-2)" fillOpacity=".12" stroke="var(--chart-2)" strokeWidth="2"/>}
        {shape.radii.map((r,i)=>r===null?null:<circle key={keys[i]} cx={point(i,r)[0]} cy={point(i,r)[1]} r="4" fill="var(--chart-2)"><title>{metrics.find(m=>m.key===keys[i])?.label}: {signed(metrics.find(m=>m.key===keys[i])?.reference_z)} SDs vs PGA; further out is better</title></circle>)}
        <text x="196" y="101" fill="var(--muted)" fontSize="10">PGA average</text>
      </svg>{baselineMetrics&&<div className="pd-legend"><span><i style={{background:"var(--chart-2)"}}/>Selected rounds</span><span><i style={{background:"var(--muted-strong)"}}/>Player’s overall baseline (dashed)</span></div>}<div className="pd-shot-values">{keys.map(k=>{const m=metrics.find(v=>v.key===k),bm=baselineMetrics?.find(v=>v.key===k);return <div key={k}><span>{k==="tee_poor"?"Poor tee-shot rate":k==="app_poor"?"Poor approach rate":axisLabel[k]}</span><b>{formatMetric(m?.value,m?.unit??"")}</b><small>{m?.higher_is_better===false?"Lower is better · ":""}{m?.n_shots??0} shots{m?.status!=="available"||(m?.n_shots??0)<100||(m?.n_rounds??0)<10?" · small sample":""}</small>{baselineMetrics&&<small>Overall baseline {formatMetric(bm?.value,bm?.unit??"")} · {bm?.n_shots??0} shots</small>}</div>;})}</div><p className="pd-muted">{strong.length?<>Standout strengths here: <b>{strong.map(m=>m.label.toLowerCase()).join(" · ")}</b>. </>:null}Each spoke compares this player with PGA players on that measure; further from the centre is better, and the dashed ring is the PGA average. {shape.complete?"":"Some spokes lack enough shots, so only the available points are drawn."}</p><details className="pd-method"><summary>How this is calculated</summary><p>Every spoke is scored separately against PGA players with at least 100 shots and 10 rounds in the last three years (reference date {date(metrics.find(m=>m.reference_as_of)?.reference_as_of)}), in standard deviations, and drawn between −2.5 and +2.5. The shape is a picture, not a single skill rating, and it does not adjust for course or shot difficulty. Distance counts every club off par 4 and 5 tees; dispersion is measured around the field’s landing line, not the target. {group==="Driving"&&"Tee-shot results leave out penalty and drop records, so the poor-shot rate is not all driving trouble. "}{group==="Putting"&&"Putting buckets are in feet. Short-putt make rate and lag leave distance reflect the actual mix of putt lengths, so they are not fixed-distance odds or three-putt rates."}</p></details>
      <details className="pd-method"><summary>All shot numbers · {metrics.filter(m=>m.status==="available").length} of {metrics.length} have enough data</summary><div className="pd-scroll"><table className="pd-table"><caption>Each measure uses different shots, so they do not add up to total strokes gained.</caption><thead><tr><th>Measure</th><th>Value</th><th title="Number of shots and rounds behind the value.">Shots / rounds</th>{hasPct&&<th title={GLOSS.percentile}>Percentile vs PGA</th>}</tr></thead><tbody>{metrics.map(m=><MetricRow key={m.key} metric={m} showPct={hasPct}/>)}</tbody></table></div>{filtered?<p className="pd-muted">Shot and round counts follow the rounds you selected. Measures overlap, so the counts do not add to a total.</p>:<p className="pd-muted">Based on {deep?.shots?.coverage?.eligible_shots!=null?Number(deep.shots.coverage.eligible_shots).toLocaleString():"the available"} shots over {String(deep?.shots?.coverage?.complete_rounds??"the available")} rounds{deep?.shots?.coverage?.last_date?`, through ${date(String(deep.shots.coverage.last_date))}`:""}.</p>}<p className="pd-muted">Percentiles only count PGA players with at least 100 shots and 10 rounds on that measure.</p></details>
    </>}
  </div>;
}

function MetricRow({metric:m,showPct}:{metric:ShotMetric;showPct:boolean}){const small=(m.n_shots??0)<100||(m.n_rounds??0)<10;return <tr><th scope="row"><details><summary>{m.label}</summary><p>{m.definition}</p>{m.last_date&&<small>Last shot {date(m.last_date)}</small>}</details></th><td>{formatMetric(m.value,m.unit)}<small>{unitText(m.unit)}{m.status!=="available"?" · not enough data":""}</small></td><td>{m.n_shots??0} / {m.n_rounds??0}</td>{showPct&&<td>{m.status==="available"&&finite(m.reference_percentile)?`${Math.round(m.reference_percentile*100)}th${small?" · small sample":""}`:"Too few shots"}</td>}</tr>;}

function BaselineChart({deep}:{deep:DeepProfile}){
  const b=deep.baseline,series=(b?.series??[]).filter(p=>finite(p.value)&&Number.isFinite(Date.parse(p.date)));
  if(!series.length)return <p className="pd-empty">No long-run skill history is published for this player.</p>;
  const times=series.map(p=>Date.parse(p.date)),left=Math.min(...times),right=Math.max(...times);
  const values=series.flatMap(p=>[p.value,...(finite(p.moving_average_90d)?[p.moving_average_90d]:[])]);
  const low=Math.min(-.25,...values)-.2,high=Math.max(.25,...values)+.2;
  const x=(t:number)=>52+(t-left)/Math.max(1,right-left)*680,y=(v:number)=>185-(v-low)/(high-low)*145;
  const line=(key:"value"|"moving_average_90d")=>series.filter(p=>finite(p[key])).map(p=>`${x(Date.parse(p.date))},${y(p[key]!)}`).join(" ");
  return <><div className="pd-baseline-head"><div><span className="eyebrow">Long-run skill (older model)</span><strong>{signed(b?.value)} <small>strokes gained per round vs the PGA average</small></strong></div><span className="pd-tag">Updated {date(b?.latest_date)}</span></div>
    <svg className="pd-line" viewBox="0 0 780 225" role="img" aria-label="Long-run skill estimate over time from the older model, with a 90-day average.">
      {[low,(low+high)/2,high].map(v=><g key={v}><line x1="52" y1={y(v)} x2="732" y2={y(v)} stroke="var(--line)"/><text x="42" y={y(v)+4} textAnchor="end" fill="var(--muted)" fontSize="11">{signed(v)}</text></g>)}
      <line x1="52" x2="732" y1={y(0)} y2={y(0)} stroke="var(--muted)" strokeDasharray="4 4"/>
      <polyline points={line("value")} fill="none" stroke="var(--chart-1)" strokeOpacity=".45" strokeWidth="1.5"/>
      <polyline points={line("moving_average_90d")} fill="none" stroke="var(--chart-2)" strokeWidth="2.5"/>
      {series.length===1&&<circle cx={x(times[0])} cy={y(series[0].value)} r="4" fill="var(--chart-1)"/>}
      <text x="52" y="215" fill="var(--muted)" fontSize="11">{date(series[0].date)}</text><text x="732" y="215" textAnchor="end" fill="var(--muted)" fontSize="11">{date(series[series.length-1].date)}</text>
    </svg><div className="pd-legend"><span><i style={{background:"var(--chart-1)",opacity:.5}}/>Monthly estimate</span><span><i style={{background:"var(--chart-2)"}}/>90-day average</span></div><p className="pd-muted">A smoothed long-run skill line from our older model, rebuilt from revised history with no course adjustment. It is not the model that sets today’s prices.</p><details className="pd-method"><summary>Technical details</summary>{b?.limitations?.map((s,i)=><p key={i}>{s}</p>)}<p>Points are the last saved daily estimate each month; the 90-day line averages the saved daily values rather than the monthly points. Reference: {b?.reference_id??"unavailable"}.</p></details></>;
}

export function ProfileDeepDive({deep,profile,comparisons=[],filters=HISTORY_DEFAULTS,referenceKey}:{deep:DeepProfile|null;profile:PlayerProfile;comparisons?:PlayerProfile[];filters?:HistoryFilters;referenceKey?:string}){
  if(!deep)return <Panel title="Deep history"><p className="pd-empty">No detailed history has been published yet for {profile.identity.name}.</p></Panel>;
  return <div className="pd-deep"><Panel title="Performance over time" eyebrow="Long-term form · not affected by the filters below"><PerformanceTrend deep={deep} anchor={profile.pga_benchmark_reference?.adjusted_sg_mean}/><details className="pd-method"><summary>Older skill model (optional)</summary><p>A separate, smoothed estimate. The main chart above follows actual results.</p><BaselineChart deep={deep}/></details></Panel><PlayerHistoryExplorer key={JSON.stringify(filters)} deep={deep} profile={profile} comparisons={comparisons} filters={filters} referenceKey={referenceKey}/></div>;
}
