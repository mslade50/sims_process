"use client";

import {useState} from "react";
import {Panel} from "./components";
import type {PlayerProfile} from "./player-profile-rules";
import {exploratoryStandouts, finite, formatMetric, shapePoints, type DeepProfile, type ShotMetric} from "./player-deep-rules";
import {HISTORY_DEFAULTS, type HistoryFilters} from "./player-history-filters";
import {PlayerHistoryExplorer} from "./PlayerHistoryExplorer";
import "./player-deep.css";
import {PerformanceTrend} from "./PlayerPerformanceTrend";

const date=(v:string|undefined)=>v&&Number.isFinite(Date.parse(v))?new Date(v).toLocaleDateString("en-US",{month:"short",day:"numeric",year:"numeric",timeZone:"UTC"}):"Unavailable";
const signed=(v:unknown)=>finite(v)?`${v>0?"+":""}${v.toFixed(2)}`:"—";
const groups={
  "All-round detail":["tee_distance","tee_accuracy","tee_dispersion","app_great","app_150_200","arg_rough"],
  "Driving":["tee_distance","tee_accuracy","tee_dispersion","tee_sg","tee_great","tee_poor"],
  "Putting":["putt_0_5","putt_5_10","putt_10_20","putt_20_plus","putt_short_make","putt_lag_leave"],
  "Approach":["app_50_100","app_100_150","app_150_200","app_200_plus","app_great","prox_rough"],
  "Short game":["arg_sand","arg_rough","arg_fairway","prox_fairway","prox_rough","app_poor"],
};
const axisLabel:Record<string,string>={tee_sg:"Tee SG",tee_great:"Great tee",tee_poor:"Poor avoidance",putt_0_5:"<5 ft SG",putt_5_10:"5–10 ft SG",putt_10_20:"10–20 ft SG",putt_20_plus:"20+ ft SG",putt_short_make:"<5 ft makes",putt_lag_leave:"Lag leave",tee_distance:"Distance",tee_accuracy:"Fairways",tee_dispersion:"Dispersion",app_great:"Great APP",app_poor:"Poor avoidance",app_50_100:"50–100",app_100_150:"100–150",app_150_200:"150–200",app_200_plus:"200+",arg_sand:"Bunker",arg_rough:"Rough ARG",arg_fairway:"Fairway ARG",prox_fairway:"FW proximity",prox_rough:"Rough proximity"};

export function ShotDetailsShape({deep,filtered=false}:{deep:DeepProfile|null;filtered?:boolean}){
  const [group,setGroup]=useState<keyof typeof groups>("All-round detail");
  const metrics=deep?.shots?.metrics??[];
  const keys=groups[group];const shape=shapePoints(metrics,keys);
  const point=(i:number,r:number)=>{const a=(i*60-90)*Math.PI/180;return [190+Math.cos(a)*106*r,158+Math.sin(a)*106*r];};
  const polygon=(r:number)=>keys.map((_,i)=>point(i,r).join(",")).join(" ");
  const strong=exploratoryStandouts(metrics,keys);
  return <div className="pd-shot"><div className="pd-heading"><div><span className="eyebrow">The details behind the score</span><h3>Shot fingerprint</h3></div><label className="pd-select"><span className="sr-only">Shot fingerprint axes</span><select value={group} onChange={e=>setGroup(e.target.value as keyof typeof groups)}>{Object.keys(groups).map(k=><option key={k}>{k}</option>)}</select></label></div>
    {!metrics.length?<p className="pd-empty">Published shot detail is unavailable for this player. Feeder and amateur results can still appear in event history.</p>:<>
      <svg className="pd-fingerprint" viewBox="0 0 380 326" role="img" aria-label="Observed shot detail relative to supported PGA players. Higher is better on every axis. Missing values have no point; exact values below.">
        {[.25,.5,.75,1].map(r=><polygon key={r} points={polygon(r)} fill="none" stroke={r===.5?"var(--muted)":"var(--line)"} strokeDasharray={r===.5?"4 4":undefined}/>)}
        {keys.map((k,i)=>{const end=point(i,1),label=point(i,1.34);return <g key={k}><line x1="190" y1="158" x2={end[0]} y2={end[1]} stroke="var(--line)"/><text x={label[0]} y={label[1]} textAnchor="middle" fill="var(--muted-strong)" fontSize="11">{axisLabel[k]}</text></g>;})}
        {shape.complete&&<polygon points={shape.radii.map((r,i)=>point(i,r!).join(",")).join(" ")} fill="var(--chart-2)" fillOpacity=".12" stroke="var(--chart-2)" strokeWidth="2"/>}
        {shape.radii.map((r,i)=>r===null?null:<circle key={keys[i]} cx={point(i,r)[0]} cy={point(i,r)[1]} r="4" fill="var(--chart-2)"><title>{metrics.find(m=>m.key===keys[i])?.label}: {signed(metrics.find(m=>m.key===keys[i])?.reference_z)} Z; higher is better</title></circle>)}
        <text x="196" y="101" fill="var(--muted)" fontSize="10">PGA mean</text>
      </svg><div className="pd-shot-values">{keys.map(k=>{const m=metrics.find(v=>v.key===k);return <div key={k}><span>{k==="tee_poor"?"Poor tee rate":k==="app_poor"?"Poor APP rate":axisLabel[k]}</span><b>{formatMetric(m?.value,m?.unit??"")}</b><small>{m?.unit==="fraction"?"rate":m?.unit??"Unavailable"}{m?.higher_is_better===false?" · lower is better":""} · {m?.n_shots??0} shots · PGA n={m?.reference_n_players??"?"}{m?.status!=="available"||(m?.n_shots??0)<100||(m?.n_rounds??0)<10?" · provisional":""}</small></div>;})}</div><p className="pd-muted">{strong.length?<>Exploratory strengths among these axes: <b>{strong.map(m=>m.label.toLowerCase()).join(" · ")}</b>. </>:null}Each axis is standardized separately against its own supported PGA cohort; higher is better. The connected area is not a composite skill score. Opportunity mix remains. Dashed ring = each metric’s cohort mean. {shape.complete?"":"Partial sample: available points only. "}Display clips at ±2.5 standard deviations. Reference as of {date(metrics.find(m=>m.reference_as_of)?.reference_as_of)}; published trailing 3 years. Tee distance includes all clubs; dispersion is around the field landing axis, not the target. {group==="Driving"&&"Tee-shot outcomes exclude separate penalty/drop records; poor-shot rate is not all driving trouble."}{group==="Putting"&&"Putting buckets use feet. Short-putt make rate and lag leave distance retain the actual distance mix; they are not fixed-distance probabilities or three-putt rates."}</p>
      <details className="pd-method"><summary>Exact shot metrics · {metrics.filter(m=>m.status==="available").length}/{metrics.length} supported</summary><div className="pd-scroll"><table className="pd-table"><caption>Observed outcomes; metrics use different opportunities and do not add to total SG.</caption><thead><tr><th>Metric</th><th>Value</th><th>Shots / rounds</th><th>PGA rank</th></tr></thead><tbody>{metrics.map(m=><MetricRow key={m.key} metric={m}/>)}</tbody></table></div>{filtered?<p className="pd-muted">Shot counts and round counts in this table follow the selected slice. Metrics use overlapping opportunities; their counts do not add to a unique-shot total.</p>:<p className="pd-muted">{String(deep?.shots?.coverage?.eligible_shots??"Unknown")} eligible physical shots · {String(deep?.shots?.coverage?.complete_rounds??"unknown")} rounds with eligible shots · through {date(String(deep?.shots?.coverage?.last_date??""))}. Excluded records: {String(deep?.shots?.coverage?.excluded_shots??"unavailable")}. {String(deep?.shots?.coverage?.note??"")}</p>}<p className="pd-muted">Ranks compare descriptive metrics with PGA players having ≥100 shots and ≥10 rounds in the same metric. No rank is shown without a supported cohort. Course, starting lie and distance mix remain. Distance uses all clubs on par 4/5 tees. Dispersion is around the field landing axis, not the intended target. A measured driver-only record is unavailable.</p></details>
    </>}
  </div>;
}

function MetricRow({metric:m}:{metric:ShotMetric}){return <tr><th scope="row"><details><summary>{m.label}</summary><p>{m.definition}</p><small>Last observation {date(m.last_date)}</small></details></th><td>{formatMetric(m.value,m.unit)}<small>{m.unit==="fraction"?"rate":m.unit}{m.status!=="available"?` · ${m.status?.replaceAll("_"," ")??"unsupported"}`:""}</small></td><td>{m.n_shots??0} / {m.n_rounds??0}</td><td>{m.status==="available"&&finite(m.reference_percentile)?`${Math.round(m.reference_percentile*100)}th${(m.n_shots??0)<100||(m.n_rounds??0)<10?" · provisional":""}`:"—"}<small>{finite(m.reference_n_players)?`PGA n=${m.reference_n_players}`:"Cohort size unavailable"}</small></td></tr>;}

function BaselineChart({deep}:{deep:DeepProfile}){
  const b=deep.baseline,series=(b?.series??[]).filter(p=>finite(p.value)&&Number.isFinite(Date.parse(p.date)));
  if(!series.length)return <p className="pd-empty">A course-neutral legacy baseline history is unavailable for this player.</p>;
  const times=series.map(p=>Date.parse(p.date)),left=Math.min(...times),right=Math.max(...times);
  const values=series.flatMap(p=>[p.value,...(finite(p.moving_average_90d)?[p.moving_average_90d]:[])]);
  const low=Math.min(-.25,...values)-.2,high=Math.max(.25,...values)+.2;
  const x=(t:number)=>52+(t-left)/Math.max(1,right-left)*680,y=(v:number)=>185-(v-low)/(high-low)*145;
  const line=(key:"value"|"moving_average_90d")=>series.filter(p=>finite(p[key])).map(p=>`${x(Date.parse(p.date))},${y(p[key]!)}`).join(" ");
  return <><div className="pd-baseline-head"><div><span className="eyebrow">Course-neutral legacy model</span><strong>{signed(b?.value)} <small>SG/round vs fixed PGA reference</small></strong></div><span className="pd-tag">Latest {date(b?.latest_date)}</span></div>
    <svg className="pd-line" viewBox="0 0 780 225" role="img" aria-label="Legacy published course-neutral model skill over time with a trailing 90-day moving average. Reconstructed from revised history, not archived forecasts.">
      {[low,(low+high)/2,high].map(v=><g key={v}><line x1="52" y1={y(v)} x2="732" y2={y(v)} stroke="var(--line)"/><text x="42" y={y(v)+4} textAnchor="end" fill="var(--muted)" fontSize="11">{signed(v)}</text></g>)}
      <line x1="52" x2="732" y1={y(0)} y2={y(0)} stroke="var(--muted)" strokeDasharray="4 4"/>
      <polyline points={line("value")} fill="none" stroke="var(--chart-1)" strokeOpacity=".45" strokeWidth="1.5"/>
      <polyline points={line("moving_average_90d")} fill="none" stroke="var(--chart-2)" strokeWidth="2.5"/>
      {series.length===1&&<circle cx={x(times[0])} cy={y(series[0].value)} r="4" fill="var(--chart-1)"/>}
      <text x="52" y="215" fill="var(--muted)" fontSize="11">{date(series[0].date)}</text><text x="732" y="215" textAnchor="end" fill="var(--muted)" fontSize="11">{date(series[series.length-1].date)}</text>
    </svg><div className="pd-legend"><span><i style={{background:"var(--chart-1)",opacity:.5}}/>Saved legacy estimate</span><span><i style={{background:"var(--chart-2)"}}/>Trailing 90-day average</span></div><p className="pd-muted">Retrospective monthly observations from our revised legacy published skill estimate; latest observation {date(b?.latest_date)}. The curve is sampled monthly, while its 90-day average uses saved daily observations. No course-specific adjustment. The current golfprice CHL pricing model uses a different estimator. This reconstructed, revised history is not an immutable record of forecasts made on those dates.</p><details className="pd-method"><summary>Baseline provenance and moving average</summary>{b?.limitations?.map((s,i)=><p key={i}>{s}</p>)}<p>Points show the last saved daily estimate each month. The 90-day line averages saved daily observations, rather than averaging the displayed monthly points. Reference: {b?.reference_id??"unavailable"}.</p></details></>;
}

export function ProfileDeepDive({deep,profile,comparisons=[],filters=HISTORY_DEFAULTS,referenceKey}:{deep:DeepProfile|null;profile:PlayerProfile;comparisons?:PlayerProfile[];filters?:HistoryFilters;referenceKey?:string}){
  if(!deep)return <Panel title="Deep history"><p className="pd-empty">No matching deep profile has been published for {profile.identity.name}.</p></Panel>;
  return <div className="pd-deep"><Panel title="Performance over time" eyebrow="Long-term form · independent of the history filters"><PerformanceTrend deep={deep} anchor={profile.pga_benchmark_reference?.adjusted_sg_mean}/><details className="pd-method"><summary>Optional legacy model history</summary><p>This separate legacy estimate blends and smooths multiple inputs; the main chart follows actual adjusted performance.</p><BaselineChart deep={deep}/></details></Panel><PlayerHistoryExplorer key={JSON.stringify(filters)} deep={deep} profile={profile} comparisons={comparisons} filters={filters} referenceKey={referenceKey}/></div>;
}
