"use client";

import { Panel } from "./components";
import { finite, type Archetype, type PlayerProfile } from "./player-profile-rules";
import type { DeepProfile } from "./player-deep-rules";
import { grouped, intervalText, markerPosition, narrative, ranked, styleOf, type PlayerStyle, type StyleDescriptor } from "./player-style-rules";
import "./player-style.css";

type TraitKey="volatility"|"relative_hot_round_upside"|"tail_balance";
const spansZero=(m:{interval_90?:number[]|null}|null|undefined)=>!!m?.interval_90 && m.interval_90.length===2 && m.interval_90[0]<=0 && m.interval_90[1]>=0;
/** Plain-English reading of one playing-style trait. Numbers, intervals and sample sizes go to the hover text. */
export function traitWords(key:TraitKey,m:Archetype["volatility"]|undefined):string{
  if(!m || !finite(m.value)) return "Not enough rounds to say";
  const pc=finite(m.percentile)?m.percentile:null, tilt=pc===null?"mid":pc>=75?"high":pc<=25?"low":"mid";
  if(key==="volatility") return tilt==="low"?"Very consistent: scores bunch close together":tilt==="high"?"Streaky: scores swing a lot from round to round":"About as consistent as a typical player";
  if(key==="relative_hot_round_upside") return tilt==="high"?"Can run much hotter than usual":tilt==="low"?"Rarely runs much hotter than usual":"Average hot-round upside";
  if(m.band_uncertain || spansZero(m)) return "Too close to call between big upside and big downside rounds";
  return tilt==="high"?"More big upside rounds than bad ones":tilt==="low"?"More big downside rounds than good ones":"Balanced between upside and downside";
}
function LegacyStyle({archetype}:{archetype:PlayerProfile["archetype"]}) {
  const a:Archetype|null=archetype && typeof archetype==="object" ? archetype : null;
  const traits=[{key:"volatility" as const,label:"Consistency",meaning:"How much this player's opening-round scores vary."},{key:"relative_hot_round_upside" as const,label:"Hot-round upside",meaning:"How far above usual the good rounds reach."},{key:"tail_balance" as const,label:"Upside vs downside",meaning:"Whether the extreme rounds lean good or bad."}];
  const shown=traits.filter(t=>finite(a?.[t.key]?.value));
  return <Panel title="Playing style" eyebrow="Descriptive only · not a forecast"><p>{a?.label ?? (typeof archetype==="string" ? archetype : "No playing style has been published.")}</p><p className="pp-muted">Based on {a?.sample?.rounds ?? 0} opening-round results in {a?.sample?.events ?? 0} events.</p>{shown.length>0 ? <div className="pp-trait-grid">{shown.map(t=>{const m=a?.[t.key];return <article key={t.key} title={`${t.meaning} Value ${finite(m?.value)?m.value.toFixed(3):"n/a"} ${m?.unit ?? ""}${m?.interval_90?.length===2?`; 90% range ${m.interval_90.map(v=>v.toFixed(3)).join(" to ")}`:""}; compared with ${m?.reference_players ?? 0} players.`}><span className="eyebrow">{t.label}</span><strong>{traitWords(t.key,m)}</strong>{finite(m?.percentile) && <p className="pp-muted">Ranks above {Math.round(m.percentile)}% of {m?.reference_players ?? "the"} comparable players on this measure</p>}</article>;})}</div> : <p className="pp-muted">Not enough opening-round history to describe this player&apos;s style.</p>}<p className="pp-muted">Descriptive only; not a forecast.</p><details className="pp-details"><summary>Technical details</summary><p>{a?.caveat ?? "Historical traits are not validated player variance or event forecasts."}</p><p>{a?.reason ?? "No trait method or supported history was published."}</p><p>Labels use the bottom quarter (low) and top quarter (high) of comparable players. Consistency needs 30 usable rounds in 10 events; the other two need 60 rounds in 15 events. Ranges are 90% intervals from resampling events; they are not forecasts.</p><p>Reliability: {a?.reliability?.status ?? "unavailable"}. {a?.reliability?.reason ?? "Verified cut-maker evidence is unavailable."}</p>{a?.notes?.map((n,i)=><p key={i}>{n}</p>)}</details></Panel>;
}

const BADGE={headline:"Standout",notable:"Leaning",typical:"Typical"} as const;
function Card({d}:{d:StyleDescriptor}){
  const iv=intervalText(d);
  return <article className={`ps-card ps-${d.level}`} title={`${d.title}. Value ${d.display??d.value.toFixed(2)}${iv?`; 90% range ${iv}`:""}; ${d.n_events??"?"} events compared with ${d.reference_players??"?"} PGA regulars; reliability ${typeof d.reliability==="number"?d.reliability.toFixed(2):"n/a"}.`}>
    <header><span className="eyebrow">{d.title}</span><em>{d.low_signal?"Low signal":BADGE[d.level]}</em></header>
    <strong>{d.label}</strong>
    <p>{d.sentence}</p>
    <div className="ps-strip" role="img" aria-label={`Between ${Math.round(d.percentile)}% of PGA regulars below and ${100-Math.round(d.percentile)}% above`}><i style={{left:`${markerPosition(d)}%`}}/></div>
    <div className="ps-axis"><span>less</span><span>PGA regulars</span><span>more</span></div>
    {d.low_signal&&<small className="pp-muted">This pattern is not consistent enough between a player&apos;s own seasons to call a trait.</small>}
  </article>;
}
function RichStyle({style,name,archetype}:{style:PlayerStyle;name?:string;archetype:Archetype|null}){
  const top=ranked(style.descriptors).filter(d=>d.level!=="typical"&&!d.low_signal),rest=style.descriptors.filter(d=>!top.includes(d));
  return <Panel title="Playing style" eyebrow="Descriptive only · not a forecast">
    <p className="ps-story">{narrative(style,name)}</p>
    {top.length>0?<div className="ps-grid">{top.map(d=><Card key={d.key} d={d}/>)}</div>:<p className="pp-muted">No single pattern clears the bar for a standout. The full set is below.</p>}
    {rest.length>0&&<details className="pp-details"><summary>All {style.descriptors.length} patterns we measure ({rest.length} typical or low-signal)</summary>{grouped({...style,descriptors:rest}).map(g=><section key={g.group}><h4 className="ps-group">{g.group}</h4><div className="ps-grid">{g.items.map(d=><Card key={d.key} d={d}/>)}</div></section>)}</details>}
    <details className="pp-details"><summary>How this is measured</summary>
      <p>Every PGA Tour round from the last three years is used, rounds 1 to 4. Each round is turned into strokes above or below the player&apos;s own usual level, after removing the course-round field (so a hard day for everyone does not count against him). Weekend rounds only exist for players who made the cut, so patterns that compare weekend with opening rounds are shown relative to what a typical PGA regular does, not to zero.</p>
      <p>&ldquo;PGA regulars&rdquo; means players with at least 60 PGA Tour rounds in that window. A percentile such as 83% means the player scores higher on that measure than 83% of those regulars. Values are pulled part of the way toward the typical player when the sample is small, and each has a 90% range from resampling whole tournaments.</p>
      <p>A pattern is a Standout only when its range clears &ldquo;typical&rdquo; and it sits in the top or bottom 15% of regulars, and is Leaning when the evidence is weaker. Patterns that are not consistent from one half of a player&apos;s events to the other (across all regulars) are labelled Low signal and never highlighted. Wind is not included because weather is not in the published rounds table.</p>
      <p>Descriptive history, not a forecast. {archetype?.caveat??""}</p>
    </details></Panel>;
}
export function PlayingStyle({archetype,name}:{archetype:PlayerProfile["archetype"];deep?:DeepProfile|null;name?:string}){
  const a:Archetype|null=archetype && typeof archetype==="object" ? archetype : null, style=styleOf(a);
  return style ? <RichStyle style={style} name={name} archetype={a}/> : <LegacyStyle archetype={archetype}/>;
}
