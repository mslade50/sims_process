"use client";

import { useMemo, useState } from "react";
import { useParams, BenchmarkCard, BenchmarkMethod, type WeeklyEvent } from "./PlayerProfilesView";
import { DataTable, EmptyState, LoadingState, PageIntro, Panel } from "./components";
import { useDashboardData } from "./data";
import { explainKeyFor, orderEvents, parseExplain, pct, type ExplainDoc, type ExPlayer } from "./explain-rules";
import { savedFieldSkill, checkpointBenchmarkValue, shortProfileDate, selectWeeklyEvent, matchingInputRun, matchesExplainCheckpoint, matchesInputCheckpoint, savedSkill, parseCatalog, profileHref, signedZ, type SavedCheckpoint } from "./player-profile-rules";
import { checkpoints } from "./distributions-rules";
import { humanCheckpoint } from "./player-deep-rules";
import { GAP_METHOD_SENTENCE, LABELS, ZERO_SENTENCE, plainName, plainPhrase } from "./labels";
import { COL, activeWeatherRounds, buildTable, nameOf, expandEntries, fieldMethod, parseInputs, reconciliationHeaders, residualFlag, residualText, thisWeekGuide, type WeeklyDeps, type WeeklyInputs } from "./weekly-rules";
import "./player-profiles.css";
import "./weekly.css";

const DEPS: WeeklyDeps = { LABELS, gapSentence: GAP_METHOD_SENTENCE, plainName, plainPhrase, signedZ, pct, savedFieldSkill, checkpointBenchmarkValue, savedSkill };

export function WeeklyPlayersView() {
  const load=useDashboardData<{events?:WeeklyEvent[]}>("golfprice/index.json"); const fields=useDashboardData<{schema_version?:string;fields?:Array<{event_uid:string;player_ids:number[]}>}>("golfprice/player_profiles/fields.json"); const events=orderEvents(load.data?.events ?? []) as WeeklyEvent[];const {params,update}=useParams();const event=selectWeeklyEvent(events,params.get("e"),params.get("player"),fields.data?.schema_version==="player_profiles.fields.v1" ? fields.data.fields : []);
  if(load.loading || (params.get("player") && !params.get("e") && fields.loading)) return <LoadingState label="Loading events"/>;
  return <div className="pp-page"><PageIntro eyebrow="Week" title="Weekly players" description="The model's view of the field for one saved run: each player's strength, the adjustments behind it, and event probabilities, with links to profiles." controls={event && <label className="pp-event" id="weekly-event-picker">Event<select value={event.event_uid} onChange={e=>update({e:e.target.value})}>{events.map(e=><option key={e.event_uid} value={e.event_uid}>{e.name ?? e.event_uid} · {e.date_start ? shortProfileDate(e.date_start) : "date unavailable"}</option>)}</select></label>}/>{!event ? <EmptyState title="No saved model runs" detail={load.error ?? "No model run has been published yet."}/> : <WeeklyBody key={event.event_uid} event={event as WeeklyEvent} playerId={params.get("player")}/>}</div>;
}
function WeeklyBody({event,playerId}:{event:WeeklyEvent;playerId:string|null}) {
  const choices=dedupeRuns(checkpoints(event)); const [run,setRun]=useState(""); const checkpoint=choices.find(r=>r.run===run); const key=checkpoint?.key ?? explainKeyFor(event); const expected=matchingInputRun(event,checkpoint);
  return <><label className="pp-event">Model run<select value={run} onChange={e=>setRun(e.target.value)}><option value="">Latest model run</option>{choices.map(r=><option key={`${r.kind}_${r.run}`} value={r.run}>{humanCheckpoint(r.as_of,r.kind)}</option>)}</select></label><WeeklyExplain key={key} event={event} dataKey={key} inputsKey={expected?.key ?? null} expected={expected} playerId={playerId}/><SavedInputs key={run || "current"} dataKey={expected?.key ?? null} expected={expected} playerId={playerId} explainKey={key}/></>;
}
/** Two runs of one kind within ten minutes are a rerun of the same moment: keep the later one. */
function dedupeRuns<T extends {kind:string;as_of:string|null}>(runs:T[]):T[] {
  const t=(r:T)=>Date.parse(r.as_of ?? "");
  const sorted=[...runs].sort((a,b)=>(t(a)||0)-(t(b)||0));
  return sorted.filter((r,i)=>{const n=sorted[i+1];return !(n && n.kind===r.kind && Number.isFinite(t(r)) && Number.isFinite(t(n)) && t(n)-t(r)<600000);}).sort((a,b)=>(t(b)||0)-(t(a)||0));
}
function WeeklyExplain({event,dataKey,inputsKey,expected,playerId}:{event:WeeklyEvent;dataKey:string;inputsKey:string|null;expected:SavedCheckpoint|null;playerId:string|null}) {
  const catalogLoad=useDashboardData<unknown>("golfprice/player_profiles/dossier-review.json");const catalog=parseCatalog(catalogLoad.data);
  const load=useDashboardData<unknown>(dataKey);const doc=useMemo(()=>parseExplain(load.data),[load.data]);const inputsLoad=useDashboardData<unknown>(inputsKey ?? "golfprice/none.json");const inputs=useMemo(()=>inputsKey && matchesInputCheckpoint(inputsLoad.data,expected) ? parseInputs(inputsLoad.data) : null,[inputsKey,inputsLoad.data,expected]);
  if(load.loading) return <LoadingState label="Loading the model run"/>;
  if(load.error || !doc || !matchesExplainCheckpoint(doc,expected)) return <EmptyState title="This model run is not available" detail={load.error ?? "No model run was published for this event and checkpoint."}/>;
  if(playerId && !doc.players.some(p=>String(p.id)===playerId)) return <EmptyState title="This player is not in this field" detail="Choose another event above, or remove the player filter to see the full field."/>;
  const historicalCoverage=doc.players.filter(p=>checkpointBenchmarkValue(catalog?.players.find(q=>q.dg_id===p.id)?.pga_benchmark,catalog?.pga_benchmark_reference,doc.as_of)!==null).length;
  const players=doc.players.filter(p=>!playerId || String(p.id)===playerId);
  return <><div className="pp-run"><span className="eyebrow">{doc.kind==="live" ? `In play · after round ${doc.after_round}` : "Pre tournament"}</span><strong>{doc.event.name}</strong><span>{humanCheckpoint(doc.as_of,doc.kind)} · {doc.players.length} players</span></div>{(!!playerId || historicalCoverage>=5) && <Panel title="Recent form vs the PGA average" eyebrow="How each player has been playing lately · not a price forecast"><p className="pp-muted">Form ratings as of {shortProfileDate(catalog?.as_of)}. Model run: {humanCheckpoint(doc.as_of,doc.kind)}. Rounds played after the model run are left out. {historicalCoverage} of {doc.players.length} players have enough rounds for a rating.</p>{playerId && players.map(p=>{const rating=catalog?.players.find(q=>q.dg_id===p.id)?.pga_benchmark;return <BenchmarkCard key={p.id} rating={rating} reference={catalog?.pga_benchmark_reference} fieldLabel={doc.event.name} checkpoint={doc.as_of} savedDoc={doc} playerId={p.id}/>;})}<BenchmarkMethod reference={catalog?.pga_benchmark_reference}/></Panel>}<Panel title="The field: skill and chances" eyebrow="Sort any column, choose columns, or download as CSV">{playerId && <a href={`/weekly-players?e=${encodeURIComponent(event.event_uid)}`}>Show the full field</a>}<WeeklyTable doc={doc} inputs={inputs} inputsPending={!!inputsKey && inputsLoad.loading} catalog={catalog} players={players}/>{!players.length && <p>No players match this selection in the saved event.</p>}</Panel></>;
}

type Catalog=ReturnType<typeof parseCatalog>;
function WeeklyTable({doc,inputs,inputsPending,catalog,players}:{doc:ExplainDoc;inputs:WeeklyInputs;inputsPending:boolean;catalog:Catalog;players:ExPlayer[]}) {
  const built=useMemo(()=>buildTable({doc,inputs,catalog,players,deps:DEPS}),[doc,inputs,catalog,players]);
  const [open,setOpen]=useState<number|null>(null);
  const method=fieldMethod(inputs,doc.kind);const picked=built.rows.find(r=>(r.__row as {id:number}).id===open) ?? null;
  const headerTitles={...built.titles,[LABELS.thisWeekPga.short]:thisWeekGuide(method,DEPS)};
  const mobile=[COL.player,LABELS.thisWeekPga.short,COL.win];
  return <div className="wk-table">
    <details className="pp-details wk-method"><summary>Method</summary>
      <p>{ZERO_SENTENCE} Recent form is judged on the opening two rounds.</p>
      <p>{GAP_METHOD_SENTENCE}</p>
      <p>{LABELS.thisWeekPga.short} is the model&apos;s saved skill plus this field&apos;s offset: {method.offsetText} strokes per round, meaning how this field compares with a normal PGA field{doc.kind==="live" ? " (kept from before the event during live runs)" : ""}.</p>
      <details className="pp-details wk-technical"><summary>Technical details</summary>
        <dl className="wk-method-list"><dt>Average of the field&apos;s ratings</dt><dd>{method.fieldAverage}</dd><dt>Ratings used</dt><dd>{method.estimator}</dd><dt>Ratings dated</dt><dd>{method.vintage}</dd><dt>First-time players estimated</dt><dd>{method.imputed}{method.nField!=="n/a" ? ` of ${method.nField} players` : ""}</dd></dl>
      </details>
    </details>
    {inputsPending && <p className="pp-muted" role="status">Loading the model inputs behind the {LABELS.thisWeekPga.short} column.</p>}
    {!inputs && !inputsPending && <p className="pp-muted" role="status">{LABELS.thisWeekPga.short} is blank because the model inputs for this checkpoint were not published.</p>}
    {built.notes.map(n=><p key={n} className="pp-muted" role="note">{n}</p>)}
    <DataTable rows={built.rows} preferredColumns={built.ordered} label="Weekly players" pageSize={40} onRowClick={row=>setOpen((row.__row as {id:number}).id)} activeRow={picked} defaultColumns={built.defaults} headerTitles={headerTitles} mobileColumns={mobile} verbatim stickyFirst/>
    {picked && <aside className="wk-expand" aria-label={`Details for ${String(picked[COL.player])}`}><header><strong>{String(picked[COL.player])}</strong><a className="pp-link" href={profileHref((picked.__row as {id:number}).id)}>Profile</a><button type="button" onClick={()=>setOpen(null)}>Close</button></header><dl>{expandEntries(picked,DEPS).map(([k,v])=><div key={k}><dt>{k}</dt><dd>{v}</dd></div>)}</dl></aside>}
    <details className="pp-details"><summary>Column guide</summary>
      {Object.entries(built.titles).filter(([k])=>built.ordered.includes(k) && !k.startsWith("Base skill: ")).map(([k,v])=><p key={k}><b>{k}:</b> {v}</p>)}
      {built.ordered.some(k=>k.startsWith("Base skill: ")) && <><p><b>Parts of base skill</b> (shown for reference, never added a second time):</p><ul>{Object.entries(built.titles).filter(([k])=>built.ordered.includes(k) && k.startsWith("Base skill: ")).map(([k,v])=><li key={k}><b>{k.replace("Base skill: ","")}:</b> {v}</li>)}</ul></>}
      <p>Hover over any column header for the same definition. A blank cell means that number does not exist for that player. Extra columns are in the column picker, in strokes per round against the field.</p>
    </details>
  </div>;
}
function SavedInputs({dataKey,expected,playerId,explainKey}:{dataKey:string|null;expected:SavedCheckpoint|null;playerId:string|null;explainKey:string}) {
  if(!dataKey) return <p className="pp-muted">The model inputs for this run were not published.</p>;
  return <SavedInputsBody key={dataKey} dataKey={dataKey} expected={expected} playerId={playerId} explainKey={explainKey}/>;
}
function SavedInputsBody({dataKey,expected,playerId,explainKey}:{dataKey:string;expected:SavedCheckpoint|null;playerId:string|null;explainKey:string}) {
  const load=useDashboardData<{players?:Array<{dg_id:number;name:string;challenger?:{breakdown?:{location?:{total?:number};override?:{total?:number}}}}>;as_of?:string;run?:string}>(dataKey);
  const explainLoad=useDashboardData<unknown>(explainKey);const legend=useMemo(()=>parseExplain(explainLoad.data)?.components_legend,[explainLoad.data]);
  if(load.loading) return <LoadingState label="Loading how the saved skill adds up"/>;
  if(load.error || !Array.isArray(load.data?.players) || !matchesInputCheckpoint(load.data,expected)) return <p role="status" className="pp-muted">The breakdown of saved skill is unavailable: {load.error ?? "the published inputs do not match this model run"}.</p>;
  const players=load.data.players.filter(p=>!playerId || String(p.dg_id)===playerId);
  if(playerId && !players.length) return <Panel title="Player not in this field"><p className="pp-muted">This model run has no inputs for this player. Choose another event above, or <a href={`/weekly-players?e=${encodeURIComponent(expected?.event_uid ?? "")}`}>show the full field</a>.</p></Panel>;
  const keys=[...new Set(players.flatMap(p=>Object.keys(savedSkill(p).families)))];
  const showResidual=players.some(p=>residualFlag(savedSkill(p).residual)),weatherRounds=activeWeatherRounds(players.map(p=>({weather:savedSkill(p).weather}))),showWeather=weatherRounds.length>0,showSum=players.some(p=>savedSkill(p).sum!==savedSkill(p).final);
  return <details className="pp-details wk-audit"><summary>How the saved skill adds up</summary><Panel title="How each player's saved skill adds up" eyebrow="Strokes per round against the field"><p className="pp-muted">Base skill is the model&apos;s estimate before course location and owner adjustments; its parts describe the base and are not added a second time. For live runs this table keeps the pre-event numbers.</p><div className="pp-table-scroll"><table className="pp-table"><caption>Saved skill by part</caption><thead><tr><th scope="col">Player</th><th scope="col" title="The model's skill estimate before course-location and owner adjustments">Base skill</th>{reconciliationHeaders(keys,legend,plainName).map(h=><th scope="col" key={h.key}>{h.label}<small>part of base skill</small></th>)}<th scope="col" title="Adjustment for travel, home region and nationality">Course location</th><th scope="col" title="A one-off change set by the owner on the private site">Owner override</th>{showSum && <th scope="col" title="Base skill plus every adjustment">Sum of parts</th>}<th scope="col" title="The model's final saved skill, before weather and live updates">Saved skill</th>{showResidual && <th scope="col" title="How far saved skill differs from the sum of its parts; it should be zero">Unexplained</th>}{showWeather && <th scope="col" title="Effect of the forecast weather on skill in each round. Not included in the saved skill.">Weather (not included)</th>}</tr></thead><tbody>{players.map(p=>{const s=savedSkill(p);return <tr key={p.dg_id}><th scope="row"><a href={profileHref(p.dg_id)}>{nameOf(p)}</a></th><td>{signedZ(s.base,3)}</td>{keys.map(k=><td key={k}>{signedZ(s.families[k],3)}</td>)}<td>{signedZ(p.challenger?.breakdown?.location?.total,3)}</td><td>{signedZ(p.challenger?.breakdown?.override?.total,3)}</td>{showSum && <td>{signedZ(s.sum,3)}</td>}<td>{signedZ(s.final,3)}</td>{showResidual && <td>{residualText(s.residual,signedZ)}</td>}{showWeather && <td>{weatherRounds.map(r=>[r,s.weather[r]] as const).map(([r,v])=><small key={r}>{r}: {signedZ(v,3)}</small>)}</td>}</tr>;})}</tbody></table></div><p className="pp-muted">Weather is separate and not part of the saved skill: positive helps the player. Live shifts and current probabilities are in the field table above.</p></Panel></details>;
}
