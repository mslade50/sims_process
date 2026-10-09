"use client";

import { useMemo, useState } from "react";
import { useParams, BenchmarkCard, BenchmarkMethod, type WeeklyEvent } from "./PlayerProfilesView";
import { DataTable, EmptyState, LoadingState, PageIntro, Panel } from "./components";
import { useDashboardData } from "./data";
import { explainKeyFor, orderEvents, parseExplain, pct, type ExplainDoc, type ExPlayer } from "./explain-rules";
import { savedFieldSkill, checkpointBenchmarkValue, shortProfileDate, selectWeeklyEvent, matchingInputRun, matchesExplainCheckpoint, matchesInputCheckpoint, savedSkill, parseCatalog, profileHref, signedZ, type SavedCheckpoint } from "./player-profile-rules";
import { checkpoints } from "./distributions-rules";
import { humanCheckpoint } from "./player-deep-rules";
import { GAP_METHOD_SENTENCE, LABELS, ZERO_SENTENCE, plainName } from "./labels";
import { ACCURACY_SENTENCE, COL, buildTable, expandEntries, fieldMethod, parseInputs, reconciliationHeaders, residualText, thisWeekGuide, type WeeklyDeps, type WeeklyInputs } from "./weekly-rules";
import "./player-profiles.css";
import "./weekly.css";

const DEPS: WeeklyDeps = { LABELS, plainName, signedZ, pct, savedFieldSkill, checkpointBenchmarkValue, savedSkill };

export function WeeklyPlayersView() {
  const load=useDashboardData<{events?:WeeklyEvent[]}>("golfprice/index.json"); const fields=useDashboardData<{schema_version?:string;fields?:Array<{event_uid:string;player_ids:number[]}>}>("golfprice/player_profiles/fields.json"); const events=orderEvents(load.data?.events ?? []) as WeeklyEvent[];const {params,update}=useParams();const event=selectWeeklyEvent(events,params.get("e"),params.get("player"),fields.data?.schema_version==="player_profiles.fields.v1" ? fields.data.fields : []);
  if(load.loading || (params.get("player") && !params.get("e") && fields.loading)) return <LoadingState label="Loading weekly player events"/>;
  return <div className="pp-page"><PageIntro eyebrow="Week" title="Weekly players" description="The saved model checkpoint: player strengths, adjustments and event probabilities, with links to historical profiles." controls={event && <label className="pp-event" id="weekly-event-picker">Saved event<select value={event.event_uid} onChange={e=>update({e:e.target.value})}>{events.map(e=><option key={e.event_uid} value={e.event_uid}>{e.name ?? e.event_uid} · {e.date_start ?? "date unavailable"}</option>)}</select></label>}/>{!event ? <EmptyState title="No saved weekly runs" detail={load.error ?? "Publish a validated golfprice run to inspect exact player components."}/> : <WeeklyBody key={event.event_uid} event={event as WeeklyEvent} playerId={params.get("player")}/>}</div>;
}
function WeeklyBody({event,playerId}:{event:WeeklyEvent;playerId:string|null}) {
  const choices=checkpoints(event); const [run,setRun]=useState(""); const checkpoint=choices.find(r=>r.run===run); const key=checkpoint?.key ?? explainKeyFor(event); const expected=matchingInputRun(event,checkpoint);
  return <><label className="pp-event">Saved checkpoint<select value={run} onChange={e=>setRun(e.target.value)}><option value="">Current saved checkpoint</option>{choices.map(r=><option key={`${r.kind}_${r.run}`} value={r.run}>{humanCheckpoint(r.as_of,r.kind)}</option>)}</select></label><WeeklyExplain key={key} event={event} dataKey={key} inputsKey={expected?.key ?? null} expected={expected} playerId={playerId}/><SavedInputs key={run || "current"} dataKey={expected?.key ?? null} expected={expected} playerId={playerId} explainKey={key}/></>;
}
function WeeklyExplain({event,dataKey,inputsKey,expected,playerId}:{event:WeeklyEvent;dataKey:string;inputsKey:string|null;expected:SavedCheckpoint|null;playerId:string|null}) {
  const catalogLoad=useDashboardData<unknown>("golfprice/player_profiles/dossier-review.json");const catalog=parseCatalog(catalogLoad.data);
  const load=useDashboardData<unknown>(dataKey);const doc=useMemo(()=>parseExplain(load.data),[load.data]);const inputsLoad=useDashboardData<unknown>(inputsKey ?? "golfprice/none.json");const inputs=useMemo(()=>inputsKey && matchesInputCheckpoint(inputsLoad.data,expected) ? parseInputs(inputsLoad.data) : null,[inputsKey,inputsLoad.data,expected]);
  if(load.loading) return <LoadingState label="Loading the saved player checkpoint"/>;
  if(load.error || !doc || !matchesExplainCheckpoint(doc,expected)) return <EmptyState title="Saved checkpoint unavailable" detail={load.error ?? "No matching explain document was published for this event."}/>;
  if(playerId && !doc.players.some(p=>String(p.id)===playerId)) return <EmptyState title="Player is not in this saved field" detail="Choose another event above or remove the player filter to inspect this event’s full field."/>;
  const historicalCoverage=doc.players.filter(p=>checkpointBenchmarkValue(catalog?.players.find(q=>q.dg_id===p.id)?.pga_benchmark,catalog?.pga_benchmark_reference,doc.as_of)!==null).length;
  const players=doc.players.filter(p=>!playerId || String(p.id)===playerId);const keys=[...new Set(doc.players.flatMap(p=>Object.keys(p.components)))];
  return <><div className="pp-run"><span className="eyebrow">{doc.kind==="live" ? `In play · after round ${doc.after_round}` : "Pre tournament"}</span><strong>{doc.event.name}</strong><span>{humanCheckpoint(doc.as_of,doc.kind)} · {doc.players.length} players</span></div><Panel title="Relative to PGA benchmark" eyebrow="Recent adjusted performance · not a simulation forecast"><p className="pp-muted">Current published historical ratings (catalog updated {shortProfileDate(catalog?.as_of)}), alongside {humanCheckpoint(doc.as_of,doc.kind)}; these do not replace the saved event model. Current ratings are recomputed at the catalog date from the latest revised history; they are not historical forecasts or as-was values. The observation-date screen only excludes later rounds. Positive means better performance (fewer strokes) than the fixed PGA reference.</p><p className="pp-muted">Supported recent historical performance: {historicalCoverage}/{doc.players.length} players in this saved field. Missing or later-history ratings stay blank. The secondary comparison uses the separately labeled saved event model.</p>{playerId && players.map(p=>{const rating=catalog?.players.find(q=>q.dg_id===p.id)?.pga_benchmark;return <BenchmarkCard key={p.id} rating={rating} reference={catalog?.pga_benchmark_reference} fieldLabel={doc.event.name} checkpoint={doc.as_of} savedDoc={doc} playerId={p.id}/>;})}<BenchmarkMethod reference={catalog?.pga_benchmark_reference}/></Panel><Panel title="Saved event components and probabilities" eyebrow="Field table · sort, pick columns, download CSV">{playerId && <a href={`/weekly-players?e=${encodeURIComponent(event.event_uid)}`}>Show the full field</a>}<WeeklyTable doc={doc} inputs={inputs} inputsPending={!!inputsKey && inputsLoad.loading} catalog={catalog} players={players}/>{!players.length && <p>No players match this selection in the saved event.</p>}<details className="pp-details"><summary>What each saved component means</summary>{keys.map(key=><p key={key}><b>{plainName(key,doc.components_legend)}:</b> {doc.components_legend[key]?.meaning ?? "No description was recorded."}</p>)}</details></Panel></>;
}

type Catalog=ReturnType<typeof parseCatalog>;
function WeeklyTable({doc,inputs,inputsPending,catalog,players}:{doc:ExplainDoc;inputs:WeeklyInputs;inputsPending:boolean;catalog:Catalog;players:ExPlayer[]}) {
  const built=useMemo(()=>buildTable({doc,inputs,catalog,players,deps:DEPS}),[doc,inputs,catalog,players]);
  const [open,setOpen]=useState<number|null>(null);
  const method=fieldMethod(inputs,doc.kind);const picked=built.rows.find(r=>(r.__row as {id:number}).id===open) ?? null;
  const headerTitles={[LABELS.vsPgaAvg.short]:GAP_METHOD_SENTENCE,[COL.gap]:GAP_METHOD_SENTENCE};
  const mobile=[COL.player,LABELS.thisWeekPga.short,COL.win];
  return <div className="wk-table">
    <details className="pp-details wk-method"><summary>Method</summary>
      <p>{ZERO_SENTENCE}</p>
      <p>{GAP_METHOD_SENTENCE}</p>
      <dl className="wk-method-list"><dt>{doc.kind==="live" ? "Pre-event field offset" : "Field offset"}</dt><dd>{method.offsetText} SG/round (field average {method.fieldAverage}{doc.kind==="live" ? "; live checkpoints keep the pre-event offset, not a recompute for the active field" : ""})</dd><dt>Estimator</dt><dd>{method.estimator}</dd><dt>Skills vintage</dt><dd>{method.vintage}</dd><dt>Debutants imputed</dt><dd>{method.imputed}{method.nField!=="n/a" ? ` of ${method.nField} players` : ""}</dd><dt>Accuracy</dt><dd>{ACCURACY_SENTENCE}</dd></dl>
      <p>{thisWeekGuide(method,DEPS)}</p>
    </details>
    {inputsPending && <p className="pp-muted" role="status">Loading the saved model inputs for the {LABELS.thisWeekPga.short} column.</p>}
    {!inputs && !inputsPending && <p className="pp-muted" role="status">{LABELS.thisWeekPga.short} is n/a: no matching saved model-inputs document is published for this checkpoint.</p>}
    <DataTable rows={built.rows} preferredColumns={built.ordered} label="Weekly players" pageSize={40} onRowClick={row=>setOpen((row.__row as {id:number}).id)} activeRow={picked} defaultColumns={built.defaults} headerTitles={headerTitles} mobileColumns={mobile} verbatim stickyFirst/>
    {picked && <aside className="wk-expand" aria-label={`Details for ${String(picked[COL.player])}`}><header><strong>{String(picked[COL.player])}</strong><a className="pp-link" href={profileHref((picked.__row as {id:number}).id)}>Profile</a><button type="button" onClick={()=>setOpen(null)}>Close</button></header><dl>{expandEntries(picked,DEPS).map(([k,v])=><div key={k}><dt>{k}</dt><dd>{v}</dd></div>)}</dl></aside>}
    <details className="pp-details"><summary>Column guide</summary>
      <p><b>{LABELS.thisWeekPga.short}:</b> {LABELS.thisWeekPga.long}.</p><p><b>{LABELS.vsField.short}:</b> {LABELS.vsField.long}.</p><p><b>{LABELS.vsPgaAvg.short} (column picker):</b> {LABELS.vsPgaAvg.long}. Shows the n&lt;12 rounds badge when the player has too little history, and is blank when the history is newer than this checkpoint.</p><p><b>Model minus form gap (column picker):</b> This week minus vs PGA avg; read it with the sentence in Method above (also the hover text on this column in the picker).</p><p><b>n (tour mix):</b> Rounds behind the PGA-average rating; a badge shows only when the primary tour is not the event tour or under half the weight is PGA.</p><p>Missing cells read n/a. Saved components, CHL families, location, override and weather by round are in the column picker. Components are field-relative SG per round and do not sum to the model skill columns.</p>
    </details>
  </div>;
}
function SavedInputs({dataKey,expected,playerId,explainKey}:{dataKey:string|null;expected:SavedCheckpoint|null;playerId:string|null;explainKey:string}) {
  if(!dataKey) return <p className="pp-muted">No matching saved model-inputs document is available for this checkpoint.</p>;
  return <SavedInputsBody key={dataKey} dataKey={dataKey} expected={expected} playerId={playerId} explainKey={explainKey}/>;
}
function SavedInputsBody({dataKey,expected,playerId,explainKey}:{dataKey:string;expected:SavedCheckpoint|null;playerId:string|null;explainKey:string}) {
  const load=useDashboardData<{players?:Array<{dg_id:number;name:string;challenger?:{breakdown?:{location?:{total?:number};override?:{total?:number}}}}>;as_of?:string;run?:string}>(dataKey);
  const explainLoad=useDashboardData<unknown>(explainKey);const legend=useMemo(()=>parseExplain(explainLoad.data)?.components_legend,[explainLoad.data]);
  if(load.loading) return <LoadingState label="Loading exact saved skill reconciliation"/>;
  if(load.error || !Array.isArray(load.data?.players) || !matchesInputCheckpoint(load.data,expected)) return <p role="status" className="pp-muted">Saved skill reconciliation unavailable: {load.error ?? "input identity, run or timestamp does not match this checkpoint"}.</p>;
  const players=load.data.players.filter(p=>!playerId || String(p.dg_id)===playerId);
  if(playerId && !players.length) return <Panel title="Player absent from this saved field"><p className="pp-muted">This checkpoint has no input row for player ID {playerId}. Choose another event above, or <a href={`/weekly-players?e=${encodeURIComponent(expected?.event_uid ?? "")}`}>show the full field</a>.</p></Panel>;
  const keys=[...new Set(players.flatMap(p=>Object.keys(savedSkill(p).families)))];
  return <details className="pp-details wk-audit"><summary>Saved baseline reconciliation (audit)</summary><Panel title="Saved event model skill decomposition" eyebrow="Saved event-centered baseline · model SG per round"><p className="pp-muted">CHL base is the saved event-centered pre-location estimate, separate from the portable PGA benchmark rating. Families describe the base; they are not added a second time. Location and overrides complete the baseline skill reconciliation. For live checkpoints this table retains the pre-event baseline; updated live skill appears below. Positive residual means baseline skill exceeds the saved component sum.</p><div className="pp-table-scroll"><table className="pp-table"><caption>Saved baseline components. Exact source is available below.</caption><thead><tr><th scope="col">Player</th><th scope="col">CHL base</th>{reconciliationHeaders(keys,legend,plainName).map(h=><th scope="col" key={h.key}>{h.label}<small>CHL family · SG/round</small></th>)}<th scope="col">Location</th><th scope="col">Override</th><th scope="col">Component sum</th><th scope="col">Baseline skill</th><th scope="col">Residual</th><th scope="col">Weather by round</th></tr></thead><tbody>{players.map(p=>{const s=savedSkill(p);return <tr key={p.dg_id}><th scope="row"><a href={profileHref(p.dg_id)}>{p.name}</a></th><td>{signedZ(s.base,3)}</td>{keys.map(k=><td key={k}>{signedZ(s.families[k],3)}</td>)}<td>{signedZ(p.challenger?.breakdown?.location?.total,3)}</td><td>{signedZ(p.challenger?.breakdown?.override?.total,3)}</td><td>{signedZ(s.sum,3)}</td><td>{signedZ(s.final,3)}</td><td>{residualText(s.residual,signedZ)}</td><td>{Object.entries(s.weather).map(([r,v])=><small key={r}>{r}: {signedZ(v,3)} SG offset</small>)}{!Object.keys(s.weather).length && "Unavailable"}</td></tr>;})}</tbody></table></div><details className="pp-details"><summary>Technical source</summary><p>{dataKey}</p></details><p className="pp-muted">Weather is separate: these saved offsets increase SG when positive and are excluded from the baseline skill sum. In-play deltas and current event probabilities appear in the checkpoint above.</p></Panel></details>;
}
