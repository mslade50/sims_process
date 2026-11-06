"use client";
/* eslint-disable jsx-a11y/no-noninteractive-tabindex -- Scrollable data tables need keyboard focus (axe scrollable-region-focusable). */
import { useMemo, useState } from "react";
import { Block } from "./score-block";
import { useDashboardData } from "./data";
import { bookName, historyRows, marketKey, matchingRecord, parseMarketScoring, signedScore, type MarketDocument, type MarketRecord } from "./market-scoring-rules";
import "./market-scoring.css";

const fmt = (v: number | null) => v == null ? "—" : v.toFixed(2);
const at = (v: string) => v ? `${v.replace("T", " ").replace(/(?:\.\d+)?Z$/, "")} UTC` : "Unknown";
export function MarketScoringPanel({ eventUid, modelRun, round }: { eventUid: string; modelRun: string; round: number | null }) {
  const source = useDashboardData<unknown>(marketKey(eventUid));
  const doc = useMemo(() => parseMarketScoring(source.data, eventUid), [source.data, eventUid]);
  return <Block id="market" title="What the books imply" eyebrow="Market versus our scoring forecast, in strokes" fold>
    {source.loading ? <p className="msc-note" role="status">Loading saved market comparisons…</p> : !doc ? <p className="msc-note">{source.errorKind === "missing" ? "No market comparison has been published for this event." : source.error ? "Market comparisons could not be loaded. Try refreshing the page." : "No valid market comparison is available for this event."}</p> : <MarketScoringContent key={`${eventUid}:${modelRun}:${round}`} doc={doc} modelRun={modelRun} round={round} />}
    {source.stale && <p className="msc-note" role="status">Showing a saved copy while the data service is unavailable.</p>}
  </Block>;
}

export function MarketScoringContent({ doc, modelRun, round }: { doc: MarketDocument; modelRun: string; round: number | null }) {
  const [picked, setPicked] = useState("");
  const matched = matchingRecord(doc.records, modelRun, round);
  const record = picked ? doc.records.find((r) => r.id === picked) ?? null : matched;
  const history = historyRows(doc.records);
  return <div className="msc">
    {!record ? <p className="msc-lead">No saved two-sided book comparison matches this simulation run and round. Choose an archived comparison below to inspect its own forecast.</p> : <Comparison record={record} different={record.model_run !== modelRun || record.round !== round} />}
    <label className="msc-select">Saved comparison<select aria-label="Saved market comparison" value={picked} onChange={(e) => setPicked(e.target.value)}><option value="">Selected simulation {matched ? "· matching comparison" : "· no comparison"}</option>{[...doc.records].reverse().map((r) => <option key={r.id} value={r.id}>R{r.round} · {at(r.capture_at)} · {r.origin === "reconstructed" ? "Reconstructed" : "Recorded"} · {r.model_run}</option>)}</select></label>
    <p className="msc-note">These are inferred field averages: each two-sided price is de-vigged, then translated into a shift using that player’s saved score distribution. The field estimate keeps our player differences. It is conditional on our shapes and ratings; disagreement alone does not identify a weather or course error. Books and players share information, so quote counts are not independent course observations.</p>
    {doc.records.some((r) => r.books.some((b) => b.freshness_unknown)) && <p className="msc-note"><b>Reconstruction limit:</b> archived timestamps record price changes. Continued availability and freshness of unchanged contracts are unknown.</p>}
    <details className="msc-details" open><summary>Round history · latest saved pre-tee comparison per book</summary>
      <div className="score-table-scroll" role="region" aria-label="Market comparison history" tabIndex={0}><table className="msc-table"><caption>Outcome errors use the same settled quoted players for our model and the book. Error = forecast minus actual; closer to zero is better.</caption><thead><tr><th>Round / book</th><th>Origin</th><th>Our field</th><th>Book field</th><th>Delta</th><th>Settlement</th><th>Actual quoted</th><th>Our error</th><th>Book error</th><th>Our / book Brier</th></tr></thead><tbody>{history.map(({ record: r, book: b }) => <tr key={`${r.round}:${b.book}`}><th scope="row">R{r.round} · {bookName(b.book)}<small>{at(r.capture_at)}<br />Model {r.model_run}</small></th><td>{r.origin === "reconstructed" ? "Reconstructed" : "Recorded"}</td><td>{fmt(b.model_field_mean)}</td><td>{fmt(b.market_field_mean)}</td><td>{signedScore(b.delta)}</td><td>{b.outcome.status} · {b.outcome.settled}/{b.quoted}</td><td>{fmt(b.outcome.actual_quoted_mean)}</td><td>{signedScore(b.outcome.model_error)}</td><td>{signedScore(b.outcome.market_error)}</td><td>{b.outcome.model_brier?.toFixed(3) ?? "—"} / {b.outcome.market_brier?.toFixed(3) ?? "—"}</td></tr>)}</tbody></table></div>
      {!history.length && <p className="msc-note">No valid pre-round comparisons are saved yet.</p>}
    </details>
    {doc.outcome_error && <p className="msc-note">Published outcomes are unavailable for this refresh. Pending rows have not been treated as zero errors.</p>}
    <details className="msc-details"><summary>All saved snapshots and method</summary><p className="msc-note">Every saved capture remains selectable above. The history table uses one latest pre-tee record per round/book, avoiding repeated scoring of the same outcome. Brier scores compare under probabilities at the original half-stroke lines; lower is better. Actual quoted averages are not full-field averages. Method: pmf-location-multiplicative-v1. Updated {at(doc.generated_at)}.</p></details>
  </div>;
}

function Comparison({ record: r, different }: { record: MarketRecord; different: boolean }) {
  const scale = Math.max(1, ...r.books.map((b) => Math.abs(b.delta)));
  return <>
    <p className="msc-lead">Round {r.round} · {r.origin === "reconstructed" ? "Reconstructed from archived prices" : "Recorded before first tee"}{different ? " · viewing a different forecast from the selected simulation" : " · paired with the selected simulation"}</p>
    <div className="score-table-scroll" role="region" aria-label="Book scoring comparison" tabIndex={0}><table className="msc-table msc-summary"><caption>Negative delta means easier scoring than our forecast. Positive means harder. All values are strokes.</caption><thead><tr><th>Book</th><th>Inferred field</th><th>Our same forecast</th><th>Delta</th><th>Quoted / field</th><th>Oldest price</th></tr></thead><tbody>{r.books.map((b) => <tr key={b.book}><th scope="row">{bookName(b.book)}</th><td data-label="Inferred field"><strong>{fmt(b.market_field_mean)}</strong></td><td data-label="Our forecast">{fmt(b.model_field_mean)}</td><td data-label="Delta">{signedScore(b.delta)}<small>{b.delta < 0 ? "Easier" : b.delta > 0 ? "Harder" : "Aligned"}</small></td><td data-label="Quoted / field">{b.quoted} / {b.field_players}</td><td data-label="Oldest price">{at(b.quote_at)}{b.freshness_unknown && <small>Availability unknown</small>}</td></tr>)}</tbody></table></div>
    <figure className="msc-deltas" aria-label="Book minus model scoring differences"><figcaption>Book minus model · easier ← 0 → harder</figcaption>{r.books.map((b) => <div className="msc-delta" key={b.book}><span>{bookName(b.book)}</span><div className="msc-track" aria-hidden="true"><i style={{ left: `${50 + Math.min(0, b.delta) / scale * 50}%`, width: `${Math.abs(b.delta) / scale * 50}%` }} /></div><b>{signedScore(b.delta)}</b></div>)}</figure>
    <p className="msc-note">Model issued {at(r.model_as_of)} · comparison {at(r.capture_at)} · first tee {at(r.first_tee_utc)}. Model run <code>{r.model_run}</code>.</p>
    <details className="msc-details"><summary>Player prices and inference checks</summary>
      {r.books.map((b) => <section key={b.book}><h3>{bookName(b.book)}</h3><p className="msc-note">Alternative field estimates: actual-wave weighted {fmt(b.wave_weighted_field_mean)} · median player shift {fmt(b.median_shift_field_mean)} · additive de-vig {fmt(b.additive_devig_field_mean)}. These are sensitivity checks, not confidence bounds.</p>{b.players.length ? <div className="score-table-scroll" role="region" aria-label={`${bookName(b.book)} player prices`} tabIndex={0}><table className="msc-table"><thead><tr><th>Player / wave</th><th>Line</th><th>Over / under decimal</th><th>No-vig under</th><th>Our mean</th><th>Inferred mean</th><th>Shift</th></tr></thead><tbody>{b.players.map((p) => <tr key={p.id}><th scope="row">{p.name}<small>{p.wave ?? "Wave unknown"} · ID {p.id}</small></th><td>{p.line.toFixed(1)}</td><td>{p.over_decimal.toFixed(3)} / {p.under_decimal.toFixed(3)}</td><td>{(p.under_no_vig * 100).toFixed(1)}%</td><td>{fmt(p.model_mean)}</td><td>{fmt(p.market_mean)}</td><td>{signedScore(p.shift)}</td></tr>)}</tbody></table></div> : <p className="msc-note">Player detail is published for the newest capture of each model run and round. This earlier capture retains its original summary.</p>}</section>)}
      <p className="msc-note">Forecast hash <code>{r.model_hash}</code> · comparison ID <code>{r.id}</code>.</p>
    </details>
  </>;
}
