"use client";

/** Player drawer (finish shape, score distribution, why-this-price waterfall, competitor overlay). Loaded when a player is opened. */
import Link from "next/link";
import { lazy, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useDialogFocus } from "./ShellParts";
import { ChevronRight, CloudSun, Search, X } from "lucide-react";
import { Kpi, SegmentedControl } from "./components";
import { MarketRows, PLAYER_COLORS, ScoreFan, ShapeBars, Waterfall } from "./ExplainBits";
import type { FinishSeries } from "./ExplainCharts";
import { EDGE_MIN, MARKET_LABEL, MAX_OVERLAY, availableMarkets, defaultCompetitors, nameMatches, pct, shapeMap, shapeOf, signed, modalFinish, modalFinishText, waterfallScale, waterfallRowScales, toParText, toggleCompetitor, topK, valueBets, waterfall, type ExPlayer, type ExplainDoc, type Market } from "./explain-rules";
import { LABELS } from "./labels";
import { Delta } from "./ui";
import "./explain.css";
import { Lazy, Tags } from "./ExplainViews";

/* ------------------------------------------------------------------ Player drawer */
const FinishChart = lazy(() => import("./ExplainCharts").then((m) => ({ default: m.FinishChart })));
const DivergeSection = lazy(() => import("./DivergeView").then((m) => ({ default: m.DivergeSection })));

export function PlayerDrawer({ doc, player, competitors, market, setCompetitors, onSelect, onClose }: {
  doc: ExplainDoc; player: ExPlayer; competitors: number[]; market: Market; setCompetitors: (ids: number[]) => void; onSelect: (id: number) => void; onClose: () => void;
}) {
  const [mode, setMode] = useState<"probability" | "cumulative">("probability");
  const [q, setQ] = useState("");
  const byId = useMemo(() => new Map(doc.players.map((p) => [p.id, p] as const)), [doc.players]);
  const wf = useMemo(() => waterfall(player, doc.components_legend), [player, doc.components_legend]);
  const wfScale = useMemo(() => waterfallScale(doc.players), [doc.players]);
  const wfScales = useMemo(() => waterfallRowScales(doc.players, doc.components_legend), [doc.players, doc.components_legend]);
  const shape = useMemo(() => (doc.kind === "live" || player.withdrawn ? null : shapeOf(player, shapeMap(doc), availableMarkets(doc))), [doc, player]);
  const chosen = [player, ...competitors.map((id) => byId.get(id)).filter((p): p is ExPlayer => !!p)];
  const withShape = chosen.filter((p) => p.finish);
  const series: FinishSeries[] = withShape.map((p, i) => ({ id: p.id, name: p.name, pos: p.finish!.pos, missCut: doc.event.cut_round ? p.finish!.p_miss_cut : null, color: PLAYER_COLORS[i % PLAYER_COLORS.length] }));
  const hits = q.trim() ? doc.players.filter((p) => p.finish && !p.withdrawn && p.id !== player.id && !competitors.includes(p.id) && nameMatches(p.name, q)).slice(0, 6) : [];
  const f = player.finish;
  const t = f ? topK(f.pos, [1, 5, 10, 20]) : [];
  const rounds = doc.event.rounds ?? 4;
  const live = player.live;
  const layerRef = useRef<HTMLDivElement>(null);
  // Modal contract (same hook as the phone More sheet): focus moves in, Tab is trapped, Escape closes, the page behind is inert, focus returns to the row that opened it.
  useDialogFocus(true, layerRef, onClose, ["main", ".sidebar", ".tabbar", ".skip-link", ".sidebar-backdrop"], { initial: ".ex-close" });
  const layer = (
    <div className="ex-drawer-layer" role="dialog" aria-modal="true" aria-labelledby="ex-drawer-title" ref={layerRef}>
      <button className="ex-drawer-backdrop" aria-label="Close player detail" tabIndex={-1} onClick={onClose} />
      <aside className="ex-drawer">
        <header>
          <div>
            <span className="eyebrow">{doc.kind === "live" ? `In play, after round ${doc.after_round}` : "Before the event"}</span>
            <h2 id="ex-drawer-title">{player.name}</h2>
            <a className="ex-link" href={`/players?player=${player.id}`}>Profile and vs PGA avg <ChevronRight size={14} /></a>
            <Tags tags={player.tags} max={8} />
          </div>
          <button type="button" className="ex-close" onClick={onClose} aria-label="Close player detail"><X size={20} /></button>
        </header>

        <div className="ex-drawer-kpis">
          <Kpi label={`${MARKET_LABEL[market]} chance`} value={pct(player.probs[market].model)} detail={doc.kind === "live" ? "our chance now" : `sportsbook ${pct(player.probs[market].market)}`} tone="model" />
          <Kpi label={`Expected skill ${LABELS.vsField.short}`} value={signed(player.mu_rel)} detail={`strokes a round · std dev ${player.sd?.toFixed(2) ?? "-"}`} tone={(player.mu_rel ?? 0) >= 0 ? "positive" : "negative"} />
          {f && <div className="ex-kpi-wrap"><Kpi label="Median finish" value={String(f.median)} detail={`Likely range: ${f.p10} to ${f.p90}`} tone="accent" /><small className="ex-modal">{modalFinishText(modalFinish(f, !!doc.event.cut_round))}</small></div>}
          {f && <Kpi label="Top 10 chance" value={pct(t[2])} detail={`Win chance ${pct(t[0])}${doc.event.cut_round ? ` · misses the cut ${pct(f.p_miss_cut, 0)}` : ""}`} tone="neutral" />}
        </div>
        {shape && <p className="ex-callout"><strong>{shape.sentence}</strong> {shape.parts_sentence}{valueBets(shape).length > 0 ? ` Value against the sportsbook: ${valueBets(shape).join(", ")}.` : ""}</p>}
        {doc.kind !== "live" && !player.withdrawn && player.diverge && <Lazy height={420}><DivergeSection doc={doc} player={player} onSelect={onSelect} /></Lazy>}

        <section>
          <h3 className="ex-h3">Why this price</h3>
          <p className="ex-muted">Strokes per round above (+) or below (−) the average player in this field ({LABELS.vsField.short}); the {LABELS.vsPgaAvg.short} view is on the profile. The skill rows add up to their expected skill{doc.kind === "live" ? "; the in-play rows are part of it" : ""}; weather is shown separately because it adjusts scores on top.</p>
          <Waterfall {...wf} scale={wfScale} scales={wfScales} />
          {shape && <p className="ex-muted">The sportsbook&rsquo;s prices imply skill {signed(shape.skill_market)} and spread {shape.spread_market.toFixed(2)}; ours are {signed(shape.skill_ours)} (including weather) and {shape.spread_ours.toFixed(2)}.</p>}
        </section>

        <section>
          <h3 className="ex-h3">Our chance vs the sportsbook</h3>
          {shape && <div className="ex-shape-wrap"><ShapeBars shape={shape} markets={availableMarkets(doc)} /><span className="ex-muted">Up: our chance is higher than the sportsbook&rsquo;s. Down: lower. Gray: within {Math.round(EDGE_MIN * 100)}%.</span></div>}
          <MarketRows player={player} hasCut={!!doc.event.cut_round} />
        </section>

        <section>
          <div className="ex-sec-head"><h3 className="ex-h3">Finish shape</h3>
            <SegmentedControl label="Chart type" value={mode} onChange={setMode} options={[{ value: "probability", label: "Position" }, { value: "cumulative", label: "Cumulative" }]} /></div>
          {doc.finish.method === "unavailable" || !f ? <p className="ex-muted">Finish odds are not available for this run.</p> : (
            <>
              <Lazy height={330}><FinishChart series={series} cutTopN={doc.event.cut_round ? doc.event.cut_top_n : null} mode={mode} /></Lazy>
              <div className="ex-compare">
                <h4>Compare with</h4>
                <div className="chip-row">
                  {competitors.map((id) => <span className="chip" key={id}>{byId.get(id)?.name ?? id}<button type="button" onClick={() => setCompetitors(competitors.filter((x) => x !== id))} aria-label={`Remove ${byId.get(id)?.name}`}><X size={13} /></button></span>)}
                  {competitors.length === 0 && <span className="ex-muted">nobody selected</span>}
                </div>
                <div className="ex-compare-actions">
                  <button type="button" onClick={() => setCompetitors(defaultCompetitors(doc, player.id, 3))}>Same-odds neighbors</button>
                  <button type="button" onClick={() => setCompetitors([])}>Clear</button>
                </div>
                <label className="search-box"><Search size={15} /><input value={q} onChange={(e) => setQ(e.target.value)} placeholder={competitors.length >= MAX_OVERLAY - 1 ? "Maximum reached" : "Add a competitor"} disabled={competitors.length >= MAX_OVERLAY - 1} aria-label="Add competitor" /></label>
                {hits.length > 0 && <ul className="ex-hits">{hits.map((p) => <li key={p.id}><button type="button" onClick={() => { setCompetitors(toggleCompetitor(competitors, p.id)); setQ(""); }}>{p.name}<small>{pct(p.probs.win.model)} win</small></button></li>)}</ul>}
              </div>
              <div className="table-scroll"><table className="ex-table ex-compare-table">
                <thead><tr><th>Player</th><th title="Expected skill: strokes per round better (+) or worse (-) than the field average.">Skill</th><th>Win</th><th>Top 5</th><th>Top 10</th><th>Top 20</th>{doc.event.cut_round ? <th title="Our chance to make the cut.">Make cut</th> : null}<th title="The finish position that half of the simulations beat and half did not. The smaller note is the single most likely finish.">Median finish</th></tr></thead>
                <tbody>{withShape.map((p, i) => (
                  <tr key={p.id}><th scope="row"><i className="ex-dot" style={{ background: PLAYER_COLORS[i % PLAYER_COLORS.length] }} aria-hidden="true" />
                    {p.id === player.id ? p.name : <button type="button" className="ex-linkbtn" onClick={() => onSelect(p.id)}>{p.name}</button>}</th>
                    <td data-label="Skill">{signed(p.mu_rel)}</td><td data-label="Win">{pct(p.probs.win.model)}</td><td data-label="Top 5">{pct(p.probs.top_5.model)}</td><td data-label="Top 10">{pct(p.probs.top_10.model)}</td><td data-label="Top 20">{pct(p.probs.top_20.model)}</td>
                    {doc.event.cut_round ? <td data-label="Make cut">{pct(p.probs.make_cut.model ?? (p.finish ? 1 - p.finish.p_miss_cut : null), 0)}</td> : null}
                    <td data-label="Median finish">{p.finish!.median}<small className="ex-modal" title="The single most likely finish position">{modalFinishText(modalFinish(p.finish, !!doc.event.cut_round)).replace("Most likely finish: ", "most likely ").replace("Most likely: ", "most likely: ")}</small></td></tr>
                ))}</tbody></table></div>
              <p className="ex-muted">Finish chances come from {doc.finish.method === "exact_tapes" ? "the simulations themselves" : `${doc.finish.n_draws?.toLocaleString() ?? "many"} replayed tournaments`}{doc.finish.calibration?.top_20 ? `; on average they sit within ${pct(doc.finish.calibration.top_20.mean_abs, 1)} of the posted top-20 chances` : ""}.</p>
            </>
          )}
        </section>

        <section>
          <h3 className="ex-h3">Score distribution</h3>
          <ScoreFan rounds={rounds} items={withShape.map((p, i) => ({ id: p.id, name: p.name, color: PLAYER_COLORS[i % PLAYER_COLORS.length], scores: p.scores }))} />
        </section>

        {live && (
          <section>
            <h3 className="ex-h3">In play</h3>
            <div className="ex-live-grid">
              <span>Position <b>{live.pos ?? "-"}</b></span><span>To par <b>{toParText(live.to_par)}</b></span>
              <span>Skill before the event → now <b>{signed(live.mu_pre)} → {signed(live.mu_live)}</b></span>
              <span>Change since last run <b>{signed(live.d_mu_since_last)}</b> (form this week {signed(live.d_b8_since_last)}{Math.abs(live.d_contention_since_last ?? 0) >= 0.005 ? `, title race ${signed(live.d_contention_since_last)}` : ""})</span>
            </div>
            {live.rounds.length > 0 && <div className="table-scroll"><table className="ex-table"><thead><tr><th>Round</th><th>Score</th><th title="Strokes gained in total versus the field.">Total</th><th title="Strokes gained off the tee.">Driving</th><th title="Strokes gained on approach shots.">Approach</th><th title="Strokes gained around the green.">Short game</th><th title="Strokes gained putting.">Putting</th></tr></thead>
              <tbody>{live.rounds.map((r) => <tr key={r.round}><th scope="row">R{r.round}</th><td>{r.score ?? "-"}</td><td>{signed(r.sg_total)}</td><td>{signed(r.sg_ott)}</td><td>{signed(r.sg_app)}</td><td>{signed(r.sg_arg)}</td><td>{signed(r.sg_putt)}</td></tr>)}</tbody></table></div>}
            <div className="table-scroll"><table className="ex-table"><thead><tr><th>Market</th><th>Before the event</th><th>Now</th><th title="Change since the previous pricing run, in percentage points.">Change since last run</th></tr></thead>
              <tbody>{availableMarkets(doc).map((m) => <tr key={m}><th scope="row">{MARKET_LABEL[m]}</th><td>{pct(live.pre_event[m])}</td><td>{pct(player.probs[m].model)}</td><td><Delta value={(live.since_last[m] ?? 0) * 100} digits={1} suffix="pp" /></td></tr>)}</tbody></table></div>
          </section>
        )}
        <p className="ex-muted ex-foot"><CloudSun size={13} /> Wave, time-of-day and wind effects are in the waterfall (weather rows) and on the <Link prefetch={false} className="ex-link" href="/weather-effects">Weather page</Link>.</p>
      </aside>
    </div>
  );
  return typeof document === "undefined" ? layer : createPortal(layer, document.body);
}

