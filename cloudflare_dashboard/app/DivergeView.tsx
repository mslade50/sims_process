"use client";

/**
 * "Why we differ" section of the player drawer (week documents from explain version 11): our chances next to Data Golf's and the sportsbooks' with the odd one out
 * marked, the Data Golf odds check, the split of our gap to the market, how the player differs from the players priced like them, and recent form.
 * Loaded on demand (lazy in ExplainViews) so the This week page does not carry it. Logic in diverge-rules.ts; numbers and sentences come from the publisher.
 */
import { useMemo } from "react";
import { Term } from "./ExplainBits";
import { availableMarkets, pct, plainName, shapeMap, shapeOf, signed, type ExPlayer, type ExplainDoc } from "./explain-rules";
import { SOURCE_KEYS, SOURCE_LABEL, chanceRows, formTable, leadSentence, oddColumn, peerLinks, peerPartRows, splitView } from "./diverge-rules";

const mean = (v: number | null) => (v === null ? "-" : signed(v));

export function DivergeSection({ doc, player, onSelect }: { doc: ExplainDoc; player: ExPlayer; onSelect: (id: number) => void }) {
  const d = player.diverge;
  const map = useMemo(() => shapeMap(doc), [doc]);
  const skillMarket = useMemo(() => {
    const markets = availableMarkets(doc);
    return (p: ExPlayer): number | null => shapeOf(p, map, markets)?.skill_market ?? null;
  }, [doc, map]);
  if (!d) return null;
  const rows = chanceRows(d.compare);
  const odd = oddColumn(d.compare);
  const split = splitView(d.edge_split);
  const peers = d.peers;
  const links = peerLinks(peers, doc);
  const parts = peerPartRows(peers);
  const form = formTable(doc, player, peers?.ids ?? [], skillMarket);
  const lead = leadSentence(d.sentences);
  const ck = d.dg_check;
  const note = doc.divergence?.course_data?.note ?? null;
  const formScale = doc.divergence?.form_source?.scale ?? "adjusted strokes gained a round vs an average PGA Tour field";
  return (
    <section className="ex-dv" aria-label="Why we differ">
      <h3 className="ex-h3">Why we differ</h3>
      {lead && <p className="ex-dv-lead">{lead}</p>}

      {rows.length > 0 && (
        <div className="table-scroll"><table className="ex-table ex-dv-table" aria-label="Chances: ours, Data Golf and the sportsbooks">
          <thead><tr><th>Chance</th>{SOURCE_KEYS.map((k) => (
            <th key={k} className={odd === k ? "is-odd" : ""}>{SOURCE_LABEL[k]}{odd === k && <small className="ex-dv-odd"><Term k="ex_odd_one_out">odd one out</Term></small>}</th>
          ))}</tr></thead>
          <tbody>{rows.map((r) => (
            <tr key={r.market}><th scope="row">{r.label}</th>{SOURCE_KEYS.map((k) => <td key={k} data-label={SOURCE_LABEL[k]} className={odd === k ? "is-odd" : ""}>{pct(r.values[k])}</td>)}</tr>
          ))}</tbody></table></div>
      )}

      {ck?.flag && ck.sentence && <p className="ex-callout is-warn" role="note"><span aria-hidden="true">⚠ </span><strong><Term k="ex_dg_check">Data Golf odds check</Term>.</strong> {ck.sentence}</p>}

      {split && (
        <div className="ex-dv-split" aria-label="Our rating and the sportsbooks' rating against the closest-priced players">
          <div className="ex-dv-pair">
            <div><span className="ex-dv-pair-label">We rate them vs the <Term k="ex_peers">closest-priced players</Term></span><b className={split.ours >= 0 ? "pos" : "neg"}>{signed(split.ours)}</b><small>strokes a round</small></div>
            <div><span className="ex-dv-pair-label">The sportsbooks rate them</span><b className={split.book >= 0 ? "pos" : "neg"}>{signed(split.book)}</b><small>strokes a round</small></div>
          </div>
          {split.particular !== null && <p className="ex-muted">The gap between the two is {signed(split.particular)} strokes a round, which is particular to this player.</p>}
        </div>
      )}
      {d.edge_split?.sentence && <p className="ex-muted ex-dv-edge">{d.edge_split.sentence}</p>}

      {peers && (parts.length > 0 || links.length > 0) && (
        <div className="ex-dv-peers">
          <h4 className="ex-dv-h4">Shared with <Term k="ex_peers">players priced like them</Term>, or particular to this player</h4>
          {links.length > 0 && <ul className="ex-dv-peerlist" aria-label="Players priced like this one">{links.map((l) => <li key={l.id}><button type="button" className="ex-linkbtn" onClick={() => onSelect(l.id)}>{plainName(l.name)}</button></li>)}</ul>}
          {parts.length > 0 && (
            <div className="table-scroll"><table className="ex-table ex-dv-table">
              <thead><tr><th>Part</th><th>This player</th><th title="Median of the players with the nearest sportsbook win chance.">Priced-like-them median</th><th title="This player minus the median: what sets them apart in our model.">Difference</th></tr></thead>
              <tbody>{parts.map((r) => (
                <tr key={r.key} className={r.standout ? "is-standout" : ""}>
                  <th scope="row">{r.label}</th>
                  <td data-label="This player">{mean(r.value)}</td><td data-label="Priced like them">{mean(r.peer)}</td>
                  <td data-label="Difference" className={r.diff === null || Math.abs(r.diff) < 0.005 ? "" : r.diff > 0 ? "pos" : "neg"}>{mean(r.diff)}</td>
                </tr>
              ))}</tbody></table></div>
          )}
          {peers.sentence && <p className="ex-muted">{peers.sentence}</p>}
        </div>
      )}

      {form && (
        <div className="ex-dv-form">
          <h4 className="ex-dv-h4">Recent form</h4>
          <div className="table-scroll"><table className="ex-table ex-dv-table ex-dv-formtable">
            <thead><tr><th>Player</th>{form.windows.map((w) => <th key={w} title={`Average over the last ${w.slice(1)} rounds before the event.`}>{w}</th>)}{form.years.map((y) => <th key={y} title={`Average over all rounds in ${y}.`}>{y}</th>)}<th title="Our skill estimate, strokes a round against this field's average.">Our skill</th><th title="The skill the sportsbook's prices imply, same scale as our skill.">Sportsbook implies</th></tr></thead>
            <tbody>{form.rows.map((r) => (
              <tr key={r.id} className={r.self ? "is-self" : ""} aria-current={r.self ? "true" : undefined}>
                <th scope="row">{r.self ? plainName(r.name) : <button type="button" className="ex-linkbtn" onClick={() => onSelect(r.id)}>{plainName(r.name)}</button>}</th>
                {r.windows.map((v, i) => <td key={form.windows[i]} data-label={form.windows[i]}>{mean(v)}</td>)}
                {r.seasons.map((v, i) => <td key={form.years[i]} data-label={String(form.years[i])}>{mean(v)}</td>)}
                <td data-label="Our skill">{mean(r.ours)}</td><td data-label="Sportsbook implies">{mean(r.market)}</td>
              </tr>
            ))}</tbody></table></div>
          <p className="ex-muted">Form columns are {formScale}, from rounds before the event (L8 is the last 8 rounds). Our skill and the sportsbook&rsquo;s are strokes a round against this field&rsquo;s average, so the two scales differ by the field&rsquo;s strength.</p>
        </div>
      )}

      {note && <p className="ex-muted ex-dv-note">{note}</p>}
    </section>
  );
}
