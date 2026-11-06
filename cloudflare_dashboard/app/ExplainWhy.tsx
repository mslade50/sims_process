"use client";

/** Why priced: every player's price against the market, with the reasons. Split from ExplainViews so This week does not ship it. */
import { useMemo, useState } from "react";
import { ArrowDown, ArrowUp, Search, X } from "lucide-react";
import { EmptyState, SegmentedControl } from "./components";
import { LegendGroups } from "./ExplainBits";
import { MARKET_LABEL, TAG_GROUPS, availableMarkets, availableTags, edgeBadge, filterWhy, gapText, pct, shapeMap, shapeOf, signed, sortWhy, tagLabel, type SortKey } from "./explain-rules";
import { etTime } from "./lib";
import { OnThisPage, Section } from "./ui";
import "./explain.css";
import { ExplainShell, PriceList, RAIL_RANGE, type Shell, type PriceRowData } from "./ExplainViews";

/* ------------------------------------------------------------------ Why priced */
export function WhyPricedView() {
  return (
    <ExplainShell title="Why priced" preEventFirst description="Every player's price against the market, with the reasons.">
      {(s) => <WhyPriced {...s} />}
    </ExplainShell>
  );
}

const SORT_LABEL: Array<{ key: SortKey; label: string }> = [
  { key: "edge_sg", label: "Gap in strokes" }, { key: "rel", label: "Gap in %" }, { key: "model", label: "Our chance" }, { key: "market", label: "Sportsbook" }, { key: "mu", label: "Expected skill" }, { key: "name", label: "Name" },
];
const SORT_HELP: Partial<Record<SortKey, string>> = {
  edge_sg: "Gap in strokes: how many strokes a round better (+) or worse (-) our numbers rate a player than the sportsbook prices imply, allowing for how boom-or-bust each side sees them.",
  rel: "Gap in %: how far our chance is from the sportsbook's, as a share of the sportsbook's chance.",
  model: "Our chance for the chosen bet type.", market: "The sportsbook's chance for the chosen bet type.",
  mu: "Expected skill: strokes per round better (+) or worse (-) than the average player in this field.", name: "Alphabetical.",
};
const COLUMN_HELP = {
  pos: "Current place on the leaderboard (MC = missed the cut, WD = withdrew).",
  model: "Our chance for this bet type, from the simulations.",
  market: "The chance implied by sportsbook odds, with the bookmaker margin removed.",
  rel: "How far our chance is from the sportsbook's, as a share of the sportsbook's. Plus means we are higher than the sportsbook.",
  edge_sg: "Skill gap: how many strokes a round better (+) or worse (-) we rate them than the sportsbook prices imply, after allowing for how much their scores swing (spread). Blank when the sportsbook prices too few bet types to tell.",
  mu: "Our estimate of their skill: strokes per round better (+) or worse (-) than the average player in this field.",
  sd: "Std dev: the standard deviation of their round scores, in strokes. Higher means a streakier, less predictable player.",
} as const;

function WhyPriced({ doc, market, setMarket, openPlayer }: Shell) {
  const [query, setQuery] = useState("");
  const [tags, setTags] = useState<string[]>([]);
  const [only, setOnly] = useState<"all" | "above" | "below">("all");
  const [top, setTop] = useState<"all" | "60" | "25">("25");
  const [sort, setSort] = useState<{ key: SortKey; dir: "asc" | "desc" }>({ key: "edge_sg", dir: "desc" });
  const live = doc.kind === "live";
  const activeFilters = (only !== "all" ? 1 : 0) + (top !== "25" ? 1 : 0) + (sort.key !== "edge_sg" || sort.dir !== "desc" ? 1 : 0) + tags.length;
  const markets = useMemo(() => availableMarkets(doc), [doc]);
  const tagOptions = useMemo(() => availableTags(doc.players), [doc.players]);
  // Before the event the strokes gap and the reason come from the bet-by-bet shape comparison (skill the sportsbook's prices imply vs ours); players it cannot be computed for show no gap.
  const players = useMemo(() => {
    if (live) return doc.players;
    const map = shapeMap(doc);
    return doc.players.map((p) => { const s = p.withdrawn ? null : shapeOf(p, map, markets); return { ...p, edge_sg: s ? s.skill_gap : null, why_edge: s ? s.sentence : null }; });
  }, [doc, live, markets]);
  const rows = useMemo(() => sortWhy(filterWhy(players, { query, tags, market, onlyEdge: only, top: top === "all" ? null : Number(top) }), sort.key, sort.dir, market), [players, query, tags, market, only, top, sort]);
  const hasMarket = !live && rows.some((p) => p.probs[market].market !== null);
  const hasModel = rows.some((p) => p.probs[market].model !== null);
  const hasGap = rows.some((p) => p.edge_sg !== null);
  const hasSd = rows.some((p) => p.sd !== null && p.sd !== undefined);
  const plRows: PriceRowData[] = rows.map((p) => {
    const e = p.probs[market];
    const b = edgeBadge(p, market);
    const nums: PriceRowData["nums"] = [];
    if (live) nums.push({ k: "Pos", v: p.live?.tier === 0 ? p.live?.pos ?? "-" : p.live?.tier === 2 ? "MC" : "WD", help: COLUMN_HELP.pos });
    if (hasModel) nums.push({ k: "Model", v: pct(e.model), help: COLUMN_HELP.model });
    if (hasMarket) nums.push({ k: "Book", v: pct(e.market), help: COLUMN_HELP.market });
    if (hasGap && hasMarket) nums.push({ k: "Skill gap", v: <>{gapText(p.edge_sg)}<small> str/rd</small></>, cls: Math.abs(p.edge_sg ?? 0) < 0.005 ? "" : (p.edge_sg ?? 0) > 0 ? "pos" : "neg", help: COLUMN_HELP.edge_sg });
    nums.push({ k: "Skill", v: signed(p.mu_rel), help: COLUMN_HELP.mu });
    if (hasSd) nums.push({ k: "Std dev", minor: true, v: p.sd?.toFixed(2) ?? "-", help: COLUMN_HELP.sd });
    return { id: p.id, name: p.name, tags: p.tags, nums, rail: hasMarket && e.model != null && e.market != null ? { model: e.model * 100, book: e.market * 100 } : null, relNote: b.tone === "neutral" ? "in line" : `${b.text} vs book`, why: (hasMarket ? p.why_edge : null) ?? p.why };
  });
  return (
    <>
      <OnThisPage sections={[{ id: "players", label: "Players" }, { id: "legend", label: "What reasons mean" }, { id: "tech", label: "Technical" }]} />
      <Section id="players" title={`${rows.length} players, ${MARKET_LABEL[market].toLowerCase()}`} purpose={`Tap a player for the driver breakdown.${hasMarket ? " Bars to the right of the zero tick: we are higher than the book." : ""}`}
        actions={<SegmentedControl label="Bet type" value={market} onChange={setMarket} options={markets.map((m) => ({ value: m, label: MARKET_LABEL[m] }))} />}>
        <div className="ex-filters">
          <label className="search-box"><Search size={15} /><input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search player" aria-label="Search player" /></label>
        </div>
        <details className="ex-details ex-typefilter ex-filterbox" open={undefined}>
          <summary>Filters{activeFilters > 0 ? ` (${activeFilters})` : ""}</summary>
          <div className="ex-filters">
            <SegmentedControl label="Show players we rate" value={only} onChange={setOnly} options={[{ value: "all", label: "All" }, { value: "above", label: "We are higher" }, { value: "below", label: "We are lower" }]} />
            <SegmentedControl label="Show" value={top} onChange={setTop} options={[{ value: "25", label: "Top 25" }, { value: "60", label: "Top 60" }, { value: "all", label: "Whole field" }]} />
            <label className="ex-sort">Sort by <select value={sort.key} title={SORT_HELP[sort.key]} onChange={(e) => setSort({ key: e.target.value as SortKey, dir: e.target.value === "name" ? "asc" : "desc" })}>{SORT_LABEL.map((o) => <option key={o.key} value={o.key} title={SORT_HELP[o.key]}>{o.label}</option>)}</select>
              <button type="button" className="ex-sortdir" onClick={() => setSort((s) => ({ ...s, dir: s.dir === "asc" ? "desc" : "asc" }))} title="Reverse the order" aria-label={sort.dir === "asc" ? "Sorted low to high, tap to reverse" : "Sorted high to low, tap to reverse"}>{sort.dir === "asc" ? <ArrowUp size={14} /> : <ArrowDown size={14} />}{sort.dir === "asc" ? "low to high" : "high to low"}</button></label>
          </div>
          <p className="ex-muted">Player type{tags.length > 0 ? ` (${tags.length} selected)` : ""}:</p>
          <div className="ex-chipgroups" role="group" aria-label="Filter by player type">
            {TAG_GROUPS.map((g) => {
              const opts = g.tags.filter((t) => tagOptions.includes(t));
              return opts.length === 0 ? null : (
                <div className="ex-chipgroup" key={g.label}>
                  <span className="ex-chipgroup-label" title={g.help}>{g.label}</span>
                  <div className="ex-chips">{opts.map((t) => <button key={t} type="button" title={tagLabel(t)} aria-pressed={tags.includes(t)} className={tags.includes(t) ? "on" : ""} onClick={() => setTags((cur) => (cur.includes(t) ? cur.filter((x) => x !== t) : [...cur, t]))}>{tagLabel(t)}</button>)}</div>
                </div>
              );
            })}
            {tags.length > 0 && <button type="button" className="ex-clear" onClick={() => setTags([])}><X size={12} /> clear player types</button>}
          </div>
        </details>
        {rows.length > 0 && <PriceList rows={plRows} range={RAIL_RANGE[market]} label="Player prices" onOpen={openPlayer} heads={["Player", "Numbers", "Model vs book (points)", "Why"]} />}
        {rows.length === 0 && <EmptyState title="No players match" detail="Clear a filter or the search." />}
        {!hasMarket && !live && rows.length > 0 && <p className="ex-muted">No sportsbook prices are posted for {MARKET_LABEL[market].toLowerCase()}, so only our own numbers are shown.</p>}
        <p className="ex-muted ex-marketnote">The strokes gap and its reasons are the same for every bet type; only the probability gap changes.{live ? " Sportsbook prices are not shown during play because they are not live." : ""}</p>
      </Section>
      <Section id="legend" title="What the reasons mean" purpose="The model components behind each price.">
        <details className="ex-glossary-wrap"><summary>Open the glossary</summary><LegendGroups legend={doc.components_legend} /></details>
      </Section>
      <Section id="tech" title="Technical details">
        <p className="ex-muted">{doc.model.n_sims ? `Prices come from ${doc.model.n_sims.toLocaleString()} simulated tournaments. ` : ""}Priced {etTime(doc.as_of)}.</p>
        <p className="ex-muted">{doc.kind === "live" ? `In-play pricing after round ${doc.after_round}` : "Pre-tournament pricing"}; finish chances {doc.finish.method === "exact_tapes" ? "come straight from the simulations" : doc.finish.method === "unavailable" ? "are not available" : "were rebuilt by replaying the hole-by-hole model"}.</p>
      </Section>
    </>
  );
}

