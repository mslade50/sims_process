"use client";

import Link from "next/link";

import { useEffect, useMemo, useState, useSyncExternalStore, type ReactNode } from "react";
import { useEventParam } from "./event-param";
import { Panel } from "./components";
import { useDashboardData } from "./data";
import { explainKeyFor, orderEvents, parseExplain, pct, signed, type ExplainDoc, type ExPlayer, type IndexEvent } from "./explain-rules";
import { STAGES, STAGE_EVIDENCE_CONTRACT, weekStart } from "./ops-rules";
import { PageHeader, Section as UiSection, useNow } from "./ui";
import { finishBuckets, matchingExplain, safeEvidenceLink, stageSignal, workflowFor, type Signal } from "./walkthrough-rules";
import "./walkthrough.css";

type Topic = "week" | "skill" | "sim" | "live" | "evidence";
const TOPICS: Array<[Topic, string, string]> = [
  ["week", "This week", "Which weekly outputs are verified"],
  ["skill", "Skill model", "How a golfer's ability is estimated"],
  ["sim", "Simulation", "Skill becomes thousands of tournaments"],
  ["live", "In-event", "Keep what happened, simulate the rest"],
  ["evidence", "Proof", "What is proven, and what is not"],
];
const fmt = (v: string | null | undefined) => v && Number.isFinite(Date.parse(v)) ? new Date(v).toLocaleString("en-US", { timeZone: "America/New_York", month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZoneName: "short" }) : "No timestamp";
const number = (v: number | null | undefined, digits = 2) => typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "Unavailable";
const subscribeNarrow = (notify: () => void) => { const q = window.matchMedia("(max-width: 640px)"); q.addEventListener("change", notify); return () => q.removeEventListener("change", notify); };
/** On a phone each explainer section starts folded behind one tap, so the page is a short list of headings, not 17 screens. */
function Fold({ label, children }: { label: string; children: ReactNode }) {
  const narrow = useSyncExternalStore(subscribeNarrow, () => window.matchMedia("(max-width: 640px)").matches, () => false);
  if (!narrow) return <>{children}</>;
  return <details className="wt-fold"><summary>{label}</summary>{children}</details>;
}
function Lamp({ signal }: { signal: Signal }) { return <span className={`wt-lamp wt-${signal.tone}`}><span aria-hidden="true">{signal.tone === "green" ? "✓" : signal.tone === "red" ? "!" : "·"}</span>{signal.label}</span>; }
function Chain({ items }: { items: Array<[string, string]> }) { return <ol className="wt-chain">{items.map(([title, detail], i) => <li key={title}><span className="wt-step">{i + 1}</span><strong>{title}</strong><p>{detail}</p></li>)}</ol>; }
function Evidence({ value }: { value: string }) { const href = safeEvidenceLink(value); return href ? <Link prefetch={false} href={href} target="_blank" rel="noreferrer">Open output evidence</Link> : <code>{value}</code>; }

export function WalkthroughView() {
  const index = useDashboardData<{ events?: IndexEvent[] }>("golfprice/index.json");
  const events = useMemo(() => orderEvents(index.data?.events ?? []), [index.data]);
  const barEvent = useEventParam();
  const selection = barEvent; // the event is chosen in the top bar only
  const event = events.find(e => e.event_uid === selection) ?? events[0];
  return <div className="wt-page">
    <PageHeader title="From golf shots to prices" purpose="How a golfer's skill becomes a price, and which weekly outputs are verified." actions={<Link prefetch={false} href="/ops" className="wt-linkbtn">Open full weekly workflow</Link>} />
    {index.error && <p className="wt-notice" role="status">Event data unavailable: {index.error} <button type="button" onClick={index.refetch}>Retry</button></p>}
    <nav aria-label="On this page: the pipeline in five steps"><ol className="wt-flow">{TOPICS.map(([key, label, detail], i) => <li key={key}><a href={`#${key}`}><span className="wt-flow-n">{i + 1}</span><strong>{label}</strong><small>{detail}</small></a></li>)}</ol></nav>
    {event ? <EventGuide key={event.event_uid} event={event} indexStale={index.stale || !!index.error} /> : <>
      <UiSection id="week" title="This week’s process" purpose="Weekly completion is unverified until an event is connected."><p className="wt-notice">No current event is connected. The explanation is available below; weekly completion remains unverified.</p></UiSection>
      <GuideSections doc={null} />
    </>}
  </div>;
}

function EventGuide({ event, indexStale }: { event: IndexEvent; indexStale: boolean }) {
  const source = useDashboardData<unknown>(explainKeyFor(event));
  const doc = matchingExplain(parseExplain(source.data), event);
  const now = useNow(15_000);
  const week = event.date_start && /^\d{4}-\d{2}-\d{2}$/.test(event.date_start) ? weekStart(Date.parse(`${event.date_start}T12:00:00Z`)) : weekStart(now);
  const [raw, setRaw] = useState<unknown>(null), [error, setError] = useState<string | null>(null), [refresh, setRefresh] = useState(0);
  useEffect(() => {
    let alive = true; const controller = new AbortController();
    let revision = 0;
    const load = async () => {
      const ownRevision = ++revision;
      try {
        const r = await fetch(`/api/ops/workflow?week=${week}&event=${encodeURIComponent(event.event_uid)}`, { cache: "no-store", credentials: "same-origin", signal: AbortSignal.any([controller.signal, AbortSignal.timeout(10_000)]) });
        const d: unknown = await r.json();
        if (!r.ok || !workflowFor(d, event.event_uid, week)) throw new Error("Current event workflow evidence could not be verified.");
        if (alive && ownRevision === revision) { setRaw(d); setError(null); }
      } catch (e) { if (alive && ownRevision === revision) setError(e instanceof Error ? e.message : "Workflow evidence is unavailable."); }
    };
    const refreshVisible = () => { if (document.visibilityState === "visible") void load(); };
    void load(); const timer = setInterval(refreshVisible, 30_000);
    document.addEventListener("visibilitychange", refreshVisible);
    return () => { alive = false; controller.abort(); clearInterval(timer); document.removeEventListener("visibilitychange", refreshVisible); };
  }, [event.event_uid, week, refresh]);
  const flow = workflowFor(raw, event.event_uid, week);
  const signals = STAGES.map(s => stageSignal(flow?.stages.find(v => v.id === s.id), flow, now, indexStale ? "The event index is stale or unavailable; refresh before trusting completion." : error));
  const green = signals.filter(s => s.tone === "green").length, red = signals.filter(s => s.tone === "red").length;
  const headline = red ? `${red} step${red === 1 ? " needs" : "s need"} attention` : green === STAGES.length ? "All weekly outputs verified" : "Weekly verification is incomplete";
  return <>
    <UiSection id="week" title="This week’s process" purpose={`Week of ${week}, checked ${fmt(flow?.now)}. A published price is not a verified handoff.`} actions={<button type="button" className="wt-btn" onClick={() => setRefresh(n => n + 1)}>Refresh</button>}>
      <div className="wt-status" role="status" aria-label="Weekly status summary">
        <div className="wt-status-head"><h3>{headline}</h3><p>{green} verified · {STAGES.length - green - red} unverified · {red} need attention</p></div>
        <ol className="wt-strip" aria-hidden="true">{signals.map((signal, i) => <li key={STAGES[i].id} className={`wt-seg wt-${signal.tone}`} title={`${STAGES[i].label}: ${signal.tone}`} />)}</ol>
        <p className="wt-status-note">Unverified means no fresh output proof yet, not a failure.</p>
      </div>
      <WeeklyRail signals={signals} flow={flow} />
    </UiSection>
    <div className="wt-artifact"><strong>{doc ? `${doc.kind === "live" ? "In-event" : "Pre-tournament"} price artifact available` : source.loading ? "Loading published price artifact…" : "Price artifact unavailable or mismatched"}</strong><span>{doc ? `${doc.event.name} · ${fmt(doc.as_of)} · run ${doc.run}` : source.error ?? "No matching event/run document has been verified."}</span>{(source.stale || indexStale) && <b>Cached data; refresh before using these prices.</b>}</div>
    <GuideSections doc={doc} />
  </>;
}

function WeeklyRail({ signals, flow }: { signals: Signal[]; flow: ReturnType<typeof workflowFor> }) {
  const [selected, setSelected] = useState("weekly");
  const spec = STAGES.find(s => s.id === selected) ?? STAGES[0];
  const stage = flow?.stages.find(s => s.id === spec.id);
  const contract = STAGE_EVIDENCE_CONTRACT[spec.id as keyof typeof STAGE_EVIDENCE_CONTRACT];
  const timing: Record<string, string> = { inputs: "Before pricing", weekly: "Sun–Mon", refresh: "Tue–first tee", boards: "After pricing", checks: "Throughout the week", in_event: "After each round", delivery: "After publication", settlement: "After final results" };
  return <div className="wt-week-grid"><ol className="wt-rail" aria-label="Weekly process">{STAGES.map((s, i) => <li key={s.id}><button type="button" aria-pressed={selected === s.id} onClick={() => setSelected(s.id)}><span className={`wt-node wt-${signals[i].tone}`}>{i + 1}</span><span><strong>{s.label}</strong><small>{timing[s.id]}</small></span><Lamp signal={signals[i]} /></button></li>)}</ol><Panel title={spec.label} className="wt-proof"><Lamp signal={signals[STAGES.indexOf(spec)]} /><p>{spec.detail}.</p><p>{signals[STAGES.indexOf(spec)].detail}</p><dl><div><dt>Output observed</dt><dd>{fmt(stage?.evidence_at)}</dd></div><div><dt>Proof valid until</dt><dd>{fmt(stage?.valid_until)}</dd></div><div><dt>Latest execution</dt><dd>{stage?.execution ? `${stage.execution.type}: ${stage.execution.status?.state ?? "unknown"}` : "No scoped execution connected"}</dd></div></dl><details><summary>What this step must prove</summary><ul>{Object.entries(contract.required_checks).map(([key, text]) => <li key={key}>{text}</li>)}</ul></details>{stage?.proof.map((p, i) => <p key={i}><Evidence value={p} /></p>)}<Link prefetch={false} href="/ops">Inspect jobs, receipts and handoffs</Link></Panel></div>;
}

function GuideSections({ doc }: { doc: ExplainDoc | null }) {
  const purpose: Record<string, string> = {
    skill: "How the model forecasts a golfer's ability, and what went into this run.",
    sim: "How one forecast becomes many possible tournaments, then prices.",
    live: "How prices update once rounds are played.",
    evidence: "What a green light proves, and what it does not.",
  };
  return <>{TOPICS.filter(([key]) => key !== "week").map(([key, label]) => <UiSection key={key} id={key} title={label === "Proof" ? "What is proven?" : label === "In-event" ? "During the event" : label === "Simulation" ? "Simulations" : label} purpose={purpose[key]}><Fold label="Read the explanation"><Guide topic={key} doc={doc} /></Fold></UiSection>)}</>;
}

function Guide({ topic, doc }: { topic: Topic; doc: ExplainDoc | null }) {
  const [playerId, setPlayerId] = useState<number | null>(null);
  const players = doc?.players.filter(p => !p.withdrawn) ?? [];
  const player = players.find(p => p.id === playerId) ?? [...players].sort((a, b) => (b.mu_rel ?? -Infinity) - (a.mu_rel ?? -Infinity))[0];
  const picker = <label className="wt-player-picker">Inspect a golfer<select value={player?.id ?? ""} disabled={!players.length} onChange={e => setPlayerId(Number(e.target.value))}>{!players.length && <option value="">No published player data</option>}{players.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}</select></label>;
  if (topic === "skill") return <>
    <Panel title="Skill is a forecast of ability, not the last score"><p>The model asks how many strokes a golfer is expected to gain per round against this field. Higher is better. A +1.0 rating means one stroke better per round in expectation; it does not promise four strokes over a tournament.</p><Chain items={[["Adjust the history", "Compare past rounds on a common field-strength scale, using only history visible before the forecast."], ["Describe the golfer", "Blend long and short histories, category strengths, trends, activity and thin-history evidence."], ["Apply fitted weights", "The CHL ridge model combines signals while shrinking unstable coefficients."], ["Adapt to this week", "Add the current course, location, shot-signal, past wave-luck and eligible DataGolf lines."], ["Carry uncertainty", "Skill uncertainty and round-to-round volatility enter simulation separately."]]} /></Panel>
    <div className="wt-split"><Panel title="What the model can learn"><dl className="wt-families"><div><dt>Level and form</dt><dd>Decayed adjusted SG at different horizons: is the golfer good, and is recent play different?</dd></div><div><dt>Category skills and trends</dt><dd>Off the tee, approach, around the green and putting; state-space estimates distinguish signal from noisy rounds.</dd></div><div><dt>Thin records and cross-tour play</dt><dd>Short histories are pulled toward fitted priors. Other tours can inform a player with little PGA or DPWT history.</dd></div><div><dt>Course and situation</dt><dd>Venue fit and history, age, layoff, activity and location effects describe the context of the next round.</dd></div></dl><details><summary>How the weights are fitted</summary><p>CHL fills missing inputs and carries missing-data indicators, centers features within events, uses tour and history-band interactions, and standardizes using training seasons. Stored season coefficients are applied to the new event. Ridge penalizes large coefficients so correlated signals do not freely add up. A named contribution is a fitted association, not proof of a causal effect.</p></details></Panel><Panel title="This run’s actual skill ingredients">{picker}{player && doc ? <PlayerIngredients player={player} doc={doc} /> : <p>No matching published player values. The diagram above describes the recipe; it is not evidence that this week ran.</p>}</Panel></div>
    <Panel title="The last adjustments have different jobs"><div className="wt-notes"><p><b>Past wave luck</b> adjusts historical performance for favorable or unfavorable weather exposure. <b>This week’s weather</b> acts on the future forecast and tee times. They describe different rounds.</p><p><b>DataGolf blend</b> is an uncertainty-weighted pull toward a verified pre-event DataGolf skill snapshot for eligible PGA runs. Established players have zero raw blend weight; thin records can move more. Missing or ineligible snapshots leave a recorded fallback.</p><p><b>Owner edits</b> are named overrides. A zero line means no contribution in this artifact. A component appearing in the legend does not prove its data was available or its layer was active.</p></div><Link prefetch={false} href="/why-priced">Inspect individual price drivers</Link> · <Link prefetch={false} href="/inputs">Inspect model inputs</Link></Panel>
  </>;
  if (topic === "sim") return <>
    <Panel title="One forecast becomes many possible tournaments"><p>The simulator preserves uncertainty. The favorite can miss the cut, a weaker player can have a great week, and players share conditions. Every simulated tournament uses the same field and event rules.</p><Chain items={[["Set the scoring environment", "Course hole tables and scoring anchor set the level. Forecast weather changes expected scoring and exposure."], ["Draw a tournament world", "Player skill uncertainty, correlated week form, hot/cold days, round variation and shared conditions create one joint world."], ["Play holes, then apply rules", "Sample integer hole outcomes, sum scores, apply the cut, withdrawals and ties, and rank the survivors."], ["Count the outcomes", "Repeat across seeds: wins, top finishes, cuts and head-to-head results become probabilities."]]} /><div className="wt-equation"><b>Expected round score</b><span>= scoring environment − player skill</span><small>Illustration: a 72-stroke environment and +1.0 strokes of skill imply 71 expected strokes, before the relevant round adjustments. Real scores remain random.</small></div></Panel>
    <div className="wt-split"><Panel title="Why draws are linked"><div className="wt-latents"><div><strong>Tournament</strong><p>A golfer’s uncertain ability and week form persist across rounds.</p></div><div><strong>Round and course</strong><p>Shared conditions move players together. Rotation groups can share a course shock.</p></div><div><strong>Player and hole</strong><p>Day form and hole outcomes create birdies, pars, bogeys and tail scores.</p></div></div><p>The hole engine tilts each hole’s outcome probabilities toward the golfer’s target round score. This creates integer scores and asymmetric tails rather than just rounding a normal round score.</p><p className="wt-caption">The engine’s different uncertainty terms have different persistence. They are not all independent errors added anew on every hole.</p></Panel><Panel title="Read the published outcomes">{picker}{player && doc ? <PlayerOutcomes player={player} doc={doc} /> : <p>Outcome graphics appear when a matching published run is available.</p>}</Panel></div>
    <Panel title="Model probability and offered price are separate"><Chain items={[["Model", "Frequency or settlement-weighted result from simulated tournaments."], ["Market comparison", "Compare against available sportsbook consensus with its margin removed."], ["Fair price and decision", "The published fair probability may combine model and market information. Betting rules use their own eligibility and price checks."]]} /><p>A 10% model win chance is about one win in ten comparable forecasts, not a prediction that this golfer wins this event. Ties, dead heats and playoff treatment affect the specific market payout. Finish-distribution graphics may be reconstructed; their basis is shown below the chart.</p><p>Listed next-round matchups use the dedicated B8 pricing route. Do not assume a tournament win curve is the source of every round-market price.</p></Panel>
  </>;
  if (topic === "live") return <>
    <Panel title="Keep what happened; simulate what is left"><Chain items={[["Read the state", "Actual strokes, completed rounds, holes played and cut/withdrawal status are fixed observations."], ["Update the next round", "B8v2 learns from total and category residuals, wave luck and the next tee clock when available."], ["Update the remaining event", "Condition week form on observed rounds; reconcile the persistent B8 shift with the engine posterior so the update acts once."], ["Reprice and verify", "Restart the tournament from that state, apply contention and variance layers where enabled, then inspect new outputs."]]} /></Panel>
    <div className="wt-split"><Panel title="A good round is evidence, not a new permanent rating"><div className="wt-update"><span>Pre-event ability</span><b>+</b><span>Shrunk evidence from rounds played</span><b>=</b><strong>Updated forecast</strong></div><p>B8 uses category-specific weights. The next-round mean and its persistence into later rounds are different quantities. Tee-time and wave effects can be next-round-only; a missing tee sheet is a neutral clock fallback, not a known advantage.</p><p>The tournament engine conditions its week-form distribution on results. Weekend scaling, bounce/state-variance and contention adjustments follow their enabled code paths. The stored run is the authority on what was applied.</p></Panel><Panel title="What changes in this published state">{doc ? <dl><div><dt>Run type</dt><dd>{doc.kind === "live" ? `In-event, after round ${doc.after_round ?? "unknown"}` : "Pre-tournament; no rounds observed yet"}</dd></div><div><dt>Weather layer</dt><dd>{doc.model.weather_layer ?? "Not reported"}</dd></div><div><dt>Owner overrides</dt><dd>{doc.model.overrides_applied.length ? doc.model.overrides_applied.join(", ") : "None reported"}</dd></div><div><dt>State timestamp</dt><dd>{fmt(doc.as_of)}</dd></div></dl> : <p>No matching state artifact.</p>}<p className="wt-notice">Mid-round pricing has a limitation: strokes-to-date and hole progress are used, but the restart does not currently model the exact lie and distance of an unfinished shot. Hole-10 starts can use an approximate partial-round par conversion.</p><Link prefetch={false} href="/why-priced">Open the published live player breakdown</Link></Panel></div>
  </>;
  return <>
    <Panel title="A successful run and an accurate model are different claims"><div className="wt-evidence-claims"><div><h3>Operational proof</h3><p>The weekly lights require current, event-scoped evidence that outputs passed content checks. Missing receipts stay yellow. A process exit code or an object existing is not enough.</p></div><div><h3>Predictive evidence</h3><p>Skill and probability accuracy require forecasts made before outcomes, comparisons with a baseline and the market, and uncertainty across whole events. Green workflow lights do not establish an edge.</p></div></div></Panel>
    <Panel title="How to judge a model change"><ol className="wt-checklist"><li>Freeze the hypothesis and baseline before testing. Use only information available at forecast time.</li><li>Test future seasons or events that were not used to choose the change. Keep golfers in the same event together when estimating uncertainty.</li><li>Check strong-player and thin-history behavior, probability calibration and tail outcomes, alongside average scoring error.</li><li>Compare with both the opening and closing market where available. A proxy replay with missing weather or shifts cannot prove production betting edge.</li><li>Require a reviewed promotion, then monitor forward forecasts and the live outputs. Retired comparison models are not part of routine production runs.</li></ol><Link prefetch={false} href="/performance">Inspect the model scorecard and results</Link></Panel>
    <Panel title="Recipe and evidence sources"><p>This guide describes the current production paths, inspected November 3, 2026. Actual component values, probabilities, run identity and weather-layer labels come from the selected published explanation. Operational status comes from the existing owner workflow API, refreshed every 30 seconds while visible. No new pricing or research run is started by this page.</p><details><summary>Source map and limits</summary><dl><div><dt>Skills and adjustments</dt><dd>engine/live/chl_live.py; golfprice/week.py; dg_blend.py; v23.py; v27_wave.py</dd></div><div><dt>Tournament simulation</dt><dd>engine/sim/holesim_m8.py; golfprice/week_arms.py; course and weather dispatch</dd></div><div><dt>In-event and round markets</dt><dd>golfprice/live/engine.py; b8v2/update_v2.py; b8v2/livewire.py</dd></div><div><dt>Stage verification</dt><dd>app/ops-rules.ts and worker/ops-api.ts. Inputs and delivery currently lack an independently connected expected occurrence; this guide leaves them unverified.</dd></div></dl><p>Recipe documentation can lag a future release. Use the artifact’s named components and the stored run manifest to establish exact active coefficients, flags, snapshots and fallback reasons.</p></details></Panel>
  </>;
}

function PlayerIngredients({ player, doc }: { player: ExPlayer; doc: ExplainDoc }) {
  const parts = Object.entries(player.components).filter(([, value]) => Number.isFinite(value));
  const max = Math.max(0.01, ...parts.map(([, v]) => Math.abs(v)));
  return <><div className="wt-player-facts"><span><b>{player.mu_rel == null ? "Unavailable" : signed(player.mu_rel, 2)}</b> SG/round vs field</span><span><b>{player.n_prior_rounds ?? "Unknown"}</b> prior rounds</span><span><b>{number(player.sd)}</b> round SD input</span></div><p className="wt-caption">Published component contributions, strokes per round. Bars share a zero at the center; their signs describe this artifact. These grouped, rounded lines are not causal effects or a fresh model calculation.</p><div className="wt-contributions">{parts.map(([key, value]) => <div key={key}><span>{doc.components_legend[key]?.label ?? key}</span><span className="wt-bar-track" aria-hidden="true"><i className={value < 0 ? "neg" : "pos"} style={{ width: `${Math.abs(value) / max * 50}%`, left: value < 0 ? `${50 - Math.abs(value) / max * 50}%` : "50%" }} /></span><b>{signed(value, 3)}</b></div>)}</div><p className="wt-caption">Positive skill lowers expected strokes. SD measures spread, not a skill bonus. Future weather can also act on scoring separately from these skill lines.</p></>;
}

function PlayerOutcomes({ player, doc }: { player: ExPlayer; doc: ExplainDoc }) {
  const bins = finishBuckets(player.finish?.pos), max = Math.max(0.01, ...bins.map(b => b.p));
  return <><p className="wt-caption">Reported simulation count: {doc.model.n_sims?.toLocaleString() ?? "not reported"}. This is the run’s setting; retained outcome draws and seed totals can differ.</p><div className="wt-probabilities">{(["win", "top_5", "top_10", "top_20", "make_cut"] as const).map(key => <div key={key}><span>{({win:"Win",top_5:"Top 5",top_10:"Top 10",top_20:"Top 20",make_cut:"Make cut"})[key]}</span><b>{pct(player.probs[key]?.model)}</b><span>model</span></div>)}</div>{bins.length > 0 && <figure className="wt-finish"><figcaption>Where the whole finish distribution sits</figcaption><div className="wt-histogram" role="img" aria-label={`Published finish distribution for ${player.name}, grouped in ten-place bins. Median ${player.finish?.median}; tenth to ninetieth percentile ${player.finish?.p10} to ${player.finish?.p90}.`}>{bins.map(b => <div key={b.label}><span style={{ height: `${b.p / max * 100}%` }} title={`${b.label}: ${pct(b.p)}`} /><small>{b.label}</small></div>)}</div><p className="wt-caption">Finish-position probability mass; equal ten-place bins, with a shorter final bin when needed. Basis: {doc.finish.method}. {doc.finish.reason}</p><details><summary>Read exact distribution values</summary><dl>{bins.map(b => <div key={b.label}><dt>Places {b.label}</dt><dd>{pct(b.p)}</dd></div>)}</dl></details></figure>}<p>{player.finish ? `Middle finish: ${player.finish.median}. The 10th–90th percentile range is ${player.finish.p10}–${player.finish.p90}.` : "Finish range not published."} A finish curve and settlement-weighted top-N chances can differ because of ties and reconstruction.</p></>;
}
