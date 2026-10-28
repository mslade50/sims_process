"use client";

import { useMemo, useState } from "react";
import { Panel } from "./components";
import { useDashboardData } from "./data";
import "./course-geometry.css";
import { DataThrough } from "./DataThrough";
import { dataThrough } from "./freshness-rules";
import { orderEvents } from "./explain-rules";
import { profileHref, selectWeeklyEvent } from "./player-profile-rules";
import {
  BADGE_NOTE, FLAG_HELP, GEOMETRY_INDEX_KEY, badgeMeta, cardView, fitLists, hasNoisy, parseCard, parseFit, parseGeoIndex, parseReference, parseTraits, playerCoverage,
  courseLabel, fitKeyFor, pickVenue, playedOverText, reliableTraits, strokesText, traitValueText, venuesForEvent, visibleBadges,
  type GeoIndex,
  type BarRow, type Badge, type FitEntry, type Fit, type Reference,
} from "./course-geometry-rules";

/** Tap-to-reveal definition: a real button (works by touch and keyboard), never a hover-only tooltip. */
function Def({ label, children }: { label: string; children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <span className="cg-def">
      <button type="button" className="cg-def-btn" aria-expanded={open} aria-label={`${open ? "Hide" : "Show"} definition: ${label}`} onClick={() => setOpen(!open)}>?</button>
      {open && <span className="cg-def-text" role="note">{children}</span>}
    </span>
  );
}

function EvidenceTag({ grade, legend }: { grade: string; legend?: string }) {
  return <span className={`cg-grade cg-grade-${grade}`} title={legend}>{grade}</span>;
}

function Bar({ row, grades }: { row: BarRow; grades: Record<string, string> }) {
  const flag = row.flag ? FLAG_HELP[row.flag] : null;
  return (
    <li className={`cg-row${row.muted ? " cg-muted" : ""}`}>
      <div className="cg-row-head">
        <span className="cg-label">{row.label}</span>
        <span className="cg-tags">
          <EvidenceTag grade={row.evidence} legend={grades[row.evidence]} />
          {flag && <span className={`cg-flag cg-flag-${row.flag}`}>{flag.tag}</span>}
          <Def label={row.label}>
            <b>{row.ordinal}.</b> Evidence {row.evidence}{grades[row.evidence] ? `: ${grades[row.evidence]}` : ""}.{flag ? ` ${flag.help}` : ""} Based on {row.editions > 0 ? `${row.editions} edition${row.editions === 1 ? "" : "s"}` : "a fixed course property"}.
          </Def>
        </span>
      </div>
      <div className="cg-track" role="img" aria-label={`${row.label}: ${row.ordinal}`}>
        <span className="cg-fill" style={{ width: `${Math.max(1, row.pct)}%` }} />
        <span className="cg-mid" />
      </div>
      <p className="cg-text">{row.text}</p>
    </li>
  );
}

function BadgeChip({ b, fit }: { b: Badge; fit: Fit }) {
  const stroke = strokesText(b);
  const pair = fit.pairs[b.pair];
  return (
    <span className={`cg-badge cg-badge-${b.kind}${b.grade === "C" ? " cg-badge-c" : ""}`}>
      <span className="cg-badge-label">{b.label}</span>
      <span className="cg-badge-meta">{badgeMeta(b)}{stroke ? ` · ${stroke}` : ""}</span>
      {pair && <Def label={b.label}><b>{pair.label}.</b> {pair.why}</Def>}
    </span>
  );
}

function Legend({ reference }: { reference: Reference | null }) {
  const g = reference?.grades ?? {};
  const keys = ["A", "B", "C"].filter((k) => g[k]);
  return keys.length ? <p className="cg-legend">{keys.map((k) => <span key={k}><EvidenceTag grade={k} /> {g[k]}</span>)}</p> : null;
}

function NoisyToggle({ on, onChange }: { on: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="cg-toggle">
      <input type="checkbox" checked={on} onChange={(e) => onChange(e.target.checked)} />
      <span>Show noisy items (grade C)</span>
    </label>
  );
}

function FitSection({ title, entries, total, fit }: { title: string; entries: FitEntry[]; total: number; fit: Fit }) {
  return (
    <section className="cg-fitcol" aria-label={title}>
      <h3>{title} <small>{total}</small></h3>
      {entries.length === 0 ? <p className="cg-text">No one clears the bar.</p> : (
        <ul>{entries.map((e) => (
          <li key={e.dg_id}>
            <a className="cg-name" href={profileHref(e.dg_id)}>{e.name}</a>
            <BadgeChip b={e.badge} fit={fit} />
            {e.others > 0 && <small className="cg-more">+{e.others} more on the profile</small>}
          </li>
        ))}</ul>
      )}
    </section>
  );
}

function FitLists({ fit, reference, eventName }: { fit: Fit; reference: Reference | null; eventName: string }) {
  const [noisy, setNoisy] = useState(false);
  const lists = useMemo(() => fitLists(fit, noisy), [fit, noisy]);
  const noisyAvailable = hasNoisy(fit);
  return (
    <div className="cg-fit">
      <div className="cg-fit-head">
        <h3 className="cg-fit-title">Who fits, who doesn&apos;t <span className="cg-pill">{BADGE_NOTE}</span></h3>
        {noisyAvailable && <NoisyToggle on={noisy} onChange={setNoisy} />}
      </div>
      <p className="cg-text">{playerCoverage(fit)} Player traits against the geometry of {eventName}; the model prices none of this.</p>
      <div className="cg-fitgrid">
        <FitSection title="Who fits" entries={lists.fits} total={lists.nFits} fit={fit} />
        <FitSection title="Who doesn't" entries={lists.misfits} total={lists.nMisfits} fit={fit} />
      </div>
      <Legend reference={reference} />
    </div>
  );
}

/** Segmented control over the courses of a multi-course event. Wraps on a phone instead of scrolling sideways. */
function CourseSwitcher({ idx, uids, value, onChange }: { idx: GeoIndex; uids: string[]; value: string; onChange: (uid: string) => void }) {
  return (
    <div className="cg-switch">
      <p className="cg-switch-title">{playedOverText(uids.length)}</p>
      <div className="cg-tabs" role="tablist" aria-label={playedOverText(uids.length)}>
        {uids.map((u) => (
          <button key={u} type="button" role="tab" aria-selected={u === value} className={`cg-tab${u === value ? " cg-tab-on" : ""}`} onClick={() => onChange(u)}>{courseLabel(idx, u)}</button>
        ))}
      </div>
    </div>
  );
}

/** "Course geometry" card for the current event's course(s). Renders nothing while loading or when the pointer, the venue or the payload is missing. */
export function CourseGeometry({ eventUid, eventName }: { eventUid: string | null | undefined; eventName?: string | null }) {
  const indexLoad = useDashboardData<unknown>(GEOMETRY_INDEX_KEY);
  const idx = useMemo(() => parseGeoIndex(indexLoad.data), [indexLoad.data]);
  const uids = useMemo(() => venuesForEvent(idx, eventUid), [idx, eventUid]);
  const [chosen, setChosen] = useState<string | null>(null);
  const uid = pickVenue(uids, chosen);
  if (!idx || !uid) return null;
  const switcher = uids.length > 1 ? <CourseSwitcher idx={idx} uids={uids} value={uid} onChange={setChosen} /> : null;
  return <CourseGeometryBody key={uid} idx={idx} uid={uid} eventName={eventName} switcher={switcher} />;
}

function CourseGeometryBody({ idx, uid, eventName, switcher }: { idx: GeoIndex; uid: string; eventName?: string | null; switcher: React.ReactNode }) {
  const entry = idx.venues[uid];
  const cardLoad = useDashboardData<unknown>(entry?.key ?? null);
  const card = useMemo(() => (cardLoad.data ? parseCard(cardLoad.data, uid) : null), [cardLoad.data, uid]);
  const fitKey = fitKeyFor(idx, uid);
  const fitLoad = useDashboardData<unknown>(card ? fitKey : null);
  const fit = useMemo(() => (fitLoad.data ? parseFit(fitLoad.data, uid) : null), [fitLoad.data, uid]);
  const refLoad = useDashboardData<unknown>(card ? idx.reference_key : null);
  const reference = useMemo(() => parseReference(refLoad.data), [refLoad.data]);
  const [more, setMore] = useState(false);

  if (!card) return switcher ? <Panel title="Course geometry" eyebrow="Several courses" className="cg-panel">{switcher}<p className="cg-text">No geometry card is published for this course.</p></Panel> : null;
  const view = cardView(card);
  const grades = reference?.grades ?? {};
  return (
    <Panel title="Course geometry" eyebrow={`${card.course_name || entry.course_name || card.event_name || eventName} · ${view.summary}`} className="cg-panel">
      {switcher}
      <DataThrough text={[dataThrough(card.as_of, "Course geometry built"), card.seasons.length ? `latest edition ${Math.max(...card.seasons)}` : null].filter(Boolean).join(" · ") || null} />
      <p className="cg-descriptive">{idx.descriptive}</p>
      {view.fallback ? (
        <p className="cg-fallback" role="note">{card.note ?? "No shot coordinates were published for this event, so only yardage is shown."}</p>
      ) : null}
      {(card.par || card.yardage) && <p className="cg-text">{[card.par ? `Par ${card.par}` : "", card.yardage ? `${Math.round(card.yardage).toLocaleString("en-US")} yd` : ""].filter(Boolean).join(" · ")}, latest edition. Bars show where this venue sits among {Object.keys(idx.venues).length} venues; the tick is the 50th percentile.</p>}
      <ul className="cg-rows">{view.main.map((r) => <Bar key={r.key} row={r} grades={grades} />)}</ul>
      {view.more.length > 0 && (
        <>
          <button type="button" className="cg-morebtn" aria-expanded={more} onClick={() => setMore(!more)}>{more ? "Hide" : "Show"} more measures</button>
          {more && <ul className="cg-rows">{view.more.map((r) => <Bar key={r.key} row={r} grades={grades} />)}</ul>}
        </>
      )}
      {fit && fit.players.length > 0 && <FitLists fit={fit} reference={reference} eventName={card.course_name || card.event_name || String(eventName)} />}
    </Panel>
  );
}

/** Player page: this golfer's reliable traits and fit badges for the event's venue. Hidden when nothing is published for the player. */
export function PlayerCourseFit({ dgId, selectedEvent }: { dgId: number | null | undefined; selectedEvent?: string | null }) {
  const eventsLoad = useDashboardData<{ events?: Array<{ event_uid: string; name?: string | null; tour?: string; date_start?: string }> }>("golfprice/index.json");
  const events = useMemo(() => orderEvents((eventsLoad.data?.events ?? []) as never[]) as unknown as Array<{ event_uid: string; name?: string | null; tour?: string; date_start?: string }>, [eventsLoad.data]);
  const event = selectWeeklyEvent(events, selectedEvent ?? null, null);
  const indexLoad = useDashboardData<unknown>(GEOMETRY_INDEX_KEY);
  const idx = useMemo(() => parseGeoIndex(indexLoad.data), [indexLoad.data]);
  const uids = useMemo(() => venuesForEvent(idx, event?.event_uid), [idx, event?.event_uid]);
  const [chosen, setChosen] = useState<string | null>(null);
  const uid = pickVenue(uids, chosen);
  const fitKey = fitKeyFor(idx, uid);
  const fitLoad = useDashboardData<unknown>(fitKey);
  const fit = useMemo(() => (uid && fitLoad.data ? parseFit(fitLoad.data, uid) : null), [fitLoad.data, uid]);
  const refLoad = useDashboardData<unknown>(idx && uid ? idx.reference_key : null);
  const reference = useMemo(() => parseReference(refLoad.data), [refLoad.data]);
  const inFit = fit?.players.find((p) => p.dg_id === dgId) ?? null;
  const traitsLoad = useDashboardData<unknown>(idx && uid && fit && !inFit ? idx.traits_key : null);
  const traitsDoc = useMemo(() => parseTraits(traitsLoad.data), [traitsLoad.data]);
  const [noisy, setNoisy] = useState(false);

  if (!idx || !uid || !fit || !reference || !dgId) return null;
  const fallbackTraits = !inFit ? traitsDoc?.players[String(dgId)]?.t : undefined;
  const traits = inFit ? inFit.traits : fallbackTraits;
  if (!inFit && !traits) return null;
  const { rows, withheld } = reliableTraits(traits ?? {}, reference);
  const badges = inFit ? visibleBadges(inFit, noisy) : [];
  const noisyAvailable = !!inFit && [...inFit.badges, ...inFit.hidden_badges].some((b) => b.grade === "C" || b.hidden);
  if (!badges.length && !rows.length && !noisyAvailable) return null;
  return (
    <Panel title="Course fit" eyebrow={`${fit.event_name} · ${BADGE_NOTE}`} className="cg-panel">
      {uids.length > 1 && <CourseSwitcher idx={idx} uids={uids} value={uid} onChange={setChosen} />}
      <p className="cg-descriptive">{fit.descriptive}</p>
      {inFit ? (
        <div className="cg-fit-head">
          <h3 className="cg-fit-title">Fit and misfit badges</h3>
          {noisyAvailable && <NoisyToggle on={noisy} onChange={setNoisy} />}
        </div>
      ) : <p className="cg-text">This golfer is not on the published fit list for this venue, so no badges are shown; reliable traits are below.</p>}
      {inFit && (badges.length ? <div className="cg-badges">{badges.map((b) => <BadgeChip key={b.pair} b={b} fit={fit} />)}</div> : <p className="cg-text">No fit or misfit badge for this golfer at this venue.</p>)}
      {rows.length > 0 && (
        <>
          <h3 className="cg-fit-title">Reliable traits</h3>
          <ul className="cg-traits">{rows.map((r) => (
            <li key={r.key}><span className="cg-label">{r.label}</span><span className="cg-traitval">{traitValueText(r)}</span><small>n {r.n.toLocaleString("en-US")}</small></li>
          ))}</ul>
        </>
      )}
      {withheld > 0 && <p className="cg-text">{withheld} trait{withheld === 1 ? "" : "s"} withheld: not enough data.</p>}
      <p className="cg-text">Descriptive, not priced: the model does not use any of this in a price.</p>
      <Legend reference={reference} />
    </Panel>
  );
}
