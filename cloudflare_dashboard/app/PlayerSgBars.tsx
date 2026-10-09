"use client";

import { Skeleton } from "./ui";
import type { DeepProfile } from "./player-deep-rules";
import { PROFILE_AXES, type PlayerProfile } from "./player-profile-rules";
import { HISTORY_DEFAULTS, filteredSkillProfile, sliceHistory, type HistoryFilters, type ObservedCategoryReference } from "./player-history-filters";
import { noCategoryReason, sgBarRows, signed } from "./player-hero-rules";

const NAMES: Record<string, string> = { sg_ott: "Off the tee", sg_app: "Approach", sg_arg: "Around green", sg_putt: "Putting" };
/** Skill-type numbers default to R1-R2 (A4). The shared history filters are a separate, visible control. */
export const SG_BAR_FILTERS: HistoryFilters = { ...HISTORY_DEFAULTS, rounds: [1, 2] };

function hasReference(ref: ObservedCategoryReference | undefined): boolean {
  if (ref?.schema_version !== "observed_category_reference.v1") return false;
  return PROFILE_AXES.some(k => { const a = ref.axes?.[k]; return Number.isFinite(a?.mean) && Number.isFinite(a?.sd) && (a?.sd ?? 0) > 0; });
}

export function PlayerSgBars({ profile, deep, loading, categoryReference, tours }: { profile: PlayerProfile; deep: DeepProfile | null; loading: boolean; categoryReference?: ObservedCategoryReference; tours?: string[] }) {
  if (loading) return <section className="ph-sg" data-testid="sg-skeleton"><span className="eyebrow">Strokes gained by category</span><Skeleton lines={4} height={12} /></section>;
  const events = deep?.history ? sliceHistory(deep.history.events, { ...SG_BAR_FILTERS, asOf: deep.as_of }) : [];
  const own = filteredSkillProfile(profile, events, categoryReference);
  const { rows, available } = sgBarRows(PROFILE_AXES.map(key => {
    const m = own.radar.find(v => v.key === key);
    return { key, label: NAMES[key], value: m?.value ?? null, z: m?.z ?? null, n: m?.n ?? 0 };
  }));
  return (
    <section className="ph-sg" data-testid="sg-bars" aria-label="Strokes gained by category">
      <div className="ph-sg-head">
        <div><span className="eyebrow">Strokes gained by category</span><h3>Off the tee, approach, around the green and putting vs the PGA average</h3></div>
        <span className="ph-muted">Rounds 1 and 2 over the last 2 years, compared with the PGA average of the past 3 years. Plain averages, not shrunk.</span>
      </div>
      {!available ? (
        <p className="ph-na" data-testid="sg-na" role="status">{noCategoryReason(tours, { hasReference: hasReference(categoryReference), deepLoaded: Boolean(deep), minRounds: 3 })}.</p>
      ) : (
        <ul className="ph-sg-rows">
          {rows.map(r => (
            <li key={r.key} data-status={r.status}>
              <span className="ph-sg-label">{r.label}</span>
              <span className="ph-sg-track" role="img" aria-label={r.status === "ok" ? `${r.label}: ${signed(r.z, 1)} standard deviations vs the PGA average` : `${r.label}: not enough category rounds`}>
                <i className="ph-sg-zero" style={{ left: `${r.zero}%` }} />
                {r.status === "ok" && <i className="ph-sg-fill" data-sign={(r.z ?? 0) >= 0 ? "pos" : "neg"} style={{ left: `${r.left}%`, width: `${r.width}%` }} />}
              </span>
              <span className="ph-sg-value">{r.status === "ok" ? <><b>{signed(r.value)}</b> SG/round · <span title="How many standard deviations above (+) or below (-) the typical PGA player. Larger numbers are further from typical.">{signed(r.z, 1)} SD</span></> : r.status === "thin" ? <span className="ph-muted">{r.n} rounds (needs 3)</span> : <span className="ph-muted" title="No rounds with strokes-gained categories in this window">No data</span>}</span>
              {r.status === "ok" ? <small>{r.n} rounds</small> : <small />}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
