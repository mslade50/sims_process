"use client";

import { useEffect, useState } from "react";
import { EmptyState, Panel } from "./components";
import { etTime } from "./lib";
import { FRESHNESS_STATUS_KEY, CHECKER_MAX_AGE_MS, fmtAge, parseFreshness, type FreshnessDoc } from "../worker/freshness-rules";
import "./freshness.css";

const URL_PATH = `/api/owner-data/${FRESHNESS_STATUS_KEY.replace(/^owner\//, "")}`;
const STATE_WORDS: Record<string, string> = { ok: "Current", stale: "Stale", missing: "Not published", unreadable: "Could not check", skipped: "Not judged now" };

/** Run page: what the 15-minute data-staleness check last found (owner/data_freshness.json, written by worker/freshness.ts). */
export function DataFreshnessPanel() {
  const [state, setState] = useState<{ phase: "loading" | "ready" | "none" | "error"; doc: FreshnessDoc | null }>({ phase: "loading", doc: null });
  const [now] = useState(() => Date.now());
  useEffect(() => {
    let live = true;
    fetch(URL_PATH, { cache: "no-store", headers: { accept: "application/json" } })
      .then(async (response) => {
        if (response.status === 404) return { phase: "none" as const, doc: null };
        if (!response.ok) return { phase: "error" as const, doc: null };
        const doc = parseFreshness(await response.json().catch(() => null));
        return doc ? { phase: "ready" as const, doc } : { phase: "error" as const, doc: null };
      })
      .catch(() => ({ phase: "error" as const, doc: null }))
      .then((next) => { if (live) setState(next); });
    return () => { live = false; };
  }, []);
  const { doc } = state;
  const checkedMs = doc ? Date.parse(doc.checked_at) : NaN;
  const checkerLate = doc && (!Number.isFinite(checkedMs) || now - checkedMs > CHECKER_MAX_AGE_MS);
  return (
    <Panel eyebrow="Data freshness" title="Is every dashboard data source current?" className="fresh-panel">
      {state.phase === "loading" ? <p className="inputs-muted">Loading the latest freshness check…</p>
        : state.phase === "none" ? <EmptyState title="No freshness check recorded yet" detail="The check runs every 15 minutes once it is deployed. If this stays empty, the check is not running." />
        : state.phase === "error" || !doc ? <EmptyState title="The freshness check could not be read" detail="Reload in a minute. If it keeps failing, the status object or the check itself is broken." />
        : <>
          <p className={doc.ok && !doc.not_judged && !checkerLate ? "data-through" : "data-through warn"} role="status" data-testid="fresh-summary">
            {checkerLate ? `The check last ran ${etTime(doc.checked_at)}, more than 45 minutes ago: it may not be running. ` : ""}
            {doc.not_judged ? "Not judged: no source could be checked this tick (off week, or the reads failed). This is not the same as healthy." : doc.ok ? "Every source checked is within its normal age." : `${doc.stale_count} source${doc.stale_count === 1 ? " is" : "s are"} stale or missing. This panel is the record; nothing is sent to your phone.`} Checked {etTime(doc.checked_at)}.
          </p>
          <table>
            <thead><tr><th>Source</th><th>State</th><th>Last updated</th><th>Normal maximum</th></tr></thead>
            <tbody>{doc.sources.map((s) => (
              <tr key={s.id} title={s.detail}>
                <td>{s.label}</td>
                <td className={s.state === "stale" || s.state === "missing" ? "fresh-stale" : s.state === "ok" ? "fresh-ok" : undefined}>{STATE_WORDS[s.state] ?? s.state}</td>
                <td>{s.observed_at ? `${etTime(s.observed_at)}${s.age_hours !== null ? ` (${fmtAge(s.age_hours)} ago)` : ""}` : "unknown"}</td>
                <td>{s.limit_hours !== null ? fmtAge(s.limit_hours) : "n/a"}</td>
              </tr>
            ))}</tbody>
          </table>
          {doc.notes.length > 0 && <p className="inputs-muted">{doc.notes.join(" · ")}</p>}
          {!doc.inbox.bound && <p className="inputs-muted">Findings are shown here only; routing to the repair inbox is switched off until the watcher side is ready.</p>}
        </>}
    </Panel>
  );
}
