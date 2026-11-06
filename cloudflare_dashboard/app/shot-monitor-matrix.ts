import { timestamp, type Source, type Tour } from "./shot-monitor-rules.ts";
import { tourSummary, type Stage } from "./shot-monitor-summary.ts";

/**
 * One row of the tour matrix: when each stage last happened, whether the status is known, why, and the single next action.
 * Presentation only: every state comes from tourSummary and the existing status rules, nothing is re-derived here.
 * Collect describes the machine (is a collector running); Check and Publish describe verification evidence (did published output validate).
 */
export type MatrixCell = { state: string; tone: Stage["tone"]; at: string | null };
export type MatrixRow = { tour: Tour; badge: string; attention: boolean; known: boolean; collect: MatrixCell; check: MatrixCell; publish: MatrixCell; reason: string; available: string; next: string };

const latest = (values: Array<string | null | undefined>): string | null => {
  const valid = values.filter((v): v is string => typeof v === "string" && Number.isFinite(timestamp(v)));
  return valid.length ? valid.reduce((a, b) => (timestamp(b) > timestamp(a) ? b : a)) : null;
};

/** Who and where for each next action, so no row ends in a bare instruction. Suffix only; the action text itself comes from tourSummary. */
const WHERE: Array<[RegExp, (primary: string) => string]> = [
  [/^Restore the approved publishing code/, () => " Owner: redeploy the approved publish code to the always-on desktop (the GolfShotRoundPublish task), then rerun that task."],
  [/^Check the publishing job failure/, () => " Owner: open the GolfShotRoundPublish task log on the always-on desktop; the reason is in the details below."],
  [/^Check the missing capture status/, (p) => ` Owner: confirm ${p} is scheduled on the always-on desktop; it has not reported.`],
  [/^Check the collection problem/, (p) => ` Owner: open ${p} on the always-on desktop and read its last error (shown in this tour's details).`],
  [/^Resolve the held data checks/, () => " Owner: read the validation limits in this tour's published datasets below; corrections are made in the publish job, not here."],
  [/^Review the reported job issues/, () => " Open this tour's details to see which collector reported the issue."],
  [/^Review the dataset validation limits/, () => " Open this tour's details, Published datasets."],
];
const PRIMARY: Record<string, string> = { pga: "GolfRichShotCollector", dpwt: "SimPrep-DPWT-forward", kft: "SimPrep-KFT-forward" };

export function tourMatrix(tour: Tour, sources: Source[], now: number): MatrixRow {
  const summary = tourSummary(tour, sources, now);
  const collectors = sources.find((s) => s.file === "collectors-desktop.json")?.data?.collectors?.filter((r) => r.tour === tour) ?? [];
  const datasets = sources.find((s) => s.file === "publications.json")?.data?.publications?.filter((r) => r.tour === tour) ?? [];
  const [collect, check, save] = summary.stages;
  const known = !["Check status", "Checking"].includes(summary.badge) && collect.state !== "Unknown";
  return {
    tour,
    badge: summary.badge,
    attention: summary.attention,
    known,
    collect: { state: collect.state, tone: collect.tone, at: latest(collectors.map((r) => r.last_success_at)) },
    check: { state: check.state, tone: check.tone, at: latest(datasets.map((r) => r.validation.checked_at)) },
    publish: { state: save.state, tone: save.tone, at: latest(datasets.map((r) => r.published_at)) },
    reason: known ? summary.headline : `Unknown: ${summary.headline}`,
    available: summary.available,
    next: summary.next ? summary.next + (WHERE.find(([re]) => re.test(summary.next!))?.[1](PRIMARY[tour]) ?? "") : "None needed.",
  };
}
