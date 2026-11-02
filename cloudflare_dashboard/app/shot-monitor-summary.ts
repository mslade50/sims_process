import { collectorStatus, publicationStatus, sourceFreshness, timestamp, type Source, type Tour } from "./shot-monitor-rules.ts";

export type Stage = { label: string; state: string; tone: "good" | "warn" | "bad" | "neutral" };
export type TourSummary = { tour: Tour; badge: string; headline: string; available: string; next?: string; attention: boolean; stages: Stage[] };
const primaryIds: Record<Tour, string> = { pga: "GolfRichShotCollector", dpwt: "SimPrep-DPWT-forward", kft: "SimPrep-KFT-forward" };

/** Presentation only. All health/readiness decisions consume the existing status rules. */
export function tourSummary(tour: Tour, sources: Source[], now: number): TourSummary {
  const capture = sources.find(s => s.file === "collectors-desktop.json");
  const published = sources.find(s => s.file === "publications.json");
  const rows = capture?.data?.collectors?.filter(r => r.tour === tour) ?? [];
  const datasets = published?.data?.publications?.filter(r => r.tour === tour) ?? [];
  const primary = rows.find(r => r.id === primaryIds[tour]) ?? (rows.length === 1 ? rows[0] : undefined);
  const base: TourSummary = { tour, badge: "Unknown", headline: "Current status is unknown.", available: "No checked data confirmed.", next: "Refresh status.", attention: true,
    stages: ["Collect", "Check", "Save"].map(label => ({ label, state: "Unknown", tone: "neutral" })) };
  if (![capture, published].every(s => s && s.status === "ready" && sourceFreshness(s, now) === "fresh")) {
    const loading = [capture, published].some(s => s?.status === "loading");
    return { ...base, badge: loading ? "Checking" : "Check status", headline: loading ? "Checking the latest snapshots…" : "Cannot check right now.", available: "Data status is unknown." };
  }
  const states = rows.map(r => collectorStatus(r, capture!, now));
  if (states.some(s => !["running", "idle", "paused", "error", "unknown"].includes(s))) return { ...base, badge: "Check status", headline: "Collector checks are out of date.", available: "Data status is unknown." };
  const status = primary ? collectorStatus(primary, capture!, now) : "unknown";
  const ready = datasets.filter(r => publicationStatus(r, published!, now) === "Ready");
  const held = datasets.some(r => publicationStatus(r, published!, now) === "held");
  const partial = datasets.some(r => publicationStatus(r, published!, now) === "partial" && Number.isFinite(timestamp(r.published_at)) && timestamp(r.published_at) <= now + 5 * 60_000);
  const allReady = datasets.length > 0 && ready.length === datasets.length;
  const publisher = rows.find(r => r.id === "GolfShotRoundPublish") ?? rows.find(r => /publish|publication/i.test(r.label));
  const publishError = !!publisher && collectorStatus(publisher, capture!, now) === "error";
  const codeMismatch = publishError && /source hash mismatch|approved.*code.*(?:drift|mismatch)|(?:drift|mismatch).*approved.*code/i.test([publisher.reason, ...publisher.issues ?? []].join(" "));
  const rowAttention = rows.some((r, i) => states[i] === "error" || states[i] === "unknown" || !!r.issues?.length);
  const paused = status === "paused";
  const attention = rowAttention || !allReady || status === "unknown";
  let available = allReady ? "Checked datasets are available within their listed validation limits." : ready.length ? "Some checked datasets are available; other data remains unverified." : partial ? "Previously published data, with limits." : "No checked dataset is confirmed available.";
  if (ready.some(r => r.id === "kft-quality")) available = "Checked historical shot measurements.";
  if (held) available = tour === "dpwt" && datasets.some(r => r.id === "dpwt-official-holes" && publicationStatus(r, published!, now) === "held") && datasets.some(r => r.id === "dpwt-rich-inputs" && !r.published_at)
    ? "Published hole scores need corrections. New shot data is not ready."
    : "Some published data is held for corrections; check the validation limits.";
  let headline = status === "unknown" ? "Current capture status is unknown." : paused ? (ready.length ? "Paused. Historical data is available." : "Paused. Data needs checking.") : status === "error" ? "Collection needs attention." : publishError ? "Capture works. Publishing is blocked." : allReady ? (status === "running" ? "Collecting. Published checks pass." : "Scheduled. Published checks pass.") : "Capture works. Data needs checking.";
  if (rowAttention && !publishError && ["running", "idle"].includes(status)) headline = "Capture reported. A job needs attention.";
  const next = !attention ? undefined : codeMismatch ? "Restore the approved publishing code." : publishError ? "Check the publishing job failure." : status === "unknown" ? "Check the missing capture status." : status === "error" ? "Check the collection problem before publishing." : held ? "Resolve the held data checks." : rowAttention ? "Review the reported job issues." : "Review the dataset validation limits.";
  return { tour, badge: attention ? "Needs attention" : paused ? "Paused" : "Checks clear", headline, available, next, attention,
    stages: [
      { label: "Collect", state: status === "running" ? "Running" : status === "idle" ? "Scheduled" : paused ? "Paused" : status === "error" ? "Needs check" : "Unknown", tone: ["running", "idle"].includes(status) ? "good" : status === "error" ? "bad" : "neutral" },
      { label: "Check", state: allReady ? "Passed" : held ? "On hold" : partial ? "Some checked" : "Unverified", tone: allReady ? "good" : "warn" },
      { label: "Save", state: publishError ? "Blocked" : !datasets.length ? "Unknown" : allReady ? "Available" : partial ? "Limited" : "Not ready", tone: publishError ? "bad" : allReady ? "good" : "warn" },
    ] };
}

export function checkedAge(sources: Source[], now: number): string {
  const required = ["collectors-desktop.json", "publications.json"];
  const times = required.map(file => {
    const source = sources.find(s => s.file === file);
    return source?.data && sourceFreshness(source, now) !== "invalid" ? timestamp(source.data.observed_at) : NaN;
  });
  if (times.some(t => !Number.isFinite(t))) return "Update time unknown";
  const seconds = Math.max(0, Math.floor((now - Math.min(...times)) / 1000));
  return `Updated ${seconds < 60 ? `${seconds}s` : seconds < 3600 ? `${Math.floor(seconds / 60)}m` : `${Math.floor(seconds / 3600)}h`} ago`;
}
