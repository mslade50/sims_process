import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { attentionItems, collectorStatus, decodeSnapshot, freshness, initialSources, publicationStatus, sourceFreshness } from "../app/shot-monitor-rules.ts";
import { tourSummary, checkedAge } from "../app/shot-monitor-summary.ts";
import { createShotMonitor } from "../app/shot-monitor-client.ts";
import { navigationFor, ownerOnlyPath, resolveRoute } from "../app/shell-rules.ts";
import { renderView } from "./helpers/render-harness.mjs";

const now = Date.parse("2026-10-27T12:00:00Z");
const iso = (delta = 0) => new Date(now + delta).toISOString();
const hash = "a".repeat(64);
const collector = (extra = {}) => ({ id: "pga-shots", tour: "pga", label: "PGA captures", state: "running", reason: "Scheduled collection", observed_at: iso(), ...extra });
const publication = (extra = {}) => ({ id: "pga-clean", tour: "pga", label: "Clean shots", published_at: "2020-01-01T00:00:00Z", generation: "g1", manifest_sha256: hash, validation: { state: "passed", manifest_sha256: hash, summary: "Counts and files checked", checked_at: iso() }, ...extra });
const envelope = (field, rows) => ({ schema_version: 1, observed_at: iso(), [field]: rows });
const sources = () => [
  { file: "collectors-desktop.json", status: "ready", data: envelope("collectors", [collector()]) },
  { file: "publications.json", status: "ready", data: envelope("publications", [publication()]) },
];
const response = (url) => Response.json(url.includes("collectors-") ? envelope("collectors", [collector()]) : envelope("publications", [publication()]));
const render = (data = initialSources(), extra = {}) => renderView("ShotDataView.tsx", "ShotDataContent", { props: { sources: data, now, refreshing: false, ...extra } });

test("source age thresholds are strict, with bounded future tolerance and timezone-required dates", () => {
  assert.equal(freshness(iso(-15 * 60_000), now, 15 * 60_000), "fresh");
  assert.equal(freshness(iso(-15 * 60_000 - 1), now, 15 * 60_000), "stale");
  assert.equal(freshness(iso(5 * 60_000), now, 15 * 60_000), "fresh");
  assert.equal(freshness(iso(5 * 60_000 + 1), now, 15 * 60_000), "invalid");
  for (const time of [null, "garbage", "2026-10-27", "2026-10-27T12:00:00"]) assert.equal(freshness(time, now, 900_000), "invalid");
  const [c, p] = sources();
  c.data.observed_at = p.data.observed_at = iso(-20 * 60_000);
  assert.equal(sourceFreshness(c, now), "stale");
  assert.equal(sourceFreshness(p, now), "fresh");
  p.data.observed_at = iso(-30 * 60_000 - 1);
  assert.equal(sourceFreshness(p, now), "stale");
});

test("stale snapshots and row observations override running; paused and errors remain distinct", () => {
  const [source] = sources();
  assert.equal(collectorStatus(collector({ state: "paused" }), source, now), "paused");
  assert.equal(collectorStatus(collector({ state: "error" }), source, now), "error");
  assert.match(collectorStatus(collector({ observed_at: iso(-16 * 60_000) }), source, now), /STALE/);
  assert.match(collectorStatus(collector({ observed_at: iso(6 * 60_000) }), source, now), /Invalid/);
  source.status = "error";
  assert.match(collectorStatus(collector(), source, now), /STALE/);
});

test("ready binds validation to exact manifest, never to old generation validation or machine publication age", () => {
  const [, source] = sources();
  assert.equal(publicationStatus(publication(), source, now), "Ready", "old published dataset remains ready when the monitor is fresh");
  for (const manifest_sha256 of [undefined, "", "b".repeat(64), "not-a-hash"]) {
    assert.notEqual(publicationStatus(publication({ manifest_sha256 }), source, now), "Ready");
  }
  const row = publication(); row.validation.manifest_sha256 = "b".repeat(64);
  assert.match(publicationStatus(row, source, now), /mismatch/);
  for (const state of ["held", "partial", "unverified"]) assert.equal(publicationStatus(publication({ validation: { state, summary: "Not passed" } }), source, now), state);
  for (const published_at of [null, "invalid", iso(6 * 60_000)]) assert.notEqual(publicationStatus(publication({ published_at }), source, now), "Ready");
  row.validation = { ...publication().validation, checked_at: iso(6 * 60_000) };
  assert.notEqual(publicationStatus(row, source, now), "Ready");
  source.data.observed_at = iso(-31 * 60_000);
  assert.match(publicationStatus(publication(), source, now), /STALE/);
});

test("decoder rejects malformed, unsupported and duplicate rows instead of silently implying health", () => {
  for (const bad of [null, {}, { schema_version: 2, observed_at: iso(), collectors: [] }, { schema_version: 1, observed_at: iso() }, envelope("collectors", [collector({ tour: "other" })]), envelope("collectors", [collector({ metrics: [{ label: "count", value: {} }] })]), envelope("collectors", [collector(), collector()])]) assert.throws(() => decodeSnapshot(bad, "collectors-desktop.json"));
  assert.throws(() => decodeSnapshot(envelope("publications", [publication({ validation: null })]), "publications.json"));
  assert.deepEqual(decodeSnapshot(envelope("collectors", []), "collectors-desktop.json").collectors, []);
  const raw = { ...envelope("publications", [publication()]), collectors: [{ malformed: true }] };
  assert.equal(decodeSnapshot(raw, "publications.json").collectors, undefined);
});

test("render loading, missing, errors, stale previous copies and escaping without any all-clear", () => {
  assert.match(render(), /Checking shot data…/);
  for (const status of ["missing", "error"]) {
    const data = initialSources().map((s) => ({ ...s, status, error: status === "missing" ? "Missing snapshot (HTTP 404)" : "Network offline" }));
    const html = render(data);
    assert.match(html, status === "missing" ? /Missing snapshot/ : /Network offline/);
    assert.doesNotMatch(html, /Reported checks are clear|sm-good/);
    assert.match(html, /readiness unknown/);
  }
  const data = sources(); data[0].status = "error"; data[0].error = "Network offline";
  data[0].data.collectors[0].label = "<script>unsafe</script>";
  const html = render(data);
  assert.match(html, /previous copy — STALE/);
  assert.match(html, /&lt;script&gt;/);
  assert.doesNotMatch(html, /<script>unsafe/);
});

test("render separates captured, cleaned and published and exposes hash mismatch instead of a passed badge", () => {
  const data = sources();
  data[0].data.collectors[0].metrics = [{ label: "Captured shots", value: 12345 }, { label: "Cleaned shots", value: 11000 }];
  data[1].data.publications[0].validation.manifest_sha256 = "b".repeat(64);
  const html = render(data);
  for (const text of ["Captured shots", "12,345", "Cleaned shots", "11,000", "Published datasets", "manifest mismatch", "No checked dataset is confirmed available."]) assert.ok(html.includes(text), text);
  assert.doesNotMatch(html, /sm-good/);
  assert.match(html, /about every 5 minutes/);
  assert.match(html, /polls every 30 seconds/);
});

test("missing tour coverage and empty arrays are explicit; complete fresh coverage can clear", () => {
  const data = sources();
  assert.ok(attentionItems(data, now).some((a) => /Korn Ferry.*no collectors/.test(a.message)));
  for (const tour of ["dpwt", "kft"]) {
    data[0].data.collectors.push(collector({ id: tour, tour, state: "idle" }));
    data[1].data.publications.push(publication({ id: tour, tour }));
  }
  assert.equal(attentionItems(data, now).length, 0);
  assert.match(render(data), /Tour checks are clear/);
  data[0].data.collectors[0].state = "paused";
  assert.match(render(data), /Paused/);
  data[0].data.collectors[0].issues = ["Capture log missing"];
  assert.match(render(data), /Capture log missing/);
  assert.match(render(data), /tour needs attention/);
});

test("only desktop and publications fetched no-store; concurrent refresh calls share the same batch", async () => {
  const calls = []; let release;
  const latch = new Promise((resolve) => { release = resolve; });
  const monitor = createShotMonitor(() => {}, { fetcher: async (url, init) => { calls.push([url, init]); await latch; return response(url); } });
  try {
    const first = monitor.refresh();
    assert.equal(monitor.refresh(), first);
    await Promise.resolve();
    assert.equal(calls.length, 2);
    assert.deepEqual(calls.map(([url]) => url.split("/").at(-1)), ["collectors-desktop.json", "publications.json"]);
    for (const [, init] of calls) { assert.equal(init.cache, "no-store"); assert.ok(init.signal instanceof AbortSignal); }
    release(); await first;
  } finally { monitor.stop(); }
});

test("independent fetch result is visible while the other source hangs; timeout is bounded", async () => {
  const states = [];
  const monitor = createShotMonitor((s) => states.push(s), { timeoutMs: 20, now: () => now, fetcher: async (url) => url.includes("collectors-") ? response(url) : new Promise(() => {}) });
  try {
    await monitor.refresh();
    assert.ok(states.some((s) => s.sources[0].status === "ready" && s.sources[1].status === "loading"));
    const last = states.at(-1);
    assert.equal(last.refreshing, false);
    assert.equal(last.sources[1].status, "error");
    assert.match(last.sources[1].error, /timed out/);
  } finally { monitor.stop(); }
});

test("failed refresh retains previous data as stale; recovery clears failure; forbidden clears private data", async () => {
  let mode = "ok"; let last;
  const monitor = createShotMonitor((s) => { last = s; }, { now: () => now, fetcher: async (url) => {
    if (mode === "network") throw new TypeError("Network offline");
    if (mode === "missing") return new Response("", { status: 404 });
    if (mode === "forbidden") return new Response("", { status: 403 });
    if (mode === "invalid") return Response.json({ schema_version: 42 });
    return response(url);
  } });
  try {
    await monitor.refresh();
    for (const failure of ["network", "missing", "invalid"]) {
      mode = failure; await monitor.refresh();
      assert.ok(last.sources.every((s) => s.data));
      assert.ok(last.sources.every((s) => sourceFreshness(s, now) === "stale"));
    }
    mode = "ok"; await monitor.refresh();
    assert.ok(last.sources.every((s) => s.status === "ready" && !s.error));
    mode = "forbidden"; await monitor.refresh();
    assert.ok(last.sources.every((s) => !s.data && s.status === "error"));
  } finally { monitor.stop(); }
});

test("stop aborts requests and prevents updates; a disposed monitor never starts fetching", async () => {
  const signals = []; let count = 0;
  const monitor = createShotMonitor(() => { count++; }, { fetcher: async (_, init) => { signals.push(init.signal); return new Promise(() => {}); } });
  const pending = monitor.refresh(); await Promise.resolve();
  monitor.stop(); const before = count; await pending;
  assert.ok(signals.every((s) => s.aborted)); assert.equal(count, before);
  await monitor.refresh(); assert.equal(signals.length, 2);
});

test("scheduled polling refreshes and stops without leaving background work", async () => {
  let calls = 0;
  const monitor = createShotMonitor(() => {}, { intervalMs: 10, fetcher: async (url) => { calls++; return response(url); } });
  try { await monitor.refresh(); await new Promise((resolve) => setTimeout(resolve, 35)); assert.ok(calls >= 4); }
  finally { monitor.stop(); }
  const before = calls; await new Promise((resolve) => setTimeout(resolve, 25)); assert.equal(calls, before);
});

test("Shot data is owner-only in navigation and shared Worker gate; registered and lazily loaded", () => {
  for (const path of ["/shot-data", "/SHOT-DATA/", "/shot-data.rsc"]) assert.equal(ownerOnlyPath(path), "page");
  assert.equal(ownerOnlyPath("/api/owner-data/shot-monitor/publications.json"), "api");
  assert.ok(navigationFor(true).find((g) => g.label === "Operate").items.some((i) => i.key === "shot-data"));
  assert.ok(!navigationFor(false).flatMap((g) => g.items).some((i) => i.key === "shot-data"));
  assert.equal(resolveRoute("shot-data", ["shot-data"]).kind, "view");
  const app = readFileSync(new URL("../app/DashboardApp.tsx", import.meta.url), "utf8");
  assert.match(app, /const ShotDataView = lazy\(/);
  assert.match(app, /"shot-data": ShotDataView/);
});

test("summary separates scheduled PGA capture from a blocked publisher and scopes repair advice", () => {
  const data = sources();
  data[0].data.collectors = [collector({ id: "GolfRichShotCollector", state: "idle" }), collector({ id: "GolfShotRoundPublish", state: "error", reason: "Delivery degraded - source hash mismatch, repair pending" })];
  data[1].data.publications[0].validation.state = "partial";
  const result = tourSummary("pga", data, now);
  assert.equal(result.headline, "Capture works. Publishing is blocked.");
  assert.equal(result.next, "Restore the approved publishing code.");
  assert.equal(result.stages[0].state, "Scheduled");
  assert.equal(result.stages[2].state, "Blocked");
  assert.equal(result.available, "Previously published data, with limits.");
  for (const reason of ["Disk unavailable", "API version mismatch", "Manifest hash mismatch"]) {
    data[0].data.collectors[1].reason = reason;
    assert.equal(tourSummary("pga", data, now).next, "Check the publishing job failure.");
  }
});

test("paused KFT with exact-hash checked measurements is neither broken nor actively collecting", () => {
  const data = sources();
  data[0].data.collectors = [collector({ id: "SimPrep-KFT-forward", tour: "kft", state: "paused" })];
  data[1].data.publications = [publication({ id: "kft-quality", tour: "kft" })];
  const result = tourSummary("kft", data, now);
  assert.equal(result.attention, false);
  assert.equal(result.badge, "Paused");
  assert.equal(result.stages[0].state, "Paused");
  assert.equal(result.next, undefined);
  assert.equal(result.available, "Checked historical shot measurements.");
  data[1].data.publications[0].validation.manifest_sha256 = "b".repeat(64);
  const mismatch = tourSummary("kft", data, now);
  assert.equal(mismatch.attention, true);
  assert.doesNotMatch(mismatch.available, /Checked historical|are available/);
  assert.notEqual(mismatch.stages[2].state, "Available");
});

test("DPWT held holes and missing rich publication never imply approved shot data", () => {
  const data = sources();
  data[0].data.collectors = [collector({ id: "SimPrep-DPWT-forward", tour: "dpwt", state: "error", reason: "Collector is running with degraded source coverage" })];
  data[1].data.publications = [publication({ id: "dpwt-official-holes", tour: "dpwt", validation: { state: "held", summary: "Corrections required" } }), publication({ id: "dpwt-rich-inputs", tour: "dpwt", published_at: undefined, validation: { state: "unverified", summary: "No rich publication found" } })];
  const result = tourSummary("dpwt", data, now);
  assert.equal(result.headline, "Collection needs attention.");
  assert.equal(result.available, "Published hole scores need corrections. New shot data is not ready.");
  assert.equal(result.stages[1].state, "On hold");
  assert.notEqual(result.stages[2].state, "Available");
  for (const reason of ["Access revoked", "Stopped", "Not running", "Stale source; previously running"]) {
    data[0].data.collectors[0].reason = reason;
    assert.doesNotMatch(tourSummary("dpwt", data, now).headline, /Collecting/);
  }
});

test("failed or stale required sources override every summary green, including retained passed publications", () => {
  for (const index of [0, 1]) for (const mode of ["error", "missing", "stale", "future"]) {
    const data = sources();
    if (["error", "missing"].includes(mode)) data[index].status = mode;
    else data[index].data.observed_at = iso(mode === "stale" ? -31 * 60_000 : 6 * 60_000);
    const result = tourSummary("pga", data, now);
    assert.equal(result.attention, true);
    assert.ok(result.stages.every(s => s.tone === "neutral"));
    assert.doesNotMatch(result.available, /are available|Checked datasets/);
    assert.equal(result.headline, "Cannot check right now.");
  }
});

test("missing tours and publications remain unknown without invented counts or readiness", () => {
  for (const data of [[], initialSources(), sources()]) {
    const result = tourSummary("dpwt", data, now);
    assert.equal(result.attention, true);
    assert.doesNotMatch(result.available, /are available|Checked historical|\d/);
    assert.ok(result.stages.every(s => s.state !== "Available"));
  }
  const data = sources(); data[1].data.publications = [];
  const result = tourSummary("pga", data, now);
  assert.equal(result.attention, true);
  assert.notEqual(result.stages[2].state, "Available");
});

test("header freshness ages the oldest monitor observation rather than the browser poll", () => {
  const data = sources();
  data.forEach(s => { s.checkedAt = now; });
  data[0].data.observed_at = iso(-4 * 60_000);
  assert.equal(checkedAge(data, now), "Updated 4m ago");
  data[1].data.observed_at = iso(-31 * 60_000);
  assert.equal(checkedAge(data, now), "Updated 31m ago");
  data[1].data.observed_at = "invalid";
  assert.equal(checkedAge(data, now), "Update time unknown");
  assert.equal(checkedAge(data.slice(0, 1), now), "Update time unknown");
});

test("all technical evidence is inside default-closed tour or monitor disclosures", () => {
  const data = sources();
  data[0].data.collectors[0].reason = "TECHNICAL_LOG_MARKER";
  const html = render(data);
  assert.equal((html.match(/class="sm-details"/g) ?? []).length, 3);
  assert.match(html, /Monitor diagnostics/);
  assert.doesNotMatch(html, /<details[^>]*\sopen(?:=|\s|>)/);
  // Remove closed native disclosures to inspect only the initial visible surface.
  let visible = html;
  while (/<details\b/.test(visible)) visible = visible.replace(/<details\b[^>]*>(?:(?!<details\b)[\s\S])*?<\/details>/g, "");
  assert.doesNotMatch(visible, /TECHNICAL_LOG_MARKER|SHA-256|Captured shots|Generation|schema|snapshot/i);
  assert.equal((visible.match(/class="sm-golf"/g) ?? []).length, 3);
  assert.equal((visible.match(/aria-label="Pipeline stages"/g) ?? []).length, 3);
});


test("error reasons never manufacture a collecting claim from free-text negations", () => {
  const data = sources();
  for (const reason of ["Collector is no longer running", "Collector isn't running", "Collector wasn't running", "Collector failed to keep running"]) {
    data[0].data.collectors = [collector({ id: "SimPrep-DPWT-forward", tour: "dpwt", state: "error", reason })];
    const result = tourSummary("dpwt", data, now);
    assert.doesNotMatch(result.headline, /Collecting/);
    assert.equal(result.stages[0].state, "Needs check");
    assert.equal(result.next, "Check the collection problem before publishing.");
  }
});
