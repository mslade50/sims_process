import assert from "node:assert/strict";
import test from "node:test";
import { FRESHNESS_REFRESH_MS, INITIAL_PANEL, checkerLate, nextPanelState, startFreshnessPolling } from "../app/freshness-poll.ts";
import { badgeView, savedCopyNotice } from "../app/ui-rules.ts";
import { FRESHNESS_SCHEMA } from "../worker/freshness-rules.ts";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { loadApp } from "./helpers/render-harness.mjs";

// No window stub: DashboardApp reads localStorage whenever window exists, as the existing shell render tests do.
const renderShell = (options) => renderToStaticMarkup(React.createElement(loadApp("DashboardApp.tsx", options).DashboardApp, { initialView: "this-week" }));

const docAt = (checked_at, ok = true) => ({ schema: FRESHNESS_SCHEMA, checked_at, ok, not_judged: false, stale_count: ok ? 0 : 1, sources: [], notes: [], inbox: { bound: false, written: [], failed: [] } });
const tick = () => new Promise((resolve) => setImmediate(resolve));

test("panel polling: reads at once, then every 5 minutes on a fake clock, and stops on unmount", async (t) => {
  t.mock.timers.enable({ apis: ["setInterval"] });
  const urls = [];
  let reply = { ok: true, status: 200, json: async () => docAt("2026-10-28T12:00:00Z") };
  const fetcher = async (url) => { urls.push(url); return reply; };
  let state = INITIAL_PANEL;
  const stop = startFreshnessPolling("/api/owner-data/data_freshness.json", fetcher, (update) => { state = update(state); });
  await tick();
  assert.equal(urls.length, 1);
  assert.equal(state.doc.checked_at, "2026-10-28T12:00:00Z");
  t.mock.timers.tick(FRESHNESS_REFRESH_MS - 1);
  await tick();
  assert.equal(urls.length, 1, "not before 5 minutes");
  reply = { ok: true, status: 200, json: async () => docAt("2026-10-28T12:15:00Z", false) };
  t.mock.timers.tick(1);
  await tick();
  assert.equal(urls.length, 2);
  assert.equal(state.doc.checked_at, "2026-10-28T12:15:00Z", "a later finding appears without a reload");
  assert.equal(state.doc.ok, false);
  reply = { ok: false, status: 503, json: async () => null };
  t.mock.timers.tick(FRESHNESS_REFRESH_MS);
  await tick();
  assert.equal(state.phase, "ready");
  assert.equal(state.refreshFailed, true, "a failed refresh keeps the last document, flagged");
  assert.equal(state.doc.checked_at, "2026-10-28T12:15:00Z");
  stop();
  t.mock.timers.tick(FRESHNESS_REFRESH_MS * 3);
  await tick();
  assert.equal(urls.length, 3, "no reads after unmount");
});

test("panel state: 404 is 'none', a first-load failure is 'error', a good read clears the failed-refresh flag", () => {
  assert.equal(nextPanelState(INITIAL_PANEL, { phase: "none", doc: null }).phase, "none");
  assert.equal(nextPanelState(INITIAL_PANEL, { phase: "error", doc: null }).phase, "error");
  const flagged = { phase: "ready", doc: docAt("2026-10-28T12:00:00Z"), refreshFailed: true };
  assert.equal(nextPanelState(flagged, { phase: "ready", doc: docAt("2026-10-28T12:15:00Z") }).refreshFailed, false);
});

test("the 45-minute warning follows a ticking clock", () => {
  const doc = docAt("2026-10-28T12:00:00Z");
  const checked = Date.parse(doc.checked_at);
  assert.equal(checkerLate(doc, checked + 44 * 60_000), false);
  assert.equal(checkerLate(doc, checked + 46 * 60_000), true, "an open page crosses 45 minutes without a reload");
  assert.equal(checkerLate(docAt("garbage"), checked), true);
  assert.equal(checkerLate(null, checked), false);
});

test("header badge: a saved copy is never live or fresh, whatever its age", () => {
  assert.deepEqual(badgeView(10 * 60_000, "Run", "run", false), { tone: "live", text: "Run 10 min ago" });
  assert.deepEqual(badgeView(10 * 60_000, "Run", "run", true), { tone: "stale", text: "Saved copy · Run 10 min ago" });
  assert.deepEqual(badgeView(null, "Run", "run", true), { tone: "stale", text: "Saved copy" });
  assert.deepEqual(badgeView(null, "Run", "run", false), { tone: "unknown", text: "No run yet" });
  assert.equal(savedCopyNotice([], 0), null);
  assert.match(savedCopyNotice([0], 3 * 3_600_000), /saved copy .*3 h ago.*Reload to retry/);
  assert.match(savedCopyNotice([0, 1], 3_600_000), /2 datasets are a saved copy/);
});

test("rendered header after an R2 outage: saved-copy badge, stale tone and a page warning (live badge otherwise)", () => {
  const asOf = new Date(Date.now() - 10 * 60_000).toISOString();
  const data = { "manifest.json": { generated_at: asOf }, "golfprice/index.json": { events: [{ name: "Test Open", course: "Test GC", date_start: "2026-10-22", runs: [{ as_of: asOf }] }] } };
  const live = renderShell({ data });
  assert.match(live, /freshness-badge fresh-live/);
  assert.doesNotMatch(live, /saved-copy-notice|Saved copy/);
  const outage = renderShell({ data, stale: ["golfprice/index.json"], saved: [Date.now() - 3_600_000] });
  assert.match(outage, /freshness-badge fresh-stale/);
  assert.match(outage, /Saved copy · Run 10 min ago/);
  assert.doesNotMatch(outage, /freshness-badge fresh-live/);
  assert.match(outage, /data-testid="saved-copy-notice"/);
  assert.match(outage, /could not be reached/);
});
