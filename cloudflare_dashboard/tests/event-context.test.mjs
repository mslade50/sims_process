// One event identity: the shell bar and every event-scoped page resolve the selected event through the same function.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { EVENT_STORAGE_KEY, resolveEventUid, shellEventFor, shellEvents } from "../app/shell-rules.ts";

const NOW = Date.parse("2026-11-05T12:00:00Z");
const index = {
  events: [
    { event_uid: "euro:2026142", name: "Abu Dhabi Aldar Championship", tour: "euro", date_start: "2026-11-05", runs: [{ as_of: "2026-11-04T10:00:00Z", kind: "pre" }] },
    { event_uid: "pga:457", name: "World Wide Technology Championship", tour: "pga", date_start: "2026-11-05", explain_key: "golfprice/x.json", runs: [{ as_of: "2026-11-04T10:00:00Z", kind: "pre" }] },
    { event_uid: "pga:old", name: "Older event", tour: "pga", date_start: "2026-10-29", runs: [] },
  ],
};

test("default is the first shell event, and ?e= wins over the remembered event", () => {
  const first = shellEvents(index, NOW)[0].uid;
  assert.equal(resolveEventUid(index, null, null, NOW), first);
  assert.equal(resolveEventUid(index, null, "bogus", NOW), first);
  assert.equal(resolveEventUid(index, "euro:2026142", "pga:457", NOW), "euro:2026142");
  assert.equal(resolveEventUid(index, "nope", "pga:457", NOW), "pga:457", "unknown ?e= falls to the remembered event");
});

test("remembered event applies only inside the newest week; an older ?e= still resolves and the bar can name it", () => {
  assert.equal(resolveEventUid(index, null, "pga:old", NOW), shellEvents(index, NOW)[0].uid);
  assert.equal(resolveEventUid(index, "pga:old", null, NOW), "pga:old");
  assert.equal(shellEventFor(index, "pga:old", NOW).name, "Older event");
  assert.equal(shellEventFor(index, "euro:2026142", NOW).uid, "euro:2026142");
  assert.equal(resolveEventUid(null, null, null, NOW), "");
});

test("event-scoped pages route their default through the shared hook, and none keeps a private default", async () => {
  const pages = ["DistributionView", "ScoringView", "WeatherEffectsView", "InputsView", "WalkthroughView", "WeatherEditsView", "ExplainViews", "WeeklyPlayersView", "PlayerProfilesView"];
  for (const page of pages) {
    const src = await readFile(new URL(`../app/${page}.tsx`, import.meta.url), "utf8");
    assert.match(src, /useEventParam\(\)/, `${page} must use useEventParam`);
    assert.doesNotMatch(src, /new URLSearchParams\(window\.location\.search\)\.get\("e"\)/, `${page} must not read ?e= itself`);
  }
  const hook = await readFile(new URL("../app/event-param.ts", import.meta.url), "utf8");
  assert.match(hook, /resolveEventUid/);
  const shell = await readFile(new URL("../app/DashboardApp.tsx", import.meta.url), "utf8");
  assert.match(shell, /resolveEventUid\(/);
  assert.ok(EVENT_STORAGE_KEY.length > 0);
});
