import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { renderView } from "./helpers/render-harness.mjs";

const fixture = (name) => JSON.parse(readFileSync(new URL(`./fixtures/player_profiles/${name}`, import.meta.url), "utf8"));
const catalog = fixture("catalog.json");
const batch = fixture("batch_0000.json");
const data = {
  "golfprice/player_profiles/dossier-review.json": catalog,
  "golfprice/player_profiles/fixture/batch_0000.json": batch,
};

test("/players fixture: catalog lists Hideki and Blair and shows the catalog as-of date", () => {
  const html = renderView("PlayerProfilesView.tsx", "PlayersView", { data, pathname: "/players" });
  assert.match(html, /Matsuyama, Hideki/);
  assert.match(html, /Blair, Zac/);
  assert.match(html, /Catalog as of Oct 8, 2026/);
  assert.match(html, /freshness-badge/);
});

test("/players?player=17639 renders Blair's PGA benchmark card value from the fixture shard", () => {
  const html = renderView("PlayerProfilesView.tsx", "PlayersView", { data, search: "?player=17639", pathname: "/players" });
  const expected = catalog.players.find((p) => p.dg_id === 17639).pga_benchmark.value;
  const signed = `${expected > 0 ? "+" : ""}${expected.toFixed(3)}`;
  assert.match(html, /Historical form vs 2025 PGA starts/);
  assert.ok(html.includes(signed), `benchmark value ${signed} rendered`);
  assert.match(html, /strokes\/round/);
  assert.match(html, /Know what the headline measures/);
});

test("/players with the catalog missing shows the not-published empty state, not a throw", () => {
  const html = renderView("PlayerProfilesView.tsx", "PlayersView", { data: {}, pathname: "/players" });
  assert.match(html, /Player profiles have not been published/);
});

test("/players: an old catalog as_of carries the amber stale badge, a recent one does not", () => {
  const old = { ...catalog, as_of: "2026-09-01T00:00:00Z" };
  const stale = renderView("PlayerProfilesView.tsx", "PlayersView", { data: { ...data, "golfprice/player_profiles/dossier-review.json": old }, pathname: "/players" });
  assert.match(stale, /Catalog as of Sep 1, 2026/);
  assert.match(stale, /fresh-aging/);
  assert.match(stale, /stale/);
  const fresh = renderView("PlayerProfilesView.tsx", "PlayersView", { data: { ...data, "golfprice/player_profiles/dossier-review.json": { ...catalog, as_of: new Date().toISOString() } }, pathname: "/players" });
  assert.match(fresh, /fresh-fresh|fresh-live/);
  assert.doesNotMatch(fresh, /fresh-aging/);
});

// ---- ?layout=next (wave 1, behind a flag) -------------------------------------------------------------------------
const next = (search = "?player=17639&layout=next", d = data) => renderView("PlayerProfilesView.tsx", "PlayersView", { data: d, search, pathname: "/players" });

test("/players?layout=next renders hero, SG bars (n/a state), filter summary bar, closed disclosures and one Method section", () => {
  const html = next();
  assert.match(html, /data-layout="next"/);
  assert.match(html, /data-testid="player-hero"/);
  assert.match(html, /vs PGA avg/);
  assert.match(html, /\+0\.73 <small/);
  assert.match(html, /Reference only; prices use the event model/);
  assert.match(html, /Descriptive, not a forecast/);
  assert.match(html, /data-testid="sg-bars"|data-testid="sg-na"/);
  assert.match(html, /data-testid="filter-bar"/);
  assert.match(html, /R1-R4 · last 2 years · no situational filter/);
  assert.match(html, /includes weekend rounds \(selected by the cut\)/);
  assert.match(html, /The player&#x27;s shape/);
  for (const summary of ["Shot detail", "Playing style", String.raw`within-field style, PGA\+LIV data only`, "Method and caveats"]) assert.match(html, new RegExp(String.raw`<summary>[\s\S]{0,40}` + summary));
  assert.equal((html.match(/data-testid="player-method"/g) ?? []).length, 1);
  // The old stacked layout pieces are gone from the new layout.
  assert.equal(html.includes("Know what the headline measures"), false);
  assert.equal(html.includes("Saved event context"), false);
  assert.equal(html.includes("Historical form vs 2025 PGA starts"), false);
});

test("/players?layout=next: Method holds anchor, window, shrinkage status, reference ids, n_eff naming and the gap sentence", () => {
  const html = next();
  assert.match(html, /180-day half-life/);
  assert.match(html, /1095-day window/);
  assert.match(html, /Shrinkage status: none/);
  assert.match(html, /pga_2025_r12:74b9dbd5ef9b8cedc861/);
  assert.match(html, /decay-weight sum/);
  assert.match(html, /Kish effective sample/);
  assert.match(html, /unshrunk, 'This week'|reads 0\.15-0\.2 higher|read 0\.15-0\.2 higher/);
  assert.equal(html.includes("DataGolf-style shrunk estimate"), false); // absent unless the catalog publishes shrunk_value
});

test("/players?layout=next shows the DataGolf-style shrunk estimate in Method only, never as the headline, when published", () => {
  const withShrunk = structuredClone(batch);
  withShrunk.profiles["17639"].pga_benchmark.shrunk_value = 0.609;
  const html = next("?player=17639&layout=next", { ...data, "golfprice/player_profiles/fixture/batch_0000.json": withShrunk });
  assert.match(html, /DataGolf-style shrunk estimate: \+0\.609/);
  assert.match(html, /\+0\.73 <small/);
});

test("/players?layout=next stores the typical-regular tick only when the catalog has one", () => {
  assert.equal(next().includes("regular-tick"), false);
  const html = next("?player=17639&layout=next", { ...data, "golfprice/player_profiles/dossier-review.json": { ...catalog, regular_tick: { value: 0.39, as_of: "2026-10-08" } } });
  assert.match(html, /data-testid="regular-tick"/);
});

test("default layout is unchanged: no hero, the old benchmark card and shape panel are still there", () => {
  const html = next("?player=17639");
  assert.equal(html.includes("data-layout=\"next\""), false);
  assert.equal(html.includes("player-hero"), false);
  assert.match(html, /Know what the headline measures/);
  assert.match(html, /Saved event context/);
  assert.match(html, /Global historical model traits · separate sample/);
});

test("B5: catalog 404 shows a hint and a retry action; an absent player ID gets a next action", () => {
  const down = renderView("PlayerProfilesView.tsx", "PlayersView", { data: {}, pathname: "/players" });
  assert.match(down, /Player profiles have not been published/);
  assert.match(down, /data-testid="catalog-hint"/);
  assert.match(down, />Retry</);
  const absent = next("?player=999999");
  assert.match(absent, /Player not in this catalog/);
  assert.match(absent, />Clear this selection</);
});
