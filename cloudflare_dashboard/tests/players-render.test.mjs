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
  assert.match(html, /R1-R2 · last 2 years · no situational filter/, "A4: next layout opens on R1-R2");
  assert.doesNotMatch(html, /includes weekend rounds \(selected by the cut\)/, "A4: weekend badge only when R3/R4 are on");
  assert.doesNotMatch(html, /Player baseline · dashed slate/, "R1-R2 default is not a situational filter");
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

// ---- mu_tour card, full-page (owner decision October 9) -------------------------------------------------------------
const explainWeek = JSON.parse(readFileSync(new URL("./fixtures/explain_week.json", import.meta.url), "utf8"));
const muInputs = JSON.parse(readFileSync(new URL("./fixtures/model_inputs_mu_tour.json", import.meta.url), "utf8"));
function savedData(inputsPatch = {}) {
  const [p0, p1] = explainWeek.players;
  const explain = { ...explainWeek, players: [{ ...p0, id: 17639, withdrawn: false, mu: 0.3 }, { ...p1, id: 5, withdrawn: false, mu: -0.3 }] };
  const run = { kind: "week", run: explainWeek.run, as_of: explainWeek.as_of, key: "golfprice/runs/fx/week_x/model_inputs.json" };
  const inputs = { ...muInputs, event: { ...muInputs.event, event_uid: explainWeek.event_uid }, generated_for_as_of: explainWeek.as_of, provenance: { run_dir: `golfprice/week/runs/fx/${explainWeek.run}` }, players: [{ dg_id: 17639, challenger: { mu_tour: 0.4002 } }], ...inputsPatch };
  return { ...data,
    "golfprice/index.json": { events: [{ event_uid: explainWeek.event_uid, name: "Fixture Open", tour: "pga", date_start: "2026-10-01", explain_run: `week_${explainWeek.run}`, explain_key: "golfprice/explain/fx.json", runs: [run] }] },
    "golfprice/player_profiles/fields.json": { schema_version: "player_profiles.fields.v1", fields: [{ event_uid: explainWeek.event_uid, player_ids: [17639, 5] }] },
    "golfprice/explain/fx.json": explain, [run.key]: inputs };
}
test("/players default layout: secondary card shows mu_tour with the PGA-scale label; vs field stays secondary", () => {
  const html = next("?player=17639", savedData());
  assert.match(html, /This week \(PGA scale\)/);
  assert.match(html, /data-testid="pga-skill"[^>]*>\+0\.400/);
  assert.match(html, /field offset \+0\.100/);
  assert.match(html, /vs field: \+0\.300/);
});
test("/players?layout=next: This-week chip shows mu_tour; the Method names the shared PGA-scale zero", () => {
  const html = next("?player=17639&layout=next", savedData());
  assert.match(html, /data-testid="pga-chip"/);
  assert.match(html, /<b>\+0\.40<\/b><small>This week \(PGA scale\)/);
  assert.match(html, /now share the PGA-scale zero; the remaining gap is form versus model/);
});
test("/players: no mu_tour in the matching document (or no matching document) keeps the old vs-field display", () => {
  for (const d of [savedData({ players: [{ dg_id: 17639, challenger: {} }] }), savedData({ generated_for_as_of: "2020-01-01T00:00:00Z" })]) {
    const html = next("?player=17639", d);
    assert.match(html, /Relative to this field/);
    assert.doesNotMatch(html, /data-testid="pga-skill"/);
    assert.doesNotMatch(next("?player=17639&layout=next", d), /data-testid="pga-chip"/);
  }
});
