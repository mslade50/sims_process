import test from "node:test";
import assert from "node:assert/strict";
import { playerDisplayName, isAmateurTag, nameInitials } from "../app/player-name.ts";

test("provider forms become First Last", () => {
  assert.equal(playerDisplayName("Scheffler, Scottie"), "Scottie Scheffler");
  assert.equal(playerDisplayName("Aberg(Am), Ludvig"), "Ludvig Aberg");
  assert.equal(playerDisplayName("Rahman (am), Daeng Abdul"), "Daeng Abdul Rahman");
  assert.equal(playerDisplayName("Ng(AM), Terrance"), "Terrance Ng");
  assert.equal(playerDisplayName("Ludvig Aberg (Am)"), "Ludvig Aberg");
});
test("particles, suffixes, single names and empties", () => {
  assert.equal(playerDisplayName("van Rooyen, Erik"), "Erik van Rooyen");
  assert.equal(playerDisplayName("del Rio, Carlos"), "Carlos del Rio");
  assert.equal(playerDisplayName("Smith, John, Jr."), "John Smith Jr.");
  assert.equal(playerDisplayName("Smith, Jr."), "Smith Jr.");
  assert.equal(playerDisplayName("Tiger"), "Tiger");
  assert.equal(playerDisplayName("Erik van Rooyen"), "Erik van Rooyen");
  assert.equal(playerDisplayName("  "), "");
  assert.equal(playerDisplayName(null), "");
  assert.equal(playerDisplayName(undefined), "");
});
test("amateur tag and initials", () => {
  assert.equal(isAmateurTag("Aberg(Am), Ludvig"), true);
  assert.equal(isAmateurTag("Aberg, Ludvig"), false);
  assert.equal(nameInitials("Aberg(Am), Ludvig"), "LA");
  assert.equal(nameInitials("Tiger"), "T");
});

// R4b: one name format site-wide (explain documents parse to "First Last"; the raw name is kept for sorting) and compact tour labels.
import { tourLabel } from "../app/player-name.ts";
import { parseExplain, sortValue } from "../app/explain-rules.ts";

test("tourLabel shows at most two tours and counts the rest", () => {
  assert.equal(tourLabel(["pga"]), "PGA");
  assert.equal(tourLabel(["pga", "dpwt"]), "PGA · DPWT");
  assert.equal(tourLabel(["pga", "dpwt", "kft", "liv"]), "PGA · DPWT +2");
  assert.equal(tourLabel([]), "");
  assert.equal(tourLabel(undefined), "");
});

test("explain players are displayed First Last everywhere; the raw name still sorts by surname", () => {
  const p = (id, name) => ({ id, name, probs: {}, components: {} });
  const doc = parseExplain({ schema: "golfprice.explain.v1", players: [p(1, "Laporta, Francesco"), p(2, "McIlroy, Rory"), p(3, "van Rooyen, Erik")], event: { name: "E" }, finish: {} });
  assert.deepEqual(doc.players.map((x) => x.name), ["Francesco Laporta", "Rory McIlroy", "Erik van Rooyen"]);
  assert.equal(sortValue(doc.players[0], "name", "win"), "Laporta, Francesco");
});
