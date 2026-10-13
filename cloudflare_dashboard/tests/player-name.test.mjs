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
