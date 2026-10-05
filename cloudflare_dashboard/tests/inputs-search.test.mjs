import assert from "node:assert/strict";
import test from "node:test";
import { fold, matchPlayers, matchRank, matchText } from "../app/inputs-search.ts";

const PLAYERS = ["Scheffler, Scottie", "Højgaard, Nicolai", "Højgaard, Rasmus", "Åberg, Ludvig", "O'Neill, Kevin", "Fitzpatrick, Matt", "Fitzpatrick, Alex", "Rodríguez, Joaquín", "Van Rooyen, Erik", "Kim, Si Woo"].map((name, i) => ({ name, dg_id: i }));
const names = (q) => matchPlayers(PLAYERS, q).map((p) => p.name);

test("fold strips accents, case, apostrophes and punctuation", () => {
  assert.equal(fold("Højgaard, Nicolai"), "hojgaard nicolai");
  assert.equal(fold("Åberg"), "aberg");
  assert.equal(fold("O'Neill"), "oneill");
  assert.equal(fold("Rodríguez, Joaquín"), "rodriguez joaquin");
});

test("matches 'first last' and 'last, first', prefix and case-insensitive", () => {
  assert.deepEqual(names("scottie scheffler"), ["Scheffler, Scottie"]);
  assert.deepEqual(names("scheffler, scottie"), ["Scheffler, Scottie"]);
  assert.deepEqual(names("SCHEF"), ["Scheffler, Scottie"]);
  assert.deepEqual(names("sco sch"), ["Scheffler, Scottie"]);
});

test("accent-insensitive in both directions", () => {
  assert.deepEqual(names("hojgaard"), ["Højgaard, Nicolai", "Højgaard, Rasmus"]);
  assert.deepEqual(names("nicolai hoj"), ["Højgaard, Nicolai"]);
  assert.deepEqual(names("åberg"), ["Åberg, Ludvig"]);
  assert.deepEqual(names("aberg"), ["Åberg, Ludvig"]);
  assert.deepEqual(names("rodriguez"), ["Rodríguez, Joaquín"]);
});

test("apostrophes, spaces in surnames and shared surnames", () => {
  assert.deepEqual(names("o'neill"), ["O'Neill, Kevin"]);
  assert.deepEqual(names("o neill"), ["O'Neill, Kevin"]);
  assert.deepEqual(names("van rooyen"), ["Van Rooyen, Erik"]);
  assert.deepEqual(names("fitz"), ["Fitzpatrick, Alex", "Fitzpatrick, Matt"]);
  assert.deepEqual(names("matt fitz"), ["Fitzpatrick, Matt"]);
});

test("empty and non-matching queries return nothing; limit is respected", () => {
  assert.deepEqual(names(""), []);
  assert.deepEqual(names("zzz"), []);
  assert.equal(matchPlayers(PLAYERS, "i", 3).length, 3);
  assert.equal(matchRank("Kim, Si Woo", "si woo kim"), 0);
});

test("matchText: every word must appear, accent and case-insensitive", () => {
  assert.equal(matchText("Kalman local-level skill driving Højgaard", "kalman HOJGAARD"), true);
  assert.equal(matchText("Layoff: log of days since last round", "layoff rust"), false);
  assert.equal(matchText("anything", "   "), true);
});
