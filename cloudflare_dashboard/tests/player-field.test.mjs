import assert from "node:assert/strict";
import test from "node:test";
import { biggestGaps, boardRows, fieldSkills, railRange, searchBoard, sortBoard } from "../app/player-field-rules.ts";

const P = (id, name, model, market, mu = 0) => ({ id, name, country: "USA", withdrawn: false, mu, probs: { win: { model, market } }, why: "w" });
const doc = { kind: "week", players: [P(1, "Smith, Al", 0.1, 0.06, 1), P(2, "Jones, Bo", 0.05, 0.08, 0), P(3, "Lee, Cy", 0.02, null, -1), { ...P(4, "Out, Wd", 0.5, 0.5), withdrawn: true }] };

test("board rows are percent, gap = model minus book, withdrawn players dropped, names displayed first-last", () => {
  const rows = boardRows(doc, "win");
  assert.equal(rows.length, 3);
  assert.deepEqual(rows.map((r) => r.name), ["Al Smith", "Bo Jones", "Cy Lee"]);
  assert.equal(rows[0].gap, 4); assert.equal(rows[1].gap, -3); assert.equal(rows[2].gap, null);
  assert.equal(rows[0].model, 10);
});
test("skill is against the field average and absent when any player lacks one", () => {
  assert.equal(fieldSkills(doc).get(1), 1);
  assert.equal(fieldSkills({ kind: "week", players: [...doc.players, P(5, "No, Mu", 0.1, 0.1, null)] }).size, 0);
});
test("sorting, search, gaps and rail range", () => {
  const rows = boardRows(doc, "win");
  assert.deepEqual(sortBoard(rows, "model").map((r) => r.id), [1, 2, 3]);
  assert.deepEqual(sortBoard(rows, "gap").map((r) => r.id).slice(0, 2), [1, 2]);
  assert.deepEqual(searchBoard(rows, "jon").map((r) => r.id), [2]);
  const g = biggestGaps(rows, 3, 0);
  assert.deepEqual(g.above.map((r) => r.id), [1]); assert.deepEqual(g.below.map((r) => r.id), [2]);
  assert.ok(railRange(rows) >= 0.5);
});
