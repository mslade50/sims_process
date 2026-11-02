import test from "node:test";
import assert from "node:assert/strict";
import { historyRoundRows, parseCourseHistory, signed } from "../app/course-history-rules.ts";

const doc = { schema: "golfprice.course_history.v1", status: "available", course: "Port Royal", by_round: [{ round: 1, editions: 2, vs_par: -0.5, wind: 15, weather_strokes: 1, field_skill: -0.8, adjusted_vs_par: -2.3 }],
  editions: [
    { event_uid: "pga:528:2025-11-13", year: 2025, start: "2025-11-13", same_event: true, courses: ["A"], rounds: [{ round: 1, players: 100, vs_par: 1, wind: 20, course: "A" }, { round: 1, players: 300, vs_par: 3, course: "B" }, { round: 2, players: 90, vs_par: null }] },
    { event_uid: "pga:528:2024-11-14", year: 2024, start: "2024-11-14", same_event: true, courses: ["A"], rounds: [{ round: 1, players: 120, vs_par: -1 }] },
    { year: 2023, rounds: [] },
  ] };

test("parses the history document and drops malformed editions", () => {
  const d = parseCourseHistory(doc);
  assert.equal(d.status, "available");
  assert.equal(d.editions.length, 2);
  assert.equal(d.editions[0].rounds[2].vs_par, null);
  assert.equal(parseCourseHistory({ schema: "other" }), null);
  assert.equal(parseCourseHistory({ ...doc, editions: [] }).status, "unavailable");
});

test("rows are rounds, bars are years oldest first, multi-course rounds player-weighted, hover cell keeps every course", () => {
  const { rows, years, rounds } = historyRoundRows(parseCourseHistory(doc), "vs_par");
  assert.deepEqual(rounds, [1, 2]);
  assert.deepEqual(years, [2024, 2025]);
  assert.equal(rows[0].label, "Round 1");
  assert.equal(rows[0].y2025, 2.5);                    // (1*100 + 3*300) / 400
  assert.equal(rows[0].c2025.players, 400);
  assert.equal(rows[0].c2025.courses.length, 2);
  assert.equal(rows[0].c2025.wind, 20);                // only the course with wind counts
  assert.equal(rows[1].y2025, null);                   // round 2 has no score: bar value null, cell kept
  assert.equal(rows[1].y2024, null);
  assert.equal(rows[1].c2024, undefined);              // 2024 had no round 2
});

test("signed numbers never show a negative zero", () => {
  assert.equal(signed(-0.001), "0.00");
  assert.equal(signed(1.234), "+1.23");
  assert.equal(signed(null), "—");
});
