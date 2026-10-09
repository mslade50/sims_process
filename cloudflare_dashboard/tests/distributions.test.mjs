import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { checkpoints, curveRows, matchup, marker } from "../app/distributions-rules.ts";
import { parseExplain } from "../app/explain-rules.ts";

const week = parseExplain(JSON.parse(await readFile(new URL("./fixtures/explain_week.json", import.meta.url), "utf8")));
test("only validated pricing checkpoints enter the timeline", () => {
  const rs = checkpoints({ distribution_runs: [
    { run: "old", kind: "week", as_of: "2026-10-07", validated: true },
    { run: "failed", kind: "live", as_of: "2026-10-08", after_round: 1, validated: false },
    { run: "r0", kind: "live", as_of: "2026-10-08", after_round: 0, validated: true },
    { run: "live", kind: "live", as_of: "2026-10-08", after_round: 1, validated: true },
  ] });
  assert.deepEqual(rs.map((r) => r.run), ["old", "live"]);
});
test("head-to-head reversal preserves conditional win probability and shared tie probability", () => {
  const d = { head_to_heads: { method: "joint_simulation", rows: [[1, 2, 0.63, 0.04]] } };
  assert.deepEqual(matchup(d, 1, 2), { p: 0.63, tie: 0.04 });
  assert.deepEqual(matchup(d, 2, 1), { p: 0.37, tie: 0.04 });
  assert.equal(matchup(d, 1, 1), null);
  assert.equal(matchup(d, 1, 3), null);
  assert.equal(matchup({ players: week.players }, 1, 2), null); // never infer H2H from marginal curves
});
test("standard finish markers use exact model prices even when rank curve or market differs", () => {
  const p = { probs: { win: { model: 0.06329, fair: 0.04411 }, top_5: { model: 0.3 } }, finish: { pos: [0.1, 0.2, 0.3] } };
  assert.equal(marker(p, 1), 0.06329);
  assert.equal(marker(p, 5), 0.3);
  assert.equal(marker(p, 2), 0.30000000000000004);
});
test("overlay curves retain independent current and prior mass and monotone CDFs", () => {
  const p = { id: 1, finish: { pos: [0.1, 0.3, 0.6] } };
  const q = { id: 1, finish: { pos: [0.3, 0.4, 0.3] } };
  const rows = curveRows([p], [q], true);
  assert.equal(rows.at(-1).now_1, 100);
  assert.equal(rows.at(-1).then_1, 100);
  assert.equal(rows[0].now_1, 10);
  assert.equal(rows[0].then_1, 30);
  assert.ok(rows.every((r, i) => !i || r.now_1 >= rows[i-1].now_1));
  assert.deepEqual(curveRows([], [], false), []);
});
import { SETTLEMENT_RANK_LABEL, markerLabel, cutReference } from "../app/distributions-rules.ts";
test("B7: the cut marker uses the priced make-cut probability, not the sliced rank array; top-N markers are unchanged", () => {
  const p = { probs: { win: { model: 0.05 }, top_5: { model: 0.2 }, top_10: { model: 0.3 }, top_20: { model: 0.5 }, make_cut: { model: 0.77 } }, finish: { pos: [0.1, 0.2, 0.3, 0.1] } };
  assert.equal(marker(p, 65, 65), 0.77);
  assert.equal(marker(p, 4), 0.1 + 0.2 + 0.3 + 0.1, "without a cut line the rank slice is used");
  assert.equal(marker(p, 4, 65), 0.1 + 0.2 + 0.3 + 0.1, "a non-cut k still slices the rank curve");
  assert.equal(marker(p, 10, 10), 0.3, "a cut line at a priced top-N keeps the model price");
  assert.equal(marker(p, 5, 65), 0.2);
  const noMarket = { probs: { win: { model: 0.05 }, top_5: { model: 0.2 }, top_10: { model: 0.3 }, top_20: { model: 0.5 }, make_cut: { model: null } }, finish: { pos: [0.5, 0.5] } };
  assert.equal(marker(noMarket, 2, 2), 1, "no make-cut price falls back to the rank slice");
  assert.equal(markerLabel(65, 65), "Make cut (top 65 and ties)");
  assert.equal(markerLabel(10, 65), "Top 10");
  assert.equal(markerLabel(65, null), "Top 65");
  assert.match(SETTLEMENT_RANK_LABEL, /finish position.*missed cuts ranked below the field/);
});

test("W1: the cut reference is independent of the finish marker and uses p_make_cut", () => {
  const p = { probs: { make_cut: { model: 0.876 } } };
  assert.deepEqual(cutReference(p, 65), { x: 65, label: "Make cut (top 65 and ties)", prob: 0.876 });
  assert.equal(cutReference(p, null), null);
  assert.equal(cutReference({ probs: {} }, 65), null);
  const src = readFileSync(new URL("../app/DistributionView.tsx", import.meta.url), "utf8");
  assert.match(src, /cutReference\(p, doc\.event\.cut_top_n\)/);
  assert.doesNotMatch(src, /ReferenceLine/, "finish chart is grouped bars with no reference lines; the cut shows as a legend line and a Missed cut bar");
  assert.match(src, /Missed cut/);
});
