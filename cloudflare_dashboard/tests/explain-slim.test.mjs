// R4b slim explain document: computed in memory by the Worker, never written; drops only the members the first screens do not read.
import assert from "node:assert/strict";
import { gzipSync } from "node:zlib";
import test from "node:test";
import { EXPLAIN_PATH, SLIM_DROP, handleExplainSlim, slimExplain } from "../worker/explain-slim.ts";
import { slimExplainKey } from "../app/shell-rules.ts";

const doc = () => ({ schema: "golfprice.explain.v1", run: "r", players: [{ id: 1, name: "Doe, Jane", probs: { win: { model: 0.1 } }, scores: { r1: 1 }, neighbors: [2], form: { n: 3 }, diverge: { dg_check: { flag: true }, edge_split: { kind: "normal" }, peers: { ids: [2] }, compare: {}, sentences: ["x"] } }, { id: 2, name: "Roe, Rich", diverge: { peers: { ids: [1] } } }], course_card: { holes: [] }, scoring: { big: [1, 2, 3] }, head_to_heads: { rows: [[1, 2, 0.5, null]] } });

test("slim drops only scoring and head_to_heads, marks itself, never mutates the input", () => {
  const d = doc();
  const s = slimExplain(d);
  assert.deepEqual(SLIM_DROP, ["scoring", "head_to_heads"]);
  assert.equal("scoring" in s, false);
  assert.equal("head_to_heads" in s, false);
  assert.equal(s.slim, true);
  assert.deepEqual(s.players[0], { id: 1, name: "Doe, Jane", probs: { win: { model: 0.1 } }, diverge: { dg_check: { flag: true }, edge_split: {} } });
  assert.deepEqual(s.players[1], { id: 2, name: "Roe, Rich", diverge: {} });
  assert.ok(d.players[0].scores && d.players[0].diverge.peers, "input untouched");
  assert.deepEqual(s.course_card, d.course_card);
  assert.ok(d.scoring && d.head_to_heads, "input untouched");
  assert.equal(slimExplain(null), null);
  assert.deepEqual(slimExplain([1]), [1]);
});

test("path rule and client key", () => {
  assert.ok(EXPLAIN_PATH.test("explain/pga_457_2026-11-05/week_20261105T103000Z_6ac7dc752a46.json"));
  assert.ok(!EXPLAIN_PATH.test("explain/../index.json"));
  assert.ok(!EXPLAIN_PATH.test("index.json"));
  assert.equal(slimExplainKey("golfprice/explain/a/b.json"), "golfprice/explain/a/b.json?slim=1");
});

function bucket(text, { gzip = false } = {}) {
  const calls = { head: 0, get: 0, put: 0 };
  const bytes = gzip ? gzipSync(Buffer.from(text)) : Buffer.from(text);
  return { calls, head: async () => { calls.head++; return { httpEtag: '"abc123"' }; },
    get: async () => { calls.get++; return { httpEtag: '"abc123"', httpMetadata: { contentEncoding: gzip ? "gzip" : undefined }, body: new Response(bytes).body }; }, put: async () => { calls.put++; } };
}
const req = (path, init) => new Request(`https://dash.example${path}`, init);
const P = "/api/golfprice/explain/pga_1_2026-11-05/week_x_abcdef123456.json";

test("endpoint answers the slim document for ?slim=1 (plain and gzip objects), without writes", async () => {
  for (const gzip of [false, true]) {
    const b = bucket(JSON.stringify(doc()), { gzip });
    const r = await handleExplainSlim(req(`${P}?slim=1`), b);
    assert.equal(r.status, 200);
    assert.equal(r.headers.get("x-explain-variant"), "slim");
    assert.equal(r.headers.get("etag"), '"abc123-slim"');
    const body = await r.json();
    assert.equal(body.slim, true);
    assert.equal(body.scoring, undefined);
    assert.equal(body.players.length, 2);
    assert.equal(b.calls.put, 0);
  }
});

test("conditional request answers 304; everything else falls through (null)", async () => {
  const b = bucket(JSON.stringify(doc()));
  const r = await handleExplainSlim(req(`${P}?slim=1`, { headers: { "if-none-match": '"abc123-slim"' } }), b);
  assert.equal(r.status, 304);
  assert.equal(await handleExplainSlim(req(P), b), null, "no slim flag");
  assert.equal(await handleExplainSlim(req(`${P}?slim=1`, { method: "POST" }), b), null, "writes never");
  assert.equal(await handleExplainSlim(req("/api/golfprice/index.json?slim=1"), b), null, "only explain documents");
  assert.equal(await handleExplainSlim(req("/api/golfprice/explain/../x.json?slim=1"), b), null);
  assert.equal(await handleExplainSlim(req(`${P}?slim=1`), undefined), null, "no bucket");
  const missing = { head: async () => null, get: async () => null };
  assert.equal(await handleExplainSlim(req(`${P}?slim=1`), missing), null, "missing object: the normal route answers 404");
  const broken = bucket("{not json");
  assert.equal(await handleExplainSlim(req(`${P}?slim=1`), broken), null, "unparseable object: the normal route answers");
});
