import assert from "node:assert/strict";
import { createSign, generateKeyPairSync } from "node:crypto";
import test from "node:test";
import {
  EDGE_BUCKETS, SCHEMA, breakdown, bucketOf, clusteredRange, edgeAt, t975, decodeDataset, defaultFilters, displayName, examples, pct, verdict,
} from "../app/bet-breakdowns-rules.ts";

// Bet breakdowns (owner-only page, owner 2026-10-21): decoding, filters, edge buckets, event-clustered ranges and the owner-only data API.

const META = {
  markets: ["t72_match", "r1_match", "win"], market_labels: { t72_match: "72-hole matchup", r1_match: "R1 2-ball", win: "Win" },
  tours: ["pga", "euro"], books: ["betonline", "pinnacle", "fanduel"], seasons: [2024, 2025], events: ["pga:1:2024-01-01", "pga:2:2025-01-01"],
  event_names: ["Event A", "Event B"], players: ["Scheffler, Scottie", "McIlroy, Rory"], hist_buckets: ["<60 rounds", "60-149", "150-299", "300+"],
  skill_buckets: ["Top 10 in field", "11-30", "31-60", "61+"], notes: ["note"],
};
const enc = (Ctor, values) => Buffer.from(new Ctor(values).buffer).toString("base64");

/** rows: {market, tour, book, season, event, p, open, close (0 = none), po, pc (null = none), hist, skill} */
function doc(rows) {
  const col = (k, f = (r) => r[k]) => rows.map(f);
  return {
    schema: SCHEMA, built_at: "2026-10-21T22:00:00Z", meta: META, n: rows.length,
    cols: {
      market: enc(Uint8Array, col("market")), tour: enc(Uint8Array, col("tour")), book: enc(Uint8Array, col("book")), season: enc(Uint8Array, col("season")),
      event: enc(Uint16Array, col("event")), player: enc(Uint16Array, rows.map(() => 0)), opp: enc(Uint16Array, rows.map(() => 1)),
      p: enc(Uint16Array, col("p", (r) => Math.round(r.p * 65535))), p_tie: enc(Uint16Array, col("pTie", (r) => Math.round(r.pTie * 65535))),
      void_tie: enc(Uint8Array, col("voidTie", (r) => (r.voidTie ? 1 : 0))), dec_open: enc(Uint16Array, col("open", (r) => Math.round(r.open * 100))),
      dec_close: enc(Uint16Array, col("close", (r) => Math.round(r.close * 100))), profit_open: enc(Int32Array, col("po", (r) => Math.round(r.po * 1000))),
      profit_close: enc(Int32Array, col("pc", (r) => (r.pc === null ? -2147483648 : Math.round(r.pc * 1000)))),
      hist: enc(Uint8Array, col("hist")), skill: enc(Uint8Array, col("skill")),
    },
  };
}
const row = (o) => ({ market: 0, tour: 0, book: 0, season: 0, event: 0, p: 0.56, open: 2.0, close: 1.9, po: 1, pc: 0.9, hist: 3, skill: 1, pTie: 0, voidTie: false, ...o });

test("decodes the published document and refuses a malformed one", () => {
  const d = decodeDataset(doc([row({}), row({ book: 1 })]));
  assert.equal(d.n, 2);
  assert.equal(d.book[1], 1);
  assert.equal(d.decOpen[0], 200);
  assert.throws(() => decodeDataset({ schema: "other" }), /expected format/);
  const bad = doc([row({})]); bad.cols.p = enc(Uint16Array, [1, 2]);
  assert.throws(() => decodeDataset(bad), /wrong length/);
});

test("edge buckets: lower bound in, upper bound out, 20%+ open-ended, below -5% excluded", () => {
  assert.equal(EDGE_BUCKETS[bucketOf(0)].key, "0");
  assert.equal(EDGE_BUCKETS[bucketOf(0.0999)].key, "5");
  assert.equal(EDGE_BUCKETS[bucketOf(0.1)].key, "10");
  assert.equal(EDGE_BUCKETS[bucketOf(3)].key, "20");
  assert.equal(EDGE_BUCKETS[bucketOf(-0.01)].key, "neg");
  assert.equal(bucketOf(-0.2), -1);
});

test("open vs close basis uses that price for the edge and that result", () => {
  // p 0.56: at 2.00 open edge +12%, at 1.90 close edge +6.4%
  const d = decodeDataset(doc([row({ po: 1, pc: 0.9 })]));
  const f = { ...defaultFilters(META), books: [0, 1, 2], markets: [0] };
  const open = breakdown(d, f);
  assert.equal(open.total[bucketOf(0.12)].n, 1);
  assert.equal(open.total[bucketOf(0.12)].roi, 1);
  const close = breakdown(d, { ...f, basis: "close" });
  assert.equal(close.total[bucketOf(0.064)].n, 1);
  assert.ok(Math.abs(close.total[bucketOf(0.064)].roi - 0.9) < 1e-9);
  // no close price -> not counted at the close, and not in any line-move filter
  const noClose = decodeDataset(doc([row({ close: 0, pc: null })]));
  assert.equal(breakdown(noClose, { ...f, basis: "close" }).considered, 0);
  assert.equal(breakdown(noClose, { ...f, move: "toward" }).considered, 0);
  assert.equal(breakdown(noClose, f).considered, 1);
});

test("void-on-tie matchups: edge is (1 - P(tie)) x (p x dec - 1), so a tie refund cannot inflate the bucket", () => {
  // p 0.53 at 2.00 = +6% plain; with a 10% tie that voids, the expected return is 0.9 x 6% = +5.4%; at 1.95, +3.35% plain -> +3.0%
  const d = decodeDataset(doc([row({ p: 0.53, open: 2.0, voidTie: true, pTie: 0.1 }), row({ p: 0.53, open: 2.0 })]));
  const f = defaultFilters(META);
  const b = breakdown(d, f);
  assert.equal(b.total[bucketOf(0.054)].n, 2, "both in 5-10%");
  assert.ok(Math.abs(b.total[bucketOf(0.054)].meanEdge - (0.054 + 0.06) / 2) < 1e-4);
  const edge = edgeAt(d, 0, 1.95);
  assert.ok(Math.abs(edge - 0.9 * (0.53 * 1.95 - 1)) < 1e-4);
  assert.equal(bucketOf(edgeAt(d, 0, 1.92)), bucketOf(0.0158), "0.9 x 1.76% stays in 0-5%");
});

test("line move filter: shorter close price = toward us", () => {
  const d = decodeDataset(doc([row({ open: 2.0, close: 1.8 }), row({ open: 2.0, close: 2.0 }), row({ open: 2.0, close: 2.2 })]));
  const f = { ...defaultFilters(META), books: [0], markets: [0] };
  const count = (move) => breakdown(d, { ...f, move }).considered;
  assert.equal(count("toward"), 1);
  assert.equal(count("flat"), 1);
  assert.equal(count("against"), 1);
  assert.equal(count("flat_or_against"), 2);
  assert.equal(count("all"), 3);
  const cell = breakdown(d, f).total[bucketOf(0.12)];
  assert.ok(Math.abs(cell.towardShare - 1 / 3) < 1e-9);
});

test("filters: market, book, season, tour, history, skill and odds range", () => {
  const d = decodeDataset(doc([
    row({}), row({ market: 1 }), row({ book: 2 }), row({ season: 1 }), row({ tour: 1 }), row({ hist: 0 }), row({ skill: 0 }), row({ open: 5, p: 0.22 }),
  ]));
  const base = { ...defaultFilters(META), markets: [0], books: [0], seasons: [0], tours: [0] };
  assert.equal(breakdown(d, base).considered, 4, "default: hist/skill any; odds any");
  assert.equal(breakdown(d, { ...base, hist: [0] }).considered, 1);
  assert.equal(breakdown(d, { ...base, skill: [0] }).considered, 1);
  assert.equal(breakdown(d, { ...base, maxDec: 3 }).considered, 3);
  assert.equal(breakdown(d, { ...base, minDec: 3 }).considered, 1);
  assert.equal(breakdown(d, { ...base, markets: [0, 1], books: [0, 2], seasons: [0, 1], tours: [0, 1] }).considered, 8);
});

test("defaults: tournament matchups, PGA, the owner's books only", () => {
  const f = defaultFilters(META);
  assert.deepEqual(f.markets, [0]);
  assert.deepEqual(f.tours, [0]);
  assert.deepEqual(f.books, [0, 1], "betonline and pinnacle; fanduel is not one of the owner's books");
  assert.equal(f.basis, "open");
});

test("rows by dimension, totals, seasons and the 'all edges >= 0' column", () => {
  const d = decodeDataset(doc([row({ book: 0, po: 1 }), row({ book: 1, po: -1 }), row({ book: 1, po: -1, season: 1 }), row({ book: 0, p: 0.49, po: -1 })]));
  const f = { ...defaultFilters(META), seasons: [0, 1] };
  const b = breakdown(d, f);
  assert.deepEqual(b.rows.map((r) => r.label).sort(), ["BetOnline", "Pinnacle"]);
  const pin = b.rows.find((r) => r.label === "Pinnacle");
  assert.equal(pin.total.n, 2);
  assert.deepEqual(pin.total.bySeason.map((s) => s.season), [2024, 2025]);
  const bo = b.rows.find((r) => r.label === "BetOnline");
  assert.equal(bo.total.n, 1, "the -2% edge row is in the -5..0 bucket, not in 'all edges >= 0'");
  assert.equal(b.grand.n, 3);
  const bySeason = breakdown(d, { ...f, rowDim: "season" });
  assert.deepEqual(bySeason.rows.map((r) => r.label), ["2024", "2025"]);
});

test("event-clustered range widens when bets in one event move together", () => {
  // 4 events, 10 bets each, every bet in an event has the same result: the effective sample is 4, not 40
  const sums = [[10, 10], [-10, 10], [10, 10], [-10, 10]];
  const [lo, hi] = clusteredRange(sums, 0, 40);
  assert.ok(hi - lo > 1.5, "much wider than an iid range on 40 bets");
  assert.equal(clusteredRange([[1, 1]], 1, 1), null);
  // two events, 15 bets each, event profits 6u and 3u: mean +30%; with 1 degree of freedom the range must cross zero
  const [lo2, hi2] = clusteredRange([[6, 15], [3, 15]], 9, 30);
  assert.ok(lo2 < 0 && hi2 > 0.6, `${lo2} ${hi2}`);
  assert.equal(t975(1), 12.706);
  assert.ok(Math.abs(t975(60) - 2.0) < 0.003);
  assert.ok(Math.abs(t975(1000) - 1.96) < 0.003);
});

test("verdict colours a cell only when the range clears zero and there are enough bets", () => {
  assert.equal(verdict({ n: 100, events: 40, lo: 0.01, hi: 0.2 }), "pos");
  assert.equal(verdict({ n: 100, events: 40, lo: -0.2, hi: -0.01 }), "neg");
  assert.equal(verdict({ n: 100, events: 40, lo: -0.1, hi: 0.1 }), "flat");
  assert.equal(verdict({ n: 10, events: 10, lo: 0.1, hi: 0.2 }), null, "too few bets");
  assert.equal(verdict({ n: 30, events: 2, lo: 0.1, hi: 0.5 }), null, "30 bets from two events is not evidence");
});

test("examples list a cell's bets, biggest edge first, with names the right way round", () => {
  const d = decodeDataset(doc([row({ p: 0.56 }), row({ p: 0.62 }), row({ p: 0.4 })]));
  const f = defaultFilters(META);
  const ex = examples(d, f, null, bucketOf(0.24));
  assert.equal(ex.total, 1);
  assert.equal(ex.list[0].player, "Scottie Scheffler");
  assert.equal(ex.list[0].opponent, "Rory McIlroy");
  assert.equal(ex.list[0].book, "BetOnline");
  assert.equal(displayName("Single"), "Single");
  assert.equal(pct(0.0123), "+1.2%");
  assert.equal(pct(null), "–");
  assert.equal(pct(-0.0001, 0), "0%", "no negative zero");
  assert.equal(pct(-0.004), "-0.4%");
});

/* ------------------------------------------------------------------ the built Worker: /api/owner-data */
const TEAM = "team.example.cloudflareaccess.com";
const AUD = "aud-123";
const OWNER = "owner@example.com";
const { publicKey, privateKey } = generateKeyPairSync("rsa", { modulusLength: 2048 });
const b64url = (value) => Buffer.from(typeof value === "string" ? value : JSON.stringify(value)).toString("base64url");
function signedHeaders(email) {
  const head = b64url({ alg: "RS256", kid: "k1" });
  const payload = b64url({ email, aud: [AUD], iss: `https://${TEAM}`, exp: Math.floor(Date.now() / 1000) + 3600, sub: "u1" });
  const sig = createSign("RSA-SHA256").update(`${head}.${payload}`).sign(privateKey).toString("base64url");
  return { "cf-access-authenticated-user-email": email, "cf-access-jwt-assertion": `${head}.${payload}.${sig}` };
}
const realFetch = globalThis.fetch;
test.before(() => {
  globalThis.fetch = async (input, init) => {
    const href = typeof input === "string" ? input : input.url;
    if (href === `https://${TEAM}/cdn-cgi/access/certs`) return Response.json({ keys: [{ ...publicKey.export({ format: "jwk" }), kid: "k1" }] });
    return realFetch(input, init);
  };
});
test.after(() => { globalThis.fetch = realFetch; });

async function loadWorker() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}-${Math.random()}`);
  return (await import(workerUrl.href)).default;
}

test("owner-data API fails closed when the owner list is missing, empty or blank, even for a signed request", async () => {
  const worker = await loadWorker();
  let reads = 0;
  const bucket = { get: async () => { reads += 1; return { body: new Response("PRIVATE").body, httpEtag: '"e1"', writeHttpMetadata() {} }; } };
  const ctx = { waitUntil() {}, passThroughOnException() {} };
  for (const owners of [undefined, "", "  ", " , "]) {
    const env = { ASSETS: { fetch: async () => new Response("asset") }, DASHBOARD_DATA: bucket, ACCESS_TEAM_DOMAIN: TEAM, ACCESS_AUD: AUD, ...(owners === undefined ? {} : { OWNER_EMAILS: owners }) };
    for (const headers of [{}, signedHeaders(OWNER)]) {
      const res = await worker.fetch(new Request("https://golf.example/api/owner-data/bet_breakdowns.json", { headers }), env, ctx);
      assert.equal(res.status, 403, `owners=${JSON.stringify(owners)}`);
    }
  }
  assert.equal(reads, 0, "nothing read from R2");
});

test("owner-data API: the owner reads R2 owner/<key>; others get 403; no packaged fallback; bad paths refused", async () => {
  const worker = await loadWorker();
  const reads = [];
  const body = JSON.stringify({ ok: 1 });
  const bucket = {
    get: async (key) => {
      reads.push(key);
      if (key !== "owner/bet_breakdowns.json") return null;
      return { body: new Response(body).body, httpEtag: '"e1"', writeHttpMetadata() {} };
    },
  };
  let assetCalls = 0;
  const env = { ASSETS: { fetch: async () => { assetCalls += 1; return new Response("public copy", { status: 200 }); } }, DASHBOARD_DATA: bucket, ACCESS_TEAM_DOMAIN: TEAM, ACCESS_AUD: AUD, OWNER_EMAILS: OWNER };
  const ctx = { waitUntil() {}, passThroughOnException() {} };
  const get = (path, headers) => worker.fetch(new Request(`https://golf.example${path}`, { headers }), env, ctx);

  const ok = await get("/api/owner-data/bet_breakdowns.json", signedHeaders(OWNER));
  assert.equal(ok.status, 200);
  assert.equal(await ok.text(), body);
  assert.equal(ok.headers.get("cache-control"), "private, no-cache");
  assert.deepEqual(reads, ["owner/bet_breakdowns.json"]);

  const colleague = await get("/api/owner-data/bet_breakdowns.json", signedHeaders("colleague@example.com"));
  assert.equal(colleague.status, 403);
  assert.equal(reads.length, 1, "nothing read for a colleague");

  const missing = await get("/api/owner-data/other.json", signedHeaders(OWNER));
  assert.equal(missing.status, 404);
  assert.equal(assetCalls, 0, "never falls back to a packaged (public) asset");

  for (const bad of ["/api/owner-data/a..b.json", "/api/owner-data/sub/%2e%2e%2fx.json", "/api/owner-data/x.txt", "/api/owner-data/%E0%A4%A"]) {
    assert.equal((await get(bad, signedHeaders(OWNER))).status, 400, bad);
  }
});
