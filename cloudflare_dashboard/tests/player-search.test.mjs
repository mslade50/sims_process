// Derived golfer search index: streaming scanner, compact table, Worker endpoint, client decoder, real-snapshot size budget.
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { gzipSync } from "node:zlib";
import test from "node:test";
import { buildFromObject, derivedKey, handleSearchIndex, readPointerGeneration, scanCatalog } from "../worker/player-search.ts";
import { parseCatalog, parseSearchIndex, profileMatches, sortProfiles } from "../app/player-profile-rules.ts";
import { searchPlayers } from "../app/shell-rules.ts";

const GEN = "44ffb7d799d33179b86d";
const OTHER = "11111111111111111111";
const player = (over) => ({ aliases: [], archetype: { label: "x", n: [1, 2, { a: "}" }] }, available_axes: 0, last_observation: "2024-05-05", n_rounds: 10, pga_benchmark: { note: "a \"quoted\" } string, with [brackets]", v: 1.5 }, photo: { url: null }, status: "unavailable", tours: ["pga"], ...over });
const fixture = () => ({
  as_of: "2026-10-09T18:17:00+00:00",
  count: 5,
  deep_coverage: { shots: { id: "s1", metrics: { a: { n_players: 3 } } } },
  generation: GEN,
  pga_benchmark_reference: { id: "pga_2025_r12:abc", n_rounds: 100 },
  players: [
    player({ dg_id: 15, name: "Bolling, Charlie", aliases: ["Bolling, Charlie", "Bolling, Chuck"], profile_id: "15", profile_key: "golfprice/player_profiles/a/batch_0001.json", status: "complete", tours: ["euro", "pga"], country: "United States", country_code: "USA" }),
    player({ dg_id: 24, name: "Brooks, Mark", profile_id: "24", profile_key: "golfprice/player_profiles/a/batch_0001.json", amateur: true }),
    player({ dg_id: 31, name: "Åberg, Ludvig", display_name: "Ludvig Åberg", profile_id: "31", profile_key: "golfprice/player_profiles/a/batch_0002.json", last_observation: null, n_rounds: 0, tours: [] }),
    player({ dg_id: null, name: "Al Shahrani, Abdulrahman", profile_id: "historical-5170aa8892ac5c6bba19", profile_key: "golfprice/player_profiles/a/batch_0002.json", last_observation: "2024-05-05T10:00:00Z" }),
    player({ dg_id: 99, name: "Late, Add", profile_id: "99", profile_key: "golfprice/player_profiles/a/batch_0003.json", status: "partial" }),
  ],
  schema_version: "player_profiles.v1",
});

const streamOf = (text, size = 0) => new ReadableStream({
  start(controller) {
    const bytes = new TextEncoder().encode(text);
    if (!size) controller.enqueue(bytes);
    else for (let i = 0; i < bytes.length; i += size) controller.enqueue(bytes.slice(i, i + size));
    controller.close();
  },
});
async function* pieces(text, size) { for (let i = 0; i < text.length; i += size) yield text.slice(i, i + size); }

test("scanner: same values as JSON.parse for every chunk size, including 1-character chunks", async () => {
  const doc = fixture();
  const text = JSON.stringify(doc);
  for (const size of [1, 2, 3, 7, 64, 1000, text.length]) {
    const meta = {}, players = [];
    await scanCatalog(pieces(text, size), { onMeta: (k, v) => { meta[k] = v; }, onPlayer: (p) => { players.push(p); } });
    const { players: expectedPlayers, ...expectedMeta } = doc;
    assert.deepEqual(meta, expectedMeta, `meta, chunk ${size}`);
    assert.deepEqual(players, expectedPlayers, `players, chunk ${size}`);
  }
});

test("scanner: bare primitives, whitespace, escapes in keys and strings, and early stop", async () => {
  const text = ' { "a\\"b" : 1.5e3 ,\n "players" : [ {"name":"x\\\\","n":[1,{"k":"}]"}]} , {"name":"y"} ] , "flag":true, "none" : null, "s":"t,\\"u" } ';
  const meta = {}, players = [];
  const stopped = await scanCatalog(pieces(text, 5), { onMeta: (k, v) => { meta[k] = v; }, onPlayer: (p) => { players.push(p); } });
  assert.equal(stopped, false);
  assert.deepEqual(meta, { 'a"b': 1500, flag: true, none: null, s: 't,"u' });
  assert.deepEqual(players, [{ name: "x\\", n: [1, { k: "}]" }] }, { name: "y" }]);
  let seen = 0;
  assert.equal(await scanCatalog(pieces(text, 5), { onPlayer: () => (++seen === 1) }), true);
  assert.equal(seen, 1);
  await assert.rejects(scanCatalog(pieces('{"a":{bad},"players":[]}', 4), { onMeta: () => undefined }));
});

test("compact table: column-major, deltas, sparse fields, and the client decoder reconstructs the identity fields", async () => {
  const doc = fixture();
  const index = await buildFromObject({ body: streamOf(JSON.stringify(doc), 97) }, GEN, "x", new Date("2026-10-14T00:00:00Z"));
  assert.equal(index.schema, "player_search.v1");
  assert.equal(index.generation, GEN);
  assert.equal(index.count, 5);
  assert.deepEqual(index.dg, [15, 9, 7, null, 68]);
  assert.deepEqual(index.pid, [[3, "historical-5170aa8892ac5c6bba19"]]);
  assert.deepEqual(index.alias, [[0, ["Bolling, Chuck"]]], "the name itself is dropped from aliases");
  assert.deepEqual(index.am, [1]);
  assert.equal(index.keys.length, 3);
  assert.equal("pga_benchmark" in index, false);
  assert.equal(JSON.stringify(index).includes("quoted"), false, "no benchmark text leaks into the index");
  assert.equal(index.meta.generation, undefined);
  assert.deepEqual(index.meta.pga_benchmark_reference, doc.pga_benchmark_reference);

  const catalog = parseSearchIndex(JSON.parse(JSON.stringify(index)));
  const full = parseCatalog(doc);
  assert.equal(catalog.slim, true);
  assert.equal(catalog.as_of, doc.as_of);
  assert.deepEqual(catalog.pga_benchmark_reference, doc.pga_benchmark_reference);
  assert.deepEqual(catalog.deep_coverage, doc.deep_coverage);
  assert.equal(catalog.players.length, full.players.length);
  full.players.forEach((p, i) => {
    const q = catalog.players[i];
    for (const key of ["dg_id", "profile_id", "name", "display_name", "country", "country_code", "status", "profile_key", "n_rounds", "amateur"]) assert.equal(q[key], p[key], `${key} of ${p.name}`);
    assert.deepEqual(q.tours, p.tours);
    assert.equal(q.last_observation ?? null, p.last_observation ?? null, `last_observation of ${p.name}`);
    assert.deepEqual(q.aliases ?? [], (p.aliases ?? []).filter((a) => a !== p.name));
    assert.equal(profileMatches(q, "chuck bolling"), profileMatches(p, "chuck bolling"));
  });
  // Search behaves the same on the slim list as on the full one.
  for (const query of ["bol", "aberg", "chuck", "shahrani", "euro", "united", "99"]) {
    assert.deepEqual(searchPlayers(catalog.players, query).map((p) => p.profile_id), searchPlayers(full.players, query).map((p) => p.profile_id), query);
  }
  assert.deepEqual(sortProfiles(catalog.players).map((p) => p.profile_id), sortProfiles(full.players).map((p) => p.profile_id));
  assert.equal(parseSearchIndex({ schema: "nope" }), null);
  assert.equal(parseSearchIndex(null), null);
  assert.equal(parseSearchIndex({ ...JSON.parse(JSON.stringify(index)), dg: [] }), null);
});

/* ------------------------------------------------------------------ Worker endpoint with a fake bucket */
function fakeBucket(files, { gzipKeys = [] } = {}) {
  const store = new Map(Object.entries(files));
  const reads = [];
  const puts = [];
  const bucket = {
    reads, puts, store,
    async get(key) {
      reads.push(key);
      const value = store.get(key);
      if (value === undefined) return null;
      const raw = typeof value === "string" ? Buffer.from(value) : value;
      const bytes = gzipKeys.includes(key) ? gzipSync(raw) : raw;
      return {
        key, httpEtag: `"etag-${key.length}-${raw.length}"`, httpMetadata: gzipKeys.includes(key) ? { contentEncoding: "gzip" } : {},
        body: new ReadableStream({ start(c) { for (let i = 0; i < bytes.length; i += 4096) c.enqueue(new Uint8Array(bytes.subarray(i, i + 4096))); c.close(); } }),
        writeHttpMetadata(h) { if (gzipKeys.includes(key)) h.set("content-encoding", "gzip"); },
      };
    },
    async put(key, value) { puts.push(key); store.set(key, typeof value === "string" ? value : Buffer.from(value)); return {}; },
  };
  return bucket;
}
const req = (path, headers = {}) => new Request(`https://golf.example/api/golfprice/${path}`, { headers });
const catalogText = () => JSON.stringify(fixture());

test("endpoint: builds once per generation, stores the derived object, then serves it with immutable cache headers", async () => {
  const bucket = fakeBucket({ [`golfprice/player_profiles/${GEN}/catalog.json`]: catalogText() });
  const first = await handleSearchIndex(req(`player_profiles/search-index.json?gen=${GEN}`), bucket);
  assert.equal(first.status, 200);
  assert.equal(first.headers.get("cache-control"), "public, max-age=31536000, immutable");
  assert.equal(first.headers.get("x-search-generation"), GEN);
  const body = await first.json();
  assert.equal(body.schema, "player_search.v1");
  assert.equal(body.count, 5);
  assert.deepEqual(bucket.puts, [derivedKey(GEN)]);
  assert.equal(derivedKey(GEN), `golfprice/player_profiles/derived/${GEN}/search-index.json`);
  const readsBefore = bucket.reads.filter((k) => k.endsWith("catalog.json")).length;
  const second = await handleSearchIndex(req(`player_profiles/search-index.json?gen=${GEN}`), bucket);
  assert.equal(second.status, 200);
  assert.equal((await second.json()).count, 5);
  assert.equal(bucket.reads.filter((k) => k.endsWith("catalog.json")).length, readsBefore, "the catalog is not read again");
  assert.equal(bucket.puts.length, 1);
  const etag = second.headers.get("etag");
  const revalidated = await handleSearchIndex(req(`player_profiles/search-index.json?gen=${GEN}`, { "if-none-match": etag }), bucket);
  assert.equal(revalidated.status, 304);
});

test("endpoint: concurrent first requests build once", async () => {
  const bucket = fakeBucket({ [`golfprice/player_profiles/${GEN}/catalog.json`]: catalogText() });
  const all = await Promise.all([1, 2, 3, 4].map(() => handleSearchIndex(req(`player_profiles/search-index.json?gen=${GEN}`), bucket)));
  assert.deepEqual(all.map((r) => r.status), [200, 200, 200, 200]);
  assert.equal(bucket.reads.filter((k) => k.endsWith("catalog.json")).length, 1);
  assert.equal(bucket.puts.length, 1);
});

test("endpoint: without gen the live pointer decides, with a short revalidated cache; gzip-stored catalogs work", async () => {
  const bucket = fakeBucket({ "golfprice/player_profiles/index.json": catalogText() }, { gzipKeys: ["golfprice/player_profiles/index.json"] });
  assert.equal(await readPointerGeneration(bucket), GEN);
  const res = await handleSearchIndex(req("player_profiles/search-index.json"), bucket);
  assert.equal(res.status, 200);
  assert.equal(res.headers.get("cache-control"), "public, max-age=15, stale-while-revalidate=120");
  assert.equal((await res.json()).generation, GEN);
  assert.deepEqual(bucket.puts, [derivedKey(GEN)]);
  // The generation folder is missing: the pointer object is used only when it IS that generation.
  const stale = await handleSearchIndex(req(`player_profiles/search-index.json?gen=${OTHER}`), bucket);
  assert.equal(stale.status, 404);
  assert.equal(bucket.puts.length, 1, "nothing is stored under the wrong generation");
});

test("endpoint: bad input and missing data answer JSON errors the client can fall back from", async () => {
  const empty = fakeBucket({});
  assert.equal((await handleSearchIndex(req("player_profiles/search-index.json?gen=../../x"), empty)).status, 400);
  const missing = await handleSearchIndex(req("player_profiles/search-index.json"), empty);
  assert.equal(missing.status, 404);
  assert.equal((await missing.json()).error, "not published yet");
  assert.equal((await handleSearchIndex(req(`player_profiles/search-index.json?gen=${GEN}`), empty)).status, 404);
  const garbage = fakeBucket({ [`golfprice/player_profiles/${GEN}/catalog.json`]: '{"schema_version":"other","players":[]}' });
  const bad = await handleSearchIndex(req(`player_profiles/search-index.json?gen=${GEN}`), garbage);
  assert.equal(bad.status, 502);
  assert.equal(garbage.puts.length, 0);
});

/* ------------------------------------------------------------------ the real snapshot (git-ignored; skipped when absent) */
const snapshot = new URL("../public/golfprice/player_profiles/index.json", import.meta.url);
test("real catalog: same golfers, field for field, and the index fits the budget", { skip: !existsSync(snapshot) }, async () => {
  const raw = await readFile(snapshot);
  const full = parseCatalog(JSON.parse(raw.toString("utf8")));
  const started = performance.now();
  const index = await buildFromObject({ body: streamOf(raw.toString("utf8"), 65536) }, "44ffb7d799d33179b86d", "snapshot", new Date());
  const buildMs = performance.now() - started;
  const text = JSON.stringify(index);
  const gz = gzipSync(text).length;
  console.log(`search index: ${full.players.length} golfers, catalog ${(raw.length / 1e6).toFixed(1)} MB (${(gzipSync(raw).length / 1e3).toFixed(0)} KB gzip) -> index ${(text.length / 1e3).toFixed(0)} KB raw, ${(gz / 1e3).toFixed(0)} KB gzip, built in ${buildMs.toFixed(0)} ms`);
  assert.ok(gz <= 400_000, `index gzip ${gz} bytes is over the 400 KB budget`);
  const slim = parseSearchIndex(JSON.parse(text));
  assert.equal(slim.players.length, full.players.length);
  for (let i = 0; i < full.players.length; i++) {
    const p = full.players[i], q = slim.players[i];
    assert.equal(q.dg_id, p.dg_id); assert.equal(q.name, p.name); assert.equal(q.profile_id, p.profile_id); assert.equal(q.profile_key, p.profile_key);
    assert.equal(q.status, p.status); assert.equal(q.n_rounds, p.n_rounds); assert.equal(q.last_observation ?? null, p.last_observation ?? null);
    assert.equal(q.country, p.country); assert.deepEqual(q.tours, p.tours);
  }
  assert.deepEqual(slim.pga_benchmark_reference, full.pga_benchmark_reference);
  for (const query of ["scheffler", "mcilroy", "jon rahm", "aberg", "kim", "al shahrani"]) {
    assert.deepEqual(searchPlayers(slim.players, query).map((p) => p.profile_id), searchPlayers(full.players, query).map((p) => p.profile_id), query);
  }
});

/* ------------------------------------------------------------------ benchmarks for a field */
import { benchmarksKey, buildBenchmarkSlice, handleBenchmarks, parseIdList } from "../worker/player-search.ts";
import { parseBenchmarks } from "../app/player-profile-rules.ts";

const bench = (id) => ({ ...fixture().players.find((p) => p.dg_id === id) });

test("benchmarks: id list parsing is strict, sorted, unique and capped at 400", () => {
  assert.deepEqual(parseIdList("31,15,15,24"), [15, 24, 31]);
  for (const bad of [null, "", "0", "1,,2", "1,a", "-3", "1.5", Array.from({ length: 401 }, (_, i) => i + 1).join(",")]) assert.equal(parseIdList(bad), null, String(bad));
  assert.equal(parseIdList(Array.from({ length: 400 }, (_, i) => i + 1).join(",")).length, 400);
});

test("benchmarks: only the requested golfers' pga_benchmark, in id order, with the reference", async () => {
  const slice = await buildBenchmarkSlice({ body: streamOf(catalogText(), 50) }, GEN, [31, 15, 12345]);
  assert.equal(slice.schema, "player_benchmarks.v1");
  assert.deepEqual(slice.players.map((p) => p.dg_id), [15, 31]);
  assert.deepEqual(slice.players[0].pga_benchmark, bench(15).pga_benchmark);
  assert.deepEqual(slice.pga_benchmark_reference, fixture().pga_benchmark_reference);
  assert.equal(slice.as_of, fixture().as_of);
  const parsed = parseBenchmarks(JSON.parse(JSON.stringify(slice)));
  assert.equal(parsed.players.length, 2);
  assert.deepEqual(parsed.pga_benchmark_reference, fixture().pga_benchmark_reference);
  assert.equal(parseBenchmarks({ schema: "x" }), null);
  await assert.rejects(buildBenchmarkSlice({ body: streamOf(catalogText()) }, OTHER, [15]));
});

test("benchmarks endpoint: derived key by generation + id hash, immutable with gen, 304, no second catalog read, validation", async () => {
  const bucket = fakeBucket({ [`golfprice/player_profiles/${GEN}/catalog.json`]: catalogText() });
  const url = `player_profiles/benchmarks.json?gen=${GEN}&ids=24,15`;
  const first = await handleBenchmarks(req(url), bucket);
  assert.equal(first.status, 200);
  assert.equal(first.headers.get("cache-control"), "public, max-age=31536000, immutable");
  assert.deepEqual((await first.json()).players.map((p) => p.dg_id), [15, 24]);
  assert.equal(bucket.puts.length, 1);
  assert.match(bucket.puts[0], new RegExp(`^golfprice/player_profiles/derived/${GEN}/benchmarks-[0-9a-f]{20}[.]json$`));
  assert.equal(benchmarksKey(GEN, "a".repeat(20)).endsWith(`benchmarks-${"a".repeat(20)}.json`), true);
  const second = await handleBenchmarks(req(`player_profiles/benchmarks.json?gen=${GEN}&ids=15,24`), bucket);
  assert.equal(bucket.reads.filter((k) => k.endsWith("catalog.json")).length, 1, "same ids in another order reuse the derived object");
  const again = await handleBenchmarks(req(url, { "if-none-match": second.headers.get("etag") }), bucket);
  assert.equal(again.status, 304);
  assert.equal((await handleBenchmarks(req(`player_profiles/benchmarks.json?gen=${GEN}`), bucket)).status, 400);
  assert.equal((await handleBenchmarks(req(`player_profiles/benchmarks.json?gen=zz&ids=1`), bucket)).status, 400);
  assert.equal((await handleBenchmarks(req(`player_profiles/benchmarks.json?gen=${OTHER}&ids=15`), bucket)).status, 404);
  const pointer = fakeBucket({ "golfprice/player_profiles/index.json": catalogText() });
  const live = await handleBenchmarks(req("player_profiles/benchmarks.json?ids=15"), pointer);
  assert.equal(live.status, 200);
  assert.equal(live.headers.get("cache-control"), "public, max-age=15, stale-while-revalidate=120");
});

test("benchmarks: size for a ~150 player field from the real snapshot", { skip: !existsSync(snapshot) }, async () => {
  const text = (await readFile(snapshot)).toString("utf8");
  const full = parseCatalog(JSON.parse(text));
  const ids = full.players.filter((p) => p.pga_benchmark && p.dg_id).slice(0, 150).map((p) => p.dg_id);
  const slice = await buildBenchmarkSlice({ body: streamOf(text, 65536) }, "44ffb7d799d33179b86d", ids);
  const raw = JSON.stringify(slice);
  console.log(`benchmarks for ${ids.length} golfers: ${(raw.length / 1e3).toFixed(0)} KB raw, ${(gzipSync(raw).length / 1e3).toFixed(0)} KB gzip (catalog 16.3 MB raw, 1199 KB gzip)`);
  assert.equal(slice.players.length, 150);
  slice.players.forEach((p) => assert.deepEqual(p.pga_benchmark, full.players.find((q) => q.dg_id === p.dg_id).pga_benchmark));
});
