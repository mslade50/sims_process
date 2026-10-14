// /api/golfprice/changes.json: validation, build + immutable store, ETag/304, errors, gzip documents, size against the second document.
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { readFile, readdir } from "node:fs/promises";
import { gzipSync } from "node:zlib";
import test from "node:test";
import { CHANGES_PATH, DERIVED_PREFIX, changesHash, handleChanges, validateChangesQuery } from "../worker/changes-api.ts";
import { cacheControlFor } from "../worker/data-http.ts";
import { CHANGES_SCHEMA, parseChangesPayload } from "../app/changes-rules.ts";
import { buildPair, clone } from "./fixtures/changes-pair.mjs";

const EVENT = "pga:554:2026-10-01";
const DIR = "golfprice/explain/pga_554_2026-10-01";
const FROM = `${DIR}/week_A.json`;
const TO = `${DIR}/week_B.json`;

/** In-memory R2 with get/put and a get counter. Values are strings or Buffers. */
function fakeBucket(initial = {}) {
  const store = new Map(Object.entries(initial).map(([k, v]) => [k, Buffer.isBuffer(v) ? v : Buffer.from(typeof v === "string" ? v : JSON.stringify(v))]));
  const gets = [];
  let version = 0;
  return {
    store, gets, putFails: false,
    async get(key) {
      gets.push(key);
      const body = store.get(key);
      if (!body) return null;
      return {
        key, httpEtag: `"v${version}-${key.length}-${body.length}"`, etag: `v${version}`,
        body: new ReadableStream({ start(c) { c.enqueue(new Uint8Array(body)); c.close(); } }),
        arrayBuffer: async () => body.buffer.slice(body.byteOffset, body.byteOffset + body.byteLength),
        text: async () => body.toString("utf8"),
        writeHttpMetadata() {},
      };
    },
    async put(key, value) {
      if (this.putFails) throw new Error("put failed");
      version += 1;
      store.set(key, Buffer.from(String(value)));
      return { key };
    },
  };
}
const url = (event, from, to) => `https://golf.example/api/golfprice/${CHANGES_PATH}?event=${encodeURIComponent(event ?? "")}&from=${encodeURIComponent(from ?? "")}&to=${encodeURIComponent(to ?? "")}`;
const ask = (bucket, event, from, to, headers = {}) => handleChanges(new Request(url(event, from, to), { headers }), bucket);
function pairBucket(extra = {}) {
  const { prev, cur } = buildPair();
  return fakeBucket({ [FROM]: prev, [TO]: cur, ...extra });
}

test("inputs are validated: event, both keys, the event's own folder, no latest.json, from != to", async () => {
  const bucket = pairBucket();
  const bad = [
    [null, FROM, TO], [EVENT, null, TO], [EVENT, FROM, null], ["a b", FROM, TO],
    [EVENT, "../x.json", TO], [EVENT, FROM, "golfprice/index.json"],
    [EVENT, `${DIR}/../other/week_A.json`, TO], [EVENT, "golfprice/explain/pga_999_x/week_A.json", TO],
    [EVENT, `${DIR}/latest.json`, TO], [EVENT, FROM, FROM], [EVENT, `${DIR}/week_A.txt`, TO],
  ];
  for (const [e, f, t] of bad) {
    const r = await ask(bucket, e, f, t);
    assert.equal(r.status, 400, `${e} ${f} ${t}`);
    assert.equal((await r.json()).ok, false);
  }
  assert.equal(validateChangesQuery(EVENT, FROM, TO), null);
  assert.equal(bucket.gets.length, 0, "nothing is read for a bad request");
});

test("builds the diff, stores it under the content-addressed key, and serves the stored copy next time", async () => {
  const bucket = pairBucket();
  const hash = await changesHash(FROM, TO);
  assert.match(hash, /^[0-9a-f]{20}$/);
  const key = `${DERIVED_PREFIX}${hash}.json`;
  assert.equal(cacheControlFor(key.replace(/^golfprice\//, "")), "public, max-age=31536000, immutable");
  const first = await ask(bucket, EVENT, FROM, TO);
  assert.equal(first.status, 200);
  assert.equal(first.headers.get("cache-control"), "public, max-age=31536000, immutable");
  assert.equal(first.headers.get("x-changes-key"), key);
  const body = await first.json();
  assert.equal(body.schema, CHANGES_SCHEMA);
  assert.equal(body.event, EVENT);
  assert.ok(bucket.store.has(key), "stored");
  assert.ok(body.change.movers.win.length > 0 && body.change.movers.win[0].causes.length > 0);
  assert.ok(parseChangesPayload(body));
  assert.deepEqual(await (await ask(bucket, EVENT, FROM, TO)).json(), body);
  bucket.gets.length = 0;
  const second = await ask(bucket, EVENT, FROM, TO);
  assert.equal(second.status, 200);
  assert.deepEqual(bucket.gets, [key], "the second request reads only the stored diff, not the two run documents");
  // the reverse direction is a different diff
  assert.notEqual(await changesHash(TO, FROM), hash);
});

test("If-None-Match answers 304 for the stored diff", async () => {
  const bucket = pairBucket();
  await ask(bucket, EVENT, FROM, TO);
  const stored = await ask(bucket, EVENT, FROM, TO);
  const etag = stored.headers.get("etag");
  assert.ok(etag);
  const hit = await ask(bucket, EVENT, FROM, TO, { "if-none-match": etag });
  assert.equal(hit.status, 304);
  assert.equal(hit.headers.get("cache-control"), "public, max-age=31536000, immutable");
  assert.equal((await ask(bucket, EVENT, FROM, TO, { "if-none-match": '"other"' })).status, 200);
});

test("a missing run document is 404; a document that is not an explain document, or one that cannot be compared, is not a 200", async () => {
  const { prev, cur } = buildPair();
  assert.equal((await ask(fakeBucket({ [FROM]: prev }), EVENT, FROM, TO)).status, 404);
  assert.equal((await ask(fakeBucket({ [TO]: cur }), EVENT, FROM, TO)).status, 404);
  const broken = fakeBucket({ [FROM]: prev, [TO]: "{not json" });
  const r = await ask(broken, EVENT, FROM, TO);
  assert.equal(r.status, 502);
  assert.equal((await r.json()).ok, false);
  assert.equal((await ask(fakeBucket({ [FROM]: prev, [TO]: { schema: "nope" } }), EVENT, FROM, TO)).status, 502);
  const other = clone(cur);
  other.event_uid = "pga:other";
  assert.equal((await ask(fakeBucket({ [FROM]: prev, [TO]: other }), EVENT, FROM, TO)).status, 400);
  const same = clone(prev);
  assert.equal((await ask(fakeBucket({ [FROM]: prev, [TO]: same }), EVENT, FROM, TO)).status, 400);
  assert.equal([...broken.store.keys()].filter((k) => k.startsWith(DERIVED_PREFIX)).length, 0, "failures store nothing");
});

test("gzip-encoded run documents are read, and a failed store still serves the diff", async () => {
  const { prev, cur } = buildPair();
  const bucket = fakeBucket({ [FROM]: gzipSync(JSON.stringify(prev)), [TO]: gzipSync(JSON.stringify(cur)) });
  bucket.putFails = true;
  const r = await ask(bucket, EVENT, FROM, TO);
  assert.equal(r.status, 200);
  assert.ok((await r.json()).change.movers.win.length > 0);
});

test("routing: the generic golfprice read hands changes.json to this module before looking in the bucket", async () => {
  const src = await readFile(new URL("../worker/overrides-api.ts", import.meta.url), "utf8");
  assert.match(src, /requested === CHANGES_PATH\) return await handleChanges\(request, env\.DASHBOARD_DATA\)/);
  assert.ok(src.indexOf("CHANGES_PATH) return") < src.indexOf("DASHBOARD_DATA.get(`golfprice/${requested}`)"));
});

test("size: the served diff against the second explain document (real published event when present)", async () => {
  const dir = new URL("../public/golfprice/explain/", import.meta.url);
  let raw = null;
  let bytes = 0;
  if (existsSync(dir)) {
    for (const d of await readdir(dir)) {
      const files = (await readdir(new URL(`${d}/`, dir))).filter((f) => f.startsWith("week_"));
      if (files.length) {
        const text = await readFile(new URL(`${d}/${files[0]}`, dir), "utf8");
        if (text.length > bytes) { raw = JSON.parse(text); bytes = text.length; }
      }
    }
  }
  if (!raw) raw = buildPair().prev; // no snapshot: the fixture stands in, only the ratio check below is skipped
  const prev = clone(raw);
  const cur = clone(raw);
  cur.run = "later"; cur.as_of = new Date(Date.parse(raw.as_of ?? "2026-10-01T00:00:00Z") + 3600_000).toISOString();
  const live = cur.players.filter((p) => !p.withdrawn);
  live[0].withdrawn = true;
  live.slice(1, 60).forEach((p, i) => { for (const m of Object.keys(p.probs)) if (p.probs[m].model) p.probs[m].model *= 1 + 0.02 * ((i % 7) - 3); if (p.probs.win.market) p.probs.win.market *= 1 + 0.01 * ((i % 5) - 2); p.mu += 0.02 * ((i % 5) - 2); p.components.kalman = (p.components.kalman ?? 0) + 0.02 * ((i % 5) - 2); });
  const ev = raw.event_uid;
  const dir2 = `golfprice/explain/${ev.replace(/:/g, "_")}`;
  const f = `${dir2}/week_A.json`, t = `${dir2}/week_B.json`;
  const bucket = fakeBucket({ [f]: prev, [t]: cur });
  const r = await ask(bucket, ev, f, t);
  assert.equal(r.status, 200);
  const text = await r.text();
  const served = Buffer.byteLength(text);
  const second = Buffer.byteLength(JSON.stringify(cur));
  console.log(`changes.json ${served} bytes (${gzipSync(text).length} gzipped) vs second document ${second} bytes (${gzipSync(JSON.stringify(cur)).length} gzipped): ${(100 * served / second).toFixed(1)}%`);
  if (bytes) assert.ok(served < second * 0.1, `${served} vs ${second}`);
  assert.ok(served < 60_000, `${served} bytes`);
});
