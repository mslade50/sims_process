import assert from "node:assert/strict";
import test from "node:test";
import { createHash } from "node:crypto";

const {
  default: worker,
  resolveMarketGeneration,
} = await import("./worker.js");

function object(bytes) {
  return {
    async arrayBuffer() {
      return Uint8Array.from(bytes).buffer;
    },
  };
}

function environment({ corrupt = false, missing = false, filename = "round_matchups.json" } = {}) {
  const market = Buffer.from(JSON.stringify({ round: 4, matchups: [{ player: "a" }] }) + "\n");
  const binding = {
    key: `generations/generation-1/${filename}`,
    sha256: createHash("sha256").update(market).digest("hex"),
    size: market.length,
  };
  const pointer = Buffer.from(JSON.stringify({
    schema_version: "odds-screen-generation/v1",
    generation: "generation-1",
    files: { [filename]: binding },
  }));
  return {
    ODDS_DATA: {
      async get(key) {
        if (key === "odds_data/meta.json") return object(pointer);
        if (key === `odds_data/${binding.key}`) {
          if (missing) return null;
          return object(corrupt ? Buffer.from("wrong") : market);
        }
        // A stale fixed object exists, but the resolver must never read it.
        if (key === "odds_data/round_matchups.json") {
          throw new Error("fixed market key was consulted");
        }
        return null;
      },
    },
  };
}

test("fixed reader URL resolves the exact pointer generation", async () => {
  const result = await resolveMarketGeneration(environment(), "round_matchups.json");
  assert.equal(result.pointer.generation, "generation-1");
  assert.equal(JSON.parse(Buffer.from(result.bytes).toString()).round, 4);
});

test("reader resolves a future JSON payload declared only by the pointer", async () => {
  const filename = "future_market.json";
  const result = await resolveMarketGeneration(
    environment({ filename }),
    filename,
  );
  assert.equal(result.pointer.files[filename].key, `generations/generation-1/${filename}`);
});

test("missing or corrupt activated objects fail closed", async () => {
  await assert.rejects(
    resolveMarketGeneration(environment({ missing: true }), "round_matchups.json"),
    /missing/,
  );
  await assert.rejects(
    resolveMarketGeneration(environment({ corrupt: true }), "round_matchups.json"),
    /(size|hash) mismatch/,
  );
});

test("snapshot reads the active pointer once and returns every bound payload", async () => {
  const payloads = {
    "meta.json": Buffer.from(JSON.stringify({ event_id: "123", round: 4 })),
    "round_matchups.json": Buffer.from(
      JSON.stringify({ matchups: [{ player: "a" }] }),
    ),
  };
  const files = Object.fromEntries(
    Object.entries(payloads).map(([filename, bytes]) => [
      filename,
      {
        key: `generations/generation-1/${filename}`,
        sha256: createHash("sha256").update(bytes).digest("hex"),
        size: bytes.length,
      },
    ]),
  );
  const pointer = Buffer.from(JSON.stringify({
    schema_version: "odds-screen-generation/v1",
    generation: "generation-1",
    files,
  }));
  let pointerReads = 0;
  const objectReads = [];
  const env = {
    ODDS_DATA: {
      async get(key) {
        if (key === "odds_data/meta.json") {
          pointerReads += 1;
          if (pointerReads > 1) {
            throw new Error("snapshot re-read the mutable publication pointer");
          }
          return object(pointer);
        }
        objectReads.push(key);
        const filename = Object.keys(files).find(
          (name) => key === `odds_data/${files[name].key}`,
        );
        return filename ? object(payloads[filename]) : null;
      },
    },
  };

  const response = await worker.fetch(
    new Request("https://example.test/odds_data/snapshot.json"),
    env,
  );
  const snapshot = await response.json();

  assert.equal(response.status, 200);
  assert.equal(response.headers.get("x-odds-generation"), "generation-1");
  assert.equal(pointerReads, 1);
  assert.deepEqual(
    objectReads.sort(),
    Object.values(files).map(({ key }) => `odds_data/${key}`).sort(),
  );
  assert.equal(snapshot.generation, "generation-1");
  assert.deepEqual(snapshot.payloads["meta.json"], { event_id: "123", round: 4 });
  assert.deepEqual(snapshot.payloads["round_matchups.json"], {
    matchups: [{ player: "a" }],
  });
});

test("fixed market endpoint rejects a stale generation pin before reading payload", async () => {
  const env = environment();
  const originalGet = env.ODDS_DATA.get.bind(env.ODDS_DATA);
  const reads = [];
  env.ODDS_DATA.get = async (key) => {
    reads.push(key);
    return originalGet(key);
  };

  const response = await worker.fetch(
    new Request(
      "https://example.test/odds_data/round_matchups.json?generation=generation-0",
    ),
    env,
  );
  const payload = await response.json();

  assert.equal(response.status, 409);
  assert.equal(payload.requested_generation, "generation-0");
  assert.equal(payload.active_generation, "generation-1");
  assert.deepEqual(reads, ["odds_data/meta.json"]);

  const unpinned = await worker.fetch(
    new Request("https://example.test/odds_data/round_matchups.json"),
    environment(),
  );
  assert.equal(unpinned.status, 200);
  assert.equal(unpinned.headers.get("x-odds-generation"), "generation-1");
});
