import assert from "node:assert/strict";
import test from "node:test";
import { createHash } from "node:crypto";

const { resolveMarketGeneration } = await import("./worker.js");

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
