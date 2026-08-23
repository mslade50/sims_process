const DATA_PREFIX = "odds_data";
const GENERATION_SCHEMA = "odds-screen-generation/v1";

class GenerationMismatchError extends Error {
  constructor(expected, actual) {
    super(
      `requested odds-screen generation ${expected} is not active (active: ${actual})`,
    );
    this.name = "GenerationMismatchError";
    this.expected = expected;
    this.actual = actual;
  }
}

function jsonResponse(payload, status = 200, generation = null) {
  const headers = new Headers({
    "content-type": "application/json; charset=utf-8",
    "cache-control": "no-store",
  });
  if (generation) headers.set("x-odds-generation", generation);
  return new Response(JSON.stringify(payload), {
    status,
    headers,
  });
}

export async function sha256Hex(bytes) {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}

async function readObject(env, key) {
  const object = await env.ODDS_DATA.get(`${DATA_PREFIX}/${key}`);
  if (!object) return null;
  return { object, bytes: await object.arrayBuffer() };
}

function parsePointer(pointerResult) {
  let pointer;
  try {
    pointer = JSON.parse(new TextDecoder().decode(pointerResult.bytes));
  } catch {
    throw new Error("odds-screen publication pointer is invalid JSON");
  }
  if (pointer?.schema_version !== GENERATION_SCHEMA) {
    throw new Error("odds-screen publication pointer schema is unsupported");
  }
  if (!/^[A-Za-z0-9_-]+$/.test(String(pointer.generation || ""))) {
    throw new Error("odds-screen publication pointer generation is unsafe");
  }
  if (
    !pointer.files ||
    typeof pointer.files !== "object" ||
    Array.isArray(pointer.files) ||
    Object.keys(pointer.files).length === 0
  ) {
    throw new Error("odds-screen publication pointer declares no files");
  }
  return pointer;
}

async function readActivePointer(env) {
  const pointerResult = await readObject(env, "meta.json");
  if (!pointerResult) {
    throw new Error("odds-screen publication pointer is missing");
  }
  return { pointer: parsePointer(pointerResult), pointerResult };
}

function requireGeneration(pointer, expectedGeneration) {
  if (
    expectedGeneration !== null &&
    String(pointer.generation) !== String(expectedGeneration)
  ) {
    throw new GenerationMismatchError(expectedGeneration, pointer.generation);
  }
}

function requireBinding(pointer, filename) {
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]*\.json$/.test(String(filename || ""))) {
    throw new Error(`odds-screen payload name is unsafe: ${filename}`);
  }
  const binding = pointer.files?.[filename];
  const expectedKey = `generations/${pointer.generation}/${filename}`;
  if (
    !binding ||
    typeof binding !== "object" ||
    String(binding.key || "") !== expectedKey ||
    !Number.isSafeInteger(binding.size) ||
    binding.size < 0 ||
    !/^[0-9a-f]{64}$/.test(String(binding.sha256 || ""))
  ) {
    throw new Error(`odds-screen pointer has no safe binding for ${filename}`);
  }
  return binding;
}

async function resolveBoundPayload(env, pointer, filename) {
  const binding = requireBinding(pointer, filename);
  const result = await readObject(env, binding.key);
  if (!result) {
    throw new Error(`activated odds-screen object is missing: ${binding.key}`);
  }
  if (result.bytes.byteLength !== binding.size) {
    throw new Error(`activated odds-screen object size mismatch: ${filename}`);
  }
  if ((await sha256Hex(result.bytes)) !== binding.sha256) {
    throw new Error(`activated odds-screen object hash mismatch: ${filename}`);
  }
  return result;
}

export async function resolveMarketGeneration(
  env,
  filename,
  expectedGeneration = null,
) {
  const { pointer } = await readActivePointer(env);
  requireGeneration(pointer, expectedGeneration);
  const result = await resolveBoundPayload(env, pointer, filename);
  return { ...result, pointer };
}

export async function resolveSnapshotGeneration(env, expectedGeneration = null) {
  // This is the snapshot's single publication-pointer read. Every subsequent
  // lookup is bound to this immutable generation, even if activation advances
  // while the payload objects are being fetched.
  const { pointer } = await readActivePointer(env);
  requireGeneration(pointer, expectedGeneration);
  const entries = Object.keys(pointer.files).sort();
  const resolved = await Promise.all(
    entries.map(async (filename) => {
      const result = await resolveBoundPayload(env, pointer, filename);
      let payload;
      try {
        payload = JSON.parse(new TextDecoder().decode(result.bytes));
      } catch {
        throw new Error(`activated odds-screen object is invalid JSON: ${filename}`);
      }
      return [filename, payload];
    }),
  );
  return {
    schema_version: "odds-screen-snapshot/v1",
    generation: pointer.generation,
    pointer,
    payloads: Object.fromEntries(resolved),
  };
}

async function serveR2Object(result, cacheControl = "max-age=30, must-revalidate") {
  const headers = new Headers();
  result.object?.writeHttpMetadata?.(headers);
  headers.set("content-type", "application/json; charset=utf-8");
  headers.set("cache-control", cacheControl);
  headers.set("x-odds-generation", result.pointer?.generation || "legacy");
  return new Response(result.bytes, { headers });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const filename = url.pathname.replace(/^\/odds_data\//, "/").split("/").pop();
    const expectedGeneration = url.searchParams.has("generation")
      ? url.searchParams.get("generation")
      : null;
    try {
      if (filename === "snapshot.json") {
        const snapshot = await resolveSnapshotGeneration(env, expectedGeneration);
        return jsonResponse(snapshot, 200, snapshot.generation);
      }
      if (filename === "meta.json") {
        const result = await readObject(env, "meta.json");
        if (!result) return jsonResponse({ error: "not published" }, 404);
        return serveR2Object({ ...result, pointer: JSON.parse(new TextDecoder().decode(result.bytes)) }, "no-store");
      }
      // Market membership is declared by the pointer, not duplicated in this
      // Worker. A future payload can therefore become readable in the same
      // generation that activates it, without a second code deployment.
      if (filename?.endsWith(".json")) {
        return serveR2Object(
          await resolveMarketGeneration(env, filename, expectedGeneration),
        );
      }
      const key = filename && filename !== "" ? filename : "index.html";
      const object = await env.ODDS_DATA.get(`${DATA_PREFIX}/${key}`);
      if (!object) return new Response("Not found", { status: 404 });
      const headers = new Headers();
      object.writeHttpMetadata(headers);
      return new Response(object.body, { headers });
    } catch (error) {
      if (error instanceof GenerationMismatchError) {
        return jsonResponse(
          {
            error: error.message,
            requested_generation: error.expected,
            active_generation: error.actual,
          },
          409,
          error.actual,
        );
      }
      // Never fall back to a fixed market key: that would recreate the mixed-
      // generation bug this resolver exists to prevent.
      return jsonResponse({ error: String(error?.message || error) }, 503);
    }
  },
};
