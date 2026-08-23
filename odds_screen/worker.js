const DATA_PREFIX = "odds_data";
function jsonResponse(payload, status = 200) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
    },
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

export async function resolveMarketGeneration(env, filename) {
  const pointerResult = await readObject(env, "meta.json");
  if (!pointerResult) {
    throw new Error("odds-screen publication pointer is missing");
  }
  const pointer = JSON.parse(new TextDecoder().decode(pointerResult.bytes));
  if (pointer.schema_version !== "odds-screen-generation/v1") {
    throw new Error("odds-screen publication pointer schema is unsupported");
  }
  const binding = pointer.files?.[filename];
  const expectedKey = `generations/${pointer.generation}/${filename}`;
  if (!binding || String(binding.key || "") !== expectedKey) {
    throw new Error(`odds-screen pointer has no safe binding for ${filename}`);
  }
  const result = await readObject(env, binding.key);
  if (!result) {
    throw new Error(`activated odds-screen object is missing: ${binding.key}`);
  }
  if (result.bytes.byteLength !== Number(binding.size)) {
    throw new Error(`activated odds-screen object size mismatch: ${filename}`);
  }
  if ((await sha256Hex(result.bytes)) !== binding.sha256) {
    throw new Error(`activated odds-screen object hash mismatch: ${filename}`);
  }
  return { bytes: result.bytes, pointer, object: result.object };
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
    try {
      if (filename === "meta.json") {
        const result = await readObject(env, "meta.json");
        if (!result) return jsonResponse({ error: "not published" }, 404);
        return serveR2Object({ ...result, pointer: JSON.parse(new TextDecoder().decode(result.bytes)) }, "no-store");
      }
      // Market membership is declared by the pointer, not duplicated in this
      // Worker. A future payload can therefore become readable in the same
      // generation that activates it, without a second code deployment.
      if (filename?.endsWith(".json")) {
        return serveR2Object(await resolveMarketGeneration(env, filename));
      }
      const key = filename && filename !== "" ? filename : "index.html";
      const object = await env.ODDS_DATA.get(`${DATA_PREFIX}/${key}`);
      if (!object) return new Response("Not found", { status: 404 });
      const headers = new Headers();
      object.writeHttpMetadata(headers);
      return new Response(object.body, { headers });
    } catch (error) {
      // Never fall back to a fixed market key: that would recreate the mixed-
      // generation bug this resolver exists to prevent.
      return jsonResponse({ error: String(error?.message || error) }, 503);
    }
  },
};
