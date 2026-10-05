/**
 * Cloudflare Access identity for write endpoints.
 *
 * Access puts the signed user assertion in `Cf-Access-Jwt-Assertion` and the email in
 * `Cf-Access-Authenticated-User-Email`. Writes are refused without both.
 * - If ACCESS_TEAM_DOMAIN and ACCESS_AUD are configured on the Worker, the RS256 signature, audience, issuer and expiry are verified
 *   against the team's published certs (strongest).
 * - Otherwise the assertion must still decode, be unexpired and carry the same email as the header. This relies on the Worker route
 *   being reachable only through the Access application (workers.dev is protected deny-by-default); `verified: false` is recorded
 *   on every history entry so the weaker mode is visible.
 */
export interface AccessEnv {
  ACCESS_TEAM_DOMAIN?: string;
  ACCESS_AUD?: string;
}

export type Identity = { email: string; verified: boolean; subject?: string };

function b64urlToBytes(value: string): Uint8Array<ArrayBuffer> {
  const pad = "=".repeat((4 - (value.length % 4)) % 4);
  const bin = atob(value.replace(/-/g, "+").replace(/_/g, "/") + pad);
  return Uint8Array.from(bin, (c) => c.charCodeAt(0));
}

function decodeJson(part: string): Record<string, unknown> | null {
  try {
    const parsed = JSON.parse(new TextDecoder().decode(b64urlToBytes(part)));
    return parsed && typeof parsed === "object" ? (parsed as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}

let certCache: { team: string; at: number; keys: Array<JsonWebKey & { kid?: string }> } | null = null;

async function signingKeys(team: string, fetcher: typeof fetch): Promise<Array<JsonWebKey & { kid?: string }>> {
  if (certCache && certCache.team === team && Date.now() - certCache.at < 3_600_000) return certCache.keys;
  const response = await fetcher(`https://${team}/cdn-cgi/access/certs`);
  if (!response.ok) throw new Error(`certs ${response.status}`);
  const body = (await response.json()) as { keys?: Array<JsonWebKey & { kid?: string }> };
  certCache = { team, at: Date.now(), keys: body.keys ?? [] };
  return certCache.keys;
}

export async function accessIdentity(request: Request, env: AccessEnv, now = Date.now(), fetcher: typeof fetch = fetch): Promise<Identity | null> {
  const headerEmail = request.headers.get("cf-access-authenticated-user-email")?.trim().toLowerCase();
  const jwt = request.headers.get("cf-access-jwt-assertion")?.trim();
  if (!headerEmail || !jwt || !/^[^@\s]+@[^@\s]+$/.test(headerEmail)) return null;
  const parts = jwt.split(".");
  if (parts.length !== 3) return null;
  const header = decodeJson(parts[0]);
  const payload = decodeJson(parts[1]);
  if (!header || !payload) return null;
  const exp = typeof payload.exp === "number" ? payload.exp * 1000 : 0;
  if (exp <= now) return null;
  if (typeof payload.email !== "string" || payload.email.toLowerCase() !== headerEmail) return null;
  const subject = typeof payload.sub === "string" ? payload.sub : undefined;

  const team = env.ACCESS_TEAM_DOMAIN?.replace(/^https?:\/\//, "").replace(/\/$/, "");
  if (!team || !env.ACCESS_AUD) return { email: headerEmail, verified: false, subject };

  try {
    if (header.alg !== "RS256") return null;
    const aud = Array.isArray(payload.aud) ? payload.aud : [payload.aud];
    if (!aud.includes(env.ACCESS_AUD) || payload.iss !== `https://${team}`) return null;
    const jwk = (await signingKeys(team, fetcher)).find((key) => key.kid === header.kid);
    if (!jwk) return null;
    const key = await crypto.subtle.importKey("jwk", jwk, { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" }, false, ["verify"]);
    const ok = await crypto.subtle.verify("RSASSA-PKCS1-v1_5", key, b64urlToBytes(parts[2]), new TextEncoder().encode(`${parts[0]}.${parts[1]}`));
    return ok ? { email: headerEmail, verified: true, subject } : null;
  } catch {
    return null;
  }
}
