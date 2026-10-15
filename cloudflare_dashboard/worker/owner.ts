/**
 * Owner-only areas (owner, October 15, 2026): colleagues may be let in through Cloudflare Access, but the Run page, weather edits,
 * Model inputs overrides and their APIs stay with the owner.
 *
 *   GET /api/whoami   {ok, email, owner}  so the app can hide owner-only links (never trusted for access; this gate is)
 *
 * OWNER_EMAILS (comma-separated, wrangler.jsonc vars) names the owners. An owner needs a signature-verified Access identity whose email is
 * on the list. Owner-only pages (app/shell-rules.ts OWNER_ONLY_VIEWS) redirect everyone else home; owner-only APIs answer 403.
 * While OWNER_EMAILS is unset the gate is off and every Access user is treated as before (unit tests, local preview); tests/wrangler-config
 * requires it in the deployed config.
 */
import { accessIdentity, type AccessEnv } from "./access";
import { ownerOnlyPath, type Viewer } from "../app/shell-rules";

export interface OwnerEnv extends AccessEnv {
  OWNER_EMAILS?: string;
}

const NO_STORE = { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" };

export function ownerEmails(env: OwnerEnv): string[] | null {
  const list = (env.OWNER_EMAILS ?? "").split(",").map((email) => email.trim().toLowerCase()).filter(Boolean);
  return list.length ? list : null;
}

export async function viewerFor(request: Request, env: OwnerEnv, now = Date.now()): Promise<Viewer> {
  const identity = await accessIdentity(request, env, now);
  const owners = ownerEmails(env);
  if (!owners) return { email: identity?.email ?? null, owner: true };
  return { email: identity?.email ?? null, owner: !!identity && identity.verified && owners.includes(identity.email) };
}

/** Answers /api/whoami and refuses owner-only paths for non-owners; null means carry on routing. */
export async function handleOwnerGate(request: Request, env: OwnerEnv, now = Date.now()): Promise<Response | null> {
  const url = new URL(request.url);
  if (url.pathname === "/api/whoami") {
    if (request.method !== "GET") return new Response(JSON.stringify({ ok: false, error: "method not allowed" }), { status: 405, headers: NO_STORE });
    return new Response(JSON.stringify({ ok: true, ...(await viewerFor(request, env, now)) }), { headers: NO_STORE });
  }
  const kind = ownerOnlyPath(url.pathname);
  if (!kind || !ownerEmails(env)) return null;
  if ((await viewerFor(request, env, now)).owner) return null;
  if (kind === "api") return new Response(JSON.stringify({ ok: false, error: "This part of the dashboard is for the owner only" }), { status: 403, headers: NO_STORE });
  return new Response(null, { status: 302, headers: { location: new URL("/this-week", url).href, "cache-control": "no-store" } });
}
