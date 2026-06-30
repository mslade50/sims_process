/**
 * Fail-closed password gate for the ENTIRE Pages project (static + /api).
 *
 * No Cloudflare Access dependency. A single shared password (APP_PASSWORD)
 * unlocks a signed, HttpOnly/Secure/SameSite=Strict session cookie (HMAC-SHA256
 * over an expiry, keyed by SESSION_SECRET). Every request without a valid cookie
 * is bounced to /__login — so /api/send is never reachable unauthenticated.
 *
 * Fails closed: if APP_PASSWORD or SESSION_SECRET is unset -> 503.
 *
 * Secrets (set with `wrangler pages secret put`):
 *   APP_PASSWORD    — the login password
 *   SESSION_SECRET  — random key for signing session cookies
 */
interface MEnv {
  APP_PASSWORD?: string;
  SESSION_SECRET?: string;
}

const COOKIE = "kx_session";
const MAX_AGE = 30 * 24 * 3600; // 30 days

function b64url(buf: ArrayBuffer): string {
  const b = new Uint8Array(buf);
  let s = "";
  for (let i = 0; i < b.length; i++) s += String.fromCharCode(b[i]);
  return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function timingSafeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

async function hmac(secret: string, msg: string): Promise<string> {
  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  return b64url(await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(msg)));
}

async function mintToken(secret: string): Promise<string> {
  const exp = Math.floor(Date.now() / 1000) + MAX_AGE;
  return `${exp}.${await hmac(secret, String(exp))}`;
}

async function validToken(token: string, secret: string): Promise<boolean> {
  const dot = token.indexOf(".");
  if (dot < 0) return false;
  const exp = token.slice(0, dot);
  const sig = token.slice(dot + 1);
  if (!/^\d+$/.test(exp) || Number(exp) < Date.now() / 1000) return false;
  return timingSafeEqual(sig, await hmac(secret, exp));
}

function getCookie(req: Request, name: string): string | null {
  const raw = req.headers.get("Cookie") || "";
  for (const part of raw.split(";")) {
    const [k, ...v] = part.trim().split("=");
    if (k === name) return v.join("=");
  }
  return null;
}

function loginPage(error = false): Response {
  const html = `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Kalshi Exec — login</title>
<style>body{background:#0e1116;color:#e6edf3;font:15px -apple-system,Segoe UI,Roboto,sans-serif;display:grid;place-items:center;height:100vh;margin:0}
form{background:#161b22;border:1px solid #2a323d;border-radius:12px;padding:28px;width:300px}
h1{font-size:17px;margin:0 0 16px}input{width:100%;box-sizing:border-box;background:#1c232d;border:1px solid #2a323d;color:#e6edf3;border-radius:7px;padding:10px;font-size:15px}
button{width:100%;margin-top:12px;background:#388bfd;border:0;color:#fff;border-radius:7px;padding:11px;font-weight:600;font-size:15px;cursor:pointer}
.err{color:#da3633;font-size:13px;margin-top:10px}</style></head>
<body><form method="POST" action="/__auth"><h1>⛳ Kalshi Exec</h1>
<input type="password" name="password" placeholder="password" autofocus autocomplete="current-password"/>
<button type="submit">Unlock</button>${error ? '<div class="err">Wrong password</div>' : ""}</form></body></html>`;
  return new Response(html, { status: error ? 401 : 200, headers: { "Content-Type": "text/html; charset=utf-8" } });
}

export const onRequest: PagesFunction<MEnv> = async (ctx) => {
  const { APP_PASSWORD, SESSION_SECRET } = ctx.env;
  if (!APP_PASSWORD || !SESSION_SECRET) {
    return new Response("Locked: gate not configured (APP_PASSWORD + SESSION_SECRET).", { status: 503 });
  }
  const url = new URL(ctx.request.url);

  // Login form
  if (url.pathname === "/__login") return loginPage(url.searchParams.get("e") === "1");

  // Login submit
  if (url.pathname === "/__auth" && ctx.request.method === "POST") {
    const form = await ctx.request.formData().catch(() => null);
    const pw = form ? String(form.get("password") || "") : "";
    if (timingSafeEqual(pw, APP_PASSWORD)) {
      const token = await mintToken(SESSION_SECRET);
      return new Response(null, {
        status: 303,
        headers: {
          Location: "/",
          "Set-Cookie": `${COOKIE}=${token}; HttpOnly; Secure; SameSite=Strict; Path=/; Max-Age=${MAX_AGE}`,
        },
      });
    }
    return new Response(null, { status: 303, headers: { Location: "/__login?e=1" } });
  }

  // Logout
  if (url.pathname === "/__logout") {
    return new Response(null, {
      status: 303,
      headers: { Location: "/__login", "Set-Cookie": `${COOKIE}=; HttpOnly; Secure; SameSite=Strict; Path=/; Max-Age=0` },
    });
  }

  // Headless pushes from the local sim/maker: let these POSTs reach their
  // handlers, which enforce their own token header. (No session cookie.)
  if (ctx.request.method === "POST" &&
      (url.pathname === "/api/proposals" || url.pathname === "/api/maker")) {
    return ctx.next();
  }

  // Gate everything else
  const tok = getCookie(ctx.request, COOKIE);
  if (tok && (await validToken(tok, SESSION_SECRET))) return ctx.next();

  // Unauthenticated: 401 JSON for API, redirect for navigations
  if (url.pathname.startsWith("/api/")) {
    return new Response(JSON.stringify({ error: "unauthorized" }), {
      status: 401,
      headers: { "Content-Type": "application/json" },
    });
  }
  return new Response(null, { status: 303, headers: { Location: "/__login" } });
};
