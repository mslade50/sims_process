/**
 * Shared Kalshi client for the execution platform.
 *
 * Used by BOTH the Pages Functions (app/functions/api/*) and the cron Worker
 * (cron/src/index.ts, which imports this file via ../../app/shared/kalshi).
 *
 * Auth mirrors kalshi_client.py / kalshi_auth.py exactly:
 *   sign message = timestampMs + METHOD + signPath
 *   signPath INCLUDES "/trade-api/v2", EXCLUDES the query string
 *   RSA-PSS, SHA-256, saltLength = 32
 */

export interface Env {
  KALSHI_API_KEY: string; // access key ID (value of KALSHI_ACCESS_KEY in .env)
  KALSHI_PRIVATE_KEY: string; // full PKCS#8 PEM text
  DB: D1Database;
}

export const API_BASE = "https://api.elections.kalshi.com/trade-api/v2";

// ── Golf ticker classification (mirrors maker_dashboard server.py) ──────────
const MARKET_TYPE_PREFIXES: [string, string][] = [
  ["KXPGATOUR", "winner"],
  ["KXPGATOP5", "top_5"],
  ["KXPGATOP10", "top_10"],
  ["KXPGATOP20", "top_20"],
  ["KXPGAH2H", "h2h"],
];

/** market_type -> Kalshi series ticker (outrights only). */
export const SERIES_BY_TYPE: Record<string, string> = {
  winner: "KXPGATOUR",
  top_5: "KXPGATOP5",
  top_10: "KXPGATOP10",
  top_20: "KXPGATOP20",
};

/** Outright market types that belong on the ticker tape (H2H excluded). */
export const OUTRIGHT_TYPES = new Set(["winner", "top_5", "top_10", "top_20"]);

export function classifyTicker(ticker: string): {
  marketType: string;
  eventCode: string;
  playerCode: string;
} {
  for (const [prefix, mt] of MARKET_TYPE_PREFIXES) {
    if (ticker.startsWith(prefix + "-")) {
      const seg = ticker.slice(prefix.length + 1).split("-");
      return { marketType: mt, eventCode: seg[0] ?? "", playerCode: seg.slice(1).join("-") };
    }
  }
  return { marketType: "", eventCode: "", playerCode: "" };
}

/** Any golf series ticker (winner/top_N/h2h). Order placement is gated on this. */
export function isGolfTicker(ticker: string): boolean {
  return classifyTicker(ticker).marketType !== "";
}

// ── RSA-PSS signing (Web Crypto) ────────────────────────────────────────────
let _cachedKey: CryptoKey | null = null;

function pemToDer(pem: string): ArrayBuffer {
  const body = pem
    .replace(/-----BEGIN [^-]+-----/, "")
    .replace(/-----END [^-]+-----/, "")
    .replace(/\s+/g, "");
  const bin = atob(body);
  const buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return buf.buffer;
}

async function importKey(pem: string): Promise<CryptoKey> {
  if (_cachedKey) return _cachedKey;
  if (!pem || !pem.includes("BEGIN PRIVATE KEY")) {
    throw new Error("KALSHI_PRIVATE_KEY must be PKCS#8 (-----BEGIN PRIVATE KEY-----).");
  }
  _cachedKey = await crypto.subtle.importKey(
    "pkcs8",
    pemToDer(pem),
    { name: "RSA-PSS", hash: "SHA-256" },
    false,
    ["sign"]
  );
  return _cachedKey;
}

function b64(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf);
  let s = "";
  for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
  return btoa(s);
}

async function signHeaders(env: Env, method: string, signPath: string): Promise<Record<string, string>> {
  const ts = Date.now().toString();
  const msg = new TextEncoder().encode(ts + method + signPath);
  const key = await importKey(env.KALSHI_PRIVATE_KEY);
  const sig = await crypto.subtle.sign({ name: "RSA-PSS", saltLength: 32 }, key, msg);
  return {
    "KALSHI-ACCESS-KEY": env.KALSHI_API_KEY,
    "KALSHI-ACCESS-TIMESTAMP": ts,
    "KALSHI-ACCESS-SIGNATURE": b64(sig),
    Accept: "application/json",
  };
}

export interface KalshiResult<T = any> {
  ok: boolean;
  status: number;
  data: T;
}

/**
 * Signed Kalshi request. `path` is WITHOUT the /trade-api/v2 prefix and WITHOUT
 * query (e.g. "/portfolio/positions"). Query params go in opts.query — they are
 * appended to the URL but excluded from the signature (matches Kalshi spec).
 */
export async function kalshi<T = any>(
  env: Env,
  method: "GET" | "POST" | "DELETE",
  path: string,
  opts: { query?: Record<string, string | number | undefined>; body?: unknown } = {}
): Promise<KalshiResult<T>> {
  const signPath = "/trade-api/v2" + path;
  let url = API_BASE + path;
  if (opts.query) {
    const qs = new URLSearchParams();
    for (const [k, v] of Object.entries(opts.query)) {
      if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
    }
    const s = qs.toString();
    if (s) url += "?" + s;
  }
  const headers = await signHeaders(env, method, signPath);
  const init: RequestInit = { method, headers };
  if (opts.body !== undefined) {
    headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(opts.body);
  }
  const resp = await fetch(url, init);
  const text = await resp.text();
  let data: any = {};
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = text;
  }
  return { ok: resp.ok, status: resp.status, data };
}

export function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}
