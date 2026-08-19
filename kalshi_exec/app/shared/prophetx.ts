/**
 * ProphetX partner ("Trading") API client for the execution platform.
 *
 * Mirrors the validated auth/discovery flow in scrapers/prophetx.py (the board
 * worktree): POST {base}/auth/login with access_key/secret_key -> Bearer
 * access_token; market data under /mm/*. Order endpoints per docs.prophetx.co:
 *   POST /mm/place_wager   { external_id, line_id, odds, stake }  (idempotent
 *                            on external_id; ~50 req/s cap)
 *   POST /mm/cancel_wager  { wager_id | external_id }
 *   GET  /v2/mm/get_wager_histories, GET /mm/get_balance,
 *   GET  /mm/get_price_ladder (only ladder odds are accepted)
 *
 * FAIL-CLOSED: without PROPHETX_ACCESS_KEY + PROPHETX_SECRET_KEY every helper
 * reports unconfigured and the /api/px/* endpoints 503. Set secrets with
 * `wrangler pages secret put`. Base URL defaults to production
 * (cash.api.prophetx.co/partner — the scraper's validated base); point
 * PROPHETX_API_BASE_URL at the sandbox for dress rehearsals.
 *
 * NOTE: written without live credentials (none exist on the dev machine) —
 * response-shape tolerances are deliberate. First run against the sandbox
 * before trusting production behavior.
 */

export interface PxEnv {
  PROPHETX_ACCESS_KEY?: string;
  PROPHETX_SECRET_KEY?: string;
  PROPHETX_API_BASE_URL?: string;
  PX_SEND_MAX_STAKE_DOLLARS?: string; // per-wager stake cap (default 500)
}

export const PX_DEFAULT_BASE = "https://cash.api.prophetx.co/partner";

export function pxConfigured(env: PxEnv): boolean {
  return Boolean(env.PROPHETX_ACCESS_KEY && env.PROPHETX_SECRET_KEY);
}

function pxBase(env: PxEnv): string {
  return (env.PROPHETX_API_BASE_URL || PX_DEFAULT_BASE).replace(/\/+$/, "");
}

export function pxJson(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

// ── Token cache (per-isolate; partner tokens are short-lived) ───────────────
let _token: { value: string; expires: number } | null = null;

async function pxToken(env: PxEnv): Promise<string> {
  const now = Date.now();
  if (_token && _token.expires > now) return _token.value;
  const r = await fetch(`${pxBase(env)}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({
      access_key: env.PROPHETX_ACCESS_KEY,
      secret_key: env.PROPHETX_SECRET_KEY,
    }),
  });
  if (!r.ok) throw new Error(`ProphetX login failed: HTTP ${r.status}`);
  const data: any = await r.json().catch(() => ({}));
  const token = String(data?.data?.access_token || data?.access_token || "");
  if (!token) throw new Error("ProphetX login returned no access token");
  _token = { value: token, expires: now + 4 * 60 * 1000 }; // refresh every 4 min
  return token;
}

export interface PxResult {
  ok: boolean;
  status: number;
  data: any;
}

export async function px(
  env: PxEnv,
  method: "GET" | "POST",
  path: string,
  opts: { body?: unknown; params?: Record<string, string> } = {}
): Promise<PxResult> {
  const token = await pxToken(env);
  const url = new URL(`${pxBase(env)}/${path.replace(/^\/+/, "")}`);
  for (const [k, v] of Object.entries(opts.params || {})) url.searchParams.set(k, v);
  const r = await fetch(url.toString(), {
    method,
    headers: {
      Authorization: `Bearer ${token}`,
      Accept: "application/json",
      ...(opts.body !== undefined ? { "Content-Type": "application/json" } : {}),
    },
    body: opts.body !== undefined ? JSON.stringify(opts.body) : undefined,
  });
  if (r.status === 401) _token = null; // stale token — next call re-logs in
  const data = await r.json().catch(() => ({}));
  return { ok: r.ok, status: r.status, data };
}

// ── Golf scoping (mirrors the scraper's classifier: sport/tournament name) ──
function isGolfTournament(t: any): boolean {
  const sport = t?.sport;
  const sportName = typeof sport === "object" && sport ? sport.name : sport;
  return `${sportName ?? ""} ${t?.name ?? ""}`.toLowerCase().includes("golf");
}

function listFrom(payload: any, ...keys: string[]): any[] {
  const data = payload?.data ?? payload;
  for (const k of keys) {
    const v = data?.[k];
    if (Array.isArray(v)) return v;
  }
  return Array.isArray(data) ? data : [];
}

export interface GolfSnapshot {
  fetchedAt: number;
  tournaments: { id: string; name: string }[];
  events: { id: string; name: string; tournament: string; status: string }[];
  eventIds: Set<string>;
}

let _snapshot: GolfSnapshot | null = null;
const SNAPSHOT_TTL_MS = 5 * 60 * 1000;

/** Active golf tournaments + events (cached ~5 min per isolate). */
export async function golfSnapshot(env: PxEnv, force = false): Promise<GolfSnapshot> {
  if (!force && _snapshot && Date.now() - _snapshot.fetchedAt < SNAPSHOT_TTL_MS) return _snapshot;
  const tp = await px(env, "GET", "/mm/get_tournaments", { params: { has_active_events: "true" } });
  if (!tp.ok) throw new Error(`get_tournaments failed: HTTP ${tp.status}`);
  const tournaments = listFrom(tp.data, "tournaments")
    .filter(isGolfTournament)
    .map((t: any) => ({ id: String(t.id ?? t.tournament_id ?? ""), name: String(t.name ?? "") }))
    .filter((t) => t.id);

  const events: GolfSnapshot["events"] = [];
  for (const t of tournaments) {
    const ep = await px(env, "GET", "/mm/get_sport_events", { params: { tournament_id: t.id } });
    if (!ep.ok) continue;
    for (const e of listFrom(ep.data, "sport_events", "events")) {
      const id = String(e?.id ?? e?.event_id ?? "");
      const status = String(e?.status ?? "").toLowerCase();
      if (!id || ["finished", "closed", "cancelled", "canceled"].includes(status)) continue;
      events.push({ id, name: String(e?.name ?? ""), tournament: t.name, status });
    }
  }
  _snapshot = {
    fetchedAt: Date.now(),
    tournaments,
    events,
    eventIds: new Set(events.map((e) => e.id)),
  };
  return _snapshot;
}

/** Markets for one event (raw payload rows — the UI flattens). */
export async function eventMarkets(env: PxEnv, eventId: string): Promise<any[]> {
  const mp = await px(env, "GET", "/mm/get_multiple_markets", { params: { event_ids: eventId } });
  if (!mp.ok) throw new Error(`get_multiple_markets failed: HTTP ${mp.status}`);
  const data = mp.data?.data ?? mp.data;
  const byEvent = data?.markets ?? data;
  if (Array.isArray(byEvent)) return byEvent;
  if (byEvent && typeof byEvent === "object") {
    const rows = byEvent[eventId];
    if (Array.isArray(rows)) return rows;
    // single-key object form
    const vals = Object.values(byEvent);
    if (vals.length === 1 && Array.isArray(vals[0])) return vals[0] as any[];
  }
  return [];
}

/** Collect every line_id present in a market payload row (shape-tolerant). */
export function marketLineIds(market: any): Set<string> {
  const out = new Set<string>();
  const visit = (node: any) => {
    if (!node || typeof node !== "object") return;
    if (Array.isArray(node)) {
      for (const item of node) visit(item);
      return;
    }
    if (node.line_id != null) out.add(String(node.line_id));
    if (node.id != null && (node.odds != null || node.line != null) && node.selections === undefined) {
      // selection-like rows sometimes carry the line id as `id`
      out.add(String(node.id));
    }
    for (const v of Object.values(node)) visit(v);
  };
  visit(market);
  return out;
}
