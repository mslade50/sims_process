/**
 * Pure rules for the app shell (site audit wave 1, October 2026): routes, navigation groups,
 * data-fallback decision, player search and the Run page's "suggested now" group. No DOM or React, so tests import it directly.
 */
import type { ProfileEntry } from "./player-profile-rules";

/* ------------------------------------------------------------------ routes */
export const DEFAULT_VIEW = "this-week";
/** Routes kept for old bookmarks that now send the visitor elsewhere (nothing is deleted; the views stay registered). */
export const REDIRECTS: Record<string, string> = { weather: "/weather-effects", "sg-distributions": "/players" };
/** Routes whose view is replaced by a "retired" notice. */
export const RETIRED: readonly string[] = ["research"];
/** Archived views fed by the legacy pipeline; the header labels them instead of showing golfprice freshness. */
export const LEGACY_VIEWS: readonly string[] = ["history", "diagnostics"];

export type RouteDecision =
  | { kind: "view"; key: string }
  | { kind: "redirect"; key: string; to: string }
  | { kind: "retired"; key: string }
  | { kind: "notfound"; requested: string };

/** `requested` is the URL segment ("" for the root); `registered` is the list of view keys the app can render. */
export function resolveRoute(requested: string | undefined | null, registered: readonly string[]): RouteDecision {
  const key = (requested ?? "").trim().replace(/^\/+|\/+$/g, "").toLowerCase();
  if (!key) return { kind: "view", key: DEFAULT_VIEW };
  if (REDIRECTS[key]) return { kind: "redirect", key, to: REDIRECTS[key] };
  if (RETIRED.includes(key)) return { kind: "retired", key };
  if (registered.includes(key)) return { kind: "view", key };
  return { kind: "notfound", requested: key.slice(0, 80) };
}

/* ------------------------------------------------------------------ navigation (data only; icons are mapped in DashboardApp) */
export type NavItem = {
  key: string;
  href: string;
  label: string;
  description: string;
  /** One nav entry that covers several routes: a small toggle under the entry switches between them. */
  toggle?: Array<{ key: string; href: string; label: string }>;
};
export type NavGroup = { label: string; collapsed?: boolean; items: NavItem[] };

export const NAVIGATION: NavGroup[] = [
  {
    label: "Week",
    items: [
      { key: "this-week", href: "/this-week", label: "This week", description: "Course, model vs market, who we favor and why" },
      { key: "weather-effects", href: "/weather-effects", label: "Weather", description: "How the weather changes scoring and who it helps or hurts" },
      {
        key: "weekly-players", href: "/weekly-players", label: "Players", description: "Everyone in the field, or one golfer's profile",
        toggle: [{ key: "weekly-players", href: "/weekly-players", label: "Whole field" }, { key: "players", href: "/players", label: "One golfer" }],
      },
      { key: "why-priced", href: "/why-priced", label: "Why priced", description: "Every player's price, shape and drivers" },
      { key: "distributions", href: "/distributions", label: "Distributions", description: "Finish-position probability curves" },
      { key: "round-scores", href: "/round-scores", label: "Scoring", description: "Expected score and how wide it could swing" },
    ],
  },
  {
    label: "Model",
    items: [
      { key: "inputs", href: "/inputs", label: "Model inputs", description: "Skill, course fit, variance, adjustments" },
      { key: "weather-edits", href: "/weather-edits", label: "Weather edits", description: "Replace forecast hours with your own numbers, then reprice" },
    ],
  },
  { label: "Operate", items: [{ key: "run", href: "/run", label: "Run", description: "Start a pricing job, even from your phone" }] },
  { label: "Review", items: [{ key: "performance", href: "/performance", label: "Scorecard and P&L", description: "Model scorecard and bet results" }] },
  {
    label: "Archive",
    collapsed: true,
    items: [
      { key: "history", href: "/history", label: "History", description: "Past tournaments (archived pipeline)" },
      { key: "diagnostics", href: "/diagnostics", label: "Diagnostics", description: "Skill accuracy checks (archived pipeline)" },
    ],
  },
];

export const NAV_ITEMS: NavItem[] = NAVIGATION.flatMap((group) => group.items);

/** The nav entry (and toggle target) that owns a view key, for the active highlight and the header title. */
export function navEntryFor(viewKey: string): { item: NavItem; label: string; description: string } | null {
  for (const item of NAV_ITEMS) {
    if (item.key === viewKey) return { item, label: item.label, description: item.description };
    const sub = item.toggle?.find((t) => t.key === viewKey);
    if (sub) return { item, label: `${item.label}: ${sub.label}`, description: item.description };
  }
  return null;
}

/* ------------------------------------------------------------------ owner-only areas (owner, October 15, 2026) */
/**
 * Colleagues may be let in through Cloudflare Access; these areas stay with the owner (OWNER_EMAILS on the Worker, worker/owner.ts).
 * The Worker enforces it (pages redirect home, APIs answer 403); the app only hides the links.
 */
export const OWNER_ONLY_VIEWS: readonly string[] = ["run", "weather-edits"];
export const OWNER_ONLY_API_PREFIXES: readonly string[] = ["/api/jobs", "/api/overrides", "/api/weather-overrides"];

/** "page" or "api" when `pathname` is owner-only, else null. Case, a trailing slash and an `.rsc` suffix do not get around it. */
export function ownerOnlyPath(pathname: string): "page" | "api" | null {
  const path = pathname.toLowerCase().replace(/\.rsc$/, "").replace(/\/+$/, "");
  if (OWNER_ONLY_API_PREFIXES.some((prefix) => path === prefix || path.startsWith(`${prefix}/`))) return "api";
  if (OWNER_ONLY_VIEWS.some((view) => path === `/${view}`)) return "page";
  return null;
}

/** The navigation a visitor sees: owner-only items are dropped for everyone else, and a group left empty goes with them. */
export function navigationFor(owner: boolean): NavGroup[] {
  if (owner) return NAVIGATION;
  return NAVIGATION.map((group) => ({ ...group, items: group.items.filter((item) => !OWNER_ONLY_VIEWS.includes(item.key)) })).filter((group) => group.items.length > 0);
}

export type Viewer = { email: string | null; owner: boolean };

/** GET /api/whoami answer to a Viewer. Anything but an explicit `owner: true` is a non-owner, so a failed check never shows owner links. */
export function viewerFrom(json: unknown): Viewer {
  const body = json && typeof json === "object" ? (json as { email?: unknown; owner?: unknown }) : {};
  return { email: typeof body.email === "string" ? body.email : null, owner: body.owner === true };
}

/* ------------------------------------------------------------------ data fallback (D2) */
export type FetchOutcome =
  | { kind: "network" }
  | { kind: "response"; ok: boolean; status: number; json: unknown };   // json === undefined when the body is not JSON

export type FallbackDecision = { action: "use" } | { action: "fallback" } | { action: "surface"; message: string };

/** The Worker answers every error as JSON with a string `error` (worker/overrides-api.ts `fail`, worker/index.ts `/api/data`). */
export function isWorkerError(json: unknown): json is { error: string } {
  return !!json && typeof json === "object" && typeof (json as { error?: unknown }).error === "string";
}

/**
 * Decide what to do with the API response. The packaged static copy is only for the no-Worker local preview, where the API route does not
 * exist: a network failure or a non-JSON body (HTML 404 page, empty body) means "no Worker here". A structured Worker error
 * (404 "not published yet", 503 "bucket is not bound") is real and is surfaced, never masked by a stale packaged copy.
 */
export function decideFallback(outcome: FetchOutcome): FallbackDecision {
  if (outcome.kind === "network") return { action: "fallback" };
  if (outcome.json === undefined) return { action: "fallback" };
  if (isWorkerError(outcome.json) || !outcome.ok) {
    const detail = isWorkerError(outcome.json) ? `: ${outcome.json.error}` : "";
    return { action: "surface", message: `This data could not be loaded (code ${outcome.status})${detail}` };
  }
  return { action: "use" };
}

/* ------------------------------------------------------------------ player search (B6) */
/** Derived by the Worker (worker/player-search.ts): identity-only column table, ~0.3 MB gzipped, instead of the 16 MB catalog. Append `?gen=<generation>` to pin a generation. */
export const PLAYER_SEARCH_KEY = "golfprice/player_profiles/search-index.json";
/** Derived by the Worker: pga_benchmark for only the golfers in `?ids=` (max 400). */
export const PLAYER_BENCHMARKS_KEY = "golfprice/player_profiles/benchmarks.json";
export const PLAYER_CATALOG_KEY = "golfprice/player_profiles/index.json";   // runner-owned, regression-gated pointer (site audit D3 end state); dossier-review.json stays frozen as last-known-good
const profileId = (p: ProfileEntry): string => p.profile_id ?? String(p.dg_id);   // same rule as player-profile-rules.playerId (kept local so this file has no value imports)
const fold = (s: string) => s.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
const haystacks = new WeakMap<ProfileEntry, string>();   // folded once per entry, not once per keystroke per golfer
const matches = (p: ProfileEntry, query: string) => {
  let haystack = haystacks.get(p);
  if (haystack === undefined) { haystack = fold([p.name, ...(p.aliases ?? []), p.country, ...(p.tours ?? []), String(p.dg_id)].join(" ")); haystacks.set(p, haystack); }
  const h = haystack;
  return fold(query).split(/\s+/).every((word) => h.includes(word));   // same rule as player-profile-rules.profileMatches
};
export const playerHref = (p: ProfileEntry): string => `/players?player=${encodeURIComponent(profileId(p))}`;

/** Up to `limit` golfers matching `query`; names that start with the query first, then complete coverage, then name. */
export function searchPlayers(players: ProfileEntry[], query: string, limit = 8): ProfileEntry[] {
  const q = query.trim();
  if (q.length < 2) return [];
  const fq = fold(q);
  const rank = (p: ProfileEntry) => {
    const name = fold(p.name);
    const last = name.split(/\s+/).at(-1) ?? name;
    return (name.startsWith(fq) ? 0 : last.startsWith(fq) ? 1 : 2) * 2 + (p.status === "complete" ? 0 : 1);
  };
  return players.filter((p) => matches(p, q)).sort((a, b) => rank(a) - rank(b) || a.name.localeCompare(b.name)).slice(0, limit);
}

/* ------------------------------------------------------------------ Run page (C4) */
export type RunGroupName = "Monday" | "Tuesday" | "Wednesday" | "Thursday" | "During event" | "Anytime";
/** The weekday group that matches today in America/New_York (the schedule's clock); Friday to Sunday are "During event". */
export function suggestedGroup(now: Date | number): RunGroupName {
  const weekday = new Intl.DateTimeFormat("en-US", { weekday: "long", timeZone: "America/New_York" }).format(now);
  return weekday === "Monday" || weekday === "Tuesday" || weekday === "Wednesday" || weekday === "Thursday" ? weekday : "During event";
}

/* ------------------------------------------------------------------ deep links (B5) */
/** True when a `?e=` value names no event in the published index (only judged once the index has loaded with at least one event). */
export function unknownEventParam(param: string | null | undefined, events: Array<{ event_uid?: string }> | undefined): boolean {
  if (!param || !events || events.length === 0) return false;
  return !events.some((e) => e.event_uid === param);
}
