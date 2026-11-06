/**
 * Pure rules for the app shell (site audit wave 1, October 2026): routes, navigation groups,
 * data-fallback decision, player search and the Run page's "suggested now" group. No DOM or React, so tests import it directly.
 */
import type { ProfileEntry } from "./player-profile-rules";
import type { IndexEventLite } from "./ui-rules";

/* ------------------------------------------------------------------ routes */
export const DEFAULT_VIEW = "this-week";
/** Routes kept for old bookmarks that now send the visitor elsewhere (nothing is deleted; the views stay registered, except the retired static weather view, which no route reaches). */
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
      { key: "model-walkthrough", href: "/model-walkthrough", label: "How the model works", description: "Graphical skill and simulation guide with this week's verified process" },
      { key: "inputs", href: "/inputs", label: "Model inputs", description: "Skill, course fit, variance, adjustments" },
      { key: "weather-edits", href: "/weather-edits", label: "Weather edits", description: "Replace forecast hours with your own numbers, then reprice" },
    ],
  },
  { label: "Operate", items: [{ key: "shot-data", href: "/shot-data", label: "Shot data", description: "Collector health, capture and cleaning, published dataset readiness" }, { key: "ops", href: "/ops", label: "Weekly workflow", description: "The week, verified outputs, open work and agent handoffs" }, { key: "run", href: "/run", label: "Run", description: "Start a pricing job, even from your phone" }] },
  {
    label: "Review",
    items: [
      { key: "performance", href: "/performance", label: "Scorecard and P&L", description: "Model scorecard and bet results" },
      { key: "bet-breakdowns", href: "/bet-breakdowns", label: "Bet breakdowns", description: "How our edges have paid, by market, book and edge size" },
      { key: "bet-card", href: "/bet-card", label: "Bet card", description: "Current suggested bets and the lowest price worth taking" },
    ],
  },
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
export const OWNER_ONLY_VIEWS: readonly string[] = ["model-walkthrough", "shot-data", "ops", "run", "weather-edits", "bet-breakdowns", "bet-card"];
export const OWNER_ONLY_API_PREFIXES: readonly string[] = ["/api/ops", "/api/jobs", "/api/overrides", "/api/weather-overrides", "/api/owner-data"];

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
let weekdayFormat: Intl.DateTimeFormat | undefined;
/** The weekday group that matches today in America/New_York (the schedule's clock); Friday to Sunday are "During event". */
export function suggestedGroup(now: Date | number): RunGroupName {
  const weekday = (weekdayFormat ??= new Intl.DateTimeFormat("en-US", { weekday: "long", timeZone: "America/New_York" })).format(now);
  return weekday === "Monday" || weekday === "Tuesday" || weekday === "Wednesday" || weekday === "Thursday" ? weekday : "During event";
}

/* ------------------------------------------------------------------ deep links (B5) */
/** True when a `?e=` value names no event in the published index (only judged once the index has loaded with at least one event). */
export function unknownEventParam(param: string | null | undefined, events: Array<{ event_uid?: string }> | undefined): boolean {
  if (!param || !events || events.length === 0) return false;
  return !events.some((e) => e.event_uid === param);
}

/* ------------------------------------------------------------------ context bar: one event identity (design system v2) */
export type ShellEvent = {
  uid: string; name: string; course: string;
  tour: string; tourLabel: string;
  /** "Pre-event", "Live, after R2" or "Final": what the newest saved run is about. */
  stage: string; stageTone: "neutral" | "accent" | "positive";
  asOf: string | null;
  /** Event stage now, which round the prices reflect, and whether they are usable (see eventStatus). */
  status: EventStatus;
};
const TOUR_LABELS: Record<string, string> = { pga: "PGA Tour", euro: "DP World Tour", dpwt: "DP World Tour", korn: "Korn Ferry" };
export function tourLabelOf(tour: string | undefined | null): string {
  const key = (tour ?? "").toLowerCase();
  return TOUR_LABELS[key] ?? (key ? key.toUpperCase() : "Tour");
}
/** Short chip text for a tour ("PGA", "DPWT"). */
export function tourShort(tour: string | undefined | null): string {
  const key = (tour ?? "").toLowerCase();
  return key === "pga" ? "PGA" : key === "euro" || key === "dpwt" ? "DP World" : key ? key.toUpperCase() : "Tour";
}
type StageEvent = { date_end?: string; runs?: Array<{ as_of?: string; kind?: string; after_round?: number | null }> };
/** Stage of an event from its newest saved run (and its end date). Honest about what is known: it says "after R2", not "in play". */
export function stageOf(e: StageEvent, now: number): { stage: string; tone: "neutral" | "accent" | "positive" } {
  const runs = [...(e.runs ?? [])].filter((r) => r.as_of).sort((a, b) => String(a.as_of).localeCompare(String(b.as_of)));
  const last = runs.at(-1);
  const end = e.date_end ? Date.parse(`${e.date_end}T23:59:59Z`) : NaN;
  if (Number.isFinite(end) && now > end + 86_400_000) return { stage: "Final", tone: "neutral" };
  if (last?.kind === "live") return { stage: typeof last.after_round === "number" && last.after_round > 0 ? `Live, after R${last.after_round}` : "Live", tone: "accent" };
  return { stage: last ? "Pre-event" : "No run yet", tone: "neutral" };
}
/**
 * Freshness is usability, not age alone. Three separate facts, one vocabulary everywhere in the shell:
 *   stageNow     where the event is ("Pre-tournament", "R2 in play", "R2 complete", "Final"), from the calendar and tour hours;
 *   pricesLabel  which round the published prices reflect ("Pre-tournament prices", "Prices after R1");
 *   state        "ok" (usable), "behind" (prices lag the event stage), "none" (no run yet) or "final".
 * Age is shown beside these but never decides the colour on its own. A published run that is old in the middle of a round it cannot
 * contain is "behind"; the same age overnight before play is fine. Hours are US Eastern: US tours play 07:00-20:00, DP World Tour 01:00-16:00.
 */
export type EventStatus = { stageNow: string; pricesLabel: string; pricedRound: number | null; inPlay: boolean; state: "ok" | "behind" | "none" | "final"; consequence: string; next: string };
const DAY_MS = 86_400_000;
const IN_PLAY_STALE_MS = 3 * 3_600_000;
let easternFormat: Intl.DateTimeFormat | undefined;
function easternParts(ms: number): { date: string; hour: number } {
  const parts = (easternFormat ??= new Intl.DateTimeFormat("en-CA", { timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", hourCycle: "h23" })).formatToParts(ms);
  const get = (type: string) => parts.find((p) => p.type === type)?.value ?? "";
  return { date: `${get("year")}-${get("month")}-${get("day")}`, hour: Number(get("hour")) };
}
const NEXT_UPDATE = "Prices change when the model publishes its next run. This bar checks for new runs by itself.";
export function eventStatus(e: StageEvent & { date_start?: string; tour?: string }, now: number): EventStatus {
  const runs = [...(e.runs ?? [])].filter((r) => r.as_of).sort((a, b) => String(a.as_of).localeCompare(String(b.as_of)));
  const last = runs.at(-1);
  const lastMs = last?.as_of ? Date.parse(last.as_of) : NaN;
  const ageMs = Number.isFinite(lastMs) ? now - lastMs : null;
  const pricedRound = last?.kind === "live" && typeof last.after_round === "number" && last.after_round > 0 ? last.after_round : 0;
  const pricesLabel = !last ? "No prices yet" : pricedRound > 0 ? `Prices after R${pricedRound}` : "Pre-tournament prices";
  const end = e.date_end ? Date.parse(`${e.date_end}T23:59:59Z`) : NaN;
  if (Number.isFinite(end) && now > end + DAY_MS) {
    return { stageNow: "Final", pricesLabel, pricedRound, inPlay: false, state: "final", consequence: "The event is over. These are the last prices published for it.", next: "No further updates for this event." };
  }
  const { date, hour } = easternParts(now);
  const start = e.date_start ? Date.parse(`${e.date_start}T00:00:00Z`) : NaN;
  const day = Number.isFinite(start) ? Math.round((Date.parse(`${date}T00:00:00Z`) - start) / DAY_MS) : NaN;
  const euro = (e.tour ?? "").toLowerCase() === "euro" || (e.tour ?? "").toLowerCase() === "dpwt";
  const [open, close] = euro ? [1, 16] : [7, 20];
  let stageNow = "Pre-tournament";
  let completed = 0;
  let inPlay = false;
  if (Number.isFinite(day) && day >= 0) {
    if (day > 3) { stageNow = "R4 complete"; completed = 4; }
    else if (hour >= close) { stageNow = `R${day + 1} complete`; completed = day + 1; }
    else if (hour >= open) { stageNow = `R${day + 1} in play`; completed = day; inPlay = true; }
    else { stageNow = day === 0 ? "Pre-tournament" : `R${day} complete`; completed = day; }
  }
  if (!last) return { stageNow, pricesLabel, pricedRound: null, inPlay, state: "none", consequence: "No model run has been published for this event yet, so there are no prices to read.", next: "The first run appears here once the model has priced the field." };
  if (pricedRound < completed) {
    const missing = pricedRound === 0 ? "Prices are still pre-tournament and do not include any round scores." : `Prices start from the end of R${pricedRound} and do not include R${completed} scores.`;
    return { stageNow, pricesLabel, pricedRound, inPlay, state: "behind", consequence: `${missing} Treat edges against live books with care until the next run.`, next: NEXT_UPDATE };
  }
  if (inPlay && day >= 1 && ageMs !== null && ageMs > IN_PLAY_STALE_MS) {
    return { stageNow, pricesLabel, pricedRound, inPlay, state: "behind", consequence: `R${day + 1} is in play and these prices were set before today's scores. Edges against live books may be out of date.`, next: NEXT_UPDATE };
  }
  const reading = inPlay ? `${stageNow.replace(" in play", "")} is in play; prices start from ${pricedRound > 0 ? `the end of R${pricedRound}` : "before the event"}.` : completed > 0 ? `Prices include every completed round (${pricedRound > 0 ? `through R${pricedRound}` : "none yet"}).` : "Prices are the current pre-tournament numbers.";
  return { stageNow, pricesLabel, pricedRound, inPlay, state: "ok", consequence: reading, next: NEXT_UPDATE };
}

/** The nav group (Week, Model, Operate, Review, Archive) that owns a view key: the one eyebrow rule for page headers. */
export function groupOf(viewKey: string): string | null {
  for (const group of NAVIGATION) {
    if (group.items.some((item) => item.key === viewKey || item.toggle?.some((t) => t.key === viewKey))) return group.label;
  }
  return null;
}

function toShellEvent(e: IndexEventLite, now: number): ShellEvent & { hasExplain: boolean } {
  const stamps = (e.runs ?? []).map((r) => r.as_of).filter((v): v is string => !!v).sort();
  const status = eventStatus(e, now);
  const stage = status.state === "final" ? "Final" : status.pricesLabel;
  return { uid: e.event_uid as string, name: e.name ?? (e.event_uid as string), course: e.course ?? "", tour: (e.tour ?? "").toLowerCase(), tourLabel: tourLabelOf(e.tour), stage, stageTone: status.inPlay ? "accent" : "neutral", asOf: stamps.at(-1) ?? null, status, hasExplain: !!e.explain_key };
}
/** The events of the newest week (same rule as the header always used), with tour, stage and freshness input. Default selection first. */
export function shellEvents(index: { events?: IndexEventLite[] } | null | undefined, now: number): ShellEvent[] {
  const dated = (index?.events ?? []).filter((e) => e.date_start && e.event_uid);
  if (!dated.length) return [];
  const newest = dated.map((e) => e.date_start as string).sort().at(-1);
  const rows = dated.filter((e) => e.date_start === newest).map((e) => toShellEvent(e, now));
  // Same order as explain-rules.orderEvents (tour ascending within the week); pages default to the first event with an explanation document, so it leads.
  rows.sort((a, b) => a.tour.localeCompare(b.tour));
  const lead = rows.findIndex((r) => r.hasExplain);
  if (lead > 0) rows.unshift(...rows.splice(lead, 1));
  return rows.map(({ hasExplain: _drop, ...rest }) => { void _drop; return rest; });
}
/** localStorage key that remembers the last event chosen, so moving between pages keeps it. The URL's ?e= always wins. */
export const EVENT_STORAGE_KEY = "golf-dashboard-event";
/**
 * THE single event resolver: the shell bar and every event-scoped page call this and get the same uid.
 * Order: ?e= when it names an event in the published index, else the remembered event when it is in the newest week, else the default (first of shellEvents).
 */
export function resolveEventUid(index: { events?: IndexEventLite[] } | null | undefined, param: string | null | undefined, remembered: string | null | undefined, now: number): string {
  const known = (index?.events ?? []).filter((e) => e.event_uid);
  if (param && known.some((e) => e.event_uid === param)) return param;
  const week = shellEvents(index, now);
  if (remembered && week.some((e) => e.uid === remembered)) return remembered;
  return week[0]?.uid ?? "";
}
/** The bar's event for a resolved uid; an older event reached by ?e= (outside the newest week) is built from the index so the bar still names it. */
export function shellEventFor(index: { events?: IndexEventLite[] } | null | undefined, uid: string, now: number): ShellEvent | null {
  const week = shellEvents(index, now);
  const hit = week.find((e) => e.uid === uid);
  if (hit) return hit;
  const old = (index?.events ?? []).find((e) => e.event_uid === uid);
  if (old) { const { hasExplain: _drop, ...rest } = toShellEvent(old, now); void _drop; return rest; }
  return week[0] ?? null;
}
/** The event the pages are showing: ?e= when it names a known event, else the default (first). Kept for callers that only have the week list. */
export function selectedShellEvent(events: ShellEvent[], param: string | null): ShellEvent | null {
  return events.find((e) => e.uid === param) ?? events[0] ?? null;
}
/** Same page, other event: replaces ?e= and keeps every other query parameter that is not tied to the old event. */
export function eventHref(pathname: string, search: string, uid: string): string {
  const q = new URLSearchParams(search);
  q.set("e", uid);
  q.delete("player"); q.delete("vs");
  return `${pathname}?${q.toString()}`;
}
/** Primary phone tabs: the first four keys the visitor can open; owners get Bet card, others get Why priced. */
export function primaryTabKeys(owner: boolean): string[] {
  return ["this-week", "weekly-players", owner ? "bet-card" : "why-priced", "weather-effects"];
}

/** The explain document This week will open for an event, when that is certain from the index alone: its latest run (the common case); null when the page may choose another run. */
export function explainWarmKey(events: Array<{ event_uid?: string; explain_key?: string | null; explain_run?: string | null; distribution_runs?: Array<{ kind?: string } | null> }> | undefined, uid: string | null): string | null {
  const event = events?.find((e) => e.event_uid === uid);
  if (!event?.explain_key) return null;
  const hasLive = (event.distribution_runs ?? []).some((r) => r?.kind === "live");
  return hasLive && !(event.explain_run ?? "").startsWith("live_") ? null : event.explain_key;
}

/** The slim variant of an explain document key (Worker-computed, without the round-score grid and head-to-head table the first screens never read). */
export const slimExplainKey = (key: string): string => `${key}?slim=1`;

/**
 * Inline <head> script: start the two requests every first screen waits on (the event index, then This week's explain document for the event
 * the page will resolve) while the framework is still downloading, instead of after hydration. app/data.ts takes each promise once
 * (window.__earlyFetch) and falls back to a normal fetch on any miss, so a wrong guess here only costs bytes, never correctness.
 * The event choice mirrors resolveEventUid + explainWarmKey above; tests/shell.test.mjs runs this script against them.
 */
export function earlyFetchScript(): string {
  return `(function(){try{if(localStorage.getItem("golf-fonts")==="1")document.documentElement.classList.add("fonts-ready")}catch(x){}try{var E={};window.__earlyFetch=E;var go=function(u,c){var p=fetch(u,{headers:{accept:"application/json"},cache:c,priority:"high",credentials:"same-origin"});p.catch(function(){});E[u]={p:p,c:c};return p};`
    + `var ix=go("/api/golfprice/index.json","no-cache"),P=location.pathname;`
    + `if(P==="/players"){var sp=new URLSearchParams(location.search),pid=sp.get("player");if(pid&&/^(?:[1-9][0-9]*|historical-[a-z0-9]+)$/.test(pid)&&!sp.get("generation"))go("/api/golfprice/player_profiles/golfer/search-index.json?id="+pid,"no-cache")}`
    + `var W=P==="/why-priced",X=P==="/weather-effects",S=P==="/round-scores",D=P==="/distributions",I=P==="/inputs";if(P!=="/"&&P!=="/this-week"&&!W&&!X&&!S&&!D&&!I)return;`
    + `ix.then(function(r){return r.ok?r.clone().json():null}).then(function(j){var ev=j&&j.events;if(!ev)return;`
    + `var known=ev.filter(function(e){return e.event_uid}),dated=known.filter(function(e){return e.date_start}),q=new URLSearchParams(location.search).get("e"),m=null,uid="";`
    + `try{m=localStorage.getItem("${EVENT_STORAGE_KEY}")}catch(x){}`
    + `if(q&&known.some(function(e){return e.event_uid===q}))uid=q;else if(dated.length){var n=dated.map(function(e){return e.date_start}).sort().pop(),`
    + `rows=dated.filter(function(e){return e.date_start===n});rows.sort(function(a,b){return String(a.tour||"").toLowerCase().localeCompare(String(b.tour||"").toLowerCase())});`
    + `var l=-1;rows.some(function(e,i){if(e.explain_key){l=i;return true}});if(l>0)rows.unshift(rows.splice(l,1)[0]);`
    + `uid=m&&rows.some(function(e){return e.event_uid===m})?m:rows[0].event_uid}`
    + `var e=known.filter(function(x){return x.event_uid===uid})[0];if(!e)return;`
    // Weather: the brief (or, without one, the weather document) is the first screen; ask for both now (mirrors briefKeyFor / weatherKey and fetchCacheMode in the view).
    + `if(X){var wc=function(k){return /(^|\\/)(latest|brief_latest)\\.json(\\?|$)/.test(k)&&!/(^|\\/)(?:[a-z]+-)*[0-9a-f]{20}(\\/|\\.json|$)/.test(k)?"no-cache":"default"};`
    + `if(e.weather_brief_key)go("/api/"+e.weather_brief_key,wc(e.weather_brief_key));var wd=e.weather_key||"golfprice/weather/"+e.event_uid.replace(/:/g,"_")+"/latest.json";go("/api/"+wd,wc(wd));return}`
    // Scoring: the newest checkpoint document and the weather document (mirrors checkpoints() and the weather key in ScoringView).
    + `if(S||D){var wc2=function(k){return /(^|\\/)(latest|brief_latest)\\.json(\\?|$)/.test(k)&&!/(^|\\/)(?:[a-z]+-)*[0-9a-f]{20}(\\/|\\.json|$)/.test(k)?"no-cache":"default"};`
    + `var cp=(e.distribution_runs||[]).filter(function(r){return r&&r.validated&&r.key&&(r.kind==="week"||(r.after_round||0)>=1)});cp.sort(function(a,b){return String(a.as_of||"").localeCompare(String(b.as_of||""))||(a.kind==="live"?1:0)-(b.kind==="live"?1:0)});`
    + `var rk=cp.length?cp[cp.length-1].key:e.explain_key;if(rk)go("/api/"+rk,wc2(rk));if(D)return;var sw=e.weather_key||"golfprice/weather/"+e.event_uid.replace(/:/g,"_")+"/latest.json";go("/api/"+sw,wc2(sw));return}`
    // Model inputs: the run the page opens (mirrors InputsView: the latest run, else the first).
    + `if(I){var ir=(e.runs||[]).filter(function(r){return r&&r.key}),ik=ir.filter(function(r){return r.key===e.latest_key})[0]||ir[0];if(ik)go("/api/"+ik.key,"default");return}`
    + `if(!e.explain_key)return;`
    + `var runs=e.distribution_runs||[],live=runs.some(function(r){return r&&r.kind==="live"});`
    + `if(W){var cur=String(e.explain_run||"").indexOf("live_")===0;if(live||cur){var v=new URLSearchParams(location.search).get("v");if(v&&v!=="pre")return;`
    + `var wk=runs.filter(function(r){return r&&r.kind==="week"&&typeof r.key==="string"&&r.key&&r.validated!==false}).sort(function(a,b){return String(a.as_of||"").localeCompare(String(b.as_of||""))}).pop();`
    + `if(wk&&(live||cur))go("/api/"+wk.key+"?slim=1","default");return}go("/api/"+e.explain_key+"?slim=1","default");return}`
    + `if(live&&String(e.explain_run||"").indexOf("live_")!==0)return;`
    + `go("/api/"+e.explain_key+"?slim=1","default");`
    // The "What changed" strip waits for the stored diff of the latest run against the run before it; ask for it now (mirrors latestChangesDataName).
    + `var rs=runs.filter(function(r){return r&&typeof r.key==="string"&&r.key&&typeof r.kind==="string"&&typeof r.run==="string"&&r.validated!==false}),L=null;`
    + `rs.forEach(function(r){if(r.kind+"_"+r.run===e.explain_run)L=r});`
    + `if(L){var at=Date.parse(L.as_of||""),pv=rs.filter(function(r){var t=Date.parse(r.as_of||"");return r.kind+"_"+r.run!==e.explain_run&&isFinite(at)&&(isFinite(t)?t:Infinity)<at}).sort(function(a,b){return (Date.parse(b.as_of||"")||0)-(Date.parse(a.as_of||"")||0)});`
    + `if(pv[0]&&pv[0].key!==L.key)go("/api/golfprice/changes.json?event="+encodeURIComponent(e.event_uid)+"&from="+encodeURIComponent(pv[0].key)+"&to="+encodeURIComponent(L.key),"default")}}).catch(function(){})}catch(x){}})()`;
}

/* ------------------------------------------------------------------ stale-chunk recovery (used by ViewBoundary.tsx) */
/** A failed dynamic import after a deploy: the old page asks for a chunk hash that no longer exists. */
export function isChunkLoadError(error: unknown): boolean {
  const e = error as { name?: string; message?: string } | null;
  const text = `${e?.name ?? ""} ${e?.message ?? String(error ?? "")}`;
  return /ChunkLoadError|Failed to fetch dynamically imported module|error loading dynamically imported module|Importing a module script failed|Loading chunk [\w-]+ failed|Loading CSS chunk/i.test(text);
}
const RELOAD_WINDOW_MS = 60_000;
/** Pure guard: a reload is allowed unless one already started within the last minute (a loop), so a tab left open across two deploys recovers twice. */
export function mayReloadForChunk(lastReloadAt: string | null, now: number): boolean {
  const last = Number(lastReloadAt);
  return !(Number.isFinite(last) && last > 0 && now - last < RELOAD_WINDOW_MS);
}
