/**
 * Bet card (owner-only page): the current suggested bets published by golfprice's bet_card stage (R2 owner/bet_card.json).
 * Suggestions only; nothing is placed automatically. Everything here is pure (no DOM) so tests import it directly.
 */
import { bookLabel, displayName } from "./bet-breakdowns-rules.ts";
import { eventStatus } from "./shell-rules.ts";

export { bookLabel, displayName };
export const BET_CARD_KEY = "bet_card.json";
export const OWNER_DATA_PREFIX = "/api/owner-data/";
export const SCHEMA = "golfprice.bet_card.v1";

export type Tier = "LEAN IN" | "SMALL STAKE";
export type Bet = {
  eventUid: string; tier: Tier; market: string; round: number; book: string; player: string; opponent: string | null;
  dec: number; minDec: number | null; p: number; edge: number; tieRule: string | null; hist: number | null; oppHist: number | null;
  ageH: number | null; why: string; stakeU: number; alsoAt: string; key: string; firstSeen: string | null; firstDec: number | null; isNew: boolean;
};
export type CardEvent = {
  eventUid: string; name: string; tour: string; dateStart: string; status: string; note: string | null; run: string | null;
  moment: string; pricedRound: number | null; nRows: number | null; coverage: Array<{ market: string; book: string; n: number }>;
  bets: Bet[]; units: number;
};
export type Card = { generatedAt: string; events: CardEvent[]; notes: string[] };

const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const str = (v: unknown): string => (typeof v === "string" ? v : "");

/** Stable sort: LEAN IN first, then edge descending, then the original order. */
export function sortBets(bets: Bet[]): Bet[] {
  return bets.map((b, i) => ({ b, i })).sort((x, y) => (Number(y.b.tier === "LEAN IN") - Number(x.b.tier === "LEAN IN")) || (y.b.edge - x.b.edge) || (x.i - y.i)).map((o) => o.b);
}

function decodeBet(r: Record<string, unknown>): Bet | null {
  const dec = num(r.dec), p = num(r.p), edge = num(r.edge);
  const tier = r.tier === "LEAN IN" || r.tier === "SMALL STAKE" ? r.tier : null;
  if (dec === null || p === null || edge === null || !tier || typeof r.player !== "string" || typeof r.market !== "string") return null;
  return {
    eventUid: str(r.event_uid), tier, market: r.market, round: num(r.round) ?? 0, book: str(r.book), player: r.player,
    opponent: typeof r.opponent === "string" && r.opponent ? r.opponent : null, dec, minDec: num(r.min_dec), p, edge,
    tieRule: typeof r.tie_rule === "string" ? r.tie_rule : null, hist: num(r.hist), oppHist: num(r.opp_hist), ageH: num(r.age_h),
    why: str(r.why), stakeU: num(r.stake_u) ?? 0, alsoAt: str(r.also_at), key: str(r.key),
    firstSeen: typeof r.first_seen === "string" ? r.first_seen : null, firstDec: num(r.first_dec), isNew: r.new === true,
  };
}

/** Decode the published document. Throws a plain-words Error when it is not the expected schema. */
export function decodeCard(json: unknown): Card {
  const doc = json as Record<string, unknown> | null;
  if (!doc || typeof doc !== "object" || doc.schema !== SCHEMA || !Array.isArray(doc.events) || !Array.isArray(doc.rows)) {
    throw new Error("The bet card file is not in the expected format");
  }
  const rows = (doc.rows as unknown[]).map((r) => (r && typeof r === "object" ? decodeBet(r as Record<string, unknown>) : null)).filter((b): b is Bet => b !== null);
  const events: CardEvent[] = (doc.events as unknown[]).filter((e): e is Record<string, unknown> => !!e && typeof e === "object").map((e) => {
    const eventUid = str(e.event_uid);
    const bets = sortBets(rows.filter((b) => b.eventUid === eventUid));
    const cov = e.coverage_fresh_owner_quotes && typeof e.coverage_fresh_owner_quotes === "object" ? (e.coverage_fresh_owner_quotes as Record<string, unknown>) : {};
    return {
      eventUid, name: str(e.event_name) || eventUid, tour: str(e.tour), dateStart: str(e.date_start), status: str(e.status),
      note: typeof e.note === "string" && e.note ? e.note : null, run: typeof e.run === "string" ? e.run : null, moment: str(e.moment),
      pricedRound: num(e.priced_round), nRows: num(e.n_rows),
      coverage: Object.entries(cov).filter(([, n]) => typeof n === "number").map(([k, n]) => { const [market, book] = k.split("|"); return { market, book: book ?? "", n: n as number }; })
        .sort((a, b) => a.market.localeCompare(b.market) || a.book.localeCompare(b.book)),
      bets, units: Math.round(bets.reduce((s, b) => s + b.stakeU, 0) * 100) / 100,
    };
  });
  return { generatedAt: str(doc.generated_at), events, notes: Array.isArray(doc.notes) ? (doc.notes as unknown[]).filter((n): n is string => typeof n === "string") : [] };
}

/* ------------------------------------------------------------------ wording */
export function americanOdds(dec: number): string {
  if (!(dec > 1)) return "n/a";
  return dec >= 2 ? `+${Math.round((dec - 1) * 100)}` : `-${Math.round(100 / (dec - 1))}`;
}

/** Minimum acceptable price, rounded so it is never below the threshold: decimal rounded UP to the hundredth; American chosen so its
 * decimal equivalent is at least the threshold (plus side rounded up, minus side toward zero). */
export function minPriceText(minDec: number): { dec: string; american: string } {
  if (!(minDec > 1)) return { dec: "n/a", american: "n/a" };
  const dec = Math.ceil(minDec * 100 - 1e-9) / 100;
  const american = minDec >= 2 ? `+${Math.ceil((minDec - 1) * 100 - 1e-9)}` : `-${Math.floor(100 / (minDec - 1) + 1e-9)}`;
  return { dec: dec.toFixed(2), american };
}

const TOP_LABEL: Record<string, string> = { top_5: "top 5", top_10: "top 10", top_20: "top 20" };
export const MARKET_LABEL: Record<string, string> = {
  matchup_round: "Round matchup", matchup_72: "72-hole matchup", make_cut: "Make the cut", mc: "Miss the cut", top_5: "Top 5", top_10: "Top 10", top_20: "Top 20",
};

/** The bet in plain words, e.g. "Rico Hoey over Kevin Yu, 72-hole". */
export function betText(b: Pick<Bet, "market" | "round" | "player" | "opponent">): string {
  const who = displayName(b.player);
  const opp = b.opponent ? displayName(b.opponent) : "";
  switch (b.market) {
    case "matchup_72": return `${who} over ${opp}, 72-hole`;
    case "matchup_round": return `${who} over ${opp}, R${b.round}`;
    case "make_cut": return `${who} to make the cut`;
    case "mc": return `${who} to miss the cut`;
    default: return TOP_LABEL[b.market] ? `${who} ${TOP_LABEL[b.market]}` : `${who} (${b.market})`;
  }
}

export const pctText = (v: number, digits = 1) => `${(v * 100).toFixed(digits)}%`;

/** "2026-10-22 06:05 ET" */
export function formatET(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "unknown time";
  const parts = Object.fromEntries(new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hourCycle: "h23" }).formatToParts(d).map((p) => [p.type, p.value]));
  return `${parts.year}-${parts.month}-${parts.day} ${parts.hour}:${parts.minute} ET`;
}

export function timeAgo(iso: string, now: Date = new Date()): string {
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return "unknown";
  const min = Math.max(0, Math.round((now.getTime() - t) / 60000));
  if (min < 1) return "just now";
  if (min < 60) return `${min} min ago`;
  const h = Math.floor(min / 60);
  if (h < 48) return min % 60 === 0 ? `${h} h ago` : `${h} h ${min % 60} min ago`;
  return `${Math.floor(h / 24)} days ago`;
}

/* ------------------------------------------------------------------ filters */
export type Group = "all" | "matchups" | "cut" | "top";
export type TierFilter = "all" | "lean" | "small";
export type Filters = { tier: TierFilter; group: Group; book: string };
export const DEFAULT_FILTERS: Filters = { tier: "all", group: "all", book: "all" };

export const GROUP_OPTIONS: Array<{ value: Group; label: string }> = [
  { value: "all", label: "All markets" }, { value: "matchups", label: "Matchups" }, { value: "cut", label: "Cut" }, { value: "top", label: "Top finish" },
];
export function groupOf(market: string): Exclude<Group, "all"> {
  if (market === "matchup_round" || market === "matchup_72") return "matchups";
  if (market === "make_cut" || market === "mc") return "cut";
  return "top";
}

export function filterBets(bets: Bet[], f: Filters): Bet[] {
  return bets.filter((b) => (f.tier === "all" || (f.tier === "lean") === (b.tier === "LEAN IN")) && (f.group === "all" || groupOf(b.market) === f.group) && (f.book === "all" || b.book === f.book));
}

export function booksIn(card: Card): string[] {
  return [...new Set(card.events.flatMap((e) => e.bets.map((b) => b.book)))].sort();
}

export function totals(bets: Bet[]): { bets: number; units: number } {
  return { bets: bets.length, units: Math.round(bets.reduce((s, b) => s + b.stakeU, 0) * 100) / 100 };
}

export function historyText(b: Pick<Bet, "hist" | "oppHist" | "opponent">): string {
  const h = b.hist === null ? "no history" : `${Math.round(b.hist)} rounds`;
  return b.opponent && b.oppHist !== null ? `${h} vs ${Math.round(b.oppHist)} for the opponent` : h;
}

/* ------------------------------------------------------------------ status wording */
/** Plain status for an event with no suggestions, e.g. "Round 1 in play". */
export function statusHeadline(e: Pick<CardEvent, "status" | "note" | "moment">): string {
  const m = /round (\d) in play/i.exec(e.note ?? "") ?? /round (\d) in play/i.exec(e.moment);
  if (m) return `Round ${m[1]} in play`;
  if (e.status === "priced") return "Priced, no bet clears the rules";
  return e.status ? e.status.charAt(0).toUpperCase() + e.status.slice(1) : "No status";
}

/** Honest next-check text: round prices only appear after the round's own run; before play the card refreshes every 30 minutes. */
export function nextCheckText(e: Pick<CardEvent, "status" | "note" | "moment">): string {
  if (/in play/i.test(`${e.note ?? ""} ${e.moment}`)) return "Next check: after the round finishes (next-round prices come from the after-round run)";
  if (/off week|not started|no event/i.test(e.note ?? "")) return "Next check: when the next event is priced";
  return "Next check: within 30 minutes (the card refreshes every 30 minutes before play)";
}

/** Share of the edge bar that is filled: 15% edge or more fills it. */
export const edgeFill = (edge: number): number => Math.max(0.04, Math.min(1, edge / 0.15));

export const activeFilterCount = (f: Filters): number => Number(f.tier !== "all") + Number(f.group !== "all") + Number(f.book !== "all");

/**
 * Where an event stands, in the same words as the context bar: the bar's own eventStatus (calendar + tour hours) supplies the stage
 * ("R1 complete", "R2 in play") and which round the prices reflect ("Prices after R1"). Without a start date the card's own text is used.
 */
export function stageText(e: Pick<CardEvent, "status" | "note" | "moment" | "pricedRound"> & Partial<Pick<CardEvent, "dateStart" | "tour" | "run">>, now: number = Date.now()): string {
  const after = /after round (\d)/i.exec(e.moment);
  const priced = after ? Number(after[1]) : 0;
  if (e.dateStart && Number.isFinite(Date.parse(`${e.dateStart}T00:00:00Z`))) {
    const end = new Date(Date.parse(`${e.dateStart}T00:00:00Z`) + 3 * 86_400_000).toISOString().slice(0, 10);
    const s = eventStatus({ date_start: e.dateStart, date_end: end, tour: e.tour, runs: e.run || e.moment || e.status === "priced" ? [{ as_of: e.run ?? new Date(now).toISOString(), kind: priced > 0 ? "live" : "week", after_round: priced > 0 ? priced : null }] : [] }, now);
    return `${s.stageNow} · ${s.pricesLabel}`;
  }
  const play = /round (\d) in play/i.exec(e.note ?? "") ?? /round (\d) in play/i.exec(e.moment);
  if (play) return `R${play[1]} in play`;
  if (after) return `Prices after R${after[1]}${e.pricedRound ? ` · pricing R${e.pricedRound}` : ""}`;
  if (/pre-?event/i.test(e.moment)) return "Pre-tournament prices";
  return e.moment || (e.status ? e.status.charAt(0).toUpperCase() + e.status.slice(1) : "No status");
}

/** One short next-check phrase for the scope strip. */
export function nextCheckShort(e: Pick<CardEvent, "status" | "note" | "moment">): string {
  if (/in play/i.test(`${e.note ?? ""} ${e.moment}`)) return "next prices after the round";
  if (/off week|not started|no event/i.test(e.note ?? "")) return "next check when the next event is priced";
  return "next check within 30 minutes";
}
