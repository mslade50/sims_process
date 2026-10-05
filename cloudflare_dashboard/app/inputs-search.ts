/** Pure text-matching helpers for the Model inputs area (player type-ahead and the Features filter). No React, no DOM: unit-tested in tests/inputs-search.test.mjs. */

const EXTRA: Record<string, string> = { ø: "o", æ: "ae", œ: "oe", ß: "ss", đ: "d", ð: "d", ł: "l", þ: "th", ı: "i" };

/** Lower-case, strip accents and apostrophes, keep letters and digits, collapse everything else to single spaces. */
export function fold(text: string): string {
  const lowered = text.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
  return lowered
    .replace(/[øæœßđðłþı]/g, (ch) => EXTRA[ch] ?? ch)
    .replace(/['’`]/g, "")
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

const tokens = (text: string): string[] => fold(text).split(" ").filter(Boolean);
const compact = (text: string): string => fold(text).replaceAll(" ", "");

/** "Scheffler, Scottie" -> "Scottie Scheffler"; names without a comma are returned unchanged. */
export function firstLast(name: string): string {
  const parts = name.split(",");
  return parts.length === 2 ? `${parts[1].trim()} ${parts[0].trim()}` : name;
}

/** Rank of a match (lower is better) or null: every typed word must start a word of the name ("sch scot", "scottie scheffler", "scheffler, scottie");
 *  a typed string with the spaces removed may also sit inside the name ("o neill" for "O'Neill", "scottiesch"). */
export function matchRank(name: string, query: string): number | null {
  const q = tokens(query);
  if (!q.length) return null;
  const forward = tokens(name);
  const swapped = tokens(firstLast(name));
  const words = [...new Set([...forward, ...swapped])];
  if (q.every((token) => words.some((word) => word.startsWith(token)))) {
    if (fold(name) === fold(query) || fold(firstLast(name)) === fold(query)) return 0;
    return forward[0]?.startsWith(q[0]) ? 1 : 2;
  }
  const flat = q.join("");
  if (compact(name).includes(flat) || compact(firstLast(name)).includes(flat)) return 3;
  return null;
}

/** Type-ahead matches, best first (then alphabetical), at most `limit`. */
export function matchPlayers<T extends { name: string }>(items: T[], query: string, limit = 8): T[] {
  const ranked: Array<{ item: T; rank: number }> = [];
  for (const item of items) {
    const rank = matchRank(item.name, query);
    if (rank !== null) ranked.push({ item, rank });
  }
  ranked.sort((a, b) => a.rank - b.rank || a.item.name.localeCompare(b.item.name));
  return ranked.slice(0, limit).map((entry) => entry.item);
}

/** Free-text filter for the Features tab: every typed word must appear (accent/case-insensitive) somewhere in the haystack. */
export function matchText(haystack: string, query: string): boolean {
  const q = tokens(query);
  if (!q.length) return true;
  const text = fold(haystack);
  return q.every((token) => text.includes(token));
}
