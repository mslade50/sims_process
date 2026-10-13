/** Golfer names as shown on the site. Dependency-free so tests can load it directly. */
const TAG = /\s*\(\s*(?:am|a|amateur)\s*\)\s*/gi;
const SUFFIX = /^(?:jr|sr|ii|iii|iv|v)\.?$/i;

/** True when the provider name carries an amateur tag such as "(Am)". */
export const isAmateurTag = (raw: string | null | undefined): boolean => !!raw && /\(\s*(?:am|a|amateur)\s*\)/i.test(raw);

/** "Aberg(Am), Ludvig" -> "Ludvig Aberg"; "van Rooyen, Erik" -> "Erik van Rooyen"; "Smith, John, Jr." -> "John Smith Jr."; single names unchanged. */
export function playerDisplayName(raw: string | null | undefined): string {
  if (!raw) return "";
  const parts = raw.replace(TAG, " ").replace(/\s+/g, " ").trim().split(",").map(s => s.trim()).filter(Boolean);
  if (parts.length === 2 && SUFFIX.test(parts[1])) return `${parts[0]} ${parts[1]}`;
  if (parts.length === 2) return `${parts[1]} ${parts[0]}`;
  if (parts.length === 3 && SUFFIX.test(parts[2])) return `${parts[1]} ${parts[0]} ${parts[2]}`;
  if (parts.length === 3 && SUFFIX.test(parts[1])) return `${parts[2]} ${parts[0]} ${parts[1]}`;
  return parts.join(" ");
}

/** Up to two initials from a display name ("Ludvig Aberg" -> "LA"). */
export const nameInitials = (raw: string | null | undefined): string => playerDisplayName(raw).split(" ").filter(Boolean).map(w => w[0]).slice(0, 2).join("").toUpperCase();
