/** American spelling for prose that arrives in published data (owner 2026-10-13: American spelling everywhere on the site).
 *  Only strings that contain a space are rewritten, so single-word codes and keys ("cancelled", "centre") are left exactly as published. */
const WORDS: Record<string, string> = {
  favourites: "favorites", favourite: "favorite", favoured: "favored", favouring: "favoring", favours: "favors", favour: "favor",
  colours: "colors", coloured: "colored", colouring: "coloring", colour: "color", neighbouring: "neighboring", neighbours: "neighbors", neighbour: "neighbor",
  centred: "centered", centre: "center", centres: "centers", travelled: "traveled", travelling: "traveling", labelled: "labeled", labelling: "labeling",
  modelled: "modeled", modelling: "modeling", grey: "gray", standardisation: "standardization", standardised: "standardized", standardise: "standardize",
  realised: "realized", normalised: "normalized", towards: "toward", cancelled: "canceled", analysed: "analyzed", analyse: "analyze", optimised: "optimized",
  minimised: "minimized", summarised: "summarized", recognised: "recognized", organised: "organized", behaviour: "behavior", penalised: "penalized",
  prioritised: "prioritized", utilised: "utilized", stabilised: "stabilized", whilst: "while", amongst: "among", programme: "program", metres: "meters", metre: "meter",
  characterised: "characterized", randomised: "randomized", generalised: "generalized", categorised: "categorized", calibre: "caliber",
};
const RX = new RegExp(`\b(${Object.keys(WORDS).sort((a, b) => b.length - a.length).join("|")})\b`, "gi");

export function americanize(text: string): string {
  if (!text.includes(" ")) return text;
  return text.replace(RX, (w) => {
    const r = WORDS[w.toLowerCase()];
    return w === w.toUpperCase() ? r.toUpperCase() : w[0] === w[0].toUpperCase() ? r[0].toUpperCase() + r.slice(1) : r;
  });
}

/** Deep copy with every prose string value americanized; object keys are never changed. */
export function americanizeDeep<T>(value: T): T {
  if (typeof value === "string") return americanize(value) as T;
  if (Array.isArray(value)) return value.map((v) => americanizeDeep(v)) as T;
  if (value && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) out[k] = americanizeDeep(v);
    return out as T;
  }
  return value;
}
