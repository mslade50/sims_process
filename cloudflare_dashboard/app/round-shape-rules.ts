/** Volatility and shape of a set of round outcomes, compared with the PGA rounds in the same slice. Dependency-free. */
export type RoundShape = { n: number; mean: number; median: number; sd: number; skew: number; hi: number; lo: number; hiCut: number };
export type ShapeRead = { headline: string; spread: string; shape: string; tails: string; caveat: string; small: boolean };

export const TAIL = 3, MIN_ROUNDS = 12;
const fin = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const pct = (x: number) => `${(x * 100).toFixed(x < .1 ? 1 : 0)}%`;

/** SD, skew and the share of rounds `cut` strokes above/below the group's own median (so the numbers describe shape, not level). */
export function roundShape(values: number[], cut = TAIL): RoundShape | null {
  const v = values.filter(fin);
  if (v.length < MIN_ROUNDS) return null;
  const n = v.length, mean = v.reduce((a, b) => a + b, 0) / n, s = [...v].sort((a, b) => a - b), median = n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
  const m2 = v.reduce((a, b) => a + (b - mean) ** 2, 0) / n, m3 = v.reduce((a, b) => a + (b - mean) ** 3, 0) / n, sd = Math.sqrt(m2 * n / (n - 1));
  return { n, mean, median, sd, skew: m2 > 1e-12 ? m3 / m2 ** 1.5 : 0, hi: v.filter(x => x - median >= cut).length / n, lo: v.filter(x => x - median <= -cut).length / n, hiCut: cut };
}

const SD_BAND = .08;
/** Plain reading of `own` vs `peer` (PGA rounds in the same slice). `withinSd` is a typical PGA regular's own round-to-round SD when published. */
export function readShape(own: RoundShape | null, peer: RoundShape | null, withinSd?: number | null, label = "PGA"): ShapeRead | null {
  if (!own) return null;
  const caveat = label !== "PGA" ? "Both groups are measured from their own median; small samples are descriptive only." : "The PGA comparison pools many players, so its spread also includes skill differences between them and runs a little wider than any one golfer's own spread. Shares and skew are measured from each group's own median.";
  if (!peer) return { headline: "No comparison", spread: `Spread ${own.sd.toFixed(1)} strokes`, shape: "", tails: `${pct(own.hi)} of rounds ≥ +${own.hiCut} and ${pct(own.lo)} ≤ −${own.hiCut} from the median`, caveat, small: own.n < 20 };
  const refSd = fin(withinSd) && withinSd > 0 ? withinSd : peer.sd, refName = fin(withinSd) && withinSd > 0 ? "typical PGA regular" : label;
  const rel = own.sd / refSd - 1, band = rel < -SD_BAND ? "tighter than typical" : rel > SD_BAND ? "looser than typical" : "about typical";
  const spread = `Spread ${own.sd.toFixed(1)} vs ${refName} ${refSd.toFixed(1)} · ${band}`;
  const z = (p: number, p0: number, n: number) => (p - p0) / Math.sqrt(Math.max(p0 * (1 - p0), .0025) / n);
  const zHi = z(own.hi, peer.hi, own.n), zLo = z(own.lo, peer.lo, own.n), dSkew = own.skew - peer.skew, seSkew = Math.sqrt(6 / own.n);
  const goodTail = zHi >= 1.5 && own.hi > peer.hi * 1.15, badTail = zLo >= 1.5 && own.lo > peer.lo * 1.15, fewBad = zLo <= -1.5, fewGood = zHi <= -1.5;
  const tails = `${pct(own.hi)} of rounds ≥ +${own.hiCut} vs ${pct(peer.hi)} for ${label} · ${pct(own.lo)} ≤ −${own.hiCut} vs ${pct(peer.lo)}`;
  let shape: string, lean = "";
  if (goodTail && !badTail) { shape = "Leans to big good rounds"; lean = "good"; }
  else if (badTail && !goodTail) { shape = "Leans to big bad rounds"; lean = "bad"; }
  else if (goodTail && badTail) shape = "Big rounds both ways";
  else if (fewGood && fewBad) shape = "Few extreme rounds either way";
  else if (fewBad) shape = "Rarely has a blow-up round";
  else if (fewGood) shape = "Rarely has a big good round";
  else if (Math.abs(dSkew) > 2 * seSkew) { shape = dSkew > 0 ? "Long tail of good rounds" : "Long tail of bad rounds"; lean = dSkew > 0 ? "good" : "bad"; }
  else shape = `Shape close to ${label === "PGA" ? "the PGA field" : label}`;
  const headline = band === "tighter than typical" ? "Steadier than typical" : band === "looser than typical" ? "Streakier than typical" : "Typical spread";
  return { headline, spread, shape: `${shape}${lean ? ` (skew ${own.skew >= 0 ? "+" : "−"}${Math.abs(own.skew).toFixed(1)} vs ${peer.skew >= 0 ? "+" : "−"}${Math.abs(peer.skew).toFixed(1)})` : ""}`, tails, caveat, small: own.n < 20 };
}
