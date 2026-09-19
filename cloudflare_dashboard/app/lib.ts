export type DataRow = Record<string, unknown>;

export const palette = ["#54d6c8", "#ffba69", "#8ca7ff", "#f27ea9", "#a58cff", "#81d77a"];

export function numberValue(value: unknown, fallback = 0): number {
  if (typeof value === "number") return Number.isFinite(value) ? value : fallback;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export function titleCase(value: unknown): string {
  return String(value ?? "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function uniqueStrings(rows: DataRow[], key: string): string[] {
  return [...new Set(rows.map((row) => String(row[key] ?? "").trim()).filter(Boolean))].sort((a, b) =>
    a.localeCompare(b),
  );
}

export function americanOdds(probability: number): string {
  if (!Number.isFinite(probability) || probability <= 0 || probability >= 1) return "—";
  if (probability >= 0.5) return String(Math.round((-probability / (1 - probability)) * 100));
  return `+${Math.round(((1 - probability) / probability) * 100)}`;
}

export function formatCell(value: unknown, key = ""): string {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "object") return JSON.stringify(value);
  if (typeof value === "number") {
    if (/prob|pct|percent/i.test(key) && Math.abs(value) <= 1) return `${(value * 100).toFixed(1)}%`;
    if (/edge|roi|rate/i.test(key)) return `${value.toFixed(1)}%`;
    if (Number.isInteger(value)) return value.toLocaleString();
    return value.toLocaleString(undefined, { maximumFractionDigits: 3 });
  }
  return titleCase(value);
}

export function displayDate(value: unknown): string {
  const date = new Date(String(value ?? ""));
  if (Number.isNaN(date.getTime())) return "Not available";
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

export function safeMean(values: number[]): number {
  const finite = values.filter(Number.isFinite);
  return finite.length ? finite.reduce((sum, value) => sum + value, 0) / finite.length : 0;
}

export function weightedMean(rows: DataRow[], valueKey: string, weightKey = "rounds"): number {
  let weightedTotal = 0;
  let totalWeight = 0;

  for (const row of rows) {
    const value = numberValue(row[valueKey], Number.NaN);
    const weight = numberValue(row[weightKey], 0);
    if (!Number.isFinite(value) || weight <= 0) continue;
    weightedTotal += value * weight;
    totalWeight += weight;
  }

  if (totalWeight > 0) return weightedTotal / totalWeight;
  return safeMean(rows.map((row) => numberValue(row[valueKey], Number.NaN)));
}

export function diagnosticRoundCount(rows: DataRow[]): number {
  const totalRows = rows.filter((row) => String(row.category) === "total");
  if (totalRows.length) return sum(totalRows.map((row) => numberValue(row.rounds)));

  const roundsByEvent = new Map<string, number>();
  for (const row of rows) {
    const eventKey = `${String(row.year ?? "")}:${String(row.event_id ?? "")}`;
    roundsByEvent.set(eventKey, Math.max(roundsByEvent.get(eventKey) ?? 0, numberValue(row.rounds)));
  }
  return sum([...roundsByEvent.values()]);
}

export function diagnosticEventCount(rows: DataRow[]): number {
  return new Set(rows.map((row) => `${String(row.year ?? "")}:${String(row.event_id ?? "")}`)).size;
}

export function sum(values: number[]): number {
  return values.filter(Number.isFinite).reduce((total, value) => total + value, 0);
}
