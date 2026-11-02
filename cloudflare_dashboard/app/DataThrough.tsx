import "./freshness.css";

/** One quiet "Data through <date>" line (or a callout when `warn`). Draws nothing when the published field is absent. */
export function DataThrough({ text, warn = false, role }: { text: string | null | undefined; warn?: boolean; role?: "status" | "alert" }) {
  if (!text) return null;
  return <p className={`data-through${warn ? " warn" : ""}`} data-testid="data-through" role={role ?? (warn ? "status" : undefined)}>{text}</p>;
}
