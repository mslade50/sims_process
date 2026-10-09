"use client";

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Check, ChevronDown, Download, Search, SlidersHorizontal, X } from "lucide-react";
import { DataRow, etTime, formatCell, numberValue, titleCase } from "./lib";
import { AnimatedNumber, Delta, SkeletonPage, Sparkline } from "./ui";

export function Panel({
  title,
  eyebrow,
  actions,
  children,
  className = "",
}: {
  title?: string;
  eyebrow?: string;
  actions?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel ${className}`}>
      {(title || eyebrow || actions) && (
        <div className="panel-heading">
          <div>
            {eyebrow && <span className="eyebrow">{eyebrow}</span>}
            {title && <h2>{title}</h2>}
          </div>
          {actions && <div className="panel-actions">{actions}</div>}
        </div>
      )}
      {children}
    </section>
  );
}

export function Kpi({
  label,
  value,
  detail,
  tone = "neutral",
  delta,
  deltaSuffix = "",
  spark,
  hint,
}: {
  label: string;
  value: string;
  detail?: string;
  tone?: "positive" | "negative" | "neutral" | "accent" | "model" | "market";
  /** Optional signed change shown with a direction glyph. */
  delta?: number;
  deltaSuffix?: string;
  /** Optional trend values drawn as a sparkline. */
  spark?: number[];
  /** Optional hover definition of the label in one plain sentence. */
  hint?: string;
}) {
  const sparkTone = tone === "neutral" ? "accent" : tone;
  return (
    <div className={`kpi kpi-${tone}`}>
      <span title={hint}>{label}</span>
      <div className="kpi-row">
        <strong><AnimatedNumber text={value} /></strong>
        {spark && spark.length > 1 && <Sparkline values={spark} tone={sparkTone} label={`${label} trend`} />}
      </div>
      {(detail || delta !== undefined) && (
        <div className="kpi-foot">
          {delta !== undefined && <Delta value={delta} suffix={deltaSuffix} />}
          {detail && <small>{detail}</small>}
        </div>
      )}
    </div>
  );
}

/** Alias for new views: the same tile, named for what it is. */
export const KpiTile = Kpi;

export function PageIntro({
  eyebrow,
  title,
  description,
  controls,
}: {
  eyebrow: string;
  title: string;
  description: string;
  controls?: React.ReactNode;
}) {
  return (
    <div className="page-intro">
      <div>
        <span className="eyebrow">{eyebrow}</span>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {controls && <div className="page-controls">{controls}</div>}
    </div>
  );
}

/** One-line banner for views fed by the retired simulation pipeline (event codes, not golfprice events). */
export function LegacyNotice({ children }: { children?: React.ReactNode }) {
  return <p className="legacy-notice" role="note"><strong>Legacy data.</strong> {children ?? "This archived view comes from the retired simulation pipeline and does not show the current golfprice week."}</p>;
}

export function EmptyState({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="empty-state">
      <strong>{title}</strong>
      <span>{detail}</span>
    </div>
  );
}

export function LoadingState({ label = "Loading dashboard data" }: { label?: string }) {
  return <SkeletonPage label={label} />;
}

export function ErrorState({ message }: { message: string }) {
  return <EmptyState title="This view is temporarily unavailable" detail={message} />;
}

export function SegmentedControl<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: Array<{ value: T; label: string }>;
  onChange: (value: T) => void;
  label: string;
}) {
  return (
    <div className="segmented" role="group" aria-label={label}>
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          className={value === option.value ? "active" : ""}
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

export function PlayerPicker({
  options,
  value,
  onChange,
  max = 5,
  label = "Players",
}: {
  options: string[];
  value: string[];
  onChange: (players: string[]) => void;
  max?: number;
  label?: string;
}) {
  const available = options.filter((option) => !value.includes(option));
  return (
    <div className="player-picker">
      <label>{label}</label>
      <div className="chip-row">
        {value.map((player) => (
          <span className="chip" key={player}>
            {titleCase(player)}
            <button type="button" onClick={() => onChange(value.filter((item) => item !== player))} aria-label={`Remove ${player}`}>
              <X size={13} />
            </button>
          </span>
        ))}
      </div>
      <select
        value=""
        disabled={value.length >= max || available.length === 0}
        onChange={(event) => event.target.value && onChange([...value, event.target.value])}
        aria-label={`Add ${label.toLowerCase()}`}
      >
        <option value="">{value.length >= max ? `Maximum ${max} selected` : "Add player…"}</option>
        {available.map((player) => (
          <option key={player} value={player}>
            {titleCase(player)}
          </option>
        ))}
      </select>
    </div>
  );
}

const ISO_TIME = /^\d{4}-\d\d-\d\d[T ]\d\d:\d\d/;

/** A stored timestamp string (ISO, with or without a zone) as Eastern time; anything else is returned unchanged. */
export function humanTimestamp(text: string): string {
  if (!ISO_TIME.test(text)) return text;
  const zoned = /(Z|[+-]\d\d:?\d\d)$/.test(text) ? text : `${text}Z`;
  return etTime(zoned.replace(" ", "T"), text);
}

/**
 * Any value as readable text: timestamps in Eastern time, yes/no, lists joined, nested objects as "Label: value; Label: value".
 * Never JSON and never "[object Object]". Use it wherever a value of unknown shape must be shown to a person.
 */
export function humanValue(value: unknown, depth = 0): string {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number") return formatCell(value);
  if (typeof value === "string") return humanTimestamp(value);
  if (depth >= 3) return "…";
  if (Array.isArray(value)) {
    const shown = value.slice(0, 8).map((item) => humanValue(item, depth + 1));
    return shown.join(", ") + (value.length > 8 ? ` and ${value.length - 8} more` : "");
  }
  if (typeof value === "object") {
    const parts = Object.entries(value as Record<string, unknown>)
      .filter(([, item]) => item !== null && item !== undefined && item !== "")
      .map(([key, item]) => `${titleCase(key)}: ${humanValue(item, depth + 1)}`);
    return parts.length ? parts.join("; ") : "—";
  }
  return String(value);
}

/** True for a value that carries nothing to show (so a column made only of these is hidden). */
const isBlank = (value: unknown): boolean => {
  if (value === null || value === undefined || value === "") return true;
  if (typeof value === "number") return Number.isNaN(value);
  if (Array.isArray(value)) return value.length === 0;
  if (typeof value === "string") return ["—", "-", "n/a", "not available", "unknown", "nan"].includes(value.trim().toLowerCase());
  return false;
};

/** Plain names and one-sentence definitions for column keys that reach tables as raw field names. A table can still override either per column. */
const COLUMN_NAMES: Record<string, string> = {
  player_name: "Player", bet_on: "Bet on", opponent: "Against", bookmaker: "Sportsbook", bet_type: "Bet type", event_name: "Event",
  edge: "Kelly edge %", raw_edge: "Raw edge %", dec_odds: "Decimal odds", pred_on: "Skill estimate", sample_on: "Sample size",
  units_wagered: "Units risked", units_won: "Units won", roi: "ROI %", wagered: "Units risked", archetype: "Player type", archetype_against: "Opponent type",
  miss: "Miss", miss_centered: "Miss vs field", max_abs_miss: "Biggest miss", ott: "Off the tee", app: "Approach", arg: "Around the green", putt: "Putting",
  predicted_sg: "Predicted SG", actual_sg: "Actual SG", pred: "Prediction", my_pred_regressed: "Prediction (adjusted)", error_raw: "Error", error_regressed: "Error (adjusted)",
  regress_helped: "Adjustment helped", mkt_adj: "Market adjustment", mu_adj: "Skill adjustment", n_eff: "Effective sample", std: "Spread", skew: "Skew",
  excess_kurtosis: "Tail weight", p1_weather_adv: "Player 1 weather edge", p2_weather_adv: "Player 2 weather edge", differential: "Difference",
};
const COLUMN_HINTS: Record<string, string> = {
  edge: "Kelly edge: how much the model likes the bet, as a percent of bankroll under the staking rule.",
  raw_edge: "The model's win chance minus the sportsbook's implied chance, in percentage points.",
  dec_odds: "The payout for a 1 unit stake, including the stake (2.50 pays 1.50 profit).",
  pred_on: "The model's skill estimate for the player bet on, in strokes gained per round.",
  sample_on: "How many rounds of data sit behind that skill estimate.",
  units_wagered: "Amount risked, in betting units.", units_won: "Profit (or loss) in betting units.", roi: "Profit divided by the amount risked.",
  archetype: "A label for the style of player (for example, a long hitter or a strong putter).", archetype_against: "The style of the opponent in a matchup.",
  miss: "Actual strokes gained minus the model's prediction. Above zero means the player beat the model.",
  miss_centered: "The miss after removing the average miss for the whole field that week.",
  max_abs_miss: "The largest miss across the four shot types, ignoring the sign.",
  predicted_sg: "Strokes gained per round the model expected.", actual_sg: "Strokes gained per round the player really produced.",
  my_pred_regressed: "The model's prediction after pulling it toward the market.", error_raw: "Actual minus the original prediction.",
  error_regressed: "Actual minus the adjusted prediction.", regress_helped: "Whether pulling toward the market made the prediction closer.",
  mkt_adj: "How far the market pulled the prediction.", mu_adj: "How far the skill adjustment moved the prediction.",
  n_eff: "Effective number of rounds behind the estimate, after giving recent rounds more weight.",
  std: "How much the player's rounds swing around their average.", skew: "Whether big misses lean good (positive) or bad (negative).",
  excess_kurtosis: "How heavy the tails are: higher means more extreme rounds.",
  differential: "Gap between the two players' weather edges; positive favors player 1.",
  ott: "Off the tee.", app: "Approach shots.", arg: "Around the green.", putt: "Putting.",
};

function csvValue(value: unknown): string {
  const text = typeof value === "object" && value !== null ? humanValue(value) : String(value ?? "");
  return `"${text.replaceAll('"', '""')}"`;
}

export function DataTable({
  rows,
  preferredColumns = [],
  label,
  pageSize = 25,
  onRowClick,
  activeRow,
  defaultColumns,
  headerLabels,
  headerTitles,
  verbatim = false,
  stickyFirst = false,
  renderCell,
  mobileColumns,
}: {
  rows: DataRow[];
  preferredColumns?: string[];
  label: string;
  pageSize?: number;
  /** Optional: makes rows clickable (used by the Model inputs player table). */
  onRowClick?: (row: DataRow) => void;
  activeRow?: DataRow | null;
  /** Initial visible set (in table order). The column picker still offers every column. Default: the first nine. */
  defaultColumns?: string[];
  /** Optional display label per column key (header and picker). */
  headerLabels?: Record<string, string>;
  /** Optional hover text (title attribute) per column, on the header and in the column picker. */
  headerTitles?: Record<string, string>;
  /** Render header, picker and text cells exactly as given (no title-casing), for tables whose columns and badges are already labels. */
  verbatim?: boolean;
  /** Keep the first column visible while scrolling sideways. */
  stickyFirst?: boolean;
  /** Optional per-column cell renderer; return undefined to fall back to the default text. */
  renderCell?: (column: string, value: unknown, row: DataRow) => ReactNode | undefined;
  /** Columns still shown at phone width (600px and under); the others are hidden by CSS. Omit to show all. */
  mobileColumns?: string[];
}) {
  const allColumns = useMemo(() => {
    // A column that is empty (or only a placeholder) for every row says nothing, so it is not offered at all.
    const found = [...new Set(rows.flatMap((row) => Object.keys(row)))].filter((column) =>
      rows.some((row) => !isBlank(row[column]) && (typeof row[column] !== "object" || Array.isArray(row[column]))),
    );
    return [...preferredColumns.filter((column) => found.includes(column)), ...found.filter((column) => !preferredColumns.includes(column))];
  }, [preferredColumns, rows]);
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<{ column: string; direction: "asc" | "desc" } | null>(null);
  const [page, setPage] = useState(0);
  const [visible, setVisible] = useState<string[]>(() => {
    const chosen = defaultColumns ? allColumns.filter((column) => defaultColumns.includes(column)) : [];
    return chosen.length ? chosen : allColumns.slice(0, Math.min(9, allColumns.length));
  });
  const headText = (column: string) => headerLabels?.[column] ?? (verbatim ? column : COLUMN_NAMES[column] ?? titleCase(column));
  const headHint = (column: string) => headerTitles?.[column] ?? COLUMN_HINTS[column];
  const cellClass = (column: string, index: number, tone = "") =>
    [tone, stickyFirst && index === 0 ? "sticky-first" : "", mobileColumns && !mobileColumns.includes(column) ? "mobile-hide" : ""].filter(Boolean).join(" ");

  const activeColumns = visible.filter((column) => allColumns.includes(column));
  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const result = needle
      ? rows.filter((row) => activeColumns.some((column) => String(row[column] ?? "").toLowerCase().includes(needle)))
      : [...rows];
    if (sort) {
      result.sort((left, right) => {
        const a = left[sort.column];
        const b = right[sort.column];
        const numeric = Number(a) - Number(b);
        const compared = Number.isNaN(numeric) ? String(a ?? "").localeCompare(String(b ?? "")) : numeric;
        return sort.direction === "asc" ? compared : -compared;
      });
    }
    return result;
  }, [activeColumns, query, rows, sort]);
  const maxPage = Math.max(0, Math.ceil(filtered.length / pageSize) - 1);
  const currentPage = Math.min(page, maxPage);
  const paged = filtered.slice(currentPage * pageSize, currentPage * pageSize + pageSize);
  // When the highlighted row is changed from outside the table (a player search), show the page that holds it. Manual paging is left alone.
  const jumpedTo = useRef<DataRow | null>(activeRow ?? null);
  useEffect(() => {
    if (!activeRow || jumpedTo.current === activeRow) return;
    jumpedTo.current = activeRow;
    const index = filtered.indexOf(activeRow);
    if (index >= 0) queueMicrotask(() => setPage(Math.floor(index / pageSize)));
  }, [activeRow, filtered, pageSize]);

  function toggleSort(column: string) {
    setSort((current) =>
      current?.column === column
        ? { column, direction: current.direction === "asc" ? "desc" : "asc" }
        : { column, direction: "desc" },
    );
  }

  function downloadCsv() {
    const csv = [
      activeColumns.map(csvValue).join(","),
      ...filtered.map((row) => activeColumns.map((column) => csvValue(row[column])).join(",")),
    ].join("\n");
    const href = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
    const anchor = document.createElement("a");
    anchor.href = href;
    anchor.download = `${label.toLowerCase().replaceAll(" ", "-")}.csv`;
    anchor.click();
    URL.revokeObjectURL(href);
  }

  if (!rows.length) return <EmptyState title={`No ${label.toLowerCase()} available`} detail="The next data publish will populate this view." />;

  return (
    <div className="data-table-wrap">
      <div className="table-toolbar">
        <label className="search-box">
          <Search size={15} />
          <input
            value={query}
            onChange={(event) => {
              setQuery(event.target.value);
              setPage(0);
            }}
            placeholder={`Search ${label.toLowerCase()}…`}
          />
        </label>
        <div className="table-tools">
          <span>{filtered.length.toLocaleString()} rows</span>
          <details className="column-menu">
            <summary><SlidersHorizontal size={15} /> Columns <ChevronDown size={13} /></summary>
            <div>
              {allColumns.map((column) => {
                const checked = activeColumns.includes(column);
                return (
                  <label key={column} title={headHint(column)}>
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={() =>
                        setVisible((current) =>
                          checked ? current.filter((item) => item !== column) : [...current, column],
                        )
                      }
                    />
                    <span className="checkbox-mark">{checked && <Check size={11} />}</span>
                    {headText(column)}
                  </label>
                );
              })}
            </div>
          </details>
          <button type="button" className="icon-button" onClick={downloadCsv} aria-label={`Download ${label} CSV`}>
            <Download size={15} />
          </button>
        </div>
      </div>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              {activeColumns.map((column, index) => (
                <th key={column} className={cellClass(column, index)}>
                  <button type="button" title={headHint(column)} onClick={() => toggleSort(column)}>
                    {headText(column)}
                    {sort?.column === column && <span>{sort.direction === "asc" ? " ↑" : " ↓"}</span>}
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {paged.map((row, rowIndex) => (
              <tr
                key={`${currentPage}-${rowIndex}`}
                className={`${onRowClick ? "clickable-row" : ""} ${activeRow && activeRow === row ? "active-row" : ""}`}
                onClick={onRowClick ? () => onRowClick(row) : undefined}
              >
                {activeColumns.map((column, index) => {
                  const value = row[column];
                  const numeric = numberValue(value, Number.NaN);
                  const tone = /edge|units_won|miss_centered/i.test(column)
                    ? numeric > 0
                      ? "positive"
                      : numeric < 0
                        ? "negative"
                        : ""
                    : "";
                  const custom = renderCell?.(column, value, row);
                  const text = custom !== undefined ? custom : typeof value === "object" && value !== null ? humanValue(value) : typeof value === "string" && ISO_TIME.test(value) ? humanTimestamp(value) : verbatim && typeof value === "string" && value !== "" ? value : formatCell(value, column);
                  return <td className={cellClass(column, index, tone)} key={column}>{text}</td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {maxPage > 0 && (
        <div className="pagination">
          <button type="button" disabled={currentPage === 0} onClick={() => setPage((value) => Math.max(0, value - 1))}>Previous</button>
          <span>Page {currentPage + 1} of {maxPage + 1}</span>
          <button type="button" disabled={currentPage === maxPage} onClick={() => setPage((value) => Math.min(maxPage, value + 1))}>Next</button>
        </div>
      )}
    </div>
  );
}
