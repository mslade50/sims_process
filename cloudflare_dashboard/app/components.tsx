"use client";

import { useMemo, useState } from "react";
import { Check, ChevronDown, Download, Search, SlidersHorizontal, X } from "lucide-react";
import { DataRow, formatCell, numberValue, titleCase } from "./lib";

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
}: {
  label: string;
  value: string;
  detail?: string;
  tone?: "positive" | "negative" | "neutral" | "accent";
}) {
  return (
    <div className={`kpi kpi-${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      {detail && <small>{detail}</small>}
    </div>
  );
}

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

export function EmptyState({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="empty-state">
      <strong>{title}</strong>
      <span>{detail}</span>
    </div>
  );
}

export function LoadingState({ label = "Loading dashboard data" }: { label?: string }) {
  return (
    <div className="loading-state" aria-live="polite">
      <span className="loading-dot" />
      <span>{label}</span>
    </div>
  );
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

function csvValue(value: unknown): string {
  const text = typeof value === "object" && value !== null ? JSON.stringify(value) : String(value ?? "");
  return `"${text.replaceAll('"', '""')}"`;
}

export function DataTable({
  rows,
  preferredColumns = [],
  label,
  pageSize = 25,
}: {
  rows: DataRow[];
  preferredColumns?: string[];
  label: string;
  pageSize?: number;
}) {
  const allColumns = useMemo(() => {
    const found = [...new Set(rows.flatMap((row) => Object.keys(row)))].filter((column) =>
      rows.some((row) => typeof row[column] !== "object" || row[column] === null),
    );
    return [...preferredColumns.filter((column) => found.includes(column)), ...found.filter((column) => !preferredColumns.includes(column))];
  }, [preferredColumns, rows]);
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<{ column: string; direction: "asc" | "desc" } | null>(null);
  const [page, setPage] = useState(0);
  const [visible, setVisible] = useState<string[]>(() => allColumns.slice(0, Math.min(9, allColumns.length)));

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
                  <label key={column}>
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
                    {titleCase(column)}
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
              {activeColumns.map((column) => (
                <th key={column}>
                  <button type="button" onClick={() => toggleSort(column)}>
                    {titleCase(column)}
                    {sort?.column === column && <span>{sort.direction === "asc" ? " ↑" : " ↓"}</span>}
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {paged.map((row, rowIndex) => (
              <tr key={`${currentPage}-${rowIndex}`}>
                {activeColumns.map((column) => {
                  const value = row[column];
                  const numeric = numberValue(value, Number.NaN);
                  const tone = /edge|units_won|miss_centered/i.test(column)
                    ? numeric > 0
                      ? "positive"
                      : numeric < 0
                        ? "negative"
                        : ""
                    : "";
                  return <td className={tone} key={column}>{formatCell(value, column)}</td>;
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
