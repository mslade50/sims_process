"use client";

import type { ReactNode } from "react";
import { Chip, type Tone } from "./ui";
import "./change-tray.css";

/**
 * One change tray for an editing page (Weather edits, Inputs > Adjust): what you changed (original -> edited), where each change stands
 * (edited on the page -> saved -> applied to prices), whether a reprice can be run, the actions, and a live region that announces the outcome.
 * Saving and repricing are separate stages and never share a word: "Saved" means kept for later runs, "Applied" means a run priced with it.
 */
export type TrayChange = { key: string; what: string; from: string; to: string; unit?: string };
export type TrayStage = { label: string; tone: Tone; value: string; detail: string };
export type TrayAnnounce = { tone: "ok" | "error"; text: string; link?: { href: string; label: string } } | null;

const MAX_ROWS = 6;

export function ChangeTray({ title = "Your changes", changes, stages, eligibility, announce, children, nothing = "Nothing changed yet." }: {
  title?: string;
  changes: TrayChange[];
  stages: TrayStage[];
  eligibility?: { ok: boolean; text: string; label?: string };
  announce: TrayAnnounce;
  children: ReactNode;
  nothing?: string;
}) {
  const shown = changes.slice(0, MAX_ROWS);
  const more = changes.slice(MAX_ROWS);
  const row = (c: TrayChange) => (
    <tr key={c.key}>
      <th scope="row">{c.what}</th>
      <td className="num">{c.from}</td>
      <td className="ct-arrow" aria-hidden="true">→</td>
      <td className="num ct-to">{c.to}</td>
      <td className="ct-unit">{c.unit ?? ""}</td>
    </tr>
  );
  return (
    <section className="ct" aria-label={title}>
      <h3 className="ct-title">{title}</h3>
      <ol className="ct-stages" aria-label="Where your changes stand">
        {stages.map((stage, i) => (
          <li key={stage.label} className="ct-stage">
            <span className="ct-stage-head"><span className="ct-stage-n" aria-hidden="true">{i + 1}</span><strong>{stage.label}</strong><Chip tone={stage.tone}>{stage.value}</Chip></span>
            <small>{stage.detail}</small>
          </li>
        ))}
      </ol>
      {changes.length === 0 ? <p className="ct-none">{nothing}</p> : (
        <div className="ct-changes">
          <table>
            <caption className="ct-sr">Original and edited values</caption>
            <thead><tr><th scope="col">Change</th><th scope="col">Original</th><th scope="col" className="ct-sr">to</th><th scope="col">Edited</th><th scope="col" className="ct-sr">Unit</th></tr></thead>
            <tbody>{shown.map(row)}</tbody>
          </table>
          {more.length > 0 && (
            <details className="ct-more"><summary>and {more.length} more</summary><table><tbody>{more.map(row)}</tbody></table></details>
          )}
        </div>
      )}
      {eligibility && <p className="ct-elig"><Chip tone={eligibility.ok ? "positive" : "neutral"}>{eligibility.label ?? (eligibility.ok ? "Reprice available" : "Reprice unavailable")}</Chip><span>{eligibility.text}</span></p>}
      <div className="ct-actions" role="group" aria-label="Save and reprice actions">{children}</div>
      <div className="ct-live" aria-live="polite" aria-atomic="true">
        {announce && (
          <div className={`ct-msg ${announce.tone}`} role={announce.tone === "error" ? "alert" : undefined}>
            <strong>{announce.tone === "error" ? "Not done." : "Done."}</strong> <span>{announce.text}</span>
            {announce.link && <> <a className="ex-link" href={announce.link.href}>{announce.link.label}</a></>}
          </div>
        )}
      </div>
    </section>
  );
}
