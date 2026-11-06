"use client";

import { useEffect, useState, type ReactNode } from "react";
import { Section } from "./ui";

/**
 * A page section for the scoring / history pages: anchored heading + one-line purpose, no card.
 * `fold` marks a secondary section: open on desktop, folded behind its heading on a phone so the page stays short.
 */
export function Block({ id, title, eyebrow, actions, fold, children }: { id: string; title: string; eyebrow?: string; actions?: ReactNode; fold?: boolean; children: ReactNode }) {
  const [open, setOpen] = useState(!fold);
  useEffect(() => {
    if (!fold) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect -- one-time read of the viewport after mount (server markup is open)
    try { setOpen(window.matchMedia("(min-width: 721px)").matches); } catch { setOpen(true); }
  }, [fold]);
  if (!fold) return <Section id={id} title={title} purpose={eyebrow} actions={actions}>{children}</Section>;
  return <Section id={id} title={title} purpose={eyebrow} actions={actions}>
    <details className="score-fold" open={open} onToggle={(e) => setOpen((e.currentTarget as HTMLDetailsElement).open)}>
      <summary>{open ? "Hide details" : "Show details"}</summary>
      {children}
    </details>
  </Section>;
}

/** A folded group inside a section: open on desktop, closed on a phone. */
export function Fold({ summary, children }: { summary: string; children: ReactNode }) {
  const [open, setOpen] = useState(true);
  // eslint-disable-next-line react-hooks/set-state-in-effect -- one-time read of the viewport after mount
  useEffect(() => { try { setOpen(window.matchMedia("(min-width: 721px)").matches); } catch { setOpen(true); } }, []);
  return <details className="score-details score-fold-inner" open={open} onToggle={(e) => setOpen((e.currentTarget as HTMLDetailsElement).open)}><summary>{summary}</summary>{children}</details>;
}
