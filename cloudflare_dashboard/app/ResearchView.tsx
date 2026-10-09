"use client";

import { EmptyState, PageIntro } from "./components";

/** Retired page (site audit C1, October 2026): the betting-backtest preview never shipped in this repository and is not part of the golfprice site. Kept registered so old links show this notice. */
export function ResearchView() {
  return (
    <div>
      <PageIntro eyebrow="Retired" title="Betting backtests" description="This page has been retired." />
      <EmptyState title="Betting backtests are no longer on this site" detail="Model experiments live in the research folder; the weekly scorecard and bet results are under Scorecard and P&L." />
      {/* eslint-disable-next-line @next/next/no-html-link-for-pages -- Native navigation avoids the deployed vinext client-router failure. */}
      <p><a className="ex-link" href="/performance">Open Scorecard and P&amp;L</a></p>
    </div>
  );
}
