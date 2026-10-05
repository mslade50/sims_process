"use client";

import { EmptyState, PageIntro } from "./components";

/**
 * Placeholder. DashboardApp.tsx at main (c9c0ddd) imports ./ResearchView, but the file was never committed to the repository, so a clean
 * checkout cannot build. The Betting backtests view (a local research preview that talks to /api/backtest on 127.0.0.1:8766) needs to be
 * restored from the machine where it was written; replace this file with it. Nothing else depends on this stub.
 */
export function ResearchView() {
  return (
    <div>
      <PageIntro eyebrow="Review" title="Betting backtests" description="Model experiments and betting results." />
      <EmptyState title="Backtest view not packaged in this build" detail="The ResearchView source was missing from the repository when the Model inputs area was added. Restore app/ResearchView.tsx to bring it back." />
    </div>
  );
}
