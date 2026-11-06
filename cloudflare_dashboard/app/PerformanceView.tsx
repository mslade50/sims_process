"use client";

import { lazy, Suspense, useState } from "react";
import { PageHeader, Section } from "./ui";
import "./performance.css";
import { LegacyNotice } from "./components";
import { GolfpriceScorecardSection } from "./ScorecardSection";

// The retired pipeline's bet results (and the Recharts they draw with, about 100 KB gzip) load only when the visitor opens them.
const LegacyBetResults = lazy(() => import("./views").then((m) => ({ default: m.LegacyBetResults })));

export function PerformanceView() {
  const [showLegacy, setShowLegacy] = useState(false);
  return (
    <>
      <PageHeader title="Scorecard and P&L" purpose="How the current model is doing against the market's closing prices, in log-loss (lower is better) and betting units." />
      <GolfpriceScorecardSection />
      <Section id="older" title="Older bet results" purpose="From the retired simulation pipeline; kept for reference, not the current model's bets." actions={<button type="button" className="icon-button" aria-expanded={showLegacy} onClick={() => setShowLegacy((value) => !value)}>{showLegacy ? "Hide older results" : "Show older results"}</button>}>
        <LegacyNotice>These bet results come from the retired simulation pipeline. They are kept for reference and do not include the current model&apos;s bets.</LegacyNotice>
      </Section>
      {showLegacy && <Suspense fallback={null}><LegacyBetResults /></Suspense>}
    </>
  );
}
