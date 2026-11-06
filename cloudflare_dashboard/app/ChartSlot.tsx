"use client";

import { Suspense, useEffect, useState, type ReactNode } from "react";

/**
 * Mounts a lazy chart after the first paint, inside a box whose height the page's CSS already sets (so nothing moves when it arrives).
 * The headline chart starts about 60 ms after the first paint; with `near`, a chart further down the page waits until it is within 400 px of the screen.
 */
export function ChartSlot({ children, near }: { children: ReactNode; near?: boolean }) {
  const [go, setGo] = useState(false);
  const [box, setBox] = useState<HTMLSpanElement | null>(null);
  useEffect(() => {
    if (!near || typeof IntersectionObserver === "undefined") { const id = window.setTimeout(() => setGo(true), near ? 0 : 60); return () => window.clearTimeout(id); }
    if (!box) return;
    const io = new IntersectionObserver((hits) => { if (hits.some((h) => h.isIntersecting)) { setGo(true); io.disconnect(); } }, { rootMargin: "400px 0px" });
    io.observe(box);
    return () => io.disconnect();
  }, [near, box]);
  return go ? <Suspense fallback={null}>{children}</Suspense> : near ? <span ref={setBox} style={{ display: "block", height: "100%" }} /> : null;
}
