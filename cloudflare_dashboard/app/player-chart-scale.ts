"use client";

import { useEffect, useState } from "react";

/**
 * Chart text that stays readable when a fixed-size SVG is scaled down to a phone. An SVG drawn on a 500-unit canvas and shown 340 px wide
 * renders its 12-unit labels at 8 px. `useSvgFont(viewW)` measures the drawn width and returns the font size, in canvas units, that
 * comes out at `min` real pixels (never smaller than `min`). Attach `ref` to the <svg>; use `fs` for every text element.
 */
export function useSvgFont(viewW: number, min = 12): { fs: number; ref: (node: SVGSVGElement | null) => void } {
  const [node, setNode] = useState<SVGSVGElement | null>(null);
  const [fs, setFs] = useState(min);
  useEffect(() => {
    if (!node) return;
    const read = () => {
      const w = node.getBoundingClientRect().width;
      if (w > 40) setFs(Math.max(min, Math.round((min * viewW) / w * 10) / 10));
    };
    read();
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(read);
    ro.observe(node);
    return () => ro.disconnect();
  }, [node, viewW, min]);
  return { fs, ref: setNode };
}
