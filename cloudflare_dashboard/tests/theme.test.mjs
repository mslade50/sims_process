import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { ACCENTS, catalogFreshnessTone, countUpText, freshnessTone, parseNumericText, parseTimestamp, relativeAge, themeBootScript } from "../app/ui-rules.ts";

const css = await readFile(new URL("../app/tokens.css", import.meta.url), "utf8");
const globals = await readFile(new URL("../app/globals.css", import.meta.url), "utf8");

/** Returns the declarations inside the first block that starts with `opener`. */
function block(opener) {
  const start = css.indexOf(opener);
  assert.ok(start >= 0, `missing block ${opener}`);
  const open = css.indexOf("{", start);
  let depth = 0;
  for (let i = open; i < css.length; i += 1) {
    if (css[i] === "{") depth += 1;
    if (css[i] === "}" && --depth === 0) return css.slice(open + 1, i);
  }
  throw new Error(`unterminated ${opener}`);
}
function tokens(text) {
  const out = {};
  for (const match of text.matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g)) out[match[1]] = match[2].trim();
  return out;
}

const dark = tokens(block(":root {"));
const lightMedia = tokens(block(":root:not([data-theme=\"dark\"])"));
const lightForced = tokens(block(":root[data-theme=\"light\"]"));
const light = { ...dark, ...lightForced };

function channel(hex) {
  const value = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255).map((v) => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
  return 0.2126 * value[0] + 0.7152 * value[1] + 0.0722 * value[2];
}
function contrast(a, b) {
  const [x, y] = [channel(a), channel(b)];
  return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05);
}
function resolve(theme, name, accent) {
  let value = theme[name];
  if (value === "var(--accent-d)") value = accent.d;
  if (value === "var(--accent-l)") value = accent.l;
  if (value?.startsWith("var(")) value = resolve(theme, value.slice(4, -1), accent);
  assert.match(value ?? "", /^#[0-9a-f]{6}$/i, `${name} should resolve to a hex color`);
  return value;
}

const SURFACES = ["--background", "--surface", "--surface-raised", "--surface-soft"];
const TEXT = ["--foreground", "--muted", "--muted-strong", "--positive", "--negative", "--warning", "--model", "--market", "--wave-am", "--wave-pm", "--wx-wind", "--wx-gust", "--wx-temp", "--wx-rain", "--accent"];
const CHARTS = Array.from({ length: 8 }, (_, i) => `--chart-${i + 1}`);

test("light theme blocks define the same tokens and the dark theme is complete", () => {
  assert.deepEqual(lightMedia, lightForced, "prefers-color-scheme light and data-theme=light must match");
  for (const name of Object.keys(lightForced)) assert.ok(name in dark, `${name} is overridden for light but not defined for dark`);
  for (const name of [...SURFACES, ...TEXT, ...CHARTS, "--accent-ink", "--wave-am-fill", "--wave-pm-fill", "--focus-ring"]) assert.ok(name in dark, `${name} missing`);
});

for (const [themeName, theme] of [["dark", dark], ["light", light]]) {
  test(`${themeName}: text and semantic colors reach 4.5:1 on every surface, for every accent`, () => {
    for (const accent of ACCENTS) {
      for (const name of TEXT) {
        for (const surface of SURFACES) {
          const ratio = contrast(resolve(theme, name, accent), resolve(theme, surface, accent));
          assert.ok(ratio >= 4.5, `${themeName} ${name} on ${surface} (accent ${accent.key}) is ${ratio.toFixed(2)}:1`);
        }
      }
      const ink = contrast(resolve(theme, "--accent-ink", accent), resolve(theme, "--accent", accent));
      assert.ok(ink >= 4.5, `${themeName} accent ${accent.key} button text is ${ink.toFixed(2)}:1`);
    }
  });
  test(`${themeName}: chart palette is 3:1 against the card surface and distinct`, () => {
    const colors = CHARTS.map((name) => resolve(theme, name, ACCENTS[0]));
    for (const [index, color] of colors.entries()) assert.ok(contrast(color, resolve(theme, "--surface", ACCENTS[0])) >= 3, `${CHARTS[index]} ${color} below 3:1`);
    assert.equal(new Set(colors).size, colors.length);
  });
}

test("stylesheet has no hard-coded hex colors (tokens only) and honors reduced motion", () => {
  assert.deepEqual([...globals.matchAll(/#[0-9a-fA-F]{3,8}\b/g)].map((m) => m[0]).filter((c) => !["#07110e", "#f3f1e8"].includes(c.toLowerCase())), []);
  assert.match(globals, /@media \(prefers-reduced-motion: reduce\)/);
  assert.match(globals, /:focus-visible/);
});

test("count-up text keeps prefix, suffix, decimals and grouping", () => {
  assert.equal(countUpText("$1,234.5", 1), "$1,234.5");
  assert.equal(countUpText("$1,234.5", 0), "$0.0");
  assert.equal(countUpText("12.3%", 0), "12.3%".replace("12.3", "0.0"));
  assert.equal(countUpText("+4.2", 0), "+0.0");
  assert.equal(countUpText("-8.0", 0), "0.0");
  assert.match(countUpText("-8.0", 0.5), /^−4\.\d$|^−6\.\d$|^−7\.\d$/);
  assert.equal(countUpText("n/a", 0.3), "n/a");
  assert.equal(parseNumericText("123 bets")?.suffix, " bets");
  assert.equal(parseNumericText("Even"), null);
});

test("freshness helpers", () => {
  assert.equal(parseTimestamp("2026-10-05T14-50-00-055214Z"), Date.parse("2026-10-05T14:50:00Z"));
  assert.equal(parseTimestamp("2026-10-05T14:50:00Z"), Date.parse("2026-10-05T14:50:00Z"));
  assert.equal(parseTimestamp("not a date"), null);
  assert.equal(relativeAge(10_000), "just now");
  assert.equal(relativeAge(12 * 60_000), "12 min ago");
  assert.equal(relativeAge(3 * 3_600_000), "3 h ago");
  assert.equal(relativeAge(3 * 86_400_000), "3 d ago");
  assert.deepEqual([null, 30 * 60_000, 5 * 3_600_000, 40 * 3_600_000, 100 * 3_600_000].map(freshnessTone), ["unknown", "live", "fresh", "aging", "stale"]);
});

test("theme boot script restores saved accent and theme before paint", () => {
  const script = themeBootScript();
  assert.match(script, /golf-dashboard-theme/);
  for (const accent of ACCENTS) assert.ok(script.includes(accent.d) && script.includes(accent.l));
  assert.doesNotThrow(() => new Function(script));
});

test("catalog freshness: fresh through 7 days, amber beyond, unknown without a date", () => {
  const day = 86_400_000;
  assert.deepEqual([null, 1000, 6.9 * day, 7 * day, 7.1 * day, 40 * day].map(catalogFreshnessTone), ["unknown", "fresh", "fresh", "fresh", "aging", "aging"]);
});
