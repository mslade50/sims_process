// Pure-rule tests for scripts/visual-rules.mjs. No browser needed: node --test tests/visual-rules.test.mjs
import assert from "node:assert/strict";
import test from "node:test";
import {
  parseColor, contrastRatio, evaluateContrast, effectiveBackground, composite, isLargeText, requiredContrast,
  intersection, meaningfulOverlap, clipTo, tapTargetIssues, classifyRequest, matchesAny, OVERLAP_ALLOW, severityOf, summarize,
} from "../scripts/visual-rules.mjs";

test("box intersection", () => {
  assert.deepEqual(intersection({ x: 0, y: 0, w: 10, h: 10 }, { x: 5, y: 5, w: 10, h: 10 }), { x: 5, y: 5, w: 5, h: 5 });
  assert.equal(intersection({ x: 0, y: 0, w: 10, h: 10 }, { x: 10, y: 0, w: 5, h: 5 }), null, "touching edges do not intersect");
  assert.equal(clipTo({ x: 0, y: 0, w: 10, h: 10 }, [{ x: 20, y: 0, w: 5, h: 5 }]), null);
});

test("meaningful overlap ignores hairlines and slivers", () => {
  const a = { x: 0, y: 0, w: 100, h: 16 };
  assert.ok(meaningfulOverlap(a, { x: 50, y: 4, w: 100, h: 16 }), "half-overlapping text boxes");
  assert.equal(meaningfulOverlap(a, { x: 0, y: 15, w: 100, h: 16 }), null, "1px line-height bleed");
  assert.equal(meaningfulOverlap(a, { x: 99, y: 0, w: 50, h: 16 }), null, "1px horizontal graze");
  assert.equal(meaningfulOverlap(a, { x: 0, y: 40, w: 100, h: 16 }), null, "disjoint");
});

test("colour parsing", () => {
  assert.deepEqual(parseColor("rgb(10, 20, 30)"), { r: 10, g: 20, b: 30, a: 1 });
  assert.deepEqual(parseColor("rgba(10, 20, 30, 0.5)"), { r: 10, g: 20, b: 30, a: 0.5 });
  assert.deepEqual(parseColor("rgb(10 20 30 / 25%)"), { r: 10, g: 20, b: 30, a: 0.25 });
  assert.deepEqual(parseColor("#fff"), { r: 255, g: 255, b: 255, a: 1 });
  assert.equal(parseColor("transparent").a, 0);
  const c = parseColor("color(srgb 1 0.5 0 / 0.5)");
  assert.equal(c.r, 255); assert.equal(c.g, 127.5); assert.equal(c.a, 0.5);
  const white = parseColor("oklab(1 0 0)");
  assert.ok(white.r > 254 && white.g > 254 && white.b > 254);
  const black = parseColor("oklch(0 0 0)");
  assert.ok(black.r < 1 && black.g < 1 && black.b < 1);
  assert.equal(parseColor("var(--x)"), null);
});

test("contrast math matches WCAG reference values", () => {
  const black = { r: 0, g: 0, b: 0, a: 1 }, white = { r: 255, g: 255, b: 255, a: 1 };
  assert.ok(Math.abs(contrastRatio(black, white) - 21) < 1e-9);
  assert.ok(Math.abs(contrastRatio(white, white) - 1) < 1e-9);
  assert.ok(Math.abs(contrastRatio(parseColor("#767676"), white) - 4.54) < 0.02, "#767676 on white is the classic AA threshold");
});

test("large text thresholds", () => {
  assert.equal(isLargeText(24, 400), true);
  assert.equal(isLargeText(19, 700), true);
  assert.equal(isLargeText(19, 400), false);
  assert.equal(requiredContrast(14, 400), 4.5);
  assert.equal(requiredContrast(24, 400), 3);
});

test("effective background walks ancestors and composites translucent layers", () => {
  const bg = effectiveBackground(["rgba(0,0,0,0)", "rgba(255,0,0,0.5)", "rgb(0,0,255)"]);
  assert.deepEqual(bg, composite({ r: 255, g: 0, b: 0, a: 0.5 }, { r: 0, g: 0, b: 255, a: 1 }));
  assert.deepEqual(effectiveBackground(["transparent"], { r: 9, g: 9, b: 9, a: 1 }), { r: 9, g: 9, b: 9, a: 1 }, "falls back to the page colour");
  assert.equal(effectiveBackground([{ unknown: true }]), null, "gradients and images cannot be judged");
});

test("evaluateContrast flags grey on white and passes dark on white; opacity counts", () => {
  const base = { r: 255, g: 255, b: 255, a: 1 };
  assert.equal(evaluateContrast({ color: "rgb(170,170,170)", chain: [], fontPx: 14, weight: 400 }, base).pass, false);
  assert.equal(evaluateContrast({ color: "rgb(20,20,20)", chain: [], fontPx: 14, weight: 400 }, base).pass, true);
  assert.equal(evaluateContrast({ color: "rgb(20,20,20)", chain: [], fontPx: 14, weight: 400, opacity: 0.3 }, base).pass, false);
  assert.equal(evaluateContrast({ color: "rgb(130,130,130)", chain: [], fontPx: 28, weight: 400 }, base).pass, true, "large text needs only 3:1");
  assert.equal(evaluateContrast({ color: "var(--x)", chain: [], fontPx: 14, weight: 400 }, base), null);
});

test("allowlists and tap targets", () => {
  assert.equal(matchesAny("div.recharts-tooltip-wrapper", OVERLAP_ALLOW), true);
  assert.equal(matchesAny("div.card", OVERLAP_ALLOW), false);
  const r = tapTargetIssues([{ w: 44, h: 44 }, { w: 20, h: 30, sel: "a" }, { w: 40, h: 39, sel: "b" }]);
  assert.equal(r.count, 2); assert.equal(r.worst[0].sel, "a"); assert.equal(r.total, 3);
});

test("request classification", () => {
  const base = "http://localhost:3111";
  assert.equal(classifyRequest({ url: `${base}/api/data/x`, status: 200, base }), "ok");
  assert.equal(classifyRequest({ url: `${base}/api/data/x`, status: 304, base }), "ok");
  assert.equal(classifyRequest({ url: `${base}/api/data/x`, status: 404, base }), "known404");
  assert.equal(classifyRequest({ url: `${base}/assets/a.js`, status: 404, base }), "failed");
  assert.equal(classifyRequest({ url: `${base}/api/data/x`, status: 500, base }), "failed");
  assert.equal(classifyRequest({ url: "https://other.example/x", status: 500, base }), "ignore");
});

test("severity and summary: only errors fail", () => {
  assert.equal(severityOf("text-overlap"), "error");
  assert.equal(severityOf("tap-target"), "warn");
  assert.equal(summarize([{ issues: [{ severity: "warn" }] }]).failed, false);
  assert.equal(summarize([{ issues: [{ severity: "error" }, { severity: "warn" }] }]).failed, true);
});
