// Contrast audit: loads every route in both themes (phone + desktop) in headless Chrome through visual-check's CDP harness and measures the real
// computed contrast of ALL visible text (body, muted, disabled controls, SVG chart labels) against its effective background (WCAG AA: 4.5, large 3).
//
//   node scripts/contrast-check.mjs [--base http://localhost:3111] [--routes a,b] [--extra] [--out dist-visual/contrast]
// Prints failure counts per route/theme/viewport and the worst offenders; exit code 1 when any AA failure remains.
import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(fileURLToPath(new URL("..", import.meta.url)));
const args = process.argv.slice(2);
const outIdx = args.indexOf("--out");
const out = resolve(ROOT, outIdx >= 0 ? args[outIdx + 1] : "dist-visual/contrast");
const passthrough = outIdx >= 0 ? args.filter((_, i) => i !== outIdx && i !== outIdx + 1) : args;
spawnSync(process.execPath, [join(ROOT, "scripts", "visual-check.mjs"), "--contrast-strict", "--no-shots", "--out", out, ...passthrough], { cwd: ROOT, stdio: "inherit" });

const report = JSON.parse(readFileSync(join(out, "report.json"), "utf8"));
let total = 0, judged = 0;
const worst = [];
for (const c of report.cases) {
  judged += c.stats?.contrastJudged ?? 0;
  const bad = c.issues.filter((i) => i.check === "contrast");
  total += bad.length;
  if (bad.length) console.log(`  ${c.route.padEnd(20)} ${c.viewport.padEnd(8)} ${c.theme.padEnd(5)} ${bad.length} failing colour group(s)`);
  for (const i of bad) worst.push(`${c.route} ${c.viewport}/${c.theme}: ${i.message} [${i.selector ?? ""}]`);
}
console.log(`\nContrast: ${judged} text elements judged across ${report.cases.length} cases, ${total} failing colour group(s)`);
worst.slice(0, 40).forEach((w) => console.log(`  ${w}`));
process.exit(total ? 1 : 0);
