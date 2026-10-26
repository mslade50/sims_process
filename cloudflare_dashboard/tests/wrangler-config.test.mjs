import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { SCHEDULE_CRONS } from "../worker/cron-rules.ts";

// wrangler.jsonc is the single source of truth for the deployed Worker (npm run deploy:cloudflare and the GitHub workflow).
const strip = (s) => s.replace(/^\s*\/\/.*$/gm, "");
const cfg = JSON.parse(strip(readFileSync(new URL("../wrangler.jsonc", import.meta.url), "utf8")));

test("cron triggers match SCHEDULE_CRONS", () => {
  assert.deepEqual(cfg.triggers?.crons, [...SCHEDULE_CRONS]);
});
test("R2 binding and Access vars are declared", () => {
  assert.deepEqual(cfg.r2_buckets, [{ binding: "DASHBOARD_DATA", bucket_name: "golf-model-dashboard-data" }]);
  assert.ok(cfg.vars?.ACCESS_TEAM_DOMAIN && cfg.vars?.ACCESS_AUD);
});
test("feature switches are declared and off", () => {
  assert.equal(cfg.vars?.LEGACY_ALERTS_ENABLED, "0");
  assert.equal(cfg.vars?.OPS_SHADOW_ENABLED, "0");
});
test("owner-only gate is on in the deployed config", () => {
  const owners = String(cfg.vars?.OWNER_EMAILS ?? "").split(",").map((e) => e.trim()).filter(Boolean);
  assert.ok(owners.length > 0, "OWNER_EMAILS must name at least one owner, or Run and overrides open to every Access user");
  for (const email of owners) assert.match(email, /^[^@\s]+@[^@\s]+$/);
});
