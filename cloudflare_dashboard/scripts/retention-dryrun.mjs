// READ-ONLY retention dry run against the live bucket through wrangler's own login (remote binding; list and get only, no put/delete).
//   node --experimental-strip-types scripts/retention-dryrun.mjs <wrangler-config.jsonc with a remote R2 binding named B> <out.json> [days]
// Output: the plan from worker/retention-rules.ts with the full delete list. Nothing is written to R2.
import { writeFileSync } from "node:fs";
import { getPlatformProxy } from "wrangler";
import { classify, planRetention } from "../worker/retention-rules.ts";
import { protectedJobIds } from "../worker/retention-rules.ts";

const [configPath, outPath, days = "7"] = process.argv.slice(2);
if (!configPath || !outPath) throw new Error("usage: retention-dryrun.mjs <config> <out.json> [days]");
const proxy = await getPlatformProxy({ configPath, persist: false });
try {
  const bucket = proxy.env.B;
  const all = [];
  const inventory = {};
  for (const prefix of ["jobs/"]) {
    let cursor;
    do {
      const res = await bucket.list({ prefix, limit: 1000, cursor });
      for (const o of res.objects) all.push({ key: o.key, size: o.size, uploaded: o.uploaded.getTime() });
      cursor = res.truncated ? res.cursor : undefined;
    } while (cursor);
  }
  for (const o of all) {
    const top = o.key.split("/").slice(0, 2).join("/") + (o.key.split("/").length > 2 ? "/" : "");
    inventory[top] ??= { count: 0, bytes: 0 };
    inventory[top].count += 1;
    inventory[top].bytes += o.size;
  }
  const byDay = {};
  for (const o of all.filter((x) => classify(x.key))) {
    const d = new Date(o.uploaded).toISOString().slice(0, 10);
    byDay[d] ??= { count: 0, bytes: 0 };
    byDay[d].count += 1;
    byDay[d].bytes += o.size;
  }
  const classified = all.filter((o) => classify(o.key));
  const plan = planRetention(classified, await protectedJobIds(bucket), Date.now(), Number(days));
  writeFileSync(outPath, JSON.stringify({ read_only: true, listed_objects_under_jobs: all.length, inventory_by_prefix: inventory, classified_by_upload_day: byDay, ...plan, delete_keys: plan.delete.map((o) => o.key) }, null, 1));
  console.log(JSON.stringify({ listed: all.length, inventory, summary: plan.summary, kinds: plan.kinds }, null, 1));
} finally {
  await proxy.dispose();
}
