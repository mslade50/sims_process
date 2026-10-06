/**
 * Cron trigger of the dashboard Worker: enqueue the jobs the schedule says are due (worker/cron-rules.ts; owner approvals October 5 and 6, 2026).
 *
 * Only the scheduled handler can create these jobs: it is not reachable over HTTP (POST /api/jobs rejects `requested_by` and always uses the
 * verified Access email, and the type allow-list of the HTTP path is unchanged). Every job goes through the same queue append and the same
 * same-type rate limit as a phone-triggered job (appendJob in worker/jobs-api.ts), so a job of that type that is still queued, claimed or running
 * (under 6 h old) blocks a duplicate. The desktop runner picks the jobs up as usual (oldest first); nothing is executed here.
 *
 * `verified: true` is set because the runner (golfprice/jobrunner.py) refuses any job without it; the trusted origin is the scheduled
 * handler itself, recorded as requested_by "cron".
 */
import { CRON_REQUESTER, dueJobs } from "./cron-rules";
import { appendJob, randomJobSuffix } from "./jobs-api";
import { isoSeconds, makeJobId, specFor, type JobRecord } from "../app/jobs-rules";

export interface CronEnv {
  DASHBOARD_DATA?: R2Bucket;
}

export type CronOutcome = { enqueued: boolean; reason: string; job?: JobRecord; jobs: JobRecord[]; skipped: { type: string; reason: string }[] };

export async function handleScheduled(controller: { scheduledTime: number; cron?: string }, env: CronEnv): Promise<CronOutcome> {
  const now = controller.scheduledTime;
  const due = dueJobs(now);
  if (due.length === 0) return { enqueued: false, reason: "nothing is due at this New York time", jobs: [], skipped: [] };
  if (!env.DASHBOARD_DATA) return { enqueued: false, reason: "dashboard bucket is not bound", jobs: [], skipped: [] };
  const jobs: JobRecord[] = [];
  const skipped: { type: string; reason: string }[] = [];
  for (const item of due) {
    if (!specFor(item.type)) {
      skipped.push({ type: item.type, reason: "not in the job allow-list" });
      continue;
    }
    const job: JobRecord = { id: makeJobId(now, randomJobSuffix()), type: item.type, params: {}, requested_by: CRON_REQUESTER, requested_at: isoSeconds(now), verified: true };
    try {
      const result = await appendJob(env.DASHBOARD_DATA, job, now);
      if (result.ok) jobs.push(job);
      else skipped.push({ type: item.type, reason: result.error });
    } catch (error) {
      skipped.push({ type: item.type, reason: error instanceof Error ? error.message : "queue write failed" });
    }
  }
  return { enqueued: jobs.length > 0, reason: jobs.length ? "queued" : (skipped[0]?.reason ?? "nothing queued"), job: jobs[0], jobs, skipped };
}
