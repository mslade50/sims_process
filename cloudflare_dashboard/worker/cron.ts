/**
 * Cron trigger of the dashboard Worker: enqueue the Monday settle job (owner approval, October 5, 2026).
 *
 * Only the scheduled handler can create this job: it is not reachable over HTTP (POST /api/jobs rejects `requested_by` and always uses the
 * verified Access email, and the type allow-list of the HTTP path is unchanged). The job goes through the same queue append and the same
 * same-type rate limit as a phone-triggered job (appendJob in worker/jobs-api.ts), so a settle that is still queued, claimed or running
 * (under 6 h old) blocks a duplicate. The desktop runner picks the job up as usual; nothing is executed here.
 *
 * `verified: true` is set because the runner (golfprice/jobrunner.py) refuses any job without it; the trusted origin is the scheduled
 * handler itself, recorded as requested_by "cron".
 */
import { CRON_JOB_TYPE, CRON_REQUESTER, isMondaySettleTime } from "./cron-rules";
import { appendJob, randomJobSuffix } from "./jobs-api";
import { isoSeconds, makeJobId, type JobRecord } from "../app/jobs-rules";

export interface CronEnv {
  DASHBOARD_DATA?: R2Bucket;
}

export type CronOutcome = { enqueued: boolean; reason: string; job?: JobRecord };

export async function handleScheduled(controller: { scheduledTime: number; cron?: string }, env: CronEnv): Promise<CronOutcome> {
  const now = controller.scheduledTime;
  if (!isMondaySettleTime(now)) return { enqueued: false, reason: "not Monday 09:30 in New York (the other daylight-saving trigger)" };
  if (!env.DASHBOARD_DATA) return { enqueued: false, reason: "dashboard bucket is not bound" };
  const job: JobRecord = { id: makeJobId(now, randomJobSuffix()), type: CRON_JOB_TYPE, params: {}, requested_by: CRON_REQUESTER, requested_at: isoSeconds(now), verified: true };
  try {
    const result = await appendJob(env.DASHBOARD_DATA, job, now);
    return result.ok ? { enqueued: true, reason: "queued", job } : { enqueued: false, reason: result.error };
  } catch (error) {
    return { enqueued: false, reason: error instanceof Error ? error.message : "queue write failed" };
  }
}
