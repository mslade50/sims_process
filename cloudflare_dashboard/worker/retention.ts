/**
 * Daily retention of job-queue objects: the Worker-side entry (rules and runner are in worker/retention-rules.ts so tests can import them without a bundler).
 * OFF unless the R2 object jobs/retention/config.json says {"mode":"dry-run"} or {"mode":"apply"}; no such object exists by default. Runs from the
 * 15-minute cron tick once a day at 03:15 America/New_York, never from HTTP.
 *   dry-run: lists, plans, writes jobs/retention/last_report.json (the full delete list), deletes nothing.
 *   apply:   same, then first writes jobs/retention/applied/<utc time>.json (what is about to go) and deletes at most MAX_DELETES_PER_RUN objects.
 */
export { handleRetention, retentionDue, protectedJobIds } from "./retention-rules";
export type { RetentionEnv, RetentionOutcome } from "./retention-rules";
