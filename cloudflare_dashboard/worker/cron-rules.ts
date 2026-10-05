/**
 * Monday settle schedule (owner approval, October 5, 2026). Pure TypeScript, no imports, shared by worker/cron.ts and tests/cron.test.mjs.
 *
 * Cloudflare cron triggers run in UTC, so "Monday 08:30 America/New_York" is 12:30 UTC in summer time (EDT) and 13:30 UTC in winter time (EST): always before
 * production's monday-grading workflow (14:00 UTC), so its results email can carry this week's golfprice scorecard.
 * The Worker is triggered at both 12:30 and 13:30 UTC on Mondays and enqueues only when the New York wall clock reads Monday, 08:30 (the other
 * trigger of the pair reads 07:30 or 09:30 and does nothing).
 */

export const MONDAY_SETTLE_CRONS = ["30 12 * * 1", "30 13 * * 1"] as const;
export const CRON_TIME_ZONE = "America/New_York";
export const CRON_JOB_TYPE = "monday_settle";
export const CRON_REQUESTER = "cron";

export type EtClock = { weekday: string; hour: number; minute: number };

export function easternClock(ms: number): EtClock {
  const parts = new Intl.DateTimeFormat("en-US", { timeZone: CRON_TIME_ZONE, weekday: "short", hour: "2-digit", minute: "2-digit", hourCycle: "h23" }).formatToParts(new Date(ms));
  const pick = (type: string) => parts.find((part) => part.type === type)?.value ?? "";
  return { weekday: pick("weekday"), hour: Number(pick("hour")), minute: Number(pick("minute")) };
}

/** True only at 08:30 on a Monday in New York; the scheduled time may arrive a few seconds late, never early, so minute 30 is exact. */
export function isMondaySettleTime(ms: number): boolean {
  const clock = easternClock(ms);
  return clock.weekday === "Mon" && clock.hour === 8 && clock.minute === 30;
}
