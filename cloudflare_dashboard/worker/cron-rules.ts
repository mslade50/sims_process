/**
 * Automatic job schedule (owner approvals: Monday settle October 5, 2026; every standard moment, the odds-move reprice and the input watch October 6, 2026).
 * Pure TypeScript, no imports, shared by worker/cron.ts, vite.config.ts and tests/cron.test.mjs.
 *
 * Cloudflare cron triggers run in UTC only. One trigger fires every 15 minutes (step expression on the minute field); the handler reads the America/New_York wall clock
 * (Intl, so daylight saving is handled by the time zone database, never by a UTC offset) and enqueues whatever the table below says is due at that
 * minute. There is no per-season pair of triggers to keep in step: the 08:30 ET Monday settle stays before production's monday-grading workflow
 * (14:00 UTC = 10:00 / 09:00 ET) in both EDT and EST.
 *
 * The fixed slots are for a normal US-timezone week. Events whose tee times differ (Asia, Europe, Pacific) are covered by the adaptive `watch` job, which
 * runs the pre-tee close from the real first tee, and by the round gate of `after_round`, which prices only a round that is actually complete.
 * Every enqueue goes through the same same-type rate limit as a phone request (appendJob): a type that is already queued or running (under 6 h old) is
 * never queued again, so a missed or slow slot never piles up duplicates.
 */

export const SCHEDULE_CRONS = ["*/15 * * * *"] as const;
export const CRON_TIME_ZONE = "America/New_York";
export const CRON_REQUESTER = "cron";

export type Day = "Mon" | "Tue" | "Wed" | "Thu" | "Fri" | "Sat" | "Sun";

export type Slot = { type: string; days: Day[]; times: string[]; note: string };
export type Window = { type: string; days: Day[]; from: string; to: string; minutes: number[]; note: string };

/** Fixed Eastern-time slots (HH:MM, 24 h). Order = enqueue order inside one tick. */
export const SLOTS: Slot[] = [
  { type: "monday_settle", days: ["Mon"], times: ["08:30", "11:30", "13:30"], note: "Settle last week at 08:30 (before production grading); later slots retry while the event is not final (a pending settle writes nothing)." },
  { type: "monday_week", days: ["Mon"], times: ["15:30"], note: "Price the new week once the settle has finished." },
  { type: "tuesday", days: ["Tue"], times: ["12:00"], note: "Tuesday refresh on the first full day of market data." },
  { type: "wednesday", days: ["Wed"], times: ["18:00", "22:30"], note: "Reprice with the tee-time gate; the 22:30 slot catches late tee sheets and the early European tee." },
  { type: "daily_health", days: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], times: ["07:30"], note: "Daily health read-back; resolved failures and missed runs clear once a later successful run covers them." },
  { type: "thursday", days: ["Thu"], times: ["05:30"], note: "Final pre-tee pricing run (after the first tee it moves its as-of back to the last pre-tee snapshot, by design)." },
  { type: "thursday_close", days: ["Thu"], times: ["06:30"], note: "Safety net only: closes every event that has not teed off and has no close yet (per event). The watch job normally runs each close 30 minutes before that event's real first tee." },
  { type: "after_round", days: ["Thu", "Fri", "Sat"], times: ["14:30", "20:30", "22:30"], note: "DP World Tour rounds finish about 14:00 ET, PGA east-coast about 19:00-20:30, west of the Mississippi about 22:00. Prices only a complete round; otherwise a no-op." },
  { type: "after_round", days: ["Fri", "Sat", "Sun"], times: ["01:00"], note: "Straggler slot after the previous evening's round (weather delays, playoffs, late Pacific finishes)." },
  { type: "after_round", days: ["Sun"], times: ["14:30"], note: "Straggler slot for a round 3 finished late (a no-op if round 3 is already priced; the event's round 4 is settled on Monday)." },
];

/** Repeating windows (ET). `minutes` are the minutes past the hour. The watch job goes first so a re-simulation it starts is the base the odds reprice then reads. */
export const WINDOWS: Window[] = [
  { type: "watch", days: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], from: "00:00", to: "23:59", minutes: [0, 15, 30, 45], note: "Every 15 minutes, all week (owner 2026-10-07, event weeks run by themselves): input changes, the per-event pre-tee close (30 minutes before each event's own first tee, any time zone), and the after-round pricing of each event as soon as the feed shows the round complete. Cheap when nothing is due; does nothing when no event is active." },
  { type: "odds_reprice", days: ["Mon"], from: "16:00", to: "23:59", minutes: [0, 30], note: "No re-simulation: re-read odds, recompute signals against the existing fairs, publish, alert." },
  { type: "odds_reprice", days: ["Tue", "Wed"], from: "00:00", to: "23:59", minutes: [0, 30], note: "Overnight and in the betting window." },
  { type: "odds_reprice", days: ["Thu"], from: "00:00", to: "11:00", minutes: [0, 30], note: "Until the first tee (the job does nothing once every event has teed off)." },
];

export type EtClock = { weekday: Day; hour: number; minute: number };

export function easternClock(ms: number): EtClock {
  const parts = new Intl.DateTimeFormat("en-US", { timeZone: CRON_TIME_ZONE, weekday: "short", hour: "2-digit", minute: "2-digit", hourCycle: "h23" }).formatToParts(new Date(ms));
  const pick = (type: string) => parts.find((part) => part.type === type)?.value ?? "";
  return { weekday: pick("weekday") as Day, hour: Number(pick("hour")), minute: Number(pick("minute")) };
}

export const hhmm = (clock: EtClock) => `${String(clock.hour).padStart(2, "0")}:${String(clock.minute).padStart(2, "0")}`;

export type DueJob = { type: string; why: string };

/** What the schedule enqueues at this instant. The scheduled time may arrive a few seconds late, never early, so the minute is exact. */
export function dueJobs(ms: number): DueJob[] {
  const clock = easternClock(ms);
  const at = hhmm(clock);
  const out: DueJob[] = [];
  for (const slot of SLOTS) {
    if (slot.days.includes(clock.weekday) && slot.times.includes(at)) out.push({ type: slot.type, why: `${slot.type} ${clock.weekday} ${at} ET` });
  }
  for (const win of WINDOWS) {
    if (win.days.includes(clock.weekday) && win.minutes.includes(clock.minute) && at >= win.from && at <= win.to) {
      out.push({ type: win.type, why: `${win.type} ${clock.weekday} ${at} ET (window ${win.from}-${win.to})` });
    }
  }
  return out;
}

/** True only at 08:30 on a Monday in New York (kept for the settle tests). */
export function isMondaySettleTime(ms: number): boolean {
  const clock = easternClock(ms);
  return clock.weekday === "Mon" && hhmm(clock) === "08:30";
}
