# Passive monitoring increment (B06)

This increment reconciles PR31 with the current dashboard. It is a subset of B06, not completed rollout acceptance. The Weekly workflow's output receipts remain separate from passive monitoring findings.

## Switches and scope

Both committed switches remain `"0"`: `LEGACY_ALERTS_ENABLED` and `OPS_SHADOW_ENABLED`. Deploying these bytes does not enable shadow recording or legacy Telegram sends. Legacy sending requires the exact `"1"` value and both existing Telegram secrets; secrets alone must not enable it. No new outcome, escalation, email or repair sender is implemented here.

When separately enabled, the existing Worker cron passively reads queue/status/heartbeat/role objects and writes only `ops/shadow/*`. It creates no jobs, changes no runner role and sends no messages. Existing product delivery and pricing remain unchanged.

The observer validates job IDs, supported job types/states, producer timestamps, status identity and heartbeat machine identity. Malformed/future evidence degrades coverage rather than becoming fresh/queued/absent. Cancellation chronology follows the record that establishes cancellation. Only a cron-origin request in the scheduler's scheduled minute establishes scheduled creation; same-minute manual request ordering remains uncertain. A definite nonterminal blocker must have pre-slot publication evidence and remain rate-limit eligible throughout the scheduled minute. Partial-minute eligibility or uncertain visibility degrades coverage. For claimed/running states, job existence is distinct from evidence that the current state was published before the slot. Queue cancellation context is retained; a later stale claim cannot erase unresolved historical ordering. Historical terminal visibility requires valid R2 publication metadata before the slot, for both runner status and queue-sourced cancellation. Execution end time alone does not establish when cron could read a terminal state. Missing, late, contradictory or equal-time evidence suppresses uncertain misses and is explicitly recorded as degraded coverage. A queue-specific read failure still permits independent heartbeat checks and an attempted degraded summary. A saturated queue may have preferentially trimmed completed background jobs; it is incomplete creation history. Missed-occurrence conclusions are suppressed with `queue_history_incomplete`, while retained-job and heartbeat checks continue. Independent occurrence history is needed for stronger coverage once the queue reaches its retention bound.

Findings are retained in create-only batches before a conditional per-day index update. Invalid index schemas/maps are preserved, with duplicate-tolerant batching and degraded coverage. Concurrent updates union keys with bounded retries. Storage is at least once: the three-day dedup window does not establish global incident identity or preserve the earliest observation for a prolonged outage. A later consumer must deduplicate across retained batches and retain the earliest `first_seen`; that consumer is not connected by this increment. No stored history is deleted.

Run and Weekly distinguish configured roles, built-in defaults, fresh heartbeat-reported roles, mismatches and unreadable configuration. Role display is observational: it does not change runner ownership, publication fencing, backup readiness or automatic takeover.

## Verification contract

Run `npm test`, `npx tsc --noEmit --incremental false`, targeted ESLint and the repository-required `python secret_free_tests.py`. Preserve existing Shot data, Ops/auth/CAS/freshness tests and policy gates. Loopback `scripts/ops-preview.mjs` has synthetic `roles_mismatch` and `roles_invalid` scenarios in addition to mixed/stale/offline. They use generated fixture identities and in-memory storage, refuse external fetches and are never deployed. Fixture screenshots do not establish live access or health.

The queue-read-error regression intentionally now requires independent heartbeat checks and degraded summary evidence instead of asserting that nothing is written. Test fixtures now use actual producer-style IDs/status identities. Existing tests are retained, including DST repeated-slot coverage, overlapping writers, corrupt indexes and zero outbound sends with credentials present. No test exclusions or relaxed assertions were added to production gates.

## Remaining B06 gates

- Reconcile an active-job registry from actual deployed schedules. Proposed/unverified deadlines remain unverified; the missed-slot grace is an observer policy, not a newly approved delivery SLA.
- Add reviewed non-golfprice receipts/content predicates and independently expected occurrences. Match occurrence/event/tour, every required output and freshness before writing `owner/ops/observations.json`. Passive findings cannot make a stage Verified. Inputs and delivery remain unknown until their occurrence sources are connected.
- Expose monitoring coverage and uncertainty in Weekly; classify upstream waits and verify repair/retry limits. Missing registry/content checks must not be reported as full-system coverage.
- Implement and test monitoring of monitoring, including a Cloudflare-independent path when both desktops and the main Worker are unavailable. This PR has no deadman implementation.
- Audit broader sender diversion/gates while preserving approved product messages. Keep all new operations outcomes/escalations/repairs suppressed.
- After reviewed activation and actual readback, record shadow start, the first full Monday-to-Sunday event week and next check. Require at least two complete event weeks plus every exit criterion in golf_ops/docs/OBSERVABILITY.md. A partial initial week does not count; no shadow clock has started merely because this code passed tests.

## Delivery and recovery

Before deployment, require final independent HIGH review, current remote/CI acceptance and a fresh bounded deployment contract with the exact Worker version/config and existing jobs left undisturbed. Verify owner/non-owner gates and real post-deploy health.

To stop shadow recording, deploy `OPS_SHADOW_ENABLED="0"`, retaining `LEGACY_ALERTS_ENABLED="0"`; keep all retained shadow/task records. Do not delete role/history objects as a rollback shortcut. A broader rollback must retain the reviewed legacy kill switch: pre-PR code used secret-only activation and is not a safe alert rollback when Telegram secrets are present. Actual activation and two-week graduation require their own recorded evidence.
