# Weekly workflow: owner and agent handoff

Open `/ops` on the Golf Model site. This is the retained work record for the rollout, with a separate measured weekly workflow. Read the [reviewed execution brief](https://github.com/mslade50/golf_ops/blob/main/docs/ROLLOUT_BRIEF.md) first. B01 remains observation pending. On October 27 the owner chose retaining the remote shared 0.4-Kelly implementation and directed investigation of reliable US Pinnacle website access, with DataGolf as fallback if needed. Exact authenticated T3 provenance is retained in golf_ops/docs/ROLLOUT_DECISIONS_20261027.json. Component reconciliation and source verification remain pending. A task note cannot expand those terms.

## Authority and storage

The Worker requires a cryptographically verified Cloudflare Access JWT and an explicitly configured owner email for both the page and API. Missing owner configuration fails closed. An email header alone is insufficient. Colleagues cannot open this page or read its API.

Future agents start at `GET /api/ops/bootstrap`, then read `/api/ops/tasks` and `/api/ops/workflow`. The server supplies current actor identity; clients cannot choose their actor. Agent access additionally requires an existing Cloudflare Access service token admitted by the Access application and its public client ID in the Worker's comma-separated `OPS_AGENT_CLIENT_IDS`. This defaults to empty: unattended agent access is not provisioned by this change. Use the approved private secret mechanism for the service-token secret; never put it in source, a task note, or a command transcript. Agents can record progress but cannot approve D01/D02. Human decisions retain exact terms, verified owner identity, time and revision.

Task documents live in the private dashboard R2 bucket at `owner/ops/tasks/<id>.json`. Missing documents are read from the reviewed seed; once written, retained documents take precedence over deployment seeds. Each write appends identity, action, note, evidence and revision. Conditional R2 writes protect concurrent changes. Invalid/corrupt records fail closed and are never replaced. History limits stop new writes; they do not discard history. The API does not execute commands, send notifications, change models or place bets.

## Claim and update protocol

1. Read `GET /api/ops/tasks/B02` (substitute the item being worked).
2. POST JSON to that URL with `op: "claim"`, the exact current `revision`, and a clear `note`. Keep the returned `lease_token` private. Claims last 15 minutes.
3. Renew with another claim request, current revision and token before expiry. Updates require the live claim, token and same authenticated actor. Another actor may recover an expired claim; the recovery is recorded.
4. POST `op: "update"` with current revision, lease token, note, state, next_step and bounded evidence references. Verified task state requires linked proof. A 409 means reread and reconcile; never retry with a guessed revision.
5. Release with `op: "release"`, current revision, token and note. A lost token cannot override a live claim; wait for expiry.

Supported task states: planned, in_progress, review, observation_pending, verified and blocked. Separate implementation delivery, timed observation and acceptance. Give a pending gate its actor, trigger and next check in `next_step`. Owner decisions use `op: "decision"`, state verified (exact approved terms) or blocked (deferred). An agent's decision note never approves a decision.

## Health evidence and current limits

The calendar uses the actual scheduler slot/window definitions and America/New_York time, including DST. It describes expected opportunities, not guaranteed useful work. The API reads only the newest 32 queued executions; absence is unverified. A completed job or exit zero cannot make a stage verified. Receipt checks require matching event, tour, occurrence, fresh timestamps and linked output checks for every selected published event. Task completion never changes these measured states. No receipt producer is connected by this page increment; B06 instruments and validates producers separately. Producer/export/email observations therefore remain unknown until their reviewed instrumentation ships.

Observations are read from `owner/ops/observations.json`, schema `golfops.observations.v1`. This file is not writable through the task API. Do not fabricate production receipts from notes or old successful jobs. Job-backed occurrences must be the exact latest known successful job ID, with every named check in `STAGE_EVIDENCE_CONTRACT`. Inputs and delivery remain unknown until their independent expected-occurrence source is connected. Every receipt state expires. The exact schema is defined by `StageReceipt` in `app/ops-rules.ts`; receipt producers require their own review before promotion. Missing, stale, wrong-event or malformed evidence must not turn green. Malformed queue, event-index or receipt documents are refused rather than silently filtering rows and exposing older success. Machine heartbeat/code agreement alone never proves backup readiness, unattended startup, replicated state, fencing or takeover.

## Verification and delivery

Build and run `npm test`, `npx tsc --noEmit --incremental false`, and ESLint on changed files. The repository also requires `python secret_free_tests.py` before independent review/PR. Full repository lint has existing unrelated findings; report them instead of weakening checks.

For local browser checks, build first, then run `node scripts/ops-preview.mjs 5317`. This helper binds loopback only, generates a test identity, uses in-memory R2 fixtures and rejects external fetches. It is never deployed. `/__fixture?scenario=mixed|stale|offline&role=owner|colleague` selects test states. Restart it after rebuilding to avoid old Worker/new asset mismatches. These screenshots are fixture verification, not live account proof.

Before accepting B02, independently review security/evidence behavior, deliver through the normal dashboard deployment, inspect the actual owner page after owner sign-in, confirm denied colleague access, persist/read back a real non-decision handoff and provision/verify agent access if needed. Keep unfulfilled acceptance gates visible. Live owner login is not supplied by the local fixture.

## Rollback

Revert the page/API commit and redeploy the prior Worker through the dashboard workflow. Leave all `owner/ops/` R2 objects intact so a later fixed release reads the retained history. To revoke machine access, remove its public client ID from `OPS_AGENT_CLIENT_IDS` and revoke that service token through Cloudflare Access. Do not delete task records or bypass Access to repair the page. This rollback does not stop or restart golf runners.
Browser regressions: after starting the fixture, set `OPS_PLAYWRIGHT_MODULE` to an existing Playwright package and run `node scripts/ops-browser-check.mjs`. The helper has no product dependency or browser installation requirement in the normal build. It exercises only loopback fixtures. Drafts retain their starting revision; polling cannot silently rebase them. Reconciliation explicitly reloads the current record and discards the local draft. Workflow requests time out after 10 seconds, task/workflow responses update independently, and displayed health expires after 60 seconds or the content receipt's earlier expiry.
