# Scoring actuals and forecast feedback

`live_stats_engine.write_actuals_to_sheet` records completed official scores
before optional observed-weather diagnostics. Missing coordinates, wind, or
dew observations do not prevent recording the actual or forecast miss.
Refreshed weather forecasts are never labeled realized observations.

`Scoring Feedback` retains content-addressed JSON receipts for forecasts,
actuals, weather diagnostics, and feedback ingestion. Statuses are `applied`,
`missing`, `failed`, and `not_due`. Zero misses and zero corrections are
`applied`. Identical backfills reuse their receipt IDs.

Original forecasts are retained when live expectations are published and when
an approved round simulation is saved. Dated tee times and the course timezone
are required; a publication at or after first tee is excluded. The scoring API
binds results to the season schedule, event, physical course, and requested
round. Partial active rounds are not recorded as completed. Completed scores
remain included when a player subsequently withdraws or misses the cut.

The actuals block is `round_config!U10:AK14`. AA is the original published
forecast, AB the official completed-field average, and AC their difference.
AF:AK retain event/year/course identity and receipt references. Feedback
ingestion verifies these against the receipt ledger before using AC, preserving
the existing two-decimal misses and evidence weights. Unreceipted legacy
structural deltas cannot enter feedback. New-event rollover clears U11:AK14;
the receipt ledger remains available for audit.

Backfills use a reviewed JSON plan with event/year/course/tournament identity,
course par, and original forecast receipts. Each forecast contains its value,
player cohort, publication and first-tee timestamps, source evidence, and
explicit missing/extra player exceptions. Source evidence must be recovered
from original artifacts or pre-start publication logs, never a hindsight
rebuild. Official actuals average all completed players; cohort exceptions
remain explicit in the receipt.

```powershell
python scoring_feedback.py --plan <verified-plan.json> --evidence-dir sheet_backups/<run>
```

This command snapshots the config tab, records official actuals, checks Sheet
readback and R1/R2/R3 feedback ingestion, and records R4 ingestion as not due.
It does not run simulations, modify future expected scores, or send email.
Inspect `verification.json` and the Sheet ledger to verify completion. Do not
trigger email-producing operational entrypoints to perform an actuals repair.
