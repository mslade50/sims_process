# Owner skill model and simulation guide

Requested: a full graphical explanation on the private owner site, with step-by-step weekly green/yellow/red status. Route: `/model-walkthrough`, owner navigation label **How the model works**.

## Scope

Five sections cover weekly execution, skill estimation, tournament simulation, in-event updating, and predictive evidence. Player contributions and finish distributions come from the matching published explanation, not illustrative numbers. Workflow lights reuse `/api/ops/workflow`, require exact event/week scope and current output evidence, and degrade to yellow on errors, stale responses, incomplete proof or expiry. Unknown inputs/delivery occurrence wiring remains explicit. Operational green is separate from model validation.

The shared owner route list protects both HTML and RSC paths. The page adds no storage writes, jobs, pricing runs, model changes, dependencies or credentials. Existing Worker bindings, access settings, cron and flags are unchanged.

## Proof before review

- `npm test`: production build succeeded; 758 main-suite tests, 757 passed, 0 failed, 1 existing bundle-budget TODO. Posttest: 6 passed, 0 failed. Exit 0.
- `npx tsc --noEmit --incremental false`: exit 0.
- Scoped ESLint on new view, pure rules and walkthrough tests plus repaired override test: exit 0.
- Full lint was run. Existing errors remain in ExplainViews.tsx, MetricTrendPanel.tsx, ShapeViews.tsx and tests/bet-breakdowns.test.mjs; no new-file errors remain.
- Browser QA: 20 combinations of five sections, 375/1440 widths and light/dark. No page errors, page overflow, new-page contrast errors or form-label errors. Keyboard and workflow-network-error checks passed. The first pass found and fixed phone select overflow. `browser-qa.json` retains measurements and warnings.
- `python secret_free_tests.py`: attempted; collection stopped with 968 tests collected and 1 unrelated existing error in scratch_scoring_audit/implementation/test_course_feedback.py: `ModuleNotFoundError: No module named 'golfprice'`. No test collection guard was weakened.
- `git diff --check`: clean. Existing source baseline: `243a27410638a5bd96fbd5dc5cb59ec930ddeaf9`.

The browser fixtures are clearly labeled **BROWSER TEST FIXTURE**. Only the test browser substitutes whoami/workflow responses; no such bypass or fake status is in application code. Local static player data in screenshots is from an older published preview and does not attest to this week's delivery.

## Test changes

Added six fail-closed evidence/run/scope/owner/link tests through npm posttest and added this route to existing rendered-view smoke coverage. Repaired one stale override test: its fixed October 5 server clock had been combined with wall-clock expiry. That one test now uses fixed-clock +3 days. The lifetime rule and assertions are unchanged.

## Final delivery evidence (current main)

The original base was stale. It was fast-forwarded in this same checkout from 243a274 to 6bffc7e65f37a0c096b398c5ec58785a751940d1, preserving the page patch. Existing cron, freshness and shared-cache repairs are unchanged relative to origin/main. No branches or worktrees were created. The first fast-forward ran the repository post-merge sync hook and cleared two tracked distributions; their absence was verified and both were restored exactly from their tracked blobs. Further source-only synchronization disabled that hook for that command only.

The initial Codex review approved with 113 independent tests and three built owner-gate checks. Opus approved the page logic but requested a current deploy base, durable main-source publication, and immediate visibility refresh. The first two are handled by current-main synchronization plus direct-main publication; the third is implemented with request ordering protection. Browser tests prove immediate refresh on tab return and that an older response cannot replace a newer failure.

The owner replaced delivery policy during this task: no additional reviews, CI launches or blanket test gates are required. Final verification uses focused page/owner/render checks and the actual live integration.

Current-base focused checks: TypeScript, scoped lint, patch hygiene, 20 viewport/theme/section cases, keyboard/error states, visibility refresh and response-order race passed. The earlier current-base full dashboard run had 776 passes, one landing-size ratchet failure and one existing 150 KB TODO; the posttest did not run in that failed command. Six page rules checks had already passed independently. The unchanged main landing bundle is 210846 bytes gzip (205.90 KB); the added owner navigation makes it approximately 206.2 KB, roughly 0.2 KB above the unchanged 206 KB ratchet. This small overage is retained explicitly; no limit/test was weakened.

The current-base Python proof command still fails collection in the unrelated scratch course-feedback test: 1037 collected, one ModuleNotFoundError for golfprice. This is not represented as a passed suite.

Final focused command: production build followed by walkthrough, built owner-gate, rendered-view, freshness-UI and cron tests. Result: 76 tests, 76 passed, 0 failed. Immediate tab-return and out-of-order browser checks also passed. Live version and source commit are recorded below after publication.

## Rollback

Before this release, the active Worker version was `25ad10f9-28f7-4d49-9a8e-699bd7974c99` (deployment `9695f259-7916-4ec6-9e33-45159f0332f5`, October 30). From cloudflare_dashboard, restore it with:

`npx wrangler rollback 25ad10f9-28f7-4d49-9a8e-699bd7974c99 --name golf-model-dashboard --message "Undo owner walkthrough release" --yes`

This restores application code; no data migration or storage rollback is involved. Source is published directly to main under the owner's replacement November 3 delivery policy. Commit messages skip automatic CI; deployment uses the existing authenticated local command. No branches or worktrees are created.
