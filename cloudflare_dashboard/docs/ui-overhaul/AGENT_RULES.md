# Rules for UI overhaul agents (shared checkout)

Goal: every rubric category (docs/ui-overhaul/RUBRIC.md) >= 9.5 from both reviewers (Claude, Codex gpt-6.1-sol).
Bar: better than Linear, Stripe Dashboard, Vercel, Apple Sports, Kalshi, Data Golf for an expert bettor on phone and desktop.
Read: RUBRIC.md, DESIGN_SYSTEM.md, the latest review-codex-r*.md and review-claude-r*.md, INVENTORY.md.

- Windows + Git Bash; `python` not python3; write files with the file-write tool. Load the `web-design` skill if available.
- Edit ONLY the files your brief lists. Other agents edit other files concurrently. If you need a change in a file you do not own, describe it in your reply instead.
- Keep your files compiling at every save.
- Build + preview + screenshots only through `node scripts/ui-check.mjs --routes <routes> --crops --out dist-visual/<label>` (lock-serialised; live data, owner mode, http://localhost:3111). Query routes allowed, e.g. `players?player=10091`, `inputs?tab=adjust`. If the build fails in someone else's file, wait and retry.
- Event in URL: Abu Dhabi `?e=euro%3A2026142%3A2026-11-05`, WWT `?e=pga%3A457%3A2026-11-05`. Pages must use `useEventParam()` (app/event-param.ts) for the event.
- Tests: run the ones related to your files after a build (`node --experimental-strip-types --test tests/<x>.test.mjs`). Update tests that assert intentionally changed markup, never weaken coverage; say so.
- Look at your screenshots (phone + desktop, dark + light), critique against the rubric, iterate until you honestly score 9.5+. 0 visual-check errors and 0 warnings on your routes.
- HARD RULES: no commit, push or deploy; no remote R2 writes; never read secrets files; no stash/reset/clean. Never change numbers, model/betting logic, data semantics, Worker/API behaviour, job types or routes. Rewording for clarity is welcome; never change what a number means.
- Round 3 framing rule: PageHeader purpose line <= 1 line on desktop, <= 2 on phone; no repeated metadata (event name, run time, stage) that the shell bar already shows; answer/comparison before any configuration; at most one section-navigation element per page. Keep bundle budgets (tests/budgets.test.mjs) green: lazy-load new heavy code.
- Reply in under 15 lines: what changed, files, screenshot folder, tests, what you think still falls short of 9.5.
