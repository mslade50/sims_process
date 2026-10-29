# Review policy

The shared review rules for this repository are in the block below. They are copied from https://github.com/mslade50/agent-rules (edit them there, not here). Repository-specific proof commands, HIGH items and CI details follow the block.

<!-- agent-rules:review-core:start (from mslade50/agent-rules repo/REVIEW_POLICY.core.md @ 92f4eaa; edit there, not here) -->
# Review policy (common core)

This is the common core of every repository's `docs/REVIEW_POLICY.md`. A repository copy should point to this file and keep only what is specific to that repository: its proof commands, its extra HIGH items, and its CI notes. Change the core in https://github.com/mslade50/agent-rules, not in the repository copy.

A coding agent writes the change. A different agent, with fresh context and no memory of the writing session, reviews it. People spend attention by risk. The repository owner does not review the code. A bad change should be cheap to undo.

## Proof before review

- Run the repository's own checks from the repo root before you open the PR. The repository policy lists the exact commands.
- Open the PR as a draft and push fixes while it is a draft. Mark it ready (`gh pr ready`) once, after the local proof passes and the review is done.
- Paste the real output into the PR description under Proof. "Tests passed" with no output is not proof.
- A change to displayed UI or to alert or notification text also needs something real: a screenshot, the sample message or a JSON sample.
- Do not weaken tests to pass. If a test must change, say why in the PR body under Test changes.

## Delivery

- Every change goes through a focused PR. No direct commits to main.
- The agent merges its own PR (merge commit unless the repository says otherwise) when both gates pass. Before marking ready: lint, types and tests pass locally with proof; the agent reviewers approved; no open review comments or nits; the diff is lean with few comments. Before merging: no open review comments or unresolved threads, all required CI is green after the PR is ready (review-check skips drafts), and nothing changed since the approvals.
- Ping the owner only for HIGH-risk decisions, or for questions that belong to the owner (model or strategy changes).

## Risk tiers

HIGH when the change touches any of these:

- money paths, trade placement or sizing, risk limits, broker or trading code;
- deploys, releases or publishing;
- alerts, emails or other notifications;
- secrets;
- workflows or CI;
- the runtime state of live agents, or scheduled tasks.

A repository adds its own HIGH items (for example pricing, signal or model logic, or published data outputs). LOW is everything else. Trivial changes (a typo, comment or doc wording, or one config value with no behavior change) need no review. If you are unsure, call it HIGH.

## Reviewers

- HIGH: two independent reviews from two different model families, a Claude Opus reviewer (a subagent with fresh context) and a Codex reviewer. Launch the Codex reviewer with T3 `delegate_task`: pick the Codex provider from `orchestrator_capabilities`, model `gpt-6.1-sol` (`gpt-6-astra` if Sol is at capacity; say which in the PR), `reasoningEffort` high or above, in its own worktree.
- LOW: one independent review.
- Each review round is a fresh task. Give it the original brief, the earlier findings, the author's responses and any open objections. Ask it to confirm that each finding is closed. Fix findings in the same branch, or in a follow-up PR if the change already merged. There is no round cap: repeat until the reviewers approve.
- Never merge HIGH work with one review. If it happened anyway, get the missing review now and tell the owner.
- Reviewers never edit, push, merge or comment on GitHub. They report a verdict, each issue with `file:line` and a concrete failure scenario, and real test output. They never touch the live checkout that scheduled agents run from.
- Every Codex reviewer brief includes: "Rate each bug from 1 to 10 on how often it would happen and how bad it would be. If the two scores multiplied are below 5, just file it as an issue and leave the code alone." A finding that scores below 5 becomes a GitHub issue and does not block the PR.

## Undo

Where practical, ship a risky change behind an environment variable or flag. Every PR states the rollback in plain words: what to revert, which flag to flip, or what to redeploy. "Revert if needed" is not a rollback step.

## Secrets

Never print or commit secrets. That includes logs, PR text and sample output. Scrub tokens, account identifiers and API keys before you paste proof.

## What CI checks

Each repository may run a read-only review-check job. It fails when the Proof or Rollback section of the PR body is missing, empty or still placeholder text. Draft PRs skip it, and marking the PR ready runs it. Where the repository has test-weakening and test-inventory guards, they fail the PR when tests get weaker and the body has no real Test changes note. Editing a guard script or its workflow counts as weakening. The repository policy lists its workflows and commands.
<!-- agent-rules:review-core:end -->

## Repository proof commands

From the repo root, run the secret-free suite before you open a pull
request or ask for review:

```
python secret_free_tests.py
```

That is the command CI runs, after
`python -m pip install -r requirements.txt "pytest>=8.1,<10"`.
It includes the review-checker tests. Paste the real summary (pytest and
script counts, plus the not-collected list) into the PR description, under
Proof. "Tests passed" with no output is not proof.

There is no linter config, so there is no lint command. Do not invent one.

If the change touches a displayed page or an alert, also attach something
real: the sample alert text, a screenshot, or a JSON sample. A description
of what the screen would look like is not evidence.

## Repository HIGH items

In addition to the shared HIGH list, call the change HIGH when it touches pricing, edge, simulation, or model logic, or published data outputs.

## CI details for this repository

`.github/workflows/review-check.yml` has four jobs. All of them use
`ubuntu-latest` and the `pull_request` event. None of them reads secrets.

The `body` job fails the check when the Proof or Rollback section is
missing, empty, or still placeholder text. It runs
`python3 scripts/check_pr_body.py` only. `## Test changes` is not required
by this job.

The `test-weakening` job runs `scripts/check_test_weakening.py` on the pull
request diff (`base...head`). It fails when tests get weaker and the body
has no real `## Test changes` note. It flags a deleted test file, a removed
test, a new skip, xfail, or importorskip (including a bare `importorskip`),
a new or wider exclusion (`--ignore`, `--deselect`, `-k`, a list-style
`"--ignore", "x"` pair, or a conftest `collect_ignore` list), or a looser
assertion. A rename, an equivalent assert rewrite, and a tighter tolerance
are not weakenings. `secret_free_tests.py` is scanned as CI config, including
`scripts/secret_free_tests.py`. Any edit to a guard checker
(`scripts/check_pr_body.py`, `scripts/check_test_weakening.py`,
`scripts/check_test_inventory.py`, or `secret_free_tests.py`) or to a
workflow that runs one of them is itself a weakening, including a
comment-only edit. That is what stops a pull request from turning its own
check off without saying so. Placeholder text does not count; that uses
`is_placeholder()` in `scripts/check_pr_body.py`.

Each of the four jobs checks out with `persist-credentials: false`, so later
steps do not keep the checkout token.

The `test-inventory` job is the main protection. It installs that commit's
`requirements.txt` and collects the secret-free suite on the base commit and
again on the head commit, using that tree's conftest, runner flags, and the
tests job's command. It records pytest node ids (collected, or skipped with
the reason, including `SKIP_TESTS`), `CREDENTIAL_IMPORTS` and `KERNEL_IMPORTS`
modules that were not collected, and each script program's reported check
count. It fails when a test id disappears, a running test becomes skipped or
not collected, or a script program disappears or reports fewer checks, unless
`## Test changes` is filled in. Added tests are fine. A plain rename is one
removed id and one added id, so it needs that section (an explicit
`old -> new` mapping). This comparison does not parse the diff, so an
exclusion that lives only in a conftest hook, the runner, the workflow run
list, or the installed requirements still shows up.

The `tests` job sets `permissions: contents: read` and nothing else. It
installs `requirements.txt` and `pytest>=8.1,<10`, sets `COEFFS_FROM_CACHE=1` so
coefficient import uses the committed cache, and runs
`python secret_free_tests.py`. Each pytest module runs in its own process.
Modules that need Google credentials, a live external service, or the
compiled `sims_kernel` extension are not collected when that dependency is
missing, and they still run locally when it is present. The list is
`conftest.py`. A collection error in a module that is not on that list
fails the job. One stale source-grep test is skipped; the rest of its
module runs. A `SCRIPT_MODULES` entry is collected anyway during the run and
fails the job if pytest finds test items there. Each script program must
report a nonzero pass count. There is no ruff config, so this job does not
run a linter.

Existing workflows are unchanged. They are scheduled or started by hand, and
some of them use a self-hosted runner. This check does not.

The PR template (`.github/PULL_REQUEST_TEMPLATE.md`) has four section
headings: Risk tier, Proof, Rollback steps, and Test changes.

Dependabot and Renovate do not fill this template, so the body check does not
fail their PRs. The weakening and inventory jobs skip them too. Every other
author, including other bots and coding agents, must fill Proof and Rollback
in, and must fill Test changes when the diff or the inventory says tests got
weaker.
