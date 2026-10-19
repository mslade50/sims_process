# Review policy

A coding agent writes the change. A different agent, with fresh context and no
memory of the writing session, reviews it. People spend attention by risk:
risky changes get a full read, small ones get a light check. The repo owner
does not review the code himself. A bad change should be cheap to undo.
Ordinary fixes still go straight to main, as AGENTS.md already says, and this
policy applies whenever a pull request is opened.

## Proof before asking for review

This repo has no secret-free application test command and no linter config.
Root `test_*.py` modules are not run in CI, and many of them need credentials
before they can be collected. Do not invent a suite command for the pull request.

From the repo root, run the review-checker tests before you open a pull
request or ask for review:

```
python -m pytest scripts/test_review_check.py -q
```

Paste the real output into the PR description, under Proof. "Tests passed"
with no output is not proof.

If the change touches a displayed page or an alert, also attach something
real: the sample alert text, a screenshot, or a JSON sample. A description
of what the screen would look like is not evidence.

## Risk tiers

Call the change **HIGH** when any of these are true. HIGH needs a full
independent review:

- it places or sizes bets or trades
- it sends alerts or notifications
- it touches pricing, edge, simulation, or model logic, or published data outputs
- it publishes or deploys
- it changes workflows or secrets

Call it **LOW** only for display, copy, or docs tweaks. LOW gets a lighter
check: the diff matches the description, and the proof is actually there.

If you are unsure, call it HIGH.

## Undo

Where practical, ship a risky change behind an environment variable or a flag
so it can be turned off without hunting through code. Every PR states the
rollback in plain words: what to revert, which flag to flip, or what to
redeploy. "Revert if needed" is not a rollback step.

## Secrets

Never print or commit secrets. That includes logs, PR text, and sample
output. Scrub tokens, chat ids, and API keys before you paste proof.

## What CI checks

`.github/workflows/review-check.yml` fails the check when the Proof or
Rollback section is missing, empty, or still placeholder text. It runs only
that body check (`python3 scripts/check_pr_body.py`). It does not install
dependencies, run the application tests, or run a linter. The job uses
`ubuntu-latest` (a GitHub-hosted runner), the `pull_request` event, read-only
permissions, and no secrets, so a pull request from a fork stays safe.

Existing workflows are unchanged. They are scheduled or started by hand, and
some of them use a self-hosted runner. This check does not.

The PR template (`.github/PULL_REQUEST_TEMPLATE.md`) has the three section
headings: Risk tier, Proof, and Rollback steps.

Dependabot and Renovate do not fill this template, so the body check does not
fail their PRs. Every other author, including other bots and coding agents,
must fill it in.
