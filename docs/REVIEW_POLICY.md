# Review policy

A coding agent writes the change. A different agent, with fresh context and no
memory of the writing session, reviews it. People spend attention by risk:
risky changes get a full read, small ones get a light check. The repo owner
does not review the code himself. A bad change should be cheap to undo.
Ordinary fixes still go straight to main, as AGENTS.md already says, and this
policy applies whenever a pull request is opened.

## Proof before asking for review

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

`.github/workflows/review-check.yml` has two jobs. Both use `ubuntu-latest`
and the `pull_request` event. Neither job reads secrets.

The `body` job is unchanged. It fails the check when the Proof or Rollback
section is missing, empty, or still placeholder text. It runs
`python3 scripts/check_pr_body.py` only.

The `tests` job sets `permissions: contents: read` and nothing else. It
installs `requirements.txt` and `pytest>=8.1,<10`, sets `COEFFS_FROM_CACHE=1` so
coefficient import uses the committed cache, and runs
`python secret_free_tests.py`. Each pytest module runs in its own process.
Modules that need Google credentials, a live external service, or the
compiled `sims_kernel` extension are not collected when that dependency is
missing, and they still run locally when it is present. The list is
`conftest.py`. A collection error in a module that is not on that list
fails the job. One stale source-grep test is skipped; the rest of its
module runs. There is no ruff config, so this job does not run a linter.

Existing workflows are unchanged. They are scheduled or started by hand, and
some of them use a self-hosted runner. This check does not.

The PR template (`.github/PULL_REQUEST_TEMPLATE.md`) has the three section
headings: Risk tier, Proof, and Rollback steps.

Dependabot and Renovate do not fill this template, so the body check does not
fail their PRs. Every other author, including other bots and coding agents,
must fill it in.
