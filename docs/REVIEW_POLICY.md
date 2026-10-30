# Review policy

Common review rules (proof, risk tiers, reviewers, structure checklist) are in `repo/REVIEW_POLICY.core.md` of `mslade50/agent-rules`, available on this machine at `~/.agent-rules/src/repo/REVIEW_POLICY.core.md`. This file holds sims_process additions only.

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
