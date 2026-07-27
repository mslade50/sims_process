# Local Golf Simulation Runner

The midweek workflow uses a hybrid execution model:

- GitHub-hosted Ubuntu runner: downloads odds and checks the Sheet transition.
- Windows self-hosted runner labeled `golf-sim`: runs
  `live_stats_engine.py --automation` and `round_sim.py`.

The self-hosted runner uses its own checkout below the runner installation
directory. It does not run jobs inside the interactive development checkout.

## Availability

The runner listener must be active and the PC must be awake to accept a job.
If either is unavailable, GitHub queues the job. The midweek coordinator repeats
its freshness and state checks after the local runner starts, so a delayed job
does not use stale odds.

Recommended setup:

1. Install the runner as a Windows service for reliable unattended starts.
2. Allow the PC to sleep normally if immediate execution is not required.
3. Keep the PC awake during the expected Thursday-Saturday transition windows
   when timely prices matter.

The service is a lightweight GitHub job listener, not an AI agent. No inbound
port or public endpoint is required; it connects outbound to GitHub.

## Read-only validation

After registration or runner maintenance, manually dispatch
`local-runner-smoke.yml`. It uses the same runner label and concurrency lock as
the simulation workflows, then verifies:

- Python dependencies and the Rust kernel
- GitHub secret delivery and Google Sheet service-account access
- DataGolf API access
- the Sheet's `course_lat_lon` value and Open-Meteo course-local forecast
- private scraped-odds access
- repository push authorization via `git push --dry-run` (no branch is created)

The smoke workflow does not update the Sheet, run a simulation, send email, or
write to the repository.

## One-time registration

GitHub requires a repository-owner registration token that expires after one
hour:

1. Open `https://github.com/mslade50/sims_process/settings/actions/runners/new`.
2. Choose Windows and x64.
3. Open an Administrator PowerShell in GitHub's recommended
   `C:\actions-runner` directory.
4. Run the download, checksum, and extraction commands GitHub displays.
5. Add `--name <computer-name>-golf-sim --labels golf-sim --runasservice`
   to the displayed `config.cmd` command.
6. Confirm the runner appears online under repository Actions settings with
   labels `self-hosted`, `Windows`, `X64`, and `golf-sim`.

Never save or commit the registration token. It is used only during
registration; normal jobs authenticate through the runner's generated
credentials.

## Manual listener alternative

To avoid an always-running service, omit `--runasservice` during registration
and launch `C:\actions-runner\run.cmd` only when the machine should accept
jobs. Pending GitHub jobs will wait for it. Closing that window takes the
runner offline again.
