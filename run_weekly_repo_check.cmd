@echo off
rem Tri-repo operational audit (Windows Task Scheduler):
rem   Monday 11:00    post-event mode  (grading landed, publication, CI week)
rem   Wednesday 17:00 readiness mode   (stale files, fan-out, kernel, odds)
rem Report-only: the skill instructs Claude not to change anything; failures
rem are alerted via Telegram (maker_alerts) using the repo .env.
cd /d "C:\Users\McKinley Slade\dev\sims_process"
"C:\Users\McKinley Slade\.local\bin\claude" -p "/weekly-repo-check" --dangerously-skip-permissions > "permanent_data\weekly_repo_check_last.log" 2>&1
