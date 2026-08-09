@echo off
rem Weekly sim health audit (Monday night, Windows Task Scheduler).
rem Report-only: the skill instructs Claude not to change anything; failures
rem are alerted via Telegram (maker_alerts) using the repo .env.
cd /d "C:\Users\McKinley Slade\dev\sims_process"
"C:\Users\McKinley Slade\.local\bin\claude" -p "/sim-health-check" --dangerously-skip-permissions > "permanent_data\sim_health_check_last.log" 2>&1
