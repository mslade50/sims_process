# GitHub Actions Setup Guide

## Files Created

1. **`.github/workflows/test-env.yml`** - Tests environment/secrets
2. **`.github/workflows/run-sim.yml`** - Runs actual simulations
3. **`requirements.txt`** - Python dependencies for Actions

## Setup Steps

### 1. Add Workflow Files to Your Repo

```bash
# Create the workflows directory
mkdir -p .github/workflows

# Copy the workflow files
# (Place test-env.yml and run-sim.yml in .github/workflows/)

# Add and commit
git add .github/workflows/test-env.yml
git add .github/workflows/run-sim.yml
git add requirements.txt
git add test_env_setup.py
git commit -m "Add GitHub Actions workflows"
git push origin main
```

### 2. Verify Secrets Are Set

Go to: `https://github.com/mslade50/sims_process/settings/secrets/actions`

You should see all 5 secrets:
- ✅ DATAGOLF_API_KEY
- ✅ EMAIL_PASSWORD
- ✅ EMAIL_RECIPIENTS
- ✅ EMAIL_USER
- ✅ GOOGLE_CREDS_JSON

---

## Testing the Setup

### Test Workflow (Verify Secrets)

1. Go to your repo on GitHub
2. Click **Actions** tab
3. Click **Test Environment Setup** in the left sidebar
4. Click **Run workflow** dropdown (top right)
5. Click **Run workflow** button

**What it does:**
- Checks all environment variables are set
- Tests Google Sheets connection
- Tests DataGolf API
- Tests email SMTP authentication
- Lists data files (some may not exist in GitHub - that's ok)

**Expected result:** Green checkmark ✅ means all secrets work!

---

## Running Simulations

### Manual Run (Recommended for Testing)

1. Go to **Actions** tab
2. Click **Run Round Simulation** in the left sidebar
3. Click **Run workflow** dropdown
4. Fill in inputs:
   - **sim_round**: `2` (which round to simulate)
   - **expected_avg**: Leave blank to read from Google Sheet, or enter like `72.5`
5. Click **Run workflow**

**What it does:**
- Runs `round_sim.py` with your secrets
- Generates Excel/CSV outputs
- Sends email with results
- Saves outputs as artifacts (downloadable for 30 days)

### Download Results

After the workflow completes:
1. Click on the completed workflow run
2. Scroll to bottom → **Artifacts** section
3. Click **simulation-results** to download ZIP

---

## Scheduled Runs (Optional)

To run automatically on tournament days, edit `.github/workflows/run-sim.yml`:

**Uncomment these lines (remove the `#`):**
```yaml
schedule:
  - cron: '0 12 * * 4'  # Thursdays at noon UTC
  - cron: '0 12 * * 5'  # Fridays at noon UTC
```

**Customize the schedule:**
- Format: `minute hour day-of-month month day-of-week`
- `0 12 * * 4` = Noon UTC on Thursday
- `0 18 * * 4` = 6pm UTC on Thursday (1pm EST)
- Use https://crontab.guru/ to help build cron expressions

**Note:** Scheduled runs will use Google Sheet config (not CLI inputs).

---

## Troubleshooting

### Test workflow fails
→ Check that secrets are set correctly in GitHub Settings

### Simulation fails with "file not found"
→ Make sure `sim_inputs.py` and data files are committed to repo
→ Run `live_stats_engine.py` first to generate prediction files

### Email not sending
→ Verify EMAIL_PASSWORD is your Gmail **App Password** (not regular password)
→ Check that 2FA is enabled on your Gmail account

### Google Sheets connection fails
→ Verify GOOGLE_CREDS_JSON is the complete JSON (no line breaks)
→ Make sure the sheet is shared with the service account email

---

## Viewing Workflow Logs

1. Go to **Actions** tab
2. Click on a workflow run
3. Click on the job name (`test-credentials` or `run-simulation`)
4. Expand each step to see detailed output

This shows you exactly what happened, including any error messages.

---

## Cost

GitHub Actions is **FREE** for public repos and includes:
- 2,000 minutes/month for private repos (free tier)
- Each workflow run takes ~1-2 minutes
- Plenty for weekly tournament simulations

---

## Security Notes

✅ Secrets are encrypted and never visible in logs
✅ Secrets can only be accessed by workflows in this repo
✅ You can rotate secrets anytime in Settings → Secrets
✅ Deleting a secret immediately revokes access
