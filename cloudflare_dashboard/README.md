# Golf Model dashboard

Cloudflare-native replacement for the legacy Dash/Render interface. Since October 2026 the site is the front end for the golfprice model and opens on **This week**. Navigation groups: **Week** (This week, Players with a Field/Player toggle, Why priced, Distributions, Scoring and weather), **Model** (Inputs), **Operate** (Run), **Review** (Scorecard and P&L) and a collapsed **Archive** (History, Diagnostics: legacy-pipeline data, labelled as such). `/weather` redirects to `/weather-effects`, `/sg-distributions` to `/players`, and `/research` shows a retired notice; nothing was deleted, and an unknown address shows a not-found notice with the navigation. Home, Outrights, Matchups, Bets and Pricer remain intentionally omitted.

The existing Python simulation remains the source of truth. `scripts/export_dashboard_data.py` converts its current CSV, Parquet, and Google Sheets-backed data into browser-safe JSON snapshots. `scripts/publish_dashboard_data.py` uploads every data object to the dedicated `golf-model-dashboard-data` R2 bucket and publishes `manifest.json` last, so the Worker never exposes a partially refreshed snapshot. Packaged assets remain a fallback only.

## Local workflow

```powershell
npm install
npm run export:data
npm run dev
```

Production verification:

```powershell
npm run lint
npx tsc --noEmit --incremental false
npm test
```

Private Cloudflare deployment (uses the authenticated Wrangler account):

```powershell
npm run deploy:cloudflare
```

The direct Cloudflare deployment binds `DASHBOARD_DATA` to the private dashboard R2 bucket. Its `workers.dev` route is protected by a Worker-specific, deny-by-default Cloudflare Access application; only members matched by an explicit Allow policy can sign in. Preview URLs remain disabled.

The normal weekly entry point remains `python push_dashboard_data.py`. It copies and syncs the simulation artifacts, exports fresh JSON, and publishes that JSON to R2. The Monday grading workflow runs the same path automatically. A separate `Publish Cloudflare Dashboard Data` workflow can refresh only the JSON on demand. `Deploy Cloudflare Dashboard` always validates application changes and deploys them automatically when the scoped `CLOUDFLARE_API_TOKEN` repository secret is present; otherwise application deployment remains an authenticated local command. Neither path depends on Render.

GitHub uses bucket-scoped `DASHBOARD_R2_ACCESS_KEY_ID` and `DASHBOARD_R2_SECRET_ACCESS_KEY` secrets for JSON publication. They are deliberately separate from the simulation repository's other R2 credentials so rotating or restricting dashboard access cannot disrupt historical-data workflows.

## Local preview and the packaged golfprice snapshot

`public/golfprice/` is an untracked, git-ignored copy of golfprice data used only by the no-Worker local preview (`RESEARCH_LOCAL_PREVIEW=1`, see `.claude/launch.json`), where `/api/golfprice/*` does not exist. `app/data.ts` falls back to it only when the API response is not the Worker's structured JSON (network failure, HTML or empty body); a real Worker answer such as 404 "not published yet" or 503 is shown as is. A normal `npm run build` removes `dist/client/golfprice` so production never ships the snapshot; `RESEARCH_LOCAL_PREVIEW=1` or `PACKAGE_GOLFPRICE_STATIC=1` keeps it. Copy fresh data into `public/golfprice/` by hand when you need it locally.

## Architecture

The interface runs on [vinext](https://github.com/cloudflare/vinext) and a Cloudflare Worker. R2 serves the current JSON snapshot at stable `data/` keys without rebuilding the interface; static assets provide a safe fallback if storage is briefly unavailable.

## Prerequisites

- Node.js `>=22.13.0`

## Private access

Signed-in visitors receive both `oai-authenticated-user-id` and `oai-authenticated-user-email`. Private Sites require every visitor to sign in; public Sites may also have anonymous visitors, for whom neither header is present.

The user ID is stable for the same user on the same Site and different across Sites. Email and name are intended for display or contact purposes.

SIWC-authenticated workspace sites may also receive
`oai-authenticated-user-full-name` when the user's SIWC profile has a non-empty
`name` claim. The full-name value is percent-encoded UTF-8 and is accompanied by
`oai-authenticated-user-full-name-encoding: percent-encoded-utf-8`.

Treat the full name as optional and fall back to email when it is absent:

```tsx
import { headers } from "next/headers";

export default async function Home() {
  const requestHeaders = await headers();
  const userId = requestHeaders.get("oai-authenticated-user-id");
  const email = requestHeaders.get("oai-authenticated-user-email");
  const encodedFullName = requestHeaders.get("oai-authenticated-user-full-name");
  const fullName =
    encodedFullName &&
    requestHeaders.get("oai-authenticated-user-full-name-encoding") ===
      "percent-encoded-utf-8"
      ? decodeURIComponent(encodedFullName)
      : null;

  const displayName = fullName ?? email;
  // ...
}
```

## Useful Commands

- `npm run dev`: start local development
- `npm run build`: verify the vinext build output
- `npm test`: build the dashboard and verify the rendered application shell

## Learn More

- [vinext Documentation](https://github.com/cloudflare/vinext)
