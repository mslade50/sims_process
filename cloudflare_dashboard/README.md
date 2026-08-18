# Golf Model dashboard

Cloudflare-native replacement for the legacy Dash/Render interface. The site keeps the eight analytical views that remain useful—Outrights, Round Scores, Weather, Finish Distributions, SG Distributions, History, Performance, and Diagnostics—and intentionally omits Home, Matchups, Bets, and Pricer.

The existing Python simulation remains the source of truth. `scripts/export_dashboard_data.py` converts its current CSV, Parquet, and Google Sheets-backed data into browser-safe JSON snapshots. The Worker reads the same keys from the `DASHBOARD_DATA` R2 binding when populated and falls back to the packaged snapshot for safe first deploys.

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

The direct Cloudflare deployment intentionally omits an R2 binding and serves the packaged JSON snapshot. Its `workers.dev` route is protected by a Worker-specific, deny-by-default Cloudflare Access application; only members matched by an explicit Allow policy can sign in. Preview URLs remain disabled. This stays within the Workers/static-assets free tier at normal dashboard traffic levels and avoids provisioning storage until live snapshot uploads are needed.

## Architecture

The interface runs on [vinext](https://github.com/cloudflare/vinext) and a Cloudflare Worker. Static snapshots make a first deployment immediately usable; R2 can replace any snapshot at the same `data/` key without rebuilding the interface.

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
