import type { Metadata, Viewport } from "next";
import { headers } from "next/headers";
import "./tokens.css";
import "./globals.css";
import "./design.css";
import { themeBootScript } from "./ui-rules";
import { earlyFetchScript } from "./shell-rules";

export const viewport: Viewport = { width: "device-width", initialScale: 1, viewportFit: "cover" };

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host = requestHeaders.get("x-forwarded-host") ?? requestHeaders.get("host") ?? "localhost:3000";
  const protocol = requestHeaders.get("x-forwarded-proto") ?? (host.startsWith("localhost") ? "http" : "https");
  const origin = `${protocol}://${host}`;
  const socialImage = `${origin}/og.png`;

  return {
    title: {
      default: "Golf Model",
      template: "%s · Golf Model",
    },
    description: "Live golf simulation intelligence, distributions, weather, performance, and model diagnostics.",
    openGraph: {
      title: "Golf Model",
      description: "Simulation intelligence, without the noise.",
      images: [{ url: socialImage, width: 1200, height: 630, alt: "Golf Model dashboard" }],
    },
    twitter: {
      card: "summary_large_image",
      title: "Golf Model",
      description: "Simulation intelligence, without the noise.",
      images: [socialImage],
    },
  };
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <meta name="color-scheme" content="dark light" />
        {/* Found by the preload scanner while stylesheets still block scripts; the same request is then reused by app/data.ts. */}
        <link rel="preload" href="/api/golfprice/index.json" as="fetch" crossOrigin="anonymous" fetchPriority="high" />
        <script dangerouslySetInnerHTML={{ __html: themeBootScript() }} />
        <script dangerouslySetInnerHTML={{ __html: earlyFetchScript() }} />
      </head>
      <body>{children}</body>
    </html>
  );
}
