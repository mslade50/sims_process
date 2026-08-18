import type { Metadata } from "next";
import { headers } from "next/headers";
import "./globals.css";

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
  return <html lang="en"><body>{children}</body></html>;
}
