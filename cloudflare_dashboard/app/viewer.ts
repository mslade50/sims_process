"use client";

/**
 * Who is looking (GET /api/whoami, worker/owner.ts), fetched once per page and shared. Only hides owner-only links; the Worker enforces access.
 * null until known, so owner-only controls never flash for a colleague.
 */
import { useEffect, useState } from "react";
import { viewerFrom, type Viewer } from "./shell-rules";

let pending: Promise<Viewer> | null = null;

function loadViewer(): Promise<Viewer> {
  pending ??= fetch("/api/whoami", { cache: "no-store", credentials: "same-origin" })
    .then((response) => response.json())
    .then(viewerFrom)
    .catch(() => viewerFrom(null));
  return pending;
}

export function useViewer(): Viewer | null {
  const [viewer, setViewer] = useState<Viewer | null>(null);
  useEffect(() => {
    let live = true;
    void loadViewer().then((value) => live && setViewer(value));
    return () => {
      live = false;
    };
  }, []);
  return viewer;
}

/** True only once the Worker has said this visitor is an owner. */
export const useIsOwner = (): boolean => useViewer()?.owner === true;
