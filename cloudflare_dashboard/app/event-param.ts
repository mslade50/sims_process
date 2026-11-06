"use client";

import { useEffect, useState } from "react";

/** The event chosen in the shell's switcher (?e=<event_uid>), read after mount so the server render matches. Empty until read, or when none is set. */
export function useEventParam(): string {
  const [uid, setUid] = useState("");
  useEffect(() => {
    queueMicrotask(() => setUid(new URLSearchParams(window.location.search).get("e") ?? ""));
  }, []);
  return uid;
}
