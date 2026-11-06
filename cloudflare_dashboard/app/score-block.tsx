"use client";

import type { ReactNode } from "react";
import { Section } from "./ui";

/** A page section for the scoring / history pages: anchored heading + one-line purpose, no card. */
export function Block({ id, title, eyebrow, actions, children }: { id: string; title: string; eyebrow?: string; actions?: ReactNode; children: ReactNode }) {
  return <Section id={id} title={title} purpose={eyebrow} actions={actions}>{children}</Section>;
}
