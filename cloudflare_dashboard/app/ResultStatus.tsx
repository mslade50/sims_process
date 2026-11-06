"use client";

/** Shared result-count announcement: polite, atomic, singular/plural aware. Visible text optional (the heading often already shows the count). */
export function plural(n: number, one: string, many = `${one}s`): string {
  return `${n} ${n === 1 ? one : many}`;
}

export function ResultStatus({ count, noun, suffix = "shown", visible = false }: { count: number; noun: string; suffix?: string; visible?: boolean }) {
  const text = count === 0 ? `No ${noun}s match` : `${plural(count, noun)} ${suffix}`;
  return <span className={visible ? "result-status num" : "sr-only"} role="status" aria-live="polite" aria-atomic="true">{text}</span>;
}
