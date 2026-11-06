"use client";

import { Component, Suspense, useEffect, type ReactNode } from "react";
import { SkeletonPage } from "./ui";
import { isChunkLoadError, mayReloadForChunk } from "./shell-rules";

const RELOAD_KEY = "golf-dashboard-chunk-reload";

/** One hard reload per minute for a stale-chunk error; sessionStorage guards against a reload loop. Returns true when a reload started. */
export function reloadOnceForStaleChunk(): boolean {
  try {
    if (!mayReloadForChunk(sessionStorage.getItem(RELOAD_KEY), Date.now())) return false;
    sessionStorage.setItem(RELOAD_KEY, String(Date.now()));
  } catch {
    return false; // storage blocked: do not risk a loop
  }
  window.location.reload();
  return true;
}

/**
 * Page-wide recovery for a deploy that happened while this page was open: any script or lazy import (shell search, customize dialog,
 * prefetches, views) that fails with a missing-chunk error reloads once to pick up the new build. Mounted once by the shell.
 */
export function useChunkLoadRecovery() {
  useEffect(() => {
    const onPreload = (event: Event) => { event.preventDefault(); reloadOnceForStaleChunk(); };
    const onRejection = (event: PromiseRejectionEvent) => { if (isChunkLoadError(event.reason)) reloadOnceForStaleChunk(); };
    const onError = (event: ErrorEvent) => { if (isChunkLoadError(event.error ?? event.message)) reloadOnceForStaleChunk(); };
    window.addEventListener("vite:preloadError", onPreload);
    window.addEventListener("unhandledrejection", onRejection);
    window.addEventListener("error", onError);
    return () => { window.removeEventListener("vite:preloadError", onPreload); window.removeEventListener("unhandledrejection", onRejection); window.removeEventListener("error", onError); };
  }, []);
}

type Props = { viewKey: string; children: ReactNode };
type State = { error: unknown; attempt: number };

/** Error boundary for one view. The shell outside it never goes blank because a view throws. */
class ViewErrorBoundary extends Component<Props, State> {
  state: State = { error: null, attempt: 0 };

  static getDerivedStateFromError(error: unknown): Partial<State> {
    return { error: error ?? new Error("Unknown error") };
  }

  componentDidCatch(error: unknown) {
    if (isChunkLoadError(error)) reloadOnceForStaleChunk();
  }

  componentDidUpdate(previous: Props) {
    if (previous.viewKey !== this.props.viewKey && this.state.error) this.setState({ error: null });
  }

  retry = () => {
    // A stale chunk will not recover by re-mounting; reload once, otherwise re-mount and try again.
    if (isChunkLoadError(this.state.error) && reloadOnceForStaleChunk()) return;
    this.setState((s) => ({ error: null, attempt: s.attempt + 1 }));
  };

  render() {
    if (this.state.error) {
      return (
        <div className="empty-state view-error" role="alert">
          <strong>This panel failed to load</strong>
          <span>{isChunkLoadError(this.state.error) ? "The dashboard was updated while this page was open." : "Something went wrong while drawing this page. The rest of the dashboard still works."}</span>
          <button type="button" className="ex-link" onClick={this.retry}>Retry</button>
        </div>
      );
    }
    return <div key={this.state.attempt} style={{ display: "contents" }}>{this.props.children}</div>;
  }
}

export function ViewBoundary({ viewKey, children }: Props) {
  return (
    <ViewErrorBoundary viewKey={viewKey}>
      <Suspense key={viewKey} fallback={<SkeletonPage label="Loading view" />}>{children}</Suspense>
    </ViewErrorBoundary>
  );
}
