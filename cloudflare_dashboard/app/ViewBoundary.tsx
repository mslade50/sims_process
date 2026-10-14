"use client";

import { Component, Suspense, type ReactNode } from "react";
import { LoadingState } from "./components";

const RELOAD_KEY = "golf-dashboard-chunk-reload";

/** A failed dynamic import after a deploy: the old page asks for a chunk hash that no longer exists. */
export function isChunkLoadError(error: unknown): boolean {
  const e = error as { name?: string; message?: string } | null;
  const text = `${e?.name ?? ""} ${e?.message ?? String(error ?? "")}`;
  return /ChunkLoadError|Failed to fetch dynamically imported module|error loading dynamically imported module|Importing a module script failed|Loading chunk [\w-]+ failed|Loading CSS chunk/i.test(text);
}

/** One hard reload per tab session for a stale-chunk error; sessionStorage guards against a reload loop. Returns true when a reload started. */
export function reloadOnceForStaleChunk(): boolean {
  try {
    if (sessionStorage.getItem(RELOAD_KEY)) return false;
    sessionStorage.setItem(RELOAD_KEY, String(Date.now()));
  } catch {
    return false; // storage blocked: do not risk a loop
  }
  window.location.reload();
  return true;
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
      <Suspense key={viewKey} fallback={<LoadingState label="Loading view" />}>{children}</Suspense>
    </ViewErrorBoundary>
  );
}
