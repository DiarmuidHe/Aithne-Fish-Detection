/**
 * Conditional polling.
 *
 * The old dashboard polled on a flat interval forever, including overnight on a
 * hidden tab. Here polling only runs while there is a job to watch, backs off
 * once the wait stops being interesting, and stops entirely when the tab is
 * hidden — TanStack Query refetches on focus, so nothing is stale on return.
 */

import * as React from 'react';

export const FAST_INTERVAL = 3_000;
export const SLOW_INTERVAL = 15_000;
/** After two minutes of watching, the operator is no longer waiting on it. */
export const BACKOFF_AFTER_MS = 120_000;

export function useDocumentVisible(): boolean {
  const [visible, setVisible] = React.useState(
    () => typeof document === 'undefined' || document.visibilityState !== 'hidden',
  );
  React.useEffect(() => {
    const onChange = () => setVisible(document.visibilityState !== 'hidden');
    document.addEventListener('visibilitychange', onChange);
    return () => document.removeEventListener('visibilitychange', onChange);
  }, []);
  return visible;
}

/**
 * The `refetchInterval` for a query that only matters while work is in flight.
 * Returns `false` when there is nothing to watch, which stops the timer.
 */
export function pollInterval(active: boolean, watchingSinceMs: number | null): number | false {
  if (!active) return false;
  if (watchingSinceMs === null) return FAST_INTERVAL;
  return Date.now() - watchingSinceMs > BACKOFF_AFTER_MS ? SLOW_INTERVAL : FAST_INTERVAL;
}

/**
 * Track how long `active` has been continuously true, so callers can back off.
 * Returns null while inactive.
 */
export function useActiveSince(active: boolean): number | null {
  const since = React.useRef<number | null>(null);
  if (active && since.current === null) since.current = Date.now();
  if (!active && since.current !== null) since.current = null;
  return since.current;
}

/** Combines the rules above into a value for TanStack Query's refetchInterval. */
export function useConditionalPoll(active: boolean): number | false {
  const visible = useDocumentVisible();
  const since = useActiveSince(active);
  if (!visible) return false;
  return pollInterval(active, since);
}
