import * as React from 'react';

/**
 * A locally-edited value that only reaches the caller once typing pauses.
 * Search runs against the whole library, so a request per keystroke is waste.
 */
export function useDebouncedValue(
  value: string,
  onCommit: (value: string) => void,
  delay = 250,
): [string, (next: string) => void] {
  const [draft, setDraft] = React.useState(value);
  const committed = React.useRef(value);

  // An external change (a saved view, the back button) wins over a stale draft.
  React.useEffect(() => {
    if (value !== committed.current) {
      committed.current = value;
      setDraft(value);
    }
  }, [value]);

  React.useEffect(() => {
    if (draft === committed.current) return;
    const timer = window.setTimeout(() => {
      committed.current = draft;
      onCommit(draft);
    }, delay);
    return () => window.clearTimeout(timer);
  }, [draft, delay, onCommit]);

  return [draft, setDraft];
}
