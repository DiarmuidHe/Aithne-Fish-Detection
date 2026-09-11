/**
 * Saved views: a name and the query string that produces the screen.
 *
 * Stored per browser, because they are one operator's shortcuts rather than a
 * shared configuration. Storage can be refused entirely, in which case the
 * built-ins still work and saving quietly does nothing.
 */

export interface SavedView {
  id: string;
  name: string;
  /** A library query string, without the leading `?`. */
  query: string;
  builtIn?: boolean;
}

const STORAGE_KEY = 'library-saved-views';

export const BUILT_IN_VIEWS: SavedView[] = [
  {
    id: 'builtin:needs-review',
    name: 'Needs review',
    // The three categories where a human decision changes the count.
    query: 'review=flagged&review=borderline&review=disputed&sort=review_status&order=desc',
    builtIn: true,
  },
  {
    id: 'builtin:failed-runs',
    name: 'Failed runs',
    query: 'status=failed',
    builtIn: true,
  },
  {
    id: 'builtin:ready-to-export',
    name: 'Ready to export',
    query: 'status=completed&review_status=complete&has=detections',
    builtIn: true,
  },
];

function read(): SavedView[] {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (view): view is SavedView =>
        typeof view?.id === 'string' &&
        typeof view?.name === 'string' &&
        typeof view?.query === 'string',
    );
  } catch {
    return [];
  }
}

function write(views: SavedView[]): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(views));
  } catch {
    // A browser can refuse storage; the current screen still works.
  }
}

export function loadSavedViews(): SavedView[] {
  return [...BUILT_IN_VIEWS, ...read()];
}

export function saveView(name: string, query: string): SavedView[] {
  const trimmed = name.trim();
  if (!trimmed) return loadSavedViews();
  const existing = read().filter((view) => view.name !== trimmed);
  write([...existing, { id: `view:${Date.now()}`, name: trimmed, query }]);
  return loadSavedViews();
}

export function deleteView(id: string): SavedView[] {
  write(read().filter((view) => view.id !== id));
  return loadSavedViews();
}

/** Which view, if any, the current query string matches. Order-insensitive. */
export function matchView(views: SavedView[], query: string): SavedView | undefined {
  const normalise = (value: string) => {
    const params = [...new URLSearchParams(value).entries()]
      .filter(([key]) => key !== 'page')
      .map(([key, entry]) => `${key}=${entry}`)
      .sort();
    return params.join('&');
  };
  const target = normalise(query);
  return views.find((view) => normalise(view.query) === target);
}
