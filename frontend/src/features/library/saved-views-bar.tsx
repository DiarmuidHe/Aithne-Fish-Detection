import * as React from 'react';

import { Button } from '@/components/base/button';
import { PopoverPanel } from '@/components/base/popups';
import { CloseIcon } from '@/icons';

import { deleteView, loadSavedViews, matchView, saveView, type SavedView } from './saved-views';
import styles from './library.module.css';
import toggles from '@/components/base/toggle.module.css';

export function SavedViewsBar({
  query,
  onApply,
}: {
  query: string;
  onApply: (query: string) => void;
}) {
  const [views, setViews] = React.useState<SavedView[]>(() => loadSavedViews());
  const [name, setName] = React.useState('');
  const active = matchView(views, query);

  return (
    <div className={styles.savedViews}>
      {views.map((view) => (
        <span key={view.id} className={toggles.chipRow}>
          <button
            type="button"
            className={toggles.chip}
            aria-pressed={active?.id === view.id}
            style={
              active?.id === view.id
                ? { background: 'var(--accent-quiet)', borderColor: 'var(--accent)' }
                : undefined
            }
            onClick={() => onApply(view.query)}
          >
            {view.name}
            {view.builtIn ? null : (
              <span
                className={toggles.chipRemove}
                role="button"
                tabIndex={0}
                aria-label={`Delete the ${view.name} view`}
                onClick={(event) => {
                  event.stopPropagation();
                  setViews(deleteView(view.id));
                }}
                onKeyDown={(event) => {
                  if (event.key !== 'Enter' && event.key !== ' ') return;
                  event.stopPropagation();
                  event.preventDefault();
                  setViews(deleteView(view.id));
                }}
              >
                <CloseIcon width={12} height={12} />
              </span>
            )}
          </button>
        </span>
      ))}

      <PopoverPanel
        title="Save this view"
        trigger={
          <Button variant="quiet" size="small">
            Save view
          </Button>
        }
      >
        <form
          style={{ display: 'grid', gap: 'var(--space-2)', minWidth: 240 }}
          onSubmit={(event) => {
            event.preventDefault();
            if (!name.trim()) return;
            setViews(saveView(name, query));
            setName('');
          }}
        >
          <label className="note" htmlFor="saved-view-name">
            Name
          </label>
          <input
            id="saved-view-name"
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder="e.g. River camera backlog"
            maxLength={60}
          />
          <p className="note">
            Saves the current filters and sort in this browser. It does not change what other
            operators see.
          </p>
          <Button type="submit" variant="primary" size="small" disabled={!name.trim()}>
            Save
          </Button>
        </form>
      </PopoverPanel>
    </div>
  );
}
