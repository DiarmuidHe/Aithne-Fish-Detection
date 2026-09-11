/**
 * Selection that survives filtering.
 *
 * A selection made before a filter narrows the list is still a selection: the
 * bulk bar keeps counting it and says how many are currently out of sight,
 * rather than silently dropping work the operator did.
 */

import * as React from 'react';

export interface Selection {
  ids: string[];
  count: number;
  /** Selected videos that the current filters are hiding. */
  hiddenCount: number;
  /** True when every currently visible row is selected. */
  allVisibleSelected: boolean;
  someVisibleSelected: boolean;
  /** True when "select all" meant the whole filtered result, not just this page. */
  wholeResultSelected: boolean;
  has: (id: string) => boolean;
  toggle: (id: string) => void;
  setMany: (ids: string[], selected: boolean) => void;
  selectAllFiltered: (ids: string[], filteredCount: number) => void;
  clear: () => void;
}

export function useSelection(visibleIds: string[]): Selection {
  const [ids, setIds] = React.useState<Set<string>>(() => new Set());
  const [wholeResultSelected, setWholeResultSelected] = React.useState(false);

  const visible = React.useMemo(() => new Set(visibleIds), [visibleIds]);

  const has = React.useCallback((id: string) => ids.has(id), [ids]);

  const toggle = React.useCallback((id: string) => {
    setWholeResultSelected(false);
    setIds((previous) => {
      const next = new Set(previous);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const setMany = React.useCallback((values: string[], selected: boolean) => {
    setWholeResultSelected(false);
    setIds((previous) => {
      const next = new Set(previous);
      for (const value of values) {
        if (selected) next.add(value);
        else next.delete(value);
      }
      return next;
    });
  }, []);

  const selectAllFiltered = React.useCallback((values: string[], filteredCount: number) => {
    setIds(new Set(values));
    // Only claim the whole result when the page really is the whole result.
    setWholeResultSelected(values.length === filteredCount);
  }, []);

  const clear = React.useCallback(() => {
    setIds(new Set());
    setWholeResultSelected(false);
  }, []);

  const selectedVisible = visibleIds.filter((id) => ids.has(id)).length;

  return {
    ids: [...ids],
    count: ids.size,
    hiddenCount: [...ids].filter((id) => !visible.has(id)).length,
    allVisibleSelected: visibleIds.length > 0 && selectedVisible === visibleIds.length,
    someVisibleSelected: selectedVisible > 0 && selectedVisible < visibleIds.length,
    wholeResultSelected,
    has,
    toggle,
    setMany,
    selectAllFiltered,
    clear,
  };
}
