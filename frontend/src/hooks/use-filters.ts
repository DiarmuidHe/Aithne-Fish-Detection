/** Library filters, held in the URL so a shared link reproduces the screen. */

import * as React from 'react';
import { useSearchParams } from 'react-router';

import {
  DEFAULT_FILTERS,
  filterReducer,
  parseFilters,
  toSearchParams,
  type FilterAction,
  type LibraryFilters,
} from '@/features/library/filter-state';

export interface FilterController {
  filters: LibraryFilters;
  dispatch: (action: FilterAction) => void;
  /** Replace the whole filter set, e.g. when a saved view is applied. */
  replace: (query: string) => void;
  /** The current filter query string, for saving a view. */
  query: string;
}

export function useFilters(): FilterController {
  const [params, setParams] = useSearchParams();
  const filters = React.useMemo(() => parseFilters(params), [params]);

  const dispatch = React.useCallback(
    (action: FilterAction) => {
      setParams((current) => toSearchParams(filterReducer(parseFilters(current), action), current), {
        // Filter changes are navigation: the back button should undo them.
        replace: false,
      });
    },
    [setParams],
  );

  const replace = React.useCallback(
    (query: string) => {
      setParams(
        (current) => toSearchParams(parseFilters(new URLSearchParams(query)), current),
        { replace: false },
      );
    },
    [setParams],
  );

  const query = React.useMemo(
    () => toSearchParams(filters).toString(),
    [filters],
  );

  return { filters, dispatch, replace, query };
}

export { DEFAULT_FILTERS };
export type { LibraryFilters };
