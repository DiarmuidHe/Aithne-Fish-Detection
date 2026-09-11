import { act, renderHook } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { useSelection } from './use-selection';

describe('selection across filtering', () => {
  it('keeps a selection that the filters later hide, and says how many', () => {
    const { result, rerender } = renderHook(({ visible }) => useSelection(visible), {
      initialProps: { visible: ['a', 'b', 'c'] },
    });

    act(() => {
      result.current.toggle('a');
      result.current.toggle('b');
    });
    expect(result.current.count).toBe(2);
    expect(result.current.hiddenCount).toBe(0);

    // A filter narrows the list; the work already done is not thrown away.
    rerender({ visible: ['a'] });

    expect(result.current.count).toBe(2);
    expect(result.current.hiddenCount).toBe(1);
    expect(result.current.ids.sort()).toEqual(['a', 'b']);
  });

  it('reports whether the visible rows are all, partly, or not selected', () => {
    const { result } = renderHook(() => useSelection(['a', 'b']));

    expect(result.current.allVisibleSelected).toBe(false);
    expect(result.current.someVisibleSelected).toBe(false);

    act(() => result.current.toggle('a'));
    expect(result.current.someVisibleSelected).toBe(true);
    expect(result.current.allVisibleSelected).toBe(false);

    act(() => result.current.toggle('b'));
    expect(result.current.allVisibleSelected).toBe(true);
    expect(result.current.someVisibleSelected).toBe(false);
  });

  it('only claims the whole result when the page really is the whole result', () => {
    const { result } = renderHook(() => useSelection(['a', 'b']));

    act(() => result.current.selectAllFiltered(['a', 'b'], 2));
    expect(result.current.wholeResultSelected).toBe(true);

    act(() => result.current.selectAllFiltered(['a', 'b'], 312));
    expect(result.current.wholeResultSelected).toBe(false);
    expect(result.current.count).toBe(2);
  });

  it('stops claiming the whole result once one row is unpicked', () => {
    const { result } = renderHook(() => useSelection(['a', 'b']));

    act(() => result.current.selectAllFiltered(['a', 'b'], 2));
    act(() => result.current.toggle('a'));

    expect(result.current.wholeResultSelected).toBe(false);
    expect(result.current.count).toBe(1);
  });

  it('selects and deselects a run of rows at once', () => {
    const { result } = renderHook(() => useSelection(['a', 'b', 'c']));

    act(() => result.current.setMany(['a', 'b'], true));
    expect(result.current.count).toBe(2);

    act(() => result.current.setMany(['a'], false));
    expect(result.current.ids).toEqual(['b']);
  });

  it('clears everything, including rows hidden by the filters', () => {
    const { result, rerender } = renderHook(({ visible }) => useSelection(visible), {
      initialProps: { visible: ['a', 'b'] },
    });

    act(() => result.current.setMany(['a', 'b'], true));
    rerender({ visible: ['a'] });
    act(() => result.current.clear());

    expect(result.current.count).toBe(0);
    expect(result.current.hiddenCount).toBe(0);
  });
});
