import * as React from 'react';
import { ScrollArea } from '@base-ui/react/scroll-area';

import { SortAscIcon, SortDescIcon } from '@/icons';

import styles from './table.module.css';

export { styles as tableStyles };

export function Scroller({
  children,
  className,
  style,
}: {
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
}) {
  return (
    <ScrollArea.Root className={`${styles.scroll} ${className ?? ''}`} style={style}>
      <ScrollArea.Viewport className={styles.viewport}>
        <ScrollArea.Content>{children}</ScrollArea.Content>
      </ScrollArea.Viewport>
      <ScrollArea.Scrollbar className={styles.scrollbar} orientation="vertical">
        <ScrollArea.Thumb className={styles.thumb} />
      </ScrollArea.Scrollbar>
      <ScrollArea.Scrollbar className={styles.scrollbar} orientation="horizontal">
        <ScrollArea.Thumb className={styles.thumb} />
      </ScrollArea.Scrollbar>
      <ScrollArea.Corner />
    </ScrollArea.Root>
  );
}

export function DataTable({
  children,
  stickyFirstColumn = true,
  ...props
}: React.TableHTMLAttributes<HTMLTableElement> & { stickyFirstColumn?: boolean }) {
  return (
    <table
      className={`${styles.table} ${stickyFirstColumn ? styles.stickyFirst : ''}`}
      {...props}
    >
      {children}
    </table>
  );
}

export interface SortHeaderProps {
  label: string;
  field: string;
  active: string;
  order: 'asc' | 'desc';
  onSort: (field: string) => void;
  numeric?: boolean;
}

export function SortHeader({
  label,
  field,
  active,
  order,
  onSort,
  numeric,
}: SortHeaderProps) {
  const isActive = active === field;
  return (
    <th
      scope="col"
      className={numeric ? styles.numeric : undefined}
      aria-sort={isActive ? (order === 'asc' ? 'ascending' : 'descending') : 'none'}
    >
      <button
        type="button"
        className={styles.sortButton}
        data-active={isActive || undefined}
        onClick={() => onSort(field)}
      >
        {label}
        {isActive ? (
          <span className={styles.sortGlyph} aria-hidden="true">
            {order === 'asc' ? <SortAscIcon /> : <SortDescIcon />}
          </span>
        ) : null}
      </button>
    </th>
  );
}

/**
 * Windowed rows for long tables.
 *
 * Past `threshold` rows the table renders only what is near the viewport, with
 * spacer rows above and below so the scrollbar still tells the truth.
 */
export function useVirtualRows<T>({
  rows,
  rowHeight = 35,
  threshold = 200,
  overscan = 12,
}: {
  rows: T[];
  rowHeight?: number;
  threshold?: number;
  overscan?: number;
}) {
  const containerRef = React.useRef<HTMLDivElement | null>(null);
  const [range, setRange] = React.useState({ start: 0, end: rows.length });
  const virtualized = rows.length > threshold;

  React.useEffect(() => {
    if (!virtualized) {
      setRange({ start: 0, end: rows.length });
      return;
    }
    const element = containerRef.current?.querySelector<HTMLElement>(`.${styles.viewport}`);
    if (!element) return;

    const update = () => {
      const start = Math.max(0, Math.floor(element.scrollTop / rowHeight) - overscan);
      const visible = Math.ceil(element.clientHeight / rowHeight) + overscan * 2;
      setRange({ start, end: Math.min(rows.length, start + visible) });
    };
    update();
    element.addEventListener('scroll', update, { passive: true });
    const observer = new ResizeObserver(update);
    observer.observe(element);
    return () => {
      element.removeEventListener('scroll', update);
      observer.disconnect();
    };
  }, [rows.length, rowHeight, overscan, virtualized]);

  return {
    containerRef,
    virtualized,
    visibleRows: virtualized ? rows.slice(range.start, range.end) : rows,
    padTop: virtualized ? range.start * rowHeight : 0,
    padBottom: virtualized ? Math.max(0, (rows.length - range.end) * rowHeight) : 0,
  };
}

export function SpacerRow({ height, columns }: { height: number; columns: number }) {
  if (height <= 0) return null;
  return (
    <tr className={styles.spacer} aria-hidden="true">
      <td colSpan={columns} style={{ height }} />
    </tr>
  );
}
