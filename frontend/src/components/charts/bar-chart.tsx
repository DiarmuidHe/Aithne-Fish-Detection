/**
 * Hand-rolled SVG charts, extending what the old product.js already did.
 *
 * Four small charts do not justify a charting dependency. Each one keeps its
 * axes while loading and carries a text summary, so nothing here is readable
 * only as a picture.
 */

import * as React from 'react';
import { Link } from 'react-router';

import { formatCount } from '@/lib/format';

import styles from './charts.module.css';

export interface Column {
  label: string;
  value: number;
  /** Drawn faintly behind `value`, e.g. all detections behind accepted ones. */
  background?: number;
}

const PLOT_WIDTH = 900;
const PLOT_HEIGHT = 300;
const PADDING = { top: 12, right: 8, bottom: 22, left: 40 };

function niceCeiling(value: number): number {
  if (value <= 0) return 1;
  const magnitude = 10 ** Math.floor(Math.log10(value));
  return Math.ceil(value / magnitude) * magnitude;
}

export function ColumnChart({
  title,
  note,
  columns,
  valueLabel,
  loading,
  formatColumnLabel = (column) => column.label,
}: {
  title: string;
  note?: React.ReactNode;
  columns: Column[];
  valueLabel: string;
  loading?: boolean;
  formatColumnLabel?: (column: Column) => string;
}) {
  const maximum = niceCeiling(
    Math.max(1, ...columns.map((column) => Math.max(column.value, column.background ?? 0))),
  );
  const innerWidth = PLOT_WIDTH - PADDING.left - PADDING.right;
  const innerHeight = PLOT_HEIGHT - PADDING.top - PADDING.bottom;
  const step = innerWidth / Math.max(1, columns.length);
  const barWidth = Math.max(1, step - 2);
  const ticks = [0, maximum / 2, maximum];

  const summary = columns.some((column) => column.value > 0)
    ? `${title}: ${columns
        .map((column) => `${formatColumnLabel(column)} ${formatCount(column.value)}`)
        .join(', ')}`
    : `${title}: no ${valueLabel} recorded.`;

  return (
    <figure className={styles.figure}>
      <figcaption className={styles.caption}>
        <span className={styles.captionTitle}>{title}</span>
        {note ? <span>{note}</span> : null}
      </figcaption>
      <svg
        className={styles.plot}
        viewBox={`0 0 ${PLOT_WIDTH} ${PLOT_HEIGHT}`}
        preserveAspectRatio="none"
        role="img"
        aria-label={loading ? `${title}: loading` : summary}
      >
        {ticks.map((tick) => {
          const y = PADDING.top + innerHeight - (tick / maximum) * innerHeight;
          return (
            <g key={tick}>
              <line
                className={tick === 0 ? styles.axis : styles.gridline}
                x1={PADDING.left}
                x2={PLOT_WIDTH - PADDING.right}
                y1={y}
                y2={y}
              />
              <text className={styles.tick} x={PADDING.left - 6} y={y + 3} textAnchor="end">
                {formatCount(Math.round(tick))}
              </text>
            </g>
          );
        })}
        <line
          className={styles.axis}
          x1={PADDING.left}
          x2={PADDING.left}
          y1={PADDING.top}
          y2={PADDING.top + innerHeight}
        />
        {loading
          ? null
          : columns.map((column, index) => {
              const x = PADDING.left + index * step + 1;
              const height = (column.value / maximum) * innerHeight;
              const backgroundHeight = ((column.background ?? 0) / maximum) * innerHeight;
              return (
                <g key={`${column.label}-${index}`} className={styles.column}>
                  {column.background === undefined ? null : (
                    <rect
                      className={styles.barMuted}
                      x={x}
                      y={PADDING.top + innerHeight - backgroundHeight}
                      width={barWidth}
                      height={Math.max(0, backgroundHeight)}
                    />
                  )}
                  <rect
                    className={styles.bar}
                    x={x}
                    y={PADDING.top + innerHeight - height}
                    width={barWidth}
                    height={Math.max(column.value > 0 ? 1 : 0, height)}
                  />
                  <rect
                    className={styles.hit}
                    x={x}
                    y={PADDING.top}
                    width={barWidth}
                    height={innerHeight}
                  >
                    <title>
                      {formatColumnLabel(column)}: {formatCount(column.value)} {valueLabel}
                    </title>
                  </rect>
                </g>
              );
            })}
      </svg>
      {columns.length > 0 && !loading ? (
        <div className={styles.caption}>
          <span className="mono">{formatColumnLabel(columns[0])}</span>
          <span className="mono">{formatColumnLabel(columns[columns.length - 1])}</span>
        </div>
      ) : null}
    </figure>
  );
}

export interface BarRow {
  label: string;
  value: number;
  to?: string;
}

/** A ranked list where the label matters as much as the number. */
export function BarList({
  title,
  note,
  rows,
  valueLabel,
  emptyMessage,
  limit = 12,
}: {
  title: string;
  note?: React.ReactNode;
  rows: BarRow[];
  valueLabel: string;
  emptyMessage: string;
  limit?: number;
}) {
  const shown = rows.slice(0, limit);
  const maximum = Math.max(1, ...shown.map((row) => row.value));

  return (
    <figure className={styles.figure}>
      {/* A list inside a titled panel carries no caption of its own. */}
      {title || note ? (
        <figcaption className={styles.caption}>
          {title ? <span className={styles.captionTitle}>{title}</span> : null}
          {note ? <span>{note}</span> : null}
        </figcaption>
      ) : null}
      {shown.length === 0 ? (
        <p className="note">{emptyMessage}</p>
      ) : (
        <div className={styles.rows}>
          {/* Two recordings can share a filename and a count, so the position
              in the ranking is what makes the key unique. */}
          {shown.map((row, index) => (
            <div className={styles.row} key={`${row.label}-${index}`}>
              <span className={styles.rowLabel} title={row.label}>
                {row.to ? (
                  <Link className={styles.rowLink} to={row.to}>
                    {row.label}
                  </Link>
                ) : (
                  row.label
                )}
              </span>
              <span className={styles.rail}>
                <span
                  className={styles.fill}
                  style={{ width: `${(row.value / maximum) * 100}%` }}
                />
              </span>
              <span className={styles.rowValue}>{formatCount(row.value)}</span>
            </div>
          ))}
        </div>
      )}
      <p className="visually-hidden">
        {shown.length === 0
          ? emptyMessage
          : `${title}: ${shown.map((row) => `${row.label} ${row.value} ${valueLabel}`).join(', ')}`}
      </p>
      {rows.length > limit ? (
        <p className="note">
          Showing the first {limit} of {formatCount(rows.length)}.
        </p>
      ) : null}
    </figure>
  );
}

/** The chart keeps its axes while it waits, so the layout does not jump. */
export function ChartSkeleton({ title }: { title: string }) {
  return (
    <figure className={styles.figure}>
      <figcaption className={styles.caption}>
        <span className={styles.captionTitle}>{title}</span>
        <span>Loading</span>
      </figcaption>
      <div className={styles.skeletonPlot} />
    </figure>
  );
}
