import * as React from 'react';

import styles from './media-grid.module.css';

/*
 * The evidence browser.
 *
 * Everywhere this application shows fish, the fish is the content and the
 * interface is not. So a tile has no card around it: the footage sits directly
 * on the band, the caption is two lines of text under it, and the actions stay
 * out of the way until the pointer or the keyboard arrives.
 *
 * The one thing allowed to mark a tile permanently is a review flag, and it
 * does so with a 2px rule along the top of the frame and a dot beside the
 * title — never a label printed across the footage.
 */

export function MediaGrid({
  children,
  size = 'default',
}: {
  children: React.ReactNode;
  /** `large` is for galleries that are the page, not a tab inside one. */
  size?: 'default' | 'large';
}) {
  return (
    <div className={styles.grid} data-size={size}>
      {children}
    </div>
  );
}

export function MediaTile({
  media,
  title,
  subtitle,
  meta,
  flag,
  flagTone = 'neutral',
  actions,
  footnote,
  ratio = '4 / 3',
}: {
  media: React.ReactNode;
  title: React.ReactNode;
  /** The identifier. Monospace, quiet, beside the name rather than under it. */
  subtitle?: React.ReactNode;
  /** One line of figures: counts, confidence, time. */
  meta?: React.ReactNode;
  flag?: React.ReactNode;
  flagTone?: 'neutral' | 'flagged' | 'named';
  /** Revealed on hover or keyboard focus, never printed on every tile. */
  actions?: React.ReactNode;
  footnote?: React.ReactNode;
  ratio?: string;
}) {
  return (
    <figure className={styles.tile} data-flag={flagTone}>
      <div className={styles.frame} style={{ ['--ratio' as string]: ratio }}>
        {media}
        {actions ? <div className={styles.actions}>{actions}</div> : null}
      </div>
      <figcaption className={styles.caption}>
        <span className={styles.line}>
          <span className={styles.title}>{title}</span>
          {subtitle ? <span className={styles.subtitle}>{subtitle}</span> : null}
        </span>
        {flag ? (
          <span className={styles.flag}>
            <span className={styles.flagDot} aria-hidden="true" />
            {flag}
          </span>
        ) : null}
        {meta ? <span className={styles.meta}>{meta}</span> : null}
        {footnote ? <span className={styles.footnote}>{footnote}</span> : null}
      </figcaption>
    </figure>
  );
}
