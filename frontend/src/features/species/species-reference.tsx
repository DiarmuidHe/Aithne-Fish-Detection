import * as React from 'react';
import { useQuery } from '@tanstack/react-query';

import type { SpeciesReference } from '@/api/types';
import { fetchSpeciesReferences } from '@/api/species';

import styles from './species-reference.module.css';

/*
 * What the named species actually looks like, beside the name.
 *
 * A classifier's answer is a claim, and until now the dashboard printed the claim
 * with nothing to check it against. Putting a stock portrait of the species next to
 * the fish's own crop turns "Gadus morhua, 82%" into something an operator can
 * agree or disagree with at a glance - which is the whole point of showing a
 * confidence figure in the first place.
 *
 * The photo is never the detected fish, and every surface says so, because a
 * reference mistaken for evidence would be worse than no reference at all.
 */

// The server looks a handful of new names up per request and reports the rest as
// `unknown`; asking again shortly after is how the remainder arrive. Found and
// missing are both final, so polling stops as soon as one of them lands.
const PENDING_POLL_MS = 4000;

/**
 * Whether a label is a scientific name worth looking up.
 *
 * Detector class names ("fish", "Unclassified") share the same fields as a
 * Fishial answer, and asking a taxonomy service about them would spend a request
 * per screen to be told what we already know. A binomial is two or three
 * capitalised-genus words and nothing else.
 */
export function looksLikeSpeciesName(name: string | null | undefined): boolean {
  return /^[A-Z][a-z-]+(?: [a-z-]+){1,2}$/.test((name ?? '').trim());
}

export function useSpeciesReference(name: string | null | undefined): SpeciesReference | null {
  const trimmed = looksLikeSpeciesName(name) ? (name as string).trim() : '';
  const query = useQuery({
    queryKey: ['species-reference', trimmed],
    queryFn: () => fetchSpeciesReferences([trimmed]),
    enabled: trimmed.length > 0,
    staleTime: Infinity,
    // A species with no photo is a cosmetic gap; retrying it is not worth a spinner.
    retry: false,
    refetchInterval: (query) =>
      query.state.data?.[0]?.state === 'unknown' ? PENDING_POLL_MS : false,
  });
  return query.data?.[0] ?? null;
}

/** How large the portrait is drawn. Table cells and tiles want very different things. */
export type SpeciesThumbSize = 'inline' | 'row' | 'tile';

/**
 * The portrait on its own, sized for wherever it sits.
 *
 * Renders nothing at all when there is no photo: a row of grey placeholders would
 * cost the same space as the pictures and carry none of the information.
 */
export function SpeciesThumb({
  name,
  size = 'inline',
}: {
  name: string | null | undefined;
  size?: SpeciesThumbSize;
}) {
  const reference = useSpeciesReference(name);
  if (!reference?.image_url) return null;
  return (
    <img
      className={styles.thumb}
      data-size={size}
      src={reference.image_url}
      alt={`Reference photo of ${reference.species}`}
      title={referenceTitle(reference)}
      loading="lazy"
      decoding="async"
    />
  );
}

/**
 * Name and portrait as one unit, for a table cell, a tile caption or a list row.
 *
 * `children` is whatever already followed the name - a confidence figure, a count -
 * so the caller keeps its own wording and only gains the picture.
 */
export function SpeciesNameplate({
  name,
  size = 'inline',
  children,
}: {
  name: string;
  size?: SpeciesThumbSize;
  children?: React.ReactNode;
}) {
  return (
    <span className={styles.nameplate} data-size={size}>
      <SpeciesThumb name={name} size={size} />
      <span className={styles.nameplateText}>
        <span className={styles.name}>{name}</span>
        {children}
      </span>
    </span>
  );
}

/**
 * A detector's or classifier's class name, with a portrait when the name is a species.
 *
 * Pipelines label a track with whatever their model emits: "fish" for a detector
 * that only finds animals, a binomial for one that names them. Both arrive in the
 * same field, so the decision of whether there is anything to illustrate is made
 * here rather than at each of the several screens that print the field.
 */
export function SpeciesLabel({
  value,
  size = 'inline',
  missing = 'Unclassified',
}: {
  value: string | null | undefined;
  size?: SpeciesThumbSize;
  missing?: string;
}) {
  if (!value) return <>{missing}</>;
  if (!looksLikeSpeciesName(value)) return <>{value}</>;
  return <SpeciesNameplate name={value} size={size} />;
}

/**
 * The full comparison: a large portrait, the common name, and who took the photo.
 *
 * Shown where an operator is deciding rather than skimming - the identification
 * panel - so this is the one place that spends vertical space on provenance. The
 * licence and photographer are printed because the photo is someone else's work
 * and the terms it is used under require saying so.
 */
export function SpeciesReferenceCard({ name }: { name: string }) {
  const reference = useSpeciesReference(name);
  if (!reference?.image_url) return null;
  return (
    <figure className={styles.card}>
      <img
        className={styles.cardImage}
        src={reference.image_url}
        alt={`Reference photo of ${reference.species}`}
        loading="lazy"
        decoding="async"
      />
      <figcaption className={styles.cardCaption}>
        <span className={styles.cardLabel}>Reference — not this fish</span>
        {reference.common_name ? (
          <span className={styles.cardCommon}>{reference.common_name}</span>
        ) : null}
        {reference.attribution ? (
          <span className={styles.cardCredit}>
            {reference.source_url ? (
              <a href={reference.source_url} target="_blank" rel="noreferrer noopener">
                {reference.attribution}
              </a>
            ) : (
              reference.attribution
            )}
          </span>
        ) : null}
      </figcaption>
    </figure>
  );
}

/** Hover text for a thumbnail: the common name if there is one, then the credit. */
function referenceTitle(reference: SpeciesReference): string {
  const parts = [`Reference photo of ${reference.species} — not this fish`];
  if (reference.common_name) parts.push(reference.common_name);
  if (reference.attribution) parts.push(reference.attribution);
  return parts.join('\n');
}
