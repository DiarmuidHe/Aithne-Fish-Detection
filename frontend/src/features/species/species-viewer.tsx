import * as React from 'react';
import { Dialog } from '@base-ui/react/dialog';
import { useQuery } from '@tanstack/react-query';

import { fetchSpeciesGallery } from '@/api/species';
import { Button } from '@/components/base/button';
import { Skeleton } from '@/components/base/feedback';
import { CloseIcon } from '@/icons';

import styles from './species-selector.module.css';

/*
 * A closer look at what a species looks like, from more than one angle.
 *
 * One stock portrait answers "is that roughly this animal?". It does not answer
 * "is this a ballan wrasse or a corkwing?", which is the question an operator is
 * actually stuck on when they open this: two fish of similar build, separated by
 * markings that one photograph at one angle may not show. So the viewer asks the
 * server for every photo we are entitled to keep of that species and lets the
 * operator page through them.
 *
 * The photos are still not evidence. Nothing here is the fish on camera, and the
 * heading says so in the same words the rest of the product uses, because a
 * reference mistaken for a detection would be worse than no reference at all.
 */

export function SpeciesReferenceViewer({
  species,
  commonName,
  onClose,
}: {
  /** Open when a species is named; `null` closes the viewer. */
  species: string | null;
  commonName?: string | null;
  onClose: () => void;
}) {
  const name = species ?? '';

  // Only ever fetched with the viewer open: this is the one call that may buy
  // several photos, and a grid of thumbnails must never set it off.
  const gallery = useQuery({
    queryKey: ['species-gallery', name],
    queryFn: () => fetchSpeciesGallery(name),
    enabled: name.length > 0,
    staleTime: Infinity,
    retry: false,
  });

  const photos = gallery.data?.photos ?? [];
  const [index, setIndex] = React.useState(0);

  // A different species is a different set; keeping the old position would open
  // the new one on whichever photo happened to share its number.
  React.useEffect(() => setIndex(0), [name]);

  const shown = photos[Math.min(index, Math.max(photos.length - 1, 0))] ?? null;
  const title = commonName ?? gallery.data?.common_name ?? null;

  const step = (delta: number) => {
    if (photos.length < 2) return;
    setIndex((current) => (current + delta + photos.length) % photos.length);
  };

  return (
    <Dialog.Root open={Boolean(species)} onOpenChange={(open) => !open && onClose()}>
      <Dialog.Portal>
        <Dialog.Backdrop className={styles.viewerBackdrop} />
        <Dialog.Viewport className={styles.viewerViewport}>
          <Dialog.Popup
            className={styles.viewer}
            // Arrow keys move between photos wherever focus is inside the viewer,
            // so the carousel is usable without tabbing onto each thumbnail.
            onKeyDown={(event) => {
              if (event.key === 'ArrowRight') step(1);
              else if (event.key === 'ArrowLeft') step(-1);
              else return;
              event.preventDefault();
            }}
          >
            <div className={styles.viewerMain}>
              <div className={styles.viewerHead}>
                <div>
                  <p className={styles.viewerLabel}>Reference — not this fish</p>
                  <Dialog.Title className={styles.viewerTitle}>
                    {title ?? name}
                  </Dialog.Title>
                  {title ? <p className={styles.viewerScientific}>{name}</p> : null}
                </div>
                <Dialog.Close render={<Button variant="quiet" iconOnly aria-label="Close" />}>
                  <CloseIcon />
                </Dialog.Close>
              </div>

              <div className={styles.viewerFrame}>
                {gallery.isLoading ? (
                  <Skeleton height={320} />
                ) : shown ? (
                  <img
                    className={styles.viewerImage}
                    src={shown.url}
                    alt={`Reference photo of ${name} — not this fish`}
                    decoding="async"
                  />
                ) : (
                  <p className={styles.viewerEmpty}>
                    No reference photo is held for {name}.
                  </p>
                )}
              </div>

              {shown?.attribution ? (
                <p className={styles.viewerCredit}>
                  {shown.source_url ? (
                    <a href={shown.source_url} target="_blank" rel="noreferrer noopener">
                      {shown.attribution}
                    </a>
                  ) : (
                    shown.attribution
                  )}
                  {shown.licence ? ` · ${shown.licence.toUpperCase()}` : ''}
                </p>
              ) : null}
            </div>

            <div className={styles.viewerSide}>
              <p className={styles.viewerLabel}>
                {photos.length > 1 ? `${photos.length} photos` : 'Other photos'}
              </p>
              {photos.length > 1 ? (
                <ul className={styles.viewerThumbs}>
                  {photos.map((photo, position) => (
                    <li key={photo.url}>
                      <button
                        type="button"
                        className={styles.viewerThumb}
                        aria-current={position === index}
                        aria-label={`Photo ${position + 1} of ${photos.length}`}
                        onClick={() => setIndex(position)}
                      >
                        <img src={photo.url} alt="" loading="lazy" decoding="async" />
                      </button>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className={styles.viewerEmpty}>
                  {gallery.isLoading
                    ? 'Looking for other photos of this species…'
                    : 'No additional reference photos are held for this species.'}
                </p>
              )}
              {photos.length > 1 ? (
                <p className={styles.viewerEmpty}>
                  Left and right arrow keys move between photos.
                </p>
              ) : null}
            </div>
          </Dialog.Popup>
        </Dialog.Viewport>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
